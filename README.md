# rvllm

Two modes, one engine: `rvllm cli` for asking questions in the terminal and
`rvllm serve` for an OpenAI-compatible HTTP server.

## Supported models

Only the SmolLM2-360M family is supported right now, in Q8_0 GGUF form with its
`tokenizer.json`:

- `SmolLM2-360M.Q8_0.gguf` (base): plain completions.
- `SmolLM2-360M-Instruct` Q8_0 GGUF (use the Instruct repo's `tokenizer.json`):
  chat, via `--chat` and `/v1/chat/completions`. Tested end to end, including
  multi-turn history and stopping on `<|im_end|>`.

These are the only models this has been developed and tested against. The loader
reads Llama-style GGUF metadata (`llama.*` keys) and dequantizes everything to F32
on load (F16 on the GPU), so other Llama-architecture models might load, but
nothing else is tested, and the tokenizer wrapper and the hardcoded ChatML chat
template are specific to SmolLM2. Base models aren't chat-tuned, so don't expect
sensible answers from `--chat` with the base file.

GGUF files store Q/K weights in llama.cpp's interleaved RoPE layout; the loader
reorders them to the rotate-half layout the kernels use (`weights.rs`).

## CLI mode

Interactive question -> answer loop (answers stream as they are generated;
Ctrl-D or `exit` quits):

```
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json
```

Pass `--chat` to wrap input in the ChatML template (use this with `-Instruct`
models); the conversation history is then kept between questions.

With `--prompt` the CLI runs the given prompts once and exits. Multiple
`--prompt`s run concurrently through continuous batching, all sharing one KV
block pool:

```
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --prompt "The sky is" \
  --prompt "2 + 2 =" \
  --max-tokens 20
```

`--arrival-step <n>` (one per `--prompt`, in order) simulates staggered
arrival: that request only becomes eligible for admission at decode step `n`
(default 0, i.e. immediately). `--block-size` and `--num-blocks` control the
KV cache's paging granularity and total capacity; `--num-blocks` defaults to
enough blocks for every request to reach its own `prompt + max_tokens`
length at once (capped at `context_length`), and can be set explicitly to
reserve more or less.

By default only the completion text is printed (one per request, in `--prompt`
order). Pass `--debug` to also print each prompt with a labelled completion,
plus the prefix cache hit rate, total block allocations, and step count /
throughput (tok/s) for the run.

### TUI

`cli --tui` opens a full-screen terminal UI (ratatui) with a model picker and a
chat view that streams the answer as it is generated:

```bash
cargo run --release -- cli --tui --models-dir .. --chat
```

The picker lists the `.gguf` files under `--models-dir` (default `.`, plus one
level of subdirectories); each model uses a `tokenizer.json` next to it, else
`--tokenizer`. `--model` skips the picker and preloads that file. Keys: Enter
send, Esc stop generating, Ctrl-O switch model, Ctrl-N new conversation,
PgUp/PgDn scroll, Ctrl-C quit. The sampling, `--chat`, `--max-tokens`,
`--device` and KV flags apply as in the plain interactive loop;
`--prompt`/`--debug`/`--arrival-step` can't be combined with `--tui`.

## Server mode

```
cargo run --release -- serve \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json \
  --port 8000
```

```
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 32, "stream": true
}'
```

Endpoints: `POST /v1/completions`, `POST /v1/chat/completions` (both with
`stream: true` SSE support, `stream_options.include_usage`, `stop`,
`temperature`, `top_p`, `seed`, `max_tokens`), `GET /v1/models`,
`GET /health` (live running/waiting counts). Only `n = 1` is supported, and
completions take a single prompt. Any OpenAI client works by pointing its base
URL at `http://localhost:8000/v1`.

### How concurrent clients are handled

All HTTP connections are served by tokio, but inference runs on one engine
thread that continuously batches whatever is in flight:

- **Shared batching.** Two agent sessions hitting the endpoint at once simply
  share decode steps; a request arriving mid-generation joins the very next
  step. Each request samples with its own RNG, so a seeded request gives the
  same output regardless of what else is running.
- **Queue.** At most `--max-running` (default 8) sequences decode together.
  Further requests wait in a FIFO queue and start as slots free up (or as KV
  blocks free up; when the pool is exhausted mid-flight the engine preempts
  the least recently used sequence and re-prefills it later).
- **Backpressure.** Once `--max-running + --max-queue` (default 8 + 64)
  requests are in flight, new ones get `429` with `Retry-After: 1` instead of
  piling up.
- **Cancellation.** If a client disconnects, its request is aborted and its KV
  blocks and queue slot are released immediately.

KV capacity is `--num-blocks` blocks of `--block-size` tokens; by default
`--max-running` x `--tokens-per-seq` (2048) tokens. A single request can use
at most the whole pool (and never more than the model's context length).
`--default-max-tokens`, `--default-temperature` and `--default-top-p` apply
to requests that omit them. SmolLM2 base models aren't chat-tuned: for real
chat use a `-Instruct` GGUF (the chat template is ChatML).

## Sampling

`--temperature` (default `0.0` in `cli`) and `--top-p` (default `1.0`) control
next-token selection. `--temperature 0.0` is greedy argmax - deterministic, no
RNG involved. Any positive temperature enables nucleus (top-p) sampling,
seeded by `--seed` (default `42`) for reproducible runs:

```
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 20 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 7
```

## Benchmarking

`bench` sweeps batch size against synthetic requests (random token ids - content
doesn't affect throughput, only sequence length and arrival timing) and reports
total tokens/sec, tokens/sec/request, prefix cache hit rate, and KV pool
utilization for each batch size:

```
cargo run --release --bin bench -- \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --batch-sizes 1,4,8,16,32 \
  --prompt-len 128 \
  --gen-len 64
```

For more realistic load - varied prompt/completion lengths and staggered
arrivals, exercising continuous batching and prefix caching the way the fixed
synthetic sweep doesn't - pass a trace file instead of a batch-size sweep:

```
cargo run --release --bin bench -- \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --trace-file trace.jsonl
```

`trace.jsonl` is one JSON object per line:

```
{"prompt_len": 128, "gen_len": 64, "arrival_step": 0}
{"prompt_len": 256, "gen_len": 32, "arrival_step": 5}
```

Each config runs twice - a discarded warmup, then the measured run - so
reported throughput is steady-state, not first-call allocation overhead.

## CUDA support

Requires an NVIDIA GPU, a CUDA 13.0-compatible driver (`nvidia-smi` shows the
driver version), and building with the `cuda` feature, which pulls in
`cudarc` and compiles every kernel under `src/cuda/kernels/*.cu` via NVRTC at
runtime - no separate build step, no `nvcc` needed.

```
cargo build --release --features cuda
cargo test --features cuda          # kernel/model unit tests; each skips
                                     # instead of failing if no CUDA device
                                     # is present (e.g. on a Mac)
```

Builds and runs unchanged on machines without a GPU - the feature is opt-in,
and the default (CPU/Candle) path is untouched by it.

Every command (`cli`, `serve`, `bench`) takes `--device cuda` to run through the
custom-written CUDA inference path (`cuda::model::CudaModel` +
`cuda::engine::CudaBackend`) instead of the CPU/Candle one, instead of `cargo build`'s `--features cuda` alone -
that just makes the CUDA code available in the binary, `--device cuda` is
what actually switches to it at runtime. Omitting it (or building without
the feature at all) always uses the CPU path.

CLI, same flags as the CPU examples above, run through CUDA (`serve` works
the same way):

```
cargo run --release --features cuda -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 20 \
  --device cuda
```

`bench`, same stats/report format as the CPU sweep, so a CPU run and a CUDA
run are directly comparable:

```
cargo run --release --features cuda --bin bench -- \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --batch-sizes 1,4,8,16,32 \
  --device cuda
```
