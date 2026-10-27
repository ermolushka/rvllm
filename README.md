# rvllm

Two modes, one engine: `rvllm cli` for asking questions in the terminal and
`rvllm serve` for an OpenAI-compatible HTTP server.

## Use as a library

```toml
rvllm = { version = "0.1", default-features = false }            # engine only
rvllm = { version = "0.1", default-features = false, features = ["server"] }  # + HTTP server module
```

```rust
use rvllm::generator::{GenerateOptions, Generator};

let generator = Generator::load("SmolLM2-360M.Q8_0.gguf", "tokenizer.json", "cpu")?;
let text = generator.complete("The capital of France is", &GenerateOptions::default())?;
```

Features: `server` (axum API), `tui` (ratatui UI), `cli` (default; both plus the
`rvllm`/`bench` binaries), `cuda`. Lower-level pieces (`Runtime`, `Engine`,
`Backend`) are public too.

## Supported models

These models are supported, in Q8_0 GGUF form with their `tokenizer.json`:

- `SmolLM2-360M.Q8_0.gguf` (base, `llama` architecture): plain completions.
- `SmolLM2-360M-Instruct` Q8_0 GGUF (`llama`): chat in ChatML.
- `Llama-3.2-1B-Instruct` Q8_0 GGUF (`llama`, with Llama 3's RoPE scaling): chat
  in Llama 3's header format.
- `Qwen2.5-0.5B-Instruct` Q8_0 GGUF (`qwen2`): chat in ChatML.
- `Qwen3-0.6B` Q8_0 GGUF (`qwen3`): chat in ChatML, with thinking off.
- `gemma-3-270m-it` Q8_0 GGUF (`gemma3`): chat in Gemma's turn format.
- `LFM2.5-1.2B-Instruct` Q8_0 GGUF (`lfm2`): chat in ChatML.

Each is tested end to end on the CPU and on CUDA, including multi-turn history
via `--chat` and `/v1/chat/completions` and stopping at the end of the model's
turn. `--chat`
uses the model's chat format where rvllm knows it and refuses otherwise; plain
completion works either way. Other models of these architectures load with a
warning on stderr, because they aren't tested: a model counts as tested when its
architecture, layer count and width match one of the models above. Base models aren't chat-tuned, so don't expect sensible answers
from `--chat` with a base file.

Everything is dequantized to F32 on load (F16 on the GPU), so a model needs
about 4 bytes of RAM per parameter: roughly 1.5 GB for SmolLM2-360M and 5 GB for
Llama-3.2-1B. The CUDA backend runs every model listed above, with the same
greedy output as the CPU path (see the CUDA support section).

`--tokenizer` defaults to the `tokenizer.json` next to the model file, then
`./tokenizer.json`. A tokenizer that gives the model's BOS or EOS token a
different id belongs to another model and is rejected.

llama.cpp stores Llama-architecture Q/K weights in an interleaved RoPE layout;
the loader reorders them to the rotate-half layout the kernels use
(`weights.rs`). The other supported families are stored that way already.

## CLI mode

Interactive question -> answer loop (answers stream as they are generated;
Ctrl-D or `exit` quits):

```
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json
```

Pass `--chat` to wrap input in the model's chat format (use this with instruct
models); the conversation history is then kept between questions. `--system
"..."` is added after the model's default system prompt, so both apply. A
conversation that outgrows the KV cache is compacted, as described in
[Long chats](#long-chats).

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
PgUp/PgDn scroll, Ctrl-C quit. The sampling, `--chat`, `--system`,
`--compact`, `--max-tokens`, `--device` and KV flags apply as in the plain
interactive loop; `--prompt`/`--debug`/`--arrival-step` can't be combined with
`--tui`.

Messages typed while an answer is streaming wait in a queue above the input.
Each goes to the model when the answer before it finishes, and it moves into
the transcript once the model has taken it. Ctrl-N clears the queue. Esc during
a summary stops the send: the message stays in the queue until you press Enter.

### Long chats

A chat resends its whole history each turn, so eventually it outgrows the KV
cache. The TUI (`cli --tui --chat`), the terminal chat (`cli --chat`) and the
server's `/v1/chat/completions` follow the same rules:

- **Budget.** The pool is `--num-blocks` × `--block-size` tokens, capped at the
  context length (the terminal chat and TUI size it to the context length, up to
  4096 tokens, unless `--num-blocks` is given). An answer keeps room for its
  `max_tokens`, but at most a quarter of the pool, and the prompt gets the rest.
- **Compaction.** A prompt over that budget is compacted to about three quarters
  of it before the message is sent. `--compact summarize` (the default) has the
  model fold the oldest turns into a summary kept in the system prompt;
  `--compact drop` removes the oldest user/assistant pairs instead. If the
  summary can't be written, turns are dropped. The TUI and terminal chat say
  what compaction did. If the newest message doesn't fit in the pool even after
  compaction, the request is refused.
- **Continuation.** If an answer fills the pool before it reaches `max_tokens`,
  the history is compacted again, the answer so far is kept, and generation
  carries on. An answer is continued at most 64 times. It stops early only when
  the history can't make room for the rest of it; the TUI and terminal chat say
  so in a notice, and the server ends the answer with `finish_reason: "length"`.
  Each continuation starts a fresh sampler, so sampled (non-greedy) output
  differs from an uninterrupted run. Greedy output doesn't.

Answers without a chat history (`/v1/completions`, and `cli` without `--chat`)
can't be compacted. They are capped at the room the pool leaves beside the
prompt, and the terminal chat says so when that cuts an answer short.

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

Chats follow [Long chats](#long-chats): `--compact` (default `summarize`) sets
how an over-budget conversation is compacted, and an answer that fills the pool
is continued. The server keeps no conversation state, so each request is
compacted from the messages it carries. With `summarize` that can cost an extra
generation per request until the conversation fits. `usage.prompt_tokens` counts
the prompt as first sent, after compaction, and `completion_tokens` counts every
generated token across the continuations. `/v1/completions` is capped at the
room left in the pool and never continued. A chat uses the system message the
client sends as it is, and the model's default system prompt only when the
client sends none. That differs from `--system` in the CLI, which is added after
the default.

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
  the least recently used sequence and re-prefills it later). A chat answer
  that is continued after compaction waits in the queue again for its next
  segment, behind requests that arrived in the meantime.
- **Backpressure.** Once `--max-running + --max-queue` (default 8 + 64)
  requests are in flight, new ones get `429` with `Retry-After: 1` instead of
  piling up.
- **Cancellation.** If a client disconnects, its request is aborted and its KV
  blocks and queue slot are released immediately.

KV capacity is `--num-blocks` blocks of `--block-size` tokens; by default
`--max-running` x `--tokens-per-seq` (2048) tokens. A single request can use
at most the whole pool (and never more than the model's context length).
`--default-max-tokens`, `--default-temperature` and `--default-top-p` apply
to requests that omit them. Base models aren't chat-tuned: for real chat use an
instruct GGUF. `/v1/chat/completions` uses the model's chat format, like `--chat`.

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
cargo test --features cuda          # kernel/model unit tests, CPU vs CUDA
RVLLM_REQUIRE_CUDA=1 cargo test --features cuda   # fail, not skip, without a GPU
```

Without a GPU the CUDA tests skip with a note on stderr, so a plain
`cargo test --features cuda` can pass without running anything on the GPU. Set
`RVLLM_REQUIRE_CUDA=1` to make a missing device a failure. CI builds and tests
only the CPU path.

Builds and runs unchanged on machines without a GPU - the feature is opt-in,
and the default (CPU/Candle) path is untouched by it.

On WSL2, a stale `libcuda.so.1` elsewhere on the loader path can shadow the
WSL driver, and then CUDA reports `CUDA_ERROR_NO_DEVICE` even though
`nvidia-smi` works. Run with `LD_LIBRARY_PATH=/usr/lib/wsl/lib` in that case,
or remove the stale library.

Which features run on the GPU, and how each was checked:

- Llama 3 RoPE scaling, Qwen2 attention biases, Qwen3 Q/K normalization,
  Gemma 3 (local sliding-window attention with its own RoPE base, GELU
  feed-forward, post-attention and post-FFN norms, scaled embeddings) and
  LFM2's short convolutions, with its conv state kept on the GPU per sequence
  and per cached block, so prefix-cache hits resume correctly.
- Each feature is compared with the CPU path by a unit test on a tiny model,
  including a batched decode and a prefix-cache resume for short convolutions.
- On RTX 4060 (WSL2), every README model's greedy output matches the CPU
  path, with the GGUF files checked against their Hugging Face LFS hashes.

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
