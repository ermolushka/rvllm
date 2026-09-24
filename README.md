# rvllm

Single request:

```
cargo run --release -- \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 20
```

Multiple concurrent requests (continuous batching - pass `--prompt` once per
request, all sharing one KV block pool):

```
cargo run --release -- \
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
enough blocks for every request to reach `context_length` at once.

By default only the completion text is printed (one per request, in `--prompt`
order). Pass `--debug` to also print each prompt with a labelled completion,
plus the prefix cache hit rate, total block allocations, and step count /
throughput (tok/s) for the run.

## Sampling

`--temperature` (default `0.0`) and `--top-p` (default `1.0`) control next-token
selection. `--temperature 0.0` is greedy argmax - deterministic, no RNG involved.
Any positive temperature enables nucleus (top-p) sampling, seeded by `--seed`
(default `42`) for reproducible runs:

```
cargo run --release -- \
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
