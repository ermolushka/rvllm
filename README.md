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
