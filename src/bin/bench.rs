// Benchmark harness: sweeps batch size against
// synthetic traces (fixed prompt/gen length, all arriving together) and,
// optionally, a trace file of varied prompt/gen lengths with staggered
// arrivals (ShareGPT-style load). Reports the plan's north-star metric,
// tokens/sec/request, alongside total throughput and cache/pool stats.
//
// Token content doesn't matter for throughput measurement, only sequence
// lengths and arrival timing, so prompts are synthetic random token ids
// rather than real text - this needs `--model` (for config: vocab size,
// layer count, block-table sizing) but not `--tokenizer`.

use std::collections::HashMap;
use std::fs;

use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rvllm::engine::{self, RequestSpec};
use rvllm::sampler::Sampler;
use rvllm::weights::SmollLM230MConfig;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    // Comma-separated batch sizes to sweep, e.g. "1,4,8,16,32". Ignored when
    // `--trace-file` is given (the file's request count is the batch size).
    #[arg(long, default_value = "1,4,8,16,32")]
    batch_sizes: String,
    #[arg(long, default_value_t = 128)]
    prompt_len: usize,
    #[arg(long, default_value_t = 64)]
    gen_len: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    // JSONL trace: one `{"prompt_len": N, "gen_len": N, "arrival_step": N}`
    // object per line. When given, runs that single trace instead of the
    // synthetic batch-size sweep - for realistic length distributions and
    // staggered arrivals exercising continuous batching / prefix caching
    // under more varied load than the fixed-length sweep.
    #[arg(long)]
    trace_file: Option<String>,
}

struct TraceEntry {
    prompt_len: usize,
    gen_len: usize,
    arrival_step: usize,
}

fn parse_trace_file(path: &str) -> Result<Vec<TraceEntry>, Box<dyn std::error::Error + Send + Sync>> {
    let content = fs::read_to_string(path)?;
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let field = |name: &str| -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
            let key = format!("\"{name}\"");
            let start = line
                .find(&key)
                .ok_or_else(|| format!("trace line missing {name:?}: {line}"))?
                + key.len();
            let rest = &line[start..];
            let digits: String = rest
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            Ok(digits.parse()?)
        };
        entries.push(TraceEntry {
            prompt_len: field("prompt_len")?,
            gen_len: field("gen_len")?,
            arrival_step: field("arrival_step").unwrap_or(0),
        });
    }
    Ok(entries)
}

fn synthetic_requests(
    rng: &mut StdRng,
    vocab_size: u32,
    batch_size: usize,
    prompt_len: usize,
    gen_len: usize,
) -> Vec<RequestSpec> {
    (0..batch_size)
        .map(|_| RequestSpec {
            tokens: (0..prompt_len).map(|_| rng.random_range(0..vocab_size)).collect(),
            max_tokens: gen_len,
            arrival_step: 0,
        })
        .collect()
}

fn requests_from_trace(rng: &mut StdRng, vocab_size: u32, trace: &[TraceEntry]) -> Vec<RequestSpec> {
    trace
        .iter()
        .map(|t| RequestSpec {
            tokens: (0..t.prompt_len).map(|_| rng.random_range(0..vocab_size)).collect(),
            max_tokens: t.gen_len,
            arrival_step: t.arrival_step,
        })
        .collect()
}

fn run_and_report(
    label: &str,
    config: &SmollLM230MConfig,
    device: &Device,
    requests: &[RequestSpec],
    block_size: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let batch_size = requests.len();
    let blocks_per_seq = (config.context_length as usize).div_ceil(block_size);
    let num_blocks = blocks_per_seq * batch_size.max(1);

    // Warmup run (discarded) so the measured run isn't paying for first-touch
    // allocation of the KV storage tensors, per the plan's "steady-state
    // throughput once warmup is excluded".
    let mut warmup_sampler = Sampler::new(0.0, 1.0, seed);
    engine::run(config, device, requests, block_size, num_blocks, &mut warmup_sampler)?;

    let mut sampler = Sampler::new(0.0, 1.0, seed);
    let result = engine::run(config, device, requests, block_size, num_blocks, &mut sampler)?;
    let stats = &result.stats;

    let secs = stats.elapsed.as_secs_f64();
    let total_tok_s = stats.output_tokens as f64 / secs;
    let per_req_tok_s = total_tok_s / batch_size as f64;
    let hit_rate = if stats.prefix_total == 0 {
        0.0
    } else {
        stats.prefix_hits as f64 / stats.prefix_total as f64 * 100.0
    };
    let pool_utilization = stats.total_block_allocs as f64 / num_blocks as f64 * 100.0;

    // Order-sensitive FNV-1a over every generated token: lets before/after
    // runs be compared for exact greedy-output equality at a glance.
    let mut checksum: u64 = 0xcbf29ce484222325;
    for tok in result.generated.iter().flatten() {
        checksum = (checksum ^ *tok as u64).wrapping_mul(0x100000001b3);
    }
    let mut ttft_ms: Vec<f64> = stats.ttft.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ttft_ms.sort_by(|a, b| a.total_cmp(b));
    let ttft_mean = ttft_ms.iter().sum::<f64>() / ttft_ms.len().max(1) as f64;
    let ttft_max = ttft_ms.last().copied().unwrap_or(0.0);
    let prefill_tok_s = stats.prefill_tokens as f64 / stats.prefill_time.as_secs_f64().max(1e-9);
    let decode_ms_step =
        stats.decode_time.as_secs_f64() * 1e3 / stats.decode_steps.max(1) as f64;

    let steps = stats.steps;
    let total_allocs = stats.total_block_allocs;
    println!(
        "{label:<24} batch={batch_size:<4} steps={steps:<5} tok/s(total)={total_tok_s:<9.1} \
         tok/s/req={per_req_tok_s:<8.2} prefix_hit={hit_rate:<5.1}% \
         pool_util={pool_utilization:<6.1}% ({total_allocs}/{num_blocks} blocks)"
    );
    println!(
        "{:<24} ttft_ms(mean/max)={ttft_mean:.1}/{ttft_max:.1} prefill_tok/s={prefill_tok_s:.1} \
         decode_ms/step={decode_ms_step:.2} checksum={checksum:016x}",
        ""
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let device = Device::Cpu;

    let mut config = SmollLM230MConfig {
        n_layers: 0,
        n_heads: 0,
        n_kv_heads: 0,
        hidden_dim: 0,
        ffn_dim: 0,
        rope_theta: 0.0,
        rms_eps: 0.0,
        vocab_size: 0,
        context_length: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        tensors: HashMap::new(),
    };
    config.read_gguf(&args.model)?;

    let mut rng = StdRng::seed_from_u64(args.seed);

    if let Some(trace_path) = &args.trace_file {
        let trace = parse_trace_file(trace_path)?;
        let requests = requests_from_trace(&mut rng, config.vocab_size, &trace);
        println!("trace: {} requests from {trace_path}", requests.len());
        run_and_report("trace", &config, &device, &requests, args.block_size, args.seed)?;
        return Ok(());
    }

    let batch_sizes: Vec<usize> = args
        .batch_sizes
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;

    println!(
        "synthetic sweep: prompt_len={} gen_len={} block_size={}",
        args.prompt_len, args.gen_len, args.block_size
    );
    for &batch_size in &batch_sizes {
        let requests = synthetic_requests(
            &mut rng,
            config.vocab_size,
            batch_size,
            args.prompt_len,
            args.gen_len,
        );
        run_and_report(
            &format!("batch_size={batch_size}"),
            &config,
            &device,
            &requests,
            args.block_size,
            args.seed,
        )?;
    }

    Ok(())
}
