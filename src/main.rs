use clap::Parser;
use rvllm::engine::{self, RequestSpec};
use rvllm::sampler::Sampler;
use rvllm::model::Model;
use rvllm::tokenizer::SmollLM230MTokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokenizer: String,
    // One request per occurrence: `--prompt "a" --prompt "b"` runs two
    // concurrent requests through continuous batching.
    #[arg(long)]
    prompt: Vec<String>,
    #[arg(long, default_value_t = 20)]
    max_tokens: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    // Defaults to enough blocks for every prompt to hit `context_length` at
    // once, once the model is loaded.
    #[arg(long)]
    num_blocks: Option<usize>,
    // One entry per `--prompt`, in the same order: the decode step at which
    // that request "arrives" (becomes eligible to be admitted). Requests
    // without a matching entry arrive at step 0. Simulates staggered arrival
    // for exercising the waiting queue and continuous batching.
    #[arg(long)]
    arrival_step: Vec<usize>,
    // 0.0 = greedy argmax (byte-identical to Phase 1's decoding, no RNG use).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    // Nucleus sampling threshold; ignored when temperature is 0.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    // Also print each prompt with its labelled completion, then prefix cache and
    // throughput stats. Without it, only the completion text is printed.
    #[arg(long)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let model = Model::load(&args.model)?;
    let config = &model.config;

    let smoll_tokenizer =
        SmollLM230MTokenizer::from_file(&args.tokenizer, config.eos_token_id, config.bos_token_id)?;

    let block_size = args.block_size;
    let blocks_per_seq = (config.context_length as usize).div_ceil(block_size);
    let num_blocks =
        args.num_blocks.unwrap_or_else(|| blocks_per_seq * args.prompt.len().max(1));

    let requests: Vec<RequestSpec> = args
        .prompt
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let tokens = smoll_tokenizer.encode(prompt)?;
            let arrival_step = args.arrival_step.get(i).copied().unwrap_or(0);
            Ok(RequestSpec { tokens, max_tokens: args.max_tokens, arrival_step })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?;

    let mut sampler = Sampler::new(args.temperature, args.top_p, args.seed);
    let result = engine::run(&model, &requests, block_size, num_blocks, &mut sampler)?;

    for (i, generated) in result.generated.iter().enumerate() {
        let text = smoll_tokenizer.decode(generated)?;
        if args.debug {
            println!("[{i}] prompt: {:?}", args.prompt[i]);
            println!("[{i}] completion: {text}");
        } else {
            println!("{text}");
        }
    }

    if args.debug {
        let stats = &result.stats;
        println!(
            "prefix cache: {} / {} tokens hit ({:.1}%), {} total block allocations",
            stats.prefix_hits,
            stats.prefix_total,
            if stats.prefix_total == 0 {
                0.0
            } else {
                stats.prefix_hits as f64 / stats.prefix_total as f64 * 100.0
            },
            stats.total_block_allocs,
        );
        println!(
            "{} steps, {} output tokens in {:.2?} ({:.1} tok/s)",
            stats.steps,
            stats.output_tokens,
            stats.elapsed,
            stats.output_tokens as f64 / stats.elapsed.as_secs_f64(),
        );
    }

    Ok(())
}
