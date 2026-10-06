use std::io::{BufRead, Write};

use clap::{Args, Parser, Subcommand};
use rvllm::chat::{self, ChatMessage};
use rvllm::engine::{self, BoxError, Engine, EngineConfig, Request, RequestSpec};
use rvllm::model::{Config, Model};
use rvllm::runtime::Runtime;
use rvllm::sampler::{Sampler, SamplingParams};
use rvllm::server;
use rvllm::tokenizer::{SmollLM230MTokenizer, StreamDecoder};

#[derive(Parser)]
#[command(about = "Small vLLM-style inference engine: terminal Q&A or an OpenAI-compatible server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ask questions in the terminal: an interactive prompt loop, or - with
    /// `--prompt` - run the given prompts once and exit.
    Cli(CliArgs),
    /// Serve an OpenAI-compatible HTTP API (/v1/completions,
    /// /v1/chat/completions, /v1/models, /health).
    Serve(ServeArgs),
}

#[derive(Args)]
struct ModelArgs {
    // Required, except with `cli --tui` where the model can be picked in the UI.
    #[arg(long)]
    model: Option<String>,
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: String,
    // "cpu" (default) or "cuda" - requires building with --features cuda.
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    // Total KV cache capacity, in blocks of `--block-size` tokens. Defaults
    // are sized per mode (see each command).
    #[arg(long)]
    num_blocks: Option<usize>,
}

#[derive(Args)]
struct CliArgs {
    #[command(flatten)]
    model: ModelArgs,
    // One request per occurrence: `--prompt "a" --prompt "b"` runs two
    // concurrent requests through continuous batching, prints the answers and
    // exits. Without any `--prompt`, starts the interactive loop.
    #[arg(long)]
    prompt: Vec<String>,
    // Defaults to 20 with `--prompt`, 256 in the interactive loop.
    #[arg(long)]
    max_tokens: Option<usize>,
    // Wrap input in the ChatML chat template (for -Instruct models). In the
    // interactive loop the conversation history is kept between questions.
    #[arg(long)]
    chat: bool,
    // One entry per `--prompt`, in the same order: the decode step at which
    // that request "arrives" (becomes eligible to be admitted). Requests
    // without a matching entry arrive at step 0. Simulates staggered arrival
    // for exercising the waiting queue and continuous batching.
    #[arg(long)]
    arrival_step: Vec<usize>,
    // 0.0 = greedy argmax (deterministic, no RNG use).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    // Nucleus sampling threshold; ignored when temperature is 0.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    // With `--prompt`: also print each prompt with its labelled completion,
    // then prefix cache and throughput stats.
    #[arg(long)]
    debug: bool,
    // Full-screen terminal UI (ratatui): pick a model from `--models-dir` and
    // chat with streamed output. `--model` preloads one instead of opening the
    // picker.
    #[arg(long, conflicts_with_all = ["prompt", "debug", "arrival_step"])]
    tui: bool,
    // Where the TUI model picker looks for .gguf files (plus one level of
    // subdirectories).
    #[arg(long, default_value = ".")]
    models_dir: String,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    // Name reported by /v1/models and echoed in responses. Defaults to the
    // model file's name.
    #[arg(long)]
    model_name: Option<String>,
    // Sequences decoded together in one batch. Further requests wait in a FIFO
    // queue until a slot frees up.
    #[arg(long, default_value_t = 8)]
    max_running: usize,
    // Requests allowed to wait beyond `--max-running`. When running + queued
    // is at this limit, new requests get HTTP 429.
    #[arg(long, default_value_t = 64)]
    max_queue: usize,
    // Defaults for requests that leave these out.
    #[arg(long, default_value_t = 256)]
    default_max_tokens: usize,
    #[arg(long, default_value_t = 1.0)]
    default_temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    default_top_p: f32,
    // KV capacity per concurrent sequence used to size the pool when
    // `--num-blocks` isn't given: pool = max_running * this many tokens.
    #[arg(long, default_value_t = 2048)]
    tokens_per_seq: usize,
}

fn main() -> Result<(), BoxError> {
    match Cli::parse().command {
        Command::Cli(args) => run_cli(args),
        Command::Serve(args) => run_serve(args),
    }
}

fn model_path(args: &ModelArgs) -> Result<&str, BoxError> {
    args.model
        .as_deref()
        .ok_or_else(|| "--model is required".into())
}

fn blocks_for(tokens: usize, block_size: usize) -> usize {
    tokens.div_ceil(block_size).max(1)
}

fn load_tokenizer(path: &str, config: &Config) -> Result<SmollLM230MTokenizer, BoxError> {
    SmollLM230MTokenizer::from_file(path, config.eos_token_id, config.bos_token_id)
}

// Tokens that end a generation: the model's EOS, plus the chat end-of-turn
// marker when the vocabulary has one.
fn stop_tokens(tokenizer: &SmollLM230MTokenizer) -> Vec<u32> {
    let mut stops = vec![tokenizer.eos_token_id];
    stops.extend(tokenizer.token_to_id(chat::END_OF_TURN));
    stops
}

// ---- cli ------------------------------------------------------------------

fn run_cli(args: CliArgs) -> Result<(), BoxError> {
    let sampling = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
    };
    if args.tui {
        return rvllm::tui::run(rvllm::tui::TuiOptions {
            device: args.model.device,
            block_size: args.model.block_size,
            num_blocks: args.model.num_blocks,
            max_tokens: args.max_tokens.unwrap_or(256),
            chat: args.chat,
            sampling,
            fallback_tokenizer: args.model.tokenizer.into(),
            models_dir: args.models_dir.into(),
            initial_model: args.model.model.map(Into::into),
        });
    }
    let model = Model::load(model_path(&args.model)?)?;
    let tokenizer = load_tokenizer(&args.model.tokenizer, &model.config)?;
    let runtime = Runtime::new(&model, &args.model.device)?;
    if args.prompt.is_empty() {
        interactive(&args, &runtime, &tokenizer, sampling)
    } else {
        one_shot(&args, &runtime, &tokenizer, sampling)
    }
}

fn one_shot(
    args: &CliArgs,
    runtime: &Runtime,
    tokenizer: &SmollLM230MTokenizer,
    sampling: SamplingParams,
) -> Result<(), BoxError> {
    let context_length = runtime.model().config.context_length as usize;
    let max_tokens = args.max_tokens.unwrap_or(20);
    let block_size = args.model.block_size;

    let requests: Vec<RequestSpec> = args
        .prompt
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let text = if args.chat {
                chat::chatml(&[ChatMessage {
                    role: "user".into(),
                    content: prompt.clone(),
                }])
            } else {
                prompt.clone()
            };
            Ok(RequestSpec {
                tokens: tokenizer.encode(&text)?,
                max_tokens,
                arrival_step: args.arrival_step.get(i).copied().unwrap_or(0),
            })
        })
        .collect::<Result<_, BoxError>>()?;

    // Enough blocks for every request to reach its own prompt+max_tokens at
    // once, capped at the model's context_length - sized off what the
    // requests actually need rather than the full context_length per
    // sequence, which on a GPU can mean OOM well before the pool is full.
    let num_blocks = args.model.num_blocks.unwrap_or_else(|| {
        let longest = requests
            .iter()
            .map(|r| (r.tokens.len() + r.max_tokens).min(context_length))
            .max()
            .unwrap_or(0);
        blocks_for(longest, block_size) * requests.len()
    });

    let backend = runtime.backend(block_size, num_blocks)?;
    let result = engine::run(
        backend,
        &requests,
        sampling,
        EngineConfig {
            block_size,
            num_blocks,
            max_running: usize::MAX,
        },
    )?;

    let stops = stop_tokens(tokenizer);
    for (i, generated) in result.generated.iter().enumerate() {
        // A request that ended on a stop token has it as its last entry.
        let ids: Vec<u32> = match generated.split_last() {
            Some((last, rest)) if stops.contains(last) => rest.to_vec(),
            _ => generated.clone(),
        };
        let text = tokenizer.decode(&ids)?;
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

fn interactive(
    args: &CliArgs,
    runtime: &Runtime,
    tokenizer: &SmollLM230MTokenizer,
    sampling: SamplingParams,
) -> Result<(), BoxError> {
    let context_length = runtime.model().config.context_length as usize;
    let block_size = args.model.block_size;
    let max_tokens = args.max_tokens.unwrap_or(256);
    let num_blocks = args
        .model
        .num_blocks
        .unwrap_or_else(|| blocks_for(context_length.min(4096), block_size));
    let mut engine = Engine::new(
        runtime.backend(block_size, num_blocks)?,
        EngineConfig {
            block_size,
            num_blocks,
            max_running: 1,
        },
    );
    let stops = stop_tokens(tokenizer);

    eprintln!("Type a question and press Enter. Ctrl-D or `exit` to quit.");
    let mut history: Vec<ChatMessage> = Vec::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "exit" | "quit") {
            break;
        }

        let prompt = if args.chat {
            history.push(ChatMessage {
                role: "user".into(),
                content: line.to_string(),
            });
            chat::chatml(&history)
        } else {
            line.to_string()
        };
        let request = Request {
            tokens: tokenizer.encode(&prompt)?,
            max_tokens,
            stop_tokens: stops.clone(),
            sampler: Sampler::from_params(sampling),
        };
        let id = match engine.add_request(request) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("error: {e}");
                if args.chat {
                    history.pop();
                }
                continue;
            }
        };

        // Stream the answer as it is generated.
        let mut decoder = StreamDecoder::new();
        let mut answer = String::new();
        'generate: while engine.has_work() {
            for ev in engine.step()? {
                debug_assert_eq!(ev.id, id);
                let stopped = ev.finish == Some(engine::FinishReason::Stop);
                let text = if stopped {
                    decoder.flush(tokenizer)?
                } else {
                    decoder.push(tokenizer, ev.token)?
                };
                answer.push_str(&text);
                write!(stdout, "{text}")?;
                stdout.flush()?;
                if ev.finish.is_some() {
                    break 'generate;
                }
            }
        }
        writeln!(stdout)?;
        if args.chat {
            history.push(ChatMessage {
                role: "assistant".into(),
                content: answer,
            });
        }
    }
    Ok(())
}

// ---- serve ----------------------------------------------------------------

fn run_serve(args: ServeArgs) -> Result<(), BoxError> {
    let model_file = model_path(&args.model)?.to_string();
    server::run(
        &model_file,
        &args.model.tokenizer,
        server::ServeOptions {
            device: args.model.device,
            addr: format!("{}:{}", args.host, args.port),
            model_name: args.model_name,
            block_size: args.model.block_size,
            num_blocks: args.model.num_blocks,
            tokens_per_seq: args.tokens_per_seq,
            max_running: args.max_running,
            max_queue: args.max_queue,
            default_max_tokens: args.default_max_tokens,
            default_temperature: args.default_temperature,
            default_top_p: args.default_top_p,
        },
    )
}
