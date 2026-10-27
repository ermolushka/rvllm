use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;

use clap::{Args, Parser, Subcommand, ValueEnum};
use rvllm::chat::{self, ChatFormat, ChatMessage};
use rvllm::conversation::{self, Chat};
use rvllm::engine::{self, BoxError, Engine, EngineConfig, RequestSpec};
use rvllm::history::Strategy;
use rvllm::model::Model;
use rvllm::runtime::Runtime;
use rvllm::sampler::SamplingParams;
use rvllm::server;
use rvllm::tokenizer::{self, Tokenizer};

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
    // Defaults to the tokenizer.json next to the model file, else
    // ./tokenizer.json.
    #[arg(long)]
    tokenizer: Option<String>,
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
    // Wrap input in the model's chat format (for instruct models). In the
    // interactive loop the conversation history is kept between questions.
    #[arg(long)]
    chat: bool,
    // System prompt for `--chat`, added after the model's default one.
    #[arg(long, requires = "chat")]
    system: Option<String>,
    // With `--tui --chat`: what to do with the oldest turns once the chat
    // outgrows the KV cache. `summarize` has the model fold them into a
    // summary kept in the system prompt; `drop` removes them.
    #[arg(long, value_enum, default_value_t = CompactArg::Summarize, requires = "chat")]
    compact: CompactArg,
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

#[derive(Clone, Copy, ValueEnum)]
enum CompactArg {
    Drop,
    Summarize,
}

impl From<CompactArg> for Strategy {
    fn from(arg: CompactArg) -> Self {
        match arg {
            CompactArg::Drop => Strategy::Drop,
            CompactArg::Summarize => Strategy::Summarize,
        }
    }
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

// The messages a chat starts with: the model's default system prompt, if it
// has one, and the `--system` text, if given.
fn opening_messages(args: &CliArgs, format: Option<ChatFormat>) -> Vec<ChatMessage> {
    chat::system_prompt(format, args.system.as_deref())
        .map(|system| ChatMessage {
            role: "system".into(),
            content: system,
        })
        .into_iter()
        .collect()
}

fn tokenizer_path(args: &ModelArgs, model: &str) -> String {
    match &args.tokenizer {
        Some(path) => path.clone(),
        None => tokenizer::path_for(Path::new(model), Path::new("tokenizer.json"))
            .to_string_lossy()
            .into_owned(),
    }
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
            system: args.system,
            sampling,
            fallback_tokenizer: args
                .model
                .tokenizer
                .unwrap_or_else(|| "tokenizer.json".into())
                .into(),
            models_dir: args.models_dir.into(),
            initial_model: args.model.model.map(Into::into),
            compact: args.compact.into(),
        });
    }
    let model_file = model_path(&args.model)?;
    let model = Model::load(model_file)?;
    let tokenizer = Tokenizer::for_model(&tokenizer_path(&args.model, model_file), &model.config)?;
    let format = ChatFormat::for_model(&model.config);
    let chat = match (args.chat, format) {
        (false, _) => None,
        (true, Some(format)) => Some(format),
        (true, None) => return Err("this model's chat format is unknown; drop --chat".into()),
    };
    let stops = chat::stop_tokens(&tokenizer, format);
    let runtime = Runtime::new(&model, &args.model.device)?;
    if args.prompt.is_empty() {
        interactive(&args, &runtime, &tokenizer, chat, &stops, sampling)
    } else {
        one_shot(&args, &runtime, &tokenizer, chat, &stops, sampling)
    }
}

// `chat` is the format to wrap prompts in, or None for plain completion.
fn one_shot(
    args: &CliArgs,
    runtime: &Runtime,
    tokenizer: &Tokenizer,
    chat: Option<ChatFormat>,
    stops: &[u32],
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
            let text = match chat {
                Some(format) => {
                    let mut messages = opening_messages(args, chat);
                    messages.push(ChatMessage {
                        role: "user".into(),
                        content: prompt.clone(),
                    });
                    format.render(&messages)
                }
                None => prompt.clone(),
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
        stops,
    )?;

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
    tokenizer: &Tokenizer,
    chat: Option<ChatFormat>,
    stops: &[u32],
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
    eprintln!("Type a question and press Enter. Ctrl-D or `exit` to quit.");
    let mut chat = chat.map(|format| {
        Chat::new(
            format,
            chat::system_prompt(Some(format), args.system.as_deref()),
            args.compact.into(),
        )
    });
    let options = conversation::Options {
        max_tokens,
        stop_tokens: stops.to_vec(),
        sampling,
    };
    // Never set here: Ctrl-C ends the process, as it always has.
    let cancel = AtomicBool::new(false);
    let stdin = std::io::stdin();
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

        let mut terminal = Terminal {
            out: std::io::stdout(),
        };
        let result = match chat.as_mut() {
            Some(chat) => match chat.begin_turn(
                &mut engine,
                tokenizer,
                line.to_string(),
                max_tokens,
                &cancel,
                &mut terminal,
            ) {
                Ok(prompt) => chat.finish_turn(
                    &mut engine,
                    tokenizer,
                    prompt,
                    &options,
                    &cancel,
                    &mut terminal,
                ),
                Err(e) => Err(e),
            },
            None => match tokenizer.encode(line) {
                Ok(prompt) => conversation::plain(
                    &mut engine,
                    tokenizer,
                    prompt,
                    &options,
                    &cancel,
                    &mut terminal,
                ),
                Err(e) => Err(e),
            },
        };
        match result {
            Ok(_) => writeln!(terminal.out)?,
            Err(e) => eprintln!("error: {e}"),
        }
    }
    Ok(())
}

// Writes an answer to the terminal as it is generated. Notices go to stderr,
// so they don't mix into the answer.
struct Terminal {
    out: std::io::Stdout,
}

impl conversation::Sink for Terminal {
    fn text(&mut self, piece: &str) -> bool {
        write!(self.out, "{piece}")
            .and_then(|_| self.out.flush())
            .is_ok()
    }

    fn notice(&mut self, line: String) {
        eprintln!("{line}");
    }
}

// ---- serve ----------------------------------------------------------------

fn run_serve(args: ServeArgs) -> Result<(), BoxError> {
    let model_file = model_path(&args.model)?.to_string();
    server::run(
        &model_file,
        &tokenizer_path(&args.model, &model_file),
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
