// One-call server entry point: loads the model and tokenizer, starts the
// engine thread, and serves until Ctrl-C. `main.rs` and `examples/server.rs`
// both go through this.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::chat::ChatFormat;
use crate::engine::{BoxError, Engine, EngineConfig};
use crate::history::Strategy;
use crate::model::Model;
use crate::runtime::Runtime;
use crate::tokenizer::Tokenizer;

use super::hub::{Active, Hub, Switcher};
use super::worker::{WorkerStatus, run_worker};
use super::{AppState, ServerConfig, serve_hub};

#[derive(Debug, Clone)]
pub struct ServeOptions {
    // "cpu", "cuda" or "metal" (the latter two need their feature).
    pub device: String,
    // `host:port` to bind.
    pub addr: String,
    // Reported by /v1/models; defaults to the model file's stem.
    pub model_name: Option<String>,
    pub block_size: usize,
    // KV pool size in blocks; defaults to `max_running * tokens_per_seq`
    // worth of tokens.
    pub num_blocks: Option<usize>,
    pub tokens_per_seq: usize,
    pub max_running: usize,
    pub max_queue: usize,
    pub default_max_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    // How a chat that outgrows the KV pool is compacted.
    pub compact: Strategy,
    // Where the chat UI's model picker looks for `.gguf` files (plus one level
    // of subdirectories). Defaults to the directory of the model being served.
    pub models_dir: Option<PathBuf>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            addr: "127.0.0.1:8000".into(),
            model_name: None,
            block_size: 16,
            num_blocks: None,
            tokens_per_seq: 2048,
            max_running: 8,
            max_queue: 64,
            default_max_tokens: 256,
            default_temperature: 1.0,
            default_top_p: 1.0,
            compact: Strategy::Summarize,
            models_dir: None,
        }
    }
}

// What the engine thread reports once the model is up.
struct Ready {
    tokenizer: Tokenizer,
    chat_format: Option<ChatFormat>,
    context_length: usize,
    max_sequence_tokens: usize,
}

// Loads `model_path` on a new engine thread and returns the request-facing
// state for it. Blocks until the model is loaded or has failed to.
fn start_engine(
    model_path: &Path,
    tokenizer_path: &Path,
    model_name: Option<String>,
    opts: &ServeOptions,
) -> Result<Active, String> {
    let block_size = opts.block_size;
    let (job_tx, job_rx) = tokio::sync::mpsc::unbounded_channel();
    let status = Arc::new(WorkerStatus::default());
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Ready, String>>();

    let thread = {
        let (model_path, tokenizer_path) = (model_path.to_path_buf(), tokenizer_path.to_path_buf());
        let (device, opts, status) = (opts.device.clone(), opts.clone(), status.clone());
        // The engine thread owns the model and builds its own runtime (CUDA
        // context included), so device state never crosses threads.
        std::thread::spawn(move || {
            let fail = |e: &dyn std::fmt::Display| drop(ready_tx.send(Err(e.to_string())));
            let model = match Model::load(&model_path.to_string_lossy()) {
                Ok(m) => m,
                Err(e) => return fail(&e),
            };
            let tokenizer =
                match Tokenizer::for_model(&tokenizer_path.to_string_lossy(), &model.config) {
                    Ok(t) => t,
                    Err(e) => return fail(&e),
                };
            let context_length = model.config.context_length as usize;
            let num_blocks = opts.num_blocks.unwrap_or_else(|| {
                opts.max_running
                    * context_length
                        .min(opts.tokens_per_seq)
                        .div_ceil(block_size)
                        .max(1)
            });
            let runtime = match Runtime::new(&model, &device) {
                Ok(r) => r,
                Err(e) => return fail(&e),
            };
            let backend = match runtime.backend(block_size, num_blocks) {
                Ok(b) => b,
                Err(e) => return fail(&e),
            };
            let engine = Engine::new(
                backend,
                EngineConfig {
                    block_size,
                    num_blocks,
                    max_running: opts.max_running,
                },
            );
            let _ = ready_tx.send(Ok(Ready {
                tokenizer,
                chat_format: ChatFormat::for_model(&model.config),
                context_length,
                max_sequence_tokens: engine.max_sequence_tokens(),
            }));
            run_worker(engine, job_rx, status);
        })
    };

    let ready = match ready_rx.recv() {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            let _ = thread.join();
            return Err(e);
        }
        Err(_) => {
            let _ = thread.join();
            return Err("engine thread died during startup".into());
        }
    };
    let model_name = model_name.unwrap_or_else(|| {
        model_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "rvllm".into())
    });
    let state = AppState::new(
        ServerConfig {
            model_name,
            max_running: opts.max_running,
            max_queue: opts.max_queue,
            default_max_tokens: opts.default_max_tokens,
            default_temperature: opts.default_temperature,
            default_top_p: opts.default_top_p,
            context_length: ready.context_length,
            max_sequence_tokens: ready.max_sequence_tokens,
            compact: opts.compact,
        },
        ready.tokenizer,
        ready.chat_format,
        job_tx,
        status,
    );
    Ok(Active {
        state,
        thread: Some(thread),
        path: model_path.to_path_buf(),
    })
}

// Blocks until the server shuts down (Ctrl-C) or fails.
pub fn run(model_path: &str, tokenizer_path: &str, opts: ServeOptions) -> Result<(), BoxError> {
    if opts.max_running == 0 {
        return Err("max_running must be at least 1".into());
    }
    let model_file = PathBuf::from(model_path);
    let fallback_tokenizer = PathBuf::from(tokenizer_path);
    let active = start_engine(
        &model_file,
        &fallback_tokenizer,
        opts.model_name.clone(),
        &opts,
    )?;
    let (model_name, num_blocks) = (
        active.state.cfg.model_name.clone(),
        opts.num_blocks.unwrap_or_else(|| {
            opts.max_running
                * (active.state.cfg.context_length)
                    .min(opts.tokens_per_seq)
                    .div_ceil(opts.block_size)
                    .max(1)
        }),
    );

    // Models picked in the UI use the tokenizer next to them, else the one
    // this server was started with.
    let models_dir = opts.models_dir.clone().unwrap_or_else(|| {
        model_file
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf()
    });
    let loader_opts = opts.clone();
    let switcher = Switcher::new(
        Box::new(move |path| {
            let tokenizer = crate::tokenizer::path_for(path, &fallback_tokenizer);
            start_engine(path, &tokenizer, None, &loader_opts)
        }),
        models_dir,
        opts.device.clone(),
    );
    let hub = Hub::switchable(active, switcher);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let addr = &opts.addr;
    let result = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!(
            "serving {model_name:?} on http://{addr} ({} on {}; {} running slots, queue of {}; KV pool {} blocks x {} tokens)",
            model_path, opts.device, opts.max_running, opts.max_queue, num_blocks, opts.block_size,
        );
        eprintln!("chat UI: http://{addr}/");
        serve_hub(listener, hub.clone()).await
    });
    // Drops any request tasks still holding the job channel, so the engine
    // thread sees it close and exits.
    rt.shutdown_background();
    hub.shutdown();
    result?;
    Ok(())
}
