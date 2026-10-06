// One-call server entry point: loads the model and tokenizer, starts the
// engine thread, and serves until Ctrl-C. `main.rs` and `examples/server.rs`
// both go through this.

use std::sync::Arc;

use crate::engine::{BoxError, Engine, EngineConfig};
use crate::model::Model;
use crate::runtime::Runtime;
use crate::tokenizer::SmollLM230MTokenizer;

use super::worker::{WorkerStatus, run_worker};
use super::{AppState, ServerConfig, serve};

#[derive(Debug, Clone)]
pub struct ServeOptions {
    // "cpu" or "cuda" (the latter needs the `cuda` feature).
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
        }
    }
}

// Blocks until the server shuts down (Ctrl-C) or fails.
pub fn run(model_path: &str, tokenizer_path: &str, opts: ServeOptions) -> Result<(), BoxError> {
    if opts.max_running == 0 {
        return Err("max_running must be at least 1".into());
    }
    let model = Model::load(model_path)?;
    let tokenizer = SmollLM230MTokenizer::from_file(
        tokenizer_path,
        model.config.eos_token_id,
        model.config.bos_token_id,
    )?;
    let context_length = model.config.context_length as usize;
    let block_size = opts.block_size;
    let num_blocks = opts.num_blocks.unwrap_or_else(|| {
        opts.max_running
            * context_length
                .min(opts.tokens_per_seq)
                .div_ceil(block_size)
                .max(1)
    });
    let engine_cfg = EngineConfig {
        block_size,
        num_blocks,
        max_running: opts.max_running,
    };
    let model_name = opts.model_name.clone().unwrap_or_else(|| {
        std::path::Path::new(model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "rvllm".into())
    });

    let (job_tx, job_rx) = tokio::sync::mpsc::unbounded_channel();
    let status = Arc::new(WorkerStatus::default());
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize, String>>();

    std::thread::scope(|scope| -> Result<(), BoxError> {
        // The engine thread builds its own runtime (CUDA context included) so
        // device state never crosses threads.
        let worker_status = status.clone();
        let device = opts.device.clone();
        let model = &model;
        scope.spawn(move || {
            let runtime = match Runtime::new(model, &device) {
                Ok(r) => r,
                Err(e) => return drop(ready_tx.send(Err(e.to_string()))),
            };
            let backend = match runtime.backend(block_size, num_blocks) {
                Ok(b) => b,
                Err(e) => return drop(ready_tx.send(Err(e.to_string()))),
            };
            let engine = Engine::new(backend, engine_cfg);
            let _ = ready_tx.send(Ok(engine.max_sequence_tokens()));
            run_worker(engine, job_rx, worker_status);
        });
        let max_sequence_tokens = ready_rx
            .recv()
            .map_err(|_| "engine thread died during startup")??;

        let state = AppState::new(
            ServerConfig {
                model_name: model_name.clone(),
                max_running: opts.max_running,
                max_queue: opts.max_queue,
                default_max_tokens: opts.default_max_tokens,
                default_temperature: opts.default_temperature,
                default_top_p: opts.default_top_p,
                context_length,
                max_sequence_tokens,
            },
            tokenizer,
            job_tx,
            status,
        );

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let addr = &opts.addr;
        let result = rt.block_on(async {
            let listener = tokio::net::TcpListener::bind(addr).await?;
            eprintln!(
                "serving {model_name:?} on http://{addr} ({} on {}; {} running slots, queue of {}; KV pool {} blocks x {} tokens)",
                model_path, opts.device, opts.max_running, opts.max_queue, num_blocks, block_size,
            );
            serve(listener, state).await
        });
        // Drops any request tasks still holding the job channel, so the
        // engine thread sees it close and exits.
        rt.shutdown_background();
        result?;
        Ok(())
    })
}
