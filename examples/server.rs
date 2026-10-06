// Minimal OpenAI-compatible server built only on the library (`server`
// feature, no clap/ratatui):
//   cargo run --release --no-default-features --features server --example server -- <model.gguf> <tokenizer.json> [addr]

use std::sync::Arc;

use rvllm::engine::{BoxError, Engine, EngineConfig};
use rvllm::model::Model;
use rvllm::runtime::Runtime;
use rvllm::server::worker::{WorkerStatus, run_worker};
use rvllm::server::{self, AppState, ServerConfig};
use rvllm::tokenizer::SmollLM230MTokenizer;

const BLOCK_SIZE: usize = 16;
const MAX_RUNNING: usize = 8;
const MAX_QUEUE: usize = 64;

fn main() -> Result<(), BoxError> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or("usage: server <model.gguf> <tokenizer.json> [addr]")?;
    let tokenizer_path = args.next().ok_or("missing tokenizer path")?;
    let addr = args.next().unwrap_or_else(|| "127.0.0.1:8000".into());

    let model = Model::load(&model_path)?;
    let tokenizer = SmollLM230MTokenizer::from_file(
        &tokenizer_path,
        model.config.eos_token_id,
        model.config.bos_token_id,
    )?;
    let context_length = model.config.context_length as usize;
    let num_blocks = MAX_RUNNING * context_length.min(2048).div_ceil(BLOCK_SIZE);
    let engine_cfg = EngineConfig {
        block_size: BLOCK_SIZE,
        num_blocks,
        max_running: MAX_RUNNING,
    };

    let (job_tx, job_rx) = tokio::sync::mpsc::unbounded_channel();
    let status = Arc::new(WorkerStatus::default());
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize, String>>();

    std::thread::scope(|scope| -> Result<(), BoxError> {
        // The engine borrows the model, so it runs on its own thread; that
        // also keeps device (CUDA) state on a single thread.
        let worker_status = status.clone();
        let model = &model;
        scope.spawn(move || {
            let runtime = match Runtime::new(model, "cpu") {
                Ok(r) => r,
                Err(e) => return drop(ready_tx.send(Err(e.to_string()))),
            };
            let backend = match runtime.backend(BLOCK_SIZE, num_blocks) {
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
                model_name: "rvllm".into(),
                max_running: MAX_RUNNING,
                max_queue: MAX_QUEUE,
                default_max_tokens: 256,
                default_temperature: 1.0,
                default_top_p: 1.0,
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
        let result = rt.block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            eprintln!("serving on http://{addr} (Ctrl-C to stop)");
            server::serve(listener, state).await
        });
        // Closes the job channel so the engine thread exits.
        rt.shutdown_background();
        result?;
        Ok(())
    })
}
