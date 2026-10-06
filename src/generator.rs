// High-level library entry point: load a GGUF model plus tokenizer once, then
// turn prompts into completions without touching the engine/scheduler types.
//
//     let generator = Generator::load("model.gguf", "tokenizer.json", "cpu")?;
//     let text = generator.complete("The capital of France is", &GenerateOptions::default())?;

use std::sync::mpsc;
use std::thread;

use crate::chat::{self, ChatMessage};
use crate::engine::{self, BoxError, EngineConfig, RequestSpec};
use crate::model::{Config, Model};
use crate::runtime::Runtime;
use crate::sampler::SamplingParams;
use crate::tokenizer::SmollLM230MTokenizer;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub block_size: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                seed: 0,
            },
            block_size: 16,
        }
    }
}

// One job for the device thread: run these requests to completion.
struct Task {
    requests: Vec<RequestSpec>,
    opts: GenerateOptions,
    num_blocks: usize,
    reply: mpsc::Sender<Result<Vec<Vec<u32>>, BoxError>>,
}

// The model and device state (CUDA context, uploaded weights) live on a
// dedicated thread for the Generator's lifetime, so they're set up once and
// never cross threads. Calls are queued to that thread and run one at a time.
pub struct Generator {
    tasks: Option<mpsc::Sender<Task>>,
    thread: Option<thread::JoinHandle<()>>,
    config: Config,
    tokenizer: SmollLM230MTokenizer,
}

fn device_thread(
    model_path: String,
    device: String,
    tasks: mpsc::Receiver<Task>,
    ready: mpsc::Sender<Result<Config, BoxError>>,
) {
    let model = match Model::load(&model_path) {
        Ok(m) => m,
        Err(e) => return drop(ready.send(Err(e.into()))),
    };
    let runtime = match Runtime::new(&model, &device) {
        Ok(r) => r,
        Err(e) => return drop(ready.send(Err(e))),
    };
    let _ = ready.send(Ok(model.config.clone()));
    while let Ok(task) = tasks.recv() {
        let result = runtime
            .backend(task.opts.block_size, task.num_blocks)
            .and_then(|backend| {
                engine::run(
                    backend,
                    &task.requests,
                    task.opts.sampling,
                    EngineConfig {
                        block_size: task.opts.block_size,
                        num_blocks: task.num_blocks,
                        max_running: usize::MAX,
                    },
                )
            })
            .map(|r| r.generated);
        let _ = task.reply.send(result);
    }
}

impl Generator {
    // `device` is "cpu" or "cuda" (the latter needs the `cuda` feature).
    pub fn load(model_path: &str, tokenizer_path: &str, device: &str) -> Result<Self, BoxError> {
        let (tasks_tx, tasks_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (model_path, device) = (model_path.to_string(), device.to_string());
        let thread = thread::spawn(move || device_thread(model_path, device, tasks_rx, ready_tx));
        let config = ready_rx
            .recv()
            .map_err(|_| "generator thread died during startup")??;
        let tokenizer = SmollLM230MTokenizer::from_file(
            tokenizer_path,
            config.eos_token_id,
            config.bos_token_id,
        )?;
        Ok(Self {
            tasks: Some(tasks_tx),
            thread: Some(thread),
            config,
            tokenizer,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &SmollLM230MTokenizer {
        &self.tokenizer
    }

    // Plain text completion of a single prompt.
    pub fn complete(&self, prompt: &str, opts: &GenerateOptions) -> Result<String, BoxError> {
        Ok(self.complete_batch(&[prompt], opts)?.remove(0))
    }

    // Several prompts run concurrently through continuous batching.
    pub fn complete_batch(
        &self,
        prompts: &[&str],
        opts: &GenerateOptions,
    ) -> Result<Vec<String>, BoxError> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let context_length = self.config.context_length as usize;
        let requests = prompts
            .iter()
            .map(|p| {
                Ok(RequestSpec {
                    tokens: self.tokenizer.encode(p)?,
                    max_tokens: opts.max_tokens,
                    arrival_step: 0,
                })
            })
            .collect::<Result<Vec<_>, BoxError>>()?;

        let longest = requests
            .iter()
            .map(|r| (r.tokens.len() + r.max_tokens).min(context_length))
            .max()
            .unwrap_or(0);
        let num_blocks = longest.div_ceil(opts.block_size).max(1) * requests.len();

        let (reply, reply_rx) = mpsc::channel();
        self.tasks
            .as_ref()
            .expect("tasks sender lives until drop")
            .send(Task {
                requests,
                opts: opts.clone(),
                num_blocks,
                reply,
            })
            .map_err(|_| "generator thread is gone")?;
        let generated = reply_rx.recv().map_err(|_| "generator thread died")??;

        let mut stops = vec![self.tokenizer.eos_token_id];
        stops.extend(self.tokenizer.token_to_id(chat::END_OF_TURN));
        generated
            .iter()
            .map(|g| {
                let ids = match g.split_last() {
                    Some((last, rest)) if stops.contains(last) => rest,
                    _ => g.as_slice(),
                };
                self.tokenizer.decode(ids)
            })
            .collect()
    }

    // Chat completion over a message history using the ChatML template
    // (SmolLM2-Instruct models).
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        opts: &GenerateOptions,
    ) -> Result<String, BoxError> {
        self.complete(&chat::chatml(messages), opts)
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        // Closing the channel ends the device thread's loop.
        self.tasks.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
