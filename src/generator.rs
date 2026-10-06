// High-level library entry point: load a GGUF model plus tokenizer once, then
// turn prompts into completions without touching the engine/scheduler types.
//
//     let generator = Generator::load("model.gguf", "tokenizer.json", "cpu")?;
//     let text = generator.complete("The capital of France is", &GenerateOptions::default())?;

use crate::chat::{self, ChatMessage};
use crate::engine::{self, BoxError, EngineConfig, RequestSpec};
use crate::model::Model;
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

pub struct Generator {
    model: Model,
    tokenizer: SmollLM230MTokenizer,
    device: String,
}

impl Generator {
    // `device` is "cpu" or "cuda" (the latter needs the `cuda` feature).
    // Device state is created per call, so on CUDA each call re-uploads the
    // weights; for long-lived GPU use, drive `Runtime` and `Engine` directly.
    pub fn load(model_path: &str, tokenizer_path: &str, device: &str) -> Result<Self, BoxError> {
        let model = Model::load(model_path)?;
        let tokenizer = SmollLM230MTokenizer::from_file(
            tokenizer_path,
            model.config.eos_token_id,
            model.config.bos_token_id,
        )?;
        if !matches!(device, "cpu" | "cuda") {
            return Err(format!("unknown device {device:?} (expected \"cpu\" or \"cuda\")").into());
        }
        Ok(Self {
            model,
            tokenizer,
            device: device.to_string(),
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
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
        let context_length = self.model.config.context_length as usize;
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
        let num_blocks = longest.div_ceil(opts.block_size).max(1) * requests.len().max(1);

        let runtime = Runtime::new(&self.model, &self.device)?;
        let backend = runtime.backend(opts.block_size, num_blocks)?;
        let result = engine::run(
            backend,
            &requests,
            opts.sampling,
            EngineConfig {
                block_size: opts.block_size,
                num_blocks,
                max_running: usize::MAX,
            },
        )?;

        let mut stops = vec![self.tokenizer.eos_token_id];
        stops.extend(self.tokenizer.token_to_id(chat::END_OF_TURN));
        result
            .generated
            .iter()
            .map(|generated| {
                let ids = match generated.split_last() {
                    Some((last, rest)) if stops.contains(last) => rest,
                    _ => generated.as_slice(),
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
