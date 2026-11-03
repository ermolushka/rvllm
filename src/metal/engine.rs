// Metal `Backend` for engine.rs: the same Scheduler-driven loop as the CPU and
// CUDA paths, with forward passes going through MetalModel/MetalKvStorage.
// Logits stay in a shared MetalBuf; token selection happens here.

use std::fmt::Display;
use std::time::{Duration, Instant};

use kv_cache_scheduler::sequence::SequenceId;

use crate::engine::{Backend, BoxError};
use crate::model::{BatchItem, Config};
use crate::sampler::Sampler;

use super::context::{MetalBuf, MetalRuntime, with_pool};
use super::kv_storage::MetalKvStorage;
use super::model::MetalModel;
use super::ops;

fn boxed<E: Display>(e: E) -> BoxError {
    e.to_string().into()
}

// Where one decode step's wall time goes, summed over the run. Printed to
// stderr when the backend is dropped if RVLLM_TIMING is set. `encode` is CPU
// time spent recording the forward pass into the command buffer; `gpu_wait` is
// how long the CPU then blocks while the GPU runs it (commit + wait).
#[derive(Default)]
struct DecodeBreakdown {
    steps: usize,
    encode: Duration,
    gpu_wait: Duration,
    copy: Duration,
    sample: Duration,
}

impl DecodeBreakdown {
    fn report(&self) {
        if self.steps == 0 || std::env::var_os("RVLLM_TIMING").is_none() {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1000.0 / self.steps as f64;
        eprintln!(
            "decode breakdown (ms/step over {} steps): encode={:.2} gpu_wait={:.2} logits_copy={:.2} sample={:.2}",
            self.steps,
            ms(self.encode),
            ms(self.gpu_wait),
            ms(self.copy),
            ms(self.sample),
        );
    }
}

pub struct MetalBackend<'a> {
    runtime: &'a MetalRuntime,
    model: &'a MetalModel,
    kv_storage: MetalKvStorage,
    breakdown: DecodeBreakdown,
}

impl<'a> MetalBackend<'a> {
    pub fn new(
        runtime: &'a MetalRuntime,
        model: &'a MetalModel,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<Self, BoxError> {
        let config = &model.config;
        let kv_storage = MetalKvStorage::new(
            runtime,
            config.n_attention_layers(),
            num_blocks,
            block_size,
            config.n_kv_heads as usize,
            config.head_dim(),
        )
        .map_err(boxed)?;
        Ok(MetalBackend {
            runtime,
            model,
            kv_storage,
            breakdown: DecodeBreakdown::default(),
        })
    }
}

impl Drop for MetalBackend<'_> {
    fn drop(&mut self) {
        self.breakdown.report();
    }
}

impl Backend for MetalBackend<'_> {
    fn config(&self) -> &Config {
        &self.model.config
    }

    // Greedy takes the argmax on the GPU and reads back one integer per row;
    // anything else reads the logits and samples on the host (they are in
    // shared memory, so "reading back" is just a copy out of the buffer).
    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError> {
        // Command buffers and encoders are autoreleased objects; scope them.
        with_pool(|| {
            let rows = items.len();
            let vocab_size = self.model.config.vocab_size as usize;
            let runtime = self.runtime;

            let encode_started = Instant::now();
            let logits = if last_only {
                self.model.forward_last(runtime, items, &mut self.kv_storage)
            } else {
                self.model.forward(runtime, items, &mut self.kv_storage)
            }
            .map_err(boxed)?;
            let greedy_ids = if samplers.iter().all(|s| s.is_greedy()) {
                let mut ids = MetalBuf::<u32>::zeros(runtime, rows);
                ops::argmax_wrapper(runtime, &logits, vocab_size as u32, &mut ids).map_err(boxed)?;
                Some(ids)
            } else {
                None
            };
            let encode = encode_started.elapsed();

            let wait_started = Instant::now();
            runtime.synchronize().map_err(boxed)?;
            let gpu_wait = wait_started.elapsed();

            let copy_started = Instant::now();
            let (next, copy, sample) = match greedy_ids {
                Some(ids) => {
                    let next = runtime.read(&ids).map_err(boxed)?;
                    (next, copy_started.elapsed(), Duration::ZERO)
                }
                None => {
                    let host = runtime.read(&logits).map_err(boxed)?;
                    let copy = copy_started.elapsed();
                    let sample_started = Instant::now();
                    let next = samplers
                        .iter_mut()
                        .enumerate()
                        .map(|(i, s)| s.sample_slice(&host[i * vocab_size..(i + 1) * vocab_size]))
                        .collect();
                    (next, copy, sample_started.elapsed())
                }
            };

            if !last_only {
                self.breakdown.steps += 1;
                self.breakdown.encode += encode;
                self.breakdown.gpu_wait += gpu_wait;
                self.breakdown.copy += copy;
                self.breakdown.sample += sample;
            }
            Ok(next)
        })
    }

    // Drops the short-conv state the sequence kept, once it finished, was
    // aborted or was preempted (the CPU and CUDA backends do the same).
    fn release(&mut self, seq_id: SequenceId) {
        self.kv_storage.conv.release(seq_id);
    }
}
