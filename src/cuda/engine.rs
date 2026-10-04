// CUDA `Backend` for engine.rs: the same Scheduler-driven loop as the CPU
// path, with forward passes going through CudaModel/CudaKvStorage. Logits come
// back from CudaModel as a flat CudaSlice<f32>; token selection happens here.

use std::fmt::Display;
use std::time::{Duration, Instant};

use crate::engine::{Backend, BoxError};
use crate::model::{BatchItem, Config};
use crate::sampler::Sampler;

use super::context::CudaRuntime;
use super::kv_storage::CudaKvStorage;
use super::model::CudaModel;
use super::ops;

fn boxed<E: Display>(e: E) -> BoxError {
    e.to_string().into()
}

// Where one decode step's wall time goes, summed over the run. Printed to
// stderr when the backend is dropped if RVLLM_TIMING is set. `enqueue` is CPU
// time spent launching the forward pass; `gpu_wait` is how long the CPU then
// blocks for the GPU to finish. If enqueue is close to enqueue + gpu_wait, the
// GPU is being starved by launch overhead, not limited by its own work.
#[derive(Default)]
struct DecodeBreakdown {
    steps: usize,
    enqueue: Duration,
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
            "decode breakdown (ms/step over {} steps): enqueue={:.2} gpu_wait={:.2} logits_copy={:.2} sample={:.2}",
            self.steps,
            ms(self.enqueue),
            ms(self.gpu_wait),
            ms(self.copy),
            ms(self.sample),
        );
    }
}

pub struct CudaBackend<'a> {
    runtime: &'a CudaRuntime,
    model: &'a CudaModel,
    kv_storage: CudaKvStorage,
    breakdown: DecodeBreakdown,
}

impl<'a> CudaBackend<'a> {
    pub fn new(
        runtime: &'a CudaRuntime,
        model: &'a CudaModel,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<Self, BoxError> {
        let config = &model.config;
        let kv_storage = CudaKvStorage::new(
            runtime,
            config.n_layers as usize,
            num_blocks,
            block_size,
            config.n_kv_heads as usize,
            config.head_dim(),
        )
        .map_err(boxed)?;
        Ok(CudaBackend {
            runtime,
            model,
            kv_storage,
            breakdown: DecodeBreakdown::default(),
        })
    }
}

impl Drop for CudaBackend<'_> {
    fn drop(&mut self) {
        self.breakdown.report();
    }
}

impl Backend for CudaBackend<'_> {
    fn config(&self) -> &Config {
        &self.model.config
    }

    // Greedy takes the argmax on the GPU and copies back one integer per row,
    // instead of the whole [rows, vocab_size] logits (megabytes at larger
    // batches) and scanning them on the CPU; anything else copies the logits
    // and samples on the host.
    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError> {
        let rows = items.len();
        let vocab_size = self.model.config.vocab_size as usize;
        let runtime = self.runtime;

        let enqueue_started = Instant::now();
        let logits = if last_only {
            self.model
                .forward_last(runtime, items, &mut self.kv_storage)
        } else {
            self.model.forward(runtime, items, &mut self.kv_storage)
        }
        .map_err(boxed)?;
        // Queue the per-row argmax right behind the forward pass.
        let greedy_ids = if samplers.iter().all(|s| s.is_greedy()) {
            let mut ids = runtime.stream.alloc_zeros::<u32>(rows).map_err(boxed)?;
            ops::argmax_wrapper(runtime, &logits, vocab_size as u32, &mut ids).map_err(boxed)?;
            Some(ids)
        } else {
            None
        };
        let enqueue = enqueue_started.elapsed();

        // Sync first so the time spent waiting on the GPU is separate from the
        // copy (clone_dtoh would sync anyway).
        let wait_started = Instant::now();
        runtime.stream.synchronize().map_err(boxed)?;
        let gpu_wait = wait_started.elapsed();

        let copy_started = Instant::now();
        let (next, copy, sample) = match greedy_ids {
            Some(ids) => {
                let next = runtime.stream.clone_dtoh(&ids).map_err(boxed)?;
                (next, copy_started.elapsed(), Duration::ZERO)
            }
            None => {
                let host = runtime.stream.clone_dtoh(&logits).map_err(boxed)?;
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
            self.breakdown.enqueue += enqueue;
            self.breakdown.gpu_wait += gpu_wait;
            self.breakdown.copy += copy;
            self.breakdown.sample += sample;
        }
        Ok(next)
    }
}
