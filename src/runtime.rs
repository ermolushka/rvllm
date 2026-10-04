// Picks and owns the device-specific pieces (CPU weights as-is, or CUDA
// context + uploaded weights) so callers can build `Backend`s without caring
// which device they're on. Create it on the thread that will run the engine.

use crate::engine::{Backend, BoxError, CpuBackend};
use crate::model::Model;

pub struct Runtime<'m> {
    model: &'m Model,
    #[cfg(feature = "cuda")]
    cuda: Option<(
        crate::cuda::context::CudaRuntime,
        crate::cuda::model::CudaModel,
    )>,
}

impl<'m> Runtime<'m> {
    // `device` is "cpu" or "cuda" (the latter needs the `cuda` feature).
    pub fn new(model: &'m Model, device: &str) -> Result<Self, BoxError> {
        match device {
            "cpu" => Ok(Runtime {
                model,
                #[cfg(feature = "cuda")]
                cuda: None,
            }),
            #[cfg(feature = "cuda")]
            "cuda" => {
                let runtime = crate::cuda::context::CudaRuntime::new(0)
                    .map_err(|e| format!("CUDA init failed: {e}"))?;
                let cuda_model = crate::cuda::model::CudaModel::upload(&runtime, model)
                    .map_err(|e| format!("weight upload failed: {e}"))?;
                Ok(Runtime {
                    model,
                    cuda: Some((runtime, cuda_model)),
                })
            }
            #[cfg(not(feature = "cuda"))]
            "cuda" => Err("built without the `cuda` feature; rebuild with --features cuda".into()),
            other => Err(format!("unknown device {other:?} (expected \"cpu\" or \"cuda\")").into()),
        }
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    // A fresh backend (and KV pool) sized for `num_blocks`.
    pub fn backend(
        &self,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<Box<dyn Backend + '_>, BoxError> {
        #[cfg(feature = "cuda")]
        if let Some((runtime, cuda_model)) = &self.cuda {
            return Ok(Box::new(crate::cuda::engine::CudaBackend::new(
                runtime, cuda_model, block_size, num_blocks,
            )?));
        }
        Ok(Box::new(CpuBackend::new(
            self.model, block_size, num_blocks,
        )?))
    }
}
