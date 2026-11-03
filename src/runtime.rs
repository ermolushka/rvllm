// Picks and owns the device-specific pieces (CPU weights as-is, or CUDA / Metal
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
    #[cfg(feature = "metal")]
    metal: Option<(
        crate::metal::context::MetalRuntime,
        crate::metal::model::MetalModel,
    )>,
}

impl<'m> Runtime<'m> {
    // `device` is "cpu", "cuda" or "metal" (the latter two need their feature).
    pub fn new(model: &'m Model, device: &str) -> Result<Self, BoxError> {
        match device {
            "cpu" => Ok(Runtime {
                model,
                #[cfg(feature = "cuda")]
                cuda: None,
                #[cfg(feature = "metal")]
                metal: None,
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
                    #[cfg(feature = "metal")]
                    metal: None,
                })
            }
            #[cfg(not(feature = "cuda"))]
            "cuda" => Err("built without the `cuda` feature; rebuild with --features cuda".into()),
            #[cfg(feature = "metal")]
            "metal" => {
                // Metal hands back autoreleased objects; scope them to the setup.
                let (runtime, metal_model) = crate::metal::context::with_pool(|| {
                    let runtime = crate::metal::context::MetalRuntime::new()
                        .map_err(|e| format!("Metal init failed: {e}"))?;
                    let metal_model = crate::metal::model::MetalModel::upload(&runtime, model)
                        .map_err(|e| format!("weight upload failed: {e}"))?;
                    Ok::<_, BoxError>((runtime, metal_model))
                })?;
                Ok(Runtime {
                    model,
                    #[cfg(feature = "cuda")]
                    cuda: None,
                    metal: Some((runtime, metal_model)),
                })
            }
            #[cfg(not(feature = "metal"))]
            "metal" => Err("built without the `metal` feature; rebuild with --features metal".into()),
            other => Err(format!(
                "unknown device {other:?} (expected \"cpu\", \"cuda\" or \"metal\")"
            )
            .into()),
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
        #[cfg(feature = "metal")]
        if let Some((runtime, metal_model)) = &self.metal {
            return Ok(Box::new(crate::metal::engine::MetalBackend::new(
                runtime, metal_model, block_size, num_blocks,
            )?));
        }
        Ok(Box::new(CpuBackend::new(
            self.model, block_size, num_blocks,
        )?))
    }
}
