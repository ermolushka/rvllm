use cudarc::driver::CudaContext;
use cudarc::driver::safe::{CudaFunction, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

pub struct CudaRuntime {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels_mapping: HashMap<&'static str, CudaFunction>,
}

impl CudaRuntime {
    pub fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = cudarc::driver::CudaContext::new(device_id)?;
        let mut cuda_runtime = Self {
            stream: ctx.default_stream(),
            ctx,
            kernels_mapping: HashMap::new(),
        };
        cuda_runtime.load_kernels()?;
        Ok(cuda_runtime)
    }
    pub fn load_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // helper that takes the kernel source, compiles it with NVRTC, and loads the module
        let silu_ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/silu.cu"))?;
        let module = self.ctx.load_module(silu_ptx)?;
        let silu_kernel = module.load_function("silu")?;
        self.kernels_mapping.insert("silu", silu_kernel);
        Ok(())
    }
}
