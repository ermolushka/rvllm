use cudarc::driver::CudaContext;
use cudarc::driver::safe::{CudaFunction, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

pub struct CudaRuntime {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels_mapping: HashMap<String, CudaFunction>,
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
        self.load_single_kernel("kernels/silu.cu".to_string(), "silu".to_string())?;
        self.load_single_kernel("kernels/silu_gate_multiply.cu".to_string(), "silu_gate_multiply".to_string())?;
        self.load_single_kernel("kernels/rmsnorm.cu".to_string(), "rmsnorm".to_string())?;
        Ok(())
    }

    pub fn load_single_kernel(&mut self, kernel_path: String, kernel_name: String) -> Result<(), Box<dyn std::error::Error>> {
        let source = std::fs::read_to_string(&kernel_path)?;
        let ptx = cudarc::nvrtc::compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        let kernel = module.load_function(kernel_name.as_str())?;
        self.kernels_mapping.insert(kernel_name, kernel);

        Ok(())
    }
}
