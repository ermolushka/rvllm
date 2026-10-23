use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaContext;
use cudarc::driver::safe::{CudaFunction, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

pub struct CudaRuntime {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels_mapping: HashMap<String, CudaFunction>,
    pub blas: CudaBlas,
}

impl CudaRuntime {
    pub fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = cudarc::driver::CudaContext::new(device_id)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let mut cuda_runtime = Self {
            stream,
            ctx,
            kernels_mapping: HashMap::new(),
            blas,
        };
        cuda_runtime.load_kernels()?;
        Ok(cuda_runtime)
    }
    pub fn load_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Embedded at compile time (relative to this file, not the process's
        // cwd at runtime) - `std::fs::read_to_string` on a relative path was
        // the earlier approach, but it silently depended on launching the
        // binary from inside src/cuda/, which no normal `cargo run`/`cargo
        // test` invocation does.
        self.load_single_kernel(include_str!("kernels/silu.cu"), "silu".to_string())?;
        self.load_single_kernel(
            include_str!("kernels/silu_gate_multiply.cu"),
            "silu_gate_multiply".to_string(),
        )?;
        self.load_single_kernel(include_str!("kernels/rmsnorm.cu"), "rmsnorm".to_string())?;
        self.load_single_kernel(include_str!("kernels/rope.cu"), "rope".to_string())?;
        self.load_single_kernel(include_str!("kernels/kv_write.cu"), "kv_write".to_string())?;
        self.load_single_kernel(
            include_str!("kernels/kv_gather.cu"),
            "kv_gather".to_string(),
        )?;
        self.load_single_kernel(include_str!("kernels/softmax.cu"), "softmax".to_string())?;
        self.load_single_kernel(include_str!("kernels/add.cu"), "add".to_string())?;
        self.load_single_kernel(
            include_str!("kernels/embed_lookup.cu"),
            "embed_lookup".to_string(),
        )?;
        self.load_single_kernel(
            include_str!("kernels/mask_add_broadcast.cu"),
            "mask_add_broadcast".to_string(),
        )?;
        self.load_single_kernel(
            include_str!("kernels/transpose_axes12.cu"),
            "transpose_axes12".to_string(),
        )?;
        self.load_single_kernel(
            include_str!("kernels/narrow_last_row.cu"),
            "narrow_last_row".to_string(),
        )?;
        self.load_single_kernel(include_str!("kernels/argmax.cu"), "argmax".to_string())?;
        self.load_single_kernel(
            include_str!("kernels/f32_to_f16.cu"),
            "f32_to_f16".to_string(),
        )?;
        self.load_single_kernel(include_str!("kernels/add_bias.cu"), "add_bias".to_string())?;
        self.load_single_kernel(
            include_str!("kernels/gelu_gate_multiply.cu"),
            "gelu_gate_multiply".to_string(),
        )?;
        self.load_single_kernel(
            include_str!("kernels/scale_inplace.cu"),
            "scale_inplace".to_string(),
        )?;
        self.load_module_kernels(
            include_str!("kernels/short_conv.cu"),
            &["short_conv_bx", "short_conv_filter_gate"],
        )?;
        Ok(())
    }

    // Like `load_single_kernel`, for a source file with several kernels: one
    // compile, then every named function is taken from that module.
    pub fn load_module_kernels(
        &mut self,
        source: &str,
        kernel_names: &[&str],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        for name in kernel_names {
            let kernel = module.load_function(name)?;
            self.kernels_mapping.insert(name.to_string(), kernel);
        }
        Ok(())
    }

    pub fn load_single_kernel(
        &mut self,
        source: &str,
        kernel_name: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        let kernel = module.load_function(kernel_name.as_str())?;
        self.kernels_mapping.insert(kernel_name, kernel);

        Ok(())
    }
}

// For GPU tests: a runtime on device 0, or None when there is no usable CUDA
// device. Without a device the test is skipped with a note on stderr; set
// RVLLM_REQUIRE_CUDA to make a missing device a test failure instead, so a
// GPU test can't pass without running anything on the GPU.
#[cfg(test)]
pub fn test_runtime() -> Option<CudaRuntime> {
    match CudaRuntime::new(0) {
        Ok(runtime) => Some(runtime),
        Err(e) => {
            assert!(
                std::env::var_os("RVLLM_REQUIRE_CUDA").is_none(),
                "RVLLM_REQUIRE_CUDA is set but CUDA init failed: {e}"
            );
            eprintln!("skipping CUDA test: {e}");
            None
        }
    }
}
