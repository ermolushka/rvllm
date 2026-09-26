use super::context::CudaRuntime;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

pub fn silu_wrapper(
    cuda_runtime: &CudaRuntime,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = input.len() as u32;

    if output.len() != input.len() {
        return Err("Silu: input and output have different lengths".into());
    }
    if n == 0 {
        return Err("Silu: input len is 0".into());
    }
    match cuda_runtime.kernels_mapping.get("silu") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(input);
            builder.arg(output);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'silu' not found".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // None when there is no usable CUDA device, so tests skip instead of fail.
    fn runtime() -> Option<CudaRuntime> {
        CudaRuntime::new(0).ok()
    }

    fn silu_cpu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn silu_matches_cpu_reference() {
        let Some(rt) = runtime() else { return };
        // 1027 is not a multiple of any block size, so the tail guard is exercised.
        // Range covers negatives, exact zero (i == 500) and large magnitudes.
        let host: Vec<f32> = (0..1027).map(|i| (i as f32 - 500.0) * 0.05).collect();

        let input = rt.stream.clone_htod(&host).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        silu_wrapper(&rt, &input, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        // Candle's CPU silu is the same op the CPU model path uses.
        let candle_want: Vec<f32> = candle_nn::ops::silu(
            &candle_core::Tensor::from_slice(&host, host.len(), &candle_core::Device::Cpu).unwrap(),
        )
        .unwrap()
        .to_vec1()
        .unwrap();

        assert_eq!(got.len(), host.len());
        for (i, (&g, &x)) in got.iter().zip(&host).enumerate() {
            let want = silu_cpu(x);
            let tol = 1e-5 * want.abs().max(1.0);
            assert!((g - want).abs() < tol, "i={i} x={x}: got {g}, want {want}");
            let c = candle_want[i];
            assert!((g - c).abs() < tol, "i={i} x={x}: got {g}, candle {c}");
        }
    }

    #[test]
    fn silu_rejects_mismatched_lengths() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(2).unwrap();
        assert!(silu_wrapper(&rt, &input, &mut output).is_err());
    }
}
