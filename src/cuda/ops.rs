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

pub fn silu_gate_multiply_wrapper(
    cuda_runtime: &CudaRuntime,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = gate.len() as u32;

    if output.len() != gate.len() || up.len() != output.len() {
        return Err("silu_gate_multiply: input and output have different lengths".into());
    }
    if n == 0 {
        return Err("silu_gate_multiply: input len is 0".into());
    }
    match cuda_runtime.kernels_mapping.get("silu_gate_multiply") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(gate);
            builder.arg(up);
            builder.arg(output);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'silu_gate_multiply' not found".into()),
    }
    Ok(())
}

pub fn rmsnorm_wrapper(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    hidden_dim: u32,
    eps: f32,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = x.len() as u32;

    if output.len() != x.len() {
        return Err("RMSNorm: input and output have different lengths".into());
    }
    if n == 0 {
        return Err("RMSNorm: input len is 0".into());
    }
    if hidden_dim == 0 || n % hidden_dim != 0 {
        return Err("RMSNorm: input length is not a multiple of hidden_dim".into());
    }
    if weight.len() as u32 != hidden_dim {
        return Err("RMSNorm: weight length must equal hidden_dim".into());
    }
    match cuda_runtime.kernels_mapping.get("rmsnorm") {
        Some(kernel) => {
            let num_rows = n / hidden_dim;
            let block_size: u32 = 256;
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(x);
            builder.arg(weight);
            builder.arg(output);
            builder.arg(&hidden_dim);
            builder.arg(&eps);
            let cfg = LaunchConfig {
                grid_dim: (num_rows, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: block_size * std::mem::size_of::<f32>() as u32,
            };
            unsafe { builder.launch(cfg) }?;
        }
        None => return Err("Error: kernel 'rmsnorm' not found".into()),
    }
    Ok(())
}

// x, out: [n_rows, head_dim] flattened. cos, sin: [n_rows, half] flattened,
// pre-expanded on the host so row i of cos/sin lines up with row i of x.
pub fn rope_wrapper(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    head_dim: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if output.len() != x.len() {
        return Err("RoPE: input and output have different lengths".into());
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err("RoPE: head_dim must be a positive even number".into());
    }
    let n = x.len() as u32;
    if n == 0 || n % head_dim != 0 {
        return Err("RoPE: input length is not a multiple of head_dim".into());
    }
    let half = head_dim / 2;
    let n_rows = n / head_dim;
    let n_pairs = n_rows * half;
    if cos.len() as u32 != n_pairs || sin.len() as u32 != n_pairs {
        return Err("RoPE: cos/sin length must equal n_rows * (head_dim / 2)".into());
    }

    match cuda_runtime.kernels_mapping.get("rope") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(x);
            builder.arg(cos);
            builder.arg(sin);
            builder.arg(output);
            builder.arg(&head_dim);
            builder.arg(&half);
            builder.arg(&n_pairs);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n_pairs)) }?;
        }
        None => return Err("Error: kernel 'rope' not found".into()),
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
    fn silu_gate_multiply_matches_candle() {
        let Some(rt) = runtime() else { return };
        // Different ramps for gate and up so a swapped-argument bug shows up.
        let gate: Vec<f32> = (0..1027).map(|i| (i as f32 - 500.0) * 0.05).collect();
        let up: Vec<f32> = (0..1027).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();

        let gate_dev = rt.stream.clone_htod(&gate).unwrap();
        let up_dev = rt.stream.clone_htod(&up).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(gate.len()).unwrap();
        silu_gate_multiply_wrapper(&rt, &gate_dev, &up_dev, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let cpu = candle_core::Device::Cpu;
        let gate_t = candle_core::Tensor::from_slice(&gate, gate.len(), &cpu).unwrap();
        let up_t = candle_core::Tensor::from_slice(&up, up.len(), &cpu).unwrap();
        let want: Vec<f32> = candle_nn::ops::silu(&gate_t).unwrap().mul(&up_t).unwrap().to_vec1().unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let hand = silu_cpu(gate[i]) * up[i];
            let tol = 1e-5 * w.abs().max(1.0);
            assert!((g - w).abs() < tol, "i={i}: got {g}, candle {w}");
            assert!((g - hand).abs() < tol, "i={i}: got {g}, formula {hand}");
        }
    }

    #[test]
    fn silu_gate_multiply_rejects_mismatched_lengths() {
        let Some(rt) = runtime() else { return };
        let gate = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let up_short = rt.stream.clone_htod(&[1.0f32, 2.0]).unwrap();
        let up_ok = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut out_short = rt.stream.alloc_zeros::<f32>(2).unwrap();
        let mut out_ok = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(silu_gate_multiply_wrapper(&rt, &gate, &up_ok, &mut out_short).is_err());
        assert!(silu_gate_multiply_wrapper(&rt, &gate, &up_short, &mut out_ok).is_err());
    }

    #[test]
    fn silu_rejects_mismatched_lengths() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(2).unwrap();
        assert!(silu_wrapper(&rt, &input, &mut output).is_err());
    }

    fn rms_norm_cpu(row: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let rms = (mean_sq + eps).sqrt();
        row.iter().zip(weight).map(|(v, w)| (v / rms) * w).collect()
    }

    #[test]
    fn rmsnorm_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        // hidden_dim = 37 doesn't divide 256 (the kernel's block size), so this
        // exercises the grid-stride tail in both the sum and the write-back loop.
        // 5 rows so the "one block per row" grid dimension is actually tested,
        // not just a single-block launch.
        let hidden_dim = 37usize;
        let num_rows = 5usize;
        let eps = 1e-5f32;

        let host: Vec<f32> = (0..hidden_dim * num_rows)
            .map(|i| (i as f32 - (hidden_dim * num_rows) as f32 / 2.0) * 0.05)
            .collect();
        // Non-uniform weight so a row/column indexing bug (e.g. weight indexed
        // by flat offset instead of column) shows up as a mismatch.
        let weight: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i as f32) * 0.1).collect();

        let input = rt.stream.clone_htod(&host).unwrap();
        let weight_dev = rt.stream.clone_htod(&weight).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        rmsnorm_wrapper(&rt, &input, &weight_dev, hidden_dim as u32, eps, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        for row in 0..num_rows {
            let start = row * hidden_dim;
            let row_in = &host[start..start + hidden_dim];
            let want = rms_norm_cpu(row_in, &weight, eps);
            for i in 0..hidden_dim {
                let g = got[start + i];
                let w = want[i];
                let tol = 1e-5 * w.abs().max(1.0);
                assert!((g - w).abs() < tol, "row={row} i={i}: got {g}, want {w}");
            }
        }
    }

    #[test]
    fn rmsnorm_matches_candle_reference() {
        let Some(rt) = runtime() else { return };
        let hidden_dim = 8usize;
        let num_rows = 3usize;
        let eps = 1e-5f32;

        let host: Vec<f32> = (0..hidden_dim * num_rows).map(|i| (i as f32 - 12.0) * 0.3).collect();
        let weight: Vec<f32> = (0..hidden_dim).map(|i| 0.5 + i as f32 * 0.2).collect();

        let input = rt.stream.clone_htod(&host).unwrap();
        let weight_dev = rt.stream.clone_htod(&weight).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        rmsnorm_wrapper(&rt, &input, &weight_dev, hidden_dim as u32, eps, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let cpu = candle_core::Device::Cpu;
        let x_t = candle_core::Tensor::from_slice(&host, (num_rows, hidden_dim), &cpu).unwrap();
        let w_t = candle_core::Tensor::from_slice(&weight, hidden_dim, &cpu).unwrap();
        let want: Vec<f32> = crate::model::rms_norm(&x_t, &w_t, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-5 * w.abs().max(1.0);
            assert!((g - w).abs() < tol, "i={i}: got {g}, candle {w}");
        }
    }

    #[test]
    fn rmsnorm_rejects_mismatched_output_length() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let weight = rt.stream.clone_htod(&[1.0f32, 1.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(rmsnorm_wrapper(&rt, &input, &weight, 2, 1e-5, &mut output).is_err());
    }

    #[test]
    fn rmsnorm_rejects_hidden_dim_not_dividing_input() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let weight = rt.stream.clone_htod(&[1.0f32, 1.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(rmsnorm_wrapper(&rt, &input, &weight, 2, 1e-5, &mut output).is_err());
    }

    #[test]
    fn rmsnorm_rejects_wrong_weight_length() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let weight = rt.stream.clone_htod(&[1.0f32, 1.0, 1.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(4).unwrap();
        assert!(rmsnorm_wrapper(&rt, &input, &weight, 2, 1e-5, &mut output).is_err());
    }

    #[test]
    fn rope_position_one_hand_computed() {
        let Some(rt) = runtime() else { return };
        // Same case as model.rs's rope_position_one_hand_computed: head_dim=4,
        // rope_theta=10000 -> theta_0=1, theta_1=0.01. Row 0 = position 0
        // (identity), row 1 = position 1. Rotate-half pairs are (x0,x2) and
        // (x1,x3); both pairs' "x" component is 1, "y" component is 0.
        let head_dim = 4u32;
        let x: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        // cos/sin pre-expanded per row: row 0 is position 0 (angle 0), row 1 is
        // position 1 (angles 1.0 and 0.01 for pair 0 and pair 1).
        let cos: Vec<f32> = vec![0f32.cos(), 0f32.cos(), 1f32.cos(), 0.01f32.cos()];
        let sin: Vec<f32> = vec![0f32.sin(), 0f32.sin(), 1f32.sin(), 0.01f32.sin()];

        let x_dev = rt.stream.clone_htod(&x).unwrap();
        let cos_dev = rt.stream.clone_htod(&cos).unwrap();
        let sin_dev = rt.stream.clone_htod(&sin).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(x.len()).unwrap();
        rope_wrapper(&rt, &x_dev, &cos_dev, &sin_dev, &mut output, head_dim).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let expected_row0 = [1f32, 1.0, 0.0, 0.0];
        let expected_row1 = [1f32.cos(), 0.01f32.cos(), 1f32.sin(), 0.01f32.sin()];
        for (got, want) in got[0..4].iter().zip(expected_row0.iter()) {
            assert!((got - want).abs() < 1e-4, "row0: got {got}, want {want}");
        }
        for (got, want) in got[4..8].iter().zip(expected_row1.iter()) {
            assert!((got - want).abs() < 1e-4, "row1: got {got}, want {want}");
        }
    }

    #[test]
    fn rope_matches_candle_reference() {
        let Some(rt) = runtime() else { return };
        let head_dim = 8usize;
        let half = head_dim / 2;
        let positions = [0usize, 1, 5, 12];
        let rope_theta = 10000.0f32;

        let host: Vec<f32> = (0..head_dim * positions.len())
            .map(|i| (i as f32 - 16.0) * 0.1)
            .collect();

        let cpu = candle_core::Device::Cpu;
        let x_t = candle_core::Tensor::from_slice(&host, (positions.len(), head_dim), &cpu).unwrap();
        let want: Vec<f32> = crate::model::rope(&x_t, rope_theta, &cpu, &positions)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Pre-expand cos/sin per row, same as the host is responsible for doing
        // before calling rope_wrapper.
        let (cos_t, sin_t) = crate::model::rope_cos_sin(rope_theta, head_dim, &positions, &cpu).unwrap();
        let cos: Vec<f32> = cos_t.flatten_all().unwrap().to_vec1().unwrap();
        let sin: Vec<f32> = sin_t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(cos.len(), positions.len() * half);

        let x_dev = rt.stream.clone_htod(&host).unwrap();
        let cos_dev = rt.stream.clone_htod(&cos).unwrap();
        let sin_dev = rt.stream.clone_htod(&sin).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        rope_wrapper(&rt, &x_dev, &cos_dev, &sin_dev, &mut output, head_dim as u32).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-4 * w.abs().max(1.0);
            assert!((g - w).abs() < tol, "i={i}: got {g}, candle {w}");
        }
    }

    #[test]
    fn rope_rejects_mismatched_output_length() {
        let Some(rt) = runtime() else { return };
        let x = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let cos = rt.stream.clone_htod(&[1.0f32, 1.0]).unwrap();
        let sin = rt.stream.clone_htod(&[0.0f32, 0.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(rope_wrapper(&rt, &x, &cos, &sin, &mut output, 4).is_err());
    }

    #[test]
    fn rope_rejects_odd_head_dim() {
        let Some(rt) = runtime() else { return };
        let x = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let cos = rt.stream.clone_htod(&[1.0f32]).unwrap();
        let sin = rt.stream.clone_htod(&[0.0f32]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(rope_wrapper(&rt, &x, &cos, &sin, &mut output, 3).is_err());
    }

    #[test]
    fn rope_rejects_wrong_cos_sin_length() {
        let Some(rt) = runtime() else { return };
        let x = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let cos = rt.stream.clone_htod(&[1.0f32]).unwrap();
        let sin = rt.stream.clone_htod(&[0.0f32]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(4).unwrap();
        assert!(rope_wrapper(&rt, &x, &cos, &sin, &mut output, 4).is_err());
    }
}
