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

// k, v: [n_tokens, n_kv_heads, head_dim] flattened. block_ids, offsets: one
// entry per token, giving its destination slot in out_k/out_v (which are
// [num_blocks, block_size, n_kv_heads, head_dim] flattened, caller-owned
// cache buffers mutated in place).
pub fn kv_write_wrapper(
    cuda_runtime: &CudaRuntime,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    block_ids: &CudaSlice<u32>,
    offsets: &CudaSlice<u32>,
    out_k: &mut CudaSlice<f32>,
    out_v: &mut CudaSlice<f32>,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if k.len() != v.len() {
        return Err("kv_write: k and v have different lengths".into());
    }
    let n = k.len() as u32;
    if n == 0 {
        return Err("kv_write: k/v len is 0".into());
    }
    if n_kv_heads == 0 || head_dim == 0 || block_size == 0 {
        return Err("kv_write: n_kv_heads/head_dim/block_size must be positive".into());
    }
    let per_token = n_kv_heads * head_dim;
    if n % per_token != 0 {
        return Err("kv_write: k/v length is not a multiple of n_kv_heads * head_dim".into());
    }
    let n_tokens = n / per_token;
    if block_ids.len() as u32 != n_tokens || offsets.len() as u32 != n_tokens {
        return Err("kv_write: block_ids/offsets length must equal n_tokens".into());
    }
    if out_k.len() != out_v.len() {
        return Err("kv_write: out_k and out_v have different lengths".into());
    }
    let slot_size = block_size * per_token;
    if out_k.len() as u32 % slot_size != 0 {
        return Err(
            "kv_write: out_k/out_v length is not a multiple of block_size * n_kv_heads * head_dim"
                .into(),
        );
    }

    match cuda_runtime.kernels_mapping.get("kv_write") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(k);
            builder.arg(v);
            builder.arg(block_ids);
            builder.arg(offsets);
            builder.arg(out_k);
            builder.arg(out_v);
            builder.arg(&n_kv_heads);
            builder.arg(&head_dim);
            builder.arg(&block_size);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'kv_write' not found".into()),
    }
    Ok(())
}

// cache: [num_blocks, block_size, n_kv_heads, head_dim] flattened, one
// layer's K (or V) storage. block_idx: [batch * max_blocks] flattened,
// built by the caller (CPU's GatherPlan equivalent) - sequence i's block
// table, padded to max_blocks by repeating its last block. output:
// [batch, n_kv_heads, ctx_len, head_dim] flattened, already in the
// transposed layout attention expects (no separate transpose step).
pub fn kv_gather_wrapper(
    cuda_runtime: &CudaRuntime,
    cache: &CudaSlice<f32>,
    block_idx: &CudaSlice<u32>,
    output: &mut CudaSlice<f32>,
    batch: u32,
    max_blocks: u32,
    ctx_len: u32,
    block_size: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch == 0
        || max_blocks == 0
        || ctx_len == 0
        || block_size == 0
        || n_kv_heads == 0
        || head_dim == 0
    {
        return Err(
            "kv_gather: batch/max_blocks/ctx_len/block_size/n_kv_heads/head_dim must be positive"
                .into(),
        );
    }
    if block_idx.len() as u32 != batch * max_blocks {
        return Err("kv_gather: block_idx length must equal batch * max_blocks".into());
    }
    let slot_size = block_size * n_kv_heads * head_dim;
    if cache.is_empty() || cache.len() as u32 % slot_size != 0 {
        return Err(
            "kv_gather: cache length is not a multiple of block_size * n_kv_heads * head_dim"
                .into(),
        );
    }
    let n = batch * n_kv_heads * ctx_len * head_dim;
    if output.len() as u32 != n {
        return Err(
            "kv_gather: output length must equal batch * n_kv_heads * ctx_len * head_dim".into(),
        );
    }
    // every position read must fall inside a real block slot
    if ctx_len > max_blocks * block_size {
        return Err("kv_gather: ctx_len must not exceed max_blocks * block_size".into());
    }

    match cuda_runtime.kernels_mapping.get("kv_gather") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(cache);
            builder.arg(block_idx);
            builder.arg(output);
            builder.arg(&batch);
            builder.arg(&max_blocks);
            builder.arg(&ctx_len);
            builder.arg(&block_size);
            builder.arg(&n_kv_heads);
            builder.arg(&head_dim);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'kv_gather' not found".into()),
    }
    Ok(())
}

// x, out: [n_rows, row_len] flattened, last-dim softmax (matches
// candle_nn::ops::softmax(&scores, D::Minus1)). Masked positions are
// expected to carry a literal f32::NEG_INFINITY, same convention as
// model.rs's batch_mask - exp(-inf - finite) == 0, no special-casing needed.
pub fn softmax_wrapper(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    row_len: u32,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = x.len() as u32;

    if output.len() != x.len() {
        return Err("Softmax: input and output have different lengths".into());
    }
    if n == 0 {
        return Err("Softmax: input len is 0".into());
    }
    if row_len == 0 || n % row_len != 0 {
        return Err("Softmax: input length is not a multiple of row_len".into());
    }
    match cuda_runtime.kernels_mapping.get("softmax") {
        Some(kernel) => {
            let num_rows = n / row_len;
            let block_size: u32 = 256;
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(x);
            builder.arg(output);
            builder.arg(&row_len);
            let cfg = LaunchConfig {
                grid_dim: (num_rows, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: block_size * std::mem::size_of::<f32>() as u32,
            };
            unsafe { builder.launch(cfg) }?;
        }
        None => return Err("Error: kernel 'softmax' not found".into()),
    }
    Ok(())
}

pub fn add_wrapper(
    cuda_runtime: &CudaRuntime,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = a.len() as u32;
    if a.len() != b.len() || a.len() != output.len() {
        return Err("add: a/b/output have different lengths".into());
    }
    if n == 0 {
        return Err("add: input len is 0".into());
    }
    match cuda_runtime.kernels_mapping.get("add") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);
            builder.arg(a);
            builder.arg(b);
            builder.arg(output);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'add' not found".into()),
    }
    Ok(())
}

// token_ids: [n_tokens]. embedding: [vocab_size, hidden_dim] flattened.
// output: [n_tokens, hidden_dim] flattened.
pub fn embed_lookup_wrapper(
    cuda_runtime: &CudaRuntime,
    token_ids: &CudaSlice<u32>,
    embedding: &CudaSlice<f32>,
    hidden_dim: u32,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if hidden_dim == 0 {
        return Err("embed_lookup: hidden_dim must be positive".into());
    }
    if embedding.is_empty() || embedding.len() as u32 % hidden_dim != 0 {
        return Err("embed_lookup: embedding length is not a multiple of hidden_dim".into());
    }
    let n = token_ids.len() as u32 * hidden_dim;
    if output.len() as u32 != n {
        return Err("embed_lookup: output length must equal n_tokens * hidden_dim".into());
    }
    if n == 0 {
        return Err("embed_lookup: token_ids is empty".into());
    }
    match cuda_runtime.kernels_mapping.get("embed_lookup") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);
            builder.arg(token_ids);
            builder.arg(embedding);
            builder.arg(output);
            builder.arg(&hidden_dim);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'embed_lookup' not found".into()),
    }
    Ok(())
}

// scores: [batch, n_kv_heads, group_size, q_len, ctx_len] flattened, mutated
// in place. mask: [batch, q_len, ctx_len] flattened.
pub fn mask_add_broadcast_wrapper(
    cuda_runtime: &CudaRuntime,
    scores: &mut CudaSlice<f32>,
    mask: &CudaSlice<f32>,
    n_kv_heads: u32,
    group_size: u32,
    q_len: u32,
    ctx_len: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if n_kv_heads == 0 || group_size == 0 || q_len == 0 || ctx_len == 0 {
        return Err(
            "mask_add_broadcast: n_kv_heads/group_size/q_len/ctx_len must be positive".into(),
        );
    }
    let per_batch = n_kv_heads * group_size * q_len * ctx_len;
    let n = scores.len() as u32;
    if n == 0 || n % per_batch != 0 {
        return Err(
            "mask_add_broadcast: scores length is not a multiple of n_kv_heads * group_size * q_len * ctx_len"
                .into(),
        );
    }
    let batch = n / per_batch;
    if mask.len() as u32 != batch * q_len * ctx_len {
        return Err("mask_add_broadcast: mask length must equal batch * q_len * ctx_len".into());
    }
    match cuda_runtime.kernels_mapping.get("mask_add_broadcast") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);
            builder.arg(scores);
            builder.arg(mask);
            builder.arg(&n_kv_heads);
            builder.arg(&group_size);
            builder.arg(&q_len);
            builder.arg(&ctx_len);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'mask_add_broadcast' not found".into()),
    }
    Ok(())
}

// Swaps the middle two axes of a 4D tensor: input [d0, d1, d2, d3] ->
// output [d0, d2, d1, d3] (both flattened).
pub fn transpose_axes12_wrapper(
    cuda_runtime: &CudaRuntime,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if d0 == 0 || d1 == 0 || d2 == 0 || d3 == 0 {
        return Err("transpose_axes12: d0/d1/d2/d3 must be positive".into());
    }
    let n = d0 * d1 * d2 * d3;
    if input.len() as u32 != n {
        return Err("transpose_axes12: input length must equal d0*d1*d2*d3".into());
    }
    if output.len() as u32 != n {
        return Err("transpose_axes12: output length must equal d0*d1*d2*d3".into());
    }
    match cuda_runtime.kernels_mapping.get("transpose_axes12") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);
            builder.arg(input);
            builder.arg(output);
            builder.arg(&d0);
            builder.arg(&d1);
            builder.arg(&d2);
            builder.arg(&d3);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'transpose_axes12' not found".into()),
    }
    Ok(())
}

// x: [batch, q_len, hidden_dim] flattened. output: [batch, hidden_dim]
// flattened, each batch item's last row only.
pub fn narrow_last_row_wrapper(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    q_len: u32,
    hidden_dim: u32,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if q_len == 0 || hidden_dim == 0 {
        return Err("narrow_last_row: q_len/hidden_dim must be positive".into());
    }
    let per_batch = q_len * hidden_dim;
    if x.is_empty() || x.len() as u32 % per_batch != 0 {
        return Err("narrow_last_row: x length is not a multiple of q_len * hidden_dim".into());
    }
    let batch = x.len() as u32 / per_batch;
    let n = batch * hidden_dim;
    if output.len() as u32 != n {
        return Err("narrow_last_row: output length must equal batch * hidden_dim".into());
    }
    match cuda_runtime.kernels_mapping.get("narrow_last_row") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);
            builder.arg(x);
            builder.arg(output);
            builder.arg(&q_len);
            builder.arg(&hidden_dim);
            builder.arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }?;
        }
        None => return Err("Error: kernel 'narrow_last_row' not found".into()),
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
        let want: Vec<f32> = candle_nn::ops::silu(&gate_t)
            .unwrap()
            .mul(&up_t)
            .unwrap()
            .to_vec1()
            .unwrap();

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
        rmsnorm_wrapper(
            &rt,
            &input,
            &weight_dev,
            hidden_dim as u32,
            eps,
            &mut output,
        )
        .unwrap();
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

        let host: Vec<f32> = (0..hidden_dim * num_rows)
            .map(|i| (i as f32 - 12.0) * 0.3)
            .collect();
        let weight: Vec<f32> = (0..hidden_dim).map(|i| 0.5 + i as f32 * 0.2).collect();

        let input = rt.stream.clone_htod(&host).unwrap();
        let weight_dev = rt.stream.clone_htod(&weight).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        rmsnorm_wrapper(
            &rt,
            &input,
            &weight_dev,
            hidden_dim as u32,
            eps,
            &mut output,
        )
        .unwrap();
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
        let x_t =
            candle_core::Tensor::from_slice(&host, (positions.len(), head_dim), &cpu).unwrap();
        let want: Vec<f32> = crate::model::rope(&x_t, rope_theta, &cpu, &positions)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Pre-expand cos/sin per row, same as the host is responsible for doing
        // before calling rope_wrapper.
        let (cos_t, sin_t) =
            crate::model::rope_cos_sin(rope_theta, head_dim, &positions, &cpu).unwrap();
        let cos: Vec<f32> = cos_t.flatten_all().unwrap().to_vec1().unwrap();
        let sin: Vec<f32> = sin_t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(cos.len(), positions.len() * half);

        let x_dev = rt.stream.clone_htod(&host).unwrap();
        let cos_dev = rt.stream.clone_htod(&cos).unwrap();
        let sin_dev = rt.stream.clone_htod(&sin).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        rope_wrapper(
            &rt,
            &x_dev,
            &cos_dev,
            &sin_dev,
            &mut output,
            head_dim as u32,
        )
        .unwrap();
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

    // Same layout as kv_storage.rs's write_tokens_then_gather_round_trips_across_blocks:
    // 3 blocks of 2 slots. Sequence A: 3 tokens over blocks 0,1. Sequence B: 1
    // token in block 2. n_kv_heads=1, head_dim=2, one distinguishable value
    // per token so cross-wiring between tokens would show up as a mismatch.
    #[test]
    fn kv_write_round_trips_across_blocks() {
        let Some(rt) = runtime() else { return };
        let n_kv_heads = 1u32;
        let head_dim = 2u32;
        let block_size = 2u32;
        let num_blocks = 3u32;

        // Sequence A: tokens at (block 0, offset 0), (block 0, offset 1),
        // (block 1, offset 0). Sequence B: token at (block 2, offset 0).
        let k: Vec<f32> = vec![10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 20.0, 20.5];
        let v = k.clone();
        let block_ids: Vec<u32> = vec![0, 0, 1, 2];
        let offsets: Vec<u32> = vec![0, 1, 0, 0];

        let k_dev = rt.stream.clone_htod(&k).unwrap();
        let v_dev = rt.stream.clone_htod(&v).unwrap();
        let block_ids_dev = rt.stream.clone_htod(&block_ids).unwrap();
        let offsets_dev = rt.stream.clone_htod(&offsets).unwrap();
        let cache_len = (num_blocks * block_size * n_kv_heads * head_dim) as usize;
        let mut out_k = rt.stream.alloc_zeros::<f32>(cache_len).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(cache_len).unwrap();

        kv_write_wrapper(
            &rt,
            &k_dev,
            &v_dev,
            &block_ids_dev,
            &offsets_dev,
            &mut out_k,
            &mut out_v,
            n_kv_heads,
            head_dim,
            block_size,
        )
        .unwrap();

        let got_k = rt.stream.clone_dtoh(&out_k).unwrap();
        let got_v = rt.stream.clone_dtoh(&out_v).unwrap();
        assert_eq!(got_k, got_v);
        // Block 0: offset 0 and offset 1 both written (sequence A's first two tokens).
        assert_eq!(&got_k[0..4], &[10.0, 10.5, 11.0, 11.5]);
        // Block 1: offset 0 written (sequence A's third token), offset 1 untouched.
        assert_eq!(&got_k[4..8], &[12.0, 12.5, 0.0, 0.0]);
        // Block 2: offset 0 written (sequence B's token), offset 1 untouched.
        assert_eq!(&got_k[8..12], &[20.0, 20.5, 0.0, 0.0]);
    }

    // Same case as kv_storage.rs's write_tokens_leaves_other_slots_untouched:
    // a single write into one slot of a cache must not disturb any other slot.
    #[test]
    fn kv_write_leaves_other_slots_untouched() {
        let Some(rt) = runtime() else { return };
        let n_kv_heads = 1u32;
        let head_dim = 2u32;
        let block_size = 4u32;

        let k = vec![1.0f32, 1.5];
        let v = k.clone();
        let block_ids: Vec<u32> = vec![0];
        let offsets: Vec<u32> = vec![2];

        let k_dev = rt.stream.clone_htod(&k).unwrap();
        let v_dev = rt.stream.clone_htod(&v).unwrap();
        let block_ids_dev = rt.stream.clone_htod(&block_ids).unwrap();
        let offsets_dev = rt.stream.clone_htod(&offsets).unwrap();
        let cache_len = (block_size * n_kv_heads * head_dim) as usize;
        let mut out_k = rt.stream.alloc_zeros::<f32>(cache_len).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(cache_len).unwrap();

        kv_write_wrapper(
            &rt,
            &k_dev,
            &v_dev,
            &block_ids_dev,
            &offsets_dev,
            &mut out_k,
            &mut out_v,
            n_kv_heads,
            head_dim,
            block_size,
        )
        .unwrap();

        let got = rt.stream.clone_dtoh(&out_k).unwrap();
        assert_eq!(got, [0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0]);
    }

    #[test]
    fn kv_write_rejects_mismatched_kv_lengths() {
        let Some(rt) = runtime() else { return };
        let k = rt.stream.clone_htod(&[1.0f32, 2.0]).unwrap();
        let v = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let block_ids = rt.stream.clone_htod(&[0u32]).unwrap();
        let offsets = rt.stream.clone_htod(&[0u32]).unwrap();
        let mut out_k = rt.stream.alloc_zeros::<f32>(8).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(8).unwrap();
        assert!(
            kv_write_wrapper(
                &rt, &k, &v, &block_ids, &offsets, &mut out_k, &mut out_v, 1, 2, 4
            )
            .is_err()
        );
    }

    // Same layout as kv_storage.rs's write_tokens_then_gather_round_trips_across_blocks:
    // 3 blocks of 2 slots. Sequence A: 3 tokens over blocks 0,1. Sequence B: 1
    // token in block 2. Writes via kv_write, then reads the whole batch back
    // via kv_gather in one call.
    #[test]
    fn kv_write_then_gather_round_trips_across_blocks() {
        let Some(rt) = runtime() else { return };
        let n_kv_heads = 1u32;
        let head_dim = 2u32;
        let block_size = 2u32;
        let num_blocks = 3u32;

        let k: Vec<f32> = vec![10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 20.0, 20.5];
        let v = k.clone();
        let block_ids: Vec<u32> = vec![0, 0, 1, 2];
        let offsets: Vec<u32> = vec![0, 1, 0, 0];

        let k_dev = rt.stream.clone_htod(&k).unwrap();
        let v_dev = rt.stream.clone_htod(&v).unwrap();
        let block_ids_dev = rt.stream.clone_htod(&block_ids).unwrap();
        let offsets_dev = rt.stream.clone_htod(&offsets).unwrap();
        let cache_len = (num_blocks * block_size * n_kv_heads * head_dim) as usize;
        let mut cache_k = rt.stream.alloc_zeros::<f32>(cache_len).unwrap();
        let mut cache_v = rt.stream.alloc_zeros::<f32>(cache_len).unwrap();
        kv_write_wrapper(
            &rt,
            &k_dev,
            &v_dev,
            &block_ids_dev,
            &offsets_dev,
            &mut cache_k,
            &mut cache_v,
            n_kv_heads,
            head_dim,
            block_size,
        )
        .unwrap();

        // Sequence A's block table [0, 1] (ctx_len 3), sequence B's [2]
        // (ctx_len 1), padded to max_blocks=2 by repeating B's last block.
        let batch = 2u32;
        let max_blocks = 2u32;
        let ctx_len = 3u32;
        let block_idx: Vec<u32> = vec![0, 1, 2, 2];
        let block_idx_dev = rt.stream.clone_htod(&block_idx).unwrap();
        let out_len = (batch * n_kv_heads * ctx_len * head_dim) as usize;
        let mut out_k = rt.stream.alloc_zeros::<f32>(out_len).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(out_len).unwrap();

        kv_gather_wrapper(
            &rt,
            &cache_k,
            &block_idx_dev,
            &mut out_k,
            batch,
            max_blocks,
            ctx_len,
            block_size,
            n_kv_heads,
            head_dim,
        )
        .unwrap();
        kv_gather_wrapper(
            &rt,
            &cache_v,
            &block_idx_dev,
            &mut out_v,
            batch,
            max_blocks,
            ctx_len,
            block_size,
            n_kv_heads,
            head_dim,
        )
        .unwrap();

        let got_k = rt.stream.clone_dtoh(&out_k).unwrap();
        let got_v = rt.stream.clone_dtoh(&out_v).unwrap();
        assert_eq!(got_k, got_v);
        // [batch=2, n_kv_heads=1, ctx_len=3, head_dim=2]
        // Sequence A's three real tokens, in order.
        assert_eq!(&got_k[..6], &[10.0, 10.5, 11.0, 11.5, 12.0, 12.5]);
        // Sequence B's one real token; the rest is masked padding, not checked.
        assert_eq!(&got_k[6..8], &[20.0, 20.5]);
    }

    #[test]
    fn kv_gather_rejects_wrong_block_idx_length() {
        let Some(rt) = runtime() else { return };
        let cache = rt.stream.alloc_zeros::<f32>(8).unwrap();
        let block_idx = rt.stream.clone_htod(&[0u32, 1]).unwrap(); // should be batch*max_blocks=4
        let mut output = rt.stream.alloc_zeros::<f32>(4).unwrap();
        assert!(kv_gather_wrapper(&rt, &cache, &block_idx, &mut output, 2, 2, 2, 2, 1, 1).is_err());
    }

    #[test]
    fn kv_gather_rejects_wrong_output_length() {
        let Some(rt) = runtime() else { return };
        let cache = rt.stream.alloc_zeros::<f32>(8).unwrap();
        let block_idx = rt.stream.clone_htod(&[0u32, 1, 0, 1]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap(); // should be 4
        assert!(kv_gather_wrapper(&rt, &cache, &block_idx, &mut output, 2, 2, 2, 2, 1, 1).is_err());
    }

    #[test]
    fn kv_gather_rejects_ctx_len_exceeding_capacity() {
        let Some(rt) = runtime() else { return };
        let cache = rt.stream.alloc_zeros::<f32>(8).unwrap();
        let block_idx = rt.stream.clone_htod(&[0u32, 1]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(10).unwrap();
        // max_blocks=1, block_size=2 -> capacity 2, ctx_len=5 is too long
        assert!(kv_gather_wrapper(&rt, &cache, &block_idx, &mut output, 2, 1, 5, 2, 1, 1).is_err());
    }

    fn softmax_cpu(row: &[f32]) -> Vec<f32> {
        let row_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|&v| (v - row_max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|&e| e / sum).collect()
    }

    #[test]
    fn softmax_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        // Same case as model.rs's gqa_attention test: scores [1, 0] scaled by
        // 1/sqrt(2) -> softmax([0.7071, 0.0]) = [0.6698, 0.3302].
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let host: Vec<f32> = vec![inv_sqrt2, 0.0];

        let input = rt.stream.clone_htod(&host).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        softmax_wrapper(&rt, &input, host.len() as u32, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let want = [0.6698f32, 0.3302];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-3, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn softmax_masks_with_neg_infinity() {
        let Some(rt) = runtime() else { return };
        // Same masking convention as model.rs's batch_mask: disallowed
        // positions carry a literal f32::NEG_INFINITY additive mask before
        // softmax. Row: two real scores, two masked-out tail positions.
        let neg_inf = f32::NEG_INFINITY;
        let host: Vec<f32> = vec![1.0, 2.0, neg_inf, neg_inf];
        let row_len = host.len() as u32;

        let input = rt.stream.clone_htod(&host).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        softmax_wrapper(&rt, &input, row_len, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let want = softmax_cpu(&host);
        // Masked positions get exactly 0, not just something small.
        assert_eq!(got[2], 0.0);
        assert_eq!(got[3], 0.0);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }
        let sum: f32 = got.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row sum {sum} != 1.0");
    }

    #[test]
    fn softmax_matches_hand_computed_multi_row_tail() {
        let Some(rt) = runtime() else { return };
        // row_len = 37 doesn't divide 256 (the kernel's block size), so this
        // exercises the grid-stride tail in all three passes. 5 rows so the
        // "one block per row" grid dimension is actually exercised.
        let row_len = 37usize;
        let num_rows = 5usize;

        let host: Vec<f32> = (0..row_len * num_rows)
            .map(|i| ((i % row_len) as f32 - (row_len as f32) / 2.0) * 0.1)
            .collect();

        let input = rt.stream.clone_htod(&host).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        softmax_wrapper(&rt, &input, row_len as u32, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        for row in 0..num_rows {
            let start = row * row_len;
            let want = softmax_cpu(&host[start..start + row_len]);
            for i in 0..row_len {
                let g = got[start + i];
                let w = want[i];
                assert!((g - w).abs() < 1e-5, "row={row} i={i}: got {g}, want {w}");
            }
        }
    }

    #[test]
    fn softmax_matches_candle_reference() {
        let Some(rt) = runtime() else { return };
        let row_len = 8usize;
        let num_rows = 3usize;

        let host: Vec<f32> = (0..row_len * num_rows)
            .map(|i| (i as f32 - 12.0) * 0.3)
            .collect();

        let input = rt.stream.clone_htod(&host).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(host.len()).unwrap();
        softmax_wrapper(&rt, &input, row_len as u32, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let cpu = candle_core::Device::Cpu;
        let x_t = candle_core::Tensor::from_slice(&host, (num_rows, row_len), &cpu).unwrap();
        let want: Vec<f32> = candle_nn::ops::softmax(&x_t, candle_core::D::Minus1)
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
    fn softmax_rejects_mismatched_output_length() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(softmax_wrapper(&rt, &input, 2, &mut output).is_err());
    }

    #[test]
    fn softmax_rejects_row_len_not_dividing_input() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(softmax_wrapper(&rt, &input, 2, &mut output).is_err());
    }

    #[test]
    fn add_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        let a: Vec<f32> = (0..1027).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..1027).map(|i| (i as f32 - 500.0) * 0.1).collect();
        let a_dev = rt.stream.clone_htod(&a).unwrap();
        let b_dev = rt.stream.clone_htod(&b).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(a.len()).unwrap();
        add_wrapper(&rt, &a_dev, &b_dev, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();
        for i in 0..a.len() {
            let want = a[i] + b[i];
            assert!(
                (got[i] - want).abs() < 1e-5,
                "i={i}: got {}, want {want}",
                got[i]
            );
        }
    }

    #[test]
    fn add_rejects_mismatched_lengths() {
        let Some(rt) = runtime() else { return };
        let a = rt.stream.clone_htod(&[1.0f32, 2.0]).unwrap();
        let b = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(2).unwrap();
        assert!(add_wrapper(&rt, &a, &b, &mut output).is_err());
    }

    #[test]
    fn embed_lookup_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        let hidden_dim = 3u32;
        // vocab_size=4: row i = [i, i+0.5, i+0.25]
        let embedding: Vec<f32> = (0..4)
            .flat_map(|i| [i as f32, i as f32 + 0.5, i as f32 + 0.25])
            .collect();
        let token_ids: Vec<u32> = vec![2, 0, 3, 2];

        let embedding_dev = rt.stream.clone_htod(&embedding).unwrap();
        let token_ids_dev = rt.stream.clone_htod(&token_ids).unwrap();
        let mut output = rt
            .stream
            .alloc_zeros::<f32>(token_ids.len() * hidden_dim as usize)
            .unwrap();
        embed_lookup_wrapper(&rt, &token_ids_dev, &embedding_dev, hidden_dim, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();

        let want: Vec<f32> = token_ids
            .iter()
            .flat_map(|&t| {
                let t = t as usize;
                embedding[t * 3..t * 3 + 3].to_vec()
            })
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn embed_lookup_rejects_wrong_output_length() {
        let Some(rt) = runtime() else { return };
        let embedding = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let token_ids = rt.stream.clone_htod(&[0u32, 1]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap(); // should be 4
        assert!(embed_lookup_wrapper(&rt, &token_ids, &embedding, 2, &mut output).is_err());
    }

    #[test]
    fn mask_add_broadcast_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        // batch=1, n_kv_heads=1, group_size=2, q_len=1, ctx_len=2.
        let (n_kv_heads, group_size, q_len, ctx_len) = (1u32, 2u32, 1u32, 2u32);
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // group0, group1 each [ctx0, ctx1]
        let mask: Vec<f32> = vec![0.0, f32::NEG_INFINITY]; // same mask for both groups

        let mut scores_dev = rt.stream.clone_htod(&scores).unwrap();
        let mask_dev = rt.stream.clone_htod(&mask).unwrap();
        mask_add_broadcast_wrapper(
            &rt,
            &mut scores_dev,
            &mask_dev,
            n_kv_heads,
            group_size,
            q_len,
            ctx_len,
        )
        .unwrap();
        let got = rt.stream.clone_dtoh(&scores_dev).unwrap();

        assert_eq!(got[0], 1.0);
        assert!(got[1].is_infinite() && got[1].is_sign_negative());
        assert_eq!(got[2], 3.0);
        assert!(got[3].is_infinite() && got[3].is_sign_negative());
    }

    #[test]
    fn mask_add_broadcast_rejects_wrong_mask_length() {
        let Some(rt) = runtime() else { return };
        let mut scores = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mask = rt.stream.clone_htod(&[0.0f32]).unwrap(); // should be 2 (batch=1 * q_len=1 * ctx_len=2)
        assert!(mask_add_broadcast_wrapper(&rt, &mut scores, &mask, 1, 2, 1, 2).is_err());
    }

    #[test]
    fn transpose_axes12_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        // [d0=1, d1=2, d2=3, d3=1] -> [1, 3, 2, 1]: a plain matrix transpose,
        // input row-major 2x3, output should be its 3x2 transpose.
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let input_dev = rt.stream.clone_htod(&input).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(6).unwrap();
        transpose_axes12_wrapper(&rt, &input_dev, &mut output, 1, 2, 3, 1).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();
        // transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]]
        assert_eq!(got, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_axes12_is_its_own_inverse() {
        let Some(rt) = runtime() else { return };
        let (d0, d1, d2, d3) = (2u32, 3u32, 4u32, 5u32);
        let n = (d0 * d1 * d2 * d3) as usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input_dev = rt.stream.clone_htod(&input).unwrap();
        let mut mid = rt.stream.alloc_zeros::<f32>(n).unwrap();
        transpose_axes12_wrapper(&rt, &input_dev, &mut mid, d0, d1, d2, d3).unwrap();
        let mut back = rt.stream.alloc_zeros::<f32>(n).unwrap();
        // mid's dims are [d0, d2, d1, d3]; swapping its middle two axes again
        // (passing d2, d1 this time) undoes the first transpose.
        transpose_axes12_wrapper(&rt, &mid, &mut back, d0, d2, d1, d3).unwrap();
        let got = rt.stream.clone_dtoh(&back).unwrap();
        assert_eq!(got, input);
    }

    #[test]
    fn transpose_axes12_rejects_wrong_length() {
        let Some(rt) = runtime() else { return };
        let input = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(3).unwrap();
        assert!(transpose_axes12_wrapper(&rt, &input, &mut output, 1, 2, 2, 1).is_err());
    }

    #[test]
    fn narrow_last_row_matches_hand_computed() {
        let Some(rt) = runtime() else { return };
        let (q_len, hidden_dim) = (3u32, 2u32);
        // batch=2: item0 rows [1,1][2,2][3,3], item1 rows [4,4][5,5][6,6].
        let x: Vec<f32> = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0];
        let x_dev = rt.stream.clone_htod(&x).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(4).unwrap();
        narrow_last_row_wrapper(&rt, &x_dev, q_len, hidden_dim, &mut output).unwrap();
        let got = rt.stream.clone_dtoh(&output).unwrap();
        assert_eq!(got, [3.0, 3.0, 6.0, 6.0]);
    }

    #[test]
    fn narrow_last_row_rejects_wrong_output_length() {
        let Some(rt) = runtime() else { return };
        let x = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mut output = rt.stream.alloc_zeros::<f32>(1).unwrap(); // should be 2
        assert!(narrow_last_row_wrapper(&rt, &x, 2, 2, &mut output).is_err());
    }

    #[test]
    fn kv_write_rejects_wrong_position_list_length() {
        let Some(rt) = runtime() else { return };
        // 2 tokens' worth of k/v but only 1 position entry.
        let k = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let v = rt.stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let block_ids = rt.stream.clone_htod(&[0u32]).unwrap();
        let offsets = rt.stream.clone_htod(&[0u32]).unwrap();
        let mut out_k = rt.stream.alloc_zeros::<f32>(8).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(8).unwrap();
        assert!(
            kv_write_wrapper(
                &rt, &k, &v, &block_ids, &offsets, &mut out_k, &mut out_v, 1, 2, 4
            )
            .is_err()
        );
    }
}
