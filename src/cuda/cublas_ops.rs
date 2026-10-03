// cuBLAS-backed matmuls for the CUDA model path: `linear` (the model's
// weight projections) and the two attention GEMMs (QK^T, probs@V), batched
// over (batch * n_kv_heads) with the GQA group folded into the row count -
// mirrors model.rs's `linear`/`gqa_attention` matmul shapes exactly, just
// expressed as cuBLAS calls instead of Candle tensor ops.
//
// cuBLAS is column-major; every buffer here is row-major. A row-major
// buffer of logical shape (p, q) is byte-identical to a column-major buffer
// of shape (q, p) - its own transpose, for free, no data movement. So every
// GEMM below is set up by writing the *desired row-major result* as a
// matrix product, taking its transpose (free, per the rule above) to get
// what cuBLAS actually needs to compute, and reading off m/n/k/lda/ldb/ldc
// and transpose flags from that - never by reasoning in column-major
// directly.
use std::ffi::c_void;

use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType,
};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig, result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::context::CudaRuntime;
use super::ops;

// Narrows x to F16 into `scratch` (which may be longer than x and is meant to
// be allocated once per forward pass and reused, so no per-call allocation).
// Call once per distinct input, then `linear_f16` for each weight that reads it.
pub fn to_f16(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    scratch: &mut CudaSlice<f16>,
) -> Result<(), Box<dyn std::error::Error>> {
    ops::f32_to_f16_wrapper(cuda_runtime, x, scratch)
}

// y = x @ weight^T. x: [rows, in_features] row-major, already narrowed to F16
// by `to_f16` (only the first rows * in_features elements are read). weight:
// [out_features, in_features] row-major, F16 (Candle Linear convention - same
// weight layout model.rs's `linear` takes). y: [rows, out_features]
// row-major, F32.
// cuBLAS has no F16-weight x F32-activation GEMM, so the activations are
// narrowed first (tiny next to the weight read) and the GEMM accumulates in
// F32 with an F32 output, so downstream kernels still see F32.
pub fn linear_f16(
    cuda_runtime: &CudaRuntime,
    x16: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: &mut CudaSlice<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if x16.len() < rows * in_features {
        return Err("linear: x16 shorter than rows * in_features".into());
    }
    if weight.len() != out_features * in_features {
        return Err("linear: weight length must equal out_features * in_features".into());
    }
    if output.len() != rows * out_features {
        return Err("linear: output length must equal rows * out_features".into());
    }
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let stream = &cuda_runtime.stream;
    let (w_ptr, _w_guard) = weight.device_ptr(stream);
    let (x_ptr, _x_guard) = x16.device_ptr(stream);
    let (o_ptr, _o_guard) = output.device_ptr_mut(stream);
    unsafe {
        result::gemm_ex(
            *cuda_runtime.blas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            out_features as i32,
            rows as i32,
            in_features as i32,
            (&alpha as *const f32).cast(),
            w_ptr as *const c_void,
            cudaDataType::CUDA_R_16F,
            in_features as i32,
            x_ptr as *const c_void,
            cudaDataType::CUDA_R_16F,
            in_features as i32,
            (&beta as *const f32).cast(),
            o_ptr as *mut c_void,
            cudaDataType::CUDA_R_32F,
            out_features as i32,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }?;
    Ok(())
}

// Convenience for a one-off projection: `to_f16` then `linear_f16`.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f16>,
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: &mut CudaSlice<f32>,
    scratch: &mut CudaSlice<f16>,
) -> Result<(), Box<dyn std::error::Error>> {
    if x.len() != rows * in_features {
        return Err("linear: x length must equal rows * in_features".into());
    }
    to_f16(cuda_runtime, x, scratch)?;
    linear_f16(
        cuda_runtime,
        scratch,
        weight,
        rows,
        in_features,
        out_features,
        output,
    )
}

// scores[b] = (1/sqrt(head_dim)) * Q[b] @ K[b]^T, batched over
// batch*n_kv_heads (the attention scale is folded into cuBLAS's alpha
// rather than a separate kernel pass).
// q: [batch, n_kv_heads, group_size*q_len, head_dim] - a free reinterpretation
// of a buffer physically laid out as [batch, n_heads, q_len, head_dim],
// valid because n_heads splits into (kv_head, group) contiguously with
// group immediately outer of q_len (same assumption model.rs's
// `gqa_attention` reshape relies on).
// k: [batch, n_kv_heads, ctx_len, head_dim].
// scores: [batch, n_kv_heads, group_size*q_len, ctx_len].
#[allow(clippy::too_many_arguments)]
pub fn attention_scores(
    cuda_runtime: &CudaRuntime,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    scores: &mut CudaSlice<f32>,
    batch: usize,
    n_kv_heads: usize,
    group_size: usize,
    q_len: usize,
    ctx_len: usize,
    head_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = group_size * q_len;
    let batch_count = batch * n_kv_heads;
    if q.len() != batch_count * rows * head_dim {
        return Err("attention_scores: q length mismatch".into());
    }
    if k.len() != batch_count * ctx_len * head_dim {
        return Err("attention_scores: k length mismatch".into());
    }
    if scores.len() != batch_count * rows * ctx_len {
        return Err("attention_scores: scores length mismatch".into());
    }
    let alpha = 1.0f32 / (head_dim as f32).sqrt();
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: ctx_len as i32,
            n: rows as i32,
            k: head_dim as i32,
            alpha,
            lda: head_dim as i32,
            ldb: head_dim as i32,
            beta: 0.0f32,
            ldc: ctx_len as i32,
        },
        batch_size: batch_count as i32,
        stride_a: (ctx_len * head_dim) as i64,
        stride_b: (rows * head_dim) as i64,
        stride_c: (rows * ctx_len) as i64,
    };
    unsafe { cuda_runtime.blas.gemm_strided_batched(cfg, k, q, scores) }?;
    Ok(())
}

// attn_out[b] = probs[b] @ V[b], batched over batch*n_kv_heads.
// probs: [batch, n_kv_heads, group_size*q_len, ctx_len] (post-softmax).
// v: [batch, n_kv_heads, ctx_len, head_dim].
// attn_out: [batch, n_kv_heads, group_size*q_len, head_dim] - a free
// reinterpretation back into [batch, n_heads, q_len, head_dim], the inverse
// of the split used for `attention_scores`'s `q`.
#[allow(clippy::too_many_arguments)]
pub fn attention_apply(
    cuda_runtime: &CudaRuntime,
    probs: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    n_kv_heads: usize,
    group_size: usize,
    q_len: usize,
    ctx_len: usize,
    head_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = group_size * q_len;
    let batch_count = batch * n_kv_heads;
    if probs.len() != batch_count * rows * ctx_len {
        return Err("attention_apply: probs length mismatch".into());
    }
    if v.len() != batch_count * ctx_len * head_dim {
        return Err("attention_apply: v length mismatch".into());
    }
    if attn_out.len() != batch_count * rows * head_dim {
        return Err("attention_apply: attn_out length mismatch".into());
    }
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: head_dim as i32,
            n: rows as i32,
            k: ctx_len as i32,
            alpha: 1.0f32,
            lda: head_dim as i32,
            ldb: ctx_len as i32,
            beta: 0.0f32,
            ldc: head_dim as i32,
        },
        batch_size: batch_count as i32,
        stride_a: (ctx_len * head_dim) as i64,
        stride_b: (rows * ctx_len) as i64,
        stride_c: (rows * head_dim) as i64,
    };
    unsafe {
        cuda_runtime
            .blas
            .gemm_strided_batched(cfg, v, probs, attn_out)
    }?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> Option<CudaRuntime> {
        CudaRuntime::new(0).ok()
    }

    #[test]
    fn linear_matches_candle_reference() {
        let Some(rt) = runtime() else { return };
        // Same case as model.rs's linear_matches_manual_matmul_for_2d_and_3d_input.
        let w: Vec<f16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect(); // [out=2, in=3]
        let x: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // [rows=2, in=3]

        let w_dev = rt.stream.clone_htod(&w).unwrap();
        let x_dev = rt.stream.clone_htod(&x).unwrap();
        let mut out = rt.stream.alloc_zeros::<f32>(4).unwrap();
        let mut scratch = rt.stream.alloc_zeros::<f16>(8).unwrap();
        linear(&rt, &x_dev, &w_dev, 2, 3, 2, &mut out, &mut scratch).unwrap();
        let got = rt.stream.clone_dtoh(&out).unwrap();

        // row0 = [1+3, 4+6] = [4, 10]; row1 = [2, 5]
        assert_eq!(got, [4.0, 10.0, 2.0, 5.0]);
    }

    #[test]
    fn attention_matches_gqa_hand_computed() {
        let Some(rt) = runtime() else { return };
        // Exact same case as model.rs's gqa_attention_hand_computed:
        // n_heads=2, n_kv_heads=1 (group_size=2), q_len=2, head_dim=2, batch=1.
        let (batch, n_kv_heads, group_size, q_len, ctx_len, head_dim) = (1, 1, 2, 2, 2, 2);
        let rows = group_size * q_len;

        // K: position0=[1,0], position1=[0,1].
        let k: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        // V: position0=[10,0], position1=[0,20].
        let v: Vec<f32> = vec![10.0, 0.0, 0.0, 20.0];
        // Q, folded as [group*q_len, head_dim]: head0 rows then head1 rows.
        let q: Vec<f32> = vec![
            1.0, 0.0, 1.0, 0.0, // head0: pos0=[1,0], pos1=[1,0]
            0.0, 1.0, 0.0, 1.0, // head1: pos0=[0,1], pos1=[0,1]
        ];
        // Causal mask: pos0 attends only to ctx0, pos1 attends to both.
        let mask: Vec<f32> = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];

        let k_dev = rt.stream.clone_htod(&k).unwrap();
        let v_dev = rt.stream.clone_htod(&v).unwrap();
        let q_dev = rt.stream.clone_htod(&q).unwrap();
        let mask_dev = rt.stream.clone_htod(&mask).unwrap();

        let mut scores = rt
            .stream
            .alloc_zeros::<f32>(batch * n_kv_heads * rows * ctx_len)
            .unwrap();
        attention_scores(
            &rt,
            &q_dev,
            &k_dev,
            &mut scores,
            batch,
            n_kv_heads,
            group_size,
            q_len,
            ctx_len,
            head_dim,
        )
        .unwrap();

        super::super::ops::mask_add_broadcast_wrapper(
            &rt,
            &mut scores,
            &mask_dev,
            n_kv_heads as u32,
            group_size as u32,
            q_len as u32,
            ctx_len as u32,
        )
        .unwrap();

        let mut probs = rt
            .stream
            .alloc_zeros::<f32>(batch * n_kv_heads * rows * ctx_len)
            .unwrap();
        super::super::ops::softmax_wrapper(&rt, &scores, ctx_len as u32, &mut probs).unwrap();

        let mut attn_out = rt
            .stream
            .alloc_zeros::<f32>(batch * n_kv_heads * rows * head_dim)
            .unwrap();
        attention_apply(
            &rt,
            &probs,
            &v_dev,
            &mut attn_out,
            batch,
            n_kv_heads,
            group_size,
            q_len,
            ctx_len,
            head_dim,
        )
        .unwrap();

        let got = rt.stream.clone_dtoh(&attn_out).unwrap();
        let expected = [
            10.0, 0.0, // head0 pos0
            6.698, 6.604, // head0 pos1
            10.0, 0.0, // head1 pos0
            3.302, 13.396, // head1 pos1
        ];
        for (g, w) in got.iter().zip(expected.iter()) {
            assert!((g - w).abs() < 1e-2, "got {got:?}, want {expected:?}");
        }
    }
}
