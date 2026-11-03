// Thin, validating wrappers around the kernels: check shapes, pick the launch
// geometry, record the dispatch. The Metal twin of cuda/ops.rs + cublas_ops.rs
// (the matmuls and attention GEMMs that cuBLAS provided are kernels here).
use metal::MTLSize;

use super::context::{Arg, MetalBuf, MetalRuntime, Result};

fn size(x: usize, y: usize, z: usize) -> MTLSize {
    MTLSize::new(x as u64, y as u64, z as u64)
}

fn check(cond: bool, msg: &str) -> Result<()> {
    if cond { Ok(()) } else { Err(msg.to_string().into()) }
}

pub fn silu_wrapper(rt: &MetalRuntime, input: &MetalBuf<f32>, output: &mut MetalBuf<f32>) -> Result<()> {
    let n = input.len();
    check(output.len() == n, "Silu: input and output have different lengths")?;
    check(n > 0, "Silu: input len is 0")?;
    rt.dispatch_1d("silu", &[input.arg(), output.arg(), Arg::U32(n as u32)], n)
}

pub fn silu_gate_multiply_wrapper(
    rt: &MetalRuntime,
    gate: &MetalBuf<f32>,
    up: &MetalBuf<f32>,
    output: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = gate.len();
    check(
        output.len() == n && up.len() == n,
        "silu_gate_multiply: input and output have different lengths",
    )?;
    check(n > 0, "silu_gate_multiply: input len is 0")?;
    rt.dispatch_1d(
        "silu_gate_multiply",
        &[gate.arg(), up.arg(), output.arg(), Arg::U32(n as u32)],
        n,
    )
}

pub fn gelu_gate_multiply_wrapper(
    rt: &MetalRuntime,
    gate: &MetalBuf<f32>,
    up: &MetalBuf<f32>,
    output: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = gate.len();
    check(
        output.len() == n && up.len() == n,
        "gelu_gate_multiply: input and output have different lengths",
    )?;
    check(n > 0, "gelu_gate_multiply: input len is 0")?;
    rt.dispatch_1d(
        "gelu_gate_multiply",
        &[gate.arg(), up.arg(), output.arg(), Arg::U32(n as u32)],
        n,
    )
}

// Adds `bias` ([n_cols]) to every row of `x` ([rows, n_cols]), in place.
pub fn add_bias_wrapper(rt: &MetalRuntime, x: &mut MetalBuf<f32>, bias: &MetalBuf<f32>) -> Result<()> {
    let n = x.len();
    let n_cols = bias.len();
    check(
        n > 0 && n_cols > 0 && n % n_cols == 0,
        "add_bias: x length must be a non-zero multiple of the bias length",
    )?;
    rt.dispatch_1d(
        "add_bias",
        &[x.arg(), bias.arg(), Arg::U32(n_cols as u32), Arg::U32(n as u32)],
        n,
    )
}

pub fn scale_inplace_wrapper(rt: &MetalRuntime, x: &mut MetalBuf<f32>, scale: f32) -> Result<()> {
    let n = x.len();
    check(n > 0, "scale_inplace: input len is 0")?;
    rt.dispatch_1d("scale_inplace", &[x.arg(), Arg::F32(scale), Arg::U32(n as u32)], n)
}

pub fn add_wrapper(
    rt: &MetalRuntime,
    a: &MetalBuf<f32>,
    b: &MetalBuf<f32>,
    out: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = a.len();
    check(b.len() == n && out.len() == n, "add: lengths differ")?;
    check(n > 0, "add: input len is 0")?;
    rt.dispatch_1d("add", &[a.arg(), b.arg(), out.arg(), Arg::U32(n as u32)], n)
}

// bx[r, c] = B[r, c] * x[r, c] from the short conv's in_proj output
// ([rows, 3 * hidden], B, C, x in that order).
pub fn short_conv_bx_wrapper(
    rt: &MetalRuntime,
    proj: &MetalBuf<f32>,
    bx: &mut MetalBuf<f32>,
    hidden: u32,
) -> Result<()> {
    let n = bx.len();
    check(
        n > 0 && hidden > 0 && n % hidden as usize == 0,
        "short_conv_bx: bx length must be a non-zero multiple of hidden",
    )?;
    check(proj.len() == 3 * n, "short_conv_bx: proj must be [rows, 3 * hidden]")?;
    rt.dispatch_1d(
        "short_conv_bx",
        &[proj.arg(), bx.arg(), Arg::U32(hidden), Arg::U32(n as u32)],
        n,
    )
}

// out[r, c] = C[r, c] * sum_k filter[c, k] * window[b, t + k, c]; see the kernel.
#[allow(clippy::too_many_arguments)]
pub fn short_conv_filter_gate_wrapper(
    rt: &MetalRuntime,
    window: &MetalBuf<f32>,
    filter: &MetalBuf<f32>,
    proj: &MetalBuf<f32>,
    out: &mut MetalBuf<f32>,
    batch: u32,
    q_len: u32,
    hidden: u32,
    kernel: u32,
) -> Result<()> {
    let n = out.len();
    check(
        n == (batch * q_len * hidden) as usize,
        "short_conv_filter_gate: out must be [batch * q_len, hidden]",
    )?;
    check(
        window.len() == (batch * (kernel - 1 + q_len) * hidden) as usize
            && filter.len() == (hidden * kernel) as usize
            && proj.len() == 3 * n,
        "short_conv_filter_gate: window, filter or proj has the wrong length",
    )?;
    rt.dispatch_1d(
        "short_conv_filter_gate",
        &[
            window.arg(),
            filter.arg(),
            proj.arg(),
            out.arg(),
            Arg::U32(batch),
            Arg::U32(q_len),
            Arg::U32(hidden),
            Arg::U32(kernel),
            Arg::U32(n as u32),
        ],
        n,
    )
}

// One 256-thread threadgroup per row of `hidden_dim`.
pub fn rmsnorm_wrapper(
    rt: &MetalRuntime,
    x: &MetalBuf<f32>,
    weight: &MetalBuf<f32>,
    hidden_dim: u32,
    eps: f32,
    output: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = x.len();
    check(output.len() == n, "RMSNorm: input and output have different lengths")?;
    check(n > 0, "RMSNorm: input len is 0")?;
    check(
        hidden_dim > 0 && n % hidden_dim as usize == 0,
        "RMSNorm: input length is not a multiple of hidden_dim",
    )?;
    check(
        weight.len() == hidden_dim as usize,
        "RMSNorm: weight length must equal hidden_dim",
    )?;
    let rows = n / hidden_dim as usize;
    rt.dispatch_groups(
        "rmsnorm",
        &[x.arg(), weight.arg(), output.arg(), Arg::U32(hidden_dim), Arg::F32(eps)],
        size(rows, 1, 1),
        size(256, 1, 1),
    )
}

// x, out: [n_rows, head_dim]. cos, sin: [n_rows, head_dim / 2], one row per x row.
pub fn rope_wrapper(
    rt: &MetalRuntime,
    x: &MetalBuf<f32>,
    cos: &MetalBuf<f32>,
    sin: &MetalBuf<f32>,
    output: &mut MetalBuf<f32>,
    head_dim: u32,
) -> Result<()> {
    check(output.len() == x.len(), "RoPE: input and output have different lengths")?;
    check(
        head_dim > 0 && head_dim % 2 == 0,
        "RoPE: head_dim must be a positive even number",
    )?;
    let n = x.len();
    check(
        n > 0 && n % head_dim as usize == 0,
        "RoPE: input length is not a multiple of head_dim",
    )?;
    let half = head_dim / 2;
    let n_pairs = n / head_dim as usize * half as usize;
    check(
        cos.len() == n_pairs && sin.len() == n_pairs,
        "RoPE: cos/sin length must equal n_rows * (head_dim / 2)",
    )?;
    rt.dispatch_1d(
        "rope",
        &[
            x.arg(),
            cos.arg(),
            sin.arg(),
            output.arg(),
            Arg::U32(head_dim),
            Arg::U32(half),
            Arg::U32(n_pairs as u32),
        ],
        n_pairs,
    )
}

// k, v: [n_tokens, n_kv_heads, head_dim]; block_ids/offsets: one per token.
// out_k/out_v: [num_blocks, block_size, n_kv_heads, head_dim], mutated in place.
#[allow(clippy::too_many_arguments)]
pub fn kv_write_wrapper(
    rt: &MetalRuntime,
    k: &MetalBuf<f32>,
    v: &MetalBuf<f32>,
    block_ids: &MetalBuf<u32>,
    offsets: &MetalBuf<u32>,
    out_k: &mut MetalBuf<f32>,
    out_v: &mut MetalBuf<f32>,
    n_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> Result<()> {
    check(k.len() == v.len(), "kv_write: k and v have different lengths")?;
    let n = k.len();
    check(n > 0, "kv_write: k/v len is 0")?;
    check(
        n_kv_heads > 0 && head_dim > 0 && block_size > 0,
        "kv_write: n_kv_heads/head_dim/block_size must be positive",
    )?;
    let per_token = (n_kv_heads * head_dim) as usize;
    check(
        n % per_token == 0,
        "kv_write: k/v length is not a multiple of n_kv_heads * head_dim",
    )?;
    let n_tokens = n / per_token;
    check(
        block_ids.len() == n_tokens && offsets.len() == n_tokens,
        "kv_write: block_ids/offsets length must equal n_tokens",
    )?;
    check(out_k.len() == out_v.len(), "kv_write: out_k and out_v have different lengths")?;
    check(
        out_k.len() % (block_size as usize * per_token) == 0,
        "kv_write: out_k/out_v length is not a multiple of block_size * n_kv_heads * head_dim",
    )?;
    rt.dispatch_1d(
        "kv_write",
        &[
            k.arg(),
            v.arg(),
            block_ids.arg(),
            offsets.arg(),
            out_k.arg(),
            out_v.arg(),
            Arg::U32(n_kv_heads),
            Arg::U32(head_dim),
            Arg::U32(block_size),
            Arg::U32(n as u32),
        ],
        n,
    )
}

// cache: one layer's K (or V) storage; block_idx: [batch * max_blocks];
// output: [batch, n_kv_heads, ctx_len, head_dim].
#[allow(clippy::too_many_arguments)]
pub fn kv_gather_wrapper(
    rt: &MetalRuntime,
    cache: &MetalBuf<f32>,
    block_idx: &MetalBuf<u32>,
    output: &mut MetalBuf<f32>,
    batch: u32,
    max_blocks: u32,
    ctx_len: u32,
    block_size: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<()> {
    check(
        batch > 0 && max_blocks > 0 && ctx_len > 0 && block_size > 0 && n_kv_heads > 0 && head_dim > 0,
        "kv_gather: all dimensions must be positive",
    )?;
    let n = (batch * n_kv_heads * ctx_len * head_dim) as usize;
    check(
        output.len() == n,
        "kv_gather: output must be [batch, n_kv_heads, ctx_len, head_dim]",
    )?;
    check(
        block_idx.len() == (batch * max_blocks) as usize,
        "kv_gather: block_idx must be [batch, max_blocks]",
    )?;
    check(
        max_blocks * block_size >= ctx_len,
        "kv_gather: block table too short for ctx_len",
    )?;
    rt.dispatch_1d(
        "kv_gather",
        &[
            cache.arg(),
            block_idx.arg(),
            output.arg(),
            Arg::U32(max_blocks),
            Arg::U32(ctx_len),
            Arg::U32(block_size),
            Arg::U32(n_kv_heads),
            Arg::U32(head_dim),
            Arg::U32(n as u32),
        ],
        n,
    )
}

// scores: [batch, n_kv_heads, group_size, q_len, ctx_len], in place; mask:
// [batch, q_len, ctx_len], broadcast over the heads.
pub fn mask_add_broadcast_wrapper(
    rt: &MetalRuntime,
    scores: &mut MetalBuf<f32>,
    mask: &MetalBuf<f32>,
    n_kv_heads: u32,
    group_size: u32,
    q_len: u32,
    ctx_len: u32,
) -> Result<()> {
    let n = scores.len();
    check(n > 0, "mask_add_broadcast: scores len is 0")?;
    let per_batch = (n_kv_heads * group_size * q_len * ctx_len) as usize;
    check(
        per_batch > 0 && n % per_batch == 0,
        "mask_add_broadcast: scores length is not a multiple of the head/row/ctx product",
    )?;
    let batch = n / per_batch;
    check(
        mask.len() == batch * (q_len * ctx_len) as usize,
        "mask_add_broadcast: mask must be [batch, q_len, ctx_len]",
    )?;
    rt.dispatch_1d(
        "mask_add_broadcast",
        &[
            scores.arg(),
            mask.arg(),
            Arg::U32(n_kv_heads),
            Arg::U32(group_size),
            Arg::U32(q_len),
            Arg::U32(ctx_len),
            Arg::U32(n as u32),
        ],
        n,
    )
}

// Softmax over rows of `row_len`: one 256-thread threadgroup per row.
pub fn softmax_wrapper(
    rt: &MetalRuntime,
    x: &MetalBuf<f32>,
    row_len: u32,
    output: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = x.len();
    check(output.len() == n, "softmax: input and output have different lengths")?;
    check(
        n > 0 && row_len > 0 && n % row_len as usize == 0,
        "softmax: input length is not a multiple of row_len",
    )?;
    rt.dispatch_groups(
        "softmax",
        &[x.arg(), output.arg(), Arg::U32(row_len)],
        size(n / row_len as usize, 1, 1),
        size(256, 1, 1),
    )
}

// ids[row] = index of the row's maximum, for x: [rows, row_len].
pub fn argmax_wrapper(
    rt: &MetalRuntime,
    x: &MetalBuf<f32>,
    row_len: u32,
    ids: &mut MetalBuf<u32>,
) -> Result<()> {
    let n = x.len();
    check(
        n > 0 && row_len > 0 && n % row_len as usize == 0,
        "argmax: input length is not a multiple of row_len",
    )?;
    let rows = n / row_len as usize;
    check(ids.len() == rows, "argmax: ids must hold one entry per row")?;
    rt.dispatch_groups(
        "argmax",
        &[x.arg(), ids.arg(), Arg::U32(row_len)],
        size(rows, 1, 1),
        size(256, 1, 1),
    )
}

pub fn embed_lookup_wrapper(
    rt: &MetalRuntime,
    token_ids: &MetalBuf<u32>,
    embedding: &MetalBuf<f32>,
    hidden_dim: u32,
    out: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = out.len();
    check(
        n == token_ids.len() * hidden_dim as usize,
        "embed_lookup: out must be [n_tokens, hidden_dim]",
    )?;
    check(n > 0, "embed_lookup: no tokens")?;
    rt.dispatch_1d(
        "embed_lookup",
        &[
            token_ids.arg(),
            embedding.arg(),
            out.arg(),
            Arg::U32(hidden_dim),
            Arg::U32(n as u32),
        ],
        n,
    )
}

// x: [batch, q_len, hidden_dim] -> out: [batch, hidden_dim] (each last row).
pub fn narrow_last_row_wrapper(
    rt: &MetalRuntime,
    x: &MetalBuf<f32>,
    q_len: u32,
    hidden_dim: u32,
    out: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = out.len();
    check(
        n > 0 && n % hidden_dim as usize == 0 && x.len() == n * q_len as usize,
        "narrow_last_row: x must be [batch, q_len, hidden_dim] for out [batch, hidden_dim]",
    )?;
    rt.dispatch_1d(
        "narrow_last_row",
        &[x.arg(), out.arg(), Arg::U32(q_len), Arg::U32(hidden_dim), Arg::U32(n as u32)],
        n,
    )
}

// [d0, d1, d2, d3] -> [d0, d2, d1, d3].
pub fn transpose_axes12_wrapper(
    rt: &MetalRuntime,
    input: &MetalBuf<f32>,
    output: &mut MetalBuf<f32>,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
) -> Result<()> {
    let n = input.len();
    check(
        n > 0 && n == (d0 * d1 * d2 * d3) as usize && output.len() == n,
        "transpose_axes12: input/output length must equal d0*d1*d2*d3",
    )?;
    rt.dispatch_1d(
        "transpose_axes12",
        &[
            input.arg(),
            output.arg(),
            Arg::U32(d0),
            Arg::U32(d1),
            Arg::U32(d2),
            Arg::U32(d3),
            Arg::U32(n as u32),
        ],
        n,
    )
}

// Which matmul kernel to run. `Small` streams the weights with a matvec (best
// for few rows); `Gemm` is the tiled simdgroup-matrix kernel (best for many).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearKernel {
    Small,
    // K-split simdgroup-matrix kernel for decode batches of up to 32 rows.
    MmaSmall,
    Gemm,
}

// Rows up to which the matvec kernel is used, then the K-split kernel up to
// MMA_SMALL_ROWS, then the tiled GEMM (thresholds from `linear_microbench`).
const SMALL_ROWS: usize = 4;
const MMA_SMALL_ROWS: usize = 16;

pub fn choose_linear_kernel(rows: usize, in_features: usize, out_features: usize) -> LinearKernel {
    // The matvec kernel loads half4/float4, which needs rows 4-element aligned.
    if rows <= SMALL_ROWS && in_features % 4 == 0 {
        LinearKernel::Small
    } else if rows <= MMA_SMALL_ROWS && in_features % 8 == 0 && out_features % 8 == 0 {
        LinearKernel::MmaSmall
    } else {
        LinearKernel::Gemm
    }
}

// y = x @ weight^T. x: [rows, in_features] F32, weight: [out_features,
// in_features] F16 row-major, y: [rows, out_features] F32.
pub fn linear(
    rt: &MetalRuntime,
    x: &MetalBuf<f32>,
    weight: &MetalBuf<half::f16>,
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: &mut MetalBuf<f32>,
) -> Result<()> {
    let kernel = choose_linear_kernel(rows, in_features, out_features);
    linear_with(rt, kernel, x, weight, rows, in_features, out_features, output)
}

#[allow(clippy::too_many_arguments)]
pub fn linear_with(
    rt: &MetalRuntime,
    kernel: LinearKernel,
    x: &MetalBuf<f32>,
    weight: &MetalBuf<half::f16>,
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: &mut MetalBuf<f32>,
) -> Result<()> {
    check(x.len() == rows * in_features, "linear: x length must equal rows * in_features")?;
    check(
        weight.len() == out_features * in_features,
        "linear: weight length must equal out_features * in_features",
    )?;
    check(
        output.len() == rows * out_features,
        "linear: output length must equal rows * out_features",
    )?;
    check(rows > 0 && out_features > 0, "linear: empty problem")?;
    check(
        kernel != LinearKernel::Small || in_features % 4 == 0,
        "linear: the matvec kernel needs in_features to be a multiple of 4",
    )?;
    check(
        kernel != LinearKernel::MmaSmall
            || (rows <= 32 && in_features % 8 == 0 && out_features % 8 == 0),
        "linear: the K-split kernel needs rows <= 32 and in/out features multiples of 8",
    )?;
    let args = [
        x.arg(),
        weight.arg(),
        output.arg(),
        Arg::U32(rows as u32),
        Arg::U32(in_features as u32),
        Arg::U32(out_features as u32),
    ];
    match kernel {
        // 4 simd groups (output features) per threadgroup, MV_ROWS rows per group of rows.
        LinearKernel::Small => rt.dispatch_groups(
            "linear_small",
            &args,
            size(out_features.div_ceil(4), rows.div_ceil(8), 1),
            size(128, 1, 1),
        ),
        LinearKernel::MmaSmall => rt.dispatch_groups(
            "linear_mma_small",
            &args,
            size(out_features / 8, 1, 1),
            size(128, 1, 1),
        ),
        LinearKernel::Gemm => rt.dispatch_groups(
            "linear_gemm_32x64",
            &args,
            size(out_features.div_ceil(64), rows.div_ceil(32), 1),
            size(128, 1, 1),
        ),
    }
}

// scores[b] = (1/sqrt(head_dim)) * Q[b] @ K[b]^T, batched over batch*n_kv_heads
// with the GQA group folded into the row count (see cuda/cublas_ops.rs).
// q: [batch, n_kv_heads, group_size*q_len, head_dim], k: [batch, n_kv_heads,
// ctx_len, head_dim], scores: [batch, n_kv_heads, group_size*q_len, ctx_len].
#[allow(clippy::too_many_arguments)]
pub fn attention_scores(
    rt: &MetalRuntime,
    q: &MetalBuf<f32>,
    k: &MetalBuf<f32>,
    scores: &mut MetalBuf<f32>,
    batch: usize,
    n_kv_heads: usize,
    group_size: usize,
    q_len: usize,
    ctx_len: usize,
    head_dim: usize,
) -> Result<()> {
    let rows = group_size * q_len;
    let batch_count = batch * n_kv_heads;
    check(q.len() == batch_count * rows * head_dim, "attention_scores: q length mismatch")?;
    check(k.len() == batch_count * ctx_len * head_dim, "attention_scores: k length mismatch")?;
    check(
        scores.len() == batch_count * rows * ctx_len,
        "attention_scores: scores length mismatch",
    )?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    rt.dispatch(
        "attention_scores",
        &[
            q.arg(),
            k.arg(),
            scores.arg(),
            Arg::U32(rows as u32),
            Arg::U32(ctx_len as u32),
            Arg::U32(head_dim as u32),
            Arg::F32(scale),
        ],
        size(ctx_len, rows, batch_count),
        size(32, 1, 1),
    )
}

// attn_out[b] = probs[b] @ V[b]. probs: [batch, n_kv_heads, group_size*q_len,
// ctx_len], v: [batch, n_kv_heads, ctx_len, head_dim], attn_out: [batch,
// n_kv_heads, group_size*q_len, head_dim].
#[allow(clippy::too_many_arguments)]
pub fn attention_apply(
    rt: &MetalRuntime,
    probs: &MetalBuf<f32>,
    v: &MetalBuf<f32>,
    attn_out: &mut MetalBuf<f32>,
    batch: usize,
    n_kv_heads: usize,
    group_size: usize,
    q_len: usize,
    ctx_len: usize,
    head_dim: usize,
) -> Result<()> {
    let rows = group_size * q_len;
    let batch_count = batch * n_kv_heads;
    check(
        probs.len() == batch_count * rows * ctx_len,
        "attention_apply: probs length mismatch",
    )?;
    check(v.len() == batch_count * ctx_len * head_dim, "attention_apply: v length mismatch")?;
    check(
        attn_out.len() == batch_count * rows * head_dim,
        "attention_apply: attn_out length mismatch",
    )?;
    rt.dispatch(
        "attention_apply",
        &[
            probs.arg(),
            v.arg(),
            attn_out.arg(),
            Arg::U32(rows as u32),
            Arg::U32(ctx_len as u32),
            Arg::U32(head_dim as u32),
        ],
        size(head_dim, rows, batch_count),
        size(head_dim.min(64), 1, 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn runtime() -> Option<MetalRuntime> {
        crate::metal::context::test_runtime()
    }

    // Deterministic pseudo-random values in [-1, 1), no rand dependency needed.
    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // y = x @ w^T in F64 over the F16-rounded weights the kernels see.
    fn linear_ref(x: &[f32], w: &[f16], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * out_f];
        for r in 0..rows {
            for n in 0..out_f {
                let mut acc = 0f64;
                for k in 0..in_f {
                    acc += x[r * in_f + k] as f64 * w[n * in_f + k].to_f32() as f64;
                }
                y[r * out_f + n] = acc as f32;
            }
        }
        y
    }

    // Both matmul kernels, on shapes that straddle every tile/stride boundary:
    // rows around the matvec/GEMM switch and the 32-row tile, in_features that
    // are and are not multiples of 4 and of the 32-wide k tile, out_features
    // that leave partial threadgroups.
    #[test]
    fn linear_matches_reference_across_shapes() {
        let Some(rt) = runtime() else { return };
        for &rows in &[1usize, 3, 4, 5, 17, 32, 33, 70] {
            for &in_f in &[4usize, 7, 32, 36, 100, 960, 64] {
                for &out_f in &[1usize, 5, 32, 33, 130, 64, 320] {
                    let x = values(rows * in_f, 1);
                    let w: Vec<f16> = values(out_f * in_f, 2).into_iter().map(f16::from_f32).collect();
                    let want = linear_ref(&x, &w, rows, in_f, out_f);
                    let xb = MetalBuf::from_slice(&rt, &x);
                    let wb = MetalBuf::from_slice(&rt, &w);
                    // Every kernel on every shape, not only the one `linear` would pick.
                    for kernel in [LinearKernel::Small, LinearKernel::MmaSmall, LinearKernel::Gemm] {
                        if kernel == LinearKernel::Small && in_f % 4 != 0 {
                            continue;
                        }
                        if kernel == LinearKernel::MmaSmall
                            && (rows > 32 || in_f % 8 != 0 || out_f % 8 != 0)
                        {
                            continue;
                        }
                        let mut yb = MetalBuf::<f32>::zeros(&rt, rows * out_f);
                        linear_with(&rt, kernel, &xb, &wb, rows, in_f, out_f, &mut yb).unwrap();
                        let got = rt.read(&yb).unwrap();
                        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                                "{kernel:?} rows={rows} in={in_f} out={out_f} i={i}: got {g}, want {w}"
                            );
                        }
                    }
                    let mut yb = MetalBuf::<f32>::zeros(&rt, rows * out_f);
                    linear(&rt, &xb, &wb, rows, in_f, out_f, &mut yb).unwrap();
                    assert_eq!(rt.read(&yb).unwrap().len(), rows * out_f);
                }
            }
        }
    }

    #[test]
    fn rmsnorm_matches_reference_for_wide_rows() {
        let Some(rt) = runtime() else { return };
        for &(rows, dim) in &[(1usize, 960usize), (3, 64), (2, 3000), (5, 33)] {
            let x = values(rows * dim, 3);
            let weight = values(dim, 4);
            let xb = MetalBuf::from_slice(&rt, &x);
            let wb = MetalBuf::from_slice(&rt, &weight);
            let mut out = MetalBuf::<f32>::zeros(&rt, rows * dim);
            rmsnorm_wrapper(&rt, &xb, &wb, dim as u32, 1e-5, &mut out).unwrap();
            let got = rt.read(&out).unwrap();
            for r in 0..rows {
                let row = &x[r * dim..(r + 1) * dim];
                let mean_sq = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / dim as f64;
                let rms = (mean_sq + 1e-5).sqrt();
                for i in 0..dim {
                    let want = (row[i] as f64 / rms) * weight[i] as f64;
                    assert!((got[r * dim + i] as f64 - want).abs() < 1e-5, "r={r} i={i}");
                }
            }
        }
    }

    #[test]
    fn softmax_handles_masked_rows_and_long_rows() {
        let Some(rt) = runtime() else { return };
        // A row with -inf entries (the attention mask), and rows longer than
        // the 256-thread threadgroup.
        for &len in &[5usize, 200, 1000] {
            let mut x = values(2 * len, 5);
            for v in x[len / 2..len].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
            let xb = MetalBuf::from_slice(&rt, &x);
            let mut out = MetalBuf::<f32>::zeros(&rt, 2 * len);
            softmax_wrapper(&rt, &xb, len as u32, &mut out).unwrap();
            let got = rt.read(&out).unwrap();
            for r in 0..2 {
                let row = &x[r * len..(r + 1) * len];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f64 = row.iter().map(|v| ((v - max) as f64).exp()).sum();
                for i in 0..len {
                    let want = ((row[i] - max) as f64).exp() / sum;
                    assert!((got[r * len + i] as f64 - want).abs() < 1e-6, "len={len} r={r} i={i}");
                }
            }
        }
    }

    #[test]
    fn argmax_matches_sampler_tie_and_nan_rules() {
        let Some(rt) = runtime() else { return };
        let len = 1000;
        let mut x = values(3 * len, 6);
        x[len + 17] = 5.0; // row 1: tie between 17 and 900, lowest index wins
        x[len + 900] = 5.0;
        x[2 * len + 3] = f32::NAN; // row 2: NaN is ignored
        x[2 * len + 40] = 7.0;
        x[7] = 9.0; // row 0: a plain maximum
        let xb = MetalBuf::from_slice(&rt, &x);
        let mut ids = MetalBuf::<u32>::zeros(&rt, 3);
        argmax_wrapper(&rt, &xb, len as u32, &mut ids).unwrap();
        assert_eq!(rt.read(&ids).unwrap(), vec![7, 17, 40]);

        // Nothing above -inf: returns 0 like the CPU scan.
        let x = vec![f32::NEG_INFINITY; 64];
        let xb = MetalBuf::from_slice(&rt, &x);
        let mut ids = MetalBuf::<u32>::zeros(&rt, 1);
        argmax_wrapper(&rt, &xb, 64, &mut ids).unwrap();
        assert_eq!(rt.read(&ids).unwrap(), vec![0]);
    }

    // Attention scores + apply against a direct CPU computation, with a GQA
    // group (2 query rows per KV head) and a head_dim that is not a multiple of 4.
    #[test]
    fn attention_kernels_match_reference() {
        let Some(rt) = runtime() else { return };
        for &head_dim in &[6usize, 64] {
            let (batch, n_kv, group, q_len, ctx) = (2usize, 2usize, 2usize, 3usize, 9usize);
            let rows = group * q_len;
            let bk = batch * n_kv;
            let q = values(bk * rows * head_dim, 7);
            let k = values(bk * ctx * head_dim, 8);
            let v = values(bk * ctx * head_dim, 9);
            let (qb, kb, vb) = (
                MetalBuf::from_slice(&rt, &q),
                MetalBuf::from_slice(&rt, &k),
                MetalBuf::from_slice(&rt, &v),
            );
            let mut scores = MetalBuf::<f32>::zeros(&rt, bk * rows * ctx);
            attention_scores(&rt, &qb, &kb, &mut scores, batch, n_kv, group, q_len, ctx, head_dim).unwrap();
            let mut out = MetalBuf::<f32>::zeros(&rt, bk * rows * head_dim);
            // Feed the raw scores through apply as if they were probabilities.
            attention_apply(&rt, &scores, &vb, &mut out, batch, n_kv, group, q_len, ctx, head_dim).unwrap();
            let (got_s, got_o) = (rt.read(&scores).unwrap(), rt.read(&out).unwrap());
            let scale = 1.0 / (head_dim as f32).sqrt();
            for b in 0..bk {
                for r in 0..rows {
                    let mut want_s = vec![0f32; ctx];
                    for c in 0..ctx {
                        let dot: f32 = (0..head_dim)
                            .map(|d| q[(b * rows + r) * head_dim + d] * k[(b * ctx + c) * head_dim + d])
                            .sum();
                        want_s[c] = dot * scale;
                        let g = got_s[(b * rows + r) * ctx + c];
                        assert!((g - want_s[c]).abs() < 1e-4, "scores b={b} r={r} c={c}");
                    }
                    for d in 0..head_dim {
                        let want: f32 = (0..ctx).map(|c| want_s[c] * v[(b * ctx + c) * head_dim + d]).sum();
                        let g = got_o[(b * rows + r) * head_dim + d];
                        assert!((g - want).abs() < 1e-3, "apply b={b} r={r} d={d}: {g} vs {want}");
                    }
                }
            }
        }
    }

    // Not a correctness test: `cargo test --release --features metal --lib
    // linear_microbench -- --ignored --nocapture` times each matmul kernel on
    // the model's real projection shapes. Effective GB/s counts the F16 weight
    // bytes once per call, which is the floor a decode step has to pay.
    #[test]
    #[ignore]
    fn linear_microbench() {
        use std::time::Instant;
        let Some(rt) = runtime() else { return };
        let shapes = [("kv 960->320", 960usize, 320usize), ("qkv/o 960x960", 960, 960), ("ffn up 960->2560", 960, 2560), ("ffn down 2560->960", 2560, 960), ("lm_head 960->49152", 960, 49152)];
        println!("{:<22} {:>5} {:>7} {:>9} {:>9} {:>9}", "shape", "rows", "kernel", "us/call", "GB/s", "GFLOP/s");
        for (name, in_f, out_f) in shapes {
            let w: Vec<f16> = values(in_f * out_f, 2).into_iter().map(f16::from_f32).collect();
            let wb = MetalBuf::from_slice(&rt, &w);
            for &rows in &[1usize, 4, 8, 16, 32, 128] {
                let xb = MetalBuf::from_slice(&rt, &values(rows * in_f, 1));
                let mut yb = MetalBuf::<f32>::zeros(&rt, rows * out_f);
                for kernel in [LinearKernel::Small, LinearKernel::MmaSmall, LinearKernel::Gemm] {
                    if kernel == LinearKernel::MmaSmall && rows > 32 {
                        continue;
                    }
                    let iters = 200;
                    for _ in 0..5 {
                        linear_with(&rt, kernel, &xb, &wb, rows, in_f, out_f, &mut yb).unwrap();
                    }
                    rt.synchronize().unwrap();
                    let start = Instant::now();
                    for _ in 0..iters {
                        linear_with(&rt, kernel, &xb, &wb, rows, in_f, out_f, &mut yb).unwrap();
                    }
                    rt.synchronize().unwrap();
                    let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
                    let gbs = (in_f * out_f * 2) as f64 / us / 1e3;
                    let gflops = 2.0 * (rows * in_f * out_f) as f64 / us / 1e3;
                    println!("{name:<22} {rows:>5} {:>7} {us:>9.1} {gbs:>9.1} {gflops:>9.1}", format!("{kernel:?}"));
                }
            }
        }
    }
}
