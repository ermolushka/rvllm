// Thin, validating wrappers around the kernels: check shapes, pick the launch
// geometry, record the dispatch. The Metal twin of cuda/ops.rs + cublas_ops.rs
// (the matmuls and attention GEMMs that cuBLAS provided are kernels here).
use metal::MTLSize;

use super::context::{Arg, MetalBuf, MetalRuntime, Result, ATTN_DECODE_KERNELS};

fn size(x: usize, y: usize, z: usize) -> MTLSize {
    MTLSize::new(x as u64, y as u64, z as u64)
}

fn check(cond: bool, msg: &str) -> Result<()> {
    if cond { Ok(()) } else { Err(msg.to_string().into()) }
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

// sum_out = a + b and norm_out = rmsnorm(sum_out) * weight, one dispatch.
pub fn add_rmsnorm_wrapper(
    rt: &MetalRuntime,
    a: &MetalBuf<f32>,
    b: &MetalBuf<f32>,
    weight: &MetalBuf<f32>,
    hidden_dim: u32,
    eps: f32,
    sum_out: &mut MetalBuf<f32>,
    norm_out: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = a.len();
    check(
        b.len() == n && sum_out.len() == n && norm_out.len() == n,
        "add_rmsnorm: lengths differ",
    )?;
    check(
        n > 0 && hidden_dim > 0 && n % hidden_dim as usize == 0,
        "add_rmsnorm: length is not a non-zero multiple of hidden_dim",
    )?;
    check(
        weight.len() == hidden_dim as usize,
        "add_rmsnorm: weight length must equal hidden_dim",
    )?;
    rt.dispatch_groups(
        "add_rmsnorm",
        &[
            a.arg(),
            b.arg(),
            weight.arg(),
            sum_out.arg(),
            norm_out.arg(),
            Arg::U32(hidden_dim),
            Arg::F32(eps),
        ],
        size(n / hidden_dim as usize, 1, 1),
        size(256, 1, 1),
    )
}

// out = act(gate) * up from the stacked gate/up matmul output [rows, 2 * ffn_dim].
pub fn glu_wrapper(
    rt: &MetalRuntime,
    gelu: bool,
    gate_up: &MetalBuf<f32>,
    ffn_dim: usize,
    out: &mut MetalBuf<f32>,
) -> Result<()> {
    let n = out.len();
    check(n > 0 && ffn_dim > 0 && n % ffn_dim == 0, "glu: out must be [rows, ffn_dim]")?;
    check(gate_up.len() == 2 * n, "glu: gate_up must be [rows, 2 * ffn_dim]")?;
    rt.dispatch_1d(
        if gelu { "gelu_glu" } else { "silu_glu" },
        &[gate_up.arg(), out.arg(), Arg::U32(ffn_dim as u32), Arg::U32(n as u32)],
        n,
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

// Everything between the QKV projection and attention: bias, per-head Q/K
// norm, RoPE, Q into `q_out` and K/V into the paged cache (see the kernel).
// qkv: [rows, (n_heads + 2 * n_kv_heads) * head_dim]; cos/sin: [rows, head_dim / 2].
#[allow(clippy::too_many_arguments)]
pub fn attn_prep(
    rt: &MetalRuntime,
    qkv: &MetalBuf<f32>,
    bias: Option<&MetalBuf<f32>>,
    q_norm: Option<&MetalBuf<f32>>,
    k_norm: Option<&MetalBuf<f32>>,
    cos: &MetalBuf<f32>,
    sin: &MetalBuf<f32>,
    block_ids: &MetalBuf<u32>,
    offsets: &MetalBuf<u32>,
    cache_k: &MetalBuf<f32>,
    cache_v: &MetalBuf<f32>,
    q_out: &mut MetalBuf<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: u32,
    eps: f32,
) -> Result<()> {
    check(head_dim > 0 && head_dim % 2 == 0, "attn_prep: head_dim must be a positive even number")?;
    let qkv_dim = (n_heads + 2 * n_kv_heads) * head_dim;
    check(qkv.len() > 0 && qkv.len() % qkv_dim == 0, "attn_prep: qkv length is not a multiple of the QKV width")?;
    let rows = qkv.len() / qkv_dim;
    check(q_out.len() == rows * n_heads * head_dim, "attn_prep: q_out must be [rows, n_heads * head_dim]")?;
    check(
        cos.len() == rows * head_dim / 2 && sin.len() == cos.len(),
        "attn_prep: cos/sin must be [rows, head_dim / 2]",
    )?;
    check(
        block_ids.len() == rows && offsets.len() == rows,
        "attn_prep: block_ids/offsets must hold one entry per row",
    )?;
    check(bias.is_none_or(|b| b.len() == qkv_dim), "attn_prep: bias must be [QKV width]")?;
    check(
        q_norm.is_none_or(|w| w.len() == head_dim) && k_norm.is_none_or(|w| w.len() == head_dim),
        "attn_prep: Q/K norm weights must be [head_dim]",
    )?;
    check(cache_k.len() == cache_v.len(), "attn_prep: K and V caches differ in length")?;
    let flags = u32::from(bias.is_some()) | u32::from(q_norm.is_some()) << 1 | u32::from(k_norm.is_some()) << 2;
    // Unused optional buffers still need something bound.
    rt.dispatch_groups(
        "attn_prep",
        &[
            qkv.arg(),
            bias.unwrap_or(qkv).arg(),
            q_norm.unwrap_or(qkv).arg(),
            k_norm.unwrap_or(qkv).arg(),
            cos.arg(),
            sin.arg(),
            block_ids.arg(),
            offsets.arg(),
            cache_k.arg(),
            cache_v.arg(),
            q_out.arg(),
            Arg::U32(n_heads as u32),
            Arg::U32(n_kv_heads as u32),
            Arg::U32(head_dim as u32),
            Arg::U32(block_size),
            Arg::U32(flags),
            Arg::F32(eps),
        ],
        size(n_heads + 2 * n_kv_heads, rows, 1),
        size(32, 1, 1),
    )
}

// Simd groups per attention threadgroup; must equal ATT_SG in attention.metal.
const ATT_SIMDGROUPS: usize = 8;

// The attention variant for a model: (name, query heads per threadgroup).
// Prefers handling the whole GQA group in one threadgroup (K/V loaded once);
// falls back to one head per threadgroup when no variant exists for the group.
fn attn_decode_variant(group: usize, head_dim: usize) -> Result<(&'static str, usize)> {
    let dpl = head_dim.div_ceil(32);
    let width = [2usize, 4, 8]
        .into_iter()
        .find(|&w| dpl <= w)
        .ok_or("attn_decode: head_dim must be at most 256")?;
    let find = |g: usize| ATTN_DECODE_KERNELS.iter().find(|&&(vg, vd, _)| vg == g && vd == width);
    let &(g, _, name) = find(group)
        .or_else(|| find(1))
        .ok_or("attn_decode: no kernel variant for this head layout")?;
    Ok((name, g))
}

// Attention over the paged cache, output in the layout the O projection reads.
// q, out: [batch * q_len, n_heads * head_dim]; mask: [batch, q_len, ctx_len];
// block_idx: [batch, max_blocks].
#[allow(clippy::too_many_arguments)]
pub fn attn_decode(
    rt: &MetalRuntime,
    q: &MetalBuf<f32>,
    cache_k: &MetalBuf<f32>,
    cache_v: &MetalBuf<f32>,
    block_idx: &MetalBuf<u32>,
    mask: &MetalBuf<f32>,
    out: &mut MetalBuf<f32>,
    batch: usize,
    q_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: u32,
    max_blocks: u32,
    ctx_len: usize,
) -> Result<()> {
    check(
        head_dim > 0 && head_dim <= 256,
        "attn_decode: head_dim must be in 1..=256",
    )?;
    check(
        n_kv_heads > 0 && n_heads % n_kv_heads == 0,
        "attn_decode: n_heads must be a multiple of n_kv_heads",
    )?;
    let rows = batch * q_len;
    check(q.len() == rows * n_heads * head_dim, "attn_decode: q length mismatch")?;
    check(out.len() == q.len(), "attn_decode: out length mismatch")?;
    check(mask.len() == rows * ctx_len, "attn_decode: mask must be [batch, q_len, ctx_len]")?;
    check(
        block_idx.len() == batch * max_blocks as usize && max_blocks as usize * block_size as usize >= ctx_len,
        "attn_decode: block table does not cover ctx_len",
    )?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (kernel, heads_per_group) = attn_decode_variant(n_heads / n_kv_heads, head_dim)?;
    rt.dispatch_groups(
        kernel,
        &[
            q.arg(),
            cache_k.arg(),
            cache_v.arg(),
            block_idx.arg(),
            mask.arg(),
            out.arg(),
            Arg::U32(n_heads as u32),
            Arg::U32(n_kv_heads as u32),
            Arg::U32(head_dim as u32),
            Arg::U32(block_size),
            Arg::U32(max_blocks),
            Arg::U32(ctx_len as u32),
            Arg::U32(q_len as u32),
            Arg::F32(scale),
        ],
        size(n_heads / heads_per_group, q_len, batch),
        size(ATT_SIMDGROUPS * 32, 1, 1),
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

    // Fused attention against a direct CPU computation over a shuffled block
    // table: every GQA group size the kernel specializes (3, 4, 2) plus ones it
    // does not (5, 9: per-head fallback), head widths in each register tier,
    // contexts shorter and longer than one 32-key chunk, ragged sequence
    // lengths (masked tails) and multi-token queries (causal mask).
    #[test]
    fn attn_decode_matches_reference() {
        let Some(rt) = runtime() else { return };
        // (n_kv_heads, group, head_dim, batch, q_len, ctx_len)
        let cases = [
            (2usize, 3usize, 64usize, 2usize, 1usize, 100usize),
            (2, 5, 64, 2, 1, 40),
            (1, 9, 32, 1, 2, 33),
            (2, 2, 6, 3, 2, 17),
            (1, 4, 128, 2, 3, 70),
            (2, 2, 200, 1, 1, 300),
        ];
        let block_size = 16usize;
        for (case, &(n_kv, group, hd, batch, q_len, ctx)) in cases.iter().enumerate() {
            let n_heads = n_kv * group;
            let rows = batch * q_len;
            let max_blocks = ctx.div_ceil(block_size);
            let num_blocks = batch * max_blocks;
            // Sequences own blocks in reverse order, so addressing is never the identity.
            let mut block_idx = Vec::new();
            for b in 0..batch {
                for i in 0..max_blocks {
                    block_idx.push((num_blocks - 1 - (b * max_blocks + i)) as u32);
                }
            }
            let mut sorted = block_idx.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), num_blocks, "test block table must be a permutation");
            let q = values(rows * n_heads * hd, 10 + case as u32);
            let cache_k = values(num_blocks * block_size * n_kv * hd, 20 + case as u32);
            let cache_v = values(num_blocks * block_size * n_kv * hd, 30 + case as u32);
            // Sequence b has ctx - 2b real tokens; its last q_len are the queries.
            let mut mask = vec![0f32; rows * ctx];
            for b in 0..batch {
                let valid = ctx - 2 * b;
                for qi in 0..q_len {
                    for j in 0..ctx {
                        if j >= valid || j + q_len > valid + qi {
                            mask[(b * q_len + qi) * ctx + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let scale = 1.0 / (hd as f32).sqrt();
            let mut want = vec![0f32; rows * n_heads * hd];
            for b in 0..batch {
                for qi in 0..q_len {
                    let row = b * q_len + qi;
                    for h in 0..n_heads {
                        let kvh = h / group;
                        let at = |c: usize| {
                            let blk = block_idx[b * max_blocks + c / block_size] as usize;
                            ((blk * block_size + c % block_size) * n_kv + kvh) * hd
                        };
                        let qh = &q[(row * n_heads + h) * hd..(row * n_heads + h + 1) * hd];
                        let scores: Vec<f64> = (0..ctx)
                            .map(|c| {
                                let dot: f64 = (0..hd).map(|d| qh[d] as f64 * cache_k[at(c) + d] as f64).sum();
                                dot * scale as f64 + mask[row * ctx + c] as f64
                            })
                            .collect();
                        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                        let total: f64 = exps.iter().sum();
                        for d in 0..hd {
                            let o: f64 = (0..ctx).map(|c| exps[c] * cache_v[at(c) + d] as f64).sum();
                            want[(row * n_heads + h) * hd + d] = (o / total) as f32;
                        }
                    }
                }
            }
            let (qb, kb, vb, bb, mb) = (
                MetalBuf::from_slice(&rt, &q),
                MetalBuf::from_slice(&rt, &cache_k),
                MetalBuf::from_slice(&rt, &cache_v),
                MetalBuf::from_slice(&rt, &block_idx),
                MetalBuf::from_slice(&rt, &mask),
            );
            let mut out = MetalBuf::<f32>::zeros(&rt, rows * n_heads * hd);
            attn_decode(
                &rt, &qb, &kb, &vb, &bb, &mb, &mut out, batch, q_len, n_heads, n_kv, hd,
                block_size as u32, max_blocks as u32, ctx,
            )
            .unwrap();
            let got = rt.read(&out).unwrap();
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "case {case} i={i}: got {g}, want {w}");
            }
        }
    }

    // The projection epilogue: bias, per-head norm, RoPE, and the scatter into
    // the paged cache, against a direct computation. Head widths past one
    // 32-lane pass, with and without bias / Q norm / K norm.
    #[test]
    fn attn_prep_matches_reference() {
        let Some(rt) = runtime() else { return };
        let block_size = 4usize;
        for &(hd, bias_on, qn_on, kn_on) in &[
            (64usize, true, true, true),
            (128, false, false, false),
            (6, true, false, true),
            (2, false, true, false),
        ] {
            let (n_heads, n_kv, rows) = (6usize, 2usize, 5usize);
            let half = hd / 2;
            let qkv_dim = (n_heads + 2 * n_kv) * hd;
            let qkv = values(rows * qkv_dim, 40);
            let bias = values(qkv_dim, 41);
            let q_norm = values(hd, 42);
            let k_norm = values(hd, 43);
            let cos = values(rows * half, 44);
            let sin = values(rows * half, 45);
            // Token r goes to (block r % 3 + 1, offset r % block_size).
            let block_ids: Vec<u32> = (0..rows as u32).map(|r| r % 3 + 1).collect();
            let offsets: Vec<u32> = (0..rows as u32).map(|r| r % block_size as u32).collect();
            let cache_len = 4 * block_size * n_kv * hd;
            let eps = 1e-5f32;

            let head = |row: usize, slot: usize| -> Vec<f32> {
                let base = row * qkv_dim + slot * hd;
                (0..hd).map(|d| qkv[base + d] + if bias_on { bias[slot * hd + d] } else { 0.0 }).collect()
            };
            let rotate = |row: usize, x: Vec<f32>, norm: Option<&Vec<f32>>| -> Vec<f32> {
                let x: Vec<f32> = match norm {
                    Some(w) => {
                        let ms = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hd as f64;
                        let rms = (ms + eps as f64).sqrt() as f32;
                        x.iter().zip(w).map(|(v, w)| v / rms * w).collect()
                    }
                    None => x,
                };
                let mut out = vec![0f32; hd];
                for j in 0..half {
                    let (c, s) = (cos[row * half + j], sin[row * half + j]);
                    out[j] = x[j] * c - x[half + j] * s;
                    out[half + j] = x[j] * s + x[half + j] * c;
                }
                out
            };
            let mut want_q = vec![0f32; rows * n_heads * hd];
            let mut want_k = vec![0f32; cache_len];
            let mut want_v = vec![0f32; cache_len];
            for row in 0..rows {
                for h in 0..n_heads {
                    let r = rotate(row, head(row, h), qn_on.then_some(&q_norm));
                    want_q[(row * n_heads + h) * hd..(row * n_heads + h + 1) * hd].copy_from_slice(&r);
                }
                for kvh in 0..n_kv {
                    let dest = (((block_ids[row] as usize * block_size + offsets[row] as usize) * n_kv) + kvh) * hd;
                    let r = rotate(row, head(row, n_heads + kvh), kn_on.then_some(&k_norm));
                    want_k[dest..dest + hd].copy_from_slice(&r);
                    want_v[dest..dest + hd].copy_from_slice(&head(row, n_heads + n_kv + kvh));
                }
            }

            let (qkv_b, bias_b, qn_b, kn_b, cos_b, sin_b, ids_b, off_b) = (
                MetalBuf::from_slice(&rt, &qkv),
                MetalBuf::from_slice(&rt, &bias),
                MetalBuf::from_slice(&rt, &q_norm),
                MetalBuf::from_slice(&rt, &k_norm),
                MetalBuf::from_slice(&rt, &cos),
                MetalBuf::from_slice(&rt, &sin),
                MetalBuf::from_slice(&rt, &block_ids),
                MetalBuf::from_slice(&rt, &offsets),
            );
            let (ck, cv) = (MetalBuf::<f32>::zeros(&rt, cache_len), MetalBuf::<f32>::zeros(&rt, cache_len));
            let mut q_out = MetalBuf::<f32>::zeros(&rt, rows * n_heads * hd);
            attn_prep(
                &rt, &qkv_b, bias_on.then_some(&bias_b), qn_on.then_some(&qn_b), kn_on.then_some(&kn_b),
                &cos_b, &sin_b, &ids_b, &off_b, &ck, &cv, &mut q_out, n_heads, n_kv, hd, block_size as u32, eps,
            )
            .unwrap();
            for (name, got, want) in [
                ("q", rt.read(&q_out).unwrap(), want_q),
                ("k cache", rt.read(&ck).unwrap(), want_k),
                ("v cache", rt.read(&cv).unwrap(), want_v),
            ] {
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert!((g - w).abs() < 1e-5, "hd={hd} {name} i={i}: got {g}, want {w}");
                }
            }
        }
    }

    #[test]
    fn add_rmsnorm_and_glu_match_reference() {
        let Some(rt) = runtime() else { return };
        for &(rows, dim) in &[(1usize, 960usize), (3, 64), (2, 3000), (5, 33)] {
            let (a, b, weight) = (values(rows * dim, 50), values(rows * dim, 51), values(dim, 52));
            let (ab, bb, wb) = (
                MetalBuf::from_slice(&rt, &a),
                MetalBuf::from_slice(&rt, &b),
                MetalBuf::from_slice(&rt, &weight),
            );
            let mut sum = MetalBuf::<f32>::zeros(&rt, rows * dim);
            let mut normed = MetalBuf::<f32>::zeros(&rt, rows * dim);
            add_rmsnorm_wrapper(&rt, &ab, &bb, &wb, dim as u32, 1e-5, &mut sum, &mut normed).unwrap();
            let (got_sum, got_norm) = (rt.read(&sum).unwrap(), rt.read(&normed).unwrap());
            for r in 0..rows {
                let s: Vec<f64> = (0..dim).map(|i| a[r * dim + i] as f64 + b[r * dim + i] as f64).collect();
                let rms = (s.iter().map(|v| v * v).sum::<f64>() / dim as f64 + 1e-5).sqrt();
                for i in 0..dim {
                    assert!((got_sum[r * dim + i] as f64 - s[i]).abs() < 1e-6, "sum r={r} i={i}");
                    let want = s[i] / rms * weight[i] as f64;
                    assert!((got_norm[r * dim + i] as f64 - want).abs() < 1e-5, "norm r={r} i={i}");
                }
            }
        }

        let (rows, ffn) = (3usize, 37usize);
        let gate_up = values(rows * 2 * ffn, 53);
        let gb = MetalBuf::from_slice(&rt, &gate_up);
        for gelu in [false, true] {
            let mut out = MetalBuf::<f32>::zeros(&rt, rows * ffn);
            glu_wrapper(&rt, gelu, &gb, ffn, &mut out).unwrap();
            let got = rt.read(&out).unwrap();
            for r in 0..rows {
                for j in 0..ffn {
                    let g = gate_up[r * 2 * ffn + j] as f64;
                    let u = gate_up[r * 2 * ffn + ffn + j] as f64;
                    let act = if gelu {
                        0.5 * g * (1.0 + (0.7978845608028654 * (g + 0.044715 * g * g * g)).tanh())
                    } else {
                        g / (1.0 + (-g).exp())
                    };
                    assert!((got[r * ffn + j] as f64 - act * u).abs() < 1e-5, "gelu={gelu} r={r} j={j}");
                }
            }
        }
    }
}
