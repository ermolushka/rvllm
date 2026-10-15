// CUDA mirror of model.rs's Layer::forward/Model::forward/forward_last.
// Same math, same shapes, but every Candle reshape/transpose that was free
// on CPU becomes an explicit kernel call here (transpose_axes12), and every
// matmul goes through cuBLAS (cublas_ops::linear/attention_scores/
// attention_apply) instead of Candle's matmul. No shared trait with the CPU
// Model - same reasoning as CudaKvStorage/CudaWeights: the CPU path is
// Candle-Tensor-typed end to end, this one is CudaSlice-typed end to end.
// Covers every family the CPU path loads: Llama 3's RoPE scaling, Qwen2's
// attention biases, Qwen3's Q/K norm, Gemma 3's local attention, post-norms,
// GELU and embedding scale, and LFM2's short convolutions.
use cudarc::driver::CudaSlice;
use half::f16;
use kv_cache_scheduler::block_pool::BlockID;

use crate::conv_cache::ConvItem;
use crate::model::{Activation, BatchItem, Config, Model};

use super::context::CudaRuntime;
use super::cublas_ops;
use super::kv_storage::{CudaGatherPlan, CudaKvStorage};
use super::ops;
use super::weights::{CudaAttention, CudaLayer, CudaMixer, CudaShortConv, CudaWeights};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// The RoPE tables and attention mask one kind of attention layer uses: the
// global ones, or a Gemma local layer's (its own RoPE base, its window).
// cos/sin are pre-expanded across the head axis here (on the host, cheap)
// since the CUDA rope kernel expects one cos/sin row per input row, and q/k
// have different head counts.
struct CudaAttnCtx {
    cos_q: CudaSlice<f32>,
    sin_q: CudaSlice<f32>,
    cos_k: CudaSlice<f32>,
    sin_k: CudaSlice<f32>,
    // [batch * q_len * ctx_len] flattened, broadcast across n_kv_heads/group_size.
    mask: CudaSlice<f32>,
}

// Everything that depends only on this call's items, not on the layer -
// same role as model.rs's StepCtx, built once per forward call and shared
// by every layer.
struct CudaStepCtx {
    global: CudaAttnCtx,
    // The local layers' tables, when the model has local layers.
    local: Option<CudaAttnCtx>,
    gather: CudaGatherPlan,
    block_ids: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    // Each item as the short-conv layers' cache sees it (empty without them).
    conv_items: Vec<ConvItem>,
    batch: usize,
    q_len: usize,
    ctx_len: usize,
}

// cos/sin for every (batch, q_len) row, not yet expanded across heads -
// same values model::rope_cos_sin computes, plain Rust instead of Candle.
// `factors` (Llama 3) divides each pair's frequency, as on the CPU.
fn rope_cos_sin_rows(
    rope_theta: f32,
    factors: Option<&[f32]>,
    head_dim: usize,
    positions: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let theta: Vec<f32> = (0..half)
        .map(|j| {
            let theta = rope_theta.powf(-2.0 * j as f32 / head_dim as f32);
            factors.map_or(theta, |factors| theta / factors[j])
        })
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for &t in &theta {
            let angle = p as f32 * t;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

// Repeats each (batch, q_len) row's cos/sin values across `n_heads` heads,
// producing rows in (batch, head, q_len) order - the layout q/k are in
// after `transpose_axes12`, which is what the rope kernel needs to line up
// against.
fn expand_across_heads(
    base: &[f32],
    batch: usize,
    q_len: usize,
    n_heads: usize,
    half: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch * n_heads * q_len * half);
    for b in 0..batch {
        for _h in 0..n_heads {
            for qi in 0..q_len {
                let row = b * q_len + qi;
                out.extend_from_slice(&base[row * half..row * half + half]);
            }
        }
    }
    out
}

impl CudaAttnCtx {
    // `window` hides keys that many positions or more behind the query, for
    // sliding-window attention; the mask rule is model::batch_mask's.
    #[allow(clippy::too_many_arguments)]
    fn new(
        cuda_runtime: &CudaRuntime,
        config: &Config,
        items: &[BatchItem],
        positions: &[usize],
        ctx_len: usize,
        rope_theta: f32,
        factors: Option<&[f32]>,
        window: Option<usize>,
    ) -> Result<Self> {
        let batch = items.len();
        let q_len = items[0].tokens.len();
        let head_dim = config.head_dim();
        let half = head_dim / 2;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;

        let (cos_rows, sin_rows) = rope_cos_sin_rows(rope_theta, factors, head_dim, positions);
        let cos_q = cuda_runtime
            .stream
            .clone_htod(&expand_across_heads(&cos_rows, batch, q_len, n_heads, half))?;
        let sin_q = cuda_runtime
            .stream
            .clone_htod(&expand_across_heads(&sin_rows, batch, q_len, n_heads, half))?;
        let cos_k = cuda_runtime.stream.clone_htod(&expand_across_heads(
            &cos_rows, batch, q_len, n_kv_heads, half,
        ))?;
        let sin_k = cuda_runtime.stream.clone_htod(&expand_across_heads(
            &sin_rows, batch, q_len, n_kv_heads, half,
        ))?;

        // Same masking rule as model::batch_mask: causal within the
        // sequence's real tokens, plus masked past its own read_num_tokens
        // (the batch-padding tail other sequences' longer contexts leave), and
        // past the window if there is one.
        let mut mask_data = vec![0f32; batch * q_len * ctx_len];
        for (b, item) in items.iter().enumerate() {
            let offset = item.position_offset;
            let valid_len = item.read_num_tokens;
            for i in 0..q_len {
                for j in 0..ctx_len {
                    let outside_window = window.is_some_and(|w| j + w <= offset + i);
                    if j >= valid_len || j > offset + i || outside_window {
                        mask_data[(b * q_len + i) * ctx_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        Ok(CudaAttnCtx {
            cos_q,
            sin_q,
            cos_k,
            sin_k,
            mask: cuda_runtime.stream.clone_htod(&mask_data)?,
        })
    }
}

impl CudaStepCtx {
    fn new(cuda_runtime: &CudaRuntime, config: &Config, items: &[BatchItem]) -> Result<Self> {
        let batch = items.len();
        let q_len = items[0].tokens.len();

        let positions: Vec<usize> = items
            .iter()
            .flat_map(|item| item.position_offset..item.position_offset + q_len)
            .collect();

        let gather_items: Vec<(&[BlockID], usize)> = items
            .iter()
            .map(|item| (item.read_blocks, item.read_num_tokens))
            .collect();
        let gather = CudaGatherPlan::new(cuda_runtime, &gather_items)?;
        let ctx_len = gather.ctx_len() as usize;

        let global = CudaAttnCtx::new(
            cuda_runtime,
            config,
            items,
            &positions,
            ctx_len,
            config.rope_theta,
            config.rope_factors.as_deref(),
            None,
        )?;
        let local = match config.local_attention {
            Some(local) => Some(CudaAttnCtx::new(
                cuda_runtime,
                config,
                items,
                &positions,
                ctx_len,
                local.rope_theta,
                None,
                Some(local.window),
            )?),
            None => None,
        };

        let writes: Vec<(BlockID, usize)> = items
            .iter()
            .flat_map(|item| item.write_positions.iter().copied())
            .collect();
        let block_ids: Vec<u32> = writes.iter().map(|&(b, _)| b.0).collect();
        let offsets: Vec<u32> = writes.iter().map(|&(_, o)| o as u32).collect();
        let block_ids = cuda_runtime.stream.clone_htod(&block_ids)?;
        let offsets = cuda_runtime.stream.clone_htod(&offsets)?;

        let conv_items = match config.conv {
            Some(_) => items
                .iter()
                .map(|item| ConvItem {
                    seq_id: item.seq_id,
                    position: item.position_offset,
                    blocks: item.read_blocks.to_vec(),
                    writes: item.write_positions.to_vec(),
                })
                .collect(),
            None => Vec::new(),
        };

        Ok(CudaStepCtx {
            global,
            local,
            gather,
            block_ids,
            offsets,
            conv_items,
            batch,
            q_len,
            ctx_len,
        })
    }
}

// Every intermediate a layer needs, allocated once per forward pass (sizes
// depend on the batch and context length, so not once per model) and reused
// by all layers, instead of allocating fresh buffers in each layer.
// Zero-initialized once; k_full/v_full padding slots keep whatever finite
// values earlier layers left there, which the attention mask hides.
struct Workspace {
    normed: CudaSlice<f32>,
    q_proj: CudaSlice<f32>,
    k_proj: CudaSlice<f32>,
    v_proj: CudaSlice<f32>,
    // Q/K after their per-head RMS norm (Qwen3, LFM2), when they have one.
    q_normed: CudaSlice<f32>,
    k_normed: CudaSlice<f32>,
    q_t: CudaSlice<f32>,
    k_t: CudaSlice<f32>,
    v_t: CudaSlice<f32>,
    q_roped: CudaSlice<f32>,
    k_roped: CudaSlice<f32>,
    k_tok: CudaSlice<f32>,
    v_tok: CudaSlice<f32>,
    k_full: CudaSlice<f32>,
    v_full: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    attn_out_t: CudaSlice<f32>,
    // The mixer's output (attention or short conv), projected back to hidden_dim.
    attn_proj: CudaSlice<f32>,
    // The mixer's output after its post-norm (Gemma), when it has one.
    attn_post: CudaSlice<f32>,
    h: CudaSlice<f32>,
    normed2: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    ffn_mid: CudaSlice<f32>,
    ffn_out: CudaSlice<f32>,
    // The FFN's output after its post-norm (Gemma), when it has one.
    ffn_post: CudaSlice<f32>,
    // Short-conv scratch: in_proj's output, B*x, the window of rows the
    // filter reads, and the gated filter output.
    conv_proj: CudaSlice<f32>,
    conv_bx: CudaSlice<f32>,
    conv_window: CudaSlice<f32>,
    conv_out: CudaSlice<f32>,
    // F16 copy of the current linear input, sized for the widest one.
    x16: CudaSlice<f16>,
}

impl Workspace {
    fn new(cuda_runtime: &CudaRuntime, config: &Config, ctx: &CudaStepCtx) -> Result<Self> {
        let hidden_dim = config.hidden_dim as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let head_dim = config.head_dim();
        let ffn_dim = config.ffn_dim as usize;
        let rows = ctx.batch * ctx.q_len;
        let attn_rows = (n_heads / n_kv_heads) * ctx.q_len;
        let q_size = rows * n_heads * head_dim;
        let kv_size = rows * n_kv_heads * head_dim;
        let full_size = ctx.batch * n_kv_heads * ctx.ctx_len * head_dim;
        let scores_size = ctx.batch * n_kv_heads * attn_rows * ctx.ctx_len;
        let widest = hidden_dim
            .max(n_heads * head_dim)
            .max(ffn_dim)
            .max(3 * hidden_dim);
        // A conv window of kernel - 1 + q_len rows per item; zero without conv.
        let kernel = config.conv.as_ref().map_or(1, |conv| conv.kernel);
        let window_rows = ctx.batch * (kernel - 1 + ctx.q_len);
        let stream = &cuda_runtime.stream;
        Ok(Workspace {
            normed: stream.alloc_zeros(rows * hidden_dim)?,
            q_proj: stream.alloc_zeros(q_size)?,
            k_proj: stream.alloc_zeros(kv_size)?,
            v_proj: stream.alloc_zeros(kv_size)?,
            q_normed: stream.alloc_zeros(q_size)?,
            k_normed: stream.alloc_zeros(kv_size)?,
            q_t: stream.alloc_zeros(q_size)?,
            k_t: stream.alloc_zeros(kv_size)?,
            v_t: stream.alloc_zeros(kv_size)?,
            q_roped: stream.alloc_zeros(q_size)?,
            k_roped: stream.alloc_zeros(kv_size)?,
            k_tok: stream.alloc_zeros(kv_size)?,
            v_tok: stream.alloc_zeros(kv_size)?,
            k_full: stream.alloc_zeros(full_size)?,
            v_full: stream.alloc_zeros(full_size)?,
            scores: stream.alloc_zeros(scores_size)?,
            probs: stream.alloc_zeros(scores_size)?,
            attn_out: stream.alloc_zeros(q_size)?,
            attn_out_t: stream.alloc_zeros(q_size)?,
            attn_proj: stream.alloc_zeros(rows * hidden_dim)?,
            attn_post: stream.alloc_zeros(rows * hidden_dim)?,
            h: stream.alloc_zeros(rows * hidden_dim)?,
            normed2: stream.alloc_zeros(rows * hidden_dim)?,
            gate: stream.alloc_zeros(rows * ffn_dim)?,
            up: stream.alloc_zeros(rows * ffn_dim)?,
            ffn_mid: stream.alloc_zeros(rows * ffn_dim)?,
            ffn_out: stream.alloc_zeros(rows * hidden_dim)?,
            ffn_post: stream.alloc_zeros(rows * hidden_dim)?,
            conv_proj: stream.alloc_zeros(rows * 3 * hidden_dim)?,
            conv_bx: stream.alloc_zeros(rows * hidden_dim)?,
            conv_window: stream.alloc_zeros((window_rows * hidden_dim).max(1))?,
            conv_out: stream.alloc_zeros(rows * hidden_dim)?,
            x16: stream.alloc_zeros(rows * widest)?,
        })
    }
}

// The attention mixer: writes its output, projected to hidden_dim, into
// ws.attn_proj. `ws.x16` must hold the normalized input as F16 on entry.
fn attention_mixer(
    cuda_runtime: &CudaRuntime,
    config: &Config,
    attn: &CudaAttention,
    ctx: &CudaStepCtx,
    kv_storage: &mut CudaKvStorage,
    ws: &mut Workspace,
) -> Result<()> {
    let hidden_dim = config.hidden_dim as usize;
    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let head_dim = config.head_dim();
    let group_size = n_heads / n_kv_heads;
    let batch = ctx.batch;
    let q_len = ctx.q_len;
    let ctx_len = ctx.ctx_len;
    let rows = batch * q_len;
    let eps = config.rms_eps;

    // Project into Q/K/V, naturally [batch, q_len, heads, head_dim] flat.
    // y = x @ weight^T. q/k/v read the same input, already narrowed to F16.
    cublas_ops::linear_f16(
        cuda_runtime,
        &ws.x16,
        &attn.q,
        rows,
        hidden_dim,
        n_heads * head_dim,
        &mut ws.q_proj,
    )?;
    cublas_ops::linear_f16(
        cuda_runtime,
        &ws.x16,
        &attn.k,
        rows,
        hidden_dim,
        n_kv_heads * head_dim,
        &mut ws.k_proj,
    )?;
    cublas_ops::linear_f16(
        cuda_runtime,
        &ws.x16,
        &attn.v,
        rows,
        hidden_dim,
        n_kv_heads * head_dim,
        &mut ws.v_proj,
    )?;
    if let Some(bias) = &attn.q_bias {
        ops::add_bias_wrapper(cuda_runtime, &mut ws.q_proj, bias)?;
    }
    if let Some(bias) = &attn.k_bias {
        ops::add_bias_wrapper(cuda_runtime, &mut ws.k_proj, bias)?;
    }
    if let Some(bias) = &attn.v_bias {
        ops::add_bias_wrapper(cuda_runtime, &mut ws.v_proj, bias)?;
    }

    // Per-head RMS norm on Q and K (each head is a head_dim-wide row), before
    // RoPE, as on the CPU.
    let q_src = match &attn.q_norm {
        Some(norm) => {
            ops::rmsnorm_wrapper(
                cuda_runtime,
                &ws.q_proj,
                norm,
                head_dim as u32,
                eps,
                &mut ws.q_normed,
            )?;
            &ws.q_normed
        }
        None => &ws.q_proj,
    };
    let k_src = match &attn.k_norm {
        Some(norm) => {
            ops::rmsnorm_wrapper(
                cuda_runtime,
                &ws.k_proj,
                norm,
                head_dim as u32,
                eps,
                &mut ws.k_normed,
            )?;
            &ws.k_normed
        }
        None => &ws.k_proj,
    };

    // [batch, q_len, heads, head_dim] -> [batch, heads, q_len, head_dim] -
    // real data movement here, unlike Candle's free `.transpose(1, 2)`.
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        q_src,
        &mut ws.q_t,
        batch as u32,
        q_len as u32,
        n_heads as u32,
        head_dim as u32,
    )?;
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        k_src,
        &mut ws.k_t,
        batch as u32,
        q_len as u32,
        n_kv_heads as u32,
        head_dim as u32,
    )?;
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        &ws.v_proj,
        &mut ws.v_t,
        batch as u32,
        q_len as u32,
        n_kv_heads as u32,
        head_dim as u32,
    )?;

    // Local layers rotate with their own RoPE base and see only their window.
    let actx = match &ctx.local {
        Some(local) if attn.local => local,
        _ => &ctx.global,
    };

    // RoPE rotates Q and K given position; V carries content, not position.
    ops::rope_wrapper(
        cuda_runtime,
        &ws.q_t,
        &actx.cos_q,
        &actx.sin_q,
        &mut ws.q_roped,
        head_dim as u32,
    )?;
    ops::rope_wrapper(
        cuda_runtime,
        &ws.k_t,
        &actx.cos_k,
        &actx.sin_k,
        &mut ws.k_roped,
        head_dim as u32,
    )?;

    // Cache wants token-major rows [batch * q_len, n_kv_heads, head_dim] -
    // transpose back out of head-major layout before writing.
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        &ws.k_roped,
        &mut ws.k_tok,
        batch as u32,
        n_kv_heads as u32,
        q_len as u32,
        head_dim as u32,
    )?;
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        &ws.v_t,
        &mut ws.v_tok,
        batch as u32,
        n_kv_heads as u32,
        q_len as u32,
        head_dim as u32,
    )?;
    kv_storage.write_tokens(
        cuda_runtime,
        attn.kv_index,
        &ctx.block_ids,
        &ctx.offsets,
        &ws.k_tok,
        &ws.v_tok,
    )?;

    kv_storage.gather(
        cuda_runtime,
        attn.kv_index,
        &ctx.gather,
        &mut ws.k_full,
        &mut ws.v_full,
    )?;

    cublas_ops::attention_scores(
        cuda_runtime,
        &ws.q_roped,
        &ws.k_full,
        &mut ws.scores,
        batch,
        n_kv_heads,
        group_size,
        q_len,
        ctx_len,
        head_dim,
    )?;
    ops::mask_add_broadcast_wrapper(
        cuda_runtime,
        &mut ws.scores,
        &actx.mask,
        n_kv_heads as u32,
        group_size as u32,
        q_len as u32,
        ctx_len as u32,
    )?;
    ops::softmax_wrapper(cuda_runtime, &ws.scores, ctx_len as u32, &mut ws.probs)?;
    cublas_ops::attention_apply(
        cuda_runtime,
        &ws.probs,
        &ws.v_full,
        &mut ws.attn_out,
        batch,
        n_kv_heads,
        group_size,
        q_len,
        ctx_len,
        head_dim,
    )?;

    // [batch, heads, q_len, head_dim] -> [batch, q_len, heads, head_dim],
    // inverse of the earlier Q/K/V split, ready to flatten into hidden_dim.
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        &ws.attn_out,
        &mut ws.attn_out_t,
        batch as u32,
        n_heads as u32,
        q_len as u32,
        head_dim as u32,
    )?;
    cublas_ops::linear(
        cuda_runtime,
        &ws.attn_out_t,
        &attn.o,
        rows,
        n_heads * head_dim,
        hidden_dim,
        &mut ws.attn_proj,
        &mut ws.x16,
    )?;
    Ok(())
}

// LFM2's short-convolution mixer, the GPU side of model::ShortConv: writes
// its output, projected to hidden_dim, into ws.attn_proj. `ws.x16` must hold
// the normalized input as F16 on entry. The state each item needs comes from
// the conv cache in `kv_storage`, which is updated for the next call.
fn short_conv_mixer(
    cuda_runtime: &CudaRuntime,
    config: &Config,
    conv: &CudaShortConv,
    ctx: &CudaStepCtx,
    kv_storage: &mut CudaKvStorage,
    ws: &mut Workspace,
) -> Result<()> {
    let hidden = config.hidden_dim as usize;
    let rows = ctx.batch * ctx.q_len;
    let q_len = ctx.q_len;
    let n = conv.kernel - 1;

    // in_proj: [rows, 3 * hidden] with B, C and x; B * x is what the filter sees.
    cublas_ops::linear_f16(
        cuda_runtime,
        &ws.x16,
        &conv.in_proj,
        rows,
        hidden,
        3 * hidden,
        &mut ws.conv_proj,
    )?;
    ops::short_conv_bx_wrapper(cuda_runtime, &ws.conv_proj, &mut ws.conv_bx, hidden as u32)?;

    // Each item's window: the n rows before its first token (from the cache),
    // then its q_len new bx rows.
    let item_rows = (n + q_len) * hidden;
    for (b, item) in ctx.conv_items.iter().enumerate() {
        let before = kv_storage
            .conv
            .rows_before(cuda_runtime, conv.conv_index, item, n, hidden)?;
        let base = b * item_rows;
        cuda_runtime.stream.memcpy_dtod(
            &before,
            &mut ws.conv_window.slice_mut(base..base + n * hidden),
        )?;
        cuda_runtime.stream.memcpy_dtod(
            &ws.conv_bx
                .slice(b * q_len * hidden..(b + 1) * q_len * hidden),
            &mut ws
                .conv_window
                .slice_mut(base + n * hidden..base + item_rows),
        )?;
    }
    kv_storage.conv.update(
        cuda_runtime,
        conv.conv_index,
        &ctx.conv_items,
        &ws.conv_window,
        q_len,
        n,
        hidden,
    )?;

    // Gate the filter's output with C, then project down to hidden_dim.
    ops::short_conv_filter_gate_wrapper(
        cuda_runtime,
        &ws.conv_window,
        &conv.filter,
        &ws.conv_proj,
        &mut ws.conv_out,
        ctx.batch as u32,
        q_len as u32,
        hidden as u32,
        conv.kernel as u32,
    )?;
    cublas_ops::linear(
        cuda_runtime,
        &ws.conv_out,
        &conv.out_proj,
        rows,
        hidden,
        hidden,
        &mut ws.attn_proj,
        &mut ws.x16,
    )?;
    Ok(())
}

// One transformer block, same structure as model.rs's Layer::forward:
//   h   = x + post_attn(mixer(rms_norm(x, attn_norm)))
//   out = h + post_ffn(ffn(rms_norm(h, ffn_norm)))
// where mixer is attention or a short conv, the post norms exist for Gemma,
// and ffn is SwiGLU or GeGLU. x: [batch * q_len, hidden_dim] flattened.
#[allow(clippy::too_many_arguments)]
fn layer_forward(
    cuda_runtime: &CudaRuntime,
    config: &Config,
    weights: &CudaLayer,
    ctx: &CudaStepCtx,
    kv_storage: &mut CudaKvStorage,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    ws: &mut Workspace,
) -> Result<()> {
    let hidden_dim = config.hidden_dim as usize;
    let ffn_dim = config.ffn_dim as usize;
    let rows = ctx.batch * ctx.q_len;
    let eps = config.rms_eps;

    ops::rmsnorm_wrapper(
        cuda_runtime,
        x,
        &weights.attn_norm,
        hidden_dim as u32,
        eps,
        &mut ws.normed,
    )?;
    cublas_ops::to_f16(cuda_runtime, &ws.normed, &mut ws.x16)?;
    match &weights.mixer {
        CudaMixer::Attention(attn) => {
            attention_mixer(cuda_runtime, config, attn, ctx, kv_storage, ws)?
        }
        CudaMixer::ShortConv(conv) => {
            short_conv_mixer(cuda_runtime, config, conv, ctx, kv_storage, ws)?
        }
    }
    if let Some(norm) = &weights.post_attn_norm {
        ops::rmsnorm_wrapper(
            cuda_runtime,
            &ws.attn_proj,
            norm,
            hidden_dim as u32,
            eps,
            &mut ws.attn_post,
        )?;
        std::mem::swap(&mut ws.attn_proj, &mut ws.attn_post);
    }

    ops::add_wrapper(cuda_runtime, x, &ws.attn_proj, &mut ws.h)?;

    ops::rmsnorm_wrapper(
        cuda_runtime,
        &ws.h,
        &weights.ffn_norm,
        hidden_dim as u32,
        eps,
        &mut ws.normed2,
    )?;
    cublas_ops::to_f16(cuda_runtime, &ws.normed2, &mut ws.x16)?;
    cublas_ops::linear_f16(
        cuda_runtime,
        &ws.x16,
        &weights.ffn_gate,
        rows,
        hidden_dim,
        ffn_dim,
        &mut ws.gate,
    )?;
    cublas_ops::linear_f16(
        cuda_runtime,
        &ws.x16,
        &weights.ffn_up,
        rows,
        hidden_dim,
        ffn_dim,
        &mut ws.up,
    )?;
    match config.activation {
        Activation::Silu => {
            ops::silu_gate_multiply_wrapper(cuda_runtime, &ws.gate, &ws.up, &mut ws.ffn_mid)?
        }
        Activation::GeluTanh => {
            ops::gelu_gate_multiply_wrapper(cuda_runtime, &ws.gate, &ws.up, &mut ws.ffn_mid)?
        }
    }
    cublas_ops::linear(
        cuda_runtime,
        &ws.ffn_mid,
        &weights.ffn_down,
        rows,
        ffn_dim,
        hidden_dim,
        &mut ws.ffn_out,
        &mut ws.x16,
    )?;
    if let Some(norm) = &weights.post_ffn_norm {
        ops::rmsnorm_wrapper(
            cuda_runtime,
            &ws.ffn_out,
            norm,
            hidden_dim as u32,
            eps,
            &mut ws.ffn_post,
        )?;
        std::mem::swap(&mut ws.ffn_out, &mut ws.ffn_post);
    }

    ops::add_wrapper(cuda_runtime, &ws.h, &ws.ffn_out, out)?;
    Ok(())
}

pub struct CudaModel {
    pub config: Config,
    pub weights: CudaWeights,
}

impl CudaModel {
    pub fn upload(
        cuda_runtime: &CudaRuntime,
        model: &Model,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        Ok(CudaModel {
            config: model.config.clone(),
            weights: CudaWeights::upload(cuda_runtime, model)?,
        })
    }

    // Embed -> N x layer_forward -> final hidden states, flat
    // [batch * q_len, hidden_dim], before the final norm and LM head.
    fn hidden(
        &self,
        cuda_runtime: &CudaRuntime,
        items: &[BatchItem],
        kv_storage: &mut CudaKvStorage,
    ) -> Result<(CudaSlice<f32>, usize, usize)> {
        let batch = items.len();
        let q_len = items[0].tokens.len();
        let hidden_dim = self.config.hidden_dim as usize;
        let rows = batch * q_len;

        let flat_tokens: Vec<u32> = items
            .iter()
            .flat_map(|item| item.tokens.iter().copied())
            .collect();
        let tokens_dev = cuda_runtime.stream.clone_htod(&flat_tokens)?;
        let mut x = cuda_runtime.stream.alloc_zeros::<f32>(rows * hidden_dim)?;
        ops::embed_lookup_wrapper(
            cuda_runtime,
            &tokens_dev,
            &self.weights.token_embd,
            hidden_dim as u32,
            &mut x,
        )?;
        if let Some(scale) = self.config.embedding_scale {
            ops::scale_inplace_wrapper(cuda_runtime, &mut x, scale)?;
        }

        let step_ctx = CudaStepCtx::new(cuda_runtime, &self.config, items)?;
        let mut ws = Workspace::new(cuda_runtime, &self.config, &step_ctx)?;
        // Ping-pong between two hidden-state buffers: a layer reads `x` and
        // writes `x_next`, then they swap.
        let mut x_next = cuda_runtime.stream.alloc_zeros::<f32>(rows * hidden_dim)?;
        for layer in self.weights.layers.iter() {
            layer_forward(
                cuda_runtime,
                &self.config,
                layer,
                &step_ctx,
                kv_storage,
                &x,
                &mut x_next,
                &mut ws,
            )?;
            std::mem::swap(&mut x, &mut x_next);
        }
        Ok((x, batch, q_len))
    }

    fn lm_head(
        &self,
        cuda_runtime: &CudaRuntime,
        x: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<CudaSlice<f32>> {
        let hidden_dim = self.config.hidden_dim as usize;
        let vocab_size = self.config.vocab_size as usize;
        let mut normed = cuda_runtime.stream.alloc_zeros::<f32>(rows * hidden_dim)?;
        ops::rmsnorm_wrapper(
            cuda_runtime,
            x,
            &self.weights.output_norm,
            hidden_dim as u32,
            self.config.rms_eps,
            &mut normed,
        )?;
        let mut x16 = cuda_runtime.stream.alloc_zeros::<f16>(rows * hidden_dim)?;
        let mut logits = cuda_runtime.stream.alloc_zeros::<f32>(rows * vocab_size)?;
        cublas_ops::linear(
            cuda_runtime,
            &normed,
            &self.weights.output,
            rows,
            hidden_dim,
            vocab_size,
            &mut logits,
            &mut x16,
        )?;
        Ok(logits)
    }

    // Full forward pass over a batch: logits, flat [batch * q_len, vocab_size].
    pub fn forward(
        &self,
        cuda_runtime: &CudaRuntime,
        items: &[BatchItem],
        kv_storage: &mut CudaKvStorage,
    ) -> Result<CudaSlice<f32>> {
        let (x, batch, q_len) = self.hidden(cuda_runtime, items, kv_storage)?;
        self.lm_head(cuda_runtime, &x, batch * q_len)
    }

    // Same, but only projects each item's last row: flat [batch, vocab_size].
    pub fn forward_last(
        &self,
        cuda_runtime: &CudaRuntime,
        items: &[BatchItem],
        kv_storage: &mut CudaKvStorage,
    ) -> Result<CudaSlice<f32>> {
        let (x, batch, q_len) = self.hidden(cuda_runtime, items, kv_storage)?;
        let hidden_dim = self.config.hidden_dim as usize;
        let mut last = cuda_runtime.stream.alloc_zeros::<f32>(batch * hidden_dim)?;
        ops::narrow_last_row_wrapper(cuda_runtime, &x, q_len as u32, hidden_dim as u32, &mut last)?;
        self.lm_head(cuda_runtime, &last, batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use kv_cache_scheduler::sequence::SequenceId;

    use crate::kv_storage::KvStorage;
    use crate::model::{Attention, ConvLayers, Ffn, Layer, LocalAttention, Mixer, ShortConv};

    fn runtime() -> Option<CudaRuntime> {
        crate::cuda::context::test_runtime()
    }

    // Same tiny fake model as model.rs's own `tiny_model` test helper: 1
    // layer, hidden_dim=4, n_heads=2, n_kv_heads=1 (head_dim=2), ffn_dim=4,
    // vocab_size=5.
    fn tiny_model() -> Model {
        let device = Device::Cpu;
        let (hidden_dim, n_heads, n_kv_heads, ffn_dim, vocab_size) =
            (4usize, 2usize, 1usize, 4usize, 5usize);
        let head_dim = hidden_dim / n_heads;
        let rand = |shape: (usize, usize)| Tensor::rand(0f32, 1., shape, &device).unwrap();
        let ones = || Tensor::ones(hidden_dim, DType::F32, &device).unwrap();

        let token_embd = rand((vocab_size, hidden_dim));
        let layer = Layer {
            attn_norm: ones(),
            mixer: Mixer::Attention(Attention {
                kv_index: 0,
                local: false,
                q: rand((n_heads * head_dim, hidden_dim)),
                k: rand((n_kv_heads * head_dim, hidden_dim)),
                v: rand((n_kv_heads * head_dim, hidden_dim)),
                o: rand((hidden_dim, n_heads * head_dim)),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
            }),
            post_attn_norm: None,
            ffn_norm: ones(),
            ffn: Ffn {
                gate: rand((ffn_dim, hidden_dim)),
                up: rand((ffn_dim, hidden_dim)),
                down: rand((hidden_dim, ffn_dim)),
            },
            post_ffn_norm: None,
        };
        Model {
            config: Config {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                hidden_dim: hidden_dim as u32,
                head_dim: head_dim as u32,
                ffn_dim: ffn_dim as u32,
                vocab_size: vocab_size as u32,
                ..Config::for_tests()
            },
            token_embd,
            layers: vec![layer],
            output_norm: ones(),
            output: rand((vocab_size, hidden_dim)),
        }
    }

    fn item<'a>(
        tokens: &'a [u32],
        offset: usize,
        blocks: &'a [BlockID],
        block_size: usize,
        writes: &'a mut Vec<(BlockID, usize)>,
    ) -> BatchItem<'a> {
        *writes = (offset..offset + tokens.len())
            .map(|p| (blocks[p / block_size], p % block_size))
            .collect();
        BatchItem {
            seq_id: SequenceId(0),
            tokens,
            write_positions: writes,
            read_blocks: blocks,
            read_num_tokens: offset + tokens.len(),
            position_offset: offset,
        }
    }

    // End-to-end correctness: the same prompt through the CPU/Candle path
    // and the CUDA path must produce the same logits (within tolerance),
    // and therefore the same greedy-argmax token - the actual outcome gate
    // PLAN.md's M6 names. Covers forward_last (prefill) and forward
    // (decode/batched) both, and a batch of 2 differently-positioned items.
    #[test]
    fn cuda_forward_matches_cpu_forward() {
        let Some(rt) = runtime() else { return };
        let device = Device::Cpu;
        let model = tiny_model();
        let cuda_model = CudaModel::upload(&rt, &model).unwrap();

        let tokens: [u32; 5] = [1, 3, 2, 0, 4];
        let blocks = [BlockID(0), BlockID(1), BlockID(2)];
        let mut writes = Vec::new();
        let items = [item(&tokens, 0, &blocks, 2, &mut writes)];

        let mut kv_cpu = KvStorage::new(1, 3, 2, 1, 2, &device).unwrap();
        let mut kv_cuda = CudaKvStorage::new(&rt, 1, 3, 2, 1, 2).unwrap();

        let cpu_logits = model.forward_last(&items, &mut kv_cpu).unwrap();
        let cpu_flat: Vec<f32> = cpu_logits.flatten_all().unwrap().to_vec1().unwrap();

        let cuda_logits = cuda_model.forward_last(&rt, &items, &mut kv_cuda).unwrap();
        let cuda_flat = rt.stream.clone_dtoh(&cuda_logits).unwrap();

        assert_eq!(cuda_flat.len(), cpu_flat.len());
        for (i, (&c, &g)) in cpu_flat.iter().zip(cuda_flat.iter()).enumerate() {
            let tol = 1e-2 * c.abs().max(1.0);
            assert!((c - g).abs() < tol, "i={i}: cpu {c}, cuda {g}");
        }
        let cpu_argmax = cpu_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let cuda_argmax = cuda_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(cpu_argmax, cuda_argmax, "greedy token mismatch");
    }

    #[test]
    fn cuda_forward_matches_cpu_forward_batched_decode() {
        let Some(rt) = runtime() else { return };
        let device = Device::Cpu;
        let model = tiny_model();
        let cuda_model = CudaModel::upload(&rt, &model).unwrap();

        let mut kv_cpu = KvStorage::new(1, 6, 2, 1, 2, &device).unwrap();
        let mut kv_cuda = CudaKvStorage::new(&rt, 1, 6, 2, 1, 2).unwrap();
        let blocks_a = [BlockID(0), BlockID(1)];
        let blocks_b = [BlockID(2), BlockID(3), BlockID(4)];

        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        model
            .forward(&[item(&[0, 2, 1], 0, &blocks_a, 2, &mut wa)], &mut kv_cpu)
            .unwrap();
        cuda_model
            .forward(
                &rt,
                &[item(&[0, 2, 1], 0, &blocks_a, 2, &mut wa)],
                &mut kv_cuda,
            )
            .unwrap();
        model
            .forward(
                &[item(&[1, 3, 0, 2, 4, 1], 0, &blocks_b, 2, &mut wb)],
                &mut kv_cpu,
            )
            .unwrap();
        cuda_model
            .forward(
                &rt,
                &[item(&[1, 3, 0, 2, 4, 1], 0, &blocks_b, 2, &mut wb)],
                &mut kv_cuda,
            )
            .unwrap();

        let blocks_b_ext = [BlockID(2), BlockID(3), BlockID(4), BlockID(5)];
        let items = [
            item(&[4], 3, &blocks_a, 2, &mut wa),
            item(&[4], 6, &blocks_b_ext, 2, &mut wb),
        ];
        let cpu_logits = model.forward(&items, &mut kv_cpu).unwrap();
        let cpu_flat: Vec<f32> = cpu_logits.flatten_all().unwrap().to_vec1().unwrap();
        let cuda_logits = cuda_model.forward(&rt, &items, &mut kv_cuda).unwrap();
        let cuda_flat = rt.stream.clone_dtoh(&cuda_logits).unwrap();

        assert_eq!(cuda_flat.len(), cpu_flat.len());
        for (i, (&c, &g)) in cpu_flat.iter().zip(cuda_flat.iter()).enumerate() {
            let tol = 1e-2 * c.abs().max(1.0);
            assert!((c - g).abs() < tol, "i={i}: cpu {c}, cuda {g}");
        }
    }

    // The tiny model's only layer's attention.
    fn attention(model: &mut Model) -> &mut Attention {
        match &mut model.layers[0].mixer {
            Mixer::Attention(attn) => attn,
            Mixer::ShortConv(_) => unreachable!("the tiny model's layer is attention"),
        }
    }

    // A 2-channel-wide random tensor in [-0.5, 0.5), for building test weights.
    fn rnd(shape: (usize, usize)) -> Tensor {
        Tensor::rand(-0.5f32, 0.5, shape, &Device::Cpu).unwrap()
    }

    fn ones_vec(n: usize) -> Tensor {
        Tensor::rand(0.5f32, 1.5, n, &Device::Cpu).unwrap()
    }

    // An attention layer of the tiny shape (hidden 4, 2 heads, 1 KV head).
    fn attention_layer(kv_index: usize, local: bool) -> Layer {
        Layer {
            attn_norm: ones_vec(4),
            mixer: Mixer::Attention(Attention {
                kv_index,
                local,
                q: rnd((4, 4)),
                k: rnd((2, 4)),
                v: rnd((2, 4)),
                o: rnd((4, 4)),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
            }),
            post_attn_norm: None,
            ffn_norm: ones_vec(4),
            ffn: Ffn {
                gate: rnd((4, 4)),
                up: rnd((4, 4)),
                down: rnd((4, 4)),
            },
            post_ffn_norm: None,
        }
    }

    // A LFM2 short-convolution layer of the tiny shape, with `kernel` taps.
    fn conv_layer(conv_index: usize, kernel: usize) -> Layer {
        Layer {
            attn_norm: ones_vec(4),
            mixer: Mixer::ShortConv(ShortConv {
                conv_index,
                in_proj: rnd((12, 4)),
                conv: rnd((4, kernel)),
                out_proj: rnd((4, 4)),
            }),
            post_attn_norm: None,
            ffn_norm: ones_vec(4),
            ffn: Ffn {
                gate: rnd((4, 4)),
                up: rnd((4, 4)),
                down: rnd((4, 4)),
            },
            post_ffn_norm: None,
        }
    }

    // A model over `layers` with the tiny shape and `tweak` applied to its config.
    fn model_with(layers: Vec<Layer>, tweak: impl FnOnce(&mut Config)) -> Model {
        let mut config = Config {
            n_layers: layers.len() as u32,
            n_heads: 2,
            n_kv_heads: 1,
            hidden_dim: 4,
            head_dim: 2,
            ffn_dim: 4,
            vocab_size: 5,
            ..Config::for_tests()
        };
        tweak(&mut config);
        Model {
            config,
            token_embd: rnd((5, 4)),
            layers,
            output_norm: ones_vec(4),
            output: rnd((5, 4)),
        }
    }

    // Each item as `item`, but for sequence `seq` (conv state is per sequence).
    fn item_seq<'a>(
        seq: i32,
        tokens: &'a [u32],
        offset: usize,
        blocks: &'a [BlockID],
        block_size: usize,
        writes: &'a mut Vec<(BlockID, usize)>,
    ) -> BatchItem<'a> {
        let mut item = item(tokens, offset, blocks, block_size, writes);
        item.seq_id = SequenceId(seq);
        item
    }

    fn assert_logits_close(cpu: &Tensor, cuda: &CudaSlice<f32>, rt: &CudaRuntime, what: &str) {
        let cpu_flat: Vec<f32> = cpu.flatten_all().unwrap().to_vec1().unwrap();
        let cuda_flat = rt.stream.clone_dtoh(cuda).unwrap();
        assert_eq!(cuda_flat.len(), cpu_flat.len(), "{what}: logits length");
        for (i, (&c, &g)) in cpu_flat.iter().zip(cuda_flat.iter()).enumerate() {
            let tol = 1e-2 * c.abs().max(1.0);
            assert!((c - g).abs() < tol, "{what} i={i}: cpu {c}, cuda {g}");
        }
    }

    // Prefill three tokens, then decode one more, on the CPU and on CUDA, and
    // compare every row's logits. Covers every feature the model uses.
    fn assert_cuda_matches_cpu(rt: &CudaRuntime, model: &Model) {
        let n_attention = model.config.n_attention_layers();
        let cuda_model = CudaModel::upload(rt, model).unwrap();
        let mut kv_cpu = KvStorage::new(n_attention, 3, 2, 1, 2, &Device::Cpu).unwrap();
        let mut kv_cuda = CudaKvStorage::new(rt, n_attention, 3, 2, 1, 2).unwrap();
        let blocks = [BlockID(0), BlockID(1)];

        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        let cpu = model
            .forward(&[item(&[1, 3, 2], 0, &blocks, 2, &mut wa)], &mut kv_cpu)
            .unwrap();
        let cuda = cuda_model
            .forward(
                rt,
                &[item(&[1, 3, 2], 0, &blocks, 2, &mut wa)],
                &mut kv_cuda,
            )
            .unwrap();
        assert_logits_close(&cpu, &cuda, rt, "prefill");

        let cpu = model
            .forward(&[item(&[4], 3, &blocks, 2, &mut wb)], &mut kv_cpu)
            .unwrap();
        let cuda = cuda_model
            .forward(rt, &[item(&[4], 3, &blocks, 2, &mut wb)], &mut kv_cuda)
            .unwrap();
        assert_logits_close(&cpu, &cuda, rt, "decode");
    }

    #[test]
    fn cuda_matches_cpu_with_attention_biases() {
        let Some(rt) = runtime() else { return };
        let mut model = tiny_model();
        let attn = attention(&mut model);
        attn.q_bias = Some(rnd((4, 1)).flatten_all().unwrap());
        attn.k_bias = Some(rnd((2, 1)).flatten_all().unwrap());
        attn.v_bias = Some(rnd((2, 1)).flatten_all().unwrap());
        assert_cuda_matches_cpu(&rt, &model);
    }

    #[test]
    fn cuda_matches_cpu_with_llama3_rope_factors() {
        let Some(rt) = runtime() else { return };
        let mut model = tiny_model();
        model.config.rope_factors = Some(vec![8.0]);
        assert_cuda_matches_cpu(&rt, &model);
    }

    #[test]
    fn cuda_matches_cpu_with_qk_norm() {
        let Some(rt) = runtime() else { return };
        let mut model = tiny_model();
        let attn = attention(&mut model);
        attn.q_norm = Some(ones_vec(2));
        attn.k_norm = Some(ones_vec(2));
        assert_cuda_matches_cpu(&rt, &model);
    }

    #[test]
    fn cuda_matches_cpu_gemma_style_layers() {
        let Some(rt) = runtime() else { return };
        // Layer 0 is a local layer (its own RoPE base, a one-token window),
        // layer 1 global; both have post norms, the FFN is GeGLU, and the
        // embeddings are scaled.
        let mut local = attention_layer(0, true);
        local.post_attn_norm = Some(ones_vec(4));
        local.post_ffn_norm = Some(ones_vec(4));
        let mut global = attention_layer(1, false);
        global.post_attn_norm = Some(ones_vec(4));
        global.post_ffn_norm = Some(ones_vec(4));
        let model = model_with(vec![local, global], |config| {
            config.activation = Activation::GeluTanh;
            config.embedding_scale = Some(2.0);
            config.local_attention = Some(LocalAttention {
                window: 1,
                rope_theta: 100.0,
            });
        });
        assert_cuda_matches_cpu(&rt, &model);
    }

    // LFM2's short conv, then attention: prefill, then a second sequence that
    // shares the first's blocks and resumes from the block-saved conv rows,
    // then a batched decode of both. Compares every step with the CPU.
    #[test]
    fn cuda_matches_cpu_short_conv_with_prefix_resume() {
        let Some(rt) = runtime() else { return };
        let model = model_with(
            vec![conv_layer(0, 3), attention_layer(0, false)],
            |config| {
                config.conv = Some(ConvLayers {
                    layers: vec![0],
                    kernel: 3,
                });
            },
        );
        let n_attention = model.config.n_attention_layers();
        assert_eq!(n_attention, 1);
        let cuda_model = CudaModel::upload(&rt, &model).unwrap();
        let mut kv_cpu = KvStorage::new(n_attention, 4, 2, 1, 2, &Device::Cpu).unwrap();
        let mut kv_cuda = CudaKvStorage::new(&rt, n_attention, 4, 2, 1, 2).unwrap();

        // Sequence 0 prefills positions 0..4 into blocks 0 and 1.
        let blocks_a = [BlockID(0), BlockID(1)];
        let (mut wa, mut wb, mut wa2, mut wb2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let cpu = model
            .forward(
                &[item_seq(0, &[1, 3, 2, 0], 0, &blocks_a, 2, &mut wa)],
                &mut kv_cpu,
            )
            .unwrap();
        let cuda = cuda_model
            .forward(
                &rt,
                &[item_seq(0, &[1, 3, 2, 0], 0, &blocks_a, 2, &mut wa)],
                &mut kv_cuda,
            )
            .unwrap();
        assert_logits_close(&cpu, &cuda, &rt, "prefill");

        // Sequence 1 shares blocks 0 and 1 (prefix cache) and resumes at 4 in
        // block 2: its conv state comes from block 1's saved rows.
        let blocks_b = [BlockID(0), BlockID(1), BlockID(2)];
        let cpu = model
            .forward(&[item_seq(1, &[4], 4, &blocks_b, 2, &mut wb)], &mut kv_cpu)
            .unwrap();
        let cuda = cuda_model
            .forward(
                &rt,
                &[item_seq(1, &[4], 4, &blocks_b, 2, &mut wb)],
                &mut kv_cuda,
            )
            .unwrap();
        assert_logits_close(&cpu, &cuda, &rt, "resumed prefill");

        // Batched decode: sequence 0 continues in block 3, sequence 1 in block 2.
        let blocks_a2 = [BlockID(0), BlockID(1), BlockID(3)];
        let cpu = model
            .forward(
                &[
                    item_seq(0, &[2], 4, &blocks_a2, 2, &mut wa2),
                    item_seq(1, &[0], 5, &blocks_b, 2, &mut wb2),
                ],
                &mut kv_cpu,
            )
            .unwrap();
        let cuda = cuda_model
            .forward(
                &rt,
                &[
                    item_seq(0, &[2], 4, &blocks_a2, 2, &mut wa2),
                    item_seq(1, &[0], 5, &blocks_b, 2, &mut wb2),
                ],
                &mut kv_cuda,
            )
            .unwrap();
        assert_logits_close(&cpu, &cuda, &rt, "batched decode");
    }
}
