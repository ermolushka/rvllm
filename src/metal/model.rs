// Metal mirror of cuda/model.rs (itself the mirror of model.rs's Layer::forward
// / Model::forward / forward_last): same math, same shapes, same kernel-per-op
// structure. Two differences from the CUDA version:
//  * the matmuls are the hand-written `ops::linear` / attention kernels instead
//    of cuBLAS, and they read F32 activations directly, so there is no F16
//    activation scratch (`to_f16`) to thread through the layers;
//  * nothing runs until the backend synchronizes: every call here only records
//    dispatches into the runtime's open command buffer.
// Covers every family the CPU path loads: Llama 3's RoPE scaling, Qwen2's
// attention biases, Qwen3's Q/K norm, Gemma 3's local attention, post-norms,
// GELU and embedding scale, and LFM2's short convolutions.
use kv_cache_scheduler::block_pool::BlockID;

use crate::conv_cache::ConvItem;
use crate::model::{Activation, BatchItem, Config, Model};

use super::context::{MetalBuf, MetalRuntime, Result};
use super::kv_storage::{MetalGatherPlan, MetalKvStorage};
use super::ops;
use super::weights::{MetalAttention, MetalLayer, MetalMixer, MetalShortConv, MetalWeights};

// The RoPE tables and attention mask one kind of attention layer uses: the
// global ones, or a Gemma local layer's (its own RoPE base, its window).
// cos/sin are pre-expanded across the head axis on the host since the rope
// kernel expects one cos/sin row per input row, and q/k have different head
// counts.
struct MetalAttnCtx {
    cos_q: MetalBuf<f32>,
    sin_q: MetalBuf<f32>,
    cos_k: MetalBuf<f32>,
    sin_k: MetalBuf<f32>,
    // [batch * q_len * ctx_len], broadcast across the heads.
    mask: MetalBuf<f32>,
}

// Everything that depends only on this call's items, not on the layer: built
// once per forward call and shared by every layer.
struct MetalStepCtx {
    global: MetalAttnCtx,
    // The local layers' tables, when the model has local layers.
    local: Option<MetalAttnCtx>,
    gather: MetalGatherPlan,
    block_ids: MetalBuf<u32>,
    offsets: MetalBuf<u32>,
    // Each item as the short-conv layers' cache sees it (empty without them).
    conv_items: Vec<ConvItem>,
    batch: usize,
    q_len: usize,
    ctx_len: usize,
}

// cos/sin for every (batch, q_len) row, not yet expanded across heads.
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

// Repeats each (batch, q_len) row's cos/sin across `n_heads` heads, in
// (batch, head, q_len) order: the layout q/k are in after `transpose_axes12`.
fn expand_across_heads(base: &[f32], batch: usize, q_len: usize, n_heads: usize, half: usize) -> Vec<f32> {
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

impl MetalAttnCtx {
    // `window` hides keys that many positions or more behind the query, for
    // sliding-window attention; the mask rule is model::batch_mask's.
    #[allow(clippy::too_many_arguments)]
    fn new(
        rt: &MetalRuntime,
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
        let cos_q = MetalBuf::from_slice(rt, &expand_across_heads(&cos_rows, batch, q_len, n_heads, half));
        let sin_q = MetalBuf::from_slice(rt, &expand_across_heads(&sin_rows, batch, q_len, n_heads, half));
        let cos_k = MetalBuf::from_slice(rt, &expand_across_heads(&cos_rows, batch, q_len, n_kv_heads, half));
        let sin_k = MetalBuf::from_slice(rt, &expand_across_heads(&sin_rows, batch, q_len, n_kv_heads, half));

        // Same masking rule as model::batch_mask: causal within the sequence's
        // real tokens, plus masked past its own read_num_tokens (the
        // batch-padding tail other sequences' longer contexts leave), and past
        // the window if there is one.
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
        Ok(MetalAttnCtx {
            cos_q,
            sin_q,
            cos_k,
            sin_k,
            mask: MetalBuf::from_slice(rt, &mask_data),
        })
    }
}

impl MetalStepCtx {
    fn new(rt: &MetalRuntime, config: &Config, items: &[BatchItem]) -> Result<Self> {
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
        let gather = MetalGatherPlan::new(rt, &gather_items)?;
        let ctx_len = gather.ctx_len() as usize;

        let global = MetalAttnCtx::new(
            rt,
            config,
            items,
            &positions,
            ctx_len,
            config.rope_theta,
            config.rope_factors.as_deref(),
            None,
        )?;
        let local = match config.local_attention {
            Some(local) => Some(MetalAttnCtx::new(
                rt,
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

        Ok(MetalStepCtx {
            global,
            local,
            gather,
            block_ids: MetalBuf::from_slice(rt, &block_ids),
            offsets: MetalBuf::from_slice(rt, &offsets),
            conv_items,
            batch,
            q_len,
            ctx_len,
        })
    }
}

// Every intermediate a layer needs, allocated once per forward pass (sizes
// depend on the batch and context length) and reused by all layers.
// Zero-initialized; k_full/v_full padding slots keep whatever finite values
// earlier layers left there, which the attention mask hides.
struct Workspace {
    normed: MetalBuf<f32>,
    q_proj: MetalBuf<f32>,
    k_proj: MetalBuf<f32>,
    v_proj: MetalBuf<f32>,
    // Q/K after their per-head RMS norm (Qwen3, LFM2), when they have one.
    q_normed: MetalBuf<f32>,
    k_normed: MetalBuf<f32>,
    q_t: MetalBuf<f32>,
    k_t: MetalBuf<f32>,
    v_t: MetalBuf<f32>,
    q_roped: MetalBuf<f32>,
    k_roped: MetalBuf<f32>,
    k_tok: MetalBuf<f32>,
    v_tok: MetalBuf<f32>,
    k_full: MetalBuf<f32>,
    v_full: MetalBuf<f32>,
    scores: MetalBuf<f32>,
    probs: MetalBuf<f32>,
    attn_out: MetalBuf<f32>,
    attn_out_t: MetalBuf<f32>,
    // The mixer's output (attention or short conv), projected back to hidden_dim.
    attn_proj: MetalBuf<f32>,
    // The mixer's output after its post-norm (Gemma), when it has one.
    attn_post: MetalBuf<f32>,
    h: MetalBuf<f32>,
    normed2: MetalBuf<f32>,
    gate: MetalBuf<f32>,
    up: MetalBuf<f32>,
    ffn_mid: MetalBuf<f32>,
    ffn_out: MetalBuf<f32>,
    // The FFN's output after its post-norm (Gemma), when it has one.
    ffn_post: MetalBuf<f32>,
    // Short-conv scratch: in_proj's output, B*x, the window of rows the filter
    // reads, and the gated filter output.
    conv_proj: MetalBuf<f32>,
    conv_bx: MetalBuf<f32>,
    conv_window: MetalBuf<f32>,
    conv_out: MetalBuf<f32>,
}

impl Workspace {
    fn new(rt: &MetalRuntime, config: &Config, ctx: &MetalStepCtx) -> Self {
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
        // A conv window of kernel - 1 + q_len rows per item; zero without conv.
        let kernel = config.conv.as_ref().map_or(1, |conv| conv.kernel);
        let window_rows = ctx.batch * (kernel - 1 + ctx.q_len);
        let z = |len: usize| MetalBuf::<f32>::zeros(rt, len);
        Workspace {
            normed: z(rows * hidden_dim),
            q_proj: z(q_size),
            k_proj: z(kv_size),
            v_proj: z(kv_size),
            q_normed: z(q_size),
            k_normed: z(kv_size),
            q_t: z(q_size),
            k_t: z(kv_size),
            v_t: z(kv_size),
            q_roped: z(q_size),
            k_roped: z(kv_size),
            k_tok: z(kv_size),
            v_tok: z(kv_size),
            k_full: z(full_size),
            v_full: z(full_size),
            scores: z(scores_size),
            probs: z(scores_size),
            attn_out: z(q_size),
            attn_out_t: z(q_size),
            attn_proj: z(rows * hidden_dim),
            attn_post: z(rows * hidden_dim),
            h: z(rows * hidden_dim),
            normed2: z(rows * hidden_dim),
            gate: z(rows * ffn_dim),
            up: z(rows * ffn_dim),
            ffn_mid: z(rows * ffn_dim),
            ffn_out: z(rows * hidden_dim),
            ffn_post: z(rows * hidden_dim),
            conv_proj: z(rows * 3 * hidden_dim),
            conv_bx: z(rows * hidden_dim),
            conv_window: z((window_rows * hidden_dim).max(1)),
            conv_out: z(rows * hidden_dim),
        }
    }
}

// The attention mixer: writes its output, projected to hidden_dim, into
// ws.attn_proj. `ws.normed` must hold the normalized input on entry.
fn attention_mixer(
    rt: &MetalRuntime,
    config: &Config,
    attn: &MetalAttention,
    ctx: &MetalStepCtx,
    kv_storage: &mut MetalKvStorage,
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
    ops::linear(rt, &ws.normed, &attn.q, rows, hidden_dim, n_heads * head_dim, &mut ws.q_proj)?;
    ops::linear(rt, &ws.normed, &attn.k, rows, hidden_dim, n_kv_heads * head_dim, &mut ws.k_proj)?;
    ops::linear(rt, &ws.normed, &attn.v, rows, hidden_dim, n_kv_heads * head_dim, &mut ws.v_proj)?;
    if let Some(bias) = &attn.q_bias {
        ops::add_bias_wrapper(rt, &mut ws.q_proj, bias)?;
    }
    if let Some(bias) = &attn.k_bias {
        ops::add_bias_wrapper(rt, &mut ws.k_proj, bias)?;
    }
    if let Some(bias) = &attn.v_bias {
        ops::add_bias_wrapper(rt, &mut ws.v_proj, bias)?;
    }

    // Per-head RMS norm on Q and K (each head is a head_dim-wide row), before
    // RoPE, as on the CPU.
    let q_src = match &attn.q_norm {
        Some(norm) => {
            ops::rmsnorm_wrapper(rt, &ws.q_proj, norm, head_dim as u32, eps, &mut ws.q_normed)?;
            &ws.q_normed
        }
        None => &ws.q_proj,
    };
    let k_src = match &attn.k_norm {
        Some(norm) => {
            ops::rmsnorm_wrapper(rt, &ws.k_proj, norm, head_dim as u32, eps, &mut ws.k_normed)?;
            &ws.k_normed
        }
        None => &ws.k_proj,
    };

    // [batch, q_len, heads, head_dim] -> [batch, heads, q_len, head_dim].
    ops::transpose_axes12_wrapper(rt, q_src, &mut ws.q_t, batch as u32, q_len as u32, n_heads as u32, head_dim as u32)?;
    ops::transpose_axes12_wrapper(rt, k_src, &mut ws.k_t, batch as u32, q_len as u32, n_kv_heads as u32, head_dim as u32)?;
    ops::transpose_axes12_wrapper(rt, &ws.v_proj, &mut ws.v_t, batch as u32, q_len as u32, n_kv_heads as u32, head_dim as u32)?;

    // Local layers rotate with their own RoPE base and see only their window.
    let actx = match &ctx.local {
        Some(local) if attn.local => local,
        _ => &ctx.global,
    };

    // RoPE rotates Q and K given position; V carries content, not position.
    ops::rope_wrapper(rt, &ws.q_t, &actx.cos_q, &actx.sin_q, &mut ws.q_roped, head_dim as u32)?;
    ops::rope_wrapper(rt, &ws.k_t, &actx.cos_k, &actx.sin_k, &mut ws.k_roped, head_dim as u32)?;

    // The cache wants token-major rows [batch * q_len, n_kv_heads, head_dim].
    ops::transpose_axes12_wrapper(rt, &ws.k_roped, &mut ws.k_tok, batch as u32, n_kv_heads as u32, q_len as u32, head_dim as u32)?;
    ops::transpose_axes12_wrapper(rt, &ws.v_t, &mut ws.v_tok, batch as u32, n_kv_heads as u32, q_len as u32, head_dim as u32)?;
    kv_storage.write_tokens(rt, attn.kv_index, &ctx.block_ids, &ctx.offsets, &ws.k_tok, &ws.v_tok)?;
    kv_storage.gather(rt, attn.kv_index, &ctx.gather, &mut ws.k_full, &mut ws.v_full)?;

    ops::attention_scores(
        rt,
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
        rt,
        &mut ws.scores,
        &actx.mask,
        n_kv_heads as u32,
        group_size as u32,
        q_len as u32,
        ctx_len as u32,
    )?;
    ops::softmax_wrapper(rt, &ws.scores, ctx_len as u32, &mut ws.probs)?;
    ops::attention_apply(
        rt,
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

    // [batch, heads, q_len, head_dim] -> [batch, q_len, heads, head_dim].
    ops::transpose_axes12_wrapper(rt, &ws.attn_out, &mut ws.attn_out_t, batch as u32, n_heads as u32, q_len as u32, head_dim as u32)?;
    ops::linear(rt, &ws.attn_out_t, &attn.o, rows, n_heads * head_dim, hidden_dim, &mut ws.attn_proj)
}

// LFM2's short-convolution mixer, the GPU side of model::ShortConv: writes its
// output, projected to hidden_dim, into ws.attn_proj. `ws.normed` must hold the
// normalized input on entry. The state each item needs comes from the conv
// cache in `kv_storage`, which is updated for the next call.
fn short_conv_mixer(
    rt: &MetalRuntime,
    config: &Config,
    conv: &MetalShortConv,
    ctx: &MetalStepCtx,
    kv_storage: &mut MetalKvStorage,
    ws: &mut Workspace,
) -> Result<()> {
    let hidden = config.hidden_dim as usize;
    let rows = ctx.batch * ctx.q_len;
    let q_len = ctx.q_len;
    let n = conv.kernel - 1;

    // in_proj: [rows, 3 * hidden] with B, C and x; B * x is what the filter sees.
    ops::linear(rt, &ws.normed, &conv.in_proj, rows, hidden, 3 * hidden, &mut ws.conv_proj)?;
    ops::short_conv_bx_wrapper(rt, &ws.conv_proj, &mut ws.conv_bx, hidden as u32)?;

    // Each item's window: the n rows before its first token (from the cache),
    // then its q_len new bx rows.
    let item_rows = (n + q_len) * hidden;
    for (b, item) in ctx.conv_items.iter().enumerate() {
        let before = kv_storage.conv.rows_before(rt, conv.conv_index, item, n, hidden)?;
        let base = b * item_rows;
        rt.copy_f32(&before, 0, &ws.conv_window, base, n * hidden)?;
        rt.copy_f32(&ws.conv_bx, b * q_len * hidden, &ws.conv_window, base + n * hidden, q_len * hidden)?;
    }
    kv_storage
        .conv
        .update(rt, conv.conv_index, &ctx.conv_items, &ws.conv_window, q_len, n, hidden)?;

    // Gate the filter's output with C, then project down to hidden_dim.
    ops::short_conv_filter_gate_wrapper(
        rt,
        &ws.conv_window,
        &conv.filter,
        &ws.conv_proj,
        &mut ws.conv_out,
        ctx.batch as u32,
        q_len as u32,
        hidden as u32,
        conv.kernel as u32,
    )?;
    ops::linear(rt, &ws.conv_out, &conv.out_proj, rows, hidden, hidden, &mut ws.attn_proj)
}

// One transformer block, same structure as model.rs's Layer::forward:
//   h   = x + post_attn(mixer(rms_norm(x, attn_norm)))
//   out = h + post_ffn(ffn(rms_norm(h, ffn_norm)))
// where mixer is attention or a short conv, the post norms exist for Gemma, and
// ffn is SwiGLU or GeGLU. x: [batch * q_len, hidden_dim] flattened.
#[allow(clippy::too_many_arguments)]
fn layer_forward(
    rt: &MetalRuntime,
    config: &Config,
    weights: &MetalLayer,
    ctx: &MetalStepCtx,
    kv_storage: &mut MetalKvStorage,
    x: &MetalBuf<f32>,
    out: &mut MetalBuf<f32>,
    ws: &mut Workspace,
) -> Result<()> {
    let hidden_dim = config.hidden_dim as usize;
    let ffn_dim = config.ffn_dim as usize;
    let rows = ctx.batch * ctx.q_len;
    let eps = config.rms_eps;

    ops::rmsnorm_wrapper(rt, x, &weights.attn_norm, hidden_dim as u32, eps, &mut ws.normed)?;
    match &weights.mixer {
        MetalMixer::Attention(attn) => attention_mixer(rt, config, attn, ctx, kv_storage, ws)?,
        MetalMixer::ShortConv(conv) => short_conv_mixer(rt, config, conv, ctx, kv_storage, ws)?,
    }
    if let Some(norm) = &weights.post_attn_norm {
        ops::rmsnorm_wrapper(rt, &ws.attn_proj, norm, hidden_dim as u32, eps, &mut ws.attn_post)?;
        std::mem::swap(&mut ws.attn_proj, &mut ws.attn_post);
    }

    ops::add_wrapper(rt, x, &ws.attn_proj, &mut ws.h)?;

    ops::rmsnorm_wrapper(rt, &ws.h, &weights.ffn_norm, hidden_dim as u32, eps, &mut ws.normed2)?;
    ops::linear(rt, &ws.normed2, &weights.ffn_gate, rows, hidden_dim, ffn_dim, &mut ws.gate)?;
    ops::linear(rt, &ws.normed2, &weights.ffn_up, rows, hidden_dim, ffn_dim, &mut ws.up)?;
    match config.activation {
        Activation::Silu => ops::silu_gate_multiply_wrapper(rt, &ws.gate, &ws.up, &mut ws.ffn_mid)?,
        Activation::GeluTanh => ops::gelu_gate_multiply_wrapper(rt, &ws.gate, &ws.up, &mut ws.ffn_mid)?,
    }
    ops::linear(rt, &ws.ffn_mid, &weights.ffn_down, rows, ffn_dim, hidden_dim, &mut ws.ffn_out)?;
    if let Some(norm) = &weights.post_ffn_norm {
        ops::rmsnorm_wrapper(rt, &ws.ffn_out, norm, hidden_dim as u32, eps, &mut ws.ffn_post)?;
        std::mem::swap(&mut ws.ffn_out, &mut ws.ffn_post);
    }

    ops::add_wrapper(rt, &ws.h, &ws.ffn_out, out)
}

pub struct MetalModel {
    pub config: Config,
    pub weights: MetalWeights,
}

impl MetalModel {
    pub fn upload(rt: &MetalRuntime, model: &Model) -> Result<Self> {
        Ok(MetalModel {
            config: model.config.clone(),
            weights: MetalWeights::upload(rt, model)?,
        })
    }

    // Embed -> N x layer_forward -> final hidden states, flat
    // [batch * q_len, hidden_dim], before the final norm and LM head.
    fn hidden(
        &self,
        rt: &MetalRuntime,
        items: &[BatchItem],
        kv_storage: &mut MetalKvStorage,
    ) -> Result<(MetalBuf<f32>, usize, usize)> {
        let batch = items.len();
        let q_len = items[0].tokens.len();
        let hidden_dim = self.config.hidden_dim as usize;
        let rows = batch * q_len;

        let flat_tokens: Vec<u32> = items
            .iter()
            .flat_map(|item| item.tokens.iter().copied())
            .collect();
        let tokens_dev = MetalBuf::from_slice(rt, &flat_tokens);
        let mut x = MetalBuf::<f32>::zeros(rt, rows * hidden_dim);
        ops::embed_lookup_wrapper(rt, &tokens_dev, &self.weights.token_embd, hidden_dim as u32, &mut x)?;
        if let Some(scale) = self.config.embedding_scale {
            ops::scale_inplace_wrapper(rt, &mut x, scale)?;
        }

        let step_ctx = MetalStepCtx::new(rt, &self.config, items)?;
        let mut ws = Workspace::new(rt, &self.config, &step_ctx);
        // Ping-pong between two hidden-state buffers: a layer reads `x` and
        // writes `x_next`, then they swap.
        let mut x_next = MetalBuf::<f32>::zeros(rt, rows * hidden_dim);
        for layer in self.weights.layers.iter() {
            layer_forward(rt, &self.config, layer, &step_ctx, kv_storage, &x, &mut x_next, &mut ws)?;
            std::mem::swap(&mut x, &mut x_next);
        }
        Ok((x, batch, q_len))
    }

    fn lm_head(&self, rt: &MetalRuntime, x: &MetalBuf<f32>, rows: usize) -> Result<MetalBuf<f32>> {
        let hidden_dim = self.config.hidden_dim as usize;
        let vocab_size = self.config.vocab_size as usize;
        let mut normed = MetalBuf::<f32>::zeros(rt, rows * hidden_dim);
        ops::rmsnorm_wrapper(
            rt,
            x,
            &self.weights.output_norm,
            hidden_dim as u32,
            self.config.rms_eps,
            &mut normed,
        )?;
        let mut logits = MetalBuf::<f32>::zeros(rt, rows * vocab_size);
        ops::linear(rt, &normed, &self.weights.output, rows, hidden_dim, vocab_size, &mut logits)?;
        Ok(logits)
    }

    // Full forward pass over a batch: logits, flat [batch * q_len, vocab_size].
    // Only records the work; the caller synchronizes before reading.
    pub fn forward(
        &self,
        rt: &MetalRuntime,
        items: &[BatchItem],
        kv_storage: &mut MetalKvStorage,
    ) -> Result<MetalBuf<f32>> {
        let (x, batch, q_len) = self.hidden(rt, items, kv_storage)?;
        self.lm_head(rt, &x, batch * q_len)
    }

    // Same, but only projects each item's last row: flat [batch, vocab_size].
    pub fn forward_last(
        &self,
        rt: &MetalRuntime,
        items: &[BatchItem],
        kv_storage: &mut MetalKvStorage,
    ) -> Result<MetalBuf<f32>> {
        let (x, batch, q_len) = self.hidden(rt, items, kv_storage)?;
        let hidden_dim = self.config.hidden_dim as usize;
        let mut last = MetalBuf::<f32>::zeros(rt, batch * hidden_dim);
        ops::narrow_last_row_wrapper(rt, &x, q_len as u32, hidden_dim as u32, &mut last)?;
        self.lm_head(rt, &last, batch)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use kv_cache_scheduler::sequence::SequenceId;

    use crate::kv_storage::KvStorage;
    use crate::model::{Attention, ConvLayers, Ffn, Layer, LocalAttention, Mixer, ShortConv};

    fn runtime() -> Option<MetalRuntime> {
        crate::metal::context::test_runtime()
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
    fn metal_forward_matches_cpu_forward() {
        let Some(rt) = runtime() else { return };
        let device = Device::Cpu;
        let model = tiny_model();
        let metal_model = MetalModel::upload(&rt, &model).unwrap();

        let tokens: [u32; 5] = [1, 3, 2, 0, 4];
        let blocks = [BlockID(0), BlockID(1), BlockID(2)];
        let mut writes = Vec::new();
        let items = [item(&tokens, 0, &blocks, 2, &mut writes)];

        let mut kv_cpu = KvStorage::new(1, 3, 2, 1, 2, &device).unwrap();
        let mut kv_metal = MetalKvStorage::new(&rt, 1, 3, 2, 1, 2).unwrap();

        let cpu_logits = model.forward_last(&items, &mut kv_cpu).unwrap();
        let cpu_flat: Vec<f32> = cpu_logits.flatten_all().unwrap().to_vec1().unwrap();

        let metal_logits = metal_model.forward_last(&rt, &items, &mut kv_metal).unwrap();
        let metal_flat = rt.read(&metal_logits).unwrap();

        assert_eq!(metal_flat.len(), cpu_flat.len());
        for (i, (&c, &g)) in cpu_flat.iter().zip(metal_flat.iter()).enumerate() {
            let tol = 1e-2 * c.abs().max(1.0);
            assert!((c - g).abs() < tol, "i={i}: cpu {c}, metal {g}");
        }
        let cpu_argmax = cpu_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let metal_argmax = metal_flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(cpu_argmax, metal_argmax, "greedy token mismatch");
    }

    #[test]
    fn metal_forward_matches_cpu_forward_batched_decode() {
        let Some(rt) = runtime() else { return };
        let device = Device::Cpu;
        let model = tiny_model();
        let metal_model = MetalModel::upload(&rt, &model).unwrap();

        let mut kv_cpu = KvStorage::new(1, 6, 2, 1, 2, &device).unwrap();
        let mut kv_metal = MetalKvStorage::new(&rt, 1, 6, 2, 1, 2).unwrap();
        let blocks_a = [BlockID(0), BlockID(1)];
        let blocks_b = [BlockID(2), BlockID(3), BlockID(4)];

        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        model
            .forward(&[item(&[0, 2, 1], 0, &blocks_a, 2, &mut wa)], &mut kv_cpu)
            .unwrap();
        metal_model
            .forward(
                &rt,
                &[item(&[0, 2, 1], 0, &blocks_a, 2, &mut wa)],
                &mut kv_metal,
            )
            .unwrap();
        model
            .forward(
                &[item(&[1, 3, 0, 2, 4, 1], 0, &blocks_b, 2, &mut wb)],
                &mut kv_cpu,
            )
            .unwrap();
        metal_model
            .forward(
                &rt,
                &[item(&[1, 3, 0, 2, 4, 1], 0, &blocks_b, 2, &mut wb)],
                &mut kv_metal,
            )
            .unwrap();

        let blocks_b_ext = [BlockID(2), BlockID(3), BlockID(4), BlockID(5)];
        let items = [
            item(&[4], 3, &blocks_a, 2, &mut wa),
            item(&[4], 6, &blocks_b_ext, 2, &mut wb),
        ];
        let cpu_logits = model.forward(&items, &mut kv_cpu).unwrap();
        let cpu_flat: Vec<f32> = cpu_logits.flatten_all().unwrap().to_vec1().unwrap();
        let metal_logits = metal_model.forward(&rt, &items, &mut kv_metal).unwrap();
        let metal_flat = rt.read(&metal_logits).unwrap();

        assert_eq!(metal_flat.len(), cpu_flat.len());
        for (i, (&c, &g)) in cpu_flat.iter().zip(metal_flat.iter()).enumerate() {
            let tol = 1e-2 * c.abs().max(1.0);
            assert!((c - g).abs() < tol, "i={i}: cpu {c}, metal {g}");
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

    fn assert_logits_close(cpu: &Tensor, metal: &MetalBuf<f32>, rt: &MetalRuntime, what: &str) {
        let cpu_flat: Vec<f32> = cpu.flatten_all().unwrap().to_vec1().unwrap();
        let metal_flat = rt.read(metal).unwrap();
        assert_eq!(metal_flat.len(), cpu_flat.len(), "{what}: logits length");
        for (i, (&c, &g)) in cpu_flat.iter().zip(metal_flat.iter()).enumerate() {
            let tol = 1e-2 * c.abs().max(1.0);
            assert!((c - g).abs() < tol, "{what} i={i}: cpu {c}, metal {g}");
        }
    }

    // Prefill three tokens, then decode one more, on the CPU and on CUDA, and
    // compare every row's logits. Covers every feature the model uses.
    fn assert_metal_matches_cpu(rt: &MetalRuntime, model: &Model) {
        let n_attention = model.config.n_attention_layers();
        let metal_model = MetalModel::upload(rt, model).unwrap();
        let mut kv_cpu = KvStorage::new(n_attention, 3, 2, 1, 2, &Device::Cpu).unwrap();
        let mut kv_metal = MetalKvStorage::new(rt, n_attention, 3, 2, 1, 2).unwrap();
        let blocks = [BlockID(0), BlockID(1)];

        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        let cpu = model
            .forward(&[item(&[1, 3, 2], 0, &blocks, 2, &mut wa)], &mut kv_cpu)
            .unwrap();
        let metal = metal_model
            .forward(
                rt,
                &[item(&[1, 3, 2], 0, &blocks, 2, &mut wa)],
                &mut kv_metal,
            )
            .unwrap();
        assert_logits_close(&cpu, &metal, rt, "prefill");

        let cpu = model
            .forward(&[item(&[4], 3, &blocks, 2, &mut wb)], &mut kv_cpu)
            .unwrap();
        let metal = metal_model
            .forward(rt, &[item(&[4], 3, &blocks, 2, &mut wb)], &mut kv_metal)
            .unwrap();
        assert_logits_close(&cpu, &metal, rt, "decode");
    }

    #[test]
    fn metal_matches_cpu_with_attention_biases() {
        let Some(rt) = runtime() else { return };
        let mut model = tiny_model();
        let attn = attention(&mut model);
        attn.q_bias = Some(rnd((4, 1)).flatten_all().unwrap());
        attn.k_bias = Some(rnd((2, 1)).flatten_all().unwrap());
        attn.v_bias = Some(rnd((2, 1)).flatten_all().unwrap());
        assert_metal_matches_cpu(&rt, &model);
    }

    #[test]
    fn metal_matches_cpu_with_llama3_rope_factors() {
        let Some(rt) = runtime() else { return };
        let mut model = tiny_model();
        model.config.rope_factors = Some(vec![8.0]);
        assert_metal_matches_cpu(&rt, &model);
    }

    #[test]
    fn metal_matches_cpu_with_qk_norm() {
        let Some(rt) = runtime() else { return };
        let mut model = tiny_model();
        let attn = attention(&mut model);
        attn.q_norm = Some(ones_vec(2));
        attn.k_norm = Some(ones_vec(2));
        assert_metal_matches_cpu(&rt, &model);
    }

    #[test]
    fn metal_matches_cpu_gemma_style_layers() {
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
        assert_metal_matches_cpu(&rt, &model);
    }

    // LFM2's short conv, then attention: prefill, then a second sequence that
    // shares the first's blocks and resumes from the block-saved conv rows,
    // then a batched decode of both. Compares every step with the CPU.
    #[test]
    fn metal_matches_cpu_short_conv_with_prefix_resume() {
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
        let metal_model = MetalModel::upload(&rt, &model).unwrap();
        let mut kv_cpu = KvStorage::new(n_attention, 4, 2, 1, 2, &Device::Cpu).unwrap();
        let mut kv_metal = MetalKvStorage::new(&rt, n_attention, 4, 2, 1, 2).unwrap();

        // Sequence 0 prefills positions 0..4 into blocks 0 and 1.
        let blocks_a = [BlockID(0), BlockID(1)];
        let (mut wa, mut wb, mut wa2, mut wb2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let cpu = model
            .forward(
                &[item_seq(0, &[1, 3, 2, 0], 0, &blocks_a, 2, &mut wa)],
                &mut kv_cpu,
            )
            .unwrap();
        let metal = metal_model
            .forward(
                &rt,
                &[item_seq(0, &[1, 3, 2, 0], 0, &blocks_a, 2, &mut wa)],
                &mut kv_metal,
            )
            .unwrap();
        assert_logits_close(&cpu, &metal, &rt, "prefill");

        // Sequence 1 shares blocks 0 and 1 (prefix cache) and resumes at 4 in
        // block 2: its conv state comes from block 1's saved rows.
        let blocks_b = [BlockID(0), BlockID(1), BlockID(2)];
        let cpu = model
            .forward(&[item_seq(1, &[4], 4, &blocks_b, 2, &mut wb)], &mut kv_cpu)
            .unwrap();
        let metal = metal_model
            .forward(
                &rt,
                &[item_seq(1, &[4], 4, &blocks_b, 2, &mut wb)],
                &mut kv_metal,
            )
            .unwrap();
        assert_logits_close(&cpu, &metal, &rt, "resumed prefill");

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
        let metal = metal_model
            .forward(
                &rt,
                &[
                    item_seq(0, &[2], 4, &blocks_a2, 2, &mut wa2),
                    item_seq(1, &[0], 5, &blocks_b, 2, &mut wb2),
                ],
                &mut kv_metal,
            )
            .unwrap();
        assert_logits_close(&cpu, &metal, &rt, "batched decode");
    }

    // Real-weights parity check, ignored by default (needs a GGUF):
    //   RVLLM_TEST_MODEL=../SmolLM2-360M.Q8_0.gguf cargo test --release \
    //     --features metal --lib real_model_prefill -- --ignored --nocapture
    // Prefills random prompts on the CPU and on Metal and compares the logits.
    // F16 weights on the GPU vs F32 on the CPU make the logits differ by a
    // little; what matters is that the greedy token only ever flips when the
    // CPU's own top-2 are nearly tied, which this prints so it can be judged.
    #[test]
    #[ignore]
    fn real_model_prefill_logits_match_cpu() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let Some(rt) = runtime() else { return };
        let Some(path) = std::env::var_os("RVLLM_TEST_MODEL") else {
            eprintln!("set RVLLM_TEST_MODEL to a .gguf to run this test");
            return;
        };
        let model = Model::load(path.to_str().unwrap()).unwrap();
        let metal_model = MetalModel::upload(&rt, &model).unwrap();
        let vocab = model.config.vocab_size;
        let (block_size, prompt_len) = (16usize, 128usize);
        let n_blocks = prompt_len.div_ceil(block_size);
        let blocks: Vec<BlockID> = (0..n_blocks as u32).map(BlockID).collect();
        // RVLLM_TEST_BENCH_SEQ=N replays `bench`'s RNG (seed 42, 128-token
        // prompts drawn one after another) up to its N-th request, to inspect
        // that exact prompt.
        let bench_seq: Option<usize> = std::env::var("RVLLM_TEST_BENCH_SEQ").ok().and_then(|v| v.parse().ok());
        let mut rng = StdRng::seed_from_u64(if bench_seq.is_some() { 42 } else { 7 });
        if let Some(n) = bench_seq {
            for _ in 0..n * prompt_len {
                let _ = rng.random_range(0..vocab);
            }
        }
        let (mut flips, mut worst_gap_of_flip) = (0, f32::MAX);
        let mut max_abs_diff = 0f32;
        let trials = if bench_seq.is_some() { 1 } else { 16 };
        for _ in 0..trials {
            let tokens: Vec<u32> = (0..prompt_len).map(|_| rng.random_range(0..vocab)).collect();
            let mut writes = Vec::new();
            let items = [item(&tokens, 0, &blocks, block_size, &mut writes)];
            let n_attn = model.config.n_attention_layers();
            let mut kv_cpu = KvStorage::new(
                n_attn, n_blocks, block_size, model.config.n_kv_heads as usize, model.config.head_dim(), &Device::Cpu,
            )
            .unwrap();
            let mut kv_metal = MetalKvStorage::new(
                &rt, n_attn, n_blocks, block_size, model.config.n_kv_heads as usize, model.config.head_dim(),
            )
            .unwrap();
            let cpu: Vec<f32> = model
                .forward_last(&items, &mut kv_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let gpu_buf = metal_model.forward_last(&rt, &items, &mut kv_metal).unwrap();
            let gpu = rt.read(&gpu_buf).unwrap();
            let top2 = |v: &[f32]| {
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
                (idx[0], v[idx[0]] - v[idx[1]])
            };
            let (cpu_top, cpu_gap) = top2(&cpu);
            let (gpu_top, _) = top2(&gpu);
            max_abs_diff = max_abs_diff.max(cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
            if cpu_top != gpu_top {
                flips += 1;
                worst_gap_of_flip = worst_gap_of_flip.min(cpu_gap);
            }
            println!("cpu top1 {cpu_top} (gap to #2: {cpu_gap:.4}), metal top1 {gpu_top}");
        }
        println!("max |logit diff| {max_abs_diff:.4}; {flips}/{trials} argmax flips (smallest CPU top-2 gap among flips: {worst_gap_of_flip})");
        // A flip is only acceptable when the CPU itself was nearly undecided.
        assert!(flips == 0 || worst_gap_of_flip < 0.1, "argmax flipped despite a clear CPU winner");
    }
}
