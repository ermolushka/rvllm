// CUDA mirror of model.rs's Layer::forward/Model::forward/forward_last.
// Same math, same shapes, but every Candle reshape/transpose that was free
// on CPU becomes an explicit kernel call here (transpose_axes12), and every
// matmul goes through cuBLAS (cublas_ops::linear/attention_scores/
// attention_apply) instead of Candle's matmul. No shared trait with the CPU
// Model - same reasoning as CudaKvStorage/CudaWeights: the CPU path is
// Candle-Tensor-typed end to end, this one is CudaSlice-typed end to end.
use cudarc::driver::CudaSlice;
use half::f16;
use kv_cache_scheduler::block_pool::BlockID;

use crate::model::{BatchItem, Config, Model};

use super::context::CudaRuntime;
use super::cublas_ops;
use super::kv_storage::{CudaGatherPlan, CudaKvStorage};
use super::ops;
use super::weights::{CudaLayer, CudaWeights};

// Everything that depends only on this call's items, not on the layer -
// same role as model.rs's StepCtx, built once per forward call and shared
// by every layer. RoPE cos/sin are pre-expanded across the head axis here
// (on the host, cheap) since the CUDA rope kernel expects one cos/sin row
// per input row, and q/k have different head counts.
struct CudaStepCtx {
    cos_q: CudaSlice<f32>,
    sin_q: CudaSlice<f32>,
    cos_k: CudaSlice<f32>,
    sin_k: CudaSlice<f32>,
    // [batch * q_len * ctx_len] flattened, broadcast across n_kv_heads/group_size.
    mask: CudaSlice<f32>,
    gather: CudaGatherPlan,
    block_ids: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    batch: usize,
    q_len: usize,
    ctx_len: usize,
}

// cos/sin for every (batch, q_len) row, not yet expanded across heads -
// same values model::rope_cos_sin computes, plain Rust instead of Candle.
fn rope_cos_sin_rows(
    rope_theta: f32,
    head_dim: usize,
    positions: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let theta: Vec<f32> = (0..half)
        .map(|j| rope_theta.powf(-2.0 * j as f32 / head_dim as f32))
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

impl CudaStepCtx {
    fn new(
        cuda_runtime: &CudaRuntime,
        config: &Config,
        items: &[BatchItem],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let batch = items.len();
        let q_len = items[0].tokens.len();
        let head_dim = config.head_dim();
        let half = head_dim / 2;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;

        let positions: Vec<usize> = items
            .iter()
            .flat_map(|item| item.position_offset..item.position_offset + q_len)
            .collect();
        let (cos_rows, sin_rows) = rope_cos_sin_rows(config.rope_theta, head_dim, &positions);
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

        let gather_items: Vec<(&[BlockID], usize)> = items
            .iter()
            .map(|item| (item.read_blocks, item.read_num_tokens))
            .collect();
        let gather = CudaGatherPlan::new(cuda_runtime, &gather_items)?;
        let ctx_len = gather.ctx_len() as usize;

        // Same masking rule as model::batch_mask: causal within the
        // sequence's real tokens, plus masked past its own read_num_tokens
        // (the batch-padding tail other sequences' longer contexts leave).
        let mut mask_data = vec![0f32; batch * q_len * ctx_len];
        for (b, item) in items.iter().enumerate() {
            let offset = item.position_offset;
            let valid_len = item.read_num_tokens;
            for i in 0..q_len {
                for j in 0..ctx_len {
                    if j >= valid_len || j > offset + i {
                        mask_data[(b * q_len + i) * ctx_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mask = cuda_runtime.stream.clone_htod(&mask_data)?;

        let writes: Vec<(BlockID, usize)> = items
            .iter()
            .flat_map(|item| item.write_positions.iter().copied())
            .collect();
        let block_ids: Vec<u32> = writes.iter().map(|&(b, _)| b.0).collect();
        let offsets: Vec<u32> = writes.iter().map(|&(_, o)| o as u32).collect();
        let block_ids = cuda_runtime.stream.clone_htod(&block_ids)?;
        let offsets = cuda_runtime.stream.clone_htod(&offsets)?;

        Ok(CudaStepCtx {
            cos_q,
            sin_q,
            cos_k,
            sin_k,
            mask,
            gather,
            block_ids,
            offsets,
            batch,
            q_len,
            ctx_len,
        })
    }
}

// One Llama transformer block, same structure as model.rs's Layer::forward:
//   h   = x + attention(rms_norm(x, attn_norm))
//   out = h + swiglu_ffn(rms_norm(h, ffn_norm))
// x: [batch * q_len, hidden_dim] flattened; returns the same shape.
// Every intermediate a layer needs, allocated once per forward pass (sizes
// depend on the batch and context length, so not once per model) and reused
// by all layers, instead of ~25 fresh allocations per layer. Zero-initialized
// once; k_full/v_full padding slots keep whatever finite values earlier
// layers left there, which the attention mask hides.
struct Workspace {
    normed: CudaSlice<f32>,
    q_proj: CudaSlice<f32>,
    k_proj: CudaSlice<f32>,
    v_proj: CudaSlice<f32>,
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
    attn_proj: CudaSlice<f32>,
    h: CudaSlice<f32>,
    normed2: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    ffn_mid: CudaSlice<f32>,
    ffn_out: CudaSlice<f32>,
    // F16 copy of the current linear input, sized for the widest one.
    x16: CudaSlice<f16>,
}

impl Workspace {
    fn new(
        cuda_runtime: &CudaRuntime,
        config: &Config,
        ctx: &CudaStepCtx,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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
        let widest = hidden_dim.max(n_heads * head_dim).max(ffn_dim);
        let stream = &cuda_runtime.stream;
        Ok(Workspace {
            normed: stream.alloc_zeros(rows * hidden_dim)?,
            q_proj: stream.alloc_zeros(q_size)?,
            k_proj: stream.alloc_zeros(kv_size)?,
            v_proj: stream.alloc_zeros(kv_size)?,
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
            h: stream.alloc_zeros(rows * hidden_dim)?,
            normed2: stream.alloc_zeros(rows * hidden_dim)?,
            gate: stream.alloc_zeros(rows * ffn_dim)?,
            up: stream.alloc_zeros(rows * ffn_dim)?,
            ffn_mid: stream.alloc_zeros(rows * ffn_dim)?,
            ffn_out: stream.alloc_zeros(rows * hidden_dim)?,
            x16: stream.alloc_zeros(rows * widest)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn layer_forward(
    cuda_runtime: &CudaRuntime,
    config: &Config,
    layer_idx: usize,
    weights: &CudaLayer,
    ctx: &CudaStepCtx,
    kv_storage: &mut CudaKvStorage,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    ws: &mut Workspace,
) -> Result<(), Box<dyn std::error::Error>> {
    let hidden_dim = config.hidden_dim as usize;
    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let head_dim = config.head_dim();
    let group_size = n_heads / n_kv_heads;
    let ffn_dim = config.ffn_dim as usize;
    let batch = ctx.batch;
    let q_len = ctx.q_len;
    let ctx_len = ctx.ctx_len;
    let rows = batch * q_len;
    let x16 = &mut ws.x16;

    ops::rmsnorm_wrapper(
        cuda_runtime,
        x,
        &weights.attn_norm,
        hidden_dim as u32,
        config.rms_eps,
        &mut ws.normed,
    )?;

    // Project into Q/K/V, naturally [batch, q_len, heads, head_dim] flat
    // (reshape is free - a plain GEMM output, no data movement needed).
    // y = x @ weight^T. q/k/v read the same input, so narrow it to F16 once.
    cublas_ops::to_f16(cuda_runtime, &ws.normed, x16)?;
    cublas_ops::linear_f16(
        cuda_runtime,
        x16,
        &weights.attn_q,
        rows,
        hidden_dim,
        n_heads * head_dim,
        &mut ws.q_proj,
    )?;
    cublas_ops::linear_f16(
        cuda_runtime,
        x16,
        &weights.attn_k,
        rows,
        hidden_dim,
        n_kv_heads * head_dim,
        &mut ws.k_proj,
    )?;
    cublas_ops::linear_f16(
        cuda_runtime,
        x16,
        &weights.attn_v,
        rows,
        hidden_dim,
        n_kv_heads * head_dim,
        &mut ws.v_proj,
    )?;

    // [batch, q_len, heads, head_dim] -> [batch, heads, q_len, head_dim] -
    // real data movement here, unlike Candle's free `.transpose(1, 2)`.
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        &ws.q_proj,
        &mut ws.q_t,
        batch as u32,
        q_len as u32,
        n_heads as u32,
        head_dim as u32,
    )?;
    ops::transpose_axes12_wrapper(
        cuda_runtime,
        &ws.k_proj,
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

    // RoPE rotates Q and K given position; V carries content, not position.
    ops::rope_wrapper(
        cuda_runtime,
        &ws.q_t,
        &ctx.cos_q,
        &ctx.sin_q,
        &mut ws.q_roped,
        head_dim as u32,
    )?;
    ops::rope_wrapper(
        cuda_runtime,
        &ws.k_t,
        &ctx.cos_k,
        &ctx.sin_k,
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
        layer_idx,
        &ctx.block_ids,
        &ctx.offsets,
        &ws.k_tok,
        &ws.v_tok,
    )?;

    kv_storage.gather(
        cuda_runtime,
        layer_idx,
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
        &ctx.mask,
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
        &weights.attn_output,
        rows,
        n_heads * head_dim,
        hidden_dim,
        &mut ws.attn_proj,
        x16,
    )?;

    ops::add_wrapper(cuda_runtime, x, &ws.attn_proj, &mut ws.h)?;

    ops::rmsnorm_wrapper(
        cuda_runtime,
        &ws.h,
        &weights.ffn_norm,
        hidden_dim as u32,
        config.rms_eps,
        &mut ws.normed2,
    )?;
    cublas_ops::to_f16(cuda_runtime, &ws.normed2, x16)?;
    cublas_ops::linear_f16(
        cuda_runtime,
        x16,
        &weights.ffn_gate,
        rows,
        hidden_dim,
        ffn_dim,
        &mut ws.gate,
    )?;
    cublas_ops::linear_f16(
        cuda_runtime,
        x16,
        &weights.ffn_up,
        rows,
        hidden_dim,
        ffn_dim,
        &mut ws.up,
    )?;
    ops::silu_gate_multiply_wrapper(cuda_runtime, &ws.gate, &ws.up, &mut ws.ffn_mid)?;
    cublas_ops::linear(
        cuda_runtime,
        &ws.ffn_mid,
        &weights.ffn_down,
        rows,
        ffn_dim,
        hidden_dim,
        &mut ws.ffn_out,
        x16,
    )?;

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
    ) -> Result<Self, Box<dyn std::error::Error>> {
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
    ) -> Result<(CudaSlice<f32>, usize, usize), Box<dyn std::error::Error>> {
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

        let step_ctx = CudaStepCtx::new(cuda_runtime, &self.config, items)?;
        let mut ws = Workspace::new(cuda_runtime, &self.config, &step_ctx)?;
        // Ping-pong between two hidden-state buffers: a layer reads `x` and
        // writes `x_next`, then they swap.
        let mut x_next = cuda_runtime.stream.alloc_zeros::<f32>(rows * hidden_dim)?;
        for (i, layer) in self.weights.layers.iter().enumerate() {
            layer_forward(
                cuda_runtime,
                &self.config,
                i,
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
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
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
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (x, batch, q_len) = self.hidden(cuda_runtime, items, kv_storage)?;
        self.lm_head(cuda_runtime, &x, batch * q_len)
    }

    // Same, but only projects each item's last row: flat [batch, vocab_size].
    pub fn forward_last(
        &self,
        cuda_runtime: &CudaRuntime,
        items: &[BatchItem],
        kv_storage: &mut CudaKvStorage,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
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

    use crate::kv_storage::KvStorage;
    use crate::model::Layer;

    fn runtime() -> Option<CudaRuntime> {
        CudaRuntime::new(0).ok()
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
            index: 0,
            attn_norm: ones(),
            attn_q: rand((n_heads * head_dim, hidden_dim)),
            attn_k: rand((n_kv_heads * head_dim, hidden_dim)),
            attn_v: rand((n_kv_heads * head_dim, hidden_dim)),
            attn_output: rand((hidden_dim, n_heads * head_dim)),
            ffn_norm: ones(),
            ffn_gate: rand((ffn_dim, hidden_dim)),
            ffn_up: rand((ffn_dim, hidden_dim)),
            ffn_down: rand((hidden_dim, ffn_dim)),
        };
        Model {
            config: Config {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                hidden_dim: hidden_dim as u32,
                ffn_dim: ffn_dim as u32,
                rope_theta: 10000.0,
                rms_eps: 1e-5,
                vocab_size: vocab_size as u32,
                context_length: 8,
                eos_token_id: 99,
                bos_token_id: 0,
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
}
