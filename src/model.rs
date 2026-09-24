use candle_core::{D, Device, Result, Tensor};
use candle_nn::ops::{silu, softmax};
use kv_cache_scheduler::block_pool::BlockID;

use crate::kv_storage::{GatherPlan, KvStorage};

// Model hyperparameters, read from the GGUF metadata by `Model::load`.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub hidden_dim: u32,
    pub ffn_dim: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        (self.hidden_dim / self.n_heads) as usize
    }
}

// One transformer block's weights, all F32 in candle Linear convention
// ([out_features, in_features]) except the two norm vectors.
pub struct Layer {
    // Index of this layer's slot in `KvStorage`.
    pub index: usize,
    pub attn_norm: Tensor,
    pub attn_q: Tensor,
    pub attn_k: Tensor,
    pub attn_v: Tensor,
    pub attn_output: Tensor,
    pub ffn_norm: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
}

// The whole model. `Model::load` (in weights.rs) builds one from a GGUF file.
pub struct Model {
    pub config: Config,
    // [vocab_size, hidden_dim]
    pub token_embd: Tensor,
    pub layers: Vec<Layer>,
    pub output_norm: Tensor,
    // [vocab_size, hidden_dim]. Shares storage with `token_embd` when the
    // model ties its input and output embeddings.
    pub output: Tensor,
}

// RMSNorm: rescale each row by its root-mean-square, then apply a learned
// per-channel scale. No mean-centering (unlike LayerNorm) - just magnitude
// control, which is cheaper and works fine for Llama-family models.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // rms = sqrt(mean(x^2) + eps), one value per row (kept as a column so it
    // broadcasts back against every element of that row). Normalizing over the
    // last dim (rather than a hardcoded 1) is what lets this same function run
    // unchanged on batched [batch, seq_len, hidden_dim] input.
    let last_dim = x.rank() - 1;
    let rms = x
        .powf(2.0)?
        .mean_keepdim(last_dim)?
        .affine(1.0, eps as f64)?
        .sqrt()?;
    // Divide each row by its own rms, then scale each channel by `weight`.
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(weight)
}

// RoPE cos/sin for `positions`, each [positions.len(), head_dim / 2]. Each
// dimension-pair rotates at its own frequency - pair 0 fastest, later pairs
// slower - so different pairs encode position at different resolutions, like
// clock hands moving at different speeds. Depends only on positions, so it is
// built once per forward call and shared by every layer and head.
pub fn rope_cos_sin(
    rope_theta: f32,
    head_dim: usize,
    positions: &[usize],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let theta: Vec<f32> = (0..half)
        .map(|j| rope_theta.powf(-2.0 * j as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::from_vec(theta, (1, half), device)?;

    let positions_f: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let positions = Tensor::from_vec(positions_f, (positions.len(), 1), device)?;

    // angle[pos, j] = pos * theta_j - how far pair j has rotated by this position.
    let angles = positions.broadcast_mul(&theta)?;
    Ok((angles.cos()?, angles.sin()?))
}

// Applies RoPE (rotate-half convention) to `x` of shape [..., n, head_dim],
// given cos/sin that broadcast against [..., n, head_dim / 2]. Because the
// rotation is elementwise over the last dim, one call covers every head (and
// every batch item) at once.
pub fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    // Rotate-half pairing: dim i pairs with dim i + half, not its neighbor.
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;

    // Standard 2D rotation applied to each (first, second) pair.
    let first_rot = first.broadcast_mul(cos)?.sub(&second.broadcast_mul(sin)?)?;
    let second_rot = first.broadcast_mul(sin)?.add(&second.broadcast_mul(cos)?)?;

    Tensor::cat(&[&first_rot, &second_rot], D::Minus1)
}

// Applies RoPE to `x` of shape [n, head_dim], one row per position given in
// `positions` (same order as `x`'s rows).
pub fn rope(x: &Tensor, rope_theta: f32, device: &Device, positions: &[usize]) -> Result<Tensor> {
    let (_, head_dim) = x.dims2()?;
    let (cos, sin) = rope_cos_sin(rope_theta, head_dim, positions, device)?;
    rope_apply(x, &cos, &sin)
}

// Looks up embedding rows for `token_ids` from `embedding_weight` ([vocab_size,
// hidden_dim]), producing a [seq_len, hidden_dim] tensor.
pub fn embed(token_ids: &[u32], embedding_weight: &Tensor, device: &Device) -> Result<Tensor> {
    // Wrap the ids so we can use them as row indices.
    let tokens_tensor = Tensor::from_slice(token_ids, token_ids.len(), device)?;
    // Pull out row `token_ids[i]` for each i - that row is that token's embedding.
    embedding_weight.index_select(&tokens_tensor, 0)
}

// y = x @ weight^T for x [..., in_features] and weight [out_features,
// in_features] (candle Linear convention). Leading dims are flattened into one
// 2D matmul, which candle runs on the transposed weight view directly. Do not
// use `broadcast_matmul` here: it calls `.contiguous()` on the broadcast
// weight, materializing a full transposed copy of it on every call, which
// made decode ~96% memcpy.
pub fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let in_features = x.dim(D::Minus1)?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = weight.dim(0)?;
    x.reshape((rows, in_features))?
        .matmul(&weight.t()?)?
        .reshape(out_dims)
}

// SwiGLU FFN: down(silu(gate(x)) * up(x)).
// x: [..., hidden_dim]
// gate_weight, up_weight: [ffn_dim, hidden_dim] (candle Linear convention: out_features x in_features)
// down_weight: [hidden_dim, ffn_dim]
pub fn swiglu_ffn(
    x: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Result<Tensor> {
    // Two parallel projections of the same input...
    let gate = linear(x, gate_weight)?;
    let up = linear(x, up_weight)?;
    // ...gate gets squashed through silu (a smooth "soft gate" nonlinearity)...
    let silu_gate = silu(&gate)?;
    // ...then used to scale up, elementwise - this is the "gating" in SwiGLU.
    let result = silu_gate.mul(&up)?;
    // Project back down to hidden_dim.
    linear(&result, down_weight)
}

// Per-batch-item attention mask [batch, 1, q_len, ctx_len]: causal within the
// sequence's real tokens (a token at position i must not see future tokens
// j > i, since at generation time those tokens don't exist yet), and
// additionally masked past that sequence's own `read_num_tokens` - that's the
// batch-padding tail every other sequence's longer context leaves behind for
// this one. Masked positions get -inf so softmax assigns them exactly 0.
// position_offset[b]: position of item b's first query row in its sequence.
pub fn batch_mask(
    position_offset: &[usize],
    read_num_tokens: &[usize],
    q_len: usize,
    ctx_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let batch = position_offset.len();
    let mut mask_data = vec![0f32; batch * q_len * ctx_len];
    for b in 0..batch {
        let offset = position_offset[b];
        let valid_len = read_num_tokens[b];
        for i in 0..q_len {
            for j in 0..ctx_len {
                if j >= valid_len || j > offset + i {
                    mask_data[(b * q_len + i) * ctx_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(mask_data, (batch, 1, q_len, ctx_len), device)
}

// Grouped-query attention over a batch, with a caller-supplied mask.
// q: [batch, n_heads, q_len, head_dim]        (RoPE already applied)
// k, v: [batch, n_kv_heads, ctx_len, head_dim] (k RoPE'd; right-padded past
//   each sequence's real length, which `mask` rules out)
// mask: [batch, 1, q_len, ctx_len], from `batch_mask`
// returns: [batch, n_heads, q_len, head_dim]
pub fn gqa_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, n_heads, q_len, head_dim) = q.dims4()?;
    let (_, n_kv_heads, ctx_len, _) = k.dims4()?;

    // GQA shares each KV head across `group_size` consecutive Q heads. Rather
    // than duplicating K and V group_size times to line up with every Q head
    // (a full copy of the context per layer), fold the group into the query
    // rows: Q heads [j*group_size .. (j+1)*group_size) all become extra query
    // rows against KV head j, so K and V are read as-is.
    let group_size = n_heads / n_kv_heads;
    let q = q
        .contiguous()?
        .reshape((batch, n_kv_heads, group_size * q_len, head_dim))?;

    // Dot product of every query against every key gives a raw "how similar
    // are these two vectors" score per (query position, key position) pair.
    let scores = q.matmul(&k.transpose(2, 3)?)?;
    // Scaling by 1/sqrt(head_dim) keeps these scores from growing huge as
    // head_dim increases, which would otherwise push softmax into saturating
    // (near one-hot) outputs regardless of how similar the vectors really are.
    let scores = scores.affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;

    // Unfold the group to add the mask, which is the same for every head.
    let scores = scores
        .reshape((batch, n_kv_heads, group_size, q_len, ctx_len))?
        .broadcast_add(&mask.unsqueeze(1)?)?;

    // Softmax turns each query's row of scores into a probability
    // distribution over key positions - "how much attention to pay to each
    // position," summing to 1.
    let probs = softmax(&scores, D::Minus1)?;
    // Using those weights to blend the V vectors gives, for each query
    // position, a weighted average of "what every position it attends to is
    // saying" - this is attention's actual output.
    probs
        .reshape((batch, n_kv_heads, group_size * q_len, ctx_len))?
        .matmul(v)?
        .reshape((batch, n_heads, q_len, head_dim))
}

// One sequence's slice of a forward call: everything the model needs to know
// about that sequence's tokens and cache bookkeeping. Prefill is a batch of
// one item with many tokens; a decode step is one item per running sequence,
// each with a single token. `tokens.len()` must be equal across the batch
// (the query axis is not padded).
pub struct BatchItem<'a> {
    // This call's new tokens for the sequence.
    pub tokens: &'a [u32],
    // (block, slot) each new token's K/V is written to, one per token.
    pub write_positions: &'a [(BlockID, usize)],
    // The sequence's full block table.
    pub read_blocks: &'a [BlockID],
    // Real (unpadded) context length after this call's tokens are written.
    pub read_num_tokens: usize,
    // Position of the first new token within the sequence (drives RoPE and
    // the causal mask).
    pub position_offset: usize,
}

// Everything that depends only on this call's items, not on the layer: RoPE
// cos/sin, the attention mask, the cache read plan and the flattened cache
// write positions. Built once per call and shared by every layer instead of
// being rebuilt per layer.
pub struct StepCtx {
    // [batch, 1, q_len, head_dim / 2]
    cos: Tensor,
    sin: Tensor,
    // [batch, 1, q_len, ctx_len]
    mask: Tensor,
    gather: GatherPlan,
    // Every item's write positions, concatenated in (batch, q_len) row-major
    // order to match the flattened K/V rows.
    writes: Vec<(BlockID, usize)>,
}

impl StepCtx {
    pub fn new(config: &Config, items: &[BatchItem], device: &Device) -> Result<Self> {
        let q_len = items[0].tokens.len();
        debug_assert!(
            items.iter().all(|item| item.tokens.len() == q_len),
            "every item must contribute the same number of new tokens"
        );
        let head_dim = config.head_dim();

        // Every item is at its own position, so build per-row positions in
        // (batch, q_len) row-major order, then fold back to broadcast shape.
        let positions: Vec<usize> = items
            .iter()
            .flat_map(|item| item.position_offset..item.position_offset + q_len)
            .collect();
        let (cos, sin) = rope_cos_sin(config.rope_theta, head_dim, &positions, device)?;
        let rope_shape = (items.len(), 1, q_len, head_dim / 2);
        let cos = cos.reshape(rope_shape)?;
        let sin = sin.reshape(rope_shape)?;

        let gather_items: Vec<(&[BlockID], usize)> = items
            .iter()
            .map(|item| (item.read_blocks, item.read_num_tokens))
            .collect();
        let gather = GatherPlan::new(&gather_items, device)?;

        let position_offsets: Vec<usize> = items.iter().map(|i| i.position_offset).collect();
        let read_num_tokens: Vec<usize> = items.iter().map(|i| i.read_num_tokens).collect();
        let mask = batch_mask(&position_offsets, &read_num_tokens, q_len, gather.ctx_len(), device)?;

        let writes = items
            .iter()
            .flat_map(|item| item.write_positions.iter().copied())
            .collect();
        Ok(StepCtx { cos, sin, mask, gather, writes })
    }
}

impl Layer {
    // One Llama transformer block:
    //   h   = x + attention(rms_norm(x, attn_norm)) // aka residual
    //   out = h + swiglu_ffn(rms_norm(h, ffn_norm))
    // x: [batch, q_len, hidden_dim]; returns the same shape.
    pub fn forward(
        &self,
        x: &Tensor,
        config: &Config,
        ctx: &StepCtx,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor> {
        let (batch, q_len, _) = x.dims3()?;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let head_dim = config.head_dim();

        // Normalize before attention (pre-norm), not after - this is the
        // convention Llama uses, and it's what keeps deep stacks of these
        // blocks stable to train/run.
        let normed = rms_norm(x, &self.attn_norm, config.rms_eps)?;

        // Project into query/key/value space. Q gets n_heads worth of output,
        // K/V get fewer (n_kv_heads) since GQA shares them across several Q
        // heads. Then split the projected dim into per-head chunks and move
        // the head axis in front of the sequence axis:
        // [batch, q_len, heads * head_dim] -> [batch, heads, q_len, head_dim].
        let q = linear(&normed, &self.attn_q)?
            .reshape((batch, q_len, n_heads, head_dim))?
            .transpose(1, 2)?;
        let k = linear(&normed, &self.attn_k)?
            .reshape((batch, q_len, n_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let v = linear(&normed, &self.attn_v)?
            .reshape((batch, q_len, n_kv_heads, head_dim))?
            .transpose(1, 2)?;

        // RoPE rotates every head identically given the position, so one call
        // per tensor covers all heads. Only Q and K get rotated - V carries
        // content, not position.
        let q = rope_apply(&q, &ctx.cos, &ctx.sin)?;
        let k = rope_apply(&k, &ctx.cos, &ctx.sin)?;

        // Write this call's tokens into their block slots. The cache wants
        // token-major rows: [batch * q_len, n_kv_heads, head_dim].
        kv_storage.write_tokens(
            self.index,
            &ctx.writes,
            &k.transpose(1, 2)?.reshape((batch * q_len, n_kv_heads, head_dim))?,
            &v.transpose(1, 2)?.reshape((batch * q_len, n_kv_heads, head_dim))?,
        )?;
        // Read back every sequence's full cached K/V (padded to the batch's
        // longest context) and attend against it.
        let (k_full, v_full) = kv_storage.gather(self.index, &ctx.gather)?;
        let attn_out = gqa_attention(&q, &k_full, &v_full, &ctx.mask)?;

        // Merge the heads back into one dimension - inverse of the earlier
        // split - and project down to hidden_dim so it can be added to the
        // residual stream.
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((batch, q_len, n_heads * head_dim))?;
        let attn_out = linear(&attn_out, &self.attn_output)?;

        // Residual connection: the block adds *to* x rather than replacing it,
        // so gradients (and, at inference time, information) can flow straight
        // through even if attention contributes very little.
        let h = x.add(&attn_out)?;

        // Same pre-norm pattern before the FFN half of the block.
        let normed2 = rms_norm(&h, &self.ffn_norm, config.rms_eps)?;
        let ffn_out = swiglu_ffn(&normed2, &self.ffn_gate, &self.ffn_up, &self.ffn_down)?;

        // Second residual connection - this is the block's output.
        h.add(&ffn_out)
    }
}

impl Model {
    pub fn device(&self) -> &Device {
        self.token_embd.device()
    }

    // Embed -> N x Layer -> final hidden states [batch, q_len, hidden_dim],
    // before the final norm and LM head.
    fn hidden(&self, items: &[BatchItem], kv_storage: &mut KvStorage) -> Result<Tensor> {
        let q_len = items[0].tokens.len();
        let device = self.device();

        // Turn token ids into vectors, in (batch, q_len) row-major order.
        let flat_tokens: Vec<u32> =
            items.iter().flat_map(|item| item.tokens.iter().copied()).collect();
        let mut x = embed(&flat_tokens, &self.token_embd, device)?.reshape((
            items.len(),
            q_len,
            self.config.hidden_dim as usize,
        ))?;

        let ctx = StepCtx::new(&self.config, items, device)?;
        // Run through every layer in sequence, each one refining x a bit more.
        for layer in &self.layers {
            x = layer.forward(&x, &self.config, &ctx, kv_storage)?;
        }
        Ok(x)
    }

    // One final norm after the last layer, then project to vocab-sized logits.
    fn lm_head(&self, x: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(x, &self.output_norm, self.config.rms_eps)?;
        linear(&normed, &self.output)
    }

    // Full forward pass over a batch: returns logits, [batch, q_len, vocab_size].
    pub fn forward(&self, items: &[BatchItem], kv_storage: &mut KvStorage) -> Result<Tensor> {
        let x = self.hidden(items, kv_storage)?;
        self.lm_head(&x)
    }

    // Same, but only projects each item's *last* row to logits:
    // [batch, 1, vocab_size]. Prefill only samples from the final position, so
    // running the vocab-sized LM head over every prompt row is wasted work.
    // RMSNorm is row-wise, so narrowing before the norm and head gives the
    // same last row.
    pub fn forward_last(&self, items: &[BatchItem], kv_storage: &mut KvStorage) -> Result<Tensor> {
        let x = self.hidden(items, kv_storage)?;
        let last = x.narrow(1, x.dim(1)? - 1, 1)?;
        self.lm_head(&last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn rms_norm_hand_computed() {
        let device = Device::Cpu;
        // x = [1, 2, 3], weight = [1, 1, 1], eps = 1e-5
        // mean(x^2) = (1 + 4 + 9) / 3 = 4.6667
        // rms = sqrt(4.6667 + 1e-5) ≈ 2.16025
        // expected = x / rms ≈ [0.46292, 0.92583, 1.38875]
        let x = Tensor::new(&[[1f32, 2., 3.]], &device).unwrap();
        let weight = Tensor::new(&[1f32, 1., 1.], &device).unwrap();
        let eps = 1e-5f32;

        let out = rms_norm(&x, &weight, eps).unwrap();
        let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        let expected = [0.46292, 0.92583, 1.38875];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32, 2., 3., 4.]], &device).unwrap(); // seq_len=1, head_dim=4, pos=0
        let out = rope(&x, 10000.0, &device, &[0]).unwrap();
        let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [1f32, 2., 3., 4.];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn rope_position_one_hand_computed() {
        let device = Device::Cpu;
        // head_dim=4, rope_theta=10000 -> theta_0=1, theta_1=0.01
        // row 0 = position 0 (identity), row 1 = position 1
        // rotate-half pairs are (x0,x2) and (x1,x3); set both pairs' "x" component
        // (indices 0 and 1) to 1, "y" component (indices 2 and 3) to 0, so both
        // pairs actually rotate instead of one of them staying at (0,0).
        // pair0=(x0,x2)=(1,0) rotates by theta_0*1=1, pair1=(x1,x3)=(1,0) rotates by theta_1*1=0.01
        let x = Tensor::new(&[[1f32, 1., 0., 0.], [1f32, 1., 0., 0.]], &device).unwrap();
        let out = rope(&x, 10000.0, &device, &[0, 1]).unwrap();
        let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // row 1 (position 1): first=[1,1], second=[0,0]
        // first_rot = first*cos - second*sin = [cos(1), cos(0.01)]
        // second_rot = first*sin + second*cos = [sin(1), sin(0.01)]
        let expected_row1 = [1f32.cos(), 0.01f32.cos(), 1f32.sin(), 0.01f32.sin()];
        for (got, want) in out[4..8].iter().zip(expected_row1.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn embed_looks_up_rows_in_order() {
        let device = Device::Cpu;
        // vocab_size=3, hidden_dim=2
        // row 0 = [1,2], row 1 = [3,4], row 2 = [5,6]
        let embedding_weight = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device).unwrap();
        let token_ids: [u32; 3] = [0, 2, 1];

        let out = embed(&token_ids, &embedding_weight, &device).unwrap();
        let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        // expect rows in token order: row0, row2, row1
        let expected = [1f32, 2., 5., 6., 3., 4.];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn swiglu_ffn_identity_weights() {
        let device = Device::Cpu;
        // hidden_dim=2, ffn_dim=2, all three weights = identity, so
        // gate(x) = up(x) = down(y) = the input unchanged.
        let identity = Tensor::new(&[[1f32, 0.], [0., 1.]], &device).unwrap();
        let x = Tensor::new(&[[1f32, 2.]], &device).unwrap();

        let out = swiglu_ffn(&x, &identity, &identity, &identity).unwrap();
        let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        // gate(x) = [1, 2], up(x) = [1, 2]
        // silu(1) = 1 * sigmoid(1) ≈ 0.731059
        // silu(2) = 2 * sigmoid(2) ≈ 1.761594
        // silu(gate) * up = [0.731059*1, 1.761594*2] = [0.731059, 3.523188]
        // down(...) = unchanged since down_weight is identity
        let expected = [0.731059f32, 3.523188];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn linear_matches_manual_matmul_for_2d_and_3d_input() {
        let device = Device::Cpu;
        // weight [out=2, in=3]; y = x @ weight^T.
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device).unwrap();
        let x = Tensor::from_vec(vec![1f32, 0., 1., 0., 1., 0.], (2, 3), &device).unwrap();
        // row0 = [1+3, 4+6] = [4, 10]; row1 = [2, 5]
        assert_eq!(flat(&linear(&x, &w).unwrap()), [4., 10., 2., 5.]);
        let x3 = x.reshape((1, 2, 3)).unwrap();
        let y3 = linear(&x3, &w).unwrap();
        assert_eq!(y3.dims(), &[1, 2, 2]);
        assert_eq!(flat(&y3), [4., 10., 2., 5.]);
    }

    #[test]
    fn gqa_attention_hand_computed() {
        let device = Device::Cpu;
        // n_heads=2, n_kv_heads=1 (group_size=2, both Q heads share the one KV head),
        // seq_len=2, head_dim=2, batch=1.

        // K (1 kv head): position0=[1,0], position1=[0,1]
        let k = Tensor::from_vec(vec![1f32, 0., 0., 1.], (1, 1, 2, 2), &device).unwrap();
        // V (1 kv head): position0=[10,0], position1=[0,20] - distinct so we can
        // see which position's value dominates the weighted sum.
        let v = Tensor::from_vec(vec![10f32, 0., 0., 20.], (1, 1, 2, 2), &device).unwrap();

        // Q head0: every position queries [1,0] (aligns with K's position0 direction)
        // Q head1: every position queries [0,1] (aligns with K's position1 direction)
        let q = Tensor::from_vec(
            vec![
                1f32, 0., 1., 0., // head0: pos0=[1,0], pos1=[1,0]
                0., 1., 0., 1., // head1: pos0=[0,1], pos1=[0,1]
            ],
            (1, 2, 2, 2),
            &device,
        )
        .unwrap();

        let mask = batch_mask(&[0], &[2], 2, 2, &device).unwrap();
        let out = flat(&gqa_attention(&q, &k, &v, &mask).unwrap());

        // Causal mask means position0 can only attend to position0, so its
        // output is always exactly V[0] regardless of head. Position1 attends
        // to both; softmax([<q,k0>, <q,k1>]/sqrt(2)) weights V[0] vs V[1].
        //
        // head0, pos1: scores=[1,0]/sqrt(2)=[0.7071,0] -> softmax≈[0.6698,0.3302]
        //   output = 0.6698*[10,0] + 0.3302*[0,20] ≈ [6.698, 6.604]
        // head1, pos1: scores=[0,1]/sqrt(2)=[0,0.7071] -> softmax≈[0.3302,0.6698]
        //   output = 0.3302*[10,0] + 0.6698*[0,20] ≈ [3.302, 13.396]
        let expected = [
            10f32, 0., // head0 pos0
            6.698, 6.604, // head0 pos1
            10., 0., // head1 pos0
            3.302, 13.396, // head1 pos1
        ];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-2, "got {got}, want {want}");
        }
    }

    // Naive per-head reference: attention for one (batch item, Q head) using
    // 2D ops only, with the KV head picked by index instead of by reshaping.
    fn naive_head(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, b: usize, h: usize, group: usize) -> Vec<f32> {
        let head_dim = q.dim(3).unwrap();
        let q_h = q.get(b).unwrap().get(h).unwrap(); // [q_len, head_dim]
        let k_h = k.get(b).unwrap().get(h / group).unwrap(); // [ctx, head_dim]
        let v_h = v.get(b).unwrap().get(h / group).unwrap();
        let m = mask.get(b).unwrap().get(0).unwrap(); // [q_len, ctx]
        let scores = q_h
            .matmul(&k_h.t().unwrap())
            .unwrap()
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
            .unwrap()
            .add(&m)
            .unwrap();
        flat(&softmax(&scores, 1).unwrap().matmul(&v_h).unwrap())
    }

    #[test]
    fn gqa_attention_matches_naive_per_head_reference() {
        let device = Device::Cpu;
        // 2 sequences of different real lengths (5 and 3, padded to ctx 5),
        // 4 Q heads over 2 KV heads (group_size 2), q_len 2 at offsets 3 and 1.
        let (batch, n_heads, n_kv_heads, q_len, ctx_len, head_dim) = (2, 4, 2, 2, 5, 4);
        let q = Tensor::randn(0f32, 1., (batch, n_heads, q_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1., (batch, n_kv_heads, ctx_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1., (batch, n_kv_heads, ctx_len, head_dim), &device).unwrap();
        let mask = batch_mask(&[3, 1], &[5, 3], q_len, ctx_len, &device).unwrap();

        let out = gqa_attention(&q, &k, &v, &mask).unwrap();
        assert_eq!(out.dims(), &[batch, n_heads, q_len, head_dim]);

        let group = n_heads / n_kv_heads;
        for b in 0..batch {
            for h in 0..n_heads {
                let want = naive_head(&q, &k, &v, &mask, b, h, group);
                let got = flat(&out.get(b).unwrap().get(h).unwrap());
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() < 1e-5, "b={b} h={h}: got {g}, want {w}");
                }
            }
        }
    }

    // Tiny fake model: 1 layer, hidden_dim=4, n_heads=2, n_kv_heads=1
    // (head_dim=2), ffn_dim=4, vocab_size=5. Random weights: these tests check
    // wiring and consistency between paths, not absolute values (each piece's
    // math is verified individually above).
    fn tiny_model() -> Model {
        let device = Device::Cpu;
        let (hidden_dim, n_heads, n_kv_heads, ffn_dim, vocab_size) = (4usize, 2usize, 1usize, 4usize, 5usize);
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

    // Prefill-shaped item for a sequence living in `blocks` (block_size slots
    // each), covering `tokens` from position `offset`.
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

    #[test]
    fn layer_forward_preserves_shape_and_changes_input() {
        let device = Device::Cpu;
        let model = tiny_model();
        let mut kv = KvStorage::new(1, 1, 2, 1, 2, &device).unwrap();
        let blocks = [BlockID(0)];
        let mut writes = Vec::new();
        let items = [item(&[0, 2], 0, &blocks, 2, &mut writes)];
        let ctx = StepCtx::new(&model.config, &items, &device).unwrap();

        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (1, 2, 4), &device).unwrap();
        let out = model.layers[0].forward(&x, &model.config, &ctx, &mut kv).unwrap();

        assert_eq!(out.dims(), &[1, 2, 4]);
        assert_ne!(flat(&out), flat(&x), "block should transform its input");
    }

    #[test]
    fn forward_produces_logits_of_expected_shape() {
        let device = Device::Cpu;
        let model = tiny_model();
        let mut kv = KvStorage::new(1, 1, 2, 1, 2, &device).unwrap();
        let blocks = [BlockID(0)];
        let mut writes = Vec::new();
        let items = [item(&[0, 2], 0, &blocks, 2, &mut writes)];

        let logits = model.forward(&items, &mut kv).unwrap();
        assert_eq!(logits.dims(), &[1, 2, model.config.vocab_size as usize]);
    }

    #[test]
    fn forward_last_matches_last_row_of_forward() {
        let device = Device::Cpu;
        let model = tiny_model();
        // 5 tokens over blocks of 2: exercises write_tokens splitting a
        // prefill into runs that end at block boundaries.
        let tokens: [u32; 5] = [1, 3, 2, 0, 4];
        let blocks = [BlockID(0), BlockID(1), BlockID(2)];
        let mut writes = Vec::new();
        let items = [item(&tokens, 0, &blocks, 2, &mut writes)];

        let mut kv_all = KvStorage::new(1, 3, 2, 1, 2, &device).unwrap();
        let mut kv_last = kv_all.clone();
        let all = model.forward(&items, &mut kv_all).unwrap();
        let last = model.forward_last(&items, &mut kv_last).unwrap();

        assert_eq!(last.dims(), &[1, 1, model.config.vocab_size as usize]);
        let want = flat(&all.get(0).unwrap().get(tokens.len() - 1).unwrap());
        let got = flat(&last);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn batched_decode_matches_unbatched_per_sequence() {
        let device = Device::Cpu;
        let model = tiny_model();
        // 4 blocks of 4 tokens each - sequence A gets block 0, sequence B
        // gets block 1, so their KV data never overlaps.
        let mut kv = KvStorage::new(1, 4, 4, 1, 2, &device).unwrap();
        let (blocks_a, blocks_b) = ([BlockID(0)], [BlockID(1)]);

        // Prefill both sequences (one call each) into disjoint blocks.
        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        model.forward(&[item(&[0, 2], 0, &blocks_a, 4, &mut wa)], &mut kv).unwrap();
        model.forward(&[item(&[1, 3], 0, &blocks_b, 4, &mut wb)], &mut kv).unwrap();

        // Snapshot post-prefill KV state so the batched and unbatched decode
        // steps below both start from the exact same cache contents.
        let mut kv_unbatched = kv.clone();

        // Unbatched: decode each sequence's one new token with its own call.
        let logits_a = model.forward(&[item(&[4], 2, &blocks_a, 4, &mut wa)], &mut kv_unbatched).unwrap();
        let logits_b = model.forward(&[item(&[4], 2, &blocks_b, 4, &mut wb)], &mut kv_unbatched).unwrap();

        // Batched: both sequences' one new token in a single call.
        let items = [
            item(&[4], 2, &blocks_a, 4, &mut wa),
            item(&[4], 2, &blocks_b, 4, &mut wb),
        ];
        let logits_batched = model.forward(&items, &mut kv).unwrap();
        assert_eq!(logits_batched.dims(), &[2, 1, model.config.vocab_size as usize]);

        for (name, row, want) in [("A", 0, &logits_a), ("B", 1, &logits_b)] {
            let got = flat(&logits_batched.get(row).unwrap());
            for (g, w) in got.iter().zip(flat(want).iter()) {
                assert!((g - w).abs() < 1e-4, "seq {name}: got {g}, want {w}");
            }
        }
    }

    #[test]
    fn batched_decode_with_unequal_lengths_matches_unbatched() {
        let device = Device::Cpu;
        let model = tiny_model();
        // Sequences of 3 and 6 tokens (blocks of 2: 2 vs 3 blocks), so the
        // batch pads A's context and masks it.
        let mut kv = KvStorage::new(1, 6, 2, 1, 2, &device).unwrap();
        let blocks_a = [BlockID(0), BlockID(1)];
        let blocks_b = [BlockID(2), BlockID(3), BlockID(4)];
        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        model.forward(&[item(&[0, 2, 1], 0, &blocks_a, 2, &mut wa)], &mut kv).unwrap();
        model.forward(&[item(&[1, 3, 0, 2, 4, 1], 0, &blocks_b, 2, &mut wb)], &mut kv).unwrap();
        let mut kv_unbatched = kv.clone();

        // A's next token lands at position 3 (block 1, slot 1); B's would be
        // position 6, so give B a block table with room via a 4th block.
        let blocks_b = [BlockID(2), BlockID(3), BlockID(4), BlockID(5)];
        let logits_a = model.forward(&[item(&[4], 3, &blocks_a, 2, &mut wa)], &mut kv_unbatched).unwrap();
        let logits_b = model.forward(&[item(&[4], 6, &blocks_b, 2, &mut wb)], &mut kv_unbatched).unwrap();

        let items = [
            item(&[4], 3, &blocks_a, 2, &mut wa),
            item(&[4], 6, &blocks_b, 2, &mut wb),
        ];
        let logits_batched = model.forward(&items, &mut kv).unwrap();
        for (name, row, want) in [("A", 0, &logits_a), ("B", 1, &logits_b)] {
            let got = flat(&logits_batched.get(row).unwrap());
            for (g, w) in got.iter().zip(flat(want).iter()) {
                assert!((g - w).abs() < 1e-4, "seq {name}: got {g}, want {w}");
            }
        }
    }
}
