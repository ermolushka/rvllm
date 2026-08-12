use candle_core::{Device, Tensor, scalar::TensorScalar};
use candle_nn::ops::{silu, softmax};

use crate::weights::SmollLM230MConfig;

// RMSNorm: rescale each row by its root-mean-square, then apply a learned
// per-channel scale. No mean-centering (unlike LayerNorm) - just magnitude
// control, which is cheaper and works fine for Llama-family models.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    // rms = sqrt(mean(x^2) + eps), one value per row (kept as a column so it
    // broadcasts back against every element of that row).
    let rms = x
        .powf(2.0)?
        .mean_keepdim(1)?
        .affine(1.0, eps as f64)?
        .sqrt()?;
    // Divide each row by its own rms, then scale each channel by `weight`.
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(weight)
}

// Applies RoPE (rotate-half convention) to `x` of shape [seq_len, head_dim],
// treating row index as the token position.
pub fn rope(
    x: &Tensor,
    rope_theta: f32,
    device: &Device,
    position_offset: usize,
) -> candle_core::Result<Tensor> {
    let (seq_len, head_dim) = x.dims2()?;
    let half = head_dim / 2;

    // Each dimension-pair rotates at its own frequency - pair 0 fastest,
    // later pairs slower - so different pairs encode position at different
    // resolutions, like clock hands moving at different speeds.
    let theta: Vec<f32> = (0..half)
        .map(|j| rope_theta.powf(-2.0 * j as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::from_vec(theta, (1, half), device)?;

    // Row index doubles as the token's position (0, 1, 2, ...).
    let positions: Vec<f32> = (position_offset..position_offset + seq_len)
        .map(|p| p as f32)
        .collect();
    let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;

    // angle[pos, j] = pos * theta_j - how far pair j has rotated by this position.
    let angles = positions.broadcast_mul(&theta)?;
    let cos = angles.cos()?;
    let sin = angles.sin()?;

    // Rotate-half pairing: dim i pairs with dim i + half, not its neighbor.
    let first = x.narrow(1, 0, half)?;
    let second = x.narrow(1, half, half)?;

    // Standard 2D rotation applied to each (first, second) pair.
    let first_rot = first.mul(&cos)?.sub(&second.mul(&sin)?)?;
    let second_rot = first.mul(&sin)?.add(&second.mul(&cos)?)?;

    Tensor::cat(&[&first_rot, &second_rot], 1)
}

// Looks up embedding rows for `token_ids` from `embedding_weight` ([vocab_size,
// hidden_dim]), producing a [seq_len, hidden_dim] tensor.
pub fn embed(
    token_ids: &[u32],
    embedding_weight: &Tensor,
    device: &Device,
) -> candle_core::Result<Tensor> {
    // Wrap the ids so we can use them as row indices.
    let tokens_tensor = Tensor::from_slice(token_ids, token_ids.len(), device)?;
    // Pull out row `token_ids[i]` for each i - that row is that token's embedding.
    embedding_weight.index_select(&tokens_tensor, 0)
}

// SwiGLU FFN: down(silu(gate(x)) * up(x)).
// x: [seq_len, hidden_dim]
// gate_weight, up_weight: [ffn_dim, hidden_dim] (candle Linear convention: out_features x in_features)
// down_weight: [hidden_dim, ffn_dim]
pub fn swiglu_ffn(
    x: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> candle_core::Result<Tensor> {
    // Two parallel projections of the same input...
    let gate = x.matmul(&gate_weight.t()?)?;
    let up = x.matmul(&up_weight.t()?)?;
    // ...gate gets squashed through silu (a smooth "soft gate" nonlinearity)...
    let silu_gate = silu(&gate)?;
    // ...then used to scale up, elementwise - this is the "gating" in SwiGLU.
    let result = silu_gate.mul(&up)?;
    // Project back down to hidden_dim.
    result.matmul(&down_weight.t()?)
}

// Grouped-query causal attention.
// q: [n_heads, seq_len, head_dim]     (RoPE already applied)
// k: [n_kv_heads, seq_len, head_dim]  (RoPE already applied)
// v: [n_kv_heads, seq_len, head_dim]
// returns: [n_heads, seq_len, head_dim]
pub fn gqa_attention(q: &Tensor, k: &Tensor, v: &Tensor, position_offset: usize) -> candle_core::Result<Tensor> {
    let (n_heads, q_seq_len, head_dim) = q.dims3()?;
    let (n_kv_heads, k_seq_len, _) = k.dims3()?;

    // GQA shares each KV head across several Q heads instead of 1:1 - this is
    // how many Q heads point at the same KV head.
    let group_size = n_heads / n_kv_heads;

    // K/V only have n_kv_heads heads, but we need to compare against all
    // n_heads Q heads. Duplicate each KV head group_size times, contiguously,
    // so KV head j lines up with Q heads [j*group_size .. (j+1)*group_size).
    let k_repeated = k
        .unsqueeze(1)? // [n_kv_heads, 1, k_seq_len, head_dim]
        .broadcast_as((n_kv_heads, group_size, k_seq_len, head_dim))?
        .reshape((n_heads, k_seq_len, head_dim))?;

    let v_repeated = v
        .unsqueeze(1)?
        .broadcast_as((n_kv_heads, group_size, k_seq_len, head_dim))?
        .reshape((n_heads, k_seq_len, head_dim))?;

    // Dot product of every query against every key gives a raw "how similar
    // are these two vectors" score per (query position, key position) pair.
    let dot_product_scores = q.matmul(&k_repeated.transpose(1, 2)?)?;
    // Scaling by 1/sqrt(head_dim) keeps these scores from growing huge as
    // head_dim increases, which would otherwise push softmax into saturating
    // (near one-hot) outputs regardless of how similar the vectors really are.
    let scaled_dot_product = dot_product_scores.affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;

    // Causal mask: a token at position i must not see future tokens (j > i),
    // since at generation time those tokens don't exist yet. Adding -inf to
    // those positions forces softmax to assign them exactly 0 weight.
    let mut mask_data = vec![0f32; q_seq_len * k_seq_len];
    for i in 0..q_seq_len {
        for j in 0..k_seq_len {
            if j > position_offset + i {
                mask_data[i * k_seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (q_seq_len, k_seq_len), q.device())?;
    let scaled_dot_product = scaled_dot_product.broadcast_add(&mask)?;

    // Softmax turns each query's row of scores into a probability
    // distribution over key positions - "how much attention to pay to each
    // position," summing to 1.
    let softmax_values = softmax(&scaled_dot_product, 2)?;
    // Using those weights to blend the V vectors gives, for each query
    // position, a weighted average of "what every position it attends to is
    // saying" - this is attention's actual output.
    softmax_values.matmul(&v_repeated)
}

// One Llama transformer block:
//   h   = x + attention(rms_norm(x, attn_norm_weight)) // aka residual
//   out = h + swiglu_ffn(rms_norm(h, ffn_norm_weight))
//
// x: [seq_len, hidden_dim]
// attn_q_weight: [n_heads*head_dim, hidden_dim], attn_k/v_weight: [n_kv_heads*head_dim, hidden_dim]
// attn_output_weight: [hidden_dim, n_heads*head_dim]
// ffn_gate/up_weight: [ffn_dim, hidden_dim], ffn_down_weight: [hidden_dim, ffn_dim]
// returns: [seq_len, hidden_dim]
#[allow(clippy::too_many_arguments)]
pub fn transformer_block(
    x: &Tensor,
    attn_norm_weight: &Tensor,
    attn_q_weight: &Tensor,
    attn_k_weight: &Tensor,
    attn_v_weight: &Tensor,
    attn_output_weight: &Tensor,
    ffn_norm_weight: &Tensor,
    ffn_gate_weight: &Tensor,
    ffn_up_weight: &Tensor,
    ffn_down_weight: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    rope_theta: f32,
    eps: f32,
    device: &Device,
    kv_cache: &mut (Tensor, Tensor),
    position_offset: usize, // where in the sequence 'x' starts, mostly for RoPE
) -> candle_core::Result<Tensor> {
    // Normalize before attention (pre-norm), not after - this is the
    // convention Llama uses, and it's what keeps deep stacks of these blocks
    // stable to train/run.
    let normed = rms_norm(x, attn_norm_weight, eps)?;

    // Project into query/key/value space. Q gets n_heads worth of output,
    // K/V get fewer (n_kv_heads) since GQA shares them across several Q heads.
    let q = normed.matmul(&attn_q_weight.t()?)?;
    let k = normed.matmul(&attn_k_weight.t()?)?;
    let v = normed.matmul(&attn_v_weight.t()?)?;
    let (seq_len, hidden_dim) = x.dims2()?;
    let head_dim = hidden_dim / n_heads;

    // Split the projected dim into per-head chunks and move the head axis to
    // the front - gqa_attention expects [n_heads, seq_len, head_dim], not one
    // big [seq_len, n_heads*head_dim] blob.
    let q = q.reshape((seq_len, n_heads, head_dim))?.transpose(0, 1)?;
    let k = k
        .reshape((seq_len, n_kv_heads, head_dim))?
        .transpose(0, 1)?;
    let v = v
        .reshape((seq_len, n_kv_heads, head_dim))?
        .transpose(0, 1)?;

    // RoPE has to be applied per head (it only knows how to rotate a single
    // [seq_len, head_dim] slice), so loop over heads and restitch the results.
    // Only Q and K get rotated - V carries content, not position.
    let q_slices: Vec<Tensor> = (0..n_heads)
        .map(|h| rope(&q.get(h)?, rope_theta, device, position_offset))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let q_roped = Tensor::stack(&q_slices, 0)?;
    let k_slices: Vec<Tensor> = (0..n_kv_heads)
        .map(|h| rope(&k.get(h)?, rope_theta, device, position_offset))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let k_roped = Tensor::stack(&k_slices, 0)?;

    // dim 1 because [n_kv_heads, cached_len, head_dim], dim 1 is growing seq length, the rest two axis are fixed
    let new_k_cache = Tensor::cat(&[&kv_cache.0, &k_roped], 1)?;
    let new_v_cache = Tensor::cat(&[&kv_cache.1, &v], 1)?;
    // update the cache with new concat value
    *kv_cache = (new_k_cache, new_v_cache);
    // The actual attention computation.
    let mut attn_out = gqa_attention(&q_roped, &kv_cache.0, &kv_cache.1, position_offset)?;

    // Merge the heads back into one dimension - inverse of the earlier split.
    attn_out = attn_out
        .transpose(0, 1)?
        .reshape((seq_len, n_heads * head_dim))?;

    // Project attention's output back down to hidden_dim so it can be added
    // to the residual stream.
    attn_out = attn_out.matmul(&attn_output_weight.t()?)?;

    // Residual connection: the block adds *to* x rather than replacing it,
    // so gradients (and, at inference time, information) can flow straight
    // through even if attention contributes very little.
    let h = x.add(&attn_out)?;

    // Same pre-norm pattern before the FFN half of the block.
    let normed2 = rms_norm(&h, ffn_norm_weight, eps)?;
    let ffn_out = swiglu_ffn(&normed2, ffn_gate_weight, ffn_up_weight, ffn_down_weight)?;

    // Second residual connection - this is the block's output.
    h.add(&ffn_out)
}

// Full forward pass: embed(tokens) -> N x transformer_block -> final norm -> LM head.
// token_ids: input sequence. Returns logits, [seq_len, vocab_size].
pub fn forward(
    config: &SmollLM230MConfig,
    token_ids: &[u32],
    device: &Device,
    kv_cache: &mut Vec<(Tensor, Tensor)>,
) -> candle_core::Result<Tensor> {
    // Turn token ids into vectors.
    let mut x = embed(token_ids, &config.tensors["token_embd.weight"], device)?;
    // Run through every layer in sequence, each one refining x a bit more.
    // Pulling weights straight out of the tensor map by the same key strings
    // weights.rs used when loading them.
    for i in 0..config.n_layers {
        let position_offset = kv_cache[i as usize].0.dim(1)?;
        x = transformer_block(
            &x,
            &config.tensors[&format!("blk.{i}.attn_norm.weight")],
            &config.tensors[&format!("blk.{i}.attn_q.weight")],
            &config.tensors[&format!("blk.{i}.attn_k.weight")],
            &config.tensors[&format!("blk.{i}.attn_v.weight")],
            &config.tensors[&format!("blk.{i}.attn_output.weight")],
            &config.tensors[&format!("blk.{i}.ffn_norm.weight")],
            &config.tensors[&format!("blk.{i}.ffn_gate.weight")],
            &config.tensors[&format!("blk.{i}.ffn_up.weight")],
            &config.tensors[&format!("blk.{i}.ffn_down.weight")],
            config.n_heads as usize,
            config.n_kv_heads as usize,
            config.rope_theta,
            config.rms_eps,
            device,
            &mut kv_cache[i as usize],
            position_offset,
        )?;
    }

    // One final norm after the last layer, then project to vocab-sized logits.
    let normed = rms_norm(&x, &config.tensors["output_norm.weight"], config.rms_eps)?;
    normed.matmul(&config.tensors["output.weight"].t()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
        let out = rope(&x, 10000.0, &device, 0 as usize).unwrap();
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
        let out = rope(&x, 10000.0, &device, 0 as usize).unwrap();
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
    fn gqa_attention_hand_computed() {
        let device = Device::Cpu;
        // n_heads=2, n_kv_heads=1 (group_size=2, both Q heads share the one KV head),
        // seq_len=2, head_dim=2.

        // K (1 kv head): position0=[1,0], position1=[0,1]
        let k = Tensor::from_vec(vec![1f32, 0., 0., 1.], (1, 2, 2), &device).unwrap();
        // V (1 kv head): position0=[10,0], position1=[0,20] - distinct so we can
        // see which position's value dominates the weighted sum.
        let v = Tensor::from_vec(vec![10f32, 0., 0., 20.], (1, 2, 2), &device).unwrap();

        // Q head0: every position queries [1,0] (aligns with K's position0 direction)
        // Q head1: every position queries [0,1] (aligns with K's position1 direction)
        let q = Tensor::from_vec(
            vec![
                1f32, 0., 1., 0., // head0: pos0=[1,0], pos1=[1,0]
                0., 1., 0., 1., // head1: pos0=[0,1], pos1=[0,1]
            ],
            (2, 2, 2),
            &device,
        )
        .unwrap();

        let out = gqa_attention(&q, &k, &v, 0).unwrap();
       let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

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

    #[test]
    fn transformer_block_preserves_shape_and_changes_input() {
        let device = Device::Cpu;
        // hidden_dim=4, n_heads=2, n_kv_heads=1, head_dim=2, ffn_dim=4, seq_len=2.
        // This isn't a hand-computed correctness test (RMSNorm, RoPE, GQA
        // attention, and SwiGLU are already individually verified above) - it's
        // a wiring/shape smoke test to catch mistakes in the reshape/transpose
        // steps that stitch those pieces together.
        let seq_len = 2;
        let hidden_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let ffn_dim = 4;

        let x = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (seq_len, hidden_dim),
            &device,
        )
        .unwrap();

        let attn_norm_weight = Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap();
        let ffn_norm_weight = Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap();

        // Non-identity but simple weights, just to exercise real matmuls.
        let attn_q_weight = Tensor::rand(0f32, 1., (n_heads * 2, hidden_dim), &device).unwrap(); // head_dim=2
        let attn_k_weight = Tensor::rand(0f32, 1., (n_kv_heads * 2, hidden_dim), &device).unwrap();
        let attn_v_weight = Tensor::rand(0f32, 1., (n_kv_heads * 2, hidden_dim), &device).unwrap();
        let attn_output_weight =
            Tensor::rand(0f32, 1., (hidden_dim, n_heads * 2), &device).unwrap();

        let ffn_gate_weight = Tensor::rand(0f32, 1., (ffn_dim, hidden_dim), &device).unwrap();
        let ffn_up_weight = Tensor::rand(0f32, 1., (ffn_dim, hidden_dim), &device).unwrap();
        let ffn_down_weight = Tensor::rand(0f32, 1., (hidden_dim, ffn_dim), &device).unwrap();
        let mut kv_cache: (Tensor, Tensor) = (
            Tensor::zeros((n_kv_heads, 0, 2), candle_core::DType::F32, &device).unwrap(),
            Tensor::zeros((n_kv_heads, 0, 2), candle_core::DType::F32, &device).unwrap(),
        );
        let out = transformer_block(
            &x,
            &attn_norm_weight,
            &attn_q_weight,
            &attn_k_weight,
            &attn_v_weight,
            &attn_output_weight,
            &ffn_norm_weight,
            &ffn_gate_weight,
            &ffn_up_weight,
            &ffn_down_weight,
            n_heads,
            n_kv_heads,
            10000.0,
            1e-5,
            &device,
            &mut kv_cache,
            0,
        )
        .unwrap();

        assert_eq!(out.dims(), &[seq_len, hidden_dim]);

        let out_vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let in_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(
            out_vals, in_vals,
            "block should transform its input, not pass it through unchanged"
        );
    }

    #[test]
    fn forward_produces_logits_of_expected_shape() {
        use crate::weights::SmollLM230MConfig;
        use std::collections::HashMap;

        let device = Device::Cpu;
        // Tiny fake model: 1 layer, hidden_dim=4, n_heads=2, n_kv_heads=1
        // (head_dim=2), ffn_dim=4, vocab_size=5. Same rationale as the
        // transformer_block smoke test: this only checks wiring/shapes, since
        // the math inside each piece is already verified above.
        let hidden_dim = 4usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = hidden_dim / n_heads;
        let ffn_dim = 4usize;
        let vocab_size = 5usize;

        let mut tensors = HashMap::new();
        tensors.insert(
            "token_embd.weight".to_string(),
            Tensor::rand(0f32, 1., (vocab_size, hidden_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.attn_norm.weight".to_string(),
            Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            Tensor::rand(0f32, 1., (n_heads * head_dim, hidden_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.attn_k.weight".to_string(),
            Tensor::rand(0f32, 1., (n_kv_heads * head_dim, hidden_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.attn_v.weight".to_string(),
            Tensor::rand(0f32, 1., (n_kv_heads * head_dim, hidden_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.attn_output.weight".to_string(),
            Tensor::rand(0f32, 1., (hidden_dim, n_heads * head_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.ffn_norm.weight".to_string(),
            Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "blk.0.ffn_gate.weight".to_string(),
            Tensor::rand(0f32, 1., (ffn_dim, hidden_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.ffn_up.weight".to_string(),
            Tensor::rand(0f32, 1., (ffn_dim, hidden_dim), &device).unwrap(),
        );
        tensors.insert(
            "blk.0.ffn_down.weight".to_string(),
            Tensor::rand(0f32, 1., (hidden_dim, ffn_dim), &device).unwrap(),
        );
        tensors.insert(
            "output_norm.weight".to_string(),
            Tensor::ones(hidden_dim, candle_core::DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "output.weight".to_string(),
            Tensor::rand(0f32, 1., (vocab_size, hidden_dim), &device).unwrap(),
        );

        let config = SmollLM230MConfig {
            n_layers: 1,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            hidden_dim: hidden_dim as u32,
            ffn_dim: ffn_dim as u32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            vocab_size: vocab_size as u32,
            context_length: 8,
            eos_token_id: 0,
            bos_token_id: 0,
            tensors,
        };

        let token_ids: [u32; 2] = [0, 2];
        let mut kv_cache: Vec<(Tensor, Tensor)> = vec![(
            Tensor::zeros((n_kv_heads, 0, head_dim), candle_core::DType::F32, &device).unwrap(),
            Tensor::zeros((n_kv_heads, 0, head_dim), candle_core::DType::F32, &device).unwrap(),
        )];
        let logits = forward(&config, &token_ids, &device, &mut kv_cache).unwrap();

        assert_eq!(logits.dims(), &[token_ids.len(), vocab_size]);
    }
}
