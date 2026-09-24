use candle_core::{DType, Device, Tensor};
use kv_cache_scheduler::block_pool::BlockID;

// [num_blocks, block_size, n_kv_heads, head_dim] per layer.
pub struct KvStorage {
    kv_k: Vec<Tensor>,
    kv_v: Vec<Tensor>,
}

// `write_tokens` mutates the cache tensors in place, and `Tensor::clone` only
// bumps a refcount on the shared storage - so a derived `Clone` would alias
// both copies. Deep-copy instead so a clone is an independent cache.
impl Clone for KvStorage {
    fn clone(&self) -> Self {
        let deep = |ts: &[Tensor]| {
            ts.iter()
                .map(|t| t.copy().expect("copying a KV cache tensor"))
                .collect()
        };
        KvStorage { kv_k: deep(&self.kv_k), kv_v: deep(&self.kv_v) }
    }
}

impl KvStorage {
    pub fn new(
        n_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> candle_core::Result<KvStorage> {
        let shape = (num_blocks, block_size, n_kv_heads, head_dim);
        let kv_k = (0..n_layers)
            .map(|_| Tensor::zeros(shape, DType::F32, device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let kv_v = (0..n_layers)
            .map(|_| Tensor::zeros(shape, DType::F32, device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(KvStorage { kv_k, kv_v })
    }

    // Writes `positions.len()` tokens' K/V into `layer`'s cache in place.
    // k, v: [n_tokens, n_kv_heads, head_dim], row i belonging to positions[i].
    // Consecutive tokens landing in consecutive slots of one block are
    // copied with a single `slice_set`, so a prefill costs about one copy per
    // block touched rather than one per token, and never copies the cache.
    pub fn write_tokens(
        &mut self,
        layer: usize,
        positions: &[(BlockID, usize)],
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<()> {
        let (n_tokens, n_kv_heads, head_dim) = k.dims3()?;
        debug_assert_eq!(n_tokens, positions.len());
        let mut start = 0;
        while start < positions.len() {
            let (block_id, offset) = positions[start];
            let mut len = 1;
            while start + len < positions.len()
                && positions[start + len] == (block_id, offset + len)
            {
                len += 1;
            }
            let block = block_id.0 as usize;
            let run_k = k.narrow(0, start, len)?.reshape((1, len, n_kv_heads, head_dim))?;
            let run_v = v.narrow(0, start, len)?.reshape((1, len, n_kv_heads, head_dim))?;
            self.kv_k[layer].narrow(0, block, 1)?.slice_set(&run_k, 1, offset)?;
            self.kv_v[layer].narrow(0, block, 1)?.slice_set(&run_v, 1, offset)?;
            start += len;
        }
        Ok(())
    }

    pub fn gather(
        &self,
        layer: usize,
        blocks: &[BlockID],
        num_tokens: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let k_blocks: Vec<Tensor> = blocks
            .iter()
            .map(|b| self.kv_k[layer].get(b.0 as usize))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let v_blocks: Vec<Tensor> = blocks
            .iter()
            .map(|b| self.kv_v[layer].get(b.0 as usize))
            .collect::<candle_core::Result<Vec<_>>>()?;

        let k = Tensor::cat(&k_blocks, 0)?.transpose(0, 1)?;
        let v = Tensor::cat(&v_blocks, 0)?.transpose(0, 1)?;
        let k = k.narrow(1, 0, num_tokens)?;
        let v = v.narrow(1, 0, num_tokens)?;
        Ok((k, v))
    }

    // Batched version of `gather`: one (blocks, num_tokens) pair per sequence in
    // the batch. Each sequence's K/V is right-padded with zeros up to the batch's
    // longest context so they can be stacked into one [batch, n_kv_heads, max_len,
    // head_dim] tensor - `gqa_attention_batch`'s mask is what keeps that padding
    // from contributing to the softmax. Returns the batched K, V, and max_len.
    pub fn gather_batch(
        &self,
        layer: usize,
        items: &[(&[BlockID], usize)],
    ) -> candle_core::Result<(Tensor, Tensor, usize)> {
        let max_len = items.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let n_kv_heads = self.kv_k[layer].dim(2)?;
        let head_dim = self.kv_k[layer].dim(3)?;
        let device = self.kv_k[layer].device();

        let mut k_padded = Vec::with_capacity(items.len());
        let mut v_padded = Vec::with_capacity(items.len());
        for &(blocks, num_tokens) in items {
            let (k, v) = self.gather(layer, blocks, num_tokens)?;
            let pad = max_len - num_tokens;
            let (k, v) = if pad > 0 {
                let zeros = Tensor::zeros((n_kv_heads, pad, head_dim), DType::F32, device)?;
                (Tensor::cat(&[&k, &zeros], 1)?, Tensor::cat(&[&v, &zeros], 1)?)
            } else {
                (k, v)
            };
            k_padded.push(k);
            v_padded.push(v);
        }
        let k = Tensor::stack(&k_padded, 0)?;
        let v = Tensor::stack(&v_padded, 0)?;
        Ok((k, v, max_len))
    }
}
