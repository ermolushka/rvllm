use candle_core::{DType, Device, Result, Tensor};
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

// Which cache blocks to read for one batched attention call, built once per
// forward call (block tables don't change between layers) and reused by every
// layer's `KvStorage::gather`.
pub struct GatherPlan {
    // Physical block ids, u32 [batch * max_blocks]: each sequence's block
    // table, padded up to `max_blocks` by repeating its last block.
    block_idx: Tensor,
    batch: usize,
    max_blocks: usize,
    // Longest real context in the batch, in tokens.
    ctx_len: usize,
}

impl GatherPlan {
    // One (block table, real token count) pair per sequence in the batch.
    pub fn new(items: &[(&[BlockID], usize)], device: &Device) -> Result<Self> {
        let batch = items.len();
        let ctx_len = items.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let max_blocks = items.iter().map(|&(blocks, _)| blocks.len()).max().unwrap_or(0);

        let mut idx: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for &(blocks, _) in items {
            idx.extend(blocks.iter().map(|b| b.0));
            // The padding only ever lands at positions the attention mask
            // rules out, so any valid block will do; the sequence's own last
            // block keeps every index in range and every value finite.
            let pad = blocks.last().map_or(0, |b| b.0);
            idx.resize(idx.len() + (max_blocks - blocks.len()), pad);
        }
        let block_idx = Tensor::from_vec(idx, batch * max_blocks, device)?;
        Ok(GatherPlan { block_idx, batch, max_blocks, ctx_len })
    }

    // Longest real context in the batch: the `ctx_len` dim of `gather`'s output.
    pub fn ctx_len(&self) -> usize {
        self.ctx_len
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
    ) -> Result<KvStorage> {
        let shape = (num_blocks, block_size, n_kv_heads, head_dim);
        let kv_k = (0..n_layers)
            .map(|_| Tensor::zeros(shape, DType::F32, device))
            .collect::<Result<Vec<_>>>()?;
        let kv_v = (0..n_layers)
            .map(|_| Tensor::zeros(shape, DType::F32, device))
            .collect::<Result<Vec<_>>>()?;
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
    ) -> Result<()> {
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

    // Reads every sequence's cached K and V for `layer`, right-padded to the
    // batch's longest context: both [batch, n_kv_heads, ctx_len, head_dim],
    // contiguous. One `index_select` over the block table per tensor. The
    // padding holds other (finite) cache data rather than zeros, which is
    // safe because `batch_mask` gives it exactly zero attention weight.
    pub fn gather(&self, layer: usize, plan: &GatherPlan) -> Result<(Tensor, Tensor)> {
        let read = |cache: &Tensor| -> Result<Tensor> {
            let blocks = cache.index_select(&plan.block_idx, 0)?;
            let (_, block_size, n_kv_heads, head_dim) = blocks.dims4()?;
            blocks
                .reshape((plan.batch, plan.max_blocks * block_size, n_kv_heads, head_dim))?
                .narrow(1, 0, plan.ctx_len)?
                .transpose(1, 2)?
                .contiguous()
        };
        Ok((read(&self.kv_k[layer])?, read(&self.kv_v[layer])?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    // K/V rows for `n` tokens with one kv head and head_dim 2: token i holds
    // [base + i, base + i + 0.5], so every slot is distinguishable.
    fn rows(base: f32, n: usize, device: &Device) -> Tensor {
        let data: Vec<f32> = (0..n).flat_map(|i| [base + i as f32, base + i as f32 + 0.5]).collect();
        Tensor::from_vec(data, (n, 1, 2), device).unwrap()
    }

    #[test]
    fn write_tokens_then_gather_round_trips_across_blocks() {
        let device = Device::Cpu;
        // 3 blocks of 2 slots. Sequence A: 3 tokens over blocks 0,1. Sequence
        // B: 1 token in block 2.
        let mut kv = KvStorage::new(1, 3, 2, 1, 2, &device).unwrap();
        let a = rows(10.0, 3, &device);
        let b = rows(20.0, 1, &device);
        kv.write_tokens(0, &[(BlockID(0), 0), (BlockID(0), 1), (BlockID(1), 0)], &a, &a).unwrap();
        kv.write_tokens(0, &[(BlockID(2), 0)], &b, &b).unwrap();

        let blocks_a = [BlockID(0), BlockID(1)];
        let blocks_b = [BlockID(2)];
        let plan = GatherPlan::new(&[(&blocks_a, 3), (&blocks_b, 1)], &device).unwrap();
        assert_eq!(plan.ctx_len(), 3);
        let (k, v) = kv.gather(0, &plan).unwrap();

        // [batch=2, n_kv_heads=1, ctx_len=3, head_dim=2]
        assert_eq!(k.dims(), &[2, 1, 3, 2]);
        assert_eq!(flat(&k), flat(&v));
        let got = flat(&k);
        // Sequence A's three real tokens, in order.
        assert_eq!(&got[..6], &[10.0, 10.5, 11.0, 11.5, 12.0, 12.5]);
        // Sequence B's one real token; the rest is masked padding, not checked.
        assert_eq!(&got[6..8], &[20.0, 20.5]);
    }

    #[test]
    fn write_tokens_leaves_other_slots_untouched() {
        let device = Device::Cpu;
        let mut kv = KvStorage::new(1, 1, 4, 1, 2, &device).unwrap();
        let t = rows(1.0, 1, &device);
        kv.write_tokens(0, &[(BlockID(0), 2)], &t, &t).unwrap();

        let blocks = [BlockID(0)];
        let plan = GatherPlan::new(&[(&blocks, 4)], &device).unwrap();
        let (k, _) = kv.gather(0, &plan).unwrap();
        assert_eq!(flat(&k), [0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0]);
    }

    #[test]
    fn clone_is_independent_of_original() {
        let device = Device::Cpu;
        let mut original = KvStorage::new(1, 1, 2, 1, 2, &device).unwrap();
        let snapshot = original.clone();
        let t = rows(1.0, 1, &device);
        original.write_tokens(0, &[(BlockID(0), 0)], &t, &t).unwrap();

        let blocks = [BlockID(0)];
        let plan = GatherPlan::new(&[(&blocks, 1)], &device).unwrap();
        let (k_orig, _) = original.gather(0, &plan).unwrap();
        let (k_snap, _) = snapshot.gather(0, &plan).unwrap();
        assert_eq!(flat(&k_orig), [1.0, 1.5]);
        assert_eq!(flat(&k_snap), [0.0, 0.0]);
    }
}
