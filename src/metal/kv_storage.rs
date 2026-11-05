// Metal mirror of cuda/kv_storage.rs: the same per-layer [num_blocks,
// block_size, n_kv_heads, head_dim] storage and the same batched write/gather
// contract, backed by shared MetalBufs and the kv_write/kv_gather kernels.
use kv_cache_scheduler::block_pool::BlockID;

use super::context::{MetalBuf, MetalRuntime, Result};
use super::conv_cache::MetalConvCache;

// Built once per forward call and reused by every layer's `gather`.
pub struct MetalGatherPlan {
    block_idx: MetalBuf<u32>,
    batch: u32,
    max_blocks: u32,
    ctx_len: u32,
}

impl MetalGatherPlan {
    // One (block table, real token count) pair per sequence in the batch.
    pub fn new(rt: &MetalRuntime, items: &[(&[BlockID], usize)]) -> Result<Self> {
        let batch = items.len();
        let ctx_len = items.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let max_blocks = items.iter().map(|&(blocks, _)| blocks.len()).max().unwrap_or(0);

        let mut idx: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for &(blocks, _) in items {
            idx.extend(blocks.iter().map(|b| b.0));
            // Pad with the sequence's own last block; the mask gives it zero weight.
            let pad = blocks.last().map_or(0, |b| b.0);
            idx.resize(idx.len() + (max_blocks - blocks.len()), pad);
        }
        Ok(MetalGatherPlan {
            block_idx: MetalBuf::from_slice(rt, &idx),
            batch: batch as u32,
            max_blocks: max_blocks as u32,
            ctx_len: ctx_len as u32,
        })
    }

    pub fn ctx_len(&self) -> u32 {
        self.ctx_len
    }

    // [batch, max_blocks] block table, padded with each sequence's last block.
    pub fn block_idx(&self) -> &MetalBuf<u32> {
        &self.block_idx
    }

    pub fn max_blocks(&self) -> u32 {
        self.max_blocks
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }
}

pub struct MetalKvStorage {
    kv_k: Vec<MetalBuf<f32>>,
    kv_v: Vec<MetalBuf<f32>>,
    block_size: u32,
    // LFM2's short-conv state, kept alongside the KV cache like the CPU KvStorage.
    pub(crate) conv: MetalConvCache,
}

impl MetalKvStorage {
    pub fn new(
        rt: &MetalRuntime,
        n_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let len = num_blocks * block_size * n_kv_heads * head_dim;
        let kv_k = (0..n_layers).map(|_| MetalBuf::zeros(rt, len)).collect();
        let kv_v = (0..n_layers).map(|_| MetalBuf::zeros(rt, len)).collect();
        Ok(MetalKvStorage {
            kv_k,
            kv_v,
            block_size: block_size as u32,
            conv: MetalConvCache::new(block_size),
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    // One layer's K and V storage, for the fused attention kernels that write
    // this call's tokens into it and read the context back without a gather.
    pub fn layer_cache(&self, layer: usize) -> (&MetalBuf<f32>, &MetalBuf<f32>) {
        (&self.kv_k[layer], &self.kv_v[layer])
    }

}
