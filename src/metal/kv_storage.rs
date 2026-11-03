// Metal mirror of cuda/kv_storage.rs: the same per-layer [num_blocks,
// block_size, n_kv_heads, head_dim] storage and the same batched write/gather
// contract, backed by shared MetalBufs and the kv_write/kv_gather kernels.
use kv_cache_scheduler::block_pool::BlockID;

use super::context::{MetalBuf, MetalRuntime, Result};
use super::conv_cache::MetalConvCache;
use super::ops::{kv_gather_wrapper, kv_write_wrapper};

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

    pub fn batch(&self) -> u32 {
        self.batch
    }
}

pub struct MetalKvStorage {
    kv_k: Vec<MetalBuf<f32>>,
    kv_v: Vec<MetalBuf<f32>>,
    block_size: u32,
    n_kv_heads: u32,
    head_dim: u32,
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
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            conv: MetalConvCache::new(block_size),
        })
    }

    // Writes this call's tokens' K/V into `layer`'s cache in place.
    // k, v: [n_tokens, n_kv_heads, head_dim].
    pub fn write_tokens(
        &mut self,
        rt: &MetalRuntime,
        layer: usize,
        block_ids: &MetalBuf<u32>,
        offsets: &MetalBuf<u32>,
        k: &MetalBuf<f32>,
        v: &MetalBuf<f32>,
    ) -> Result<()> {
        kv_write_wrapper(
            rt,
            k,
            v,
            block_ids,
            offsets,
            &mut self.kv_k[layer],
            &mut self.kv_v[layer],
            self.n_kv_heads,
            self.head_dim,
            self.block_size,
        )
    }

    // Every sequence's cached K and V for `layer`, right-padded to the batch's
    // longest context: both [batch, n_kv_heads, ctx_len, head_dim].
    pub fn gather(
        &self,
        rt: &MetalRuntime,
        layer: usize,
        plan: &MetalGatherPlan,
        out_k: &mut MetalBuf<f32>,
        out_v: &mut MetalBuf<f32>,
    ) -> Result<()> {
        for (cache, out) in [(&self.kv_k[layer], out_k), (&self.kv_v[layer], out_v)] {
            kv_gather_wrapper(
                rt,
                cache,
                &plan.block_idx,
                out,
                plan.batch,
                plan.max_blocks,
                plan.ctx_len,
                self.block_size,
                self.n_kv_heads,
                self.head_dim,
            )?;
        }
        Ok(())
    }
}
