// CUDA mirror of kv_storage::{KvStorage, GatherPlan}: same per-layer
// [num_blocks, block_size, n_kv_heads, head_dim] storage and the same
// batched write/gather contract, but backed by CudaSlice<f32> buffers and
// the kv_write/kv_gather kernels instead of Candle tensor ops. No shared
// trait with the CPU type - the CPU API is Candle-Tensor-typed end to end,
// so unifying the two would mean leaking CUDA types into it or vice versa.
use cudarc::driver::CudaSlice;
use kv_cache_scheduler::block_pool::BlockID;

use super::context::CudaRuntime;
use super::ops::{kv_gather_wrapper, kv_write_wrapper};

// Built once per forward call (block tables don't change between layers)
// and reused by every layer's `CudaKvStorage::gather`, same role as the
// CPU `GatherPlan`.
pub struct CudaGatherPlan {
    block_idx: CudaSlice<u32>,
    batch: u32,
    max_blocks: u32,
    ctx_len: u32,
}

impl CudaGatherPlan {
    // One (block table, real token count) pair per sequence in the batch.
    pub fn new(
        cuda_runtime: &CudaRuntime,
        items: &[(&[BlockID], usize)],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let batch = items.len();
        let ctx_len = items.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let max_blocks = items
            .iter()
            .map(|&(blocks, _)| blocks.len())
            .max()
            .unwrap_or(0);

        let mut idx: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for &(blocks, _) in items {
            idx.extend(blocks.iter().map(|b| b.0));
            // Same padding convention as the CPU GatherPlan: repeat the
            // sequence's own last block, which the batch mask then gives
            // exactly zero attention weight.
            let pad = blocks.last().map_or(0, |b| b.0);
            idx.resize(idx.len() + (max_blocks - blocks.len()), pad);
        }
        let block_idx = cuda_runtime.stream.clone_htod(&idx)?;
        Ok(CudaGatherPlan {
            block_idx,
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

pub struct CudaKvStorage {
    kv_k: Vec<CudaSlice<f32>>,
    kv_v: Vec<CudaSlice<f32>>,
    block_size: u32,
    n_kv_heads: u32,
    head_dim: u32,
}

impl CudaKvStorage {
    pub fn new(
        cuda_runtime: &CudaRuntime,
        n_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let len = num_blocks * block_size * n_kv_heads * head_dim;
        let kv_k = (0..n_layers)
            .map(|_| cuda_runtime.stream.alloc_zeros::<f32>(len))
            .collect::<Result<Vec<_>, _>>()?;
        let kv_v = (0..n_layers)
            .map(|_| cuda_runtime.stream.alloc_zeros::<f32>(len))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CudaKvStorage {
            kv_k,
            kv_v,
            block_size: block_size as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
        })
    }

    // Writes this call's tokens' K/V into `layer`'s cache in place.
    // block_ids, offsets: one entry per token (flat across the whole
    // batch). k, v: [n_tokens, n_kv_heads, head_dim] flattened.
    pub fn write_tokens(
        &mut self,
        cuda_runtime: &CudaRuntime,
        layer: usize,
        block_ids: &CudaSlice<u32>,
        offsets: &CudaSlice<u32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        kv_write_wrapper(
            cuda_runtime,
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

    // Reads every sequence's cached K and V for `layer`, right-padded to
    // the batch's longest context: both [batch, n_kv_heads, ctx_len,
    // head_dim], into caller-owned output buffers (sized by the caller via
    // `plan.batch() * n_kv_heads * plan.ctx_len() * head_dim`).
    pub fn gather(
        &self,
        cuda_runtime: &CudaRuntime,
        layer: usize,
        plan: &CudaGatherPlan,
        out_k: &mut CudaSlice<f32>,
        out_v: &mut CudaSlice<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        kv_gather_wrapper(
            cuda_runtime,
            &self.kv_k[layer],
            &plan.block_idx,
            out_k,
            plan.batch,
            plan.max_blocks,
            plan.ctx_len,
            self.block_size,
            self.n_kv_heads,
            self.head_dim,
        )?;
        kv_gather_wrapper(
            cuda_runtime,
            &self.kv_v[layer],
            &plan.block_idx,
            out_v,
            plan.batch,
            plan.max_blocks,
            plan.ctx_len,
            self.block_size,
            self.n_kv_heads,
            self.head_dim,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> Option<CudaRuntime> {
        CudaRuntime::new(0).ok()
    }

    // Same layout as kv_storage.rs's write_tokens_then_gather_round_trips_across_blocks:
    // 3 blocks of 2 slots. Sequence A: 3 tokens over blocks 0,1. Sequence B:
    // 1 token in block 2.
    #[test]
    fn write_tokens_then_gather_round_trips_across_blocks() {
        let Some(rt) = runtime() else { return };
        let n_kv_heads = 1usize;
        let head_dim = 2usize;
        let block_size = 2usize;
        let num_blocks = 3usize;

        let mut kv =
            CudaKvStorage::new(&rt, 1, num_blocks, block_size, n_kv_heads, head_dim).unwrap();

        let k: Vec<f32> = vec![10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 20.0, 20.5];
        let v = k.clone();
        let block_ids: Vec<u32> = vec![0, 0, 1, 2];
        let offsets: Vec<u32> = vec![0, 1, 0, 0];
        let k_dev = rt.stream.clone_htod(&k).unwrap();
        let v_dev = rt.stream.clone_htod(&v).unwrap();
        let block_ids_dev = rt.stream.clone_htod(&block_ids).unwrap();
        let offsets_dev = rt.stream.clone_htod(&offsets).unwrap();

        kv.write_tokens(&rt, 0, &block_ids_dev, &offsets_dev, &k_dev, &v_dev)
            .unwrap();

        let blocks_a = [BlockID(0), BlockID(1)];
        let blocks_b = [BlockID(2)];
        let plan = CudaGatherPlan::new(&rt, &[(&blocks_a, 3), (&blocks_b, 1)]).unwrap();
        assert_eq!(plan.ctx_len(), 3);

        let out_len =
            (plan.batch() * n_kv_heads as u32 * plan.ctx_len() * head_dim as u32) as usize;
        let mut out_k = rt.stream.alloc_zeros::<f32>(out_len).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(out_len).unwrap();
        kv.gather(&rt, 0, &plan, &mut out_k, &mut out_v).unwrap();

        let got_k = rt.stream.clone_dtoh(&out_k).unwrap();
        let got_v = rt.stream.clone_dtoh(&out_v).unwrap();
        assert_eq!(got_k, got_v);
        // Sequence A's three real tokens, in order.
        assert_eq!(&got_k[..6], &[10.0, 10.5, 11.0, 11.5, 12.0, 12.5]);
        // Sequence B's one real token; the rest is masked padding, not checked.
        assert_eq!(&got_k[6..8], &[20.0, 20.5]);
    }

    #[test]
    fn write_tokens_leaves_other_slots_untouched() {
        let Some(rt) = runtime() else { return };
        let mut kv = CudaKvStorage::new(&rt, 1, 1, 4, 1, 2).unwrap();
        let t = vec![1.0f32, 1.5];
        let t_dev = rt.stream.clone_htod(&t).unwrap();
        let block_ids = rt.stream.clone_htod(&[0u32]).unwrap();
        let offsets = rt.stream.clone_htod(&[2u32]).unwrap();
        kv.write_tokens(&rt, 0, &block_ids, &offsets, &t_dev, &t_dev)
            .unwrap();

        let blocks = [BlockID(0)];
        let plan = CudaGatherPlan::new(&rt, &[(&blocks, 4)]).unwrap();
        let mut out_k = rt.stream.alloc_zeros::<f32>(8).unwrap();
        let mut out_v = rt.stream.alloc_zeros::<f32>(8).unwrap();
        kv.gather(&rt, 0, &plan, &mut out_k, &mut out_v).unwrap();

        let got = rt.stream.clone_dtoh(&out_k).unwrap();
        assert_eq!(got, [0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0]);
    }
}
