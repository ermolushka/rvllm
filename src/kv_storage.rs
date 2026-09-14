use candle_core::{DType, Device, Tensor};
use kv_cache_scheduler::block_pool::BlockID;

// [num_blocks, block_size, n_kv_heads, head_dim] per layer.
pub struct KvStorage {
    kv_k: Vec<Tensor>,
    kv_v: Vec<Tensor>,
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

    pub fn write(
        &mut self,
        layer: usize,
        block_id: BlockID,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<()> {
        let (n_kv_heads, head_dim) = k.dims2()?;
        let k = k.reshape((1, 1, n_kv_heads, head_dim))?;
        let v = v.reshape((1, 1, n_kv_heads, head_dim))?;
        let block_idx = block_id.0 as usize;
        let ranges = [
            block_idx..block_idx + 1,
            offset..offset + 1,
            0..n_kv_heads,
            0..head_dim,
        ];
        self.kv_k[layer] = self.kv_k[layer].slice_assign(&ranges, &k)?;
        self.kv_v[layer] = self.kv_v[layer].slice_assign(&ranges, &v)?;
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
}
