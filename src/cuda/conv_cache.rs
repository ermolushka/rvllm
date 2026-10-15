// LFM2's short-convolution state on the GPU: the device-side counterpart of
// conv_cache.rs, with the same bookkeeping and the same rules (see that file
// for why a prefill that reuses cached blocks needs the rows a block kept),
// but CudaSlices instead of Candle tensors.
use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::SequenceId;

use crate::conv_cache::ConvItem;

use super::context::CudaRuntime;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) struct CudaConvCache {
    block_size: usize,
    // (sequence, conv layer) -> position of the sequence's next token, and
    // the `kernel - 1` rows before it. Each row is [hidden] floats.
    live: HashMap<(SequenceId, usize), (usize, CudaSlice<f32>)>,
    // (block, conv layer) -> the `kernel` rows ending at the block's last token.
    blocks: HashMap<(BlockID, usize), CudaSlice<f32>>,
}

// A fresh device copy of `len` floats of `src` starting at `start`.
fn copy_range(
    runtime: &CudaRuntime,
    src: &CudaSlice<f32>,
    start: usize,
    len: usize,
) -> Result<CudaSlice<f32>> {
    let view = src.slice(start..start + len);
    Ok(runtime.stream.clone_dtod(&view)?)
}

impl CudaConvCache {
    pub(crate) fn new(block_size: usize) -> Self {
        CudaConvCache {
            block_size,
            live: HashMap::new(),
            blocks: HashMap::new(),
        }
    }

    // The `n` rows before `item`'s first token for conv layer `layer`, as
    // [n * hidden] floats on the device: the sequence's own if it stopped right
    // there, zeros at the start of a sequence, else the rows kept by the cached
    // block holding the token before it.
    pub(crate) fn rows_before(
        &self,
        runtime: &CudaRuntime,
        layer: usize,
        item: &ConvItem,
        n: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>> {
        if let Some((next, rows)) = self.live.get(&(item.seq_id, layer))
            && *next == item.position
        {
            return Ok(runtime.stream.clone_dtod(rows)?);
        }
        if item.position == 0 {
            return Ok(runtime.stream.alloc_zeros::<f32>(n * hidden)?);
        }
        let index = (item.position - 1) / self.block_size;
        let end = (index + 1) * self.block_size;
        let saved = item
            .blocks
            .get(index)
            .and_then(|b| self.blocks.get(&(*b, layer)));
        match saved {
            // The block's rows are those of tokens end - n - 1 .. end, so the
            // rows before `position` start at row position + 1 - end.
            Some(saved) if item.position + 1 >= end => copy_range(
                runtime,
                saved,
                (item.position + 1 - end) * hidden,
                n * hidden,
            ),
            _ => Err(format!(
                "no short-conv state for position {} of sequence {}",
                item.position, item.seq_id.0
            )
            .into()),
        }
    }

    // Keeps conv layer `layer`'s rows after a forward call over `items`.
    // `window` is [batch, n + q_len, hidden] on the device: each item's `n`
    // rows before its first token, then its `q_len` new ones.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update(
        &mut self,
        runtime: &CudaRuntime,
        layer: usize,
        items: &[ConvItem],
        window: &CudaSlice<f32>,
        q_len: usize,
        n: usize,
        hidden: usize,
    ) -> Result<()> {
        let item_rows = (n + q_len) * hidden;
        for (i, item) in items.iter().enumerate() {
            let base = i * item_rows;
            let last = copy_range(runtime, window, base + q_len * hidden, n * hidden)?;
            self.live
                .insert((item.seq_id, layer), (item.position + q_len, last));
            for (t, &(block, slot)) in item.writes.iter().enumerate() {
                if slot + 1 == self.block_size {
                    let rows = copy_range(runtime, window, base + t * hidden, (n + 1) * hidden)?;
                    self.blocks.insert((block, layer), rows);
                }
            }
        }
        Ok(())
    }

    // Forgets `seq_id`'s rows, once the engine is done with the sequence.
    pub(crate) fn release(&mut self, seq_id: SequenceId) {
        self.live.retain(|&(seq, _), _| seq != seq_id);
    }
}
