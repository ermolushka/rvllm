// LFM2's short-convolution state on the GPU: the Metal counterpart of
// cuda/conv_cache.rs (see conv_cache.rs for why a prefill that reuses cached
// blocks needs the rows a block kept). Copies run as GPU kernels so they stay
// in dispatch order with the forward pass that produced the rows.
use std::collections::HashMap;

use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::SequenceId;

use crate::conv_cache::ConvItem;

use super::context::{MetalBuf, MetalRuntime, Result};

pub(crate) struct MetalConvCache {
    block_size: usize,
    // (sequence, conv layer) -> position of the sequence's next token, and the
    // `kernel - 1` rows before it. Each row is [hidden] floats.
    live: HashMap<(SequenceId, usize), (usize, MetalBuf<f32>)>,
    // (block, conv layer) -> the `kernel` rows ending at the block's last token.
    blocks: HashMap<(BlockID, usize), MetalBuf<f32>>,
}

// A fresh buffer holding `len` floats of `src` starting at `start`.
fn copy_range(rt: &MetalRuntime, src: &MetalBuf<f32>, start: usize, len: usize) -> Result<MetalBuf<f32>> {
    let out = MetalBuf::<f32>::zeros(rt, len);
    rt.copy_f32(src, start, &out, 0, len)?;
    Ok(out)
}

impl MetalConvCache {
    pub(crate) fn new(block_size: usize) -> Self {
        MetalConvCache {
            block_size,
            live: HashMap::new(),
            blocks: HashMap::new(),
        }
    }

    // The `n` rows before `item`'s first token for conv layer `layer`, as
    // [n * hidden] floats: the sequence's own if it stopped right there, zeros
    // at the start of a sequence, else the rows kept by the cached block holding
    // the token before it.
    pub(crate) fn rows_before(
        &self,
        rt: &MetalRuntime,
        layer: usize,
        item: &ConvItem,
        n: usize,
        hidden: usize,
    ) -> Result<MetalBuf<f32>> {
        if let Some((next, rows)) = self.live.get(&(item.seq_id, layer))
            && *next == item.position
        {
            return rt.clone_buf(rows);
        }
        if item.position == 0 {
            return Ok(MetalBuf::zeros(rt, n * hidden));
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
            Some(saved) if item.position + 1 >= end => {
                copy_range(rt, saved, (item.position + 1 - end) * hidden, n * hidden)
            }
            _ => Err(format!(
                "no short-conv state for position {} of sequence {}",
                item.position, item.seq_id.0
            )
            .into()),
        }
    }

    // Keeps conv layer `layer`'s rows after a forward call over `items`.
    // `window` is [batch, n + q_len, hidden]: each item's `n` rows before its
    // first token, then its `q_len` new ones.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update(
        &mut self,
        rt: &MetalRuntime,
        layer: usize,
        items: &[ConvItem],
        window: &MetalBuf<f32>,
        q_len: usize,
        n: usize,
        hidden: usize,
    ) -> Result<()> {
        let item_rows = (n + q_len) * hidden;
        for (i, item) in items.iter().enumerate() {
            let base = i * item_rows;
            let last = copy_range(rt, window, base + q_len * hidden, n * hidden)?;
            self.live
                .insert((item.seq_id, layer), (item.position + q_len, last));
            for (t, &(block, slot)) in item.writes.iter().enumerate() {
                if slot + 1 == self.block_size {
                    let rows = copy_range(rt, window, base + t * hidden, (n + 1) * hidden)?;
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
