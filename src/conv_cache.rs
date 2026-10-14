// State of LFM2's short-convolution layers between forward calls.
//
// A short-conv layer mixes each token's B*x row with the rows of the
// `kernel - 1` tokens before it (see `model::ShortConv`), so a sequence
// carries those rows from one forward call to the next.
//
// A prompt that reuses cached KV blocks skips the tokens those rows came
// from, so every block a forward call completes also keeps its last `kernel`
// rows. That covers both places a prefill resumes from cached blocks: the end
// of the last one, or one token before it (when the whole prompt is cached,
// the engine recomputes the last token for its logits). The scheduler drops a
// block from its prefix cache before handing it out again, so a block's rows
// are never read for other data; they are replaced when the block is next
// completed.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::SequenceId;

// What the cache needs from one batch item of a forward call.
pub(crate) struct ConvItem {
    pub(crate) seq_id: SequenceId,
    // Position of the item's first new token.
    pub(crate) position: usize,
    // The sequence's block table.
    pub(crate) blocks: Vec<BlockID>,
    // (block, slot) of each new token.
    pub(crate) writes: Vec<(BlockID, usize)>,
}

#[derive(Clone)]
pub(crate) struct ConvCache {
    block_size: usize,
    // (sequence, conv layer) -> position of the sequence's next token, and
    // the `kernel - 1` rows before it.
    live: HashMap<(SequenceId, usize), (usize, Tensor)>,
    // (block, conv layer) -> the `kernel` rows ending at the block's last token.
    blocks: HashMap<(BlockID, usize), Tensor>,
}

impl ConvCache {
    pub(crate) fn new(block_size: usize) -> Self {
        ConvCache {
            block_size,
            live: HashMap::new(),
            blocks: HashMap::new(),
        }
    }

    // The `n` rows before `item`'s first token for conv layer `layer`, as
    // [n, hidden]: the sequence's own if it stopped right there, zeros at the
    // start of a sequence, else those kept by the cached block that holds the
    // token before it.
    pub(crate) fn rows_before(
        &self,
        layer: usize,
        item: &ConvItem,
        n: usize,
        hidden: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if let Some((next, rows)) = self.live.get(&(item.seq_id, layer))
            && *next == item.position
        {
            return Ok(rows.clone());
        }
        if item.position == 0 {
            return Tensor::zeros((n, hidden), DType::F32, device);
        }
        let index = (item.position - 1) / self.block_size;
        let end = (index + 1) * self.block_size;
        let saved = item
            .blocks
            .get(index)
            .and_then(|b| self.blocks.get(&(*b, layer)));
        match saved {
            // The block's rows are those of tokens end - n - 1 .. end.
            Some(saved) if item.position + 1 >= end => saved.narrow(0, item.position + 1 - end, n),
            _ => candle_core::bail!(
                "no short-conv state for position {} of sequence {}",
                item.position,
                item.seq_id.0
            ),
        }
    }

    // Keeps conv layer `layer`'s rows after a forward call over `items`:
    // `rows` is [batch, n + q_len, hidden], each item's `n` rows before its
    // first token followed by its new ones.
    pub(crate) fn update(
        &mut self,
        layer: usize,
        items: &[ConvItem],
        rows: &Tensor,
        n: usize,
    ) -> Result<()> {
        for (i, item) in items.iter().enumerate() {
            let rows = rows.get(i)?;
            let q_len = item.writes.len();
            // `contiguous` copies the rows out of the call's buffer.
            let last = rows.narrow(0, q_len, n)?.contiguous()?;
            self.live
                .insert((item.seq_id, layer), (item.position + q_len, last));
            for (t, &(block, slot)) in item.writes.iter().enumerate() {
                if slot + 1 == self.block_size {
                    let block_rows = rows.narrow(0, t, n + 1)?.contiguous()?;
                    self.blocks.insert((block, layer), block_rows);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    // An item of `seq` writing tokens `from..to` into blocks of 2 slots,
    // block i of the sequence being `BlockID(10 + i)`.
    fn item(seq: i32, from: usize, to: usize) -> ConvItem {
        let block = |position: usize| BlockID(10 + (position / 2) as u32);
        ConvItem {
            seq_id: SequenceId(seq),
            position: from,
            blocks: (0..to.div_ceil(2))
                .map(|b| BlockID(10 + b as u32))
                .collect(),
            writes: (from..to).map(|p| (block(p), p % 2)).collect(),
        }
    }

    // One row per position, holding the position, for one channel: rows for
    // positions `from - 2 .. to` as the cache sees them with a kernel of 3.
    fn rows(from: usize, to: usize) -> Tensor {
        let data: Vec<f32> = (from..to + 2).map(|p| p as f32 - 2.).collect();
        Tensor::from_vec(data, (1, to - from + 2, 1), &Device::Cpu).unwrap()
    }

    #[test]
    fn a_sequence_continues_from_its_own_rows() {
        let mut cache = ConvCache::new(2);
        let rows_at = |cache: &ConvCache, item: &ConvItem| {
            flat(&cache.rows_before(0, item, 2, 1, &Device::Cpu).unwrap())
        };
        assert_eq!(rows_at(&cache, &item(1, 0, 3)), [0., 0.]);
        cache.update(0, &[item(1, 0, 3)], &rows(0, 3), 2).unwrap();
        assert_eq!(rows_at(&cache, &item(1, 3, 4)), [1., 2.]);
        // Another layer, or another sequence at that position, has no rows
        // of its own yet; position 3 is mid-block, so there are no block rows
        // to resume from either.
        assert!(
            cache
                .rows_before(1, &item(1, 3, 4), 2, 1, &Device::Cpu)
                .is_err()
        );
        assert!(
            cache
                .rows_before(0, &item(2, 3, 4), 2, 1, &Device::Cpu)
                .is_err()
        );
        cache.release(SequenceId(1));
        assert!(
            cache
                .rows_before(0, &item(1, 3, 4), 2, 1, &Device::Cpu)
                .is_err()
        );
    }

    #[test]
    fn a_prefill_resumes_from_a_cached_block() {
        let mut cache = ConvCache::new(2);
        // Sequence 1 completes blocks 10 (positions 0, 1) and 11 (2, 3).
        cache.update(0, &[item(1, 0, 5)], &rows(0, 5), 2).unwrap();
        cache.release(SequenceId(1));
        let rows_at =
            |item: &ConvItem| flat(&cache.rows_before(0, item, 2, 1, &Device::Cpu).unwrap());
        // Sequence 2 shares those blocks and resumes at the end of block 11,
        // or one token before it.
        assert_eq!(rows_at(&item(2, 4, 6)), [2., 3.]);
        assert_eq!(rows_at(&item(2, 3, 4)), [1., 2.]);
        assert_eq!(rows_at(&item(2, 2, 4)), [0., 1.]);
    }
}
