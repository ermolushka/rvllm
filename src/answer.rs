// The rules for one answer whose pool may run out part way through it.
//
// An answer runs as segments. A segment generates up to the room the pool
// leaves beside the prompt and the answer so far. When a segment ends with the
// pool full, the caller compacts the history, which gives the answer room
// back, and the next segment resumes from the new prompt plus the tokens
// generated so far. Only chat answers can be continued, because only chat
// history can be compacted; other answers are capped at the room they get.
//
// Nothing here touches the engine or the tokenizer. `conversation` (the
// interactive frontends) and the server's handler drive these rules.

use crate::engine::FinishReason;

// How many times one answer may be continued after compaction.
pub const MAX_CONTINUATIONS: usize = 64;

// The room kept free for an answer while the history is compacted: the
// answer's length, but at most a quarter of the pool, so a long answer can't
// squeeze the prompt out. What it outgrows is continued after compaction.
pub fn reserve_for(max_tokens: usize, pool: usize) -> usize {
    max_tokens.min(pool / 4).max(1)
}

// Why an answer stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    // A stop token: the answer is complete.
    Stop,
    // It reached its length, or the room the pool left for it.
    Length,
    // The pool filled while this was the only request running.
    PoolFull,
    // The caller cancelled it.
    Cancelled,
}

impl From<FinishReason> for Ended {
    fn from(finish: FinishReason) -> Self {
        match finish {
            FinishReason::Stop => Ended::Stop,
            FinishReason::Length => Ended::Length,
            FinishReason::PoolFull => Ended::PoolFull,
        }
    }
}

// What to do once a segment has been recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum After {
    // The answer is over, for this reason.
    Finished(Ended),
    // Compact the history, call `rebase`, then run the next segment.
    Compact,
    // The continuation limit was reached; the answer ends where it is.
    Limit,
}

// One answer: the prompt it runs from, the tokens generated so far, and how
// much of it may still be generated.
#[derive(Debug, Clone)]
pub struct Answer {
    pool: usize,
    max_tokens: usize,
    prompt: Vec<u32>,
    ids: Vec<u32>,
    continuations: usize,
    continuable: bool,
}

impl Answer {
    // `pool` is the most one request can hold, prompt included. A continuable
    // answer may be compacted and continued; any other is capped at the room
    // left in the pool.
    pub fn new(pool: usize, max_tokens: usize, prompt: Vec<u32>, continuable: bool) -> Self {
        Answer {
            pool,
            max_tokens,
            prompt,
            ids: Vec::new(),
            continuations: 0,
            continuable,
        }
    }

    // Tokens the answer may still generate.
    pub fn remaining(&self) -> usize {
        self.max_tokens.saturating_sub(self.ids.len())
    }

    // The tokens generated so far, in order.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn continuations(&self) -> usize {
        self.continuations
    }

    // The room to keep free when compacting, for the rest of the answer.
    pub fn reserve(&self) -> usize {
        reserve_for(self.remaining(), self.pool)
    }

    // What the next segment starts from: the prompt, then the answer so far.
    pub fn tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt.clone();
        tokens.extend_from_slice(&self.ids);
        tokens
    }

    // How many tokens the next segment may generate, or None if nothing fits
    // or the answer is complete.
    pub fn next_segment(&self) -> Option<usize> {
        let room = self.pool.saturating_sub(self.prompt.len() + self.ids.len());
        let tokens = self.remaining().min(room);
        (tokens > 0).then_some(tokens)
    }

    // Records what a segment generated, and why it ended.
    pub fn record(&mut self, ids: &[u32], ended: Ended) -> After {
        self.ids.extend_from_slice(ids);
        match ended {
            Ended::Stop | Ended::Cancelled => After::Finished(ended),
            _ if self.remaining() == 0 => After::Finished(Ended::Length),
            _ if !self.continuable => After::Finished(ended),
            // The segment ended for want of room, and the answer isn't done.
            _ if self.continuations == MAX_CONTINUATIONS => After::Limit,
            _ => {
                self.continuations += 1;
                After::Compact
            }
        }
    }

    // Takes `prompt` as the new prompt, once the history has been compacted to
    // it. False if the pool then leaves less than `reserve` beside the prompt
    // and the answer so far, so the answer can't go on.
    pub fn rebase(&mut self, prompt: Vec<u32>) -> bool {
        self.prompt = prompt;
        self.pool.saturating_sub(self.prompt.len() + self.ids.len()) >= self.reserve()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Records `n` generated tokens, which ended the segment with `ended`.
    fn record(answer: &mut Answer, n: usize, ended: Ended) -> After {
        let ids: Vec<u32> = (0..n as u32).collect();
        answer.record(&ids, ended)
    }

    #[test]
    fn reserve_is_the_length_but_at_most_a_quarter_of_the_pool() {
        assert_eq!(reserve_for(256, 4096), 256);
        assert_eq!(reserve_for(2048, 4096), 1024);
        assert_eq!(reserve_for(300, 256), 64);
        assert_eq!(reserve_for(0, 256), 1);
        assert_eq!(reserve_for(10, 0), 1);
    }

    #[test]
    fn a_segment_takes_what_the_pool_leaves_beside_the_prompt() {
        assert_eq!(
            Answer::new(100, 50, vec![0; 30], true).next_segment(),
            Some(50)
        );
        assert_eq!(
            Answer::new(100, 50, vec![0; 80], true).next_segment(),
            Some(20)
        );
        // Exactly the room: the whole room is usable.
        assert_eq!(
            Answer::new(100, 50, vec![0; 50], true).next_segment(),
            Some(50)
        );
    }

    #[test]
    fn no_segment_when_the_prompt_fills_or_overflows_the_pool() {
        assert_eq!(
            Answer::new(100, 50, vec![0; 100], true).next_segment(),
            None
        );
        assert_eq!(
            Answer::new(100, 50, vec![0; 101], true).next_segment(),
            None
        );
        assert_eq!(
            Answer::new(100, 50, Vec::new(), true).next_segment(),
            Some(50)
        );
    }

    #[test]
    fn no_segment_once_the_answer_is_complete() {
        let mut answer = Answer::new(100, 5, vec![0; 10], true);
        assert_eq!(
            record(&mut answer, 5, Ended::Length),
            After::Finished(Ended::Length)
        );
        assert_eq!(answer.remaining(), 0);
        assert_eq!(answer.next_segment(), None);
    }

    #[test]
    fn a_plain_answer_is_capped_at_the_room_it_gets() {
        let mut answer = Answer::new(100, 500, vec![0; 60], false);
        assert_eq!(answer.next_segment(), Some(40));
        assert_eq!(
            record(&mut answer, 40, Ended::Length),
            After::Finished(Ended::Length)
        );
        assert_eq!(answer.continuations(), 0);
    }

    #[test]
    fn a_plain_answer_with_a_full_pool_reports_it() {
        let mut answer = Answer::new(100, 500, vec![0; 60], false);
        assert_eq!(
            record(&mut answer, 40, Ended::PoolFull),
            After::Finished(Ended::PoolFull)
        );
    }

    #[test]
    fn a_stop_ends_the_answer_wherever_it_is() {
        let mut answer = Answer::new(100, 50, vec![0; 10], true);
        assert_eq!(
            record(&mut answer, 2, Ended::Stop),
            After::Finished(Ended::Stop)
        );
        assert_eq!(answer.ids().len(), 2);
    }

    #[test]
    fn cancellation_ends_the_answer() {
        let mut answer = Answer::new(100, 50, vec![0; 10], true);
        assert_eq!(
            record(&mut answer, 0, Ended::Cancelled),
            After::Finished(Ended::Cancelled)
        );
    }

    #[test]
    fn a_segment_capped_by_the_room_asks_for_compaction() {
        let mut answer = Answer::new(100, 100, vec![0; 60], true);
        assert_eq!(answer.next_segment(), Some(40));
        assert_eq!(record(&mut answer, 40, Ended::Length), After::Compact);
        assert_eq!(answer.continuations(), 1);
        assert_eq!(answer.remaining(), 60);
    }

    #[test]
    fn a_full_pool_asks_for_compaction() {
        let mut answer = Answer::new(100, 100, vec![0; 60], true);
        assert_eq!(record(&mut answer, 40, Ended::PoolFull), After::Compact);
    }

    #[test]
    fn a_segment_that_reaches_the_length_is_the_end() {
        let mut answer = Answer::new(100, 40, vec![0; 60], true);
        assert_eq!(
            record(&mut answer, 40, Ended::Length),
            After::Finished(Ended::Length)
        );
        assert_eq!(answer.continuations(), 0);
    }

    #[test]
    fn continuations_stop_at_the_limit_and_do_not_count_past_it() {
        let mut answer = Answer::new(100, 10_000, vec![0; 10], true);
        for _ in 0..MAX_CONTINUATIONS {
            assert_eq!(record(&mut answer, 1, Ended::PoolFull), After::Compact);
            assert!(answer.rebase(vec![0; 10]));
        }
        assert_eq!(answer.continuations(), MAX_CONTINUATIONS);
        assert_eq!(record(&mut answer, 1, Ended::PoolFull), After::Limit);
        assert_eq!(answer.continuations(), MAX_CONTINUATIONS);
    }

    #[test]
    fn rebase_needs_room_for_the_rest_of_the_answer() {
        // Pool 100, answer of 60 with 40 generated: reserve = min(20, 25) = 20.
        let mut answer = Answer::new(100, 60, vec![0; 90], true);
        assert_eq!(record(&mut answer, 40, Ended::PoolFull), After::Compact);
        assert_eq!(answer.reserve(), 20);
        // 100 - 41 - 40 = 19 is one short of the reserve.
        assert!(!answer.rebase(vec![0; 41]));
        // 100 - 40 - 40 = 20 is exactly enough.
        assert!(answer.rebase(vec![0; 40]));
        assert_eq!(answer.next_segment(), Some(20));
    }

    #[test]
    fn rebase_fails_when_the_prompt_alone_fills_the_pool() {
        let mut answer = Answer::new(100, 60, vec![0; 90], true);
        assert_eq!(record(&mut answer, 40, Ended::PoolFull), After::Compact);
        assert!(!answer.rebase(vec![0; 100]));
        assert!(!answer.rebase(vec![0; 150]));
    }

    #[test]
    fn the_tokens_are_the_prompt_then_the_answer_so_far() {
        let mut answer = Answer::new(100, 50, vec![7, 8], true);
        assert_eq!(answer.tokens(), vec![7, 8]);
        record(&mut answer, 3, Ended::Length);
        assert_eq!(answer.tokens(), vec![7, 8, 0, 1, 2]);
    }

    #[test]
    fn the_next_segment_counts_the_answer_so_far_against_the_room() {
        let mut answer = Answer::new(100, 80, vec![0; 50], true);
        assert_eq!(answer.next_segment(), Some(50));
        record(&mut answer, 30, Ended::Length);
        // 50 prompt + 30 answer leaves 20 in the pool, and 50 remain.
        assert_eq!(answer.next_segment(), Some(20));
    }

    #[test]
    fn a_compacted_prompt_gives_the_answer_its_room_back() {
        let mut answer = Answer::new(100, 80, vec![0; 50], true);
        record(&mut answer, 50, Ended::PoolFull);
        assert_eq!(answer.next_segment(), None);
        assert!(answer.rebase(vec![0; 10]));
        // 10 prompt + 50 answer leaves 40 for a remaining 30.
        assert_eq!(answer.next_segment(), Some(30));
    }
}
