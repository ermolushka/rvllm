// Continuous-batching decode loop, shared by the interactive CLI (`main.rs`)
// and the benchmark harness (`bench.rs`). Lifted out of `main.rs` unchanged
// except: (1) parameterized over an arbitrary set of requests instead of
// `Args`, (2) next-token selection goes through `Sampler` instead of a bare
// `argmax`, (3) it now returns per-request outputs plus timing/pool stats
// instead of printing them.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use candle_core::Result;
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::{Scheduler, SequenceId, TokenId};

use crate::kv_storage::KvStorage;
use crate::model::{BatchItem, Model};
use crate::sampler::Sampler;

pub struct RequestSpec {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    // Decode step at which this request becomes eligible for admission.
    pub arrival_step: usize,
}

#[derive(Default)]
pub struct RunStats {
    pub steps: usize,
    pub elapsed: Duration,
    pub output_tokens: usize,
    pub prefix_hits: usize,
    pub prefix_total: usize,
    pub total_block_allocs: usize,
    // Wall time spent inside prefill forward calls / batched decode forward
    // calls (incl. sampling), and how many tokens each processed.
    pub prefill_time: Duration,
    pub prefill_tokens: usize,
    pub decode_time: Duration,
    pub decode_steps: usize,
    // Per-request time-to-first-token: prefill start -> first sampled token.
    // Same order/length as the input `requests` slice.
    pub ttft: Vec<Duration>,
}

pub struct RunResult {
    // Same order/length as the input `requests` slice.
    pub generated: Vec<Vec<u32>>,
    pub stats: RunStats,
}

struct PendingRequest {
    arrival_step: usize,
    req_idx: usize,
    tokens: Vec<u32>,
}

// Everything the engine tracks per sequence, keyed by the scheduler's id.
struct SeqState {
    // Index into the caller's `requests` slice.
    req_idx: usize,
    // Length of the original prompt, before any generated tokens were appended.
    prompt_len: usize,
    max_tokens: usize,
    generated: Vec<u32>,
    // Token to feed the next decode step: the last one sampled.
    next_input: u32,
}

// One running sequence's contribution to a decode step. Owned here so the
// `BatchItem`s can borrow from it; the block table is borrowed straight from
// the scheduler instead of copied.
struct DecodeRow {
    seq_id: SequenceId,
    token: [u32; 1],
    write: [(BlockID, usize); 1],
    position_offset: usize,
}

pub fn run(
    model: &Model,
    requests: &[RequestSpec],
    block_size: usize,
    num_blocks: usize,
    sampler: &mut Sampler,
) -> Result<RunResult> {
    let config = &model.config;

    let mut kv_storage = KvStorage::new(
        config.n_layers as usize,
        num_blocks,
        block_size,
        config.n_kv_heads as usize,
        config.head_dim(),
        model.device(),
    )?;
    let mut scheduler = Scheduler::new(num_blocks as u32, block_size);

    let mut pending: VecDeque<PendingRequest> = requests
        .iter()
        .enumerate()
        .map(|(i, r)| PendingRequest {
            arrival_step: r.arrival_step,
            req_idx: i,
            tokens: r.tokens.clone(),
        })
        .collect();
    pending.make_contiguous().sort_by_key(|r| r.arrival_step);

    let mut seqs: HashMap<SequenceId, SeqState> = HashMap::new();

    let started = Instant::now();
    let mut step_idx = 0usize;
    let mut prefill_time = Duration::ZERO;
    let mut prefill_tokens = 0usize;
    let mut decode_time = Duration::ZERO;
    let mut decode_steps = 0usize;
    let mut ttft = vec![Duration::ZERO; requests.len()];
    loop {
        // 1. Admit every request whose arrival step has been reached - just
        //    hands it to the scheduler (-> `waiting`), no blocks allocated yet.
        while pending.front().is_some_and(|r| r.arrival_step <= step_idx) {
            let req = pending.pop_front().unwrap();
            let prompt_len = req.tokens.len();
            let seq_id =
                scheduler.add_request(req.tokens.iter().map(|&t| TokenId(t as i32)).collect());
            seqs.insert(
                seq_id,
                SeqState {
                    req_idx: req.req_idx,
                    prompt_len,
                    max_tokens: requests[req.req_idx].max_tokens,
                    generated: Vec::new(),
                    next_input: 0,
                },
            );
        }

        // 2. Admit as many waiting requests as fit into freed/available
        //    capacity, prefilling each with its own forward call.
        while let Some(&seq_id) = scheduler.waiting.front() {
            let needed_blocks =
                scheduler.sequences[&seq_id].token_ids.len().div_ceil(block_size);
            if scheduler.pool.available() < needed_blocks {
                break;
            }

            let token_ids = scheduler.sequences[&seq_id].token_ids.clone();
            let (_matched_tokens, matched_blocks) =
                scheduler.prefix_cache.match_prefix(&token_ids);
            let skip_tokens = (matched_blocks.len() * block_size).min(token_ids.len() - 1);

            let prefill_started = Instant::now();
            scheduler.prefill(seq_id);

            let seq = &scheduler.sequences[&seq_id];
            let tokens: Vec<u32> = seq.token_ids.iter().map(|t| t.0 as u32).collect();
            let write_positions: Vec<(BlockID, usize)> = (skip_tokens..tokens.len())
                .map(|i| seq.block_table.logical_to_physical(i))
                .collect();

            // Only the tokens past the cached prefix run through the model,
            // but attention reads the sequence's whole block table.
            let item = BatchItem {
                tokens: &tokens[skip_tokens..],
                write_positions: &write_positions,
                read_blocks: seq.block_table.blocks(),
                read_num_tokens: seq.block_table.num_tokens(),
                position_offset: skip_tokens,
            };
            let logits = model.forward_last(&[item], &mut kv_storage)?; // [1, 1, vocab_size]
            let next_id = sampler.sample(&logits.get(0)?.get(0)?)?;
            let prefill_elapsed = prefill_started.elapsed();
            prefill_time += prefill_elapsed;
            prefill_tokens += tokens.len() - skip_tokens;

            let state = seqs.get_mut(&seq_id).unwrap();
            ttft[state.req_idx] = prefill_elapsed;
            state.next_input = next_id;
        }

        if scheduler.running.is_empty() && scheduler.waiting.is_empty() && pending.is_empty() {
            break;
        }

        // 3. Batched decode: every currently running sequence contributes
        //    exactly one new token to a single forward call.
        if !scheduler.running.is_empty() {
            let decode_started = Instant::now();
            let running = scheduler.running.clone();
            let position_offsets: HashMap<SequenceId, usize> = running
                .iter()
                .map(|&id| (id, scheduler.sequences[&id].block_table.num_tokens()))
                .collect();

            scheduler.step();

            // A sequence that dropped out of `running` was preempted (its
            // blocks evicted): rewind its token list to the real tokens so
            // it re-prefills cleanly when readmitted.
            for &seq_id in &running {
                if !scheduler.running.contains(&seq_id) {
                    let state = &seqs[&seq_id];
                    let real_len = state.prompt_len + state.generated.len();
                    scheduler.sequences.get_mut(&seq_id).unwrap().token_ids.truncate(real_len);
                }
            }

            let mut rows: Vec<DecodeRow> = Vec::with_capacity(running.len());
            for &seq_id in &running {
                if !scheduler.running.contains(&seq_id) {
                    continue;
                }
                let next_input = seqs[&seq_id].next_input;
                let seq = scheduler.sequences.get_mut(&seq_id).unwrap();
                if let Some(t) = seq.token_ids.last_mut() {
                    *t = TokenId(next_input as i32);
                }
                let position_offset = position_offsets[&seq_id];
                rows.push(DecodeRow {
                    seq_id,
                    token: [next_input],
                    write: [seq.block_table.logical_to_physical(position_offset)],
                    position_offset,
                });
            }

            let logits = {
                let items: Vec<BatchItem> = rows
                    .iter()
                    .map(|row| {
                        let table = &scheduler.sequences[&row.seq_id].block_table;
                        BatchItem {
                            tokens: &row.token,
                            write_positions: &row.write,
                            read_blocks: table.blocks(),
                            read_num_tokens: table.num_tokens(),
                            position_offset: row.position_offset,
                        }
                    })
                    .collect();
                model.forward(&items, &mut kv_storage)? // [batch, 1, vocab_size]
            };

            for (i, row) in rows.iter().enumerate() {
                let next_id = sampler.sample(&logits.get(i)?.get(0)?)?;
                let req = seqs.get_mut(&row.seq_id).unwrap();
                req.generated.push(next_id);
                let finished =
                    next_id == config.eos_token_id || req.generated.len() >= req.max_tokens;
                if finished {
                    scheduler.finish_sequence(row.seq_id);
                } else {
                    req.next_input = next_id;
                }
            }
            decode_time += decode_started.elapsed();
            decode_steps += 1;
        }

        step_idx += 1;
    }

    let elapsed = started.elapsed();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); requests.len()];
    let mut output_tokens = 0usize;
    for state in seqs.into_values() {
        output_tokens += state.generated.len();
        generated[state.req_idx] = state.generated;
    }

    Ok(RunResult {
        generated,
        stats: RunStats {
            steps: step_idx,
            elapsed,
            output_tokens,
            prefix_hits: scheduler.metrics.prefix_hits,
            prefix_total: scheduler.metrics.prefix_total,
            total_block_allocs: scheduler.pool.total_allocs,
            prefill_time,
            prefill_tokens,
            decode_time,
            decode_steps,
            ttft,
        },
    })
}
