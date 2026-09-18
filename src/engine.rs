// Continuous-batching decode loop, shared by the interactive CLI (`main.rs`)
// and the benchmark harness (`bench.rs`). Lifted out of `main.rs` unchanged
// except: (1) parameterized over an arbitrary set of requests instead of
// `Args`, (2) next-token selection goes through `Sampler` instead of a bare
// `argmax`, (3) it now returns per-request outputs plus timing/pool stats
// instead of printing them.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use candle_core::Device;
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::{Scheduler, SequenceId, TokenId};

use crate::kv_storage::KvStorage;
use crate::model::{self, BatchItem};
use crate::sampler::Sampler;
use crate::weights::SmollLM230MConfig;

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

struct RequestState {
    max_tokens: usize,
    generated: Vec<u32>,
    next_input: u32,
}

pub fn run(
    config: &SmollLM230MConfig,
    device: &Device,
    requests: &[RequestSpec],
    block_size: usize,
    num_blocks: usize,
    sampler: &mut Sampler,
) -> candle_core::Result<RunResult> {
    let head_dim = (config.hidden_dim / config.n_heads) as usize;

    let mut kv_storage = KvStorage::new(
        config.n_layers as usize,
        num_blocks,
        block_size,
        config.n_kv_heads as usize,
        head_dim,
        device,
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

    let mut state: HashMap<SequenceId, RequestState> = HashMap::new();
    let mut req_idx_of: HashMap<SequenceId, usize> = HashMap::new();
    let mut orig_prompt_len: HashMap<SequenceId, usize> = HashMap::new();

    let started = Instant::now();
    let mut step_idx = 0usize;
    loop {
        // 1. Admit every request whose arrival step has been reached - just
        //    hands it to the scheduler (-> `waiting`), no blocks allocated yet.
        while pending.front().is_some_and(|r| r.arrival_step <= step_idx) {
            let req = pending.pop_front().unwrap();
            let prompt_len = req.tokens.len();
            let seq_id =
                scheduler.add_request(req.tokens.iter().map(|&t| TokenId(t as i32)).collect());
            req_idx_of.insert(seq_id, req.req_idx);
            orig_prompt_len.insert(seq_id, prompt_len);
        }

        // 2. Admit as many waiting requests as fit into freed/available
        //    capacity, prefilling each with its own (unbatched) forward call.
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

            scheduler.prefill(seq_id);

            let seq = &scheduler.sequences[&seq_id];
            let tokens: Vec<u32> = seq.token_ids.iter().map(|t| t.0 as u32).collect();
            let write_positions: Vec<(BlockID, usize)> = (skip_tokens..tokens.len())
                .map(|i| seq.block_table.logical_to_physical(i))
                .collect();
            let read_blocks = seq.block_table.blocks().to_vec();
            let read_num_tokens = seq.block_table.num_tokens();

            let logits = model::forward(
                config,
                &tokens[skip_tokens..],
                device,
                &mut kv_storage,
                &write_positions,
                &read_blocks,
                read_num_tokens,
                skip_tokens,
            )?; // [prompt_len - skip_tokens, vocab_size]
            let (seq_len, _) = logits.dims2()?;
            let next_id = sampler.sample(&logits.get(seq_len - 1)?)?;

            state
                .entry(seq_id)
                .or_insert_with(|| RequestState {
                    max_tokens: requests[req_idx_of[&seq_id]].max_tokens,
                    generated: Vec::new(),
                    next_input: 0,
                })
                .next_input = next_id;
        }

        if scheduler.running.is_empty() && scheduler.waiting.is_empty() && pending.is_empty() {
            break;
        }

        // 3. Batched decode: every currently running sequence contributes
        //    exactly one new token to a single forward_batch call.
        if !scheduler.running.is_empty() {
            let running = scheduler.running.clone();
            let position_offsets: HashMap<SequenceId, usize> = running
                .iter()
                .map(|&id| (id, scheduler.sequences[&id].block_table.num_tokens()))
                .collect();

            scheduler.step();

            for &seq_id in &running {
                if !scheduler.running.contains(&seq_id) {
                    let real_len = orig_prompt_len[&seq_id] + state[&seq_id].generated.len();
                    scheduler.sequences.get_mut(&seq_id).unwrap().token_ids.truncate(real_len);
                }
            }

            let mut write_positions_by_seq: HashMap<SequenceId, [(BlockID, usize); 1]> =
                HashMap::new();
            let mut still_running = Vec::new();
            for &seq_id in &running {
                if !scheduler.running.contains(&seq_id) {
                    continue;
                }
                let next_input = state[&seq_id].next_input;
                if let Some(t) = scheduler.sequences.get_mut(&seq_id).unwrap().token_ids.last_mut()
                {
                    *t = TokenId(next_input as i32);
                }

                let offset = position_offsets[&seq_id];
                let block_pos =
                    scheduler.sequences[&seq_id].block_table.logical_to_physical(offset);
                write_positions_by_seq.insert(seq_id, [block_pos]);
                still_running.push(seq_id);
            }

            let tokens_by_seq: HashMap<SequenceId, [u32; 1]> =
                still_running.iter().map(|&id| (id, [state[&id].next_input])).collect();
            let read_blocks_by_seq: HashMap<SequenceId, Vec<BlockID>> = still_running
                .iter()
                .map(|&id| (id, scheduler.sequences[&id].block_table.blocks().to_vec()))
                .collect();

            let items: Vec<BatchItem> = still_running
                .iter()
                .map(|&id| BatchItem {
                    tokens: &tokens_by_seq[&id],
                    write_positions: &write_positions_by_seq[&id],
                    read_blocks: &read_blocks_by_seq[&id],
                    read_num_tokens: scheduler.sequences[&id].block_table.num_tokens(),
                    position_offset: position_offsets[&id],
                })
                .collect();

            let logits = model::forward_batch(config, &items, device, &mut kv_storage)?; // [batch, 1, vocab_size]

            for (i, &seq_id) in still_running.iter().enumerate() {
                let next_id = sampler.sample(&logits.get(i)?.get(0)?)?;
                let req = state.get_mut(&seq_id).unwrap();
                req.generated.push(next_id);
                let finished =
                    next_id == config.eos_token_id || req.generated.len() >= req.max_tokens;
                if finished {
                    scheduler.finish_sequence(seq_id);
                } else {
                    req.next_input = next_id;
                }
            }
        }

        step_idx += 1;
    }

    let elapsed = started.elapsed();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); requests.len()];
    let mut output_tokens = 0usize;
    for (seq_id, req_idx) in &req_idx_of {
        let tokens = state[seq_id].generated.clone();
        output_tokens += tokens.len();
        generated[*req_idx] = tokens;
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
        },
    })
}
