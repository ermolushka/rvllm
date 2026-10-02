// CUDA mirror of engine.rs's continuous-batching decode loop: same
// Scheduler-driven admission/prefill/decode structure, but forward passes
// go through CudaModel/CudaKvStorage. The seam PLAN.md's M6 names: logits
// come back from CudaModel as a flat CudaSlice<f32>, copied to host once
// per call, then wrapped per-row as a 1D CPU Tensor so the existing
// Sampler (unchanged, CPU-only) can sample from it exactly as it does on
// the CPU path.
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor};
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::{Scheduler, SequenceId, TokenId};

use crate::model::BatchItem;
use crate::sampler::Sampler;

use super::context::CudaRuntime;
use super::kv_storage::CudaKvStorage;
use super::model::CudaModel;

pub use crate::engine::{RequestSpec, RunResult, RunStats};

struct PendingRequest {
    arrival_step: usize,
    req_idx: usize,
    tokens: Vec<u32>,
}

struct SeqState {
    req_idx: usize,
    prompt_len: usize,
    max_tokens: usize,
    generated: Vec<u32>,
    next_input: u32,
}

struct DecodeRow {
    seq_id: SequenceId,
    token: [u32; 1],
    write: [(BlockID, usize); 1],
    position_offset: usize,
}

// Copies one row of a flat [rows, vocab_size] device buffer already on the
// host back into a 1D CPU Tensor, the shape `Sampler::sample` expects.
fn row_tensor(host: &[f32], row: usize, vocab_size: usize) -> candle_core::Result<Tensor> {
    Tensor::from_slice(
        &host[row * vocab_size..(row + 1) * vocab_size],
        vocab_size,
        &Device::Cpu,
    )
}

pub fn run(
    cuda_runtime: &CudaRuntime,
    model: &CudaModel,
    requests: &[RequestSpec],
    block_size: usize,
    num_blocks: usize,
    sampler: &mut Sampler,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    let config = &model.config;
    let vocab_size = config.vocab_size as usize;

    let mut kv_storage = CudaKvStorage::new(
        cuda_runtime,
        config.n_layers as usize,
        num_blocks,
        block_size,
        config.n_kv_heads as usize,
        config.head_dim(),
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

        while let Some(&seq_id) = scheduler.waiting.front() {
            let needed_blocks = scheduler.sequences[&seq_id]
                .token_ids
                .len()
                .div_ceil(block_size);
            if scheduler.pool.available() < needed_blocks {
                break;
            }

            let token_ids = scheduler.sequences[&seq_id].token_ids.clone();
            let (_matched_tokens, matched_blocks) = scheduler.prefix_cache.match_prefix(&token_ids);
            let skip_tokens = (matched_blocks.len() * block_size).min(token_ids.len() - 1);

            let prefill_started = Instant::now();
            scheduler.prefill(seq_id);

            let seq = &scheduler.sequences[&seq_id];
            let tokens: Vec<u32> = seq.token_ids.iter().map(|t| t.0 as u32).collect();
            let write_positions: Vec<(BlockID, usize)> = (skip_tokens..tokens.len())
                .map(|i| seq.block_table.logical_to_physical(i))
                .collect();

            let item = BatchItem {
                tokens: &tokens[skip_tokens..],
                write_positions: &write_positions,
                read_blocks: seq.block_table.blocks(),
                read_num_tokens: seq.block_table.num_tokens(),
                position_offset: skip_tokens,
            };
            let logits = model.forward_last(cuda_runtime, &[item], &mut kv_storage)?;
            let host = cuda_runtime.stream.clone_dtoh(&logits)?;
            let next_id = sampler.sample(&row_tensor(&host, 0, vocab_size)?)?;
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

        if !scheduler.running.is_empty() {
            let decode_started = Instant::now();
            let running = scheduler.running.clone();
            let position_offsets: HashMap<SequenceId, usize> = running
                .iter()
                .map(|&id| (id, scheduler.sequences[&id].block_table.num_tokens()))
                .collect();

            scheduler.step();

            for &seq_id in &running {
                if !scheduler.running.contains(&seq_id) {
                    let state = &seqs[&seq_id];
                    let real_len = state.prompt_len + state.generated.len();
                    scheduler
                        .sequences
                        .get_mut(&seq_id)
                        .unwrap()
                        .token_ids
                        .truncate(real_len);
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

            let host = {
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
                let logits = model.forward(cuda_runtime, &items, &mut kv_storage)?;
                cuda_runtime.stream.clone_dtoh(&logits)?
            };

            for (i, row) in rows.iter().enumerate() {
                let next_id = sampler.sample(&row_tensor(&host, i, vocab_size)?)?;
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
