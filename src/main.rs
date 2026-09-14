pub mod model;
pub mod tokenizer;
pub mod weights;
pub mod kv_storage;

use std::collections::{HashMap, VecDeque};

use candle_core::Device;
use clap::Parser;
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::{Scheduler, SequenceId, TokenId};
use kv_storage::KvStorage;
use model::BatchItem;
use tokenizer::SmollLM230MTokenizer;
use weights::SmollLM230MConfig;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokenizer: String,
    // One request per occurrence: `--prompt "a" --prompt "b"` runs two
    // concurrent requests through continuous batching.
    #[arg(long)]
    prompt: Vec<String>,
    #[arg(long, default_value_t = 20)]
    max_tokens: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    // Defaults to enough blocks for every prompt to hit `context_length` at
    // once, once the model is loaded.
    #[arg(long)]
    num_blocks: Option<usize>,
    // One entry per `--prompt`, in the same order: the decode step at which
    // that request "arrives" (becomes eligible to be admitted). Requests
    // without a matching entry arrive at step 0. Simulates staggered arrival
    // for exercising the waiting queue and continuous batching.
    #[arg(long)]
    arrival_step: Vec<usize>,
}

// A request that hasn't been handed to the scheduler yet - it's just sitting
// in `pending`, waiting for its `arrival_step` to be reached.
struct PendingRequest {
    arrival_step: usize,
    prompt_idx: usize,
    tokens: Vec<u32>,
}

// Per-sequence bookkeeping the scheduler itself doesn't track: this run's
// generated token ids, how many have been produced, and the next token to
// feed into the following decode step.
struct RequestState {
    max_tokens: usize,
    generated: Vec<u32>,
    next_input: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let device = Device::Cpu;

    let mut config = SmollLM230MConfig {
        n_layers: 0,
        n_heads: 0,
        n_kv_heads: 0,
        hidden_dim: 0,
        ffn_dim: 0,
        rope_theta: 0.0,
        rms_eps: 0.0,
        vocab_size: 0,
        context_length: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        tensors: HashMap::new(),
    };
    config.read_gguf(&args.model)?;

    let smoll_tokenizer =
        SmollLM230MTokenizer::from_file(&args.tokenizer, config.eos_token_id, config.bos_token_id)?;

    let head_dim = (config.hidden_dim / config.n_heads) as usize;
    let block_size = args.block_size;
    let blocks_per_seq = (config.context_length as usize).div_ceil(block_size);
    let num_blocks = args
        .num_blocks
        .unwrap_or_else(|| blocks_per_seq * args.prompt.len().max(1));

    let mut kv_storage = KvStorage::new(
        config.n_layers as usize,
        num_blocks,
        block_size,
        config.n_kv_heads as usize,
        head_dim,
        &device,
    )?;
    let mut scheduler = Scheduler::new(num_blocks as u32, block_size);

    // Every prompt starts as a `PendingRequest`, ordered by arrival step -
    // none of these have touched the scheduler (or a block) yet.
    let mut pending: VecDeque<PendingRequest> = args
        .prompt
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let tokens = smoll_tokenizer.encode(prompt)?;
            let arrival_step = args.arrival_step.get(i).copied().unwrap_or(0);
            Ok(PendingRequest { arrival_step, prompt_idx: i, tokens })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?
        .into_iter()
        .collect();
    pending.make_contiguous().sort_by_key(|r| r.arrival_step);

    let mut state: HashMap<SequenceId, RequestState> = HashMap::new();
    let mut prompt_idx_of: HashMap<SequenceId, usize> = HashMap::new();
    let mut orig_prompt_len: HashMap<SequenceId, usize> = HashMap::new();

    let mut step_idx = 0usize;
    loop {
        // 1. Admit every request whose arrival step has been reached - just
        //    hands it to the scheduler (-> `waiting`), no blocks allocated yet.
        while pending.front().is_some_and(|r| r.arrival_step <= step_idx) {
            let req = pending.pop_front().unwrap();
            let prompt_len = req.tokens.len();
            let seq_id = scheduler
                .add_request(req.tokens.iter().map(|&t| TokenId(t as i32)).collect());
            prompt_idx_of.insert(seq_id, req.prompt_idx);
            orig_prompt_len.insert(seq_id, prompt_len);
        }

        // 2. Admit as many waiting requests as fit into freed/available
        //    capacity, prefilling each with its own (unbatched) forward call -
        //    prompt lengths differ per request, so prefill isn't batched.
        //    FIFO order is preserved, so a request that doesn't fit blocks
        //    everything behind it rather than being skipped over.
        while let Some(&seq_id) = scheduler.waiting.front() {
            // token_ids.len() rather than the original prompt_len: a sequence
            // preempted mid-decode (evicted back to `waiting` by
            // handle_pool_full) carries its already-generated tokens too, and
            // needs blocks for all of them on re-prefill, not just the prompt.
            let needed_blocks = scheduler.sequences[&seq_id]
                .token_ids
                .len()
                .div_ceil(block_size);
            if scheduler.pool.available() < needed_blocks {
                break;
            }

            // Read-only pre-check of what `prefill` is about to reuse from
            // the prefix cache, so we know how many of this prompt's tokens
            // to skip recomputing - `prefill` itself doesn't return this.
            // Only *full* cached blocks are skippable (mirrors `prefill`'s
            // own `cache_block_count * block_size`); clamp below the last
            // token so there's always at least one real forward call left
            // to produce this step's logits, even on a 100% cache hit.
            let token_ids = scheduler.sequences[&seq_id].token_ids.clone();
            let (_matched_tokens, matched_blocks) = scheduler.prefix_cache.match_prefix(&token_ids);
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
                &config,
                &tokens[skip_tokens..],
                &device,
                &mut kv_storage,
                &write_positions,
                &read_blocks,
                read_num_tokens,
                skip_tokens,
            )?; // [prompt_len - skip_tokens, vocab_size]
            let (seq_len, _) = logits.dims2()?;
            let next_id = logits.get(seq_len - 1)?.argmax(0)?.to_scalar::<u32>()?;

            state
                .entry(seq_id)
                .or_insert_with(|| RequestState {
                    max_tokens: args.max_tokens,
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

            // Allocates each running sequence's next token slot. May evict a
            // sequence back to `waiting` under pool pressure (handled below -
            // it just sits out this decode step and gets a fresh prefill,
            // recomputing its KV from `token_ids`, once capacity frees up).
            scheduler.step();

            // `step()` unconditionally pushes a placeholder TokenId(0) onto a
            // sequence's token_ids *before* it's known whether that sequence
            // (or some other running sequence, while handling pool pressure)
            // ends up evicted this call. Any sequence that was running before
            // `step()` but isn't after it is exactly one such stray
            // placeholder ahead of its real history - drop it, or the next
            // re-prefill would replay a bogus token into that sequence's
            // context.
            for &seq_id in &running {
                if !scheduler.running.contains(&seq_id) {
                    let real_len = orig_prompt_len[&seq_id] + state[&seq_id].generated.len();
                    scheduler
                        .sequences
                        .get_mut(&seq_id)
                        .unwrap()
                        .token_ids
                        .truncate(real_len);
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
                if let Some(t) = scheduler
                    .sequences
                    .get_mut(&seq_id)
                    .unwrap()
                    .token_ids
                    .last_mut()
                {
                    *t = TokenId(next_input as i32);
                }

                let offset = position_offsets[&seq_id];
                let block_pos = scheduler.sequences[&seq_id]
                    .block_table
                    .logical_to_physical(offset);
                write_positions_by_seq.insert(seq_id, [block_pos]);
                still_running.push(seq_id);
            }

            let tokens_by_seq: HashMap<SequenceId, [u32; 1]> = still_running
                .iter()
                .map(|&id| (id, [state[&id].next_input]))
                .collect();
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

            let logits = model::forward_batch(&config, &items, &device, &mut kv_storage)?; // [batch, 1, vocab_size]

            for (i, &seq_id) in still_running.iter().enumerate() {
                let next_id = logits.get(i)?.get(0)?.argmax(0)?.to_scalar::<u32>()?;
                let req = state.get_mut(&seq_id).unwrap();
                req.generated.push(next_id);
                let finished = next_id == config.eos_token_id || req.generated.len() >= req.max_tokens;
                if finished {
                    scheduler.finish_sequence(seq_id);
                } else {
                    req.next_input = next_id;
                }
            }
        }

        step_idx += 1;
    }

    let mut results: Vec<(usize, &Vec<u32>)> = prompt_idx_of
        .iter()
        .map(|(seq_id, &prompt_idx)| (prompt_idx, &state[seq_id].generated))
        .collect();
    results.sort_by_key(|(prompt_idx, _)| *prompt_idx);
    for (i, generated) in results {
        let text = smoll_tokenizer.decode(generated)?;
        println!("[{i}] prompt: {:?}", args.prompt[i]);
        println!("[{i}] completion: {text}");
    }

    println!(
        "prefix cache: {} / {} tokens hit ({:.1}%), {} total block allocations",
        scheduler.metrics.prefix_hits,
        scheduler.metrics.prefix_total,
        scheduler.metrics.hit_rate() * 100.0,
        scheduler.pool.total_allocs,
    );

    Ok(())
}
