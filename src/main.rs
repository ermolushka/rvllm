pub mod model;
pub mod tokenizer;
pub mod weights;
pub mod kv_storage;

use std::collections::HashMap;

use candle_core::Device;
use clap::Parser;
use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::{Scheduler, TokenId};
use kv_storage::KvStorage;
use tokenizer::SmollLM230MTokenizer;
use weights::SmollLM230MConfig;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokenizer: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 20)]
    max_tokens: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    // Defaults to context_length / block_size (rounded up) once the model is loaded.
    #[arg(long)]
    num_blocks: Option<usize>,
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

    let mut ids: Vec<u32> = smoll_tokenizer.encode(&args.prompt)?;

    let head_dim = (config.hidden_dim / config.n_heads) as usize;
    let block_size = args.block_size;
    let num_blocks = args
        .num_blocks
        .unwrap_or_else(|| (config.context_length as usize).div_ceil(block_size));

    let mut kv_storage = KvStorage::new(
        config.n_layers as usize,
        num_blocks,
        block_size,
        config.n_kv_heads as usize,
        head_dim,
        &device,
    )?;

    let mut scheduler = Scheduler::new(num_blocks as u32, block_size);
    let seq_id =
        scheduler.add_request(ids.iter().map(|&t| TokenId(t as i32)).collect());

    // Greedy decode loop. First step primes the cache with the whole prompt
    // (via prefill, which allocates blocks for it); every step after only
    // feeds the newly generated token, since everything before it is already
    // in kv_storage.
    let mut next_input = ids.clone();
    for step_idx in 0..args.max_tokens {
        // Tokens already cached before this call's allocation - 0 on the
        // first (prefill) call.
        let position_offset = scheduler.sequences[&seq_id].block_table.num_tokens();

        if step_idx == 0 {
            scheduler.prefill(seq_id);
        } else {
            // Allocates this call's one new token slot. step() pushes a
            // placeholder TokenId(0) onto the sequence's token history -
            // patch it with the real id we already generated.
            scheduler.step();
            let last = scheduler
                .sequences
                .get_mut(&seq_id)
                .unwrap()
                .token_ids
                .last_mut()
                .unwrap();
            *last = TokenId(next_input[0] as i32);
        }

        let seq = &scheduler.sequences[&seq_id];
        let write_positions: Vec<(BlockID, usize)> = (0..next_input.len())
            .map(|i| seq.block_table.logical_to_physical(position_offset + i))
            .collect();
        let read_blocks = seq.block_table.blocks().to_vec();
        let read_num_tokens = seq.block_table.num_tokens();

        // 1. Run the model on only the tokens not yet cached.
        let logits = model::forward(
            &config,
            &next_input,
            &device,
            &mut kv_storage,
            &write_positions,
            &read_blocks,
            read_num_tokens,
            position_offset,
        )?; // [seq_len, vocab_size]

        // 2. Take the last row of logits (the prediction for the next token).
        let (seq_len, _vocab_size) = logits.dims2()?;
        let last_logits = logits.get(seq_len - 1)?; // [vocab_size]

        // 3. Argmax to get the next token id. candle's Tensor::argmax(dim) on a
        //    1D tensor returns a scalar tensor; pull it out with .to_scalar::<u32>().
        let next_id = last_logits.argmax(0)?.to_scalar::<u32>()?;

        // 4. Append next_id, then stop generating once EOS is produced —
        //    there's no point running `forward` again on a sequence that's
        //    already "finished" from the model's point of view.
        ids.push(next_id);
        if next_id == config.eos_token_id {
            break;
        }
        next_input = vec![next_id];
    }

    // 5. After the loop, decode the full `ids` (or just the newly generated
    //    slice, if you want to print only the completion) and print it.
    let text = smoll_tokenizer.decode(&ids)?;
    println!("{text}");

    Ok(())
}
