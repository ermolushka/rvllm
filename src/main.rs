pub mod model;
pub mod tokenizer;
pub mod weights;

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use clap::Parser;
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
    let mut kv_cache: Vec<(Tensor, Tensor)> = (0..config.n_layers)
        .map(|_| {
            Ok((
                Tensor::zeros(
                    (config.n_kv_heads as usize, 0, head_dim),
                    candle_core::DType::F32,
                    &device,
                )?,
                Tensor::zeros(
                    (config.n_kv_heads as usize, 0, head_dim),
                    candle_core::DType::F32,
                    &device,
                )?,
            ))
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    // Greedy decode loop. First step primes the cache with the whole prompt;
    // every step after only feeds the newly generated token, since everything
    // before it is already in kv_cache.
    let mut next_input = ids.clone();
    for _ in 0..args.max_tokens {
        // 1. Run the model on only the tokens not yet cached.
        let logits = model::forward(&config, &next_input, &device, &mut kv_cache)?; // [seq_len, vocab_size]

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
