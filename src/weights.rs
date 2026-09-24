use candle_core::quantized::gguf_file;
use candle_core::{Device, Result, Tensor};

use crate::model::{Config, Layer, Model};

fn metadata<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    match content.metadata.get(key) {
        Some(value) => Ok(value),
        None => candle_core::bail!("missing GGUF metadata key {key:?}"),
    }
}

impl Model {
    // Reads the model's hyperparameters and every weight tensor from a GGUF
    // file, dequantizing everything to F32 (no quantized matmul yet). Weights
    // stay on the CPU.
    pub fn load(path: &str) -> Result<Model> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        let device = Device::Cpu;
        let mut load = |name: &str| -> Result<Tensor> {
            content.tensor(&mut file, name, &device)?.dequantize(&device)
        };

        let token_embd = load("token_embd.weight")?;
        let output_norm = load("output_norm.weight")?;
        // Models that tie their input and output embeddings ship no separate
        // output matrix; sharing the tensor avoids a second full copy.
        let output = if content.tensor_infos.contains_key("output.weight") {
            load("output.weight")?
        } else {
            token_embd.clone()
        };

        let config = Config {
            n_layers: metadata(&content, "llama.block_count")?.to_u32()?,
            n_heads: metadata(&content, "llama.attention.head_count")?.to_u32()?,
            n_kv_heads: metadata(&content, "llama.attention.head_count_kv")?.to_u32()?,
            hidden_dim: metadata(&content, "llama.embedding_length")?.to_u32()?,
            ffn_dim: metadata(&content, "llama.feed_forward_length")?.to_u32()?,
            rope_theta: metadata(&content, "llama.rope.freq_base")?.to_f32()?,
            rms_eps: metadata(&content, "llama.attention.layer_norm_rms_epsilon")?.to_f32()?,
            vocab_size: token_embd.dim(0)? as u32,
            context_length: metadata(&content, "llama.context_length")?.to_u32()?,
            eos_token_id: metadata(&content, "tokenizer.ggml.eos_token_id")?.to_u32()?,
            bos_token_id: metadata(&content, "tokenizer.ggml.bos_token_id")?.to_u32()?,
        };

        let layers = (0..config.n_layers as usize)
            .map(|i| {
                Ok(Layer {
                    index: i,
                    attn_norm: load(&format!("blk.{i}.attn_norm.weight"))?,
                    attn_q: load(&format!("blk.{i}.attn_q.weight"))?,
                    attn_k: load(&format!("blk.{i}.attn_k.weight"))?,
                    attn_v: load(&format!("blk.{i}.attn_v.weight"))?,
                    attn_output: load(&format!("blk.{i}.attn_output.weight"))?,
                    ffn_norm: load(&format!("blk.{i}.ffn_norm.weight"))?,
                    ffn_gate: load(&format!("blk.{i}.ffn_gate.weight"))?,
                    ffn_up: load(&format!("blk.{i}.ffn_up.weight"))?,
                    ffn_down: load(&format!("blk.{i}.ffn_down.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Model { config, token_embd, layers, output_norm, output })
    }
}
