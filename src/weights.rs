use std::collections::HashMap;

use candle_core::{DType, Tensor, quantized::gguf_file};

#[derive(Debug)]
pub struct SmollLM230MConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub hidden_dim: u32,
    pub ffn_dim: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub tensors: HashMap<String, Tensor>,
}
impl SmollLM230MConfig {
    pub fn read_gguf(
        &mut self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut file = std::fs::File::open(path)?;
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;

        // firstly, we want to fetch metadata values
        for (key, value) in &model.metadata {
            match key.as_str() {
                "llama.embedding_length" => self.hidden_dim = value.to_u32()?,
                "llama.block_count" => self.n_layers = value.to_u32()?,
                "llama.attention.head_count" => self.n_heads = value.to_u32()?,
                "llama.attention.head_count_kv" => self.n_kv_heads = value.to_u32()?,
                "llama.feed_forward_length" => self.ffn_dim = value.to_u32()?,
                "llama.rope.freq_base" => self.rope_theta = value.to_f32()?,
                "llama.attention.layer_norm_rms_epsilon" => self.rms_eps = value.to_f32()?,
                "llama.context_length" => self.context_length = value.to_u32()?,
                "llama.vocab_size" => self.vocab_size = value.to_u32()?,
                "tokenizer.ggml.eos_token_id" => self.eos_token_id = value.to_u32()?,
                "tokenizer.ggml.bos_token_id" => self.bos_token_id = value.to_u32()?,
                _ => {}
            }
        }
        // later, we want to fetch tensors and dequantize them to bf16
        // hardcoding device for now, can reconsider later
        let device = candle_core::Device::Cpu;
        let embd = model
            .tensor(&mut file, "token_embd.weight", &device)?
            .dequantize(&device)?;
        self.tensors.insert("token_embd.weight".to_string(), embd);

        let output_norm_weight = model
            .tensor(&mut file, "output_norm.weight", &device)?
            .dequantize(&device)?;
        self.tensors
            .insert("output_norm.weight".to_string(), output_norm_weight);

        let output_weight_name = if model.tensor_infos.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output_weight = model
            .tensor(&mut file, output_weight_name, &device)?
            .dequantize(&device)?;
        self.tensors
            .insert("output.weight".to_string(), output_weight);

        for i in 0..self.n_layers {
            for suffix in [
                "attn_norm",
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_norm",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                let name = format!("blk.{i}.{suffix}.weight");
                let t = model
                    .tensor(&mut file, &name, &device)?
                    .dequantize(&device)?;
                self.tensors.insert(name, t);
            }
        }

        Ok(())
    }
}
