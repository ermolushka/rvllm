use candle_core::quantized::gguf_file;
use candle_core::{Device, Result, Tensor};

use crate::model::{Config, Layer, Model};

fn metadata<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    match content.metadata.get(key) {
        Some(value) => Ok(value),
        None => candle_core::bail!("missing GGUF metadata key {key:?}"),
    }
}

// llama.cpp's converter reorders each head's Q/K output rows so that RoPE
// rotates *adjacent* pairs (row 2j with row 2j+1). This engine's RoPE uses the
// rotate-half convention (row j with row j + head_dim/2), which is how
// HuggingFace checkpoints are laid out. Undo the reordering at load time:
// within each head, even rows move to the first half and odd rows to the
// second. Without this, attention degrades as the context gets longer.
// `w` is [n_heads * head_dim, in_features].
fn unpermute_rope_rows(w: &Tensor, n_heads: usize) -> Result<Tensor> {
    let (rows, cols) = w.dims2()?;
    let half = rows / n_heads / 2;
    w.reshape((n_heads, half, 2, cols))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((rows, cols))
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
            content
                .tensor(&mut file, name, &device)?
                .dequantize(&device)
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
                    attn_q: unpermute_rope_rows(
                        &load(&format!("blk.{i}.attn_q.weight"))?,
                        config.n_heads as usize,
                    )?,
                    attn_k: unpermute_rope_rows(
                        &load(&format!("blk.{i}.attn_k.weight"))?,
                        config.n_kv_heads as usize,
                    )?,
                    attn_v: load(&format!("blk.{i}.attn_v.weight"))?,
                    attn_output: load(&format!("blk.{i}.attn_output.weight"))?,
                    ffn_norm: load(&format!("blk.{i}.ffn_norm.weight"))?,
                    ffn_gate: load(&format!("blk.{i}.ffn_gate.weight"))?,
                    ffn_up: load(&format!("blk.{i}.ffn_up.weight"))?,
                    ffn_down: load(&format!("blk.{i}.ffn_down.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Model {
            config,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // llama.cpp's convert_hf_to_gguf.py `permute`, on a [rows, cols] matrix.
    fn llama_cpp_permute(w: &Tensor, n_heads: usize) -> Result<Tensor> {
        let (rows, cols) = w.dims2()?;
        w.reshape((n_heads, 2, rows / n_heads / 2, cols))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((rows, cols))
    }

    #[test]
    fn unpermute_inverts_llama_cpp_layout() -> Result<()> {
        // 2 heads x head_dim 4, 3 columns; each row's values identify it.
        let data: Vec<f32> = (0..8).flat_map(|r| [r as f32; 3]).collect();
        let hf = Tensor::from_vec(data, (8, 3), &Device::Cpu)?;
        let gguf = llama_cpp_permute(&hf, 2)?;
        // Per head, GGUF order is [0, 2, 1, 3] of the HF rows.
        let row_ids =
            |t: &Tensor| -> Result<Vec<f32>> { Ok(t.narrow(1, 0, 1)?.flatten_all()?.to_vec1()?) };
        assert_eq!(row_ids(&gguf)?, vec![0., 2., 1., 3., 4., 6., 5., 7.]);
        assert_eq!(row_ids(&unpermute_rope_rows(&gguf, 2)?)?, row_ids(&hf)?);
        Ok(())
    }
}
