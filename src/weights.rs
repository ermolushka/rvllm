use std::collections::HashMap;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::{Device, Result, Tensor};

use crate::model::{Arch, Config, Layer, Model};

// Reads `key` with `get` (e.g. `Value::to_u32`), naming the key if it's
// missing or holds another type.
fn metadata<'a, T>(
    md: &'a HashMap<String, Value>,
    key: &str,
    get: impl FnOnce(&'a Value) -> Result<T>,
) -> Result<T> {
    match md.get(key) {
        Some(value) => get(value).map_err(|e| e.context(format!("GGUF metadata key {key:?}"))),
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

impl Arch {
    fn from_gguf(name: &str) -> Result<Arch> {
        match name {
            "llama" => Ok(Arch::Llama),
            _ => candle_core::bail!("unsupported model architecture {name:?} (supported: llama)"),
        }
    }

    // Whether llama.cpp's converter stored this family's Q/K rows reordered
    // (see `unpermute_rope_rows`). It only does that for families whose RoPE
    // rotates adjacent pairs, like Llama.
    fn qk_permuted(self) -> bool {
        match self {
            Arch::Llama => true,
        }
    }
}

impl Config {
    fn from_gguf_metadata(md: &HashMap<String, Value>, vocab_size: u32) -> Result<Config> {
        let arch_name = metadata(md, "general.architecture", Value::to_string)?;
        let arch = Arch::from_gguf(arch_name)?;
        let key = |name: &str| format!("{arch_name}.{name}");

        let n_heads = metadata(md, &key("attention.head_count"), Value::to_u32)?;
        let n_kv_heads = metadata(md, &key("attention.head_count_kv"), Value::to_u32)?;
        // Grouped-query attention shares each KV head with a whole group of
        // query heads.
        if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            candle_core::bail!("invalid attention head counts: {n_heads} query, {n_kv_heads} KV")
        }
        let hidden_dim = metadata(md, &key("embedding_length"), Value::to_u32)?;
        // Optional in GGUF; llama.cpp also falls back to hidden_dim / n_heads.
        let key_length = key("attention.key_length");
        let head_dim = if md.contains_key(&key_length) {
            metadata(md, &key_length, Value::to_u32)?
        } else if hidden_dim % n_heads == 0 {
            hidden_dim / n_heads
        } else {
            candle_core::bail!(
                "no {key_length:?} and {hidden_dim} isn't divisible by {n_heads} heads"
            )
        };
        // RoPE rotates pairs of dimensions.
        if head_dim == 0 || head_dim % 2 != 0 {
            candle_core::bail!("invalid attention head size {head_dim}: must be even and non-zero")
        }

        Ok(Config {
            arch,
            n_layers: metadata(md, &key("block_count"), Value::to_u32)?,
            n_heads,
            n_kv_heads,
            hidden_dim,
            head_dim,
            ffn_dim: metadata(md, &key("feed_forward_length"), Value::to_u32)?,
            rope_theta: metadata(md, &key("rope.freq_base"), Value::to_f32)?,
            rms_eps: metadata(md, &key("attention.layer_norm_rms_epsilon"), Value::to_f32)?,
            vocab_size,
            context_length: metadata(md, &key("context_length"), Value::to_u32)?,
            eos_token_id: metadata(md, "tokenizer.ggml.eos_token_id", Value::to_u32)?,
            bos_token_id: metadata(md, "tokenizer.ggml.bos_token_id", Value::to_u32)?,
        })
    }
}

impl Model {
    // Reads the model's hyperparameters and every weight tensor from a GGUF
    // file, dequantizing everything to F32 (no quantized matmul yet). Weights
    // stay on the CPU.
    pub fn load(path: &str) -> Result<Model> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        // Parse the config before reading any weights, so an unsupported file
        // fails right away.
        let Some(token_embd_info) = content.tensor_infos.get("token_embd.weight") else {
            candle_core::bail!("missing GGUF tensor \"token_embd.weight\"")
        };
        let config =
            Config::from_gguf_metadata(&content.metadata, token_embd_info.shape.dim(0)? as u32)?;
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

        let layers = (0..config.n_layers as usize)
            .map(|i| {
                let mut attn_q = load(&format!("blk.{i}.attn_q.weight"))?;
                let mut attn_k = load(&format!("blk.{i}.attn_k.weight"))?;
                if config.arch.qk_permuted() {
                    attn_q = unpermute_rope_rows(&attn_q, config.n_heads as usize)?;
                    attn_k = unpermute_rope_rows(&attn_k, config.n_kv_heads as usize)?;
                }
                Ok(Layer {
                    index: i,
                    attn_norm: load(&format!("blk.{i}.attn_norm.weight"))?,
                    attn_q,
                    attn_k,
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
    use candle_core::DType;
    use candle_core::quantized::{GgmlDType, QTensor};

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

    // The keys `Config::from_gguf_metadata` reads, with SmolLM2-360M-Instruct's values.
    fn smollm2_metadata() -> HashMap<String, Value> {
        [
            ("general.architecture", Value::String("llama".into())),
            ("llama.block_count", Value::U32(32)),
            ("llama.attention.head_count", Value::U32(15)),
            ("llama.attention.head_count_kv", Value::U32(5)),
            ("llama.embedding_length", Value::U32(960)),
            ("llama.feed_forward_length", Value::U32(2560)),
            ("llama.rope.freq_base", Value::F32(100000.0)),
            ("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("llama.context_length", Value::U32(8192)),
            ("tokenizer.ggml.eos_token_id", Value::U32(2)),
            ("tokenizer.ggml.bos_token_id", Value::U32(1)),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }

    #[test]
    fn config_from_llama_metadata() -> Result<()> {
        let config = Config::from_gguf_metadata(&smollm2_metadata(), 49152)?;
        assert_eq!(config.arch, Arch::Llama);
        assert_eq!(
            (config.n_layers, config.n_heads, config.n_kv_heads),
            (32, 15, 5)
        );
        // No key_length, so head_dim = 960 / 15.
        assert_eq!(
            (config.hidden_dim, config.head_dim, config.ffn_dim),
            (960, 64, 2560)
        );
        assert_eq!((config.rope_theta, config.rms_eps), (100000.0, 1e-5));
        assert_eq!((config.vocab_size, config.context_length), (49152, 8192));
        assert_eq!((config.eos_token_id, config.bos_token_id), (2, 1));
        Ok(())
    }

    #[test]
    fn key_length_overrides_derived_head_dim() -> Result<()> {
        let mut md = smollm2_metadata();
        md.insert("llama.attention.key_length".into(), Value::U32(128));
        assert_eq!(Config::from_gguf_metadata(&md, 49152)?.head_dim, 128);
        Ok(())
    }

    #[test]
    fn invalid_metadata_is_reported() {
        // Each case replaces (Some) or removes (None) one key of valid metadata
        // and names text the error must contain.
        let cases = [
            ("general.architecture", None, "general.architecture"),
            (
                "general.architecture",
                Some(Value::U32(1)),
                "general.architecture",
            ),
            (
                "general.architecture",
                Some(Value::String("gemma3".into())),
                "unsupported model architecture",
            ),
            ("llama.context_length", None, "llama.context_length"),
            (
                "llama.block_count",
                Some(Value::U64(32)),
                "llama.block_count",
            ),
            (
                "llama.attention.key_length",
                Some(Value::F32(64.)),
                "llama.attention.key_length",
            ),
            (
                "llama.attention.head_count",
                Some(Value::U32(0)),
                "head counts",
            ),
            (
                "llama.attention.head_count_kv",
                Some(Value::U32(0)),
                "head counts",
            ),
            (
                "llama.attention.head_count_kv",
                Some(Value::U32(4)),
                "head counts",
            ),
            ("llama.embedding_length", Some(Value::U32(961)), "divisible"),
            (
                "llama.attention.key_length",
                Some(Value::U32(0)),
                "head size",
            ),
            (
                "llama.attention.key_length",
                Some(Value::U32(63)),
                "head size",
            ),
        ];
        for (key, value, expected) in cases {
            let mut md = smollm2_metadata();
            match value {
                Some(value) => md.insert(key.into(), value),
                None => md.remove(key),
            };
            let err = Config::from_gguf_metadata(&md, 49152)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{key}: {err}");
        }
    }

    // An [n, cols] matrix whose row r is all r, so rows can be told apart.
    fn rows(n: usize, cols: usize) -> Result<Tensor> {
        let data: Vec<f32> = (0..n).flat_map(|r| vec![r as f32; cols]).collect();
        Tensor::from_vec(data, (n, cols), &Device::Cpu)
    }

    // One layer with two query heads and one KV head of size 4: big enough
    // that llama.cpp's Q/K permutation actually moves rows.
    fn tiny_llama_metadata() -> HashMap<String, Value> {
        let mut md = smollm2_metadata();
        for (key, value) in [
            ("llama.block_count", 1),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 1),
            ("llama.embedding_length", 8),
            ("llama.feed_forward_length", 8),
        ] {
            md.insert(key.into(), Value::U32(value));
        }
        md
    }

    // Writes `md` and `tensors` (stored as F32) to a temporary GGUF file and
    // runs `Model::load` on it.
    fn load_gguf(
        name: &str,
        md: &HashMap<String, Value>,
        tensors: &[(&str, Tensor)],
    ) -> Result<Model> {
        let path = std::env::temp_dir().join(format!("rvllm-{}-{name}.gguf", std::process::id()));
        let tensors = tensors
            .iter()
            .map(|(name, t)| Ok((*name, QTensor::quantize(t, GgmlDType::F32)?)))
            .collect::<Result<Vec<_>>>()?;
        let tensors: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (*n, t)).collect();
        let md: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
        gguf_file::write(&mut std::fs::File::create(&path)?, &md, &tensors)?;
        let model = Model::load(path.to_str().unwrap());
        std::fs::remove_file(&path)?;
        model
    }

    #[test]
    fn load_undoes_llama_qk_permutation() -> Result<()> {
        let (q, k) = (rows(8, 8)?, rows(4, 8)?);
        let norm = Tensor::ones(8, DType::F32, &Device::Cpu)?;
        let tensors = [
            ("token_embd.weight", rows(5, 8)?),
            ("output_norm.weight", norm.clone()),
            ("blk.0.attn_norm.weight", norm.clone()),
            ("blk.0.attn_q.weight", llama_cpp_permute(&q, 2)?),
            ("blk.0.attn_k.weight", llama_cpp_permute(&k, 1)?),
            ("blk.0.attn_v.weight", rows(4, 8)?),
            ("blk.0.attn_output.weight", rows(8, 8)?),
            ("blk.0.ffn_norm.weight", norm),
            ("blk.0.ffn_gate.weight", rows(8, 8)?),
            ("blk.0.ffn_up.weight", rows(8, 8)?),
            ("blk.0.ffn_down.weight", rows(8, 8)?),
        ];
        let model = load_gguf("llama", &tiny_llama_metadata(), &tensors)?;
        assert_eq!((model.config.vocab_size, model.config.head_dim), (5, 4));
        let values = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1() };
        assert_eq!(values(&model.layers[0].attn_q)?, values(&q)?);
        assert_eq!(values(&model.layers[0].attn_k)?, values(&k)?);
        Ok(())
    }

    #[test]
    fn load_checks_architecture_before_reading_weights() -> Result<()> {
        let mut md = tiny_llama_metadata();
        md.insert(
            "general.architecture".into(),
            Value::String("gemma3".into()),
        );
        // Every other tensor is missing, so reading any weight would fail differently.
        let Err(err) = load_gguf("gemma3", &md, &[("token_embd.weight", rows(5, 8)?)]) else {
            panic!("loaded a gemma3 file");
        };
        assert!(err.to_string().contains("\"gemma3\""), "{err}");
        Ok(())
    }

    #[test]
    fn load_requires_token_embd() {
        let Err(err) = load_gguf("no-embd", &tiny_llama_metadata(), &[]) else {
            panic!("loaded a file without token_embd.weight");
        };
        assert!(err.to_string().contains("token_embd.weight"), "{err}");
    }
}
