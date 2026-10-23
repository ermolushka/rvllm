use std::collections::HashMap;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::{Device, Result, Tensor};

use crate::model::{
    Activation, Arch, Attention, Config, ConvLayers, Ffn, Layer, LocalAttention, Mixer, Model,
    ShortConv,
};

// Reads `key` with `get` (e.g. `Value::to_u32`) if present, naming the key if
// it holds another type.
fn optional_metadata<'a, T>(
    md: &'a HashMap<String, Value>,
    key: &str,
    get: impl FnOnce(&'a Value) -> Result<T>,
) -> Result<Option<T>> {
    match md.get(key) {
        Some(value) => get(value)
            .map(Some)
            .map_err(|e| e.context(format!("GGUF metadata key {key:?}"))),
        None => Ok(None),
    }
}

// Like `optional_metadata`, for keys the model can't do without.
fn metadata<'a, T>(
    md: &'a HashMap<String, Value>,
    key: &str,
    get: impl FnOnce(&'a Value) -> Result<T>,
) -> Result<T> {
    match optional_metadata(md, key, get)? {
        Some(value) => Ok(value),
        None => candle_core::bail!("missing GGUF metadata key {key:?}"),
    }
}

// The (architecture, layers, width) of each model the README lists as tested.
const TESTED_SHAPES: &[(Arch, u32, u32)] = &[
    (Arch::Llama, 32, 960),  // SmolLM2-360M, base and Instruct
    (Arch::Llama, 16, 2048), // Llama-3.2-1B-Instruct
    (Arch::Qwen2, 24, 896),  // Qwen2.5-0.5B-Instruct
    (Arch::Qwen3, 28, 1024), // Qwen3-0.6B
    (Arch::Gemma3, 18, 640), // gemma-3-270m-it
    (Arch::Lfm2, 16, 2048),  // LFM2.5-1.2B-Instruct
];

// Whether the model has the shape of one the README lists as tested. Shapes
// are the only thing to compare: `general.name` varies between conversions.
fn is_tested_shape(config: &Config) -> bool {
    TESTED_SHAPES.contains(&(config.arch, config.n_layers, config.hidden_dim))
}

// The RoPE scaling the metadata declares, from `{arch}.rope.scaling.type`.
fn rope_scaling_type(md: &HashMap<String, Value>) -> Result<Option<&str>> {
    let arch = metadata(md, "general.architecture", Value::to_string)?;
    let key = format!("{arch}.rope.scaling.type");
    Ok(optional_metadata(md, &key, Value::to_string)?.map(String::as_str))
}

// A count from a GGUF integer array: llama.cpp writes per-layer values as
// I32, single ones as U32.
fn count(value: &Value) -> Result<u32> {
    match value {
        Value::I32(n) => {
            u32::try_from(*n).map_err(|_| candle_core::Error::Msg(format!("negative count {n}")))
        }
        _ => value.to_u32(),
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
            "qwen2" => Ok(Arch::Qwen2),
            "qwen3" => Ok(Arch::Qwen3),
            "gemma3" => Ok(Arch::Gemma3),
            "lfm2" => Ok(Arch::Lfm2),
            _ => candle_core::bail!(
                "unsupported model architecture {name:?} (supported: llama, qwen2, qwen3, gemma3, lfm2)"
            ),
        }
    }

    // Whether llama.cpp's converter stored this family's Q/K rows reordered
    // (see `unpermute_rope_rows`). It only does that for families whose RoPE
    // rotates adjacent pairs, like Llama.
    fn qk_permuted(self) -> bool {
        match self {
            Arch::Llama => true,
            Arch::Qwen2 | Arch::Qwen3 | Arch::Gemma3 | Arch::Lfm2 => false,
        }
    }

    // Whether the Q/K/V projections have biases.
    fn qkv_bias(self) -> bool {
        matches!(self, Arch::Qwen2)
    }

    // Whether each Q and K head is RMS-normalized before RoPE.
    fn qk_norm(self) -> bool {
        matches!(self, Arch::Qwen3 | Arch::Gemma3 | Arch::Lfm2)
    }

    // Whether the attention and FFN outputs are RMS-normalized before the
    // residual adds.
    fn post_norms(self) -> bool {
        matches!(self, Arch::Gemma3)
    }

    // Whether layer `i` is a local layer: Gemma 3 follows every five of them
    // with one global layer.
    fn local_layer(self, i: usize) -> bool {
        matches!(self, Arch::Gemma3) && i % 6 < 5
    }

    fn activation(self) -> Activation {
        match self {
            Arch::Llama | Arch::Qwen2 | Arch::Qwen3 | Arch::Lfm2 => Activation::Silu,
            Arch::Gemma3 => Activation::GeluTanh,
        }
    }

    // The final norm's tensor. LFM2's is named after the embeddings (HF's
    // `embedding_norm`), though it comes after the last layer.
    fn output_norm(self) -> &'static str {
        match self {
            Arch::Lfm2 => "token_embd_norm.weight",
            Arch::Llama | Arch::Qwen2 | Arch::Qwen3 | Arch::Gemma3 => "output_norm.weight",
        }
    }
}

impl Config {
    fn from_gguf_metadata(md: &HashMap<String, Value>, vocab_size: u32) -> Result<Config> {
        let arch_name = metadata(md, "general.architecture", Value::to_string)?;
        let arch = Arch::from_gguf(arch_name)?;
        let key = |name: &str| format!("{arch_name}.{name}");

        let n_layers = metadata(md, &key("block_count"), Value::to_u32)?;
        let n_heads = metadata(md, &key("attention.head_count"), Value::to_u32)?;
        // LFM2 gives the KV head count per layer, 0 for its short-conv layers.
        let kv_key = key("attention.head_count_kv");
        let kv_heads = match md.get(&kv_key) {
            Some(Value::Array(per_layer)) => per_layer
                .iter()
                .map(count)
                .collect::<Result<Vec<_>>>()
                .map_err(|e| e.context(format!("GGUF metadata key {kv_key:?}")))?,
            _ => vec![metadata(md, &kv_key, Value::to_u32)?; n_layers as usize],
        };
        if kv_heads.len() != n_layers as usize {
            candle_core::bail!(
                "{kv_key:?} has {} entries for {n_layers} layers",
                kv_heads.len()
            )
        }
        let conv_layers: Vec<usize> = match arch {
            Arch::Lfm2 => (0..kv_heads.len()).filter(|&i| kv_heads[i] == 0).collect(),
            Arch::Llama | Arch::Qwen2 | Arch::Qwen3 | Arch::Gemma3 => Vec::new(),
        };
        let mut attention_kv_heads = (0..kv_heads.len())
            .filter(|i| !conv_layers.contains(i))
            .map(|i| kv_heads[i]);
        let Some(n_kv_heads) = attention_kv_heads.next() else {
            candle_core::bail!("no attention layers")
        };
        if attention_kv_heads.any(|n| n != n_kv_heads) {
            candle_core::bail!("attention layers with different KV head counts aren't supported")
        }
        // Grouped-query attention shares each KV head with a whole group of
        // query heads.
        if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            candle_core::bail!("invalid attention head counts: {n_heads} query, {n_kv_heads} KV")
        }
        let hidden_dim = metadata(md, &key("embedding_length"), Value::to_u32)?;
        // Optional in GGUF; llama.cpp also falls back to hidden_dim / n_heads.
        let key_length = key("attention.key_length");
        let head_dim = match optional_metadata(md, &key_length, Value::to_u32)? {
            Some(head_dim) => head_dim,
            None if hidden_dim % n_heads == 0 => hidden_dim / n_heads,
            None => candle_core::bail!(
                "no {key_length:?} and {hidden_dim} isn't divisible by {n_heads} heads"
            ),
        };
        // RoPE rotates pairs of dimensions.
        if head_dim == 0 || head_dim % 2 != 0 {
            candle_core::bail!("invalid attention head size {head_dim}: must be even and non-zero")
        }

        let tokenizer_model = optional_metadata(md, "tokenizer.ggml.model", Value::to_string)?;
        let tokenizer_pre = optional_metadata(md, "tokenizer.ggml.pre", Value::to_string)?.cloned();
        // When the GGUF doesn't say, follow llama.cpp's defaults: SentencePiece
        // vocabularies and Llama 3's BPE tokenizer start prompts with BOS.
        let add_bos = optional_metadata(md, "tokenizer.ggml.add_bos_token", Value::to_bool)?
            .unwrap_or(
                tokenizer_model.is_some_and(|model| model == "llama")
                    || matches!(
                        tokenizer_pre.as_deref(),
                        Some("llama-bpe" | "llama3" | "llama-v3")
                    ),
            );
        let conv = if conv_layers.is_empty() {
            None
        } else {
            let kernel = metadata(md, &key("shortconv.l_cache"), Value::to_u32)?;
            if kernel == 0 {
                candle_core::bail!("invalid short-conv kernel size 0")
            }
            Some(ConvLayers {
                layers: conv_layers,
                kernel: kernel as usize,
            })
        };

        // Gemma 3's local layers; llama.cpp defaults their RoPE base to 10000.
        let local_attention = match arch {
            Arch::Gemma3 => {
                let window = metadata(md, &key("attention.sliding_window"), Value::to_u32)?;
                if window == 0 {
                    candle_core::bail!("invalid sliding window of 0 positions")
                }
                Some(LocalAttention {
                    window: window as usize,
                    rope_theta: optional_metadata(md, &key("rope.freq_base_swa"), Value::to_f32)?
                        .unwrap_or(10000.0),
                })
            }
            Arch::Llama | Arch::Qwen2 | Arch::Qwen3 | Arch::Lfm2 => None,
        };

        let eos_token_id = metadata(md, "tokenizer.ggml.eos_token_id", Value::to_u32)?;
        let bos_token_id = metadata(md, "tokenizer.ggml.bos_token_id", Value::to_u32)?;
        let tokens = optional_metadata(md, "tokenizer.ggml.tokens", Value::to_vec)?;
        let token = |id: u32| {
            tokens
                .and_then(|tokens| tokens.get(id as usize))
                .and_then(|token| token.to_string().ok())
                .cloned()
        };

        // Only Llama 3's frequency factors are implemented, and they arrive as a
        // tensor (see `Model::load`). Any other scaling would silently run
        // unscaled RoPE, so refuse it instead.
        match rope_scaling_type(md)? {
            None | Some("none") | Some("llama3") => {}
            Some(other) => candle_core::bail!(
                "unsupported RoPE scaling {other:?} in {arch_name} (only llama3 is implemented)"
            ),
        }

        Ok(Config {
            arch,
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            head_dim,
            ffn_dim: metadata(md, &key("feed_forward_length"), Value::to_u32)?,
            rope_theta: metadata(md, &key("rope.freq_base"), Value::to_f32)?,
            rope_factors: None,
            local_attention,
            conv,
            rms_eps: metadata(md, &key("attention.layer_norm_rms_epsilon"), Value::to_f32)?,
            vocab_size,
            context_length: metadata(md, &key("context_length"), Value::to_u32)?,
            eos_token_id,
            bos_token_id,
            add_bos,
            tokenizer_pre,
            bos_token: token(bos_token_id),
            eos_token: token(eos_token_id),
            embedding_scale: (arch == Arch::Gemma3).then(|| (hidden_dim as f32).sqrt()),
            activation: arch.activation(),
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
        let mut config =
            Config::from_gguf_metadata(&content.metadata, token_embd_info.shape.dim(0)? as u32)?;
        // Other sizes of a supported family load, since their shapes are checked
        // below, but nothing has been verified for them, so say so.
        if !is_tested_shape(&config) {
            let arch_name = metadata(&content.metadata, "general.architecture", Value::to_string)?;
            eprintln!(
                "warning: this {arch_name} model ({} layers, width {}) is not one of rvllm's tested models; its output is unverified",
                config.n_layers, config.hidden_dim
            );
        }
        let device = Device::Cpu;
        // Checking each weight's shape against the config makes a file whose
        // tensors and metadata disagree fail here, not on the first request.
        let mut load = |name: &str, dims: &[usize]| -> Result<Tensor> {
            let tensor = content.tensor(&mut file, name, &device)?;
            if tensor.shape().dims() != dims {
                candle_core::bail!(
                    "GGUF tensor {name:?} has shape {:?}, expected {dims:?}",
                    tensor.shape().dims()
                )
            }
            tensor.dequantize(&device)
        };
        let hidden_dim = config.hidden_dim as usize;
        let vocab_size = config.vocab_size as usize;
        let ffn_dim = config.ffn_dim as usize;
        let q_dim = config.n_heads as usize * config.head_dim();
        let kv_dim = config.n_kv_heads as usize * config.head_dim();

        let token_embd = load("token_embd.weight", &[vocab_size, hidden_dim])?;
        let output_norm = load(config.arch.output_norm(), &[hidden_dim])?;
        // Models that tie their input and output embeddings ship no separate
        // output matrix; sharing the tensor avoids a second full copy.
        let output = if content.tensor_infos.contains_key("output.weight") {
            load("output.weight", &[vocab_size, hidden_dim])?
        } else {
            token_embd.clone()
        };
        // Llama 3 ships its RoPE scaling as a tensor rather than metadata.
        if content.tensor_infos.contains_key("rope_freqs.weight") {
            let factors = load("rope_freqs.weight", &[config.head_dim() / 2])?;
            config.rope_factors = Some(factors.to_vec1()?);
        } else if rope_scaling_type(&content.metadata)? == Some("llama3") {
            candle_core::bail!(
                "the GGUF declares llama3 RoPE scaling but has no rope_freqs.weight tensor"
            );
        }

        let conv_layers = config
            .conv
            .as_ref()
            .map_or(&[][..], |conv| &conv.layers[..]);
        let kernel = config.conv.as_ref().map_or(0, |conv| conv.kernel);
        let layers = (0..config.n_layers as usize)
            .map(|i| {
                let mixer = match conv_layers.iter().position(|&layer| layer == i) {
                    Some(conv_index) => Mixer::ShortConv(ShortConv {
                        conv_index,
                        in_proj: load(
                            &format!("blk.{i}.shortconv.in_proj.weight"),
                            &[3 * hidden_dim, hidden_dim],
                        )?,
                        conv: load(
                            &format!("blk.{i}.shortconv.conv.weight"),
                            &[hidden_dim, kernel],
                        )?,
                        out_proj: load(
                            &format!("blk.{i}.shortconv.out_proj.weight"),
                            &[hidden_dim, hidden_dim],
                        )?,
                    }),
                    None => {
                        let mut q = load(&format!("blk.{i}.attn_q.weight"), &[q_dim, hidden_dim])?;
                        let mut k = load(&format!("blk.{i}.attn_k.weight"), &[kv_dim, hidden_dim])?;
                        if config.arch.qk_permuted() {
                            q = unpermute_rope_rows(&q, config.n_heads as usize)?;
                            k = unpermute_rope_rows(&k, config.n_kv_heads as usize)?;
                        }
                        let bias = config.arch.qkv_bias();
                        let qk_norm = config.arch.qk_norm();
                        let head_dim = config.head_dim();
                        Mixer::Attention(Attention {
                            // Only attention layers have KV slots.
                            kv_index: i - conv_layers.iter().filter(|&&layer| layer < i).count(),
                            local: config.arch.local_layer(i),
                            q,
                            k,
                            v: load(&format!("blk.{i}.attn_v.weight"), &[kv_dim, hidden_dim])?,
                            o: load(&format!("blk.{i}.attn_output.weight"), &[hidden_dim, q_dim])?,
                            q_bias: bias
                                .then(|| load(&format!("blk.{i}.attn_q.bias"), &[q_dim]))
                                .transpose()?,
                            k_bias: bias
                                .then(|| load(&format!("blk.{i}.attn_k.bias"), &[kv_dim]))
                                .transpose()?,
                            v_bias: bias
                                .then(|| load(&format!("blk.{i}.attn_v.bias"), &[kv_dim]))
                                .transpose()?,
                            q_norm: qk_norm
                                .then(|| load(&format!("blk.{i}.attn_q_norm.weight"), &[head_dim]))
                                .transpose()?,
                            k_norm: qk_norm
                                .then(|| load(&format!("blk.{i}.attn_k_norm.weight"), &[head_dim]))
                                .transpose()?,
                        })
                    }
                };
                let post_norms = config.arch.post_norms();
                Ok(Layer {
                    attn_norm: load(&format!("blk.{i}.attn_norm.weight"), &[hidden_dim])?,
                    mixer,
                    post_attn_norm: post_norms
                        .then(|| {
                            load(
                                &format!("blk.{i}.post_attention_norm.weight"),
                                &[hidden_dim],
                            )
                        })
                        .transpose()?,
                    ffn_norm: load(&format!("blk.{i}.ffn_norm.weight"), &[hidden_dim])?,
                    ffn: Ffn {
                        gate: load(&format!("blk.{i}.ffn_gate.weight"), &[ffn_dim, hidden_dim])?,
                        up: load(&format!("blk.{i}.ffn_up.weight"), &[ffn_dim, hidden_dim])?,
                        down: load(&format!("blk.{i}.ffn_down.weight"), &[hidden_dim, ffn_dim])?,
                    },
                    post_ffn_norm: post_norms
                        .then(|| load(&format!("blk.{i}.post_ffw_norm.weight"), &[hidden_dim]))
                        .transpose()?,
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
            ("tokenizer.ggml.model", Value::String("gpt2".into())),
            ("tokenizer.ggml.pre", Value::String("smollm".into())),
            ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
            (
                "tokenizer.ggml.tokens",
                Value::Array(
                    ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
                        .map(|t| Value::String(t.into()))
                        .to_vec(),
                ),
            ),
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
        assert_eq!(config.eos_token.as_deref(), Some("<|im_end|>"));
        assert_eq!(config.bos_token.as_deref(), Some("<|im_start|>"));
        assert_eq!(config.tokenizer_pre.as_deref(), Some("smollm"));
        assert!(!config.add_bos);
        Ok(())
    }

    #[test]
    fn add_bos_follows_the_gguf_then_llama_cpp_defaults() -> Result<()> {
        // (add_bos_token, tokenizer.ggml.model, tokenizer.ggml.pre, expected)
        let cases = [
            (Some(true), "gpt2", "smollm", true),
            (Some(false), "gpt2", "llama-bpe", false),
            (Some(false), "llama", "default", false),
            (None, "gpt2", "llama-bpe", true),
            (None, "llama", "default", true),
            (None, "gpt2", "smollm", false),
        ];
        for (flag, model, pre, expected) in cases {
            let mut md = smollm2_metadata();
            md.insert("tokenizer.ggml.model".into(), Value::String(model.into()));
            md.insert("tokenizer.ggml.pre".into(), Value::String(pre.into()));
            match flag {
                Some(flag) => md.insert("tokenizer.ggml.add_bos_token".into(), Value::Bool(flag)),
                None => md.remove("tokenizer.ggml.add_bos_token"),
            };
            let add_bos = Config::from_gguf_metadata(&md, 49152)?.add_bos;
            assert_eq!(add_bos, expected, "{flag:?} {model} {pre}");
        }
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
                Some(Value::String("mamba".into())),
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
            (
                "tokenizer.ggml.add_bos_token",
                Some(Value::U32(1)),
                "tokenizer.ggml.add_bos_token",
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

    // `tiny_llama_metadata` for another architecture: the same values under
    // that architecture's key prefix.
    fn tiny_metadata(arch: &str) -> HashMap<String, Value> {
        tiny_llama_metadata()
            .into_iter()
            .map(|(key, value)| match key.strip_prefix("llama.") {
                Some(name) => (format!("{arch}.{name}"), value),
                None => (key, value),
            })
            .chain([("general.architecture".into(), Value::String(arch.into()))])
            .collect()
    }

    // Layer `i`'s attention.
    fn attention(model: &Model, i: usize) -> &Attention {
        match &model.layers[i].mixer {
            Mixer::Attention(attn) => attn,
            Mixer::ShortConv(_) => panic!("layer {i} is a short-conv layer"),
        }
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

    // Every weight `tiny_llama_metadata` needs, with `q` and `k` stored as
    // given.
    fn tiny_tensors(q: Tensor, k: Tensor) -> Result<Vec<(&'static str, Tensor)>> {
        let norm = Tensor::ones(8, DType::F32, &Device::Cpu)?;
        Ok(vec![
            ("token_embd.weight", rows(5, 8)?),
            ("output_norm.weight", norm.clone()),
            ("blk.0.attn_norm.weight", norm.clone()),
            ("blk.0.attn_q.weight", q),
            ("blk.0.attn_k.weight", k),
            ("blk.0.attn_v.weight", rows(4, 8)?),
            ("blk.0.attn_output.weight", rows(8, 8)?),
            ("blk.0.ffn_norm.weight", norm),
            ("blk.0.ffn_gate.weight", rows(8, 8)?),
            ("blk.0.ffn_up.weight", rows(8, 8)?),
            ("blk.0.ffn_down.weight", rows(8, 8)?),
        ])
    }

    // `tiny_tensors` with `q` and `k` (HF order) stored the way llama.cpp
    // permutes them for Llama.
    fn tiny_llama_tensors(q: &Tensor, k: &Tensor) -> Result<Vec<(&'static str, Tensor)>> {
        tiny_tensors(llama_cpp_permute(q, 2)?, llama_cpp_permute(k, 1)?)
    }

    #[test]
    fn load_undoes_llama_qk_permutation() -> Result<()> {
        let (q, k) = (rows(8, 8)?, rows(4, 8)?);
        let model = load_gguf(
            "llama",
            &tiny_llama_metadata(),
            &tiny_llama_tensors(&q, &k)?,
        )?;
        assert_eq!((model.config.vocab_size, model.config.head_dim), (5, 4));
        let values = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1() };
        let attn = attention(&model, 0);
        assert_eq!(values(&attn.q)?, values(&q)?);
        assert_eq!(values(&attn.k)?, values(&k)?);
        assert!(attn.q_bias.is_none());
        Ok(())
    }

    #[test]
    fn load_qwen2_keeps_qk_rows_and_reads_biases() -> Result<()> {
        let (q, k) = (rows(8, 8)?, rows(4, 8)?);
        // llama.cpp stores Qwen2's Q/K rows in HF order, unpermuted.
        let mut tensors = tiny_tensors(q.clone(), k.clone())?;
        let md = tiny_metadata("qwen2");
        let Err(err) = load_gguf("qwen2-no-bias", &md, &tensors) else {
            panic!("loaded a qwen2 file without its biases");
        };
        assert!(err.to_string().contains("blk.0.attn_q.bias"), "{err}");

        let q_bias = Tensor::arange(0f32, 8., &Device::Cpu)?;
        let k_bias = Tensor::arange(10f32, 14., &Device::Cpu)?;
        let v_bias = Tensor::arange(20f32, 24., &Device::Cpu)?;
        tensors.extend([
            ("blk.0.attn_q.bias", q_bias.clone()),
            ("blk.0.attn_k.bias", k_bias.clone()),
            ("blk.0.attn_v.bias", v_bias.clone()),
        ]);
        let model = load_gguf("qwen2", &md, &tensors)?;
        assert_eq!(model.config.arch, Arch::Qwen2);
        let values = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1() };
        let attn = attention(&model, 0);
        assert_eq!(values(&attn.q)?, values(&q)?);
        assert_eq!(values(&attn.k)?, values(&k)?);
        for (got, want) in [
            (&attn.q_bias, &q_bias),
            (&attn.k_bias, &k_bias),
            (&attn.v_bias, &v_bias),
        ] {
            assert_eq!(values(got.as_ref().unwrap())?, values(want)?);
        }
        Ok(())
    }

    #[test]
    fn load_reads_rope_factors() -> Result<()> {
        let mut tensors = tiny_llama_tensors(&rows(8, 8)?, &rows(4, 8)?)?;
        let model = load_gguf("no-rope-freqs", &tiny_llama_metadata(), &tensors)?;
        assert_eq!(model.config.rope_factors, None);

        // head_dim 4: one factor per dimension pair.
        let factors = Tensor::new(&[1f32, 8.], &Device::Cpu)?;
        tensors.push(("rope_freqs.weight", factors));
        let model = load_gguf("rope-freqs", &tiny_llama_metadata(), &tensors)?;
        assert_eq!(model.config.rope_factors, Some(vec![1., 8.]));
        Ok(())
    }

    #[test]
    fn tested_shapes_are_recognised() -> Result<()> {
        // SmolLM2-360M's shape is in the tested list; the tiny test model's isn't.
        let config = Config::from_gguf_metadata(&smollm2_metadata(), 49152)?;
        assert!(is_tested_shape(&config));
        let tiny = Config::from_gguf_metadata(&tiny_llama_metadata(), 5)?;
        assert!(!is_tested_shape(&tiny));
        Ok(())
    }

    #[test]
    fn rope_scaling_other_than_llama3_is_refused() {
        let mut md = tiny_llama_metadata();
        md.insert(
            "llama.rope.scaling.type".into(),
            Value::String("yarn".into()),
        );
        let err = Config::from_gguf_metadata(&md, 5).unwrap_err();
        assert!(err.to_string().contains("yarn"), "{err}");

        md.insert(
            "llama.rope.scaling.type".into(),
            Value::String("none".into()),
        );
        assert!(Config::from_gguf_metadata(&md, 5).is_ok());
    }

    #[test]
    fn llama3_scaling_needs_its_factor_tensor() -> Result<()> {
        let mut md = tiny_llama_metadata();
        md.insert(
            "llama.rope.scaling.type".into(),
            Value::String("llama3".into()),
        );
        let mut tensors = tiny_llama_tensors(&rows(8, 8)?, &rows(4, 8)?)?;
        let Err(err) = load_gguf("llama3-no-freqs", &md, &tensors) else {
            panic!("loaded llama3 scaling without rope_freqs.weight");
        };
        assert!(err.to_string().contains("rope_freqs.weight"), "{err}");

        // head_dim 4: one factor per dimension pair.
        tensors.push(("rope_freqs.weight", Tensor::new(&[1f32, 8.], &Device::Cpu)?));
        let model = load_gguf("llama3-freqs", &md, &tensors)?;
        assert_eq!(model.config.rope_factors, Some(vec![1., 8.]));
        Ok(())
    }

    #[test]
    fn load_qwen3_reads_qk_norms() -> Result<()> {
        let (q, k) = (rows(8, 8)?, rows(4, 8)?);
        let mut tensors = tiny_tensors(q.clone(), k.clone())?;
        let md = tiny_metadata("qwen3");
        let Err(err) = load_gguf("qwen3-no-norm", &md, &tensors) else {
            panic!("loaded a qwen3 file without its Q/K norms");
        };
        assert!(
            err.to_string().contains("blk.0.attn_q_norm.weight"),
            "{err}"
        );

        // One weight per element of a head (head_dim 4).
        let q_norm = Tensor::arange(0f32, 4., &Device::Cpu)?;
        let k_norm = Tensor::arange(4f32, 8., &Device::Cpu)?;
        tensors.extend([
            ("blk.0.attn_q_norm.weight", q_norm.clone()),
            ("blk.0.attn_k_norm.weight", k_norm.clone()),
        ]);
        let model = load_gguf("qwen3", &md, &tensors)?;
        assert_eq!(model.config.arch, Arch::Qwen3);
        let values = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1() };
        let attn = attention(&model, 0);
        assert_eq!(values(&attn.q)?, values(&q)?);
        assert_eq!(values(attn.q_norm.as_ref().unwrap())?, values(&q_norm)?);
        assert_eq!(values(attn.k_norm.as_ref().unwrap())?, values(&k_norm)?);
        assert!(attn.q_bias.is_none());
        Ok(())
    }

    // `tiny_metadata("gemma3")` plus the sliding window Gemma 3 requires.
    fn tiny_gemma3_metadata() -> HashMap<String, Value> {
        let mut md = tiny_metadata("gemma3");
        md.insert("gemma3.attention.sliding_window".into(), Value::U32(512));
        md
    }

    #[test]
    fn config_from_gemma3_metadata() -> Result<()> {
        let mut md = tiny_gemma3_metadata();
        let config = Config::from_gguf_metadata(&md, 5)?;
        assert_eq!(config.arch, Arch::Gemma3);
        // No `rope.freq_base_swa`, so llama.cpp's default base.
        let local = LocalAttention {
            window: 512,
            rope_theta: 10000.0,
        };
        assert_eq!(config.local_attention, Some(local));
        assert_eq!(config.embedding_scale, Some(8f32.sqrt()));
        assert_eq!(config.activation, Activation::GeluTanh);

        md.insert("gemma3.rope.freq_base_swa".into(), Value::F32(5000.0));
        let config = Config::from_gguf_metadata(&md, 5)?;
        assert_eq!(config.local_attention.unwrap().rope_theta, 5000.0);

        md.insert("gemma3.attention.sliding_window".into(), Value::U32(0));
        let err = Config::from_gguf_metadata(&md, 5).unwrap_err();
        assert!(err.to_string().contains("sliding window"), "{err}");
        md.remove("gemma3.attention.sliding_window");
        let err = Config::from_gguf_metadata(&md, 5).unwrap_err();
        assert!(
            err.to_string().contains("gemma3.attention.sliding_window"),
            "{err}"
        );

        let llama = Config::from_gguf_metadata(&tiny_llama_metadata(), 5)?;
        assert_eq!(llama.local_attention, None);
        assert_eq!(llama.embedding_scale, None);
        assert_eq!(llama.activation, Activation::Silu);
        Ok(())
    }

    #[test]
    fn gemma3_follows_five_local_layers_with_a_global_one() {
        let local: Vec<bool> = (0..12).map(|i| Arch::Gemma3.local_layer(i)).collect();
        let pattern = [true, true, true, true, true, false];
        assert_eq!(local, pattern.repeat(2));
        assert!(!(0..12).any(|i| Arch::Llama.local_layer(i)));
    }

    #[test]
    fn load_gemma3_reads_post_norms() -> Result<()> {
        let (q, k) = (rows(8, 8)?, rows(4, 8)?);
        let mut tensors = tiny_tensors(q, k)?;
        let head_norm = Tensor::ones(4, DType::F32, &Device::Cpu)?;
        tensors.extend([
            ("blk.0.attn_q_norm.weight", head_norm.clone()),
            ("blk.0.attn_k_norm.weight", head_norm),
        ]);
        let md = tiny_gemma3_metadata();
        let Err(err) = load_gguf("gemma3-no-post-norm", &md, &tensors) else {
            panic!("loaded a gemma3 file without its post norms");
        };
        assert!(
            err.to_string().contains("blk.0.post_attention_norm.weight"),
            "{err}"
        );

        let post_attn = Tensor::arange(0f32, 8., &Device::Cpu)?;
        let post_ffn = Tensor::arange(8f32, 16., &Device::Cpu)?;
        tensors.extend([
            ("blk.0.post_attention_norm.weight", post_attn.clone()),
            ("blk.0.post_ffw_norm.weight", post_ffn.clone()),
        ]);
        let model = load_gguf("gemma3", &md, &tensors)?;
        let values = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1() };
        let layer = &model.layers[0];
        assert!(attention(&model, 0).local);
        assert!(attention(&model, 0).q_norm.is_some());
        assert_eq!(
            values(layer.post_attn_norm.as_ref().unwrap())?,
            values(&post_attn)?
        );
        assert_eq!(
            values(layer.post_ffn_norm.as_ref().unwrap())?,
            values(&post_ffn)?
        );
        Ok(())
    }

    // Two LFM2 layers: a short convolution, then attention with one KV head.
    fn tiny_lfm2_metadata() -> HashMap<String, Value> {
        let mut md = tiny_metadata("lfm2");
        md.insert("lfm2.block_count".into(), Value::U32(2));
        md.insert(
            "lfm2.attention.head_count_kv".into(),
            Value::Array(vec![Value::I32(0), Value::I32(1)]),
        );
        md.insert("lfm2.shortconv.l_cache".into(), Value::U32(3));
        md
    }

    #[test]
    fn config_from_lfm2_metadata() -> Result<()> {
        let config = Config::from_gguf_metadata(&tiny_lfm2_metadata(), 5)?;
        assert_eq!(config.arch, Arch::Lfm2);
        assert_eq!(config.n_kv_heads, 1);
        let conv = ConvLayers {
            layers: vec![0],
            kernel: 3,
        };
        assert_eq!(config.conv, Some(conv));
        assert_eq!(config.n_attention_layers(), 1);

        // (per-layer KV head counts, text the error must contain)
        let cases = [
            (vec![0, 0], "no attention layers"),
            (vec![2, 1], "different KV head counts"),
            (vec![0], "has 1 entries for 2 layers"),
            (vec![0, -1], "negative count"),
        ];
        for (per_layer, expected) in cases {
            let mut md = tiny_lfm2_metadata();
            let per_layer = per_layer.into_iter().map(Value::I32).collect();
            md.insert(
                "lfm2.attention.head_count_kv".into(),
                Value::Array(per_layer),
            );
            let err = Config::from_gguf_metadata(&md, 5).unwrap_err().to_string();
            assert!(err.contains(expected), "{expected}: {err}");
        }
        let mut md = tiny_lfm2_metadata();
        md.remove("lfm2.shortconv.l_cache");
        let err = Config::from_gguf_metadata(&md, 5).unwrap_err().to_string();
        assert!(err.contains("lfm2.shortconv.l_cache"), "{err}");

        // Only LFM2 has short-conv layers; elsewhere 0 KV heads is invalid.
        let mut md = tiny_llama_metadata();
        md.insert(
            "llama.attention.head_count_kv".into(),
            Value::Array(vec![Value::I32(0)]),
        );
        let err = Config::from_gguf_metadata(&md, 5).unwrap_err().to_string();
        assert!(err.contains("head counts"), "{err}");
        Ok(())
    }

    #[test]
    fn load_lfm2_layers() -> Result<()> {
        let norm = Tensor::ones(8, DType::F32, &Device::Cpu)?;
        let head_norm = Tensor::ones(4, DType::F32, &Device::Cpu)?;
        let final_norm = Tensor::arange(0f32, 8., &Device::Cpu)?;
        let conv = Tensor::arange(0f32, 24., &Device::Cpu)?.reshape((8, 3))?;
        let q = rows(8, 8)?;
        let tensors = [
            ("token_embd.weight", rows(5, 8)?),
            ("token_embd_norm.weight", final_norm.clone()),
            ("blk.0.attn_norm.weight", norm.clone()),
            ("blk.0.shortconv.in_proj.weight", rows(24, 8)?),
            ("blk.0.shortconv.conv.weight", conv.clone()),
            ("blk.0.shortconv.out_proj.weight", rows(8, 8)?),
            ("blk.0.ffn_norm.weight", norm.clone()),
            ("blk.0.ffn_gate.weight", rows(8, 8)?),
            ("blk.0.ffn_up.weight", rows(8, 8)?),
            ("blk.0.ffn_down.weight", rows(8, 8)?),
            ("blk.1.attn_norm.weight", norm.clone()),
            ("blk.1.attn_q.weight", q.clone()),
            ("blk.1.attn_k.weight", rows(4, 8)?),
            ("blk.1.attn_v.weight", rows(4, 8)?),
            ("blk.1.attn_output.weight", rows(8, 8)?),
            ("blk.1.attn_q_norm.weight", head_norm.clone()),
            ("blk.1.attn_k_norm.weight", head_norm),
            ("blk.1.ffn_norm.weight", norm),
            ("blk.1.ffn_gate.weight", rows(8, 8)?),
            ("blk.1.ffn_up.weight", rows(8, 8)?),
            ("blk.1.ffn_down.weight", rows(8, 8)?),
        ];
        let model = load_gguf("lfm2", &tiny_lfm2_metadata(), &tensors)?;
        let values = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1() };
        assert_eq!(values(&model.output_norm)?, values(&final_norm)?);
        let Mixer::ShortConv(short_conv) = &model.layers[0].mixer else {
            panic!("layer 0 isn't a short-conv layer");
        };
        assert_eq!(short_conv.conv_index, 0);
        assert_eq!(values(&short_conv.conv)?, values(&conv)?);
        // The first attention layer takes the first KV slot.
        let attn = attention(&model, 1);
        assert_eq!(attn.kv_index, 0);
        assert_eq!(values(&attn.q)?, values(&q)?);
        Ok(())
    }

    #[test]
    fn load_rejects_a_tensor_of_the_wrong_shape() -> Result<()> {
        let mut tensors = tiny_llama_tensors(&rows(8, 8)?, &rows(4, 8)?)?;
        // One KV head of size 4 needs a [4, 8] V projection.
        for (name, tensor) in &mut tensors {
            if *name == "blk.0.attn_v.weight" {
                *tensor = rows(3, 8)?;
            }
        }
        let Err(err) = load_gguf("bad-shape", &tiny_llama_metadata(), &tensors) else {
            panic!("loaded a tensor of the wrong shape");
        };
        let want = "\"blk.0.attn_v.weight\" has shape [3, 8], expected [4, 8]";
        assert!(err.to_string().contains(want), "{err}");
        Ok(())
    }

    #[test]
    fn load_checks_architecture_before_reading_weights() -> Result<()> {
        let mut md = tiny_llama_metadata();
        md.insert("general.architecture".into(), Value::String("mamba".into()));
        // Every other tensor is missing, so reading any weight would fail differently.
        let Err(err) = load_gguf("mamba", &md, &[("token_embd.weight", rows(5, 8)?)]) else {
            panic!("loaded a mamba file");
        };
        assert!(err.to_string().contains("\"mamba\""), "{err}");
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
