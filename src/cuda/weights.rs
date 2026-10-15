// One-time CPU -> CudaSlice weight upload. Weights are loaded via
// Candle/GGUF on CPU as usual (model.rs/weights.rs, untouched), then copied
// once into flat device buffers here. Same tensor set as `model::Layer`/
// `model::Model`, just CudaSlice-typed instead of Candle-Tensor-typed -
// no shared trait, same reasoning as CudaKvStorage vs KvStorage.
// The big projection matrices are narrowed to F16 on upload (decode is
// memory bound, so this halves the bytes moved); norm weights and the token
// embedding table stay F32.
use candle_core::Tensor;
use cudarc::driver::CudaSlice;
use half::f16;

use crate::model::{Mixer, Model};

use super::context::CudaRuntime;

// An attention layer's weights, the GPU side of model::Attention.
pub struct CudaAttention {
    // Index of this layer's slot in the KV cache.
    pub kv_index: usize,
    pub local: bool,
    pub q: CudaSlice<f16>,
    pub k: CudaSlice<f16>,
    pub v: CudaSlice<f16>,
    pub o: CudaSlice<f16>,
    pub q_bias: Option<CudaSlice<f32>>,
    pub k_bias: Option<CudaSlice<f32>>,
    pub v_bias: Option<CudaSlice<f32>>,
    pub q_norm: Option<CudaSlice<f32>>,
    pub k_norm: Option<CudaSlice<f32>>,
}

// A short-convolution layer's weights, the GPU side of model::ShortConv.
pub struct CudaShortConv {
    // Index of this layer's state in the conv cache.
    pub conv_index: usize,
    // Taps per channel: the conv's kernel size.
    pub kernel: usize,
    pub in_proj: CudaSlice<f16>,
    // [hidden, kernel], one filter per channel, oldest tap first.
    pub filter: CudaSlice<f32>,
    pub out_proj: CudaSlice<f16>,
}

#[allow(clippy::large_enum_variant)]
pub enum CudaMixer {
    Attention(CudaAttention),
    ShortConv(CudaShortConv),
}

pub struct CudaLayer {
    pub attn_norm: CudaSlice<f32>,
    pub mixer: CudaMixer,
    pub post_attn_norm: Option<CudaSlice<f32>>,
    pub ffn_norm: CudaSlice<f32>,
    pub ffn_gate: CudaSlice<f16>,
    pub ffn_up: CudaSlice<f16>,
    pub ffn_down: CudaSlice<f16>,
    pub post_ffn_norm: Option<CudaSlice<f32>>,
}

pub struct CudaWeights {
    pub token_embd: CudaSlice<f32>,
    pub layers: Vec<CudaLayer>,
    pub output_norm: CudaSlice<f32>,
    pub output: CudaSlice<f16>,
}

fn upload(
    cuda_runtime: &CudaRuntime,
    t: &Tensor,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let host: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(cuda_runtime.stream.clone_htod(&host)?)
}

fn upload_f16(
    cuda_runtime: &CudaRuntime,
    t: &Tensor,
) -> Result<CudaSlice<f16>, Box<dyn std::error::Error>> {
    let host: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let host: Vec<f16> = host.into_iter().map(f16::from_f32).collect();
    Ok(cuda_runtime.stream.clone_htod(&host)?)
}

// `upload` for an optional tensor: None stays None.
fn upload_optional(
    cuda_runtime: &CudaRuntime,
    t: Option<&Tensor>,
) -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
    t.map(|t| upload(cuda_runtime, t)).transpose()
}

impl CudaWeights {
    pub fn upload(
        cuda_runtime: &CudaRuntime,
        model: &Model,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let token_embd = upload(cuda_runtime, &model.token_embd)?;
        let layers = model
            .layers
            .iter()
            .map(|l| -> Result<CudaLayer, Box<dyn std::error::Error>> {
                let mixer = match &l.mixer {
                    Mixer::Attention(attn) => CudaMixer::Attention(CudaAttention {
                        kv_index: attn.kv_index,
                        local: attn.local,
                        q: upload_f16(cuda_runtime, &attn.q)?,
                        k: upload_f16(cuda_runtime, &attn.k)?,
                        v: upload_f16(cuda_runtime, &attn.v)?,
                        o: upload_f16(cuda_runtime, &attn.o)?,
                        q_bias: upload_optional(cuda_runtime, attn.q_bias.as_ref())?,
                        k_bias: upload_optional(cuda_runtime, attn.k_bias.as_ref())?,
                        v_bias: upload_optional(cuda_runtime, attn.v_bias.as_ref())?,
                        q_norm: upload_optional(cuda_runtime, attn.q_norm.as_ref())?,
                        k_norm: upload_optional(cuda_runtime, attn.k_norm.as_ref())?,
                    }),
                    Mixer::ShortConv(conv) => CudaMixer::ShortConv(CudaShortConv {
                        conv_index: conv.conv_index,
                        kernel: conv.conv.dim(1)?,
                        in_proj: upload_f16(cuda_runtime, &conv.in_proj)?,
                        filter: upload(cuda_runtime, &conv.conv)?,
                        out_proj: upload_f16(cuda_runtime, &conv.out_proj)?,
                    }),
                };
                Ok(CudaLayer {
                    attn_norm: upload(cuda_runtime, &l.attn_norm)?,
                    mixer,
                    post_attn_norm: upload_optional(cuda_runtime, l.post_attn_norm.as_ref())?,
                    ffn_norm: upload(cuda_runtime, &l.ffn_norm)?,
                    ffn_gate: upload_f16(cuda_runtime, &l.ffn.gate)?,
                    ffn_up: upload_f16(cuda_runtime, &l.ffn.up)?,
                    ffn_down: upload_f16(cuda_runtime, &l.ffn.down)?,
                    post_ffn_norm: upload_optional(cuda_runtime, l.post_ffn_norm.as_ref())?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_norm = upload(cuda_runtime, &model.output_norm)?;
        let output = upload_f16(cuda_runtime, &model.output)?;
        Ok(CudaWeights {
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
    use crate::model::{Attention, Config, Ffn, Layer};
    use candle_core::{DType, Device};

    fn runtime() -> Option<CudaRuntime> {
        crate::cuda::context::test_runtime()
    }

    // Tiny fake model, same shape convention as model.rs's own `tiny_model`
    // test helper: 1 layer, hidden_dim=4, n_heads=2, n_kv_heads=1, ffn_dim=4,
    // vocab_size=5.
    fn tiny_model() -> Model {
        let device = Device::Cpu;
        let (hidden_dim, n_heads, n_kv_heads, ffn_dim, vocab_size) =
            (4usize, 2usize, 1usize, 4usize, 5usize);
        let head_dim = hidden_dim / n_heads;
        let rand = |shape: (usize, usize)| Tensor::rand(0f32, 1., shape, &device).unwrap();
        let ones = || Tensor::ones(hidden_dim, DType::F32, &device).unwrap();

        let token_embd = rand((vocab_size, hidden_dim));
        let layer = Layer {
            attn_norm: ones(),
            mixer: Mixer::Attention(Attention {
                kv_index: 0,
                local: false,
                q: rand((n_heads * head_dim, hidden_dim)),
                k: rand((n_kv_heads * head_dim, hidden_dim)),
                v: rand((n_kv_heads * head_dim, hidden_dim)),
                o: rand((hidden_dim, n_heads * head_dim)),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
            }),
            post_attn_norm: None,
            ffn_norm: ones(),
            ffn: Ffn {
                gate: rand((ffn_dim, hidden_dim)),
                up: rand((ffn_dim, hidden_dim)),
                down: rand((hidden_dim, ffn_dim)),
            },
            post_ffn_norm: None,
        };
        Model {
            config: Config {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                hidden_dim: hidden_dim as u32,
                head_dim: head_dim as u32,
                ffn_dim: ffn_dim as u32,
                vocab_size: vocab_size as u32,
                ..Config::for_tests()
            },
            token_embd,
            layers: vec![layer],
            output_norm: ones(),
            output: rand((vocab_size, hidden_dim)),
        }
    }

    #[test]
    fn upload_matches_cpu_tensor_values() {
        let Some(rt) = runtime() else { return };
        let model = tiny_model();
        let weights = CudaWeights::upload(&rt, &model).unwrap();

        let want: Vec<f32> = model.token_embd.flatten_all().unwrap().to_vec1().unwrap();
        let got = rt.stream.clone_dtoh(&weights.token_embd).unwrap();
        assert_eq!(got, want);

        let Mixer::Attention(attn) = &model.layers[0].mixer else {
            unreachable!("the tiny model's layer is attention");
        };
        let want_q: Vec<f32> = attn.q.flatten_all().unwrap().to_vec1().unwrap();
        let CudaMixer::Attention(cuda_attn) = &weights.layers[0].mixer else {
            unreachable!("the tiny model's layer is attention");
        };
        let got_q = rt.stream.clone_dtoh(&cuda_attn.q).unwrap();
        for (g, w) in got_q.iter().zip(want_q.iter()) {
            assert!(
                (g.to_f32() - w).abs() < 1e-3,
                "got {got_q:?}, want {want_q:?}"
            );
        }

        let want_out: Vec<f32> = model.output.flatten_all().unwrap().to_vec1().unwrap();
        let got_out = rt.stream.clone_dtoh(&weights.output).unwrap();
        for (g, w) in got_out.iter().zip(want_out.iter()) {
            assert!(
                (g.to_f32() - w).abs() < 1e-3,
                "got {got_out:?}, want {want_out:?}"
            );
        }
    }
}
