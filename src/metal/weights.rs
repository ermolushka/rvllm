// One-time CPU -> Metal weight upload, the twin of cuda/weights.rs. Weights are
// loaded via Candle/GGUF on the CPU as usual, then copied once into shared
// buffers. The big projection matrices are narrowed to F16 on upload (decode is
// memory bound, so this halves the bytes streamed per token); norm weights and
// the token embedding table stay F32.
use candle_core::Tensor;
use half::f16;

use crate::model::{Attention, Mixer, Model};

use super::context::{MetalBuf, MetalRuntime, Result};

pub struct MetalAttention {
    // Index of this layer's slot in the KV cache.
    pub kv_index: usize,
    pub local: bool,
    // Q, K and V projections stacked row-wise: [(n_heads + 2 * n_kv_heads) * head_dim, hidden_dim],
    // so one matmul produces all three.
    pub qkv: MetalBuf<f16>,
    pub o: MetalBuf<f16>,
    // The three biases stacked the same way (zeros for one a model lacks), when any exists.
    pub qkv_bias: Option<MetalBuf<f32>>,
    pub q_norm: Option<MetalBuf<f32>>,
    pub k_norm: Option<MetalBuf<f32>>,
}

pub struct MetalShortConv {
    pub conv_index: usize,
    // Taps per channel: the conv's kernel size.
    pub kernel: usize,
    pub in_proj: MetalBuf<f16>,
    // [hidden, kernel], one filter per channel, oldest tap first.
    pub filter: MetalBuf<f32>,
    pub out_proj: MetalBuf<f16>,
}

#[allow(clippy::large_enum_variant)]
pub enum MetalMixer {
    Attention(MetalAttention),
    ShortConv(MetalShortConv),
}

pub struct MetalLayer {
    pub attn_norm: MetalBuf<f32>,
    pub mixer: MetalMixer,
    pub post_attn_norm: Option<MetalBuf<f32>>,
    pub ffn_norm: MetalBuf<f32>,
    // Gate and up projections stacked row-wise: [2 * ffn_dim, hidden_dim].
    pub ffn_gate_up: MetalBuf<f16>,
    pub ffn_down: MetalBuf<f16>,
    pub post_ffn_norm: Option<MetalBuf<f32>>,
}

pub struct MetalWeights {
    pub token_embd: MetalBuf<f32>,
    pub layers: Vec<MetalLayer>,
    pub output_norm: MetalBuf<f32>,
    pub output: MetalBuf<f16>,
}

fn upload(rt: &MetalRuntime, t: &Tensor) -> Result<MetalBuf<f32>> {
    let host: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(MetalBuf::from_slice(rt, &host))
}

fn upload_f16(rt: &MetalRuntime, t: &Tensor) -> Result<MetalBuf<f16>> {
    let host: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let host: Vec<f16> = host.into_iter().map(f16::from_f32).collect();
    Ok(MetalBuf::from_slice(rt, &host))
}

fn upload_optional(rt: &MetalRuntime, t: Option<&Tensor>) -> Result<Option<MetalBuf<f32>>> {
    t.map(|t| upload(rt, t)).transpose()
}

// The Q/K/V biases as one vector in QKV order; a missing one becomes zeros
// (adding 0.0 changes nothing). None when the layer has no bias at all.
fn stacked_bias(rt: &MetalRuntime, attn: &Attention) -> Result<Option<MetalBuf<f32>>> {
    if attn.q_bias.is_none() && attn.k_bias.is_none() && attn.v_bias.is_none() {
        return Ok(None);
    }
    let part = |bias: &Option<Tensor>, weight: &Tensor| -> Result<Vec<f32>> {
        match bias {
            Some(b) => Ok(b.flatten_all()?.to_vec1()?),
            None => Ok(vec![0.0; weight.dim(0)?]),
        }
    };
    let mut all = part(&attn.q_bias, &attn.q)?;
    all.extend(part(&attn.k_bias, &attn.k)?);
    all.extend(part(&attn.v_bias, &attn.v)?);
    Ok(Some(MetalBuf::from_slice(rt, &all)))
}

impl MetalWeights {
    pub fn upload(rt: &MetalRuntime, model: &Model) -> Result<Self> {
        let token_embd = upload(rt, &model.token_embd)?;
        let layers = model
            .layers
            .iter()
            .map(|l| -> Result<MetalLayer> {
                let mixer = match &l.mixer {
                    Mixer::Attention(attn) => MetalMixer::Attention(MetalAttention {
                        kv_index: attn.kv_index,
                        local: attn.local,
                        qkv: upload_f16(rt, &Tensor::cat(&[&attn.q, &attn.k, &attn.v], 0)?)?,
                        o: upload_f16(rt, &attn.o)?,
                        qkv_bias: stacked_bias(rt, attn)?,
                        q_norm: upload_optional(rt, attn.q_norm.as_ref())?,
                        k_norm: upload_optional(rt, attn.k_norm.as_ref())?,
                    }),
                    Mixer::ShortConv(conv) => MetalMixer::ShortConv(MetalShortConv {
                        conv_index: conv.conv_index,
                        kernel: conv.conv.dim(1)?,
                        in_proj: upload_f16(rt, &conv.in_proj)?,
                        filter: upload(rt, &conv.conv)?,
                        out_proj: upload_f16(rt, &conv.out_proj)?,
                    }),
                };
                Ok(MetalLayer {
                    attn_norm: upload(rt, &l.attn_norm)?,
                    mixer,
                    post_attn_norm: upload_optional(rt, l.post_attn_norm.as_ref())?,
                    ffn_norm: upload(rt, &l.ffn_norm)?,
                    ffn_gate_up: upload_f16(rt, &Tensor::cat(&[&l.ffn.gate, &l.ffn.up], 0)?)?,
                    ffn_down: upload_f16(rt, &l.ffn.down)?,
                    post_ffn_norm: upload_optional(rt, l.post_ffn_norm.as_ref())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MetalWeights {
            token_embd,
            layers,
            output_norm: upload(rt, &model.output_norm)?,
            output: upload_f16(rt, &model.output)?,
        })
    }
}
