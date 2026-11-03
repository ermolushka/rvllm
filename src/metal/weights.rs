// One-time CPU -> Metal weight upload, the twin of cuda/weights.rs. Weights are
// loaded via Candle/GGUF on the CPU as usual, then copied once into shared
// buffers. The big projection matrices are narrowed to F16 on upload (decode is
// memory bound, so this halves the bytes streamed per token); norm weights and
// the token embedding table stay F32.
use candle_core::Tensor;
use half::f16;

use crate::model::{Mixer, Model};

use super::context::{MetalBuf, MetalRuntime, Result};

pub struct MetalAttention {
    // Index of this layer's slot in the KV cache.
    pub kv_index: usize,
    pub local: bool,
    pub q: MetalBuf<f16>,
    pub k: MetalBuf<f16>,
    pub v: MetalBuf<f16>,
    pub o: MetalBuf<f16>,
    pub q_bias: Option<MetalBuf<f32>>,
    pub k_bias: Option<MetalBuf<f32>>,
    pub v_bias: Option<MetalBuf<f32>>,
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
    pub ffn_gate: MetalBuf<f16>,
    pub ffn_up: MetalBuf<f16>,
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
                        q: upload_f16(rt, &attn.q)?,
                        k: upload_f16(rt, &attn.k)?,
                        v: upload_f16(rt, &attn.v)?,
                        o: upload_f16(rt, &attn.o)?,
                        q_bias: upload_optional(rt, attn.q_bias.as_ref())?,
                        k_bias: upload_optional(rt, attn.k_bias.as_ref())?,
                        v_bias: upload_optional(rt, attn.v_bias.as_ref())?,
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
                    ffn_gate: upload_f16(rt, &l.ffn.gate)?,
                    ffn_up: upload_f16(rt, &l.ffn.up)?,
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
