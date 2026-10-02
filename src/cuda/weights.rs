// One-time CPU -> CudaSlice weight upload. Weights are loaded via
// Candle/GGUF on CPU as usual (model.rs/weights.rs, untouched), then copied
// once into flat device buffers here. Same tensor set as `model::Layer`/
// `model::Model`, just CudaSlice<f32>-typed instead of Candle-Tensor-typed -
// no shared trait, same reasoning as CudaKvStorage vs KvStorage.
use candle_core::Tensor;
use cudarc::driver::CudaSlice;

use crate::model::Model;

use super::context::CudaRuntime;

pub struct CudaLayer {
    pub attn_norm: CudaSlice<f32>,
    pub attn_q: CudaSlice<f32>,
    pub attn_k: CudaSlice<f32>,
    pub attn_v: CudaSlice<f32>,
    pub attn_output: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    pub ffn_down: CudaSlice<f32>,
}

pub struct CudaWeights {
    pub token_embd: CudaSlice<f32>,
    pub layers: Vec<CudaLayer>,
    pub output_norm: CudaSlice<f32>,
    pub output: CudaSlice<f32>,
}

fn upload(
    cuda_runtime: &CudaRuntime,
    t: &Tensor,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let host: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(cuda_runtime.stream.clone_htod(&host)?)
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
                Ok(CudaLayer {
                    attn_norm: upload(cuda_runtime, &l.attn_norm)?,
                    attn_q: upload(cuda_runtime, &l.attn_q)?,
                    attn_k: upload(cuda_runtime, &l.attn_k)?,
                    attn_v: upload(cuda_runtime, &l.attn_v)?,
                    attn_output: upload(cuda_runtime, &l.attn_output)?,
                    ffn_norm: upload(cuda_runtime, &l.ffn_norm)?,
                    ffn_gate: upload(cuda_runtime, &l.ffn_gate)?,
                    ffn_up: upload(cuda_runtime, &l.ffn_up)?,
                    ffn_down: upload(cuda_runtime, &l.ffn_down)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_norm = upload(cuda_runtime, &model.output_norm)?;
        let output = upload(cuda_runtime, &model.output)?;
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
    use crate::model::{Config, Layer};
    use candle_core::{DType, Device};

    fn runtime() -> Option<CudaRuntime> {
        CudaRuntime::new(0).ok()
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
            index: 0,
            attn_norm: ones(),
            attn_q: rand((n_heads * head_dim, hidden_dim)),
            attn_k: rand((n_kv_heads * head_dim, hidden_dim)),
            attn_v: rand((n_kv_heads * head_dim, hidden_dim)),
            attn_output: rand((hidden_dim, n_heads * head_dim)),
            ffn_norm: ones(),
            ffn_gate: rand((ffn_dim, hidden_dim)),
            ffn_up: rand((ffn_dim, hidden_dim)),
            ffn_down: rand((hidden_dim, ffn_dim)),
        };
        Model {
            config: Config {
                n_layers: 1,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                hidden_dim: hidden_dim as u32,
                ffn_dim: ffn_dim as u32,
                rope_theta: 10000.0,
                rms_eps: 1e-5,
                vocab_size: vocab_size as u32,
                context_length: 8,
                eos_token_id: 99,
                bos_token_id: 0,
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

        let want_q: Vec<f32> = model.layers[0]
            .attn_q
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got_q = rt.stream.clone_dtoh(&weights.layers[0].attn_q).unwrap();
        assert_eq!(got_q, want_q);

        let want_out: Vec<f32> = model.output.flatten_all().unwrap().to_vec1().unwrap();
        let got_out = rt.stream.clone_dtoh(&weights.output).unwrap();
        assert_eq!(got_out, want_out);
    }
}
