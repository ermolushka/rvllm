use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Temperature + top-p (nucleus) sampling over a 1D logits tensor
// ([vocab_size]). `temperature <= 0.0` is a special case: skip softmax
// entirely and fall back to greedy argmax, both to avoid a divide-by-zero
// and to keep the T=0 path byte-identical to Phase 1's greedy decoding.
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Sampler { temperature, top_p, rng: StdRng::seed_from_u64(seed) }
    }

    pub fn sample(&mut self, logits: &Tensor) -> candle_core::Result<u32> {
        if self.temperature <= 0.0 {
            return logits.argmax(0)?.to_scalar::<u32>();
        }

        let logits: Vec<f32> = logits.to_vec1()?;
        let inv_temp = 1.0 / self.temperature;
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b * inv_temp));
        let mut probs: Vec<f32> =
            logits.iter().map(|&l| (l * inv_temp - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Nucleus filtering: keep the smallest prefix (by descending
        // probability) whose cumulative mass reaches top_p, then renormalize
        // and sample from just that set.
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut cumulative = 0.0f32;
        let mut cutoff = order.len();
        for (i, &idx) in order.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= self.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let kept = &order[..cutoff];
        let kept_sum: f32 = kept.iter().map(|&i| probs[i]).sum();

        let r: f32 = self.rng.random::<f32>() * kept_sum;
        let mut acc = 0.0f32;
        for &idx in kept {
            acc += probs[idx];
            if acc >= r {
                return Ok(idx as u32);
            }
        }
        Ok(*kept.last().unwrap() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn zero_temperature_matches_argmax() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 0.9, 0.05, -0.2], &device).unwrap();
        let mut sampler = Sampler::new(0.0, 1.0, 42);
        assert_eq!(sampler.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn near_zero_temperature_and_top_p_near_one_matches_greedy() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, -3.0, 0.5], &device).unwrap();
        let expected = logits.argmax(0).unwrap().to_scalar::<u32>().unwrap();
        let mut sampler = Sampler::new(1e-6, 0.999, 7);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits).unwrap(), expected);
        }
    }

    #[test]
    fn top_p_excludes_low_probability_tokens() {
        let device = Device::Cpu;
        // Token 0 dominates the softmax at T=1; a tight top_p should never
        // let the low-probability tail get sampled.
        let logits = Tensor::new(&[10.0f32, -10.0, -10.0, -10.0], &device).unwrap();
        let mut sampler = Sampler::new(1.0, 0.5, 123);
        for _ in 0..50 {
            assert_eq!(sampler.sample(&logits).unwrap(), 0);
        }
    }

    #[test]
    fn sampling_is_deterministic_given_seed() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &device).unwrap();
        let mut a = Sampler::new(1.0, 1.0, 99);
        let mut b = Sampler::new(1.0, 1.0, 99);
        let seq_a: Vec<u32> = (0..20).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..20).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b);
    }
}
