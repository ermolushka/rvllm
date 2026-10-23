// GeGLU gate: out = gelu_tanh(gate) * up, with GELU's tanh approximation
// (what Gemma's FFN uses, and what candle's Tensor::gelu computes).
__device__ float gelu_tanh_helper(float x) {
    const float k = 0.7978845608028654f; // sqrt(2 / pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void gelu_gate_multiply(const float *gate, const float *up, float *out, unsigned int n) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = gelu_tanh_helper(gate[i]) * up[i];
    }
}
