__device__ float silu_helper(float in) {
    return in * (1.0f / (1.0f + expf(-in)));
}

extern "C" __global__ void silu_gate_multiply(const float *gate, const float *up, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gateVal = gate[i];
        float upVal = up[i];
        out[i] = silu_helper(gateVal) * upVal;
    }
}
