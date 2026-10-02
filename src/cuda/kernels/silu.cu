extern "C" __global__ void silu(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = in[i];
        out[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}
