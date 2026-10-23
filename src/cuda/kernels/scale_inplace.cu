// x *= scale, elementwise, in place (Gemma's sqrt(hidden_dim) embedding scale).
extern "C" __global__ void scale_inplace(float *x, float scale, unsigned int n) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        x[i] *= scale;
    }
}
