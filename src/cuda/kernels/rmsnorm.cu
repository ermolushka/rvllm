extern "C" __global__ void rmsnorm(
    const float *x,
    const float *weight,
    float *out,
    unsigned int hidden_dim,
    float eps
) {
    extern __shared__ float sdata[];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float *row_in = x + (size_t)row * hidden_dim;
    float *row_out = out + (size_t)row * hidden_dim;

    float partial = 0.0f;
    for (unsigned int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = row_in[i];
        partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    __shared__ float rms;
    if (tid == 0) {
        float mean_sq = sdata[0] / (float)hidden_dim;
        rms = sqrtf(mean_sq + eps);
    }
    __syncthreads();

    for (unsigned int i = tid; i < hidden_dim; i += blockDim.x) {
        row_out[i] = (row_in[i] / rms) * weight[i];
    }
}
