// One block per row: index of the row's maximum. Ties go to the lowest index
// and NaN is ignored, same as Sampler::sample_slice's greedy scan, so the
// GPU and CPU paths pick identical tokens. A row with no value above -inf
// (all -inf / NaN) returns 0, like the CPU scan.
extern "C" __global__ void argmax(
    const float *x,
    unsigned int *out,
    unsigned int row_len
) {
    extern __shared__ unsigned char smem[];
    float *vals = (float *)smem;
    unsigned int *idxs = (unsigned int *)(smem + blockDim.x * sizeof(float));

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float *row_in = x + (size_t)row * row_len;

    // NVRTC has no INFINITY; build -inf from its bit pattern.
    float best = -__int_as_float(0x7f800000);
    unsigned int best_idx = 0xffffffffu;
    for (unsigned int i = tid; i < row_len; i += blockDim.x) {
        float v = row_in[i];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = vals[tid + s];
            unsigned int i2 = idxs[tid + s];
            if (v2 > vals[tid] || (v2 == vals[tid] && i2 < idxs[tid])) {
                vals[tid] = v2;
                idxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = idxs[0] == 0xffffffffu ? 0u : idxs[0];
    }
}
