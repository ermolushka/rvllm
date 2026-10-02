extern "C" __global__ void softmax(
    const float *x,
    float *out,
    unsigned int row_len
) {
    extern __shared__ float sdata[];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float *row_in = x + (size_t)row * row_len;
    float *row_out = out + (size_t)row * row_len;

    // pass 1: row max, for numerical stability. Identity is -inf (not 0),
    // so a row that's legitimately all-negative still reduces correctly.
    // NVRTC compiles without <math.h>, so INFINITY isn't defined - build
    // -inf from its IEEE-754 bit pattern instead (0x7f800000 = +inf).
    float partial_max = -__int_as_float(0x7f800000);
    for (unsigned int i = tid; i < row_len; i += blockDim.x) {
        partial_max = fmaxf(partial_max, row_in[i]);
    }
    sdata[tid] = partial_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    __shared__ float row_max;
    if (tid == 0) {
        row_max = sdata[0];
    }
    __syncthreads();

    // pass 2: sum of exp(x - row_max). Store the exp values into row_out
    // so pass 3 doesn't need to recompute expf. -inf inputs fall out
    // naturally: expf(-inf - finite) == 0, contributing nothing to the sum.
    float partial_sum = 0.0f;
    for (unsigned int i = tid; i < row_len; i += blockDim.x) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;
        partial_sum += e;
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    __shared__ float row_sum;
    if (tid == 0) {
        row_sum = sdata[0];
    }
    __syncthreads();

    // pass 3: normalize the exp values already sitting in row_out.
    for (unsigned int i = tid; i < row_len; i += blockDim.x) {
        row_out[i] = row_out[i] / row_sum;
    }
}
