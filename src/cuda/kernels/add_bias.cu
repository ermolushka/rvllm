// x: [n_elements / n_cols, n_cols] flattened, mutated in place. bias:
// [n_cols]; every row gets the same bias added (Qwen2's Q/K/V projections).
extern "C" __global__ void add_bias(float *x, const float *bias, unsigned int n_cols, unsigned int n_elements) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        x[index] += bias[index % n_cols];
    }
}
