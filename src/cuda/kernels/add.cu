extern "C" __global__ void add(
    const float *a,
    const float *b,
    float *out,
    unsigned int n
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        out[index] = a[index] + b[index];
    }
}
