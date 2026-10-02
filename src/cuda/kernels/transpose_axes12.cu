// Swaps the middle two axes of a 4D tensor: in [d0, d1, d2, d3] flattened ->
// out [d0, d2, d1, d3] flattened. Candle's `.transpose(1, 2)` is a free
// restrided view; on a raw buffer it's real data movement, which is what
// this kernel does. Its own inverse: calling it again with d1/d2 swapped
// (i.e. passing the output's dims) undoes it.
extern "C" __global__ void transpose_axes12(
    const float *in,
    float *out,
    unsigned int d0,
    unsigned int d1,
    unsigned int d2,
    unsigned int d3,
    unsigned int n_elements
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        unsigned int i3 = index % d3;
        unsigned int i2 = (index / d3) % d2;
        unsigned int i1 = (index / (d3 * d2)) % d1;
        unsigned int i0 = index / (d3 * d2 * d1);

        unsigned int out_index = ((i0 * d2 + i2) * d1 + i1) * d3 + i3;
        out[out_index] = in[index];
    }
}
