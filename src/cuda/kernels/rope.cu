extern "C" __global__ void rope(
    const float *x,
    const float *cos,
    const float *sin,
    float *out,
    unsigned int head_dim,
    unsigned int half,
    unsigned int n_pairs
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n_pairs) {
        unsigned int row = index / half;
        unsigned int j = index % half;
        float first = x[row * head_dim + j];
        float second = x[row * head_dim + half + j];
        float c = cos[row * half + j];
        float s = sin[row * half + j];
        float first_rot  = first * c - second * s;
        float second_rot = first * s + second * c;
        out[row * head_dim + j] = first_rot;
        out[row * head_dim + half + j] = second_rot;
    }
}
