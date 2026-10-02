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
        // which row it takes care of
        unsigned int row = index / half;
        // which pair within that row
        unsigned int j = index % half;
        // it loads one pair using offsets
        float first = x[row * head_dim + j];
        float second = x[row * head_dim + half + j];
        // pre comouted values
        float c = cos[row * half + j];
        float s = sin[row * half + j];
        // actual rotation
        float first_rot  = first * c - second * s;
        float second_rot = first * s + second * c;
        // store it back
        out[row * head_dim + j] = first_rot;
        out[row * head_dim + half + j] = second_rot;
    }
}
