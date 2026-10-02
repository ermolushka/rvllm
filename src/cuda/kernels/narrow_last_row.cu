// x: [batch, q_len, hidden_dim] flattened. out: [batch, hidden_dim]
// flattened, each batch item's last (q_len-th) row only.
extern "C" __global__ void narrow_last_row(
    const float *x,
    float *out,
    unsigned int q_len,
    unsigned int hidden_dim,
    unsigned int n_elements
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        unsigned int d = index % hidden_dim;
        unsigned int b = index / hidden_dim;
        unsigned int src = (b * q_len + (q_len - 1)) * hidden_dim + d;
        out[index] = x[src];
    }
}
