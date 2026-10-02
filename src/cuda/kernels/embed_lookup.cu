// token_ids: [n_tokens]. embedding: [vocab_size, hidden_dim] flattened.
// out: [n_tokens, hidden_dim] flattened.
extern "C" __global__ void embed_lookup(
    const unsigned int *token_ids,
    const float *embedding,
    float *out,
    unsigned int hidden_dim,
    unsigned int n_elements
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        unsigned int d = index % hidden_dim;
        unsigned int token = index / hidden_dim;
        unsigned int row = token_ids[token];
        out[index] = embedding[row * hidden_dim + d];
    }
}
