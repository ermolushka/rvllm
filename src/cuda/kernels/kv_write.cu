extern "C" __global__ void kv_write(
    const float *k,
    const float *v,
    const unsigned int *block_ids,
    const unsigned int *offsets,
    float *out_k,
    float *out_v,
    unsigned int n_kv_heads,
    unsigned int head_dim,
    unsigned int block_size,
    unsigned int n_elements
) {
    // one thread per scalar element of k/v
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        // decompose flat index into (token, head, d) over [n_tokens, n_kv_heads, head_dim]
        unsigned int d = index % head_dim;
        // which head's vector
        unsigned int head = (index / head_dim) % n_kv_heads;
        // which vector it belongs to
        unsigned int token = index / (head_dim * n_kv_heads);

        // this token's destination slot in the cache
        unsigned int block = block_ids[token];
        unsigned int offset = offsets[token];

        // flat offset into out_k/out_v: [num_blocks, block_size, n_kv_heads, head_dim]
        unsigned int dest = block * block_size * n_kv_heads * head_dim
            + offset * n_kv_heads * head_dim
            + head * head_dim
            + d;

        // no synchronization needed, every thread writes a single scalar to a distinct slot
        out_k[dest] = k[index];
        out_v[dest] = v[index];
    }
}
