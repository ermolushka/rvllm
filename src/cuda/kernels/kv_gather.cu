extern "C" __global__ void kv_gather(
    const float *cache,
    const unsigned int *block_idx,
    float *out,
    unsigned int batch,
    unsigned int max_blocks,
    unsigned int ctx_len,
    unsigned int block_size,
    unsigned int n_kv_heads,
    unsigned int head_dim,
    unsigned int n_elements
) {
    // one thread per scalar element of out: [batch, n_kv_heads, ctx_len, head_dim]
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        // decompose flat index into (b, head, pos, d), head_dim innermost
        unsigned int d = index % head_dim;
        unsigned int pos = (index / head_dim) % ctx_len;
        unsigned int head = (index / (head_dim * ctx_len)) % n_kv_heads;
        unsigned int b = index / (head_dim * ctx_len * n_kv_heads);

        // pos splits into which block (within this sequence's block table)
        // and which slot inside that block
        unsigned int block_pos = pos / block_size;
        unsigned int offset = pos % block_size;
        unsigned int block = block_idx[b * max_blocks + block_pos];

        // flat offset into cache: [num_blocks, block_size, n_kv_heads, head_dim]
        unsigned int src = block * block_size * n_kv_heads * head_dim
            + offset * n_kv_heads * head_dim
            + head * head_dim
            + d;

        out[index] = cache[src];
    }
}
