// scores: [batch, n_kv_heads, group_size, q_len, ctx_len] flattened, mutated
// in place. mask: [batch, q_len, ctx_len] flattened, broadcast across
// n_kv_heads and group_size (same mask applies to every Q head sharing a
// KV head, and every KV head - batch_mask never depends on either).
extern "C" __global__ void mask_add_broadcast(
    float *scores,
    const float *mask,
    unsigned int n_kv_heads,
    unsigned int group_size,
    unsigned int q_len,
    unsigned int ctx_len,
    unsigned int n_elements
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_elements) {
        unsigned int ci = index % ctx_len;
        unsigned int qi = (index / ctx_len) % q_len;
        unsigned int b = index / (ctx_len * q_len * group_size * n_kv_heads);

        unsigned int mask_index = (b * q_len + qi) * ctx_len + ci;
        scores[index] += mask[mask_index];
    }
}
