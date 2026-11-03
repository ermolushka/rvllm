// Layout shuffling and the attention GEMMs. Everything that cuBLAS did for the
// CUDA path is a hand-written kernel here.

// RoPE on [n_rows, head_dim]; cos/sin are [n_rows, half], one row per x row.
kernel void rope(
    device const float *x [[buffer(0)]],
    device const float *cos_t [[buffer(1)]],
    device const float *sin_t [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &half_dim [[buffer(5)]],
    constant uint &n_pairs [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n_pairs) {
        uint row = i / half_dim;
        uint j = i % half_dim;
        float first = x[(ulong)row * head_dim + j];
        float second = x[(ulong)row * head_dim + half_dim + j];
        float c = cos_t[(ulong)row * half_dim + j];
        float s = sin_t[(ulong)row * half_dim + j];
        out[(ulong)row * head_dim + j] = first * c - second * s;
        out[(ulong)row * head_dim + half_dim + j] = first * s + second * c;
    }
}

// [d0, d1, d2, d3] -> [d0, d2, d1, d3].
kernel void transpose_axes12(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &d0 [[buffer(2)]],
    constant uint &d1 [[buffer(3)]],
    constant uint &d2 [[buffer(4)]],
    constant uint &d3 [[buffer(5)]],
    constant uint &n [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint i3 = i % d3;
        uint i2 = (i / d3) % d2;
        uint i1 = (i / (d3 * d2)) % d1;
        uint i0 = i / (d3 * d2 * d1);
        out[(((ulong)i0 * d2 + i2) * d1 + i1) * d3 + i3] = in[i];
    }
}

// Scatter this step's K and V into the paged cache
// [num_blocks, block_size, n_kv_heads, head_dim].
kernel void kv_write(
    device const float *k [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device const uint *block_ids [[buffer(2)]],
    device const uint *offsets [[buffer(3)]],
    device float *out_k [[buffer(4)]],
    device float *out_v [[buffer(5)]],
    constant uint &n_kv_heads [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &block_size [[buffer(8)]],
    constant uint &n [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint d = i % head_dim;
        uint head = (i / head_dim) % n_kv_heads;
        uint token = i / (head_dim * n_kv_heads);
        ulong dest = ((ulong)block_ids[token] * block_size + offsets[token]) * n_kv_heads * head_dim
            + (ulong)head * head_dim + d;
        out_k[dest] = k[i];
        out_v[dest] = v[i];
    }
}

// Gather each sequence's blocks into [batch, n_kv_heads, ctx_len, head_dim].
kernel void kv_gather(
    device const float *cache [[buffer(0)]],
    device const uint *block_idx [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &max_blocks [[buffer(3)]],
    constant uint &ctx_len [[buffer(4)]],
    constant uint &block_size [[buffer(5)]],
    constant uint &n_kv_heads [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &n [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint d = i % head_dim;
        uint pos = (i / head_dim) % ctx_len;
        uint head = (i / (head_dim * ctx_len)) % n_kv_heads;
        uint b = i / (head_dim * ctx_len * n_kv_heads);
        uint block = block_idx[b * max_blocks + pos / block_size];
        ulong src = ((ulong)block * block_size + pos % block_size) * n_kv_heads * head_dim
            + (ulong)head * head_dim + d;
        out[i] = cache[src];
    }
}

// scores: [batch, n_kv_heads, group * q_len, ctx_len]; mask: [batch, q_len, ctx_len].
kernel void mask_add_broadcast(
    device float *scores [[buffer(0)]],
    device const float *mask [[buffer(1)]],
    constant uint &n_kv_heads [[buffer(2)]],
    constant uint &group_size [[buffer(3)]],
    constant uint &q_len [[buffer(4)]],
    constant uint &ctx_len [[buffer(5)]],
    constant uint &n [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint ci = i % ctx_len;
        uint qi = (i / ctx_len) % q_len;
        uint b = i / (ctx_len * q_len * group_size * n_kv_heads);
        scores[i] += mask[((ulong)b * q_len + qi) * ctx_len + ci];
    }
}

// scores[bk, r, c] = scale * dot(q[bk, r, :], k[bk, c, :]); bk = batch * n_kv_heads,
// r runs over group_size * q_len. One thread per score; grid is (ctx_len, rows, bk).
kernel void attention_scores(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &ctx_len [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint c = gid.x, r = gid.y, bk = gid.z;
    if (c >= ctx_len || r >= rows) return;
    device const float *qs = q + ((ulong)bk * rows + r) * head_dim;
    device const float *ks = k + ((ulong)bk * ctx_len + c) * head_dim;
    float acc = 0.0f;
    if (head_dim % 4 == 0) {
        // Uniform branch: 16-byte loads are only aligned when head_dim is a multiple of 4.
        device const float4 *qr = (device const float4 *)qs;
        device const float4 *kr = (device const float4 *)ks;
        for (uint d = 0; d < head_dim / 4; ++d) acc += dot(qr[d], kr[d]);
    } else {
        for (uint d = 0; d < head_dim; ++d) acc += qs[d] * ks[d];
    }
    scores[((ulong)bk * rows + r) * ctx_len + c] = acc * scale;
}

// out[bk, r, d] = sum_c probs[bk, r, c] * v[bk, c, d]. Grid is (head_dim, rows, bk):
// neighbouring threads read neighbouring d of v, so the loads coalesce.
kernel void attention_apply(
    device const float *probs [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &ctx_len [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, r = gid.y, bk = gid.z;
    if (d >= head_dim || r >= rows) return;
    device const float *p = probs + ((ulong)bk * rows + r) * ctx_len;
    device const float *vb = v + (ulong)bk * ctx_len * head_dim + d;
    float acc = 0.0f;
    for (uint c = 0; c < ctx_len; ++c) acc += p[c] * vb[(ulong)c * head_dim];
    out[((ulong)bk * rows + r) * head_dim + d] = acc;
}
