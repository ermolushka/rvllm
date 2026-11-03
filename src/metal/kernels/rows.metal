// Row-wise kernels: one threadgroup (256 threads = 8 simd groups) per row.

kernel void rmsnorm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float scratch[32];
    device const float *row_in = x + (ulong)row * hidden_dim;
    device float *row_out = out + (ulong)row * hidden_dim;

    float partial = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float v = row_in[i];
        partial += v * v;
    }
    float sum_sq = tg_sum(partial, scratch, sg, lane, tg_size / 32);
    float rms = sqrt(sum_sq / (float)hidden_dim + eps);

    for (uint i = tid; i < hidden_dim; i += tg_size) {
        row_out[i] = (row_in[i] / rms) * weight[i];
    }
}

// Softmax over the last axis. -inf inputs (the attention mask) fall out
// naturally: exp(-inf - finite) == 0. Needs fast-math OFF so -inf survives.
kernel void softmax(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &row_len [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float scratch[32];
    device const float *row_in = x + (ulong)row * row_len;
    device float *row_out = out + (ulong)row * row_len;
    uint n_sg = tg_size / 32;

    float partial_max = -INFINITY;
    for (uint i = tid; i < row_len; i += tg_size) partial_max = max(partial_max, row_in[i]);
    float row_max = tg_max(partial_max, scratch, sg, lane, n_sg);

    float partial_sum = 0.0f;
    for (uint i = tid; i < row_len; i += tg_size) {
        float e = exp(row_in[i] - row_max);
        row_out[i] = e;
        partial_sum += e;
    }
    float row_sum = tg_sum(partial_sum, scratch, sg, lane, n_sg);

    // Each thread re-reads only the slots it wrote itself, so no barrier is
    // needed between the exp pass and this one.
    for (uint i = tid; i < row_len; i += tg_size) row_out[i] = row_out[i] / row_sum;
}

// Index of each row's maximum. Ties go to the lowest index and NaN is ignored,
// same as Sampler::sample_slice's greedy scan, so GPU and CPU pick the same
// token. A row with nothing above -inf returns 0.
kernel void argmax(
    device const float *x [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &row_len [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float vals[256];
    threadgroup uint idxs[256];
    device const float *row_in = x + (ulong)row * row_len;

    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    for (uint i = tid; i < row_len; i += tg_size) {
        float v = row_in[i];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = vals[tid + s];
            uint i2 = idxs[tid + s];
            if (v2 > vals[tid] || (v2 == vals[tid] && i2 < idxs[tid])) {
                vals[tid] = v2;
                idxs[tid] = i2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = idxs[0] == 0xffffffffu ? 0u : idxs[0];
}
