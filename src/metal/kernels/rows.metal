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

// Residual add fused with the RMS norm that follows it: sum_out = a + b, and
// norm_out = rmsnorm(sum_out) * weight. One threadgroup per row. The sum is
// recomputed in the second pass instead of re-read, so no thread depends on
// another's store.
kernel void add_rmsnorm(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *sum_out [[buffer(3)]],
    device float *norm_out [[buffer(4)]],
    constant uint &hidden_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float scratch[32];
    ulong base = (ulong)row * hidden_dim;

    float partial = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float v = a[base + i] + b[base + i];
        sum_out[base + i] = v;
        partial += v * v;
    }
    float sum_sq = tg_sum(partial, scratch, sg, lane, tg_size / 32);
    float rms = sqrt(sum_sq / (float)hidden_dim + eps);

    for (uint i = tid; i < hidden_dim; i += tg_size) {
        norm_out[base + i] = ((a[base + i] + b[base + i]) / rms) * weight[i];
    }
}
