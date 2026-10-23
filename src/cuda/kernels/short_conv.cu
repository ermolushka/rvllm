// LFM2's short convolution, after the in_proj matmul. proj: [rows, 3 * hidden]
// with B, C and x side by side (in that order). Computes, per row r and
// channel c:  B[r, c] * x[r, c]  is the signal the causal filter sees.
//
// short_conv_bx: bx[r, c] = B[r, c] * x[r, c]. rows is [batch * q_len].
extern "C" __global__ void short_conv_bx(const float *proj, float *bx, unsigned int hidden, unsigned int n_elements) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        unsigned int row = i / hidden;
        unsigned int c = i % hidden;
        bx[i] = proj[row * 3 * hidden + c] * proj[row * 3 * hidden + 2 * hidden + c];
    }
}

// short_conv_filter_gate: out[r, c] = C[r, c] * sum_k filter[c, k] * window[r, k, c]
// where window holds, for each batch item, the kernel - 1 bx rows that came
// before its q_len new rows followed by those new rows:
// window: [batch, kernel - 1 + q_len, hidden]. filter: [hidden, kernel], oldest
// tap first. The output of new row t mixes window rows t .. t + kernel - 1, so
// the token itself and the kernel - 1 before it.
extern "C" __global__ void short_conv_filter_gate(
    const float *window,
    const float *filter,
    const float *proj,
    float *out,
    unsigned int batch,
    unsigned int q_len,
    unsigned int hidden,
    unsigned int kernel,
    unsigned int n_elements
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        unsigned int c = i % hidden;
        unsigned int t = (i / hidden) % q_len;
        unsigned int b = i / (hidden * q_len);
        unsigned int window_len = kernel - 1 + q_len;
        const float *item = window + (size_t)b * window_len * hidden;
        float acc = 0.0f;
        for (unsigned int k = 0; k < kernel; ++k) {
            acc += filter[c * kernel + k] * item[(t + k) * hidden + c];
        }
        // C is the second chunk of proj's row for this (b, t).
        unsigned int row = b * q_len + t;
        out[i] = proj[row * 3 * hidden + hidden + c] * acc;
    }
}
