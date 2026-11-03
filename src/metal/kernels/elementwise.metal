// One thread per element. Every kernel bounds-checks even though the host
// dispatches exactly `n` threads, so a rounded-up grid would also be safe.

kernel void add(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) out[i] = a[i] + b[i];
}

kernel void silu(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float v = in[i];
        out[i] = v * (1.0f / (1.0f + exp(-v)));
    }
}

kernel void silu_gate_multiply(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        out[i] = (g * (1.0f / (1.0f + exp(-g)))) * up[i];
    }
}

// GeGLU gate: gelu_tanh(gate) * up (Gemma's FFN; same formula as candle).
kernel void gelu_gate_multiply(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float x = gate[i];
        const float k = 0.7978845608028654f; // sqrt(2 / pi)
        out[i] = 0.5f * x * (1.0f + tanh(k * (x + 0.044715f * x * x * x))) * up[i];
    }
}

// x: [n / n_cols, n_cols], bias: [n_cols], added to every row in place.
kernel void add_bias(
    device float *x [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant uint &n_cols [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) x[i] += bias[i % n_cols];
}

kernel void scale_inplace(
    device float *x [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) x[i] *= scale;
}

// token_ids: [n_tokens], embedding: [vocab, hidden_dim], out: [n_tokens, hidden_dim].
kernel void embed_lookup(
    device const uint *token_ids [[buffer(0)]],
    device const float *embedding [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint d = i % hidden_dim;
        uint token = i / hidden_dim;
        out[i] = embedding[(ulong)token_ids[token] * hidden_dim + d];
    }
}

// x: [batch, q_len, hidden_dim] -> out: [batch, hidden_dim], each item's last row.
kernel void narrow_last_row(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &q_len [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint d = i % hidden_dim;
        uint b = i / hidden_dim;
        out[i] = x[((ulong)b * q_len + (q_len - 1)) * hidden_dim + d];
    }
}

// Plain copy of n floats. The host binds each buffer at a byte offset, which is
// how sub-ranges are copied (the short-conv window, the per-block conv rows).
kernel void copy_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) dst[i] = src[i];
}

// LFM2 short conv, after in_proj. proj: [rows, 3 * hidden] holding B, C, x
// side by side; bx[r, c] = B[r, c] * x[r, c].
kernel void short_conv_bx(
    device const float *proj [[buffer(0)]],
    device float *bx [[buffer(1)]],
    constant uint &hidden [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint row = i / hidden;
        uint c = i % hidden;
        bx[i] = proj[(ulong)row * 3 * hidden + c] * proj[(ulong)row * 3 * hidden + 2 * hidden + c];
    }
}

// out[r, c] = C[r, c] * sum_k filter[c, k] * window[b, t + k, c] for new row
// r = (b, t). window: [batch, kernel - 1 + q_len, hidden], filter: [hidden, kernel].
kernel void short_conv_filter_gate(
    device const float *window [[buffer(0)]],
    device const float *filter [[buffer(1)]],
    device const float *proj [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &q_len [[buffer(5)]],
    constant uint &hidden [[buffer(6)]],
    constant uint &kernel_size [[buffer(7)]],
    constant uint &n [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        uint c = i % hidden;
        uint t = (i / hidden) % q_len;
        uint b = i / (hidden * q_len);
        uint window_len = kernel_size - 1 + q_len;
        device const float *item = window + (ulong)b * window_len * hidden;
        float acc = 0.0f;
        for (uint k = 0; k < kernel_size; ++k) {
            acc += filter[c * kernel_size + k] * item[(t + k) * hidden + c];
        }
        uint row = b * q_len + t;
        out[i] = proj[(ulong)row * 3 * hidden + hidden + c] * acc;
    }
}
