// Layout shuffling and the attention GEMMs. Everything that cuBLAS did for the
// CUDA path is a hand-written kernel here.

// ---------------------------------------------------------------------------
// Fused decode/prefill attention path (replaces bias + Q/K norm + transposes +
// RoPE + kv_write + kv_gather + scores + mask + softmax + apply).

// Everything between the QKV projection and the attention kernel, per head.
// qkv: [rows, (n_heads + 2 * n_kv_heads) * head_dim] (Q heads, then K, then V).
// One 32-thread threadgroup per (head slot, row): optional bias, optional
// per-head RMS norm (Q and K), RoPE (Q and K), then Q goes to q_out
// [rows, n_heads * head_dim] and K/V are scattered straight into the paged
// cache. flags: bit 0 = bias, bit 1 = Q norm, bit 2 = K norm.
kernel void attn_prep(
    device const float *qkv [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device const float *q_norm [[buffer(2)]],
    device const float *k_norm [[buffer(3)]],
    device const float *cos_t [[buffer(4)]],
    device const float *sin_t [[buffer(5)]],
    device const uint *block_ids [[buffer(6)]],
    device const uint *offsets [[buffer(7)]],
    device float *cache_k [[buffer(8)]],
    device float *cache_v [[buffer(9)]],
    device float *q_out [[buffer(10)]],
    constant uint &n_heads [[buffer(11)]],
    constant uint &n_kv_heads [[buffer(12)]],
    constant uint &head_dim [[buffer(13)]],
    constant uint &block_size [[buffer(14)]],
    constant uint &flags [[buffer(15)]],
    constant float &eps [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint slot = tg.x;
    uint row = tg.y;
    uint half_dim = head_dim / 2;
    uint qkv_dim = (n_heads + 2 * n_kv_heads) * head_dim;
    bool use_bias = (flags & 1u) != 0;
    device const float *src = qkv + (ulong)row * qkv_dim + (ulong)slot * head_dim;
    device const float *b = bias + (ulong)slot * head_dim;

    if (slot >= n_heads + n_kv_heads) {
        uint kvh = slot - n_heads - n_kv_heads;
        ulong dest = (((ulong)block_ids[row] * block_size + offsets[row]) * n_kv_heads + kvh) * head_dim;
        for (uint d = lane; d < head_dim; d += 32) cache_v[dest + d] = use_bias ? src[d] + b[d] : src[d];
        return;
    }

    bool is_q = slot < n_heads;
    bool use_norm = (flags & (is_q ? 2u : 4u)) != 0;
    device const float *w = is_q ? q_norm : k_norm;
    float rms = 1.0f;
    if (use_norm) {
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) {
            float v = use_bias ? src[d] + b[d] : src[d];
            partial += v * v;
        }
        rms = sqrt(simd_sum(partial) / (float)head_dim + eps);
    }

    ulong dest = 0;
    if (!is_q) {
        uint kvh = slot - n_heads;
        dest = (((ulong)block_ids[row] * block_size + offsets[row]) * n_kv_heads + kvh) * head_dim;
    }
    for (uint j = lane; j < half_dim; j += 32) {
        float first = use_bias ? src[j] + b[j] : src[j];
        float second = use_bias ? src[half_dim + j] + b[half_dim + j] : src[half_dim + j];
        if (use_norm) {
            first = (first / rms) * w[j];
            second = (second / rms) * w[half_dim + j];
        }
        float c = cos_t[(ulong)row * half_dim + j];
        float s = sin_t[(ulong)row * half_dim + j];
        float r1 = first * c - second * s;
        float r2 = first * s + second * c;
        if (is_q) {
            device float *dst = q_out + (ulong)row * n_heads * head_dim + (ulong)slot * head_dim;
            dst[j] = r1;
            dst[half_dim + j] = r2;
        } else {
            cache_k[dest + j] = r1;
            cache_k[dest + half_dim + j] = r2;
        }
    }
}

// Attention straight over the paged cache, one threadgroup per (G query heads
// that share a KV head, query position, sequence). Eight simd groups split the
// keys between them; each runs an online softmax (running max, sum and weighted
// V, per head) over its keys and the eight partial results are merged at the
// end. K and V are loaded once per key and used by all G heads. Keys the mask
// hides (-inf) are skipped entirely, V load included. Each lane owns every
// 32nd element of the head, so K/V loads are contiguous across the simd group.
// Specialized per G and DPL (= ceil(head_dim / 32)) so the per-lane state lives
// in registers; the host picks the variant (see ATTN_DECODE_KERNELS). The merge
// buffer is ATT_SG * G * DPL * 32 floats, so G * DPL stays under 32 (32 KB limit).
// q, out: [rows, n_heads * head_dim]; mask: [batch, q_len, ctx_len];
// block_idx: [batch, max_blocks]; cache: [num_blocks, block_size, n_kv, head_dim].
#define ATT_SG 8

template <uint G, uint DPL>
inline void attn_decode_impl(
    device const float *q,
    device const float *cache_k,
    device const float *cache_v,
    device const uint *block_idx,
    device const float *mask,
    device float *out,
    uint n_heads,
    uint n_kv_heads,
    uint head_dim,
    uint block_size,
    uint max_blocks,
    uint ctx_len,
    uint q_len,
    float scale,
    uint3 tg,
    uint tid,
    uint sg,
    uint lane,
    threadgroup float *m_s,   // [ATT_SG][G]
    threadgroup float *l_s,   // [ATT_SG][G]
    threadgroup float *acc_s  // [ATT_SG][G][DPL * 32]
) {
    uint first_head = tg.x * G;
    uint kvh = first_head / (n_heads / n_kv_heads);
    uint qi = tg.y, b = tg.z;
    ulong row = (ulong)b * q_len + qi;

    float qv[G][DPL];
    float acc[G][DPL];
    float m[G];
    float l[G];
    _Pragma("clang loop unroll(full)")
    for (uint g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < DPL; ++i) {
            uint d = lane + 32 * i;
            qv[g][i] = d < head_dim ? q[row * n_heads * head_dim + (ulong)(first_head + g) * head_dim + d] : 0.0f;
            acc[g][i] = 0.0f;
        }
    }

    device const float *mrow = mask + row * ctx_len;
    device const uint *blocks = block_idx + (ulong)b * max_blocks;
    // Each simd group takes chunks of 32 consecutive keys. Lane j first loads
    // the mask and cache address of key j of the chunk (one memory round trip
    // for all 32), then the keys are walked with those broadcast, so a key's
    // critical path is just its K/V loads, not mask -> block table -> K.
    for (uint c0 = sg * 32; c0 < ctx_len; c0 += ATT_SG * 32) {
        uint cl = c0 + lane;
        float mk_lane = cl < ctx_len ? mrow[cl] : -INFINITY;
        ulong base_lane = 0;
        if (mk_lane != -INFINITY) {
            base_lane = (((ulong)blocks[cl / block_size] * block_size + cl % block_size) * n_kv_heads + kvh) * head_dim;
        }
        uint base_lo = (uint)base_lane;
        uint base_hi = (uint)(base_lane >> 32);
        for (uint j = 0; j < 32; ++j) {
            float mk = simd_broadcast(mk_lane, j);
            if (mk == -INFINITY) continue; // uniform across the simd group
            ulong base = ((ulong)simd_broadcast(base_hi, j) << 32) | simd_broadcast(base_lo, j);
            float kv[DPL];
            float vv[DPL];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < DPL; ++i) {
                uint d = lane + 32 * i;
                kv[i] = d < head_dim ? cache_k[base + d] : 0.0f;
                vv[i] = d < head_dim ? cache_v[base + d] : 0.0f;
            }
            _Pragma("clang loop unroll(full)")
            for (uint g = 0; g < G; ++g) {
                float p = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < DPL; ++i) p += qv[g][i] * kv[i];
                float s = simd_sum(p) * scale + mk;
                float m_new = max(m[g], s);
                float corr = exp(m[g] - m_new);
                float e = exp(s - m_new);
                l[g] = l[g] * corr + e;
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < DPL; ++i) acc[g][i] = acc[g][i] * corr + e * vv[i];
                m[g] = m_new;
            }
        }
    }

    _Pragma("clang loop unroll(full)")
    for (uint g = 0; g < G; ++g) {
        if (lane == 0) {
            m_s[sg * G + g] = m[g];
            l_s[sg * G + g] = l[g];
        }
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < DPL; ++i) acc_s[(sg * G + g) * DPL * 32 + lane + 32 * i] = acc[g][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < G * head_dim; idx += ATT_SG * 32) {
        uint g = idx / head_dim;
        uint d = idx % head_dim;
        float big_m = m_s[g];
        for (uint s = 1; s < ATT_SG; ++s) big_m = max(big_m, m_s[s * G + g]);
        float big_l = 0.0f;
        float o = 0.0f;
        for (uint s = 0; s < ATT_SG; ++s) {
            float w = exp(m_s[s * G + g] - big_m);
            big_l += l_s[s * G + g] * w;
            o += acc_s[(s * G + g) * DPL * 32 + d] * w;
        }
        out[row * n_heads * head_dim + (ulong)(first_head + g) * head_dim + d] = o / big_l;
    }
}

#define ATTN_DECODE_KERNEL(NAME, G_, DPL_)                                               \
    kernel void NAME(                                                                    \
        device const float *q [[buffer(0)]],                                             \
        device const float *cache_k [[buffer(1)]],                                       \
        device const float *cache_v [[buffer(2)]],                                       \
        device const uint *block_idx [[buffer(3)]],                                      \
        device const float *mask [[buffer(4)]],                                          \
        device float *out [[buffer(5)]],                                                 \
        constant uint &n_heads [[buffer(6)]],                                            \
        constant uint &n_kv_heads [[buffer(7)]],                                         \
        constant uint &head_dim [[buffer(8)]],                                           \
        constant uint &block_size [[buffer(9)]],                                         \
        constant uint &max_blocks [[buffer(10)]],                                        \
        constant uint &ctx_len [[buffer(11)]],                                           \
        constant uint &q_len [[buffer(12)]],                                             \
        constant float &scale [[buffer(13)]],                                            \
        uint3 tg [[threadgroup_position_in_grid]],                                       \
        uint tid [[thread_index_in_threadgroup]],                                        \
        uint sg [[simdgroup_index_in_threadgroup]],                                      \
        uint lane [[thread_index_in_simdgroup]]                                          \
    ) {                                                                                  \
        threadgroup float m_s[ATT_SG * G_];                                              \
        threadgroup float l_s[ATT_SG * G_];                                              \
        threadgroup float acc_s[ATT_SG * G_ * DPL_ * 32];                                \
        attn_decode_impl<G_, DPL_>(q, cache_k, cache_v, block_idx, mask, out, n_heads,   \
            n_kv_heads, head_dim, block_size, max_blocks, ctx_len, q_len, scale, tg, tid, \
            sg, lane, m_s, l_s, acc_s);                                                  \
    }

ATTN_DECODE_KERNEL(attn_decode_g1_d2, 1, 2)
ATTN_DECODE_KERNEL(attn_decode_g2_d2, 2, 2)
ATTN_DECODE_KERNEL(attn_decode_g3_d2, 3, 2)
ATTN_DECODE_KERNEL(attn_decode_g4_d2, 4, 2)
ATTN_DECODE_KERNEL(attn_decode_g6_d2, 6, 2)
ATTN_DECODE_KERNEL(attn_decode_g7_d2, 7, 2)
ATTN_DECODE_KERNEL(attn_decode_g8_d2, 8, 2)
ATTN_DECODE_KERNEL(attn_decode_g1_d4, 1, 4)
ATTN_DECODE_KERNEL(attn_decode_g2_d4, 2, 4)
ATTN_DECODE_KERNEL(attn_decode_g3_d4, 3, 4)
ATTN_DECODE_KERNEL(attn_decode_g4_d4, 4, 4)
ATTN_DECODE_KERNEL(attn_decode_g6_d4, 6, 4)
ATTN_DECODE_KERNEL(attn_decode_g7_d4, 7, 4)
ATTN_DECODE_KERNEL(attn_decode_g1_d8, 1, 8)
ATTN_DECODE_KERNEL(attn_decode_g2_d8, 2, 8)
ATTN_DECODE_KERNEL(attn_decode_g3_d8, 3, 8)
