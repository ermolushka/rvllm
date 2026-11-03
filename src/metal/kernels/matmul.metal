// y = x @ w^T with x [rows, in] F32, w [out, in] F16 (the Candle/GGUF Linear
// layout), y [rows, out] F32. Two kernels, picked by the host on row count:
//
//  * linear_small: decode. Few rows, so the job is purely "stream the weights
//    once at full memory bandwidth". One simd group owns one output feature and
//    its 32 lanes walk that weight row with 8-byte half4 loads; simd_sum adds
//    the lanes up. Up to MV_ROWS rows share every weight load.
//  * linear_gemm_*: prefill and bigger batches. An output tile per threadgroup,
//    staged through threadgroup memory, multiplied with the hardware 8x8
//    simdgroup_matrix units. 32x64 tile: a 64x64 variant was tried and was no
//    faster once the accumulator loops were force-unrolled (before that it
//    spilled its 16 accumulators to memory and ran 10x slower).

#define MV_ROWS 8
#define MV_SG 4

kernel void linear_small(
    device const float *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_f [[buffer(4)]],
    constant uint &out_f [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint n = tg.x * MV_SG + sg;
    // All 32 lanes of a simd group share n, so this exit is uniform.
    if (n >= out_f) return;
    uint r0 = tg.y * MV_ROWS;
    uint nr = min((uint)MV_ROWS, rows - r0);

    device const half4 *wrow = (device const half4 *)(w + (ulong)n * in_f);
    uint n4 = in_f / 4;
    float acc[MV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = lane; i < n4; i += 32) {
        float4 wv = float4(wrow[i]);
        for (uint r = 0; r < nr; ++r) {
            float4 xv = ((device const float4 *)(x + (ulong)(r0 + r) * in_f))[i];
            acc[r] += dot(wv, xv);
        }
    }
    for (uint r = 0; r < nr; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0) y[(ulong)(r0 + r) * out_f + n] = s;
    }
}

// linear_mma_small: decode batches of up to 32 rows. A threadgroup owns 8 output
// features, and its 4 simd groups split K between them (interleaved, 8 wide), so
// even a 320-wide projection spreads over 40 threadgroups x 4 simd groups instead
// of the 5 tiles the GEMM would get. Weight fragments are loaded straight from
// device memory into half 8x8 matrices (no staging, no barriers in the loop);
// the x tile is 8 rows x 8 k, bounced through a small per-simd-group scratch so
// rows past `rows` read as zero. Row tiles of 8 share each weight fragment (up
// to 4 of them = 32 rows). The four partial sums are added at the end.
// Needs in_f % 8 == 0 and out_f % 8 == 0 (the host checks).
kernel void linear_mma_small(
    device const float *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &in_f [[buffer(4)]],
    constant uint &out_f [[buffer(5)]],
    uint tgi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float a_scratch[4][4][64]; // [simd group][row tile][8x8]
    threadgroup float part[4][4][64];      // [simd group][row tile][8x8]
    uint n0 = tgi * 8;
    uint n_tiles = (rows + 7) / 8;
    uint r = lane / 4;       // this lane's row within an 8x8 tile
    uint c = (lane % 4) * 2; // and its first of two columns

    simdgroup_float8x8 acc[4];
    _Pragma("clang loop unroll(full)")
    for (uint t = 0; t < 4; ++t) acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = sg * 8; k < in_f; k += 32) {
        simdgroup_half8x8 b;
        // w is [n][k]; the product needs [k][n], hence the transposed load.
        simdgroup_load(b, w + (ulong)n0 * in_f + k, in_f, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4; ++t) {
            if (t < n_tiles) {
                uint row = t * 8 + r;
                float2 v = row < rows ? *(device const float2 *)(x + (ulong)row * in_f + k + c) : float2(0.0f);
                *(threadgroup float2 *)(&a_scratch[sg][t][r * 8 + c]) = v;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 a;
                simdgroup_load(a, &a_scratch[sg][t][0], 8);
                simdgroup_multiply_accumulate(acc[t], a, b, acc[t]);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    _Pragma("clang loop unroll(full)")
    for (uint t = 0; t < 4; ++t) simdgroup_store(acc[t], &part[sg][t][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < n_tiles * 64; e += 128) {
        uint t = e / 64;
        uint row = t * 8 + (e % 64) / 8;
        if (row < rows) {
            float sum = part[0][t][e % 64] + part[1][t][e % 64] + part[2][t][e % 64] + part[3][t][e % 64];
            y[(ulong)row * out_f + n0 + (e % 8)] = sum;
        }
    }
}

// Tiled GEMM: a BM x BN output tile per threadgroup of 4 simd groups (2x2), K
// walked in BK=32 slices staged through threadgroup memory. Tile loads are
// 16-byte vectors (when in_f is a multiple of 4) and tile rows are padded by 4
// floats so the simdgroup_load strides do not all land in the same bank.
#define BK 32
#define LDK 36

template <uint BM, uint BN>
inline void linear_gemm_impl(
    device const float *x,
    device const half *w,
    device float *y,
    uint rows,
    uint in_f,
    uint out_f,
    uint2 tg,
    uint tid,
    uint sg,
    threadgroup float *xs, // x tile, [BM][LDK]
    threadgroup float *ws, // w tile, [BN][LDK] (n-major, k contiguous)
    threadgroup float *stage // result tile [BM][BN], aliases xs/ws: they are dead after the last barrier.
    // (Threadgroup arrays can only be declared in the kernel entry point itself.)
) {
    constexpr uint TM = BM / 16; // 8x8 tiles per simd group, down
    constexpr uint TN = BN / 16; // and across
    uint m0 = tg.y * BM;
    uint n0 = tg.x * BN;
    uint sm = sg / 2;
    uint sn = sg % 2;
    bool vec = (in_f % 4) == 0;

    simdgroup_float8x8 acc[TM][TN];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < TM; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < TN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < in_f; k0 += BK) {
        if (vec) {
            for (uint e = tid; e < BM * (BK / 4); e += 128) {
                uint r = e / (BK / 4);
                uint c = (e % (BK / 4)) * 4;
                uint gk = k0 + c;
                uint gm = m0 + r;
                float4 v = float4(0.0f);
                if (gm < rows && gk < in_f) v = *(device const float4 *)(x + (ulong)gm * in_f + gk);
                *(threadgroup float4 *)(xs + r * LDK + c) = v;
            }
            for (uint e = tid; e < BN * (BK / 4); e += 128) {
                uint r = e / (BK / 4);
                uint c = (e % (BK / 4)) * 4;
                uint gk = k0 + c;
                uint gn = n0 + r;
                float4 v = float4(0.0f);
                if (gn < out_f && gk < in_f) v = float4(*(device const half4 *)(w + (ulong)gn * in_f + gk));
                *(threadgroup float4 *)(ws + r * LDK + c) = v;
            }
        } else {
            for (uint e = tid; e < BM * BK; e += 128) {
                uint r = e / BK;
                uint c = e % BK;
                uint gk = k0 + c;
                uint gm = m0 + r;
                xs[r * LDK + c] = (gm < rows && gk < in_f) ? x[(ulong)gm * in_f + gk] : 0.0f;
            }
            for (uint e = tid; e < BN * BK; e += 128) {
                uint r = e / BK;
                uint c = e % BK;
                uint gk = k0 + c;
                uint gn = n0 + r;
                ws[r * LDK + c] = (gn < out_f && gk < in_f) ? float(w[(ulong)gn * in_f + gk]) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[TM];
            simdgroup_float8x8 b[TN];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < TM; ++i)
                simdgroup_load(a[i], xs + (sm * (BM / 2) + i * 8) * LDK + kk, LDK);
            // w is stored [n][k]; the product needs [k][n], hence the transposed load.
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < TN; ++j)
                simdgroup_load(b[j], ws + (sn * (BN / 2) + j * 8) * LDK + kk, LDK, ulong2(0, 0), true);
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < TM; ++i)
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage the result through threadgroup memory so edge tiles can be bounds
    // checked (simdgroup_store would write a full 8x8 regardless).
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < TM; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < TN; ++j)
            simdgroup_store(acc[i][j], stage + (sm * (BM / 2) + i * 8) * BN + (sn * (BN / 2) + j * 8), BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = e / BN;
        uint c = e % BN;
        if (m0 + r < rows && n0 + c < out_f) y[(ulong)(m0 + r) * out_f + n0 + c] = stage[e];
    }
}

#define LINEAR_GEMM_KERNEL(NAME, BM_, BN_)                                              \
    kernel void NAME(                                                                    \
        device const float *x [[buffer(0)]],                                             \
        device const half *w [[buffer(1)]],                                              \
        device float *y [[buffer(2)]],                                                   \
        constant uint &rows [[buffer(3)]],                                               \
        constant uint &in_f [[buffer(4)]],                                               \
        constant uint &out_f [[buffer(5)]],                                              \
        uint2 tg [[threadgroup_position_in_grid]],                                       \
        uint tid [[thread_index_in_threadgroup]],                                        \
        uint sg [[simdgroup_index_in_threadgroup]]                                       \
    ) {                                                                                  \
        /* One array: the x and w tiles, reused as the result tile at the end. */       \
        threadgroup float tiles[(BM_ + BN_) * LDK];                                      \
        linear_gemm_impl<BM_, BN_>(x, w, y, rows, in_f, out_f, tg, tid, sg, tiles, tiles + BM_ * LDK, tiles); \
    }

LINEAR_GEMM_KERNEL(linear_gemm_32x64, 32, 64)
