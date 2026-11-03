// Shared header, prepended to every kernel source before the one-shot runtime
// compile (the Metal analogue of NVRTC: no offline `metal` toolchain needed).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Sum / max over a whole threadgroup, result visible to every thread. A simd
// group (32 lanes on Apple GPUs) reduces in registers with simd_sum/simd_max,
// lane 0 of each simd group parks its partial in threadgroup memory, and every
// thread then folds the (at most 32) partials. `scratch` needs 32 floats. The
// trailing barrier lets the caller reuse `scratch` straight away.
inline float tg_sum(float v, threadgroup float *scratch, uint sg, uint lane, uint n_sg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < n_sg; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

inline float tg_max(float v, threadgroup float *scratch, uint sg, uint lane, uint n_sg) {
    v = simd_max(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = scratch[0];
    for (uint i = 1; i < n_sg; ++i) best = max(best, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return best;
}
