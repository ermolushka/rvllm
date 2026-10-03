// Raw 16-bit halves (no cuda_fp16.h, which NVRTC may not find); the cvt
// does round-to-nearest-even, matching host-side half::f16::from_f32.
extern "C" __global__ void f32_to_f16(
    const float *input,
    unsigned short *out,
    unsigned int n
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(input[index]));
        out[index] = h;
    }
}
