#pragma once
#include "cast_common.cuh"

#define G4_LANES 64

#define GEMV_TR 8

#define MMA_KC 32

__device__ __forceinline__ double shfl_down_f64(double v, int off) {
    unsigned lo = __double2loint(v), hi = __double2hiint(v);
    lo = __shfl_down_sync(0xffffffffu, lo, off);
    hi = __shfl_down_sync(0xffffffffu, hi, off);
    return __hiloint2double(hi, lo);
}

__device__ __forceinline__ float b2f(unsigned short v) {
    return __uint_as_float(((unsigned)v) << 16);
}

__device__ __forceinline__ unsigned pk2bf(const unsigned short* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 16);
}

__device__ __forceinline__ unsigned hilo_pack(float v) {
    const unsigned short h = f2h(v);
    const unsigned short l = f2h(v - h2f(h));
    return (unsigned)h | ((unsigned)l << 16);
}

__device__ __forceinline__ unsigned hilo_hi(unsigned p0, unsigned p1) {
    return (p0 & 0x0000ffffu) | (p1 << 16);
}

__device__ __forceinline__ unsigned hilo_lo(unsigned p0, unsigned p1) {
    return (p0 >> 16) | (p1 & 0xffff0000u);
}

__device__ __forceinline__ unsigned short deq_hi_e(unsigned qw, int j, float sc) {
    const int nib = (int)((qw >> (4 * j)) & 0xFu) - 8;
    return f2h((float)nib * sc);
}

__device__ __forceinline__ void mma_f16(float* c, unsigned a0, unsigned a1,
                                        unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(sa), "l"(gmem));
}
