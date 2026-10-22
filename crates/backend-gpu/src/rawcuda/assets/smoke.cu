#include <cuda_bf16.h>

extern "C" __global__ void llm170_mma_smoke(const float* a, const float* b, float* c) {
    __shared__ float sa[16 * 16];
    __shared__ float sb[16 * 8];
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
        sa[i] = a[i];
    }
    for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) {
        sb[i] = b[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    auto pk = [](float x, float y) {
        __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
        return *reinterpret_cast<unsigned*>(&h);
    };
    const unsigned a0 = pk(sa[g * 16 + 2 * t], sa[g * 16 + 2 * t + 1]);
    const unsigned a1 = pk(sa[(g + 8) * 16 + 2 * t], sa[(g + 8) * 16 + 2 * t + 1]);
    const unsigned a2 = pk(sa[g * 16 + 2 * t + 8], sa[g * 16 + 2 * t + 9]);
    const unsigned a3 = pk(sa[(g + 8) * 16 + 2 * t + 8], sa[(g + 8) * 16 + 2 * t + 9]);
    const unsigned b0 = pk(sb[(2 * t) * 8 + g], sb[(2 * t + 1) * 8 + g]);
    const unsigned b1 = pk(sb[(2 * t + 8) * 8 + g], sb[(2 * t + 9) * 8 + g]);
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    c[g * 8 + 2 * t] = c0;
    c[g * 8 + 2 * t + 1] = c1;
    c[(g + 8) * 8 + 2 * t] = c2;
    c[(g + 8) * 8 + 2 * t + 1] = c3;
}
