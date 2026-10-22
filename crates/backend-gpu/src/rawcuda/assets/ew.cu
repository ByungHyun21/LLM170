
#include <cuda_fp16.h>
#include "cast_common.cuh"
__device__ __forceinline__ float ew_expf(float x)
{
    return __expf(x);
}

extern "C" __global__ void ew(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    float* __restrict__ y32,
    unsigned short* __restrict__ y16,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float e = ew_expf(-v);
    float act = (v / (1.0f + e)) * u[j];
    y[j] = act;
    if (y32 != (float*)0) {
        const __half h = __float2half_rn(act);
        y32[j] = __half2float(h);
        if (y16 != (unsigned short*)0) {
            y16[j] = __half_as_ushort(h);
        }
    } else if (y16 != (unsigned short*)0) {
        y16[j] = __half_as_ushort(__float2half_rn(act));
    }
}


extern "C" __global__ void ew_gelu(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    float* __restrict__ y32,
    unsigned short* __restrict__ y16,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float x3 = v * v * v;
    float act = 0.5f * v * (1.0f + tanhf(0.7978846f * (v + 0.044715f * x3))) * u[j];
    y[j] = act;
    if (y32 != (float*)0) {
        const __half h = __float2half_rn(act);
        y32[j] = __half2float(h);
        if (y16 != (unsigned short*)0) {
            y16[j] = __half_as_ushort(h);
        }
    } else if (y16 != (unsigned short*)0) {
        y16[j] = __half_as_ushort(__float2half_rn(act));
    }
}

