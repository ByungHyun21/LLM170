extern "C" __global__ void exl3_norm_resid_p(
    float* __restrict__ x,
    const float* __restrict__ nw_row,
    const float* __restrict__ ab,
    float* __restrict__ xn,
    int t_len)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * 5120;
    float v0 = x[base + tid] + ab[base + tid];
    float v1 = x[base + 1024 + tid] + ab[base + 1024 + tid];
    float v2 = x[base + 2048 + tid] + ab[base + 2048 + tid];
    float v3 = x[base + 3072 + tid] + ab[base + 3072 + tid];
    float v4 = x[base + 4096 + tid] + ab[base + 4096 + tid];
    float ss = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4;
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / 5120.0f + 1e-6f);
    xn[base + tid] = v0 * inv * nw_row[tid];
    xn[base + 1024 + tid] = v1 * inv * nw_row[1024 + tid];
    xn[base + 2048 + tid] = v2 * inv * nw_row[2048 + tid];
    xn[base + 3072 + tid] = v3 * inv * nw_row[3072 + tid];
    xn[base + 4096 + tid] = v4 * inv * nw_row[4096 + tid];
    x[base + tid] = v0;
    x[base + 1024 + tid] = v1;
    x[base + 2048 + tid] = v2;
    x[base + 3072 + tid] = v3;
    x[base + 4096 + tid] = v4;
}
// 마커 nrp

// EXL3 ew(silu·mul) — T=1 디코드용: y[n] = silu(g[n])·u[n].
extern "C" __global__ void exl3_ew(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    y[j] = (v / (1.0f + __expf(-v))) * u[j];
}

// exl3_argmax — [n] 로짓에서 최대 인덱스 1개(단일 WG 리덕션, n<=1M).
extern "C" __global__ void exl3_argmax(
    const float* __restrict__ lg,
    unsigned* __restrict__ out,
    int n)
{
    __shared__ float sv[1024];
    __shared__ unsigned si[1024];
    int tid = threadIdx.x;
    float best = -1e30f;
    unsigned idx = 0u;
    for (int i = tid; i < n; i += 1024) {
        float v = lg[i];
        if (v > best) {
            best = v;
            idx = (unsigned)i;
        }
    }
    sv[tid] = best;
    si[tid] = idx;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) {
            if (sv[tid + st] > sv[tid]) {
                sv[tid] = sv[tid + st];
                si[tid] = si[tid + st];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = si[0];
}

#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

