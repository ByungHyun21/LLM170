#include <cuda_fp16.h>
#include "cast_common.cuh"

extern "C" __global__ void norm_resid(
    float* __restrict__ x,
    const float* __restrict__ nw,
    const float* __restrict__ ab,
    float* __restrict__ xn,
    float* __restrict__ xn32,
    unsigned short* __restrict__ xn16,
    int t_len, int w_off, int hidden)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * hidden;
    int nper = (hidden + 1023) >> 10;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        v[j] = x[e] + ab[e];
        ss += v[j] * v[j];
    }
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / (float)hidden + 1e-6f);
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        float y = v[j] * inv * nw[w_off + (j << 10) + tid];
        xn[e] = y;
        if (xn32 != (float*)0) {
            const __half h = __float2half_rn(y);
            xn32[e] = __half2float(h);
            if (xn16 != (unsigned short*)0) {
                xn16[e] = __half_as_ushort(h);
            }
        } else if (xn16 != (unsigned short*)0) {
            xn16[e] = __half_as_ushort(__float2half_rn(y));
        }
        x[e] = v[j];
    }
}

extern "C" __global__ void rms_rows(
    const float* __restrict__ x,
    const float* __restrict__ nw,
    float* __restrict__ xn,
    int t_len, int w_off, int hidden, float eps)
{
    extern __shared__ float sm[];
    float* sx = sm;
    float* sp = sm + hidden;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    if (t >= t_len) return;
    const int base = t * hidden;
    for (int e = tid; e < hidden; e += nth) {
        sx[e] = x[base + e];
    }
    __syncthreads();
    const int SEG = 32;
    const int chunk = (hidden + SEG - 1) / SEG;
    if (tid < SEG) {
        int lo = tid * chunk;
        int hi = lo + chunk < hidden ? lo + chunk : hidden;
        float part = 0.0f;
        for (int i = lo; i < hi; i++) {
            float v = sx[i];
            part = __fadd_rn(part, __fmul_rn(v, v));
        }
        sp[tid] = part;
    }
    __syncthreads();
    if (tid == 0) {
        double dsum = 0.0;
        for (int u = 0; u < SEG; u++) {
            if (u * chunk >= hidden) break;
            dsum += (double)sp[u];
        }
        double mean = dsum / (double)hidden + (double)eps;
        sp[SEG] = 1.0f / (float)sqrt(mean);
    }
    __syncthreads();
    const float scale = sp[SEG];
    for (int e = tid; e < hidden; e += nth) {
        xn[base + e] = __fmul_rn(__fmul_rn(sx[e], scale), nw[w_off + e]);
    }
}

extern "C" __global__ void norm_resid_g(
    float* __restrict__ x,
    const float* __restrict__ nw,
    const float* __restrict__ ab,
    float* __restrict__ xn,
    float* __restrict__ xn32,
    unsigned short* __restrict__ xn16,
    int t_len, int w_off, int hidden)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * hidden;
    int lim = base + hidden;
    int nper = (hidden + 1023) >> 10;
    if (nper > 8) return;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        if (e < lim) {
            v[j] = x[e] + ab[e];
            ss += v[j] * v[j];
        } else {
            v[j] = 0.0f;
        }
    }
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / (float)hidden + 1e-6f);
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        if (e >= lim) continue;
        float y = v[j] * inv * nw[w_off + (j << 10) + tid];
        xn[e] = y;
        if (xn32 != (float*)0) {
            const __half h = __float2half_rn(y);
            xn32[e] = __half2float(h);
            if (xn16 != (unsigned short*)0) {
                xn16[e] = __half_as_ushort(h);
            }
        } else if (xn16 != (unsigned short*)0) {
            xn16[e] = __half_as_ushort(__float2half_rn(y));
        }
        x[e] = v[j];
    }
}
