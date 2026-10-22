#include "g4_common.cuh"

extern "C" __global__ void w4a16_argmax_min(
    const float* __restrict__ lg, int n, unsigned* __restrict__ out)
{
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    const int tid = threadIdx.x;
    float v = -INFINITY;
    unsigned ix = 0u;
    const int n4 = n >> 2;
    const float4* l4 = reinterpret_cast<const float4*>(lg);
    for (int i = tid; i < n4; i += 1024) {
        const float4 q = l4[i];
        const float w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (w[j] > v) {
                v = w[j];
                ix = (unsigned)(4 * i + j);
            }
        }
    }
    for (int i = 4 * n4 + tid; i < n; i += 1024) {
        if (lg[i] > v) {
            v = lg[i];
            ix = (unsigned)i;
        }
    }
    bv[tid] = v;
    bi[tid] = ix;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) {
            const float ov = bv[tid + st];
            const unsigned oi = bi[tid + st];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                bv[tid] = ov;
                bi[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = bi[0];
    }
}

extern "C" __global__ void w4a16_argmax_min_t(
    const float* __restrict__ lg, int n, int t, unsigned* __restrict__ out)
{
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    const int row = blockIdx.x;
    if (row >= t) {
        return;
    }
    const int tid = threadIdx.x;
    const float* lr = lg + (size_t)row * n;
    float v = -INFINITY;
    unsigned ix = 0u;
    const int n4 = n >> 2;
    const float4* l4 = reinterpret_cast<const float4*>(lr);
    for (int i = tid; i < n4; i += 1024) {
        const float4 q = l4[i];
        const float w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (w[j] > v) {
                v = w[j];
                ix = (unsigned)(4 * i + j);
            }
        }
    }
    for (int i = 4 * n4 + tid; i < n; i += 1024) {
        if (lr[i] > v) {
            v = lr[i];
            ix = (unsigned)i;
        }
    }
    bv[tid] = v;
    bi[tid] = ix;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) {
            const float ov = bv[tid + st];
            const unsigned oi = bi[tid + st];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                bv[tid] = ov;
                bi[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = bi[0];
    }
}

template <int T>
__device__ __forceinline__ void gemv_i8_trow_body(
    const signed char* __restrict__ w, const float* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o,
    double* __restrict__ red)
{
    const int l = threadIdx.x;
    const unsigned* q32 = reinterpret_cast<const unsigned*>(w + (size_t)o * k);
    const float s = sc[o];
    const int k4 = k >> 2;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    const float4* xb = reinterpret_cast<const float4*>(x);
#pragma unroll 16
    for (int i = l; i < k4; i += 64) {
        const unsigned qw = q32[i];
        const float q0 = (float)(signed char)(qw & 0xFFu);
        const float q1 = (float)(signed char)((qw >> 8) & 0xFFu);
        const float q2 = (float)(signed char)((qw >> 16) & 0xFFu);
        const float q3 = (float)(signed char)(qw >> 24);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float4 xv = xb[(size_t)r * k4 + i];
            acc[r] += q0 * xv.x;
            acc[r] += q1 * xv.y;
            acc[r] += q2 * xv.z;
            acc[r] += q3 * xv.w;
        }
    }
#pragma unroll
    for (int r = 0; r < T; ++r) {
        red[l] = (double)acc[r];
        __syncthreads();
        if (l < 32) {
            double v = red[l] + red[l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(v, off);
                if (l < off) {
                    v += oth;
                }
            }
            if (l == 0) {
                out[(size_t)r * n + o] = (float)v * s;
            }
        }
        __syncthreads();
    }
}

#define GEMV_I8_TROW_CASE(N) \
    case N: gemv_i8_trow_body<N>(w, sc, x, out, n, k, o, red); break;
extern "C" __global__ void w4a16_gemv_i8_t(
    const signed char* __restrict__ w, const float* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t)
{
    const int o = blockIdx.x;
    if (o >= n || t <= 0 || t > 16) {
        return;
    }
    __shared__ double red[64];
    switch (t) {
        GEMV_I8_TROW_CASE(1)
        GEMV_I8_TROW_CASE(2)
        GEMV_I8_TROW_CASE(3)
        GEMV_I8_TROW_CASE(4)
        GEMV_I8_TROW_CASE(5)
        GEMV_I8_TROW_CASE(6)
        GEMV_I8_TROW_CASE(7)
        GEMV_I8_TROW_CASE(8)
        GEMV_I8_TROW_CASE(9)
        GEMV_I8_TROW_CASE(10)
        GEMV_I8_TROW_CASE(11)
        GEMV_I8_TROW_CASE(12)
        GEMV_I8_TROW_CASE(13)
        GEMV_I8_TROW_CASE(14)
        GEMV_I8_TROW_CASE(15)
        GEMV_I8_TROW_CASE(16)
        default: break;
    }
}

extern "C" __global__ void w4a16_gemv_i8(
    const signed char* __restrict__ w, const float* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ out, int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    const unsigned* q32 = reinterpret_cast<const unsigned*>(w + (size_t)o * k);
    const float s = sc[o];
    const int k4 = k >> 2;
    const float4* xp = reinterpret_cast<const float4*>(x) + l;
    float acc = 0.0f;
#pragma unroll 16
    for (int i = l; i < k4; i += 64) {
        const unsigned qw = q32[i];
        const float4 xv = *xp;
        acc += (float)(signed char)(qw & 0xFFu) * xv.x;
        acc += (float)(signed char)((qw >> 8) & 0xFFu) * xv.y;
        acc += (float)(signed char)((qw >> 16) & 0xFFu) * xv.z;
        acc += (float)(signed char)(qw >> 24) * xv.w;
        xp += 64;
    }
    __shared__ double red[64];
    red[l] = (double)acc;
    __syncthreads();
    if (l < 32) {
        double v = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) {
                v += oth;
            }
        }
        if (l == 0) {
            out[o] = (float)v * s;
        }
    }
}
