#include <cuda_fp16.h>


#include "cast_common.cuh"
#include "g4_common.cuh"


extern "C" __global__ void w4a16_cast_x32(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          unsigned short* __restrict__ out16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned short h = f2h(in[i]);
        if (out != (float*)0) {
            out[i] = h2f(h);
        }
        if (out16 != (unsigned short*)0) {
            out16[i] = h;
        }
    }
}


#define G4_ROWS 8

#define G4_SCMAX 256

template <bool BF16>
__device__ __forceinline__ float ld_scale(const unsigned short* __restrict__ p, int i) {
    return BF16 ? b2f(p[i]) : h2f(p[i]);
}


template <int SHIFT>
__device__ __forceinline__ int sidx2(int l, int j) {
    if constexpr (SHIFT == 7) {
        return j;
    } else {
        return (l >> 4) + (j << 2);
    }
}

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_row_body(
    const unsigned* __restrict__ q,
    const unsigned short* __restrict__ s,
    const float* __restrict__ x,
    float* __restrict__ out,
    int o,
    int k)
{
    if (o < 0) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    __shared__ float sc[G4_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> SHIFT;
    for (int g = l; g < kg; g += G4_LANES) {
        sc[g] = ld_scale<BF16>(s + (size_t)o * kg, g);
    }
    __syncthreads();
    const unsigned* qrow = q + (size_t)o * k8;
    const int jn = k >> 7;
    const int sh = 8 * (l & 3);
    float acc = 0.0f;
    int j = 0;
    const unsigned* qp = qrow + (l >> 2);
    const float2* xp = reinterpret_cast<const float2*>(x) + l;
    for (; j + 8 <= jn; j += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int jj = j + u;
            const unsigned qw = __ldcs(qp);
            const float2 xv = *xp;
            const unsigned byte = (qw >> sh) & 0xFFu;
            const float scv = sc[sidx2<SHIFT>(l, jj)];
            const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
            const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
            acc = fmaf(w0, xv.x, acc);
            acc = fmaf(w1, xv.y, acc);
            qp += 16;
            xp += 64;
        }
    }
    for (; j < jn; ++j) {
        const unsigned qw = __ldcs(qp);
        const float2 xv = *xp;
        const unsigned byte = (qw >> sh) & 0xFFu;
        const float scv = sc[sidx2<SHIFT>(l, j)];
        const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
        acc = fmaf(w0, xv.x, acc);
        acc = fmaf(w1, xv.y, acc);
        qp += 16;
        xp += 64;
    }
    red[l] = (double)acc;
    __syncthreads();
    if (l < 32) {
        double r = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(r, off);
            if (l < off) {
                r += oth;
            }
        }
        if (l == 0) {
            out[o] = (float)r;
        }
    }
}

template <int SHIFT, bool BF16, bool GELU>
__device__ __forceinline__ void gemv_row_pair_body(
    const unsigned* __restrict__ qg, const unsigned short* __restrict__ sg,
    const unsigned* __restrict__ qu, const unsigned short* __restrict__ su,
    const float* __restrict__ x, float* __restrict__ out, int o, int k)
{
    if (o < 0) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    __shared__ float scg[G4_SCMAX];
    __shared__ float scu[G4_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> SHIFT;
    for (int g = l; g < kg; g += G4_LANES) {
        scg[g] = ld_scale<BF16>(sg + (size_t)o * kg, g);
        scu[g] = ld_scale<BF16>(su + (size_t)o * kg, g);
    }
    __syncthreads();
    const unsigned* qgp = qg + (size_t)o * k8 + (l >> 2);
    const unsigned* qup = qu + (size_t)o * k8 + (l >> 2);
    const int jn = k >> 7;
    const int sh = 8 * (l & 3);
    float accg = 0.0f;
    float accu = 0.0f;
    int j = 0;
    const float2* xp = reinterpret_cast<const float2*>(x) + l;
    for (; j + 8 <= jn; j += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int jj = j + u;
            const unsigned qwg = __ldcs(qgp);
            const unsigned qwu = __ldcs(qup);
            const float2 xv = *xp;
            const unsigned bg = (qwg >> sh) & 0xFFu;
            const unsigned bu = (qwu >> sh) & 0xFFu;
            const float scgv = scg[sidx2<SHIFT>(l, jj)];
            const float scuv = scu[sidx2<SHIFT>(l, jj)];
            const float wg0 = fmaf((float)(bg & 0xFu), scgv, -8.0f * scgv);
            const float wg1 = fmaf((float)(bg >> 4), scgv, -8.0f * scgv);
            const float wu0 = fmaf((float)(bu & 0xFu), scuv, -8.0f * scuv);
            const float wu1 = fmaf((float)(bu >> 4), scuv, -8.0f * scuv);
            accg = fmaf(wg0, xv.x, accg);
            accg = fmaf(wg1, xv.y, accg);
            accu = fmaf(wu0, xv.x, accu);
            accu = fmaf(wu1, xv.y, accu);
            qgp += 16;
            qup += 16;
            xp += 64;
        }
    }
    for (; j < jn; ++j) {
        const unsigned qwg = __ldcs(qgp);
        const unsigned qwu = __ldcs(qup);
        const float2 xv = *xp;
        const unsigned bg = (qwg >> sh) & 0xFFu;
        const unsigned bu = (qwu >> sh) & 0xFFu;
        const float scgv = scg[sidx2<SHIFT>(l, j)];
        const float scuv = scu[sidx2<SHIFT>(l, j)];
        const float wg0 = fmaf((float)(bg & 0xFu), scgv, -8.0f * scgv);
        const float wg1 = fmaf((float)(bg >> 4), scgv, -8.0f * scgv);
        const float wu0 = fmaf((float)(bu & 0xFu), scuv, -8.0f * scuv);
        const float wu1 = fmaf((float)(bu >> 4), scuv, -8.0f * scuv);
        accg = fmaf(wg0, xv.x, accg);
        accg = fmaf(wg1, xv.y, accg);
        accu = fmaf(wu0, xv.x, accu);
        accu = fmaf(wu1, xv.y, accu);
        qgp += 16;
        qup += 16;
        xp += 64;
    }
    red[l] = (double)accg;
    __syncthreads();
    float g = 0.0f;
    if (l < 32) {
        double r = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(r, off);
            if (l < off) {
                r += oth;
            }
        }
        if (l == 0) {
            g = (float)r;
        }
    }
    __syncthreads();
    red[l] = (double)accu;
    __syncthreads();
    if (l < 32) {
        double r = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(r, off);
            if (l < off) {
                r += oth;
            }
        }
        if (l == 0) {
            const float u = (float)r;
            float a;
            if constexpr (GELU) {
                const float x3 = g * g * g;
                a = (0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * x3)))) * u;
            } else {
                a = (g / (1.0f + __expf(-g))) * u;
            }
            out[o] = a;
        }
    }
}

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    gemv_row_body<SHIFT, BF16>(q, s, x, out, o, k);
}

template <int SHIFT>
__device__ __forceinline__ int sidx256(int l, int jj) {
    return (jj << (8 - SHIFT)) + (l >> (SHIFT - 2));
}

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_row_body4(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int o, int k)
{
    if (o < 0) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    __shared__ float sc[G4_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> SHIFT;
    for (int g = l; g < kg; g += G4_LANES) {
        sc[g] = ld_scale<BF16>(s + (size_t)o * kg, g);
    }
    __syncthreads();
    const int jn = k >> 8;
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    const float4* xp = reinterpret_cast<const float4*>(x) + l;
    float acc = 0.0f;
#pragma unroll 16
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = (unsigned)qh[jj * 64];
        const float scv = sc[sidx256<SHIFT>(l, jj)];
        const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
        const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
        const float w3 = fmaf((float)(qw >> 12), scv, -8.0f * scv);
        const float4 xv = *xp;
        acc = fmaf(w0, xv.x, acc);
        acc = fmaf(w1, xv.y, acc);
        acc = fmaf(w2, xv.z, acc);
        acc = fmaf(w3, xv.w, acc);
        xp += 64;
    }
    red[l] = (double)acc;
    __syncthreads();
    if (l < 32) {
        double r = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(r, off);
            if (l < off) {
                r += oth;
            }
        }
        if (l == 0) {
            out[o] = (float)r;
        }
    }
}

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_body4(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    if ((k & 255) != 0) {
        gemv_row_body<SHIFT, BF16>(q, s, x, out, o, k);
        return;
    }
    gemv_row_body4<SHIFT, BF16>(q, s, x, out, o, k);
}

extern "C" __global__ void w4a16_gemv_experts_g32_bf16(
    const unsigned long long* __restrict__ tab, int base,
    const int* __restrict__ idx, int nslots,
    const float* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{
    const int sl = blockIdx.x / n;
    if (sl >= nslots) {
        return;
    }
    const int o = blockIdx.x - sl * n;
    const unsigned long long* e = tab + (size_t)(base + idx[sl] * 3) * 2;
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    gemv_row_body<5, true>((const unsigned*)e[0], (const unsigned short*)e[1],
                           x + xoff, out + (size_t)sl * n, o, k);
}

extern "C" __global__ void w4a16_gemv_experts_glu(
    const unsigned long long* __restrict__ tab, int base,
    const int* __restrict__ idx, int nslots,
    const float* __restrict__ x, int xstride, int sp,
    float* __restrict__ act, int n, int k)
{
    const int sl = blockIdx.x / n;
    if (sl >= nslots) {
        return;
    }
    const int o = blockIdx.x - sl * n;
    const unsigned long long* eg = tab + (size_t)(base + idx[sl] * 3) * 2;
    const unsigned long long* eu = tab + (size_t)(base + 1 + idx[sl] * 3) * 2;
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    float* slot = act + (size_t)sl * n;
    gemv_row_pair_body<5, true, false>(
        (const unsigned*)eg[0], (const unsigned short*)eg[1],
        (const unsigned*)eu[0], (const unsigned short*)eu[1], x + xoff, slot, o,
        k);
}

extern "C" __global__ void w4a16_gemv_experts_glu_gelu(
    const unsigned long long* __restrict__ tab, int base,
    const int* __restrict__ idx, int nslots,
    const float* __restrict__ x, int xstride, int sp,
    float* __restrict__ act, int n, int k)
{
    const int sl = blockIdx.x / n;
    if (sl >= nslots) {
        return;
    }
    const int o = blockIdx.x - sl * n;
    const unsigned long long* eg = tab + (size_t)(base + idx[sl] * 3) * 2;
    const unsigned long long* eu = tab + (size_t)(base + 1 + idx[sl] * 3) * 2;
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    float* slot = act + (size_t)sl * n;
    gemv_row_pair_body<5, true, true>(
        (const unsigned*)eg[0], (const unsigned short*)eg[1],
        (const unsigned*)eu[0], (const unsigned short*)eu[1], x + xoff, slot, o,
        k);
}

extern "C" __global__ void w4a16_moe_align(
    const int* __restrict__ idx, int nslots,
    unsigned* __restrict__ gslot, unsigned* __restrict__ gcnt,
    unsigned* __restrict__ goff, int n_exp)
{
    __shared__ unsigned cnt[1024];
    __shared__ unsigned off[1024];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    for (int i = tid; i < n_exp; i += nt) {
        cnt[i] = 0;
    }
    __syncthreads();
    for (int j = tid; j < nslots; j += nt) {
        atomicAdd(&cnt[idx[j]], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned acc = 0;
        for (int i = 0; i < n_exp; ++i) {
            off[i] = acc;
            acc += cnt[i];
            gcnt[i] = cnt[i];
            goff[i] = off[i];
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < n_exp; ++i) {
            cnt[i] = 0;
        }
        for (int j = 0; j < nslots; ++j) {
            const int e = idx[j];
            gslot[off[e] + cnt[e]] = (unsigned)j;
            cnt[e] += 1;
        }
    }
}

extern "C" __global__ void w4a16_gemv_g128(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body4<7, false>(q, s, x, out, n, k);
}
extern "C" __global__ void w4a16_gemv_g32_bf16(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body4<5, true>(q, s, x, out, n, k);
}

extern "C" __global__ void w4a16_gemv_g128_multi(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ x,
    int count)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int s = 0; s < count; ++s) {
        const int ns = (int)tab[s * 5 + 3];
        if (blk < base + ns) {
            seg = s;
            break;
        }
        base += ns;
    }
    const int n = (int)tab[seg * 5 + 3];
    const int o = blk - base;
    if (o < 0 || o >= n) {
        return;
    }
    const int k = (int)tab[seg * 5 + 4];
    gemv_row_body<7, false>((const unsigned*)(size_t)tab[seg * 5 + 0],
                            (const unsigned short*)(size_t)tab[seg * 5 + 1],
                            x, (float*)(size_t)tab[seg * 5 + 2], o, k);
}



#define GEMV_T_CASE(N) case N: gemv_g128_t_body<N>(q, s, x, out, n, k, o0, red, sc); break;


template <int T>
__device__ __forceinline__ void gemv_g128_t_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o0,
    double (*red)[32], float (*sc)[G4_SCMAX])
{
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5;
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 7;
    if (live) {
        for (int gg = l; gg < kg; gg += 32) {
            sc[g][gg] = ld_scale<false>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    const int k4 = k >> 2;
#pragma unroll 32
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = (unsigned)qh[jj * 32];
        const float scv = sc[g][jj];
        const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
        const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
        const float w3 = fmaf((float)(qw >> 12), scv, -8.0f * scv);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float4 xv = xb[(size_t)r * k4 + jj * 32];
            acc[r] = fmaf(w0, xv.x, acc[r]);
            acc[r] = fmaf(w1, xv.y, acc[r]);
            acc[r] = fmaf(w2, xv.z, acc[r]);
            acc[r] = fmaf(w3, xv.w, acc[r]);
        }
    }
#pragma unroll
    for (int rr = 0; rr < T; ++rr) {
        red[g][l] = (double)acc[rr];
        __syncthreads();
        double v = (l < 16) ? (red[g][l] + red[g][l + 16]) : 0.0;
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) {
                v += oth;
            }
        }
        if (l == 0 && live) {
            out[(size_t)rr * n + o] = (float)v;
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g128_t(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, int k, int t)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE(1)
        GEMV_T_CASE(2)
        GEMV_T_CASE(3)
        GEMV_T_CASE(4)
        GEMV_T_CASE(5)
        GEMV_T_CASE(6)
        GEMV_T_CASE(7)
        GEMV_T_CASE(8)
        GEMV_T_CASE(9)
        GEMV_T_CASE(10)
        GEMV_T_CASE(11)
        GEMV_T_CASE(12)
        GEMV_T_CASE(13)
        GEMV_T_CASE(14)
        GEMV_T_CASE(15)
        GEMV_T_CASE(16)
        default: break;
    }
}

template <int T>
__device__ __forceinline__ void gemv_g32_t_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o0,
    double (*red)[32], float (*sc)[G4_SCMAX])
{
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5;
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    if (live) {
        for (int gg = l; gg < kg; gg += 32) {
            sc[g][gg] = ld_scale<true>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    const int k4 = k >> 2;
#pragma unroll 32
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = (unsigned)qh[jj * 32];
        const float scv = sc[g][(l >> 3) + (jj << 2)];
        const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
        const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
        const float w3 = fmaf((float)(qw >> 12), scv, -8.0f * scv);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float4 xv = xb[(size_t)r * k4 + jj * 32];
            acc[r] = fmaf(w0, xv.x, acc[r]);
            acc[r] = fmaf(w1, xv.y, acc[r]);
            acc[r] = fmaf(w2, xv.z, acc[r]);
            acc[r] = fmaf(w3, xv.w, acc[r]);
        }
    }
#pragma unroll
    for (int rr = 0; rr < T; ++rr) {
        red[g][l] = (double)acc[rr];
        __syncthreads();
        double v = (l < 16) ? (red[g][l] + red[g][l + 16]) : 0.0;
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) {
                v += oth;
            }
        }
        if (l == 0 && live) {
            out[(size_t)rr * n + o] = (float)v;
        }
        __syncthreads();
    }
}

#define GEMV_T_CASE_G32(N) \
    case N: gemv_g32_t_body<N>(q, s, x, out, n, k, o0, red, sc); break;

extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g32_bf16_t(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE_G32(1)
        GEMV_T_CASE_G32(2)
        GEMV_T_CASE_G32(3)
        GEMV_T_CASE_G32(4)
        GEMV_T_CASE_G32(5)
        GEMV_T_CASE_G32(6)
        GEMV_T_CASE_G32(7)
        GEMV_T_CASE_G32(8)
        GEMV_T_CASE_G32(9)
        GEMV_T_CASE_G32(10)
        GEMV_T_CASE_G32(11)
        GEMV_T_CASE_G32(12)
        GEMV_T_CASE_G32(13)
        GEMV_T_CASE_G32(14)
        GEMV_T_CASE_G32(15)
        GEMV_T_CASE_G32(16)
        default: break;
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_g32_bf16_tall(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out,
    int n, int k, int t16)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    constexpr int T = 16;
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5;
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    if (live) {
        for (int gg = l; gg < kg; gg += 32) {
            sc[g][gg] = ld_scale<true>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    const int k4 = k >> 2;
    for (int t0 = 0; t0 < t16; t0 += T) {
        float acc[T];
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acc[r] = 0.0f;
        }
#pragma unroll 32
        for (int jj = 0; jj < jn; ++jj) {
            const unsigned qw = (unsigned)qh[jj * 32];
            const float scv = sc[g][(l >> 3) + (jj << 2)];
            const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
            const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
            const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
            const float w3 = fmaf((float)((qw >> 12) & 0xFu), scv, -8.0f * scv);
#pragma unroll
            for (int r = 0; r < T; ++r) {
                const float4 xv = xb[(size_t)(t0 + r) * k4 + jj * 32];
                acc[r] = fmaf(w0, xv.x, acc[r]);
                acc[r] = fmaf(w1, xv.y, acc[r]);
                acc[r] = fmaf(w2, xv.z, acc[r]);
                acc[r] = fmaf(w3, xv.w, acc[r]);
            }
        }
#pragma unroll
        for (int rr = 0; rr < T; ++rr) {
            red[g][l] = (double)acc[rr];
            __syncthreads();
            double v = (l < 16) ? (red[g][l] + red[g][l + 16]) : 0.0;
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(v, off);
                if (l < off) {
                    v += oth;
                }
            }
            if (l == 0 && live) {
                out[(size_t)(t0 + rr) * n + o] = (float)v;
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g128_multi_t(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ x,
    int count, int t)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int ss = 0; ss < count; ++ss) {
        const int ns = ((int)tab[ss * 5 + 3] + GEMV_TR - 1) / GEMV_TR;
        if (blk < base + ns) {
            seg = ss;
            break;
        }
        base += ns;
    }
    const unsigned* q = (const unsigned*)(size_t)tab[seg * 5 + 0];
    const unsigned short* s = (const unsigned short*)(size_t)tab[seg * 5 + 1];
    float* out = (float*)(size_t)tab[seg * 5 + 2];
    const int n = (int)tab[seg * 5 + 3];
    const int k = (int)tab[seg * 5 + 4];
    const int o0 = (blk - base) * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE(1)
        GEMV_T_CASE(2)
        GEMV_T_CASE(3)
        GEMV_T_CASE(4)
        GEMV_T_CASE(5)
        GEMV_T_CASE(6)
        GEMV_T_CASE(7)
        GEMV_T_CASE(8)
        GEMV_T_CASE(9)
        GEMV_T_CASE(10)
        GEMV_T_CASE(11)
        GEMV_T_CASE(12)
        GEMV_T_CASE(13)
        GEMV_T_CASE(14)
        GEMV_T_CASE(15)
        GEMV_T_CASE(16)
        default: break;
    }
}


extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g32_bf16_multi_t(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ x,
    int count, int t)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int ss = 0; ss < count; ++ss) {
        const int ns = ((int)tab[ss * 5 + 3] + GEMV_TR - 1) / GEMV_TR;
        if (blk < base + ns) {
            seg = ss;
            break;
        }
        base += ns;
    }
    const unsigned* q = (const unsigned*)(size_t)tab[seg * 5 + 0];
    const unsigned short* s = (const unsigned short*)(size_t)tab[seg * 5 + 1];
    float* out = (float*)(size_t)tab[seg * 5 + 2];
    const int n = (int)tab[seg * 5 + 3];
    const int k = (int)tab[seg * 5 + 4];
    const int o0 = (blk - base) * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE_G32(1)
        GEMV_T_CASE_G32(2)
        GEMV_T_CASE_G32(3)
        GEMV_T_CASE_G32(4)
        GEMV_T_CASE_G32(5)
        GEMV_T_CASE_G32(6)
        GEMV_T_CASE_G32(7)
        GEMV_T_CASE_G32(8)
        GEMV_T_CASE_G32(9)
        GEMV_T_CASE_G32(10)
        GEMV_T_CASE_G32(11)
        GEMV_T_CASE_G32(12)
        GEMV_T_CASE_G32(13)
        GEMV_T_CASE_G32(14)
        GEMV_T_CASE_G32(15)
        GEMV_T_CASE_G32(16)
        default: break;
    }
}

#define G4_GTMAX 32
#define G4_KC 128

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemm_body(
    const unsigned* __restrict__ q,
    const unsigned short* __restrict__ s,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n,
    int k,
    int t)
{
    const int o0 = blockIdx.x * G4_ROWS;
    const int g = threadIdx.x >> 6;
    const int l = threadIdx.x & (G4_LANES - 1);
    const int o = o0 + g;
    const bool live = (o < n) && (t > 0) && (t <= G4_GTMAX);
    __shared__ float sc[G4_ROWS][G4_SCMAX];
    __shared__ float xs[G4_KC][36];
    __shared__ double red[G4_ROWS][G4_LANES];
    float acc[G4_GTMAX];
#pragma unroll
    for (int u = 0; u < G4_GTMAX; ++u) {
        acc[u] = 0.0f;
    }
    const int k8 = k >> 3;
    const int kg = k >> SHIFT;
    if (live) {
        for (int gg = l; gg < kg; gg += G4_LANES) {
            sc[g][gg] = ld_scale<BF16>(s + (size_t)o * kg, gg);
        }
    }
    const int li = l & 7;
    for (int base = 0; base < k; base += G4_KC) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < G4_GTMAX * G4_KC; idx += blockDim.x) {
            const int ti = idx / G4_KC;
            const int il = idx - ti * G4_KC;
            const int gi = base + il;
            xs[il][ti] = (ti < t && gi < k) ? x[(size_t)ti * k + gi] : 0.0f;
        }
        __syncthreads();
        if (live) {
            const unsigned* qrow = q + (size_t)o * k8;
            const int nch = min(G4_KC, k - base);
#pragma unroll
            for (int q4 = 0; q4 < G4_KC / G4_LANES; ++q4) {
                const int il = l + (q4 << 6);
                if (il < nch) {
                    const int i = base + il;
                    const unsigned qw = __ldcs(&qrow[i >> 3]);
                    const int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
                    const float w = (float)v * sc[g][i >> SHIFT];
                    const float4 xa = *reinterpret_cast<const float4*>(&xs[il][0]);
                    const float4 xb = *reinterpret_cast<const float4*>(&xs[il][4]);
                    const float4 xc = *reinterpret_cast<const float4*>(&xs[il][8]);
                    const float4 xd = *reinterpret_cast<const float4*>(&xs[il][12]);
                    const float4 xe = *reinterpret_cast<const float4*>(&xs[il][16]);
                    const float4 xf = *reinterpret_cast<const float4*>(&xs[il][20]);
                    const float4 xg2 = *reinterpret_cast<const float4*>(&xs[il][24]);
                    const float4 xh = *reinterpret_cast<const float4*>(&xs[il][28]);
                    if (t > 0) acc[0] += w * xa.x;
                    if (t > 1) acc[1] += w * xa.y;
                    if (t > 2) acc[2] += w * xa.z;
                    if (t > 3) acc[3] += w * xa.w;
                    if (t > 4) acc[4] += w * xb.x;
                    if (t > 5) acc[5] += w * xb.y;
                    if (t > 6) acc[6] += w * xb.z;
                    if (t > 7) acc[7] += w * xb.w;
                    if (t > 8) acc[8] += w * xc.x;
                    if (t > 9) acc[9] += w * xc.y;
                    if (t > 10) acc[10] += w * xc.z;
                    if (t > 11) acc[11] += w * xc.w;
                    if (t > 12) acc[12] += w * xd.x;
                    if (t > 13) acc[13] += w * xd.y;
                    if (t > 14) acc[14] += w * xd.z;
                    if (t > 15) acc[15] += w * xd.w;
                    if (t > 16) acc[16] += w * xe.x;
                    if (t > 17) acc[17] += w * xe.y;
                    if (t > 18) acc[18] += w * xe.z;
                    if (t > 19) acc[19] += w * xe.w;
                    if (t > 20) acc[20] += w * xf.x;
                    if (t > 21) acc[21] += w * xf.y;
                    if (t > 22) acc[22] += w * xf.z;
                    if (t > 23) acc[23] += w * xf.w;
                    if (t > 24) acc[24] += w * xg2.x;
                    if (t > 25) acc[25] += w * xg2.y;
                    if (t > 26) acc[26] += w * xg2.z;
                    if (t > 27) acc[27] += w * xg2.w;
                    if (t > 28) acc[28] += w * xh.x;
                    if (t > 29) acc[29] += w * xh.y;
                    if (t > 30) acc[30] += w * xh.z;
                    if (t > 31) acc[31] += w * xh.w;
                }
            }
        }
    }
    for (int ti = 0; ti < G4_GTMAX; ++ti) {
        if (ti < t) {
            red[g][l] = (double)acc[ti];
        }
        __syncthreads();
        if (ti < t && live && l < 32) {
            double r = red[g][l] + red[g][l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(r, off);
                if (l < off) {
                    r += oth;
                }
            }
            if (l == 0) {
                out[(size_t)ti * n + o] = (float)r;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void w4a16_gemm_g128(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t) {
    gemm_body<7, false>(q, s, x, out, n, k, t);
}
extern "C" __global__ void w4a16_gemm_g32_bf16(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t) {
    gemm_body<5, true>(q, s, x, out, n, k, t);
}



extern "C" __global__ void w4a16_copy(const float4* __restrict__ src,
                                      float4* __restrict__ dst, long n4) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n4) {
        dst[i] = src[i];
    }
}

#define MMA_M 64
#define MMA_N 128
#define MMA_KC_G 64

#define GRP_M 64
#define GRP_N 32



__device__ __forceinline__ unsigned deq_pair_b(unsigned qw, int tt, float sc) {
    const unsigned sub8 = 0x64086408u;
    const unsigned byte = (qw >> (8 * tt)) & 0xFFu;
    unsigned w = ((byte & 0xFu) | ((byte & 0xF0u) << 12)) & 0x000f000fu;
    w |= 0x64006400u;
    __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&w),
                        *reinterpret_cast<const __half2*>(&sub8));
    const float2 f = __half22float2(h);
    const __half2 r = __float22half2_rn(make_float2(f.x * sc, f.y * sc));
    return *reinterpret_cast<const unsigned*>(&r);
}

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemm_g128_mma(
    const unsigned* __restrict__ q,
    const unsigned short* __restrict__ s,
    const unsigned short* __restrict__ x,
    float* __restrict__ out, int n, int k, int t)
{
    __shared__ __align__(16) unsigned short xs[2][MMA_M][MMA_KC_G + 8];
    __shared__ unsigned wsp[2][MMA_N][MMA_KC_G / 8];
    __shared__ float ssm[2][MMA_N];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * MMA_M;
    const int n0 = blockIdx.y * MMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 2) * 32;
    const int ntw = (warp & 3) * 32;
    const int k8 = k >> 3;
    const int kg = k >> 7;
    float c[2][4][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    int cur = 0;
    int k0 = 0;
    for (int e = tid; e < MMA_M * MMA_KC_G / 8; e += 256) {
        const int r = e / (MMA_KC_G / 8);
        const int c8 = e % (MMA_KC_G / 8);
        const int gi = c8 * 8;
        if (m0 + r < t && gi + 7 < k) {
            cp_async16(&xs[0][r][c8 * 8], &x[(size_t)(m0 + r) * k + gi]);
        } else if (m0 + r < t) {
            unsigned short t8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int j = 0; j < 8; ++j) {
                if (gi + j < k) {
                    t8[j] = x[(size_t)(m0 + r) * k + gi + j];
                }
            }
            *reinterpret_cast<uint4*>(&xs[0][r][c8 * 8]) =
                *reinterpret_cast<const uint4*>(t8);
        } else {
            *reinterpret_cast<uint4*>(&xs[0][r][c8 * 8]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int e = tid; e < MMA_N * (MMA_KC_G / 8); e += 256) {
        const int r = e / (MMA_KC_G / 8);
        const int c8 = e % (MMA_KC_G / 8);
        const int gi = c8 * 8;
        const bool live = (n0 + r < n) && (gi + 7 < k);
        wsp[0][r][c8] = live ? q[(size_t)(n0 + r) * k8 + (gi >> 3)] : 0x88888888u;
    }
    for (int e = tid; e < MMA_N; e += 256) {
        unsigned short scb = 0;
        if (n0 + e < n) {
            scb = s[(size_t)(n0 + e) * kg];
        }
        ssm[0][e] = __half2float(*reinterpret_cast<const __half*>(&scb));
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (; k0 < k; k0 += MMA_KC_G, cur ^= 1) {
        const int kn = k0 + MMA_KC_G;
        if (kn < k) {
            for (int e = tid; e < MMA_M * MMA_KC_G / 8; e += 256) {
                const int r = e / (MMA_KC_G / 8);
                const int c8 = e % (MMA_KC_G / 8);
                const int gi = kn + c8 * 8;
                if (m0 + r < t && gi + 7 < k) {
                    cp_async16(&xs[cur ^ 1][r][c8 * 8], &x[(size_t)(m0 + r) * k + gi]);
                } else if (m0 + r < t) {
                    unsigned short t8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int j = 0; j < 8; ++j) {
                        if (gi + j < k) {
                            t8[j] = x[(size_t)(m0 + r) * k + gi + j];
                        }
                    }
                    *reinterpret_cast<uint4*>(&xs[cur ^ 1][r][c8 * 8]) =
                        *reinterpret_cast<const uint4*>(t8);
                } else {
                    *reinterpret_cast<uint4*>(&xs[cur ^ 1][r][c8 * 8]) =
                        make_uint4(0u, 0u, 0u, 0u);
                }
            }
            if (tid < MMA_N) {
                const int r = tid;
                if (n0 + r < n) {
                    cp_async16(&wsp[cur ^ 1][r][0], &q[(size_t)(n0 + r) * k8 + (kn >> 3)]);
#pragma unroll
                    for (int c2 = 4; c2 < MMA_KC_G / 8; c2 += 4) {
                        cp_async16(&wsp[cur ^ 1][r][c2],
                                   &q[(size_t)(n0 + r) * k8 + (kn >> 3) + c2]);
                    }
                } else {
#pragma unroll
                    for (int c8 = 0; c8 < MMA_KC_G / 8; ++c8) {
                        wsp[cur ^ 1][r][c8] = 0x88888888u;
                    }
                }
            }
            for (int e = tid; e < MMA_N; e += 256) {
                unsigned short scb = 0;
                if (n0 + e < n) {
                    scb = s[(size_t)(n0 + e) * kg + (kn >> 7)];
                }
                ssm[cur ^ 1][e] = __half2float(*reinterpret_cast<const __half*>(&scb));
            }
            asm volatile("cp.async.commit_group;");
        }
#pragma unroll
        for (int ks = 0; ks < MMA_KC_G / 16; ++ks) {
            const int kb = ks * 16;
#pragma unroll
            for (int mf = 0; mf < 2; ++mf) {
                const int mr = mt + mf * 16;
                const unsigned a0 = pk2bf(&xs[cur][mr + g][kb + 2 * tt]);
                const unsigned a1 = pk2bf(&xs[cur][mr + g + 8][kb + 2 * tt]);
                const unsigned a2 = pk2bf(&xs[cur][mr + g][kb + 2 * tt + 8]);
                const unsigned a3 = pk2bf(&xs[cur][mr + g + 8][kb + 2 * tt + 8]);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int nb = ntw + nt * 8;
                    const float bsc = ssm[cur][nb + g];
                    const unsigned b0 = deq_pair_b(wsp[cur][nb + g][kb >> 3], tt, bsc);
                    const unsigned b1 = deq_pair_b(wsp[cur][nb + g][(kb >> 3) + 1], tt, bsc);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[mf][nt][0]), "+f"(c[mf][nt][1]), "+f"(c[mf][nt][2]),
                          "+f"(c[mf][nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }
#pragma unroll
    for (int mf = 0; mf < 2; ++mf) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int col = n0 + ntw + nt * 8 + 2 * tt;
            const int r0 = m0 + mt + mf * 16 + g;
            if (r0 < t) {
                if (col < n) {
                    out[(size_t)r0 * n + col] = c[mf][nt][0];
                }
                if (col + 1 < n) {
                    out[(size_t)r0 * n + col + 1] = c[mf][nt][1];
                }
            }
            if (r0 + 8 < t) {
                if (col < n) {
                    out[(size_t)(r0 + 8) * n + col] = c[mf][nt][2];
                }
                if (col + 1 < n) {
                    out[(size_t)(r0 + 8) * n + col + 1] = c[mf][nt][3];
                }
            }
        }
    }
}

#define GRP_KC 32
extern "C" __global__ void w4a16_gemm_g32_mma_grp(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{
    const int e = blockIdx.x;
    const unsigned cnt = gcnt[e];
    if (cnt == 0) {
        return;
    }
    const int n0 = blockIdx.y * GRP_N;
    const unsigned long long* wp = tab + (size_t)(base + e * 3) * 2;
    const unsigned* q = (const unsigned*)wp[0];
    const unsigned short* s = (const unsigned short*)wp[1];
    const unsigned* gs = gslot + (size_t)goff[e];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 1) * 16;
    const int ntw = (warp & 1) * 16;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    __shared__ __align__(16) unsigned short xs[2][GRP_M][GRP_KC + 8];
    __shared__ unsigned short ws[2][GRP_N][GRP_KC + 8];
    const unsigned mtiles = (cnt + GRP_M - 1) / GRP_M;
    for (unsigned mtile = 0; mtile < mtiles; ++mtile) {
        const unsigned rbase = mtile * GRP_M;
        float c[2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        auto stage_ab = [&](int st, int kn) {
            for (int ee = tid; ee < GRP_M * GRP_KC / 8; ee += 256) {
                const int r = ee / (GRP_KC / 8);
                const int c8 = ee % (GRP_KC / 8);
                const int gi = kn + c8 * 8;
                if (rbase + r < cnt) {
                    const unsigned sl = gs[rbase + r];
                    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
                    if (gi + 7 < k) {
                        cp_async16(&xs[st][r][c8 * 8], &x[xoff + gi]);
                    } else {
                        unsigned short t8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        for (int j = 0; j < 8; ++j) {
                            if (gi + j < k) {
                                t8[j] = x[xoff + gi + j];
                            }
                        }
                        *reinterpret_cast<uint4*>(&xs[st][r][c8 * 8]) =
                            *reinterpret_cast<const uint4*>(t8);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&xs[st][r][c8 * 8]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            for (int ee = tid; ee < GRP_N * GRP_KC / 8; ee += 256) {
                const int r = ee / (GRP_KC / 8);
                const int c8 = ee % (GRP_KC / 8);
                const int gi = kn + c8 * 8;
                const bool live = (n0 + r < n) && (gi + 7 < k);
                unsigned short scb = 0;
                if (n0 + r < n) {
                    scb = s[(size_t)(n0 + r) * kg + (gi >> 5)];
                }
                const float scf = __uint_as_float((unsigned)scb << 16);
                const unsigned qw = live ? __ldcs(&q[(size_t)(n0 + r) * k8 + (gi >> 3)]) : 0u;
                if (live) {
#pragma unroll
                    for (int jp = 0; jp < 4; ++jp) {
                        *reinterpret_cast<unsigned*>(&ws[st][r][c8 * 8 + jp * 2]) =
                            deq_pair_b(qw, jp, scf);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[st][r][c8 * 8 + j] = 0;
                    }
                }
            }
        };
        stage_ab(0, 0);
        asm volatile("cp.async.commit_group;");
        int cur = 0;
        for (int k0 = 0; k0 < k; k0 += GRP_KC, cur ^= 1) {
            const int kn = k0 + GRP_KC;
            if (kn < k) {
                stage_ab(cur ^ 1, kn);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < GRP_KC / 16; ++ks) {
                const int kb = ks * 16;
                const unsigned a0 = pk2bf(&xs[cur][mt + g][kb + 2 * tt]);
                const unsigned a1 = pk2bf(&xs[cur][mt + g + 8][kb + 2 * tt]);
                const unsigned a2 = pk2bf(&xs[cur][mt + g][kb + 2 * tt + 8]);
                const unsigned a3 = pk2bf(&xs[cur][mt + g + 8][kb + 2 * tt + 8]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int nb = ntw + nt * 8;
                    const unsigned b0 = pk2bf(&ws[cur][nb + g][kb + 2 * tt]);
                    const unsigned b1 = pk2bf(&ws[cur][nb + g][kb + 2 * tt + 8]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[nt][0]), "+f"(c[nt][1]), "+f"(c[nt][2]), "+f"(c[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = n0 + ntw + nt * 8 + 2 * tt;
            const int r0 = mt + g;
            if (rbase + r0 < cnt) {
                const unsigned sl = gs[rbase + r0];
                if (col < n) {
                    out[(size_t)sl * n + col] = c[nt][0];
                }
                if (col + 1 < n) {
                    out[(size_t)sl * n + col + 1] = c[nt][1];
                }
            }
            if (rbase + r0 + 8 < cnt) {
                const unsigned sl = gs[rbase + r0 + 8];
                if (col < n) {
                    out[(size_t)sl * n + col] = c[nt][2];
                }
                if (col + 1 < n) {
                    out[(size_t)sl * n + col + 1] = c[nt][3];
                }
            }
        }
    }
}


extern "C" __global__ void w4a16_gemm_g32_mma_grp2(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, float* __restrict__ out2, int n, int k)
{
    const int e = blockIdx.x;
    const unsigned cnt = gcnt[e];
    if (cnt == 0) {
        return;
    }
    const int n0 = blockIdx.y * GRP_N;
    const unsigned long long* wp0 = tab + (size_t)(base + e * 3) * 2;
    const unsigned long long* wp1 = tab + (size_t)(base + e * 3 + 1) * 2;
    const unsigned* q0 = (const unsigned*)wp0[0];
    const unsigned short* s0 = (const unsigned short*)wp0[1];
    const unsigned* q1 = (const unsigned*)wp1[0];
    const unsigned short* s1 = (const unsigned short*)wp1[1];
    const unsigned* gs = gslot + (size_t)goff[e];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 1) * 16;
    const int ntw = (warp & 1) * 32;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    __shared__ __align__(16) unsigned xs[2][GRP_M][GRP_KC + 8];
    __shared__ unsigned short ws[2][2 * GRP_N][GRP_KC + 8];
    const unsigned mtiles = (cnt + GRP_M - 1) / GRP_M;
    for (unsigned mtile = 0; mtile < mtiles; ++mtile) {
        const unsigned rbase = mtile * GRP_M;
        float c[4][4];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        auto stage_ab = [&](int st, int kn) {
            for (int ee = tid; ee < GRP_M * GRP_KC / 4; ee += 256) {
                const int r = ee / (GRP_KC / 4);
                const int c4 = ee % (GRP_KC / 4);
                const int gi = kn + c4 * 4;
                if (rbase + r < cnt) {
                    const unsigned sl = gs[rbase + r];
                    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
                    if (gi + 3 < k) {
                        cp_async16(&xs[st][r][c4 * 4], &x[xoff + gi]);
                    } else {
                        unsigned t4[4] = {0u, 0u, 0u, 0u};
                        for (int j = 0; j < 4; ++j) {
                            if (gi + j < k) {
                                t4[j] = x[xoff + gi + j];
                            }
                        }
                        *reinterpret_cast<uint4*>(&xs[st][r][c4 * 4]) =
                            *reinterpret_cast<const uint4*>(t4);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&xs[st][r][c4 * 4]) =
                        make_uint4(0u, 0u, 0u, 0u);
                }
            }
            for (int ee = tid; ee < 2 * GRP_N * GRP_KC / 8; ee += 256) {
                const int r = ee / (GRP_KC / 8);
                const int c8 = ee % (GRP_KC / 8);
                const int gi = kn + c8 * 8;
                const int rr = n0 + (r & (GRP_N - 1));
                const unsigned* q = (r < GRP_N) ? q0 : q1;
                const unsigned short* s = (r < GRP_N) ? s0 : s1;
                const bool live = (rr < n) && (gi + 7 < k);
                unsigned short scb = 0;
                if (rr < n) {
                    scb = s[(size_t)rr * kg + (gi >> 5)];
                }
                const float scf = __uint_as_float((unsigned)scb << 16);
                const unsigned qw = live ? __ldcs(&q[(size_t)rr * k8 + (gi >> 3)]) : 0u;
                if (live) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[st][r][c8 * 8 + j] = deq_hi_e(qw, j, scf);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[st][r][c8 * 8 + j] = 0;
                    }
                }
            }
        };
        stage_ab(0, 0);
        asm volatile("cp.async.commit_group;");
        int cur = 0;
        for (int k0 = 0; k0 < k; k0 += GRP_KC, cur ^= 1) {
            const int kn = k0 + GRP_KC;
            if (kn < k) {
                stage_ab(cur ^ 1, kn);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
            __syncthreads();
#pragma unroll
            for (int ks = 0; ks < GRP_KC / 16; ++ks) {
                const int kb = ks * 16;
                const unsigned p00 = xs[cur][mt + g][kb + 2 * tt];
                const unsigned p01 = xs[cur][mt + g][kb + 2 * tt + 1];
                const unsigned p10 = xs[cur][mt + g + 8][kb + 2 * tt];
                const unsigned p11 = xs[cur][mt + g + 8][kb + 2 * tt + 1];
                const unsigned p20 = xs[cur][mt + g][kb + 2 * tt + 8];
                const unsigned p21 = xs[cur][mt + g][kb + 2 * tt + 9];
                const unsigned p30 = xs[cur][mt + g + 8][kb + 2 * tt + 8];
                const unsigned p31 = xs[cur][mt + g + 8][kb + 2 * tt + 9];
                const unsigned ah0 = hilo_hi(p00, p01);
                const unsigned al0 = hilo_lo(p00, p01);
                const unsigned ah1 = hilo_hi(p10, p11);
                const unsigned al1 = hilo_lo(p10, p11);
                const unsigned ah2 = hilo_hi(p20, p21);
                const unsigned al2 = hilo_lo(p20, p21);
                const unsigned ah3 = hilo_hi(p30, p31);
                const unsigned al3 = hilo_lo(p30, p31);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int nb = ntw + nt * 8;
                    const unsigned b0 = pk2bf(&ws[cur][nb + g][kb + 2 * tt]);
                    const unsigned b1 = pk2bf(&ws[cur][nb + g][kb + 2 * tt + 8]);
                    mma_f16(c[nt], ah0, ah1, ah2, ah3, b0, b1);
                    mma_f16(c[nt], al0, al1, al2, al3, b0, b1);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int col64 = ntw + nt * 8 + 2 * tt;
            const int col = n0 + (col64 & (GRP_N - 1));
            float* o = (col64 >= GRP_N) ? out2 : out;
            const int r0 = mt + g;
            if (rbase + r0 < cnt) {
                const unsigned sl = gs[rbase + r0];
                if (col < n) {
                    o[(size_t)sl * n + col] = c[nt][0];
                }
                if (col + 1 < n) {
                    o[(size_t)sl * n + col + 1] = c[nt][1];
                }
            }
            if (rbase + r0 + 8 < cnt) {
                const unsigned sl = gs[rbase + r0 + 8];
                if (col < n) {
                    o[(size_t)sl * n + col] = c[nt][2];
                }
                if (col + 1 < n) {
                    o[(size_t)sl * n + col + 1] = c[nt][3];
                }
            }
        }
    }
}
