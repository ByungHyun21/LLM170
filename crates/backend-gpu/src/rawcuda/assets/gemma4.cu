#include <cuda_fp16.h>
#include "g4_common.cuh"

#define G4_SCMAX 256

template <bool BF16>
__device__ __forceinline__ float ld_scale(const unsigned short* __restrict__ p, int i) {
    return BF16 ? b2f(p[i]) : h2f(p[i]);
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_g32_bf16_tall2(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out,
    int n, int k, int t16)
{
    const int o0 = blockIdx.x * 16;
    const int tb0 = blockIdx.y * 128;
    const int tb1 = min(tb0 + 128, t16);
    const int l = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int oa = o0 + 2 * w;
    const int ob = oa + 1;
    const bool la = oa < n;
    const bool lb = ob < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    __shared__ double red[16][32];
    __shared__ float sc[16][G4_SCMAX];
    for (int idx = threadIdx.x; idx < 16 * kg; idx += 256) {
        const int row = idx / kg;
        const int gg = idx - row * kg;
        if (o0 + row < n) {
            sc[row][gg] = ld_scale<true>(s + (size_t)(o0 + row) * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    const int k4 = k >> 2;
    const unsigned short* blank =
        reinterpret_cast<const unsigned short*>(q) + l;
    const unsigned short* qa = la
        ? reinterpret_cast<const unsigned short*>(q + (size_t)oa * k8) + l
        : blank;
    const unsigned short* qb = lb
        ? reinterpret_cast<const unsigned short*>(q + (size_t)ob * k8) + l
        : blank;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    constexpr int T = 16;
    for (int t0 = tb0; t0 < tb1; t0 += T) {
        float acca[T], accb[T];
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acca[r] = 0.0f;
            accb[r] = 0.0f;
        }
#pragma unroll 32
        for (int jj = 0; jj < jn; ++jj) {
            const unsigned qwa = (unsigned)qa[jj * 32];
            const unsigned qwb = (unsigned)qb[jj * 32];
            const float sca = sc[2 * w][(l >> 3) + (jj << 2)];
            const float scb = sc[2 * w + 1][(l >> 3) + (jj << 2)];
            const float a0 = fmaf((float)(qwa & 0xFu), sca, -8.0f * sca);
            const float a1 = fmaf((float)((qwa >> 4) & 0xFu), sca, -8.0f * sca);
            const float a2 = fmaf((float)((qwa >> 8) & 0xFu), sca, -8.0f * sca);
            const float a3 = fmaf((float)(qwa >> 12), sca, -8.0f * sca);
            const float b0 = fmaf((float)(qwb & 0xFu), scb, -8.0f * scb);
            const float b1 = fmaf((float)((qwb >> 4) & 0xFu), scb, -8.0f * scb);
            const float b2 = fmaf((float)((qwb >> 8) & 0xFu), scb, -8.0f * scb);
            const float b3 = fmaf((float)(qwb >> 12), scb, -8.0f * scb);
#pragma unroll
            for (int r = 0; r < T; ++r) {
                const float4 xv = xb[(size_t)(t0 + r) * k4 + jj * 32];
                acca[r] = fmaf(a0, xv.x, acca[r]);
                acca[r] = fmaf(a1, xv.y, acca[r]);
                acca[r] = fmaf(a2, xv.z, acca[r]);
                acca[r] = fmaf(a3, xv.w, acca[r]);
                accb[r] = fmaf(b0, xv.x, accb[r]);
                accb[r] = fmaf(b1, xv.y, accb[r]);
                accb[r] = fmaf(b2, xv.z, accb[r]);
                accb[r] = fmaf(b3, xv.w, accb[r]);
            }
        }
        for (int rr = 0; rr < T; ++rr) {
            red[2 * w][l] = (double)acca[rr];
            red[2 * w + 1][l] = (double)accb[rr];
            __syncthreads();
            double va = (l < 16) ? (red[2 * w][l] + red[2 * w][l + 16]) : 0.0;
            double vb = (l < 16) ? (red[2 * w + 1][l] + red[2 * w + 1][l + 16])
                                 : 0.0;
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                const double da = shfl_down_f64(va, off);
                const double db = shfl_down_f64(vb, off);
                if (l < off) {
                    va += da;
                    vb += db;
                }
            }
            if (l == 0) {
                if (la) {
                    out[(size_t)(t0 + rr) * n + oa] = (float)va;
                }
                if (lb) {
                    out[(size_t)(t0 + rr) * n + ob] = (float)vb;
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_g32_bf16_tall4(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out,
    int n, int k, int t16)
{
    const int o0 = blockIdx.x * 32;
    const int tb0 = blockIdx.y * 128;
    const int tb1 = min(tb0 + 128, t16);
    const int l = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int oa = o0 + 4 * w;
    const int ob = oa + 1;
    const int oc = oa + 2;
    const int od = oa + 3;
    const bool la = oa < n;
    const bool lb = ob < n;
    const bool lc = oc < n;
    const bool ld = od < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    __shared__ double red[32][32];
    __shared__ float sc[32][G4_SCMAX];
    for (int idx = threadIdx.x; idx < 32 * kg; idx += 256) {
        const int row = idx / kg;
        const int gg = idx - row * kg;
        if (o0 + row < n) {
            sc[row][gg] = ld_scale<true>(s + (size_t)(o0 + row) * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    const int k4 = k >> 2;
    const unsigned short* blank =
        reinterpret_cast<const unsigned short*>(q) + l;
    const unsigned short* qa = la
        ? reinterpret_cast<const unsigned short*>(q + (size_t)oa * k8) + l
        : blank;
    const unsigned short* qb = lb
        ? reinterpret_cast<const unsigned short*>(q + (size_t)ob * k8) + l
        : blank;
    const unsigned short* qc = lc
        ? reinterpret_cast<const unsigned short*>(q + (size_t)oc * k8) + l
        : blank;
    const unsigned short* qd = ld
        ? reinterpret_cast<const unsigned short*>(q + (size_t)od * k8) + l
        : blank;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    constexpr int T = 8;
    for (int t0 = tb0; t0 < tb1; t0 += T) {
        float acca[T], accb[T], accc[T], accd[T];
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acca[r] = 0.0f;
            accb[r] = 0.0f;
            accc[r] = 0.0f;
            accd[r] = 0.0f;
        }
#pragma unroll 32
        for (int jj = 0; jj < jn; ++jj) {
            const unsigned qwa = (unsigned)qa[jj * 32];
            const unsigned qwb = (unsigned)qb[jj * 32];
            const unsigned qwc = (unsigned)qc[jj * 32];
            const unsigned qwd = (unsigned)qd[jj * 32];
            const int sj = (l >> 3) + (jj << 2);
            const float sca = sc[4 * w][sj];
            const float scb = sc[4 * w + 1][sj];
            const float scc = sc[4 * w + 2][sj];
            const float scd = sc[4 * w + 3][sj];
            const float a0 = fmaf((float)(qwa & 0xFu), sca, -8.0f * sca);
            const float a1 = fmaf((float)((qwa >> 4) & 0xFu), sca, -8.0f * sca);
            const float a2 = fmaf((float)((qwa >> 8) & 0xFu), sca, -8.0f * sca);
            const float a3 = fmaf((float)(qwa >> 12), sca, -8.0f * sca);
            const float b0 = fmaf((float)(qwb & 0xFu), scb, -8.0f * scb);
            const float b1 = fmaf((float)((qwb >> 4) & 0xFu), scb, -8.0f * scb);
            const float b2 = fmaf((float)((qwb >> 8) & 0xFu), scb, -8.0f * scb);
            const float b3 = fmaf((float)(qwb >> 12), scb, -8.0f * scb);
            const float c0 = fmaf((float)(qwc & 0xFu), scc, -8.0f * scc);
            const float c1 = fmaf((float)((qwc >> 4) & 0xFu), scc, -8.0f * scc);
            const float c2 = fmaf((float)((qwc >> 8) & 0xFu), scc, -8.0f * scc);
            const float c3 = fmaf((float)(qwc >> 12), scc, -8.0f * scc);
            const float d0 = fmaf((float)(qwd & 0xFu), scd, -8.0f * scd);
            const float d1 = fmaf((float)((qwd >> 4) & 0xFu), scd, -8.0f * scd);
            const float d2 = fmaf((float)((qwd >> 8) & 0xFu), scd, -8.0f * scd);
            const float d3 = fmaf((float)(qwd >> 12), scd, -8.0f * scd);
#pragma unroll
            for (int r = 0; r < T; ++r) {
                const float4 xv = xb[(size_t)(t0 + r) * k4 + jj * 32];
                acca[r] = fmaf(a0, xv.x, acca[r]);
                acca[r] = fmaf(a1, xv.y, acca[r]);
                acca[r] = fmaf(a2, xv.z, acca[r]);
                acca[r] = fmaf(a3, xv.w, acca[r]);
                accb[r] = fmaf(b0, xv.x, accb[r]);
                accb[r] = fmaf(b1, xv.y, accb[r]);
                accb[r] = fmaf(b2, xv.z, accb[r]);
                accb[r] = fmaf(b3, xv.w, accb[r]);
                accc[r] = fmaf(c0, xv.x, accc[r]);
                accc[r] = fmaf(c1, xv.y, accc[r]);
                accc[r] = fmaf(c2, xv.z, accc[r]);
                accc[r] = fmaf(c3, xv.w, accc[r]);
                accd[r] = fmaf(d0, xv.x, accd[r]);
                accd[r] = fmaf(d1, xv.y, accd[r]);
                accd[r] = fmaf(d2, xv.z, accd[r]);
                accd[r] = fmaf(d3, xv.w, accd[r]);
            }
        }
        for (int rr = 0; rr < T; ++rr) {
            red[4 * w][l] = (double)acca[rr];
            red[4 * w + 1][l] = (double)accb[rr];
            red[4 * w + 2][l] = (double)accc[rr];
            red[4 * w + 3][l] = (double)accd[rr];
            __syncthreads();
            double va = (l < 16) ? (red[4 * w][l] + red[4 * w][l + 16]) : 0.0;
            double vb = (l < 16) ? (red[4 * w + 1][l] + red[4 * w + 1][l + 16]) : 0.0;
            double vc = (l < 16) ? (red[4 * w + 2][l] + red[4 * w + 2][l + 16]) : 0.0;
            double vd = (l < 16) ? (red[4 * w + 3][l] + red[4 * w + 3][l + 16]) : 0.0;
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                const double da = shfl_down_f64(va, off);
                const double db = shfl_down_f64(vb, off);
                const double dc = shfl_down_f64(vc, off);
                const double dd = shfl_down_f64(vd, off);
                if (l < off) {
                    va += da;
                    vb += db;
                    vc += dc;
                    vd += dd;
                }
            }
            if (l == 0) {
                if (la) {
                    out[(size_t)(t0 + rr) * n + oa] = (float)va;
                }
                if (lb) {
                    out[(size_t)(t0 + rr) * n + ob] = (float)vb;
                }
                if (lc) {
                    out[(size_t)(t0 + rr) * n + oc] = (float)vc;
                }
                if (ld) {
                    out[(size_t)(t0 + rr) * n + od] = (float)vd;
                }
            }
            __syncthreads();
        }
    }
}
