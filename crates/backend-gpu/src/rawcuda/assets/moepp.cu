
#include <cuda_fp16.h>

#include "cast_common.cuh"
#include "g4_common.cuh"

#define GRP_M 64
#define GRP_N 32
#define GRP_KC 64
#define DGRP_M 64
#define DGRP_N 64
#define DGRP_KC 64
#define BMMA_N 64
#define PP_MZ 4

__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                       unsigned& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ void ldm_x2(unsigned& r0, unsigned& r1, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}

__device__ __forceinline__ unsigned short deq_hi_v(unsigned qw, int j, float sc) {
    const int nib = (int)((qw >> (4 * j)) & 0xFu) - 8;
    return __half_as_ushort(__float2half_rn((float)nib * sc));
}

extern "C" __global__ void w4a16_cast_f16(const float* __restrict__ in,
                                          unsigned short* __restrict__ out, int n) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i4 + 1) * 4 <= n) {
        const float4 v = *reinterpret_cast<const float4*>(in + i4 * 4);
        unsigned* o = reinterpret_cast<unsigned*>(out + i4 * 4);
        o[0] = (unsigned)__half_as_ushort(__float2half_rn(v.x))
             | ((unsigned)__half_as_ushort(__float2half_rn(v.y)) << 16);
        o[1] = (unsigned)__half_as_ushort(__float2half_rn(v.z))
             | ((unsigned)__half_as_ushort(__float2half_rn(v.w)) << 16);
    } else {
        for (int j = i4 * 4; j < n; ++j) {
            out[j] = __half_as_ushort(__float2half_rn(in[j]));
        }
    }
}

__device__ __forceinline__ unsigned short f2bf16_v(float v) {
    const unsigned u = __float_as_uint(v);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void mma_bf16(float* c, unsigned a0, unsigned a1,
                                         unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MM>
__device__ __forceinline__ void bmma_body2(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int t, int xbf)
{
    __shared__ __align__(16) unsigned short xs[2][MM][MMA_KC + 8];
    __shared__ __align__(16) unsigned short ws[2][BMMA_N][MMA_KC + 8];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * MM;
    const int n0 = blockIdx.y * BMMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (MM == 32) ? (warp >> 2) * 16 : 0;
    const int ntw = (MM == 32) ? (warp & 3) * 16 : warp * 8;
    constexpr int NF = (MM == 32) ? 2 : 1;
    float c[NF][4];
#pragma unroll
    for (int i = 0; i < 4 * NF; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    auto stage = [&](int st, int k0) {
        if (xbf) {
            const unsigned short* x16 = reinterpret_cast<const unsigned short*>(x);
            for (int e = tid; e < MM * MMA_KC / 8; e += 256) {
                const int r = e / (MMA_KC / 8);
                const int c8 = e % (MMA_KC / 8);
                const int gi = k0 + c8 * 8;
                if (m0 + r < t && gi + 7 < k) {
                    cp_async16(&xs[st][r][c8 * 8], &x16[(size_t)(m0 + r) * k + gi]);
                } else {
                    for (int j = 0; j < 8; ++j) {
                        xs[st][r][c8 * 8 + j] = (m0 + r < t && gi + j < k)
                            ? x16[(size_t)(m0 + r) * k + gi + j]
                            : (unsigned short)0;
                    }
                }
            }
        } else {
            for (int e = tid; e < MM * MMA_KC / 4; e += 256) {
                const int r = e / (MMA_KC / 4);
                const int c4 = e % (MMA_KC / 4);
                const int gi = k0 + c4 * 4;
                const float4 v = (m0 + r < t)
                    ? *reinterpret_cast<const float4*>(&x[(size_t)(m0 + r) * k + gi])
                    : make_float4(0.f, 0.f, 0.f, 0.f);
                if (gi + 3 < k) {
                    xs[st][r][c4 * 4] = f2bf16_v(v.x);
                    xs[st][r][c4 * 4 + 1] = f2bf16_v(v.y);
                    xs[st][r][c4 * 4 + 2] = f2bf16_v(v.z);
                    xs[st][r][c4 * 4 + 3] = f2bf16_v(v.w);
                } else {
                    xs[st][r][c4 * 4] = (gi < k) ? f2bf16_v(v.x) : (unsigned short)0;
                    xs[st][r][c4 * 4 + 1] = (gi + 1 < k) ? f2bf16_v(v.y) : (unsigned short)0;
                    xs[st][r][c4 * 4 + 2] = (gi + 2 < k) ? f2bf16_v(v.z) : (unsigned short)0;
                    xs[st][r][c4 * 4 + 3] = (gi + 3 < k) ? f2bf16_v(v.w) : (unsigned short)0;
                }
            }
        }
        for (int e = tid; e < BMMA_N * MMA_KC / 8; e += 256) {
            const int r = e / (MMA_KC / 8);
            const int c8 = e % (MMA_KC / 8);
            const int gi = k0 + c8 * 8;
            if (n0 + r < n && gi + 7 < k) {
                cp_async16(&ws[st][r][c8 * 8], &w[(size_t)(n0 + r) * k + gi]);
            } else {
                for (int j = 0; j < 8; ++j) {
                    ws[st][r][c8 * 8 + j] = (n0 + r < n && gi + j < k)
                        ? w[(size_t)(n0 + r) * k + gi + j]
                        : (unsigned short)0;
                }
            }
        }
    };
    stage(0, 0);
    asm volatile("cp.async.commit_group;");
    int cur = 0;
    for (int k0 = 0; k0 < k; k0 += MMA_KC, cur ^= 1) {
        const int kn = k0 + MMA_KC;
        if (kn < k) {
            stage(cur ^ 1, kn);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < MMA_KC / 16; ++ks) {
            const int kb = ks * 16;
            const int row = (lane & 15);
            const int colblk = (lane & 16) ? 8 : 0;
            unsigned a0, a1, a2, a3;
            ldm_x4(a0, a1, a2, a3, &xs[cur][mt + row][kb + colblk]);
#pragma unroll
            for (int nt = 0; nt < NF; ++nt) {
                const int nb = ntw + nt * 8;
                unsigned b0, b1;
                ldm_x2(b0, b1, &ws[cur][nb + (lane & 7)][kb + ((lane & 8) ? 8 : 0)]);
                mma_bf16(c[nt], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < NF; ++nt) {
        const int col = n0 + ntw + nt * 8 + 2 * tt;
        const int r0 = m0 + mt + g;
        if (r0 < t) {
            if (col < n) {
                out[(size_t)r0 * n + col] = c[nt][0];
            }
            if (col + 1 < n) {
                out[(size_t)r0 * n + col + 1] = c[nt][1];
            }
        }
        if (r0 + 8 < t) {
            if (col < n) {
                out[(size_t)(r0 + 8) * n + col] = c[nt][2];
            }
            if (col + 1 < n) {
                out[(size_t)(r0 + 8) * n + col + 1] = c[nt][3];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 6) w4a16_gemm_bf16_mma2(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int t, int xbf)
{
    bmma_body2<32>(w, x, out, n, k, t, xbf);
}

extern "C" __global__ void w4a16_gemm_bf16_mma2_16(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int t, int xbf)
{
    bmma_body2<16>(w, x, out, n, k, t, xbf);
}

#define BMMA3_KC 64
extern __shared__ __align__(16) unsigned short b3sm[];
#define B3_XS(st, r, c) (b3sm + ((st) * 64 + (r)) * (BMMA3_KC + 8) + (c))
#define B3_WS(st, r, c) \
    (b3sm + 2 * 64 * (BMMA3_KC + 8) + ((st) * 128 + (r)) * (BMMA3_KC + 8) + (c))
 extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemm_bf16_mma3(
     const unsigned short* __restrict__ w, const unsigned short* __restrict__ x16,
     float* __restrict__ out, int n, int k, int t)
 {
     const int tid = threadIdx.x;
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * 128;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 2) * 32;
    const int ntw = (warp & 3) * 32;
    float c[2][4][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    auto stage = [&](int st, int k0) {
        for (int e = tid; e < 64 * BMMA3_KC / 8; e += 256) {
            const int r = e / (BMMA3_KC / 8);
            const int c8 = e % (BMMA3_KC / 8);
            const int gi = k0 + c8 * 8;
            if (m0 + r < t && gi + 7 < k) {
                cp_async16(B3_XS(st, r, c8 * 8), &x16[(size_t)(m0 + r) * k + gi]);
            } else {
                for (int j = 0; j < 8; ++j) {
                    B3_XS(st, r, c8 * 8)[j] =
                        (m0 + r < t && gi + j < k)
                            ? x16[(size_t)(m0 + r) * k + gi + j]
                            : (unsigned short)0;
                }
            }
        }
        for (int e = tid; e < 128 * BMMA3_KC / 8; e += 256) {
            const int r = e / (BMMA3_KC / 8);
            const int c8 = e % (BMMA3_KC / 8);
            const int gi = k0 + c8 * 8;
            if (n0 + r < n && gi + 7 < k) {
                cp_async16(B3_WS(st, r, c8 * 8), &w[(size_t)(n0 + r) * k + gi]);
            } else {
                for (int j = 0; j < 8; ++j) {
                    B3_WS(st, r, c8 * 8)[j] =
                        (n0 + r < n && gi + j < k)
                            ? w[(size_t)(n0 + r) * k + gi + j]
                            : (unsigned short)0;
                }
            }
        }
    };
    stage(0, 0);
    asm volatile("cp.async.commit_group;");
    int cur = 0;
    for (int k0 = 0; k0 < k; k0 += BMMA3_KC, cur ^= 1) {
        const int kn = k0 + BMMA3_KC;
        if (kn < k) {
            stage(cur ^ 1, kn);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < BMMA3_KC / 16; ++ks) {
            const int kb = ks * 16;
            const int row = (lane & 15);
            const int colblk = (lane & 16) ? 8 : 0;
            unsigned a[2][4];
#pragma unroll
            for (int mf = 0; mf < 2; ++mf) {
                ldm_x4(a[mf][0], a[mf][1], a[mf][2], a[mf][3],
                       B3_XS(cur, mt + mf * 16 + row, kb + colblk));
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int nb = ntw + nt * 8;
                unsigned b0, b1;
                ldm_x2(b0, b1, B3_WS(cur, nb + (lane & 7), kb + ((lane & 8) ? 8 : 0)));
#pragma unroll
                for (int mf = 0; mf < 2; ++mf) {
                    mma_bf16(c[mf][nt], a[mf][0], a[mf][1], a[mf][2], a[mf][3], b0, b1);
                }
            }
        }
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
#undef BMMA3_KC
#undef B3_XS
#undef B3_WS
}

template <bool SC_F16>
__device__ void grp2_f16_impl(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, float* __restrict__ out2, int n, int k)
{
    const int e = blockIdx.x / PP_MZ;
    const int mz = blockIdx.x % PP_MZ;
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
    __shared__ __align__(16) unsigned short xs[2][GRP_M][GRP_KC + 8];
    __shared__ unsigned short ws[2][2 * GRP_N][GRP_KC + 8];
    const unsigned mtiles = (cnt + GRP_M - 1) / GRP_M;
    for (unsigned mtile = (unsigned)mz; mtile < mtiles; mtile += PP_MZ) {
        const unsigned rbase = mtile * GRP_M;
        float c[4][4];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
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
                        unsigned t4[4] = {0u, 0u, 0u, 0u};
                        for (int j = 0; j < 4; ++j) {
                            const int g2 = gi + 2 * j;
                            const unsigned short lo =
                                (g2 < k) ? x[xoff + g2] : (unsigned short)0;
                            const unsigned short hi =
                                (g2 + 1 < k) ? x[xoff + g2 + 1] : (unsigned short)0;
                            t4[j] = (unsigned)lo | ((unsigned)hi << 16);
                        }
                        *reinterpret_cast<uint4*>(&xs[st][r][c8 * 8]) =
                            *reinterpret_cast<const uint4*>(t4);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&xs[st][r][c8 * 8]) =
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
                const float scf = SC_F16
                    ? __half2float(__ushort_as_half(scb))
                    : __uint_as_float((unsigned)scb << 16);
                const unsigned qw = live ? __ldcs(&q[(size_t)rr * k8 + (gi >> 3)]) : 0u;
                if (live) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[st][r][c8 * 8 + j] = deq_hi_v(qw, j, scf);
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
                const int lrow = (lane & 15);
                const int lcol = (lane & 16) ? 8 : 0;
                unsigned a0, a1, a2, a3;
                ldm_x4(a0, a1, a2, a3, &xs[cur][mt + lrow][kb + lcol]);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int nb = ntw + nt * 8;
                    unsigned b0, b1;
                    ldm_x2(b0, b1, &ws[cur][nb + (lane & 7)][kb + ((lane & 8) ? 8 : 0)]);
                    mma_f16(c[nt], a0, a1, a2, a3, b0, b1);
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

template <bool SC_F16>
__device__ void grp_d64_f16_impl(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{
    const int e = blockIdx.x / PP_MZ;
    const int mz = blockIdx.x % PP_MZ;
    const unsigned cnt = gcnt[e];
    if (cnt == 0) {
        return;
    }
    const int n0 = blockIdx.y * DGRP_N;
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
    const int ntw = (warp & 1) * 32;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    __shared__ __align__(16) unsigned short xs[2][DGRP_M][DGRP_KC + 8];
    __shared__ unsigned short ws[2][DGRP_N][DGRP_KC + 8];
    const unsigned mtiles = (cnt + DGRP_M - 1) / DGRP_M;
    for (unsigned mtile = (unsigned)mz; mtile < mtiles; mtile += PP_MZ) {
        const unsigned rbase = mtile * DGRP_M;
        float c[4][4];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        auto stage_ab = [&](int st, int kn) {
            for (int ee = tid; ee < DGRP_M * DGRP_KC / 8; ee += 256) {
                const int r = ee / (DGRP_KC / 8);
                const int c8 = ee % (DGRP_KC / 8);
                const int gi = kn + c8 * 8;
                if (rbase + r < cnt) {
                    const unsigned sl = gs[rbase + r];
                    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
                    if (gi + 7 < k) {
                        cp_async16(&xs[st][r][c8 * 8], &x[xoff + gi]);
                    } else {
                        unsigned t4[4] = {0u, 0u, 0u, 0u};
                        for (int j = 0; j < 4; ++j) {
                            const int g2 = gi + 2 * j;
                            const unsigned short lo =
                                (g2 < k) ? x[xoff + g2] : (unsigned short)0;
                            const unsigned short hi =
                                (g2 + 1 < k) ? x[xoff + g2 + 1] : (unsigned short)0;
                            t4[j] = (unsigned)lo | ((unsigned)hi << 16);
                        }
                        *reinterpret_cast<uint4*>(&xs[st][r][c8 * 8]) =
                            *reinterpret_cast<const uint4*>(t4);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&xs[st][r][c8 * 8]) =
                        make_uint4(0u, 0u, 0u, 0u);
                }
            }
            for (int ee = tid; ee < DGRP_N * DGRP_KC / 8; ee += 256) {
                const int r = ee / (DGRP_KC / 8);
                const int c8 = ee % (DGRP_KC / 8);
                const int gi = kn + c8 * 8;
                const int rr = n0 + r;
                const bool live = (rr < n) && (gi + 7 < k);
                unsigned short scb = 0;
                if (rr < n) {
                    scb = s[(size_t)rr * kg + (gi >> 5)];
                }
                const float scf = SC_F16
                    ? __half2float(__ushort_as_half(scb))
                    : __uint_as_float((unsigned)scb << 16);
                const unsigned qw = live ? __ldcs(&q[(size_t)rr * k8 + (gi >> 3)]) : 0u;
                if (live) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[st][r][c8 * 8 + j] = deq_hi_v(qw, j, scf);
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
        for (int k0 = 0; k0 < k; k0 += DGRP_KC, cur ^= 1) {
            const int kn = k0 + DGRP_KC;
            if (kn < k) {
                stage_ab(cur ^ 1, kn);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
            __syncthreads();
#pragma unroll
            for (int ks = 0; ks < DGRP_KC / 16; ++ks) {
                const int kb = ks * 16;
                const int lrow = (lane & 15);
                const int lcol = (lane & 16) ? 8 : 0;
                unsigned a0, a1, a2, a3;
                ldm_x4(a0, a1, a2, a3, &xs[cur][mt + lrow][kb + lcol]);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int nb = ntw + nt * 8;
                    unsigned b0, b1;
                    ldm_x2(b0, b1, &ws[cur][nb + (lane & 7)][kb + ((lane & 8) ? 8 : 0)]);
                    mma_f16(c[nt], a0, a1, a2, a3, b0, b1);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
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

extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemm_g32_mma_grp2_f16(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, float* __restrict__ out2, int n, int k)
{ grp2_f16_impl<false>(tab, base, gslot, gcnt, goff, x, xstride, sp, out, out2, n, k); }
extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemm_g32_mma_grp2_sf16(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, float* __restrict__ out2, int n, int k)
{ grp2_f16_impl<true>(tab, base, gslot, gcnt, goff, x, xstride, sp, out, out2, n, k); }
extern "C" __global__ void w4a16_gemm_g32_mma_grp_d64_f16(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{ grp_d64_f16_impl<false>(tab, base, gslot, gcnt, goff, x, xstride, sp, out, n, k); }
extern "C" __global__ void w4a16_gemm_g32_mma_grp_d64_sf16(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned short* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{ grp_d64_f16_impl<true>(tab, base, gslot, gcnt, goff, x, xstride, sp, out, n, k); }
