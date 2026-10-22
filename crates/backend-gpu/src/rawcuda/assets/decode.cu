#include <cuda_fp16.h>
#include "g4_common.cuh"

#define SK_KC 32
#define SK_STAGES 4


__device__ __forceinline__ unsigned sk_and_or(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ void sk_deq(unsigned q, unsigned* lo, unsigned* hi) {
    const unsigned sub8 = 0x64086408u;
    const unsigned mul16 = 0x2c002c00u;
    const unsigned add72 = 0xd480d480u;
    const unsigned a = sk_and_or(q, 0x000f000fu, 0x64006400u);
    const unsigned b = sk_and_or(q, 0x00f000f0u, 0x64006400u);
    const __half2 la = __hsub2(*reinterpret_cast<const __half2*>(&a),
                               *reinterpret_cast<const __half2*>(&sub8));
    const __half2 hb = __hfma2(*reinterpret_cast<const __half2*>(&b),
                               *reinterpret_cast<const __half2*>(&mul16),
                               *reinterpret_cast<const __half2*>(&add72));
    *lo = *reinterpret_cast<const unsigned*>(&la);
    *hi = *reinterpret_cast<const unsigned*>(&hb);
}

__device__ __forceinline__ unsigned sk_scale(unsigned v, unsigned s2) {
    const __half2 a = *reinterpret_cast<const __half2*>(&v);
    const __half2 b = *reinterpret_cast<const __half2*>(&s2);
    const __half2 r = __hmul2(a, b);
    return *reinterpret_cast<const unsigned*>(&r);
}

__device__ __forceinline__ void sk_mma(float* c, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void sk_marlin_body(
    const unsigned* __restrict__ qm, const unsigned short* __restrict__ s,
    const unsigned short* __restrict__ x, float* __restrict__ out,
    float* __restrict__ part, int n, int k, int n0, int gblk, int ks,
    int ksp, int gi_lo, int gp)
{
    constexpr int A_STAGE = 16 * (SK_KC + 8);
    constexpr int B_STAGE = (SK_KC / 16) * 4 * 128;
    __shared__ __align__(16) unsigned short xs[SK_STAGES * A_STAGE];
    __shared__ __align__(16) unsigned wb[SK_STAGES * B_STAGE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wn = tid >> 5;
    const int nt_n = n >> 6;
    const int kg = k >> 7;
    const int n16 = lane >> 2;
    const int k2 = (lane & 3) * 2;

    auto stage = [&](int st, int k0) {
        if (tid < 16 * (SK_KC / 8)) {
            const int r = tid >> 2;
            const int c8 = (tid & 3) * 8;
            unsigned short* const d = xs + st * A_STAGE + r * (SK_KC + 8) + c8;
            if (r == 0) {
                cp_async16(d, x + k0 + c8);
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (int e = tid; e < (SK_KC / 16) * 128; e += 128) {
            const int tl = e >> 5;
            const int j = e & 31;
            const int ntg = (n0 >> 6) + (tl & 3);
            unsigned* const d = wb + st * B_STAGE + tl * 128 + j * 4;
            if (ntg < nt_n) {
                cp_async16(
                    d, &qm[((size_t)((k0 >> 4) + (tl >> 2)) * nt_n + ntg) * 128 + j * 4]);
            } else {
                *reinterpret_cast<uint4*>(d) =
                    make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
            }
        }
    };

    float c[8][4];
    unsigned s2lo[4], s2hi[4];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        (&c[0][0])[i] = 0.0f;
    }

    stage(0, gi_lo << 7);
    asm volatile("cp.async.commit_group;");

    const int nch = gp * (128 / SK_KC);
    int cid = 0;
    for (int gi = gi_lo; gi < gi_lo + gp; ++gi) {
        {
            const int nb = n0 + wn * 64;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                int cl = nb + 16 * w + n16;
                cl = cl < n ? cl : n - 1;
                const unsigned sl = s[(size_t)cl * kg + gi];
                const unsigned sh = s[(size_t)(cl + 8 < n ? cl + 8 : n - 1) * kg + gi];
                s2lo[w] = sl | (sl << 16);
                s2hi[w] = sh | (sh << 16);
            }
        }
#pragma unroll
        for (int ci = 0; ci < 128 / SK_KC; ++ci, ++cid) {
            const int k0 = (gi << 7) + ci * SK_KC;
            const int stc = cid & (SK_STAGES - 1);
            if (cid + 1 < nch) {
                stage((cid + 1) & (SK_STAGES - 1), k0 + SK_KC);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();

            const unsigned short* const ab =
                xs + stc * A_STAGE + n16 * (SK_KC + 8) + k2;
            const unsigned* const bb = wb + stc * B_STAGE + wn * 128;
#pragma unroll
            for (int kl = 0; kl < SK_KC / 16; ++kl) {
                const int kb = kl * 16;
                const unsigned short* const ar = ab + kb;
                const unsigned a0 = *reinterpret_cast<const unsigned*>(ar);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(ar + 8 * (SK_KC + 8));
                const unsigned a2 = *reinterpret_cast<const unsigned*>(ar + 8);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(ar + 8 * (SK_KC + 8) + 8);
                const uint4 bv = reinterpret_cast<const uint4*>(bb + kl * 512)[lane];
                const unsigned bq[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned lo0, hi0, lo1, hi1;
                    sk_deq(bq[w], &lo0, &hi0);
                    sk_deq(bq[w] >> 8, &lo1, &hi1);
                    sk_mma(c[2 * w], a0, a1, a2, a3, sk_scale(lo0, s2lo[w]),
                           sk_scale(hi0, s2lo[w]));
                    sk_mma(c[2 * w + 1], a0, a1, a2, a3, sk_scale(lo1, s2hi[w]),
                           sk_scale(hi1, s2hi[w]));
                }
            }
        }
    }

#pragma unroll
    for (int f = 0; f < 8; ++f) {
        if (n16 != 0) {
            continue;
        }
        const int col = n0 + wn * 64 + f * 8 + 2 * (lane & 3);
        if (ksp == 1) {
            if (col < n) {
                out[col] = c[f][0];
            }
            if (col + 1 < n) {
                out[col + 1] = c[f][1];
            }
        } else {
            float* const p =
                part + ((size_t)gblk * ksp + ks) * 256 + (col - n0);
            p[0] = c[f][0];
            p[1] = c[f][1];
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) w4a16_gemv_sk_marlin(
    const unsigned long long* __restrict__ tab,
    const unsigned short* __restrict__ x,
    float* __restrict__ part,
    int count, int ksp)
{
    const int blk = blockIdx.x;
    const unsigned long long* seg = tab + (count - 1) * 5;
    int base = 0;
    for (int i = 0; i < count; ++i) {
        const int nb = (int)((tab[i * 5 + 3] + 255) >> 8);
        if (blk < base + nb) {
            seg = tab + i * 5;
            break;
        }
        base += nb;
    }
    const unsigned* qm = (const unsigned*)(size_t)seg[0];
    const unsigned short* sc = (const unsigned short*)(size_t)seg[1];
    float* out = (float*)(size_t)seg[2];
    const int n = (int)seg[3];
    const int k = (int)seg[4];
    const int gp = (k >> 7) / ksp;
    sk_marlin_body(qm, sc, x, out, part, n, k, (blk - base) * 256, blk,
                   blockIdx.y, ksp, blockIdx.y * gp, gp);
}

extern "C" __global__ void w4a16_gemv_sk_reduce(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ part, int count, int ksp)
{
    const int gblk = blockIdx.x;
    const unsigned long long* seg = 0;
    int base = 0;
    for (int i = 0; i < count; ++i) {
        const int nb = (int)((tab[i * 5 + 3] + 255) >> 8);
        if (gblk < base + nb) {
            seg = tab + i * 5;
            break;
        }
        base += nb;
    }
    if (seg == 0) {
        return;
    }
    const int n = (int)seg[3];
    float* out = (float*)(size_t)seg[2];
    const int col = (gblk - base) * 256 + threadIdx.x;
    if (col >= n) {
        return;
    }
    const float* p = part + (size_t)gblk * ksp * 256 + threadIdx.x;
    float v = p[0];
#pragma unroll 4
    for (int ks = 1; ks < ksp; ++ks) {
        v += p[ks * 256];
    }
    out[col] = v;
}

extern "C" __global__ void __launch_bounds__(128, 4) w4a16_gemv_sk1_marlin(
    const unsigned* __restrict__ qm,
    const unsigned short* __restrict__ s,
    const unsigned short* __restrict__ x,
    float* __restrict__ out,
    float* __restrict__ part,
    int n, int k, int ksp)
{
    const int gp = (k >> 7) / ksp;
    sk_marlin_body(qm, s, x, out, part, n, k, blockIdx.x * 256, blockIdx.x,
                   blockIdx.y, ksp, blockIdx.y * gp, gp);
}

extern "C" __global__ void w4a16_gemv_sk1_reduce(
    const float* __restrict__ part, float* __restrict__ out, int n, int ksp)
{
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (col >= n) {
        return;
    }
    const float* p = part + (size_t)blockIdx.x * ksp * 256 + threadIdx.x;
    float v = p[0];
#pragma unroll 4
    for (int ks = 1; ks < ksp; ++ks) {
        v += p[ks * 256];
    }
    out[col] = v;
}
