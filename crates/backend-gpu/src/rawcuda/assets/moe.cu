
#include <cuda_fp16.h>

#include "cast_common.cuh"
#include "g4_common.cuh"


extern "C" __global__ void w4a16_gemv_bf16(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    const unsigned short* wrow = w + (size_t)o * k;
    float acc = 0.0f;
    for (int i = l; i < k; i += G4_LANES) {
        acc += b2f(wrow[i]) * x[i];
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

extern "C" __global__ void w4a16_gemv_multi(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ x,
    int count)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int s = 0; s < count; ++s) {
        const int ns = (int)tab[s * 4 + 2];
        if (blk < base + ns) {
            seg = s;
            break;
        }
        base += ns;
    }
    const int n = (int)tab[seg * 4 + 2];
    const int o = blk - base;
    if (o < 0 || o >= n) {
        return;
    }
    const int k = (int)tab[seg * 4 + 3];
    const unsigned short* w = (const unsigned short*)(size_t)tab[seg * 4 + 0];
    float* out = (float*)(size_t)tab[seg * 4 + 1];
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    const unsigned short* wrow = w + (size_t)o * k;
    float acc = 0.0f;
    for (int i = l; i < k; i += G4_LANES) {
        acc += b2f(wrow[i]) * x[i];
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

#define GEMV_T_CASE_BF(N) \
    case N: gemv_bf16_t_body<N>(w, x, out, n, k, o0, red); break;

template <int T>
__device__ __forceinline__ void gemv_bf16_t_body(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int o0, double (*red)[G4_LANES])
{
    const int l = threadIdx.x & (G4_LANES - 1);
    const int g = threadIdx.x >> 6;
    const int o = o0 + g;
    const bool live = o < n;
    const unsigned short* wrow = w + (size_t)o * k;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    const float* xb = x;
#pragma unroll 32
    for (int i = l; i < k; i += G4_LANES) {
        const float wv = b2f(wrow[i]);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acc[r] += wv * xb[(size_t)r * k + i];
        }
    }
#pragma unroll
    for (int rr = 0; rr < T; ++rr) {
        red[g][l] = (double)acc[rr];
        __syncthreads();
        if (l < 32 && live) {
            double v = red[g][l] + red[g][l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(v, off);
                if (l < off) {
                    v += oth;
                }
            }
            if (l == 0) {
                out[(size_t)rr * n + o] = (float)v;
            }
        }
        __syncthreads();
    }
}

#define G4_TMAX 16

extern "C" __global__ void w4a16_gemv_bf16_t(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, int k, int t)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n || t <= 0 || t > G4_TMAX) {
        return;
    }
    __shared__ double red[GEMV_TR][G4_LANES];
    switch (t) {
        GEMV_T_CASE_BF(1)
        GEMV_T_CASE_BF(2)
        GEMV_T_CASE_BF(3)
        GEMV_T_CASE_BF(4)
        GEMV_T_CASE_BF(5)
        GEMV_T_CASE_BF(6)
        GEMV_T_CASE_BF(7)
        GEMV_T_CASE_BF(8)
        GEMV_T_CASE_BF(9)
        GEMV_T_CASE_BF(10)
        GEMV_T_CASE_BF(11)
        GEMV_T_CASE_BF(12)
        GEMV_T_CASE_BF(13)
        GEMV_T_CASE_BF(14)
        GEMV_T_CASE_BF(15)
        GEMV_T_CASE_BF(16)
        default: break;
    }
}

extern "C" __global__ void w4a16_gemv_bf16_multi_t(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ x,
    int count, int t)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int ss = 0; ss < count; ++ss) {
        const int ns = ((int)tab[ss * 4 + 2] + GEMV_TR - 1) / GEMV_TR;
        if (blk < base + ns) {
            seg = ss;
            break;
        }
        base += ns;
    }
    const unsigned short* w = (const unsigned short*)(size_t)tab[seg * 4 + 0];
    float* out = (float*)(size_t)tab[seg * 4 + 1];
    const int n = (int)tab[seg * 4 + 2];
    const int k = (int)tab[seg * 4 + 3];
    const int o0 = (blk - base) * GEMV_TR;
    if (o0 >= n || t <= 0 || t > G4_TMAX) {
        return;
    }
    __shared__ double red[GEMV_TR][G4_LANES];
    switch (t) {
        GEMV_T_CASE_BF(1)
        GEMV_T_CASE_BF(2)
        GEMV_T_CASE_BF(3)
        GEMV_T_CASE_BF(4)
        GEMV_T_CASE_BF(5)
        GEMV_T_CASE_BF(6)
        GEMV_T_CASE_BF(7)
        GEMV_T_CASE_BF(8)
        GEMV_T_CASE_BF(9)
        GEMV_T_CASE_BF(10)
        GEMV_T_CASE_BF(11)
        GEMV_T_CASE_BF(12)
        GEMV_T_CASE_BF(13)
        GEMV_T_CASE_BF(14)
        GEMV_T_CASE_BF(15)
        GEMV_T_CASE_BF(16)
        default: break;
    }
}

template <int T>
__device__ __forceinline__ void gemv_bf16_t_v4_body(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int o0)
{
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5;
    const int o = o0 + g;
    const bool live = o < n;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) acc[r] = 0.0f;
    if (live) {
        const unsigned short* wrow = w + (size_t)o * k;
        const bool vec = ((k & 7) == 0) && (((size_t)wrow & 15) == 0);
        if (vec) {
            int base = l * 8;
            for (; base + 7 < k; base += 256) {
                const uint4 wv = *reinterpret_cast<const uint4*>(wrow + base);
                const float w0 = b2f((unsigned short)(wv.x & 0xffffu));
                const float w1 = b2f((unsigned short)(wv.x >> 16));
                const float w2 = b2f((unsigned short)(wv.y & 0xffffu));
                const float w3 = b2f((unsigned short)(wv.y >> 16));
                const float w4 = b2f((unsigned short)(wv.z & 0xffffu));
                const float w5 = b2f((unsigned short)(wv.z >> 16));
                const float w6 = b2f((unsigned short)(wv.w & 0xffffu));
                const float w7 = b2f((unsigned short)(wv.w >> 16));
#pragma unroll
                for (int r = 0; r < T; ++r) {
                    const float* xr = x + (size_t)r * k + base;
                    const float4 xa = *reinterpret_cast<const float4*>(xr);
                    const float4 xb = *reinterpret_cast<const float4*>(xr + 4);
                    acc[r] += w0 * xa.x;
                    acc[r] += w1 * xa.y;
                    acc[r] += w2 * xa.z;
                    acc[r] += w3 * xa.w;
                    acc[r] += w4 * xb.x;
                    acc[r] += w5 * xb.y;
                    acc[r] += w6 * xb.z;
                    acc[r] += w7 * xb.w;
                }
            }
            for (int i = base; i < k; ++i) {
                const float wv = b2f(wrow[i]);
#pragma unroll
                for (int r = 0; r < T; ++r) acc[r] += wv * x[(size_t)r * k + i];
            }
        } else {
            for (int i = l; i < k; i += 32) {
                const float wv = b2f(wrow[i]);
#pragma unroll
                for (int r = 0; r < T; ++r) acc[r] += wv * x[(size_t)r * k + i];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < T; ++r) {
        double v = (double)acc[r];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) v += oth;
        }
        if (live && l == 0) out[(size_t)r * n + o] = (float)v;
    }
}

#define GEMV_T_CASE_BF_V4(N) \
    case N: gemv_bf16_t_v4_body<N>(w, x, out, n, k, o0); break;

extern "C" __global__ void w4a16_gemv_bf16_multi_t_v4(
    const unsigned long long* __restrict__ tab,
    const float* __restrict__ x,
    int count, int t)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int ss = 0; ss < count; ++ss) {
        const int ns = ((int)tab[ss * 4 + 2] + GEMV_TR - 1) / GEMV_TR;
        if (blk < base + ns) {
            seg = ss;
            break;
        }
        base += ns;
    }
    const unsigned short* w = (const unsigned short*)(size_t)tab[seg * 4 + 0];
    float* out = (float*)(size_t)tab[seg * 4 + 1];
    const int n = (int)tab[seg * 4 + 2];
    const int k = (int)tab[seg * 4 + 3];
    const int o0 = (blk - base) * GEMV_TR;
    if (o0 >= n || t <= 0 || t > G4_TMAX) {
        return;
    }
    switch (t) {
        GEMV_T_CASE_BF_V4(1)
        GEMV_T_CASE_BF_V4(2)
        GEMV_T_CASE_BF_V4(3)
        GEMV_T_CASE_BF_V4(4)
        GEMV_T_CASE_BF_V4(5)
        GEMV_T_CASE_BF_V4(6)
        GEMV_T_CASE_BF_V4(7)
        GEMV_T_CASE_BF_V4(8)
        GEMV_T_CASE_BF_V4(9)
        GEMV_T_CASE_BF_V4(10)
        GEMV_T_CASE_BF_V4(11)
        GEMV_T_CASE_BF_V4(12)
        GEMV_T_CASE_BF_V4(13)
        GEMV_T_CASE_BF_V4(14)
        GEMV_T_CASE_BF_V4(15)
        GEMV_T_CASE_BF_V4(16)
        default: break;
    }
}


#define BMMA_N 64
#define BMMA_M 32

__device__ __forceinline__ unsigned short f2bf16(float v) {
    unsigned u = __float_as_uint(v);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

extern "C" __global__ void w4a16_cast_bf16(const float* __restrict__ in,
                                           unsigned short* __restrict__ out, int n) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i4 + 1) * 4 <= n) {
        const float4 v = *reinterpret_cast<const float4*>(in + i4 * 4);
        unsigned short* o = out + i4 * 4;
        o[0] = f2bf16(v.x);
        o[1] = f2bf16(v.y);
        o[2] = f2bf16(v.z);
        o[3] = f2bf16(v.w);
    } else {
        for (int j = i4 * 4; j < n; ++j) {
            out[j] = f2bf16(in[j]);
        }
    }
}

extern "C" __global__ void w4a16_cast_hilo(const float* __restrict__ in,
                                           unsigned* __restrict__ out, int n) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i4 + 1) * 4 <= n) {
        const float4 v = *reinterpret_cast<const float4*>(in + i4 * 4);
        unsigned* o = out + i4 * 4;
        o[0] = hilo_pack(v.x);
        o[1] = hilo_pack(v.y);
        o[2] = hilo_pack(v.z);
        o[3] = hilo_pack(v.w);
    } else {
        for (int j = i4 * 4; j < n; ++j) {
            out[j] = hilo_pack(in[j]);
        }
    }
}

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

template <int MM>
__device__ __forceinline__ void bmma_body(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, int k, int t, int xbf)
{
    __shared__ unsigned short xs[MM][MMA_KC + 8];
    __shared__ unsigned short ws[BMMA_N][MMA_KC + 8];
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
    for (int k0 = 0; k0 < k; k0 += MMA_KC) {
        __syncthreads();
        if (xbf) {
            const unsigned short* x16 = reinterpret_cast<const unsigned short*>(x);
            for (int e = tid; e < MM * MMA_KC / 8; e += 256) {
                const int r = e / (MMA_KC / 8);
                const int c8 = e % (MMA_KC / 8);
                const int gi = k0 + c8 * 8;
                unsigned short* dst = &xs[r][c8 * 8];
                if (m0 + r < t && gi + 7 < k) {
                    *reinterpret_cast<uint4*>(dst) =
                        *reinterpret_cast<const uint4*>(&x16[(size_t)(m0 + r) * k + gi]);
                } else {
                    for (int j = 0; j < 8; ++j) {
                        dst[j] = (m0 + r < t && gi + j < k)
                            ? x16[(size_t)(m0 + r) * k + gi + j]
                            : (unsigned short)0;
                    }
                }
            }
        } else
        for (int e = tid; e < MM * MMA_KC / 4; e += 256) {
            const int r = e / (MMA_KC / 4);
            const int c4 = e % (MMA_KC / 4);
            const int gi = k0 + c4 * 4;
            const float4 v = (m0 + r < t)
                ? *reinterpret_cast<const float4*>(&x[(size_t)(m0 + r) * k + gi])
                : make_float4(0.f, 0.f, 0.f, 0.f);
            if (gi + 3 < k) {
                xs[r][c4 * 4] = f2bf16(v.x);
                xs[r][c4 * 4 + 1] = f2bf16(v.y);
                xs[r][c4 * 4 + 2] = f2bf16(v.z);
                xs[r][c4 * 4 + 3] = f2bf16(v.w);
            } else {
                xs[r][c4 * 4] = (gi < k) ? f2bf16(v.x) : (unsigned short)0;
                xs[r][c4 * 4 + 1] = (gi + 1 < k) ? f2bf16(v.y) : (unsigned short)0;
                xs[r][c4 * 4 + 2] = (gi + 2 < k) ? f2bf16(v.z) : (unsigned short)0;
                xs[r][c4 * 4 + 3] = (gi + 3 < k) ? f2bf16(v.w) : (unsigned short)0;
            }
        }
        for (int e = tid; e < BMMA_N * MMA_KC / 8; e += 256) {
            const int r = e / (MMA_KC / 8);
            const int c8 = e % (MMA_KC / 8);
            const int gi = k0 + c8 * 8;
            unsigned short* dst = &ws[r][c8 * 8];
            if (n0 + r < n && gi + 7 < k) {
                const uint4 v = *reinterpret_cast<const uint4*>(&w[(size_t)(n0 + r) * k + gi]);
                *reinterpret_cast<uint4*>(dst) = v;
            } else {
                for (int j = 0; j < 8; ++j) {
                    dst[j] = (n0 + r < n && gi + j < k)
                        ? w[(size_t)(n0 + r) * k + gi + j]
                        : (unsigned short)0;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < MMA_KC / 16; ++ks) {
            const int kb = ks * 16;
            const int row = (lane & 15);
            const int colblk = (lane & 16) ? 8 : 0;
            unsigned a0, a1, a2, a3;
            ldm_x4(a0, a1, a2, a3, &xs[mt + row][kb + colblk]);
#pragma unroll
            for (int nt = 0; nt < NF; ++nt) {
                const int nb = ntw + nt * 8;
                unsigned b0, b1;
                ldm_x2(b0, b1, &ws[nb + (lane & 7)][kb + ((lane & 8) ? 8 : 0)]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c[nt][0]), "+f"(c[nt][1]), "+f"(c[nt][2]), "+f"(c[nt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
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

extern "C" __global__ void __launch_bounds__(256, 6) w4a16_gemm_bf16_mma(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int t, int xbf)
{
    bmma_body<32>(w, x, out, n, k, t, xbf);
}

extern "C" __global__ void w4a16_gemm_bf16_mma16(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int t, int xbf)
{
    bmma_body<16>(w, x, out, n, k, t, xbf);
}

extern "C" __global__ void w4a16_gemm_bf16(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, int k, int t)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    const unsigned short* wrow = w + (size_t)o * k;
    float acc[G4_TMAX];
#pragma unroll
    for (int u = 0; u < G4_TMAX; ++u) {
        acc[u] = 0.0f;
    }
    for (int i = l; i < k; i += G4_LANES) {
        const float wv = b2f(wrow[i]);
#pragma unroll
        for (int u = 0; u < G4_TMAX; ++u) {
            if (u < t) {
                acc[u] += wv * x[(size_t)u * k + i];
            }
        }
    }
    for (int u = 0; u < t; ++u) {
        red[l] = (double)acc[u];
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
                out[(size_t)u * n + o] = (float)r;
            }
        }
        __syncthreads();
    }
}

#define G4_TMAX2 32
#define G4_ROWS_B 8
#define G4_KC_B 128
extern "C" __global__ void w4a16_gemm_bf16_t(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, int k, int t)
{
    const int g = threadIdx.x >> 6;
    const int l = threadIdx.x & (G4_LANES - 1);
    const int o = blockIdx.x * G4_ROWS_B + g;
    const bool live = o < n;
    __shared__ double red[G4_ROWS_B][G4_LANES];
    __shared__ unsigned short ws[G4_ROWS_B][G4_KC_B];
    float acc[G4_TMAX2];
#pragma unroll
    for (int u = 0; u < G4_TMAX2; ++u) {
        acc[u] = 0.0f;
    }
    for (int base = 0; base < k; base += G4_KC_B) {
        const int nch = min(G4_KC_B, k - base);
        for (int idx = threadIdx.x; idx < G4_ROWS_B * G4_KC_B; idx += blockDim.x) {
            const int r = idx / G4_KC_B;
            const int i = idx - r * G4_KC_B;
            const int ro = blockIdx.x * G4_ROWS_B + r;
            ws[r][i] = (ro < n && i < nch) ? w[(size_t)ro * k + base + i] : (unsigned short)0;
        }
        __syncthreads();
        const unsigned short* wsrow = ws[g];
        const float* xp = x + base + l;
        for (int i = l; i < nch; i += G4_LANES, xp += G4_LANES) {
            const float wv = live ? b2f(wsrow[i]) : 0.0f;
#pragma unroll
            for (int u = 0; u < G4_TMAX2; ++u) {
                if (u < t) {
                    acc[u] += wv * xp[(size_t)u * k];
                }
            }
        }
        __syncthreads();
    }
    for (int u = 0; u < t; ++u) {
        red[g][l] = (double)acc[u];
        __syncthreads();
        if (live && l < 32) {
            double r = red[g][l] + red[g][l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(r, off);
                if (l < off) {
                    r += oth;
                }
            }
            if (l == 0) {
                out[(size_t)u * n + o] = (float)r;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void w4a16_axpy(float w, const float* __restrict__ x,
                                      float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += w * x[i];
    }
}

extern "C" __global__ void w4a16_moe_pes_scale(
    float* __restrict__ wt, const int* __restrict__ idx,
    const float* __restrict__ pes, int nslots)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nslots) {
        return;
    }
    wt[i] *= pes[idx[i]];
}

extern "C" __global__ void w4a16_scale(float* __restrict__ y, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] *= s;
    }
}

extern "C" __global__ void w4a16_moe_accum(const float* __restrict__ w,
                                           const float* __restrict__ d,
                                           float* __restrict__ y, int sp, int nslots,
                                           int n) {
    const int spt = (sp > 0) ? sp : nslots;
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int ti = row * spt;
        const int end = (ti + spt < nslots) ? ti + spt : nslots;
        float acc = 0.0f;
        for (int s = ti; s < end; ++s) {
            acc += w[s] * d[(size_t)s * n + i];
        }
        y[(size_t)row * n + i] = acc;
    }
}


#define MOE_TK_WARPS 8

extern "C" __global__ void w4a16_moe_topk_t(
    const float* __restrict__ lg,
    unsigned* __restrict__ idx,
    float* __restrict__ wt,
    int t, int n, int k)
{
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ti = blockIdx.x * MOE_TK_WARPS + w;
    if (ti >= t || n > 1024) {
        return;
    }
    const float* l = lg + (size_t)ti * n;
    float mx = -INFINITY;
    for (int i = lane; i < n; i += 32) {
        mx = fmaxf(mx, l[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    float pv[32];
    unsigned pidx[32];
    int cnt = 0;
    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) {
        pv[cnt] = __expf(l[i] - mx);
        pidx[cnt] = (unsigned)i;
        sum += pv[cnt];
        ++cnt;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    unsigned used = 0u;
    float wsum = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        unsigned bi = 0xffffffffu;
        for (int j = 0; j < cnt; ++j) {
            if (used & (1u << j)) {
                continue;
            }
            if (pv[j] > bv || (pv[j] == bv && pidx[j] < bi)) {
                bv = pv[j];
                bi = pidx[j];
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        for (int j = 0; j < cnt; ++j) {
            if (pidx[j] == bi) {
                used |= (1u << j);
            }
        }
        if (lane == 0) {
            idx[(size_t)ti * k + r] = bi;
            wt[(size_t)ti * k + r] = bv;
        }
        wsum += bv;
    }
    __syncwarp();
    for (int r = lane; r < k; r += 32) {
        wt[(size_t)ti * k + r] = wt[(size_t)ti * k + r] / wsum;
    }
}

#define MOE_TK256_N 256
#define MOE_TK256_C 8

extern "C" __global__ void w4a16_moe_topk256(
    const float* __restrict__ lg,
    unsigned* __restrict__ idx,
    float* __restrict__ wt,
    int k)
{
    const int lane = threadIdx.x & 31;
    float v[MOE_TK256_C];
#pragma unroll
    for (int j = 0; j < MOE_TK256_C; ++j) {
        v[j] = lg[lane + 32 * j];
    }
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < MOE_TK256_C; ++j) {
        mx = fmaxf(mx, v[j]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    float p[MOE_TK256_C];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MOE_TK256_C; ++j) {
        p[j] = __expf(v[j] - mx);
        sum += p[j];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    unsigned used = 0u;
    float wsum = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        unsigned bi = 0xffffffffu;
#pragma unroll
        for (int j = 0; j < MOE_TK256_C; ++j) {
            if (used & (1u << j)) {
                continue;
            }
            const unsigned ii = (unsigned)(lane + 32 * j);
            if (p[j] > bv || (p[j] == bv && ii < bi)) {
                bv = p[j];
                bi = ii;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
#pragma unroll
        for (int j = 0; j < MOE_TK256_C; ++j) {
            if ((unsigned)(lane + 32 * j) == bi) {
                used |= (1u << j);
            }
        }
        if (lane == 0) {
            idx[r] = bi;
            wt[r] = bv;
        }
        wsum += bv;
    }
    __syncwarp();
    for (int r = lane; r < k; r += 32) {
        wt[r] = wt[r] / wsum;
    }
}

extern "C" __global__ void w4a16_moe_accum_sg(const float* __restrict__ w,
                                              const float* __restrict__ d,
                                              const float* __restrict__ sg,
                                              const float* __restrict__ sx,
                                              float* __restrict__ y, int sp, int nslots,
                                              int n) {
    const int spt = (sp > 0) ? sp : nslots;
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int ti = row * spt;
        const int end = (ti + spt < nslots) ? ti + spt : nslots;
        float acc = 0.0f;
        for (int s = ti; s < end; ++s) {
            acc += w[s] * d[(size_t)s * n + i];
        }
        const float g = 1.0f / (1.0f + expf(-sg[row]));
        y[(size_t)row * n + i] = acc + g * sx[(size_t)row * n + i];
    }
}

extern "C" __global__ void w4a16_moe_align2(
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
    if (tid < 32) {
        const int l = tid;
        const unsigned lt = (1u << l) - 1u;
        for (int i = l; i < n_exp; i += 32) {
            cnt[i] = 0;
        }
        __syncwarp();
        for (int base = 0; base < nslots; base += 32) {
            const int j = base + l;
            const int e = (j < nslots) ? idx[j] : -1;
            const unsigned m = __match_any_sync(0xFFFFFFFFu, e);
            if (e >= 0 && j < nslots) {
                gslot[off[e] + cnt[e] + __popc(m & lt)] = (unsigned)j;
            }
            __syncwarp();
            if (e >= 0 && j < nslots && (1u << l) == (m & (0u - m))) {
                cnt[e] += __popc(m);
            }
            __syncwarp();
        }
    }
}

extern "C" __global__ void w4a16_shared_add(const float* __restrict__ sg,
                                            const float* __restrict__ x,
                                            float* __restrict__ y, int n) {
    const int ti = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < n) {
        const float s = 1.0f / (1.0f + expf(-sg[ti]));
        y[(size_t)ti * n + i] += s * x[(size_t)ti * n + i];
    }
}


__device__ __forceinline__ unsigned deq_pair_b_d(unsigned qw, int tt, float sc) {
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

#define DGRP_M 64
#define DGRP_N 64
#define DGRP_KC 32

extern "C" __global__ void w4a16_gemm_g32_mma_grp_d64(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{
    const int e = blockIdx.x;
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
    __shared__ __align__(16) unsigned xs[2][DGRP_M][DGRP_KC + 8];
    __shared__ unsigned short ws[2][DGRP_N][DGRP_KC + 8];
    const unsigned mtiles = (cnt + DGRP_M - 1) / DGRP_M;
    for (unsigned mtile = 0; mtile < mtiles; ++mtile) {
        const unsigned rbase = mtile * DGRP_M;
        float c[4][4];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        auto stage_ab = [&](int st, int kn) {
            for (int ee = tid; ee < DGRP_M * DGRP_KC / 4; ee += 256) {
                const int r = ee / (DGRP_KC / 4);
                const int c4 = ee % (DGRP_KC / 4);
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

#undef DGRP_M
#undef DGRP_N
#undef DGRP_KC


#define MAR_M 64
#define MAR_N 128
#define MAR_KC 32
#define MAR_STAGES 4

__device__ __forceinline__ unsigned mar_and_or(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ void mar_deq(unsigned q, unsigned* lo, unsigned* hi) {
    const unsigned sub8 = 0x64086408u;
    const unsigned mul16 = 0x2c002c00u;
    const unsigned add72 = 0xd480d480u;
    const unsigned a = mar_and_or(q, 0x000f000fu, 0x64006400u);
    const unsigned b = mar_and_or(q, 0x00f000f0u, 0x64006400u);
    const __half2 la = __hsub2(*reinterpret_cast<const __half2*>(&a),
                               *reinterpret_cast<const __half2*>(&sub8));
    const __half2 hb = __hfma2(*reinterpret_cast<const __half2*>(&b),
                               *reinterpret_cast<const __half2*>(&mul16),
                               *reinterpret_cast<const __half2*>(&add72));
    *lo = *reinterpret_cast<const unsigned*>(&la);
    *hi = *reinterpret_cast<const unsigned*>(&hb);
}

__device__ __forceinline__ unsigned mar_scale(unsigned v, unsigned s2) {
    const __half2 a = *reinterpret_cast<const __half2*>(&v);
    const __half2 b = *reinterpret_cast<const __half2*>(&s2);
    const __half2 r = __hmul2(a, b);
    return *reinterpret_cast<const unsigned*>(&r);
}

__device__ __forceinline__ void mar_mma(float* c, unsigned a0, unsigned a1,
                                        unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemm_g128_marlin(
    const unsigned* __restrict__ qm,
    const unsigned short* __restrict__ s,
    const unsigned short* __restrict__ x,
    float* __restrict__ out, int n, int k, int t)
{
    constexpr int A_STAGE = MAR_M * (MAR_KC + 8);
    constexpr int B_STAGE = (MAR_KC / 16) * 2 * 128;
    __shared__ __align__(16) unsigned short xs[MAR_STAGES * A_STAGE];
    __shared__ __align__(16) unsigned wb[MAR_STAGES * B_STAGE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 1;
    const int wn = warp & 1;
    const int m0 = blockIdx.x * MAR_M;
    const int n0 = blockIdx.y * MAR_N;
    const int nt_n = n >> 6;
    const int kg = k >> 7;
    const int n16 = lane >> 2;
    const int k2 = (lane & 3) * 2;

    const int a_cpr = MAR_KC / 8;
    const int b_tl = tid >> 5;
    const int b_j = tid & 31;
    const int b_off = b_tl * 128 + b_j * 4;
    const int b_ntg = (n0 >> 6) + (b_tl & 1);
    const bool b_ok = b_ntg < nt_n;

    auto stage = [&](int st, int k0) {
#pragma unroll
        for (int e = tid; e < MAR_M * a_cpr; e += 256) {
            const int r = e / a_cpr;
            const int c8 = (e % a_cpr) * 8;
            unsigned short* const d = xs + st * A_STAGE + r * (MAR_KC + 8) + c8;
            if (m0 + r < t) {
                cp_async16(d, x + (size_t)(m0 + r) * k + k0 + c8);
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        if (tid < 128) {
            unsigned* const db = wb + st * B_STAGE + b_off;
            if (b_ok) {
                cp_async16(db, &qm[((size_t)((k0 >> 4) + (b_tl >> 1)) * nt_n + b_ntg) * 128 + b_j * 4]);
            } else {
                *reinterpret_cast<uint4*>(db) =
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

    stage(0, 0);
    asm volatile("cp.async.commit_group;");

    const int nch = k / MAR_KC;
    const int cpg = 128 / MAR_KC;
    const int ngroups = k >> 7;
    int cid = 0;
    for (int gi = 0; gi < ngroups; ++gi) {
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
        for (int ci = 0; ci < cpg; ++ci, ++cid) {
            const int k0 = (gi << 7) + ci * MAR_KC;
            const int stc = cid & (MAR_STAGES - 1);
            if (cid + 1 < nch) {
                stage((cid + 1) & (MAR_STAGES - 1), k0 + MAR_KC);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();

            const unsigned short* const ab =
                xs + stc * A_STAGE + (wm * 16 + n16) * (MAR_KC + 8) + k2;
            const unsigned* const bb = wb + stc * B_STAGE + wn * 128;
#pragma unroll
            for (int kl = 0; kl < MAR_KC / 16; ++kl) {
                const int kb = kl * 16;
                const unsigned short* const ar = ab + kb;
                const unsigned a0 = *reinterpret_cast<const unsigned*>(ar);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(ar + 8 * (MAR_KC + 8));
                const unsigned a2 = *reinterpret_cast<const unsigned*>(ar + 8);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(ar + 8 * (MAR_KC + 8) + 8);
                const uint4 bv = reinterpret_cast<const uint4*>(bb + kl * 256)[lane];
                const unsigned bq[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned lo0, hi0, lo1, hi1;
                    mar_deq(bq[w], &lo0, &hi0);
                    mar_deq(bq[w] >> 8, &lo1, &hi1);
                    mar_mma(c[2 * w], a0, a1, a2, a3, mar_scale(lo0, s2lo[w]),
                            mar_scale(hi0, s2lo[w]));
                    mar_mma(c[2 * w + 1], a0, a1, a2, a3, mar_scale(lo1, s2hi[w]),
                            mar_scale(hi1, s2hi[w]));
                }
            }
        }
    }

#pragma unroll
    for (int f = 0; f < 8; ++f) {
        const int col = n0 + wn * 64 + f * 8 + 2 * (lane & 3);
        const int r0 = m0 + wm * 16 + n16;
        if (r0 < t) {
            if (col < n) {
                out[(size_t)r0 * n + col] = c[f][0];
            }
            if (col + 1 < n) {
                out[(size_t)r0 * n + col + 1] = c[f][1];
            }
        }
        if (r0 + 8 < t) {
            if (col < n) {
                out[(size_t)(r0 + 8) * n + col] = c[f][2];
            }
            if (col + 1 < n) {
                out[(size_t)(r0 + 8) * n + col + 1] = c[f][3];
            }
        }
    }
}


extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemm_g32_marlin(
    const unsigned* __restrict__ qm,
    const unsigned short* __restrict__ s,
    const unsigned short* __restrict__ x,
    float* __restrict__ out, int n, int k, int t)
{
    constexpr int A_STAGE = MAR_M * (MAR_KC + 8);
    constexpr int B_STAGE = (MAR_KC / 16) * 2 * 128;
    __shared__ __align__(16) unsigned short xs[MAR_STAGES * A_STAGE];
    __shared__ __align__(16) unsigned wb[MAR_STAGES * B_STAGE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 1;
    const int wn = warp & 1;
    const int m0 = blockIdx.x * MAR_M;
    const int n0 = blockIdx.y * MAR_N;
    const int nt_n = n >> 6;
    const int kg = k >> 5;
    const int n16 = lane >> 2;
    const int k2 = (lane & 3) * 2;

    const int a_cpr = MAR_KC / 8;
    const int b_tl = tid >> 5;
    const int b_j = tid & 31;
    const int b_off = b_tl * 128 + b_j * 4;
    const int b_ntg = (n0 >> 6) + (b_tl & 1);
    const bool b_ok = b_ntg < nt_n;

    auto stage = [&](int st, int k0) {
#pragma unroll
        for (int e = tid; e < MAR_M * a_cpr; e += 256) {
            const int r = e / a_cpr;
            const int c8 = (e % a_cpr) * 8;
            unsigned short* const d = xs + st * A_STAGE + r * (MAR_KC + 8) + c8;
            if (m0 + r < t) {
                cp_async16(d, x + (size_t)(m0 + r) * k + k0 + c8);
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        if (tid < 128) {
            unsigned* const db = wb + st * B_STAGE + b_off;
            if (b_ok) {
                cp_async16(db, &qm[((size_t)((k0 >> 4) + (b_tl >> 1)) * nt_n + b_ntg) * 128 + b_j * 4]);
            } else {
                *reinterpret_cast<uint4*>(db) =
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

    stage(0, 0);
    asm volatile("cp.async.commit_group;");

    const int nch = k / MAR_KC;
    const int cpg = 128 / MAR_KC;
    const int ngroups = k >> 7;
    int cid = 0;
    for (int gi = 0; gi < ngroups; ++gi) {
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
        for (int ci = 0; ci < cpg; ++ci, ++cid) {
            const int k0 = (gi << 7) + ci * MAR_KC;
            const int stc = cid & (MAR_STAGES - 1);
            if (cid + 1 < nch) {
                stage((cid + 1) & (MAR_STAGES - 1), k0 + MAR_KC);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();

            {
                const int g = k0 >> 5;
                const int nb = n0 + wn * 64;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    int cl = nb + 16 * w + n16;
                    cl = cl < n ? cl : n - 1;
                    const unsigned sl = (unsigned)__half_as_ushort(
                        __float2half_rn(b2f(s[(size_t)cl * kg + g])));
                    const unsigned sh = (unsigned)__half_as_ushort(__float2half_rn(
                        b2f(s[(size_t)(cl + 8 < n ? cl + 8 : n - 1) * kg + g])));
                    s2lo[w] = sl | (sl << 16);
                    s2hi[w] = sh | (sh << 16);
                }
            }

            const unsigned short* const ab =
                xs + stc * A_STAGE + (wm * 16 + n16) * (MAR_KC + 8) + k2;
            const unsigned* const bb = wb + stc * B_STAGE + wn * 128;
#pragma unroll
            for (int kl = 0; kl < MAR_KC / 16; ++kl) {
                const int kb = kl * 16;
                const unsigned short* const ar = ab + kb;
                const unsigned a0 = *reinterpret_cast<const unsigned*>(ar);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(ar + 8 * (MAR_KC + 8));
                const unsigned a2 = *reinterpret_cast<const unsigned*>(ar + 8);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(ar + 8 * (MAR_KC + 8) + 8);
                const uint4 bv = reinterpret_cast<const uint4*>(bb + kl * 256)[lane];
                const unsigned bq[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned lo0, hi0, lo1, hi1;
                    mar_deq(bq[w], &lo0, &hi0);
                    mar_deq(bq[w] >> 8, &lo1, &hi1);
                    mar_mma(c[2 * w], a0, a1, a2, a3, mar_scale(lo0, s2lo[w]),
                            mar_scale(hi0, s2lo[w]));
                    mar_mma(c[2 * w + 1], a0, a1, a2, a3, mar_scale(lo1, s2hi[w]),
                            mar_scale(hi1, s2hi[w]));
                }
            }
        }
    }

#pragma unroll
    for (int f = 0; f < 8; ++f) {
        const int col = n0 + wn * 64 + f * 8 + 2 * (lane & 3);
        const int r0 = m0 + wm * 16 + n16;
        if (r0 < t) {
            if (col < n) {
                out[(size_t)r0 * n + col] = c[f][0];
            }
            if (col + 1 < n) {
                out[(size_t)r0 * n + col + 1] = c[f][1];
            }
        }
        if (r0 + 8 < t) {
            if (col < n) {
                out[(size_t)(r0 + 8) * n + col] = c[f][2];
            }
            if (col + 1 < n) {
                out[(size_t)(r0 + 8) * n + col + 1] = c[f][3];
            }
        }
    }
}



#define MM16_M 16
#define MM16_N 256
#define MM16_KC 32
#define MM16_S 4

extern "C" __global__ void __launch_bounds__(128, 4) w4a16_gemm_g128_marlin_m16(
    const unsigned* __restrict__ qm,
    const unsigned short* __restrict__ s,
    const unsigned short* __restrict__ x,
    float* __restrict__ out, int n, int k, int t)
{
    constexpr int A_STAGE = MM16_M * (MM16_KC + 8);
    constexpr int B_STAGE = (MM16_KC / 16) * 4 * 128;
    __shared__ __align__(16) unsigned short xs[MM16_S * A_STAGE];
    __shared__ __align__(16) unsigned wb[MM16_S * B_STAGE];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wn = tid >> 5;
    const int m0 = blockIdx.x * MM16_M;
    const int n0 = blockIdx.y * MM16_N;
    const int nt_n = n >> 6;
    const int kg = k >> 7;
    const int n16 = lane >> 2;
    const int k2 = (lane & 3) * 2;

    auto stage = [&](int st, int k0) {
        if (tid < MM16_M * (MM16_KC / 8)) {
            const int r = tid >> 2;
            const int c8 = (tid & 3) * 8;
            unsigned short* const d = xs + st * A_STAGE + r * (MM16_KC + 8) + c8;
            if (m0 + r < t) {
                cp_async16(d, x + (size_t)(m0 + r) * k + k0 + c8);
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (int e = tid; e < (MM16_KC / 16) * 128; e += 128) {
            const int tl = e >> 5;
            const int j = e & 31;
            const int ntg = (n0 >> 6) + (tl & 3);
            unsigned* const d = wb + st * B_STAGE + tl * 128 + j * 4;
            if (ntg < nt_n) {
                cp_async16(
                    d,
                    &qm[((size_t)((k0 >> 4) + (tl >> 2)) * nt_n + ntg) * 128 + j * 4]);
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

    stage(0, 0);
    asm volatile("cp.async.commit_group;");

    const int nch = k / MM16_KC;
    const int cpg = 128 / MM16_KC;
    const int ngroups = k >> 7;
    int cid = 0;
    for (int gi = 0; gi < ngroups; ++gi) {
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
        for (int ci = 0; ci < cpg; ++ci, ++cid) {
            const int k0 = (gi << 7) + ci * MM16_KC;
            const int stc = cid & (MM16_S - 1);
            if (cid + 1 < nch) {
                stage((cid + 1) & (MM16_S - 1), k0 + MM16_KC);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();

            const unsigned short* const ab = xs + stc * A_STAGE + n16 * (MM16_KC + 8) + k2;
            const unsigned* const bb = wb + stc * B_STAGE + wn * 128;
#pragma unroll
            for (int kl = 0; kl < MM16_KC / 16; ++kl) {
                const int kb = kl * 16;
                const unsigned short* const ar = ab + kb;
                const unsigned a0 = *reinterpret_cast<const unsigned*>(ar);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(ar + 8 * (MM16_KC + 8));
                const unsigned a2 = *reinterpret_cast<const unsigned*>(ar + 8);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(ar + 8 * (MM16_KC + 8) + 8);
                const uint4 bv = reinterpret_cast<const uint4*>(bb + kl * 512)[lane];
                const unsigned bq[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned lo0, hi0, lo1, hi1;
                    mar_deq(bq[w], &lo0, &hi0);
                    mar_deq(bq[w] >> 8, &lo1, &hi1);
                    mar_mma(c[2 * w], a0, a1, a2, a3, mar_scale(lo0, s2lo[w]),
                            mar_scale(hi0, s2lo[w]));
                    mar_mma(c[2 * w + 1], a0, a1, a2, a3, mar_scale(lo1, s2hi[w]),
                            mar_scale(hi1, s2hi[w]));
                }
            }
        }
    }

#pragma unroll
    for (int f = 0; f < 8; ++f) {
        const int col = n0 + wn * 64 + f * 8 + 2 * (lane & 3);
        const int r0 = m0 + n16;
        if (r0 < t) {
            if (col < n) {
                out[(size_t)r0 * n + col] = c[f][0];
            }
            if (col + 1 < n) {
                out[(size_t)r0 * n + col + 1] = c[f][1];
            }
        }
        if (r0 + 8 < t) {
            if (col < n) {
                out[(size_t)(r0 + 8) * n + col] = c[f][2];
            }
            if (col + 1 < n) {
                out[(size_t)(r0 + 8) * n + col + 1] = c[f][3];
            }
        }
    }
}

extern "C" __global__ void w4a16_h2f(const unsigned short* __restrict__ in,
                                     float* __restrict__ out, int n2) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += stride) {
        const __half2 h = *reinterpret_cast<const __half2*>(&in[2 * i]);
        const float2 f = __half22float2(h);
        *reinterpret_cast<float2*>(&out[2 * i]) = f;
    }
}

extern "C" __global__ void w4a16_moe_align_marlin(
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff, int* __restrict__ sorted_ids,
    int* __restrict__ expert_ids, int* __restrict__ num_post, int n_exp,
    int block, int sentinel) {
    __shared__ int boff[1024];
    if (blockIdx.x != 0) {
        return;
    }
    const int tid = threadIdx.x;
    if (tid == 0) {
        int pos = 0;
        for (int e = 0; e < n_exp; ++e) {
            boff[e] = pos;
            const int c = (int)gcnt[e];
            pos += ((c + block - 1) / block) * block;
        }
        num_post[0] = pos;
    }
    __syncthreads();
    for (int e = tid; e < n_exp; e += blockDim.x) {
        const int c = (int)gcnt[e];
        const int base = boff[e];
        const unsigned* sl = gslot + goff[e];
        int blk_i = base / block;
        for (int s = 0; s < c; s += block) {
            expert_ids[blk_i] = e;
            const int m = min(block, c - s);
            for (int k2 = 0; k2 < block; ++k2) {
                sorted_ids[base + s + k2] = (k2 < m) ? (int)sl[s + k2] : sentinel;
            }
            blk_i += 1;
        }
    }
}

extern "C" __global__ void w4a16_moe_act_gelu_f16(
    const unsigned short* __restrict__ c13, unsigned short* __restrict__ act,
    int ns, int n_ff) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ns * n_ff; i += stride) {
        const int r = i / n_ff, c = i % n_ff;
        const float g = __half2float(*reinterpret_cast<const __half*>(&c13[r * 2 * n_ff + c]));
        const float u = __half2float(*reinterpret_cast<const __half*>(&c13[r * 2 * n_ff + n_ff + c]));
        const float x3 = g * g * g;
        const float a = (0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * x3)))) * u;
        act[i] = __half_as_ushort(__float2half_rn(a));
    }
}

extern "C" __global__ void w4a16_moe_act_silu_f16(
    const unsigned short* __restrict__ c13, unsigned short* __restrict__ act,
    int ns, int n_ff) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ns * n_ff; i += stride) {
        const int r = i / n_ff, c = i % n_ff;
        const float g = __half2float(*reinterpret_cast<const __half*>(&c13[r * 2 * n_ff + c]));
        const float u = __half2float(*reinterpret_cast<const __half*>(&c13[r * 2 * n_ff + n_ff + c]));
        const float e = __expf(-g);
        act[i] = __half_as_ushort(__float2half_rn((g / (1.0f + e)) * u));
    }
}

extern "C" __global__ void w4a16_moe_topk_sum_f16(
    const unsigned short* __restrict__ c2, float* __restrict__ out, int t, int h,
    int top_k) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < t * h; i += stride) {
        const int r = i / h, c = i % h;
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            acc += __half2float(*reinterpret_cast<const __half*>(&c2[(r * top_k + k) * h + c]));
        }
        out[i] = acc;
    }
}
