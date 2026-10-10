// ── MoE·플레인 bf16 커널 — [R5 2026-10-10] gptq4.cu에서 분할 ──
// 35B-A3B의 비양자화(플레인) 경로 전부: bf16 GEMV/GEMM(mma 포함) + 라우터
// top-k·전문가 가중 누적·게이트 보조. 분할(g128/g32) 양자화 커널과 전문가
// 배치 GEMV는 gptq4.cu 소유(공용 define·헬퍼는 g4_common.cuh).
// [빌드 계약] -fmad=false — 플레인 경로도 mul·add 분리 산식(CPU 미러 계약).

#include <cuda_fp16.h>

#include "g4_common.cuh"

// ── MoE(W4-1, 35B-A3B) 보조 커널 ──
// 플레인 bf16 GEMV — 행=블록(64레인), 레인 l = i=l,l+64,… f32 누산 → tree64.
// split 커널과 동일한 레인·환원 구조(플레인 경로 판정은 토큰 수준 — 골든).
// w는 [n][k] 행 우선(업로드 원본 그대로 — 전치 없음).

extern "C" __global__ void w4a16_gemv_bf16(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [k] f32
    float* __restrict__ out,               // [n]
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

// [B-4 2026-10-10] 다중 세그먼트 GEMV — 같은 x를 공유하는 소형 선형 여럿을
// 1런치로(소형 커널 런치 플로어 제거: 35B 실측 4런치 ~18µs → 1런치 ~5µs/층).
// tab[seg] = (w, out, n, k) u64×4(호스트가 층별로 작성). 1블록 = 1행,
// 세그먼트는 행 누적으로 선택(count≤8 선형 탐색 — 전 블록 공통 테이블).
// **행별 산술은 w4a16_gemv_bf16과 비트동일**(i=l,l+64… mul·add 분리·tree64).
extern "C" __global__ void w4a16_gemv_multi(
    const unsigned long long* __restrict__ tab, // [count][4]: w, out, n, k
    const float* __restrict__ x,                // 공통 x [k]
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

// [R14] t-커널 스위치(BF 판) — gptq4.cu GEMV_T_CASE의 플레인 대응.
#define GEMV_T_CASE_BF(N) \
    case N: gemv_bf16_t_body<N>(w, x, out, n, k, o0, red); break;

// [A9 2026-10-10] t행 GEMV(플레인 bf16) — 가중 판독 1회를 t행이 공유.
// 행별 산술은 w4a16_gemv_bf16과 **비트동일**(i 오름차순 mul·add 분리·tree64).
// t 템플릿 상수 + GEMV_TR행/블록(A9-fix1/2 — split 판 동형).
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
    const float* xp[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        xp[r] = x + (size_t)r * k + l;
    }
#pragma unroll 32
    for (int i = l; i < k; i += G4_LANES) {
        const float wv = b2f(wrow[i]);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acc[r] += wv * *xp[r];
            xp[r] += G4_LANES;
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

// t 상한 — 호스트 BATCH_DEC_MAX 미러(디코드 배치 상한; GEMV_T_CASE_BF 1..8).
// [R5 2026-10-10] 플레인 v1 GEMM(gemm_bf16)과 공유 — 종전 재정의 2곳을
// 단일 정의로 통합(값 8 불변).
#define G4_TMAX 8

extern "C" __global__ void w4a16_gemv_bf16_t(
    const unsigned short* __restrict__ w, // [n][k] bf16
    const float* __restrict__ x,          // [t][k] f32(원시 — h2f 왕복 없음)
    float* __restrict__ out,              // [t][n]
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
        default: break; // 호스트 계약 밖(t≤8)
    }
}

// ── T2(2026-10-09): 플레인 bf16 mma GEMM — 계약 완화 승인 후 첫 본체 ──
// A=x(bf16 반올림, 스테이징에서 f32→bf16 RNE), B=w bf16(스토어 원본 [n][k]),
// f32 누적. 프래그먼트 규약은 --mma-smoke에서 검증(비트동일 0.00e0):
//   A: a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t..]} a2={A[g][2t+8..]} a3={A[g+8][2t+8..]}
//   B(col-major, B[k][n]=w[n][k]): b0={w[g][2t],w[g][2t+1]} b1={w[g][2t+8],w[g][2t+9]}
//   C: c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]  (g=lane>>2, t=lane&3)
// 타일: 블록 256스레드(8워프) = M32 × N64, k청크 32(스테이징 ~6KB) → 워프당
// m16×n16(2× n8 mma). 워프 w: m타일 w/4, n타일 (w%4)*16.
// [B1/B3 2026-10-10] ncu 이중 병목(L2 86% x 재판독 ÷2·ALU 57.7 디퀀트)·KC32
// 실측: ffn_up t512 1.58→1.35ms. 그룹(MoE) = GRP_M64×GRP_N32(슬롯행 재사용).

// [marlin-A2 결함수정] w4a16_gemm_bf16_mma는 자체 N64 고정(워프 매핑 하드코딩)
// — MMA_N 공유 시 N128에서 상위 절반 미계산(35B 4000 골든 실측).
// [FLA-7 기각 2026-10-10] N64→128(A f32 재판독 절반 — L2 68% 바운드 실측)
// 골든은 불변이나 bf16_mma 27→32ms·프리필 145→149ms — 점유 6→4블록 손실
// 우세(마린-A3 KC64와 동형). 이 커널은 L2보다 점유 민감. 되돌림.
#define BMMA_N 64
#define BMMA_M 32

__device__ __forceinline__ unsigned short f2bf16(float v) {
    // RNE — __floats2bfloat162_rn과 동일 비트(스모크에서 대조 검증).
    unsigned u = __float_as_uint(v);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// [FLA-10 2026-10-10] f32 → bf16 미러 — bf16_mma의 A 재판독(L2 68% 실측)
// 절반화. f2bf16과 동일 산술(비트동일 — 커널 내 변환을 대체). float4 벡터.
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

// ldmatrix — 프래그먼트 smem 재판독(8× 중복)을 1명령으로. A는 x4(비전치),
// B는 x2.trans([n][k]→[k][n] 전치 = col-major 프래그먼트).
__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                       unsigned& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ void ldm_x2(unsigned& r0, unsigned& r1, const void* p) {
    // [버그 수정] B는 비전치 — ldmatrix 분배(스레드 i = 행 i/4, 열 2(i%4),+1)가
    // b0={w[g][2t],w[g][2t+1]}와 정확히 일치한다. .trans를 쓰면 전치돼
    // 값이 계통적으로 틀린다(실측 maxabs 3e10·불량 100%).
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}

extern "C" __global__ void __launch_bounds__(256, 6) w4a16_gemm_bf16_mma(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32 (xbf=0) / bf16 미러
    float* __restrict__ out,               // [t][n]
    int n, int k, int t, int xbf)
{
    // [뱅크 충돌] 행 stride를 128B(전 뱅크 주기)로 두면 프래그먼트 로드가
    // 8-way 충돌(실측: 패딩 없음 0.056ms = v3와 동일). +8 bf16(16B) 패딩으로
    // 행마다 4뱅크씩 이동 → conflict-free.
    __shared__ unsigned short xs[BMMA_M][MMA_KC + 8];
    __shared__ unsigned short ws[BMMA_N][MMA_KC + 8];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * BMMA_M;
    const int n0 = blockIdx.y * BMMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 2) * 16;      // [B1/B3] 2 m워프 × 16 = M32
    const int ntw = (warp & 3) * 16;      // 4 n워프 × 16 = N64(워프 16×16)
    float c[2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += MMA_KC) {
        __syncthreads();
        // [FLA-10] A 스테이징 — xbf=1이면 bf16 미러 직접 복사(uint4), 아니면
        // f32→bf16 변환(float4). 두 경로의 xs 값은 비트동일.
        if (xbf) {
            const unsigned short* x16 = reinterpret_cast<const unsigned short*>(x);
            for (int e = tid; e < BMMA_M * MMA_KC / 8; e += 256) {
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
        for (int e = tid; e < BMMA_M * MMA_KC / 4; e += 256) {
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
        // ws 스테이징: N64×KC64 bf16 (uint4 벡터화)
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
            // A: ldmatrix.x4 — 레인 0-15이 행 0-15(열 kb), 16-31이 행 0-15(열 kb+8).
            const int row = (lane & 15);
            const int colblk = (lane & 16) ? 8 : 0;
            unsigned a0, a1, a2, a3;
            ldm_x4(a0, a1, a2, a3, &xs[mt + row][kb + colblk]);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int nb = ntw + nt * 8;
                // B: ldmatrix.x2.trans — 레인 0-7이 n행(열 kb), 8-15가 n행(열 kb+8).
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
    // 에필로그 — 프래그먼트 규약 그대로 기록(범위 가드).
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
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

// [R5 2026-10-10] G4_TMAX 재정의 제거 — 단일 정의(gemv_bf16_t 상단)로 통합.
// 플레인 bf16 GEMM(t≤8) — 행=블록(64레인), **가중치 1회 판독 × t토큰 재사용**
// (t=1 경로와 같은 레인·환원 순서 — 판정은 토큰 수준, 골든). x는 원시 f32
// [t][k](전치 없음), out [t][n]. 프리필 청크(t≤8)의 dense 경로.
extern "C" __global__ void w4a16_gemm_bf16(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32
    float* __restrict__ out,               // [t][n]
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

// 플레인 bf16 GEMM v3(t≤32) — **8행/블록(512스레드) + 행별 smem 가중치 + k청크**.
// [v2(1행/블록)는 x를 행마다 재판독해 L2 트래픽이 n×t×k×4(실측 8GB/층) —
// v3는 8그룹이 같은 x를 L1 공유(÷8)하고 가중치도 청크 단위 smem 재사용.]
// 산술 순서는 v1/v2와 동일(레인 l = i=l,l+64,… f32 누산 → tree64 — 토큰 독립).
#define G4_TMAX2 32
#define G4_ROWS_B 8
#define G4_KC_B 128
extern "C" __global__ void w4a16_gemm_bf16_t(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32
    float* __restrict__ out,               // [t][n]
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
        __syncthreads(); // 다음 청크 스테이징 전 소비 완료
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

// 가중 누적 — y[i] += w·x[i] (mul·add 분리 — CPU MoE 스테이지와 동일 산식).
extern "C" __global__ void w4a16_axpy(float w, const float* __restrict__ x,
                                      float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += w * x[i];
    }
}

// 전문가 가중 누적 — y[row][i] = Σ_{s∈row} w[s]·d[s][i] (선택 순서 가산 — CPU 미러).
// [FLA-6 2026-10-10] 행(토큰) 병렬 — grid.y = 출력 행. 종전엔 열만 병렬(그리드
// n/256 = 8블록)로 토큰 루프가 스레드당 4096회 직렬(실측 GPU 99% 유휴,
// 343µs/런치 = DRAM의 1/10). 가산 순서·값 불변(비트동일).
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

// [R3 2026-10-10] 구 w4a16_moe_topk(단일 스레드·32스레드 블록) 제거 —
// t=1은 워프 병렬 w4a16_moe_topk_t(B-3), n=128은 w4a16_moe_topk256(B-4)이
// 대체. 시맨틱 미러 설명은 아래 topk_t 주석 참조.

// [P11] 프리필용 배치 라우터 top-k — 워프당 1토큰(t 토큰 동시). 종전엔
// 층·청크마다 d2h(32KB)+sync+호스트 전체 정렬(512×32)로 프리필의 ~10%.
// 시맨틱은 호스트 moe_topk 미러: softmax(전문가 전체) → k라운드 선택
// (p 내림차순, 동률 낮은 idx) → 재정규화. 플레인 경로 — 토큰 골든 판정.
// 워프 환원 순서는 호스트와 다르다(계약 완화 승인 — 허용오차/토큰 판정).
#define MOE_TK_WARPS 8

extern "C" __global__ void w4a16_moe_topk_t(
    const float* __restrict__ lg,   // [t][n_exp]
    unsigned* __restrict__ idx,     // [t][k]
    float* __restrict__ wt,         // [t][k]
    int t, int n, int k)
{
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ti = blockIdx.x * MOE_TK_WARPS + w;
    if (ti >= t || n > 1024) {
        return;
    }
    const float* l = lg + (size_t)ti * n;
    // softmax(최대 빼기) — 워프 환원.
    float mx = -INFINITY;
    for (int i = lane; i < n; i += 32) {
        mx = fmaxf(mx, l[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    // p를 레지스터에 보관(라운드마다 재계산 금지) — 레인당 n/32 ≤ 32개.
    float pv[32];
    unsigned pidx[32];
    int cnt = 0;
    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) {
        // [B-4 후속 2026-10-10] 정밀 expf → __expf(단문 실측 10.4µs = 의존성
        // 지연 지배, IPC 0.36). ew와 같은 완화 등급 — 토큰 골든 판정.
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
        // 워프 환원 — (v, i) 쌍, 동률 낮은 idx.
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        // 승자 마킹(승자를 소유한 레인이 표시).
        for (int j = 0; j < cnt; ++j) {
            if (pidx[j] == bi) {
                used |= (1u << j);
            }
        }
        if (lane == 0) {
            idx[(size_t)ti * k + r] = bi;
            wt[(size_t)ti * k + r] = bv; // 원시 p — 재정규화에서 나눈다.
        }
        wsum += bv;
    }
    // 재정규화 — 호스트 moe_topk의 (p/sum)/(Σtop-k p/sum) = raw/Σraw와 동일
    // (sum이 분자·분모에서 상쇄). 여기서 sum을 또 나누면 가중치가 1/sum배 작아진다.
    // [레이스 수정] 위 루프의 기록(레인 0)과 아래 판독(전 레인) 사이에
    // __syncwarp 필요 — 없으면 실행마다 토큰이 달라진다(실측).
    __syncwarp();
    for (int r = lane; r < k; r += 32) {
        wt[(size_t)ti * k + r] = wt[(size_t)ti * k + r] / wsum;
    }
}

// [B-4 후속 2026-10-10] 라우터 top-k t=1 전용(n=256 = 32레인×8) — **전부
// 레지스터**. 종전 topk_t는 pv[32]/pidx[32]의 동적 인덱싱이 로컬 메모리
// 왕복을 강제 → 단일 워프 임계 경로 2,474명령/21,445사이클(실측 10.2µs,
// 층당 = 디코드의 ~5%). 산술·환원 순서·동률 규칙은 topk_t와 **비트동일**
// (softmax 순차 누산·shuffle_xor 16→1·라운드 스캔 순서 그대로).
#define MOE_TK256_N 256
#define MOE_TK256_C 8

extern "C" __global__ void w4a16_moe_topk256(
    const float* __restrict__ lg, // [256]
    unsigned* __restrict__ idx,   // [k]
    float* __restrict__ wt,       // [k]
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

// shared 게이트 가산 — y[i] += sigmoid(sg[0])·x[i] (sigmoid = 1/(1+e^-v)).
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
