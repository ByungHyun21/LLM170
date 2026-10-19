// ── MoE·플레인 bf16 커널 — [R5 2026-10-10] gptq4.cu에서 분할 ──
// 35B-A3B의 비양자화(플레인) 경로 전부: bf16 GEMV/GEMM(mma 포함) + 라우터
// top-k·전문가 가중 누적·게이트 보조. 분할(g128/g32) 양자화 커널과 전문가
// 배치 GEMV는 gptq4.cu 소유(공용 define·헬퍼는 g4_common.cuh).
// [빌드 계약] -fmad=false — 플레인 경로도 mul·add 분리 산식(CPU 미러 계약).

#include <cuda_fp16.h>

#include "cast_common.cuh"
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
// [S1-a 실측 기각 2026-10-11] 환원 재배치(상반부 슬롯+1배리어) 시도 — 실측
// 역행으로 원복(gptq4.cu gemv_g128_t_body 기록 참조).
// [S1-b 실측 기각 2026-10-11] GEMV_TR 8→16(공용 define) — g128_t에서 27B
// 프리필 NaN, 본 커널(bf16_t)은 35B 프리필 정상이나 공용 define 원복에 포함
// (기록: gptq4.cu).
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
    // [T16-reg 2026-10-15] 행별 x 포인터 배열(xp[T]) 제거 — T=16에서 REG가
    // 48→94까지 올라 점유가 절반으로 떨어졌다(cuobjdump 실측). 단일 베이스 +
    // (r·k) 오프셋 로드로 전환 — 적재 값·순서 불변(행별 산술 비트동일).
    // 주의: 루프 인덱스 i가 l로 시작하므로 **베이스에 l을 더하지 않는다**
    // (초판 이중 가산 버그 — 골든 붕괴로 검출).
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

// t 상한 — 호스트 G4_CHUNK_MAX 미러(gemma4 t행 청크 상한; GEMV_T_CASE_BF 1..16).
// [R5 2026-10-10] 플레인 v1 GEMM(gemm_bf16)과 공유 — 종전 재정의 2곳을
// 단일 정의로 통합(값 8 불변).
#define G4_TMAX 16

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
        GEMV_T_CASE_BF(9)
        GEMV_T_CASE_BF(10)
        GEMV_T_CASE_BF(11)
        GEMV_T_CASE_BF(12)
        GEMV_T_CASE_BF(13)
        GEMV_T_CASE_BF(14)
        GEMV_T_CASE_BF(15)
        GEMV_T_CASE_BF(16)
        default: break; // 호스트 계약 밖(t≤16)
    }
}

// [S2 2026-10-11] 다중 세그먼트 t행 GEMV(플레인 bf16) — 테이블은 t=1
// w4a16_gemv_multi와 공용((w,out,n,k)). 세그먼트 블록수 ceil(n/GEMV_TR),
// 호스트 grid = Σceil. 행별 산술은 w4a16_gemv_bf16_t와 **비트동일**.
extern "C" __global__ void w4a16_gemv_bf16_multi_t(
    const unsigned long long* __restrict__ tab, // [count][4]: w, out, n, k
    const float* __restrict__ x,                // [t][k] f32(원시)
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
        default: break; // 호스트 계약 밖(t≤16)
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

// [A2b 2026-10-15] f32 → (hi|lo) f16 쌍 패킹 — 그룹 mma 2-pass의 A 미러.
// 4B/원소(f32와 동일 바이트)라 스테이징·cp.async 16B 정렬 계약이 그대로다.
// float4 벡터(캐스트_bf16 동형).
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

// [M16 2026-10-11] 타일 M을 템플릿화 — MM=16 변형은 그리드 기아 형상
// (t/32×n/64 < SM수)에서 블록 2배. 워프 매핑: MM32=2m×4n(워프 16×16),
// MM16=1m×8n(워프 16×8). k 순서·mma 동일 — 두 변형 값 비트동일.
template <int MM>
__device__ __forceinline__ void bmma_body(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32 (xbf=0) / bf16 미러
    float* __restrict__ out,               // [t][n]
    int n, int k, int t, int xbf)
{
    // [뱅크 충돌] 행 stride를 128B(전 뱅크 주기)로 두면 프래그먼트 로드가
    // 8-way 충돌(실측: 패딩 없음 0.056ms = v3와 동일). +8 bf16(16B) 패딩으로
    // 행마다 4뱅크씩 이동 → conflict-free.
    __shared__ unsigned short xs[MM][MMA_KC + 8];
    __shared__ unsigned short ws[BMMA_N][MMA_KC + 8];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * MM;
    const int n0 = blockIdx.y * BMMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (MM == 32) ? (warp >> 2) * 16 : 0; // MM32: 2m워프 / MM16: 1m
    const int ntw = (MM == 32) ? (warp & 3) * 16 : warp * 8; // MM32: 4n / MM16: 8n
    constexpr int NF = (MM == 32) ? 2 : 1; // n프래그먼트 수
    float c[NF][4];
#pragma unroll
    for (int i = 0; i < 4 * NF; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += MMA_KC) {
        __syncthreads();
        // [FLA-10] A 스테이징 — xbf=1이면 bf16 미러 직접 복사(uint4), 아니면
        // f32→bf16 변환(float4). 두 경로의 xs 값은 비트동일.
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
            for (int nt = 0; nt < NF; ++nt) {
                const int nb = ntw + nt * 8;
                // B: ldmatrix.x2 — 레인 0-7이 n행(열 kb), 8-15가 n행(열 kb+8).
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

// [M16] 그리드 기아 형상(t/32×n/64 < SM) 전용 — 블록 2배(워프 16×8).
// 값은 MM32와 비트동일(같은 k 순서·같은 mma) — 골든 판정.
extern "C" __global__ void w4a16_gemm_bf16_mma16(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, int n, int k, int t, int xbf)
{
    bmma_body<16>(w, x, out, n, k, t, xbf);
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

// [gemma4 P2 2026-10-14] per-expert 사전 스케일 — top-k 가중치(전문가 전체
// softmax → top-k → 재정규화(합 1) **후**)에 μ_e = per_expert_scale[idx[i]]를
// 곱한다. HF Gemma4TextRouter 순서 미러(사전 정규화 후 곱 — CPU core gemma4
// stages::route와 동일). top_k(≤1024) 한 런치, nslots = t×top_k.
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

// 전문가 가중 누적 — y[row][i] = Σ_{s∈row} w[s]·d[s][i] (선택 순서 가산 — CPU 미러).
// [FLA-6 2026-10-10] 행(토큰) 병렬 — grid.y = 출력 행. 종전엔 열만 병렬(그리드
// n/256 = 8블록)로 토큰 루프가 스레드당 4096회 직렬(실측 GPU 99% 유휴,
// 343µs/런치 = DRAM의 1/10). 가산 순서·값 불변(비트동일).
// [gemma4 P2 2026-10-14] 층 스칼라 곱 — y[i] = y[i] * s (gemma4 `layer_scalar`).
// axpy(w=s-1, x=y, y=y)와 수치적으로 동일하지 않다(1회 반올림 vs 2회) —
// 전용 커널로 정확히 x*s (HF `hidden_states *= self.layer_scalar`).
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
// [§4.5 2026-10-11] 전문가 누적 + shared 가산 융합 — 종전 accum(덮어쓰기)
// 과 shared_add(+= σ(sg)·s3) 2런치를 1런치로(런치 -1/층). 값은
// acc + σ(sg)·s3 — 종전과 같은 f32 가산 1회·같은 expf 산식(비트동일).
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

// [2026-10-18] 전문가-우선 슬롯 순열 v2 — gptq4.cu w4a16_moe_align의 단일
// 스레드 O(nslots) 직렬 스캐터를 대체. 출력은 완전 동일(전문가별 슬롯 j
// 오름차순 — C4 계약·입력만의 함수, 원자 획득 순 무관). 카운트(원자합)·
// 접두(스레드 0)는 종전과 동일. gptq4.cu 원본은 코드젠 드로우 동결
// (2026-10-11 경고)로 잔존 — 심볼 등록만. n_exp ≤ 1024(호스트 가드)·
// 블록 256. [실측 2026-10-18] 발사당 103.8µs(원본) → 34.7µs(본 커널) —
// 중간 시안 "전문가별 전슬롯 스캔"(65.6µs)은 명령량 열세로 폐기.
// 35B A/B pp512 +1.7%·pp1024 +1.4%·비트동일(해시 불변).
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
    // 스캐터 = 단일 워프 32-슬롯 청크 순차 스캔. 청크 내 랭크는
    // __match_any_sync(같은 전문가 레인 마스크)·popc로 일괄 — 스레드별
    // 전슬롯 스캔(v2 = 66µs 실측, 중복 명령 처리량 바운드)을 청크당
    // ~20명령으로 대체. 진행 카운트는 값별 리더 1인이 비원자 가산
    // (청크 내 값당 그룹 유일 — 결정론). 순서는 j 오름차순 계약 그대로.
    if (tid < 32) {
        const int l = tid;
        const unsigned lt = (1u << l) - 1u;
        for (int i = l; i < n_exp; i += 32) {
            cnt[i] = 0; // 진행 카운터 재사용(접두는 goff/gcnt에 기기록)
        }
        __syncwarp();
        for (int base = 0; base < nslots; base += 32) {
            const int j = base + l;
            const int e = (j < nslots) ? idx[j] : -1;
            const unsigned m = __match_any_sync(0xFFFFFFFFu, e);
            if (e >= 0 && j < nslots) {
                gslot[off[e] + cnt[e] + __popc(m & lt)] = (unsigned)j;
            }
            __syncwarp(); // 전원 랭크 확정 후 카운트 갱신
            if (e >= 0 && j < nslots && (1u << l) == (m & (0u - m))) {
                cnt[e] += __popc(m);
            }
            __syncwarp(); // 갱신 가시화 후 차기 청크
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


// ── [B2 2026-10-12] down 투영 전용 N64 변형 ──
// 근거(ncu): grp 병목 = 청크당 A/B 스테이징 + 배리어 2회(배리어 스톨 41%+
// MIO 12%, mma 파이프 유휴·DRAM 17%). 게이트/업은 grp2(N64 융합)로 상각했고,
// down(N=hidden, GRP_N=32)만 N32로 남아 같은 일을 배리어 2배로 지불했다.
// down은 단일 투영이라 grp2식 2텐서 융합 대신 N을 64로 확장(워프 2n×32=64열,
// n타일 4) — 청크당 mma 작업 2배로 배리어·스테이징 상각.
// 계약: 산술은 grp와 동일 계열(deq_pair_b·pk2bf·mma 누산 순서) — down 값
// 골든 판정(토큰). deq_pair_b는 gptq4.cu 정의의 **비트동일 복제**다(g4_common
// 이동은 전 자산 재빌드 → gptq4 코드젠 추첨 리셋 — 금지).
__device__ __forceinline__ unsigned deq_pair_b_d(unsigned qw, int tt, float sc) {
    const unsigned sub8 = 0x64086408u; // half2(1024+8)
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

// [A2b 2026-10-15] 2-pass 정밀도 회수: A = (hi|lo) f16 쌍 packed u32(4B/원소),
// B = f16 단일 — mma 2회(hi·B + lo·B). A의 f32→f16 반올림(지배 오차원)을
// ~2^-22로 낮춰 f32 FFMA 클래스에 근접시킨다(잔여 = B 반올림·누산 순서).
// B hi/lo 3-pass는 en/ko 비트동일·+26% 비용으로 기각(실측 — g4_common 주석).
// A 스테이징만 2×(packed) — cp.async 16B = 4원소, 정렬 계약 동일.
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
    const int mt = (warp >> 1) * 16; // 4 m워프 × 16 = M64
    const int ntw = (warp & 1) * 32; // 2 n워프 × 32 = N64
    const int k8 = k >> 3;
    const int kg = k >> 5; // g32
    __shared__ __align__(16) unsigned xs[2][DGRP_M][DGRP_KC + 8]; // packed hi|lo
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

// ── [marlin 스테이지 2 착수 2026-10-12] marlin B 배치 mma GEMM(g128·f16) ──
// 소비 레이아웃 = core::marlin::marlin_repack_q 산출물 [k/16][n/64][128] u32.
// 타일 내 u32 index = lane*4 + w (repack.cu의 th_id*4+warp_id 그대로 — lane과
// th_id의 (tc_col, tc_row) 역할이 동일). u32 w = mma B 프래그먼트 쌍:
//   니블 {0,4} = k행 {2t,2t+1}·열 n16, 니블 {1,5} = 같은 열 k {2t+8,2t+9},
//   그리고 q>>8의 니블 {2,6}/{3,7} = 열 n16+8의 같은 k행들 → u32 하나가
//   프래그먼트 2개(f=2w, 2w+1)를 직결(마린 원문 marlin_template.h의
//   b_quant_1 = b_quant_0 >> 8과 동형).
// dequant = marlin 매직 상수(lop3 0xEA + hsub2/hfma2). 값은 종전
// deq_pair_b와 f16 라운딩 동일 — (n−8)·s의 단일 RNE(exact 곱의 RNE).
// 스케일 = 열별 f16 hmul2(청크 KC=64 ⊂ g128 그룹 1개 → 청크당 1회 로드).
// x·k 순서가 기존 w4a16_gemm_g128_mma와 같으므로 출력 비트동일 전제
// (A/B 게이트가 판정). 레퍼런스 대조: reference/GPTQModel/gptqmodel_ext/marlin/
// {gptq_marlin_repack.cu, dequant.h, marlin_template.h} 실측(2026-10-12).

#define MAR_M 64
#define MAR_N 128
#define MAR_KC 32
#define MAR_STAGES 4

// lop3 = (a & b) | c (lut 0xEA) — 마스크/가산 상수는 imm.
__device__ __forceinline__ unsigned mar_and_or(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// u32 → (lo, hi) half2 2개: lo = (니블0−8, 니블4−8), hi = (니블1−8, 니블5−8).
// (dequant.h kU4B8 sym: SUB=0x64086408 · MUL=0x2c002c00 · ADD=0xd480d480)
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

// f16x2 × {s,s} (hmul2) — s는 u32 = (s|s<<16).
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
    const unsigned* __restrict__ qm,      // [k/16][n/64][128] u32 (marlin B)
    const unsigned short* __restrict__ s, // [n][k/128] f16
    const unsigned short* __restrict__ x, // [t][k] f16
    float* __restrict__ out, int n, int k, int t)
{
    // 스테이지 크기: A = 64행×72열, B = (KC/16)×2×128 u32.
    constexpr int A_STAGE = MAR_M * (MAR_KC + 8);
    constexpr int B_STAGE = (MAR_KC / 16) * 2 * 128;
    __shared__ __align__(16) unsigned short xs[MAR_STAGES * A_STAGE];
    __shared__ __align__(16) unsigned wb[MAR_STAGES * B_STAGE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 1; // m16 블록 0..3
    const int wn = warp & 1;  // n64 블록 0..1
    const int m0 = blockIdx.x * MAR_M;
    const int n0 = blockIdx.y * MAR_N;
    const int nt_n = n >> 6;
    const int kg = k >> 7;
    const int n16 = lane >> 2;      // 프래그먼트 열(repack tc_col)
    const int k2 = (lane & 3) * 2;  // 프래그먼트 k행쌍(repack tc_row)

    // 스테이징 주소·프레디케이트 사전계산(루프 내 정수 연산 제거 —
    // SASS 실측: 주소/프레디케이트가 지배 비용이었다).
    const int a_cpr = MAR_KC / 8; // A 행당 16B 청크 수(KC=32 → 4)
    // B 청크(tid<128): (kl*2+ntl) = tid>>5, j = tid&31.
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
    const int cpg = 128 / MAR_KC; // 그룹당 청크 수(KC가 128을 나눔)
    const int ngroups = k >> 7;
    int cid = 0; // 전역 청크 index(스테이지 = cid % MAR_STAGES)
    for (int gi = 0; gi < ngroups; ++gi) {
        // 그룹 스케일 — 그룹 내 모든 청크가 동일(청크 ⊆ g128 그룹).
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

    // 에필로그 — m16n8k16 C 프래그먼트: 행 (g, g+8), 열 (2t, 2t+1).
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


// ── [marlin g32 2026-10-12] g32·bf16 MoE용 변형 — g128 커널과 동일 구조,
//    kg=k/32 · 스케일 = 청크마다 bf16→f16 정확 변환. (MoE 트랙 1조각 —
//    전문가 업로드 repack·g32 게이트는 다음 단계.) ──
extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemm_g32_marlin(
    const unsigned* __restrict__ qm,      // [k/16][n/64][128] u32 (marlin B)
    const unsigned short* __restrict__ s, // [n][k/128] f16
    const unsigned short* __restrict__ x, // [t][k] f16
    float* __restrict__ out, int n, int k, int t)
{
    // 스테이지 크기: A = 64행×72열, B = (KC/16)×2×128 u32.
    constexpr int A_STAGE = MAR_M * (MAR_KC + 8);
    constexpr int B_STAGE = (MAR_KC / 16) * 2 * 128;
    __shared__ __align__(16) unsigned short xs[MAR_STAGES * A_STAGE];
    __shared__ __align__(16) unsigned wb[MAR_STAGES * B_STAGE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 1; // m16 블록 0..3
    const int wn = warp & 1;  // n64 블록 0..1
    const int m0 = blockIdx.x * MAR_M;
    const int n0 = blockIdx.y * MAR_N;
    const int nt_n = n >> 6;
    const int kg = k >> 5;
    const int n16 = lane >> 2;      // 프래그먼트 열(repack tc_col)
    const int k2 = (lane & 3) * 2;  // 프래그먼트 k행쌍(repack tc_row)

    // 스테이징 주소·프레디케이트 사전계산(루프 내 정수 연산 제거 —
    // SASS 실측: 주소/프레디케이트가 지배 비용이었다).
    const int a_cpr = MAR_KC / 8; // A 행당 16B 청크 수(KC=32 → 4)
    // B 청크(tid<128): (kl*2+ntl) = tid>>5, j = tid&31.
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
    const int cpg = 128 / MAR_KC; // 그룹당 청크 수(KC가 128을 나눔)
    const int ngroups = k >> 7;
    int cid = 0; // 전역 청크 index(스테이지 = cid % MAR_STAGES)
    for (int gi = 0; gi < ngroups; ++gi) {
        // 그룹 스케일 — 그룹 내 모든 청크가 동일(청크 ⊆ g128 그룹).
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

            // [g32] 청크(32k)마다 스케일 재장전 — 스케일 원시는 bf16(35B-A3B
            // MoE 계약) → f16 정확 변환(bf16 유효수는 f16 범위 내 정확 표현)
            // → hmul2 = 기존 g32 경로의 f32 곱+RNE와 동일 값.
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

    // 에필로그 — m16n8k16 C 프래그먼트: 행 (g, g+8), 열 (2t, 2t+1).
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



// ── [marlin M16 2026-10-12] 소형-M 변형 — 디코드(t≤16)용. 블록 M16×N256·
// 128스레드(워프4 = n64 4개). 디퀀트·k순서가 M64 커널과 동일 → t≤16에서
// 같은 입력이면 행별 비트동일(전체합 t≤16). ──
#define MM16_M 16
#define MM16_N 256
#define MM16_KC 32
#define MM16_S 4

extern "C" __global__ void __launch_bounds__(128, 4) w4a16_gemm_g128_marlin_m16(
    const unsigned* __restrict__ qm,      // [k/16][n/64][128] u32 (marlin B)
    const unsigned short* __restrict__ s, // [n][k/128] f16
    const unsigned short* __restrict__ x, // [t][k] f16
    float* __restrict__ out, int n, int k, int t)
{
    constexpr int A_STAGE = MM16_M * (MM16_KC + 8);
    constexpr int B_STAGE = (MM16_KC / 16) * 4 * 128;
    __shared__ __align__(16) unsigned short xs[MM16_S * A_STAGE];
    __shared__ __align__(16) unsigned wb[MM16_S * B_STAGE];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wn = tid >> 5; // n64 블록 0..3
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
            const int tl = e >> 5; // kl*4 + ntl
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

// [2026-10-18] f16→f32 변환 — 마린 GEMM의 f16 출력 → 엔진 f32 스트림.
// 순수 값 변환(__half2float — 정확) · n2 = half2 원소 수(n/2) · 그리드-스트라이드.
extern "C" __global__ void w4a16_h2f(const unsigned short* __restrict__ in,
                                     float* __restrict__ out, int n2) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += stride) {
        const __half2 h = *reinterpret_cast<const __half2*>(&in[2 * i]);
        const float2 f = __half22float2(h);
        *reinterpret_cast<float2*>(&out[2 * i]) = f;
    }
}

// [2026-10-19] 마린 MoE 정렬 변환 — gslot/gcnt/goff(전문가별 슬롯 j 오름차순)을
// 마린 형식(sorted_token_ids·expert_ids·num_tokens_post_padded)으로 변환.
// 블록 = block(마지막 블록 패딩 = sentinel ≥ n_slots). 1단계(스레드 0) 블록
// 접두, 2단계(전문가별 병렬 복사) — align2와 동형의 결정론.
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

// [2026-10-19] 마린 MoE 활성 — f16 C13([ns][2*n_ff] = gate|up) → f16 act([ns][n_ff])
// = gelu_pytorch_tanh(gate)*up (ew_gelu·glu_gelu와 동일 수식·f16 반올림).
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
        // 주의: __half를 unsigned short에 직접 대입하면 __half→float→정수
        // 절단 변환이 일어난다(0.0005→0) — 반드시 비트 보존으로 쓴다.
        act[i] = __half_as_ushort(__float2half_rn(a));
    }
}

// [2026-10-19] 마린 MoE 활성(silu) — f16 C13([ns][2*n_ff]) → f16 act([ns][n_ff]).
// qwen35 MoE = silu·mul(엔진 ew(silu·mul)와 동일식: g/(1+__expf(-g))*u).
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

// [2026-10-19] 마린 MoE topk 합산 — f16 C2([ns][h]) → f32 out([t][h]).
// 행 순서 = 슬롯(token*topk+k) — 합산 순서 k 오름차순(마린 mul_topk 가중 적용 후).
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
