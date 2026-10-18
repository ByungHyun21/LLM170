// ── [⑥ 2026-10-18] MoE 프리필 grp 커널의 **A=f16 단일패스** 변형 ──
// 원본: gptq4.cu `w4a16_gemm_g32_mma_grp2`(게이트/업 2N 융합) ·
//       moe.cu `w4a16_gemm_g32_mma_grp_d64`(down N64) — A = (hi|lo) u32
//       2-pass(mma 2회: A_hi·B + A_lo·B, k스텝당 A 로드 8 + hilo 분해 8ALU).
//
// 배경(ncu 실측 2026-10-18, 35B pp512·180W):
//   grp2 5.37ms/층 — 스톨 지배 = wait(고정지연 의존) 33.6% + math_pipe 9.6%,
//   mma 파이프 유휴·SM 48.6%·reg 79(점유 3블록/SM). A 2패스가 k스텝당
//   mma·프래그먼트 ALU를 배증하는 구조라 지배 스톨에 정면으로 겹친다.
//
// 본 변형: A = f16(RNE) 단일 — cast_hilo의 hi 성분과 비트동일(lo 항 제거).
//   k스텝당 mma 8→4회·A 프래그먼트 (로드 8+hilo ALU 8) → 로드 4.
//   A 스테이징 바이트도 절반(2B/원소). B(가중치)는 원판 그대로 f16 단일.
//
// 수치 클래스: f16-A는 과거 실측(2026-10-15) en PPL +0.17%·agree 98.6%
// 밴드 — 채택 판정은 eval 골든/게이트(2026-10-18 개정·노이즈 클래스 조항).
//
// gptq4.cu/moe.cu는 바이트 동결(코드젠 드로우 보존) 유지 — 본 파일이 대체
// 경로이며 채택 확정 시 원판 정리(삭제+드로우 채택)를 별도로 한다.
// [빌드 계약] -fmad=false — gptq4/moe와 동일 산식 계약.

#include <cuda_fp16.h>

#include "cast_common.cuh"
#include "g4_common.cuh"

#define GRP_M 64
#define GRP_N 32
#define GRP_KC 32
#define DGRP_M 64
#define DGRP_N 64
#define DGRP_KC 32
// [⑥-c2 2026-10-18] m-타일 z분할 수 — 슬롯 분포 편중(97+ 슬롯 전문가 존재,
// 실측 moe-cnt)으로 단일 블록이 mtiles를 직렬 처리하면 테일이 지배:
// grid.x = 전문가×PP_MZ, 블록 = (e = x/PP_MZ, mz = x%PP_MZ), m타일 = mz부터
// PP_MZ 스트라이드. 산술 불변 = 비트동일. 호스트 MOE_PP_MZ와 동일 계약.
#define PP_MZ 4

// [⑥-e 2026-10-18] ldmatrix — fragment smem 재판독(8× 중복)을 1명령으로.
// A는 x4(비전치), B는 x2(비전치 — mma.col B 프래그먼트와 분배 일치. .trans는
// 전치돼 계통 오류 — moe.cu 실측 이력).
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

// [⑥-f 2026-10-18] 디큐트 전용 초경량 f32→f16 — `cvt.rn.f16.f32` 단일 명령.
// (g4_common의 f2h는 호스트 전수 대조용 서브노멀 정규화 루프 + 분기 —
// SASS 실측 원소당 ~20명령·BSSY/BSYNC 2쌍. 본 입력 = (nib−8)·sc — 정확한
// ≤15비트 곱(sc는 f16 값의 f32 정확 표현)이라 HW cvt RNE = 소프트 f2h RNE와
// 동일 결과(서브노멀 출력 포함) — 비트동일은 벤치 해시로 검증.)
__device__ __forceinline__ unsigned short deq_hi_v(unsigned qw, int j, float sc) {
    const int nib = (int)((qw >> (4 * j)) & 0xFu) - 8;
    return __half_as_ushort(__float2half_rn((float)nib * sc));
}

// [⑥] f32 → f16(RNE) 미러 — 단일패스 A 소스(hilo_pack의 hi 성분과 비트동일).
extern "C" __global__ void w4a16_cast_f16(const float* __restrict__ in,
                                          unsigned short* __restrict__ out, int n) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i4 + 1) * 4 <= n) {
        const float4 v = *reinterpret_cast<const float4*>(in + i4 * 4);
        unsigned* o = reinterpret_cast<unsigned*>(out + i4 * 4);
        o[0] = (unsigned)f2h(v.x) | ((unsigned)f2h(v.y) << 16);
        o[1] = (unsigned)f2h(v.z) | ((unsigned)f2h(v.w) << 16);
    } else {
        for (int j = i4 * 4; j < n; ++j) {
            out[j] = f2h(in[j]);
        }
    }
}

// grp2 f16-A — 게이트·업 2N 융합(N64 = 게이트 32 + 업 32), M=전문가 슬롯 수.
// [⑥-d 2026-10-18] __launch_bounds__(256,4) — 점유 3→4블록(reg 70→≤64, 이슈율
// 2.4/4·wait 32% 스톨을 워프 수로 메움; xs 스트라이드 반감으로 smem 20.5KB).
extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemm_g32_mma_grp2_f16(
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
    const int mt = (warp >> 1) * 16; // 4 m워프 × 16 = M64
    const int ntw = (warp & 1) * 32; // 2 n워프 × 32 = N64(게이트 32+업 32)
    const int k8 = k >> 3;
    const int kg = k >> 5; // g32
    // A f16 — 행 스트라이드 40half(80B): 원소 32 + 뱅크패딩 8. 워드 인덱스
    // (r*20 + j/2) → 레인 뱅크 20g+tt = 전역 무충돌(160B 스트라이드의 2-way
    // 충돌 제거) + smem 반감([⑥-d 2026-10-18]).
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
                const float scf = __uint_as_float((unsigned)scb << 16);
                // 스트리밍 힌트 — B는 블록별 1회 판독(재사용 없음),
                // L2를 A 재사용에 남긴다.
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
                // [⑥-e] ldmatrix: A x4(레인 0-15 = 행 mt+0..15·열 kb, 16-31 = +8열),
                // B x2(레인 0-7 = n행·열 kb, 8-15 = +8열) — 프래그먼트 값 동일(비트동일).
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

// grp_d64 f16-A — down 단일 투영 N64.
extern "C" __global__ void w4a16_gemm_g32_mma_grp_d64_f16(
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
    const int mt = (warp >> 1) * 16; // 4 m워프 × 16 = M64
    const int ntw = (warp & 1) * 32; // 2 n워프 × 32 = N64
    const int k8 = k >> 3;
    const int kg = k >> 5; // g32
    // A f16 — 행 스트라이드 40half(80B): 원소 32 + 뱅크패딩 8(무충돌 — ⑥-d).
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
                const float scf = __uint_as_float((unsigned)scb << 16);
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
                // [⑥-e] ldmatrix (grp2_f16와 동일 계약 — 비트동일).
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
