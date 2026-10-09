#include <cuda_fp16.h>

// ── W4A16(GPTQ4·g128·sym) GEMV/GEMM CUDA — 비트 계약: core dot_row_w4a16_lane ──
// 산술 계약(crates/core/src/quant/lane.rs와 1:1):
//  - 레인 l = 0..63: i = l, l+64, … f32 누산
//      w = (nib - 8) as f32 * h2f(scale[g]);  acc += w * h2f(x[i])   (mul·add 각각 반올림)
//  - 레인값 f64 변환 → tree64: a[i]+=a[i+32](0..32) 후 off=16,8,4,2,1
//  - 결과 f32 = (float)tree64  (Rust `as f32` 동일 — RN)
// zp=8 상수(sym — zero-point 미저장). h2f는 Rust half_to_f32와 비트동일
// 구현(서브노멀 정규화 루프 포함, libdevice 미사용).
//
// [빌드 계약] -fmad=false 필수 — `acc += w*x`가 FMA로 수축되면 CPU 미러와
// 비트가 어긋난다(scripts/build_cuda_kernels.sh). x·scale은 f16 비트(u16),
// packed는 u32(니블 8개/워드, lsb-first).

#include "cast_common.cuh"

// 활성 f32 [n] → f16 비트 [n] (GEMV 입력 스테이징 계약).
extern "C" __global__ void w4a16_cast_f16(const float* __restrict__ in,
                                          unsigned short* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = f2h(in[i]);
    }
}

// 활성 f32 [n] → h2f(f2h(v)) f32 [n] — t=1 GEMV 입력의 사전 변환.
// 계약: 커널 안에서 h2f(xt[i])하던 값을 밖에서 한 번 계산해 두는 것과 동일
// (f2h→h2f 왕복이 비트를 보존). 반드시 f2h/h2f와 동형 수정.
extern "C" __global__ void w4a16_cast_x32(const float* __restrict__ in,
                                          float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = h2f(f2h(in[i]));
    }
}

#define G4_LANES 64
#define G4_ROWS 8

// t=1 전용 GEMV — 한 블록 = 한 행(64레인). 구 GEMM 커널은 블록이 8행을
// 순차 처리해 메모리 지연을 못 숨기고(실측 GPU 88ms/토큰), 여기서는 격자를
// n으로 늘려 지연을 숨긴다. 스케일은 smem f32로 선변환, 활성 x는
// w4a16_cast_x32가 만든 f32(h2f 왕복). 산술 순서는 구 커널과 동일:
//   w = (nib-8) as f32 * sc[g];  acc += w * x[i];   (mul·add 분리, 레인 l은
//   i=l,l+64,… 오름차순 → f64 tree64). k ≤ 128*G4_SCMAX 계약.
#define G4_SCMAX 256

// double 셔플(__shfl_down_sync는 32bit) — 트리 병렬화용.
__device__ __forceinline__ double shfl_down_f64(double v, int off) {
    unsigned lo = __double2loint(v), hi = __double2hiint(v);
    lo = __shfl_down_sync(0xffffffffu, lo, off);
    hi = __shfl_down_sync(0xffffffffu, hi, off);
    return __hiloint2double(hi, lo);
}

// f64 tree64 — core lane.rs와 동일 순서(전 호출부 공용).
__device__ __forceinline__ float tree64(double* red) {
    for (int i = 0; i < 32; ++i) {
        red[i] += red[i + 32];
    }
    for (int off = 16; off >= 1; off >>= 1) {
        for (int i = 0; i < off; ++i) {
            red[i] += red[i + off];
        }
    }
    return (float)red[0];
}

// ── 그룹·스케일 dtype 변형(W4-1: 35B-A3B = g32·BF16 스케일) ──
// bf16 → f32: 상위 16비트 좌시프트(정확) — core quant::deq::bf16_to_f32 동일.
__device__ __forceinline__ float b2f(unsigned short v) {
    return __uint_as_float(((unsigned)v) << 16);
}

// 스케일 로드 — BF16이면 b2f, 아니면 h2f(기존 g128·f16 경로와 동일 값).
template <bool BF16>
__device__ __forceinline__ float ld_scale(const unsigned short* __restrict__ p, int i) {
    return BF16 ? b2f(p[i]) : h2f(p[i]);
}

// 스케일 그룹 인덱스 — i = l + 64j (l = 레인, j = 64원소 스텝).
// g128(SHIFT=7): (l+64j)>>7 = j>>1 (l<64라 레인 공통), g32(SHIFT=5):
// (l>>5) + 2j. 일반식 (l>>SHIFT)+(j<<(6-SHIFT))는 SHIFT=7에서 시프트가
// 음수라 분기한다.
template <int SHIFT>
__device__ __forceinline__ int sidx(int l, int j) {
    if constexpr (SHIFT == 7) {
        return j >> 1;
    } else {
        return (l >> SHIFT) + (j << (6 - SHIFT));
    }
}

// 2행/블록 — x를 두 행이 공유(L1 x 트래픽 ÷2)하고 레인당 미결 로드가 2배.
// 스케일은 행별(각 행의 srow), 산술 순서는 1행 커널과 동일(계약 불변).
// SHIFT = 그룹 로그2(7 = g128, 5 = g32) — 산술 계약은 core
// dot_row_w4a16_lane_group과 1:1.
template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_row_body(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/2^SHIFT] 스케일(f16/bf16 비트)
    const float* __restrict__ x,           // [k] f32 = h2f(f2h(활성))
    float* __restrict__ out,               // [해당 행 1개]
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
    // 스케일 그룹은 i>>7 = (l + 64j)>>7 = j>>1 — 전 레인 공통(유니폼 로드).
    // 16이터레이션 언롤 — 미결 q·x 로드를 16개까지 겹친다(지연 은닉, 실측 8→16
    // = 27.9→27.2ms). 2행/블록 변형은 역효과(34.8ms — 레지스터·L1 압박).
    const int jn = k >> 6;
    const int li = l & 7;
    float acc = 0.0f;
    int j = 0;
    for (; j + 16 <= jn; j += 16) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int jj = j + u;
            const int i = l + (jj << 6);
            // evict-first — 한 번 읽는 가중치가 L2를 오염시키지 않게(스트리밍).
            unsigned qw = __ldcs(&qrow[i >> 3]);
            int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
            float w = (float)v * sc[sidx<SHIFT>(l, jj)];
            acc += w * x[i];
        }
    }
    for (; j < jn; ++j) {
        const int i = l + (j << 6);
        unsigned qw = __ldcs(&qrow[i >> 3]);
        int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
        float w = (float)v * sc[sidx<SHIFT>(l, j)];
        acc += w * x[i];
    }
    red[l] = (double)acc;
    __syncthreads();
    // 트리 병렬화(계약 순서 유지 — 1단 a[i]+=a[i+32] 후 셔플로 off 16..1).
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

// 전문가 배치 GEMV — 전문가 포인터 테이블(tab, (q,s) 쌍) + 슬롯 인덱스.
// 테이블 레이아웃 = (il, e, proj) 순서 — 커널 항목 = base + idx[sl]·3
// (같은 층·proj의 전문가). xstride: 게이트/업은 0(x 공통 브로드캐스트),
// 다운은 k(슬롯별 활성 — x = [nslots][k]). grid = n × nslots(블록 = (행,슬롯)),
// out = [nslots][n]. 상주 전용(스트리밍은 호스트 h2d 경로).
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
    // sp = 토큰당 슬롯 수(프리필 = top_k, 디코드 = 1) — x는 토큰 단위 공유.
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    gemv_row_body<5, true>((const unsigned*)e[0], (const unsigned short*)e[1],
                           x + xoff, out + (size_t)sl * n, o, k);
}

// t=1 GEMV 래퍼 — g128·f16(27B) / g32·bf16(35B 전문가).
extern "C" __global__ void w4a16_gemv_g128(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body<7, false>(q, s, x, out, n, k);
}
extern "C" __global__ void w4a16_gemv_g32_bf16(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body<5, true>(q, s, x, out, n, k);
}

// out[t][n] = x[t][k] · W4A16(g128, sym) — t≥2(프리필 배치) 전용.
// [2026-10-08 P3-b 최종 — 채택 변형] 1라운드 진화(벤치 실측): 행=블록 43 →
// smem(R=4) 51 → smem f16→f32 변환 63 → **R=8(512스레드) 72~77(채택)** →
// 레지스터 블로킹 RL=2 61(기각) → f32 직접 52(기각) → cp.async 2단 62(기각:
// 스테이징 지연은 병목 아님). ncu는 권한(ERR_NVGPUCTRPERM)으로 카운터 불가 —
// 잔여 지연 요인은 미규명(모델: warp-jj ~30사이클 × 2.78M ≈ 380µs vs 실측 620).
// 채택본: R=8 + x f32(cast_x32 공유 — 변환·h2f 없음) + smem 스테이징 + 언롤.
// 산술 계약 불변(행·토큰별 레인 l은 i=l,l+64,… 오름차순 f32 누산 → tree64).
#define G4_GTMAX 32            // g128 GEMM 토큰 상한(2026-10-09: 8→32)
#define G4_ROWS 8
#define G4_LANES 64
#define G4_KC 128              // k-청크(64의 배수 · x smem 128×36×4=18KB)

// SHIFT = 그룹 로그2(7 = g128, 5 = g32), BF16 = 스케일 dtype.
template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemm_body(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/2^SHIFT] 스케일(f16/bf16 비트)
    const float* __restrict__ x,           // [t][k] f32 = h2f(f2h(활성))
    float* __restrict__ out,               // [t][n]
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
    // x 스테이징 = 전치+패딩 레이아웃([il][36]): 레인당 float4 8회 판독
    // (종전 8토큰 float4 2회 — 2026-10-09 t≤32 확대). 36 = 32토큰 + 4패딩
    // (16B 정렬·뱅크 충돌 회피: 행 stride 144B).
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
        __syncthreads(); // 이전 청크 소비 완료 대기
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
                    const unsigned qw = __ldcs(&qrow[i >> 3]); // 가중치 1회 = t토큰 공유
                    const int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
                    const float w = (float)v * sc[g][i >> SHIFT];
                    // 전치 패딩 레이아웃에서 float4 2회로 8토큰 판독(smem 트래픽 ÷4).
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
    // 토큰별 트리 — 계약 순서(1단 a[i]+=a[i+32] 후 off 16,8,4,2,1)를 병렬화:
    // 워프 셔플로 각 단을 병렬 가산(각 가산은 독립 — 비트 동일), 배리어는
    // red 기록용 1회/토큰만.
    // [2026-10-09 결함 수정] 종전엔 ti≥t 행도 트리·기록을 수행했다 — 출력
    // 스크래치(ensure_dyt)가 t행만 할당되므로 t<32에서 버퍼 밖에 유한값을
    // 기록하는 무증상 잠복 결함(값이 유한해 크래시·오염이 드러나지 않음).
    // 유효 토큰(ti<t)만 트리·기록한다 — 산술 계약 무변.
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

// t≥2 GEMM 래퍼 — g128·f16(27B) / g32·bf16(35B 전문가).
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

// ── T2(2026-10-09): 플레인 bf16 mma GEMM — 계약 완화 승인 후 첫 본체 ──
// A=x(bf16 반올림, 스테이징에서 f32→bf16 RNE), B=w bf16(스토어 원본 [n][k]),
// f32 누적. 프래그먼트 규약은 --mma-smoke에서 검증(비트동일 0.00e0):
//   A: a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t..]} a2={A[g][2t+8..]} a3={A[g+8][2t+8..]}
//   B(col-major, B[k][n]=w[n][k]): b0={w[g][2t],w[g][2t+1]} b1={w[g][2t+8],w[g][2t+9]}
//   C: c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]  (g=lane>>2, t=lane&3)
// 타일: 블록 256스레드(8워프) = M32 × N64, k청크 64(스테이징 12KB) → 워프당
// m16×n16(2× n8 mma). 워프 w: m타일 w/4, n타일 (w%4)*16.
#define MMA_M 32
#define MMA_N 32
#define MMA_KC 64   // 128은 smem 증가로 블록 감소(0.18 vs 0.13ms 실측)

__device__ __forceinline__ unsigned short f2bf16(float v) {
    // RNE — __floats2bfloat162_rn과 동일 비트(스모크에서 대조 검증).
    unsigned u = __float_as_uint(v);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ unsigned pk2bf(const unsigned short* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 16);
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

extern "C" __global__ void w4a16_gemm_bf16_mma(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32
    float* __restrict__ out,               // [t][n]
    int n, int k, int t)
{
    // [뱅크 충돌] 행 stride를 128B(전 뱅크 주기)로 두면 프래그먼트 로드가
    // 8-way 충돌(실측: 패딩 없음 0.056ms = v3와 동일). +8 bf16(16B) 패딩으로
    // 행마다 4뱅크씩 이동 → conflict-free.
    __shared__ unsigned short xs[MMA_M][MMA_KC + 8];
    __shared__ unsigned short ws[MMA_N][MMA_KC + 8];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * MMA_M;
    const int n0 = blockIdx.y * MMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 2) * 16;      // 워프 m 오프셋(0/16)
    const int ntw = (warp & 3) * 8;      // [병렬도] T1과 동일(16×8)
    float c[2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += MMA_KC) {
        __syncthreads();
        // xs 스테이징: M32×KC64 f32→bf16 (float4 벡터화)
        for (int e = tid; e < MMA_M * MMA_KC / 4; e += 256) {
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
        for (int e = tid; e < MMA_N * MMA_KC / 8; e += 256) {
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
            for (int nt = 0; nt < 1; ++nt) {
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
    for (int nt = 0; nt < 1; ++nt) {
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

// ── T1(2026-10-09): split(g128·f16) mma GEMM — int4 디퀀트→f16→mma ──
// ncu: split GEMM은 L1/명령 바운드(DRAM 3.7%·compute 46%) → mma 여지 큼.
// A=xh(f16 — split 경로가 이미 f2h 캐스트 제공, 계약과 동일 값), B=디퀀트 f16.
// 디퀀트는 marlin식 마법 상수: f16(1024+n) = 0x6400|n (n<16이 mantissa 하위
// 비트에 정확히 더해짐) → w=(n−8)·s = (1024+n)·s − 1032·s = hfma2 1회.
// 니블 순서는 __byte_perm으로 (e0,e1),(e2,e3) 정렬(8원소/u32당 ~8 op).
// 타일·ldmatrix는 T2와 동일(M32×N64, k청크 64).
extern "C" __global__ void w4a16_gemm_g128_mma(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/128] f16 스케일
    const float* __restrict__ x,           // [t][k] f32 = h2f(f2h(활성)) —
                                           // split 경로 계약(cast_x32 산출).
                                           // 커널이 __float2half_rn로 f16화
                                           // (동일 값 — cast_x32와 같은 반올림).
    float* __restrict__ out, int n, int k, int t)
{
    // [지연 은닉] 더블 버퍼 — ncu: Compute 27%·점유 35%·No Eligible 77%(지연
    // 바운드). 스테이징(k+1)과 mma(k)를 겹친다(버퍼 2×27.6KB = 55KB < 100KB).
    __shared__ unsigned short xs[2][MMA_M][MMA_KC + 8];
    __shared__ unsigned short ws[2][MMA_N][MMA_KC + 8];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * MMA_M;
    const int n0 = blockIdx.y * MMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    // [병렬도] 워프 타일 16×8(N=32, 8워프 = 2m×4n) — ncu: 스케줄러당 활성
    // 워프 4.24(No Eligible 76%)로 지연 노출. 블록 32×32로 grid 272→544.
    // 가중치 트래픽은 타일과 무관(각 블록이 자기 행만 읽음) — x 재판독만 2배.
    const int mt = (warp >> 2) * 16;
    const int ntw = (warp & 3) * 8;
    const int k8 = k >> 3;
    const int kg = k >> 7;
    float c[2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    int cur = 0;
    int k0 = 0;
    // 초기 스테이징(청크 0).
    for (int e = tid; e < MMA_M * MMA_KC / 4; e += 256) {
        const int r = e / (MMA_KC / 4);
        const int c4 = e % (MMA_KC / 4);
        const int gi = c4 * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (m0 + r < t && gi + 3 < k) {
            v = *reinterpret_cast<const float4*>(&x[(size_t)(m0 + r) * k + gi]);
        } else if (m0 + r < t) {
            float t4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int j = 0; j < 4; ++j) {
                if (gi + j < k) {
                    t4[j] = x[(size_t)(m0 + r) * k + gi + j];
                }
            }
            v = make_float4(t4[0], t4[1], t4[2], t4[3]);
        }
        xs[0][r][c4 * 4 + 0] = __half_as_ushort(__float2half_rn(v.x));
        xs[0][r][c4 * 4 + 1] = __half_as_ushort(__float2half_rn(v.y));
        xs[0][r][c4 * 4 + 2] = __half_as_ushort(__float2half_rn(v.z));
        xs[0][r][c4 * 4 + 3] = __half_as_ushort(__float2half_rn(v.w));
    }
    for (int e = tid; e < MMA_N * MMA_KC / 8; e += 256) {
        const int r = e / (MMA_KC / 8);
        const int c8 = e % (MMA_KC / 8);
        const int gi = c8 * 8;
        const bool live = (n0 + r < n) && (gi + 7 < k);
        unsigned short scb = 0;
        if (n0 + r < n) {
            scb = s[(size_t)(n0 + r) * kg + (gi >> 7)];
        }
        const float scf = __half2float(*reinterpret_cast<const __half*>(&scb));
        const unsigned qw = live ? q[(size_t)(n0 + r) * k8 + (gi >> 3)] : 0u;
        if (live) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int nib = (int)((qw >> (4 * j)) & 0xFu) - 8;
                ws[0][r][c8 * 8 + j] = __half_as_ushort(__float2half_rn((float)nib * scf));
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                ws[0][r][c8 * 8 + j] = 0;
            }
        }
    }
    for (; k0 < k; k0 += MMA_KC, cur ^= 1) {
        __syncthreads(); // 이전 compute 완료(버퍼 재사용) + 스테이징 가시화
        // 다음 청크 스테이징(다른 버퍼) — mma와 겹친다.
        const int kn = k0 + MMA_KC;
        if (kn < k) {
            for (int e = tid; e < MMA_M * MMA_KC / 4; e += 256) {
                const int r = e / (MMA_KC / 4);
                const int c4 = e % (MMA_KC / 4);
                const int gi = kn + c4 * 4;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (m0 + r < t && gi + 3 < k) {
                    v = *reinterpret_cast<const float4*>(&x[(size_t)(m0 + r) * k + gi]);
                } else if (m0 + r < t) {
                    float t4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int j = 0; j < 4; ++j) {
                        if (gi + j < k) {
                            t4[j] = x[(size_t)(m0 + r) * k + gi + j];
                        }
                    }
                    v = make_float4(t4[0], t4[1], t4[2], t4[3]);
                }
                xs[cur ^ 1][r][c4 * 4 + 0] = __half_as_ushort(__float2half_rn(v.x));
                xs[cur ^ 1][r][c4 * 4 + 1] = __half_as_ushort(__float2half_rn(v.y));
                xs[cur ^ 1][r][c4 * 4 + 2] = __half_as_ushort(__float2half_rn(v.z));
                xs[cur ^ 1][r][c4 * 4 + 3] = __half_as_ushort(__float2half_rn(v.w));
            }
            for (int e = tid; e < MMA_N * MMA_KC / 8; e += 256) {
                const int r = e / (MMA_KC / 8);
                const int c8 = e % (MMA_KC / 8);
                const int gi = kn + c8 * 8;
                const bool live = (n0 + r < n) && (gi + 7 < k);
                unsigned short scb = 0;
                if (n0 + r < n) {
                    scb = s[(size_t)(n0 + r) * kg + (gi >> 7)];
                }
                const float scf = __half2float(*reinterpret_cast<const __half*>(&scb));
                const unsigned qw = live ? q[(size_t)(n0 + r) * k8 + (gi >> 3)] : 0u;
                if (live) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int nib = (int)((qw >> (4 * j)) & 0xFu) - 8;
                        ws[cur ^ 1][r][c8 * 8 + j] =
                            __half_as_ushort(__float2half_rn((float)nib * scf));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[cur ^ 1][r][c8 * 8 + j] = 0;
                    }
                }
            }
        }
#pragma unroll
        for (int ks = 0; ks < MMA_KC / 16; ++ks) {
            const int kb = ks * 16;
            // 프래그먼트 = 스모크 검증 방식(명시 판독). ldmatrix는 전치/분배
            // 불일치로 계통 오차(실측 maxabs 3e10) — 정확성 우선으로 되돌림.
            const unsigned a0 = pk2bf(&xs[cur][mt + g][kb + 2 * tt]);
            const unsigned a1 = pk2bf(&xs[cur][mt + g + 8][kb + 2 * tt]);
            const unsigned a2 = pk2bf(&xs[cur][mt + g][kb + 2 * tt + 8]);
            const unsigned a3 = pk2bf(&xs[cur][mt + g + 8][kb + 2 * tt + 8]);
#pragma unroll
            for (int nt = 0; nt < 1; ++nt) {
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
    }
#pragma unroll
    for (int nt = 0; nt < 1; ++nt) {
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

#define G4_TMAX 8              // 플레인 v1 GEMM 전용 토큰 상한

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

// 전문가 가중 누적 — y[i] += Σ_s w[s]·d[s][i] (선택 순서 가산 — CPU 미러).
extern "C" __global__ void w4a16_moe_accum(const float* __restrict__ w,
                                           const float* __restrict__ d,
                                           float* __restrict__ y, int sp, int nslots,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int spt = (sp > 0) ? sp : nslots;
        for (int ti = 0; ti < nslots; ti += spt) {
            float acc = 0.0f;
            const int end = (ti + spt < nslots) ? ti + spt : nslots;
            for (int s = ti; s < end; ++s) {
                acc += w[s] * d[(size_t)s * n + i];
            }
            y[(size_t)(ti / spt) * n + i] = acc;
        }
    }
}

// [2026-10-09 P1] MoE 라우터 top-k — 호스트 moe_topk 미러:
// softmax(f32, max-빼기) → k라운드 최대 선택(p 내림차순, 동률 낮은 idx) →
// 재정규화. 단일 스레드 순차(결정적·캡처 안전 — 호스트 왕복 제거). n ≤ 1024.
// exp는 CUDA expf — 호스트 libm exp와 ulp 차이 허용(플레인 경로 — 토큰 골든
// 판정). idx/wt는 디바이스에 남아 배치 GEMV·moe_accum이 직접 소비한다.
extern "C" __global__ void w4a16_moe_topk(const float* __restrict__ lg,
                                          unsigned* __restrict__ idx,
                                          float* __restrict__ wt, int n, int k) {
    __shared__ float p[1024];
    __shared__ unsigned char used[1024];
    if (threadIdx.x != 0) {
        return;
    }
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) {
        mx = fmaxf(mx, lg[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        p[i] = expf(lg[i] - mx);
        sum += p[i];
    }
    for (int i = 0; i < n; i++) {
        p[i] /= sum;
        used[i] = 0;
    }
    float wsum = 0.0f;
    for (int r = 0; r < k; r++) {
        int bi = 0;
        float bv = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (used[i]) {
                continue;
            }
            if (p[i] > bv) {
                bv = p[i];
                bi = i;
            }
        }
        used[bi] = 1;
        idx[r] = (unsigned)bi;
        wt[r] = bv;
        wsum += bv;
    }
    for (int r = 0; r < k; r++) {
        wt[r] /= wsum;
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
