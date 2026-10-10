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

// [T3-2] 쌍 배치(128폭 청크) 스케일 인덱스 — i = 2l + 128j (2l < 128).
// g128: (i)>>7 = j, g32: (i)>>5 = (l>>4) + 4j.
template <int SHIFT>
__device__ __forceinline__ int sidx2(int l, int j) {
    if constexpr (SHIFT == 7) {
        return j;
    } else {
        return (l >> 4) + (j << 2);
    }
}

// 2행/블록 — x를 두 행이 공유(L1 x 트래픽 ÷2)하고 레인당 미결 로드가 2배.
// 스케일은 행별(각 행의 srow), 산술 순서는 1행 커널과 동일(계약 불변).
// SHIFT = 그룹 로그2(7 = g128, 5 = g32) — 산술 계약은 core
// dot_row_w4a16_lane_group과 1:1.
//
// [T3-2 2026-10-09] 니블 2개/레인 + float2 x: 128폭 청크에서 레인 l이
// k = 2l, 2l+1(연속)을 담당 — q 워드는 4레인 공유(바이트 1개 = 니블 쌍),
// x는 float2(8B) 로드 1회. 로드/주소 연산 ÷2. 누산 묶음이 바뀌므로
// 비트동일 아님(승인 완화 — 골든 판정). k는 128 배수 계약(전 모델 형상 충족).
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
    // [T3-1/2 2026-10-09] FMA 3-op(가중=fmaf(nib,sc,−8sc), 누산=fmaf) +
    // 포인터 진행 + 니블 쌍/float2 — 레인당 2가중/이터, 128폭 청크.
    // 종전 대비: 가중당 I2F+FFMA+FFMA(3), LDG ÷2, 주소 증분 상수.
    // 비트동일 아님(승인 완화 — 골든·허용오차 게이트가 판정, 비트 게이트 진단용).
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
            // evict-first — 한 번 읽는 가중치가 L2를 오염시키지 않게(스트리밍).
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
// [P11] 전문가-우선 슬롯 순열 — 같은 전문가의 슬롯을 연속 배치. 목적:
// 전문가 GEMV 블록의 실행 순서를 전문가 단위로 묶어 같은 가중치 행을 L2에서
// 재사용(토큰-우선 순서는 매 슬롯이 전문가 집합 전체를 재판독 — t=128 프리필
// gemv 47%의 원인). 1블록·스레드 병렬: 카운트(원자) → 접두합(스레드 0) →
// 스캐터(커서 원자). 슬롯 내부 순서는 비결정이나 **슬롯별 산술이 독립**이라
// 결과는 순서 무관(비트 동일). n_exp ≤ 1024 계약(호스트 가드).
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
            goff[i] = off[i]; // 연속 레이아웃 기저(그룹 커널 소비)
        }
    }
    __syncthreads();
    for (int j = tid; j < nslots; j += nt) {
        const unsigned e = (unsigned)idx[j];
        const unsigned p = atomicAdd(&off[e], 1u);
        gslot[p] = (unsigned)j;
    }
}

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

// [A9 2026-10-10] t행 GEMV(split g128) — 가중 판독 1회를 t행이 공유.
// 행별 산술은 w4a16_gemv_g128(gemv_row_body)과 **비트동일**(레인 i는
// l, l+64,… 오름차순 f32 누산 → tree64; 행별로 같은 시퀀스). t≤8
// (호스트 BATCH_DEC_MAX 미러). 배치 디코드 전용.
// [A9-fix] t는 템플릿 상수(T)로 받는다 — 런타임 t면 acc[r]이 로컬 스필
// (실측 0.125ms vs 0.032ms, 4×). 언롤 16 + q·x 포인터 진행(1행판 동형).
// [A9 2026-10-10] t행 GEMV(split g128) — 가중 판독 1회를 t행이 공유.
// 행별 산술은 w4a16_gemv_g128(gemv_row_body)과 **비트동일**(레인 i는
// l, l+64,… 오름차순 f32 누산 → tree64; 행별로 같은 시퀀스). t≤8
// (호스트 BATCH_DEC_MAX 미러). 배치 디코드 전용.
// [A9-fix1] t는 템플릿 상수(T) — 런타임 t면 acc[r] 로컬 스필(실측 4×).
// [A9-fix2] 블록당 GEMV_TR행 — x(t×k)를 블록 내 재사용(L1)해 행당 x
// L1/L2 재판독을 ÷GEMV_TR. 실측 t=4: 0.114 → 아래 수치.
#define GEMV_TR 8

// [R14 2026-10-10] t-커널 스위치 매크로 — t 1..8 케이스 나열을 한 줄로.
// (인스턴스는 종전과 동일 — 코드젠 불변.)
#define GEMV_T_CASE(N) case N: gemv_g128_t_body<N>(q, s, x, out, n, k, o0, red, sc); break;
#define GEMV_T_CASE_BF(N) \
    case N: gemv_bf16_t_body<N>(w, x, out, n, k, o0, red); break;

// red/sc는 extern 커널에서 1회 할당 후 전달(템플릿 인라인 시 인스턴스별
// 중복 할당 — TR=8에서 8×96KB > 48KB ptxas 한계로 실측).
template <int T>
__device__ __forceinline__ void gemv_g128_t_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o0,
    double (*red)[G4_LANES], float (*sc)[G4_SCMAX])
{
    const int l = threadIdx.x & (G4_LANES - 1);
    const int g = threadIdx.x >> 6; // 행 그룹 0..GEMV_TR-1
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 7;
    if (live) {
        for (int gg = l; gg < kg; gg += G4_LANES) {
            sc[g][gg] = ld_scale<false>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    // [A9-fix3] T3-2(1행판) 구조 그대로: 128폭 청크·니블 쌍·float2 x —
    // 행별 누산 시퀀스가 w4a16_gemv_g128(T3-2)과 동일(비트동일).
    const unsigned* qrow = q + (size_t)o * k8;
    const int jn = k >> 7;
    const int sh = 8 * (l & 3);
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    const unsigned* qp = qrow + (l >> 2);
    const float2* xp[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        xp[r] = reinterpret_cast<const float2*>(x) + (size_t)r * (k >> 1) + l;
    }
#pragma unroll 32
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = __ldcs(qp);
        qp += 16;
        const unsigned byte = (qw >> sh) & 0xFFu;
        const float scv = sc[g][jj];
        const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float2 xv = *xp[r];
            // T3-2와 동일: 단일 누산 체인(쌍 순차) — 이 분리가 비트동일 조건.
            acc[r] = fmaf(w0, xv.x, acc[r]);
            acc[r] = fmaf(w1, xv.y, acc[r]);
            xp[r] += 64;
        }
    }
#pragma unroll
    for (int rr = 0; rr < T; ++rr) {
        // 행당 64레인 tree — 1행판과 동일 구조(누산 시퀀스도 동일).
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

extern "C" __global__ void w4a16_gemv_g128_t(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x,   // [t][k] f32 (cast_x32 산출값)
    float* __restrict__ out,       // [t][n]
    int n, int k, int t)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][G4_LANES];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE(1) // gemv_g128_t_body<1>
        GEMV_T_CASE(2) // gemv_g128_t_body<2>
        GEMV_T_CASE(3) // gemv_g128_t_body<3>
        GEMV_T_CASE(4) // gemv_g128_t_body<4>
        GEMV_T_CASE(5) // gemv_g128_t_body<5>
        GEMV_T_CASE(6) // gemv_g128_t_body<6>
        GEMV_T_CASE(7) // gemv_g128_t_body<7>
        GEMV_T_CASE(8) // gemv_g128_t_body<8>
        default: break; // 호스트 계약 밖(t≤8)
    }
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
#define MMA_M 32
#define MMA_N 64
#define GRP_M 64
#define GRP_N 32
#define MMA_KC 32   // [B1/B3 실험] 64→32: smem 절반 → 점유 2배(지연 노출 완화)

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

extern "C" __global__ void __launch_bounds__(256, 6) w4a16_gemm_bf16_mma(
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
    const int mt = (warp >> 2) * 16;      // [B1/B3] 2 m워프 × 16 = M32
    const int ntw = (warp & 3) * 16;      // 4 n워프 × 16 = N64(워프 16×16)
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

// ── T1(2026-10-09): split(g128·f16) mma GEMM — int4 디퀀트→f16→mma ──
// ncu: split GEMM은 L1/명령 바운드(DRAM 3.7%·compute 46%) → mma 여지 큼.
// [T3 실측 2026-10-10] t<16 강제(배치 디코드 시험): 실효 가중 판독 ~300GB/s
// (27B pp512 656ms ≈ m타일 16패스 × 12.16GB) — 디코드 t-GEMV(380GB/s)에
// 열세라 배치는 t-GEMV 유지. 이 스테이징(스칼라 디퀀트+STS)이 프리필 지배
// 비용 — 디퀀트 벡터화(PRMT/half2)로 판독률 역전 시 양쪽 재론.
// A=xh(f16 — split 경로가 이미 f2h 캐스트 제공, 계약과 동일 값), B=디퀀트 f16.
// 디퀀트는 marlin식 마법 상수: f16(1024+n) = 0x6400|n (n<16이 mantissa 하위
// 비트에 정확히 더해짐) → w=(n−8)·s = (1024+n)·s − 1032·s = hfma2 1회.
// 니블 순서는 __byte_perm으로 (e0,e1),(e2,e3) 정렬(8원소/u32당 ~8 op).
// 타일·ldmatrix는 T2와 동일(M32×N64, k청크 32 — B1/B3).
extern "C" __global__ void __launch_bounds__(256, 6) w4a16_gemm_g128_mma(
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
    // [B1/B3 2026-10-10] 워프 타일 16×16(M32×N64, 8워프 = 2m×4n) — ncu:
    // ALU 57.7%(디퀀트)·L2 86%(x f32 재판독) 이중 병목. N64로 x 재판독 절반.
    const int mt = (warp >> 2) * 16;
    const int ntw = (warp & 3) * 16;
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
    }
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

// ── [P11] 그룹 mma GEMM(g32·bf16 스케일) — MoE 프리필 전문가 묶음 ──
// 블록 = (전문가, n타일). 타일 GRP_M64×GRP_N32(워프 4m×2n 16×16, B1/B3 —
// 슬롯행 64 재사용으로 W 디퀀트 절반). M = 그 전문가의 슬롯 수(가변 — m타일 루프),
// A = 슬롯→토큰 매핑으로 모은 x 행(f16 스테이징), B = g32 디퀀트(f16).
// 프래그먼트·타일 계약은 T1과 동일(명시 판독 — ldmatrix 전치 불일치 회피).
// 판정은 토큰 골든(계약 완화 — 허용오차 등급).
extern "C" __global__ void w4a16_gemm_g32_mma_grp(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const float* __restrict__ x, int xstride, int sp,
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
    const int mt = (warp >> 1) * 16; // [B1/B3] 4 m워프 × 16 = M64
    const int ntw = (warp & 1) * 16; // 2 n워프 × 16 = N32(워프 16×16)
    const int k8 = k >> 3;
    const int kg = k >> 5; // g32
    __shared__ unsigned short xs[GRP_M][MMA_KC + 8];
    __shared__ unsigned short ws[GRP_N][MMA_KC + 8];
    const unsigned mtiles = (cnt + GRP_M - 1) / GRP_M;
    for (unsigned mtile = 0; mtile < mtiles; ++mtile) {
        const unsigned rbase = mtile * GRP_M;
        float c[2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        for (int k0 = 0; k0 < k; k0 += MMA_KC) {
            __syncthreads(); // 이전 mma 완료(버퍼 재사용) + 스테이징 가시화
            // A 스테이징 — 슬롯 r의 x 행(슬롯→토큰: sl/sp·xstride), f16.
            for (int ee = tid; ee < GRP_M * MMA_KC / 4; ee += 256) {
                const int r = ee / (MMA_KC / 4);
                const int c4 = ee % (MMA_KC / 4);
                const int gi = k0 + c4 * 4;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (rbase + r < cnt) {
                    const unsigned sl = gs[rbase + r];
                    const size_t xoff =
                        (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
                    if (gi + 3 < k) {
                        v = *reinterpret_cast<const float4*>(&x[xoff + gi]);
                    } else {
                        float t4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        for (int j = 0; j < 4; ++j) {
                            if (gi + j < k) {
                                t4[j] = x[xoff + gi + j];
                            }
                        }
                        v = make_float4(t4[0], t4[1], t4[2], t4[3]);
                    }
                }
                xs[r][c4 * 4 + 0] = __half_as_ushort(__float2half_rn(v.x));
                xs[r][c4 * 4 + 1] = __half_as_ushort(__float2half_rn(v.y));
                xs[r][c4 * 4 + 2] = __half_as_ushort(__float2half_rn(v.z));
                xs[r][c4 * 4 + 3] = __half_as_ushort(__float2half_rn(v.w));
            }
            // B 스테이징 — g32 디퀀트(스케일 bf16 → f32은 비트 상위 시프트).
            for (int ee = tid; ee < GRP_N * MMA_KC / 8; ee += 256) {
                const int r = ee / (MMA_KC / 8);
                const int c8 = ee % (MMA_KC / 8);
                const int gi = k0 + c8 * 8;
                const bool live = (n0 + r < n) && (gi + 7 < k);
                unsigned short scb = 0;
                if (n0 + r < n) {
                    scb = s[(size_t)(n0 + r) * kg + (gi >> 5)];
                }
                const float scf = __uint_as_float((unsigned)scb << 16);
                const unsigned qw = live ? q[(size_t)(n0 + r) * k8 + (gi >> 3)] : 0u;
                if (live) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int nib = (int)((qw >> (4 * j)) & 0xFu) - 8;
                        ws[r][c8 * 8 + j] =
                            __half_as_ushort(__float2half_rn((float)nib * scf));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        ws[r][c8 * 8 + j] = 0;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int ks = 0; ks < MMA_KC / 16; ++ks) {
                const int kb = ks * 16;
                const unsigned a0 = pk2bf(&xs[mt + g][kb + 2 * tt]);
                const unsigned a1 = pk2bf(&xs[mt + g + 8][kb + 2 * tt]);
                const unsigned a2 = pk2bf(&xs[mt + g][kb + 2 * tt + 8]);
                const unsigned a3 = pk2bf(&xs[mt + g + 8][kb + 2 * tt + 8]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int nb = ntw + nt * 8;
                    const unsigned b0 = pk2bf(&ws[nb + g][kb + 2 * tt]);
                    const unsigned b1 = pk2bf(&ws[nb + g][kb + 2 * tt + 8]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[nt][0]), "+f"(c[nt][1]), "+f"(c[nt][2]), "+f"(c[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
        // 에필로그 — 원본 슬롯 행으로 기록(범위 가드).
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
