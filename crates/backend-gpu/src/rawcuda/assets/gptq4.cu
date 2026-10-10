#include <cuda_fp16.h>

// ── W4A16(GPTQ4·g128·sym) GEMV/GEMM CUDA — 비트 계약: core dot_row_w4a16_lane ──
// [R5 2026-10-10] 분할 — 플레인 bf16·MoE 보조 커널(top-k·누적·게이트)은
// moe.cu, 공용 device 헬퍼/define은 g4_common.cuh. 이 파일은 분할(g128/g32)
// 계열 전부(FFMA GEMV/GEMM·mma·전문가 배치·cast·copy)를 소유한다.
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
#include "g4_common.cuh"

// ── [R3 2026-10-10] define 소유권 표 (값 변경 시 영향 범위 — 비트/토큰 재판정) ──
// 커널별 전용은 접두 분리(BMMA_* 전례 — MMA_M/N 공유 사고 2회).
// [R5 2026-10-10] 공유 항목(G4_LANES·GEMV_TR·MMA_KC·b2f·pk2bf·shfl_down_f64·
// cp_async16/4)은 g4_common.cuh 단일 정의. 이 파일 소유:
//  G4_ROWS — FFMA GEMM 행/블록.  G4_SCMAX — t=1 GEMV(k 상한).
//  G4_GTMAX/G4_KC — FFMA GEMM 패밀리.
//  MMA_M/MMA_N — g128 mma GEMM(MMA_KC는 공용 헤더).
//  GRP_M/GRP_N — g32 그룹 mma.
// moe.cu 소유: G4_TMAX(배치 GEMV·플레인 v1 GEMM — 재정의 통합) ·
//  G4_TMAX2/G4_ROWS_B/G4_KC_B(플레인 v3) · BMMA_M/BMMA_N(bf16 mma) ·
//  MOE_TK_WARPS/MOE_TK256_N/MOE_TK256_C(MoE topk_t/256).

// 활성 f32 [n] → h2f(f2h(v)) f32 [n] — t=1 GEMV 입력의 사전 변환.
// 계약: 커널 안에서 h2f(xt[i])하던 값을 밖에서 한 번 계산해 두는 것과 동일
// (f2h→h2f 왕복이 비트를 보존). 반드시 f2h/h2f와 동형 수정.
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
            out16[i] = h; // f16 미러 — mma GEMM A(대역 절반, 비트 동일).
        }
    }
}

// [R5 2026-10-10] G4_LANES는 g4_common.cuh(플레인 공용)로 이동. 이 파일 전용:

#define G4_ROWS 8

// t=1 전용 GEMV — 한 블록 = 한 행(64레인). 구 GEMM 커널은 블록이 8행을
// 순차 처리해 메모리 지연을 못 숨기고(실측 GPU 88ms/토큰), 여기서는 격자를
// n으로 늘려 지연을 숨긴다. 스케일은 smem f32로 선변환, 활성 x는
// w4a16_cast_x32가 만든 f32(h2f 왕복). 산술 순서는 구 커널과 동일:
//   w = (nib-8) as f32 * sc[g];  acc += w * x[i];   (mul·add 분리, 레인 l은
//   i=l,l+64,… 오름차순 → f64 tree64). k ≤ 128*G4_SCMAX 계약.
#define G4_SCMAX 256

// ── 그룹·스케일 dtype 변형(W4-1: 35B-A3B = g32·BF16 스케일) ──
// [R5] b2f는 g4_common.cuh(플레인 공용)로 이동.
// 스케일 로드 — BF16이면 b2f, 아니면 h2f(기존 g128·f16 경로와 동일 값).
template <bool BF16>
__device__ __forceinline__ float ld_scale(const unsigned short* __restrict__ p, int i) {
    return BF16 ? b2f(p[i]) : h2f(p[i]);
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

// [B-1/B-2 2026-10-10] 전문가 gate+up+ew 융합 — 1블록이 (행, 슬롯)의 gate·up
// 두 행을 계산해 act = silu(gate)·up을 직접 기록. ew 커널·게이트/업 중간
// 버퍼 왕복·x 재판독(게이트+업 각 1회 → 1회) 제거. 산술: 행별은
// gemv_row_body 그대로(비트동일), silu·mul은 ew 커널과 동일식(__expf,
// f32 div, -fmad=false) → act 비트동일(골든 판정). t=1 상주 전용.
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
    gemv_row_body<5, true>((const unsigned*)eg[0], (const unsigned short*)eg[1],
                           x + xoff, slot, o, k);
    __syncthreads();
    const float g = slot[o];
    gemv_row_body<5, true>((const unsigned*)eu[0], (const unsigned short*)eu[1],
                           x + xoff, slot, o, k);
    __syncthreads();
    const float u = slot[o];
    if (threadIdx.x == 0) {
        const float e = __expf(-g);
        slot[o] = (g / (1.0f + e)) * u;
    }
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
// [A9-fix1] t는 템플릿 상수(T) — 런타임 t면 acc[r] 로컬 스필(실측 4×).


// [R14 2026-10-10] t-커널 스위치 매크로 — t 1..8 케이스 나열을 한 줄로.
// (인스턴스는 종전과 동일 — 코드젠 불변. BF 판은 moe.cu GEMV_T_CASE_BF.)
#define GEMV_T_CASE(N) case N: gemv_g128_t_body<N>(q, s, x, out, n, k, o0, red, sc); break;


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
// [R3] G4_ROWS/G4_LANES 중복 정의 제거 — 상단(t=1 GEMV) 정의(8/64) 공유.
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



// [A-1 2026-10-10] D2D 복사 — 스펙 롤백용. cuMemcpyDtoDAsync가 드라이버에서
// 세그폴트(초유일 사용 경로, 실측 백트레이스)라 커널로 대체.
extern "C" __global__ void w4a16_copy(const float4* __restrict__ src,
                                      float4* __restrict__ dst, long n4) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n4) {
        dst[i] = src[i];
    }
}

// [marlin-A4 2026-10-10] 32→64: B 재판독이 m타일 수에 비례(16) — M64면
// 절반. 워프당 32행(누산 c[2][4][4]), smem A 2×64×40×2=10KB.
#define MMA_M 64
// [marlin-A2 2026-10-10] 64→128: A 재판독이 n타일 수에 비례 — ncu 실측 L2
// 4.12GB/런치(= 예상 2.1GB의 2×, L2 90% 바운드). N128이면 n타일 절반 →
// A 트래픽 절반. (marlin이 thread_n_blocks=16=N256을 쓰는 이유와 동형.)
// [FLA-11 기각 2026-10-10] N256 재시도 = 27B 프리필 348→404ms(+17%) —
// 누산 64개(c[2][8][4])의 레지스터 압박으로 점유 3→2블록, A 절반 이득을
// 압도(bf16_mma N128 기각과 동형: 이 계열은 L2보다 점유 민감). 되돌림.
#define MMA_N 128

#define GRP_M 64
#define GRP_N 32

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


// [marlin-B 2026-10-10] 패킹 u32에서 (2tt, 2tt+1) 니블 2개 → f16 쌍(u32).
// 종전 smem f16 디큐트 `__float2half_rn((float)(nib-8)*scf)`와 **비트동일**:
// hsub2(1024+n)-(1024+8) = n-8 정확(f16 정수) → f32 곱 → RNE. 스테이징이
// 순수 복사가 되고 smem B가 1/4(L1·smem 압력↓).
__device__ __forceinline__ unsigned deq_pair_b(unsigned qw, int tt, float sc) {
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

extern "C" __global__ void __launch_bounds__(256, 3) w4a16_gemm_g128_mma(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/128] f16 스케일
    const unsigned short* __restrict__ x,  // [t][k] f16 — split 경로 계약
                                           // (cast_x32/norm이 h2f(f2h)와 함께
                                           // 기록한 미러). 종전 f32 재판독이
                                           // L2의 80%(ncu 실측) → 16B 복사
                                           // 스테이징으로 대역 절반(marlin A).
    float* __restrict__ out, int n, int k, int t)
{
    // [지연 은닉] 더블 버퍼 — ncu: Compute 27%·점유 35%·No Eligible 77%(지연
    // 바운드). 스테이징(k+1)과 mma(k)를 겹친다(버퍼 2×27.6KB = 55KB < 100KB).
    __shared__ __align__(16) unsigned short xs[2][MMA_M][MMA_KC + 8];
    // [marlin-B] B는 패킹 int4 그대로(디큐트는 레지스터, mma 전) + 청크 스케일.
    __shared__ unsigned wsp[2][MMA_N][MMA_KC / 8];
    __shared__ float ssm[2][MMA_N];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * MMA_M;
    const int n0 = blockIdx.y * MMA_N;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    // [B1/B3 2026-10-10] 워프 타일 16×16(M32×N64, 8워프 = 2m×4n) — ncu:
    // ALU 57.7%(디퀀트)·L2 86%(x f32 재판독) 이중 병목. N64로 x 재판독 절반.
    const int mt = (warp >> 2) * 32; // [marlin-A4] M64: 2 m워프 × 32 = M64
    const int ntw = (warp & 3) * 32; // [marlin-A2] N128: 워프당 32열
    const int k8 = k >> 3;
    const int kg = k >> 7;
    float c[2][4][4]; // [marlin-A4] M64: 2 m프래그먼트 × 4 n프래그먼트
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        ((float*)c)[i] = 0.0f;
    }
    int cur = 0;
    int k0 = 0;
    // 초기 스테이징(청크 0) — f16 16B 로드 → 그대로 복사(변환 없음).
    for (int e = tid; e < MMA_M * MMA_KC / 8; e += 256) {
        const int r = e / (MMA_KC / 8);
        const int c8 = e % (MMA_KC / 8);
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
    for (int e = tid; e < MMA_N * (MMA_KC / 8); e += 256) {
        const int r = e / (MMA_KC / 8);
        const int c8 = e % (MMA_KC / 8);
        const int gi = c8 * 8;
        const bool live = (n0 + r < n) && (gi + 7 < k);
        // 패딩 니블 = 8(값 0) — 0u면 (0-8)*sc ≠ 0이 된다.
        wsp[0][r][c8] = live ? q[(size_t)(n0 + r) * k8 + (gi >> 3)] : 0x88888888u;
    }
    for (int e = tid; e < MMA_N; e += 256) {
        unsigned short scb = 0;
        if (n0 + e < n) {
            scb = s[(size_t)(n0 + e) * kg];
        }
        ssm[0][e] = __half2float(*reinterpret_cast<const __half*>(&scb));
    }
    // 초기 스테이지 완료 대기 + 가시화(이후 루프가 관리).
    // [marlin-C4 기각 2026-10-10] 4스테이지 확장 = gemm 236→277ms —
    // smem 30.5KB/블록으로 L1 캐시 축소(프래그먼트·스테이징 L1 의존).
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (; k0 < k; k0 += MMA_KC, cur ^= 1) {
        // 다음 청크 스테이징을 cp.async로 발사 — 아래 mma와 겹친다.
        const int kn = k0 + MMA_KC;
        if (kn < k) {
            for (int e = tid; e < MMA_M * MMA_KC / 8; e += 256) {
                const int r = e / (MMA_KC / 8);
                const int c8 = e % (MMA_KC / 8);
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
            for (int e = tid; e < MMA_N * (MMA_KC / 8); e += 256) {
                const int r = e / (MMA_KC / 8);
                const int c8 = e % (MMA_KC / 8);
                const int gi = kn + c8 * 8;
                const bool live = (n0 + r < n) && (gi + 7 < k);
                if (live) {
                    cp_async4(&wsp[cur ^ 1][r][c8], &q[(size_t)(n0 + r) * k8 + (gi >> 3)]);
                } else {
                    wsp[cur ^ 1][r][c8] = 0x88888888u;
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
        for (int ks = 0; ks < MMA_KC / 16; ++ks) {
            const int kb = ks * 16;
            // 프래그먼트 = 스모크 검증 방식(명시 판독). ldmatrix는 전치/분배
            // 불일치로 계통 오차(실측 maxabs 3e10) — 정확성 우선으로 되돌림.
            // [marlin-A4] M64: mf 2(16행 프래그먼트 2개) × nt 4.
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
                    // [marlin-B] 레지스터 디큐트(비트동일 — deq_pair_b 주석).
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
        // 다음 스테이지 완료 대기 + 가시화(다음 반복 compute 전).
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
    // [marlin-A5 2026-10-10] A는 f16 — 호스트가 cast로 기록한 미러(xn/act의
    // f2h). 종전 f32 재판독(콜당 ~536MB, 16 n타일 재판독) → 절반. 값 동일
    // (커널 내부 __float2half_rn과 같은 반올림).
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int mt = (warp >> 1) * 16; // [B1/B3] 4 m워프 × 16 = M64
    const int ntw = (warp & 1) * 16; // 2 n워프 × 16 = N32(워프 16×16)
    const int k8 = k >> 3;
    const int kg = k >> 5; // g32
    // [marlin-C3 2026-10-10] A·B 완전 더블버퍼 + cp.async — 청크 직렬
    // (스테이징→sync→mma→sync ×64청크)이 지배 병목(ncu 배리어 스톨).
    // A는 cp.async, B는 디큐트 ALU(다음 청크를 mma와 겹쳐 발사).
    // [marlin-B 재기각] 파이프라인 확보 후에도 레지스터 디큐트 = 88.6→90.5ms
    // (프래그먼트 디큐트 체인이 mma 의존 사슬에 직렬). f16 smem 디큐트 유지.
    // [FLA-5 기각 2026-10-10] B 스테이징 2단 분리(q/s cp.async 선행 + smem
    // 디큐트, 전역 지연 은닉) — 이득 0(P512 158 vs 159ms) + 600토큰에서
    // illegal access(경계 결함). 되돌림. g32 시도 누적 6회 전부 실측 기각
    // (레지스터 디큐트×2·cp.async·KC·NSPLIT·FLA-5) — SASS상 디큐트는 이미
    // 컴파일러 최적(PRMT 팩·STS.64).
    __shared__ __align__(16) unsigned short xs[2][GRP_M][MMA_KC + 8];
    __shared__ unsigned short ws[2][GRP_N][MMA_KC + 8];
    const unsigned mtiles = (cnt + GRP_M - 1) / GRP_M;
    for (unsigned mtile = 0; mtile < mtiles; ++mtile) {
        const unsigned rbase = mtile * GRP_M;
        float c[2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        // 스테이지 발사(스테이지 st, 청크 kn) — A는 cp.async, B는 디큐트 ALU.
        auto stage_ab = [&](int st, int kn) {
            for (int ee = tid; ee < GRP_M * MMA_KC / 8; ee += 256) {
                const int r = ee / (MMA_KC / 8);
                const int c8 = ee % (MMA_KC / 8);
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
            for (int ee = tid; ee < GRP_N * MMA_KC / 8; ee += 256) {
                const int r = ee / (MMA_KC / 8);
                const int c8 = ee % (MMA_KC / 8);
                const int gi = kn + c8 * 8;
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
                        ws[st][r][c8 * 8 + j] =
                            __half_as_ushort(__float2half_rn((float)nib * scf));
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
        for (int k0 = 0; k0 < k; k0 += MMA_KC, cur ^= 1) {
            // 다음 청크를 발사(mma와 겹침) — A cp.async + B 디큐트.
            const int kn = k0 + MMA_KC;
            if (kn < k) {
                stage_ab(cur ^ 1, kn);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < MMA_KC / 16; ++ks) {
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
            // [경합 수정 2026-10-10] 이 스테이지 소비 완료 배리어 — 없으면 다음
            // 반복의 stage_ab(B의 **동기 smem 저장**)이 타 워프의 mma 판독과
            // 경합(스모크 간헐 실패 실측).
            __syncthreads();
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
