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

// [A2-float4 2026-10-11] t=1 로우 커널 float4판 — **[256폭 청크·64레인 유지]**
// (종전 128폭·2원소/레인 → 256폭·4원소/레인: 레인 l = k 4l..4l+3/256청크).
// 블록·grid 구조는 종전과 동일(행=블록 64스레드·grid n) — 레인 수 불변이라
// 병렬도 보존, x float4(1판독/4원소)·웨이트 u16(코얼레싱)만 회수.
// 수치 클래스 — t=1 PPL(eval) 직접 판정(비트 계약 아님).
// 기각(2026-10-11): 32레인/행·2행/블록(128폭) 변형 = tg −3.5%(49.5 t/s) —
// 행당 스레드 반감으로 지연 은닉 손실. 64레인 유지가 정답이었다.
template <int SHIFT>
__device__ __forceinline__ int sidx256(int l, int jj) {
    // (4l + 256j) >> SHIFT — g128: 2j + (l>>5), g32: 8j + (l>>3).
    return (jj << (8 - SHIFT)) + (l >> (SHIFT - 2));
}

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_row_body4(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int o, int k)
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
    const int jn = k >> 8; // 256폭 청크(전 모델 k는 256 배수 — 128 배수 상위)
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    const float4* xp = reinterpret_cast<const float4*>(x) + l;
    float acc = 0.0f;
#pragma unroll 16
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = (unsigned)qh[jj * 64];
        const float scv = sc[sidx256<SHIFT>(l, jj)];
        const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
        const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
        const float w3 = fmaf((float)(qw >> 12), scv, -8.0f * scv);
        const float4 xv = *xp;
        acc = fmaf(w0, xv.x, acc);
        acc = fmaf(w1, xv.y, acc);
        acc = fmaf(w2, xv.z, acc);
        acc = fmaf(w3, xv.w, acc);
        xp += 64;
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

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_body4(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    // [gemma4 2026-10-14] float4 본체는 256폭 청크 전제 — k가 256 배수가
    // 아니면 꼬리가 누락된다(실측 0.158: gemma4 dense down_proj k=2176 =
    // 8.5×256, w4a16-gemv 게이트 적발). 해당 형상은 **스칼라 본체**로
    // (128폭 청크 — k%128==0이면 정확, 전문가 경로가 쓰는 검증된 경로).
    // 수치 클래스: 양쪽 모두 FMA(t=1 GEMV = 비트 계약 아님, 진단 게이트).
    if ((k & 255) != 0) {
        gemv_row_body<SHIFT, BF16>(q, s, x, out, o, k);
        return;
    }
    gemv_row_body4<SHIFT, BF16>(q, s, x, out, o, k);
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

// [gemma4 P2 2026-10-14] 전문가 gate+up+활성 융합 — gelu_tanh판.
// 구조·행별 산술은 w4a16_gemv_experts_glu와 동일(gemv_row_body) — 활성만
// silu → gelu_pytorch_tanh(0.5x(1+tanh(√(2/π)(x+0.044715x³)))). tanhf는
// 트랜센던털 등급(eval 판정 — ew_gelu와 동일 클래스). gemma4 전용(모델별 선택).
extern "C" __global__ void w4a16_gemv_experts_glu_gelu(
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
        const float x3 = g * g * g;
        slot[o] = (0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * x3)))) * u;
    }
}

// t=1 GEMV 래퍼 — g128·f16(27B) / g32·bf16(35B 전문가).
// [P11] 전문가-우선 슬롯 순열 — 같은 전문가의 슬롯을 연속 배치. 목적:
// 전문가 GEMV 블록의 실행 순서를 전문가 단위로 묶어 같은 가중치 행을 L2에서
// 재사용(토큰-우선 순서는 매 슬롯이 전문가 집합 전체를 재판독 — t=128 프리필
// gemv 47%의 원인). 1블록·스레드 병렬: 카운트(원자 — 합은 결정론) → 접두합
// (스레드 0) → **결정론 스캐터**. n_exp ≤ 1024 계약(호스트 가드).
// [C4 2026-10-15] 구 스캐터는 원자 커서(atomicAdd(&off[e]))라 **그룹 내
// 슬롯 순서가 원자 획득 순을 따라 런마다 흔들렸다**(실측 1e-7급 노이즈 —
// TreeWY 검증 중 관측·실재 확인). 대체 = 슬롯 j의 그룹 내 순위를 j'<j 중
// 동일 전문가 수(슬롯 순서 프리픽스)로 직접 계산 — 어떤 실행에서도 그룹 내
// 오름차순 j로 고정(재현성 = 골든·그래프 캡처·지문의 전제). 비용 = O(nslots²/2)
// 스캔이나 ns ≤ 4096(청크512×top8)에서 전체 예산 <0.2%, 소형(ns≤1024)은 µs급.
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
    // [P2 2026-10-16] O(n²/nt) 선행 검색 → O(n) 단일 스레드 순회 — 순위 =
    // "같은 전문가가 이전에 몇 번 나왔나"는 앞 슬롯을 전부 훑지 않고 실행
    // 카운터(shared cnt 재사용 — 프리픽스 후 0으로 리셋)로 직접 추적.
    // ns≤4096 → 4096 회 = 수 µs(기존 8.4M 비교)·완전 결정론(순회 순서 고정).
    // 병렬화가 필요해지면 워프별 프리픽스 2패스로 확장(결정론 유지 조건).
    if (tid == 0) {
        for (int i = 0; i < n_exp; ++i) {
            cnt[i] = 0; // 실행 카운터로 전환(합은 이미 gcnt에 기록됨)
        }
        for (int j = 0; j < nslots; ++j) {
            const int e = idx[j];
            gslot[off[e] + cnt[e]] = (unsigned)j;
            cnt[e] += 1;
        }
    }
}

extern "C" __global__ void w4a16_gemv_g128(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body4<7, false>(q, s, x, out, n, k); // [A2-float4] 2행/블록
}
extern "C" __global__ void w4a16_gemv_g32_bf16(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body4<5, true>(q, s, x, out, n, k); // [A2-float4] 2행/블록
}

// [S2 2026-10-11] 다중 세그먼트 t=1 GEMV(split) — 같은 x를 공유하는 소형
// 선형 여럿을 1런치로(런치 꼬리 제거 — moe.cu w4a16_gemv_multi의 split 판).
// tab[seg] = (q, s, out, n, k) u64×5 — 호스트가 업로드 후 1회 작성(그래프
// 캡처 정적 계약: 리플레이 간 내용 불변). 행별 산술은 w4a16_gemv_g128과
// **비트동일**(gemv_row_body 그대로 — 행 독립이라 세그먼트 결합 무영향).
extern "C" __global__ void w4a16_gemv_g128_multi(
    const unsigned long long* __restrict__ tab, // [count][5]: q, s, out, n, k
    const float* __restrict__ x,                // 공통 x [k] f32(h2f 왕복)
    int count)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int s = 0; s < count; ++s) {
        const int ns = (int)tab[s * 5 + 3];
        if (blk < base + ns) {
            seg = s;
            break;
        }
        base += ns;
    }
    const int n = (int)tab[seg * 5 + 3];
    const int o = blk - base;
    if (o < 0 || o >= n) {
        return;
    }
    const int k = (int)tab[seg * 5 + 4];
    gemv_row_body<7, false>((const unsigned*)(size_t)tab[seg * 5 + 0],
                            (const unsigned short*)(size_t)tab[seg * 5 + 1],
                            x, (float*)(size_t)tab[seg * 5 + 2], o, k);
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
// [A-float4 2026-10-11] 4-연속-k 레인 재매핑(수치 변경 — eval 판정 클래스):
// 레인 l(0..31) = k 4l..4l+3/128청크, 웨이트 u16(바이트 2l,2l+1 = 니블 4개)
// 완전 코얼레싱, x float4(16B 정렬), FMA 4연쇄. 레인당 로드 명령 절반(진단
// 상한 +24%의 회수 시도). 행당 32레인 f64 트리(∪l,l+16 → 셔플 8..1).
// 블록 = 32×GEMV_TR(256스레드·행/블록 = GEMV_TR — 호스트 grid 불변).
template <int T>
__device__ __forceinline__ void gemv_g128_t_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o0,
    double (*red)[32], float (*sc)[G4_SCMAX])
{
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5; // 행 그룹 0..GEMV_TR-1
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 7;
    if (live) {
        for (int gg = l; gg < kg; gg += 32) {
            sc[g][gg] = ld_scale<false>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    // 청크 j의 레인별 u16 = 바이트 64j+2l,64j+2l+1 — 워프 전량 코얼레싱.
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    // [T16-reg 2026-10-15] 행별 x 포인터 배열(xp[T]) 제거 — T=16에서 레지스터
    // 32개를 추가로 먹어 스필(STACK 256B·점유 4블록, ncu: Block Limit
    // Registers=4 vs SM 24)을 유발했다. 단일 베이스 + (r·k4) 루프 불변 오프셋
    // 로드로 전환 — 적재 값·FMA 순서 불변(행별 산술 비트동일).
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    const int k4 = k >> 2;
#pragma unroll 32
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = (unsigned)qh[jj * 32];
        const float scv = sc[g][jj];
        const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
        const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
        const float w3 = fmaf((float)(qw >> 12), scv, -8.0f * scv);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float4 xv = xb[(size_t)r * k4 + jj * 32];
            acc[r] = fmaf(w0, xv.x, acc[r]);
            acc[r] = fmaf(w1, xv.y, acc[r]);
            acc[r] = fmaf(w2, xv.z, acc[r]);
            acc[r] = fmaf(w3, xv.w, acc[r]);
        }
    }
#pragma unroll
    for (int rr = 0; rr < T; ++rr) {
        red[g][l] = (double)acc[rr];
        __syncthreads();
        // [A-float4] 전 워프 참여 셔플(마스크 0xffffffff 계약 — l≥16도 참여),
        // 합산·기록은 하위 레인만(첫 결합 l<16, 셔플 8..1).
        double v = (l < 16) ? (red[g][l] + red[g][l + 16]) : 0.0;
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) {
                v += oth;
            }
        }
        if (l == 0 && live) {
            out[(size_t)rr * n + o] = (float)v;
        }
        __syncthreads();
    }
}

// [§np4 실측 기각 2026-10-11] launch_bounds(512,3)(점유 2→3블록): np4 103→
// 98.9 tok/s(스필 우세). ncu: t=5에서 점유 2블록(64 regs)·long_scoreboard
// 12.63·barrier 4.49 — 지연은 트리 배리어(계약)+x float2 패턴에 내재.
// 대안(Gemm 경로 21 tok/s·lb) 전부 실측 기각 — 잔여는 구조 재설계 영역.
// [S1-a 실측 기각 2026-10-11] 환원 재배치(행별 2배리어 → 상반부 슬롯 기록+
// 배리어 1, 산술 비트동일) 시도: 27B np4 109.6→107.1 tok/s(동일 세션 2×2
// fatbin A/B −2.2%)·t=4 격리 bench 0.065→0.066ms. 격리 대조(o2v=old+4KB
// smem(0.065) · o58=old maxrregcount=58(0.064) · n12=new sc축소 12KB(0.066))
// 로 smem·레지스터 가설 배제 — barrier 4.49는 임계 경로 아님(long_scoreboard
// 지배). 무변경 원복.
// [S1-b 실측 기각 2026-10-11] GEMV_TR 8→16(x L2 재판독 ÷2 가설 — np 스텝
// 스케일링 +5~6ms/토큰 근거): 27B 프리필 t=2..8이 0층부터 NaN(토큰 골든
// 붕괴). 대조 실측: t=1 체인·35B 프리필(bf16_t)·27B 스펙 검증 t=4는 정상,
// TR=12도 동일 실패(1024스레드·레지스터 정확 적합 가설 배제), T=2 단독 SASS는
// TR 스케일 상수 5개 외 동일. TR-비의존 국소화 실패(원인 미규명) — TR=8
// 원복. 가설 자체는 미검증으로 잔존.
// [m8 역할교환 mma 전면 재작성 실측 기각 2026-10-11 — 마8 축 종결] marlin
// m8 동형 설계(가중=A m16·활성=B n8·특징=M·토큰=N·x를 블록당 1회 — x 트래픽
// ÷16)를 3구성 실측: KC32 단일스테이지 50.3(−62%)·KC32 cp.async 이중버퍼
// 70.8(−47%)·KC128 94.0(−30%)·KC256 94.1(smem 58.6KB — 포화). **정합은
// 완전**(eval chunk2 NLL 1.7733·골든 4종 유지 — 활성 f16+mma 누산 클래스).
// 벽 = 워프 수: mma 판은 출력 타일이 커 워프 총수가 GEMV의 **1/16**(n/16 vs
// n) — mma의 MACs/명령 우위(≈5×)가 지연 은닉 상실을 못 메움(같은 세션의
// GEMV 행-내부 변형들 −26~−53%와 동형 패턴). **t≤8 디코드는 본 GPU/형상에서
// GEMV가 최적** — vLLM 190 격차는 타 요인(KV/스케줄/어텐션 문맥)으로 추정.
// [v2 행-내부 재설계 실측 기각 2026-10-11 — 가장 어려운 축 시도] x 로드
// 절감 설계(워프=8행·레인=k슬라이스 — x를 4~8행이 공유, 트리 셔플만)를
// 3변형 실측: ①rows=8/T≤4: T=8 케이스 acc[8][8]이 커널 레지스터 배분을
// 지배 → 스필 9756B·np4 62.1(−53%) ②rows=4/unroll8: 64reg·스필 16B(클린)
// 이나 np4 99.1(−26%) ③rows=4/unroll16·32: 94.4(−29%). **x 부재 상한
// +24%(진단)는 스케줄 변경으로 회수 불가** — 단순 1행 루프의 심층 스트리밍
// ILP가 상실분(cp.async 스테이징 기각과 동형). np4 t-GEMV는 국소 최적 —
// 남은 길은 marlin식 전면 재설계(레이아웃+ldmatrix)뿐(t=4 mma = GEMV 동률
// 실측이 그 경로의 필요조건을 시사).
// [A 진단·기각 2026-10-11] (진단)x 로드 제거(값 상수 대체) = 27B np4
// 110.8→137.0 t/s(**+24%** — x 축이 실재·상한). (시도)cp.async smem
// 스테이징(2×2048 f32·KC=2048/T 청크·프리페치 파이프): np4 80.4(프리페치
// 미겹침)→87.4(겹침) — **base 110.8 대비 대폭 역행**. 원인: 청크 경계
// 배리어가 unroll-32 ILP를 4로 축소 — x 비용은 대역이 아니라 **ILP/스케줄
// 성분**. 단계적 보류 후 —
// **[A-float4 채택 2026-10-11]** 4-연속-k 레인 재매핑(위 body): 27B np4
// 110.8→118.6(**+7.0%**)·tg/pp 불변. 수치 클래스(eval 판정): chunk4
// PPL 델타 en +0.0211%·ko +0.0000%·argmax 일치 99.904%/99.770%(발산은
// 모델 오답 위치의 ulp 플립) — eval.md 후보 표 기록. 골든(프로브·spec·
// np4 serve) 3종 유지·smoke 10/10. 초기 구현의 하프워프 셔플(마스크 UB·
// 행)은 전 워프 참여형으로 수정(위 rr 루프).
// [A1-점유 2026-10-11] 72→64reg 강제(4블록/SM — np4 27B 119.0→128.0(+7.6%),
// 스필 228/344B 감수). ncu 실측: DRAM 38%·long_scoreboard 13.7·점유 50%(3블록
// 한계) = 완전 지연 바운드 — cp.async 스테이징이 아니라 **점유**가 정답이었다.
extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g128_t(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x,   // [t][k] f32 (cast_x32 산출값)
    float* __restrict__ out,       // [t][n]
    int n, int k, int t)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32]; // [A-float4] 32레인 트리
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
        GEMV_T_CASE(9)
        GEMV_T_CASE(10)
        GEMV_T_CASE(11)
        GEMV_T_CASE(12)
        GEMV_T_CASE(13)
        GEMV_T_CASE(14)
        GEMV_T_CASE(15)
        GEMV_T_CASE(16)
        default: break; // 호스트 계약 밖(t≤16)
    }
}

template <int T>
__device__ __forceinline__ void gemv_g32_t_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o0,
    double (*red)[32], float (*sc)[G4_SCMAX])
{
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5; // 행 그룹 0..GEMV_TR-1
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    if (live) {
        for (int gg = l; gg < kg; gg += 32) {
            sc[g][gg] = ld_scale<true>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    // 청크 j의 레인별 u16 = 바이트 64j+2l,64j+2l+1 — 워프 전량 코얼레싱.
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    // [T16-reg 2026-10-15] 행별 x 포인터 배열 제거(위 g128_t_body 동형 — T=16
    // 스필/점유 원인). 단일 베이스 + (r·k4) 오프셋 — 값·순서 불변.
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    const int k4 = k >> 2;
#pragma unroll 32
    for (int jj = 0; jj < jn; ++jj) {
        const unsigned qw = (unsigned)qh[jj * 32];
        // [g32 포트] 레인당 4니블 = 한 32그룹 내부(4l..4l+3 ⊂ [32⌊l/8⌋, +32)).
        const float scv = sc[g][(l >> 3) + (jj << 2)];
        const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
        const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
        const float w3 = fmaf((float)(qw >> 12), scv, -8.0f * scv);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float4 xv = xb[(size_t)r * k4 + jj * 32];
            acc[r] = fmaf(w0, xv.x, acc[r]);
            acc[r] = fmaf(w1, xv.y, acc[r]);
            acc[r] = fmaf(w2, xv.z, acc[r]);
            acc[r] = fmaf(w3, xv.w, acc[r]);
        }
    }
#pragma unroll
    for (int rr = 0; rr < T; ++rr) {
        red[g][l] = (double)acc[rr];
        __syncthreads();
        // [A-float4] 전 워프 참여 셔플(마스크 0xffffffff 계약 — l≥16도 참여),
        // 합산·기록은 하위 레인만(첫 결합 l<16, 셔플 8..1).
        double v = (l < 16) ? (red[g][l] + red[g][l + 16]) : 0.0;
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) {
                v += oth;
            }
        }
        if (l == 0 && live) {
            out[(size_t)rr * n + o] = (float)v;
        }
        __syncthreads();
    }
}

// [gemma4 P3 2026-10-14] g32·bf16 스케일 t행 GEMV — gemv_g128_t_body의 g32 포트
// (차이 3곳: kg = k>>5 · 스케일 인덱스 (l>>3)+4j · ld_scale<true>). 행별 산술
// 순서는 g128_t와 동일 — 디코드 배치(t≤8)가 t=1 GEMV와 같은 클래스.
#define GEMV_T_CASE_G32(N) \
    case N: gemv_g32_t_body<N>(q, s, x, out, n, k, o0, red, sc); break;

extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g32_bf16_t(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE_G32(1)
        GEMV_T_CASE_G32(2)
        GEMV_T_CASE_G32(3)
        GEMV_T_CASE_G32(4)
        GEMV_T_CASE_G32(5)
        GEMV_T_CASE_G32(6)
        GEMV_T_CASE_G32(7)
        GEMV_T_CASE_G32(8)
        GEMV_T_CASE_G32(9)
        GEMV_T_CASE_G32(10)
        GEMV_T_CASE_G32(11)
        GEMV_T_CASE_G32(12)
        GEMV_T_CASE_G32(13)
        GEMV_T_CASE_G32(14)
        GEMV_T_CASE_G32(15)
        GEMV_T_CASE_G32(16)
        default: break; // 호스트 계약 밖(t≤16)
    }
}

// ── [P8 2026-10-15] 타일 루프 t행 GEMV(g32·bf16) — dense 청크 전체를 1런치로 ──
// 동기: 청크512에서 dense t행 GEMV가 GPU 예산 61.7%(프로파일 실측) — 16행
// 호출 t/16회가 **가중 행렬 전체를 매 호출 재스트림**한다. 본 커널은 블록당
// 출력행(종전과 동일 그리드 = ceil(n/GEMV_TR)×256)에서 t0 타일(16행) 루프로
// 청크를 전부 소비 — 가중 DRAM 스트림 1회(32× 절감, L2 재사용은 타일 간).
// 행별 산술은 gemv_g32_t_body<16>과 **비트동일**: 레인 l·j-시퀀스·f32 acc·
// tree64 환원이 행 독립이라 t0 그룹화와 무관(잔여 tail<16행은 호스트가
// 기존 ≤16 스위치로 발사 — 핫루프에 가드 없음).
extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_g32_bf16_tall(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out,
    int n, int k, int t16) // t16 = 16의 배수(호스트 계약 — tail은 별도 발사)
{
    const int o0 = blockIdx.x * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32];
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    constexpr int T = 16;
    const int l = threadIdx.x & 31;
    const int g = threadIdx.x >> 5;
    const int o = o0 + g;
    const bool live = o < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    if (live) {
        for (int gg = l; gg < kg; gg += 32) {
            sc[g][gg] = ld_scale<true>(s + (size_t)o * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    const unsigned short* qh =
        reinterpret_cast<const unsigned short*>(q + (size_t)o * k8) + l;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    const int k4 = k >> 2;
    for (int t0 = 0; t0 < t16; t0 += T) {
        float acc[T];
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acc[r] = 0.0f;
        }
#pragma unroll 32
        for (int jj = 0; jj < jn; ++jj) {
            const unsigned qw = (unsigned)qh[jj * 32];
            const float scv = sc[g][(l >> 3) + (jj << 2)];
            const float w0 = fmaf((float)(qw & 0xFu), scv, -8.0f * scv);
            const float w1 = fmaf((float)((qw >> 4) & 0xFu), scv, -8.0f * scv);
            const float w2 = fmaf((float)((qw >> 8) & 0xFu), scv, -8.0f * scv);
            const float w3 = fmaf((float)((qw >> 12) & 0xFu), scv, -8.0f * scv);
#pragma unroll
            for (int r = 0; r < T; ++r) {
                const float4 xv = xb[(size_t)(t0 + r) * k4 + jj * 32];
                acc[r] = fmaf(w0, xv.x, acc[r]);
                acc[r] = fmaf(w1, xv.y, acc[r]);
                acc[r] = fmaf(w2, xv.z, acc[r]);
                acc[r] = fmaf(w3, xv.w, acc[r]);
            }
        }
#pragma unroll
        for (int rr = 0; rr < T; ++rr) {
            red[g][l] = (double)acc[rr];
            __syncthreads();
            double v = (l < 16) ? (red[g][l] + red[g][l + 16]) : 0.0;
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(v, off);
                if (l < off) {
                    v += oth;
                }
            }
            if (l == 0 && live) {
                out[(size_t)(t0 + rr) * n + o] = (float)v;
            }
            __syncthreads();
        }
    }
}

// [S2 2026-10-11] 다중 세그먼트 t행 GEMV(배치 디코드·검증 t≤8) — 테이블은
// t=1 판 w4a16_gemv_g128_multi와 공용((q,s,out,n,k)). 세그먼트 블록수는
// ceil(n/GEMV_TR), 호스트 grid = Σceil(n_s/GEMV_TR). 행별 산술은
// w4a16_gemv_g128_t와 **비트동일**(gemv_g128_t_body 그대로 — 행 독립).
// [A1-점유] 4블록 강제(위 g128_t와 동형 — multi_t도 72reg/3블록 한계).
extern "C" __global__ void __launch_bounds__(256, 4) w4a16_gemv_g128_multi_t(
    const unsigned long long* __restrict__ tab, // [count][5]: q, s, out, n, k
    const float* __restrict__ x,                // [t][k] f32(cast_x32 산출)
    int count, int t)
{
    const int blk = blockIdx.x;
    int seg = count - 1;
    int base = 0;
    for (int ss = 0; ss < count; ++ss) {
        const int ns = ((int)tab[ss * 5 + 3] + GEMV_TR - 1) / GEMV_TR;
        if (blk < base + ns) {
            seg = ss;
            break;
        }
        base += ns;
    }
    const unsigned* q = (const unsigned*)(size_t)tab[seg * 5 + 0];
    const unsigned short* s = (const unsigned short*)(size_t)tab[seg * 5 + 1];
    float* out = (float*)(size_t)tab[seg * 5 + 2];
    const int n = (int)tab[seg * 5 + 3];
    const int k = (int)tab[seg * 5 + 4];
    const int o0 = (blk - base) * GEMV_TR;
    if (o0 >= n) {
        return;
    }
    __shared__ double red[GEMV_TR][32]; // [A-float4] 32레인 트리
    __shared__ float sc[GEMV_TR][G4_SCMAX];
    switch (t) {
        GEMV_T_CASE(1)
        GEMV_T_CASE(2)
        GEMV_T_CASE(3)
        GEMV_T_CASE(4)
        GEMV_T_CASE(5)
        GEMV_T_CASE(6)
        GEMV_T_CASE(7)
        GEMV_T_CASE(8)
        GEMV_T_CASE(9)
        GEMV_T_CASE(10)
        GEMV_T_CASE(11)
        GEMV_T_CASE(12)
        GEMV_T_CASE(13)
        GEMV_T_CASE(14)
        GEMV_T_CASE(15)
        GEMV_T_CASE(16)
        default: break; // 호스트 계약 밖(t≤16)
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
// [S6 실측 기각 2026-10-11] 프리필 디큐트 최적화 시도 — (진단)디큐트 제거
// (return qw·로드 유지) A/B: pp512 288→260ms(−9.5%)·pp4096 −9.5%·pp16384
// −8.2% = 디큐트 체인 총 비중 ~9%가 상한. ① B 디큐트 mf-호이스팅(중복
// 2→1회, 비트동일, 레지스터 79·스필 0 동일): pp512/4096 +1.7~2.0% 역행 —
// 커널은 issue-bound가 아니라 지연/스케줄 민감(ncu No Eligible 77% 정합).
// ② 잔여 lop3/PRMT-lite 상한(추출·전개 ~5/12 op) <1% — 무의미. 현행 유지.
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

// [§1.3 실측 기각 2026-10-11] B smem-f16 디큐트(블록 1회 — 레지스터 디큐트의
// m워프 중복 제거): 골든 비트동일·math_pipe 부하 감소에도 **전면 악화**
// (P4096 2765→2838ms·pp16384 14494→15241ms·커널 7712→7958ms) — smem
// 30.7KB로 L1 축소가 우세(marlin-C4 30.5KB 실측과 동형). 되돌림. 잔여
// math_pipe_throttle(2.88) 해소는 smem 비증가 구조에서만 시도할 것.
// [실측 기각 원장 — 2026-10-18 ⑭/⑰②] (1) launch_bounds(256,3→4)(REG 79→64·
// 스필 8B·4블록): 비트동일·**pp512 −4.4%** — reg 79 할당의 스케줄/ILP가
// load-bearing(sk_marlin lb5 −2.1%와 동형). (2) t=2 행-분할 매핑(워프=1행×2열·
// 셔플 트리 — x 재판독 8→2워프): 비트동일·**np2 −13%**(np4 −0.5%) — 행공유 v2
// (−26~53%)·TR16·x 제거 상한 +24%와 합쳐 **x 축 3차 종결**. 잔여 = M-타일 확대
// (M128 — 가중 재판독 t/M ÷2)급 재설계뿐.
// (3) M128 타일(512스레드·4m×4n·REG64 캡·점유 50%·가중 재판독 t/M ÷2):
// 비트동일·**pp512/1024 −6.5%** — M32→M64 채택([marlin-A4]) 이후 M64이 국소
// 최적(M128 = 그리드 축소·캡 스케줄 손실 > 트래픽 이득). 타일 축 소진.
// [⑰② 실측 2026-10-18] 배치 np2 합산 = +2%뿐(per-slot 43.6ms=2.0×) — t=2의
// per-row 비용이 t=1 sk와 동급(가중 공유 이득 없음 — issue/지연 바운드).
// ncu(multi_t t=2): L1/TEX 77%·DRAM 6%·SM 38%. np4는 +36%(디큐트 상각 — t≥4 수익).
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
            // [§1.3 재프로파일 2026-10-11] .cg 적용 후: 538→479.7µs(−11%)·
            // math_pipe_throttle 4.10(디큐트 명령 — 계약 고정)+L2 77%(A/B
            // 트래픽). 둘 다 구조/계약 고정 — 잔여는 영속 타일 재편뿐.
            // [§1.3 2026-10-11] B 스테이징 16B .cg — 종전 4B .ca는 가중치
            // (1회 판독·재사용 없음)를 L1에 채워 프래그먼트/A의 L1을 압박
            // (carveout 실측: 이 커널은 L1 민감). 16B 청크 = 행당 4 u32
            // (전 k청크 — k%128=0 계약으로 경계 정합) → .cg(L2 전용).
            if (tid < MMA_N) {
                const int r = tid;
                if (n0 + r < n) {
                    cp_async16(&wsp[cur ^ 1][r][0], &q[(size_t)(n0 + r) * k8 + (kn >> 3)]);
                } else {
#pragma unroll
                    for (int c8 = 0; c8 < MMA_KC / 8; ++c8) {
                        wsp[cur ^ 1][r][c8] = 0x88888888u;
                    }
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
// [S5-lite 채택 2026-10-11] GRP_KC 32→64 이력 — 근거: grp 병목 = 청크당
// 배리어 2회 스톨 41%(§1.2 ncu) — KC 2배 = 배리어·스테이징 횟수 절반. 값
// 비트동일(mma k순서 불변 — 35B 골든 일치 실측). 실측: pp512
// 4520→4695(+3.7%)·pp4096 4110→4240(+3.2%). smem 27.6KB·스필 0.
// (라우팅 분포 실측 부기: 256전문가·4096슬롯 → 전문가당 1-16 다수 —
// M타일 축소(M32/16)는 B 디큐트 타일 중복 때문에 역효과로 모델링됨.)
// [A2b 2026-10-15] 2-pass(A packed hi/lo + B f16 단일)로 smem이 늘어 **32로
// 환원** — A 20.5KB + B 10.2KB = 30.7KB(2~3블록/SM). KC=64면 55KB(블록 1개)
// 로 점유가 반감 — 배리어 상각(위 실측)보다 점유를 우선한다. (B hi/lo
// 3-pass 실측 = en/ko 비트동일·비용 +26% — 기각.)
#define GRP_KC 32
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
    // [FLA-14 기각 2026-10-10] A 프래그먼트 ldmatrix.x4 — 명시 판독 4×pk2bf를
    // 1명령으로(mio 후보). 골든 비트동일(35B 600/4000 — 분배 규약 정확)이나
    // **성능 무이득**: 교차 A/B P512 127~128ms 동일 · g32_grp 58.7~59.0ms
    // 동일(P4096 정상상태 457.6→456.8). 1차 측정의 −5%는 기준선 아웃라이어
    // (62ms) 오염이었다. 되돌림.
    // (레지스터 디큐트×2·cp.async·KC·NSPLIT·FLA-5) — SASS상 디큐트는 이미
    // 컴파일러 최적(PRMT 팩·STS.64).
    __shared__ __align__(16) unsigned short xs[2][GRP_M][GRP_KC + 8];
    __shared__ unsigned short ws[2][GRP_N][GRP_KC + 8];
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
            for (int ee = tid; ee < GRP_N * GRP_KC / 8; ee += 256) {
                const int r = ee / (GRP_KC / 8);
                const int c8 = ee % (GRP_KC / 8);
                const int gi = kn + c8 * 8;
                const bool live = (n0 + r < n) && (gi + 7 < k);
                unsigned short scb = 0;
                if (n0 + r < n) {
                    scb = s[(size_t)(n0 + r) * kg + (gi >> 5)];
                }
                const float scf = __uint_as_float((unsigned)scb << 16);
                // [실험] 스트리밍 힌트 — B는 블록별 1회 판독(재사용 없음).
                const unsigned qw = live ? __ldcs(&q[(size_t)(n0 + r) * k8 + (gi >> 3)]) : 0u;
                if (live) {
                    // [half2 비트매직 2026-10-11] 원소별 (float)nib*scf+f2h
                    // (ALU ~7/elem·STS.16×8) → deq_pair_b 쌍 처리(hsub2 정확
                    // 경로 — 비트동일, g128 mma 입증 부품 재사용·STS.32×4).
                    // 실측 35B pp512/4096 +0.3%(골든 유지).
#pragma unroll
                    for (int jp = 0; jp < 4; ++jp) {
                        *reinterpret_cast<unsigned*>(&ws[st][r][c8 * 8 + jp * 2]) =
                            deq_pair_b(qw, jp, scf);
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
            // 다음 청크를 발사(mma와 겹침) — A cp.async + B 디큐트.
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


// [§1.2 2026-10-11] 게이트·업 2N 융합 — grp와 동일 파이프라인에서 N타일을
// 2×GRP_N(행 0..31=게이트, 32..63=업)으로 확장. ncu 확정: grp 병목은
// 배리어 스톨 41%+MIO 12%(청크당 A/B 스테이징+배리어 2회가 지배, DRAM 17%,
// mma 파이프 유휴) — 청크당 작업을 2배로 해 배리어·스테이징을 상각한다.
// A 스테이징 1회 공유(발사 1회로 게이트+업 동시), B만 2벌. 워프 n매핑
// ntw=(warp&1)*32·nt 0..3. 요소별 mma 누산 순서 동일 — 토큰 골든 판정.
// [A2b 2026-10-15] 2-pass 정밀도 회수(아래 grp_d64와 동형): A = (hi|lo) packed
// u32, B = f16 단일, mma 2회(A_hi·B + A_lo·B — A 반올림 제거, f32 클래스 근접).
// 활성은 gemma4·35B 공용(모델별 선택 아님). B hi/lo 3-pass는 실측 기각.
extern "C" __global__ void w4a16_gemm_g32_mma_grp2(
    const unsigned long long* __restrict__ tab, int base,
    const unsigned* __restrict__ gslot, const unsigned* __restrict__ gcnt,
    const unsigned* __restrict__ goff,
    const unsigned* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, float* __restrict__ out2, int n, int k)
{
    const int e = blockIdx.x;
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
    __shared__ __align__(16) unsigned xs[2][GRP_M][GRP_KC + 8]; // packed hi|lo
    __shared__ unsigned short ws[2][2 * GRP_N][GRP_KC + 8];
    const unsigned mtiles = (cnt + GRP_M - 1) / GRP_M;
    for (unsigned mtile = 0; mtile < mtiles; ++mtile) {
        const unsigned rbase = mtile * GRP_M;
        float c[4][4];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ((float*)c)[i] = 0.0f;
        }
        auto stage_ab = [&](int st, int kn) {
            for (int ee = tid; ee < GRP_M * GRP_KC / 4; ee += 256) {
                const int r = ee / (GRP_KC / 4);
                const int c4 = ee % (GRP_KC / 4);
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
                // [실험] 스트리밍 힌트 — B는 블록별 1회 판독(재사용 없음),
                // L2를 A(x16) 재사용에 남긴다.
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
