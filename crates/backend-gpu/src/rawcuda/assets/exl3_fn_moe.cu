// ── Flash-Next MoE FFN CUDA 커널 (plans/124 G001 FNA→FNE, 2026-10-05) ──
// 계약 소스는 이 .cu — 빌드 자산 exl3_fn_moe.fatbin은 scripts/build_cuda.bat이
// 같은 디렉터리에 쓰고 함께 커밋된다(rawhip co/*.co 미러: 소스·자산 동반).
//
// [산술 계약 — 원천: crates/core/src/qwen4exp/stages/moe.rs (CPU 황금 기준)]
// - 라우팅: softmax(max-sub) → total_cmp 내림 정렬(동률은 오름차 id — Rust
//   sort_by 안정성) → top-n_used → wsum 하한 6.1035156e-5 정규화 → w==0
//   스킵. 원천 moe.rs L46-70. 커널은 이 선택을 "상향 스캔·초과 갱신(최초
//   등장 우선)" 반복 선택으로 동치 구현한다(내림차 안정 정렬의 위너 =
//   남은 것 중 최댓값의 최저 id 최초 등장).
// - 전문가/shared 선형: y = Σ_i x[i]·W[o][i] 순차 f32 누산(오름차 i) —
//   core matmul CPU 경로(cpu.rs L91-94: acc += x[i]*scratch[i])와 동일
//   순서·동일 곱/가산 2회 반올림. 가중치 F16 행 우선 [n_out][n_in]
//   (EXL3 라우터 mlp.gate.weight F16[512,2560] 실측 계급 — 트렐리스 아님,
//   fn_support.rs 계약 지도 "MoE 라우트"/"MoE shared" 항).
// - 게이트 활성화: y = (v/(1+exp(−v)))·u — ops.rs silu L127-131 + moe.rs
//   silu_rows L10-16·r[i] *= u[i] (L249-255).
// - 결합: 토큰별 전문가 기여 e-오름차 누산(moe.rs L198-216 토큰-메이저
//   (ti,e,w) 페어 정렬 계약 — 전문가별 서브배치 경로와 누산 순서 동일,
//   L228-297 class) 후 shared sigmoid(sgate) 가중 가산(L313-318:
//   sh_w = sigmoid(sgate_all[ti][0]); o[i] += sh_w*shout[ti][i]).
// - sigmoid: 1/(1+exp(−x)) — ops.rs L133-135.
//
// [트랜센던트 계약 — 치환이 아닌 미러]
// core exp_cr(ops.rs L52-89)은 f64 FMA 호너 13차 DAG — fma()의 IEEE 단일
// 반올림·round_ties_even(rint 기본 모드)·비트 재구성이 전부 CUDA로 이식
// 된다. fn_moe_exp_cr은 core exp_cr과 비트동일(G5/G7의 트윈 "치환"과 달리
// 같은 DAG의 직이식). 예외는 라우팅 softmax의 exp뿐: core moe.rs L48은
// std f32::exp(시스템 libm — 기기 간 이식 불가)를 쓰므로, 커널과 검증층
// 오라클 양측 모두 exp_cr 미러로 치환한다(G5/G7 노선 계승, plans/124 §6
// — exp는 단조라 이산 선택 순서 불변, 동일 비트 입력에 동일 비트 출력으로
// 결정론 유지).
//
// [빌드 계약] -fmad=false 별도 블록(scripts/build_cuda.bat) — G5/G6/G7/G8/
// FNA 노선: acc += x[i]*w 및 acc += w*y가 FMA 수축되면 호스트 미러(core
// cpu.rs 순차 2회 반올림)와 비트가 어긋난다. 정합 판정은 값 maxdiff
// (argmax·토큰 판정 금지 — plans/124 §6. 단 라우팅 전문가 선택은 이산
// 계약이라 모듈↔오라클 exact-match로 별도 판정한다 — 과제 계약).
//
// [NaN 도메인] 라우팅 로짓은 finite 계약 — core의 NaN 로짓 경로(moe.rs
// L57-59 total_cmp NaN 내성 + L72-77 trace 보고)는 서버 nan_guard 영역.
// 본 모듈 프로브는 finite 픽스처만 다룬다(계약 경계 문서화).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
// - fn_moe_gemm_f16_ptr: 그리드 (ceil(n_out/128), np)·블록 128 — 스레드당
//   출력 1개·내부 순차 누산(비트동일 계약이 우선). 행 우선 F16에서 인접
//   스레드는 n_in 스트라이드로 읽어 완전 coalesce가 아니나, MoE 전문가
//   GEMM은 활성 x가 전문가당 1행(재사용 없음 — 토큰-메이저 페어 계약)이라
//   HBM2e 관점 W 스트리밍 본드가 지배적이다. sm_80 실측 장비 도착 후
//   mma(m16n8k16) 타일 재판정(plans/124 §1 "배치 GEMM — CUDA의 승부처").
//   개발기(RTX 4070 SUPER, sm_89)는 정합 검증 전용 — 타이밍 판단 근거 아님.
// - fn_moe_route: 토큰당 단일 스레드 순차(softmax zs·선택 누산의 core
//   순서 계약) — 512 exp·5k 비교/토큰으로 ALU 미미, 전문가 GEMM 대비
//   <0.1%. 블록 32(활성 1레인)는 발사 최소단위.
// - fn_moe_ew: G7 exl3_ew 동일 등급(그리드 ceil(n/128)·블록 128).
// - fn_moe_combine: 그리드 (t,1)·블록 256 — 원소별 coalesce 스트리밍,
//   페어 루프는 레지스터 누산(HBM2e 관점 순수 스트리밍).

#include <cuda_fp16.h>
#include <math.h>

// ── core exp_cr 직이식(ops.rs L52-89 — 비트동일 미러) ──
// f64 FMA 호너 13차: k=round_ties_even(x·log2e) → r=x−k·ln2(hi/lo) →
// 테일러 → 2^k 비트 재구성. 연산 순서·상수 변경 금지(호스트 오라클
// moe_cuda_probe.rs exp_cr_mirror와 리터럴까지 동일해야 비트동일).
__device__ __forceinline__ float fn_moe_exp_cr(float x)
{
    double xd = (double)x;
    if (xd > 88.72) return INFINITY;
    if (xd < -103.97) return 0.0f;
    const double LN2_HI = 6.9314718036912382e-01;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634; // log2(e) — std 상수 비트동일
    double kd = rint(xd * INV_LN2);            // f64 round_ties_even 미러
    double r = fma(-kd, LN2_HI, xd);
    r = fma(-kd, LN2_LO, r);
    double p = 1.0 / 1307674368000.0;
    p = fma(p, r, 1.0 / 479001600.0);
    p = fma(p, r, 1.0 / 39916800.0);
    p = fma(p, r, 1.0 / 3628800.0);
    p = fma(p, r, 1.0 / 362880.0);
    p = fma(p, r, 1.0 / 40320.0);
    p = fma(p, r, 1.0 / 5040.0);
    p = fma(p, r, 1.0 / 720.0);
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    long long k = (long long)kd;
    if (k > 127) return INFINITY;
    double scale = __longlong_as_double((long long)(k + 1023) << 52);
    return (float)(p * scale);
}

// ── 라우팅: softmax → top-n_used → 정규화(moe.rs L46-70 미러) ──
// 토큰당 블록 1개·활성 스레드 1(레인 0). zs·wsum·선택 전부 core 순차
// 순서(오름차 e / 선택 순서 k). 출력: ids[t][n_used]·wts[t][n_used]에
// 선택 순서(확률 내림, 동률 id 오름차)로 w!=0 만 압축 기록, cnt[t]는
// 유지 원소 수(w==0 스킵 — moe.rs L66-68 by_expert push 가드).
// 도메인: n_exp·5B ≤ 동적 공유(실측 계급 256/512 ≪ 48KB)·n_used ≤ 16.
extern "C" __global__ void fn_moe_route(
    const float* __restrict__ logits,
    int n_exp,
    int n_used,
    int* __restrict__ ids,
    float* __restrict__ wts,
    int* __restrict__ cnt)
{
    extern __shared__ float prob[];                    // [n_exp]
    unsigned char* taken = (unsigned char*)(prob + n_exp); // [n_exp]
    if (threadIdx.x != 0) return;
    const float* lg = logits + (size_t)blockIdx.x * (size_t)n_exp;
    float mx = -INFINITY;                              // L47 fold(NEG_INF,max)
    for (int e = 0; e < n_exp; ++e) mx = fmaxf(mx, lg[e]);
    float zs = 0.0f;                                   // L48-52(오름차 순차)
    for (int e = 0; e < n_exp; ++e) {
        float v = fn_moe_exp_cr(lg[e] - mx);
        prob[e] = v;
        zs += v;
    }
    for (int e = 0; e < n_exp; ++e) prob[e] /= zs;     // L53-55(나눗셈)
    for (int e = 0; e < n_exp; ++e) taken[e] = 0;
    int sel[16];                                       // n_used ≤ 16 도메인
    for (int k = 0; k < n_used; ++k) {                 // L57-59 동치 선택
        int be = -1;
        float bv = -INFINITY;
        for (int e = 0; e < n_exp; ++e) {
            if (!taken[e] && prob[e] > bv) { bv = prob[e]; be = e; }
        }
        taken[be] = 1;
        sel[k] = be;
    }
    float wsum = 0.0f;                                 // L61(선택 순서 순차)
    for (int k = 0; k < n_used; ++k) wsum += prob[sel[k]];
    wsum = fmaxf(wsum, 6.1035156e-5f);                 // L62 하한 가드
    int c = 0;
    for (int k = 0; k < n_used; ++k) {                 // L63-68(w!=0 스킵)
        float w = prob[sel[k]] / wsum;
        if (w != 0.0f) {
            ids[(size_t)blockIdx.x * n_used + c] = sel[k];
            wts[(size_t)blockIdx.x * n_used + c] = w;
            ++c;
        }
    }
    cnt[blockIdx.x] = c;
}

// ── F16 선형(포인터 배열 GEMM) — core cpu.rs L91-94 순차 누산 미러 ──
// out[p][o] = Σ_i x[p][i]·W_p[o][i]. 페어 p마다 독립 가중치 포인터
// wptr[p](mm_paired 계급 — moe.rs "전문가별 1행 입력·상이 가중치" 디스패치,
// stages/mod.rs mm_paired L50-75). 라우터·shared 재사용도 동일 커널:
// wptr 전 원소가 같은 포인터면 배치 GEMV가 된다. -fmad=false 하에서
// 곱·가산 2회 반올림 = 호스트 미러와 비트동일.
extern "C" __global__ void fn_moe_gemm_f16_ptr(
    const float* __restrict__ x,
    const unsigned long long* __restrict__ wptr,
    float* __restrict__ out,
    int n_in,
    int n_out)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    size_t p = blockIdx.y;
    const __half* w = (const __half*)(uintptr_t)wptr[p];
    const float* xr = x + p * (size_t)n_in;
    float acc = 0.0f;
    for (int i = 0; i < n_in; ++i)
        acc += xr[i] * __half2float(w[(size_t)o * n_in + i]);
    out[p * (size_t)n_out + o] = acc;
}

// ── 게이트 활성화 silu·mul(ops.rs silu L127-131 + moe.rs L249-255) ──
// y[j] = (v/(1+exp(−v)))·u — G7 exl3_ew와 동일 구조, exp만 exp_cr 직이식.
// 전문가 게이트·업(np·n_ff)과 shared(t·n_ff_sh) 공용.
extern "C" __global__ void fn_moe_ew(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float e = fn_moe_exp_cr(-v);
    y[j] = (v / (1.0f + e)) * u[j];
}

// ── 결합: 전문가 가중 누산(e-오름차 페어) + shared sigmoid 게이트 ──
// moe.rs L204-216(페어 정렬 = 토큰 내 e 오름차)·L313-318. 원소별:
// acc = Σ_p w_p·yo[p][i](순차, 곱·가산 분리) → + sigmoid(sgate[ti])·shout.
// sigmoid = 1/(1+exp(−x)) — ops.rs L133-135 미러.
extern "C" __global__ void fn_moe_combine(
    const float* __restrict__ yo,
    const float* __restrict__ wpair,
    const int* __restrict__ poff,
    const float* __restrict__ shout,
    const float* __restrict__ sgate,
    float* __restrict__ out,
    int n_embd)
{
    int ti = blockIdx.x;
    float e = fn_moe_exp_cr(-sgate[ti]);
    float shw = 1.0f / (1.0f + e);
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float acc = 0.0f;
        for (int p = poff[ti]; p < poff[ti + 1]; ++p)
            acc += wpair[p] * yo[(size_t)p * n_embd + i];
        out[(size_t)ti * n_embd + i] = acc + shw * shout[(size_t)ti * n_embd + i];
    }
}
