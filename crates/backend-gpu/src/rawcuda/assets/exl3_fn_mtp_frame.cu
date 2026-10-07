// llm170 rawcuda Flash-Next MTP 드래프트 프레임 커널 (plans/124 FNG,
// 2026-10-05).
//
// [계약] 본 파일의 산술 원천은 crates/core/src/qwen4exp/frame/mtp.rs
// (mtp_draft_frame — CPU 황금 계약, plans/124 §6 유일 기준)이다. 체인의
// 대부분 단계는 기착 스테이지 모듈이 이미 검증했다 — 본 파일은 그 중
// 어디에도 없는 MTP 고유 2커널만 담는다(재사용 지도는 아래):
//   - eh_proj/투영 4종(q·k·v·o)/logits 헤드 GEMV: llm170_fn_hc_gemv
//     (assets/exl3_fn_hc.cu — cpu.rs matmul L64-76 순차 f32 누산 미러,
//     FNC 비트동일 원장) 재사용. 모듈(mtp_fn_cuda.rs)이 exl3_fn_hc.fatbin을
//     함께 적재해 발사한다 — 커널 중복 파생 금지(rawhip 원칙 계승).
//   - hc 저랭크 믹서 3종(attn/ffn/nextn_head): hc_cuda.rs HcCuda(FNC)
//     모듈 재사용(grouped rms·down/inject·silu(lo/hc)·up·gate_mean).
//   - MoE 라우팅·전문가·shared: moe_cuda.rs MoeCuda(FNE) 모듈 재사용.
//   - dense 어텐션의 norm/rope/KV/softmax/게이트: core 프레임 경로와
//     동일하게 CPU 유지(frame/mtp.rs L14 헤더 계약 — mtp_attn_cpu_row
//     공유 코어의 모듈층 미러가 호스트에서 구동, GPU화는 별계 목표).
//
// [llm170_fn_mtp_combine] FrameOp::HcCombine(matmul/traits.rs L918-925)의
// CUDA 미러 — 원천 수식: layers.rs hc_combine L2575-2586
//   res[t][s*n+i] += out[t][i] * (2*sigmoid(inj[t][s]/hc))
// - sigmoid = ops.rs sigmoid L131-133(1/(1+exp_cr(-x))) — exp_cr 트윈을
//   리터럴까지 직이식(아래 mtp_exp_cr — exl3_fn_hc.cu hc_exp_cr과 동일
//   DAG. mul_add(호스트)≡__fma_rn(디바이스) 올림-정확 단일 반올림).
// - 곱·가산 2회 반올림 유지(오라클 hc_combine의 ov*w → += 순서 미러)
//   → -fmad=false 필수(아래 [빌드 계약]).
// - t=1 드래프트 전용: 발사 grid(ceil(hc*n/256))×256, 블록당 원소 스트림
//   s는 i/n에서 유도(스트림 경계 정렬 — n=2560은 256의 배수라 블록이
//   스트림을 가로지르지 않는다; 일반성을 위해 s 인덱싱은 원소 단위로
//   계산, inj[s]는 스레드 공통 스칼라가 아닌 per-stream 로드).
//
// [llm170_fn_mtp_argmax] greedy 미러 — 원천: cpu.rs greedy_from L235-253.
// v > bv(초과 갱신) 순차 스캔 = 동률 최저 인덱스 승리. n은 로짓 길이
// (vocab=248320 — 결함 8호: 행수 아님). 정합 우선 설계: 1스레드 순차
// 스캔(248320 f32 ≈ 0.25ms 계급 — 병렬 환원의 타이 브레이크·순서 계약을
// 현재는 단순화; sm_80 실측 후 원장 18호 계급으로 재판정한다).
//
// [빌드 계약] 별도 -fmad=false 블록(scripts/build_cuda.bat) — G5/G6/G7/
// G8/FNA/FNC 노선: combine의 acc += ov*w 가 FFMA 수축되면 호스트 미러
// (layers.rs hc_combine의 2회 반올림)와 비트가 어긋난다. argmax는 비교
// 전용이라 플래그 무관이나 동일 fatbin.
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
// 드래프트 1스텝의 지배 비용은 헤드 GEMV(248320×2560 f32 = 2.5GB →
// HBM2e 하한 ~1.7ms)와 MoE(전문가 스트리밍) — 본 파일 2커널은 combine
// (40KB r/w)·argmax(1MB r)로 대역폭 관점 무시 가능. 병렬화 여지는
// sm_80 실측 후 판단(개발기 sm_89 타이밍은 판단 근거 아님 — plans/124 §0).
//
// 독립 컴파일 계약(plans/124 G1): 표준 CUDA C++만 — 외부 의존 없음.

#include <math.h>

// ── core exp_cr 직이식(ops.rs L63-119 — 비트동일 미러) ──
// f64 FMA 호너 13차: k=round_ties_even(x·log2e) → r=x−k·ln2(hi/lo) →
// 테일러 → 2^k 비트 재구성. 리터럴·연산 순서 변경 금지(호스트 오라클과
// 비트동일 조건 — exl3_fn_hc.cu hc_exp_cr과 동일 DAG).
__device__ __forceinline__ float mtp_exp_cr(float x)
{
    double xd = (double)x;
    if (xd > 88.72) return __int_as_float(0x7f800000);
    if (xd < -103.97) return 0.0f;
    const double LN2_HI = 6.9314718036912382e-01;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634; // log2(e) — std 상수 비트동일
    double kd = rint(xd * INV_LN2);            // f64 round_ties_even 미러
    double r = __fma_rn(-kd, LN2_HI, xd);
    r = __fma_rn(-kd, LN2_LO, r);
    double p = 1.0 / 1307674368000.0;          // 1/13!
    p = __fma_rn(p, r, 1.0 / 479001600.0);     // 1/12!
    p = __fma_rn(p, r, 1.0 / 39916800.0);      // 1/11!
    p = __fma_rn(p, r, 1.0 / 3628800.0);       // 1/10!
    p = __fma_rn(p, r, 1.0 / 362880.0);        // 1/9!
    p = __fma_rn(p, r, 1.0 / 40320.0);         // 1/8!
    p = __fma_rn(p, r, 1.0 / 5040.0);          // 1/7!
    p = __fma_rn(p, r, 1.0 / 720.0);           // 1/6!
    p = __fma_rn(p, r, 1.0 / 120.0);           // 1/5!
    p = __fma_rn(p, r, 1.0 / 24.0);            // 1/4!
    p = __fma_rn(p, r, 1.0 / 6.0);             // 1/3!
    p = __fma_rn(p, r, 0.5);
    p = __fma_rn(p, r, 1.0);
    p = __fma_rn(p, r, 1.0);
    if (kd > 127.0) return __int_as_float(0x7f800000);
    long long k = (long long)kd;
    double scale = __longlong_as_double((k + 1023) << 52);
    return (float)(p * scale);
}

// ── hc combine — FrameOp::HcCombine / layers.rs hc_combine L2575-2586 ──
// res[s*n+i] += out[i] * (2*sigmoid(inj[s]/hc)) — t=1 드래프트 단일 행.
// w는 스트림당 1회 산출(호스트 hc_combine의 s 루프 순서·2회 반올림 미러:
// ov*w 곱 → += 가산, FMA 수축 없음).
extern "C" __global__ void llm170_fn_mtp_combine(
    float* __restrict__ res,        // [hc*n] 드래프트 잔차(제자리 갱신)
    const float* __restrict__ out,  // [n] combine 입력(attn/ffn 출력)
    const float* __restrict__ inj,  // [hc] inject 스칼라
    int n, int hc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    int s = i / n;
    float w = 2.0f * (1.0f / (1.0f + mtp_exp_cr(-inj[s] / (float)hc)));
    res[i] += out[i - s * n] * w;
}

// ── argmax — cpu.rs greedy_from L235-253 미러(동률 최저 인덱스) ──
// 1스레드 순차 스캔(v > bv 초과 갱신). n = 로짓 길이(결함 8호).
extern "C" __global__ void llm170_fn_mtp_argmax(
    const float* __restrict__ logits,  // [n]
    int n,
    unsigned int* __restrict__ tok)    // [1]
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned best = 0u;
    float bv = -INFINITY;
    for (int i = 0; i < n; ++i) {
        float v = logits[i];
        if (v > bv) {
            bv = v;
            best = (unsigned)i;
        }
    }
    tok[0] = best;
}
