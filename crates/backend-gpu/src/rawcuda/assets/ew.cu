// ── EW(silu·mul) CUDA 포팅 (G7, 2026-10-04) ──
// 산술은 구 rawhip 커널 ew(L898-908) 1:1 직이식 — y[j] = silu(g[j])·u[j].
// 트랜센던트: [계약 완화 2026-10-09] f32 __expf. 종전엔 CPU 참조
// (core/ops.rs silu)와의 비트동일을 위해 f64 DAG 트윈(ew_exp_d)을 썼으나
// 독립 벤치(--bench-ew, n=245760)에서 0.015ms/발사(197GB/s) vs
// __expf 0.004ms(722GB/s) = 커널 3.7× — 허용오차 등급(골든·장문 판정 통과).
// 빌드는 -fmad=false(FMA 수축 제거 — f32 add/div/mul 순서 보존).
//
// [CMP 170HX(sm_80) 설계 근거] 그리드 (ceil(n/128),1)·블록 128 — 원본 hip
// 발사와 동일. 27B FFN n=17408 → 136블록, g/u/y 완전 coalesce 스트리밍
// (블록당 512B×3) — 순수 메모리 본드 원소별 연산.

#include "cast_common.cuh"
__device__ __forceinline__ float ew_expf(float x)
{
    return __expf(x);
}

// ── ew 본체(구 rawhip 커널 L898-908 직이식) ──
// y[j] = silu(g[j])·u[j] — 디코드 FFN 게이트·업 곱(배치 t는 호출부가
// n = t×ff로 편다).
// [4.1 2026-10-11] 생산자 융합 — 선택 출력 y32/y16(nullable) 동시 기록.
// y32 = h2f(f2h(act))·y16 = f2h(act) — cast_x32/cast_x16 커널과 동일 산식
// (norm_resid xn32/xn16 P6 패턴, cast_common 동일 함수 → 비트 동일).
// split 체인의 독립 cast 노드 제거용 — MoE 등 원시 f32 소비 경로는 null.
extern "C" __global__ void ew(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    float* __restrict__ y32,
    unsigned short* __restrict__ y16,
    int n)
{
    // [float4 벡터화 실측 기각 2026-10-11] 4원소/스레드 변형 = pp4096 −3.0%
    // (교차 A/B old 1666 vs new 1617 t/s·원소별 산술 동일인데도 역행 —
    // 원인 미규명, 소형 커널 특성으로 판단) — 스칼라 유지.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float e = ew_expf(-v);
    float act = (v / (1.0f + e)) * u[j];
    y[j] = act;
    if (y32 != (float*)0) {
        y32[j] = h2f(f2h(act));
    }
    if (y16 != (unsigned short*)0) {
        y16[j] = f2h(act);
    }
}

// 마커 ewc

// ── ew_gelu — gemma4 FFN 게이트(gelu tanh 근사)·업 곱 (2026-10-14) ──
// HF `gelu_pytorch_tanh`: 0.5x(1+tanh(√(2/π)(x+0.044715x³))).
// 산술: CPU 참조(core/ops.rs gelu_tanh)와 동일 식. tanhf는 트랜센던털
// 등급(eval 판정 — 비트 아님; ew silu의 __expf 완화와 동일 클래스).
// y32/y16 생산자 융합은 ew와 동일 규약(선택 출력 — null이면 미기록).
extern "C" __global__ void ew_gelu(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    float* __restrict__ y32,
    unsigned short* __restrict__ y16,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float x3 = v * v * v;
    float act = 0.5f * v * (1.0f + tanhf(0.7978846f * (v + 0.044715f * x3))) * u[j];
    y[j] = act;
    if (y32 != (float*)0) {
        y32[j] = h2f(f2h(act));
    }
    if (y16 != (unsigned short*)0) {
        y16[j] = f2h(act);
    }
}

