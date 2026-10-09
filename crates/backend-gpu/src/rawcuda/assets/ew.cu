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
__device__ __forceinline__ float ew_expf(float x)
{
    return __expf(x);
}

// ── ew 본체(구 rawhip 커널 L898-908 직이식) ──
// y[j] = silu(g[j])·u[j] — 디코드 FFN 게이트·업 곱(배치 t는 호출부가
// n = t×ff로 편다).
extern "C" __global__ void ew(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float e = ew_expf(-v);
    y[j] = (v / (1.0f + e)) * u[j];
}

// 마커 ewc
