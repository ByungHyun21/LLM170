#include "cast_common.cuh"

// ── norm_resid CUDA 포팅 (G3, 2026-10-04) ──
// 산술은 구 rawhip 커널의 norm_resid 1:1 직이식(원본
// 그대로 베낌 — 부동소수 적산 순서·정밀 sqrt 계약 포함). 차이는 hip 판이
// 5120으로 경직된 행 폭을 hidden 인자로 일반화한 것뿐(hidden=5120이면
// 원본과 원소 순서까지 동일 — v0..v4 스트라이드 적재가 루프 좌결합으로
// 재현된다).
//
// 계약:
// - xn = rms_norm(x+ab)·nw[w행] — eps=1e-6, 평균은 hidden으로 나눈다.
// - w는 nw 배열의 행 포인터: 오프셋 w·hidden(27B는 w·5120 — 오프셋
//   누락이 결함 2호: 전 노름이 L0 행 판독).
// - 잔차 스트림은 제자리 가산 — 커널이 x = x + ab를 기록한다.
// - 인자 순서 (x, nw, ab, xn) 고정 — 교차 시 잔차에 노름가중치가
//   더해진다(결함 7호).
// - inv는 정밀 sqrtf(vk 원본 주석): rsqrt 근사는 drift 씨앗.
//
// [CMP 170HX(sm_80, GA100, HBM2e ~1.5TB/s) 설계 근거]
// 노름은 순수 메모리 본드(행당 읽기 x·ab·w + 쓰기 x·xn = 5·hidden·4B,
// 5120 기준 100KB). 런치 기하(블록 1024=32워프, 그리드 (t_len,1), 스레드당
// hidden/1024원소 스트라이드 적재 — 5120이면 ILP-5)는 hip 검증 배치 그대로:
// GA100에서 smem red[1024]=4KB·레지스터 상한(실측 cuobjdump)으로
// 2블록/SM 상주 = 2048스레드 = 64워프 풀 점유가 성립하고, ILP-5 연속
// f32 로드로 HBM2e 스트리밍 병렬성을 확보한다. float4 벡터화는 산술
// 미러(§3.2 1:1)를 깨지 않는 선에서 후속 실측 후 판단 — 개발 호스트
// 4070(sm_89) 타이밍은 판단 근거가 아니다(정합 검증 전용).
//
// hidden 계약: 1024의 배수, 8192 이하(v[8] 레지스터 적재 상한).
// 27B hidden=5120(nper=5) · 35B hidden=2048(nper=2). 256원소 q/k_norm
// 행(결함 9호 패딩 쟁점)은 본 커널 범위 밖(G4+ MTP 단계).
extern "C" __global__ void norm_resid(
    float* __restrict__ x,        // [T][hidden] r/w — 잔차 스트림(제자리 가산)
    const float* __restrict__ nw, // [rows][hidden] 노름 가중 배열
    const float* __restrict__ ab, // [T][hidden]
    float* __restrict__ xn,       // [T][hidden]
    float* __restrict__ xn32,     // [T][hidden] h2f(f2h(xn)) — 융합 캐스트.
                                  // 0이면 생략(cast_x32 커널과 비트 동일 계약:
                                  // 노드 −2/층 → 그래프 노드·런치 절감).
    int t_len, int w_off, int hidden)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * hidden;
    int nper = (hidden + 1023) >> 10;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        v[j] = x[e] + ab[e];
        ss += v[j] * v[j];
    }
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    // 정밀 sqrt 계약(vk 원본 주석): rsq 근사는 drift 씨앗.
    float inv = 1.0f / sqrtf(red[0] / (float)hidden + 1e-6f);
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        float y = v[j] * inv * nw[w_off + (j << 10) + tid];
        xn[e] = y;
        if (xn32 != (float*)0) {
            xn32[e] = h2f(f2h(y)); // cast_x32 커널과 동일 산식(비트 동일)
        }
        x[e] = v[j];
    }
}
