// head 자산 — 로짓 argmax(min-index-on-tie) 커널.
// [A-4 2026-10-10] 종전 head_bf16/head_bf16_t/head_transpose는 제거됐다:
// head GEMV는 플레인 표준 경로 w4a16_gemv_bf16(_t)(행=블록, [n][k] 원본 —
// gptq4.cu)로 이관. 로드시 전치(head_transpose)도 불필요.
// 실측 근거(2026-10-10, RTX 4090): 27B head 2.55GB @951GB/s·35B 1.02GB
// @942GB/s — 양쪽 다 DRAM 벽(가중치 바이트가 하한). 종전 [k][n] 전치
// 판독(head_bf16)과 성능 동일 → 단순한 쪽(원본 레이아웃) 채택.
// 산술 계약: 종전 행별 k직렬(f32 mul·add 분리)에서 행=블록 GEMV의
// 레인 직렬(l,l+64,…) + f64 tree64로 변경 — 토큰 수준 판정(골든·smoke 10/10).

// [P3] 로짓 argmax — min-index-on-tie(CPU greedy_from 계약 미러: 엄격 비교
// v > best → 동률은 최저 인덱스, NaN은 비교 false로 순위 제외). 1블록 1024
// 스레드, float4 판독 + 공유 트리 리덕션(결정적). out[0] = u32 인덱스 —
// 그래프 캡처 가능(커널 1 + 4B d2h).
extern "C" __global__ void w4a16_argmax_min(
    const float* __restrict__ lg, int n, unsigned* __restrict__ out)
{
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    const int tid = threadIdx.x;
    float v = -INFINITY;
    unsigned ix = 0u;
    const int n4 = n >> 2;
    const float4* l4 = reinterpret_cast<const float4*>(lg);
    for (int i = tid; i < n4; i += 1024) {
        const float4 q = l4[i];
        const float w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (w[j] > v) {
                v = w[j];
                ix = (unsigned)(4 * i + j);
            }
        }
    }
    for (int i = 4 * n4 + tid; i < n; i += 1024) {
        if (lg[i] > v) {
            v = lg[i];
            ix = (unsigned)i;
        }
    }
    bv[tid] = v;
    bi[tid] = ix;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) {
            const float ov = bv[tid + st];
            const unsigned oi = bi[tid + st];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                bv[tid] = ov;
                bi[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = bi[0];
    }
}

// [A9 2026-10-10] 배치 argmax — 행당 1블록(grid=t), out[row]=인덱스.
// 행별 알고리즘·동률 규칙은 w4a16_argmax_min과 동일(비트동일).
extern "C" __global__ void w4a16_argmax_min_t(
    const float* __restrict__ lg, int n, int t, unsigned* __restrict__ out)
{
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    const int row = blockIdx.x;
    if (row >= t) {
        return;
    }
    const int tid = threadIdx.x;
    const float* lr = lg + (size_t)row * n;
    float v = -INFINITY;
    unsigned ix = 0u;
    const int n4 = n >> 2;
    const float4* l4 = reinterpret_cast<const float4*>(lr);
    for (int i = tid; i < n4; i += 1024) {
        const float4 q = l4[i];
        const float w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (w[j] > v) {
                v = w[j];
                ix = (unsigned)(4 * i + j);
            }
        }
    }
    for (int i = 4 * n4 + tid; i < n; i += 1024) {
        if (lr[i] > v) {
            v = lr[i];
            ix = (unsigned)i;
        }
    }
    bv[tid] = v;
    bi[tid] = ix;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) {
            const float ov = bv[tid + st];
            const unsigned oi = bi[tid + st];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                bv[tid] = ov;
                bi[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = bi[0];
    }
}
