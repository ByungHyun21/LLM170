#include "g4_common.cuh" // [I8head] shfl_down_f64(tree64) 공용
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

// [실측 기각 원장 — 2026-10-18 ⑮①/⑱: head 축 4연속 기각] (1) FMAD on
// (__fmaf_rn): SASS 명령 −20.8%(i8_t 44008→34864)에도 gemma4 tg −2.4%·pp −0.4%
// — 본 커널 = long_sb 74%(로드 지연) 바운드라 명령 수 비지배(⑥-a와 동형).
// (2) R=4 동거(256스레드·비트동일) = tg −3.4%·명명 배리어판 −3.9%.
// (3) R=4 + x smem 스테이징(x L2 재판독 262144블록×16KB ÷4 — ncu DRAM 14.4%·
// 3.7GB 로드/발사): 비트동일·양 모델 해시 동일에도 gemma4 ±0.0%·27B −1.2% =
// **x 트래픽도 병목 아님**(주 판독 = 가중 DRAM 스트림 지연). 잔여 후보 =
// tolerance-class 재결합(벡터·다중 acc — 골든/eval 게이트)뿐.
// ── [I8head 2026-10-11] int8 head GEMV — 품질 프로브(eval) 통과판 ──
// bf16 head는 디코드마다 2.54GB(27B)를 재판독(디코드의 14%) — int8 팩킹으로
// 절반(1.27GB+행스케일 1MB). eval chunk4: en PPL +0.061%·argmax 99.139% /
// ko +0.0002%·99.347%(KV int8 밴드 내) · 27B 프로브 골든 유지.
// 행별 스케일(amax/127) · 산술 = (Σ q·x)·s — 토큰 수준 판정(골든·smoke).
// (int4-g128 head는 기각: en +0.59%/92.7% — 4090.md.)
// [I8head t판 — 행=블록 확장형] t=1 행 커널과 동일 몸체에 T 루프 추가:
// 블록 64스레드가 한 행(o)의 웨이트를 1회만 읽고 t토큰에 재사용(amortization
// 유지). 512스레드 8행그룹 판(구현 이력의 코드젠 결함 — 진단 미규명)을
// 피해 검증된 행=블록 형태만 사용.
template <int T>
__device__ __forceinline__ void gemv_i8_trow_body(
    const signed char* __restrict__ w, const float* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int o,
    double* __restrict__ red)
{
    const int l = threadIdx.x;
    const unsigned* q32 = reinterpret_cast<const unsigned*>(w + (size_t)o * k);
    const float s = sc[o];
    const int k4 = k >> 2;
    float acc[T];
#pragma unroll
    for (int r = 0; r < T; ++r) {
        acc[r] = 0.0f;
    }
    // [T16-reg 2026-10-15] 행별 x 포인터 배열 제거 — T=16에서 REG 65→90 상승분의
    // 주 원인이었다(cuobjdump). 단일 베이스 + (r·k4) 오프셋 — 값·순서 불변.
    // 주의: 루프 인덱스 i가 l로 시작하므로 **베이스에 l을 더하지 않는다**
    // (초판 이중 가산 버그 — 골든 붕괴로 검출).
    const float4* xb = reinterpret_cast<const float4*>(x);
#pragma unroll 16
    for (int i = l; i < k4; i += 64) {
        const unsigned qw = q32[i];
        const float q0 = (float)(signed char)(qw & 0xFFu);
        const float q1 = (float)(signed char)((qw >> 8) & 0xFFu);
        const float q2 = (float)(signed char)((qw >> 16) & 0xFFu);
        const float q3 = (float)(signed char)(qw >> 24);
#pragma unroll
        for (int r = 0; r < T; ++r) {
            const float4 xv = xb[(size_t)r * k4 + i];
            acc[r] += q0 * xv.x;
            acc[r] += q1 * xv.y;
            acc[r] += q2 * xv.z;
            acc[r] += q3 * xv.w;
        }
    }
#pragma unroll
    for (int r = 0; r < T; ++r) {
        red[l] = (double)acc[r];
        __syncthreads();
        if (l < 32) {
            double v = red[l] + red[l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(v, off);
                if (l < off) {
                    v += oth;
                }
            }
            if (l == 0) {
                out[(size_t)r * n + o] = (float)v * s;
            }
        }
        __syncthreads();
    }
}

#define GEMV_I8_TROW_CASE(N) \
    case N: gemv_i8_trow_body<N>(w, sc, x, out, n, k, o, red); break;
extern "C" __global__ void w4a16_gemv_i8_t(
    const signed char* __restrict__ w, const float* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t)
{
    const int o = blockIdx.x;
    if (o >= n || t <= 0 || t > 16) {
        return;
    }
    __shared__ double red[64];
    switch (t) {
        GEMV_I8_TROW_CASE(1)
        GEMV_I8_TROW_CASE(2)
        GEMV_I8_TROW_CASE(3)
        GEMV_I8_TROW_CASE(4)
        GEMV_I8_TROW_CASE(5)
        GEMV_I8_TROW_CASE(6)
        GEMV_I8_TROW_CASE(7)
        GEMV_I8_TROW_CASE(8)
        GEMV_I8_TROW_CASE(9)
        GEMV_I8_TROW_CASE(10)
        GEMV_I8_TROW_CASE(11)
        GEMV_I8_TROW_CASE(12)
        GEMV_I8_TROW_CASE(13)
        GEMV_I8_TROW_CASE(14)
        GEMV_I8_TROW_CASE(15)
        GEMV_I8_TROW_CASE(16)
        default: break; // 호스트 계약 밖(t≤16)
    }
}

// [I8head t=1] 행=블록·64레인 — 검증된 주 경로(디코드 tg +7.6%/9.3% 실측).
extern "C" __global__ void w4a16_gemv_i8(
    const signed char* __restrict__ w, const float* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ out, int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    const unsigned* q32 = reinterpret_cast<const unsigned*>(w + (size_t)o * k);
    const float s = sc[o];
    const int k4 = k >> 2;
    const float4* xp = reinterpret_cast<const float4*>(x) + l;
    float acc = 0.0f;
#pragma unroll 16
    for (int i = l; i < k4; i += 64) {
        const unsigned qw = q32[i];
        const float4 xv = *xp;
        acc += (float)(signed char)(qw & 0xFFu) * xv.x;
        acc += (float)(signed char)((qw >> 8) & 0xFFu) * xv.y;
        acc += (float)(signed char)((qw >> 16) & 0xFFu) * xv.z;
        acc += (float)(signed char)(qw >> 24) * xv.w;
        xp += 64;
    }
    __shared__ double red[64];
    red[l] = (double)acc;
    __syncthreads();
    if (l < 32) {
        double v = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(v, off);
            if (l < off) {
                v += oth;
            }
        }
        if (l == 0) {
            out[o] = (float)v * s;
        }
    }
}
