// bf16 head — 로드시 전치([k][n]) + 응집 판독 GEMV.
// 비트 계약: core matmul::cpu의 행별 직렬 f32 내적(acc += x[i]*w[i], k
// 오름차순, mul·add 분리)과 동일 순서 — -fmad=false 필수.
// bf16→f32는 상위 16비트 시프트(deq.rs bf16_to_f32와 동일 — 정확).
//
// [2026-10-08 성능] 종전 n-major([n][k]) 커널은 스레드가 자기 행을 2B씩
// 걸어 워프가 매 스텝 32개 라인을 건드렸다(L1 스래싱, 실효 ~210GB/s).
// uint4 벡터화만으로는 부족(11.3ms 실측) — **로드시 1회 전치**로 [k][n]을
// 만들어 k스텝마다 연속 n을 응집 판독한다(스텝당 워프 256B 연속).
// 산술 순서는 불변 — 골든 대조가 판정(종전과 비트 동일이어야 한다).

__device__ __forceinline__ float b2f(unsigned short b) {
    return __uint_as_float(((unsigned)b) << 16);
}

// in[n][k] → out[k][n] (bf16). 1D 블록 1024 = 32×32 타일(smem 패딩 33).
// 판독·기록 모두 tx가 연속 → 응집. k·n 32배수 여부는 경계 가드로 일반화.
extern "C" __global__ void head_transpose(const unsigned short* __restrict__ in,
                                          unsigned short* __restrict__ out,
                                          int n,
                                          int k)
{
    __shared__ unsigned short t[32][33];
    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;
    const int x = blockIdx.x * 32 + tx; // k 성분
    const int y = blockIdx.y * 32 + ty; // n 성분
    if (x < k && y < n) {
        t[ty][tx] = in[(size_t)y * k + x];
    }
    __syncthreads();
    const int x2 = blockIdx.y * 32 + tx; // n 성분
    const int y2 = blockIdx.x * 32 + ty; // k 성분
    if (x2 < n && y2 < k) {
        out[(size_t)y2 * n + x2] = t[tx][ty];
    }
}

// 스레드당 8출력 — k스텝마다 uint4(bf16 8개 = 16B) 응집 판독, x는 브로드캐스트.
// 정렬: n%8==0일 때만 벡터 경로(출력 시작 오프셋 16B 정렬), 아니면 스칼라.
#define HEAD_OUTS 8

extern "C" __global__ void head_bf16(const unsigned short* __restrict__ wt, // [k][n]
                                     const float* __restrict__ x,          // [k]
                                     float* __restrict__ out,              // [n]
                                     int n,
                                     int k)
{
    const int r0 = (blockIdx.x * blockDim.x + threadIdx.x) * HEAD_OUTS;
    if (r0 >= n) {
        return;
    }
    if (r0 + HEAD_OUTS <= n && (n & (HEAD_OUTS - 1)) == 0) {
        float a[HEAD_OUTS];
#pragma unroll
        for (int u = 0; u < HEAD_OUTS; ++u) {
            a[u] = 0.0f;
        }
        for (int i = 0; i < k; ++i) {
            const float xv = x[i];
            const uint4 v =
                *reinterpret_cast<const uint4*>(wt + (size_t)i * n + r0);
            const unsigned short* p = reinterpret_cast<const unsigned short*>(&v);
#pragma unroll
            for (int u = 0; u < HEAD_OUTS; ++u) {
                a[u] += xv * b2f(p[u]);
            }
        }
#pragma unroll
        for (int u = 0; u < HEAD_OUTS; ++u) {
            out[r0 + u] = a[u];
        }
        return;
    }
    // 꼬리(8 미만 잔여) — 스칼라, 순서 동일.
    for (int r = r0; r < n; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) {
            acc += x[i] * b2f(wt[(size_t)i * n + r]);
        }
        out[r] = acc;
    }
}
