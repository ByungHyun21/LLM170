// bf16 head GEMV — output.weight[n][k] bf16 비트 × xn f32 → logits f32.
// 비트 계약: core matmul::cpu의 행별 직렬 f32 내적(acc += x[i]*w[i], k
// 오름차순, mul·add 분리)과 동일 순서 — -fmad=false 필수.
// bf16→f32는 상위 16비트 시프트(deq.rs bf16_to_f32와 동일 — 정확).

__device__ __forceinline__ float b2f(unsigned short b) {
    return __uint_as_float(((unsigned)b) << 16);
}

// 스레드당 4행 — x[i]를 4행이 공유해 로드 수를 줄인다(행별 누산 순서 불변).
#define HEAD_ROWS 4
extern "C" __global__ void head_bf16(const unsigned short* __restrict__ w,
                                     const float* __restrict__ x,
                                     float* __restrict__ out,
                                     int n,
                                     int k)
{
    const int r0 = (blockIdx.x * blockDim.x + threadIdx.x) * HEAD_ROWS;
    if (r0 >= n) {
        return;
    }
    if (r0 + HEAD_ROWS <= n) {
        const unsigned short* w0 = w + (size_t)r0 * k;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float xv = x[i];
            a0 += xv * b2f(w0[i]);
            a1 += xv * b2f(w0[k + i]);
            a2 += xv * b2f(w0[2 * k + i]);
            a3 += xv * b2f(w0[3 * k + i]);
        }
        out[r0] = a0;
        out[r0 + 1] = a1;
        out[r0 + 2] = a2;
        out[r0 + 3] = a3;
        return;
    }
    for (int r = r0; r < n; ++r) {
        const unsigned short* wr = w + (size_t)r * k;
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) {
            acc += x[i] * b2f(wr[i]);
        }
        out[r] = acc;
    }
}
