// ─── CLIP ViT (plans/17) — f32 가중 GEMM + ViT 전용 EW ───

// f32 GEMM 타일: 256스레드 = 32토큰 × 8행 (1D 수동 분해). 그리드 gx=t/32, gy=n_out/8.
// 4와이드 전개 ILP — 실측 최적 (min-traffic 1토큰판보다 빠름: 지연 은닉 우위).
extern "C" __global__ void gemm_f32t(const float* x, const float* w, const float* b,
                                     float* out, int ni, int n_out, int t) {
    int tx = threadIdx.x & 31;
    int ty = threadIdx.x >> 5;
    int ti = blockIdx.x * 32 + tx;
    int o = (blockIdx.y + blockIdx.z * gridDim.y) * 8 + ty;
    if (ti >= t || o >= n_out) return;
    const float* xr = x + (size_t)ti * ni;
    const float* wr = w + (size_t)o * ni;
    float acc = b ? b[o] : 0.0f;
    int i = 0;
    for (; i + 4 <= ni; i += 4) {
        acc += xr[i] * wr[i] + xr[i+1] * wr[i+1] + xr[i+2] * wr[i+2] + xr[i+3] * wr[i+3];
    }
    for (; i < ni; i++) acc += xr[i] * wr[i];
    out[(size_t)ti * n_out + o] = acc;
}

// LayerNorm(w, b) [t][n] — in-place.
extern "C" __global__ void layernorm_t(float* x, const float* w, const float* b,
                                       int n, float eps, int t) {
    int ti = blockIdx.x;
    if (ti >= t) return;
    float* r = x + (size_t)ti * n;
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += r[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = r[i] - mean; var += d * d; }
    var /= n;
    float inv = rsqrtf(var + eps);
    for (int i = 0; i < n; i++) r[i] = (r[i] - mean) * inv * w[i] + b[i];
}

// 비전 M-RoPE — q/k [t][n_head][72]: 쌍 (i, i+36), i<18 → y, ≥18 → x.
extern "C" __global__ void vit_rope(float* qk, const int* yx, int n_head, int d_head,
                                    int t, int part_stride) {
    int ti = blockIdx.x;
    int h = blockIdx.y;
    if (ti >= t || h >= n_head) return;
    int half = d_head / 2;
    int base = ti * part_stride + h * d_head;
    int py = yx[ti * 2];
    int px = yx[ti * 2 + 1];
    for (int i = 0; i < half; i++) {
        int sec = i < half / 2 ? py : px;
        int local = i < half / 2 ? i : i - half / 2;
        // theta = sec · 10000^(-2/half)^local — cos/sin은 호스트 사전계산 테이블 재사용
        float theta = sec * __powf(10000.0f, -2.0f / half * local);
        float c = cosf(theta), s2 = sinf(theta);
        float x0 = qk[base + i];
        float x1 = qk[base + i + half];
        qk[base + i] = x0 * c - x1 * s2;
        qk[base + i + half] = x0 * s2 + x1 * c;
    }
}

// ViT flash attention v2 — 무게이트, d_head=72 특화.
// 분업: 스레드 72..255 = 점수 (스레드당 ~13 위치 직렬 도트, ILP),
// 스레드 0..71 = 가중 V 누적 (출력 차원당 1스레드). 셔플체인 제거.
extern "C" __global__ void flash_vit(const float* q, const float* k, const float* v,
                                     float* out, int n_past, int n_head, int d_head, float scale) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= gridDim.x || h >= n_head) return;
    __shared__ float sc[2304];
    __shared__ float red[8];
    __shared__ float acc[128];
    const int d = d_head; // 72
    int tid = threadIdx.x;
    const float* qp = q + (size_t)t * n_head * d + h * d;
    // A: 점수 — 184스레드 (72..255) 분담
    if (tid >= d) {
        int lane = tid - d;
        int nthr = blockDim.x - d; // 184
        for (int p = lane; p < n_past; p += nthr) {
            const float* kp = k + (size_t)p * n_head * d + h * d;
            float acc2 = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < d; i++) acc2 += qp[i] * kp[i];
            sc[p] = acc2 * scale;
        }
    }
    __syncthreads();
    // B: max/sum — 전 스레드 협력 (스트라이드)
    float lmax = -3.4028235e38f;
    for (int p = tid; p < n_past; p += blockDim.x) lmax = fmaxf(lmax, sc[p]);
    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        lmax = fmaxf(lmax, __shfl_down_sync(0xffffffffffffffffull, lmax, off));
    if ((tid & 31) == 0) red[tid >> 5] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = red[0];
        for (int j = 1; j < 8; j++) m = fmaxf(m, red[j]);
        red[0] = m;
    }
    __syncthreads();
    float mx = red[0];
    float lsum = 0.0f;
    for (int p = tid; p < n_past; p += blockDim.x) {
        float e = __expf(sc[p] - mx);
        sc[p] = e;
        lsum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        lsum += __shfl_down_sync(0xffffffffffffffffull, lsum, off);
    if ((tid & 31) == 0) red[tid >> 5] = lsum;
    __syncthreads();
    if (tid == 0) {
        float sm2 = red[0];
        for (int j = 1; j < 8; j++) sm2 += red[j];
        red[0] = 1.0f / sm2;
    }
    __syncthreads();
    float inv = red[0];
    // C: 가중 V 누적 — 72스레드 (차원당 1)
    if (tid < d) {
        float a = 0.0f;
        for (int p = 0; p < n_past; p++) {
            const float* vp = v + (size_t)p * n_head * d + h * d;
            a += sc[p] * vp[tid];
        }
        out[(size_t)t * n_head * d + h * d + tid] = a * inv;
    }
}

// GELU (tanh 근사) [t][n] — in-place.
extern "C" __global__ void gelu_t(float* x, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float xx = x[j];
    const float A = 0.044715f;
    const float SQ2OPI = 0.7978845608f;
    x[j] = 0.5f * xx * (1.0f + tanhf(SQ2OPI * xx * (1.0f + A * xx * xx)));
}

// y += x (요소별)
extern "C" __global__ void add_f32(float* y, const float* x, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    y[j] += x[j];
}

// src [t][stride] + off → dst [t][n] 팩
extern "C" __global__ void pack_strided(const float* src, float* dst, int n, int stride, int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y;
    if (j >= n || ti >= t) return;
    dst[(size_t)ti * n + j] = src[(size_t)ti * stride + j];
}


