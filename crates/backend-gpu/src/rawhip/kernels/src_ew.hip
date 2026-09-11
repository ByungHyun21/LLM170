// ─── ew 계열 (큐브cl ew.rs 산술 이식) ───
extern "C" __global__ void silu_mul_f32(const float* g, const float* u, float* out, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    out[j] = (v / (1.0f + __expf(-v))) * u[j];
}

extern "C" __global__ void silu_mul(const float* g, const float* u, float* out, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    out[j] = (v / (1.0f + exp_cr(-v))) * u[j];
}
// silu_mul+quant 융합 (t=1) — 동일 산술: v/(1+exp(-v))·u 레지스터 통과 후
// 32블록 quant. 레인당 quant 블록 1개 (64스레드×32 = n%2048==0 사용).
extern "C" __global__ void silu_mulq(const float* g, const float* u, unsigned* xq, int n) {
    int lb = blockIdx.x * blockDim.x + threadIdx.x;
    if (lb >= (n >> 5)) return;
    int base = lb << 5;
    float xv[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = g[base + i];
        xv[i] = (v / (1.0f + exp_cr(-v))) * u[base + i];
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xv[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    int nwords = n >> 2;
    int qs0 = 0, qs1 = 0;
    #pragma unroll
    for (int wi = 0; wi < 8; wi++) {
        unsigned word = 0u;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float v = xv[wi * 4 + k] * id;
            float r = v >= 0.0f ? (float)(int)(v + 0.5f)
                                : -((float)(int)(0.5f - v));
            float c = r > 127.0f ? 127.0f : (r < -127.0f ? -127.0f : r);
            word |= (((unsigned)(int)c) & 0xFFu) << (k * 8);
        }
        xq[lb * 8 + wi] = word;
        if (wi < 4) qs0 = dot4(0x01010101u, word, qs0);
        else qs1 = dot4(0x01010101u, word, qs1);
    }
    xq[nwords + lb] = __float_as_uint(d);
    xq[nwords + (n >> 5) + 2 * lb] = (unsigned)qs0;
    xq[nwords + (n >> 5) + 2 * lb + 1] = (unsigned)qs1;
}

extern "C" __global__ void axpy_scaled(float* y, const float* x, const float* s, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    y[j] = y[j] + x[j] * s[0];
}
extern "C" __global__ void copy_rows(const float* src, float* dst, int src_off, int dst_off, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    dst[dst_off + j] = src[src_off + j];
}
// rms 2-커널 (part: 32유닛 세그 f64, finish: 순차 결합+스케일)
extern "C" __global__ void rms_part(const float* x, double* part, int n) {
    int row = blockIdx.x;
    int u = threadIdx.x;
    if (u >= 32) return;
    int chunk = (n + 31) >> 5;
    int lo = u * chunk;
    if (lo >= n) { part[row * 32 + u] = 0.0; return; }
    int hi = min(lo + chunk, n);
    const float* xr = x + (size_t)row * n;
    float acc = 0.0f;  // f32 세그먼트 (f64 1/16 레이트) — 미러 sq_sum과 쌍
    int i = lo;
    // 4원소 벡터 로드 — 가산 순서는 스칼라판과 동일(x,y,z,w). n%4!=0이면 스칼라만.
    if ((n & 3) == 0) {
        for (; i + 4 <= hi; i += 4) {
            float4 v = *reinterpret_cast<const float4*>(xr + i);
            float d0 = v.x; acc += d0 * d0;
            float d1 = v.y; acc += d1 * d1;
            float d2 = v.z; acc += d2 * d2;
            float d3 = v.w; acc += d3 * d3;
        }
    }
    for (; i < hi; i++) {
        float dv = xr[i];
        acc += dv * dv;
    }
    part[row * 32 + u] = (double)acc;
}
extern "C" __global__ void rms_finish(const float* x, const float* w, const double* part,
                                      float* out, float eps, int n, int w_reps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    // 모든 스레드가 inv 직접 계산 (shared 없음 — 수치 경로 원본과 동일)
    double sum = 0.0;
    #pragma unroll
    for (int u2 = 0; u2 < 32; u2++) {
        if (u2 * ((n + 31) >> 5) < n) sum += part[row * 32 + u2];
    }
    float scale32 = (float)sqrt(sum / (double)n + (double)eps);
    float inv = 1.0f / scale32;
    if ((n & 3) != 0) {
        // n%4!=0 폴백 — 원본 매핑 유지 (벡터 경로 정렬 불가)
        int schunk = (n + 31) >> 5;
        int lo = (tid % 32) * schunk;
        if (lo < n) {
            int hi = min(lo + schunk, n);
            int g = tid >> 5;
            int gs = (hi - lo + 7) >> 3;
            int l2 = lo + g * gs;
            int h2 = min(l2 + gs, hi);
            for (int i = l2; i < h2; i++) out[(size_t)row * n + i] = x[(size_t)row * n + i] * inv * w[(size_t)(row % w_reps) * n + i];
        }
        return;
    }
    // 원소별 독립 — float4 스트라이드(완전 코얼레스). 곱 결합은 (x*inv)*w 유지.
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)row * n);
    const float4* w4 = reinterpret_cast<const float4*>(w + (size_t)(row % w_reps) * n);
    float4* o4 = reinterpret_cast<float4*>(out + (size_t)row * n);
    int n4 = n >> 2;
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 vx = x4[i];
        float4 vw = w4[i];
        float4 r;
        r.x = vx.x * inv * vw.x;
        r.y = vx.y * inv * vw.y;
        r.z = vx.z * inv * vw.z;
        r.w = vx.w * inv * vw.w;
        o4[i] = r;
    }
}
// q/k norm+rope (f64 중간 — FMA 수축 면역, qk_norm_rope 이식)
extern "C" __global__ void qk_norm_rope(float* xq, float* xk, const float* qw, const float* kw,
                                        const float* cs, float eps, float kqs, int pos,
                                        int n_head, int n_kv, int hd, int n_rot) {
    // gy=t 배치 — aq [t][n_head·2hd] (q‖gate 인터리브), ak [t][n_kv·hd].
    // 32레인 협력 (2026-09-05: 구버전은 블록당 1스레드 — 11.75ms/청크 병목).
    // 비트보존: 각 레인이 shfl로 32 세그먼트를 전부 모아 **동일 순서** f64 합산
    // (중복 연산이나 결과 동일), 스케일/로프는 원소별 독립라 순서 무관.
    int r0 = blockIdx.x;
    int u = threadIdx.x;   // 0..31
    int y = blockIdx.y;
    if (r0 >= n_head + n_kv) return;
    bool is_q = r0 < n_head;
    int half = n_rot >> 1;
    int csbase = (pos + y) * half * 2;
    int row_base = is_q ? y * (n_head * 2 * hd) + r0 * 2 * hd
                        : y * (n_kv * hd) + (r0 - n_head) * hd;
    float* xv = is_q ? xq : xk;
    const float* wv = is_q ? qw + r0 * hd : kw + (r0 - n_head) * hd;
    int chunk = (hd + 31) >> 5;
    // 레인 u가 세그먼트 u 계산
    float mypart = 0.0f;
    {
        int lo = u * chunk;
        int hi = min(lo + chunk, hd);
        for (int i = lo; i < hi; i++) {
            float dv = xv[row_base + i];
            mypart += dv * dv;
        }
    }
    // 전 레인이 32 세그먼트를 shfl로 수집 후 동일 순서 f64 합 (비트동일)
    float parts[32];
    #pragma unroll
    for (int uu = 0; uu < 32; uu++)
        parts[uu] = __shfl_sync(0xFFFFFFFFFFFFFFFFull, mypart, uu);
    double sum = 0.0;
    #pragma unroll
    for (int uu = 0; uu < 32; uu++) sum += parts[uu];
    float scale = 1.0f / (float)sqrt(sum / (double)hd + (double)eps);
    for (int i = u; i < hd; i += 32)
        xv[row_base + i] = xv[row_base + i] * scale * wv[i] * (is_q ? 1.0f : kqs);
    __syncwarp();
    for (int p = u; p < half; p += 32) {
        double c = (double)cs[csbase + p * 2];
        double sf = (double)cs[csbase + p * 2 + 1];
        int a = row_base + p, b = a + half;
        double x0 = (double)xv[a], x1 = (double)xv[b];
        xv[a] = (float)(x0 * c - x1 * sf);
        xv[b] = (float)(x0 * sf + x1 * c);
    }
}


