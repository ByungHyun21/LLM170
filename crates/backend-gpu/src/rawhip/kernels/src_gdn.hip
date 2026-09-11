// GDN conv1d + ring shift + silu (t=1)
extern "C" __global__ void gdn_conv(const float* qkv, const float* cw, float* state,
                                    float* out, int ch, int k) {
    // 병렬화 (부록78): 채널=스레드 — 구형 블록당 1스레드(1/64 활용) 폐지.
    // 채널별 산술·순서 동일 (비트불변).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    int xb = c; // t=0
    float sum = cw[c * k + (k - 1)] * qkv[xb];
    for (int j = 0; j < k - 1; j++) sum += cw[c * k + j] * state[j * ch + c];
    float oc = sum / (1.0f + exp_cr(-sum));
    for (int j = 0; j < k - 2; j++) state[j * ch + c] = state[(j + 1) * ch + c];
    state[(k - 2) * ch + c] = qkv[xb];
    out[xb] = oc;
}
// GDN conv1d + ring — t토큰 순차 (thread0/채널, blockIdx.y=t)
extern "C" __global__ void gdn_conv_t2(const float* qkv, const float* cw, float* state,
                                       float* out, int ch, int k, int t) {
    // 완전 병렬 (c × t): 인과 conv는 이전 '입력'만 참조(출력 아님) — 청크 내
    // 의존 없음. 가중 합 순서/식은 구형과 원소별 동일열 (비트무영향).
    // 상태 갱신: ti ∈ [t-(k-1), t) 각자 자기 슬롯만 기입 (경합 없음).
    // 전제 t >= k-1 (짧은 꼬리 청크는 구형 순차 커널 사용).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y;
    if (c >= ch || ti >= t) return;
    const float* row = qkv + (size_t)ti * ch;
    float sum = cw[c * k + (k - 1)] * row[c];
    for (int j = 0; j < k - 1; j++) {
        int pos = ti - (k - 1) + j;
        float xv = pos >= 0 ? qkv[(size_t)pos * ch + c] : state[(pos + (k - 1)) * ch + c];
        sum += cw[c * k + j] * xv;
    }
    float oc = sum / (1.0f + exp_cr(-sum));
    out[(size_t)ti * ch + c] = oc;
}
extern "C" __global__ void gdn_conv_t2_f32(const float* qkv, const float* cw, float* state,
                                       float* out, int ch, int k, int t) {
    // 완전 병렬 (c × t): 인과 conv는 이전 '입력'만 참조(출력 아님) — 청크 내
    // 의존 없음. 가중 합 순서/식은 구형과 원소별 동일열 (비트무영향).
    // 상태 갱신: ti ∈ [t-(k-1), t) 각자 자기 슬롯만 기입 (경합 없음).
    // 전제 t >= k-1 (짧은 꼬리 청크는 구형 순차 커널 사용).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y;
    if (c >= ch || ti >= t) return;
    const float* row = qkv + (size_t)ti * ch;
    float sum = cw[c * k + (k - 1)] * row[c];
    for (int j = 0; j < k - 1; j++) {
        int pos = ti - (k - 1) + j;
        float xv = pos >= 0 ? qkv[(size_t)pos * ch + c] : state[(pos + (k - 1)) * ch + c];
        sum += cw[c * k + j] * xv;
    }
    float oc = sum / (1.0f + __expf(-sum));
    out[(size_t)ti * ch + c] = oc;
}

// conv 상태 갱신 (conv_t2 이후 별도 런치 — 읽기/쓰기 경합 제거)
extern "C" __global__ void gdn_conv_state(const float* qkv, float* state, int ch, int k, int t) {
    int j = blockIdx.x;                 // 0..k-2
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= ch || j >= k - 1) return;
    state[j * ch + c] = qkv[(size_t)(t - (k - 1) + j) * ch + c];
}

extern "C" __global__ void gdn_conv_t(const float* qkv, const float* cw, float* state,
                                      float* out, int ch, int k, int t) {
    int c = blockIdx.x;
    int u = threadIdx.x;
    if (u != 0 || c >= ch) return;
    for (int ti = 0; ti < t; ti++) {
        const float* row = qkv + (size_t)ti * ch;
        float sum = cw[c * k + (k - 1)] * row[c];
        for (int j = 0; j < k - 1; j++) sum += cw[c * k + j] * state[j * ch + c];
        float oc = sum / (1.0f + exp_cr(-sum));
        for (int j = 0; j < k - 2; j++) state[j * ch + c] = state[(j + 1) * ch + c];
        state[(k - 2) * ch + c] = row[c];
        out[(size_t)ti * ch + c] = oc;
    }
}

// beta / e^g precompute
extern "C" __global__ void gdn_beta_g(const float* b, const float* a, const float* dtb,
                                      const float* sa, float* bg, int n_h, int dt_rank) {
    // n_h = t·dt_rank (배치) — dtb/sa는 토큰 공유 [dt_rank].
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_h) return;
    int h0 = h % dt_rank;
    float bv = b[h];
    bg[h * 2] = 1.0f / (1.0f + exp_cr(-bv));
    float x = fminf(a[h] + dtb[h0], 80.0f);
    float sp = log1p_cr(exp_cr(x));
    bg[h * 2 + 1] = exp_cr(sp * sa[h0]);
}
extern "C" __global__ void gdn_beta_g_f32(const float* b, const float* a, const float* dtb,
                                      const float* sa, float* bg, int n_h, int dt_rank) {
    // n_h = t·dt_rank (배치) — dtb/sa는 토큰 공유 [dt_rank].
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_h) return;
    int h0 = h % dt_rank;
    float bv = b[h];
    bg[h * 2] = 1.0f / (1.0f + __expf(-bv));
    float x = fminf(a[h] + dtb[h0], 80.0f);
    float sp = log1p_cr(__expf(x));
    bg[h * 2 + 1] = __expf(sp * sa[h0]);
}
// norm_gated silu (sequential 32-segment f64 — bit-identical to CPU ops)
extern "C" __global__ void norm_gated_silu(const float* o, const float* z, const float* w,
                                           float* out, float eps, int d, int n_h) {
    // 워프(32레인)당 1행 — 레인 u가 세그먼트 u의 순차 f32 부분합(기존과 동일
    // 경계), f64 결합은 전 레인이 p0..p31 동일 순서로 수행 (미러 산술열 보존).
    // 초과 세그먼트 부분합 0 가산은 IEEE 항등 — 비트무영향.
    int row = blockIdx.y * n_h + blockIdx.x;
    int u = threadIdx.x;
    if (blockIdx.x >= n_h) return;
    int xb = row * d;
    int wb = blockIdx.x * d;
    int chunk = (d + 31) >> 5;
    float part = 0.0f;
    {
        int lo = u * chunk;
        if (lo < d) {
            int hi = min(lo + chunk, d);
            for (int i = lo; i < hi; i++) {
                float dv = o[xb + i];
                part += dv * dv;
            }
        }
    }
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part, j);
        sum += (double)pj;
    }
    float scale32 = (float)sqrt(sum / (double)d + (double)eps);
    float inv = 1.0f / scale32;
    for (int i = u; i < d; i += 32) {
        float nrm = o[xb + i] * inv * w[wb + i];
        float zz = z[xb + i];
        out[xb + i] = nrm * (zz / (1.0f + exp_cr(-zz)));
    }
}
extern "C" __global__ void norm_gated_silu_f32(const float* o, const float* z, const float* w,
                                           float* out, float eps, int d, int n_h) {
    // 워프(32레인)당 1행 — 레인 u가 세그먼트 u의 순차 f32 부분합(기존과 동일
    // 경계), f64 결합은 전 레인이 p0..p31 동일 순서로 수행 (미러 산술열 보존).
    // 초과 세그먼트 부분합 0 가산은 IEEE 항등 — 비트무영향.
    int row = blockIdx.y * n_h + blockIdx.x;
    int u = threadIdx.x;
    if (blockIdx.x >= n_h) return;
    int xb = row * d;
    int wb = blockIdx.x * d;
    int chunk = (d + 31) >> 5;
    float part = 0.0f;
    {
        bool vec = (d >= 128) && ((d & 127) == 0);
        int lo = vec ? (u << 2) : (u * chunk);
        if (lo < d) {
            if (vec) {
                float4 v = *reinterpret_cast<const float4*>(o + xb + lo);
                float a0 = v.x; part += a0 * a0;
                float a1 = v.y; part += a1 * a1;
                float a2 = v.z; part += a2 * a2;
                float a3 = v.w; part += a3 * a3;
            } else {
                int hi = min(lo + chunk, d);
                for (int i = lo; i < hi; i++) {
                    float dv = o[xb + i];
                    part += dv * dv;
                }
            }
        }
    }
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part, j);
        sum += (double)pj;
    }
    float scale32 = (float)sqrt(sum / (double)d + (double)eps);
    float inv = 1.0f / scale32;
    if ((d >= 128) && ((d & 127) == 0)) {
        // 원소별 독립 — 레인당 연속 4원소 float4 경로 (기존 스트라이드 매핑과 동일 값)
        int i = u << 2;
        float4 ov = *reinterpret_cast<const float4*>(o + xb + i);
        float4 zv = *reinterpret_cast<const float4*>(z + xb + i);
        float4 wv = *reinterpret_cast<const float4*>(w + wb + i);
        float4 r;
        float n0 = ov.x * inv * wv.x, z0 = zv.x; r.x = n0 * (z0 / (1.0f + __expf(-z0)));
        float n1 = ov.y * inv * wv.y, z1 = zv.y; r.y = n1 * (z1 / (1.0f + __expf(-z1)));
        float n2 = ov.z * inv * wv.z, z2 = zv.z; r.z = n2 * (z2 / (1.0f + __expf(-z2)));
        float n3 = ov.w * inv * wv.w, z3 = zv.w; r.w = n3 * (z3 / (1.0f + __expf(-z3)));
        *reinterpret_cast<float4*>(out + xb + i) = r;
        return;
    }
    for (int i = u; i < d; i += 32) {
        float nrm = o[xb + i] * inv * w[wb + i];
        float zz = z[xb + i];
        out[xb + i] = nrm * (zz / (1.0f + __expf(-zz)));
    }
}
// norm_gated_silu+quant 융합 (t=1, d%1024==0) — 축소 산술은 원본과
// 동일열(세그먼트 f32합→셔플 f64 순서합), 원소별 nrm·silu(z)는 레지스터
// 통과 후 32블록 quant. 워프=1행 그대로.
extern "C" __global__ void gatedq(const float* o, const float* z, const float* w,
                                  unsigned* xq, float eps, int d, int n_h, int n_tot) {
    // 행 = d원소 (호출축: d=d_state, grid.x=dt_rank) — 워프당 1행:
    // 세그먼트 f32합→셔플 f64 순서합 (원본 norm_gated_silu와 동일열),
    // 이후 행 내 d/32개 32블록 quant (레인 lb<d/32).
    int row = blockIdx.y * n_h + blockIdx.x;
    int u = threadIdx.x;
    if (blockIdx.x >= n_h) return;
    int xb = row * d;
    int wb = blockIdx.x * d;
    int chunk = (d + 31) >> 5;
    float part = 0.0f;
    {
        int lo = u * chunk;
        if (lo < d) {
            int hi = min(lo + chunk, d);
            for (int i = lo; i < hi; i++) {
                float dv = o[xb + i];
                part += dv * dv;
            }
        }
    }
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part, j);
        sum += (double)pj;
    }
    float scale32 = (float)sqrt(sum / (double)d + (double)eps);
    float inv = 1.0f / scale32;
    int nblk = d >> 5;              // 행 내 블록 수 (d_state=128 → 4)
    int gb0 = xb >> 5;              // 플랫 xq 기준 글로벌 블록 베이스
    int nwords = n_tot >> 2;
    int nblk_tot = n_tot >> 5;
    for (int lb = u; lb < nblk; lb += 32) {
        int base = lb << 5;
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            float nrm = o[xb + base + i] * inv * w[wb + base + i];
            float zz = z[xb + base + i];
            xv[i] = nrm * (zz / (1.0f + exp_cr(-zz)));
        }
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xv[i]));
        float dd = amax / 127.0f;
        float id = dd != 0.0f ? 1.0f / dd : 0.0f;
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
            xq[(gb0 + lb) * 8 + wi] = word;
            if (wi < 4) qs0 = dot4(0x01010101u, word, qs0);
            else qs1 = dot4(0x01010101u, word, qs1);
        }
        xq[nwords + gb0 + lb] = __float_as_uint(dd);
        xq[nwords + nblk_tot + 2 * (gb0 + lb)] = (unsigned)qs0;
        xq[nwords + nblk_tot + 2 * (gb0 + lb) + 1] = (unsigned)qs1;
    }
}

// GDN AR state update (cube = (b,h) pair, unit = dv column)
// GDN AR — t토큰 순차 (block=128=d_state 열, blockIdx.x=pair)
extern "C" __global__ void gdn_ar_t(float* s, const float* q, const float* k, const float* v,
                                    const float* beta_ge, float* out, int d, int k_stride,
                                    int v_stride, int h_v, int h_k, float scale, int t) {
    __shared__ float ss[128 * 64];
    __shared__ float ks[128], qs[128], vs[64];
    int pair = blockIdx.x;
    int lu = threadIdx.x;
    int u = lu + blockIdx.y * 64;
    if (u >= d) return;
    int base_s = pair * d * d;
    // 스레드 lu는 자기 열(128원소) 전체 적재 — 이전 버그: 2원소만 적재
    #pragma unroll 8
    for (int j = 0; j < d; j++) ss[j * 64 + lu] = s[base_s + u * d + j];  // 전치
    for (int ti = 0; ti < t; ti++) {
        int h = pair % h_v;
        int kh = h % h_k;
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
        for (int e = lu; e < d; e += 64) { ks[e] = k[qk0 + e]; qs[e] = q[qk0 + e]; }
        __syncthreads();
        float sk = 0.0f;
        for (int kdim = 0; kdim < d; kdim++) {
            float sv = ss[kdim * 64 + lu] * g_exp;
            ss[kdim * 64 + lu] = sv;
            sk += sv * ks[kdim];
        }
        float delta = (v[v0 + u] - sk) * beta;
        // 업데이트+출력 융합 — 원소 순서·식 동일 (비트무영향)
        float o = 0.0f;
        for (int kdim = 0; kdim < d; kdim++) {
            float sv = ss[kdim * 64 + lu] + ks[kdim] * delta;
            ss[kdim * 64 + lu] = sv;
            o += (sv * qs[kdim]) * scale;
        }
        out[v0 + u] = o;
        __syncthreads();
    }
    #pragma unroll 8
    for (int j = 0; j < d; j++) s[base_s + u * d + j] = ss[j * 64 + lu];  // 전치
}

// GDN AR 워프 분할 (fast 실험) — wave32당 1열, 레인이 kdim 4개 분담,
// sk/o는 셔플 트리 환원 (순차 순서 아님 — 스트림 계약).
extern "C" __global__ void gdn_ar_w_swap(float* s, const float* q, const float* k, const float* v,
                                   const float* beta_ge, float* out, int d, int k_stride,
                                   int v_stride, int h_v, int h_k, float scale, int t) {
    // (ks/qs shared 스테이징 제거: 연산이 레인-로컬이라 크로스레인 동기 불필요 — 셔플 트리만으로 충분)
    int u = blockIdx.x;      // 축스왑 (부록88): u블록 인접발사 → k/q L2 국소성
    int pair = blockIdx.y;                     // 열 0..127
    int lane = threadIdx.x;                 // 0..31 — kdim 4개씩
    int base_s = pair * d * d;
    // 내 레인의 kdim: lane*4+j (레인당 4)
    float ssr[4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        ssr[j] = s[base_s + u * d + ((lane << 2) + j)];  // 전치
    for (int ti = 0; ti < t; ti++) {
        int h = pair % h_v;
        int kh = h % h_k;
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
        // k/q 직접 인덱싱 (레인-로컬 — shared 왕복·동기 제거)
        // sk 부분합 + 트리 환원
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            ssr[j] *= g_exp;
            part += ssr[j] * k[qk0 + (lane << 2) + j];
        }
        // 트리 환원 (shfl_sync 기반 — 검증된 프리미티브, xor 마스크 회피)
        float sk = part;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, sk, (lane ^ off) & 31);
            sk += pj;
        }
        float delta = (v[v0 + u] - sk) * beta;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            ssr[j] += k[qk0 + (lane << 2) + j] * delta;
        float op = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            op += ssr[j] * q[qk0 + (lane << 2) + j];
        float otot = op;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, otot, (lane ^ off) & 31);
            otot += pj;
        }
        if (lane == 0) out[v0 + u] = otot * scale;
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
        s[base_s + u * d + ((lane << 2) + j)] = ssr[j];  // 전치
}

// ══ AR 청크 스캔 (세션 3 부록 72) — t 직렬을 ch 청크로 병렬화 ══
// A: 청크 로컬 스캔 (제로 초기) + L_end(lend)/PG(pgb)/P 접두곱(pbuf) 기록.
extern "C" __global__ void gdn_ar_w(float* s, const float* q, const float* k, const float* v,
                                   const float* beta_ge, float* out, int d, int k_stride,
                                   int v_stride, int h_v, int h_k, float scale, int t) {
    // (ks/qs shared 스테이징 제거: 연산이 레인-로컬이라 크로스레인 동기 불필요 — 셔플 트리만으로 충분)
    int pair = blockIdx.x;
    int u = blockIdx.y;                     // 열 0..127
    int lane = threadIdx.x;                 // 0..31 — kdim 4개씩
    int base_s = pair * d * d;
    // 내 레인의 kdim: lane*4+j (레인당 4)
    float ssr[4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        ssr[j] = s[base_s + u * d + ((lane << 2) + j)];  // 전치
    for (int ti = 0; ti < t; ti++) {
        int h = pair % h_v;
        int kh = h % h_k;
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
        // k/q 직접 인덱싱 (레인-로컬 — shared 왕복·동기 제거)
        // sk 부분합 + 트리 환원
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            ssr[j] *= g_exp;
            part += ssr[j] * k[qk0 + (lane << 2) + j];
        }
        // 트리 환원 (shfl_sync 기반 — 검증된 프리미티브, xor 마스크 회피)
        float sk = part;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, sk, (lane ^ off) & 31);
            sk += pj;
        }
        float delta = (v[v0 + u] - sk) * beta;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            ssr[j] += k[qk0 + (lane << 2) + j] * delta;
        float op = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            op += ssr[j] * q[qk0 + (lane << 2) + j];
        float otot = op;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, otot, (lane ^ off) & 31);
            otot += pj;
        }
        if (lane == 0) out[v0 + u] = otot * scale;
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
        s[base_s + u * d + ((lane << 2) + j)] = ssr[j];  // 전치
}

// ══ AR 청크 스캔 (세션 3 부록 72) — t 직렬을 ch 청크로 병렬화 ══
// A: 청크 로컬 스캔 (제로 초기) + L_end(lend)/PG(pgb)/P 접두곱(pbuf) 기록.
extern "C" __global__ void gdn_ar_chunk_a(float* s, const float* q, const float* k, const float* v,
                                   const float* beta_ge, float* out, int d, int k_stride,
                                   int v_stride, int h_v, int h_k, float scale, int t, int ch,
                                   float* lend, float* pgb, float* pbuf) {
    int pair = blockIdx.x;
    int u = blockIdx.y;
    int c = blockIdx.z;
    int npair = gridDim.x;
    int lane = threadIdx.x;
    int c0 = c * ch;
    int c1 = min(c0 + ch, t);
    float ssr[4];
    float pg[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { ssr[j] = 0.0f; pg[j] = 1.0f; }
    for (int ti = c0; ti < c1; ti++) {
        int h = pair % h_v;
        int kh = h % h_k;
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            ssr[j] *= g_exp;
            pg[j] *= g_exp;
            part += ssr[j] * k[qk0 + (lane << 2) + j];
        }
        float sk = part;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, sk, (lane ^ off) & 31);
            sk += pj;
        }
        float delta = (v[v0 + u] - sk) * beta;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            ssr[j] += k[qk0 + (lane << 2) + j] * delta;
        float op = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++)
            op += ssr[j] * q[qk0 + (lane << 2) + j];
        float otot = op;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, otot, (lane ^ off) & 31);
            otot += pj;
        }
        if (lane == 0) out[v0 + u] = otot * scale;
        // P 접두곱: pbuf[c][pair][ti-c0][j] — u==0 블록만 기록
        if (u == 0) {
            float* pd = pbuf + (((size_t)c * npair + pair) * ch + (ti - c0)) * d;
            #pragma unroll
            for (int j = 0; j < 4; j++) pd[(lane << 2) + j] = pg[j];
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
        lend[((size_t)c * npair + pair) * d * d + u * d + ((lane << 2) + j)] = ssr[j];  // 전치
    if (u == 0) {
        float* pgs = pgb + ((size_t)c * npair + pair) * d;
        #pragma unroll
        for (int j = 0; j < 4; j++) pgs[(lane << 2) + j] = pg[j];
    }
    (void)s;
}

// B: 캐리 전파 — s_start[c] 저장 후 최종 상태 s에 기록.
// 스레드 (pair, j)당 u열 루프? → 그리드 (npair), 128스레드 = j, u는 내부 128 루프.
extern "C" __global__ void gdn_ar_chunk_b(float* s, const float* lend, const float* pgb,
                                          float* sstart, int d, int t, int ch, int npair, int nc) {
    int pair = blockIdx.x;
    int j = threadIdx.x;
    if (j >= d) return;
    // 컬럼 버퍼 (레지스터 대신 로컬 배열 — d=128 가정, 128 f32/스레드는 과함 →
    // u-블록 재병렬화: 그리드 (npair, 8), 16스레드=j청크? 단순화: 스레드당 u 8개씩.
    // 여기선 순차 c 루프 × u 128 (컴파일러 언롤) — 병목 아님 (소형).
    float col[128];
    for (int u2 = 0; u2 < d; u2++)
        col[u2] = s[(size_t)pair * d * d + u2 * d + j];  // 전치
    for (int c = 0; c < nc; c++) {
        // s_start[c] 기록
        float* ss = sstart + ((size_t)c * npair + pair) * d * d;  // 전치: [u][j] 열
        for (int u2 = 0; u2 < d; u2++) ss[u2 * d + j] = col[u2];
        float g = pgb[((size_t)c * npair + pair) * d + j];
        const float* le = lend + ((size_t)c * npair + pair) * d * d;  // 전치
        for (int u2 = 0; u2 < d; u2++) col[u2] = col[u2] * g + le[u2 * d + j];
    }
    // 최종 상태 기록
    float* sf = s + (size_t)pair * d * d;
    for (int u2 = 0; u2 < d; u2++) sf[u2 * d + j] = col[u2];
}

// AR 배치 smem 스테이징 (부록82): 블록=pair, 128스레드=u, 상태 64KB 동적
// smem + k/q 층마다 1회 스테이징 — u블록 128중 k/q 재독 제거.
// 산술: 스레드(u)별 kdim 순차 환원 — gdn_ar_w와 원소 동일순서 (비트계약
// 대상 아님: _w 계열 셔플 환원과 동일 부류).
extern "C" __global__ void gdn_ar_sm(float* s, const float* q, const float* k, const float* v,
                                   const float* beta_ge, float* out, int d, int k_stride,
                                   int v_stride, int h_v, int h_k, float scale, int t) {
    // 부록82 수정판: 블록=(pair, u절반 64) — 상태 [64][128] f32 = 32KB 정적.
    __shared__ float st[64][128];
    __shared__ float ks_sh[128], qs_sh[128];
    int pair = blockIdx.x;
    int uh = blockIdx.y;              // 0..1
    int u0 = uh * 64;
    int lu = threadIdx.x;            // 0..63
    int u = u0 + lu;
    int h = pair % h_v;
    int kh = h % h_k;
    int base_s = pair * d * d;
    // 상태 적재 (전치행)
    for (int j = 0; j < d; j++) st[lu][j] = s[base_s + u * d + j];
    __syncthreads();
    for (int ti = 0; ti < t; ti++) {
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
        if (lu < 64 && uh == 0) { ks_sh[lu] = k[qk0 + lu]; qs_sh[lu] = q[qk0 + lu]; }
        __syncthreads();
        float sk = 0.0f;
        for (int j = 0; j < d; j++) sk += st[lu][j] * ks_sh[j];
        float delta = (v[v0 + u] - sk) * beta;
        float o = 0.0f;
        for (int j = 0; j < d; j++) {
            float sv = st[lu][j] * g_exp + ks_sh[j] * delta;
            st[lu][j] = sv;
            o += sv * qs_sh[j];
        }
        out[v0 + u] = o * scale;
        __syncthreads();
    }
    for (int j = 0; j < d; j++) s[base_s + u * d + j] = st[lu][j];
}

// C2: S f16 smem 스테이징 (32KB) — S 재독 제거 (부록 72 처방).
// 그리드 (npair, nc), 블록 128 = u열.
extern "C" __global__ void gdn_ar_chunk_c2(float* out, const float* q, const float* pbuf,
                                          const float* sstart, int d, int k_stride, int v_stride,
                                          int h_v, int h_k, float scale, int t, int ch, int npair) {
    int pair = blockIdx.x;
    int c = blockIdx.y;
    int t0 = c * ch;
    __shared__ half S[128][128];
    const float* ssc = sstart + ((size_t)c * npair + pair) * d * d;
    for (int i = threadIdx.x; i < d * d; i += blockDim.x)
        S[i / d][i % d] = __float2half(ssc[i]);
    __syncthreads();
    int u = threadIdx.x;
    int h = pair % h_v;
    int kh = h % h_k;
    for (int r = 0; r < ch; r++) {
        int tt2 = t0 + r;
        if (tt2 >= t) return;
        int qk0 = tt2 * k_stride + kh * d;
        const float* pd = pbuf + (((size_t)c * npair + pair) * ch + r) * d;
        float acc = 0.0f;
        for (int j = 0; j < d; j++)
            acc += __half2float(S[j][u]) * (q[qk0 + j] * pd[j]);
        out[tt2 * v_stride + h * d + u] += acc * scale;
    }
}


// L2 norm rows (sequential f64 — l2_rows arithmetic) + scale (q only)
extern "C" __global__ void l2_rows2_scale(float* gq, float* gk, float eps, float scale,
                                          int d, int n_group) {
    // gy=t 배치 — gq/gk 각각 [t][n_group·d] 분리 버퍼.
    int y = blockIdx.y;
    int x = blockIdx.x;
    int u = threadIdx.x;
    if (u != 0 || x >= 2 * n_group) return;
    bool is_q = x < n_group;
    int tb = y * n_group * d;
    int xb = is_q ? tb + x * d : tb + (x - n_group) * d;
    float* g = is_q ? gq : gk;
    float sum = 0.0f;  // 순차 f32 — 미러 l2_norm과 쌍
    for (int i = 0; i < d; i++) {
        float dv = g[xb + i];
        sum += dv * dv;
    }
    float scale32 = sqrtf(sum);
    float inv = 1.0f / fmaxf(scale32, eps);
    if (is_q) {
        for (int i = 0; i < d; i++) g[xb + i] = g[xb + i] * inv;
    } else {
        for (int i = 0; i < d; i++) g[xb + i] = g[xb + i] * inv;
    }
}
// KV append 배치 — gy=t, 고정 src 행 → 캐시 pos0+y 행.
extern "C" __global__ void kv_append_t(const float* src, float* dst, int n, int pos0) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (j >= n) return;
    dst[(size_t)(pos0 + y) * n + j] = src[(size_t)y * n + j];
}

// 3-way split (q/k/v)
extern "C" __global__ void split3(const float* src, float* d0, float* d1, float* d2,
                                   int n0, int n1, int n2) {
    // 배치 — j는 [t][n0+n1+n2] 전역: 토큰 내 위치로 분해.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n0 + n1 + n2;
    // ew_l는 gx 전체 범위 런치 — 그리드 초과분 없음 (과잉 스레드는 아래 분기 밖)
    int tok = j / total;
    int jl = j - tok * total;
    if (jl < n0) d0[tok * n0 + jl] = src[j];
    else if (jl < n0 + n1) d1[tok * n1 + jl - n0] = src[j];
    else d2[tok * n2 + jl - n0 - n1] = src[j];
}


// L2 rows (워프 병렬 fast) — 32레인×4원소 세그먼트 부분합 + 레인 순서 결합.
// 순차 미러와 f32 그룹핑 다름 (스트림 계약 — EXACT는 구형 유지).
extern "C" __global__ void l2_rows2_scale_w(float* gq, float* gk, float eps, float scale,
                                            int d, int n_group) {
    int y = blockIdx.y;
    int x = blockIdx.x;
    int lane = threadIdx.x;
    if (x >= 2 * n_group) return;
    bool is_q = x < n_group;
    int tb = y * n_group * d;
    int xb = is_q ? tb + x * d : tb + (x - n_group) * d;
    float* g = is_q ? gq : gk;
    // 세그먼트 부분합 (레인당 4)
    float part = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float dv = g[xb + (lane << 2) + j];
        part += dv * dv;
    }
    // 레인 순서 결합
    float sum = 0.0f;
    #pragma unroll
    for (int l2 = 0; l2 < 32; l2++) {
        float pj = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part, l2);
        sum += pj;
    }
    float scale32 = sqrtf(sum);
    float inv = 1.0f / fmaxf(scale32, eps);
    #pragma unroll
    for (int j = 0; j < 4; j++)
        g[xb + (lane << 2) + j] = g[xb + (lane << 2) + j] * inv;
}

