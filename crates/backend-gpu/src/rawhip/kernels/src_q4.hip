// ─── qwen4exp 값 경로 커널 (q4acc) — 2026-09-12 ───
// 계약: core 미러(quant.rs dot_row_w4a8_* / ops.rs)와 동일 산술 계열.
// 정수 내적은 정확, 그룹 스케일 적용·f64 부분합 순서만 CPU와 다르다
// (문서화된 ≤2.4e-7 계열). 활성 xq = [n/4 워드][n/32 f32 스케일].

// q5_1 (블록 32 = d(2) m(2) qh(4) qs(16), 24B). value = d·q + m (q 무부호 5bit).
// q = lo4 + 16·hi1 로 분해 → dot4 2회 + Σxq 1회. 그룹 스케일 d/m 은 f16.
extern "C" __global__ void q4_gemm_q5_1(const unsigned* xq, const unsigned* w,
                                        double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 24;
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 24;
        float d = f16w(w, wb);
        float mn = f16w(w, wb + 2);
        unsigned qh = w[(wb + 4) >> 2];
        unsigned qsw0 = w[(wb + 8) >> 2];
        unsigned qsw1 = w[(wb + 12) >> 2];
        unsigned qsw2 = w[(wb + 16) >> 2];
        unsigned qsw3 = w[(wb + 20) >> 2];
        int xw = (sb << 5) >> 2;
        int s_lo = 0, s_hi = 0, s1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // 요소 4k..4k+3 — xq 워드 k, qs 워드 k&3 (k<4 저니블, k>=4 고니블)
            unsigned qs4 = (k & 3) == 0 ? qsw0 : (k & 3) == 1 ? qsw1 : (k & 3) == 2 ? qsw2 : qsw3;
            unsigned nib = (k < 4) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
            unsigned hb = (qh >> (4 * k)) & 0xFu;
            // 요소 i 의 고비트 → 자기 바이트의 bit4 (값 16 — 별도 <<4 불필요)
            unsigned hb_p = (hb & 1u) << 4 | (hb & 2u) << 11 | (hb & 4u) << 18 | (hb & 8u) << 25;
            unsigned y = xq[xw + k];
            s_lo += dot4(nib, y, 0);
            s_hi += dot4(hb_p, y, 0);   // hb_p 바이트 = 16 → 여기서 이미 ×16
            s1 += dot4(ones, y, 0);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        int isum = s_lo + s_hi;
        acc += yd * (d * (float)isum + mn * (float)s1);
    }
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}

// f32 가중치 GEMV/GEMM (MoE 라우터 ffn_gate_inp 등) — 무양자화.
// grid (t, n_out), 64레인, x 행 스트라이드 = x_stride (float).
extern "C" __global__ void q4_gemm_f32(const float* x, const float* w, float* out,
                                       int n_in, int n_out, int x_stride) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    x += (size_t)blockIdx.x * x_stride;
    out += (size_t)blockIdx.x * n_out;
    const float* row = w + (size_t)o * n_in;
    float acc = 0.0f;
    for (int i = l; i < n_in; i += 64) acc += x[i] * row[i];
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}

// QSA 마스크드 밀집 GQA — 값 경로 브리지 (f32 KV, 게이트 인터리브 q).
// q [t][n_head·2·hd] (앞 hd = q, 뒤 hd = gate) · ck/cv [n_past][n_kv·hd] ·
// mask [t][n_past] u32 (0 = 차단) · out [t][n_head·hd].
// 온라인 소프트맥스(청크 128) — CPU 참조(max 후 exp/합)와 수학 동일.
// 차단 위치는 Q4_NEG(=-FLT_MAX) — hipRTC에는 INFINITY 매크로가 없다.
#define Q4_NEG (-3.4028235e38f)
extern "C" __global__ void q4_qsa_attn(const float* q, const float* ck, const float* cv,
                                       const unsigned* mask, float* out, float kq_scale,
                                       int n_past, int n_head, int n_kv, int hd, int t) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;              // dim = hd (256)
    if (tok >= t || h >= n_head || tid >= hd) return;
    int kvh = h / (n_head / n_kv);
    const float* qh = q + ((size_t)tok * n_head + h) * 2 * hd;
    const float* gate = qh + hd;
    const unsigned* mrow = mask + (size_t)tok * n_past;
    float acc = 0.0f;                   // dim tid
    float mrun = Q4_NEG;
    float lrun = 0.0f;
    __shared__ float sc[128];
    __shared__ float wexp[128];
    __shared__ float red[8];
    for (int base = 0; base < n_past; base += 128) {
        int cnt = min(128, n_past - base);
        // 1) 청크 점수 — 스레드 = 위치 (전체 hd 순차 dot)
        if (tid < cnt) {
            int p = base + tid;
            float s = Q4_NEG;
            if (mrow[p] != 0u) {
                const float* k = ck + (size_t)p * n_kv * hd + kvh * hd;
                s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * k[i];
                s *= kq_scale;
            }
            sc[tid] = s;
        }
        __syncthreads();
        // 2) 청크 최대 (블록 환원)
        float cmax = (tid < cnt) ? sc[tid] : Q4_NEG;
        for (int off = 16; off > 0; off >>= 1) cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffffffffffull, cmax, off));
        if ((tid & 31) == 0) red[tid >> 5] = cmax;
        __syncthreads();
        if (tid == 0) {
            float v = red[0];
            for (int w = 1; w < 8; w++) v = fmaxf(v, red[w]);
            red[0] = v;
        }
        __syncthreads();
        if (red[0] <= Q4_NEG) {         // 청크 전체 차단 — 기여 없음
            __syncthreads();
            continue;
        }
        float nrun = fmaxf(mrun, red[0]);
        float corr = (mrun <= Q4_NEG) ? 0.0f : exp_cr(mrun - nrun);
        acc *= corr;
        lrun *= corr;
        mrun = nrun;
        // 3) exp + 청크 합
        float psum = 0.0f;
        if (tid < cnt && sc[tid] > Q4_NEG) {
            float e = exp_cr(sc[tid] - mrun);
            wexp[tid] = e;
            psum = e;
        } else if (tid < cnt) {
            wexp[tid] = 0.0f;
        }
        for (int off = 16; off > 0; off >>= 1) psum += __shfl_xor_sync(0xffffffffffffffffull, psum, off);
        __syncthreads();
        if ((tid & 31) == 0) red[tid >> 5] = psum;
        __syncthreads();
        if (tid == 0) {
            float v = red[0];
            for (int w = 1; w < 8; w++) v += red[w];
            red[0] = v;
        }
        __syncthreads();
        lrun += red[0];
        // 4) V 누적 — 스레드 = dim
        for (int j = 0; j < cnt; j++) {
            float wv = wexp[j];
            if (wv == 0.0f) continue;
            acc += wv * cv[(size_t)(base + j) * n_kv * hd + kvh * hd + tid];
        }
        __syncthreads();
    }
    float o = (lrun > 0.0f) ? acc / lrun : 0.0f;
    float g = 1.0f / (1.0f + exp_cr(-gate[tid]));
    out[((size_t)tok * n_head + h) * hd + tid] = o * g;
}
