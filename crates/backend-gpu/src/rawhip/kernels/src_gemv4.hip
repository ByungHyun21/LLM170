// np 소형 배치(t=2..4)용 4-토큰 GEMV (2026-09-12) — 타일의 128열 패딩 낭비 제거.
// 가중 워드를 1회 읽고 토큰 t개의 y와 dot4 → 토큰별 독립 누산 후 각각 트리 환원.
// 토큰별 산술 순서는 gemm_q5k(t=1)와 동일 — 프리필 타일과 다른 클래스지만
// np 디코드 경로는 spec 계약(비트동일) 대상이 아니다(검증은 트렁크가 수행).
extern "C" __global__ void gemm_q5k4(const unsigned* xq, const unsigned* w,
                                     float* out, int n_in, int n_out, int xq_w, int t) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 176;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int js = sb & 7;
        int it = js >> 1;
        int half = js & 1;
        int wb = row_base + (sb >> 3) * 176;
        int wq = wb >> 2;
        unsigned w0 = w[wq];
        float d = bits_f16(w0 & 0xFFFFu);
        float dm = bits_f16(w0 >> 16);
        unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
        unsigned r = (js & 3) * 8;
        unsigned b_j   = js < 4 ? (sc0 >> r) & 0xFFu : (sc1 >> r) & 0xFFu;
        unsigned b_j4  = js < 4 ? (sc1 >> r) & 0xFFu : (sc2 >> r) & 0xFFu;
        unsigned b_jm4 = (sc0 >> r) & 0xFFu;
        unsigned sc_v, m_v;
        if (js < 4) { sc_v = b_j & 63u; m_v = b_j4 & 63u; }
        else {
            sc_v = (b_j4 & 0xFu) | ((b_jm4 >> 6) << 4);
            m_v  = (b_j4 >> 4) | ((b_j >> 6) << 4);
        }
        int qhw = wq + 4;
        unsigned h0 = w[qhw], h1 = w[qhw+1], h2 = w[qhw+2], h3 = w[qhw+3];
        unsigned h4 = w[qhw+4], h5 = w[qhw+5], h6 = w[qhw+6], h7 = w[qhw+7];
        int qlb = wq + 12 + it * 8;
        unsigned q0 = w[qlb], q1 = w[qlb+1], q2 = w[qlb+2], q3 = w[qlb+3];
        unsigned q4 = w[qlb+4], q5 = w[qlb+5], q6 = w[qlb+6], q7 = w[qlb+7];
        int sh = 2 * it + half;
        int nsh = half << 2;
        unsigned hbit = 1u << sh;
        int qsb = (n_in >> 2) + (n_in >> 5);
        int xw = (sb << 5) >> 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (k >= t) break;
            const unsigned* xk = xq + (size_t)k * xq_w;
            unsigned y0 = xk[xw], y1 = xk[xw+1], y2 = xk[xw+2], y3 = xk[xw+3];
            unsigned y4 = xk[xw+4], y5 = xk[xw+5], y6 = xk[xw+6], y7 = xk[xw+7];
            int isum = 0;
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                unsigned qv = kk < 4 ? (kk==0?q0:kk==1?q1:kk==2?q2:q3) : (kk==4?q4:kk==5?q5:kk==6?q6:q7);
                unsigned hv = kk < 4 ? (kk==0?h0:kk==1?h1:kk==2?h2:h3) : (kk==4?h4:kk==5?h5:kk==6?h6:h7);
                unsigned yv = kk < 4 ? (kk==0?y0:kk==1?y1:kk==2?y2:y3) : (kk==4?y4:kk==5?y5:kk==6?y6:y7);
                unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
                unsigned bitw = ((hv & (hbit * 0x01010101u)) >> sh) << 4;
                isum = dot4(nibw | bitw, yv, isum);
            }
            int qsum = (int)xk[qsb + (sb << 1)] + (int)xk[qsb + (sb << 1) + 1];
            float yd = __uint_as_float(xk[(n_in >> 2) + sb]);
            acc[k] += yd * (d * (float)sc_v) * (float)isum;
            acc[k] -= yd * (dm * (float)m_v) * (float)qsum;
        }
    }
    // 토큰별 트리 환원 (gemm_q5k와 동일 순서)
    __shared__ double sh32[4][32];
    double accd[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) accd[k] = (double)acc[k];
    if (l >= 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) sh32[k][l - 32] = accd[k];
    }
    __syncthreads();
    if (l < 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            accd[k] += sh32[k][l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                accd[k] += __shfl_down_sync(0xffffffffffffffffull, accd[k], off);
        }
        if (l == 0) {
            #pragma unroll
            for (int k = 0; k < 4; k++) out[(size_t)k * n_out + o] = (float)accd[k];
        }
    }
}

// iq4_xs 4-토큰 GEMV — gemm_xs(t=1)와 동일 산술, 가중 1회 독서.
extern "C" __global__ void gemm_xs4(const unsigned* xq, const unsigned* w, const unsigned* ktab2,
                                    float* out, int n_in, int n_out, int xq_w, int t) {
    __shared__ unsigned kt_s[256];
    for (int i = threadIdx.x; i < 256; i += 64) kt_s[i] = ktab2[i];
    __syncthreads();
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 136;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int b = sb >> 3;
        int ib = sb & 7;
        int wb = row_base + b * 136;
        int wq = wb >> 2;
        unsigned w0 = w[wq];
        float d = bits_f16(w0 & 0xFFFFu);
        int qw = (wb + 8 + ib * 16) >> 2;
        int ls = (int)((w[wq + 1] >> (((ib >> 1) * 8 + (ib & 1) * 4))) & 0xFu)
              | (int)((((w0 >> 16) >> (2 * ib)) & 3u) << 4);
        float dl = d * (float)(ls - 32);
        unsigned qv0 = w[qw], qv1 = w[qw+1], qv2 = w[qw+2], qv3 = w[qw+3];
        unsigned lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
        #pragma unroll
        for (int bb = 3; bb >= 0; bb--) {
            unsigned t0 = kt_s[(qv0 >> (8 * bb)) & 0xFFu];
            lo0 = (lo0 << 8) | (t0 & 0xFFu);
            hi0 = (hi0 << 8) | (t0 >> 8);
            unsigned t1 = kt_s[(qv1 >> (8 * bb)) & 0xFFu];
            lo1 = (lo1 << 8) | (t1 & 0xFFu);
            hi1 = (hi1 << 8) | (t1 >> 8);
        }
        unsigned lv0 = 0, hv0 = 0, lv1 = 0, hv1 = 0;
        #pragma unroll
        for (int bb = 3; bb >= 0; bb--) {
            unsigned t2 = kt_s[(qv2 >> (8 * bb)) & 0xFFu];
            lv0 = (lv0 << 8) | (t2 & 0xFFu);
            hv0 = (hv0 << 8) | (t2 >> 8);
            unsigned t3 = kt_s[(qv3 >> (8 * bb)) & 0xFFu];
            lv1 = (lv1 << 8) | (t3 & 0xFFu);
            hv1 = (hv1 << 8) | (t3 >> 8);
        }
        int xw = (sb << 5) >> 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (k >= t) break;
            const unsigned* xk = xq + (size_t)k * xq_w;
            int isum = 0;
            isum = dot4(lo0, xk[xw + 0], isum);
            isum = dot4(hi0, xk[xw + 4], isum);
            isum = dot4(lo1, xk[xw + 1], isum);
            isum = dot4(hi1, xk[xw + 5], isum);
            isum = dot4(lv0, xk[xw + 2], isum);
            isum = dot4(hv0, xk[xw + 6], isum);
            isum = dot4(lv1, xk[xw + 3], isum);
            isum = dot4(hv1, xk[xw + 7], isum);
            float yd = __uint_as_float(xk[(n_in >> 2) + sb]);
            acc[k] += yd * dl * (float)isum;
        }
    }
    __shared__ double sh32[4][32];
    double accd[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) accd[k] = (double)acc[k];
    if (l >= 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) sh32[k][l - 32] = accd[k];
    }
    __syncthreads();
    if (l < 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            accd[k] += sh32[k][l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                accd[k] += __shfl_down_sync(0xffffffffffffffffull, accd[k], off);
        }
        if (l == 0) {
            #pragma unroll
            for (int k = 0; k < 4; k++) out[(size_t)k * n_out + o] = (float)accd[k];
        }
    }
}

// q4_K 4-토큰 GEMV — gemm_q4k(t=1)와 동일 산술·순서, 가중 1회 독서.
extern "C" __global__ void gemm_q4k4(const unsigned* xq, const unsigned* w,
                                     float* out, int n_in, int n_out, int xq_w, int t) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 144;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int js = sb & 7;
        int it = js >> 1;
        int half = js & 1;
        int wb = row_base + (sb >> 3) * 144;
        int wq = wb >> 2;
        unsigned w0 = w[wq];
        float d = bits_f16(w0 & 0xFFFFu);
        float dm = bits_f16(w0 >> 16);
        unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
        unsigned r = (js & 3) * 8;
        unsigned b_j   = js < 4 ? (sc0 >> r) & 0xFFu : (sc1 >> r) & 0xFFu;
        unsigned b_j4  = js < 4 ? (sc1 >> r) & 0xFFu : (sc2 >> r) & 0xFFu;
        unsigned b_jm4 = (sc0 >> r) & 0xFFu;
        unsigned sc_v, m_v;
        if (js < 4) { sc_v = b_j & 63u; m_v = b_j4 & 63u; }
        else {
            sc_v = (b_j4 & 0xFu) | ((b_jm4 >> 6) << 4);
            m_v  = (b_j4 >> 4) | ((b_j >> 6) << 4);
        }
        int qlb = wq + 4 + it * 8;
        unsigned q0 = w[qlb], q1 = w[qlb+1], q2 = w[qlb+2], q3 = w[qlb+3];
        unsigned q4 = w[qlb+4], q5 = w[qlb+5], q6 = w[qlb+6], q7 = w[qlb+7];
        int xw = (sb << 5) >> 2;
        int nsh = half << 2;
        int qsb = (n_in >> 2) + (n_in >> 5);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (k >= t) break;
            const unsigned* xk = xq + (size_t)k * xq_w;
            unsigned y0 = xk[xw], y1 = xk[xw+1], y2 = xk[xw+2], y3 = xk[xw+3];
            unsigned y4 = xk[xw+4], y5 = xk[xw+5], y6 = xk[xw+6], y7 = xk[xw+7];
            int isum = 0;
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                unsigned qv = kk < 4 ? (kk==0?q0:kk==1?q1:kk==2?q2:q3) : (kk==4?q4:kk==5?q5:kk==6?q6:q7);
                unsigned yv = kk < 4 ? (kk==0?y0:kk==1?y1:kk==2?y2:y3) : (kk==4?y4:kk==5?y5:kk==6?y6:y7);
                unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
                isum = dot4(nibw, yv, isum);
            }
            int qsum = (int)xk[qsb + (sb << 1)] + (int)xk[qsb + (sb << 1) + 1];
            float yd = __uint_as_float(xk[(n_in >> 2) + sb]);
            acc[k] += yd * (d * (float)sc_v) * (float)isum;
            acc[k] -= yd * (dm * (float)m_v) * (float)qsum;
        }
    }
    __shared__ double sh32[4][32];
    double accd[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) accd[k] = (double)acc[k];
    if (l >= 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) sh32[k][l - 32] = accd[k];
    }
    __syncthreads();
    if (l < 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            accd[k] += sh32[k][l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                accd[k] += __shfl_down_sync(0xffffffffffffffffull, accd[k], off);
        }
        if (l == 0) {
            #pragma unroll
            for (int k = 0; k < 4; k++) out[(size_t)k * n_out + o] = (float)accd[k];
        }
    }
}

// q6_K 4-토큰 GEMV — gemm_q6k(t=1)와 동일 산술·순서, 가중 1회 독서.
extern "C" __global__ void gemm_q6k4(const unsigned* xq, const unsigned* w,
                                     float* out, int n_in, int n_out, int xq_w, int t) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_g = n_in >> 4;
    int cnt = (n_g + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 210;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < cnt; m++) {
        int g = l + (m << 6);
        int blk = g >> 4;
        int kloc = g - (blk << 4);
        int wb = row_base + blk * 210;
        int h = kloc >> 3;
        int src = (kloc - (h << 3)) >> 1;
        int p = kloc & 1;
        float d = f16w(w, wb + 208);
        int sc = sext8(byte(w, wb + 192 + kloc));
        int ql_rel = h * 64 + p * 16 + ((src & 1) << 5);
        int qh_rel = 128 + h * 32 + p * 16;
        bool al = (wb & 3) == 0;
        int qlw = (wb + ql_rel) >> 2;
        int qhw = (wb + qh_rel) >> 2;
        unsigned qa0, qa1, qa2, qa3, ha0, ha1, ha2, ha3;
        if (al) {
            qa0 = w[qlw]; qa1 = w[qlw+1]; qa2 = w[qlw+2]; qa3 = w[qlw+3];
            ha0 = w[qhw]; ha1 = w[qhw+1]; ha2 = w[qhw+2]; ha3 = w[qhw+3];
        } else {
            qa0 = (w[qlw] >> 16) | (w[qlw+1] << 16);
            qa1 = (w[qlw+1] >> 16) | (w[qlw+2] << 16);
            qa2 = (w[qlw+2] >> 16) | (w[qlw+3] << 16);
            qa3 = (w[qlw+3] >> 16) | (w[qlw+4] << 16);
            ha0 = (w[qhw] >> 16) | (w[qhw+1] << 16);
            ha1 = (w[qhw+1] >> 16) | (w[qhw+2] << 16);
            ha2 = (w[qhw+2] >> 16) | (w[qhw+3] << 16);
            ha3 = (w[qhw+3] >> 16) | (w[qhw+4] << 16);
        }
        int xw = (g << 4) >> 2;
        int nsh = (src == 0 || src == 1) ? 0 : 4;
        int hsh = 2 * src;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (k >= t) break;
            const unsigned* xk = xq + (size_t)k * xq_w;
            unsigned y0 = xk[xw], y1 = xk[xw+1], y2 = xk[xw+2], y3 = xk[xw+3];
            int isum = 0;
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                unsigned qv = kk==0?qa0:kk==1?qa1:kk==2?qa2:qa3;
                unsigned hv = kk==0?ha0:kk==1?ha1:kk==2?ha2:ha3;
                unsigned yv = kk==0?y0:kk==1?y1:kk==2?y2:y3;
                unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
                unsigned hi16w = (hv >> hsh) & 0x03030303u;
                isum = dot4(nibw + (hi16w << 4), yv, isum);
            }
            isum -= 32 * (int)xk[(n_in >> 2) + (n_in >> 5) + g];
            float yd = __uint_as_float(xk[(n_in >> 2) + (g >> 1)]);
            acc[k] += yd * d * (float)sc * (float)isum;
        }
    }
    __shared__ double sh32[4][32];
    double accd[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) accd[k] = (double)acc[k];
    if (l >= 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) sh32[k][l - 32] = accd[k];
    }
    __syncthreads();
    if (l < 32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            accd[k] += sh32[k][l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                accd[k] += __shfl_down_sync(0xffffffffffffffffull, accd[k], off);
        }
        if (l == 0) {
            #pragma unroll
            for (int k = 0; k < 4; k++) out[(size_t)k * n_out + o] = (float)accd[k];
        }
    }
}

// 배치 MTP 입력 결합: out[row*2n + i] = e[row*n + i], out[row*2n + n + i] = h[row*n + i].
extern "C" __global__ void cat2_rows(const float* e, const float* h, float* out, int n, int t) {
    int row = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= t || i >= n) return;
    out[(size_t)row * 2 * n + i] = e[(size_t)row * n + i];
    out[(size_t)row * 2 * n + n + i] = h[(size_t)row * n + i];
}
