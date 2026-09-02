//! 원시 HIP 커널 소스 — core 미러(dot_row_w4a8_*_lane)와 동일 연산열.
//! 스트라이드 레인(레인 l: 서브블록 l, l+64, …), f64 부분합, 레인 순서
//! 합 후 1회 f32 캐스트. 그룹핑 무관 비트 일치 설계 (2026-09-02 k2).

pub const SRC: &str = r#"
#define DEV __device__ __forceinline__

DEV int sext8(unsigned v) { int x = (int)(v & 0xFFu); return x - ((x & 0x80) ? 256 : 0); }
DEV float bits_f16(unsigned h) {
    unsigned sign = (h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 0x1Fu;
    unsigned frac = h & 0x3FFu;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign);
        float v = (float)frac * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    if (exp == 31) return __int_as_float(sign | 0x7F800000u | (frac << 13));
    return __int_as_float(sign | ((exp + 112) << 23) | (frac << 13));
}
DEV float f16w(const unsigned* w, int wb) {
    return bits_f16((w[wb >> 2] >> ((wb & 3) * 8)) & 0xFFFFu);
}
DEV unsigned byte(const unsigned* w, int i) {
    return (w[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
}

// 활성 양자화 — rust quantize_row_q8_ref 미러 (round half away 산술 구현)
// d는 별도 float 저장이 간헐 유실되는 코드젠 결함(2026-09-03 RCA) 회피차
// xq 워드 스트림 뒤 영역(xq[nwords + b])에 u32 비트로 편승 — 저장 경로
// 단일화. GEMV는 xd[sb] = __uint_as_float(xq[nwords + sb])로 읽음.
extern "C" __global__ void quant_q8(const float* x, unsigned* xq, int n) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int nblk = n >> 5;
    if (b >= nblk) return;
    int nwords = n >> 2;
    int base = b << 5;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        amax = fmaxf(amax, fabsf(x[base + i]));
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    #pragma unroll
    for (int wi = 0; wi < 8; wi++) {
        unsigned word = 0u;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float xv = x[base + wi * 4 + k] * id;
            float r = xv >= 0.0f ? (float)(int)(xv + 0.5f)
                                : -((float)(int)(0.5f - xv));
            float c = r > 127.0f ? 127.0f : (r < -127.0f ? -127.0f : r);
            word |= (((unsigned)(int)c) & 0xFFu) << (k * 8);
        }
        xq[b * 8 + wi] = word;
    }
    xq[nwords + b] = __float_as_uint(d); // d 비트 편승 (u32 저장 경로)
}

// reduce: [n_out×64] f64 → [n_out] f32 (레인 순서 합, 1회 캐스트)
extern "C" __global__ void reduce64(const double* part, float* out, int n_out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    double acc = 0.0;
    #pragma unroll
    for (int l = 0; l < 64; l++) acc += part[o * 64 + l];
    out[o] = (float)acc;
}

// ─── W4A8 GEMV t=1 — 스트라이드 레인, f64 부분합 ───
// iq4_xs (ty16)
extern "C" __global__ void gemm_xs(const unsigned* xq, const unsigned* w,
                                   double* part, const unsigned* ktab2, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 136;
    double acc = 0.0;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int b = sb >> 3;
        int ib = sb - (b << 3);
        int wb = row_base + b * 136;
        int wq = wb >> 2;
        unsigned w0 = w[wq];
        unsigned w1 = w[wq + 1];
        float d = bits_f16(w0 & 0xFFFFu);
        unsigned scales_h = w0 >> 16;
        int qw = (wb + 8 + ib * 16) >> 2;
        unsigned q0 = w[qw], q1 = w[qw+1], q2 = w[qw+2], q3 = w[qw+3];
        unsigned nib = (w1 >> (((ib >> 1) * 8 + (ib & 1) * 4))) & 0xFu;
        int ls = (int)nib | (int)(((scales_h >> (2 * ib)) & 3u) << 4);
        float dl = d * (float)(ls - 32);
        int isum = 0;
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned qv, ylv, yhv;
            int wi = j >> 2, sk = (j & 3) * 8;
            switch (wi) {
                case 0: qv = q0; ylv = y0; yhv = y4; break;
                case 1: qv = q1; ylv = y1; yhv = y5; break;
                case 2: qv = q2; ylv = y2; yhv = y6; break;
                default: qv = q3; ylv = y3; yhv = y7; break;
            }
            unsigned t = ktab2[(qv >> sk) & 0xFFu];
            isum += sext8(t & 0xFFu) * sext8((ylv >> sk) & 0xFFu);
            isum += sext8((t >> 8) & 0xFFu) * sext8((yhv >> sk) & 0xFFu);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += (double)(yd * dl * (float)isum);
    }
    part[o * 64 + l] = acc;
}

// q5_K (ty13) — 분할 형태(곱 체인 2개), qh 비트, 스케일 scale_min_k4
extern "C" __global__ void gemm_q5k(const unsigned* xq, const unsigned* w,
                                    double* part, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 176;
    double acc = 0.0;
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
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        int isum = 0, qsum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned qv, hv, yv;
            int wi = j >> 2, k = j & 3;
            switch (wi) {
                case 0: qv = q0; hv = h0; yv = y0; break;
                case 1: qv = q1; hv = h1; yv = y1; break;
                case 2: qv = q2; hv = h2; yv = y2; break;
                case 3: qv = q3; hv = h3; yv = y3; break;
                case 4: qv = q4; hv = h4; yv = y4; break;
                case 5: qv = q5; hv = h5; yv = y5; break;
                case 6: qv = q6; hv = h6; yv = y6; break;
                default: qv = q7; hv = h7; yv = y7; break;
            }
            int sk = k * 8;
            unsigned nib = half == 0 ? (qv >> sk) & 0xFu : ((qv >> sk) >> 4) & 0xFu;
            unsigned t = (hv >> (sk + sh)) & 1u;
            int hi = (int)t * 16;
            int y8 = sext8((yv >> sk) & 0xFFu);
            isum += ((int)nib + hi) * y8;
            qsum += y8;
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += (double)(yd * (d * (float)sc_v) * (float)isum);
        acc -= (double)(yd * (dm * (float)m_v) * (float)qsum);
    }
    part[o * 64 + l] = acc;
}

// q8_0 (ty8)
extern "C" __global__ void gemm_q8_0(const unsigned* xq, const unsigned* w,
                                     double* part, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 34;
    double acc = 0.0;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 34;
        float d = f16w(w, wb);
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            int qv = sext8(byte(w, wb + 2 + j));
            unsigned yv; int wi = j >> 2, sk = (j & 3) * 8;
            switch (wi) {
                case 0: yv = y0; break; case 1: yv = y1; break;
                case 2: yv = y2; break; case 3: yv = y3; break;
                case 4: yv = y4; break; case 5: yv = y5; break;
                case 6: yv = y6; break; default: yv = y7; break;
            }
            isum += qv * sext8((yv >> sk) & 0xFFu);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += (double)(yd * d * (float)isum);
    }
    part[o * 64 + l] = acc;
}

// q4_K (ty12) — 분할 형태, qh 없음 (qs 16..143)
extern "C" __global__ void gemm_q4k(const unsigned* xq, const unsigned* w,
                                    double* part, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 144;
    double acc = 0.0;
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
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        int isum = 0, qsum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned qv, yv;
            int wi = j >> 2, sk = (j & 3) * 8;
            switch (wi) {
                case 0: qv = q0; yv = y0; break; case 1: qv = q1; yv = y1; break;
                case 2: qv = q2; yv = y2; break; case 3: qv = q3; yv = y3; break;
                case 4: qv = q4; yv = y4; break; case 5: qv = q5; yv = y5; break;
                case 6: qv = q6; yv = y6; break; default: qv = q7; yv = y7; break;
            }
            unsigned nib = half == 0 ? (qv >> sk) & 0xFu : ((qv >> sk) >> 4) & 0xFu;
            int y8 = sext8((yv >> sk) & 0xFFu);
            isum += (int)nib * y8;
            qsum += y8;
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += (double)(yd * (d * (float)sc_v) * (float)isum);
        acc -= (double)(yd * (dm * (float)m_v) * (float)qsum);
    }
    part[o * 64 + l] = acc;
}

// q6_K (ty14) — 16원소 그룹, 가상워드 ql/qh (비정렬 5워드 슬라이딩)
extern "C" __global__ void gemm_q6k(const unsigned* xq, const unsigned* w,
                                    double* part, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_g = n_in >> 4;
    int cnt = (n_g + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 210;
    double acc = 0.0;
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
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned qv, hv, yv;
            int wi = j >> 2, sk = (j & 3) * 8;
            switch (wi) {
                case 0: qv = qa0; hv = ha0; yv = y0; break;
                case 1: qv = qa1; hv = ha1; yv = y1; break;
                case 2: qv = qa2; hv = ha2; yv = y2; break;
                default: qv = qa3; hv = ha3; yv = y3; break;
            }
            unsigned nib = (src == 0 || src == 1) ? (qv >> sk) & 0xFu : ((qv >> sk) >> 4) & 0xFu;
            int hi2 = (src == 0) ? (int)((hv >> sk) & 3u)
                    : (src == 1) ? (int)(((hv >> sk) >> 2) & 3u)
                    : (src == 2) ? (int)(((hv >> sk) >> 4) & 3u)
                    : (int)(((hv >> sk) >> 6) & 3u);
            int y8 = sext8((yv >> sk) & 0xFFu);
            isum += (((int)nib | (hi2 << 4)) - 32) * y8;
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + (g >> 1)]);
        acc += (double)(yd * d * (float)sc * (float)isum);
    }
    part[o * 64 + l] = acc;
}

// iq4_nl (ty20) — ktab2 룩업, 블록 32원소
extern "C" __global__ void gemm_nl(const unsigned* xq, const unsigned* w,
                                   double* part, const unsigned* ktab2, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 18;
    double acc = 0.0;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 18;
        float d = f16w(w, wb);
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned qb = byte(w, wb + 2 + j);
            unsigned t = ktab2[qb];
            unsigned ylv, yhv;
            int wi = (j) >> 2, sk = (j & 3) * 8;
            switch (wi) {
                case 0: ylv = y0; yhv = y4; break;
                case 1: ylv = y1; yhv = y5; break;
                case 2: ylv = y2; yhv = y6; break;
                default: ylv = y3; yhv = y7; break;
            }
            isum += sext8(t & 0xFFu) * sext8((ylv >> sk) & 0xFFu);
            isum += sext8((t >> 8) & 0xFFu) * sext8((yhv >> sk) & 0xFFu);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += (double)(yd * d * (float)isum);
    }
    part[o * 64 + l] = acc;
}

// q3_K (ty11) — 16원소 하프블록, ql/hm byte() 로드(비정렬)
extern "C" __global__ void gemm_q3k(const unsigned* xq, const unsigned* w,
                                    double* part, int n_in, int n_out) {
    int o = blockIdx.x + blockIdx.z * gridDim.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_h = n_in >> 4;
    int cnt = (n_h + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 110;
    double acc = 0.0;
    for (int m = 0; m < cnt; m++) {
        int h = l + (m << 6);
        int blk = h >> 4;
        int kloc = h - (blk << 4);
        int hh = kloc >> 3;
        int src = (kloc - (hh << 3)) >> 1;
        int p = kloc & 1;
        int wb = row_base + blk * 110;
        // 스케일 12바이트(wb+96..107) — wb%4 ∈ {0,2} 워드 경계 산술
        int sw = (wb + 96) >> 2;
        int off = (wb + 96) & 3;
        unsigned a0, a1, tmp;
        if (off == 0) { a0 = w[sw]; a1 = w[sw+1]; tmp = w[sw+2]; }
        else { a0 = (w[sw] >> 16) | (w[sw+1] << 16); a1 = (w[sw+1] >> 16) | (w[sw+2] << 16); tmp = (w[sw+2] >> 16) | (w[sw+3] << 16); }
        unsigned k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
        unsigned aux2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
        unsigned aux3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
        unsigned aux0 = (a0 & k2) | ((tmp & k1) << 4);
        unsigned aux1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
        int ai = hh * 8 + src * 2 + p;
        unsigned aux = ai < 4 ? aux0 : (ai < 8 ? aux1 : (ai < 12 ? aux2 : aux3));
        unsigned scb = (aux >> ((ai % 4) * 8)) & 0xFFu;
        float dl = f16w(w, wb + 108) * (float)(sext8(scb) - 32);
        int qlb2 = wb + 32 + hh * 32 + p * 16;
        int hbase2 = wb + p * 16;
        int xw = (h << 4) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned qb = byte(w, qlb2 + j);
            unsigned hb = byte(w, hbase2 + j);
            unsigned yv;
            int wi = j >> 2, sk = (j & 3) * 8;
            switch (wi) {
                case 0: yv = y0; break; case 1: yv = y1; break;
                case 2: yv = y2; break; default: yv = y3; break;
            }
            int qv = (int)((qb >> (src * 2)) & 3u);
            int bit = (int)((hb >> src) & 1u);
            int sub = 4 - bit * 4;
            isum += (qv - sub) * sext8((yv >> sk) & 0xFFu);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + (h >> 1)]);
        acc += (double)(yd * dl * (float)isum);
    }
    part[o * 64 + l] = acc;
}


// ─── ew 계열 (큐브cl ew.rs 산술 이식) ───
extern "C" __global__ void silu_mul(const float* g, const float* u, float* out, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    out[j] = (v / (1.0f + __expf(-v))) * u[j];
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
    int xb = row * n;
    double acc = 0.0;
    for (int i = lo; i < hi; i++) {
        double dv = (double)x[xb + i];
        acc += dv * dv;
    }
    part[row * 32 + u] = acc;
}
extern "C" __global__ void rms_finish(const float* x, const float* w, const double* part,
                                      float* out, float eps, int n, int w_reps) {
    int row = blockIdx.x;
    int u = threadIdx.x;
    if (u >= 32) return;
    double sum = 0.0;
    #pragma unroll
    for (int u2 = 0; u2 < 32; u2++) {
        if (u2 * ((n + 31) >> 5) < n) sum += part[row * 32 + u2];
    }
    float scale32 = (float)sqrt(sum / (double)n + (double)eps);
    float inv = 1.0f / scale32;
    int xb = row * n;
    int wb = (row % w_reps) * n;
    int schunk = (n + 31) >> 5;
    int lo = u * schunk;
    if (lo < n) {
        int hi = min(lo + schunk, n);
        for (int i = lo; i < hi; i++) out[xb + i] = x[xb + i] * inv * w[wb + i];
    }
}
// q/k norm+rope (f64 중간 — FMA 수축 면역, qk_norm_rope 이식)
extern "C" __global__ void qk_norm_rope(float* xq, float* xk, const float* qw, const float* kw,
                                        const float* cs, float eps, float kqs, int pos,
                                        int n_head, int n_kv, int hd, int n_rot) {
    int r = blockIdx.x;
    int u = threadIdx.x;
    int rows = n_head + n_kv;
    if (r >= rows || u != 0) return;
    bool is_q = r < n_head;
    int half = n_rot >> 1;
    int xq_len = n_head * 2 * hd;
    int csbase = pos * half * 2;
    int row_base = is_q ? r * 2 * hd : (r - n_head) * hd;
    // rms 32세그 순차 재현 (ops::sq_sum 순서)
    double parts[32];
    int chunk = (hd + 31) >> 5;
    for (int uu = 0; uu < 32; uu++) {
        int lo = uu * chunk;
        int hi = min(lo + chunk, hd);
        double acc = 0.0;
        for (int i = lo; i < hi; i++) {
            double dv = (double)(is_q ? xq[row_base + i] : xk[row_base + i]);
            acc += dv * dv;
        }
        parts[uu] = acc;
    }
    double sum = 0.0;
    for (int uu = 0; uu < 32; uu++) sum += parts[uu];
    float scale = 1.0f / (float)sqrt(sum / (double)hd + (double)eps);
    if (is_q) {
        for (int i = 0; i < hd; i++)
            xq[row_base + i] = xq[row_base + i] * scale * qw[r * hd + i];
        for (int p = 0; p < half; p++) {
            double c = (double)cs[csbase + p * 2];
            double sf = (double)cs[csbase + p * 2 + 1];
            int a = row_base + p, b = a + half;
            double x0 = (double)xq[a], x1 = (double)xq[b];
            xq[a] = (float)(x0 * c - x1 * sf);
            xq[b] = (float)(x0 * sf + x1 * c);
        }
    } else {
        for (int i = 0; i < hd; i++)
            xk[row_base + i] = xk[row_base + i] * scale * kw[row_base + i] * kqs;
        for (int p = 0; p < half; p++) {
            double c = (double)cs[csbase + p * 2];
            double sf = (double)cs[csbase + p * 2 + 1];
            int a = row_base + p, b = a + half;
            double x0 = (double)xk[a], x1 = (double)xk[b];
            xk[a] = (float)(x0 * c - x1 * sf);
            xk[b] = (float)(x0 * sf + x1 * c);
        }
    }
}


// GDN conv1d + ring shift + silu (t=1)
extern "C" __global__ void gdn_conv(const float* qkv, const float* cw, float* state,
                                    float* out, int ch, int k) {
    int c = blockIdx.x;
    int u = threadIdx.x;
    if (u != 0) return;
    int xb = c; // t=0
    float sum = cw[c * k + (k - 1)] * qkv[xb];
    for (int j = 0; j < k - 1; j++) sum += cw[c * k + j] * state[j * ch + c];
    float oc = sum / (1.0f + __expf(-sum));
    for (int j = 0; j < k - 2; j++) state[j * ch + c] = state[(j + 1) * ch + c];
    state[(k - 2) * ch + c] = qkv[xb];
    out[xb] = oc;
}
// beta / e^g precompute
extern "C" __global__ void gdn_beta_g(const float* b, const float* a, const float* dtb,
                                      const float* sa, float* bg, int n_h) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_h) return;
    float bv = b[h];
    bg[h * 2] = 1.0f / (1.0f + __expf(-bv));
    float x = fminf(a[h] + dtb[h], 80.0f);
    float sp = log1pf(__expf(x));
    bg[h * 2 + 1] = __expf(sp * sa[h]);
}
// norm_gated silu (sequential 32-segment f64 — bit-identical to CPU ops)
extern "C" __global__ void norm_gated_silu(const float* o, const float* z, const float* w,
                                           float* out, float eps, int d, int n_h) {
    int row = blockIdx.x;
    int u = threadIdx.x;
    if (u != 0) return;
    int xb = row * d;
    int wb = (row % n_h) * d;
    double sum = 0.0;
    int chunk = (d + 31) >> 5;
    for (int sg = 0; sg < 32; sg++) {
        int lo = sg * chunk;
        if (lo >= d) break;
        int hi = min(lo + chunk, d);
        double part = 0.0;
        for (int i = lo; i < hi; i++) {
            double dv = (double)o[xb + i];
            part += dv * dv;
        }
        sum += part;
    }
    float scale32 = (float)sqrt(sum / (double)d + (double)eps);
    float inv = 1.0f / scale32;
    for (int i = 0; i < d; i++) {
        float nrm = o[xb + i] * inv * w[wb + i];
        float zz = z[xb + i];
        out[xb + i] = nrm * (zz / (1.0f + __expf(-zz)));
    }
}
// GDN AR state update (cube = (b,h) pair, unit = dv column)
extern "C" __global__ void gdn_ar(float* s, const float* q_scaled, const float* k, const float* v,
                                  const float* beta_ge, float* out, int d, int k_stride,
                                  int v_stride, int h_v, int h_k) {
    int pair = blockIdx.x;
    int u = threadIdx.x;
    if (u >= d) return;
    int b = pair / h_v;
    int h = pair % h_v;
    int kh = h % h_k;
    int base_s = pair * d * d;
    int qk0 = b * k_stride + kh * d;
    int v0 = b * v_stride + h * d;
    float beta = beta_ge[pair * 2];
    float g_exp = beta_ge[pair * 2 + 1];
    float sk = 0.0f;
    for (int kdim = 0; kdim < d; kdim++) {
        float sv = s[base_s + kdim * d + u] * g_exp;
        s[base_s + kdim * d + u] = sv;
        sk += sv * k[qk0 + kdim];
    }
    float delta = (v[v0 + u] - sk) * beta;
    for (int kdim = 0; kdim < d; kdim++) {
        s[base_s + kdim * d + u] += k[qk0 + kdim] * delta;
    }
    float o = 0.0f;
    for (int kdim = 0; kdim < d; kdim++)
        o += s[base_s + kdim * d + u] * q_scaled[qk0 + kdim];
    out[v0 + u] = o;
}
// L2 norm rows (sequential f64 — l2_rows arithmetic) + scale (q only)
extern "C" __global__ void l2_rows2_scale(float* gq, float* gk, float eps, float scale,
                                          int d, int n_group) {
    int row = blockIdx.x;
    int u = threadIdx.x;
    if (u != 0 || row >= 2 * n_group) return;
    bool is_q = row < n_group;
    int r = is_q ? row : row - n_group;
    int xb = r * d;
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        double dv = (double)(is_q ? gq[xb + i] : gk[xb + i]);
        sum += dv * dv;
    }
    float scale32 = (float)sqrt(sum);
    float inv = 1.0f / fmaxf(scale32, eps);
    if (is_q) {
        for (int i = 0; i < d; i++) gq[xb + i] = gq[xb + i] * inv * scale;
    } else {
        for (int i = 0; i < d; i++) gk[xb + i] = gk[xb + i] * inv;
    }
}
// 3-way split (q/k/v)
extern "C" __global__ void split3(const float* src, float* d0, float* d1, float* d2,
                                  int n0, int n1, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n0 + n1 + n2;
    if (i >= total) return;
    if (i < n0) d0[i] = src[i];
    else if (i < n0 + n1) d1[i - n0] = src[i];
    else d2[i - n0 - n1] = src[i];
}

"#;

pub const NAMES: &[&str] = &[
    "quant_q8", "reduce64",
    "gemm_xs", "gemm_q5k", "gemm_q8_0", "gemm_q4k", "gemm_q6k", "gemm_nl", "gemm_q3k",
    "silu_mul", "axpy_scaled", "copy_rows", "rms_part", "rms_finish", "qk_norm_rope",
    "gdn_conv", "gdn_beta_g", "norm_gated_silu", "gdn_ar", "l2_rows2_scale", "split3",
];
// ─── 원시 HIP ew 계열 (큐브cl ew.rs 산술 이식, 다음 검증 대상) ───

