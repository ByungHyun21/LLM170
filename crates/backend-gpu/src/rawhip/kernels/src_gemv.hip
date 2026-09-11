// q8_0 듀얼 GEMV — beta/alpha 동시 (출력 2개).
extern "C" __global__ void gemm_q8_0_dual(const unsigned* xq, const unsigned* w1, const unsigned* w2,
                                    float* out1, float* out2, int n_in, int no1, int no2, int xq_w) {
    int o0 = blockIdx.y + blockIdx.z * gridDim.y;
    if (o0 >= no1 + no2) return;
    const unsigned* w;
    float* out;
    int o;
    if (o0 < no1) { w = w1; out = out1; o = o0; }
    else { w = w2; out = out2; o = o0 - no1; }
    int l = threadIdx.x;
    if (l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 34;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 34;
        float d = f16w(w, wb);
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        int wq = wb >> 2;
        bool al = (wb & 3) == 2;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qv;
            if (al) {
                qv = w[wq + 1 + k];
            } else {
                qv = (w[wq + k] >> 16) | (w[wq + k + 1] << 16);
            }
            unsigned yv = k==0?y0:k==1?y1:k==2?y2:k==3?y3:k==4?y4:k==5?y5:k==6?y6:y7;
            isum = dot4(qv, yv, isum);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * d * (float)isum;
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


// ─── W4A8 GEMV t=1 — 스트라이드 레인, f64 부분합 ───
// iq4_xs (ty16)
extern "C" __global__ void gemm_xs(const unsigned* xq, const unsigned* w,
                                   double* part, const unsigned* ktab2, float* out, int n_in, int n_out, int xq_w) {
    __shared__ unsigned kt_s[256];
    for (int i = threadIdx.x; i < 256; i += 64) kt_s[i] = ktab2[i];
    __syncthreads();

    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 136;
    float acc = 0.0f;
    int m = 0;
    // 2-서브블록 ILP — isum 독립 체인, acc 가산은 m 순서 유지 (비트계약)
    for (; m + 1 < cnt; m += 2) {
        int sb0 = l + (m << 6);
        int sb1 = sb0 + 64;
        int b0 = sb0 >> 3, ib0 = sb0 & 7;
        int b1 = sb1 >> 3, ib1 = sb1 & 7;
        int wb0 = row_base + b0 * 136;
        int wb1 = row_base + b1 * 136;
        unsigned w00 = w[wb0 >> 2], w10 = w[wb1 >> 2];
        float d0 = bits_f16(w00 & 0xFFFFu), d1 = bits_f16(w10 & 0xFFFFu);
        int qw0 = (wb0 + 8 + ib0 * 16) >> 2;
        int qw1 = (wb1 + 8 + ib1 * 16) >> 2;
        unsigned ls0 = (int)((w[(wb0 >> 2) + 1] >> (((ib0 >> 1) * 8 + (ib0 & 1) * 4))) & 0xFu)
                     | (int)((((w00 >> 16) >> (2 * ib0)) & 3u) << 4);
        unsigned ls1 = (int)((w[(wb1 >> 2) + 1] >> (((ib1 >> 1) * 8 + (ib1 & 1) * 4))) & 0xFu)
                     | (int)((((w10 >> 16) >> (2 * ib1)) & 3u) << 4);
        float dl0 = d0 * (float)((int)ls0 - 32);
        float dl1 = d1 * (float)((int)ls1 - 32);
        int is0 = 0, is1 = 0;
        int xw0 = (sb0 << 5) >> 2;
        int xw1 = (sb1 << 5) >> 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qv0 = w[qw0 + k];
            unsigned qv1 = w[qw1 + k];
            unsigned lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
            #pragma unroll
            for (int b = 3; b >= 0; b--) {
                unsigned t0 = kt_s[(qv0 >> (8 * b)) & 0xFFu];
                lo0 = (lo0 << 8) | (t0 & 0xFFu);
                hi0 = (hi0 << 8) | (t0 >> 8);
                unsigned t1 = kt_s[(qv1 >> (8 * b)) & 0xFFu];
                lo1 = (lo1 << 8) | (t1 & 0xFFu);
                hi1 = (hi1 << 8) | (t1 >> 8);
            }
            is0 = dot4(lo0, xq[xw0 + k], is0);
            is0 = dot4(hi0, xq[xw0 + 4 + k], is0);
            is1 = dot4(lo1, xq[xw1 + k], is1);
            is1 = dot4(hi1, xq[xw1 + 4 + k], is1);
        }
        float yd0 = __uint_as_float(xq[(n_in >> 2) + sb0]);
        float yd1 = __uint_as_float(xq[(n_in >> 2) + sb1]);
        acc += yd0 * dl0 * (float)is0;
        acc += yd1 * dl1 * (float)is1;
    }
    for (; m < cnt; m++) {
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
        int isum = 0;
        int xw = (sb << 5) >> 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qv = w[qw + k];
            unsigned lo = 0, hi = 0;
            #pragma unroll
            for (int b2 = 3; b2 >= 0; b2--) {
                unsigned t = kt_s[(qv >> (8 * b2)) & 0xFFu];
                lo = (lo << 8) | (t & 0xFFu);
                hi = (hi << 8) | (t >> 8);
            }
            isum = dot4(lo, xq[xw + k], isum);
            isum = dot4(hi, xq[xw + 4 + k], isum);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * dl * (float)isum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
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

// q5_K (ty13) — 분할 형태(곱 체인 2개), qh 비트, 스케일 scale_min_k4
extern "C" __global__ void gemm_q5k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 176;
    float acc = 0.0f;  // f32 레인 누산 (f64 1/16 레이트 RCA)
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
        // dot4 재작성 — 워드 단위 SIMD-in-register (llama.cpp MMVQ 패턴)
        int nsh = half << 2;
        unsigned hbit = 1u << sh;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qv = k < 4 ? (k==0?q0:k==1?q1:k==2?q2:q3) : (k==4?q4:k==5?q5:k==6?q6:q7);
            unsigned hv = k < 4 ? (k==0?h0:k==1?h1:k==2?h2:h3) : (k==4?h4:k==5?h5:k==6?h6:h7);
            unsigned yv = k < 4 ? (k==0?y0:k==1?y1:k==2?y2:y3) : (k==4?y4:k==5?y5:k==6?y6:y7);
            unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
            unsigned bitw = ((hv & (hbit * 0x01010101u)) >> sh) << 4; // 레인 0x00/0x10
            isum = dot4(nibw | bitw, yv, isum);
        }
        int qsb = (n_in >> 2) + (n_in >> 5);
        int qsum = (int)xq[qsb + (sb << 1)] + (int)xq[qsb + (sb << 1) + 1];
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * (d * (float)sc_v) * (float)isum;
        acc -= yd * (dm * (float)m_v) * (float)qsum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;  // f64 트리 — 미러 tree64와 동일 순서
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

// q5_K GEMV v2 (세션5 부록76): llama mmvq vdr=2 구조 이식 — 스레드당
// 연속 2 서브블록 (88B 국소성) + stride-128 그리드-스트라이드 (m-아핀).
// 합산 순서 변경으로 자체 스트림 게이트 대상 (비트계약 아님).
extern "C" __global__ void gemm_q5k_v2(const unsigned* xq, const unsigned* w,
                                    float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 176;
    float acc = 0.0f;
    int qsb = (n_in >> 2) + (n_in >> 5);
    // vdr=2: 페어 인덱스 p = l, sb0 = 2l + 128m (m-아핀)
    for (int m = 0; ; m++) {
        int sb0 = (l << 1) + (m << 7);
        if (sb0 >= n_sub) break;
        #pragma unroll
        for (int half2 = 0; half2 < 2; half2++) {
            int sb = sb0 + half2;
            if (sb >= n_sub) break;
            int js = sb & 7;
            int it = js >> 1;
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
            int qlb = wq + 12 + it * 8;
            unsigned nsh = (js & 1) << 2;
            int sh = 2 * it + (js & 1);
            unsigned hbit = 1u << sh;
            int xw = (sb << 5) >> 2;
            int isum = 0;
            for (int k = 0; k < 8; k++) {
                unsigned qv = w[qlb + k];
                unsigned hv = w[qhw + k];
                unsigned yv = xq[xw + k];
                unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
                unsigned bitw = ((hv & (hbit * 0x01010101u)) >> sh) << 4;
                isum = dot4(nibw | bitw, yv, isum);
            }
            int qsum = (int)xq[qsb + (sb << 1)] + (int)xq[qsb + (sb << 1) + 1];
            float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
            acc += yd * (d * (float)sc_v) * (float)isum;
            acc -= yd * (dm * (float)m_v) * (float)qsum;
        }
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

// 듀얼 텐서 q5_K GEMV (2026-09-05): qkv+gate처럼 같은 xq 입력·같은 포맷의
// 독립 GEMV 2개를 1런치로 — GEMV 사이의 스트리밍 리셋 제거 (KTRACE 계측 근거).
// 행별 산술은 gemm_q5k와 완전 동일 (w/o/out만 행 소속에 따라 선택).
extern "C" __global__ void gemm_q5k2(const unsigned* xq, const unsigned* w1, const unsigned* w2,
                                    float* out1, float* out2, int n_in, int no1, int no2, int xq_w) {
    int o0 = blockIdx.y + blockIdx.z * gridDim.y;
    if (o0 >= no1 + no2) return;
    const unsigned* w;
    float* out;
    int o;
    if (o0 < no1) { w = w1; out = out1; o = o0; }
    else { w = w2; out = out2; o = o0 - no1; }
    int l = threadIdx.x;
    if (l >= 64) return;   // o0 범위는 상단에서 이미 가드
    xq += (int)blockIdx.x * xq_w;
    // t=1 전용 (듀얼): blockIdx.x=0 — out 스트라이드 불필요
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 176;
    float acc = 0.0f;  // f32 레인 누산 (f64 1/16 레이트 RCA)
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
        // dot4 재작성 — 워드 단위 SIMD-in-register (llama.cpp MMVQ 패턴)
        int nsh = half << 2;
        unsigned hbit = 1u << sh;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qv = k < 4 ? (k==0?q0:k==1?q1:k==2?q2:q3) : (k==4?q4:k==5?q5:k==6?q6:q7);
            unsigned hv = k < 4 ? (k==0?h0:k==1?h1:k==2?h2:h3) : (k==4?h4:k==5?h5:k==6?h6:h7);
            unsigned yv = k < 4 ? (k==0?y0:k==1?y1:k==2?y2:y3) : (k==4?y4:k==5?y5:k==6?y6:y7);
            unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
            unsigned bitw = ((hv & (hbit * 0x01010101u)) >> sh) << 4; // 레인 0x00/0x10
            isum = dot4(nibw | bitw, yv, isum);
        }
        int qsb = (n_in >> 2) + (n_in >> 5);
        int qsum = (int)xq[qsb + (sb << 1)] + (int)xq[qsb + (sb << 1) + 1];
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * (d * (float)sc_v) * (float)isum;
        acc -= yd * (dm * (float)m_v) * (float)qsum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;  // f64 트리 — 미러 tree64와 동일 순서
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

// q8_0 (ty8)
extern "C" __global__ void gemm_q8_0(const unsigned* xq, const unsigned* w,
                                     double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 34;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 34;
        float d = f16w(w, wb);
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        // dot4 — 가중 워드 (wb+2, 34B 블록 → 짝수블록 정렬/홀수 +2 슬라이딩)
        int wq = wb >> 2;
        bool al = (wb & 3) == 2; // wb+2가 워드 정렬인 경우
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qv;
            if (al) {
                qv = w[wq + 1 + k];                    // wb+2+4k 워드 정렬
            } else {
                qv = (w[wq + k] >> 16) | (w[wq + k + 1] << 16); // +2 슬라이딩
            }
            unsigned yv = k < 4 ? (k==0?y0:k==1?y1:k==2?y2:y3) : (k==4?y4:k==5?y5:k==6?y6:y7);
            isum = dot4(qv, yv, isum);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * d * (float)isum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
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

// 진단 전용: q6_K 스칼라 기준 GEMV — 명시적 인덱스, 벡터화/트리/게더 없음.
// yf는 양자화 전(또는 복원된) f32 활성. 목적: 엔진 GEMV와 같은 버퍼를
// 서로 다른 코드로 계산해 커널 인덱싱 오류를 분리 판정한다.
extern "C" __global__ void q6k_ref_scalar(const float* yf, const unsigned char* w, float* out,
                                          int n_in, int n_out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    int blocks = n_in >> 8;
    const unsigned char* row = w + (size_t)o * (size_t)blocks * 210u;
    float acc = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const unsigned char* blk = row + b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = (const signed char*)(blk + 192);
        unsigned short dh = *(const unsigned short*)(blk + 208);
        float d = bits_f16((unsigned)dh);
        int base = b * 256;
        #pragma unroll
        for (int n128 = 0; n128 < 2; n128++) {
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;
                int q1 = (int)((ql[l +             n128*64] & 0x0Fu) | (((qh[l + n128*32] >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((ql[l + 32 +        n128*64] & 0x0Fu) | (((qh[l + n128*32] >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((ql[l +             n128*64] >> 4)   | (((qh[l + n128*32] >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((ql[l + 32 +        n128*64] >> 4)   | (((qh[l + n128*32] >> 6) & 3u) << 4)) - 32;
                int p = base + n128*128;
                acc += d * (float)sc[is + 0 + n128*8] * (float)q1 * yf[p + l +  0];
                acc += d * (float)sc[is + 2 + n128*8] * (float)q2 * yf[p + l + 32];
                acc += d * (float)sc[is + 4 + n128*8] * (float)q3 * yf[p + l + 64];
                acc += d * (float)sc[is + 6 + n128*8] * (float)q4 * yf[p + l + 96];
            }
        }
    }
    out[o] = acc;
}

// 프리필 MTP h_shift: 행 시프트 (dst[0]=carry, dst[i]=src[i-1]) — 호스트 왕복 제거.
// src는 본체 step_batch가 남긴 device hidden(xs_t), carry는 이전 청크 마지막 행.
extern "C" __global__ void row_shift_gather(const float* src, const float* carry, float* dst,
                                            int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * t) return;
    int row = i / n;
    int col = i - row * n;
    dst[i] = (row == 0) ? carry[col] : src[(row - 1) * n + col];
}
