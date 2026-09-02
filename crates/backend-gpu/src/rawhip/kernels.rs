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

// i8×4 레인 정수 dot (v_dot4_i32_i8) — llama.cpp dp4a 상당.
// 비트계약: 정수라 내부 순서 무관, 미러와 동일 isum 보장.
typedef char __attribute__((ext_vector_type(4))) c4v;
DEV int dot4(unsigned a, unsigned b, int c) {
    c4v va = __builtin_bit_cast(c4v, a);
    c4v vb = __builtin_bit_cast(c4v, b);
    return __ockl_sdot4(va, vb, c, false);
}

// 올림-정확 exp(f32→f32) — f64 fma 호너 다항. glibc/Rust expf는 올림-정확
// (ARM routines) → 결과 비트 일치 (경계 2^-28 확률 제외). 디바이스
// expf는 1ulp 빗나감(244/4096 실측, 2026-09-03) — W4A8 비트계약 위반.
DEV float exp_cr(float xf) {
    double x = (double)xf;
    if (x > 88.72) return __int_as_float(0x7f800000u);
    if (x < -103.97) return 0.0f;
    const double LN2_HI = 6.93147180369123816490e-01;
    const double LN2_LO = 1.90821492927058770002e-10;
    const double INV_LN2 = 1.44269504088896338700e+00;
    double kd = rint(x * INV_LN2);
    int k = (int)kd;
    double r = fma(-kd, LN2_HI, x);
    r = fma(-kd, LN2_LO, r);
    double p = 1.0 / 1307674368000.0;
    p = fma(p, r, 1.0 / 479001600.0);
    p = fma(p, r, 1.0 / 39916800.0);
    p = fma(p, r, 1.0 / 3628800.0);
    p = fma(p, r, 1.0 / 362880.0);
    p = fma(p, r, 1.0 / 40320.0);
    p = fma(p, r, 1.0 / 5040.0);
    p = fma(p, r, 1.0 / 720.0);
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    if (k > 127) return __int_as_float(0x7f800000u);
    // 2^k 스케일 — f64 지수 비트 직접 구성 (k ≥ -1022 보장: x ≥ -103.97)
    long long eb = (long long)(k + 1023) << 52;
    double scale = __longlong_as_double(eb);
    return (float)(p * scale);
}

// f64 자연로그 — atanh 급수 fma 호너 (Rust ops::ln_cr과 동일 연산열).
DEV double ln_cr(double v) {
    long long bits = __double_as_longlong(v);
    long long e = (bits >> 52) & 0x7ff;
    double k = (double)(e - 1023);
    double m = __longlong_as_double((bits & ~(0x7ffLL << 52)) | (1023LL << 52));
    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;
    double q = 1.0 / 25.0;
    q = fma(q, t2, 1.0 / 23.0);
    q = fma(q, t2, 1.0 / 21.0);
    q = fma(q, t2, 1.0 / 19.0);
    q = fma(q, t2, 1.0 / 17.0);
    q = fma(q, t2, 1.0 / 15.0);
    q = fma(q, t2, 1.0 / 13.0);
    q = fma(q, t2, 1.0 / 11.0);
    q = fma(q, t2, 1.0 / 9.0);
    q = fma(q, t2, 1.0 / 7.0);
    q = fma(q, t2, 1.0 / 5.0);
    q = fma(q, t2, 1.0 / 3.0);
    q = fma(q, t2, 1.0);
    double lnm = 2.0 * t * q;
    const double LN2_HI = 6.93147180369123816490e-01;
    const double LN2_LO = 1.90821492927058770002e-10;
    double kh = k * LN2_HI;
    double kl = k * LN2_LO;
    double s1 = lnm + kh;
    double s2 = (lnm - s1) + kh;
    return s1 + (s2 + kl);
}

// log1p — ln_cr 급수 경유 (Rust ops::log1p_cr과 동일 연산열).
DEV float log1p_cr(float y) {
    return (float)ln_cr((double)y + 1.0);
}

// 활성 양자화 — rust quantize_row_q8_ref 미러 (round half away 산술 구현)
// d는 별도 float 저장이 간헐 유실되는 코드젠 결함(2026-09-03 RCA) 회피차
// xq 워드 스트림 뒤 영역(xq[nwords + b])에 u32 비트로 편승 — 저장 경로
// 단일화. GEMV는 xd[sb] = __uint_as_float(xq[nwords + sb])로 읽음.
extern "C" __global__ void quant_q8(const float* x, unsigned* xq, int n, int xq_w) {
    int nblk = n >> 5;
    int lb = blockIdx.x * blockDim.x + threadIdx.x;
    if (lb >= nblk) return; // 토큰 내 가드
    x += (size_t)blockIdx.y * n;
    xq += (size_t)blockIdx.y * xq_w;
    int nwords = n >> 2;
    int qs0 = 0, qs1 = 0;
    int base = lb << 5;
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
        xq[lb * 8 + wi] = word;
        if (wi < 4) qs0 = dot4(0x01010101u, word, qs0);
        else qs1 = dot4(0x01010101u, word, qs1);
    }
    xq[nwords + lb] = __float_as_uint(d); // d 비트 편승 (u32 저장 경로)
    xq[nwords + nblk + 2 * lb] = (unsigned)qs0;      // q16 테이블 (하위 16원소합)
    xq[nwords + nblk + 2 * lb + 1] = (unsigned)qs1;
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

// q5_K 타일 배치 (ty13) — 블록=1출력행, TT=16 토큰 타일, 가중 1회 독서.
// 토큰별 isum 레지스터 배열 → 토큰별 트리 환원. 미러와 토큰당 동일열.
#define TT 16
extern "C" __global__ void gemm_q5k_bt(const unsigned* xq, const unsigned* w,
                                       float* out, int n_in, int n_out, int xq_w, int t) {
    int o = blockIdx.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 176;
    float accs[TT];  // f32 레인 누산 — f64 1/16 레이트가 병목 (RCA 2026-09-04)
    #pragma unroll
    for (int q = 0; q < TT; q++) accs[q] = 0.0f;
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
        unsigned nsh = half << 2;
        int sh = 2 * it + half;
        unsigned hbit = 1u << sh;
        unsigned nib0 = (q0 >> nsh) & 0x0F0F0F0Fu, nib1 = (q1 >> nsh) & 0x0F0F0F0Fu;
        unsigned nib2 = (q2 >> nsh) & 0x0F0F0F0Fu, nib3 = (q3 >> nsh) & 0x0F0F0F0Fu;
        unsigned nib4 = (q4 >> nsh) & 0x0F0F0F0Fu, nib5 = (q5 >> nsh) & 0x0F0F0F0Fu;
        unsigned nib6 = (q6 >> nsh) & 0x0F0F0F0Fu, nib7 = (q7 >> nsh) & 0x0F0F0F0Fu;
        unsigned bit0 = ((h0 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit1 = ((h1 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit2 = ((h2 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit3 = ((h3 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit4 = ((h4 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit5 = ((h5 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit6 = ((h6 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned bit7 = ((h7 & (hbit * 0x01010101u)) >> sh) << 4;
        unsigned wv0 = nib0 | bit0, wv1 = nib1 | bit1, wv2 = nib2 | bit2, wv3 = nib3 | bit3;
        unsigned wv4 = nib4 | bit4, wv5 = nib5 | bit5, wv6 = nib6 | bit6, wv7 = nib7 | bit7;
        int xw = (sb << 5) >> 2;
        // 토큰 타일 — isum/토큰, 가중 워드는 루프 외 1회
        for (int ti = 0; ti < t; ti++) {
            const unsigned* xt = xq + ti * xq_w;
            int isum = 0;
            unsigned y0v = xt[xw], y1v = xt[xw+1], y2v = xt[xw+2], y3v = xt[xw+3];
            unsigned y4v = xt[xw+4], y5v = xt[xw+5], y6v = xt[xw+6], y7v = xt[xw+7];
            isum = dot4(wv0, y0v, isum); isum = dot4(wv1, y1v, isum);
            isum = dot4(wv2, y2v, isum); isum = dot4(wv3, y3v, isum);
            isum = dot4(wv4, y4v, isum); isum = dot4(wv5, y5v, isum);
            isum = dot4(wv6, y6v, isum); isum = dot4(wv7, y7v, isum);
            int qsb = (n_in >> 2) + (n_in >> 5);
            int qsum = (int)xt[qsb + (sb << 1)] + (int)xt[qsb + (sb << 1) + 1];
            float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
            int q = ti & (TT - 1);
            accs[q] += yd * (d * (float)sc_v) * (float)isum;
            accs[q] -= yd * (dm * (float)m_v) * (float)qsum;
        }
    }
    // 토큰별 트리 환원 — accs[TT] 중 ti 카운트
    __shared__ double sh32[32];
    for (int ti = 0; ti < t; ti++) {
        double acc = (double)accs[ti & (TT - 1)];
        if (l >= 32) sh32[l - 32] = acc;
        __syncthreads();
        if (l < 32) {
            acc += sh32[l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
            if (l == 0) out[(size_t)ti * n_out + o] = (float)acc;
        }
        __syncthreads();
    }
}

// ─── 타일 배치 GEMM (블록=1행, TT 토큰 레지스터, 가중 1회 독서) ───
// gemm_q5k_bt 참조. 미러와 토큰당 동일 연산열.

// q4_K 타일 (ty12)
extern "C" __global__ void gemm_q4k_bt(const unsigned* xq, const unsigned* w,
                                       float* out, int n_in, int n_out, int xq_w, int t) {
    int o = blockIdx.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 144;
    float accs[TT];
    #pragma unroll
    for (int q = 0; q < TT; q++) accs[q] = 0.0f;
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
        unsigned nsh = half << 2;
        unsigned wv0 = (w[qlb]   >> nsh) & 0x0F0F0F0Fu;
        unsigned wv1 = (w[qlb+1] >> nsh) & 0x0F0F0F0Fu;
        unsigned wv2 = (w[qlb+2] >> nsh) & 0x0F0F0F0Fu;
        unsigned wv3 = (w[qlb+3] >> nsh) & 0x0F0F0F0Fu;
        unsigned wv4 = (w[qlb+4] >> nsh) & 0x0F0F0F0Fu;
        unsigned wv5 = (w[qlb+5] >> nsh) & 0x0F0F0F0Fu;
        unsigned wv6 = (w[qlb+6] >> nsh) & 0x0F0F0F0Fu;
        unsigned wv7 = (w[qlb+7] >> nsh) & 0x0F0F0F0Fu;
        int xw = (sb << 5) >> 2;
        for (int ti = 0; ti < t; ti++) {
            const unsigned* xt = xq + ti * xq_w;
            int isum = 0;
            unsigned y0v = xt[xw], y1v = xt[xw+1], y2v = xt[xw+2], y3v = xt[xw+3];
            unsigned y4v = xt[xw+4], y5v = xt[xw+5], y6v = xt[xw+6], y7v = xt[xw+7];
            isum = dot4(wv0, y0v, isum); isum = dot4(wv1, y1v, isum);
            isum = dot4(wv2, y2v, isum); isum = dot4(wv3, y3v, isum);
            isum = dot4(wv4, y4v, isum); isum = dot4(wv5, y5v, isum);
            isum = dot4(wv6, y6v, isum); isum = dot4(wv7, y7v, isum);
            int qsb = (n_in >> 2) + (n_in >> 5);
            int qsum = (int)xt[qsb + (sb << 1)] + (int)xt[qsb + (sb << 1) + 1];
            float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
            int q = ti & (TT - 1);
            accs[q] += yd * (d * (float)sc_v) * (float)isum;
            accs[q] -= yd * (dm * (float)m_v) * (float)qsum;
        }
    }
    __shared__ double sh32[32];
    for (int ti = 0; ti < t; ti++) {
        double acc = (double)accs[ti & (TT - 1)];
        if (l >= 32) sh32[l - 32] = acc;
        __syncthreads();
        if (l < 32) {
            acc += sh32[l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
            if (l == 0) out[(size_t)ti * n_out + o] = (float)acc;
        }
        __syncthreads();
    }
}

// q6_K 타일 (ty14) — 16원소 그룹, w = nib + 16·hi2 − 32 분해
extern "C" __global__ void gemm_q6k_bt(const unsigned* xq, const unsigned* w,
                                       float* out, int n_in, int n_out, int xq_w, int t) {
    int o = blockIdx.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_g = n_in >> 4;
    int cnt = (n_g + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 210;
    float accs[TT];
    #pragma unroll
    for (int q = 0; q < TT; q++) accs[q] = 0.0f;
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
        int nsh = (src == 0 || src == 1) ? 0 : 4;
        int hsh = 2 * src;
        unsigned wv0 = ((qa0 >> nsh) & 0x0F0F0F0Fu) + ((((ha0 >> hsh) & 0x03030303u)) << 4);
        unsigned wv1 = ((qa1 >> nsh) & 0x0F0F0F0Fu) + ((((ha1 >> hsh) & 0x03030303u)) << 4);
        unsigned wv2 = ((qa2 >> nsh) & 0x0F0F0F0Fu) + ((((ha2 >> hsh) & 0x03030303u)) << 4);
        unsigned wv3 = ((qa3 >> nsh) & 0x0F0F0F0Fu) + ((((ha3 >> hsh) & 0x03030303u)) << 4);
        int xw = (g << 4) >> 2;
        for (int ti = 0; ti < t; ti++) {
            const unsigned* xt = xq + ti * xq_w;
            int isum = 0;
            unsigned y0v = xt[xw], y1v = xt[xw+1], y2v = xt[xw+2], y3v = xt[xw+3];
            isum = dot4(wv0, y0v, isum); isum = dot4(wv1, y1v, isum);
            isum = dot4(wv2, y2v, isum); isum = dot4(wv3, y3v, isum);
            isum -= 32 * (int)xt[(n_in >> 2) + (n_in >> 5) + g];
            float yd = __uint_as_float(xt[(n_in >> 2) + (g >> 1)]);
            int q = ti & (TT - 1);
            accs[q] += yd * d * (float)sc * (float)isum;
        }
    }
    __shared__ double sh32[32];
    for (int ti = 0; ti < t; ti++) {
        double acc = (double)accs[ti & (TT - 1)];
        if (l >= 32) sh32[l - 32] = acc;
        __syncthreads();
        if (l < 32) {
            acc += sh32[l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
            if (l == 0) out[(size_t)ti * n_out + o] = (float)acc;
        }
        __syncthreads();
    }
}

// iq4_xs 타일 (ty23) — kvalues 룩업을 m 루프로 호이스트(토큰 불변)
extern "C" __global__ void gemm_xs_bt(const unsigned* xq, const unsigned* w,
                                      float* out, const unsigned* ktab2, int n_in, int n_out,
                                      int xq_w, int t) {
    __shared__ unsigned kt_s[256];
    for (int i = threadIdx.x; i < 256; i += 64) kt_s[i] = ktab2[i];
    __syncthreads();

    int o = blockIdx.x;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 136;
    float accs[TT];
    #pragma unroll
    for (int q = 0; q < TT; q++) accs[q] = 0.0f;
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
        // 룩업+패킹 — 토큰 불변, 1회
        unsigned lo[4], hi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qv = w[qw + k];
            unsigned lov = 0, hiv = 0;
            #pragma unroll
            for (int b2 = 3; b2 >= 0; b2--) {
                unsigned tt2 = kt_s[(qv >> (8 * b2)) & 0xFFu];
                lov = (lov << 8) | (tt2 & 0xFFu);
                hiv = (hiv << 8) | (tt2 >> 8);
            }
            lo[k] = lov; hi[k] = hiv;
        }
        int xw = (sb << 5) >> 2;
        for (int ti = 0; ti < t; ti++) {
            const unsigned* xt = xq + ti * xq_w;
            int isum = 0;
            isum = dot4(lo[0], xt[xw], isum); isum = dot4(lo[1], xt[xw+1], isum);
            isum = dot4(lo[2], xt[xw+2], isum); isum = dot4(lo[3], xt[xw+3], isum);
            isum = dot4(hi[0], xt[xw+4], isum); isum = dot4(hi[1], xt[xw+5], isum);
            isum = dot4(hi[2], xt[xw+6], isum); isum = dot4(hi[3], xt[xw+7], isum);
            float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
            int q = ti & (TT - 1);
            accs[q] += yd * dl * (float)isum;
        }
    }
    __shared__ double sh32[32];
    for (int ti = 0; ti < t; ti++) {
        double acc = (double)accs[ti & (TT - 1)];
        if (l >= 32) sh32[l - 32] = acc;
        __syncthreads();
        if (l < 32) {
            acc += sh32[l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffffffffffull, acc, off);
            if (l == 0) out[(size_t)ti * n_out + o] = (float)acc;
        }
        __syncthreads();
    }
}

// q4_K (ty12) — 분할 형태, qh 없음 (qs 16..143)
extern "C" __global__ void gemm_q4k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 144;
    float acc = 0.0f;
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
        int nsh = half << 2;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qv = k < 4 ? (k==0?q0:k==1?q1:k==2?q2:q3) : (k==4?q4:k==5?q5:k==6?q6:q7);
            unsigned yv = k < 4 ? (k==0?y0:k==1?y1:k==2?y2:y3) : (k==4?y4:k==5?y5:k==6?y6:y7);
            unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
            isum = dot4(nibw, yv, isum);
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

// q6_K (ty14) — 16원소 그룹, 가상워드 ql/qh (비정렬 5워드 슬라이딩)
extern "C" __global__ void gemm_q6k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_g = n_in >> 4;
    int cnt = (n_g + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 210;
    float acc = 0.0f;
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
        // dot4 — 레인 간 빌림 방지 분해: w = nib + 16·hi2 − 32
        //   Σw·y = Σ(nib + 16·hi2)·y − 32·Σy  (레인 내 덧셈 ≤ 63 — 캐리 없음)
        int nsh = (src == 0 || src == 1) ? 0 : 4;
        int hsh = 2 * src;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qv = k==0?qa0:k==1?qa1:k==2?qa2:qa3;
            unsigned hv = k==0?ha0:k==1?ha1:k==2?ha2:ha3;
            unsigned yv = k==0?y0:k==1?y1:k==2?y2:y3;
            unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
            unsigned hi16w = (hv >> hsh) & 0x03030303u;
            isum = dot4(nibw + (hi16w << 4), yv, isum);
        }
        isum -= 32 * (int)xq[(n_in >> 2) + (n_in >> 5) + g]; // q16[g]=Σy, 체인은 32×Σy
        float yd = __uint_as_float(xq[(n_in >> 2) + (g >> 1)]);
        acc += yd * d * (float)sc * (float)isum;
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

// iq4_nl (ty20) — ktab2 룩업, 블록 32원소
extern "C" __global__ void gemm_nl(const unsigned* xq, const unsigned* w,
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
    int row_base = o * n_sub * 18;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 18;
        float d = f16w(w, wb);
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        // dot4 — kvalues 룩업 후 i8×4 패킹 (하프 워드 lo=elements 0..15, hi=16..31)
        bool al = (wb & 3) == 2;
        int wq2 = wb >> 2;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qvw; // 4 weight byte
            if (al) qvw = w[wq2 + 1 + k];
            else qvw = (w[wq2 + k] >> 16) | (w[wq2 + k + 1] << 16);
            unsigned lo = 0, hi = 0;
            #pragma unroll
            for (int b = 3; b >= 0; b--) {
                unsigned t = kt_s[(qvw >> (8 * b)) & 0xFFu];
                lo = (lo << 8) | (t & 0xFFu);
                hi = (hi << 8) | (t >> 8);
            }
            unsigned ylv = k==0?y0:k==1?y1:k==2?y2:y3;
            unsigned yhv = k==0?y4:k==1?y5:k==2?y6:y7;
            isum = dot4(lo, ylv, isum);
            isum = dot4(hi, yhv, isum);
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

// q3_K (ty11) — 16원소 하프블록, ql/hm byte() 로드(비정렬)
extern "C" __global__ void gemm_q3k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_h = n_in >> 4;
    int cnt = (n_h + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 110;
    float acc = 0.0f;
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
        // dot4 — Σ(qv − 4 + 4bit)·y = Σ qv·y − 4Σy + 4Σ bit·y
        int qsh = src * 2;
        bool alq = (qlb2 & 3) == 0;
        bool alh = (hbase2 & 3) == 0;
        int qlw2 = qlb2 >> 2;
        int qhw2 = hbase2 >> 2;
        int isum = 0, qsum = 0, bsum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qvw, hvw;
            if (alq) qvw = w[qlw2 + k]; else qvw = (w[qlw2 + k] >> 16) | (w[qlw2 + k + 1] << 16);
            if (alh) hvw = w[qhw2 + k]; else hvw = (w[qhw2 + k] >> 16) | (w[qhw2 + k + 1] << 16);
            unsigned yv = k==0?y0:k==1?y1:k==2?y2:y3;
            isum = dot4((qvw >> qsh) & 0x03030303u, yv, isum);
            qsum = dot4(0x04040404u, yv, qsum);
            unsigned bitw = ((hvw >> src) & 0x01010101u) << 2;
            bsum = dot4(bitw, yv, bsum);
        }
        isum += bsum - qsum;
        float yd = __uint_as_float(xq[(n_in >> 2) + (h >> 1)]);
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


// iq3_s (ty21) — 32원소 서브블록, grid4×부호 정수 가중, f64 레인 누산
__device__ const unsigned IQ3S_GRID[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

extern "C" __global__ void gemm_iq3s(const unsigned* xq, const unsigned* w,
                                     double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 110;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sub = l + (m << 6);
        int blk = sub >> 3;
        int h = sub & 7;
        int wb = row_base + blk * 110;
        float d = f16w(w, wb);
        unsigned scb = byte(w, wb + 106 + (h >> 1));
        int nib = (h & 1) ? (int)(scb >> 4) : (int)(scb & 0xFu);
        float db = d * (float)(1 + 2 * nib);
        unsigned qhb = byte(w, wb + 66 + h);
        int qs_base = wb + 2 + h * 8;
        int sg_base = wb + 74 + h * 4;
        long isum = 0;
        #pragma unroll
        for (int ll = 0; ll < 4; ll++) {
            unsigned idx1 = byte(w, qs_base + 2*ll) | ((qhb << (8 - 2*ll)) & 256u);
            unsigned idx2 = byte(w, qs_base + 2*ll + 1) | ((qhb << (7 - 2*ll)) & 256u);
            unsigned g1 = IQ3S_GRID[idx1];
            unsigned g2 = IQ3S_GRID[idx2];
            unsigned sgb = byte(w, sg_base + ll);
            int e0 = 8 * ll;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int w1 = sext8((g1 >> (8*j)) & 0xFFu) * ((sgb & (1u << j)) ? -1 : 1);
                int w2 = sext8((g2 >> (8*j)) & 0xFFu) * ((sgb & (1u << (4+j))) ? -1 : 1);
                int e1 = e0 + j, e2 = e0 + 4 + j;
                int y1 = sext8(byte(xq, ((sub << 5) + e1)));
                int y2 = sext8(byte(xq, ((sub << 5) + e2)));
                isum += (long)w1 * y1 + (long)w2 * y2;
            }
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sub]);
        acc += yd * db * (float)isum;
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

// 디버그: row0의 특정 sub 하나만 계산해 part[0]에 기록.
extern "C" __global__ void gemm_iq3s_sub(const unsigned* xq, const unsigned* w,
                                         double* part, int n_in, long sub) {
    int blk = (int)(sub >> 3);
    int h = (int)(sub & 7);
    int wb = blk * 110;
    float d = f16w(w, wb);
    unsigned scb = byte(w, wb + 106 + (h >> 1));
    int nib = (h & 1) ? (int)(scb >> 4) : (int)(scb & 0xFu);
    float db = d * (float)(1 + 2 * nib);
    unsigned qhb = byte(w, wb + 66 + h);
    int qs_base = wb + 2 + h * 8;
    int sg_base = wb + 74 + h * 4;
    long isum = 0;
    #pragma unroll
    for (int ll = 0; ll < 4; ll++) {
        unsigned idx1 = byte(w, qs_base + 2*ll) | ((qhb << (8 - 2*ll)) & 256u);
        unsigned idx2 = byte(w, qs_base + 2*ll + 1) | ((qhb << (7 - 2*ll)) & 256u);
        unsigned g1 = IQ3S_GRID[idx1];
        unsigned g2 = IQ3S_GRID[idx2];
        unsigned sgb = byte(w, sg_base + ll);
        int e0 = 8 * ll;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int w1 = sext8((g1 >> (8*j)) & 0xFFu) * ((sgb & (1u << j)) ? -1 : 1);
            int w2 = sext8((g2 >> (8*j)) & 0xFFu) * ((sgb & (1u << (4+j))) ? -1 : 1);
            int e1 = e0 + j, e2 = e0 + 4 + j;
            int y1 = sext8(byte(xq, ((sub << 5) + e1)));
            int y2 = sext8(byte(xq, ((sub << 5) + e2)));
            isum += (long)w1 * y1 + (long)w2 * y2;
        }
    }
    float yd = __uint_as_float(xq[(n_in >> 2) + sub]);
    part[0] = (double)(yd * db * (float)isum);
    part[1] = (double)isum;
    part[2] = (double)__uint_as_float(__float_as_uint(db));
    part[3] = (double)d;
    part[4] = (double)yd;
    part[5] = (double)(yd * db);
    part[6] = (double)((yd * db) * (float)isum);
}


// 디버그: expf 비트 동일성 프로브 — host Rust exp와 비교.
extern "C" __global__ void exp_probe(const float* x, unsigned* bits, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bits[i] = __float_as_uint((float)exp((double)x[i]));
}

// dot intrinsic 프로브 — sdot8(i4x8) 확정 + ext_vector_type sdot2/4 시도
typedef short __attribute__((ext_vector_type(2))) s2v;
extern "C" __global__ void dp4a_probe(const unsigned* x, int* out) {
    out[0] = __ockl_sdot8((int)x[0], (int)x[1], 0, false);   // i4x8: 3
    out[1] = (int)__ockl_udot8(x[2], x[3], 0u, false);       // 5
    s2v a2 = {12, 0}, b2 = {17, 0};
    out[2] = __ockl_sdot2(a2, b2, 0, false);                 // 204
    c4v a4 = {12, 0, 0, 0}, b4 = {17, 0, 0, 0};
    out[3] = __ockl_sdot4(a4, b4, 0, false);                 // 204
    // 음수 레인: {1,1,1,1}·{-1,-1,-1,-1} = -4
    c4v o4 = {1, 1, 1, 1}, m4 = {-1, -1, -1, -1};
    out[4] = __ockl_sdot4(o4, m4, 0, false);
    // {-32,2,0,0}·{17,3,5,7} = -544+6 = -538
    c4v n4 = {-32, 2, 0, 0}, q4 = {17, 3, 5, 7};
    out[5] = __ockl_sdot4(n4, q4, 0, false);
    // bitcast 경로: {0,-32,0,2}·{7,17,5,3} = 0 - 544 + 0 + 6 = -538
    out[6] = dot4(0x02E00000u, 0x03110705u, 0);
    out[7] = dot4(0x02E00000u, 0x03110705u, 1000);           // 누적 확인 462
}

// 순수 대역폭 프로브 — gemm과 동일 그리드/레인/서브블록 접근, 읽기+XOR만.
extern "C" __global__ void bw_probe(const unsigned* w, double* part, int n_in, int n_out, int bsize) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * bsize;
    unsigned acc = 0;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + (sb >> 3) * bsize;
        int qw = wb >> 2;
        acc ^= w[qw] ^ w[qw+1] ^ w[qw+2] ^ w[qw+3];
    }
    part[o * 64 + l] = (double)acc + (double)l + (double)o;
}

// 디버그: q6_K 단일 그룹 old/new isum 동시 계산
extern "C" __global__ void q6k_ab(const unsigned* w, const unsigned* xq, double* part,
                                  int row, int g) {
    int blocks = 0; // 계산 재사용
    (void)blocks;
    int wb = row * (16) * 210 + (g >> 4) * 210; // n_in=256 가정 (호출부가 조정)
    int klocal = g & 15;
    int h = klocal >> 3;
    int src = (klocal & 7) >> 1;
    int p = klocal & 1;
    // 워드 로드 (al)
    int ql_rel = h * 64 + p * 16 + ((src & 1) << 5);
    int qh_rel = 128 + h * 32 + p * 16;
    bool al = (wb & 3) == 0;
    int qlw = (wb + ql_rel) >> 2;
    int qhw = (wb + qh_rel) >> 2;
    unsigned qa[4], ha[4];
    if (al) {
        for (int k = 0; k < 4; k++) { qa[k] = w[qlw+k]; ha[k] = w[qhw+k]; }
    } else {
        for (int k = 0; k < 4; k++) {
            qa[k] = (w[qlw+k] >> 16) | (w[qlw+k+1] << 16);
            ha[k] = (w[qhw+k] >> 16) | (w[qhw+k+1] << 16);
        }
    }
    int xw = (g << 4) >> 2;
    unsigned yv[4];
    for (int k = 0; k < 4; k++) yv[k] = xq[xw + k];
    // old 스칼라
    int isum_old = 0;
    for (int j = 0; j < 16; j++) {
        unsigned qv = qa[j >> 2], hv = ha[j >> 2];
        int sk = (j & 3) * 8;
        unsigned nib = (src == 0 || src == 1) ? (qv >> sk) & 0xFu : ((qv >> sk) >> 4) & 0xFu;
        int hi2 = (src == 0) ? (int)((hv >> sk) & 3u)
                : (src == 1) ? (int)(((hv >> sk) >> 2) & 3u)
                : (src == 2) ? (int)(((hv >> sk) >> 4) & 3u)
                : (int)(((hv >> sk) >> 6) & 3u);
        int y8 = sext8((yv[j >> 2] >> sk) & 0xFFu);
        isum_old += (((int)nib | (hi2 << 4)) - 32) * y8;
    }
    // new dot4
    int nsh = (src == 0 || src == 1) ? 0 : 4;
    int hsh = 2 * src;
    int isum_new = 0;
    for (int k = 0; k < 4; k++) {
        unsigned nibw = (qa[k] >> nsh) & 0x0F0F0F0Fu;
        unsigned hi2w = (ha[k] >> hsh) & 0x03030303u;
        unsigned wv = (nibw | (hi2w << 4)) - 0x20202020u;
        isum_new = dot4(wv, yv[k], isum_new);
    }
    // v3: 패킹 후 스칼라 (dot4 없이)
    int isum_v3 = 0;
    for (int k = 0; k < 4; k++) {
        for (int b = 0; b < 4; b++) {
            unsigned nib = (qa[k] >> (8 * b + nsh)) & 0xFu;
            unsigned hi2 = (ha[k] >> (8 * b + hsh)) & 3u;
            int w = ((int)nib | ((int)hi2 << 4)) - 32;
            int y8 = sext8((yv[k] >> (8 * b)) & 0xFFu);
            isum_v3 += w * y8;
        }
    }
    part[0] = (double)isum_old;
    part[1] = (double)isum_new;
    part[2] = (double)isum_v3;
    part[3] = (double)(wb & 3);
}

// 트리 환원 순서 프로브 — 변형별 검증
extern "C" __global__ void tree_probe(double* out) {
    int l = threadIdx.x;
    double v = (double)l; // lane 인덱스 — 정수값
    // a) 1스텝 off=32: lane0 = 0+32
    double a = v + __shfl_down_sync(0xffffffffffffffffull, v, 32);
    // b) 1스텝 off=1: lane0 = 0+1
    double b = v + __shfl_down_sync(0xffffffffffffffffull, v, 1);
    // c) 전체 트리: 0+1+...+63 = 2016
    double c = v;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        c += __shfl_down_sync(0xffffffffffffffffull, c, off);
    // d) width=64 명시
    double d = v + __shfl_down_sync(0xffffffffffffffffull, v, 32, 64);
    // e) width 명시 전체 트리
    double e = v;
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        e += __shfl_down_sync(0xffffffffffffffffull, e, off, 64);
    if (l == 0) { out[0] = a; out[1] = b; out[2] = c; out[3] = d; out[4] = e; }
}

// dot4 루프-오버헤드 프로브 — 순수 레지스터 체인 vs L1/L2 로드 혼합.
extern "C" __global__ void dot_roof(const unsigned* xq, const unsigned* w,
                                    double* out, int mode, int iters, int n_in) {
    int l = threadIdx.x;
    unsigned a = xq[l], b = w[l];
    int acc = 0;
    if (mode == 0) {
        // 순수 레지스터 dot 체인 (루프 카운트만)
        for (int i = 0; i < iters; i++) acc = dot4(a, b, acc);
    } else if (mode == 1) {
        // L1/글로벌 로드 동반 (같은 주소 — 캐시 히트)
        for (int i = 0; i < iters; i++) {
            unsigned yv = xq[l];
            unsigned wv = w[l];
            acc = dot4(wv, yv, acc);
        }
    } else {
        // 스트라이드 로드 (실제 커널 유사 — 캐시라인 분산)
        for (int i = 0; i < iters; i++) {
            unsigned yv = xq[(l + i * 64) % (n_in >> 2)];
            unsigned wv = w[(l + i * 64) % (n_in >> 2)];
            acc = dot4(wv, yv, acc);
        }
    }
    if (acc == 0x7fffffff) out[0] = 1.0;
    out[1] = (double)acc;
}

// ─── ew 계열 (큐브cl ew.rs 산술 이식) ───
extern "C" __global__ void silu_mul(const float* g, const float* u, float* out, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    out[j] = (v / (1.0f + exp_cr(-v))) * u[j];
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
    float acc = 0.0f;  // f32 세그먼트 (f64 1/16 레이트) — 미러 sq_sum과 쌍
    for (int i = lo; i < hi; i++) {
        float dv = x[xb + i];
        acc += dv * dv;
    }
    part[row * 32 + u] = (double)acc;
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
    // gy=t 배치 — aq [t][n_head·2hd] (q‖gate 인터리브), ak [t][n_kv·hd].
    int r0 = blockIdx.x;
    int u = threadIdx.x;
    int y = blockIdx.y;
    if (r0 >= n_head + n_kv || u != 0) return;
    bool is_q = r0 < n_head;
    int half = n_rot >> 1;
    int csbase = (pos + y) * half * 2;
    int row_base = is_q ? y * (n_head * 2 * hd) + r0 * 2 * hd
                        : y * (n_kv * hd) + (r0 - n_head) * hd;
    float* xv = is_q ? xq : xk;
    const float* wv = is_q ? qw + r0 * hd : kw + (r0 - n_head) * hd;
    float parts[32];  // f32 세그먼트 — 미러 sq_sum과 쌍
    int chunk = (hd + 31) >> 5;
    for (int uu = 0; uu < 32; uu++) {
        int lo = uu * chunk;
        int hi = min(lo + chunk, hd);
        float acc = 0.0f;
        for (int i = lo; i < hi; i++) {
            float dv = xv[row_base + i];
            acc += dv * dv;
        }
        parts[uu] = acc;
    }
    double sum = 0.0;
    for (int uu = 0; uu < 32; uu++) sum += parts[uu];
    float scale = 1.0f / (float)sqrt(sum / (double)hd + (double)eps);
    for (int i = 0; i < hd; i++)
        xv[row_base + i] = xv[row_base + i] * scale * wv[i] * (is_q ? 1.0f : kqs);
    for (int p = 0; p < half; p++) {
        double c = (double)cs[csbase + p * 2];
        double sf = (double)cs[csbase + p * 2 + 1];
        int a = row_base + p, b = a + half;
        double x0 = (double)xv[a], x1 = (double)xv[b];
        xv[a] = (float)(x0 * c - x1 * sf);
        xv[b] = (float)(x0 * sf + x1 * c);
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
    float oc = sum / (1.0f + exp_cr(-sum));
    for (int j = 0; j < k - 2; j++) state[j * ch + c] = state[(j + 1) * ch + c];
    state[(k - 2) * ch + c] = qkv[xb];
    out[xb] = oc;
}
// GDN conv1d + ring — t토큰 순차 (thread0/채널, blockIdx.y=t)
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
// norm_gated silu (sequential 32-segment f64 — bit-identical to CPU ops)
extern "C" __global__ void norm_gated_silu(const float* o, const float* z, const float* w,
                                           float* out, float eps, int d, int n_h) {
    int row = blockIdx.y * n_h + blockIdx.x;
    int u = threadIdx.x;
    if (u != 0 || blockIdx.x >= n_h) return;
    int xb = row * d;
    int wb = blockIdx.x * d;
    double sum = 0.0;
    int chunk = (d + 31) >> 5;
    for (int sg = 0; sg < 32; sg++) {
        int lo = sg * chunk;
        if (lo >= d) break;
        int hi = min(lo + chunk, d);
        float part = 0.0f;  // f32 세그먼트 — 미러와 쌍
        for (int i = lo; i < hi; i++) {
            float dv = o[xb + i];
            part += dv * dv;
        }
        sum += (double)part;
    }
    float scale32 = (float)sqrt(sum / (double)d + (double)eps);
    float inv = 1.0f / scale32;
    for (int i = 0; i < d; i++) {
        float nrm = o[xb + i] * inv * w[wb + i];
        float zz = z[xb + i];
        out[xb + i] = nrm * (zz / (1.0f + exp_cr(-zz)));
    }
}
// GDN AR state update (cube = (b,h) pair, unit = dv column)
extern "C" __global__ void gdn_ar(float* s, const float* q, const float* k, const float* v,
                                  const float* beta_ge, float* out, int d, int k_stride,
                                  int v_stride, int h_v, int h_k, float scale) {
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
        o += (s[base_s + kdim * d + u] * q[qk0 + kdim]) * scale;
    out[v0 + u] = o;
}
// GDN AR — t토큰 순차 (block=128=d_state 열, blockIdx.x=pair)
extern "C" __global__ void gdn_ar_t(float* s, const float* q, const float* k, const float* v,
                                    const float* beta_ge, float* out, int d, int k_stride,
                                    int v_stride, int h_v, int h_k, float scale, int t) {
    int pair = blockIdx.x;
    int u = threadIdx.x;
    if (u >= d) return;
    int base_s = pair * d * d;
    for (int ti = 0; ti < t; ti++) {
        int b = 0;
        int h = pair % h_v;
        int kh = h % h_k;
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
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
            o += (s[base_s + kdim * d + u] * q[qk0 + kdim]) * scale;
        out[v0 + u] = o;
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


// QSA attention score: dot(q_head, ck[p]) — grid (n_past, n_head, t)
extern "C" __global__ void qsa_score(const float* q, const float* ck, const unsigned* mask,
                                     float* scores, int n_past, int n_head, int n_kv,
                                     int hd, int t_len) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    if (p >= n_past || h >= n_head || t >= t_len) return;
    if (mask[t * n_past + p] == 0u) {
        scores[(t * n_head + h) * n_past + p] = -3.0e38f;
        return;
    }
    int kvh = h / (n_head / n_kv);
    int qb = t * n_head * 2 * hd + h * 2 * hd;
    int kb = p * n_kv * hd + kvh * hd;
    float d = 0.0f;
    for (int i = 0; i < hd; i++) d += q[qb + i] * ck[kb + i];
    scores[(t * n_head + h) * n_past + p] = d;
}
// QSA attention mix: softmax-weighted V sum + gate — sequential deterministic order
extern "C" __global__ void qsa_mix(const float* q, const float* scores, const float* cv,
                                   float* out, int n_past, int n_head, int n_kv,
                                   int hd, int t_len) {
    int d_i = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    if (d_i >= hd || h >= n_head || t >= t_len) return;
    int sbase = (t * n_head + h) * n_past;
    float maxv = scores[sbase];
    for (int p = 0; p < n_past; p++) {
        float sv = scores[sbase + p];
        if (sv > maxv) maxv = sv;
    }
    float sum = exp_cr(scores[sbase] - maxv);
    for (int p = 1; p < n_past; p++) sum += exp_cr(scores[sbase + p] - maxv);
    int kvh = h / (n_head / n_kv);
    float a = 0.0f;
    for (int p = 0; p < n_past; p++) {
        float w = exp_cr(scores[sbase + p] - maxv) / sum;
        if (w != 0.0f) {
            int kb = p * n_kv * hd + kvh * hd;
            a += w * cv[kb + d_i];
        }
    }
    int gb = t * n_head * 2 * hd + h * 2 * hd + hd;
    float g = 1.0f / (1.0f + exp_cr(-q[gb + d_i]));
    out[t * n_head * hd + h * hd + d_i] = a * g;
}

"#;

pub const NAMES: &[&str] = &[
    "quant_q8", "reduce64",
    "gemm_xs", "gemm_q5k", "gemm_q8_0", "gemm_q4k", "gemm_q6k", "gemm_nl", "gemm_q3k",
    "silu_mul", "axpy_scaled", "copy_rows", "rms_part", "rms_finish", "qk_norm_rope",
    "gdn_conv", "gdn_beta_g", "norm_gated_silu", "gdn_ar", "l2_rows2_scale", "split3",
    "qsa_score", "qsa_mix", "gemm_iq3s", "gemm_iq3s_sub", "exp_probe", "dp4a_probe", "bw_probe", "q6k_ab", "tree_probe", "gdn_conv_t", "gdn_ar_t", "kv_append_t", "gemm_q5k_bt", "dot_roof", "gemm_q4k_bt", "gemm_q6k_bt", "gemm_xs_bt",
];
// ─── 원시 HIP ew 계열 (큐브cl ew.rs 산술 이식, 다음 검증 대상) ───


