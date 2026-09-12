
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
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
#ifdef LLM170_FASTEXP
    // 빠른 경로(옵트인): 디바이스 __expf — 1ulp 이내 오차, glibc 올림정확과는
    // 경계에서 다를 수 있다(f64 호너 제거로 커널당 수 µs → 수백 ns).
    return __expf(xf);
#else
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
#endif
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
