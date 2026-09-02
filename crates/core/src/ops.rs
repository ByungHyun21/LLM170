//! 기본 연산: norms, 활성화, RoPE — llama.cpp ggml-cpu 시맨틱스 대응.
//!
//! - rms_norm: f64 누적, scale = 1/sqrt(mean(x²)+eps), 감마는 GGUF 저장값(1+w) 그대로 곱.
//! - l2_norm: scale = 1/max(sqrt(Σx²), eps) (ops.cpp — eps 는 floor).
//! - rope: 인접 페어 (2i, 2i+1), θ_p = base^(−p/n_pairs). IMROPE의 텍스트 전용 퇴화형
//!   (모든 위치 성분이 동일하면 표준 RoPE와 동일 — ggml mrope_cache_init 참조).

/// 제곱합 표준 순서 (2026-09-02, P0): 32세그먼트 병렬 친화적 구조 —
/// 각 세그먼트 f64 순차 누산 후 세그먼트 합을 순차 결합.
/// GPU rms_rows_part/finish 커널과 동일 순서 → 병렬화해도 비트 일치.
pub fn sq_sum(x: &[f32]) -> f64 {
    const SEG: usize = 32;
    let n = x.len();
    let chunk = n.div_ceil(SEG);
    let mut sum = 0.0f64;
    for u in 0..SEG {
        let lo = u * chunk;
        if lo >= n { break; }
        let hi = (lo + chunk).min(n);
        let mut part = 0.0f64;
        for &v in &x[lo..hi] {
            part += (v as f64) * (v as f64);
        }
        sum += part;
    }
    sum
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

pub fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / (sum.sqrt() as f32).max(eps);
    x.iter().map(|&v| v * scale).collect()
}


/// f64 fma 호너 올림-정확 exp — HIP 커널 exp_cr과 동일 연산열 (비트 동일).
/// glibc expf는 ½ulp 오차(2026-09-03 실측 244/4096 불일치) — W4A8
/// 비트계약상 양 플랫폼이 같은 알고리즘을 쓴다.
pub fn exp_cr(x: f32) -> f32 {
    let xd = x as f64;
    if xd > 88.72 {
        return f32::INFINITY;
    }
    if xd < -103.97 {
        return 0.0;
    }
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const INV_LN2: f64 = 1.44269504088896338700e+00;
    let kd = (xd * INV_LN2).round_ties_even();
    let k = kd as i64;
    let mut r = (-kd).mul_add(LN2_HI, xd);
    r = (-kd).mul_add(LN2_LO, r);
    let mut p = 1.0f64 / 1307674368000.0;
    p = p.mul_add(r, 1.0 / 479001600.0);
    p = p.mul_add(r, 1.0 / 39916800.0);
    p = p.mul_add(r, 1.0 / 3628800.0);
    p = p.mul_add(r, 1.0 / 362880.0);
    p = p.mul_add(r, 1.0 / 40320.0);
    p = p.mul_add(r, 1.0 / 5040.0);
    p = p.mul_add(r, 1.0 / 720.0);
    p = p.mul_add(r, 1.0 / 120.0);
    p = p.mul_add(r, 1.0 / 24.0);
    p = p.mul_add(r, 1.0 / 6.0);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);
    if k > 127 {
        return f32::INFINITY;
    }
    let scale = f64::from_bits(((k + 1023) as u64) << 52);
    (p * scale) as f32
}

/// f64 자연로그 — atanh 급수 fma 호너. HIP 커널 ln_cr과 동일 연산열.
/// 정규수 v ≥ 1 전용 (softplus log1p 경로 — y+1 ≥ 1).
pub fn ln_cr(v: f64) -> f64 {
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64;
    let k = e - 1023;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52));
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // atanh 급수: ln(m) = 2t(1 + t²/3 + t⁴/5 + …) — s^25/25 절단
    let mut q = 1.0f64 / 25.0;
    q = q.mul_add(t2, 1.0 / 23.0);
    q = q.mul_add(t2, 1.0 / 21.0);
    q = q.mul_add(t2, 1.0 / 19.0);
    q = q.mul_add(t2, 1.0 / 17.0);
    q = q.mul_add(t2, 1.0 / 15.0);
    q = q.mul_add(t2, 1.0 / 13.0);
    q = q.mul_add(t2, 1.0 / 11.0);
    q = q.mul_add(t2, 1.0 / 9.0);
    q = q.mul_add(t2, 1.0 / 7.0);
    q = q.mul_add(t2, 1.0 / 5.0);
    q = q.mul_add(t2, 1.0 / 3.0);
    q = q.mul_add(t2, 1.0);
    // ln(v) = ln(m) + k·ln2 — 2단 ln2 (LN2_HI + LN2_LO) fma 합
    let lnm = 2.0 * t * q;
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    let kh = (k as f64) * LN2_HI;
    let kl = (k as f64) * LN2_LO;
    let s1 = lnm + kh;
    let s2 = (lnm - s1) + kh;
    s1 + (s2 + kl)
}

pub fn log1p_cr(y: f32) -> f32 {
    ln_cr(y as f64 + 1.0) as f32
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp_cr(-x))
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_cr(-x))
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { log1p_cr(exp_cr(x)) }
}

/// 헤드 벡터의 첫 n_rot 차원에 RoPE 적용 (나머지 통과).
/// qwen35 = IMROPE → **NEOX 페어링**: 페어 p = (p, p + n_rot/2),
/// θ_p = base^(−2p/n_rot). 텍스트 전용에서는 모든 위치 성분이 같아 표준 NEOX와 동일.
/// (ggml.h: GGML_ROPE_TYPE_IMROPE 40 — "still NEOX ordering")
/// RoPE 단일 헤드 — f64 중간연산: f32×f32 곱·차는 f64에서 전부 정확(
/// FMA 수축 무영향), 최종 1회 f32 반올림. GPU ew::qk_norm_rope와 동일
/// 연산열 (2026-09-02 브리지 제거 — P1 수축 RCA의 해법).
pub fn rope_head(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for p in 0..half {
        let theta = base.powf(-(2.0 * p as f32) / n_rot as f32);
        let angle = pos as f32 * theta;
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (head[p] as f64, head[p + half] as f64);
        let (cf, sf) = (c as f64, s as f64);
        head[p] = (x0 * cf - x1 * sf) as f32;
        head[p + half] = (x0 * sf + x1 * cf) as f32;
    }
}
