
pub fn sq_sum(x: &[f32]) -> f64 {
    const SEG: usize = 32;
    let n = x.len();
    let chunk = n.div_ceil(SEG);
    let mut sum = 0.0f64;
    for u in 0..SEG {
        let lo = u * chunk;
        if lo >= n {
            break;
        }
        let hi = (lo + chunk).min(n);
        let mut part = 0.0f32;
        for &v in &x[lo..hi] {
            part += v * v;
        }
        sum += part as f64;
    }
    sum
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "rms_norm: x/w 길이 불일치");
    let sum = sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

pub fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / sum.sqrt().max(eps);
    x.iter().map(|&v| v * scale).collect()
}

pub fn exp_cr(x: f32) -> f32 {
    let xd = x as f64;
    if xd > 88.72 {
        return f32::INFINITY;
    }
    if xd < -103.97 {
        return 0.0;
    }
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
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

pub fn ln_cr(v: f64) -> f64 {
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64;
    let k = e - 1023;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52));
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
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
    let lnm = 2.0 * t * q;
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
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

pub fn rope_head(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    rope_head_dim(head, pos, n_rot, n_rot, base);
}

pub fn rope_head_dim(head: &mut [f32], pos: u32, n_rot: usize, head_dim: usize, base: f32) {
    let half = head_dim / 2;
    let pairs = n_rot / 2;
    for p in 0..pairs {
        let theta = base.powf(-(2.0 * p as f32) / head_dim as f32);
        let angle = pos as f32 * theta;
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (head[p] as f64, head[p + half] as f64);
        let (cf, sf) = (c as f64, s as f64);
        head[p] = (x0 * cf - x1 * sf) as f32;
        head[p + half] = (x0 * sf + x1 * cf) as f32;
    }
}

pub fn rms_norm_plain(x: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().map(|&v| v * scale).collect()
}

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    let x3 = x * x * x;
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x3)).tanh())
}

#[inline]
pub fn logit_softcap(x: f32, c: f32) -> f32 {
    if c > 0.0 { c * (x / c).tanh() } else { x }
}

pub fn bf16_rne(v: f32) -> f32 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    f32::from_bits(rounded & 0xFFFF_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "길이 불일치")]
    fn rms_norm_length_mismatch_panics() {
        let _ = rms_norm(&[1.0, 2.0], &[1.0], 1e-6);
    }

    #[test]
    fn rms_norm_unit_scale() {
        let y = rms_norm(&[1.0f32; 4], &[1.0f32; 4], 0.0);
        for v in y {
            assert!((v - 1.0).abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn logit_softcap_identity_monotone_saturates() {
        assert_eq!(logit_softcap(3.5, 0.0), 3.5, "c=0 항등");
        assert_eq!(logit_softcap(0.0, 30.0), 0.0);
        let y = logit_softcap(30.0, 30.0);
        assert!((y - 30.0 * 1.0f32.tanh()).abs() < 1e-6, "{y}");
        assert!(logit_softcap(-10.0, 30.0) < logit_softcap(10.0, 30.0));
        let big = logit_softcap(1000.0, 30.0);
        assert!(big > 29.9 && big <= 30.0, "{big}");
    }
}
