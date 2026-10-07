//! 공통 트윈 연산 — RMS·QAT 시뮬(FP8/FP4)·Hadamard·YaRN RoPE·활성화.
//!
//! 모든 스케일 산술은 참조 커널(kernel.py `act_quant`·`fp4_act_quant`·
//! `fast_round_scale`)의 비트 조작 경로를 그대로 재현한다(보고서 §9.1-9.3):
//! - FP8: 블록 amax → 하한 1e-4 → s = 2^ceil(log2(amax/448)) (ue8m0,
//!   `pow2_ceil`) → x/s를 ±448 클램프 후 e4m3 RNE → ×s.
//! - FP4: 32원소 블록, amax 하한 6·2^-126, s = 2^ceil(log2(amax/6)), ±6 클램프
//!   e2m1 RNE → ×s.
//!
//! 활성화(sigmoid/silu/softplus)는 `crate::ops`의 exp_cr 계열을 재사용 —
//! GPU 커널과 동일 연산열(프로젝트 비트 동일성 규율).

use crate::ops::{silu, softplus};

/// 참조 구현의 bf16 경계 캐스트 — RNE(짝수로 반올림) 1회.
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let hi = ((b >> 16) as u16) as u32 & 1;
    f32::from_bits(((b + 0x7FFF + hi) >> 16) << 16)
}

/// 슬라이스 전체 bf16 반올림.
#[inline]
pub fn bf16_round_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = bf16_round(*v);
    }
}

/// 가중 RMSNorm (참조 RMSNorm.forward) — f32 내부, 출력은 bf16 경계 반올림.
/// var = mean(x²), x·rsqrt(var+eps)·w. 합은 f32 순차(오름차순).
pub fn rms_norm_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| bf16_round(v * scale * g))
        .collect()
}

/// 비가중 RMS 배율 — q 헤드 정규화(`q *= rsqrt(mean(q²)+eps)`)·hc_pre용.
/// 반환값은 f32 스케일(반올림 없음 — 곱하는 쪽에서 경계 처리).
#[inline]
pub fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}

/// y[m,n] = Σ_k x[m,k]·w[k,n] — f32 누산, k 오름차순(결정적).
/// w는 k-major([k][n], torch Linear.weight[out,in]의 전치) 저장.
pub fn gemm_nt(x: &[f32], w: &[f32], k: usize, n: usize, y: &mut [f32]) {
    let m = x.len() / k;
    assert_eq!(x.len(), m * k);
    y.fill(0.0);
    for i in 0..m {
        let xr = &x[i * k..(i + 1) * k];
        let yr = &mut y[i * n..(i + 1) * n];
        for (kk, &xv) in xr.iter().enumerate() {
            let wr = &w[kk * n..(kk + 1) * n];
            for (j, &wv) in wr.iter().enumerate() {
                yr[j] += xv * wv;
            }
        }
    }
}

/// 2^ceil(log2(x)) — kernel.py `fast_log2_ceil`+`fast_pow2` 비트 경로 재현.
/// x > 0 가정. 지수만 보고 가수비트 있으면 올림.
#[inline]
pub fn pow2_ceil(x: f32) -> f32 {
    debug_assert!(x > 0.0 && x.is_finite());
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    let l2 = e - 127 + i32::from(b & 0x7F_FFFF != 0);
    f32::from_bits(((l2 + 127) as u32) << 23)
}

/// f32 → e4m3fn 비트 (RNE). 입력은 |x| ≤ 448 클램프 후라고 가정
/// (클램프는 호출부 책임 — 참조 커널도 clamp 후 cast).
/// 하위 비트: S|EEEE|MMM, 바이어스 7, 최대 정규수 448, 0x7F=NaN.
pub fn f32_to_e4m3(x: f32) -> u8 {
    // -0.0도 부호 비트 보존(x<0.0은 -0.0에서 거짓).
    let sign = u8::from(x.is_sign_negative()) << 7;
    let a = x.abs();
    if a < 2.0f32.powi(-10) {
        // 절반 최소 비정규(2^-10) 미만 → 0 (RNE 동점 짝수=0).
        return sign;
    }
    if a < 2.0f32.powi(-6) {
        // 비정규 영역: ulp = 2^-9, q = a·2^9 를 정수로 RNE.
        let q = a * 512.0;
        let r = q.round_ties_even();
        if r >= 8.0 {
            // 8·2^-9 = 2^-6 → 최소 정규수로 승격.
            return sign | (1 << 3);
        }
        return sign | r as u8;
    }
    let b = a.to_bits();
    let e = (((b >> 23) & 0xFF) as i32) - 127;
    let mant = b & 0x7F_FFFF;
    // 가수 23비트 → 3비트 RNE (버려지는 20비트의 반올림).
    const SH: u32 = 20;
    let rem = mant & ((1 << SH) - 1);
    let half = 1u32 << (SH - 1);
    let mut mm = mant >> SH;
    let mut ee = e;
    if rem > half || (rem == half && (mm & 1) == 1) {
        mm += 1;
    }
    if mm == 8 {
        mm = 0;
        ee += 1;
    }
    let e4 = (ee + 7) as u8;
    if e4 > 15 || (e4 == 15 && mm > 7) {
        // 448 초과 — 참조에선 클램프로 불가. 방어적으로 NaN.
        return sign | 0x7F;
    }
    sign | (e4 << 3) | mm as u8
}

/// e4m3fn 비트 → f32 (정확).
pub fn e4m3_to_f32(u: u8) -> f32 {
    // 부호를 ±1.0으로 인코딩(±0.0이면 곱셈에서 전부 0이 된다).
    let sign = f32::from_bits((((u & 0x80) as u32) << 24) | 0x3F80_0000);
    let e4 = ((u >> 3) & 0xF) as i32;
    let m = (u & 7) as u32;
    if e4 == 15 && m == 7 {
        return f32::NAN.copysign(sign);
    }
    if e4 == 0 {
        // 비정규: m·2^-9
        return sign * (m as f32) * 2.0f32.powi(-9);
    }
    let bits = (((e4 - 7 + 127) as u32) << 23) | (m << 20);
    sign * f32::from_bits(bits)
}

/// f32 → e2m1 값 (RNE) — 그리드 {0,.5,1,1.5,2,3,4,6}. |x| ≤ 6 가정.
/// 동점(정확히 중점)은 짝수 인덱스 그리드로 (RNE).
#[inline]
pub fn f32_to_e2m1(x: f32) -> f32 {
    let a = x.abs();
    if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    }
}

/// FP8-sim(inplace 양자화-디양자화) — 행렬 [rows×cols] 블록 `block`(128 또는 64).
/// amax 하한 1e-4, s=pow2_ceil(amax·(1/448)), e4m3 RNE 후 ×s.
/// e4m3값×2^k는 bf16/f32에서 정확 — 추가 반올림 없음.
pub fn fp8_sim(x: &mut [f32], cols: usize, block: usize) {
    let rows = x.len() / cols;
    for r in 0..rows {
        for c0 in (0..cols).step_by(block) {
            let blk = &mut x[r * cols + c0..r * cols + (c0 + block).min(cols)];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = pow2_ceil(amax * (1.0 / 448.0));
            for v in blk.iter_mut() {
                let q = (*v / s).clamp(-448.0, 448.0);
                *v = e4m3_to_f32(f32_to_e4m3(q)) * s;
            }
        }
    }
}

/// FP4-sim — 32원소 블록, amax 하한 6·2^-126, s=pow2_ceil(amax·(1/6)), ±6 클램프.
pub fn fp4_sim(x: &mut [f32]) {
    for blk in x.chunks_mut(32) {
        let mut amax = 0.0f32;
        for &v in blk.iter() {
            amax = amax.max(v.abs());
        }
        amax = amax.max(6.0 * 2.0f32.powi(-126));
        let s = pow2_ceil(amax * (1.0 / 6.0));
        for v in blk.iter_mut() {
            let q = (*v / s).clamp(-6.0, 6.0);
            *v = f32_to_e2m1(q).copysign(q) * s;
        }
    }
}

/// MXFP4 전문가 가중치 디양자화 — GGUF/원본 체크포인트 경로용(보고서 §5).
/// `packed`: K방향 2개/바이트(낮은 니블이 k=2j, 높은 니블이 k=2j+1 —
/// float4_e2m1fn_x2 패킹 관례), `scales`: e8m0(바이어스 127) [n][k/32].
/// w[n][k] = e2m1 × 2^(scale−127) → f32 k-major 반환([k][n]).
pub fn mxfp4_dequant(packed: &[u8], scales: &[u8], n: usize, k: usize) -> Vec<f32> {
    /// 니블 → e2m1 값 (부호 비트 3, 크기 3비트).
    #[inline]
    fn e2m1(nib: u8) -> f32 {
        const TAB: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let v = TAB[(nib & 7) as usize];
        if nib & 8 != 0 { -v } else { v }
    }
    let mut w = vec![0.0f32; k * n];
    for row in 0..n {
        let prow = &packed[row * (k / 2)..(row + 1) * (k / 2)];
        for kb in 0..k.div_ceil(32) {
            let s = 2.0f32.powi(scales[row * k.div_ceil(32) + kb] as i32 - 127);
            for j in 0..32 {
                let kk = kb * 32 + j;
                if kk >= k {
                    break;
                }
                let byte = prow[kk / 2];
                let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
                w[kk * n + row] = e2m1(nib) * s;
            }
        }
    }
    w
}

/// 자연 순서 WHT-128 + N^-0.5 스케일 — kernel.py `rotate_activation`
/// (hadamard_transform, scale=N^-0.5) 재현. f32 순차 버터플라이.
/// N은 2의 거듭제곱(인덱서 q/kv의 128).
pub fn hadamard_rotate(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut width = 1;
    while width < n {
        let mut base = 0;
        while base < n {
            for i in 0..width {
                let (a, b) = (x[base + i], x[base + width + i]);
                x[base + i] = a + b;
                x[base + width + i] = a - b;
            }
            base += 2 * width;
        }
        width *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// YaRN 주파수 테이블 — model.py `precompute_freqs_cis` 재현.
/// dim=64(페어 32), 원본 길이>0이면 YaRN 램프 blend.
/// 저장: [len][dim/2][2] = (cos, sin) 쌍, f32.
pub struct RopeTable {
    pub cs: Vec<f32>,
    pub half: usize,
}

#[inline]
fn find_correction_dim(num_rot: f64, dim: usize, base: f64, max_len: usize) -> f64 {
    dim as f64 * (max_len as f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
}

impl RopeTable {
    /// base·yarn 파라미터로 [0, len) 위치 테이블 구축.
    pub fn build(
        dim: usize,
        len: usize,
        base: f64,
        yarn: bool,
        factor: f64,
        orig_len: usize,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Self {
        let half = dim / 2;
        let mut freqs = vec![0.0f32; half];
        for (p, f) in freqs.iter_mut().enumerate() {
            *f = (base.powf((2 * p) as f64 / dim as f64) as f32).recip();
        }
        if yarn && orig_len > 0 {
            // find_correction_range(beta_fast, beta_slow) — clamp [0, dim-1].
            let low = find_correction_dim(beta_fast, dim, base, orig_len)
                .floor()
                .max(0.0) as usize;
            let high = (find_correction_dim(beta_slow, dim, base, orig_len)
                .ceil()
                .min(dim as f64 - 1.0)) as usize;
            // linear_ramp_factor(min=low, max=high) — smooth = 1 - ramp.
            let (lo, hi) = (low as f32, high as f32);
            for (p, f) in freqs.iter_mut().enumerate() {
                let mut ramp = ((p as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
                if low == high {
                    ramp = ((p as f32 - lo) / (hi - lo + 0.001)).clamp(0.0, 1.0);
                }
                let smooth = 1.0 - ramp;
                *f = (*f / factor as f32) * (1.0 - smooth) + *f * smooth;
            }
        }
        let mut cs = vec![0.0f32; len * half * 2];
        for pos in 0..len {
            for (p, &f) in freqs.iter().enumerate() {
                let angle = pos as f32 * f;
                cs[pos * half * 2 + p * 2] = angle.cos();
                cs[pos * half * 2 + p * 2 + 1] = angle.sin();
            }
        }
        RopeTable { cs, half }
    }

    /// 위치 pos의 (cos,sin) 슬라이스 [half][2].
    #[inline]
    pub fn at(&self, pos: usize) -> &[f32] {
        &self.cs[pos * self.half * 2..(pos + 1) * self.half * 2]
    }
}

/// 인터리브 페어 RoPE — (x[2p], x[2p+1]) 복소수 곱 (model.py
/// `apply_rotary_emb`, view_as_complex 짝 페어링). inverse=true → 켤레
/// (출력 역회전, 어텐션 o 비회전용). f32 산술, bf16 반올림은 호출부.
pub fn rope_apply(x: &mut [f32], cs: &[f32], inverse: bool) {
    let half = cs.len() / 2;
    debug_assert_eq!(x.len(), half * 2);
    for p in 0..half {
        let (c, s) = (
            cs[p * 2],
            if inverse {
                -cs[p * 2 + 1]
            } else {
                cs[p * 2 + 1]
            },
        );
        let (x0, x1) = (x[p * 2], x[p * 2 + 1]);
        x[p * 2] = x0 * c - x1 * s;
        x[p * 2 + 1] = x0 * s + x1 * c;
    }
}

/// sqrtsoftplus — Gate.score_func (보고서 §5). 게이트는 f32 전용 산술.
#[inline]
pub fn sqrtsoftplus(g: f32) -> f32 {
    softplus(g).sqrt()
}

/// SwiGLU limit — 비대칭 클램프(보고서 §5): up-proj [-L, L], gate-proj max L.
#[inline]
pub fn swiglu_limit(gate: f32, up: f32, limit: f32) -> f32 {
    silu(gate.min(limit)) * up.clamp(-limit, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// e4m3 RNE 왕복 — 대표값·비정규·동점.
    #[test]
    fn e4m3_roundtrip() {
        // 정확히 표현되는 값들.
        for v in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 12.0, 448.0] {
            assert_eq!(e4m3_to_f32(f32_to_e4m3(v)), v, "v={v}");
            assert_eq!(e4m3_to_f32(f32_to_e4m3(-v)), -v, "v=-{v}");
        }
        // 비정규 최소 2^-9, 절반(2^-10)은 RNE 짝수(0)로.
        assert_eq!(e4m3_to_f32(f32_to_e4m3(2.0f32.powi(-9))), 2.0f32.powi(-9));
        assert_eq!(f32_to_e4m3(2.0f32.powi(-10)), 0);
        assert_eq!(
            e4m3_to_f32(f32_to_e4m3(1.5 * 2.0f32.powi(-10))),
            2.0f32.powi(-9)
        );
        // 8비트 그리드 밀도: e4m3_to_f32는 전부 정확.
        for u in 0..=255u8 {
            if u == 0x7F || u == 0xFF {
                continue;
            }
            let v = e4m3_to_f32(u);
            assert!(v.is_finite());
            assert_eq!(f32_to_e4m3(v), u, "u={u:#x} v={v}");
        }
    }

    /// fp8_sim — 스케일 공식·왕복 오차 상한·amax 하한.
    #[test]
    fn fp8_sim_known_values() {
        // amax=448·2^k 블록: s=2^k, 그리드 간격 = s·2^(e-3) 최대 상대오차 6.25%.
        let mut x = vec![100.0f32, -200.0, 448.0, 1.0];
        let orig = x.clone();
        fp8_sim(&mut x, 4, 4);
        let s = pow2_ceil(448.0 * (1.0 / 448.0));
        assert_eq!(s, 1.0);
        for (&o, &v) in orig.iter().zip(x.iter()) {
            let err = (o - v).abs() / o.abs().max(1e-6);
            assert!(err < 0.07, "o={o} v={v} err={err}");
        }
        // 전부 0인 블록 — amax 하한 1e-4 → 0 유지.
        let mut z = vec![0.0f32; 128];
        fp8_sim(&mut z, 128, 128);
        assert!(z.iter().all(|&v| v == 0.0));
        // ue8m0 스케일: amax=449 → s=2.
        assert_eq!(pow2_ceil(449.0 * (1.0 / 448.0)), 2.0);
        assert_eq!(pow2_ceil(448.0 * (1.0 / 448.0)), 1.0);
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.5), 2.0);
    }

    /// fp4_sim — 그리드 값 정확 재현 + 하한 스케일.
    #[test]
    fn fp4_sim_grid() {
        let mut x = vec![6.0f32, -3.0, 0.5, 1.25, 5.9, 0.1];
        let expect = [6.0f32, -3.0, 0.5, 1.0, 6.0, 0.0]; // amax=6 → s=1
        fp4_sim(&mut x);
        for (i, (&v, &e)) in x.iter().zip(expect.iter()).enumerate() {
            assert_eq!(v, e, "i={i}");
        }
        // 하한: 전부 0 → s = pow2ceil(6·2^-126/6) = 2^-126, 0 유지.
        let mut z = vec![0.0f32; 32];
        fp4_sim(&mut z);
        assert!(z.iter().all(|&v| v == 0.0));
        // 동점 규칙: 정확히 중점 2.5 → RNE 짝수 그리드 2.
        assert_eq!(f32_to_e2m1(2.5), 2.0);
        assert_eq!(f32_to_e2m1(3.5), 4.0);
        assert_eq!(f32_to_e2m1(0.25), 0.0);
        assert_eq!(f32_to_e2m1(0.75), 1.0);
    }

    /// MXFP4 디양자화 — 패킹·스케일 공식 w=e2m1·2^(s-127).
    #[test]
    fn mxfp4_dequant_known() {
        // 1행 k=8: 니블 [1,2,9,0,8,15,4,7] → [0.5,1,-0.5,0,-0? ...]
        let packed: [u8; 4] = [0x21, 0x09, 0xF8, 0x74];
        let scales: [u8; 1] = [127]; // 2^0
        let w = mxfp4_dequant(&packed, &scales, 1, 8);
        assert_eq!(w.len(), 8);
        assert_eq!(w[0], 0.5); // 니블 1
        assert_eq!(w[1], 1.0); // 니블 2
        assert_eq!(w[2], -0.5); // 니블 9 = 부호비트 + 크기 1 → -0.5
        assert_eq!(w[3], 0.0);
        // 스케일 128 → 2^1.
        let w2 = mxfp4_dequant(&packed, &[128], 1, 8);
        assert_eq!(w2[0], 1.0);
    }

    /// Hadamard — 상수 입력 → H_orth·(c·1) = c·√N (scale=N^-0.5 —
    /// fast_hadamard_transform 규약: 비정규 WHT에 N^-0.5를 곱한다).
    #[test]
    fn hadamard_constant() {
        let mut x = vec![3.0f32; 128];
        hadamard_rotate(&mut x);
        assert!((x[0] - 3.0 * 128.0f32.sqrt()).abs() < 1e-3, "x0={}", x[0]);
        assert!(x[1..].iter().all(|&v| v.abs() < 1e-4));
    }

    /// YaRN RoPE — 회전/역회전 항등 + YaRN 주파수 감소 + L0-1(비YaRN) 동일성.
    #[test]
    fn rope_roundtrip_and_yarn() {
        let dim = 64;
        let t = RopeTable::build(dim, 16, 10000.0, false, 16.0, 65536, 32.0, 1.0);
        let mut x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.37 - 4.0).collect();
        let orig = x.clone();
        rope_apply(&mut x, t.at(7), false);
        assert_ne!(x, orig);
        rope_apply(&mut x, t.at(7), true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // pos=0 → 항등.
        let mut y = orig.clone();
        rope_apply(&mut y, t.at(0), false);
        assert_eq!(y, orig);
        // YaRN: factor 16 → 고차원(빠른 주파수)은 freq/16 블렌드.
        let ty = RopeTable::build(dim, 4, 160000.0, true, 16.0, 65536, 32.0, 1.0);
        // p=0: freq=1 불변 검증 (angle=pos·1 동일).
        let a_plain = RopeTable::build(dim, 2, 160000.0, false, 16.0, 0, 32.0, 1.0);
        assert_eq!(ty.at(1)[0], a_plain.at(1)[0]); // cos(1·1)
        // p=half-1(가장 빠른 주파수): ramp=1 → smooth=0 → freq/16.
        // 작은 각도의 sin ≈ 각도 — acos 대신 sin 성분으로 비교(정밀도).
        let p = (dim / 2 - 1) * 2 + 1; // sin 성분 인덱스
        let sin_y = ty.at(1)[p];
        let sin_p = a_plain.at(1)[p];
        assert!(
            (sin_y * 16.0 - sin_p).abs() < 1e-4 * sin_p.max(1e-6),
            "sin_y={sin_y} sin_p={sin_p} (16배 관계 기대)"
        );
    }

    /// 활성화 — sqrtsoftplus·swiglu 비대칭 클램프·bf16 RNE.
    #[test]
    fn activations_and_bf16() {
        assert!((sqrtsoftplus(0.0) - 2.0f32.ln().sqrt()).abs() < 1e-6);
        // gate=11 → min 10, up=11 → clamp 10.
        let v = swiglu_limit(11.0, 11.0, 10.0);
        let want = crate::ops::silu(10.0) * 10.0;
        assert!((v - want).abs() < 1e-6);
        // up 하한: -11 → -10.
        let v2 = swiglu_limit(0.5, -11.0, 10.0);
        assert!((v2 - crate::ops::silu(0.5) * -10.0).abs() < 1e-6);
        // gate는 하한 없음: -50 그대로.
        let v3 = swiglu_limit(-50.0, 1.0, 10.0);
        assert!((v3 - crate::ops::silu(-50.0) * 1.0).abs() < 1e-6);
        // bf16 RNE: 1.0 + 2^-9 (0.5ulp 동점) → 짝수(1.0);
        // 1.0+1.5ulp 동점 → 짝수 가수 2ulp = 1.0+2^-7; 3.0+0.375ulp → 3.0.
        assert_eq!(bf16_round(1.0 + 2.0f32.powi(-9)), 1.0);
        assert_eq!(
            bf16_round(1.0 + 2.0f32.powi(-8) + 2.0f32.powi(-9)),
            1.0 + 2.0f32.powi(-7)
        );
        assert_eq!(bf16_round(3.0 + 2.0f32.powi(-8) + 2.0f32.powi(-9)), 3.0);
        assert_eq!(bf16_round(0.0), 0.0);
    }
}
