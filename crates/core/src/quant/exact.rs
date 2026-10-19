//! [2026-10-19] **정확 내적 — 진리층(accuracy truth)**.
//!
//! W4A16의 수학적 정답은 실제로 계산 가능하다: 항 = (nib−8)·scale·x 에서
//! (nib−8) = 정수(≤15, 정확) · scale = bf16/f16(mantissa ≤11비트, f64 정확) ·
//! x = f32(정확) → **곱이 f64에 정확히 표현**된다. f64 보상(Neumaier) 누산이면
//! 남는 오차 = 가산 반올림뿐(~1e-16). GPU/vLLM의 f16/f32 클래스(1e-3~1e-5)와의
//! 거리를 재는 기준 — "vLLM보다 정확"의 판정은 이 값과의 거리로 정의한다.
//! 검증 = i128 정수 정확합 대조 테스트(하단) — 부동소수 가정 없이 독립 계산.
//!
//! 한계(기록): CPU 참조의 초월함수(exp/tanh)·norm 누산은 기존 f32 클래스 그대로 —
//! 잔여 오차 ~1e-7(행렬 누산 1e-5~1e-4 대비 소항). 필요 시 후속 모드로 확장.

use super::deq::{bf16_to_f32, half_to_f32};

/// 보상 누산(Neumaier) — 항이 정확하면 합의 반올림 오차만 남는다.
#[derive(Default, Clone, Copy)]
struct Neumaier {
    sum: f64,
    c: f64,
}

impl Neumaier {
    #[inline]
    fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.c += (self.sum - t) + v;
        } else {
            self.c += (v - t) + self.sum;
        }
        self.sum = t;
    }
    #[inline]
    fn value(self) -> f64 {
        self.sum + self.c
    }
}

/// W4A16 행 내적의 정확값 — 디스크 규약 바이트(q/s) + f32 x 그대로.
/// qrow = 그 행의 k/2바이트, srow = 그 행의 (k/group)×2바이트.
pub fn dot_row_w4a16_exact_bytes(
    qrow: &[u8],
    srow: &[u8],
    x: &[f32],
    k: usize,
    group: usize,
    scale_bf16: bool,
) -> f64 {
    let nb = k / group;
    let mut acc = Neumaier::default();
    for b in 0..nb {
        let bits = u16::from_le_bytes([srow[2 * b], srow[2 * b + 1]]);
        let sc = if scale_bf16 {
            bf16_to_f32(bits)
        } else {
            half_to_f32(bits)
        } as f64;
        for i in 0..group {
            let woff = 4 * (b * (group / 8) + i / 8);
            let w =
                u32::from_le_bytes([qrow[woff], qrow[woff + 1], qrow[woff + 2], qrow[woff + 3]]);
            let nib = ((w >> (4 * (i % 8))) & 0xF) as i32;
            acc.add((nib - 8) as f64 * sc * x[b * group + i] as f64);
        }
    }
    acc.value()
}

/// 진리층 전역 스위치 — CPU matmul(W4A16 split)이 정확 경로를 탄다.
/// 프로브(w4a16-ref)가 켠다. env 아님(연산 경로 아님 — 진단 모드).
static HI_PREC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_hi_prec(on: bool) {
    HI_PREC.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn hi_prec() -> bool {
    HI_PREC.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32(양의 정규수) → (mantissa, exp): v = m·2^e, m 정수.
    fn dyadic(v: f32) -> (i128, i32) {
        let b = v.to_bits();
        assert_eq!(b >> 31, 0, "테스트: 양수만");
        let e = ((b >> 23) & 0xFF) as i32 - 127 - 23;
        let m = (b & 0x7F_FFFF) as i128 | (1 << 23);
        (m, e)
    }

    /// 정확 내적 = i128 정수 정확합과 1e-14 상대 이내 일치.
    #[test]
    fn exact_dot_matches_integer_sum() {
        let (k, group) = (64usize, 32usize);
        // 결정적 재료 — splitmix64(core::sampler 규약과 동일 상수).
        let mut st = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = st;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // q: k/8 워드(니블 랜덤), s: group당 bf16, x: f32 [0.25, 2)
        let q: Vec<u8> = (0..k / 2).map(|_| (next() & 0xFF) as u8).collect();
        let s_bf16: Vec<u16> = (0..k / group)
            .map(|_| (next() % 2048) as u16 + 0x3A00) // ~[0.0005, 0.1] 근방
            .collect();
        let s: Vec<u8> = s_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x: Vec<f32> = (0..k)
            .map(|_| 0.25 + (next() % 1_000_000) as f32 / 1_000_000.0 * 1.75)
            .collect();

        // i128 정확합 — 각 항 = (nib−8)·m_s·m_x · 2^(e_s+e_x).
        let mut terms: Vec<(i128, i32)> = Vec::new();
        for b in 0..k / group {
            let sc = bf16_to_f32(s_bf16[b]);
            let (ms, es) = dyadic(sc);
            for i in 0..group {
                let woff = 4 * (b * (group / 8) + i / 8);
                let w = u32::from_le_bytes([q[woff], q[woff + 1], q[woff + 2], q[woff + 3]]);
                let nib = ((w >> (4 * (i % 8))) & 0xF) as i128;
                let (mx, ex) = dyadic(x[b * group + i]);
                terms.push(((nib - 8) * ms * mx, es + ex));
            }
        }
        let min_e = terms.iter().map(|t| t.1).min().unwrap();
        let mut isum: i128 = 0;
        for (m, e) in &terms {
            isum += m << (e - min_e);
        }
        let exact = isum as f64 * 2f64.powi(min_e);

        let got = dot_row_w4a16_exact_bytes(&q, &s, &x, k, group, true);
        let rel = ((got - exact) / exact).abs();
        assert!(
            rel < 1e-14,
            "정확 내적 상대오차 {rel:.3e} (got {got}, exact {exact})"
        );
        // 대조: 기존 f32 순차 누산은 같은 값에서 더 멀다(클래스 확인 — 통계적).
        let mut acc = 0.0f32;
        for b in 0..k / group {
            let sc = bf16_to_f32(s_bf16[b]);
            for i in 0..group {
                let woff = 4 * (b * (group / 8) + i / 8);
                let w = u32::from_le_bytes([q[woff], q[woff + 1], q[woff + 2], q[woff + 3]]);
                let nib = ((w >> (4 * (i % 8))) & 0xF) as i32;
                acc += (nib - 8) as f32 * sc * x[b * group + i];
            }
        }
        let rel32 = ((acc as f64 - exact) / exact).abs();
        eprintln!("[exact-dot] f64 보상 {rel:.3e} · f32 순차 {rel32:.3e}");
        assert!(rel32 > rel, "f32 순차가 정확값보다 멀어야 함(클래스)");
    }
}
