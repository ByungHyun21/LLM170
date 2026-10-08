//! 양자화 블록 역양자화 — llama.cpp `ggml/src/ggml-quants.c` 대응 (2026-08-30 판).
//!
//! 블록 레이아웃은 `ggml-common.h` 구조체 선언 순서 그대로 바이트 오프셋로 해석한다.
//! 모든 함수는 한 "행"(row, ne[0] 축)의 연속 블록을 f32 로 펼친다.

pub mod deq;
pub mod lane;
pub mod mm;
pub mod q8;

#[cfg(test)]
#[cfg(test)]
use crate::wtype::WType;
pub use deq::*;
pub use lane::*;
pub use mm::*;
pub use q8::*;
#[cfg(test)]
mod w4a8_tests {
    use super::*;

    /// q5_1 산술 정밀 검증 — f32 디퀀트 기준 vs dot_q5_1_q8 vs 레인 미러.
    /// (m 항을 반드시 포함: d=1.0, m=−0.5 f16 고정, 4블록 = 128원소)
    #[test]
    fn q5_1_block_exact() {
        let mut seed = 0x5A5A_1234u64;
        let mut lcg = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        let n = 128usize;
        let mut bytes = vec![0u8; (n / 32) * 24];
        for b in bytes.iter_mut() {
            *b = (lcg() & 0xFF) as u8;
        }
        for blk in bytes.chunks_mut(24) {
            // d = 1.0 (f16 0x3C00), m = -0.5 (f16 0xB800)
            blk[0] = 0x00;
            blk[1] = 0x3C;
            blk[2] = 0x00;
            blk[3] = 0xB8;
        }
        let x: Vec<f32> = (0..n)
            .map(|_| ((lcg() >> 8) as f32 / (1u32 << 24) as f32) - 0.5)
            .collect();
        let y = quantize_row_q8_ref(&x);
        // 기준: f32 디퀀트 × q8 재구성 (측정 대상 산술만 남긴다)
        let mut wv = vec![0.0f32; n];
        dequant_row(WType::Q5_1, &bytes, 0, n as u64, &mut wv);
        let mut want = 0.0f64;
        for i in 0..n {
            want += (wv[i] as f64) * (y[i / 32].d as f64) * (y[i / 32].qs[i % 32] as f64);
        }
        // 블록 단위 미러(전 블록 합)와 레인 미러가 f32 디퀀트 기준과 일치해야 한다
        let mut got_block = 0.0f64;
        for b in 0..n / 32 {
            got_block += dot_q5_1_q8(&bytes[b * 24..b * 24 + 24], &y[b..b + 1]) as f64;
        }
        let got_lane = dot_row_w4a8_q5_1_lane(&bytes, n as u64, &y) as f64;
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-3);
        assert!(rel(got_block, want) < 1e-4, "block {got_block} vs {want}");
        assert!(rel(got_lane, want) < 1e-4, "lane {got_lane} vs {want}");
    }

    /// 각 타입: 임의 블록 바이트 → f32 dequant 내적 vs dot_*_q8 — 상대오차 < 1.5e-2
    /// (q8 활성 양자화 오차가 유일한 차이원).
    #[test]
    fn w4a8_dots_match_f32() {
        let mut seed = 170u64;
        let mut lcg = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        // 임의 x (f32) → q8 양자화 → 재구성 y_f 를 f32 기준으로 삼으면
        // 차이는 오직 (a) 블록별 정수그룹화 (b) q8 양자화 0 — 아니, f32 기준은
        // 원본 x와 y 재구성을 같이 쓴다: w_f32[i]·x[i] vs dot(q8(x)) — q8 오차 포함.
        let cases: Vec<(WType, usize)> = vec![
            (WType::Q4K, 144),
            (WType::Q5K, 176),
            (WType::Q6K, 210),
            (WType::Q3K, 110),
            (WType::Q8_0, 34),
            (WType::Iq4Xs, 136),
            (WType::Iq4Nl, 18),
            (WType::Iq3S, 110),
        ];
        for (ty, bsize) in cases {
            let blck = ty.blck_size() as usize;
            let n = blck * 4; // 4블록
            let mut bytes = vec![0u8; n / blck * bsize];
            for b in bytes.iter_mut() {
                *b = (lcg() & 0xFF) as u8;
            }
            // d 필드가 극단적(0/ff)이면 값이 퇴화 — 스케일 바이트만 온화하게
            for blk in bytes.chunks_mut(bsize) {
                match ty {
                    WType::Q4K | WType::Q5K => {
                        blk[0] = 0x30;
                        blk[1] = 0x10;
                        blk[2] = 0x28;
                        blk[3] = 0x10;
                    }
                    WType::Q6K => {
                        blk[208] = 0x50;
                        blk[209] = 0x11;
                    }
                    WType::Q3K => {
                        blk[108] = 0x40;
                        blk[109] = 0x11;
                    }
                    WType::Q8_0 => {
                        blk[0] = 0x50;
                        blk[1] = 0x11;
                    }
                    WType::Iq4Xs | WType::Iq4Nl => {
                        blk[0] = 0x50;
                        blk[1] = 0x11;
                    }
                    WType::Iq3S => {
                        blk[0] = 0x50;
                        blk[1] = 0x11;
                    }
                    _ => {}
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (lcg() as f32 / 2147483648.0) - 0.5)
                .collect();
            let y = quantize_row_q8_ref(&x);
            let mut wf = vec![0.0f32; n];
            for b in 0..n / blck {
                dequant_row(
                    ty,
                    &bytes[b * bsize..],
                    0,
                    blck as u64,
                    &mut wf[b * blck..(b + 1) * blck],
                );
            }
            let f32_dot: f32 = x.iter().zip(wf.iter()).map(|(a, b)| a * b).sum();
            let w4a8 = dot_row_w4a8(ty, &bytes, n as u64, &y);
            let rel = (f32_dot - w4a8).abs() / f32_dot.abs().max(1.0);
            assert!(
                rel < 5e-2,
                "{ty:?}: f32={f32_dot:.5} w4a8={w4a8:.5} rel={rel:.4}"
            );
        }
    }
}
