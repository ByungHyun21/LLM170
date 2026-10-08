//! W4A16(GPTQ4·g128·sym) 레인 미러 — **CPU 비트 계약** (GPU 커널 판정 기준).
//!
//! AutoRound auto_gptq 규약: qrow 워드 j의 니블 j%8 = 원소 8·(j/8)+(j%8).
//! lsb-first **확정**(2026-10-08, w4a16-xcheck — 동일 기저 27B 원본 대조
//! corr(lsb) 0.991~0.994 vs corr(msb) ≈0.01, §3.6 종결).
//! zrow = 로더가 행별 언팩한 zp(그룹당 1개, sym 전형 8), srow = 그룹 f16 스케일.
//! GPU gemm_gptq4(64레인)와 동일 연산열: 레인 l = 원소 l, l+64, … f32 누산 →
//! f64 레인 배열 → tree64.

/// 64레인 환원 — warp 트리 순서 (GPU __shfl_down 16,8,4,2,1 ×2 + 상위 가산).
/// 비트계약: 커널 환원과 동일 순서.
pub fn tree64(v: &[f64; 64]) -> f64 {
    let mut a = *v;
    // 상/하 절반 쌍 가산 후 32레인 트리 (GPU 커널 환원과 동일 순서)
    for i in 0..32 {
        a[i] += a[i + 32];
    }
    for &off in &[16usize, 8, 4, 2, 1] {
        for i in 0..off {
            a[i] += a[i + off];
        }
    }
    a[0]
}

/// f16 비트(u16) → f32 (IEEE half, 정확 변환 — deq::half_to_f32 공유).
#[inline]
fn f16b(v: u16) -> f32 {
    super::deq::half_to_f32(v)
}

/// W4A16 행 내적 미러 — 활성 x는 f16 비트(u16) 슬라이스(res_hc f16 버스 계약).
/// zrow는 로더가 행별 언팩한 zp(그룹당 1워드, 값 0..15).
pub fn dot_row_w4a16_lane(qrow: &[u32], zrow: &[u32], srow: &[u16], x: &[u16]) -> f32 {
    tree64(&dot_row_w4a16_lane_parts(qrow, zrow, srow, x)) as f32
}

pub fn dot_row_w4a16_lane_parts(qrow: &[u32], zrow: &[u32], srow: &[u16], x: &[u16]) -> [f64; 64] {
    let k = x.len();
    assert!(k.is_multiple_of(128), "w4a16: k는 g128 그룹정렬 필요");
    assert_eq!(qrow.len(), k / 8, "w4a16: qrow 길이");
    assert_eq!(srow.len(), k / 128, "w4a16: srow 길이");
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let mut acc = 0.0f32;
        let mut i = l;
        while i < k {
            let g = i >> 7;
            let q = ((qrow[i >> 3] >> (4 * (i & 7))) & 0xF) as i32;
            let w = (q - zrow[g] as i32) as f32 * f16b(srow[g]);
            acc += w * f16b(x[i]);
            i += 64;
        }
        lane[l] = acc as f64;
    }
    lane
}

#[cfg(test)]
mod w4a16_tests {
    use super::*;

    /// f32 → f16 비트 (정규수 round-to-nearest — 테스트 전용).
    fn to_f16(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        let man = b & 0x7F_FFFF;
        if exp <= 0 {
            return sign;
        }
        if exp >= 31 {
            return sign | 0x7C00;
        }
        let half_man = man >> 13;
        let round = (man >> 12) & 1;
        let m = half_man + round;
        let (m, e) = if m & 0x400 != 0 {
            (m & 0x3FF, exp + 1)
        } else {
            (m, exp)
        };
        sign | ((e as u16) << 10) | (m as u16)
    }

    #[test]
    fn w4a16_lane_matches_scalar() {
        let k = 256usize; // g128 × 2그룹
        let mut seed = 0x9e37_79b9u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut qrow = vec![0u32; k / 8];
        for v in qrow.iter_mut() {
            *v = rnd() as u32;
        }
        // zp 그룹별 상이(8 고정 아님) — 감산 경로 포함 검증
        let mut zrow = vec![0u32; k / 128];
        for v in zrow.iter_mut() {
            *v = (rnd() % 16) as u32;
        }
        let mut srow = vec![0u16; k / 128];
        for v in srow.iter_mut() {
            *v = to_f16(0.25f32 + (rnd() % 100) as f32 / 400.0);
        }
        let mut x = vec![0u16; k];
        for v in x.iter_mut() {
            *v = to_f16((rnd() % 2000) as f32 / 1000.0 - 1.0);
        }

        let got = dot_row_w4a16_lane(&qrow, &zrow, &srow, &x);
        let mut want = 0.0f64;
        for i in 0..k {
            let g = i / 128;
            let q = ((qrow[i / 8] >> (4 * (i % 8))) & 0xF) as i32;
            let w = (q - zrow[g] as i32) as f32 * f16b(srow[g]);
            want += (w * f16b(x[i])) as f64;
        }
        let rel = (got as f64 - want).abs() / want.abs().max(1e-9);
        assert!(
            rel < 1e-6,
            "w4a16 미러 불일치: got={got} want={want} rel={rel}"
        );
    }

    #[test]
    fn f16b_is_ieee_half() {
        // 회귀 가드: bf16식 비트 시프트로 퇴화하지 않는다(2026-10-08 정정 —
        // 스케일은 실측 f16, 예: 0x3800=0.5).
        assert_eq!(f16b(0x3C00), 1.0);
        assert_eq!(f16b(0x3800), 0.5);
        assert_eq!(f16b(0xC000), -2.0);
    }

    #[test]
    fn w4a16_sym_zp8_center() {
        // zp=8·q=8 → 전 기여 0 — 니블 추출 자체 검증
        let k = 128usize;
        let qrow = vec![0x8888_8888u32; k / 8];
        let zrow = vec![8u32; 1];
        let srow = vec![to_f16(0.5); 1];
        let x = vec![to_f16(1.0); k];
        assert_eq!(dot_row_w4a16_lane(&qrow, &zrow, &srow, &x), 0.0);
    }
}
