//! MM 변형(mm-mirror) + ktab2 팩.

/// 누산, 4파편 순차 결합 p0+p1+p2+p3 (gemm_q5k_mm와 쌍).
use super::deq::f16;
use super::lane::scale_min_k4_local;
use super::q8::y_el;
use super::q8::*;
use crate::tables::KVALUES_IQ4NL;
pub fn dot_row_w4a8_q5k_mm(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    // mm2 쌍: 순차 sb f32 누산 (wvs 레지스터 적재형 커널과 동일 순서)
    let n_sub = (k / 32) as usize;
    let mut acc = 0.0f32;
    for sb in 0..n_sub {
        let js = sb % 8;
        let (it, half) = (js / 2, js % 2);
        let wb = &data[(sb / 8) * 176..(sb / 8) * 176 + 176];
        let d = f16(wb, 0);
        let dm = f16(wb, 2);
        let (sc, m_) = scale_min_k4_local(wb, js);
        let u = if half == 0 {
            1u8 << (2 * it)
        } else {
            2u8 << (2 * it)
        };
        let mut isum = 0i64;
        let mut qsum = 0i64;
        for j in 0..32 {
            let nib = if half == 0 {
                wb[48 + it * 32 + j] & 0xF
            } else {
                wb[48 + it * 32 + j] >> 4
            };
            let hi2 = if wb[16 + j] & u != 0 { 16i64 } else { 0i64 };
            let yv = y_el(y, sb * 32 + j);
            isum += (nib as i64 + hi2) * yv;
            qsum += yv;
        }
        let yd = y[sb].d;
        acc += yd * (d * sc as f32) * isum as f32;
        acc -= yd * (dm * m_ as f32) * qsum as f32;
    }
    acc
}

/// MMQ 포트 미러(q4_K) — 파편 sb%4 오름차순 f32, p0+p1+p2+p3.
pub fn dot_row_w4a8_q4k_mm(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    // mm2 쌍: 순차 sb f32 누산
    let n_sub = (k / 32) as usize;
    let mut acc = 0.0f32;
    for sb in 0..n_sub {
        let js = sb % 8;
        let (it, half) = (js / 2, js % 2);
        let wb = &data[(sb / 8) * 144..(sb / 8) * 144 + 144];
        let d = f16(wb, 0);
        let dm = f16(wb, 2);
        let (sc, m_) = scale_min_k4_local(wb, js);
        let mut isum = 0i64;
        let mut qsum = 0i64;
        for j in 0..32 {
            let nib = if half == 0 {
                wb[16 + it * 32 + j] & 0xF
            } else {
                wb[16 + it * 32 + j] >> 4
            };
            let yv = y_el(y, sb * 32 + j);
            isum += nib as i64 * yv;
            qsum += yv;
        }
        let yd = y[sb].d;
        acc += yd * (d * sc as f32) * isum as f32;
        acc -= yd * (dm * m_ as f32) * qsum as f32;
    }
    acc
}

/// MMQ 포트 미러(q6_K) — 파편 그룹 g%4 (16원소), 단일 체인.
pub fn dot_row_w4a8_q6k_mm(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let n_g = (k / 16) as usize;
    let mut acc = 0.0f32;
    for g in 0..n_g {
        let blk = g / 16;
        let klocal = g % 16;
        let wb = &data[blk * 210..blk * 210 + 210];
        let h = klocal / 8;
        let src = (klocal % 8) / 2;
        let p2 = klocal % 2;
        let d = f16(wb, 208);
        let sc = wb[192 + klocal] as i8;
        let mut isum = 0i64;
        for jj in 0..16 {
            let l2 = p2 * 16 + jj;
            let (nib, hi2) = match src {
                0 => (wb[h * 64 + l2] & 0xF, (wb[128 + h * 32 + l2] & 3) as i64),
                1 => (
                    wb[h * 64 + l2 + 32] & 0xF,
                    ((wb[128 + h * 32 + l2] >> 2) & 3) as i64,
                ),
                2 => (
                    wb[h * 64 + l2] >> 4,
                    ((wb[128 + h * 32 + l2] >> 4) & 3) as i64,
                ),
                _ => (
                    wb[h * 64 + l2 + 32] >> 4,
                    ((wb[128 + h * 32 + l2] >> 6) & 3) as i64,
                ),
            };
            let elem = blk * 256 + h * 128 + src * 32 + p2 * 16 + jj;
            // −32는 qsum 경로로 1회 (커널 wvs = nib|hi2<<4, isum -= 32·qst)
            isum += ((nib as i64) | (hi2 << 4)) * y_el(y, elem);
        }
        let qsum: i64 = (0..16)
            .map(|jj| y_el(y, blk * 256 + h * 128 + src * 32 + p2 * 16 + jj))
            .sum();
        isum -= 32 * qsum;
        let yd = y[blk * 8 + h * 4 + src].d;
        acc += yd * (d * sc as f32) * isum as f32;
    }
    acc
}

/// MMQ 포트 미러(iq4_xs) — kvalues 룩업, 단일 체인.
pub fn dot_row_w4a8_iq4xs_mm(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let n_sub = (k / 32) as usize;
    let mut acc = 0.0f32;
    for sb in 0..n_sub {
        let ib = sb % 8;
        let wb = &data[(sb / 8) * 136..(sb / 8) * 136 + 136];
        let d = f16(wb, 0);
        let scales_h = u16::from_le_bytes([wb[2], wb[3]]);
        let ls = ((wb[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let mut isum = 0i64;
        for j in 0..16 {
            let b = wb[8 + ib * 16 + j];
            let t = ktab2_word(b as usize);
            let lo = (t & 0xFF) as i8 as i64;
            let hi = ((t >> 8) & 0xFF) as i8 as i64;
            isum += lo * y_el(y, sb * 32 + j);
            isum += hi * y_el(y, sb * 32 + 16 + j);
        }
        let yd = y[sb].d;
        acc += yd * dl * isum as f32;
    }
    acc
}

#[inline]
fn ktab2_word(b: usize) -> u32 {
    let lo = KVALUES_IQ4NL[b & 0xF] as u8 as u32;
    let hi = KVALUES_IQ4NL[b >> 4] as u8 as u32;
    lo | (hi << 8)
}
