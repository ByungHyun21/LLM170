//! 양자화 블록 역양자화 — llama.cpp `ggml/src/ggml-quants.c` 대응 (2026-08-30 판).
//!
//! 블록 레이아웃은 `ggml-common.h` 구조체 선언 순서 그대로 바이트 오프셋로 해석한다.
//! 모든 함수는 한 "행"(row, ne[0] 축)의 연속 블록을 f32 로 펼친다.

use crate::tables::{KVALUES_IQ4NL, IQ3S_GRID};
use llm170_gguf::GgmlType;

#[inline]
fn f16(b: &[u8], off: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// IEEE 754 binary16 → f32
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // subnormal
                let mut e = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                (sign << 31) | (e << 23) | (f << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// get_scale_min_k4 (ggml-quants.c:880)
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// q4_K 블록: d(2) dmin(2) scales(12) qs(128)
fn deq_q4_k(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let min = f16(blk, 2);
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    let mut is = 0;
    let mut qi = 0;
    let mut yi = 0;
    for _ in 0..4 {
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        for l in 0..32 {
            y[yi + l] = d1 * (qs[qi + l] & 0xF) as f32 - mm1;
            y[yi + 32 + l] = d2 * (qs[qi + l] >> 4) as f32 - mm2;
        }
        qi += 32;
        yi += 64;
        is += 2;
    }
}

/// q5_K 블록: d(2) dmin(2) scales(12) qh(32) qs(128)
fn deq_q5_k(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let min = f16(blk, 2);
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let ql = &blk[48..176];
    let mut is = 0;
    let mut qi = 0;
    let mut yi = 0;
    let (mut u1, mut u2) = (1u8, 2u8);
    for _ in 0..4 {
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        for l in 0..32 {
            y[yi + l] = d1 * ((ql[qi + l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32 - mm1;
            y[yi + 32 + l] = d2 * ((ql[qi + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32 - mm2;
        }
        qi += 32;
        yi += 64;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// q6_K 블록: ql(128) qh(64) scales(16, i8) d(2)
fn deq_q6_k(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 208);
    let ql = &blk[0..128];
    let qh = &blk[128..192];
    let sc: Vec<i8> = blk[192..208].iter().map(|&b| b as i8).collect();
    let mut yi = 0;
    let mut qli = 0;
    let mut qhi = 0;
    let mut sci = 0;
    for _ in 0..2 {
        for l in 0..32 {
            let is = sci + l / 16;
            let q1 = (((ql[qli + l] & 0xF) | (((qh[qhi + l]) & 3) << 4)) as i32) - 32;
            let q2 = (((ql[qli + l + 32] & 0xF) | (((qh[qhi + l] >> 2) & 3) << 4)) as i32) - 32;
            let q3 = (((ql[qli + l] >> 4) | (((qh[qhi + l] >> 4) & 3) << 4)) as i32) - 32;
            let q4 = (((ql[qli + l + 32] >> 4) | (((qh[qhi + l] >> 6) & 3) << 4)) as i32) - 32;
            y[yi + l] = d * sc[is] as f32 * q1 as f32;
            y[yi + 32 + l] = d * sc[is + 2] as f32 * q2 as f32;
            y[yi + 64 + l] = d * sc[is + 4] as f32 * q3 as f32;
            y[yi + 96 + l] = d * sc[is + 6] as f32 * q4 as f32;
        }
        yi += 128;
        qli += 64;
        qhi += 32;
        sci += 8;
    }
}

/// q3_K 블록: hmask(32) qs(64) scales(12) d(2)
fn deq_q3_k(blk: &[u8], y: &mut [f32]) {
    let d_all = f16(blk, 108);
    let hm = &blk[0..32];
    let q = &blk[32..96];
    // scales 12바이트 → 16개 i8 (ggml-quants.c:1323-1328)
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([blk[96], blk[97], blk[98], blk[99]]);
    aux[1] = u32::from_le_bytes([blk[100], blk[101], blk[102], blk[103]]);
    let tmp = u32::from_le_bytes([blk[104], blk[105], blk[106], blk[107]]);
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let scales: Vec<i8> = aux.iter().flat_map(|w| w.to_le_bytes()).map(|b| b as i8).collect();

    // n(2회) × shift(4) × half(2) × 16 값; q는 n마다 32바이트, hm은 비트팩(값 index = n*128 + j*32 + half*16 + l → hm 바이트 (qi+half+l), 비트 m=1<<j)
    let mut yi = 0;
    let mut qi = 0;
    let mut is = 0;
    for _ in 0..2 {
        let mut m = 1u8;
        for shift in [0u8, 2, 4, 6] {
            for half in [0usize, 16] {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = ((q[qi + half + l] >> shift) & 3) as i32;
                    let sub = if hm[half + l] & m != 0 { 0 } else { 4 };
                    y[yi + l] = dl * (qv - sub) as f32;
                }
                yi += 16;
            }
            m <<= 1;
        }
        qi += 32;
    }
}

/// q8_0 블록: d(2) qs(32, i8)
fn deq_q8_0(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    for j in 0..32 {
        y[j] = blk[2 + j] as i8 as f32 * d;
    }
}

/// q5_1 블록: d(2) m(2) qh(4) qs(16)
fn deq_q5_1(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let m = f16(blk, 2);
    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
    let qs = &blk[8..24];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as i32;
        let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
        let x0 = (qs[j] & 0x0F) as i32 | xh0;
        let x1 = (qs[j] >> 4) as i32 | xh1;
        y[j] = x0 as f32 * d + m;
        y[16 + j] = x1 as f32 * d + m;
    }
}

/// iq4_xs 블록: d(2) scales_h(2) scales_l(4) qs(128)
fn deq_iq4_xs(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
    let scales_l = &blk[4..8];
    let qs = &blk[8..136];
    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let q = &qs[ib * 16..ib * 16 + 16];
        for j in 0..16 {
            y[ib * 32 + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            y[ib * 32 + 16 + j] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}

/// iq4_nl 블록: d(2) qs(16)
fn deq_iq4_nl(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let qs = &blk[2..18];
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        y[16 + j] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

#[inline]
fn grid4(idx: usize) -> [f32; 4] {
    let w = IQ3S_GRID[idx];
    [
        (w & 0xFF) as i8 as f32,
        ((w >> 8) & 0xFF) as i8 as f32,
        ((w >> 16) & 0xFF) as i8 as f32,
        ((w >> 24) & 0xFF) as i8 as f32,
    ]
}

/// iq3_s 블록: d(2) qs(64) qh(8) signs(32) scales(4) — d가 맨 앞 (ggml-common.h 구조체)
fn deq_iq3_s(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let qs = &blk[2..66];
    let qh = &blk[66..74];
    let signs = &blk[74..106];
    let scales = &blk[106..110];
    let mut yi = 0;
    let mut qsi = 0;
    let mut qhi = 0;
    let mut sgi = 0;
    for ib32_pair in 0..4 {
        let db1 = d * (1 + 2 * (scales[ib32_pair] & 0xF)) as f32;
        let db2 = d * (1 + 2 * (scales[ib32_pair] >> 4)) as f32;
        for l in 0..4 {
            let g1 = grid4(qs[qsi + 2 * l] as usize | (((qh[qhi] as usize) << (8 - 2 * l)) & 256));
            let g2 = grid4(qs[qsi + 2 * l + 1] as usize | (((qh[qhi] as usize) << (7 - 2 * l)) & 256));
            for j in 0..4 {
                y[yi + j] = db1 * g1[j] * if signs[sgi + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[yi + 4 + j] = db1 * g2[j] * if signs[sgi + l] & KMASK_IQ2XS[4 + j] != 0 { -1.0 } else { 1.0 };
            }
            yi += 8;
        }
        qsi += 8;
        sgi += 4;
        for l in 0..4 {
            let g1 = grid4(qs[qsi + 2 * l] as usize | (((qh[qhi + 1] as usize) << (8 - 2 * l)) & 256));
            let g2 = grid4(qs[qsi + 2 * l + 1] as usize | (((qh[qhi + 1] as usize) << (7 - 2 * l)) & 256));
            for j in 0..4 {
                y[yi + j] = db2 * g1[j] * if signs[sgi + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[yi + 4 + j] = db2 * g2[j] * if signs[sgi + l] & KMASK_IQ2XS[4 + j] != 0 { -1.0 } else { 1.0 };
            }
            yi += 8;
        }
        qsi += 8;
        sgi += 4;
        qhi += 2;
    }
}

/// 한 행(k 원소, k 는 블록 크기의 배수)을 f32 로 펼친다.
/// `data` 는 해당 텐서의 데이터 시작 바이트.
pub fn dequant_row(ty: GgmlType, data: &[u8], row: u64, k: u64, out: &mut [f32]) {
    let (blck, bsize) = ty.block_info();
    let blocks = (k / blck) as usize;
    let bsize = bsize as usize;
    debug_assert_eq!(out.len(), k as usize);
    let base = row as usize * blocks * bsize;
    match ty {
        GgmlType::F32 => {
            for j in 0..k as usize {
                let o = base + j * 4;
                out[j] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            }
        }
        GgmlType::F16 | GgmlType::Bf16 => {
            for j in 0..k as usize {
                let h = u16::from_le_bytes([data[base + j * 2], data[base + j * 2 + 1]]);
                out[j] = if ty == GgmlType::F16 { half_to_f32(h) } else { bf16_to_f32(h) };
            }
        }
        GgmlType::Q4K => {
            for b in 0..blocks {
                deq_q4_k(&data[base + b * bsize..][..bsize], &mut out[b * 256..b * 256 + 256]);
            }
        }
        GgmlType::Q5K => {
            for b in 0..blocks {
                deq_q5_k(&data[base + b * bsize..][..bsize], &mut out[b * 256..b * 256 + 256]);
            }
        }
        GgmlType::Q6K => {
            for b in 0..blocks {
                deq_q6_k(&data[base + b * bsize..][..bsize], &mut out[b * 256..b * 256 + 256]);
            }
        }
        GgmlType::Q3K => {
            for b in 0..blocks {
                deq_q3_k(&data[base + b * bsize..][..bsize], &mut out[b * 256..b * 256 + 256]);
            }
        }
        GgmlType::Q8_0 => {
            for b in 0..blocks {
                deq_q8_0(&data[base + b * bsize..][..bsize], &mut out[b * 32..b * 32 + 32]);
            }
        }
        GgmlType::Q5_1 => {
            for b in 0..blocks {
                deq_q5_1(&data[base + b * bsize..][..bsize], &mut out[b * 32..b * 32 + 32]);
            }
        }
        GgmlType::Iq4Xs => {
            for b in 0..blocks {
                deq_iq4_xs(&data[base + b * bsize..][..bsize], &mut out[b * 256..b * 256 + 256]);
            }
        }
        GgmlType::Iq4Nl => {
            for b in 0..blocks {
                deq_iq4_nl(&data[base + b * bsize..][..bsize], &mut out[b * 32..b * 32 + 32]);
            }
        }
        GgmlType::Iq3S => {
            for b in 0..blocks {
                deq_iq3_s(&data[base + b * bsize..][..bsize], &mut out[b * 256..b * 256 + 256]);
            }
        }
        other => unimplemented!("dequant for {other:?}"),
    }
}
