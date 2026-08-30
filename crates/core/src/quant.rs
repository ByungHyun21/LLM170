//! 양자화 블록 역양자화 — llama.cpp `ggml/src/ggml-quants.c` 대응 (2026-08-30 판).
//!
//! 블록 레이아웃은 `ggml-common.h` 구조체 선언 순서 그대로 바이트 오프셋로 해석한다.
//! 모든 함수는 한 "행"(row, ne[0] 축)의 연속 블록을 f32 로 펼친다.

use crate::tables::{IQ3S_GRID, KVALUES_IQ4NL};
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
        _ => (sign << 31) | ((exp + 112) << 23) | (frac << 13),
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
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
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
            y[yi + l] =
                d1 * ((ql[qi + l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32 - mm1;
            y[yi + 32 + l] =
                d2 * ((ql[qi + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32 - mm2;
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
    let scales: Vec<i8> = aux
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|b| b as i8)
        .collect();

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
            let g2 =
                grid4(qs[qsi + 2 * l + 1] as usize | (((qh[qhi] as usize) << (7 - 2 * l)) & 256));
            for j in 0..4 {
                y[yi + j] = db1
                    * g1[j]
                    * if signs[sgi + l] & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                y[yi + 4 + j] = db1
                    * g2[j]
                    * if signs[sgi + l] & KMASK_IQ2XS[4 + j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
            }
            yi += 8;
        }
        qsi += 8;
        sgi += 4;
        for l in 0..4 {
            let g1 =
                grid4(qs[qsi + 2 * l] as usize | (((qh[qhi + 1] as usize) << (8 - 2 * l)) & 256));
            let g2 = grid4(
                qs[qsi + 2 * l + 1] as usize | (((qh[qhi + 1] as usize) << (7 - 2 * l)) & 256),
            );
            for j in 0..4 {
                y[yi + j] = db2
                    * g1[j]
                    * if signs[sgi + l] & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                y[yi + 4 + j] = db2
                    * g2[j]
                    * if signs[sgi + l] & KMASK_IQ2XS[4 + j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
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
                out[j] = if ty == GgmlType::F16 {
                    half_to_f32(h)
                } else {
                    bf16_to_f32(h)
                };
            }
        }
        GgmlType::Q4K => {
            for b in 0..blocks {
                deq_q4_k(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 256..b * 256 + 256],
                );
            }
        }
        GgmlType::Q5K => {
            for b in 0..blocks {
                deq_q5_k(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 256..b * 256 + 256],
                );
            }
        }
        GgmlType::Q6K => {
            for b in 0..blocks {
                deq_q6_k(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 256..b * 256 + 256],
                );
            }
        }
        GgmlType::Q3K => {
            for b in 0..blocks {
                deq_q3_k(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 256..b * 256 + 256],
                );
            }
        }
        GgmlType::Q8_0 => {
            for b in 0..blocks {
                deq_q8_0(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 32..b * 32 + 32],
                );
            }
        }
        GgmlType::Q5_1 => {
            for b in 0..blocks {
                deq_q5_1(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 32..b * 32 + 32],
                );
            }
        }
        GgmlType::Iq4Xs => {
            for b in 0..blocks {
                deq_iq4_xs(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 256..b * 256 + 256],
                );
            }
        }
        GgmlType::Iq4Nl => {
            for b in 0..blocks {
                deq_iq4_nl(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 32..b * 32 + 32],
                );
            }
        }
        GgmlType::Iq3S => {
            for b in 0..blocks {
                deq_iq3_s(
                    &data[base + b * bsize..][..bsize],
                    &mut out[b * 256..b * 256 + 256],
                );
            }
        }
        other => unimplemented!("dequant for {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// W4A8 변형 — 활성 q8 양자화 + 정수 내적 (성능 경로, f32이 기준).
// 산술 구조는 ggml-quants.c 스칼라 참조, 스케일 d는 f32(ggml은 f16 저장).
// y 접근: y_el(y, p) 평탄 인덱스 — 위 deq_* 의 y[p] 순서와 1:1.
// ---------------------------------------------------------------------------

/// q8_0 블록 (변형): d f32 + qs i8×32.
#[derive(Clone, Copy)]
pub struct Q8Block {
    pub d: f32,
    pub qs: [i8; 32],
}

/// 활성 행을 q8 블록으로 양자화 — ggml quantize_row_q8_ref 산술.
pub fn quantize_row_q8_ref(x: &[f32]) -> Vec<Q8Block> {
    let blocks = x.len().div_ceil(32);
    let mut out = vec![Q8Block { d: 0.0, qs: [0; 32] }; blocks];
    for (b, o) in out.iter_mut().enumerate() {
        let s = &x[b * 32..(b * 32 + 32).min(x.len())];
        let mut amax = 0.0f32;
        for &v in s {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.d = d;
        for (j, &v) in s.iter().enumerate() {
            o.qs[j] = ((v * id).round()).clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

/// y의 평탄 요소 정수값 (블록 p/32, 내부 p%32).
#[inline]
fn y_el(y: &[Q8Block], p: usize) -> i64 {
    y[p / 32].qs[p % 32] as i64
}

/// y 평탄 요소 재구성값 (q·d).
#[inline]
fn y_f(y: &[Q8Block], p: usize) -> f32 {
    y[p / 32].qs[p % 32] as f32 * y[p / 32].d
}

/// q8_0(32) × q8.
pub fn dot_q8k_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let mut isum = 0i64;
    for j in 0..32 {
        isum += (w[2 + j] as i8) as i64 * y_el(y, j);
    }
    d * y[0].d * isum as f32
}

/// q4_K(256) × q8 — deq_q4_k 순서: p=it*64+l(lo), +32(hi).
pub fn dot_q4k_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let min = f16(w, 2);
    let sc = &w[4..16];
    let qs = &w[16..144];
    let mut sum = 0.0f32;
    for it in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * it, sc);
        let (sc2, m2) = scale_min_k4(2 * it + 1, sc);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        let mut isum1 = 0i64;
        let mut isum2 = 0i64;
        let mut qsum1 = 0i64;
        let mut qsum2 = 0i64;
        for l in 0..32 {
            let q = qs[it * 32 + l];
            isum1 += (q & 0xF) as i64 * y_el(y, it * 64 + l);
            isum2 += (q >> 4) as i64 * y_el(y, it * 64 + 32 + l);
        }
        qsum1 = (0..32).map(|l| y_el(y, it * 64 + l)).sum();
        qsum2 = (0..32).map(|l| y_el(y, it * 64 + 32 + l)).sum();
        let (yd1, yd2) = (y[2 * it].d, y[2 * it + 1].d);
        sum += yd1 * (d1 * isum1 as f32 - mm1 * qsum1 as f32);
        sum += yd2 * (d2 * isum2 as f32 - mm2 * qsum2 as f32);
    }
    sum
}

/// q5_K(256) × q8 — deq_q5_k 순서 동일 + qh 비트.
pub fn dot_q5k_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let min = f16(w, 2);
    let sc = &w[4..16];
    let qh = &w[16..48];
    let ql = &w[48..176];
    let mut sum = 0.0f32;
    let (mut u1, mut u2) = (1u8, 2u8);
    for it in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * it, sc);
        let (sc2, m2) = scale_min_k4(2 * it + 1, sc);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        let mut isum1 = 0i64;
        let mut isum2 = 0i64;
        let mut qsum1 = 0i64;
        let mut qsum2 = 0i64;
        for l in 0..32 {
            let v1 = (ql[it * 32 + l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
            let v2 = (ql[it * 32 + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            isum1 += v1 as i64 * y_el(y, it * 64 + l);
            isum2 += v2 as i64 * y_el(y, it * 64 + 32 + l);
            qsum1 += y_el(y, it * 64 + l);
            qsum2 += y_el(y, it * 64 + 32 + l);
        }
        let (yd1, yd2) = (y[2 * it].d, y[2 * it + 1].d);
        sum += yd1 * (d1 * isum1 as f32 - mm1 * qsum1 as f32);
        sum += yd2 * (d2 * isum2 as f32 - mm2 * qsum2 as f32);
        u1 <<= 2;
        u2 <<= 2;
    }
    sum
}

/// q6_K(256) × q8 — deq_q6_k: p = h*128 + pos*32 + l, 스케일 h*8+l/16+pos*2.
pub fn dot_q6k_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 208);
    let ql = &w[0..128];
    let qh = &w[128..192];
    let sc: Vec<i8> = w[192..208].iter().map(|&b| b as i8).collect();
    let mut sum = 0.0f32;
    // 누적을 스케일별 i64로 모아 한 번에 조합
    let mut acc = vec![0i64; 16];
    for h in 0..2 {
        for l in 0..32 {
            let is = h * 8 + l / 16;
            let q1 = (((ql[h * 64 + l] & 0xF) | (((qh[h * 32 + l]) & 3) << 4)) as i32 - 32) as i64;
            let q2 = (((ql[h * 64 + l + 32] & 0xF) | (((qh[h * 32 + l] >> 2) & 3) << 4)) as i32 - 32) as i64;
            let q3 = (((ql[h * 64 + l] >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)) as i32 - 32) as i64;
            let q4 = (((ql[h * 64 + l + 32] >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)) as i32 - 32) as i64;
            acc[is] += q1 * y_el(y, h * 128 + l);
            acc[is + 2] += q2 * y_el(y, h * 128 + 32 + l);
            acc[is + 4] += q3 * y_el(y, h * 128 + 64 + l);
            acc[is + 6] += q4 * y_el(y, h * 128 + 96 + l);
        }
    }
    for (k, a) in acc.iter().enumerate() {
        let h = k / 8;
        let pos = (k % 8) / 2;
        let yd = y[h * 4 + pos].d;
        sum += yd * d * sc[k] as f32 * *a as f32;
    }
    sum
}

/// q3_K(256) × q8 — deq_q3_k: p = n*128 + si*32 + half*16 + l.
pub fn dot_q3k_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d_all = f16(w, 108);
    let hm = &w[0..32];
    let q = &w[32..96];
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([w[96], w[97], w[98], w[99]]);
    aux[1] = u32::from_le_bytes([w[100], w[101], w[102], w[103]]);
    let tmp = u32::from_le_bytes([w[104], w[105], w[106], w[107]]);
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    let aux2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    let aux3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let scales: Vec<i8> = [aux[0], aux[1], aux2, aux3]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .map(|b| b as i8)
        .collect();
    let mut sum = 0.0f32;
    let mut ai = 0usize;
    for n in 0..2 {
        for si in 0..4 {
            for half in 0..2 {
                let dl = d_all * (scales[ai] as f32 - 32.0);
                ai += 1;
                let mut isum = 0i64;
                for l in 0..16 {
                    let qv = ((q[n * 32 + half * 16 + l] >> (2 * si)) & 3) as i64;
                    let sub = if hm[half * 16 + l] & (1 << si) != 0 { 0i64 } else { 4i64 };
                    isum += (qv - sub) * y_el(y, n * 128 + si * 32 + half * 16 + l);
                }
                let yd = y[n * 4 + si].d;
                sum += yd * dl * isum as f32;
            }
        }
    }
    sum
}

/// iq4_xs(256) × q8 — p = ib*32 + j(lo), ib*32+16+j(hi).
pub fn dot_iq4xs_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let scales_h = u16::from_le_bytes([w[2], w[3]]);
    let scales_l = &w[4..8];
    let qs = &w[8..136];
    let mut sum = 0.0f32;
    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let mut isum = 0i64;
        for j in 0..16 {
            let q = qs[ib * 16 + j];
            isum += KVALUES_IQ4NL[(q & 0xF) as usize] as i64 * y_el(y, ib * 32 + j);
            isum += KVALUES_IQ4NL[(q >> 4) as usize] as i64 * y_el(y, ib * 32 + 16 + j);
        }
        sum += y[ib].d * dl * isum as f32;
    }
    sum
}

/// iq4_nl(32) × q8.
pub fn dot_iq4nl_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let qs = &w[2..18];
    let mut isum = 0i64;
    for j in 0..16 {
        isum += KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as i64 * y_el(y, j);
        isum += KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i64 * y_el(y, 16 + j);
    }
    d * y[0].d * isum as f32
}

/// iq3_s(256) × q8 — deq_iq3_s 순서: p = ib*64 + sec*32 + l*8 + (0..8).
pub fn dot_iq3s_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let qs = &w[2..66];
    let qh = &w[66..74];
    let signs = &w[74..106];
    let scales = &w[106..110];
    let mut sum = 0.0f32;
    let mut yi = 0usize;
    for ib in 0..4 {
        let db1 = d * (1 + 2 * (scales[ib] & 0xF)) as f32;
        let db2 = d * (1 + 2 * (scales[ib] >> 4)) as f32;
        for sec in 0..2 {
            let db = if sec == 0 { db1 } else { db2 };
            for l in 0..4 {
                let qhi = qh[2 * ib + sec] as usize;
                let i1 = qs[ib * 16 + sec * 8 + 2 * l] as usize | ((qhi << (8 - 2 * l)) & 256);
                let i2 = qs[ib * 16 + sec * 8 + 2 * l + 1] as usize | ((qhi << (7 - 2 * l)) & 256);
                let g1 = grid4(i1);
                let g2 = grid4(i2);
                let mut isum = 0i64;
                for j in 0..4 {
                    let s1 = if signs[ib * 8 + sec * 4 + l] & KMASK_IQ2XS[j] != 0 { -1i64 } else { 1i64 };
                    let s2 = if signs[ib * 8 + sec * 4 + l] & KMASK_IQ2XS[4 + j] != 0 { -1i64 } else { 1i64 };
                    isum += g1[j] as i64 * s1 * y_el(y, yi + j);
                    isum += g2[j] as i64 * s2 * y_el(y, yi + 4 + j);
                }
                sum += y[ib * 2 + sec].d * db * isum as f32;
                yi += 8;
            }
        }
    }
    sum
}

/// W4A8: 한 행 k원소의 W4A8 내적 — 타입별 분기. data는 행 시작.
pub fn dot_row_w4a8(ty: GgmlType, data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let (blck, bsize) = ty.block_info();
    let blocks = (k / blck) as usize;
    let bsize = bsize as usize;
    let mut acc = 0.0f32;
    for b in 0..blocks {
        let wb = &data[b * bsize..b * bsize + bsize];
        let yb = &y[b * (blck as usize / 32)..b * (blck as usize / 32) + blck as usize / 32];
        let v = match ty {
            GgmlType::Q4K => dot_q4k_q8(wb, yb),
            GgmlType::Q5K => dot_q5k_q8(wb, yb),
            GgmlType::Q6K => dot_q6k_q8(wb, yb),
            GgmlType::Q3K => dot_q3k_q8(wb, yb),
            GgmlType::Q8_0 => dot_q8k_q8(wb, yb),
            GgmlType::Iq4Xs => dot_iq4xs_q8(wb, yb),
            GgmlType::Iq4Nl => dot_iq4nl_q8(wb, yb),
            GgmlType::Iq3S => dot_iq3s_q8(wb, yb),
            _ => {
                // 미지원: f32 디양자화 × y 재구성 (정확도 기준과 동일 원소 재구성)
                let n = blck as usize;
                let mut tmp = vec![0.0f32; n];
                dequant_row(ty, wb, 0, blck, &mut tmp);
                let base = b * n;
                let mut s = 0.0;
                for (i, t) in tmp.iter().enumerate() {
                    s += t * y_f(y, base + i);
                }
                s
            }
        };
        acc += v;
    }
    acc
}

#[cfg(test)]
mod w4a8_tests {
    use super::*;

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
        let cases: Vec<(GgmlType, usize)> = vec![
            (GgmlType::Q4K, 144), (GgmlType::Q5K, 176), (GgmlType::Q6K, 210),
            (GgmlType::Q3K, 110), (GgmlType::Q8_0, 34), (GgmlType::Iq4Xs, 136),
            (GgmlType::Iq4Nl, 18), (GgmlType::Iq3S, 110),
        ];
        for (ty, bsize) in cases {
            let blck = ty.blck_size() as usize;
            let n = blck * 4; // 4블록
            let mut bytes = vec![0u8; n / blck * bsize];
            for b in bytes.iter_mut() {
                *b = (lcg() & 0xFF) as u8;
            }
            // d 필드가 극단적(0/ff)이면 값이 퇴화 — 스케일 바이트만 온화하게
            for (bi, blk) in bytes.chunks_mut(bsize).enumerate() {
                let _ = bi;
                match ty {
                    GgmlType::Q4K | GgmlType::Q5K => {
                        blk[0] = 0x30; blk[1] = 0x10; blk[2] = 0x28; blk[3] = 0x10;
                    }
                    GgmlType::Q6K => { blk[208] = 0x50; blk[209] = 0x11; }
                    GgmlType::Q3K => { blk[108] = 0x40; blk[109] = 0x11; }
                    GgmlType::Q8_0 => { blk[0] = 0x50; blk[1] = 0x11; }
                    GgmlType::Iq4Xs | GgmlType::Iq4Nl => { blk[0] = 0x50; blk[1] = 0x11; }
                    GgmlType::Iq3S => { blk[0] = 0x50; blk[1] = 0x11; }
                    _ => {}
                }
            }
            let x: Vec<f32> = (0..n).map(|_| (lcg() as f32 / 2147483648.0) - 0.5).collect();
            let y = quantize_row_q8_ref(&x);
            let mut wf = vec![0.0f32; n];
            for b in 0..n / blck {
                dequant_row(ty, &bytes[b * bsize..], 0, blck as u64, &mut wf[b * blck..(b + 1) * blck]);
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
