//! Q8 활성 양자화 참조(Q8Block)와 스칼라 dot 패밀리.
use super::deq::*;
use super::deq::{KMASK_IQ2XS, grid4, scale_min_k4};
use super::lane::tree64;
use crate::tables::KVALUES_IQ4NL;
/// q8_0 블록 (변형): d f32 + qs i8×32.
#[derive(Clone, Copy)]
pub struct Q8Block {
    pub d: f32,
    pub qs: [i8; 32],
}

/// 활성 행을 q8 블록으로 양자화 — ggml quantize_row_q8_ref 산술.
pub fn quantize_row_q8_ref(x: &[f32]) -> Vec<Q8Block> {
    let blocks = x.len().div_ceil(32);
    let mut out = vec![
        Q8Block {
            d: 0.0,
            qs: [0; 32]
        };
        blocks
    ];
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
pub(crate) fn y_el(y: &[Q8Block], p: usize) -> i64 {
    y[p / 32].qs[p % 32] as i64
}

/// y 평탄 요소 재구성값 (q·d).
#[inline]
pub(crate) fn y_f(y: &[Q8Block], p: usize) -> f32 {
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

/// q5_1(32) × q8 — value = d·q + m (q 무부호 5bit; lo4 + 16·hi1 분해).
pub fn dot_q5_1_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let m = f16(w, 2);
    let qh = u32::from_le_bytes([w[4], w[5], w[6], w[7]]);
    let mut s_lo = 0i64;
    let mut s_hi = 0i64;
    let mut s1 = 0i64;
    for j in 0..32usize {
        let byte = w[8 + (j & 15)] as u32;
        let lo4 = if j < 16 {
            byte & 0xF
        } else {
            (byte >> 4) & 0xF
        };
        let hi1 = (qh >> j) & 1;
        let yv = y_el(y, j);
        s_lo += yv * lo4 as i64;
        s_hi += yv * hi1 as i64;
        s1 += yv;
    }
    let isum = s_lo + (s_hi << 4);
    y[0].d * (d * isum as f32 + m * s1 as f32)
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

        for l in 0..32 {
            let q = qs[it * 32 + l];
            isum1 += (q & 0xF) as i64 * y_el(y, it * 64 + l);
            isum2 += (q >> 4) as i64 * y_el(y, it * 64 + 32 + l);
        }
        let qsum1: i64 = (0..32).map(|l| y_el(y, it * 64 + l)).sum();
        let qsum2: i64 = (0..32).map(|l| y_el(y, it * 64 + 32 + l)).sum();
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
    let mut acc = [0i64; 16];
    for h in 0..2 {
        for l in 0..32 {
            let is = h * 8 + l / 16;
            let q1 = (((ql[h * 64 + l] & 0xF) | (((qh[h * 32 + l]) & 3) << 4)) as i32 - 32) as i64;
            let q2 = (((ql[h * 64 + l + 32] & 0xF) | (((qh[h * 32 + l] >> 2) & 3) << 4)) as i32
                - 32) as i64;
            let q3 =
                (((ql[h * 64 + l] >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)) as i32 - 32) as i64;
            let q4 = (((ql[h * 64 + l + 32] >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)) as i32 - 32)
                as i64;
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
                    let sub = if hm[half * 16 + l] & (1 << si) != 0 {
                        0i64
                    } else {
                        4i64
                    };
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

/// W4A8 레인 미러(iq4_xs) — GPU gemm_q8i와 동일 64레인 연속 분할·f64
/// 부분합으로 행 전체를 비트 일치 재현. 그룹핑: n_sub=⌈k/32⌉개 서브블록
/// base/rem 연속 분할, 레인 내 오름차순 f64 누산, 레인 순서 f64 합 후
pub fn dot_row_w4a8_iq4xs_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_iq4xs_lane_parts(data, k, y);
    tree64(&lane) as f32
}

/// 레인별 f64 부분합 (디버그·GPU 대조용) — gemm_q8i 그룹핑 미러.
pub fn dot_row_w4a8_iq4xs_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64; // 스트라이드 매핑 — gemm_q8i 미러
        let start = l;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = start + m * 64;
            let (b, ib) = (sb / 8, sb % 8);
            let wb = &data[b * 136..b * 136 + 136];
            let d = f16(wb, 0);
            let scales_h = u16::from_le_bytes([wb[2], wb[3]]);
            let ls = ((wb[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            let mut isum = 0i64;
            for j in 0..16 {
                let q = wb[8 + ib * 16 + j];
                isum += KVALUES_IQ4NL[(q & 0xF) as usize] as i64 * y_el(y, sb * 32 + j);
                isum += KVALUES_IQ4NL[(q >> 4) as usize] as i64 * y_el(y, sb * 32 + 16 + j);
            }
            acc += y[sb].d * dl * isum as f32;
        }
        lane[l] = acc as f64;
    }
    lane
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
                    let s1 = if signs[ib * 8 + sec * 4 + l] & KMASK_IQ2XS[j] != 0 {
                        -1i64
                    } else {
                        1i64
                    };
                    let s2 = if signs[ib * 8 + sec * 4 + l] & KMASK_IQ2XS[4 + j] != 0 {
                        -1i64
                    } else {
                        1i64
                    };
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
