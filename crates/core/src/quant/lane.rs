//! W4A8 레인 미러 정수 내적(64레인 f64 부분합) + 형식 디스패치.

use super::deq::KMASK_IQ2XS;
/// W4A8: 한 행 k원소의 W4A8 내적 — 타입별 분기. data는 행 시작.
use super::deq::{dequant_row, f16};
use super::q8::*;
#[allow(unused_imports)]
use super::q8::{y_el, y_f};
use crate::tables::{IQ3S_GRID, KVALUES_IQ4NL};
use crate::wtype::WType;
pub fn dot_row_w4a8(ty: WType, data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let (blck, bsize) = ty.block_info();
    let blocks = (k / blck) as usize;
    let bsize = bsize as usize;
    let mut acc = 0.0f32;
    for b in 0..blocks {
        let wb = &data[b * bsize..b * bsize + bsize];
        let yb = &y[b * (blck as usize / 32)..b * (blck as usize / 32) + blck as usize / 32];
        let v = match ty {
            WType::Q4K => dot_q4k_q8(wb, yb),
            WType::Q5K => dot_q5k_q8(wb, yb),
            WType::Q6K => dot_q6k_q8(wb, yb),
            WType::Q3K => dot_q3k_q8(wb, yb),
            WType::Q8_0 => dot_q8k_q8(wb, yb),
            WType::Q5_1 => dot_q5_1_q8(wb, yb),
            WType::Iq4Xs => dot_iq4xs_q8(wb, yb),
            WType::Iq4Nl => dot_iq4nl_q8(wb, yb),
            WType::Iq3S => dot_iq3s_q8(wb, yb),
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

/// W4A8 레인 미러(q3_K) — GPU gemm_q8i_q3k와 동일 구조. 16요소 하프블록
/// (n,si,half)별 c = yd·dl·isum (f32 곱 2), 스트라이드 레인(레인 l:
/// 하프블록 l, l+64, …) f64 누산 → 레인 순서 합 → f32 1회 캐스트.
pub fn dot_row_w4a8_q3k_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_q3k_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_q3k_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_h = (k / 16) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_h + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let h = l + m * 64;
            let local = h % 16;
            let (n, si, half) = (local / 8, (local % 8) / 2, local % 2);
            let wb = &data[(h / 16) * 110..(h / 16) * 110 + 110];
            let d_all = f16(wb, 108);
            let (a0, a1, tmp) = (
                u32::from_le_bytes([wb[96], wb[97], wb[98], wb[99]]),
                u32::from_le_bytes([wb[100], wb[101], wb[102], wb[103]]),
                u32::from_le_bytes([wb[104], wb[105], wb[106], wb[107]]),
            );
            let (k1, k2) = (0x03030303u32, 0x0f0f0f0fu32);
            let aux2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            let aux3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            let aux0 = (a0 & k2) | ((tmp & k1) << 4);
            let aux1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
            let ai = n * 8 + si * 2 + half;
            let aux = [aux0, aux1, aux2, aux3][ai / 4];
            let scb = (aux >> (8 * (ai % 4))) & 0xFF;
            let dl = d_all * (scb as i8 as f32 - 32.0);
            let mut isum = 0i64;
            for j in 0..16 {
                let qv = ((wb[32 + n * 32 + half * 16 + j] >> (2 * si)) & 3) as i64;
                let sub = if wb[half * 16 + j] & (1 << si) != 0 {
                    0i64
                } else {
                    4i64
                };
                isum += (qv - sub) * y_el(y, h * 16 + j);
            }
            let yd = y[h / 2].d;
            acc += yd * dl * isum as f32;
        }
        lane[l] = acc as f64;
    }
    lane
}

/// W4A8 레인 미러(q5_K) — 분할 형태: 뺄셈(d·isum − m·qsum)을 순수 곱
/// 체인 2개로 분리해 FMA 수축 면역 (GPU gemm_q8i_q5k와 동일 연산열).
/// 서브블록 32원소, 스트라이드 레인, f64 누산.
pub fn dot_row_w4a8_q5k_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_q5k_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_q5k_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        // f32 레인 누산 — GPU f64가 1/16 레이트라 병목 (2026-09-04 RCA).
        // 커널과 동일 열: f32 가산 후 f64 트리 (tree64)로 결합.
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = l + m * 64;
            let js = sb % 8;
            let (it, half) = (js / 2, js % 2);
            let wb = &data[(sb / 8) * 176..(sb / 8) * 176 + 176];
            let d = f16(wb, 0);
            let dm = f16(wb, 2);
            let (sc, m_) = scale_min_k4_local(wb, sb % 8);
            let (mut isum, mut qsum) = (0i64, 0i64);
            let u = if half == 0 {
                1u8 << (2 * it)
            } else {
                2u8 << (2 * it)
            };
            for j in 0..32 {
                let nib = if half == 0 {
                    wb[48 + it * 32 + j] & 0xF
                } else {
                    wb[48 + it * 32 + j] >> 4
                };
                let hi = if wb[16 + j] & u != 0 { 16i64 } else { 0i64 };
                let yv = y_el(y, sb * 32 + j);
                isum += (nib as i64 + hi) * yv;
                qsum += yv;
            }
            let yd = y[sb].d;
            // 분할: c1 = yd·(d·sc)·isum, c2 = yd·(dm·m)·qsum — 곱 체인만
            acc += yd * (d * sc as f32) * isum as f32;
            acc -= yd * (dm * m_ as f32) * qsum as f32;
        }
        lane[l] = acc as f64;
    }
    lane
}

/// q4_K/q5_K 6비트 scale/min (스토어 내부, 블록 절대 오프셋 버전).
pub fn scale_min_k4_local(wb: &[u8], sb: usize) -> (u32, u32) {
    let sc = &wb[4..16];
    let j = sb; // 서브블록 = scale 인덱스 (0..7)
    if j < 4 {
        ((sc[j] & 63) as u32, (sc[j + 4] & 63) as u32)
    } else {
        (
            ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as u32,
            ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as u32,
        )
    }
}

/// W4A8 레인 미러(q4_K) — q5_K과 동일 분할 형태, qh 없음 (qs 128B).
pub fn dot_row_w4a8_q4k_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_q4k_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_q4k_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = l + m * 64;
            let js = sb % 8;
            let (it, half) = (js / 2, js % 2);
            let wb = &data[(sb / 8) * 144..(sb / 8) * 144 + 144];
            let d = f16(wb, 0);
            let dm = f16(wb, 2);
            let (sc, m_) = scale_min_k4_local(wb, js);
            let (mut isum, mut qsum) = (0i64, 0i64);
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
        lane[l] = acc as f64;
    }
    lane
}

/// W4A8 레인 미러(q8_0) — 32원소 블록 = 서브블록 (블록 상위 구조 없음).
pub fn dot_row_w4a8_q8_0_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_q8_0_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_q8_0_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = l + m * 64;
            let wb = &data[sb * 34..sb * 34 + 34];
            let d = f16(wb, 0);
            let mut isum = 0i64;
            for j in 0..32 {
                isum += (wb[2 + j] as i8) as i64 * y_el(y, sb * 32 + j);
            }
            let yd = y[sb].d;
            acc += yd * d * isum as f32;
        }
        lane[l] = acc as f64;
    }
    lane
}

/// W4A8 레인 미러(q5_1) — 32원소 블록 = d(f16)·q + m(f16), q 무부호 5bit.
/// q = lo4 + 16·hi1 분해 → isum = Σxq·lo4 + 16·Σxq·hi1, s1 = Σxq.
/// HIP `q4_gemm_q5_1`과 동일 연산열 (block: yd·(d·isum + m·s1) f32 누산).
pub fn dot_row_w4a8_q5_1_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_q5_1_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_q5_1_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = l + m * 64;
            let wb = &data[sb * 24..sb * 24 + 24];
            let d = f16(wb, 0);
            let mn = f16(wb, 2);
            let qh = u32::from_le_bytes([wb[4], wb[5], wb[6], wb[7]]);
            let mut s_lo = 0i64;
            let mut s_hi = 0i64;
            let mut s1 = 0i64;
            for j in 0..32usize {
                let byte = wb[8 + (j & 15)] as u32;
                let lo4 = if j < 16 {
                    byte & 0xF
                } else {
                    (byte >> 4) & 0xF
                };
                let hi1 = (qh >> j) & 1;
                let yv = y_el(y, sb * 32 + j);
                s_lo += yv * lo4 as i64;
                s_hi += yv * hi1 as i64;
                s1 += yv;
            }
            let yd = y[sb].d;
            let isum = s_lo + (s_hi << 4);
            acc += yd * (d * isum as f32 + mn * s1 as f32);
        }
        lane[l] = acc as f64;
    }
    lane
}

/// W4A8 레인 미러(iq4_nl) — 32원소 블록, ktab 정수 룩업.
pub fn dot_row_w4a8_iq4nl_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_iq4nl_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_iq4nl_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = l + m * 64;
            let wb = &data[sb * 18..sb * 18 + 18];
            let d = f16(wb, 0);
            let mut isum = 0i64;
            for j in 0..16 {
                let q = wb[2 + j];
                isum += KVALUES_IQ4NL[(q & 0xF) as usize] as i64 * y_el(y, sb * 32 + j);
                isum += KVALUES_IQ4NL[(q >> 4) as usize] as i64 * y_el(y, sb * 32 + 16 + j);
            }
            let yd = y[sb].d;
            acc += yd * d * isum as f32;
        }
        lane[l] = acc as f64;
    }
    lane
}

/// W4A8 레인 미러(q6_K) — 16원소 그룹(16개/256블록), 그룹 g = h*8 + src*2 + p
/// (src∈0..3: ql lo/hi × 오프셋0/32, p∈{0,1}). c = ((yd·d)·sc)·isum (순수
/// 곱 체인). 스트라이드 레인, f64 누산 — GPU gemm_q8i_q6k 미러.
pub fn dot_row_w4a8_q6k_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_q6k_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_q6k_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_g = (k / 16) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_g + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let g = l + m * 64;
            let blk = g / 16;
            let klocal = g % 16;
            let wb = &data[blk * 210..blk * 210 + 210];
            let h = klocal / 8;
            let src = (klocal % 8) / 2; // 0..3
            let p = klocal % 2;
            let d = f16(wb, 208);
            let sc = wb[192 + klocal] as i8;
            let mut isum = 0i64;
            for jj in 0..16 {
                let ll = p * 16 + jj;
                // qh는 블록 오프셋 128부터 (ql은 0부터) — RCA: +128 누락이
                // 미러·f32 7× 발산 원인이었음.
                let (nib, hi2) = match src {
                    0 => (wb[h * 64 + ll] & 0xF, (wb[128 + h * 32 + ll] & 3) as i64),
                    1 => (
                        wb[h * 64 + ll + 32] & 0xF,
                        ((wb[128 + h * 32 + ll] >> 2) & 3) as i64,
                    ),
                    2 => (
                        wb[h * 64 + ll] >> 4,
                        ((wb[128 + h * 32 + ll] >> 4) & 3) as i64,
                    ),
                    _ => (
                        wb[h * 64 + ll + 32] >> 4,
                        ((wb[128 + h * 32 + ll] >> 6) & 3) as i64,
                    ),
                };
                let elem = blk * 256 + h * 128 + src * 32 + p * 16 + jj;
                isum += (((nib as i64) | (hi2 << 4)) - 32) * y_el(y, elem);
            }
            let pos = src;
            let yd = y[blk * 8 + h * 4 + pos].d;
            acc += yd * d * sc as f32 * isum as f32;
        }
        lane[l] = acc as f64;
    }
    lane
}

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

// ─── W4A16(GPTQ4·g128·sym) 레인 미러 ───
// AutoRound auto_gptq 규약: qrow 워드 j의 니블 j%8 = 원소 8·(j/8)+(j%8).
// lsb-first **확정**(2026-10-08, w4a16-xcheck — 동일 기저 27B 원본 대조
// corr(lsb) 0.991~0.994 vs corr(msb) ≈0.01, §3.6 종결).
// zrow = 로더가 행별 언팩한 zp(그룹당 1개, sym 전형 8), srow = 그룹 f16 스케일.
// GPU gemm_gptq4(64레인)와 동일 연산열: 레인 l = 원소 l, l+64, … f32 누산 →
// f64 레인 배열 → tree64(기존 레인 계약 준수).

/// f16 비트(u16) → f32 (IEEE half: 상위 16비트 시프트, 정확 변환).
#[inline]
fn f16b(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
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

/// (yd·db·isum) f32 → f64 레인 누산 (GPU gemm_iq3s와 동일 연산열).
pub fn dot_row_w4a8_iq3s_lane(data: &[u8], k: u64, y: &[Q8Block]) -> f32 {
    let lane = dot_row_w4a8_iq3s_lane_parts(data, k, y);
    tree64(&lane) as f32
}

pub fn dot_row_w4a8_iq3s_lane_parts(data: &[u8], k: u64, y: &[Q8Block]) -> [f64; 64] {
    let n_sub = (k / 32) as usize;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sub = l + m * 64;
            let blk = sub / 8;
            let h = sub % 8;
            let wb = &data[blk * 110..blk * 110 + 110];
            let d_all = f16(wb, 0);
            let scb = wb[106 + (h >> 1)];
            let nib = if h & 1 != 0 { scb >> 4 } else { scb & 0xF };
            let db = d_all * (1 + 2 * nib as i32) as f32;
            let qhb = wb[66 + h] as u32;
            let qs_base = 2 + h * 8;
            let sg_base = 74 + h * 4;
            let mut isum = 0i64;
            for ll in 0..4usize {
                let i1 = (wb[qs_base + 2 * ll] as u32) | ((qhb << (8 - 2 * ll)) & 256);
                let i2 = (wb[qs_base + 2 * ll + 1] as u32) | ((qhb << (7 - 2 * ll)) & 256);
                let g1 = IQ3S_GRID[i1 as usize];
                let g2 = IQ3S_GRID[i2 as usize];
                let sgb = wb[sg_base + ll];
                let e0 = 8 * ll;
                for j in 0..4usize {
                    let w1 = ((g1 >> (8 * j)) & 0xFF) as i8 as i32
                        * if sgb & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                    let w2 = ((g2 >> (8 * j)) & 0xFF) as i8 as i32
                        * if sgb & KMASK_IQ2XS[4 + j] != 0 { -1 } else { 1 };
                    let e1 = e0 + j;
                    let e2 = e0 + 4 + j;
                    isum +=
                        (w1 as i64) * y_el(y, sub * 32 + e1) + (w2 as i64) * y_el(y, sub * 32 + e2);
                }
            }
            let yd = y[sub].d;
            acc += yd * db * isum as f32;
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
