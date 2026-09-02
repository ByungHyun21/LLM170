//! gemm_q2 — v2 양자화 GEMM: k-레인 병렬 (코얼레싱·레이턴시 병렬화).
//!
//! v1(gemm_q)는 유닛=행 순차 스챈: 워프 lanes가 서로 다른 행을 읽어 비코얼레싱 +
//! 유닛당 의존 부하 사슬로 레이턴시 병목 (~17× 대역폭 미달).
//! v2는 1행×SLICES레인: lanes가 같은 iteration에 연속 k를 읽어 완전 코얼레싱.
//! 요소 단위 디양자화(k 임의 접근) 후 레인 부분합 → part[t,o,s] → 결정적 reduce.
//! 누산 순서가 CPU(블록 순차)와 달라 f32 반올림 수준 차이만 발생.

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// 공용 헬퍼 (gemm_q와 동일 방식 — u32 워드 운반)
// ---------------------------------------------------------------------------

/// u32 워드에서 i바이트 추출 (리틀 엔디안).
#[cube]
pub(crate) fn byte(w: &Tensor<u32>, i: usize) -> u32 {
    (w[i >> 2] >> (((i & 3) * 8) as u32)) & 0xFF
}

/// f16 바이트 2개 → f32 (리틀 엔디안) — WGSL에 u16/f16이 없어 u32 비트 산술.
#[cube]
pub(crate) fn f16_at(w: &Tensor<u32>, off: usize) -> f32 {
    let h = byte(w, off) | (byte(w, off + 1) << 8);
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign)
        } else {
            let v = (frac as f32) * (1.0 / 16777216.0);
            if sign != 0 { -v } else { v }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F800000 | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
}

/// 바이트 값 → 부호 있는 f32 (ggml i8 재해석).
#[cube]
pub(crate) fn byte_signed(v: u32) -> f32 {
    let x = v as i32;
    if x > 127 { (x - 256) as f32 } else { x as f32 }
}

/// q4_K/q5_K 공용 6비트 스케일+min 추출 (get_scale_min_k4, ggml-quants.c:880).
#[cube]
pub(crate) fn scale_min_k4(j: usize, w: &Tensor<u32>, base: usize) -> (u32, u32) {
    if j < 4 {
        (byte(w, base + j) & 63, byte(w, base + j + 4) & 63)
    } else {
        (
            (byte(w, base + j + 4) & 0xF) | ((byte(w, base + j - 4) >> 6) << 4),
            (byte(w, base + j + 4) >> 4) | ((byte(w, base + j) >> 6) << 4),
        )
    }
}

// ---------------------------------------------------------------------------
// 요소 단위 디양자화: 블록 기준 오프셋 wb, 블록 내 인덱스 j → 값
// ---------------------------------------------------------------------------

#[cube]
fn de_elem_q8_0(w: &Tensor<u32>, wb: usize, j: usize) -> f32 {
    byte_signed(byte(w, wb + 2 + j)) * f16_at(w, wb)
}

#[cube]
fn de_elem_q4_k(w: &Tensor<u32>, wb: usize, j: usize) -> f32 {
    let sb = j / 32;
    let h = j % 32;
    let (sc, m) = scale_min_k4(sb, w, wb + 4);
    let qlb = byte(w, wb + 16 + (sb / 2) * 32 + h);
    let nib = if sb % 2 == 0 { qlb & 0xF } else { qlb >> 4 };
    f16_at(w, wb) * sc as f32 * nib as f32 - f16_at(w, wb + 2) * m as f32
}

#[cube]
fn de_elem_q5_k(w: &Tensor<u32>, wb: usize, j: usize) -> f32 {
    let sb = j / 32;
    let h = j % 32;
    let (sc, m) = scale_min_k4(sb, w, wb + 4);
    let qlb = byte(w, wb + 48 + (sb / 2) * 32 + h);
    let nib = if sb % 2 == 0 { qlb & 0xF } else { qlb >> 4 };
    let bit = (byte(w, wb + 16 + h) >> (sb as u32)) & 1;
    let v = nib + bit * 16;
    f16_at(w, wb) * sc as f32 * v as f32 - f16_at(w, wb + 2) * m as f32
}

#[cube]
fn de_elem_q6_k(w: &Tensor<u32>, wb: usize, j: usize) -> f32 {
    let h = j / 128;
    let r = j % 128;
    let pos = r / 32;
    let l = r % 32;
    let qlb = byte(w, wb + h * 64 + l + (pos % 2) * 32);
    let qhb = byte(w, wb + 128 + h * 32 + l);
    let nib: u32 = if pos == 0 {
        (qlb & 0xF) | ((qhb & 3) << 4)
    } else if pos == 1 {
        (qlb & 0xF) | (((qhb >> 2) & 3) << 4)
    } else if pos == 2 {
        (qlb >> 4) | (((qhb >> 4) & 3) << 4)
    } else {
        (qlb >> 4) | (((qhb >> 6) & 3) << 4)
    };
    let qv = (nib as i32) - 32;
    let sc = byte_signed(byte(w, wb + 192 + h * 8 + l / 16 + pos * 2));
    f16_at(w, wb + 208) * sc * qv as f32
}

#[cube]
fn de_elem_q3_k(w: &Tensor<u32>, wb: usize, j: usize) -> f32 {
    let n = j / 128;
    let r = j % 128;
    let si = r / 32;
    // CPU의 half는 {0,16} 바이트 오프셋 — 인덱스(hi)와 오프셋(half) 분리
    let hi = (r % 32) / 16;
    let half = hi * 16;
    let l = j % 16;
    // 12바이트 스케일 → 16 i8 (ggml-quants.c:1323-1328) — 요소마다 재계산
    let a0 = byte(w, wb + 96) | (byte(w, wb + 97) << 8) | (byte(w, wb + 98) << 16) | (byte(w, wb + 99) << 24);
    let a1 = byte(w, wb + 100) | (byte(w, wb + 101) << 8) | (byte(w, wb + 102) << 16) | (byte(w, wb + 103) << 24);
    let tmp = byte(w, wb + 104) | (byte(w, wb + 105) << 8) | (byte(w, wb + 106) << 16) | (byte(w, wb + 107) << 24);
    let k1 = 0x03030303u32;
    let k2 = 0x0f0f0f0fu32;
    let aux2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    let aux3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    let aux0 = (a0 & k2) | ((tmp & k1) << 4);
    let aux1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
    let ai = n * 8 + si * 2 + hi;
    let aux_v = if ai < 4 { aux0 } else if ai < 8 { aux1 } else if ai < 12 { aux2 } else { aux3 };
    let scb = (aux_v >> (((ai % 4) * 8) as u32)) & 0xFF;
    let dl = f16_at(w, wb + 108) * (byte_signed(scb) - 32.0);
    let qv = ((byte(w, wb + 32 + n * 32 + half + l) >> ((si * 2) as u32)) & 3) as i32;
    let bit = (byte(w, wb + half + l) >> (si as u32)) & 1;
    // 주의: `!=`/bool 조건 if-식은 HIP 코드젠 오컴파일 — 순수 산술 유지
    let sub = 4 - ((bit * 4) as i32);
    dl * (qv - sub) as f32
}

/// q5_1 요소: d(2) m(2) qh(4) qs(16) — x0|xh·16 형태 + m 가산.
#[cube]
fn de_elem_q5_1(w: &Tensor<u32>, wb: usize, j: usize) -> f32 {
    let d = f16_at(w, wb);
    let m = f16_at(w, wb + 2);
    let qh = byte(w, wb + 4) | (byte(w, wb + 5) << 8) | (byte(w, wb + 6) << 16) | (byte(w, wb + 7) << 24);
    let q = byte(w, wb + 8 + (j % 16));
    // ggml: xh0 = ((qh >> j) << 4) & 0x10; xh1 = (qh >> (j+12)) & 0x10
    let xh: u32 = if j < 16 {
        ((qh >> (j as u32)) << 4) & 0x10
    } else {
        (qh >> ((j as u32) - 16 + 12)) & 0x10
    };
    let nib: u32 = if j < 16 { q & 0xF } else { q >> 4 };
    let v = nib | xh;
    v as f32 * d + m
}

#[cube]
fn de_elem_iq4_xs(w: &Tensor<u32>, wb: usize, j: usize, ktab: &Tensor<f32>) -> f32 {
    let ib = j / 32;
    let h = j % 32;
    let qb = byte(w, wb + 8 + ib * 16 + (h % 16));
    let nib = if h < 16 { qb & 0xF } else { qb >> 4 };
    let scales_h = byte(w, wb + 2) | (byte(w, wb + 3) << 8);
    let ls = ((byte(w, wb + 4 + ib / 2) >> (((ib % 2) * 4) as u32)) & 0xF)
        | (((scales_h >> ((2 * ib) as u32)) & 3) << 4);
    f16_at(w, wb) * (ls as i32 - 32) as f32 * ktab[nib as usize]
}

#[cube]
fn de_elem_iq4_nl(w: &Tensor<u32>, wb: usize, j: usize, ktab: &Tensor<f32>) -> f32 {
    if j < 16 {
        f16_at(w, wb) * ktab[(byte(w, wb + 2 + j) & 0xF) as usize]
    } else {
        f16_at(w, wb) * ktab[(byte(w, wb + 2 + j - 16) >> 4) as usize]
    }
}

#[cube]
fn de_elem_iq3_s(
    w: &Tensor<u32>,
    wb: usize,
    j: usize,
    grid3: &Tensor<u32>,
) -> f32 {
    let ib = j / 64;
    let r = j % 64;
    let l = (r % 32) / 8; // sec 비트 제외 — 블록 내 루프 위치
    let q = r % 8;
    let second = r >= 32;
    let g2 = q >= 4;
    // qs 바이트: first loop {2l, 2l+1}, second loop {8+2l, 8+2l+1}
    let sec = r / 32; // 0|1 — second
    let g2i = q / 4; // 0|1
    let qs_off = sec * 8 + 2 * l + g2i;
    let qh_b = byte(w, wb + 66 + 2 * ib + sec);
    let shift = 8 - 2 * l - g2i;
    let idx = byte(w, wb + 2 + ib * 16 + qs_off) | ((qh_b << (shift as u32)) & 256);
    let g = grid3[idx as usize];
    let gval = byte_signed((g >> (((q % 4) * 8) as u32)) & 0xFF);
    // signs 바이트 오프셋은 루프(second), 비트 오프셋은 벡터(g2) — 산술 유도
    let sgn_b = byte(w, wb + 74 + ib * 8 + l + sec * 4);
    let bit = (sgn_b >> (((q % 4 + g2i * 4)) as u32)) & 1;
    let s = byte(w, wb + 106 + ib);
    let sel = (s >> ((sec * 4) as u32)) & 0xF;
    let db = f16_at(w, wb) * (1 + 2 * sel) as f32;
    db * gval * (1.0 - 2.0 * bit as f32)
}

/// 요소 값 산출 — qtype comptime 분기.
#[cube]
pub(crate) fn de_elem(
    w: &Tensor<u32>,
    wb: usize,
    j: usize,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    #[comptime] qtype: usize,
) -> f32 {
    if qtype == 0 {
        // F32: 블록 없음 — wb가 곧 요소 오프셋
        let bits = byte(w, wb) | (byte(w, wb + 1) << 8) | (byte(w, wb + 2) << 16) | (byte(w, wb + 3) << 24);
        f32::from_bits(bits)
    } else if qtype == 1 {
        f16_at(w, wb)
    } else if qtype == 30 {
        f32::from_bits((byte(w, wb) | (byte(w, wb + 1) << 8)) << 16)
    } else if qtype == 8 {
        de_elem_q8_0(w, wb, j)
    } else if qtype == 7 {
        de_elem_q5_1(w, wb, j)
    } else if qtype == 12 {
        de_elem_q4_k(w, wb, j)
    } else if qtype == 13 {
        de_elem_q5_k(w, wb, j)
    } else if qtype == 14 {
        de_elem_q6_k(w, wb, j)
    } else if qtype == 11 {
        de_elem_q3_k(w, wb, j)
    } else if qtype == 23 {
        de_elem_iq4_xs(w, wb, j, ktab)
    } else if qtype == 20 {
        de_elem_iq4_nl(w, wb, j, ktab)
    } else {
        de_elem_iq3_s(w, wb, j, grid3)
    }
}

// ---------------------------------------------------------------------------
// 커널
// ---------------------------------------------------------------------------

/// k-레인 GEMM: part[(t·n_out + o)·slices + l] = Σ_{k≡l (mod slices)} x[k]·W[o,k].
/// 큐브 = 1행(o) × 4토큰(t) × 64레인. lanes이 같은 iteration에 연속 k — 완전 코얼레싱.
#[cube(launch_unchecked)]
pub fn gemm_q2(
    x: &Tensor<f32>,
    w: &Tensor<u32>,
    part: &mut Tensor<f32>,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    n_in: usize,
    n_out: usize,
    t_len: usize,
    gx: usize,
    #[comptime] qtype: usize,
    #[comptime] slices: usize,
) {
    // o-차원 접힘: wgpu 그리드 X 상한 65,535 — 초과분 Z로 (2026-09-01).
    let o = CUBE_POS_X as usize + CUBE_POS_Z as usize * gx;
    let t = (CUBE_POS_Y * 4 + UNIT_POS_Y) as usize;
    let l = UNIT_POS_X as usize;
    if o >= n_out || t >= t_len || l >= slices {
        terminate!();
    }
    let blck = if qtype == 0 {
        1
    } else if qtype == 1 {
        1
    } else if qtype == 30 {
        1
    } else if qtype == 8 {
        32
    } else if qtype == 20 {
        32
    } else if qtype == 7 {
        32
    } else {
        256
    };
    let bsize = if qtype == 0 {
        4
    } else if qtype == 1 {
        2
    } else if qtype == 30 {
        2
    } else if qtype == 8 {
        34
    } else if qtype == 20 {
        18
    } else if qtype == 7 {
        24
    } else if qtype == 12 {
        144
    } else if qtype == 13 {
        176
    } else if qtype == 14 {
        210
    } else if qtype == 11 {
        110
    } else if qtype == 23 {
        136
    } else {
        110 // iq3_s (21)
    };
    let blocks = n_in / blck;
    let row_base = o * blocks * bsize;
    let xb = t * n_in;
    let mut acc = 0.0f32;
    for k in range_stepped(l, n_in, slices) {
        let b = k / blck;
        let j = k % blck;
        let v = de_elem(w, row_base + b * bsize, j, ktab, grid3, qtype);
        acc += x[xb + k] * v;
    }
    part[((t * n_out + o) * slices) + l] = acc;
}

/// k-레인 토큰-블록 GEMM (prefill) — q2의 토큰당 가중치 재디양자화 제거.
/// 큐브 = (행 o, 16토큰 블록) × 64레인: 가중치 원소 1회 디양자화로
/// 16토큰 동시 누산 (마이크로벤치 평탄 ~4 GFLOPS의 원인 — 2026-09-02).
/// 누산 순서는 q2와 동일(레인 분할 + reduce_parts 순차 축소) — 수치 불변.
#[cube(launch_unchecked)]
pub fn gemm_q7(
    x: &Tensor<f32>,
    w: &Tensor<u32>,
    part: &mut Tensor<f32>,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    n_in: usize,
    n_out: usize,
    t_len: usize,
    gx: usize,
    #[comptime] qtype: usize,
    #[comptime] slices: usize,
) {
    let o = CUBE_POS_X as usize + CUBE_POS_Z as usize * gx;
    let tb = CUBE_POS_Y as usize * BT;
    let l = UNIT_POS_X as usize;
    if o >= n_out || l >= slices {
        terminate!();
    }
    let blck = if qtype == 8 || qtype == 20 || qtype == 7 {
        32
    } else if qtype == 0 || qtype == 1 || qtype == 30 {
        1
    } else {
        256
    };
    let bsize = if qtype == 0 {
        4
    } else if qtype == 1 {
        2
    } else if qtype == 30 {
        2
    } else if qtype == 8 {
        34
    } else if qtype == 20 {
        18
    } else if qtype == 7 {
        24
    } else if qtype == 12 {
        144
    } else if qtype == 13 {
        176
    } else if qtype == 14 {
        210
    } else if qtype == 11 {
        110
    } else if qtype == 23 {
        136
    } else {
        110
    };
    let blocks = n_in / blck;
    let row_base = o * blocks * bsize;
    const BT: usize = 16;
    let mut acc = Array::<f32>::new(BT);
    for ti in 0..BT {
        acc[ti] = 0.0;
    }
    for k in range_stepped(l, n_in, slices) {
        let b = k / blck;
        let j = k % blck;
        let v = de_elem(w, row_base + b * bsize, j, ktab, grid3, qtype);
        for ti in 0..BT {
            let tt = tb + ti;
            if tt < t_len {
                acc[ti] += x[tt * n_in + k] * v;
            }
        }
    }
    for ti in 0..BT {
        let tt = tb + ti;
        if tt < t_len {
            part[(tt * n_out + o) * slices + l] = acc[ti];
        }
    }
}

/// 디버그: 블록 0의 요소별 디양자화 값 덤프 — gpu-de 서브커맨드 (python 대조용).
#[cube(launch_unchecked)]
pub fn debug_de(
    w: &Tensor<u32>,
    out: &mut Tensor<f32>,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    #[comptime] qtype: usize,
    #[comptime] n: usize,
) {
    let j = ABSOLUTE_POS_X as usize;
    if j >= n {
        terminate!();
    }
    let bsize = if qtype == 8 { 34 } else if qtype == 20 { 18 } else if qtype == 12 { 144 } else if qtype == 13 { 176 } else if qtype == 14 { 210 } else if qtype == 11 { 110 } else if qtype == 23 { 136 } else { 110 };
    out[j] = de_elem(w, 0, j, ktab, grid3, qtype);
    let _ = bsize;
}

/// 디버그: q3_K 중간값 덤프 — mode 0:bit 1:sub 2:qv 3:scb(signed).
#[cube(launch_unchecked)]
pub fn debug_q3(w: &Tensor<u32>, out: &mut Tensor<f32>, #[comptime] n: usize, #[comptime] mode: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j >= n {
        terminate!();
    }
    let n_ = j / 128;
    let r = j % 128;
    let si = r / 32;
    let hi = (r % 32) / 16;
    let half = hi * 16;
    let l = j % 16;
    let a0 = byte(w, 96) | (byte(w, 97) << 8) | (byte(w, 98) << 16) | (byte(w, 99) << 24);
    let a1 = byte(w, 100) | (byte(w, 101) << 8) | (byte(w, 102) << 16) | (byte(w, 103) << 24);
    let tmp = byte(w, 104) | (byte(w, 105) << 8) | (byte(w, 106) << 16) | (byte(w, 107) << 24);
    let k1 = 0x03030303u32;
    let k2 = 0x0f0f0f0fu32;
    let aux2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    let aux3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    let aux0 = (a0 & k2) | ((tmp & k1) << 4);
    let aux1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
    let ai = n_ * 8 + si * 2 + hi;
    let aux_v = if ai < 4 { aux0 } else if ai < 8 { aux1 } else if ai < 12 { aux2 } else { aux3 };
    let scb = (aux_v >> (((ai % 4) * 8) as u32)) & 0xFF;
    let bit = (byte(w, half + l) >> (si as u32)) & 1;
    let sub = 4 - ((bit * 4) as i32);
    let qv = ((byte(w, 32 + n_ * 32 + half + l) >> ((si * 2) as u32)) & 3) as i32;
    if mode == 0 {
        out[j] = bit as f32;
    } else if mode == 1 {
        out[j] = sub as f32;
    } else if mode == 2 {
        out[j] = qv as f32;
    } else {
        out[j] = byte_signed(scb);
    }
}

/// 디버그: 블록 0의 원시 바이트 값 덤프 — 전송 정합성 확인용.
#[cube(launch_unchecked)]
pub fn debug_bytes(w: &Tensor<u32>, out: &mut Tensor<f32>, #[comptime] n: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j >= n {
        terminate!();
    }
    out[j] = byte(w, j) as f32;
}

/// 디코드 전용 k-레인 GEMM — 토큰 상각: 디양자화 1회로 tlen 토큰 동시 누산.
/// 유닛 = (행 o, k-레인 l). tlen comptime {1,2,4,8}, x는 tlen행으로 패딩 업로드.
#[cube(launch_unchecked)]
pub fn gemm_q3(
    x: &Tensor<f32>,
    w: &Tensor<u32>,
    part: &mut Tensor<f32>,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    n_in: usize,
    n_out: usize,
    gx: usize,
    #[comptime] tlen: usize,
    #[comptime] qtype: usize,
) {
    // o-차원 접힘 (wgpu 65,535 상한).
    let o = CUBE_POS_X as usize + CUBE_POS_Z as usize * gx;
    let l = UNIT_POS_X as usize;
    if o >= n_out || l >= 64 {
        terminate!();
    }
    let blck = if qtype == 0 {
        1
    } else if qtype == 1 {
        1
    } else if qtype == 30 {
        1
    } else if qtype == 8 {
        32
    } else if qtype == 20 {
        32
    } else if qtype == 7 {
        32
    } else {
        256
    };
    let bsize = if qtype == 0 {
        4
    } else if qtype == 1 {
        2
    } else if qtype == 30 {
        2
    } else if qtype == 8 {
        34
    } else if qtype == 20 {
        18
    } else if qtype == 7 {
        24
    } else if qtype == 12 {
        144
    } else if qtype == 13 {
        176
    } else if qtype == 14 {
        210
    } else if qtype == 11 {
        110
    } else if qtype == 23 {
        136
    } else {
        110 // iq3_s (21)
    };
    let blocks = n_in / blck;
    let row_base = o * blocks * bsize;
    let mut acc = Array::<f32>::new(tlen);
    #[unroll]
    for ti in 0..tlen {
        acc[ti] = 0.0;
    }
    for k in range_stepped(l, n_in, 64) {
        let b = k / blck;
        let j = k % blck;
        let v = de_elem(w, row_base + b * bsize, j, ktab, grid3, qtype);
        #[unroll]
        for ti in 0..tlen {
            acc[ti] += x[ti * n_in + k] * v;
        }
    }
    #[unroll]
    for ti in 0..tlen {
        part[(ti * n_out + o) * 64 + l] = acc[ti];
    }
}

/// W4A8 변형 디코드 GEMM — x를 q8로 사전 양자화해 전송(qs u32 워드 + 블록 d).
/// q3 구조(토큰 상각·k-레인)에서 x[k] 대신 q8[k]·d[k/32] 재구성.
/// CMP 정수 누산 버전(dp4a류)은 하드웨어 도착 후 변형으로 추가 예정.
#[cube(launch_unchecked)]
pub fn gemm_q6(
    xq: &Tensor<u32>,   // i8 4개/워드
    xd: &Tensor<f32>,   // q8 블록 스케일 [tlen·n_in/32]
    w: &Tensor<u32>,
    part: &mut Tensor<f32>,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    n_in: usize,
    n_out: usize,
    #[comptime] tlen: usize,
    #[comptime] qtype: usize,
) {
    let o = CUBE_POS_X as usize;
    let l = UNIT_POS_X as usize;
    if o >= n_out || l >= 64 {
        terminate!();
    }
    let blck = if qtype == 0 {
        1
    } else if qtype == 1 {
        1
    } else if qtype == 30 {
        1
    } else if qtype == 8 {
        32
    } else if qtype == 20 {
        32
    } else if qtype == 7 {
        32
    } else {
        256
    };
    let bsize = if qtype == 0 {
        4
    } else if qtype == 1 {
        2
    } else if qtype == 30 {
        2
    } else if qtype == 8 {
        34
    } else if qtype == 20 {
        18
    } else if qtype == 7 {
        24
    } else if qtype == 12 {
        144
    } else if qtype == 13 {
        176
    } else if qtype == 14 {
        210
    } else if qtype == 11 {
        110
    } else if qtype == 23 {
        136
    } else {
        110 // iq3_s (21)
    };
    let blocks = n_in / blck;
    let row_base = o * blocks * bsize;
    let mut acc = Array::<f32>::new(tlen);
    #[unroll]
    for ti in 0..tlen {
        acc[ti] = 0.0;
    }
    for k in range_stepped(l, n_in, 64) {
        let b = k / blck;
        let j = k % blck;
        let v = de_elem(w, row_base + b * bsize, j, ktab, grid3, qtype);
        // x 재구성: q8 정수 × 블록 d
        let word = xq[k >> 2];
        let shift = ((k & 3) * 8) as u32;
        let qi = byte_signed((word >> shift) & 0xFF);
        let xrec = qi * xd[k / 32];
        #[unroll]
        for ti in 0..tlen {
            acc[ti] += xrec * v;
        }
    }
    #[unroll]
    for ti in 0..tlen {
        part[(ti * n_out + o) * 64 + l] = acc[ti];
    }
}

/// 부분합 결정적 축소: out[t·n_out+o] = Σ_s part[...s] (s 순차).
#[cube(launch_unchecked)]
pub fn reduce_parts(
    part: &Tensor<f32>,
    out: &mut Tensor<f32>,
    n_out: usize,
    t_len: usize,
    gx: usize,
    #[comptime] slices: usize,
) {
    // o = 유닛 절대 인덱스(큐브×64+유닛) + Z레이어 오프셋 — wgpu 65,535 접힘.
    let o = ABSOLUTE_POS_X as usize + CUBE_POS_Z as usize * gx * 64;
    let t = ABSOLUTE_POS_Y as usize;
    if o >= n_out || t >= t_len {
        terminate!();
    }
    let base = (t * n_out + o) * slices;
    let mut acc = 0.0f32;
    for s in 0..slices {
        acc += part[base + s];
    }
    out[t * n_out + o] = acc;
}
