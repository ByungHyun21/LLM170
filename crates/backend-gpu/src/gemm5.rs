//! gemm_q5 — MoE 전문가 배치 down GEMM (mul_mat_id류).
//!
//! 짝(matmul_paired)이 전문가당 1런치×K동기화였다면, 이 커널은
//! **K개 전문가의 down을 1런치**로: 그리드 (n_out, K), 큐브당 (전문가 e,
//! 출력 o). x는 전문가별 1행이 평탄하게 이어진 [K·n_in].
//! 가중치는 3D 스택 [K][n_out][n_in 블록] — expert_w 뷰 대신 스택 전체를
//! 1개 상주 가중치로 업로드한다 (P2-2, 2026-09-01).
//!
//! 구조는 gemm_q3(k-레인·part+reduce)와 동일 — 워드 오프셋에
//! e·per_expert_words 만 더한다. ADR-0005: mul_add 금지.

use cubecl::prelude::*;

/// K 전문가 × (n_in→n_out) 1행 GEMM — part[(e·n_out+o)·64+l] = Σ_{k≡l} x[e·n_in+k]·W[e,o,k]
#[cube(launch_unchecked)]
pub fn gemm_q5(
    x: &Tensor<f32>,          // [K·n_in] 행 순서 = exps 순
    w: &Tensor<u32>,          // 스택 [n_expert][rows] 원시 워드
    part: &mut Tensor<f32>,   // [K·n_out·64]
    exps: &Tensor<u32>,       // [K] 행→전문가 id 매핑
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    n_in: usize,
    n_out: usize,
    exp_bytes: usize,          // 전문가 1개의 바이트 수 (row_base는 바이트 단위)
    gx: usize,
    #[comptime] qtype: usize,
) {
    // o-차원 접힘 (wgpu 65,535 상한).
    let o = CUBE_POS_X as usize + CUBE_POS_Z as usize * gx;
    let gy = CUBE_POS_Y as usize;
    let e = exps[gy] as usize; // 실제 전문가 id — 행 순서와 독립
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
    let row_base = e * exp_bytes + o * blocks * bsize;
    let xb = gy * n_in;
    let mut acc = 0.0f32;
    for k in range_stepped(l, n_in, 64) {
        let b = k / blck;
        let j = k % blck;
        let v = crate::gemm2::de_elem(w, row_base + b * bsize, j, ktab, grid3, qtype);
        acc += x[xb + k] * v;
    }
    part[(gy * n_out + o) * 64 + l] = acc;
}
