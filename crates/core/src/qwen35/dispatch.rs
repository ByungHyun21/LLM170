//! qwen35 CPU 디스패치 — 가속기 계층 제거(2026-10-08)로 직접 호출만 남는다.

use crate::matmul::cpu::{matmul, matmul_batch};
use crate::matmul::weight::Weight;

/// 그룹 배치 — x는 공통 입력, ws[i] → outs[i] (각각 CPU 배치 내적).
pub fn mm_group(xs: &[Vec<f32>], ws: &[Weight], outs: &mut [Vec<Vec<f32>>]) {
    for (w, out) in ws.iter().zip(outs.iter_mut()) {
        matmul_batch(xs, w, out);
    }
}

/// 단일 가중 배치 내적.
pub fn mm_batch(xs: &[Vec<f32>], w: &Weight, outs: &mut [Vec<f32>]) {
    matmul_batch(xs, w, outs);
}

/// [P12 E3] 참조 행 그룹 변형.
pub fn mm_group_ref(xs: &[&[f32]], ws: &[Weight], outs: &mut [Vec<Vec<f32>>]) {
    for (w, out) in ws.iter().zip(outs.iter_mut()) {
        crate::matmul::cpu::matmul_batch_ref(xs, w, out);
    }
}

/// 단일 행 내적.
pub fn mm(x: &[f32], w: &Weight, out: &mut [f32]) {
    matmul(x, w, out);
}
