//! qwen35 mm_group 디스패치 — 가속기 없으면 CPU 개별 배치 (matmul/에서 이관,
//! qwen35 전용 ModelError 반환이라 소속을 qwen35로).
/// matmul_group 디스패치 — 가속기 없으면 CPU 개별 배치.
use crate::matmul::cpu::{matmul, matmul_batch};
use crate::matmul::traits::*;
use crate::matmul::weight::Weight;
pub fn mm_group(
    acc: &Acc,
    xs: &[Vec<f32>],
    ws: &[Weight],
    outs: &mut [Vec<Vec<f32>>],
) -> Result<(), crate::qwen35::ModelError> {
    match acc.as_deref() {
        Some(a) => a
            .matmul_group(xs, ws, outs)
            .map_err(crate::qwen35::ModelError::Accel),
        None => {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                matmul_batch(xs, w, out);
            }
            Ok(())
        }
    }
}

pub type Acc = Option<std::sync::Arc<dyn Accelerator>>;

/// matmul_batch 디스패치 — 가속기 없으면 CPU 스레드 경로.
pub fn mm_batch(
    acc: &Acc,
    xs: &[Vec<f32>],
    w: &Weight,
    outs: &mut [Vec<f32>],
) -> Result<(), crate::qwen35::ModelError> {
    match acc.as_deref() {
        Some(a) => a
            .matmul_batch(xs, w, outs)
            .map_err(crate::qwen35::ModelError::Accel),
        None => {
            matmul_batch(xs, w, outs);
            Ok(())
        }
    }
}

/// matmul 디스패치.
pub fn mm(
    acc: &Acc,
    x: &[f32],
    w: &Weight,
    out: &mut [f32],
) -> Result<(), crate::qwen35::ModelError> {
    match acc.as_deref() {
        Some(a) => a
            .matmul(x, w, out)
            .map_err(crate::qwen35::ModelError::Accel),
        None => {
            matmul(x, w, out);
            Ok(())
        }
    }
}
