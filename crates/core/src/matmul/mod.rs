//! matmul — 가속기 트레이트·CPU 기준·디스패치 (107 W6: 단일 파일 → 모듈 절단).
pub mod cpu;
pub mod raw;
pub mod traits;
pub mod weight;

pub use cpu::{greedy_from, matmul, matmul_batch, matmul_w4a8, n_threads, w4a8_enabled, w4a8_ty};
pub use raw::RawDecode;
pub use traits::*;
pub use weight::Weight;
