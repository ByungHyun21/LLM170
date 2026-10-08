//! matmul — CPU 기준 (가속기 계층 제거 2026-10-08).

pub mod cpu;
pub mod weight;

pub use cpu::{greedy_from, matmul, matmul_batch, n_threads};
pub use weight::Weight;
