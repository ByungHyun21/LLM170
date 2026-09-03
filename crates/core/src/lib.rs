//! llm170-core — 모델 구현 코어 (CPU 참조 백엔드).

pub mod gdn;
pub mod mode;
pub mod matmul;
pub mod model;
pub mod qwen4exp;
pub mod clip;
mod ops;
pub mod quant;
mod tables;
pub use tables::{IQ3S_GRID, KVALUES_IQ4NL};
