//! llm170-core — 모델 구현 코어 (CPU 참조 백엔드).

pub mod gdn;
pub mod gdn_norm;
pub mod matmul;
pub mod qwen35;
pub mod sampler;
pub mod qwen4exp;
pub mod clip;
pub mod clip_preproc;
pub mod ops;
pub mod quant;
mod tables;
pub use tables::{IQ3S_GRID, KVALUES_IQ4NL, ktab2_packed};
