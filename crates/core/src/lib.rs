//! llm170-core — 모델 구현 코어 (CPU 참조 백엔드).

pub mod clip;
pub mod clip_preproc;
pub mod deepseek4;
pub mod gdn;
pub mod gdn_norm;
pub mod json;
pub mod matmul;
pub mod ops;
pub mod quant;
pub mod qwen35;
pub mod qwen4exp;
pub mod sampler;
pub mod st;
pub mod w4a16;
pub use tables::{IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_IQ4NL, ktab2_packed};
mod tables;
