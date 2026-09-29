//! rawvk/checks — 진단 체커(plans/90 B1: gemv.rs에서 순수 이동; plans/107 W4: 모듈 트리 분할).
//! `llm170 vk-*-check` 계열 CLI가 호출하는 CPU 대조 검증 — 프로덕션 경로와 무관.

mod attention;
mod fault;
mod frame_check;
mod harness;
mod misc;
mod ops;

pub use attention::{ft32_check, gdn_chunk_check, gemv_check, gemv8_check};
pub use fault::fault_probe;
pub use frame_check::frame_check;
pub use misc::ple_mt_check;
pub use ops::idot_probe;
