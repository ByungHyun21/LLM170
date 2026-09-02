//! GPU 백엔드 — 순수 Rust 원시 HIP 실행기 (2026-09-03 cubecl 제거).
//!
//! `rawhip`: hipRTC로 임베디드 HIP C++를 컴파일해 hipModuleLaunchKernel로 실행.
//! 커널 산술은 core 미러(dot_row_w4a8_*_lane)와 토큰당 동일 연산열 — to_bits 검증 게이트.
//! 비트계약: raw-HIP greedy 스트림 ≡ CPU W4A8 참조 엔진 (12+64토큰 교차검증).

pub mod rawhip;

pub use rawhip::decode::RawDecoder;
pub use rawhip::{bw_test, dp4a_test, exp_ab, qk_check, q6k_ab_test, raw_probe, tree_test};
