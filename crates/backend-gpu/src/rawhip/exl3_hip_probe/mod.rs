//! EXL3 hip 검증 프로브 — 프로브별 파일 분해(plans/129 R6, 순수 이동).
//! 각 파일이 프로브 1개(검증기·하네스 저작 원칙은 server/probes/mod.rs 헤더).
//! 측정 원장·corr 기준은 각 프로브 헤더와 plans/archive 참조.

// ── EXL3 hip GEMV 체인 프로브(plans/121 CMP 포팅 · todo 2/4) ──
// vk 가중치를 그대로 투입해 hipRTC 컴파일 exl3_had_in→gemv→had_out 체인을
// 실행, vk 트레이리던트 참조(tr.linear)와 대조 — 8060S hipRTC로 검증.

mod a1;
mod attn;
mod batch;
mod decode;
mod gdn;
mod gemm;
mod gemv;
mod graph;
mod hcmp;
mod linear;
mod mtp;
mod mtp_round;
mod nr;
mod tbench;

pub use a1::*;
pub use attn::*;
pub use batch::*;
pub use decode::*;
pub use gdn::*;
pub use gemm::*;
pub use gemv::*;
pub use graph::*;
pub use hcmp::*;
pub use linear::*;
pub use mtp::*;
pub use mtp_round::*;
pub use nr::*;
pub use tbench::*;
