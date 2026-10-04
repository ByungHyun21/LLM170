//! rawvk/checks — 진단 체커(plans/90 B1: gemv.rs에서 순수 이동; plans/107 W4: 모듈 트리 분할).
//! `llm170 vk-*-check` 계열 CLI가 호출하는 CPU 대조 검증 — 프로덕션 경로와 무관.

mod attention;
mod exl3_bench;
pub(crate) mod exl3_probes;
mod fault;
mod frame_check;
mod gdn;
mod harness;
mod misc;
mod ops;

pub use crate::rawvk::exl3::decode::exl3_decode;
pub use crate::rawvk::exl3::decode::exl3_mtp;
pub use crate::rawvk::exl3::decode::exl3_mtp2;
pub use crate::rawvk::exl3::decode::exl3_pp;
pub use crate::rawvk::exl3::decode::{
    SeqState, decode_step, exl3_spec_step, new_seq_state, prefill_batch, prefill_batch_spec,
};
pub use crate::rawvk::exl3::resident::TrellisResident;
pub use crate::rawvk::exl3::util::exl3_vk_check;
pub use attention::{ft32_check, gemv_check, gemv8_check};
pub use exl3_bench::exl3_bench;
pub use exl3_probes::{attn_check, chain_check, ffn_check, gemm_check, nr_check, scan_check};
pub use fault::fault_probe;
pub use frame_check::frame_check;
pub use misc::ple_mt_check;
pub use ops::idot_probe;

pub use gdn::{gdn_check, smoke_test};
