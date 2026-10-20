//! [FN 2단계 2026-10-17] 레이어 그래프 스테이지 — 판독 검증
//! (실제 텐서 forward). [D2 2026-10-20] 상태ful 오라클 완성 —
//! hc(hyper-connection)·qsa(indexer+선택 집합 어텐션) 추가.

pub mod full;
pub mod hc;
pub mod linear;
pub mod moe;
pub mod ple;
pub mod qsa;
