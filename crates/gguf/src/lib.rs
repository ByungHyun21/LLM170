//! GGML 가중치 타입 사전 — **GGUF 파서는 탈락**(2026-10-08 방향 전환,
//! plans/w4a16-cuda.md §5: EXL3·GGUF 탈락·CUDA W4A16 단일).
//!
//! `GgmlType`은 `Weight.ty` 계약의 타입 태그로 존치한다 — GGUF 파일 파싱
//! (GgufFile/Value/dump)은 삭제됐고, 레이아웃 상수(block_info)와 이름만 유지.

mod types;

pub use types::GgmlType;
