//! llm170-diag — 공유 진단 계층 (plans/82).
//!
//! 층 계약: 이 크레이트는 순수 Rust(GPU 의존 0)로, 백엔드가 공급하는
//! **resolved 이벤트**(시간 계산 완료)를 표현·저장·분석한다.
//! 의존 방향: `diag ← core, gguf, backend-gpu, server`.

pub mod trace;
pub mod writer;
pub mod flag;
pub mod fp;
pub mod span;
pub mod dump;
pub mod alloc;

pub use trace::Ev;
pub use trace::capture_on;
pub use flag::env_on;
pub use fp::{fp_record, fp_diff};
