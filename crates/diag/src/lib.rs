//! llm170-diag — 공유 진단 계층.
//!
//! 층 계약: 이 크레이트는 순수 Rust(GPU 의존 0)로, 백엔드가 공급하는
//! **resolved 이벤트**(시간 계산 완료)를 표현·저장·분석한다.
//! 의존 방향: `diag ← core, backend-gpu, server`.

pub mod dump;
pub mod fb;
pub mod flag;
pub mod fp;
pub mod span;
pub mod watchdog;

pub use fp::{fp_diff, fp_record};

/// [tuning-sweep W3 2026-10-16] profile feature(계측) 빌드 판정 — ab-bench의
/// 이진 strings 검사와 동일 조건을 인프로세스로. bench --sweep이 측정을 거부한다
/// (실측 2026-10-15: 계측 포함 빌드로 청크 효과를 오독할 뻔).
pub const PROFILE_BUILD: bool = cfg!(feature = "profile");
