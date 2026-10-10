//! 원시 CUDA 실행기 — NVIDIA 드라이버 API 직접 경로 (2026-10-04).
//!
//! **[2026-10-08 정리]** 단일 트랙(CUDA W4A16)으로
//! 재편: 컨텍스트/FFI와 **커널 자산**(활성 연산 — 어텐션·GDN·rms·EW·smoke)만
//! 존치한다. 구 포맷·타깃 스택(구 디코더·DS4·FN 체인)은 삭제됐다.
//! W4A16 커널 호스트는 W2에서 이 디렉터리에 새로 얹는다(자산 .cu가 계약 소스).
//!
//! dead_code 허용 — 커널 자산은 소비자(W2 디코더) 배선 전까지 미판독이다.

pub mod assets;
pub mod ctx;
pub mod ffi;
pub mod gptq4;
pub mod w4a16_dec;
