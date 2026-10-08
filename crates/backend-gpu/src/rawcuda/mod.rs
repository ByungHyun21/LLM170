//! 원시 CUDA 실행기 — NVIDIA 드라이버 API 직접 경로 (plans/124 2026-10-04).
//!
//! **[2026-10-08 방향 전환]** hip/vulkan·EXL3·GGUF·DS4·FN 탈락(plans/w4a16-cuda.md §5):
//! EXL3 디코더·DS4/FN 체인 모듈을 삭제하고, **W4A16 상주 디코더(W3)가 재사용할
//! 커널 호스트만 존치**한다 — attn(어텐션)·gdn(conv/scan/gate)·norm(rms)·
//! ew_argmax(silu/mul/argmax)·q4(GGUF 값경로 — 호스트 패턴 템플릿)·q4acc(어댑터).
//! 각 호스트의 kernel-facing 계약은 파일 머리 주석(원장) 참조.
//!
//! dead_code 허용 — 호스트 모듈은 소비자(W3 디코더) 배선 전까지 미판독으로 남는다
//! (cuda_probe_shim의 동일 allow 관례).
#![allow(dead_code)]
//!
//! 계층(3층 분리, plans/124 §5):
//! - ffi:  드라이버 API 수동 바인딩(런타임 해석 — 기본 빌드 녹색 계약)
//! - ctx:  CudaCtx 디바이스 컨텍스트(할당·복사·런치·동기화)
//! - q4_cuda/q4_cuda_probe: Q4 GGUF 값경로 커널 호스트(호스트 스테이징 패턴)
//!
//! assets/: .cu가 계약 소스, .fatbin이 빌드 자산(소스·자산 함께 커밋).
//! exl3_attn/exl3_gdn/exl3_norm/exl3_ew는 **커널 자산으로 존치** — 가중치
//! 비의존(어텐션·GDN·rms·EW의 활성 연산)이라 W4A16 상주 디코더(W3)가 재사용
//! 예정. 호스트 impl은 구 Exl3CudaDecoder 확장 파일이었으므로 삭제됨(W3에서
//! 새 디코더 골격에 재배선).

pub mod ctx;
pub mod ffi;
pub mod q4_cuda;
pub mod q4_cuda_probe;
// q4acc_cuda는 Accelerator 어댑터로 크레이트 의존(llm170_core·parking_lot)
// 을 갖는다 — rawcuda 단독 컴파일 계약(cuda_probe_shim, std 전용)에서는
// 제외한다. 셔임 빌드는 `rustc --cfg cuda_probe_shim`(verify_cuda.sh).
#[cfg(not(cuda_probe_shim))]
pub mod q4acc_cuda;
