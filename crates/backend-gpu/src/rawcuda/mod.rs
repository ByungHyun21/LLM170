//! 원시 CUDA 실행기 — NVIDIA 드라이버 API 직접 경로 (plans/124 2026-10-04).
//! 범위: 모듈 단위 구현·검증까지 — 메인 부착·전체 모델 실행 없음(계약 §0).
//! ENV에 따른 계산 경로 분기 없음(계약 위반 금지 항목).
//!
//! dead_code 허용 — G4+ 필드(어텐션·MTP·배치·그래프 버퍼)는 plans/124 §0
//! 범위상 조립 단계(G3+)까지 미판독으로 남는다(2026-10-07 리눅스 포팅 —
//! cuda_probe_shim의 동일 allow 관례를 크레이트 층으로 승격).
#![allow(dead_code)]
//!
//! 계층(3층 분리, plans/124 §5; G10 파일 분할 2026-10-04):
//! - ffi:    드라이버 API 수동 바인딩(런타임 해석 — 기본 빌드 녹색 계약)
//! - ctx:    CudaCtx 디바이스 컨텍스트(할당·복사·런치·동기화)
//! - exl3_cuda:       모듈층 공유 글루(디코더 레지스트리·선형 상주 로드·
//!   파서·fatbin 리졸버)
//! - exl3_cuda_forward: 순차 forward(S5 호스트 스테이징 경로 — 회귀 기준)
//! - exl3_cuda_device: 디바이스 상주 forward(S10 — 서버 기본 경로)
//! - exl3_cuda_device_probe/_bench/_gemv_probe: S10 두 경로 대조 계기
//! - gemv_cuda/gemm2_cuda/norm_cuda/gdn_cuda/attn_cuda/ew_argmax_cuda:
//!   Exl3CudaDecoder 모듈별 임플 블록(G2-G7 분리 파일 —
//!   머리에 용도·정합·속도 원장, plans/129-cuda C2)
//! - exl3_cuda_probe: 검증층 공유 지원(스모크 + 공용 오라클·RNG·판정 +
//!   커널 파일→프루브 매핑 표, plans/129-cuda C5)
//! - *_cuda_probe(모듈별): 모듈 검증층(G10 분할 — 머리에 129 A10 체크리스트,
//!   plans/129-cuda C1)
//! - q4_cuda/q4_cuda_probe(G8)·mtp_cuda/mtp_cuda_probe(G9): 독립 모듈 파일.
//!
//! assets/: .cu가 계약 소스, .fatbin이 빌드 자산(scripts/build_cuda.bat —
//! rawhip co/*.co 미러: 소스·자산 함께 커밋). 기능 게이트 없음 — rawhip과
//! 동일하게 상시 컴파일(드라이버 부재는 CudaCtx::new에서 런타임 Err).

pub mod attn_cuda;
pub mod exl3_cuda_batch;
pub mod exl3_cuda_batch_probe;
pub mod attn_cuda_probe;
pub mod ctx;
pub mod ew_argmax_cuda;
pub mod ew_argmax_cuda_probe;
pub mod exl3_cuda;
pub mod exl3_cuda_device;
pub mod exl3_cuda_device_bench;
pub mod exl3_cuda_device_probe;
pub mod exl3_cuda_gemv_probe;
pub mod exl3_cuda_forward;
pub mod exl3_cuda_probe;
pub mod ffi;
pub mod fn_gdn_cuda;
pub mod fn_gdn_cuda_probe;
pub mod fn_hc_cuda;
pub mod fn_moe_cuda;
pub mod fn_ple_cuda;
pub mod fn_qsa_cuda;
pub mod fn_support;
pub mod gdn_cuda;
pub mod gdn_cuda_probe;
pub mod gemm2_cuda;
pub mod gemm2_cuda_probe;
pub mod gemv_cuda;
pub mod gemv_cuda_probe;
pub mod hc_cuda;
pub mod hc_cuda_probe;
pub mod moe_cuda;
pub mod moe_cuda_probe;
pub mod mtp_cuda;
pub mod mtp_cuda_probe;
pub mod norm_cuda;
pub mod norm_cuda_probe;
pub mod ple_cuda;
pub mod ple_cuda_probe;
pub mod q4_cuda;
pub mod q4_cuda_probe;

pub mod ds4_attn_cuda;
pub mod ds4_attn_cuda_probe;
pub mod ds4_hc_cuda;
pub mod ds4_hc_cuda_probe;
pub mod ds4_moe_cuda;
pub mod ds4_moe_cuda_probe;
pub mod ds4_mtp_cuda;
pub mod ds4_mtp_cuda_probe;
pub mod fn_chain_cuda_probe;
pub mod mtp_fn_cuda;
pub mod mtp_fn_cuda_probe;
pub mod qsa_cuda;
pub mod qsa_cuda_probe;
