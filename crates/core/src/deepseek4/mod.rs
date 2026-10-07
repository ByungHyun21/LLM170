//! DeepSeek-V4-Flash(0731 계열) CPU 골든 레퍼런스 — plans/130 B2.
//!
//! 본 모듈의 산술은 후속 CUDA 모듈 패밀리(plans/130 B3)의 **수치 계약**이다.
//! 1차 출처: 공식 참조 구현 `inference/model.py`·`kernel.py`(0731 레포 동봉,
//! 논문 §2.3 "모호하지 않은 명세") + 연구 보고서
//! `.omo/evidence/ds4-architecture-report.md`(2026-10-05, 인용 §N).
//!
//! 구조 선례: `crate::qwen4exp`(mod/layers/stages/frame 배치, 순수 함수 스테이지,
//! file:line 인용 주석). 로더 선례: `crate::qwen4exp::Model4`(mmap·오프셋 읽기)를
//! EXL3 safetensors(`llm170_exl3::StArchive` + trellis 참조 디코드)로 이식.
//!
//! dtype 규율(보고서 §9.4): 저장은 전부 f32. 참조 구현의 bf16 경계
//! (`.to(dtype)`/`type_as` 지점)마다 `ops::bf16_round`를 명시적으로 적용하고,
//! QAT 시뮬 지점(FP8 128/64블록, FP4 32블록, 인덱서 Hadamard)은 참조 커널과
//! 동일한 스케일 공식으로 재현한다. 압축·hc·게이트·전문가 FFN·RMSNorm 내부는 fp32.
//!
//! MoE/인덱서 top-k 동점 규칙(보고서 "UNVERIFIED 잔여 (a)"): 값 내림차순,
//! 동점이면 **낮은 인덱스 우선**(안정 정렬) — 실측 패리티는 B4에서 대조.

pub mod config;
pub mod frame;
pub mod layers;
pub mod loader;
pub mod ops;
pub mod stages;

pub use config::{Deepseek4Config, LayerKind};
pub use loader::Ds4Loader;
