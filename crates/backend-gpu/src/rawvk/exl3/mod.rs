//! EXL3 vk 프로덕션 런타임 (R3, plans/129 — checks/에서 순수 이동).
//! 3층 원칙(B-1): 모듈(여기) / 검증(checks/) / 본체 배선(server/).
//! 검증 하네스(exl3_probes·exl3_bench)는 checks/ 잔류.
pub(crate) mod attn;
pub(crate) mod cpu;
pub(crate) mod decode;
pub(crate) mod frame;
pub(crate) mod gdn;
pub(crate) mod mtp;
pub(crate) mod resident;
pub(crate) mod staging;
#[allow(clippy::module_inception)]
pub(crate) mod util;
pub(crate) mod wire;
