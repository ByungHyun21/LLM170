//! GPU 백엔드 — 순수 Rust 원시 HIP 실행기 (2026-09-03 cubecl 제거).
//!
//! `rawhip`: hipRTC로 임베디드 HIP C++를 컴파일해 hipModuleLaunchKernel로 실행.
//! 커널 산술은 core 미러(dot_row_w4a8_*_lane)와 토큰당 동일 연산열 — to_bits 검증 게이트.
//! 비트계약: raw-HIP greedy 스트림 ≡ CPU W4A8 참조 엔진 (12+64토큰 교차검증).

pub mod rawhip;
pub mod rawvk;

pub use rawhip::decode::{RawDecoder, inject as inject_rawhip};
pub use rawhip::q4acc::new_acc_with_sources as new_q4_acc_with_sources;
/// plans/86 §6 — 파트 소스 지정판: 대형 가중 업로드가 mmap 폴트(20-180 MB/s)
/// 대신 pread 스테이징(~1.2 GB/s)을 쓴다(hip staged_upload 미러).
pub fn new_q4_acc_vk_with_sources(
    parts: Vec<(usize, usize, std::path::PathBuf)>,
) -> Result<std::sync::Arc<dyn llm170_core::matmul::Accelerator>, String> {
    let acc = rawvk::gemv::VkAcc::new_with_sources(parts)?;
    Ok(std::sync::Arc::new(acc))
}
pub use rawhip::{bw_test, dp4a_test, qk_check, raw_probe};
pub use rawhip::probes::gpu_mem_free;
pub use rawvk::decoder::inject as inject_rawvk;
