//! GPU 백엔드 — **rawcuda 단일** (2026-10-08: 단일 트랙 CUDA W4A16).
//!
//! `rawcuda`: CUDA 드라이버 API 수동 바인딩(ffi) + 컨텍스트(ctx) + 커널 자산
//! (assets/*.cu, fatbin). 커널 산술 계약은 core 미러(`dot_row_w4a16_lane` 등)가
//! 판정 기준 — W2에서 W4A16 커널·호스트를 이 크레이트에 얹는다.

pub mod rawcuda;

pub use rawcuda::gptq4::Gptq4;
pub use rawcuda::gptq4::h2f;
pub use rawcuda::w4a16_dec::{AttnDims, GdnDims, W4a16Dec, f32_to_f16};

/// VRAM 프로브 — 가드 preflight가 사용(미측정 시 B17 거부).
pub fn cuda_mem_free() -> Option<(u64, u64)> {
    rawcuda::ctx::cuda_mem_free()
}
