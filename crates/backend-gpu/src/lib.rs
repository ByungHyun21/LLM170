//! GPU 백엔드 — **rawcuda 단일** (2026-10-08 방향 전환: hip/vulkan 탈락,
//! plans/w4a16-cuda.md §5 · CUDA W4A16 트랙).
//!
//! `rawcuda`: CUDA 드라이버 API 수동 바인딩(ffi) + fatbin 모듈 + GEMV/GEMM
//! 커널 호스트. 커널 산술 계약은 core 미러(dot_row_w4a16_lane 등)가 판정 기준.

pub mod rawcuda;

pub use rawcuda::q4acc_cuda::new_q4_acc_cuda;

/// B6: CUDA 런타임 VRAM 프로브 — 가드 preflight가 사용(미측정 시 B17 거부).
pub fn cuda_mem_free() -> Option<(u64, u64)> {
    rawcuda::ctx::cuda_mem_free()
}
