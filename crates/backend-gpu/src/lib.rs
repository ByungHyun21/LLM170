
pub mod rawcuda;

pub use rawcuda::ctx::{DeviceCaps, VramSampler};
pub use rawcuda::gptq4::Gptq4;
pub use rawcuda::gptq4::h2f;
pub use rawcuda::w4a16_dec::{
    AttnDims, AttnLayer, BATCH_DEC_MAX, CTX_OVERFLOW, G4_CHUNK_MAX, GEMM_FFMA_TMAX, GdnDims,
    LayerWindow, SPEC_VERIFY_TMAX, W4a16Dec, f32_to_f16, forward_2shard, forward_2shard_argmax,
    forward_2shard_toks,
};

pub fn cuda_mem_free() -> Option<(u64, u64)> {
    rawcuda::ctx::cuda_mem_free()
}

pub fn cuda_mem_free_dev(ordinal: i32) -> Option<(u64, u64)> {
    rawcuda::ctx::cuda_mem_free_dev(ordinal)
}
