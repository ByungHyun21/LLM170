//! rawvk — Vulkan 백엔드 (plans/12). 모듈 루트.

pub mod checks;
pub use checks::{gdn_check, smoke_test};
pub mod context;
pub mod decoder;
pub mod flashcheck;
pub mod vkacc;


const SMOKE_SPV: &[u8] = include_bytes!("spv/smoke.spv");
const COOPMAT_PROBE_SPV: &[u8] = include_bytes!("spv/coopmat_probe.spv");
pub const AXPY_SPV: &[u8] = include_bytes!("spv/axpy_scaled.spv");

/// vk-check — 디바이스 역량 + 트리비얼 컴퓨트 값 검증 (M1 게이트).
/// subsum-check — 서브그룹 리덕션 프로브 (xor 트리 / add / broadcast).
pub fn subsum_check() -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
    let mut ctx = VkCtx::new()?;
    let ob = ctx.alloc(16)?;
    let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(include_bytes!("spv/subsum.spv"), 1, 4)?;
    ctx.bind_bufs(ds, &[ob.buf]);
    for mode in 0..3 {
        let push = (mode as u32).to_le_bytes().to_vec();
        // gy=128 (gdn_ar과 동일 그리드 형상) — WG 수 의존성 검출
        ctx.run(pl, ds, pipe, &push, 1, 128, 1)?;
    }
    let mut r = vec![0f32; 3];
    unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const f32, r.as_mut_ptr(), 3) };
    Ok(format!(
        "subgroup: xor_tree={} (기대 496) add={} broadcast={}",
        r[0], r[1], r[2]
    ))
}

