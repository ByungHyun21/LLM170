//! 폴트 프로브(plans/107 W4: checks.rs에서 분할).

use crate::rawvk::vkacc::{Slot, VkAcc, push_u32s};

/// plans/87 §1 — 의도적 GPUVM 폴트 프로브: 실제 결함 패턴(디스크립터
/// 오프셋이 버퍼 끝 너머 — pipeline robustness가 주소 자체를 못 구한다)으로
/// 폴트를 유발해 RADV 주소 → va-lookup 체인을 검증한다. DEVICE_LOST가 정상.
pub fn fault_probe() -> Result<String, String> {
    let acc = VkAcc::new()?;
    llm170_diag::alloc::set_on(true);
    llm170_diag::alloc::set_vaddr(true);
    let mut ctx = acc.ctx.lock();
    let b = ctx.alloc_host(4096)?; // 원장 기록(VA 포함)
    let p = acc.pipeline(&mut ctx, Slot::Scale)?;
    let ds = ctx.fresh_ds_for(&p, 1)?;
    // 실효 패턴: 12-바인딩 gemv 파이프라인에 1개만 바인딩 — 미바인딩
    // 디스크립터(3..11)를 커널이 읽는다. 오프셋 초과는 RADV가 빈 범위로
    // 클램프해 폴트가 안 나는 것을 실측했다(정렬 무관).
    let _ = ds;
    let ds2 = ctx.bind_ds(&p, &[b.buf])?;
    let push = push_u32s(&[32u32, 32u32, 8u32, 8u32, 1u32, 1024u32]);
    let r = ctx.run(p.pl, ds2, p.pipe, &push, 1, 1, 1);
    let tsv = llm170_diag::alloc::tsv_path().unwrap_or_else(|| "(없음)".into());
    Ok(format!(
        "발사 결과: {r:?} (Err=DEVICE_LOST 정상) — tsv: {tsv} 에서 RADV 폴트 주소를 va-lookup 하라"
    ))
}
