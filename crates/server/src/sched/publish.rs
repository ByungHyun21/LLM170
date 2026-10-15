//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::{SCHED, Slot};
use crate::engine::Engine;
use std::sync::atomic::Ordering;

/// [R7] 모니터링 게시 스로틀(≤1Hz — 최신값만, 핫패스 할당 최소화) —
/// 슬롯 루프에서 순수 이동.
pub(super) fn maybe_publish(
    slots: &[Slot],
    eng: &Engine,
    copy_base: [(u64, u64, u64); 3],
    last_pub: &mut std::time::Instant,
) {
    if last_pub.elapsed() >= std::time::Duration::from_millis(900) {
        *last_pub = std::time::Instant::now();
        publish_slot_views(slots);
        publish_stats(eng, copy_base);
    }
}

/// 모니터링 — 메모리 분류 + 복사 계측 게시(최신값만).
pub(super) fn publish_stats(eng: &Engine, base: [(u64, u64, u64); 3]) {
    let (weights_gpu, kv_gpu, offload_cpu, ple_cpu) = eng.mem_stats();
    crate::metrics::publish_memory(crate::metrics::MemView {
        weights_gpu,
        kv_gpu,
        offload_cpu,
        ple_cpu,
        active_per_token: eng.active_weight_bytes(),
        moe_mode: eng.moe_mode().to_string(),
        moe_cache: eng.mcache_stats(),
    });
    let now = eng.copy_stats();
    let sub = |i: usize| -> (u64, u64, u64) {
        (
            now[i].0.saturating_sub(base[i].0),
            now[i].1.saturating_sub(base[i].1),
            now[i].2.saturating_sub(base[i].2),
        )
    };
    crate::metrics::publish_io(crate::metrics::IoView {
        runtime: [sub(0), sub(1), sub(2)],
        load: base,
    });
}

/// 모니터링 게시 — 슬롯 뷰 + 활성 수(최신값 덮어쓰기, 히스토리 없음).
pub(super) fn publish_slot_views(slots: &[Slot]) {
    let views: Vec<crate::metrics::SlotView> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| crate::metrics::SlotView {
            id: i,
            is_processing: s.job.is_some(),
            n_prompt_tokens: s.job.as_ref().map(|j| j.tokens.len()).unwrap_or(0),
            n_prompt_processed: s.prefilled,
            n_cache: s.cached.len(),
            n_decoded: s.generated as usize,
            err_count: s.err_count,
        })
        .collect();
    SCHED.slots_active.store(
        views.iter().filter(|v| v.is_processing).count() as u64,
        Ordering::Relaxed,
    );
    crate::metrics::publish_slots(views);
}
