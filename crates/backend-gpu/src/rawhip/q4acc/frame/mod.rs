//! q4acc 프레임 — FrameState·FrameHost (활성 상주 디코드, plans/78 R1).

use super::*;

/// GEMM 패밀리 핀 — 프리필(t>1, np 아님)만 타일 강제(원장 18). np 디코드는
/// 언핀(행별 t=1 패밀리 — 원장 124 VERIFY_ROW_PIN 산술, plans/115 P5).
fn prefill_pin(t: usize, np: bool) {
    let pin = t > 1 && !np;
    crate::rawhip::ctx::PREFILL_PIN.store(pin, std::sync::atomic::Ordering::Relaxed);
}

/// plans/115 D: MoE 폴백(비 Q4K/Q5_1) 도달 — 그래프 캡처 호환성 판정.
pub static MOE_FALLBACK_USED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

mod host;
mod state;
