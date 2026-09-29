//! 그래프 캡처/재생 — 폐기(ledger 음성 판정, 107 W2). capture_mark는 호스트
//! 왕복 마커 인터페이스로만 남는다(항상 no-op).

/// 호스트 왕복(d2h/h2d) 마커 — 그래프 미사용 이후 항상 no-op.
///
/// # Safety
/// 호출부는 단일 스텝 스레드에서만 호출한다(과거 그래프 캡처 경계 규약).
pub unsafe fn capture_mark(
    _stream: crate::rawhip::hip::hipStream_t,
    _tag: &str,
) -> Result<(), String> {
    Ok(())
}

pub fn nolaunch_on() -> bool {
    *NOLAUNCH
}

static NOLAUNCH: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("LLM170_NOLAUNCH").is_some());
