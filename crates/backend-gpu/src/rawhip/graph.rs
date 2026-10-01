//! 그래프 캡처/재생 — plans/115 D(원장 152): 프리필 dispatch 고정비 ~500ms
//! (≈4,400런치 × ~114µs) 제거. Strata layer.cpp:893 준거 — capacity/청크
//! 고정 인자로 캡처, 청크마다 재캡처(인자가 pos 의존). 캡처 중 동기 호출
//! (sync·d2h_wait·ktr_ev)는 ctx.capturing 가드로 건너뛴다.

use crate::rawhip::hip;
use crate::rawhip::{RawCtx, ck};

/// 캡처 시작 — Relaxed 모드(캡처 불가 연산이면 그냥 통과시켜 폭탄 회피).
pub fn capture_begin(ctx: &RawCtx) -> Result<(), String> {
    unsafe {
        ck(
            hip::hipStreamBeginCapture(
                ctx.stream,
                hip::hipStreamCaptureMode_hipStreamCaptureModeRelaxed,
            ),
            "pf-cap-begin",
        )?;
    }
    ctx.capturing
        .store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// 캡처 종료 → 즉시 인스턴스화 → 발행. 그래프/실행체는 호출부가 소유.
pub fn capture_end_and_launch(ctx: &RawCtx) -> Result<(), String> {
    ctx.capturing
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let mut graph: hip::hipGraph_t = std::ptr::null_mut();
    unsafe {
        ck(
            hip::hipStreamEndCapture(ctx.stream, &mut graph),
            "pf-cap-end",
        )?;
        if graph.is_null() {
            return Err("pf-cap: 그래프 null".into());
        }
        let mut exec: hip::hipGraphExec_t = std::ptr::null_mut();
        ck(
            hip::hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ),
            "pf-cap-inst",
        )?;
        // 발행 — 그래프 노드들이 스트림에서 실행된다.
        ck(hip::hipGraphLaunch(exec, ctx.stream), "pf-cap-launch")?;
        // 파괴는 발행 뒤에도 안전(실행체는 발행 후 수명 유지, ROCm가 완료 전
        // 파괴 시 블록한다 — 여기서는 즉시 파괴하면 직렬화하므로 유지했다가
        // 다음 캡처 시작 시 파괴한다).
        PENDING.with(|p| {
            let mut g = p.borrow_mut();
            if let Some(old) = g.take() {
                _ = hip::hipGraphExecDestroy(old);
            }
            *g = Some(exec);
            // 그래프 원본은 인스턴스화 후 불필요.
            _ = hip::hipGraphDestroy(graph);
        });
    }
    Ok(())
}

thread_local! {
    /// 직전 청크의 실행체 — 다음 캡처 때 파괴(발행 완료 보장 후).
    static PENDING: std::cell::RefCell<Option<hip::hipGraphExec_t>> =
        const { std::cell::RefCell::new(None) };
}

/// 호스트 왕복(d2h/h2d) 마커 — 캡처 중 no-op.
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
