//! 프레임 그래프 캡처/재생 — 스텝 내 호스트 왕복(capture_mark)을 경계로 스트림 캡처를
//! 세그먼트로 끊어 그래프로 굳히고, 재생 시에는 런치 함수가 즉시 반환된다.

use crate::rawhip::hip;
use crate::rawhip::ck;

pub unsafe fn capture_mark(stream: hip::hipStream_t, tag: &str) -> Result<(), String> {
    let dbg = std::env::var_os("LLM170_GRAPH_DEBUG").is_some();
    let mut g = GRAPH.lock().map_err(|e| e.to_string())?;
    match &mut *g {
        GraphMode::Off => Ok(()),
        // 규약: `*_in` = 호스트 왕복 *직전* → 현재 세그먼트 종료(그래프 굳힘).
        //       `*_out` = 왕복 *직후* → 다음 세그먼트 시작. 왕복 구간(d2h/h2d)은
        //       어느 그래프에도 들어가지 않는다(캡처 불가 연산).
        GraphMode::Capture { segs, open } => unsafe {
            if is_end(tag) {
                if *open {
                    let mut gr: hip::hipGraph_t = std::ptr::null_mut();
                    let r = hip::hipStreamEndCapture(stream, &mut gr);
                    if r != hip::hipError_t_hipSuccess {
                        return Err(format!("EndCapture 실패({r:?}) tag={tag} seg={}", segs.len()));
                    }
                    segs.push(gr);
                    *open = false;
                    if dbg {
                        eprintln!("# graph-mark {tag} 종료 (seg {})", segs.len());
                    }
                }
            } else if !*open {
                let r = hip::hipStreamBeginCapture(stream, hip::hipStreamCaptureMode_hipStreamCaptureModeThreadLocal);
                if r != hip::hipError_t_hipSuccess {
                    return Err(format!("BeginCapture 실패({r:?}) tag={tag}"));
                }
                *open = true;
                if dbg {
                    eprintln!("# graph-mark {tag} 시작 (seg {})", segs.len());
                }
            }
            Ok(())
        },
        GraphMode::Replay { execs, idx } => {
            // 세그먼트는 `*_in`(종료) 지점에서 발사된다 — 그 그래프가 직전 구간.
            if is_end(tag) && *idx < execs.len() {
                unsafe { ck(hip::hipGraphLaunch(execs[*idx], stream), "GraphLaunch")?; }
                *idx += 1;
            }
            Ok(())
        }
    }
}

fn is_end(tag: &str) -> bool {
    tag.ends_with("_in")
}

/// 그래프 상태 폐기 — 캡처/재생을 끄고 런치 스킵도 해제한다(경로 전환 시 필수).
pub fn graph_abort() {
    if let Ok(mut g) = GRAPH.lock() {
        *g = GraphMode::Off;
    }
    GRAPH_SKIP.store(false, std::sync::atomic::Ordering::Relaxed);
}

pub unsafe fn graph_replay(on: bool) -> Result<(), String> {
    if on {
        let mut g = GRAPH.lock().map_err(|e| e.to_string())?;
        if let GraphMode::Replay { idx, .. } = &mut *g {
            *idx = 0;
        }
    }
    GRAPH_SKIP.store(on, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

pub unsafe fn graph_capture_end(stream: hip::hipStream_t) -> Result<(), String> {
    let mut g = GRAPH.lock().map_err(|e| e.to_string())?;
    let GraphMode::Capture { segs, open } = &mut *g else {
        return Err("graph_capture_end: 캡처 중이 아님".into());
    };
    let mut segs = std::mem::take(segs);
    if *open {
        unsafe {
            let mut gr: hip::hipGraph_t = std::ptr::null_mut();
            ck(hip::hipStreamEndCapture(stream, &mut gr), "EndCapture")?;
            segs.push(gr);
        }
        *open = false;
    }
    let mut execs = Vec::with_capacity(segs.len());
    for g0 in &segs {
        unsafe {
            let mut ex: hip::hipGraphExec_t = std::ptr::null_mut();
            ck(hip::hipGraphInstantiate(&mut ex, *g0, std::ptr::null_mut(), std::ptr::null_mut(), 0), "GraphInstantiate")?;
            execs.push(ex);
        }
    }
    eprintln!("# graph: 세그먼트 {}개 캡처·인스턴스화", execs.len());
    *g = GraphMode::Replay { execs, idx: 0 };
    Ok(())
}

pub unsafe fn graph_capture_begin(_stream: hip::hipStream_t) -> Result<(), String> {
    *GRAPH.lock().map_err(|e| e.to_string())? =
        GraphMode::Capture { segs: Vec::new(), open: false };
    Ok(())
}

pub fn nolaunch_on() -> bool {
    *NOLAUNCH.get_or_init(|| std::env::var_os("LLM170_NOLAUNCH").is_some())
}

static NOLAUNCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub static GRAPH_SKIP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub static GRAPH: std::sync::Mutex<GraphMode> = std::sync::Mutex::new(GraphMode::Off);

pub enum GraphMode {
    Off,
    Capture { segs: Vec<hip::hipGraph_t>, open: bool },
    Replay { execs: Vec<hip::hipGraphExec_t>, idx: usize },
}

