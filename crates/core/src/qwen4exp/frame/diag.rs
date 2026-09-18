//! frame 진단 자산 — 체크섬·스텝 타이밍·스테이지 스킵 (plans/78 R9 선분리).
//! 청크 불변성 결함 조사(plans/77 §1)의 프로브가 여기에 모인다 —
//! forward 본체(frame/mod.rs)와 분리해 조사 중 편집 충돌을 줄인다.

use super::*;

// 스테이지 동기 마커 (LLM170_FRAME_SYNC=1) — 스티키 폴트의 발생 지점을
// 즉시 드러낸다(폴트는 다음 API 호출에서야 보고된다).
thread_local! {
    /// 스테이지 누적 시간 — (마지막 경계 시각, [(접미사, us, 호출수)]).
    static FT: std::cell::RefCell<(std::time::Instant, Vec<(String, u64, u64)>)> =
        std::cell::RefCell::new((std::time::Instant::now(), Vec::new()));
}

/// 프레임 스테이지 시간 계측 (LLM170_FRAME_TIME=1). sync_mark가 만드는 경계
/// 에서만 측정한다 — 프레임 op는 비동기라 호출 시간만으로는 GPU 시간이 안 나온다.
pub(super) fn ftime_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LLM170_FRAME_TIME").is_some())
}

/// 진단용 스테이지 스킵(LLM170_STAGE_SKIP="qsa,gdn,moe") — 비용 분해 전용.
pub fn stage_skipped(name: &str) -> bool {
    std::env::var("LLM170_STAGE_SKIP")
        .map(|v| v.split(',').any(|x| x.trim() == name))
        .unwrap_or(false)
}

/// 진단용 프레임 체크섬 — `LLM170_DUMP=...,checksum`. np·단일·배치 경로 공용.
/// 버퍼 앞 t·n개를 전부 읽어 합과 행 표본(첫·중간·마지막 행의 첫 원소)을
/// 보고한다. 청크 크기가 다른 두 실행에서 "같은 층·같은 단계·같은 토큰 수"를
/// 맞대어 첫 발산 지점을 찾는 용도 — 합만으로는 상쇄로 가려질 수 있어 행
/// 표본을 함께 낸다. 기본 꺼짐(1회 판독).
pub(super) fn frame_ck(acc: &dyn Accelerator, h: u64, n: usize, t: usize, tag: &str) {
    let o = llm170_diag::dump::opts();
    if !o.checksum && !o.row_on(tag) {
        return;
    }
    let rows_on = o.row_on(tag);
    let mut v = vec![0.0f32; n * t];
    if acc.frame_read(h, &mut v).is_ok() {
        if o.checksum {
            let s: f64 = v.iter().map(|&x| x as f64).sum();
            let mid = v[(t / 2) * n];
            let last = v[(t - 1) * n];
            eprintln!(
                "[npck] {tag} t={t} sum={s:.6} v0={:.6} mid0={mid:.6} last0={last:.6}",
                v[0]
            );
        }
        if rows_on {
            let cap = t.min(720);
            // 행당 4표본(첫·둘째·100번째·끝 원소) — 요소 축 커버리지.
            let e1 = 1.min(n.saturating_sub(1));
            let e2 = 100.min(n.saturating_sub(1));
            let e3 = n.saturating_sub(1);
            let vals: Vec<String> = (0..cap)
                .map(|r| {
                    let b = r * n;
                    format!(
                        "{:08x},{:08x},{:08x},{:08x}",
                        v[b].to_bits(),
                        v[b + e1].to_bits(),
                        v[b + e2].to_bits(),
                        v[b + e3].to_bits()
                    )
                })
                .collect();
            eprintln!("[nprd] {tag} t={t} n={n} {}", vals.join(" "));
        }
        if rows_on && o.row0full {
            let full = n.min(2560);
            let rows_n = t.min(16);
            for r in 0..rows_n {
                let vals: Vec<String> = v[r * n..r * n + full]
                    .iter()
                    .map(|x| format!("{:08x}", x.to_bits()))
                    .collect();
                eprintln!("[npr0] {tag} r={r} n={n} {}", vals.join(" "));
            }
        }
    }
}

/// 진단용 버퍼 FNV 해시 — `LLM170_DUMP=...,bufhash`. 앞 len 원소의 비트를
/// 해시해 한 줄로 찍는다. 층 경계마다 전 버퍼를 찍어 첫 오염 버퍼를
/// 찾는 용도(plans/80 §A — 메모리 결함 추적).
pub(super) fn buf_hash(acc: &dyn Accelerator, h: u64, len: usize, tag: &str) {
    if !llm170_diag::dump::opts().bufhash || len == 0 {
        return;
    }
    let mut v = vec![0.0f32; len];
    if acc.frame_read(h, &mut v).is_ok() {
        let mut x = 0xcbf29ce484222325u64;
        for f in &v {
            x ^= f.to_bits() as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
        eprintln!("[npbh] {tag} len={len} h={x:016x}");
    }
}

pub(super) fn sync_mark(acc: &dyn Accelerator, tag: &str, h: u64) -> Result<(), Q4Error> {
    match (ftime_on(), std::env::var_os("LLM170_FRAME_SYNC").is_some()) {
        (false, false) => return Ok(()),
        (ft, sync) => {
            // 1원소 판독 = 동기 + 폴트 보고 (barrier는 오류를 삼킨다).
            let mut v = [0.0f32; 1];
            acc.frame_read(h, &mut v)
                .map_err(|e| Q4Error::Io(format!("fsync {tag}: {e}")))?;
            if ft {
                FT.with(|s| {
                    let mut s = s.borrow_mut();
                    let dt = s.0.elapsed().as_micros() as u64;
                    s.0 = std::time::Instant::now();
                    let key = tag.rsplit('.').next().unwrap_or(tag).to_string();
                    match s.1.iter_mut().find(|e| e.0 == key) {
                        Some(e) => {
                            e.1 += dt;
                            e.2 += 1;
                        }
                        None => s.1.push((key, dt, 1)),
                    }
                });
            }
            if sync {
                eprintln!("# fsync {tag}");
            }
        }
    }
    Ok(())
}

/// 청크/스텝 단위 리포트 — 누적이 있으면 한 줄 출력 후 초기화.
/// (디코드 t=1도 찍는다: 프리필과 달리 동기 지점이 많아 누적-지연 왜곡이 없다.)
pub(super) fn ftime_report(_t: usize) {
    if !ftime_on() {
        return;
    }
    FT.with(|s| {
        let mut s = s.borrow_mut();
        if !s.1.is_empty() {
            s.1.sort_by_key(|b| std::cmp::Reverse(b.1));
            let mut line = String::from("# frame-time(t) ");
            for (k, us, n) in s.1.iter() {
                line.push_str(&format!("{k}={:.1}ms×{n} ", *us as f64 / 1e3));
            }
            eprintln!("{line}");
            s.1.clear();
        }
        s.0 = std::time::Instant::now();
    });
}

/// 단계 덤프 (LLM170_Q4_DBG=1) — 값 경로와 같은 양을 찍어 대조한다.
pub(super) fn dbg(tag: &str, acc: &dyn Accelerator, h: u64, n: usize) {
    if std::env::var_os("LLM170_Q4_DBG").is_none() {
        return;
    }
    let mut v = vec![0.0f32; n];
    if acc.frame_read(h, &mut v).is_err() {
        return;
    }
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    let mx = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    eprintln!("# fdbg {tag}: sum={s:.6} max={mx:.6} v0..3={:?}", &v[..3.min(n)]);
}
