//! trace — resolved 커널 이벤트 캡처 스토어.
//!
//! 백엔드(hipEvent/VK timestamp)가 시간을 계산해 완료한 이벤트만 받는다.
//! 캡처 게이트는 AtomicBool(Relaxed) — 꺼졌을 때 런치패스 비용 = 원자 1회.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// resolved 이벤트 — 백엔드 어댑터가 채운 완성품.
#[derive(Debug, Clone)]
pub struct Ev {
    pub name: &'static str,
    /// lane(그리드-y 등 병렬 축 식별자)
    pub lane: u32,
    pub dur_ms: f64,
    /// 다음 이벤트까지의 갭(ms) — dump 시 채운다(None = 마지막).
    pub gap_next_ms: Option<f64>,
    /// 시작부터의 순차 시각(ms) — dump 시 채운다.
    pub seq_ms: f64,
}

static CAPTURE: AtomicBool = AtomicBool::new(false);
static DROPPED: AtomicU64 = AtomicU64::new(0);

fn store() -> &'static Mutex<Vec<Ev>> {
    static STORE: std::sync::OnceLock<Mutex<Vec<Ev>>> = std::sync::OnceLock::new();
    STORE.get_or_init(|| Mutex::new(Vec::new()))
}

/// 캡처 시작 — 기존 이벤트 청소.
pub fn capture_begin() {
    CAPTURE.store(true, Ordering::Relaxed);
    DROPPED.store(0, Ordering::Relaxed);
    if let Ok(mut s) = store().lock() {
        s.clear();
    }
}

/// 캡처 종료 — 이후 push는 무시.
pub fn capture_end() {
    CAPTURE.store(false, Ordering::Relaxed);
}

/// 캡처 중 여부 — 런치패스에서 원자 1회 판독.
pub fn capture_on() -> bool {
    CAPTURE.load(Ordering::Relaxed)
}

/// resolved 이벤트 push — 캡처 중이 아니면 원자 1회로 회피.
pub fn push(ev: Ev) {
    if !capture_on() {
        return;
    }
    match store().lock() {
        Ok(mut s) => {
            if s.len() < 200_000 {
                s.push(ev);
            } else {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
        Err(_) => {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// 캡처된 이벤트 소비 + 갭/순차 시각 계산 — dump 시 1회 호출.
/// 갭은 end(k)→start(k+1)이 아니라 **전임자 이름 기준**으로 귀속한다
/// (갭의 주인은 후속 op의 호스트 비용이므로).
pub fn take() -> (Vec<Ev>, u64) {
    capture_end();
    let dropped = DROPPED.load(Ordering::Relaxed);
    let mut evs = match store().lock() {
        Ok(mut s) => std::mem::take(&mut *s),
        Err(_) => Vec::new(),
    };
    // 순차 시각 + 갭
    let mut t_acc = 0.0f64;
    for i in 0..evs.len() {
        evs[i].seq_ms = t_acc;
        t_acc += evs[i].dur_ms;
        if i + 1 < evs.len() {
            evs[i].gap_next_ms = Some(0.0); // TODO: 백엔드가 절대 시각 제공 시 계산
        }
    }
    (evs, dropped)
}

/// 커널명별 집계 — writer가 소비.
pub fn summarize(evs: &[Ev]) -> Vec<(String, u32, f64, u32)> {
    use std::collections::HashMap;
    let mut m: HashMap<(&'static str, u32), (f64, u32)> = HashMap::new();
    for ev in evs {
        let e = m.entry((ev.name, ev.lane)).or_insert((0.0, 0));
        e.0 += ev.dur_ms;
        e.1 += 1;
    }
    let mut v: Vec<_> = m.into_iter()
        .map(|((n, g), (ms, c))| (n.to_string(), g, ms, c))
        .collect();
    v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// 갭 집계 — 전임자 커널명별.
pub fn gap_by_pred(evs: &[Ev]) -> Vec<(String, f64, u32)> {
    use std::collections::HashMap;
    let mut m: HashMap<&'static str, (f64, u32)> = HashMap::new();
    let mut gap_tot = 0.0f64;
    for w in 0..evs.len().saturating_sub(1) {
        let g = evs[w].gap_next_ms.unwrap_or(0.0);
        if g > 0.0 {
            let e = m.entry(evs[w].name).or_insert((0.0, 0));
            e.0 += g;
            e.1 += 1;
            gap_tot += g;
        }
    }
    let _ = gap_tot;
    let mut v: Vec<_> = m.into_iter()
        .map(|(n, (ms, c))| (n.to_string(), ms, c))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_off_push_noop() {
        assert!(!capture_on());
        push(Ev { name: "x", lane: 1, dur_ms: 1.0, gap_next_ms: None, seq_ms: 0.0 });
        let (evs, _) = take();
        assert!(evs.is_empty(), "off 상태 push는 무시돼야 한다");
    }

    #[test]
    fn seq_and_take() {
        capture_begin();
        push(Ev { name: "a", lane: 1, dur_ms: 10.0, gap_next_ms: None, seq_ms: 0.0 });
        push(Ev { name: "b", lane: 2, dur_ms: 5.0, gap_next_ms: None, seq_ms: 0.0 });
        let (evs, dropped) = take();
        assert_eq!(evs.len(), 2);
        assert_eq!(dropped, 0);
        assert_eq!(evs[0].seq_ms, 0.0);
        assert_eq!(evs[1].seq_ms, 10.0);
    }

    #[test]
    fn summarize_sorted() {
        let evs = vec![
            Ev { name: "b", lane: 1, dur_ms: 5.0, gap_next_ms: None, seq_ms: 0.0 },
            Ev { name: "a", lane: 1, dur_ms: 10.0, gap_next_ms: None, seq_ms: 0.0 },
            Ev { name: "a", lane: 1, dur_ms: 10.0, gap_next_ms: None, seq_ms: 0.0 },
        ];
        let s = summarize(&evs);
        assert_eq!(s[0].0, "a");
        assert_eq!(s[0].3, 2); // count
        assert_eq!(s[1].0, "b");
    }
}
