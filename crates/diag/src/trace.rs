//! trace — resolved 커널 이벤트 표현·시각 계산 (plans/82 §1, plans/109 P2).
//!
//! 백엔드(hipEvent/VK timestamp)가 시간을 계산해 완료한 이벤트(Ev)를
//! 받는다. 캡처 스토어(capture_begin/push/take)·집계(summarize/gap_by_pred)는
//! 프로덕션 호출 0으로 삭제했다 — 구 백엔드 ktrace는 자체 스토어를 쓰고
//! 집계는 writer::table이 담당한다.

/// resolved 이벤트 — 백엔드 어댑터가 채운 완성품.
#[derive(Debug, Clone)]
pub struct Ev {
    pub name: &'static str,
    /// lane(그리드-y 등 병렬 축 식별자)
    pub lane: u32,
    /// 절대 시작 시각(ms) — 첫 이벤트 기준 0. 어댑터가 채운다.
    pub start_ms: f64,
    pub dur_ms: f64,
    /// 다음 이벤트까지의 갭(ms) — resolve 시 계산(None = 마지막).
    pub gap_next_ms: Option<f64>,
    /// 시작부터의 순차 시각(ms) — resolve 시 계산.
    pub seq_ms: f64,
}

/// 절대 시작 시각 기반 갭/순차 계산 — 어댑터가 start_ms를 채운 뒤 호출.
/// 갭은 end(k)→start(k+1)로 잰 뒤 **전임자 이벤트에 귀속**한다
/// (갭의 주인은 후속 op의 호스트 비용이므로).
pub fn resolve(evs: &mut [Ev]) {
    if evs.is_empty() {
        return;
    }
    let t0 = evs[0].start_ms;
    for i in 0..evs.len() {
        evs[i].seq_ms = evs[i].start_ms - t0;
        if i + 1 < evs.len() {
            let gap = (evs[i + 1].start_ms - (evs[i].start_ms + evs[i].dur_ms)).max(0.0);
            evs[i].gap_next_ms = Some(gap);
        } else {
            evs[i].gap_next_ms = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(name: &'static str, lane: u32, start_ms: f64, dur_ms: f64) -> Ev {
        Ev {
            name,
            lane,
            start_ms,
            dur_ms,
            gap_next_ms: None,
            seq_ms: 0.0,
        }
    }

    #[test]
    fn resolve_seq_and_gap_attribution() {
        let mut evs = vec![ev("a", 1, 0.0, 10.0), ev("b", 2, 12.0, 5.0)];
        resolve(&mut evs);
        assert_eq!(evs[0].seq_ms, 0.0);
        assert_eq!(evs[1].seq_ms, 12.0);
        // 갭: b 시작(12) − (a 시작 0 + dur 10) = 2ms, 전임자 a에 귀속.
        assert_eq!(evs[0].gap_next_ms, Some(2.0));
        assert_eq!(evs[1].gap_next_ms, None, "마지막 이벤트 갭 없음");
    }

    #[test]
    fn resolve_overlap_clamps_to_zero() {
        let mut evs = vec![ev("a", 1, 0.0, 10.0), ev("b", 1, 5.0, 1.0)];
        resolve(&mut evs);
        assert_eq!(evs[0].gap_next_ms, Some(0.0), "겹침은 갭 0으로 클램프");
    }
}
