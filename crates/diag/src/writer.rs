//! writer — 텍스트 테이블 출력 (plans/82 §1).
//!
//! ktrace dump 포맷과 항등한 출력을 단위테스트로 핀한다.

use crate::trace::Ev;

/// 커널명별 집계 테이블 — 기존 KTRACE dump 포맷.
pub fn table(evs: &[Ev], dropped: u64) -> String {
    use std::collections::HashMap;
    let mut out = String::new();
    if evs.is_empty() && dropped == 0 {
        return out;
    }

    // 이름·lane별 집계
    let mut sums: HashMap<(&'static str, u32), (f64, u32)> = HashMap::new();
    for ev in evs {
        let e = sums.entry((ev.name, ev.lane)).or_insert((0.0, 0));
        e.0 += ev.dur_ms;
        e.1 += 1;
    }
    let total_ms: f64 = evs.iter().map(|e| e.dur_ms).sum();
    let gap_tot: f64 = evs.iter().filter_map(|e| e.gap_next_ms).sum();
    out.push_str(&format!("TOTAL {:.1}ms GAPS {:.1}ms\n", total_ms, gap_tot));

    let mut v: Vec<_> = sums.iter().collect();
    v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    for ((n, g), (ms, cnt)) in v.iter().take(40) {
        out.push_str(&format!("{:30} gy={:<4} {:9.3}ms x{:4}\n", n, g, ms, cnt));
    }

    // 갭 집계 (전임자 이름별)
    let mut gap_by: HashMap<&'static str, (f64, u32)> = HashMap::new();
    for w in 0..evs.len().saturating_sub(1) {
        let g = evs[w].gap_next_ms.unwrap_or(0.0);
        if g > 0.0 {
            let e = gap_by.entry(evs[w].name).or_insert((0.0, 0));
            e.0 += g;
            e.1 += 1;
        }
    }
    out.push_str(&format!("LAUNCH GAPS total {:.1}ms\n", gap_tot));
    let mut gv: Vec<_> = gap_by.iter().collect();
    gv.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal));
    for (n, (ms, c)) in gv.iter().take(12) {
        out.push_str(&format!("  after {:26} {:8.1}ms x{:4}\n", n, ms, c));
    }

    if dropped > 0 {
        out.push_str(&format!("WARNING: {dropped} events dropped (capacity)\n"));
    }
    out
}

/// 순차 덤프 — `LLM170_KTRACE_SEQ=N`.
pub fn seq_dump(evs: &[Ev], max: usize) -> String {
    let mut out = String::new();
    for (k, ev) in evs.iter().enumerate().take(max) {
        out.push_str(&format!(
            "# seq {k:4} {:<28} gy={:<6} {:8.3}ms\n",
            ev.name, ev.lane, ev.dur_ms
        ));
    }
    out
}


/// 통합 덤프 — 집계 테이블 + 갭 + (LLM170_KTRACE_SEQ=N 지정시) 순차 목록.
/// 백엔드 어댑터(ktrace 등)의 단일 호출 프론트엔드.
pub fn dump(evs: &[Ev], dropped: u64) -> String {
    let mut out = table(evs, dropped);
    if let Ok(v) = std::env::var("LLM170_KTRACE_SEQ") {
        let n: usize = v.parse().unwrap_or(64);
        out.push_str(&seq_dump(evs, n));
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::Ev;

    fn ev(name: &'static str, lane: u32, dur: f64, gap: Option<f64>) -> Ev {
        Ev { name, lane, start_ms: 0.0, dur_ms: dur, gap_next_ms: gap, seq_ms: 0.0 }
    }

    #[test]
    fn table_format() {
        let evs = vec![
            ev("kernel_a", 1, 10.5, Some(2.0)),
            ev("kernel_b", 2, 5.0, None),
            ev("kernel_a", 1, 10.5, Some(2.0)),
        ];
        let s = table(&evs, 0);
        assert!(s.contains("TOTAL 26.0ms GAPS 4.0ms"), "합계/갡: {s}");
        assert!(s.contains("kernel_a"), "커널명 포함: {s}");
        assert!(s.contains("x   2"), "카운트: {s}");
    }

    #[test]
    fn empty_table() {
        assert_eq!(table(&[], 0), "");
    }

    #[test]
    fn dropped_warning() {
        let s = table(&[], 42);
        assert!(s.contains("42 events dropped"));
    }

    #[test]
    fn seq_dump_format() {
        let evs = vec![ev("k1", 4, 1.5, None)];
        let s = seq_dump(&evs, 64);
        assert!(s.contains("# seq    0 k1                           gy=4         1.500ms"));
    }
}
