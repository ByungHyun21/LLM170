//! alloc — GPU 메모리 할당 원장 (plans/86 §5 진단).
//!
//! `LLM170_DUMP=alloc` 으로 켠다: 백엔드의 모든 (해제 없는) 버퍼 할당을
//! 사이트 태그별로 기록해 누적·타임라인을 찍는다. 대형 컨텍스트에서
//! GTT/VRAM 예산이 어느 사이트에 소진되는지 직접 관측이 목적이다
//! (RADV의 OOM 덤프는 크기·도메인만 알려준다).
//!
//! 계층 계약: 이 크레이트는 이벤트 표현만 한다 — 기록 호출은 백엔드가
//! 넣는다. 꺼져 있으면 원자 판독 1회로 제로 코스트.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ON: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct Ledger {
    /// site → (할당 횟수, 누적 바이트)
    sites: HashMap<&'static str, (u64, u64)>,
    total: u64,
}

static LEDGER: Mutex<Option<Ledger>> = Mutex::new(None);

/// 진단 게이트 — 덤프 옵션 파싱 후 백엔드 초기화 시 1회 호출.
pub fn set_on(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// 할당 1건 기록 + 타임라인 라인. 꺼져 있으면 no-op.
pub fn record(site: &'static str, bytes: usize) {
    if !on() {
        return;
    }
    let mut g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    let l = g.get_or_insert_with(Ledger::default);
    let e = l.sites.entry(site).or_insert((0, 0));
    e.0 += 1;
    e.1 += bytes as u64;
    l.total += bytes as u64;
    let (cnt, sum) = *e;
    drop(g);
    eprintln!(
        "[alloc] +{:<10} {site:<14} n={cnt:<5} site_sum={:<12} total={:.2} GiB",
        format_bytes(bytes as u64),
        format_bytes(sum),
        l_total_gib(),
    );
}

/// 사이트별 요약 — 실패 경로 등에서 호출.
pub fn report() {
    if !on() {
        return;
    }
    let g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(l) = g.as_ref() else { return };
    let mut rows: Vec<(&&'static str, &(u64, u64))> = l.sites.iter().collect();
    rows.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
    eprintln!("[alloc] ─ 사이트별 누적 (총 {} ─ {})", format_bytes(l.total), l.sites.len());
    for (site, (cnt, sum)) in rows {
        eprintln!("[alloc]   {site:<14} n={cnt:<5} {}", format_bytes(*sum));
    }
}

fn l_total_gib() -> f64 {
    let g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().map(|l| l.total as f64 / (1 << 30) as f64).unwrap_or(0.0)
}

fn format_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}
