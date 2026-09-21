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
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

static ON: AtomicBool = AtomicBool::new(false);

/// plans/87 §1 — 버퍼 VA 로깅 게이트(LLM170_DUMP=vaddr). 켜면 할당마다
/// tsv 한 줄을 append 로 남겨 폴트 주소 매칭에 쓴다.
static VADDR: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct Ledger {
    /// site → (할당 횟수, 누적 바이트, 풀 반납 바이트)
    sites: HashMap<&'static str, (u64, u64, u64)>,
    total: u64,
    /// 최초 기록 시점의 커널 카운터(미추적 산출 기준).
    base_counters: Option<(u64, u64)>,
    /// 증분 tsv(개방된 파일) — vaddr 게이트.
    tsv: Option<std::fs::File>,
    seq: u64,
}

static LEDGER: Mutex<Option<Ledger>> = Mutex::new(None);

/// 진단 게이트 — 덤프 옵션 파싱 후 백엔드 초기화 시 1회 호출.
pub fn set_on(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

/// VA 로깅 게이트(plans/87 §1).
pub fn set_vaddr(v: bool) {
    VADDR.store(v, Ordering::Relaxed);
}

pub fn vaddr_on() -> bool {
    VADDR.load(Ordering::Relaxed)
}

/// 진행 심박(plans/87 §2) — 백엔드 디스패치마다 증가. 와치독이 감시한다.
pub static HEARTBEAT: AtomicU64 = AtomicU64::new(0);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// plans/87 §1 — VA 원장 항목(폴트 매처 데이터).
pub struct VaEntry {
    pub site: &'static str,
    pub bytes: u64,
    pub va: u64,
    pub va_end: u64,
    pub seq: u64,
}

/// 할당 1건 기록 + 타임라인 라인. 꺼져 있으면 no-op.
/// va != 0 이면 vaddr 모드에서 tsv 로도 남긴다.
pub fn record(site: &'static str, bytes: usize) {
    record_va(site, bytes, 0, 0);
}

pub fn record_va(site: &'static str, bytes: usize, va: u64, va_end: u64) {
    if !on() {
        return;
    }
    let mut g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    let l = g.get_or_insert_with(Ledger::default);
    if l.base_counters.is_none() {
        l.base_counters = Some(drm_counters());
    }
    l.seq += 1;
    let e = l.sites.entry(site).or_insert((0, 0, 0));
    e.0 += 1;
    e.1 += bytes as u64;
    l.total += bytes as u64;
    let (cnt, sum, _) = *e;
    let seq = l.seq;
    if l.tsv.is_none() && vaddr_on() {
        l.tsv = std::fs::File::create(format!("/tmp/llm170-alloc-{}.tsv", std::process::id())).ok();
    }
    if let Some(t) = l.tsv.as_mut() {
        let _ = writeln!(t, "{site}\t{bytes}\t{va:#x}\t{va_end:#x}\t{seq}");
    }
    drop(g);
    eprintln!(
        "[alloc] +{:<10} {site:<14} n={cnt:<5} site_sum={:<12} total={:.2} GiB",
        format_bytes(bytes as u64),
        format_bytes(sum),
        l_total_gib(),
    );
}

/// plans/87 §4 — 풀 반납(frame_free) 기록: 사이트별 재사용 가능 재고.
pub fn recycle(site: &'static str, bytes: usize) {
    if !on() {
        return;
    }
    let mut g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = g.as_mut() {
        let e = l.sites.entry(site).or_insert((0, 0, 0));
        e.2 += bytes as u64;
    }
}

/// 이 프로세스의 tsv 경로(vaddr 모드에서 만들어진 경우).
pub fn tsv_path() -> Option<String> {
    let g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().map(|_| format!("/tmp/llm170-alloc-{}.tsv", std::process::id()))
}

/// 사이트별 요약 — 실패 경로 등에서 호출.
pub fn report() {
    if !on() {
        return;
    }
    let g = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(l) = g.as_ref() else { return };
    let mut rows: Vec<(&&'static str, &(u64, u64, u64))> = l.sites.iter().collect();
    rows.sort_by_key(|(_, (_, b, _))| std::cmp::Reverse(*b));
    eprintln!("[alloc] ─ 사이트별 누적 (총 {} ─ {})", format_bytes(l.total), l.sites.len());
    for (site, (cnt, sum, rec)) in rows {
        eprintln!(
            "[alloc]   {site:<14} n={cnt:<5} {:>10}  풀반납 {}",
            format_bytes(*sum),
            format_bytes(*rec)
        );
    }
    // plans/87 §4 — 커널 카운터 대사: 미추적(RADV 내부+무태그) 분리.
    let (g0, v0) = l.base_counters.unwrap_or((0, 0));
    let (g1, v1) = drm_counters();
    eprintln!(
        "[alloc] ─ 카운터 대사: GTT {}→{} (+{}), VRAM {}→{} (+{}) | 원장 {} | 미추적 GTT {} VRAM {}",
        format_bytes(g0),
        format_bytes(g1),
        format_bytes(g1.saturating_sub(g0)),
        format_bytes(v0),
        format_bytes(v1),
        format_bytes(v1.saturating_sub(v0)),
        format_bytes(l.total),
        format_bytes(g1.saturating_sub(g0).saturating_sub(l.total)),
        format_bytes(v1.saturating_sub(v0).saturating_sub(l.total)),
    );
}

/// amdgpu 커널 카운터 (GTT used, VRAM used) — 진단 경로 전용.
fn drm_counters() -> (u64, u64) {
    let read = |name: &str| -> u64 {
        let mut best = 0u64;
        let Ok(rd) = std::fs::read_dir("/sys/class/drm") else { return 0 };
        for entry in rd.flatten() {
            let p = entry.path().join("device").join(name);
            if let Ok(s) = std::fs::read_to_string(&p)
                && let Ok(v) = s.trim().parse()
            {
                best = best.max(v);
            }
        }
        best
    };
    (read("mem_info_gtt_used"), read("mem_info_vram_used"))
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
