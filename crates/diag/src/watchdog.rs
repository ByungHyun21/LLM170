
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

static HEARTBEAT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn bump() {
    HEARTBEAT.fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone)]
pub struct OpMark {
    pub tag: &'static str,
    pub at: Instant,
}

static RING: Mutex<Option<VecDeque<OpMark>>> = Mutex::new(None);
static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static REPORTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

pub fn reports() -> u64 {
    REPORTS.load(Ordering::Relaxed)
}

pub fn record_op(tag: &'static str) {
    if !on() {
        return;
    }
    let mut g = RING.lock().unwrap_or_else(|e| e.into_inner());
    let q = g.get_or_insert_with(VecDeque::new);
    q.push_back(OpMark {
        tag,
        at: Instant::now(),
    });
    while q.len() > 32 {
        q.pop_front();
    }
}

pub fn spawn(sec: u64, fail: bool) {
    ON.store(true, Ordering::Relaxed);
    let period = Duration::from_secs(sec.clamp(1, 3600));
    std::thread::spawn(move || {
        let mut last = HEARTBEAT.load(Ordering::Relaxed);
        let mut since = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let cur = HEARTBEAT.load(Ordering::Relaxed);
            if cur != last {
                last = cur;
                since = Instant::now();
                continue;
            }
            if since.elapsed() >= period {
                REPORTS.fetch_add(1, Ordering::Relaxed);
                let ring: Vec<OpMark> = RING
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .map(|q| q.iter().rev().take(8).cloned().collect())
                    .unwrap_or_default();
                eprintln!(
                    "[watchdog] 스텔 {}s — 마지막 op: {}",
                    since.elapsed().as_secs(),
                    ring.first()
                        .map(|m| m.tag)
                        .unwrap_or("(기록 없음 — 로드/CPU 단계)")
                );
                for (i, m) in ring.iter().enumerate() {
                    eprintln!(
                        "[watchdog]   역순[{i}] {} (+{:.1}s)",
                        m.tag,
                        since.elapsed().as_secs_f32() - m.at.elapsed().as_secs_f32()
                    );
                }
                if fail {
                    eprintln!("[watchdog] FAIL 모드 — 프로세스 강제 종료");
                    std::process::exit(137);
                }
                since = Instant::now();
            }
        }
    });
}
