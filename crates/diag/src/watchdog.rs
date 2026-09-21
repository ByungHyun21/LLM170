//! watchdog — 진행 스텔 감시 (plans/87 §2).
//!
//! `LLM170_WATCHDOG=<sec>`: 백엔드 디스패치마다 `alloc::HEARTBEAT`가
//! 진동한다. 별도 스레드가 `<sec>` 이상 무진동이면 op 링의 마지막 태그와
//! 최근 기록을 덤프한다. `LLM170_WATCHDOG_FAIL=1`이면 SIGKILL 로 자결 —
//! 벤치 스크립트가 무한 대기하지 않게 한다.
//!
//! op 링은 `record_op(tag)` 로 채운다(길이 32, Mutex — 디스패치 경로이나
//! 와치독이 켜졌을 때만 기록해 상시 비용을 없앤다).

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::alloc::HEARTBEAT;

#[derive(Clone)]
pub struct OpMark {
    pub tag: &'static str,
    pub at: Instant,
}

static RING: Mutex<Option<VecDeque<OpMark>>> = Mutex::new(None);
static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// 디스패치 1건 기록(와치독 활성 시에만 링에 적립).
pub fn record_op(tag: &'static str) {
    if !on() {
        return;
    }
    let mut g = RING.lock().unwrap_or_else(|e| e.into_inner());
    let q = g.get_or_insert_with(VecDeque::new);
    q.push_back(OpMark { tag, at: Instant::now() });
    while q.len() > 32 {
        q.pop_front();
    }
}

/// 와치독 스폰 — env 파싱은 호출부(main). `sec` 클램프 [1, 3600].
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
                let ring: Vec<OpMark> = RING
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .map(|q| q.iter().rev().take(8).cloned().collect())
                    .unwrap_or_default();
                eprintln!(
                    "[watchdog] 스텔 {}s — 마지막 op: {}",
                    since.elapsed().as_secs(),
                    ring.first().map(|m| m.tag).unwrap_or("(기록 없음 — 로드/CPU 단계)")
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
                // 비 FAIL: 재보고 주기 리셋(스팸 방지) — 스텔 지속 시 1회/period.
                since = Instant::now();
            }
        }
    });
}
