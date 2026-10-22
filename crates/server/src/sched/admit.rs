
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub const RESERVE_SLACK: usize = llm170_backend_gpu::SPEC_VERIFY_TMAX - 1;

pub fn need_tokens(prompt: usize, n_predict: usize) -> usize {
    prompt + n_predict.max(1) + RESERVE_SLACK
}

pub fn binpack_order(
    needs: &[usize],
    free_blocks: usize,
    freeze: bool,
    mut blocks_needed: impl FnMut(usize) -> usize,
) -> Vec<usize> {
    let mut free = free_blocks;
    let mut left: Vec<usize> = (0..needs.len()).collect();
    let mut order = Vec::new();
    loop {
        let mut progressed = false;
        let mut k = 0;
        while k < left.len() {
            let idx = left[k];
            if freeze && idx != 0 {
                k += 1;
                continue;
            }
            let nb = blocks_needed(needs[idx]);
            if nb <= free {
                free -= nb;
                order.push(idx);
                left.remove(k);
                progressed = true;
                if freeze {
                    return order;
                }
            } else {
                k += 1;
            }
        }
        if !progressed {
            break;
        }
    }
    order
}

pub fn predict_start_delay(
    need_blocks: usize,
    free_blocks: usize,
    running: &[(usize, usize)],
    tps: f64,
) -> f64 {
    let mut avail = free_blocks;
    if avail >= need_blocks {
        return 0.0;
    }
    let tps = tps.max(1e-6);
    let mut r: Vec<(f64, usize)> = running.iter().map(|(t, b)| (*t as f64 / tps, *b)).collect();
    r.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (t, held) in r {
        avail += held;
        if avail >= need_blocks {
            return t;
        }
    }
    f64::INFINITY
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AgingStats {
    pub frozen: bool,
    pub d_s: f64,
    pub freeze_events: u64,
    pub freeze_ms: u64,
    pub blocked_jobs: u64,
    pub blocked_tokens: u64,
    pub head_wait_p95_s: f64,
}

pub struct Aging {
    d_s: f64,
    fixed: Option<f64>,
    boot_d_s: f64,
    beta: f64,
    durations: VecDeque<f64>,
    head_waits: VecDeque<f64>,
    frozen: bool,
    freeze_events: u64,
    freeze_ms: u64,
    blocked_jobs: u64,
    blocked_tokens: u64,
    win_freeze_ms: u64,
    win_idle_ms: u64,
    win_blocked_tokens: u64,
    last_adapt: Instant,
}

impl Aging {
    pub fn new(boot_d_s: f64, fixed_d_s: Option<f64>) -> Self {
        let d = fixed_d_s.unwrap_or(boot_d_s);
        Aging {
            d_s: d,
            fixed: fixed_d_s,
            boot_d_s,
            beta: 1.0,
            durations: VecDeque::new(),
            head_waits: VecDeque::new(),
            frozen: false,
            freeze_events: 0,
            freeze_ms: 0,
            blocked_jobs: 0,
            blocked_tokens: 0,
            win_freeze_ms: 0,
            win_idle_ms: 0,
            win_blocked_tokens: 0,
            last_adapt: Instant::now(),
        }
    }

    pub fn decide(
        &mut self,
        dt_ms: u64,
        head_need_blocks: usize,
        free_blocks: usize,
        running: &[(usize, usize)],
        tps: f64,
    ) -> bool {
        if self.frozen {
            self.freeze_ms += dt_ms;
            self.win_freeze_ms += dt_ms;
            if free_blocks > 0 {
                self.win_idle_ms += dt_ms;
            }
        }
        if tps <= 0.0 {
            self.frozen = false;
            return false;
        }
        let delay = predict_start_delay(head_need_blocks, free_blocks, running, tps);
        let freeze = delay > self.d_s;
        if freeze && !self.frozen {
            self.freeze_events += 1;
            eprintln!(
                "# aging: 동결 시작 — 선두 예상 입장 {delay:.1}s > D {:.1}s (자유 {free_blocks}블록)",
                self.d_s
            );
        }
        self.frozen = freeze;
        freeze
    }

    pub fn idle(&mut self) {
        self.frozen = false;
    }

    pub fn note_head_wait(&mut self, wait_s: f64) {
        if wait_s.is_finite() && wait_s >= 0.0 {
            push_ring(&mut self.head_waits, wait_s);
        }
    }

    pub fn observe_est(&mut self, dur_s: f64) {
        if dur_s.is_finite() && dur_s > 0.0 {
            push_ring(&mut self.durations, dur_s);
        }
    }

    pub fn note_blocked(&mut self, tokens: usize) {
        self.blocked_jobs += 1;
        self.blocked_tokens += tokens as u64;
        self.win_blocked_tokens += tokens as u64;
    }

    pub fn maybe_adapt(&mut self, now: Instant) -> Option<String> {
        if self.fixed.is_some() {
            return None;
        }
        if now.duration_since(self.last_adapt) < Duration::from_secs(30) {
            return None;
        }
        self.last_adapt = now;
        let old = self.d_s;
        let base = if self.durations.len() >= 16 {
            let mut v: Vec<f64> = self.durations.iter().copied().collect();
            v.sort_by(f64::total_cmp);
            (self.beta * v[v.len() / 2]).clamp(10.0, 120.0)
        } else {
            self.boot_d_s
        };
        let mut d = base;
        let idle_ratio = if self.win_freeze_ms > 0 {
            self.win_idle_ms as f64 / self.win_freeze_ms as f64
        } else {
            0.0
        };
        if self.win_blocked_tokens > 0 && idle_ratio > 0.6 {
            d = (d * 0.5).max(5.0);
        } else if self.head_waits.len() >= 16
            && self.head_wait_p95().map(|p| p > 2.4 * old).unwrap_or(false)
        {
            d = (d * 1.5).min(120.0);
        }
        self.d_s = d;
        self.win_freeze_ms = 0;
        self.win_idle_ms = 0;
        self.win_blocked_tokens = 0;
        if (d - old).abs() / old.max(1e-9) >= 0.05 {
            Some(format!(
                "D {old:.1}s → {d:.1}s (동결 {}회·차단 {}tok·표본 지속 {}·대기 {})",
                self.freeze_events,
                self.blocked_tokens,
                self.durations.len(),
                self.head_waits.len()
            ))
        } else {
            None
        }
    }

    fn head_wait_p95(&self) -> Option<f64> {
        if self.head_waits.is_empty() {
            return None;
        }
        let mut v: Vec<f64> = self.head_waits.iter().copied().collect();
        v.sort_by(f64::total_cmp);
        Some(v[(v.len() - 1) * 95 / 100])
    }

    pub fn stats(&self) -> AgingStats {
        AgingStats {
            frozen: self.frozen,
            d_s: self.d_s,
            freeze_events: self.freeze_events,
            freeze_ms: self.freeze_ms,
            blocked_jobs: self.blocked_jobs,
            blocked_tokens: self.blocked_tokens,
            head_wait_p95_s: self.head_wait_p95().unwrap_or(0.0),
        }
    }
}

fn push_ring(r: &mut VecDeque<f64>, v: f64) {
    if r.len() >= 64 {
        r.pop_front();
    }
    r.push_back(v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn need_tokens_slack_contract() {
        assert_eq!(need_tokens(100, 16), 100 + 16 + 8);
        assert_eq!(need_tokens(10, 0), 10 + 1 + 8, "생성 0 = 최소 1 토큰");
    }

    #[test]
    fn binpack_examples() {
        let bn = |t: usize| t.div_ceil(512);
        let needs = [100_000usize, 200_000, 50_000];
        assert_eq!(binpack_order(&needs, 294, false, bn), vec![0, 2]);
        let needs2 = [200_000usize, 100_000, 50_000];
        assert_eq!(binpack_order(&needs2, 489, false, bn), vec![0, 2]);
        assert!(binpack_order(&needs, 10, false, bn).is_empty());
    }

    #[test]
    fn binpack_freeze_head_only() {
        let bn = |t: usize| t.div_ceil(512);
        let needs = [100_000usize, 200_000, 50_000];
        assert_eq!(binpack_order(&needs, 294, true, bn), vec![0]);
        assert!(binpack_order(&needs, 100, true, bn).is_empty());
    }

    #[test]
    fn predict_delay_accumulates() {
        let running = [(50usize, 70usize), (100, 30)];
        let d = predict_start_delay(100, 40, &running, 10.0);
        assert!((d - 5.0).abs() < 1e-9, "5초(50토큰/10tps=5s)에 도달: {d}");
        assert_eq!(predict_start_delay(30, 40, &running, 10.0), 0.0);
        assert!(predict_start_delay(1000, 0, &running, 10.0).is_infinite());
    }

    #[test]
    fn aging_adapts() {
        let mut a = Aging::new(30.0, None);
        for _ in 0..20 {
            a.observe_est(60.0);
        }
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let msg = a.maybe_adapt(Instant::now());
        assert!(msg.is_some());
        assert!((a.stats().d_s - 60.0).abs() < 1e-6);
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 800;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 30.0).abs() < 1e-6, "{:?}", a.stats());
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 800;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 30.0).abs() < 1e-6);
    }

    #[test]
    fn aging_hysteresis_margin() {
        let mut a = Aging::new(30.0, None);
        for _ in 0..20 {
            a.observe_est(60.0);
        }
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 550;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 60.0).abs() < 1e-6, "{:?}", a.stats());
        for _ in 0..20 {
            a.note_head_wait(132.0);
        }
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 60.0).abs() < 1e-6, "{:?}", a.stats());
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 650;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 30.0).abs() < 1e-6, "{:?}", a.stats());
    }

    #[test]
    fn aging_clamps() {
        let mut a = Aging::new(30.0, None);
        for _ in 0..20 {
            a.observe_est(10_000.0);
        }
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 120.0).abs() < 1e-6);
        let mut f = Aging::new(30.0, Some(45.0));
        for _ in 0..20 {
            f.observe_est(999.0);
        }
        f.last_adapt = Instant::now() - Duration::from_secs(31);
        assert!(f.maybe_adapt(Instant::now()).is_none());
        assert!((f.stats().d_s - 45.0).abs() < 1e-6);
    }

    #[test]
    fn aging_decide_gates() {
        let mut a = Aging::new(30.0, None);
        let running = [(50usize, 150usize)];
        assert!(!a.decide(100, 100, 0, &running, 10.0));
        let mut b = Aging::new(30.0, Some(1.0));
        assert!(b.decide(100, 100, 0, &running, 10.0));
        assert!(b.stats().frozen);
        assert_eq!(b.stats().freeze_events, 1);
        assert!(!b.decide(100, 100, 0, &running, 0.0));
        assert!(!b.stats().frozen);
    }
}
