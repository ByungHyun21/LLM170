
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemAdvice {
    Hold,
    PreferKv,
    PreferCache,
}

impl MemAdvice {
    pub fn name(self) -> &'static str {
        match self {
            MemAdvice::Hold => "hold",
            MemAdvice::PreferKv => "prefer_kv",
            MemAdvice::PreferCache => "prefer_cache",
        }
    }
}

pub struct MemCtl {
    cur: MemAdvice,
    streak: u32,
}

impl MemCtl {
    pub const fn new() -> Self {
        Self {
            cur: MemAdvice::Hold,
            streak: 0,
        }
    }

    pub fn sample(&mut self, offload_ratio: f64, expert_hit_rate: f64) -> MemAdvice {
        let want = if offload_ratio > 0.10 && expert_hit_rate < 0.9 {
            MemAdvice::PreferKv
        } else if offload_ratio < 0.02 && expert_hit_rate > 0.85 {
            MemAdvice::PreferCache
        } else {
            MemAdvice::Hold
        };
        if want == self.cur {
            self.streak = 0;
            return self.cur;
        }
        self.streak += 1;
        if self.streak >= 8 {
            self.cur = want;
            self.streak = 0;
        }
        self.cur
    }

    pub fn current(&self) -> MemAdvice {
        self.cur
    }
}

static MEMCTL: std::sync::Mutex<MemCtl> = std::sync::Mutex::new(MemCtl::new());

pub fn sample(offload_ratio: f64, expert_hit_rate: f64) -> MemAdvice {
    let mut c = MEMCTL.lock().unwrap_or_else(|e| e.into_inner());
    let prev = c.current();
    let cur = c.sample(offload_ratio, expert_hit_rate);
    if prev != cur {
        eprintln!(
            "# memctl: 배치 권고 {} → {} (KV 오프로드 {offload_ratio:.2} · 전문가 히트 {expert_hit_rate:.2})",
            prev.name(),
            cur.name()
        );
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memctl_window_transitions() {
        let mut c = MemCtl::new();
        for i in 1..=7 {
            assert_eq!(c.sample(0.5, 0.5), MemAdvice::Hold, "표본 {i}");
        }
        assert_eq!(c.sample(0.5, 0.5), MemAdvice::PreferKv, "8연속 = 전환");
        assert_eq!(c.sample(0.0, 0.5), MemAdvice::PreferKv);
        assert_eq!(c.sample(0.0, 0.5), MemAdvice::PreferKv);
        let mut c2 = MemCtl::new();
        for _ in 1..=7 {
            assert_eq!(c2.sample(0.0, 0.9), MemAdvice::Hold);
        }
        assert_eq!(c2.sample(0.0, 0.9), MemAdvice::PreferCache);
        let mut d = MemCtl::new();
        for _ in 0..16 {
            d.sample(0.10, 0.9);
        }
        assert_eq!(d.current(), MemAdvice::Hold);
    }
}
