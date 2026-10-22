
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Hbm,
    HostRam,
}

impl Tier {
    fn idx(self) -> usize {
        match self {
            Tier::Hbm => 0,
            Tier::HostRam => 1,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Tier::Hbm => "HBM",
            Tier::HostRam => "host",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheCategory {
    Kv,
    Checkpoint,
    SpecState,
    ExpertWeights,
}

impl CacheCategory {
    fn tag(self) -> &'static str {
        match self {
            CacheCategory::Kv => "kv",
            CacheCategory::Checkpoint => "ckpt",
            CacheCategory::SpecState => "spec",
            CacheCategory::ExpertWeights => "expert",
        }
    }
}

pub type EntryId = String;

pub mod base_util {
    pub const KV: f64 = 100.0;
    pub const EXPERT_LRU: f64 = 50.0;
    pub const SPEC_STATE: f64 = 20.0;
    pub const CKPT: f64 = 10.0;
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub id: EntryId,
    pub category: CacheCategory,
    pub tier: Tier,
    pub bytes: u64,
    pub utility: f64,
    pub last_used: u64,
}

pub struct UnifiedCacheRegistry {
    entries: HashMap<EntryId, CacheEntry>,
    totals: [u64; 2],
    budgets: [u64; 2],
    tick: u64,
    enforce: bool,
    evicted: Vec<CacheEntry>,
    evict_count: u64,
    evict_bytes: u64,
    over_budget_events: u64,
}

impl Default for UnifiedCacheRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedCacheRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            totals: [0; 2],
            budgets: [u64::MAX; 2],
            tick: 0,
            enforce: true,
            evicted: Vec::new(),
            evict_count: 0,
            evict_bytes: 0,
            over_budget_events: 0,
        }
    }

    pub fn set_budget(&mut self, tier: Tier, bytes: u64) {
        let t = tier.idx();
        self.budgets[t] = bytes;
        if self.enforce {
            while self.totals[t] > self.budgets[t] {
                match self.evict_lowest(tier) {
                    Some(_) => {}
                    None => break,
                }
            }
        } else if self.totals[t] > self.budgets[t] {
            self.over_budget_events += 1;
        }
    }

    pub fn set_enforce(&mut self, on: bool) {
        self.enforce = on;
    }

    pub fn allocate(
        &mut self,
        id: &str,
        category: CacheCategory,
        tier: Tier,
        bytes: u64,
        utility: f64,
    ) -> Result<EntryId, String> {
        if self.entries.contains_key(id) {
            return Err(format!("ucache: 중복 id \"{id}\" — release 먼저"));
        }
        if !utility.is_finite() {
            return Err(format!("ucache: utility는 유한값이어야 함(\"{id}\")"));
        }
        let t = tier.idx();
        let headroom = self.budgets[t].saturating_sub(self.totals[t]);
        if bytes > headroom {
            if !self.enforce {
                self.over_budget_events += 1;
                eprintln!(
                    "[ucache] 예산 초과 등록(로그 전용): {id}({}) {}MiB > {} 여유MiB",
                    category.tag(),
                    bytes >> 20,
                    headroom >> 20
                );
            } else {
                self.evict_until_fits(tier, bytes)?;
            }
        }
        self.tick += 1;
        let entry = CacheEntry {
            id: id.to_string(),
            category,
            tier,
            bytes,
            utility,
            last_used: self.tick,
        };
        self.totals[t] = self.totals[t].saturating_add(bytes);
        self.entries.insert(id.to_string(), entry);
        Ok(id.to_string())
    }

    pub fn release(&mut self, id: &str) -> Result<(), String> {
        let Some(e) = self.entries.remove(id) else {
            return Err(format!("ucache: 해제 대상 없음 \"{id}\""));
        };
        let t = e.tier.idx();
        self.totals[t] = self.totals[t].saturating_sub(e.bytes);
        Ok(())
    }

    pub fn release_if_present(&mut self, id: &str) -> bool {
        self.release(id).is_ok()
    }

    pub fn touch(&mut self, id: &str) -> Result<(), String> {
        let Some(e) = self.entries.get_mut(id) else {
            return Err(format!("ucache: touch 대상 없음 \"{id}\""));
        };
        self.tick += 1;
        e.last_used = self.tick;
        Ok(())
    }

    pub fn utility_effective(&self, id: &str) -> Option<f64> {
        self.entries.get(id).map(|e| {
            let age = self.tick.saturating_sub(e.last_used) as f64;
            e.utility / (1.0 + age)
        })
    }

    pub fn entry(&self, id: &str) -> Option<&CacheEntry> {
        self.entries.get(id)
    }

    pub fn total(&self, tier: Tier) -> u64 {
        self.totals[tier.idx()]
    }

    pub fn budget(&self, tier: Tier) -> u64 {
        self.budgets[tier.idx()]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn evict_count(&self) -> u64 {
        self.evict_count
    }

    pub fn evict_bytes(&self) -> u64 {
        self.evict_bytes
    }

    pub fn over_budget_events(&self) -> u64 {
        self.over_budget_events
    }

    pub fn take_evicted(&mut self) -> Vec<CacheEntry> {
        std::mem::take(&mut self.evicted)
    }

    pub fn log_line(&self, id: &str) -> String {
        let e = self.entries.get(id);
        let detail = match e {
            Some(en) => format!(
                "{}({}) {}MiB u={}",
                en.id,
                en.category.tag(),
                en.bytes >> 20,
                en.utility
            ),
            None => format!("{id} (미등록)"),
        };
        let hbm = if self.budgets[Tier::Hbm.idx()] == u64::MAX {
            format!("{}MiB/∞", self.totals[Tier::Hbm.idx()] >> 20)
        } else {
            format!(
                "{}/{}MiB",
                self.totals[Tier::Hbm.idx()] >> 20,
                self.budgets[Tier::Hbm.idx()] >> 20
            )
        };
        format!(
            "[ucache] {detail} · HBM {hbm} · {}엔트리{}",
            self.entries.len(),
            if self.enforce {
                ""
            } else {
                " · 예산 미강제(트랜셰 1)"
            }
        )
    }

    fn evict_lowest(&mut self, tier: Tier) -> Option<CacheEntry> {
        let tick = self.tick;
        let mut victim: Option<(f64, u64, EntryId)> = None;
        for e in self.entries.values() {
            if e.tier != tier || e.category == CacheCategory::Kv {
                continue;
            }
            let age = tick.saturating_sub(e.last_used);
            let eff = e.utility / (1.0 + age as f64);
            let replace = match &victim {
                None => true,
                Some(v) => eff < v.0 || (eff == v.0 && (age > v.1 || (age == v.1 && e.id < v.2))),
            };
            if replace {
                victim = Some((eff, age, e.id.clone()));
            }
        }
        let (_, _, vid) = victim?;
        let e = self
            .entries
            .remove(&vid)
            .expect("victim은 방금 스캔한 엔트리");
        let t = tier.idx();
        self.totals[t] = self.totals[t].saturating_sub(e.bytes);
        self.evict_count += 1;
        self.evict_bytes += e.bytes;
        self.evicted.push(e.clone());
        Some(e)
    }

    fn evict_until_fits(&mut self, tier: Tier, bytes: u64) -> Result<(), String> {
        let t = tier.idx();
        while bytes > self.budgets[t].saturating_sub(self.totals[t]) {
            if self.evict_lowest(tier).is_none() {
                let b = self.budgets[t];
                return Err(format!(
                    "ucache: 예산 초과 — 청구 {}MiB > {} 예산 {}MiB",
                    bytes >> 20,
                    tier.tag(),
                    b >> 20
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> UnifiedCacheRegistry {
        let mut r = UnifiedCacheRegistry::new();
        r.set_budget(Tier::Hbm, 100);
        r
    }

    #[test]
    fn allocate_release_accounting() {
        let mut r = reg();
        r.allocate("a", CacheCategory::Kv, Tier::Hbm, 60, base_util::KV)
            .unwrap();
        r.allocate("b", CacheCategory::ExpertWeights, Tier::Hbm, 40, 1.0)
            .unwrap();
        assert_eq!(r.total(Tier::Hbm), 100);
        assert_eq!(r.len(), 2);
        r.release("a").unwrap();
        assert_eq!(r.total(Tier::Hbm), 40);
        assert!(r.release("a").is_err(), "이중 해제는 부기 오류");
        assert_eq!(r.total(Tier::Hbm), 40, "실패 해제는 점유 불변");
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut r = reg();
        r.allocate("a", CacheCategory::Kv, Tier::Hbm, 10, 1.0)
            .unwrap();
        assert!(
            r.allocate("a", CacheCategory::Kv, Tier::Hbm, 10, 1.0)
                .is_err()
        );
    }

    #[test]
    fn non_finite_utility_rejected() {
        let mut r = reg();
        assert!(
            r.allocate("a", CacheCategory::Kv, Tier::Hbm, 10, f64::NAN)
                .is_err()
        );
        assert!(r.is_empty());
    }

    #[test]
    fn eviction_under_budget_pressure() {
        let mut r = reg();
        r.allocate("A", CacheCategory::Kv, Tier::Hbm, 60, 3.0)
            .unwrap();
        r.allocate("B", CacheCategory::Checkpoint, Tier::Hbm, 40, 1.0)
            .unwrap();
        r.allocate("C", CacheCategory::SpecState, Tier::Hbm, 30, 2.0)
            .unwrap();
        assert!(r.entry("B").is_none(), "최저 utility B가 퇴거");
        assert!(r.entry("A").is_some() && r.entry("C").is_some());
        assert_eq!(r.total(Tier::Hbm), 90);
        assert_eq!(r.evict_count(), 1);
        assert_eq!(r.evict_bytes(), 40);
        let ev = r.take_evicted();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].id, "B");
        assert!(r.take_evicted().is_empty(), "take는 소비 후 소진");
    }

    #[test]
    fn recency_beats_equal_utility() {
        let mut r = reg();
        r.allocate("A", CacheCategory::ExpertWeights, Tier::Hbm, 50, 1.0)
            .unwrap();
        r.allocate("B", CacheCategory::ExpertWeights, Tier::Hbm, 50, 1.0)
            .unwrap();
        r.touch("A").unwrap();
        assert!(r.utility_effective("B").unwrap() < r.utility_effective("A").unwrap());
        r.allocate("C", CacheCategory::ExpertWeights, Tier::Hbm, 10, 9.0)
            .unwrap();
        assert!(r.entry("B").is_none(), "같은 utility 중 오래된 B 퇴거");
        assert!(r.entry("A").is_some());
    }

    #[test]
    fn touch_resets_recency_decay() {
        let mut r = UnifiedCacheRegistry::new();
        r.allocate("a", CacheCategory::Kv, Tier::Hbm, 1, 8.0)
            .unwrap();
        r.allocate("b", CacheCategory::Kv, Tier::Hbm, 1, 1.0)
            .unwrap();
        r.allocate("c", CacheCategory::Kv, Tier::Hbm, 1, 1.0)
            .unwrap();
        let decayed = r.utility_effective("a").unwrap();
        assert!(decayed < 8.0, "age>0이면 기준값 미만");
        r.touch("a").unwrap();
        assert_eq!(r.utility_effective("a").unwrap(), 8.0, "touch 후 기준 회귀");
        assert!(r.touch("없음").is_err());
    }

    #[test]
    fn set_budget_shrink_evicts() {
        let mut r = UnifiedCacheRegistry::new();
        r.allocate("kv", CacheCategory::Kv, Tier::Hbm, 600, base_util::KV)
            .unwrap();
        r.allocate(
            "ckpt",
            CacheCategory::Checkpoint,
            Tier::Hbm,
            400,
            base_util::CKPT,
        )
        .unwrap();
        r.set_budget(Tier::Hbm, 700);
        assert!(r.entry("ckpt").is_none(), "최저 utility 퇴거로 예산 확보");
        assert!(r.entry("kv").is_some());
        assert_eq!(r.total(Tier::Hbm), 600);
    }

    #[test]
    fn oversize_request_rejected() {
        let mut r = reg();
        let err = r
            .allocate("big", CacheCategory::Kv, Tier::Hbm, 101, 1.0)
            .unwrap_err();
        assert!(err.contains("예산"), "거절 사유 명시: {err}");
        assert!(r.is_empty() && r.total(Tier::Hbm) == 0);
    }

    #[test]
    fn non_enforce_registers_over_budget() {
        let mut r = reg();
        r.set_enforce(false);
        r.allocate("a", CacheCategory::Kv, Tier::Hbm, 60, 1.0)
            .unwrap();
        r.allocate("b", CacheCategory::Kv, Tier::Hbm, 60, 1.0)
            .unwrap();
        assert_eq!(r.len(), 2, "초과 등록 유지");
        assert_eq!(r.total(Tier::Hbm), 120);
        assert_eq!(r.over_budget_events(), 1);
        assert_eq!(r.evict_count(), 0, "로그 전용은 퇴거 없음");
    }

    #[test]
    fn tier_isolation() {
        let mut r = UnifiedCacheRegistry::new();
        r.set_budget(Tier::Hbm, 100);
        r.set_budget(Tier::HostRam, 10);
        r.allocate("host", CacheCategory::ExpertWeights, Tier::HostRam, 10, 1.0)
            .unwrap();
        r.allocate("gpu", CacheCategory::ExpertWeights, Tier::Hbm, 100, 5.0)
            .unwrap();
        r.set_budget(Tier::Hbm, 50);
        assert!(r.entry("gpu").is_none());
        assert!(r.entry("host").is_some());
        assert_eq!(r.total(Tier::HostRam), 10);
    }

    #[test]
    fn kv_pinned_survives_pressure() {
        let mut r = UnifiedCacheRegistry::new();
        r.allocate("kv", CacheCategory::Kv, Tier::Hbm, 600, base_util::KV)
            .unwrap();
        r.allocate(
            "ckpt",
            CacheCategory::Checkpoint,
            Tier::Hbm,
            100,
            base_util::CKPT,
        )
        .unwrap();
        r.allocate(
            "spec",
            CacheCategory::SpecState,
            Tier::Hbm,
            100,
            base_util::SPEC_STATE,
        )
        .unwrap();
        r.set_budget(Tier::Hbm, 500);
        assert!(r.entry("kv").is_some(), "KV는 절대 퇴거 불가(고정)");
        assert!(r.entry("ckpt").is_none() && r.entry("spec").is_none());
        assert_eq!(r.total(Tier::Hbm), 600, "소진 후 과점유 허용");
        assert_eq!(r.evict_count(), 2);
        let ev = r.take_evicted();
        assert_eq!(
            ev.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["ckpt", "spec"],
            "utility 오름차순 퇴거(10 → 20)"
        );
    }

    #[test]
    fn oversize_with_only_pinned_left() {
        let mut r = reg();
        r.allocate("kv", CacheCategory::Kv, Tier::Hbm, 60, base_util::KV)
            .unwrap();
        let err = r
            .allocate("big", CacheCategory::ExpertWeights, Tier::Hbm, 50, 1.0)
            .unwrap_err();
        assert!(err.contains("예산"), "거절 사유 명시: {err}");
        assert!(r.entry("big").is_none());
    }
}
