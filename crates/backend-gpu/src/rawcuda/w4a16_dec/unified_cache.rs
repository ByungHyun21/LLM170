//! [§3.4 ② 2026-10-12] 통합 메모리 네임스페이스 — 회계+정책 레이어(호스트 전용).
//!
//! K6 트랜셰 2(블록 테이블 전면 간접화, 1038912e) 이후 엔진에는 4개의
//! 독립 정책 캐시가 있다: KV(dkc/dvc/dksc/dvsc) · K3 체크포인트(dchk —
//! 512경계 슬롯당 4환형) · 스펙 검증 상태(dsave/dsnap) · 전문가 LRU
//! (mcache_q/s). 이 모듈은 네 캐시를 단일 네임스페이스로 묶어 **VRAM을
//! utility로 배분하는 런타임 통화**의 회계 기반을 제공한다.
//!
//! 범위(트랜셰 1): 레지스트리는 **부기만 한다** — 실제 메모리 이동/해제는
//! 하지 않는다. 할당 지점은 이 레지스트리에 등록·해제하며, 예산 초과는
//! 로그(enforce=false 국면). 트랜셰 2가 dchk/dsave 등을 레지스트리 조회
//! 하에 두고 enforce를 켜면 evict_until_fits가 실제 해제를 지시하게 된다.
//!
//! 정책: 예산 초과 시 같은 티어에서 **유효 utility(기준 utility × recency
//! 감쇠) 최저** 엔트리부터 퇴거. 동률은 오래된 것(last_used 낮은 것),
//! 그다음 id 사전순 — 판정 완전 결정론(테스트 동일성 보장).

use std::collections::HashMap;

/// 메모리 티어 — 현재 캐시 전부 HBM. HostRam은 전문가 호스트 스테이징
/// (host_stage)·KV 넘침 등 트랜셰 2 이후 이관 대상.
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

/// 캐시 범주 — 통합 네임스페이스의 4원 분류.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheCategory {
    /// KV 캐시(슬롯별 연속 — 블록 테이블 간접). 디코드 필수.
    Kv,
    /// K3 접두 재사용 체크포인트(dchk) — 재계산 가능한 TTFT 최적화.
    Checkpoint,
    /// 스펙 검증 상태 저장(dsave/dsnap) — 스펙 라운드 직결.
    SpecState,
    /// 35B 스트리밍 전문가 LRU 캐시(mcache_q/s) — 스트리밍 t/s 직결.
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

/// 네임스페이스 엔트리 식별자 — 문자열 이름(로그 가독성 계약.
/// 예: "attn:kv" · "spec:ckpt" · "moe:expert-lru").
pub type EntryId = String;

/// 기준 utility 권고 — 할당 지점이 상황에 맞게 조정할 수 있는 출발점.
/// 채택 사유(2026-10-12): KV 퇴거는 디코드 자체를 죽인다(최고) · 전문가
/// 캐시는 스트리밍 t/s 직결 · 스펙 상태는 스펙 가속 상실(기능 유지) ·
/// 체크포인트는 재계산 가능한 순수 최적화(최저 — 예산 압박 시 1순위 퇴거).
pub mod base_util {
    pub const KV: f64 = 100.0;
    pub const EXPERT_LRU: f64 = 50.0;
    pub const SPEC_STATE: f64 = 20.0;
    pub const CKPT: f64 = 10.0;
}

/// 단일 네임스페이스 엔트리.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub id: EntryId,
    pub category: CacheCategory,
    pub tier: Tier,
    pub bytes: u64,
    /// 기준 utility(호출자 지정, 유한값 계약). 유효 utility는
    /// `utility × recency 감쇠` — recency는 touch마다 갱신.
    pub utility: f64,
    /// 마지막 사용 tick(레지스트리 내부 단조 카운터 — wall-clock 아님).
    pub last_used: u64,
}

/// 통합 캐시 레지스트리 — 엔트리 부기·티어별 예산·utility 퇴거 정책.
///
/// `enforce=false`(로그 전용 국면): 예산 초과 할당도 등록하고 경고
/// 카운터만 올린다(퇴거 없음). `enforce=true`: 예산 확보를 위해 같은
/// 티어에서 유효 utility 최저 순 퇴거 — 퇴거분은 `take_evicted()`로
/// 호출자가 소비한다(트랜셰 2: 실제 해제·이관).
pub struct UnifiedCacheRegistry {
    entries: HashMap<EntryId, CacheEntry>,
    totals: [u64; 2],
    budgets: [u64; 2],
    tick: u64,
    enforce: bool,
    /// 퇴거 로그(아직 소비 안 된 것) — 트랜셰 2의 실제 해제 지시 원.
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
    /// 예산 무제한·enforce=true(정책 온전) 기본값.
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

    /// 티어 예산 설정. 예산이 현재 점유 이하로 내려가면 enforce 국면에서
    /// 즉시 퇴거 스윕을 돈다(런타임 VRAM 압박 반영 — 예: 타 프로세스 점유).
    pub fn set_budget(&mut self, tier: Tier, bytes: u64) {
        let t = tier.idx();
        self.budgets[t] = bytes;
        if self.enforce {
            while self.totals[t] > self.budgets[t] {
                match self.evict_lowest(tier) {
                    Some(_) => {}
                    None => break, // 엔트리 소진 — 점유 0이므로 사실상 도달 불가
                }
            }
        } else if self.totals[t] > self.budgets[t] {
            self.over_budget_events += 1;
        }
    }

    /// 예산 강제 토글 — false = 로그 전용(트랜셰 1 국면).
    pub fn set_enforce(&mut self, on: bool) {
        self.enforce = on;
    }

    /// 엔트리 등록(=통화 지출). 예산 부족 시 같은 티어에서 유효 utility
    /// 최저 순 퇴거로 확보한다. 실패 = 예산보다 큰 단일 청구 또는 중복 id.
    /// enforce=false 국면에서는 예산 초과도 등록(경고 카운트) — 실패는
    /// 중복 id뿐.
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

    /// 엔트리 해제(=통화 환수). 부재 id는 Err(부기 정합성 가드).
    pub fn release(&mut self, id: &str) -> Result<(), String> {
        let Some(e) = self.entries.remove(id) else {
            return Err(format!("ucache: 해제 대상 없음 \"{id}\""));
        };
        let t = e.tier.idx();
        self.totals[t] = self.totals[t].saturating_sub(e.bytes);
        Ok(())
    }

    /// 재등록 경로 관용 해제 — 부재 시 무동작·false(할당 성공 후에만
    /// 등록되는 항목의 "구것 해제 → 새것 할당" 순서와 정합).
    pub fn release_if_present(&mut self, id: &str) -> bool {
        self.release(id).is_ok()
    }

    /// 사용 갱신 — LRU tick 진행 + 유효 utility 재계산(감쇠 리셋: 유효
    /// utility가 기준값으로 회귀). 트랜셰 2 퇴거 판정의 recency 원천.
    pub fn touch(&mut self, id: &str) -> Result<(), String> {
        let Some(e) = self.entries.get_mut(id) else {
            return Err(format!("ucache: touch 대상 없음 \"{id}\""));
        };
        self.tick += 1;
        e.last_used = self.tick;
        Ok(())
    }

    /// 유효 utility = 기준 × 1/(1+age). age=0이면 기준값 그대로.
    /// 퇴거 우선순위(오름차순)의 단일 척도.
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

    /// 로그 전용 국면의 예산 초과 발생 수(진단 — /stats 노출 후보).
    pub fn over_budget_events(&self) -> u64 {
        self.over_budget_events
    }

    /// 퇴거분 소비(트랜셰 2: 여기 받은 엔트리의 실제 메모리를 해제·이관).
    pub fn take_evicted(&mut self) -> Vec<CacheEntry> {
        std::mem::take(&mut self.evicted)
    }

    /// 등록 직후 진단 로그 1행(할당 지점이 eprintln로 출력).
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

    /// 같은 티어에서 유효 utility 최저 엔트리 1개 퇴거(등록부에서만).
    /// 동률: age 큰 것(오래된 것) → id 사전순. 대상 없음 = None.
    fn evict_lowest(&mut self, tier: Tier) -> Option<CacheEntry> {
        let tick = self.tick;
        let mut victim: Option<(f64, u64, EntryId)> = None; // (유효util, age, id)
        for e in self.entries.values() {
            if e.tier != tier {
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

    /// bytes 확보까지 퇴거 반복. 전부 퇴거해도 부족하면 Err(단일 청구가
    /// 예산 초과 — 로그 전용 국면이 아니면 기능 단위 축소가 호출자 책임).
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

    /// 핵심 시나리오: 예산 압박 하 최저 utility 퇴거 — B(u1)가 나가고
    /// A(u3)·C(u2)만 남는다.
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

    /// 동일 utility면 최근 사용(touch)한 쪽이 살고 오래된 쪽이 퇴거.
    #[test]
    fn recency_beats_equal_utility() {
        let mut r = reg();
        r.allocate("A", CacheCategory::Kv, Tier::Hbm, 50, 1.0)
            .unwrap();
        r.allocate("B", CacheCategory::Kv, Tier::Hbm, 50, 1.0)
            .unwrap();
        r.touch("A").unwrap(); // A 최신 → B가 더 낡음
        assert!(r.utility_effective("B").unwrap() < r.utility_effective("A").unwrap());
        r.allocate("C", CacheCategory::Kv, Tier::Hbm, 10, 9.0)
            .unwrap();
        assert!(r.entry("B").is_none(), "같은 utility 중 오래된 B 퇴거");
        assert!(r.entry("A").is_some());
    }

    /// touch의 유효 utility 재계산 — 감쇠 리셋 확인.
    #[test]
    fn touch_resets_recency_decay() {
        let mut r = UnifiedCacheRegistry::new();
        r.allocate("a", CacheCategory::Kv, Tier::Hbm, 1, 8.0)
            .unwrap();
        // 진행 tick으로 age 증가 → 유효 utility 감소.
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

    /// 예산 축소 스윕 — 낮은 utility부터 내쫓아 예산 이하로.
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

    /// 예산보다 큰 단일 청구 — 빈 레지스트리여도 거절.
    #[test]
    fn oversize_request_rejected() {
        let mut r = reg();
        let err = r
            .allocate("big", CacheCategory::Kv, Tier::Hbm, 101, 1.0)
            .unwrap_err();
        assert!(err.contains("예산"), "거절 사유 명시: {err}");
        assert!(r.is_empty() && r.total(Tier::Hbm) == 0);
    }

    /// 로그 전용 국면(enforce=false) — 초과 등록 허용 + 경고 계수,
    /// 퇴거는 일어나지 않는다(트랜셰 2 전 실메모리 보존 계약).
    #[test]
    fn non_enforce_registers_over_budget() {
        let mut r = reg();
        r.set_enforce(false);
        r.allocate("a", CacheCategory::Kv, Tier::Hbm, 60, 1.0)
            .unwrap();
        r.allocate("b", CacheCategory::Kv, Tier::Hbm, 60, 1.0)
            .unwrap(); // 120 > 100
        assert_eq!(r.len(), 2, "초과 등록 유지");
        assert_eq!(r.total(Tier::Hbm), 120);
        assert_eq!(r.over_budget_events(), 1);
        assert_eq!(r.evict_count(), 0, "로그 전용은 퇴거 없음");
    }

    /// 티어 격리 — HBM 압박이 host RAM 엔트리를 건드리지 않는다.
    #[test]
    fn tier_isolation() {
        let mut r = UnifiedCacheRegistry::new();
        r.set_budget(Tier::Hbm, 100);
        r.set_budget(Tier::HostRam, 10);
        r.allocate("host", CacheCategory::ExpertWeights, Tier::HostRam, 10, 1.0)
            .unwrap();
        r.allocate("kv", CacheCategory::Kv, Tier::Hbm, 100, 5.0)
            .unwrap();
        // HBM 예산 100 → 50으로 축소: kv 퇴거, host 무사.
        r.set_budget(Tier::Hbm, 50);
        assert!(r.entry("kv").is_none());
        assert!(r.entry("host").is_some());
        assert_eq!(r.total(Tier::HostRam), 10);
    }
}
