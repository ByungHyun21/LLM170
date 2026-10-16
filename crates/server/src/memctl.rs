//! (2026-10-16) 메모리 배치 권고 — 이산 컨트롤러(순수 로직 + 1Hz 표본).
//!
//! 신호는 전부 기존 계측이다: KV 호스트 오프로드율(호스트로 밀린 KV = 매 토큰
//! PCIe 재독 비용), MoE 전문가 캐시 히트율(스트리밍 미스 = 디스크/PCIe 비용),
//! (서버가 게시하는 큐 대기는 aging·/stats가 별도 소비).
//!
//! **범위(2026-10-16 확정)**: 현재는 **관측·로그 전용**이다. 액추에이터
//! (experts↔KV VRAM 이동)는 KV 아레나 탄력(VMM 청킹 — 2단계 id 매핑)이 선행
//! 조건이고, 트리거 실측(개발기 RTX4090·Gen4: `--kv-mem` 512M→8G에서 전문가
//! 캐시 k 96→78·히트 0.878→0.863·tg 53.9→52.0 t/s = **soft 상충**)에선 이동
//! 이득이 작다. CMP(Gen2 x4) 도착 시 재실측 후 채택/기각 — 설계의 "정적 배분
//! v0 + 트리거" 규율. 그때 이 모듈의 신호·결정이 그대로 액추에이터 입력이 된다.
//!
//! 임계는 **초기값**이다(실측 캘리브레이션 대상): PreferKv = 오프로드율 > 0.10
//! (KV가 호스트로 밀림 — 매-토큰 재독 비용 발생), PreferCache = 오프로드 ≈ 0
//! (0.02 미만) & 히트율 > 0.85(측정 밴드 0.86~0.88). 플래핑 방지 = 8표본(≈8s)
//! 연속 동일 판정에서만 전환(창 자체가 히스테리시스).

/// 배치 권고(이산).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemAdvice {
    /// 중립(기본 — 현행 정적 배분 유지).
    Hold,
    /// KV에 VRAM 우선: 호스트 오프로드 발생 & 전문가 캐시 이득 상대적 낮음.
    PreferKv,
    /// 전문가 핫셋에 VRAM 우선: 오프로드 0 & 캐시 히트율 높음(상각 큼).
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

/// 컨트롤러 상태 — 지속 창(연속 동일 판정) 카운터.
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

    /// 표본 1회 — (KV 호스트 오프로드율, 전문가 캐시 히트율) → 권고.
    /// 전환은 `WINDOW`(8 = 1Hz 8초) 연속 동일 판정에서만.
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

/// 1Hz 표본 진입점(게시 경로) — 전환 시 1줄 로그. 반환 = 현재 권고.
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

    /// 이산 전환 — 8표본 연속에서만(창 = 히스테리시스), 중립 복귀 포함.
    #[test]
    fn memctl_window_transitions() {
        let mut c = MemCtl::new();
        // KV 오프로드 + 낮은 히트 — 7표본까지는 Hold, 8번째에서 전환.
        for i in 1..=7 {
            assert_eq!(c.sample(0.5, 0.5), MemAdvice::Hold, "표본 {i}");
        }
        assert_eq!(c.sample(0.5, 0.5), MemAdvice::PreferKv, "8연속 = 전환");
        // 창이 깨지면(중립 표본) 스트릭 리셋 — 유지.
        assert_eq!(c.sample(0.0, 0.5), MemAdvice::PreferKv);
        assert_eq!(c.sample(0.0, 0.5), MemAdvice::PreferKv);
        // 오프로드 0 + 높은 히트 8연속 → PreferCache(새 컨트롤러 = 창 의미 명확).
        let mut c2 = MemCtl::new();
        for _ in 1..=7 {
            assert_eq!(c2.sample(0.0, 0.9), MemAdvice::Hold);
        }
        assert_eq!(c2.sample(0.0, 0.9), MemAdvice::PreferCache);
        // 경계: 오프로드 0.10 정확히 = PreferKv 아님(초과만), 히트 0.9 = KV 아님.
        let mut d = MemCtl::new();
        for _ in 0..16 {
            d.sample(0.10, 0.9);
        }
        assert_eq!(d.current(), MemAdvice::Hold);
    }
}
