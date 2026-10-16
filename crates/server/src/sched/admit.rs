//! [2026-10-16] 입장·aging 정책 — 순수 로직 분리(테스트 계약).
//! slot_intake가 호출하고, 규칙(슬랙·bin-pack·EASY 동결·D 적응)은 여기에만 있다.
//!
//! - **슬랙 계약**: 스펙 검증은 전방 t=1+k행(현 최대 9)을 기록한다. 예약이
//!   이 몫을 빼면 블록 경계+풀 포화에서 생성 중 `KvPool::ensure` 실패 창이
//!   열린다(2026-10-16 확인 — 종전 슬랙 2 부족). 슬랙 = 8 = tmax−1.
//! - **bin-pack**: 선착순 차단 금지 — 뒤의 소형이 '지금 들어갈 수 있으면'
//!   입장(스킵어헤드 스캔). 순서 규칙은 `binpack_order`(순수)가 정본.
//! - **aging(EASY 백필식 동결)**: 선두(최장 대기)의 예상 입장 지연이 임계 D를
//!   넘으면 신규 입장을 동결(선두 1건만 허용). 실행 중 요청은 건드리지 않는다
//!   (비선점 — "사용 중 실패 금지"). 선두 입장 시 해제(다음 틱 재판정).
//! - **D 적응**: 부트스트랩 30s → β×P50(요청 예상 지속) 자동 스케일 +
//!   비용 피드백(트리거 히스테리시스 ±20%: 동결 중 유휴 60%+ & 차단>0 = 실손해
//!   → ×0.5 / 선두 대기 P95 > 2.4×D = 보호 부족 → ×1.5), 창 30s·1회·클램프
//!   5~120s. D는 공정성↔스루풋 노브.
//!   `LLM170_AGING_D`(초) 지정 시 고정 D·적응 OFF(운영 노브·A/B).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// 예약 슬랙(토큰) — 스펙 검증 전방 행(tmax−1)을 예약이 덮는다. tmax는 백엔드
/// 단일 출처(`llm170_backend_gpu::SPEC_VERIFY_TMAX` = 1+kmax = 9) — 초안 깊이를
/// 올리면 그 상수만 수정하면 된다.
pub const RESERVE_SLACK: usize = llm170_backend_gpu::SPEC_VERIFY_TMAX - 1;

/// 요청 예약 토큰 = 프롬프트 + 생성한도 + 슬랙.
pub fn need_tokens(prompt: usize, n_predict: usize) -> usize {
    prompt + n_predict.max(1) + RESERVE_SLACK
}

/// bin-pack 입장 순서(순수 모델) — 앞에서부터 '지금 들어가면' 입장, 안 맞으면
/// 스킵어헤드(뒤의 소형이 먼저), 소비 후 재스캔. `freeze` = 선두(원 인덱스 0)
/// 1건만 허용. 반환 = 입장 순서(원 인덱스).
/// 예(512×층 가정 아님 — 테스트는 blocks_needed로 환산): 큐 100k·200k·50k,
/// 여유 150k → [0, 2](200k는 이후). 큐 200k·100k·50k, 여유 250k → [0, 2].
///
/// **소비처 = slot_intake 런타임**(매 반복 첫 후보 선택 — 검토 반영 2026-10-16,
/// 인라인 이원 구현 제거) · 계약 회귀 = 이 모듈 테스트.
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
                    return order; // 동결 중엔 선두 1건만
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

/// 선두 예상 입장 지연(초) — 자유(+유휴 회수) 블록이 충분하면 0, 아니면 실행
/// 중 작업의 완료(점유 해제) 누적으로 필요 블록이 확보되는 시각의 최솟값.
/// `running` = (남은 토큰, 점유 블록), `tps` = 슬롯당 토큰/s.
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

/// aging 관측 스냅샷(게시용).
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

/// EASY 백필식 동결 + D 적응 상태.
pub struct Aging {
    d_s: f64,
    /// 운영자 고정 D(env) — Some면 적응 OFF.
    fixed: Option<f64>,
    boot_d_s: f64,
    beta: f64,
    /// 완료(예상) 지속 표본 — P50 자동 스케일(링 64).
    durations: VecDeque<f64>,
    /// 선두 입장 대기 표본(링 64) — 보호 부족 판정.
    head_waits: VecDeque<f64>,
    frozen: bool,
    freeze_events: u64,
    freeze_ms: u64,
    blocked_jobs: u64,
    blocked_tokens: u64,
    /// 적응 창(30s) 카운터.
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

    /// 틱 판정 — 동결 여부 갱신. `dt_ms` = 직전 틱과의 간격(회계).
    /// `head_need_blocks` = 선두 예약 블록, `free_blocks` = 자유+유휴 회수분.
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
            // 동결 중 여유 블록 > 0 = 백필을 막아 만든 유휴(실손해 신호).
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

    /// 보류열이 빈 틱 — 동결 해제(보호 대상 없음).
    pub fn idle(&mut self) {
        self.frozen = false;
    }

    /// 선두 입장 실측 대기(초).
    pub fn note_head_wait(&mut self, wait_s: f64) {
        if wait_s.is_finite() && wait_s >= 0.0 {
            push_ring(&mut self.head_waits, wait_s);
        }
    }

    /// 입장 시점 예상 지속(초) = 예약 토큰/tps — P50 자동 스케일 입력.
    pub fn observe_est(&mut self, dur_s: f64) {
        if dur_s.is_finite() && dur_s > 0.0 {
            push_ring(&mut self.durations, dur_s);
        }
    }

    /// 동결로 차단된 백필(작업당 1회 호출 — slot_intake).
    pub fn note_blocked(&mut self, tokens: usize) {
        self.blocked_jobs += 1;
        self.blocked_tokens += tokens as u64;
        self.win_blocked_tokens += tokens as u64;
    }

    /// 창(30s)마다 D 적응 — 변경 시 로그 문자열 반환.
    pub fn maybe_adapt(&mut self, now: Instant) -> Option<String> {
        if self.fixed.is_some() {
            return None;
        }
        if now.duration_since(self.last_adapt) < Duration::from_secs(30) {
            return None;
        }
        self.last_adapt = now;
        let old = self.d_s;
        // 1) 자동 스케일 — β×P50(예상 지속). 표본 < 16 = 부트스트랩 유지.
        let base = if self.durations.len() >= 16 {
            let mut v: Vec<f64> = self.durations.iter().copied().collect();
            v.sort_by(f64::total_cmp);
            (self.beta * v[v.len() / 2]).clamp(10.0, 120.0)
        } else {
            self.boot_d_s
        };
        let mut d = base;
        // 2) 비용 피드백(이산 조정 — **트리거 히스테리시스 ±20%**: 명목 임계의
        //    1.2배를 넘어야 발화한다. 창당 1회 — 플래핑 방지, 설계 §10.5).
        let idle_ratio = if self.win_freeze_ms > 0 {
            self.win_idle_ms as f64 / self.win_freeze_ms as f64
        } else {
            0.0
        };
        if self.win_blocked_tokens > 0 && idle_ratio > 0.6 {
            // 동결이 스루풋 실손해(동결 중 유휴 60%+) — 보호 완화.
            d = (d * 0.5).max(5.0);
        } else if self.head_waits.len() >= 16
            && self.head_wait_p95().map(|p| p > 2.4 * old).unwrap_or(false)
        {
            // 선두 대기가 목표(2×D)의 20% 초과 — 보호 강화.
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

    /// 슬랙 계약 — tmax−1 = 8(스펙 검증 전방 행).
    #[test]
    fn need_tokens_slack_contract() {
        assert_eq!(need_tokens(100, 16), 100 + 16 + 8);
        assert_eq!(need_tokens(10, 0), 10 + 1 + 8, "생성 0 = 최소 1 토큰");
    }

    /// bin-pack — 설계 예제 2종(스킵어헤드·소비 후 재스캔).
    /// 블록 환산 = ceil(토큰/512)×1층(단순화 — 순서 규칙만 판정).
    #[test]
    fn binpack_examples() {
        let bn = |t: usize| t.div_ceil(512);
        // 큐 100k·200k·50k, 여유 ~150k(294블록) → 100k(196)+50k(98) = 294.
        let needs = [100_000usize, 200_000, 50_000];
        assert_eq!(binpack_order(&needs, 294, false, bn), vec![0, 2]);
        // 큐 200k·100k·50k, 여유 ~250k(489블록) → 200k(391)+50k(98) = 489.
        let needs2 = [200_000usize, 100_000, 50_000];
        assert_eq!(binpack_order(&needs2, 489, false, bn), vec![0, 2]);
        // 여유 부족 = 빈 순서(전부 대기).
        assert!(binpack_order(&needs, 10, false, bn).is_empty());
    }

    /// 동결 — 선두 1건만. 선두가 안 맞으면 백필도 차단(빈 순서).
    #[test]
    fn binpack_freeze_head_only() {
        let bn = |t: usize| t.div_ceil(512);
        let needs = [100_000usize, 200_000, 50_000];
        // 선두가 맞으면 [0]만(백필 차단).
        assert_eq!(binpack_order(&needs, 294, true, bn), vec![0]);
        // 선두(100k=196블록)가 안 맞으면 전부 대기.
        assert!(binpack_order(&needs, 100, true, bn).is_empty());
    }

    /// 예상 입장 지연 — 자유 충분 = 0, 완료 누적으로 도달, 불가 = ∞.
    #[test]
    fn predict_delay_accumulates() {
        // 자유 40, 필요 100, 실행 (5s 후 70블록)+(10s 후 30블록) → 5초에 110.
        let running = [(50usize, 70usize), (100, 30)]; // (남은 토큰, 블록)
        let d = predict_start_delay(100, 40, &running, 10.0);
        assert!((d - 5.0).abs() < 1e-9, "5초(50토큰/10tps=5s)에 도달: {d}");
        // 자유로 충분 → 0.
        assert_eq!(predict_start_delay(30, 40, &running, 10.0), 0.0);
        // 누적해도 부족 → ∞.
        assert!(predict_start_delay(1000, 0, &running, 10.0).is_infinite());
    }

    /// D 적응 — 자동 스케일(β×P50)·비용 피드백(완화/강화)·클램프.
    #[test]
    fn aging_adapts() {
        // 표본 ≥ 16, P50 = 60s → D 60s.
        let mut a = Aging::new(30.0, None);
        for _ in 0..20 {
            a.observe_est(60.0);
        }
        // 창 경과 강제.
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let msg = a.maybe_adapt(Instant::now());
        assert!(msg.is_some());
        assert!((a.stats().d_s - 60.0).abs() < 1e-6);
        // 동결이 유휴를 만들며 백필을 차단 → ×0.5 = 30s.
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 800;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 30.0).abs() < 1e-6, "{:?}", a.stats());
        // 표본 P50이 60이라 매 창 60 → 피드백도 매 창 적용(조건 지속 시 하강).
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 800;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 30.0).abs() < 1e-6);
    }

    /// [검토 반영 2026-10-16] 트리거 히스테리시스 ±20% — 명목 임계 미달 신호는
    /// 발화하지 않는다(유휴 55%는 ×0.5 미발화, p95 2.2×D는 ×1.5 미발화).
    #[test]
    fn aging_hysteresis_margin() {
        let mut a = Aging::new(30.0, None);
        for _ in 0..20 {
            a.observe_est(60.0); // P50 60s → base 60s
        }
        // 유휴 55%(0.5 초과·0.6 미만) = 데드밴드 — 조정 없음(60s 유지).
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 550;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 60.0).abs() < 1e-6, "{:?}", a.stats());
        // p95 = 2.2×D(2.0 초과·2.4 미만) = 데드밴드 — 조정 없음.
        for _ in 0..20 {
            a.note_head_wait(132.0); // 2.2 × 60
        }
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 60.0).abs() < 1e-6, "{:?}", a.stats());
        // 유휴 65% = 발화(×0.5).
        a.win_blocked_tokens = 1000;
        a.win_freeze_ms = 1000;
        a.win_idle_ms = 650;
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 30.0).abs() < 1e-6, "{:?}", a.stats());
    }

    /// D 적응 — 클램프 경계(자동 상한 120·하한 5).
    #[test]
    fn aging_clamps() {
        let mut a = Aging::new(30.0, None);
        for _ in 0..20 {
            a.observe_est(10_000.0); // P50 초과 → 상한 클램프 120.
        }
        a.last_adapt = Instant::now() - Duration::from_secs(31);
        let _ = a.maybe_adapt(Instant::now());
        assert!((a.stats().d_s - 120.0).abs() < 1e-6);
        // 고정 D = 적응 OFF.
        let mut f = Aging::new(30.0, Some(45.0));
        for _ in 0..20 {
            f.observe_est(999.0);
        }
        f.last_adapt = Instant::now() - Duration::from_secs(31);
        assert!(f.maybe_adapt(Instant::now()).is_none());
        assert!((f.stats().d_s - 45.0).abs() < 1e-6);
    }

    /// decide — 지연 > D = 동결, tps 미상 = 동결 없음(보수).
    #[test]
    fn aging_decide_gates() {
        let mut a = Aging::new(30.0, None);
        let running = [(50usize, 150usize)]; // 5s 후 150블록 해제
        // 지연 5s < 30 → 비동결.
        assert!(!a.decide(100, 100, 0, &running, 10.0));
        // 지연 5s 이지만 D=1(고정) → 동결.
        let mut b = Aging::new(30.0, Some(1.0));
        assert!(b.decide(100, 100, 0, &running, 10.0));
        assert!(b.stats().frozen);
        assert_eq!(b.stats().freeze_events, 1);
        // tps 0 = 예측 불가 → 비동결(동결 오판 방지).
        assert!(!b.decide(100, 100, 0, &running, 0.0));
        assert!(!b.stats().frozen);
    }
}
