//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::SCHED;
use std::sync::atomic::{AtomicU64, Ordering};

// ── [2.A-④ 2026-10-13] 런타임 경제성 게이트 ─────────────────────────────
// 라운드 실측(ms/방출토큰) vs 평범 스텝 실측(ms/토큰) EMA로 3택
// (mtp/ngram/평범)을 자동 선택한다 — 스펙은 무손실(검증 정본)이라 스위칭은
// 처리량만 바꾼다(토큰 불변). 히스테리시스(±2%) + OFF 후 주기 재프로브,
// k는 {2,4,6,8} 밴드 EMA(3% 여유·32라운드 탐사). 판정은 순수 함수(테스트 표면).

/// k 밴드 — 인덱스 ↔ 초안 폭.
const ECON_KS: [usize; 4] = [2, 4, 6, 8];
/// 스펙/평범 히스테리시스 여유(±2%).
const ECON_MARGIN: f64 = 0.02;
/// k 전환 여유(3%).
const ECON_K_MARGIN: f64 = 0.03;
/// OFF 후 재프로브 주기(토큰).
/// [P5 2026-10-15 512→128 단축] 재프로브가 요청 프리앰블(구조적 저수용)에만
/// 착륙해 복사/반복 구간 표본이 영영 안 들어오는 교착 실측 — 128이면 요청
/// 내 복사 구간에 도달한다. 산문 비용 = OFF 중 1라운드/128토큰(≈0.27ms/tok).
const ECON_PROBE_TOKENS: u64 = 128;

/// EMA 갱신 — 초기 표본은 1/(n+1)(빠른 수렴), 이후 1/8. x = ms/토큰.
fn econ_ema(v: &AtomicU64, n: &AtomicU64, x: f64) {
    let k = n.fetch_add(1, Ordering::Relaxed) + 1;
    let a = 1.0 / k.min(8) as f64;
    let old = f64::from_bits(v.load(Ordering::Relaxed));
    v.store((old + a * (x - old)).to_bits(), Ordering::Relaxed);
}

fn econ_get(v: &AtomicU64, n: &AtomicU64) -> Option<f64> {
    (n.load(Ordering::Relaxed) > 0).then(|| f64::from_bits(v.load(Ordering::Relaxed)))
}

fn econ_kidx(k: usize) -> usize {
    match k {
        2 => 0,
        6 => 2,
        8 => 3,
        _ => 1,
    }
}

/// 평범 표본 — (ms, 방출 토큰 수).
pub(super) fn econ_plain_sample(ms: f64, toks: usize) {
    if toks > 0 {
        econ_ema(&SCHED.econ_plain, &SCHED.econ_plain_n, ms / toks as f64);
    }
}

/// [2.A-④] MTP 초안 유지 여부 — 게이트가 스펙 ON·현 소스 MTP일 때만.
/// OFF/ngram 소스에서는 유지비(체인 ~2.5ms/토큰 수준)를 내지 않는다 —
/// 재선택 시 해당 라운드의 post_round가 초안을 재구축(복귀 지연 1라운드).
pub(super) fn econ_mtp_maintain() -> bool {
    SCHED.econ_on.load(Ordering::Relaxed) == 1 && SCHED.econ_src_cur.load(Ordering::Relaxed) == 0
}

/// 스펙 라운드 표본 — (소스, 초안 폭, 수용 수, 라운드 ms).
/// 비용/토큰 = 라운드 ms ÷ 방출 토큰(수용 접두 + 보정/첫 토큰 1).
pub(super) fn econ_spec_sample(src: u8, k: usize, acc: usize, ms: f64) {
    let per_tok = ms / (acc + 1) as f64;
    // [P5 2026-10-15 재프로브 재부트스트랩] OFF 상태에서 온 재프로브 라운드가
    // 확실히 우수(평범 대비 2% 여유 하회)하면 해당 소스 EMA를 그 표본으로
    // 리셋(n=1). 근거(실측): 복사 워크로드에서 초반 저수용 표본(acc 1·0)이
    // EMA를 오염시키면 OFF→512토큰 재프로브가 1표본/주기로만 들어와 가중치
    // 1/8로는 EMA가 회복 불가 — 좋은 재프로브를 봐도 사실상 영구 OFF(라운드
    // 4↔47 요동의 원인). 레짐 전환 감지용이며 ON/OFF 판정 자체(±2% 히스테리시스)는
    // 불변 — 산문에서 나쁜 재프로브는 EMA를 낮추지 못해 OFF 유지. 마커 소비 =
    // swap(0) — 재프로브 여부는 econ_plan이 표시(econ_on은 재프로브 전에 1로
    // 저장되므로 상태로 판별 불가 — 실측).
    let probe = SCHED.econ_probe_mark.swap(0, Ordering::Relaxed) == 1;
    if probe {
        let plain = econ_get(&SCHED.econ_plain, &SCHED.econ_plain_n);
        if let Some(p) = plain
            && per_tok < p * (1.0 - ECON_MARGIN)
        {
            SCHED.econ_src[src as usize].store(per_tok.to_bits(), Ordering::Relaxed);
            SCHED.econ_src_n[src as usize].store(1, Ordering::Relaxed);
            let i = src as usize * 4 + econ_kidx(k);
            SCHED.econ_k[i].store(per_tok.to_bits(), Ordering::Relaxed);
            SCHED.econ_k_n[i].store(1, Ordering::Relaxed);
            SCHED.econ_rounds.fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
    econ_ema(
        &SCHED.econ_src[src as usize],
        &SCHED.econ_src_n[src as usize],
        per_tok,
    );
    let i = src as usize * 4 + econ_kidx(k);
    econ_ema(&SCHED.econ_k[i], &SCHED.econ_k_n[i], per_tok);
    SCHED.econ_rounds.fetch_add(1, Ordering::Relaxed);
}

/// 소스 선택 — 가용성·측정 반영(둘 다 측정이면 낮은 비용; 미측정이면 MTP부터).
pub(super) fn econ_pick_src(mtp: Option<f64>, ng: Option<f64>, mtp_avail: bool) -> u8 {
    if !mtp_avail {
        return 1;
    }
    match (mtp, ng) {
        (Some(m), Some(g)) => {
            if g < m {
                1
            } else {
                0
            }
        }
        (None, Some(_)) => 1,
        _ => 0,
    }
}

/// 스펙/평범 히스테리시스 — ratio = 스펙 비용/평범 비용.
/// ON 유지: ratio ≤ 1+m · OFF→ON: ratio < 1−m. 스펙 미측정이면 측정 위해 ON.
pub(super) fn econ_should_spec(spec: Option<f64>, plain: Option<f64>, on: bool) -> bool {
    match (spec, plain) {
        (None, _) | (_, None) => true,
        (Some(s), Some(p)) => {
            let r = s / p;
            if on {
                r <= 1.0 + ECON_MARGIN
            } else {
                r < 1.0 - ECON_MARGIN
            }
        }
    }
}

/// k 선택 — 미측정 탐색(4→6→2→8) 후 최소 EMA(3% 여유 = 히스테리시스).
pub(super) fn econ_pick_k(cur: usize, emas: [Option<f64>; 4]) -> usize {
    for &j in &[1usize, 2, 0, 3] {
        if emas[j].is_none() {
            return j;
        }
    }
    let mut best = cur;
    let mut best_v = emas[cur].unwrap_or(f64::INFINITY);
    for (j, v) in emas.iter().enumerate() {
        let v = v.unwrap(); // 미측정 없음(위 루프)
        if v < best_v * (1.0 - ECON_K_MARGIN) {
            best = j;
            best_v = v;
        }
    }
    best
}

/// [2.A-④] 게이트 판정 — (소스, k). None = 평범(라운드 생략).
pub(super) fn econ_plan(mtp_avail: bool) -> Option<(u8, usize)> {
    // [진단 2026-10-13] 비결정성 격리용 — 스펙 전면 off(평범 전용). 무손실
    // 경로 선택(성능만) · 운영 진단 한정. [2.J] CLI `--spec off`(정책) 우선.
    if crate::engine::SPEC_POLICY.load(Ordering::Relaxed) == 3
        || llm170_diag::flag::on_nonzero("LLM170_SPEC_OFF")
    {
        return None;
    }
    let plain = econ_get(&SCHED.econ_plain, &SCHED.econ_plain_n);
    let mtp = econ_get(&SCHED.econ_src[0], &SCHED.econ_src_n[0]);
    let ng = econ_get(&SCHED.econ_src[1], &SCHED.econ_src_n[1]);
    let mut src = econ_pick_src(mtp, ng, mtp_avail);
    // 소스 탐사 — 미측정 소스 우선(비교 성립), 이후 32라운드마다 교차 1회.
    // (스펙이 계속 켜지면 안 뽑힌 소스는 영원히 미측정 — 자기참조 차단.)
    if mtp_avail {
        let other = 1 - src;
        let n_other = SCHED.econ_src_n[other as usize].load(Ordering::Relaxed);
        if n_other < 3 || SCHED.econ_rounds.load(Ordering::Relaxed) % 32 == 15 {
            src = other;
        }
    }
    // 평범 프로브 — 초기 3표본 부트스트랩(4라운드마다 1회 스킵) 후 64라운드
    // 마다 1회 — 스펙이 계속 켜지면 평범 실측이 갱신되지 않는 자기참조 차단.
    // (상태(econ_on)는 유지 — 스킵 1회 뒤 재판정.)
    let rounds = SCHED.econ_rounds.load(Ordering::Relaxed);
    let plain_n = SCHED.econ_plain_n.load(Ordering::Relaxed);
    if (plain_n < 3 && rounds % 4 == 3) || rounds % 64 == 47 {
        return None;
    }
    let spec = if src == 0 { mtp } else { ng };
    let on = SCHED.econ_on.load(Ordering::Relaxed) == 1;
    let mut use_spec = econ_should_spec(spec, plain, on);
    if !use_spec {
        let gtok = SCHED.econ_gen.load(Ordering::Relaxed);
        let last = SCHED.econ_probe_gen.load(Ordering::Relaxed);
        if gtok.saturating_sub(last) < ECON_PROBE_TOKENS {
            SCHED.econ_on.store(0, Ordering::Relaxed);
            return None;
        }
        // 재프로브 1라운드 — 결과 표본이 다음 판정을 갱신(밴드 밖이면 OFF 복귀).
        // OFF 동안에는 완료 라운드가 드물어 rounds 기반 소스 교차가 얼어붙는다 —
        // 프로브마다 소스를 교차해 MTP/ngram 둘 다 주기적으로 재측정한다.
        if mtp_avail {
            let alt = SCHED.econ_probe_alt.load(Ordering::Relaxed) ^ 1;
            SCHED.econ_probe_alt.store(alt, Ordering::Relaxed);
            src = alt as u8;
        }
        SCHED.econ_probe_gen.store(gtok, Ordering::Relaxed);
        SCHED.econ_probe_mark.store(1, Ordering::Relaxed);
        use_spec = true;
    }
    SCHED.econ_on.store(use_spec as u64, Ordering::Relaxed);
    SCHED.econ_src_cur.store(src as u64, Ordering::Relaxed);
    let k = if src == 1 {
        let emas: [Option<f64>; 4] =
            std::array::from_fn(|j| econ_get(&SCHED.econ_k[4 + j], &SCHED.econ_k_n[4 + j]));
        let cur = (SCHED.econ_kidx.load(Ordering::Relaxed) as usize).min(3);
        let mut idx = econ_pick_k(cur, emas);
        // 32라운드마다 밴드 순환 1회(탐사 — EMA가 낡은 순위에 갇히지 않게).
        if SCHED.econ_rounds.load(Ordering::Relaxed) % 32 == 31 {
            idx = (idx + 1) % 4;
        }
        SCHED.econ_kidx.store(idx as u64, Ordering::Relaxed);
        ECON_KS[idx]
    } else {
        4 // MTP 저장 초안 폭 고정(체인 K=4)
    };
    Some((src, k))
}

#[cfg(test)]
mod tests {
    use super::{econ_pick_k, econ_pick_src, econ_should_spec};

    /// [2.A-④ 2026-10-13] 경제성 게이트 판정 — 소스 선택·히스테리시스·k밴드.
    #[test]
    fn econ_gate_decisions() {
        // 소스: 둘 다 측정 → 낮은 비용. MTP 미가용 → ngram 고정.
        assert_eq!(econ_pick_src(Some(24.0), Some(17.0), true), 1);
        assert_eq!(econ_pick_src(Some(17.0), Some(24.0), true), 0);
        assert_eq!(econ_pick_src(Some(1.0), Some(1.0), false), 1);
        // 미측정 — MTP 가용이면 MTP로 측정 개시, 아니면 ngram.
        assert_eq!(econ_pick_src(None, None, true), 0);
        assert_eq!(econ_pick_src(None, None, false), 1);
        // 스펙/평범 히스테리시스 — ON 유지(r≤1.02), OFF→ON은 r<0.98 필요.
        assert!(econ_should_spec(Some(10.0), Some(10.0), true), "동률 유지");
        assert!(
            !econ_should_spec(Some(10.3), Some(10.0), true),
            "ON에서 +3%면 OFF"
        );
        assert!(
            !econ_should_spec(Some(10.0), Some(10.0), false),
            "OFF에서 동률이면 OFF 유지(밴드)"
        );
        assert!(
            econ_should_spec(Some(9.7), Some(10.0), false),
            "OFF에서 −3%면 ON"
        );
        assert!(
            econ_should_spec(None, Some(10.0), false),
            "미측정은 ON(탐색)"
        );
        // k밴드: 미측정 탐색(4→6→2→8) → 최소 EMA(3% 여유).
        assert_eq!(econ_pick_k(1, [None, None, None, None]), 1);
        assert_eq!(econ_pick_k(1, [None, Some(1.0), None, None]), 2);
        assert_eq!(
            econ_pick_k(1, [Some(9.0), Some(10.0), Some(11.0), Some(12.0)]),
            0,
            "9.0 < 10.0×0.97 — 전환"
        );
        assert_eq!(
            econ_pick_k(1, [Some(9.8), Some(10.0), Some(10.1), Some(10.2)]),
            1,
            "9.8 > 10.0×0.97 — 유지(히스테리시스)"
        );
    }
}
