use super::SCHED;
use std::sync::atomic::{AtomicU64, Ordering};


const ECON_KS: [usize; 4] = [2, 4, 6, 8];
const ECON_MARGIN: f64 = 0.02;
const ECON_K_MARGIN: f64 = 0.03;
const ECON_PROBE_TOKENS: u64 = 512;

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

pub(super) fn econ_plain_sample(ms: f64, toks: usize) {
    if toks > 0 {
        econ_ema(&SCHED.econ_plain, &SCHED.econ_plain_n, ms / toks as f64);
    }
}

pub(super) fn econ_mtp_maintain() -> bool {
    SCHED.econ_on.load(Ordering::Relaxed) == 1 && SCHED.econ_src_cur.load(Ordering::Relaxed) == 0
}

pub(super) fn econ_spec_sample(src: u8, k: usize, acc: usize, ms: f64) {
    let per_tok = ms / (acc + 1) as f64;
    econ_ema(
        &SCHED.econ_src[src as usize],
        &SCHED.econ_src_n[src as usize],
        per_tok,
    );
    let i = src as usize * 4 + econ_kidx(k);
    econ_ema(&SCHED.econ_k[i], &SCHED.econ_k_n[i], per_tok);
    SCHED.econ_rounds.fetch_add(1, Ordering::Relaxed);
}

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

pub(super) fn econ_pick_k(cur: usize, emas: [Option<f64>; 4]) -> usize {
    for &j in &[1usize, 2, 0, 3] {
        if emas[j].is_none() {
            return j;
        }
    }
    let mut best = cur;
    let mut best_v = emas[cur].unwrap_or(f64::INFINITY);
    for (j, v) in emas.iter().enumerate() {
        let v = v.unwrap();
        if v < best_v * (1.0 - ECON_K_MARGIN) {
            best = j;
            best_v = v;
        }
    }
    best
}

pub(super) fn econ_plan(mtp_avail: bool) -> Option<(u8, usize)> {
    if !crate::engine::spec_enabled_by_policy()
        || llm170_diag::flag::on_nonzero("LLM170_SPEC_OFF")
    {
        return None;
    }
    let plain = econ_get(&SCHED.econ_plain, &SCHED.econ_plain_n);
    let mtp = econ_get(&SCHED.econ_src[0], &SCHED.econ_src_n[0]);
    let ng = econ_get(&SCHED.econ_src[1], &SCHED.econ_src_n[1]);
    let mut src = econ_pick_src(mtp, ng, mtp_avail);
    if mtp_avail {
        let other = 1 - src;
        let n_other = SCHED.econ_src_n[other as usize].load(Ordering::Relaxed);
        if n_other < 3 || SCHED.econ_rounds.load(Ordering::Relaxed) % 32 == 15 {
            src = other;
        }
    }
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
        if mtp_avail {
            let alt = SCHED.econ_probe_alt.load(Ordering::Relaxed) ^ 1;
            SCHED.econ_probe_alt.store(alt, Ordering::Relaxed);
            src = alt as u8;
        }
        SCHED.econ_probe_gen.store(gtok, Ordering::Relaxed);
        use_spec = true;
    }
    SCHED.econ_on.store(use_spec as u64, Ordering::Relaxed);
    SCHED.econ_src_cur.store(src as u64, Ordering::Relaxed);
    let k = if src == 1 {
        let emas: [Option<f64>; 4] =
            std::array::from_fn(|j| econ_get(&SCHED.econ_k[4 + j], &SCHED.econ_k_n[4 + j]));
        let cur = (SCHED.econ_kidx.load(Ordering::Relaxed) as usize).min(3);
        let mut idx = econ_pick_k(cur, emas);
        if SCHED.econ_rounds.load(Ordering::Relaxed) % 32 == 31 {
            idx = (idx + 1) % 4;
        }
        SCHED.econ_kidx.store(idx as u64, Ordering::Relaxed);
        ECON_KS[idx]
    } else {
        4
    };
    Some((src, k))
}

#[cfg(test)]
mod tests {
    use super::{econ_pick_k, econ_pick_src, econ_should_spec};

    #[test]
    fn econ_gate_decisions() {
        assert_eq!(econ_pick_src(Some(24.0), Some(17.0), true), 1);
        assert_eq!(econ_pick_src(Some(17.0), Some(24.0), true), 0);
        assert_eq!(econ_pick_src(Some(1.0), Some(1.0), false), 1);
        assert_eq!(econ_pick_src(None, None, true), 0);
        assert_eq!(econ_pick_src(None, None, false), 1);
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
