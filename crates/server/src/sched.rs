use crate::engine::{Engine, InferResult};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) mod admit;
mod assign;
mod decode;
mod econ;
mod prefill;
mod publish;
mod spec;
use self::admit::Aging;
use self::assign::*;
use self::decode::*;
use self::prefill::*;
use self::publish::*;

const PF_BUDGET_TOKENS: usize = 1024;
const PF_CHUNK_MAX: usize = 512;
const PF_CHUNK_MIN: usize = 64;
const PF_LONG_PROMPT: usize = 2048;

pub enum ProgressFrame {
    Tok(u32),
    Ping,
}

pub struct SlotJob {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
    pub constraint: Option<crate::oai::schema::Schema>,
    pub sampler: Option<llm170_core::sampler::SamplerParams>,
    pub stops: Vec<u32>,
    pub progress: Option<std::sync::mpsc::Sender<ProgressFrame>>,
    pub out: std::sync::mpsc::Sender<InferResult>,
    pub queued: std::time::Instant,
    pub aging_blocked: bool,
}
struct Slot {
    job: Option<SlotJob>,
    prefilled: usize,
    next: u32,
    generated: u32,
    tokens: Vec<u32>,
    touch: u64,
    cancelled: bool,
    cached: Vec<u32>,
    sampler: Option<llm170_core::sampler::Sampler>,
    err_count: u32,
    failed: Option<String>,
    constraint: Option<crate::oai::schema::Constraint>,
    spec_miss: u32,
    t_first: Option<std::time::Instant>,
}

impl Slot {
    fn free() -> Self {
        Slot {
            job: None,
            prefilled: 0,
            next: 0,
            generated: 0,
            tokens: Vec::new(),
            touch: 0,
            cancelled: false,
            cached: Vec::new(),
            sampler: None,
            err_count: 0,
            failed: None,
            t_first: None,
            spec_miss: 0,
            constraint: None,
        }
    }
}

fn slot_fail(s: &mut Slot, msg: String) {
    s.err_count += 1;
    if s.err_count >= 3 && s.failed.is_none() {
        eprintln!("# slot 확정 실패(연속 {}회): {}", s.err_count, msg);
        s.failed = Some(msg);
    }
}

pub(crate) fn load_gpu_retry(
    p: &std::path::Path,
    n_slots: usize,
    ctx: usize,
    kv_budget: Option<u64>,
    kv_reserve: Option<u64>,
    kv_vram_cap: Option<u64>,
    cards: usize,
) -> crate::gpu_engine::GpuEngine {
    let _ = crate::engine::ARCH.set(llm170_core::qwen35::bind::dir_arch(p).unwrap_or_default());
    for i in 0..5 {
        let loaded = if cards == 2 {
            crate::gpu_engine::GpuEngine::load_pp(
                p,
                n_slots,
                ctx,
                kv_budget,
                kv_reserve,
                kv_vram_cap,
            )
        } else {
            crate::gpu_engine::GpuEngine::load(p, n_slots, ctx, kv_budget, kv_reserve, kv_vram_cap)
        };
        match loaded {
            Ok(mut e) => {
                if let Err(err) = e.enable_spec() {
                    eprintln!("# spec 비활성: {err}");
                }
                return e;
            }
            Err(err) => {
                eprintln!("# gpu 엔진 로드 재시도 {}/5: {err}", i + 1);
                if err.contains("메모리 부족") || err.contains("OUT_OF_MEMORY") {
                    eprintln!("# gpu 엔진 로드 실패(VRAM 부족): {err}");
                    eprintln!("# --ctx 를 줄이세요(예: 32768).");
                    std::process::exit(1);
                }
                if !(err.contains("ENOPT") || err.contains("cuModule")) {
                    panic!("gpu 엔진 로드 실패(재시도 불가): {} — {err}", p.display());
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    panic!("gpu 엔진 로드 최종 실패: {}", p.display())
}

pub static SCHED: Sched = Sched {
    jobs: AtomicU64::new(0),
    queue_wait_us: AtomicU64::new(0),
    prefix_tokens: AtomicU64::new(0),
    prefix_full_hits: AtomicU64::new(0),
    prefix_ckpt_hits: AtomicU64::new(0),
    trie_boundaries: AtomicU64::new(0),
    ttft_us: AtomicU64::new(0),
    ticks_decode: AtomicU64::new(0),
    ms_decode: AtomicU64::new(0),
    chunks_prefill: AtomicU64::new(0),
    ms_prefill: AtomicU64::new(0),
    spec_rounds: AtomicU64::new(0),
    spec_accepted: AtomicU64::new(0),
    spec_ms: AtomicU64::new(0),
    slots_active: AtomicU64::new(0),
    prompt_tokens: AtomicU64::new(0),
    gen_tokens: AtomicU64::new(0),
    requests: AtomicU64::new(0),
    requests_failed: AtomicU64::new(0),
    queue_depth: AtomicU64::new(0),
    econ_plain: AtomicU64::new(0),
    econ_plain_n: AtomicU64::new(0),
    econ_src: [AtomicU64::new(0), AtomicU64::new(0)],
    econ_src_n: [AtomicU64::new(0), AtomicU64::new(0)],
    econ_k: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
    econ_k_n: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
    econ_on: AtomicU64::new(1),
    econ_kidx: AtomicU64::new(1),
    econ_src_cur: AtomicU64::new(0),
    econ_probe_alt: AtomicU64::new(0),
    econ_gen: AtomicU64::new(0),
    econ_probe_gen: AtomicU64::new(0),
    econ_rounds: AtomicU64::new(0),
    aging_frozen: AtomicU64::new(0),
    aging_d_ms: AtomicU64::new(0),
    aging_freeze_ms: AtomicU64::new(0),
    aging_events: AtomicU64::new(0),
    aging_blocked_jobs: AtomicU64::new(0),
    aging_blocked_tokens: AtomicU64::new(0),
    aging_head_wait_p95_ms: AtomicU64::new(0),
};
pub struct Sched {
    pub jobs: AtomicU64,
    pub queue_wait_us: AtomicU64,
    pub prefix_tokens: AtomicU64,
    pub prefix_full_hits: AtomicU64,
    pub prefix_ckpt_hits: AtomicU64,
    pub trie_boundaries: AtomicU64,
    pub ttft_us: AtomicU64,
    pub ticks_decode: AtomicU64,
    pub ms_decode: AtomicU64,
    pub chunks_prefill: AtomicU64,
    pub ms_prefill: AtomicU64,
    pub spec_rounds: AtomicU64,
    pub spec_accepted: AtomicU64,
    pub spec_ms: AtomicU64,
    pub slots_active: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub gen_tokens: AtomicU64,
    pub requests: AtomicU64,
    pub requests_failed: AtomicU64,
    pub queue_depth: AtomicU64,
    pub econ_plain: AtomicU64,
    pub econ_plain_n: AtomicU64,
    pub econ_src: [AtomicU64; 2],
    pub econ_src_n: [AtomicU64; 2],
    pub econ_k: [AtomicU64; 8],
    pub econ_k_n: [AtomicU64; 8],
    pub econ_on: AtomicU64,
    pub econ_kidx: AtomicU64,
    pub econ_src_cur: AtomicU64,
    pub econ_probe_alt: AtomicU64,
    pub econ_gen: AtomicU64,
    pub econ_probe_gen: AtomicU64,
    pub econ_rounds: AtomicU64,
    pub aging_frozen: AtomicU64,
    pub aging_d_ms: AtomicU64,
    pub aging_freeze_ms: AtomicU64,
    pub aging_events: AtomicU64,
    pub aging_blocked_jobs: AtomicU64,
    pub aging_blocked_tokens: AtomicU64,
    pub aging_head_wait_p95_ms: AtomicU64,
}
impl Sched {
    pub fn summary(&self) -> String {
        let jobs = self.jobs.load(Ordering::Relaxed);
        let qw = self.queue_wait_us.load(Ordering::Relaxed);
        let px = self.prefix_tokens.load(Ordering::Relaxed);
        let (fh, ch) = (
            self.prefix_full_hits.load(Ordering::Relaxed),
            self.prefix_ckpt_hits.load(Ordering::Relaxed),
        );
        let td = self.ticks_decode.load(Ordering::Relaxed);
        let md = self.ms_decode.load(Ordering::Relaxed);
        let cp = self.chunks_prefill.load(Ordering::Relaxed);
        let mp = self.ms_prefill.load(Ordering::Relaxed);
        let sr = self.spec_rounds.load(Ordering::Relaxed);
        let sa = self.spec_accepted.load(Ordering::Relaxed);
        let sms = self.spec_ms.load(Ordering::Relaxed);
        let qd = self.queue_depth.load(Ordering::Relaxed);
        format!(
            "[sched] jobs {jobs} | queue {qd} | queue-wait avg {:.0}ms | prefix-reuse {px}tok(f{fh}/c{ch}) | decode {td}x avg {:.1}ms | prefill {cp}x avg {:.1}ms | spec {sr}r acc {sa} ({:.2}/r) avg {:.1}ms",
            if jobs > 0 {
                qw as f64 / jobs as f64 / 1e3
            } else {
                0.0
            },
            if td > 0 { md as f64 / td as f64 } else { 0.0 },
            if cp > 0 { mp as f64 / cp as f64 } else { 0.0 },
            if sr > 0 { sa as f64 / sr as f64 } else { 0.0 },
            if sr > 0 {
                sms as f64 / sr as f64 / 1e3
            } else {
                0.0
            },
        )
    }
}

pub fn slot_loop(mut eng: Engine, rx: std::sync::mpsc::Receiver<SlotJob>, n_slots: usize) {
    let mut ptrie = crate::tree::PrefixTrie::new();
    let eos = eng.eos();
    let mut pending: Vec<SlotJob> = Vec::new();
    let d_fix = llm170_diag::flag::val("LLM170_AGING_D")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|d| *d > 0.0);
    let mut aging = Aging::new(30.0, d_fix);
    let mut tps = 0f64;
    let mut last_tick = std::time::Instant::now();
    let copy_base = eng.copy_stats();
    {
        let warm: Vec<u32> = vec![1u32; 528];
        for slot in 0..n_slots {
            if let Err(e) = eng.0.kv_grow(slot, warm.len() + 8) {
                eprintln!("# warmup KV 확보 실패(치명 아님) slot{slot}: {e}");
            }
            let w: Result<(), String> = eng.0.prefill(slot, &warm).and_then(|l| {
                let t = llm170_core::qwen35::greedy(&l);
                eng.0.decode_greedy(slot, t).map(|_| ())
            });
            if let Err(err) = w {
                eprintln!("# warmup 실패(치명 아님) slot{slot}: {err}");
            }
            let _ = eng.0.reset_seq(slot);
        }
        if let Err(e) = eng.0.warm_spec_graphs(n_slots, eos) {
            eprintln!("# warmup spec 캡처 실패(치명 아님): {e}");
        }
    }
    crate::http::READY.store(true, std::sync::atomic::Ordering::Release);
    let mut slots: Vec<Slot> = (0..n_slots).map(|_| Slot::free()).collect();
    let mut tick: u64 = 0;
    let (mut n_dec, mut n_pf) = (0u64, 0u64);
    let (mut ms_dec, mut ms_pf, mut ms_fin) = (0f64, 0f64, 0f64);
    let npw = llm170_diag::dump::opts().key("wall_time");
    let t0w = std::time::Instant::now();
    let mut last_wt = std::time::Instant::now();
    let mut last_pub = std::time::Instant::now();
    loop {
        llm170_diag::watchdog::bump();
        if npw {
            let now = std::time::Instant::now();
            let dt = last_wt.elapsed().as_secs_f64();
            if dt > 0.05 {
                eprintln!("[wall] +{dt:.2}s @{}s", t0w.elapsed().as_secs_f64());
            }
            last_wt = now;
        }
        let now = std::time::Instant::now();
        let dt_ms = now.duration_since(last_tick).as_millis().min(60_000) as u64;
        last_tick = now;
        let freeze = match pending.first() {
            Some(j) => {
                let mut free_eff = eng.0.kv_free_blocks();
                let mut running: Vec<(usize, usize)> = Vec::new();
                for (i, s) in slots.iter().enumerate() {
                    let held = eng.0.kv_used_blocks(i);
                    match &s.job {
                        Some(j) => {
                            let rem = j.tokens.len().saturating_sub(s.prefilled)
                                + j.n_predict.max(1).saturating_sub(s.generated as usize);
                            running.push((rem, held));
                        }
                        None => free_eff += held,
                    }
                }
                let hb = eng.0.kv_blocks_needed(j_need_tokens(j));
                aging.decide(dt_ms, hb, free_eff, &running, tps)
            }
            None => {
                aging.idle();
                false
            }
        };
        if let Some(msg) = aging.maybe_adapt(now) {
            eprintln!("# aging: {msg}");
        }
        let live_tokens: usize = slots
            .iter()
            .filter(|s| s.job.is_none() && s.cached.len() >= 512)
            .map(|s| s.cached.len())
            .sum();
        if ptrie.n_nodes > live_tokens.saturating_mul(2).saturating_add(4096) {
            let before = ptrie.n_nodes;
            ptrie.rebuild(
                slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.job.is_none() && s.cached.len() >= 512)
                    .map(|(i, s)| (i, s.cached.as_slice())),
            );
            eprintln!(
                "# prefix-trie: 재구축 {before}→{}노드 (live {live_tokens}토큰)",
                ptrie.n_nodes
            );
        }
        for (i, s) in slots.iter().enumerate() {
            if s.job.is_none() && s.cached.len() >= 512 {
                ptrie.insert(&s.cached, i);
                SCHED
                    .trie_boundaries
                    .store(ptrie.n_boundaries as u64, Ordering::Relaxed);
            }
        }
        slot_intake(
            &mut slots,
            &mut eng,
            &rx,
            tick,
            &mut ptrie,
            &mut pending,
            freeze,
            &mut aging,
            tps,
        );
        maybe_publish(&slots, &eng, copy_base, &mut last_pub, &aging);
        tick += 1;
        let _it0 = std::time::Instant::now();

        let (active, decoded, dec_ms, fin_ms) = slot_decode(&mut slots, &mut eng, eos);
        ms_fin += fin_ms;

        let pending_prefill = slots
            .iter()
            .any(|s| s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len());
        if !decoded || pending_prefill {
            if let Some(pf_ms) = slot_prefill(&mut eng, &mut slots, &active, tick, eos, npw, &t0w) {
                n_pf += 1;
                ms_pf += pf_ms;
            } else {
                continue;
            }
        }

        if decoded {
            if dec_ms > 0.0 {
                let r = 1000.0 / dec_ms;
                tps = if tps <= 0.0 { r } else { 0.9 * tps + 0.1 * r };
            }
            n_dec += 1;
            ms_dec += dec_ms;
            SCHED.ticks_decode.fetch_add(1, Ordering::Relaxed);
            SCHED.ms_decode.fetch_add(dec_ms as u64, Ordering::Relaxed);
            if llm170_diag::dump::opts().key("srv_time") && n_dec % 32 == 0 {
                eprintln!(
                    "[srv] steps={} decode avg {:.1}ms (결과 처리 {:.2}ms) | prefill {}x avg {:.1}ms | {}",
                    n_dec,
                    ms_dec / n_dec as f64,
                    ms_fin / n_dec as f64,
                    n_pf,
                    if n_pf > 0 { ms_pf / n_pf as f64 } else { 0.0 },
                    SCHED.summary()
                );
                if let Ok(p) = eng.0.prof_report("serve")
                    && !p.is_empty()
                {
                    eprintln!("{p}");
                }
            }
        }
        let busy = slots.iter().any(|s| s.job.is_some());
        if !busy {
            publish_slot_views(&slots);
            publish_stats(&eng, copy_base);
            publish_aging(&aging);
            if !pending.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
        }
        if !busy {
            match rx.recv() {
                Ok(j) => {
                    SCHED.queue_depth.fetch_sub(1, Ordering::Relaxed);
                    let need = j_need_tokens(&j);
                    if !admit_ok(&eng, &j) {
                        if eng.0.kv_blocks_needed(need) > eng.0.kv_total_blocks() {
                            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
                            let _ = j.out.send(InferResult {
                                tokens: Vec::new(),
                                error: Some(format!(
                                    "request exceeds the KV pool: need {need} tokens (prompt + max_tokens) > pool capacity"
                                )),
                            });
                            continue;
                        }
                        while !admit_ok(&eng, &j) && reclaim_idle(&mut eng, &mut slots) {}
                        if !admit_ok(&eng, &j) {
                            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
                            let _ = j.out.send(InferResult {
                                tokens: Vec::new(),
                                error: Some(format!(
                                    "request exceeds the KV pool: need {need} tokens (prompt + max_tokens) > pool capacity"
                                )),
                            });
                            continue;
                        }
                    }
                    assign_slot(&mut slots, &mut eng, j, tick, &mut ptrie);
                }
                Err(_) => break,
            }
        }
    }
    eprintln!("{}", SCHED.summary());
}

#[allow(clippy::too_many_arguments)]
fn slot_intake(
    slots: &mut [Slot],
    eng: &mut Engine,
    rx: &std::sync::mpsc::Receiver<SlotJob>,
    tick: u64,
    ptrie: &mut crate::tree::PrefixTrie,
    pending: &mut Vec<SlotJob>,
    freeze: bool,
    aging: &mut Aging,
    tps: f64,
) {
    loop {
        if !slots.iter().any(|s| s.job.is_none()) {
            break;
        }
        let needs: Vec<usize> = pending.iter().map(j_need_tokens).collect();
        if freeze {
            for (k, j) in pending.iter_mut().enumerate().skip(1) {
                if !j.aging_blocked && eng.0.kv_blocks_needed(needs[k]) <= eng.0.kv_total_blocks() {
                    j.aging_blocked = true;
                    aging.note_blocked(needs[k]);
                }
            }
        }
        let pick = admit::binpack_order(&needs, eng.0.kv_free_blocks(), freeze, |n| {
            eng.0.kv_blocks_needed(n)
        })
        .first()
        .copied();
        match pick {
            Some(i) => {
                let head = i == 0;
                let j = pending.remove(i);
                if head {
                    aging.note_head_wait(j.queued.elapsed().as_secs_f64());
                }
                if tps > 0.0 {
                    aging.observe_est(j_need_tokens(&j) as f64 / tps);
                }
                assign_slot(slots, eng, j, tick, ptrie);
            }
            None => {
                let mut rejected = false;
                let mut k = 0;
                while k < pending.len() {
                    let need = j_need_tokens(&pending[k]);
                    if eng.0.kv_blocks_needed(need) > eng.0.kv_total_blocks() {
                        let j = pending.remove(k);
                        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
                        let _ = j.out.send(InferResult {
                            tokens: Vec::new(),
                            error: Some(format!(
                                "request exceeds the KV pool: need {need} tokens (prompt + max_tokens) > pool capacity"
                            )),
                        });
                        rejected = true;
                    } else {
                        k += 1;
                    }
                }
                if rejected {
                    continue;
                }
                if reclaim_idle(eng, slots) {
                    continue;
                }
                break;
            }
        }
    }
    loop {
        if !slots.iter().any(|s| s.job.is_none()) {
            break;
        }
        let Ok(mut j) = rx.try_recv() else {
            break;
        };
        SCHED.queue_depth.fetch_sub(1, Ordering::Relaxed);
        if freeze {
            j.aging_blocked = true;
            aging.note_blocked(j_need_tokens(&j));
            pending.push(j);
        } else if admit_ok(eng, &j) {
            if tps > 0.0 {
                aging.observe_est(j_need_tokens(&j) as f64 / tps);
            }
            assign_slot(slots, eng, j, tick, ptrie);
        } else if j_need_tokens(&j) > 0
            && eng.0.kv_blocks_needed(j_need_tokens(&j)) > eng.0.kv_total_blocks()
        {
            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
            let _ = j.out.send(InferResult {
                tokens: Vec::new(),
                error: Some(format!(
                    "request exceeds the KV pool: need {} tokens (prompt + max_tokens) > pool capacity",
                    j_need_tokens(&j)
                )),
            });
        } else {
            let _ = reclaim_idle(eng, slots);
            pending.push(j);
        }
    }
}

fn j_need_tokens(j: &SlotJob) -> usize {
    admit::need_tokens(j.tokens.len(), j.n_predict)
}

fn reclaim_idle(eng: &mut Engine, slots: &mut [Slot]) -> bool {
    match reclaim_victim(slots) {
        Some(i) => {
            let freed = eng.0.kv_release_slot(i);
            if freed > 0 {
                slots[i].cached.clear();
                eprintln!("# kv 압박: 슬롯{i} 접두 캐시 회수({freed}블록)");
                true
            } else {
                false
            }
        }
        None => false,
    }
}

fn reclaim_victim(slots: &[Slot]) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.job.is_none() && !s.cached.is_empty())
        .min_by_key(|(_, s)| (s.cached.len(), s.touch))
        .map(|(i, _)| i)
}

fn admit_ok(eng: &Engine, j: &SlotJob) -> bool {
    let need = j_need_tokens(j);
    eng.0.kv_blocks_needed(need) <= eng.0.kv_free_blocks()
}

#[cfg(test)]
mod tests {
    use super::{Slot, reclaim_victim};

    #[test]
    fn spec_gate_boundary() {
        let mut s = Slot::free();
        s.spec_miss = 0;
        s.spec_miss += 1;
        assert_eq!(s.spec_miss, 1);
        s.generated = 256;
        assert_eq!(s.generated & 255, 0, "256 배수에서 재시도 허용");
        s.generated = 257;
        assert_ne!(s.generated & 255, 0, "256 배수 외 재시도 차단");
    }

    #[test]
    fn reclaim_victim_flop_aware() {
        let mut long = Slot::free();
        long.cached = vec![0; 700];
        long.touch = 10;
        let mut short = Slot::free();
        short.cached = vec![0; 100];
        short.touch = 1;
        let mut empty = Slot::free();
        empty.touch = 0;
        assert_eq!(
            reclaim_victim(&[long, short, empty]),
            Some(1),
            "짧은 접두 우선"
        );
        let mut a = Slot::free();
        a.cached = vec![0; 100];
        a.touch = 5;
        let mut b = Slot::free();
        b.cached = vec![0; 100];
        b.touch = 2;
        assert_eq!(reclaim_victim(&[a, b]), Some(1));
        assert_eq!(reclaim_victim(&[Slot::free()]), None);
    }

    #[test]
    fn prefix_trie_boundaries() {
        let mut t = crate::tree::PrefixTrie::new();
        let tokens: Vec<u32> = (1..=600).collect();
        t.insert(&tokens, 0);
        assert_eq!(t.longest_prefix(&tokens), Some((0, 512)));
        let short: Vec<u32> = (1..100).collect();
        assert_eq!(t.longest_prefix(&short), None);
        t.remove(0);
        assert_eq!(t.longest_prefix(&tokens), None);
    }
}
