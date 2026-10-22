use super::decode::pick;
use super::econ::econ_mtp_maintain;
use super::spec::slot_emit;
use super::{
    Engine, InferResult, PF_BUDGET_TOKENS, PF_CHUNK_MAX, PF_CHUNK_MIN, PF_LONG_PROMPT,
    ProgressFrame, SCHED, Slot, slot_fail,
};
use std::sync::atomic::Ordering;

pub(super) fn slot_prefill(
    eng: &mut Engine,
    slots: &mut [Slot],
    active: &[usize],
    tick: u64,
    eos: u32,
    npw: bool,
    t0w: &std::time::Instant,
) -> Option<f64> {
    let others_decoding = slots
        .iter()
        .any(|s| s.job.is_some() && s.prefilled == s.job.as_ref().unwrap().tokens.len());
    let backlog = slots
        .iter()
        .filter(|s| s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len())
        .count();
    let cap_now = if others_decoding && backlog > 1 {
        PF_CHUNK_MAX
    } else if others_decoding {
        eng.0.prefill_mixed_cap()
    } else {
        PF_CHUNK_MAX
    };
    let pf = (0..slots.len())
        .filter(|&i| {
            slots[i].job.is_some()
                && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
        })
        .min_by_key(|&i| {
            let left = slots[i].job.as_ref().unwrap().tokens.len() - slots[i].prefilled;
            (if left <= cap_now { 0 } else { 1 }, slots[i].touch)
        });
    let i = pf?;
    let _pft = std::time::Instant::now();
    llm170_diag::watchdog::bump();
    let left = slots[i].job.as_ref().unwrap().tokens.len() - slots[i].prefilled;
    let mut chunk = PF_BUDGET_TOKENS
        .saturating_sub(active.len())
        .clamp(PF_CHUNK_MIN, PF_CHUNK_MAX);
    if backlog > 1 && left > PF_LONG_PROMPT {
        chunk = chunk.min(PF_CHUNK_MAX / 2);
    }
    if others_decoding {
        chunk = chunk.min(cap_now);
    }
    let chunk = chunk.min(left);
    let (start, logits) = {
        let end = (slots[i].prefilled + chunk).min(slots[i].job.as_ref().unwrap().tokens.len());
        let part: Vec<u32> =
            slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
        let samp = slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
        let cons = slots[i].constraint.is_some();
        let r: Result<u32, String> = if end < slots[i].job.as_ref().unwrap().tokens.len() {
            eng.0.prefill_state(i, &part).map(|_| 0)
        } else if samp || cons {
            eng.0
                .prefill(i, &part)
                .map(|mut l| pick(&mut slots[i], &mut l))
        } else {
            eng.0.prefill_greedy(i, &part)
        };
        (end, r)
    };
    slots[i].touch = tick;
    if let Some(p) = slots[i].job.as_ref().and_then(|j| j.progress.as_ref())
        && p.send(ProgressFrame::Ping).is_err()
    {
        if let Some(j) = slots[i].job.take() {
            let _ = j.out.send(InferResult {
                tokens: Vec::new(),
                error: None,
            });
        }
        eng.reset_seq(i);
        slots[i] = Slot::free();
        return None;
    }
    if npw {
        eprintln!(
            "[wall] prefill slot{i} {start}tok done @{}s",
            t0w.elapsed().as_secs_f64()
        );
    }
    let ms = _pft.elapsed().as_secs_f64() * 1e3;
    SCHED.chunks_prefill.fetch_add(1, Ordering::Relaxed);
    SCHED.ms_prefill.fetch_add(ms as u64, Ordering::Relaxed);
    match logits {
        Ok(t) => {
            slots[i].prefilled = start;
            if start == slots[i].job.as_ref().unwrap().tokens.len() {
                slot_emit(&mut slots[i], t);
                if eng.0.mtp_drafts_on()
                    && econ_mtp_maintain()
                    && let Err(err) = eng.0.mtp_first(i, t)
                {
                    eprintln!("# mtp(무시): {err}");
                }
            }
        }
        Err(err) => slot_fail(&mut slots[i], format!("prefill: {err}")),
    }
    finish_slot(&mut slots[i], eng, i, eos);
    Some(ms)
}

pub(super) fn finish_slot(s: &mut Slot, eng: &mut Engine, i: usize, eos: u32) {
    if let Some(err) = s.failed.take()
        && s.job.is_some()
    {
        if let Some(j) = s.job.take() {
            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
            let _ = j.out.send(InferResult {
                tokens: Vec::new(),
                error: Some(err.clone()),
            });
        }
        if err.contains(llm170_backend_gpu::CTX_OVERFLOW)
            && s.prefilled == 0
            && !s.cached.is_empty()
        {
            let keep = std::mem::take(&mut s.cached);
            *s = Slot::free();
            s.cached = keep;
            return;
        }
        eng.reset_seq(i);
        *s = Slot::free();
        return;
    }
    let done = s.job.as_ref().is_some_and(|j| {
        s.prefilled == j.tokens.len()
            && (s.cancelled
                || s.next == eos
                || j.stops.contains(&s.next)
                || s.generated as usize >= j.n_predict.max(1))
    });
    if done {
        if let Some(j) = s.job.take() {
            let mut toks = s.tokens.clone();
            while toks.last() == Some(&eos) || j.stops.contains(toks.last().unwrap_or(&0)) {
                toks.pop();
            }
            toks.truncate(j.n_predict);
            SCHED
                .prompt_tokens
                .fetch_add(j.tokens.len() as u64, Ordering::Relaxed);
            SCHED
                .gen_tokens
                .fetch_add(toks.len() as u64, Ordering::Relaxed);
            let _ = j.out.send(InferResult {
                tokens: toks.clone(),
                error: None,
            });
            let total_ms = j.queued.elapsed().as_secs_f64() * 1e3;
            let ttft_ms = s
                .t_first
                .map(|t| t.saturating_duration_since(j.queued).as_secs_f64() * 1e3)
                .unwrap_or(total_ms);
            SCHED
                .ttft_us
                .fetch_add((ttft_ms * 1e3) as u64, Ordering::Relaxed);
            eprintln!(
                "# req: slot{i} ttft={ttft_ms:.1}ms total={total_ms:.1}ms toks={} ({:.1} tok/s) reuse={}",
                toks.len(),
                toks.len() as f64 / (total_ms / 1e3).max(1e-9),
                s.prefilled
            );
            let mut full = j.tokens.clone();
            full.extend(toks);
            s.cached = full;
        } else {
            eng.reset_seq(i);
        }
        let c = std::mem::take(&mut s.cached);
        *s = Slot::free();
        s.cached = c;
    }
}
