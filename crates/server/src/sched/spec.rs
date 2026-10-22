use super::ProgressFrame;
use super::econ::{econ_plan, econ_spec_sample};
use super::{SCHED, Slot};
use std::sync::atomic::Ordering;

pub(super) fn spec_round(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    spec_round_impl(i, s, e, false)
}

pub(super) fn spec_round_sampled(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    spec_round_impl(i, s, e, true)
}

fn spec_round_impl(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
    sampled: bool,
) -> Result<bool, String> {
    if s.spec_miss >= 1 && (s.generated & 255) != 0 {
        return Ok(false);
    }
    if sampled && s.sampler.is_none() {
        return Ok(false);
    }
    if s.constraint.is_some() {
        return Ok(false);
    }
    let mtp_avail = !sampled && e.mtp_drafts_on();
    let Some((src, kk)) = econ_plan(mtp_avail) else {
        return Ok(false);
    };
    let mut hist: Vec<u32> = Vec::new();
    if let Some(j) = &s.job {
        hist.extend_from_slice(&j.tokens);
    }
    hist.extend_from_slice(&s.tokens);
    let mtp_on = src == 0;
    let (nn, nm) = e.ngram_knobs();
    let draft: Vec<u32> = if mtp_on {
        match e.mtp_take_draft(i) {
            Some(d) => d,
            None if nn == 0 => llm170_core::spec::ngram_draft_multi(&hist, nm, kk - 1),
            None => llm170_core::spec::ngram_draft(&hist, nn, nm, kk - 1),
        }
    } else if nn == 0 {
        llm170_core::spec::ngram_draft_multi(&hist, nm, kk - 1)
    } else {
        llm170_core::spec::ngram_draft(&hist, nn, nm, kk - 1)
    };
    if draft.is_empty() {
        return Ok(false);
    }
    let t = 1 + draft.len();
    let mut rows: Vec<f32> = Vec::with_capacity(t * e.n_embd());
    rows.extend_from_slice(&e.embed_row(s.next)?);
    for &tk in &draft {
        rows.extend_from_slice(&e.embed_row(tk)?);
    }
    let rt0 = std::time::Instant::now();
    let dbg = llm170_diag::flag::on_nonzero("LLM170_MOE_DBG");
    let (toks, ms_ver, acc) = if sampled {
        let lg = e.spec_verify_logits(i, &rows, t)?;
        let vocab = e.vocab();
        if lg.len() != t * vocab {
            return Err(format!(
                "spec(샘플) 로짓 크기 {} != {}",
                lg.len(),
                t * vocab
            ));
        }
        let mut acc = 0usize;
        for k in 0..draft.len() {
            let mut lk = lg[k * vocab..(k + 1) * vocab].to_vec();
            let sm = s.sampler.as_mut().unwrap();
            sm.apply_penalties_ext(&mut lk, &draft[..k]);
            let px = sm.prob_of(&lk, draft[k]);
            let coin = sm.next_coin();
            if coin < px {
                acc += 1;
            } else {
                break;
            }
        }
        (Vec::new(), 0.0, acc)
    } else {
        let t_ver = std::time::Instant::now();
        let toks = e.spec_verify(i, &rows, t)?;
        let ms_ver = t_ver.elapsed().as_secs_f64() * 1e3;
        let mut acc = 0usize;
        while acc < draft.len() && toks[acc] == draft[acc] {
            acc += 1;
        }
        if dbg {
            eprintln!(
                "[mtp-dbg] draft={draft:?} verify0={:?} acc={acc}",
                toks.first()
            );
            s.spec_miss = 0;
        }
        (toks, ms_ver, acc)
    };
    let spec_mtp = mtp_on;
    if spec_mtp && let Err(err) = e.mtp_pre_round(acc) {
        eprintln!("# mtp(무시): {err}");
    }
    let last = if acc >= 1 { draft[acc - 1] } else { s.next };
    let corr_by_decode = spec_mtp || sampled;
    let fused_keep9 = !corr_by_decode && acc + 1 > 8;
    if corr_by_decode || fused_keep9 {
        if acc >= 1 {
            e.spec_rollback(i, acc)?;
            e.spec_rewind_pos(i, (t - acc) as u32)?;
        } else {
            e.spec_rollback(i, 0)?;
            e.spec_rewind_pos(i, t as u32)?;
        }
    } else {
        e.spec_rollback(i, acc + 1)?;
        e.spec_rewind_pos(i, (t - acc - 1) as u32)?;
    }
    if acc == 0 {
        s.spec_miss = (s.spec_miss + 1).min(3);
    } else {
        s.spec_miss = 0;
    }
    if dbg && !sampled {
        s.spec_miss = 0;
    }
    if sampled {
        for &tk in &draft[..acc] {
            slot_emit(s, tk);
        }
        let lg1 = e.decode(&[i], &[last])?.pop().unwrap_or_default();
        let vocab = e.vocab();
        if lg1.len() != vocab {
            return Err(format!("spec(샘플) 보정 로짓 {} != {vocab}", lg1.len()));
        }
        let mut corr = {
            let mut l = lg1.clone();
            s.sampler.as_mut().unwrap().sample(&mut l)
        };
        if acc < draft.len() {
            let x = draft[acc];
            let mut guard = 0;
            while corr == x && guard < 64 {
                let mut l = lg1.clone();
                corr = s.sampler.as_mut().unwrap().sample(&mut l);
                guard += 1;
            }
            if corr == x {
                eprintln!("# spec(샘플): 보정 재추출 가드 소진 — corr == x 방출");
            }
        }
        slot_emit(s, corr);
        SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
        SCHED.spec_accepted.fetch_add(acc as u64, Ordering::Relaxed);
        econ_spec_sample(1, draft.len(), acc, rt0.elapsed().as_secs_f64() * 1e3);
        if dbg {
            eprintln!(
                "[spec-dbg] gen={} src=1 k={} t={} acc={} corr={} sampled=1",
                s.generated,
                draft.len(),
                t,
                acc,
                corr
            );
        }
        return Ok(true);
    }
    let mut ms_corr = 0.0f64;
    let corr = if spec_mtp || fused_keep9 {
        let t_corr = std::time::Instant::now();
        let d = e.decode_greedy(i, last)?;
        ms_corr = t_corr.elapsed().as_secs_f64() * 1e3;
        d
    } else {
        toks[acc]
    };
    if dbg {
        eprintln!(
            "[spec-dbg] gen={} src={} k={} t={} acc={} corr={}",
            s.generated, src, kk, t, acc, corr
        );
    }
    let t_mtp = std::time::Instant::now();
    if spec_mtp && let Err(err) = e.mtp_post_round(i, &draft[..acc], corr) {
        eprintln!("# mtp(무시): {err}");
    }
    let ms_mtp = t_mtp.elapsed().as_secs_f64() * 1e3;
    if dbg {
        eprintln!(
            "[spec-ms] verify={ms_ver:.2} corr={ms_corr:.2} mtp={ms_mtp:.2} tot={:.2}",
            rt0.elapsed().as_secs_f64() * 1e3
        );
    }
    for &tk in &draft[..acc] {
        slot_emit(s, tk);
    }
    slot_emit(s, corr);
    SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
    SCHED.spec_accepted.fetch_add(acc as u64, Ordering::Relaxed);
    SCHED
        .spec_ms
        .fetch_add(rt0.elapsed().as_micros() as u64, Ordering::Relaxed);
    econ_spec_sample(src, draft.len(), acc, rt0.elapsed().as_secs_f64() * 1e3);
    Ok(true)
}

pub(super) fn slot_emit(s: &mut Slot, t: u32) {
    if s.t_first.is_none() {
        s.t_first = Some(std::time::Instant::now());
    }
    s.next = t;
    s.tokens.push(t);
    s.generated += 1;
    SCHED.econ_gen.fetch_add(1, Ordering::Relaxed);
    s.err_count = 0;
    if let Some(sm) = &mut s.sampler {
        sm.push_tokens([t]);
    }
    if let Some(j) = &s.job
        && let Some(p) = &j.progress
        && p.send(ProgressFrame::Tok(t)).is_err()
    {
        s.cancelled = true;
    }
}
