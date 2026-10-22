use super::econ::{econ_mtp_maintain, econ_plain_sample};
use super::prefill::finish_slot;
use super::spec::{slot_emit, spec_round, spec_round_sampled};
use super::{Engine, Slot, slot_fail};

pub(super) fn slot_decode(
    slots: &mut [Slot],
    eng: &mut Engine,
    eos: u32,
) -> (Vec<usize>, bool, f64, f64) {
    let active: Vec<usize> = (0..slots.len())
        .filter(|&i| {
            slots[i].job.is_some()
                && slots[i].prefilled == slots[i].job.as_ref().unwrap().tokens.len()
        })
        .collect();
    let mut decoded = false;
    let mut dec_ms = 0f64;
    let mut fin_ms = 0f64;
    if !active.is_empty() {
        decoded = true;
        let _dt = std::time::Instant::now();
        llm170_diag::watchdog::bump();
        q35_decode(&mut eng.0, slots, &active, eos);
        dec_ms = _dt.elapsed().as_secs_f64() * 1e3;
        let _ft = std::time::Instant::now();
        for &i in &active {
            finish_slot(&mut slots[i], eng, i, eos);
        }
        fin_ms = _ft.elapsed().as_secs_f64() * 1e3;
    }
    (active, decoded, dec_ms, fin_ms)
}

fn q35_decode(e: &mut crate::gpu_engine::GpuEngine, slots: &mut [Slot], seqs: &[usize], eos: u32) {
    let (mut greedy, mut sampled): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for &i in seqs {
        if slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy())
            || slots[i].constraint.is_some()
        {
            sampled.push(i);
        } else {
            greedy.push(i);
        }
    }
    let spec_ok = e.spec_on();
    let mut g2: Vec<usize> = Vec::with_capacity(greedy.len());
    for &i in &greedy {
        if !spec_ok {
            g2.push(i);
            continue;
        }
        match spec_round(i, &mut slots[i], e) {
            Ok(true) => {}
            Ok(false) => g2.push(i),
            Err(err) => slot_fail(&mut slots[i], format!("spec: {err}")),
        }
    }
    let greedy = g2;
    let mut s2: Vec<usize> = Vec::with_capacity(sampled.len());
    for &i in &sampled {
        if !spec_ok {
            s2.push(i);
            continue;
        }
        match spec_round_sampled(i, &mut slots[i], e) {
            Ok(true) => {}
            Ok(false) => s2.push(i),
            Err(err) => slot_fail(&mut slots[i], format!("spec(샘플): {err}")),
        }
    }
    let sampled = s2;
    if !greedy.is_empty() && !sampled.is_empty() {
        let mut uslots: Vec<usize> = Vec::with_capacity(greedy.len() + sampled.len());
        let mut utoks: Vec<u32> = Vec::with_capacity(uslots.capacity());
        for &i in greedy.iter().chain(sampled.iter()) {
            uslots.push(i);
            utoks.push(slots[i].next);
        }
        let _pmm = std::time::Instant::now();
        match e.decode_np_mixed(&uslots, &utoks) {
            Ok((toks, mut flat)) => {
                econ_plain_sample(_pmm.elapsed().as_secs_f64() * 1e3, uslots.len());
                let hn = flat.len() / uslots.len();
                for (row, (chunk, &i)) in flat.chunks_mut(hn).zip(uslots.iter()).enumerate() {
                    let smp = slots[i].constraint.is_some()
                        || slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
                    let t = if smp {
                        pick(&mut slots[i], chunk)
                    } else {
                        toks[row]
                    };
                    slot_emit(&mut slots[i], t);
                }
                return;
            }
            Err(err) => eprintln!("# 혼합 배치 폴백(분리): {err}"),
        }
    }
    if greedy.len() >= 2 {
        let toks: Vec<u32> = greedy.iter().map(|&i| slots[i].next).collect();
        let _pt = std::time::Instant::now();
        match e.decode_np_greedy(&greedy, &toks) {
            Ok(ts) => {
                econ_plain_sample(_pt.elapsed().as_secs_f64() * 1e3, greedy.len());
                for (row, &i) in greedy.iter().enumerate() {
                    slot_emit(&mut slots[i], ts[row]);
                }
            }
            Err(err) => {
                eprintln!("# np-greedy 실패({err}) — 이번 회차 건너뜀");
                for &i2 in &greedy {
                    slot_fail(&mut slots[i2], format!("decode_np_greedy: {err}"));
                }
            }
        }
    } else if let Some(&i) = greedy.first() {
        const KSTEP_K: usize = 4;
        let rem = slots[i]
            .job
            .as_ref()
            .map(|j| {
                j.n_predict
                    .max(1)
                    .saturating_sub(slots[i].generated as usize)
            })
            .unwrap_or(0);
        let k = KSTEP_K.min(rem);
        let _pt = std::time::Instant::now();
        let kres = if k >= 2 {
            e.decode_greedy_k(i, slots[i].next, k, eos)
        } else {
            Err("kstep: k<2".into())
        };
        match kres {
            Ok(toks) => {
                econ_plain_sample(_pt.elapsed().as_secs_f64() * 1e3, toks.len());
                for t in toks {
                    slot_emit(&mut slots[i], t);
                }
                if e.mtp_drafts_on()
                    && econ_mtp_maintain()
                    && let Err(err) = e.mtp_single(i, slots[i].next)
                {
                    eprintln!("# mtp(무시): {err}");
                }
            }
            Err(_) => {
                let _pt2 = std::time::Instant::now();
                match e.decode_greedy(i, slots[i].next) {
                    Ok(t) => {
                        econ_plain_sample(_pt2.elapsed().as_secs_f64() * 1e3, 1);
                        if e.mtp_drafts_on()
                            && econ_mtp_maintain()
                            && let Err(err) = e.mtp_single(i, t)
                        {
                            eprintln!("# mtp(무시): {err}");
                        }
                        slot_emit(&mut slots[i], t);
                    }
                    Err(err) => slot_fail(&mut slots[i], format!("decode_greedy: {err}")),
                }
            }
        }
    }
    if !sampled.is_empty() {
        let toks: Vec<u32> = sampled.iter().map(|&i| slots[i].next).collect();
        let _plg = std::time::Instant::now();
        match e.decode_np_logits(&sampled, &toks) {
            Ok(mut flat) => {
                econ_plain_sample(_plg.elapsed().as_secs_f64() * 1e3, sampled.len());
                let hn = flat.len() / sampled.len();
                let mut off = 0usize;
                for &i in &sampled {
                    let (row, _rest) = flat[off..].split_at_mut(hn);
                    let t = pick(&mut slots[i], row);
                    slot_emit(&mut slots[i], t);
                    off += hn;
                }
            }
            Err(err) => {
                eprintln!("# decode 실패({err}) — 이번 회차 건너뜀");
                for &i2 in &sampled {
                    slot_fail(&mut slots[i2], format!("decode: {err}"));
                }
            }
        }
    }
}

pub(super) fn pick(s: &mut Slot, logits: &mut [f32]) -> u32 {
    if let Some(c) = &mut s.constraint {
        let tok = crate::engine::TOKENIZER.get();
        let vocab = logits.len() as u32;
        let allowed: Vec<u32> = match tok {
            Some(t) => c
                .allowed(|id| t.piece(id).unwrap_or("").to_string(), vocab)
                .to_vec(),
            None => Vec::new(),
        };
        if allowed.is_empty() {
            return s
                .job
                .as_ref()
                .and_then(|j| j.stops.first().copied())
                .unwrap_or(0);
        }
        let mut best = f32::NEG_INFINITY;
        let mut bt = allowed[0];
        for &t in &allowed {
            let v = logits[t as usize];
            if v > best {
                best = v;
                bt = t;
            }
        }
        if let Some(t) = tok
            && let Some(txt) = t.piece(bt)
        {
            c.commit(txt);
        }
        return bt;
    }
    match &mut s.sampler {
        Some(sm) if !sm.is_greedy() => sm.sample(logits),
        _ => llm170_core::qwen35::greedy(logits),
    }
}
