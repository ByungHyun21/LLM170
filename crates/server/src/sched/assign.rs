use super::{SCHED, Slot, SlotJob};
use crate::engine::{Engine, InferResult};
use std::sync::atomic::Ordering;

pub(super) fn assign_slot(
    slots: &mut [Slot],
    eng: &mut Engine,
    j: SlotJob,
    tick: u64,
    ptrie: &mut crate::tree::PrefixTrie,
) {
    SCHED.requests.fetch_add(1, Ordering::Relaxed);
    SCHED
        .queue_wait_us
        .fetch_add(j.queued.elapsed().as_micros() as u64, Ordering::Relaxed);
    if j.tokens.is_empty() {
        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
        let _ = j.out.send(InferResult {
            tokens: Vec::new(),
            error: Some("empty prompt".into()),
        });
        return;
    }
    SCHED.jobs.fetch_add(1, Ordering::Relaxed);
    let (i, l) = if let Some((ti, _tl)) = ptrie.longest_prefix(&j.tokens)
        && ti < slots.len()
        && slots[ti].job.is_none()
        && !slots[ti].cached.is_empty()
    {
        let l = slots[ti]
            .cached
            .iter()
            .zip(j.tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        (ti, l)
    } else {
        let Some((i, l)) = pick_prefix_slot(slots, &j.tokens) else {
            return;
        };
        (i, l)
    };
    let epos = eng.0.seq_pos(i) as usize;
    let l = l.min(epos).min(j.tokens.len().saturating_sub(1));
    let reok = eng.0.reuse_valid(i, l, j.tokens.len());
    let full = reok && l > 0 && l == slots[i].cached.len() && l == epos;
    let mut reuse = 0usize;
    if full {
        reuse = l;
        SCHED
            .prefix_tokens
            .fetch_add(reuse as u64, Ordering::Relaxed);
        SCHED.prefix_full_hits.fetch_add(1, Ordering::Relaxed);
        eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(전체)");
    } else if reok && l > 0 {
        match eng.0.restore_prefix(i, l as u32) {
            Ok(p) if p > 0 && p as usize <= l => {
                reuse = p as usize;
                SCHED
                    .prefix_tokens
                    .fetch_add(reuse as u64, Ordering::Relaxed);
                SCHED.prefix_ckpt_hits.fetch_add(1, Ordering::Relaxed);
                eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(부분)");
            }
            _ => {}
        }
    }
    if reuse == 0 {
        eng.reset_seq(i);
    }
    let need_tokens = super::admit::need_tokens(j.tokens.len(), j.n_predict);
    if let Err(e) = eng.0.kv_grow(i, need_tokens) {
        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
        let _ = j.out.send(InferResult {
            tokens: Vec::new(),
            error: Some(format!("kv 입장 실패: {e}")),
        });
        let freed = eng.0.kv_release_slot(i);
        let c = if freed > 0 {
            Vec::new()
        } else {
            std::mem::take(&mut slots[i].cached)
        };
        slots[i] = Slot::free();
        slots[i].cached = c;
        return;
    }
    if reuse > 0 {
        eng.0.mtp_slot_reused(i);
    } else {
        eng.0.mtp_slot_fresh(i);
        if let Err(err) = eng.0.mtp_slot_kv_reset() {
            eprintln!("# mtp(무시): {err}");
        }
    }
    let prev = std::mem::take(&mut slots[i].cached);
    let mut sampler_new = j.sampler.clone().map(llm170_core::sampler::Sampler::new);
    if let Some(sm) = &mut sampler_new
        && !sm.is_greedy()
    {
        sm.push_tokens(j.tokens.iter().copied());
    }
    let constraint = j
        .constraint
        .as_ref()
        .map(crate::oai::schema::Constraint::new);
    slots[i] = Slot {
        job: Some(j),
        prefilled: reuse,
        next: 0,
        generated: 0,
        tokens: Vec::new(),
        touch: tick,
        cancelled: false,
        cached: prev,
        sampler: sampler_new,
        err_count: 0,
        failed: None,
        t_first: None,
        spec_miss: 0,
        constraint,
    };
}

pub(super) fn pick_prefix_slot(slots: &[Slot], tokens: &[u32]) -> Option<(usize, usize)> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.job.is_none())
        .map(|(i, s)| {
            let l = s
                .cached
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            (i, l)
        })
        .max_by_key(|&(i, l)| (l, std::cmp::Reverse(i)))
}

#[cfg(test)]
mod tests {
    use super::super::Slot;
    use super::pick_prefix_slot;

    fn slot_with(cached: Vec<u32>) -> Slot {
        let mut s = Slot::free();
        s.cached = cached;
        s
    }

    #[test]
    fn pick_none_cases() {
        assert!(pick_prefix_slot(&[], &[1, 2, 3]).is_none());
        let mut busy = slot_with(vec![1, 2]);
        busy.job = None;
        let free = slot_with(Vec::new());
        assert_eq!(pick_prefix_slot(&[free], &[1, 2, 3]), Some((0, 0)));
    }

    #[test]
    fn pick_longest_prefix() {
        let a = slot_with(vec![1, 2, 3, 4]);
        let b = slot_with(vec![1, 2, 9]);
        let c = slot_with(vec![7, 8]);
        assert_eq!(pick_prefix_slot(&[a, b, c], &[1, 2, 3, 4, 5]), Some((0, 4)));
    }

    #[test]
    fn pick_exact_length_full_prefix() {
        let a = slot_with(vec![1, 2, 3]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 3)));
    }

    #[test]
    fn pick_no_prefix_still_slot() {
        let a = slot_with(vec![9, 9]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 0)));
    }
}
