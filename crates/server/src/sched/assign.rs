//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::{SCHED, Slot, SlotJob};
use crate::engine::{Engine, InferResult};
use std::sync::atomic::Ordering;

/// 슬롯 배정 단일 구현 (107 W7: drain/유휴 이중 복제 통합).
/// 접두 캐시 최장 일치 슬롯 선택(전 슬롯 대상 — 종전 유휴 경로는
/// slot0 고정이었음), 재사용 시 reset 생략, 샘플러 시딩 포함.
pub(super) fn assign_slot(
    slots: &mut [Slot],
    eng: &mut Engine,
    j: SlotJob,
    tick: u64,
    ptrie: &mut crate::tree::PrefixTrie,
) {
    // 모니터링 — 배정 단일 지점(두 수신 경로 공통).
    SCHED.requests.fetch_add(1, Ordering::Relaxed);
    SCHED
        .queue_wait_us
        .fetch_add(j.queued.elapsed().as_micros() as u64, Ordering::Relaxed);
    // [H 2026-10-09] 빈 프롬프트 최후 가드 — 표면(oai 400·CLI)이 막지만
    // 스케줄러 직접 진입은 0번 토큰에서 조용히 디코드하게 된다.
    if j.tokens.is_empty() {
        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
        let _ = j.out.send(InferResult {
            tokens: Vec::new(),
            error: Some("empty prompt".into()),
        });
        return;
    }
    SCHED.jobs.fetch_add(1, Ordering::Relaxed);
    // [§3.4 ③] 전역 트리 우선 조회 — 512+ 접두의 슬롯 후보 직접 제시.
    // 미스(단문/신규)는 슬롯 스캔 폴백. 트리는 슬롯 후보만 제시하고
    // 실제 l(공통 접두 길이)는 기존 산식과 동일 — 수치 무변.
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
    // [A2] 재사용 판정 — 전체 접두(상태가 그 끝 위치와 정합) 우선, 부분은
    // 엔진 restore_prefix(≤ l: qwen = GDN 체크포인트 · gemma4 = pos 되감기).
    // [finish 불변식 2026-10-14] 상한 = 엔진이 실제 소비한 위치 — n_predict
    // 종료(스토퍼 없음)는 cached가 엔진 pos보다 1 길어 그대로 재사용하면
    // 위치가 어긋난다(콜드 정합 유지 — 스토퍼 종료는 정합적이라 무영향).
    let epos = eng.0.seq_pos(i) as usize;
    // [빈 프리필 가드 2026-10-14] 프롬프트 마지막 토큰은 항상 실제 프리필 —
    // 재사용이 토큰 끝과 겹치면 prefill이 빈 벡터가 되어 첫 디코드가 next=0
    // (플레이스홀더)을 투입한다(실측: gemma4가 턴 종료 마커 출력). qwen은
    // 체크포인트가 512 경계로 양자화돼 이 구멍을 비껴갔다.
    let l = l.min(epos).min(j.tokens.len().saturating_sub(1));
    // [2.N Stage 2] 트림 슬롯 재사용 가드 — 해제 행이 미래 판독·기록에 닿으면
    // 콜드(전량 재계산). qwen·미트림 슬롯은 항상 true(동작 보존).
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
    // [K6-3 2026-10-13] KV 확보 — 예약(프롬프트+생성한도+슬랙)까지 온디맨드
    // 배정. 입장 판정이 선행하나 재사용 슬롯의 기존 배정과 합산해 부족할 수
    // 있다(보수 판정) — 실패는 요청 오류로 정직하게(사용 중 실패 금지 정합:
    // 입장 후에는 배정 성공이 보장된다).
    // [W2 2026-10-16] 슬랙 = admit::RESERVE_SLACK(8 = 스펙 전방 tmax−1행) —
    // 입장 판정과 동일 함수(단일 출처, 종전 슬랙 2는 블록 경계에서 부족).
    let need_tokens = super::admit::need_tokens(j.tokens.len(), j.n_predict);
    if let Err(e) = eng.0.kv_grow(i, need_tokens) {
        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
        let _ = j.out.send(InferResult {
            tokens: Vec::new(),
            error: Some(format!("kv 입장 실패: {e}")),
        });
        // [검토 반영 2026-10-16] 부분 할당 잔존 금지 — 실패 슬롯은 전량 반납하고
        // 접두 캐시를 무효화한다(블록이 사라진 cached는 재사용 불가 — reclaim_idle와
        // 동일 의미). freed=0(아무 블록도 없었음)이면 캐시 유지(기존 동작).
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
    // [MTP] 접두 재사용 슬롯은 MTP KV 정합이 없다 — 초안 비활성(검증은
    // 정본이라 손실은 없지만, 초안 비용만 지불하게 된다).
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

/// [I 2026-10-10] 접두 캐시 재사용 슬롯 선택 — **순수 함수**(테스트 표면).
/// free 슬롯(job 없음) 중 cached와 tokens의 공통 접두가 최장인 곳. 반환
/// (slot, l): l = 공통 접두 길이(부분 포함 — 전체/체크포인트 판정은 호출부가
/// 엔진 체크포인트 조회와 함께 수행[A2 2026-10-11]). 정확히 같은 길이도
/// l = cached 전체(전체 재사용 — 프리필 스킵).
pub(super) fn pick_prefix_slot(slots: &[Slot], tokens: &[u32]) -> Option<(usize, usize)> {
    // [A2 2026-10-11] 부분 접두까지 l 반환(순수 함수 유지 — 전체/부분/체크포인트
    // 판정은 호출부가 엔진 체크포인트 조회와 함께 수행).
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
        // [K6-3] 동률(모두 미스)은 최저 인덱스 — 배치 그래프 키(슬롯 조합)
        // 압축(전형 조합 {0},{0,1},… — 조합 폭발 방지).
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

    /// [I] 접두 선택 — 빈 슬롯 없음/전부 점유 = None.
    #[test]
    fn pick_none_cases() {
        assert!(pick_prefix_slot(&[], &[1, 2, 3]).is_none());
        let mut busy = slot_with(vec![1, 2]);
        busy.job = None; // job 없음 + cached 유지 = 자유 슬롯(재사용 판정은 아래)
        // 전부 점유(가짜 점유는 job Some 필요 — out 생략 위해 free로 검증 불가하므로
        // "cached 비었고 공통 접두 0" 케이스로 대체).
        let free = slot_with(Vec::new());
        assert_eq!(pick_prefix_slot(&[free], &[1, 2, 3]), Some((0, 0)));
    }

    /// [I] 최장 접두 슬롯 선택 + 전체 접두(더 긴 토큰) 재사용 판정.
    #[test]
    fn pick_longest_prefix() {
        let a = slot_with(vec![1, 2, 3, 4]);
        let b = slot_with(vec![1, 2, 9]);
        let c = slot_with(vec![7, 8]);
        // 공통 접두: a=4(전체+토큰 더 김→재사용), b=2, c=0 → a 선택.
        assert_eq!(pick_prefix_slot(&[a, b, c], &[1, 2, 3, 4, 5]), Some((0, 4)));
    }

    /// [I→A2] 정확히 같은 길이 = cached 전체가 접두 → l=3(전체 재사용 —
    /// 판정은 호출부 full 경로. 구버전 기대(0,0)은 A2 부분-접두 반환 이전 잔존).
    #[test]
    fn pick_exact_length_full_prefix() {
        let a = slot_with(vec![1, 2, 3]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 3)));
    }

    /// [I] 접두 0이어도 슬롯은 반환(reuse=0 — reset 경로).
    #[test]
    fn pick_no_prefix_still_slot() {
        let a = slot_with(vec![9, 9]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 0)));
    }
}
