//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::decode::pick;
use super::econ::econ_mtp_maintain;
use super::spec::slot_emit;
use super::{
    Engine, InferResult, PF_BUDGET_TOKENS, PF_CHUNK_MAX, PF_CHUNK_MIN, PF_CHUNK_MIXED,
    PF_LONG_PROMPT, ProgressFrame, SCHED, Slot, slot_fail,
};
use std::sync::atomic::Ordering;

/// [R7] 틱 ③ 프리필 1청크(슬롯 루프에서 순수 이동). 반환 Some(ms) = 청크
/// 완료(호출부 n_pf·ms_pf 가산), None = 클라이언트 절단 — 종전 continue
/// 경로(잔여 카운터·finish 생략) 그대로.
pub(super) fn slot_prefill(
    eng: &mut Engine,
    slots: &mut [Slot],
    active: &[usize],
    tick: u64,
    eos: u32,
    npw: bool,
    t0w: &std::time::Instant,
) -> Option<f64> {
    let pf = (0..slots.len())
        .filter(|&i| {
            slots[i].job.is_some()
                && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
        })
        .min_by_key(|&i| slots[i].touch);
    let i = pf?;
    let _pft = std::time::Instant::now();
    llm170_diag::watchdog::bump();
    // [B5] 가변 청크 — 잔여 예산(디코드 차감) + 경합 시 long 캡.
    let left = slots[i].job.as_ref().unwrap().tokens.len() - slots[i].prefilled;
    let backlog = slots
        .iter()
        .filter(|s| s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len())
        .count();
    let mut chunk = PF_BUDGET_TOKENS
        .saturating_sub(active.len())
        .clamp(PF_CHUNK_MIN, PF_CHUNK_MAX);
    if backlog > 1 && left > PF_LONG_PROMPT {
        chunk = chunk.min(PF_CHUNK_MAX / 2);
    }
    // [§2] 혼합 — 타 슬롯이 이미 디코드 중(프리필 완료+잔존)이면
    // 청크 캡으로 디코드 정지 시간을 유계화(프리필 총량은 동일).
    let others_decoding = slots.iter().enumerate().any(|(j, s)| {
        j != i && s.job.is_some() && s.prefilled == s.job.as_ref().unwrap().tokens.len()
    });
    if others_decoding {
        chunk = chunk.min(PF_CHUNK_MIXED);
    }
    let chunk = chunk.min(left);
    // Q4(FN)는 prefill_greedy — 청크마다 어휘 152k
    // 로짓 pageable D2H(슬로패스 수십 ms) 대신 GPU argmax 8B 회수.
    // Q35(27B)는 종전 전사 경로(원시 프리필 내부 d2h).
    let (start, logits) = {
        let end = (slots[i].prefilled + chunk).min(slots[i].job.as_ref().unwrap().tokens.len());
        let part: Vec<u32> =
            slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
        // 샘플링 슬롯은 로짓 판(마지막 청크만 판정에 사용) — Q4도
        // prefill_greedy 대신 prefill. greedy는 종전 최적 경로.
        let samp = slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
        let r: Result<u32, String> = eng.0.prefill(i, &part).map(|mut l| {
            if samp {
                pick(&mut slots[i], &mut l)
            } else {
                llm170_core::qwen35::greedy(&l)
            }
        });
        (end, r)
    };
    // [B5] 라운드로빈 — 방금 전진한 슬롯을 최신으로 갱신, 다음
    // 청크는 더 오래 기다린(pending) 슬롯이 받는다. 종전엔 touch가
    // 배정 시점에만 갱신돼 한 슬롯이 전량 프리필을 독점했다.
    slots[i].touch = tick;
    // [K0 2026-10-12] 프리필 중 절단 감지 — 청크마다 Ping 프로브로
    // 수신자 생존 확인(keep-alive 타임아웃까지 대기 없이 즉시).
    // 절단이면 잔여 프리필을 버리고 즉시 반납(캐시 폐기 — 절단된
    // 요청의 상태 신뢰 불가). 종전엔 프리필 완료 후에야 done 판정이
    // cancelled를 봐 긴 프리필이 슬롯을 끝까지 점유했다(체감 지연).
    if let Some(p) = slots[i].job.as_ref().and_then(|j| j.progress.as_ref())
        && p.send(ProgressFrame::Ping).is_err()
    {
        if let Some(j) = slots[i].job.take() {
            // 수신측 이미 절단 — out도 drop 상태라 전송은 최선만.
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
                // [MTP] 프리필 완료 — 유예 호출(prev hidden, 첫
                // 토큰)로 첫 초안 생성(다음 틱 스펙 라운드 소비).
                if eng.0.mtp_drafts_on()
                    && econ_mtp_maintain()
                    && let Err(err) = eng.0.mtp_first(i, t)
                {
                    eprintln!("# mtp(무시): {err}");
                }
            }
        }
        // QA-1: 프리필 실패 적립 — 종전엔 무시돼 prefilled가 영구
        // 갱신되지 않는 스피너였다(매 틱 동일 청크 재시도).
        Err(err) => slot_fail(&mut slots[i], format!("prefill: {err}")),
    }
    finish_slot(&mut slots[i], eng, i, eos);
    Some(ms)
}

pub(super) fn finish_slot(s: &mut Slot, eng: &mut Engine, i: usize, eos: u32) {
    // QA-1: 확정 실패 슬롯 — 에러 결과 전파 후 반환. 종전엔 토큰 방출이
    // 없어 완료 조건이 영구 거짓 → 서버 수명 동안 재시도하는 스피너였다.
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
        // [K4 2026-10-12] 용량 초과(사전 검증 에러·프리필 미소비) — 엔진
        // 상태는 이전 cached와 정합이므로 캐시를 버리지 않는다(재요청
        // 접두 히트 유지). 그 외 런타임 에러는 현행 폐기(신뢰 불가).
        if err.contains("context overflow") && s.prefilled == 0 && !s.cached.is_empty() {
            let keep = std::mem::take(&mut s.cached);
            *s = Slot::free();
            s.cached = keep;
            return;
        }
        eng.reset_seq(i);
        *s = Slot::free(); // 접두 캐시 폐기 — 엔진 시퀀스 상태 신뢰 불가
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
            // 접두 캐시 — 상태 유지 (프롬프트+생성 = 구워진 열).
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
