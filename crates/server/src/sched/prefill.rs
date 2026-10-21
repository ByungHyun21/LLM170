//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::decode::pick;
use super::econ::econ_mtp_maintain;
use super::spec::slot_emit;
use super::{
    Engine, InferResult, PF_BUDGET_TOKENS, PF_CHUNK_MAX, PF_CHUNK_MIN, PF_LONG_PROMPT,
    ProgressFrame, SCHED, Slot, slot_fail,
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
    // [§2] 혼합 — 타 슬롯이 이미 디코드 중(프리필 완료+잔존)이면 청크 캡으로
    // 디코드 정지 시간을 유계화(프리필 총량은 동일). 캡 = 엔진별 시간 등급
    // [파이프라인 2026-10-22]: dense 27B 128토큰(≈102ms) 기준, MoE 상주 35B는
    // 512토큰(≈110ms) 전폭 — 슬롯 8 상향 후 혼합 구간 증가로 128 캡의
    // 청크 수 4배·디코드 틱 직렬화가 서비스 시간을 부풀렸다(C=8 실측).
    // 디코드 중 슬롯(prefilled==len)과 프리필 후보(prefilled<len)는 서로
    // 배타라 후보 공통 상수로 호이스팅해도 동일하다.
    let others_decoding = slots
        .iter()
        .any(|s| s.job.is_some() && s.prefilled == s.job.as_ref().unwrap().tokens.len());
    let cap_now = if others_decoding {
        eng.0.prefill_mixed_cap()
    } else {
        PF_CHUNK_MAX
    };
    // [파이프라인 2026-10-22] 완료-우선 선택 — 이번 틱에 끝낼 수 있는 슬롯
    // (잔여 ≤ 현재 청크 상한)을 라운드로빈보다 먼저 뽑는다. 종전 순수 touch
    // 라운드로빈에서 접두 히트 슬롯의 2토큰 접미 청크가 콜드 512청크들 뒤로
    // 밀려 TTFT 150ms→900ms가 되는 사례(27B ttft_harness C=2 — 첫 토큰 emit은
    // 최종 청크에서만 일어남). 잔여>상한 슬롯은 종전대로 touch 순서(기아 없음 —
    // 긴 프롬프트도 잔여가 상한 이하로 줄면 완료-우선 대상이 된다).
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
    if others_decoding {
        chunk = chunk.min(cap_now);
    }
    let chunk = chunk.min(left);
    // [파이프라인 2026-10-22] greedy 슬롯 = 디바이스 argmax 프리필 — 최종 청크
    // 토큰 1개(4B)만 D2H. 종전 greedy는 로짓 전사 D2H(어휘×4B ≈ 600KB
    // pageable — 슬로패스 수 ms) + 호스트 argmax를 요청마다 치렀다(35B 서빙
    // nsys: 로짓 D2H API 163ms/6회). 토큰 불변 = w4a16_argmax_min(min-index
    // tie)은 greedy_from 미러 — 디코드 greedy와 동일 계약(엔진별 softcap
    // 가드는 gpu_engine/qwen.rs prefill_greedy 주석).
    // 샘플링/제약 슬롯은 로짓 판이 필요(마지막 청크만 판정) — 기존 prefill.
    let (start, logits) = {
        let end = (slots[i].prefilled + chunk).min(slots[i].job.as_ref().unwrap().tokens.len());
        let part: Vec<u32> =
            slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
        let samp = slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
        // [P8] 제약 슬롯은 prefill 마지막 로짓도 pick(마스킹) 경유 — 첫
        // 토큰부터 스키마 강제(실측: 우회 시 1토큰 프리앰블 새는 결함).
        let cons = slots[i].constraint.is_some();
        let r: Result<u32, String> = if end < slots[i].job.as_ref().unwrap().tokens.len() {
            // [⑰-1 2026-10-18] 중간 청크 = 헤드·로짓 d2h 스킵(낭비 제거).
            // 최종 청크만 로짓 판정(emit 토큰). KV·MTP 전진 동일 → 토큰열 불변.
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
        // 마커 = 백엔드 단일 출처(CTX_OVERFLOW) — 메시지 변경에도 분기 유지.
        // (청크 프리필 중간 실패(prefilled>0)는 통폐기 — 이미 소비된 청크도
        // 유효 접두지만 상태 신뢰 우선. 부분 히트 개선 후보로 기록만.)
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
            // [관측 2026-10-14] 요청별 TTFT/처리량 — ttft = 큐 진입→첫 토큰.
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
