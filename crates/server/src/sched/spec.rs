//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::ProgressFrame;
use super::econ::{econ_plan, econ_spec_sample};
use super::{SCHED, Slot};
use std::sync::atomic::Ordering;

/// [§7.1 2026-10-11] 스펙 라운드 — 초안(4-gram 최대 k밴드 {2,4,6,8}-1) 발화 시
/// 1회. Ok(true)=라운드 수행(토큰 방출), Ok(false)=초안 없음(종전 경로).
/// 값 보존: 검증 배치가 전 위치 토큰을 내고, 수용분+보정(t=1 재계산)이
/// 단독 디코드와 동일한 토큰열을 만든다(프로브 [A-1 완화 ①]과 동일 절차 —
/// 35B/27B 스펙 골든 접두가 판정). 초안 폭은 k밴드 EMA(2.A-④)가 결정.
/// [MTP-CHAIN §3.3] 초안 소스 정책(mtp/ngram)은 엔진 메서드
/// GpuEngine::mtp_drafts_on — 프리필 KV 구축 게이트와 단일 정의.
pub(super) fn spec_round(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    spec_round_impl(i, s, e, false)
}

/// [§7 2단계 2026-10-11] 비greedy 스펙 라운드 — target-only 거절샘플링.
/// 수용: coin < p(x_k)(prob_of — 후처리 체인 미러, 수용 초안 가산 페널티).
/// 보정: t=1 재계산 로짓의 표본 — 거절 시 잔차(≠x 재추출 = 조건분포 정확).
/// 방출 분포 = 표적 분포(기대 수용 = E[p]). 그리디 라운드와 롤백 구조 동일.
pub(super) fn spec_round_sampled(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    spec_round_impl(i, s, e, true)
}

/// [R6] 스펙 라운드 공용 골격 — 그리디(확정)/샘플드(거절샘플링) 단일 경로.
/// 게이트·hist 조립·초안·rows·상태 저장·롤백·게이트 갱신은 공유하고, 검증
/// 및 보정만 `sampled` 정책 분기(내부 로직·호출 순서·로그·주석 원문 유지 —
/// 동작 불변).
fn spec_round_impl(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
    sampled: bool,
) -> Result<bool, String> {
    // [§7.5 2026-10-11] 적응 게이트 — 연속 무수용 시 256토큰마다만 재시도.
    // 산문 벤치 실측(서빙 A/B): 초안이 발화하나 수용 0이 반복되면 검증 비용
    // (~0.85틱)이 순손실 — tg −1.6%·np4 −5.7%. 수용 시 즉시 해제.
    // [MTP 2026-10-12] MTP 초안(수용 80~82%) 편입 시 임계 3이 적정(실측:
    // 임계 1이면 첫 경계 미스로 256토큰 억제 → 42.7 t/s). 현재 MTP는
    // 1-초안이 순손실(런치 오버헤드 ~5.5ms/콜 — 그래프 캡처·MTP-2 후 재평가)
    // 이라 파킹 상태이므로 n-gram 기준 임계 1 유지.
    // [MTP-CHAIN §3.3 2026-10-12] 체인(호스트 왕복 0)으로 재귀 치환 후
    // MTP 재활성 — 임계는 서빙 A/B 재실측 전까지 1 유지(선행 실측 근거
    // 80~82%는 순차 경로 기준 — 체인 편입 후 재계측 필요).
    // [§7.5] 무수용 억제 — 연속 무수용 시 재시도 주기마다만 재시도(경제성 게이트와
    // 별개의 빠른 억제기: 수용 0이 반복되는 발화는 비용만 쓰고 수용이 없음).
    // [P5 2026-10-15 백오프] 주기 = miss 단계별 32/128/256. 근거(실측): 고정
    // 256은 요청 프리앰블(구조적 저수용 — 모델 자기 서술)의 미스 1~2개로
    // 스펙 황금지(복사·반복 구간) 시작 전에 요청을 잠가버렸다(512토큰 중
    // 라운드 2~3). 단계 백오프는 프리앰블 미스를 빠르게 벗어나게 하고,
    // 산문(연속 미스)은 종전 256으로 수렴 — 산문 튜닝(§7.5)의 안정 상태 불변.
    let retry_mask: u32 = match s.spec_miss {
        0 => 0,
        1 => 31,
        2 => 127,
        _ => 255,
    };
    if s.spec_miss >= 1 && (s.generated & retry_mask) != 0 {
        return Ok(false);
    }
    // 샘플드 정책 한정 — 비greedy는 샘플러 필수(종전 조건 그대로).
    if sampled && s.sampler.is_none() {
        return Ok(false);
    }
    // [2.A-④ 2026-10-13] 경제성 게이트 — 3택(mtp/ngram/평범). None = 평범.
    // 스펙 무손실 — 스위칭은 처리량만(토큰 불변). spec_miss 억제 뒤에 둔다
    // (억제된 라운드는 표본·프로브 시계를 소비하지 않게).
    let mtp_avail = !sampled && e.mtp_drafts_on();
    let Some((src, kk)) = econ_plan(mtp_avail) else {
        return Ok(false);
    };
    let mut hist: Vec<u32> = Vec::new();
    if let Some(j) = &s.job {
        hist.extend_from_slice(&j.tokens);
    }
    hist.extend_from_slice(&s.tokens);
    // [2.A-④] 소스·폭은 게이트가 결정(위 econ_plan) — MTP는 저장 초안(체인
    // K=4 고정), n-gram은 k밴드 {2,4,6,8} EMA. 초안은 검증이 정본 — 소스/폭
    // 교체는 정책 계층(무손실). 샘플드는 n-gram 전용(MTP 초안 없음).
    // [MTP-CHAIN §3.3] mtp_drafts_on=false(강제 ngram)면 생산(mtp_first/single/
    // post_round)·소비 모두 스킵(순수 n-gram 경로).
    let mtp_on = src == 0;
    let draft: Vec<u32> = if mtp_on {
        match e.mtp_take_draft(i) {
            Some(d) => d,
            None => llm170_core::spec::ngram_draft(&hist, 3, 48, kk - 1),
        }
    } else {
        llm170_core::spec::ngram_draft(&hist, 3, 48, kk - 1)
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
    // [dsave 대체 2026-10-14] 라운드 전 상태 저장 호출 폐지 — 롤백 재료는
    // WY 표현(검증 체인이 solve 실행)과 conv가 기록하는 라운드 전 링
    // (dring0)이다. 저장 비용·VRAM(dsave 27B/4슬롯 약 315MiB) 소멸.
    let dbg = llm170_diag::flag::on_nonzero("LLM170_MOE_DBG");
    // 검증 + 수용(acc) 판정 — 정책 분기.
    // 그리디: 확정 일치 접두(토큰열). 샘플드: target-only 거절샘플링 —
    // 수용: coin < p(x_k)(prob_of — 후처리 체인 미러, 수용 초안 가산 페널티).
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
            s.spec_miss = 0; // 진단: 게이트 우회(수용률 계측)
        }
        (toks, ms_ver, acc)
    };
    // [MTP] 롤백·보정 전에 검증 dres의 수용 접두를 초안기 hidden으로 스냅샷.
    // (그리디 경로 한정 — 샘플드는 종전대로 MTP 콜 없음.)
    // [2026-10-12 시도·기각] pre_round(acc+1)로 보정행까지 검증 dres에서
    // 푸시하면 디코드 제거 가능하다는 가설 — 실측 MTP-on 0/381 수용으로
    // 기각(검증 dres 행 acc ≠ 보정 hidden — 롤백/그래프 경로 차이). 디코드를
    // 유지한다(MTP 경로 한정 — n-gram/플레인은 융합 유지).
    let spec_mtp = mtp_on;
    if spec_mtp && let Err(err) = e.mtp_pre_round(acc) {
        eprintln!("# mtp(무시): {err}");
    }
    let last = if acc >= 1 { draft[acc - 1] } else { s.next };
    // [비결정성 수정 2026-10-13] 상태 재구성 계약 — 검증 행은 [s.next, d0..]이고,
    // 라운드 후 목표 상태 = "수용 draft 소비 완료 + 보정 토큰 대기"(pos = P+acc+1).
    //   · 보정 디코드가 있는 경로(MTP·샘플드): 디코드가 마지막 수용 draft를
    //     재소비/전진시키므로 종전 (acc, t−acc)가 맞다.
    //   · 융합 경로(ngram 그리디): 디코드가 없다 → keep=acc+1(수용 소비 포함)·
    //     rewind=t−acc−1로 직접 재구성해야 한다. 종전 (acc, t−acc)는 한 행
    //     뒤처져 후속 토큰을 오염시켰다(하네스 재현: 동일 요청 idx 83 분기).
    //   · acc+1>8(전량 수용 t=9): WY 링 상한(8) — 종전 경로 + 보정 디코드
    //     (반환값 = toks[acc] 동일 계약)로 폴백.
    let corr_by_decode = spec_mtp || sampled;
    let fused_keep9 = !corr_by_decode && acc + 1 > 8;
    if corr_by_decode || fused_keep9 {
        if acc >= 1 {
            e.spec_rollback(i, acc)?;
            e.spec_rewind_pos(i, (t - acc) as u32)?;
        } else {
            // [dsave 대체 2026-10-14] 전량 거절 복원 = keep=0 (m=0 재구성 +
            // dring0) — 종전 spec_load_state(dsave 정밀 스냅샷) 대체.
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
        s.spec_miss = 0; // 진단: 게이트 우회(수용률 계측) — 증가 뒤에 둬야 유효.
    }
    // [2.A-④ 2026-10-13] k 적응은 밴드 EMA가 대체(구 "전량 수용 +1/부진 4 복귀"
    // 휴리스틱 제거) — 수용·비용이 cost/token EMA에 함께 반영된다.
    // 보정 토큰 — 정책 분기.
    if sampled {
        // 거절 시 잔차(≠x 재추출 = 조건분포 정확). 방출 분포 = 표적 분포
        // (기대 수용 = E[p]). 그리디 라운드와 롤백 구조 동일.
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
        }
        slot_emit(s, corr);
        SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
        SCHED.spec_accepted.fetch_add(acc as u64, Ordering::Relaxed);
        // [2.A-④] 경제성 표본(샘플드 = ngram) — 라운드 비용/방출토큰.
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
    // 그리디 보정 — [2026-10-12] 보정 토큰 = 검증 행 acc의 argmax(toks[acc]) —
    // 종전 decode_greedy(last) 재계산은 **동일 값의 중복**이었다(라운드 30%).
    // 근거: t-그래프 행 acc의 토큰 = last(수용 접두 마지막)이고 그 다음
    // 토큰이 정확히 보정 토큰. 커널·순서 동일 = 비트동일(스펙 골든 계약).
    // acc=draft.len()(전량 수용)이면 toks[acc]도 검증 범위 안(길이 1+len).
    //
    // [MTP 한정 예외] MTP post_round는 보정 토큰의 **hidden 상태 전진**을
    // 요구한다(초안 체인의 입력) — 융합하면 dres/norm.dxn이 검증 상태에
    // 머물러 초안이 붕괴한다(실측: MTP-on 0/508 수용). MTP 활성 경로만
    // 디코드를 유지(토큰 값은 toks[acc]와 동일 — 부수효과가 계약).
    let mut ms_corr = 0.0f64;
    let corr = if spec_mtp || fused_keep9 {
        let t_corr = std::time::Instant::now();
        let d = e.decode_greedy(i, last)?;
        ms_corr = t_corr.elapsed().as_secs_f64() * 1e3;
        d
    } else {
        toks[acc]
    };
    // [진단 2026-10-13] 라운드 마커 — 비결정성 분기 지점 대조용(토큰 인덱스 포함).
    if dbg {
        eprintln!(
            "[spec-dbg] gen={} src={} k={} t={} acc={} corr={} tail={:?} draft={:?} verify={:?}",
            s.generated,
            src,
            kk,
            t,
            acc,
            corr,
            &s.tokens[s.tokens.len().saturating_sub(4)..],
            &draft[..draft.len().min(4)],
            &toks[..toks.len().min(4)]
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
    // [2.A-④] 경제성 표본 — 라운드 비용/방출토큰(수용 접두 + 보정 1).
    econ_spec_sample(src, draft.len(), acc, rt0.elapsed().as_secs_f64() * 1e3);
    Ok(true)
}

pub(super) fn slot_emit(s: &mut Slot, t: u32) {
    // [관측 2026-10-14] 첫 emit 시각 — TTFT 원료(큐 진입은 job.queued).
    if s.t_first.is_none() {
        s.t_first = Some(std::time::Instant::now());
    }
    s.next = t;
    s.tokens.push(t);
    s.generated += 1;
    // [2.A-④] 토큰 시계 — 경제성 게이트의 프로브 주기 판정용.
    SCHED.econ_gen.fetch_add(1, Ordering::Relaxed);
    s.err_count = 0; // QA-1: 성공 — 연속 실패 카운터 리셋
    if let Some(sm) = &mut s.sampler {
        sm.push_tokens([t]);
    }
    if let Some(j) = &s.job
        && let Some(p) = &j.progress
        && p.send(ProgressFrame::Tok(t)).is_err()
    {
        // SSE 수신자 소멸(클라이언트 절단) — 즉시 취소 표시
        s.cancelled = true;
    }
}
