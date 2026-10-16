//! sched.rs에서 분해(순수 이동 — 코드 무변).
use super::econ::{econ_mtp_maintain, econ_plain_sample};
use super::prefill::finish_slot;
use super::spec::{slot_emit, spec_round, spec_round_sampled};
use super::{Engine, Slot, slot_fail};

/// [R7] 틱 ② 디코드 우선 — prefill 완료 슬롯 전부(슬롯 루프에서 순수 이동).
/// 반환 = (활성 슬롯, 디코드 여부, 디코드 ms, 완료 처리 ms) — 호출부 계측
/// 가산에 그대로 쓴다.
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
        // 완료 슬롯 정리 — 결과 전송·반환
        let _ft = std::time::Instant::now();
        for &i in &active {
            finish_slot(&mut slots[i], eng, i, eos);
        }
        fin_ms = _ft.elapsed().as_secs_f64() * 1e3;
    }
    (active, decoded, dec_ms, fin_ms)
}

/// Q35 np 디코드 — 샘플링 슬롯 포함시 logits 경로(decode), 아니면 GPU argmax 판.
/// [A9-3 2026-10-10] 혼합 워크로드: 그리디 슬롯은 배치(가중 상각), 샘플링
/// 슬롯만 직렬 — 슬롯별 상태 독립이라 순서 무관. 그리디가 1개 이하면 종전 직렬.
/// [K-STEP 2026-10-12] eos 인자 추가 — 단독 그리디 K스텝 장치 루프가
/// 컨트롤러 eos(캡처 키)로 쓴다.
fn q35_decode(e: &mut crate::gpu_engine::GpuEngine, slots: &mut [Slot], seqs: &[usize], eos: u32) {
    // 그리디/샘플링 분할.
    let (mut greedy, mut sampled): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for &i in seqs {
        // [P8] 제약 슬롯 = 로그짓 판 경로(pick이 마스킹). GPU argmax/스펙은
        // 마스크를 모른다.
        if slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy())
            || slots[i].constraint.is_some()
        {
            sampled.push(i);
        } else {
            greedy.push(i);
        }
    }
    // [§7.1 2026-10-11] 스펙 서빙 — 초안이 발화하는 그리디 슬롯은 검증
    // 라운드(직렬). 값 보존: 검증+보정 토큰이 t=1 디코드와 비트동일
    // (스펙 골든 계약 — spec_round 주석 참조). 나머지는 종전 배치/단독.
    // [2026-10-12] 스펙 비활성(스트리밍·VRAM 부족)이면 라운드 시도 금지 —
    // 비활성 상태 시도가 요청을 죽이던 결함(실측: 스트리밍 서브 500).
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
    // [§7 2단계] 샘플드 스펙 — 초안 발화 슬롯은 검증 라운드(직렬), 나머지 배치.
    // [S11 2026-10-11] 혼합 통합을 위해 그리디 배치보다 먼저 처리(슬롯 독립 —
    // 순서 무관).
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
    // [S11] 혼합(그리디+샘플) — 한 배치 체인으로 통합(종전 np_k + np_m 2회).
    // 실패 시 분리 경로로 폴백.
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
                    // [P0-D1 2026-10-16] 제약 슬롯도 pick 경유 — 혼합 배치에서
                    // GPU argmax(toks[row])가 마스크를 우회해 스키마를 위반했다
                    // (이슈 문서 D1·np2 혼합 재현 조건).
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
        // [E-1 후속 2026-10-10] 단독 그리디는 **GPU argmax**(4B d2h) —
        // 종전 logits 경로 합류는 토큰당 로짓 993KB pageable D2H + CPU
        // argmax(248K)를 지불했다(실측 ~0.5ms/토큰). 샘플링 슬롯만 logits 판.
        // [K-STEP §3.4 ① 2026-10-12] 먼저 K스텝 장치 루프(스텝 그래프 k회
        // 리플레이 — 호스트 체크인 K스텝마다)를 시도한다. k = min(K, 잔여
        // max_tokens) — cap 잔여는 엔진이 클램프(2 미만 Err). EOS는 장치
        // 컨트롤러가 감지(이후 스텝 미기록)하므로 방출 순서·값은 순차 단일
        // 스텝과 동일(토큰열 판정). 계약·캡처 실패는 기존 토큰당 경로 폴백.
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
                // 방출 루프 — 슬롯 상태(tokn·generated·SSE 프레임)는 기존
                // 틱당 방출과 동일 경로로 갱신(eos가 로그 끝에 최대 1회).
                for t in toks {
                    slot_emit(&mut slots[i], t);
                }
                // [MTP] 비-스펙 단독 디코드 후 초안 생성(마지막 방출 토큰 기준
                // 1회 — 순차 경로의 토큰당 호출과 같은 상태·입력).
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
                        // [MTP] 비-스펙 단독 디코드 후 초안 생성(다음 라운드 소비).
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
        // [S11 2026-10-11] t≥2 샘플링은 배치 체인 1회(로짓 d2h) — 종전
        // 슬롯별 t=1 직렬(≈19.5ms×N) 대비 대폭 상각. 로짓은 행별 t=1 head와
        // 비트동일이라 호스트 샘플러 토큰열 불변. t=1은 종전 직렬 유지.
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

/// 슬롯 로짓 → 토큰: 활성 샘플러면 sample, 아니면 greedy (동률 최저 인덱스).
pub(super) fn pick(s: &mut Slot, logits: &mut [f32]) -> u32 {
    // [P8 2026-10-16] json_schema 제약 — 허용 토큰 집합으로 마스킹 후 greedy.
    // 완결 시 EOS(자연 종결). 토큰 텍스트는 전역 TOKENIZER로 디코드.
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
            // 완결 — 채팅 종결자로 정지(스케줄러 stop 계약). 부재 시 id 0.
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
