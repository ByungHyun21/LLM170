//! qwen4exp MTP(스펙 디코드) — 드래프트 체인·검증·수용 보행 (plans/129 R5 이동).
//! spec_step 계열(단일·multi·frame 변형)·suffix 드래프터·CPU MTP층 헤드.
use super::super::stages::{self, Ctx};
use super::super::{Hparams4, Q4Error};
use super::{Engine4, SeqState4};
use crate::matmul::Accelerator;
use crate::ops::sigmoid;
use crate::quant::dequant_row;

impl Engine4 {
    /// MTP 드래프트 스텝 (CPU 참조, plans/109 P15②) — 외장 nextn 블록
    /// (blk.{n_layer}) 1회 포워드. 입력: 직전 타깃 hidden h(프리-헤드,
    /// hc_mix_head 출력) + 채택 토큰 x. 출력: 드래프트 로짓 [vocab].
    /// 산술: qwen3next MTP 패턴(qwen4exp HC 변형) —
    ///   e=emb(x) → enorm·hnorm → eh_proj([en;hn]) → x̃
    ///   res_hc = x̃ 방송 → hc_mix(attn) → QSA(블록 n_layer, 자체 KV) →
    ///   hc_combine → hc_mix(ffn) → MoE → hc_combine →
    ///   hc_mix(nextn.hc_head) → output(@본체 공유) → logits.
    /// 드래프트 상태 pos는 호출부가 1씩 진행(mtp_seqs[seq].pos).
    /// MTP 스펙 스텝 (plans/109 P15③, CPU 참조판) — 드래프트 k-1토oken 체인
    /// 제안 + 타깃 순차 디코드 검증. q35 spec_step과 동일 계약:
    /// 반환 (수용 토큰열[보너스 포함], 타깃 forward 수).
    /// 상태 안전: 검증 전 타깃+드래프트 상태를 스냅샷해 거부 시 복원
    /// (SeqState4는 Vec 필드라 clone이 완전 복사). 값경로 전제 — 프레임
    /// 경로는 ④에서 프레임 트랜잭션(plans/86 §3)으로 동일 보장.
    pub fn mtp_spec_step(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        if k == 0 || !self.model.has_mtp() {
            return Ok((Vec::new(), 0));
        }
        // ── plans/110 W2: 프레임 경로 — 배치 검증(1회 t=k-1 포워드) ──
        // 실패 시 fb 카운터 + 순차(값경로) 폴백.
        if k >= 2 {
            match self.mtp_spec_step_frame(seq, last_token, k) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    super::super::frame::fb_incr(super::super::frame::FbId::MtpSpec);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("# mtp-spec-frame: 실패 — 순차 경로 폴백 ({e})"));
                }
            }
        }
        // ① 직전의 pre-mixer 잔차 = last_token의 예측자 hidden(① 전에 확보 —
        // decode1이 덮어쓴다). 이것이 주기 시작 커밋 토큰의 드래프트 쌍 h다.
        let h_prev = if self.spec_h_prev[seq].is_empty() {
            self.last_res_hc.clone()
        } else {
            std::mem::take(&mut self.spec_h_prev[seq])
        };
        let l = self.decode1(seq, last_token)?;
        let t0 = crate::qwen35::greedy(&l);
        let mut forwards = 1usize;
        let h_after_first = self.last_res_hc.clone();
        // ④′ 주기 시작 커밋 토큰(last_token)의 드래프트 KV 행을 **진위치에
        // 기입** — 종전 미기입이 드래프트 문맥에서 가장 최근 행을 잃게 했다.
        {
            let hp_hc = self.model.hp.hc * self.model.hp.n_embd;
            let hp0 = self.mtp_seqs[seq].pos;
            let _ = hp_hc;
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let _ = hp0;
        }
        // ② 드래프트 체인 — h는 pre-mix 멀티[10240]로 연결(chain export).
        // plans/141: 프레임 판과 동일하게 k회 반복 — 마지막 예측 g_{k-1}까지
        // 제안에 실어야 검증 비교가 성립한다(종전 k-1회 = 제안 k-1행이라
        // 마지막 항을 버리고 수용 판정 대상이 사라졌다).
        let snap_d = self.mtp_seqs[seq].clone();
        let snap_t = self.seqs[seq].clone();
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k {
            let _acc = self.acc.clone();
            let (next_d, dh) = self.mtp_draft_step_h(seq, next, &chain_h, _acc.as_deref())?;
            proposals.push(next);
            chain_h = dh;
            next = next_d;
        }
        // ③ 검증 — 제안 순차 타깃 디코드 후 수용 접두 판정.
        let mut tgt_out = Vec::new();
        for &p in &proposals {
            let l = self.decode1(seq, p)?;
            forwards += 1;
            tgt_out.push(crate::qwen35::greedy(&l));
        }
        let mut n_acc = proposals.len();
        for i in 0..proposals.len() {
            if let Some(e) = proposals.get(i + 1)
                && tgt_out[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if n_acc + 1 >= proposals.len() {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*tgt_out.last().unwrap_or(&t0));
        } else {
            // 거부 — ① 직후 스냅샷 복원 후 주기시작 행 재기입 + 수용분 재적립.
            self.seqs[seq] = snap_t;
            self.mtp_seqs[seq] = snap_d;
            let hp_hc = self.model.hp.hc * self.model.hp.n_embd;
            let _ = hp_hc;
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc.min(proposals.len() - 1)] {
                let l = self.decode1(seq, p)?;
                forwards += 1;
                let _acc = self.acc.clone();
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh, _acc.as_deref())?;
                dh = ndh;
                let _ = l;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc.min(proposals.len() - 1)]);
            accepted.push(tgt_out[n_acc]);
        }
        self.spec_h_prev[seq] = self.last_res_hc.clone();
        Ok((accepted, forwards))
    }

    /// plans/115 P12 (Strata P0-3 서픽스 드래프터): 히스토리 접미 n-gram
    /// (길이 2..=8, 긴 것 우선)의 가장 최근 선행 출현 이후 k 토큰.
    /// llama.cpp prompt-lookup과 동일 규칙 — 드래프트 비용 0.
    pub fn suffix_drafts(hist: &[u32], k: usize) -> Vec<u32> {
        let h = hist.len();
        if h < 4 || k == 0 {
            return Vec::new();
        }
        let max_n = 8.min(h - 1);
        for n in (2..=max_n).rev() {
            let suf = &hist[h - n..];
            for start in (0..h - n).rev() {
                if &hist[start..start + n] == suf {
                    let out: Vec<u32> = hist[start + n..].iter().take(k).copied().collect();
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
        Vec::new()
    }

    /// plans/115 P12: 서픽스 제안 스펙 라운드 — 검증/수용/기각 재실행은
    /// mtp_spec_step_frame과 동일 기계(배치 verify + 배치 재실행, 43/43 등가
    /// 계약). 드래프트 헤드가 없다(MTP 무관 — 모델 공통). 제안이 비면 호출부가
    /// MTP/plain으로 폴백한다(이 메서드는 제안 있는 경우만 담당).
    pub fn suffix_spec_step(
        &mut self,
        seq: usize,
        last_token: u32,
        drafts: &[u32],
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        if drafts.is_empty() {
            return Ok((Vec::new(), 0));
        }
        if self.frame_on(true) && self.frame_ensure() {
            return self.suffix_spec_step_frame(seq, last_token, drafts);
        }
        // 프레임 경로 불가 — 순차 검증(값경로). 수용 접두 판정은 동일식.
        let l = self.decode1(seq, last_token)?;
        let t0 = crate::qwen35::greedy(&l);
        let mut forwards = 1usize;
        let proposals: Vec<u32> = std::iter::once(t0).chain(drafts.iter().copied()).collect();
        let snap_t = self.seqs[seq].clone();
        let mut tgt_out = Vec::new();
        for &p in &proposals {
            let li = self.decode1(seq, p)?;
            forwards += 1;
            tgt_out.push(crate::qwen35::greedy(&li));
        }
        let n_all = proposals.len() - 1;
        let mut n_acc = n_all;
        for i in 0..n_all {
            if let Some(e) = proposals.get(i + 1)
                && tgt_out[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = vec![t0];
        if n_acc >= n_all {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*tgt_out.last().unwrap_or(&t0));
        } else {
            self.seqs[seq] = snap_t;
            for &p in &proposals[..=n_acc] {
                self.decode1(seq, p)?;
                forwards += 1;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(tgt_out[n_acc]);
        }
        Ok((accepted, forwards))
    }

    /// 서픽스 스펙 프레임 판 — 배치 검증 + (기각 시) 배치 재실행.
    fn suffix_spec_step_frame(
        &mut self,
        seq: usize,
        last_token: u32,
        drafts: &[u32],
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        // ① 주기 시작 — last_token 디코드 1회(t0 산출·상태 전진).
        let l = self.decode1(seq, last_token)?;
        let t0 = crate::qwen35::greedy(&l);
        let mut forwards = 1usize;
        let proposals: Vec<u32> = std::iter::once(t0).chain(drafts.iter().copied()).collect();
        let snap_t = self.seqs[seq].clone();
        // ② 배치 검증 — 스냅샷 → t=len(proposals) 포워드 → 행별 argmax.
        let y: Vec<u32>;
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("suffix-spec: 프레임 없음".into()));
            };
            super::super::frame::verify_snap_capture(a, f, seq)?;
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            y = super::super::frame::frame_forward_verify(
                a,
                model,
                &ctx,
                seqs.as_mut_slice(),
                seq,
                f,
                &proposals,
            )?;
        }
        forwards += 1;
        // ③ 수용 접두 판정 + 정착 — full은 배치 상태 그대로, 기각은 복원 후
        // 배치 재실행(수용분 1회 포워드, 등가 계약).
        let n_all = proposals.len() - 1;
        let mut n_acc = n_all;
        for i in 0..n_all {
            if let Some(e) = proposals.get(i + 1)
                && y[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if n_acc >= n_all {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*y.last().unwrap_or(&t0));
            self.seqs[seq].pos += proposals.len() as u32;
            Ok((accepted, forwards))
        } else {
            // 기각 — CPU 상태 복원 + GDN 스냅샷 복원 + 수용분 배치 재실행.
            self.seqs[seq] = snap_t;
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("suffix-spec: 프레임 없음".into()));
                };
                super::super::frame::verify_snap_restore(a, f, seq)?;
            }
            let win: Vec<u32> = proposals[..=n_acc].to_vec();
            {
                let Engine4 {
                    model,
                    frame,
                    seqs,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("suffix-spec: 프레임 없음".into()));
                };
                let ctx = crate::qwen4exp::stages::Ctx {
                    model,
                    acc: Some(a),
                };
                super::super::frame::frame_forward_verify(
                    a,
                    model,
                    &ctx,
                    seqs.as_mut_slice(),
                    seq,
                    f,
                    &win,
                )?;
            }
            forwards += 1;
            self.seqs[seq].pos += win.len() as u32;
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(y[n_acc]);
            Ok((accepted, forwards))
        }
    }

    /// MTP 스펙 스텝 프레임 판 (plans/110 W2) — 검증을 t=k-1 배치 포워드
    /// 1회로 통합(np 불변식: 배치 == 순차 decode1 비트 동일). 기각 시
    /// GDN 디바이스 상태 스냅샷 복원 + PLE 링 pos 되감기 + 수용분 재실행.
    /// 수용 산출식은 순차 판과 동일 — 프레임 경로 상태 부패(스펙≠비스펙
    /// 토큰 분기, 110 W2 발견)도 이 트랜잭션으로 해소된다.
    fn mtp_spec_step_frame(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        // ① 주기 시작 커밋 토큰의 타깃 forward — greedy 판정만 회수.
        let h_prev = if self.spec_h_prev[seq].is_empty() {
            self.last_res_hc.clone()
        } else {
            std::mem::take(&mut self.spec_h_prev[seq])
        };
        let t0 = self.decode1_greedy(seq, last_token)?;
        let mut forwards = 1usize;
        let h_after_first = self.last_res_hc.clone();
        // ② 드래프트 체인 — proposals = [t0, g1, .., g_{k-1}](k개).
        //
        // plans/141: 종전 `k-1`회 반복은 마지막 예측 `g_{k-1}`을 계산만 하고
        // 버려 검증 목록에서 제외했다. 결과적으로 검증 행이 t0로 끝나 다음
        // 제안과 비교할 행이 없어 매 라운드 matched=0·full=true(수용 0)이 되었고,
        // 드래프트 비용만 더해 순수 디코드보다 느렸다(k=2 실측 3.09 vs 18.14 t/s).
        // k회 반복은 두 가지를 함께 맞춘다: 검증 대상이 k행이 되어 g_{k-1}까지
        // 비교되고, 드래프트 KV도 전 라운드 커밋 토큰(t0·g_1..g_{k-1})을 덮는다
        // — 전 수용 시 다음 라운드 ④′ 이전 드래프트 문맥이 비지 않는다.
        let snap_t = self.seqs[seq].clone();
        // 거각 복원 기준 — ④′ 이전. 복원 후 재체인(아래)이 last_token 행부터
        // 다시 적립하므로, ④′까지 진행한 상태를 기준으로 잡으면 last_token 행이
        // 한 번 더 쌓여 드래프트 위치가 어긋난다.
        let snap_d = self.mtp_seqs[seq].clone();
        // ④′ 주기 시작 커밋 토큰의 드래프트 KV 행 진위치 기입.
        {
            let _acc = self.acc.clone();
            self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
        }
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k {
            let _acc = self.acc.clone();
            let (next_d, dh) = self.mtp_draft_step_h(seq, next, &chain_h, _acc.as_deref())?;
            proposals.push(next);
            chain_h = dh;
            next = next_d;
        }
        // ── 배치 검증: t=k행 1회 포워드 + 행별 GPU argmax ──
        let y: Vec<u32>;
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("mtp-spec-frame: 프레임 없음".into()));
            };
            // 검증 직전 GDN 디바이스 상태 스냅샷(기각 복원용).
            super::super::frame::verify_snap_capture(a, f, seq)?;
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            y = super::super::frame::frame_forward_verify(
                a,
                model,
                &ctx,
                seqs.as_mut_slice(),
                seq,
                f,
                &proposals,
            )?;
            // 다음 라운드 h 입력 — 배치 export 행 풀(마지막 행 = 마지막 처리 행).
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                self.last_res_hc_rows = f.last_res_hc_rows.clone();
                self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
            }
        }
        forwards += 1;
        // y[i] = proposals[i] 처리 후 greedy — 수용 접두 판정(순차 판과 동일식).
        let mut n_acc = proposals.len();
        for i in 0..proposals.len() {
            if let Some(e) = proposals.get(i + 1)
                && y[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let full = n_acc + 1 >= proposals.len();
        // plans/141: 수용률 0의 원인을 GPU 수치 전에 판별한다. 프레임 스펙의
        // 실제 제안열·검증열과 채택 수를 함께 남겨 제안 누락을 식별한다.
        // `n_acc` = 검증 출력과 일치한 드래프트 수(y[i] == proposals[i+1]).
        // 드래프트 pos도 함께 찍는다 — 거각 복원·재체인이 어긋나면 pos가
        // 라운드마다 상승해 CPU 어텐션 비용이 늘어난다(spawn O(pos)).
        if llm170_diag::dump::opts().key("spec_accept") {
            eprintln!(
                "# spec-accept pos={} dpos={} k={k} proposals={proposals:?} verify={y:?} matched={n_acc} full={full}",
                snap_t.pos, self.mtp_seqs[seq].pos,
            );
        }
        // 그림자 진단(LLM170_DUMP=spec_check) — 배치 y·상태와 순차 decode1
        // 재현을 전수 대조. 그림자 종료 상태 = 순차 전이(배치가 도달해야 할
        // 상태)라 관측이 스트림을 오염시키지 않는다.
        let shadow = llm170_diag::dump::opts().key("spec_check");
        if shadow {
            // 배치가 남긴 GDN 디바이스 상태.
            let batch_gdn = {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    unreachable!("프레임 존재");
                };
                let mut snap = vec![Vec::new(); f.st_gdn[seq].len()];
                for (ri, &h) in f.st_gdn[seq].iter().enumerate() {
                    snap[ri] = vec![0.0f32; f.gdn_state_len];
                    a.frame_read(h, &mut snap[ri]).map_err(Q4Error::Io)?;
                }
                snap
            };
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    unreachable!("프레임 존재");
                };
                super::super::frame::verify_snap_restore(a, f, seq)?;
            }
            self.seqs[seq] = snap_t.clone();
            self.mtp_seqs[seq] = snap_d.clone();
            let seq_y: Vec<u32> = proposals
                .iter()
                .map(|&p| self.decode1_greedy(seq, p))
                .collect::<Result<_, _>>()?;
            let mism: Vec<String> = (0..proposals.len())
                .filter(|&i| y[i] != seq_y[i])
                .map(|i| format!("y[{i}]={} seq={}", y[i], seq_y[i]))
                .collect();
            let mut gdn_bad = 0usize;
            let mut first = String::new();
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    unreachable!("프레임 존재");
                };
                for (ri, &h) in f.st_gdn[seq].iter().enumerate() {
                    let mut v = vec![0.0f32; f.gdn_state_len];
                    a.frame_read(h, &mut v).map_err(Q4Error::Io)?;
                    let bad = v
                        .iter()
                        .zip(batch_gdn[ri].iter())
                        .filter(|(x, b)| x.to_bits() != b.to_bits())
                        .count();
                    if bad > 0 && first.is_empty() {
                        first = format!("gdn[ri={ri}] {bad}");
                    }
                    gdn_bad += bad;
                }
            }
            eprintln!(
                "# spec-check pos={} full={} mismatch {} gdn_bad={gdn_bad} {first}",
                snap_t.pos,
                full,
                mism.len(),
            );
            if full {
                // 전수용: 순차 상태가 곧 정답 — 이 상태로 계속.
                let mut acc_v = vec![t0];
                acc_v.extend_from_slice(&proposals[1..]);
                acc_v.push(*seq_y.last().unwrap_or(&t0));
                return Ok((acc_v, forwards));
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if full {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*y.last().unwrap_or(&t0));
            // 배치가 정확히 proposals행만큼 상태를 전진시켰다 — pos 정산.
            self.seqs[seq].pos += proposals.len() as u32;
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok((accepted, forwards))
        } else {
            // 기각 — 스냅샷 복원(GDN 디바이스 + CPU) 후 수용분 재실행.
            let snap_pos = snap_t.pos;
            self.seqs[seq] = snap_t;
            self.mtp_seqs[seq] = snap_d;
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("mtp-spec-frame: 프레임 없음".into()));
                };
                super::super::frame::verify_snap_restore(a, f, seq)?;
            }
            // PLE 링: CPU snap_t.ple_conv가 정합 — 이후 첫 PLE 디바이스 스텝이
            // pos 기반 워터마크 되감기로 호스트 링을 리프레시한다(백엔드 계약).
            //
            // plans/115 P0-1(Strata commit-replay 1단계): 수용분 재실행을 종전
            // m× 순차 decode1(스텝당 ~55ms×m — 기각 라운드의 지배 비용)에서
            // **배치 verify 1회**로. t=k-1 검증 배치는 순차 decode1과 완전
            // 등가(43/43, 원장 128) — 상태 전진과 y 모두 동일 산술이다.
            {
                let Engine4 {
                    model,
                    frame,
                    seqs,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("mtp-spec-frame: 프레임 없음".into()));
                };
                let ctx = crate::qwen4exp::stages::Ctx {
                    model,
                    acc: Some(a),
                };
                let win: Vec<u32> = proposals[..=n_acc].to_vec();
                let ry = super::super::frame::frame_forward_verify(
                    a,
                    model,
                    &ctx,
                    seqs.as_mut_slice(),
                    seq,
                    f,
                    &win,
                )?;
                forwards += 1;
                if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                    self.last_res_hc_rows = f.last_res_hc_rows.clone();
                    self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
                }
                // 진단(spec_check) — 재실행 ry[i]는 원 배치 y[i]와 일치해야 한다.
                if shadow {
                    let mm: Vec<String> = (0..=n_acc)
                        .filter(|&i| ry[i] != y[i])
                        .map(|i| format!("r[{i}]={} y={}", ry[i], y[i]))
                        .collect();
                    eprintln!(
                        "# spec-reject pos={snap_pos} n_acc={} replay_mismatch {}{}",
                        n_acc,
                        mm.len(),
                        if mm.is_empty() {
                            String::new()
                        } else {
                            format!(" first={}", mm[0])
                        }
                    );
                }
            }
            self.seqs[seq].pos += (n_acc + 1) as u32;
            // 드래프트 재체인 — 종전대로(수용 토큰별 mtp_draft_step_h).
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc] {
                let _acc = self.acc.clone();
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh, _acc.as_deref())?;
                dh = ndh;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(y[n_acc]);
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok((accepted, forwards))
        }
    }

    /// np×spec 병합 스펙 라운드 (plans/110 W5) — 다중 슬롯의 라운드 시작
    /// decode1을 **1회 np 배치 포워드**로 묶고(무게 패스 공유), 드래프트·
    /// 검증·롤백은 슬롯별(배치 검증 기계 재사용). 반환 [slot][accepted],
    /// forwards 총합. 큐35 spec_step_multi의 Q4판 — 검증 배치 병합(원자
    /// 의미론)은 후속, 여기선 라운드 시작 병합만.
    pub fn mtp_spec_step_multi(
        &mut self,
        slots: &[usize],
        last_tokens: &[u32],
        k: usize,
    ) -> Result<(Vec<Vec<u32>>, usize), Q4Error> {
        if k == 0 || !self.model.has_mtp() || slots.is_empty() {
            return Ok((Vec::new(), 0));
        }
        if !(self.frame_on(true) && self.frame_ensure()) || slots.len() < 2 {
            // 폴백: 단일 슬롯 순차판(기존 mtp_spec_step).
            let mut out = Vec::with_capacity(slots.len());
            let mut fw = 0usize;
            for (&s, &t) in slots.iter().zip(last_tokens.iter()) {
                let (acc, f) = self.mtp_spec_step(s, t, k)?;
                fw += f;
                out.push(acc);
            }
            return Ok((out, fw));
        }
        // ① 다중 슬롯 라운드 시작 — 1회 np 배치(행핀으로 decode1 비트 동일)
        // + pre-mixer res_hc 행 export(드래프트 h_after_first).
        let toks: Vec<u32>;
        let mut row_h: Vec<Vec<f32>> = Vec::new();
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("mtp-spec-multi: 프레임 없음".into()));
            };
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            // 행핀 없음 — serve 비스펙(np) 경로와 동일 산술·무게 상각 유지.
            // (근접 타이 플립은 수용률 저하로만 나타난다 — 스펙 고유 성질.)
            let r = super::super::frame::frame_forward_np_greedy_h(
                a,
                model,
                &ctx,
                slots,
                seqs.as_mut_slice(),
                f,
                last_tokens,
            );
            toks = r?;
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                row_h = f.last_res_hc_rows.clone();
            }
        }
        let mut forwards = 1usize;
        let mut out = Vec::with_capacity(slots.len());
        for (i, (&s, &lt)) in slots.iter().zip(last_tokens.iter()).enumerate() {
            let t0 = toks.get(i).copied().unwrap_or(0);
            // ② 이 슬롯의 h_after_first = np export 행 i.
            let h_after_first = row_h.get(i).cloned().unwrap_or_default();
            let accepted = self.mtp_spec_round_rest(s, lt, t0, h_after_first, k, &mut forwards)?;
            out.push(accepted);
        }
        Ok((out, forwards))
    }

    /// 스펙 라운드의 잔여(④′ 드래프트 + 체인 + 배치 검증 + 수용/롤백) —
    /// 라운드 시작이 외부(다중 병합)에서 처리된 경우의 공유 본체.
    fn mtp_spec_round_rest(
        &mut self,
        seq: usize,
        last_token: u32,
        t0: u32,
        h_after_first: Vec<f32>,
        k: usize,
        forwards: &mut usize,
    ) -> Result<Vec<u32>, Q4Error> {
        // ④′ + 체인 드래프트 — h_prev는 슬롯별 직전 라운드 최종 export
        // (spec_h_prev; 110 W5 — 전역 last_res_hc는 타 슬롯이 덮는다).
        let h_prev = if self.spec_h_prev[seq].is_empty() {
            return Err(Q4Error::Io("spec_h_prev 미시드".into()));
        } else {
            std::mem::take(&mut self.spec_h_prev[seq])
        };
        {
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
        }
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k.saturating_sub(1) {
            let _acc = self.acc.clone();
            let (next_d, dh) = self.mtp_draft_step_h(seq, next, &chain_h, _acc.as_deref())?;
            proposals.push(next);
            chain_h = dh;
            next = next_d;
        }
        // 배치 검증 + 수용/롤백 — mtp_spec_step_frame의 ②이후와 동일 기계.
        let snap_t = self.seqs[seq].clone();
        let snap_d = self.mtp_seqs[seq].clone();
        let y: Vec<u32>;
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("mtp-spec-round: 프레임 없음".into()));
            };
            super::super::frame::verify_snap_capture(a, f, seq)?;
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            y = super::super::frame::frame_forward_verify(
                a,
                model,
                &ctx,
                seqs.as_mut_slice(),
                seq,
                f,
                &proposals,
            )?;
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                self.last_res_hc_rows = f.last_res_hc_rows.clone();
                self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
            }
        }
        *forwards += 1;
        let mut n_acc = proposals.len();
        for i in 0..proposals.len() {
            if let Some(e) = proposals.get(i + 1)
                && y[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if n_acc + 1 >= proposals.len() {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*y.last().unwrap_or(&t0));
            self.seqs[seq].pos += proposals.len() as u32;
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok(accepted)
        } else {
            self.seqs[seq] = snap_t;
            self.mtp_seqs[seq] = snap_d;
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("mtp-spec-round: 프레임 없음".into()));
                };
                super::super::frame::verify_snap_restore(a, f, seq)?;
            }
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc] {
                self.decode1_greedy(seq, p)?;
                *forwards += 1;
                let _acc = self.acc.clone();
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh, _acc.as_deref())?;
                dh = ndh;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(y[n_acc]);
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok(accepted)
        }
    }
    /// MTP 드래프트 프리필 (P15④) — 타깃 프리필 직후 호출. 프롬프트 토큰
    /// c_1..c_{T-1}을 (c_{i+1}, h_i) 쌍으로 드래프트 계층에 적립해 드래프트
    /// KV가 전체 문맥을 갖게 한다(빈 문맥 시작이 수용률 붕괴 원인 — 실측).
    /// last_h_rows는 직전 값경로 prefill의 h 행 전체.
    pub fn mtp_draft_prefill(
        &mut self,
        seq: usize,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<(), Q4Error> {
        if !self.model.has_mtp() || self.mtp_seqs.is_empty() {
            return Ok(());
        }
        if self.last_res_hc_rows.len() != tokens.len() {
            return Err(Q4Error::Io(format!(
                "mtp prefill: pre-mix h행({}) ≠ 토큰({}) — 값경로 프리필 직후에 호출",
                self.last_res_hc_rows.len(),
                tokens.len()
            )));
        }
        // 쌍 규약: 위치 p의 드래프트 입력은 (c_p, h_{p-1}). h_rows[i]는
        // c_i 처리 후 h — 즉 위치 i+1의 쌍은 (c_{i+1}, h_rows[i]).
        // **드래프트 KV 위치는 타깃 위치와 1:1**(vLLM "cell for cell") —
        // c_1은 pos 1에 적립(위치 0의 h_{-1}은 없음). 종전 pos 0 시작은
        // 로프 위치 전체를 1 어긋나게 했다(수용률 억제 원인, P15④).
        // base_pos(P1-3): 접두 복원 잡은 [cp..)만 재생 — 드래프트 KV [0..cp)는
        // 접두 불변이라 이미 유효(체크포인트가 pos만 되감았다).
        self.mtp_seqs[seq].pos = (base_pos + 1) as u32;
        for i in 0..tokens.len().saturating_sub(1) {
            let x = tokens[i + 1];
            let hi = self.last_res_hc_rows[i].clone();
            let _acc = self.acc.clone();
            let (_lg, _h) = self.mtp_draft_step_h(seq, x, &hi, _acc.as_deref())?;
        }
        // 110 W5: 슬롯별 스펙 h 시드 — 첫 스펙 라운드의 h_prev(프리필 최종 h).
        self.spec_h_prev[seq] = self.last_res_hc.clone();
        Ok(())
    }

    /// plans/115 D2: 잡 h행 세션 시작 — 슬롯 프리필 첫 청크 직전 호출.
    /// 잔존 행(이전 잡/스펙 라운드)이 mtp_draft_prefill의 len 검사를
    /// 깨고 드래프트 프리필을 영구 생략시켰다(수용률 붕괴).
    fn mtp_draft_step_h(
        &mut self,
        seq: usize,
        x: u32,
        h_pre: &[f32],
        acc: Option<&dyn Accelerator>,
    ) -> Result<(u32, Vec<f32>), Q4Error> {
        if !self.model.has_mtp() || self.mtp_seqs.is_empty() {
            return Err(Q4Error::Io("mtp_draft_step: MTP 미적재".into()));
        }
        let hp = &self.model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let il = hp.n_layer;
        let hc_dim = hc * n;
        if h_pre.len() != hc_dim {
            return Err(Q4Error::Io(format!(
                "mtp h 계약 위반: {} ≠ hc·n {}",
                h_pre.len(),
                hc_dim
            )));
        }
        let embd = self.model.w4("token_embd.weight")?;
        let mut e = vec![0.0f32; n];
        dequant_row(embd.ty, embd.data, x as u64, n as u64, &mut e);
        // enorm: 임베딩 플랫 정규화 [n]. hnorm: **멀티 스트림 전체 플랫 RMS**
        // [hc·n](vLLM GemmaRMSNorm(hidden*hc_count) 평탄 1회 — 스트림별 아님).
        // 두 norm은 공통(값·프레임 경로 동일 입력).
        let en = crate::ops::rms_norm(
            &e,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.enorm.weight"))?,
            hp.eps,
        );
        let hn = crate::ops::rms_norm(
            h_pre,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.hnorm.weight"))?,
            hp.eps,
        );
        // ── plans/110 W1: 프레임 경로 — 상주 버퍼 GEMV 체인 ──
        let mtp_t = std::time::Instant::now();
        if self.frame_on(true) && self.frame_ensure() {
            let Engine4 {
                model,
                frame,
                mtp_seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                unreachable!("frame_ensure 성공 직후");
            };
            match super::super::frame::mtp_draft_frame(a, model, f, &mut mtp_seqs[seq], &en, &hn) {
                Ok((tok, chain_h)) => {
                    if llm170_diag::dump::opts().key("mtp_time") {
                        eprintln!(
                            "# mtp-draft-frame: {:.2}ms",
                            mtp_t.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    mtp_seqs[seq].pos += 1;
                    return Ok((tok, chain_h));
                }
                Err(e) => {
                    super::super::frame::fb_incr(super::super::frame::FbId::MtpDraft);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("# mtp-draft-frame: 실패 — 값경로 폴백 ({e})"));
                }
            }
        }
        // ── 값경로(종전 판) ──
        // eh_proj [2n→n] = 융합 [fc_embedding | fc_hidden]: 스트림 s 초기값 =
        //   fc_hidden(hn_s) + fc_embedding(en)  (vLLM amd: emb.unsqueeze + hidden).
        // fc_embedding = 입력 반쪽 [..n], fc_hidden = 입력 뒤반쪽 [n..2n].
        let weh = self.model.w4(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let mut res_hc: Vec<Vec<f32>> = Vec::with_capacity(1);
        {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            let mut r = vec![0.0f32; hc_dim];
            let mut x_t = vec![0.0f32; n];
            for s_i in 0..hc {
                let mut cat = vec![0.0f32; 2 * n];
                cat[..n].clone_from_slice(&en);
                cat[n..].clone_from_slice(&hn[s_i * n..(s_i + 1) * n]);
                ctx.mm(&cat, &weh, &mut x_t)?;
                r[s_i * n..(s_i + 1) * n].copy_from_slice(&x_t);
            }
            res_hc.push(r);
        }
        let (mix, inject) = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::hc_mix(&ctx, il, "attn", &res_hc)?
        };
        let attn_out = self.mtp_dense_attn(seq, il, &mix, acc)?;
        hc_combine(&mut res_hc, &attn_out, &inject, hc);
        let (mix2, inject2) = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::hc_mix(&ctx, il, "ffn", &res_hc)?
        };
        let ffn_out = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::moe_ffn(&ctx, il, &mix2)?
        };
        hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        let head_rows = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::hc_mix_nextn_head(&ctx, il, &res_hc)?
        };
        let h1 = head_rows
            .last()
            .ok_or(Q4Error::BadMeta("mtp 빈 헤드"))?
            .clone();
        let wout = self
            .model
            .w4("output.weight")
            .map_err(|_| Q4Error::MissingTensor("output.weight".into()))?;
        let mut logits = vec![0.0f32; wout.n_out as usize];
        {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            ctx.mm(&h1, &wout, &mut logits)?;
        }
        self.mtp_seqs[seq].pos += 1;
        // 체인 반출 = pre-mix 멀티 스트림(마지막 행).
        let chain_h = res_hc.last().cloned().unwrap_or_default();
        Ok((crate::qwen35::greedy(&logits), chain_h))
    }

    /// MTP dense 게이트드 어텐션 (plans/109 P15②) — 값경로 판. 투영(q/k/v)은
    /// mm_group, norm·rope·KV·softmax·게이트는 mtp_attn_cpu_row(공유 코어),
    /// wo 투영 mm_group. 프레임 경로(110 W1)는 frame/mtp.rs가 동일 코어를
    /// 판독한 q/k/v 행으로 호출한다.
    fn mtp_dense_attn(
        &mut self,
        seq: usize,
        il: usize,
        xs: &[Vec<f32>],
        acc: Option<&dyn Accelerator>,
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        let hp = &self.model.hp;
        let wq = self.model.w4(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.w4(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.w4(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.w4(&format!("blk.{il}.attn_output.weight"))?;
        let qn = self
            .model
            .f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
        let kn = self
            .model
            .f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
        let n_tok = xs.len();
        let mut qg = vec![vec![0.0f32; wq.n_out as usize]; n_tok];
        let mut kk = vec![vec![0.0f32; wk.n_out as usize]; n_tok];
        let mut vv = vec![vec![0.0f32; wv.n_out as usize]; n_tok];
        {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            let mut gi = vec![
                std::mem::take(&mut qg),
                std::mem::take(&mut kk),
                std::mem::take(&mut vv),
            ];
            ctx.mm_group(xs, &[wq, wk, wv], &mut gi)?;
            qg = std::mem::take(&mut gi[0]);
            kk = std::mem::take(&mut gi[1]);
            vv = std::mem::take(&mut gi[2]);
        }
        let mtp_st = &mut self.mtp_seqs[seq];
        let mut attn_all = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            attn_all.push(mtp_attn_cpu_row(
                hp, &mut qg[t], &mut kk[t], &vv[t], mtp_st, &qn, &kn,
            ));
        }
        // wo 투영 (대여 분리 — KV 적립 종료 후 새 Ctx).
        let ctx = Ctx {
            model: &self.model,
            acc,
        };
        let mut out_rows = vec![vec![vec![0.0f32; wo.n_out as usize]; n_tok]; 1];
        ctx.mm_group(&attn_all, std::slice::from_ref(&wo), &mut out_rows)?;
        Ok(std::mem::take(&mut out_rows[0]))
    }

    pub fn mtp_draft_step(&mut self, seq: usize, x: u32, h: &[f32]) -> Result<Vec<f32>, Q4Error> {
        if !self.model.has_mtp() || self.mtp_seqs.is_empty() {
            return Err(Q4Error::Io("mtp_draft_step: MTP 미적재".into()));
        }
        let hp = &self.model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let il = hp.n_layer; // 블록 48
        let hc_dim = hc * n;
        // 1) e = emb(x) — 본체 임베딩 공유
        let embd = self.model.w4("token_embd.weight")?;
        let mut e = vec![0.0f32; n];
        dequant_row(embd.ty, embd.data, x as u64, n as u64, &mut e);
        // 2) enorm/hnorm
        let en = crate::ops::rms_norm(
            &e,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.enorm.weight"))?,
            hp.eps,
        );
        let hn = crate::ops::rms_norm(
            h,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.hnorm.weight"))?,
            hp.eps,
        );
        // 3) eh_proj: [en;hn](2n) → x̃(n)
        let mut cat = vec![0.0f32; 2 * n];
        cat[..n].clone_from_slice(&en);
        cat[n..].clone_from_slice(&hn);
        let weh = self.model.w4(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let mut x_t = vec![0.0f32; n];
        {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            ctx.mm(&cat, &weh, &mut x_t)?;
        }
        // 4) res_hc 방송 (트렁크 hc_init와 동일)
        let mut res_hc: Vec<Vec<f32>> = Vec::with_capacity(1);
        {
            let mut r = vec![0.0f32; hc_dim];
            for s in 0..hc {
                r[s * n..(s + 1) * n].copy_from_slice(&x_t);
            }
            res_hc.push(r);
        }
        // 5) 어텐션 반쪽 — MTP 헤드는 **dense** 게이트드 어텐션(llama.cpp
        // qwen3next MTP 패턴: 인덱서 미사용, compress[48]=0과 일관). 트렁크
        // qsa의 cpu_attn_row 산술(전체 위치 마스크)과 동일 열.
        // 대여 분리: 각 스테이지가 스코프 ctx로 self.model만 빌린다.
        let (mix, inject) = {
            let ctx = Ctx {
                model: &self.model,
                acc: None, // CPU 참조 — GPU 경로는 ④(frame)에서
            };
            stages::hc_mix(&ctx, il, "attn", &res_hc)?
        };
        let _acc = self.acc.clone();
        let attn_out = self.mtp_dense_attn(seq, il, &mix, _acc.as_deref())?;
        hc_combine(&mut res_hc, &attn_out, &inject, hc);
        // 6) FFN 반쪽 (MoE + shexp)
        let (mix2, inject2) = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            stages::hc_mix(&ctx, il, "ffn", &res_hc)?
        };
        let ffn_out = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            stages::moe_ffn(&ctx, il, &mix2)?
        };
        hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        // 7) 드래프트 헤드 — nextn.hc_head 믹서 → 본체 output 공유
        let head_rows = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            stages::hc_mix_nextn_head(&ctx, il, &res_hc)?
        };
        let h1 = head_rows.last().ok_or(Q4Error::BadMeta("mtp 빈 헤드"))?;
        let wout = self
            .model
            .w4("output.weight")
            .map_err(|_| Q4Error::MissingTensor("output.weight".into()))?;
        let mut logits = vec![0.0f32; wout.n_out as usize];
        {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            ctx.mm(h1, &wout, &mut logits)?;
        }
        self.mtp_seqs[seq].pos += 1;
        Ok(logits)
    }
}

/// MTP dense 어텐션 CPU 코어(plans/110 W1 분리) — q/k/v 원시 행 1개에
/// norm·rope·KV 적립·softmax(cell 0 스킵)·게이트를 적용해 [n_head·hd] 반환.
/// 값경로(mtp_dense_attn)와 프레임 경로(frame/mtp.rs)가 공유 — 산술 단일 소스.
pub(crate) fn mtp_attn_cpu_row(
    hp: &Hparams4,
    q_row: &mut [f32],
    k_row: &mut [f32],
    v_row: &[f32],
    st: &mut SeqState4,
    qn: &[f32],
    kn: &[f32],
) -> Vec<f32> {
    let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
    let pos = st.pos;
    let kq_scale = hp.kq_scale();
    // q: 헤드별 norm+rope(전반 hd) — 게이트 후반은 미가공.
    for h in 0..n_head {
        let lo = h * 2 * hd;
        let mut qh = crate::ops::rms_norm(&q_row[lo..lo + hd], qn, hp.eps);
        crate::ops::rope_head(&mut qh, pos, n_rot, hp.rope_base);
        q_row[lo..lo + hd].copy_from_slice(&qh);
    }
    // k: kv헤드별 norm+rope → 캐시 적립. v: 원문 그대로.
    let kbase = pos as usize * n_kv * hd;
    for h in 0..n_kv {
        let lo = h * hd;
        let mut kh = crate::ops::rms_norm(&k_row[lo..lo + hd], kn, hp.eps);
        crate::ops::rope_head(&mut kh, pos, n_rot, hp.rope_base);
        st.kv_k[0][kbase + lo..kbase + lo + hd].copy_from_slice(&kh);
    }
    st.kv_v[0][kbase..kbase + n_kv * hd].copy_from_slice(v_row);
    // dense softmax 어텐션 + 게이트 — cpu_attn_row 열에서 **cell 0은
    // 스킵**(드래프트 KV는 위치 1부터 기입 — 팬텀 0키가 softmax 질량을
    // 훔치는 결함, P15④-5).
    //
    // plans/141: 헤드별 루프를 스레드 파티션으로 나눴다. 이 계산은 드래프트
    // 스텝에서 컨텍스트에 비례해 grow하는 유일한 부분이다(스텝당 5ms → 12ms,
    // pp512 → 수천 토큰 구간). 헤드는 q/k/v와 KV를 **읽기만** 하고 out의
    // 서로 겹치지 않는 구간[ob, ob+hd)에 쓴다 — 헤드 간 공유 상태가 없어
    // 파티션이 정확하며, 각 헤드의 산술 순서(부분합 → softmax → 가중합 → 게이트)는
    // 그대로여서 비트 동일하다. frame/forward.rs의 ple_gatherMT와 같은 관례.
    let n_past = pos as usize;
    let (ck, cv) = (&st.kv_k[0], &st.kv_v[0]);
    let mut out = vec![0.0f32; n_head * hd];
    // q는 헤드 앞처리(rms_norm+rope)가 앞에서 in-place로 끝났으므로 읽기 전용이다.
    let q_ro: &[f32] = q_row;
    let nt = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_head)
        .min(32);
    let per = n_head.div_ceil(nt);
    std::thread::scope(|sc| {
        let mut rest: &mut [f32] = &mut out;
        let mut base = 0usize;
        while base < n_head {
            let take = per.min(n_head - base);
            let (chunk, tail) = rest.split_at_mut(take * hd);
            rest = tail;
            let b = base;
            sc.spawn(move || {
                for j in 0..take {
                    head_attn(
                        b + j,
                        n_head,
                        n_kv,
                        hd,
                        n_past,
                        kq_scale,
                        q_ro,
                        ck,
                        cv,
                        chunk,
                        j * hd,
                    );
                }
            });
            base += take;
        }
    });
    out
}

/// 단일 헤드의 dense 어텐션 + 게이트. `slot`은 파티션 버퍼 내 오프셋
/// (헤드 h는 `slot..slot+hd`를 쓴다). 산술 순서는 mtp_attn_cpu_row의 종전
/// 본문과 동일 — 병렬화만 바뀌고 계산 순서는 그대로다.
#[allow(clippy::too_many_arguments)]
fn head_attn(
    h: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_past: usize,
    kq_scale: f32,
    q_row: &[f32],
    ck: &[f32],
    cv: &[f32],
    out: &mut [f32],
    slot: usize,
) {
    let kvh = h / (n_head / n_kv);
    let qb = h * 2 * hd;
    let mut scores = vec![0.0f32; n_past];
    let mut maxv = f32::NEG_INFINITY;
    for (p, sc) in scores.iter_mut().enumerate() {
        let p = p + 1; // cell 0 스킵
        let b = p * n_kv * hd + kvh * hd;
        let mut d = 0.0f32;
        for i in 0..hd {
            d += q_row[qb + i] * ck[b + i];
        }
        *sc = d * kq_scale;
        maxv = maxv.max(*sc);
    }
    let mut sum = 0.0f32;
    for sc in scores.iter_mut() {
        *sc = (*sc - maxv).exp();
        sum += *sc;
    }
    for (p0, sc) in scores.iter().enumerate() {
        let w = sc / sum;
        if w == 0.0 {
            continue;
        }
        let b = (p0 + 1) * n_kv * hd + kvh * hd;
        for i in 0..hd {
            out[slot + i] += w * cv[b + i];
        }
    }
    let gb = h * 2 * hd + hd;
    for i in 0..hd {
        out[slot + i] *= sigmoid(q_row[gb + i]);
    }
}
/// hc_combine: res[s] += out·(2·σ(inject_s/4)).
pub(super) fn hc_combine(
    res_hc: &mut [Vec<f32>],
    out: &[Vec<f32>],
    inject: &[Vec<f32>],
    hc: usize,
) {
    for (t, o) in out.iter().enumerate() {
        for s in 0..hc {
            let w = 2.0 * sigmoid(inject[t][s] / hc as f32);
            let base = s * o.len();
            for (i, ov) in o.iter().enumerate() {
                res_hc[t][base + i] += ov * w;
            }
        }
    }
}
