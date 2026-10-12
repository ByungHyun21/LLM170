//! GPU 서빙 엔진 — W4A16 CUDA 디바이스 체인 + GPU head.
//!
//! W3-3 착륙 후 성능 캠페인(2026-10-08): 활성은 디바이스 상주(왕복은
//! 토큰당 임베딩 업로드 + 로짓 판독뿐), head는 head_bf16 커널(bf16
//! →f32 시프트 + k 직렬 f32 내적 — CPU 참조와 동일 순서).
//! LLM170_STAGED=1이면 구 스테이징 경로(호스트 왕복)로 폴백한다.
//! 판정 기준은 토큰열(골든)이다.

use llm170_backend_gpu::{AttnDims, GdnDims, W4a16Dec};
use std::path::Path;

pub struct GpuEngine {
    model: llm170_core::qwen35::Model,
    dec: W4a16Dec,
    n_slots: usize,
    /// 스테이징 폴백(LLM170_STAGED=1 — 비교/디버그 전용).
    staged: bool,
    /// GPU head 상주 여부(bf16 output.weight 업로드 성공).
    head_gpu: bool,
    /// [MTP 2026-10-12] 모델 자체 MTP 초안기 활성(27B dense v1).
    mtp: bool,
}

/// [R16] 프리필 청크 tmax 정책 — 순수 함수(테스트 표면).
/// 규칙: MoE 스트리밍은 캐시 스테이징 기반 t≤8 청크(종전 t=1 폴백 — 512토큰
/// ~9.3s), 그 외는 `LLM170_PREFILL_T`(진단 오버라이드) 또는 기본 512
/// (가중치 상각), 1..=512.
pub fn prefill_tmax(is_moe: bool, resident: bool, env: Option<&str>) -> usize {
    if is_moe && !resident {
        return 8;
    }
    env.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        .clamp(1, 512)
}

impl GpuEngine {
    /// 모델 로드 + 가중치 상주 업로드(간헐 ENOENT 재시도는 호출부 소관).
    pub fn load(dir: &Path, n_slots: usize, ctx: usize) -> Result<Self, String> {
        let model = llm170_core::qwen35::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        upload_model(&mut dec, &model, ctx)?;
        // 5) head — bf16 output.weight는 GPU 커널 경로(아니면 CPU 참조).
        let mut head_gpu = false;
        if let Some(w) = model.w("output.weight")
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            dec.upload_head(w.data, w.n_out as usize, w.n_in as usize)?;
            head_gpu = true;
        }
        // [S3-1 2026-10-11] token_embd 상주 — 프리필 행 조립을 GPU gather로.
        if let Some(w) = model.w("token_embd.weight")
            && matches!(
                w.ty,
                llm170_core::wtype::WType::F16 | llm170_core::wtype::WType::Bf16
            )
        {
            dec.upload_embed(
                w.data,
                w.n_out as usize,
                w.n_in as usize,
                w.ty == llm170_core::wtype::WType::Bf16,
            )?;
        }
        let mtp = upload_mtp(&mut dec, &model)?;
        Ok(GpuEngine {
            model,
            dec,
            n_slots: n_slots.max(1),
            staged: llm170_diag::flag::on("LLM170_STAGED"),
            head_gpu,
            mtp,
        })
    }

    /// 메모리 분류(모니터링) — (상주 가중치, KV, CPU 오프로드, PLE 오프로드).
    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        self.dec.mem_stats()
    }

    /// 토큰당 활성 가중치 바이트(실효 대역폭 계산용).
    pub fn active_weight_bytes(&self) -> u64 {
        self.dec.active_weight_bytes()
    }

    /// MoE 배치 모드 — "none" | "resident" | "streaming".
    pub fn moe_mode(&self) -> &'static str {
        self.dec.moe_mode()
    }

    /// 복사 계측 — [h2d, d2h, d2d] × (바이트, ns, 호출).
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.dec.copy_stats()
    }

    /// 1토큰 체인(로짓 없이 상태만 진행) — prefill 중간 토큰용.
    fn forward(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.staged {
            self.dec.forward(slot, row)
        } else {
            self.dec.forward_device(slot, row)
        }
    }

    /// 1토큰 다음 로짓 — GPU head 상주 시 로짓만 회수(xn 판독 생략),
    /// 아니면 디바이스/스테이징 체인 + CPU 참조 head.
    /// [P3] greedy 토큰 — GPU 경로는 디바이스 argmax(로짓 d2h 1MB → 4B).
    fn token_greedy(&mut self, slot: usize, row: &[f32]) -> Result<u32, String> {
        if self.head_gpu && !self.staged {
            return self.dec.forward_device_argmax(slot, row);
        }
        let lg = self.logits(slot, row)?;
        Ok(llm170_core::matmul::greedy_from(&lg))
    }

    fn logits(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_gpu && !self.staged {
            return self.dec.forward_device_head(slot, row);
        }
        let xn = self.forward(slot, row)?;
        self.head_logits(&xn)
    }

    /// head 로짓 — CPU 참조 경로(bf16 디퀀트 f32 내적, 골든과 동일 계급).
    /// A9: 로더가 보증하더라도 패닉 대신 Result — 서버 오류 응답으로 유도.
    fn head_logits(&self, xn: &[f32]) -> Result<Vec<f32>, String> {
        let head = self
            .model
            .w("output.weight")
            .ok_or_else(|| "output.weight 부재 — head 계약 위반".to_string())?;
        let mut lg = vec![0.0f32; head.n_out as usize];
        llm170_core::matmul::matmul(xn, &head, &mut lg);
        Ok(lg)
    }

    /// 시퀀스 prefill → 마지막 로짓. t≤8 배치 청크(GEMM t≥2 경로)로 처리하고
    /// head는 마지막 토큰만 계산한다(중간 토큰 상각).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.model.hp.n_embd;
        let n = tokens.len();
        if n == 0 {
            // A10: 빈 프롬프트는 조용한 제로 로짓 대신 명시 오류.
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = None;
        if self.staged {
            // 구 스테이징 경로는 t=1 전용 — 한 토큰씩.
            for (i, tok) in tokens.iter().enumerate() {
                let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
                if i + 1 == n {
                    let xn = self.dec.forward(seq, &row)?;
                    last = Some(self.head_logits(&xn)?);
                } else {
                    self.dec.forward(seq, &row)?;
                }
            }
            return last.ok_or_else(|| "prefill: 빈 프롬프트".to_string());
        }
        // 청크 크기 정책 — 순수 함수(테스트 표면, [R16]).
        let tmax = prefill_tmax(
            self.dec.is_moe(),
            self.dec.moe_experts_resident(),
            llm170_diag::flag::val("LLM170_PREFILL_T"),
        );
        let mut i = 0usize;
        while i < n {
            let t = (n - i).min(tmax);
            let is_last = i + t == n;
            let part = &tokens[i..i + t];
            // [MTP] 청크 시작 위치(메인 forward 전에 캡처 — forward가 전진).
            let mtp_pos = if self.mtp {
                Some(self.dec.slot_pos_of(seq))
            } else {
                None
            };
            if t == 1 {
                let row = self.model.embed_row(part[0]).map_err(|e| e.to_string())?;
                if is_last && self.head_gpu {
                    last = Some(self.dec.forward_prefill(seq, &row, 1, true)?);
                } else if is_last {
                    let xn = self.dec.forward_prefill(seq, &row, 1, false)?;
                    last = Some(self.head_logits(&xn)?);
                } else {
                    self.dec.forward_prefill(seq, &row, 1, false)?;
                }
            } else if self.dec.embed_ready() {
                // [S3-1 2026-10-11] 임베딩 GPU gather — 행 조립(mmap 디코드+
                // 할당+h2d t×h×4B)을 id h2d(4B/행)+gather로 대체. 값 동일
                // (h2f/b2f = decode_f32 동일 함수 — 골든 계약).
                if is_last && self.head_gpu {
                    last = Some(self.dec.forward_prefill_toks(seq, part, t, true)?);
                } else if is_last {
                    let xn = self.dec.forward_prefill_toks(seq, part, t, false)?;
                    last = Some(self.head_logits(&xn)?);
                } else {
                    self.dec.forward_prefill_toks(seq, part, t, false)?;
                }
            } else {
                let mut rows = Vec::with_capacity(t * h);
                for &tok in part {
                    rows.extend_from_slice(&self.model.embed_row(tok).map_err(|e| e.to_string())?);
                }
                if is_last && self.head_gpu {
                    last = Some(self.dec.forward_prefill(seq, &rows, t, true)?);
                } else if is_last {
                    let xn = self.dec.forward_prefill(seq, &rows, t, false)?;
                    last = Some(self.head_logits(&xn)?);
                } else {
                    self.dec.forward_prefill(seq, &rows, t, false)?;
                }
            }
            // [MTP] 호출열 = (P의 hidden, P+1의 토큰) → P+2 초안 — 토큰 시프트.
            // 마지막 청크의 P=L-1은 토큰 at L이 없어 **유예**(마지막 hidden
            // 저장 → 첫 샘플 후 mtp_first).
            if let Some(ps) = mtp_pos {
                let (calls, save_row): (&[u32], Option<usize>) = if is_last {
                    (&tokens[i + 1..i + t], Some(t - 1))
                } else {
                    (&tokens[i + 1..=i + t], None)
                };
                let mut emb: Vec<f32> = Vec::with_capacity(calls.len() * h);
                for &tk in calls {
                    emb.extend_from_slice(&self.model.embed_row(tk).map_err(|e| e.to_string())?);
                }
                if let Err(err) = self.dec.mtp_prefill_chunk(seq, calls, &emb, ps, save_row) {
                    eprintln!("# mtp(무시): {err}");
                }
            }
            i += t;
        }
        last.ok_or_else(|| "prefill: 빈 프롬프트 — 토큰 1개 이상 필요".to_string())
    }

    /// 배치 디코드 — seq별 1토큰 forward → 로짓.
    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(seq_ids.len());
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.push(self.logits(*s, &row)?);
        }
        Ok(out)
    }

    /// 배치 greedy — 토큰만 회수.
    /// [A9 2026-10-10] t∈2..=BATCH_DEC_MAX이면 단일 체인 배치 경로(가중
    /// 상각)를 시도, 전제 미충족·실패 시 슬롯별 직렬로 폴백(로그 1회성).
    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        if (2..=llm170_backend_gpu::BATCH_DEC_MAX).contains(&seq_ids.len()) {
            let h = self.model.hp.n_embd;
            let mut rows = Vec::with_capacity(seq_ids.len() * h);
            for tok in tokens {
                rows.extend_from_slice(&self.model.embed_row(*tok).map_err(|e| e.to_string())?);
            }
            match self.dec.forward_device_argmax_batch(seq_ids, &rows) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    eprintln!("# np-batch 폴백(직렬): {e}");
                }
            }
        }
        let mut out = Vec::with_capacity(seq_ids.len());
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.push(self.token_greedy(*s, &row)?);
        }
        Ok(out)
    }

    /// [MTP 2026-10-12] 초안기 활성.
    pub fn mtp_enabled(&self) -> bool {
        self.mtp
    }

    /// [MTP] 라운드 전 — 검증 체인 dres의 수용 접두 acc행을 hidden 어셈블리로
    /// (롤백·보정 디코드 전에 호출해야 dres가 살아있다).
    pub fn mtp_pre_round(&mut self, acc: usize) -> Result<(), String> {
        self.dec.mtp_reset_hidden();
        for r in 0..acc {
            self.dec.mtp_push_hidden(r)?;
        }
        Ok(())
    }

    /// [MTP-2] 초안 폭 — 총 초안 수(첫 초안 + 재귀). 검증 t = 1+k ≤ 8(TR GEMV).
    const MTP_K: usize = 4;

    /// [MTP] 라운드 후 — 보정 행 hidden 추가 + t행 순전파 + **MTP-2 재귀**.
    /// pos_start = slot_pos - (수용+1) — 수용 초안·보정 토큰이 그 위치들이다.
    /// 첫 초안(draft_0)은 (P, P+1)→P+2 규칙 그대로, 이후 k-1개는 직전 fin을
    /// hidden으로 하는 재귀(각 런이 자기 위치 KV를 갱신).
    pub fn mtp_post_round(
        &mut self,
        slot: usize,
        accepted: &[u32],
        corrected: u32,
    ) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        self.dec.mtp_push_hidden(0)?;
        let mut toks: Vec<u32> = accepted.to_vec();
        toks.push(corrected);
        let mut embeds: Vec<f32> = Vec::with_capacity(toks.len() * self.model.hp.n_embd);
        for &tk in &toks {
            embeds.extend_from_slice(&self.model.embed_row(tk).map_err(|e| e.to_string())?);
        }
        let t = toks.len() as u32;
        let pos_start = self.dec.slot_pos_of(slot).saturating_sub(t);
        let draft = self.dec.mtp_run(slot, &toks, &embeds, pos_start)?;
        let mut drafts = vec![draft];
        let mut tok = draft;
        for i in 0..(Self::MTP_K.saturating_sub(1)) {
            let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
            let d = self
                .dec
                .mtp_recur_step(slot, tok, &emb, pos_start + t + i as u32)?;
            drafts.push(d);
            tok = d;
        }
        // lim=pos+t+1 슬랙이 읽는 pos+1 행의 이전 라운드 잔존분 제거.
        if let Err(err) = self
            .dec
            .mtp_zero_row(self.dec.slot_pos_of(slot) as usize + 1)
        {
            eprintln!("# mtp(무시): {err}");
        }
        self.dec.mtp_store_drafts(slot, &drafts);
        Ok(())
    }

    /// [MTP] 비-스펙 단독 디코드 후 — 리셋 + 1행 순전파 + 초안 보관.
    /// (슬롯 off면 no-op — 접두 재사용 슬롯.)
    pub fn mtp_single(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        self.mtp_pre_round(0)?;
        self.mtp_post_round(slot, &[], tok)
    }

    /// [MTP] 프리필 직후 첫 초안 — 유예 호출 (prev hidden, 첫 토큰) → 초안.
    pub fn mtp_first(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
        let d = self.dec.mtp_run_prev(slot, tok, &emb)?;
        let mut drafts = vec![d];
        let mut tk = d;
        // 유예 호출은 P = slot_pos-1 위치 → 재귀는 slot_pos..(k-2) 위치.
        let base = self.dec.slot_pos_of(slot);
        for i in 0..(Self::MTP_K.saturating_sub(1)) {
            let eb = self.model.embed_row(tk).map_err(|e| e.to_string())?;
            let nd = self.dec.mtp_recur_step(slot, tk, &eb, base + i as u32)?;
            drafts.push(nd);
            tk = nd;
        }
        self.dec.mtp_store_drafts(slot, &drafts);
        Ok(())
    }

    /// [MTP] 슬롯 재사용(접두 캐시) — KV 정합 없음: 초안 비활성/재개.
    pub fn mtp_slot_reused(&mut self, slot: usize) {
        self.dec.mtp_slot_off(slot);
    }

    pub fn mtp_slot_fresh(&mut self, slot: usize) {
        self.dec.mtp_slot_on(slot);
    }

    /// [MTP] 슬롯 KV 초기화(새 시퀀스 — 잔존 행 제거).
    pub fn mtp_slot_kv_reset(&mut self) -> Result<(), String> {
        self.dec.mtp_kv_reset()
    }

    /// [MTP] 초안 소비(현 위치 유효분만).
    pub fn mtp_take_draft(&mut self, slot: usize) -> Option<Vec<u32>> {
        self.dec.mtp_take_draft(slot)
    }

    /// 단일 greedy 디코드.
    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        let row = self.model.embed_row(token).map_err(|e| e.to_string())?;
        self.token_greedy(seq, &row)
    }

    /// [S11 2026-10-11] 혼합 배치(그리디+샘플 슬롯 한 배치) — 배치 체인이
    /// 토큰(argmax)과 로짓(dbatch_lg)을 동시 산출하므로 한 번에 회수한다.
    /// 반환 = (전 행 토큰, 전 행 로짓 평탄 [t][head_n]). 그리디 행은 토큰,
    /// 샘플 행은 로짓 사용 — 둘 다 단독 경로와 비트동일.
    pub fn decode_np_mixed(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        if !(2..=llm170_backend_gpu::BATCH_DEC_MAX).contains(&seq_ids.len()) {
            return Err("mixed: t 2..=BATCH_DEC_MAX 전용".into());
        }
        let h = self.model.hp.n_embd;
        let mut rows = Vec::with_capacity(seq_ids.len() * h);
        for tok in tokens {
            rows.extend_from_slice(&self.model.embed_row(*tok).map_err(|e| e.to_string())?);
        }
        let toks = self.dec.forward_device_argmax_batch(seq_ids, &rows)?;
        let lg = self.dec.batch_read_logits(seq_ids.len())?;
        Ok((toks, lg))
    }

    /// [S11 2026-10-11] 배치 디코드 로짓(샘플링 슬롯) — 배치 체인(argmax
    /// 그래프 재사용: dbatch_lg가 이미 로짓) + 로짓 d2h. 행별 로짓은 t=1
    /// forward_device_head와 비트동일 → 호스트 샘플러 토큰열 불변.
    /// 반환 = 행 우선 평탄 [t][head_n]. 전제 위반 시 슬롯별 직렬 폴백.
    pub fn decode_np_logits(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        if (2..=llm170_backend_gpu::BATCH_DEC_MAX).contains(&seq_ids.len()) {
            let h = self.model.hp.n_embd;
            let mut rows = Vec::with_capacity(seq_ids.len() * h);
            for tok in tokens {
                rows.extend_from_slice(&self.model.embed_row(*tok).map_err(|e| e.to_string())?);
            }
            match self.dec.forward_device_argmax_batch(seq_ids, &rows) {
                Ok(_tok_discard) => match self.dec.batch_read_logits(seq_ids.len()) {
                    Ok(lg) => return Ok(lg),
                    Err(e) => eprintln!("# np-logits 판독 폴백(직렬): {e}"),
                },
                Err(e) => eprintln!("# np-logits 배치 폴백(직렬): {e}"),
            }
        }
        let mut out = Vec::new();
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.extend_from_slice(&self.logits(*s, &row)?);
        }
        Ok(out)
    }

    /// [S13 2026-10-11] GPU 커널 범주 프로파일(LLM170_TIME=1 이벤트 계측) —
    /// serve 스텝 분해(srv_time 요약에 연동). off면 빈 문자열.
    pub fn prof_report(&mut self, label: &str) -> Result<String, String> {
        self.dec.prof_report(label)
    }

    /// 슬롯 상태 리셋(GDN 링/스캔 + pp + pos).
    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        self.dec.reset_state(seq)
    }

    /// [§7.1 2026-10-11] 스펙 서빙 — GDN 토큰별 스냅샷 버퍼 확보(모델당 1회).
    /// VRAM: 27B 654MB+dsave(슬롯당 ~154MB)·35B 436MB+dsave.
    /// [2026-10-12] 스펙 활성 여부 — 비활성(스트리밍·VRAM 부족)이면 스케줄러가
    /// 라운드를 시도하지 않는다.
    pub fn spec_on(&self) -> bool {
        self.dec.spec_on()
    }
    pub fn enable_spec(&mut self) -> Result<(), String> {
        self.dec.enable_spec()
    }
    pub fn spec_save_state(&mut self, slot: usize) -> Result<(), String> {
        self.dec.spec_save_state(slot)
    }
    pub fn spec_load_state(&mut self, slot: usize) -> Result<(), String> {
        self.dec.spec_load_state(slot)
    }
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        self.dec.spec_verify(slot, rows, t)
    }
    /// [§7 2단계] 검증 체인 로짓([t][vocab]) — 비greedy 거절샘플링용.
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        self.dec.spec_verify_logits(slot, rows, t)
    }
    pub fn vocab(&self) -> usize {
        self.model.hp.vocab
    }
    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        self.dec.spec_rollback(slot, keep)
    }
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        self.dec.spec_rewind_pos(slot, n)
    }
    /// [A2 2026-10-11] 접두 재사용 체크포인트 — 위치 조회(0=없음)/복원.
    pub fn checkpoint_pos(&self, slot: usize) -> u32 {
        self.dec.checkpoint_pos(slot)
    }
    pub fn restore_gdn(&mut self, slot: usize) -> Result<u32, String> {
        self.dec.restore_gdn(slot)
    }
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, String> {
        self.model.embed_row(tok).map_err(|e| e.to_string())
    }
    pub fn n_embd(&self) -> usize {
        self.model.hp.n_embd
    }

    /// 전 슬롯 리셋(서버 워밍업 종료 후).
    pub fn reset_states(&mut self) -> Result<(), String> {
        for s in 0..self.n_slots {
            self.dec.reset_state(s)?;
        }
        // [MTP] 층 슬라이스 KV도 리셋(본체 reset_state는 본체 층만).
        self.dec.mtp_kv_reset()?;
        Ok(())
    }

    /// 표면형 근사 디토크(표시용 — 정식 BPE 디토크는 TOKENIZER 몫).
    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(String::as_str)
            .unwrap_or("")
            .replace('Ġ', " ")
            .replace('Ċ', "\n")
    }
}

/// MoE(35B-A3B) 상주 배선 — 플레인 bf16 업로드 + 전문가 슬라이스 테이블.
/// GpuEngine·w4a16-gpu 프로브 공용.
///
/// SAFETY(전문가 테이블): 항목은 모델 스토어의 mmap 슬라이스 주소다 —
/// 호출자가 모델을 디코더와 함께 보유하는 수명 계약(디코더가 먼저 drop).
pub(crate) fn upload_moe(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    for name in model.engine_names() {
        // token_embd(호스트 embed_row)·output(head 전용 업로드)는 체인이
        // 선형으로 쓰지 않는다 — VRAM 절약(각 ~970MiB @35B).
        if name == "token_embd.weight" || name == "output.weight" {
            continue;
        }
        if let Some(w) = model.w_raw(&name)
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            dec.upload_plain(&name, w.data, w.n_out as usize, w.n_in as usize)?;
        }
    }
    let plain_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    dec.set_plain_mode(true);
    let (group, scale_bf16) = model.expert_quant();
    dec.set_moe(
        hp.n_experts,
        hp.top_k,
        hp.moe_ffn,
        hp.shared_ffn,
        group,
        scale_bf16,
    );
    let mut tab: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(hp.n_layer * hp.n_experts * 3);
    for il in 0..hp.n_layer {
        for e in 0..hp.n_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let (q, s, _, _) = model
                    .expert_slice(il, e, proj)
                    .ok_or_else(|| format!("전문가 슬라이스 부재: L{il} e{e} {proj}"))?;
                tab.push((
                    q.as_ptr() as u64,
                    q.len() as u64,
                    s.as_ptr() as u64,
                    s.len() as u64,
                ));
            }
        }
    }
    // 상주 가능(170HX 64GB 등)이면 전문가 전량 VRAM — 토큰당 PCIe 스트리밍 제거.
    // 판정: 전문가 바이트 + 1GiB 여유. 불가면 스트리밍 테이블(호스트 mmap).
    let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
    let free = llm170_backend_gpu::cuda_mem_free()
        .map(|(f, _)| f)
        .unwrap_or(0);
    let resident = match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
        Some("0") => false,
        Some("1") => true,
        _ => free > expert_bytes + (1 << 30),
    };
    let tab_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    if resident {
        dec.upload_experts_resident(&tab)?;
    } else {
        dec.set_expert_table(tab);
    }
    eprintln!(
        "[moe] 플레인 {plain_ms:.0}ms · 테이블 {tab_ms:.0}ms · 전문가 {:.0}ms ({:.1}GB/s) — 전문가 {:.2}GiB · 여유 {:.2}GiB → {}",
        t2.elapsed().as_secs_f64() * 1e3,
        (expert_bytes as f64 * 1e-9) / t2.elapsed().as_secs_f64().max(1e-9),
        expert_bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        if resident {
            "전량 상주"
        } else {
            "호스트 스트리밍"
        }
    );
    Ok(())
}

/// 모델 상주 업로드(1~4단계) — GpuEngine·w4a16-gpu 프로브 공용.
/// head는 호출부 소관(프로브는 --no-head/스테이징 분기가 있다).
pub(crate) fn upload_model(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    ctx: usize,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let interval = hp.full_attn_interval.max(1);
    if interval != 4 {
        return Err(format!(
            "디코더는 4층 주기(full_attention_interval) 전용 — interval={interval}"
        ));
    }
    let moe = hp.n_experts > 0;
    // 1) 선형 상주 — GPU 체인은 원본(HF) 무게(순열은 커널 내부 처리).
    if moe {
        upload_moe(dec, model)?;
    } else {
        for name in model.engine_names() {
            if let Some(w) = model.w_raw(&name)
                && w.ty == llm170_core::wtype::WType::W4a16Split
            {
                let s = w.aux.ok_or_else(|| format!("{name}: aux 부재"))?;
                dec.upload_lin(&name, w.data, s, w.n_out as usize, w.n_in as usize)?;
            }
        }
    }
    // 2) 노름 nw [2L+1][hidden] (+1 보정 — f32_vec).
    let mut nw: Vec<f32> = Vec::new();
    for il in 0..hp.n_layer {
        nw.extend(
            model
                .f32_vec(&format!("blk.{il}.attn_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
        nw.extend(
            model
                .f32_vec(&format!("blk.{il}.post_attention_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
    }
    nw.extend(
        model
            .f32_vec("output_norm.weight")
            .map_err(|e| e.to_string())?,
    );
    dec.set_norm_weights(&nw, 2 * hp.n_layer + 1)?;
    // 3) GDN 상수 — 원본(HF) 순서(a 먼저 b 다음 — 커널 ab 색인).
    let n_gdn = hp.n_layer - hp.n_layer / interval;
    let gd = GdnDims {
        n_gdn,
        hidden: hp.n_embd,
        h_k: hp.n_group,
        h_v: hp.dt_rank,
        d: hp.d_state,
    };
    let (mut cw, mut ab, mut alog, mut dtb, mut gnw): (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for il in 0..hp.n_layer {
        if (il + 1) % interval == 0 {
            continue;
        }
        cw.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))
                .map_err(|e| e.to_string())?,
        );
        ab.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_alpha.weight"))
                .map_err(|e| e.to_string())?,
        );
        ab.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_beta.weight"))
                .map_err(|e| e.to_string())?,
        );
        alog.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_a"))
                .map_err(|e| e.to_string())?,
        );
        dtb.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_dt.bias"))
                .map_err(|e| e.to_string())?,
        );
        gnw.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
    }
    dec.set_gdn(gd, &cw, &ab, &alog, &dtb, &gnw)?;
    // 4) 어텐션 q/k 노름(+1 — f32_vec 보정).
    // [MTP 2026-10-12] MTP 존재 시 가상 전체-어텐션 층 1개 추가(n_attn+1,
    // 마지막 인덱스) — KV 1층 확장 + q/k_norm 행 추가. 커널 무수정.
    let mtp_attn = model
        .w_raw("mtp.layers.0.self_attn.q_proj.weight")
        .is_some();
    let n_attn = hp.n_layer / interval + usize::from(mtp_attn);
    let ad = AttnDims {
        n_attn,
        q_heads: hp.n_head,
        kv_heads: hp.n_kv,
        d: hp.head_dim,
        cap: ctx,
        // [A11] 모델 config에서 동적(종전 하드코딩 4) — 신규 비율 모델 대응.
        interval,
    };
    let (mut qnw, mut knw): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for ai in 0..n_attn - usize::from(mtp_attn) {
        let il = ai * interval + interval - 1;
        qnw.extend(
            model
                .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
        knw.extend(
            model
                .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
    }
    if mtp_attn {
        qnw.extend(
            model
                .f32_vec("mtp.layers.0.self_attn.q_norm.weight")
                .map_err(|e| e.to_string())?,
        );
        knw.extend(
            model
                .f32_vec("mtp.layers.0.self_attn.k_norm.weight")
                .map_err(|e| e.to_string())?,
        );
    }
    dec.set_attn(ad, &qnw, &knw)?;
    Ok(())
}

/// [MTP 2026-10-12] MTP 블록 업로드 — 27B dense 한정(v1). 35B MoE MTP는
/// 전문가 스택(3D bf16)이라 차기. 성공 시 true.
pub(crate) fn upload_mtp(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
) -> Result<bool, String> {
    // [MTP 2026-10-12 — 파킹(정확성 수정 완료)] 정확성 결함 3건을 모두
    // 수정해 **수용률 80~82% 실측**(초안=정답 일치). 수정 내역:
    //  (1) 꼬리(8~11단계: o_proj→post_attention_layernorm→MLP→mtp.norm)
    //      미구현 — head가 어텐션 前 x를 소비하던 것(CPU 대조 rel≈1e-7 일치).
    //  (2) norm_resid의 ab 판독이 t행 전체인데 zero 버퍼가 1행(h)이라 t≥2
    //      에서 OOB(행1+ NaN) → 프리필 K/V 오염 → 전 어텐션 NaN.
    //  (3) bind의 노름 매칭 ends_with("norm.weight")가 pre_fc_norm_hidden/
    //      embedding을 놓쳐 +1 없이 raw 사용(그 두 텐서만 zero-centered).
    //  (4) hidden 소스 = pre-norm 잔차(dres) — vLLM speculator.py 298
    //      "(logits_hidden, feedback_hidden) for final-norm correctness".
    // 파킹 사유: **1-초안 스펙이 순손실** — MTP 콜당 런치 오버헤드 ~5.5ms
    // (연산 0.2ms, ~20런치×250µs) + 검증/보정 고정비 → 서빙 38 t/s
    // (평범 디코드 ~50+). **MTP-2(k=4) 구현·실측 완료**: 라운드당 2.62토큰
    // (부분수용 71%)이나 콜 4회 오버헤드(+18ms)로 67.7ms/라운드 = 38.7 t/s.
    // **[MTP-GRAPH 구현·실측 2026-10-12]** t=1..5 핀 임베드+캡처 인프라
    // (ensure_mtp_graph·mtp_fwd_launch 분리·graph_invalidate take 계약) —
    // 캡처 5개 성공에도 **라운드 67.7→68.9ms(무이득)**: 이 기기의 cuGraphLaunch
    // 자체가 느려 콜 비용 ~5ms는 런치 오버헤드가 아니었다. 라운드 분해 =
    // 검증 t=5 ~30ms + 보정 t=1 18ms + GDN 스냅샷 ~15ms + MTP ~3ms.
    // 재개 절차(갱신): ①GDN 스냅샷 축소(f16 §5 또는 증분) ②검증 marlin급
    // ③MTP-2를 **단일 그래프 내 디바이스 체인**(초안→GPU 임베드 gather→
    // 다음 런 — 호스트 왕복 제거)으로 재구성 → A/B 목표 1.4× → 게이트 3 →
    // MTP_ENABLED=true → 스모크·eval.
    const MTP_ENABLED: bool = false;
    if !MTP_ENABLED || model.w_raw("mtp.fc.weight").is_none() {
        return Ok(false);
    }
    if model
        .w_raw("mtp.layers.0.mlp.experts.gate_up_proj")
        .is_some()
    {
        return Ok(false); // MoE MTP — 차기
    }
    let mut q_out = 0usize;
    for name in [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ] {
        let w = model
            .w_raw(name)
            .ok_or_else(|| format!("mtp: {name} 부재"))?;
        if name.ends_with("attn.q_proj.weight") {
            q_out = w.n_out as usize;
        }
        dec.upload_plain(name, w.data, w.n_out as usize, w.n_in as usize)?;
    }
    let mut norms: Vec<f32> = Vec::new();
    for name in [
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ] {
        norms.extend(model.f32_vec(name).map_err(|e| e.to_string())?);
    }
    let hp = &model.hp;
    let ffn = model
        .w_raw("mtp.layers.0.mlp.gate_proj.weight")
        .map(|w| w.n_out as usize)
        .unwrap_or(0);
    dec.set_mtp(&norms, q_out, hp.n_kv * hp.head_dim, ffn)?;
    let _ = hp;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::prefill_tmax;

    /// [R16] 프리필 tmax 정책 — 스트리밍 MoE 폴백·env 오버라이드·상한.
    #[test]
    fn prefill_tmax_policy() {
        assert_eq!(
            prefill_tmax(true, false, None),
            8,
            "MoE 스트리밍 = t=1 폴백"
        );
        assert_eq!(prefill_tmax(true, true, None), 512, "MoE 상주 = 기본 512");
        assert_eq!(prefill_tmax(false, false, None), 512, "dense = 기본 512");
        assert_eq!(
            prefill_tmax(false, false, Some("128")),
            128,
            "env 오버라이드"
        );
        assert_eq!(
            prefill_tmax(false, false, Some("99999")),
            512,
            "상한 클램프"
        );
        assert_eq!(prefill_tmax(false, false, Some("0")), 1, "하한 클램프");
        assert_eq!(
            prefill_tmax(false, false, Some("abc")),
            512,
            "파싱 실패 = 기본"
        );
    }
}
