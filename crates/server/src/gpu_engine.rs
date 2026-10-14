//! GPU 서빙 엔진 — W4A16 CUDA 디바이스 체인 + GPU head.
//!
//! W3-3 착륙 후 성능 캠페인(2026-10-08): 활성은 디바이스 상주(왕복은
//! 토큰당 임베딩 업로드 + 로짓 판독뿐), head는 head_bf16 커널(bf16
//! →f32 시프트 + k 직렬 f32 내적 — CPU 참조와 동일 순서).
//! 판정 기준은 토큰열(골든)이다.

use llm170_backend_gpu::{AttnDims, GdnDims, W4a16Dec};
use std::path::Path;

pub struct QwenEngine {
    model: llm170_core::qwen35::Model,
    dec: W4a16Dec,
    #[allow(dead_code)] // reset_states(진단/복구) 전용
    n_slots: usize,
    /// GPU head 상주 여부(bf16 output.weight 업로드 성공).
    head_gpu: bool,
    /// [gemma4 P2 2026-10-14] 최종 로짓 softcap(0=미적용) — 장치 argmax는
    /// 단조 불변이라 커널 무접촉, **호스트로 나오는 로짓**(샘플링·PPL/eval·
    /// 검증)에만 적용한다. 값은 config.json 파생(모델별).
    softcap: f32,
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

impl QwenEngine {
    /// 모델 로드 + 가중치 상주 업로드(간헐 ENOENT 재시도는 호출부 소관).
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
    ) -> Result<Self, String> {
        let model = llm170_core::qwen35::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        // [K6-3 예산] 풀 예산(여유 VRAM 파생) — upload_model(내부 set_attn) 전에 설정.
        dec.set_kv_budget(kv_budget);
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        // [D2 검수 2026-10-13] 적재 실패 시 부분 할당 정리 — 재시도 루프가
        // VRAM을 되찾지 못하던 문제(첫 시도 실패 후 전부 OOM)의 해소.
        match Self::fill(&mut dec, &model, ctx, kv_reserve.unwrap_or(0)) {
            Ok((head_gpu, mtp)) => Ok(QwenEngine {
                model,
                dec,
                n_slots: n_slots.max(1),
                head_gpu,
                softcap: llm170_core::qwen35::bind::final_logit_softcap(dir),
                mtp,
            }),
            Err(e) => {
                dec.free_all_device();
                Err(e)
            }
        }
    }

    /// [D2 검수 2026-10-13] 컨텍스트 생성 **후** 적재 본체(head·embed·MTP).
    /// 실패는 호출부(load)가 `dec.free_all_device()`로 정리한다.
    fn fill(
        dec: &mut W4a16Dec,
        model: &llm170_core::qwen35::Model,
        ctx: usize,
        kv_reserve: u64,
    ) -> Result<(bool, bool), String> {
        upload_model(dec, model, ctx, kv_reserve)?;
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
        let mtp = upload_mtp(dec, model)?;
        Ok((head_gpu, mtp))
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
        self.dec.forward_device(slot, row)
    }

    /// 1토큰 다음 로짓 — GPU head 상주 시 로짓만 회수(xn 판독 생략),
    /// 아니면 디바이스 체인 + CPU 참조 head.
    /// [P3] greedy 토큰 — GPU 경로는 디바이스 argmax(로짓 d2h 1MB → 4B).
    fn token_greedy(&mut self, slot: usize, row: &[f32]) -> Result<u32, String> {
        if self.head_gpu {
            return self.dec.forward_device_argmax(slot, row);
        }
        let lg = self.logits(slot, row)?;
        Ok(llm170_core::matmul::greedy_from(&lg))
    }

    fn logits(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_gpu {
            let mut lg = self.dec.forward_device_head(slot, row)?;
            self.softcap_row(&mut lg);
            return Ok(lg);
        }
        let xn = self.forward(slot, row)?;
        let mut lg = self.head_logits(&xn)?;
        self.softcap_row(&mut lg);
        Ok(lg)
    }

    /// 최종 로짓 softcap 적용(0 = 무연산) — 호스트 로짓 경로 단일 지점.
    fn softcap_row(&self, lg: &mut [f32]) {
        if self.softcap > 0.0 {
            for v in lg.iter_mut() {
                *v = llm170_core::ops::logit_softcap(*v, self.softcap);
            }
        }
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

    /// [R5] 토큰 열 → 임베딩 행 평탄 조립(t×h) — 배치 디코드 부트스트랩
    /// 공용(동작 불변: 토큰별 embed_row 동일 순서·동일 오류 전파).
    fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.model.hp.n_embd;
        let mut rows = Vec::with_capacity(tokens.len() * h);
        for tok in tokens {
            rows.extend_from_slice(&self.model.embed_row(*tok).map_err(|e| e.to_string())?);
        }
        Ok(rows)
    }

    /// [R5] 배치 디코드 부트스트랩 공용 골격 — t∈2..=BATCH_DEC_MAX 전제
    /// 검사 + 임베딩 행 조립 + 배치 argmax 체인 실행. Some(토큰) = 배치
    /// 성공, None = 전제 미충족 또는 체인 실패(직렬 폴백 — 폴백 로그는
    /// `label`로 호출부 원문 유지, 동작 불변). 행 조립 실패만 Err 전파.
    fn np_batch_argmax(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
        label: &str,
    ) -> Result<Option<Vec<u32>>, String> {
        if (2..=llm170_backend_gpu::BATCH_DEC_MAX).contains(&seq_ids.len()) {
            let rows = self.embed_rows(tokens)?;
            match self.dec.forward_device_argmax_batch(seq_ids, &rows) {
                Ok(toks) => return Ok(Some(toks)),
                Err(e) => eprintln!("# {label} 폴백(직렬): {e}"),
            }
        }
        Ok(None)
    }

    /// 시퀀스 prefill → 마지막 로짓. t≤8 배치 청크(GEMM t≥2 경로)로 처리하고
    /// head는 마지막 토큰만 계산한다(중간 토큰 상각).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let n = tokens.len();
        if n == 0 {
            // A10: 빈 프롬프트는 조용한 제로 로짓 대신 명시 오류.
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = None;
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
            // [MTP-CHAIN §3.3] 강제 n-gram 시 MTP 프리필 KV 구축도 스킵
            // (초안을 만들지 않으면 읽을 곳이 없다 — [TC] mma mtp 라인 없음).
            let mtp_pos = if self.mtp_drafts_on() {
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
                // [R5] 행 조립 — embed_rows 공용(동작 불변).
                let rows = self.embed_rows(part)?;
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
                // [R5] 행 조립 — embed_rows 공용(동작 불변).
                let emb = self.embed_rows(calls)?;
                if let Err(err) = self.dec.mtp_prefill_chunk(seq, calls, &emb, ps, save_row) {
                    eprintln!("# mtp(무시): {err}");
                }
            }
            i += t;
        }
        last.ok_or_else(|| "prefill: 빈 프롬프트 — 토큰 1개 이상 필요".to_string())
    }

    /// 배치 디코드 — seq별 1토큰 forward → 로짓.
    /// [R5] 행 조립을 embed_rows 공용으로(동작 불변 — 로짓 슬라이스 동일).
    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let rows = self.embed_rows(tokens)?;
        let h = self.model.hp.n_embd;
        let mut out = Vec::with_capacity(seq_ids.len());
        for (i, &s) in seq_ids.iter().enumerate() {
            out.push(self.logits(s, &rows[i * h..(i + 1) * h])?);
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
        // [R5] 부트스트랩 골격 공용 — 전제 검사·행 조립·배치 체인(로그 원문 유지).
        if let Some(v) = self.np_batch_argmax(seq_ids, tokens, "np-batch")? {
            return Ok(v);
        }
        let mut out = Vec::with_capacity(seq_ids.len());
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.push(self.token_greedy(*s, &row)?);
        }
        Ok(out)
    }

    /// [MTP-CHAIN §3.3 2026-10-12] 초안 소스 정책 — MTP 헤드 가용 + 강제
    /// n-gram 아님. LLM170_SPEC_DRAFT_SOURCE: "mtp"(기본/미설정 — MTP 가용
    /// 시 MTP, 스테일·부재 시 n-gram 폴백) | "ngram"(강제 n-gram — MTP
    /// 상태 구축[프리필 KV]·생산[mtp_first/single/post_round] 전면 스킵,
    /// 종전 파킹 경로와 동일). 초안은 검증이 정본이라 이 게이트는 무손실
    /// (운영 정책 — 수치 경로 분기 아님, 단일화 방침 준수).
    pub fn mtp_drafts_on(&self) -> bool {
        // [2.J] CLI 정책 우선 — 1=mtp 강제(가용 시) · 2=ngram 강제 · 0=auto(env 진단).
        match crate::engine::SPEC_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
            1 => self.mtp,
            2 => false,
            _ => self.mtp && llm170_diag::flag::val("LLM170_SPEC_DRAFT_SOURCE") != Some("ngram"),
        }
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

    /// [MTP-CHAIN §3.3 ② 2026-10-12] 앵커 직후 k-1 재귀 스텝을 장치 체인
    /// (그래프 리플레이 — 호스트 왕복 0)으로 치환해 초안 [d_1..d_k] 반환.
    /// t>1 앵커는 진입 계약 시드(mtp_chain_seed — 토큰 레지스터·dpp)가
    /// 선행, t=1은 멱등 재기입. 체인 실패(계약·캡처 — 리플레이 前 판정)는
    /// Err → 호출부가 순차 재귀(mtp_recur_step, 스텝당 sync+회수 왕복)로
    /// 폴백. 순차·체인 두 경로의 소비 입력(ptr5 행0·d_1·위치)은 동일 —
    /// 토큰열 판정 동일(모듈 계약 승계).
    fn mtp_chain_round(&mut self, slot: usize, d1: u32, base_pos: u32) -> Result<Vec<u32>, String> {
        self.dec.mtp_chain_seed(slot, d1, base_pos)?;
        self.dec.mtp_chain_drafts(slot, Self::MTP_K, base_pos)
    }

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
        // 보정행 hidden = 액티브 경로가 전진시킨 현재 dres 행 0.
        // (pre_round(acc+1) 융합은 실측 기각 — sched 주석 참조.)
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
        // [2.A-③ 2026-10-13] 재귀 시드 = 앵커 마지막 행(t-1)의 fin — 체인/순차
        // 재귀의 "ptr5 행 0 복사" 계약에 맞춰 정규화한다(종전 t≥2에서 행 0을
        // 읽어 d_2+가 스테일 조건에서 생성되던 결함). t=1은 무연산.
        self.dec.mtp_fin_seed(t as usize - 1)?;
        // [MTP-CHAIN §3.3] 재귀 k-1스텝 = 체인 1회(호스트 왕복 0 — 종전
        // 4콜 ≈10.4ms/라운드 → 그래프 리플레이). base_pos = 앵커 마지막 행
        // 위치(pos_start+t-1): 리플레이의 dpp+1 → 첫 스텝 pos_start+t =
        // 순차 경로와 동일 위치. 실패(계약·캡처) 시 순차 재귀 폴백.
        let base_pos = pos_start + t - 1;
        let drafts = match self.mtp_chain_round(slot, draft, base_pos) {
            Ok(ds) => ds,
            Err(err) => {
                eprintln!("# mtp-chain(폴백 순차): {err}");
                let mut ds = vec![draft];
                let mut tok = draft;
                for i in 0..(Self::MTP_K.saturating_sub(1)) {
                    let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
                    let d = self
                        .dec
                        .mtp_recur_step(slot, tok, &emb, pos_start + t + i as u32)?;
                    ds.push(d);
                    tok = d;
                }
                ds
            }
        };
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
        // [MTP-CHAIN §3.3] 유예 앵커(t=1) 위치 = 저장 시점 slot_pos-1
        // (프리필 완료 직후 호출 계약 — 디코드가 slot_pos를 전진시키기 전).
        // base = 그 위치 → 리플레이 dpp+1 = 첫 스텝 slot_pos(순차 base+0과
        // 동일). 실패 시 순차 재귀 폴백.
        let base_pos = self.dec.slot_pos_of(slot).saturating_sub(1);
        let drafts = match self.mtp_chain_round(slot, d, base_pos) {
            Ok(ds) => ds,
            Err(err) => {
                eprintln!("# mtp-chain(폴백 순차): {err}");
                let mut ds = vec![d];
                let mut tk = d;
                for i in 0..(Self::MTP_K.saturating_sub(1)) {
                    let eb = self.model.embed_row(tk).map_err(|e| e.to_string())?;
                    let nd = self
                        .dec
                        .mtp_recur_step(slot, tk, &eb, base_pos + 1 + i as u32)?;
                    ds.push(nd);
                    tk = nd;
                }
                ds
            }
        };
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

    /// [K-STEP §3.4 ① 2026-10-12] 단독 그리디 K-스텝 디코드 — 체인 스텝
    /// 그래프를 k회 리플레이해 호스트 체크인을 K스텝마다로. 방출 토큰열은
    /// 순차 단일 스텝(decode_greedy ×k)과 동일(토큰열 판정 — kstep 모듈
    /// 계약, 패리티 테스트 승계). 실패(계약·캡처) 시 Err → 호출부(sched
    /// 단독 그리디 분기)가 기존 토큰당 경로로 폴백.
    pub fn decode_greedy_k(
        &mut self,
        seq: usize,
        token: u32,
        k: usize,
        eos: u32,
    ) -> Result<Vec<u32>, String> {
        if !self.head_gpu {
            return Err("kstep: GPU head 미상주".into());
        }
        self.dec.decode_k_greedy(seq, token, k, eos)
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
        // [R5] 행 조립 — embed_rows 공용(동작 불변).
        let rows = self.embed_rows(tokens)?;
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
        // [R5] 부트스트랩 골격 공용 — 전제 검사·행 조립·배치 체인(로그 원문 유지).
        if self
            .np_batch_argmax(seq_ids, tokens, "np-logits 배치")?
            .is_some()
        {
            match self.dec.batch_read_logits(seq_ids.len()) {
                Ok(mut lg) => {
                    self.softcap_row(&mut lg);
                    return Ok(lg);
                }
                Err(e) => eprintln!("# np-logits 판독 폴백(직렬): {e}"),
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

    /// 슬롯 상태 리셋(GDN 링/스캔 + pp + pos). [K6-3] KV 블록 반납 포함 —
    /// 정상 완료(접두 캐시 유지)는 리셋을 부르지 않으므로 영향 없음.
    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        self.dec.kv_release(seq);
        self.dec.reset_state(seq)
    }

    /// [K6-3 2026-10-13] 슬롯 KV 블록 확보(온디맨드·입장) — 신규 블록 수.
    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        self.dec.kv_grow(slot, n_tokens)
    }

    /// [K6-3] 풀 여유 블록 수.
    pub fn kv_free_blocks(&self) -> usize {
        self.dec.kv_free_blocks()
    }

    /// [K6-3] 풀 총 블록 수(용량).
    pub fn kv_total_blocks(&self) -> usize {
        self.dec.kv_total_blocks()
    }

    /// [K6-3] 슬롯 KV 반납(유휴 슬롯 접두 캐시 회수 — 압박 사다리 1단계).
    pub fn kv_release_slot(&mut self, slot: usize) -> usize {
        self.dec.kv_release(slot)
    }

    /// [K6-3] n_tokens 예약 필요 블록 수(입장 판정).
    pub fn kv_blocks_needed(&self, n_tokens: usize) -> usize {
        self.dec.kv_blocks_needed(n_tokens)
    }

    /// [§7.1 2026-10-11] 스펙 서빙 — WY 표현 버퍼 확보(모델당 1회).
    /// [dsave 대체 2026-10-14] 정밀 상태 스냅샷(dsave)은 프로브 전용 지연
    /// 할당으로 강등 — serve 경로 상주분(27B/4슬롯 315MiB, dring0 5.6MiB
    /// 대체)은 예약·컨텍스트로 환원됐다(RESERVE 2.25GiB→1995MiB).
    /// [2026-10-12] 스펙 활성 여부 — 비활성(스트리밍·VRAM 부족)이면 스케줄러가
    /// 라운드를 시도하지 않는다.
    pub fn spec_on(&self) -> bool {
        self.dec.spec_on()
    }
    /// 정지 토큰 — Q35 아키텍처 상수.
    pub fn eos(&self) -> u32 {
        llm170_core::qwen35::EOS_EOT
    }
    pub fn enable_spec(&mut self) -> Result<(), String> {
        self.dec.enable_spec()
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
        let mut lg = self.dec.spec_verify_logits(slot, rows, t)?;
        self.softcap_row(&mut lg);
        Ok(lg)
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
    /// [A2→K3 2026-10-12] 접두 재사용 체크포인트 — upto 이하 최대 경계
    /// 조회(0=없음)/복원(512 경계 스택 4개).
    pub fn checkpoint_pos(&self, slot: usize, upto: u32) -> u32 {
        self.dec.checkpoint_pos(slot, upto)
    }
    pub fn restore_gdn(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        self.dec.restore_gdn(slot, upto)
    }
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, String> {
        self.model.embed_row(tok).map_err(|e| e.to_string())
    }
    pub fn n_embd(&self) -> usize {
        self.model.hp.n_embd
    }

    /// 전 슬롯 리셋(서버 워밍업 종료 후).
    #[allow(dead_code)] // [K6-3] 워밍업은 reset_seq(0)(KV 반납 포함)로 대체 — 진단/복구용 유지
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
    kv_reserve: u64,
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
        // [gemma4 P2 2026-10-14] 전문가 GLU 활성 — qwen은 silu(기본), gemma4는
        // gelu_tanh(배선 시 true). 모델별 단일 경로(런타임 수치 플래그 아님).
        false,
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
    // [2.J] CLI 정책 우선 — 1=resident 강제 · 2=streaming 강제 · 0=auto(env/여유).
    // [2.M 사다리] auto는 **KV 우선**: KV 풀 몫(kv_reserve — 서버 산출)을 떼고
    // 남는 VRAM으로만 상주 판정. KV가 ctx를 보장한 뒤 남는 잔여가 전문가 핫셋.
    // (kv_reserve=0 = 종전 규칙 — 프로브 경로.)
    let resident = match crate::engine::MOE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
            Some("0") => false,
            Some("1") => true,
            _ => free > expert_bytes + kv_reserve + (1 << 30),
        },
    };
    let tab_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    if resident {
        dec.upload_experts_resident(&tab)?;
    } else {
        dec.set_expert_table(tab);
    }
    eprintln!(
        "[moe] 플레인 {plain_ms:.0}ms · 테이블 {tab_ms:.0}ms · 전문가 {:.0}ms ({:.1}GB/s) — 전문가 {:.2}GiB · 여유 {:.2}GiB − KV 예약 {:.2}GiB → {}",
        t2.elapsed().as_secs_f64() * 1e3,
        (expert_bytes as f64 * 1e-9) / t2.elapsed().as_secs_f64().max(1e-9),
        expert_bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        kv_reserve as f64 / (1u64 << 30) as f64,
        if resident {
            "전량 상주"
        } else {
            "호스트 스트리밍"
        }
    );
    Ok(())
}

/// [gemma4 P2-3 2026-10-14] gemma4 v0 체인(호스트 스테이징) 호스트 설정 —
/// 서버 프로브가 층 산술(노름·잔차·layer_scalar)을 조립하는 데 필요한 것 전부.
/// 행 순서·계약 출처 = CPU 오라클(`crates/core/src/gemma4/`).
pub(crate) struct G4HostCfg {
    pub softcap: f32,
    /// 노름 행 8L+1 × hidden — [input, post_attn, pre_ff, post_ff1, pre_ff2,
    /// post_ff2, post_ff, router(=moe_scale·h^-0.5)] × 층 + output_norm.
    pub nw: Vec<f32>,
    /// 층별 layer_scalar(출력 곱).
    pub layer_scalar: Vec<f32>,
    /// [L][n_experts] per-expert 사전 스케일(μ_e).
    pub pes: Vec<f32>,
    /// QK 노름 [L][d_max](층별 d까지 유효 — 나머지 0 패딩).
    pub qnw: Vec<f32>,
    pub knw: Vec<f32>,
    pub d_max: usize,
}

impl G4HostCfg {
    pub fn nw_row(&self, il: usize, k: usize, hidden: usize) -> &[f32] {
        &self.nw[(il * 8 + k) * hidden..(il * 8 + k + 1) * hidden]
    }

    pub fn out_norm_row(&self, n_layer: usize, hidden: usize) -> &[f32] {
        &self.nw[n_layer * 8 * hidden..(n_layer * 8 + 1) * hidden]
    }

    pub fn qk_rows(&self, il: usize, d: usize) -> (&[f32], &[f32]) {
        let b = il * self.d_max;
        (&self.qnw[b..b + d], &self.knw[b..b + d])
    }

    pub fn pes_row(&self, il: usize, n_exp: usize) -> &[f32] {
        &self.pes[il * n_exp..(il + 1) * n_exp]
    }
}

/// [gemma4 P2-3 2026-10-14] gemma4 상주 업로드 — qwen35 `upload_model` 미러.
/// ① 양자화 선형(층별 q/k/(v)/o + dense gate/up/down + 라우터) ② 노름 8행/층
/// (라우터 입력 행 = moe_scale·h^-0.5 — CPU route의 가중 RMS와 동일 산식)
/// ③ QK 노름(d_max 패딩) + AttnLayer 테이블(층별 d/kv/sliding/k_eq_v)
/// ④ MoE(무 shared·gelu 전문가·g32 테이블) ⑤ head(원본 embed — 행별 int8)
/// ⑥ 호스트 설정 반환(노름 사본·rope·layer_scalar·pes).
pub(crate) fn upload_model_g4(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    ctx: usize,
) -> Result<G4HostCfg, String> {
    use llm170_core::wtype::WType;
    let hp = model.hp.clone();
    let h = hp.n_embd;
    let t0 = std::time::Instant::now();
    // ① 선형 — 밀집(양자화) + 라우터(플레인/양자화 무관 분기).
    for il in 0..hp.n_layer {
        for suf in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "moe_gate.weight",
        ] {
            let name = format!("blk.{il}.{suf}");
            let Some(w) = model.w(&name) else {
                continue; // k_eq_v 층의 v_proj 부재
            };
            match w.ty {
                WType::W4a16Split => {
                    let s = w.aux.ok_or_else(|| format!("{name}: aux 부재"))?;
                    dec.upload_lin_g(
                        &name,
                        w.data,
                        s,
                        w.n_out as usize,
                        w.n_in as usize,
                        w.group,
                        w.scale_bf16,
                    )?;
                }
                WType::Bf16 => {
                    dec.upload_plain(&name, w.data, w.n_out as usize, w.n_in as usize)?;
                }
                _ => {}
            }
        }
    }
    // ② 노름 nw — 8행/층 + 최종 1행. 라우터 행 = moe_scale·h^-0.5(가중 RMS).
    let inv_h = 1.0f32 / (h as f32).sqrt();
    let mut nw: Vec<f32> = Vec::with_capacity((8 * hp.n_layer + 1) * h);
    for il in 0..hp.n_layer {
        for suf in [
            "attn_norm.weight",
            "post_attn_norm.weight",
            "pre_ff_norm.weight",
            "post_ff_norm1.weight",
            "pre_ff_norm2.weight",
            "post_ff_norm2.weight",
            "post_ff_norm.weight",
        ] {
            let v = model
                .f32_vec(&format!("blk.{il}.{suf}"))
                .map_err(|e| e.to_string())?;
            if v.len() != h {
                return Err(format!("g4 노름 형상 위반 blk.{il}.{suf}: {}", v.len()));
            }
            nw.extend(v);
        }
        let sc = model
            .f32_vec(&format!("blk.{il}.moe_scale"))
            .map_err(|e| e.to_string())?;
        if sc.len() != h {
            return Err(format!("g4 moe_scale 형상 위반 L{il}: {}", sc.len()));
        }
        nw.extend(sc.iter().map(|v| v * inv_h));
    }
    nw.extend(
        model
            .f32_vec("output_norm.weight")
            .map_err(|e| e.to_string())?,
    );
    dec.set_norm_weights(&nw, 8 * hp.n_layer + 1)?;
    // ③ QK 노름(d_max 패딩) + 어텐션 층 테이블(KV 아레나 = 층별 최대 클래스).
    let d_max = hp.head_dim.iter().copied().max().unwrap_or(0);
    let mut qnw = vec![0f32; hp.n_layer * d_max];
    let mut knw = vec![0f32; hp.n_layer * d_max];
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        let q = model
            .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
            .map_err(|e| e.to_string())?;
        let k = model
            .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
            .map_err(|e| e.to_string())?;
        if q.len() != d || k.len() != d {
            return Err(format!(
                "g4 QK 노름 형상 L{il}: q={} k={} d={d}",
                q.len(),
                k.len()
            ));
        }
        qnw[il * d_max..il * d_max + d].copy_from_slice(&q);
        knw[il * d_max..il * d_max + d].copy_from_slice(&k);
    }
    let layers: Vec<llm170_backend_gpu::AttnLayer> = (0..hp.n_layer)
        .map(|il| llm170_backend_gpu::AttnLayer {
            kv_heads: hp.n_kv[il],
            d: hp.head_dim[il],
            sliding: !hp.full[il],
            k_eq_v: model.kv_eq_v(il),
            kq_scale: 1.0, // gemma4 = 1/√d 없음(노름이 스케일 흡수)
        })
        .collect();
    dec.set_attn(
        llm170_backend_gpu::AttnDims {
            n_attn: hp.n_layer,
            q_heads: hp.n_head,
            kv_heads: hp.n_kv[0],
            d: hp.head_dim[0],
            cap: ctx,
            interval: 1,
            sliding_window: hp.sw,
        },
        layers,
        &qnw,
        &knw,
    )?;
    let attn_ms = t0.elapsed().as_secs_f64() * 1e3;
    // ④ MoE — 전문가 테이블(무 shared·gelu GLU) + 상주/스트리밍 판정.
    let t1 = std::time::Instant::now();
    if hp.n_experts > 0 {
        let (group, scale_bf16) = model.expert_quant();
        dec.set_moe(
            hp.n_experts,
            hp.top_k,
            hp.moe_ffn,
            0,
            group,
            scale_bf16,
            true,
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
        let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
        let free = llm170_backend_gpu::cuda_mem_free()
            .map(|(f, _)| f)
            .unwrap_or(0);
        let resident = match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
            Some("0") => false,
            Some("1") => true,
            _ => free > expert_bytes + (1 << 30),
        };
        if resident {
            dec.upload_experts_resident(&tab)?;
        } else {
            dec.set_expert_table(tab);
        }
        eprintln!(
            "[g4-moe] 전문가 {:.2}GiB · 여유 {:.2}GiB → {} (g{group}{})",
            expert_bytes as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
            if resident {
                "전량 상주"
            } else {
                "호스트 스트리밍"
            },
            if scale_bf16 { "/bf16" } else { "" }
        );
    }
    let moe_ms = t1.elapsed().as_secs_f64() * 1e3;
    // ⑤ head — 원본 embed(tied) 행별 int8 W8A16(단일 head 계약).
    let t2 = std::time::Instant::now();
    {
        let hw = model.wchk("output.weight").map_err(|e| e.to_string())?;
        dec.upload_head(hw.data, hw.n_out as usize, hw.n_in as usize)?;
    }
    let head_ms = t2.elapsed().as_secs_f64() * 1e3;
    // ⑥ 호스트 설정(rope·scalar·pes — 층 산술 조립용) + rope 테이블 등록.
    let mut layer_scalar = Vec::with_capacity(hp.n_layer);
    let mut pes: Vec<f32> = Vec::with_capacity(hp.n_layer * hp.n_experts);
    let mut rp: Vec<f32> = Vec::with_capacity(hp.n_layer * 7);
    for il in 0..hp.n_layer {
        let ls = model
            .f32_vec(&format!("blk.{il}.layer_scalar"))
            .map_err(|e| e.to_string())?;
        layer_scalar.push(
            *ls.first()
                .ok_or_else(|| format!("layer_scalar 부재 L{il}"))?,
        );
        pes.extend(
            model
                .f32_vec(&format!("blk.{il}.moe_per_expert_scale"))
                .map_err(|e| e.to_string())?,
        );
        // 커널 rp 규약: {d, pairs, span, denom, base, vn_on, k_eq_v}.
        let d = hp.head_dim[il];
        rp.extend([
            d as f32,
            (hp.n_rot[il] / 2) as f32,
            (d / 2) as f32,
            d as f32,
            hp.rope_base[il],
            1.0, // v_norm(무가중) — gemma4 전 층
            if model.kv_eq_v(il) { 1.0 } else { 0.0 },
        ]);
    }
    dec.set_g4_rope(&rp)?;
    eprintln!("[g4] 업로드 — 선형·노름 {attn_ms:.0}ms · MoE {moe_ms:.0}ms · head {head_ms:.0}ms");
    Ok(G4HostCfg {
        softcap: hp.softcap,
        nw,
        layer_scalar,
        pes,
        qnw,
        knw,
        d_max,
    })
}

/// 모델 상주 업로드(1~4단계) — GpuEngine·w4a16-gpu 프로브 공용.
/// head는 호출부 소관(프로브는 --no-head/스테이징 분기가 있다).
pub(crate) fn upload_model(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    ctx: usize,
    kv_reserve: u64,
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
        upload_moe(dec, model, kv_reserve)?;
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
        sliding_window: 0,
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
    dec.set_attn(ad, Vec::new(), &qnw, &knw)?;
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
    // **[MTP-GRAPH·구간 실측 2026-10-12 — 최종]** t=1..8 캡처 인프라 완비
    // (핀 임베드·argmax 핀 회수·mtp_fwd_launch 분리·invalidate take·dx32/16
    // 사전 보장). 라운드 구간 실측(27B 프로즈, k=4): **save 0.00ms · 검증
    // t=5 37.1ms · 보정 t=1 19.8ms · MTP 4콜 10.4ms · 합 67.4ms**. 즉
    // (a) GDN 스냅샷은 d2d async라 사실상 무료(추정 15ms는 오판),
    // (b) 검증 t=5가 t=1(17.9)의 **2.07×**로 지배, (c) 그래프 후 MTP 콜은
    // 2.6ms/call이나 cuGraphLaunch 자체가 느린 기기라 총합 이득 ~0.
    // k=7(검증 t=8 풀블록 가설) 실측: 검증 61.7ms로 오히려 악화(블록 완충
    // 가설 기각), 4.82s/128토큰 = 26.6 t/s. k=4 = 3.4s(37.6 t/s) — 평범
    // (~50) 대비 여전히 순손실. **정직한 결론: 27B는 가중치 바운드라
    // 검증(t=k+1)+보정(t=1) 두 체인 고정비가 스펙 수익을 잠식 — 수익화는
    // ①검증 경로 marlin급(가중치 스트리밍 절반) ②검증+보정 체인 통합
    // ③MTP-2 단일 그래프 디바이스 체인(호스트 왕복 제거)이 선행돼야 한다.
    // **[보정 융합 채택 2026-10-12]** 스펙 라운드의 보정 토큰 = 검증 행 acc
    // argmax(toks[acc]) 재사용 — decode_greedy 중복 제거. n-gram 스펙 서빙
    // **2.6s→2.12s/128토큰(49→60 t/s, +22%)** · 라운드 53.2→32.0ms ·
    // 스펙 골든 불변(비트 계약 유지). MTP 경로만 디코드 유지(보정 hidden
    // 상태 전진이 초안 체인 입력 — 융합 시 0% 실측). MTP-on(k=4)은 3.07s
    // (41.7 t/s)로 여전히 평범 미달 — 파킹 유지(MTP_ENABLED=false).
    // **[MTP-CHAIN §3.3 재활성 2026-10-12]** 파킹 사유의 하나였던 재귀
    // 4콜 호스트 왕복(≈10.4ms/라운드)을 장치 체인(mtp_chain_drafts —
    // 그래프 리플레이 k-1회, 중간 왕복 0)으로 치환해 재활성. 초안 소스
    // 정책은 sched 계층(LLM170_SPEC_DRAFT_SOURCE="ngram" 강제 시 n-gram
    // 전용, 기본 MTP 가용 시 MTP — 35B MoE MTP는 여기서 early-return
    // false라 자동 n-gram). 순차 재귀는 체인 실패(계약·캡처) 폴백으로
    // 유지. 순손실 재발 시 이 자리에서 재파킹(실측 수치 갱신).
    const MTP_ENABLED: bool = true;
    if !MTP_ENABLED || model.w_raw("mtp.fc.weight").is_none() {
        return Ok(false);
    }
    // [D1 검수 수정 2026-10-13] dense MTP 마커 필수 — 종전
    // `experts.gate_up_proj` 검사는 bind가 None(비 .weight 이름)이라 w_raw가
    // 볼 수 없어 **무력**했고(35B MoE MTP가 dense 경로로 새어
    // `mtp.layers.0.mlp.gate_proj.weight 부재`로 로드 실패), 이 자리가
    // MoE MTP의 스킵 지점이다(자동 n-gram 폴백). MoE MTP 지원은 차기.
    if model.w_raw("mtp.layers.0.mlp.gate_proj.weight").is_none() {
        eprintln!("# mtp 스킵: dense MTP 아님(MoE MTP는 차기) — n-gram 폴백");
        return Ok(false);
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

/// [gemma4 P3 2026-10-14] t행 청크(≤8) — 프리필·검증 스텝.
/// 하이브리드: 노름·gelu·잔차·layer_scalar는 호스트(f32 = CPU 오라클 계약),
/// GEMM(t행)·어텐션(g4 호스트 경로 — 토큰별)·MoE(t행 디바이스)·head(t행 TR)는
/// GPU(전부 기존 검증 커널). KV = t=1 경로와 동일 슬롯·연속위치.
/// 반환 = 행별 로짓(t개, softcap 적용).
pub(crate) fn g4_chunk(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &G4HostCfg,
    slots: &[usize],
    pos: &[u32],
    rows: &[f32],
    t: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let hp = &model.hp;
    let h = hp.n_embd;
    if t == 0 || t > 8 || rows.len() != t * h || slots.len() != t || pos.len() != t {
        return Err(format!(
            "g4_chunk: t={t} rows={} slots={} pos={} (1..=8 · t*h 계약)",
            rows.len(),
            slots.len(),
            pos.len()
        ));
    }
    let mut x: Vec<f32> = rows.to_vec();
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        // ① input_layernorm → t행 q/k/(v) GEMM.
        let xn = g4_norm_rows(&x, cfg.nw_row(il, 0, h), hp.eps, t);
        // ② q/k/v GEMM + 어텐션 + o_proj 전부 **디바이스**(중간 d2h 0 —
        // 토큰별 호스트 경로의 왕복을 청크당 3런치로). KV 누적 = t=1과 동일.
        let sw = if hp.full[il] { 0 } else { hp.sw };
        let qd = hp.n_head * d;
        let dxn = dec.g4_dev_in(0, &xn)?;
        let q = dec.g4_dev_gemm(&format!("blk.{il}.attn_q.weight"), dxn, t, h, 1)?;
        let k = dec.g4_dev_gemm(&format!("blk.{il}.attn_k.weight"), dxn, t, h, 2)?;
        let v = if model.kv_eq_v(il) {
            k // v_norm(k 원시) — prep_g4 vn_on 계약
        } else {
            dec.g4_dev_gemm(&format!("blk.{il}.attn_v.weight"), dxn, t, h, 3)?
        };
        let outv = dec.attn_g4_dev(
            slots,
            pos,
            il,
            q,
            k,
            v,
            hp.n_head,
            hp.n_kv[il],
            d,
            sw,
            1.0,
            2,
            6,
            7,
            8,
        )?;
        // ③ o_proj(t행 — 디바이스) → d2h → post_attn_norm + 잔차.
        let o = dec.g4_dev_gemm(&format!("blk.{il}.attn_output.weight"), outv, t, qd, 9)?;
        let _ = o;
        let aop = dec.g4_dev_out(9, t * h)?;
        let hh = g4_norm_rows(&aop, cfg.nw_row(il, 1, h), hp.eps, t);
        for i in 0..t * h {
            x[i] += hh[i];
        }
        // ④ dense MLP — 디바이스 체인(pre_ff → gate/up GEMM → ew_gelu 커널 →
        // down GEMM) + d2h 1회. gelu = t=1 경로와 동일 커널 심볼(수치 정합).
        let nd = g4_norm_rows(&x, cfg.nw_row(il, 2, h), hp.eps, t);
        let dnd = dec.g4_dev_in(0, &nd)?;
        let g = dec.g4_dev_gemm(&format!("blk.{il}.ffn_gate.weight"), dnd, t, h, 1)?;
        let u = dec.g4_dev_gemm(&format!("blk.{il}.ffn_up.weight"), dnd, t, h, 2)?;
        let act = dec.g4_dev_gelu(g, u, 3, t * hp.n_ff)?;
        let _ = act;
        let dm_ = dec.g4_dev_gemm(&format!("blk.{il}.ffn_down.weight"), act, t, hp.n_ff, 4)?;
        let _ = dm_;
        let m = dec.g4_dev_out(4, t * h)?;
        let h1 = g4_norm_rows(&m, cfg.nw_row(il, 3, h), hp.eps, t);
        // ⑤ MoE — 라우터 입력(hn)·전문가 입력(nm) 분리 · pes 디바이스.
        let hn = g4_norm_rows(&x, cfg.nw_row(il, 7, h), hp.eps, t);
        let nm = g4_norm_rows(&x, cfg.nw_row(il, 4, h), hp.eps, t);
        let pes_dev = dec.g4_pes_dev(il);
        if pes_dev == 0 {
            return Err(format!("g4_chunk: L{il} pes 미등록(set_g4_pes 선행)"));
        }
        let e_out = dec.g4_moe_host_t(il, &hn, &nm, cfg.pes_row(il, hp.n_experts), t, pes_dev)?;
        let h2 = g4_norm_rows(&e_out, cfg.nw_row(il, 5, h), hp.eps, t);
        // ⑥ 병합 → post_ff_norm → 잔차 → layer_scalar.
        let mut merged = h1;
        for i in 0..t * h {
            merged[i] += h2[i];
        }
        let hf = g4_norm_rows(&merged, cfg.nw_row(il, 6, h), hp.eps, t);
        let ls = cfg.layer_scalar[il];
        for i in 0..t * h {
            x[i] = (x[i] + hf[i]) * ls;
        }
    }
    // ⑦ 최종 노름 + t행 head(TR — 가중 1판독) + softcap.
    let xf = g4_norm_rows(&x, cfg.out_norm_row(hp.n_layer, h), hp.eps, t);
    let flat = dec.g4_head_host_t(&xf, t)?;
    let n_vocab = hp.vocab;
    let mut out = Vec::with_capacity(t);
    for kk in 0..t {
        let mut lg = flat[kk * n_vocab..(kk + 1) * n_vocab].to_vec();
        if cfg.softcap > 0.0 {
            for l in lg.iter_mut() {
                *l = cfg.softcap * (*l / cfg.softcap).tanh();
            }
        }
        out.push(lg);
    }
    Ok(out)
}

/// [t][h] 행렬 행별 가중 RMS — CPU 계약(ops::rms_norm)과 동일 함수.
fn g4_norm_rows(x: &[f32], w: &[f32], eps: f32, t: usize) -> Vec<f32> {
    let h = w.len();
    let mut out = Vec::with_capacity(t * h);
    for r in 0..t {
        out.extend_from_slice(&llm170_core::ops::rms_norm(&x[r * h..(r + 1) * h], w, eps));
    }
    out
}

/// [gemma4 P2-3 2026-10-14] gemma4 v0 체인 1토큰 스텝(프로브·GemmaEngine 공용).
/// 층 산술(노름·잔차·layer_scalar)은 CPU 코어 계약(`llm170_core::ops`)으로
/// 조립하고, GPU가 담당하는 것은 GEMV(그룹 인식)·g4 어텐션·gelu·MoE(라우터
/// topk·pes·전문가)·head다. 계약 출처 = `crates/core/src/gemma4/`(CPU 오라클).
pub(crate) fn g4_step(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &crate::gpu_engine::G4HostCfg,
    slot: usize,
    tok: u32,
    pos: u32,
) -> Result<Vec<f32>, String> {
    let hp = &model.hp;
    let h = hp.n_embd;
    let mut x = model.embed_row(tok).map_err(|e| e.to_string())?;
    // 첫 풀층(층별 어텐션 도입부) 중간값 지문 — NaN/발산 위치 특정용.
    let stat = |v: &[f32]| -> String {
        let (mut mx, mut nan) = (f32::NEG_INFINITY, false);
        for &t in v {
            if t.is_nan() {
                nan = true;
            }
            mx = mx.max(t);
        }
        format!("max={mx:.4} nan={nan} len={}", v.len())
    };
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        let dbg = dec.debug_layers && hp.full[il] && il <= 5;
        // ① 어텐션 — input_layernorm → q/k/(v) → prep_g4+f3s_g4 → post_attn+잔차.
        let xn = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 0, h), hp.eps);
        let q = dec.gemv_f32_host(&format!("blk.{il}.attn_q.weight"), &xn)?;
        let k = dec.gemv_f32_host(&format!("blk.{il}.attn_k.weight"), &xn)?;
        let v = if model.kv_eq_v(il) {
            k.clone() // v_norm(k 원시) — prep_g4의 vn_on이 적용(HF 계약)
        } else {
            dec.gemv_f32_host(&format!("blk.{il}.attn_v.weight"), &xn)?
        };
        if dbg {
            eprintln!("  L{il} xn {} / q {} / k {}", stat(&xn), stat(&q), stat(&k));
        }
        let (qnw, knw) = cfg.qk_rows(il, d);
        let sw = if hp.full[il] { 0 } else { hp.sw };
        let attn_out = dec.attn_g4_host(
            slot,
            &q,
            &k,
            &v,
            qnw,
            knw,
            il,
            1,
            hp.n_head,
            hp.n_kv[il],
            d,
            sw,
            1.0,
            2,
            pos,
        )?;
        if dbg {
            eprintln!(
                "  L{il} attn_out {} / qnw {} / knw {}",
                stat(&attn_out),
                stat(qnw),
                stat(knw)
            );
        }
        // o_proj — 헤드 결합(16×256=4096) → hidden(2816).
        let ao = dec.gemv_f32_host(&format!("blk.{il}.attn_output.weight"), &attn_out)?;
        let hh = llm170_core::ops::rms_norm(&ao, cfg.nw_row(il, 1, h), hp.eps);
        if dbg {
            eprintln!(
                "  L{il} ao {} / hh {} / wo {}",
                stat(&ao),
                stat(&hh),
                stat(cfg.nw_row(il, 1, h))
            );
        }
        for i in 0..h {
            x[i] += hh[i];
        }
        // ② dense MLP — pre_ff → gate/up → gelu_tanh → down → post_ff1.
        let nd = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 2, h), hp.eps);
        let g = dec.gemv_f32_host(&format!("blk.{il}.ffn_gate.weight"), &nd)?;
        let u = dec.gemv_f32_host(&format!("blk.{il}.ffn_up.weight"), &nd)?;
        let act = dec.ew_gelu_host(&g, &u)?;
        let m = dec.gemv_f32_host(&format!("blk.{il}.ffn_down.weight"), &act)?;
        let h1 = llm170_core::ops::rms_norm(&m, cfg.nw_row(il, 3, h), hp.eps);
        // ③ MoE — 라우터 입력(hn)과 전문가 입력(nm)이 **별개** 노름(gemma4 계약).
        let hn = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 7, h), hp.eps);
        let nm = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 4, h), hp.eps);
        let e_out = dec.g4_moe_host(il, &hn, &nm, cfg.pes_row(il, hp.n_experts))?;
        let h2 = llm170_core::ops::rms_norm(&e_out, cfg.nw_row(il, 5, h), hp.eps);
        // ④ 병합 — post_ff_norm(h1+h2) → 잔차 → ×layer_scalar.
        let mut merged = h1;
        for i in 0..h {
            merged[i] += h2[i];
        }
        let hf = llm170_core::ops::rms_norm(&merged, cfg.nw_row(il, 6, h), hp.eps);
        let ls = cfg.layer_scalar[il];
        for i in 0..h {
            x[i] = (x[i] + hf[i]) * ls;
        }
        // 층별 지문 — CPU 오라클(debug_layers)과 같은 형식(최초 발산 층 추적).
        if dec.debug_layers {
            let mut mx = f32::NEG_INFINITY;
            let mut nan = false;
            for &v in x.iter() {
                if v.is_nan() {
                    nan = true;
                }
                mx = mx.max(v);
            }
            eprintln!("gemma4[gpu] layer {il:>2} max|x|={mx:.4} nan={nan}");
        }
    }
    // ⑤ 최종 노름 + GPU head + softcap(호스트 — argmax 단조 불변).
    let xf = llm170_core::ops::rms_norm(&x, cfg.out_norm_row(hp.n_layer, h), hp.eps);
    if dec.debug_layers {
        let mut mx = f32::NEG_INFINITY;
        let mut nan = false;
        for &v in xf.iter() {
            if v.is_nan() {
                nan = true;
            }
            mx = mx.max(v);
        }
        eprintln!("gemma4[gpu] final max|x|={mx:.4} nan={nan}");
    }
    let mut lg = dec.head_logits_host(&xf)?;
    if cfg.softcap > 0.0 {
        for l in lg.iter_mut() {
            *l = cfg.softcap * (*l / cfg.softcap).tanh();
        }
    }
    Ok(lg)
}

// ──────────────────── [gemma4 P3 2026-10-14] 서빙 아치 디스패치 ────────────────────
// qwen(QwenEngine)·gemma4(GemmaEngine)를 단일 타입으로 — P2-3 v0 체인 그대로
// (호스트 스테이징·f32 활성 클래스·g4 커널군·g32 GEMV·int8 KV/head).

/// gemma4 서빙 엔진 — v0 체인(P2-3). spec·kstep·배치 체인은 후속 트랙
/// (순수 어텐션 롤백 = KV 절단 — v0는 슬롯 직렬).
pub struct GemmaEngine {
    model: llm170_core::gemma4::Model,
    cfg: G4HostCfg,
    dec: W4a16Dec,
    #[allow(dead_code)]
    n_slots: usize,
}

impl GemmaEngine {
    /// 모델 로드 + 상주 업로드(upload_model_g4 — 선형·노름 8행/층·어텐션
    /// 테이블·rope·MoE(g32/전문가)·head(원문 embed 행별 int8)).
    pub fn load(dir: &Path, n_slots: usize, ctx: usize) -> Result<Self, String> {
        let model = llm170_core::gemma4::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        let cfg = upload_model_g4(&mut dec, &model, ctx)?;
        // 층별 pes(μ_e) 디바이스 상주(t행 MoE 경로용 — 1회).
        for il in 0..model.hp.n_layer {
            dec.set_g4_pes(il, cfg.pes_row(il, model.hp.n_experts))?;
        }
        Ok(GemmaEngine {
            model,
            cfg,
            dec,
            n_slots,
        })
    }

    /// 1토큰 스텝 — **t행 청크 t=1**(단일 경로: 배치·프리필·spec과 동일 커널·
    /// 클래스 — t행 GEMV는 t=1 GEMV와 같은 산술 순서). 호스트 체인의 층당
    /// ~14왕복(h2d/launch/d2h/sync)이 청크에선 ~8로 줄어든다.
    fn step(&mut self, seq: usize, tok: u32) -> Result<Vec<f32>, String> {
        let pos = self.dec.slot_pos[seq];
        self.dec.kv_grow(seq, pos as usize + 1)?;
        let rows = self.model.embed_row(tok).map_err(|e| e.to_string())?;
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            &[seq],
            &[pos],
            &rows,
            1,
        )?;
        self.dec.slot_pos[seq] = pos + 1;
        Ok(lgs.into_iter().next().unwrap_or_default())
    }

    /// 프리필 — t행 청크(≤8) 경로(가중 상각). 반환 = 마지막 로짓.
    /// 스트리밍(비상주 전문가)은 t행 미지원 → t=1 스텝 폴백.
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = Vec::new();
        let t_chunk = if self.dec.moe_experts_resident() {
            8
        } else {
            1
        };
        for chunk in tokens.chunks(t_chunk) {
            let pos = self.dec.slot_pos[seq];
            self.dec.kv_grow(seq, pos as usize + chunk.len())?;
            if chunk.len() == 1 {
                last = self.step(seq, chunk[0])?;
            } else {
                let mut rows: Vec<f32> = Vec::with_capacity(chunk.len() * self.model.hp.n_embd);
                for &tk in chunk {
                    rows.extend_from_slice(&self.model.embed_row(tk).map_err(|e| e.to_string())?);
                }
                let csv: Vec<usize> = vec![seq; chunk.len()];
                let cpos: Vec<u32> = (0..chunk.len()).map(|i| pos + i as u32).collect();
                let lgs = g4_chunk(
                    &mut self.dec,
                    &self.model,
                    &self.cfg,
                    &csv,
                    &cpos,
                    &rows,
                    chunk.len(),
                )?;
                self.dec.slot_pos[seq] = pos + chunk.len() as u32;
                last = lgs.into_iter().last().unwrap_or_default();
            }
        }
        Ok(last)
    }

    /// 슬롯 배치 1토큰 — **이종 슬롯·위치 t행 청크**(가중 상각) + 행별 로짓.
    /// 스트리밍 전문가·t>8은 슬롯별 t=1 폴백.
    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        if seq_ids.len() != tokens.len() {
            return Err(format!(
                "decode: seqs={} toks={} 불일치",
                seq_ids.len(),
                tokens.len()
            ));
        }
        if seq_ids.is_empty() {
            return Err("decode: 빈 배치".into());
        }
        let t = seq_ids.len();
        if t == 1 {
            return Ok(vec![self.step(seq_ids[0], tokens[0])?]);
        }
        if t > 8 || !self.dec.moe_experts_resident() {
            let mut out = Vec::with_capacity(t);
            for (&s, &tok) in seq_ids.iter().zip(tokens.iter()) {
                out.push(self.step(s, tok)?);
            }
            return Ok(out);
        }
        // 배치 청크 — 행 = 슬롯(임베딩·pos·KV 확보) → 디바이스 t행 체인.
        let mut rows = Vec::with_capacity(t * self.model.hp.n_embd);
        let mut pos: Vec<u32> = Vec::with_capacity(t);
        for (&s, &tok) in seq_ids.iter().zip(tokens.iter()) {
            rows.extend_from_slice(&self.model.embed_row(tok).map_err(|e| e.to_string())?);
            let p = self.dec.slot_pos[s];
            self.dec.kv_grow(s, p as usize + 1)?;
            pos.push(p);
        }
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            seq_ids,
            &pos,
            &rows,
            t,
        )?;
        for (i, &s) in seq_ids.iter().enumerate() {
            self.dec.slot_pos[s] = pos[i] + 1;
        }
        Ok(lgs)
    }

    /// 슬롯 배치 greedy — step + argmax.
    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        Ok(self
            .decode(seq_ids, tokens)?
            .iter()
            .map(|l| llm170_core::matmul::greedy_from(l))
            .collect())
    }

    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        let lg = self.step(seq, token)?;
        Ok(llm170_core::matmul::greedy_from(&lg))
    }

    /// 혼합 배치 — (그리디 토큰, 평탄 로짓). v0는 슬롯 직렬(그리디 행=토큰,
    /// 샘플 행=로짓 소비 — qwen 계약과 동일 의미).
    pub fn decode_np_mixed(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let rows = self.decode(seq_ids, tokens)?;
        let toks = rows
            .iter()
            .map(|l| llm170_core::matmul::greedy_from(l))
            .collect();
        let mut flat = Vec::with_capacity(rows.len() * self.model.hp.vocab);
        for l in &rows {
            flat.extend_from_slice(l);
        }
        Ok((toks, flat))
    }

    /// 배치 로짓(샘플링 슬롯) — 평탄 [t][head_n].
    pub fn decode_np_logits(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        let rows = self.decode(seq_ids, tokens)?;
        let mut flat = Vec::with_capacity(rows.len() * self.model.hp.vocab);
        for l in &rows {
            flat.extend_from_slice(l);
        }
        Ok(flat)
    }

    /// 슬롯 리셋 — KV 블록 회수 + pos 0(새 요청).
    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        let _ = self.dec.kv_release(seq);
        if seq < self.dec.slot_pos.len() {
            self.dec.slot_pos[seq] = 0;
        }
        Ok(())
    }

    /// spec 활성 — 순수 어텐션 롤백 = **pos 되감기**(KV는 다음 쓰기가 덮고,
    /// 판독은 pos까지만 — 상태 스냅샷 불요). 초안 = n-gram(스케줄러 측).
    pub fn enable_spec(&mut self) -> Result<(), String> {
        Ok(())
    }
    pub fn spec_on(&self) -> bool {
        true
    }
    /// 검증 라운드 — [last]+drafts 행(연속 위치)을 t행 청크로 통과 → 행별 그리디.
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        let lgs = self.spec_run(slot, rows, t)?;
        Ok(lgs.iter().map(|l| llm170_core::gemma4::greedy(l)).collect())
    }
    /// 검증 행별 로짓(샘플드 거절샘플링 — 평탄 [t][vocab]).
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let lgs = self.spec_run(slot, rows, t)?;
        let mut flat = Vec::with_capacity(t * self.model.hp.vocab);
        for l in &lgs {
            flat.extend_from_slice(l);
        }
        Ok(flat)
    }
    /// 검증 공용 — t행 청크 + KV 확보(pos 전진은 청크가 수행).
    fn spec_run(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<Vec<f32>>, String> {
        if !(2..=8).contains(&t) {
            return Err(format!("g4 spec: t={t} (2..=8 계약)"));
        }
        let pos0 = self.dec.slot_pos[slot];
        self.dec.kv_grow(slot, pos0 as usize + t)?;
        let csv: Vec<usize> = vec![slot; t];
        let cpos: Vec<u32> = (0..t).map(|i| pos0 + i as u32).collect();
        let lgs = g4_chunk(&mut self.dec, &self.model, &self.cfg, &csv, &cpos, rows, t)?;
        self.dec.slot_pos[slot] = pos0 + t as u32;
        Ok(lgs)
    }
    pub fn decode_greedy_k(
        &mut self,
        _seq: usize,
        _token: u32,
        _k: usize,
        _eos: u32,
    ) -> Result<Vec<u32>, String> {
        Err("gemma4: kstep 미지원(v0)".into())
    }
    pub fn mtp_drafts_on(&self) -> bool {
        false
    }

    /// 접두 재사용 체크포인트 — v0 미구현(순수 어텐션의 KV 절단 재사용은
    /// 후속 트랙). 반환 = 0(재사용 없음 — 전체 재프리필).
    pub fn checkpoint_pos(&self, _slot: usize, _upto: u32) -> u32 {
        0
    }
    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        self.dec.kv_grow(slot, n_tokens)
    }
    pub fn kv_total_blocks(&self) -> usize {
        self.dec.kv_total_blocks()
    }
    pub fn kv_free_blocks(&self) -> usize {
        self.dec.kv_free_blocks()
    }
    pub fn kv_blocks_needed(&self, n_tokens: usize) -> usize {
        self.dec.kv_blocks_needed(n_tokens)
    }
    pub fn kv_release_slot(&mut self, slot: usize) -> usize {
        self.dec.kv_release(slot)
    }
    /// 표시용 조각(gemma4 모델 조각표 — ▁→공백).
    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(|s| s.replace('▁', " "))
            .unwrap_or_default()
    }
    pub fn prof_report(&mut self, _label: &str) -> Result<String, String> {
        Err("gemma4: prof_report 미지원(v0)".into())
    }
    // MTP·GDN·spec 계열 — gemma4 v0 미지원(호출부는 spec/mtp 비활성 게이트로
    // 진입 자체가 없다. 시그니처 정합용 명시 거부).
    pub fn restore_gdn(&mut self, _slot: usize, _upto: u32) -> Result<u32, String> {
        Err("gemma4: GDN 없음(순수 어텐션)".into())
    }
    pub fn spec_rollback(&mut self, _slot: usize, _keep: usize) -> Result<(), String> {
        // 순수 어텐션 — 복원할 상태 없음. 수용 접두 이후 KV 행은 다음 쓰기가
        // 덮는다(판독은 pos까지 — 잔존 행 무해). pos는 spec_rewind_pos 소관.
        Ok(())
    }
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        let p = self.dec.slot_pos[slot];
        self.dec.slot_pos[slot] = p.saturating_sub(n);
        Ok(())
    }
    pub fn mtp_first(&mut self, _slot: usize, _tok: u32) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_single(&mut self, _slot: usize, _tok: u32) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_pre_round(&mut self, _acc: usize) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_post_round(
        &mut self,
        _slot: usize,
        _accepted: &[u32],
        _corrected: u32,
    ) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_slot_kv_reset(&mut self) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_slot_fresh(&mut self, _slot: usize) {}
    pub fn mtp_slot_reused(&mut self, _slot: usize) {}
    pub fn mtp_take_draft(&mut self, _slot: usize) -> Option<Vec<u32>> {
        None
    }

    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        self.dec.mem_stats()
    }
    pub fn active_weight_bytes(&self) -> u64 {
        self.dec.active_weight_bytes()
    }
    pub fn moe_mode(&self) -> &'static str {
        self.dec.moe_mode()
    }
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.dec.copy_stats()
    }
    /// 정지 토큰 — gemma4 EOS는 [1,106,50], 대표 1(<eos>). 전체 목록은
    /// 서빙 스톱 레지스트리(oai chat_spec)·스모크가 소유.
    pub fn eos(&self) -> u32 {
        1
    }
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, String> {
        self.model.embed_row(tok).map_err(|e| e.to_string())
    }
    pub fn vocab(&self) -> usize {
        self.model.hp.vocab
    }
    pub fn n_embd(&self) -> usize {
        self.model.hp.n_embd
    }
}

/// 서빙 엔진 — 아치 디스패치 단일 타입(호출부는 이 타입만 쓴다).
pub enum GpuEngine {
    Qwen(Box<QwenEngine>),
    Gemma(Box<GemmaEngine>),
}

/// 위임 매크로 — &mut self 메서드.
macro_rules! engine_delegate_mut {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
        impl GpuEngine {
            $(
                pub fn $name(&mut self $(, $a: $t)*) -> $r {
                    match self {
                        GpuEngine::Qwen(e) => e.$name($($a),*),
                        GpuEngine::Gemma(e) => e.$name($($a),*),
                    }
                }
            )*
        }
    };
}

/// 위임 매크로 — &self 메서드.
macro_rules! engine_delegate_ref {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
        impl GpuEngine {
            $(
                #[allow(dead_code)]
                pub fn $name(&self $(, $a: $t)*) -> $r {
                    match self {
                        GpuEngine::Qwen(e) => e.$name($($a),*),
                        GpuEngine::Gemma(e) => e.$name($($a),*),
                    }
                }
            )*
        }
    };
}

engine_delegate_mut! {
    prefill(seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String>;
    kv_grow(slot: usize, n_tokens: usize) -> Result<usize, String>;
    kv_release_slot(slot: usize) -> usize;
    prof_report(label: &str) -> Result<String, String>;
    restore_gdn(slot: usize, upto: u32) -> Result<u32, String>;
    spec_rollback(slot: usize, keep: usize) -> Result<(), String>;
    spec_rewind_pos(slot: usize, n: u32) -> Result<(), String>;
    spec_verify(slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String>;
    mtp_first(slot: usize, tok: u32) -> Result<(), String>;
    mtp_single(slot: usize, tok: u32) -> Result<(), String>;
    mtp_pre_round(acc: usize) -> Result<(), String>;
    mtp_post_round(slot: usize, accepted: &[u32], corrected: u32) -> Result<(), String>;
    mtp_slot_kv_reset() -> Result<(), String>;
    mtp_slot_fresh(slot: usize) -> ();
    mtp_slot_reused(slot: usize) -> ();
    mtp_take_draft(slot: usize) -> Option<Vec<u32>>;
    decode(seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String>;
    decode_np_greedy(seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<u32>, String>;
    decode_greedy(seq: usize, token: u32) -> Result<u32, String>;
    decode_greedy_k(seq: usize, token: u32, k: usize, eos: u32) -> Result<Vec<u32>, String>;
    decode_np_mixed(seq_ids: &[usize], tokens: &[u32]) -> Result<(Vec<u32>, Vec<f32>), String>;
    decode_np_logits(seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<f32>, String>;
    reset_seq(seq: usize) -> Result<(), String>;
    enable_spec() -> Result<(), String>;
    spec_verify_logits(slot: usize, rows: &[f32], t: usize) -> Result<Vec<f32>, String>;
}

engine_delegate_ref! {
    spec_on() -> bool;
    checkpoint_pos(slot: usize, upto: u32) -> u32;
    kv_total_blocks() -> usize;
    kv_free_blocks() -> usize;
    kv_blocks_needed(n_tokens: usize) -> usize;
    piece(tok: u32) -> String;
    mtp_drafts_on() -> bool;
    mem_stats() -> (u64, u64, u64, u64);
    active_weight_bytes() -> u64;
    moe_mode() -> &'static str;
    copy_stats() -> [(u64, u64, u64); 3];
    eos() -> u32;
    embed_row(tok: u32) -> Result<Vec<f32>, String>;
    vocab() -> usize;
    n_embd() -> usize;
}

impl GpuEngine {
    /// 아치 스니핑 후 해당 엔진 로드 — 서빙 진입 단일 경로.
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
    ) -> Result<Self, String> {
        let arch = llm170_core::qwen35::bind::dir_arch(dir).unwrap_or_default();
        if llm170_core::gemma4::bind::arch_ok(&arch) {
            if kv_budget.is_some() || kv_reserve.is_some() {
                eprintln!("# gemma4: kv_budget/kv_reserve는 v0 미지원 — 기본 전량 할당");
            }
            return Ok(GpuEngine::Gemma(Box::new(GemmaEngine::load(
                dir, n_slots, ctx,
            )?)));
        }
        Ok(GpuEngine::Qwen(Box::new(QwenEngine::load(
            dir, n_slots, ctx, kv_budget, kv_reserve,
        )?)))
    }
}
