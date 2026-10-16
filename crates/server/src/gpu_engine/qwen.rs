//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use super::QwenEngine;
use super::mtp::upload_mtp;
use super::prefill_tmax;
use super::upload::upload_model;
use llm170_backend_gpu::W4a16Dec;
use std::path::Path;

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
        // [W4 2026-10-16] head/embed — O_DIRECT 직결 스트림(mmap 우회).
        // head는 행-블록 양자화(호스트 RAM 상수 — 2.4GB 물질화 회피).
        let mut ld = super::upload::DirectLoad::new(model.store());
        let mut head_gpu = false;
        if let Some(w) = model.w("output.weight")
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            let (tn, _) = model
                .w_raw_names("output.weight")
                .ok_or_else(|| "output.weight: 원시 텐서 이름 부재".to_string())?;
            let len = ld.tensor_len(&tn)? as usize;
            let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
            dec.upload_head_stream(w.n_out as usize, w.n_in as usize, len, &mut f)?;
            head_gpu = true;
        }
        // [S3-1 2026-10-11] token_embd 상주 — 프리필 행 조립을 GPU gather로.
        if let Some(w) = model.w("token_embd.weight")
            && matches!(
                w.ty,
                llm170_core::wtype::WType::F16 | llm170_core::wtype::WType::Bf16
            )
        {
            let (tn, _) = model
                .w_raw_names("token_embd.weight")
                .ok_or_else(|| "token_embd.weight: 원시 텐서 이름 부재".to_string())?;
            let len = ld.tensor_len(&tn)? as usize;
            let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
            dec.upload_embed_stream(
                w.n_out as usize,
                w.n_in as usize,
                w.ty == llm170_core::wtype::WType::Bf16,
                len,
                &mut f,
            )?;
        }
        ld.log_summary();
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

    /// [§2.M ①] 스트리밍 전문가 LRU 캐시 계측 위임.
    pub fn mcache_stats(&self) -> (u64, u64, usize) {
        self.dec.mcache_stats()
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
    pub(super) fn softcap_row(&self, lg: &mut [f32]) {
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

    /// [W1 2026-10-16] 실블록 유래 단일 요청 상한(토큰).
    pub fn kv_cap_tokens(&self) -> usize {
        self.dec.kv_cap_tokens()
    }

    /// [W2 2026-10-16] 슬롯 점유 블록(aging 예측 입력).
    pub fn kv_used_blocks(&self, slot: usize) -> usize {
        self.dec.kv_used_blocks(slot)
    }

    /// 정지 토큰 — Q35 아키텍처 상수.
    pub fn eos(&self) -> u32 {
        llm170_core::qwen35::EOS_EOT
    }
    pub fn vocab(&self) -> usize {
        self.model.hp.vocab
    }
    /// [A2→K3 2026-10-12] 접두 재사용 체크포인트 — upto 이하 최대 경계
    /// 조회(0=없음)/복원(512 경계 스택 4개).
    /// [접두 재사용 2026-10-14] 엔진이 실제로 소비한 위치 — 재사용 상한의 정본.
    pub fn seq_pos(&self, slot: usize) -> u32 {
        self.dec.slot_pos_of(slot)
    }
    /// 부분 접두 복원 — GDN 체크포인트(≤ upto)로 상태·pos 복원. 반환 = 복원 pos.
    pub fn restore_prefix(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        self.dec.restore_gdn(slot, upto)
    }
    /// [2.N Stage 2] 재사용 유효 — qwen은 KV 트림을 하지 않아 항상 유효
    /// (Stage-1 동작 보존 — 유효 판정은 restore_prefix/체크포인트가 수행).
    pub fn reuse_valid(&self, _slot: usize, _l: usize, _prompt_len: usize) -> bool {
        true
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
