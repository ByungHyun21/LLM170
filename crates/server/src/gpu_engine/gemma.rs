//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use super::g4::{g4_chunk, g4_step};
use super::upload_g4::{G4HostCfg, upload_model_g4};
use llm170_backend_gpu::W4a16Dec;
use std::path::Path;

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
    /// kv_reserve는 시그니처 통일용으로 받고 소비하지 않는다 — gemma4는 KV
    /// 실할당(set_attn)이 MoE 판정보다 먼저라 여유가 이미 KV를 제외한 값이다
    /// (차감 추가 = 이중차감. qwen은 반대 순서라 reserve를 쓴다).
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
    ) -> Result<Self, String> {
        let _ = kv_reserve;
        let model = llm170_core::gemma4::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        // [디버그 배선 2026-10-15] qwen과 동형 — 층별 지문(dump 키) + 그래프 자동 OFF
        // (g4_graph_ok). 종전엔 GemmaEngine이 세팅하지 않아 서빙에선 무출력이었다.
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        // [K6-3 예산] 풀 예산(여유 VRAM 파생) — upload_model_g4(내부 set_attn) 전에 설정.
        dec.set_kv_budget(kv_budget);
        // [D2 검수] 적재 실패 시 부분 할당 정리 — qwen load와 동일 위생.
        match Self::fill(&mut dec, &model, ctx) {
            Ok(cfg) => Ok(GemmaEngine {
                model,
                cfg,
                dec,
                n_slots,
            }),
            Err(e) => {
                dec.free_all_device();
                Err(e)
            }
        }
    }

    /// 컨텍스트 생성 **후** 적재 본체(upload_model_g4 + 층별 pes).
    /// 실패는 호출부(load)가 `dec.free_all_device()`로 정리한다.
    fn fill(
        dec: &mut W4a16Dec,
        model: &llm170_core::gemma4::Model,
        ctx: usize,
    ) -> Result<G4HostCfg, String> {
        let cfg = upload_model_g4(dec, model, ctx)?;
        // 층별 pes(μ_e) 디바이스 상주(t행 MoE 경로용 — 1회). layer_scalar는
        // upload_model_g4가 등록(엔진·eval·프로브 공용 단일 계약).
        for il in 0..model.hp.n_layer {
            dec.set_g4_pes(il, cfg.pes_row(il, model.hp.n_experts))?;
        }
        Ok(cfg)
    }

    /// 1토큰 스텝 — **t행 청크 t=1**(단일 경로: 배치·프리필·spec과 동일 커널·
    /// 클래스 — t행 GEMV는 t=1 GEMV와 같은 산술 순서). 호스트 체인의 층당
    /// ~14왕복(h2d/launch/d2h/sync)이 청크에선 ~8로 줄어든다.
    /// **전문가 비상주(스트리밍)에선 v0 호스트 체인(g4_step)으로 폴백** — t행
    /// MoE(g4_moe_dev_t)가 스트리밍 미지원이라, 배치·프리필의 비상주 분기가
    /// 모두 수렴하는 이 스텝이 저하 사다리의 단일 통로다.
    fn step(&mut self, seq: usize, tok: u32) -> Result<Vec<f32>, String> {
        let pos = self.dec.slot_pos[seq];
        self.dec.kv_grow(seq, pos as usize + 1)?;
        if !self.dec.moe_experts_resident() {
            let lg = g4_step(&mut self.dec, &self.model, &self.cfg, seq, tok, pos)?;
            self.dec.slot_pos[seq] = pos + 1;
            // [2.N Stage 2] 스트리밍 경로도 확정 위치 — KV 트림은 MoE 상주와 무관.
            self.maybe_trim(seq)?;
            return Ok(lg);
        }
        let rows = self.model.embed_row(tok).map_err(|e| e.to_string())?;
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            &[seq],
            &[pos],
            &rows,
            1,
            false,
        )?;
        self.dec.slot_pos[seq] = pos + 1;
        // [2.N Stage 2] 확정 1토큰 — 플로어 전진 가능.
        self.maybe_trim(seq)?;
        Ok(lgs.into_iter().next().unwrap_or_default())
    }

    /// 프리필 — t행 청크 경로(가중 상각). 반환 = 마지막 로짓.
    /// 스트리밍(비상주 전문가)은 t행 미지원 → t=1 스텝 폴백.
    /// [2.N T16-reg 2026-10-15] **청크 16 채택** — T=16 인스턴스의 스필/레지스터
    /// 압박(gptq4 g32_t STACK 256B·moe bf16_t REG 48→94·head i8_t 65→90)이
    /// pp −13~14%의 원인이었다. 커널 행별 x 포인터 배열(xp[T])을 단일 베이스 +
    /// 컴파일타임 오프셋으로 제거한 뒤 동일 세션 교차 A/B(수정 fatbin, 2라운드·
    /// SM 2805MHz): **pp16 +13.5% · pp256 +16.2%**(대 chunk8) · 디코드 불변 —
    /// 방향 역전으로 청크 16이 정본.
    /// [A1 2026-10-15 nsys/ncu 직측 — 차기 t 확대 근거] pp 갭(vLLM 30~58×)의
    /// 지배 항 = **청크 t 상각**: 전문가 가중치 = 청크당 ~11.4GB 고정(union
    /// 포화·L2 부분 상쇄 — t 무관) → bytes/토큰 = 11.4GB/t: t16 712MB vs
    /// vLLM(배치≈1024) 13.7MB = 52×(잔차 포함 실측 pp1024 갭 57.9×). glu/down은
    /// DRAM 87%/48% 포화(커널 개선 여지 없음)·dense t행 GEMV는 DRAM 8~11%·
    /// SM 38~41% 지연(청크 예산 33%)·head는 청크당 고정 738MB. 차기 = t=64
    /// (dense는 mma 재개 조건 — M64 타일 포화).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = Vec::new();
        // [A2 2026-10-15] **청크 64 실험** — A1 귀속(전문가 bytes/토큰 = 11.4GB/t)
        // 검증: 전문가·topk·attn은 t 직지원, t행 GEMV 계열(선형·라우터)은 호스트
        // 행 분할(≤16 스위치 — 행별 비트동일). 상한 = G4_CHUNK_MAX(단일 출처).
        let t_chunk = if self.dec.moe_experts_resident() {
            llm170_backend_gpu::G4_CHUNK_MAX
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
                    true, // 프리필 — 마지막 행 로짓만
                )?;
                self.dec.slot_pos[seq] = pos + chunk.len() as u32;
                last = lgs.into_iter().last().unwrap_or_default();
            }
        }
        // [2.N Stage 2] 프리필 확정 — 플로어 전진 가능.
        self.maybe_trim(seq)?;
        Ok(last)
    }

    /// 슬롯 배치 1토큰 — **이종 슬롯·위치 t행 청크**(가중 상각) + 행별 로짓.
    /// 스트리밍 전문가·t>16은 슬롯별 t=1 폴백.
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
        if t > 16 || !self.dec.moe_experts_resident() {
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
            false,
        )?;
        for (i, &s) in seq_ids.iter().enumerate() {
            self.dec.slot_pos[s] = pos[i] + 1;
        }
        // [2.N Stage 2] 배치 확정 — 슬롯별 플로어 전진 가능.
        for &s in seq_ids {
            self.maybe_trim(s)?;
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
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            &csv,
            &cpos,
            rows,
            t,
            false,
        )?;
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

    /// [접두 재사용 2026-10-14] 엔진 소비 위치 — 순수 어텐션은 상태가 pos뿐.
    pub fn seq_pos(&self, slot: usize) -> u32 {
        self.dec.slot_pos.get(slot).copied().unwrap_or(0)
    }
    /// 부분 접두 복원 — 순수 어텐션: 상태 = KV + pos뿐이라 pos만 되돌린다
    /// (KV 잔존 행은 다음 쓰기가 덮고, 판독은 pos까지 — spec_rollback과 동일
    /// 계약). upto > pos는 불가(호출부가 콜드 경로로). 반환 = 복원 pos.
    /// [2.N Stage 2] 플로어 백스톱 — `floor > upto`면 해제 행이 접미사 재기록
    /// 아래에 있어 복원 불가(호출부는 콜드 경로로). prompt−sw 조건은 스케줄러
    /// 측 reuse_valid가 본다(여기에 prompt 길이가 없다).
    pub fn restore_prefix(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        if slot >= self.n_slots {
            return Err(format!(
                "gemma4: slot {slot} 범위 밖(문맥 {})",
                self.n_slots
            ));
        }
        let p = self.dec.slot_pos[slot];
        if upto == 0 || upto > p {
            return Err(format!("gemma4: restore_prefix {upto} > pos {p}"));
        }
        if self.dec.kv_floor_of(slot) > upto {
            return Err(format!(
                "gemma4: restore_prefix {upto} < floor {}",
                self.dec.kv_floor_of(slot)
            ));
        }
        self.dec.slot_pos[slot] = upto;
        Ok(upto)
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
    // MTP·spec 계열 — gemma4 v0 미지원(호출부는 spec/mtp 비활성 게이트로
    // 진입 자체가 없다. 시그니처 정합용 명시 거부).
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

    /// [2.N Stage 2] 접두 재사용 유효 판정 — 스케줄러가 restore_prefix 호출
    /// 전에 문의한다(전체·부분 공용). 조건(둘 다 필요):
    /// ① `floor ≤ l` — 접미사 프리필이 l부터 전부 재기록하므로 floor 미만
    /// 행은 다시 기록되지 않는다. l이 floor보다 낮으면 그 행들은 해제된
    /// 채로 남아 이후 판독을 오염시킨다 → 콜드(전량 재계산).
    /// ② `floor ≤ prompt_len − sw` — 디코드는 `[lim−sw, lim)`만 읽고 lim은
    /// prompt_len부터 커지므로, 이 조건이 floor 미만 행의 비마스크 판독을
    /// 원천 차단한다(FA2의 마스크 밖 물리 판독은 무조건 −1e30 대입이라 무해).
    /// 풀층은 트림 대상이 아니라 조건과 무관하다.
    pub fn reuse_valid(&self, slot: usize, l: usize, prompt_len: usize) -> bool {
        reuse_valid(
            self.dec.kv_floor_of(slot),
            l as u32,
            prompt_len as u32,
            self.model.hp.sw as u32,
        )
    }

    /// [2.N Stage 2] 확정 위치 기준 플로어 전진 — 커밋 경로(step·prefill·
    /// decode) 꼬리에서만 호출한다. spec 검증(spec_run)은 투기 위치라 트림
    /// 금지(되감기 시 플로어가 위치를 앞지르면 오염).
    fn maybe_trim(&mut self, slot: usize) -> Result<(), String> {
        let pos = self.dec.slot_pos.get(slot).copied().unwrap_or(0);
        let f = floor_for_pos(pos, self.model.hp.sw as u32);
        if f > self.dec.kv_floor_of(slot) {
            let n = self.dec.kv_release_below(slot, f)?;
            eprintln!("# g4-trim: slot{slot} floor {f}행(해제 {n}블록)");
        }
        Ok(())
    }
}

/// [2.N Stage 2] 위치→플로어(행) — `pos−sw−여유`를 512 경계로 내림.
/// 여유 512(1블록)는 spec 되감기(≤8)·청크 정렬을 흡수한다. pos가 짧으면 0.
fn floor_for_pos(pos: u32, sw: u32) -> u32 {
    const SLACK: u32 = 512;
    if pos > sw + SLACK {
        ((pos - sw - SLACK) / 512) * 512
    } else {
        0
    }
}

/// [2.N Stage 2] 재사용 유효 순수 판정(테스트 표면).
fn reuse_valid(floor: u32, l: u32, prompt_len: u32, sw: u32) -> bool {
    floor <= l && floor <= prompt_len.saturating_sub(sw)
}

#[cfg(test)]
mod tests {
    use super::{floor_for_pos, reuse_valid};

    /// 플로어 양자화 — 512 경계 내림·단조·짧으면 0.
    #[test]
    fn floor_quantized_monotone() {
        let sw = 1024;
        assert_eq!(floor_for_pos(0, sw), 0);
        assert_eq!(floor_for_pos(1536, sw), 0, "sw+여유 이하는 미트림");
        assert_eq!(floor_for_pos(1537, sw), 0, "512 미만 내림");
        assert_eq!(floor_for_pos(2048, sw), 512);
        assert_eq!(floor_for_pos(8192, sw), 6656, "(8192−1536)/512 내림");
        assert!(floor_for_pos(9000, sw) >= floor_for_pos(8192, sw), "단조");
    }

    /// 재사용 가드 — 트림 슬롯의 짧은 접두 복원은 콜드.
    #[test]
    fn reuse_guard_needs_floor_below_l_and_window() {
        let sw = 1024;
        assert!(reuse_valid(0, 0, 100, sw), "미트림은 항상 유효");
        assert!(reuse_valid(0, 512, 600, sw), "미트림 단문 유효");
        assert!(reuse_valid(6500, 8000, 8500, sw), "장문 계속 유효");
        assert!(!reuse_valid(6500, 512, 8000, sw), "l이 floor 미만 → 콜드");
        assert!(
            !reuse_valid(6500, 7000, 7000, sw),
            "prompt−sw가 floor 미만 → 콜드"
        );
        assert!(reuse_valid(6500, 7000, 8000, sw), "둘 다 만족 → 유효");
    }
}
