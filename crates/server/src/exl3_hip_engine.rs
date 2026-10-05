//! EXL3 hip 엔진 어댑터 (plans/121 · exl3-sched) — Exl3HipDecoder 단일 슬롯 v1.
//!
//! 기본 서빙 경로(사용자 2026-10-04 지시: MTP·병렬 슬롯 없는 구동 먼저):
//! 프리필 = forward_batch 청크(64행 — plans/128 P1), 디코드 = greedy는
//! step_tok GPU argmax(plans/130 A2), 샘플링은 forward_tok 로짓 판.
//!
//! 단일 슬롯: Exl3HipDecoder는 상태(dring/dgst/KV/pos)를 1세트만 보유 —
//! n_slots>1 요청은 Err(다중 슬롯은 병렬-슬롯 캠페인에서 상태 분리 후 개방).
//! reset은 제자리(dring/dgst 0-fill + pos 0 — KV는 pos 도달 시 자연 갱신).

use llm170_backend_gpu::rawhip::exl3_hip::Exl3HipDecoder;

use crate::exl3_engine::exl3_eos_of;

pub struct Exl3HipEngine {
    dec: Exl3HipDecoder,
    /// MTP 스펙 상태(plans/130 D2): 마지막 커밋 시점 hidden(V3 post-final-norm)
    /// + 타깃 자신의 다음 예측(미처리). spec_round가 소비·갱신.
    mtp_h: Option<Vec<f32>>,
    mtp_pp: u32,
    /// 정지 토큰(plans/130 F5) — tokenizer_config.json eos_token_id 파생,
    /// 실패 시 qwen 계열 기본 248044.
    pub eos: u32,
}

// SAFETY: decoder의 모든 GPU 접근은 slot_loop 단일 스레드에서 직렬 실행 —
// vk 엔진 어댑터의 Send 근거와 동일(매핑 포인터 단일 소유).
unsafe impl Send for Exl3HipEngine {}

impl Exl3HipEngine {
    pub fn load(dir: &str, _n_slots: usize, ctx_len: usize) -> Result<Self, String> {
        // kvcap = 서빙 ctx(plans/128 P0) — 과거 req.ctx를 무시해 KV가 1024로
        // 고정됐고 32k 요청이 pos≈1023에서 attn_prep 폴트로 죽었다.
        // 상한 32768: 그 이상은 이 UMA 기기의 KV RAM 예산(GTT 동결 위험) 초과 —
        // 범위 밖 요청은 클램프 후 로그로 알린다(엄청난 --ctx 오타 방지).
        let kvcap = if ctx_len == 0 {
            4096
        } else {
            ctx_len.clamp(64, 32768)
        };
        if kvcap != ctx_len {
            eprintln!("# hip kvcap: ctx {ctx_len} → {kvcap} (범위 [64, 32768]로 클램프)");
        }
        let dec = Exl3HipDecoder::load(dir, 64, kvcap)?;
        Ok(Self {
            dec,
            mtp_h: None,
            mtp_pp: 0,
            eos: exl3_eos_of(dir),
        })
    }

    /// 프리필 — 청크 64행(plans/128 P1: 어텐션 t-런치 배치화+had16 수리 완료로
    /// t=64 형상 활성 — mma 상각 개선. 반환 = 마지막 로짓.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("빈 프리필".into());
        }
        let mut last = Vec::new();
        let mut h = Vec::new();
        for chunk in tokens.chunks(64) {
            // plans/130 A3: 임베딩 행 d2h→h2d 왕복 대신 토큰 id 디바이스 gather.
            let (lgs, lh) = self.dec.forward_batch_toks(chunk)?;
            last = lgs.last().cloned().ok_or("빈 배치")?;
            h = lh;
        }
        // 스펙 상태 시딩(D2): h = 마지막 행 post-final-norm hidden, pp = 다음 예측.
        self.mtp_h = Some(h);
        self.mtp_pp = llm170_core::qwen35::greedy(&last);
        Ok(last)
    }

    /// 1토큰 순차 디코드 — 반환 로짓.
    pub fn decode1(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        self.dec.forward_tok(tok)
    }

    /// 1토큰 순차 디코드(greedy) — GPU argmax, 로짓 1MB d2h 스킵(plans/130 A2).
    pub fn step_tok(&mut self, tok: u32) -> Result<u32, String> {
        self.dec.step_tok(tok)
    }

    /// 제자리 리셋 — 링/스캔 상태 0화 + pos 초기화(KV는 pos 의미론으로 무해).
    pub fn reset_seq(&mut self) -> Result<(), String> {
        self.dec.reset_state()?;
        self.mtp_h = None;
        self.mtp_pp = 0;
        Ok(())
    }

    /// MTP 스펙 라운드(plans/130 D2 — 프로브 mtp-round v2에 롤백 추가).
    /// greedy 전용. 반환 = 이번 라운드 새로 확정 토큰(수용 드래프트 + 교정;
    /// pp 자체는 호출자가 이미 배출). 거부 시 GDN 스냅샷 복원 후 수용 접두+
    /// 교정 재처리로 진짜 상태 정렬(무롤백 재사용 오염 없음).
    pub fn spec_round(&mut self, k: usize) -> Result<Vec<u32>, String> {
        let h = self.mtp_h.clone().ok_or("spec_round: prefill 선행 필요")?;
        let pp = self.mtp_pp;
        let pos_now = self.dec.pos;
        // 드래프트 k체인 — MTP 헤드가 pp를 자체 처리해 pp+1..을 예측.
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut hcur = h.clone();
        let mut prev = pp;
        for i in 0..k {
            let (d, hd) = self.dec.mtp_draft_gpu(prev, &hcur, pos_now + i as u32)?;
            drafts.push(d);
            prev = d;
            hcur = hd;
        }
        // 검증 직전 스냅샷(롤백 포인트 — 지속 버퍼).
        self.dec.gdn_save()?;
        let mut toks = vec![pp];
        toks.extend(drafts.iter().copied());
        let rows: Vec<Vec<f32>> = toks.iter().map(|&t| self.dec.embed_row_host(t)).collect();
        let (lgs, hnew) = self.dec.forward_batch_with_mtp(&rows, &toks)?;
        let am = |row: &[f32]| llm170_core::qwen35::greedy(row);
        let mut out: Vec<u32> = Vec::with_capacity(k);
        let mut rejected_at: Option<usize> = None;
        for j in 0..k {
            let amj = am(&lgs[j]);
            if amj == drafts[j] {
                out.push(drafts[j]);
            } else {
                out.push(amj);
                rejected_at = Some(j);
                break;
            }
        }
        match rejected_at {
            None => {
                // 전부 수용 — row_k(마지막 드래프트 후)가 다음 pp.
                self.mtp_pp = am(&lgs[k]);
                self.mtp_h = Some(hnew);
            }
            Some(j) => {
                // 롤백 → [pp, 수용 접두 d_1..d_j, corr] 재처리 = 진짜 상태.
                let corr = out.last().copied().unwrap_or(pp);
                self.dec.gdn_rollback()?;
                let mut replay = vec![pp];
                replay.extend_from_slice(&drafts[..j]);
                replay.push(corr);
                let rows2: Vec<Vec<f32>> =
                    replay.iter().map(|&t| self.dec.embed_row_host(t)).collect();
                let (lgr, hr) = self.dec.forward_batch_with_mtp(&rows2, &replay)?;
                self.mtp_pp = llm170_core::qwen35::greedy(&lgr.last().cloned().unwrap_or_default());
                self.mtp_h = Some(hr);
            }
        }
        Ok(out)
    }
}
