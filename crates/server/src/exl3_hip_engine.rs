//! EXL3 hip 엔진 어댑터 (plans/121 · exl3-sched) — Exl3HipDecoder 단일 슬롯 v1.
//!
//! 기본 서빙 경로(사용자 2026-10-04 지시: MTP·병렬 슬롯 없는 구동 먼저):
//! 프리필 = forward_batch 청크(상각곡선 고정 ~320ms + 35ms/토큰 — T=16 청크로
//! 18.8 t/s), 디코드 = forward_tok 순차(6.25 t/s, greedy-4 무결).
//!
//! 단일 슬롯: Exl3HipDecoder는 상태(dring/dgst/KV/pos)를 1세트만 보유 —
//! n_slots>1 요청은 Err(다중 슬롯은 병렬-슬롯 캠페인에서 상태 분리 후 개방).
//! reset은 제자리(dring/dgst 0-fill + pos 0 — KV는 pos 도달 시 자연 갱신).

use llm170_backend_gpu::rawhip::exl3_hip::Exl3HipDecoder;

pub struct Exl3HipEngine {
    dec: Exl3HipDecoder,
}

// SAFETY: decoder의 모든 GPU 접근은 slot_loop 단일 스레드에서 직렬 실행 —
// vk 엔진 어댑터의 Send 근거와 동일(매핑 포인터 단일 소유).
unsafe impl Send for Exl3HipEngine {}

impl Exl3HipEngine {
    pub fn load(dir: &str, _n_slots: usize, _ctx_len: usize) -> Result<Self, String> {
        let dec = Exl3HipDecoder::load(dir, 64)?;
        Ok(Self { dec })
    }

    /// 프리필 — 청크 16행(상각 곡선의 실용점). 반환 = 마지막 로짓.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("빈 프리필".into());
        }
        let mut last = Vec::new();
        for chunk in tokens.chunks(16) {
            let rows: Vec<Vec<f32>> = chunk.iter().map(|&t| self.dec.embed_row_host(t)).collect();
            let (lgs, _) = self.dec.forward_batch(&rows)?;
            last = lgs.last().cloned().ok_or("빈 배치")?;
        }
        Ok(last)
    }

    /// 1토큰 순차 디코드 — 반환 로짓.
    pub fn decode1(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        self.dec.forward_tok(tok)
    }

    /// 제자리 리셋 — 링/스캔 상태 0화 + pos 초기화(KV는 pos 의미론으로 무해).
    pub fn reset_seq(&mut self) -> Result<(), String> {
        self.dec.reset_state()
    }
}
