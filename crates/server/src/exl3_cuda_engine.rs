//! EXL3 cuda 엔진 어댑터 — Exl3CudaDecoder 단일 슬롯(rawcuda 포팅, plans/124).
//!
//! S4 배선 계약: `--backend cuda` 라우팅 + 디바이스 가중치 상주 적재까지.
//! forward 체인(prefill/decode1/step_tok)은 디코더의 G3+ 조립 스텁을 그대로
//! 위임한다 — 호출 시 plans/124 TODO Err가 그대로 상면에 올라간다(은폐 금지).
//! hip 어댑터(exl3_hip_engine.rs) 미러. MTP spec_round는 디코더 mtp_* 스텁
//! (G4+) 해소 후 개방.

use crate::exl3_engine::exl3_eos_of;
use llm170_backend_gpu::rawcuda::exl3_cuda::Exl3CudaDecoder;

pub struct Exl3CudaEngine {
    dec: Exl3CudaDecoder,
    /// 정지 토큰 — tokenizer_config.json eos_token_id 파생(exl3_eos_of 공용).
    pub eos: u32,
}

// SAFETY: hip 어댑터(exl3_hip_engine.rs)와 동일 근거 — 모든 GPU 접근은
// slot_loop 단일 스레드에서 직렬 실행(디코더 버퍼 단일 소유).
unsafe impl Send for Exl3CudaEngine {}

impl Exl3CudaEngine {
    pub fn load(dir: &str, _n_slots: usize, _ctx_len: usize) -> Result<Self, String> {
        // lim_layers=usize::MAX → 전층 상주(단일 슬롯 전체 모델 계약).
        // KV 버퍼는 load 시점 할당 없음(G4+ 필드 0 유지 — plans/124 §0).
        let dec = Exl3CudaDecoder::load(dir, usize::MAX)?;
        Ok(Self {
            dec,
            eos: exl3_eos_of(dir),
        })
    }

    /// 프리필 — cuda 디코더에는 hip forward_batch_toks(토큰 id 디바이스
    /// gather) 대응이 아직 없다(G3+). 여기서 명확히 막는다(무음 오염 금지).
    pub fn prefill(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, String> {
        Err("exl3-cuda: prefill 미구현 — 디코더 조립(G3+) 전 (plans/124 §0)".into())
    }

    /// 1토큰 순차 디코드 — 반환 로짓(디코더 G3+ 스텁 위임).
    pub fn decode1(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        self.dec.forward_tok(tok)
    }

    /// 1토큰 greedy — GPU argmax(디코더 G3+ 스텁 위임).
    pub fn step_tok(&mut self, tok: u32) -> Result<u32, String> {
        self.dec.step_tok(tok)
    }

    /// 제자리 리셋 — 현 시점 pos 초기화만 한다. GDN 링/스캔 상태 영화는
    /// 디코더 reset_state(G3+)에서 담당 — forward 체인이 미개방인 동안
    /// pos 외 상태는 어차피 불변이다.
    pub fn reset_seq(&mut self) -> Result<(), String> {
        self.dec.pos = 0;
        Ok(())
    }
}
