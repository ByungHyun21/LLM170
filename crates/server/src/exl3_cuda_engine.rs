//! EXL3 cuda 엔진 어댑터 — Exl3CudaDecoder 다중 슬롯(rawcuda 포팅, plans/124).
//!
//! S5 배선 계약(plans/cuda-port.md): ctx 기반 KV 할당 + 순차 프리필·디코드.
//! S8: 슬롯은 디코더가 GDN 링/스캔 상태·KV 캐시·pos를 슬롯별로 보유한다
//! (가중치는 전 슬롯 공유). hip 어댑터(exl3_hip_engine.rs) 미러.
//! MTP spec_round는 디코더 mtp_* 스텁(G4+) 해소 후 개방.

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
    /// n_slots개 동시 시퀀스. 슬롯당 VRAM이 선형으로 증가하므로
    /// (27B·ctx 4096 기준 GDN 157MB + KV 536MB) 상한을 넘어가면
    /// cuMemAlloc이 조용히 실패하지 않고 Err로 거절한다 — 세그먼트
    /// 폴트 대신 우아한 거절이 계약이다(plans/128 P0와 동일 논리).
    pub fn load(dir: &str, n_slots: usize, ctx_len: usize) -> Result<Self, String> {
        let slots = n_slots.max(1);
        // plans/cuda-port.md S5: hip와 동일한 ctx 범위로 KV 용량을 정한다.
        let kvcap = if ctx_len == 0 {
            4096
        } else {
            ctx_len.clamp(64, 32768)
        };
        if kvcap != ctx_len {
            eprintln!("# cuda kvcap: ctx {ctx_len} → {kvcap} (범위 [64, 32768]로 클램프)");
        }
        let dec = Exl3CudaDecoder::load_slots(dir, usize::MAX, kvcap, slots)?;
        Ok(Self {
            dec,
            eos: exl3_eos_of(dir),
        })
    }

    /// 프리필 — S5 호스트 임베딩 경로(plans/cuda-port.md): 토큰을 순차 처리해
    /// 마지막 로짓을 반환한다. 빈 프롬프트는 디코더 상태를 건드리지 않고 거부.
    pub fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("빈 프리필".into());
        }
        let mut last = Vec::new();
        for &tok in tokens {
            last = self.dec.forward_tok(slot, tok)?;
        }
        Ok(last)
    }

    /// 1토큰 순차 디코드 — 반환 로짓.
    pub fn decode1(&mut self, slot: usize, tok: u32) -> Result<Vec<f32>, String> {
        self.dec.forward_tok(slot, tok)
    }

    /// 1토큰 순차 디코드(greedy) — GPU argmax.
    pub fn step_tok(&mut self, slot: usize, tok: u32) -> Result<u32, String> {
        self.dec.step_tok(slot, tok)
    }

    /// 슬롯 제자리 리셋 — GDN 링/스캔 상태와 pos를 디코더에서 함께 초기화.
    pub fn reset_seq(&mut self, slot: usize) -> Result<(), String> {
        self.dec.reset_state(slot)
    }

    /// 전 슬롯 리셋(워밍업 종료 후).
    pub fn reset_states(&mut self) -> Result<(), String> {
        self.dec.reset_states()
    }
}
