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
        // plans/cuda-port.md S12: 옛엔 여기서 fwd3s 점수 scratch가 공유메모리
        // 1024행이라 ctx>1024를 로드 시점에 경고했다(S9). 커널이 위치 청크
        // 온라인 소프트맥스로 바뀌면서 그 상한이 사라졌으므로 경고도 함께
        // 걷는다 — 이제 위치축 상한은 kvcap(=clamp된 ctx) 자체다.
        let dec = Exl3CudaDecoder::load_slots(dir, usize::MAX, kvcap, slots)?;
        Ok(Self {
            dec,
            eos: exl3_eos_of(dir),
        })
    }

    /// 1토큰 순차 디코드 — 반환 로짓(디바이스 상주 경로, S10).
    pub fn decode1(&mut self, slot: usize, tok: u32) -> Result<Vec<f32>, String> {
        self.dec.forward_tok_device(slot, tok)
    }

    /// 프리필 — 배치 경로(S11)로 T≤8토큰씩 처리해 마지막 로짓을 반환한다.
    ///
    /// [왜 배치인가] T=1 순차는 토큰당 64층을 한 번씩 돈다. T행으로 넘기면
    /// GEMM2·norm이 행 병렬로 처리한다. T 상한은 어텐션 fwd3s의
    /// ATTN_F3S_TMAX(8) — 그보다 큰 청크는 커널이 거부한다.
    ///
    /// [정합] S11 동치성 게이트(cuda_probe s11)가 배치가 T=1과 같은 토큰을
    /// 고름을 실물로 확인했다(어텐션 0.000e0·GDN 7.5e-4·fwd argmax 일치).
    /// f16 mma 누산 차이로 logit maxdiff는 ~1.6e-2지만 argmax는 같고,
    /// 그게 실사용 판정이다(게이트 스크립트와 동일 기준).
    pub fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("빈 프리필".into());
        }
        let tmax = llm170_backend_gpu::rawcuda::attn_cuda::ATTN_F3S_TMAX;
        let mut last = Vec::new();
        for chunk in tokens.chunks(tmax) {
            let mut rows: Vec<f32> = Vec::with_capacity(chunk.len() * self.dec.hidden);
            for &tok in chunk {
                let row = self.dec.embed_row_host(tok);
                if row.len() != self.dec.hidden {
                    return Err(format!("exl3-cuda: 임베딩 토큰 {tok} 범위 밖 또는 미적재"));
                }
                rows.extend_from_slice(&row);
            }
            last = self.dec.forward_batch_device(slot, &rows)?.0;
        }
        Ok(last)
    }

    /// 1토큰 순차 디코드(greedy) — 디바이스 상주 경로(S10). 서버 기본 디코드
    /// 경로다: 토큰당 왕복이 임베딩 업로드·로짓 판독 2회뿐이다(호스트
    /// 스테이징은 GEMV마다 d2h→h2d를 반복해 층당 ~8회).
    ///
    /// [S10 게이트] 두 경로의 종단 토큰열이 동일한 것을 프로브
    /// (cuda_probe s10)가 실측 검증한다 — 산술이 아니라 값으로 증명한다.
    pub fn step_tok_device(&mut self, slot: usize, tok: u32) -> Result<u32, String> {
        // argmax_host가 1MB(로짓 벡터) 장치 버퍼를 cuMemAlloc하므로
        // forward의 가드 밖에서 부르면 INVALID_CONTEXT로 죽는다. 슬롯
        // 스레드에 current 컨텍스트가 전파되지 않기 때문이다(plans/cuda-port.md S5).
        let _g = self.dec.cc.guard()?;
        let logits = self.decode1(slot, tok)?;
        self.dec.argmax_host(&logits)
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
