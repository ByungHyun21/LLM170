//! gemma4 하이퍼파라미터 — config.json(text_config 중첩)에서 동적 로드.
//! qwen35와 달리 **층별 dims가 1급 필드**다: 슬라이딩(256/8kv·SW 1024)과
//! 풀(512/2kv·k_eq_v)이 섞이고 rope도 2종(기본 전차원·비례 부분회전)이다.

#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    /// dense MLP 폭(2176 — 2026-10-14 실측판; 구판 2112는 config 파생이라 무영향) — MoE와 병렬.
    pub n_ff: usize,
    /// Q 헤드 수(16 — 전 층 공통).
    pub n_head: usize,
    pub vocab: usize,
    pub eps: f32,
    /// 슬라이딩 창(0 = 없음) — 슬라이딩 층 마스크 폭(자기 포함).
    pub sw: usize,
    /// 층별 — 풀 어텐션 여부·head_dim(256/512)·KV 헤드(8/2).
    pub full: Vec<bool>,
    pub head_dim: Vec<usize>,
    pub n_kv: Vec<usize>,
    /// 층별 rope — 회전 차원 수(슬라이딩 256·풀 128)·base(1e4/1e6).
    /// 주파수 분모 = head_dim(256/512) — ops::rope_head_dim.
    pub n_rot: Vec<usize>,
    pub rope_base: Vec<f32>,
    /// MoE — 전 층 병렬 병합(0이면 dense 단독).
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_ffn: usize,
    /// 최종 로짓 softcap(0 = 미적용) — logits = c·tanh(l/c).
    pub softcap: f32,
    pub max_pos: usize,
}

impl Hparams {
    /// KV 바이트/토큰(f32 참조 상태 = K+V) — 진단·회계용.
    pub fn kv_bytes_per_token(&self) -> u64 {
        let mut b = 0u64;
        for il in 0..self.n_layer {
            b += 2 * (self.n_kv[il] as u64) * (self.head_dim[il] as u64) * 4;
        }
        b
    }
}
