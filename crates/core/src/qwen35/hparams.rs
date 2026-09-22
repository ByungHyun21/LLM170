//! qwen35 하이퍼파라미터 — GGUF 메타에서 동적 로드.

#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize, // 본체 층수 (MTP 제외)
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub full_attn_interval: usize,
    pub d_inner: usize,
    pub n_group: usize, // GDN K heads
    pub dt_rank: usize, // GDN V heads
    pub d_state: usize,
    pub conv_k: usize,
    pub vocab: usize,
}

impl Hparams {
    pub fn conv_ch(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
    /// rope cos/sin 테이블 [ctx_n][n_rot/2][2] — ops::rope_head과 동일 값.
    /// (raw 디코더 상수 주입과 frame 모듈이 공유 — 단일 소스, plans/35 P6)
    pub fn rope_cs(&self, ctx_n: usize) -> Vec<f32> {
        crate::ops::rope_cs_table(self.n_rot, self.rope_base, ctx_n)
    }
}
