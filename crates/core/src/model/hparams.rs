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
}
