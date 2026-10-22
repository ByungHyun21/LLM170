
#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub vocab: usize,
    pub eps: f32,
    pub sw: usize,
    pub full: Vec<bool>,
    pub head_dim: Vec<usize>,
    pub n_kv: Vec<usize>,
    pub n_rot: Vec<usize>,
    pub rope_base: Vec<f32>,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_ffn: usize,
    pub softcap: f32,
    pub max_pos: usize,
}

impl Hparams {
    pub fn kv_bytes_per_token(&self) -> u64 {
        let mut b = 0u64;
        for il in 0..self.n_layer {
            b += 2 * (self.n_kv[il] as u64) * (self.head_dim[il] as u64) * 4;
        }
        b
    }
}
