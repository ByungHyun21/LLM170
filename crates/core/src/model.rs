//! qwen35 (Qwen3.8-27B 계열) CPU 참조 엔진.
//!
//! 그래프 배선: `~/local_llm/llama.cpp/src/models/qwen35.cpp` (2026-08-30 판).
//! - 잔차: h += attn(rms(h)); h += ffn(rms_post(h))
//! - GDN층(interval≠3): qkv → depthwise conv+SiLU → L2 norm(q,k) → GDN → rms_norm·silu(z) → ssm_out
//! - Full-attn층(interval==3): q‖gate fused → per-head rms norm(q,k) → RoPE(n_rot, base, 인접 페어)
//!   → GQA(scale 1/√head_dim) → ⊙sigmoid(gate) → wo
//! - 하이퍼파라미터는 GGUF 메타에서 동적 로드 (소형 검증 모델 지원).
//! - f32 KV, f32 GDN 상태 (참조 정확도 우선).

use llm170_gguf::GgufFile;
use llm170_profiler::profile_span;
use memmap2::Mmap;

use crate::matmul::{matmul, matmul_batch, Weight};
use crate::ops::{l2_norm, rms_norm, rope_head, silu, sigmoid, softplus};
use crate::quant::dequant_row;

#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize,      // 본체 층수 (MTP 제외)
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
    pub n_group: usize,   // GDN K heads
    pub dt_rank: usize,   // GDN V heads
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

#[derive(Debug)]
pub enum ModelError {
    MissingTensor(String),
    UnsupportedLayout { name: String, why: &'static str },
    BadHparam(&'static str),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ModelError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            ModelError::UnsupportedLayout { name, why } => write!(f, "unsupported layout {name}: {why}"),
            ModelError::BadHparam(w) => write!(f, "bad hyperparameter: {w}"),
        }
    }
}

impl std::error::Error for ModelError {}

pub struct Model {
    pub gguf: GgufFile,
    pub hp: Hparams,
    mmap: Mmap,
    pub token_pieces: Vec<String>,
}

macro_rules! span_block {
    ($name:literal, $body:block) => {{
        llm170_profiler::profile_span!($name);
        $body
    }};
}
pub(crate) use span_block;

impl Model {
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        profile_span!("model::load");
        let gguf = GgufFile::open(path)?;
        let file = std::fs::File::open(path)?;
        // SAFETY: 읽기 전용 무게 매핑 — 수정하지 않는다
        let mmap = unsafe { Mmap::map(&file)? };

        let u = |k: &str| gguf.arch_kv_u64(k);
        let n_embd = u("embedding_length").ok_or(ModelError::BadHparam("embedding_length"))? as usize;
        let block_count = u("block_count").unwrap_or(64) as usize;
        let head_count = u("attention.head_count").ok_or(ModelError::BadHparam("head_count"))? as usize;
        let head_dim = u("attention.key_length").ok_or(ModelError::BadHparam("key_length"))? as usize;
        let d_state = u("ssm.state_size").ok_or(ModelError::BadHparam("ssm.state_size"))? as usize;
        let n_group = u("ssm.group_count").ok_or(ModelError::BadHparam("ssm.group_count"))? as usize;
        let dt_rank = u("ssm.time_step_rank").ok_or(ModelError::BadHparam("ssm.time_step_rank"))? as usize;
        let d_inner = u("ssm.inner_size").ok_or(ModelError::BadHparam("ssm.inner_size"))? as usize;

        let hp = Hparams {
            n_layer: block_count.min(64), // block_count=65 → 64본체 + MTP(그래프 외)
            n_embd,
            n_ff: u("feed_forward_length").ok_or(ModelError::BadHparam("feed_forward_length"))? as usize,
            n_head: head_count,
            n_kv: u("attention.head_count_kv").unwrap_or(head_count as u64) as usize,
            head_dim,
            n_rot: u("rope.dimension_count").unwrap_or(head_dim as u64) as usize,
            rope_base: gguf.arch_kv("rope.freq_base").and_then(llm170_gguf::Value::as_f64).unwrap_or(1e7) as f32,
            eps: gguf.arch_kv("attention.layer_norm_rms_epsilon").and_then(llm170_gguf::Value::as_f64).unwrap_or(1e-6) as f32,
            full_attn_interval: u("full_attention_interval").unwrap_or(4).max(1) as usize,
            d_inner,
            n_group,
            dt_rank,
            d_state,
            conv_k: u("ssm.conv_kernel").unwrap_or(4) as usize,
            vocab: 0, // 아래에서 embd 텐서로 확정
        };

        // 정합성: d_inner = dt_rank × head_v_dim, head_v_dim == d_state (delta-net-base assert)
        if d_inner % dt_rank != 0 || d_inner / dt_rank != d_state {
            return Err(ModelError::BadHparam("d_inner/dt_rank != d_state").into());
        }
        if hp.n_head % hp.n_kv != 0 {
            return Err(ModelError::BadHparam("n_head % n_kv != 0").into());
        }

        let mut token_pieces = Vec::new();
        if let Some((_, toks)) = gguf.kv("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
            for t in toks {
                token_pieces.push(t.as_str().unwrap_or("").to_string());
            }
        }

        let vocab = gguf.find_tensor("token_embd.weight").map(|t| t.ne[1]).unwrap_or(0) as usize;
        let hp = Hparams { vocab, ..hp };

        let m = Model { gguf, mmap, hp, token_pieces };
        for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
            m.w(name).ok_or(ModelError::MissingTensor(name.into()))?;
        }
        Ok(m)
    }

    /// 무게 뷰.
    pub fn w(&self, name: &str) -> Option<Weight<'_>> {
        let t = self.gguf.find_tensor(name)?;
        let (start, end) = t.file_range(self.gguf.data_offset)?;
        Some(Weight {
            data: &self.mmap[start as usize..end as usize],
            ty: t.ty,
            n_in: t.ne[0],
            n_out: t.ne[1] * t.ne[2] * t.ne[3],
        })
    }

    fn wchk(&self, name: &str) -> Result<Weight<'_>, ModelError> {
        self.w(name).ok_or_else(|| ModelError::MissingTensor(name.into()))
    }

    fn f32_vec(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        Ok(self.wchk(name)?.dequant_f32_vec())
    }

    pub fn is_recr(&self, il: usize) -> bool {
        il % self.hp.full_attn_interval != self.hp.full_attn_interval - 1
    }
}

/// 시퀀스 상태
pub struct SeqState {
    pub pos: u32,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    gdn_s: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
}

impl SeqState {
    pub fn new(model: &Model, ctx: usize) -> Self {
        let n_full = (0..model.hp.n_layer).filter(|&il| !model.is_recr(il)).count();
        let n_recr = model.hp.n_layer - n_full;
        let (n_kv, hd) = (model.hp.n_kv, model.hp.head_dim);
        let state_size = model.hp.dt_rank * model.hp.d_state * model.hp.d_state;
        let conv_len = (model.hp.conv_k - 1) * model.hp.conv_ch();
        SeqState {
            pos: 0,
            kv_k: vec![vec![0.0; ctx * n_kv * hd]; n_full],
            kv_v: vec![vec![0.0; ctx * n_kv * hd]; n_full],
            gdn_s: vec![vec![0.0; state_size]; n_recr],
            conv: vec![vec![0.0; conv_len]; n_recr],
        }
    }
}

pub struct Engine {
    pub model: Model,
    pub seqs: Vec<SeqState>,
}

impl Engine {
    pub fn new(model: Model, n_seqs: usize, ctx: usize) -> Self {
        let seqs = (0..n_seqs).map(|_| SeqState::new(&model, ctx)).collect();
        Engine { model, seqs }
    }

    /// 배치 포워드: batch[i] = 시퀀스 i 토큰(모든 시퀀스 동일 길이).
    fn forward(&mut self, batch: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("engine::forward");
        let n_seqs = batch.len();
        let t_len = batch.first().map(|v| v.len()).unwrap_or(0);
        assert!(n_seqs > 0 && t_len > 0);
        assert!(batch.iter().all(|v| v.len() == t_len), "배치 내 동일 길이 필수 (equal_seqs)");

        let hp = self.model.hp.clone();
        let n_embd = hp.n_embd;
        let n_tok = n_seqs * t_len;
        let embd = self.model.wchk("token_embd.weight")?;

        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        for seq_tokens in batch {
            for &tok in seq_tokens {
                let mut row = vec![0.0f32; n_embd];
                dequant_row(embd.ty, embd.data, tok as u64, n_embd as u64, &mut row);
                xs.push(row);
            }
        }

        let mut full_idx = 0usize;
        let mut recr_idx = 0usize;
        for il in 0..hp.n_layer {
            let norm_w = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
            // 잔차: pre-norm 원본 보존 (qwen35.cpp:162-184 — inpSA)
            let residual: Vec<Vec<f32>> = xs.clone();
            for x in xs.iter_mut() {
                *x = rms_norm(x, &norm_w, hp.eps);
            }

            let attn_out = if self.model.is_recr(il) {
                let o = self.gdn_layer(il, &xs, n_seqs, t_len, recr_idx)?;
                recr_idx += 1;
                o
            } else {
                let o = self.attn_layer(il, &xs, n_seqs, t_len, full_idx)?;
                full_idx += 1;
                o
            };

            for (t, a) in attn_out.iter().enumerate() {
                for i in 0..n_embd {
                    xs[t][i] = residual[t][i] + a[i];
                }
            }
            let ffn_residual = xs.clone();
            let dbg_layer = il == 0 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some();
            if dbg_layer {
                let m = xs.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
                eprintln!("  rs stage post-attn-residual max={m:.5}");
            }

            let post_w = self.model.f32_vec(&format!("blk.{il}.post_attention_norm.weight"))?;
            let gate_w = self.model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
            let up_w = self.model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
            let down_w = self.model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
            let n_ff = hp.n_ff;

            let mut normed: Vec<Vec<f32>> = xs.iter().map(|x| rms_norm(x, &post_w, hp.eps)).collect();
            let mut gate_y = vec![vec![0.0f32; n_ff]; n_tok];
            let mut up_y = vec![vec![0.0f32; n_ff]; n_tok];
            {
                span_block!("cpu::ffn_gate", {
                    matmul_batch(&normed, &gate_w, &mut gate_y);
                });
            }
            {
                span_block!("cpu::ffn_up", {
                    matmul_batch(&normed, &up_w, &mut up_y);
                });
            }
            for t in 0..n_tok {
                for i in 0..n_ff {
                    gate_y[t][i] = silu(gate_y[t][i]) * up_y[t][i];
                }
            }
            {
                span_block!("cpu::ffn_down", {
                    matmul_batch(&gate_y, &down_w, &mut xs);
                });
            }
            if dbg_layer {
                let m = gate_y.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
                eprintln!("  rs stage ffn_siluup max={m:.5}");
            }
            for t in 0..n_tok {
                for i in 0..n_embd {
                    xs[t][i] += ffn_residual[t][i];
                }
            }
            if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                let m = xs[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = xs[0].iter().any(|v| v.is_nan());
                let v4: Vec<String> = xs[0][..4].iter().map(|v| format!("{v:.5}")).collect();
                eprintln!("layer {il:>2} recr={} max|x|={m:.4} nan={nan} head={}", self.model.is_recr(il), v4.join(","));
            }
        }

        // output_norm + logits (시퀀스별 마지막 토큰만)
        let out_norm = self.model.f32_vec("output_norm.weight")?;
        let head = self.model.wchk("output.weight")?;
        let mut result = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let last = &xs[(s + 1) * t_len - 1];
            let h = rms_norm(last, &out_norm, hp.eps);
            let mut logits = vec![0.0f32; head.n_out as usize];
            matmul(&h, &head, &mut logits);
            if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = logits.iter().any(|v| v.is_nan());
                eprintln!("logits: max={m:.4} nan={nan} argmax={}", greedy(&logits));
            }
            result.push(logits);
        }
        Ok(result)
    }

    /// GDN층. xs: [n_tok][n_embd], seq-major.
    fn gdn_layer(
        &mut self,
        il: usize,
        xs: &[Vec<f32>],
        n_seqs: usize,
        t_len: usize,
        recr_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_gdn");
        let hp = &self.model.hp;
        let n_tok = n_seqs * t_len;
        let (d_state, dt_rank, n_group, d_inner) = (hp.d_state, hp.dt_rank, hp.n_group, hp.d_inner);
        let (conv_k, conv_ch) = (hp.conv_k, hp.conv_ch());
        let wqkv = self.model.wchk(&format!("blk.{il}.attn_qkv.weight"))?;
        let wgate = self.model.wchk(&format!("blk.{il}.attn_gate.weight"))?;
        let wbeta = self.model.wchk(&format!("blk.{il}.ssm_beta.weight"))?;
        let walpha = self.model.wchk(&format!("blk.{il}.ssm_alpha.weight"))?;
        let ssm_a = self.model.f32_vec(&format!("blk.{il}.ssm_a"))?;
        let dt_bias = self.model.f32_vec(&format!("blk.{il}.ssm_dt.bias"))?;
        let conv_w = self.model.f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))?; // [conv_k][conv_ch] 행 우선
        let ssm_norm_w = self.model.f32_vec(&format!("blk.{il}.ssm_norm.weight"))?;
        let wout = self.model.wchk(&format!("blk.{il}.ssm_out.weight"))?;

        let dbg0 = il == 0 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some();
        let mut qkv = vec![vec![0.0f32; conv_ch]; n_tok];
        {
            span_block!("cpu::gdn_qkv", {
                matmul_batch(xs, &wqkv, &mut qkv);
            });
        }
        let mut z = vec![vec![0.0f32; d_inner]; n_tok];
        {
            span_block!("cpu::gdn_z", {
                matmul_batch(xs, &wgate, &mut z);
            });
        }

        if dbg0 {
            let mz = z.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let mq = qkv.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let mc = xs.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!("  rs stage cur max={mc:.5} qkv max={mq:.5} z max={mz:.5}");
        }
        let mut beta_all = vec![0.0f32; n_tok * dt_rank];
        let mut g_all = vec![0.0f32; n_tok * dt_rank];
        {
            span_block!("cpu::gdn_bg", {
                let mut b = vec![vec![0.0f32; dt_rank]; n_tok];
                let mut a = vec![vec![0.0f32; dt_rank]; n_tok];
                matmul_batch(xs, &wbeta, &mut b);
                matmul_batch(xs, &walpha, &mut a);
                for t in 0..n_tok {
                    for h in 0..dt_rank {
                        beta_all[t * dt_rank + h] = sigmoid(b[t][h]);
                        g_all[t * dt_rank + h] = softplus(a[t][h] + dt_bias[h]) * ssm_a[h];
                    }
                }
            });
        }

        let k_len = n_group * d_state;
        let v_len = dt_rank * d_state;
        let mut q_all = vec![0.0f32; n_tok * k_len];
        let mut k_all = vec![0.0f32; n_tok * k_len];
        let mut v_all = vec![0.0f32; n_tok * v_len];
        let mut o_all = vec![0.0f32; n_tok * v_len];

        {
            profile_span!("cpu::gdn_conv");
            for s in 0..n_seqs {
                let conv_state = &mut self.seqs[s].conv[recr_idx];
                for t in 0..t_len {
                    let row = s * t_len + t;
                    for c in 0..conv_ch {
                        // ggml ssm_conv: weight {d_conv, d_inner} 행 우선 → w[c*conv_k + j]
                        let mut sum = conv_w[c * conv_k + (conv_k - 1)] * qkv[row][c];
                        for j in 0..conv_k - 1 {
                            sum += conv_w[c * conv_k + j] * conv_state[j * conv_ch + c];
                        }
                        let out_c = silu(sum);
                        for j in 0..conv_k - 2 {
                            conv_state[j * conv_ch + c] = conv_state[(j + 1) * conv_ch + c];
                        }
                        conv_state[(conv_k - 2) * conv_ch + c] = qkv[row][c];
                        // 레이아웃: q [k heads] | k [k heads] | v [v heads]
                        if c < k_len {
                            q_all[row * k_len + c] = out_c;
                        } else if c < 2 * k_len {
                            k_all[row * k_len + c - k_len] = out_c;
                        } else {
                            v_all[row * v_len + c - 2 * k_len] = out_c;
                        }
                    }
                }
            }
        }

        {
            profile_span!("cpu::gdn_l2norm");
            for row in 0..n_tok {
                for h in 0..n_group {
                    let b0 = row * k_len + h * d_state;
                    let head: Vec<f32> = q_all[b0..b0 + d_state].to_vec();
                    q_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&head, hp.eps));
                    let headk: Vec<f32> = k_all[b0..b0 + d_state].to_vec();
                    k_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&headk, hp.eps));
                }
            }
        }

        {
            profile_span!("cpu::gdn_core");
            for s in 0..n_seqs {
                let r0 = s * t_len;
                let r1 = r0 + t_len;
                let st = &mut self.seqs[s].gdn_s[recr_idx];
                if t_len == 1 {
                    crate::gdn::gdn_ar_batch(
                        &q_all[r0 * k_len..r1 * k_len],
                        &k_all[r0 * k_len..r1 * k_len],
                        &v_all[r0 * v_len..r1 * v_len],
                        &beta_all[r0 * dt_rank..r1 * dt_rank],
                        &g_all[r0 * dt_rank..r1 * dt_rank],
                        st,
                        &mut o_all[r0 * v_len..r1 * v_len],
                        1,
                        n_group,
                        dt_rank,
                    );
                } else {
                    crate::gdn::gdn_chunk_seq(
                        &q_all[r0 * k_len..r1 * k_len],
                        &k_all[r0 * k_len..r1 * k_len],
                        &v_all[r0 * v_len..r1 * v_len],
                        &beta_all[r0 * dt_rank..r1 * dt_rank],
                        &g_all[r0 * dt_rank..r1 * dt_rank],
                        st,
                        &mut o_all[r0 * v_len..r1 * v_len],
                        t_len,
                        n_group,
                        dt_rank,
                    );
                }
            }
        }

        if dbg0 {
            let mq = q_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mk = k_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mv = v_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mo = o_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let c4: Vec<String> = o_all[..4].iter().map(|v| format!("{v:.6}")).collect();
            let z4: Vec<String> = z[0][..4].iter().map(|v| format!("{v:.6}")).collect();
            eprintln!("  rs stage core[:4]={c4:?} z[:4]={z4:?}");
        }
        // norm_gated: rms_norm(core)·silu(z) per head → ssm_out
        let mut gated = vec![vec![0.0f32; d_inner]; n_tok];
        {
            profile_span!("cpu::gdn_normgated");
            for t in 0..n_tok {
                for h in 0..dt_rank {
                    let b0 = t * v_len + h * d_state;
                    let head: Vec<f32> = o_all[b0..b0 + d_state].to_vec();
                    let n = rms_norm(&head, &ssm_norm_w, hp.eps);
                    let zb = h * d_state;
                    for i in 0..d_state {
                        gated[t][zb + i] = n[i] * silu(z[t][zb + i]);
                    }
                }
            }
        }
        if dbg0 {
            let mg = gated.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let fmt = |o: usize| -> String { gated[0][o..o + 4].iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(",") };
            eprintln!("  rs gated h0={} h1={} h2={} h3={} (max={mg:.5})", fmt(0), fmt(16), fmt(32), fmt(48));
        }
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        {
            span_block!("cpu::gdn_out", {
                matmul_batch(&gated, &wout, &mut out);
            });
        }
        if dbg0 {
            let m = out.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!("  rs stage ssm_out max={m:.5}");
        }
        Ok(out)
    }

    /// Full-attention층 (gated, IMROPE 텍스트 퇴화형).
    fn attn_layer(
        &mut self,
        il: usize,
        xs: &[Vec<f32>],
        n_seqs: usize,
        t_len: usize,
        full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_attn");
        let hp = &self.model.hp;
        let n_tok = n_seqs * t_len;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = self.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;

        let mut qg = vec![vec![0.0f32; wq.n_out as usize]; n_tok];
        let mut kk = vec![vec![0.0f32; wk.n_out as usize]; n_tok];
        let mut vv = vec![vec![0.0f32; wv.n_out as usize]; n_tok];
        {
            span_block!("cpu::attn_qkv", {
                matmul_batch(xs, &wq, &mut qg);
                matmul_batch(xs, &wk, &mut kk);
                matmul_batch(xs, &wv, &mut vv);
            });
        }

        let kq_scale = hp.kq_scale();
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        for s in 0..n_seqs {
            let pos0 = self.seqs[s].pos;
            let seq = &mut self.seqs[s];
            let (cache_k, cache_v) =
                (seq.kv_k[full_idx].as_mut_slice(), seq.kv_v[full_idx].as_mut_slice());

            for t in 0..t_len {
                let row = s * t_len + t;
                let pos = pos0 + t as u32;
                for h in 0..n_kv {
                    let src = kk[row][h * hd..h * hd + hd].to_vec();
                    let mut head = rms_norm(&src, &k_norm_w, hp.eps);
                    rope_head(&mut head, pos, n_rot, hp.rope_base);
                    let b = (pos as usize) * n_kv * hd + h * hd;
                    cache_k[b..b + hd].copy_from_slice(&head);
                    cache_v[b..b + hd].copy_from_slice(&vv[row][h * hd..h * hd + hd]);
                }
                let mut attn_out = vec![0.0f32; n_head * hd];
                for h in 0..n_head {
                    let src = qg[row][h * 2 * hd..h * 2 * hd + hd].to_vec();
                    let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                    rope_head(&mut qh, pos, n_rot, hp.rope_base);
                    let kvh = h / (n_head / n_kv);
                    let n_past = pos as usize + 1;
                    let mut scores = vec![0.0f32; n_past];
                    let mut maxv = f32::NEG_INFINITY;
                    for (p, sc) in scores.iter_mut().enumerate() {
                        let b = p * n_kv * hd + kvh * hd;
                        let mut d = 0.0f32;
                        for i in 0..hd {
                            d += qh[i] * cache_k[b + i];
                        }
                        *sc = d * kq_scale;
                        maxv = maxv.max(*sc);
                    }
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - maxv).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    let ob = h * hd;
                    for p in 0..n_past {
                        let w = scores[p];
                        if w == 0.0 {
                            continue;
                        }
                        let b = p * n_kv * hd + kvh * hd;
                        for i in 0..hd {
                            attn_out[ob + i] += w * cache_v[b + i];
                        }
                    }
                    // q‖gate 인접 인터리브: gate at h*2*hd + hd
                    let gb = h * 2 * hd + hd;
                    for i in 0..hd {
                        attn_out[ob + i] *= sigmoid(qg[row][gb + i]);
                    }
                }
                matmul(&attn_out, &wo, &mut out[row]);
            }
        }
        Ok(out)
    }

    /// 배치 디코드: seq_ids 각각 토큰 1개. 위치 진행 후 logits 반환.
    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        let batch: Vec<Vec<u32>> = tokens.iter().map(|t| vec![*t]).collect();
        let logits = self.forward(&batch)?;
        for s in seq_ids {
            self.seqs[*s].pos += 1;
        }
        Ok(logits)
    }

    /// 시퀀스 prefill: 전체 토큰 적립 + 마지막 logits.
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let logits = self.forward(&[tokens.to_vec()])?;
        self.seqs[seq].pos += tokens.len() as u32;
        Ok(logits.into_iter().next().unwrap())
    }

    /// 표면형 근사 디토크 (표시용 — 정식 BPE 디토크나이저는 후속)
    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(String::as_str)
            .unwrap_or("")
            .replace('Ġ', " ")
            .replace('Ċ', "\n")
    }
}

pub fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}
