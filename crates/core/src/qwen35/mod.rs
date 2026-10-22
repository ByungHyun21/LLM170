
pub mod bind;
mod diag;
mod dispatch;
use dispatch::{mm, mm_batch, mm_group};
pub const EOS_EOT: u32 = 248044;
pub mod hparams;
pub mod prefill;
pub mod stages;

use hparams::Hparams;
use llm170_diag::profile_span;

use crate::matmul::Weight;
use crate::ops::{rms_norm, silu};
use crate::quant::dequant_row;

#[derive(Debug)]
pub enum ModelError {
    MissingTensor(String),
    BadHparam(&'static str),
    W4a16(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            ModelError::BadHparam(w) => write!(f, "bad hyperparameter: {w}"),
            ModelError::W4a16(e) => write!(f, "w4a16: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

pub struct Model {
    w4: Box<crate::w4a16::W4a16Model>,
    cfg: bind::QwenCfg,
    perm: std::sync::OnceLock<bind::PermStore>,
    pub hp: Hparams,
    pub token_pieces: Vec<String>,
    f32_cache: std::cell::RefCell<std::collections::HashMap<String, Vec<f32>>>,
}

macro_rules! span_block {
    ($name:literal, $body:block) => {{
        llm170_diag::profile_span!($name);
        $body
    }};
}
pub(crate) use span_block;

impl Model {
    pub fn load(dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        profile_span!("model::load_w4a16");
        let w4 =
            crate::w4a16::W4a16Model::open(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let cfg = bind::QwenCfg::load(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let rep = bind::validate(&w4, &cfg).map_err(|e| ModelError::W4a16(e.to_string()))?;
        if !rep.ok() {
            return Err(
                ModelError::W4a16(format!("커버리지 검증 실패:\n{}", rep.summary())).into(),
            );
        }
        let c = &cfg;
        let d_state = c.linear_key_head_dim;
        let d_inner = c.linear_num_value_heads * c.linear_key_head_dim;
        let hp = Hparams {
            n_layer: c.layers,
            n_embd: c.hidden,
            n_ff: c.ffn,
            n_head: c.heads,
            n_kv: c.kv_heads,
            head_dim: c.head_dim,
            n_rot: (c.head_dim as f64 * c.partial_rotary_factor) as usize,
            rope_base: c.rope_theta as f32,
            eps: c.rms_norm_eps as f32,
            full_attn_interval: c.full_interval,
            d_inner,
            n_group: c.linear_num_key_heads,
            dt_rank: c.linear_num_value_heads,
            d_state,
            conv_k: c.linear_conv_kernel,
            vocab: c.vocab,
            max_pos: c.max_pos,
            n_experts: c.n_experts,
            top_k: c.top_k,
            moe_ffn: c.moe_ffn,
            shared_ffn: c.shared_ffn,
        };
        if !d_inner.is_multiple_of(hp.dt_rank) || d_inner / hp.dt_rank != d_state {
            return Err(ModelError::BadHparam("d_inner/dt_rank != d_state").into());
        }
        let token_pieces =
            crate::w4a16::load_pieces(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let m = Model {
            w4: Box::new(w4),
            cfg,
            perm: std::sync::OnceLock::new(),
            hp,
            token_pieces,
            f32_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        };
        for name in ["token_embd.weight", "output.weight"] {
            m.w(name).ok_or(ModelError::MissingTensor(name.into()))?;
        }
        m.f32_vec("output_norm.weight")?;
        Ok(m)
    }

    pub fn w(&self, name: &str) -> Option<Weight<'_>> {
        w4_weight(self, name)
    }

    pub fn engine_names(&self) -> Vec<String> {
        bind::engine_names(&self.cfg)
    }

    pub fn store(&self) -> &crate::w4a16::W4a16Model {
        &self.w4
    }

    pub fn w_raw_names(&self, name: &str) -> Option<(String, Option<String>)> {
        use bind::Eng;
        match bind::eng(name)? {
            Eng::Quant { base, .. } => {
                if self.w4.lin_shape(&base).is_some() {
                    Some((
                        format!("{base}.weight_packed"),
                        Some(format!("{base}.weight_scale")),
                    ))
                } else {
                    Some((format!("{base}.weight"), None))
                }
            }
            Eng::Plain { name: hf, .. } => Some((hf, None)),
            Eng::Synth { .. } => None,
        }
    }

    pub fn expert_names(&self, il: usize, e: usize, proj: &str) -> (String, String) {
        let base = bind::expert_base(il, e, proj);
        (
            format!("{base}.weight_packed"),
            format!("{base}.weight_scale"),
        )
    }

    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, ModelError> {
        let Some(bind::Eng::Plain { name, .. }) = bind::eng("token_embd.weight") else {
            return Err(ModelError::MissingTensor("token_embd.weight".into()));
        };
        if let (Some(buf), Some(e)) = (self.w4.tensor_slice(&name), self.w4.entry(&name))
            && e.shape.len() == 2
        {
            let cols = e.shape[1] as usize;
            let nb = e.dtype.nbytes() as usize;
            let off = tok as usize * cols * nb;
            if let Some(row) = buf.get(off..off + cols * nb) {
                return crate::w4a16::decode_f32(row, e.dtype, &name)
                    .map_err(|err| ModelError::W4a16(err.to_string()));
            }
        }
        self.w4
            .plain_rows_f32(&name, tok as u64, tok as u64 + 1)
            .map_err(|e| ModelError::W4a16(e.to_string()))
    }

    pub fn expert_slice(
        &self,
        il: usize,
        e: usize,
        proj: &str,
    ) -> Option<(&[u8], &[u8], usize, usize)> {
        self.w4.expert_at(&bind::expert_base(il, e, proj))
    }

    pub fn expert_quant(&self) -> (usize, bool) {
        let base = bind::expert_base(0, 0, "gate_proj");
        (self.w4.group(), self.w4.scale_is_bf16(&base))
    }

    pub fn expert_w(&self, il: usize, e: usize, proj: &str) -> Option<Weight<'_>> {
        let base = bind::expert_base(il, e, proj);
        let (data, scale, n, k) = self.w4.expert_at(&base)?;
        Some(Weight {
            data,
            aux: Some(scale),
            ty: crate::wtype::WType::W4a16Split,
            n_in: k as u64,
            n_out: n as u64,
            group: self.w4.group(),
            scale_bf16: self.w4.scale_is_bf16(&base),
        })
    }

    pub fn w_raw(&self, name: &str) -> Option<Weight<'_>> {
        w4_weight_raw(&self.w4, name)
    }

    pub fn raw_f32_vec(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let hf = match bind::eng(name) {
            Some(bind::Eng::Synth { name, .. }) | Some(bind::Eng::Plain { name, .. }) => name,
            _ => return Err(ModelError::W4a16(format!("{name}: Synth/Plain 아님"))),
        };
        let e = self
            .w4
            .entry(&hf)
            .ok_or_else(|| ModelError::MissingTensor(hf.clone()))?;
        if e.shape.len() == 1 {
            self.w4
                .plain_vec_f32(&hf)
                .map_err(|e| ModelError::W4a16(e.to_string()))
        } else {
            let rows = *e.shape.first().unwrap_or(&0);
            let raw = self
                .w4
                .raw_rows(&hf, 0, rows)
                .map_err(|e| ModelError::W4a16(e.to_string()))?;
            crate::w4a16::decode_f32(&raw, e.dtype, &hf)
                .map_err(|e| ModelError::W4a16(e.to_string()))
        }
    }

    fn perm_store(&self) -> &bind::PermStore {
        self.perm
            .get_or_init(|| bind::build_perm(&self.w4, &self.cfg))
    }

    pub fn wchk(&self, name: &str) -> Result<Weight<'_>, ModelError> {
        self.w(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        if let Some(v) = self.f32_cache.borrow().get(name) {
            return Ok(v.clone());
        }
        let w4 = &self.w4;
        let v = match bind::eng(name) {
            Some(bind::Eng::Synth {
                name: hf,
                plus1,
                neg_exp,
                perm,
            }) => {
                let mut v = if perm == 2 {
                    bind::conv_rows_f32_permuted(w4, &self.cfg, &hf)
                        .map_err(|e| ModelError::W4a16(e.to_string()))?
                } else {
                    let raw = w4
                        .plain_vec_f32(&hf)
                        .map_err(|e| ModelError::W4a16(e.to_string()))?;
                    if perm == 1 {
                        bind::permute_heads_f32(&self.cfg, &raw)
                    } else {
                        raw
                    }
                };
                if plus1 {
                    for x in &mut v {
                        *x += 1.0;
                    }
                }
                if neg_exp {
                    for x in &mut v {
                        *x = -x.exp();
                    }
                }
                v
            }
            _ => self.wchk(name)?.dequant_f32_vec(),
        };
        self.f32_cache
            .borrow_mut()
            .insert(name.to_string(), v.clone());
        Ok(v)
    }

    pub fn is_recr(&self, il: usize) -> bool {
        il % self.hp.full_attn_interval != self.hp.full_attn_interval - 1
    }
}

fn plain_weight<'a>(w4: &'a crate::w4a16::W4a16Model, hf: &str) -> Option<Weight<'a>> {
    use crate::wtype::WType;
    let e = w4.entry(hf)?;
    let ty = match e.dtype {
        crate::st::StDtype::Bf16 => WType::Bf16,
        crate::st::StDtype::F16 => WType::F16,
        crate::st::StDtype::F32 => WType::F32,
        _ => return None,
    };
    let n_out = *e.shape.first()?;
    let n_in = e.shape[1..].iter().product::<u64>();
    Some(Weight {
        data: w4.tensor_slice(hf)?,
        aux: None,
        ty,
        n_in,
        n_out,
        group: 0,
        scale_bf16: false,
    })
}

fn w4_weight_raw<'a>(w4: &'a crate::w4a16::W4a16Model, name: &str) -> Option<Weight<'a>> {
    use crate::wtype::WType;
    use bind::Eng;
    match bind::eng(name)? {
        Eng::Quant { base, .. } => {
            let Some((n, k)) = w4.lin_shape(&base) else {
                return plain_weight(w4, &format!("{base}.weight"));
            };
            Some(Weight {
                data: w4.tensor_slice(&format!("{base}.weight_packed"))?,
                aux: Some(w4.tensor_slice(&format!("{base}.weight_scale"))?),
                ty: WType::W4a16Split,
                n_in: k as u64,
                n_out: n as u64,
                group: w4.group(),
                scale_bf16: w4.scale_is_bf16(&base),
            })
        }
        Eng::Plain { name: hf, .. } => plain_weight(w4, &hf),
        Eng::Synth { .. } => None,
    }
}

fn w4_weight<'a>(m: &'a Model, name: &str) -> Option<Weight<'a>> {
    use crate::wtype::WType;
    use bind::{Eng, PV};
    let w4 = &m.w4;
    match bind::eng(name)? {
        Eng::Quant { base, vperm } => {
            let Some((n, k)) = w4.lin_shape(&base) else {
                let hf = format!("{base}.weight");
                return match vperm {
                    PV::None => plain_weight(w4, &hf),
                    _ => {
                        let e = w4.entry(&hf)?;
                        let ty = match e.dtype {
                            crate::st::StDtype::Bf16 => WType::Bf16,
                            crate::st::StDtype::F16 => WType::F16,
                            crate::st::StDtype::F32 => WType::F32,
                            _ => return None,
                        };
                        let data = m.perm_store().plain(&hf)?;
                        let n_out = *e.shape.first()?;
                        let n_in = e.shape[1..].iter().product::<u64>();
                        Some(Weight {
                            data,
                            aux: None,
                            ty,
                            n_in,
                            n_out,
                            group: 0,
                            scale_bf16: false,
                        })
                    }
                };
            };
            let (data, aux) = match vperm {
                PV::None => (
                    w4.tensor_slice(&format!("{base}.weight_packed"))?,
                    w4.tensor_slice(&format!("{base}.weight_scale"))?,
                ),
                _ => m.perm_store().quant(&base)?,
            };
            Some(Weight {
                data,
                aux: Some(aux),
                ty: WType::W4a16Split,
                n_in: k as u64,
                n_out: n as u64,
                group: w4.group(),
                scale_bf16: w4.scale_is_bf16(&base),
            })
        }
        Eng::Plain {
            name: hf,
            rows_perm,
        } => {
            let e = w4.entry(&hf)?;
            let ty = match e.dtype {
                crate::st::StDtype::Bf16 => WType::Bf16,
                crate::st::StDtype::F16 => WType::F16,
                crate::st::StDtype::F32 => WType::F32,
                _ => return None,
            };
            let data = if rows_perm {
                m.perm_store().plain(&hf)?
            } else {
                w4.tensor_slice(&hf)?
            };
            let n_out = *e.shape.first()?;
            let n_in = e.shape[1..].iter().product::<u64>();
            Some(Weight {
                data,
                aux: None,
                ty,
                n_in,
                n_out,
                group: 0,
                scale_bf16: false,
            })
        }
        Eng::Synth { .. } => None,
    }
}

pub struct SeqState {
    pub pos: u32,
    pub(crate) gdn_s: Vec<Vec<f32>>,
    pub(crate) conv: Vec<Vec<f32>>,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
}

impl SeqState {
    pub fn new(model: &Model, ctx: usize) -> Self {
        let n_full = (0..model.hp.n_layer)
            .filter(|&il| !model.is_recr(il))
            .count();
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

    pub fn reset_states(&mut self) {
        let n_kv = self.model.hp.n_kv;
        let hd = self.model.hp.head_dim;
        let ctx = self
            .seqs
            .first()
            .and_then(|s| s.kv_k.first().map(|k| k.len() / (n_kv * hd)))
            .unwrap_or(4096);
        for i in 0..self.seqs.len() {
            self.seqs[i] = SeqState::new(&self.model, ctx);
        }
    }

    pub fn reset_seq(&mut self, seq: usize) {
        let n_kv = self.model.hp.n_kv;
        let hd = self.model.hp.head_dim;
        let ctx = self.seqs[seq]
            .kv_k
            .first()
            .map(|k| k.len() / (n_kv * hd))
            .unwrap_or(4096);
        self.seqs[seq] = SeqState::new(&self.model, ctx);
    }

    fn forward(
        &mut self,
        seq_ids: &[usize],
        batch: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.forward_emb(seq_ids, batch, None)
    }

    fn forward_emb(
        &mut self,
        seq_ids: &[usize],
        batch: &[Vec<u32>],
        rows: Option<&[Vec<f32>]>,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("engine::forward");
        let n_seqs = batch.len();
        let t_len = batch.first().map(|v| v.len()).unwrap_or(0);
        assert!(n_seqs > 0 && t_len > 0);
        assert!(
            batch.iter().all(|v| v.len() == t_len),
            "배치 내 동일 길이 필수 (equal_seqs)"
        );

        let hp = self.model.hp.clone();
        let n_embd = hp.n_embd;
        let n_tok = n_seqs * t_len;

        let mut xs: Vec<Vec<f32>> = match rows {
            Some(r) if r.len() == n_tok => r.to_vec(),
            _ => {
                let embd = self.model.wchk("token_embd.weight")?;
                let mut v = Vec::with_capacity(n_tok);
                for seq_tokens in batch {
                    for &tok in seq_tokens {
                        let mut row = vec![0.0f32; n_embd];
                        dequant_row(embd.ty, embd.data, tok as u64, n_embd as u64, &mut row);
                        v.push(row);
                    }
                }
                v
            }
        };

        let mut full_idx = 0usize;
        let mut recr_idx = 0usize;
        for il in 0..hp.n_layer {
            let norm_w = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
            let residual: Vec<Vec<f32>> = xs.clone();
            for x in xs.iter_mut() {
                *x = rms_norm(x, &norm_w, hp.eps);
            }

            let ctx = stages::Ctx { model: &self.model };
            let attn_out = if self.model.is_recr(il) {
                let o = stages::gdn_layer(&ctx, &mut self.seqs, il, &xs, seq_ids, t_len, recr_idx)?;
                recr_idx += 1;
                o
            } else {
                let o =
                    stages::attn_layer(&ctx, &mut self.seqs, il, &xs, seq_ids, t_len, full_idx)?;
                full_idx += 1;
                o
            };

            for (t, a) in attn_out.iter().enumerate() {
                for i in 0..n_embd {
                    xs[t][i] = residual[t][i] + a[i];
                }
            }
            let ffn_residual = xs.clone();
            if llm170_diag::dump::opts().key("debug_layers") {
                let sum: f64 = xs[0].iter().map(|&v| v as f64).sum();
                eprintln!("  A{il} xs sum={sum:.6}");
            }

            let post_w = self
                .model
                .f32_vec(&format!("blk.{il}.post_attention_norm.weight"))?;
            if hp.n_experts > 0 {
                let mut normed: Vec<Vec<f32>> = vec![vec![0.0f32; n_embd]; n_tok];
                for (i, x) in xs.iter().enumerate() {
                    normed[i] = rms_norm(x, &post_w, hp.eps);
                }
                let mut ffn_out = vec![vec![0.0f32; n_embd]; n_tok];
                span_block!("cpu::moe", {
                    stages::moe_ffn(&ctx, il, &normed, &mut ffn_out)?;
                });
                for t in 0..n_tok {
                    for i in 0..n_embd {
                        xs[t][i] = ffn_residual[t][i] + ffn_out[t][i];
                    }
                }
                if llm170_diag::dump::opts().key("debug_layers") {
                    let m = xs[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let nan = xs[0].iter().any(|v| v.is_nan());
                    let v4: Vec<String> = xs[0][..4].iter().map(|v| format!("{v:.5}")).collect();
                    let sum: f64 = xs[0].iter().map(|&v| v as f64).sum();
                    eprintln!(
                        "layer {il:>2} recr={} moe max|x|={m:.4} nan={nan} head={} sum={sum:.6}",
                        self.model.is_recr(il),
                        v4.join(","),
                    );
                }
                continue;
            }
            let gate_w = self.model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
            let up_w = self.model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
            let down_w = self.model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
            let n_ff = hp.n_ff;

            let mut normed: Vec<Vec<f32>> = vec![vec![0.0f32; hp.n_embd]; n_tok];
            for (i, x) in xs.iter().enumerate() {
                normed[i] = rms_norm(x, &post_w, hp.eps);
            }
            let mut ffn_group: [Vec<Vec<f32>>; 2] = [
                vec![vec![0.0f32; n_ff]; n_tok],
                vec![vec![0.0f32; n_ff]; n_tok],
            ];
            {
                span_block!("cpu::ffn_gate_up", {
                    mm_group(&normed, &[gate_w, up_w], &mut ffn_group);
                });
            }
            let [mut gate_y, up_y] = ffn_group;
            for t in 0..n_tok {
                for i in 0..n_ff {
                    gate_y[t][i] = silu(gate_y[t][i]) * up_y[t][i];
                }
            }
            {
                span_block!("cpu::ffn_down", {
                    mm_batch(&gate_y, &down_w, &mut xs);
                });
            }
            for t in 0..n_tok {
                for i in 0..n_embd {
                    xs[t][i] += ffn_residual[t][i];
                }
            }
            if llm170_diag::dump::opts().key("debug_layers") {
                let m = xs[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = xs[0].iter().any(|v| v.is_nan());
                let v4: Vec<String> = xs[0][..4].iter().map(|v| format!("{v:.5}")).collect();
                let sum: f64 = xs[0].iter().map(|&v| v as f64).sum();
                eprintln!(
                    "layer {il:>2} recr={} max|x|={m:.4} nan={nan} head={} sum={sum:.6}",
                    self.model.is_recr(il),
                    v4.join(","),
                );
            }
        }

        let out_norm = self.model.f32_vec("output_norm.weight")?;
        let head = self.model.wchk("output.weight")?;
        let mut result = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let last = &xs[(s + 1) * t_len - 1];
            let h = rms_norm(last, &out_norm, hp.eps);
            let mut logits = vec![0.0f32; head.n_out as usize];
            mm(&h, &head, &mut logits);
            if llm170_diag::dump::opts().key("debug_layers") {
                let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = logits.iter().any(|v| v.is_nan());
                eprintln!("logits: max={m:.4} nan={nan} argmax={}", greedy(&logits));
            }
            result.push(logits);
        }
        Ok(result)
    }

    pub fn decode(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        let batch: Vec<Vec<u32>> = tokens.iter().map(|t| vec![*t]).collect();
        let logits = self.forward(seq_ids, &batch)?;
        for s in seq_ids {
            self.seqs[*s].pos += 1;
        }
        Ok(logits)
    }

    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, ModelError> {
        let logits = self.decode(seq_ids, tokens)?;
        Ok(logits.iter().map(|l| greedy(l)).collect())
    }

    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, ModelError> {
        let logits = self.decode(&[seq], &[token])?;
        Ok(crate::qwen35::greedy(&logits[0]))
    }

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

pub use crate::matmul::greedy_from as greedy;

pub struct CtxInfo {
    pub ctx: usize,
    pub gdn_state_bytes_per_slot: u64,
    pub kv_bytes_per_token: u64,
    pub dense: bool,
}

pub fn ctx_info(dir: &std::path::Path) -> Result<CtxInfo, String> {
    let c = bind::QwenCfg::load(dir).map_err(|e| e.to_string())?;
    let full = (c.layers as u64).div_ceil(c.full_interval.max(1) as u64);
    let kv = full * c.kv_heads as u64 * c.head_dim as u64 * 2;
    let sc = full * c.kv_heads as u64 * 2 * 4;
    Ok(CtxInfo {
        ctx: c.max_pos,
        gdn_state_bytes_per_slot: {
            let n_gdn = c.layers as u64 - c.layers as u64 / c.full_interval.max(1) as u64;
            let hv = c.linear_num_value_heads as u64;
            n_gdn * hv * 128 * 128 * 2 + n_gdn * hv * 128 * 4
        },
        kv_bytes_per_token: kv + sc,
        dense: c.n_experts == 0,
    })
}
