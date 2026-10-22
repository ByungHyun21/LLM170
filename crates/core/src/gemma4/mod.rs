
pub mod bind;
pub mod hparams;
pub mod prefill;
mod stages;

use hparams::Hparams;
use std::path::Path;

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

use crate::matmul::Weight;

pub struct Model {
    w4: Box<crate::w4a16::W4a16Model>,
    cfg: bind::GemmaCfg,
    pub hp: Hparams,
    pub token_pieces: Vec<String>,
    f32_cache: std::cell::RefCell<std::collections::HashMap<String, Vec<f32>>>,
    embed_scale: f32,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self, ModelError> {
        let w4 =
            crate::w4a16::W4a16Model::open(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let cfg = bind::GemmaCfg::load(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let rep = bind::validate(&w4, &cfg).map_err(|e| ModelError::W4a16(e.to_string()))?;
        if !rep.ok() {
            return Err(ModelError::W4a16(format!(
                "커버리지 검증 실패:\n{}",
                rep.summary()
            )));
        }
        let c = &cfg;
        let hp = Hparams {
            n_layer: c.layers,
            n_embd: c.hidden,
            n_ff: c.ffn,
            n_head: c.heads,
            vocab: c.vocab,
            eps: c.eps as f32,
            sw: c.sw,
            full: c.layer_full.clone(),
            head_dim: (0..c.layers).map(|il| c.head_dim(il)).collect(),
            n_kv: (0..c.layers).map(|il| c.n_kv(il)).collect(),
            n_rot: (0..c.layers).map(|il| c.n_rot(il)).collect(),
            rope_base: (0..c.layers).map(|il| c.rope_base(il)).collect(),
            n_experts: if c.moe { c.n_experts } else { 0 },
            top_k: c.top_k,
            moe_ffn: c.moe_ffn,
            softcap: c.softcap as f32,
            max_pos: c.max_pos,
        };
        let token_pieces =
            crate::w4a16::load_pieces(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let embed_scale = (c.hidden as f32).sqrt();
        let m = Model {
            w4: Box::new(w4),
            cfg,
            hp,
            token_pieces,
            f32_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            embed_scale,
        };
        for name in ["token_embd.weight", "output.weight"] {
            m.w(name).ok_or(ModelError::MissingTensor(name.into()))?;
        }
        m.f32_vec("output_norm.weight")?;
        Ok(m)
    }

    pub fn kv_eq_v(&self, il: usize) -> bool {
        self.cfg.kv_eq_v(il)
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

    pub fn engine_names(&self) -> Vec<String> {
        let mut v = vec![
            "token_embd.weight".to_string(),
            "output.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        for il in 0..self.hp.n_layer {
            for suf in [
                "attn_norm.weight",
                "post_attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "pre_ff_norm.weight",
                "pre_ff_norm2.weight",
                "post_ff_norm.weight",
                "post_ff_norm1.weight",
                "post_ff_norm2.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "moe_gate.weight",
                "moe_scale",
                "moe_per_expert_scale",
                "layer_scalar",
            ] {
                v.push(format!("blk.{il}.{suf}"));
            }
        }
        v
    }

    pub fn embed_scale(&self) -> f32 {
        self.embed_scale
    }

    pub fn w(&self, name: &str) -> Option<Weight<'_>> {
        use bind::Eng;
        let w4 = &self.w4;
        match bind::eng(name)? {
            Eng::Quant { base } => Some(Weight {
                data: w4.tensor_slice(&format!("{base}.weight_packed"))?,
                aux: Some(w4.tensor_slice(&format!("{base}.weight_scale"))?),
                ty: crate::wtype::WType::W4a16Split,
                n_in: w4.lin_shape(&base)?.1 as u64,
                n_out: w4.lin_shape(&base)?.0 as u64,
                group: w4.group(),
                scale_bf16: w4.scale_is_bf16(&base),
            }),
            Eng::Plain { name: hf } => plain_weight(w4, &hf),
        }
    }

    pub fn wchk(&self, name: &str) -> Result<Weight<'_>, ModelError> {
        self.w(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))
    }

    pub fn store(&self) -> &crate::w4a16::W4a16Model {
        &self.w4
    }

    pub fn w_names(&self, name: &str) -> Option<(String, Option<String>)> {
        use bind::Eng;
        match bind::eng(name)? {
            Eng::Quant { base } => Some((
                format!("{base}.weight_packed"),
                Some(format!("{base}.weight_scale")),
            )),
            Eng::Plain { name: hf } => Some((hf, None)),
        }
    }

    pub fn expert_names(&self, il: usize, e: usize, proj: &str) -> (String, String) {
        let base = bind::expert_base(il, e, proj);
        (
            format!("{base}.weight_packed"),
            format!("{base}.weight_scale"),
        )
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        if let Some(v) = self.f32_cache.borrow().get(name) {
            return Ok(v.clone());
        }
        let hf = match bind::eng(name) {
            Some(bind::Eng::Plain { name }) => name,
            _ => return Err(ModelError::W4a16(format!("{name}: Plain 아님"))),
        };
        let e = self
            .w4
            .entry(&hf)
            .ok_or_else(|| ModelError::MissingTensor(hf.clone()))?;
        let v = if e.shape.len() == 1 {
            self.w4
                .plain_vec_f32(&hf)
                .map_err(|err| ModelError::W4a16(err.to_string()))?
        } else {
            self.wchk(name)?.dequant_f32_vec()
        };
        self.f32_cache
            .borrow_mut()
            .insert(name.to_string(), v.clone());
        Ok(v)
    }

    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, ModelError> {
        let (buf, cols) = self.embed_table()?;
        let mut v = Vec::with_capacity(cols);
        bake_embed_row(buf, cols, tok, self.embed_scale, &mut v);
        Ok(v)
    }

    pub fn embed_table(&self) -> Result<(&[u8], usize), ModelError> {
        let name = "model.language_model.embed_tokens.weight";
        let e = self
            .w4
            .entry(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let buf = self
            .w4
            .tensor_slice(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        Ok((buf, e.shape[1] as usize))
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
}

pub fn bake_embed_row(buf: &[u8], cols: usize, tok: u32, scale: f32, out: &mut Vec<f32>) {
    out.clear();
    let off = tok as usize * cols * 2;
    let row = &buf[off..off + cols * 2];
    for c in row.as_chunks::<2>().0 {
        let f = crate::quant::deq::bf16_to_f32(u16::from_le_bytes(*c));
        out.push(crate::ops::bf16_rne(f * scale));
    }
}

fn plain_weight<'a>(w4: &'a crate::w4a16::W4a16Model, hf: &str) -> Option<Weight<'a>> {
    use crate::wtype::WType;
    let e = w4.entry(hf)?;
    if e.shape.len() < 2 {
        return None;
    }
    let ty = match e.dtype {
        crate::st::StDtype::Bf16 => WType::Bf16,
        crate::st::StDtype::F16 => WType::F16,
        crate::st::StDtype::F32 => WType::F32,
        _ => return None,
    };
    Some(Weight {
        data: w4.tensor_slice(hf)?,
        aux: None,
        ty,
        n_in: e.shape[1..].iter().product::<u64>(),
        n_out: e.shape[0],
        group: 0,
        scale_bf16: false,
    })
}

pub struct SeqState {
    pub pos: u32,
    pub(crate) kv_k: Vec<Vec<f32>>,
    pub(crate) kv_v: Vec<Vec<f32>>,
}

impl SeqState {
    pub fn new(model: &Model, ctx: usize) -> Self {
        let hp = &model.hp;
        let mk = || {
            (0..hp.n_layer)
                .map(|il| vec![0.0f32; ctx * hp.n_kv[il] * hp.head_dim[il]])
                .collect()
        };
        SeqState {
            pos: 0,
            kv_k: mk(),
            kv_v: mk(),
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
        let ctx = self
            .seqs
            .first()
            .map(|s| s.kv_k[0].len() / (self.model.hp.n_kv[0] * self.model.hp.head_dim[0]))
            .unwrap_or(4096);
        for i in 0..self.seqs.len() {
            self.seqs[i] = SeqState::new(&self.model, ctx);
        }
    }

    pub fn reset_seq(&mut self, seq: usize) {
        let hp = &self.model.hp;
        let ctx = self.seqs[seq].kv_k[0].len() / (hp.n_kv[0] * hp.head_dim[0]);
        self.seqs[seq] = SeqState::new(&self.model, ctx);
    }

    pub fn forward(
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
        let n_seqs = batch.len();
        let t_len = batch.first().map(|v| v.len()).unwrap_or(0);
        assert!(n_seqs > 0 && t_len > 0);
        assert!(
            batch.iter().all(|v| v.len() == t_len),
            "배치 내 동일 길이 필수"
        );
        let hp = self.model.hp.clone();
        let n_embd = hp.n_embd;
        let n_tok = n_seqs * t_len;
        let mut xs: Vec<Vec<f32>> = match rows {
            Some(r) if r.len() == n_tok => r.to_vec(),
            _ => {
                let mut v = Vec::with_capacity(n_tok);
                for seq_tokens in batch {
                    for &tok in seq_tokens {
                        v.push(self.model.embed_row(tok)?);
                    }
                }
                v
            }
        };
        for il in 0..hp.n_layer {
            let residual: Vec<Vec<f32>> = xs.clone();
            let attn_norm = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
            for x in xs.iter_mut() {
                *x = crate::ops::rms_norm(x, &attn_norm, hp.eps);
            }
            let attn_out =
                stages::attn_layer(&self.model, &mut self.seqs, il, &xs, seq_ids, t_len)?;
            let post_attn = self
                .model
                .f32_vec(&format!("blk.{il}.post_attn_norm.weight"))?;
            for t in 0..n_tok {
                let h = crate::ops::rms_norm(&attn_out[t], &post_attn, hp.eps);
                for i in 0..n_embd {
                    xs[t][i] = residual[t][i] + h[i];
                }
            }
            let residual2: Vec<Vec<f32>> = xs.clone();
            let pre_ff = self
                .model
                .f32_vec(&format!("blk.{il}.pre_ff_norm.weight"))?;
            let n_dense: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| crate::ops::rms_norm(&residual2[t], &pre_ff, hp.eps))
                .collect();
            let mut m = vec![vec![0.0f32; n_embd]; n_tok];
            stages::dense_mlp(&self.model, il, &n_dense, &mut m)?;
            let post_ff1 = self
                .model
                .f32_vec(&format!("blk.{il}.post_ff_norm1.weight"))?;
            let h1: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| crate::ops::rms_norm(&m[t], &post_ff1, hp.eps))
                .collect();
            let sel = stages::route(&self.model, il, &residual2)?;
            let pre_ff2 = self
                .model
                .f32_vec(&format!("blk.{il}.pre_ff_norm2.weight"))?;
            let n_moe: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| crate::ops::rms_norm(&residual2[t], &pre_ff2, hp.eps))
                .collect();
            let mut e_out = vec![vec![0.0f32; n_embd]; n_tok];
            stages::moe_block(&self.model, il, &n_moe, &sel, &mut e_out)?;
            let post_ff2 = self
                .model
                .f32_vec(&format!("blk.{il}.post_ff_norm2.weight"))?;
            let h2: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| crate::ops::rms_norm(&e_out[t], &post_ff2, hp.eps))
                .collect();
            let post_ff = self
                .model
                .f32_vec(&format!("blk.{il}.post_ff_norm.weight"))?;
            let ls = self.model.f32_vec(&format!("blk.{il}.layer_scalar"))?[0];
            let mut merged = vec![0.0f32; n_embd];
            for t in 0..n_tok {
                for i in 0..n_embd {
                    merged[i] = h1[t][i] + h2[t][i];
                }
                let h = crate::ops::rms_norm(&merged, &post_ff, hp.eps);
                for i in 0..n_embd {
                    xs[t][i] = (residual2[t][i] + h[i]) * ls;
                }
            }
            if llm170_diag::dump::opts().key("debug_layers") {
                let mx = xs[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = xs[0].iter().any(|v| v.is_nan());
                eprintln!("gemma4 layer {il:>2} max|x|={mx:.4} nan={nan}");
            }
        }
        let out_norm = self.model.f32_vec("output_norm.weight")?;
        let head = self.model.wchk("output.weight")?;
        let mut result = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let last = &xs[(s + 1) * t_len - 1];
            let h = crate::ops::rms_norm(last, &out_norm, hp.eps);
            let mut logits = vec![0.0f32; head.n_out as usize];
            crate::matmul::cpu::matmul(&h, &head, &mut logits);
            for l in logits.iter_mut() {
                *l = crate::ops::logit_softcap(*l, hp.softcap);
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
        Ok(greedy(&logits[0]))
    }

    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(|s| s.replace('▁', " "))
            .unwrap_or_default()
    }
}

pub use crate::matmul::greedy_from as greedy;

pub fn ctx_info(dir: &std::path::Path) -> Result<crate::qwen35::CtxInfo, String> {
    let cfg = bind::GemmaCfg::load(dir).map_err(|e| e.to_string())?;
    Ok(ctx_info_from_cfg(&cfg))
}

pub fn ctx_info_from_cfg(cfg: &bind::GemmaCfg) -> crate::qwen35::CtxInfo {
    let kvd = (cfg.kv_local as u64 * cfg.head_dim_local as u64)
        .max(cfg.kv_global as u64 * cfg.head_dim_global as u64);
    let kvh = (cfg.kv_local as u64).max(cfg.kv_global as u64);
    let per_layer = 2 * (kvd + kvh * 4);
    crate::qwen35::CtxInfo {
        ctx: cfg.max_pos,
        gdn_state_bytes_per_slot: 0,
        kv_bytes_per_token: cfg.layers as u64 * per_layer,
        dense: !cfg.moe,
    }
}
