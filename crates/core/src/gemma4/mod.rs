//! gemma4 (Gemma-4-26B-A4B) CPU 참조 엔진 — qwen35 미러·병렬(trait 추출 없음).
//!
//! 그래프(2026-10-14 HF `modeling_gemma4.py` 대조):
//! - 30층 = 25 sliding(SW 1024·Q16×256·KV8·θ1e4) + 5 full(5/11/17/23/29 —
//!   Q16×512·KV2·k_eq_v(V:=v_norm(k_proj))) · 층별 dims라 rope도 2종.
//! - 층 = `attn → post_attn_norm → +r` → dense MLP(pre_ff)와 MoE(pre_ff_2)를
//!   **병렬** 계산 → `post_ff_norm1(m) + post_ff_norm2(e)` → post_ff_norm → +r
//!   → **×layer_scalar**(전 층).
//! - RMSNorm = 직접 승수(+1 아님) · 노름류는 f32 · 라우터 = 가중 없는 RMS.
//! - embed = E·√h(fp32 곱 → BF16 라운드 — 입력 임베딩만. head는 원본 E).
//! - head tied(텐서 없음) · logits = softcap(30·tanh(l/30)).

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
    /// W4A16 디렉터리 로더 오류.
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

/// 가중치 소스 — W4A16 디렉터리(스토어 mmap 직결).
pub struct Model {
    w4: Box<crate::w4a16::W4a16Model>,
    cfg: bind::GemmaCfg,
    pub hp: Hparams,
    pub token_pieces: Vec<String>,
    /// f32 노름·스칼라 디양자화 캐시(1회).
    f32_cache: std::cell::RefCell<std::collections::HashMap<String, Vec<f32>>>,
    /// 입력 임베딩 √h(곱은 f32, 저장은 BF16 라운드 — CPU·GPU 동일 바이트).
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

    /// 풀층 k_eq_v — v_proj 없이 v = v_norm(k_proj 원시).
    pub fn kv_eq_v(&self, il: usize) -> bool {
        self.cfg.kv_eq_v(il)
    }

    /// MoE 전문가 슬라이스 — (packed ptr/len, scale ptr/len) × (il,e,proj).
    /// qwen35 `expert_slice` 미러(업로드 테이블 조립용).
    pub fn expert_slice(
        &self,
        il: usize,
        e: usize,
        proj: &str,
    ) -> Option<(&[u8], &[u8], usize, usize)> {
        self.w4.expert_at(&bind::expert_base(il, e, proj))
    }

    /// MoE 전문가 양자 파라미터 — (group, scale_bf16) 실측.
    pub fn expert_quant(&self) -> (usize, bool) {
        let base = bind::expert_base(0, 0, "gate_proj");
        (self.w4.group(), self.w4.scale_is_bf16(&base))
    }

    /// 엔진 접점 이름 전수(업로드·체인 배선용) — qwen35 engine_names 미러.
    /// 층 접미사는 gemma4 바인딩(`bind::eng`)과 1:1.
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

    /// 임베딩 √h 스케일(병렬 프리필용).
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

    /// f32 벡터(노름·스칼라·라우터 가중) — 1회 디양자화 캐시.
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

    /// 입력 임베딩 행 — E(tok)·√h. 곱은 f32, **BF16 라운드 저장**(베이크) —
    /// GPU 업로드 테이블과 동일 바이트 계약.
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, ModelError> {
        let (buf, cols) = self.embed_table()?;
        let mut v = Vec::with_capacity(cols);
        bake_embed_row(buf, cols, tok, self.embed_scale, &mut v);
        Ok(v)
    }

    /// 임베딩 원시 뷰 — (bf16 바이트, 열 수). 병렬 프리필이 Model 비경유로 쓴다.
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

    /// 전문가 트리플 슬라이스 → Weight (`expert_at` 무경유 접근자).
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

/// 임베딩 원시 행(bf16) → √h 베이크 f32 행 — 스레드 안전(Model 비경유).
/// 곱은 f32, 저장은 BF16 라운드(CPU·GPU 동일 바이트).
pub fn bake_embed_row(buf: &[u8], cols: usize, tok: u32, scale: f32, out: &mut Vec<f32>) {
    out.clear();
    let off = tok as usize * cols * 2;
    let row = &buf[off..off + cols * 2];
    for c in row.as_chunks::<2>().0 {
        let f = crate::quant::deq::bf16_to_f32(u16::from_le_bytes(*c));
        out.push(crate::ops::bf16_rne(f * scale));
    }
}

/// 플레인(BF16) Weight — 2D 텐서(라우터·head). 1D는 `f32_vec` 전용.
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

/// 시퀀스 상태 — 전 층 KV(f32 참조).
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

    /// 시퀀스 상태 전체 초기화 — ctx는 기존 KV 용량에서 역산.
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

    /// 슬롯 단위 초기화(연속 배칭).
    pub fn reset_seq(&mut self, seq: usize) {
        let hp = &self.model.hp;
        let ctx = self.seqs[seq].kv_k[0].len() / (hp.n_kv[0] * hp.head_dim[0]);
        self.seqs[seq] = SeqState::new(&self.model, ctx);
    }

    /// 배치 포워드 — batch[i] = seq_ids[i]의 토큰열(행간 동일 길이).
    pub fn forward(
        &mut self,
        seq_ids: &[usize],
        batch: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.forward_emb(seq_ids, batch, None)
    }

    /// forward 변형 — 임베딩 행 사전 조달(프리필).
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
            // ── 어텐션 블록 ──
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
            // ── FFN 병렬(dense + MoE) ──
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
        // ── 최종 노름 + head(tied) + softcap ──
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

    /// np 배치 greedy — 토큰만 회수.
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

    /// 표시용 근사 디토크(정식 BPE 디토크는 tokenize.rs 소관).
    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(|s| s.replace('▁', " "))
            .unwrap_or_default()
    }
}

/// greedy argmax — 동률 최저인덱스(공용 계약).
pub use crate::matmul::greedy_from as greedy;

/// [P3 2026-10-14] 서빙 계획 파생 — **층별 KV 일반화**(이종 dims: 슬라이딩
/// kv_local×head_dim_local·풀 kv_global×head_dim_global). kv_bpt = int8
/// K+V(1B/원소) + 행×헤드 스케일 f32(4B). 순수 어텐션이므로 GDN 상태 0.
pub fn ctx_info(dir: &std::path::Path) -> Result<crate::qwen35::CtxInfo, String> {
    let cfg = bind::GemmaCfg::load(dir).map_err(|e| e.to_string())?;
    let n_full = cfg.layer_full.iter().filter(|&&f| f).count() as u64;
    let n_slide = cfg.layers as u64 - n_full;
    let kv = n_slide * (cfg.kv_local as u64 * cfg.head_dim_local as u64) * 2
        + n_full * (cfg.kv_global as u64 * cfg.head_dim_global as u64) * 2;
    let sc = n_slide * cfg.kv_local as u64 * 4 + n_full * cfg.kv_global as u64 * 4;
    Ok(crate::qwen35::CtxInfo {
        ctx: cfg.max_pos,
        gdn_state_bytes_per_slot: 0,
        kv_bytes_per_token: kv + sc,
        dense: !cfg.moe,
    })
}
