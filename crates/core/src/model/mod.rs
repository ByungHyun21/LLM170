//! qwen35 (Qwen3.8-27B 계열) CPU 참조 엔진.
//!
//! 그래프 배선: `~/local_llm/llama.cpp/src/models/qwen35.cpp` (2026-08-30 판).
//! - 잔차: h += attn(rms(h)); h += ffn(rms_post(h))
//! - GDN층(interval≠3): qkv → depthwise conv+SiLU → L2 norm(q,k) → GDN → rms_norm·silu(z) → ssm_out
//! - Full-attn층(interval==3): q‖gate fused → per-head rms norm(q,k) → RoPE(n_rot, base, 인접 페어)
//!   → GQA(scale 1/√head_dim) → ⊙sigmoid(gate) → wo
//! - 하이퍼파라미터는 GGUF 메타에서 동적 로드 (소형 검증 모델 지원).
//! - f32 KV, f32 GDN 상태 (참조 정확도 우선).

pub mod hparams;
mod layers;
pub(crate) mod frame35;
pub use frame35::Frame35;

use hparams::Hparams;
use llm170_gguf::GgufFile;
use llm170_profiler::profile_span;
use memmap2::Mmap;

use crate::matmul::{Weight, mm, mm_batch, mm_group};
use crate::ops::{rms_norm, silu};
use crate::quant::dequant_row;
#[derive(Debug)]
pub enum ModelError {
    MissingTensor(String),
    UnsupportedLayout { name: String, why: &'static str },
    BadHparam(&'static str),
    Accel(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ModelError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            ModelError::UnsupportedLayout { name, why } => {
                write!(f, "unsupported layout {name}: {why}")
            }
            ModelError::BadHparam(w) => write!(f, "bad hyperparameter: {w}"),
            ModelError::Accel(e) => write!(f, "accelerator: {e}"),
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
        let n_embd =
            u("embedding_length").ok_or(ModelError::BadHparam("embedding_length"))? as usize;
        let block_count = u("block_count").unwrap_or(64) as usize;
        let head_count =
            u("attention.head_count").ok_or(ModelError::BadHparam("head_count"))? as usize;
        let head_dim =
            u("attention.key_length").ok_or(ModelError::BadHparam("key_length"))? as usize;
        let d_state = u("ssm.state_size").ok_or(ModelError::BadHparam("ssm.state_size"))? as usize;
        let n_group =
            u("ssm.group_count").ok_or(ModelError::BadHparam("ssm.group_count"))? as usize;
        let dt_rank =
            u("ssm.time_step_rank").ok_or(ModelError::BadHparam("ssm.time_step_rank"))? as usize;
        let d_inner = u("ssm.inner_size").ok_or(ModelError::BadHparam("ssm.inner_size"))? as usize;

        let hp = Hparams {
            n_layer: block_count.min(64), // block_count=65 → 64본체 + MTP(그래프 외)
            n_embd,
            n_ff: u("feed_forward_length").ok_or(ModelError::BadHparam("feed_forward_length"))?
                as usize,
            n_head: head_count,
            n_kv: u("attention.head_count_kv").unwrap_or(head_count as u64) as usize,
            head_dim,
            n_rot: u("rope.dimension_count").unwrap_or(head_dim as u64) as usize,
            rope_base: gguf
                .arch_kv("rope.freq_base")
                .and_then(llm170_gguf::Value::as_f64)
                .unwrap_or(1e7) as f32,
            eps: gguf
                .arch_kv("attention.layer_norm_rms_epsilon")
                .and_then(llm170_gguf::Value::as_f64)
                .unwrap_or(1e-6) as f32,
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

        let vocab = gguf
            .find_tensor("token_embd.weight")
            .map(|t| t.ne[1])
            .unwrap_or(0) as usize;
        let hp = Hparams { vocab, ..hp };

        let m = Model {
            gguf,
            mmap,
            hp,
            token_pieces,
        };
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

    pub fn wchk(&self, name: &str) -> Result<Weight<'_>, ModelError> {
        self.w(name)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        Ok(self.wchk(name)?.dequant_f32_vec())
    }

    pub fn is_recr(&self, il: usize) -> bool {
        il % self.hp.full_attn_interval != self.hp.full_attn_interval - 1
    }
}

/// 시퀀스 상태
pub struct SeqState {
    pub pos: u32,
    pub(crate) gdn_s: Vec<Vec<f32>>,
    pub(crate) conv: Vec<Vec<f32>>,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    /// MTP draft층 KV (blk.64 full-attn 1층분) — nextn 미탑재 모델은 빈 벡터.
    mtp_kv_k: Vec<f32>,
    mtp_kv_v: Vec<f32>,
    /// MTP draft h 입력 — 직전 확정 토큰 t의 본체 hidden (h_t).
    pub mtp_h: Vec<f32>,
}

impl SeqState {
    /// 프레임 KV 동기용 읽기 접근 (2026-09-02 P1).
    pub(crate) fn kv_k_ref(&self) -> &[Vec<f32>] {
        &self.kv_k
    }
    pub(crate) fn kv_v_ref(&self) -> &[Vec<f32>] {
        &self.kv_v
    }
    pub fn new(model: &Model, ctx: usize) -> Self {
        let n_full = (0..model.hp.n_layer)
            .filter(|&il| !model.is_recr(il))
            .count();
        let n_recr = model.hp.n_layer - n_full;
        let (n_kv, hd) = (model.hp.n_kv, model.hp.head_dim);
        let state_size = model.hp.dt_rank * model.hp.d_state * model.hp.d_state;
        let conv_len = (model.hp.conv_k - 1) * model.hp.conv_ch();
        let has_mtp = model.gguf.find_tensor("blk.64.nextn.eh_proj.weight").is_some();
        SeqState {
            pos: 0,
            kv_k: vec![vec![0.0; ctx * n_kv * hd]; n_full],
            kv_v: vec![vec![0.0; ctx * n_kv * hd]; n_full],
            gdn_s: vec![vec![0.0; state_size]; n_recr],
            conv: vec![vec![0.0; conv_len]; n_recr],
            mtp_kv_k: vec![0.0; if has_mtp { ctx * n_kv * hd } else { 0 }],
            mtp_kv_v: vec![0.0; if has_mtp { ctx * n_kv * hd } else { 0 }],
            mtp_h: vec![0.0; if has_mtp { model.hp.n_embd } else { 0 }],
        }
    }
}
pub struct Engine {
    pub model: Model,
    pub seqs: Vec<SeqState>,
    /// 런타임 주입 가속기 (None = CPU 참조 경로). 구현은 backend-gpu.
    pub acc: crate::matmul::Acc,
    /// qwen35 디코드 프레임 (t=1) — LLM170_FRAME35=1 첫 디코드에서 생성.
    pub raw_decode: Option<std::sync::Arc<dyn crate::matmul::RawDecode>>,
    pub frame35: Option<Frame35>,
    /// 시퀀스별 프레임 상태 유효 플래그 — 값 경로 실행(prefill 등)마다 무효화.
    pub(crate) frame35_clean: Vec<bool>,
}

impl Engine {
    /// MTP(nextn) 텐서 탑재 여부 — --spec 사용 가능 판정.
    pub fn has_mtp(&self) -> bool {
        !self.seqs.first().map(|s| s.mtp_h.is_empty()).unwrap_or(true)
    }

    pub fn new(model: Model, n_seqs: usize, ctx: usize) -> Self {
        let seqs = (0..n_seqs).map(|_| SeqState::new(&model, ctx)).collect();
        Engine {
            raw_decode: None,
            frame35: None,
            frame35_clean: vec![false; n_seqs],
            model,
            seqs,
            acc: None,
        }
    }

    /// 시퀀스 상태 전체 초기화 (무상태 HTTP 서버용) — mmap은 유지.
    /// ctx는 기존 KV 용량에서 역산 (첫 kv_k 길이).
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

    /// 슬롯 단위 상태 초기화 (연속 배칭 서버 — 04). 해당 시퀀스의
    /// KV/GDN/conv 상태만 영점화, 다른 슬롯은 간섭 없음.
    pub fn reset_seq(&mut self, seq: usize) {
        let n_kv = self.model.hp.n_kv;
        let hd = self.model.hp.head_dim;
        let ctx = self.seqs[seq].kv_k.first().map(|k| k.len() / (n_kv * hd)).unwrap_or(4096);
        self.seqs[seq] = SeqState::new(&self.model, ctx);
    }

    /// 가속기 주입 (server --backend gpu).
    pub fn with_acc(mut self, acc: std::sync::Arc<dyn crate::matmul::Accelerator>) -> Self {
        self.acc = Some(acc);
        self
    }

    /// 배치 포워드: batch[i] = seq_ids[i] 시퀀스의 토큰(행간 동일 길이).
    /// seq_ids: 배치 행 → 엔진 시퀀스 id 매핑 (prefill은 단일, decode는 활성 집합).
    fn forward(
        &mut self,
        seq_ids: &[usize],
        batch: &[Vec<u32>],
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
        // 값 경로 실행 — 프레임 GPU 상태는 CPU 상태와 어긋나 무효화.
        for s in seq_ids {
            self.frame35_clean[*s] = false;
        }
        // 가속기 아크 복제 — self 차입 충돌 없이 층 내부까지 전달
        let acc = self.acc.clone();
        for il in 0..hp.n_layer {
            let norm_w = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
            // 잔차: pre-norm 원본 보존 (qwen35.cpp:162-184 — inpSA)
            let residual: Vec<Vec<f32>> = xs.clone();
            for x in xs.iter_mut() {
                *x = rms_norm(x, &norm_w, hp.eps);
            }

            let attn_out = if self.model.is_recr(il) {
                let o = self.gdn_layer(il, &xs, seq_ids, t_len, recr_idx)?;
                recr_idx += 1;
                o
            } else {
                let o = self.attn_layer(il, &xs, seq_ids, t_len, full_idx)?;
                full_idx += 1;
                o
            };

            for (t, a) in attn_out.iter().enumerate() {
                for i in 0..n_embd {
                    xs[t][i] = residual[t][i] + a[i];
                }
            }
            let ffn_residual = xs.clone();
            if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                let sum: f64 = xs[0].iter().map(|&v| v as f64).sum();
                eprintln!("  A{il} xs sum={sum:.6}");
            }

            let post_w = self
                .model
                .f32_vec(&format!("blk.{il}.post_attention_norm.weight"))?;
            let gate_w = self.model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
            let up_w = self.model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
            let down_w = self.model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
            let n_ff = hp.n_ff;

            let normed: Vec<Vec<f32>> =
                xs.iter().map(|x| rms_norm(x, &post_w, hp.eps)).collect();
            let mut ffn_group: [Vec<Vec<f32>>; 2] =
                [vec![vec![0.0f32; n_ff]; n_tok], vec![vec![0.0f32; n_ff]; n_tok]];
            {
                span_block!("cpu::ffn_gate_up", {
                    mm_group(&acc, &normed, &[gate_w, up_w], &mut ffn_group)?;
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
                    mm_batch(&acc, &gate_y, &down_w, &mut xs)?;
                });
            }
            let _ = &gate_y;
            for t in 0..n_tok {
                for i in 0..n_embd {
                    xs[t][i] += ffn_residual[t][i];
                }
            }
            if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
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

        // output_norm + logits (시퀀스별 마지막 토큰만)
        let out_norm = self.model.f32_vec("output_norm.weight")?;
        let head = self.model.wchk("output.weight")?;
        let mut result = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let last = &xs[(s + 1) * t_len - 1];
            // MTP draft용 h_t 스냅샷 (06) — 본체 hidden을 시퀀스 상태에 보관.
            if !self.seqs[seq_ids[s]].mtp_h.is_empty() {
                self.seqs[seq_ids[s]].mtp_h.copy_from_slice(last);
            }
            let h = rms_norm(last, &out_norm, hp.eps);
            let mut logits = vec![0.0f32; head.n_out as usize];
            mm(&acc, &h, &head, &mut logits)?;
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
    pub fn decode(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        // 원시 HIP 디코드 (t=1 단일) — LLM170_RAWHIP=1, 최우선 게이트.
        if tokens.len() == 1
            && seq_ids.len() == 1
            && self.raw_decode.is_some()
            && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
        {
            let seq = seq_ids[0];
            let token = tokens[0];
            let n = self.model.hp.n_embd;
            let embd = self.model.wchk("token_embd.weight")?;
            let mut row = vec![0.0f32; n];
            crate::quant::dequant_row(embd.ty, embd.data, token as u64, n as u64, &mut row);
            let rd = self.raw_decode.as_ref().unwrap();
            let pos = self.seqs[seq].pos as usize;
            let logits = rd.raw_step(seq, pos, &row).map_err(ModelError::Accel)?;
            if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = logits.iter().any(|v| v.is_nan());
                eprintln!("logits: max={m:.4} nan={nan} argmax={}", greedy(&logits));
            }
            self.seqs[seq].pos += 1;
            return Ok(vec![logits]);
        }
        // qwen35 프레임 (t=1 단일) — LLM170_FRAME35=1 게이트, 실패는 Err 전파.
        if tokens.len() == 1
            && seq_ids.len() == 1
            && self.acc.is_some()
            && std::env::var("LLM170_FRAME35").is_ok_and(|v| v != "0")
        {
            let logits = self.decode1_frame35(seq_ids[0], tokens[0])?;
            self.seqs[seq_ids[0]].pos += 1;
            return Ok(vec![logits]);
        }
        let batch: Vec<Vec<u32>> = tokens.iter().map(|t| vec![*t]).collect();
        let logits = self.forward(seq_ids, &batch)?;
        for s in seq_ids {
            self.seqs[*s].pos += 1;
        }
        Ok(logits)
    }

    /// greedy 디코드 — GPU argmax 경로 (logits 전사 없음). raw 활성 시 유효.
    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, ModelError> {
        let Some(rd) = self.raw_decode.as_ref() else {
            // 폴백: 일반 decode + greedy
            let logits = self.decode(&[seq], &[token])?;
            return Ok(crate::model::greedy(&logits[0]));
        };
        let n = self.model.hp.n_embd;
        let embd = self.model.wchk("token_embd.weight")?;
        let mut row = vec![0.0f32; n];
        crate::quant::dequant_row(embd.ty, embd.data, token as u64, n as u64, &mut row);
        let pos = self.seqs[seq].pos as usize;
        let tok = rd.raw_step_greedy(seq, pos, &row).map_err(ModelError::Accel)?;
        self.seqs[seq].pos += 1;
        Ok(tok)
    }

    /// MTP draft층 1 forward (06) — eh_proj([enorm(embd(tok)); hnorm(h_t)]) →
    /// 게이티드 어텐션(자체 KV) → FFN → shared head 로짓. 체인 draft용:
    /// 반환 (logits, h_{t+1}) — h는 다음 draft 스텝의 hnorm 입력.
    pub fn mtp_forward(
        &mut self,
        seq: usize,
        token: u32,
        h_in: &[f32],
        pos: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        profile_span!("cpu::mtp_forward");
        let hp = self.model.hp.clone();
        let n_embd = hp.n_embd;
        let il = 64; // blk.64 — MTP층
        let w_eh = self.model.wchk(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let enorm = self.model.f32_vec(&format!("blk.{il}.nextn.enorm.weight"))?;
        let hnorm = self.model.f32_vec(&format!("blk.{il}.nextn.hnorm.weight"))?;

        // 1) embd(tok) 디양자화 → enorm / h_t → hnorm, concat → eh_proj
        let embd = self.model.wchk("token_embd.weight")?;
        let mut tok_row = vec![0.0f32; n_embd];
        crate::quant::dequant_row(embd.ty, embd.data, token as u64, n_embd as u64, &mut tok_row);
        let e_n = crate::ops::rms_norm(&tok_row, &enorm, hp.eps);
        let h_n = crate::ops::rms_norm(h_in, &hnorm, hp.eps);
        let mut cat = vec![0.0f32; 2 * n_embd];
        cat[..n_embd].copy_from_slice(&e_n);
        cat[n_embd..].copy_from_slice(&h_n);
        let acc = self.acc.clone();
        let mut cur = vec![0.0f32; n_embd];
        crate::matmul::mm(&acc, &cat, &w_eh, &mut cur)?;

        // 2) 게이티드 어텐션 — attn_layer와 동일 구조, 자체 KV(mtp_kv_*) 사용
        let attn_out = self.mtp_attn(seq, il, &cur, pos)?;

        // 3) 잔차 + post_attention_norm + FFN
        for i in 0..n_embd {
            cur[i] += attn_out[i];
        }
        let ffn_res = cur.clone();
        let post_w = self.model.f32_vec(&format!("blk.{il}.post_attention_norm.weight"))?;
        let gate_w = self.model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
        let up_w = self.model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
        let down_w = self.model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
        let normed = rms_norm(&cur, &post_w, hp.eps);
        let mut gu: [Vec<Vec<f32>>; 2] =
            [vec![vec![0.0f32; hp.n_ff]; 1], vec![vec![0.0f32; hp.n_ff]; 1]];
        crate::matmul::mm_group(&acc, &[normed], &[gate_w, up_w], &mut gu)?;
        let [mut g, u] = gu;
        for i in 0..hp.n_ff {
            g[0][i] = crate::ops::silu(g[0][i]) * u[0][i];
        }
        let mut ffn_out = vec![vec![0.0f32; n_embd]; 1];
        crate::matmul::mm_batch(&acc, &g, &down_w, &mut ffn_out)?;
        for i in 0..n_embd {
            cur[i] = ffn_out[0][i] + ffn_res[i];
        }

        // 4) shared head — shared_head_norm + output.weight (mtp_use_dedicated_embeddings=false)
        let sh_norm = self.model.f32_vec(&format!("blk.{il}.nextn.shared_head_norm.weight"))?;
        let head = self.model.wchk("output.weight")?;
        let h = rms_norm(&cur, &sh_norm, hp.eps);
        let mut logits = vec![0.0f32; head.n_out as usize];
        crate::matmul::mm(&acc, &h, &head, &mut logits)?;
        Ok((logits, cur))
    }

    fn mtp_attn(
        &mut self,
        seq: usize,
        il: usize,
        x: &[f32],
        pos: u32,
    ) -> Result<Vec<f32>, ModelError> {
        let hp = self.model.hp.clone();
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = self.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;
        let kq_scale = hp.kq_scale();
        let acc = self.acc.clone();

        let mut group: [Vec<Vec<f32>>; 3] = [
            vec![vec![0.0f32; wq.n_out as usize]; 1],
            vec![vec![0.0f32; wk.n_out as usize]; 1],
            vec![vec![0.0f32; wv.n_out as usize]; 1],
        ];
        {
            let xs = vec![x.to_vec()];
            crate::matmul::mm_group(&acc, &xs, &[wq, wk, wv], &mut group)?;
        }
        let [qg, kk, vv] = group;

        // k norm+rope → 자체 KV 캐시 적립
        {
            let st = &mut self.seqs[seq];
            for h in 0..n_kv {
                let src = kk[0][h * hd..h * hd + hd].to_vec();
                let mut head = crate::ops::rms_norm(&src, &k_norm_w, hp.eps);
                crate::ops::rope_head(&mut head, pos, n_rot, hp.rope_base);
                let b = pos as usize * n_kv * hd + h * hd;
                st.mtp_kv_k[b..b + hd].copy_from_slice(&head);
                st.mtp_kv_v[b..b + hd].copy_from_slice(&vv[0][h * hd..h * hd + hd]);
            }
        }
        let st = &self.seqs[seq];
        let cache_k = &st.mtp_kv_k;
        let cache_v = &st.mtp_kv_v;

        let mut attn_out = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            let src = qg[0][h * 2 * hd..h * 2 * hd + hd].to_vec();
            let mut qh = crate::ops::rms_norm(&src, &q_norm_w, hp.eps);
            crate::ops::rope_head(&mut qh, pos, n_rot, hp.rope_base);
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
                *sc = crate::ops::exp_cr(*sc - maxv);
                sum += *sc;
            }
            let ob = h * hd;
            for p in 0..n_past {
                let w = scores[p] / sum;
                if w == 0.0 {
                    continue;
                }
                let b = p * n_kv * hd + kvh * hd;
                for i in 0..hd {
                    attn_out[ob + i] += w * cache_v[b + i];
                }
            }
            let gb = h * 2 * hd + hd;
            for i in 0..hd {
                attn_out[ob + i] *= crate::ops::sigmoid(qg[0][gb + i]);
            }
        }
        // wo 프로젝션
        let mut out = vec![vec![0.0f32; wo.n_out as usize]; 1];
        crate::matmul::mm_batch(&acc, &[attn_out], &wo, &mut out)?;
        Ok(out.into_iter().next().unwrap())
    }

    /// 시퀀스 prefill: 전체 토큰 적립 + 마지막 logits.
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        // 1024토큰 청크 — qwen4exp와 동일 근거: 단일 초대형 forward는 GPU
        // 스크래치·상태 크기를 폭주시킨다 (qwen4exp GPF 실측, 2026-08-31).
        // 청킹은 수치 불변 (GDN chunked·attention 캐시는 순차 적립).
        let chunk: usize = std::env::var("LLM170_Q35_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
            .clamp(16, 1024);
        let mut last = None;
        // 원시 HIP 활성 시 프리필도 t=1 raw 스텝으로 — 상태 동기화 불필요
        // (KV/GDN/conv 링이 raw 디코더에 직접 적립).
        if std::env::var("LLM170_T1_PREFILL").is_ok()
            || (self.raw_decode.is_some()
                && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true))
        {
            let use_batch = std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
                && std::env::var_os("LLM170_T1_PREFILL").is_none()
                && (tokens.len() > 1 || std::env::var_os("LLM170_FORCE_BATCH").is_some());
            if use_batch {
                let rd = self.raw_decode.as_ref().unwrap();
                let n = self.model.hp.n_embd as usize;
                let embd = self.model.wchk("token_embd.weight")?;
                let mut pos = self.seqs[seq].pos as usize;
                let mut cache: Vec<Vec<f32>> = Vec::new();
                for &tok in tokens {
                    let mut row = vec![0.0f32; n];
                    crate::quant::dequant_row(
                        embd.ty,
                        embd.data,
                        tok as u64,
                        n as u64,
                        &mut row,
                    );
                    cache.push(row);
                }
                // 청크 ≤ 64: 소유권형 MMQ (shared ~40KB, acc[16] 레지스터)
                let ch_sz = std::env::var("LLM170_CHUNK").ok().and_then(|v| v.parse().ok())
                    .unwrap_or(if std::env::var_os("LLM170_CO_PATH").is_some() && std::env::var_os("LLM170_EXACT").is_none() { 128 } else { 64 });
                for ch in cache.chunks(ch_sz) {
                    let flat: Vec<f32> = ch.iter().flatten().copied().collect();
                    let logits = rd.raw_prefill(seq, pos, &flat).map_err(ModelError::Accel)?;
                    if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                        let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        eprintln!("logits(batch): max={m:.4} argmax={}", greedy(&logits));
                    }
                    pos += ch.len();
                    last = Some(logits);
                }
                self.seqs[seq].pos = pos as u32;
                return Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]));
            }
            for &t in tokens {
                let logits = self.decode(&[seq], &[t])?;
                last = Some(logits.into_iter().next().unwrap());
            }
            return Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]));
        }
        for ch in tokens.chunks(chunk) {
            let logits = self.forward(&[seq], &[ch.to_vec()])?;
            self.seqs[seq].pos += ch.len() as u32;
            last = Some(logits.into_iter().next().unwrap());
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
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

    /// 스펙 디코드 1사이클 (06) — k draft 생성 → 타깃 forward 연쇄 검증.
    /// v1 단순화: 검증은 타깃 t=1 순차 forward (정확성 동일 — 06 §4.3
    /// "1차는 슬롯별 순차도 허용"). 반환: (수용 토큰들, 타깃 forward 수).
    #[allow(clippy::type_complexity)]
    pub fn spec_step(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), ModelError> {
        let eos = 248044u32;
        // ── draft: 체인 k개 — mtp_forward는 자체 KV에 순차 적립
        let mut h = self.seqs[seq].mtp_h.clone();
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut tok = last_token;
        let base_pos = self.seqs[seq].pos;
        for _ in 0..k {
            let dpos = base_pos + drafts.len() as u32;
            let (logits, nh) = self.mtp_forward(seq, tok, &h, dpos)?;
            let d = greedy(&logits);
            drafts.push(d);
            tok = d;
            h = nh;
            if d == eos {
                break;
            }
        }
        // ── verify: 타깃 t=1 순차 forward — draft와 순서 비교 연쇄 수용
        let mut accepted: Vec<u32> = Vec::new();
        let mut cur_tok = last_token;
        let mut total = 0usize;
        for (i, &d) in drafts.iter().enumerate() {
            let logits = self.decode(&[seq], &[cur_tok])?;
            total += 1;
            let t = greedy(&logits[0]);
            if t == d {
                accepted.push(t);
                cur_tok = t;
                if i + 1 == drafts.len() {
                    let l2 = self.decode(&[seq], &[cur_tok])?;
                    total += 1;
                    accepted.push(greedy(&l2[0]));
                }
            } else {
                accepted.push(t); // 첫 불일치 위치의 타깃 argmax = 보너스
                break;
            }
            if t == eos {
                break;
            }
        }
        Ok((accepted, total))
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
