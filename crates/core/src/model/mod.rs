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
    /// MTP 훅이 계산한 직전 토큰의 draft 로짓/hidden (spec step-0 재사용).
    pub mtp_draft_logits: Vec<f32>,
    pub mtp_draft_tok: u32,
    pub mtp_draft_h: Vec<f32>,
    /// 직전 처리 토큰의 trunk hidden — MTP 시프트 페어링 (tok_p, h_{p-1}) (llama.cpp pending_h).
    pub mtp_pending_h: Vec<f32>,
    /// GDN 미확정 행(토큰) — 부분수용 후 다음 verify 배치에서 재실행 (롤백 재실행 대체).
    pub gdn_carried: Vec<u32>,
    pub mtp_h_next: Vec<f32>,
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
            mtp_draft_logits: Vec::new(),
            mtp_draft_tok: 0,
            mtp_draft_h: Vec::new(),
            mtp_pending_h: vec![0.0; if has_mtp { model.hp.n_embd } else { 0 }],
            gdn_carried: Vec::new(),
            mtp_h_next: vec![0.0; if has_mtp { model.hp.n_embd } else { 0 }],
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
    /// token_embd 원시 복사 캐시 (spec 토큰 행 디양자화용 — 매 스텝 to_vec 폭주 방지).
    pub embd_cache: Option<(llm170_gguf::GgmlType, std::sync::Arc<Vec<u8>>)>,
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
            embd_cache: None,
            frame35: None,
            frame35_clean: vec![false; n_seqs],
            model,
            seqs,
            acc: None,
        }
    }

    /// KV 용량에서 역산한 컨텍스트 길이 (rawhip 상수 테이블 크기 등).
    pub fn ctx_len(&self) -> usize {
        let (n_kv, hd) = (self.model.hp.n_kv, self.model.hp.head_dim);
        self.seqs
            .first()
            .and_then(|s| s.kv_k.first().map(|k| k.len() / (n_kv * hd)))
            .unwrap_or(4096)
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
            let mut xn: Vec<Vec<f32>> = vec![vec![0.0f32; hp.n_embd]; n_tok];
            let rms_ok = match acc.as_deref() {
                Some(a) => a.rms_norm(&xs, &norm_w, hp.eps, &mut xn).is_ok(),
                None => false,
            };
            if rms_ok {
                xs = xn;
            } else {
                for x in xs.iter_mut() {
                    *x = rms_norm(x, &norm_w, hp.eps);
                }
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

            let mut normed: Vec<Vec<f32>> = vec![vec![0.0f32; hp.n_embd]; n_tok];
            let rms_ok = match acc.as_deref() {
                Some(a) => a.rms_norm(&xs, &post_w, hp.eps, &mut normed).is_ok(),
                None => false,
            };
            if !rms_ok {
                for (i, x) in xs.iter().enumerate() {
                    normed[i] = rms_norm(x, &post_w, hp.eps);
                }
            }
            // FFN 상주 체인 (가속기 지원 시): 업/다운로드 1회씩.
            let mut ffn_out: Vec<Vec<f32>> = vec![vec![0.0f32; hp.n_embd]; n_tok];
            let mut ffn_chained = false;
            if let Some(a) = acc.as_deref() {
                ffn_chained = a.ffn_chain(&normed, &gate_w, &up_w, &down_w, &mut ffn_out).is_ok();
            }
            if ffn_chained {
                for (x, o) in xs.iter_mut().zip(ffn_out.iter()) {
                    for (xi, oi) in x.iter_mut().zip(o.iter()) {
                        *xi += *oi;
                    }
                }
                continue;
            }
            let mut ffn_group: [Vec<Vec<f32>>; 2] =
                [vec![vec![0.0f32; n_ff]; n_tok], vec![vec![0.0f32; n_ff]; n_tok]];
            {
                span_block!("cpu::ffn_gate_up", {
                    mm_group(&acc, &normed, &[gate_w, up_w], &mut ffn_group)?;
                });
            }
            let [mut gate_y, up_y] = ffn_group;
            let mut glu: Vec<Vec<f32>> = vec![vec![0.0f32; n_ff]; n_tok];
            let silu_ok = match acc.as_deref() {
                Some(a) => a.silu_mul(&gate_y, &up_y, &mut glu).is_ok(),
                None => false,
            };
            if silu_ok {
                gate_y = glu;
            } else {
                for t in 0..n_tok {
                    for i in 0..n_ff {
                        gate_y[t][i] = silu(gate_y[t][i]) * up_y[t][i];
                    }
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
        // MTP nextn KV 적립 — 프롬프트/배치 토큰 전체 (draft 어텐션 컨텍스트).
        // h_in = 본체 최종 hidden (output_norm 전). 로짓 없이 1층만.
        if std::env::var_os("LLM170_SPEC_DBG").is_some() {
            eprintln!("  [hookguard] mtp_h.len={} seq0={}", self.seqs[seq_ids[0]].mtp_h.len(), seq_ids[0]);
        }
        if !self.seqs[seq_ids[0]].mtp_h.is_empty() {
            for s in 0..n_seqs {
                let sid = seq_ids[s];
                let pos0 = self.seqs[sid].pos as usize;
                let mut prev_h = std::mem::take(&mut self.seqs[sid].mtp_pending_h);
                for t in 0..t_len {
                    // 마지막 토큰만 로짓 (spec step-0 재사용), 나머지는 KV 적립
                    let wl = t + 1 == t_len;
                    let h_t = xs[s * t_len + t].clone();
                    // 시프트 페어링: MTP(tok_p, h_{p-1})
                    let (lg, hn) =
                        self.mtp_step(sid, batch[s][t], &prev_h, (pos0 + t) as u32, wl)?;
                    prev_h.copy_from_slice(&h_t);
                    if wl {
                        let am = crate::model::greedy(&lg);
                        self.seqs[sid].mtp_draft_logits = lg;
                        self.seqs[sid].mtp_h_next = hn;
                        if std::env::var_os("LLM170_SPEC_DBG").is_some() {
                            eprintln!("  [hook] sid={sid} tok={:?} am={am}", batch[s][t]);
                        }
                    }
                }
                self.seqs[sid].mtp_pending_h = prev_h;
            }
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
            let mut h_t = Vec::new();
            let logits = if !self.seqs[seq].mtp_h.is_empty() {
                let lg = rd.raw_step_h(seq, pos, &row, &mut h_t).map_err(ModelError::Accel)?;
                let rd2 = rd.clone();
                let prev_h = std::mem::take(&mut self.seqs[seq].mtp_pending_h);
                let (am, _) = rd2
                    .mtp_step_gpu(seq, &row, &prev_h, pos)
                    .map_err(ModelError::Accel)?;
                let st = &mut self.seqs[seq];
                st.mtp_draft_tok = am;
                st.mtp_pending_h = h_t;
                lg
            } else {
                rd.raw_step(seq, pos, &row).map_err(ModelError::Accel)?
            };
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
    /// 게이티드 어텐션(자체 KV) → FFN → (선택) shared head 로짓.
    /// with_logits=false → KV 적립·h만 (프리필/디코드 훅 — head GEMV 고가).
    pub fn mtp_step(
        &mut self,
        seq: usize,
        token: u32,
        h_in: &[f32],
        pos: u32,
        with_logits: bool,
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
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] cat e0={:.6} e1={:.6} esum={:.4} | h0={:.6} h1={:.6} hsum={:.4}", e_n[0], e_n[1], e_n.iter().map(|&x| x as f64).sum::<f64>(), h_n[0], h_n[1], h_n.iter().map(|&x| x as f64).sum::<f64>());
        }
        let acc = self.acc.clone();
        let mut cur = vec![0.0f32; n_embd];
        crate::matmul::mm(&acc, &cat, &w_eh, &mut cur)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] eh sum={:.5} x0={:.5} x1={:.5}", cur.iter().map(|&x| x as f64).sum::<f64>(), cur[0], cur[1]);
        }

        // 2) 게이티드 어텐션 — attn_layer와 동일 구조, 자체 KV(mtp_kv_*) 사용
        let attn_out = self.mtp_attn(seq, il, &cur, pos)?;

        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] wo sum={:.5} x0={:.5}", attn_out.iter().map(|&x| x as f64).sum::<f64>(), attn_out[0]);
        }
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
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] ff sum={:.5} x0={:.5}", cur.iter().map(|&x| x as f64).sum::<f64>(), cur[0]);
        }

        // 4) shared head — with_logits에만 (output.weight GEMV는 고가)
        if !with_logits {
            return Ok((Vec::new(), cur));
        }
        let sh_norm = self.model.f32_vec(&format!("blk.{il}.nextn.shared_head_norm.weight"))?;
        let head = self.model.wchk("output.weight")?;
        let h = rms_norm(&cur, &sh_norm, hp.eps);
        let mut logits = vec![0.0f32; head.n_out as usize];
        crate::matmul::mm(&acc, &h, &head, &mut logits)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] head L0..7={:?} hnorm0..3={:?}", &logits[0..8], &h[0..4]);
        }
        Ok((logits, cur))
    }

    /// MTP draft forward — 로짓 포함 (spec 체인용).
    pub fn mtp_forward(
        &mut self,
        seq: usize,
        token: u32,
        h_in: &[f32],
        pos: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        self.mtp_step(seq, token, h_in, pos, true)
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

        // llama.cpp: eh_proj 출력 → attn_norm → q/k/v (누락이 0-수용률 주벅)
        let attn_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
        let xn = crate::ops::rms_norm(x, &attn_norm_w, hp.eps);
        let mut group: [Vec<Vec<f32>>; 3] = [
            vec![vec![0.0f32; wq.n_out as usize]; 1],
            vec![vec![0.0f32; wk.n_out as usize]; 1],
            vec![vec![0.0f32; wv.n_out as usize]; 1],
        ];
        {
            let xs = vec![xn];
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
                let rd = self.raw_decode.clone().unwrap();
                let n = self.model.hp.n_embd as usize;
                // 소유 복사 — 루프 내 self.mtp_step (&mut self)와 공존
                let (embd_ty, embd_data) = {
                    let t = self.model.wchk("token_embd.weight")?;
                    (t.ty, t.data.to_vec())
                };
                let mut pos = self.seqs[seq].pos as usize;
                let mut cache: Vec<Vec<f32>> = Vec::new();
                for &tok in tokens {
                    let mut row = vec![0.0f32; n];
                    crate::quant::dequant_row(
                        embd_ty,
                        &embd_data,
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
                    let logits = if !self.seqs[seq].mtp_h.is_empty() {
                        // MTP KV 적립: 전 토큰 hidden 회수 후 훅 (마지막만 로짓)
                        let mut h_all: Vec<f32> = Vec::new();
                        let lg = rd
                            .raw_prefill_h(seq, pos, &flat, &mut h_all)
                            .map_err(ModelError::Accel)?;
                        let n_e = self.model.hp.n_embd;
                        let mut prev_h = self.seqs[seq].mtp_pending_h.clone();
                        for ti in 0..ch.len() {
                            let wl = ti + 1 == ch.len();
                            let h_t = h_all[ti * n_e..(ti + 1) * n_e].to_vec();
                            let t_tok = tokens[pos + ti];
                            let mut trow = vec![0.0f32; n_e];
                            crate::quant::dequant_row(
                                embd_ty, &embd_data, t_tok as u64, n_e as u64, &mut trow);
                            let rd2 = rd.clone();
                            // llama.cpp 시프트 페어링: MTP(tok_p, h_{p-1}) — h_{-1}=0
                            let (am, _hn) = rd2
                                .mtp_step_gpu(seq, &trow, &prev_h, pos + ti)
                                .map_err(ModelError::Accel)?;
                            prev_h.copy_from_slice(&h_t);
                            if wl {
                                let st = &mut self.seqs[seq];
                                st.mtp_draft_tok = am;
                                st.mtp_pending_h = h_t;
                            }
                        }
                        lg
                    } else {
                        rd.raw_prefill(seq, pos, &flat).map_err(ModelError::Accel)?
                    };
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
        // GPU 검증 경로 (rawhip): draft 체인(CPU MTP층 + GPU head) → 1배치 검증.
        if self.raw_decode.is_some()
            && !self.seqs[seq].mtp_h.is_empty()
            && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
            && std::env::var_os("LLM170_SPEC_GPU").is_some()
        {
            return self.spec_step_gpu(seq, last_token, k);
        }
        let eos = 248044u32;
        // 교차 검증: 타깃 decode(토큰) → (hook이 계산한 (토큰,h) 쌍의 draft 로짓) 비교.
        // draft 체인은 직전 draft 토큰 쌍으로 순차 — target decode가 h를 갱신하는 즉시.
        let base_pos = self.seqs[seq].pos; // 슬롯 0..base_pos-1 처리됨; last_token = 위치 base_pos 토큰(미처리)
        let mut accepted: Vec<u32> = Vec::new();
        let mut total = 0usize;
        let mut cur = last_token;
        // 체인 상태: (draft 토큰, 그 pair의 h_next) — j=0은 hook 저장분 사용
        let mut chain_tok: Option<u32> = None;
        let mut chain_h: Vec<f32> = Vec::new();
        let mut chain_pos = base_pos; // 다음 mtp_forward가 쓸 슬롯
        for j in 0..=k {
            let logits = self.decode(&[seq], &[cur])?;
            total += 1;
            let t = greedy(&logits[0]);
            // draft 예측
            let d = if j == 0 {
                greedy(&self.seqs[seq].mtp_draft_logits)
            } else {
                // 직전 루프에서 준비한 체인 로짓
                let (lgt, nh) = self.mtp_forward(seq, chain_tok.unwrap(), &chain_h, chain_pos)?;
                chain_h = nh;
                chain_pos += 1;
                greedy(&lgt)
            };
            accepted.push(t);
            if std::env::var_os("LLM170_SPEC_DBG").is_some() {
                eprintln!("  verify j={j} target={t} draft={d} {}", if t == d { "OK" } else { "MISS" });
            }
            if t != d || t == eos {
                break;
            }
            // 수용: 다음 비교용 체인 준비 — (d, h_next) pair는 다음 루프에서 forward
            chain_tok = Some(d);
            if chain_h.is_empty() {
                chain_h = self.seqs[seq].mtp_h_next.clone();
            }
            cur = t;
        }
        Ok((accepted, total))
    }

    /// GPU 검증 스펙: draft k개(순차) → raw_verify 1배치 → 최장 수용 접두 + 보너스.
    fn spec_step_gpu(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), ModelError> {
        let eos = 248044u32;
        let sp_t0 = std::time::Instant::now();
        let rd = self.raw_decode.clone().ok_or(ModelError::Accel("raw 없음".into()))?;
        let base_pos = self.seqs[seq].pos; // 슬롯 0..base_pos-1 처리됨
        let t_draft0 = std::time::Instant::now();
        // ── draft: step-0 = (last_token, pending_h) 시프트 페어링; j≥1 = 체인 자가 h
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let n_e = self.model.hp.n_embd;
        {
            if self.embd_cache.is_none() {
                let t = self.model.wchk("token_embd.weight")?;
                self.embd_cache = Some((t.ty, std::sync::Arc::new(t.data.to_vec())));
            }
            let (embd_ty, embd_arc) = self.embd_cache.as_ref().unwrap().clone();
            let embd_data: &Vec<u8> = &embd_arc;
            let mut trow = vec![0.0f32; n_e];
            crate::quant::dequant_row(embd_ty, &embd_data, last_token as u64, n_e as u64, &mut trow);
            let pending = std::mem::take(&mut self.seqs[seq].mtp_pending_h);
            let t_d0 = std::time::Instant::now();
            let (d0, _) = rd
                .mtp_step_gpu(seq, &trow, &pending, base_pos as usize)
                .map_err(ModelError::Accel)?;
            if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
                eprintln!("[d0] draft0={:.1}ms", t_d0.elapsed().as_secs_f64() * 1e3);
            }
            // pending은 다시 저장 (verify 후 마지막 행 hidden으로 갱신)
            self.seqs[seq].mtp_pending_h = pending;
            drafts.push(d0);
            let mut tok = d0;
            for j in 1..k {
                let dpos = (base_pos + drafts.len() as u32 - 1) as usize;
                let tc = std::time::Instant::now();
                let mut trow = vec![0.0f32; n_e];
                crate::quant::dequant_row(embd_ty, &embd_data, tok as u64, n_e as u64, &mut trow);
                let td = std::time::Instant::now();
                let d = rd.mtp_step_chain(seq, &trow, dpos).map_err(ModelError::Accel)?;
                if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
                    eprintln!("[ch] j={j} deq={:.2}ms step={:.2}ms", td.duration_since(tc).as_secs_f64()*1e3, td.elapsed().as_secs_f64()*1e3);
                }
                drafts.push(d);
                tok = d;
                if d == eos {
                    break;
                }
            }
        }
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[sp] draft chain={:.1}ms k={}", t_draft0.elapsed().as_secs_f64() * 1e3, drafts.len());
        }
        let t_v0 = std::time::Instant::now();
        // ── verify: [carried..., last_token, d0, d1, ...] 1배치 — 행별 argmax = 다음 토큰 정답
        // carried = 직전 부분수용에서 GDN이 미확정인 행 — 같은 토큰·같은 위치 재실행
        // (결정론적 커널 → 동일 결과, KV는 동일값 재기입). 재실행 배치를 대체한다.
        let mut carried: Vec<u32> = std::mem::take(&mut self.seqs[seq].gdn_carried);
        // carried 상한 — 초과 시 GDN 커밋 배치(헤드 무의미하지만 간단)로 소화 후 청소.
        // 전체수용이 드문 높은 k에서 배치 무한 증식 방지 (롤백 재실행의 분할 상환).
        if carried.len() + 1 + k > 16 {
            let n_c = self.model.hp.n_embd;
            let (embd_ty_c, embd_arc_c) = self.embd_cache.as_ref().unwrap().clone();
            let mut crows: Vec<f32> = Vec::with_capacity(carried.len() * n_c);
            for &tk in carried.iter() {
                let mut r = vec![0.0f32; n_c];
                crate::quant::dequant_row(embd_ty_c, &embd_arc_c, tk as u64, n_c as u64, &mut r);
                crows.extend(r);
            }
            let mut cam: Vec<u32> = Vec::new();
            let mut ch_all: Vec<f32> = Vec::new();
            rd.raw_verify(seq, (base_pos - carried.len() as u32) as usize, &crows, &mut cam, &mut ch_all)
                .map_err(ModelError::Accel)?;
            // 커밋 후 mtp 진행도 보강 (멱등 — 신규 행만)
            for i in 1..carried.len() {
                let h_prev = ch_all[(i - 1) * n_c..i * n_c].to_vec();
                rd.mtp_step_adv(seq, &crows[i * n_c..(i + 1) * n_c], &h_prev, (base_pos as usize) - carried.len() + i)
                    .map_err(ModelError::Accel)?;
            }
            carried = Vec::new();
        }
        let carried_n = carried.len();
        let pos0 = base_pos - carried_n as u32;
        let t = 1 + drafts.len() + carried_n;
        let n = self.model.hp.n_embd;
        if self.embd_cache.is_none() {
            let tw = self.model.wchk("token_embd.weight")?;
            self.embd_cache = Some((tw.ty, std::sync::Arc::new(tw.data.to_vec())));
        }
        let (embd_ty, embd_arc) = self.embd_cache.as_ref().unwrap().clone();
        let embd_data: &Vec<u8> = &embd_arc;
        let mut rows: Vec<f32> = Vec::with_capacity(t * n);
        let mut row_toks: Vec<u32> = Vec::with_capacity(t);
        for &tk in carried
            .iter()
            .chain(std::iter::once(&last_token))
            .chain(drafts.iter())
        {
            let mut r = vec![0.0f32; n];
            crate::quant::dequant_row(embd_ty, embd_data, tk as u64, n as u64, &mut r);
            rows.extend(r);
            row_toks.push(tk);
        }
        // 부분수용 대비 GDN/conv 스냅샷 (KV는 위치 색인이라 자가치유)
        rd.gdn_snapshot().map_err(ModelError::Accel)?;
        let mut am: Vec<u32> = Vec::new();
        let mut h_all: Vec<f32> = Vec::new();
        rd.raw_verify(seq, pos0 as usize, &rows, &mut am, &mut h_all)
            .map_err(ModelError::Accel)?;
        // 수용 (신규 세그먼트만): am[carried_n + j] vs drafts[j]
        let mut accepted: Vec<u32> = Vec::new();
        for j in 0..drafts.len() {
            let i = carried_n + j;
            accepted.push(am[i]);
            if am[i] != drafts[j] || am[i] == eos {
                break;
            }
        }
        if std::env::var_os("LLM170_SPEC_DBG").is_some() {
            eprintln!(
                "  gpu-verify pos={base_pos} carried={carried_n} drafts={drafts:?} am={am:?} acc_n={}",
                if accepted.len() == drafts.len()
                    && drafts.iter().zip(accepted.iter()).all(|(d, a)| d == a)
                { accepted.len() + 1 } else { accepted.len().max(1) }
            );
        }
        let all_acc = accepted.len() == drafts.len()
            && drafts.iter().zip(accepted.iter()).all(|(d, a)| d == a);
        let kept_new = if all_acc {
            accepted.push(am[t - 1]);
            1 + drafts.len() // last + 전부
        } else {
            accepted.len() // last + 매칭 draft 수
        };
        if all_acc {
            // 전부 수용 — 상태 그대로 유효, carried 소멸.
            self.seqs[seq].gdn_carried = Vec::new();
        } else {
            // 부분수용 — GDN 복원 후 유지 행을 carried로 (다음 배치에서 재실행).
            rd.gdn_restore().map_err(ModelError::Accel)?;
            self.seqs[seq].gdn_carried = row_toks[..carried_n + kept_new].to_vec();
        }
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[sp] verify+decide={:.1}ms", t_v0.elapsed().as_secs_f64() * 1e3);
        }
        let t_adv0 = std::time::Instant::now();
        // ── MTP 상태 진행 (시프트 페어링): 행 0은 draft step-0이 이미 처리.
        // carried 구간은 직전 스텝이 이미 적립(멱등) — 신규 행부터만.
        {
            let kept = carried_n + kept_new;
            for i in (carried_n.max(1))..kept {
                let h_prev = h_all[(i - 1) * n..i * n].to_vec();
                rd.mtp_step_adv(seq, &rows[i * n..(i + 1) * n], &h_prev, pos0 as usize + i)
                    .map_err(ModelError::Accel)?;
            }
            // pending = 마지막 유지 행의 trunk hidden (다음 draft step-0 페어링)
            self.seqs[seq].mtp_pending_h = h_all[(kept - 1) * n..kept * n].to_vec();
        }
        // 시퀀스 pos 동기 — 유지 신규 행 수만 반영
        self.seqs[seq].pos = base_pos + (kept_new as u32);
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[sp] advance={:.1}ms | step total={:.1}ms acc={}", t_adv0.elapsed().as_secs_f64() * 1e3, sp_t0.elapsed().as_secs_f64() * 1e3, accepted.len());
        }
        let n = accepted.len().max(1);
        Ok((accepted, n))
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
