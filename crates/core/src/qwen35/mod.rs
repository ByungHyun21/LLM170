//! qwen35 (Qwen3.8-27B 계열) CPU 참조 엔진.
//!
//! 그래프 배선: `~/local_llm/llama.cpp/src/models/qwen35.cpp` (2026-08-30 판).
//! 하이퍼파라미터(27B): n_embd 5120, FFN 17408(SwiGLU), vocab 248320,
//! rms_eps 1e-6, ctx 262144. 64층 = 12×(3×GDN→FFN + 1×GatedAttn→FFN)
//! (full_attention_interval=4 → full-attn il∈{3,7,…,63}),
//! block_count=65 (MTP blk.64, mtp_num_hidden_layers=1).
//! - 잔차: h += attn(rms(h)); h += ffn(rms_post(h)) — FFN 분기는 **post**-attn norm
//! - GDN층(interval≠3): qkv → depthwise conv+SiLU → L2 norm(q,k) → GDN → rms_norm·silu(z) → ssm_out
//! - Full-attn층(interval==3): wq는 Q‖gate 퓨전 [5120, 12288] 헤드별 인터리브
//!   (스트라이드 2·head_dim) → per-head rms norm(q,k) [256] → RoPE half-split
//!   (n_rot 64, base 1e7 — 텍스트 토큰에서 mrope 구간 [11,11,10]과 동치)
//!   → GQA(24Q/4KV, scale 1/√256) → ⊙sigmoid(gate) → wo. KV f16 64 KiB/token(16층).
//! - 하이퍼파라미터는 config.json 메타에서 동적 로드 (소형 검증 모델 지원).
//! - f32 KV, f32 GDN 상태 (참조 정확도 우선).

mod dispatch;
use dispatch::{Acc, mm, mm_batch, mm_group};
mod diag;
pub(crate) mod frame;
pub const EOS_EOT: u32 = 248044;
pub mod hparams;
pub mod prefill;
pub mod spec;
pub mod stages;
pub use frame::Frame;

use hparams::Hparams;
use llm170_diag::profile_span;

use crate::matmul::Weight;
use crate::ops::{rms_norm, silu};
use crate::quant::dequant_row;
#[derive(Debug)]
pub enum ModelError {
    MissingTensor(String),
    UnsupportedLayout {
        name: String,
        why: &'static str,
    },
    BadHparam(&'static str),
    Accel(String),
    /// W4A16 디렉터리 로더 오류(§3.5 A안 직접 로드).
    W4a16(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            ModelError::UnsupportedLayout { name, why } => {
                write!(f, "unsupported layout {name}: {why}")
            }
            ModelError::BadHparam(w) => write!(f, "bad hyperparameter: {w}"),
            ModelError::Accel(e) => write!(f, "accelerator: {e}"),
            ModelError::W4a16(e) => write!(f, "w4a16: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// 가중치 소스 — **W4A16 디렉터리 단일**(2026-10-08, plans/w4a16-cuda.md §5).
/// 세부 접근은 `w4a16` 로더(mmap 슬라이스 + 순열 사본).
pub struct Model {
    w4: Box<crate::w4a16::W4a16Model>,
    pub hp: Hparams,
    pub token_pieces: Vec<String>,
    /// plans/113(llama-vllm P13): f32 norm 가중 디양자화 캐시 — 첫 호출 1회
    /// 디양자화 후 재사용(수치 불변). 값 경로 매 포워드 층별 norm 재디양자 제거.
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
    /// W4A16 디렉터리 로드(§3.5 직접 로드) — config.json에서 하이퍼파라미터,
    /// tokenizer.json/vocab.json에서 조각표. 소스 검증은 W4A16 로더의
    /// validate()가 전수(트리플·커버리지). MTP는 미매핑(스펙 별도 과제).
    pub fn load(dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        profile_span!("model::load_w4a16");
        let w4 =
            crate::w4a16::W4a16Model::open(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let rep = w4
            .validate()
            .map_err(|e| ModelError::W4a16(e.to_string()))?;
        if !rep.ok() {
            return Err(
                ModelError::W4a16(format!("커버리지 검증 실패:\n{}", rep.summary())).into(),
            );
        }
        let c = &w4.cfg;
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
        };
        if !d_inner.is_multiple_of(hp.dt_rank) || d_inner / hp.dt_rank != d_state {
            return Err(ModelError::BadHparam("d_inner/dt_rank != d_state").into());
        }
        let token_pieces =
            crate::w4a16::load_pieces(dir).map_err(|e| ModelError::W4a16(e.to_string()))?;
        let m = Model {
            w4: Box::new(w4),
            hp,
            token_pieces,
            f32_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        };
        for name in ["token_embd.weight", "output.weight"] {
            m.w(name).ok_or(ModelError::MissingTensor(name.into()))?;
        }
        // output_norm은 Synth(norm +1) — f32_vec 경유로 실재·형상 검증.
        m.f32_vec("output_norm.weight")?;
        Ok(m)
    }

    /// plans/40 페이지 반납은 구 mmap 소스 전용이었음 — 2026-10-08 단일 트랙으로
    /// 무동작(W4A16 소스는 전부 CPU 경로). 호출부 계약 유지를 위해 잔존.
    pub fn discard_weight_pages(&self, _keep: &[&str]) -> u64 {
        0
    }

    /// 무게 뷰 — W4A16 이름맵(§3.5)으로 슬라이스/분리버퍼 구성.
    pub fn w(&self, name: &str) -> Option<Weight<'_>> {
        w4_weight(&self.w4, name)
    }

    /// 텐서 실재 판정 — MTP 탑재 여부 등.
    pub fn has_tensor(&self, name: &str) -> bool {
        crate::w4a16::eng(name).is_some()
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
        let v = match crate::w4a16::eng(name) {
            // 정규화는 HF zero-centered(w−1 저장) — 로드 시 +1 보정.
            // ssm_a는 −exp(A_log), dt_bias/A_log·conv는 V헤드 순열 합성
            // (엔진 subhead-major 계약 — perm: 1 인덱스·2 채널).
            Some(crate::w4a16::Eng::Synth {
                name: hf,
                plus1,
                neg_exp,
                perm,
            }) => {
                let mut v = if perm == 2 {
                    w4.conv_rows_f32_permuted(&hf)
                        .map_err(|e| ModelError::W4a16(e.to_string()))?
                } else {
                    let raw = w4
                        .plain_vec_f32(&hf)
                        .map_err(|e| ModelError::W4a16(e.to_string()))?;
                    if perm == 1 {
                        w4.permute_heads_f32(&raw)
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
            // 플레인 2D(BF16) — Weight 경유 디양자화(행 순열 포함).
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

/// W4A16 소스의 Weight 구성(§3.5 A안) — Quant는 분리 버퍼(packed+scale),
/// Plain은 원시 슬라이스(ty는 dtype에서). Synth(norm류)는 None —
/// f32_vec 전용 계약(호출부가 w()로 요구하지 않는다).
fn w4_weight<'a>(w4: &'a crate::w4a16::W4a16Model, name: &str) -> Option<Weight<'a>> {
    use crate::w4a16::{Eng, PV};
    use crate::wtype::WType;
    match crate::w4a16::eng(name)? {
        Eng::Quant { base, vperm } => {
            let (n, k) = w4.lin_shape(&base)?;
            let (data, aux) = match vperm {
                PV::None => (
                    w4.tensor_slice(&format!("{base}.weight_packed"))?,
                    w4.tensor_slice(&format!("{base}.weight_scale"))?,
                ),
                _ => w4.perm_quant(&base, vperm)?,
            };
            Some(Weight {
                data,
                aux: Some(aux),
                ty: WType::W4a16G128Split,
                n_in: k as u64,
                n_out: n as u64,
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
                w4.perm_plain(&hf)?
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
            })
        }
        Eng::Synth { .. } => None,
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
        let has_mtp = model.has_tensor("blk.64.nextn.eh_proj.weight");
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
    pub acc: Acc,
    /// qwen35 디코드 프레임 (t=1) — LLM170_FRAME35=1 첫 디코드에서 생성.
    pub raw_decode: Option<std::sync::Arc<dyn crate::matmul::RawDecode>>,
    /// token_embd 원시 복사 캐시 (spec 토큰 행 디양자화용 — 매 스텝 to_vec 폭주 방지).
    pub embd_cache: Option<(crate::wtype::WType, std::sync::Arc<Vec<u8>>)>,
    pub frame: Option<Frame>,
    /// MTP 스펙 의도 — true일 때만 prefill/decode 훅 활성 (미사용 시
    /// 훅 비용으로 prefill 3배 저하 방지, 2026-09-04 계측).
    pub mtp_wanted: bool,
    /// 시퀀스별 프레임 상태 유효 플래그 — 값 경로 실행(prefill 등)마다 무효화.
    pub(crate) frame_clean: Vec<bool>,
}

impl Engine {
    /// MTP(nextn) 텐서 탑재 여부 — --spec 사용 가능 판정.
    pub fn has_mtp(&self) -> bool {
        !self
            .seqs
            .first()
            .map(|s| s.mtp_h.is_empty())
            .unwrap_or(true)
    }

    pub fn new(model: Model, n_seqs: usize, ctx: usize) -> Self {
        let seqs = (0..n_seqs).map(|_| SeqState::new(&model, ctx)).collect();
        Engine {
            raw_decode: None,
            embd_cache: None,
            frame: None,
            mtp_wanted: false,
            frame_clean: vec![false; n_seqs],
            model,
            seqs,
            acc: None,
        }
    }

    /// KV 용량에서 역산한 컨텍스트 길이.
    pub fn ctx_len(&self) -> usize {
        let (n_kv, hd) = (self.model.hp.n_kv, self.model.hp.head_dim);
        self.seqs
            .first()
            .and_then(|s| s.kv_k.first().map(|k| k.len() / (n_kv * hd)))
            .unwrap_or(4096)
    }
    /// 시퀀스 상태 전체 초기화 (무상태 HTTP 서버용) — mmap은 유지.
    /// ctx는 기존 KV 용량에서 역산 (첫 kv_k 길이).
    ///
    /// GPU 상주 상태(GDN S/conv 링)도 전 슬롯 영점화 — 2026-09-20 plans/84 A:
    /// CPU SeqState만 교체하면 raw 프리필이 이전 대화의 더러운 초기 상태를
    /// 읽어 두 번째 동일 프리필부터 logits이 발산했다 (chunk-check 재현:
    /// 1회째 bits-identical, 2회째 max|Δ|≈14). reset_seq은 이미 raw_reset.
    pub fn reset_states(&mut self) {
        let n_kv = self.model.hp.n_kv;
        let hd = self.model.hp.head_dim;
        let ctx = self
            .seqs
            .first()
            .and_then(|s| s.kv_k.first().map(|k| k.len() / (n_kv * hd)))
            .unwrap_or(4096);
        if let Some(rd) = self.raw_decode.as_ref() {
            for seq in 0..self.seqs.len() {
                let _ = rd.raw_reset(seq);
            }
        }
        for i in 0..self.seqs.len() {
            self.seqs[i] = SeqState::new(&self.model, ctx);
            // GPU 프레임 상태는 새 SeqState와 어긋남 — 재구축 유도 (qwen4exp reset_states와 동일 원칙).
            self.frame_clean[i] = false;
        }
    }

    /// 슬롯 단위 상태 초기화 (연속 배칭 서버 — 04). 해당 시퀀스의
    /// KV/GDN/conv 상태만 영점화, 다른 슬롯은 간섭 없음.
    pub fn reset_seq(&mut self, seq: usize) {
        let n_kv = self.model.hp.n_kv;
        let hd = self.model.hp.head_dim;
        let ctx = self.seqs[seq]
            .kv_k
            .first()
            .map(|k| k.len() / (n_kv * hd))
            .unwrap_or(4096);
        // raw 디코더 상주 상태(GDN/conv)도 제로화 — 슬롯 재사용 시 누수 방지.
        if let Some(rd) = self.raw_decode.as_ref() {
            let _ = rd.raw_reset(seq);
        }
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
        self.forward_emb(seq_ids, batch, None)
    }

    /// forward 변형 — 임베딩 행 사전 조달(비전 스플라이스) 시 rows 직접 사용.
    /// rows.len() == 전체 토큰 수 필수. None이면 token_embd 디양자화.
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
        // 값 경로 실행 — 프레임 GPU 상태는 CPU 상태와 어긋나 무효화.
        for s in seq_ids {
            self.frame_clean[*s] = false;
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

            let ctx = stages::Ctx {
                model: &self.model,
                acc: &self.acc,
            };
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
                ffn_chained = a
                    .ffn_chain(&normed, &gate_w, &up_w, &down_w, &mut ffn_out)
                    .is_ok();
            }
            if ffn_chained {
                for (x, o) in xs.iter_mut().zip(ffn_out.iter()) {
                    for (xi, oi) in x.iter_mut().zip(o.iter()) {
                        *xi += *oi;
                    }
                }
                continue;
            }
            let mut ffn_group: [Vec<Vec<f32>>; 2] = [
                vec![vec![0.0f32; n_ff]; n_tok],
                vec![vec![0.0f32; n_ff]; n_tok],
            ];
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
            if llm170_diag::dump::opts().key("debug_layers") {
                let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = logits.iter().any(|v| v.is_nan());
                eprintln!("logits: max={m:.4} nan={nan} argmax={}", greedy(&logits));
            }
            result.push(logits);
        }
        // MTP nextn KV 적립 — 프롬프트/배치 토큰 전체 (draft 어텐션 컨텍스트).
        // h_in = 본체 최종 hidden (output_norm 전). 로짓 없이 1층만.
        if !self.seqs[seq_ids[0]].mtp_h.is_empty() && self.mtp_wanted {
            // plans/46: raw 백엔드는 GPU MTP 스텝을 사용 — CPU mtp_step은 토큰당 ~150ms로
            // 프리필·검증을 30× 악화시켰다. GPU 경로는 argmax만 반환 → mtp_draft_tok 사용.
            let raw = self.raw_decode.clone();
            let n_e = self.model.hp.n_embd;
            for s in 0..n_seqs {
                let sid = seq_ids[s];
                let pos0 = self.seqs[sid].pos as usize;
                let mut prev_h = std::mem::take(&mut self.seqs[sid].mtp_pending_h);
                for t in 0..t_len {
                    let wl = t + 1 == t_len;
                    let h_t = xs[s * t_len + t].clone();
                    if let Some(rd) = &raw {
                        let embd = self.model.wchk("token_embd.weight")?;
                        let mut row = vec![0.0f32; n_e];
                        crate::quant::dequant_row(
                            embd.ty,
                            embd.data,
                            batch[s][t] as u64,
                            n_e as u64,
                            &mut row,
                        );
                        let (am, hn) = rd
                            .mtp_step_gpu(sid, &row, &prev_h, pos0 + t)
                            .map_err(ModelError::Accel)?;
                        prev_h.copy_from_slice(&h_t);
                        if wl {
                            let st = &mut self.seqs[sid];
                            st.mtp_draft_tok = am;
                            st.mtp_draft_logits.clear();
                            st.mtp_h_next = hn;
                        }
                    } else {
                        // 시프트 페어링: MTP(tok_p, h_{p-1}) — CPU 폴백
                        let (lg, hn) =
                            self.mtp_step(sid, batch[s][t], &prev_h, (pos0 + t) as u32, wl)?;
                        prev_h.copy_from_slice(&h_t);
                        if wl {
                            let _am = crate::qwen35::greedy(&lg);
                            self.seqs[sid].mtp_draft_logits = lg;
                            self.seqs[sid].mtp_h_next = hn;
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
            && llm170_diag::flag::ne0("LLM170_RAWHIP")
            && let Some(rd) = self.raw_decode.as_ref()
        {
            let seq = seq_ids[0];
            let token = tokens[0];
            let n = self.model.hp.n_embd;
            let embd = self.model.wchk("token_embd.weight")?;
            let mut row = vec![0.0f32; n];
            crate::quant::dequant_row(embd.ty, embd.data, token as u64, n as u64, &mut row);
            let pos = self.seqs[seq].pos as usize;
            let mut h_t = Vec::new();
            let logits = if !self.seqs[seq].mtp_h.is_empty() && self.mtp_wanted {
                let lg = rd
                    .raw_step_h(seq, pos, &row, &mut h_t)
                    .map_err(ModelError::Accel)?;
                let rd2 = rd.clone();
                let prev_h = std::mem::take(&mut self.seqs[seq].mtp_pending_h);
                let (am, hn) = rd2
                    .mtp_step_gpu(seq, &row, &prev_h, pos)
                    .map_err(ModelError::Accel)?;
                let st = &mut self.seqs[seq];
                st.mtp_draft_tok = am;
                // 체인용 MTP층 hidden — 미저장이면 j>=1 초안이 0입력으로 계산된다
                // (2026-09-12 RCA: j>=1 수락 0/12).
                if st.mtp_h_next.len() == hn.len() {
                    st.mtp_h_next.copy_from_slice(&hn);
                }
                st.mtp_pending_h = h_t;
                lg
            } else {
                rd.raw_step(seq, pos, &row).map_err(ModelError::Accel)?
            };
            if llm170_diag::dump::opts().key("debug_layers") {
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
            && llm170_diag::flag::on_nonzero("LLM170_FRAME35")
        {
            let logits = self.decode1_frame(seq_ids[0], tokens[0])?;
            self.seqs[seq_ids[0]].pos += 1;
            return Ok(vec![logits]);
        }
        // 구 np 배치 GPU 경로 — 각 seq 1토큰, GEMM 공유 (plans/15)
        if tokens.len() > 1
            && seq_ids.len() > 1
            && self.raw_decode.is_some()
            && llm170_diag::flag::ne0("LLM170_RAWHIP")
        {
            let rd = self.raw_decode.clone().unwrap();
            let n = self.model.hp.n_embd;
            // plans/92 P6: token_embd 은 mmap 에서 행 단위 직판독(t=1 경로와 동일) —
            // 종전 2.5GB to_vec 캐시를 첫 np 호출(계측 구간 내)에 만들어 agg 셀에
            // ~1s 를 삼키고 RAM 을 상주시켰다. wchk 는 mmap 뷰라 복사 불필요.
            let embd = self.model.wchk("token_embd.weight")?;
            let poss: Vec<u32> = seq_ids.iter().map(|&s| self.seqs[s].pos).collect();
            let mut rows: Vec<f32> = Vec::with_capacity(tokens.len() * n);
            for &tk in tokens {
                let mut r = vec![0.0f32; n];
                crate::quant::dequant_row(embd.ty, embd.data, tk as u64, n as u64, &mut r);
                rows.extend(r);
            }
            let lgs = rd
                .raw_step_multi(seq_ids, &poss, &rows)
                .map_err(ModelError::Accel)?;
            for s in seq_ids {
                self.seqs[*s].pos += 1;
            }
            return Ok(lgs);
        }
        let batch: Vec<Vec<u32>> = tokens.iter().map(|t| vec![*t]).collect();
        let logits = self.forward(seq_ids, &batch)?;
        for s in seq_ids {
            self.seqs[*s].pos += 1;
        }
        Ok(logits)
    }

    /// np 배치 greedy 디코드 — 토큰만 회수 (logits 전사 회피, plans/74 N1).
    /// raw np 경로가 없으면 decode+CPU greedy 폴백. LLM170_NP_GREEDY=0 게이트.
    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, ModelError> {
        if tokens.len() > 1
            && seq_ids.len() > 1
            && self.raw_decode.is_some()
            && llm170_diag::flag::ne0("LLM170_RAWHIP")
        {
            let rd = self.raw_decode.clone().unwrap();
            let n = self.model.hp.n_embd;
            // plans/92 P6: mmap 직판독(decode_greedy 단일 경로와 동일) —
            // 2.5GB to_vec 캐시 빌드를 계측 구간에 삼키지 않는다.
            let embd = self.model.wchk("token_embd.weight")?;
            let poss: Vec<u32> = seq_ids.iter().map(|&s| self.seqs[s].pos).collect();
            let mut rows: Vec<f32> = Vec::with_capacity(tokens.len() * n);
            for &tk in tokens {
                let mut r = vec![0.0f32; n];
                crate::quant::dequant_row(embd.ty, embd.data, tk as u64, n as u64, &mut r);
                rows.extend(r);
            }
            let toks = rd
                .raw_step_multi_greedy(seq_ids, &poss, &rows)
                .map_err(ModelError::Accel)?;
            for s in seq_ids {
                self.seqs[*s].pos += 1;
            }
            return Ok(toks);
        }
        let logits = self.decode(seq_ids, tokens)?;
        Ok(logits.iter().map(|l| greedy(l)).collect())
    }

    /// greedy 디코드 — GPU argmax 경로 (logits 전사 없음). raw 활성 시 유효.
    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, ModelError> {
        let Some(rd) = self.raw_decode.as_ref() else {
            // 폴백: 일반 decode + greedy
            let logits = self.decode(&[seq], &[token])?;
            return Ok(crate::qwen35::greedy(&logits[0]));
        };
        let n = self.model.hp.n_embd;
        let embd = self.model.wchk("token_embd.weight")?;
        let mut row = vec![0.0f32; n];
        crate::quant::dequant_row(embd.ty, embd.data, token as u64, n as u64, &mut row);
        let pos = self.seqs[seq].pos as usize;
        let tok = rd
            .raw_step_greedy(seq, pos, &row)
            .map_err(ModelError::Accel)?;
        self.seqs[seq].pos += 1;
        Ok(tok)
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

/// greedy argmax — `matmul::greedy_from`과 동일 의미(동률 최저인덱스).
/// 구현 중복 제거(plans/90 A1 D3): 단일 구현 재수출.
pub use crate::matmul::greedy_from as greedy;
