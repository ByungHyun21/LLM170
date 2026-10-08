//! qwen35 (Qwen3.8-27B 계열) CPU 참조 엔진.
//!
//! 그래프 배선: `~/local_llm/llama.cpp/src/models/qwen35.cpp` (2026-08-30 판).
//! 하이퍼파라미터(27B): n_embd 5120, FFN 17408(SwiGLU), vocab 248320,
//! rms_eps 1e-6, ctx 262144. 64층 = 12×(3×GDN→FFN + 1×GatedAttn→FFN)
//! (full_attention_interval=4 → full-attn il∈{3,7,…,63}), block_count=65.
//! - 잔차: h += attn(rms(h)); h += ffn(rms_post(h)) — FFN 분기는 **post**-attn norm
//! - GDN층(interval≠3): qkv → depthwise conv+SiLU → L2 norm(q,k) → GDN → rms_norm·silu(z) → ssm_out
//! - Full-attn층(interval==3): wq는 Q‖gate 퓨전 [5120, 12288] 헤드별 인터리브
//!   (스트라이드 2·head_dim) → per-head rms norm(q,k) [256] → RoPE half-split
//!   (n_rot 64, base 1e7 — 텍스트 토큰에서 mrope 구간 [11,11,10]과 동치)
//!   → GQA(24Q/4KV, scale 1/√256) → ⊙sigmoid(gate) → wo. KV f16 64 KiB/token(16층).
//! - 하이퍼파라미터는 config.json 메타에서 동적 로드 (소형 검증 모델 지원).
//! - f32 KV, f32 GDN 상태 (참조 정확도 우선).

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
    /// W4A16 디렉터리 로더 오류(§3.5 A안 직접 로드).
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

/// 가중치 소스 — **W4A16 디렉터리 단일**(2026-10-08).
/// 세부 접근은 `w4a16` 로더(mmap 슬라이스 + 순열 사본).
pub struct Model {
    w4: Box<crate::w4a16::W4a16Model>,
    /// qwen3_5 아키텍처 바인딩 설정(config.json).
    cfg: bind::QwenCfg,
    /// V축 순열 사본(지연 1회 구축).
    perm: std::sync::OnceLock<bind::PermStore>,
    pub hp: Hparams,
    pub token_pieces: Vec<String>,
    /// f32 norm 가중 디양자화 캐시(llama-vllm P13 계열) — 첫 호출 1회
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
    /// validate()가 전수(트리플·커버리지).
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
        // output_norm은 Synth(norm +1) — f32_vec 경유로 실재·형상 검증.
        m.f32_vec("output_norm.weight")?;
        Ok(m)
    }

    /// 무게 뷰 — W4A16 이름맵으로 슬라이스/분리버퍼 구성.
    pub fn w(&self, name: &str) -> Option<Weight<'_>> {
        w4_weight(self, name)
    }

    /// 엔진 접점 이름 전수(상주 로더/W3 배선용).
    pub fn engine_names(&self) -> Vec<String> {
        bind::engine_names(&self.cfg)
    }

    /// 임베딩 행 1개(tok) — W3 GPU 체인 입력(호스트 스테이징).
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, ModelError> {
        let Some(bind::Eng::Plain { name, .. }) = bind::eng("token_embd.weight") else {
            return Err(ModelError::MissingTensor("token_embd.weight".into()));
        };
        self.w4
            .plain_rows_f32(&name, tok as u64, tok as u64 + 1)
            .map_err(|e| ModelError::W4a16(e.to_string()))
    }

    /// MoE 전문가 트리플 슬라이스(W4-1) — 서버 GPU 전문가 테이블 구성용.
    pub fn expert_slice(
        &self,
        il: usize,
        e: usize,
        proj: &str,
    ) -> Option<(&[u8], &[u8], usize, usize)> {
        self.w4.expert_slice(il, e, proj)
    }

    /// MoE 전문가 양자 파라미터 — (group, scale_bf16) 실측.
    pub fn expert_quant(&self) -> (usize, bool) {
        let base = crate::w4a16::W4a16Model::expert_base(0, 0, "gate_proj");
        (self.w4.group(), self.w4.scale_is_bf16(&base))
    }

    /// MoE 전문가 Weight(W4-1) — 스토어 트리플에서 직접 구성(30k 이름맵 무경유).
    pub fn expert_w(&self, il: usize, e: usize, proj: &str) -> Option<Weight<'_>> {
        let base = crate::w4a16::W4a16Model::expert_base(il, e, proj);
        let (data, scale, n, k) = self.w4.expert_slice(il, e, proj)?;
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

    /// 원본(HF) 무게 — W3 GPU 체인용. GPU 커널은 순열을 내부에서
    /// 처리하므로(l2perm scatter·gate p_inv) 순열 사본을 주면 이중 순열이
    /// 된다. CPU 엔진은 순열 사본(w())을 쓰고, GPU는 이쪽을 쓴다.
    pub fn w_raw(&self, name: &str) -> Option<Weight<'_>> {
        w4_weight_raw(&self.w4, name)
    }

    /// 원본(HF) f32 벡터 — 순열·synth 보정 없이(1D/2D 모두).
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
            // 랭크 ≥2(conv1d는 [conv_ch,1,4]) — 행 원시 바이트 → f32.
            let rows = *e.shape.first().unwrap_or(&0);
            let raw = self
                .w4
                .raw_rows(&hf, 0, rows)
                .map_err(|e| ModelError::W4a16(e.to_string()))?;
            crate::w4a16::decode_f32(&raw, e.dtype, &hf)
                .map_err(|e| ModelError::W4a16(e.to_string()))
        }
    }

    /// Synth 원값(변환 없이) + 순열만 — W3 GPU 조립용.
    /// (예: A_log — f32_vec는 CPU 계약(-exp)이라 커널이 -exp를 스스로
    /// 산출하도록 원값이 필요하다.)
    pub fn synth_raw(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        match bind::eng(name) {
            Some(bind::Eng::Synth { name: hf, perm, .. }) => {
                let raw = self
                    .w4
                    .plain_vec_f32(&hf)
                    .map_err(|e| ModelError::W4a16(e.to_string()))?;
                Ok(match perm {
                    1 => bind::permute_heads_f32(&self.cfg, &raw),
                    2 => bind::conv_rows_f32_permuted(&self.w4, &self.cfg, &hf)
                        .map_err(|e| ModelError::W4a16(e.to_string()))?,
                    _ => raw,
                })
            }
            _ => Err(ModelError::W4a16(format!("{name}: Synth 아님"))),
        }
    }

    /// V축 순열 사본(지연 1회 구축 — 48 선형층 qkv-v/z/out + alpha/beta).
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
            // 정규화는 HF zero-centered(w−1 저장) — 로드 시 +1 보정.
            // ssm_a는 −exp(A_log), dt_bias/A_log·conv는 V헤드 순열 합성
            // (엔진 subhead-major 계약 — perm: 1 인덱스·2 채널).
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

/// 원본(HF) Weight 구성 — 순열 사본 비경유(W3 GPU 체인용).
/// 플레인(무양자) Weight — MoE(35B) 어텐션 투영이 BF16인 케이스의 공용 경로.
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
            // MoE(35B) 폴백: 트리플이 없고 같은 베이스의 플레인(BF16)이 있으면
            // 그쪽을 쓴다 — 스토어 내용이 진실(아치 분기 없이 동일 엔진 이름).
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

/// W4A16 소스의 Weight 구성 — Quant는 분리 버퍼(packed+scale), Plain은
/// 원시 슬라이스(ty는 dtype에서). Synth(norm류)는 None —
/// f32_vec 전용 계약(호출부가 w()로 요구하지 않는다).
fn w4_weight<'a>(m: &'a Model, name: &str) -> Option<Weight<'a>> {
    use crate::wtype::WType;
    use bind::{Eng, PV};
    let w4 = &m.w4;
    match bind::eng(name)? {
        Eng::Quant { base, vperm } => {
            // MoE(35B) 폴백 — 같은 베이스의 플레인(BF16). V축 순열은 양자화
            // 경로와 동일 규약으로 적용한다(엔진 subhead-major 계약).
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

/// 시퀀스 상태
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

    /// KV 용량에서 역산한 컨텍스트 길이.
    pub fn ctx_len(&self) -> usize {
        let (n_kv, hd) = (self.model.hp.n_kv, self.model.hp.head_dim);
        self.seqs
            .first()
            .and_then(|s| s.kv_k.first().map(|k| k.len() / (n_kv * hd)))
            .unwrap_or(4096)
    }

    /// 시퀀스 상태 전체 초기화 (무상태 HTTP 서버용).
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

    /// 슬롯 단위 상태 초기화 (연속 배칭 서버) — 해당 시퀀스의
    /// KV/GDN/conv 상태만 영점화, 다른 슬롯은 간섭 없음.
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

    /// 배치 포워드: batch[i] = seq_ids[i] 시퀀스의 토큰(행간 동일 길이).
    /// seq_ids: 배치 행 → 엔진 시퀀스 id 매핑 (prefill은 단일, decode는 활성 집합).
    fn forward(
        &mut self,
        seq_ids: &[usize],
        batch: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.forward_emb(seq_ids, batch, None)
    }

    /// forward 변형 — 임베딩 행 사전 조달 시 rows 직접 사용.
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
        for il in 0..hp.n_layer {
            let norm_w = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
            // 잔차: pre-norm 원본 보존 (qwen35.cpp:162-184 — inpSA)
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
                // MoE FFN(W4-1, 35B-A3B) — dense와 동일 계약: normed 계산 후
                // 잔차(ffn_residual) 가산은 여기서.
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

        // output_norm + logits (시퀀스별 마지막 토큰만)
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

    /// np 배치 greedy 디코드 — 토큰만 회수 (logits 전사 회피).
    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, ModelError> {
        let logits = self.decode(seq_ids, tokens)?;
        Ok(logits.iter().map(|l| greedy(l)).collect())
    }

    /// greedy 디코드 — decode + argmax.
    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, ModelError> {
        let logits = self.decode(&[seq], &[token])?;
        Ok(crate::qwen35::greedy(&logits[0]))
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
/// 구현 중복 제거: 단일 구현 재수출.
pub use crate::matmul::greedy_from as greedy;
