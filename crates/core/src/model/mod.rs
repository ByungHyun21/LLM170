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
pub mod prefill;
pub mod rawinject;
pub mod spec;
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

    /// plans/40: 가중 텐서의 mmap 페이지를 커널에 반납 (MADV_DONTNEED).
    /// GPU 상주 엔진 업로드 후 호출 — 파일 지원 클린 페이지라 즉시 회수되고,
    /// 이후 재접근 시 디스크에서 다시 읽힘. keep에 포함된 이름·4MiB 미만은 유지.
    pub fn discard_weight_pages(&self, keep: &[&str]) -> u64 {
        let mut total = 0u64;
        for t in &self.gguf.tensors {
            let name = t.name.as_str();
            let Some((start, end)) = t.file_range(self.gguf.data_offset) else { continue };
            let len = (end - start) as usize;
            if len < (4 << 20) || keep.contains(&name) {
                continue;
            }
            // madvise는 페이지 정렬 필수 — 시작을 내림, 길이 보정
            const PG: usize = 4096;
            let s_pg = (start as usize) & !(PG - 1);
            let e_pg = ((start as usize + len + PG - 1) & !(PG - 1)).min(self.mmap.len());
            if e_pg <= s_pg {
                continue;
            }
            let rc = unsafe {
                libc::madvise(
                    self.mmap.as_ptr().add(s_pg) as *mut libc::c_void,
                    e_pg - s_pg,
                    libc::MADV_DONTNEED,
                )
            };
            if rc == 0 {
                total += len as u64;
            } else {
                eprintln!("[madvise] {name}: rc={rc} err={}", std::io::Error::last_os_error());
            }
        }
        total
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
            mtp_h: vec![0.0; if has_mtp && std::env::var_os("LLM170_NOMTP").is_none() { model.hp.n_embd } else { 0 }],
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
    /// MTP 스펙 의도 — true일 때만 prefill/decode 훅 활성 (미사용 시
    /// 훅 비용으로 prefill 3배 저하 방지, 2026-09-04 계측).
    pub mtp_wanted: bool,
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
            mtp_wanted: false,
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
                            embd.ty, embd.data, batch[s][t] as u64, n_e as u64, &mut row);
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
                            let am = crate::model::greedy(&lg);
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
            let logits = if !self.seqs[seq].mtp_h.is_empty() && self.mtp_wanted {
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
        // np 배치 (rawhip) — 각 seq 1토큰, GEMM 공유 (plans/15)
        if tokens.len() > 1
            && seq_ids.len() > 1
            && self.raw_decode.is_some()
            && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
        {
            let rd = self.raw_decode.clone().unwrap();
            let n = self.model.hp.n_embd;
            if self.embd_cache.is_none() {
                let t = self.model.wchk("token_embd.weight")?;
                self.embd_cache = Some((t.ty, std::sync::Arc::new(t.data.to_vec())));
            }
            let (embd_ty, embd_arc) = self.embd_cache.as_ref().unwrap().clone();
            let poss: Vec<u32> = seq_ids.iter().map(|&s| self.seqs[s].pos).collect();
            let mut rows: Vec<f32> = Vec::with_capacity(tokens.len() * n);
            for &tk in tokens {
                let mut r = vec![0.0f32; n];
                crate::quant::dequant_row(embd_ty, &embd_arc, tk as u64, n as u64, &mut r);
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
