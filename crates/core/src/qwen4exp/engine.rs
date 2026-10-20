//! [FN 2단계 2026-10-17] qwen4_exp(FN) CPU 오라클 엔진 골격 — 실행 레이어
//! 그래프. **스토어 직접 접근**(StArchive + 샤드 mmap) — FN 체크포인트는
//! auto_round:auto_gptq 팩(qweight/qzeros/scales)이라 W4a16Model
//! (compressed-tensors weight_packed 계약)로 열 수 없다.
//!
//! 2차 마일스톤 계약(골격+판독 검증):
//! - 하이브리드 층 분기: linear_attn 36 / self_attn(full) 12(interval 4).
//! - linear층 = GDN 계열(Qwen3-Next GatedDeltaNet 미러): in_proj_qkv/a/b/z
//!   · conv1d(k=4) · A_log/dt_bias · norm[128] · out_proj — 코어는
//!   `crate::gdn`(qwen35와 동일 형상 계약: h_k=16·h_v=48·d=128) 재사용.
//! - full층 = Q‖gate 퓨전 q_proj[12288] + q/k_norm[256] + GQA 24Q/2KV
//!   + RoPE(n_rot 64·base 1e7 — qwen35와 동일 계약).
//! - MoE: 라우터 BF16[512] top-10(전체 softmax·재정규화 — qwen35
//!   select_topk 재사용) + gptq int4 전문가 + shared(sigmoid 게이트).
//! - head·embed: embed_tokens/lm_head BF16 직접.
//!
//! **미구현 지점(3차 — todo가 아니라 명시적 Err + 문서)**:
//! ① hyper-connection(4스트림 믹서·hc_norm[10240]·block_inject) — 프로브는
//!   단순 잔차(x += out) 근사. ② PLE FP8 ngram 임베딩 디코드 + indexer
//!   top-k 경로. ③ gptq 팩 nibble 순서·zero(+1) 바이어스의 골든 검증.
//! ④ 상태ful 디코드(KV/GDN 상태 재사용)·골든 토큰열. ⑤ MTP n-gram 폴백.

use super::bind::Qwen4ExpCfg;
use crate::st::{StArchive, StDtype, StEntry};
use std::path::Path;

/// gptq 그룹 크기(FN 실측 — qzeros [k/128, n/8]·scales [k/128, n]).
const GPTQ_GROUP: usize = 128;

pub struct Model {
    pub cfg: Qwen4ExpCfg,
    ar: StArchive,
    mmaps: Vec<memmap2::Mmap>,
    /// rms_norm_eps.
    pub eps: f32,
}

impl Model {
    /// 디렉터리 열기 — config + 아카이브 헤더 + 샤드 mmap.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = Qwen4ExpCfg::load(dir)?;
        let ar = StArchive::open(dir).map_err(|e| format!("safetensors: {e}"))?;
        let mut mmaps = Vec::new();
        for p in ar.shard_paths() {
            let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            // SAFETY: 읽기 전용 매핑 — 수정하지 않는다(w4a16 mmap 동일 계약).
            mmaps.push(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? });
        }
        Ok(Model {
            cfg,
            ar,
            mmaps,
            eps: 1e-6,
        })
    }

    pub(crate) fn ent(&self, name: &str) -> Result<&StEntry, String> {
        self.ar
            .entry(name)
            .ok_or_else(|| format!("텐서 부재: {name}"))
    }

    /// 텐서 원시 슬라이스(mmap 무카피).
    pub(crate) fn slice(&self, name: &str) -> Result<&[u8], String> {
        let e = self.ent(name)?;
        let base = self
            .ar
            .shard_data_base(e.shard)
            .ok_or_else(|| format!("샤드 좌표 부재: {name}"))?;
        let m = self
            .mmaps
            .get(e.shard)
            .ok_or_else(|| format!("샤드 mmap 부재: {name}"))?;
        Ok(&m[(base + e.begin) as usize..(base + e.end) as usize])
    }

    /// 1-D BF16/F16 텐서 → f32 벡터.
    pub(crate) fn f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let e = self.ent(name)?;
        let d = self.slice(name)?;
        let n = e.nbytes() as usize / e.dtype.nbytes() as usize;
        let mut v = vec![0.0f32; n];
        match e.dtype {
            StDtype::Bf16 => {
                for i in 0..n {
                    v[i] = f32::from_bits(
                        u32::from(u16::from_le_bytes(d[2 * i..2 * i + 2].try_into().unwrap()))
                            << 16,
                    );
                }
            }
            StDtype::F16 => {
                for i in 0..n {
                    v[i] = half_f32(u16::from_le_bytes(d[2 * i..2 * i + 2].try_into().unwrap()));
                }
            }
            other => return Err(format!("{name}: 1-D f32 변환 불가 dtype {other:?}")),
        }
        Ok(v)
    }

    /// zero-centered(Gemma식) 노름 가중치 판독 — f32_vec + 1 합성.
    /// **FN norm+1 계약의 단일 지점**(qwen35 로더 `Eng::Synth{plus1}`의
    /// StArchive 등가 — FN은 W4a16Model 스토어가 아니라 지연 mmap 판독
    ///이라 가중치 합성은 읽기 시점 이 한 곳에서만 일어난다).
    /// 근거: 논문 §2.1.1 "consistently applied to all other RMSNorm layers"
    /// ·vLLM GemmaRMSNorm/GroupedGemmaRMSNorm(`normalized*(1+w)` —
    /// qsa.py q/k_norm·indexer_qsa.py q/k_layernorm·ple_layer.py
    /// Qwen4ExpPLEGroupedNorm·hyperconnection.py hc_norm 전부).
    /// 대상: self_attn.{q,k}_norm·mlp_hyper_connection.hc_norm·ple.
    /// norm_{conv,key,query}·indexer.{q,k}_layernorm·
    /// hyper_connection_mixer.hc_norm. **예외 — linear_attn.norm은 +1 금지**:
    /// vLLM은 RMSNormGated(plain `x_normed*weight`, layernorm.py
    /// forward_static)이고 체크포인트 실측 평균 ≈0.97(원시 w≈0인
    /// zero-centered 분포가 아님) — qwen3_next 계약 그대로 평범 가중치.
    pub(crate) fn norm_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let mut v = self.f32_vec(name)?;
        plus1(&mut v);
        Ok(v)
    }

    /// [n_out, n_in] BF16 행렬 × x → y(스트리밍 디양자화 — 사본 없음).
    pub(crate) fn matvec_bf16(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let e = self.ent(name)?;
        if e.dtype != StDtype::Bf16 || e.shape.len() != 2 {
            return Err(format!(
                "{name}: [n,k] BF16 계약 위반 {:?} {:?}",
                e.dtype, e.shape
            ));
        }
        let (n, k) = (e.shape[0] as usize, e.shape[1] as usize);
        if k != x.len() {
            return Err(format!("{name}: k={k} ≠ 입력 {}]", x.len()));
        }
        let d = self.slice(name)?;
        let mut y = vec![0.0f32; n];
        for (r, yr) in y.iter_mut().enumerate() {
            let row = &d[2 * r * k..2 * (r + 1) * k];
            let mut acc = 0.0f64;
            for (j, &xv) in x.iter().enumerate() {
                let b = u16::from_le_bytes([row[2 * j], row[2 * j + 1]]);
                acc += (f32::from_bits((b as u32) << 16)) as f64 * xv as f64;
            }
            *yr = acc as f32;
        }
        Ok(y)
    }

    /// BF16 임베딩 행 1개 → f32.
    pub(crate) fn embed_row(&self, name: &str, r: usize) -> Result<Vec<f32>, String> {
        let e = self.ent(name)?;
        let k = *e.shape.last().ok_or("임베딩 shape 부재")? as usize;
        let d = self.slice(name)?;
        let row = &d[2 * r * k..2 * (r + 1) * k];
        Ok((0..k)
            .map(|j| {
                f32::from_bits((u16::from_le_bytes([row[2 * j], row[2 * j + 1]]) as u32) << 16)
            })
            .collect())
    }

    /// gptq int4 전문가 행 디양자화 — base는 `.qweight` 접미사 전 이름.
    /// 레이아웃(FN 실측): qweight I32 [k/8, n] · qzeros I32 [k/g, n/8]
    /// · scales F16 [k/g, n] · g=128. auto_gptq 팩 계약: 니블은 입력축
    /// 8개 팩(LE), zero = 언팩값 **+ 1**(vLLM q_gemm.cu `zero_offset=1`
    /// — classic auto_gptq 팩. [D1] 전수 스캔: FN qzeros 전 니블=7
    /// 상수 → zero=8, sym. CT weight_packed zp=8과 동일 계약).
    pub(crate) fn expert_row(&self, base: &str, r: usize) -> Result<Vec<f32>, String> {
        let (qw, qz, sc, n, k) = self.expert_triple(base)?;
        let g = GPTQ_GROUP;
        let mut row = vec![0.0f32; k];
        for (j, v) in row.iter_mut().enumerate() {
            let word = u32::from_le_bytes(
                qw[4 * ((j / 8) * n + r)..4 * ((j / 8) * n + r) + 4]
                    .try_into()
                    .unwrap(),
            );
            let nib = ((word >> (4 * (j % 8))) & 0xF) as i32;
            let gi = j / g;
            let zw = u32::from_le_bytes(
                qz[4 * (gi * (n / 8) + r / 8)..4 * (gi * (n / 8) + r / 8) + 4]
                    .try_into()
                    .unwrap(),
            );
            let zero = (((zw >> (4 * (r % 8))) & 0xF) as i32) + 1;
            let sb = 2 * (gi * n + r);
            let s = half_f32(u16::from_le_bytes([sc[sb], sc[sb + 1]]));
            *v = (nib - zero) as f32 * s;
        }
        Ok(row)
    }

    /// gptq 트리플 슬라이스 + 형상 검증 → (qweight, qzeros, scales, n, k).
    fn expert_triple(&self, base: &str) -> Result<(&[u8], &[u8], &[u8], usize, usize), String> {
        let eq = self.ent(&format!("{base}.qweight"))?;
        let ez = self.ent(&format!("{base}.qzeros"))?;
        let es = self.ent(&format!("{base}.scales"))?;
        if eq.dtype != StDtype::I32 || ez.dtype != StDtype::I32 || es.dtype != StDtype::F16 {
            return Err(format!("{base}: gptq 트리플 dtype 위반"));
        }
        let (n, k) = (eq.shape[1] as usize, eq.shape[0] as usize * 8);
        if ez.shape != [k.div_ceil(GPTQ_GROUP) as u64, (n / 8) as u64]
            || es.shape != [k.div_ceil(GPTQ_GROUP) as u64, n as u64]
        {
            return Err(format!(
                "{base}: gptq 트리플 형상 위반 qw={:?} qz={:?} sc={:?} (n={n},k={k})",
                eq.shape, ez.shape, es.shape
            ));
        }
        Ok((
            self.slice(&format!("{base}.qweight"))?,
            self.slice(&format!("{base}.qzeros"))?,
            self.slice(&format!("{base}.scales"))?,
            n,
            k,
        ))
    }

    /// 전문가 프로젝션 전체 행 디양자화([n,k] f32 — 프로브 소형 전용).
    pub(crate) fn expert_mat(&self, base: &str) -> Result<(Vec<Vec<f32>>, usize, usize), String> {
        let (_, _, _, n, k) = self.expert_triple(base)?;
        let mut rows = Vec::with_capacity(n);
        for r in 0..n {
            rows.push(self.expert_row(base, r)?);
        }
        Ok((rows, n, k))
    }

    /// 층 타입 판정 — layer_types 실측 우선(config load에서 이미 반영).
    pub fn is_linear(&self, il: usize) -> bool {
        (il % self.cfg.full_interval.max(1)) != self.cfg.full_interval.max(1) - 1
    }

    /// 임베딩 → hidden.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        tokens
            .iter()
            .map(|&t| {
                if t as usize >= self.cfg.vocab {
                    Err(format!("토큰 {t} ≥ vocab {}", self.cfg.vocab))
                } else {
                    self.embed_row("model.language_model.embed_tokens.weight", t as usize)
                }
            })
            .collect()
    }

    /// 마지막 hidden → 로짓(lm_head [vocab, hidden] 스트리밍 matvec).
    pub fn head(&self, h: &[f32]) -> Result<Vec<f32>, String> {
        self.matvec_bf16("lm_head.weight", h)
    }
}

/// 노름 가중치 zero-centered 합성(w → 1+w) — `norm_vec` 전용(단일 지점).
pub(crate) fn plus1(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x += 1.0;
    }
}

/// F16 → f32(참조 경로 — 정확도 우선).
pub(crate) fn half_f32(b: u16) -> f32 {
    let s: f32 = if b >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1F) as i32;
    let m = (b & 0x3FF) as f32;
    match e {
        0 => s * m * 2.0f32.powi(-24),
        31 => f32::NAN * s,
        _ => s * (1.0 + m / 1024.0) * 2.0f32.powi(e - 15),
    }
}

/// hyper-connection(4스트림 잔차 믹서) — **미구현(3차)**.
/// 텐서: 층별 attn/mlp_hyper_connection(input_mix up/down [10240,320]·
/// hc_norm [10240]·block_inject [4,10240]) + 최상위 hyper_connection_mixer.
/// 프로브는 단순 잔차(x += out) 근사로 대체한다.
pub fn hyper_connection_forward(_m: &Model, _il: usize) -> Result<(), String> {
    Err("hyper-connection 4스트림 믹서 미구현 — FN 3차 (프로브는 단순 잔차 근사)".into())
}

/// 그래프 프로브 — 구조 검증 + linear/full 1층분 forward(형상·dtype 체크)
/// + PLE·indexer·hyper-connection 미구현 지점 명시. w4a16-ref 진입점.
pub fn probe(dir: &Path) -> Result<String, String> {
    let m = Model::load(dir)?;
    let cfg = &m.cfg;
    // ① 구조 검증 — 텐서 실측 층 분포 vs config.
    let mut n_lin = 0;
    let mut n_ful = 0;
    let mut ple_layers = Vec::new();
    for il in 0..cfg.num_hidden_layers {
        let has_lin = m
            .ent(&format!(
                "model.language_model.layers.{il}.linear_attn.A_log"
            ))
            .is_ok();
        let has_ful = m
            .ent(&format!(
                "model.language_model.layers.{il}.self_attn.q_proj.weight"
            ))
            .is_ok();
        if has_lin == has_ful {
            return Err(format!("층 {il}: linear/full 어텐션 텐서 이중/부재"));
        }
        n_lin += has_lin as usize;
        n_ful += has_ful as usize;
        if m.ent(&format!(
            "model.language_model.layers.{il}.ple.conv1d.weight"
        ))
        .is_ok()
        {
            ple_layers.push(il);
        }
    }
    if (n_lin, n_ful) != (cfg.n_linear, cfg.n_full) {
        return Err(format!(
            "층 분포 실측 ({n_lin}, {n_ful}) ≠ config ({}, {})",
            cfg.n_linear, cfg.n_full
        ));
    }
    // 전문가 수 실측(층 0 기준).
    let n_exp_real = (0..cfg.n_experts)
        .filter(|&e| {
            m.ent(&format!(
                "model.language_model.layers.0.mlp.experts.{e}.gate_proj.qweight"
            ))
            .is_ok()
        })
        .count();
    if n_exp_real != cfg.n_experts {
        return Err(format!("전문가 실측 {n_exp_real}/{}", cfg.n_experts));
    }

    // ② 실행 — 임베딩 → (linear층 0 · full층 3) 1층분 forward.
    let tokens: Vec<u32> = [9707, 11, 728, 264].to_vec();
    let mut xs = m.embed(&tokens)?;
    let mut lines = vec![format!(
        "fn-probe {} (CPU 오라클 골격)\n  구조: linear {} / full {} (interval {}) · PLE 층 {:?} · 전문가 {n_exp_real} top-{} · shared {}",
        dir.display(),
        cfg.n_linear,
        cfg.n_full,
        cfg.full_interval,
        ple_layers,
        cfg.top_k,
        cfg.shared_ffn
    )];
    for il in [0usize, 3usize] {
        let attn_out = if m.is_linear(il) {
            super::stages::linear::forward(&m, il, &xs)?
        } else {
            super::stages::full::forward(&m, il, &xs)?
        };
        // 잔차(hyper-connection 근사) + MoE.
        for t in 0..xs.len() {
            for i in 0..cfg.hidden_size {
                xs[t][i] += attn_out[t][i];
            }
        }
        let ffn = super::stages::moe::forward(&m, il, &xs)?;
        for t in 0..xs.len() {
            for i in 0..cfg.hidden_size {
                xs[t][i] += ffn[t][i];
            }
        }
        let sum: f64 = xs[0].iter().map(|&v| v as f64).sum();
        let mx = xs[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        if !xs[0].iter().all(|v| v.is_finite()) {
            return Err(format!("층 {il} forward 비유한값"));
        }
        lines.push(format!(
            "  층 {il} ({}) forward 통과 — |x|max={mx:.4} sum={sum:.4}",
            if m.is_linear(il) {
                "linear/GDN"
            } else {
                "full-attn"
            }
        ));
    }
    // ③ head — 마지막 토큰 로짓 top-1.
    let logits = m.head(&xs[xs.len() - 1])?;
    let argmax = crate::matmul::greedy_from(&logits);
    let lmax = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    lines.push(format!(
        "  head 통과 — vocab {} 로짓 max={lmax:.4} top1={argmax}",
        logits.len()
    ));
    // ④ 미구현 지점(3차) — 판독 검증 + 명시적 Err.
    if let Some(&pil) = ple_layers.first() {
        let v = super::stages::ple::verify(&m, pil)?;
        let e = super::stages::ple::forward(&m, pil, &xs).unwrap_err();
        lines.push(format!(
            "  PLE 층 {pil} 판독: {v}\n  PLE forward: Err — {e}"
        ));
    }
    let ie = super::stages::full::indexer_forward(&m, 3, &xs).unwrap_err();
    let he = hyper_connection_forward(&m, 0).unwrap_err();
    lines.push(format!(
        "  indexer: Err — {ie}\n  hyper-connection: Err — {he}\n  판정: FN 2단계 골격 통과 — 3차: PLE FP8 디코드·indexer top-k·hyper-connection·gptq 팩 골든·상태ful 디코드·MTP n-gram"
    ));
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_decode() {
        assert_eq!(half_f32(0x3C00), 1.0);
        assert_eq!(half_f32(0xBC00), -1.0);
        assert_eq!(half_f32(0x4000), 2.0);
        assert_eq!(half_f32(0x0000), 0.0);
        assert!((half_f32(0x3555) - 0.33325195).abs() < 1e-6);
    }

    /// 실모델 그래프 프로브 — LLM170_FN_MODEL 환경변수로 FN 디렉터리 지정
    /// 시에만 실행(CI·회귀 무관). 서버 진입(리소스 가드)과 무관하게
    /// 골격 판독 검증을 직접 수행한다.
    #[test]
    fn fn_probe_real_model() {
        let Ok(d) = std::env::var("LLM170_FN_MODEL") else {
            return;
        };
        let rep = probe(std::path::Path::new(&d)).expect("fn-probe 실패");
        eprintln!("{rep}");
        assert!(rep.contains("골격 통과"), "리포트: {rep}");
    }

    /// [D0] zero-centered 노름 합성 — plus1 원소 단위 증명.
    #[test]
    fn norm_plus1() {
        let mut v = vec![0.0f32, -0.25, 0.5, -1.0];
        plus1(&mut v);
        assert_eq!(v, vec![1.0, 0.75, 1.5, 0.0]);
    }

    /// [D0] 실모델 norm 계약 — Gemma 노름은 norm_vec = f32_vec+1,
    /// linear_attn.norm은 평범 가중치(원시 w가 이미 ~1 분포 — +1 금지).
    #[test]
    fn fn_norm_contract_real_model() {
        let Ok(d) = std::env::var("LLM170_FN_MODEL") else {
            return;
        };
        let m = Model::load(std::path::Path::new(&d)).expect("로드 실패");
        let gemma = "model.language_model.layers.3.self_attn.q_norm.weight";
        let raw = m.f32_vec(gemma).expect("q_norm 부재");
        let syn = m.norm_vec(gemma).expect("norm_vec 실패");
        assert_eq!(raw.len(), syn.len());
        for (r, s) in raw.iter().zip(syn.iter()) {
            assert_eq!(*s, r + 1.0, "Gemma 노름은 정확히 +1");
        }
        // linear_attn.norm = 평범 가중치 — 원시 평균이 1 근처(zero-centered
        // 라면 0 근처). +1 합성 실수 방지 가드.
        let gdn = m
            .f32_vec("model.language_model.layers.0.linear_attn.norm.weight")
            .expect("linear_attn.norm 부재");
        let mean = gdn.iter().sum::<f32>() / gdn.len() as f32;
        assert!(
            (0.8..1.2).contains(&mean),
            "linear_attn.norm 평균 {mean} — 평범 가중치 계약 위반"
        );
    }

    /// [D1] gptq qzeros sym 판정 — 전문가 qzeros **전수 스캔**(mmap CPU).
    /// LLM170_FN_MODEL 지정 시에만 실행. 대상 48층×512전문가×3투영의
    /// qzeros I32 니블 전부(약 9.4억)를 히스토그램화해 단일 상수(=sym
    /// zero-point)인지 판정한다. 언팩 계약은 expert_row와 동일(LE 워드,
    /// 니블 시프트 4·i) — 원시 니블 v의 zero-point는 v−1.
    /// 함께: ① 표본 qweight 니블 분포(팩 무결성) ② expert_row 수동
    /// 디큐트 교차검증(스캔 모달 니블로 재계산해 비트일치 확인).
    #[test]
    fn fn_qzeros_sym_scan_real_model() {
        let Ok(d) = std::env::var("LLM170_FN_MODEL") else {
            return;
        };
        let m = Model::load(std::path::Path::new(&d)).expect("로드 실패");
        let mut hist = [0u64; 16];
        let mut tensors = 0u64;
        let mut const_tensors = 0u64;
        let mut const_value_hist = [0u64; 16];
        let mut layer_hist: std::collections::BTreeMap<u64, [u64; 16]> =
            std::collections::BTreeMap::new();
        let mut worst: Vec<(u64, u64, u64)> = Vec::new(); // (il, e, 상이 니블 수)
        let mut names: Vec<&String> =
            m.ar.entries()
                .keys()
                .filter(|n| n.contains(".mlp.experts.") && n.ends_with(".qzeros"))
                .collect();
        names.sort_unstable();
        assert_eq!(names.len(), 48 * 512 * 3, "전문가 qzeros 텐서 수 위반");
        for name in &names {
            let ent = m.ent(name).expect("엔트리");
            assert_eq!(ent.dtype, StDtype::I32, "{name}: dtype");
            let d = m.slice(name).expect("슬라이스");
            let mut th = [0u64; 16];
            for c in d.as_chunks::<4>().0 {
                let w = u32::from_le_bytes(*c);
                for i in 0..8 {
                    th[((w >> (4 * i)) & 0xF) as usize] += 1;
                }
            }
            let used: u64 = th.iter().filter(|&&c| c > 0).count() as u64;
            let (mode_i, &mode_c) = th.iter().enumerate().max_by_key(|(_, c)| *c).unwrap();
            let il: u64 = name
                .split('.')
                .nth(3)
                .and_then(|s| s.parse().ok())
                .expect("층 파싱");
            tensors += 1;
            if used == 1 {
                const_tensors += 1;
                const_value_hist[mode_i] += 1;
            } else {
                let ex: u64 = name
                    .split('.')
                    .nth(5)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                worst.push((il, ex, th.iter().sum::<u64>() - mode_c));
                worst.sort_by_key(|&(_, _, dv)| std::cmp::Reverse(dv));
                worst.truncate(8);
            }
            for i in 0..16 {
                hist[i] += th[i];
                layer_hist.entry(il).or_insert([0; 16])[i] += th[i];
            }
        }
        let total: u64 = hist.iter().sum();
        let mode = hist.iter().enumerate().max_by_key(|(_, c)| *c).unwrap().0;
        let off_mode: u64 = total - hist[mode];
        eprintln!(
            "qzeros 스캔: 텐서 {tensors} · 니블 {total} · 최빠값 {mode}({}) · 상이 {off_mode} ({:.6}%)",
            hist[mode],
            100.0 * off_mode as f64 / total as f64
        );
        eprintln!("전역 히스토그램: {hist:?}");
        eprintln!("상수 텐서 {const_tensors}/{tensors} — 상숫값 분포 {const_value_hist:?}");
        eprintln!("비상수 최악 8(il,e,상이수): {worst:?}");
        for &il in &[0u64, 1, 2, 23, 47] {
            if let Some(h) = layer_hist.get(&il) {
                let t: u64 = h.iter().sum();
                let md = h.iter().enumerate().max_by_key(|(_, c)| *c).unwrap().0;
                eprintln!("층 {il}: 니블 {t} · 최빠 {md} · 상이 {}", t - h[md]);
            }
        }
        assert!(
            off_mode == 0,
            "비상수 qzeros 발견 — GPTQ-asym 스토어 경로 필요"
        );

        // 표본 qweight 니블 분포(팩 무결성 — 전체 60GB는 읽지 않고 스프레드).
        for base in [
            "model.language_model.layers.0.mlp.experts.0.gate_proj",
            "model.language_model.layers.1.mlp.experts.7.down_proj",
            "model.language_model.layers.2.mlp.experts.13.up_proj",
            "model.language_model.layers.23.mlp.experts.5.gate_proj",
            "model.language_model.layers.47.mlp.experts.511.down_proj",
        ] {
            let d = m.slice(&format!("{base}.qweight")).expect("qweight");
            let mut qh = [0u64; 16];
            for c in d.as_chunks::<4>().0 {
                let w = u32::from_le_bytes(*c);
                for i in 0..8 {
                    qh[((w >> (4 * i)) & 0xF) as usize] += 1;
                }
            }
            eprintln!("qweight {base}: {qh:?}");
            assert!(qh.iter().all(|&c| c > 0), "니블 전 범위 미사용 — 팩 의심");
        }

        // 교차검증: expert_row(base, r) 원소를 상수 zero=8(스캔 니블 7
        // + vLLM zero_offset 1 — engine과 별개 경로)로 수동 재계산.
        let base = "model.language_model.layers.23.mlp.experts.5.gate_proj";
        let r = 7usize;
        let row = m.expert_row(base, r).expect("expert_row");
        let (qw, qz, sc, n, k) = m.expert_triple(base).expect("트리플");
        const ZERO: i32 = 8; // 스캔 상수(니블 7) + 1 — CT sym zp=8 동일
        for j in [0usize, 1, 63, 127, 128, k / 2, k - 1] {
            let word = u32::from_le_bytes(qw[4 * ((j / 8) * n + r)..][..4].try_into().unwrap());
            let nib = ((word >> (4 * (j % 8))) & 0xF) as i32;
            let sb = 2 * ((j / GPTQ_GROUP) * n + r);
            let s = half_f32(u16::from_le_bytes([sc[sb], sc[sb + 1]]));
            assert_eq!(row[j], (nib - ZERO) as f32 * s, "j={j} 디큐트 불일치");
        }
        // qzeros 실측 워드도 expert_row 인덱싱과 동일 위치에서 상수 확인.
        let zw = u32::from_le_bytes(qz[4 * (r / 8)..][..4].try_into().unwrap());
        assert_eq!(((zw >> (4 * (r % 8))) & 0xF) as usize, mode);
        eprintln!("교차검증 통과: zero=8(스캔 니블 7 + vLLM zero_offset 1) · expert_row 비트일치");
    }
}
