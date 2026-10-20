//! [D2 2026-10-20] FN 상태ful CPU 오라클 — 전 48층 forward 체인.
//!
//! vLLM amd/model.py Qwen4ExpModel.forward 전수(단일 시퀀스):
//! - 초기 R = embed × hc_count 반복([T,2560] → [T,10240]).
//! - 층 루프: (PLE층이면 R += ple 출력 — 어텐션 mix **앞** 직접 가산)
//!   → attn_hc.mix → 어텐션(linear=GDN 상태ful / full=QSA 선택 집합)
//!   → combine(R += y⊙2σ(inj/4)) → mlp_hc.mix → MoE → combine.
//!   (AMD 지연 결합과 동일 수학 — 오라클은 즉시 결합.)
//! - 최종: 마지막 mlp combine 후 `hyper_connection_mixer`(use_combine
//!   =False) mix → 샘플 스트림 [2560] → lm_head. 최종 norm 없음.
//!
//! 상태 축 4종: ① full층 KV(12) + 인덱서 캐시(원시 K·압축 키)
//! ② GDN 상태(36) + conv 히스토리 ③ PLE conv 상태(9×10240)
//! ④ 전체 토큰열(PLE n-gram 선행 참조). 접두 체크포인트 = 이들 그대로.

use super::engine::Model;
use super::stages::{hc, linear, moe, ple, qsa};
use crate::matmul::greedy_from;
use std::path::Path;

/// 층별 상태.
enum LayerSt {
    /// linear(GDN)층 — conv 히스토리(k−1 qkv 행)·GDN 상태·qkv 폭.
    Lin {
        conv: Vec<Vec<f32>>,
        s: Vec<f32>,
        qkv_n: usize,
    },
    /// full(QSA)층 — 본 KV + 인덱서 캐시.
    Full {
        kv: qsa::KvCache,
        idx: qsa::QsaCache,
    },
}

/// 골든 프롬프트(swap_models.sh GOLDEN 스크립트 패턴 — 고정 토큰열).
pub const GOLDEN_PROMPT: &[u32] = &[156037, 16072, 154029, 209495, 30];

/// 상태ful 오라클 — prefill/디코드 같은 `forward`(청크 길이만 다름).
pub struct Oracle {
    pub m: Model,
    layers: Vec<LayerSt>,
    ple: Option<ple::PleState>,
    ple_il: Option<usize>,
    /// 소비 위치(다음 청크 첫 토큰 절대 위치).
    pos: usize,
    /// 전체 토큰열(PLE n-gram 선행·기록).
    hist: Vec<u32>,
}

impl Oracle {
    pub fn new(dir: &Path) -> Result<Self, String> {
        let m = Model::load(dir)?;
        let cfg = &m.cfg;
        let d = cfg.linear_head_dim;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for il in 0..cfg.num_hidden_layers {
            if m.is_linear(il) {
                let qkv_n = m
                    .ent(&format!(
                        "model.language_model.layers.{il}.linear_attn.in_proj_qkv.weight"
                    ))?
                    .shape[0] as usize;
                layers.push(LayerSt::Lin {
                    conv: Vec::new(),
                    s: vec![0.0f32; cfg.linear_value_heads * d * d],
                    qkv_n,
                });
            } else {
                layers.push(LayerSt::Full {
                    kv: qsa::KvCache::new(),
                    idx: qsa::QsaCache::new(),
                });
            }
        }
        // PLE 층 — 체크포인트 실물 스캔(config ple_layer_ids는 1-based라
        // 1 차이 — 프로브와 동일한 판별).
        let ple_il = (0..cfg.num_hidden_layers).find(|&il| {
            m.ent(&format!(
                "model.language_model.layers.{il}.ple.conv1d.weight"
            ))
            .is_ok()
        });
        let ple_state = ple_il.map(|_| ple::PleState::new(cfg.hc_count * cfg.hidden_size));
        Ok(Oracle {
            m,
            layers,
            ple: ple_state,
            ple_il,
            pos: 0,
            hist: Vec::new(),
        })
    }

    /// 소비 위치(그간 프리필+디코드 토큰 수).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 청크 forward — 마지막 토큰 로짓 반환. prefill(≥1토큰)·디코드
    /// (1토큰) 동일 경로(상태 축 연속).
    pub fn forward(&mut self, chunk: &[u32]) -> Result<Vec<f32>, String> {
        if chunk.is_empty() {
            return Err("빈 청크".into());
        }
        let base = self.pos;
        let cfg = &self.m.cfg;
        let (hc_n, hidden, n_layers) = (cfg.hc_count, cfg.hidden_size, cfg.num_hidden_layers);
        // 초기 멀티스트림 상태 — embed ×hc 반복.
        let xs = self.m.embed(chunk)?;
        let mut r: Vec<Vec<f32>> = xs.iter().map(|e| e.repeat(hc_n)).collect();
        self.hist.extend_from_slice(chunk);
        if llm170_diag::dump::opts().key("debug_layers") {
            let mx = r[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("fn[cpu] init R0max={mx:.4}");
        }
        for il in 0..n_layers {
            // PLE — 어텐션 HC mix 앞에서 멀티스트림 상태에 직접 가산.
            if Some(il) == self.ple_il {
                let ple_st = self
                    .ple
                    .as_mut()
                    .ok_or_else(|| format!("PLE 상태 부재(층 {il})"))?;
                ple::forward(&self.m, il, &self.hist, base, &mut r, ple_st)?;
            }
            let lp = format!("model.language_model.layers.{il}");
            // 어텐션 블록.
            let (block, inj) = hc::mix(&self.m, &format!("{lp}.attn_hyper_connection"), &r)?;
            let attn_out = match &mut self.layers[il] {
                LayerSt::Lin { conv, s, qkv_n } => {
                    linear::forward_state(&self.m, il, &block, conv, s, *qkv_n)?
                }
                LayerSt::Full { kv, idx } => qsa::full_forward(&self.m, il, &block, base, kv, idx)?,
            };
            for t in 0..r.len() {
                hc::combine_tok(
                    &mut r[t],
                    &attn_out[t],
                    &inj[t * hc_n..(t + 1) * hc_n],
                    hc_n,
                    hidden,
                );
            }
            if llm170_diag::dump::opts().key("debug_layers") {
                let ym = attn_out[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mx = r[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                eprintln!("fn[cpu] layer {il:>2} y0max={ym:.4} attn R0max={mx:.4}");
            }
            // MoE 블록.
            let (block2, inj2) = hc::mix(&self.m, &format!("{lp}.mlp_hyper_connection"), &r)?;
            let ffn = moe::forward(&self.m, il, &block2)?;
            for t in 0..r.len() {
                hc::combine_tok(
                    &mut r[t],
                    &ffn[t],
                    &inj2[t * hc_n..(t + 1) * hc_n],
                    hc_n,
                    hidden,
                );
            }
            // [디버그 배선 2026-10-20] 층별 지문 — R 행 0 max|x|(GPU fn[gpu]와
            // 동일 계약). LLM170_DUMP=debug_layers.
            if llm170_diag::dump::opts().key("debug_layers") {
                let mx = r[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = r[0].iter().any(|v| v.is_nan());
                eprintln!("fn[cpu] layer {il:>2} R0max={mx:.4} nan={nan}");
            }
        }
        self.pos += chunk.len();
        // 최종 mixer(use_combine=False) — hc_norm+mix → lm_head 입력.
        let sample = hc::mix_final(&self.m, &r)?;
        self.m.head(&sample[sample.len() - 1])
    }

    /// greedy — 프롬프트 prefill → 시드 1토큰 + `n_predict` 디코드
    /// (총 n_predict+1 토큰. w4a16-ref 참조 러너 계약과 동일).
    pub fn greedy(&mut self, prompt: &[u32], n_predict: usize) -> Result<Vec<u32>, String> {
        let mut logits = self.forward(prompt)?;
        let mut out = vec![greedy_from(&logits)];
        for _ in 0..n_predict {
            logits = self.forward(&[out[out.len() - 1]])?;
            out.push(greedy_from(&logits));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [D2 자기일관성] 상태ful prefill ≡ 토큰별 재계산 — 5토큰 프롬프트를
    /// 1회 프리필한 마지막 로짓 vs 2+1+1+1 청크 분할 프리필의 로짓.
    /// GDN 청크(수학 동일·부동 소수 경로 차)·나머지 경로 원소 동치 —
    /// 허용오차 1e-3 + argmax 일치. LLM170_FN_MODEL 지정 시에만 실행.
    #[test]
    fn fn_oracle_equiv_real_model() {
        let Ok(d) = std::env::var("LLM170_FN_MODEL") else {
            return;
        };
        let p = GOLDEN_PROMPT;
        let mut a = Oracle::new(std::path::Path::new(&d)).expect("오라클 A 로드");
        let la = a.forward(p).expect("일괄 프리필");
        let mut b = Oracle::new(std::path::Path::new(&d)).expect("오라클 B 로드");
        let lb = b
            .forward(&p[..2])
            .and_then(|_| b.forward(&p[2..3]))
            .and_then(|_| b.forward(&p[3..4]))
            .and_then(|_| b.forward(&p[4..5]))
            .expect("분할 프리필");
        let mut worst = 0.0f32;
        for (x, y) in la.iter().zip(lb.iter()) {
            worst = worst.max((x - y).abs());
        }
        eprintln!(
            "prefill↔재계산: max|Δlogit| = {worst:.3e} ({} 로짓)",
            la.len()
        );
        assert_eq!(greedy_from(&la), greedy_from(&lb), "argmax 불일치");
        assert!(worst < 1e-3, "로짓 편차 {worst} ≥ 1e-3");
    }

    /// [D2 골든] 고정 프롬프트 greedy 8토큰 — 동결값 대조(재현성 회귀).
    /// vLLM 대조는 다음 단계(과제 범위 밖) — 우선 자체 일관성 기준.
    #[test]
    fn fn_golden_real_model() {
        let Ok(d) = std::env::var("LLM170_FN_MODEL") else {
            return;
        };
        let mut o = Oracle::new(std::path::Path::new(&d)).expect("오라클 로드");
        let toks = o.greedy(GOLDEN_PROMPT, 8).expect("greedy");
        eprintln!("FN 골든 프롬프트 {:?} → {:?}", GOLDEN_PROMPT, toks);
        assert_eq!(
            toks,
            vec![2977u32, 198, 198, 248044, 67, 2423, 271, 16436, 604],
            "골든 토큰열 변경 — 골든 재동결 필요(swap_models GOLDEN 규율)"
        );
    }
}
