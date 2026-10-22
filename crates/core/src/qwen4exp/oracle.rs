
use super::engine::Model;
use super::stages::{hc, linear, moe, ple, qsa};
use crate::matmul::greedy_from;
use std::path::Path;

enum LayerSt {
    Lin {
        conv: Vec<Vec<f32>>,
        s: Vec<f32>,
        qkv_n: usize,
    },
    Full {
        kv: qsa::KvCache,
        idx: qsa::QsaCache,
    },
}

pub const GOLDEN_PROMPT: &[u32] = &[156037, 16072, 154029, 209495, 30];

pub struct Oracle {
    pub m: Model,
    layers: Vec<LayerSt>,
    ple: Option<ple::PleState>,
    ple_il: Option<usize>,
    pos: usize,
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

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn forward(&mut self, chunk: &[u32]) -> Result<Vec<f32>, String> {
        if chunk.is_empty() {
            return Err("빈 청크".into());
        }
        let base = self.pos;
        let cfg = &self.m.cfg;
        let (hc_n, hidden, n_layers) = (cfg.hc_count, cfg.hidden_size, cfg.num_hidden_layers);
        let xs = self.m.embed(chunk)?;
        let mut r: Vec<Vec<f32>> = xs.iter().map(|e| e.repeat(hc_n)).collect();
        self.hist.extend_from_slice(chunk);
        if llm170_diag::dump::opts().key("debug_layers") {
            let mx = r[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("fn[cpu] init R0max={mx:.4}");
        }
        for il in 0..n_layers {
            if Some(il) == self.ple_il {
                let ple_st = self
                    .ple
                    .as_mut()
                    .ok_or_else(|| format!("PLE 상태 부재(층 {il})"))?;
                ple::forward(&self.m, il, &self.hist, base, &mut r, ple_st)?;
            }
            let lp = format!("model.language_model.layers.{il}");
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
                let f16s = r[0]
                    .iter()
                    .take(12)
                    .map(|x| format!("{x:.4}"))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("fn[cpu] layer {il:>2} y0max={ym:.4} attn R0max={mx:.4} r0[..12]={f16s}");
            }
            if il == 0 && base == 0 && llm170_diag::dump::opts().key("r_full") {
                let mut buf = Vec::with_capacity(r.len() * hc_n * hidden * 4);
                for row in &r {
                    for v in row {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                let _ = std::fs::write("/tmp/cpu_r0.bin", &buf);
            }
            let (block2, inj2) = hc::mix(&self.m, &format!("{lp}.mlp_hyper_connection"), &r)?;
            let ffn = moe::forward(&self.m, il, &block2)?;
            if llm170_diag::dump::opts().key("debug_layers") {
                let xm = block2[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let ym = ffn[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let f12: String = block2[0]
                    .iter()
                    .take(12)
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("fn[cpu] layer {il:>2} x2max={xm:.4} ymmax={ym:.4} x2[..12]={f12}");
            }
            for t in 0..r.len() {
                hc::combine_tok(
                    &mut r[t],
                    &ffn[t],
                    &inj2[t * hc_n..(t + 1) * hc_n],
                    hc_n,
                    hidden,
                );
            }
            if llm170_diag::dump::opts().key("debug_layers") {
                let mx = r[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = r[0].iter().any(|v| v.is_nan());
                eprintln!("fn[cpu] layer {il:>2} R0max={mx:.4} nan={nan}");
            }
        }
        self.pos += chunk.len();
        let sample = hc::mix_final(&self.m, &r)?;
        self.m.head(&sample[sample.len() - 1])
    }

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

    #[test]
    fn fn_oracle_equiv_real_model() {
        let Some(d) = llm170_diag::flag::val("LLM170_FN_MODEL") else {
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

    #[test]
    fn fn_golden_real_model() {
        let Some(d) = llm170_diag::flag::val("LLM170_FN_MODEL") else {
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
