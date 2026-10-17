//! MoE FFN 스테이지 — 512전문가 top-10 + shared(gemma4/qwen35 MoE 미러).
//!
//! 텐서 계약(FN 실측): gate(라우터) BF16 [512,2560] · 전문가 gptq int4
//! 트리플(gate/up [640,2560] · down [2560,640], g128) · shared_expert
//! gate/up/down BF16 + shared_expert_gate [1,2560].
//! 라우팅 = 전체 softmax → top-k → 재정규화(qwen35 select_topk 재사용 —
//! vllm Qwen3NextSparseMoeBlock 시맨틱).

use super::super::engine::Model;
use crate::ops::{sigmoid, silu};
use crate::qwen35::stages::moe::select_topk;

pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let cfg = &m.cfg;
    let l = format!("model.language_model.layers.{il}.mlp");
    let n_exp = cfg.n_experts;
    let g = m.ent(&format!("{l}.gate.weight"))?;
    if g.shape != [n_exp as u64, cfg.hidden_size as u64] {
        return Err(format!(
            "mlp.gate {:?} ≠ [{n_exp},{}]",
            g.shape, cfg.hidden_size
        ));
    }
    let mut out = vec![vec![0.0f32; cfg.hidden_size]; xs.len()];
    for (t, x) in xs.iter().enumerate() {
        let logits = m.matvec_bf16(&format!("{l}.gate.weight"), x)?;
        let sel = select_topk(&logits, cfg.top_k);
        // 선택 전문가 — silu(gate)⊙up → down, 라우터 가중 가산.
        for &(e, w) in &sel {
            let gw = m.expert_mat(&format!("{l}.experts.{e}.gate_proj"))?;
            let uw = m.expert_mat(&format!("{l}.experts.{e}.up_proj"))?;
            let mut h = vec![0.0f32; gw.1];
            for (r, row) in gw.0.iter().enumerate() {
                h[r] = silu(dot(row, x)) * dot(&uw.0[r], x);
            }
            let dw = m.expert_mat(&format!("{l}.experts.{e}.down_proj"))?;
            for (r, row) in dw.0.iter().enumerate() {
                out[t][r] += w * dot(row, &h);
            }
        }
        // shared expert — sigmoid(shared_gate)·silu(gate)⊙up → down.
        let sgate = m.matvec_bf16(&format!("{l}.shared_expert_gate.weight"), x)?[0];
        let sgate = sigmoid(sgate);
        let s_up = m.matvec_bf16(&format!("{l}.shared_expert.gate_proj.weight"), x)?;
        let s_val = m.matvec_bf16(&format!("{l}.shared_expert.up_proj.weight"), x)?;
        let mut h = vec![0.0f32; s_up.len()];
        for (r, (&a, &b)) in s_up.iter().zip(s_val.iter()).enumerate() {
            h[r] = silu(a) * b;
        }
        let s_out = m.matvec_bf16(&format!("{l}.shared_expert.down_proj.weight"), &h)?;
        for (o, v) in out[t].iter_mut().zip(s_out.iter()) {
            *o += sgate * v;
        }
    }
    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x as f64 * y as f64;
    }
    acc as f32
}
