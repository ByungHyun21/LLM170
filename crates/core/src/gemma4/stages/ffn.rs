//! gemma4 FFN 스테이지 — dense MLP(2112·gelu_tanh) + MoE(라우터 top-8) 병렬.
//!
//! 계약(2026-10-14 HF 대조):
//! - 라우터 = **가중 없는 RMS** → `·scale·h^-0.5` → BF16 proj → softmax(전문가
//!   전체, f32) → top-8 → **재정규화(합 1)** → `·per_expert_scale[e]`.
//!   입력 = **MLP 이전 h**(어텐션 잔차) — MLP 입력과 별개.
//! - 전문가 = gelu_tanh(gate)·up → down, 선택 가중치 합산.
//! - dense(pre_ff)와 MoE(pre_ff_2)는 **별개 노름**으로 병렬 계산 후
//!   `post_ff_norm1(dense) + post_ff_norm2(moe)` 병합(호출부).

use super::super::{Model, ModelError};
use crate::matmul::cpu::{matmul, matmul_batch};
use crate::ops::{gelu_tanh, rms_norm_plain};

/// 라우터 — (전문가, 가중) 내림차순. 동률 = 낮은 인덱스(결정적).
/// `h` = 라우터 입력(MLP 이전 잔차 h) 행들.
pub(crate) fn route(
    model: &Model,
    il: usize,
    h: &[Vec<f32>],
) -> Result<Vec<Vec<(usize, f32)>>, ModelError> {
    let hp = &model.hp;
    let n_exp = hp.n_experts;
    let top_k = hp.top_k;
    let scale = model.f32_vec(&format!("blk.{il}.moe_scale"))?;
    let pes = model.f32_vec(&format!("blk.{il}.moe_per_expert_scale"))?;
    let w = model.wchk(&format!("blk.{il}.moe_gate.weight"))?;
    assert_eq!(w.n_out as usize, n_exp, "라우터 출력 = 전문가 수");
    let inv = 1.0 / (hp.n_embd as f32).sqrt();
    let mut out = Vec::with_capacity(h.len());
    let mut logits = vec![0.0f32; n_exp];
    for x in h {
        let normed = rms_norm_plain(x, hp.eps);
        let r: Vec<f32> = normed
            .iter()
            .zip(scale.iter())
            .map(|(v, s)| v * s * inv)
            .collect();
        matmul(&r, &w, &mut logits);
        // softmax(f32·전체) → top-k → 재정규화 → per_expert_scale.
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32; n_exp];
        let mut sum = 0.0f32;
        for (e, pv) in p.iter_mut().enumerate() {
            *pv = crate::ops::exp_cr(logits[e] - mx);
            sum += *pv;
        }
        for pv in p.iter_mut() {
            *pv /= sum;
        }
        let kk = top_k.min(n_exp);
        let mut idx: Vec<usize> = (0..n_exp).collect();
        let mut ord = |&a: &usize, &b: &usize| p[b].total_cmp(&p[a]).then(a.cmp(&b));
        idx.select_nth_unstable_by(kk - 1, &mut ord);
        idx.truncate(kk);
        idx.sort_unstable_by(&mut ord);
        let wsum: f32 = idx.iter().map(|&e| p[e]).sum();
        out.push(
            idx.iter()
                .map(|&e| (e, p[e] / wsum * pes[e]))
                .collect::<Vec<_>>(),
        );
    }
    Ok(out)
}

/// dense MLP — out[t] = down(gelu_tanh(gate(n_t))·up(n_t)). `x` = pre_ff 노름 행.
pub(crate) fn dense_mlp(
    model: &Model,
    il: usize,
    x: &[Vec<f32>],
    out: &mut [Vec<f32>],
) -> Result<(), ModelError> {
    let hp = &model.hp;
    let n_tok = x.len();
    let n_ff = hp.n_ff;
    assert_eq!(out.len(), n_tok);
    let gate_w = model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
    let up_w = model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
    let down_w = model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
    let mut g = vec![vec![0.0f32; n_ff]; n_tok];
    let mut u = vec![vec![0.0f32; n_ff]; n_tok];
    matmul_batch(x, &gate_w, &mut g);
    matmul_batch(x, &up_w, &mut u);
    for (gi, ui) in g.iter_mut().zip(u.iter()) {
        for i in 0..n_ff {
            gi[i] = gelu_tanh(gi[i]) * ui[i];
        }
    }
    matmul_batch(&g, &down_w, out);
    Ok(())
}

/// MoE — `sel`(route 결과) 기준 전문가별 토큰 묶음 계산 → 가중 합산.
/// `x` = pre_ff_2 노름 행(라우터 입력과 별개 계약).
pub(crate) fn moe_block(
    model: &Model,
    il: usize,
    x: &[Vec<f32>],
    sel: &[Vec<(usize, f32)>],
    out: &mut [Vec<f32>],
) -> Result<(), ModelError> {
    let hp = &model.hp;
    let n_tok = x.len();
    let n_embd = hp.n_embd;
    let n_ff = hp.moe_ffn;
    assert_eq!(out.len(), n_tok);
    for row in out.iter_mut() {
        row.fill(0.0);
    }
    // 선택 전문가별 토큰 묶음 — 행 디양자화 1회 상각.
    let mut by_exp: Vec<Vec<(usize, f32)>> = vec![Vec::new(); hp.n_experts];
    for (t, s) in sel.iter().enumerate() {
        for &(e, w) in s {
            by_exp[e].push((t, w));
        }
    }
    for (e, toks) in by_exp.iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        let xs: Vec<&[f32]> = toks.iter().map(|&(t, _)| x[t].as_slice()).collect();
        let gw = model
            .expert_w(il, e, "gate_proj")
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{il}.expert{e}.gate_proj")))?;
        let uw = model
            .expert_w(il, e, "up_proj")
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{il}.expert{e}.up_proj")))?;
        let mut g = vec![vec![0.0f32; n_ff]; toks.len()];
        let mut u = vec![vec![0.0f32; n_ff]; toks.len()];
        crate::matmul::cpu::matmul_batch_ref(&xs, &gw, &mut g);
        crate::matmul::cpu::matmul_batch_ref(&xs, &uw, &mut u);
        for (gi, ui) in g.iter_mut().zip(u.iter()) {
            for i in 0..n_ff {
                gi[i] = gelu_tanh(gi[i]) * ui[i];
            }
        }
        let dw = model
            .expert_w(il, e, "down_proj")
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{il}.expert{e}.down_proj")))?;
        let mut d = vec![vec![0.0f32; n_embd]; toks.len()];
        matmul_batch(&g, &dw, &mut d);
        for (ti, &(t, w)) in toks.iter().enumerate() {
            for i in 0..n_embd {
                out[t][i] += w * d[ti][i];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ops::gelu_tanh;

    /// gelu_tanh = HF `gelu_pytorch_tanh` 근사 — torch 참조값 고정(2026-10-14 실측).
    #[test]
    fn gelu_tanh_reference_values() {
        let cases = [
            (0.0f32, 0.0f32),
            (1.0, 0.841_192_0),
            (-1.0, -0.158_808_0),
            (2.5, 2.484_915_7),
            (-3.0, -0.003_637_4),
            (0.5, 0.345_714_0),
            (-0.25, -0.100_324_6),
        ];
        for (x, want) in cases {
            let got = gelu_tanh(x);
            assert!(
                (got - want).abs() < 1e-5,
                "gelu_tanh({x}) = {got} != {want}"
            );
        }
    }
}
