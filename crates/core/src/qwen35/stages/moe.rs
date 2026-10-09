//! MoE FFN 스테이지(W4-1, 35B-A3B) — 라우터 top-k 소프트맥스 + 전문가 가중합
//! + shared_expert(sigmoid 게이트).
//!
//! 시맨틱 기준(vllm qwen3_next — Qwen3NextSparseMoeBlock): 라우터 로짓에
//! **전문가 전체 softmax** → top-k → 재정규화(norm_topk_prob 기본 True),
//! shared = sigmoid(shared_expert_gate·x)·MLP(gate/up/down, silu).
//! 전문가는 스토어 트리플 슬라이스에서 직접 구성(Model::expert_w — 30k
//! 이름맵 무경유). 선택 전문가별 토큰 묶음으로 행 디양자화 1회 상각.

use super::Ctx;
use crate::ops::silu;
use crate::qwen35::ModelError;
use crate::qwen35::dispatch::mm_group_ref;
use crate::qwen35::{mm_batch, mm_group};

/// 라우터 로짓 → softmax(전문가 전체) → top-k → 재정규화 — (전문가, 가중)
/// 내림차순. 결정적(동률 = 낮은 인덱스). backend-gpu `moe_topk`(배치 경로)가
/// 같은 시맨틱을 미러한다(크레이트 의존 방향 제약 — 변경 시 동시 갱신).
pub(crate) fn select_topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = logits.len();
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (e, pv) in p.iter_mut().enumerate() {
        *pv = (logits[e] - mx).exp();
        sum += *pv;
    }
    for pv in p.iter_mut() {
        *pv /= sum;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let wsum: f32 = idx.iter().map(|&e| p[e]).sum();
    idx.iter().map(|&e| (e, p[e] / wsum)).collect()
}

/// MoE FFN — out[t] = Σ_i w_i·E_i(x_t) + sigmoid(sgate·x_t)·S(x_t).
/// out은 덮어쓴다(잔차 가산은 호출부 — dense FFN과 동일 계약).
pub(crate) fn moe_ffn(
    ctx: &Ctx,
    il: usize,
    x: &[Vec<f32>],
    out: &mut [Vec<f32>],
) -> Result<(), ModelError> {
    let m = ctx.model;
    let hp = &m.hp;
    let n_tok = x.len();
    let n_embd = hp.n_embd;
    let n_exp = hp.n_experts;
    let top_k = hp.top_k;
    let n_ff = hp.moe_ffn;
    assert_eq!(out.len(), n_tok);

    // 라우터 로짓 [n_tok][n_exp] — 플레인 가중(35B = BF16).
    let gate_w = m.wchk(&format!("blk.{il}.moe_gate.weight"))?;
    assert_eq!(gate_w.n_out as usize, n_exp, "라우터 출력 = 전문가 수");
    let mut logits = vec![vec![0.0f32; n_exp]; n_tok];
    mm_batch(x, &gate_w, &mut logits);

    let mut sel: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tok);
    for l in logits.iter() {
        sel.push(select_topk(l, top_k));
    }

    // 전문가 디스패치 — 선택된 전문가별 토큰 묶음.
    let mut by_exp: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_exp];
    for (t, s) in sel.iter().enumerate() {
        for &(e, w) in s {
            by_exp[e].push((t, w));
        }
    }
    let mut moe_out = vec![vec![0.0f32; n_embd]; n_tok];
    for (e, toks) in by_exp.iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        // [P12 E3] 참조 수집 — 종전 행 clone(프리필 1024토큰에서 층당 수십 MB).
        let xs: Vec<&[f32]> = toks.iter().map(|&(t, _)| x[t].as_slice()).collect();
        let gw = m
            .expert_w(il, e, "gate_proj")
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{il}.expert{e}.gate_proj")))?;
        let uw = m
            .expert_w(il, e, "up_proj")
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{il}.expert{e}.up_proj")))?;
        let mut outs = [
            vec![vec![0.0f32; n_ff]; toks.len()],
            vec![vec![0.0f32; n_ff]; toks.len()],
        ];
        mm_group_ref(&xs, &[gw, uw], &mut outs);
        let [mut g, u] = outs;
        for (gi, ui) in g.iter_mut().zip(u.iter()) {
            for i in 0..n_ff {
                gi[i] = silu(gi[i]) * ui[i];
            }
        }
        let dw = m
            .expert_w(il, e, "down_proj")
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{il}.expert{e}.down_proj")))?;
        let mut d = vec![vec![0.0f32; n_embd]; toks.len()];
        mm_batch(&g, &dw, &mut d); // g는 소유 행(게이트 활성) — 래퍼가 참조 수집.
        for (ti, &(t, w)) in toks.iter().enumerate() {
            for i in 0..n_embd {
                moe_out[t][i] += w * d[ti][i];
            }
        }
    }

    // shared_expert — sigmoid(sgate·x)·down(silu(gate·x)·up·x).
    if hp.shared_ffn > 0 {
        let sf = hp.shared_ffn;
        let sgw = m.wchk(&format!("blk.{il}.moe_shared_gate.weight"))?;
        let suw = m.wchk(&format!("blk.{il}.moe_shared_up.weight"))?;
        let sdw = m.wchk(&format!("blk.{il}.moe_shared_down.weight"))?;
        let sggw = m.wchk(&format!("blk.{il}.moe_shared_sgate.weight"))?;
        let mut outs = [vec![vec![0.0f32; sf]; n_tok], vec![vec![0.0f32; sf]; n_tok]];
        mm_group(x, &[sgw, suw], &mut outs);
        let [mut g, u] = outs;
        for (gi, ui) in g.iter_mut().zip(u.iter()) {
            for i in 0..sf {
                gi[i] = silu(gi[i]) * ui[i];
            }
        }
        let mut d = vec![vec![0.0f32; n_embd]; n_tok];
        mm_batch(&g, &sdw, &mut d);
        let mut sgate = vec![vec![0.0f32; 1]; n_tok];
        mm_batch(x, &sggw, &mut sgate);
        for t in 0..n_tok {
            let s = 1.0 / (1.0 + (-sgate[t][0]).exp());
            for i in 0..n_embd {
                out[t][i] = moe_out[t][i] + s * d[t][i];
            }
        }
    } else {
        for t in 0..n_tok {
            out[t].copy_from_slice(&moe_out[t]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::select_topk;

    /// 시맨틱 고정 — softmax 전체 → top-k → 재정규화(합 1), 내림차순.
    #[test]
    fn select_topk_renormalizes_and_orders() {
        let sel = select_topk(&[2.0, 1.0, 0.5, -1.0], 2);
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].0, 0);
        assert_eq!(sel[1].0, 1);
        assert!(sel[0].1 > sel[1].1, "내림차순");
        let sum: f32 = sel.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "재정규화 합={sum}");
        // 원 softmax 값 비례: p0/p1 = exp(1) 비율 유지.
        let ratio = sel[0].1 / sel[1].1;
        assert!((ratio - std::f32::consts::E).abs() < 1e-4, "비율={ratio}");
    }

    /// 동률은 낮은 인덱스 우선(결정적) — backend-gpu moe_topk 미러 계약.
    #[test]
    fn select_topk_tie_breaks_low_index() {
        let sel = select_topk(&[1.0, 1.0, 1.0], 2);
        assert_eq!(sel[0].0, 0);
        assert_eq!(sel[1].0, 1);
        let sum: f32 = sel.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
