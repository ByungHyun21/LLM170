//! 블록 조립 — hc_attn → attn → hc_ffn → ffn (model.py `Block.forward`).
//!
//! 배선: residual=X → hc_pre(hc_attn) → y → attn_norm → attn → hc_post →
//! X'; 다시 ffn 동일 구조. 잔차는 4스트림 전체, norm은 믹스 후 단일 스트림.

use crate::deepseek4::config::Deepseek4Config;
use crate::deepseek4::ops::{RopeTable, rms_norm_weighted};
use crate::deepseek4::stages::attn::{LayerAttn, attention_forward};
use crate::deepseek4::stages::hc::{HcParams, hc_post, hc_pre};
use crate::deepseek4::stages::moe::{
    ExpertWeights, GateWeights, expert_ffn, gate_scores, moe_forward,
};

/// 한 블록의 비-전문가 가중치 — 전문가는 지연 디양자화(moe_forward 콜백).
pub struct BlockWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn: HcParams,
    pub hc_ffn: HcParams,
    pub attn: LayerAttn,
    pub gate: GateWeights,
    pub shared: ExpertWeights,
    pub is_hash: bool,
}

/// 블록 프리필 — x [t×hc·d] (스트림 방송 상태) → 동일 형상.
/// `expert` 콜백: 전문가 id → 가중치(로더 지연 적재).
pub fn block_forward(
    bw: &BlockWeights,
    x: &[f32],
    input_ids: &[u32],
    cfg: &Deepseek4Config,
    rope: &RopeTable,
    mut expert: impl FnMut(usize) -> ExpertWeights,
) -> Vec<f32> {
    let (d, hc, t) = (cfg.dim, cfg.hc_mult, x.len() / (cfg.hc_mult * cfg.dim));
    // --- 어텐션 서브블록 (hc 스테이지는 토큰 단위 순수 함수) ---
    let mut y = Vec::with_capacity(t * d);
    let mut posts = Vec::with_capacity(t);
    let mut combs = Vec::with_capacity(t);
    for i in 0..t {
        let (yi, post, comb) = hc_pre(
            &x[i * hc * d..(i + 1) * hc * d],
            d,
            hc,
            &bw.hc_attn,
            cfg.rms_eps,
            cfg.hc_eps,
            cfg.hc_sinkhorn_iters,
        );
        y.extend_from_slice(&yi);
        posts.push(post);
        combs.push(comb);
    }
    let mut xn = Vec::with_capacity(t * d);
    for i in 0..t {
        xn.extend_from_slice(&rms_norm_weighted(
            &y[i * d..(i + 1) * d],
            &bw.attn_norm,
            cfg.rms_eps,
        ));
    }
    let a = attention_forward(&bw.attn, &xn, cfg, rope);
    let mut x2 = Vec::with_capacity(t * hc * d);
    for i in 0..t {
        x2.extend_from_slice(&hc_post(
            &a[i * d..(i + 1) * d],
            &x[i * hc * d..(i + 1) * hc * d],
            &posts[i],
            &combs[i],
            d,
            hc,
        ));
    }
    // --- FFN(MoE) 서브블록 ---
    let mut y2 = Vec::with_capacity(t * d);
    let mut posts2 = Vec::with_capacity(t);
    let mut combs2 = Vec::with_capacity(t);
    for i in 0..t {
        let (yi, post, comb) = hc_pre(
            &x2[i * hc * d..(i + 1) * hc * d],
            d,
            hc,
            &bw.hc_ffn,
            cfg.rms_eps,
            cfg.hc_eps,
            cfg.hc_sinkhorn_iters,
        );
        y2.extend_from_slice(&yi);
        posts2.push(post);
        combs2.push(comb);
    }
    let mut xn2 = Vec::with_capacity(t * d);
    for i in 0..t {
        xn2.extend_from_slice(&rms_norm_weighted(
            &y2[i * d..(i + 1) * d],
            &bw.ffn_norm,
            cfg.rms_eps,
        ));
    }
    let f = moe_forward(
        &xn2,
        input_ids,
        &bw.gate,
        &bw.shared,
        cfg,
        bw.is_hash,
        &mut expert,
    );
    let mut out = Vec::with_capacity(t * hc * d);
    for i in 0..t {
        out.extend_from_slice(&hc_post(
            &f[i * d..(i + 1) * d],
            &x2[i * hc * d..(i + 1) * hc * d],
            &posts2[i],
            &combs2[i],
            d,
            hc,
        ));
    }
    out
}

/// 공유 전문가만 쓰는 1토큰 FFN — 스모크/디버그용 (라우팅 없음).
pub fn shared_ffn_only(x: &[f32], shared: &ExpertWeights, cfg: &Deepseek4Config) -> Vec<f32> {
    expert_ffn(x, shared, 1.0, cfg.dim, cfg.moe_inter, cfg.swiglu_limit)
}

/// 게이트 스코어 재노출 — 프루브용.
pub fn block_gate_scores(x: &[f32], gw: &GateWeights, cfg: &Deepseek4Config) -> Vec<f32> {
    gate_scores(x, gw, cfg.n_routed, cfg.dim)
}
