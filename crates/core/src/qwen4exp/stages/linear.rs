//! linear(GDN류)층 스테이지 — Qwen3-Next GatedDeltaNet 미러.
//!
//! 텐서 계약(FN 실측, 2560 hidden): in_proj_qkv [10240,2560](q 2048|k 2048|v
//! 6144 — qwen35와 동일 레이아웃) · in_proj_a/b [48,2560] · in_proj_z
//! [6144,2560] · conv1d [10240,1,4] · A_log/dt_bias [48] · norm [128]
//! · out_proj [2560,6144]. 코어 = `crate::gdn::gdn_chunk_seq`(h_k=16·h_v=48
//! ·d=128 — qwen35와 동일 형상 계약).
//!
//! 프로브 계약: 상태 0에서 1청크 forward(conv 히스토리 좌측 0패딩).
//! q/k 정규화 = l2_norm(가중치 없음 — Qwen3-Next 계약). [D0 2026-10-20]
//! 출력게이트 = sigmoid(config output_gate_type — FN 실측; 논문 Eq.11
//! o=W_o[σ(W_zx)⊙RMSNorm(y)], vLLM RMSNormGated activation 인자).
//! norm[128] = **평범 RMSNorm 가중치(+1 금지)** — vLLM RMSNormGated는
//! plain `x_normed*weight`(layernorm.py forward_static)이고 체크포인트
//! 실측 평균 ≈0.97(zero-centered 원시 w가 아님) → `f32_vec` 그대로
//! (q/k_norm·hc_norm 등 Gemma 노름만 `Model::norm_vec` +1).
//! [D2 2026-10-20] `forward_state` — conv 히스토리(k−1토큰 in_proj_qkv
//! 출력)·GDN 상태를 전달하는 상태ful 경로(오라클 디코드).

use super::super::engine::Model;
use crate::gdn::gdn_chunk_seq;
use crate::gdn_norm::{GdnGate, gdn_norm_gated};
use crate::ops::{l2_norm, sigmoid, silu, softplus};

/// 상태 0 프로브 — 상태ful `forward_state`의 1청크 래퍼.
pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let qkv_n = m.ent(&format!(
        "model.language_model.layers.{il}.linear_attn.in_proj_qkv.weight"
    ))?
    .shape[0] as usize;
    let mut st = vec![0.0f32; m.cfg.linear_value_heads * m.cfg.linear_head_dim * m.cfg.linear_head_dim];
    let mut hist: Vec<Vec<f32>> = Vec::new();
    forward_state(m, il, xs, &mut hist, &mut st, qkv_n)
}

/// [D2] 상태ful GDN forward — `conv_hist` = 직전 토큰 in_proj_qkv 출력
/// (최대 k−1개·vLLM conv 상태 계약), `st` = GDN 상태[h_v·d·d](입출).
pub fn forward_state(
    m: &Model,
    il: usize,
    xs: &[Vec<f32>],
    conv_hist: &mut Vec<Vec<f32>>,
    st: &mut [f32],
    qkv_n: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let l = format!("model.language_model.layers.{il}.linear_attn");
    let cfg = &m.cfg;
    let n_tok = xs.len();
    let h_v = cfg.linear_value_heads;
    let d = cfg.linear_head_dim;
    let v_len = h_v * d;

    // 형상·dtype 체크 — qkv 폭은 호출부 전달값과 대조.
    let eq = m.ent(&format!("{l}.in_proj_qkv.weight"))?;
    if *eq.shape.first().ok_or("in_proj_qkv shape 부재")? as usize != qkv_n {
        return Err("in_proj_qkv 폭 불일치".into());
    }
    if qkv_n <= v_len || !(qkv_n - v_len).is_multiple_of(2 * d) {
        return Err(format!(
            "in_proj_qkv n={qkv_n} — v({v_len}) 외 q/k 균분 위반"
        ));
    }
    let h_k = (qkv_n - v_len) / (2 * d);
    let k_len = h_k * d;
    let conv_k = m
        .ent(&format!("{l}.conv1d.weight"))?
        .shape
        .last()
        .copied()
        .unwrap_or(0) as usize;
    let conv_w = m.f32_vec(&format!("{l}.conv1d.weight"))?; // [ch][1][k] 행 우선
    let dt_bias = m.f32_vec(&format!("{l}.dt_bias"))?;
    let a_log = m.f32_vec(&format!("{l}.A_log"))?;
    let norm_w = m.f32_vec(&format!("{l}.norm.weight"))?;
    if norm_w.len() != d
        || dt_bias.len() != h_v
        || a_log.len() != h_v
        || conv_w.len() != qkv_n * conv_k
    {
        return Err(format!(
            "linear_attn[{il}] 형상 위반 norm={} dt={} alog={} conv={}",
            norm_w.len(),
            dt_bias.len(),
            a_log.len(),
            conv_w.len()
        ));
    }

    // 투영 — qkv·z·a·b 동일 입력.
    let qkv: Vec<Vec<f32>> = (0..n_tok)
        .map(|t| m.matvec_bf16(&format!("{l}.in_proj_qkv.weight"), &xs[t]))
        .collect::<Result<_, _>>()?;
    let z: Vec<Vec<f32>> = (0..n_tok)
        .map(|t| m.matvec_bf16(&format!("{l}.in_proj_z.weight"), &xs[t]))
        .collect::<Result<_, _>>()?;
    let a: Vec<Vec<f32>> = (0..n_tok)
        .map(|t| m.matvec_bf16(&format!("{l}.in_proj_a.weight"), &xs[t]))
        .collect::<Result<_, _>>()?;
    let b: Vec<Vec<f32>> = (0..n_tok)
        .map(|t| m.matvec_bf16(&format!("{l}.in_proj_b.weight"), &xs[t]))
        .collect::<Result<_, _>>()?;

    // 게이트 파라미터 — g = softplus(a + dt_bias)·(−exp(A_log)) (qwen35 계약).
    let mut beta_all = vec![0.0f32; n_tok * h_v];
    let mut g_all = vec![0.0f32; n_tok * h_v];
    for t in 0..n_tok {
        for h in 0..h_v {
            beta_all[t * h_v + h] = sigmoid(b[t][h]);
            g_all[t * h_v + h] = softplus(a[t][h] + dt_bias[h]) * -a_log[h].exp();
        }
    }

    // causal conv1d(청크 앞은 conv_hist — 부족분/미초기화 0) + silu → q|k|v.
    let mut q_all = vec![0.0f32; n_tok * k_len];
    let mut k_all = vec![0.0f32; n_tok * k_len];
    let mut v_all = vec![0.0f32; n_tok * v_len];
    for t in 0..n_tok {
        for c in 0..qkv_n {
            let mut acc = 0.0f32;
            for j in 0..conv_k {
                let q = t as isize + j as isize - (conv_k as isize - 1);
                let src = if q >= 0 {
                    qkv[q as usize][c]
                } else {
                    let hq = conv_hist.len() as isize + q;
                    if hq >= 0 {
                        conv_hist[hq as usize][c]
                    } else {
                        0.0
                    }
                };
                acc += conv_w[c * conv_k + j] * src;
            }
            let o = silu(acc);
            if c < k_len {
                q_all[t * k_len + c] = o;
            } else if c < 2 * k_len {
                k_all[t * k_len + c - k_len] = o;
            } else {
                v_all[t * v_len + c - 2 * k_len] = o;
            }
        }
    }
    // conv 히스토리 갱신 — 최근 k−1개의 원시 qkv.
    for row in qkv {
        conv_hist.push(row);
        if conv_hist.len() > conv_k - 1 {
            conv_hist.remove(0);
        }
    }
    // q/k per-head l2_norm.
    for row in 0..n_tok {
        for h in 0..h_k {
            let b0 = row * k_len + h * d;
            let qh = l2_norm(&q_all[b0..b0 + d], m.eps);
            q_all[b0..b0 + d].copy_from_slice(&qh);
            let kh = l2_norm(&k_all[b0..b0 + d], m.eps);
            k_all[b0..b0 + d].copy_from_slice(&kh);
        }
    }

    // GDN 코어 — 상태 입출(청크 경계 연속).
    let mut o_all = vec![0.0f32; n_tok * v_len];
    gdn_chunk_seq(
        &q_all, &k_all, &v_all, &beta_all, &g_all, st, &mut o_all, n_tok, h_k, h_v,
    );

    // norm_gated: rms_norm(o, norm[128])·게이트(config — FN=sigmoid) → out_proj.
    let gate = if cfg.output_gate_sigmoid {
        GdnGate::Sigmoid
    } else {
        GdnGate::Silu
    };
    let mut gated = vec![vec![0.0f32; v_len]; n_tok];
    gdn_norm_gated(
        gate, &o_all, &z, &norm_w, m.eps, n_tok, h_v, d, v_len, &mut gated,
    );
    let mut out = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        out.push(m.matvec_bf16(&format!("{l}.out_proj.weight"), &gated[t])?);
    }
    Ok(out)
}
