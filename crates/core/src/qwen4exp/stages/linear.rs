//! linear(GDN류)층 스테이지 — Qwen3-Next GatedDeltaNet 미러.
//!
//! 텐서 계약(FN 실측, 2560 hidden): in_proj_qkv [10240,2560](q 2048|k 2048|v
//! 6144 — qwen35와 동일 레이아웃) · in_proj_a/b [48,2560] · in_proj_z
//! [6144,2560] · conv1d [10240,1,4] · A_log/dt_bias [48] · norm [128]
//! · out_proj [2560,6144]. 코어 = `crate::gdn::gdn_chunk_seq`(h_k=16·h_v=48
//! ·d=128 — qwen35와 동일 형상 계약).
//!
//! 프로브 계약: 상태 0에서 1청크 forward(conv 히스토리 좌측 0패딩).
//! q/k 정규화 = l2_norm(가중치 없음 — Qwen3-Next 계약), 출력 게이팅 =
//! rms_norm(norm[128])·silu(z). 상태ful 디코드는 3차.

use super::super::engine::Model;
use crate::gdn::gdn_chunk_seq;
use crate::gdn_norm::{GdnGate, gdn_norm_gated};
use crate::ops::{l2_norm, sigmoid, silu, softplus};

pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let l = format!("model.language_model.layers.{il}.linear_attn");
    let cfg = &m.cfg;
    let n_tok = xs.len();
    let h_v = cfg.linear_value_heads;
    let d = cfg.linear_head_dim;
    let v_len = h_v * d;

    // 형상·dtype 체크 — qkv 폭에서 q/k 헤드 수 유도.
    let eq = m.ent(&format!("{l}.in_proj_qkv.weight"))?;
    let qkv_n = *eq.shape.first().ok_or("in_proj_qkv shape 부재")? as usize;
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

    // causal conv1d(좌측 0패딩) + silu → q|k|v 분할.
    let mut q_all = vec![0.0f32; n_tok * k_len];
    let mut k_all = vec![0.0f32; n_tok * k_len];
    let mut v_all = vec![0.0f32; n_tok * v_len];
    for t in 0..n_tok {
        for c in 0..qkv_n {
            let mut acc = 0.0f32;
            for j in 0..conv_k {
                let hist = t + j >= conv_k - 1; // x[t + j - (conv_k-1)]
                if hist {
                    let src = qkv[t + j - (conv_k - 1)][c];
                    acc += conv_w[c * conv_k + j] * src;
                }
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

    // GDN 코어 — 상태 0 프로브(1청크).
    let mut st = vec![0.0f32; h_v * d * d];
    let mut o_all = vec![0.0f32; n_tok * v_len];
    gdn_chunk_seq(
        &q_all, &k_all, &v_all, &beta_all, &g_all, &mut st, &mut o_all, n_tok, h_k, h_v,
    );

    // norm_gated: rms_norm(o, norm[128])·silu(z) → out_proj.
    let mut gated = vec![vec![0.0f32; v_len]; n_tok];
    gdn_norm_gated(
        GdnGate::Silu,
        &o_all,
        &z,
        &norm_w,
        m.eps,
        n_tok,
        h_v,
        d,
        v_len,
        &mut gated,
    );
    let mut out = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        out.push(m.matvec_bf16(&format!("{l}.out_proj.weight"), &gated[t])?);
    }
    Ok(out)
}
