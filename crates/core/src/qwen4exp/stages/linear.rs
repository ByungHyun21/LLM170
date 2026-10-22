
use super::super::engine::Model;
use crate::gdn::gdn_chunk_seq;
use crate::gdn_norm::{GdnGate, gdn_norm_gated};
use crate::ops::{l2_norm, sigmoid, silu, softplus};

pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let qkv_n = m
        .ent(&format!(
            "model.language_model.layers.{il}.linear_attn.in_proj_qkv.weight"
        ))?
        .shape[0] as usize;
    let mut st =
        vec![0.0f32; m.cfg.linear_value_heads * m.cfg.linear_head_dim * m.cfg.linear_head_dim];
    let mut hist: Vec<Vec<f32>> = Vec::new();
    forward_state(m, il, xs, &mut hist, &mut st, qkv_n)
}

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
    let conv_w = m.f32_vec(&format!("{l}.conv1d.weight"))?;
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

    let mut beta_all = vec![0.0f32; n_tok * h_v];
    let mut g_all = vec![0.0f32; n_tok * h_v];
    for t in 0..n_tok {
        for h in 0..h_v {
            beta_all[t * h_v + h] = sigmoid(b[t][h]);
            g_all[t * h_v + h] = softplus(a[t][h] + dt_bias[h]) * -a_log[h].exp();
        }
    }

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
    for row in qkv {
        conv_hist.push(row);
        if conv_hist.len() > conv_k - 1 {
            conv_hist.remove(0);
        }
    }
    for row in 0..n_tok {
        for h in 0..h_k {
            let b0 = row * k_len + h * d;
            let qh = l2_norm(&q_all[b0..b0 + d], m.eps);
            q_all[b0..b0 + d].copy_from_slice(&qh);
            let kh = l2_norm(&k_all[b0..b0 + d], m.eps);
            k_all[b0..b0 + d].copy_from_slice(&kh);
        }
    }

    if llm170_diag::dump::opts().key("debug_layers") {
        let f2 = |v: &[f32]| -> String {
            v.iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        eprintln!(
            "fn[cpu] core-in L{il} betaN={} gN={}",
            f2(&beta_all[..h_v]),
            f2(&g_all[..h_v]),
        );
    }
    let g_gva = h_v / h_k;
    let hf_of = |e: usize| g_gva * (e % h_k) + e / h_k;
    let (mut v_sub, mut beta_sub, mut g_sub, mut o_sub) = (
        vec![0.0f32; n_tok * v_len],
        vec![0.0f32; n_tok * h_v],
        vec![0.0f32; n_tok * h_v],
        vec![0.0f32; n_tok * v_len],
    );
    for t in 0..n_tok {
        for e in 0..h_v {
            let h = hf_of(e);
            beta_sub[t * h_v + e] = beta_all[t * h_v + h];
            g_sub[t * h_v + e] = g_all[t * h_v + h];
            v_sub[t * v_len + e * d..t * v_len + (e + 1) * d]
                .copy_from_slice(&v_all[t * v_len + h * d..t * v_len + (h + 1) * d]);
        }
    }
    let mut o_all = vec![0.0f32; n_tok * v_len];
    gdn_chunk_seq(
        &q_all, &k_all, &v_sub, &beta_sub, &g_sub, st, &mut o_sub, n_tok, h_k, h_v,
    );
    for t in 0..n_tok {
        for e in 0..h_v {
            let h = hf_of(e);
            o_all[t * v_len + h * d..t * v_len + (h + 1) * d]
                .copy_from_slice(&o_sub[t * v_len + e * d..t * v_len + (e + 1) * d]);
        }
    }

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
        if t == 0 && llm170_diag::dump::opts().key("debug_layers") {
            let mx = gated[0].iter().fold(0f32, |a, v| a.max(v.abs()));
            let om = o_all[..v_len].iter().fold(0f32, |a, v| a.max(v.abs()));
            eprintln!("fn[cpu] gated L{il} R0max={mx:.4} oN={om:.4}");
        }
        out.push(m.matvec_bf16(&format!("{l}.out_proj.weight"), &gated[t])?);
    }
    Ok(out)
}
