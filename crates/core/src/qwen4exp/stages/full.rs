
use super::super::engine::Model;
use crate::ops::{rms_norm, rope_head, sigmoid};

pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let l = format!("model.language_model.layers.{il}.self_attn");
    let cfg = &m.cfg;
    let (n_head, n_kv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let n_tok = xs.len();
    let q = m.ent(&format!("{l}.q_proj.weight"))?;
    let q_n = *q.shape.first().ok_or("q_proj shape 부재")? as usize;
    if q_n != n_head * 2 * hd {
        return Err(format!("q_proj n={q_n} ≠ Q‖gate {n_head}×2×{hd}"));
    }
    let q_norm_w = m.norm_vec(&format!("{l}.q_norm.weight"))?;
    let k_norm_w = m.norm_vec(&format!("{l}.k_norm.weight"))?;
    let n_rot = hd / 4;
    let kq_scale = 1.0f32 / (hd as f32).sqrt();

    let mut out = vec![vec![0.0f32; n_head * hd]; n_tok];
    let mut k_cache = vec![0.0f32; n_tok * n_kv * hd];
    let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let qg = m.matvec_bf16(&format!("{l}.q_proj.weight"), &xs[t])?;
        let kk = m.matvec_bf16(&format!("{l}.k_proj.weight"), &xs[t])?;
        let vv = m.matvec_bf16(&format!("{l}.v_proj.weight"), &xs[t])?;
        for h in 0..n_kv {
            let src = kk[h * hd..h * hd + hd].to_vec();
            let mut head = rms_norm(&src, &k_norm_w, m.eps);
            rope_head(&mut head, t as u32, n_rot, cfg_rope_base());
            let b = t * n_kv * hd + h * hd;
            k_cache[b..b + hd].copy_from_slice(&head);
        }
        v_cache.push(vv);
        let mut attn = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            let src = qg[h * 2 * hd..h * 2 * hd + hd].to_vec();
            let mut qh = rms_norm(&src, &q_norm_w, m.eps);
            rope_head(&mut qh, t as u32, n_rot, cfg_rope_base());
            let kvh = h / (n_head / n_kv);
            let mut scores = vec![0.0f32; t + 1];
            let mut mx = f32::NEG_INFINITY;
            for p in 0..=t {
                let kb = p * n_kv * hd + kvh * hd;
                let mut dp = 0.0f64;
                for i in 0..hd {
                    dp += qh[i] as f64 * k_cache[kb + i] as f64;
                }
                scores[p] = (dp as f32) * kq_scale;
                mx = mx.max(scores[p]);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let ob = h * hd;
            for (i, o) in attn[ob..ob + hd].iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (p, &sv) in scores.iter().enumerate() {
                    acc += sv as f64 * v_cache[p][kvh * hd + i] as f64;
                }
                *o = (acc / sum as f64) as f32;
            }
            let gb = h * 2 * hd + hd;
            for i in 0..hd {
                attn[ob + i] *= sigmoid(qg[gb + i]);
            }
        }
        out[t] = attn;
    }
    for t in 0..n_tok {
        out[t] = m.matvec_bf16(&format!("{l}.o_proj.weight"), &out[t])?;
    }
    Ok(out)
}

fn cfg_rope_base() -> f32 {
    1e7
}
