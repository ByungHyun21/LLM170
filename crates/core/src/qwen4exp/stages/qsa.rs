
use super::super::engine::Model;
use crate::ops::{rms_norm, rope_head, sigmoid};

pub struct QsaCache {
    pub raw_k: Vec<Vec<f32>>,
    pub comp_k: Vec<Vec<f32>>,
}

impl QsaCache {
    pub fn new() -> Self {
        QsaCache {
            raw_k: Vec::new(),
            comp_k: Vec::new(),
        }
    }
}

impl Default for QsaCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new() -> Self {
        KvCache {
            k: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

fn indexer_project(
    m: &Model,
    l: &str,
    x: &[f32],
    pos: u32,
    q_ln: &[f32],
) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
    let (n_ih, ihd, n_rot) = (
        m.cfg.indexer_n_heads,
        m.cfg.indexer_head_dim,
        m.cfg.indexer_head_dim / 2,
    );
    let qk = m.matvec_bf16(&format!("{l}.indexer.index_qk_proj.weight"), x)?;
    let mut q = Vec::with_capacity(n_ih);
    for h in 0..n_ih {
        let b = h * ihd;
        let mut head = rms_norm(&qk[b..b + ihd], q_ln, m.eps);
        rope_head(&mut head, pos, n_rot, 1e7);
        q.push(head);
    }
    let k = qk[n_ih * ihd..n_ih * ihd + ihd].to_vec();
    Ok((q, k))
}

pub fn select(
    idx: &QsaCache,
    q: &[Vec<f32>],
    p: usize,
    ratio: usize,
    block_topk: usize,
) -> Vec<usize> {
    let visible = (p + 1) / ratio;
    if visible == 0 {
        return (0..=p).collect();
    }
    let ihd = q[0].len();
    let mut scores = Vec::with_capacity(visible);
    for j in 0..visible {
        let kb = &idx.comp_k[j];
        let mut s = 0.0f32;
        for qh in q {
            let mut dot = 0.0f64;
            for i in 0..ihd {
                dot += qh[i] as f64 * kb[i] as f64;
            }
            s += (dot as f32).max(0.0);
        }
        scores.push(s / (ihd as f32).sqrt());
    }
    let mut order: Vec<usize> = (0..visible).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(block_topk);
    let mut toks: Vec<usize> = Vec::with_capacity(block_topk * ratio + ratio);
    for &b in &order {
        toks.extend(b * ratio..b * ratio + ratio);
    }
    toks.extend(visible * ratio..=p);
    toks
}

pub fn full_forward(
    m: &Model,
    il: usize,
    xs: &[Vec<f32>],
    base_pos: usize,
    kv: &mut KvCache,
    idx: &mut QsaCache,
) -> Result<Vec<Vec<f32>>, String> {
    let cfg = &m.cfg;
    let l = format!("model.language_model.layers.{il}.self_attn");
    let (n_head, n_kv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let ratio = cfg.indexer_compress_ratio.max(1);
    let block_topk = cfg.indexer_budget / ratio;
    let q_norm_w = m.norm_vec(&format!("{l}.q_norm.weight"))?;
    let k_norm_w = m.norm_vec(&format!("{l}.k_norm.weight"))?;
    let q_ln = m.norm_vec(&format!("{l}.indexer.q_layernorm.weight"))?;
    let k_ln = m.norm_vec(&format!("{l}.indexer.k_layernorm.weight"))?;
    let i_n_rot = cfg.indexer_head_dim / 2;
    let n_rot = hd / 4;
    let kq_scale = 1.0f32 / (hd as f32).sqrt();
    let n_tok = xs.len();

    let mut out = vec![vec![0.0f32; n_head * hd]; n_tok];
    for t in 0..n_tok {
        let p = base_pos + t;
        let kk = m.matvec_bf16(&format!("{l}.k_proj.weight"), &xs[t])?;
        let vv = m.matvec_bf16(&format!("{l}.v_proj.weight"), &xs[t])?;
        let mut krow = vec![0.0f32; n_kv * hd];
        for h in 0..n_kv {
            let mut head = rms_norm(&kk[h * hd..h * hd + hd], &k_norm_w, m.eps);
            rope_head(&mut head, p as u32, n_rot, 1e7);
            krow[h * hd..h * hd + hd].copy_from_slice(&head);
        }
        kv.k.push(krow);
        kv.v.push(vv);
        let (iq, ik) = indexer_project(m, &l, &xs[t], p as u32, &q_ln)?;
        idx.raw_k.push(ik);
        if (p + 1).is_multiple_of(ratio) {
            let ihd = cfg.indexer_head_dim;
            let mut pooled = vec![0.0f32; ihd];
            for off in 0..ratio {
                let src = &idx.raw_k[idx.raw_k.len() - ratio + off];
                for (d, v) in pooled.iter_mut().enumerate() {
                    *v += src[d];
                }
            }
            for v in pooled.iter_mut() {
                *v /= ratio as f32;
            }
            let mut nk = rms_norm(&pooled, &k_ln, m.eps);
            rope_head(&mut nk, (p + 1 - ratio) as u32, i_n_rot, 1e7);
            idx.comp_k.push(nk);
        }
        let sel = select(idx, &iq, p, ratio, block_topk);
        let qg = m.matvec_bf16(&format!("{l}.q_proj.weight"), &xs[t])?;
        for h in 0..n_head {
            let qb = h * 2 * hd;
            let mut qh = rms_norm(&qg[qb..qb + hd], &q_norm_w, m.eps);
            rope_head(&mut qh, p as u32, n_rot, 1e7);
            let kvh = h / (n_head / n_kv);
            let mut scores = Vec::with_capacity(sel.len());
            let mut mx = f32::NEG_INFINITY;
            for &s in &sel {
                let kb = kvh * hd;
                let mut dp = 0.0f64;
                for i in 0..hd {
                    dp += qh[i] as f64 * kv.k[s][kb + i] as f64;
                }
                let sc = (dp as f32) * kq_scale;
                scores.push(sc);
                mx = mx.max(sc);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let ob = h * hd;
            for i in 0..hd {
                let mut acc = 0.0f64;
                for (j, &s) in scores.iter().enumerate() {
                    acc += s as f64 * kv.v[sel[j]][kvh * hd + i] as f64;
                }
                out[t][ob + i] = (acc / sum as f64) as f32;
            }
            for i in 0..hd {
                out[t][ob + i] *= sigmoid(qg[qb + hd + i]);
            }
        }
    }
    for t in 0..n_tok {
        out[t] = m.matvec_bf16(&format!("{l}.o_proj.weight"), &out[t])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_math() {
        let ihd = 4usize;
        let mut idx = QsaCache::new();
        for j in 0..3usize {
            let mut k = vec![0.0f32; ihd];
            k[j % ihd] = 1.0;
            idx.comp_k.push(k);
        }
        let q = vec![vec![0.0f32, 0.0, 1.0, 0.0]];
        let ratio = 4;
        let sel = select(&idx, &q, 11, ratio, 1);
        assert_eq!(sel, vec![8, 9, 10, 11], "visible 3·top1=블록2");
        let selb = select(&idx, &q, 10, ratio, 1);
        assert_eq!(selb, vec![0, 1, 2, 3, 8, 9, 10], "동점→블록0·꼬리 8..10");
        let sel2 = select(&idx, &q, 7, ratio, 2);
        assert_eq!(sel2, vec![0, 1, 2, 3, 4, 5, 6, 7], "visible 2·top2 전체");
        let sel3 = select(&idx, &q, 3, ratio, 1);
        assert_eq!(sel3, vec![0, 1, 2, 3]);
    }
}
