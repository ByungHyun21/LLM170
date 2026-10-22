
use super::super::engine::Model;
use crate::ops::{rms_norm, sigmoid, silu};

pub struct PleState {
    pub conv: Vec<Vec<f32>>,
    pub has_init: bool,
}

impl PleState {
    pub fn new(ch: usize) -> Self {
        PleState {
            conv: vec![vec![0.0f32; ch]; conv_state_len()],
            has_init: false,
        }
    }
}

pub fn conv_state_len() -> usize {
    9
}

#[allow(clippy::doc_lazy_continuation)]
pub fn gather_emb(m: &Model, hist: &[u32], base_pos: usize) -> Result<Vec<Vec<f32>>, String> {
    let cfg = &m.cfg;
    let n_tok = hist.len() - base_pos;
    let emb_dim = cfg.ple_embed_dim;
    let ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    let head_dim = emb_dim / ngram_heads;
    let eb = "model.language_model.layers.1.ple.ple_embedding".to_string();
    let mult = i64_vec(m, &format!("{eb}.layer_multipliers"))?;
    let sizes = i64_vec(m, &format!("{eb}.ngram_heads_vocab_sizes"))?;
    let offsets = i64_vec(m, &format!("{eb}.ngram_heads_offsets"))?;
    let scale = m.f32_vec(&format!("{eb}.ngram_embedding.weight_scale"))?[0];
    let shard_rows = m
        .ent(&format!("{eb}.ngram_embedding.shard_0.weight"))?
        .shape[0] as usize;
    let eos = cfg.eos_token_id;
    let mut out = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let ids = ngram_ids(
            hist,
            base_pos + t,
            &mult,
            &sizes,
            &offsets,
            eos,
            cfg.heads_per_ngram,
        );
        let mut emb = vec![0.0f32; emb_dim];
        for (h, &id) in ids.iter().enumerate() {
            ple_row(
                m,
                &eb,
                shard_rows,
                id,
                scale,
                &mut emb[h * head_dim..(h + 1) * head_dim],
            )?;
        }
        out.push(emb);
    }
    Ok(out)
}

pub fn forward(
    m: &Model,
    il: usize,
    hist: &[u32],
    base_pos: usize,
    r: &mut [Vec<f32>],
    st: &mut PleState,
) -> Result<(), String> {
    let cfg = &m.cfg;
    let (hc, hidden) = (cfg.hc_count, cfg.hidden_size);
    let l = format!("model.language_model.layers.{il}.ple");
    let n_tok = r.len();
    let emb_dim = cfg.ple_embed_dim;
    let ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    let head_dim = emb_dim / ngram_heads;

    let eb = format!("{l}.ple_embedding");
    let mult = i64_vec(m, &format!("{eb}.layer_multipliers"))?;
    let sizes = i64_vec(m, &format!("{eb}.ngram_heads_vocab_sizes"))?;
    let offsets = i64_vec(m, &format!("{eb}.ngram_heads_offsets"))?;
    if mult.len() != cfg.ngram_size || sizes.len() != ngram_heads || offsets.len() != ngram_heads {
        return Err(format!(
            "PLE[{il}] ngram 버퍼 위반 mult={} sizes={} offsets={} (ngram_size {}·heads {ngram_heads})",
            mult.len(),
            sizes.len(),
            offsets.len(),
            cfg.ngram_size
        ));
    }
    let scale = m.f32_vec(&format!("{eb}.ngram_embedding.weight_scale"))?[0];
    let shard_rows = m
        .ent(&format!("{eb}.ngram_embedding.shard_0.weight"))?
        .shape[0] as usize;
    let conv_w = m.f32_vec(&format!("{l}.conv1d.weight"))?;
    let nk = m.norm_vec(&format!("{l}.norm_key.weight"))?;
    let nq = m.norm_vec(&format!("{l}.norm_query.weight"))?;
    let nc = m.norm_vec(&format!("{l}.norm_conv.weight"))?;

    let eos = cfg.eos_token_id;
    let mut gated_all = vec![vec![0.0f32; hc * hidden]; n_tok];
    let mut normed_all = vec![vec![0.0f32; hc * hidden]; n_tok];
    for t in 0..n_tok {
        let ids = ngram_ids(
            hist,
            base_pos + t,
            &mult,
            &sizes,
            &offsets,
            eos,
            cfg.heads_per_ngram,
        );
        let mut emb = vec![0.0f32; emb_dim];
        for (h, &id) in ids.iter().enumerate() {
            ple_row(
                m,
                &eb,
                shard_rows,
                id,
                scale,
                &mut emb[h * head_dim..(h + 1) * head_dim],
            )?;
        }
        let key = m.matvec_bf16(&format!("{l}.key_proj.weight"), &emb)?;
        let value = m.matvec_bf16(&format!("{l}.value_proj.weight"), &emb)?;
        for s in 0..hc {
            let b = s * hidden;
            let k_n = rms_norm(&key[b..b + hidden], &nk[b..b + hidden], m.eps);
            let q_n = rms_norm(&r[t][b..b + hidden], &nq[b..b + hidden], m.eps);
            let mut dot = 0.0f64;
            for i in 0..hidden {
                dot += k_n[i] as f64 * q_n[i] as f64;
            }
            let d = (dot as f32) / (hidden as f32).sqrt();
            let sign = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            let magnitude = d.abs().max(1e-6).sqrt();
            let g = sigmoid(sign * magnitude);
            let gated = &mut gated_all[t][b..b + hidden];
            for i in 0..hidden {
                gated[i] = g * value[i];
            }
            let normed = rms_norm(gated, &nc[b..b + hidden], m.eps);
            normed_all[t][b..b + hidden].copy_from_slice(&normed);
        }
    }

    let ch = hc * hidden;
    let wlen = conv_state_len();
    let conv_y = |t: usize, c: usize| -> f32 {
        let mut acc = 0.0f32;
        for k in 0..4usize {
            let q = (base_pos + t + 3 * k) as isize - 9;
            let tap: f32 = if q < 0 {
                0.0
            } else if (q as usize) < base_pos {
                if st.has_init {
                    st.conv[q as usize + wlen - base_pos][c]
                } else {
                    0.0
                }
            } else {
                normed_all[q as usize - base_pos][c]
            };
            acc += conv_w[c * 4 + k] * tap;
        }
        acc
    };
    let mut y = vec![vec![0.0f32; ch]; n_tok];
    for t in 0..n_tok {
        for c in 0..ch {
            y[t][c] = conv_y(t, c);
        }
    }
    for t in 0..n_tok {
        for c in 0..ch {
            r[t][c] += gated_all[t][c] + silu(y[t][c]);
        }
    }
    let t_end = base_pos + n_tok;
    let mut new_st = vec![vec![0.0f32; ch]; wlen];
    for i in 0..wlen {
        let q = t_end as isize + i as isize - 9;
        if q < 0 {
            continue;
        }
        let q = q as usize;
        if q >= base_pos {
            new_st[i].copy_from_slice(&normed_all[q - base_pos]);
        } else if st.has_init {
            new_st[i].copy_from_slice(&st.conv[q + wlen - base_pos]);
        }
    }
    st.conv = new_st;
    st.has_init = true;
    Ok(())
}

pub fn ngram_ids(
    hist: &[u32],
    pos: usize,
    mult: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    eos: u32,
    heads_per: usize,
) -> Vec<i64> {
    let ngram_ctx = mult.len() - 1;
    let n_heads = sizes.len();
    let tok = hist[pos] as i64;
    let mut mixed = vec![tok.wrapping_mul(mult[0]); n_heads];
    let mut crossed = false;
    for shift in 1..=ngram_ctx {
        let mut cand = if pos >= shift {
            hist[pos - shift] as i64
        } else {
            eos as i64
        };
        if crossed {
            cand = eos as i64;
        }
        crossed = crossed || cand == eos as i64;
        let term = cand.wrapping_mul(mult[shift]);
        for (h, m) in mixed.iter_mut().enumerate() {
            let order = h / heads_per + 2;
            if order > shift {
                *m ^= term;
            }
        }
    }
    mixed
        .iter()
        .enumerate()
        .map(|(h, &m)| m.rem_euclid(sizes[h]) + offsets[h])
        .collect()
}

pub fn e4m3_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    match e {
        0 => s * (m / 8.0) * 2f32.powi(-6),
        15 if m == 7.0 => f32::NAN * s,
        _ => s * (1.0 + m / 8.0) * 2f32.powi(e - 7),
    }
}

fn ple_row(
    m: &Model,
    eb: &str,
    shard_rows: usize,
    id: i64,
    scale: f32,
    out: &mut [f32],
) -> Result<(), String> {
    let si = (id / shard_rows as i64) as usize;
    let row = (id % shard_rows as i64) as u64;
    let name = format!("{eb}.ngram_embedding.shard_{si}.weight");
    let e = m.ent(&name)?;
    if row >= e.shape[0] {
        return Err(format!("PLE id {id} — 샤드 {si} 행 {row} ≥ {}", e.shape[0]));
    }
    let d = m.slice(&name)?;
    let base = (row as usize) * out.len();
    for (j, o) in out.iter_mut().enumerate() {
        *o = e4m3_f32(d[base + j]) * scale;
    }
    Ok(())
}

pub fn verify(m: &Model, il: usize) -> Result<String, String> {
    let l = format!("model.language_model.layers.{il}.ple");
    let mut rep = Vec::new();
    for p in ["conv1d.weight", "key_proj.weight", "value_proj.weight"] {
        let e = m.ent(&format!("{l}.{p}"))?;
        rep.push(format!("{p}={:?}", e.shape));
    }
    for p in ["norm_conv.weight", "norm_key.weight", "norm_query.weight"] {
        let e = m.ent(&format!("{l}.{p}"))?;
        rep.push(format!("{p}=[{}]", e.shape.first().unwrap_or(&0)));
    }
    let eb = format!("{l}.ple_embedding");
    let e0 = m.ent(&format!("{eb}.ngram_embedding.shard_0.weight"))?;
    let n_shards = (0..m.cfg.split_ngram_parts)
        .filter(|&s| {
            m.ent(&format!("{eb}.ngram_embedding.shard_{s}.weight"))
                .is_ok()
        })
        .count();
    let offs = i64_vec(m, &format!("{eb}.ngram_heads_offsets"))?;
    let vocabs = i64_vec(m, &format!("{eb}.ngram_heads_vocab_sizes"))?;
    let mult = i64_vec(m, &format!("{eb}.layer_multipliers"))?;
    m.ent(&format!("{eb}.ngram_embedding.weight_scale"))?;
    if n_shards != m.cfg.split_ngram_parts {
        return Err(format!(
            "ngram shard {n_shards}/{}",
            m.cfg.split_ngram_parts
        ));
    }
    if offs.len() != m.cfg.heads_per_ngram * 2 || vocabs.len() != m.cfg.heads_per_ngram * 2 {
        return Err(format!(
            "ngram offsets {}·vocabs {} — heads_per_ngram {}×2 계약 위반",
            offs.len(),
            vocabs.len(),
            m.cfg.heads_per_ngram
        ));
    }
    rep.push(format!(
        "shards {n_shards}×{:?} {dt:?} · offsets {} · vocab합 {} · multipliers {:?}",
        e0.shape,
        offs.len(),
        vocabs.iter().sum::<i64>(),
        mult,
        dt = e0.dtype,
    ));
    Ok(rep.join(" · "))
}

fn i64_vec(m: &Model, name: &str) -> Result<Vec<i64>, String> {
    let d = m.slice(name)?;
    Ok(d.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_decode() {
        assert_eq!(e4m3_f32(0x00), 0.0);
        assert_eq!(e4m3_f32(0x80), -0.0);
        assert_eq!(e4m3_f32(0x40), 2.0);
        assert_eq!(e4m3_f32(0xC0), -2.0);
        assert_eq!(e4m3_f32(0x30), 0.5);
        assert_eq!(e4m3_f32(0x38), 1.0);
        assert_eq!(e4m3_f32(0x3C), 1.5);
        assert_eq!(e4m3_f32(0x7E), 448.0);
        assert_eq!(e4m3_f32(0xFE), -448.0);
        assert!(e4m3_f32(0x7F).is_nan());
        assert!(e4m3_f32(0xFF).is_nan());
        assert_eq!(e4m3_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_f32(0x07), 0.875 * 2f32.powi(-6));
        assert_eq!(e4m3_f32(0x08), 2f32.powi(-6));
    }

    #[test]
    fn ngram_id_math() {
        let mult = vec![7i64, 11, 13];
        let sizes: Vec<i64> = (0..16).map(|h| 1_000_003 + h as i64).collect();
        let offsets: Vec<i64> = (0..16).map(|h| h as i64 * 100).collect();
        let eos = 999u32;
        let hist = vec![5u32, 11, 3, 999, 8, 2];
        let ids = ngram_ids(&hist, 2, &mult, &sizes, &offsets, eos, 8);
        let base2 = 3i64.wrapping_mul(7) ^ 11i64.wrapping_mul(11) ^ 5i64.wrapping_mul(13);
        let tri2 = 3i64.wrapping_mul(7) ^ 11i64.wrapping_mul(11);
        for h in 0..16 {
            let expect = if h < 8 { tri2 } else { base2 };
            assert_eq!(ids[h], expect.rem_euclid(sizes[h]) + offsets[h], "헤드 {h}");
        }
        let ids4 = ngram_ids(&hist, 4, &mult, &sizes, &offsets, eos, 8);
        let tri4 = 8i64.wrapping_mul(7) ^ 999i64.wrapping_mul(11);
        let base4 = tri4 ^ 999i64.wrapping_mul(13);
        for h in 0..16 {
            let expect = if h < 8 { tri4 } else { base4 };
            assert_eq!(ids4[h], expect.rem_euclid(sizes[h]) + offsets[h]);
        }
        let ids0 = ngram_ids(&hist, 0, &mult, &sizes, &offsets, eos, 8);
        let tri0 = 5i64.wrapping_mul(7) ^ 999i64.wrapping_mul(11);
        let base0 = tri0 ^ 999i64.wrapping_mul(13);
        for h in 0..16 {
            let expect = if h < 8 { tri0 } else { base0 };
            assert_eq!(ids0[h], expect.rem_euclid(sizes[h]) + offsets[h]);
        }
        let big = vec![248319u32, 1];
        let bm = vec![0x9E3779B97F4A7C15u64 as i64, 3, 5];
        let ids_b = ngram_ids(&big, 1, &bm, &sizes, &offsets, eos, 8);
        for (h, &v) in ids_b.iter().enumerate() {
            let rel = v - offsets[h];
            assert!((0..sizes[h]).contains(&rel), "헤드 {h} id {v}");
        }
    }
}
