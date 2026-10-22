
use super::super::engine::{Model, SendPtr, par_ranges};
use crate::ops::{sigmoid, silu};
use crate::qwen35::stages::moe::select_topk;

const OUT_BAND: usize = 256;

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
    let (hidden, top_k) = (cfg.hidden_size, cfg.top_k);
    let n_tok = xs.len();
    let mut sels: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tok);
    for x in xs {
        let logits = m.matvec_bf16(&format!("{l}.gate.weight"), x)?;
        sels.push(select_topk(&logits, top_k));
    }
    for sel in &sels {
        for &(e, _) in sel {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                m.ent(&format!("{l}.experts.{e}.{p}.qweight"))?;
            }
        }
    }
    let mut hs: Vec<Vec<f32>> = vec![Vec::new(); n_tok * top_k];
    {
        let hs_ptr = SendPtr::new(hs.as_mut_ptr());
        let m_r: &Model = m;
        let l_r = l.as_str();
        par_ranges(n_tok * top_k, |b0, b1| {
            let hsp = unsafe { hs_ptr.as_ptr::<Vec<f32>>() };
            for u in b0..b1 {
                let (t, ei) = (u / top_k, u % top_k);
                let (e, _w) = sels[t][ei];
                let x = &xs[t];
                let gw = m_r
                    .expert_mat(&format!("{l_r}.experts.{e}.gate_proj"))
                    .expect("사전검증됨");
                let uw = m_r
                    .expert_mat(&format!("{l_r}.experts.{e}.up_proj"))
                    .expect("사전검증됨");
                let mut h = vec![0.0f32; gw.1];
                for (r, row) in gw.0.iter().enumerate() {
                    h[r] = silu(dot(row, x)) * dot(&uw.0[r], x);
                }
                unsafe { *hsp.add(u) = h };
            }
        });
    }
    let mut sgate = vec![0.0f32; n_tok];
    let mut s_out: Vec<Vec<f32>> = vec![Vec::new(); n_tok];
    {
        let (sg_ptr, so_ptr) = (
            SendPtr::new(sgate.as_mut_ptr()),
            SendPtr::new(s_out.as_mut_ptr()),
        );
        par_ranges(n_tok, |b0, b1| {
            let sgp = unsafe { sg_ptr.as_ptr::<f32>() };
            let sop = unsafe { so_ptr.as_ptr::<Vec<f32>>() };
            for t in b0..b1 {
                let x = &xs[t];
                let sg = sigmoid(
                    m.matvec_bf16(&format!("{l}.shared_expert_gate.weight"), x)
                        .unwrap()[0],
                );
                let s_up = m
                    .matvec_bf16(&format!("{l}.shared_expert.gate_proj.weight"), x)
                    .unwrap();
                let s_val = m
                    .matvec_bf16(&format!("{l}.shared_expert.up_proj.weight"), x)
                    .unwrap();
                let mut h = vec![0.0f32; s_up.len()];
                for (r, (&a, &b)) in s_up.iter().zip(s_val.iter()).enumerate() {
                    h[r] = silu(a) * b;
                }
                let so = m
                    .matvec_bf16(&format!("{l}.shared_expert.down_proj.weight"), &h)
                    .unwrap();
                unsafe {
                    *sgp.add(t) = sg;
                    *sop.add(t) = so;
                }
            }
        });
    }
    let mut flat = vec![0.0f32; n_tok * hidden];
    {
        let out_ptr = SendPtr::new(flat.as_mut_ptr());
        let m_r: &Model = m;
        let l_r = l.as_str();
        let bands = hidden.div_ceil(OUT_BAND);
        par_ranges(n_tok * bands, |b0, b1| {
            let op = unsafe { out_ptr.as_ptr::<f32>() };
            for u in b0..b1 {
                let (t, bi) = (u / bands, u % bands);
                let (r0, r1) = (bi * OUT_BAND, ((bi + 1) * OUT_BAND).min(hidden));
                for r in r0..r1 {
                    let mut acc = 0.0f32;
                    for (ei, &(e, w)) in sels[t].iter().enumerate() {
                        let drow = m_r
                            .expert_row(&format!("{l_r}.experts.{e}.down_proj"), r)
                            .expect("down_proj 행");
                        acc += w * dot(&drow, &hs[t * top_k + ei]);
                    }
                    acc += sgate[t] * s_out[t][r];
                    unsafe { *op.add(t * hidden + r) = acc };
                }
            }
        });
    }
    Ok(flat.chunks(hidden).map(<[f32]>::to_vec).collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x as f64 * y as f64;
    }
    acc as f32
}
