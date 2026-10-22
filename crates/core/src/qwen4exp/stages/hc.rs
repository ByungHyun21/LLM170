
use super::super::engine::Model;
use crate::ops::{rms_norm, sigmoid, silu};

pub fn hc_norm_tok(r: &[f32], hc: usize, hidden: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut xn = Vec::with_capacity(r.len());
    for s in 0..hc {
        let b = s * hidden;
        xn.extend(rms_norm(&r[b..b + hidden], &w[b..b + hidden], eps));
    }
    xn
}

pub fn gate_mean_tok(xn: &[f32], gate: &[f32], hc: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    for s in 0..hc {
        let b = s * hidden;
        for j in 0..hidden {
            out[j] += sigmoid(gate[b + j]) * xn[b + j];
        }
    }
    for v in out.iter_mut() {
        *v /= hc as f32;
    }
    out
}

pub fn combine_tok(r: &mut [f32], y: &[f32], inj_logits: &[f32], hc: usize, hidden: usize) {
    for s in 0..hc {
        let g = 2.0 * sigmoid(inj_logits[s] / hc as f32);
        let b = s * hidden;
        for j in 0..hidden {
            r[b + j] += y[j] * g;
        }
    }
}

pub fn mix(m: &Model, p: &str, r: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
    let cfg = &m.cfg;
    let (hc, hidden) = (cfg.hc_count, cfg.hidden_size);
    let use_combine = m.ent(&format!("{p}.block_inject_weight.weight")).is_ok();
    let w = m.norm_vec(&format!("{p}.hc_norm.weight"))?;
    let mut block = Vec::with_capacity(r.len());
    let mut inj = Vec::with_capacity(if use_combine { r.len() * hc } else { 0 });
    for rt in r {
        let xn = hc_norm_tok(rt, hc, hidden, &w, m.eps);
        let down = m.matvec_bf16(&format!("{p}.input_mix_weight_down.weight"), &xn)?;
        let lora: Vec<f32> = down.iter().map(|&v| silu(v / hc as f32)).collect();
        let gate = m.matvec_bf16(&format!("{p}.input_mix_weight_up.weight"), &lora)?;
        block.push(gate_mean_tok(&xn, &gate, hc, hidden));
        if use_combine {
            let logits = m.matvec_bf16(&format!("{p}.block_inject_weight.weight"), &xn)?;
            inj.extend_from_slice(&logits);
        }
    }
    Ok((block, inj))
}

pub fn mix_final(m: &Model, r: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let (block, inj) = mix(m, "model.language_model.hyper_connection_mixer", r)?;
    debug_assert!(inj.is_empty(), "최종 mixer에 block_inject 부재 계약");
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_math() {
        let (hc, hidden) = (2usize, 4usize);
        let r = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let w = vec![1.0f32; 8];
        let xn = hc_norm_tok(&r, hc, hidden, &w, 1e-6);
        let s0 = 1.0f32 / (7.5f32 + 1e-6).sqrt();
        for j in 0..4 {
            assert!((xn[j] - r[j] * s0).abs() < 1e-6, "스트림0 정규화");
            assert!((xn[4 + j] - r[4 + j] * s0).abs() < 1e-6, "스트림1 정규화");
        }
        let gate = vec![0.0f32; 8];
        let m0 = gate_mean_tok(&xn, &gate, hc, hidden);
        for j in 0..4 {
            assert!(
                (m0[j] - (xn[j] + xn[4 + j]) / 4.0).abs() < 1e-7,
                "게이트 평균"
            );
        }
        let y = vec![1.0f32, -1.0, 2.0, 0.5];
        let mut rc = r.clone();
        combine_tok(&mut rc, &y, &[0.0f32; 2], hc, hidden);
        for j in 0..4 {
            assert_eq!(rc[j], r[j] + y[j]);
            assert_eq!(rc[4 + j], r[4 + j] + y[j]);
        }
        let mut rc2 = vec![0.0f32; 8];
        combine_tok(&mut rc2, &y, &[-30.0f32, 30.0], hc, hidden);
        assert!(rc2[..4].iter().all(|&v| v.abs() < 5e-6), "σ(−15)→0");
        for j in 0..4 {
            assert!((rc2[4 + j] - 2.0 * y[j]).abs() < 5e-6, "σ(15)→2");
        }
    }
}
