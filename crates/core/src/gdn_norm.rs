
use crate::ops::{rms_norm, sigmoid, silu};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GdnGate {
    Silu,
    Sigmoid,
}

#[inline]
fn gate_act(g: GdnGate, x: f32) -> f32 {
    match g {
        GdnGate::Silu => silu(x),
        GdnGate::Sigmoid => sigmoid(x),
    }
}

pub fn gdn_norm_gated(
    gate: GdnGate,
    o_all: &[f32],
    z: &[Vec<f32>],
    ssm_norm_w: &[f32],
    eps: f32,
    n_tok: usize,
    dt_rank: usize,
    d_state: usize,
    v_len: usize,
    gated: &mut [Vec<f32>],
) {
    for t in 0..n_tok {
        for h in 0..dt_rank {
            let b0 = t * v_len + h * d_state;
            let head: Vec<f32> = o_all[b0..b0 + d_state].to_vec();
            let n = rms_norm(&head, ssm_norm_w, eps);
            let zb = h * d_state;
            for i in 0..d_state {
                gated[t][zb + i] = n[i] * gate_act(gate, z[t][zb + i]);
            }
        }
    }
}
