//! D1(plans/91 P5) — qwen35/qwen4exp GDN층 공유 norm_gated 코어.
//! z-게이트 활성화(silu↔sigmoid)만 파라미터화 — 양쪽 레이어의 나머지
//! 오케스트레이션(상태 유형·GPU 훅·시퀀스 레이아웃)은 구조적으로 달라
//! 그대로 둔다(원장 90 D8/D9 판정 준거 — 중복 없는 간접화 회피).

use crate::ops::{rms_norm, silu, sigmoid};

/// 게이트 활성화 종류 — qwen35=silu, qwen4exp(Flash-Next)=sigmoid.
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

/// norm_gated: 헤드별 rms_norm(o) · gate(z) → gated[t][h·d_state .. +d_state].
/// 행 레이아웃: o_all/z는 [n_tok][v_len] 플랫, gated는 [n_tok][d_inner].
/// 원문 루프와 동일 순서(헤드→원소, rms_norm 벡터 복제 포함) — 비트 불변.
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
