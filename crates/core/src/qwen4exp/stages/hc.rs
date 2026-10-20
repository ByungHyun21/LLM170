//! hyper-connection(Gated Residual) 스테이지 — [D2 2026-10-20].
//!
//! vLLM qwen4_exp common/hyperconnection.py `GatedResidual`
//! (hc_per_branch_norm=True) 전수:
//! - 읽기 `mix`: R̂ = 스트림별 GemmaRMSNorm(1+w)(hc_norm[10240],
//!   그룹 2560) → down W_d[320,10240]/n_r → SiLU → up W_u[10240,320] →
//!   σ → G[4,2560], block_in = mean_s(G_s ⊙ R̂_s). use_combine면
//!   인젝션 로짓 W_w[4,10240]·R̂ (raw — combine에서 2σ(·/n_r)).
//! - 쓰기 `combine`: R′_s = R_s + y ⊙ 2σ(w_s/n_r).
//! - 최종 mixer(`hyper_connection_mixer`, use_combine=False) = mix만
//!   (block_inject 부재 — lm_head 입력).
//!
//! 체크포인트 실측 레이아웃: down [320,10240]·up [10240,320]·block_inject
//! [4,10240] 전부 자연 [out,in] — `matvec_bf16` 직접.
//! (AMD 지연 결합은 위 수학의 스케줄 차이일 뿐 — 오라클은 즉시 결합.)

use super::super::engine::Model;
use crate::ops::{rms_norm, sigmoid, silu};

/// hc_norm 그룹 정규화 — 스트림별(H 그룹) GemmaRMSNorm(1+w), 토큰 1개분.
pub fn hc_norm_tok(r: &[f32], hc: usize, hidden: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut xn = Vec::with_capacity(r.len());
    for s in 0..hc {
        let b = s * hidden;
        xn.extend(rms_norm(&r[b..b + hidden], &w[b..b + hidden], eps));
    }
    xn
}

/// 게이트 믹스 코어 — σ(gate)⊙xn 스트림 평균(토큰 1개분,
/// hc_gate_mix 커널: acc = Σ_s σ(gate_s)·xn_s / hc).
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

/// combine — R_s += y ⊙ 2σ(inj_logit_s / n_r)(hc_combine 커널 전수, 토큰 1개분).
pub fn combine_tok(r: &mut [f32], y: &[f32], inj_logits: &[f32], hc: usize, hidden: usize) {
    for s in 0..hc {
        let g = 2.0 * sigmoid(inj_logits[s] / hc as f32);
        let b = s * hidden;
        for j in 0..hidden {
            r[b + j] += y[j] * g;
        }
    }
}

/// mix — 정규화·로우랭크 게이트·인젝션 로짓. 반환: (block_in [T,H],
/// inj_logits [T·hc] — block_inject 부재(최종 mixer)면 빈 Vec).
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

/// 최종 mixer — use_combine=False(블록 인젝션 없음): mix만 → lm_head 입력.
pub fn mix_final(m: &Model, r: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let (block, inj) = mix(m, "model.language_model.hyper_connection_mixer", r)?;
    debug_assert!(inj.is_empty(), "최종 mixer에 block_inject 부재 계약");
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 소형 fixture — hc=2·H=4: 정규화·게이트·combine 수식 원소 검증.
    #[test]
    fn hc_math() {
        let (hc, hidden) = (2usize, 4usize);
        // R = [1,2,3,4 | 4,3,2,1], w = 1(norm_vec(0)=1 합성 후 값)
        let r = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let w = vec![1.0f32; 8];
        let xn = hc_norm_tok(&r, hc, hidden, &w, 1e-6);
        // 스트림 0: rms = sqrt(30/4) → x/√7.5
        let s0 = 1.0f32 / (7.5f32 + 1e-6).sqrt();
        for j in 0..4 {
            assert!((xn[j] - r[j] * s0).abs() < 1e-6, "스트림0 정규화");
            assert!((xn[4 + j] - r[4 + j] * s0).abs() < 1e-6, "스트림1 정규화");
        }
        // gate 0 전구간: σ(0)=0.5 → (0.5·xn0 + 0.5·xn1)/2 = (xn0+xn1)/4.
        let gate = vec![0.0f32; 8];
        let m0 = gate_mean_tok(&xn, &gate, hc, hidden);
        for j in 0..4 {
            assert!(
                (m0[j] - (xn[j] + xn[4 + j]) / 4.0).abs() < 1e-7,
                "게이트 평균"
            );
        }
        // combine: inj 로짓 0 → g = 2σ(0) = 1 → R += y 그대로.
        let y = vec![1.0f32, -1.0, 2.0, 0.5];
        let mut rc = r.clone();
        combine_tok(&mut rc, &y, &[0.0f32; 2], hc, hidden);
        for j in 0..4 {
            assert_eq!(rc[j], r[j] + y[j]);
            assert_eq!(rc[4 + j], r[4 + j] + y[j]);
        }
        // inj 로짓 ±30: σ(±15) → 0 / 2배.
        let mut rc2 = vec![0.0f32; 8];
        combine_tok(&mut rc2, &y, &[-30.0f32, 30.0], hc, hidden);
        assert!(rc2[..4].iter().all(|&v| v.abs() < 5e-6), "σ(−15)→0");
        for j in 0..4 {
            assert!((rc2[4 + j] - 2.0 * y[j]).abs() < 5e-6, "σ(15)→2");
        }
    }
}
