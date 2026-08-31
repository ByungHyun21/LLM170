//! gdn(delta 순환) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use super::super::layers::SeqState4;
use crate::ops::{l2_norm, rms_norm, sigmoid, silu, softplus};
use llm170_profiler::profile_span;

    /// GDN층 — qwen35와 동일 모듈, 차이: z-gate가 sigmoid.
    pub fn gdn_layer(
        ctx: &Ctx,
        seq: &mut SeqState4,
        il: usize,
        xs: &[Vec<f32>],
        t_len: usize,
        recr_idx: usize,
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::layer_gdn");
        let hp = ctx.model.hp.clone();
        let n_tok = t_len;
        let (d_state, dt_rank, n_group, d_inner) = (hp.d_state, hp.dt_rank, hp.n_group, hp.d_inner);
        let conv_ch = n_group * d_state * 2 + dt_rank * d_state;
        let wqkv = ctx.model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
        let wgate = ctx.model.w4(&format!("blk.{il}.attn_gate.weight"))?;
        let wbeta = ctx.model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
        let walpha = ctx.model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
        let ssm_a = ctx.model.f32_vec4(&format!("blk.{il}.ssm_a"))?;
        let dt_bias = ctx.model.f32_vec4(&format!("blk.{il}.ssm_dt.bias"))?;
        let conv_w = ctx.model.f32_vec4(&format!("blk.{il}.ssm_conv1d.weight"))?;
        let ssm_norm_w = ctx.model.f32_vec4(&format!("blk.{il}.ssm_norm.weight"))?;
        let wout = ctx.model.w4(&format!("blk.{il}.ssm_out.weight"))?;

        // qkv/gate/beta/alpha는 동일 입력 xs — 그룹 1호출 (왕복 4→1).
        let mut qkv = vec![vec![0.0f32; conv_ch]; n_tok];
        let mut z = vec![vec![0.0f32; d_inner]; n_tok];
        let mut b = vec![vec![0.0f32; dt_rank]; n_tok];
        let mut a = vec![vec![0.0f32; dt_rank]; n_tok];
        {
            let mut gi = vec![
                std::mem::take(&mut qkv),
                std::mem::take(&mut z),
                std::mem::take(&mut b),
                std::mem::take(&mut a),
            ];
            ctx.mm_group(xs, &[wqkv, wgate, wbeta, walpha], &mut gi)?;
            qkv = std::mem::take(&mut gi[0]);
            z = std::mem::take(&mut gi[1]);
            b = std::mem::take(&mut gi[2]);
            a = std::mem::take(&mut gi[3]);
        }

        let mut beta_all = vec![0.0f32; n_tok * dt_rank];
        let mut g_all = vec![0.0f32; n_tok * dt_rank];
        for t in 0..n_tok {
            for h in 0..dt_rank {
                beta_all[t * dt_rank + h] = sigmoid(b[t][h]);
                g_all[t * dt_rank + h] = softplus(a[t][h] + dt_bias[h]) * ssm_a[h];
            }
        }

        let k_len = n_group * d_state;
        let v_len = dt_rank * d_state;
        let mut q_all = vec![0.0f32; n_tok * k_len];
        let mut k_all = vec![0.0f32; n_tok * k_len];
        let mut v_all = vec![0.0f32; n_tok * v_len];
        let mut o_all = vec![0.0f32; n_tok * v_len];
        {
            let conv_state = &mut seq.conv[recr_idx];
            for t in 0..t_len {
                for c in 0..conv_ch {
                    let mut sum = conv_w[c * hp.conv_k + (hp.conv_k - 1)] * qkv[t][c];
                    for j in 0..hp.conv_k - 1 {
                        sum += conv_w[c * hp.conv_k + j] * conv_state[j * conv_ch + c];
                    }
                    let out_c = silu(sum);
                    for j in 0..hp.conv_k - 2 {
                        conv_state[j * conv_ch + c] = conv_state[(j + 1) * conv_ch + c];
                    }
                    conv_state[(hp.conv_k - 2) * conv_ch + c] = qkv[t][c];
                    if c < k_len {
                        q_all[t * k_len + c] = out_c;
                    } else if c < 2 * k_len {
                        k_all[t * k_len + c - k_len] = out_c;
                    } else {
                        v_all[t * v_len + c - 2 * k_len] = out_c;
                    }
                }
            }
        }
        for row in 0..n_tok {
            for h in 0..n_group {
                let b0 = row * k_len + h * d_state;
                let head: Vec<f32> = q_all[b0..b0 + d_state].to_vec();
                q_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&head, hp.eps));
                let headk: Vec<f32> = k_all[b0..b0 + d_state].to_vec();
                k_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&headk, hp.eps));
            }
        }
        {
            let st = &mut seq.gdn_s[recr_idx];
            if t_len == 1 {
                crate::gdn::gdn_ar_batch(
                    &q_all, &k_all, &v_all, &beta_all, &g_all, st, &mut o_all, 1, n_group, dt_rank,
                );
            } else {
                crate::gdn::gdn_chunk_seq(
                    &q_all, &k_all, &v_all, &beta_all, &g_all, st, &mut o_all, t_len, n_group, dt_rank,
                );
            }
        }
        // norm_gated: rms·sigmoid(z) — qwen35(silu)와의 유일 차이
        let mut gated = vec![vec![0.0f32; d_inner]; n_tok];
        for t in 0..n_tok {
            for h in 0..dt_rank {
                let b0 = t * v_len + h * d_state;
                let head: Vec<f32> = o_all[b0..b0 + d_state].to_vec();
                let n = rms_norm(&head, &ssm_norm_w, hp.eps);
                for i in 0..d_state {
                    gated[t][h * d_state + i] = n[i] * sigmoid(z[t][h * d_state + i]);
                }
            }
        }
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        ctx.mm_batch(&gated, &wout, &mut out)?;
        Ok(out)
    }

