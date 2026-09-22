//! GDN층 스테이지 — layers.rs에서 이동(qwen4exp P1 패턴, plans/90 B4).
//! 수치 경로 불변 — 시그니처만 Ctx/seqs 분리.

use super::super::{ModelError, SeqState, span_block};
use super::Ctx;
use crate::matmul::{mm_batch, mm_group};
use crate::ops::{l2_norm, rms_norm, silu, softplus, sigmoid};
use llm170_diag::profile_span;

    /// GDN층: qkv/게이트/베타/알파/아웃 프로젝션 — 디스패치 경유.
pub(crate) fn gdn_layer(
    ctx: &Ctx,
    seqs: &mut [SeqState],
        il: usize,
        xs: &[Vec<f32>],
        seq_ids: &[usize],
        t_len: usize,
        recr_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_gdn");
        let acc = ctx.acc.clone();
        let hp = &ctx.model.hp;
        let n_seqs = seq_ids.len();
        let n_tok = n_seqs * t_len;
        let (d_state, dt_rank, n_group, d_inner) = (hp.d_state, hp.dt_rank, hp.n_group, hp.d_inner);
        let (conv_k, conv_ch) = (hp.conv_k, hp.conv_ch());
        let wqkv = ctx.model.wchk(&format!("blk.{il}.attn_qkv.weight"))?;
        let wgate = ctx.model.wchk(&format!("blk.{il}.attn_gate.weight"))?;
        let wbeta = ctx.model.wchk(&format!("blk.{il}.ssm_beta.weight"))?;
        let walpha = ctx.model.wchk(&format!("blk.{il}.ssm_alpha.weight"))?;
        let ssm_a = ctx.model.f32_vec(&format!("blk.{il}.ssm_a"))?;
        let dt_bias = ctx.model.f32_vec(&format!("blk.{il}.ssm_dt.bias"))?;
        let conv_w = ctx.model.f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))?; // [conv_k][conv_ch] 행 우선
        let ssm_norm_w = ctx.model.f32_vec(&format!("blk.{il}.ssm_norm.weight"))?;
        let wout = ctx.model.wchk(&format!("blk.{il}.ssm_out.weight"))?;

        let dbg0 = il
            == std::env::var("LLM170_DEBUG_LAYER")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
            && std::env::var_os("LLM170_DEBUG_LAYERS").is_some();
        // qkv·z·beta·alpha는 전부 동일 입력 xs — 1그룹 배치 (GPU: 업로드 1회+동기 1회)
        let mut group: [Vec<Vec<f32>>; 4] = [
            vec![vec![0.0f32; conv_ch]; n_tok],
            vec![vec![0.0f32; d_inner]; n_tok],
            vec![vec![0.0f32; dt_rank]; n_tok],
            vec![vec![0.0f32; dt_rank]; n_tok],
        ];
        {
            span_block!("cpu::gdn_qkvzba", {
                mm_group(&acc, xs, &[wqkv, wgate, wbeta, walpha], &mut group)?;
            });
        }
        let [qkv, z, b, a] = group;

        if dbg0 {
            let mz = z
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            let mq = qkv
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            let mc = xs
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!("  rs stage cur max={mc:.5} qkv max={mq:.5} z max={mz:.5}");
        }
        let mut beta_all = vec![0.0f32; n_tok * dt_rank];
        let mut g_all = vec![0.0f32; n_tok * dt_rank];
        {
            span_block!("cpu::gdn_bg", {
                for t in 0..n_tok {
                    for h in 0..dt_rank {
                        beta_all[t * dt_rank + h] = sigmoid(b[t][h]);
                        g_all[t * dt_rank + h] = softplus(a[t][h] + dt_bias[h]) * ssm_a[h];
                    }
                }
            });
        }

        let k_len = n_group * d_state;
        let v_len = dt_rank * d_state;
        let mut q_all = vec![0.0f32; n_tok * k_len];
        let mut k_all = vec![0.0f32; n_tok * k_len];
        let mut v_all = vec![0.0f32; n_tok * v_len];
        let mut o_all = vec![0.0f32; n_tok * v_len];
        // ── GPU 연결 (02-2): t=1 디코드를 conv·β/g·AR·norm_gated 커널로.
        // 값 스타일(업/다운로드). LLM170_GDN_CPU=1 또는 어느 단계 실패 시 CPU 전체 폴백.
        let mut gated = vec![vec![0.0f32; d_inner]; n_tok];
        let mut gpu_done = false;
        if t_len == 1 && n_seqs == 1 && std::env::var_os("LLM170_GDN_CPU").is_none()
            && let Some(acc_ref) = acc.as_deref() {
                let mut conv_out = vec![0.0f32; conv_ch];
                let st = &mut seqs[seq_ids[0]].conv[recr_idx];
                if acc_ref
                    .gdn_conv(&qkv[0], &conv_w, st, &mut conv_out, conv_ch, conv_k)
                    .is_ok()
                {
                    for c in 0..conv_ch {
                        if c < k_len {
                            q_all[c] = conv_out[c];
                        } else if c < 2 * k_len {
                            k_all[c - k_len] = conv_out[c];
                        } else {
                            v_all[c - 2 * k_len] = conv_out[c];
                        }
                    }
                    for h in 0..n_group {
                        let b0 = h * d_state;
                        let head: Vec<f32> = q_all[b0..b0 + d_state].to_vec();
                        q_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&head, hp.eps));
                        let headk: Vec<f32> = k_all[b0..b0 + d_state].to_vec();
                        k_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&headk, hp.eps));
                    }
                    let mut beta_ge = vec![0.0f32; dt_rank * 2];
                    if acc_ref
                        .gdn_beta_g(&b[0], &a[0], &dt_bias, &ssm_a, &mut beta_ge)
                        .is_ok()
                    {
                        let scale = 1.0f32 / (d_state as f32).sqrt();
                        let qs: Vec<f32> = q_all.iter().map(|x| x * scale).collect();
                        let st = &mut seqs[seq_ids[0]].gdn_s[recr_idx];
                        if acc_ref
                            .gdn_ar(&qs, &k_all, &v_all, &beta_ge, st, &mut o_all, 1, n_group, dt_rank, d_state)
                            .is_ok()
                        {
                            let w_tiled: Vec<f32> = ssm_norm_w
                                .iter()
                                .copied()
                                .cycle()
                                .take(ssm_norm_w.len() * dt_rank)
                                .collect();
                            let mut grow = vec![0.0f32; d_inner];
                            if acc_ref
                                .gdn_norm_gated_silu(&o_all, &z[0], &w_tiled, &mut grow, hp.eps, d_state)
                                .is_ok()
                            {
                                gated[0].copy_from_slice(&grow);
                                gpu_done = true;
                            }
                        }
                    }
                }
            }
        if !gpu_done {
            {
                profile_span!("cpu::gdn_conv");
                for s in 0..n_seqs {
                    let conv_state = &mut seqs[seq_ids[s]].conv[recr_idx];
                    for t in 0..t_len {
                        let row = s * t_len + t;
                        for c in 0..conv_ch {
                            // ggml ssm_conv: weight {d_conv, d_inner} 행 우선 → w[c*conv_k + j]
                            let mut sum = conv_w[c * conv_k + (conv_k - 1)] * qkv[row][c];
                            for j in 0..conv_k - 1 {
                                sum += conv_w[c * conv_k + j] * conv_state[j * conv_ch + c];
                            }
                            let out_c = silu(sum);
                            for j in 0..conv_k - 2 {
                                conv_state[j * conv_ch + c] = conv_state[(j + 1) * conv_ch + c];
                            }
                            conv_state[(conv_k - 2) * conv_ch + c] = qkv[row][c];
                            // 레이아웃: q [k heads] | k [k heads] | v [v heads]
                            if c < k_len {
                                q_all[row * k_len + c] = out_c;
                            } else if c < 2 * k_len {
                                k_all[row * k_len + c - k_len] = out_c;
                            } else {
                                v_all[row * v_len + c - 2 * k_len] = out_c;
                            }
                        }
                    }
                }
            }
            {
                profile_span!("cpu::gdn_l2norm");
                for row in 0..n_tok {
                    for h in 0..n_group {
                        let b0 = row * k_len + h * d_state;
                        let head: Vec<f32> = q_all[b0..b0 + d_state].to_vec();
                        q_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&head, hp.eps));
                        let headk: Vec<f32> = k_all[b0..b0 + d_state].to_vec();
                        k_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&headk, hp.eps));
                    }
                }
            }
            profile_span!("cpu::gdn_core");
            for s in 0..n_seqs {
                let r0 = s * t_len;
                let r1 = r0 + t_len;
                let st = &mut seqs[seq_ids[s]].gdn_s[recr_idx];
                if t_len == 1 {
                    crate::gdn::gdn_ar_batch(
                        &q_all[r0 * k_len..r1 * k_len],
                        &k_all[r0 * k_len..r1 * k_len],
                        &v_all[r0 * v_len..r1 * v_len],
                        &beta_all[r0 * dt_rank..r1 * dt_rank],
                        &g_all[r0 * dt_rank..r1 * dt_rank],
                        st,
                        &mut o_all[r0 * v_len..r1 * v_len],
                        1,
                        n_group,
                        dt_rank,
                    );
                    if il == 0 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                        crate::qwen35::diag::g0_gdn(r0, r1, &o_all, &q_all, &k_all, &v_all, &beta_all, &g_all, v_len, k_len, dt_rank);
                    }
                } else {
                    // GPU 청크 (03 §3.1) — 값 스타일, 실패 시 CPU 청크.
                    let mut done = false;
                    if std::env::var_os("LLM170_GDN_CPU").is_none()
                        && let Some(acc_ref) = acc.as_deref() {
                            let flat_st: &mut [f32] = st;
                            if acc_ref
                                .gdn_chunk(
                                    &q_all[r0 * k_len..r1 * k_len],
                                    &k_all[r0 * k_len..r1 * k_len],
                                    &v_all[r0 * v_len..r1 * v_len],
                                    &beta_all[r0 * dt_rank..r1 * dt_rank],
                                    &g_all[r0 * dt_rank..r1 * dt_rank],
                                    flat_st,
                                    &mut o_all[r0 * v_len..r1 * v_len],
                                    t_len,
                                    n_group,
                                    dt_rank,
                                    d_state,
                                )
                                .is_ok()
                            {
                                done = true;
                            }
                        }
                    if !done {
                        crate::gdn::gdn_chunk_seq(
                            &q_all[r0 * k_len..r1 * k_len],
                            &k_all[r0 * k_len..r1 * k_len],
                            &v_all[r0 * v_len..r1 * v_len],
                            &beta_all[r0 * dt_rank..r1 * dt_rank],
                            &g_all[r0 * dt_rank..r1 * dt_rank],
                            st,
                            &mut o_all[r0 * v_len..r1 * v_len],
                            t_len,
                            n_group,
                            dt_rank,
                        );
                    }
                }
            }
        }

        if dbg0 {
            let _mq = q_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let _mk = k_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let _mv = v_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let _mo = o_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let c4: Vec<String> = o_all[..4].iter().map(|v| format!("{v:.6}")).collect();
            let z4: Vec<String> = z[0][..4].iter().map(|v| format!("{v:.6}")).collect();
            eprintln!("  rs stage core[:4]={c4:?} z[:4]={z4:?}");
        }
        // norm_gated: rms_norm(core)·silu(z) per head → ssm_out (GPU 경로가 이미 채움)
        if !gpu_done {
            profile_span!("cpu::gdn_normgated");
            for t in 0..n_tok {
                for h in 0..dt_rank {
                    let b0 = t * v_len + h * d_state;
                    let head: Vec<f32> = o_all[b0..b0 + d_state].to_vec();
                    let n = rms_norm(&head, &ssm_norm_w, hp.eps);
                    let zb = h * d_state;
                    for i in 0..d_state {
                        gated[t][zb + i] = n[i] * silu(z[t][zb + i]);
                    }
                }
            }
        }
        if dbg0 {
            let mg = gated
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            let fmt = |o: usize| -> String {
                gated[0][o..o + 4]
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            eprintln!(
                "  rs gated h0={} h1={} h2={} h3={} (max={mg:.5})",
                fmt(0),
                fmt(16),
                fmt(32),
                fmt(48)
            );
        }
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        {
            span_block!("cpu::gdn_out", {
                mm_batch(&acc, &gated, &wout, &mut out)?;
            });
        }
        if std::env::var_os("LLM170_CPU_TRACE").is_some() {
            let last = out.len() - 1;
            let sum: f64 = out[last].iter().map(|&v| v as f64).sum();
            eprintln!("  CGDN il={il} out_sum={sum:.6} row={last}");
        }
        if dbg0 {
            let m = out
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            let (mut mi, mut mv) = (0usize, f32::NEG_INFINITY);
            for (r, row) in out.iter().enumerate() {
                for v in row.iter() {
                    if v.abs() > mv {
                        mv = v.abs();
                        mi = r;
                    }
                }
            }
            eprintln!(
                "  rs stage ssm_out max={m:.5} @row{mi} out[:4]={:?} out[{mi}][:3]={:?}",
                out[0][..4]
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>(),
                out[mi][..3]
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
            );
        }
        Ok(out)
    }
