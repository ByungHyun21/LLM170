//! GDN층 스테이지 — CPU 참조 경로. 수치 경로 불변.

use super::super::{ModelError, SeqState, span_block};
use super::Ctx;
use crate::ops::{l2_norm, sigmoid, silu, softplus};
use crate::qwen35::{mm_batch, mm_group};
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

    // qkv·z·beta·alpha는 전부 동일 입력 xs — 1그룹 배치.
    let mut group: [Vec<Vec<f32>>; 4] = [
        vec![vec![0.0f32; conv_ch]; n_tok],
        vec![vec![0.0f32; d_inner]; n_tok],
        vec![vec![0.0f32; dt_rank]; n_tok],
        vec![vec![0.0f32; dt_rank]; n_tok],
    ];
    {
        span_block!("cpu::gdn_qkvzba", {
            mm_group(xs, &[wqkv, wgate, wbeta, walpha], &mut group);
        });
    }
    let [qkv, z, b, a] = group;
    if il == 0 && llm170_diag::dump::opts().key("debug_layers") {
        eprintln!("  C0 xn[0..4]={:?}", &xs[0][..4]);
        eprintln!("  C0 qkv[0..4]={:?}", &qkv[0][..4]);
        eprintln!("  C0 z[0..4]={:?}", &z[0][..4]);
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
            if il == 0 && llm170_diag::dump::opts().key("debug_layers") {
                crate::qwen35::diag::g0_gdn(
                    r0, r1, &o_all, &q_all, &k_all, &v_all, &beta_all, &g_all, v_len, k_len,
                    dt_rank,
                );
            }
        } else {
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
        // [2.G 진단 2026-10-13] GDN 라이브 상태 크기 통계(LLM170_DUMP=gdst) —
        // 상태 양자화 후보 판정용: 원소 크기 분포(f16 서브노멀 질량)와 행(kdim)
        // 별 최대값(행-스케일 설계 여지), 전역 최대(오버플로 여지).
        // 실측(27B en 64토큰): |S|max≤35·서브노멀(<6e-5) 12~28% — raw f16은
        // 35B에서 밴드 초과(+1.55%)의 주 원인(채널 스케일 부재, bf16 대조로 확인).
        if llm170_diag::dump::opts().key("gdst") {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static GDST_N: AtomicUsize = AtomicUsize::new(0);
            let n = GDST_N.fetch_add(1, Ordering::Relaxed);
            let dd = d_state * d_state;
            let (mut gmax, mut nsub, mut nlow, mut ntot) = (0f32, 0usize, 0usize, 0usize);
            let (mut rmin, mut rmax) = (f32::MAX, 0f32);
            for hc in st.chunks(dd) {
                for i in 0..d_state {
                    let row = &hc[i * d_state..(i + 1) * d_state];
                    let rm = row.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    if rm > 0.0 {
                        rmin = rmin.min(rm);
                        rmax = rmax.max(rm);
                    }
                    for &x in row {
                        let a = x.abs();
                        if a > 0.0 {
                            if a < 6e-5 {
                                nsub += 1;
                            }
                            if a < 1e-4 {
                                nlow += 1;
                            }
                        }
                        ntot += 1;
                    }
                }
                gmax = hc.iter().fold(gmax, |a, &x| a.max(x.abs()));
            }
            eprintln!(
                "[gdst] n={n} il={il} max={gmax:.4e} row[{rmin:.3e},{rmax:.3e}] sub6e-5={nsub} low1e-4={nlow} tot={ntot}"
            );
        }
    }

    // norm_gated: rms_norm(core)·silu(z) per head → ssm_out
    let mut gated = vec![vec![0.0f32; d_inner]; n_tok];
    {
        profile_span!("cpu::gdn_normgated");
        crate::gdn_norm::gdn_norm_gated(
            crate::gdn_norm::GdnGate::Silu,
            &o_all,
            &z,
            &ssm_norm_w,
            hp.eps,
            n_tok,
            dt_rank,
            d_state,
            v_len,
            &mut gated,
        );
    }
    let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
    {
        span_block!("cpu::gdn_out", {
            mm_batch(&gated, &wout, &mut out);
        });
    }
    if il == 0 && llm170_diag::dump::opts().key("debug_layers") {
        eprintln!("  C0 gated[0..4]={:?}", &gated[0][..4]);
        eprintln!("  C0 out[0..4]={:?}", &out[0][..4]);
    }
    Ok(out)
}
