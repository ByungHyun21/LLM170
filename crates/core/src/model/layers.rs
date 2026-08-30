//! qwen35 층 구현 — GDN층(gdn_layer)과 full-attention층(attn_layer).
//! 그래프 배선: ~/local_llm/llama.cpp/src/models/qwen35.cpp (2026-08-30 판).

use super::{Engine, ModelError};
use crate::matmul::{matmul, matmul_batch};
use crate::ops::{l2_norm, rms_norm, rope_head, silu, sigmoid, softplus};
use llm170_profiler::profile_span;
use super::span_block;

impl Engine {
    pub(super) fn gdn_layer(
        &mut self,
        il: usize,
        xs: &[Vec<f32>],
        n_seqs: usize,
        t_len: usize,
        recr_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_gdn");
        let hp = &self.model.hp;
        let n_tok = n_seqs * t_len;
        let (d_state, dt_rank, n_group, d_inner) = (hp.d_state, hp.dt_rank, hp.n_group, hp.d_inner);
        let (conv_k, conv_ch) = (hp.conv_k, hp.conv_ch());
        let wqkv = self.model.wchk(&format!("blk.{il}.attn_qkv.weight"))?;
        let wgate = self.model.wchk(&format!("blk.{il}.attn_gate.weight"))?;
        let wbeta = self.model.wchk(&format!("blk.{il}.ssm_beta.weight"))?;
        let walpha = self.model.wchk(&format!("blk.{il}.ssm_alpha.weight"))?;
        let ssm_a = self.model.f32_vec(&format!("blk.{il}.ssm_a"))?;
        let dt_bias = self.model.f32_vec(&format!("blk.{il}.ssm_dt.bias"))?;
        let conv_w = self.model.f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))?; // [conv_k][conv_ch] 행 우선
        let ssm_norm_w = self.model.f32_vec(&format!("blk.{il}.ssm_norm.weight"))?;
        let wout = self.model.wchk(&format!("blk.{il}.ssm_out.weight"))?;

        let dbg0 = il == std::env::var("LLM170_DEBUG_LAYER").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) && std::env::var_os("LLM170_DEBUG_LAYERS").is_some();
        let mut qkv = vec![vec![0.0f32; conv_ch]; n_tok];
        {
            span_block!("cpu::gdn_qkv", {
                matmul_batch(xs, &wqkv, &mut qkv);
            });
        }
        let mut z = vec![vec![0.0f32; d_inner]; n_tok];
        {
            span_block!("cpu::gdn_z", {
                matmul_batch(xs, &wgate, &mut z);
            });
        }

        if dbg0 {
            let mz = z.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let mq = qkv.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let mc = xs.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!("  rs stage cur max={mc:.5} qkv max={mq:.5} z max={mz:.5}");
        }
        let mut beta_all = vec![0.0f32; n_tok * dt_rank];
        let mut g_all = vec![0.0f32; n_tok * dt_rank];
        {
            span_block!("cpu::gdn_bg", {
                let mut b = vec![vec![0.0f32; dt_rank]; n_tok];
                let mut a = vec![vec![0.0f32; dt_rank]; n_tok];
                matmul_batch(xs, &wbeta, &mut b);
                matmul_batch(xs, &walpha, &mut a);
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
                let conv_state = &mut self.seqs[s].conv[recr_idx];
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

        {
            profile_span!("cpu::gdn_core");
            for s in 0..n_seqs {
                let r0 = s * t_len;
                let r1 = r0 + t_len;
                let st = &mut self.seqs[s].gdn_s[recr_idx];
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
            }
        }

        if dbg0 {
            let mq = q_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mk = k_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mv = v_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mo = o_all.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let c4: Vec<String> = o_all[..4].iter().map(|v| format!("{v:.6}")).collect();
            let z4: Vec<String> = z[0][..4].iter().map(|v| format!("{v:.6}")).collect();
            eprintln!("  rs stage core[:4]={c4:?} z[:4]={z4:?}");
        }
        // norm_gated: rms_norm(core)·silu(z) per head → ssm_out
        let mut gated = vec![vec![0.0f32; d_inner]; n_tok];
        {
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
            let mg = gated.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let fmt = |o: usize| -> String { gated[0][o..o + 4].iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(",") };
            eprintln!("  rs gated h0={} h1={} h2={} h3={} (max={mg:.5})", fmt(0), fmt(16), fmt(32), fmt(48));
        }
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        {
            span_block!("cpu::gdn_out", {
                matmul_batch(&gated, &wout, &mut out);
            });
        }
        if dbg0 {
            let m = out.iter().flat_map(|r| r.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            let (mut mi, mut mv) = (0usize, f32::NEG_INFINITY);
            for (r, row) in out.iter().enumerate() {
                for (c, v) in row.iter().enumerate() {
                    if v.abs() > mv { mv = v.abs(); mi = r; }
                }
            }
            eprintln!("  rs stage ssm_out max={m:.5} @row{mi} out[:4]={:?} out[{mi}][:3]={:?}", 
                out[0][..4].iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>(),
                out[mi][..3].iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>());
        }
        Ok(out)
    }

    /// Full-attention층 (gated, IMROPE 텍스트 퇴화형).
    pub(super) fn attn_layer(
        &mut self,
        il: usize,
        xs: &[Vec<f32>],
        n_seqs: usize,
        t_len: usize,
        full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_attn");
        let hp = &self.model.hp;
        let n_tok = n_seqs * t_len;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = self.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;

        let mut qg = vec![vec![0.0f32; wq.n_out as usize]; n_tok];
        let mut kk = vec![vec![0.0f32; wk.n_out as usize]; n_tok];
        let mut vv = vec![vec![0.0f32; wv.n_out as usize]; n_tok];
        {
            span_block!("cpu::attn_qkv", {
                matmul_batch(xs, &wq, &mut qg);
                matmul_batch(xs, &wk, &mut kk);
                matmul_batch(xs, &wv, &mut vv);
            });
        }

        let kq_scale = hp.kq_scale();
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        for s in 0..n_seqs {
            let pos0 = self.seqs[s].pos;
            let seq = &mut self.seqs[s];
            let (cache_k, cache_v) =
                (seq.kv_k[full_idx].as_mut_slice(), seq.kv_v[full_idx].as_mut_slice());

            for t in 0..t_len {
                let row = s * t_len + t;
                let pos = pos0 + t as u32;
                for h in 0..n_kv {
                    let src = kk[row][h * hd..h * hd + hd].to_vec();
                    let mut head = rms_norm(&src, &k_norm_w, hp.eps);
                    rope_head(&mut head, pos, n_rot, hp.rope_base);
                    let b = (pos as usize) * n_kv * hd + h * hd;
                    cache_k[b..b + hd].copy_from_slice(&head);
                    cache_v[b..b + hd].copy_from_slice(&vv[row][h * hd..h * hd + hd]);
                }
                let mut attn_out = vec![0.0f32; n_head * hd];
                for h in 0..n_head {
                    let src = qg[row][h * 2 * hd..h * 2 * hd + hd].to_vec();
                    let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                    rope_head(&mut qh, pos, n_rot, hp.rope_base);
                    let kvh = h / (n_head / n_kv);
                    let n_past = pos as usize + 1;
                    let mut scores = vec![0.0f32; n_past];
                    let mut maxv = f32::NEG_INFINITY;
                    for (p, sc) in scores.iter_mut().enumerate() {
                        let b = p * n_kv * hd + kvh * hd;
                        let mut d = 0.0f32;
                        for i in 0..hd {
                            d += qh[i] * cache_k[b + i];
                        }
                        *sc = d * kq_scale;
                        maxv = maxv.max(*sc);
                    }
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - maxv).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    let ob = h * hd;
                    for p in 0..n_past {
                        let w = scores[p];
                        if w == 0.0 {
                            continue;
                        }
                        let b = p * n_kv * hd + kvh * hd;
                        for i in 0..hd {
                            attn_out[ob + i] += w * cache_v[b + i];
                        }
                    }
                    // q‖gate 인접 인터리브: gate at h*2*hd + hd
                    let gb = h * 2 * hd + hd;
                    for i in 0..hd {
                        attn_out[ob + i] *= sigmoid(qg[row][gb + i]);
                    }
                }
                matmul(&attn_out, &wo, &mut out[row]);
            }
        }
        Ok(out)
    }
}
