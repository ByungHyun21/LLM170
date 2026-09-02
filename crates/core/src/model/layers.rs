//! qwen35 층 구현 — GDN층(gdn_layer)과 full-attention층(attn_layer).
//! 그래프 배선: ~/local_llm/llama.cpp/src/models/qwen35.cpp (2026-08-30 판).

use super::{Engine, ModelError, span_block};
use crate::matmul::{mm_batch, mm_group};
use crate::ops::{l2_norm, rms_norm, rope_head, sigmoid, silu, softplus};
use llm170_profiler::profile_span;
impl Engine {
    /// GDN층: qkv/게이트/베타/알파/아웃 프로젝션 — 디스패치 경유.
    pub(super) fn gdn_layer(
        &mut self,
        il: usize,
        xs: &[Vec<f32>],
        seq_ids: &[usize],
        t_len: usize,
        recr_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_gdn");
        let acc = self.acc.clone();
        let hp = &self.model.hp;
        let n_seqs = seq_ids.len();
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
        if t_len == 1 && n_seqs == 1 && std::env::var_os("LLM170_GDN_CPU").is_none() {
            if let Some(acc_ref) = acc.as_deref() {
                let mut conv_out = vec![0.0f32; conv_ch];
                let st = &mut self.seqs[seq_ids[0]].conv[recr_idx];
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
                        let st = &mut self.seqs[seq_ids[0]].gdn_s[recr_idx];
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
        }
        if !gpu_done {
            {
                profile_span!("cpu::gdn_conv");
                for s in 0..n_seqs {
                    let conv_state = &mut self.seqs[seq_ids[s]].conv[recr_idx];
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
                let st = &mut self.seqs[seq_ids[s]].gdn_s[recr_idx];
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
                        let sumo: f64 = o_all[r0 * v_len..r1 * v_len].iter().map(|&v| v as f64).sum();
                        let sumq: f64 = q_all[r0 * k_len..r1 * k_len].iter().map(|&v| v as f64).sum();
                        let mut xco: u64 = 0; let mut xcq: u64 = 0;
                        for &v in &o_all[r0 * v_len..r1 * v_len] { xco ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        for &v in &q_all[r0 * k_len..r1 * k_len] { xcq ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        eprintln!("  G0dbg o_all sum={sumo:.6} xor={xco:016x} q_all xor={xcq:016x}");
                        let mut xck: u64 = 0; let mut xcv: u64 = 0; let mut xcb: u64 = 0; let mut xcg: u64 = 0;
                        for &v in &k_all[r0 * k_len..r1 * k_len] { xck ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        for &v in &v_all[r0 * v_len..r1 * v_len] { xcv ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        for &v in &beta_all[r0 * dt_rank..r1 * dt_rank] { xcb ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        for &v in &g_all[r0 * dt_rank..r1 * dt_rank] { xcg ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        eprintln!("  G0dbg k_all xor={xck:016x} v_all xor={xcv:016x} beta xor={xcb:016x} g_all xor={xcg:016x}");
                        let mut xce: u64 = 0;
                        for &v in &g_all[r0 * dt_rank..r1 * dt_rank] { xce ^= (crate::ops::exp_cr(v).to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        eprintln!("  G0dbg exp_cr(g) xor={xce:016x}");
                    }
                } else {
                    // GPU 청크 (03 §3.1) — 값 스타일, 실패 시 CPU 청크.
                    let mut done = false;
                    if std::env::var_os("LLM170_GDN_CPU").is_none() {
                        if let Some(acc_ref) = acc.as_deref() {
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
        if dbg0 {
            let m = out
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            let (mut mi, mut mv) = (0usize, f32::NEG_INFINITY);
            for (r, row) in out.iter().enumerate() {
                for (_c, v) in row.iter().enumerate() {
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

    /// Full-attention층 (gated, IMROPE 텍스트 퇴화형).
    pub(super) fn attn_layer(
        &mut self,
        il: usize,
        xs: &[Vec<f32>],
        seq_ids: &[usize],
        t_len: usize,
        full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_attn");
        let acc = self.acc.clone();
        let hp = &self.model.hp;
        let n_seqs = seq_ids.len();
        let n_tok = n_seqs * t_len;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = self.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = self
            .model
            .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = self
            .model
            .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;

        if il == 3 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
            eprintln!("  A3dbg normed[0..6]={:?}", &xs[0][0..6]);
            if let Some(qb) = crate::quant::quantize_row_q8_ref(&xs[0]).first() {
                let w0 = qb.qs.iter().take(4).fold(0u32, |a, &v| a | ((v as u8 as u32) << (8 * (a.count_ones() as usize % 4))));
                let mut word = 0u32;
                for (i, b) in qb.qs.iter().take(4).enumerate() {
                    word |= (*b as u8 as u32) << (8 * i);
                }
                eprintln!("  A3dbg cpu q word0={word:#010x} d={:e} q[0..6]={:?}", qb.d, qb.qs.iter().take(6).collect::<Vec<_>>());
                let _ = w0;
            }
        }
        // q·k·v 동일 입력 xs — 1그룹 배치
        let mut group: [Vec<Vec<f32>>; 3] = [
            vec![vec![0.0f32; wq.n_out as usize]; n_tok],
            vec![vec![0.0f32; wk.n_out as usize]; n_tok],
            vec![vec![0.0f32; wv.n_out as usize]; n_tok],
        ];
        {
            span_block!("cpu::attn_qkv", {
                mm_group(&acc, xs, &[wq, wk, wv], &mut group)?;
            });
        }
        let [qg, kk, vv] = group;

        let kq_scale = hp.kq_scale();
        // GPU 연결 (02-3): t=1 디코드 score/softmax/V-mix를 qsa_attention 커널
        // 재사용(마스크=prefix 전체). norm·rope·캐시 기록은 CPU 유지(저렴).
        // LLM170_ATTN_CPU=1 또는 실패 시 CPU 루프.
        let attn_gpu = acc.is_some() && t_len == 1 && n_seqs == 1
            && std::env::var_os("LLM170_ATTN_CPU").is_none();
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        let mut attn_all = vec![vec![0.0f32; n_head * hd]; n_tok];
        let mut gpu_qrow: Option<Vec<f32>> = None;
        let mut gpu_done = false;
        for s in 0..n_seqs {
            let pos0 = self.seqs[seq_ids[s]].pos;
            let seq = &mut self.seqs[seq_ids[s]];
            let (cache_k, cache_v) = (
                seq.kv_k[full_idx].as_mut_slice(),
                seq.kv_v[full_idx].as_mut_slice(),
            );

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
                if attn_gpu {
                    // q norm+rope → q‖gate 인터리브 플랫 [n_head·2·hd]
                    // (qsa_score/mix 커널 계약 — 게이트는 커널이 sigmoid 적용)
                    let mut qrow = vec![0.0f32; n_head * 2 * hd];
                    for h in 0..n_head {
                        let src = qg[row][h * 2 * hd..h * 2 * hd + hd].to_vec();
                        let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                        rope_head(&mut qh, pos, n_rot, hp.rope_base);
                        qrow[h * 2 * hd..h * 2 * hd + hd].copy_from_slice(&qh);
                        let gb = h * 2 * hd + hd;
                        for i in 0..hd {
                            qrow[gb + i] = qg[row][gb + i];
                        }
                    }
                    gpu_qrow = Some(qrow);
                    continue;
                }
                let mut attn_out = std::mem::take(&mut attn_all[row]);
                let dbg3 = il == 3 && t == 0 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some();
                if dbg3 {
                    let b0 = (pos as usize) * n_kv * hd;
                    eprintln!("  A3dbg pos{pos} cache_k[b0..4]={:?} cache_k[0..4]={:?}", &cache_k[b0..b0 + 4], &cache_k[0..4]);
                    eprintln!("  A3dbg cache_v[0..4]={:?}", &cache_v[b0..b0 + 4]);
                    eprintln!("  A3dbg gate h0 [0..4]={:?}", &qg[row][hd..hd + 4]);
                    eprintln!("  A3dbg sigmoid(g)={:?}", (0..4).map(|i| sigmoid(qg[row][hd + i])).collect::<Vec<_>>());
                }
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
                        *sc = crate::ops::exp_cr(*sc - maxv);
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
                    if dbg3 && h == 0 {
                        let sumsc: f64 = scores.iter().map(|&v| v as f64).sum();
                        eprintln!("  A3dbg h0 scores_sum={sumsc:.6} n_past={} attn_out[0..4]={:?}", n_past, &attn_out[0..4]);
                    }
                }
                attn_all[row] = attn_out;
            }
        }
        if let Some(qrow) = gpu_qrow.as_ref() {
            if let Some(acc_ref) = acc.as_deref() {
                let seq = &self.seqs[seq_ids[0]];
                let n_past = seq.pos as usize + 1;
                let ck = &seq.kv_k[full_idx][..n_past * n_kv * hd];
                let cv = &seq.kv_v[full_idx][..n_past * n_kv * hd];
                let mask: Vec<u32> = (0..n_past).map(|p| (p < n_past) as u32).collect();
                if let Ok(res) = acc_ref.qsa_attention(
                    qrow, ck, cv, &mask, kq_scale, n_past, n_head, n_kv, hd, 1,
                ) {
                    // 커널 출력에 게이트 이미 적용 (qsa_mix)
                    attn_all[0].copy_from_slice(&res[..n_head * hd]);
                    gpu_done = true;
                }
            }
        }
        if gpu_qrow.is_some() && !gpu_done {
            // GPU 시도 실패 → CPU 재계산 (row 0, t=1)
            let pos = self.seqs[seq_ids[0]].pos;
            let seq = &self.seqs[seq_ids[0]];
            let (cache_k, cache_v) = (
                seq.kv_k[full_idx].as_slice(),
                seq.kv_v[full_idx].as_slice(),
            );
            let mut attn_out = std::mem::take(&mut attn_all[0]);
            for h in 0..n_head {
                let src = qg[0][h * 2 * hd..h * 2 * hd + hd].to_vec();
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
                    *sc = crate::ops::exp_cr(*sc - maxv);
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
                let gb = h * 2 * hd + hd;
                for i in 0..hd {
                    attn_out[ob + i] *= sigmoid(qg[0][gb + i]);
                }
            }
            attn_all[0] = attn_out;
        }
        // wo 프로젝션 — 전 토큰 배치 1회
        {
            span_block!("cpu::attn_wo", {
                mm_batch(&acc, &attn_all, &wo, &mut out)?;
            });
        }
        Ok(out)
    }
}
