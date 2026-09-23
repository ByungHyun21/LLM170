//! vk decoder 스텝 본체·배치 프리필 (plans/79 B).

use super::*;

impl DecoderState {

    /// t=1 단일 스텝 본체 — 배치 모드로 전 층 단일 제출·다운로드 1회.
    /// 로짓은 b_lg에만 남는다 (전사는 step() 래퍼).
    pub(super) fn step_core(&mut self, seq: usize, pos: usize, emb: &[f32]) -> Result<(), String> {
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
        let n = self.n_embd;
        debug_assert_eq!(emb.len(), n);
        let (dt_rank, d_state, d_inner) = (self.dt_rank, self.d_state, self.d_inner);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        unsafe { std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, n) };
        let vk_t0 = std::time::Instant::now();
        self.ctx.begin_batch()?;
        let _tw_rec = std::time::Instant::now();
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let layer_cut = std::env::var("LLM170_VK_LAYERS").ok().and_then(|v| v.parse::<usize>().ok());
        let npck = std::env::var_os("LLM170_VK_NPCK").is_some();
        for il in 0..self.n_layer {
            if layer_cut.is_some_and(|c| il >= c) {
                break;
            }
            // ── attn_norm — 0층만 (이후 층 입력 규격화는 하단 fdown addrms에 융합)
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, 1)?;
            }
            if self.is_recr[il] {
                // GDN 4 GEMV — 독립 그룹 (gemv_stage: 내부 배리어 생략·dead quant 스킵)
                self.gemv_stage(n, 1, &[
                    (format!("blk.{il}.attn_qkv.weight"), self.b_xq_n.buf, self.b_gqkv.buf),
                    (format!("blk.{il}.attn_gate.weight"), self.b_xq_n.buf, self.b_gz.buf),
                    (format!("blk.{il}.ssm_beta.weight"), self.b_xq_n.buf, self.b_gb.buf),
                    (format!("blk.{il}.ssm_alpha.weight"), self.b_xq_n.buf, self.b_ga.buf),
                ])?;
                let gskip = std::env::var("LLM170_VK_GDN_SKIP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
                // conv (t=1 — ring)
                {
                    let cw = self.consts.get(&format!("blk.{il}.conv_w")).cloned().ok_or("conv_w")?;
                    // PC: {int ch; int k; int t} + local 64 — gdn-check 준거
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, 1u32]);
                    self.run_pipe("gdn_conv", GDN_CONV_SPV, 4, 12,
                        &[self.b_gqkv.buf, cw.buf, self.st_conv[recr_idx][seq].buf, self.b_gconv.buf],
                        &push, conv_ch.div_ceil(64) as u32, 1, 1)?;
                }
                if npck && il < 3 {
                    let b = self.b_gconv.clone();
                    self.npck_mark("conv", il, &b, 0, 64);
                }
                if npck && il == 0 {
                    let s0b = self.st_gdn[0][seq].clone();
                    self.npck_mark("stin", il, &s0b, 0, 64);
                    let gb = self.b_gb.clone();
                    self.npck_mark("gb", il, &gb, 0, 16);
                    let ga = self.b_ga.clone();
                    self.npck_mark("ga", il, &ga, 0, 16);
                }
                let arf_on = std::env::var("LLM170_VK_ARF").map(|v| v != "0").unwrap_or(true);
                if arf_on && gskip & 1 == 0 {
                    let dtb2 = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa2 = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let scale2 = 1.0f32 / (d_state as f32).sqrt();
                    let mut push2 = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32,
                        dt_rank as u32, self.n_group as u32]);
                    push2.extend_from_slice(&scale2.to_le_bytes());
                    push2.extend_from_slice(&1u32.to_le_bytes());
                    push2.extend_from_slice(&self.eps.to_le_bytes());
                    self.run_pipe("gdn_arf", GDN_ARF_SPV, 7, 32,
                        &[self.st_gdn[recr_idx][seq].buf, self.b_gconv.buf,
                          self.b_gb.buf, self.b_ga.buf, dtb2.buf, ssa2.buf, self.b_go.buf],
                        &push2, dt_rank as u32, d_state as u32, 1)?;
                    if npck && il == 0 {
                        let s0b = self.st_gdn[0][seq].clone();
                        self.npck_mark("stout", il, &s0b, 0, 64);
                    }
                if npck && il < 3 {
                    let b = self.b_go.clone();
                    self.npck_mark("ar", il, &b, 0, 64);
                }
                } else {
                {
                    let total = 2 * k_len + v_len;
                    let push = Self::push_u32s(&[k_len as u32, k_len as u32, v_len as u32]);
                    self.run_pipe("split3", SPLIT3_SPV, 4, 12,
                        &[self.b_gconv.buf, self.b_gq.buf, self.b_gk.buf, self.b_gv.buf],
                        &push, total.div_ceil(64) as u32, 1, 1)?;
                }
                // l2 — PC: {float eps; int d; int ng} (스케일은 AR이 적용 — rawhip l2_rows2_scale 준거)
                {
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, self.n_group as u32]));
                    self.run_pipe("l2", L2_SPV, 2, 12,
                        &[self.b_gq.buf, self.b_gk.buf], &push, (2 * self.n_group) as u32, 1, 1)?;
                }
                // beta_g
                {
                    let dtb = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let push = Self::push_u32s(&[dt_rank as u32, dt_rank as u32]);
                    self.run_pipe("beta_g", BETA_G_SPV, 5, 8,
                        &[self.b_gb.buf, self.b_ga.buf, dtb.buf, ssa.buf, self.b_gbg.buf],
                        &push, dt_rank.div_ceil(64) as u32, 1, 1)?;
                }
                } // else (구 체인)
                if !arf_on && gskip & 1 == 0 {
                {
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32, dt_rank as u32, self.n_group as u32]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&1u32.to_le_bytes());
                    self.run_pipe("gdn_ar", GDN_AR_SPV, 6, 28,
                        &[self.st_gdn[recr_idx][seq].buf, self.b_gq.buf, self.b_gk.buf,
                          self.b_gv.buf, self.b_gbg.buf, self.b_go.buf],
                        &push, dt_rank as u32, d_state as u32, 1)?;
                }
                }
                // norm_gated (비트 2)
                if gskip & 2 == 0 {
                {
                    let sn = self.consts.get(&format!("blk.{il}.ssm_norm")).cloned().ok_or("sn")?;
                    // PC: {float eps; int d=d_state; int n_h=dt_rank} — rawhip norm_gated_silu 준거
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, dt_rank as u32]));
                    self.run_pipe("norm_gated", NORM_GATED_SPV, 4, 12,
                        &[self.b_go.buf, self.b_gz.buf, sn.buf, self.b_ggated.buf],
                        &push, dt_rank as u32, 1, 1)?;
                }
                }
                if gskip & 4 == 0 {
                    self.gemv_w(self.b_ggated.buf, self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, 1, d_inner)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il < 2 {
                    self.ctx.end_batch_wait().ok();
                    self.ctx.begin_batch().ok();
                    let mut v = vec![0f32; n];
                    unsafe { std::ptr::copy_nonoverlapping(self.b_gout.ptr as *const f32, v.as_mut_ptr(), n) };
                    let sum: f64 = v.iter().map(|&x| x as f64).sum();
                    let s = |b: &VkBuf, len: usize| -> f64 {
                        let mut x = vec![0f32; len];
                        unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, x.as_mut_ptr(), len) };
                        x.iter().map(|&q| q as f64).sum()
                    };
                    let srow = |b: &VkBuf, r: usize, len: usize| -> f64 {
                        let mut x = vec![0f32; len];
                        unsafe { std::ptr::copy_nonoverlapping(b.ptr.add(r * len * 4) as *const f32, x.as_mut_ptr(), len) };
                        x.iter().map(|&q| q as f64).sum()
                    };
                    eprintln!("#  G0 il={il} gout={sum:.6} | xs0={:.4} xs1={:.4} xn0={:.4} xn1={:.4} gqkv0={:.4} gconv0={:.4} gq0={:.4} go0={:.4} ggated0={:.4}",
                        srow(&self.b_xs, 0, 64), srow(&self.b_xs, 1, 64),
                        srow(&self.b_xn, 0, 64), srow(&self.b_xn, 1, 64),
                        s(&self.b_gqkv, self.conv_ch.min(64)),
                        s(&self.b_gconv, self.conv_ch.min(64)), s(&self.b_gq, self.k_len.min(64)),
                        s(&self.b_go, self.v_len.min(64)), s(&self.b_ggated, self.d_inner.min(64)));
                }
                recr_idx += 1;
            } else {
                // 어텐션 (LLM170_VK_ATTN: 1=qkv gemv만, 2=+rope/kv, 3=+flash, 4=+wo)
                let attn_cut = std::env::var("LLM170_VK_ATTN").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 3 {
                    let (bufs, tyq, niq, noq) = self.w.get(&format!("blk.{il}.attn_q.weight")).unwrap();
                    let total: usize = bufs.iter().map(|b| b.bytes).sum();
                    let (gbufs, _, _, gno) = self.w.get("blk.0.attn_qkv.weight").unwrap();
                    let gtotal: usize = gbufs.iter().map(|b| b.bytes).sum();
                    eprintln!("#  ATTN3 ty={} ni={} no={} chunks={} bytes={} max_ssbo={} | L0qkv no={} chunks={} bytes={}", tyq, niq, noq, bufs.len(), total, self.max_ssbo, gno, gbufs.len(), gtotal);
                }
                self.gemv_stage(n, 1, &[
                    (format!("blk.{il}.attn_q.weight"), self.b_xq_n.buf, self.b_aq.buf),
                    (format!("blk.{il}.attn_k.weight"), self.b_xq_n.buf, self.b_ak.buf),
                    (format!("blk.{il}.attn_v.weight"), self.b_xq_n.buf, self.b_av.buf),
                ])?;
                // qk_rope
                {
                    let qn = self.consts.get(&format!("blk.{il}.attn_q_norm")).cloned().ok_or("qn")?;
                    let kn = self.consts.get(&format!("blk.{il}.attn_k_norm")).cloned().ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    // PC: {float eps; float kqs; int pos; int nh; int nk; int hd; int nr} — gdn-check 준거
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
                    self.run_pipe("qk_rope2", QK_ROPE2_SPV, 5, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push, (n_head + n_kv) as u32, 1, 1)?;
                }
                if attn_cut >= 2 {
                // kv append
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
                    // k/v 어펜드는 상호 독립 — k 배리어 생략, v가 종결 (flash는 둘 다 판독)
                    if kv8 {
                        let gq = (n_kv * hd).div_ceil(32).div_ceil(64) as u32;
                        self.run_pipe_b("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push, gq, 1, 1, false)?;
                        self.run_pipe("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push, gq, 1, 1)?;
                    } else {
                        self.run_pipe_b("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, 1, 1, false)?;
                        self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
                    }
                }
                if attn_cut >= 3 {
                // flash
                {
                    let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32]);
                    if kv8 {
                        self.run_pipe("qsa_flash_q8", QSA_FLASH_Q8_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, 1, n_head as u32, 1)?;
                    } else {
                        self.run_pipe("qsa_flash", QSA_FLASH_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, 1, n_head as u32, 1)?;
                    }
                }
                if npck && il < 4 {
                    let b = self.b_aout.clone();
                    self.npck_mark("flash", il, &b, 0, 64);
                }
                if attn_cut >= 4 {
                    self.gemv_w(self.b_aout.buf, self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, 1, n_head * hd)?;
                }
                }
                }
                full_idx += 1;
            }
            // 잔차 + post_norm — addrms 융합 (plans/36 G2: axpy+rms 비트동일)
            self.addrms(self.b_xs.buf, self.b_gout.buf, &format!("blk.{il}.post_norm"), self.b_xn.buf, n, 1)?;
            // ── FFN
            self.gemv_stage(n, 1, &[
                (format!("blk.{il}.ffn_gate.weight"), self.b_xq_n.buf, self.b_fgate.buf),
                (format!("blk.{il}.ffn_up.weight"), self.b_xq_n.buf, self.b_fup.buf),
            ])?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
            self.gemv_w(self.b_fglu.buf, self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, 1, self.n_ff)?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합
            let is_last = il + 1 >= self.n_layer || layer_cut.is_some_and(|c| il + 1 >= c);
            let nkey = if is_last {
                "output_norm".to_string()
            } else {
                format!("blk.{}.attn_norm", il + 1)
            };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, 1)?;
            if npck {
                let b = self.b_xn.clone();
                self.npck_mark("L", il, &b, 0, 64);
            }
            // 실험: L0 FFN 직후 attn_q gemv 강제 (층 위치 vs 가중치 분리)
            if std::env::var_os("LLM170_VK_FORCE_AQ").is_some() && il == 0 {
                self.gemv_w(self.b_xn.buf, self.b_xq_n.buf, "blk.3.attn_q.weight", self.b_aq.buf, 1, n)?;
            }
        }
        // ── head: gemv(output) — output_norm은 마지막 addrms에 융합, quant는
        // gemv_w 폴백 시 내부 수행. 트렁크와 동일 배치로 단일 제출·대기 (G3).
        let tw_head1 = std::time::Instant::now();
        self.gemv_w(self.b_xn.buf, self.b_xq_n.buf, "output.weight", self.b_lg.buf, 1, n)?;
        self.ctx.end_batch_wait()?;
        self.ctx.ts_report();
        if std::env::var_os("LLM170_DBG_WALL").is_some() {
            eprintln!("[step] head+wait={:.2}ms step총={:.2}ms",
                tw_head1.elapsed().as_secs_f64()*1e3, vk_t0.elapsed().as_secs_f64()*1e3);
        }
        if self.ktime {
            let mut v: Vec<_> = self.ktimes.iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let tot: f64 = v.iter().map(|(_, (e, _))| *e).sum();
            eprintln!("[ktime1] wall {:.1}ms · 커널합 {tot:.1}ms", vk_t0.elapsed().as_secs_f32() * 1e3);
            for (k, (e, c)) in v.iter().take(12) {
                eprintln!("[ktime1] {:22} {:9.1}ms ({}회)", k, e, c);
            }
            self.ktimes.clear();
        }
        if std::env::var_os("LLM170_VK_PROF").is_some() { eprintln!("[vkprof] step: {:.1}ms (pos {})", vk_t0.elapsed().as_secs_f32()*1e3, pos); }
        if std::env::var_os("LLM170_VKD_TRACE").is_some() {
            let s = |b: &VkBuf, len: usize| -> f64 {
                let mut x = vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, x.as_mut_ptr(), len) };
                x.iter().map(|&q| q as f64).sum()
            };
            eprintln!("#  ST state seq={seq} conv0={:.6} gdn0={:.6} kvk3r0={:.6} kvv3r0={:.6} xsL={:.6}",
                s(&self.st_conv[0][seq], 30720.min(self.conv_ch * 3)),
                s(&self.st_gdn[0][seq], 4096),
                s(&self.kv_k[0][seq], 1024), s(&self.kv_v[0][seq], 1024),
                s(&self.b_xs, 64));
        }
        Ok(())
    }

    /// step_core + 전사 로짓 (기존 계약). raw_step_greedy는 아래 lg_argmax 경로로
    /// 608KB CPU 판독을 우회한다 (GTT 비캐시 판독 ~5ms/토큰 절감, plans/46).
    pub fn step(&mut self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        self.step_core(seq, pos, emb)?;
        let mut logits = vec![0f32; self.n_vocab];
        unsafe { std::ptr::copy_nonoverlapping(self.b_lg.ptr as *const f32, logits.as_mut_ptr(), self.n_vocab) };
        Ok(logits)
    }

    /// b_lg 상주 로짓의 GPU argmax — CPU greedy_from과 동일 의미(첫 최댓값=최저 인덱스).
    pub(super) fn lg_argmax(&mut self) -> Result<u32, String> {
        let nthr = 256usize;
        let chunk = 8usize;
        let n = self.n_vocab;
        let n_wg = n.div_ceil(nthr * chunk);
        let push0 = Self::push_u32s(&[n as u32, 0u32]);
        let push1 = Self::push_u32s(&[n_wg as u32, 1u32]);
        let binds = [self.b_lg.buf, self.b_ams.buf, self.b_am.buf];
        self.ctx.begin_batch()?;
        self.run_pipe_b("argmax2", crate::rawvk::vkacc::ARGMAX2_SPV, 3, 8,
            &binds, &push0, n_wg as u32, 1, 1, true)?;
        self.run_pipe_b("argmax2", crate::rawvk::vkacc::ARGMAX2_SPV, 3, 8,
            &binds, &push1, 1, 1, 1, true)?;
        self.ctx.end_batch_wait()?;
        Ok(unsafe { *(self.b_am.ptr as *const u32) })
    }

    /// t행 배치 스텝 (plans/20) — 가중 1회 판독 분할 상각. 행별 산술은
    /// step()과 비트 동일(gemv3 행별 lane 축산·AR 내부 순차·conv 이력 판독).
    /// all_logits=true: 전 행 head 로짓 [t][n_vocab] (verify용 — b_lg_t).
    /// 아니면 마지막 행만 (b_lg). emb는 [t][n_embd].
    pub fn step_batch(&mut self, seq: usize, pos0: usize, emb: &[f32], all_logits: bool) -> Result<Vec<f32>, String> {
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
        let _vk_t0b = std::time::Instant::now();
        let n = self.n_embd;
        let t = emb.len() / n;
        if t == 0 || emb.len() != t * n || t > T_MAX {
            return Err(format!("step_batch t={t} (1..={T_MAX})"));
        }
        let (dt_rank, d_state, d_inner) = (self.dt_rank, self.d_state, self.d_inner);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        // plans/92 P2: [pfck] 업로드·제출대기·헤드·판독 4분해 (LLM170_PFCK=1).
        let pfck = std::env::var_os("LLM170_PFCK").is_some();
        let pf_up0 = std::time::Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, t * n);
        }
        let pf_up = pf_up0.elapsed().as_secs_f64() * 1e3;
        let pf_gpu0 = std::time::Instant::now();
        if std::env::var_os("LLM170_VKD_TRACE").is_some() {
            let mut x = vec![0f32; 64];
            unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, x.as_mut_ptr(), 64) };
            let s0: f64 = x.iter().map(|&v| v as f64).sum();
            eprintln!("#  SB upload t={t} xs0={s0:.4}");
        }
        self.ctx.begin_batch()?;
        let tw_rec = std::time::Instant::now();
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let vkd_stage = std::env::var_os("LLM170_VKD_STAGE").is_some();
        let mut il_t = std::time::Instant::now();
        for il in 0..self.n_layer {
            if vkd_stage {
                // 직전 레이어 시간 출력(루프 끝을 몰라도 되는 형태) 후 리셋.
                if il > 0 {
                    eprintln!("# vkd L{} {:.1}ms", il - 1, il_t.elapsed().as_secs_f64() * 1e3);
                }
                il_t = std::time::Instant::now();
            }
            // ── attn_norm — 0층만 (이후 fdown addrms 융합). xq는 gemv_stage 지연 양자화.
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, t)?;
            }
            if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                let _x = vec![0f32; 64];
                let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                eprintln!("#  SB post-rms xs0={s0:.4}");
            }
            if self.is_recr[il] {
                // plans/30: gemm_i8/quant_b8 경로는 배치 상태를 오염(실측 —
                // VK_NOI8=1로 재현 해소). LLM170_VK_I8ON=1 옵트인만 사용.
                if t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some() {
                    self.quant_b8(self.b_xn.buf, n, t)?;
                }
                self.gemv_stage(n, t, &[
                    (format!("blk.{il}.attn_qkv.weight"), self.b_xq_n.buf, self.b_gqkv.buf),
                    (format!("blk.{il}.attn_gate.weight"), self.b_xq_n.buf, self.b_gz.buf),
                    (format!("blk.{il}.ssm_beta.weight"), self.b_xq_n.buf, self.b_gb.buf),
                    (format!("blk.{il}.ssm_alpha.weight"), self.b_xq_n.buf, self.b_ga.buf),
                ])?;
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    let mut g = vec![0f32; 8];
                    unsafe { std::ptr::copy_nonoverlapping(self.b_gqkv.ptr as *const f32, g.as_mut_ptr(), 8) };
                    let gq: f64 = unsafe { std::slice::from_raw_parts(self.b_gqkv.ptr as *const f32, 128) }.iter().map(|&v| v as f64).sum();
                    let d0 = format!("{:?}", g);
                    let gz8: Vec<f32> = unsafe { std::slice::from_raw_parts(self.b_gz.ptr as *const f32, 8) }.to_vec();
                    let gb8: Vec<f32> = unsafe { std::slice::from_raw_parts(self.b_gb.ptr as *const f32, 4) }.to_vec();
                    eprintln!("#  stage2 gz={:?} gb={:?}", gz8, gb8);
                    let mut b8v = [0i8; 16];
                    unsafe { std::ptr::copy_nonoverlapping(self.b8.ptr as *const i8, b8v.as_mut_ptr(), 16) };
                    let b8r1: Vec<i8> = unsafe { std::slice::from_raw_parts(self.b8.ptr as *const i8, 32) }[16..].to_vec();
                    let mut ydv = [0f32; 4];
                    unsafe { std::ptr::copy_nonoverlapping(self.ydb.ptr as *const f32, ydv.as_mut_ptr(), 4) };
                    let mut qsv = [0i32; 4];
                    unsafe { std::ptr::copy_nonoverlapping(self.qsb.ptr as *const i32, qsv.as_mut_ptr(), 4) };
                    eprintln!("#  SB post-gemv4 xs0={s0:.4} gqkv0={gq:.4} first8={d0} b8={:?} b8tail={:?} yd={:?} qs={:?}", b8v.to_vec(), b8r1, ydv.to_vec(), qsv.to_vec());
                }
                // conv — gy=t (이력은 qkv에서 판독, t>1은 링을 conv_state가 갱신)
                {
                    let cw = self.consts.get(&format!("blk.{il}.conv_w")).cloned().ok_or("conv_w")?;
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32]);
                    self.run_pipe("gdn_conv", GDN_CONV_SPV, 4, 12,
                        &[self.b_gqkv.buf, cw.buf, self.st_conv[recr_idx][seq].buf, self.b_gconv.buf],
                        &push, conv_ch.div_ceil(64) as u32, t as u32, 1)?;
                    if t > 1 {
                        let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32]);
                        self.run_pipe("gdn_conv_state", GDN_CONV_STATE_SPV, 2, 12,
                            &[self.b_gqkv.buf, self.st_conv[recr_idx][seq].buf],
                            &push, conv_ch.div_ceil(64) as u32, 1, 1)?;
                    }
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-conv xs0={s0:.4}");
                }
                // plans/46: 프리필 융합 AR8 (split3+l2+beta_g 인라인) — 기본.
                let ar8f_on = std::env::var("LLM170_VK_AR8F").map(|v| v != "0").unwrap_or(true);
                if ar8f_on {
                    let dtb = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32,
                        dt_rank as u32, self.n_group as u32]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&(t as u32).to_le_bytes());
                    push.extend_from_slice(&self.eps.to_le_bytes());
                    self.run_pipe("gdn_ar8f", GDN_AR8F_SPV, 7, 32,
                        &[self.st_gdn[recr_idx][seq].buf, self.b_gconv.buf,
                          self.b_gb.buf, self.b_ga.buf, dtb.buf, ssa.buf, self.b_go.buf],
                        &push, dt_rank as u32, d_state as u32 / 8, 1)?;
                } else {
                // split3 — flat total*t
                {
                    let total = 2 * k_len + v_len;
                    let push = Self::push_u32s(&[k_len as u32, k_len as u32, v_len as u32]);
                    self.run_pipe("split3", SPLIT3_SPV, 4, 12,
                        &[self.b_gconv.buf, self.b_gq.buf, self.b_gk.buf, self.b_gv.buf],
                        &push, (total * t).div_ceil(64) as u32, 1, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-split3 xs0={s0:.4}");
                }
                // l2 — grid (2*ng, t)
                {
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, self.n_group as u32]));
                    self.run_pipe("l2", L2_SPV, 2, 12,
                        &[self.b_gq.buf, self.b_gk.buf], &push, (2 * self.n_group) as u32, t as u32, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-l2 xs0={s0:.4}");
                }
                // beta_g — n_h = dt_rank*t
                {
                    let dtb = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let push = Self::push_u32s(&[(dt_rank * t) as u32, dt_rank as u32]);
                    self.run_pipe("beta_g", BETA_G_SPV, 5, 8,
                        &[self.b_gb.buf, self.b_ga.buf, dtb.buf, ssa.buf, self.b_gbg.buf],
                        &push, (dt_rank * t).div_ceil(64) as u32, 1, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-betag xs0={s0:.4}");
                }
                // AR — PC.t 내부 순차
                {
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32, dt_rank as u32, self.n_group as u32]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&(t as u32).to_le_bytes());
                    // plans/40: ar4 — 열 4개 ILP. 옵트아웃 LLM170_VK_AR4=0.
                    let arsel = std::env::var("LLM170_VK_AR4").unwrap_or_else(|_| "8".into());
                    let (arnm, arspv, argy) = match arsel.as_str() {
                        "0" => ("gdn_ar", GDN_AR_SPV, d_state as u32),
                        "4" => ("gdn_ar4", GDN_AR4_SPV, d_state as u32 / 4),
                        _ => ("gdn_ar8", GDN_AR8_SPV, d_state as u32 / 8),
                    };
                    self.run_pipe(arnm, arspv, 6, 28,
                        &[self.st_gdn[recr_idx][seq].buf, self.b_gq.buf, self.b_gk.buf,
                          self.b_gv.buf, self.b_gbg.buf, self.b_go.buf],
                        &push, dt_rank as u32, argy, 1)?;
                }
                } // else (구 체인)
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-ar xs0={s0:.4}");
                }
                // norm_gated — grid (dt_rank, t)
                {
                    let sn = self.consts.get(&format!("blk.{il}.ssm_norm")).cloned().ok_or("sn")?;
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, dt_rank as u32]));
                    self.run_pipe("norm_gated", NORM_GATED_SPV, 4, 12,
                        &[self.b_go.buf, self.b_gz.buf, sn.buf, self.b_ggated.buf],
                        &push, dt_rank as u32, t as u32, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-normgated xs0={s0:.4}");
                }
                self.gemv_w(self.b_ggated.buf, self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, t, d_inner)?;
                recr_idx += 1;
            } else {
                // i8 활성(소비 조건과 동일)일 때만 b8 양자화 — 기본 경로의 dead dispatch 제거
                if t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some()
                    && std::env::var_os("LLM170_VK_NOI8").is_none()
                    && self.i8w.contains_key(&format!("blk.{il}.attn_q.weight")) {
                    self.quant_b8(self.b_xn.buf, n, t)?;
                }
                self.gemv_stage(n, t, &[
                    (format!("blk.{il}.attn_q.weight"), self.b_xq_n.buf, self.b_aq.buf),
                    (format!("blk.{il}.attn_k.weight"), self.b_xq_n.buf, self.b_ak.buf),
                    (format!("blk.{il}.attn_v.weight"), self.b_xq_n.buf, self.b_av.buf),
                ])?;
                // qk_rope — grid (nh+nk, t), pos = pos0+행
                {
                    let qn = self.consts.get(&format!("blk.{il}.attn_q_norm")).cloned().ok_or("qn")?;
                    let kn = self.consts.get(&format!("blk.{il}.attn_k_norm")).cloned().ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
                    self.run_pipe("qk_rope2", QK_ROPE2_SPV, 5, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push, (n_head + n_kv) as u32, t as u32, 1)?;
                }
                // kv append — grid (n/64, t). k/v 상호 독립 — k 배리어 생략, v가 종결
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos0 as u32]);
                    if kv8 {
                        let gq = (n_kv * hd).div_ceil(32).div_ceil(64) as u32;
                        self.run_pipe_b("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push, gq, t as u32, 1, false)?;
                        self.run_pipe("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push, gq, t as u32, 1)?;
                    } else {
                        self.run_pipe_b("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, t as u32, 1, false)?;
                        self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, t as u32, 1)?;
                    }
                }
                // flash — 프리필(t≥2, GQA ≤6:1)은 다중쿼리 판(plans/83 D):
                // K/V 타일을 24쿼리가 공유해 장문 프리필(pp4096) 어텐션 트래픽·
                // 지연을 1/24로 줄인다. 폴백(구 판)은 LLM170_VK_NOGQ=1.
                if t >= 2 && !kv8 && n_head / n_kv.max(1) <= 6 && std::env::var_os("LLM170_VK_NOGQ").is_none() {
                    // plans/92 P3: 레지스터 상주판(qsa_flash_reg) — hip wk16 구조
                    // 이식(LDS·배리어 0, 점유 8WG/CU급). 종전 gq는 LDS 61KB/WG로
                    // 점유 1WG/CU — 장문 프리필 어텐션이 npmax 선형 지연의 주벚.
                    // hd≠256·킬스위치(LLM170_VK_NOREG=1)는 gq로.
                    if hd == 256 && std::env::var("LLM170_VK_NOREG").map(|v| v != "1").unwrap_or(true) {
                        let push = Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32, t as u32]);
                        self.run_pipe("qsa_flash_reg", QSA_FLASH_REG_SPV, 4, 20,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, (t as u32).div_ceil(16), n_head as u32, 1)?;
                    } else {
                        let push = Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32, t as u32]);
                        self.run_pipe("qsa_flash_gq", QSA_FLASH_GQ_SPV, 4, 20,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, (t as u32).div_ceil(4), n_kv as u32, 1)?;
                    }
                } else {
                    let push = Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32]);
                    if kv8 {
                        self.run_pipe("qsa_flash_q8", QSA_FLASH_Q8_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, t as u32, n_head as u32, 1)?;
                    } else {
                        self.run_pipe("qsa_flash", QSA_FLASH_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, t as u32, n_head as u32, 1)?;

                    }
                }
                self.gemv_w(self.b_aout.buf, self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, t, n_head * hd)?;
                full_idx += 1;
            }
            // 잔차 + post_norm — addrms 융합 (t행)
            self.addrms(self.b_xs.buf, self.b_gout.buf, &format!("blk.{il}.post_norm"), self.b_xn.buf, n, t)?;
            // FFN — xq는 gemv_stage 지연 양자화
            if t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some() {
                self.quant_b8(self.b_xn.buf, n, t)?;
            }
            self.gemv_stage(n, t, &[
                (format!("blk.{il}.ffn_gate.weight"), self.b_xq_n.buf, self.b_fgate.buf),
                (format!("blk.{il}.ffn_up.weight"), self.b_xq_n.buf, self.b_fup.buf),
            ])?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff * t)?;
            self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, t)?;
            self.gemv_w(self.b_fglu.buf, self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, t, self.n_ff)?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합
            let is_last = il + 1 >= self.n_layer;
            let nkey = if is_last {
                "output_norm".to_string()
            } else {
                format!("blk.{}.attn_norm", il + 1)
            };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, t)?;
        }
        // ── head (all_logits) — output_norm은 마지막 addrms에 융합. 트렁크와
        // 동일 배치로 단일 제출·대기 (G3). quant는 gemv_w 폴백 시 내부 수행.
        if all_logits {
            self.gemv_w(self.b_xn.buf, self.b_xq_n.buf, "output.weight", self.b_lg_t.buf, t, n)?;
        }
        if std::env::var_os("LLM170_DBG_REC").is_some() {
            let d = self.dbg_drain_ms;
            self.dbg_drain_ms = 0.0;
            eprintln!("#  rec t={t} span={:.1}ms drain={:.1}ms", tw_rec.elapsed().as_secs_f64() * 1e3, d);
        }
        let pf_rec = pf_gpu0.elapsed().as_secs_f64() * 1e3;
        let pf_w0 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        let pf_wait = pf_w0.elapsed().as_secs_f64() * 1e3;
        self.ctx.ts_report();
        let pf_head0 = std::time::Instant::now();
        if self.ktime {
            let mut v: Vec<_> = self.ktimes.iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let tot: f64 = v.iter().map(|(_, (e, _))| *e).sum();
            eprintln!("[ktime] t={t} 총 {tot:.0}ms");
            for (k, (e, c)) in v.iter().take(14) {
                eprintln!("[ktime] {:22} {:9.1}ms ({}회)", k, e, c);
            }
        }
        if std::env::var_os("LLM170_VKD_TRACE").is_some() {
            let s = |b: &VkBuf, len: usize| -> f64 {
                let mut x = vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, x.as_mut_ptr(), len) };
                x.iter().map(|&q| q as f64).sum()
            };
            eprintln!("#  SB state seq={seq} conv0={:.6} gdn0={:.6} kvk3r0={:.6} kvv3r0={:.6} xsL={:.6}",
                s(&self.st_conv[0][seq], 30720.min(self.conv_ch * 3)),
                s(&self.st_gdn[0][seq], 4096),
                s(&self.kv_k[0][seq], 1024), s(&self.kv_v[0][seq], 1024),
                s(&self.b_xs, 64));
        }
        // 마지막 행 head — b_xn 마지막 행이 이미 output_norm 융합 결과
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.b_xn.ptr.add((t - 1) * n * 4) as *const f32,
                self.m_e.ptr as *mut f32, n);
        }
        self.ctx.begin_batch()?;
        self.gemv_w(self.m_e.buf, self.m_xq.buf, "output.weight", self.b_lg.buf, 1, n)?;
        self.ctx.end_batch_wait()?;
        let pf_head = pf_head0.elapsed().as_secs_f64() * 1e3;
        let pf_rd0 = std::time::Instant::now();
        let mut logits = vec![0f32; self.n_vocab];
        unsafe { std::ptr::copy_nonoverlapping(self.b_lg.ptr as *const f32, logits.as_mut_ptr(), self.n_vocab) };
        if pfck {
            eprintln!("[pfck] step_batch t={t} up={pf_up:.1}ms rec={pf_rec:.1}ms wait={pf_wait:.1}ms head={pf_head:.1}ms rd={:.1}ms",
                pf_rd0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(logits)
    }

    /// np 배치 디코드 (plans/91 P0) — t행(슬롯별 시퀀스) 단일 패스.
    /// GEMM/요소커널은 t행 공유(무게 1회 판독 — 종전 순차 루프의 t배 비용 제거),
    /// 상태커널(conv/AR/rope/KV/flash)은 행별 상태를 디바이스 주소 테이블로
    /// 단일 런치(rawhip step_batch_np 구조의 vk 이식). 행별 산술은 step()
    /// (t=1 순차 루프)과 비트 동일 — gemv8/gemv3 행별 축산·conv 링 시프트·
    /// AR 내부 순서·flash 루프 상한 모두 행 독립.
    /// greedy=false: 전 행 로짓 [t][n_vocab] 반환. greedy=true: GPU 행별
    /// argmax(fn_argmax_rows — 동률 최저 인덱스, CPU greedy와 동일 의미).
    pub fn step_batch_np_ex(
        &mut self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
        greedy: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<u32>), String> {
        let t = seqs.len();
        let n = self.n_embd;
        debug_assert_eq!(emb.len(), t * n);
        if t == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        // 비기본 경로(kv8, ARF 폴백)는 순차 루프 — 계약 동일, np 커널은 기본
        // 경로(arf 융합·f32 KV)만 커버.
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
        let arf_on = std::env::var("LLM170_VK_ARF").map(|v| v != "0").unwrap_or(true);
        if kv8 || !arf_on || t == 1 {
            let mut out = Vec::with_capacity(t);
            let mut toks = Vec::with_capacity(t);
            for (i, (&sq, &ps)) in seqs.iter().zip(poss.iter()).enumerate() {
                if greedy {
                    self.step_core(sq, ps as usize, &emb[i * n..(i + 1) * n])?;
                    toks.push(self.lg_argmax()?);
                } else {
                    out.push(self.step(sq, ps as usize, &emb[i * n..(i + 1) * n])?);
                }
            }
            return Ok((out, toks));
        }
        let ns = self.st_gdn.first().map(|g| g.len()).unwrap_or(0);
        if seqs.iter().any(|&s| s >= ns) {
            return Err("np: 슬롯 범위 초과".into());
        }
        let (dt_rank, d_state, d_inner) = (self.dt_rank, self.d_state, self.d_inner);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        let npck = std::env::var_os("LLM170_VK_NPCK").is_some();
        if npck {
            let sl: Vec<u32> = unsafe { std::slice::from_raw_parts(self.np_slot.ptr as *const u32, t) }.to_vec();
            let ps: Vec<u32> = unsafe { std::slice::from_raw_parts(self.np_pos.ptr as *const u32, t) }.to_vec();
            let tb: Vec<u64> = unsafe { std::slice::from_raw_parts(self.np_gdn_tbl.ptr as *const u64, 2.min(ns)) }.to_vec();
            let va: Vec<u64> = (0..2.min(ns)).map(|s| self.ctx.buffer_va(self.st_gdn[0][s].buf)).collect();
            eprintln!("[npck] t={t} ns={ns} slot={sl:?} pos={ps:?} tbl={tb:#x?} va={va:#x?}");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, t * n);
            // 행별 pos/slot 맵 (u32 테이블 — 셰이더가 행 인덱스로 판독).
            // slot: usize 배열의 저바이트 재해석 금지 — 원소 변환 후 기입
            // (종전 as *const u32 재해석이 [0,0]을 써 np 상태가 전행 슬롯0에
            // 겹쳐 쓰였다 — [npck] st1out 불변으로 국소화, plans/91 P0).
            let slot_v: Vec<u32> = seqs.iter().map(|&s| s as u32).collect();
            std::ptr::copy_nonoverlapping(poss.as_ptr(), self.np_pos.ptr as *mut u32, t);
            std::ptr::copy_nonoverlapping(slot_v.as_ptr(), self.np_slot.ptr as *mut u32, t);
        }
        self.ctx.begin_batch()?;
        let np_t0 = std::time::Instant::now();
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let layer_cut = std::env::var("LLM170_VK_LAYERS").ok().and_then(|v| v.parse::<usize>().ok());
        let n_layer_eff = layer_cut.map(|c| c.min(self.n_layer)).unwrap_or(self.n_layer);
        for il in 0..n_layer_eff {
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, t)?;
            }
            if self.is_recr[il] {
                // GDN 4 GEMV 공유 (t행 — 무게 1회)
                self.gemv_stage(n, t, &[
                    (format!("blk.{il}.attn_qkv.weight"), self.b_xq_n.buf, self.b_gqkv.buf),
                    (format!("blk.{il}.attn_gate.weight"), self.b_xq_n.buf, self.b_gz.buf),
                    (format!("blk.{il}.ssm_beta.weight"), self.b_xq_n.buf, self.b_gb.buf),
                    (format!("blk.{il}.ssm_alpha.weight"), self.b_xq_n.buf, self.b_ga.buf),
                ])?;
                if npck && il == 0 {
                    let s0b = self.st_gdn[0][seqs[0]].clone();
                    self.npck_mark("stin", il, &s0b, 0, 64);
                    let gb = self.b_gb.clone();
                    self.npck_mark("gb", il, &gb, 0, 16);
                    let ga = self.b_ga.clone();
                    self.npck_mark("ga", il, &ga, 0, 16);
                }
                // conv — 행별 링 상태 1런치 (plans/91 P0)
                {
                    let cw = self.consts.get(&format!("blk.{il}.conv_w")).cloned().ok_or("conv_w")?;
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32,
                        recr_idx as u32, ns as u32]);
                    self.run_pipe("gdn_conv_np", GDN_CONV_NP_SPV, 5, 20,
                        &[self.b_gqkv.buf, cw.buf, self.np_conv_tbl.buf, self.b_gconv.buf, self.np_slot.buf],
                        &push, conv_ch.div_ceil(64) as u32, t as u32, 1)?;
                }
                if npck && il < 3 {
                    let b = self.b_gconv.clone();
                    self.npck_mark("conv", il, &b, 0, 64);
                }
                // AR — 융합 판(arf)의 행별 상태 1런치 (split3+l2+beta_g+AR 인라인)
                {
                    let dtb = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32,
                        dt_rank as u32, self.n_group as u32]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&(t as u32).to_le_bytes());
                    push.extend_from_slice(&self.eps.to_le_bytes());
                    push.extend_from_slice(&(recr_idx as u32).to_le_bytes());
                    push.extend_from_slice(&(ns as u32).to_le_bytes());
                    self.run_pipe("gdn_arf_np", GDN_ARF_NP_SPV, 8, 40,
                        &[self.np_gdn_tbl.buf, self.b_gconv.buf,
                          self.b_gb.buf, self.b_ga.buf, dtb.buf, ssa.buf, self.b_go.buf,
                          self.np_slot.buf],
                        &push, dt_rank as u32, d_state as u32, 1)?;
                }
                if npck && il == 0 {
                    let s0b = self.st_gdn[0][seqs[0]].clone();
                    self.npck_mark("stout", il, &s0b, 0, 64);
                }
                if npck && il == 0 && seqs.len() > 1 {
                    let s1b = self.st_gdn[0][seqs[1]].clone();
                    self.npck_mark("st1out", il, &s1b, 0, 64);
                }
                if npck && il < 3 {
                    let b = self.b_go.clone();
                    self.npck_mark("ar", il, &b, 0, 64);
                }
                // norm_gated — grid (dt_rank, t)
                {
                    let sn = self.consts.get(&format!("blk.{il}.ssm_norm")).cloned().ok_or("sn")?;
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, dt_rank as u32]));
                    self.run_pipe("norm_gated", NORM_GATED_SPV, 4, 12,
                        &[self.b_go.buf, self.b_gz.buf, sn.buf, self.b_ggated.buf],
                        &push, dt_rank as u32, t as u32, 1)?;
                }
                self.gemv_w(self.b_ggated.buf, self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, t, d_inner)?;
                recr_idx += 1;
            } else {
                // 어텐션 3 GEMV 공유 (t행)
                self.gemv_stage(n, t, &[
                    (format!("blk.{il}.attn_q.weight"), self.b_xq_n.buf, self.b_aq.buf),
                    (format!("blk.{il}.attn_k.weight"), self.b_xq_n.buf, self.b_ak.buf),
                    (format!("blk.{il}.attn_v.weight"), self.b_xq_n.buf, self.b_av.buf),
                ])?;
                // rope — 행별 pos 테이블 (grid nh+nk, t)
                {
                    let qn = self.consts.get(&format!("blk.{il}.attn_q_norm")).cloned().ok_or("qn")?;
                    let kn = self.consts.get(&format!("blk.{il}.attn_k_norm")).cloned().ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
                    self.run_pipe("qk_rope2_np", QK_ROPE2_NP_SPV, 6, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf, self.np_pos.buf],
                        &push, (n_head + n_kv) as u32, t as u32, 1)?;
                }
                // kv append — 행별 KV 캐시 (k 배리어 생략, v가 종결 — flash는 둘 다 판독)
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, full_idx as u32, ns as u32]);
                    let binds_k: Vec<vk::Buffer> = vec![self.b_ak.buf, self.np_kvk_tbl.buf, self.np_pos.buf, self.np_slot.buf];
                    self.run_pipe_b("kv_app_np", KV_APP_NP_SPV, 4, 12,
                        &binds_k, &push, (n_kv * hd).div_ceil(64) as u32, t as u32, 1, false)?;
                    let binds_v: Vec<vk::Buffer> = vec![self.b_av.buf, self.np_kv_v_tbl.buf, self.np_pos.buf, self.np_slot.buf];
                    self.run_pipe("kv_app_np", KV_APP_NP_SPV, 4, 12,
                        &binds_v, &push, (n_kv * hd).div_ceil(64) as u32, t as u32, 1)?;
                }
                // flash — 행별 KV·pos (grid t, n_head)
                {
                    let push = Self::push_u32s(&[n_head as u32, n_kv as u32, hd as u32, full_idx as u32, ns as u32]);
                    self.run_pipe("qsa_flash_np", QSA_FLASH_NP_SPV, 6, 20,
                        &[self.b_aq.buf, self.np_kvk_tbl.buf, self.np_kv_v_tbl.buf,
                          self.np_pos.buf, self.np_slot.buf, self.b_aout.buf],
                        &push, t as u32, n_head as u32, 1)?;
                }
                if npck && il < 4 {
                    let b = self.b_aout.clone();
                    self.npck_mark("flash", il, &b, 0, 64);
                }
                self.gemv_w(self.b_aout.buf, self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, t, n_head * hd)?;
                full_idx += 1;
            }
            // 잔차 + post_norm (t행)
            self.addrms(self.b_xs.buf, self.b_gout.buf, &format!("blk.{il}.post_norm"), self.b_xn.buf, n, t)?;
            // FFN 공유 (t행)
            self.gemv_stage(n, t, &[
                (format!("blk.{il}.ffn_gate.weight"), self.b_xq_n.buf, self.b_fgate.buf),
                (format!("blk.{il}.ffn_up.weight"), self.b_xq_n.buf, self.b_fup.buf),
            ])?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff * t)?;
            self.gemv_w(self.b_fglu.buf, self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, t, self.n_ff)?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합 (t행)
            let is_last = il + 1 >= self.n_layer;
            let nkey = if is_last { "output_norm".to_string() } else { format!("blk.{}.attn_norm", il + 1) };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, t)?;
            if npck {
                let b = self.b_xn.clone();
                self.npck_mark("L", il, &b, 0, 64);
            }
        }
        // ── head — b_xn 전 행이 이미 output_norm 융합 결과. t행 GEMV 1회.
        self.gemv_w(self.b_xn.buf, self.b_xq_n.buf, "output.weight", self.b_lg_t.buf, t, n)?;
        if greedy {
            // 행별 GPU argmax — fn_argmax_rows 2단계 (동률 최저 인덱스).
            // 트렁크와 동일 배치로 제출(별도 서브미션 제거).
            let nv = self.n_vocab;
            let n_wg = nv.div_ceil(256 * 8);
            let push0 = Self::push_u32s(&[nv as u32, 0u32, n_wg as u32]);
            let push1 = Self::push_u32s(&[nv as u32, 1u32, n_wg as u32]);
            let binds = [self.b_lg_t.buf, self.b_amsc.buf, self.b_amr.buf];
            self.run_pipe_b("fn_argmax_rows", crate::rawvk::vkacc::FN_ARGMAX_ROWS_SPV, 3, 12,
                &binds, &push0, n_wg as u32, t as u32, 1, true)?;
            self.run_pipe_b("fn_argmax_rows", crate::rawvk::vkacc::FN_ARGMAX_ROWS_SPV, 3, 12,
                &binds, &push1, 1, t as u32, 1, true)?;
        }
        let np_t1 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        if std::env::var_os("LLM170_NP_TIME").is_some() {
            eprintln!("[npstep] vk t={t} greedy={greedy} rec={:.1}ms wait={:.1}ms",
                (np_t1 - np_t0).as_secs_f64() * 1e3, np_t1.elapsed().as_secs_f64() * 1e3);
        }
        if greedy {
            let mut toks = vec![0u32; t];
            unsafe { std::ptr::copy_nonoverlapping(self.b_amr.ptr as *const u32, toks.as_mut_ptr(), t) };
            return Ok((Vec::new(), toks));
        }
        // plans/92 P6.4: 중간 flat 2.4MB + 행별 재복사 폐지 — 행 버퍼로 직복사.
        let mut rows: Vec<Vec<f32>> = (0..t).map(|_| vec![0f32; self.n_vocab]).collect();
        for (i, r) in rows.iter_mut().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.b_lg_t.ptr.add(i * self.n_vocab * 4) as *const f32,
                    r.as_mut_ptr(), self.n_vocab)
            };
        }
        Ok((rows, Vec::new()))
    }
    /// plans/91 P0 — [npck] 스테이지 마크: 배치-순차 대조용 행 합 덤프.
    fn npck_mark(&mut self, tag: &str, il: usize, b: &VkBuf, row: usize, len: usize) {
        self.ctx.end_batch_wait().ok();
        self.ctx.begin_batch().ok();
        let mut v = vec![0f32; len];
        unsafe { std::ptr::copy_nonoverlapping(b.ptr.add(row * len * 4) as *const f32, v.as_mut_ptr(), len) };
        let s: f64 = v.iter().map(|&x| x as f64).sum();
        eprintln!("[npck] {tag} il={il} sum={s:.6} v0={:.6}", v[0]);
    }
}
