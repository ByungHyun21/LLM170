//! vk decoder 단일 스텝 본체 (plans/79 B). 배치/np 경로는 step_batch·step_np 모듈.

use super::*;

impl DecoderState {
    /// t=1 단일 스텝 본체 — 배치 모드로 전 층 단일 제출·다운로드 1회.
    /// 로짓은 b_lg에만 남는다 (전사는 step() 래퍼).
    pub(super) fn step_core(&mut self, seq: usize, pos: usize, emb: &[f32]) -> Result<(), String> {
        let n = self.n_embd;
        debug_assert_eq!(emb.len(), n);
        let (dt_rank, d_state, d_inner) = (self.dt_rank, self.d_state, self.d_inner);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        // SAFETY (107 W8): b_xs 매핑 기입 — n 이하(단일 행), begin_batch 전이라 GPU 접근 없음.
        unsafe { std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, n) };
        let vk_t0 = std::time::Instant::now();
        self.ctx.begin_batch()?;
        let _tw_rec = std::time::Instant::now();
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let layer_cut =
            llm170_diag::flag::val("LLM170_VK_LAYERS").and_then(|v| v.parse::<usize>().ok());
        let npck = llm170_diag::flag::on("LLM170_VK_NPCK");
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
                self.gemv_stage(
                    n,
                    1,
                    &[
                        (
                            format!("blk.{il}.attn_qkv.weight"),
                            self.b_xq_n.buf,
                            self.b_gqkv.buf,
                        ),
                        (
                            format!("blk.{il}.attn_gate.weight"),
                            self.b_xq_n.buf,
                            self.b_gz.buf,
                        ),
                        (
                            format!("blk.{il}.ssm_beta.weight"),
                            self.b_xq_n.buf,
                            self.b_gb.buf,
                        ),
                        (
                            format!("blk.{il}.ssm_alpha.weight"),
                            self.b_xq_n.buf,
                            self.b_ga.buf,
                        ),
                    ],
                )?;
                let gskip = llm170_diag::flag::val("LLM170_VK_GDN_SKIP")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(0);
                // conv (t=1 — ring)
                {
                    let cw = self
                        .consts
                        .get(&format!("blk.{il}.conv_w"))
                        .cloned()
                        .ok_or("conv_w")?;
                    // PC: {int ch; int k; int t} + local 64 — gdn-check 준거
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, 1u32]);
                    self.run_pipe(
                        "gdn_conv",
                        GDN_CONV_SPV,
                        4,
                        12,
                        &[
                            self.b_gqkv.buf,
                            cw.buf,
                            self.st_conv[recr_idx][seq].buf,
                            self.b_gconv.buf,
                        ],
                        &push,
                        conv_ch.div_ceil(64) as u32,
                        1,
                        1,
                    )?;
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
                if gskip & 1 == 0 {
                    let dtb2 = self
                        .consts
                        .get(&format!("blk.{il}.dt_bias"))
                        .cloned()
                        .ok_or("dtb")?;
                    let ssa2 = self
                        .consts
                        .get(&format!("blk.{il}.ssm_a"))
                        .cloned()
                        .ok_or("ssa")?;
                    let scale2 = 1.0f32 / (d_state as f32).sqrt();
                    let mut push2 = Self::push_u32s(&[
                        d_state as u32,
                        k_len as u32,
                        v_len as u32,
                        dt_rank as u32,
                        self.n_group as u32,
                    ]);
                    push2.extend_from_slice(&scale2.to_le_bytes());
                    push2.extend_from_slice(&1u32.to_le_bytes());
                    push2.extend_from_slice(&self.eps.to_le_bytes());
                    self.run_pipe(
                        "gdn_arf",
                        GDN_ARF_SPV,
                        7,
                        32,
                        &[
                            self.st_gdn[recr_idx][seq].buf,
                            self.b_gconv.buf,
                            self.b_gb.buf,
                            self.b_ga.buf,
                            dtb2.buf,
                            ssa2.buf,
                            self.b_go.buf,
                        ],
                        &push2,
                        dt_rank as u32,
                        d_state as u32,
                        1,
                    )?;
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
                        self.run_pipe(
                            "split3",
                            SPLIT3_SPV,
                            4,
                            12,
                            &[
                                self.b_gconv.buf,
                                self.b_gq.buf,
                                self.b_gk.buf,
                                self.b_gv.buf,
                            ],
                            &push,
                            total.div_ceil(64) as u32,
                            1,
                            1,
                        )?;
                    }
                    // l2 — PC: {float eps; int d; int ng} (스케일은 AR이 적용 — rawhip l2_rows2_scale 준거)
                    {
                        let mut push = self.eps.to_le_bytes().to_vec();
                        push.extend(Self::push_u32s(&[d_state as u32, self.n_group as u32]));
                        self.run_pipe(
                            "l2",
                            L2_SPV,
                            2,
                            12,
                            &[self.b_gq.buf, self.b_gk.buf],
                            &push,
                            (2 * self.n_group) as u32,
                            1,
                            1,
                        )?;
                    }
                    // beta_g
                    {
                        let dtb = self
                            .consts
                            .get(&format!("blk.{il}.dt_bias"))
                            .cloned()
                            .ok_or("dtb")?;
                        let ssa = self
                            .consts
                            .get(&format!("blk.{il}.ssm_a"))
                            .cloned()
                            .ok_or("ssa")?;
                        let push = Self::push_u32s(&[dt_rank as u32, dt_rank as u32]);
                        self.run_pipe(
                            "beta_g",
                            BETA_G_SPV,
                            5,
                            8,
                            &[
                                self.b_gb.buf,
                                self.b_ga.buf,
                                dtb.buf,
                                ssa.buf,
                                self.b_gbg.buf,
                            ],
                            &push,
                            dt_rank.div_ceil(64) as u32,
                            1,
                            1,
                        )?;
                    }
                } // else (구 체인)
                // norm_gated (비트 2)
                if gskip & 2 == 0 {
                    {
                        let sn = self
                            .consts
                            .get(&format!("blk.{il}.ssm_norm"))
                            .cloned()
                            .ok_or("sn")?;
                        // PC: {float eps; int d=d_state; int n_h=dt_rank} — rawhip norm_gated_silu 준거
                        let mut push = self.eps.to_le_bytes().to_vec();
                        push.extend(Self::push_u32s(&[d_state as u32, dt_rank as u32]));
                        self.run_pipe(
                            "norm_gated",
                            NORM_GATED_SPV,
                            4,
                            12,
                            &[self.b_go.buf, self.b_gz.buf, sn.buf, self.b_ggated.buf],
                            &push,
                            dt_rank as u32,
                            1,
                            1,
                        )?;
                    }
                }
                if gskip & 4 == 0 {
                    self.gemv_w(
                        self.b_ggated.buf,
                        self.b_xq_g.buf,
                        &format!("blk.{il}.ssm_out.weight"),
                        self.b_gout.buf,
                        1,
                        d_inner,
                    )?;
                }
                recr_idx += 1;
            } else {
                // 어텐션 (LLM170_VK_ATTN: 1=qkv gemv만, 2=+rope/kv, 3=+flash, 4=+wo)
                let attn_cut = llm170_diag::flag::val("LLM170_VK_ATTN")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(4);
                self.gemv_stage(
                    n,
                    1,
                    &[
                        (
                            format!("blk.{il}.attn_q.weight"),
                            self.b_xq_n.buf,
                            self.b_aq.buf,
                        ),
                        (
                            format!("blk.{il}.attn_k.weight"),
                            self.b_xq_n.buf,
                            self.b_ak.buf,
                        ),
                        (
                            format!("blk.{il}.attn_v.weight"),
                            self.b_xq_n.buf,
                            self.b_av.buf,
                        ),
                    ],
                )?;
                // qk_rope
                {
                    let qn = self
                        .consts
                        .get(&format!("blk.{il}.attn_q_norm"))
                        .cloned()
                        .ok_or("qn")?;
                    let kn = self
                        .consts
                        .get(&format!("blk.{il}.attn_k_norm"))
                        .cloned()
                        .ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    // PC: {float eps; float kqs; int pos; int nh; int nk; int hd; int nr} — gdn-check 준거
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[
                        pos as u32,
                        n_head as u32,
                        n_kv as u32,
                        hd as u32,
                        n_rot as u32,
                    ]));
                    self.run_pipe(
                        "qk_rope2",
                        QK_ROPE2_SPV,
                        5,
                        28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push,
                        (n_head + n_kv) as u32,
                        1,
                        1,
                    )?;
                }
                if attn_cut >= 2 {
                    // kv append
                    {
                        let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
                        // k/v 어펜드는 상호 독립 — k 배리어 생략, v가 종결 (flash는 둘 다 판독)
                        self.run_pipe(
                            "kv_app",
                            KV_APPEND_SPV,
                            2,
                            8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf],
                            &push,
                            (n_kv * hd).div_ceil(64) as u32,
                            1,
                            1,
                        )?;
                        self.run_pipe(
                            "kv_app",
                            KV_APPEND_SPV,
                            2,
                            8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf],
                            &push,
                            (n_kv * hd).div_ceil(64) as u32,
                            1,
                            1,
                        )?;
                    }
                    if attn_cut >= 3 {
                        // flash
                        {
                            let push = Self::push_u32s(&[
                                pos as u32,
                                n_head as u32,
                                n_kv as u32,
                                hd as u32,
                            ]);
                            self.run_pipe(
                                "qsa_flash",
                                QSA_FLASH_SPV,
                                4,
                                16,
                                &[
                                    self.b_aq.buf,
                                    self.kv_k[full_idx][seq].buf,
                                    self.kv_v[full_idx][seq].buf,
                                    self.b_aout.buf,
                                ],
                                &push,
                                1,
                                n_head as u32,
                                1,
                            )?;
                        }
                        if npck && il < 4 {
                            let b = self.b_aout.clone();
                            self.npck_mark("flash", il, &b, 0, 64);
                        }
                        if attn_cut >= 4 {
                            self.gemv_w(
                                self.b_aout.buf,
                                self.b_xq_g.buf,
                                &format!("blk.{il}.attn_output.weight"),
                                self.b_gout.buf,
                                1,
                                n_head * hd,
                            )?;
                        }
                    }
                }
                full_idx += 1;
            }
            // 잔차 + post_norm — addrms 융합 (plans/36 G2: axpy+rms 비트동일)
            self.addrms(
                self.b_xs.buf,
                self.b_gout.buf,
                &format!("blk.{il}.post_norm"),
                self.b_xn.buf,
                n,
                1,
            )?;
            // ── FFN
            self.gemv_stage(
                n,
                1,
                &[
                    (
                        format!("blk.{il}.ffn_gate.weight"),
                        self.b_xq_n.buf,
                        self.b_fgate.buf,
                    ),
                    (
                        format!("blk.{il}.ffn_up.weight"),
                        self.b_xq_n.buf,
                        self.b_fup.buf,
                    ),
                ],
            )?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
            self.gemv_w(
                self.b_fglu.buf,
                self.b_xq_f.buf,
                &format!("blk.{il}.ffn_down.weight"),
                self.b_fdown.buf,
                1,
                self.n_ff,
            )?;
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
            // (VK_FORCE_AQ L0 강제 실험은 plans/115 env 정리로 삭제)
        }
        // ── head: gemv(output) — output_norm은 마지막 addrms에 융합, quant는
        // gemv_w 폴백 시 내부 수행. 트렁크와 동일 배치로 단일 제출·대기 (G3).
        self.gemv_w(
            self.b_xn.buf,
            self.b_xq_n.buf,
            "output.weight",
            self.b_lg.buf,
            1,
            n,
        )?;
        self.ctx.end_batch_wait()?;
        self.ctx.ts_report();
        if self.ktime {
            let mut v: Vec<_> = self.ktimes.iter().collect();
            v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let tot: f64 = v.iter().map(|(_, (e, _))| *e).sum();
            eprintln!(
                "[ktime1] wall {:.1}ms · 커널합 {tot:.1}ms",
                vk_t0.elapsed().as_secs_f32() * 1e3
            );
            for (k, (e, c)) in v.iter().take(12) {
                eprintln!("[ktime1] {:22} {:9.1}ms ({}회)", k, e, c);
            }
            self.ktimes.clear();
        }
        if llm170_diag::flag::on("LLM170_VK_PROF") {
            eprintln!(
                "[vkprof] step: {:.1}ms (pos {})",
                vk_t0.elapsed().as_secs_f32() * 1e3,
                pos
            );
        }
        Ok(())
    }

    /// step_core + 전사 로짓 (기존 계약). raw_step_greedy는 아래 lg_argmax 경로로
    /// 608KB CPU 판독을 우회한다 (GTT 비캐시 판독 ~5ms/토큰 절감, plans/46).
    pub fn step(&mut self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        self.step_core(seq, pos, emb)?;
        let mut logits = vec![0f32; self.n_vocab];
        // SAFETY (107 W8): b_lg 매핑 판독 — step_core는 end_batch_wait로 동기 완료; n_vocab 읽기는 할당 크기와 일치.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.b_lg.ptr as *const f32,
                logits.as_mut_ptr(),
                self.n_vocab,
            )
        };
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
        self.run_pipe_b(
            "argmax2",
            crate::rawvk::vkacc::ARGMAX2_SPV,
            3,
            8,
            &binds,
            &push0,
            n_wg as u32,
            1,
            1,
            true,
        )?;
        self.run_pipe_b(
            "argmax2",
            crate::rawvk::vkacc::ARGMAX2_SPV,
            3,
            8,
            &binds,
            &push1,
            1,
            1,
            1,
            true,
        )?;
        self.ctx.end_batch_wait()?;
        // SAFETY (107 W8): b_am u32 1원소 판독 — 직전 end_batch_wait로 GPU 유휴.
        Ok(unsafe { *(self.b_am.ptr as *const u32) })
    }
}
