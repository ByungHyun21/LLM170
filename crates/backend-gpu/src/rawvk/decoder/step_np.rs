//! vk decoder np 배치 디코드 — step.rs에서 분리 (plans/110 P12d).

use super::*;

impl DecoderState {
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
        // 비기본 경로(ARF 폴백)는 순차 루프 — 계약 동일, np 커널은 기본
        // 경로(arf 융합·f32 KV)만 커버.
        let arf_on = llm170_diag::flag::ne0("LLM170_VK_ARF");
        if !arf_on || t == 1 {
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
        let npck = llm170_diag::flag::on("LLM170_VK_NPCK");
        if npck {
            let sl: Vec<u32> =
                unsafe { std::slice::from_raw_parts(self.np_slot.ptr as *const u32, t) }.to_vec();
            let ps: Vec<u32> =
                unsafe { std::slice::from_raw_parts(self.np_pos.ptr as *const u32, t) }.to_vec();
            let tb: Vec<u64> =
                unsafe { std::slice::from_raw_parts(self.np_gdn_tbl.ptr as *const u64, 2.min(ns)) }
                    .to_vec();
            let va: Vec<u64> = (0..2.min(ns))
                .map(|s| self.ctx.buffer_va(self.st_gdn[0][s].buf))
                .collect();
            eprintln!("[npck] t={t} ns={ns} slot={sl:?} pos={ps:?} tbl={tb:#x?} va={va:#x?}");
        }
        // SAFETY (107 W8): b_xs 매핑 기입 — t*n 원소, 할당 용량 이내; 배치 시작 전.
        unsafe {
            std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, t * n);
            // 행별 pos/slot 맵 (u32 테이블 — 셰이더가 행 인덱스로 판독).
            // slot: usize 배열의 저바이트 재해석 금지 — 원소 변환 후 기입
            // (종전 as *const u32 재해석이 [0,0]을 써 np 상태가 전행 슬롯0에
            // 겹쳐 쓰였다 — [npck] st1out 불변으로 국소화, plans/91 P0).
            let slot_v: Vec<u32> = seqs.iter().map(|&s| s as u32).collect();
            std::ptr::copy_nonoverlapping(poss.as_ptr(), self.np_pos.ptr as *mut u32, t);
            std::ptr::copy_nonoverlapping(slot_v.as_ptr(), self.np_slot.ptr as *mut u32, t);
            // HOST_CACHED 비결합 — 스텝마다 호스트 기입 후 GPU 판독 전 flush 의무
            // (context.rs 297 주석 · plans/135 §21-3 14차 — 무-flush가 원장 87/90
            // 확산형 비결정의 근원). 스텝당 16B×2 — flush 비용 무시 가능.
            self.ctx.flush_range(&self.np_pos, t * 4);
            self.ctx.flush_range(&self.np_slot, t * 4);
        }
        self.ctx.begin_batch()?;
        let np_t0 = std::time::Instant::now();
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let layer_cut =
            llm170_diag::flag::val("LLM170_VK_LAYERS").and_then(|v| v.parse::<usize>().ok());
        let n_layer_eff = layer_cut
            .map(|c| c.min(self.n_layer))
            .unwrap_or(self.n_layer);
        for il in 0..n_layer_eff {
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, t)?;
            }
            if self.is_recr[il] {
                // GDN 4 GEMV 공유 (t행 — 무게 1회)
                self.gemv_stage(
                    n,
                    t,
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
                    let cw = self
                        .consts
                        .get(&format!("blk.{il}.conv_w"))
                        .cloned()
                        .ok_or("conv_w")?;
                    let push = Self::push_u32s(&[
                        conv_ch as u32,
                        self.conv_k as u32,
                        t as u32,
                        recr_idx as u32,
                        ns as u32,
                    ]);
                    self.run_pipe(
                        "gdn_conv_np",
                        GDN_CONV_NP_SPV,
                        5,
                        20,
                        &[
                            self.b_gqkv.buf,
                            cw.buf,
                            self.np_conv_tbl.buf,
                            self.b_gconv.buf,
                            self.np_slot.buf,
                        ],
                        &push,
                        conv_ch.div_ceil(64) as u32,
                        t as u32,
                        1,
                    )?;
                }
                if npck && il < 3 {
                    let b = self.b_gconv.clone();
                    self.npck_mark("conv", il, &b, 0, 64);
                }
                // AR — 융합 판(arf)의 행별 상태 1런치 (split3+l2+beta_g+AR 인라인)
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
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[
                        d_state as u32,
                        k_len as u32,
                        v_len as u32,
                        dt_rank as u32,
                        self.n_group as u32,
                    ]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&(t as u32).to_le_bytes());
                    push.extend_from_slice(&self.eps.to_le_bytes());
                    push.extend_from_slice(&(recr_idx as u32).to_le_bytes());
                    push.extend_from_slice(&(ns as u32).to_le_bytes());
                    self.run_pipe(
                        "gdn_arf_np",
                        GDN_ARF_NP_SPV,
                        8,
                        40,
                        &[
                            self.np_gdn_tbl.buf,
                            self.b_gconv.buf,
                            self.b_gb.buf,
                            self.b_ga.buf,
                            dtb.buf,
                            ssa.buf,
                            self.b_go.buf,
                            self.np_slot.buf,
                        ],
                        &push,
                        dt_rank as u32,
                        d_state as u32,
                        1,
                    )?;
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
                    let sn = self
                        .consts
                        .get(&format!("blk.{il}.ssm_norm"))
                        .cloned()
                        .ok_or("sn")?;
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
                        t as u32,
                        1,
                    )?;
                }
                self.gemv_w(
                    self.b_ggated.buf,
                    self.b_xq_g.buf,
                    &format!("blk.{il}.ssm_out.weight"),
                    self.b_gout.buf,
                    t,
                    d_inner,
                )?;
                recr_idx += 1;
            } else {
                // 어텐션 3 GEMV 공유 (t행)
                self.gemv_stage(
                    n,
                    t,
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
                // rope — 행별 pos 테이블 (grid nh+nk, t)
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
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[
                        n_head as u32,
                        n_kv as u32,
                        hd as u32,
                        n_rot as u32,
                    ]));
                    self.run_pipe(
                        "qk_rope2_np",
                        QK_ROPE2_NP_SPV,
                        6,
                        28,
                        &[
                            self.b_aq.buf,
                            self.b_ak.buf,
                            qn.buf,
                            kn.buf,
                            cs.buf,
                            self.np_pos.buf,
                        ],
                        &push,
                        (n_head + n_kv) as u32,
                        t as u32,
                        1,
                    )?;
                }
                // kv append — 행별 KV 캐시 (k 배리어 생략, v가 종결 — flash는 둘 다 판독)
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, full_idx as u32, ns as u32]);
                    let binds_k: Vec<vk::Buffer> = vec![
                        self.b_ak.buf,
                        self.np_kvk_tbl.buf,
                        self.np_pos.buf,
                        self.np_slot.buf,
                    ];
                    self.run_pipe(
                        "kv_app_np",
                        KV_APP_NP_SPV,
                        4,
                        12,
                        &binds_k,
                        &push,
                        (n_kv * hd).div_ceil(64) as u32,
                        t as u32,
                        1,
                    )?;
                    let binds_v: Vec<vk::Buffer> = vec![
                        self.b_av.buf,
                        self.np_kv_v_tbl.buf,
                        self.np_pos.buf,
                        self.np_slot.buf,
                    ];
                    self.run_pipe(
                        "kv_app_np",
                        KV_APP_NP_SPV,
                        4,
                        12,
                        &binds_v,
                        &push,
                        (n_kv * hd).div_ceil(64) as u32,
                        t as u32,
                        1,
                    )?;
                }
                // flash — 행별 KV·pos (grid t, n_head)
                {
                    let push = Self::push_u32s(&[
                        n_head as u32,
                        n_kv as u32,
                        hd as u32,
                        full_idx as u32,
                        ns as u32,
                    ]);
                    self.run_pipe(
                        "qsa_flash_np",
                        QSA_FLASH_NP_SPV,
                        6,
                        20,
                        &[
                            self.b_aq.buf,
                            self.np_kvk_tbl.buf,
                            self.np_kv_v_tbl.buf,
                            self.np_pos.buf,
                            self.np_slot.buf,
                            self.b_aout.buf,
                        ],
                        &push,
                        t as u32,
                        n_head as u32,
                        1,
                    )?;
                }
                if npck && il < 4 {
                    let b = self.b_aout.clone();
                    self.npck_mark("flash", il, &b, 0, 64);
                }
                self.gemv_w(
                    self.b_aout.buf,
                    self.b_xq_g.buf,
                    &format!("blk.{il}.attn_output.weight"),
                    self.b_gout.buf,
                    t,
                    n_head * hd,
                )?;
                full_idx += 1;
            }
            // 잔차 + post_norm (t행)
            self.addrms(
                self.b_xs.buf,
                self.b_gout.buf,
                &format!("blk.{il}.post_norm"),
                self.b_xn.buf,
                n,
                t,
            )?;
            // FFN 공유 (t행)
            self.gemv_stage(
                n,
                t,
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
            self.silu_mul(
                self.b_fgate.buf,
                self.b_fup.buf,
                self.b_fglu.buf,
                self.n_ff * t,
            )?;
            self.gemv_w(
                self.b_fglu.buf,
                self.b_xq_f.buf,
                &format!("blk.{il}.ffn_down.weight"),
                self.b_fdown.buf,
                t,
                self.n_ff,
            )?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합 (t행)
            let is_last = il + 1 >= self.n_layer;
            let nkey = if is_last {
                "output_norm".to_string()
            } else {
                format!("blk.{}.attn_norm", il + 1)
            };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, t)?;
            if npck {
                let b = self.b_xn.clone();
                self.npck_mark("L", il, &b, 0, 64);
            }
        }
        // ── head — b_xn 전 행이 이미 output_norm 융합 결과. t행 GEMV 1회.
        self.gemv_w(
            self.b_xn.buf,
            self.b_xq_n.buf,
            "output.weight",
            self.b_lg_t.buf,
            t,
            n,
        )?;
        if greedy {
            // 행별 GPU argmax — fn_argmax_rows 2단계 (동률 최저 인덱스).
            // 트렁크와 동일 배치로 제출(별도 서브미션 제거).
            let nv = self.n_vocab;
            let n_wg = nv.div_ceil(256 * 8);
            let push0 = Self::push_u32s(&[nv as u32, 0u32, n_wg as u32]);
            let push1 = Self::push_u32s(&[nv as u32, 1u32, n_wg as u32]);
            let binds = [self.b_lg_t.buf, self.b_amsc.buf, self.b_amr.buf];
            self.run_pipe_b(
                "fn_argmax_rows",
                crate::rawvk::vkacc::FN_ARGMAX_ROWS_SPV,
                3,
                12,
                &binds,
                &push0,
                n_wg as u32,
                t as u32,
                1,
                true,
            )?;
            self.run_pipe_b(
                "fn_argmax_rows",
                crate::rawvk::vkacc::FN_ARGMAX_ROWS_SPV,
                3,
                12,
                &binds,
                &push1,
                1,
                t as u32,
                1,
                true,
            )?;
        }
        let np_t1 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        if llm170_diag::dump::opts().key("np_time") {
            eprintln!(
                "[npstep] vk t={t} greedy={greedy} rec={:.1}ms wait={:.1}ms",
                (np_t1 - np_t0).as_secs_f64() * 1e3,
                np_t1.elapsed().as_secs_f64() * 1e3
            );
        }
        if greedy {
            let mut toks = vec![0u32; t];
            // SAFETY (107 W8): b_amr u32 t원소 판독 — 직전 end_batch_wait로 GPU 유휴.
            unsafe {
                std::ptr::copy_nonoverlapping(self.b_amr.ptr as *const u32, toks.as_mut_ptr(), t)
            };
            return Ok((Vec::new(), toks));
        }
        // plans/92 P6.4: 중간 flat 2.4MB + 행별 재복사 폐지 — 행 버퍼로 직복사.
        let mut rows: Vec<Vec<f32>> = (0..t).map(|_| vec![0f32; self.n_vocab]).collect();
        for (i, r) in rows.iter_mut().enumerate() {
            // SAFETY (107 W8): b_lg_t 행 판독 — .add(i*n_vocab*4), i<t이고 버퍼는 t*n_vocab f32; 대기 완료 후.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.b_lg_t.ptr.add(i * self.n_vocab * 4) as *const f32,
                    r.as_mut_ptr(),
                    self.n_vocab,
                )
            };
        }
        Ok((rows, Vec::new()))
    }
    /// plans/91 P0 — [npck] 스테이지 마크: 배치-순차 대조용 행 합 덤프.
    pub(super) fn npck_mark(&mut self, tag: &str, il: usize, b: &VkBuf, row: usize, len: usize) {
        self.ctx.end_batch_wait().ok();
        self.ctx.begin_batch().ok();
        let mut v = vec![0f32; len];
        // SAFETY (107 W8): npck_mark — .add(row*len*4)는 호출부 계약상 버퍼 내 행; end_batch_wait 직후 GPU 유휴.
        unsafe {
            std::ptr::copy_nonoverlapping(
                b.ptr.add(row * len * 4) as *const f32,
                v.as_mut_ptr(),
                len,
            )
        };
        let s: f64 = v.iter().map(|&x| x as f64).sum();
        eprintln!("[npck] {tag} il={il} sum={s:.6} v0={:.6}", v[0]);
    }
}
