//! vk decoder 배치(프리필) 스텝 — step.rs에서 분리 (plans/110 P12d).
//! step_core와 트렁크 구조가 유사하나 행 인자(1 vs t)·conv_state·AR 판
//! 선택(ar4/ar8/gq flash)이 곳곳에서 달라 완전 일치 구간이 없음 —
//! 비트동일 보장을 위해 별도 유지 (부분 일치 조각만 존재).

use super::*;

impl DecoderState {
    /// t행 배치 스텝 (plans/20) — 가중 1회 판독 분할 상각. 행별 산술은
    /// step()과 비트 동일(gemv3 행별 lane 축산·AR 내부 순차·conv 이력 판독).
    /// all_logits=true: 전 행 head 로짓 [t][n_vocab] (verify용 — b_lg_t).
    /// 아니면 마지막 행만 (b_lg). emb는 [t][n_embd].
    pub fn step_batch(
        &mut self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        all_logits: bool,
    ) -> Result<Vec<f32>, String> {
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
        let pfck = llm170_diag::dump::opts().key("pfck");
        let pf_up0 = std::time::Instant::now();
        // SAFETY (107 W8): b_xs 매핑 기입 — t*n 원소, b_xs는 t행 용량으로 할당; begin_batch 전 유휴.
        unsafe {
            std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, t * n);
        }
        let pf_up = pf_up0.elapsed().as_secs_f64() * 1e3;
        let pf_gpu0 = std::time::Instant::now();
        self.ctx.begin_batch()?;
        let tw_rec = std::time::Instant::now();
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        for il in 0..self.n_layer {
            // ── attn_norm — 0층만 (이후 fdown addrms 융합). xq는 gemv_stage 지연 양자화.
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, t)?;
            }
            if self.is_recr[il] {
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
                // conv — gy=t (이력은 qkv에서 판독, t>1은 링을 conv_state가 갱신)
                {
                    let cw = self
                        .consts
                        .get(&format!("blk.{il}.conv_w"))
                        .cloned()
                        .ok_or("conv_w")?;
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32]);
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
                        t as u32,
                        1,
                    )?;
                    if t > 1 {
                        let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32]);
                        self.run_pipe(
                            "gdn_conv_state",
                            GDN_CONV_STATE_SPV,
                            2,
                            12,
                            &[self.b_gqkv.buf, self.st_conv[recr_idx][seq].buf],
                            &push,
                            conv_ch.div_ceil(64) as u32,
                            1,
                            1,
                        )?;
                    }
                }
                // plans/46: 프리필 융합 AR8 (split3+l2+beta_g 인라인) — 기본.
                {
                    // plans/46: 프리필 융합 AR8 — 기본(plans/115 env 정리).
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
                    self.run_pipe(
                        "gdn_ar8f",
                        GDN_AR8F_SPV,
                        7,
                        32,
                        &[
                            self.st_gdn[recr_idx][seq].buf,
                            self.b_gconv.buf,
                            self.b_gb.buf,
                            self.b_ga.buf,
                            dtb.buf,
                            ssa.buf,
                            self.b_go.buf,
                        ],
                        &push,
                        dt_rank as u32,
                        d_state as u32 / 8,
                        1,
                    )?;
                }
                // (구 체인 폴백 삭제 — plans/115 env 정리: 융합 AR8 상시)
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
                // qk_rope — grid (nh+nk, t), pos = pos0+행
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
                        pos0 as u32,
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
                        t as u32,
                        1,
                    )?;
                }
                // kv append — grid (n/64, t). k/v 상호 독립 — k 배리어 생략, v가 종결
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos0 as u32]);
                    self.run_pipe(
                        "kv_app",
                        KV_APPEND_SPV,
                        2,
                        8,
                        &[self.b_ak.buf, self.kv_k[full_idx][seq].buf],
                        &push,
                        (n_kv * hd).div_ceil(64) as u32,
                        t as u32,
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
                        t as u32,
                        1,
                    )?;
                }
                // flash — 프리필(t≥2, GQA ≤6:1)은 다중쿼리 판(plans/83 D):
                // K/V 타일을 24쿼리가 공유해 장문 프리필(pp4096) 어텐션 트래픽·
                // 지연을 1/24로 줄인다. 폴백(구 판)은 LLM170_VK_NOGQ=1.
                if t >= 2 && n_head / n_kv.max(1) <= 6 {
                    // plans/92 P3: 레지스터 상주판(qsa_flash_reg) — hip wk16 구조
                    // 이식(LDS·배리어 0, 점유 8WG/CU급). 종전 gq는 LDS 61KB/WG로
                    // 점유 1WG/CU — 장문 프리필 어텐션이 npmax 선형 지연의 주벚.
                    // hd≠256·킬스위치(LLM170_VK_NOREG=1)는 gq로.
                    if hd == 256
                        && std::env::var("LLM170_VK_NOREG")
                            .map(|v| v != "1")
                            .unwrap_or(true)
                    {
                        let push = Self::push_u32s(&[
                            pos0 as u32,
                            n_head as u32,
                            n_kv as u32,
                            hd as u32,
                            t as u32,
                        ]);
                        self.run_pipe(
                            "qsa_flash_reg",
                            QSA_FLASH_REG_SPV,
                            4,
                            20,
                            &[
                                self.b_aq.buf,
                                self.kv_k[full_idx][seq].buf,
                                self.kv_v[full_idx][seq].buf,
                                self.b_aout.buf,
                            ],
                            &push,
                            (t as u32).div_ceil(16),
                            n_head as u32,
                            1,
                        )?;
                    } else {
                        let push = Self::push_u32s(&[
                            pos0 as u32,
                            n_head as u32,
                            n_kv as u32,
                            hd as u32,
                            t as u32,
                        ]);
                        self.run_pipe(
                            "qsa_flash_gq",
                            QSA_FLASH_GQ_SPV,
                            4,
                            20,
                            &[
                                self.b_aq.buf,
                                self.kv_k[full_idx][seq].buf,
                                self.kv_v[full_idx][seq].buf,
                                self.b_aout.buf,
                            ],
                            &push,
                            (t as u32).div_ceil(4),
                            n_kv as u32,
                            1,
                        )?;
                    }
                } else {
                    let push =
                        Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32]);
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
                        t as u32,
                        n_head as u32,
                        1,
                    )?;
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
            // 잔차 + post_norm — addrms 융합 (t행)
            self.addrms(
                self.b_xs.buf,
                self.b_gout.buf,
                &format!("blk.{il}.post_norm"),
                self.b_xn.buf,
                n,
                t,
            )?;
            // FFN — xq는 gemv_stage 지연 양자화
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
            self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, t)?;
            self.gemv_w(
                self.b_fglu.buf,
                self.b_xq_f.buf,
                &format!("blk.{il}.ffn_down.weight"),
                self.b_fdown.buf,
                t,
                self.n_ff,
            )?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합
            let is_last = il + 1 >= self.n_layer;
            let nkey = if is_last {
                "output_norm".to_string()
            } else {
                format!("blk.{}.attn_norm", il + 1)
            };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, t)?;
            if llm170_diag::dump::opts().key("vkd_lsum") {
                // 107 W1: vk 레이스 국소화 — 층별 b_xn 첫 64합(il % MOD).
                // 판독 직전 배치를 닫았다 재시작(진단 전용 모드).
                let m = std::env::var("LLM170_VKD_LSUM_MOD")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8);
                if il % m == 0 {
                    let mut v = vec![0f32; 64.min(n)];
                    self.ctx.end_batch_wait()?;
                    // SAFETY (107 W8): b_xn 매핑 판독 — 직전 end_batch_wait로 GPU 유휴(디버그 [lsum] 덤프).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.b_xn.ptr as *const f32,
                            v.as_mut_ptr(),
                            v.len(),
                        )
                    };
                    self.ctx.begin_batch()?;
                    let s: f64 = v.iter().map(|&x| x as f64).sum();
                    eprintln!("[lsum] il={il} t={t} sum={s:.6}");
                }
            }
        }
        // ── head (all_logits) — output_norm은 마지막 addrms에 융합. 트렁크와
        // 동일 배치로 단일 제출·대기 (G3). quant는 gemv_w 폴백 시 내부 수행.
        if all_logits {
            self.gemv_w(
                self.b_xn.buf,
                self.b_xq_n.buf,
                "output.weight",
                self.b_lg_t.buf,
                t,
                n,
            )?;
        }
        if llm170_diag::flag::on("LLM170_DBG_REC") {
            let d = self.dbg_drain_ms;
            self.dbg_drain_ms = 0.0;
            eprintln!(
                "#  rec t={t} span={:.1}ms drain={:.1}ms",
                tw_rec.elapsed().as_secs_f64() * 1e3,
                d
            );
        }
        let pf_rec = pf_gpu0.elapsed().as_secs_f64() * 1e3;
        let pf_w0 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        let pf_wait = pf_w0.elapsed().as_secs_f64() * 1e3;
        self.ctx.ts_report();
        let pf_head0 = std::time::Instant::now();
        if all_logits {
            // 107 W1 계약 변경: 전 행 로짓은 b_lg_t에 상주한 채 반환하지
            // 않는다(전사 폐지) — 소비자(verify_rows)는 fn_argmax_rows로
            // GPU 행별 argmax를 수행한다. 반환은 빈 Vec.
            return Ok(Vec::new());
        }
        if self.ktime {
            let mut v: Vec<_> = self.ktimes.iter().collect();
            v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let tot: f64 = v.iter().map(|(_, (e, _))| *e).sum();
            eprintln!("[ktime] t={t} 총 {tot:.0}ms");
            for (k, (e, c)) in v.iter().take(14) {
                eprintln!("[ktime] {:22} {:9.1}ms ({}회)", k, e, c);
            }
        }
        // 마지막 행 head — b_xn 마지막 행이 이미 output_norm 융합 결과
        // SAFETY (107 W8): b_xn 마지막 행 — .add((t-1)*n*4)은 t*n 할당 이내; 제출 전 상태.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.b_xn.ptr.add((t - 1) * n * 4) as *const f32,
                self.m_e.ptr as *mut f32,
                n,
            );
        }
        self.ctx.begin_batch()?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "output.weight",
            self.b_lg.buf,
            1,
            n,
        )?;
        self.ctx.end_batch_wait()?;
        let pf_head = pf_head0.elapsed().as_secs_f64() * 1e3;
        let pf_rd0 = std::time::Instant::now();
        let mut logits = vec![0f32; self.n_vocab];
        // SAFETY (107 W8): b_lg 매핑 판독 — 직전 end_batch_wait로 GPU 유휴, n_vocab 원소.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.b_lg.ptr as *const f32,
                logits.as_mut_ptr(),
                self.n_vocab,
            )
        };
        if pfck {
            eprintln!(
                "[pfck] step_batch t={t} up={pf_up:.1}ms rec={pf_rec:.1}ms wait={pf_wait:.1}ms head={pf_head:.1}ms rd={:.1}ms",
                pf_rd0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(logits)
    }
}
