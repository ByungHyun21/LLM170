//! 멀티-스텝 스펙 검증(verify_batch_ms 계열) (R7 이동).
/// np×spec 병합 verify (plans/18) — seq-major 행 그룹. group_starts[i] = seq_i 그룹의
use super::*;
use crate::rawhip::env_on;

impl DecodeState {
    /// 첫 행 인덱스, group_starts 끝은 t. 행별 argmax 반환 (logits_all 재사용).
    #[allow(clippy::too_many_arguments)]
    pub fn verify_batch_ms(
        &self,
        seqs: &[usize],
        poss: &[usize],
        group_starts: &[usize],
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        if env_on("LLM170_LAUNCH_BT") {
            eprintln!(
                "[xf] verify_batch_ms t={} seqs={:?} poss={:?} gs={:?}",
                emb.len() / self.n_embd,
                seqs,
                poss,
                group_starts
            );
        }
        let t = emb.len() / self.n_embd;
        if t > 64 {
            return Err(format!("verify_batch_ms t={t} > 64"));
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        let d_inner = self.d_inner;
        let xq_sn = crate::rawhip::q4acc::xq_words(n);
        let xq_sf = crate::rawhip::q4acc::xq_words(self.n_ff);
        let xq_sg = crate::rawhip::q4acc::xq_words(d_inner);

        // 행 메타데이터 (호스트 조립 → 소형 업로드)
        let mut row_seq = vec![0i32; t];
        let mut row_pos = vec![0i32; t];
        let mut seg_start = vec![0i32; t];
        let mut seg_end = vec![0i32; t];
        let mut row_np = vec![0i32; t];
        for gi in 0..group_starts.len() {
            let (g0, g1) = (
                group_starts[gi],
                if gi + 1 < group_starts.len() {
                    group_starts[gi + 1]
                } else {
                    t
                },
            );
            for r in g0..g1 {
                row_seq[r] = seqs[gi] as i32;
                row_pos[r] = (poss[gi] + (r - g0)) as i32;
                seg_start[r] = g0 as i32;
                seg_end[r] = g1 as i32;
                row_np[r] = row_pos[r] + 1; // 절대 위치+1 (attention 범위)
            }
        }
        self.h2d_i32_ms(&row_seq, &row_pos, &seg_start, &seg_end, &row_np)?;
        // per-row KV 포인터 (레이어별) — 업로드는 레이어 루프 내 (kv_k[il][seq])
        self.ctx.h2d(self.xs_t, bytemuck::cast_slice(emb))?;
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        for il in 0..self.n_layer {
            let wn = *self
                .consts
                .get(&format!("blk.{il}.attn_norm"))
                .ok_or("attn_norm")?;
            self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            if self.is_recr[il] {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                self.mm_b2(
                    self.xn_t,
                    self.xq_n_t,
                    xq_sn,
                    wp,
                    ty,
                    ni,
                    no,
                    self.gqkv_t,
                    t,
                )?;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                self.mm_b2(
                    self.xn_t,
                    self.xq_n_t,
                    xq_sn,
                    wg2,
                    tg2,
                    nig2,
                    nog2,
                    self.gz_t,
                    t,
                )?;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                self.mm_b2(
                    self.xn_t,
                    self.xq_n_t,
                    xq_sn,
                    wb2,
                    tb2,
                    nib2,
                    nob2,
                    self.gb_t,
                    t,
                )?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                self.mm_b2(
                    self.xn_t,
                    self.xq_n_t,
                    xq_sn,
                    wa2,
                    ta2,
                    nia2,
                    noa2,
                    self.ga_t,
                    t,
                )?;
                if il == 0 {
                    self.trace_rows("ms_gqkv", self.gqkv_t, conv_ch, t)?;
                }
                let cw = *self
                    .consts
                    .get(&format!("blk.{il}.conv_w"))
                    .ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self
                    .consts
                    .get(&format!("blk.{il}.ssm_norm"))
                    .ok_or("ssm_norm")?;
                // conv _ms (행별 seq 상태)
                {
                    let mut qp = self.gqkv_t as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.ms_conv_ptr(recr_idx, &row_seq)? as *mut std::ffi::c_void;
                    let mut op = self.gconv_t as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = t as i32;
                    let mut rs = self.ms_rowseq as *mut std::ffi::c_void;
                    let mut sg = self.ms_segstart as *mut std::ffi::c_void;
                    let mut args = vec![
                        Self::p(&mut qp),
                        Self::p(&mut cp),
                        Self::p(&mut sp),
                        Self::p(&mut op),
                        Self::p(&mut ch),
                        Self::p(&mut kk),
                        Self::p(&mut tt),
                        Self::p(&mut rs),
                        Self::p(&mut sg),
                    ];
                    self.ctx.launch3(
                        "gdn_conv_t2_ms_f32",
                        conv_ch.div_ceil(64) as u32,
                        t as u32,
                        1,
                        64,
                        &mut args,
                    )?;
                    let mut qp2 = self.gqkv_t as *mut std::ffi::c_void;
                    let mut sp2 = self.ms_conv_ptr(recr_idx, &row_seq)? as *mut std::ffi::c_void;
                    let mut ch2 = conv_ch as i32;
                    let mut kk2 = self.conv_k as i32;
                    let mut tt2 = t as i32;
                    let mut rs2 = self.ms_rowseq as *mut std::ffi::c_void;
                    let mut se = self.ms_segend as *mut std::ffi::c_void;
                    let mut args2 = vec![
                        Self::p(&mut qp2),
                        Self::p(&mut sp2),
                        Self::p(&mut ch2),
                        Self::p(&mut kk2),
                        Self::p(&mut tt2),
                        Self::p(&mut rs2),
                        Self::p(&mut se),
                    ];
                    self.ctx.launch3(
                        "gdn_conv_state_ms",
                        t as u32,
                        conv_ch.div_ceil(64) as u32,
                        1,
                        64,
                        &mut args2,
                    )?;
                }
                if il == 0 {
                    self.trace_rows("ms_gconv", self.gconv_t, conv_ch, t)?;
                }
                // split3 공유 — 공용 헬퍼(plans/109 P10)
                self.gdn_split3(
                    self.gconv_t,
                    self.gq_t,
                    self.gk_t,
                    self.gv_t,
                    k_len,
                    v_len,
                    Some(t),
                )?;
                // l2 공유
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![
                        Self::p(&mut qp),
                        Self::p(&mut kp),
                        Self::p(&mut ep),
                        Self::p(&mut sc),
                        Self::p(&mut d),
                        Self::p(&mut ng),
                    ];
                    self.ctx.launch3(
                        "l2_rows2_scale_w",
                        (2 * self.n_group) as u32,
                        t as u32,
                        1,
                        32,
                        &mut args,
                    )?;
                }
                // beta/e^g 공유
                {
                    let mut bp = self.gb_t as *mut std::ffi::c_void;
                    let mut ap = self.ga_t as *mut std::ffi::c_void;
                    let mut dp = dtb as *mut std::ffi::c_void;
                    let mut sp2 = ssa as *mut std::ffi::c_void;
                    let mut bgp = self.gbg_t as *mut std::ffi::c_void;
                    let mut nh = (self.dt_rank * t) as i32;
                    let mut dr = self.dt_rank as i32;
                    let mut args = vec![
                        Self::p(&mut bp),
                        Self::p(&mut ap),
                        Self::p(&mut dp),
                        Self::p(&mut sp2),
                        Self::p(&mut bgp),
                        Self::p(&mut nh),
                        Self::p(&mut dr),
                    ];
                    self.ew_l("gdn_beta_g_f32", self.dt_rank * t, &mut args)?;
                }
                // AR — per-seq 슬라이스 (gdn_ar_w_ms 비대칭 RCA 회피 — 트렁크 검증 커널 재사용)
                for gi in 0..group_starts.len() {
                    let (g0, g1) = (
                        group_starts[gi],
                        if gi + 1 < group_starts.len() {
                            group_starts[gi + 1]
                        } else {
                            t
                        },
                    );
                    let gt = g1 - g0;
                    let sq = seqs[gi];
                    let mut sp3 = self.st_gdn[recr_idx][sq] as *mut std::ffi::c_void;
                    let mut qp = unsafe { self.gq_t.add(g0 * k_len * 4) } as *mut std::ffi::c_void;
                    let mut kp = unsafe { self.gk_t.add(g0 * k_len * 4) } as *mut std::ffi::c_void;
                    let mut vp = unsafe { self.gv_t.add(g0 * v_len * 4) } as *mut std::ffi::c_void;
                    let mut bgp = unsafe { self.gbg_t.add(g0 * self.dt_rank * 2 * 4) }
                        as *mut std::ffi::c_void;
                    let mut op = unsafe { self.go_t.add(g0 * v_len * 4) } as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = gt as i32;
                    let mut args = vec![
                        Self::p(&mut sp3),
                        Self::p(&mut qp),
                        Self::p(&mut kp),
                        Self::p(&mut vp),
                        Self::p(&mut bgp),
                        Self::p(&mut op),
                        Self::p(&mut d),
                        Self::p(&mut ks),
                        Self::p(&mut vs),
                        Self::p(&mut hv),
                        Self::p(&mut hk),
                        Self::p(&mut asc),
                        Self::p(&mut tt),
                    ];
                    self.ctx.launch3(
                        "gdn_ar_w",
                        self.dt_rank as u32,
                        self.d_state as u32,
                        1,
                        32,
                        &mut args,
                    )?;
                }
                if il == 0 {
                    self.trace_rows("ms_go", self.go_t, v_len, t)?;
                }
                // norm_gated 공유
                {
                    let mut op = self.go_t as *mut std::ffi::c_void;
                    let mut zp = self.gz_t as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut outp = self.ggated_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut args = vec![
                        Self::p(&mut op),
                        Self::p(&mut zp),
                        Self::p(&mut wp),
                        Self::p(&mut outp),
                        Self::p(&mut ep),
                        Self::p(&mut d),
                        Self::p(&mut nh),
                    ];
                    self.ctx.launch3(
                        "norm_gated_silu_f32",
                        self.dt_rank as u32,
                        t as u32,
                        1,
                        32,
                        &mut args,
                    )?;
                }
                self.ctx
                    .quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
                self.mm_b2(
                    self.ggated_t,
                    self.xq_g_t,
                    xq_sg,
                    wp,
                    ty,
                    ni,
                    no,
                    self.gout_t,
                    t,
                )?;
                recr_idx += 1;
            } else {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
                let qn = *self
                    .consts
                    .get(&format!("blk.{il}.attn_q_norm"))
                    .ok_or("qn")?;
                let kn = *self
                    .consts
                    .get(&format!("blk.{il}.attn_k_norm"))
                    .ok_or("kn")?;
                // per-seq 슬라이스 (np 검증 경로 — _ms 의심 바이팩)
                for gi in 0..group_starts.len() {
                    let (g0, g1) = (
                        group_starts[gi],
                        if gi + 1 < group_starts.len() {
                            group_starts[gi + 1]
                        } else {
                            t
                        },
                    );
                    let gt = g1 - g0;
                    let sq = seqs[gi];
                    let aq_row = unsafe { self.aq_t.add(g0 * n_head * 2 * hd * 4) };
                    let ak_row = unsafe { self.ak_t.add(g0 * n_kv * hd * 4) };
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut kp = ak_row as *mut std::ffi::c_void;
                        let mut qwp = qn as *mut std::ffi::c_void;
                        let mut kwp = kn as *mut std::ffi::c_void;
                        let mut csp = cs as *mut std::ffi::c_void;
                        let mut ep = self.eps;
                        let mut kq = self.kq_scale;
                        let mut pp = (poss[gi]) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut nr = n_rot as i32;
                        let rows = n_head + n_kv;
                        let mut args = vec![
                            Self::p(&mut qp),
                            Self::p(&mut kp),
                            Self::p(&mut qwp),
                            Self::p(&mut kwp),
                            Self::p(&mut csp),
                            Self::p(&mut ep),
                            Self::p(&mut kq),
                            Self::p(&mut pp),
                            Self::p(&mut nh),
                            Self::p(&mut nk),
                            Self::p(&mut h),
                            Self::p(&mut nr),
                        ];
                        self.ctx.launch3(
                            "qk_norm_rope",
                            rows as u32,
                            gt as u32,
                            1,
                            32,
                            &mut args,
                        )?;
                    }
                    let pos0 = poss[gi];
                    let av_row = unsafe { self.av_t.add(g0 * n_kv * hd * 4) };
                    // f32 원본은 legacy 전용(flash 기본에선 kv_k/v 가 NULL — 가드 필수),
                    // 어텐션은 f16 미러만 읽으므로 미러 변환도 여기서 반드시 수행한다.
                    if legacy_f32() {
                        for (src, table) in [(ak_row, &self.kv_k), (av_row, &self.kv_v)] {
                            let mut sp = src as *mut std::ffi::c_void;
                            let mut dp = table[full_idx][sq] as *mut std::ffi::c_void;
                            let mut na = (n_kv * hd) as i32;
                            let mut p0 = pos0 as i32;
                            let mut args = vec![
                                Self::p(&mut sp),
                                Self::p(&mut dp),
                                Self::p(&mut na),
                                Self::p(&mut p0),
                            ];
                            self.ctx.launch3(
                                "kv_append_t",
                                (n_kv * hd).div_ceil(64) as u32,
                                gt as u32,
                                1,
                                64,
                                &mut args,
                            )?;
                        }
                    }
                    {
                        let doff = pos0 * n_kv * hd;
                        let cnt = gt * n_kv * hd;
                        kv_to_f16(&self.ctx, ak_row, self.kv_k16[full_idx][sq], 0, doff, cnt)?;
                        kv_to_f16(&self.ctx, av_row, self.kv_v16[full_idx][sq], 0, doff, cnt)?;
                    }
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut ckp = self.kv_k16[full_idx][sq] as *mut std::ffi::c_void;
                        let mut cvp = self.kv_v16[full_idx][sq] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut op = unsafe { self.aout_t.add(g0 * n_head * hd * 4) }
                            as *mut std::ffi::c_void;
                        let mut np_ = (pos0 + gt) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = gt as i32;
                        let mut ss = self.ctx_len as i32;
                        let mut p0 = pos0 as i32;
                        let mut args = vec![
                            Self::p(&mut qp),
                            Self::p(&mut ckp),
                            Self::p(&mut cvp),
                            Self::p(&mut mp),
                            Self::p(&mut op),
                            Self::p(&mut np_),
                            Self::p(&mut nh),
                            Self::p(&mut nk),
                            Self::p(&mut h),
                            Self::p(&mut tl),
                            Self::p(&mut ss),
                            Self::p(&mut p0),
                        ];
                        self.ctx.launch3(
                            "qsa_flash",
                            gt as u32,
                            n_head as u32,
                            1,
                            256,
                            &mut args,
                        )?;
                    }
                }
                self.ctx
                    .quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(
                    self.aout_t,
                    self.xq_g_t,
                    xq_sg,
                    wp,
                    ty,
                    ni,
                    no,
                    self.gout_t,
                    t,
                )?;
                if full_idx == 1 {
                    self.trace_rows("ms_aout", self.aout_t, n_head * hd, t)?;
                }
                full_idx += 1;
            }
            self.axpy(self.xs_t, self.gout_t, n * t)?;
            // FFN 공유
            let pw = *self
                .consts
                .get(&format!("blk.{il}.post_norm"))
                .ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            if !self.grp_mmq(
                &[
                    format!("blk.{il}.ffn_gate.weight"),
                    format!("blk.{il}.ffn_up.weight"),
                ],
                t,
            ) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            self.mm_b2(
                self.xn_t,
                self.xq_n_t,
                xq_sn,
                wg,
                tg,
                nig,
                nog,
                self.fgate_t,
                t,
            )?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            self.mm_b2(
                self.xn_t,
                self.xq_n_t,
                xq_sn,
                wu,
                tu,
                niu,
                nou,
                self.fup_t,
                t,
            )?;
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![
                    Self::p(&mut gp),
                    Self::p(&mut up),
                    Self::p(&mut op),
                    Self::p(&mut na),
                ];
                self.ew_l("silu_mul_f32", self.n_ff * t, &mut args)?;
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx
                    .quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(
                self.fglu_t,
                self.xq_f_t,
                xq_sf,
                wd,
                td,
                nid,
                nod,
                self.fdown_t,
                t,
            )?;
            self.axpy(self.xs_t, self.fdown_t, n * t)?;
        }
        // head (j128 강제) + 행별 argmax
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.ctx.gemm_tile_head(
            self.xq_n_t as *const u8,
            wh as *const u8,
            self.ktab2 as *const u8,
            th,
            nih,
            noh,
            xq_sn,
            t,
            self.logits_all,
        )?;
        argmaxes.clear();
        // GPU argmax — t×vocab 전사 회피 (plans/74 N1). MS_LOGITS 진단만 전사.
        if env_on("LLM170_MS_LOGITS") {
            let mut all_buf = vec![0f32; t * noh];
            self.ctx.d2h(
                bytemuck::cast_slice_mut(&mut all_buf).as_mut(),
                self.logits_all,
            )?;
            for ti in 0..t {
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in all_buf[ti * noh..(ti + 1) * noh].iter().enumerate() {
                    if v > bv {
                        bv = v;
                        best = i;
                    }
                }
                argmaxes.push(best as u32);
            }
            for ti in 0..t {
                let mut top: Vec<(u32, f32)> = all_buf[ti * noh..(ti + 1) * noh]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top.truncate(8);
                eprintln!(
                    "[mslg] row{ti}: {}",
                    top.iter()
                        .map(|(i, v)| format!("{i}:{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        } else {
            argmaxes.extend(self.argmax_rows(self.logits_all, t, noh)?);
        }
        h_all.clear();
        h_all.resize(t * self.n_embd, 0.0);
        self.ctx
            .d2h(bytemuck::cast_slice_mut(h_all).as_mut(), self.xs_t)?;
        Ok(())
    }

    /// 행 메타 업로드 (i32 5종) — 고정 ms 버퍼.
    pub(crate) fn h2d_i32_ms(
        &self,
        row_seq: &[i32],
        row_pos: &[i32],
        seg_start: &[i32],
        seg_end: &[i32],
        row_np: &[i32],
    ) -> Result<(), String> {
        self.ctx
            .h2d(self.ms_rowseq, bytemuck::cast_slice(row_seq))?;
        self.ctx
            .h2d(self.ms_rowpos, bytemuck::cast_slice(row_pos))?;
        self.ctx
            .h2d(self.ms_segstart, bytemuck::cast_slice(seg_start))?;
        self.ctx
            .h2d(self.ms_segend, bytemuck::cast_slice(seg_end))?;
        self.ctx.h2d(self.ms_rownp, bytemuck::cast_slice(row_np))?;
        Ok(())
    }

    /// 레이어별 per-seq 상태 포인터 테이블 업로드 (t 엔트리 — 행 → 자기 seq 상태).
    pub(crate) fn ms_conv_ptr(&self, il: usize, row_seq: &[i32]) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq
            .iter()
            .map(|&sq| self.st_conv[il][sq as usize])
            .collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(self.ms_ptrbuf, bytemuck::cast_slice(&raw))?;
        Ok(self.ms_ptrbuf)
    }

    pub(in crate::rawhip::decode) fn ms_gdn_ptr(
        &self,
        il: usize,
        row_seq: &[i32],
    ) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq
            .iter()
            .map(|&sq| self.st_gdn[il][sq as usize])
            .collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(self.ms_ptrbuf, bytemuck::cast_slice(&raw))?;
        Ok(self.ms_ptrbuf)
    }
}
