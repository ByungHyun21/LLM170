//! decode np — 멀티슬롯 배치 디코드 스텝 (plans/78 R2).

use super::*;

impl DecodeState {
    /// np 배치 디코드 — GEMM/요소커널은 t=n_seqs 행 공유, 상태커널(conv/AR/rope/KV/flash)은
    /// 행 슬라이스로 per-seq 실행 (plans/15). 반환: seq별 logits.
    pub fn step_batch_np(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.step_batch_np_ex(seqs, poss, emb, false).map(|(l, _)| l)
    }

    /// np greedy판 — head 후 logits 전사 대신 GPU argmax, 토큰만 회수.
    /// 산술·상태 갱신은 동일(greedy 플래그는 tail 판독만 갈린다).
    pub fn step_batch_np_greedy(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<u32>, String> {
        self.step_batch_np_ex(seqs, poss, emb, true).map(|(_, t)| t)
    }

    fn step_batch_np_ex(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
        greedy: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<u32>), String> {
        if std::env::var_os("LLM170_LAUNCH_BT").is_some() {
            eprintln!("[xf] step_batch_np");
        }
        let t = seqs.len();
        let np_t0 = std::time::Instant::now();
        let n = self.n_embd;
        debug_assert_eq!(emb.len(), t * n);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        // 레이아웃 (DecodeState 필드 준거): conv_ch = d_inner + 2·n_group·d_state,
        // q/k 행 = n_group·d_state, v 행 = dt_rank·d_state(=d_inner).
        let (conv_ch, k_len, v_len) = (self.conv_ch, self.k_len, self.v_len);
        let d_inner = self.d_inner;
        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = self.n_ff / 4 + self.n_ff / 32 + self.n_ff / 16;
        let xq_sg = d_inner / 4 + d_inner / 32 + d_inner / 16;
        self.ctx.h2d(self.xs_t, bytemuck::cast_slice(emb))?;
        // _ms 커널(conv) 행 메타 — 행별 그룹 크기 1 (plans/74 N3).
        if t > 1 {
            let row_seq: Vec<i32> = seqs.iter().map(|&s| s as i32).collect();
            let row_pos: Vec<i32> = poss.iter().map(|&p| p as i32).collect();
            let seg_start: Vec<i32> = (0..t).map(|r| r as i32).collect();
            let seg_end: Vec<i32> = (0..t).map(|r| (r + 1) as i32).collect();
            let row_np: Vec<i32> = poss.iter().map(|&p| (p + 1) as i32).collect();
            self.h2d_i32_ms(&row_seq, &row_pos, &seg_start, &seg_end, &row_np)?;
        }
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        for il in 0..self.n_layer {
            let wn = *self.consts.get(&format!("blk.{il}.attn_norm")).ok_or("attn_norm")?;
            self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            if self.is_recr[il] {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
                // conv — gdn_conv_np 테이블판 1런치 (plans/74 N3; 종전 행당 1런치,
                // npt4 3.9ms). 산술은 gdn_conv_t(t=1) 과 동일(exp_cr + 링 시프트).
                if t > 1 && std::env::var_os("LLM170_NO_NPCONV").is_none() {
                    let row_seq: Vec<i32> = seqs.iter().map(|&s2| s2 as i32).collect();
                    let mut qp = self.gqkv_t as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.ms_conv_ptr(recr_idx, &row_seq)? as *mut std::ffi::c_void;
                    let mut op = self.gconv_t as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = t as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_conv_np", conv_ch.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                } else {
                for s in 0..t {
                    let row = unsafe { self.gqkv_t.add(s * conv_ch * 4) };
                    let mut qp = row as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.st_conv[recr_idx][seqs[s]] as *mut std::ffi::c_void;
                    let mut op = unsafe { self.gconv_t.add(s * conv_ch * 4) } as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = 1i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_conv_t", conv_ch as u32, 1, 1, 32, &mut args)?;
                }
                }
                // split3 공유 (행별 요소)
                {
                    let mut sp = self.gconv_t as *mut std::ffi::c_void;
                    let mut q0 = self.gq_t as *mut std::ffi::c_void;
                    let mut q1 = self.gk_t as *mut std::ffi::c_void;
                    let mut q2 = self.gv_t as *mut std::ffi::c_void;
                    let mut n0 = k_len as i32;
                    let mut n1 = k_len as i32;
                    let mut n2 = v_len as i32;
                    let total = (2 * k_len + v_len) * t;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut q0), Self::p(&mut q1), Self::p(&mut q2), Self::p(&mut n0), Self::p(&mut n1), Self::p(&mut n2)];
                    self.ew_l("split3", total, &mut args)?;
                }
                // l2 공유
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut ep), Self::p(&mut sc), Self::p(&mut d), Self::p(&mut ng)];
                    self.ctx.launch3("l2_rows2_scale_w", (2 * self.n_group) as u32, t as u32, 1, 32, &mut args)?;
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
                    let mut args = vec![Self::p(&mut bp), Self::p(&mut ap), Self::p(&mut dp), Self::p(&mut sp2), Self::p(&mut bgp), Self::p(&mut nh), Self::p(&mut dr)];
                    self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_beta_g_f32" } else { "gdn_beta_g" }, self.dt_rank * t, &mut args)?;
                }
                // AR — 행별 상태 포인터 테이블로 1런치 (plans/74 N3; 종전 행당
                // 1런치×t, npt4 4.9ms). LLM170_NO_NPAR=1 이면 종전 행 슬라이스.
                if t > 1 && std::env::var_os("LLM170_NO_NPAR").is_none() {
                    let row_seq: Vec<i32> = seqs.iter().map(|&s2| s2 as i32).collect();
                    let tbl = self.ms_gdn_ptr(recr_idx, &row_seq)?;
                    let mut tp = tbl as *mut std::ffi::c_void;
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut vp = self.gv_t as *mut std::ffi::c_void;
                    let mut bgp = self.gbg_t as *mut std::ffi::c_void;
                    let mut op = self.go_t as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = t as i32;
                    let mut args = vec![Self::p(&mut tp), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_ar_w_np", self.dt_rank as u32, self.d_state as u32, 1, 32, &mut args)?;
                } else {
                for s in 0..t {
                    let mut sp3 = self.st_gdn[recr_idx][seqs[s]] as *mut std::ffi::c_void;
                    let mut qp = unsafe { self.gq_t.add(s * k_len * 4) } as *mut std::ffi::c_void;
                    let mut kp = unsafe { self.gk_t.add(s * k_len * 4) } as *mut std::ffi::c_void;
                    let mut vp = unsafe { self.gv_t.add(s * v_len * 4) } as *mut std::ffi::c_void;
                    let mut bgp = unsafe { self.gbg_t.add(s * self.dt_rank * 2 * 4) } as *mut std::ffi::c_void;
                    let mut op = unsafe { self.go_t.add(s * v_len * 4) } as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = 1i32;
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, self.d_state as u32, 1, 32, &mut args)?;
                }
                }
                if std::env::var_os("LLM170_NP_DBG6").is_some() && il == 0 {
                    self.ctx.sync()?;
                    if std::env::var_os("LLM170_NP_DBG7").is_some() {
                        let mut hq2 = vec![0f32; conv_ch * t];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq2).as_mut(), self.gqkv_t)?;
                        let mut hc = vec![0f32; conv_ch * t];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hc).as_mut(), self.gconv_t)?;
                        let cs = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
                        eprintln!("[g0q] qkv=[{:.4},{:.4}] conv=[{:.4},{:.4}]",
                            cs(&hq2[..conv_ch]), cs(&hq2[conv_ch..]),
                            cs(&hc[..conv_ch]), cs(&hc[conv_ch..]));
                    }
                    let mut hq = vec![0f32; k_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.gq_t)?;
                    let mut hk2 = vec![0f32; k_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hk2).as_mut(), self.gk_t)?;
                    let mut hv2 = vec![0f32; v_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv2).as_mut(), self.gv_t)?;
                    let mut ho = vec![0f32; v_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.go_t)?;
                    let cs = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
                    eprintln!("[g0] q=[{:.4},{:.4}] k=[{:.4},{:.4}] v=[{:.4},{:.4}] ar_out=[{:.4},{:.4}]",
                        cs(&hq[..k_len.min(hq.len())]), cs(&hq[k_len.min(hq.len())..]),
                        cs(&hk2[..k_len.min(hk2.len())]), cs(&hk2[k_len.min(hk2.len())..]),
                        cs(&hv2[..v_len.min(hv2.len())]), cs(&hv2[v_len.min(hv2.len())..]),
                        cs(&ho[..v_len.min(ho.len())]), cs(&ho[v_len.min(ho.len())..]));
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
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "norm_gated_silu_f32" } else { "norm_gated_silu" }, self.dt_rank as u32, t as u32, 1, 32, &mut args)?;
                }
                self.ctx.quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
                self.mm_b2(self.ggated_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                recr_idx += 1;
            } else {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
                let qn = *self.consts.get(&format!("blk.{il}.attn_q_norm")).ok_or("qn")?;
                let kn = *self.consts.get(&format!("blk.{il}.attn_k_norm")).ok_or("kn")?;
                let cs = *self.consts.get("cs").ok_or("cs")?;
                for s in 0..t {
                    let aq_row = unsafe { self.aq_t.add(s * n_head * 2 * hd * 4) };
                    let ak_row = unsafe { self.ak_t.add(s * n_kv * hd * 4) };
                    // rope per-seq (t=1)
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut kp = ak_row as *mut std::ffi::c_void;
                        let mut qwp = qn as *mut std::ffi::c_void;
                        let mut kwp = kn as *mut std::ffi::c_void;
                        let mut csp = cs as *mut std::ffi::c_void;
                        let mut ep = self.eps;
                        let mut kq = self.kq_scale;
                        let mut pp = poss[s] as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut nr = n_rot as i32;
                        let rows = n_head + n_kv;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                        self.ctx.launch3("qk_norm_rope", rows as u32, 1, 1, 32, &mut args)?;
                    }
                    // KV append per-seq
                    let pos = poss[s] as usize;
                    let av_row = unsafe { self.av_t.add(s * n_kv * hd * 4) };
                    if legacy_f32() {
                        for (src, table) in [(ak_row, &self.kv_k), (av_row, &self.kv_v)] {
                            let mut sp = src as *mut std::ffi::c_void;
                            let mut dp = table[full_idx][seqs[s]] as *mut std::ffi::c_void;
                            let mut na = (n_kv * hd) as i32;
                            let mut p0 = pos as i32;
                            let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                            self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, 1, 1, 64, &mut args)?;
                        }
                    }
                    // f16 미러: 활성 행에서 직접 변환(어텐션은 미러만 읽는다)
                    let si = seqs[s];
                    kv_to_f16(&self.ctx, ak_row, self.kv_k16[full_idx][si], 0, pos * n_kv * hd, n_kv * hd)?;
                    kv_to_f16(&self.ctx, av_row, self.kv_v16[full_idx][si], 0, pos * n_kv * hd, n_kv * hd)?;
                    // flash per-seq (t=1) — plans/74 N3: 종전 평문 qsa_flash
                    // (gx=1, gy=n_head → 24블록)는 npt4 에서 64런치 직렬
                    // 20.4ms/step 였다. t=1 디코드 경로와 동일 gqa2d(v_dot2
                    // f16) + merge — sg 세그먼트 병렬. part 스크래치는 행별
                    // 분할(공유 충돌 회피). LLM170_NO_NPGQA=1 이면 종전 판.
                    let np_gqa = std::env::var_os("LLM170_NO_NPGQA").is_none()
                        && std::env::var_os("LLM170_NO_FLASH").is_none()
                        && std::env::var_os("LLM170_NO_GQA2D").is_none()
                        && hd <= 256
                        && n_head % n_kv == 0;
                    if np_gqa {
                        let sg = ((pos + 1) / 64).clamp(32, 256);
                        let nseg = (pos + 1).div_ceil(sg).max(1);
                        // 행별 nseg 가 pos 로 달라진다 — stride 는 최대 nseg 기준
                        // 단일 버퍼(스크래치 크기 키 균일), 행은 오프셋 분할.
                        let pos_max = poss.iter().map(|&p| p as usize).max().unwrap_or(0);
                        let nseg_max = ((pos_max + 1).div_ceil(((pos_max + 1) / 64).clamp(32, 256))).max(1);
                        let part = self.ctx.scratch(t * n_head * nseg_max * (hd + 2) * 4)?;
                        let prow = unsafe { part.add(s * n_head * nseg_max * (hd + 2) * 4) };
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut k16 = self.kv_k16[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut v16 = self.kv_v16[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut pp2 = prow as *mut std::ffi::c_void;
                        let mut np_ = (pos + 1) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = 1i32;
                        let mut ss = self.ctx_len as i32;
                        let mut p0 = pos as i32;
                        let mut sg_a = sg as i32;
                        let args16: Vec<*mut std::ffi::c_void> = vec![
                            Self::p(&mut qp), Self::p(&mut k16), Self::p(&mut v16), Self::p(&mut mp),
                            Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk),
                            Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0),
                            Self::p(&mut sg_a),
                        ];
                        let mut args16 = args16;
                        self.ctx.launch3("qsa_flash_gqa2d", 1, n_kv as u32, nseg as u32, 256, &mut args16)?;
                        let mut op = unsafe { self.aout_t.add(s * n_head * hd * 4) } as *mut std::ffi::c_void;
                        let mut margs = vec![Self::p(&mut qp), Self::p(&mut pp2), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut sg_a)];
                        self.ctx.launch3("qsa_flash_merge", 1, n_head as u32, 1, 256, &mut margs)?;
                    } else {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut ckp = self.kv_k16[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut cvp = self.kv_v16[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut op = unsafe { self.aout_t.add(s * n_head * hd * 4) } as *mut std::ffi::c_void;
                        let mut np_ = (pos + 1) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = 1i32;
                        let mut ss = self.ctx_len as i32;
                        let mut p0 = pos as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
                    }
                }
                if !self.grp_mmq(&[format!("blk.{il}.attn_output.weight")], t) {
                    self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(self.aout_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                full_idx += 1;
            }
            self.axpy(self.xs_t, self.gout_t, n * t)?;
            // FFN 공유
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            if !self.grp_mmq(
                &[format!("blk.{il}.ffn_gate.weight"), format!("blk.{il}.ffn_up.weight")],
                t,
            ) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "silu_mul_f32" } else { "silu_mul" }, self.n_ff * t, &mut args)?;
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(self.fglu_t, self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
            self.axpy(self.xs_t, self.fdown_t, n * t)?;
            if std::env::var_os("LLM170_NP_DBG3").is_some() && il % 8 == 0 {
                self.ctx.sync()?;
                let mut hv = vec![0f32; n * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.xs_t)?;
                let r0 = &hv[..n];
                let r1 = &hv[n.min(hv.len())..];
                eprintln!(
                    "[npl] L{il} r0={:.4} r1={:.4}",
                    r0.iter().map(|&v| v as f64).sum::<f64>(),
                    r1.iter().map(|&v| v as f64).sum::<f64>()
                );
            }
        }
        // head — 전체 t행 logits (j128 강제 타일) → 행별 d2h
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        // 소형 t(2..=4)는 4-토큰 GEMV — 타일 헤드는 128열 고정이라 t=4에서 18%를 먹는다
        // (1.04GB q6_K 헤드를 j128 타일로 읽어 29.7ms/스텝, 실측).
        if (2..=4).contains(&t)
            && th == 14
            && std::env::var_os("LLM170_NO_G4").is_none()
        {
            self.ctx.gemm_g4(
                th,
                self.xq_n_t as *const u8,
                wh as *const u8,
                self.ktab2 as *const u8,
                nih,
                noh,
                xq_sn,
                t,
                self.logits_all,
            )?;
        } else {
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
        }
        let (out, toks) = if greedy {
            // GPU argmax — 4×608KB d2h + CPU 스캔 + 행별 동기 왕복 제거.
            // 산술은 CPU greedy와 동일 의미(동률 최저 인덱스, argmax64).
            (Vec::new(), self.argmax_rows(self.logits_all, t, noh)?)
        } else {
            let mut out = Vec::with_capacity(t);
            let mut row = vec![0f32; noh];
            for s in 0..t {
                let src = unsafe { self.logits_all.add(s * noh * 4) } as *const u8;
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut row).as_mut(), src)?;
                out.push(row.clone());
            }
            (out, Vec::new())
        };
        if std::env::var_os("LLM170_NP_TIME").is_some() {
            eprintln!("[npstep] t={t} greedy={greedy} {:.1}ms", np_t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok((out, toks))
    }

}
