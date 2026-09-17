//! decode 스텝 — 단일 시퀀스 prefill 배치 스텝 (plans/78 R2).

use super::*;

impl DecodeState {
    /// 프리필 배치 스텝 — t 토큰 (emb: [t][n_embd], pos0..pos0+t-1), 마지막 logits.
    /// mm/quant/rms/silu/l2/split3/beta_g/norm_gated/qk_norm_rope 배치,
    /// conv/AR/KV/qsa 순차·토큰 의존 — 토큰 루프. 산술은 step()과 토큰당 동일열.
    #[allow(clippy::too_many_lines)]
    pub fn step_batch(&self, seq: usize, pos0: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        if std::env::var_os("LLM170_LAUNCH_BT").is_some() {
            eprintln!("[xf] step_batch");
        }
        let t = emb.len() / self.n_embd;
        debug_assert!(t >= 1 && t <= self.b_t_max);
        let n = self.n_embd;
        let prof = std::env::var_os("LLM170_PP_PROF").is_some();
        let t0w = std::time::Instant::now();
        let mut marks: Vec<(String, hip::hipEvent_t)> = Vec::new();
        let gmark = |lab: &str, marks: &mut Vec<(String, hip::hipEvent_t)>| {
            if prof {
                let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                unsafe { hip::hipEventCreate(&mut ev); hip::hipEventRecord(ev, self.ctx.stream); }
                marks.push((lab.to_string(), ev));
            }
        };
        let (k_len, v_len, conv_ch) = (self.k_len, self.v_len, self.conv_ch);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = self.n_ff / 4 + self.n_ff / 32 + self.n_ff / 16;
        let xq_sg = self.d_inner / 4 + self.d_inner / 32 + self.d_inner / 16;
        self.ctx.h2d(self.xs_t, bytemuck::cast_slice(emb))?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        let mask = *self.consts.get("mask").ok_or("mask")?;
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        for il in 0..self.n_layer {
            let wn = *self.consts.get(&format!("blk.{il}.attn_norm")).ok_or("attn_norm")?;
            self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.mmq_y_bump();  // 부록81: xn_t 재기 → quant_y 캐시 무효화
            // qkv/gate/beta/alpha(또는 q/k/v)가 모두 f32 직소비면 q8 활성은 사장 —
            // MMQ는 y_f32를 직접 읽고 내부에서 mmq 레이아웃으로 재양자화한다.
            let proj_names: Vec<String> = if self.is_recr[il] {
                ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"].iter()
                    .map(|k2| format!("blk.{il}.{k2}.weight")).collect()
            } else {
                ["attn_q", "attn_k", "attn_v"].iter()
                    .map(|k2| format!("blk.{il}.{k2}.weight")).collect()
            };
            if !self.grp_mmq(&proj_names, t) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
gmark("norm", &mut marks);
            if self.is_recr[il] {
                // 2스트림: qkv+beta(주) ‖ gate+alpha(사이드) — 4독립 GEMM
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                let qkv_tile = matches!(ty, 12 | 13 | 14 | 23) && t > 64;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                let gate_tile = matches!(tg2, 12 | 13 | 14 | 23) && t > 64;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                if qkv_tile && gate_tile {
                    // 사이드는 타일형만 (beta/alpha는 q8_0 gemv — 주 스트림)
                    // (사이드 이동 시도: L2 경합으로 -5 t/s 회귀 — 2026-09-05 측정)
                    self.ctx.side_wait_main()?;
                    self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                    self.ctx.join2()?;
                } else {
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                }
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
gmark("gdn_mm", &mut marks);
                // conv+ring 배치 (채널 블록 × t 내부 순차)
                {
                    let mut qp = self.gqkv_t as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.st_conv[recr_idx][seq] as *mut std::ffi::c_void;
                    let mut op = self.gconv_t as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = t as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt)];
                    if t >= self.conv_k - 1 {
                        self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_conv_t2_f32" } else { "gdn_conv_t2" }, conv_ch.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                        let mut qp2 = self.gqkv_t as *mut std::ffi::c_void;
                        let mut sp2 = self.st_conv[recr_idx][seq] as *mut std::ffi::c_void;
                        let mut ch2 = conv_ch as i32;
                        let mut kk2 = self.conv_k as i32;
                        let mut tt2 = t as i32;
                        let mut args2 = vec![Self::p(&mut qp2), Self::p(&mut sp2), Self::p(&mut ch2), Self::p(&mut kk2), Self::p(&mut tt2)];
                        self.ctx.launch3("gdn_conv_state", (self.conv_k - 1) as u32, conv_ch.div_ceil(64) as u32, 1, 64, &mut args2)?;
                    } else {
                        self.ctx.launch3("gdn_conv_t", conv_ch as u32, 1, 1, 32, &mut args)?;
                    }
                }
                if il == 0 {
                    self.trace_rows("tr_gconv", self.gconv_t, conv_ch, t)?;
                }
                // split3 전체 배치 (요소별)
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
                // l2 전체 배치 (gy=t)
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut ep), Self::p(&mut sc), Self::p(&mut d), Self::p(&mut ng)];
                    let l2k = if std::env::var_os("LLM170_EXACT").is_some() { "l2_rows2_scale" } else { "l2_rows2_scale_w" };
                    self.ctx.launch3(l2k, (2 * self.n_group) as u32, t as u32, 1, 32, &mut args)?;
                }
gmark("split+l2", &mut marks);
                // beta/e^g 전체 배치 (요소별)
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
gmark("betag", &mut marks);
                // AR 배치 (pair 블록 × t 내부 순차)
                {
                    let mut sp3 = self.st_gdn[recr_idx][seq] as *mut std::ffi::c_void;
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
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    if std::env::var_os("LLM170_EXACT").is_some() || std::env::var_os("LLM170_AR_T").is_some() {
                        self.ctx.launch3("gdn_ar_t", self.dt_rank as u32, (self.d_state / 64) as u32, 1, 64, &mut args)?;
                    } else if t >= 128 && self.d_state == 128 && std::env::var_os("LLM170_ARCHUNK").is_some() {
                        // 청크 스캔 (부록 72): A(로컬)→B(캐리)→C(보정)
                        const CH: usize = 64;
                        let npair = self.dt_rank;
                        let mut d = self.d_state as i32;
                        let nc = t.div_ceil(CH);
                        let nc_max = self.b_t_max.div_ceil(CH);
                        // ffn 버퍼 별칭 (AR은 층 내 ffn 이전 — 라이프타임 무충돌)
                        let mut sstart = self.fgate_t as *mut u8;
                        let mut pbuf = self.fup_t as *mut u8;
                        let mut pgb = self.fglu_t as *mut u8;
                        let mut lend = (self.fglu_t as *mut u8).wrapping_add(262144);
                        let _ = nc_max;
                        let mut chi = CH as i32;
                        let mut ca: Vec<*mut std::ffi::c_void> = args.clone();
                        ca.extend([Self::p(&mut chi), Self::p(&mut lend), Self::p(&mut pgb), Self::p(&mut pbuf)].iter().copied());
                        self.ctx.launch3("gdn_ar_chunk_a", npair as u32, d as u32, nc as u32, 32, &mut ca)?;
                        let (mut bnp, mut bnc) = (npair as i32, nc as i32);
                        let mut cb: Vec<*mut std::ffi::c_void> = vec![
                            Self::p(&mut sp3), Self::p(&mut lend), Self::p(&mut pgb),
                            Self::p(&mut sstart), Self::p(&mut d), Self::p(&mut tt),
                            Self::p(&mut chi), Self::p(&mut bnp), Self::p(&mut bnc),
                        ];
                        self.ctx.launch3("gdn_ar_chunk_b", npair as u32, 1, 1, d as u32, &mut cb)?;
                        let mut cc: Vec<*mut std::ffi::c_void> = vec![
                            Self::p(&mut op), Self::p(&mut qp), Self::p(&mut pbuf),
                            Self::p(&mut sstart), Self::p(&mut d), Self::p(&mut ks),
                            Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk),
                            Self::p(&mut asc), Self::p(&mut tt), Self::p(&mut chi),
                            Self::p(&mut bnp),
                        ];
                        self.ctx.launch3("gdn_ar_chunk_c2", npair as u32, nc as u32, 1, d as u32, &mut cc)?;
                    } else if std::env::var_os("LLM170_ARSM").is_some() && self.d_state == 128 {
                        // 부록82: smem 스테이징 AR (k/q 128중 재독 제거)
                        self.ctx.launch3("gdn_ar_sm", self.dt_rank as u32, 2, 1, 64, &mut args)?;
                    } else {
                        // 부록88 기본: 축스왑(u블록 인접) — k/q L2 국소성 +1.1% (350-354)
                        if std::env::var_os("LLM170_NO_ARSWAP").is_none() {
                            self.ctx.launch3("gdn_ar_w_swap", self.d_state as u32, self.dt_rank as u32, 1, 32, &mut args)?;
                        } else {
                            // 8워프/블록 (워프당 1열) — 블록 수 1/8
                            let uw = 8u32;
                            self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, (self.d_state as u32).div_ceil(uw), 1, 32 * uw, &mut args)?;
                        }
                    }
                }
                if il == 0 {
                    self.trace_rows("tr_go", self.go_t, v_len, t)?;
                }
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 0 {
                    self.ctx.sync()?;
                    let mut hq = vec![0f32; k_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.gq_t)?;
                    let mut xq_: u64 = 0;
                    for &v in &hq { xq_ ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    let mut ho = vec![0f32; v_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.go_t)?;
                    let mut xo_: u64 = 0;
                    for &v in &ho { xo_ ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    eprintln!("#  G0dbg batch gq xor={xq_:016x} go xor={xo_:016x}");
                }
gmark("trace", &mut marks);
                // norm_gated 전체 배치 (gy=t)
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
gmark("gdn", &mut marks);
gmark("normg", &mut marks);
                // out proj 배치
                if !self.grp_mmq(&[format!("blk.{il}.ssm_out.weight")], t) {
                    self.ctx.quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
gmark("outproj", &mut marks);
                self.mm_b2(self.ggated_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                recr_idx += 1;
} else {
gmark("attn", &mut marks);
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
                let qn = *self.consts.get(&format!("blk.{il}.attn_q_norm")).ok_or("qn")?;
                let kn = *self.consts.get(&format!("blk.{il}.attn_k_norm")).ok_or("kn")?;
                // q/k norm+rope 전체 배치 (gy=t — 커널 pos+y)
                {
                    let mut qp = self.aq_t as *mut std::ffi::c_void;
                    let mut kp = self.ak_t as *mut std::ffi::c_void;
                    let mut qwp = qn as *mut std::ffi::c_void;
                    let mut kwp = kn as *mut std::ffi::c_void;
                    let mut csp = cs as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut kq = self.kq_scale;
                    let mut pp = pos0 as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut nr = n_rot as i32;
                    let rows = n_head + n_kv;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                    self.ctx.launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)?;
                }
                // KV append 배치 (gy=t)
                {
                    if legacy_f32() {
                        let mut sp = self.ak_t as *mut std::ffi::c_void;
                        let mut dp = self.kv_k[full_idx][seq] as *mut std::ffi::c_void;
                        let mut na = (n_kv * hd) as i32;
                        let mut p0 = pos0 as i32;
                        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                        self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                    }
                    let dst = pos0 * n_kv * hd;
                    let n = t * n_kv * hd;
                    kv_to_f16(&self.ctx, self.ak_t, self.kv_k16[full_idx][seq], 0, dst, n)?;
                }
                {
                    if legacy_f32() {
                        let mut sp = self.av_t as *mut std::ffi::c_void;
                        let mut dp = self.kv_v[full_idx][seq] as *mut std::ffi::c_void;
                        let mut na = (n_kv * hd) as i32;
                        let mut p0 = pos0 as i32;
                        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                        self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                    }
                    let dst = pos0 * n_kv * hd;
                    let n = t * n_kv * hd;
                    kv_to_f16(&self.ctx, self.av_t, self.kv_v16[full_idx][seq], 0, dst, n)?;
                }
                // qsa 배치 (단일 런치 — sstride=ctx, n_past=pos0+t 최대;
                // 초과 p는 마스크→-3e38→w=0 기여로 원소 산술열 불변)
                {
                    // flash 시 기존 score/mix2 스킵 (이중실행 방지)
                    let flash_only = std::env::var_os("LLM170_NO_FLASH").is_none() && hd <= 256;
                    if !flash_only {
                    let np_max = (pos0 + t) as i32;
                    let sstr = self.ctx_len as i32;
                    {
                        let mut qp = self.aq_t as *mut std::ffi::c_void;
                        let mut ckp = self.kv_k16[full_idx][seq] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut scp = self.scores_t as *mut std::ffi::c_void;
                        let mut np_ = np_max;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = t as i32;
                        let mut ss = sstr;
                        let mut p0 = pos0 as i32;
                        let gx = np_max.unsigned_abs().div_ceil(64);
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut mp), Self::p(&mut scp), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_score", gx, n_head as u32, t as u32, 64, &mut args)?;
                    }
                    {
                        let mut qp = self.aq_t as *mut std::ffi::c_void;
                        let mut scp = self.scores_t as *mut std::ffi::c_void;
                        let mut cvp = self.kv_v16[full_idx][seq] as *mut std::ffi::c_void;
                        let mut op = self.aout_t as *mut std::ffi::c_void;
                        let mut np_ = np_max;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = t as i32;
                        let mut ss = sstr;
                        let mut p0 = pos0 as i32;
                        let gx = hd.div_ceil(64) as u32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut scp), Self::p(&mut cvp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_mix2", gx, n_head as u32, t as u32, 64, &mut args)?;
                    }
                    }
                }
                // qsa fused flash: score+softmax+mix 단일 패스 (maxrel 1e-6 검증, 스트림★)
                if std::env::var_os("LLM170_NO_FLASH").is_none() && hd <= 256 {
                    let mut qp = self.aq_t as *mut std::ffi::c_void;
                    let mut ckp = self.kv_k16[full_idx][seq] as *mut std::ffi::c_void;
                    let mut cvp = self.kv_v16[full_idx][seq] as *mut std::ffi::c_void;
                    let mut mp = mask as *mut std::ffi::c_void;
                    let mut op = self.aout_t as *mut std::ffi::c_void;
                    let mut np_ = (pos0 + t) as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut tl = t as i32;
                    let mut ss = self.ctx_len as i32;
                    let mut p0 = pos0 as i32;
                    // 분할 flash 기본 ON (2026-09-05: pp512 +5 — 청크 2-4의 np 성장
                    // 구간 병렬화; LLM170_NO_QSA_SPLIT으로 원경로)
                    if std::env::var_os("LLM170_NO_QSA_SPLIT").is_none() && np_ > std::env::var("LLM170_QSA_TH").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(128) {
                        // 세그먼트 기본 1024 (2026-09-12 실측): 128→1024 로 pp3314 331.9→339.5 t/s,
                        // pp512 359.9→362.8. part 중간버퍼 트래픽이 세그먼트 수에 비례해 줄어든다.
                        let sg = std::env::var("LLM170_QSA_SEG").ok().and_then(|v| v.parse().ok()).unwrap_or(1024usize).max(64);
                        let nseg = (pos0 + t).div_ceil(sg);
                        // part: [t][n_head][nseg][hd+2]
                        let part = self.ctx.scratch(t * n_head * nseg * (hd + 2) * 4)?;
                        let mut pp2 = part as *mut std::ffi::c_void;
                        let mut sg_a = sg as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0), Self::p(&mut sg_a)];
                        // q4 다중화 기본: ck/cv 1회 로드로 t 4행 공유 (레지스터 여유 내 최대 배율)
                        // wk는 t>8(프리필) 전용 — 소형 배치(검증 t<=8)는 디코드와 같은
                        // split4q4를 써서 spec/greedy 계약을 구조적으로 만든다.
                        let wk = t > 8 && std::env::var_os("LLM170_NO_WKFLASH").is_none();
                        // hd=256 프리필은 16레인/행 판이 기본 (판정기 17/19 유지).
                        // 3회 평균 pp3314 303.7 vs 종전 299.9 (+1.3%, 구동 잡음 ±1.5%).
                        // LLM170_NO_WK16=1 이면 종전 32레인 판으로 복귀.
                        let wk16 = wk && hd == 256 && std::env::var_os("LLM170_NO_WK16").is_none();
                        if wk16 {
                            // WMMA 타일 판은 **옵트인**(LLM170_WK_WMMA=1)으로 강등
                            // (2026-09-14, plans/69): 이 기기의 ROCm/HIP 빌드에서
                            // wmma_ok() 프로브는 통과하지만 실측이 파탄이다 —
                            // pp512 38.2 t/s vs wk8 364.2 t/s(**9.5×**). 커널이
                            // 에뮬레이션/스필 경로로 떨어지는 것으로 추정(원인은
                            // 미상 — VK 시대 기록 361.3과 같은 수치를 냈던 판이다).
                            // launch3_dyn이 KTRACE 이벤트를 안 남겨 'kv_f16 뒤
                            // 갭 12s'로 위장한 게 이 어텐션 커널 시간이었다.
                            if wk
                                && hd == 256
                                && std::env::var_os("LLM170_WK_WMMA").is_some()
                                && super::probes::wmma_ok()
                            {
                                // WMMA 타일 판(기본): Q_in_reg + Q 버퍼를 K/V 로 재사용. 공유 32768B.
                                // pp512 361.3 / pp3314 329.2 vs 스칼라 360.2 / 324.7 (2026-09-12).
                                // 산술이 f16 누적이라 스칼라와 다른데, 토큰 동일성은 600토큰
                                // 다중 청크에서 확인했고 커널 정확성은 wmma-attn-check 가 보증한다.
                                // LLM170_NO_WK_WMMA=1 이면 wk8 로 복귀.
                                self.ctx.launch3_dyn("qsa_flash_wmma", t.div_ceil(64) as u32, n_head as u32, nseg as u32, 256, 32768, &mut args)?;
                            } else {
                            // hd=256 프리필은 8레인/행 판(셔플 3단)이 기본 — wk16 대비 페어 +2.3%.
                            // 산술(트리 깊이)이 달라 장문 궤적이 갈리지만 커널 정확성은
                            // `llm170 attn-check` 로 보증된다(사용자 결정 2026-09-12).
                            // LLM170_NO_WK8=1 이면 wk16(4단)으로 복귀.
                            // plans/73: v_dot2 판은 **옵트인**(LLM170_WK8D=1). QK 는
                            // 4× 빨라지지만 PV(스레드=dim × 8쿼리)가 스칼라 f32 FMA 로
                            // 병목을 넘어가 실측 역행 — wk8i 253 vs wk8d 234 t/s@pp16k.
                            // 구조 교훈(PV/기록의 8쿼리 전부 규약, 완전마스크 타일의
                            // e=0)은 커널 주석에 남긴다. WMMA급 해법이 다음 과제.
                            // plans/74 N4: raw-builtin WMMA(w32) 판 — ABI 는
                            // wmma2-map2 프로브로 확정, wmma2-attn-check PASS.
                            // f16 Q/P 산술 클래스라 평탄분포 아그맥스를 흔든다(실측:
                            // 2302토큰 프롬프트 스트림 분기, 첫 플립 참조갭 2.07nat
                            // → 근접티 ε=1.5 밖). 표준 검증면(게이트·verify·MTP·VL,
                            // 모두 ctx≤8k)은 바이트 불변을 유지하고 **n_past>8192
                            // 장문 프리필에만** 적용한다(pp16k +11%; 임계 2560 — 검증
                            // 프롬프트 최대 2326토큰은 wk8i 클래스 유지). 폭 넓은 채택은
                            // llama 참조 재수집 후 재판정 과제. LLM170_NO_WMMA2=1
                            // 이면 전 구간 wk8i.
                            if std::env::var_os("LLM170_NO_WMMA2").is_none()
                                && np_ > 2560 {
                                let v2k = std::env::var_os("LLM170_NO_WMMA2V2").is_none();
                                self.ctx.launch3(if v2k { "qsa_flash_wmma2v2" } else { "qsa_flash_wmma2" }, t.div_ceil(16) as u32, n_head as u32, nseg as u32, 64, &mut args)?;
                            } else if std::env::var_os("LLM170_WK8D").is_some() {
                                self.ctx.launch3("qsa_flash_wk8d", t.div_ceil(8) as u32, n_head as u32, nseg as u32, 256, &mut args)?;
                            } else if std::env::var_os("LLM170_NO_WK8I").is_none() {
                                self.ctx.launch3("qsa_flash_wk8i", t.div_ceil(32) as u32, n_head as u32, nseg as u32, 256, &mut args)?;
                            } else if std::env::var_os("LLM170_NO_WK8").is_none() {
                                self.ctx.launch3("qsa_flash_wk8", t.div_ceil(32) as u32, n_head as u32, nseg as u32, 256, &mut args)?;
                            } else {
                                self.ctx.launch3("qsa_flash_wk16", t.div_ceil(16) as u32, n_head as u32, nseg as u32, 256, &mut args)?;
                            }
                            }
                        } else {
                            let (kn, gx) = if wk { ("qsa_flash_wk", t.div_ceil(32) as u32) } else { ("qsa_flash_split4q4", t.div_ceil(4) as u32) };
                            self.ctx.launch3(kn, gx, n_head as u32, nseg as u32, 256, &mut args)?;
                        }
                        let mut margs = vec![Self::p(&mut qp), Self::p(&mut pp2), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut sg_a)];
                        self.ctx.launch3("qsa_flash_merge", t as u32, n_head as u32, 1, 256, &mut margs)?;
                        if let Some(path) = std::env::var_os("LLM170_ATTN_DUMP")
                            && full_idx == 0 {
                                self.ctx.sync().ok();
                                let mut v = vec![0f32; t * n_head * hd];
                                self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.aout_t)?;
                                std::fs::write(&path, bytemuck::cast_slice(&v)).ok();
                                eprintln!("# attn-dump L0 t={t} n_head={n_head} hd={hd}");
                            }
                    } else {
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", t as u32, n_head as u32, 1, 256, &mut args)?;
                    }
                }
                // wo 배치
                if !self.grp_mmq(&[format!("blk.{il}.attn_output.weight")], t) {
                    self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(self.aout_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                full_idx += 1;
            }
self.axpy(self.xs_t, self.gout_t, n * t)?;
            gmark("proj", &mut marks);
            // FFN 배치
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            self.ctx.mmq_y_bump();  // 부록81: xn_t 재기 → quant_y 캐시 무효화
            if !self.grp_mmq(
                &[format!("blk.{il}.ffn_gate.weight"), format!("blk.{il}.ffn_up.weight")],
                t,
            ) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
gmark("ffn_quant", &mut marks);
            // 2스트림: gate(사이드) ‖ up(주) — 독립 GEMM, 출력버퍼 분리
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            let gate_tile = matches!(tg, 12 | 13 | 14 | 23) && t > 64;
            let up_tile = matches!(tu, 12 | 13 | 14 | 23) && t > 64;
            if std::env::var_os("LLM170_PP_PAIRS").is_none() {
                // 기본: 직렬 — 2스트림 페어는 join2(이벤트) 오버헤드가 이득을 넘는다
                // (2026-09-12 A/B: 직렬 +0.75%, LLM170_PP_PAIRS=1로 페어 복원).
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            } else if gate_tile && !up_tile {
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.ctx.side_wait_main()?;
                self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else if !gate_tile && up_tile {
                self.ctx.side_wait_main()?;
                self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else if gate_tile && up_tile {
                // 둘 다 타일 — 하나 사이드
                self.ctx.side_wait_main()?;
                self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else {
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            }
gmark("ffn_gate", &mut marks);
gmark("ffn_up", &mut marks);
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "silu_mul_f32" } else { "silu_mul" }, self.n_ff * t, &mut args)?;
            }
gmark("ffn_silu", &mut marks);
            if std::env::var_os("LLM170_DUMP_XQN").is_some() && il == 0 {
                self.ctx.sync()?;
                let mut bytes = vec![0u8; xq_sn * 4 * t];
                self.ctx.d2h(bytes.as_mut_slice(), self.xq_n_t)?;
                let _ = std::fs::write(std::env::var_os("LLM170_DUMP_XQN").unwrap(), &bytes);
                eprintln!("#  xq_n_t dumped: {} words", xq_sn * t);
            }
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 0 {
                self.ctx.sync()?;
                let mut hf = vec![0f32; self.n_ff * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hf).as_mut(), self.fgate_t)?;
                let bg = hf.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                let mut hl = vec![0f32; self.n_ff * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hl).as_mut(), self.fglu_t)?;
                let bl = hl.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                eprintln!("#  F0 fgate nan/inf {bg} | fglu nan/inf {bl}");
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
gmark("ffn_quant2", &mut marks);
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(self.fglu_t, self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
self.axpy(self.xs_t, self.fdown_t, n * t)?;
            gmark("ffn", &mut marks);
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() {
                self.ctx.sync()?;
                let mut hv = vec![0f32; n * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.xs_t)?;
                let sum0: f64 = hv[..n].iter().map(|&v| v as f64).sum();
                let suml: f64 = hv[n * (t - 1)..].iter().map(|&v| v as f64).sum();
                eprintln!("#  E{il} t={t} xs0={sum0:.6} xs_last={suml:.6}");
            }
        }
        // head — 마지막 토큰만
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        let last = unsafe { self.xs_t.add((t - 1) * n * 4) } as *mut u8;
        {
            let mut xp = last as *mut std::ffi::c_void;
            let mut pp = self.p64 as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut a1 = vec![Self::p(&mut xp), Self::p(&mut pp), Self::p(&mut na)];
            self.ctx.launch("rms_part", 1, 1, 32, &mut a1)?;
            let mut wp = wn as *mut std::ffi::c_void;
            let mut op = self.xn as *mut std::ffi::c_void;
            let mut ep = self.eps;
            let mut wr = 1i32;
            let mut a2 = vec![Self::p(&mut xp), Self::p(&mut wp), Self::p(&mut pp), Self::p(&mut op), Self::p(&mut ep), Self::p(&mut na), Self::p(&mut wr)];
            self.ctx.launch("rms_finish", 1, 1, 256, &mut a2)?;
        }
        self.ctx.quant_q8(self.xn, self.xq_n, n)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.mm_into(self.xq_n, wh, th, nih, noh, self.logits)?;
        gmark("head", &mut marks);
        if prof {
            let wall = t0w.elapsed().as_secs_f64() * 1e3;
            eprintln!("cpu_submit={:.1}ms (CPU returned from launches)", wall);
            if let Some((_, last)) = marks.last() {
                unsafe { hip::hipEventSynchronize(*last); }
            }
            let mut acc: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
            for w in marks.windows(2) {
                let mut d = 0f32;
                unsafe { hip::hipEventElapsedTime(&mut d, w[0].1, w[1].1); }
                *acc.entry(w[1].0.clone()).or_insert(0.0) += d;
            }
            let mut v: Vec<_> = acc.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tot: f32 = v.iter().map(|x| x.1).sum();
            eprintln!("pprof t={t} total_marks={tot:.1}ms");
            for (k, ms) in v { eprintln!("  pprof[{k}] {ms:.1}ms"); }
        }
        Ok(Vec::new()) // logits 상주 — d2h는 호출부 선택
    }

    /// logits 전체 d2h (비-greedy 샘플링용).
    pub fn read_logits(&self) -> Result<Vec<f32>, String> {
        let noh = self.n_vocab;
        let mut out = vec![0f32; noh];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out).as_mut(), self.logits)?;
        Ok(out)
    }

    /// 마지막 스텝의 최종 hidden (output_norm 전) d2h — MTP 훅용.
    pub fn read_hidden(&self, out: &mut Vec<f32>) -> Result<(), String> {
        out.resize(self.n_embd, 0.0);
        self.ctx.d2h(bytemuck::cast_slice_mut(out).as_mut(), self.xs)
    }

}
