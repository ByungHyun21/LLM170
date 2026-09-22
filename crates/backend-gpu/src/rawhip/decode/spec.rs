//! decode MTP/spec — 드래프트·검증 배치 (plans/78 R2).

use super::*;
use crate::rawhip::env_on;

impl DecodeState {
    /// 정규화된 h(공유 head norm 적용됨) → GPU head GEMV + argmax — MTP draft 토큰.
    pub fn mtp_head_argmax(&self, h_normed: &[f32]) -> Result<u32, String> {
        let n = self.n_embd;
        self.ctx.h2d(self.xn, bytemuck::cast_slice(h_normed))?;
        self.ctx.quant_q8(self.xn, self.xq_n, n)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.mm_into(self.xq_n, wh, th, nih, noh, self.logits)?;
        self.argmax_token()
    }

    /// 배치 검증: 토큰 [pos0..pos0+t) 처리 + 전 토큰 head argmax 반환 (MTP spec).
    /// logits_all 버퍼 [t][vocab] — 행별 argmax는 per-row 발사.
    pub fn verify_batch(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
    ) -> Result<(), String> {
        let t = emb.len() / self.n_embd;
        if t > 32 {
            return Err(format!("verify_batch t={t} > 64 (logits_all 상한)"));
        }
        let n = self.n_embd;
        let t_b0 = std::time::Instant::now();
        self.step_batch(seq, pos0, emb)?;
        if env_on("LLM170_SPEC_TIMING") {
            eprintln!("[vb] trunk t={t}: {:.1}ms", t_b0.elapsed().as_secs_f64() * 1e3);
        }
        if env_on("LLM170_SPEC_DBG") { eprintln!("[vb] step_batch ok"); }
        // head: 전 행 rms → quant → output 타일 → 행별 argmax
        let t_r0 = std::time::Instant::now();
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.mmq_y_bump();  // 부록81: xn_t 재기 → quant_y 캐시 무효화
        let xq_sn = crate::rawhip::q4acc::xq_words(n);
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        if env_on("LLM170_SPEC_TIMING") {
            eprintln!("[vb] head prep: {:.1}ms", t_r0.elapsed().as_secs_f64() * 1e3);
        }
        if env_on("LLM170_SPEC_DBG") { eprintln!("[vb] head tile t={t} ty={th} no={noh}"); }
        if env_on("LLM170_SPEC_TIMING") {
            let t_s0 = std::time::Instant::now();
            self.ctx.sync()?;
            eprintln!("[vb] trunk drain: {:.1}ms", t_s0.elapsed().as_secs_f64() * 1e3);
        } else {
            self.ctx.sync()?;
        }
        // plans/73: t≤8 은 mm_b 라우팅(g4 = 무게 1회 독서) — 직접 tile 호출은
        let t_h0 = std::time::Instant::now();
        if t <= 8 && matches!(th, 8 | 12 | 13 | 14 | 23) {
            self.mm_b(
                self.xq_n_t,
                xq_sn,
                wh,
                th,
                nih,
                noh,
                self.logits_all,
                t,
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
        if env_on("LLM170_SPEC_TIMING") {
            self.ctx.sync()?;
            eprintln!("[vb] head mm t={t}: {:.1}ms", t_h0.elapsed().as_secs_f64() * 1e3);
        }
        self.ctx.sync()?;
        if env_on("LLM170_SPEC_DBG") { eprintln!("[vb] head ok"); }
        self.ctx.sync()?;
        let _t_a0 = std::time::Instant::now();
        argmaxes.clear();
        // GPU argmax — t×vocab 플로트 d2h + CPU 스캔 제거 (2026-09-15, plans/74 N1).
        // LLM170_MS_LOGITS 진단은 전사 경로를 유지한다.
        if env_on("LLM170_MS_LOGITS") {
            let mut all_buf = vec![0f32; t * noh];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut all_buf).as_mut(), self.logits_all)?;
            for ti in 0..t {
                let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
                for (i, &v) in all_buf[ti * noh..(ti + 1) * noh].iter().enumerate() {
                    if v > bv { bv = v; best = i; }
                }
                argmaxes.push(best as u32);
            }
        } else {
            argmaxes.extend(self.argmax_rows(self.logits_all, t, noh)?);
        }
        if env_on("LLM170_SPEC_TIMING") {
            eprintln!("[vb] argmax={:.1}ms", _t_a0.elapsed().as_secs_f64() * 1e3);
        }

        Ok(())
    }

    /// GDN+conv 상태 GPU 스냅샷 (d2d).
    /// MTP (blk.64) 1스텝 GPU 실행 — CPU mtp_step과 동일 산술 순서.
    /// tok_emb: 토큰 임베딩 행 [n], h: 트런크 hidden [n], 반환: (argmax, h_next)
    pub fn mtp_step_g(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h_gpu: *mut u8,
        pos: usize,
        with_head: bool,
    ) -> Result<Option<u32>, String> {
        if !self.mtp_on { return Err("mtp_step_gpu: MTP 미로드".into()); }
        let n = self.n_embd;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv, self.hd);
        let n_ao = n_head * hd; // wo 입력 길이
        assert_eq!(tok_emb.len(), n);
        // 입력 업로드 (h는 GPU 버퍼 직접)
        self.ctx.h2d(self.mtp_e, bytemuck::cast_slice(tok_emb))?;
        let t0s = std::time::Instant::now();
        // enorm → cat[0..n], hnorm → cat[n..2n]
        let en = *self.consts.get("blk.64.nextn.enorm").ok_or("enorm")?;
        let hn = *self.consts.get("blk.64.nextn.hnorm").ok_or("hnorm")?;
        self.rms(self.mtp_e, en, self.mtp_cat, n)?;
        let cat_h = unsafe { self.mtp_cat.add(n * 4) };
        self.rms(h_gpu, hn, cat_h, n)?;
        // eh_proj [2n → n]
        if env_on("LLM170_MTP_STAGE") {
            self.ctx.sync()?;
            let mut v = vec![0f32; 2 * n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cat)?;
            let (a, b) = v.split_at(n);
            eprintln!("[g] cat e0={:.6} e1={:.6} esum={:.4} | h0={:.6} h1={:.6} hsum={:.4}", a[0], a[1], a.iter().map(|&x| x as f64).sum::<f64>(), b[0], b[1], b.iter().map(|&x| x as f64).sum::<f64>());
        }
        self.quant(self.mtp_cat, self.mtp_xq2, 2 * n)?;
        if env_on("LLM170_MTP_DBG") { self.ctx.sync()?; eprintln!("[mtp] quant2 ok"); }
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        if env_on("LLM170_MTP_DBG") {
            eprintln!("[mtp] eh_proj ty={te} ni={nie} no={noe} w={we:p} xq2={:p} cur={:p} cat={:p}", self.mtp_xq2, self.mtp_cur, self.mtp_cat);
        }
        // RCA 대상: gemv_q8_out 경로가 ni=10240에서만 700 — 직접 launch는 동일 파라미터로
        // 성공(gy 스위프 검증). 동일 직접 경로로 실행 (산술은 gemm_q6k로 동일).
        self.mm_direct(self.mtp_xq2, we, te, nie, noe, self.mtp_cur)?;
        // 진단 덤프 (MTP 헤드 1단계 수치 미러 대조): tok_emb/h/cat/eh
        if let Some(pref) = std::env::var_os("LLM170_MTP_DUMP") {
            let pref = pref.to_string_lossy().to_string();
            self.ctx.sync()?;
            let mut hv = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), h_gpu)?;
            let mut cv = vec![0f32; 2 * n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut cv).as_mut(), self.mtp_cat)?;
            let mut ev = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut ev).as_mut(), self.mtp_cur)?;
            let mut out = Vec::with_capacity((5 * n + 2 * n) * 4);
            out.extend_from_slice(bytemuck::cast_slice(tok_emb));
            out.extend_from_slice(bytemuck::cast_slice(&hv));
            out.extend_from_slice(bytemuck::cast_slice(&cv));
            out.extend_from_slice(bytemuck::cast_slice(&ev));
            std::fs::write(format!("{pref}.f32"), &out).map_err(|e| e.to_string())?;
            // y(q8) 원시 워드 — quant 레이아웃 검증용 (2n: int8 워드 + 스케일 + qsum)
            // 업로드된 가중치 버퍼 앞부분(GPU) vs 파일 비교용
            let mut wv = vec![0u32; 64];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut wv).as_mut(), we)?;
            std::fs::write(format!("{pref}.wfirst.u32"), bytemuck::cast_slice(&wv)).map_err(|e| e.to_string())?;
            // 중간/끝 구간도 대조 (부분 업로드 탐지): row 2500 / row 5119 시작
            let off_mid = 2500usize * 40 * 210;
            let off_end = 5119usize * 40 * 210;
            let mut wm = vec![0u32; 64];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut wm).as_mut(), unsafe { we.add(off_mid) })?;
            std::fs::write(format!("{pref}.wmid.u32"), bytemuck::cast_slice(&wm)).map_err(|e| e.to_string())?;
            let mut we2 = vec![0u32; 64];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut we2).as_mut(), unsafe { we.add(off_end) })?;
            std::fs::write(format!("{pref}.wend.u32"), bytemuck::cast_slice(&we2)).map_err(|e| e.to_string())?;
            let xq_words = crate::rawhip::q4acc::xq_words(2 * n);
            let mut xv = vec![0u32; xq_words];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut xv).as_mut(), self.mtp_xq2)?;
            std::fs::write(format!("{pref}.xq.u32"), bytemuck::cast_slice(&xv)).map_err(|e| e.to_string())?;
        }
        if env_on("LLM170_MTP_STAGE") {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cur)?;
            eprintln!("[g] eh sum={:.5} x0={:.5} x1={:.5}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0], v[1]);
        }
        // (구 gemv_q8_out 경로 mm_into 호출 제거 — RCA: ni=10240에서 오값이
        //  mm_direct 결과를 덮어써 MTP 초안 품질이 무너졌다. 2026-09-12)
        // attn_norm → q/k/v
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms(self.mtp_cur, an, self.mtp_e, n)?;
        self.quant(self.mtp_e, self.mtp_xq, n)?;
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        self.mm_into(self.mtp_xq, wq, tq, niq, noq, self.aq)?;
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_into(self.mtp_xq, wk, tk, nik, nok, self.ak)?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_into(self.mtp_xq, wv, tv, niv, nov, self.av)?;
        // q/k norm+rope
        let qn = *self.consts.get("blk.64.attn_q_norm").ok_or("qn")?;
        let kn = *self.consts.get("blk.64.attn_k_norm").ok_or("kn")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        {
            let mut qp = self.aq as *mut std::ffi::c_void;
            let mut kp = self.ak as *mut std::ffi::c_void;
            let mut qwp = qn as *mut std::ffi::c_void;
            let mut kwp = kn as *mut std::ffi::c_void;
            let mut csp = cs as *mut std::ffi::c_void;
            let mut ep = self.eps;
            let mut kq = self.kq_scale;
            let mut pp = pos as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut hh = hd as i32;
            let mut nr = self.n_rot as i32;
            let rows = n_head + n_kv;
            let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut hh), Self::p(&mut nr)];
            self.ctx.launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
        }
        // MTP KV append
        self.copy(self.ak, self.mtp_kv_k[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        self.copy(self.av, self.mtp_kv_v[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        kv_to_f16(&self.ctx, self.ak, self.mtp_kv_k16[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        kv_to_f16(&self.ctx, self.av, self.mtp_kv_v16[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        // flash attention (np = pos+1)
        {
            let n_past = pos + 1;
            let mask = self.consts.get("mask").copied().ok_or("mask")?;
            let mut qp = self.aq as *mut std::ffi::c_void;
            let mut ckp = self.mtp_kv_k16[seq] as *mut std::ffi::c_void;
            let mut cvp = self.mtp_kv_v16[seq] as *mut std::ffi::c_void;
            let mut mp = mask as *mut std::ffi::c_void;
            let mut op = self.mtp_ao as *mut std::ffi::c_void;
            let mut np_ = n_past as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut hh = hd as i32;
            let mut tl = 1i32;
            let mut ss = self.ctx_len as i32;
            let mut p0 = pos as i32;
            let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut hh), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
            self.ctx.launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
        }
        // wo + 잔차 (입력 길이 = n_head*hd)
        self.quant(self.mtp_ao, self.mtp_xq, n_ao)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_direct(self.mtp_xq, wo, two, nio, noo, self.gout)?;
        if env_on("LLM170_MTP_STAGE") {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.gout)?;
            eprintln!("[g] wo sum={:.5} x0={:.5}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0]);
        }
        self.axpy(self.mtp_cur, self.gout, n)?;
        // FFN
        let pn = *self.consts.get("blk.64.post_attention_norm").ok_or("post_norm")?;
        self.rms(self.mtp_cur, pn, self.mtp_e, n)?;
        self.quant(self.mtp_e, self.mtp_xq, n)?;
        let (wg, tg, nig, nog) = self.w("blk.64.ffn_gate.weight")?;
        self.mm_into(self.mtp_xq, wg, tg, nig, nog, self.fgate)?;
        let (wu, tu, niu, nou) = self.w("blk.64.ffn_up.weight")?;
        self.mm_into(self.mtp_xq, wu, tu, niu, nou, self.fup)?;
        {
            let mut gp = self.fgate as *mut std::ffi::c_void;
            let mut up = self.fup as *mut std::ffi::c_void;
            let mut op = self.fglu as *mut std::ffi::c_void;
            let mut na = self.n_ff as i32;
            let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
            self.ew_l("silu_mul", self.n_ff, &mut args)?;
        }
        self.quant(self.fglu, self.xq_f, self.n_ff)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_into(self.xq_f, wd, td, nid, nod, self.fdown)?;
        self.axpy(self.mtp_cur, self.fdown, n)?;
        if env_on("LLM170_MTP_STAGE") {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cur)?;
            eprintln!("[g] ff sum={:.5} x0={:.5}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0]);
        }
        if !with_head {
            if env_on("LLM170_SPEC_TIMING") {
                eprintln!("[mt] step(nohead)={:.2}ms", t0s.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(None);
        }
        // shared head norm → output head → argmax
        let shn = *self.consts.get("blk.64.nextn.shared_head_norm").ok_or("shn")?;
        self.rms(self.mtp_cur, shn, self.mtp_e, n)?;
        let _t0h = std::time::Instant::now();
        let am = self.head_argmax_gpu(self.mtp_e)?;
        if env_on("LLM170_SPEC_TIMING") {
            eprintln!("[mt] head={:.2}ms", t0s.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Some(am))
    }

    /// MTP 프리필 배치 — blk.64를 t행 한 번에 처리 (t=1 스텝 × 토큰수 대체).
    /// tok_embs: [t][n] 토큰 임베딩, h_shift: [t][n] = [h_prev, h_all[0..t-1]].
    /// 반환: 마지막 행의 드래프트 토큰 (헤드는 마지막 행만).
    pub fn mtp_prefill_batch(
        &self,
        seq: usize,
        tok_embs: &[f32],
        carry_h: &[f32],
        t: usize,
        pos0: usize,
        with_head: bool,
    ) -> Result<u32, String> {
        if env_on("LLM170_LAUNCH_BT") {
            eprintln!("[xf] mtp_prefill_batch");
        }
        if !self.mtp_on {
            return Err("mtp_prefill_batch: MTP 미로드".into());
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        if t == 0 || t > self.t_max_mtp {
            return Err(format!("mtp_prefill_batch: t={t} 범위 밖"));
        }
        let xq2_w = crate::rawhip::q4acc::xq_words(2 * n);
        let xq_n = crate::rawhip::q4acc::xq_words(n);
        let xq_sf = crate::rawhip::q4acc::xq_words(self.n_ff);
        let n_ao = n_head * hd; // attn_output 입력 길이
        let xq_sg = crate::rawhip::q4acc::xq_words(n_ao);
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        let t_mtp = std::time::Instant::now();
        let mtp_time = env_on("LLM170_MTP_TIMING");
        let mark = |label: &str, last: &mut std::time::Instant| {
            if mtp_time {
                self.ctx.sync().ok();
                eprintln!("[mtpb] {label}: {:.2}ms", last.elapsed().as_secs_f64() * 1e3);
                *last = std::time::Instant::now();
            }
        };
        let mut cp = std::time::Instant::now();
        // KV-only 프리필 제어: 프롬프트 행의 attention/wo/FFN 출력은 쓰이지 않는다
        // (헤드는 마지막 행, 체인은 디코드 h 사용). LLM170_MTP_FULL=1이면 전행.
        let full = env_on("LLM170_MTP_FULL");
        let qstride = n_head * 2 * hd;
        let ostride = n_head * hd;
        let nrow_attn = if full { t } else { 1 };
        let qoff = if full { 0 } else { (t - 1) * qstride };
        let ooff = if full { 0 } else { (t - 1) * ostride };
        // ① enorm(tok) ‖ hnorm(h_{p-1}) → cat [t][2n]  (mtp_b_cur/mtp_b_e는 임시)
        if self.mtp_prefetched.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.ctx.join2()?;   // 사이드 h2d 완료 대기 (메인 프리필과 중첩됨)
        } else {
            self.ctx.h2d(self.mtp_b_e, bytemuck::cast_slice(tok_embs))?;
        }
        // h_shift는 device에서 조립 — src=본체 hidden(xs_t), carry=이전 청크 마지막 행.
        // (호스트 왕복 2×t·n·4B 제거)
        if carry_h.len() != n {
            return Err(format!("mtp_prefill_batch: carry_h len {} != {n}", carry_h.len()));
        }
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(carry_h))?;
        {
            let mut sp = self.xs_t as *mut std::ffi::c_void;
            let mut cp2 = self.mtp_h as *mut std::ffi::c_void;
            let mut dp = self.mtp_b_hs as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut ta = t as i32;
            let mut args = vec![Self::p(&mut sp), Self::p(&mut cp2), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut ta)];
            let gx = ((n * t) as u32).div_ceil(256);
            self.ctx.launch3("row_shift_gather", gx, 1, 1, 256, &mut args)?;
        }
        let en = *self.consts.get("blk.64.nextn.enorm").ok_or("enorm")?;
        let hn = *self.consts.get("blk.64.nextn.hnorm").ok_or("hnorm")?;
        self.rms_rows(self.mtp_b_e, en, self.mtp_b_cur, n, t)?;
        self.rms_rows(self.mtp_b_hs, hn, self.mtp_b_e, n, t)?;
        {
            let mut ep = self.mtp_b_cur as *mut std::ffi::c_void;
            let mut hp = self.mtp_b_e as *mut std::ffi::c_void;
            let mut op = self.mtp_b_cat as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut ta = t as i32;
            let gx = (n.div_ceil(256)) as u32;
            let mut args = vec![
                Self::p(&mut ep), Self::p(&mut hp), Self::p(&mut op),
                Self::p(&mut na), Self::p(&mut ta),
            ];
            self.ctx.launch3("cat2_rows", gx, t as u32, 1, 256, &mut args)?;
        }
        mark("norms+cat", &mut cp);
        // ② eh_proj [2n → n]
        self.ctx.quant_q8_b(self.mtp_b_cat, self.mtp_b_xq2, 2 * n, xq2_w, t)?;
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        self.mm_b2(
            self.mtp_b_cat, self.mtp_b_xq2, xq2_w, we, te, nie, noe,
            self.mtp_b_cur, t,
        )?;
        mark("eh_proj", &mut cp);
        // ③ attn_norm → q/k/v
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms_rows(self.mtp_b_cur, an, self.mtp_b_e, n, t)?;
        self.ctx.quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, t)?;
        // q 투영은 attention(=종료 청크)에서만 필요 — KV는 k/v만 적립한다.
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        if with_head {
            self.mm_b2(self.mtp_b_e, self.mtp_b_xqn, xq_n, wq, tq, niq, noq, self.aq_t, t)?;
        }
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_b2(self.mtp_b_e, self.mtp_b_xqn, xq_n, wk, tk, nik, nok, self.ak_t, t)?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_b2(self.mtp_b_e, self.mtp_b_xqn, xq_n, wv, tv, niv, nov, self.av_t, t)?;
        mark("qkv", &mut cp);
        // ④ q/k norm+rope (배치, pos+y) + KV 적립 + flash (배치)
        let qn = *self.consts.get("blk.64.attn_q_norm").ok_or("qn")?;
        let kn = *self.consts.get("blk.64.attn_k_norm").ok_or("kn")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
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
            let mut args = vec![
                Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp),
                Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp),
                Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr),
            ];
            self.ctx.launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)?;
        }
        for (src, dst, dst16) in [
            (self.ak_t, self.mtp_kv_k[seq], self.mtp_kv_k16[seq]),
            (self.av_t, self.mtp_kv_v[seq], self.mtp_kv_v16[seq]),
        ] {
            let mut sp = src as *mut std::ffi::c_void;
            let mut dp = dst as *mut std::ffi::c_void;
            let mut na = (n_kv * hd) as i32;
            let mut p0 = pos0 as i32;
            let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
            self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
            let dstoff = pos0 * n_kv * hd;
            kv_to_f16(&self.ctx, src, dst16, 0, dstoff, t * n_kv * hd)?;
        }
        {
            // KV-only: 원소 i의 MTP층 출력은 (a) 헤드에서 마지막 행만, (b) 체인은
            // 디코드 스텝의 h를 쓰므로 프롬프트 행들의 attention/wo/FFN은 불필요.
            // 인과 구조상 마지막 행의 출력은 앞 행들의 *KV*만 필요하다 (이미 적립).
            // LLM170_MTP_FULL=1이면 종전 전행 경로.
            let mut qp = unsafe { self.aq_t.add(qoff * 4) } as *mut std::ffi::c_void;
            let mut ckp = self.mtp_kv_k16[seq] as *mut std::ffi::c_void;
            let mut cvp = self.mtp_kv_v16[seq] as *mut std::ffi::c_void;
            let mut mp = mask as *mut std::ffi::c_void;
            let mut op = unsafe { self.aout_t.add(ooff * 4) } as *mut std::ffi::c_void;
            let mut np_ = (pos0 + t) as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut tl = nrow_attn as i32;
            let mut ss = self.ctx_len as i32;
            let mut p0 = (pos0 + t - nrow_attn) as i32;
            if np_ > std::env::var("LLM170_QSA_TH").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(128) {
                let sg = std::env::var("LLM170_QSA_SEG").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize).max(64);
                let nseg = (pos0 + t).div_ceil(sg);
                let part = self.ctx.scratch(nrow_attn * n_head * nseg * (hd + 2) * 4)?;
                let mut pp2 = part as *mut std::ffi::c_void;
                let mut sg_a = sg as i32;
                let mut args = vec![
                    Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp),
                    Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk),
                    Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0),
                    Self::p(&mut sg_a),
                ];
                let wk = nrow_attn > 8;
                let (kn2, gx) = if wk { ("qsa_flash_wk", (nrow_attn.div_ceil(32)) as u32) } else { ("qsa_flash_split4q4", (nrow_attn.div_ceil(4)) as u32) };
                self.ctx.launch3(kn2, gx, n_head as u32, nseg as u32, 256, &mut args)?;
                let mut margs = vec![
                    Self::p(&mut qp), Self::p(&mut pp2), Self::p(&mut op), Self::p(&mut np_),
                    Self::p(&mut nh), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut sg_a),
                ];
                self.ctx.launch3("qsa_flash_merge", nrow_attn as u32, n_head as u32, 1, 256, &mut margs)?;
            } else {
                let mut args = vec![
                    Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp),
                    Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk),
                    Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0),
                ];
                self.ctx.launch3("qsa_flash", nrow_attn as u32, n_head as u32, 1, 256, &mut args)?;
            }
        }
        mark("attn+kv", &mut cp);
        // 중간 청크(with_head=false)는 KV 적립만 — 마지막 행의 attention/FFN/헤드도
        // 아무도 읽지 않는다 (초안은 프롬프트 종료 청크에서만 필요).
        if !with_head {
            if mtp_time {
                eprintln!("[mtpb] TOTAL {:.2}ms (t={t}, kv-only)", t_mtp.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(0);
        }
        // ⑤⑥ KV-only: 마지막 행만 (앞 행들의 wo/FFN 출력은 아무도 쓰지 않는다)
        let nrow_ffn = if full { t } else { 1 };
        let coff = if full { 0 } else { (t - 1) * n };
        let cur_p = unsafe { self.mtp_b_cur.add(coff * 4) };
        let aout_p = unsafe { self.aout_t.add(ooff * 4) };
        let gout_p = unsafe { self.gout_t.add(coff * 4) };
        // ⑤ attn_output + 잔차
        self.ctx.quant_q8_b(aout_p, self.mtp_b_xqn, n_head * hd, xq_sg, nrow_ffn)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_b2(
            aout_p, self.mtp_b_xqn, xq_sg, wo, two, nio, noo,
            gout_p, nrow_ffn,
        )?;
        self.axpy(cur_p, gout_p, n * nrow_ffn)?;
        // ⑥ FFN + 잔차
        let pn = *self.consts.get("blk.64.post_attention_norm").ok_or("post_norm")?;
        self.rms_rows(cur_p, pn, self.mtp_b_e, n, nrow_ffn)?;
        self.ctx.quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, nrow_ffn)?;
        let (wg, tg, nig, nog) = self.w("blk.64.ffn_gate.weight")?;
        self.mm_b2(self.mtp_b_e, self.mtp_b_xqn, xq_n, wg, tg, nig, nog, self.fgate_t, nrow_ffn)?;
        let (wu, tu, niu, nou) = self.w("blk.64.ffn_up.weight")?;
        self.mm_b2(self.mtp_b_e, self.mtp_b_xqn, xq_n, wu, tu, niu, nou, self.fup_t, nrow_ffn)?;
        {
            let mut gp = self.fgate_t as *mut std::ffi::c_void;
            let mut up = self.fup_t as *mut std::ffi::c_void;
            let mut op = self.fglu_t as *mut std::ffi::c_void;
            let mut na = (self.n_ff * nrow_ffn) as i32;
            let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
            self.ew_l(
                "silu_mul_f32",
                self.n_ff * nrow_ffn,
                &mut args,
            )?;
        }
        self.ctx.quant_q8_b(self.fglu_t, self.mtp_b_xq2, self.n_ff, xq_sf, nrow_ffn)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_b2(
            self.fglu_t, self.mtp_b_xq2, xq_sf, wd, td, nid, nod,
            self.fdown_t, nrow_ffn,
        )?;
        self.axpy(cur_p, self.fdown_t, n * nrow_ffn)?;
        mark("ffn", &mut cp);
        // ⑦ 마지막 행만 헤드 — 공유 head norm → output GEMV → argmax
        let shn = *self.consts.get("blk.64.nextn.shared_head_norm").ok_or("shn")?;
        let last = unsafe { self.mtp_b_cur.add((t - 1) * n * 4) };
        self.rms(last, shn, self.mtp_e, n)?;
        let am = self.head_argmax_gpu(self.mtp_e)?;
        if mtp_time {
            eprintln!("[mtpb] TOTAL {:.2}ms (t={})", t_mtp.elapsed().as_secs_f64() * 1e3, t);
        }
        Ok(am)
    }

    /// MTP 1스텝 (호스트 h, head, h_next 회수) — 프리필/디코드 훅용.
    pub fn mtp_step_gpu(        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(h))?;
        let am = self
            .mtp_step_g(seq, tok_emb, self.mtp_h, pos, true)?
            .ok_or_else(|| "mtp head".to_string())?;
        let mut h_next = vec![0f32; self.n_embd];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut h_next).as_mut(), self.mtp_cur)?;
        Ok((am, h_next))
    }

    /// MTP 체인 스텝 — h를 내부 mtp_cur(직전 h_next)에서 직접 읽음.
    pub fn mtp_step_chain(
        &self,
        seq: usize,
        tok_emb: &[f32],
        pos: usize,
    ) -> Result<u32, String> {
        self.mtp_step_g(seq, tok_emb, self.mtp_cur, pos, true)?
            .ok_or("mtp head".to_string())
    }

    /// MTP 상태 진행 스텝 (호스트 trunk h, head 없음) — spec 수용 후 KV 동기.
    pub fn mtp_step_adv(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(), String> {
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(h))?;
        self.mtp_step_g(seq, tok_emb, self.mtp_h, pos, false)?;
        Ok(())
    }

    /// 정규화 입력 x → output GEMV → GPU argmax
    fn head_argmax_gpu(&self, x: *mut u8) -> Result<u32, String> {
        let n = self.n_embd;
        self.quant(x, self.mtp_xq, n)?;
        let (wo, to, nio, noo) = self.w("output.weight")?;
        self.mm_into(self.mtp_xq, wo, to, nio, noo, self.logits)?;
        if env_on("LLM170_MTP_STAGE") {
            self.ctx.sync()?;
            let mut v = vec![0f32; 8];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.logits)?;
            let mut hn8 = vec![0f32; 8];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut hn8).as_mut(), x)?;
            eprintln!("[g] head L0..7={:?} hnorm0..3={:?}", v, &hn8[0..4]);
        }
        let mut xp = self.logits as *mut std::ffi::c_void;
        let mut np_ = noo as i32;
        let mut op = self.ctx.scratch(16)?;
        let mut args = vec![Self::p(&mut xp), Self::p(&mut np_), Self::p(&mut op)];
        self.ctx.launch("argmax64", 1, 1, 64, &mut args)?;
        let mut b8 = [0u8; 8]; // int2: x=최댓값, y=인덱스
        self.ctx.d2h(&mut b8, op)?;
        Ok(u32::from_le_bytes([b8[4], b8[5], b8[6], b8[7]]))
    }

}

    /// np×spec 병합 verify (plans/18) — seq-major 행 그룹. group_starts[i] = seq_i 그룹의
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
            eprintln!("[xf] verify_batch_ms t={} seqs={:?} poss={:?} gs={:?}", emb.len() / self.n_embd, seqs, poss, group_starts);
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
            let (g0, g1) = (group_starts[gi], if gi + 1 < group_starts.len() { group_starts[gi + 1] } else { t });
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
                if il == 0 {
                    self.trace_rows("ms_gqkv", self.gqkv_t, conv_ch, t)?;
                }
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
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
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt), Self::p(&mut rs), Self::p(&mut sg)];
                    self.ctx.launch3("gdn_conv_t2_ms_f32", conv_ch.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                    let mut qp2 = self.gqkv_t as *mut std::ffi::c_void;
                    let mut sp2 = self.ms_conv_ptr(recr_idx, &row_seq)? as *mut std::ffi::c_void;
                    let mut ch2 = conv_ch as i32;
                    let mut kk2 = self.conv_k as i32;
                    let mut tt2 = t as i32;
                    let mut rs2 = self.ms_rowseq as *mut std::ffi::c_void;
                    let mut se = self.ms_segend as *mut std::ffi::c_void;
                    let mut args2 = vec![Self::p(&mut qp2), Self::p(&mut sp2), Self::p(&mut ch2), Self::p(&mut kk2), Self::p(&mut tt2), Self::p(&mut rs2), Self::p(&mut se)];
                    self.ctx.launch3("gdn_conv_state_ms", t as u32, conv_ch.div_ceil(64) as u32, 1, 64, &mut args2)?;
                }
                if il == 0 {
                    self.trace_rows("ms_gconv", self.gconv_t, conv_ch, t)?;
                }
                // split3 공유
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
                    self.ew_l("gdn_beta_g_f32", self.dt_rank * t, &mut args)?;
                }
                                // AR — per-seq 슬라이스 (gdn_ar_w_ms 비대칭 RCA 회피 — 트렁크 검증 커널 재사용)
                for gi in 0..group_starts.len() {
                    let (g0, g1) = (group_starts[gi], if gi + 1 < group_starts.len() { group_starts[gi + 1] } else { t });
                    let gt = g1 - g0;
                    let sq = seqs[gi];
                    let mut sp3 = self.st_gdn[recr_idx][sq] as *mut std::ffi::c_void;
                    let mut qp = unsafe { self.gq_t.add(g0 * k_len * 4) } as *mut std::ffi::c_void;
                    let mut kp = unsafe { self.gk_t.add(g0 * k_len * 4) } as *mut std::ffi::c_void;
                    let mut vp = unsafe { self.gv_t.add(g0 * v_len * 4) } as *mut std::ffi::c_void;
                    let mut bgp = unsafe { self.gbg_t.add(g0 * self.dt_rank * 2 * 4) } as *mut std::ffi::c_void;
                    let mut op = unsafe { self.go_t.add(g0 * v_len * 4) } as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = gt as i32;
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, self.d_state as u32, 1, 32, &mut args)?;
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
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch3("norm_gated_silu_f32", self.dt_rank as u32, t as u32, 1, 32, &mut args)?;
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
                                // per-seq 슬라이스 (np 검증 경로 — _ms 의심 바이팩)
                for gi in 0..group_starts.len() {
                    let (g0, g1) = (group_starts[gi], if gi + 1 < group_starts.len() { group_starts[gi + 1] } else { t });
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
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                        self.ctx.launch3("qk_norm_rope", rows as u32, gt as u32, 1, 32, &mut args)?;
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
                            let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                            self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, gt as u32, 1, 64, &mut args)?;
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
                        let mut op = unsafe { self.aout_t.add(g0 * n_head * hd * 4) } as *mut std::ffi::c_void;
                        let mut np_ = (pos0 + gt) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = gt as i32;
                        let mut ss = self.ctx_len as i32;
                        let mut p0 = pos0 as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", gt as u32, n_head as u32, 1, 256, &mut args)?;
                    }
                }
self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(self.aout_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                if full_idx == 1 {
                    self.trace_rows("ms_aout", self.aout_t, n_head * hd, t)?;
                }
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
                self.ew_l("silu_mul_f32", self.n_ff * t, &mut args)?;
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(self.fglu_t, self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
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
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut all_buf).as_mut(), self.logits_all)?;
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
                eprintln!("[mslg] row{ti}: {}", top.iter()
                    .map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>().join(" "));
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
    pub(super) fn h2d_i32_ms(
        &self,
        row_seq: &[i32],
        row_pos: &[i32],
        seg_start: &[i32],
        seg_end: &[i32],
        row_np: &[i32],
    ) -> Result<(), String> {
        self.ctx.h2d(self.ms_rowseq, bytemuck::cast_slice(row_seq))?;
        self.ctx.h2d(self.ms_rowpos, bytemuck::cast_slice(row_pos))?;
        self.ctx.h2d(self.ms_segstart, bytemuck::cast_slice(seg_start))?;
        self.ctx.h2d(self.ms_segend, bytemuck::cast_slice(seg_end))?;
        self.ctx.h2d(self.ms_rownp, bytemuck::cast_slice(row_np))?;
        Ok(())
    }

    /// 레이어별 per-seq 상태 포인터 테이블 업로드 (t 엔트리 — 행 → 자기 seq 상태).
    pub(super) fn ms_conv_ptr(&self, il: usize, row_seq: &[i32]) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq.iter().map(|&sq| self.st_conv[il][sq as usize]).collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(self.ms_ptrbuf, bytemuck::cast_slice(&raw))?;
        Ok(self.ms_ptrbuf)
    }

    pub(super) fn ms_gdn_ptr(&self, il: usize, row_seq: &[i32]) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq.iter().map(|&sq| self.st_gdn[il][sq as usize]).collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(self.ms_ptrbuf, bytemuck::cast_slice(&raw))?;
        Ok(self.ms_ptrbuf)
    }

}
