//! 스펙 검증·드래프트(배치·GPU·체인·advanced) (plans/129 R7 이동).
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
        self.step_batch(seq, pos0, emb)?;
        // head: 전 행 rms → quant → output 타일 → 행별 argmax
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
        let xq_sn = crate::rawhip::q4acc::xq_words(n);
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        // plans/73: t≤8 은 mm_b 라우팅(g4 = 무게 1회 독서) — 직접 tile 호출은
        if t <= 8 && matches!(th, 8 | 12 | 13 | 14 | 23) {
            self.mm_b(self.xq_n_t, xq_sn, wh, th, nih, noh, self.logits_all, t)?;
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
        self.ctx.sync()?;
        self.ctx.sync()?;
        let _t_a0 = std::time::Instant::now();
        argmaxes.clear();
        // GPU argmax — t×vocab 플로트 d2h + CPU 스캔 제거 (2026-09-15, plans/74 N1).
        // LLM170_MS_LOGITS 진단은 전사 경로를 유지한다.
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
        } else {
            argmaxes.extend(self.argmax_rows(self.logits_all, t, noh)?);
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
        if !self.mtp_on {
            return Err("mtp_step_gpu: MTP 미로드".into());
        }
        // plans/135 §22 MTP k3: 패스 위상 분해 (LLM170_DUMP=spec_time)
        let mtp_tm = llm170_diag::dump::opts().key("spec_time");
        let mut mt = std::time::Instant::now();
        let mut mtw = std::time::Instant::now();
        let mut ph = [0u128; 8];
        let mut phw = [0u128; 8];
        macro_rules! mmark { ($i:expr) => {{
            if mtp_tm {{
                phw[$i] += mtw.elapsed().as_micros() as u128; // wall (호스트 포함)
                self.ctx.sync().ok(); // 커널 비동기 — sync 후 GPU 실측
                ph[$i] += mt.elapsed().as_micros() as u128;
                mt = std::time::Instant::now();
                mtw = std::time::Instant::now();
            }
        }} }}
        let n = self.n_embd;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv, self.hd);
        let n_ao = n_head * hd; // wo 입력 길이
        assert_eq!(tok_emb.len(), n);
        // 입력 업로드 (h는 GPU 버퍼 직접)
        self.ctx.h2d(self.mtp_e, bytemuck::cast_slice(tok_emb))?;
        // enorm → cat[0..n], hnorm → cat[n..2n]
        let en = *self.consts.get("blk.64.nextn.enorm").ok_or("enorm")?;
        let hn = *self.consts.get("blk.64.nextn.hnorm").ok_or("hnorm")?;
        self.rms(self.mtp_e, en, self.mtp_cat, n)?;
        let cat_h = unsafe { self.mtp_cat.add(n * 4) };
        self.rms(h_gpu, hn, cat_h, n)?;
        mmark!(0);
        // eh_proj [2n → n]
        if llm170_diag::dump::opts().key("mtp_stage") {
            self.ctx.sync()?;
            let mut v = vec![0f32; 2 * n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cat)?;
            let (a, b) = v.split_at(n);
            eprintln!(
                "[g] cat e0={:.6} e1={:.6} esum={:.4} | h0={:.6} h1={:.6} hsum={:.4}",
                a[0],
                a[1],
                a.iter().map(|&x| x as f64).sum::<f64>(),
                b[0],
                b[1],
                b.iter().map(|&x| x as f64).sum::<f64>()
            );
        }
        self.quant(self.mtp_cat, self.mtp_xq2, 2 * n)?;
        mmark!(1);
        if env_on("LLM170_MTP_DBG") {
            self.ctx.sync()?;
            eprintln!("[mtp] quant2 ok");
        }
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        if env_on("LLM170_MTP_DBG") {
            eprintln!(
                "[mtp] eh_proj ty={te} ni={nie} no={noe} w={we:p} xq2={:p} cur={:p} cat={:p}",
                self.mtp_xq2, self.mtp_cur, self.mtp_cat
            );
        }
        // RCA 대상: gemv_q8_out 경로가 ni=10240에서만 700 — 직접 launch는 동일 파라미터로
        // 성공(gy 스위프 검증). 동일 직접 경로로 실행 (산술은 gemm_q6k로 동일).
        self.mm_direct(self.mtp_xq2, we, te, nie, noe, self.mtp_cur)?;
        mmark!(2);
        // 진단 덤프 (MTP 헤드 1단계 수치 미러 대조): tok_emb/h/cat/eh
        if let Some(pref) = llm170_diag::dump::opts().key_arg("mtp_dump") {
            let pref = pref.to_string();
            self.ctx.sync()?;
            let mut hv = vec![0f32; n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), h_gpu)?;
            let mut cv = vec![0f32; 2 * n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut cv).as_mut(), self.mtp_cat)?;
            let mut ev = vec![0f32; n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut ev).as_mut(), self.mtp_cur)?;
            let mut out = Vec::with_capacity((5 * n + 2 * n) * 4);
            out.extend_from_slice(bytemuck::cast_slice(tok_emb));
            out.extend_from_slice(bytemuck::cast_slice(&hv));
            out.extend_from_slice(bytemuck::cast_slice(&cv));
            out.extend_from_slice(bytemuck::cast_slice(&ev));
            std::fs::write(format!("{pref}.f32"), &out).map_err(|e| e.to_string())?;
            // y(q8) 원시 워드 — quant 레이아웃 검증용 (2n: int8 워드 + 스케일 + qsum)
            // 업로드된 가중치 버퍼 앞부분(GPU) vs 파일 비교용
            let mut wv = vec![0u32; 64];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut wv).as_mut(), we)?;
            std::fs::write(format!("{pref}.wfirst.u32"), bytemuck::cast_slice(&wv))
                .map_err(|e| e.to_string())?;
            // 중간/끝 구간도 대조 (부분 업로드 탐지): row 2500 / row 5119 시작
            let off_mid = 2500usize * 40 * 210;
            let off_end = 5119usize * 40 * 210;
            let mut wm = vec![0u32; 64];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut wm).as_mut(), unsafe {
                    we.add(off_mid)
                })?;
            std::fs::write(format!("{pref}.wmid.u32"), bytemuck::cast_slice(&wm))
                .map_err(|e| e.to_string())?;
            let mut we2 = vec![0u32; 64];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut we2).as_mut(), unsafe {
                    we.add(off_end)
                })?;
            std::fs::write(format!("{pref}.wend.u32"), bytemuck::cast_slice(&we2))
                .map_err(|e| e.to_string())?;
            let xq_words = crate::rawhip::q4acc::xq_words(2 * n);
            let mut xv = vec![0u32; xq_words];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut xv).as_mut(), self.mtp_xq2)?;
            std::fs::write(format!("{pref}.xq.u32"), bytemuck::cast_slice(&xv))
                .map_err(|e| e.to_string())?;
        }
        if llm170_diag::dump::opts().key("mtp_stage") {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cur)?;
            eprintln!(
                "[g] eh sum={:.5} x0={:.5} x1={:.5}",
                v.iter().map(|&x| x as f64).sum::<f64>(),
                v[0],
                v[1]
            );
        }
        // (구 gemv_q8_out 경로 mm_into 호출 제거 — RCA: ni=10240에서 오값이
        //  mm_direct 결과를 덮어써 MTP 초안 품질이 무너졌다. 2026-09-12)
        // attn_norm → q/k/v
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms(self.mtp_cur, an, self.mtp_e, n)?;
        self.quant(self.mtp_e, self.mtp_xq, n)?;
        mmark!(3);
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        self.mm_into(self.mtp_xq, wq, tq, niq, noq, self.aq)?;
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_into(self.mtp_xq, wk, tk, nik, nok, self.ak)?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_into(self.mtp_xq, wv, tv, niv, nov, self.av)?;
        mmark!(4);
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
                Self::p(&mut hh),
                Self::p(&mut nr),
            ];
            self.ctx
                .launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
        }
        // MTP KV append
        self.copy(self.ak, self.mtp_kv_k[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        self.copy(self.av, self.mtp_kv_v[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        kv_to_f16(
            &self.ctx,
            self.ak,
            self.mtp_kv_k16[seq],
            0,
            pos * n_kv * hd,
            n_kv * hd,
        )?;
        kv_to_f16(
            &self.ctx,
            self.av,
            self.mtp_kv_v16[seq],
            0,
            pos * n_kv * hd,
            n_kv * hd,
        )?;
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
            let mut args = vec![
                Self::p(&mut qp),
                Self::p(&mut ckp),
                Self::p(&mut cvp),
                Self::p(&mut mp),
                Self::p(&mut op),
                Self::p(&mut np_),
                Self::p(&mut nh),
                Self::p(&mut nk),
                Self::p(&mut hh),
                Self::p(&mut tl),
                Self::p(&mut ss),
                Self::p(&mut p0),
            ];
            self.ctx
                .launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
        }
        mmark!(5);
        // wo + 잔차 (입력 길이 = n_head*hd)
        self.quant(self.mtp_ao, self.mtp_xq, n_ao)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_direct(self.mtp_xq, wo, two, nio, noo, self.gout)?;
        if llm170_diag::dump::opts().key("mtp_stage") {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.gout)?;
            eprintln!(
                "[g] wo sum={:.5} x0={:.5}",
                v.iter().map(|&x| x as f64).sum::<f64>(),
                v[0]
            );
        }
        self.axpy(self.mtp_cur, self.gout, n)?;
        // FFN
        let pn = *self
            .consts
            .get("blk.64.post_attention_norm")
            .ok_or("post_norm")?;
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
            let mut args = vec![
                Self::p(&mut gp),
                Self::p(&mut up),
                Self::p(&mut op),
                Self::p(&mut na),
            ];
            self.ew_l("silu_mul", self.n_ff, &mut args)?;
        }
        self.quant(self.fglu, self.xq_f, self.n_ff)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_into(self.xq_f, wd, td, nid, nod, self.fdown)?;
        self.axpy(self.mtp_cur, self.fdown, n)?;
        mmark!(6);
        if llm170_diag::dump::opts().key("mtp_stage") {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cur)?;
            eprintln!(
                "[g] ff sum={:.5} x0={:.5}",
                v.iter().map(|&x| x as f64).sum::<f64>(),
                v[0]
            );
        }
        if !with_head {
            return Ok(None);
        }
        // shared head norm → output head → argmax
        let shn = *self
            .consts
            .get("blk.64.nextn.shared_head_norm")
            .ok_or("shn")?;
        self.rms(self.mtp_cur, shn, self.mtp_e, n)?;
        let _t0h = std::time::Instant::now();
        // 마지막 위상 — fn 종료로 리셋 불요 (데드 할당 경고 회피)
        if mtp_tm {
            self.ctx.sync().ok();
            ph[7] += mt.elapsed().as_micros();
            phw[7] += mtw.elapsed().as_micros();
        }
        if mtp_tm {
            eprintln!(
                "[mtpP] cat={:6.3}/{:6.3} q8={:6.3}/{:6.3} eh={:6.3}/{:6.3} anq={:6.3}/{:6.3} qkv={:6.3}/{:6.3} attn={:6.3}/{:6.3} ffn={:6.3}/{:6.3} head={:6.3}/{:6.3} ms wall/GPU",
                phw[0] as f64 / 1000.0, ph[0] as f64 / 1000.0,
                phw[1] as f64 / 1000.0, ph[1] as f64 / 1000.0,
                phw[2] as f64 / 1000.0, ph[2] as f64 / 1000.0,
                phw[3] as f64 / 1000.0, ph[3] as f64 / 1000.0,
                phw[4] as f64 / 1000.0, ph[4] as f64 / 1000.0,
                phw[5] as f64 / 1000.0, ph[5] as f64 / 1000.0,
                phw[6] as f64 / 1000.0, ph[6] as f64 / 1000.0,
                phw[7] as f64 / 1000.0, ph[7] as f64 / 1000.0
            );
        }
        let am = self.head_argmax_gpu(self.mtp_e)?;
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
                eprintln!(
                    "[mtpb] {label}: {:.2}ms",
                    last.elapsed().as_secs_f64() * 1e3
                );
                *last = std::time::Instant::now();
            }
        };
        let mut cp = std::time::Instant::now();
        // KV-only 프리필 제어: 프롬프트 행의 attention/wo/FFN 출력은 쓰이지 않는다
        // (헤드는 마지막 행, 체인은 디코드 h 사용). 전행 경로는 MTP_FULL 옵트인
        // 부록으로 plans/109 P6에서 삭제(미사용 판정).
        let qstride = n_head * 2 * hd;
        let ostride = n_head * hd;
        let nrow_attn = 1usize;
        let qoff = (t - 1) * qstride;
        let ooff = (t - 1) * ostride;
        // ① enorm(tok) ‖ hnorm(h_{p-1}) → cat [t][2n]  (mtp_b_cur/mtp_b_e는 임시)
        if self
            .mtp_prefetched
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            self.ctx.join2()?; // 사이드 h2d 완료 대기 (메인 프리필과 중첩됨)
        } else {
            self.ctx.h2d(self.mtp_b_e, bytemuck::cast_slice(tok_embs))?;
        }
        // h_shift는 device에서 조립 — src=본체 hidden(xs_t), carry=이전 청크 마지막 행.
        // (호스트 왕복 2×t·n·4B 제거)
        if carry_h.len() != n {
            return Err(format!(
                "mtp_prefill_batch: carry_h len {} != {n}",
                carry_h.len()
            ));
        }
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(carry_h))?;
        {
            let mut sp = self.xs_t as *mut std::ffi::c_void;
            let mut cp2 = self.mtp_h as *mut std::ffi::c_void;
            let mut dp = self.mtp_b_hs as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut ta = t as i32;
            let mut args = vec![
                Self::p(&mut sp),
                Self::p(&mut cp2),
                Self::p(&mut dp),
                Self::p(&mut na),
                Self::p(&mut ta),
            ];
            let gx = ((n * t) as u32).div_ceil(256);
            self.ctx
                .launch3("row_shift_gather", gx, 1, 1, 256, &mut args)?;
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
                Self::p(&mut ep),
                Self::p(&mut hp),
                Self::p(&mut op),
                Self::p(&mut na),
                Self::p(&mut ta),
            ];
            self.ctx
                .launch3("cat2_rows", gx, t as u32, 1, 256, &mut args)?;
        }
        mark("norms+cat", &mut cp);
        // ② eh_proj [2n → n]
        self.ctx
            .quant_q8_b(self.mtp_b_cat, self.mtp_b_xq2, 2 * n, xq2_w, t)?;
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        self.mm_b2(
            self.mtp_b_cat,
            self.mtp_b_xq2,
            xq2_w,
            we,
            te,
            nie,
            noe,
            self.mtp_b_cur,
            t,
        )?;
        mark("eh_proj", &mut cp);
        // ③ attn_norm → q/k/v
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms_rows(self.mtp_b_cur, an, self.mtp_b_e, n, t)?;
        self.ctx
            .quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, t)?;
        // q 투영은 attention(=종료 청크)에서만 필요 — KV는 k/v만 적립한다.
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        if with_head {
            self.mm_b2(
                self.mtp_b_e,
                self.mtp_b_xqn,
                xq_n,
                wq,
                tq,
                niq,
                noq,
                self.aq_t,
                t,
            )?;
        }
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wk,
            tk,
            nik,
            nok,
            self.ak_t,
            t,
        )?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wv,
            tv,
            niv,
            nov,
            self.av_t,
            t,
        )?;
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
            self.ctx
                .launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)?;
        }
        for (src, dst, dst16) in [
            (self.ak_t, self.mtp_kv_k[seq], self.mtp_kv_k16[seq]),
            (self.av_t, self.mtp_kv_v[seq], self.mtp_kv_v16[seq]),
        ] {
            let mut sp = src as *mut std::ffi::c_void;
            let mut dp = dst as *mut std::ffi::c_void;
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
                t as u32,
                1,
                64,
                &mut args,
            )?;
            let dstoff = pos0 * n_kv * hd;
            kv_to_f16(&self.ctx, src, dst16, 0, dstoff, t * n_kv * hd)?;
        }
        {
            // KV-only: 원소 i의 MTP층 출력은 (a) 헤드에서 마지막 행만, (b) 체인은
            // 디코드 스텝의 h를 쓰므로 프롬프트 행들의 attention/wo/FFN은 불필요.
            // 인과 구조상 마지막 행의 출력은 앞 행들의 *KV*만 필요하다 (이미 적립).
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
            if np_ > 128 {
                let sg = 128;
                let nseg = (pos0 + t).div_ceil(sg);
                let part = self.ctx.scratch(nrow_attn * n_head * nseg * (hd + 2) * 4)?;
                let mut pp2 = part as *mut std::ffi::c_void;
                let mut sg_a = sg as i32;
                let mut args = vec![
                    Self::p(&mut qp),
                    Self::p(&mut ckp),
                    Self::p(&mut cvp),
                    Self::p(&mut mp),
                    Self::p(&mut pp2),
                    Self::p(&mut np_),
                    Self::p(&mut nh),
                    Self::p(&mut nk),
                    Self::p(&mut h),
                    Self::p(&mut tl),
                    Self::p(&mut ss),
                    Self::p(&mut p0),
                    Self::p(&mut sg_a),
                ];
                let wk = nrow_attn > 8;
                let (kn2, gx) = if wk {
                    ("qsa_flash_wk", (nrow_attn.div_ceil(32)) as u32)
                } else {
                    ("qsa_flash_split4q4", (nrow_attn.div_ceil(4)) as u32)
                };
                self.ctx
                    .launch3(kn2, gx, n_head as u32, nseg as u32, 256, &mut args)?;
                let mut margs = vec![
                    Self::p(&mut qp),
                    Self::p(&mut pp2),
                    Self::p(&mut op),
                    Self::p(&mut np_),
                    Self::p(&mut nh),
                    Self::p(&mut h),
                    Self::p(&mut tl),
                    Self::p(&mut sg_a),
                ];
                self.ctx.launch3(
                    "qsa_flash_merge",
                    nrow_attn as u32,
                    n_head as u32,
                    1,
                    256,
                    &mut margs,
                )?;
            } else {
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
                    nrow_attn as u32,
                    n_head as u32,
                    1,
                    256,
                    &mut args,
                )?;
            }
        }
        mark("attn+kv", &mut cp);
        // 중간 청크(with_head=false)는 KV 적립만 — 마지막 행의 attention/FFN/헤드도
        // 아무도 읽지 않는다 (초안은 프롬프트 종료 청크에서만 필요).
        if !with_head {
            if mtp_time {
                eprintln!(
                    "[mtpb] TOTAL {:.2}ms (t={t}, kv-only)",
                    t_mtp.elapsed().as_secs_f64() * 1e3
                );
            }
            return Ok(0);
        }
        // ⑤⑥ KV-only: 마지막 행만 (앞 행들의 wo/FFN 출력은 아무도 쓰지 않는다)
        let nrow_ffn = 1usize;
        let coff = (t - 1) * n;
        let cur_p = unsafe { self.mtp_b_cur.add(coff * 4) };
        let aout_p = unsafe { self.aout_t.add(ooff * 4) };
        let gout_p = unsafe { self.gout_t.add(coff * 4) };
        // ⑤ attn_output + 잔차
        self.ctx
            .quant_q8_b(aout_p, self.mtp_b_xqn, n_head * hd, xq_sg, nrow_ffn)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_b2(
            aout_p,
            self.mtp_b_xqn,
            xq_sg,
            wo,
            two,
            nio,
            noo,
            gout_p,
            nrow_ffn,
        )?;
        self.axpy(cur_p, gout_p, n * nrow_ffn)?;
        // ⑥ FFN + 잔차
        let pn = *self
            .consts
            .get("blk.64.post_attention_norm")
            .ok_or("post_norm")?;
        self.rms_rows(cur_p, pn, self.mtp_b_e, n, nrow_ffn)?;
        self.ctx
            .quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, nrow_ffn)?;
        let (wg, tg, nig, nog) = self.w("blk.64.ffn_gate.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wg,
            tg,
            nig,
            nog,
            self.fgate_t,
            nrow_ffn,
        )?;
        let (wu, tu, niu, nou) = self.w("blk.64.ffn_up.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wu,
            tu,
            niu,
            nou,
            self.fup_t,
            nrow_ffn,
        )?;
        {
            let mut gp = self.fgate_t as *mut std::ffi::c_void;
            let mut up = self.fup_t as *mut std::ffi::c_void;
            let mut op = self.fglu_t as *mut std::ffi::c_void;
            let mut na = (self.n_ff * nrow_ffn) as i32;
            let mut args = vec![
                Self::p(&mut gp),
                Self::p(&mut up),
                Self::p(&mut op),
                Self::p(&mut na),
            ];
            self.ew_l("silu_mul_f32", self.n_ff * nrow_ffn, &mut args)?;
        }
        self.ctx
            .quant_q8_b(self.fglu_t, self.mtp_b_xq2, self.n_ff, xq_sf, nrow_ffn)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_b2(
            self.fglu_t,
            self.mtp_b_xq2,
            xq_sf,
            wd,
            td,
            nid,
            nod,
            self.fdown_t,
            nrow_ffn,
        )?;
        self.axpy(cur_p, self.fdown_t, n * nrow_ffn)?;
        mark("ffn", &mut cp);
        // ⑦ 마지막 행만 헤드 — 공유 head norm → output GEMV → argmax
        let shn = *self
            .consts
            .get("blk.64.nextn.shared_head_norm")
            .ok_or("shn")?;
        let last = unsafe { self.mtp_b_cur.add((t - 1) * n * 4) };
        self.rms(last, shn, self.mtp_e, n)?;
        let am = self.head_argmax_gpu(self.mtp_e)?;
        if mtp_time {
            eprintln!(
                "[mtpb] TOTAL {:.2}ms (t={})",
                t_mtp.elapsed().as_secs_f64() * 1e3,
                t
            );
        }
        Ok(am)
    }

    /// plans/135 항목 4 — MTP 드래프트 스텝 슬롯 배칭: 슬롯 t행을 한 번에.
    /// eh_proj·qkv·wo·FFN·head 가중을 t행 상각(head 0.95GB를 4→1회, GEMV 재독
    /// 상각 합계 ≈ -23ms/드래프트 스텝 @np4). rope·KV 적립·flash만 슬롯별
    /// 런치(MTP KV 테이블·pos가 슬롯마다 상이). 산술은 슬롯별 t=1과 같은
    /// mm_b2 패밀리(t=4 → g4 상간) — 초안은 제안이라 수용률에만 영향.
    /// 반환: [t] 초안 토큰 + hs_out = [t][n] h_next (체인 다음 단계 피드 —
    /// 층 출력 mtp_b_cur 행의 d2h). h 입력은 호출자가 관리(초기 = mtp_pending_h).
    pub fn mtp_draft_batch(
        &self,
        seqs: &[usize],
        tok_embs: &[f32],
        hs: &[f32],
        poss: &[usize],
        hs_out: &mut [f32],
    ) -> Result<Vec<u32>, String> {
        if !self.mtp_on {
            return Err("mtp_draft_batch: MTP 미로드".into());
        }
        let t = seqs.len();
        if t == 0 || t > self.b_t_max {
            return Err(format!("mtp_draft_batch: t={t} 범위 밖"));
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv, self.hd);
        let xq_n = crate::rawhip::q4acc::xq_words(n);
        let xq2_w = crate::rawhip::q4acc::xq_words(2 * n);
        let xq_sf = crate::rawhip::q4acc::xq_words(self.n_ff);
        let xq_sg = crate::rawhip::q4acc::xq_words(n_head * hd);
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        // ① 토큰 임베딩·h 입력 적체 → enorm/hnorm → cat
        self.ctx.h2d(self.mtp_b_e, bytemuck::cast_slice(tok_embs))?;
        self.ctx.h2d(self.mtp_b_hs, bytemuck::cast_slice(hs))?;
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
                Self::p(&mut ep),
                Self::p(&mut hp),
                Self::p(&mut op),
                Self::p(&mut na),
                Self::p(&mut ta),
            ];
            self.ctx
                .launch3("cat2_rows", gx, t as u32, 1, 256, &mut args)?;
        }
        // ② eh_proj [2n → n] (t행)
        self.ctx
            .quant_q8_b(self.mtp_b_cat, self.mtp_b_xq2, 2 * n, xq2_w, t)?;
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        self.mm_b2(
            self.mtp_b_cat,
            self.mtp_b_xq2,
            xq2_w,
            we,
            te,
            nie,
            noe,
            self.mtp_b_cur,
            t,
        )?;
        // ③ attn_norm → q/k/v (t행)
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms_rows(self.mtp_b_cur, an, self.mtp_b_e, n, t)?;
        self.ctx
            .quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, t)?;
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wq,
            tq,
            niq,
            noq,
            self.aq_t,
            t,
        )?;
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wk,
            tk,
            nik,
            nok,
            self.ak_t,
            t,
        )?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wv,
            tv,
            niv,
            nov,
            self.av_t,
            t,
        )?;
        let qn = *self.consts.get("blk.64.attn_q_norm").ok_or("qn")?;
        let kn = *self.consts.get("blk.64.attn_k_norm").ok_or("kn")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        // ④ rope·KV 적립·flash — 슬롯별 (테이블·pos 상이). 행 스트라이드는
        // mm_b2가 쓴 실제 n_out(noq/nok/nov) 준수.
        for si in 0..t {
            let sq = seqs[si];
            let pos = poss[si];
            let aq_row = unsafe { self.aq_t.add(si * noq * 4) };
            let ak_row = unsafe { self.ak_t.add(si * nok * 4) };
            let av_row = unsafe { self.av_t.add(si * nov * 4) };
            {
                let mut qp = aq_row as *mut std::ffi::c_void;
                let mut kp = ak_row as *mut std::ffi::c_void;
                let mut qwp = qn as *mut std::ffi::c_void;
                let mut kwp = kn as *mut std::ffi::c_void;
                let mut csp = cs as *mut std::ffi::c_void;
                let mut ep = self.eps;
                let mut kq = self.kq_scale;
                let mut pp = pos as i32;
                let mut nh = n_head as i32;
                let mut nk = n_kv as i32;
                let mut h = hd as i32;
                let mut nr = self.n_rot as i32;
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
                self.ctx
                    .launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
            }
            self.copy(ak_row, self.mtp_kv_k[sq], 0, pos * n_kv * hd, n_kv * hd)?;
            self.copy(av_row, self.mtp_kv_v[sq], 0, pos * n_kv * hd, n_kv * hd)?;
            kv_to_f16(
                &self.ctx,
                ak_row,
                self.mtp_kv_k16[sq],
                0,
                pos * n_kv * hd,
                n_kv * hd,
            )?;
            kv_to_f16(
                &self.ctx,
                av_row,
                self.mtp_kv_v16[sq],
                0,
                pos * n_kv * hd,
                n_kv * hd,
            )?;
            {
                let aout_row = unsafe { self.aout_t.add(si * n_head * hd * 4) };
                let mut qp = aq_row as *mut std::ffi::c_void;
                let mut ckp = self.mtp_kv_k16[sq] as *mut std::ffi::c_void;
                let mut cvp = self.mtp_kv_v16[sq] as *mut std::ffi::c_void;
                let mut mp = mask as *mut std::ffi::c_void;
                let mut op = aout_row as *mut std::ffi::c_void;
                let mut np_ = (pos + 1) as i32;
                let mut nh = n_head as i32;
                let mut nk = n_kv as i32;
                let mut h = hd as i32;
                let mut tl = 1i32;
                let mut ss = self.ctx_len as i32;
                let mut p0 = pos as i32;
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
                self.ctx
                    .launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
            }
        }
        // ⑤ wo + 잔차 (t행)
        self.ctx
            .quant_q8_b(self.aout_t, self.mtp_b_xqn, n_head * hd, xq_sg, t)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_b2(
            self.aout_t,
            self.mtp_b_xqn,
            xq_sg,
            wo,
            two,
            nio,
            noo,
            self.gout_t,
            t,
        )?;
        self.axpy(self.mtp_b_cur, self.gout_t, n * t)?;
        // ⑥ FFN + 잔차 (t행)
        let pn = *self
            .consts
            .get("blk.64.post_attention_norm")
            .ok_or("post_norm")?;
        self.rms_rows(self.mtp_b_cur, pn, self.mtp_b_e, n, t)?;
        self.ctx
            .quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, t)?;
        let (wg, tg, nig, nog) = self.w("blk.64.ffn_gate.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
            wg,
            tg,
            nig,
            nog,
            self.fgate_t,
            t,
        )?;
        let (wu, tu, niu, nou) = self.w("blk.64.ffn_up.weight")?;
        self.mm_b2(
            self.mtp_b_e,
            self.mtp_b_xqn,
            xq_n,
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
        self.ctx
            .quant_q8_b(self.fglu_t, self.mtp_b_xq2, self.n_ff, xq_sf, t)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_b2(
            self.fglu_t,
            self.mtp_b_xq2,
            xq_sf,
            wd,
            td,
            nid,
            nod,
            self.fdown_t,
            t,
        )?;
        self.axpy(self.mtp_b_cur, self.fdown_t, n * t)?;
        // ⑦ 헤드 배치: shared head norm → gemm_tile_head(t행, 가중 1회 독서)
        // → argmax_rows (행별 GPU argmax)
        let shn = *self
            .consts
            .get("blk.64.nextn.shared_head_norm")
            .ok_or("shn")?;
        self.rms_rows(self.mtp_b_cur, shn, self.xn_t, n, t)?;
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_n, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.ctx.gemm_tile_head(
            self.xq_n_t as *const u8,
            wh as *const u8,
            self.ktab2 as *const u8,
            th,
            nih,
            noh,
            xq_n,
            t,
            self.logits_all,
        )?;
        let out = self.argmax_rows(self.logits_all, t, noh)?;
        // h_next 회수 — 체인 다음 단계의 h 입력(층 출력 = mtp_b_cur 행).
        // argmax d2h가 이미 큐를 드레인하므로 동기 비용 추가 없음.
        if hs_out.len() >= t * n {
            self.ctx.d2h(
                bytemuck::cast_slice_mut(&mut hs_out[..t * n]),
                self.mtp_b_cur,
            )?;
        }
        Ok(out)
    }

    /// MTP 1스텝 (호스트 h, head, h_next 회수) — 프리필/디코드 훅용.
    pub fn mtp_step_gpu(
        &self,
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
    pub fn mtp_step_chain(&self, seq: usize, tok_emb: &[f32], pos: usize) -> Result<u32, String> {
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
    pub(in crate::rawhip::decode) fn head_argmax_gpu(&self, x: *mut u8) -> Result<u32, String> {
        let n = self.n_embd;
        self.quant(x, self.mtp_xq, n)?;
        let (wo, to, nio, noo) = self.w("output.weight")?;
        self.mm_into(self.mtp_xq, wo, to, nio, noo, self.logits)?;
        if llm170_diag::dump::opts().key("mtp_stage") {
            self.ctx.sync()?;
            let mut v = vec![0f32; 8];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.logits)?;
            let mut hn8 = vec![0f32; 8];
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut hn8).as_mut(), x)?;
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
