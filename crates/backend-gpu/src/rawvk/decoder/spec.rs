//! vk decoder MTP/spec·상태 스냅샷 (plans/19, plans/79 B).

use super::*;

// ══ Phase A: MTP·spec·np — 전부 t=1 검증 커널 재사용 (plans/19) ══

impl DecoderState {
    /// b_xs 최종 hidden [n] 판독 (step 완료 후 — GPU 유휴 보장).
    pub(super) fn hidden_row(&self) -> Vec<f32> {
        let n = self.n_embd;
        let mut v = vec![0f32; n];
        // SAFETY (107 W8): b_xs 매핑 판독 — 호출 계약상 step 완료(end_batch_wait) 후 GPU 유휴; n=n_embd 이하.
        unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, v.as_mut_ptr(), n) };
        v
    }

    /// copy_off: src[0..n) → dst[dst_off..).
    pub(super) fn copy_off(
        &mut self,
        src: vk::Buffer,
        dst: vk::Buffer,
        n: usize,
        dst_off: usize,
    ) -> Result<(), String> {
        let push = Self::push_u32s(&[n as u32, dst_off as u32]);
        self.run_pipe(
            "copy_off",
            COPY_OFF_SPV,
            2,
            8,
            &[src, dst],
            &push,
            n.div_ceil(256) as u32,
            1,
            1,
        )
    }

    /// shared head — 정규화 입력(m_e) → 로짓 → GPU 2단계 argmax(107 W1:
    /// 종전 b_lg 608KB 매핑 판독 + CPU greedy 폐지 — lg_argmax와 동일 판).
    pub(super) fn head_argmax(&mut self) -> Result<u32, String> {
        let n = self.n_embd;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.ctx.begin_batch()?;
        self.gemv(self.m_xq.buf, "output.weight", self.b_lg.buf, 1)?;
        let nthr = 256usize;
        let chunk = 8usize;
        let nv = self.n_vocab;
        let n_wg = nv.div_ceil(nthr * chunk);
        let binds = [self.b_lg.buf, self.b_ams.buf, self.b_am.buf];
        self.run_pipe_b(
            "argmax2",
            crate::rawvk::vkacc::ARGMAX2_SPV,
            3,
            8,
            &binds,
            &Self::push_u32s(&[nv as u32, 0u32]),
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
            &Self::push_u32s(&[n_wg as u32, 1u32]),
            1,
            1,
            1,
            true,
        )?;
        self.ctx.end_batch_wait()?;
        let mut tok = 0u32;
        // SAFETY (107 W8): b_am u32 판독 — 직전 end_batch_wait로 GPU 유휴, 1원소.
        unsafe { std::ptr::copy_nonoverlapping(self.b_am.ptr as *const u32, &mut tok, 1) };
        Ok(tok)
    }

    /// MTP (blk.64) 1스텝 — rawhip mtp_step_g 산술 미러 (t=1 커널 재사용).
    /// h_from_cur=true: h 입력을 내부 m_cur에서 (체인). 반환: with_head면 argmax.
    pub(super) fn mtp_step_g(
        &mut self,
        seq: usize,
        tok_emb: &[f32],
        h_from_cur: bool,
        h_host: &[f32],
        pos: usize,
        with_head: bool,
    ) -> Result<Option<u32>, String> {
        if !self.mtp_on {
            return Err("mtp_step_gpu: MTP 미로드".into());
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        debug_assert_eq!(tok_emb.len(), n);
        // SAFETY (107 W8): m_e/m_h는 n(n_embd) f32로 할당 — 기입/0-채우기 n 이내, 제출 전이라 GPU 접근 없음.
        unsafe {
            std::ptr::copy_nonoverlapping(tok_emb.as_ptr(), self.m_e.ptr as *mut f32, n);
            if !h_from_cur {
                if h_host.len() >= n {
                    std::ptr::copy_nonoverlapping(h_host.as_ptr(), self.m_h.ptr as *mut f32, n);
                } else {
                    std::ptr::write_bytes(self.m_h.ptr as *mut f32, 0, n);
                }
            }
        }
        let h_buf = if h_from_cur {
            self.m_cur.buf
        } else {
            self.m_h.buf
        };
        self.ctx.begin_batch()?;
        // enorm → cat[0..n] ‖ hnorm → cat[n..2n]
        let en = self
            .consts
            .get("blk.64.nextn.enorm")
            .cloned()
            .ok_or("enorm")?;
        let hn = self
            .consts
            .get("blk.64.nextn.hnorm")
            .cloned()
            .ok_or("hnorm")?;
        self.rms(self.m_e.buf, "blk.64.nextn.enorm", self.m_cat.buf, n, 1)?;
        self.rms(h_buf, "blk.64.nextn.hnorm", self.b_xn.buf, n, 1)?;
        self.copy_off(self.b_xn.buf, self.m_cat.buf, n, n)?;
        let _ = (en, hn);
        // eh_proj [2n → n]
        self.gemv_w(
            self.m_cat.buf,
            self.m_xq2.buf,
            "blk.64.nextn.eh_proj.weight",
            self.m_cur.buf,
            1,
            2 * n,
        )?;
        // attn_norm → q/k/v
        self.rms(self.m_cur.buf, "blk.64.attn_norm", self.m_e.buf, n, 1)?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.attn_q.weight",
            self.b_aq.buf,
            1,
            n,
        )?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.attn_k.weight",
            self.b_ak.buf,
            1,
            n,
        )?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.attn_v.weight",
            self.b_av.buf,
            1,
            n,
        )?;
        // q/k norm+rope (t=1 — step과 동일 디스패치)
        {
            let qn = self.consts.get("blk.64.attn_q_norm").cloned().ok_or("qn")?;
            let kn = self.consts.get("blk.64.attn_k_norm").cloned().ok_or("kn")?;
            let cs = self.consts.get("cs").cloned().ok_or("cs")?;
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
        // MTP 자체 KV 적립 + flash (np = pos+1)
        {
            let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
            self.run_pipe(
                "kv_app",
                KV_APPEND_SPV,
                2,
                8,
                &[self.b_ak.buf, self.m_kv_k[seq].buf],
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
                &[self.b_av.buf, self.m_kv_v[seq].buf],
                &push,
                (n_kv * hd).div_ceil(64) as u32,
                1,
                1,
            )?;
        }
        {
            let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32]);
            self.run_pipe(
                "qsa_flash",
                QSA_FLASH_SPV,
                4,
                16,
                &[
                    self.b_aq.buf,
                    self.m_kv_k[seq].buf,
                    self.m_kv_v[seq].buf,
                    self.b_aout.buf,
                ],
                &push,
                1,
                n_head as u32,
                1,
            )?;
        }
        // wo + 잔차
        self.gemv_w(
            self.b_aout.buf,
            self.b_xq_g.buf,
            "blk.64.attn_output.weight",
            self.b_gout.buf,
            1,
            n_head * hd,
        )?;
        self.axpy(self.m_cur.buf, self.b_gout.buf, n)?;
        // FFN
        self.rms(
            self.m_cur.buf,
            "blk.64.post_attention_norm",
            self.m_e.buf,
            n,
            1,
        )?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.ffn_gate.weight",
            self.b_fgate.buf,
            1,
            n,
        )?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.ffn_up.weight",
            self.b_fup.buf,
            1,
            n,
        )?;
        self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
        self.gemv_w(
            self.b_fglu.buf,
            self.b_xq_f.buf,
            "blk.64.ffn_down.weight",
            self.b_fdown.buf,
            1,
            self.n_ff,
        )?;
        self.axpy(self.m_cur.buf, self.b_fdown.buf, n)?;
        self.ctx.end_batch_wait()?;
        if !with_head {
            return Ok(None);
        }
        // shared head norm → head → argmax
        self.rms(
            self.m_cur.buf,
            "blk.64.nextn.shared_head_norm",
            self.m_e.buf,
            n,
            1,
        )?;
        Ok(Some(self.head_argmax()?))
    }

    /// per-token 검증 — 행별 step + argmax + hidden 회수.
    pub(super) fn verify_rows(
        &mut self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        // 배치 검증 기본 (plans/91 P2): step_batch t행 == step() 행별 비트 동일
        // (plans/20 계약 + P0 verify_np_self 재확보 — gemv8t 2..4토큰 포함).
        // 배치 검증 (plans/91 P2): step_batch t행 == step() 행별 비트 동일
        // (plans/20 계약 + P0 verify_np_self 재확보 — gemv8t 2..4토큰 포함).
        // per-token 복원(VKD_SPEC_BATCH=0)은 plans/109 P6 삭제.

        let n = self.n_embd;
        for (off, ch) in emb.chunks(T_MAX * n).enumerate() {
            let t = ch.len() / n;
            let _tt = std::time::Instant::now();
            // 107 W1: all_logits=true는 이제 전사 없이 b_lg_t에 상주
            // (step_batch 계약 변경 — 유일 소비자가 이 경로다).
            let _ = self.step_batch(seq, pos0 + off, ch, true)?;
            // fn_argmax_rows 2단계 GPU argmax — t×608KB 전사·CPU 스캔 폐지.
            let nv = self.n_vocab;
            let n_wg = nv.div_ceil(256 * 8);
            let push0 = Self::push_u32s(&[nv as u32, 0u32, n_wg as u32]);
            let push1 = Self::push_u32s(&[nv as u32, 1u32, n_wg as u32]);
            let binds = [self.b_lg_t.buf, self.b_amsc.buf, self.b_amr.buf];
            self.ctx.begin_batch()?;
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
            self.ctx.end_batch_wait()?;
            let mut toks = vec![0u32; t];
            unsafe {
                std::ptr::copy_nonoverlapping(self.b_amr.ptr as *const u32, toks.as_mut_ptr(), t)
            };
            argmaxes.extend_from_slice(&toks);
            let mut hv = vec![0f32; t * n];
            unsafe {
                std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, hv.as_mut_ptr(), t * n)
            };
            h_all.extend_from_slice(&hv);
        }
        Ok(Vec::new())
    }

    /// MTP 프리필 배치 (plans/91 P2) — blk.64를 t행 1패스. rawhip
    /// mtp_prefill_batch 구조 이식: KV-only 최적화(중간 청크는 KV 적립만,
    /// 종료 청크는 마지막 행의 attention/FFN/헤드만) + h_shift 디바이스 조립.
    /// GEMM은 gemv_stage(t행 — 무게 1회 판독), rope/kv/flash는 기존 t행 커널.
    pub(super) fn mtp_prefill_batch_ex(
        &mut self,
        seq: usize,
        tok_embs: &[f32],
        carry_h: &[f32],
        t: usize,
        pos0: usize,
        with_head: bool,
    ) -> Result<u32, String> {
        if !self.mtp_on {
            return Err("mtp_prefill_batch: MTP 미로드".into());
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        if t == 0 || t > T_MAX {
            return Err(format!("mtp_prefill_batch: t={t} 범위 밖"));
        }
        let mtp_time = llm170_diag::flag::on("LLM170_MTP_TIMING");
        let t0 = std::time::Instant::now();
        // ① 임베딩: 선반입(있으면 그대로, 없으면 업로드) + h_shift 디바이스 조립
        if !self
            .m_prefetched
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            unsafe {
                std::ptr::copy_nonoverlapping(tok_embs.as_ptr(), self.m_be.ptr as *mut f32, t * n);
            }
        }
        if carry_h.len() != n {
            return Err(format!(
                "mtp_prefill_batch: carry_h len {} != {n}",
                carry_h.len()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(carry_h.as_ptr(), self.m_h.ptr as *mut f32, n);
        }
        self.ctx.begin_batch()?;
        {
            let push = Self::push_u32s(&[n as u32, t as u32]);
            self.run_pipe(
                "row_shift_gather",
                ROW_SHIFT_GATHER_SPV,
                3,
                8,
                &[self.b_xs.buf, self.m_h.buf, self.m_bhs.buf],
                &push,
                (n * t).div_ceil(256) as u32,
                1,
                1,
            )?;
        }
        // enorm(tok) ‖ hnorm(h_shift) → cat
        self.rms(self.m_be.buf, "blk.64.nextn.enorm", self.m_bcur.buf, n, t)?;
        self.rms(self.m_bhs.buf, "blk.64.nextn.hnorm", self.m_be.buf, n, t)?;
        {
            let push = Self::push_u32s(&[n as u32, t as u32]);
            self.run_pipe(
                "cat2_rows",
                CAT2_ROWS_SPV,
                3,
                8,
                &[self.m_bcur.buf, self.m_be.buf, self.m_bcat.buf],
                &push,
                n.div_ceil(256) as u32,
                t as u32,
                1,
            )?;
        }
        // ② eh_proj [2n → n]
        self.quant(self.m_bcat.buf, self.m_bxq2.buf, 2 * n, t)?;
        self.gemv(
            self.m_bxq2.buf,
            "blk.64.nextn.eh_proj.weight",
            self.m_bcur.buf,
            t,
        )?;
        // ③ attn_norm → q/k/v (q는 종료 청크만)
        self.rms(self.m_bcur.buf, "blk.64.attn_norm", self.m_be.buf, n, t)?;
        self.quant(self.m_be.buf, self.m_bxqn.buf, n, t)?;
        if with_head {
            self.gemv(self.m_bxqn.buf, "blk.64.attn_q.weight", self.b_aq.buf, t)?;
        }
        self.gemv(self.m_bxqn.buf, "blk.64.attn_k.weight", self.b_ak.buf, t)?;
        self.gemv(self.m_bxqn.buf, "blk.64.attn_v.weight", self.b_av.buf, t)?;
        // ④ rope(배치) + KV 적립(배치)
        {
            let qn = self.consts.get("blk.64.attn_q_norm").cloned().ok_or("qn")?;
            let kn = self.consts.get("blk.64.attn_k_norm").cloned().ok_or("kn")?;
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
        {
            let push = Self::push_u32s(&[(n_kv * hd) as u32, pos0 as u32]);
            self.run_pipe(
                "kv_app",
                KV_APPEND_SPV,
                2,
                8,
                &[self.b_ak.buf, self.m_kv_k[seq].buf],
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
                &[self.b_av.buf, self.m_kv_v[seq].buf],
                &push,
                (n_kv * hd).div_ceil(64) as u32,
                t as u32,
                1,
            )?;
        }
        if !with_head {
            // 중간 청크: KV 적립만 — 마지막 행의 attention/FFN/헤드는 아무도
            // 읽지 않는다(초안은 종료 청크에서만).
            self.ctx.end_batch_wait()?;
            if mtp_time {
                eprintln!(
                    "[mtpb] vk TOTAL {:.2}ms (t={t}, kv-only)",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            return Ok(0);
        }
        // ⑤ 종료 청크: 마지막 행만 attention → wo → FFN → 헤드.
        // 마지막 행 q를 b_aq 행0으로 복사(매핑 ptr 직접) 후 t=1 플래시
        // (pos = pos0+t-1 — 그 행의 인과 상한 np = pos+1).
        let qstride = n_head * 2 * hd;
        // SAFETY (107 W8): b_aq 내부 이동 — .add((t-1)*qstride*4)는 t행 할당의 마지막 행; overlap되는 ptr::copy(memmove) 사용. 직전 end_batch_wait로 유휴.
        unsafe {
            std::ptr::copy(
                self.b_aq.ptr.add((t - 1) * qstride * 4) as *const f32,
                self.b_aq.ptr as *mut f32,
                qstride,
            );
        }
        {
            let pos_last = (pos0 + t - 1) as u32;
            let push = Self::push_u32s(&[pos_last, n_head as u32, n_kv as u32, hd as u32]);
            self.run_pipe(
                "qsa_flash",
                QSA_FLASH_SPV,
                4,
                16,
                &[
                    self.b_aq.buf,
                    self.m_kv_k[seq].buf,
                    self.m_kv_v[seq].buf,
                    self.b_aout.buf,
                ],
                &push,
                1,
                n_head as u32,
                1,
            )?;
        }
        // SAFETY (107 W8): m_bcur 마지막 행 판독 — (t-1)*n*4는 t*n 할당 이내; 청크 종료 대기(GPU 유휴) 후.
        // 마지막 행 잔여 트렁크 — t=1 버퍼로 복사해 기존 헬퍼 재사용.
        unsafe {
            let cur_last = self.m_bcur.ptr.add((t - 1) * n * 4) as *const f32;
            std::ptr::copy_nonoverlapping(cur_last, self.m_cur.ptr as *mut f32, n);
        }
        self.gemv_w(
            self.b_aout.buf,
            self.m_xq.buf,
            "blk.64.attn_output.weight",
            self.b_gout.buf,
            1,
            n_head * hd,
        )?;
        self.axpy(self.m_cur.buf, self.b_gout.buf, n)?;
        self.rms(
            self.m_cur.buf,
            "blk.64.post_attention_norm",
            self.m_e.buf,
            n,
            1,
        )?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.ffn_gate.weight",
            self.b_fgate.buf,
            1,
            n,
        )?;
        self.gemv_w(
            self.m_e.buf,
            self.m_xq.buf,
            "blk.64.ffn_up.weight",
            self.b_fup.buf,
            1,
            n,
        )?;
        self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
        self.gemv_w(
            self.b_fglu.buf,
            self.m_xq.buf,
            "blk.64.ffn_down.weight",
            self.b_fdown.buf,
            1,
            self.n_ff,
        )?;
        self.axpy(self.m_cur.buf, self.b_fdown.buf, n)?;
        self.ctx.end_batch_wait()?;
        // ⑥ shared head norm → head → argmax
        self.rms(
            self.m_cur.buf,
            "blk.64.nextn.shared_head_norm",
            self.m_e.buf,
            n,
            1,
        )?;
        let am = self.head_argmax()?;
        if mtp_time {
            eprintln!(
                "[mtpb] vk TOTAL {:.2}ms (t={t})",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(am)
    }

    /// GDN/conv 상태 스냅샷 — 단일 디바이스 버퍼 D2D 복사(107 W1 spec2 수리).
    /// 종전 호스트 왕복(매핑 GTT 판독 ~580ms/사이클) 폐지. lazy 1회 할당.
    pub(super) fn snapshot_states(&mut self) -> Result<(), String> {
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.ctx.end_batch_wait()?;
        }
        let (gl, cl) = (self.gdn_len(), self.conv_len());
        let mut copies: Vec<(vk::Buffer, u64, vk::Buffer, u64, u64)> = Vec::new();
        let mut off = 0usize;
        for rows in self.st_gdn.iter() {
            for b in rows.iter() {
                copies.push((b.buf, 0, vk::Buffer::null(), off as u64 * 4, gl as u64 * 4));
                off += gl;
            }
        }
        for rows in self.st_conv.iter() {
            for b in rows.iter() {
                copies.push((b.buf, 0, vk::Buffer::null(), off as u64 * 4, cl as u64 * 4));
                off += cl;
            }
        }
        let total = off * 4;
        let snap = match self.gdn_snap.as_ref() {
            Some(b) if b.bytes >= total => b.buf,
            _ => {
                let b = self.ctx.alloc(total)?;
                let buf = b.buf;
                self.gdn_snap = Some(b);
                buf
            }
        };
        for c in copies.iter_mut() {
            c.2 = snap;
        }
        self.ctx.copy_dev(&copies)
    }

    pub(super) fn restore_states(&mut self) -> Result<(), String> {
        let Some(snap) = self.gdn_snap.clone() else {
            return Ok(());
        };
        let (gl, cl) = (self.gdn_len(), self.conv_len());
        let mut copies: Vec<(vk::Buffer, u64, vk::Buffer, u64, u64)> = Vec::new();
        let mut off = 0usize;
        for rows in self.st_gdn.iter() {
            for b in rows.iter() {
                copies.push((snap.buf, off as u64 * 4, b.buf, 0, gl as u64 * 4));
                off += gl;
            }
        }
        for rows in self.st_conv.iter() {
            for b in rows.iter() {
                copies.push((snap.buf, off as u64 * 4, b.buf, 0, cl as u64 * 4));
                off += cl;
            }
        }
        self.ctx.copy_dev(&copies)
    }

    /// per-seq 복원 (np×spec 부분수용) — 해당 슬롯 행만.
    pub(super) fn restore_seq_states(&mut self, seq: usize) -> Result<(), String> {
        let Some(snap) = self.gdn_snap.clone() else {
            return Ok(());
        };
        let (gl, cl) = (self.gdn_len(), self.conv_len());
        let mut copies: Vec<(vk::Buffer, u64, vk::Buffer, u64, u64)> = Vec::new();
        let mut off = 0usize;
        for rows in self.st_gdn.iter() {
            let stride = rows.len();
            if seq < stride {
                copies.push((
                    snap.buf,
                    (off + seq * gl) as u64 * 4,
                    rows[seq].buf,
                    0,
                    gl as u64 * 4,
                ));
            }
            off += stride * gl;
        }
        for rows in self.st_conv.iter() {
            let stride = rows.len();
            if seq < stride {
                copies.push((
                    snap.buf,
                    (off + seq * cl) as u64 * 4,
                    rows[seq].buf,
                    0,
                    cl as u64 * 4,
                ));
            }
            off += stride * cl;
        }
        self.ctx.copy_dev(&copies)
    }

    fn gdn_len(&self) -> usize {
        self.dt_rank * self.d_state * self.d_state
    }
    fn conv_len(&self) -> usize {
        (self.conv_k - 1) * self.conv_ch
    }
}
