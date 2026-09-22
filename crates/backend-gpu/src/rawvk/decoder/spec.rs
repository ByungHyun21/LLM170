//! vk decoder MTP/spec·상태 스냅샷 (plans/19, plans/79 B).

use super::*;

    // ══ Phase A: MTP·spec·np — 전부 t=1 검증 커널 재사용 (plans/19) ══

impl DecoderState {

    /// b_xs 최종 hidden [n] 판독 (step 완료 후 — GPU 유휴 보장).
    pub(super) fn hidden_row(&self) -> Vec<f32> {
        let n = self.n_embd;
        let mut v = vec![0f32; n];
        unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, v.as_mut_ptr(), n) };
        v
    }

    /// copy_off: src[0..n) → dst[dst_off..).
    pub(super) fn copy_off(&mut self, src: vk::Buffer, dst: vk::Buffer, n: usize, dst_off: usize) -> Result<(), String> {
        let push = Self::push_u32s(&[n as u32, dst_off as u32]);
        self.run_pipe("copy_off", COPY_OFF_SPV, 2, 8,
            &[src, dst], &push, n.div_ceil(256) as u32, 1, 1)
    }

    /// shared head — 정규화 입력(m_e) → 로짓 argmax (b_lg 매핑 판독 + CPU greedy).
    pub(super) fn head_argmax(&mut self) -> Result<u32, String> {
        let n = self.n_embd;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.ctx.begin_batch()?;
        self.gemv(self.m_xq.buf, "output.weight", self.b_lg.buf, 1)?;
        self.ctx.end_batch_wait()?;
        let lgr: &[f32] = unsafe { std::slice::from_raw_parts(self.b_lg.ptr as *const f32, self.n_vocab) };
        Ok(llm170_core::matmul::greedy_from(lgr))
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
        let h_buf = if h_from_cur { self.m_cur.buf } else { self.m_h.buf };
        let noba = std::env::var_os("LLM170_VK_NOBATCH").is_some();
        if !noba { self.ctx.begin_batch()?; }
        // enorm → cat[0..n] ‖ hnorm → cat[n..2n]
        let en = self.consts.get("blk.64.nextn.enorm").cloned().ok_or("enorm")?;
        let hn = self.consts.get("blk.64.nextn.hnorm").cloned().ok_or("hnorm")?;
        self.rms(self.m_e.buf, "blk.64.nextn.enorm", self.m_cat.buf, n, 1)?;
        self.rms(h_buf, "blk.64.nextn.hnorm", self.b_xn.buf, n, 1)?;
        self.copy_off(self.b_xn.buf, self.m_cat.buf, n, n)?;
        let _ = (en, hn);
        // eh_proj [2n → n]
        self.quant(self.m_cat.buf, self.m_xq2.buf, 2 * n, 1)?;
        self.gemv(self.m_xq2.buf, "blk.64.nextn.eh_proj.weight", self.m_cur.buf, 1)?;
        // attn_norm → q/k/v
        self.rms(self.m_cur.buf, "blk.64.attn_norm", self.m_e.buf, n, 1)?;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.attn_q.weight", self.b_aq.buf, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.attn_k.weight", self.b_ak.buf, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.attn_v.weight", self.b_av.buf, 1)?;
        // q/k norm+rope (t=1 — step과 동일 디스패치)
        {
            let qn = self.consts.get("blk.64.attn_q_norm").cloned().ok_or("qn")?;
            let kn = self.consts.get("blk.64.attn_k_norm").cloned().ok_or("kn")?;
            let cs = self.consts.get("cs").cloned().ok_or("cs")?;
            let mut push = self.eps.to_le_bytes().to_vec();
            push.extend_from_slice(&self.kq_scale.to_le_bytes());
            push.extend(Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
            self.run_pipe("qk_rope2", QK_ROPE2_SPV, 5, 28,
                &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                &push, (n_head + n_kv) as u32, 1, 1)?;
        }
        // MTP 자체 KV 적립 + flash (np = pos+1)
        {
            let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
            self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                &[self.b_ak.buf, self.m_kv_k[seq].buf], &push,
                (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
            self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                &[self.b_av.buf, self.m_kv_v[seq].buf], &push,
                (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
        }
        {
            let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32]);
            self.run_pipe("qsa_flash", QSA_FLASH_SPV, 4, 16,
                &[self.b_aq.buf, self.m_kv_k[seq].buf, self.m_kv_v[seq].buf, self.b_aout.buf],
                &push, 1, n_head as u32, 1)?;
        }
        // wo + 잔차
        self.quant(self.b_aout.buf, self.b_xq_g.buf, n_head * hd, 1)?;
        self.gemv(self.b_xq_g.buf, "blk.64.attn_output.weight", self.b_gout.buf, 1)?;
        self.axpy(self.m_cur.buf, self.b_gout.buf, n)?;
        // FFN
        self.rms(self.m_cur.buf, "blk.64.post_attention_norm", self.m_e.buf, n, 1)?;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.ffn_gate.weight", self.b_fgate.buf, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.ffn_up.weight", self.b_fup.buf, 1)?;
        self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
        self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, 1)?;
        self.gemv_w(self.b_fglu.buf, self.b_xq_f.buf, "blk.64.ffn_down.weight", self.b_fdown.buf, 1, self.n_ff)?;
        self.axpy(self.m_cur.buf, self.b_fdown.buf, n)?;
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; }
        if !with_head {
            return Ok(None);
        }
        // shared head norm → head → argmax
        self.rms(self.m_cur.buf, "blk.64.nextn.shared_head_norm", self.m_e.buf, n, 1)?;
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
        // 배치 검증 — LLM170_VKD_SPEC_BATCH=1 명시 옵트인만 (기본 per-token:
        // gemv8 t≥2 검증 배치는 기지 간헐 레이스의 의심 트리거 — plans/36 §8).
        if std::env::var("LLM170_VKD_SPEC_BATCH").map(|v| v == "1").unwrap_or(false) {
            let n = self.n_embd;
            let mut last = Vec::new();
            for (off, ch) in emb.chunks(T_MAX * n).enumerate() {
                let t = ch.len() / n;
                let _tt = std::time::Instant::now();
                let rows = self.step_batch(seq, pos0 + off, ch, true)?;
                if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
                    eprintln!("[vb] step_batch t={t} = {:.1}ms", _tt.elapsed().as_secs_f64()*1e3);
                }
                for r in 0..t {
                    argmaxes.push(llm170_core::matmul::greedy_from(
                        &rows[r * self.n_vocab..(r + 1) * self.n_vocab]));
                }
                let mut hv = vec![0f32; t * n];
                unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, hv.as_mut_ptr(), t * n) };
                h_all.extend_from_slice(&hv);
                last = rows[(t - 1) * self.n_vocab..].to_vec();
            }
            return Ok(last);
        }
        let n = self.n_embd;
        let mut last = Vec::new();
        for (ti, ch) in emb.chunks(n).enumerate() {
            let lg = self.step(seq, pos0 + ti, ch)?;
            argmaxes.push(llm170_core::matmul::greedy_from(&lg));
            h_all.extend_from_slice(&self.hidden_row());
            last = lg;
        }
        Ok(last)
    }

    /// GDN/conv 상태 스냅샷 (매핑 ptr 직접 — GPU 유휴 시).
    pub(super) fn snapshot_states(&mut self) -> Result<(), String> {
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.ctx.end_batch_wait()?;
        }
        let gl = self.dt_rank * self.d_state * self.d_state;
        let cl = (self.conv_k - 1) * self.conv_ch;
        for (r, rows) in self.st_gdn.iter().enumerate() {
            let stride = rows.len();
            for (s, b) in rows.iter().enumerate() {
                let src: &[f32] = unsafe { std::slice::from_raw_parts(b.ptr as *const f32, gl) };
                self.snap_gdn[r * stride + s] = src.to_vec();
                let cs: &[f32] = unsafe { std::slice::from_raw_parts(self.st_conv[r][s].ptr as *const f32, cl) };
                self.snap_conv[r * stride + s] = cs.to_vec();
            }
        }
        Ok(())
    }

    pub(super) fn restore_states(&mut self) -> Result<(), String> {
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.ctx.end_batch_wait()?;
        }
        let gl = self.dt_rank * self.d_state * self.d_state;
        let cl = (self.conv_k - 1) * self.conv_ch;
        for (r, rows) in self.st_gdn.iter().enumerate() {
            let stride = rows.len();
            for (s, b) in rows.iter().enumerate() {
                let snap = self.snap_gdn[r * stride + s].clone();
                if snap.len() == gl {
                    unsafe { std::ptr::copy_nonoverlapping(snap.as_ptr(), b.ptr as *mut f32, gl) };
                }
                let snapc = self.snap_conv[r * stride + s].clone();
                if snapc.len() == cl {
                    unsafe { std::ptr::copy_nonoverlapping(snapc.as_ptr(), self.st_conv[r][s].ptr as *mut f32, cl) };
                }
            }
        }
        Ok(())
    }
}
