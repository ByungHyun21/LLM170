use super::*;

mod moe_align;
mod moe_gemm;
mod moe_gemv;

impl W4a16Dec {
    pub fn moe_pes_scale_host(
        &mut self,
        wt: &[f32],
        idx: &[i32],
        pes: &[f32],
    ) -> Result<Vec<f32>, String> {
        if wt.is_empty() || wt.len() != idx.len() {
            return Err(format!("pes: wt={} idx={} 계약 위반", wt.len(), idx.len()));
        }
        let n = wt.len();
        let dwt = self.cc.alloc(n * 4)?;
        let dix = self.cc.alloc(n * 4)?;
        let dpes = self.cc.alloc(pes.len().max(1) * 4)?;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32b = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let r = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dwt, &f32b(wt))?;
            self.cc.h2d(dix, &i32b(idx))?;
            self.cc.h2d(dpes, &f32b(pes))?;
            let f = self.cc.function("w4a16_moe_pes_scale")?;
            let (mut a0, mut a1, mut a2, mut a3) = (dwt, dix, dpes, n as i32);
            self.cc.launch(
                f,
                1,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut a3),
            )?;
            self.cc.sync()?;
            let mut ob = vec![0u8; n * 4];
            self.cc.d2h(&mut ob, dwt)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dwt);
        let _ = self.cc.free(dix);
        let _ = self.cc.free(dpes);
        r
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_accum_dev(
        &mut self,
        w_dev: CUdeviceptr,
        d_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        sp: usize,
        nslots: usize,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_accum")?;
        let (mut p_w, mut p_d, mut p_y) = (w_dev, d_dev, y_dev);
        let (mut p_sp, mut p_ns, mut nn) = (sp as i32, nslots as i32, n as i32);
        let spt = if sp > 0 { sp } else { nslots.max(1) };
        let rows = nslots.div_ceil(spt) as u32;
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            rows,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_d, &mut p_y, &mut p_sp, &mut p_ns, &mut nn,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_ffn_dev(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.moe.n_experts;
        if n_exp == 0 || self.moe.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        self.ensure_moe_bufs()?;
        if self.moe.moe_resident {
            self.moe_multi_gate(il, xn)?;
            self.moe_topk_dev()?;
            self.moe_experts_batch(il, xn, self.moe.top_k)?;
        } else {
            self.moe_multi_gate(il, xn)?;
            self.moe_topk_dev()?;
            let sel = self.moe_read_selection()?;
            Self::zero_dev(&self.cc, self.moe.dmo.ptr, self.hidden * 4)?;
            self.moe_experts_streaming(il, xn, &sel)?;
        }
        self.moe_shared(il, xn)?;
        Ok(self.moe.dmo.ptr)
    }

    pub(in crate::rawcuda::w4a16_dec) fn build_moe_multi_tab(&mut self) -> Result<(), String> {
        let nl = self.n_layers;
        let (d0, d1) = (self.moe.dsh[0], self.moe.dsh[1]);
        let drt = self.moe.drt.ptr;
        let shared = self.moe.shared_ffn > 0;
        let win = self.window;
        let mut ent: Vec<u64> = Vec::with_capacity(nl * 4 * 4);
        let mut rows: Vec<u32> = vec![0; nl];
        let mut rows_t3: Vec<u32> = vec![0; nl];
        for il in 0..nl {
            if !win.contains(il) {
                ent.extend([0u64; 16]);
                continue;
            }
            let (gw, gn, gk) = self.plain_spec(&format!("blk.{il}.moe_gate.weight"))?;
            ent.extend([gw, drt, gn as u64, gk as u64]);
            let mut r = gn as u32;
            let mut rt3 = gn.div_ceil(GEMV_TR) as u32;
            if shared {
                for (suf, out) in [("moe_shared_gate", d0), ("moe_shared_up", d1)] {
                    let (w, n, k) = self.plain_spec(&format!("blk.{il}.{suf}.weight"))?;
                    ent.extend([w, out, n as u64, k as u64]);
                    r += n as u32;
                    rt3 += n.div_ceil(GEMV_TR) as u32;
                }
                let (w, n, k) = self.plain_spec(&format!("blk.{il}.moe_shared_sgate.weight"))?;
                ent.extend([w, self.moe.dsg, n as u64, k as u64]);
                r += n as u32;
                rt3 += n.div_ceil(GEMV_TR) as u32;
            } else {
                ent.extend([0u64; 12]);
            }
            rows[il] = r;
            rows_t3[il] = rt3;
        }
        let bytes = ent.len() * 8;
        if self.moe.moe_multi_tab != 0 {
            self.cc.free(self.moe.moe_multi_tab)?;
            self.moe.moe_multi_tab = 0;
        }
        let p = self.cc.alloc(bytes)?;
        let mut buf: Vec<u8> = Vec::with_capacity(bytes);
        for v in &ent {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(e) = Self::h2d_chunked(&self.cc, p, &buf) {
            let _ = self.cc.free(p);
            return Err(e);
        }
        self.moe.moe_multi_tab = p;
        self.moe.moe_multi_rows = rows;
        self.moe.moe_multi_rows_t3 = rows_t3;
        Ok(())
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_multi_gate_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.ensure_moe_bufs()?;
        let total = *self
            .moe
            .moe_multi_rows_t3
            .get(il)
            .ok_or("moe: 다중 세그먼트 t 테이블 미작성")?;
        if total == 0 {
            return Ok(());
        }
        self.gemv_multi_t_launch(
            true,
            self.moe.moe_multi_tab + (il as u64) * 4 * 4 * 8,
            xn,
            total as usize,
            4,
            t,
        )
    }

    fn moe_shared_tail_t(
        &mut self,
        il: usize,
        _xn: CUdeviceptr,
        t: usize,
        forked: bool,
    ) -> Result<(), String> {
        let h = self.hidden;
        let sf = self.moe.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        if forked {
            let e2 = self.moe.ev_fork[il * 2 + 1];
            self.cc.stream_wait(self.cc.stream, e2)?;
            return self.shared_add_dev(self.moe.dsg, s3, self.moe.dmo.ptr, h, t);
        }
        self.ew_dev(self.moe.dsh[0], self.moe.dsh[1], s2, t * sf)?;
        self.plain_gemv_t_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.shared_add_dev(self.moe.dsg, s3, self.moe.dmo.ptr, h, t)
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_multi_gate(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<(), String> {
        self.ensure_moe_bufs()?;
        let rows = *self
            .moe
            .moe_multi_rows
            .get(il)
            .ok_or("moe: 다중 세그먼트 테이블 미작성")?;
        if rows == 0 {
            return Ok(());
        }
        let f = self.cc.function("w4a16_gemv_multi")?;
        let (mut p_tab, mut p_x, mut p_c) =
            (self.moe.moe_multi_tab + (il as u64) * 4 * 4 * 8, xn, 4i32);
        self.cc.launch(
            f,
            rows,
            1,
            64,
            &mut crate::rawcuda::args::l3(&mut p_tab, &mut p_x, &mut p_c),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_shared(
        &mut self,
        il: usize,
        _xn: CUdeviceptr,
    ) -> Result<(), String> {
        if self.moe.shared_ffn == 0 {
            return Ok(());
        }
        self.moe_shared_tail(il)
    }

    fn moe_shared_tail(&mut self, il: usize) -> Result<(), String> {
        let h = self.hidden;
        let sf = self.moe.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        self.ew_dev(self.moe.dsh[0], self.moe.dsh[1], s2, sf)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3)?;
        let sg = self.moe.dsg;
        if self.moe.moe_resident {
            self.moe_accum_sg_dev(
                self.moe.moe_wt,
                self.moe.exp.dn,
                sg,
                s3,
                self.moe.dmo.ptr,
                self.moe.top_k.max(1),
                h,
            )?;
        } else {
            self.shared_add_dev(sg, s3, self.moe.dmo.ptr, h, 1)?;
        }
        Ok(())
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_ffn_dev_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.moe.n_experts;
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let tk = self.moe.top_k;
        if n_exp == 0 || self.moe.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        self.ensure_moe_bufs()?;
        let fused = self.moe.shared_ffn > 0 && t <= BATCH_DEC_MAX;
        if fused {
            self.moe_multi_gate_t(il, xn, t)?;
        } else {
            self.plain_gemm_launch(
                &format!("blk.{il}.moe_gate.weight"),
                xn,
                self.moe.drt.ptr,
                t,
            )?;
        }
        let forked = if fused {
            self.moe_shared_head_t_fork(il, xn, t)?
        } else {
            false
        };
        {
            let f = self.cc.function("w4a16_moe_topk_t")?;
            let (mut p_lg, mut p_ix, mut p_wt) =
                (self.moe.drt.ptr, self.moe.moe_idx, self.moe.moe_wt);
            let (mut p_t, mut p_n, mut p_k) = (t as i32, n_exp as i32, tk as i32);
            self.cc.launch(
                f,
                t.div_ceil(8) as u32,
                1,
                256,
                &mut crate::rawcuda::args::l6(
                    &mut p_lg, &mut p_ix, &mut p_wt, &mut p_t, &mut p_n, &mut p_k,
                ),
            )?;
        }
        let ns = t * tk;
        if !self.moe.moe_resident {
            self.moe_stage_t(il, t)?;
        }
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            let mut vb = vec![0u8; t * h * 4];
            self.cc.d2h_async(vb.as_mut_ptr(), xn, t * h * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let rows: Vec<usize> = (0..t)
                .filter(|&r| v[r * h..(r + 1) * h].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] xn nan-rows={rows:?} t={t}");
        }
        let base = il * n_exp * 3;
        let group = (t > 8 || self.moe.moe_resident) && n_exp <= 1024;
        let mut marlin_done = false;
        if group {
            self.moe_align_launch(ns)?;
            if llm170_diag::dump::opts().key("moe_cnt") {
                let mut cb = vec![0u8; n_exp * 4];
                self.cc
                    .d2h_async(cb.as_mut_ptr(), self.moe.moe_gcnt, n_exp * 4)?;
                self.cc.sync()?;
                let cnts: Vec<u32> = cb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                let mut hist = [0usize; 8];
                let mut sum = 0usize;
                let (mut t64, mut t32, mut t16) = (0usize, 0usize, 0usize);
                for &c in &cnts {
                    let c = c as usize;
                    sum += c;
                    t64 += c.div_ceil(64);
                    t32 += c.div_ceil(32);
                    t16 += c.div_ceil(16);
                    let b = match c {
                        0 => 0,
                        1..=8 => 1,
                        9..=16 => 2,
                        17..=32 => 3,
                        33..=48 => 4,
                        49..=64 => 5,
                        65..=96 => 6,
                        _ => 7,
                    };
                    hist[b] += 1;
                }
                eprintln!(
                    "[moe-cnt] il={il} slots={ns} sum={sum} tiles M64={t64} M32={t32} M16={t16} hist[0|1-8|9-16|17-32|33-48|49-64|65-96|97+]={hist:?}"
                );
            }
            marlin_done =
                self.marlin_moe_ffn_dev(il, xn, t, h, n_ff, tk, false, self.moe.dmo.ptr)?;
            if !marlin_done {
                let x16 = self.cast_f16_dev(xn, t * h)?;
                self.gemm_g32_mma_grp2_launch(
                    base,
                    x16,
                    h,
                    tk,
                    self.moe.exp.gate,
                    self.moe.exp.up,
                    n_ff,
                    h,
                )?;
            }
        } else {
            self.gemv_experts_launch(base, ns, xn, h, tk, self.moe.exp.gate, n_ff, h)?;
            self.gemv_experts_launch(base + 1, ns, xn, h, tk, self.moe.exp.up, n_ff, h)?;
        }
        if marlin_done {
            if fused {
                self.moe_shared_tail_t(il, xn, t, forked)?;
            } else {
                self.moe_shared_t(il, xn, t)?;
            }
            return Ok(self.moe.dmo.ptr);
        }
        self.ew_dev(
            self.moe.exp.gate,
            self.moe.exp.up,
            self.moe.exp.act,
            ns * n_ff,
        )?;
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            let mut vb = vec![0u8; ns * n_ff * 4];
            self.cc
                .d2h_async(vb.as_mut_ptr(), self.moe.exp.act, ns * n_ff * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let bad: Vec<usize> = (0..ns)
                .filter(|&s| v[s * n_ff..(s + 1) * n_ff].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] act nan-slots={bad:?} ns={ns}");
        }
        if group {
            let ap = self.cast_f16_dev(self.moe.exp.act, ns * n_ff)?;
            self.gemm_g32_mma_grp_d64_launch(base + 2, ap, n_ff, 1, self.moe.exp.dn, h, n_ff)?;
        } else {
            self.gemv_experts_launch(
                base + 2,
                ns,
                self.moe.exp.act,
                n_ff,
                1,
                self.moe.exp.dn,
                h,
                n_ff,
            )?;
        }
        self.moe_accum_dev(
            self.moe.moe_wt,
            self.moe.exp.dn,
            self.moe.dmo.ptr,
            tk,
            ns,
            h,
        )?;
        if fused {
            self.moe_shared_tail_t(il, xn, t, forked)?;
        } else {
            self.moe_shared_t(il, xn, t)?;
        }
        Ok(self.moe.dmo.ptr)
    }

    fn moe_shared_head_t_fork(
        &mut self,
        il: usize,
        _xn: CUdeviceptr,
        t: usize,
    ) -> Result<bool, String> {
        let aux = self.cc.stream2;
        if aux.is_null() || self.moe.dsg == 0 || self.moe.ev_fork.len() < (il + 1) * 2 {
            return Ok(false);
        }
        let sf = self.moe.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        let (e1, e2) = (self.moe.ev_fork[il * 2], self.moe.ev_fork[il * 2 + 1]);
        self.cc.event_record_on(e1, self.cc.stream)?;
        self.cc.stream_wait(aux, e1)?;
        self.ew_dev_on(aux, self.moe.dsh[0], self.moe.dsh[1], s2, t * sf)?;
        self.plain_gemv_t_launch_on(aux, &format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.cc.event_record_on(e2, aux)?;
        Ok(true)
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_shared_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        if self.moe.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.moe.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0, t)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1, t)?;
        self.ew_dev(s0, s1, s2, t * sf)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.plain_gemm_launch(
            &format!("blk.{il}.moe_shared_sgate.weight"),
            xn,
            self.moe.drt.ptr,
            t,
        )?;
        self.shared_add_dev(self.moe.drt.ptr, s3, self.moe.dmo.ptr, h, t)
    }
}
