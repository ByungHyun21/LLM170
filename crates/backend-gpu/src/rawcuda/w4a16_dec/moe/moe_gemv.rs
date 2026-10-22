use super::super::*;

impl W4a16Dec {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn gemv_experts_launch(
        &self,
        base: usize,
        nslots: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let sym = match (self.moe.moe_group, self.moe.moe_scale_bf16) {
            (32, false) => "w4a16_gemv_experts_g32_f16",
            (32, true) => "w4a16_gemv_experts_g32_bf16",
            (128, false) => "w4a16_gemv_experts_g128_f16",
            (g, b) => return Err(format!("gemv_experts: 미지원 포맷 group={g} bf16={b}")),
        };
        let f = self.cc.function(sym)?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) = (
            self.moe.moe_dev_tab,
            base as i32,
            self.moe.moe_idx,
            nslots as i32,
        );
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            out_dev,
            n as i32,
            k as i32,
        );
        self.cc.launch(
            f,
            (n * nslots) as u32,
            1,
            64,
            &mut crate::rawcuda::args::l10(
                &mut p_t, &mut p_b, &mut p_i, &mut p_ns, &mut p_x, &mut p_xs, &mut p_sp, &mut p_o,
                &mut p_n, &mut p_k,
            ),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn gemv_experts_glu_launch(
        &self,
        base: usize,
        nslots: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        act_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function(if self.moe.glu_gelu {
            "w4a16_gemv_experts_glu_gelu"
        } else {
            "w4a16_gemv_experts_glu"
        })?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) = (
            self.moe.moe_dev_tab,
            base as i32,
            self.moe.moe_idx,
            nslots as i32,
        );
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            act_dev,
            n as i32,
            k as i32,
        );
        self.cc.launch(
            f,
            (n * nslots) as u32,
            1,
            64,
            &mut crate::rawcuda::args::l10(
                &mut p_t, &mut p_b, &mut p_i, &mut p_ns, &mut p_x, &mut p_xs, &mut p_sp, &mut p_o,
                &mut p_n, &mut p_k,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_experts_batch(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        ns: usize,
    ) -> Result<(), String> {
        let n_exp = self.moe.n_experts;
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let base = il * n_exp * 3;
        self.gemv_experts_glu_launch(base, ns, xn, 0, ns, self.moe.exp.act, n_ff, h)?;
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
        Ok(())
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_experts_streaming(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        sel: &[(usize, f32)],
    ) -> Result<(), String> {
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        if sel.is_empty() {
            return Ok(());
        }
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe.moe_group * 2;
        debug_assert!(
            self.moe.stg.cap.0 >= qsz * sel.len() * 3 && self.moe.stg.cap.1 >= ssz * sel.len() * 3
        );
        let use_cache = self.moe.mcache_k > 0;
        let hot = self.moe.hot_per_layer;
        let mut qd_list: Vec<[(u64, u64); 3]> = Vec::with_capacity(sel.len());
        for (j, &(e, _)) in sel.iter().enumerate() {
            let trio = if hot > 0 && e < hot {
                let b = (il * self.moe.n_experts + e) * 3;
                let t = &self.moe.hot_tab[b..b + 3];
                [(t[0].0, t[0].2), (t[1].0, t[1].2), (t[2].0, t[2].2)]
            } else {
                let (qd, sd, need_h2d) = if use_cache {
                    match self.mcache_lookup(il, e) {
                        Some(slot) => {
                            self.moe.mcache_hits += 1;
                            let qb = self.moe.mcache_q
                                + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
                            let sb = self.moe.mcache_s
                                + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
                            (qb, sb, false)
                        }
                        None => {
                            self.moe.mcache_miss += 1;
                            let slot = self.mcache_insert(il, e)?;
                            let qb = self.moe.mcache_q
                                + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
                            let sb = self.moe.mcache_s
                                + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
                            (qb, sb, true)
                        }
                    }
                } else {
                    let qb = self.moe.stg.q + ((j * 3) * qsz) as u64;
                    let sb = self.moe.stg.s + ((j * 3) * ssz) as u64;
                    (qb, sb, true)
                };
                if need_h2d {
                    self.mcache_upload_to(il, e, qd, sd)?;
                }
                [
                    (qd, sd),
                    (qd + qsz as u64, sd + ssz as u64),
                    (qd + 2 * qsz as u64, sd + 2 * ssz as u64),
                ]
            };
            qd_list.push(trio);
        }
        self.cc.sync()?;
        if use_cache
            && llm170_diag::flag::on_nonzero("LLM170_MOE_DBG")
            && self.moe.mcache_hits + self.moe.mcache_miss > 0
            && (self.moe.mcache_hits + self.moe.mcache_miss).is_multiple_of(2560)
        {
            eprintln!(
                "[mcache] hit {} / {} ({:.1}%)",
                self.moe.mcache_hits,
                self.moe.mcache_hits + self.moe.mcache_miss,
                100.0 * self.moe.mcache_hits as f64
                    / (self.moe.mcache_hits + self.moe.mcache_miss) as f64
            );
        }
        for (j, &(_e, w)) in sel.iter().enumerate() {
            let [(q0, p0), (q1, p1), (q2, p2)] = qd_list[j];
            self.gemv_launch_raw(q0, p0, n_ff, h, xn, s0)?;
            self.gemv_launch_raw(q1, p1, n_ff, h, xn, s1)?;
            self.ew_dev(s0, s1, s2, n_ff)?;
            self.gemv_launch_raw(q2, p2, h, n_ff, s2, s3)?;
            self.axpy_dev(w, s3, self.moe.dmo.ptr, h)?;
        }
        Ok(())
    }

    fn mcache_upload_to(&mut self, il: usize, e: usize, qd: u64, sd: u64) -> Result<(), String> {
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
        let base = (il * self.moe.n_experts + e) * 3;
        for p in 0..3 {
            let (qp, ql, sp, sl) = self.moe.moe_tab[base + p];
            let qdd = qd + (p * qsz) as u64;
            let sdd = sd + (p * ssz) as u64;
            let qb = unsafe { std::slice::from_raw_parts(qp as *const u8, ql as usize) };
            let sb = unsafe { std::slice::from_raw_parts(sp as *const u8, sl as usize) };
            self.cc.h2d_async(qdd, qb)?;
            self.cc.h2d_async(sdd, sb)?;
        }
        Ok(())
    }

    fn mcache_tick_bump(&mut self) {
        self.moe.mcache_tick = self.moe.mcache_tick.wrapping_add(1);
        if self.moe.mcache_tick == 0 {
            self.moe.mcache_age.fill(0);
        }
    }

    fn mcache_lookup(&mut self, il: usize, e: usize) -> Option<usize> {
        self.mcache_tick_bump();
        let base = il * self.moe.mcache_k;
        for slot in 0..self.moe.mcache_k {
            if self.moe.mcache_ids[base + slot] == e as u16 {
                self.moe.mcache_age[base + slot] = self.moe.mcache_tick;
                return Some(slot);
            }
        }
        None
    }

    fn mcache_insert(&mut self, il: usize, e: usize) -> Result<usize, String> {
        self.mcache_tick_bump();
        let base = il * self.moe.mcache_k;
        let mut slot = 0usize;
        let mut best = u32::MAX;
        for s in 0..self.moe.mcache_k {
            if self.moe.mcache_ids[base + s] == u16::MAX {
                slot = s;
                break;
            }
            if self.moe.mcache_age[base + s] < best {
                best = self.moe.mcache_age[base + s];
                slot = s;
            }
        }
        self.moe.mcache_ids[base + slot] = e as u16;
        self.moe.mcache_age[base + slot] = self.moe.mcache_tick;
        if self.moe.moe_dev_tab != 0 {
            let h = self.hidden;
            let n_ff = self.moe.moe_ffn;
            let qsz = h * n_ff / 2;
            let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
            let qb = self.moe.mcache_q + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
            let sb = self.moe.mcache_s + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
            let mut bytes = [0u8; 48];
            for p in 0..3 {
                let q = qb + (p * qsz) as u64;
                let sv = sb + (p * ssz) as u64;
                bytes[p * 16..p * 16 + 8].copy_from_slice(&q.to_le_bytes());
                bytes[p * 16 + 8..p * 16 + 16].copy_from_slice(&sv.to_le_bytes());
            }
            let off = ((il * self.moe.n_experts + e) * 3) as u64 * 2 * 8;
            self.cc.h2d_async(self.moe.moe_dev_tab + off, &bytes)?;
        }
        Ok(slot)
    }

    pub(super) fn moe_stage_t(&mut self, il: usize, t: usize) -> Result<(), String> {
        if self.moe.mcache_k == 0 || self.moe.moe_dev_tab == 0 {
            return Err("moe 스트리밍 t>1: 전문가 캐시 미구성".into());
        }
        let tk = self.moe.top_k.max(1);
        let ns = t * tk;
        let need = ns * 4;
        if self.moe.pin_moe_idx_cap < need {
            if !self.moe.pin_moe_idx.is_null() {
                self.cc.pinned_free(self.moe.pin_moe_idx)?;
                self.moe.pin_moe_idx = std::ptr::null_mut();
                self.moe.pin_moe_idx_cap = 0;
            }
            self.moe.pin_moe_idx = self.cc.pinned_alloc(need * 2)?;
            self.moe.pin_moe_idx_cap = need * 2;
        }
        let ib: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(self.moe.pin_moe_idx as *mut u8, need) };
        self.cc.d2h_async(ib.as_mut_ptr(), self.moe.moe_idx, need)?;
        self.cc.sync()?;
        let mut seen = vec![false; self.moe.n_experts];
        let mut distinct = 0usize;
        let hot = self.moe.hot_per_layer;
        for j in 0..ns {
            let e = u32::from_le_bytes(ib[j * 4..j * 4 + 4].try_into().unwrap()) as usize;
            if e >= self.moe.n_experts || e < hot || seen[e] {
                continue;
            }
            seen[e] = true;
            distinct += 1;
        }
        if distinct > self.moe.mcache_k {
            return Err(format!(
                "moe streaming: 청크 distinct {distinct} > 캐시 {} — t 상한 초과",
                self.moe.mcache_k
            ));
        }
        for j in 0..ns {
            let e = u32::from_le_bytes(ib[j * 4..j * 4 + 4].try_into().unwrap()) as usize;
            if e >= self.moe.n_experts || !seen[e] {
                continue;
            }
            if self.mcache_lookup(il, e).is_none() {
                self.moe.mcache_miss += 1;
                let slot = self.mcache_insert(il, e)?;
                let h = self.hidden;
                let n_ff = self.moe.moe_ffn;
                let qsz = h * n_ff / 2;
                let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
                let qb =
                    self.moe.mcache_q + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
                let sb =
                    self.moe.mcache_s + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
                self.mcache_upload_to(il, e, qb, sb)?;
            } else {
                self.moe.mcache_hits += 1;
            }
        }
        Ok(())
    }
}
