use super::*;

impl W4a16Dec {
    pub(super) fn graph_ok(&self) -> bool {
        llm170_diag::flag::ne0("LLM170_GRAPH")
            && !self.debug_layers
            && !self.graph.graph_failed
            && (self.moe.n_experts == 0 || self.moe.moe_resident)
    }

    pub(super) fn graph_invalidate(&mut self) {
        for e in self.graph.graph_cache.drain(..) {
            let _ = self.cc.graph_destroy(e.exec, e.handle);
        }
        for (_, e, g) in self.graph.batch_graphs.drain(..) {
            let _ = self.cc.graph_destroy(e, g);
        }
        for (_, e, g) in self.graph.spec_graphs.drain(..) {
            let _ = self.cc.graph_destroy(e, g);
        }
        for g in self.mtp.mtp_graphs.iter_mut() {
            if let Some((e, h)) = g.take() {
                let _ = self.cc.graph_destroy(e, h);
            }
        }
        for g in self.mtp.mtp_chain_graphs.iter_mut() {
            if let Some((e, h)) = g.take() {
                let _ = self.cc.graph_destroy(e, h);
            }
        }
        for g in self.kstep.graphs.iter_mut() {
            if let Some((_, e, h)) = g.take() {
                let _ = self.cc.graph_destroy(e, h);
            }
        }
        for e in self.graph.g4_graphs.drain(..) {
            let _ = self.cc.graph_destroy(e.exec, e.handle);
        }
    }

    pub(super) fn ensure_graph(
        &mut self,
        slot: usize,
        head: bool,
        argmax: bool,
    ) -> Result<ffi::CUgraphExec, String> {
        if let Some(e) = self
            .graph
            .graph_cache
            .iter()
            .find(|e| e.slot == slot && e.head == head && e.argmax == argmax)
        {
            return Ok(e.exec);
        }
        self.warm_for_capture()?;
        if self.graph.pin_embed.is_null() {
            self.cc.create_stream()?;
            self.graph.pin_embed = self.cc.pinned_alloc(self.hidden * 4)?;
            self.graph.pin_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
        }
        let out_len = if head {
            self.head_n * 4
        } else {
            self.hidden * 4
        };
        if self.graph.pin_out.is_null() || self.graph.pin_out_len < out_len {
            self.graph_invalidate();
            if !self.graph.pin_out.is_null() {
                let _ = self.cc.pinned_free(self.graph.pin_out);
            }
            self.graph.pin_out = self.cc.pinned_alloc(out_len)?;
            self.graph.pin_out_len = out_len;
        }
        self.cc.capture_begin()?;
        self.graph.capture_pinned_src = true;
        let cap = (|| -> Result<(), String> {
            let row = vec![0f32; self.hidden];
            let xn = self.chain_device(slot, &row)?;
            if head {
                self.head_gemv_launch(xn, self.head_out)?;
                if argmax {
                    let fa = self.cc.function("w4a16_argmax_min")?;
                    let (mut p_l, mut p_n, mut p_o) =
                        (self.head_out, self.head_n as i32, self.argmax_out);
                    self.cc.launch(
                        fa,
                        1,
                        1,
                        1024,
                        &mut crate::rawcuda::args::l3(&mut p_l, &mut p_n, &mut p_o),
                    )?;
                    self.cc
                        .d2h_async(self.graph.pin_out as *mut u8, self.argmax_out, 4)?;
                } else {
                    self.cc.d2h_async(
                        self.graph.pin_out as *mut u8,
                        self.head_out,
                        self.head_n * 4,
                    )?;
                }
            } else {
                self.cc
                    .d2h_async(self.graph.pin_out as *mut u8, xn, self.hidden * 4)?;
            }
            Ok(())
        })();
        self.graph.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end();
            return Err(format!("캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?;
        let cap = self.n_slots.max(1) * 3;
        if self.graph.graph_cache.len() >= cap {
            let old = self.graph.graph_cache.remove(0);
            let _ = self.cc.graph_destroy(old.exec, old.handle);
        }
        self.graph.graph_cache.push(GraphEntry {
            exec: e,
            handle: g,
            slot,
            head,
            argmax,
        });
        eprintln!(
            "[graph] captured slot={slot} head={head} argmax={argmax} moe_resident={}",
            self.moe.moe_resident
        );
        Ok(e)
    }

    pub(super) fn graph_replay(
        &mut self,
        exec: ffi::CUgraphExec,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        let pos = self.slot_pos[slot];
        self.kv_ensure(slot, pos as usize + 3)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                embed_row.as_ptr() as *const u8,
                self.graph.pin_embed as *mut u8,
                self.hidden * 4,
            );
            std::ptr::copy_nonoverlapping(
                pos.to_le_bytes().as_ptr(),
                (self.graph.pin_pos as *mut u8).add(slot * 4),
                4,
            );
        }
        let posb =
            unsafe { std::slice::from_raw_parts(self.graph.pin_pos as *const u8, self.n_slots * 4) };
        self.cc.h2d_async(
            self.attn.dpp + (slot as u64) * 4,
            &posb[slot * 4..slot * 4 + 4],
        )?;
        self.cc.graph_launch(exec)?;
        self.cc.sync()?;
        let out =
            unsafe { std::slice::from_raw_parts(self.graph.pin_out as *const f32, self.graph.pin_out_len / 4) };
        let v = out.to_vec();
        self.slot_pos[slot] = pos + 1;
        Ok(v)
    }

    pub(super) fn graph_replay_argmax(
        &mut self,
        exec: ffi::CUgraphExec,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<u32, String> {
        let pos = self.slot_pos[slot];
        self.kv_ensure(slot, pos as usize + 3)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                embed_row.as_ptr() as *const u8,
                self.graph.pin_embed as *mut u8,
                self.hidden * 4,
            );
            std::ptr::copy_nonoverlapping(
                pos.to_le_bytes().as_ptr(),
                (self.graph.pin_pos as *mut u8).add(slot * 4),
                4,
            );
        }
        let posb =
            unsafe { std::slice::from_raw_parts(self.graph.pin_pos as *const u8, self.n_slots * 4) };
        self.cc.h2d_async(
            self.attn.dpp + (slot as u64) * 4,
            &posb[slot * 4..slot * 4 + 4],
        )?;
        self.cc.graph_launch(exec)?;
        self.cc.sync()?;
        let ob = unsafe { std::slice::from_raw_parts(self.graph.pin_out as *const u8, 4) };
        self.slot_pos[slot] = pos + 1;
        Ok(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]))
    }

    pub fn forward_device_argmax(&mut self, slot: usize, embed_row: &[f32]) -> Result<u32, String> {
        if self.head_w == 0 || self.argmax_out == 0 {
            return Err("forward_device_argmax: head/argmax 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, true, true)
                .and_then(|exec| self.graph_replay_argmax(exec, slot, embed_row))
            {
                Ok(t) => return Ok(t),
                Err(e) => {
                    eprintln!("[graph] argmax 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        self.head_gemv_launch(xn, self.head_out)?;
        let fa = self.cc.function("w4a16_argmax_min")?;
        let (mut p_l, mut p_nn, mut p_a) = (self.head_out, self.head_n as i32, self.argmax_out);
        self.cc.launch(
            fa,
            1,
            1,
            1024,
            &mut crate::rawcuda::args::l3(&mut p_l, &mut p_nn, &mut p_a),
        )?;
        let mut ob = [0u8; 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.argmax_out, 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(u32::from_le_bytes(ob))
    }

    pub fn forward_device_argmax_batch(
        &mut self,
        slots: &[usize],
        rows: &[f32],
    ) -> Result<Vec<u32>, String> {
        let t = slots.len();
        if !(2..=BATCH_DEC_MAX).contains(&t) {
            return Err("batch: t 2..=BATCH_DEC_MAX 전용".into());
        }
        let _g = self.cc.guard()?;
        if rows.len() != t * self.hidden
            || (self.moe.n_experts > 0 && !self.moe.moe_resident)
            || self.head_w == 0
            || slots.iter().any(|&s| s >= self.n_slots)
        {
            return Err(format!("batch: 전제 위반 t={t}"));
        }
        let cap = self.attn.attn.ok_or("attn: 형상 미등록")?.cap;
        for &s in slots {
            if self.slot_pos[s] as usize + 1 > cap {
                return Err(format!("batch: slot{s} 컨텍스트 초과"));
            }
        }
        self.ensure_batch_bufs()?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                rows.as_ptr() as *const u8,
                self.graph.pin_batch_in as *mut u8,
                t * self.hidden * 4,
            );
            let pp = self.graph.pin_batch_pos as *mut u8;
            for s in 0..self.n_slots {
                let pos = self.slot_pos[s];
                std::ptr::copy_nonoverlapping(pos.to_le_bytes().as_ptr(), pp.add(s * 4), 4);
            }
        }
        if let (Some(gd), Some(ad)) = (self.gdn.gdn, self.attn.attn) {
            let ring_stride = (gd.n_gdn * 3 * gd.conv_ch()) as u64;
            let blk_stride = (ad.n_attn * ad.cap.div_ceil(512) * 4) as u64;
            unsafe {
                for (k, &s) in slots.iter().enumerate() {
                    let blk = self.attn.dblk_tab + s as u64 * blk_stride * 8;
                    let pp = self.attn.dpp + s as u64 * 4;
                    let ring = self.gdn.dring + s as u64 * ring_stride * 4;
                    let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&gd, s);
                    let st = self.gdn.dgst + st_slot as u64 * 2;
                    let sc = self.gdn.dsc + sc_slot as u64 * 4;
                    for (pin, v) in [
                        (self.graph.pin_row_blk, blk),
                        (self.graph.pin_row_pp, pp),
                        (self.graph.pin_row_ring, ring),
                        (self.graph.pin_row_st, st),
                        (self.graph.pin_row_sc, sc),
                    ] {
                        std::ptr::copy_nonoverlapping(
                            v.to_le_bytes().as_ptr(),
                            (pin as *mut u8).add(k * 8),
                            8,
                        );
                    }
                }
            }
        }
        let key: (usize, Vec<usize>) = (t, slots.to_vec());
        let hit = self
            .graph
            .batch_graphs
            .iter()
            .position(|(ks, _, _)| *ks == key);
        if let Some(k) = hit {
            for &s in slots {
                self.kv_ensure(s, self.slot_pos[s] as usize + 3)?;
            }
            let exec = self.graph.batch_graphs[k].1;
            self.cc.graph_launch(exec)?;
            self.cc.sync()?;
            let out = self.batch_read_tokens(t);
            for &s in slots {
                self.slot_pos[s] += 1;
            }
            return Ok(out);
        }
        self.batch_launch(slots, t)?;
        self.cc.sync()?;
        let out = self.batch_read_tokens(t);
        if !self.graph.batch_capture_failed {
            match self.batch_capture(&key, t) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# batch 그래프 캡처 실패 — 직접 발사 유지: {e}");
                    self.graph.batch_capture_failed = true;
                    self.cc.sync().ok();
                }
            }
        }
        for &s in slots {
            self.slot_pos[s] += 1;
        }
        Ok(out)
    }

    pub(super) fn batch_capture(
        &mut self,
        key: &(usize, Vec<usize>),
        t: usize,
    ) -> Result<(), String> {
        let _tt = std::time::Instant::now();
        if self.graph.batch_graphs.len() >= 8 {
            let (_, e, g) = self.graph.batch_graphs.remove(0);
            let _ = self.cc.graph_destroy(e, g);
        }
        self.cc.capture_begin()?;
        if let Err(e) = self.batch_launch(&key.1, t) {
            let _ = self.cc.capture_end();
            return Err(e);
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?;
        self.graph.batch_graphs.push((key.clone(), e, g));
        eprintln!(
            "[batch-graph] captured t={} slots={:?} ({:.2}ms)",
            key.0,
            key.1,
            _tt.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    pub fn forward_prefill(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        if t == 1 {
            let row = &rows[..self.hidden];
            return if head {
                self.forward_device_head(slot, row)
            } else {
                self.forward_device(slot, row)
            };
        }
        self.chain_device_t(slot, rows, t, head)
    }

    pub fn forward_device(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, false, false)
                .and_then(|exec| self.graph_replay(exec, slot, embed_row))
            {
                Ok(xn) => {
                    llm170_diag::fp::fp_record("gpu.xn", &xn);
                    return Ok(xn);
                }
                Err(e) => {
                    eprintln!("[graph] xn 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn_final = self.chain_device(slot, embed_row)?;
        let mut ob = vec![0u8; self.hidden * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), xn_final, self.hidden * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        let xn =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, self.hidden) }.to_vec();
        llm170_diag::fp::fp_record("gpu.xn", &xn);
        Ok(xn)
    }

    pub fn forward_prefill_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        if !(2..=GEMM_FFMA_TMAX).contains(&t) {
            return Err(format!("prefill_logits: t={t} — 2..={GEMM_FFMA_TMAX} 전용"));
        }
        if self.head_w == 0 {
            return Err("prefill_logits: head 미등록 — upload_head 선행".into());
        }
        self.chain_t(slot, rows, t, true, true, true, false)
    }

    pub fn forward_device_head(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("forward_device_head: head 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, true, false)
                .and_then(|exec| self.graph_replay(exec, slot, embed_row))
            {
                Ok(lg) => {
                    llm170_diag::fp::fp_record("gpu.logits", &lg);
                    return Ok(lg);
                }
                Err(e) => {
                    eprintln!("[graph] head 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        self.head_gemv_launch(xn, self.head_out)?;
        let mut ob = vec![0u8; self.head_n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        let lg: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        llm170_diag::fp::fp_record("gpu.logits", &lg);
        Ok(lg)
    }
}
