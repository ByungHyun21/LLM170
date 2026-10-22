
use super::*;

impl W4a16Dec {
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        let _g = self.cc.guard()?;
        if self.head_w == 0 {
            return Err("spec_verify: head 미등록".into());
        }
        if !self.spec_on {
            return Err("spec_verify: enable_spec 미호출".into());
        }
        if !(2..=8).contains(&t) {
            return Err(format!("spec_verify: t={t} 계약(2..=8)"));
        }
        {
            let pos0 = self.slot_pos[slot] as usize;
            self.kv_ensure(slot, pos0 + t + 2)?;
        }
        if self.graph.spec_graph_ok
            && !self.graph.pin_batch_in.is_null()
            && !self.graph.pin_pos.is_null()
            && self.n_slots > 0
        {
            let h = self.hidden;
            if rows.len() != t * h {
                return Err(format!("spec_verify: rows 계약 {}", rows.len()));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rows.as_ptr() as *const u8,
                    self.graph.pin_batch_in as *mut u8,
                    t * h * 4,
                );
            }
            let pos = self.slot_pos[slot];
            unsafe {
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
            if let Some(k) = self
                .graph
                .spec_graphs
                .iter()
                .position(|(key, _, _)| *key == (slot, t))
            {
                let exec = self.graph.spec_graphs[k].1;
                self.cc.graph_launch(exec)?;
                self.cc.sync()?;
                let toks = self.batch_read_tokens(t);
                self.slot_pos[slot] = pos + t as u32;
                return Ok(toks);
            }
            let pin =
                unsafe { std::slice::from_raw_parts(self.graph.pin_batch_in as *const f32, t * h) };
            let v = self.chain_t(slot, pin, t, false, true, false, false)?;
            let toks: Vec<u32> = v.into_iter().map(|x| x as u32).collect();
            if self.graph.spec_graphs.len() >= 8 {
                let (_, e, g) = self.graph.spec_graphs.remove(0);
                let _ = self.cc.graph_destroy(e, g);
            }
            let cap = (|| -> Result<(ffi::CUgraphExec, ffi::CUgraph), String> {
                self.cc.capture_begin()?;
                self.graph.capture_pinned_src = true;
                self.graph.capture_no_host = true;
                let r = self.chain_t(slot, pin, t, false, true, false, false);
                self.graph.capture_no_host = false;
                self.graph.capture_pinned_src = false;
                if let Err(e) = r {
                    let _ = self.cc.capture_end();
                    return Err(e);
                }
                let g = self.cc.capture_end()?;
                let e = self.cc.graph_instantiate(g)?;
                self.cc.graph_upload(e)?;
                Ok((e, g))
            })();
            match cap {
                Ok((e, g)) => self.graph.spec_graphs.push(((slot, t), e, g)),
                Err(err) => {
                    eprintln!("[spec-capture 실패] slot={slot} t={t}: {err}");
                    self.graph.spec_graph_ok = false;
                    self.cc.sync().ok();
                }
            }
            return Ok(toks);
        }
        let v = self.chain_t(slot, rows, t, false, true, false, false)?;
        Ok(v.into_iter().map(|x| x as u32).collect())
    }

    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if self.head_w == 0 {
            return Err("spec_verify_logits: head 미등록".into());
        }
        if !self.spec_on {
            return Err("spec_verify_logits: enable_spec 미호출".into());
        }
        if !(2..=8).contains(&t) {
            return Err(format!("spec_verify_logits: t={t} 계약(2..=8)"));
        }
        self.chain_t(slot, rows, t, false, true, true, false)
    }

    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if !self.spec_on || keep > 8 {
            return Err(format!("spec_rollback: 계약 위반 keep={keep}"));
        }
        if self.gdn.dwy_s0 == 0 {
            return Err("spec_rollback: WY 버퍼 미확보(enable_spec)".into());
        }
        if keep == 0 && self.gdn.dring0 == 0 {
            return Err("spec_rollback: 링 스냅샷 미확보(enable_spec)".into());
        }
        let _g = self.cc.guard()?;
        let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&dm, slot);
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let ring_elems = (3 * dm.conv_ch()) as u64;
        let f = self.cc.function("gdn_wy_reconstruct")?;
        for gi in 0..dm.n_gdn {
            let (mut w0, mut w1, mut w2, mut w3) = (
                self.gdn.dwy_k,
                self.gdn.dwy_s0,
                self.gdn.dwy_dc,
                self.gdn.dwy_g,
            );
            let mut w4 = self.gdn.dgst + (st_slot as u64) * 2;
            let mut w5 = self.gdn.dsc + (sc_slot as u64) * 4;
            let (mut wm, mut hk, mut hv, mut dd, mut lay) = (
                keep as i32,
                dm.h_k as i32,
                dm.h_v as i32,
                dm.d as i32,
                gi as i32,
            );
            self.cc.launch(
                f,
                dm.h_v as u32,
                1,
                128,
                &mut crate::rawcuda::args::l11(
                    &mut w0, &mut w1, &mut w2, &mut w3, &mut w4, &mut w5, &mut wm, &mut hk,
                    &mut hv, &mut dd, &mut lay,
                ),
            )?;
            self.gdn_rescale_slot(slot, gi)?;
            let rsrc = if keep == 0 {
                self.gdn.dring0 + gi as u64 * ring_elems * 4
            } else {
                self.gdn.dsnap_ring + (gi as u64 * 8 + (keep - 1) as u64) * ring_elems * 4
            };
            let rdst = self.gdn.dring + (ring_slot as u64 + gi as u64 * ring_elems) * 4;
            self.copy_dev(rdst, rsrc, ring_elems * 4)?;
        }
        Ok(())
    }

    pub(super) fn copy_dev(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: u64,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let f = self.cc.function("w4a16_copy")?;
        let n4 = (bytes / 16) as i64;
        let (mut p_s, mut p_d, mut p_n) = (src, dst, n4);
        self.cc.launch(
            f,
            ((n4 as u64).div_ceil(256)) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut p_s, &mut p_d, &mut p_n),
        )
    }

    pub fn spec_set_scan(&mut self, on: bool) {
        self.spec_scan_on = on;
    }

    pub fn spec_hv(&self) -> usize {
        self.gdn.gdn.map(|d| d.h_v).unwrap_or(0)
    }

    pub fn spec_dump_outv(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let mut b = vec![0u8; n * 4];
        self.cc.d2h(&mut b, self.gdn.dgo)?;
        self.cc.sync()?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub fn spec_dump_state(&mut self, slot: usize) -> Result<Vec<f32>, String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let (st_slot, _) = super::gdn::gdn_state_offsets(&dm, slot);
        let bytes = dm.n_gdn * dm.h_v * dm.d * dm.d * 2;
        let mut b = vec![0u8; bytes];
        self.cc.d2h(&mut b, self.gdn.dgst + st_slot as u64 * 2)?;
        self.cc.sync()?;
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::rawcuda::gptq4::h2f(u16::from_le_bytes(*c)))
            .collect())
    }

    pub fn checkpoint_gdn(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.gdn.dchk.is_null() || slot >= self.gdn.dchk_pos.len() || !pos.is_multiple_of(512) {
            return Ok(());
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let k = Self::ckpt_victim(&self.gdn.dchk_saved[slot], &self.gdn.dchk_use[slot]);
        self.gdn.dchk_tick += 1;
        let tick = self.gdn.dchk_tick;
        let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&dm, slot);
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 2) as u64;
        let sc_bytes = (dm.n_gdn * dm.h_v * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        let per = st_bytes + sc_bytes + ring_bytes;
        let off = ((slot as u64 * 4 + k as u64) * per) as isize;
        unsafe {
            let base = self.gdn.dchk as *mut u8;
            self.cc.d2h_async(
                base.offset(off),
                self.gdn.dgst + st_slot as u64 * 2,
                st_bytes as usize,
            )?;
            self.cc.d2h_async(
                base.offset(off + st_bytes as isize),
                self.gdn.dsc + sc_slot as u64 * 4,
                sc_bytes as usize,
            )?;
            self.cc.d2h_async(
                base.offset(off + (st_bytes + sc_bytes) as isize),
                self.gdn.dring + ring_slot * 4,
                ring_bytes as usize,
            )?;
        }
        self.gdn.dchk_pos[slot][k] = pos;
        self.gdn.dchk_use[slot][k] = tick;
        self.gdn.dchk_saved[slot][k] = 0;
        self.ucache.touch("spec:ckpt")?;
        Ok(())
    }

    pub(super) fn ckpt_victim(saved: &[u32; 4], use_tick: &[u64; 4]) -> usize {
        let mut best = 0usize;
        for s in 1..4 {
            if (saved[s], use_tick[s]) < (saved[best], use_tick[best]) {
                best = s;
            }
        }
        best
    }

    fn ck_pick(&self, slot: usize, upto: u32) -> u32 {
        let mut best = 0u32;
        if let Some(ks) = self.gdn.dchk_pos.get(slot) {
            for &p in ks {
                if p > 0 && p <= upto && p > best {
                    best = p;
                }
            }
        }
        best
    }

    pub fn restore_gdn(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        let pos = self.ck_pick(slot, upto);
        if self.gdn.dchk.is_null() || pos == 0 {
            return Ok(0);
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let Some(k) = self.gdn.dchk_pos[slot].iter().position(|&p| p == pos) else {
            return Ok(0);
        };
        let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&dm, slot);
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 2) as u64;
        let sc_bytes = (dm.n_gdn * dm.h_v * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        let per = st_bytes + sc_bytes + ring_bytes;
        let off = ((slot as u64 * 4 + k as u64) * per) as isize;
        let total = (st_bytes + sc_bytes + ring_bytes) as usize;
        let host =
            unsafe { std::slice::from_raw_parts((self.gdn.dchk as *const u8).offset(off), total) };
        self.cc.h2d_async(
            self.gdn.dgst + st_slot as u64 * 2,
            &host[..st_bytes as usize],
        )?;
        self.cc.h2d_async(
            self.gdn.dsc + sc_slot as u64 * 4,
            &host[st_bytes as usize..(st_bytes + sc_bytes) as usize],
        )?;
        self.cc.h2d_async(
            self.gdn.dring + ring_slot * 4,
            &host[(st_bytes + sc_bytes) as usize..],
        )?;
        self.slot_pos[slot] = pos;
        self.attn_set_pos(slot, pos)?;
        self.gdn.dchk_saved[slot][k] =
            self.gdn.dchk_saved[slot][k].saturating_add(upto.saturating_sub(pos));
        self.gdn.dchk_tick += 1;
        self.gdn.dchk_use[slot][k] = self.gdn.dchk_tick;
        self.ucache.touch("spec:ckpt")?;
        Ok(pos)
    }

    pub fn checkpoint_pos(&self, slot: usize, upto: u32) -> u32 {
        self.ck_pick(slot, upto)
    }

    pub fn spec_save_state(&mut self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if self.gdn.dsave == 0 {
            let per = dm.n_gdn * dm.h_v * dm.d * dm.d * 2
                + dm.n_gdn * dm.h_v * dm.d * 4
                + dm.n_gdn * 3 * dm.conv_ch() * 4;
            self.gdn.dsave = self.cc.alloc(self.n_slots * per)?;
        }
        let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&dm, slot);
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 2) as u64;
        let sc_bytes = (dm.n_gdn * dm.h_v * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        self.copy_dev(self.gdn.dsave, self.gdn.dgst + st_slot as u64 * 2, st_bytes)?;
        self.copy_dev(
            self.gdn.dsave + st_bytes,
            self.gdn.dsc + sc_slot as u64 * 4,
            sc_bytes,
        )?;
        self.copy_dev(
            self.gdn.dsave + st_bytes + sc_bytes,
            self.gdn.dring + ring_slot * 4,
            ring_bytes,
        )?;
        self.ucache.touch("spec:state")
    }

    pub fn spec_load_state(&mut self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if self.gdn.dsave == 0 {
            return Err("spec_load_state: 저장본 없음(spec_save_state 미호출)".into());
        }
        let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&dm, slot);
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 2) as u64;
        let sc_bytes = (dm.n_gdn * dm.h_v * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        self.copy_dev(self.gdn.dgst + st_slot as u64 * 2, self.gdn.dsave, st_bytes)?;
        self.copy_dev(
            self.gdn.dsc + sc_slot as u64 * 4,
            self.gdn.dsave + st_bytes,
            sc_bytes,
        )?;
        self.copy_dev(
            self.gdn.dring + ring_slot * 4,
            self.gdn.dsave + st_bytes + sc_bytes,
            ring_bytes,
        )?;
        self.ucache.touch("spec:state")
    }

    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        let p = self.slot_pos[slot].saturating_sub(n);
        self.slot_pos[slot] = p;
        self.attn_set_pos(slot, p)?;
        Ok(())
    }
}
