use super::*;

impl W4a16Dec {
    pub fn forward(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "{CTX_OVERFLOW}: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        self.kv_ensure(slot, pos as usize + 3)?;
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.gdn.is_none() {
            return Err("forward: 디코더 상수 미등록".into());
        }
        let dres = self.cc.alloc(self.hidden * 4)?;
        let result = (|| {
            let row = unsafe {
                std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4)
            };
            self.cc.h2d(dres, row)?;
            self.forward_resident(slot, pos, dres)
        })();
        let freed = self.cc.free(dres);
        match (result, freed) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }

    pub(super) fn forward_resident(
        &mut self,
        slot: usize,
        pos: u32,
        dres: CUdeviceptr,
    ) -> Result<Vec<f32>, String> {
        let mut ab = vec![0.0f32; self.hidden];
        let interval = self.attn.attn.map(|a| a.interval).unwrap_or(4);
        let win = self.window;
        let mut gi = win.gdn_start(interval);
        for off in 0..win.n() {
            let il = win.abs(off);
            let xn = self
                .norm_resid_staged(2 * il, dres, &ab)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            if il == 0 && self.debug_layers {
                eprintln!("  G0 xn[0..4]={:?}", &xn[..4]);
            }
            ab = if (il + 1).is_multiple_of(interval) {
                let ai = il / interval;
                let q = self
                    .gemv_host(&format!("blk.{il}.attn_q.weight"), &xn)
                    .map_err(|e| format!("L{il} q: {e}"))?;
                let k = self
                    .gemv_host(&format!("blk.{il}.attn_k.weight"), &xn)
                    .map_err(|e| format!("L{il} k: {e}"))?;
                let v = self
                    .gemv_host(&format!("blk.{il}.attn_v.weight"), &xn)
                    .map_err(|e| format!("L{il} v: {e}"))?;
                let out = self
                    .attn_chain_host(slot, ai, 1, &q, &k, &v, pos)
                    .map_err(|e| format!("L{il} attn: {e}"))?;
                self.gemv_host(&format!("blk.{il}.attn_output.weight"), &out)
                    .map_err(|e| format!("L{il} o: {e}"))?
            } else {
                let qkv = self
                    .gemv_host(&format!("blk.{il}.attn_qkv.weight"), &xn)
                    .map_err(|e| format!("L{il} qkv: {e}"))?;
                let z = self
                    .gemv_host(&format!("blk.{il}.attn_gate.weight"), &xn)
                    .map_err(|e| format!("L{il} z: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 qkv[0..4]={:?}", &qkv[..4]);
                    eprintln!("  G0 z[0..4]={:?}", &z[..4]);
                }
                let gated = self
                    .gdn_chain_host(slot, gi, 1, &xn, &qkv, &z)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 gated[0..4]={:?}", &gated[..4]);
                }
                let o = self
                    .gemv_host(&format!("blk.{il}.ssm_out.weight"), &gated)
                    .map_err(|e| format!("L{il} ssm_out: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 out[0..4]={:?}", &o[..4]);
                }
                o
            };
            let xn = self
                .norm_resid_staged(2 * il + 1, dres, &ab)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            let gate = self
                .gemv_host(&format!("blk.{il}.ffn_gate.weight"), &xn)
                .map_err(|e| format!("L{il} gate: {e}"))?;
            let up = self
                .gemv_host(&format!("blk.{il}.ffn_up.weight"), &xn)
                .map_err(|e| format!("L{il} up: {e}"))?;
            let act = self
                .ew_host(&gate, &up)
                .map_err(|e| format!("L{il} ew: {e}"))?;
            ab = self
                .gemv_host(&format!("blk.{il}.ffn_down.weight"), &act)
                .map_err(|e| format!("L{il} down: {e}"))?;
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, dres)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = ab.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "  G{il:>2} recr={} sum={:.6}",
                    !(il + 1).is_multiple_of(interval),
                    d + a
                );
            }
        }
        let xn = if win.owns_final(self.n_layers) {
            self.norm_resid_staged(2 * self.n_layers, dres, &ab)
                .map_err(|e| format!("final norm: {e}"))?
        } else {
            ab
        };
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(xn)
    }

    pub fn reset_state(&mut self, slot: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if slot >= self.n_slots {
            return Err(format!("reset: slot={slot} >= {}", self.n_slots));
        }
        self.slot_pos[slot] = 0;
        self.attn_set_pos(slot, 0)?;
        if let Some(dims) = self.gdn.gdn {
            let ring_bytes = dims.n_gdn * 3 * dims.conv_ch() * 4;
            let st_bytes = dims.n_gdn * dims.h_v * 128 * 128 * 2;
            let sc_n = dims.n_gdn * dims.h_v * 128;
            let ring_off = (slot * dims.n_gdn * 3 * dims.conv_ch()) as u64 * 4;
            let (st_slot, sc_slot) = super::gdn::gdn_state_offsets(&dims, slot);
            let st_off = st_slot as u64 * 2;
            let sc_off = sc_slot as u64 * 4;
            Self::zero_dev(&self.cc, self.gdn.dring + ring_off, ring_bytes)?;
            Self::zero_dev(&self.cc, self.gdn.dgst + st_off, st_bytes)?;
            self.cc
                .d2d(self.gdn.dsc + sc_off, self.gdn.dsc_ones, sc_n * 4)?;
        }
        Ok(())
    }


    pub(super) fn chain_device(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward_device: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "{CTX_OVERFLOW}: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        self.kv_ensure(slot, pos as usize + 3)?;
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.gdn.is_none() {
            return Err("forward_device: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        let row = if self.graph.capture_pinned_src {
            unsafe {
                std::slice::from_raw_parts(self.graph.pin_embed as *const u8, self.hidden * 4)
            }
        } else {
            unsafe { std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4) }
        };
        self.cc.h2d_async(self.dres, row)?;
        self.chain_device_body(slot)
    }

    pub(super) fn chain_device_body(&mut self, slot: usize) -> Result<CUdeviceptr, String> {
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let mut ab = self.dab_dev;
        let interval = self.attn.attn.ok_or("attn: 형상 미등록")?.interval;
        let win = self.window;
        let mut gi = win.gdn_start(interval);
        let plain = self.plain_weights;
        for off in 0..win.n() {
            let il = win.abs(off);
            let x32 = if plain {
                0
            } else {
                self.ensure_dx32(self.hidden)?
            };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, 1, x32)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            let branch = if (il + 1).is_multiple_of(interval) {
                let xin = if plain { xn } else { x32 };
                if !plain && self.sk_multi_dev(il, 0, self.dx16.ptr)? {
                } else {
                    self.gemv_multi_dev(plain, il, 0, xin)
                        .map_err(|e| format!("L{il} qkv: {e}"))?;
                }
                self.attn_chain_dev_run(slot, il / interval, 1, s0, s1, s1b)
                    .map_err(|e| format!("L{il} attn: {e}"))?
            } else {
                let xin = if plain { xn } else { x32 };
                if !plain && self.sk_multi_dev(il, 0, self.dx16.ptr)? {
                } else {
                    self.gemv_multi_dev(plain, il, 0, xin)
                        .map_err(|e| format!("L{il} qkvz: {e}"))?;
                }
                let g = self
                    .gdn_chain_dev_run(slot, gi, 1, xn, s0, s1)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1).is_multiple_of(interval) {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            let out = if plain {
                self.plain_gemv_dev(&lo, branch)
                    .map_err(|e| format!("L{il} {lo}: {e}"))?
            } else {
                let (_, _, n, ko) = self.lin_spec(&lo)?;
                if self.dx32.cap < ko || self.dx16.cap < ko {
                    return Err(format!("L{il} branch 융합: dx32/dx16 부족 {ko}"));
                }
                let dy = self.ensure_dy(n)?;
                if self.sk_single_dev(&lo, self.dx16.ptr, dy)? {
                    dy
                } else {
                    self.gemv_dev_x32(&lo, self.dx32.ptr)
                        .map_err(|e| format!("L{il} {lo}: {e}"))?
                }
            };
            let x32 = self.ensure_dx32(self.hidden)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, out, 1, x32)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev(il, xn2)
                    .map_err(|e| format!("L{il} moe: {e}"))?;
            } else {
                let x32n = x32;
                let _ = xn2;
                if !self.sk_multi_dev(il, 1, self.dx16.ptr)? {
                    self.gemv_multi_dev(false, il, 1, x32n)
                        .map_err(|e| format!("L{il} gateup: {e}"))?;
                }
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                self.ew_dev_fused(s0, s1, s2, kd)
                    .map_err(|e| format!("L{il} ew: {e}"))?;
                if !self.sk_single_dev(&dn, self.dx16.ptr, s3)? {
                    self.gemv_launch(&dn, self.dx32.ptr, s3)
                        .map_err(|e| format!("L{il} down: {e}"))?;
                }
                ab = s3;
            }
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                let mut abv = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, self.dres)?;
                self.cc.d2h(&mut abv, ab)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = abv
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                eprintln!(
                    "  G{il:>2} recr={} sum={:.6}",
                    !(il + 1).is_multiple_of(4),
                    d + a
                );
            }
        }
        let xn_final = if win.owns_final(self.n_layers) {
            self.norm_resid_dev(2 * self.n_layers, self.dres, ab, 1, 0)
                .map_err(|e| format!("final norm: {e}"))?
        } else {
            self.axpy_dev(1.0, ab, self.dres, self.hidden)?;
            self.dres
        };
        Ok(xn_final)
    }

    pub(super) fn batch_launch(&mut self, slots: &[usize], t: usize) -> Result<(), String> {
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.gdn.is_none() {
            return Err("batch_launch: 디코더 상수 미등록".into());
        }
        for &sl in slots {
            let p = self.slot_pos.get(sl).copied().unwrap_or(0) as usize;
            self.kv_ensure(sl, p + t + 2)?;
        }
        self.ensure_chain_bufs()?;
        self.ensure_norm_bufs(t)?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?;
        let dyt = self.ensure_dyt(t)?;
        {
            let rb = unsafe {
                std::slice::from_raw_parts(
                    self.graph.pin_batch_in as *const u8,
                    t * self.hidden * 4,
                )
            };
            self.cc.h2d_async(self.dres, rb)?;
            let pb = unsafe {
                std::slice::from_raw_parts(self.graph.pin_batch_pos as *const u8, self.n_slots * 4)
            };
            self.cc.h2d_async(self.attn.dpp, pb)?;
            for (pin, dev) in [
                (self.graph.pin_row_blk, self.drow_blk),
                (self.graph.pin_row_pp, self.drow_pp),
                (self.graph.pin_row_ring, self.drow_ring),
                (self.graph.pin_row_st, self.drow_st),
                (self.graph.pin_row_sc, self.drow_sc),
            ] {
                let b = unsafe { std::slice::from_raw_parts(pin as *const u8, t * 8) };
                self.cc.h2d_async(dev, b)?;
            }
        }
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let h = self.hidden;
        let mut ab = self.dab_dev;
        let interval = self.attn.attn.ok_or("attn: 형상 미등록")?.interval;
        let win = self.window;
        if !win.owns_final(self.n_layers) {
            return Err("batch_launch: 비최종 창 — head/argmax는 마지막 샤드 소관".into());
        }
        let mut gi = win.gdn_start(interval);
        let plain = self.plain_weights;
        if t > 32 {
            return Err(format!("chain_device_batch: t={t} > 32 FFMA 상한"));
        }
        for off in 0..win.n() {
            let il = win.abs(off);
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("B{il} input norm: {e}"))?;
            let branch = if (il + 1).is_multiple_of(interval) {
                let xin = if plain { xn } else { xh };
                self.gemv_multi_t_dev(plain, il, 0, xin, t)
                    .map_err(|e| format!("B{il} qkv: {e}"))?;
                self.attn_chain_dev_batch(slots, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("B{il} attn: {e}"))?
            } else {
                let xin = if plain { xn } else { xh };
                self.gemv_multi_t_dev(plain, il, 0, xin, t)
                    .map_err(|e| format!("B{il} qkvz: {e}"))?;
                let g = self
                    .gdn_chain_dev_batch(slots, gi, t, xn, s0, s1)
                    .map_err(|e| format!("B{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1).is_multiple_of(interval) {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            if plain {
                self.lin_forward(plain, LinPath::GemvT, &lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                if self.dx32.cap < t * ko || self.dx16.cap < t * ko {
                    return Err(format!("B{il} branch 융합: dx32/dx16 부족 {}", t * ko));
                }
                self.lin_forward(plain, LinPath::GemvT, &lo, self.dx32.ptr, dyt, t)?;
            }
            let xh = self.ensure_dx32(t * h)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, dyt, t, xh)
                .map_err(|e| format!("B{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev_t(il, xn2, t)
                    .map_err(|e| format!("B{il} moe: {e}"))?;
            } else {
                let xh3 = xh;
                let _ = xn2;
                self.gemv_multi_t_dev(false, il, 1, xh3, t)
                    .map_err(|e| format!("B{il} gateup: {e}"))?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                self.ew_dev_fused(s0, s1, s2, t * kd)?;
                self.lin_forward(false, LinPath::GemvT, &dn, self.dx32.ptr, s3, t)?;
                ab = s3;
            }
        }
        let xn_all = self
            .norm_resid_dev(2 * self.n_layers, self.dres, ab, t, 0)
            .map_err(|e| format!("B final norm: {e}"))?;
        self.head_gemv_t_launch(xn_all, self.dbatch_lg, t)?;
        {
            let fa = self.cc.function("w4a16_argmax_min_t")?;
            let (mut p_l, mut p_n, mut p_t, mut p_a) =
                (self.dbatch_lg, self.head_n as i32, t as i32, self.dbatch_am);
            self.cc.launch(
                fa,
                t as u32,
                1,
                1024,
                &mut crate::rawcuda::args::l4(&mut p_l, &mut p_n, &mut p_t, &mut p_a),
            )?;
        }
        let tb =
            unsafe { std::slice::from_raw_parts_mut(self.graph.pin_batch_tok as *mut u8, t * 4) };
        self.cc.d2h_async(tb.as_mut_ptr(), self.dbatch_am, t * 4)?;
        Ok(())
    }

    pub(super) fn batch_read_tokens(&self, t: usize) -> Vec<u32> {
        let tb =
            unsafe { std::slice::from_raw_parts(self.graph.pin_batch_tok as *const u8, t * 4) };
        (0..t)
            .map(|k| u32::from_le_bytes([tb[k * 4], tb[k * 4 + 1], tb[k * 4 + 2], tb[k * 4 + 3]]))
            .collect()
    }

    pub fn batch_read_logits(&mut self, t: usize) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if t == 0 || t > BATCH_DEC_MAX || self.dbatch_lg == 0 {
            return Err(format!("batch_read_logits: 계약 위반 t={t}"));
        }
        self.ensure_batch_bufs()?;
        let bytes = t * self.head_n * 4;
        self.cc
            .d2h_async(self.graph.pin_batch_lg as *mut u8, self.dbatch_lg, bytes)?;
        self.cc.sync()?;
        let pb = unsafe { std::slice::from_raw_parts(self.graph.pin_batch_lg as *const u8, bytes) };
        Ok(pb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}
