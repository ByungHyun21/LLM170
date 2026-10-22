
use super::*;

const MTP_TMAX: usize = 64;
const MTP_PF_CHUNK: usize = 64;

impl super::W4a16Dec {
    pub fn mtp_enabled(&self) -> bool {
        self.mtp.mtp_ok
    }

    pub fn slot_pos_of(&self, slot: usize) -> u32 {
        self.slot_pos.get(slot).copied().unwrap_or(0)
    }

    pub fn set_mtp(
        &mut self,
        norms: &[f32],
        q_out: usize,
        kv_dim: usize,
        ffn: usize,
    ) -> Result<(), String> {
        let h = self.hidden;
        let dm = self.attn.attn.ok_or("mtp: attn 미등록")?;
        if norms.len() != 5 * h {
            return Err(format!("mtp: norms != 5x{h}"));
        }
        if dm.n_attn == 0 {
            return Err("mtp: 전체-어텐션 층 없음".into());
        }
        let tmax = MTP_TMAX;
        let (o_emb, o_hid, o_cat) = (0, tmax * h, 2 * tmax * h);
        let o_fc = o_cat + 2 * tmax * h;
        let (o_res, o_x, o_o) = (o_fc + tmax * h, o_fc + 2 * tmax * h, o_fc + 3 * tmax * h);
        let o_qg = o_fc + 4 * tmax * h;
        let o_kin = o_qg + tmax * q_out;
        let o_vin = o_kin + tmax * kv_dim;
        let o_g = o_vin + tmax * kv_dim;
        let o_u = o_g + tmax * ffn;
        let o_act = o_u + tmax * ffn;
        let o_dn = o_act + tmax * ffn;
        let o_lg = o_dn + tmax * h;
        let o_xn_e = o_lg + 8 * self.head_n;
        let o_xn_h = o_xn_e + tmax * h;
        let o_prev = o_xn_h + tmax * h;
        let total = o_prev + self.n_slots * h;
        if self.mtp.mtp_buf != 0 {
            self.cc.free(self.mtp.mtp_buf)?;
            self.mtp.mtp_buf = 0;
        }
        if self.mtp.mtp_nw != 0 {
            self.cc.free(self.mtp.mtp_nw)?;
            self.mtp.mtp_nw = 0;
        }
        if self.mtp.mtp_zero != 0 {
            self.cc.free(self.mtp.mtp_zero)?;
            self.mtp.mtp_zero = 0;
        }
        for g in self.mtp.mtp_chain_graphs.iter_mut() {
            if let Some((e, h)) = g.take() {
                let _ = self.cc.graph_destroy(e, h);
            }
        }
        if self.mtp.mtp_chain_ctr != 0 {
            self.cc.free(self.mtp.mtp_chain_ctr)?;
            self.mtp.mtp_chain_ctr = 0;
        }
        if self.mtp.mtp_chain_log != 0 {
            self.cc.free(self.mtp.mtp_chain_log)?;
            self.mtp.mtp_chain_log = 0;
        }
        let nw = self.cc.alloc(5 * h * 4)?;
        let nb =
            unsafe { std::slice::from_raw_parts(norms.as_ptr() as *const u8, norms.len() * 4) };
        if let Err(e) = Self::h2d_chunked(&self.cc, nw, nb) {
            let _ = self.cc.free(nw);
            return Err(e);
        }
        let zero = self.cc.alloc(tmax * h * 4)?;
        Self::zero_dev(&self.cc, zero, tmax * h * 4)?;
        let buf = self.cc.alloc(total * 4)?;
        let cctr = self.cc.alloc(4)?;
        Self::zero_dev(&self.cc, cctr, 4)?;
        let clog = self.cc.alloc(8 * 4)?;
        Self::zero_dev(&self.cc, clog, 8 * 4)?;
        self.mtp.mtp_nw = nw;
        self.mtp.mtp_zero = zero;
        self.mtp.mtp_buf = buf;
        self.mtp.mtp_off = [
            o_emb, o_hid, o_cat, o_fc, o_res, o_x, o_o, o_qg, o_kin, o_vin, o_g, o_u, o_act, o_dn,
            o_lg, o_xn_e, o_xn_h,
        ];
        self.mtp.mtp_prev_off = o_prev;
        self.mtp.mtp_layer = dm.n_attn - 1;
        self.mtp.mtp_rows = 0;
        self.mtp.mtp_ffn = ffn;
        self.mtp.mtp_draft = vec![Vec::new(); self.n_slots];
        self.mtp.mtp_pos = vec![u32::MAX; self.n_slots];
        self.mtp.mtp_prev_pos = vec![u32::MAX; self.n_slots];
        self.mtp.mtp_off_slot = vec![false; self.n_slots];
        self.mtp.mtp_chain_ctr = cctr;
        self.mtp.mtp_chain_log = clog;
        self.mtp.mtp_chain_graphs = vec![None; self.n_slots];
        self.mtp.mtp_ok = true;
        eprintln!(
            "[mtp] 등록 — q_out {q_out}·kv {kv_dim}·ffn {ffn}·아레나 {:.1}MiB·층 {}",
            (total * 4) as f64 / (1 << 20) as f64,
            self.mtp.mtp_layer
        );
        {
            let dm = self.attn.attn.ok_or("mtp: attn")?;
            let arena = (buf, buf + total as u64 * 4);
            let nwr = (nw, nw + 5 * h as u64 * 4);
            let zr = (zero, zero + h as u64 * 4);
            let dq = self.attn.dqh_a;
            let dqe = dq + (dm.q_dim() * 8) as u64 * 4;
            eprintln!("[mtp-ptr] arena={arena:?} nw={nwr:?} zero={zr:?} dqh={dq}..{dqe}");
        }
        Ok(())
    }

    fn mtp_ptr(&self, idx: usize) -> CUdeviceptr {
        self.mtp.mtp_buf + (self.mtp.mtp_off[idx] as u64) * 4
    }

    pub fn mtp_push_hidden(&mut self, row: usize) -> Result<(), String> {
        if !self.mtp.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        if self.mtp.mtp_rows >= MTP_TMAX {
            return Err("mtp: hidden 행 상한".into());
        }
        let h = self.hidden;
        let src = self.dres + (row * h) as u64 * 4;
        let dst = self.mtp_ptr(1) + (self.mtp.mtp_rows * h) as u64 * 4;
        self.copy_dev(dst, src, (h * 4) as u64)?;
        self.mtp.mtp_rows += 1;
        Ok(())
    }

    pub fn mtp_reset_hidden(&mut self) {
        self.mtp.mtp_rows = 0;
    }

    pub fn mtp_store_drafts(&mut self, slot: usize, toks: &[u32]) {
        if slot < self.mtp.mtp_draft.len() {
            self.mtp.mtp_draft[slot] = toks.to_vec();
            self.mtp.mtp_pos[slot] = self.slot_pos[slot] + 1;
        }
    }

    pub fn mtp_recur_step(
        &mut self,
        slot: usize,
        tok: u32,
        embed: &[f32],
        pos: u32,
    ) -> Result<u32, String> {
        let h = self.hidden;
        if embed.len() != h {
            return Err("mtp: 재귀 임베딩 폭".into());
        }
        self.copy_dev(self.mtp_ptr(1), self.mtp_ptr(5), (h * 4) as u64)?;
        self.mtp.mtp_rows = 1;
        self.mtp_run_h(slot, &[tok], embed, pos, true)
    }

    pub fn mtp_fin_seed(&mut self, last_row: usize) -> Result<(), String> {
        if !self.mtp.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        if last_row == 0 {
            return Ok(());
        }
        if last_row >= MTP_TMAX {
            return Err(format!("mtp fin seed: row={last_row} >= {MTP_TMAX}"));
        }
        let h = self.hidden;
        self.copy_dev(
            self.mtp_ptr(5),
            self.mtp_ptr(5) + (last_row * h) as u64 * 4,
            (h * 4) as u64,
        )
    }

    pub fn mtp_zero_row(&mut self, pos: usize) -> Result<(), String> {
        if !self.mtp.mtp_ok {
            return Ok(());
        }
        let dm = self.attn.attn.ok_or("mtp: attn")?;
        if pos >= dm.cap {
            return Ok(());
        }
        let lay = self.mtp.mtp_layer;
        for slot in 0..self.n_slots {
            let Ok((k, v, ks, vs)) = self.attn_row_ptrs(slot, lay, pos) else {
                continue;
            };
            Self::zero_dev(&self.cc, k, dm.kv_dim())?;
            Self::zero_dev(&self.cc, v, dm.kv_dim())?;
            Self::zero_dev(&self.cc, ks, dm.kv_heads * 4)?;
            Self::zero_dev(&self.cc, vs, dm.kv_heads * 4)?;
        }
        Ok(())
    }

    pub fn mtp_take_draft(&mut self, slot: usize) -> Option<Vec<u32>> {
        if slot >= self.mtp.mtp_draft.len() {
            return None;
        }
        if self.mtp.mtp_pos[slot] != self.slot_pos[slot] + 1 {
            return None;
        }
        if self.mtp.mtp_draft[slot].is_empty() {
            return None;
        }
        Some(self.mtp.mtp_draft[slot].clone())
    }

    pub fn mtp_run(
        &mut self,
        slot: usize,
        tokens: &[u32],
        embeds: &[f32],
        pos_start: u32,
    ) -> Result<u32, String> {
        self.mtp_run_h(slot, tokens, embeds, pos_start, true)
    }

    pub fn mtp_run_h(
        &mut self,
        slot: usize,
        tokens: &[u32],
        embeds: &[f32],
        pos_start: u32,
        head: bool,
    ) -> Result<u32, String> {
        let _g = self.cc.guard()?;
        if !self.mtp.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        let h = self.hidden;
        let t = tokens.len();
        if !(1..=MTP_TMAX).contains(&t) || self.mtp.mtp_rows != t || embeds.len() != t * h {
            return Err(format!(
                "mtp: t={t} rows={} embeds={} 계약 위반",
                self.mtp.mtp_rows,
                embeds.len()
            ));
        }
        let pos_end = pos_start as usize + t;
        let cap = self.attn.attn.ok_or("mtp: attn")?.cap;
        if pos_end > cap {
            return Err(format!("mtp: pos{pos_start}+{t} > cap{cap}"));
        }
        self.kv_ensure(slot, pos_end + 2)?;
        self.ensure_chain_bufs()?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?;
        if self.mtp.mtp_pin_emb.is_null() {
            self.mtp.mtp_pin_emb = self.cc.pinned_alloc(MTP_TMAX * h * 4)?;
            self.mtp.mtp_pin_out = self.cc.pinned_alloc(8 * 4)?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                embeds.as_ptr() as *const u8,
                self.mtp.mtp_pin_emb as *mut u8,
                t * h * 4,
            );
        }
        if head && t <= 8 && !self.mtp.mtp_graph_failed {
            match self.ensure_mtp_graph(slot, t) {
                Ok(exec) => {
                    let save = self.slot_pos[slot];
                    self.slot_pos[slot] = pos_start;
                    self.attn_set_pos(slot, pos_start)?;
                    self.cc.graph_launch(exec)?;
                    self.cc.sync()?;
                    self.slot_pos[slot] = save;
                    let ob = unsafe {
                        std::slice::from_raw_parts(self.mtp.mtp_pin_out as *const u8, t * 4)
                    };
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&ob[(t - 1) * 4..t * 4]);
                    return Ok(u32::from_le_bytes(b));
                }
                Err(e) => {
                    eprintln!("[mtp-graph] 캡처 실패 — 직접 경로: {e}");
                    self.mtp.mtp_graph_failed = true;
                }
            }
        }
        let save = self.slot_pos[slot];
        self.slot_pos[slot] = pos_start;
        let r = self.mtp_fwd_launch(slot, t, head, true);
        self.slot_pos[slot] = save;
        r?;
        if !head {
            return Ok(0);
        }
        self.cc.sync()?;
        let n = if t > 8 { 4 } else { t * 4 };
        let ob = unsafe { std::slice::from_raw_parts(self.mtp.mtp_pin_out as *const u8, n) };
        let mut b = [0u8; 4];
        b.copy_from_slice(&ob[n - 4..n]);
        Ok(u32::from_le_bytes(b))
    }

    fn ensure_mtp_graph(&mut self, slot: usize, t: usize) -> Result<ffi::CUgraphExec, String> {
        if let Some((e, _)) = self.mtp.mtp_graphs[t] {
            return Ok(e);
        }
        self.ensure_chain_bufs()?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?;
        if self.cc.stream.is_null() {
            self.cc.create_stream()?;
        }
        {
            let dm = self.attn.attn.ok_or("mtp: attn")?;
            let nm = t * dm.q_heads * 256;
            self.ensure_dx32(nm)?;
            self.ensure_dx16(nm)?;
        }
        self.cc.capture_begin()?;
        self.graph.capture_pinned_src = true;
        let cap = self.mtp_fwd_launch(slot, t, true, true);
        self.graph.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end();
            return Err(format!("mtp 캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?;
        self.mtp.mtp_graphs[t] = Some((e, g));
        eprintln!("[mtp-graph] captured t={t}");
        Ok(e)
    }

    fn mtp_fwd_launch(
        &mut self,
        slot: usize,
        t: usize,
        head: bool,
        emb_h2d: bool,
    ) -> Result<(), String> {
        let h = self.hidden;
        if emb_h2d {
            let eb =
                unsafe { std::slice::from_raw_parts(self.mtp.mtp_pin_emb as *const u8, t * h * 4) };
            self.cc.h2d_async(self.mtp_ptr(0), eb)?;
        }
        self.mtp_norm(1, self.mtp_ptr(0), self.mtp_ptr(15), t)?;
        self.mtp_norm(0, self.mtp_ptr(1), self.mtp_ptr(16), t)?;
        for i in 0..t {
            let cat = self.mtp_ptr(2) + (i * 2 * h) as u64 * 4;
            let xe = self.mtp_ptr(15) + (i * h) as u64 * 4;
            let xh = self.mtp_ptr(16) + (i * h) as u64 * 4;
            self.copy_dev(cat, xe, (h * 4) as u64)?;
            self.copy_dev(cat + (h * 4) as u64, xh, (h * 4) as u64)?;
        }
        self.mtp_plain_gemv_t("mtp.fc.weight", self.mtp_ptr(2), self.mtp_ptr(3), t)?;
        Self::zero_dev(&self.cc, self.mtp_ptr(4), t * h * 4)?;
        self.mtp_norm_acc(2, self.mtp_ptr(4), self.mtp_ptr(3), self.mtp_ptr(5), t)?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.q_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(7),
            t,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.k_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(8),
            t,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.v_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(9),
            t,
        )?;
        let dout = self.attn_chain_dev_run(
            slot,
            self.mtp.mtp_layer,
            t,
            self.mtp_ptr(7),
            self.mtp_ptr(8),
            self.mtp_ptr(9),
        )?;
        if !head {
            return Ok(());
        }
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.o_proj.weight",
            dout,
            self.mtp_ptr(6),
            t,
        )?;
        self.mtp_norm_acc(3, self.mtp_ptr(4), self.mtp_ptr(6), self.mtp_ptr(5), t)?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.mlp.gate_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(10),
            t,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.mlp.up_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(11),
            t,
        )?;
        self.ew_dev(
            self.mtp_ptr(10),
            self.mtp_ptr(11),
            self.mtp_ptr(12),
            t * self.mtp.mtp_ffn,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.mlp.down_proj.weight",
            self.mtp_ptr(12),
            self.mtp_ptr(13),
            t,
        )?;
        self.mtp_norm_acc(4, self.mtp_ptr(4), self.mtp_ptr(13), self.mtp_ptr(5), t)?;
        if t > 8 {
            let xl = self.mtp_ptr(5) + ((t - 1) * h) as u64 * 4;
            self.head_gemv_launch(xl, self.mtp_ptr(14))?;
            let fa = self.cc.function("w4a16_argmax_min")?;
            let (mut p_l, mut p_nn, mut p_a) =
                (self.mtp_ptr(14), self.head_n as i32, self.dbatch_am);
            self.cc.launch(
                fa,
                1,
                1,
                1024,
                &mut crate::rawcuda::args::l3(&mut p_l, &mut p_nn, &mut p_a),
            )?;
            self.cc
                .d2h_async(self.mtp.mtp_pin_out as *mut u8, self.dbatch_am, 4)?;
            return Ok(());
        }
        self.head_gemv_t_launch(self.mtp_ptr(5), self.mtp_ptr(14), t)?;
        {
            let fa = self.cc.function("w4a16_argmax_min_t")?;
            let (mut p_l, mut p_n, mut p_t, mut p_a) = (
                self.mtp_ptr(14),
                self.head_n as i32,
                t as i32,
                self.dbatch_am,
            );
            self.cc.launch(
                fa,
                t as u32,
                1,
                1024,
                &mut crate::rawcuda::args::l4(&mut p_l, &mut p_n, &mut p_t, &mut p_a),
            )?;
        }
        self.cc
            .d2h_async(self.mtp.mtp_pin_out as *mut u8, self.dbatch_am, t * 4)?;
        Ok(())
    }

    pub fn mtp_prefill_chunk(
        &mut self,
        slot: usize,
        call_tokens: &[u32],
        embeds: &[f32],
        pos_start: u32,
        save_last_row: Option<usize>,
    ) -> Result<(), String> {
        let h = self.hidden;
        if !self.mtp.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        if embeds.len() != call_tokens.len() * h {
            return Err("mtp: embeds 계약".into());
        }
        let mut r = 0usize;
        while r < call_tokens.len() {
            let t = (call_tokens.len() - r).min(MTP_PF_CHUNK);
            self.mtp_reset_hidden();
            for k in 0..t {
                self.mtp_push_hidden(r + k)?;
            }
            let eb = &embeds[r * h..(r + t) * h];
            let ps = pos_start + r as u32;
            self.mtp_run_h(slot, &call_tokens[r..r + t], eb, ps, false)?;
            r += t;
        }
        if let Some(row) = save_last_row {
            self.mtp_save_prev(slot, row)?;
        }
        Ok(())
    }

    pub fn mtp_save_prev(&mut self, slot: usize, row: usize) -> Result<(), String> {
        if !self.mtp.mtp_ok || slot >= self.n_slots {
            return Err("mtp: 미등록/슬롯".into());
        }
        let h = self.hidden;
        let src = self.dres + (row * h) as u64 * 4;
        let dst = self.mtp.mtp_buf + (self.mtp.mtp_prev_off + slot * h) as u64 * 4;
        self.copy_dev(dst, src, (h * 4) as u64)?;
        self.mtp.mtp_prev_pos[slot] = self.slot_pos_of(slot).saturating_sub(1);
        Ok(())
    }

    pub fn mtp_run_prev(&mut self, slot: usize, tok: u32, embed: &[f32]) -> Result<u32, String> {
        if self.mtp.mtp_prev_pos[slot] == u32::MAX {
            return Err("mtp: 유예 호출 없음".into());
        }
        let h = self.hidden;
        self.mtp_reset_hidden();
        let src = self.mtp.mtp_buf + (self.mtp.mtp_prev_off + slot * h) as u64 * 4;
        self.copy_dev(self.mtp_ptr(1), src, (h * 4) as u64)?;
        self.mtp.mtp_rows = 1;
        let pos = self.mtp.mtp_prev_pos[slot];
        let out = self.mtp_run_h(slot, &[tok], embed, pos, true)?;
        self.mtp.mtp_prev_pos[slot] = u32::MAX;
        Ok(out)
    }

    pub fn mtp_kv_reset(&mut self) -> Result<(), String> {
        if !self.mtp.mtp_ok {
            return Ok(());
        }
        let dm = self.attn.attn.ok_or("mtp: attn")?;
        let layer = self.mtp.mtp_layer;
        let mb = dm.cap.div_ceil(512);
        for slot in 0..self.n_slots {
            let pool = self.attn.kv_pool.as_ref().ok_or("mtp: KvPool 미등록")?;
            for bi in 0..mb {
                let Some(_) = pool.get(slot, layer, bi) else {
                    continue;
                };
                let pos = bi * 512;
                let rows = (dm.cap - pos).min(512);
                let (k, v, ks, vs) = self.attn_row_ptrs(slot, layer, pos)?;
                Self::zero_dev(&self.cc, k, rows * dm.kv_dim())?;
                Self::zero_dev(&self.cc, v, rows * dm.kv_dim())?;
                Self::zero_dev(&self.cc, ks, rows * dm.kv_heads * 4)?;
                Self::zero_dev(&self.cc, vs, rows * dm.kv_heads * 4)?;
            }
        }
        Ok(())
    }

    pub fn mtp_slot_off(&mut self, slot: usize) {
        if slot < self.mtp.mtp_off_slot.len() {
            self.mtp.mtp_off_slot[slot] = true;
            self.mtp.mtp_prev_pos[slot] = u32::MAX;
        }
    }

    pub fn mtp_slot_on(&mut self, slot: usize) {
        if slot < self.mtp.mtp_off_slot.len() {
            self.mtp.mtp_off_slot[slot] = false;
        }
    }

    pub fn mtp_slot_enabled(&self, slot: usize) -> bool {
        self.mtp.mtp_ok && slot < self.mtp.mtp_off_slot.len() && !self.mtp.mtp_off_slot[slot]
    }

    fn mtp_norm(
        &mut self,
        row: usize,
        x: CUdeviceptr,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let ab = self.mtp.mtp_zero;
        self.mtp_norm_acc(row, x, ab, xn, t)
    }

    fn mtp_norm_acc(
        &mut self,
        row: usize,
        x: CUdeviceptr,
        ab: CUdeviceptr,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("norm_resid")?;
        let (mut p_x, mut p_w) = (x, self.mtp.mtp_nw);
        let mut p_ab = ab;
        let mut p_xn = xn;
        let (mut p_z1, mut p_z2) = (0u64, 0u64);
        let (mut p_t, mut p_wo, mut p_h) =
            (t as i32, (row * self.hidden) as i32, self.hidden as i32);
        self.cc.launch(
            f,
            t as u32,
            1,
            1024,
            &mut crate::rawcuda::args::l9(
                &mut p_x, &mut p_w, &mut p_ab, &mut p_xn, &mut p_z1, &mut p_z2, &mut p_t,
                &mut p_wo, &mut p_h,
            ),
        )
    }

    fn mtp_plain_gemv_t(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        if t == 1 {
            self.plain_gemv_launch(name, x, y)
        } else if t <= 8 {
            self.plain_gemv_t_launch(name, x, y, t)
        } else {
            self.plain_gemm_launch(name, x, y, t)
        }
    }
}

impl super::W4a16Dec {
    pub fn mtp_chain_drafts(
        &mut self,
        slot: usize,
        k: usize,
        base_pos: u32,
    ) -> Result<Vec<u32>, String> {
        let _g = self.cc.guard()?;
        if !self.mtp.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        if slot >= self.n_slots {
            return Err(format!("mtp chain: 슬롯 {slot} >= {}", self.n_slots));
        }
        if !(2..=8).contains(&k) {
            return Err(format!("mtp chain: k={k} — 2..=8 전용(로그 8칸 계약)"));
        }
        if !self.embed_ready() {
            return Err("mtp chain: 임베딩 미등록(upload_embed 선행)".into());
        }
        if self.head_n == 0 || self.dbatch_am == 0 {
            return Err("mtp chain: head/argmax 미등록".into());
        }
        if self.head_n > self.emb_n {
            return Err(format!(
                "mtp chain: head_n {} > emb_n {} — 초안 토큰 임베딩 행 부재",
                self.head_n, self.emb_n
            ));
        }
        let dm = self.attn.attn.ok_or("mtp: attn")?;
        if base_pos as usize + k > dm.cap {
            return Err(format!(
                "mtp chain: pos{base_pos}+{k} > cap{} (초안 위치 초과)",
                dm.cap
            ));
        }
        self.kv_ensure(slot, base_pos as usize + k + 2)?;
        let exec = self.ensure_mtp_chain_graph(slot)?;
        self.cc
            .h2d_async(self.mtp.mtp_chain_ctr, &0u32.to_le_bytes())?;
        for _ in 0..(k - 1) {
            self.cc.graph_launch(exec)?;
        }
        let mut lb = vec![0u8; (k - 1) * 4];
        self.cc
            .d2h_async(lb.as_mut_ptr(), self.mtp.mtp_chain_log, (k - 1) * 4)?;
        self.cc.sync()?;
        let ob = unsafe { std::slice::from_raw_parts(self.mtp.mtp_pin_out as *const u8, 4) };
        let mut drafts: Vec<u32> = lb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        drafts.push(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]));
        debug_assert_eq!(drafts.len(), k);
        self.attn_set_pos(slot, base_pos)?;
        Ok(drafts)
    }

    pub fn mtp_chain_seed(&mut self, slot: usize, d1: u32, base_pos: u32) -> Result<(), String> {
        if self.dbatch_am == 0 {
            return Err("mtp chain seed: argmax 미등록".into());
        }
        self.cc.h2d_async(self.dbatch_am, &d1.to_le_bytes())?;
        self.attn_set_pos(slot, base_pos)
    }

    fn ensure_mtp_chain_graph(&mut self, slot: usize) -> Result<ffi::CUgraphExec, String> {
        if let Some((e, _)) = self.mtp.mtp_chain_graphs[slot] {
            return Ok(e);
        }
        self.ensure_chain_bufs()?;
        self.ensure_attn_bufs(1)?;
        self.ensure_batch_bufs()?;
        if self.cc.stream.is_null() {
            self.cc.create_stream()?;
        }
        {
            let dm = self.attn.attn.ok_or("mtp: attn")?;
            let nm = dm.q_heads * 256;
            self.ensure_dx32(nm)?;
            self.ensure_dx16(nm)?;
        }
        if self.mtp.mtp_pin_emb.is_null() {
            self.mtp.mtp_pin_emb = self.cc.pinned_alloc(MTP_TMAX * self.hidden * 4)?;
            self.mtp.mtp_pin_out = self.cc.pinned_alloc(8 * 4)?;
        }
        self.cc.capture_begin()?;
        self.graph.capture_pinned_src = true;
        let cap = self.mtp_chain_step_launch(slot);
        self.graph.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end();
            return Err(format!("mtp 체인 캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?;
        self.mtp.mtp_chain_graphs[slot] = Some((e, g));
        eprintln!("[mtp-chain] captured slot={slot}");
        Ok(e)
    }

    pub fn warm_spec_graphs(&mut self, n_slots: usize, eos: u32) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let mut first: Option<String> = None;
        for slot in 0..n_slots {
            if let Err(e) = self.ensure_kstep_graph(slot, eos) {
                first.get_or_insert(e);
            }
        }
        if n_slots >= 2 {
            if let Err(e) = self.ensure_batch_bufs() {
                first.get_or_insert(e);
            } else {
                for t in 2..=n_slots.min(super::BATCH_DEC_MAX) {
                    let slots: Vec<usize> = (0..t).collect();
                    if let Err(e) = self.batch_capture(&(t, slots), t) {
                        first.get_or_insert(e);
                    }
                }
            }
        }
        if self.mtp.mtp_ok {
            for slot in 0..n_slots.min(self.mtp.mtp_chain_graphs.len()) {
                if let Err(e) = self.ensure_mtp_chain_graph(slot) {
                    first.get_or_insert(e);
                }
            }
            for t in 1..=n_slots.min(MTP_TMAX) {
                if let Err(e) = self.ensure_mtp_graph(0, t) {
                    first.get_or_insert(e);
                }
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn mtp_chain_step_launch(&mut self, slot: usize) -> Result<(), String> {
        self.mtp_chain_bookkeeping_launch(slot)?;
        self.embed_gather_launch(self.dbatch_am, self.mtp_ptr(0), 1)?;
        self.copy_dev(self.mtp_ptr(1), self.mtp_ptr(5), (self.hidden * 4) as u64)?;
        self.mtp_fwd_launch(slot, 1, true, false)
    }

    fn mtp_chain_bookkeeping_launch(&mut self, slot: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_mtp_chain_step")?;
        let (mut p_t, mut p_l, mut p_c, mut p_p) = (
            self.dbatch_am,
            self.mtp.mtp_chain_log,
            self.mtp.mtp_chain_ctr,
            self.attn.dpp,
        );
        let (mut p_s, mut p_k) = (slot as i32, 8i32);
        self.cc.launch(
            f,
            1,
            1,
            32,
            &mut crate::rawcuda::args::l6(
                &mut p_t, &mut p_l, &mut p_c, &mut p_p, &mut p_s, &mut p_k,
            ),
        )
    }
}
