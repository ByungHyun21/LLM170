
use super::*;

impl W4a16Dec {
    pub fn decode_k_greedy(
        &mut self,
        slot: usize,
        seed_tok: u32,
        k: usize,
        eos: u32,
    ) -> Result<Vec<u32>, String> {
        let _g = self.cc.guard()?;
        if slot >= self.n_slots {
            return Err(format!("kstep: 슬롯 {slot} >= {}", self.n_slots));
        }
        if !(2..=KSTEP_KMAX).contains(&k) {
            return Err(format!("kstep: k={k} — 2..={KSTEP_KMAX} 전용"));
        }
        {
            let p = self.slot_pos.get(slot).copied().unwrap_or(0) as usize;
            self.kv_ensure(slot, p + k + 2)?;
        }
        if !self.embed_ready() {
            return Err("kstep: 임베딩 미등록(upload_embed 선행)".into());
        }
        if self.head_w == 0 || self.argmax_out == 0 {
            return Err("kstep: head/argmax 미등록".into());
        }
        if self.head_n > self.emb_n {
            return Err(format!(
                "kstep: head_n {} > emb_n {} — 방출 토큰 임베딩 행 부재",
                self.head_n, self.emb_n
            ));
        }
        if seed_tok as usize >= self.emb_n || seed_tok == eos {
            return Err(format!("kstep: 시드 토큰 {seed_tok} 부적합(범위/EOS)"));
        }
        let dm = self.attn.attn.ok_or("kstep: attn 미등록")?;
        let pos = self.slot_pos[slot];
        let k = k.min(dm.cap.saturating_sub(pos as usize));
        if k < 2 {
            return Err(format!("kstep: cap 잔여 부족 pos{pos}/cap{}", dm.cap));
        }
        if !self.graph_ok() || self.kstep.failed {
            return Err("kstep: 그래프 모드 불가(폴백)".into());
        }
        let exec = self.ensure_kstep_graph(slot, eos)?;
        let rst = [0u8; 8];
        self.cc.h2d_async(self.kstep.buf, &rst)?;
        self.cc
            .h2d_async(self.argmax_out, &seed_tok.to_le_bytes())?;
        self.cc
            .h2d_async(self.attn.dpp + (slot as u64) * 4, &pos.to_le_bytes())?;
        for _ in 0..k {
            self.cc.graph_launch(exec)?;
        }
        let mut ob = vec![0u8; 8 + k * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.kstep.buf, 8 + k * 4)?;
        self.cc.sync()?;
        let ctr = u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]) as usize;
        if ctr == 0 || ctr > k {
            return Err(format!(
                "kstep: 카운터 부적합 ctr={ctr}(k={k}) — 컨트롤러 계약 위반"
            ));
        }
        let toks: Vec<u32> = ob[8..8 + ctr * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        self.slot_pos[slot] = pos + ctr as u32;
        self.attn_set_pos(slot, pos + ctr as u32)?;
        Ok(toks)
    }

    pub(super) fn ensure_kstep_graph(
        &mut self,
        slot: usize,
        eos: u32,
    ) -> Result<ffi::CUgraphExec, String> {
        if slot >= self.kstep.graphs.len() {
            self.kstep.graphs.resize(slot + 1, None);
        }
        if let Some((e_eos, e, h)) = self.kstep.graphs[slot].take() {
            if e_eos == eos {
                self.kstep.graphs[slot] = Some((e_eos, e, h));
                return Ok(e);
            }
            let _ = self.cc.graph_destroy(e, h);
        }
        self.warm_for_capture()?;
        if self.cc.stream.is_null() {
            self.cc.create_stream()?;
        }
        if self.kstep.buf == 0 {
            let b = self.cc.alloc(8 + KSTEP_KMAX * 4)?;
            Self::zero_dev(&self.cc, b, 8 + KSTEP_KMAX * 4)?;
            self.kstep.buf = b;
        }
        self.cc.capture_begin()?;
        self.graph.capture_pinned_src = true;
        let cap = self.kstep_launch(slot, eos);
        self.graph.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end();
            self.kstep.failed = true;
            return Err(format!("kstep 캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?;
        self.kstep.graphs[slot] = Some((eos, e, g));
        eprintln!("[kstep] captured slot={slot} eos={eos}");
        Ok(e)
    }

    fn kstep_launch(&mut self, slot: usize, eos: u32) -> Result<(), String> {
        self.embed_gather_launch(self.argmax_out, self.dres, 1)?;
        let xn = self.chain_device_body(slot)?;
        self.head_gemv_launch(xn, self.head_out)?;
        let fa = self.cc.function("w4a16_argmax_min")?;
        let (mut p_l, mut p_n, mut p_o) = (self.head_out, self.head_n as i32, self.argmax_out);
        self.cc.launch(
            fa,
            1,
            1,
            1024,
            &mut crate::rawcuda::args::l3(&mut p_l, &mut p_n, &mut p_o),
        )?;
        let fc = self.cc.function("w4a16_kstep_ctrl")?;
        let (mut p_t, mut p_o, mut p_c) = (self.argmax_out, self.kstep.buf + 8, self.kstep.buf);
        let (mut p_s, mut p_p) = (self.kstep.buf + 4, self.attn.dpp);
        let (mut p_sl, mut p_e) = (slot as i32, eos as i32);
        self.cc.launch(
            fc,
            1,
            1,
            32,
            &mut crate::rawcuda::args::l7(
                &mut p_t, &mut p_o, &mut p_c, &mut p_s, &mut p_p, &mut p_sl, &mut p_e,
            ),
        )
    }
}
