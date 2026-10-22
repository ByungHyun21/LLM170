
use super::*;

impl W4a16Dec {

    pub fn forward_prefill_toks(
        &mut self,
        slot: usize,
        toks: &[u32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !self.embed_ready() {
            return Err("forward_prefill_toks: 임베딩 미등록".into());
        }
        if !(2..=CHAIN_TMAX).contains(&t) || toks.len() != t || slot >= self.n_slots {
            return Err(format!(
                "forward_prefill_toks: t={t} toks={} 계약 위반",
                toks.len()
            ));
        }
        if let Some(&bad) = toks.iter().find(|&&x| x as usize >= self.emb_n) {
            return Err(format!("forward_prefill_toks: 토큰 범위 위반 {bad}"));
        }
        self.ensure_chain_bufs()?;
        let tb = unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u8, t * 4) };
        self.cc.h2d_async(self.dtg, tb)?;
        self.embed_gather_launch(self.dtg, self.dres, t)?;
        self.chain_t_sel(slot, None, t, head, false, false, false)
    }

    pub fn forward_prefill_toks_greedy(
        &mut self,
        slot: usize,
        toks: &[u32],
        t: usize,
    ) -> Result<u32, String> {
        let _g = self.cc.guard()?;
        if !self.embed_ready() {
            return Err("forward_prefill_toks_greedy: 임베딩 미등록".into());
        }
        if !(2..=CHAIN_TMAX).contains(&t) || toks.len() != t || slot >= self.n_slots {
            return Err(format!(
                "forward_prefill_toks_greedy: t={t} toks={} 계약 위반",
                toks.len()
            ));
        }
        if let Some(&bad) = toks.iter().find(|&&x| x as usize >= self.emb_n) {
            return Err(format!("forward_prefill_toks_greedy: 토큰 범위 위반 {bad}"));
        }
        self.ensure_chain_bufs()?;
        let tb = unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u8, t * 4) };
        self.cc.h2d_async(self.dtg, tb)?;
        self.embed_gather_launch(self.dtg, self.dres, t)?;
        let v = self.chain_t_sel(slot, None, t, true, false, false, true)?;
        Ok(v[0] as u32)
    }

    pub(super) fn chain_device_t(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        self.chain_t(slot, rows, t, head, false, false, false)
    }

    pub(super) fn chain_t(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
        all: bool,
        logits: bool,
        last_argmax: bool,
    ) -> Result<Vec<f32>, String> {
        self.chain_t_sel(slot, Some(rows), t, head, all, logits, last_argmax)
    }

    pub(super) fn chain_t_sel(
        &mut self,
        slot: usize,
        rows: Option<&[f32]>,
        t: usize,
        head: bool,
        all: bool,
        logits: bool,
        last_argmax: bool,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !(2..=CHAIN_TMAX).contains(&t)
            || rows.is_some_and(|r| r.len() != t * self.hidden)
            || slot >= self.n_slots
        {
            return Err(format!(
                "chain_device_t: t={t} rows={:?} 계약 위반",
                rows.map(<[f32]>::len)
            ));
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize + t > cap {
            return Err(format!("{CTX_OVERFLOW}: pos{pos}+T{t} > kvcap{cap}"));
        }
        self.kv_ensure(slot, pos as usize + t + 2)?;
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.gdn.is_none() {
            return Err("chain_device_t: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        self.ensure_norm_bufs(t)?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        let dyt = self.ensure_dyt(t)?;
        if let Some(r) = rows {
            let rb = unsafe { std::slice::from_raw_parts(r.as_ptr() as *const u8, r.len() * 4) };
            self.cc.h2d_async(self.dres, rb)?;
        }
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let h = self.hidden;
        let mut ab = self.dab_dev;
        let interval = self.attn.attn.ok_or("attn: 형상 미등록")?.interval;
        let win = self.window;
        let owns_final = win.owns_final(self.n_layers);
        let mut gi = win.gdn_start(interval);
        let plain = self.plain_weights;
        let gemm_mma = t >= 16;
        if t > GEMM_FFMA_TMAX && !gemm_mma {
            return Err(format!(
                "chain_device_t: t={t} > {GEMM_FFMA_TMAX} — FFMA 폴백 상한(t<16 진단 또는 t<16)"
            ));
        }
        let lp = if t <= GEMV_TR {
            LinPath::GemvT
        } else {
            LinPath::Gemm
        };
        let fuse = matches!(lp, LinPath::GemvT);
        for off in 0..win.n() {
            let il = win.abs(off);
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("T{il} input norm: {e}"))?;
            let branch = if (il + 1).is_multiple_of(interval) {
                let xin = if plain { xn } else { xh };
                if fuse {
                    self.gemv_multi_t_dev(plain, il, 0, xin, t)
                        .map_err(|e| format!("T{il} qkv: {e}"))?;
                } else {
                    self.lin_forward(plain, lp, &format!("blk.{il}.attn_q.weight"), xin, s0, t)?;
                    self.lin_forward(plain, lp, &format!("blk.{il}.attn_k.weight"), xin, s1, t)?;
                    self.lin_forward(plain, lp, &format!("blk.{il}.attn_v.weight"), xin, s1b, t)?;
                }
                self.attn_chain_dev_run(slot, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("T{il} attn: {e}"))?
            } else {
                let xin = if plain { xn } else { xh };
                if fuse {
                    self.gemv_multi_t_dev(plain, il, 0, xin, t)
                        .map_err(|e| format!("T{il} qkvz: {e}"))?;
                } else {
                    self.lin_forward(plain, lp, &format!("blk.{il}.attn_qkv.weight"), xin, s0, t)?;
                    self.lin_forward(plain, lp, &format!("blk.{il}.attn_gate.weight"), xin, s1, t)?;
                }
                let g = self
                    .gdn_chain_dev_run(slot, gi, t, xn, s0, s1)
                    .map_err(|e| format!("T{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1).is_multiple_of(interval) {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            if plain {
                self.lin_forward(plain, lp, &lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                if self.dx32.cap < t * ko || self.dx16.cap < t * ko {
                    return Err(format!("T{il} branch 융합: dx32/dx16 부족 {}", t * ko));
                }
                self.lin_forward(plain, lp, &lo, self.dx32.ptr, dyt, t)?;
            }
            let xh = self.ensure_dx32(t * h)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, dyt, t, xh)
                .map_err(|e| format!("T{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev_t(il, xn2, t)
                    .map_err(|e| format!("T{il} moe: {e}"))?;
            } else {
                let xh3 = xh;
                let _ = xn2;
                if fuse {
                    self.gemv_multi_t_dev(false, il, 1, xh3, t)
                        .map_err(|e| format!("T{il} gateup: {e}"))?;
                } else {
                    self.lin_forward(false, lp, &format!("blk.{il}.ffn_gate.weight"), xh3, s0, t)?;
                    self.lin_forward(false, lp, &format!("blk.{il}.ffn_up.weight"), xh3, s1, t)?;
                }
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                self.ew_dev_fused(s0, s1, s2, t * kd)?;
                self.lin_forward(false, lp, &dn, self.dx32.ptr, s3, t)?;
                ab = s3;
            }
        }
        let endp = pos + t as u32;
        if !self.graph.capture_no_host && endp.is_multiple_of(512) {
            let _ = self.checkpoint_gdn(slot, endp);
        }
        if all {
            if !owns_final {
                return Err("chain_t: 비최종 창 — 전 위치 head/argmax는 마지막 샤드 소관".into());
            }
            let xn_all = self
                .norm_resid_dev(2 * self.n_layers, self.dres, ab, t, 0)
                .map_err(|e| format!("S final norm: {e}"))?;
            self.ensure_batch_bufs()?;
            if logits && t > BATCH_DEC_MAX {
                for r in 0..t {
                    self.head_gemv_launch(
                        xn_all + (r * self.hidden) as u64 * 4,
                        self.dbatch_lg + (r * self.head_n) as u64 * 4,
                    )?;
                }
            } else {
                self.head_gemv_t_launch(xn_all, self.dbatch_lg, t)?;
            }
            if logits {
                let mut ob = vec![0u8; t * self.head_n * 4];
                self.cc
                    .d2h_async(ob.as_mut_ptr(), self.dbatch_lg, ob.len())?;
                if !self.graph.capture_no_host {
                    self.cc.sync()?;
                    let pos_after = pos + t as u32;
                    self.slot_pos[slot] = pos_after;
                    self.attn_set_pos(slot, pos_after)?;
                }
                return Ok(ob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect());
            }
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
            let tb = unsafe {
                std::slice::from_raw_parts_mut(self.graph.pin_batch_tok as *mut u8, t * 4)
            };
            self.cc.d2h_async(tb.as_mut_ptr(), self.dbatch_am, t * 4)?;
            if !self.graph.capture_no_host {
                self.cc.sync()?;
                let pos_after = pos + t as u32;
                self.slot_pos[slot] = pos_after;
                self.attn_set_pos(slot, pos_after)?;
            }
            return Ok((0..t)
                .map(|k| {
                    u32::from_le_bytes([tb[k * 4], tb[k * 4 + 1], tb[k * 4 + 2], tb[k * 4 + 3]])
                        as f32
                })
                .collect());
        }
        let last = (t - 1) as u64 * (h as u64) * 4;
        let (xn_last, out_rows) = if owns_final {
            (
                self.norm_resid_dev(2 * self.n_layers, self.dres + last, ab + last, 1, 0)
                    .map_err(|e| format!("T final norm: {e}"))?,
                h,
            )
        } else {
            self.axpy_dev(1.0, ab, self.dres, t * h)?;
            (self.dres, t * h)
        };
        let mut ob = vec![
            0u8;
            if head && !last_argmax {
                self.head_n * 4
            } else {
                out_rows * 4
            }
        ];
        let mut tok4 = [0u8; 4];
        if head {
            if self.head_w == 0 || !owns_final {
                return Err("chain_device_t: head 미등록/비최종 창".into());
            }
            if last_argmax && self.argmax_out == 0 {
                return Err("chain_device_t: argmax 미등록 — upload_head 선행".into());
            }
            self.head_gemv_launch(xn_last, self.head_out)?;
            if last_argmax {
                let fa = self.cc.function("w4a16_argmax_min")?;
                let (mut p_l, mut p_nn, mut p_a) =
                    (self.head_out, self.head_n as i32, self.argmax_out);
                self.cc.launch(
                    fa,
                    1,
                    1,
                    1024,
                    &mut crate::rawcuda::args::l3(&mut p_l, &mut p_nn, &mut p_a),
                )?;
                self.cc.d2h_async(tok4.as_mut_ptr(), self.argmax_out, 4)?;
            } else {
                self.cc
                    .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
            }
        } else {
            self.cc.d2h_async(ob.as_mut_ptr(), xn_last, out_rows * 4)?;
        }
        self.cc.sync()?;
        let pos_after = pos + t as u32;
        self.slot_pos[slot] = pos_after;
        self.attn_set_pos(slot, pos_after)?;
        if head && last_argmax {
            return Ok(vec![u32::from_le_bytes(tok4) as f32]);
        }
        let v: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if head {
            llm170_diag::fp::fp_record("gpu.logits", &v);
        } else {
            llm170_diag::fp::fp_record("gpu.xn", &v[v.len() - h..]);
        }
        Ok(v)
    }
}
