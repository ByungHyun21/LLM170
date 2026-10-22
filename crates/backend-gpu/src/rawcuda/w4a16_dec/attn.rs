use super::*;

impl W4a16Dec {
    pub fn attn_set_pos(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.attn.dpp == 0 || slot >= self.n_slots {
            return Err("attn: pp 미할당/슬롯 범위".into());
        }
        if self.graph.capture_pinned_src {
            return Ok(());
        }
        self.cc
            .h2d_async(self.attn.dpp + (slot as u64) * 4, &pos.to_le_bytes())
    }

    pub(super) fn attn_pp_ptr(&self, slot: usize) -> CUdeviceptr {
        self.attn.dpp + (slot as u64) * 4
    }

    pub(super) fn attn_row_ptrs(
        &self,
        slot: usize,
        layer: usize,
        pos: usize,
    ) -> Result<(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr), String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        let pool = self.attn.kv_pool.as_ref().ok_or("attn: KvPool 미등록")?;
        if slot >= self.n_slots {
            return Err(format!("attn row: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        let bi = pos / 512;
        let off = (pos % 512) as u64;
        let id = pool
            .get(slot, layer, bi)
            .ok_or_else(|| format!("attn row: KV 블록 미할당 slot={slot} layer={layer} bi={bi}"))?
            as u64;
        let kvd = dm.kv_dim() as u64;
        let kvh = dm.kv_heads as u64;
        let bk = 512 * kvd;
        let bs = 512 * kvh * 4;
        let vb = self.attn.kv_vram_blocks;
        Ok((
            Self::kv_block_addr(self.attn.dkc, self.attn.dkc_h, id, bk, vb) + off * kvd,
            Self::kv_block_addr(self.attn.dvc, self.attn.dvc_h, id, bk, vb) + off * kvd,
            Self::kv_block_addr(self.attn.dksc, self.attn.dksc_h, id, bs, vb) + off * kvh * 4,
            Self::kv_block_addr(self.attn.dvsc, self.attn.dvsc_h, id, bs, vb) + off * kvh * 4,
        ))
    }

    pub(super) fn attn_max_kv(&self) -> Result<(usize, usize), String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if self.attn.layers.is_empty() {
            return Ok((dm.kv_dim(), dm.kv_heads));
        }
        Ok((
            self.attn
                .layers
                .iter()
                .map(|l| l.kv_heads * l.d)
                .max()
                .unwrap_or(dm.kv_dim()),
            self.attn
                .layers
                .iter()
                .map(|l| l.kv_heads)
                .max()
                .unwrap_or(dm.kv_heads),
        ))
    }

    #[allow(dead_code)]
    pub(super) fn attn_layer_of(&self, il: usize) -> Result<AttnLayer, String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        Ok(self
            .attn
            .layers
            .get(il)
            .copied()
            .unwrap_or_else(|| dm.layer(il)))
    }

    pub(super) fn attn_blk_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn.attn {
            Some(dm) => {
                let mb = dm.cap.div_ceil(512);
                self.attn.dblk_tab + (slot * dm.n_attn * mb * 4) as u64 * 8
            }
            None => self.attn.dblk_tab,
        }
    }

    pub(super) fn attn_prep_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let f = self.cc.function("attn_prep_q")?;
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        #[allow(clippy::type_complexity)]
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut aa) = (
            qg_dev,
            kin_dev,
            vin_dev,
            self.attn.dqnw_a,
            self.attn.dknw_a,
            self.attn.dqh_a,
            self.attn_blk_ptr(slot),
            self.attn_pp_ptr(slot),
        );
        let (mut n0, mut n1) = (0u64, 0u64);
        self.cc.launch(
            f,
            t_len as u32,
            (dm.q_heads + dm.kv_heads) as u32,
            128,
            &mut crate::rawcuda::args::l15(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut aa, &mut n0,
                &mut n1, &mut tl, &mut lay, &mut qh, &mut kvh, &mut cp,
            ),
        )
    }

    pub(super) fn attn_fwd3s_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
    ) -> Result<(), String> {
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fwd3s: T={t_len} — 소형 전용 도메인 위반"));
        }
        if t_len > 8 {
            return self.attn_fa2_launch(slot, layer, t_len, qg_dev);
        }
        self.attn_fwd3s_legacy_launch(slot, layer, t_len, qg_dev)
    }

    pub(super) fn attn_fwd3s_legacy_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fwd3s: T={t_len} — 소형 전용 도메인 위반"));
        }
        if self.attn.dattn_part == 0 {
            return Err("attn fwd3s: dattn_part 미할당 — ensure_attn_bufs 선행".into());
        }
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut n0, mut n1) = (0u64, 0u64);
        let spv = if t_len <= 8 {
            ATTN_SPLITS_D
        } else {
            ATTN_SPLITS
        };
        let mut sp = spv as i32;
        let fp = self.cc.function("attn_fwd3s_part_qg")?;
        let (mut f0, mut f1, mut f2, mut f3) = (
            self.attn.dqh_a,
            self.attn_blk_ptr(slot),
            self.attn.dattn_part,
            self.attn_pp_ptr(slot),
        );
        let mut scl = dm.layer(layer).scale();
        self.cc.launch(
            fp,
            t_len as u32,
            (dm.kv_heads * spv) as u32,
            256,
            &mut crate::rawcuda::args::l13(
                &mut f0, &mut f1, &mut f2, &mut f3, &mut n0, &mut n1, &mut tl, &mut lay, &mut qh,
                &mut kvh, &mut cp, &mut sp, &mut scl,
            ),
        )?;
        let nm = t_len * dm.q_heads * 256;
        let (m32, m16) = (self.ensure_dx32(nm)?, self.ensure_dx16(nm)?);
        let fm = self.cc.function("attn_fwd3s_merge")?;
        let (mut mp, mut mg, mut mo) = (self.attn.dattn_part, qg_dev, self.attn.doutv_a);
        let (mut mo32, mut mo16) = (m32, m16);
        let (mut tl2, mut qh2) = (t_len as i32, dm.q_heads as i32);
        self.cc.launch(
            fm,
            t_len as u32,
            dm.q_heads as u32,
            256,
            &mut crate::rawcuda::args::l8(
                &mut mp, &mut mg, &mut mo, &mut mo32, &mut mo16, &mut tl2, &mut qh2, &mut sp,
            ),
        )
    }

    pub(super) fn attn_fa2_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fa2: T={t_len} — 도메인 위반"));
        }
        let f = self.cc.function("attn_fa2_qg")?;
        let nm = t_len * dm.q_heads * 256;
        let (m32, m16) = (self.ensure_dx32(nm)?, self.ensure_dx16(nm)?);
        #[allow(clippy::type_complexity)]
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6) = (
            self.attn.dqh_a,
            self.attn_blk_ptr(slot),
            qg_dev,
            self.attn.doutv_a,
            m32,
            m16,
            self.attn_pp_ptr(slot),
        );
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let mut scl = dm.layer(layer).scale();
        let (mut swv, mut ngate) = (0i32, 0i32);
        self.cc.launch(
            f,
            t_len.div_ceil(32) as u32,
            dm.q_heads as u32,
            128,
            &mut crate::rawcuda::args::l15(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut tl, &mut lay,
                &mut qh, &mut kvh, &mut cp, &mut scl, &mut swv, &mut ngate,
            ),
        )
    }

    pub(super) fn attn_qsa_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
        sel_dev: CUdeviceptr,
        nsel_dev: CUdeviceptr,
        sel_stride: usize,
    ) -> Result<(), String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn qsa: T={t_len} — 도메인 위반"));
        }
        self.attn_prep_launch(slot, layer, t_len, qg_dev, kin_dev, vin_dev)?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let mut scl = dm.layer(layer).scale();
        let mut st = sel_stride as i32;
        let f = self.cc.function("attn_qsa_part")?;
        let (mut a0, mut a1, mut a2, mut a3) = (
            self.attn.dqh_a,
            self.attn_blk_ptr(slot),
            self.attn.dattn_part,
            self.attn_pp_ptr(slot),
        );
        let (mut a4, mut a5) = (sel_dev, nsel_dev);
        self.cc.launch(
            f,
            t_len as u32,
            dm.q_heads as u32,
            256,
            &mut crate::rawcuda::args::l13(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut st, &mut tl, &mut lay,
                &mut qh, &mut kvh, &mut cp, &mut scl,
            ),
        )?;
        let nm = t_len * dm.q_heads * 256;
        let (m32, m16) = (self.ensure_dx32(nm)?, self.ensure_dx16(nm)?);
        let fm = self.cc.function("attn_fwd3s_merge")?;
        let (mut mp, mut mg, mut mo) = (self.attn.dattn_part, qg_dev, self.attn.doutv_a);
        let (mut mo32, mut mo16) = (m32, m16);
        let (mut tl2, mut qh2, mut sp) = (t_len as i32, dm.q_heads as i32, 1i32);
        self.cc.launch(
            fm,
            t_len as u32,
            dm.q_heads as u32,
            256,
            &mut crate::rawcuda::args::l8(
                &mut mp, &mut mg, &mut mo, &mut mo32, &mut mo16, &mut tl2, &mut qh2, &mut sp,
            ),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_prep_g4_q_host(
        &mut self,
        q: &[f32],
        qnw: &[f32],
        t_len: usize,
        q_heads: usize,
        d: usize,
        pairs: usize,
        span: usize,
        denom: usize,
        base: f32,
        pos0: u32,
    ) -> Result<Vec<f32>, String> {
        if q.len() != t_len * q_heads * d || qnw.len() != d {
            return Err(format!(
                "prep_g4: q={} qnw={} 계약 위반",
                q.len(),
                qnw.len()
            ));
        }
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dq = self.cc.alloc(q.len() * 4)?;
        let dqnw = self.cc.alloc(qnw.len() * 4)?;
        let dqh = self.cc.alloc(q.len() * 4)?;
        let rp: Vec<f32> = vec![
            d as f32,
            pairs as f32,
            span as f32,
            denom as f32,
            base,
            1.0,
            0.0,
        ];
        let drp = self.cc.alloc(rp.len() * 4)?;
        let dpp = self.cc.alloc(4)?;
        let rr = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dq, &f32b(q))?;
            self.cc.h2d(dqnw, &f32b(qnw))?;
            self.cc.h2d(drp, &f32b(&rp))?;
            self.cc.h2d(dpp, &pos0.to_le_bytes())?;
            let f = self.cc.function("attn_prep_g4")?;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
                (dq, 0u64, 0u64, dqnw, 0u64, dqh, 0u64, dpp, drp);
            let (mut rblk, mut rpp) = (0u64, 0u64);
            let (mut tl, mut lay, mut qh, mut kvh, mut cp) =
                (t_len as i32, 0i32, q_heads as i32, 0i32, 32i32);
            self.cc.launch(
                f,
                t_len as u32,
                q_heads as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                    &mut rblk, &mut rpp, &mut a8, &mut tl, &mut lay, &mut qh, &mut kvh, &mut cp,
                ),
            )?;
            self.cc.sync()?;
            let mut ob = vec![0u8; q.len() * 4];
            self.cc.d2h(&mut ob, dqh)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dq);
        let _ = self.cc.free(dqnw);
        let _ = self.cc.free(dqh);
        let _ = self.cc.free(drp);
        let _ = self.cc.free(dpp);
        rr
    }

    pub fn set_g4_rope(&mut self, rp: &[f32]) -> Result<(), String> {
        if rp.is_empty() || !rp.len().is_multiple_of(7) {
            return Err(format!("set_g4_rope: len={} — 7의 배수 계약", rp.len()));
        }
        let _g = self.cc.guard()?;
        let bytes: Vec<u8> = rp.iter().flat_map(|x| x.to_le_bytes()).collect();
        if self.attn.g4_rp == 0 || self.attn.attn.map(|d| d.n_attn) != Some(rp.len() / 7) {
            if self.attn.g4_rp != 0 {
                self.cc.free(self.attn.g4_rp)?;
            }
            self.attn.g4_rp = self.cc.alloc(bytes.len())?;
        }
        self.cc.h2d(self.attn.g4_rp, &bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_g4_host(
        &mut self,
        slot: usize,
        q: &[f32],
        kin: &[f32],
        vin: &[f32],
        qnw: &[f32],
        knw: &[f32],
        layer: usize,
        t_len: usize,
        q_heads: usize,
        kv_heads: usize,
        d: usize,
        sw: usize,
        kq_scale: f32,
        splits: usize,
        pos0: u32,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if self.attn.g4_rp == 0 || layer >= dm.n_attn {
            return Err(format!(
                "attn_g4: rope 테이블 부재/층 초과 — layer={layer} n_attn={}",
                dm.n_attn
            ));
        }
        if q.len() != t_len * q_heads * d
            || kin.len() != t_len * kv_heads * d
            || vin.len() != t_len * kv_heads * d
            || qnw.len() != d
            || knw.len() != d
        {
            return Err("attn_g4: 입력 형상 계약 위반".into());
        }
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dq = self.cc.alloc(q.len() * 4)?;
        let dk = self.cc.alloc(kin.len() * 4)?;
        let dv = self.cc.alloc(vin.len() * 4)?;
        let dqnw = self.cc.alloc(qnw.len() * 4)?;
        let dknw = self.cc.alloc(knw.len() * 4)?;
        let dqh = self.cc.alloc(q.len() * 4)?;
        let npart = t_len * q_heads * splits;
        let dpart = self.cc.alloc(npart * (d + 2) * 4)?;
        let dout = self.cc.alloc(t_len * q_heads * d * 4)?;
        let drp = self.attn.g4_rp;
        let mb = dm.cap.div_ceil(512);
        let dblk = self.attn.dblk_tab + (slot * dm.n_attn * mb * 4 * 8) as u64;
        let dpp = self.cc.alloc(4)?;
        let rr = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dq, &f32b(q))?;
            self.cc.h2d(dk, &f32b(kin))?;
            self.cc.h2d(dv, &f32b(vin))?;
            self.cc.h2d(dqnw, &f32b(qnw))?;
            self.cc.h2d(dknw, &f32b(knw))?;
            self.cc.h2d(dpp, &pos0.to_le_bytes())?;
            {
                let f = self.cc.function("attn_prep_g4")?;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
                    (dq, dk, dv, dqnw, dknw, dqh, dblk, dpp, drp);
                let (mut rblk, mut rpp) = (0u64, 0u64);
                let (mut tl, mut lay, mut qh, mut kvh, mut cp) = (
                    t_len as i32,
                    layer as i32,
                    q_heads as i32,
                    kv_heads as i32,
                    dm.cap as i32,
                );
                self.cc.launch(
                    f,
                    t_len as u32,
                    (q_heads + kv_heads) as u32,
                    128,
                    &mut crate::rawcuda::args::l16(
                        &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                        &mut rblk, &mut rpp, &mut a8, &mut tl, &mut lay, &mut qh, &mut kvh,
                        &mut cp,
                    ),
                )?;
            }
            {
                let f = self.cc.function("attn_fwd3s_part_g4")?;
                let (mut a0, mut a1, mut a2, mut a3) = (dqh, dblk, dpart, dpp);
                let (mut rblk, mut rpp) = (0u64, 0u64);
                let (mut tl, mut lay, mut qh, mut kvh, mut cp) = (
                    t_len as i32,
                    layer as i32,
                    q_heads as i32,
                    kv_heads as i32,
                    dm.cap as i32,
                );
                let (mut sp, mut dd, mut swv) = (splits as i32, d as i32, sw as i32);
                let mut scl = kq_scale;
                self.cc.launch(
                    f,
                    t_len as u32,
                    (q_heads * splits) as u32,
                    d as u32,
                    &mut crate::rawcuda::args::l15(
                        &mut a0, &mut a1, &mut a2, &mut a3, &mut rblk, &mut rpp, &mut tl, &mut lay,
                        &mut qh, &mut kvh, &mut cp, &mut sp, &mut dd, &mut swv, &mut scl,
                    ),
                )?;
            }
            {
                let f = self.cc.function("attn_fwd3s_merge_g4")?;
                let (mut a0, mut a1, mut a2, mut a3) = (dpart, dout, 0u64, 0u64);
                let (mut tl, mut qh, mut sp, mut dd) =
                    (t_len as i32, q_heads as i32, splits as i32, d as i32);
                self.cc.launch(
                    f,
                    t_len as u32,
                    q_heads as u32,
                    d as u32,
                    &mut crate::rawcuda::args::l8(
                        &mut a0, &mut a1, &mut a2, &mut a3, &mut tl, &mut qh, &mut sp, &mut dd,
                    ),
                )?;
            }
            self.cc.sync()?;
            let mut ob = vec![0u8; t_len * q_heads * d * 4];
            self.cc.d2h(&mut ob, dout)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        for p in [dq, dk, dv, dqnw, dknw, dqh, dpart, dout, dpp] {
            let _ = self.cc.free(p);
        }
        rr
    }

    pub fn attn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg: &[f32],
        kin: &[f32],
        vin: &[f32],
        pos0: u32,
    ) -> Result<Vec<f32>, String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if layer >= dm.n_attn || slot >= self.n_slots {
            return Err(format!("attn: 범위 위반 layer={layer} slot={slot}"));
        }
        if qg.len() != t_len * dm.qg_dim()
            || kin.len() != t_len * dm.kv_dim()
            || vin.len() != t_len * dm.kv_dim()
        {
            return Err("attn: 입력 형상 계약 위반".into());
        }
        if pos0 as usize + t_len > dm.cap {
            return Err(format!(
                "attn: pos0={pos0} + T={t_len} > cap={}(--ctx 상향)",
                dm.cap
            ));
        }
        self.ensure_attn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.attn.dqg_a, b(qg))?;
        self.cc.h2d(self.attn.dkin_a, b(kin))?;
        self.cc.h2d(self.attn.dvin_a, b(vin))?;
        self.attn_set_pos(slot, pos0)?;
        let (aqg, akin, avin) = (self.attn.dqg_a, self.attn.dkin_a, self.attn.dvin_a);
        self.attn_prep_launch(slot, layer, t_len, aqg, akin, avin)?;
        self.attn_fwd3s_launch(slot, layer, t_len, aqg)?;
        let mut buf = vec![0u8; t_len * dm.q_dim() * 4];
        self.cc.d2h(&mut buf, self.attn.doutv_a)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, t_len * dm.q_dim()) }
                .to_vec(),
        )
    }


    pub(super) fn attn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX || layer >= dm.n_attn || slot >= self.n_slots {
            return Err("attn dev: 도메인/범위 위반".into());
        }
        let pos = self.slot_pos[slot];
        if pos as usize + t_len > dm.cap {
            return Err(format!("attn dev: pos{pos}+T{t_len} > cap{}", dm.cap));
        }
        self.attn_set_pos(slot, pos)?;
        self.ensure_attn_bufs(t_len)?;
        self.attn_prep_launch(slot, layer, t_len, qg_dev, kin_dev, vin_dev)?;
        self.attn_fwd3s_launch(slot, layer, t_len, qg_dev)?;
        Ok(self.attn.doutv_a)
    }

    pub(super) fn attn_chain_dev_batch(
        &mut self,
        slots: &[usize],
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0
            || t_len > BATCH_DEC_MAX
            || t_len != slots.len()
            || layer >= dm.n_attn
            || self.attn.dattn_part == 0
            || slots.iter().any(|&s| s >= self.n_slots)
        {
            return Err("attn batch: 도메인/범위 위반".into());
        }
        for &s in slots {
            let pos = self.slot_pos[s];
            if pos as usize + 1 > dm.cap {
                return Err(format!("attn batch: slot{s} pos{pos} > cap{}", dm.cap));
            }
        }
        self.ensure_attn_bufs(t_len)?;
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let spv = if t_len <= 8 {
            ATTN_SPLITS_D
        } else {
            ATTN_SPLITS
        };
        let mut sp = spv as i32;
        let (row_blk, row_pp) = (self.drow_blk, self.drow_pp);
        {
            let f = self.cc.function("attn_prep_q")?;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut aa) = (
                qg_dev,
                kin_dev,
                vin_dev,
                self.attn.dqnw_a,
                self.attn.dknw_a,
                self.attn.dqh_a,
                self.attn.dblk_tab,
                self.attn.dpp,
            );
            let (mut tl, mut lay, mut qh2, mut kvh2, mut cp2) =
                (t_len as i32, layer as i32, qh, kvh, cp);
            let (mut rb, mut rp) = (row_blk, row_pp);
            self.cc.launch(
                f,
                t_len as u32,
                (dm.q_heads + dm.kv_heads) as u32,
                128,
                &mut crate::rawcuda::args::l15(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut aa,
                    &mut rb, &mut rp, &mut tl, &mut lay, &mut qh2, &mut kvh2, &mut cp2,
                ),
            )?;
        }
        let fp = self.cc.function("attn_fwd3s_part_q")?;
        let fm = self.cc.function("attn_fwd3s_merge")?;
        {
            let nm = t_len * dm.q_heads * 256;
            self.ensure_dx32(nm)?;
            self.ensure_dx16(nm)?;
        }
        {
            let (mut f0, mut f1, mut f2, mut f3) = (
                self.attn.dqh_a,
                self.attn.dblk_tab,
                self.attn.dattn_part,
                self.attn.dpp,
            );
            let (mut tl, mut lay) = (t_len as i32, layer as i32);
            let (mut rb, mut rp) = (row_blk, row_pp);
            let mut scl = dm.layer(layer).scale();
            self.cc.launch(
                fp,
                t_len as u32,
                (dm.q_heads * spv) as u32,
                256,
                &mut crate::rawcuda::args::l13(
                    &mut f0, &mut f1, &mut f2, &mut f3, &mut rb, &mut rp, &mut tl, &mut lay,
                    &mut qh, &mut kvh, &mut cp, &mut sp, &mut scl,
                ),
            )?;
        }
        {
            let (mut mp, mut mg, mut mo) = (self.attn.dattn_part, qg_dev, self.attn.doutv_a);
            let (mut mo32, mut mo16) = (self.dx32.ptr, self.dx16.ptr);
            let (mut tl2, mut qh2) = (t_len as i32, dm.q_heads as i32);
            self.cc.launch(
                fm,
                t_len as u32,
                dm.q_heads as u32,
                256,
                &mut crate::rawcuda::args::l8(
                    &mut mp, &mut mg, &mut mo, &mut mo32, &mut mo16, &mut tl2, &mut qh2, &mut sp,
                ),
            )?;
        }
        Ok(self.attn.doutv_a)
    }
}
