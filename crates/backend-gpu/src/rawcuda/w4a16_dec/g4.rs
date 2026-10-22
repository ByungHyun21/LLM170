
use super::*;

impl W4a16Dec {
    pub fn g4_moe_host(
        &mut self,
        il: usize,
        hn: &[f32],
        xn: &[f32],
        pes: &[f32],
    ) -> Result<Vec<f32>, String> {
        let h = self.hidden;
        let n_exp = self.moe.n_experts;
        if hn.len() != h || xn.len() != h {
            return Err(format!(
                "g4_moe: 입력 계약 위반 hn={} xn={} != hidden={h}",
                hn.len(),
                xn.len()
            ));
        }
        if n_exp == 0 || pes.len() != n_exp {
            return Err(format!(
                "g4_moe: pes={} != n_experts={n_exp} (set_moe 선행 확인)",
                pes.len()
            ));
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dh = self.cc.alloc(h * 4)?;
        let dx = self.cc.alloc(h * 4)?;
        let dp = self.cc.alloc(n_exp * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dh, &f32b(hn))?;
            self.cc.h2d(dx, &f32b(xn))?;
            self.cc.h2d(dp, &f32b(pes))?;
            let rn = format!("blk.{il}.moe_gate.weight");
            if self.lins.contains_key(&rn) {
                self.gemv_launch(&rn, dh, self.moe.drt.ptr)?;
            } else {
                self.plain_gemv_launch(&rn, dh, self.moe.drt.ptr)?;
            }
            self.moe_topk_dev()?;
            {
                let f = self.cc.function("w4a16_moe_pes_scale")?;
                let (mut p_w, mut p_i, mut p_p) = (self.moe.moe_wt, self.moe.moe_idx, dp);
                let mut p_n = self.moe.top_k as i32;
                self.cc.launch(
                    f,
                    (self.moe.top_k as u32).div_ceil(256).max(1),
                    1,
                    256,
                    &mut crate::rawcuda::args::l4(&mut p_w, &mut p_i, &mut p_p, &mut p_n),
                )?;
            }
            if self.moe.moe_resident {
                self.moe_experts_batch(il, dx, self.moe.top_k)?;
                self.moe_accum_dev(
                    self.moe.moe_wt,
                    self.moe.exp.dn,
                    self.moe.dmo.ptr,
                    0,
                    self.moe.top_k,
                    h,
                )?;
            } else {
                let sel = self.moe_read_selection()?;
                Self::zero_dev(&self.cc, self.moe.dmo.ptr, h * 4)?;
                self.moe_experts_streaming(il, dx, &sel)?;
            }
            self.cc.sync()?;
            let mut ob = vec![0u8; h * 4];
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.moe.dmo.ptr, h * 4)?;
            self.cc.sync()?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dh);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(dp);
        r
    }

    pub fn g4_gemm_host(
        &mut self,
        name: &str,
        x: &[f32],
        t: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let (_, _, n, kk) = self.lin_spec(name)?;
        if t == 0 || k != kk || x.len() != t * k {
            return Err(format!(
                "g4_gemm: {name} 계약 위반 t={t} k={k}(선형 {kk}) x={}",
                x.len()
            ));
        }
        if self.g4_a.cap < t * k {
            if self.g4_a.ptr != 0 {
                self.cc.free(self.g4_a.ptr)?;
            }
            self.g4_a.ptr = self.cc.alloc(t * k * 4)?;
            self.g4_a.cap = t * k;
        }
        if self.g4_y.cap < t * n {
            if self.g4_y.ptr != 0 {
                self.cc.free(self.g4_y.ptr)?;
            }
            self.g4_y.ptr = self.cc.alloc(t * n * 4)?;
            self.g4_y.cap = t * n;
        }
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.g4_a.ptr, xb)?;
        self.gemm_launch(name, self.g4_a.ptr, self.g4_y.ptr, t)?;
        self.cc.sync()?;
        let mut ob = vec![0u8; t * n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.g4_y.ptr, t * n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub fn build_multi_tables_g4(&mut self) -> Result<(), String> {
        let nl = self.n_layers;
        let dm = self.attn.attn.ok_or("g4 multi: attn 미등록")?;
        let tmax = G4_CHUNK_MAX;
        let qd_max = self
            .attn
            .layers
            .iter()
            .map(|l| dm.q_heads * l.d)
            .max()
            .unwrap_or(0);
        let mut nff_max = 0usize;
        for il in 0..nl {
            let (_, _, n, _) = self.lin_spec(&format!("blk.{il}.ffn_gate.weight"))?;
            nff_max = nff_max.max(n);
        }
        let wmax = qd_max.max(nff_max);
        let (b1, b2, b3) = (
            self.g4_scratch_slot(Self::G4_B1, tmax * wmax * 4)?,
            self.g4_scratch_slot(Self::G4_B2, tmax * wmax * 4)?,
            self.g4_scratch_slot(Self::G4_B3, tmax * wmax * 4)?,
        );
        let mut tab: Vec<u64> = vec![0; nl * 25];
        let mut meta: Vec<[u32; 8]> = vec![[0u32; 8]; nl];
        for il in 0..nl {
            let keqv = self.attn.layers[il].k_eq_v;
            let mut segs0: Vec<(String, u64)> = vec![
                (format!("blk.{il}.attn_q.weight"), b1),
                (format!("blk.{il}.attn_k.weight"), b2),
            ];
            if !keqv {
                segs0.push((format!("blk.{il}.attn_v.weight"), b3));
            }
            let segs1: Vec<(String, u64)> = vec![
                (format!("blk.{il}.ffn_gate.weight"), b1),
                (format!("blk.{il}.ffn_up.weight"), b2),
            ];
            let mut wr = il * 25;
            let mut gm = [[0u32; 4]; 2];
            for (gi, segs) in [&segs0, &segs1].into_iter().enumerate() {
                let off = wr as u32;
                let (mut total, mut tgrid, mut cnt) = (0usize, 0usize, 0u32);
                for (name, out) in segs {
                    let (p0, p1, n, k) = self.lin_spec(name)?;
                    if n > wmax {
                        return Err(format!("g4 multi({name}): n={n} > 슬롯 {wmax}"));
                    }
                    tab[wr] = p0;
                    wr += 1;
                    tab[wr] = p1;
                    wr += 1;
                    tab[wr] = *out;
                    wr += 1;
                    tab[wr] = n as u64;
                    wr += 1;
                    tab[wr] = k as u64;
                    wr += 1;
                    total += n;
                    tgrid += n.div_ceil(GEMV_TR);
                    cnt += 1;
                }
                gm[gi] = [off, cnt, total as u32, tgrid as u32];
            }
            meta[il] = [
                gm[0][0], gm[0][1], gm[0][2], gm[1][0], gm[1][1], gm[1][2], gm[0][3], gm[1][3],
            ];
        }
        let bytes = tab.len() * 8;
        let buf = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, bytes) };
        if self.dmulti == 0 {
            self.dmulti = self.cc.alloc(bytes)?;
        }
        self.cc.h2d(self.dmulti, buf)?;
        self.cc.sync()?;
        self.dmulti_meta = meta;
        Ok(())
    }

    pub fn g4_scratch_slot(&mut self, idx: usize, bytes: usize) -> Result<CUdeviceptr, String> {
        if self.g4_pool.len() <= idx {
            self.g4_pool.resize(idx + 1, Buf { ptr: 0, cap: 0 });
        }
        let (ptr, cap) = {
            let b = &self.g4_pool[idx];
            (b.ptr, b.cap)
        };
        if cap >= bytes {
            return Ok(ptr);
        }
        self.cc.capture_guard("g4_slot");
        self.graph_invalidate();
        self.cc.sync()?;
        let newp = self.cc.alloc(bytes)?;
        if ptr != 0 {
            self.cc.free(ptr)?;
        }
        self.g4_pool[idx] = Buf {
            ptr: newp,
            cap: bytes,
        };
        Ok(newp)
    }

    pub fn g4_dev_gemm(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        t: usize,
        k: usize,
        y_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let (_, _, n, kk) = self.lin_spec(name)?;
        if t == 0 || k != kk {
            return Err(format!("g4_dev_gemm: {name} k={k} != {kk}"));
        }
        let y = self.g4_scratch_slot(y_slot, t * n * 4)?;
        self.gemv_t_rows(name, x_dev, y, t)?;
        Ok(y)
    }

    fn gemv_t_rows(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.gemv_t_launch(name, x_dev, y_dev, t)
    }

    fn plain_gemv_t_rows(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (_, n, k) = self.plain_spec(name)?;
        let mut r0 = 0usize;
        while r0 < t {
            let rows = G4_TROW_MAX.min(t - r0);
            self.plain_gemv_t_launch(
                name,
                x_dev + (r0 * k * 4) as u64,
                y_dev + (r0 * n * 4) as u64,
                rows,
            )?;
            r0 += rows;
        }
        Ok(())
    }

    pub fn g4_dev_gelu(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_slot: usize,
        n: usize,
    ) -> Result<CUdeviceptr, String> {
        if n == 0 {
            return Err("g4_dev_gelu: n=0".into());
        }
        let _g = self.cc.guard()?;
        let y = self.g4_scratch_slot(y_slot, n * 4)?;
        let f = self.cc.function("ew_gelu")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y, 0u64, 0u64);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )?;
        Ok(y)
    }

    pub fn g4_dev_ptr(&self, slot: usize) -> CUdeviceptr {
        self.g4_pool.get(slot).map(|b| b.ptr).unwrap_or(0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_g4_dev(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        layer: usize,
        q_dev: CUdeviceptr,
        k_dev: CUdeviceptr,
        v_dev: CUdeviceptr,
        q_heads: usize,
        kv_heads: usize,
        d: usize,
        sw: usize,
        kq_scale: f32,
        splits: usize,
        qh_slot: usize,
        part_slot: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let t = slots.len();
        if t == 0 || pos.len() != t {
            return Err(format!("attn_g4_dev: t={t} pos={}", pos.len()));
        }
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if self.attn.g4_rp == 0 || layer >= dm.n_attn {
            return Err(format!("attn_g4_dev: layer={layer} n_attn={}", dm.n_attn));
        }
        let qn = t * q_heads * d;
        let qh = self.g4_scratch_slot(qh_slot, qn * 4)?;
        let part = self.g4_scratch_slot(part_slot, t * q_heads * splits * (d + 2) * 4)?;
        let outv = self.g4_scratch_slot(out_slot, qn * 4)?;
        self.g4_atab_prepare(slots, pos)?;
        let pos_dev = self.g4_scratch_slot(Self::G4_ATAB_POS, t * 4)?;
        let rpp_dev = self.g4_scratch_slot(Self::G4_ATAB_RPP, t * 8)?;
        let rblk_dev = self.g4_scratch_slot(Self::G4_ATAB_RBLK, t * 8)?;
        let dpp = pos_dev;
        let d_max = self.attn.layers.iter().map(|l| l.d).max().unwrap_or(d);
        let qnw = self.attn.dqnw_a + (layer * d_max * 4) as u64;
        let knw = self.attn.dknw_a + (layer * d_max * 4) as u64;
        let dblk = self.attn.dblk_tab;
        let drp = self.attn.g4_rp;
        {
            let f = self.cc.function("attn_prep_g4")?;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
                (q_dev, k_dev, v_dev, qnw, knw, qh, dblk, dpp, drp);
            let (mut rblk, mut rpp) = (rblk_dev, rpp_dev);
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            self.cc.launch(
                f,
                t as u32,
                (q_heads + kv_heads) as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                    &mut rblk, &mut rpp, &mut a8, &mut tl, &mut lay, &mut qh_, &mut kvh, &mut cp,
                ),
            )?;
        }
        let contig =
            slots.windows(2).all(|w| w[0] == w[1]) && pos.windows(2).all(|w| w[1] == w[0] + 1);
        if t > 8 && (d == 256 || d == 512) && contig {
            let f = self.cc.function(if d == 512 {
                "attn_fa2_g4_512"
            } else {
                "attn_fa2_qg"
            })?;
            let rows = if d == 512 { 16 } else { 32 };
            let (mut a0, mut a1, mut a2, mut a3) = (qh, self.attn_blk_ptr(slots[0]), 0u64, outv);
            let (mut a4, mut a5, mut a6) = (0u64, 0u64, pos_dev);
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            let (mut swv, mut ngate) = (sw as i32, 1i32);
            let mut scl = kq_scale;
            self.cc.launch(
                f,
                t.div_ceil(rows) as u32,
                q_heads as u32,
                128,
                &mut crate::rawcuda::args::l15(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut tl,
                    &mut lay, &mut qh_, &mut kvh, &mut cp, &mut scl, &mut swv, &mut ngate,
                ),
            )?;
            return Ok(outv);
        }
        {
            let f = self.cc.function("attn_fwd3s_part_g4")?;
            let (mut a0, mut a1, mut a2, mut a3) = (qh, dblk, part, dpp);
            let (mut rblk, mut rpp) = (rblk_dev, rpp_dev);
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            let (mut sp, mut dd, mut swv) = (splits as i32, d as i32, sw as i32);
            let mut scl = kq_scale;
            self.cc.launch(
                f,
                t as u32,
                (q_heads * splits) as u32,
                d as u32,
                &mut crate::rawcuda::args::l15(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut rblk, &mut rpp, &mut tl, &mut lay,
                    &mut qh_, &mut kvh, &mut cp, &mut sp, &mut dd, &mut swv, &mut scl,
                ),
            )?;
        }
        {
            let f = self.cc.function("attn_fwd3s_merge_g4")?;
            let (mut a0, mut a1, mut a2, mut a3) = (part, outv, 0u64, 0u64);
            let (mut tl, mut qh_, mut sp, mut dd) =
                (t as i32, q_heads as i32, splits as i32, d as i32);
            self.cc.launch(
                f,
                t as u32,
                q_heads as u32,
                d as u32,
                &mut crate::rawcuda::args::l8(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut tl, &mut qh_, &mut sp, &mut dd,
                ),
            )?;
        }
        Ok(outv)
    }

    pub fn set_g4_pes(&mut self, il: usize, pes: &[f32]) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.g4_pes.len() <= il {
            self.g4_pes.resize(il + 1, 0);
        }
        if self.g4_pes[il] == 0 {
            let bytes: Vec<u8> = pes.iter().flat_map(|x| x.to_le_bytes()).collect();
            let p = self.cc.alloc(bytes.len())?;
            self.cc.h2d(p, &bytes)?;
            self.g4_pes[il] = p;
        }
        Ok(())
    }

    pub fn g4_pes_dev(&self, il: usize) -> CUdeviceptr {
        self.g4_pes.get(il).copied().unwrap_or(0)
    }

    pub fn set_g4_layer_scalar(&mut self, ls: &[f32]) -> Result<(), String> {
        if ls.len() != self.n_layers {
            return Err(format!(
                "g4 layer_scalar: {} != n_layer={}",
                ls.len(),
                self.n_layers
            ));
        }
        self.g4_ls = ls.to_vec();
        Ok(())
    }

    pub fn g4_atab_prepare(&mut self, slots: &[usize], pos: &[u32]) -> Result<(), String> {
        let t = slots.len();
        if t == 0 || pos.len() != t || t > G4_CHUNK_MAX {
            return Err(format!("g4_atab: t={t} pos={} (1..=16)", pos.len()));
        }
        if self.g4_atab.0 == slots && self.g4_atab.1 == pos {
            return Ok(());
        }
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if slots.iter().any(|&sl| sl >= self.n_slots) {
            return Err("g4_atab: 슬롯 범위 밖".into());
        }
        let mb = dm.cap.div_ceil(512);
        let pos_dev = self.g4_scratch_slot(Self::G4_ATAB_POS, t * 4)?;
        let rpp_dev = self.g4_scratch_slot(Self::G4_ATAB_RPP, t * 8)?;
        let rblk_dev = self.g4_scratch_slot(Self::G4_ATAB_RBLK, t * 8)?;
        self.cc.capture_guard("g4_atab");
        let mut pb: Vec<u8> = Vec::with_capacity(t * 4);
        for &pv in pos {
            pb.extend_from_slice(&pv.to_le_bytes());
        }
        self.cc.h2d_async(pos_dev, &pb)?;
        let mut rppb: Vec<u8> = Vec::with_capacity(t * 8);
        let mut rblkb: Vec<u8> = Vec::with_capacity(t * 8);
        for (i, &sl) in slots.iter().enumerate() {
            rppb.extend_from_slice(&(pos_dev + (i * 4) as u64).to_le_bytes());
            let base = self.attn.dblk_tab + (sl * dm.n_attn * mb * 4 * 8) as u64;
            rblkb.extend_from_slice(&base.to_le_bytes());
        }
        self.cc.h2d_async(rpp_dev, &rppb)?;
        self.cc.h2d_async(rblk_dev, &rblkb)?;
        self.g4_atab = (slots.to_vec(), pos.to_vec());
        Ok(())
    }

    const G4_X: usize = 0;
    const G4_A: usize = 1;
    const G4_B1: usize = 2;
    const G4_B2: usize = 3;
    const G4_B3: usize = 4;
    const G4_ATAB_POS: usize = 10;
    const G4_ATAB_RPP: usize = 11;
    const G4_ATAB_RBLK: usize = 12;
    const G4_HN: usize = 13;
    const G4_NM: usize = 14;
    const G4_MOE: usize = 15;
    const G4_H2: usize = 16;
    const G4_HF: usize = 17;
    const G4_HEADX: usize = 18;
    const G4_HEADOUT: usize = 19;

    pub fn g4_warm_slots(&mut self, t_max: usize) -> Result<(), String> {
        let h = self.hidden;
        let dm = self.attn.attn.ok_or("g4_warm: attn 미등록")?;
        let mut qd_max = 0usize;
        let mut part_max = 0usize;
        for l in &self.attn.layers {
            qd_max = qd_max.max(dm.q_heads * l.d);
            part_max = part_max.max(dm.q_heads * 2 * (l.d + 2));
        }
        let th = t_max * h * 4;
        for s in [
            Self::G4_X,
            Self::G4_A,
            Self::G4_HEADX,
            Self::G4_HN,
            Self::G4_NM,
            Self::G4_MOE,
            Self::G4_H2,
            Self::G4_HF,
        ] {
            self.g4_scratch_slot(s, th)?;
        }
        for s in [Self::G4_B1, Self::G4_B2, Self::G4_B3] {
            self.g4_scratch_slot(s, t_max * qd_max.max(h) * 4)?;
        }
        self.g4_scratch_slot(5, t_max * qd_max * 4)?;
        self.g4_scratch_slot(6, t_max * part_max * 4)?;
        self.g4_scratch_slot(7, t_max * qd_max * 4)?;
        self.g4_scratch_slot(8, th)?;
        self.g4_scratch_slot(9, th)?;
        self.g4_scratch_slot(Self::G4_HEADOUT, t_max * self.head_n * 4)?;
        if self.moe.moe_ffn > 0 && self.moe.n_experts > 0 {
            let ns_max = t_max * self.moe.top_k.max(1);
            self.ensure_dxp(ns_max * self.moe.moe_ffn.max(h))?;
        }
        Ok(())
    }

    fn g4_chunk_body(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        let h = self.hidden;
        let n = t * h;
        let dm = self.attn.attn.ok_or("g4_chunk_dev: attn 미등록")?;
        let x = self.g4_scratch_slot(Self::G4_X, n * 4)?;
        for il in 0..self.n_layers {
            let l = self.attn.layers[il];
            let d = l.d;
            let qd = dm.q_heads * d;
            let nb = il * nw_per_layer;
            let sw = if l.sliding { dm.sliding_window } else { 0 };
            let a = self.g4_scratch_slot(Self::G4_A, n * 4)?;
            self.rms_rows_dev(nb, x, a, t, eps)?;
            let (q, k, v) = if t <= 16 {
                self.gemv_multi_g32_t_dev(il, 0, a, t)?;
                let q = self.g4_scratch_slot(Self::G4_B1, t * qd * 4)?;
                let k = self.g4_scratch_slot(Self::G4_B2, t * qd * 4)?;
                let v = if l.k_eq_v {
                    k
                } else {
                    self.g4_scratch_slot(Self::G4_B3, t * qd * 4)?
                };
                (q, k, v)
            } else {
                let q =
                    self.g4_dev_gemm(&format!("blk.{il}.attn_q.weight"), a, t, h, Self::G4_B1)?;
                let k =
                    self.g4_dev_gemm(&format!("blk.{il}.attn_k.weight"), a, t, h, Self::G4_B2)?;
                let v = if l.k_eq_v {
                    k
                } else {
                    self.g4_dev_gemm(&format!("blk.{il}.attn_v.weight"), a, t, h, Self::G4_B3)?
                };
                (q, k, v)
            };
            let sp_dec = if t <= 8 { ATTN_SPLITS_D } else { 2 };
            let outv = self.attn_g4_dev(
                slots, pos, il, q, k, v, dm.q_heads, l.kv_heads, d, sw, l.kq_scale, sp_dec, 5, 6, 7,
            )?;
            let o = self.g4_dev_gemm(&format!("blk.{il}.attn_output.weight"), outv, t, qd, 8)?;
            let hh = self.g4_scratch_slot(9, n * 4)?;
            self.rms_rows_dev(nb + 1, o, hh, t, eps)?;
            self.axpy_dev(1.0, hh, x, n)?;
            self.rms_rows_dev(nb + 2, x, a, t, eps)?;
            let (_, _, _, n_ff) = self.lin_spec(&format!("blk.{il}.ffn_down.weight"))?;
            let (g, u) = if t <= 16 {
                self.gemv_multi_g32_t_dev(il, 1, a, t)?;
                (
                    self.g4_scratch_slot(Self::G4_B1, t * n_ff * 4)?,
                    self.g4_scratch_slot(Self::G4_B2, t * n_ff * 4)?,
                )
            } else {
                (
                    self.g4_dev_gemm(&format!("blk.{il}.ffn_gate.weight"), a, t, h, Self::G4_B1)?,
                    self.g4_dev_gemm(&format!("blk.{il}.ffn_up.weight"), a, t, h, Self::G4_B2)?,
                )
            };
            let act = self.g4_dev_gelu(g, u, Self::G4_B3, t * n_ff)?;
            let m = self.g4_dev_gemm(&format!("blk.{il}.ffn_down.weight"), act, t, n_ff, 8)?;
            let h1 = self.g4_scratch_slot(9, n * 4)?;
            self.rms_rows_dev(nb + 3, m, h1, t, eps)?;
            let hn = self.g4_scratch_slot(Self::G4_HN, n * 4)?;
            let nm = self.g4_scratch_slot(Self::G4_NM, n * 4)?;
            self.rms_rows_dev(nb + 7, x, hn, t, eps)?;
            self.rms_rows_dev(nb + 4, x, nm, t, eps)?;
            let pes = self.g4_pes_dev(il);
            if pes == 0 {
                return Err(format!("g4_chunk_dev: L{il} pes 미등록(set_g4_pes 선행)"));
            }
            let moe = self.g4_moe_dev_t(il, hn, nm, pes, t, Self::G4_MOE)?;
            let h2 = self.g4_scratch_slot(Self::G4_H2, n * 4)?;
            self.rms_rows_dev(nb + 5, moe, h2, t, eps)?;
            self.axpy_dev(1.0, h2, h1, n)?;
            let hf = self.g4_scratch_slot(Self::G4_HF, n * 4)?;
            self.rms_rows_dev(nb + 6, h1, hf, t, eps)?;
            self.axpy_dev(1.0, hf, x, n)?;
            let ls = *self
                .g4_ls
                .get(il)
                .ok_or("g4_chunk_dev: layer_scalar 미등록(set_g4_layer_scalar)")?;
            self.scale_dev(x, ls, n)?;
            if self.debug_layers {
                let mut rb = vec![0u8; h * 4];
                self.cc.d2h_async(rb.as_mut_ptr(), x, h * 4)?;
                self.cc.sync()?;
                let row: Vec<f32> = rb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = row.iter().any(|v| v.is_nan());
                eprintln!("gemma4[gpu-chunk] layer {il:>2} max|x|={mx:.4} nan={nan}");
            }
        }
        let hx = self.g4_scratch_slot(Self::G4_HEADX, n * 4)?;
        let out_row = self.n_layers * nw_per_layer;
        if head_last {
            let xr = x + ((t - 1) * h * 4) as u64;
            self.rms_rows_dev(out_row, xr, hx, 1, eps)?;
            self.g4_head_dev_t(hx, 1, Self::G4_HEADOUT)?;
        } else {
            self.rms_rows_dev(out_row, x, hx, t, eps)?;
            self.g4_head_dev_t(hx, t, Self::G4_HEADOUT)?;
        }
        Ok(())
    }

    fn scale_dev(&mut self, y: CUdeviceptr, s: f32, n: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_scale")?;
        let mut ss = s;
        let mut p_y = y;
        let mut nn = n as i32;
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut p_y, &mut ss, &mut nn),
        )
    }

    pub fn g4_moe_dev_t(
        &mut self,
        il: usize,
        dhn: CUdeviceptr,
        dxn: CUdeviceptr,
        pes_dev: CUdeviceptr,
        t: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let h = self.hidden;
        let n_exp = self.moe.n_experts;
        let tk = self.moe.top_k.max(1);
        let n_ff = self.moe.moe_ffn;
        if t == 0 || t > G4_CHUNK_MAX || dhn == 0 || dxn == 0 || pes_dev == 0 {
            return Err(format!(
                "g4_moe_dev: 계약 위반 t={t} dhn={dhn} dxn={dxn} pes={pes_dev}"
            ));
        }
        if n_exp == 0 || !self.moe.moe_resident {
            return Err("g4_moe_dev: 전문가 비상주(스트리밍) — t행 미지원(호출부 t=1 폴백)".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let rn = format!("blk.{il}.moe_gate.weight");
        if self.lins.contains_key(&rn) {
            self.gemv_t_rows(&rn, dhn, self.moe.drt.ptr, t)?;
        } else {
            self.plain_gemv_t_rows(&rn, dhn, self.moe.drt.ptr, t)?;
        }
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
        if self.route_capture {
            let mut buf = vec![0u32; t * tk];
            self.cc
                .d2h_async(buf.as_mut_ptr() as *mut u8, self.moe.moe_idx, buf.len() * 4)?;
            self.cc.sync()?;
            self.route_out.extend_from_slice(&buf);
        }
        {
            let ns = t * tk;
            let f = self.cc.function("w4a16_moe_pes_scale")?;
            let (mut p_w, mut p_i, mut p_p) = (self.moe.moe_wt, self.moe.moe_idx, pes_dev);
            let mut p_n = ns as i32;
            self.cc.launch(
                f,
                (ns as u32).div_ceil(256).max(1),
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_w, &mut p_i, &mut p_p, &mut p_n),
            )?;
        }
        let ns = t * tk;
        let base = il * n_exp * 3;
        let out = self.g4_scratch_slot(out_slot, t * h * 4)?;
        let mut marlin_done = false;
        if t > G4_TROW_MAX && n_exp <= 1024 {
            self.moe_align_launch(ns)?;
            marlin_done = self.marlin_moe_ffn_dev(il, dxn, t, h, n_ff, tk, true, out)?;
            if !marlin_done {
                let xp = self.cast_f16_dev(dxn, t * h)?;
                self.gemm_g32_mma_grp2_launch(
                    base,
                    xp,
                    h,
                    tk,
                    self.moe.exp.gate,
                    self.moe.exp.up,
                    n_ff,
                    h,
                )?;
                self.ew_gelu_dev(
                    self.moe.exp.gate,
                    self.moe.exp.up,
                    self.moe.exp.act,
                    ns * n_ff,
                )?;
                let ap = self.cast_f16_dev(self.moe.exp.act, ns * n_ff)?;
                self.gemm_g32_mma_grp_d64_launch(base + 2, ap, n_ff, 1, self.moe.exp.dn, h, n_ff)?;
            }
        } else {
            self.gemv_experts_glu_launch(base, ns, dxn, h, tk, self.moe.exp.act, n_ff, h)?;
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
        if !marlin_done {
            self.moe_accum_dev(self.moe.moe_wt, self.moe.exp.dn, out, tk, ns, h)?;
        }
        Ok(out)
    }

    pub fn g4_head_dev_t(
        &mut self,
        x_dev: CUdeviceptr,
        t: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        if self.head_w == 0 {
            return Err("g4_head_dev: head 미등록".into());
        }
        if t == 0 || t > G4_CHUNK_MAX || x_dev == 0 {
            return Err(format!("g4_head_dev: t={t} x={x_dev}"));
        }
        let y = self.g4_scratch_slot(out_slot, t * self.head_n * 4)?;
        let mut r0 = 0usize;
        while r0 < t {
            let rows = G4_TROW_MAX.min(t - r0);
            if rows == 1 {
                self.head_gemv_i8_launch(
                    x_dev + (r0 * self.hidden * 4) as u64,
                    y + (r0 * self.head_n * 4) as u64,
                )?;
            } else {
                self.head_gemv_t_launch(
                    x_dev + (r0 * self.hidden * 4) as u64,
                    y + (r0 * self.head_n * 4) as u64,
                    rows,
                )?;
            }
            r0 += rows;
        }
        Ok(y)
    }

    pub fn g4_chunk_dev(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        rows: &[f32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<Vec<Vec<f32>>, String> {
        let h = self.hidden;
        if t == 0 || t > G4_CHUNK_MAX || slots.len() != t || pos.len() != t || rows.len() != t * h {
            return Err(format!(
                "g4_chunk_dev: t={t} rows={} slots={} pos={} (1..=16 · t*h 계약)",
                rows.len(),
                slots.len(),
                pos.len()
            ));
        }
        if self.head_w == 0 {
            return Err("g4_chunk_dev: head 미등록(upload_head 선행)".into());
        }
        let _g = self.cc.guard()?;
        if self.cc.stream.is_null() {
            self.cc.create_stream()?;
        }
        self.g4_warm_slots(G4_CHUNK_MAX)?;
        self.ensure_moe_bufs()?;
        self.g4_atab_prepare(slots, pos)?;
        let x = self.g4_scratch_slot(Self::G4_X, t * h * 4)?;
        let xb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d_async(x, xb)?;
        self.g4_ensure_pin_out(G4_CHUNK_MAX * self.head_n * 4)?;
        if self.g4_graph_ok() {
            if let Err(e) = self.g4_graph_run(slots, pos, t, head_last, nw_per_layer, eps) {
                eprintln!("[g4-graph] 실패 — 직접 경로 폴백: {e}");
                self.graph.g4_capture_failed = true;
                let _ = self.cc.sync();
                self.g4_run_direct(slots, pos, t, head_last, nw_per_layer, eps)?;
            }
        } else {
            self.g4_run_direct(slots, pos, t, head_last, nw_per_layer, eps)?;
        }
        self.cc.sync()?;
        let nrow = if head_last { 1 } else { t };
        let flat = unsafe {
            std::slice::from_raw_parts(self.graph.pin_g4_out as *const f32, nrow * self.head_n)
        };
        Ok(flat
            .chunks(self.head_n)
            .map(|r| r.to_vec())
            .collect::<Vec<_>>())
    }

    fn g4_graph_ok(&self) -> bool {
        llm170_diag::flag::ne0("LLM170_GRAPH")
            && !self.debug_layers
            && !self.route_capture
            && !self.graph.g4_capture_failed
            && self.moe.n_experts > 0
            && self.moe.moe_resident
    }

    fn g4_run_direct(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.g4_chunk_body(slots, pos, t, head_last, nw_per_layer, eps)?;
        self.g4_out_d2h(t, head_last)
    }

    fn g4_out_d2h(&mut self, t: usize, head_last: bool) -> Result<(), String> {
        let nrow = if head_last { 1 } else { t };
        let hd = self.g4_scratch_slot(Self::G4_HEADOUT, t * self.head_n * 4)?;
        self.cc
            .d2h_async(self.graph.pin_g4_out as *mut u8, hd, nrow * self.head_n * 4)
    }

    fn g4_graph_run(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        if let Some(k) = self
            .graph
            .g4_graphs
            .iter()
            .position(|e| e.t == t && e.head_last == head_last && e.slots == slots)
        {
            let exec = self.graph.g4_graphs[k].exec;
            return self.cc.graph_launch(exec);
        }
        self.g4_run_direct(slots, pos, t, head_last, nw_per_layer, eps)?;
        self.cc.sync()?;
        if !self.graph.g4_capture_failed {
            match self.g4_capture(slots, pos, t, head_last, nw_per_layer, eps) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# g4 그래프 캡처 실패 — 직접 발사 유지: {e}");
                    self.graph.g4_capture_failed = true;
                    let _ = self.cc.capture_end();
                    let _ = self.cc.sync();
                }
            }
        }
        Ok(())
    }

    fn g4_capture(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        let tt = std::time::Instant::now();
        if self.graph.g4_graphs.len() >= 32 {
            let old = self.graph.g4_graphs.remove(0);
            let _ = self.cc.graph_destroy(old.exec, old.handle);
        }
        self.cc.capture_begin()?;
        let r = self
            .g4_chunk_body(slots, pos, t, head_last, nw_per_layer, eps)
            .and_then(|_| self.g4_out_d2h(t, head_last));
        if let Err(e) = r {
            let _ = self.cc.capture_end();
            return Err(e);
        }
        let g = self.cc.capture_end()?;
        let e = match self.cc.graph_instantiate(g) {
            Ok(e) => e,
            Err(err) => {
                let _ = self.cc.graph_destroy(std::ptr::null_mut(), g);
                return Err(err);
            }
        };
        if let Err(err) = self.cc.graph_upload(e) {
            let _ = self.cc.graph_destroy(e, g);
            return Err(err);
        }
        self.graph.g4_graphs.push(G4GraphEntry {
            t,
            slots: slots.to_vec(),
            head_last,
            exec: e,
            handle: g,
        });
        eprintln!(
            "[g4-graph] captured t={t} slots={slots:?} head_last={head_last} ({:.2}ms)",
            tt.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    fn g4_ensure_pin_out(&mut self, bytes: usize) -> Result<(), String> {
        if !self.graph.pin_g4_out.is_null() && self.graph.pin_g4_out_len >= bytes {
            return Ok(());
        }
        self.graph_invalidate();
        if !self.graph.pin_g4_out.is_null() {
            let _ = self.cc.pinned_free(self.graph.pin_g4_out);
        }
        self.graph.pin_g4_out = self.cc.pinned_alloc(bytes)?;
        self.graph.pin_g4_out_len = bytes;
        Ok(())
    }

    pub fn head_logits_host(&mut self, xn: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("head_logits_host: head 미등록 — upload_head 선행".into());
        }
        if xn.len() != self.hidden {
            return Err(format!(
                "head_logits_host: xn={} != hidden={}",
                xn.len(),
                self.hidden
            ));
        }
        let _g = self.cc.guard()?;
        let dx = self.cc.alloc(self.hidden * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            let b: Vec<u8> = xn.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.cc.h2d(dx, &b)?;
            self.head_gemv_launch(dx, self.head_out)?;
            let mut ob = vec![0u8; self.head_n * 4];
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
            self.cc.sync()?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dx);
        r
    }
}
