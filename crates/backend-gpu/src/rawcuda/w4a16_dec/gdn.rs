use super::*;

pub(super) fn gdn_state_offsets(dm: &GdnDims, slot: usize) -> (usize, usize) {
    (
        slot * dm.n_gdn * dm.h_v * 128 * 128,
        slot * dm.n_gdn * dm.h_v * 128,
    )
}

impl W4a16Dec {
    pub(super) fn gdn_chain_dev(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if layer >= dm.n_gdn || slot >= self.n_slots || t_len == 0 {
            return Err(format!(
                "GDN: 범위 위반 layer={layer} slot={slot} t={t_len}"
            ));
        }
        self.ensure_gdn_bufs(t_len)?;
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let (st_slot, sc_slot) = gdn_state_offsets(&dm, slot);
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut hk, mut hv, mut dd) = (dm.h_k as i32, dm.h_v as i32, dm.d as i32);
        let (mut hd, mut kl, mut vl, mut cch) = (
            dm.hidden as i32,
            dm.k_len() as i32,
            dm.v_len() as i32,
            dm.conv_ch() as i32,
        );

        let f = self.cc.function("gdn_conv")?;
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (
            qkv_dev,
            self.gdn.dcw,
            self.gdn.dring + (ring_slot as u64) * 4,
            self.gdn.dgq,
            self.gdn.dgk,
            self.gdn.dgv,
        );
        let mut c6 = if self.spec_on && (2..=8).contains(&t_len) {
            self.gdn.dsnap_ring + (layer as u64 * 8 * 3 * dm.conv_ch() as u64) * 4
        } else {
            0
        };
        let mut c7 = if c6 != 0 && self.gdn.dring0 != 0 {
            self.gdn.dring0 + (layer as u64 * 3 * dm.conv_ch() as u64) * 4
        } else {
            0
        };
        let mut c8 = 0u64;
        self.cc.launch(
            f,
            (dm.conv_ch() / 128) as u32,
            (t_len.div_ceil(16)) as u32,
            128,
            &mut crate::rawcuda::args::l14(
                &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7, &mut c8,
                &mut tl, &mut lay, &mut kl, &mut vl, &mut cch,
            ),
        )?;

        let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6, mut l7, mut l8, mut l9, mut l10) = (
            self.gdn.dgq,
            self.gdn.dgk,
            self.gdn.dgv,
            xn_dev,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dq2,
            self.gdn.dk2,
            self.gdn.dv2,
            self.gdn.dbg,
        );
        if t_len >= 4 {
            let f = self.cc.function("gdn_l2perm8")?;
            let smem = 2 * dm.hidden * 4 + 8 * 128 * 4;
            self.cc.set_dynamic_smem(f, smem as u32)?;
            self.cc.launch_shared(
                f,
                dm.h_v as u32,
                t_len.div_ceil(8) as u32,
                1024,
                smem as u32,
                &mut crate::rawcuda::args::l16(
                    &mut l0, &mut l1, &mut l2, &mut l3, &mut l4, &mut l5, &mut l6, &mut l7,
                    &mut l8, &mut l9, &mut l10, &mut tl, &mut lay, &mut hk, &mut hv, &mut hd,
                ),
            )?;
        } else {
            let f = self.cc.function("gdn_l2perm")?;
            self.cc.launch(
                f,
                dm.h_v as u32,
                t_len as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut l0, &mut l1, &mut l2, &mut l3, &mut l4, &mut l5, &mut l6, &mut l7,
                    &mut l8, &mut l9, &mut l10, &mut tl, &mut lay, &mut hk, &mut hv, &mut hd,
                ),
            )?;
        }

        let mut skip_scan = false;
        if t_len == 1 {
            let (mut n0, mut n1) = (0u64, 0u64);
            let fp = self.cc.function("gdn1_part")?;
            let (mut p0, mut p1, mut p2, mut p3, mut p4) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dgst + (st_slot as u64) * 2,
                self.gdn.dsc + (sc_slot as u64) * 4,
                self.gdn.dgpart,
            );
            self.cc.launch(
                fp,
                dm.h_v as u32,
                8,
                128,
                &mut crate::rawcuda::args::l11(
                    &mut p0, &mut p1, &mut p2, &mut p3, &mut p4, &mut n0, &mut n1, &mut hk,
                    &mut hv, &mut dd, &mut lay,
                ),
            )?;
            let fc = self.cc.function("gdn1_comb")?;
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dgpart,
                self.gdn.dgdc,
                self.gdn.dgo,
            );
            self.cc.launch(
                fc,
                dm.h_v as u32,
                1,
                128,
                &mut crate::rawcuda::args::l10(
                    &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut hk,
                    &mut hv, &mut dd,
                ),
            )?;
            let fu = self.cc.function("gdn1_upd")?;
            let (mut u0, mut u1, mut u2, mut u3, mut u4) = (
                self.gdn.dk2,
                self.gdn.dbg,
                self.gdn.dgdc,
                self.gdn.dgst + (st_slot as u64) * 2,
                self.gdn.dsc + (sc_slot as u64) * 4,
            );
            self.cc.launch(
                fu,
                dm.h_v as u32,
                8,
                128,
                &mut crate::rawcuda::args::l11(
                    &mut u0, &mut u1, &mut u2, &mut u3, &mut u4, &mut n0, &mut n1, &mut hk,
                    &mut hv, &mut dd, &mut lay,
                ),
            )?;
            skip_scan = true;
        }
        if !skip_scan && self.spec_on && self.spec_scan_on && (2..=8).contains(&t_len) {
            if self.gdn.dwy_s0 == 0 {
                return Err("GDN spec: enable_spec 미호출".into());
            }
            let fw = self.cc.function("gdn_wy_solve")?;
            let (mut w0, mut w1, mut w2, mut w3) = (
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dgst + (st_slot as u64) * 2,
            );
            let (mut w8, mut w4, mut w5, mut w6, mut w7) = (
                self.gdn.dsc + (sc_slot as u64) * 4,
                self.gdn.dwy_s0,
                self.gdn.dwy_dc,
                self.gdn.dwy_g,
                self.gdn.dwy_k,
            );
            self.cc.launch(
                fw,
                dm.h_v as u32,
                1,
                128,
                &mut crate::rawcuda::args::l14(
                    &mut w0, &mut w1, &mut w2, &mut w3, &mut w8, &mut w4, &mut w5, &mut w6,
                    &mut w7, &mut tl, &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
            let f = self.cc.function("gdn_spec_scan")?;
            let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dgst + (st_slot as u64) * 2,
                self.gdn.dgo,
                self.gdn.dsc + (sc_slot as u64) * 4,
            );
            self.cc.launch(
                f,
                dm.h_v as u32,
                1,
                128,
                &mut crate::rawcuda::args::l12(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s6, &mut s5, &mut tl,
                    &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
            skip_scan = true;
        }
        if !skip_scan {
            let nch = t_len.div_ceil(GDN_CS) as u32;
            let fp = self.cc.function("gdn_scan_akq")?;
            let (mut p0, mut p1, mut p2, mut p3) =
                (self.gdn.dq2, self.gdn.dk2, self.gdn.dbg, self.gdn.dakq);
            let (mut pt, mut phk, mut phv, mut pd) =
                (t_len as i32, dm.h_k as i32, dm.h_v as i32, dm.d as i32);
            self.cc.launch(
                fp,
                dm.h_v as u32,
                nch,
                512,
                &mut crate::rawcuda::args::l8(
                    &mut p0, &mut p1, &mut p2, &mut p3, &mut pt, &mut phk, &mut phv, &mut pd,
                ),
            )?;

            let fw = self.cc.function("gdn_wu")?;
            self.cc.set_dynamic_smem(fw, GDN_WU_SMEM)?;
            let nch_w = t_len.div_ceil(GDN_CS) as u32;
            let (mut w0, mut w1, mut w2, mut w3, mut w4) = (
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dakq,
                self.gdn.dwu,
            );
            self.cc.launch_shared(
                fw,
                nch_w,
                dm.h_v as u32,
                (2 * dm.d) as u32,
                GDN_WU_SMEM,
                &mut crate::rawcuda::args::l9(
                    &mut w0, &mut w1, &mut w2, &mut w3, &mut w4, &mut tl, &mut hk, &mut hv, &mut dd,
                ),
            )?;
            let fh = self.cc.function("gdn_h")?;
            self.cc.set_dynamic_smem(fh, GDN_H_SMEM)?;
            let (mut h0, mut h1, mut h2, mut h3, mut h4, mut h5, mut h6) = (
                self.gdn.dk2,
                self.gdn.dbg,
                self.gdn.dwu,
                self.gdn.dgst + (st_slot as u64) * 2,
                self.gdn.dsc + (sc_slot as u64) * 4,
                self.gdn.dhbuf,
                self.gdn.ddcbuf,
            );
            self.cc.launch_shared(
                fh,
                (dm.h_v * GDN_NSPLIT) as u32,
                1,
                (GDN_NGRP * (128 / GDN_NSPLIT)) as u32,
                GDN_H_SMEM,
                &mut crate::rawcuda::args::l12(
                    &mut h0, &mut h1, &mut h2, &mut h3, &mut h4, &mut h5, &mut h6, &mut tl,
                    &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
            let fo = self.cc.function("gdn_o")?;
            self.cc.set_dynamic_smem(fo, GDN_O_SMEM)?;
            let (mut o0, mut o1, mut o2, mut o3, mut o4, mut o5, mut o6) = (
                self.gdn.dq2,
                self.gdn.dbg,
                self.gdn.dakq,
                self.gdn.ddcbuf,
                self.gdn.dhbuf,
                self.gdn.dsc + (sc_slot as u64) * 4,
                self.gdn.dgo,
            );
            self.cc.launch_shared(
                fo,
                nch_w * dm.h_v as u32,
                1,
                1024,
                GDN_O_SMEM,
                &mut crate::rawcuda::args::l12(
                    &mut o0, &mut o1, &mut o2, &mut o3, &mut o4, &mut o5, &mut o6, &mut tl,
                    &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
        }

        let ng = t_len * dm.v_len();
        let (g32, g16) = (self.ensure_dx32(ng)?, self.ensure_dx16(ng)?);
        let f = self.cc.function(if self.gdn.sig_gate {
            "gdn_gate_sig"
        } else {
            "gdn_gate"
        })?;
        let (mut g0, mut g1, mut g2, mut g3) = (self.gdn.dgo, z_dev, self.gdn.dnwg, self.gdn.dgate);
        let (mut g4, mut g5) = (g32, g16);
        self.cc.launch(
            f,
            dm.h_v as u32,
            t_len as u32,
            128,
            &mut crate::rawcuda::args::l10(
                &mut g0, &mut g1, &mut g2, &mut g3, &mut g4, &mut g5, &mut tl, &mut lay, &mut hk,
                &mut hv,
            ),
        )?;
        if layer == dm.n_gdn - 1 {
            self.gdn_rescale_all(slot)?;
        }
        Ok(())
    }

    pub(super) fn gdn_rescale_slot(&self, slot: usize, layer: usize) -> Result<(), String> {
        self.gdn_rescale_layers(slot, layer, 1)
    }

    pub(super) fn gdn_rescale_all(&self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        self.gdn_rescale_layers(slot, 0, dm.n_gdn)
    }

    fn gdn_rescale_layers(
        &self,
        slot: usize,
        layer0: usize,
        n_layers: usize,
    ) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let f = self.cc.function("gdn_rescale")?;
        let (st_slot, sc_slot) = gdn_state_offsets(&dm, slot);
        let (mut p0, mut p1) = (
            self.gdn.dgst + st_slot as u64 * 2,
            self.gdn.dsc + sc_slot as u64 * 4,
        );
        let (mut hv, mut dd, mut lay) = (dm.h_v as i32, dm.d as i32, layer0 as i32);
        let mut head = 256.0f32;
        self.cc.launch(
            f,
            dm.h_v as u32,
            n_layers as u32,
            128,
            &mut crate::rawcuda::args::l6(&mut p0, &mut p1, &mut hv, &mut dd, &mut lay, &mut head),
        )
    }

    pub fn gdn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>, String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if xn.len() != t_len * dm.hidden
            || qkv.len() != t_len * dm.conv_ch()
            || z.len() != t_len * dm.v_len()
        {
            return Err("GDN: 입력 형상 계약 위반".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.gdn.dgxn, b(xn))?;
        self.cc.h2d(self.gdn.dqkv, b(qkv))?;
        self.cc.h2d(self.gdn.dzv, b(z))?;
        let (gxn, gqkv, gzv) = (self.gdn.dgxn, self.gdn.dqkv, self.gdn.dzv);
        self.gdn_chain_dev(slot, layer, t_len, gxn, gqkv, gzv)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.gdn.dgate)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()) }
                .to_vec(),
        )
    }


    pub fn spec_on(&self) -> bool {
        self.spec_on
    }

    pub fn enable_spec(&mut self) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.moe.n_experts > 0 && !self.moe.moe_resident {
            return Err("스트리밍(비상주 MoE) — 검증 체인 미지원(t>1 상주 전용)".into());
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if self.gdn.dwy_s0 != 0 {
            self.spec_on = true;
            return Ok(());
        }
        let st_elems = dm.n_gdn * dm.h_v * dm.d * dm.d;
        self.gdn.dwy_s0 = self.cc.alloc(st_elems * 4)?;
        self.gdn.dwy_dc = self.cc.alloc(8 * dm.n_gdn * dm.h_v * dm.d * 4)?;
        self.gdn.dwy_g = self.cc.alloc(8 * dm.n_gdn * dm.h_v * 4)?;
        self.gdn.dwy_k = self.cc.alloc(8 * dm.n_gdn * dm.k_len() * 4)?;
        let ring_snap = 8 * dm.n_gdn * 3 * dm.conv_ch() * 4;
        self.gdn.dsnap_ring = self.cc.alloc(ring_snap)?;
        let ring_pre = dm.n_gdn * 3 * dm.conv_ch() * 4;
        self.gdn.dring0 = self.cc.alloc(ring_pre)?;
        let spec_bytes = (st_elems * 4
            + 8 * dm.n_gdn * dm.h_v * dm.d * 4
            + 8 * dm.n_gdn * dm.h_v * 4
            + 8 * dm.n_gdn * dm.k_len() * 4
            + ring_snap
            + ring_pre) as u64;
        if let Err(e) = self.ucache.allocate(
            "spec:state",
            CacheCategory::SpecState,
            Tier::Hbm,
            spec_bytes,
            base_util::SPEC_STATE,
        ) {
            eprintln!("[ucache] spec:state 등록 실패(회계 전용): {e}");
        } else {
            eprintln!("{}", self.ucache.log_line("spec:state"));
        }
        let chk = 4
            * self.n_slots
            * (dm.n_gdn * dm.h_v * dm.d * dm.d * 2
                + dm.n_gdn * dm.h_v * dm.d * 4
                + dm.n_gdn * 3 * dm.conv_ch() * 4);
        match self.cc.pinned_alloc(chk) {
            Ok(p) => {
                self.gdn.dchk = p;
                self.gdn.dchk_pos = vec![[0u32; 4]; self.n_slots];
                self.gdn.dchk_saved = vec![[0u32; 4]; self.n_slots];
                self.gdn.dchk_use = vec![[0u64; 4]; self.n_slots];
                self.gdn.dchk_tick = 0;
                if let Err(e) = self.ucache.allocate(
                    "spec:ckpt",
                    CacheCategory::Checkpoint,
                    Tier::HostRam,
                    chk as u64,
                    base_util::CKPT,
                ) {
                    eprintln!("[ucache] spec:ckpt 등록 실패(회계 전용): {e}");
                } else {
                    eprintln!("{}", self.ucache.log_line("spec:ckpt"));
                }
            }
            Err(e) => {
                eprintln!("# prefix-ckpt 비활성(VRAM): {e}");
            }
        }
        self.spec_on = true;
        self.ucache_sweep()?;
        if self.gdn.dwy_s0 == 0 {
            self.spec_on = false;
        }
        Ok(())
    }

    pub(super) fn gdn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        self.gdn_chain_dev(slot, layer, t_len, xn_dev, qkv_dev, z_dev)?;
        Ok(self.gdn.dgate)
    }

    pub(super) fn gdn_chain_dev_batch(
        &mut self,
        slots: &[usize],
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if t_len == 0 || t_len != slots.len() || slots.iter().any(|&s| s >= self.n_slots) {
            return Err("GDN batch: 슬롯/토큰 계약 위반".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let (mut kl, mut vl, mut cch, mut hv, mut hd) = (
            dm.k_len() as i32,
            dm.v_len() as i32,
            dm.conv_ch() as i32,
            dm.h_v as i32,
            dm.hidden as i32,
        );
        let (mut hk, mut dd) = (dm.h_k as i32, dm.d as i32);
        let mut lay = layer as i32;
        let mut one = 1i32;
        let (row_ring, row_st, row_sc) = (self.drow_ring, self.drow_st, self.drow_sc);
        let f = self.cc.function("gdn_conv")?;
        {
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (
                qkv_dev,
                self.gdn.dcw,
                self.gdn.dring,
                self.gdn.dgq,
                self.gdn.dgk,
                self.gdn.dgv,
            );
            let (mut c6, mut c7, mut c8) = (0u64, 0u64, row_ring);
            self.cc.launch(
                f,
                (dm.conv_ch() / 128) as u32,
                t_len as u32,
                128,
                &mut crate::rawcuda::args::l14(
                    &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7,
                    &mut c8, &mut one, &mut lay, &mut kl, &mut vl, &mut cch,
                ),
            )?;
        }
        {
            let f = self.cc.function("gdn_l2perm")?;
            let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6) = (
                self.gdn.dgq,
                self.gdn.dgk,
                self.gdn.dgv,
                xn_dev,
                self.gdn.dab_c,
                self.gdn.dalog,
                self.gdn.ddtb,
            );
            let (mut l7, mut l8, mut l9, mut l10) =
                (self.gdn.dq2, self.gdn.dk2, self.gdn.dv2, self.gdn.dbg);
            let mut tl = t_len as i32;
            self.cc.launch(
                f,
                dm.h_v as u32,
                t_len as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut l0, &mut l1, &mut l2, &mut l3, &mut l4, &mut l5, &mut l6, &mut l7,
                    &mut l8, &mut l9, &mut l10, &mut tl, &mut lay, &mut hk, &mut hv, &mut hd,
                ),
            )?;
        }
        let (fp, fc, fu) = (
            self.cc.function("gdn1_part")?,
            self.cc.function("gdn1_comb")?,
            self.cc.function("gdn1_upd")?,
        );
        {
            let (mut p0, mut p1, mut p2, mut p3, mut p4) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dgst,
                self.gdn.dsc,
                self.gdn.dgpart,
            );
            let (mut p5, mut p6) = (row_st, row_sc);
            self.cc.launch(
                fp,
                dm.h_v as u32,
                (8 * t_len) as u32,
                128,
                &mut crate::rawcuda::args::l11(
                    &mut p0, &mut p1, &mut p2, &mut p3, &mut p4, &mut p5, &mut p6, &mut hk,
                    &mut hv, &mut dd, &mut lay,
                ),
            )?;
        }
        {
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dgpart,
                self.gdn.dgdc,
                self.gdn.dgo,
            );
            self.cc.launch(
                fc,
                dm.h_v as u32,
                t_len as u32,
                128,
                &mut crate::rawcuda::args::l10(
                    &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut hk,
                    &mut hv, &mut dd,
                ),
            )?;
        }
        {
            let (mut u0, mut u1, mut u2, mut u3, mut u4) = (
                self.gdn.dk2,
                self.gdn.dbg,
                self.gdn.dgdc,
                self.gdn.dgst,
                self.gdn.dsc,
            );
            let (mut u5, mut u6) = (row_st, row_sc);
            self.cc.launch(
                fu,
                dm.h_v as u32,
                (8 * t_len) as u32,
                128,
                &mut crate::rawcuda::args::l11(
                    &mut u0, &mut u1, &mut u2, &mut u3, &mut u4, &mut u5, &mut u6, &mut hk,
                    &mut hv, &mut dd, &mut lay,
                ),
            )?;
        }
        if layer == dm.n_gdn - 1 {
            for &slot in slots {
                self.gdn_rescale_all(slot)?;
            }
        }
        {
            let ng = t_len * dm.v_len();
            let (g32, g16) = (self.ensure_dx32(ng)?, self.ensure_dx16(ng)?);
            let f = self.cc.function(if self.gdn.sig_gate {
                "gdn_gate_sig"
            } else {
                "gdn_gate"
            })?;
            let (mut g0, mut g1, mut g2, mut g3) =
                (self.gdn.dgo, z_dev, self.gdn.dnwg, self.gdn.dgate);
            let (mut g4, mut g5) = (g32, g16);
            let mut tl = t_len as i32;
            self.cc.launch(
                f,
                dm.h_v as u32,
                t_len as u32,
                128,
                &mut crate::rawcuda::args::l10(
                    &mut g0, &mut g1, &mut g2, &mut g3, &mut g4, &mut g5, &mut tl, &mut lay,
                    &mut hk, &mut hv,
                ),
            )?;
        }
        Ok(self.gdn.dgate)
    }
}
