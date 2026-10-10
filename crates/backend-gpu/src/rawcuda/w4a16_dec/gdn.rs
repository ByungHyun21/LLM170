use super::*;

impl W4a16Dec {
    /// GDN 체인 디바이스 — [A3 2026-10-09] 입력(xn·qkv·z)을 스테이징 버퍼로
    /// d2d 복사하지 않고 **커널 인자로 직접 소비**한다(호출자 버퍼가 곧 입력).
    pub(super) fn gdn_chain_dev(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if layer >= dm.n_gdn || slot >= self.n_slots || t_len == 0 {
            return Err(format!(
                "GDN: 범위 위반 layer={layer} slot={slot} t={t_len}"
            ));
        }
        self.ensure_gdn_bufs(t_len)?;
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let st_slot = slot * dm.n_gdn * dm.h_v * 128 * 128;
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
            self.dcw,
            self.dring + (ring_slot as u64) * 4,
            self.dgq,
            self.dgk,
            self.dgv,
        );
        // [A-1] 스펙 검증이면 토큰별 링 스냅샷 기록(층 슬라이스 = layer×8×3×ch).
        let mut c6 = if self.spec_on && (2..=8).contains(&t_len) {
            self.dsnap_ring + (layer as u64 * 8 * 3 * dm.conv_ch() as u64) * 4
        } else {
            0
        };
        self.cc.launch(
            f,
            (dm.conv_ch() / 128) as u32,
            // [토큰축 병렬] y = ceil(t_len/16) — 4탭은 입력 이력만 필요.
            (t_len.div_ceil(16)) as u32,
            128,
            &mut crate::rawcuda::args::l12(
                &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut tl, &mut lay,
                &mut kl, &mut vl, &mut cch,
            ),
        )?;

        let f = self.cc.function("gdn_l2perm")?;
        let (
            mut l0,
            mut l1,
            mut l2,
            mut l3,
            mut l4,
            mut l5,
            mut l6,
            mut l7,
            mut l8,
            mut l9,
            mut l10,
        ) = (
            self.dgq, self.dgk, self.dgv, xn_dev, self.dab_c, self.dalog, self.ddtb, self.dq2,
            self.dk2, self.dv2, self.dbg,
        );
        self.cc.launch(
            f,
            dm.h_v as u32,
            t_len as u32,
            128,
            &mut crate::rawcuda::args::l16(
                &mut l0, &mut l1, &mut l2, &mut l3, &mut l4, &mut l5, &mut l6, &mut l7, &mut l8,
                &mut l9, &mut l10, &mut tl, &mut lay, &mut hk, &mut hv, &mut hd,
            ),
        )?;

        // [P9] t=1 전용 — i축 분할 3커널(grid h_v×4 = 192블록). ncu 실측
        // gdn_scan 점유 8.3%(지연 바운드) → 분할로 병렬도 확보. t>1은 종전.
        let mut skip_scan = false;
        if t_len == 1 {
            let fp = self.cc.function("gdn1_part")?;
            let (mut p0, mut p1, mut p2, mut p3) = (
                self.dq2,
                self.dk2,
                self.dgst + (st_slot as u64) * 4,
                self.dgpart,
            );
            self.cc.launch(
                fp,
                dm.h_v as u32,
                8,
                128,
                &mut crate::rawcuda::args::l8(
                    &mut p0, &mut p1, &mut p2, &mut p3, &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
            let fc = self.cc.function("gdn1_comb")?;
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) = (
                self.dq2,
                self.dk2,
                self.dv2,
                self.dbg,
                self.dgpart,
                self.dgdc,
                self.dgo,
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
            let (mut u0, mut u1, mut u2, mut u3) = (
                self.dk2,
                self.dbg,
                self.dgdc,
                self.dgst + (st_slot as u64) * 4,
            );
            self.cc.launch(
                fu,
                dm.h_v as u32,
                8,
                128,
                &mut crate::rawcuda::args::l8(
                    &mut u0, &mut u1, &mut u2, &mut u3, &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
            skip_scan = true;
        }
        if !skip_scan && self.spec_on && (2..=8).contains(&t_len) {
            // [A-1] 스펙 검증 — 토큰 루프 + 토큰별 상태 스냅샷(부분 수용
            // 롤백 지점). 산술은 t=1 trio와 비트동일.
            if self.dsnap == 0 {
                return Err("GDN spec: enable_spec 미호출".into());
            }
            let f = self.cc.function("gdn_spec_scan")?;
            let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6) = (
                self.dq2,
                self.dk2,
                self.dv2,
                self.dbg,
                self.dgst + (st_slot as u64) * 4,
                self.dsnap,
                self.dgo,
            );
            let (mut tl, mut hk, mut hv, mut dd) =
                (t_len as i32, dm.h_k as i32, dm.h_v as i32, dm.d as i32);
            let (mut lay, mut nl) = (layer as i32, self.n_layers as i32);
            self.cc.launch(
                f,
                dm.h_v as u32,
                1,
                128,
                &mut crate::rawcuda::args::l13(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut tl,
                    &mut hk, &mut hv, &mut dd, &mut lay, &mut nl,
                ),
            )?;
            return Ok(());
        }
        if !skip_scan {
            // [A5-4] FLA 2단: A/KQ 청크 병렬 prepass(값 비트동일) →
            // 상태/출력 V-타일 스캔. 스크래치는 dakq(로드 시 확보).
            let nch = t_len.div_ceil(GDN_CS) as u32;
            let fp = self.cc.function("gdn_scan_akq")?;
            let (mut p0, mut p1, mut p2, mut p3) = (self.dq2, self.dk2, self.dbg, self.dakq);
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

            let f = self.cc.function("gdn_scan")?;
            self.cc.set_dynamic_smem(f, GDN_SCAN_SMEM)?;
            let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6) = (
                self.dq2,
                self.dk2,
                self.dv2,
                self.dbg,
                self.dakq,
                self.dgst + (st_slot as u64) * 4,
                self.dgo,
            );
            // grid = h_v×NSPLIT(블록 = GDN_NGRP×GDN_VS = 512스레드).
            self.cc.launch_shared(
                f,
                (dm.h_v * GDN_NSPLIT) as u32,
                1,
                (GDN_NGRP * (128 / GDN_NSPLIT)) as u32,
                GDN_SCAN_SMEM,
                &mut crate::rawcuda::args::l12(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut tl,
                    &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
        }

        let f = self.cc.function("gdn_gate")?;
        let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, z_dev, self.dnwg, self.dgate);
        self.cc.launch(
            f,
            dm.h_v as u32,
            t_len as u32,
            128,
            &mut crate::rawcuda::args::l8(
                &mut g0, &mut g1, &mut g2, &mut g3, &mut tl, &mut lay, &mut hk, &mut hv,
            ),
        )?;
        Ok(())
    }

    /// GDN 체인 호스트 진입 — xn·qkv·z 업로드 → 4커널 → gated 판독.
    pub fn gdn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if xn.len() != t_len * dm.hidden
            || qkv.len() != t_len * dm.conv_ch()
            || z.len() != t_len * dm.v_len()
        {
            return Err("GDN: 입력 형상 계약 위반".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dgxn, b(xn))?;
        self.cc.h2d(self.dqkv, b(qkv))?;
        self.cc.h2d(self.dzv, b(z))?;
        let (gxn, gqkv, gzv) = (self.dgxn, self.dqkv, self.dzv);
        self.gdn_chain_dev(slot, layer, t_len, gxn, gqkv, gzv)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        Ok(
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()) }
                .to_vec(),
        )
    }

    // ── 어텐션 ──

    /// GDN 체인 디바이스 상주 — xn·qkv·z(디바이스) → dgate.
    /// [A-1] 스펙 검증 활성화 — GDN 토큰별 스냅샷 버퍼 확보(KMAX=8 고정).
    /// VRAM: 8 × n_layers × h_v × 128×128 × 4B (35B 436MB · 27B 654MB).
    pub fn enable_spec(&mut self) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if self.dsnap != 0 {
            self.spec_on = true;
            return Ok(());
        }
        let elems = 8 * self.n_layers * dm.h_v * dm.d * dm.d;
        self.dsnap = self.cc.alloc(elems * 4)?;
        self.dsnap_ring = self.cc.alloc(8 * self.n_layers * 3 * dm.conv_ch() * 4)?;
        self.spec_on = true;
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
        // [A3] 스테이징 d2d 3회/층 제거 — 호출자 버퍼를 커널 인자로 직접 소비.
        // (gdn_chain_dev가 형상·범위 검증·버퍼 보장을 겸한다.)
        self.gdn_chain_dev(slot, layer, t_len, xn_dev, qkv_dev, z_dev)?;
        Ok(self.dgate)
    }

    /// [A9 2026-10-10] GDN 디코드 배치 — 토큰별 슬롯 상태(링·스캔)를 쓴다.
    /// 청크 스캔은 혼합 슬롯에서 의미론이 깨지므로(타 슬롯 토큰과 intra-chunk
    /// 어텐션) t=1 트리오(gdn1_*)를 토큰 수만큼 발사한다 — 행 단위 커널
    /// (l2perm·gate)만 1회. 각 토큰 산술은 단독 t=1 경로와 동일(골든 계약).
    pub(super) fn gdn_chain_dev_batch(
        &mut self,
        slots: &[usize],
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
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
        let st_stride = (dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let ring_stride = (dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let bg_stride = dm.bg_len() as u64;
        // conv — 토큰별(링 = 슬롯).
        let f = self.cc.function("gdn_conv")?;
        for (k, &slot) in slots.iter().enumerate() {
            let mut c0 = qkv_dev + k as u64 * cch as u64 * 4;
            let mut c1 = self.dcw;
            let mut c2 = self.dring + (slot as u64 * ring_stride) * 4;
            let mut c3 = self.dgq + k as u64 * kl as u64 * 4;
            let mut c4 = self.dgk + k as u64 * kl as u64 * 4;
            let mut c5 = self.dgv + k as u64 * vl as u64 * 4;
            let mut c6 = 0u64;
            self.cc.launch(
                f,
                (dm.conv_ch() / 128) as u32,
                1,
                128,
                &mut crate::rawcuda::args::l12(
                    &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut one,
                    &mut lay, &mut kl, &mut vl, &mut cch,
                ),
            )?;
        }
        // l2perm — 행 단위 1회.
        {
            let f = self.cc.function("gdn_l2perm")?;
            let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6) = (
                self.dgq, self.dgk, self.dgv, xn_dev, self.dab_c, self.dalog, self.ddtb,
            );
            let (mut l7, mut l8, mut l9, mut l10) = (self.dq2, self.dk2, self.dv2, self.dbg);
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
        // t=1 트리오 — 토큰별(상태 = 슬롯).
        let part_stride = (dm.h_v * 8 * 256) as u64;
        let dc_stride = (dm.h_v * 128) as u64;
        let (fp, fc, fu) = (
            self.cc.function("gdn1_part")?,
            self.cc.function("gdn1_comb")?,
            self.cc.function("gdn1_upd")?,
        );
        for (k, &slot) in slots.iter().enumerate() {
            let q2k = self.dq2 + k as u64 * kl as u64 * 4;
            let k2k = self.dk2 + k as u64 * kl as u64 * 4;
            let v2k = self.dv2 + k as u64 * vl as u64 * 4;
            let bgk = self.dbg + k as u64 * bg_stride * 4;
            let stk = self.dgst + slot as u64 * st_stride * 4;
            let partk = self.dgpart + k as u64 * part_stride * 4;
            let dck = self.dgdc + k as u64 * dc_stride * 4;
            let outk = self.dgo + k as u64 * vl as u64 * 4;
            {
                let (mut p0, mut p1, mut p2, mut p3) = (q2k, k2k, stk, partk);
                self.cc.launch(
                    fp,
                    dm.h_v as u32,
                    8,
                    128,
                    &mut crate::rawcuda::args::l8(
                        &mut p0, &mut p1, &mut p2, &mut p3, &mut hk, &mut hv, &mut dd, &mut lay,
                    ),
                )?;
            }
            {
                let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) =
                    (q2k, k2k, v2k, bgk, partk, dck, outk);
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
            }
            {
                let (mut u0, mut u1, mut u2, mut u3) = (k2k, bgk, dck, stk);
                self.cc.launch(
                    fu,
                    dm.h_v as u32,
                    8,
                    128,
                    &mut crate::rawcuda::args::l8(
                        &mut u0, &mut u1, &mut u2, &mut u3, &mut hk, &mut hv, &mut dd, &mut lay,
                    ),
                )?;
            }
        }
        // gate — 행 단위 1회.
        {
            let f = self.cc.function("gdn_gate")?;
            let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, z_dev, self.dnwg, self.dgate);
            let mut tl = t_len as i32;
            self.cc.launch(
                f,
                dm.h_v as u32,
                t_len as u32,
                128,
                &mut crate::rawcuda::args::l8(
                    &mut g0, &mut g1, &mut g2, &mut g3, &mut tl, &mut lay, &mut hk, &mut hv,
                ),
            )?;
        }
        Ok(self.dgate)
    }
}
