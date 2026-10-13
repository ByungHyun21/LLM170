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
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
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
            self.gdn.dcw,
            self.gdn.dring + (ring_slot as u64) * 4,
            self.gdn.dgq,
            self.gdn.dgk,
            self.gdn.dgv,
        );
        // [A-1] 스펙 검증이면 토큰별 링 스냅샷 기록(층 슬라이스 = layer×8×3×ch).
        let mut c6 = if self.spec_on && (2..=8).contains(&t_len) {
            self.gdn.dsnap_ring + (layer as u64 * 8 * 3 * dm.conv_ch() as u64) * 4
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
            let (mut p0, mut p1, mut p2, mut p3, mut p4) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dgst + (st_slot as u64) * 2,
                self.gdn.dsc + (st_slot as u64) * 4,
                self.gdn.dgpart,
            );
            self.cc.launch(
                fp,
                dm.h_v as u32,
                8,
                128,
                &mut crate::rawcuda::args::l9(
                    &mut p0, &mut p1, &mut p2, &mut p3, &mut p4, &mut hk, &mut hv, &mut dd,
                    &mut lay,
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
                self.gdn.dsc + (st_slot as u64) * 4,
            );
            self.cc.launch(
                fu,
                dm.h_v as u32,
                8,
                128,
                &mut crate::rawcuda::args::l9(
                    &mut u0, &mut u1, &mut u2, &mut u3, &mut u4, &mut hk, &mut hv, &mut dd,
                    &mut lay,
                ),
            )?;
            skip_scan = true;
        }
        if !skip_scan && self.spec_on && self.spec_scan_on && (2..=8).contains(&t_len) {
            // [A-1] 스펙 검증 — 토큰 루프 순차 스캔(출력·최종 상태 — 산술은
            // t=1 trio와 비트동일) + [TreeWY 2.B 2026-10-12] WY solve(롤백
            // 데이터 — 토큰별 상태 스냅샷 dsnap 폐지, 27B 1.50GiB→약 164MB).
            if self.gdn.dwy_s0 == 0 {
                return Err("GDN spec: enable_spec 미호출".into());
            }
            // WY solve를 순차 스캔 **앞에** 발사 — 스캔이 상태를 전진시키므로
            // S_0 판독·저장이 먼저여야 한다(동일 스트림 순서 보장).
            let fw = self.cc.function("gdn_wy_solve")?;
            let (mut w0, mut w1, mut w2, mut w3) = (
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dgst + (st_slot as u64) * 2,
            );
            let (mut w8, mut w4, mut w5, mut w6, mut w7) = (
                self.gdn.dsc + (st_slot as u64) * 4,
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
                self.gdn.dsc + (st_slot as u64) * 4,
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
            // [결함수정] return으로 gdn_chain_dev 전체를 빠져나가면 **게이트가
            // 건너뛰어져**(dgate 미갱신) 검증 토큰이 붕괴한다(실측 328,73,73,73).
            // akq+scan만 건너뛰고 게이트로 진행한다.
            skip_scan = true;
        }
        if !skip_scan {
            // [A5-4] FLA 2단: A/KQ 청크 병렬 prepass(값 비트동일) →
            // 상태/출력 V-타일 스캔. 스크래치는 dakq(로드 시 확보).
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

            let f = self.cc.function("gdn_scan")?;
            self.cc.set_dynamic_smem(f, GDN_SCAN_SMEM)?;
            let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dakq,
                self.gdn.dgst + (st_slot as u64) * 2,
                self.gdn.dgo,
                self.gdn.dsc + (st_slot as u64) * 4,
            );
            // grid = h_v×NSPLIT(블록 = GDN_NGRP×GDN_VS = 512스레드).
            self.cc.launch_shared(
                f,
                (dm.h_v * GDN_NSPLIT) as u32,
                1,
                (GDN_NGRP * (128 / GDN_NSPLIT)) as u32,
                GDN_SCAN_SMEM,
                &mut crate::rawcuda::args::l13(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s7, &mut s6,
                    &mut tl, &mut hk, &mut hv, &mut dd, &mut lay,
                ),
            )?;
        }

        // [4.1 2026-10-11] gate가 x32/미러 동시 기록 — split 체인의 cast_x32
        // 노드 제거(값 비트동일). dx32/dx16 = t_len×v_len 보장.
        let ng = t_len * dm.v_len();
        let (g32, g16) = (self.ensure_dx32(ng)?, self.ensure_dx16(ng)?);
        let f = self.cc.function("gdn_gate")?;
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
        // [행-스케일 2026-10-13] 상태 기록 직후 스케일 정규화(체인 내 — 그래프
        // 캡처 포함: 포인터·그리드 고정). 히스테리시스 = 커널 내부.
        self.gdn_rescale_slot(slot, layer)?;
        Ok(())
    }

    /// [행-스케일 2026-10-13] 상태 행별 스케일 재정규화 — 행당 1스레드.
    /// head(256) = 저장 최대 → 서브노멀 문턱 ≈ 2.3e-7·행max · 오버플로 여유
    /// ≈ 256× 한-토큰 성장(기록측 ±65504 클램프). 기록 직후 호출(디코드 체인·
    /// 프리필 청크·스펙 스캔·롤백 재구성).
    pub(super) fn gdn_rescale_slot(&self, slot: usize, layer: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let f = self.cc.function("gdn_rescale")?;
        let (mut p0, mut p1) = (
            self.gdn.dgst + (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64 * 2,
            self.gdn.dsc + (slot * dm.n_gdn * dm.h_v * 128) as u64 * 4,
        );
        let (mut hv, mut dd, mut lay) = (dm.h_v as i32, dm.d as i32, layer as i32);
        let mut head = 256.0f32;
        self.cc.launch(
            f,
            dm.h_v as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut p0, &mut p1, &mut hv, &mut dd, &mut lay, &mut head),
        )
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
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
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
        self.cc.h2d(self.gdn.dgxn, b(xn))?;
        self.cc.h2d(self.gdn.dqkv, b(qkv))?;
        self.cc.h2d(self.gdn.dzv, b(z))?;
        let (gxn, gqkv, gzv) = (self.gdn.dgxn, self.gdn.dqkv, self.gdn.dzv);
        self.gdn_chain_dev(slot, layer, t_len, gxn, gqkv, gzv)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.gdn.dgate)?;
        self.cc.sync()?;
        Ok(
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()) }
                .to_vec(),
        )
    }

    // ── 어텐션 ──

    /// GDN 체인 디바이스 상주 — xn·qkv·z(디바이스) → dgate.
    /// [A-1→TreeWY 2.B 2026-10-12] 스펙 검증 활성화 — WY 표현 버퍼 확보
    /// (S_0+dc+G+k — 27B 약 164MB·35B 약 69MB. 종전 dsnap 1.50GiB/640MiB
    /// 폐지, conv 링 스냅샷 dsnap_ring은 유지).
    /// [2026-10-12] 스펙 활성 여부 — 스케줄러가 라운드 시도 전에 확인한다
    /// (비활성 상태의 라운드 시도가 요청을 죽이던 결함 — 실측 500).
    pub fn spec_on(&self) -> bool {
        self.spec_on
    }

    pub fn enable_spec(&mut self) -> Result<(), String> {
        // [2026-10-12] 스트리밍(비상주 MoE)은 검증 체인(t>1 체인) 미지원 —
        // 검증 라운드가 실패해 요청이 죽는다(실측 500). 시작 시 명시 비활성.
        if self.moe.n_experts > 0 && !self.moe.moe_resident {
            return Err("스트리밍(비상주 MoE) — 검증 체인 미지원(t>1 상주 전용)".into());
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if self.gdn.dwy_s0 != 0 {
            self.spec_on = true;
            return Ok(());
        }
        // [TreeWY 2.B 2026-10-12] 토큰별 상태 스냅샷 dsnap(8×n_layers×
        // h_v×128²×4B — 27B 1.50GiB·35B 640MiB) 폐지 → WY 표현: S_0 1개 +
        // 유사값 dc + 누적감쇠 G + k행. 롤백은 gdn_wy_reconstruct 재구성
        // (spec_rollback). 수치 판정 = PoC(evidence/TreeWY — fp32
        // |ΔS_m| ≤1.2e-7, eval 클래스 채택 기준).
        let st_elems = dm.n_gdn * dm.h_v * dm.d * dm.d;
        self.gdn.dwy_s0 = self.cc.alloc(st_elems * 4)?;
        self.gdn.dwy_dc = self.cc.alloc(8 * dm.n_gdn * dm.h_v * dm.d * 4)?;
        self.gdn.dwy_g = self.cc.alloc(8 * dm.n_gdn * dm.h_v * 4)?;
        self.gdn.dwy_k = self.cc.alloc(8 * dm.n_gdn * dm.k_len() * 4)?;
        // conv 링 스냅샷은 유지(입력 이력 — WY 대상 아님). 첨자는 GDN 서수
        // (layer·gi)라 n_gdn 스트라이드가 정확(종전 n_layers 할당은 여유분).
        let ring_snap = 8 * dm.n_gdn * 3 * dm.conv_ch() * 4;
        self.gdn.dsnap_ring = self.cc.alloc(ring_snap)?;
        // [A-1 진단] 슬롯 상태 저장 스크래치(상태 + conv 링) — save/load용.
        // [행-스케일] 상태 f16(×2) + 스케일 f32(×4) + conv 링 f32(×4).
        self.gdn.dsave = self.cc.alloc(
            self.n_slots
                * (dm.n_gdn * dm.h_v * dm.d * dm.d * 2
                    + dm.n_gdn * dm.h_v * dm.d * 4
                    + dm.n_gdn * 3 * dm.conv_ch() * 4),
        )?;
        // [§3.4 ② 2026-10-12] 통합 캐시 회계 — 스펙 상태(WY+dsnap_ring+
        // dsave). 트랜셰 2: 예산 압박 시 이 등록 자체가 ckpt·expert-lru를
        // 퇴거시킬 수 있다(실제 해제는 함수 끝 ucache_sweep).
        let spec_bytes = (st_elems * 4
            + 8 * dm.n_gdn * dm.h_v * dm.d * 4
            + 8 * dm.n_gdn * dm.h_v * 4
            + 8 * dm.n_gdn * dm.k_len() * 4
            + ring_snap
            + self.n_slots
                * (dm.n_gdn * dm.h_v * dm.d * dm.d * 2
                    + dm.n_gdn * dm.h_v * dm.d * 4
                    + dm.n_gdn * 3 * dm.conv_ch() * 4)) as u64;
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
        // [A2] 접두 재사용 체크포인트 — [K3] 512 경계 스택 4개(환형).
        // 실패(VRAM 부족) 시 비활성(기능만 off).
        let chk = 4
            * self.n_slots
            * (dm.n_gdn * dm.h_v * dm.d * dm.d * 2
                + dm.n_gdn * dm.h_v * dm.d * 4
                + dm.n_gdn * 3 * dm.conv_ch() * 4);
        match self.cc.alloc(chk) {
            Ok(p) => {
                self.gdn.dchk = p;
                self.gdn.dchk_pos = vec![[0u32; 4]; self.n_slots];
                // [§3.4 ②] 체크포인트 회계 — 슬롯×4환형 전체. 실패(VRAM)
                // 경로는 미등록(기능 off와 정합).
                if let Err(e) = self.ucache.allocate(
                    "spec:ckpt",
                    CacheCategory::Checkpoint,
                    Tier::Hbm,
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
        // [§3.4 ② 트랜셰 2] 등록 중 압박 퇴거 실행 — spec:state가 퇴거됐다면
        // WY S_0(dwy_s0)가 0화됐으므로 활성화를 되돌린다(해제 버퍼로
        // 라운드 시도 방지 — 스케줄러는 spec_on()으로 선판정).
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
        // [A3] 스테이징 d2d 3회/층 제거 — 호출자 버퍼를 커널 인자로 직접 소비.
        // (gdn_chain_dev가 형상·범위 검증·버퍼 보장을 겸한다.)
        self.gdn_chain_dev(slot, layer, t_len, xn_dev, qkv_dev, z_dev)?;
        Ok(self.gdn.dgate)
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
        let st_stride = (dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let ring_stride = (dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let bg_stride = dm.bg_len() as u64;
        // conv — 토큰별(링 = 슬롯).
        let f = self.cc.function("gdn_conv")?;
        for (k, &slot) in slots.iter().enumerate() {
            let mut c0 = qkv_dev + k as u64 * cch as u64 * 4;
            let mut c1 = self.gdn.dcw;
            let mut c2 = self.gdn.dring + (slot as u64 * ring_stride) * 4;
            let mut c3 = self.gdn.dgq + k as u64 * kl as u64 * 4;
            let mut c4 = self.gdn.dgk + k as u64 * kl as u64 * 4;
            let mut c5 = self.gdn.dgv + k as u64 * vl as u64 * 4;
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
        // t=1 트리오 — 토큰별(상태 = 슬롯).
        let part_stride = (dm.h_v * 8 * 256) as u64;
        let dc_stride = (dm.h_v * 128) as u64;
        let (fp, fc, fu) = (
            self.cc.function("gdn1_part")?,
            self.cc.function("gdn1_comb")?,
            self.cc.function("gdn1_upd")?,
        );
        for (k, &slot) in slots.iter().enumerate() {
            let q2k = self.gdn.dq2 + k as u64 * kl as u64 * 4;
            let k2k = self.gdn.dk2 + k as u64 * kl as u64 * 4;
            let v2k = self.gdn.dv2 + k as u64 * vl as u64 * 4;
            let bgk = self.gdn.dbg + k as u64 * bg_stride * 4;
            let stk = self.gdn.dgst + slot as u64 * st_stride * 2; // f16 상태
            let partk = self.gdn.dgpart + k as u64 * part_stride * 4;
            let dck = self.gdn.dgdc + k as u64 * dc_stride * 4;
            let outk = self.gdn.dgo + k as u64 * vl as u64 * 4;
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
        // gate — 행 단위 1회. [4.1] x32/미러 융합(배치 cast_x32 노드 제거).
        {
            let ng = t_len * dm.v_len();
            let (g32, g16) = (self.ensure_dx32(ng)?, self.ensure_dx16(ng)?);
            let f = self.cc.function("gdn_gate")?;
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
