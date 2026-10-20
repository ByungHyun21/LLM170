use super::*;

/// [결함 정정 단일 출처 2026-10-17] 슬롯별 GDN 상태(f16)·행 스케일(f32)
/// **원소** 오프셋 = (상태, 스케일). 상태는 슬롯당 n_gdn×h_v×128×128,
/// 스케일은 n_gdn×h_v×128(행 스칼라 1개) — 정확히 128배 차이. 두 배열에 같은
/// 오프셋을 쓰면 슬롯≥1에서 128× OOB: ①10-13 gdn scan("np·다중 슬롯 serve
/// illegal access") ②10-17 spec 롤백 재구성(kv 8G C≥4 700 — sanitizer
/// gdn_wy_reconstruct sc 로드 4B OOB, 1G는 우연 매핑으로 침묵 오염).
/// 슬롯 파생은 이 함수만 쓴다(파생식 중복 = 재발).
pub(super) fn gdn_state_offsets(dm: &GdnDims, slot: usize) -> (usize, usize) {
    (
        slot * dm.n_gdn * dm.h_v * 128 * 128,
        slot * dm.n_gdn * dm.h_v * 128,
    )
}

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
        // [결함수정 2026-10-13 → 단일 출처 2026-10-17] 스케일 슬롯 기저 —
        // 스케일 배열은 슬롯당 n_gdn×h_v×128(스칼라 1/행)이므로 st_slot(×128
        // 포함)을 쓰면 슬롯≥1에서 128× OOB(슬롯 0만 정확해 단일-슬롯 골든·
        // 스모크가 통과했고, np·다중 슬롯 serve가 illegal access로 죽었다).
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
        // [A-1] 스펙 검증이면 토큰별 링 스냅샷 기록(층 슬라이스 = layer×8×3×ch).
        let mut c6 = if self.spec_on && (2..=8).contains(&t_len) {
            self.gdn.dsnap_ring + (layer as u64 * 8 * 3 * dm.conv_ch() as u64) * 4
        } else {
            0
        };
        // [dsave 대체 2026-10-14] 라운드 전 링 스냅샷(층 슬라이스 [3][ch]) —
        // acc==0 복원(spec_rollback keep=0)의 S_0 동반 상태. 같은 조건에서만.
        let mut c7 = if c6 != 0 && self.gdn.dring0 != 0 {
            self.gdn.dring0 + (layer as u64 * 3 * dm.conv_ch() as u64) * 4
        } else {
            0
        };
        // [행 퓨전 2026-10-14] 단일 체인은 행별 링 배열 불요(null).
        let mut c8 = 0u64;
        self.cc.launch(
            f,
            (dm.conv_ch() / 128) as u32,
            // [토큰축 병렬] y = ceil(t_len/16) — 4탭은 입력 이력만 필요.
            (t_len.div_ceil(16)) as u32,
            128,
            &mut crate::rawcuda::args::l14(
                &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7, &mut c8,
                &mut tl, &mut lay, &mut kl, &mut vl, &mut cch,
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
            // [행 퓨전 2026-10-14] 행별 배열 인자(null — 단일 체인) ×2.
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

            // [GDN-FLA K1/K2 2026-10-19] wu(청크 병렬 해·W'/U_local) → h(순수
            // 재귀) → o(청크 병렬 출력). FlashKDA 정본: 토큰 병렬 작업 전부 K1.
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
            // grid = h_v×NSPLIT(블록 = GDN_NGRP×GDN_VS = 512스레드).
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

        // [4.1 2026-10-11] gate가 x32/미러 동시 기록 — split 체인의 cast_x32
        // 노드 제거(값 비트동일). dx32/dx16 = t_len×v_len 보장.
        let ng = t_len * dm.v_len();
        let (g32, g16) = (self.ensure_dx32(ng)?, self.ensure_dx16(ng)?);
        // [FN 2026-10-20] 출력게이트 = config output_gate_type(qwen35 silu /
        // FN sigmoid) — gdn_gate_sig 심볼(fn.cu) 선택.
        let f = self
            .cc
            .function(if self.gdn.sig_gate { "gdn_gate_sig" } else { "gdn_gate" })?;
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
        // [rescale 통합 2026-10-13] 마지막 GDN 층에서 **전 층 일괄** 정규화 —
        // 층별 런치(디코드 48회/토큰)를 1회로(그래프 노드 감소 = 채택 비용
        // 회복). 정합: 층 L 상태는 같은 체인 안 재판독 없음(다음 토큰이 첫
        // 판독)·체크포인트(청크 끝)는 항상 이 지점 뒤. 히스테리시스 = 커널 내부.
        if layer == dm.n_gdn - 1 {
            self.gdn_rescale_all(slot)?;
        }
        Ok(())
    }

    /// [행-스케일 2026-10-13] 상태 행별 스케일 재정규화 — 행당 1스레드.
    /// head(256) = 저장 최대 → 서브노멀 문턱 ≈ 2.3e-7·행max · 오버플로 여유
    /// ≈ 256× 한-토큰 성장(기록측 ±65504 클램프). 기록 직후 호출(디코드 체인·
    /// 프리필 청크·스펙 스캔·롤백 재구성).
    pub(super) fn gdn_rescale_slot(&self, slot: usize, layer: usize) -> Result<(), String> {
        self.gdn_rescale_layers(slot, layer, 1)
    }

    /// [rescale 통합 2026-10-13] 전 층 일괄 — 체인 끝(마지막 GDN 층)에서 1회.
    /// 층별 수학 동일(비트동일): 층 L 상태는 같은 체인 안에서 재판독이 없다.
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
        // (spec_rollback). 수치 판정 = PoC(scripts/poc_treewy.py — fp32
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
        // [dsave 대체 2026-10-14] 라운드 전 conv 링 1벌([n_gdn][3][conv_ch]) —
        // acc==0(전량 거절) 복원용(라운드 conv가 갱신 전 링을 기록). 종전
        // dsave(슬롯×[상태 f16+스케일+링] — 27B/4슬롯 약 315MiB)는 **프로브
        // 전용 지연 할당**(spec_save_state)으로 이동 — serve 경로 VRAM 회수.
        let ring_pre = dm.n_gdn * 3 * dm.conv_ch() * 4;
        self.gdn.dring0 = self.cc.alloc(ring_pre)?;
        // [§3.4 ② 2026-10-12] 통합 캐시 회계 — 스펙 상태(WY+dsnap_ring+dring0).
        // 트랜셰 2: 예산 압박 시 이 등록 자체가 ckpt·expert-lru를 퇴거시킬 수
        // 있다(실제 해제는 함수 끝 ucache_sweep).
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
        // [A2] 접두 재사용 체크포인트 — [K3] 512 경계 스택 4개(환형).
        // 실패(VRAM 부족) 시 비활성(기능만 off).
        let chk = 4
            * self.n_slots
            * (dm.n_gdn * dm.h_v * dm.d * dm.d * 2
                + dm.n_gdn * dm.h_v * dm.d * 4
                + dm.n_gdn * 3 * dm.conv_ch() * 4);
        // [호스트 이관 2026-10-14] 콜드 백업 — VRAM 대신 pinned 호스트 RAM.
        // 체크포인트 기록 = d2h(청크당 ~77MB)·복원 = h2d — 스트림 순서 보존,
        // 수치(바이트)는 d2d 시절과 동일. VRAM −1.26GiB = 컨텍스트 +37K토큰급.
        match self.cc.pinned_alloc(chk) {
            Ok(p) => {
                self.gdn.dchk = p;
                self.gdn.dchk_pos = vec![[0u32; 4]; self.n_slots];
                self.gdn.dchk_saved = vec![[0u32; 4]; self.n_slots];
                self.gdn.dchk_use = vec![[0u64; 4]; self.n_slots];
                self.gdn.dchk_tick = 0;
                // [§3.4 ②] 체크포인트 회계 — 슬롯×4환형 전체. 실패(호스트)
                // 경로는 미등록(기능 off와 정합).
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
        // [배치 행 퓨전 2026-10-14] 층당 4런치(행 퓨전 = PersistentKV 워크큐류) —
        // 종전 토큰별 6t런치. 행별 슬롯 링/상태는 drow_ring/drow_st/drow_sc
        // (캡처 h2d 노드가 기록), k-스트라이드 인자는 베이스 + 커널 내 행 인덱스
        // — 주소·산술은 종전 토큰별 발사와 동일(행 k = k번째 토큰). 비트동일 계약.
        let (row_ring, row_st, row_sc) = (self.drow_ring, self.drow_st, self.drow_sc);
        // conv — 1런치: 그리드 (ch/128, t) · 행 = y.
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
        // t=1 트리오 — 층당 3런치(행 = y: part/upd는 행×분할 인코딩).
        let (fp, fc, fu) = (
            self.cc.function("gdn1_part")?,
            self.cc.function("gdn1_comb")?,
            self.cc.function("gdn1_upd")?,
        );
        {
            // part — 그리드 (h_v, 8·t).
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
            // comb — 그리드 (h_v, t) · 상태 무접촉(베이스 전달 — 커널이 행 인덱스).
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) = (
                self.gdn.dq2,
                self.gdn.dk2,
                self.gdn.dv2,
                self.gdn.dbg,
                self.gdn.dgpart,
                self.gdn.dgdc,
                self.gdn.dgo,
            );
            // comb은 행 인자를 받지 않는다(blockIdx.y = 행).
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
            // upd — 그리드 (h_v, 8·t).
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
        // [rescale 통합 2026-10-13] 마지막 GDN 층에서 전 층·전 슬롯 일괄.
        if layer == dm.n_gdn - 1 {
            for &slot in slots {
                self.gdn_rescale_all(slot)?;
            }
        }
        // gate — 행 단위 1회. [4.1] x32/미러 융합(배치 cast_x32 노드 제거).
        {
            let ng = t_len * dm.v_len();
            let (g32, g16) = (self.ensure_dx32(ng)?, self.ensure_dx16(ng)?);
            let f = self
                .cc
                .function(if self.gdn.sig_gate { "gdn_gate_sig" } else { "gdn_gate" })?;
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
