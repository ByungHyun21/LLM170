//! [A-1] 스페큘러티브 디코딩 — 검증/롤백 호스트.
//!
//! 설계(llama.cpp §4.1 이식):
//! - 초안: n-gram(core::spec) — 검증으로만 수용되므로 정확성 위험 없음.
//! - 검증: t토큰 배치 체인 + 전 위치 argmax(chain_t all=true) — 가중 커널이
//!   t행으로 상각되어 1토큰 비용에 가까움.
//! - GDN 상태: gdn_spec_scan이 토큰별 스냅샷([t][L][h_v][d²]) 기록 —
//!   부분 수용 시 스냅샷[keep-1]로 상태·conv 링을 복원(인덱스 전환 + D2D).
//! - KV/pos는 위치 카운터만 되감으면 됨(스테일 엔트리는 다음 쓰기가 덮음).

use super::*;

impl W4a16Dec {
    /// [A-1] 스펙 검증 — t토큰(2..=8) 배치 체인 + 전 위치 그리디 토큰.
    /// 상태(GDN 스냅샷 포함)·pos는 전진한다(수용 판정 후 롤백은 호출부).
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        if self.head_w == 0 {
            return Err("spec_verify: head 미등록".into());
        }
        if !self.spec_on {
            return Err("spec_verify: enable_spec 미호출".into());
        }
        if !(2..=8).contains(&t) {
            return Err(format!("spec_verify: t={t} 계약(2..=8)"));
        }
        let v = self.chain_t(slot, rows, t, false, true, false)?;
        Ok(v.into_iter().map(|x| x as u32).collect())
    }

    /// [A-1] 롤백 — 배치 `keep`개 토큰 처리 시점(스냅샷[keep-1])으로 슬롯
    /// GDN 상태·conv 링 복원. pos 되감기는 spec_rewind_pos 소관.
    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if !self.spec_on || !(1..=8).contains(&keep) {
            return Err(format!("spec_rollback: 계약 위반 keep={keep}"));
        }
        let _g = self.cc.guard()?;
        let st_slot = slot * dm.n_gdn * dm.h_v * 128 * 128;
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let st_elems = (dm.h_v * dm.d * dm.d) as u64;
        let ring_elems = (3 * dm.conv_ch()) as u64;
        for layer in 0..self.n_layers {
            let src = self.dsnap
                + ((keep - 1) as u64 * self.n_layers as u64 + layer as u64) * st_elems * 4;
            let dst = self.dgst + (st_slot as u64 + layer as u64 * st_elems) * 4;
            self.copy_dev(dst, src, st_elems * 4)?;
            let rsrc = self.dsnap_ring
                + (layer as u64 * 8 * ring_elems + (keep - 1) as u64 * ring_elems) * 4;
            let rdst = self.dring + (ring_slot as u64 + layer as u64 * ring_elems) * 4;
            self.copy_dev(rdst, rsrc, ring_elems * 4)?;
        }
        Ok(())
    }

    /// [A-1] D2D 복사(커널) — 롤백 전용. cuMemcpyDtoDAsync가 드라이버
    /// 세그폴트(초유일 사용 경로 — 실측 백트레이스)라 커널 복사로 대체.
    pub(super) fn copy_dev(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: u64,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_copy")?;
        let n4 = (bytes / 16) as i64;
        let (mut p_s, mut p_d, mut p_n) = (src, dst, n4);
        self.cc.launch(
            f,
            ((n4 as u64).div_ceil(256)) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut p_s, &mut p_d, &mut p_n),
        )
    }

    /// [A-1 진단] spec scan 커널 토글(교차 대조용).
    pub fn spec_set_scan(&mut self, on: bool) {
        self.spec_scan_on = on;
    }

    /// [A-1 진단] GDN 출력(o_lc) 덤프 — 마지막 GDN 층 값.
    pub fn spec_dump_outv(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let mut b = vec![0u8; n * 4];
        self.cc.d2h(&mut b, self.dgo)?;
        self.cc.sync()?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [A-1 진단] 슬롯 GDN 상태 덤프(d2h) — trio vs spec scan 대조용.
    pub fn spec_dump_state(&mut self, slot: usize) -> Result<Vec<f32>, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let bytes = dm.n_gdn * dm.h_v * dm.d * dm.d * 4;
        let mut b = vec![0u8; bytes];
        self.cc.d2h(&mut b, self.dgst + st_slot * 4)?;
        self.cc.sync()?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [A-1 진단] 슬롯 GDN 상태 저장/복원(전 층·헤드) — 검증 대조용.
    pub fn spec_save_state(&mut self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        self.copy_dev(self.dsave, self.dgst + st_slot * 4, st_bytes)?;
        self.copy_dev(
            self.dsave + st_bytes,
            self.dring + ring_slot * 4,
            ring_bytes,
        )
    }

    /// [A-1 진단] 저장 상태 복원(스냅샷 기반 롤백과 별개 — 대조 실험용).
    pub fn spec_load_state(&mut self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        self.copy_dev(self.dgst + st_slot * 4, self.dsave, st_bytes)?;
        self.copy_dev(
            self.dring + ring_slot * 4,
            self.dsave + st_bytes,
            ring_bytes,
        )
    }

    /// [A-1] pos 되감기 — slot_pos/attn pos를 n만큼 되돌린다(스테일 KV는
    /// 다음 쓰기가 덮는다).
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        let p = self.slot_pos[slot].saturating_sub(n);
        self.slot_pos[slot] = p;
        self.attn_set_pos(slot, p)?;
        Ok(())
    }
}
