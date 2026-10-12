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
        // [A6-1 2026-10-11] 검증 체인 (slot,t) 그래프 캐시 — 런치 오버헤드 제거.
        // 행은 핀드 고정 버퍼(캡처 h2d 소스 고정), pos는 replay 직전 그래프
        // 밖 h2d(같은 스트림 순서). 미스 = 직접 발사(정답·호스트 갱신 포함)
        // 후 캡처(미실행) — batch_capture 동형. 캡처 실패 시 영구 비활성.
        if self.graph.spec_graph_ok
            && !self.graph.pin_batch_in.is_null()
            && !self.graph.pin_pos.is_null()
            && self.n_slots > 0
        {
            let h = self.hidden;
            if rows.len() != t * h {
                return Err(format!("spec_verify: rows 계약 {}", rows.len()));
            }
            // SAFETY: 핀드 스크래치는 self 소유(수명=self)·복사 크기 t*h*4.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rows.as_ptr() as *const u8,
                    self.graph.pin_batch_in as *mut u8,
                    t * h * 4,
                );
            }
            let pos = self.slot_pos[slot];
            // SAFETY: pin_pos = n_slots*4 핀드(그래프 경로 선행 할당).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pos.to_le_bytes().as_ptr(),
                    (self.graph.pin_pos as *mut u8).add(slot * 4),
                    4,
                );
            }
            let posb =
                // SAFETY: self.pin_*는 pinned_alloc 소유(수명=self).
                unsafe { std::slice::from_raw_parts(self.graph.pin_pos as *const u8, self.n_slots * 4) };
            self.cc.h2d_async(
                self.attn.dpp + (slot as u64) * 4,
                &posb[slot * 4..slot * 4 + 4],
            )?;
            if let Some(k) = self
                .graph
                .spec_graphs
                .iter()
                .position(|(key, _, _)| *key == (slot, t))
            {
                let exec = self.graph.spec_graphs[k].1;
                self.cc.graph_launch(exec)?;
                self.cc.sync()?;
                let toks = self.batch_read_tokens(t);
                self.slot_pos[slot] = pos + t as u32;
                return Ok(toks);
            }
            // SAFETY: 위 복사와 동일 범위(t*h f32) — 핀드 유효.
            let pin =
                unsafe { std::slice::from_raw_parts(self.graph.pin_batch_in as *const f32, t * h) };
            let v = self.chain_t(slot, pin, t, false, true, false)?;
            let toks: Vec<u32> = v.into_iter().map(|x| x as u32).collect();
            if self.graph.spec_graphs.len() >= 8 {
                let (_, e, g) = self.graph.spec_graphs.remove(0);
                let _ = self.cc.graph_destroy(e, g);
            }
            let cap = (|| -> Result<(ffi::CUgraphExec, ffi::CUgraph), String> {
                self.cc.capture_begin()?;
                self.graph.capture_pinned_src = true;
                self.graph.capture_no_host = true;
                let r = self.chain_t(slot, pin, t, false, true, false);
                self.graph.capture_no_host = false;
                self.graph.capture_pinned_src = false;
                if let Err(e) = r {
                    let _ = self.cc.capture_end();
                    return Err(e);
                }
                let g = self.cc.capture_end()?;
                let e = self.cc.graph_instantiate(g)?;
                self.cc.graph_upload(e)?;
                Ok((e, g))
            })();
            match cap {
                Ok((e, g)) => self.graph.spec_graphs.push(((slot, t), e, g)),
                Err(_) => {
                    self.graph.spec_graph_ok = false;
                    self.cc.sync().ok();
                }
            }
            return Ok(toks);
        }
        let v = self.chain_t(slot, rows, t, false, true, false)?;
        Ok(v.into_iter().map(|x| x as u32).collect())
    }

    /// [§7 2단계 2026-10-11] 검증 체인 + [t][vocab] 로짓 회수 — 비greedy
    /// 거절샘플링용(eval logits 경로 재사용). 상태·pos는 전진(호출부 롤백).
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("spec_verify_logits: head 미등록".into());
        }
        if !self.spec_on {
            return Err("spec_verify_logits: enable_spec 미호출".into());
        }
        if !(2..=8).contains(&t) {
            return Err(format!("spec_verify_logits: t={t} 계약(2..=8)"));
        }
        self.chain_t(slot, rows, t, false, true, true)
    }

    /// [A-1] 롤백 — 배치 `keep`개 토큰 처리 시점(스냅샷[keep-1])으로 슬롯
    /// GDN 상태·conv 링 복원. pos 되감기는 spec_rewind_pos 소관.
    /// [결함수정 2026-10-11] 루프를 **GDN 층(n_gdn) 한정** — 종전 n_layers
    /// (모델 층) 순회는 상태/링 버퍼([slot][n_gdn] 크기)를 넘어 illegal
    /// access(compute-sanitizer 실측: w4a16_copy +1537B 초과). 스냅샷
    /// 스트라이드는 n_layers(스캔 계약), 상태·링 스트라이드는 n_gdn.
    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        if !self.spec_on || !(1..=8).contains(&keep) {
            return Err(format!("spec_rollback: 계약 위반 keep={keep}"));
        }
        let _g = self.cc.guard()?;
        let st_slot = slot * dm.n_gdn * dm.h_v * 128 * 128;
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let st_elems = (dm.h_v * dm.d * dm.d) as u64;
        let ring_elems = (3 * dm.conv_ch()) as u64;
        let nl = self.n_layers as u64;
        for gi in 0..dm.n_gdn {
            let src = self.gdn.dsnap + ((keep - 1) as u64 * nl + gi as u64) * st_elems * 4;
            let dst = self.gdn.dgst + (st_slot as u64 + gi as u64 * st_elems) * 4;
            self.copy_dev(dst, src, st_elems * 4)?;
            let rsrc = self.gdn.dsnap_ring + (gi as u64 * 8 + (keep - 1) as u64) * ring_elems * 4;
            let rdst = self.gdn.dring + (ring_slot as u64 + gi as u64 * ring_elems) * 4;
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
        // [2026-10-12] 컨텍스트 가드 — 스케줄 스레드 등 비-로드 스레드에서
        // 무가드 런치는 invalid device context(201)로 죽는다(실측: 스트리밍
        // 서브 스펙 라운드).
        let _g = self.cc.guard()?;
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
        self.cc.d2h(&mut b, self.gdn.dgo)?;
        self.cc.sync()?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [A-1 진단] 슬롯 GDN 상태 덤프(d2h) — trio vs spec scan 대조용.
    pub fn spec_dump_state(&mut self, slot: usize) -> Result<Vec<f32>, String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let bytes = dm.n_gdn * dm.h_v * dm.d * dm.d * 4;
        let mut b = vec![0u8; bytes];
        self.cc.d2h(&mut b, self.gdn.dgst + st_slot * 4)?;
        self.cc.sync()?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [A2→K3 2026-10-12] 접두 재사용 체크포인트 — 슬롯 GDN 상태+conv 링
    /// 저장 + 위치 기억. **512 경계마다** 호출 — 환형 스택 4개
    /// (512/1024/1536/2048 — k=(pos/512-1)%4)에 최근 4경계 유지.
    /// 값은 순수 복사(산술 불변).
    pub fn checkpoint_gdn(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.gdn.dchk == 0 || slot >= self.gdn.dchk_pos.len() || !pos.is_multiple_of(512) {
            return Ok(());
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let k = (pos / 512 - 1) as usize % 4;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        let per = st_bytes + ring_bytes;
        let base = self.gdn.dchk + (slot as u64 * 4 + k as u64) * per;
        self.copy_dev(base, self.gdn.dgst + st_slot * 4, st_bytes)?;
        self.copy_dev(base + st_bytes, self.gdn.dring + ring_slot * 4, ring_bytes)?;
        self.gdn.dchk_pos[slot][k] = pos;
        Ok(())
    }

    /// [K3] upto 이하 최대 체크포인트 경계 선택(0 = 없음).
    fn ck_pick(&self, slot: usize, upto: u32) -> u32 {
        let mut best = 0u32;
        if let Some(ks) = self.gdn.dchk_pos.get(slot) {
            for &p in ks {
                if p > 0 && p <= upto && p > best {
                    best = p;
                }
            }
        }
        best
    }

    /// [A2→K3] 체크포인트 복원 — upto 이하 최대 경계의 상태+링을 슬롯으로
    /// 되돌리고 pos를 해당 위치로(절대 설정). 반환 = 복원 위치(0 = 없음).
    pub fn restore_gdn(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        let pos = self.ck_pick(slot, upto);
        if self.gdn.dchk == 0 || pos == 0 {
            return Ok(0);
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let k = (pos / 512 - 1) as usize % 4;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        let per = st_bytes + ring_bytes;
        let base = self.gdn.dchk + (slot as u64 * 4 + k as u64) * per;
        self.copy_dev(self.gdn.dgst + st_slot * 4, base, st_bytes)?;
        self.copy_dev(self.gdn.dring + ring_slot * 4, base + st_bytes, ring_bytes)?;
        self.slot_pos[slot] = pos;
        self.attn_set_pos(slot, pos)?;
        Ok(pos)
    }

    /// [A2→K3] 체크포인트 위치 — upto 이하 최대(0 = 없음).
    pub fn checkpoint_pos(&self, slot: usize, upto: u32) -> u32 {
        self.ck_pick(slot, upto)
    }

    /// [A-1 진단] 슬롯 GDN 상태 저장/복원(전 층·헤드) — 검증 대조용.
    pub fn spec_save_state(&mut self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        self.copy_dev(self.gdn.dsave, self.gdn.dgst + st_slot * 4, st_bytes)?;
        self.copy_dev(
            self.gdn.dsave + st_bytes,
            self.gdn.dring + ring_slot * 4,
            ring_bytes,
        )
    }

    /// [A-1 진단] 저장 상태 복원(스냅샷 기반 롤백과 별개 — 대조 실험용).
    pub fn spec_load_state(&mut self, slot: usize) -> Result<(), String> {
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let st_slot = (slot * dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let st_bytes = (dm.n_gdn * dm.h_v * dm.d * dm.d * 4) as u64;
        let ring_slot = (slot * dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let ring_bytes = (dm.n_gdn * 3 * dm.conv_ch() * 4) as u64;
        self.copy_dev(self.gdn.dgst + st_slot * 4, self.gdn.dsave, st_bytes)?;
        self.copy_dev(
            self.gdn.dring + ring_slot * 4,
            self.gdn.dsave + st_bytes,
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
