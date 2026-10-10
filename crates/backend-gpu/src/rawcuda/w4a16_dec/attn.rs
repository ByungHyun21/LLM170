use super::*;

impl W4a16Dec {
    pub fn attn_set_pos(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.dpp == 0 || slot >= self.n_slots {
            return Err("attn: pp 미할당/슬롯 범위".into());
        }
        if self.capture_pinned_src {
            // 캡처 중: dpp 갱신은 replay 전 1회(그래프 밖·같은 스트림)로 옮긴다 —
            // 캡처 중 스택 임시를 소스로 잡으면 replay에서 무효 주소가 된다.
            return Ok(());
        }
        // 비동기 — 층마다 동기 H2D를 걸면 스트림이 매번 배수된다(층당 8ms 실측).
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &pos.to_le_bytes())
    }

    pub(super) fn attn_pp_ptr(&self, slot: usize) -> CUdeviceptr {
        self.dpp + (slot as u64) * 4
    }

    pub(super) fn attn_kv_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => {
                // [KVQ 채택 2026-10-10] int8 KV — 슬롯 오프셋 = 원소 수(1B/원소).
                self.dkc + dm.kv_slot_elems(slot) as u64
            }
            None => self.dkc,
        }
    }

    pub(super) fn attn_vc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => {
                // [KVQ 채택 2026-10-10] int8 KV — 슬롯 오프셋 = 원소 수(1B/원소).
                self.dvc + dm.kv_slot_elems(slot) as u64
            }
            None => self.dvc,
        }
    }

    /// [P13] KV 스케일 포인터(슬롯 기저) — KVQ 전용(f32).
    pub(super) fn attn_ksc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dksc + (slot * dm.n_attn * dm.cap * dm.kv_heads) as u64 * 4,
            None => self.dksc,
        }
    }

    pub(super) fn attn_vsc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dvsc + (slot * dm.n_attn * dm.cap * dm.kv_heads) as u64 * 4,
            None => self.dvsc,
        }
    }

    /// 어텐션 prep — [A3 2026-10-09] qg·kin·vin은 커널 인자 직접 소비
    /// (스테이징 d2d 제거). 출력은 dqh_a(정규화 q)·KV 캐시.
    /// [KVQ 채택 2026-10-10] int8 KV 단일 경로 — ksc/vsc(행×헤드 f32) 인자.
    pub(super) fn attn_prep_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let f = self.cc.function("attn_prep_q")?;
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        #[allow(clippy::type_complexity)]
        let (
            mut a0,
            mut a1,
            mut a2,
            mut a3,
            mut a4,
            mut a5,
            mut a6,
            mut a7,
            mut a8,
            mut a9,
            mut aa,
        ) = (
            qg_dev,
            kin_dev,
            vin_dev,
            self.dqnw_a,
            self.dknw_a,
            self.dqh_a,
            self.attn_kv_ptr(slot),
            self.attn_vc_ptr(slot),
            self.attn_ksc_ptr(slot),
            self.attn_vsc_ptr(slot),
            self.attn_pp_ptr(slot),
        );
        self.cc.launch(
            f,
            t_len as u32,
            (dm.q_heads + dm.kv_heads) as u32,
            128,
            &mut crate::rawcuda::args::l16(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7, &mut a8,
                &mut a9, &mut aa, &mut tl, &mut lay, &mut qh, &mut kvh, &mut cp,
            ),
        )
    }

    /// fwd3s — [A3 2026-10-09] gate(qg)는 커널 인자 직접 소비(스테이징 제거).
    /// [KVQ 채택 2026-10-10] int8 KV 판독 단일 경로 — 분할 s + 병합. 분할 수는
    /// 커널이 lim으로 결정(그래프 캡처 무관) — 호스트는 항상 분할 경로.
    pub(super) fn attn_fwd3s_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fwd3s: T={t_len} — 소형 전용 도메인 위반"));
        }
        if self.dattn_part == 0 {
            return Err("attn fwd3s: dattn_part 미할당 — ensure_attn_bufs 선행".into());
        }
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let mut sp = ATTN_SPLITS as i32;
        let fp = self.cc.function("attn_fwd3s_part_q")?;
        #[allow(clippy::type_complexity)]
        let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
            self.dqh_a,
            self.attn_kv_ptr(slot),
            self.attn_vc_ptr(slot),
            self.attn_ksc_ptr(slot),
            self.attn_vsc_ptr(slot),
            self.dattn_part,
        );
        let mut f6 = self.attn_pp_ptr(slot);
        self.cc.launch(
            fp,
            t_len as u32,
            (dm.q_heads * ATTN_SPLITS) as u32,
            256,
            &mut crate::rawcuda::args::l13(
                &mut f0, &mut f1, &mut f2, &mut f3, &mut f4, &mut f5, &mut f6, &mut tl, &mut lay,
                &mut qh, &mut kvh, &mut cp, &mut sp,
            ),
        )?;
        let fm = self.cc.function("attn_fwd3s_merge")?;
        let (mut mp, mut mg, mut mo) = (self.dattn_part, qg_dev, self.doutv_a);
        let (mut tl2, mut qh2) = (t_len as i32, dm.q_heads as i32);
        self.cc.launch(
            fm,
            t_len as u32,
            dm.q_heads as u32,
            256,
            &mut crate::rawcuda::args::l6(&mut mp, &mut mg, &mut mo, &mut tl2, &mut qh2, &mut sp),
        )
    }

    /// 어텐션 체인 호스트 진입 — qg·kin·vin 업로드 → pp=pos0 → prep → fwd3s
    /// → outv 판독.
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
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
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
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dqg_a, b(qg))?;
        self.cc.h2d(self.dkin_a, b(kin))?;
        self.cc.h2d(self.dvin_a, b(vin))?;
        self.attn_set_pos(slot, pos0)?;
        let (aqg, akin, avin) = (self.dqg_a, self.dkin_a, self.dvin_a);
        self.attn_prep_launch(slot, layer, t_len, aqg, akin, avin)?;
        self.attn_fwd3s_launch(slot, layer, t_len, aqg)?;
        let mut buf = vec![0u8; t_len * dm.q_dim() * 4];
        self.cc.d2h(&mut buf, self.doutv_a)?;
        self.cc.sync()?;
        Ok(
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, t_len * dm.q_dim()) }
                .to_vec(),
        )
    }

    // ── 순차 forward ──

    /// 어텐션 체인 디바이스 상주 — qg·kin·vin(디바이스) → doutv.
    pub(super) fn attn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX || layer >= dm.n_attn || slot >= self.n_slots {
            return Err("attn dev: 도메인/범위 위반".into());
        }
        let pos = self.slot_pos[slot];
        if pos as usize + t_len > dm.cap {
            return Err(format!("attn dev: pos{pos}+T{t_len} > cap{}", dm.cap));
        }
        self.attn_set_pos(slot, pos)?;
        self.ensure_attn_bufs(t_len)?;
        // [A3] 스테이징 d2d 3회/층 제거 — 호출자 버퍼를 커널 인자로 직접 소비.
        self.attn_prep_launch(slot, layer, t_len, qg_dev, kin_dev, vin_dev)?;
        self.attn_fwd3s_launch(slot, layer, t_len, qg_dev)?;
        Ok(self.doutv_a)
    }

    /// [A9 2026-10-10] 어텐션 디코드 배치 — 토큰별 슬롯의 KV/pos를 쓰고,
    /// prep·part·merge를 토큰 수만큼 발사(각 t=1 — 단독 경로와 동일 산술).
    /// [KVQ 채택 2026-10-10] int8 KV 커널(_q)로 포팅 — 직렬 폴백 제거.
    pub(super) fn attn_chain_dev_batch(
        &mut self,
        slots: &[usize],
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0
            || t_len > BATCH_DEC_MAX
            || t_len != slots.len()
            || layer >= dm.n_attn
            || self.dattn_part == 0
            || slots.iter().any(|&s| s >= self.n_slots)
        {
            return Err("attn batch: 도메인/범위 위반".into());
        }
        // dpp(슬롯 pos)는 배치 진입부가 1회 일괄 h2d(pin_batch_pos) — 캡처
        // 그래프 replay 시에도 갱신되도록 여기서는 호출하지 않는다.
        for &s in slots {
            let pos = self.slot_pos[s];
            if pos as usize + 1 > dm.cap {
                return Err(format!("attn batch: slot{s} pos{pos} > cap{}", dm.cap));
            }
        }
        self.ensure_attn_bufs(t_len)?;
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (qgd, kvd, qdd) = (dm.qg_dim() as u64, dm.kv_dim() as u64, dm.q_dim() as u64);
        let pstride = (dm.q_heads * ATTN_SPLITS * 258) as u64;
        let mut sp = ATTN_SPLITS as i32;
        // prep — 토큰별.
        let f = self.cc.function("attn_prep_q")?;
        for (k, &slot) in slots.iter().enumerate() {
            let mut a0 = qg_dev + k as u64 * qgd * 4;
            let mut a1 = kin_dev + k as u64 * kvd * 4;
            let mut a2 = vin_dev + k as u64 * kvd * 4;
            let mut a3 = self.dqnw_a;
            let mut a4 = self.dknw_a;
            let mut a5 = self.dqh_a + k as u64 * qdd * 4;
            let mut a6 = self.attn_kv_ptr(slot);
            let mut a7 = self.attn_vc_ptr(slot);
            let mut a8 = self.attn_ksc_ptr(slot);
            let mut a9 = self.attn_vsc_ptr(slot);
            let mut aa = self.attn_pp_ptr(slot);
            let (mut tl, mut lay, mut qh2, mut kvh2, mut cp2) = (1i32, layer as i32, qh, kvh, cp);
            self.cc.launch(
                f,
                1,
                (dm.q_heads + dm.kv_heads) as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                    &mut a8, &mut a9, &mut aa, &mut tl, &mut lay, &mut qh2, &mut kvh2, &mut cp2,
                ),
            )?;
        }
        // fwd3s part + merge — 토큰별.
        let fp = self.cc.function("attn_fwd3s_part_q")?;
        let fm = self.cc.function("attn_fwd3s_merge")?;
        for (k, &slot) in slots.iter().enumerate() {
            let qhk = self.dqh_a + k as u64 * qdd * 4;
            let partk = self.dattn_part + k as u64 * pstride * 4;
            {
                let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
                    qhk,
                    self.attn_kv_ptr(slot),
                    self.attn_vc_ptr(slot),
                    self.attn_ksc_ptr(slot),
                    self.attn_vsc_ptr(slot),
                    partk,
                );
                let mut f6 = self.attn_pp_ptr(slot);
                let (mut tl, mut lay) = (1i32, layer as i32);
                self.cc.launch(
                    fp,
                    1,
                    (dm.q_heads * ATTN_SPLITS) as u32,
                    256,
                    &mut crate::rawcuda::args::l13(
                        &mut f0, &mut f1, &mut f2, &mut f3, &mut f4, &mut f5, &mut f6, &mut tl,
                        &mut lay, &mut qh, &mut kvh, &mut cp, &mut sp,
                    ),
                )?;
            }
            {
                let (mut mp, mut mg, mut mo) = (
                    partk,
                    qg_dev + k as u64 * qgd * 4,
                    self.doutv_a + k as u64 * qdd * 4,
                );
                let (mut tl2, mut qh2) = (1i32, dm.q_heads as i32);
                self.cc.launch(
                    fm,
                    1,
                    dm.q_heads as u32,
                    256,
                    &mut crate::rawcuda::args::l6(
                        &mut mp, &mut mg, &mut mo, &mut tl2, &mut qh2, &mut sp,
                    ),
                )?;
            }
        }
        Ok(self.doutv_a)
    }
}
