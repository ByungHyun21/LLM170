use super::*;

impl W4a16Dec {
    pub fn attn_set_pos(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.attn.dpp == 0 || slot >= self.n_slots {
            return Err("attn: pp 미할당/슬롯 범위".into());
        }
        if self.graph.capture_pinned_src {
            // 캡처 중: dpp 갱신은 replay 전 1회(그래프 밖·같은 스트림)로 옮긴다 —
            // 캡처 중 스택 임시를 소스로 잡으면 replay에서 무효 주소가 된다.
            return Ok(());
        }
        // 비동기 — 층마다 동기 H2D를 걸면 스트림이 매번 배수된다(층당 8ms 실측).
        self.cc
            .h2d_async(self.attn.dpp + (slot as u64) * 4, &pos.to_le_bytes())
    }

    pub(super) fn attn_pp_ptr(&self, slot: usize) -> CUdeviceptr {
        self.attn.dpp + (slot as u64) * 4
    }

    /// [K6-3 2026-10-13] (slot, layer, pos) 행의 KV 4포인터 — **블록 테이블
    /// 간접(호스트 정본 = KvPool)**. 종전 슬롯 연속 계산을 대체한다(mtp 위생·
    /// 리셋 등 직접 주소가 필요했던 지점). 미할당 블록은 Err — 호출부가
    /// ensure_upto를 선행하거나 회피한다.
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
        Ok((
            self.attn.dkc + id * bk + off * kvd,
            self.attn.dvc + id * bk + off * kvd,
            self.attn.dksc + id * bs + off * kvh * 4,
            self.attn.dvsc + id * bs + off * kvh * 4,
        ))
    }

    /// [K6-2 2026-10-12] KV 블록 테이블 포인터(슬롯 기저) — 디바이스 u64
    /// [layer][mb][4](k, v, ksc, vsc). set_attn이 연속 할당 기반으로 구축
    /// (내용 불변 — 그래프 캡처 계약: 캡처는 테이블 주소만 베이크).
    /// 어텐션 커널 4종(prep_q·part_q·part_qg·fa2_qg)이 기존 KV 4포인터를
    /// 대신해 이 1포인터로 간접 어드레싱한다(merge는 KV 무판독 — 무관).
    pub(super) fn attn_blk_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn.attn {
            Some(dm) => {
                let mb = dm.cap.div_ceil(512);
                self.attn.dblk_tab + (slot * dm.n_attn * mb * 4) as u64 * 8
            }
            None => self.attn.dblk_tab,
        }
    }

    /// 어텐션 prep — [A3 2026-10-09] qg·kin·vin은 커널 인자 직접 소비
    /// (스테이징 d2d 제거). 출력은 dqh_a(정규화 q)·KV 캐시.
    /// [KVQ 채택 2026-10-10] int8 KV 단일 경로. [K6-2 2026-10-12] KV·스케일
    /// 4포인터 → 블록 테이블 1포인터 간접화(attn_blk_ptr).
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
        // [행 퓨전 2026-10-14] 단일 체인은 행별 배열 불요(null).
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
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fwd3s: T={t_len} — 소형 전용 도메인 위반"));
        }
        // [FA2 2026-10-12] 프리필(t>8)은 FA2 타일 경로 — part_qg+merge 대체.
        // (수치 클래스: q f16·P f16·스코어 mma — eval 대역 판정·골든 재동결
        // 절차. 디코드 t≤8은 기존 part_q+merge 유지.)
        if t_len > 8 {
            return self.attn_fa2_launch(slot, layer, t_len, qg_dev);
        }
        self.attn_fwd3s_legacy_launch(slot, layer, t_len, qg_dev)
    }

    /// [FA2] 기존 part_qg+merge 경로 — 디코드(t≤8) 프로덕션 + A/B 참조 전용.
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
        // [행 퓨전 2026-10-14] 단일 체인은 행별 배열 불요(null).
        let (mut n0, mut n1) = (0u64, 0u64);
        // [§8.1] 디코드(t≤8)는 분할 32 — 점유 회복(ncu: 192블록 25%).
        let spv = if t_len <= 8 {
            ATTN_SPLITS_D
        } else {
            ATTN_SPLITS
        };
        let mut sp = spv as i32;
        // [KV 읽기 공유 채택 2026-10-13] GQA 패킹 단일 경로 — kv헤드 그룹의
        // gq개 q헤드가 K/V 행 판독을 공유(그리드 y = kv_heads×S). 종전 디코드는
        // q헤드별 중복 판독(part_q — gq회). 장문 곡선 실측(27B): 디코드 지연
        // 기울기 ≈ 컨텍스트 토큰당 146ns(64K에서 tg 35.3) — 실측 분해 결과
        // **DRAM 대역이 아니라 점유/지연 바운드**(패킹만으로 +2%, 분할 64로
        // +10%: 실측표는 attn.cu part_qg 주석). part_qg는 AV의 V·vsc 곱 공유와
        // 행 오름차순 누산을 설계 계약으로 하므로 **비트동일**(§8.2 주석 —
        // 골든·지문 재판정으로 확인). 전프리필(t>8)은 FA2 담당, 이 커널은
        // t 도메인 전체(디코드+스펙 검증) 단일 경로.
        let fp = self.cc.function("attn_fwd3s_part_qg")?;
        // [K6-2] KV 4포인터 → 블록 테이블(attn_blk_ptr) 간접화.
        let (mut f0, mut f1, mut f2, mut f3) = (
            self.attn.dqh_a,
            self.attn_blk_ptr(slot),
            self.attn.dattn_part,
            self.attn_pp_ptr(slot),
        );
        // [gemma4 P2 2026-10-14] q·k 스케일 인자화 — qwen = 1/√d(0.0625 동일값,
        // 비트 불변), gemma4 = 1.0. 층별 테이블 폴백 포함(`AttnLayer::scale`).
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
        // [4.1 2026-10-11] merge가 x32/미러 동시 기록 — cast_x32 노드 제거.
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

    /// [FA2 2026-10-12] 프리필 어텐션 FA2 경로 — t>8 전용. prep이 기록한
    /// dqh·int8 KV를 소비해 outv + out32/out16 융합 기록(merge 불요).
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
        // [K6-2] KV 4포인터 → 블록 테이블(attn_blk_ptr) 간접화.
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
        let mut scl = dm.layer(layer).scale(); // [gemma4] q·k 스케일 인자
        self.cc.launch(
            f,
            t_len.div_ceil(32) as u32,
            dm.q_heads as u32,
            128,
            &mut crate::rawcuda::args::l13(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut tl, &mut lay,
                &mut qh, &mut kvh, &mut cp, &mut scl,
            ),
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
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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
        // [A3] 스테이징 d2d 3회/층 제거 — 호출자 버퍼를 커널 인자로 직접 소비.
        self.attn_prep_launch(slot, layer, t_len, qg_dev, kin_dev, vin_dev)?;
        self.attn_fwd3s_launch(slot, layer, t_len, qg_dev)?;
        Ok(self.attn.doutv_a)
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
        // [§8.1] 배치 디코드도 분할 32(t_len=슬롯 수 ≤ 8).
        let spv = if t_len <= 8 {
            ATTN_SPLITS_D
        } else {
            ATTN_SPLITS
        };
        let mut sp = spv as i32;
        // [배치 행 퓨전 2026-10-14] 층당 3런치(행 퓨전 = PersistentKV 워크큐류)
        // — 종전 토큰별 3t런치. 행별 blk/pp는 drow_blk/drow_pp(캡처 h2d 노드가
        // 기록) · 인자 베이스 + 커널 내 행 인덱스(row=t) — 주소·산술은 종전
        // 토큰별 발사와 동일(비트동일 계약: 같은 lim·같은 K/V 행·같은 누산 순서).
        // part는 행 퓨전으로 그리드가 (t, q_heads×S)가 되어 점유가 회복되므로
        // 패킹(part_qg — 그리드 (t, kv_heads×S))도 성립한다: 종전 "토큰별 패킹
        // 기각"(−2%, 2026-10-14 원장)은 런치당 t=1(256블록 25% 점유) 한정 실측.
        let (row_blk, row_pp) = (self.drow_blk, self.drow_pp);
        // prep — 1런치: 그리드 (t, q+kv) · 행 = t.
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
        // fwd3s part + merge — 2런치(행 퓨전).
        // [2026-10-14 실측 노트] 토큰별(비퓨전) part_qg 전환은 −2%로 기각됐고
        // (배치 점유/지연 바운드), 퓨전 후에는 그리드 (t, kv_heads×S) = t×256
        // 블록으로 점유가 회복된다 — 그래도 배치 기본은 part_q(그리드 t×1536)
        // 유지: 초기 채택은 보수적으로, 곡선 재측정은 차기.
        let fp = self.cc.function("attn_fwd3s_part_q")?;
        let fm = self.cc.function("attn_fwd3s_merge")?;
        // [4.1] 배치 merge 융합 기록 — dx32/dx16 = t_len×q_heads×256 보장.
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
            let mut scl = dm.layer(layer).scale(); // [gemma4] q·k 스케일 인자
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
