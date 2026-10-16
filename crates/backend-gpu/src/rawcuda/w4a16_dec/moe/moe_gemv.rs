use super::super::*;

impl W4a16Dec {
    /// 배치 전문가 GEMV(상주) — tab 간접 참조, grid = n × nslots.
    /// xstride = 0(x 공통) 또는 k(슬롯/토큰별 활성), sp = 토큰당 슬롯 수
    /// (프리필 = top_k → x는 토큰 단위 [t][k], 디코드 = 1 → 슬롯별 [nslots][k]).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn gemv_experts_launch(
        &self,
        base: usize,
        nslots: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_experts_g32_bf16")?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) = (
            self.moe.moe_dev_tab,
            base as i32,
            self.moe.moe_idx,
            nslots as i32,
        );
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            out_dev,
            n as i32,
            k as i32,
        );
        self.cc.launch(
            f,
            (n * nslots) as u32,
            1,
            64,
            &mut crate::rawcuda::args::l10(
                &mut p_t, &mut p_b, &mut p_i, &mut p_ns, &mut p_x, &mut p_xs, &mut p_sp, &mut p_o,
                &mut p_n, &mut p_k,
            ),
        )
    }

    /// [B-1/B-2 2026-10-10] gate+up+ew 융합 런치(act 직접 기록) — t=1 상주.
    /// grid = n(행)×nslots, 블록 64(행당 2회 gemv_row_body).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn gemv_experts_glu_launch(
        &self,
        base: usize,
        nslots: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        act_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        // [gemma4 P2 2026-10-14] 활성 선택 — 기본 silu, gemma4는 gelu_tanh
        // (모델별 단일 경로 — 런타임 수치 플래그 아님: 로드 시 확정).
        let f = self.cc.function(if self.moe.glu_gelu {
            "w4a16_gemv_experts_glu_gelu"
        } else {
            "w4a16_gemv_experts_glu"
        })?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) = (
            self.moe.moe_dev_tab,
            base as i32,
            self.moe.moe_idx,
            nslots as i32,
        );
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            act_dev,
            n as i32,
            k as i32,
        );
        self.cc.launch(
            f,
            (n * nslots) as u32,
            1,
            64,
            &mut crate::rawcuda::args::l10(
                &mut p_t, &mut p_b, &mut p_i, &mut p_ns, &mut p_x, &mut p_xs, &mut p_sp, &mut p_o,
                &mut p_n, &mut p_k,
            ),
        )
    }

    /// 전문가 배치(상주) — 테이블 간접 GEMV + 가중 누적.
    /// [P1] idx/wt는 호출자가 디바이스에 기록한다(moe_route_dev) — 종전
    /// h2d 2회(층당) 제거. gDN 경로(moe_ffn_dev_t)는 호스트 선택이 남아
    /// 이 함수 앞에서 moe_idx/moe_wt를 h2d로 채운 뒤 ns를 넘긴다.
    pub(in crate::rawcuda::w4a16_dec) fn moe_experts_batch(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        ns: usize,
    ) -> Result<(), String> {
        let n_exp = self.moe.n_experts;
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        // [B-1/B-2 2026-10-10] gate+up+ew 융합 1런치(x 공통) → down → 누적.
        // ew 커널(실측 0.35ms/토큰)과 중간 버퍼 왕복·x 재판독 제거 — act는
        // 융합 커널이 직접 기록(산술 비트동일).
        let base = il * n_exp * 3;
        self.gemv_experts_glu_launch(base, ns, xn, 0, ns, self.moe.exp.act, n_ff, h)?;
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
        // [§4.5] 누적은 꼬리(shared down 이후)의 accum_sg가 수행 — 호출부 참조.
        Ok(())
    }

    /// 전문가 스트리밍(비상주) — 스테이징 1쌍 재사용 h2d + 개별 GEMV.
    pub(in crate::rawcuda::w4a16_dec) fn moe_experts_streaming(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        sel: &[(usize, f32)],
    ) -> Result<(), String> {
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        if sel.is_empty() {
            return Ok(());
        }
        // [P7 2026-10-09] 층 단위 배치 스테이징 — (top_k×3) q/s 슬라이스를
        // async 복사로 전부 올린 뒤 sync 1회(종전 proj마다 h2d_chunked —
        // proj당 sync 2회 × 3 × top_k = 층당 수십 sync). 소스는 호스트
        // 스테이징(C2)·mmap — 둘 다 호출 내 수명이면 충분(드라이버 스테이징).
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe.moe_group * 2;
        debug_assert!(
            self.moe.stg.cap.0 >= qsz * sel.len() * 3 && self.moe.stg.cap.1 >= ssz * sel.len() * 3
        );
        // [B4-2 2026-10-12] 전문가 LRU 캐시 — hit은 h2d 생략(가중치 불변이라
        // 수치 무변화), miss는 캐시 슬롯(빈/LRU 퇴거)으로 직접 h2d. 캐시 없음
        // (mcache_k=0)이면 종전 stg 경로 그대로.
        let use_cache = self.moe.mcache_k > 0;
        let mut qd_list: Vec<(u64, u64)> = Vec::with_capacity(sel.len());
        for (j, &(e, _)) in sel.iter().enumerate() {
            let (qd, sd, need_h2d) = if use_cache {
                match self.mcache_lookup(il, e) {
                    Some(slot) => {
                        self.moe.mcache_hits += 1;
                        let qb = self.moe.mcache_q
                            + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
                        let sb = self.moe.mcache_s
                            + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
                        (qb, sb, false)
                    }
                    None => {
                        self.moe.mcache_miss += 1;
                        let slot = self.mcache_insert(il, e)?;
                        let qb = self.moe.mcache_q
                            + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
                        let sb = self.moe.mcache_s
                            + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
                        (qb, sb, true)
                    }
                }
            } else {
                let qb = self.moe.stg.q + ((j * 3) * qsz) as u64;
                let sb = self.moe.stg.s + ((j * 3) * ssz) as u64;
                (qb, sb, true)
            };
            if need_h2d {
                // [B4-1 결함수정 2026-10-13] 업로드는 공용 헬퍼 — 스테이징(moe_stage_t)
                // 삽입도 반드시 업로드를 동반한다(종전엔 t=1 경로만 업로드해 t≤8
                // GEMV가 미업로드 슬롯을 읽었다: 실측 오염 — 워밍업/프리필 유령 슬롯).
                self.mcache_upload_to(il, e, qd, sd)?;
            }
            qd_list.push((qd, sd));
        }
        self.cc.sync()?;
        if use_cache
            && llm170_diag::flag::on_nonzero("LLM170_MOE_DBG")
            && self.moe.mcache_hits + self.moe.mcache_miss > 0
            && (self.moe.mcache_hits + self.moe.mcache_miss).is_multiple_of(2560)
        {
            eprintln!(
                "[mcache] hit {} / {} ({:.1}%)",
                self.moe.mcache_hits,
                self.moe.mcache_hits + self.moe.mcache_miss,
                100.0 * self.moe.mcache_hits as f64
                    / (self.moe.mcache_hits + self.moe.mcache_miss) as f64
            );
        }
        for (j, &(_e, w)) in sel.iter().enumerate() {
            let (q0, p0) = qd_list[j];
            let q1 = q0 + qsz as u64;
            let p1 = p0 + ssz as u64;
            let q2 = q0 + (2 * qsz) as u64;
            let p2 = p0 + (2 * ssz) as u64;
            // proj 순서 = 테이블 조립 순서(gate, up, down) — 스테이징과 일치.
            self.gemv_launch_raw(q0, p0, n_ff, h, xn, s0)?;
            self.gemv_launch_raw(q1, p1, n_ff, h, xn, s1)?;
            self.ew_dev(s0, s1, s2, n_ff)?;
            self.gemv_launch_raw(q2, p2, h, n_ff, s2, s3)?;
            self.axpy_dev(w, s3, self.moe.dmo.ptr, h)?;
        }
        Ok(())
    }

    /// [B4-1 결함수정 2026-10-13] 슬롯 업로드 — (il,e)의 3투영 q/s를 대상
    /// 포인터 3슬롯(qd/sd, 투영 스트라이드 qsz/ssz)으로 async 복사.
    /// 스테이징(**moe_stage_t 삽입 포함**)과 t=1 스트리밍이 공용한다.
    /// 결함 이력: 종전 t≤8 스테이징은 ids/장치 테이블만 갱신하고 업로드를
    /// 빼먹어 GEMV가 미업로드 슬롯(쓰레기/옛 전문가)을 읽었다 — 워밍업
    /// 프리필이 유령 슬롯을 남기고 이후 t=1 디코드까지 오염(실측).
    fn mcache_upload_to(&mut self, il: usize, e: usize, qd: u64, sd: u64) -> Result<(), String> {
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
        let base = (il * self.moe.n_experts + e) * 3;
        for p in 0..3 {
            let (qp, ql, sp, sl) = self.moe.moe_tab[base + p];
            let qdd = qd + (p * qsz) as u64;
            let sdd = sd + (p * ssz) as u64;
            // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap/호스트
            // 스테이징), ql/sl은 qsz/ssz 형상 계약을 따른다.
            let qb = unsafe { std::slice::from_raw_parts(qp as *const u8, ql as usize) };
            let sb = unsafe { std::slice::from_raw_parts(sp as *const u8, sl as usize) };
            self.cc.h2d_async(qdd, qb)?;
            self.cc.h2d_async(sdd, sb)?;
        }
        Ok(())
    }

    /// [B4-2] 캐시 조회 — hit이면 슬롯 반환 + 나이 갱신.
    fn mcache_lookup(&mut self, il: usize, e: usize) -> Option<usize> {
        self.moe.mcache_tick = self.moe.mcache_tick.wrapping_add(1);
        let base = il * self.moe.mcache_k;
        for slot in 0..self.moe.mcache_k {
            if self.moe.mcache_ids[base + slot] == e as u16 {
                self.moe.mcache_age[base + slot] = self.moe.mcache_tick;
                return Some(slot);
            }
        }
        None
    }

    /// [B4-2] 빈 슬롯 우선, 없으면 LRU(최소 나이) 퇴거 — 새 expert id/나이를
    /// 기록하고 슬롯 반환(호출자가 그 슬롯으로 h2d).
    /// [B4-1] 삽입 시 장치 테이블 엔트리((q,s)쌍 ×3투영, 48B)도 갱신한다 —
    /// t≤8 프리필 경로가 테이블 간접 참조로 이 슬롯을 소비한다.
    fn mcache_insert(&mut self, il: usize, e: usize) -> Result<usize, String> {
        self.moe.mcache_tick = self.moe.mcache_tick.wrapping_add(1);
        let base = il * self.moe.mcache_k;
        let mut slot = 0usize;
        let mut best = u32::MAX;
        for s in 0..self.moe.mcache_k {
            if self.moe.mcache_ids[base + s] == u16::MAX {
                slot = s;
                break;
            }
            if self.moe.mcache_age[base + s] < best {
                best = self.moe.mcache_age[base + s];
                slot = s;
            }
        }
        self.moe.mcache_ids[base + slot] = e as u16;
        self.moe.mcache_age[base + slot] = self.moe.mcache_tick;
        if self.moe.moe_dev_tab != 0 {
            let h = self.hidden;
            let n_ff = self.moe.moe_ffn;
            let qsz = h * n_ff / 2;
            let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
            let qb = self.moe.mcache_q + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
            let sb = self.moe.mcache_s + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
            let mut bytes = [0u8; 48];
            for p in 0..3 {
                let q = qb + (p * qsz) as u64;
                let sv = sb + (p * ssz) as u64;
                bytes[p * 16..p * 16 + 8].copy_from_slice(&q.to_le_bytes());
                bytes[p * 16 + 8..p * 16 + 16].copy_from_slice(&sv.to_le_bytes());
            }
            let off = ((il * self.moe.n_experts + e) * 3) as u64 * 2 * 8;
            self.cc.h2d_async(self.moe.moe_dev_tab + off, &bytes)?;
        }
        Ok(slot)
    }

    /// [B4-1 2026-10-12] 스트리밍 t≤8 전문가 스테이징 — topk_t 기록(moe_idx)을
    /// 판독해 distinct 전문가를 LRU 캐시에 보장(삽입이 장치 테이블 엔트리 갱신).
    /// 이후 전문가 GEMV(융합 t≤8 경로)가 테이블 간접 참조로 동작한다.
    pub(super) fn moe_stage_t(&mut self, il: usize, t: usize) -> Result<(), String> {
        if self.moe.mcache_k == 0 || self.moe.moe_dev_tab == 0 {
            return Err("moe 스트리밍 t>1: 전문가 캐시 미구성".into());
        }
        let tk = self.moe.top_k.max(1);
        let ns = t * tk;
        // [P4 2026-10-16] 핀드 상시 버퍼 — 페이지러블 Vec는 드라이버 내부
        // 스테이징을 거쳐 d2h가 진짜 비동기가 아니다. 핀드 DMA 직행으로
        // 동기 대기 구간을 줄인다(용량 성장 시 재할당).
        let need = ns * 4;
        if self.moe.pin_moe_idx_cap < need {
            if !self.moe.pin_moe_idx.is_null() {
                self.cc.pinned_free(self.moe.pin_moe_idx)?;
            }
            // 여유 있게 — 청크 tmax는 세션 고정이라 2배 성장으로 상각.
            self.moe.pin_moe_idx = self.cc.pinned_alloc(need * 2)?;
            self.moe.pin_moe_idx_cap = need * 2;
        }
        let ib: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(self.moe.pin_moe_idx as *mut u8, need) };
        self.cc.d2h_async(ib.as_mut_ptr(), self.moe.moe_idx, need)?;
        self.cc.sync()?;
        let mut seen = vec![false; self.moe.n_experts];
        let mut distinct = 0usize;
        for j in 0..ns {
            let e = u32::from_le_bytes(ib[j * 4..j * 4 + 4].try_into().unwrap()) as usize;
            if e >= self.moe.n_experts || seen[e] {
                continue;
            }
            seen[e] = true;
            distinct += 1;
        }
        // [P6 2026-10-15] union 가드 — 청크 distinct > 캐시 슬롯이면 스테이징이
        // 캐시를 스래싱해 디바이스 테이블 항목을 무효화한다(조용한 오염 방지 —
        // 즉시 실패). 운용 tmax는 통계적으로 여유를 두고 산출(호스트 prefill_tmax).
        if distinct > self.moe.mcache_k {
            return Err(format!(
                "moe streaming: 청크 distinct {distinct} > 캐시 {} — t 상한 초과",
                self.moe.mcache_k
            ));
        }
        for j in 0..ns {
            let e = u32::from_le_bytes(ib[j * 4..j * 4 + 4].try_into().unwrap()) as usize;
            if e >= self.moe.n_experts || !seen[e] {
                continue;
            }
            if self.mcache_lookup(il, e).is_none() {
                self.moe.mcache_miss += 1;
                // [B4-1 결함수정 2026-10-13] 삽입 = 가중치 업로드 동반(필수).
                let slot = self.mcache_insert(il, e)?;
                let h = self.hidden;
                let n_ff = self.moe.moe_ffn;
                let qsz = h * n_ff / 2;
                let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
                let qb =
                    self.moe.mcache_q + ((il * self.moe.mcache_k + slot) * 3) as u64 * qsz as u64;
                let sb =
                    self.moe.mcache_s + ((il * self.moe.mcache_k + slot) * 3) as u64 * ssz as u64;
                self.mcache_upload_to(il, e, qb, sb)?;
            } else {
                // [§2.M ① 2026-10-15] 프리필 스테이징 히트율 계측(t=1 경로와 동일
                // 계수기 — /stats moe_cache 노출).
                self.moe.mcache_hits += 1;
            }
        }
        Ok(())
    }
}
