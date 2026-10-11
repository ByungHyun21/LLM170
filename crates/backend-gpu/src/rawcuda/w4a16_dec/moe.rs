use super::*;

impl W4a16Dec {
    /// 배치 전문가 GEMV(상주) — tab 간접 참조, grid = n × nslots.
    /// xstride = 0(x 공통) 또는 k(슬롯/토큰별 활성), sp = 토큰당 슬롯 수
    /// (프리필 = top_k → x는 토큰 단위 [t][k], 디코드 = 1 → 슬롯별 [nslots][k]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_experts_launch(
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
        let (mut p_t, mut p_b, mut p_i, mut p_ns) =
            (self.moe_dev_tab, base as i32, self.moe_idx, nslots as i32);
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
    pub(super) fn gemv_experts_glu_launch(
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
        let f = self.cc.function("w4a16_gemv_experts_glu")?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) =
            (self.moe_dev_tab, base as i32, self.moe_idx, nslots as i32);
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

    /// [P11] 전문가-우선 슬롯 순열 발사(프리필 전용) — w4a16_moe_align.
    /// n_exp ≤ 1024 계약(초과 시 호출부가 종전 경로로 폴백).
    pub(super) fn moe_align_launch(&self, nslots: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_align")?;
        let (mut p_i, mut p_ns) = (self.moe_idx, nslots as i32);
        let (mut p_g, mut p_c, mut p_o) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
        let mut p_ne = self.n_experts as i32;
        self.cc.launch(
            f,
            1,
            1,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_i, &mut p_ns, &mut p_g, &mut p_c, &mut p_o, &mut p_ne,
            ),
        )
    }

    /// [P11] 그룹 mma GEMM(g32) 발사 — grid (전문가 × n타일), M=전문가 슬롯 수.
    /// [B2 2026-10-12] down 전용 N64 발사 — grp의 배리어 상각판(그리드 y =
    /// ceil(n/64)). 값은 grp와 동일 계열(골든 판정) — 커널 주석 참조.
    pub(super) fn gemm_g32_mma_grp_d64_launch(
        &self,
        base: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemm_g32_mma_grp_d64")?;
        let (mut p_t, mut p_b) = (self.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
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
            self.n_experts as u32,
            n.div_ceil(64) as u32,
            256,
            &mut crate::rawcuda::args::l11(
                &mut p_t, &mut p_b, &mut p_g, &mut p_c, &mut p_of, &mut p_x, &mut p_xs, &mut p_sp,
                &mut p_o, &mut p_n, &mut p_k,
            ),
        )
    }

    /// [§1.2 2026-10-11] 게이트·업 2N 융합 발사 — 한 커널이 base·base+1 두
    /// 투영을 동시 처리(N64 = 게이트 32 + 업 32). A 스테이징·배리어를 상각.
    /// [B2 2026-10-12] KC32 변형 채택 시 미사용(비교용 보존).
    #[allow(dead_code)]
    pub(super) fn gemm_g32_mma_grp2_launch(
        &self,
        base: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_gate: CUdeviceptr,
        out_up: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemm_g32_mma_grp2")?;
        let (mut p_t, mut p_b) = (self.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
        let (mut p_x, mut p_xs, mut p_sp) = (x_dev, xstride as i32, sp as i32);
        let (mut p_o, mut p_o2, mut p_n, mut p_k) = (out_gate, out_up, n as i32, k as i32);
        self.cc.launch(
            f,
            self.n_experts as u32,
            n.div_ceil(GEMM_GRP_N) as u32,
            256,
            &mut crate::rawcuda::args::l12(
                &mut p_t, &mut p_b, &mut p_g, &mut p_c, &mut p_of, &mut p_x, &mut p_xs, &mut p_sp,
                &mut p_o, &mut p_o2, &mut p_n, &mut p_k,
            ),
        )
    }

    /// 선택 순서 가중 누적 — y[ti] = Σ_{s∈ti} w[s]·d[s][i] (sp = 토큰당 슬롯).
    pub(super) fn moe_accum_dev(
        &mut self,
        w_dev: CUdeviceptr,
        d_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        sp: usize,
        nslots: usize,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_accum")?;
        let (mut p_w, mut p_d, mut p_y) = (w_dev, d_dev, y_dev);
        let (mut p_sp, mut p_ns, mut nn) = (sp as i32, nslots as i32, n as i32);
        // [FLA-6] grid.y = 출력 행(토큰) — 종전 1은 토큰 루프 직렬(99% 유휴).
        let spt = if sp > 0 { sp } else { nslots.max(1) };
        let rows = nslots.div_ceil(spt) as u32;
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            rows,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_d, &mut p_y, &mut p_sp, &mut p_ns, &mut nn,
            ),
        )
    }

    /// MoE FFN(35B-A3B) — 라우터(bf16 GEMV→호스트 top-k) + 전문가 스트리밍
    /// GEMV + shared. 반환 = self.dmo.ptr(잔차 ab로 소비).
    /// 시맨틱은 CPU 스테이지(core qwen35::stages::moe)와 동일: 라우터 전체
    /// softmax → top-k → 재정규화, shared = sigmoid(sgate)·MLP.
    pub(super) fn moe_ffn_dev(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.n_experts;
        if n_exp == 0 || self.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        self.ensure_moe_bufs()?;
        // [2026-10-09 P1] 상주는 라우터를 디바이스에서 마무리(top-k 디바이스) —
        // d2h+sync+호스트 moe_topk+h2d 왕복(층당 1회) 제거, 그래프 캡처 가능화.
        // 스트리밍은 전문가 파일 스테이징에 호스트 선택이 필요해 종전 경로 유지
        // (zero_dev도 axpy 누적 전용 — 상주는 moe_accum이 전 행 덮어씀, P2).
        if self.moe_resident {
            // [B-4 2026-10-10] 라우터 게이트 + shared gate/up를 1런치로
            // (동일 x=xn·상호 무의존) → topk는 게이트 결과(drt) 소비.
            self.moe_multi_gate(il, xn)?;
            self.moe_topk_dev()?;
            self.moe_experts_batch(il, xn, self.top_k)?;
        } else {
            // [B7+B6 2026-10-12] 스트리밍도 multi_tab 재사용 — 라우터+shared
            // 게이트/업/sgate 1런치(종전 4런치: 라우터 + shared gate/up/sgate).
            // topk는 디바이스(topk_t) — 호스트로는 id/wt 8쌍만(종전 로짓 128
            // d2h + 호스트 전체 정렬). 스테이징에 id 선택만 필요해 호스트 몫
            // 최소화. 시맨틱 동일(softmax→k→재정규화 미러) — 골든 판정.
            self.moe_multi_gate(il, xn)?;
            self.moe_topk_dev()?;
            let sel = self.moe_read_selection()?;
            Self::zero_dev(&self.cc, self.dmo.ptr, self.hidden * 4)?;
            self.moe_experts_streaming(il, xn, &sel)?;
        }
        self.moe_shared(il, xn)?;
        Ok(self.dmo.ptr)
    }

    /// 라우터 top-k 디바이스(P1) — 게이트 GEMV는 [B-4] 다중 세그먼트 런치가
    /// 수행(moe_multi_gate). 여기는 drt 로짓 → idx/wt 선택만.
    pub(super) fn moe_topk_dev(&mut self) -> Result<(), String> {
        // [B-3 2026-10-10] t=1도 **워프 병렬 topk_t** — 종전 단일 스레드
        // 커널(R3에서 제거 — 구 w4a16_moe_topk)은 128전문가 softmax+8라운드를
        // 한 스레드가 순차 처리(층당 수µs × 40층). 시맨틱은 동일 미러
        // (gptq4.cu 주석) — 골든으로 판정.
        // [B-4 후속 2026-10-10] n=128은 레지스터 전용 커널(비트동일) —
        // 종전 topk_t t=1 실측 10.2µs(2,474명령·로컬메모리 왕복).
        if self.n_experts == 256 {
            let f = self.cc.function("w4a16_moe_topk256")?;
            let (mut p_lg, mut p_ix, mut p_wt) = (self.drt.ptr, self.moe_idx, self.moe_wt);
            let mut p_k = self.top_k as i32;
            return self.cc.launch(
                f,
                1,
                1,
                32,
                &mut crate::rawcuda::args::l4(&mut p_lg, &mut p_ix, &mut p_wt, &mut p_k),
            );
        }
        let f = self.cc.function("w4a16_moe_topk_t")?;
        let (mut p_lg, mut p_ix, mut p_wt) = (self.drt.ptr, self.moe_idx, self.moe_wt);
        let (mut p_t, mut p_n, mut p_k) = (1i32, self.n_experts as i32, self.top_k as i32);
        self.cc.launch(
            f,
            1,
            1,
            32,
            &mut crate::rawcuda::args::l6(
                &mut p_lg, &mut p_ix, &mut p_wt, &mut p_t, &mut p_n, &mut p_k,
            ),
        )
    }

    /// [B6/B7 2026-10-12] 디바이스 top-k(id/wt, moe_idx/moe_wt) → 호스트 목록.
    /// 스트리밍 스테이징은 id 선택만 필요 — 종전 라우터 로짓 128 d2h+호스트
    /// 정렬(moe_route)을 대체한다(시맨틱은 topk_t가 미러 — 골든 판정).
    fn moe_read_selection(&mut self) -> Result<Vec<(usize, f32)>, String> {
        let tk = self.top_k.max(1);
        let mut ib = vec![0u8; tk * 4];
        let mut wb = vec![0u8; tk * 4];
        self.cc.d2h_async(ib.as_mut_ptr(), self.moe_idx, tk * 4)?;
        self.cc.d2h_async(wb.as_mut_ptr(), self.moe_wt, tk * 4)?;
        self.cc.sync()?;
        let mut out: Vec<(usize, f32)> = Vec::with_capacity(tk);
        for k in 0..tk {
            let e = u32::from_le_bytes(ib[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
            let w = f32::from_le_bytes(wb[k * 4..k * 4 + 4].try_into().unwrap());
            out.push((e, w));
        }
        Ok(out)
    }

    /// 전문가 배치(상주) — 테이블 간접 GEMV + 가중 누적.
    /// [P1] idx/wt는 호출자가 디바이스에 기록한다(moe_route_dev) — 종전
    /// h2d 2회(층당) 제거. gDN 경로(moe_ffn_dev_t)는 호스트 선택이 남아
    /// 이 함수 앞에서 moe_idx/moe_wt를 h2d로 채운 뒤 ns를 넘긴다.
    pub(super) fn moe_experts_batch(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        ns: usize,
    ) -> Result<(), String> {
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        // [B-1/B-2 2026-10-10] gate+up+ew 융합 1런치(x 공통) → down → 누적.
        // ew 커널(실측 0.35ms/토큰)과 중간 버퍼 왕복·x 재판독 제거 — act는
        // 융합 커널이 직접 기록(산술 비트동일).
        let base = il * n_exp * 3;
        self.gemv_experts_glu_launch(base, ns, xn, 0, ns, self.exp.act, n_ff, h)?;
        self.gemv_experts_launch(base + 2, ns, self.exp.act, n_ff, 1, self.exp.dn, h, n_ff)?;
        // [§4.5] 누적은 꼬리(shared down 이후)의 accum_sg가 수행 — 호출부 참조.
        Ok(())
    }

    /// 전문가 스트리밍(비상주) — 스테이징 1쌍 재사용 h2d + 개별 GEMV.
    pub(super) fn moe_experts_streaming(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        sel: &[(usize, f32)],
    ) -> Result<(), String> {
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        if sel.is_empty() {
            return Ok(());
        }
        // [P7 2026-10-09] 층 단위 배치 스테이징 — (top_k×3) q/s 슬라이스를
        // async 복사로 전부 올린 뒤 sync 1회(종전 proj마다 h2d_chunked —
        // proj당 sync 2회 × 3 × top_k = 층당 수십 sync). 소스는 호스트
        // 스테이징(C2)·mmap — 둘 다 호출 내 수명이면 충분(드라이버 스테이징).
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe_group * 2;
        debug_assert!(
            self.stg.cap.0 >= qsz * sel.len() * 3 && self.stg.cap.1 >= ssz * sel.len() * 3
        );
        // [B4-2 2026-10-12] 전문가 LRU 캐시 — hit은 h2d 생략(가중치 불변이라
        // 수치 무변화), miss는 캐시 슬롯(빈/LRU 퇴거)으로 직접 h2d. 캐시 없음
        // (mcache_k=0)이면 종전 stg 경로 그대로.
        let use_cache = self.mcache_k > 0;
        let mut qd_list: Vec<(u64, u64)> = Vec::with_capacity(sel.len());
        for (j, &(e, _)) in sel.iter().enumerate() {
            let (qd, sd, need_h2d) = if use_cache {
                match self.mcache_lookup(il, e) {
                    Some(slot) => {
                        self.mcache_hits += 1;
                        let qb =
                            self.mcache_q + ((il * self.mcache_k + slot) * 3) as u64 * qsz as u64;
                        let sb =
                            self.mcache_s + ((il * self.mcache_k + slot) * 3) as u64 * ssz as u64;
                        (qb, sb, false)
                    }
                    None => {
                        self.mcache_miss += 1;
                        let slot = self.mcache_insert(il, e);
                        let qb =
                            self.mcache_q + ((il * self.mcache_k + slot) * 3) as u64 * qsz as u64;
                        let sb =
                            self.mcache_s + ((il * self.mcache_k + slot) * 3) as u64 * ssz as u64;
                        (qb, sb, true)
                    }
                }
            } else {
                let qb = self.stg.q + ((j * 3) * qsz) as u64;
                let sb = self.stg.s + ((j * 3) * ssz) as u64;
                (qb, sb, true)
            };
            if need_h2d {
                let base = (il * n_exp + e) * 3;
                for p in 0..3 {
                    let (qp, ql, sp, sl) = self.moe_tab[base + p];
                    let qdd = qd + (p * qsz) as u64;
                    let sdd = sd + (p * ssz) as u64;
                    // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap/호스트
                    // 스테이징), ql/sl은 위 qsz/ssz 형상 계약을 따른다.
                    let qb = unsafe { std::slice::from_raw_parts(qp as *const u8, ql as usize) };
                    let sb = unsafe { std::slice::from_raw_parts(sp as *const u8, sl as usize) };
                    self.cc.h2d_async(qdd, qb)?;
                    self.cc.h2d_async(sdd, sb)?;
                }
            }
            qd_list.push((qd, sd));
        }
        self.cc.sync()?;
        if use_cache
            && llm170_diag::flag::on_nonzero("LLM170_MOE_DBG")
            && self.mcache_hits + self.mcache_miss > 0
            && (self.mcache_hits + self.mcache_miss).is_multiple_of(2560)
        {
            eprintln!(
                "[mcache] hit {} / {} ({:.1}%)",
                self.mcache_hits,
                self.mcache_hits + self.mcache_miss,
                100.0 * self.mcache_hits as f64 / (self.mcache_hits + self.mcache_miss) as f64
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
            self.axpy_dev(w, s3, self.dmo.ptr, h)?;
        }
        Ok(())
    }

    /// [B4-2] 캐시 조회 — hit이면 슬롯 반환 + 나이 갱신.
    fn mcache_lookup(&mut self, il: usize, e: usize) -> Option<usize> {
        self.mcache_tick = self.mcache_tick.wrapping_add(1);
        let base = il * self.mcache_k;
        for slot in 0..self.mcache_k {
            if self.mcache_ids[base + slot] == e as u16 {
                self.mcache_age[base + slot] = self.mcache_tick;
                return Some(slot);
            }
        }
        None
    }

    /// [B4-2] 빈 슬롯 우선, 없으면 LRU(최소 나이) 퇴거 — 새 expert id/나이를
    /// 기록하고 슬롯 반환(호출자가 그 슬롯으로 h2d).
    fn mcache_insert(&mut self, il: usize, e: usize) -> usize {
        self.mcache_tick = self.mcache_tick.wrapping_add(1);
        let base = il * self.mcache_k;
        let mut slot = 0usize;
        let mut best = u32::MAX;
        for s in 0..self.mcache_k {
            if self.mcache_ids[base + s] == u16::MAX {
                slot = s;
                break;
            }
            if self.mcache_age[base + s] < best {
                best = self.mcache_age[base + s];
                slot = s;
            }
        }
        self.mcache_ids[base + slot] = e as u16;
        self.mcache_age[base + slot] = self.mcache_tick;
        slot
    }

    /// [B-4 2026-10-10] 다중 세그먼트 GEMV 테이블 — 라우터 게이트 + shared
    /// gate/up(x=xn 공통·상호 무의존)을 1런치로 묶는다. 소형 커널 런치
    /// 플로어 제거(35B 실측: 게이트+shared 4런치 ~18µs → 1런치 ~5µs, 층당).
    /// 포인터는 버퍼 수명 내 불변 — 그래프 캡처가 테이블을 참조한다.
    pub(super) fn build_moe_multi_tab(&mut self) -> Result<(), String> {
        let nl = self.n_layers;
        // [B7 2026-10-12] shared 게이트/업은 전용 슬롯(dsh)으로 — dchain은
        // 스트리밍 전문가 스크래치와 충돌(게이트/업이 전문가보다 선계산).
        let (d0, d1) = (self.dsh[0], self.dsh[1]);
        let drt = self.drt.ptr;
        let shared = self.shared_ffn > 0;
        let mut ent: Vec<u64> = Vec::with_capacity(nl * 4 * 4);
        let mut rows: Vec<u32> = Vec::with_capacity(nl);
        let mut rows_t3: Vec<u32> = Vec::with_capacity(nl);
        for il in 0..nl {
            let (gw, gn, gk) = self.plain_spec(&format!("blk.{il}.moe_gate.weight"))?;
            ent.extend([gw, drt, gn as u64, gk as u64]);
            let mut r = gn as u32;
            let mut rt3 = gn.div_ceil(GEMV_TR) as u32;
            if shared {
                for (suf, out) in [("moe_shared_gate", d0), ("moe_shared_up", d1)] {
                    let (w, n, k) = self.plain_spec(&format!("blk.{il}.{suf}.weight"))?;
                    ent.extend([w, out, n as u64, k as u64]);
                    r += n as u32;
                    rt3 += n.div_ceil(GEMV_TR) as u32;
                }
                // [§4.5 2026-10-11] sgate를 4세그먼트로 편입 — 꼬리 GEMV
                // 런치 -1/층. [B3-2b 2026-10-12] 출력은 전용 슬롯(dsg)으로
                // 통일 + t 경로(rows_t3)에도 편입 — t≤8 꼬리 런치 추가 -1.
                let (w, n, k) = self.plain_spec(&format!("blk.{il}.moe_shared_sgate.weight"))?;
                ent.extend([w, self.dsg, n as u64, k as u64]);
                r += n as u32;
                rt3 += n.div_ceil(GEMV_TR) as u32;
            } else {
                ent.extend([0u64; 12]);
            }
            rows.push(r);
            rows_t3.push(rt3);
        }
        let bytes = ent.len() * 8;
        if self.moe_multi_tab != 0 {
            self.cc.free(self.moe_multi_tab)?;
            self.moe_multi_tab = 0;
        }
        let p = self.cc.alloc(bytes)?;
        let mut buf: Vec<u8> = Vec::with_capacity(bytes);
        for v in &ent {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(e) = Self::h2d_chunked(&self.cc, p, &buf) {
            let _ = self.cc.free(p);
            return Err(e);
        }
        self.moe_multi_tab = p;
        self.moe_multi_rows = rows;
        self.moe_multi_rows_t3 = rows_t3;
        Ok(())
    }

    /// [S10-잔여 2026-10-11] t행(2..8) 상주 라우터+shared gate/up 융합 —
    /// t=1 multi_gate의 t판(moe_multi_tab 세그먼트 0..2 공용). 종전 층당
    /// 3런치(라우터 게이트 + shared gate/up plain_gemm) → 1런치.
    /// [B3-2b 2026-10-12] sgate를 4세그먼트로 확장(count=4) — 꼬리 런치 -1/층
    /// (출력은 dsg 전용 슬롯). 값은 단독 GemvT와 비트동일(커널 계약).
    pub(super) fn moe_multi_gate_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.ensure_moe_bufs()?;
        let total = *self
            .moe_multi_rows_t3
            .get(il)
            .ok_or("moe: 다중 세그먼트 t 테이블 미작성")?;
        if total == 0 {
            return Ok(());
        }
        self.gemv_multi_t_launch(
            true,
            self.moe_multi_tab + (il as u64) * 4 * 4 * 8,
            xn,
            total as usize,
            4,
            t,
        )
    }

    /// [S10-잔여] shared 꼬리(t판) — ew → down → 가산([B3-2b 2026-10-12]
    /// sgate는 multi_gate_t 4세그먼트가 dsg에 선기록 — 런치 -1/층). 게이트/
    /// 업은 multi_gate_t가 선기록(moe_shared_t에서 gate/up 런치 제거판).
    /// [B3] forked=true면 머리(ew/down)가 보조 스트림에 선발사됨 — 조인 후
    /// 가산만(조인 대상은 dsg·s3).
    fn moe_shared_tail_t(
        &mut self,
        il: usize,
        _xn: CUdeviceptr,
        t: usize,
        forked: bool,
    ) -> Result<(), String> {
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        if forked {
            // [B3] 보조 스트림 조인 후 가산(ew/down은 포크가 선발사).
            let e2 = self.ev_fork[il * 2 + 1];
            self.cc.stream_wait(self.cc.stream, e2)?;
            return self.shared_add_dev(self.dsg, s3, self.dmo.ptr, h, t);
        }
        self.ew_dev(self.dsh[0], self.dsh[1], s2, t * sf)?;
        // [S10-잔여 확장] down은 t행 GEMV — v3 GEMM 대비 동값
        // (plain-gemm-selfcheck 0/65536 실측)·소형 t에서 우세.
        self.plain_gemv_t_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.shared_add_dev(self.dsg, s3, self.dmo.ptr, h, t)
    }

    /// [B-4] 다중 세그먼트 GEMV 런치(라우터 게이트 + shared gate/up).
    pub(super) fn moe_multi_gate(&mut self, il: usize, xn: CUdeviceptr) -> Result<(), String> {
        self.ensure_moe_bufs()?;
        let rows = *self
            .moe_multi_rows
            .get(il)
            .ok_or("moe: 다중 세그먼트 테이블 미작성")?;
        if rows == 0 {
            return Ok(());
        }
        let f = self.cc.function("w4a16_gemv_multi")?;
        let (mut p_tab, mut p_x, mut p_c) =
            (self.moe_multi_tab + (il as u64) * 4 * 4 * 8, xn, 4i32);
        self.cc.launch(
            f,
            rows,
            1,
            64,
            &mut crate::rawcuda::args::l3(&mut p_tab, &mut p_x, &mut p_c),
        )
    }

    /// shared 전문가 — sigmoid(sgate·xn)·down(silu(gate·xn)·up·xn).
    /// [B7 2026-10-12] 게이트/업/sgate는 multi_gate(multi_tab 4세그먼트)가
    /// 선기록(상주·스트리밍 공통) — 꼬리만 수행(스트리밍 분기 런치 -3/층).
    pub(super) fn moe_shared(&mut self, il: usize, _xn: CUdeviceptr) -> Result<(), String> {
        if self.shared_ffn == 0 {
            return Ok(());
        }
        self.moe_shared_tail(il)
    }

    /// shared 꼬리 — ew(silu·mul) → down → 누적(sgate는 multi_gate 4세그먼트가
    /// dsg에 선기록 — 런치 -1/층).
    fn moe_shared_tail(&mut self, il: usize) -> Result<(), String> {
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        self.ew_dev(self.dsh[0], self.dsh[1], s2, sf)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3)?;
        // [B7] sgate는 multi_gate 4세그먼트([B3-2b] 출력 dsg)가 선기록 —
        // 상주·스트리밍 공통.
        let sg = self.dsg;
        if self.moe_resident {
            // [§4.5] 상주 — 전문가 누적 + shared 가산 융합(런치 -1/층).
            // 누적은 여기(shared down 이후)로 이동 — 값 순서 동일(acc + σ·s3).
            self.moe_accum_sg_dev(
                self.moe_wt,
                self.exp.dn,
                sg,
                s3,
                self.dmo.ptr,
                self.top_k.max(1),
                h,
            )?;
        } else {
            self.shared_add_dev(sg, s3, self.dmo.ptr, h, 1)?;
        }
        Ok(())
    }

    /// MoE FFN 배치(t≤8) — 라우터 플레인 GEMM → 디바이스 top-k(t×top_k 슬롯,
    /// 토큰 우선) → 전문가 배치 GEMV(gate/up x=토큰 단위 sp=top_k, down x=슬롯
    /// 단위 sp=1) → 토큰별 누적 → shared. **상주 모드 전용**(스트리밍 프리필은
    /// 미구현 — 호출부가 t=1로 떨어뜨린다).
    pub(super) fn moe_ffn_dev_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let tk = self.top_k;
        if n_exp == 0 || self.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        if !self.moe_resident {
            return Err("moe t>1: 상주 모드 전용(스트리밍 프리필 미구현)".into());
        }
        self.ensure_moe_bufs()?;
        // [S10-잔여 2026-10-11] t≤8(디코드 배치·검증): 라우터+shared 게이트/
        // 업/sgate를 multi_t 1런치로 융합(종전 4런치). t>8(프리필)은 종전 유지.
        let fused = self.shared_ffn > 0 && t <= BATCH_DEC_MAX;
        // 1) 라우터 [t][n_exp] — 플레인 GEMM → **디바이스 top-k**(P11).
        // 종전: d2h(32KB)+sync+호스트 전체 정렬(512×t)이 층·청크마다 — 프리필의
        // ~10%. 시맨틱은 moe_topk 미러(softmax→k라운드→재정규화).
        // 실측(2026-10-09): 35B 512토큰 프리필 1294→1088ms.
        if fused {
            self.moe_multi_gate_t(il, xn, t)?;
        } else {
            self.plain_gemm_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt.ptr, t)?;
        }
        // [B3 2026-10-12] shared 꼬리 선포크 — ew→down→sgate(게이트/업은
        // multi_gate_t가 선기록)를 보조 스트림에 선발사해 topk·전문가 GEMV와
        // 병렬화한다. 조인은 shared_add 직전(꼬리). sgate 출력은 전용 슬롯
        // (dsg — 종전 drt 재사용은 topk와의 경합). 실측 상한(꼬리 제거):
        // tg +4.1%·np4 +5.6% — 기대 실효는 발사 몫 회수분.
        let forked = if fused {
            self.moe_shared_head_t_fork(il, xn, t)?
        } else {
            false
        };
        {
            let f = self.cc.function("w4a16_moe_topk_t")?;
            let (mut p_lg, mut p_ix, mut p_wt) = (self.drt.ptr, self.moe_idx, self.moe_wt);
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
        let ns = t * tk;
        // 2) 전문가 배치.
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            // xn(정규화 출력) 행별 NaN — 업스트림 vs 전문가 GEMV 판별.
            let mut vb = vec![0u8; t * h * 4];
            self.cc.d2h_async(vb.as_mut_ptr(), xn, t * h * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let rows: Vec<usize> = (0..t)
                .filter(|&r| v[r * h..(r + 1) * h].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] xn nan-rows={rows:?} t={t}");
        }
        // (P11) idx/wt는 이미 디바이스에 있다(topk_t) — h2d 없음.
        let base = il * n_exp * 3;
        // [P11] 프리필(t>1)은 전문가-우선 정렬 + 그룹 GEMV — 같은 전문가의
        // 슬롯을 연속 처리해 가중치 행을 L2 재사용(슬롯별 산술 동일 = 비트 동일).
        // [S10-잔여 확장 2026-10-11] t≤8(np4·검증)은 GEMV 전문가 경로 —
        // t=1 단독 디코드와 **동일 커널 계열**(f32 x·f64 트리)로 통일해 배치
        // 디코드 = 단독 디코드 일관성을 회복하고 +8.1% np4. grp2(mma·f16
        // 스테이징)와는 수치 클래스 상이 — eval chunk4 A/B: en PPL +0.1697%·
        // agree 98.566% / ko +0.0091%·97.337%(밴드: int8 KV 채택 이력보다 작음).
        // t≥9(프리필)는 종전 grp2 유지.
        let group = t > 8 && n_exp <= 1024;
        if group {
            // [P11] 그룹 mma GEMM — 전문가별 슬롯 묶음(M=슬롯 수), T1 계약 미러.
            self.moe_align_launch(ns)?;
            // [S13/S5 2026-10-11] 라우팅 분포 진단(LLM170_DUMP=moe_cnt) —
            // gcnt 히스토그램 + Σceil(cnt/GRP_M) 타일 수(M64/32/16 비교).
            if llm170_diag::dump::opts().key("moe_cnt") {
                let mut cb = vec![0u8; n_exp * 4];
                self.cc
                    .d2h_async(cb.as_mut_ptr(), self.moe_gcnt, n_exp * 4)?;
                self.cc.sync()?;
                let cnts: Vec<u32> = cb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                let mut hist = [0usize; 8];
                let mut sum = 0usize;
                let (mut t64, mut t32, mut t16) = (0usize, 0usize, 0usize);
                for &c in &cnts {
                    let c = c as usize;
                    sum += c;
                    t64 += c.div_ceil(64);
                    t32 += c.div_ceil(32);
                    t16 += c.div_ceil(16);
                    let b = match c {
                        0 => 0,
                        1..=8 => 1,
                        9..=16 => 2,
                        17..=32 => 3,
                        33..=48 => 4,
                        49..=64 => 5,
                        65..=96 => 6,
                        _ => 7,
                    };
                    hist[b] += 1;
                }
                eprintln!(
                    "[moe-cnt] il={il} slots={ns} sum={sum} tiles M64={t64} M32={t32} M16={t16} hist[0|1-8|9-16|17-32|33-48|49-64|65-96|97+]={hist:?}"
                );
            }
            // [marlin-A5] A는 f16(xn 미러) — 그룹 GEMM A 재판독 절반.
            let x16 = self.cast_x16(xn, t * h)?;
            // [§1.2 2026-10-11] 게이트·업 2N 융합 — 발사 1회(A 스테이징·배리어 상각).
            self.gemm_g32_mma_grp2_launch(base, x16, h, tk, self.exp.gate, self.exp.up, n_ff, h)?;
        } else {
            self.gemv_experts_launch(base, ns, xn, h, tk, self.exp.gate, n_ff, h)?;
            self.gemv_experts_launch(base + 1, ns, xn, h, tk, self.exp.up, n_ff, h)?;
        }
        if group {
            // [4.1 2026-10-11] ew가 f16 미러 동시 기록 — cast_x16 노드 제거.
            self.ew_dev_mirror(self.exp.gate, self.exp.up, self.exp.act, ns * n_ff)?;
        } else {
            self.ew_dev(self.exp.gate, self.exp.up, self.exp.act, ns * n_ff)?;
        }
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            let mut vb = vec![0u8; ns * n_ff * 4];
            self.cc
                .d2h_async(vb.as_mut_ptr(), self.exp.act, ns * n_ff * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let bad: Vec<usize> = (0..ns)
                .filter(|&s| v[s * n_ff..(s + 1) * n_ff].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] act nan-slots={bad:?} ns={ns}");
        }
        if group {
            // [B2 2026-10-12] down N64 변형 — grp 배리어 상각(골든 판정).
            self.gemm_g32_mma_grp_d64_launch(
                base + 2,
                self.dx16.ptr,
                n_ff,
                1,
                self.exp.dn,
                h,
                n_ff,
            )?;
        } else {
            self.gemv_experts_launch(base + 2, ns, self.exp.act, n_ff, 1, self.exp.dn, h, n_ff)?;
        }
        self.moe_accum_dev(self.moe_wt, self.exp.dn, self.dmo.ptr, tk, ns, h)?;
        if fused {
            self.moe_shared_tail_t(il, xn, t, forked)?;
        } else {
            self.moe_shared_t(il, xn, t)?;
        }
        Ok(self.dmo.ptr)
    }

    /// [B3 2026-10-12] shared 꼬리 머리(ew→down→sgate)를 보조 스트림에 선발사.
    /// 반환 true면 ew/down/sgate가 이미 발사됨(조인은 moe_shared_tail_t).
    fn moe_shared_head_t_fork(
        &mut self,
        il: usize,
        _xn: CUdeviceptr,
        t: usize,
    ) -> Result<bool, String> {
        let aux = self.cc.stream2;
        if aux.is_null() || self.dsg == 0 || self.ev_fork.len() < (il + 1) * 2 {
            return Ok(false);
        }
        let sf = self.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        let (e1, e2) = (self.ev_fork[il * 2], self.ev_fork[il * 2 + 1]);
        // 포크: 게이트/업(multi_gate_t) 완료 → 보조 스트림에 ew/down.
        // (sgate는 [B3-2b] multi_gate_t 4세그먼트가 dsg에 선기록.)
        self.cc.event_record_on(e1, self.cc.stream)?;
        self.cc.stream_wait(aux, e1)?;
        self.ew_dev_on(aux, self.dsh[0], self.dsh[1], s2, t * sf)?;
        self.plain_gemv_t_launch_on(aux, &format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.cc.event_record_on(e2, aux)?;
        Ok(true)
    }

    /// shared 전문가 배치(t≤8) — 플레인 GEMM ×3 + 토큰별 게이트 가산.
    pub(super) fn moe_shared_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        if self.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0, t)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1, t)?;
        self.ew_dev(s0, s1, s2, t * sf)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.plain_gemm_launch(
            &format!("blk.{il}.moe_shared_sgate.weight"),
            xn,
            self.drt.ptr,
            t,
        )?;
        self.shared_add_dev(self.drt.ptr, s3, self.dmo.ptr, h, t)
    }
}
