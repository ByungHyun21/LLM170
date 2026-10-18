use super::*;

mod moe_align;
mod moe_gemm;
mod moe_gemv;

impl W4a16Dec {
    /// [gemma4 P2 2026-10-14] per-expert 사전 스케일 — top-k 가중치(재정규화 후)에
    /// μ_e를 곱한다(HF Gemma4TextRouter 순서). 호스트 검증용 직접 런처
    /// (`--moe-pes-check` — ew_gelu_host와 동일 패턴).
    pub fn moe_pes_scale_host(
        &mut self,
        wt: &[f32],
        idx: &[i32],
        pes: &[f32],
    ) -> Result<Vec<f32>, String> {
        if wt.is_empty() || wt.len() != idx.len() {
            return Err(format!("pes: wt={} idx={} 계약 위반", wt.len(), idx.len()));
        }
        let n = wt.len();
        let dwt = self.cc.alloc(n * 4)?;
        let dix = self.cc.alloc(n * 4)?;
        let dpes = self.cc.alloc(pes.len().max(1) * 4)?;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32b = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let r = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dwt, &f32b(wt))?;
            self.cc.h2d(dix, &i32b(idx))?;
            self.cc.h2d(dpes, &f32b(pes))?;
            let f = self.cc.function("w4a16_moe_pes_scale")?;
            let (mut a0, mut a1, mut a2, mut a3) = (dwt, dix, dpes, n as i32);
            self.cc.launch(
                f,
                1,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut a3),
            )?;
            self.cc.sync()?;
            let mut ob = vec![0u8; n * 4];
            self.cc.d2h(&mut ob, dwt)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dwt);
        let _ = self.cc.free(dix);
        let _ = self.cc.free(dpes);
        r
    }

    /// 선택 순서 가중 누적 — y[ti] = Σ_{s∈ti} w[s]·d[s][i] (sp = 토큰당 슬롯).
    pub(in crate::rawcuda::w4a16_dec) fn moe_accum_dev(
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
    /// GEMV + shared. 반환 = self.moe.dmo.ptr(잔차 ab로 소비).
    /// 시맨틱은 CPU 스테이지(core qwen35::stages::moe)와 동일: 라우터 전체
    /// softmax → top-k → 재정규화, shared = sigmoid(sgate)·MLP.
    pub(in crate::rawcuda::w4a16_dec) fn moe_ffn_dev(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.moe.n_experts;
        if n_exp == 0 || self.moe.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        self.ensure_moe_bufs()?;
        // [2026-10-09 P1] 상주는 라우터를 디바이스에서 마무리(top-k 디바이스) —
        // d2h+sync+호스트 moe_topk+h2d 왕복(층당 1회) 제거, 그래프 캡처 가능화.
        // 스트리밍은 전문가 파일 스테이징에 호스트 선택이 필요해 종전 경로 유지
        // (zero_dev도 axpy 누적 전용 — 상주는 moe_accum이 전 행 덮어씀, P2).
        if self.moe.moe_resident {
            // [B-4 2026-10-10] 라우터 게이트 + shared gate/up를 1런치로
            // (동일 x=xn·상호 무의존) → topk는 게이트 결과(drt) 소비.
            self.moe_multi_gate(il, xn)?;
            self.moe_topk_dev()?;
            self.moe_experts_batch(il, xn, self.moe.top_k)?;
        } else {
            // [B7+B6 2026-10-12] 스트리밍도 multi_tab 재사용 — 라우터+shared
            // 게이트/업/sgate 1런치(종전 4런치: 라우터 + shared gate/up/sgate).
            // topk는 디바이스(topk_t) — 호스트로는 id/wt 8쌍만(종전 로짓 128
            // d2h + 호스트 전체 정렬). 스테이징에 id 선택만 필요해 호스트 몫
            // 최소화. 시맨틱 동일(softmax→k→재정규화 미러) — 골든 판정.
            self.moe_multi_gate(il, xn)?;
            self.moe_topk_dev()?;
            let sel = self.moe_read_selection()?;
            Self::zero_dev(&self.cc, self.moe.dmo.ptr, self.hidden * 4)?;
            self.moe_experts_streaming(il, xn, &sel)?;
        }
        self.moe_shared(il, xn)?;
        Ok(self.moe.dmo.ptr)
    }

    /// [B-4 2026-10-10] 다중 세그먼트 GEMV 테이블 — 라우터 게이트 + shared
    /// gate/up(x=xn 공통·상호 무의존)을 1런치로 묶는다. 소형 커널 런치
    /// 플로어 제거(35B 실측: 게이트+shared 4런치 ~18µs → 1런치 ~5µs, 층당).
    /// 포인터는 버퍼 수명 내 불변 — 그래프 캡처가 테이블을 참조한다.
    pub(in crate::rawcuda::w4a16_dec) fn build_moe_multi_tab(&mut self) -> Result<(), String> {
        let nl = self.n_layers;
        // [B7 2026-10-12] shared 게이트/업은 전용 슬롯(dsh)으로 — dchain은
        // 스트리밍 전문가 스크래치와 충돌(게이트/업이 전문가보다 선계산).
        let (d0, d1) = (self.moe.dsh[0], self.moe.dsh[1]);
        let drt = self.moe.drt.ptr;
        let shared = self.moe.shared_ffn > 0;
        // [k2 (b) 2026-10-17] 층 창 — 창 층만 테이블에 기입한다(엔트리는
        // 층당 16u64 고정 폭이라 창 밖은 제로 행 유지 — 실행 경로가 닿지
        // 않는다). 기본 전체창 → 내용 불변.
        let win = self.window;
        let mut ent: Vec<u64> = Vec::with_capacity(nl * 4 * 4);
        let mut rows: Vec<u32> = vec![0; nl];
        let mut rows_t3: Vec<u32> = vec![0; nl];
        for il in 0..nl {
            if !win.contains(il) {
                ent.extend([0u64; 16]);
                continue;
            }
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
                ent.extend([w, self.moe.dsg, n as u64, k as u64]);
                r += n as u32;
                rt3 += n.div_ceil(GEMV_TR) as u32;
            } else {
                ent.extend([0u64; 12]);
            }
            rows[il] = r;
            rows_t3[il] = rt3;
        }
        let bytes = ent.len() * 8;
        if self.moe.moe_multi_tab != 0 {
            self.cc.free(self.moe.moe_multi_tab)?;
            self.moe.moe_multi_tab = 0;
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
        self.moe.moe_multi_tab = p;
        self.moe.moe_multi_rows = rows;
        self.moe.moe_multi_rows_t3 = rows_t3;
        Ok(())
    }

    /// [S10-잔여 2026-10-11] t행(2..8) 상주 라우터+shared gate/up 융합 —
    /// t=1 multi_gate의 t판(moe_multi_tab 세그먼트 0..2 공용). 종전 층당
    /// 3런치(라우터 게이트 + shared gate/up plain_gemm) → 1런치.
    /// [B3-2b 2026-10-12] sgate를 4세그먼트로 확장(count=4) — 꼬리 런치 -1/층
    /// (출력은 dsg 전용 슬롯). 값은 단독 GemvT와 비트동일(커널 계약).
    pub(in crate::rawcuda::w4a16_dec) fn moe_multi_gate_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.ensure_moe_bufs()?;
        let total = *self
            .moe
            .moe_multi_rows_t3
            .get(il)
            .ok_or("moe: 다중 세그먼트 t 테이블 미작성")?;
        if total == 0 {
            return Ok(());
        }
        self.gemv_multi_t_launch(
            true,
            self.moe.moe_multi_tab + (il as u64) * 4 * 4 * 8,
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
        let sf = self.moe.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        if forked {
            // [B3] 보조 스트림 조인 후 가산(ew/down은 포크가 선발사).
            let e2 = self.moe.ev_fork[il * 2 + 1];
            self.cc.stream_wait(self.cc.stream, e2)?;
            return self.shared_add_dev(self.moe.dsg, s3, self.moe.dmo.ptr, h, t);
        }
        self.ew_dev(self.moe.dsh[0], self.moe.dsh[1], s2, t * sf)?;
        // [S10-잔여 확장] down은 t행 GEMV — v3 GEMM 대비 동값
        // (plain-gemm-selfcheck 0/65536 실측)·소형 t에서 우세.
        self.plain_gemv_t_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.shared_add_dev(self.moe.dsg, s3, self.moe.dmo.ptr, h, t)
    }

    /// [B-4] 다중 세그먼트 GEMV 런치(라우터 게이트 + shared gate/up).
    pub(in crate::rawcuda::w4a16_dec) fn moe_multi_gate(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<(), String> {
        self.ensure_moe_bufs()?;
        let rows = *self
            .moe
            .moe_multi_rows
            .get(il)
            .ok_or("moe: 다중 세그먼트 테이블 미작성")?;
        if rows == 0 {
            return Ok(());
        }
        let f = self.cc.function("w4a16_gemv_multi")?;
        let (mut p_tab, mut p_x, mut p_c) =
            (self.moe.moe_multi_tab + (il as u64) * 4 * 4 * 8, xn, 4i32);
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
    pub(in crate::rawcuda::w4a16_dec) fn moe_shared(
        &mut self,
        il: usize,
        _xn: CUdeviceptr,
    ) -> Result<(), String> {
        if self.moe.shared_ffn == 0 {
            return Ok(());
        }
        self.moe_shared_tail(il)
    }

    /// shared 꼬리 — ew(silu·mul) → down → 누적(sgate는 multi_gate 4세그먼트가
    /// dsg에 선기록 — 런치 -1/층).
    fn moe_shared_tail(&mut self, il: usize) -> Result<(), String> {
        let h = self.hidden;
        let sf = self.moe.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        self.ew_dev(self.moe.dsh[0], self.moe.dsh[1], s2, sf)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3)?;
        // [B7] sgate는 multi_gate 4세그먼트([B3-2b] 출력 dsg)가 선기록 —
        // 상주·스트리밍 공통.
        let sg = self.moe.dsg;
        if self.moe.moe_resident {
            // [§4.5] 상주 — 전문가 누적 + shared 가산 융합(런치 -1/층).
            // 누적은 여기(shared down 이후)로 이동 — 값 순서 동일(acc + σ·s3).
            self.moe_accum_sg_dev(
                self.moe.moe_wt,
                self.moe.exp.dn,
                sg,
                s3,
                self.moe.dmo.ptr,
                self.moe.top_k.max(1),
                h,
            )?;
        } else {
            self.shared_add_dev(sg, s3, self.moe.dmo.ptr, h, 1)?;
        }
        Ok(())
    }

    /// MoE FFN 배치(t≤8) — 라우터 플레인 GEMM → 디바이스 top-k(t×top_k 슬롯,
    /// 토큰 우선) → 전문가 배치 GEMV(gate/up x=토큰 단위 sp=top_k, down x=슬롯
    /// 단위 sp=1) → 토큰별 누적 → shared. **상주 모드 전용**(스트리밍 프리필은
    /// 미구현 — 호출부가 t=1로 떨어뜨린다).
    pub(in crate::rawcuda::w4a16_dec) fn moe_ffn_dev_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.moe.n_experts;
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let tk = self.moe.top_k;
        if n_exp == 0 || self.moe.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        // [B4-1 2026-10-12 → P6 2026-10-15] 스트리밍 t≤8은 캐시 스테이징 + 장치
        // 테이블 융합 경로(종전 t=1 폴백 — 512토큰 ~9.3s). **t>8 해제** — 그룹
        // 경로(grp2/grp_d64)는 moe_dev_tab 간접 판독이라 스트리밍 캐시 테이블과
        // 동일 계약(레이아웃 [il][n_exp][3쌍] — mcache_insert가 갱신). 정확성
        // 상한 = 청크 distinct 전문가 ≤ 캐시 슬롯(moe_stage_t 가드) — 초과 시
        // 스테이징 스래싱이 테이블을 무효화하므로 가드가 즉시 실패시킨다.
        // tmax 운용점(호스트 prefill_tmax STREAM=12)은 가드와 함께 산출됨.
        self.ensure_moe_bufs()?;
        // [S10-잔여 2026-10-11] t≤8(디코드 배치·검증): 라우터+shared 게이트/
        // 업/sgate를 multi_t 1런치로 융합(종전 4런치). t>8(프리필)은 종전 유지.
        let fused = self.moe.shared_ffn > 0 && t <= BATCH_DEC_MAX;
        // 1) 라우터 [t][n_exp] — 플레인 GEMM → **디바이스 top-k**(P11).
        // 종전: d2h(32KB)+sync+호스트 전체 정렬(512×t)이 층·청크마다 — 프리필의
        // ~10%. 시맨틱은 moe_topk 미러(softmax→k라운드→재정규화).
        // 실측(2026-10-09): 35B 512토큰 프리필 1294→1088ms.
        if fused {
            self.moe_multi_gate_t(il, xn, t)?;
        } else {
            self.plain_gemm_launch(
                &format!("blk.{il}.moe_gate.weight"),
                xn,
                self.moe.drt.ptr,
                t,
            )?;
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
            let (mut p_lg, mut p_ix, mut p_wt) =
                (self.moe.drt.ptr, self.moe.moe_idx, self.moe.moe_wt);
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
        // [B4-1] 스트리밍: topk_t idx의 distinct 전문가를 캐시에 보장(장치
        // 테이블 갱신 완료) — 이후 GEMV가 테이블 간접 참조로 소비.
        // [P4 2026-10-16 판정] cast_hilo 선포크(d2h 앞 배치)는 실측 기각 —
        // 패킹(3-pass, t=512에서 ~50-100µs)이 d2h+호스트 루프보다 길어 버블을
        // 채우지 못하고 sync 임계경로에 가산(pp1024 −4.1%). 동일 스트림 순서
        // 보장이라 캐스팅을 뒤로 미뤄도 총량 불변 — 이득 경로 없음.
        if !self.moe.moe_resident {
            self.moe_stage_t(il, t)?;
        }
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
                    .d2h_async(cb.as_mut_ptr(), self.moe.moe_gcnt, n_exp * 4)?;
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
            // [⑥ 2026-10-18] A = f16(RNE) 단일 미러 — [A2b] (hi|lo) 2-pass
            // 제거. ncu: grp2 스톨 지배 = wait(고정지연 의존) 33.6%·
            // math_pipe 9.6% — mma 2패스·A 분해 ALU를 절반화(moepp.cu).
            // 수치 클래스 = f16-A(과거 en PPL +0.17% 밴드) — eval 게이트 판정.
            let x16 = self.cast_f16_dev(xn, t * h)?;
            // [§1.2 2026-10-11] 게이트·업 2N 융합 — 발사 1회(A 스테이징·배리어 상각).
            self.gemm_g32_mma_grp2_launch(
                base,
                x16,
                h,
                tk,
                self.moe.exp.gate,
                self.moe.exp.up,
                n_ff,
                h,
            )?;
        } else {
            self.gemv_experts_launch(base, ns, xn, h, tk, self.moe.exp.gate, n_ff, h)?;
            self.gemv_experts_launch(base + 1, ns, xn, h, tk, self.moe.exp.up, n_ff, h)?;
        }
        // act는 f32 기록(미러 없음) — down A f16 미러는 cast_f16이 만든다.
        // ([⑥ 2026-10-18] 종전 [A2b] hi|lo packed·ew_dev_mirror는 삭제 경로.)
        self.ew_dev(
            self.moe.exp.gate,
            self.moe.exp.up,
            self.moe.exp.act,
            ns * n_ff,
        )?;
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            let mut vb = vec![0u8; ns * n_ff * 4];
            self.cc
                .d2h_async(vb.as_mut_ptr(), self.moe.exp.act, ns * n_ff * 4)?;
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
            // down N64 변형 — A는 f16 단일 미러([⑥] hi|lo 2-pass 제거).
            let ap = self.cast_f16_dev(self.moe.exp.act, ns * n_ff)?;
            self.gemm_g32_mma_grp_d64_launch(base + 2, ap, n_ff, 1, self.moe.exp.dn, h, n_ff)?;
        } else {
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
        }
        self.moe_accum_dev(
            self.moe.moe_wt,
            self.moe.exp.dn,
            self.moe.dmo.ptr,
            tk,
            ns,
            h,
        )?;
        if fused {
            self.moe_shared_tail_t(il, xn, t, forked)?;
        } else {
            self.moe_shared_t(il, xn, t)?;
        }
        Ok(self.moe.dmo.ptr)
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
        if aux.is_null() || self.moe.dsg == 0 || self.moe.ev_fork.len() < (il + 1) * 2 {
            return Ok(false);
        }
        let sf = self.moe.shared_ffn;
        let [_s0, _s1, _s1b, s2, s3] = self.dchain;
        let (e1, e2) = (self.moe.ev_fork[il * 2], self.moe.ev_fork[il * 2 + 1]);
        // 포크: 게이트/업(multi_gate_t) 완료 → 보조 스트림에 ew/down.
        // (sgate는 [B3-2b] multi_gate_t 4세그먼트가 dsg에 선기록.)
        self.cc.event_record_on(e1, self.cc.stream)?;
        self.cc.stream_wait(aux, e1)?;
        self.ew_dev_on(aux, self.moe.dsh[0], self.moe.dsh[1], s2, t * sf)?;
        self.plain_gemv_t_launch_on(aux, &format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.cc.event_record_on(e2, aux)?;
        Ok(true)
    }

    /// shared 전문가 배치(t≤8) — 플레인 GEMM ×3 + 토큰별 게이트 가산.
    pub(in crate::rawcuda::w4a16_dec) fn moe_shared_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        if self.moe.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.moe.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0, t)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1, t)?;
        self.ew_dev(s0, s1, s2, t * sf)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.plain_gemm_launch(
            &format!("blk.{il}.moe_shared_sgate.weight"),
            xn,
            self.moe.drt.ptr,
            t,
        )?;
        self.shared_add_dev(self.moe.drt.ptr, s3, self.moe.dmo.ptr, h, t)
    }
}
