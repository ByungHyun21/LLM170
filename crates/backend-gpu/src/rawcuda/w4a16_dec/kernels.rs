use super::*;

impl W4a16Dec {
    pub(super) fn norm_resid_at(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
        xn32: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        if self.dnw == 0 {
            return Err("norm: 노름 가중 미등록".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("norm: w={w} >= rows={}", self.norm_w_rows));
        }
        // [marlin-A] xn16 미러 — xn32와 같은 조건으로 함께 기록.
        let xn16 = if xn32 != 0 {
            let need = t_len * self.hidden;
            if need > self.dx16.cap {
                self.graph_invalidate();
                self.cc.sync()?;
                if self.dx16.ptr != 0 {
                    self.cc.free(self.dx16.ptr)?;
                }
                self.dx16.ptr = 0;
                self.dx16.cap = 0;
                self.dx16.ptr = self.cc.alloc(need * 2)?;
                self.dx16.cap = need;
            }
            self.dx16.ptr
        } else {
            0
        };
        let f = self.cc.function("norm_resid")?;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32;
        let mut hd = self.hidden as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
            (x_dev, self.dnw, ab_dev, self.norm.dxn, xn32, xn16);
        self.cc.launch(
            f,
            t_len as u32,
            1,
            1024,
            &mut crate::rawcuda::args::l9(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut tl, &mut wo, &mut hd,
            ),
        )?;
        Ok(self.norm.dxn)
    }

    /// 잔차 x_dev에 ab(호스트) 가산 + 노름 xn 판독(스테이징 1행).
    pub(super) fn norm_resid_staged(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.hidden == 0 || ab.len() != self.hidden || x_dev == 0 {
            return Err("norm: 순차 잔차/분기 폭 계약 위반".into());
        }
        self.ensure_norm_bufs(1)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let abb = unsafe { std::slice::from_raw_parts(ab.as_ptr() as *const u8, ab.len() * 4) };
        self.cc.h2d(self.norm.dab, abb)?;
        let xn = self.norm_resid_at(w, x_dev, self.norm.dab, 1, 0)?;
        let mut bytes = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut bytes, xn)?;
        self.cc.sync()?;
        Ok(
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.hidden) }
                .to_vec(),
        )
    }

    /// [marlin-A5] f32 → dx16(f16) 캐스트만 — MoE 그룹 GEMM의 A(xn/act)용.
    /// 값은 cast_x32와 동일 반올림(비트 동일) — GEMM A 대역 절반.
    pub(super) fn cast_x16(&mut self, x_dev: CUdeviceptr, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx16.cap {
            self.cc.capture_guard("dx16_cap"); // [R12]
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dx16.ptr != 0 {
                self.cc.free(self.dx16.ptr)?;
            }
            self.dx16.ptr = 0;
            self.dx16.cap = 0;
            self.dx16.ptr = self.cc.alloc(n * 2)?;
            self.dx16.cap = n;
        }
        let f = self.cc.function("w4a16_cast_x32")?;
        let mut nn = n as i32;
        let (mut c0, mut c1, mut c2) = (x_dev, 0u64, self.dx16.ptr);
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l4(&mut c0, &mut c1, &mut c2, &mut nn),
        )?;
        Ok(self.dx16.ptr)
    }

    /// t≥2 GEMM 발사 — x f16 [t][k] → out [t][n] 직접 쓰기.
    pub(super) fn gemm_launch(
        &mut self,
        name: &str,
        xh_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        // [T1] split mma 경로(기본 ON — `LLM170_TC=0`으로 해제, t≥16).
        // 검증: 27B 4k·35B 600 토큰이 원본과 동일, 골든 유지, 허용오차 ~6e-4.
        // A=xh(f16 — split 경로가 이미 f2h 캐스트 제공), 계약 완화 승인 후.
        if t >= 16 && llm170_diag::flag::ne0("LLM170_TC") {
            // [marlin-A 2026-10-10] A는 f16 미러(dx16) — 불변식: split 경로
            // GEMM A는 직전 cast_x32/norm_resid 산출(dx32)이고 dx16이 같은
            // [t][k] 값을 갖는다(두 경로가 항상 함께 기록). 종전 f32 A는
            // ncu상 L2 트래픽의 80%였다.
            if self.dx16.cap < t * k {
                return Err(format!("gemm: dx16 부족 {} < {}", self.dx16.cap, t * k));
            }
            let f = self.cc.function("w4a16_gemm_g128_mma")?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx16.ptr, y_out);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            return self.cc.launch(
                f,
                t.div_ceil(GEMM_MMA_M) as u32,
                n.div_ceil(GEMM_MMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
                ),
            );
        }
        let f = self.cc.function("w4a16_gemm_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, xh_dev, y_out);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        // 8행/블록 커널(512스레드 = 8그룹×64레인) — grid = ceil(n/8).
        self.cc.launch(
            f,
            n.div_ceil(8) as u32,
            1,
            512,
            &mut crate::rawcuda::args::l7(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    /// GEMV 발사(공용) — x32 입력 → y_out 직접 쓰기(dy·d2d 경유 제거).
    /// [기각 2026-10-10] t=1을 TR 커널(8행/블록)로 전환하는 안 — A/B 실측
    /// 27B 21.1→25.3ms·35B 9.7→10.1ms/토큰(역효과). t=1은 x가 단일 행이라
    /// L1 재사용 이득이 이미 캐시로 상쇄되고 smem(red/sc ×TR) 점유만 악화.
    /// 1행/블록 유지(TR 커널은 배치 t≥2 전용).
    /// [실측 기각 2026-10-10 — 재시도 금지] ① re-layout(오프라인 재배치·프리페치
    /// 링·오토튜너): 대형 형상 ncu DRAM 81~89% 포화(ffn_up 89%·attn_qkv 818GB/s)
    /// — 여지 소. 소형 형상은 그리드 미포화(점유 15%)가 원인. ② q8_1+dp4a 정수
    /// 경로: ①과 동일 사유. ③ half2 누산(MoE 전문가 GEMV): DRAM 46%·ALU 29%·
    /// No Eligible 63%로 지연 바운드 — ALU 절감 무효 + f16 누산 골든 리스크.
    pub(super) fn gemv_launch(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let f = self.cc.function("w4a16_gemv_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x32_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )
    }

    /// GEMV(x32 입력) → self.dy.ptr — 반환 포인터는 다음 gemv가 덮는다(스트림 순서).
    pub(super) fn gemv_dev_x32(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        let dy = self.ensure_dy(n)?;
        self.gemv_launch(name, x32_dev, dy)?;
        Ok(dy)
    }

    /// GEMV(x32 입력) → dst 직접 쓰기 + 폭 검사(스테이징 공유 폭 계약).
    pub(super) fn gemv_stage_x32(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        if n > w {
            return Err(format!("gemv_stage({name}): n={n} > 스테이징 {w}"));
        }
        self.gemv_launch(name, x32_dev, dst)
    }

    /// [A-4] head GEMV([n][k] 원본) — TR 커널 t=1(8행/블록).
    /// 행별 누산 시퀀스가 1행판과 동일(i=l,l+64,… + f64 트리) → 비트 동일.
    /// 이득: 블록 내 8행 그룹이 x를 L1 공유 → x L2 재판독 ÷8
    /// (35B head 실측 1.05ms — x L2 1.2GB가 가중치 622MB와 경합).
    pub(super) fn head_gemv_launch(&self, x: CUdeviceptr, out: CUdeviceptr) -> Result<(), String> {
        self.head_gemv_t_launch(x, out, 1)
    }

    /// [A-4] head GEMV 배치(t≤8) — TR 커널(행=블록 공유).
    pub(super) fn head_gemv_t_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_bf16_t")?;
        let (mut p_w, mut p_x, mut p_o) = (self.head_w, x, out);
        let (mut p_n, mut p_k, mut p_t) = (self.head_n as i32, self.head_k as i32, t as i32);
        self.cc.launch(
            f,
            self.head_n.div_ceil(GEMV_TR) as u32,
            1,
            (64 * GEMV_TR) as u32,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    /// [R10 2026-10-10] 체인 선형 경로 — 3체인(단독 t=1 / 프리필 t≥2 /
    /// 배치 t≤8)의 차이는 이 열거 하나다. 방출 커널·인자 순서는 종전과 동일
    /// (그래프 캡처 불변식 — 골든·스모크로 검증).
    pub(super) fn lin_forward(
        &mut self,
        plain: bool,
        path: LinPath,
        name: &str,
        x: CUdeviceptr,
        dst: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        match (plain, path) {
            (false, LinPath::Gemv1 { w }) => self.gemv_stage_x32(name, x, dst, w),
            (false, LinPath::Gemm) => self.gemm_launch(name, x, dst, t),
            (false, LinPath::GemvT) => self.gemv_t_launch(name, x, dst, t),
            (true, LinPath::Gemv1 { w }) => self.plain_stage_x32(name, x, dst, w),
            (true, LinPath::Gemm) => self.plain_gemm_launch(name, x, dst, t),
            (true, LinPath::GemvT) => self.plain_gemv_t_launch(name, x, dst, t),
        }
    }

    // ── 플레인 bf16(MoE 모델 — 35B) + MoE FFN ──

    /// bf16 GEMV — 행=블록(w4a16_gemv_bf16), x는 **원시 f32**(h2f 왕복 없음).
    /// 플레인 경로 판정은 토큰 수준(골든) — split 경로의 레인/환원 구조 미러.
    pub(super) fn plain_gemv_launch(
        &self,
        name: &str,
        x_dev: CUdeviceptr,
        out_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        let f = self.cc.function("w4a16_gemv_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, out_dev);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l5(&mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k),
        )
    }

    /// 플레인 bf16 GEMM(t≤8) — x는 원시 f32 [t][k](h2f 왕복 없음),
    /// 가중치 1회 판독 × t토큰 재사용(프리필 청크의 dense 경로).
    pub(super) fn plain_gemm_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        // [T2] bf16 mma GEMM(기본 ON — `LLM170_TC=0`으로 해제). t≥16에서만
        // (타일 M32 — 부분 타일은 가드로 동작하나 이득이 작음). 검증: 35B 600
        // 토큰 동일, 골든 유지, 허용오차 ~1e-3(bf16 활성 반올림).
        if t >= 16 && llm170_diag::flag::ne0("LLM170_TC") {
            {
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[TC] mma GEMM 경로 진입: {name} n={n} k={k} t={t}");
                }
            }
            // [FLA-10 2026-10-10] A 미러 — f32 x를 bf16으로 선변환(L2 A
            // 재판독 절반 — bf16_mma L2 68% 바운드 실측). cast_bf16과 커널
            // 내 f2bf16이 비트동일 → 결과 불변.
            let need = t * k * 2;
            if need > self.dabf.cap {
                self.graph_invalidate();
                self.cc.sync()?;
                if self.dabf.ptr != 0 {
                    self.cc.free(self.dabf.ptr)?;
                }
                self.dabf.ptr = 0;
                self.dabf.cap = 0;
                self.dabf.ptr = self.cc.alloc(need)?;
                self.dabf.cap = need;
            }
            {
                let fc = self.cc.function("w4a16_cast_bf16")?;
                let (mut c_in, mut c_out) = (x_dev, self.dabf.ptr);
                let mut c_n = (t * k) as i32;
                self.cc.launch(
                    fc,
                    (t * k).div_ceil(1024) as u32,
                    1,
                    256,
                    &mut crate::rawcuda::args::l3(&mut c_in, &mut c_out, &mut c_n),
                )?;
            }
            // [M16 2026-10-11] 그리드 기아 형상이면 M16 변형(블록 2배) —
            // 값은 MM32와 비트동일(같은 k 순서·mma). 판정: M32 타일 수 < SM 수.
            let tiles32 = t.div_ceil(GEMM_BMMA_M) as u64 * n.div_ceil(GEMM_BMMA_N) as u64;
            let m16 = tiles32 < 2 * self.cc.sm_count().max(1) as u64;
            let f = self.cc.function(if m16 {
                "w4a16_gemm_bf16_mma16"
            } else {
                "w4a16_gemm_bf16_mma"
            })?;
            let (mut p_w, mut p_x, mut p_o) = (w, self.dabf.ptr, y_out);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            let mut p_b = 1i32;
            return self.cc.launch(
                f,
                t.div_ceil(if m16 { 16 } else { GEMM_BMMA_M }) as u32,
                n.div_ceil(GEMM_BMMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t, &mut p_b,
                ),
            );
        }
        // [B2] v3(8행/블록 + 행별 smem + k청크) 상시 — t=1은 GEMV 경로라
        // 여기 오지 않으므로(t≥2) 실질 임계치 = t≥2. v1(t≤8 전용)은 x 재판독이
        // 있어 v3 대비 열위, 산술 순서는 동일 명시(plain_gemm_selfcheck 게이트).
        let v3 = t > 1;
        let f = self.cc.function(if v3 {
            "w4a16_gemm_bf16_t"
        } else {
            "w4a16_gemm_bf16"
        })?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_out);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        if v3 {
            self.cc.launch(
                f,
                n.div_ceil(8) as u32,
                1,
                512,
                &mut crate::rawcuda::args::l6(
                    &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
                ),
            )
        } else {
            self.cc.launch(
                f,
                n as u32,
                1,
                64,
                &mut crate::rawcuda::args::l6(
                    &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
                ),
            )
        }
    }

    /// [A9 2026-10-10] 배치 t행 GEMV(split g128) — 가중 판독 1회를 t행 공유,
    /// 행별 산술은 GEMV와 비트동일. x는 cast_x32 산출 f32.
    /// [T3 실측 2026-10-10] T1 mma t<16 강제 배치 시험은 열세(27B 13.2 vs
    /// 본 커널 21.1 tok/s/사용자 — mma 스테이징 실효 판독 < 380GB/s; gptq4.cu T1).
    pub(super) fn gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let f = self.cc.function("w4a16_gemv_g128_t")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_dev);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        self.cc.launch(
            f,
            n.div_ceil(GEMV_TR) as u32,
            1,
            (32 * GEMV_TR) as u32, // [A-float4] 32레인/행 — 호스트 미러
            &mut crate::rawcuda::args::l7(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    /// [S2 2026-10-11] 다중세그먼트 t=1 GEMV 런치 — 테이블 u64 오프셋·행 합.
    /// split = (q,s,out,n,k)(w4a16_gemv_g128_multi) / 플레인 = (w,out,n,k)
    /// (w4a16_gemv_multi). 값은 세그먼트 단독 런치와 비트동일(행 독립).
    pub(super) fn gemv_multi_launch(
        &self,
        plain: bool,
        table: CUdeviceptr,
        x_dev: CUdeviceptr,
        total: usize,
        count: usize,
    ) -> Result<(), String> {
        let name = if plain {
            "w4a16_gemv_multi"
        } else {
            "w4a16_gemv_g128_multi"
        };
        let f = self.cc.function(name)?;
        let (mut p_t, mut p_x) = (table, x_dev);
        let mut p_c = count as i32;
        self.cc.launch(
            f,
            total as u32,
            1,
            64,
            &mut crate::rawcuda::args::l3(&mut p_t, &mut p_x, &mut p_c),
        )
    }

    /// [S2] 층·그룹의 다중세그먼트 런치(테이블 메타 조회 — group 0=branch,
    /// 1=gate/up). 테이블은 ensure_chain_bufs가 1회 작성.
    pub(super) fn gemv_multi_dev(
        &mut self,
        plain: bool,
        il: usize,
        group: usize,
        x_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let m = *self
            .dmulti_meta
            .get(il)
            .ok_or_else(|| format!("gemv_multi: 메타 부족 il={il}"))?;
        let (off, cnt, total) = if group == 0 {
            (m[0], m[1], m[2])
        } else {
            (m[3], m[4], m[5])
        };
        if cnt == 0 {
            return Err(format!("gemv_multi: 세그먼트 없음 il={il} g={group}"));
        }
        self.gemv_multi_launch(
            plain,
            self.dmulti + off as u64 * 8,
            x_dev,
            total as usize,
            cnt as usize,
        )
    }

    /// [S2 2026-10-11] 다중세그먼트 t행 GEMV 런치(t≤8) — grid =
    /// Σceil(n_s/GEMV_TR)(테이블 메타 *_tgrid). 값은 단독 GemvT와 비트동일.
    pub(super) fn gemv_multi_t_launch(
        &self,
        plain: bool,
        table: CUdeviceptr,
        x_dev: CUdeviceptr,
        total: usize,
        count: usize,
        t: usize,
    ) -> Result<(), String> {
        let name = if plain {
            "w4a16_gemv_bf16_multi_t"
        } else {
            "w4a16_gemv_g128_multi_t"
        };
        let f = self.cc.function(name)?;
        let (mut p_t, mut p_x) = (table, x_dev);
        let (mut p_c, mut p_tt) = (count as i32, t as i32);
        // [A-float4] split은 32레인/행 — 플레인(bf16)은 종전 64레인.
        let blk = if plain { 64 * GEMV_TR } else { 32 * GEMV_TR } as u32;
        self.cc.launch(
            f,
            total as u32,
            1,
            blk,
            &mut crate::rawcuda::args::l4(&mut p_t, &mut p_x, &mut p_c, &mut p_tt),
        )
    }

    /// [S2] 층·그룹의 다중세그먼트 t행 런치(메타 idx 6/7 = *_tgrid).
    pub(super) fn gemv_multi_t_dev(
        &mut self,
        plain: bool,
        il: usize,
        group: usize,
        x_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let m = *self
            .dmulti_meta
            .get(il)
            .ok_or_else(|| format!("gemv_multi_t: 메타 부족 il={il}"))?;
        let (off, cnt, tgrid) = if group == 0 {
            (m[0], m[1], m[6])
        } else {
            (m[3], m[4], m[7])
        };
        if cnt == 0 {
            return Err(format!("gemv_multi_t: 세그먼트 없음 il={il} g={group}"));
        }
        self.gemv_multi_t_launch(
            plain,
            self.dmulti + off as u64 * 8,
            x_dev,
            tgrid as usize,
            cnt as usize,
            t,
        )
    }

    /// [A9] 배치 t행 GEMV(플레인 bf16) — 원시 f32 x, 행별 GEMV 비트동일.
    pub(super) fn plain_gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        let f = self.cc.function("w4a16_gemv_bf16_t")?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_dev);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        self.cc.launch(
            f,
            n.div_ceil(GEMV_TR) as u32,
            1,
            (64 * GEMV_TR) as u32,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    /// 플레인 GEMV → 스테이징 dst 직접 쓰기 + 폭 검사.
    pub(super) fn plain_stage_x32(
        &self,
        name: &str,
        x_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let (_, n, _) = self.plain_spec(name)?;
        if n > w {
            return Err(format!("plain_stage({name}): n={n} > 스테이징 {w}"));
        }
        self.plain_gemv_launch(name, x_dev, dst)
    }

    /// 플레인 GEMV → self.dy.ptr.
    pub(super) fn plain_gemv_dev(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let (_, n, _) = self.plain_spec(name)?;
        let dy = self.ensure_dy(n)?;
        self.plain_gemv_launch(name, x_dev, dy)?;
        Ok(dy)
    }

    /// 분리 GEMV 발사(이름 무경유 — MoE 전문가 스테이징).
    pub(super) fn gemv_launch_raw(
        &self,
        dq: CUdeviceptr,
        ds: CUdeviceptr,
        n: usize,
        k: usize,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let sym = crate::rawcuda::gptq4::kernel_sym(false, self.moe_group, self.moe_scale_bf16)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )
    }

    /// 가중 누적 — y += w·x.
    pub(super) fn axpy_dev(
        &mut self,
        w: f32,
        x: CUdeviceptr,
        y: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_axpy")?;
        let mut ww = w;
        let (mut p_x, mut p_y) = (x, y);
        let mut nn = n as i32;
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l4(&mut ww, &mut p_x, &mut p_y, &mut nn),
        )
    }

    /// shared 게이트 가산 — y[t][i] += sigmoid(sg[t])·x[t][i] (grid = t).
    /// [§4.5] 전문가 누적 + shared 가산 융합 런치 — accum_dev와 동일 그리드
    /// ((n/256, 행), 256스레드) + sg/sx 인자.
    pub(super) fn moe_accum_sg_dev(
        &mut self,
        w: CUdeviceptr,
        d: CUdeviceptr,
        sg: CUdeviceptr,
        sx: CUdeviceptr,
        y: CUdeviceptr,
        nslots: usize,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_accum_sg")?;
        let (mut p_w, mut p_d, mut p_sg, mut p_sx, mut p_y) = (w, d, sg, sx, y);
        let (mut p_sp, mut p_ns, mut p_n) = (0i32, nslots as i32, n as i32);
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l8(
                &mut p_w, &mut p_d, &mut p_sg, &mut p_sx, &mut p_y, &mut p_sp, &mut p_ns, &mut p_n,
            ),
        )
    }

    pub(super) fn shared_add_dev(
        &mut self,
        sg: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        n: usize,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_shared_add")?;
        let (mut p_sg, mut p_x, mut p_y) = (sg, x, y);
        let mut nn = n as i32;
        self.cc.launch(
            f,
            t.max(1) as u32,
            n.div_ceil(256) as u32,
            256,
            &mut crate::rawcuda::args::l4(&mut p_sg, &mut p_x, &mut p_y, &mut nn),
        )
    }

    /// 노름 1회(디바이스 x·ab) — xn은 self.norm.dxn(다음 노름이 덮는다).
    pub(super) fn norm_resid_dev(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
        xn32: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        self.ensure_norm_bufs(t_len)?;
        self.norm_resid_at(w, x_dev, ab_dev, t_len, xn32)
    }

    // ── ew(silu·mul) ──

    /// ew(silu·mul) 디바이스 발사 — g·u → y(원시 f32 — MoE 소비 경로).
    pub(super) fn ew_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y_dev, 0u64, 0u64);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )
    }

    /// [4.1 2026-10-11] ew + x32/미러 동시 기록 — split 체인의 cast_x32 노드
    /// 제거(값 비트동일: h2f(f2h(act))·f2h(act) — cast 커널과 동일 산식).
    /// dx32/dx16을 n까지 보장하고 커널에 넘긴다.
    pub(super) fn ew_dev_fused(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        self.ensure_dx32(n)?;
        self.ensure_dx16(n)?;
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) =
            (g_dev, u_dev, y_dev, self.dx32.ptr, self.dx16.ptr);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )
    }

    /// [4.1 2026-10-11] ew + f16 미러만 — MoE 그룹 GEMM A(cast_x16 대체).
    pub(super) fn ew_dev_mirror(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        self.ensure_dx16(n)?;
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y_dev, 0u64, self.dx16.ptr);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )
    }
}
