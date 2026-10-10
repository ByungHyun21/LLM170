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

    // ── ew(silu·mul) ──

    /// 활성 f32 → x32(h2f 왕복) 캐스트 1회 — 같은 xn을 쓰는 GEMV들이 공유한다
    /// (q/k/v·gate/up: 종전 gemv마다 캐스트 = 런치 2배). 반환은 self.dx32.ptr.
    /// [A3 판정 2026-10-10] GEMV 로드에 h2f를 인라인해 이 노드를 없애는 안은
    /// 기각 — 디코드 GEMV +1op/가중치(+15%, ~+2.4ms/토큰) > 런치 절감
    /// (단일 소비자 2회/층 ~0.4ms).
    pub(super) fn cast_x32(&mut self, x_dev: CUdeviceptr, k: usize) -> Result<CUdeviceptr, String> {
        if k > self.dx32.cap {
            // [P10] 재할당 전 무효화+동기 — 프리필 cast_x32(t×k) 성장이
            // 비행 중 norm xn32 기록을 해제 버퍼로 보낸다(새니타이저 실측).
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = 0; // G1
            self.dx32.cap = 0;
            self.dx32.ptr = self.cc.alloc(k * 4)?;
            self.dx32.cap = k;
        }
        // [marlin-A] f16 미러 동시 기록 — mma GEMM A(대역 절반, 비트 동일).
        if k > self.dx16.cap {
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dx16.ptr != 0 {
                self.cc.free(self.dx16.ptr)?;
            }
            self.dx16.ptr = 0;
            self.dx16.cap = 0;
            self.dx16.ptr = self.cc.alloc(k * 2)?;
            self.dx16.cap = k;
        }
        let f = self.cc.function("w4a16_cast_x32")?;
        let mut nn = k as i32;
        let (mut c0, mut c1, mut c2) = (x_dev, self.dx32.ptr, self.dx16.ptr);
        self.cc.launch(
            f,
            k.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l4(&mut c0, &mut c1, &mut c2, &mut nn),
        )?;
        Ok(self.dx32.ptr)
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
        &self,
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
            let f = self.cc.function("w4a16_gemm_bf16_mma")?;
            let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_out);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            return self.cc.launch(
                f,
                t.div_ceil(GEMM_BMMA_M) as u32,
                n.div_ceil(GEMM_BMMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l6(
                    &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
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
            (64 * GEMV_TR) as u32,
            &mut crate::rawcuda::args::l7(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
            ),
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

    /// ew(silu·mul) 디바이스 발사 — g·u → y.
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
        let (mut a0, mut a1, mut a2) = (g_dev, u_dev, y_dev);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut nn),
        )
    }
}
