use super::super::*;

impl W4a16Dec {
    // ── 플레인 bf16(MoE 모델 — 35B) + MoE FFN ──

    /// bf16 GEMV — 행=블록(w4a16_gemv_bf16), x는 **원시 f32**(h2f 왕복 없음).
    /// 플레인 경로 판정은 토큰 수준(골든) — split 경로의 레인/환원 구조 미러.
    pub(in crate::rawcuda::w4a16_dec) fn plain_gemv_launch(
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
    pub(in crate::rawcuda::w4a16_dec) fn plain_gemm_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        // [T2] bf16 mma GEMM(단일 경로). t≥16에서만
        // (타일 M32 — 부분 타일은 가드로 동작하나 이득이 작음). 검증: 35B 600
        // 토큰 동일, 골든 유지, 허용오차 ~1e-3(bf16 활성 반올림).
        if t >= 16 {
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
                self.dabf = self.realloc_buf(self.dabf, need, need, "dabf_cap")?;
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
                "w4a16_gemm_bf16_mma2_16"
            } else {
                "w4a16_gemm_bf16_mma2"
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

    /// [A9] 배치 t행 GEMV(플레인 bf16) — 원시 f32 x, 행별 GEMV 비트동일.
    pub(in crate::rawcuda::w4a16_dec) fn plain_gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.plain_gemv_t_launch_on(self.cc.stream, name, x_dev, y_dev, t)
    }

    /// [B3] t행 플레인 GEMV — 보조 스트림 발사 변형(shared 꼬리 포크).
    pub(in crate::rawcuda::w4a16_dec) fn plain_gemv_t_launch_on(
        &mut self,
        s: ffi::CUstream,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        let f = self.cc.function("w4a16_gemv_bf16_t")?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_dev);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        self.cc.launch_on(
            s,
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
    pub(in crate::rawcuda::w4a16_dec) fn plain_stage_x32(
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
    pub(in crate::rawcuda::w4a16_dec) fn plain_gemv_dev(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let (_, n, _) = self.plain_spec(name)?;
        let dy = self.ensure_dy(n)?;
        self.plain_gemv_launch(name, x_dev, dy)?;
        Ok(dy)
    }
}
