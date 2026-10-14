use super::*;

mod gemm;
mod gemv;
mod plain;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn norm_resid_at(
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
                self.dx16 = self.realloc_buf(self.dx16, need * 2, need, "dx16_cap")?;
            }
            self.dx16.ptr
        } else {
            0
        };
        let f = self.cc.function(if self.hidden.is_multiple_of(1024) {
            "norm_resid"
        } else {
            // [gemma4 2026-10-14] hidden 2816(1024 배수 아님) — 행 경계 가드 변형.
            "norm_resid_g"
        })?;
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
    pub(in crate::rawcuda::w4a16_dec) fn norm_resid_staged(
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
    pub(in crate::rawcuda::w4a16_dec) fn cast_x16(
        &mut self,
        x_dev: CUdeviceptr,
        n: usize,
    ) -> Result<CUdeviceptr, String> {
        if n > self.dx16.cap {
            self.dx16 = self.realloc_buf(self.dx16, n * 2, n, "dx16_cap")?;
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

    /// [R10 2026-10-10] 체인 선형 경로 — 3체인(단독 t=1 / 프리필 t≥2 /
    /// 배치 t≤8)의 차이는 이 열거 하나다. 방출 커널·인자 순서는 종전과 동일
    /// (그래프 캡처 불변식 — 골든·스모크로 검증).
    pub(in crate::rawcuda::w4a16_dec) fn lin_forward(
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

    /// 가중 누적 — y += w·x.
    pub(in crate::rawcuda::w4a16_dec) fn axpy_dev(
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
    pub(in crate::rawcuda::w4a16_dec) fn moe_accum_sg_dev(
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

    pub(in crate::rawcuda::w4a16_dec) fn shared_add_dev(
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

    /// [gemma4 P3 2026-10-14] 잔차 없는 순수 행별 RMS — `rms_rows` 커널
    /// (호스트 `ops::rms_norm` **비트 미러**: 32세그먼트 f32 순차 누산 →
    /// f64 결합 → 정밀 sqrt). gemma4 디바이스 상주 청크의 노름 대체 —
    /// `norm_resid*`(잔차 융합·1024트리)와는 다른 산술이라 호스트 경로
    /// 교체에 쓸 수 없다. x·y 비중첩(제자리 불가 — 부모 행을 읽는다).
    pub(in crate::rawcuda::w4a16_dec) fn rms_rows_dev(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t_len: usize,
        eps: f32,
    ) -> Result<(), String> {
        if self.dnw == 0 {
            return Err("rms_rows: 노름 가중 미등록".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("rms_rows: w={w} >= rows={}", self.norm_w_rows));
        }
        if t_len == 0 || x_dev == 0 || y_dev == 0 {
            return Err("rms_rows: t=0/포인터 0 계약 위반".into());
        }
        let f = self.cc.function("rms_rows")?;
        // 동적 smem = [hidden]x + [32]세그먼트 + [1]스케일 (2816 = 11.4KB).
        let smem = ((self.hidden + 33) * 4) as u32;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32;
        let mut hd = self.hidden as i32;
        let mut ep = eps;
        let (mut a0, mut a1, mut a2) = (x_dev, self.dnw, y_dev);
        self.cc.launch_shared(
            f,
            t_len as u32,
            1,
            256,
            smem,
            &mut crate::rawcuda::args::l7(
                &mut a0, &mut a1, &mut a2, &mut tl, &mut wo, &mut hd, &mut ep,
            ),
        )
    }

    /// 노름 1회(디바이스 x·ab) — xn은 self.norm.dxn(다음 노름이 덮는다).
    pub(in crate::rawcuda::w4a16_dec) fn norm_resid_dev(
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
    pub(in crate::rawcuda::w4a16_dec) fn ew_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        self.ew_dev_on(self.cc.stream, g_dev, u_dev, y_dev, n)
    }

    /// [B3] ew — 보조 스트림 발사 변형(shared 꼬리 포크).
    pub(in crate::rawcuda::w4a16_dec) fn ew_dev_on(
        &mut self,
        s: ffi::CUstream,
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
        self.cc.launch_on(
            s,
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
    pub(in crate::rawcuda::w4a16_dec) fn ew_dev_fused(
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
    pub(in crate::rawcuda::w4a16_dec) fn ew_dev_mirror(
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
