use super::super::*;

impl W4a16Dec {

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

    pub(in crate::rawcuda::w4a16_dec) fn plain_gemm_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        if t >= 16 {
            {
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[TC] mma GEMM 경로 진입: {name} n={n} k={k} t={t}");
                }
            }
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
            if t >= 64 && n >= 2048 && n.is_multiple_of(128) && k.is_multiple_of(64) {
                let f = self.cc.function("w4a16_gemm_bf16_mma3")?;
                const B3_SMEM: u32 = 2 * (64 + 128) * (64 + 8) * 2;
                self.cc.set_dynamic_smem(f, B3_SMEM)?;
                let (mut p_w, mut p_x, mut p_o) = (w, self.dabf.ptr, y_out);
                let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
                return self.cc.launch_shared(
                    f,
                    t.div_ceil(64) as u32,
                    n.div_ceil(128) as u32,
                    256,
                    B3_SMEM,
                    &mut crate::rawcuda::args::l6(
                        &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
                    ),
                );
            }
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

    pub(in crate::rawcuda::w4a16_dec) fn plain_gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.plain_gemv_t_launch_on(self.cc.stream, name, x_dev, y_dev, t)
    }

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
