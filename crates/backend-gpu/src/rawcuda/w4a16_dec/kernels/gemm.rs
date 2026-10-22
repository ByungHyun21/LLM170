use super::super::*;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn gemm_launch(
        &mut self,
        name: &str,
        xh_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let (grp, sbf) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        if (t >= 16 || dq == 0) && grp == 128 && !sbf {
            if self.dx16.cap < t * k {
                return Err(format!("gemm: dx16 부족 {} < {}", self.dx16.cap, t * k));
            }
            let a16 = self.dx16.ptr;
            if self.marlin_gemm_try(name, t, n, k, a16, y_out, 128)? {
                return Ok(());
            }
            if dq == 0 {
                return Err(format!("gemm {name}: 마린 경로 실패 — 원본 q 폐기됨"));
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
        let f = self
            .cc
            .function(crate::rawcuda::gptq4::kernel_sym(true, grp, sbf)?)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, xh_dev, y_out);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
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
}
