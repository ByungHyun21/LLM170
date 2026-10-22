use super::super::*;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn gemm_g32_mma_grp_d64_launch(
        &self,
        base: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let sym = if self.moe.moe_group == 32 && !self.moe.moe_scale_bf16 {
            "w4a16_gemm_g32_mma_grp_d64_sf16"
        } else {
            "w4a16_gemm_g32_mma_grp_d64_f16"
        };
        let f = self.cc.function(sym)?;
        let (mut p_t, mut p_b) = (self.moe.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) =
            (self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff);
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
            (self.moe.n_experts * MOE_PP_MZ) as u32,
            n.div_ceil(64) as u32,
            256,
            &mut crate::rawcuda::args::l11(
                &mut p_t, &mut p_b, &mut p_g, &mut p_c, &mut p_of, &mut p_x, &mut p_xs, &mut p_sp,
                &mut p_o, &mut p_n, &mut p_k,
            ),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn gemm_g32_mma_grp2_launch(
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
        let sym = if self.moe.moe_group == 32 && !self.moe.moe_scale_bf16 {
            "w4a16_gemm_g32_mma_grp2_sf16"
        } else {
            "w4a16_gemm_g32_mma_grp2_f16"
        };
        let f = self.cc.function(sym)?;
        let (mut p_t, mut p_b) = (self.moe.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) =
            (self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff);
        let (mut p_x, mut p_xs, mut p_sp) = (x_dev, xstride as i32, sp as i32);
        let (mut p_o, mut p_o2, mut p_n, mut p_k) = (out_gate, out_up, n as i32, k as i32);
        self.cc.launch(
            f,
            (self.moe.n_experts * MOE_PP_MZ) as u32,
            n.div_ceil(GEMM_GRP_N) as u32,
            256,
            &mut crate::rawcuda::args::l12(
                &mut p_t, &mut p_b, &mut p_g, &mut p_c, &mut p_of, &mut p_x, &mut p_xs, &mut p_sp,
                &mut p_o, &mut p_o2, &mut p_n, &mut p_k,
            ),
        )
    }
}
