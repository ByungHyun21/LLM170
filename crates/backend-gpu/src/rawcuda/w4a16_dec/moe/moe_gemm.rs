use super::super::*;

impl W4a16Dec {
    /// [P11] 그룹 mma GEMM(g32) 발사 — grid (전문가 × n타일), M=전문가 슬롯 수.
    /// [B2 2026-10-12] down 전용 N64 발사 — grp의 배리어 상각판(그리드 y =
    /// ceil(n/64)). 값은 grp와 동일 계열(골든 판정) — 커널 주석 참조.
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
        let f = self.cc.function("w4a16_gemm_g32_mma_grp_d64")?;
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
            self.moe.n_experts as u32,
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
        let f = self.cc.function("w4a16_gemm_g32_mma_grp2")?;
        let (mut p_t, mut p_b) = (self.moe.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) =
            (self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff);
        let (mut p_x, mut p_xs, mut p_sp) = (x_dev, xstride as i32, sp as i32);
        let (mut p_o, mut p_o2, mut p_n, mut p_k) = (out_gate, out_up, n as i32, k as i32);
        self.cc.launch(
            f,
            self.moe.n_experts as u32,
            n.div_ceil(GEMM_GRP_N) as u32,
            256,
            &mut crate::rawcuda::args::l12(
                &mut p_t, &mut p_b, &mut p_g, &mut p_c, &mut p_of, &mut p_x, &mut p_xs, &mut p_sp,
                &mut p_o, &mut p_o2, &mut p_n, &mut p_k,
            ),
        )
    }
}
