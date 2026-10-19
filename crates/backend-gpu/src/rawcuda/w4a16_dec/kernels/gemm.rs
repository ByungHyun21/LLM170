use super::super::*;

impl W4a16Dec {
    /// t≥2 GEMM 발사 — x f16 [t][k] → out [t][n] 직접 쓰기.
    pub(in crate::rawcuda::w4a16_dec) fn gemm_launch(
        &mut self,
        name: &str,
        xh_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        // [gemma4 P2-3 2026-10-14] 그룹별 심볼 — mma(t≥16)는 g128 전용,
        // 그 외는 FFMA(kernel_sym — g32 대응). g32의 t≥16은 FFMA 폴백
        // (mma g32 커널은 후속 트랙 — 정확성 우선).
        // [2.N 2026-10-15 실측 기각 — 재론 금지(조건부)] g32 dense mma 포트
        // (`w4a16_gemm_g32_mma` = g128_mma의 SHIFT=5·bf16 스케일 + cast_x16 A
        // 미러 — 커널 REG 76·STACK 0) 배선 시도: gemma4 청크16에서 동일 세션
        // 교차 A/B **pp16 −6.7% · pp256 −8.9%**(디코드 불변) — M64 타일이
        // t=16에서 16행만 유효(1 m타일)라 스테이징·디큐트 오버헤드를 상쇄하지
        // 못한다(t<16 역행 기록과 같은 클래스). 후보 삭제(커널·배선 제거,
        // 기록만 — 단일 경로 규율). 재개 조건 = 청크 t≥64 정책 도입 시.
        let (grp, sbf) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        // [T1] split mma 경로(단일 경로, t≥16). 검증: 27B 4k·35B 600
        // 토큰이 원본과 동일, 골든 유지, 허용오차 ~6e-4.
        // A=xh(f16 — split 경로가 이미 f2h 캐스트 제공), 계약 완화 승인 후.
        if t >= 16 && grp == 128 && !sbf {
            // [marlin-A 2026-10-10] A는 f16 미러(dx16) — 불변식: split 경로
            // GEMM A는 직전 cast_x32/norm_resid 산출(dx32)이고 dx16이 같은
            // [t][k] 값을 갖는다(두 경로가 항상 함께 기록). 종전 f32 A는
            // ncu상 L2 트래픽의 80%였다.
            if self.dx16.cap < t * k {
                return Err(format!("gemm: dx16 부족 {} < {}", self.dx16.cap, t * k));
            }
            // [2026-10-18] vLLM 마린(벤더) 시도 — 사본 가용 시 단일 경로.
            let a16 = self.dx16.ptr;
            if self.marlin_gemm_try(name, t, n, k, a16, y_out, 128)? {
                return Ok(());
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
}
