use super::super::*;

impl W4a16Dec {
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
    pub(in crate::rawcuda::w4a16_dec) fn gemv_launch(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        // [gemma4 P2-3 2026-10-14] 그룹별 심볼 — g128/f16(기본) vs g32/bf16.
        // 런치 구성은 동일(grid=n·block=64) — 게이트 `w4a16-gemv`와 같은 표.
        let (g, sb) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        if k > g * 256 {
            return Err(format!(
                "gemv {name}: k={k} > {} (t=1 smem sc 계약, g{g})",
                g * 256
            ));
        }
        let sym = crate::rawcuda::gptq4::kernel_sym(false, g, sb)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x32_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32, // [A2-float4 v2] 256폭·64레인 — 종전 grid/블록 유지
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )
    }

    /// GEMV(x32 입력) → self.dy.ptr — 반환 포인터는 다음 gemv가 덮는다(스트림 순서).
    pub(in crate::rawcuda::w4a16_dec) fn gemv_dev_x32(
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
    pub(in crate::rawcuda::w4a16_dec) fn gemv_stage_x32(
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
    pub(in crate::rawcuda::w4a16_dec) fn head_gemv_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        // [I8head] t=1은 전용 행=블록 커널(TR 판은 t=1에서 역행 실측 이력).
        self.head_gemv_i8_launch(x, out)
    }

    /// [A-4 / I8head] head GEMV 배치(t≤8) — TR 커널(가중 1판독 t행 공유).
    /// [I8head 2026-10-11] int8 판(행별 스케일) — bf16 대비 대역 절반.
    pub(in crate::rawcuda::w4a16_dec) fn head_gemv_t_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_i8_t")?;
        let (mut p_w, mut p_sc, mut p_x, mut p_o) = (self.head_w, self.head_sc, x, out);
        let (mut p_n, mut p_k, mut p_t) = (self.head_n as i32, self.head_k as i32, t as i32);
        self.cc.launch(
            f,
            self.head_n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l7(
                &mut p_w, &mut p_sc, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    /// [I8head] head GEMV t=1 — 행=블록 64레인 int8(디코드 주 경로).
    pub(in crate::rawcuda::w4a16_dec) fn head_gemv_i8_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_i8")?;
        let (mut p_w, mut p_sc, mut p_x, mut p_o) = (self.head_w, self.head_sc, x, out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        self.cc.launch(
            f,
            self.head_n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_sc, &mut p_x, &mut p_o, &mut p_n, &mut p_k,
            ),
        )
    }

    /// [A9 2026-10-10] 배치 t행 GEMV(split g128) — 가중 판독 1회를 t행 공유,
    /// 행별 산술은 GEMV와 비트동일. x는 cast_x32 산출 f32.
    /// [T3 실측 2026-10-10] T1 mma t<16 강제 배치 시험은 열세(27B 13.2 vs
    /// 본 커널 21.1 tok/s/사용자 — mma 스테이징 실효 판독 < 380GB/s; gptq4.cu T1).
    pub(in crate::rawcuda::w4a16_dec) fn gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        // [gemma4 P2-3 2026-10-14] t행 GEMV는 g128 전용 — 비기본 그룹(예: gemma4
        // g32)은 조용한 오값 대신 명시 거부(커널 미비, 후속 트랙).
        if let Some(&(g, sb)) = self.lin_grp.get(name)
            && (g != 128 || sb)
        {
            return Err(format!(
                "gemv_t {name}: g{g}/{} 커널 미비 — t행 프리필 미지원",
                if sb { "bf16" } else { "f16" }
            ));
        }
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
    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_launch(
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
    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_dev(
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
    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_t_launch(
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
    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_t_dev(
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

    /// 분리 GEMV 발사(이름 무경유 — MoE 전문가 스테이징).
    pub(in crate::rawcuda::w4a16_dec) fn gemv_launch_raw(
        &self,
        dq: CUdeviceptr,
        ds: CUdeviceptr,
        n: usize,
        k: usize,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let sym =
            crate::rawcuda::gptq4::kernel_sym(false, self.moe.moe_group, self.moe.moe_scale_bf16)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32, // [A2-float4 v2] 256폭·64레인 — 종전 grid 유지
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )
    }
}
