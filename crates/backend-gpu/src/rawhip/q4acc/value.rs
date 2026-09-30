//! q4acc 값 경로 — GEMM/GEMV 런치 기계 + MatmulHost·EwOps (plans/78 R1).

use super::*;
use crate::rawhip::env_on;

impl Q4Acc {
    /// 프레임 활성 q8 준비 — x(프레임 f32) → xq 스크래치. (xq, xq_w)
    pub(super) fn frame_quant(
        &self,
        x: *mut u8,
        n_in: usize,
        t: usize,
    ) -> Result<(*mut u8, usize), String> {
        let xq_w = xq_words(n_in);
        let buf = {
            let mut b = self.fxq.lock().map_err(|e| e.to_string())?;
            b.ensure(&self.ctx, t * xq_w * 4)?
        };
        self.ctx.quant_q8_b(x, buf, n_in, xq_w, t)?;
        Ok((buf, xq_w))
    }

    /// 프레임 GEMM 1건 — x는 프레임 f32, 무게는 mmap 참조(업로드 캐시).
    pub(super) fn frame_gemm(
        &self,
        x: *mut u8,
        w: &llm170_core::matmul::Weight<'_>,
        out: *mut u8,
        t: usize,
    ) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let (wd, f32w) = self.dev_weight(w)?;
        if f32w {
            return self.launch_gemm_f32(x, wd, n_in, n_out, t, out);
        }
        let ty = ggml_id(w.ty);
        // f16 경로는 t=1에서만 검증됨(plans/65 §19): t>1(프리필)은 x 취급이 어긋나
        // 값이 깨진다(f16-map t=4 프로브로 재현). t>1 해결 전에는 배선하지 않는다.
        let (xq, xq_w) = self.frame_quant(x, n_in, t)?;
        self.launch_gemm(ty, xq, wd, n_in, n_out, xq_w, t, out)
    }

    /// 프레임 op 런치 헬퍼 — gx/gy/gz + 32/64/128/256 스레드.
    pub(super) fn kop(
        &self,
        kern: &'static str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        self.ctx.launch3(kern, gx, gy, gz, block, args)
    }

    /// GEMV/GEMM 1런치 — xq는 이미 업로드·양자화된 활성 포인터.
    /// q8_0 듀얼 GEMV (t=1) — 같은 xq를 쓰는 인접 2 가중을 1런치로.
    /// gemm_q8_0_dual의 행 산술은 원판 gemm_q8_0과 동일(비트 불변).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm_q8_dual(
        &self,
        xq: *mut u8,
        w1: *mut u8,
        no1: usize,
        out1: *mut u8,
        w2: *mut u8,
        no2: usize,
        out2: *mut u8,
        ni: usize,
    ) -> Result<(), String> {
        let gy = (no1 + no2).min(65535) as u32;
        let gz = (no1 + no2).div_ceil(65535) as u32;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w1p = w1 as *mut std::ffi::c_void;
        let mut w2p = w2 as *mut std::ffi::c_void;
        let mut o1 = out1 as *mut std::ffi::c_void;
        let mut o2 = out2 as *mut std::ffi::c_void;
        let mut ni_a = ni as i32;
        let mut no1a = no1 as i32;
        let mut no2a = no2 as i32;
        let mut xw = crate::rawhip::q4acc::xq_words(ni) as i32;
        let mut args = vec![
            (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w1p) as *mut _ as *mut std::ffi::c_void,
            (&mut w2p) as *mut _ as *mut std::ffi::c_void,
            (&mut o1) as *mut _ as *mut std::ffi::c_void,
            (&mut o2) as *mut _ as *mut std::ffi::c_void,
            (&mut ni_a) as *mut _ as *mut std::ffi::c_void,
            (&mut no1a) as *mut _ as *mut std::ffi::c_void,
            (&mut no2a) as *mut _ as *mut std::ffi::c_void,
            (&mut xw) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("gemm_q8_0_dual", 1, gy, gz, 64, &mut args)
    }

    /// f32 듀얼 GEMV (t=1) — 같은 f32 활성을 쓰는 인접 2 가중을 1런치로.
    /// q4_gemm_f32_w2의 워프-퍼-출력 기하 = 원판과 동일 → 비트 불변.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm_f32_dual(
        &self,
        xf: *const u8,
        w1: *mut u8,
        no1: usize,
        out1: *mut u8,
        w2: *mut u8,
        no2: usize,
        out2: *mut u8,
        ni: usize,
    ) -> Result<(), String> {
        let gy = (no1 + no2).div_ceil(8) as u32;
        let mut x_p = xf as *mut std::ffi::c_void;
        let mut w1p = w1 as *mut std::ffi::c_void;
        let mut o1 = out1 as *mut std::ffi::c_void;
        let mut w2p = w2 as *mut std::ffi::c_void;
        let mut o2 = out2 as *mut std::ffi::c_void;
        let mut ni_a = ni as i32;
        let mut no1a = no1 as i32;
        let mut no2a = no2 as i32;
        let mut args = vec![
            (&mut x_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w1p) as *mut _ as *mut std::ffi::c_void,
            (&mut o1) as *mut _ as *mut std::ffi::c_void,
            (&mut w2p) as *mut _ as *mut std::ffi::c_void,
            (&mut o2) as *mut _ as *mut std::ffi::c_void,
            (&mut ni_a) as *mut _ as *mut std::ffi::c_void,
            (&mut no1a) as *mut _ as *mut std::ffi::c_void,
            (&mut no2a) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("q4_gemm_f32_w2", gy, 1, 1, 256, &mut args)
    }

    /// 혼합 듀얼 GEMV (t=1) — (q8_0, f32) 인접쌍을 1런치로.
    /// 블록 파티션: 전반은 gemm_q8_0 판, 후반은 q4_gemm_f32_w 판 → 비트 불변.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm_mix_dual(
        &self,
        xq: *mut u8,
        w8: *mut u8,
        no8: usize,
        out8: *mut u8,
        xf: u64,
        w4: *mut u8,
        no4: usize,
        out4: *mut u8,
        ni: usize,
    ) -> Result<(), String> {
        let gy = (no8 + no4.div_ceil(8)) as u32;
        let xqw = crate::rawhip::q4acc::xq_words(ni);
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w8p = w8 as *mut std::ffi::c_void;
        let mut o8 = out8 as *mut std::ffi::c_void;
        let mut no8a = no8 as i32;
        let mut xf_p = xf as *mut std::ffi::c_void; // f32 원활성 (프레임 핸들)
        let mut w4p = w4 as *mut std::ffi::c_void;
        let mut o4 = out4 as *mut std::ffi::c_void;
        let mut no4a = no4 as i32;
        let mut ni_a = ni as i32;
        let mut xw = xqw as i32;
        let mut args = vec![
            (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w8p) as *mut _ as *mut std::ffi::c_void,
            (&mut o8) as *mut _ as *mut std::ffi::c_void,
            (&mut no8a) as *mut _ as *mut std::ffi::c_void,
            (&mut xf_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w4p) as *mut _ as *mut std::ffi::c_void,
            (&mut o4) as *mut _ as *mut std::ffi::c_void,
            (&mut no4a) as *mut _ as *mut std::ffi::c_void,
            (&mut ni_a) as *mut _ as *mut std::ffi::c_void,
            (&mut xw) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("gemm_mix_dual", gy, 1, 1, 256, &mut args)
    }

    pub(super) fn launch_gemm(
        &self,
        ty: u32,
        xq: *mut u8,
        w: *mut u8,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        // q4_K MMQ급 타일 — 산술은 core dot_q4k_q8과 동일 순서로 썼다.
        // 2026-09-14 재검증: q4k-micro가 max_abs 0.0(CPU 참조와 비트 동일)이고
        // Flash-Next diverse(24토큰 프리필+8디코드)도 기준열과 비트 동일하다 —
        // 과거 "실측 오답(ffn_gate_exps t=20)" 기록은 이후 수정으로 해소됐다.
        // 다만 pp2048 실측이 9,012~9,118 → 8,899~9,155ms로 중립(노이즈 범위)이라
        // 기본은 여전히 끈 상태다: 이득이 아니라 속도 근거로 옵트인 유지.
        if ty == ggml_id(GgmlType::Q5_1) {
            // 타일 판은 출력 4개/블록 — 그리드도 4로 나눈다.
            // plans/78 R6: 타일 판 확정(복원 경로는 plans/109 P11 삭제).
            const TILED: bool = true;
            let outs_per_block = if TILED { 4usize } else { 1 };
            let nblk = n_out.div_ceil(outs_per_block);
            let gy = nblk.min(65535) as u32;
            let gz = nblk.div_ceil(65535) as u32;
            let part = self.ctx.scratch(n_out * 64 * 8)?;
            let mut xq_p = xq as *mut std::ffi::c_void;
            let mut w_p = w as *mut std::ffi::c_void;
            let mut part_p = part as *mut std::ffi::c_void;
            let mut o_p = out as *mut std::ffi::c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut tt = t as i32;
            // 16행 타일 판(2026-09-13) — 가중치 1회 독서로 상각. 원판은 행마다
            // 같은 가중치 행을 다시 읽어 MoE expert-down(20행 그룹)에서 20배
            // 증폭이었다(실측 2715ms/청크). 산술 순서는 동일 = 비트 동일.
            // MMQ급 판(스레드당 (출력,행) 누산) — 기본. 누산 순서가 달라
            // 비트 동일이 아니지만 q4-acc-check 실측 max_abs 5.96e-8 /
            // max_rel 2.1e-5 (q5_1 양자화 오차 ~1e-2의 1/500)이고 230토큰
            // greedy 스트림이 동일하다 — llama.cpp/vLLM과 같은 허용 오차 계약.
            // 비트 동일 판은 LLM170_Q5_1_EXACT=1로 복귀.
            let mmq = t >= 16 && !env_on("LLM170_Q5_1_EXACT") && ty == ggml_id(GgmlType::Q5_1);
            let kern = if mmq {
                "q4_gemm_q5_1_m"
            } else if TILED {
                "q4_gemm_q5_1_t"
            } else {
                "q4_gemm_q5_1"
            };
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            if kern.ends_with("_m") {
                let nblk = n_out.div_ceil(16);
                let smem = (16 * (n_in / 32) * 24) as u32;
                return self.ctx.launch3_dyn(
                    kern,
                    nblk.min(65535) as u32,
                    t.div_ceil(16) as u32,
                    1,
                    256,
                    smem,
                    &mut args,
                );
            }
            let gx = if TILED {
                t.div_ceil(16) as u32
            } else {
                t as u32
            };
            return self
                .ctx
                .launch3(kern, gx, gy, gz, if TILED { 256 } else { 64 }, &mut args);
        }
        // t≥16: MMQ 타일 우선 — 가중치 1회 독서 + 토큰 타일 상각(raw 디코더
        // mm_b와 동일 게이트). 타일 커널이 없는 타입은 GEMV 폴백.
        // 실측(2026-09-14, q4k-bench 2560x6144): 2.5-2.7 TFLOPS로 **t에 걸쳐 평탄**하다
        // (t=128 1.474ms 2.73, t=512 6.401 2.52, t=1024 12.581 2.56, t=2048 27.203
        // 2.37 TFLOPS). 즉 점유율 문제가 아니라 이 형상의 커널 고유 비용이고
        // 가중치 대역도 0.3-6 GB/s뿐이라 연산·대역폭 어느 쪽도 아니다 — llama.cpp
        // 대비 프리필 1.33x가 사는 곳이다. 27B가 같은 계열로 19.5 TFLOPS를 내는 것은
        // n_in/n_out이 더 큰 형상(5120x17408)이라 행당 상각이 크기 때문이다.
        // 같은 형상에서 q4_K MMQ 타일(LLM170_Q4K_MMQ)은 오히려 느렸고(33-34ms),
        // Q6K/Q4_K f16 융합 dequant도 중립이었다. 남은 방향은 그래프당 dequant 캐시.
        // t≥16: MMQ 타일 우선 — 단 **128토큰 이하로 쪼개서** 호출한다.
        // j128 CO는 gz>1(다중 토큰 사분면)일 때 n_in=6144 형상에서 폴트한다
        // (2026-09-12 실측: t=129 폴트, t=128 정상, GEMV 경로는 비트 동일).
        // plans/84 E.2 근본 원인/수정: MoE 폴백은 전문가별 행수 r을 t로 넘긴다 —
        // r은 청킹에 의존하므로(같은 전문가가 208청크에선 r=40, 16청크에선 r=3)
        // t>=16 임계값이면 같은 (토큰,전문가) 계산이 타일/GEMV 패밀리로 갈라져
        // ~1ulp 발산이 청크 불변성을 깬다. 프리필 핀 중에는 **모든 t**가 타일
        // 패밀리(핀이 j128+large-t로 고정)를 쓰게 한다 — 패밀리가 t 무관으로
        // 동일해져 행별 산술이 청킹과 무관해진다. 디코드(t=1)는 핀이 꺼져
        // 있어 기존 GEMV 그대로다.
        // plans/110 W2: 검증 배치 핀 중에는 타일 패밀리를 끈다 — frame_begin이
        // t>1에서 PREFILL_PIN을 켜는데, 검증 배치는 t=1 GEMV(디코드)와
        // 비트 동일이어야 한다(행핀이 gemv 경로를 잡는다).
        let pin_tile = crate::rawhip::ctx::PREFILL_PIN.load(std::sync::atomic::Ordering::Relaxed)
            && !crate::rawhip::ctx::VERIFY_ROW_PIN.load(std::sync::atomic::Ordering::Relaxed);
        if (t >= 16 || pin_tile) && !env_on("LLM170_Q4_NO_TILE") {
            // j128/v4 계열(=8/12/13/14/23)은 사분면 지원 — 그 외 타입만 128씩 분할.
            let tq_mode = matches!(
                ty,
                x if x == ggml_id(GgmlType::Q8_0)
                    || x == ggml_id(GgmlType::Q4K)
                    || x == ggml_id(GgmlType::Q5K)
                    || x == ggml_id(GgmlType::Q6K)
                    || x == ggml_id(GgmlType::Q3K)
            );
            let mut ok = true;
            for c in 0..if tq_mode { 1 } else { t.div_ceil(128) } {
                let t0 = c * 128;
                let tc = if tq_mode { t } else { 128.min(t - t0) };
                let xsrc = unsafe { xq.add(t0 * xq_w * 4) };
                let osrc = unsafe { out.add(t0 * n_out * 4) };
                if let Err(_e) = self
                    .ctx
                    .gemm_tile(xsrc, w, self.ktab2, ty, n_in, n_out, xq_w, tc, osrc)
                {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(());
            }
        }
        self.ctx.gemv_q8_out(
            xq as *const u8,
            w as *const u8,
            self.ktab2 as *const u8,
            ty,
            n_in,
            n_out,
            out,
            xq_w,
            t,
        )
    }

    /// f32 무게(라우터 등) GEMV — 양자화 없이 업로드한 활성을 직접 소비.
    pub(super) fn launch_gemm_f32(
        &self,
        x: *mut u8,
        w: *mut u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        // plans/110 W2: 검증 배치 핀 — 행별 t=1 디스패치(mt 판은 w판과
        // 누산 재배열 편차가 있다). 라우터/hc inject 등 f32 무게 전용.
        if (2..=8).contains(&t)
            && n_in.is_multiple_of(4)
            && crate::rawhip::ctx::VERIFY_ROW_PIN.load(std::sync::atomic::Ordering::Relaxed)
        {
            for r in 0..t {
                let xs = unsafe { x.add(r * n_in * 4) };
                let os = unsafe { out.add(r * n_out * 4) };
                self.launch_gemm_f32(xs, w, n_in, n_out, 1, os)?;
            }
            return Ok(());
        }
        let mut x_p = x as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut o_p = out as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut st = n_in as i32;
        // MMQ급 타일 — 커널 자체는 276→176ms로 빨라지지만(스레드당 40 MAC →
        // 2560 MAC) 엔드투엔드 pp512는 136.5 vs 137.8로 **차이 없음**(파이프라인
        // 뒤에 숨음). 이득 없는 계약 변경이라 기본에서 제외 — 옵트인만 남긴다.
        // f32 MMQ 판 기본(2026-09-13): pp2048 10,781→10,307ms, 토큰 동일.
        if t >= 16 {
            let mut tt = t as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut st) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3(
                "q4_gemm_f32_m",
                n_out.div_ceil(16) as u32,
                t.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            );
        }
        // plans/73: t=1은 워프-퍼-출력판 — 저출력(hc inject [10240→4])·라우터
        // 형상에서 원판 대비 3-6×. 누산 재배열 편차는 게이트로 검증.
        if t == 1 && n_in.is_multiple_of(4) {
            // plans/78 R6: F32W 폐기 — f32 직독 기본
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3(
                "q4_gemm_f32_w",
                n_out.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            );
        }
        // plans/74: t=2..8 은 멀티토큰 워프판(무게 1회 독서) — 종전 t판은
        // np 라우터/PLE 투영에서 17GB/s였다. LLM170_NO_F32MT=1 복귀.
        if (2..=8).contains(&t) && n_in.is_multiple_of(4) {
            let mut tt = t as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3(
                "q4_gemm_f32_mt",
                n_out.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            );
        }
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut x_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut no) as *mut _ as *mut std::ffi::c_void,
            (&mut st) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx
            .launch3("q4_gemm_f32", t as u32, gy, gz, 64, &mut args)
    }

    /// 배치 GEMM 본체 — xs [t][n_in] f32 → outs [t][n_out] f32.
    /// `x_start`/`w_off`은 moe_down의 전문가 그룹 런치용 부분 범위.
    /// 활성 준비 — 업로드(+q8 양자화) 1회. 그룹/전문가 호출이 공유한다.
    /// 반환: (xf 포인터, xq 포인터(양자화 전이면 null), xq_w, t). w_f32면 xf를 쓴다.
    fn prepare_x(
        &self,
        xs: &[Vec<f32>],
        n_in: usize,
        w_f32: bool,
    ) -> Result<(*mut u8, *mut u8, usize, usize), String> {
        let t = xs.len();
        let mut xflat = Vec::with_capacity(t * n_in);
        for row in xs {
            if row.len() != n_in {
                return Err(format!("matmul: x({}) != n_in({n_in})", row.len()));
            }
            xflat.extend_from_slice(row);
        }
        let xdev = {
            let mut xb = self.xf.lock().map_err(|e| e.to_string())?;
            xb.ensure(&self.ctx, t * n_in * 4)?
        };
        self.ctx.h2d(xdev, bytemuck::cast_slice(&xflat))?;
        if w_f32 {
            return Ok((xdev, std::ptr::null_mut(), 0, t));
        }
        let xq_w = xq_words(n_in);
        let xq_buf = {
            let mut xb = self.xq.lock().map_err(|e| e.to_string())?;
            xb.ensure(&self.ctx, t * xq_w * 4)?
        };
        self.ctx.quant_q8_b(xdev, xq_buf, n_in, xq_w, t)?;
        Ok((xdev, xq_buf, xq_w, t))
    }

    /// 준비된 활성으로 1회 런치 + 판독.
    fn run_prepared(
        &self,
        xf: *mut u8,
        xq: *mut u8,
        xq_w: usize,
        t: usize,
        w: &llm170_core::matmul::Weight<'_>,
        w_off_bytes: usize,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let tt = llm170_diag::dump::opts().key("q4acc_time");

        let t_up = std::time::Instant::now();
        let (w_dev, w_f32) = self.dev_weight(w)?;
        let up_ns = t_up.elapsed().as_nanos() as u64;
        let w_slice = unsafe { w_dev.add(w_off_bytes) };
        let ydev = {
            let mut yb = self.yf.lock().map_err(|e| e.to_string())?;
            // f16 경로는 128 사분면 경계까지 쓰므로 여유를 둔다(행 < t 만 사용).
            let need = if env_on("LLM170_F16_ACC") {
                t.div_ceil(128) * 128 * n_out * 4
            } else {
                t * n_out * 4
            };
            yb.ensure(&self.ctx, need)?
        };
        // 실측(2026-09-14, pp2048): 아래 할당+d2h+행 산포가 청크당 ~0.75s(8%)를
        // 쓴다(LLM170_Q4ACC_TIME으로 d2h=1.5s/400콜). 스테이지가 행 벡터 대신
        // 디바이스 상주 버퍼를 받으면 사라지는 비용 — QSA 선택목록 물질화와 같은 뿌리.
        //
        // 장문맥 디코드(pp8192, t=1)에서는 이 d2h가 **호출 비용의 96%**다:
        // LLM170_Q4ACC_TIME 600콜 기준 평균 5.4ms/호출인데 upload 0.2s·quant 0.0s·
        // launch 0.0s(비동기)이고 d2h만 3.1s(=5.2ms/호출)다. d2h는 앞서 큐에 넣은
        // 디바이스 작업을 기다리므로 이 값은 "그 호출이 동기화하는 디바이스 시간"이다.
        // t=1 GEMV 자체는 수십 us이므로, 스텝 비용을 결정하는 것은 **동기화 횟수**다
        // (스텝당 ~25회 x 5.2ms ≈ 130ms = 프레임 t=1 실측 137ms와 일치).
        // 다음 지렛대: 동기 횟수를 줄이거나(그룹핑) d2h를 뒤로 미루는 것.
        //
        // d2h 대역 자체도 실측했다(q4-d2h-bench): 8MB 0.476ms = **17.6 GB/s**.
        // 따라서 프리필의 호출당 5.2ms는 대부분 출력 전송이다(t=2048 x n_out 6144
        // = 50MB -> 2.8ms + 런치/커널). 즉 **출력을 호스트로 가져오는 한 이 비용은
        // 사라지지 않는다** — 스테이지 API를 디바이스 상주 버퍼로 바꾸는 것이
        // 프리필(8%)과 장문맥 디코드(58% QSA 스테이지)의 공통 해법이다.
        // 출력 스테이징은 **영속 버퍼**를 재사용한다: d2h가 전체를 덮어쓰므로
        // 매 호출 `vec![0.0; t*n_out]`로 할당+0-채움할 필요가 없다(프리필에서
        // 호출당 50MB — 그룹 5회면 250MB의 memset이 사라진다).
        let mut yb = self.ybuf.lock().map_err(|e| e.to_string())?;
        if yb.len() < t * n_out {
            yb.resize(t * n_out, 0.0); // 확장 시에만 0 채움
        }
        let t_k = std::time::Instant::now();
        if w_f32 {
            self.launch_gemm_f32(xf, w_slice, n_in, n_out, t, ydev)?;
        } else {
            // f16 경로 A/B — 실모델 텐서·실활성으로 검증(q4-acc-check가 미러와 대조).
            // 실측(2026-09-14): 게이트를 Q4_K/Q6_K(12/14)로 넓혀도 pp2048 중립이었다
            // (9,087.8 vs 9,043.9ms). 경로는 실제로 타고(F16_DBG=48콜: QSA wq
            // [6144x2560]x12, wk/wv [2560x512]x24) 200토큰 프롬프트 greedy 스트림도
            // 동일했지만, dequant가 **호출마다** 돌아 상각되지 않는다. plans/66 P1의
            // 실제 내용은 "그래프당 1회 dequant 후 캐시"이고 그게 빠져 있다.
            let ty0 = ggml_id(w.ty);
            if t >= 32
                && ty0 == 8
                && env_on("LLM170_F16_ACC")
                && self
                    .ctx
                    .gemm_f16_deq(ty0, xf as *const u8, w_slice, n_in, n_out, t, ydev)
                    .is_ok()
            {
            } else {
                self.launch_gemm(ty0, xq, w_slice, n_in, n_out, xq_w, t, ydev)?;
            }
        }
        let k_ns = t_k.elapsed().as_nanos() as u64;
        let t_d = std::time::Instant::now();
        let yflat = &mut yb[..t * n_out];
        self.ctx.d2h(bytemuck::cast_slice_mut(yflat), ydev)?;
        let d_ns = t_d.elapsed().as_nanos() as u64;
        if tt {
            self.note(up_ns, 0, k_ns, d_ns);
        }
        for (o, v) in outs.iter_mut().zip(yflat.chunks_exact(n_out)) {
            o.copy_from_slice(v);
        }
        Ok(())
    }

    /// 배치 GEMM 본체 — xs [t][n_in] f32 → outs [t][n_out] f32.
    fn batch_into(
        &self,
        xs: &[Vec<f32>],
        outs: &mut [Vec<f32>],
        w: &llm170_core::matmul::Weight<'_>,
        w_off_bytes: usize,
    ) -> Result<(), String> {
        if xs.is_empty() {
            return Ok(());
        }
        let n_in = w.n_in as usize;
        // f32 계열은 양자화를 건너뛰므로 준비 단계가 w_f32를 알아야 한다.
        let w_f32 = matches!(w.ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let (xf, xq, xq_w, t) = self.prepare_x(xs, n_in, w_f32)?;
        self.run_prepared(xf, xq, xq_w, t, w, w_off_bytes, outs)
    }
}

/// PleSsd 블록 확보 — LRU(접근 tick 갱신, 지연 힙 증발: 자주 쓰면 RAM 유지,
/// 오래 안 쓰면 증발 → SSD 재판독). gather 본체와 예열(ple_ssd_warm)이 공유.
fn ple_ensure_block(st: &mut crate::rawhip::q4acc::PleSsd, bidx: u64) -> Result<u32, String> {
    use std::os::unix::fs::FileExt as _;
    st.tick += 1;
    // QA-4(plans/114): 힙 상한 — 접근(힛 포함)마다 push되는 항은
    // free 소진 없이 팝되지 않아, 워킹셋이 캐시에 드는 부하(디코드
    // t=1)에서 접근 수에 비례해 영구 증가했다. 상한 초과 시 현재
    // 점유 슬롯 기준 유효 항만으로 재구축(O(n_slots), amortized).
    if st.evict.len() > st.blocks.len() * 2 + 1024 {
        st.evict = st
            .blocks
            .values()
            .map(|&sl| std::cmp::Reverse((st.slot_tick[sl as usize], sl)))
            .collect();
    }
    if let Some(&s) = st.blocks.get(&bidx) {
        st.slot_tick[s as usize] = st.tick;
        st.evict.push(std::cmp::Reverse((st.tick, s)));
        return Ok(s);
    }
    let s = match st.free.pop() {
        Some(s) => s,
        None => {
            // stale 항목(재삽입 이전 tick)은 버리고 최저 접근 슬롯 증발.
            loop {
                let std::cmp::Reverse((tick, s)) =
                    st.evict.pop().ok_or("ple_gather: 캐시 증발 큐 비정상")?;
                if st.slot_tick[s as usize] == tick {
                    st.blocks.remove(&st.slot_block[s as usize]);
                    break s;
                }
            }
        }
    };
    let base = s as usize * crate::rawhip::q4acc::PLE_SSD_BLOCK;
    // QA-7(plans/114): 파일 끝의 미만 블록 — read_exact(4KB) 고정은
    // UnexpectedEof로 실패했다. 잔여 바이트만 pread, 나머지는 0
    // (테이블 범위 밖 미판독 영역).
    let boff = bidx * crate::rawhip::q4acc::PLE_SSD_BLOCK as u64;
    let want = crate::rawhip::q4acc::PLE_SSD_BLOCK.min(st.file_len.saturating_sub(boff) as usize);
    if want == 0 {
        return Err(format!("ple_gather: 블록 {bidx}가 파일 범위 밖"));
    }
    // 4차 W-O: 전체 블록은 O_DIRECT(정렬 scratch) — 실패 시 버퍼드
    // 폴백 + direct 영구 해제(파일시스템 미지원 등).
    let mut read_err = None;
    if want == crate::rawhip::q4acc::PLE_SSD_BLOCK {
        let df = st.direct.take();
        if let Some(f) = &df
            && let Err(e) = f.read_exact_at(&mut st.scratch.0[..], boff)
        {
            read_err = Some(e);
        }
        if read_err.is_none() {
            st.direct = df;
            st.arena[base..base + crate::rawhip::q4acc::PLE_SSD_BLOCK]
                .copy_from_slice(&st.scratch.0);
        } else {
            eprintln!("# ple-ssd: O_DIRECT 실패({:?}) — 버퍼드 폴백", read_err);
        }
    }
    if read_err.is_some() || want < crate::rawhip::q4acc::PLE_SSD_BLOCK {
        let mut buf = vec![0u8; crate::rawhip::q4acc::PLE_SSD_BLOCK];
        st.file
            .read_exact_at(&mut buf[..want], boff)
            .map_err(|e| format!("ple_gather: pread {bidx}: {e}"))?;
        st.arena[base..base + crate::rawhip::q4acc::PLE_SSD_BLOCK].copy_from_slice(&buf);
    }
    st.blocks.insert(bidx, s);
    st.slot_block[s as usize] = bidx;
    st.slot_tick[s as usize] = st.tick;
    st.evict.push(std::cmp::Reverse((st.tick, s)));
    Ok(s)
}

impl llm170_core::matmul::MatmulHost for Q4Acc {
    fn barrier(&self) {
        unsafe {
            let _ = ck(hip::hipDeviceSynchronize(), "hipDeviceSynchronize");
        }
    }

    fn matmul(
        &self,
        x: &[f32],
        w: &llm170_core::matmul::Weight<'_>,
        out: &mut [f32],
    ) -> Result<(), String> {
        let mut o = vec![vec![0.0f32; w.n_out as usize]];
        let xs = [x.to_vec()];
        self.batch_into(&xs, &mut o, w, 0)?;
        out.copy_from_slice(&o[0]);
        Ok(())
    }

    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &llm170_core::matmul::Weight<'_>,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.batch_into(xs, outs, w, 0)
    }

    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[llm170_core::matmul::Weight<'_>],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        // 동일 입력 — 업로드·양자화 1회를 그룹 전체가 공유한다(값 경로에서
        // 왕복이 스텝 비용의 대부분이라 그룹 호출당 3→1로 줄인다).
        if ws.len() != outs.len() {
            return Err(format!(
                "matmul_group: ws({}) != outs({})",
                ws.len(),
                outs.len()
            ));
        }
        if ws.is_empty() || xs.is_empty() {
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        let f32_family = |t: GgmlType| matches!(t, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let w_f32 = f32_family(ws[0].ty);
        // 타입 계열·n_in이 섞이면 준비를 공유할 수 없다 — 개별 경로로.
        if ws
            .iter()
            .any(|w| w.n_in as usize != n_in || f32_family(w.ty) != w_f32)
        {
            for (w, o) in ws.iter().zip(outs.iter_mut()) {
                self.batch_into(xs, o, w, 0)?;
            }
            return Ok(());
        }
        let (xf, xq, xq_w, t) = self.prepare_x(xs, n_in, w_f32)?;
        for (w, o) in ws.iter().zip(outs.iter_mut()) {
            self.run_prepared(xf, xq, xq_w, t, w, 0, o)?;
        }
        Ok(())
    }

    fn matmul_paired(
        &self,
        xs: &[Vec<f32>],
        ws: &[llm170_core::matmul::Weight<'_>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        if ws.len() != xs.len() || ws.len() != outs.len() {
            return Err(format!(
                "matmul_paired: 형상 불일치 ws={} xs={} outs={}",
                ws.len(),
                xs.len(),
                outs.len()
            ));
        }
        for ((x, w), o) in xs.iter().zip(ws.iter()).zip(outs.iter_mut()) {
            let one = [x.clone()];
            let mut oo = [std::mem::take(o)];
            self.batch_into(&one, &mut oo, w, 0)?;
            *o = std::mem::take(&mut oo[0]);
        }
        Ok(())
    }

    /// MoE 전문가 스택 배치 — ids로 전문가를 묶어 그룹별 런치.
    /// ids가 가리키는 전문가 슬라이스는 스택에서 연속이므로 바이트 오프셋만
    /// 옮기면 기존 GEMV 커널이 그대로 성립한다(mul_mat_id의 오프셋 형태).
    fn moe_down(
        &self,
        xs: &[Vec<f32>],
        ws: &llm170_core::matmul::Weight<'_>,
        expert_ids: &[u32],
        n_expert_stack: usize,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        if t != expert_ids.len() || t != outs.len() {
            return Err(format!(
                "moe_down: 형상 불일치 xs={} ids={} outs={}",
                t,
                expert_ids.len(),
                outs.len()
            ));
        }
        if t == 0 {
            return Ok(());
        }
        let per_expert = ws.data.len() / n_expert_stack.max(1);
        let n_in = ws.n_in as usize;
        // 3D 전문가 스택은 n_out = 전문가수×전문가당 행으로 온다 — 런치·출력
        // 버퍼는 전문가당 행 기준이다(mul_mat_id와 동일한 해석).
        let n_out = ws.n_out as usize / n_expert_stack.max(1);
        let (w_dev, w_f32) = self.dev_weight(ws)?;
        // 활성 업로드 + 양자화 1회 (전문가 공통)
        let (xdev_f32, xq_buf, xq_w, t) = self.prepare_x(xs, n_in, w_f32)?;
        let mut yflat = vec![0.0f32; t * n_out];
        let ydev = {
            let mut yb = self.yf.lock().map_err(|e| e.to_string())?;
            yb.ensure(&self.ctx, t * n_out * 4)?
        };
        // 전문가 순 그룹화 (2026-09-13): ids는 확률순이라 연속 런이 1행씩
        // 흩어진다(프레임 실측 t=512·k=10 ≈4000런치/층). 카운팅 정렬로 묶어
        // 런치 수를 전문가 수 수준으로 줄인다. x 행은 순열 gather로 모으고,
        // 결과 행 순서는 d2h 후 호스트 산란으로 복원한다(가중합이 원래 행
        // 순서를 요구 — 호스트 비용은 perm 인덱싱뿐).
        let ne = n_expert_stack.max(1);
        // 카운팅 정렬 테이블 — common 공용판(프레임 호스트 빌드·vk 폴백과
        // 동일 코드, P13).
        let off = crate::common::moe::grp_offsets(expert_ids, ne);
        let perm = crate::common::moe::grp_perm(expert_ids, ne, &off);
        let row_u32 = if w_f32 { n_in } else { xq_w };
        let xbase = if w_f32 { xdev_f32 } else { xq_buf };
        let xg = {
            let mut g = self.xperm.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, t * row_u32 * 4)?
        };
        self.rows_permute(xbase, &perm, xg, row_u32, t)?;
        for e in 0..ne {
            let rows = off[e + 1] - off[e];
            if rows == 0 {
                continue;
            }
            let start = off[e];
            let xsrc = unsafe { xg.add(start * row_u32 * 4) };
            let wsrc = unsafe { w_dev.add(e * per_expert) };
            let dst = unsafe { ydev.add(start * n_out * 4) };
            if w_f32 {
                self.launch_gemm_f32(xsrc, wsrc, n_in, n_out, rows, dst)?;
            } else {
                self.launch_gemm(ggml_id(ws.ty), xsrc, wsrc, n_in, n_out, xq_w, rows, dst)?;
            }
        }
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut yflat), ydev)?;
        for (g, v) in yflat.chunks_exact(n_out).enumerate() {
            outs[perm[g] as usize].copy_from_slice(v);
        }
        Ok(())
    }

    /// QSA 마스크드 밀집 GQA (값 경로 브리지) — f32 캐시.
    fn total_mem_bytes(&self) -> u64 {
        let (mut f, mut t) = (0usize, 0usize);
        unsafe {
            if hip::hipMemGetInfo(&mut f, &mut t) != hip::hipError_t_hipSuccess {
                return 0;
            }
        }
        t as u64
    }
}

impl llm170_core::matmul::EwOps for Q4Acc {
    fn shexp_gu(
        &self,
        x: u64,
        wg: &llm170_core::matmul::Weight,
        wu: &llm170_core::matmul::Weight,
        h: u64,
        n_in: usize,
        n_hidden: usize,
    ) -> Result<(), String> {
        let mut xp = self.fptr(x)? as *mut std::ffi::c_void;
        let (wgd, _) = self.dev_weight(wg)?;
        let (wud, _) = self.dev_weight(wu)?;
        let mut wgp = wgd as *mut std::ffi::c_void;
        let mut wup = wud as *mut std::ffi::c_void;
        let mut hp = self.fptr(h)? as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut nh = n_hidden as i32;
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wgp) as *mut _ as *mut std::ffi::c_void,
            (&mut wup) as *mut _ as *mut std::ffi::c_void,
            (&mut hp) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
        ];
        // n_hidden=640, warp당 1행 → 640 워프 = 20블록(256스레드=8워프)
        self.ctx.launch3(
            "q4_shexp_gu",
            n_hidden.div_ceil(8) as u32,
            1,
            1,
            256,
            &mut args,
        )
    }

    fn shexp_da(
        &self,
        h: u64,
        wd: &llm170_core::matmul::Weight,
        s: u64,
        mout: u64,
        n_in: usize,
        n_hidden: usize,
    ) -> Result<(), String> {
        let mut hp = self.fptr(h)? as *mut std::ffi::c_void;
        let (wdd, _) = self.dev_weight(wd)?;
        let mut wdp = wdd as *mut std::ffi::c_void;
        let mut sp = self.fptr(s)? as *mut std::ffi::c_void;
        let mut mp = self.fptr(mout)? as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut nh = n_hidden as i32;
        let mut args = vec![
            (&mut hp) as *mut _ as *mut std::ffi::c_void,
            (&mut wdp) as *mut _ as *mut std::ffi::c_void,
            (&mut sp) as *mut _ as *mut std::ffi::c_void,
            (&mut mp) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
        ];
        // n_in=2560, warp당 1행 → 2560 워프 = 320블록(8워프/블록)
        self.ctx
            .launch3("q4_shexp_da", n_in.div_ceil(8) as u32, 1, 1, 256, &mut args)
    }

    fn ple_math_dev(
        &self,
        res: u64,
        key: u64,
        value: u64,
        nk: &[f32],
        nq: &[f32],
        nc: &[f32],
        conv_w: &[f32],
        gated: u64,
        conv_out: u64,
        gate_out: u64,
        seq: usize,
        pos0: usize,
        t: usize,
        eps: f32,
        n_embd: usize,
        hc: usize,
        kern: usize,
        dil: usize,
        hist: usize,
        host_ring: &[f32],
    ) -> Result<(), String> {
        // plans/111 W4c: t>1(프리필) 디바이스 경로 해제(사용자 승인, 원장 132
        // 부정 1의 재적용) — forward.rs 체인 재배선으로 GPU gather 출력이
        // frame_mm_group→ple_math 로 이어진다. key/value GEMM 산술 클래스는
        // 골든 재캡처로 승인(디코드 t=1이 이미 같은 GPU GEMM 패밀리 사용).
        let hc_dim = hc * n_embd;
        let ring_bytes = hist * hc_dim * 4;
        // 링 워터마크는 pos 기반(vk 판과 동일 의미론) — t 기반은 되감기
        // (스펙 검증 롤백 재실행, plans/110 W2)을 감지하지 못해 디바이스 링이
        // 기각된 타임라인의 활성을 담은 채로 남는다. pos0 < 워터마크면 호스트
        // 링(정합 상태)으로 리프레시.
        let rewind = {
            let mut wm = self.ple_ring_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry(seq).or_insert(0);
            let rw = pos0 < *w;
            *w = pos0 + t;
            rw
        };
        let ring = {
            let mut m = self.ple_ring.lock().map_err(|e| e.to_string())?;
            let g = m.entry(seq).or_insert_with(|| GBuf::new("ple_ring"));
            // 주의: ensure 가 ptr 을 세우므로 최초 판정은 ensure **전**에.
            let fresh = g.ptr.is_null();
            g.ensure(&self.ctx, ring_bytes)?;
            if fresh || rewind {
                // 최초/되감기: 호스트 링(정합 상태)으로 초기화 — 동기 h2d 1회.
                self.ctx.h2d(g.ptr, bytemuck::cast_slice(host_ring))?;
            }
            g.ptr
        };
        let (resp, keyp, valp, gp, cop, gop) = (
            self.fptr(res)?,
            self.fptr(key)?,
            self.fptr(value)?,
            self.fptr(gated)?,
            self.fptr(conv_out)?,
            self.fptr(gate_out)?,
        );
        let nk_d = self.upload_map(&self.ple_nk, "ple_nk", nk)?;
        let nq_d = self.upload_map(&self.ple_nq, "ple_nq", nq)?;
        let nc_d = self.upload_map(&self.ple_nc, "ple_nc", nc)?;
        let cw_d = self.upload_map(&self.ple_cw, "ple_cw", conv_w)?;
        // (1) gate + 방송 + 그룹 norm — 워프당 (t,s), 레인 0 실행.
        {
            let (mut rp, mut kp, mut vp) = (
                resp as *mut std::ffi::c_void,
                keyp as *mut std::ffi::c_void,
                valp as *mut std::ffi::c_void,
            );
            let (mut nk_, mut nq_, mut nc_) = (
                nk_d as *mut std::ffi::c_void,
                nq_d as *mut std::ffi::c_void,
                nc_d as *mut std::ffi::c_void,
            );
            let (mut gp_, mut gop_) = (gp as *mut std::ffi::c_void, gop as *mut std::ffi::c_void);
            let (mut e, mut ne, mut hcc, mut tt) = (eps, n_embd as i32, hc as i32, t as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut rp) as *mut _ as *mut std::ffi::c_void,
                (&mut kp) as *mut _ as *mut std::ffi::c_void,
                (&mut vp) as *mut _ as *mut std::ffi::c_void,
                (&mut nk_) as *mut _ as *mut std::ffi::c_void,
                (&mut nq_) as *mut _ as *mut std::ffi::c_void,
                (&mut nc_) as *mut _ as *mut std::ffi::c_void,
                (&mut gp_) as *mut _ as *mut std::ffi::c_void,
                (&mut gop_) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut ne) as *mut _ as *mut std::ffi::c_void,
                (&mut hcc) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_ple_gate",
                hc.div_ceil(8) as u32,
                t as u32,
                1,
                256,
                &mut args,
            )?;
        }
        // (2) dilated conv + silu + 링 갱신.
        {
            let (mut gp_, mut cw_, mut ring_, mut cop_) = (
                gp as *mut std::ffi::c_void,
                cw_d as *mut std::ffi::c_void,
                ring as *mut std::ffi::c_void,
                cop as *mut std::ffi::c_void,
            );
            let (mut hd, mut tt, mut k2, mut d2, mut h2) = (
                hc_dim as i32,
                t as i32,
                kern as i32,
                dil as i32,
                hist as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut gp_) as *mut _ as *mut std::ffi::c_void,
                (&mut cw_) as *mut _ as *mut std::ffi::c_void,
                (&mut ring_) as *mut _ as *mut std::ffi::c_void,
                (&mut cop_) as *mut _ as *mut std::ffi::c_void,
                (&mut hd) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut k2) as *mut _ as *mut std::ffi::c_void,
                (&mut d2) as *mut _ as *mut std::ffi::c_void,
                (&mut h2) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_ple_conv",
                hc_dim.div_ceil(256) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        // (3) 잔차.
        {
            let (mut rp, mut vp, mut gop_, mut cop_) = (
                resp as *mut std::ffi::c_void,
                valp as *mut std::ffi::c_void,
                gop as *mut std::ffi::c_void,
                cop as *mut std::ffi::c_void,
            );
            let (mut ne, mut hcc, mut tt) = (n_embd as i32, hc as i32, t as i32);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut rp) as *mut _ as *mut std::ffi::c_void,
                (&mut vp) as *mut _ as *mut std::ffi::c_void,
                (&mut gop_) as *mut _ as *mut std::ffi::c_void,
                (&mut cop_) as *mut _ as *mut std::ffi::c_void,
                (&mut ne) as *mut _ as *mut std::ffi::c_void,
                (&mut hcc) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_ple_residual",
                n_embd.div_ceil(256) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        Ok(())
    }

    /// plans/110 W2 — 디바이스 PLE 링 → 호스트 판독(hip). 종전 미구현으로
    /// 디코드가 CPU ple_conv를 동기화하지 못했고, 스펙 롤백의 pos 기반
    /// 되감기 refresh가 stale 호스트 링으로 디바이스 링을 오염시켰다.
    /// 계약: GPU 유휴 시점(판독 동기 후) 호출.
    fn ple_ring_sync(&self, seq: usize, ring_out: &mut [f32]) -> Result<(), String> {
        let m = self.ple_ring.lock().map_err(|e| e.to_string())?;
        let Some(g) = m.get(&seq) else {
            return Err("ple_ring_sync: 링 없음".into());
        };
        let n = ring_out.len().min(g.bytes / 4);
        // SAFETY (107 W8): 링 GBuf 매핑 판독 — n은 g.bytes/4 상한 클램프,
        // 호출 계약상 선행 판독이 스트림을 동기화했다.
        unsafe {
            std::ptr::copy_nonoverlapping(g.ptr as *const f32, ring_out.as_mut_ptr(), n);
        }
        Ok(())
    }

    /// plans/111 W2 — token_embd(Q8_0) gather + hc 방송(vk plans/97 의 hip 이식).
    /// 종전 hip 미구현으로 매 스텝 CPU 디양자화+h2d(~9ms)·프리필 청크당
    /// 48-70ms를 냈다. 테이블은 weights 캐시(mmap ptr 키)로 1회 상주.
    fn emb_q8_gather_dev(
        &self,
        table_key: usize,
        table: &[u8],
        tokens: &[u32],
        out: u64,
        n: usize,
        hc: usize,
    ) -> Result<(), String> {
        if !n.is_multiple_of(32) {
            return Err(format!("emb_q8g: n%32 != 0 ({n})"));
        }
        let t = tokens.len();
        if t == 0 {
            return Ok(());
        }
        let bpr = n / 32;
        // 테이블 1회 상주 — host-pinned 제로카피(plans/111 W4c: 카브아웃이 풀이라
        // hipMalloc(675MB)이 serve에서 간헐 실패해 CPU 폴백했었다. PLE과 동일 패턴).
        let tbl = {
            let mut c = self.ple_tbl.lock().map_err(|e| e.to_string())?;
            if let Some((k, p)) = *c {
                if k != table_key {
                    return Err("emb_q8g: 테이블 키 교체 미지원(단일 모델)".into());
                }
                p
            } else {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                // SAFETY: hipMallocHost 원본 바인딩 — p 미초기화 포인터 인자.
                unsafe {
                    crate::rawhip::hip::hipMallocHost(&mut p, table.len().max(1));
                }
                if p.is_null() {
                    return Err("emb_q8g: hipMallocHost 실패".into());
                }
                let p = p as *mut u8;
                // SAFETY (107 W8): p는 hipMallocHost(table.len()) 바이트 — 1회 순차 복사.
                unsafe {
                    std::ptr::copy_nonoverlapping(table.as_ptr(), p, table.len());
                }
                *c = Some((table_key, p));
                p
            }
        };
        // ids 업로드 — 매 콜(직전 것 교체).
        let ids = {
            let mut g = self.emb_ids.lock().map_err(|e| e.to_string())?;
            let p = g.ensure(&self.ctx, t * 4)?;
            // SAFETY (107 W8): ids 업로드 — p는 ensure(t*4) 바이트, 재해석 길이 일치.
            self.ctx.h2d(p, unsafe {
                std::slice::from_raw_parts(tokens.as_ptr() as *const u8, t * 4)
            })?;
            p
        };
        // res_hc 버스 규약 — f16 쌍팩(hip 기본) 또는 f32.
        let f16 = llm170_core::qwen4exp::frame::res_f16_on();
        let need = if f16 { t * hc * n * 2 } else { t * hc * n * 4 };
        let cap = self.fcap(out)?;
        if cap < need {
            return Err(format!("emb_q8g: res_hc {cap}B < {need}B"));
        }
        let op = self.fptr(out)?;
        let (mut idp, mut tp) = (ids as *mut std::ffi::c_void, tbl as *mut std::ffi::c_void);
        let mut opv = op as *mut std::ffi::c_void;
        let (mut n_, mut t_, mut hc_, mut bpr_) = (n as i32, t as i32, hc as i32, bpr as i32);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut idp) as *mut _ as *mut std::ffi::c_void,
            (&mut tp) as *mut _ as *mut std::ffi::c_void,
            (&mut opv) as *mut _ as *mut std::ffi::c_void,
            (&mut n_) as *mut _ as *mut std::ffi::c_void,
            (&mut t_) as *mut _ as *mut std::ffi::c_void,
            (&mut hc_) as *mut _ as *mut std::ffi::c_void,
            (&mut bpr_) as *mut _ as *mut std::ffi::c_void,
        ];
        let name = if f16 { "q4_emb_q8g_f16" } else { "q4_emb_q8g" };
        self.ctx
            .launch(name, (t * bpr).div_ceil(256) as u32, 1, 256, &mut args)?;
        Ok(())
    }

    /// plans/111 4차 W-P — ssd 블록 캐시 선예열: 행이 닿는 4KB 블록 전수를
    /// 적재(이미 상주한 블록 스킵). 프리페치 워커가 호출 — 본경로 gather는
    /// 예열된 캐시에서 즉시 적중. 잠금은 gather와 공유(예열 중 gather 대기
    /// 가능하나 그 블록을 어차피 읽어야 했으므로 순손해 없음).
    fn ple_ssd_warm(&self, rows: &[u32]) {
        let Ok(mut guard) = self.ple_ssd.lock() else {
            return;
        };
        let Some(st) = guard.as_mut() else { return };
        const B: u64 = crate::rawhip::q4acc::PLE_SSD_BLOCK as u64;
        for &r in rows {
            let goff = st.base_off + r as u64 * st.row_bytes as u64;
            let end = goff + st.row_bytes as u64;
            let mut bidx = goff / B;
            let last = end.div_ceil(B);
            while bidx < last {
                if !st.blocks.contains_key(&bidx) {
                    let _ = ple_ensure_block(st, bidx);
                }
                bidx += 1;
            }
        }
    }

    /// ssd 오프로드 활성 판정(ram 모드·미초기화 = false).
    fn ple_table_ssd_active(&self) -> bool {
        self.ple_ssd.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// plans/111 W4c — PLE 임베딩 테이블(IQ4_NL) gather(vk plans/93 의 hip 이식).
    /// 서빙 옵션 `--ple-table ram|ssd|auto`(기본 auto — 엔진 set_ple_table_mode):
    /// - ram: 전체 테이블 host-pinned 상주. 소형 테이블 전용 — FN 테이블은
    ///   [160, 320M] = 28.8GB라 시스템 RAM(30GB)에 불가(283MB 라는 vk 주석은
    ///   오기 — 양 백엔드 GPU gather는 지금까지 실동작 0회).
    /// - ssd: 4KB 블록 캐시(1GiB FIFO) + pread 미스 + 컴팩트 업로드 — 접근된
    ///   행만 RAM에 상주(Engram 성장 대비 공통 경로). pread도 페이지캐시 경유.
    /// - auto: 테이블 ≤ 캐시 예산이면 ram, 아니면 ssd. 선택은 ONCE 로그.
    ///
    /// 블록 산술은 CPU deq_iq4_nl(deq.rs)과 비트 동일(커널 q4_ple_gather).
    #[allow(clippy::too_many_arguments)]
    fn ple_gather_dev(
        &self,
        table_key: usize,
        table: &[u8],
        rows: &[u32],
        out: u64,
        hd: usize,
    ) -> Result<(), String> {
        use std::collections::HashMap;

        let nrows = rows.len();
        if nrows == 0 {
            return Ok(());
        }
        let bpr = hd.div_ceil(32);
        let row_bytes = bpr * 18;
        let mode = crate::rawhip::q4acc::ple_table_mode_str();
        let mode = match mode {
            "ram" => 1,
            "ssd" => 2,
            // auto — 캐시 예산에 들어가면 ram, 아니면 ssd.
            _ => usize::from(table.len() <= crate::rawhip::q4acc::ple_ssd_cache_bytes()),
        };
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let name = match mode {
                    1 => "ram(전체 pinned)",
                    _ => "ssd(블록 캐시+pread)",
                };
                eprintln!(
                    "# ple-table: {name} — 테이블 {:.1} GiB",
                    table.len() as f64 / (1u64 << 30) as f64
                );
            });
        }
        if mode == 1 {
            // ── ram: 전체 host-pinned(제로카피) — 무게 카브아웃은 그대로. ──
            let tbl = {
                let mut c = self.ple_tbl.lock().map_err(|e| e.to_string())?;
                if let Some((k, p)) = *c {
                    if k != table_key {
                        return Err("ple_gather: 테이블 키 교체 미지원(단일 모델)".into());
                    }
                    p
                } else {
                    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                    // SAFETY: hipMallocHost 원본 바인딩 — p 미초기화 포인터 인자.
                    unsafe {
                        crate::rawhip::hip::hipMallocHost(&mut p, table.len().max(1));
                    }
                    if p.is_null() {
                        return Err("ple_gather: hipMallocHost 실패".into());
                    }
                    let p = p as *mut u8;
                    // SAFETY (107 W8): p는 hipMallocHost(table.len()) 바이트.
                    unsafe {
                        std::ptr::copy_nonoverlapping(table.as_ptr(), p, table.len());
                    }
                    *c = Some((table_key, p));
                    p
                }
            };
            let ids = {
                let mut g = self.ple_rows_buf.lock().map_err(|e| e.to_string())?;
                let p = g.ensure(&self.ctx, nrows * 4)?;
                // SAFETY (107 W8): rows 업로드 — ensure(nrows*4) 바이트, 길이 일치.
                self.ctx.h2d(p, unsafe {
                    std::slice::from_raw_parts(rows.as_ptr() as *const u8, nrows * 4)
                })?;
                p
            };
            self.ple_gather_launch(tbl as *const u8, ids, out, hd, nrows, bpr)?;
            return Ok(());
        }
        // ── ssd: 파트 파일 pread + 4KB 블록 캐시 + 컴팩트 업로드. ──
        let (file, base_off) = {
            let src = self
                .sources
                .iter()
                .find(|p| p.covers(table_key, table.len()).is_some())
                .ok_or("ple_gather: 테이블 파트 미커버")?;
            let off = src.covers(table_key, table.len()).unwrap();
            (src.file.try_clone().map_err(|e| e.to_string())?, off)
        };
        let mut guard = self.ple_ssd.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            let n_slots =
                crate::rawhip::q4acc::ple_ssd_cache_bytes() / crate::rawhip::q4acc::PLE_SSD_BLOCK;
            let file_len = file
                .metadata()
                .map_err(|e| format!("ple_gather: 파일 메타: {e}"))?
                .len();
            // 4차 W-O: O_DIRECT 재오픈 — 블록 캐시가 자체 LRU라 커널 페이지캐시와
            // 이중으로 쌓여 핫 페이지(가중 mmap)를 밀어내던 것을 끊는다
            // (ninfer read_direct 교훈). 미지원 FS는 폴백(버퍼드).
            let direct = {
                use std::os::unix::fs::OpenOptionsExt as _;
                let src2 = self
                    .sources
                    .iter()
                    .find(|p| p.covers(table_key, table.len()).is_some());
                src2.and_then(|p2| {
                    std::fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(0o40000) // Linux O_DIRECT
                        .open(&p2.path)
                        .ok()
                })
            };
            if direct.is_some() {
                eprintln!("# ple-ssd: O_DIRECT pread (페이지캐시 미경유)");
            }
            *guard = Some(crate::rawhip::q4acc::PleSsd {
                file,
                direct,
                scratch: Default::default(),
                file_len,
                base_off,
                row_bytes,
                blocks: HashMap::new(),
                slot_tick: vec![0; n_slots],
                slot_block: vec![0; n_slots],
                evict: std::collections::BinaryHeap::new(),
                tick: 0,
                arena: vec![0u8; n_slots * crate::rawhip::q4acc::PLE_SSD_BLOCK],
                free: (0..n_slots as u32).rev().collect(),
            });
        }
        let st = guard.as_mut().unwrap();
        // 블록 확보 — LRU(접근 tick 갱신, 지연 힙 증발: 자주 쓰면 RAM 유지,
        // 오래 안 쓰면 증발 → SSD 재판독. 사용자 의미론 계약). 자유함수로
        // 분리(4차 W-P: 예열 경로와 공유).
        let ensure_block = ple_ensure_block;

        // 행 dedupe → 컴팩트 사본 + slot id 재매핑.
        let mut slot_of: HashMap<u32, u32> = HashMap::with_capacity(nrows);
        let mut compact: Vec<u8> = Vec::with_capacity(nrows * row_bytes);
        let mut ids: Vec<u32> = Vec::with_capacity(nrows);
        for &r in rows {
            let n = slot_of.len() as u32;
            let slot = *slot_of.entry(r).or_insert(n);
            if slot == n {
                // 새 고유 행 — 블록 캐시에서 조립(블록 경계 관통 처리).
                let goff = st.base_off + r as u64 * st.row_bytes as u64;
                let mut in_off = (goff % crate::rawhip::q4acc::PLE_SSD_BLOCK as u64) as usize;
                let mut bidx = goff / crate::rawhip::q4acc::PLE_SSD_BLOCK as u64;
                let mut left = st.row_bytes;
                while left > 0 {
                    let s = ensure_block(st, bidx)?;
                    let base = s as usize * crate::rawhip::q4acc::PLE_SSD_BLOCK;
                    let take = left.min(crate::rawhip::q4acc::PLE_SSD_BLOCK - in_off);
                    compact.extend_from_slice(&st.arena[base + in_off..base + in_off + take]);
                    left -= take;
                    in_off = 0;
                    bidx += 1;
                }
            }
            ids.push(slot);
        }
        let n_uniq = slot_of.len();
        drop(guard);
        // 컴팩트 테이블 + slot ids 업로드 → 동일 커널.
        let tbl_d = {
            let mut g = self.ple_compact.lock().map_err(|e| e.to_string())?;
            let p = g.ensure(&self.ctx, n_uniq * row_bytes)?;
            self.ctx.h2d(p, &compact)?;
            p
        };
        let ids_d = {
            let mut g = self.ple_rows_buf.lock().map_err(|e| e.to_string())?;
            let p = g.ensure(&self.ctx, nrows * 4)?;
            // SAFETY (107 W8): ids 업로드 — ensure(nrows*4) 바이트, 길이 일치.
            self.ctx.h2d(p, unsafe {
                std::slice::from_raw_parts(ids.as_ptr() as *const u8, nrows * 4)
            })?;
            p
        };
        self.ple_gather_launch(tbl_d, ids_d, out, hd, nrows, bpr)?;
        Ok(())
    }
}

impl Q4Acc {
    /// ple_gather 공용 런치 — tbl/ids 포인터만 모드별로 다르다.
    fn ple_gather_launch(
        &self,
        tbl: *const u8,
        ids: *mut u8,
        out: u64,
        hd: usize,
        nrows: usize,
        bpr: usize,
    ) -> Result<(), String> {
        let need = nrows * hd * 4;
        let cap = self.fcap(out)?;
        if cap < need {
            return Err(format!("ple_gather: out {cap}B < {need}B"));
        }
        let op = self.fptr(out)?;
        let (mut rp, mut tp) = (ids as *mut std::ffi::c_void, tbl as *mut std::ffi::c_void);
        let mut opv = op as *mut std::ffi::c_void;
        let (mut hd_, mut nr) = (hd as i32, nrows as i32);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut rp) as *mut _ as *mut std::ffi::c_void,
            (&mut tp) as *mut _ as *mut std::ffi::c_void,
            (&mut opv) as *mut _ as *mut std::ffi::c_void,
            (&mut hd_) as *mut _ as *mut std::ffi::c_void,
            (&mut nr) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch(
            "q4_ple_gather",
            (nrows * bpr).div_ceil(256) as u32,
            1,
            256,
            &mut args,
        )?;
        Ok(())
    }
}
