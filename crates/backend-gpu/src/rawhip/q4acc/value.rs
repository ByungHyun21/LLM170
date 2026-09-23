//! q4acc 값 경로 — GEMM/GEMV 런치 기계 + MatmulHost·EwOps (plans/78 R1).

use super::*;
use crate::rawhip::env_on;

impl Q4Acc {


    /// 프레임 활성 q8 준비 — x(프레임 f32) → xq 스크래치. (xq, xq_w)
    pub(super) fn frame_quant(&self, x: *mut u8, n_in: usize, t: usize) -> Result<(*mut u8, usize), String> {
        let xq_w = xq_words(n_in);
        let buf = {
            let mut b = self.fxq.lock().map_err(|e| e.to_string())?;
            b.ensure(&self.ctx, t * xq_w * 4)?
        };
        self.ctx.quant_q8_b(x, buf, n_in, xq_w, t)?;
        Ok((buf, xq_w))
    }

    /// 프레임 GEMM 1건 — x는 프레임 f32, 무게는 mmap 참조(업로드 캐시).
    pub(super) fn frame_gemm(&self, x: *mut u8, w: &llm170_core::matmul::Weight<'_>, out: *mut u8, t: usize) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let (wd, f32w) = self.dev_weight(w)?;
        if f32w {
            return self.launch_gemm_f32(x, wd, n_in, n_out, t, out);
        }
        // llama MMQ 경로(부록5: q4_K maxrel 6e-4) — qwen35 raw 디코더가 쓰는
        // 바로 그 mul_mat_q 커널. f32 활성을 직접 양자화하므로 frame_quant를
        // 건너뛴다. 형상은 qwen35와 같은 게이트(t>=32).
        let ty = ggml_id(w.ty);
        if t >= 32
            && matches!(ty, 12 | 13 | 14 | 23)
            && env_on("LLM170_Q4_MMQ")
            && self.ctx.gemm_mmq(ty, x as *const u8, wd, n_in, n_out, t, out).is_ok()
        {
            return Ok(());
        }
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
        let mut xf_p = xf as *mut std::ffi::c_void;  // f32 원활성 (프레임 핸들)
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
        if ty == ggml_id(GgmlType::Q4K)
            && t >= 16
            && (env_on("LLM170_Q4K_MMQ")
                || env_on("LLM170_Q4K_OUTS"))
        {
            // 행-배치 타일(plans/65) — 가중치 디퀀트를 행 루프 밖으로.
            if env_on("LLM170_Q4K_Y") {
                let rpt: usize = std::env::var("LLM170_Q4K_YRPT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(16);
                // 커널이 += 누산이므로 출력을 0으로 초기화한다(출력 버퍼는 매
                // 호출 새로 쓰이는 스크래치라 안전).
                unsafe {
                    std::ptr::write_bytes(out as *mut f32, 0, t * n_out);
                }
                let mut xq_p = xq as *mut std::ffi::c_void;
                let mut w_p = w as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = out as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt, mut rp) =
                    (n_in as i32, n_out as i32, xq_w as i32, t as i32, rpt as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut rp) as *mut _ as *mut std::ffi::c_void,
                ];
                return self.ctx.launch3(
                    "q4_gemm_q4k_y",
                    n_out.div_ceil(256) as u32,
                    t.div_ceil(rpt) as u32,
                    1,
                    256,
                    &mut args,
                );
            }
            // x-스테이징 타일(plans/65) — 출력별 x 재독 제거. 로직·순서는 _m과 동일.
            if env_on("LLM170_Q4K_X") {
                let mut xq_p = xq as *mut std::ffi::c_void;
                let mut w_p = w as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = out as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt) =
                    (n_in as i32, n_out as i32, xq_w as i32, t as i32);
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
                return self.ctx.launch3(
                    "q4_gemm_q4k_x",
                    n_out.div_ceil(16) as u32,
                    t.div_ceil(16) as u32,
                    1,
                    256,
                    &mut args,
                );
            }
            // 형상 스윕용 가변 타일(plans/65) — outs/rows를 env로 지정.
            if let (Ok(outs), Ok(rows)) = (
                std::env::var("LLM170_Q4K_OUTS").map(|v| v.parse::<usize>()),
                std::env::var("LLM170_Q4K_ROWS").map(|v| v.parse::<usize>()),
            ) {
                let (outs, rows) = (outs.unwrap_or(16), rows.unwrap_or(16));
                let mut xq_p = xq as *mut std::ffi::c_void;
                let mut w_p = w as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = out as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt) =
                    (n_in as i32, n_out as i32, xq_w as i32, t as i32);
                let (mut oo, mut rr) = (outs as i32, rows as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut oo) as *mut _ as *mut std::ffi::c_void,
                    (&mut rr) as *mut _ as *mut std::ffi::c_void,
                ];
                return self.ctx.launch3(
                    "q4_gemm_q4k_g",
                    n_out.div_ceil(outs) as u32,
                    t.div_ceil(rows) as u32,
                    1,
                    (outs * rows) as u32,
                    &mut args,
                );
            }
            let mut xq_p = xq as *mut std::ffi::c_void;
            let mut w_p = w as *mut std::ffi::c_void;
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut o_p = out as *mut std::ffi::c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut tt = t as i32;
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
            return self.ctx.launch3(
                "q4_gemm_q4k_m",
                n_out.div_ceil(16) as u32,
                t.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            );
        }
        if ty == ggml_id(GgmlType::Q5_1) {
            // 타일 판은 출력 4개/블록 — 그리드도 4로 나눈다.
            let tiled = true; // plans/78 R6: NO_Q5_1_T 폐기 — 타일 판 확정
            let outs_per_block = if tiled { 4usize } else { 1 };
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
            let mmq = t >= 16
                && !env_on("LLM170_Q5_1_EXACT")
                && ty == ggml_id(GgmlType::Q5_1);
            let kern = match (mmq, tiled) {
                (true, _) => "q4_gemm_q5_1_m",
                (false, true) => "q4_gemm_q5_1_t",
                (false, false) => "q4_gemm_q5_1",
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
            let gx = if tiled { t.div_ceil(16) as u32 } else { t as u32 };
            return self.ctx.launch3(kern, gx, gy, gz, if tiled { 256 } else { 64 }, &mut args);
        }
        // t≥16: MMQ 타일 우선 — 가중치 1회 독서 + 토큰 타일 상각(raw 디코더
        // mm_b와 동일 게이트). 타일 커널이 없는 타입은 GEMV 폴백.
        // 실측(2026-09-14, q4k-bench 2560x6144): 2.5-2.7 TFLOPS로 **t에 걸쳐 평탄**하다
        // (t=128 1.474ms 2.73, t=512 6.401 2.52, t=1024 12.581 2.56, t=2048 27.203
        // 2.37 TFLOPS). 즉 점유율 문제가 아니라 이 형상의 커널 고유 비용이고
        // 가중치 대역도 0.3-6 GB/s뿐이라 연산·대역폭 어느 쪽도 아니다 — llama.cpp
        // 대비 프리필 1.33x가 사는 곳이다. 27B가 같은 계열로 19.5 TFLOPS를 내는 것은
        // n_in/n_out이 더 큰 형상(5120x17408)이라 행당 상각이 크기 때문이다.
        // 같은 형상에서 q4_K MMQ 타일(LLM170_Q4K_MMQ/Y)은 오히려 느렸고(33-34ms),
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
        let pin_tile = crate::rawhip::ctx::PREFILL_PIN.load(std::sync::atomic::Ordering::Relaxed);
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
                if let Err(e) = self
                    .ctx
                    .gemm_tile(xsrc, w, self.ktab2, ty, n_in, n_out, xq_w, tc, osrc)
                {
                    if env_on("LLM170_Q4_DBG") {
                        use std::sync::Mutex;
                        use std::sync::OnceLock;
                        static SEEN: OnceLock<Mutex<Vec<(u32, usize, usize, usize)>>> = OnceLock::new();
                        let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
                        if let Ok(mut v) = seen.lock() {
                            // (ty, n_in, n_out) 별 1회 + t는 128 단위 구간으로 구분.
                            let key = (ty, n_in, n_out, (tc / 128) * 128);
                            if !v.contains(&key) && v.len() < 24 {
                                v.push(key);
                                eprintln!(
                                    "# gemm_tile 폴백: ty={ty} n_in={n_in} n_out={n_out} t={tc} err={e}"
                                );
                            }
                        }
                    }
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
        if t == 1 && n_in.is_multiple_of(4) { // plans/78 R6: F32W 폐기 — f32 직독 기본
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
        self.ctx.launch3("q4_gemm_f32", t as u32, gy, gz, 64, &mut args)
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
        let tt = env_on("LLM170_Q4ACC_TIME");

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
                // 임시 진단: LLM170_F16_DBG=1 이면 호출 직후 동기화해 실패 지점을 명명한다.
                if env_on("LLM170_F16_DBG") {
                    self.ctx.sync().map_err(|e| format!("f16 sync [{n_in}x{n_out}] t={t}: {e}"))?;
                    eprintln!("f16-deq OK [{n_in}x{n_out}] t={t}");
                }
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
            return Err(format!("matmul_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        if ws.is_empty() || xs.is_empty() {
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        let f32_family = |t: GgmlType| matches!(t, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let w_f32 = f32_family(ws[0].ty);
        // 타입 계열·n_in이 섞이면 준비를 공유할 수 없다 — 개별 경로로.
        if ws.iter().any(|w| w.n_in as usize != n_in || f32_family(w.ty) != w_f32) {
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
        let mut off = vec![0usize; ne + 1];
        for &e in expert_ids {
            off[(e as usize).min(ne - 1) + 1] += 1;
        }
        for e in 0..ne {
            off[e + 1] += off[e];
        }
        let mut cur = off[..ne].to_vec();
        let mut perm = vec![0u32; t];
        for (i, &e) in expert_ids.iter().enumerate() {
            let e = (e as usize).min(ne - 1);
            let p = cur[e];
            perm[p] = i as u32;
            cur[e] += 1;
        }
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
        &self, x: u64, wg: &llm170_core::matmul::Weight, wu: &llm170_core::matmul::Weight,
        h: u64, n_in: usize, n_hidden: usize,
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
        self.ctx.launch3("q4_shexp_gu", n_hidden.div_ceil(8) as u32, 1, 1, 256, &mut args)
    }

    fn shexp_da(
        &self, h: u64, wd: &llm170_core::matmul::Weight, s: u64, mout: u64,
        n_in: usize, n_hidden: usize,
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
        self.ctx.launch3("q4_shexp_da", n_in.div_ceil(8) as u32, 1, 1, 256, &mut args)
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
        t: usize,
        eps: f32,
        n_embd: usize,
        hc: usize,
        kern: usize,
        dil: usize,
        hist: usize,
        host_ring: &[f32],
    ) -> Result<(), String> {
        if t != 1 {
            return Err(format!("ple_math_dev: t={t} (디코드 전용)"));
        }
        let hc_dim = hc * n_embd;
        let ring_bytes = hist * hc_dim * 4;
        // 링 풀 + 워터마크(되감기면 호스트 링으로 리프레시).
        let rewind = {
            let mut wm = self.ple_ring_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry(seq).or_insert(0);
            let rw = *w > t; // pos0=0 재시작(벤치 워밍업 등)
            *w = t;          // t=1: 이번 토큰까지 유효
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
            let (mut e, mut ne, mut hcc, mut tt) =
                (eps, n_embd as i32, hc as i32, t as i32);
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
}
