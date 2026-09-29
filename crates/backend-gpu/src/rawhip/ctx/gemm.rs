//! ctx/gemm — GEMV/GEMM·타일·MMQ 디스패치 (ctx.rs 절단, plans/107 W5; 내용 무변경).

use super::*;

/// W4A8 GEMV 커널명 테이블 — gemv_q8 계열 3중 복제 통합(plans/109 P9).
fn gemv_kern(ty: u32) -> Result<&'static str, String> {
    Ok(match ty {
        23 => "gemm_xs",
        13 => "gemm_q5k",
        8 => "gemm_q8_0",
        12 => "gemm_q4k",
        14 => "gemm_q6k",
        20 => "gemm_nl",
        11 => "gemm_q3k",
        21 => "gemm_iq3s",
        _ => return Err(format!("미지원 타입 {ty}")),
    })
}

impl RawCtx {
    /// W4A8 t=1 GEMV — 타입별 커널 선택, 부분합 reduce까지 수행.
    /// 반환 [n_out] f32. 수치: dot_row_w4a8_*_lane 미러와 동일열.
    pub fn gemv_q8(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let out = self.scratch(n_out * 4)?;
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let kern = gemv_kern(ty)?;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut gx_a = 1i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let _ = &mut gx_a;
        let mut out_p0 = out as *mut std::ffi::c_void;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let mut xw_a = crate::rawhip::q4acc::xq_words(n_in) as i32;
        args_v.push(&mut xw_a as *mut _ as *mut std::ffi::c_void);
        self.launch3(kern, 1, gy, gz, 64, &mut args_v)?;
        let mut res = vec![0f32; n_out];
        self.sync()?;
        self.d2h(
            bytemuck::cast_slice_mut(&mut res).as_mut(),
            out as *const u8,
        )?;
        Ok(res)
    }

    /// plans/108 P7 — t=1 q8_0 dmmv: f32 활성 직소비 GEMV(vk gemv8_q8b 이식).
    /// 활성 quant를 건너뛰는 경로 — mm_b2 디스패치와 grp_mmq(dmmv_used)
    /// 게이트가 동일 조건이어야 한다(어긋나면 stale xq를 읽는다).
    pub fn gemv_q8_dmmv_out(
        &self,
        x: *const u8,
        w: *const u8,
        n_in: usize,
        n_out: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let mut x_p = x as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut o_p = out as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut x_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut no) as *mut _ as *mut std::ffi::c_void,
        ];
        let wgs = n_out.div_ceil(2);
        self.launch3(
            "gemm_q8_0_dmmv",
            1,
            wgs.min(65535) as u32,
            wgs.div_ceil(65535) as u32,
            64,
            &mut args,
        )
    }

    pub fn gemv_q8_out(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        out: *mut u8,
        xq_w: usize,
        t: usize,
    ) -> Result<(), String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let gy = n_out.min(65535) as u32;
        let _gz = n_out.div_ceil(65535) as u32;
        let kern = gemv_kern(ty)?;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let q8tr = env_on("LLM170_Q8_TRACE");
        if q8tr {
            eprintln!("# q8tr ty={ty} n_in={n_in} n_out={n_out} t={t}");
        }
        let mut out_p0 = out as *mut std::ffi::c_void;
        let mut xw_a = xq_w as i32;
        let mut tt_a = t as i32;
        // plans/73 (2026-09-16): t=1 q8_0 전 형상을 **16레인×4사분면** 판으로 —
        // 종전 64레인/행은 n_sub=80(qkv/gate)에서 62.5%, n_sub=10(hc up)에서 31%
        // 레인 효율이었고 그만큼 대역폭이 깎였다(GDN mm_group 15.0ms = 106GB/s).
        // 산술은 비트 동일(gemm_q8_0_w4 주석의 트리 재구성). 킬스위치 LLM170_Q8W4=0.
        // 멀티토큰 판(2026-09-16, np 배치): t=2..8 q8_0은 무게 행 1회 독서로
        // 토큰별 내적 — grid=(t,n_out) 배치가 토큰마다 무게를 재독하는 것과
        // 달리 가중치 트래픽이 t배 증가하지 않는다. 산술 비트 동일.
        // plans/74 N2: 소형 n_sub(≤32 — hc up/down=10)는 16레인/행 mt16 판 —
        // mt(64레인)은 이 형상에서 레인 효율 15%(np t=4 182us/호출 실측).
        // 킬스위치 LLM170_Q8MT16=0.
        if (2..=8).contains(&t)
            && ty == 8
            && n_in / 32 <= 32
            && !PREFILL_PIN.load(std::sync::atomic::Ordering::Relaxed)
        {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
                &mut tt_a as *mut _ as *mut std::ffi::c_void,
            ];
            if q8tr {
                eprintln!("# q8tr->mt16 n_in={n_in} n_out={n_out} t={t}");
            }
            return self.launch3(
                "gemm_q8_0_mt16",
                n_out.div_ceil(8) as u32,
                1,
                1,
                128,
                &mut args,
            );
        }
        // plans/74 (2026-09-16): 멀티토큰 q8_0 은 **워프=행 판**(gemm_q8_0_mt_w)이
        // 기본 — n_sub>32 에서 종전 64레인 판 대비 +40..65% 실측(마이크로벤치
        // 117→190GB/s @ n_sub=80, 119→180 @ n_sub=64). 산술은 정수 사슬 분리
        // (결합법칙 — 값 불변) + 레인별 f32 사슬/ f64 32레인 트리(mt16 계열과
        // 동일 정밀도 클래스). 킬스위치 LLM170_Q8MTW=0.
        if (2..=8).contains(&t)
            && ty == 8
            && n_in / 32 > 32
            && !PREFILL_PIN.load(std::sync::atomic::Ordering::Relaxed)
        {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
                &mut tt_a as *mut _ as *mut std::ffi::c_void,
            ];
            if q8tr {
                eprintln!("# q8tr->mt_w n_in={n_in} n_out={n_out} t={t}");
            }
            return self.launch3(
                "gemm_q8_0_mt_w",
                1,
                n_out.min(65535) as u32,
                n_out.div_ceil(65535) as u32,
                32,
                &mut args,
            );
        }
        if (2..=8).contains(&t)
            && ty == 8
            && !PREFILL_PIN.load(std::sync::atomic::Ordering::Relaxed)
        {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
                &mut tt_a as *mut _ as *mut std::ffi::c_void,
            ];
            if q8tr {
                eprintln!("# q8tr->mt64 n_in={n_in} n_out={n_out} t={t}");
            }
            return self.launch3(
                "gemm_q8_0_mt",
                1,
                n_out.min(65535) as u32,
                n_out.div_ceil(65535) as u32,
                64,
                &mut args,
            );
        }
        // 소형 n_sub(≤32) 구간은 w16(16레인/행, 레인 효율 62.5-100% vs 워프판
        // 31%)으로 — FN tg128 17.2 → 18.0 t/s (+4.8%, 2026-09-16 실측, 게이트 동일).
        // 킬스위치 LLM170_Q8W16_SMALL=0.
        if t == 1 && ty == 8 && n_in / 32 <= 32 {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
            ];
            return self.launch3(
                "gemm_q8_0_w16",
                n_out.div_ceil(8) as u32,
                t as u32,
                1,
                128,
                &mut args,
            );
        }
        // plans/83 D2: 저출력 GEMV(hc down 등 n_out ≤ 2048, n_sub > 32)는
        // 워프판이 유리 — FN tg32 18.10 → 18.25 (+0.8%). 단 축소 순서가
        // 달라 27B 게이트 타이를 뒤집는다(토큰5 실측) — 전역 디스패처라 모델
        // 구분이 없어 옵트인으로만 둔다. 기본 적용은 형상 스코프 분리 후.
        // plans/84 E1: 모델 스코프 분리 — Flash-Next 기본 적용(FN tg +0.8%,
        // 게이트 통과), qwen35는 옵트인(타이 플립 방지). 킬스위치 =0.
        if t == 1 && ty == 8 && n_out <= 2048 && n_in / 32 > 32 && self.scope_is_flashnext() {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
            ];
            return self.launch3(
                "gemm_q8_0_w",
                n_out.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            );
        }
        let gz2 = n_out.div_ceil(65535) as u32;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let xw_ptr = &mut xw_a as *mut _ as *mut std::ffi::c_void;
        args_v.push(xw_ptr);
        if q8tr {
            eprintln!("# q8tr->fallback {kern} n_in={n_in} n_out={n_out} t={t}");
        }
        self.launch3(kern, t as u32, gy, gz2, 64, &mut args_v)?;
        Ok(())
    }

    /// np 소형 배치(t=2..4) 4-토큰 GEMV — 가중 1회 독서.
    /// y는 [t][xq_w], out은 [t][n_out] (토큰별 독립 누산·환원).
    pub fn gemm_g4(
        &self,
        ty: u32,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        // plans/74 N3: q5_K 은 워프=행(블록 32, 스트라이드 32레인)이 기본.
        // 4워프/블록 판은 -3% 측정(부정, 2026-09-17). LLM170_NO_Q5K4W2=1 원판.
        // ILP-2(서브블록 2개/반복, 워드 선적재)는 -1.5% 측정(부정, 2026-09-17).
        let w2 = ty == 13;
        // q4_K 도 워프=행 기본(인터리브 3회: 30.6/30.0 vs 31.0/31.1/32.2).
        let w2q4 = ty == 12;
        let kern = match ty {
            12 => {
                if w2q4 {
                    "gemm_q4k4_w2"
                } else {
                    "gemm_q4k4"
                }
            }
            13 => {
                if w2 {
                    "gemm_q5k4_w2"
                } else {
                    "gemm_q5k4"
                }
            }
            14 => "gemm_q6k4",
            23 => "gemm_xs4",
            _ => return Err(format!("g4 미지원 타입 {ty}")),
        };
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut kt = ktab2 as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
        ];
        if ty == 23 {
            args.push(&mut kt as *mut _ as *mut std::ffi::c_void);
        }
        args.push(&mut op as *mut _ as *mut std::ffi::c_void);
        args.push(&mut ni as *mut _ as *mut std::ffi::c_void);
        args.push(&mut no as *mut _ as *mut std::ffi::c_void);
        args.push(&mut xw as *mut _ as *mut std::ffi::c_void);
        args.push(&mut tt as *mut _ as *mut std::ffi::c_void);
        let blk: u32 = if w2 || w2q4 { 32 } else { 64 };
        self.launch3(kern, 1, gy, gz, blk, &mut args)
    }

    /// AR 청크 버퍼 조기 확보 (DecodeState init에서 호출 — 조각화 회피).
    pub fn ar_chunk_prealloc(
        &self,
        npair: usize,
        d: usize,
        nc_max: usize,
        ch: usize,
    ) -> Result<(), String> {
        self.ar_chunk_bufs(npair, d, nc_max, ch).map(|_| ())
    }

    /// AR 청크 스캔 버퍼 (lend/sstart/pgb/pbuf) — b_t_max 기준 1회 할당.
    pub fn ar_chunk_bufs(
        &self,
        npair: usize,
        d: usize,
        nc_max: usize,
        ch: usize,
    ) -> Result<(*mut u8, *mut u8, *mut u8, *mut u8), String> {
        let mut g = self.ar_cache.lock().map_err(|e| e.to_string())?;
        if let Some(b) = *g {
            return Ok(b);
        }
        let lend = self.alloc(npair * d * d * nc_max)?;
        let sstart = self.alloc(npair * d * d * nc_max)?;
        let pgb = self.alloc(npair * d * nc_max)?;
        let pbuf = self.alloc(npair * ch * d * nc_max)?;
        let b = (lend, sstart, pgb, pbuf);
        *g = Some(b);
        Ok(b)
    }

    /// 사이드 스트림판 — 호출자가 side_wait_main 후 발사/ join2로 합류.
    pub fn gemv_q8_out_s(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        out: *mut u8,
        xq_w: usize,
        t: usize,
    ) -> Result<(), String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let gy = n_out.min(65535) as u32;
        let _gz = n_out.div_ceil(65535) as u32;
        let kern = gemv_kern(ty)?;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let gz = n_out.div_ceil(65535) as u32;
        let mut out_p0 = out as *mut std::ffi::c_void;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let mut xw_a = xq_w as i32;
        let xw_ptr = &mut xw_a as *mut _ as *mut std::ffi::c_void;
        args_v.push(xw_ptr);
        self.launch3s(kern, t as u32, gy, gz, 64, &mut args_v)?;
        Ok(())
    }

    fn tile_core(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<TileLaunch, String> {
        // plans/84 E.2: 프리필 핀 중 large-t 패밀리(j128 강제 + big — odd(v4)는
        // big만 본다). 진입점: hc down(q8_0)의 t 키 GEMV/j128 갈림.
        if !env_on("LLM170_EXACT") && PREFILL_PIN.load(std::sync::atomic::Ordering::Relaxed) {
            let j128 = self.co_loaded(CO_J128);
            return self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128, true);
        }
        let j128 = !env_on("LLM170_EXACT") && self.co_loaded(CO_J128) && t > 64;
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128, false)
    }

    /// head 강제판 — j128 타일을 t≤64에서도 (n_out 초대형일 때 이득).
    fn tile_core_head(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<TileLaunch, String> {
        let j128 = !env_on("LLM170_EXACT") && self.co_loaded(CO_J128);
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128, false)
    }

    /// 프리필 핀판 (plans/84 A) — j128 강제 + large-t 패밀리(wm/v4) 고정.
    /// 청크 불변성: 같은 텐서는 t에 무관하게 항상 동일 커널 산술을 쓴다.
    fn tile_core_pin(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<TileLaunch, String> {
        let j128 = !env_on("LLM170_EXACT") && self.co_loaded(CO_J128);
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128, true)
    }

    fn tile_core_inner(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
        j128: bool,
        large_t: bool,
    ) -> Result<TileLaunch, String> {
        // wm·mm 상한 64: t>64 무CO는 유효 커널 없음 — 침묵 오답 대신 에러
        // (핀판은 j128 강제 — large-t 패밀리가 곧 j128/v4이므로 무CO면 에러가 정당)
        if t > 64 && !j128 {
            return Err(format!("타일 미지원: t={t}는 CO 사전컴파일(j128/v4) 필요"));
        }
        let big = t >= 32 || large_t;
        let (v4, odd) = (self.co_loaded(CO_V4), self.co_loaded(CO_ODD));
        let kern: &'static str = match ty {
            13 => {
                if j128 && v4 {
                    "gemm_q5k_v4"
                } else if j128 {
                    "gemm_q5k_j128"
                } else if !env_on("LLM170_EXACT") && big {
                    "gemm_q5k_wm"
                } else {
                    "gemm_q5k_mm"
                }
            }
            12 => {
                if j128 && v4 {
                    "gemm_q4k_v4"
                } else if j128 {
                    "gemm_q4k_j128"
                } else if !env_on("LLM170_EXACT") && big {
                    "gemm_q4k_wm"
                } else {
                    "gemm_q4k_mm"
                }
            }
            14 => {
                if j128 {
                    "gemm_q6k_j128"
                } else if !env_on("LLM170_EXACT") && big {
                    "gemm_q6k_wm"
                } else {
                    "gemm_q6k_mm"
                }
            }
            // 107 P0-7: gemm_xs_v4u 분기 삭제 — NAMES·CO 어디에도 미등록,
            // LLM170_XS_V4U=1 도달 시 런치 즉실패하는 죽은 분기였음.
            23 => {
                if j128 {
                    "gemm_xs_j128"
                } else if v4 && !env_on("LLM170_EXACT") && big {
                    "gemm_xs_v4"
                } else if !env_on("LLM170_EXACT") && big {
                    "gemm_xs_wm"
                } else {
                    "gemm_xs_mm"
                }
            }
            20 => {
                if odd && !env_on("LLM170_EXACT") && big {
                    "gemm_nl_v4"
                } else {
                    return Err("타일 미지원 타입 20 (GEMV 경로 사용)".into());
                }
            }
            11 => {
                if odd && !env_on("LLM170_EXACT") && big {
                    "gemm_q3k_v4"
                } else {
                    return Err("타일 미지원 타입 11 (GEMV 경로 사용)".into());
                }
            }
            21 => {
                if odd && !env_on("LLM170_EXACT") && big {
                    "gemm_iq3s_v4"
                } else {
                    return Err("타일 미지원 타입 21 (GEMV 경로 사용)".into());
                }
            }
            8 => {
                if j128 {
                    "gemm_q8_j128"
                } else {
                    return Err("타일 미지원 타입 8 (GEMV 경로 사용)".into());
                }
            }
            _ => return Err(format!("타일 미지원 타입 {ty}")),
        };
        if env_on("LLM170_TILE_SHAPES") {
            use std::sync::Mutex;
            use std::sync::OnceLock;
            static SEEN: OnceLock<Mutex<Vec<(String, usize, usize, usize)>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
            if let Ok(mut v) = seen.lock() {
                let key = (kern.to_string(), n_in, n_out, (t / 128) * 128);
                if !v.contains(&key) {
                    v.push(key.clone());
                    eprintln!("# tile-shape {kern} n_in={n_in} n_out={n_out} t~{t}");
                }
            }
        }
        let mm = kern.ends_with("_mm")
            || kern.ends_with("_wm")
            || kern.ends_with("_j128")
            || kern.ends_with("_v4");
        let rows_per_block: usize = if kern.ends_with("_j128") || kern.ends_with("_v4") {
            128
        } else if mm {
            64
        } else {
            1
        };
        let nblocks = n_out.div_ceil(rows_per_block);
        Ok(TileLaunch {
            kern,
            xp: xq as *mut std::ffi::c_void,
            wp: w as *mut std::ffi::c_void,
            op: out as *mut std::ffi::c_void,
            ktp: ktab2 as *mut std::ffi::c_void,
            ni: n_in as i32,
            no: n_out as i32,
            xw: xq_w as i32,
            // z-그리드 토큰 사분면: t≤128이면 gz=1 (무변형). 초과분은 128씩.
            // j128/v4 판은 토큰 사분면(blockIdx.z) 지원 — t를 그대로 넘긴다(§51 수정판).
            tt: if kern.ends_with("_j128") || kern.ends_with("_v4") {
                t as i32
            } else {
                t.min(128) as i32
            },
            gx: nblocks.min(65535) as u32,
            gz: (nblocks.div_ceil(65535) * t.div_ceil(128)) as u32,
            block: if mm { 256 } else { 64 },
            ktab: ty == 23 || kern == "gemm_nl_v4",
        })
    }

    fn tile_args(l: &mut TileLaunch) -> Vec<*mut std::ffi::c_void> {
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut l.xp) as *mut _ as *mut std::ffi::c_void,
            (&mut l.wp) as *mut _ as *mut std::ffi::c_void,
            (&mut l.op) as *mut _ as *mut std::ffi::c_void,
        ];
        if l.ktab {
            args.push((&mut l.ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut l.ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.tt) as *mut _ as *mut std::ffi::c_void);
        args
    }

    /// spec verify head 전용 — j128/v4 타일 강제 (가중 1회 독서).
    /// 산술은 동일 W4A8이나 환원 순서가 mm 계열와 달라 스트림 비트계약 대상 아님
    /// (spec 내부 draft↔verify 일관성만 요구).
    pub fn gemm_tile_head(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let mut l = self.tile_core_head(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    pub fn gemm_tile(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let mut l = self.tile_core(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        let r = self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args);
        if env_on("LLM170_TILE_PROF") {
            self.sync().ok();
            let ti = std::time::Instant::now();
            self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args).ok();
            self.sync().ok();
            eprintln!(
                "tileprof ty={ty} {n_in}x{n_out} t={t} kern={} {:.3}ms",
                l.kern,
                ti.elapsed().as_secs_f64() * 1e3
            );
        }
        r
    }

    /// gemm_tile의 프리필 핀판 — large-t 패밀리 고정 (plans/84 A, 청크 불변성).
    pub fn gemm_tile_pin(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let mut l = self.tile_core_pin(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    /// gemm_tile_s의 프리필 핀판 — large-t 패밀리 고정·사이드 스트림 (plans/84 A).
    pub fn gemm_tile_pin_s(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let mut l = self.tile_core_pin(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3s(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    /// 커널 속성 조회 — (레지스터, 로컬 바이트, 최대 스레드). 점유율 진단용.
    pub fn kern_attrs(&self, name: &str) -> Option<(i32, usize, i32)> {
        let f = *self.fns.get(name)?;
        let mut a: hip::hipFuncAttributes = unsafe { std::mem::zeroed() };
        let r = unsafe { hip::hipFuncGetAttributes(&mut a, f as *const std::ffi::c_void) };
        if r == hip::hipError_t_hipSuccess {
            Some((a.numRegs, a.localSizeBytes, a.maxThreadsPerBlock))
        } else {
            None
        }
    }

    /// gemm_tile의 사이드 스트림판 — 커널 선택·인자 구성은 공용 코어에 위임.
    pub fn gemm_tile_s(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let mut l = self.tile_core(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3s(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    /// q6_K → f16 전개 + gemm_f16_v4 (deq-f16 경로, 부록42).
    pub fn gemm_f16_q6(
        &self,
        y_f32: *const u8,
        w: *const u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let fns = &self.fns;
        let fq = *fns.get("dequant_q6k_f16").ok_or("dequant_q6k_f16 없음")?;
        let fm = *fns.get("gemm_f16_v4").ok_or("gemm_f16_v4 없음")?;
        // f16 전개 버퍼 (지속: w 주소 키 캐시)
        let key = w as usize;
        let wf16 = {
            let mut c = self.f16_cache.lock().map_err(|e| e.to_string())?;
            if let Some(&p) = c.get(&key) {
                p
            } else {
                let blocks = n_in.div_ceil(256); // 256요소 슈퍼블록 격자(부분 허용)
                let p = self.alloc(n_out * n_in * 2)?;
                unsafe {
                    let mut a1 = w as *mut std::ffi::c_void;
                    let mut a2 = p as *mut std::ffi::c_void;
                    let mut a3 = blocks as i32;
                    let mut a4 = n_out as i32;
                    let mut args = vec![
                        &mut a1 as *mut _ as *mut _,
                        &mut a2 as *mut _ as *mut _,
                        &mut a3 as *mut _ as *mut _,
                        &mut a4 as *mut _ as *mut _,
                    ];
                    ck(
                        hip::hipModuleLaunchKernel(
                            fq,
                            n_out as u32,
                            blocks as u32,
                            1,
                            256,
                            1,
                            1,
                            0,
                            self.stream,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        "dequant_q6k_f16",
                    )?;
                }
                c.insert(key, p);
                p
            }
        };
        // y: f32 → **llama q8_1**(mmq_quant_y, 144B/128원소) — v4 GEMM은 llama
        // 계열이라 이 레이아웃을 기대한다. 우리 quant_q8(1.375B/원소)을 넣으면
        // 레이아웃이 어긋나 쓰레기 토큰이 나온다(plans/65 §13-14 실측).
        let xq_w = (n_in / 128) * 36;
        let tr = t.div_ceil(128) * 128; // 커널의 128 단위 사분면 경계 (범위 밖 쓰기 방지)
        let mut xq = self.mmq_y2.lock().map_err(|e| e.to_string())?;
        let xq_p = if xq.0 < xq_w * tr {
            let p = self.alloc(xq_w * tr * 4)?;
            *xq = (xq_w * tr, p);
            p
        } else {
            xq.1
        };
        unsafe {
            let mut a1 = y_f32 as *mut std::ffi::c_void;
            let mut a2 = xq_p as *mut std::ffi::c_void;
            let mut a3 = n_in as i32;
            let mut a4 = xq_w as i32;
            let mut a5 = t as i32;
            let mut args = vec![
                &mut a1 as *mut _ as *mut _,
                &mut a2 as *mut _ as *mut _,
                &mut a3 as *mut _ as *mut _,
                &mut a4 as *mut _ as *mut _,
                &mut a5 as *mut _ as *mut _,
            ];
            // quant_q8_b: grid(nblk/64, t) block 64 — kernels.rs quant_q8 시그니처 (x, xq, n, xq_w)
            let fq8 = *fns.get("mmq_quant_y").ok_or("mmq_quant_y 없음")?;
            ck(
                hip::hipModuleLaunchKernel(
                    fq8,
                    (n_in / 128) as u32,
                    t as u32,
                    1,
                    32,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "mmq_quant_y",
            )?;
            let mut b2 = wf16 as *mut std::ffi::c_void;
            let mut b4 = n_in as i32;
            let mut b5 = n_out as i32;
            let mut b6 = xq_w as i32;
            // z-그리드 사분면 CO: 단일 런치 (tt=min(t,128), gz=사분면)
            {
                let mut z1 = xq_p as *mut std::ffi::c_void;
                let mut z3 = out as *mut std::ffi::c_void;
                let mut z7 = t.min(128) as i32;
                let mut az: Vec<*mut std::ffi::c_void> = vec![
                    &mut z1 as *mut _ as *mut _,
                    &mut b2 as *mut _ as *mut _,
                    &mut z3 as *mut _ as *mut _,
                    &mut b4 as *mut _ as *mut _,
                    &mut b5 as *mut _ as *mut _,
                    &mut b6 as *mut _ as *mut _,
                    &mut z7 as *mut _ as *mut _,
                ];
                ck(
                    hip::hipModuleLaunchKernel(
                        fm,
                        n_out.div_ceil(128) as u32,
                        1,
                        (tr / 128) as u32,
                        256,
                        1,
                        1,
                        0,
                        self.stream,
                        az.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "gemm_f16_v4",
                )?;
            }
        }
        Ok(())
    }
    pub fn gemm_f16_deq(
        &self,
        ty: u32,
        y_f32: *const u8,
        w: *const u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let fns = &self.fns;
        // f16 전개 커널 선택 — 우리 .co의 GEMM이 소비하는 레이아웃으로 전개한다.
        let fq = match ty {
            14 => *fns.get("dequant_q6k_f16").ok_or("dequant_q6k_f16 없음")?,
            12 => *fns.get("dequant_q4k_f16").ok_or("dequant_q4k_f16 없음")?,
            8 => *fns.get("dequant_q8_0_f16").ok_or("dequant_q8_0_f16 없음")?,
            _ => return Err(format!("f16 경로 미지원 타입 {ty}")),
        };
        let fm = *fns.get("gemm_f16_v4").ok_or("gemm_f16_v4 없음")?;
        // f16 전개 버퍼 (지속: w 주소 키 캐시)
        let key = (w as usize) ^ ((ty as usize) << 60);
        // 크기 가드: 전문가 스택(수십 GB)은 f16 캐시 불가 → 호출자가 거른다.
        if (n_out as u64) * (n_in as u64) * 2 > 512 * 1024 * 1024 {
            return Err("f16 캐시 상한 초과".into());
        }
        let wf16 = {
            let mut c = self.f16_cache.lock().map_err(|e| e.to_string())?;
            if let Some(&p) = c.get(&key) {
                p
            } else {
                let _blocks = n_in / 256;
                let p = self.alloc(n_out * n_in * 2)?;
                unsafe {
                    let mut a1 = w as *mut std::ffi::c_void;
                    let mut a2 = p as *mut std::ffi::c_void;
                    let mut a3 = (n_in / 32) as i32; // 행당 32블록 수 = 행 스트라이드
                    let mut a4 = n_out as i32;
                    let mut a5 = n_in as i32; // 유효 요소 수(부분 블록 가드)
                    let mut args = vec![
                        &mut a1 as *mut _ as *mut _,
                        &mut a2 as *mut _ as *mut _,
                        &mut a3 as *mut _ as *mut _,
                        &mut a4 as *mut _ as *mut _,
                        &mut a5 as *mut _ as *mut _,
                    ];
                    ck(
                        hip::hipModuleLaunchKernel(
                            fq,
                            n_out as u32,
                            n_in.div_ceil(256) as u32,
                            1,
                            256,
                            1,
                            1,
                            0,
                            self.stream,
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        "dequant_f16",
                    )?;
                }
                c.insert(key, p);
                p
            }
        };
        // y: f32 → 우리 xq (quant_q8) — y_f32 에서 직접
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        // 커널은 t를 128 단위 사분면으로 소비하고 드레인도 그 경계까지 쓴다 →
        // 부분 t에서 범위 밖 쓰기가 생긴다. 런치 t를 128 배수로 올려 in-bounds로 만든다
        // (행 < t 만 유효, 호출자가 그만큼만 읽는다).
        let tr = t.div_ceil(128) * 128;
        let mut xq = self.mmq_y2.lock().map_err(|e| e.to_string())?;
        let xq_p = if xq.0 < xq_w * tr {
            let p = self.alloc(xq_w * tr * 4)?;
            *xq = (xq_w * tr, p);
            p
        } else {
            xq.1
        };
        unsafe {
            let mut a1 = y_f32 as *mut std::ffi::c_void;
            let mut a2 = xq_p as *mut std::ffi::c_void;
            let mut a3 = n_in as i32;
            let mut a4 = xq_w as i32;
            let mut a5 = t as i32;
            let mut args = vec![
                &mut a1 as *mut _ as *mut _,
                &mut a2 as *mut _ as *mut _,
                &mut a3 as *mut _ as *mut _,
                &mut a4 as *mut _ as *mut _,
                &mut a5 as *mut _ as *mut _,
            ];
            // quant_q8_b: grid(nblk/64, t) block 64 — kernels.rs quant_q8 시그니처 (x, xq, n, xq_w)
            let fq8 = *fns.get("quant_q8").ok_or("quant_q8 없음")?;
            ck(
                hip::hipModuleLaunchKernel(
                    fq8,
                    ((n_in / 32).div_ceil(64)) as u32,
                    t as u32,
                    1,
                    64,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "quant_q8",
            )?;
            let mut b2 = wf16 as *mut std::ffi::c_void;
            let mut b4 = n_in as i32;
            let mut b5 = n_out as i32;
            let mut b6 = xq_w as i32;
            // z-그리드 사분면 CO: 단일 런치 (tt=min(t,128), gz=사분면)
            {
                let mut z1 = xq_p as *mut std::ffi::c_void;
                let mut z3 = out as *mut std::ffi::c_void;
                let mut z7 = t.min(128) as i32;
                let mut az: Vec<*mut std::ffi::c_void> = vec![
                    &mut z1 as *mut _ as *mut _,
                    &mut b2 as *mut _ as *mut _,
                    &mut z3 as *mut _ as *mut _,
                    &mut b4 as *mut _ as *mut _,
                    &mut b5 as *mut _ as *mut _,
                    &mut b6 as *mut _ as *mut _,
                    &mut z7 as *mut _ as *mut _,
                ];
                ck(
                    hip::hipModuleLaunchKernel(
                        fm,
                        n_out.div_ceil(128) as u32,
                        1,
                        (tr / 128) as u32,
                        256,
                        1,
                        1,
                        0,
                        self.stream,
                        az.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "gemm_f16_v4",
                )?;
            }
        }
        Ok(())
    }

    /// llama MMQ (mul_mat_q<q4_K/q5_K,128>) — f32 활성 직양자화 + 원형 런치.
    /// 하니스 검증: q4_K maxrel 6e-4, q5_K maxrel 1.5e-3 (plans/27 부록5·14).
    fn gemm_mmq_impl(
        &self,
        ty: u32,
        y_f32: *const u8,
        w: *const u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
        stream: hip::hipStream_t,
        side: bool,
    ) -> Result<(), String> {
        let fns = &self.fns;
        // D4 타입(q6_K/iq4_xs)은 f32-d 전용 양자화 (mmq.cuh ds_layout 계약)
        // DS 레이아웃(mmq.cuh): Q6K/IQ4XS/Q8_0 → D4, Q4K/Q5K → DS4.
        // Q8_0(8)도 D4라 기존 quant_y_d4와 포맷 공유를 기대(plans/71 실험).
        let fq = *fns
            .get(if matches!(ty, 8 | 14 | 23) {
                "mmq_quant_y_d4"
            } else {
                "mmq_quant_y"
            })
            .ok_or("mmq quant 없음")?;
        let j: usize = 128;
        let sym = match ty {
            12 => {
                let js = if j == 64 { "64" } else { "128" };
                format!(
                    "_ZL9mul_mat_qIL9ggml_type12ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                    js
                )
            }
            13 => {
                let js = if j == 64 { "64" } else { "128" };
                format!(
                    "_ZL9mul_mat_qIL9ggml_type13ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                    js
                )
            }
            14 => {
                let js = if j == 64 { "64" } else { "128" };
                format!(
                    "_ZL9mul_mat_qIL9ggml_type14ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                    js
                )
            }
            23 => {
                let js = if j == 64 { "64" } else { "128" };
                format!(
                    "_ZL9mul_mat_qIL9ggml_type23ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                    js
                )
            }
            8 => {
                let js = if j == 64 { "64" } else { "128" };
                format!(
                    "_ZL9mul_mat_qIL9ggml_type8ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                    js
                )
            }
            _ => return Err(format!("MMQ 미지원 타입 {ty}")),
        };
        let fm = *fns.get(&sym[..]).ok_or("mul_mat_q 없음")?;
        // q6_K는 GGUF(=ggml 정준) 레이아웃을 그대로 쓴다. 과거의
        // requant_q6k_canonical(d-first 재배열)은 정준 입력을 깨뜨려 쓰레기
        // 토큰을 냈다(2026-09-12 실측) — 107 W4로 레거시 분기 삭제.
        let w_eff = w as *mut u8;
        // 전용 y 버퍼 — scratch 풀은 동일 크기 호출에 같은 포인터 반환(비동기
        // 재작성 위험). MMQ y는 단일 소유로 격리.
        let yb = {
            // 마지막 128행 타일은 t를 넘어 읽는다 — llama.cpp도 y 버퍼에
            // J_max*sizeof(block_q8_1_mmq) 슬랙을 둔다(mmq.cu nbytes_src1_q8_1).
            // 슬랙이 없으면 t가 128의 배수가 아닐 때(예: 검증 배치 t=33) OOB read.
            const MMQ_Y_SLACK: usize = 128 * 144;
            let need = (n_in / 128) * t * 144 + MMQ_Y_SLACK;
            let mut sc = if side {
                self.mmq_y_s.lock().map_err(|e| e.to_string())?
            } else {
                self.mmq_y.lock().map_err(|e| e.to_string())?
            };
            if sc.0 < need {
                if !sc.1.is_null() {
                    unsafe { hip::hipFree(sc.1 as *mut _) };
                }
                sc.1 = self.alloc(need)?;
                sc.0 = need;
            }
            sc.1
        };
        let mut yp = yb as *mut std::ffi::c_void;
        let mut ysrc = y_f32 as *const std::ffi::c_void;
        let mut nt = t as i32;
        let mut ni_a = n_in as i32;
        {
            if ty == 8 {
                // plans/71: Q8_0의 y양자화는 신형 quantize_mmq_q8_1<D4,false>
                // (ROCm 10 빌드) — 구형 mmq_quant_y*는 block_q8_1_mmq ABI가 달라
                // 혼합 시 HIP 700. 인자: (x, ids=null, vy, ne00, s01, s02, s03,
                // ne0, ne1, ne2, n_expert_used) 그리드 (t, ceil(n_in/512), 1) 128.
                let fq2 = *self
                    .fns
                    .get("_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvllllliii")
                    .ok_or("quantize_mmq_q8_1<D4> 없음")?;
                let mut xp2 = y_f32 as *mut std::ffi::c_void;
                let mut idsp: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut ne00 = n_in as i64;
                let mut s01 = n_in as i64;
                let mut s02 = 0i64;
                let mut s03 = 0i64;
                let mut ne0 = n_in as i64;
                let mut ne1 = t as i32;
                let mut ne2 = 1i32;
                let mut neu = 0i32;
                let mut q2 = vec![
                    &mut xp2 as *mut _ as *mut std::ffi::c_void,
                    &mut idsp as *mut _ as *mut std::ffi::c_void,
                    &mut yp as *mut _ as *mut std::ffi::c_void,
                    &mut ne00 as *mut _ as *mut std::ffi::c_void,
                    &mut s01 as *mut _ as *mut std::ffi::c_void,
                    &mut s02 as *mut _ as *mut std::ffi::c_void,
                    &mut s03 as *mut _ as *mut std::ffi::c_void,
                    &mut ne0 as *mut _ as *mut std::ffi::c_void,
                    &mut ne1 as *mut _ as *mut std::ffi::c_void,
                    &mut ne2 as *mut _ as *mut std::ffi::c_void,
                    &mut neu as *mut _ as *mut std::ffi::c_void,
                ];
                // ne0는 128 배수여야 함(assert 위) — n_in이 128 미만 배수면
                // 상위 경로에서 이미 128 정렬(27B/Flash 폭은 전부 128배수).
                unsafe {
                    let gy = n_in.div_ceil(512) as u32;
                    ck(
                        hip::hipModuleLaunchKernel(
                            fq2,
                            t as u32,
                            gy,
                            1,
                            128,
                            1,
                            1,
                            0,
                            stream,
                            q2.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        "quantize_mmq_q8_1",
                    )?;
                }
            } else {
                unsafe {
                    let mut qargs = vec![
                        &mut ysrc as *mut _ as *mut std::ffi::c_void,
                        &mut yp as *mut _ as *mut std::ffi::c_void,
                        &mut nt as *mut _ as *mut std::ffi::c_void,
                        &mut ni_a as *mut _ as *mut std::ffi::c_void,
                    ];
                    self.ktr_mark("mmq_quant_y", t as u32);
                    ck(
                        hip::hipModuleLaunchKernel(
                            fq,
                            (n_in / 128) as u32,
                            t as u32,
                            1,
                            32,
                            1,
                            1,
                            0,
                            stream,
                            qargs.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        "mmq_quant_y",
                    )?;
                    self.ktr_mark("mmq_quant_y", t as u32);
                }
            }
        }
        fn fd3(d: u32) -> [u32; 3] {
            let mut l = 0u32;
            while l < 32 && (1u32 << l) < d {
                l += 1;
            }
            let mp = ((((1u64) << 32) * (((1u64) << l) - d as u64)) / d as u64 + 1) as u32;
            [mp, l, d]
        }
        let j: usize = 128;
        // 블록 원소수(qk): K계열 256, Q8_0은 32 — launcher의 ncols_x/qk 계약.
        // n_in/256 하드코딩은 Q8_0에서 8배 작아 인덱싱 붕괴(가비지)였다(plans/71).
        let qk: usize = if ty == 8 { 32 } else { 256 };
        let nbk = (n_in / qk) as u32;
        let mut bpn = fd3(nbk);
        let mut one = fd3(1);
        let j_now: usize = 128;
        let mut ntx_fd = fd3(t.div_ceil(j_now) as u32);
        let z3: [u32; 3] = [0, 0, 0];
        let mut ax = w_eff as *mut std::ffi::c_void;
        let mut ay = yb as *mut std::ffi::c_void;
        let mut aid: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut aeb: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut adst = out as *mut std::ffi::c_void;
        let mut afx: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut ays: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut p_nrows = n_out as i32;
        let mut p_ncolsdst = t as i32;
        let mut p_srow = (n_in / qk) as i32;
        let mut p_ncolsy = t as i32;
        let mut p_scol = n_out as i32;
        let smem: i32 = (j * 4 + 128 * 76 * 4 + (j * 144).div_ceil(1024) * 1024) as i32;
        unsafe {
            ck(
                hip::hipFuncSetAttribute(
                    fm as *const _,
                    hip::hipFuncAttribute_hipFuncAttributeMaxDynamicSharedMemorySize,
                    smem,
                ),
                "mmq smem attr",
            )?;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut ax as *mut _ as *mut _,
                &mut ay as *mut _ as *mut _,
                &mut aid as *mut _ as *mut _,
                &mut aeb as *mut _ as *mut _,
                &mut adst as *mut _ as *mut _,
                &mut afx as *mut _ as *mut _,
                &mut ays as *mut _ as *mut _,
                bpn.as_mut_ptr() as *mut _,
                &mut p_nrows as *mut _ as *mut _,
                &mut p_ncolsdst as *mut _ as *mut _,
                &mut p_srow as *mut _ as *mut _,
                &mut p_ncolsy as *mut _ as *mut _,
                &mut p_scol as *mut _ as *mut _,
                one.as_mut_ptr() as *mut _,
                one.as_mut_ptr() as *mut _,
                z3.as_ptr() as *mut _,
                z3.as_ptr() as *mut _,
                z3.as_ptr() as *mut _,
                one.as_mut_ptr() as *mut _,
                one.as_mut_ptr() as *mut _,
                z3.as_ptr() as *mut _,
                z3.as_ptr() as *mut _,
                z3.as_ptr() as *mut _,
                ntx_fd.as_mut_ptr() as *mut _,
            ];
            let tag: &'static str = match ty {
                12 => "mmq_q4k",
                13 => "mmq_q5k",
                14 => "mmq_q6k",
                23 => "mmq_xs",
                _ => "mmq_other",
            };
            self.ktr_mark(tag, t as u32);
            ck(
                hip::hipModuleLaunchKernel(
                    fm,
                    n_out.div_ceil(128) as u32,
                    t.div_ceil(128) as u32,
                    1,
                    32,
                    8,
                    1,
                    smem as u32,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "mul_mat_q",
            )?;
            self.ktr_mark(tag, t as u32);
            if env_on("LLM170_MMQ_ARGS") {
                eprintln!(
                    "mmq_args ty={ty} n_in={n_in} n_out={n_out} t={t} grid=({},{},1) blk=(32,8) smem={smem} srow={} scol={} nrows={}",
                    n_out.div_ceil(128),
                    t.div_ceil(128),
                    n_in / 256,
                    n_out,
                    n_out
                );
            }
        }
        Ok(())
    }

    /// MMQ — 메인 스트림판.
    pub fn gemm_mmq(
        &self,
        ty: u32,
        y_f32: *const u8,
        w: *const u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        self.gemm_mmq_impl(ty, y_f32, w, n_in, n_out, t, out, self.stream, false)
    }

    /// MMQ — 사이드 스트림판 (mmq_y_s 전용 버퍼는 호출부가 지정).
    pub fn gemm_mmq_s(
        &self,
        ty: u32,
        y_f32: *const u8,
        w: *const u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        self.gemm_mmq_impl(ty, y_f32, w, n_in, n_out, t, out, self.stream2, true)
    }
}
