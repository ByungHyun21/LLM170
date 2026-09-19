//! q4acc 프레임 — FrameState·FrameHost (활성 상주 디코드, plans/78 R1).

use super::*;
use crate::rawhip::{env_on, env_eq};

impl llm170_core::matmul::FrameState for Q4Acc {
    fn set_ctx_len(&self, n: usize) {
        self.ctx_len.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn frame_begin(&self, t: usize) {
        self.cur_t.store(t.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// GDN AR (프레임) — qwen35 raw 디코더와 동일 커널(gdn_ar_w_swap).
    /// q는 호출부에서 1/√d 스케일이 끝난 상태 → 커널 scale=1.0.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar(
        &self,
        q_scaled: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        states: u64,
        out: u64,
        n_seqs: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        if n_seqs != 1 {
            return Err("q4acc: frame_gdn_ar np 미지원".into());
        }
        let (mut sp, mut qp, mut kp, mut vp, mut bp, mut op_) = (
            self.fptr(states)?,
            self.fptr(q_scaled)?,
            self.fptr(k)?,
            self.fptr(v)?,
            self.fptr(beta_ge)?,
            self.fptr(out)?,
        );
        let mut dd = d as i32;
        let mut ks = (h_k * d) as i32;
        let mut vs = (h_v * d) as i32;
        let mut hv = h_v as i32;
        let mut hk = h_k as i32;
        let mut sc = 1.0f32;
        // t토큰 순차 재귀 — 커널 내부 ti 루프가 상태를 이어간다(1런치).
        let mut tt = self.t_cur() as i32;
        if env_on("LLM170_Q4_DBG") {
            eprintln!(
                "# ar-args s={:?} q={:?} k={:?} v={:?} bg={:?} out={:?} d={dd} ks={ks} vs={vs} hv={hv} hk={hk}",
                sp as usize, qp as usize, kp as usize, vp as usize, bp as usize, op_ as usize
            );
        }
        // gdn_ar_w_swap: 전치 상태 레이아웃(s[dv*d+kdim]) + d=128 고정(레인당
        // kdim 4개). 구 q4_gdn_ar_w의 열 단위 접근은 512B 스트라이드였다.
        self.ctx.launch3(
            "gdn_ar_w_swap",
            d as u32,
            h_v as u32,
            1,
            32,
            &mut cargs!(&mut sp, &mut qp, &mut kp, &mut vp, &mut bp, &mut op_, &mut dd, &mut ks, &mut vs, &mut hv, &mut hk, &mut sc, &mut tt),
        )
    }


    fn frame_moe_gather(
        &self,
        mix: u64,
        xsel: u64,
        n: usize,
        k_sel: usize,
        t: usize,
    ) -> Result<(), String> {
        let (mp, xs) = (self.fptr(mix)?, self.fptr(xsel)?);
        let total = (t * k_sel * n) as u32;
        let (mut a, mut b) = (mp, xs);
        let (mut nn, mut ks, mut tt) = (n as i32, k_sel as i32, t as i32);
        self.kop("q4_moe_gather", total.div_ceil(128), 1, 1, 128, &mut cargs!(&mut a, &mut b, &mut nn, &mut ks, &mut tt))
    }

    fn frame_moe_scatter(
        &self,
        ys: u64,
        wt: u64,
        out: u64,
        k_sel: usize,
        n: usize,
        t: usize,
    ) -> Result<(), String> {
        let (yp, wp, op_) = (self.fptr(ys)?, self.fptr(wt)?, self.fptr(out)?);
        let (mut a, mut b, mut c) = (yp, wp, op_);
        let (mut ks, mut nn, mut tt) = (k_sel as i32, n as i32, t as i32);
        self.kop("q4_moe_scatter", ((t * n) as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut a, &mut b, &mut c, &mut ks, &mut nn, &mut tt))
    }

    /// MoE ids 구동 전문가 GEMM — 스택 + ids(프레임 상주). ids는 행당 u32.
    /// ids는 확률순(전문가순 아님)이라 연속 런이 1행씩 흩어진다 — 실측
    /// t=512·k_sel=10에서 런치 ~4000회/층. 카운팅 정렬로 전문가 순으로 묶어
    /// 런치 수를 전문가 수 수준으로 줄이고(순열은 relu 없이 안정), 결과 행
    /// 순서는 역순열 산란으로 복원한다(가중합이 원래 행 순서를 요구).
    fn frame_moe_gemm(
        &self,
        x: u64,
        ws: &llm170_core::matmul::Weight<'_>,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
        k_sel: usize,
    ) -> Result<(), String> {
        let tm = env_on("LLM170_MOE_TIME");
        let t0 = std::time::Instant::now();
        let mut lap = t0;
        let phase = |name: &str, lap: &mut std::time::Instant| {
            if tm {
                let ms = lap.elapsed().as_secs_f64() * 1e3;
                if ms >= 0.05 {
                    eprintln!("# moe-phase {name}={ms:.2}ms");
                }
                *lap = std::time::Instant::now();
            }
        };
        let n_in = ws.n_in as usize;
        let n_out = ws.n_out as usize / n_expert_stack.max(1);
        // 행 수 = t·k_sel — 버퍼는 t_max 크기라 길이에서 유도할 수 없다.
        let rows = self.t_cur() * k_sel.max(1);
        let xp = self.fptr(x)?;
        let op_ = self.fptr(out)?;
        let (wd, f32w) = self.dev_weight(ws)?;
        let per_expert = ws.data.len() / n_expert_stack.max(1);
        phase("weight", &mut lap);
        let gen_q = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
        let (xq, xq_w) = if f32w {
            (std::ptr::null_mut(), 0usize)
        } else {
            let key = (xp as usize, rows, gen_q);
            let hit = {
                let c = self.quant_cache.lock().map_err(|e| e.to_string())?;
                c.as_ref()
                    .filter(|(xp0, r0, g0, _, _)| (*xp0, *r0, *g0) == key)
                    .map(|(_, _, _, q, w)| (*q as *mut u8, *w))
            };
            if llm170_diag::dump::opts().moe {
                let c = self.quant_cache.lock().map_err(|e| e.to_string())?;
                let (h, ck) = match c.as_ref() {
                    Some((xp0, r0, g0, _, _)) => (
                        (*xp0, *r0, *g0) == key,
                        format!("({xp0:#x},{r0},{g0})"),
                    ),
                    None => (false, "empty".into()),
                };
                eprintln!(
                    "# qcache key=({:#x},{rows},{gen_q}) {} cached={ck}",
                    xp as usize,
                    if h { "HIT" } else { "MISS" }
                );
            }
            match hit {
                Some(v) => v,
                None => {
                    let (q, w) = self.frame_quant(xp, n_in, rows)?;
                    let mut c = self.quant_cache.lock().map_err(|e| e.to_string())?;
                    *c = Some((key.0, key.1, key.2, q as u64, w));
                    (q, w)
                }
            }
        };
        phase("quant", &mut lap);
        // direct-ids(t=1, LLM170_MOE_DIRECT=1): 그룹화 테이블·gather·scatter를
        // 전부 건너뛰고 커널이 ids[row]를 직접 읽는다. 행 순서가 곧 ids 순서라
        // 가중합(ys[e*n+i])이 그대로 맞고, 호스트 왕복(ids d2h+빌드+h2d)도 없다.
        // t=1에서는 k_sel행이 같은 벡터이므로 스트라이드 0으로 0번 행을 읽는다.
        // plans/74: direct-ids 는 행 수만 보면 된다 — np t=4(k_sel×4=40행)도
        // 이 빠른 경로로(그룹화 경로는 프리필 대량 행 전용). rows<=64 게이트로
        if (self.t_cur() == 1 || rows <= 64)
            && ws.ty == GgmlType::Q4K
            && !f32w
            && !env_on("LLM170_MOE_GROUPED")
        {
            let idp = self.fptr(ids)?;
            // K-분할: 타일 40블록(=1/CU)이던 점유율을 ksplit배로. 부분합은 part에
            // 남기고 reduce가 k 오름차순 합산(결정적, 순서 재결합만 다른 미세 드리프트).
            let ksplit: u32 = std::env::var("LLM170_MOE_KSPLIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4)
                .clamp(1, 8);
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let part_buf = self
                .ctx
                .scratch(rows * n_out * ksplit as usize * 8)?;
            let mut part_p = part_buf as *mut std::ffi::c_void;
            let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let (mut xw, mut tt, mut eb) = (0i32, rows as i32, per_expert as i32);
            let mut rp: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ip) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut eb) as *mut _ as *mut std::ffi::c_void,
                (&mut rp) as *mut _ as *mut std::ffi::c_void,
            ];
            // 실측(2026-09-14): 호출당 10전문가 x 0.92MB = 9.2MB를 51us에 옮긴다
            // = 180 GB/s ≈ DRAM(236)의 76%. 이미 최적에 가까워 K-분할(4배 블록,
            // -1.4%), GEMV형 그리드(16배 블록 + 트리 환원, 중립), 접근 패턴
            // 프로브(235-264 GB/s로 평탄)가 모두 중립이었다 — 격차가 아니라 산술이었다.
            self.ctx.launch3(
                "q4_gemm_q4k_ge_ids",
                n_out.div_ceil(16) as u32,
                rows.div_ceil(16) as u32,
                ksplit,
                256,
                &mut args,
            )?;
            if ksplit > 1 {
                let mut pp = part_buf as *mut std::ffi::c_void;
                let mut op2 = self.fptr(out)? as *mut std::ffi::c_void;
                let mut nn = (rows * n_out) as i32;
                let mut ks = ksplit as i32;
                let mut rargs: Vec<*mut std::ffi::c_void> = vec![
                    (&mut pp) as *mut _ as *mut std::ffi::c_void,
                    (&mut op2) as *mut _ as *mut std::ffi::c_void,
                    (&mut nn) as *mut _ as *mut std::ffi::c_void,
                    (&mut ks) as *mut _ as *mut std::ffi::c_void,
                ];
                self.ctx.launch3(
                    "q4_gemm_q4k_ids_reduce",
                    ((rows * n_out) as u32).div_ceil(256),
                    1,
                    1,
                    256,
                    &mut rargs,
                )?;
            }
            return Ok(());
        }
        // plans/73: Q5_1 다운의 direct-ids를 **그룹화 캐시 평가 전에** 올린다 —
        // 종전엔 캐시 미스가 q4_moe_group_t1 커널 + 비동기 d2h를 매층 발사하고
        // 곧바로 direct-ids로 반환해 그 작업이 전부 쓰레기였다(0.034ms × 48층
        // + 스텝당 48회의 d2h_issue).
        if (self.t_cur() == 1 || rows <= 64)
            && ws.ty == GgmlType::Q5_1
            && !f32w
            && rows > 0
            && n_in / 32 <= 32
            && !env_on("LLM170_MOE_GROUPED")
            && !env_eq("LLM170_Q5W", "0")
        {
            let idp = self.fptr(ids)?;
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let (mut xw, mut tt) = (xq_w as i32, rows as i32);
            let mut ew = (per_expert / 4) as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ip) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ew) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_gemm_q5_1_w_ids",
                n_out.div_ceil(8) as u32,
                rows as u32,
                1,
                256,
                &mut args,
            )?;
            return Ok(());
        }
        // plans/73: Q8_0 다운 전문가도 direct-ids 워프판으로 — 종전엔 이 층들이
        if (self.t_cur() == 1 || rows <= 64)
            && ws.ty == GgmlType::Q8_0
            && !f32w
            && rows > 0
            && n_in / 32 <= 32
            && !env_on("LLM170_MOE_GROUPED")
            && !env_eq("LLM170_Q8IDS", "0")
        {
            let idp = self.fptr(ids)?;
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let mut ew = per_expert as i32; // 바이트 — 34B 행 비정렬 오프셋용
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let (mut xw, mut tt) = (xq_w as i32, rows as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ip) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ew) as *mut _ as *mut std::ffi::c_void,
            ];
            if env_on("LLM170_Q8IDS_DBG") {
                eprintln!("# q8ids launch n_in={n_in} n_out={n_out} rows={rows} per_expert={per_expert}");
            }
            self.ctx.launch3(
                "gemm_q8_0_ids",
                n_out.div_ceil(8) as u32,
                rows as u32,
                1,
                256,
                &mut args,
            )?;
            return Ok(());
        }
        let ne = n_expert_stack.max(1);
        // 그룹화 캐시 — gate/up/down 3개 투영이 같은 라우팅을 공유한다. 게이트가
        // 1회만 d2h(동기)+정렬+순열 업로드하고 나머지는 디바이스 순열을 재사용.
        // 실측: 호출마다 동기하던 시절 층당 9회 → MoE 77ms/층(청크 59%).
        let generation = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
        let hit = {
            let c = self.moe_group.lock().map_err(|e| e.to_string())?;
            c.as_ref()
                .filter(|g| g.generation == generation && g.rows == rows)
                .map(|g| (g.perm_d, g.inv_d, g.rowexp_d, g.perm_pad_d, g.inv_pad_d, g.tilexp_d, g.rows_pad, g.rows_pad_d, g.off.clone(), g.off_d, g.pinned_off))
        };
        if tm {
            eprintln!("# moe-cache {}", if hit.is_some() { "HIT" } else { "MISS" });
        }
        let (perm_d, inv_d, rowexp_d, perm_pad_d, inv_pad_d, tilexp_d, rows_pad, rows_pad_d, off, off_d, pinned_off) = match hit {
            Some(v) => v,
            None => {
                // t=1(디코드): 그룹화를 GPU에서 한다. 호스트 왕복(동기 d2h + 테이블
                // 빌드 + h2d 3회)이 스텝의 44%(48층×0.9ms)였다 — 테이블은 ids의
                // 순수 함수라 커널로 옮기면 사라진다. 결과는 호스트판과 동일 순서라
                // 비트 동일. (프리필은 행 수가 커서 기존 호스트 경로 유지.)
                // 기본은 호스트 경로 — 2026-09-14 A/B: 호스트 755.4ms vs
                // 디바이스 1006.9ms(tg8, 같은 바이너리). 디바이스판은 테이블이
                // 비트 동일하고 호스트 빌드·h2d 3회를 없애지만, 추가분(그룹 커널
                // + 상한(rows*16+16) 크기로 커진 gather/scatter + 폴백의 이벤트
                // 대기)이 그보다 커서 +31ms/스텝이다. down(q8_0) 폴백까지 그룹
                // 커널로 덮으면 재평가한다. 옵트인: LLM170_MOE_GROUP_DEV=1.
                // t=1 전용 (프리필은 아직 불가). 2026-09-14 리팩터: 상한을 한 곳에서
                // 계산(rows + 16*ne)해 커널 인자·모든 버퍼에 쓰고, 소비 지점에서
                // 디바이스가 보고한 rows_pad를 검증한다 — 리팩터 전에는 상한이
                // 5곳에 흩어져 이 경로 자체가 잠재 OOB였다(rows*16+16=176 vs 실제
                // ≤8,202). 리팩터 후 t=1은 비트 동일로 검증됨.
                // 프리필(t>1)은 아직 5번째 상한 축이 남아 실패한다(h2d 8MB — 크기는
                // h2d 진단이 보고한다). 켜려면 그 축부터 찾아야 한다.
                // t=1(디코드) 전용 — 프리필(t>1)은 plans/68에서 레이아웃 혼재
                // (패딩/비패딩 gather·scatter·폴백 오프셋)를 전면 교정했으나
                // 잔여 발산(16토큰 중 마지막 1개 플립)과 진단 동기화 시에만
                // 재현되는 폴백 행 수 오염이 남아 기본 경로는 유지한다.
                let pf_exp = env_eq("LLM170_MOE_GROUP_PF", "1");
                if !env_eq("LLM170_MOE_GROUP_DEV", "0")
                    && (self.t_cur() == 1 || pf_exp)
                    && ne <= 512
                    && rows > 0
                {
                    // **단일 상한**: Σ_e ceil(r_e/16)*16 ≤ rows + 16*ne
                    // (전문가당 ≤15행 패딩). 종전 rows*16+16은 16배 과대였고,
                    // 그 값으로 커널 zero-fill·호스트 버퍼가 어긋나 OOB가 났다.
                    let bound = rows + 16 * ne;
                    let (pd, ivd, rxd) = {
                        let mut a = self.rperm.lock().map_err(|e| e.to_string())?;
                        let pd = a.ensure(&self.ctx, rows * 4)? as u64;
                        let mut b = self.rperm2.lock().map_err(|e| e.to_string())?;
                        let ivd = b.ensure(&self.ctx, rows * 4)? as u64;
                        let mut c = self.rexp.lock().map_err(|e| e.to_string())?;
                        // GEMM이 rows_pad까지 rowexp를 읽는다 → bound 크기.
                        let rxd = c.ensure(&self.ctx, bound * 4)? as u64;
                        (pd, ivd, rxd)
                    };
                    let (ppd, ipd, txd, offd, rpd) = {
                        let mut a = self.gp.lock().map_err(|e| e.to_string())?;
                        let ppd = a.ensure(&self.ctx, (bound + 1) * 4)? as u64;
                        let mut b = self.gi.lock().map_err(|e| e.to_string())?;
                        let ipd = b.ensure(&self.ctx, rows * 4)? as u64;
                        let mut c = self.texp.lock().map_err(|e| e.to_string())?;
                        let txd = c.ensure(&self.ctx, (bound / 16 + 1) * 4)? as u64;
                        let mut d = self.gyp.lock().map_err(|e| e.to_string())?;
                        let base = d.ensure(&self.ctx, (ne + 2) * 4)? as u64;
                        let offd = base;
                        let rpd = base + (ne as u64 + 1) * 4; // off 뒤 4B = rows_pad
                        (ppd, ipd, txd, offd, rpd)
                    };
                    self.moe_group_dev(ids, ne, rows, offd, pd, ivd, rxd, ppd, ipd, txd, rpd, bound)?;
                    if env_on("LLM170_MOE_GCHECK") {
                        // 진단: 디바이스 테이블과 호스트 재계산을 비교(첫 불일치 지점 출력).
                        self.ctx.sync().map_err(|e| e.to_string())?;
                        let mut dev_off = vec![0i32; ne + 2];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut dev_off), offd as *const u8)?;
                        let mut dev_perm = vec![0u32; rows];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut dev_perm), pd as *const u8)?;
                        let mut dev_rowexp = vec![0u32; bound];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut dev_rowexp), rxd as *const u8)?;
                        let idp = self.fptr(ids)?;
                        let mut idv = vec![0u32; rows];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
                        let mut cnt = vec![0i32; ne];
                        for &e in &idv { cnt[(e as usize).min(ne - 1)] += 1; }
                        let mut hoff = vec![0usize; ne + 1];
                        let mut acc2 = 0;
                        for e in 0..ne { hoff[e] = acc2; acc2 += cnt[e] as usize; }
                        hoff[ne] = acc2;
                        let mut bad = 0;
                        for e in 0..=ne {
                            if dev_off[e] as usize != hoff[e] {
                                eprintln!("# gcheck off[{e}] dev={} host={}", dev_off[e], hoff[e]);
                                bad += 1;
                                if bad > 4 { break; }
                            }
                        }
                        if bad == 0 {
                            let mut cur = hoff[..ne].to_vec();
                            for (i, &e) in idv.iter().enumerate() {
                                let e2 = (e as usize).min(ne - 1);
                                let ppos = cur[e2]; cur[e2] += 1;
                                if dev_perm[ppos] as usize != i { 
                                    eprintln!("# gcheck perm@{ppos} dev={} host={i}", dev_perm[ppos]);
                                    bad += 1;
                                    if bad > 4 { break; }
                                }
                                if dev_rowexp[ppos] as usize != e2 {
                                    eprintln!("# gcheck rowexp@{ppos} dev={} host={e2}", dev_rowexp[ppos]);
                                    bad += 1;
                                    if bad > 4 { break; }
                                }
                            }
                        }
                        eprintln!("# gcheck rows={rows} rows_pad_dev={} bad={bad}", dev_off[ne + 1]);
                    }
                    // 폴백(비 Q4K/Q5_1 타입)용 오프셋. 기본은 비동기로 미리 걸어
                    // 소비 시점(층 하단)까지 gate/up GEMM이 지연을 덮는다.
                    // LLM170_MOE_GROUP_SYNC=1이면 즉시 동기(스트림 드레인) —
                    // 호스트 경로와 같은 순서 조건을 만들어 순서 효과를 검정한다.
                    // 이분법: 비동기 예약 자체를 건너뛴다(폴백은 동기 d2h로).
                    let pinned_off = if env_on("LLM170_MOE_GROUP_NOD2H") {
                        std::ptr::null_mut()
                    } else if env_on("LLM170_MOE_GROUP_SYNC") {
                        let buf = self.ctx.d2h_issue((ne + 2) * 4, offd as *const u8)?;
                        self.ctx.d2h_wait()?;
                        buf
                    } else {
                        // +4B: 오프셋 뒤에 디바이스가 계산한 rows_pad가 붙어 있다(가드용).
                        self.ctx.d2h_issue((ne + 2) * 4, offd as *const u8)?
                    };
                    let mut c = self.moe_group.lock().map_err(|e| e.to_string())?;
                    *c = Some(MoeGroup {
                        generation, rows, perm_d: pd, inv_d: ivd, rowexp_d: rxd,
                        perm_pad_d: ppd, inv_pad_d: ipd, tilexp_d: txd,
                        rows_pad: bound, rows_pad_d: rpd, off_d: offd, pinned_off, off: Vec::new(),
                    });
                    (pd, ivd, rxd, ppd, ipd, txd, bound, rpd, Vec::new(), offd, pinned_off)
                } else {
                // 그래프 캡처 경계 — 이 블록은 d2h(라우팅 판독)+호스트 정렬+h2d를
                // 하므로 캡처 밖이어야 한다(세그먼트 분할점).
                unsafe { crate::rawhip::capture_mark(self.ctx.stream, "moe_group_in") }?;
                let mut lp = std::time::Instant::now();
                let idp = self.fptr(ids)?;
                let mut idv = vec![0u32; rows];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
                if tm {
                    let ms = lp.elapsed().as_secs_f64() * 1e3;
                    if ms >= 0.05 { eprintln!("# moe-miss d2h={ms:.2}ms rows={rows}"); }
                    lp = std::time::Instant::now();
                }
                let mut off = vec![0usize; ne + 1];
                for &e in &idv {
                    off[(e as usize).min(ne - 1) + 1] += 1;
                }
                for e in 0..ne {
                    off[e + 1] += off[e];
                }
                let mut cur = off[..ne].to_vec();
                let mut perm = vec![0u32; rows];
                let mut inv = vec![0u32; rows];
                for (i, &e) in idv.iter().enumerate() {
                    let e = (e as usize).min(ne - 1);
                    let p = cur[e];
                    perm[p] = i as u32;
                    inv[i] = p as u32;
                    cur[e] += 1;
                }
                if tm {
                    let ms = lp.elapsed().as_secs_f64() * 1e3;
                    if ms >= 0.05 { eprintln!("# moe-miss sort={ms:.2}ms"); }
                    lp = std::time::Instant::now();
                }
                let (pd, ivd, rxd) = {
                    let mut a = self.rperm.lock().map_err(|e| e.to_string())?;
                    let pd = a.ensure(&self.ctx, rows * 4)? as u64;
                    let mut b = self.rperm2.lock().map_err(|e| e.to_string())?;
                    let ivd = b.ensure(&self.ctx, rows * 4)? as u64;
                    let mut c = self.rexp.lock().map_err(|e| e.to_string())?;
                    let rxd = c.ensure(&self.ctx, rows * 4)? as u64;
                    (pd, ivd, rxd)
                };
                self.ctx.h2d(pd as *mut u8, bytemuck::cast_slice(&perm))?;
                self.ctx.h2d(ivd as *mut u8, bytemuck::cast_slice(&inv))?;
                // rowexp: 순열 후 행 p의 전문가 = idv[perm[p]]
                let mut rowexp = vec![0u32; rows];
                for p in 0..rows {
                    rowexp[p] = idv[(perm[p] as usize).min(rows - 1)].min((ne - 1) as u32);
                }
                self.ctx.h2d(rxd as *mut u8, bytemuck::cast_slice(&rowexp))?;
                let mut off_pad = vec![0usize; ne + 1];
                for e in 0..ne {
                    off_pad[e + 1] = off_pad[e] + (off[e + 1] - off[e]).div_ceil(16) * 16;
                }
                let rows_pad = off_pad[ne].max(16);
                let mut perm_pad = vec![0u32; rows_pad];
                let mut inv_pad = vec![0u32; rows];
                for e in 0..ne {
                    let r = off[e + 1] - off[e];
                    for i in 0..(off_pad[e + 1] - off_pad[e]) {
                        let pd = off_pad[e] + i;
                        if i < r {
                            let src = off[e] + i;
                            perm_pad[pd] = perm[src];
                            inv_pad[perm[src] as usize] = pd as u32;
                        } else {
                            perm_pad[pd] = 0;
                        }
                    }
                }
                let mut tilexp = vec![0u32; rows_pad / 16];
                for e in 0..ne {
                    for tg in off_pad[e] / 16..off_pad[e + 1] / 16 {
                        tilexp[tg] = e as u32;
                    }
                }
                let (ppd, ipd, txd) = {
                    let mut a = self.gp.lock().map_err(|e| e.to_string())?;
                    let ppd = a.ensure(&self.ctx, rows_pad * 4)? as u64;
                    let mut b = self.gi.lock().map_err(|e| e.to_string())?;
                    let ipd = b.ensure(&self.ctx, rows * 4)? as u64;
                    let mut c = self.texp.lock().map_err(|e| e.to_string())?;
                    let txd = c.ensure(&self.ctx, (rows_pad / 16).max(1) * 4)? as u64;
                    (ppd, ipd, txd)
                };
                if env_on("LLM170_GE5_DBG") {
                    eprintln!(
                        "# ge5 rows={rows} rows_pad={rows_pad} ne={ne} ppd={ppd} ipd={ipd} txd={txd} \
perm_pad[0..4]={:?} inv_pad[0..4]={:?} tile[0..4]={:?} off[0..4]={:?}",
                        &perm_pad[..perm_pad.len().min(4)],
                        &inv_pad[..inv_pad.len().min(4)],
                        &tilexp[..tilexp.len().min(4)],
                        &off[..off.len().min(4)]
                    );
                }
                self.ctx.h2d(ppd as *mut u8, bytemuck::cast_slice(&perm_pad))?;
                self.ctx.h2d(ipd as *mut u8, bytemuck::cast_slice(&inv_pad))?;
                self.ctx.h2d(txd as *mut u8, bytemuck::cast_slice(&tilexp))?;
                if tm {
                    let ms = lp.elapsed().as_secs_f64() * 1e3;
                    if ms >= 0.05 { eprintln!("# moe-miss h2d={ms:.2}ms"); }
                }
                unsafe { crate::rawhip::capture_mark(self.ctx.stream, "moe_group_out") }?;
                let mut c = self.moe_group.lock().map_err(|e| e.to_string())?;
                *c = Some(MoeGroup { generation, rows, perm_d: pd, inv_d: ivd, rowexp_d: rxd,
                    perm_pad_d: ppd, inv_pad_d: ipd, tilexp_d: txd, rows_pad, rows_pad_d: 0, off_d: 0, pinned_off: std::ptr::null_mut(), off: off.clone() });
                (pd, ivd, rxd, ppd, ipd, txd, rows_pad, 0u64, off, 0u64, std::ptr::null_mut())
                }
            }
        };
        phase("group", &mut lap);
        let row_u32 = if f32w { n_in } else { xq_w };
        // plans/68 레이아웃 실험 플래그 — t=1의 기존(검증된) 동작은 그대로 두고
        // 프리필 디바이스 그룹화 실험에서만 패딩 도메인 레이아웃을 쓴다.
        let pad_layout = rows_pad_d != 0 && self.t_cur() > 1;
        // 디바이스 그룹화 경로의 GEMM은 t = rows_pad로 x를 읽는다(패딩 행의 출력은
        // scatter가 버리므로 값은 무관, 크기만 rows_pad까지 필요).
        let xbuf_rows = if rows_pad_d != 0 { rows + 16 * ne } else { rows };
        let xg = {
            let mut g = self.xperm.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, xbuf_rows * row_u32 * 4)?
        };
        let yg = {
            let mut g = self.yperm.lock().map_err(|e| e.to_string())?;
            // ★ 5번째 축(plans/68): q4_gemm_q4k_ge는 r < *rows_pad까지
            // out[r·n_out+o]에 기록한다 — 디바이스 그룹화 경로(rows_pad_d≠0)는
            // 패딩 행(≤16·ne)분까지 버퍼를 확보해야 한다. 종전 rows 크기여서
            // 프리필에서 out 끝을 넘는 쓰기 → HIP 700(10차 소거의 정체).
            let ybuf_rows = if rows_pad_d != 0 { rows + 16 * ne } else { rows };
            g.ensure(&self.ctx, ybuf_rows * n_out * 4)?
        };
        let xsrc0 = if f32w { xp } else { xq };
        // plans/68 레이아웃 일관화: 디바이스 그룹화(rows_pad_d≠0)는 GEMM이
        // **패딩 도메인**(r < *rows_pad, rowexp=패딩 인덱스)으로 읽는다 — gather도
        // perm_pad/bound행으로. 호스트 경로는 종전대로 비패딩 perm_d/rows.
        if pad_layout {
            self.rows_permute_dev(xsrc0, perm_pad_d as *mut u8, xg, row_u32, rows + 16 * ne)?;
        } else {
            self.rows_permute_dev(xsrc0, perm_d as *mut u8, xg, row_u32, rows)?;
        }
        phase("gather", &mut lap);
        if llm170_core::qwen4exp::frame::stage_skipped("moe") {
            // 진단용(LLM170_STAGE_SKIP=moe): 전문가 GEMM 생략 — 비용 분해, 출력 무효.
            return Ok(());
        }
        // 그룹 런치(옵트인) — q4_K 전문가를 한 번에: 청크당 런치 7.4만 → 48.
        // 호스트/갭 ~2.7초@pp2048 제거(KTRACE 실측). 산술은 _m과 동일(비트 동일).
        // 기본 경로 — 비트 동일(토큰 검증), pp2048 −1.8%, 런치 7.4만→48/청크.
        // q5_1(다운) 그룹판 — 16배수 패딩 레이아웃으로 타일=전문가, 가중치 재독 1회.
        if ws.ty == GgmlType::Q5_1 && !f32w && rows > 0 {
            // direct-ids(t=1): down도 그룹화 없이 — 게이트/up만 direct로는 down에서
            // 그룹화(d2h+빌드+h2d)가 1회 발생해 이득이 사라진다.
            // 실측(2026-09-14): down은 72 GB/s로 게이트/up의 180에 크게 못 미친다.
            // 원인은 워프가 출력행마다 480B 스트라이드로 읽어 32B 섹터당 4B만
            // 쓰는 8배 증폭. gm의 협조 적재 이식은 **불가능**하다(그 불변식은
            // 타일당 단일 전문가인데 비정렬 direct는 행마다 전문가가 다르다 —
            // o-행 하나에 16전문가 가중치가 필요해 공유 버퍼로 표현 불가, 시도 후 복원).
            // 워프-퍼-행 재설계도 q5_1의 6워드 슈퍼블록 입도 때문에 3배가 한계였다.
            // 즉 6ms는 Q5_1 레이아웃 고유 비용이다.
            if self.t_cur() == 1 && !env_on("LLM170_MOE_GROUPED") {
                let idp = self.fptr(ids)?;
                let mut x_p = xq as *mut std::ffi::c_void;
                let mut w_p = wd as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
                let mut ip = idp as *mut std::ffi::c_void;
                let (mut ni, mut no) = (n_in as i32, n_out as i32);
                let (mut xw, mut tt) = (xq_w as i32, rows as i32);
                let mut ew = (per_expert / 4) as i32;
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ip) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut ew) as *mut _ as *mut std::ffi::c_void,
                ];
                self.ctx.launch3(
                    "q4_gemm_q5_1_gm_ids",
                    n_out.div_ceil(16) as u32,
                    rows.div_ceil(16) as u32,
                    1,
                    256,
                    &mut args,
                )?;
                return Ok(());
            }
            let xgp = {
                let mut g = self.gxp.lock().map_err(|e| e.to_string())?;
                g.ensure(&self.ctx, rows_pad * xq_w * 4)?
            };
            self.rows_permute_dev(xq, perm_pad_d as *mut u8, xgp, xq_w, rows_pad)?;
            let ygp = {
                let mut g = self.gyp.lock().map_err(|e| e.to_string())?;
                g.ensure(&self.ctx, rows_pad * n_out * 4)?
            };
            {
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut x_p = xgp as *mut std::ffi::c_void;
                let mut w_p = wd as *mut std::ffi::c_void;
                let mut o_p = ygp as *mut std::ffi::c_void;
                let mut tx_p = tilexp_d as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt, mut ew) = (
                    n_in as i32,
                    n_out as i32,
                    xq_w as i32,
                    rows_pad as i32,
                    (per_expert / 4) as i32,
                );
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut tx_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut ew) as *mut _ as *mut std::ffi::c_void,
                ];
                let smem = (16 * (n_in / 32) * 24) as u32;
                self.ctx.launch3_dyn(
                    "q4_gemm_q5_1_gm",
                    n_out.div_ceil(16).min(65535) as u32,
                    rows_pad.div_ceil(16) as u32,
                    1,
                    256,
                    smem,
                    &mut args,
                )?;
            }
            self.rows_permute_dev(ygp, inv_pad_d as *mut u8, op_, n_out, rows)?;
            if tm {
                eprintln!("# moe-phase TOTAL={:.2}ms rows={rows} ge5", t0.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(());
        }
        if ws.ty == GgmlType::Q4K && !f32w && rows > 0 {
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut x_p = xg as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut o_p = yg as *mut std::ffi::c_void;
            let mut rx_p = rowexp_d as *mut std::ffi::c_void;
            // 디바이스 그룹화 경로면 rows_pad를 커널이 디바이스에서 읽는다(가드).
            let mut rpd_p = rows_pad_d as *mut u8;
            let (mut ni, mut no, mut xw, mut tt, mut eb) = (
                n_in as i32,
                n_out as i32,
                xq_w as i32,
                rows as i32,
                per_expert as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut rx_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut eb) as *mut _ as *mut std::ffi::c_void,
                (&mut rpd_p) as *mut _ as *mut std::ffi::c_void,
            ];
            {
                use std::sync::Mutex;
                use std::sync::OnceLock;
                static SEEN: OnceLock<Mutex<Vec<(usize, usize, usize)>>> = OnceLock::new();
                if env_on("LLM170_Q4_DBG") {
                    let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
                    if let Ok(mut v) = seen.lock() {
                        let key = (n_in, n_out, rows);
                        if !v.contains(&key) && v.len() < 8 {
                            v.push(key);
                            eprintln!(
                                "# q4_gemm_q4k_ge: n_in={n_in} n_out={n_out} rows={rows} blocks={}x{}",
                                n_out.div_ceil(16),
                                rows.div_ceil(16)
                            );
                        }
                    }
                }
            }
            if llm170_diag::dump::opts().moe {
                // 진단(plans/80): GEMM의 숨은 입력(xq 전체·rowexp·perm)을
                // FNV 해시로 비교한다. mxsel/ids가 같은데 이들이 다르면
                // quant/그룹화 산출물이 오염된 것(버퍼 결함).
                self.ctx.sync().map_err(|e| e.to_string())?;
                let fnv = |b: &[u32]| -> u64 {
                    b.iter().fold(0xcbf29ce484222325u64, |a, &w| {
                        a.wrapping_mul(0x100000001b3) ^ (w as u64)
                    })
                };
                let mut xqf = vec![0u32; rows.min(160) * xq_w];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut xqf), xsrc0 as *const u8)?;
                let mut xf = vec![0u32; rows.min(160) * n_in];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut xf), xp as *const u8)?;
                let mut rxf = vec![0u32; rows];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut rxf), rowexp_d as *const u8)?;
                let mut pmf = vec![0u32; rows];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut pmf), perm_d as *const u8)?;
                let mut ivf = vec![0u32; rows];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut ivf), inv_d as *const u8)?;
                eprintln!(
                    "# moedump rows={rows} x_h={:016x} xq_h={:016x} rx_h={:016x} pm_h={:016x} iv_h={:016x}",
                    fnv(&xf),
                    fnv(&xqf),
                    fnv(&rxf),
                    fnv(&pmf),
                    fnv(&ivf)
                );
            }
            self.ctx.launch3(
                "q4_gemm_q4k_ge",
                n_out.div_ceil(16) as u32,
                rows_pad.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            )?;
            let scat = if pad_layout { inv_pad_d } else { inv_d };
            self.rows_permute_dev(yg, scat as *mut u8, op_, n_out, rows)?;
            phase("scatter", &mut lap);
            if tm {
                eprintln!("# moe-phase TOTAL={:.2}ms rows={rows}", t0.elapsed().as_secs_f64() * 1e3);
            }
            self.moe_hash_check("ge", op_, rows, n_out)?;
            return Ok(());
        }
        // 디바이스 그룹화 경로(off 비어 있음): 폴백(비 Q4K/Q5_1 타입, 예: down의
        // q8_0)은 전문가별 런치를 위해 오프셋만 읽는다 — 그룹화·테이블 빌드·업로드
        // 왕복은 GPU가 이미 끝냈으므로 여기서는 (ne+1)개 int만 받는다.
        let mut off_d2h;
        // plans/68: 디바이스 그룹화 경로의 xg는 **패딩 도메인** — 폴백(전문가별
        // 런치)도 패딩 오프셋(off_pad)에서 구간을 읽어야 행이 맞는다.
        let off: &[usize] = if off.is_empty() && rows_pad_d != 0 {
            self.ctx.d2h_wait()?;
            // 오프셋 + 그 뒤 4B(디바이스가 계산한 rows_pad)를 함께 읽어 **상한을
            // 검증**한다. 초과하면 크래시(h2d 700) 대신 진단 메시지로 실패시킨다 —
            // 2026-09-14에 상한 가정이 5곳에 흩어져 있어 디버깅이 오래 걸렸다.
            let mut b = vec![0i32; ne + 2];
            if !pinned_off.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(pinned_off as *const u8, b.as_mut_ptr() as *mut u8, (ne + 2) * 4);
                }
            } else {
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut b), off_d as *const u8)?;
            }
            let _ = &b;
            let rows_pad_dev = b[ne + 1].max(0) as usize;
            let bound = self.t_cur() * k_sel.max(1) + 16 * ne;
            if env_on("LLM170_MOE_BCHECK") {
                // b(pinned off) 무결성 — r 오염(gemm_q5k gx=1.04억)의 원본 관찰.
                let mut mono_ok = true;
                for i in 0..ne {
                    if b[i] > b[i + 1] { mono_ok = false; break; }
                }
                let total = b[ne];
                if !mono_ok || total < 0 || total as usize > bound || b[..ne.min(8)].iter().any(|&x| x < 0) {
                    eprintln!(
                        "# bcheck BAD rows={rows} ne={ne} mono={mono_ok} total={total} bound={bound} b0..7={:?} rp={}",
                        &b[..8.min(ne)], b[ne + 1]
                    );
                }
            }
            if rows_pad_dev > bound {
                return Err(format!(
                    "moe 그룹화: rows_pad {rows_pad_dev} > bound {bound} (ne={ne}) — 상한 가정 위반"
                ));
            }
            if pad_layout {
                // [프리필 실험] 시작점은 패딩 도메인, 행 수는 실제 카운트 — 패딩
                // 행을 타일 GEMM에 넘기면 블록 단위 처리가 실제 행 결과를 흔든다
                // (해시 국소화로 확인, 2026-09-14). starts = Σ ceil16(cnt).
                let mut starts = vec![0usize; ne + 1];
                let mut accp2 = 0usize;
                for e in 0..ne {
                    starts[e] = accp2;
                    let c = (b[e + 1].max(0) as usize).saturating_sub(b[e].max(0) as usize);
                    accp2 += c.div_ceil(16) * 16;
                }
                starts[ne] = accp2;
                off_d2h = vec![0usize; ne + 1];
                for e in 0..ne {
                    off_d2h[e] = starts[e];
                    off_d2h[e + 1] = starts[e]
                        + (b[e + 1].max(0) as usize).saturating_sub(b[e].max(0) as usize);
                }
            } else {
                // t=1 종전 동작: 비패딩 off를 그대로(gather도 perm_d라 일관).
                off_d2h = b[..ne + 1].iter().map(|&x| x.max(0) as usize).collect();
            }
            &off_d2h
        } else {
            &off
        };
        for e in 0..ne {
            let r = off[e + 1] - off[e];
            if r == 0 {
                continue;
            }
            let start = off[e];
            let xsrc = unsafe { xg.add(start * row_u32 * 4) };
            let wsrc = unsafe { wd.add(e * per_expert) };
            let dst = unsafe { yg.add(start * n_out * 4) };
            if f32w {
                self.launch_gemm_f32(xsrc, wsrc, n_in, n_out, r, dst)?;
            } else {
                self.launch_gemm(ggml_id(ws.ty), xsrc, wsrc, n_in, n_out, xq_w, r, dst)?;
            }
        }
        phase("gemms", &mut lap);
        let scat2 = if pad_layout { inv_pad_d } else { inv_d };
        self.rows_permute_dev(yg, scat2 as *mut u8, op_, n_out, rows)?;
        phase("scatter", &mut lap);
        if tm {
            eprintln!("# moe-phase TOTAL={:.2}ms rows={rows}", t0.elapsed().as_secs_f64() * 1e3);
        }
        self.moe_hash_check("fb", op_, rows, n_out)?;
        Ok(())
    }


}

impl llm170_core::matmul::FrameHost for Q4Acc {

    /// np 행별 conv 1런치 (plans/74 N2) — gdn_conv(t=1) 산술, 상태는 행
    /// 포인터 테이블. qkv/out은 [t][ch] 연속 프레임 버퍼.
    fn frame_gdn_conv_np(
        &self,
        qkv: u64,
        out: u64,
        states: &[u64],
        cw: u64,
        ch: usize,
        k: usize,
    ) -> Result<(), String> {
        let t = states.len();
        if t == 0 {
            return Ok(());
        }
        let mut ptrs: Vec<usize> = Vec::with_capacity(t);
        for &h in states {
            ptrs.push(self.fptr(h)? as usize);
        }
        let tbl = self.ctx.scratch(t * 8)?;
        self.ctx.h2d(tbl, bytemuck::cast_slice(&ptrs))?;
        let (mut q, mut c, mut s_, mut o_) = (
            self.fptr(qkv)?,
            self.fptr(cw)?,
            tbl as *mut std::ffi::c_void,
            self.fptr(out)?,
        );
        let (mut chh, mut kk, mut tt) = (ch as i32, k as i32, t as i32);
        self.kop(
            "gdn_conv_np",
            (ch as u32).div_ceil(64),
            t as u32,
            1,
            64,
            &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk, &mut tt),
        )
    }
    /// np 행별 AR 1런치 (plans/74 N2) — gdn_ar_w_swap(t=1) 산술(scale=1,
    /// q는 L2Rows+Scale 로 선스케일), 상태는 행 포인터 테이블.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar_np(
        &self,
        q: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        out: u64,
        states: &[u64],
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        let t = states.len();
        if t == 0 {
            return Ok(());
        }
        let mut ptrs: Vec<usize> = Vec::with_capacity(t);
        for &h in states {
            ptrs.push(self.fptr(h)? as usize);
        }
        let tbl = self.ctx.scratch(t * 8)?;
        self.ctx.h2d(tbl, bytemuck::cast_slice(&ptrs))?;
        let (mut sp, mut qp, mut kp, mut vp, mut bp, mut op_) = (
            tbl as *mut std::ffi::c_void,
            self.fptr(q)?,
            self.fptr(k)?,
            self.fptr(v)?,
            self.fptr(beta_ge)?,
            self.fptr(out)?,
        );
        let (mut dd, mut ks, mut vs, mut hv, mut hk, mut sc, mut tt) = (
            d as i32,
            (h_k * d) as i32,
            (h_v * d) as i32,
            h_v as i32,
            h_k as i32,
            1.0f32,
            t as i32,
        );
        // gx=h_v(페어 축 — 커널의 blockIdx.x), gy=d(u 축). 27B rawhip 판과
        // 동일 순서(2026-09-16 실수로 (d,h_v)로 바꿔써 GPU 메모리 폴트).
        self.ctx.launch3(
            "gdn_ar_w_np",
            h_v as u32,
            d as u32,
            1,
            32,
            &mut cargs!(&mut sp, &mut qp, &mut kp, &mut vp, &mut bp, &mut op_, &mut dd, &mut ks, &mut vs, &mut hv, &mut hk, &mut sc, &mut tt),
        )
    }
    /// plans/73(np): 프레임 버퍼 행 뷰 — 배치 디코드의 per-seq 상태 op용.
    fn frame_slice(&self, h: u64, off_elems: usize, len: usize) -> Result<u64, String> {
        let mut v = self.frames.lock().map_err(|e| e.to_string())?;
        let idx = (h.checked_sub(1).ok_or("frame 핸들 0")?) as usize;
        let (base, cap) = *v
            .get(idx)
            .ok_or_else(|| format!("frame 핸들 없음: {h}"))?;
        let need = (off_elems + len) * 4;
        if need > cap {
            return Err(format!("frame_slice 범위 초과: need {need} > cap {cap}"));
        }
        let ptr = unsafe { base.add(off_elems * 4) };
        v.push((ptr, len * 4));
        Ok(v.len() as u64)
    }

    fn frame_qk_norm_rope(
        &self,
        q: u64,
        k: u64,
        q_norm: &[f32],
        k_norm: &[f32],
        cs: &[f32],
        eps: f32,
        pos0: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
        t: usize,
    ) -> Result<(), String> {
        // 상수 3개(qn/kn/cs) — plans/73: 매 호출 h2d(+sync)가 스텝당 36회의
        // 동기를 만들었다. **키는 (ptr,len)** — 내용 해시는 층마다 값이 달라
        // 단일 슬롯 캐시가 매 층 미스했고(24KB+2KB 동기 복사 ×12층 = 40ms/스텝),
        // 프레임이 헤드 타일을 1회 만들어 상주시키므로 포인터가 곧 신원이다.
        // (2026-09-16: LLM170_Q4_TIME 계측 — qsa.mm+rope 3.4ms/층의 전부가 이 복사였다)
        let (qnd, knd, csd) = {
            let qnd = self.upload_map(&self.qn_map, "qn_t", q_norm)?;
            let knd = self.upload_map(&self.kn_map, "kn_t", k_norm)?;
            let qh = q_norm.as_ptr() as u64;
            let kh = k_norm.as_ptr() as u64;
            let _ = (qh, kh);
            let cskey = (cs.as_ptr() as usize, cs.len());
            let mut c = (self.cst.lock().map_err(|e| e.to_string())?, self.cst_cache.lock().map_err(|e| e.to_string())?);
            if *c.1 != cskey || c.0.ptr.is_null() {
                c.0.ensure(&self.ctx, cs.len().max(1) * 4)?;
                self.ctx.h2d(c.0.ptr, bytemuck::cast_slice(cs))?;
                *c.1 = cskey;
            }
            (qnd, knd, c.0.ptr)
        };
        let mut qp = self.fptr(q)? as *mut std::ffi::c_void;
        let mut kp = self.fptr(k)? as *mut std::ffi::c_void;
        let mut qwp = qnd as *mut std::ffi::c_void;
        let mut kwp = knd as *mut std::ffi::c_void;
        let mut csp = csd as *mut std::ffi::c_void;
        // kq_scale은 이 커널의 decode 판은 k에 구워 넣지만(kqs=self.kq_scale),
        // QSA 프레임 경로는 **k를 무척도(1.0)로 둔다** — QSA KV 캐시 규약이
        // 무척도 k이고 qsa_attn_sel6가 q·k에 kq_scale을 곱하기 때문. 초기 구현은
        // 0.0을 넘겨 k를 전부 0으로 만드는 잠복 결함이었음(미호출 경로라 미발견,
        // 2026-09-14 plans/67 2c 연결 시 발견·수정).
        let mut kq = 1.0f32;
        let mut ep = eps;
        let mut pp = pos0 as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut nr = n_rot as i32;
        let rows = n_head + n_kv;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut qp) as *mut _ as *mut std::ffi::c_void,
            (&mut kp) as *mut _ as *mut std::ffi::c_void,
            (&mut qwp) as *mut _ as *mut std::ffi::c_void,
            (&mut kwp) as *mut _ as *mut std::ffi::c_void,
            (&mut csp) as *mut _ as *mut std::ffi::c_void,
            (&mut ep) as *mut _ as *mut std::ffi::c_void,
            (&mut kq) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut nr) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)
    }

    // ─── 프레임(활성화 GPU 상주) — plans/64 P1 ───
    // 계약: core `qwen4exp/frame.rs`의 op 순서·산술 그대로. 프레임 경로는
    // 스텝당 동기를 ~14회로 줄인다(값 경로 ~1300회).

    /// 버퍼 할당 — `len`은 **원소 수**(f32 4바이트/u32 1워드). core frame.rs
    /// 규약(`a(k_len)`, `a(v.len())`)을 따른다.
    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        let p = self.ctx.alloc((len.max(4)) * 4)?;
        let mut v = self.frames.lock().map_err(|e| e.to_string())?;
        // cap은 **바이트**다 — frame_slice가 need=(off+len)*4와 비교하고 슬라이스
        // 핸들도 len*4로 적는다. 종전엔 원소 수를 적어 슬라이스 상한이 실제
        // 버퍼의 1/4행이었다(np 1행 뷰는 안 걸렸고 다중 프리필 행 대역이 걸림,
        // 2026-09-17). 검사만 완화되므로 기존 경로는 불변.
        v.push((p, len * 4));
        Ok(v.len() as u64)
    }

    fn frame_free(&self, _h: u64) -> Result<(), String> {
        // 해제 없음 (ADR-0014) — 풀은 영구.
        Ok(())
    }

    fn frame_write(&self, h: u64, data: &[f32]) -> Result<(), String> {
        self.fchk(h, data.len() * 4, "frame_write")?;
        let p = self.fptr(h)?;
        self.ctx.h2d(p, bytemuck::cast_slice(data))
    }

    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        self.fchk(h, data.len() * 4, "frame_write_u32")?;
        let p = self.fptr(h)?;
        self.ctx.h2d(p, bytemuck::cast_slice(data))
    }

    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        self.fchk(h, out.len() * 4, "frame_read")?;
        let p = self.fptr(h)?;
        // 동기 hipMemcpy — 공유 핀 스테이징(d2h 헬퍼)의 재사용 상태에 의존하지
        // 않는다. 프레임 판독은 스텝당 몇 회뿐이라 동기 경로 비용이 무의미하다.
        unsafe {
            ck(
                hip::hipMemcpy(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    p as *const std::ffi::c_void,
                    out.len() * 4,
                    hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                ),
                "frame_read",
            )
        }
    }

    /// KTRACE 진단 훅 — 스텝 단위 덤프+재시작(core→백엔드 의존 방향 존중).
    fn ktrace_tick(&self) {
        eprintln!("{}", crate::rawhip::ktrace_dump());
        crate::rawhip::ktrace_on();
    }

    /// 슬롯 반납 — PLE 링/워터마크 제거(2026-09-16 RCA: reset_seq 이 링을
    /// 못 지워 새 대화가 이전 대화의 n-gram 링을 읽었다 — np4 잔여 비결정성
    /// 및 슬롯 재사용 오염의 원인).
    fn acc_reset_seq(&self, seq: usize) {
        if let Ok(mut m) = self.ple_ring.lock() {
            m.remove(&seq);
        }
        if let Ok(mut wm) = self.ple_ring_pos.lock() {
            wm.remove(&seq);
        }
    }

    /// [t][vocab] logits 행별 GPU argmax — np greedy 판정 (plans/74 N1).
    /// argmax64 = CPU greedy와 동일 의미(동률 최저 인덱스).
    fn frame_argmax_rows(&self, logits: u64, t: usize, vocab: usize) -> Result<Vec<u32>, String> {
        let base = self.fptr(logits)?;
        // 병렬 2단계(plans/74 N3) — argmax64 1블록 판은 vocab 248k 에서
        // ~1.2ms/행 직렬 꼬리(FN np4 KTRACE 4.6ms/step).
        let nblk = (vocab / 4096).clamp(1, 64) as u32;
        let part = self.ctx.scratch(t.max(1) * nblk as usize * 8)?;
        let outb = self.ctx.scratch(t.max(1) * 8 + 8 * nblk as usize * t.max(1))?;
        let out = unsafe { outb.add(t.max(1) * nblk as usize * 8) };
        {
            let mut xp = base as *mut std::ffi::c_void;
            let mut vb = vocab as i32;
            let mut pp = part as *mut std::ffi::c_void;
            let mut nb = nblk as i32;
            let mut args = vec![
                (&mut xp) as *mut _ as *mut std::ffi::c_void,
                (&mut vb) as *mut _ as *mut std::ffi::c_void,
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("argmax_rows_s1", nblk, t.max(1) as u32, 1, 256, &mut args)?;
        }
        {
            let mut pp = part as *mut std::ffi::c_void;
            let mut op = out as *mut std::ffi::c_void;
            let mut nb = nblk as i32;
            let mut args = vec![
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("argmax_rows_s2", t.max(1) as u32, 1, 1, 64, &mut args)?;
        }
        let mut r8 = vec![0u8; t * 8];
        self.ctx.d2h(&mut r8, out)?;
        Ok((0..t)
            .map(|s| {
                let b = &r8[s * 8..s * 8 + 8];
                u32::from_le_bytes([b[4], b[5], b[6], b[7]])
            })
            .collect())
    }

    fn frame_mm(&self, x: u64, w: &llm170_core::matmul::Weight<'_>, out: u64, t: usize) -> Result<(), String> {
        let (xp, op) = (self.fptr(x)?, self.fptr(out)?);
        self.frame_gemm(xp, w, op, t)
    }

    fn frame_mm_group(&self, x: u64, ws: &[llm170_core::matmul::Weight<'_>], outs: &[u64], t: usize) -> Result<(), String> {
        if ws.len() != outs.len() {
            return Err(format!("frame_mm_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        let xp = self.fptr(x)?;
        // 동일 입력 — 양자화 1회 공유 (f32 계열이 섞이면 개별).
        let f32_family = |ty: GgmlType| matches!(ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let f32w = f32_family(ws[0].ty);
        if ws.iter().all(|w| w.n_in == ws[0].n_in && f32_family(w.ty) == f32w) && !f32w {
            // llama MMQ 우선(옵트인) — 같은 입력을 여러 커널이 공유하는 그룹이라
            // 항목별로 MMQ 가능 타입이면 MMQ를 쓰고 나머지는 기존 타일로 간다.
            let mmq_on = t >= 32 && env_on("LLM170_Q4_MMQ");
            let (xq, xq_w) = if mmq_on {
                (std::ptr::null_mut(), 0usize)
            } else {
                self.frame_quant(xp, ws[0].n_in as usize, t)?
            };
            // plans/83 D2: t=1 디코드에서 그룹 내 q8_0 인접쌍을 듀얼 커널로
            // 융합 — 런치 수 절반, 블록 수 합산(점유 개선). dual 커널의 행
            // 산술은 원판 gemm_q8_0과 동일 트리 → 비트 불변.
            let dual_ok = t == 1 && !mmq_on && std::env::var_os("LLM170_NO_DUAL").is_none();
            let mut idx = 0usize;
            while idx < ws.len() {
                let w = &ws[idx];
                let ty = ggml_id(w.ty);
                if dual_ok
                    && ty == 8
                    && idx + 1 < ws.len()
                    && ggml_id(ws[idx + 1].ty) == 8
                {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    let n_in = w.n_in as usize;
                    let no1 = w.n_out as usize;
                    let no2 = ws[idx + 1].n_out as usize;
                    self.gemm_q8_dual(xq, wd1, no1, o1, wd2, no2, o2, n_in)?;
                    idx += 2;
                    continue;
                }
                let (wd, _) = self.dev_weight(w)?;
                let op = self.fptr(outs[idx])?;
                let n_in = w.n_in as usize;
                let n_out = w.n_out as usize;
                if mmq_on && matches!(ty, 12 | 13 | 14 | 23)
                    && self.ctx.gemm_mmq(ty, xp as *const u8, wd, n_in, n_out, t, op).is_ok()
                {
                    idx += 1;
                    continue;
                }
                // f16 경로 미검증(위 frame_gemm 주석 참조) — 배선 보류.
                let (xqi, xwi) = if mmq_on { self.frame_quant(xp, n_in, t)? } else { (xq, xq_w) };
                self.launch_gemm(ty, xqi, wd, n_in, n_out, xwi, t, op)?;
                idx += 1;
            }
            return Ok(());
        }
        // plans/71: q8_0 가중치 + t>=32는 MMQ(int8 dp4a) — f32 활성을 직접 받아
        // 자체 양자화. 종전 j128 타일 대비 측정 이득은 벤치로 검증.
        // 혼합 패밀리 그룹(GDN [q8,q8,f32,f32] 등) — t=1에서 인접 q8_0 쌍을
        // 듀얼로 융합(plans/83 D2). 첫 분기의 동일-패밀리 조건에 걸리지 않는
        // 그룹의 q8_0 쌍도 같은 이득을 받는다. 비트 불변(행 산술 동일).
        // plans/83 D2(계속): f32 인접쌍(β/α)은 f32 듀얼로, (q8,f32) 인접쌍
        // (hc down+inject)은 혼합 듀얼로 — 각 1런치. 행 산술은 소스 커널과
        // 동일 → 비트 불변.
        let f32fam = |ty: GgmlType| matches!(ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let dual_any = t == 1 && std::env::var_os("LLM170_NO_DUAL").is_none();
        let mut idx = 0usize;
        while idx < ws.len() {
            let w = &ws[idx];
            if dual_any && idx + 1 < ws.len() && ws[idx + 1].n_in == w.n_in {
                let a8 = w.ty == GgmlType::Q8_0;
                let b8 = ws[idx + 1].ty == GgmlType::Q8_0;
                let af = f32fam(w.ty);
                let bf = f32fam(ws[idx + 1].ty);
                if a8 && b8 {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    let (xq, _xw) = self.frame_quant(xp, w.n_in as usize, t)?;
                    self.gemm_q8_dual(xq, wd1, w.n_out as usize, o1, wd2, ws[idx + 1].n_out as usize, o2, w.n_in as usize)?;
                    idx += 2;
                    continue;
                }
                if af && bf {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    self.gemm_f32_dual(xp as *const u8, wd1, w.n_out as usize, o1, wd2, ws[idx + 1].n_out as usize, o2, w.n_in as usize)?;
                    idx += 2;
                    continue;
                }
                if a8 && bf {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    let ni = w.n_in as usize;
                    let (xq, _xw) = self.frame_quant(xp, ni, t)?;
                    self.gemm_mix_dual(xq, wd1, w.n_out as usize, o1, xp as u64, wd2, ws[idx + 1].n_out as usize, o2, ni)?;
                    idx += 2;
                    continue;
                }
            }
            if t == 1
                && w.ty == GgmlType::Q8_0
                && idx + 1 < ws.len()
                && ws[idx + 1].ty == GgmlType::Q8_0
                && ws[idx + 1].n_in == w.n_in
                && std::env::var_os("LLM170_NO_DUAL").is_none()
            {
                let (wd1, _) = self.dev_weight(w)?;
                let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                let o1 = self.fptr(outs[idx])?;
                let o2 = self.fptr(outs[idx + 1])?;
                let (xq, xq_w) = self.frame_quant(xp, w.n_in as usize, t)?;
                let _ = xq_w;
                self.gemm_q8_dual(xq, wd1, w.n_out as usize, o1, wd2, ws[idx + 1].n_out as usize, o2, w.n_in as usize)?;
                idx += 2;
                continue;
            }
            let op = self.fptr(outs[idx])?;
            if w.ty == GgmlType::Q8_0 && t >= 32
                && env_eq("LLM170_Q8MMQ", "1")
            {
                let (wd, _) = self.dev_weight(w)?;
                self.ctx
                    .gemm_mmq(8, xp as *const u8, wd, w.n_in as usize, w.n_out as usize, t, op)
                    .map_err(|e| format!("q8mmq: {e}"))?;
                idx += 1;
                continue;
            }
            self.frame_gemm(xp, w, op, t)?;
            idx += 1;
        }
        Ok(())
    }

    /// 상주 elementwise/RoPE/인덱서 연산 — qwen4exp 프레임이 쓰는 변형만 구현.
    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp as O;
        match *op {
            O::SiluDiv { t, div, n } => {
                let mut p = self.fptr(t)?;
                let mut d = div;
                let mut nn = n as i32;
                self.kop("q4_silu_div", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut d, &mut nn))
            }
            O::SiluMul { g, u, out, n } => {
                let (mut gp, mut up, mut op) = (self.fptr(g)?, self.fptr(u)?, self.fptr(out)?);
                let mut nn = n as i32;
                self.kop("silu_mul", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut gp, &mut up, &mut op, &mut nn))
            }
            O::Sigmoid { t, n } => {
                let mut p = self.fptr(t)?;
                let mut nn = n as i32;
                self.kop("q4_sigmoid", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut nn))
            }
            O::RmsRows { x, w, out, eps, n, w_reps } => {
                let (xp, wp) = (self.fptr(x)?, self.fptr(w)?);
                let rows = w_reps * self.t_cur();
                // plans/73: 융합 판은 측정 역행(16.78→16.28 t/s) — 옵트인 자산.
                // 워프=행의 320-원소 직렬 f32 체인이 part/finish 의 병렬 2런치보다 느리다.
                if rows <= 32 && env_on("LLM170_RMSSMALL") {
                    let mut xa = xp;
                    let mut wa = wp;
                    let mut op_ = self.fptr(out)?;
                    let mut e = eps;
                    let mut nn = n as i32;
                    let (mut rws, mut rr) = (rows as i32, w_reps as i32);
                    return self.kop(
                        "rms_small",
                        rows.div_ceil(8) as u32,
                        1,
                        1,
                        256,
                        &mut cargs!(&mut xa, &mut wa, &mut op_, &mut e, &mut nn, &mut rws, &mut rr),
                    );
                }
                let part = {
                    let mut b = self.fpart.lock().map_err(|e| e.to_string())?;
                    b.ensure(&self.ctx, rows * 32 * 8)?
                };
                {
                    let mut xa = xp;
                    let mut pa = part;
                    let mut nn = n as i32;
                    self.kop("rms_part", rows as u32, 1, 1, 32, &mut cargs!(&mut xa, &mut pa, &mut nn))?;
                }
                let mut xa = xp;
                let mut wa = wp;
                let mut pa = part;
                let mut op_ = self.fptr(out)?;
                let mut e = eps;
                let mut nn = n as i32;
                let mut rr = w_reps as i32;
                // 256스레드 = 8 그룹 × 32레인 (raw 디코더와 동일 기하 — 128로
                // 줄이면 행 절반이 미기록)
                self.kop("rms_finish", rows as u32, 1, 1, 256, &mut cargs!(&mut xa, &mut wa, &mut pa, &mut op_, &mut e, &mut nn, &mut rr))
            }
            O::NormGated { o, z, w, out, eps, d, n_h } => {
                let mut op_ = self.fptr(o)?;
                let mut zp = self.fptr(z)?;
                let mut wp = self.fptr(w)?;
                let mut outp = self.fptr(out)?;
                let mut e = eps;
                let mut dd = d as i32;
                let mut nh = n_h as i32;
                let rows = n_h * self.t_cur();
                self.kop("q4_norm_gated_sig", n_h as u32, (rows / n_h.max(1)) as u32, 1, 32, &mut cargs!(&mut op_, &mut zp, &mut wp, &mut outp, &mut e, &mut dd, &mut nh))
            }
            O::L2Rows { x, eps, d, n } => {
                let mut xp = self.fptr(x)?;
                let mut e = eps;
                let mut dd = d as i32;
                // 행 수는 *토큰 수*에서 온다. 버퍼 길이(t_max)를 쓰면 t=1에서도
                // t_max행을 처리해 33.7ms/스텝을 낭비한다(2026-09-14 실측).
                let rows = (n / d).max(1) as u32;
                self.kop("q4_l2_rows", rows, 1, 1, 32, &mut cargs!(&mut xp, &mut e, &mut dd))
            }
            O::Scale { t, s, n } => {
                let mut p = self.fptr(t)?;
                let mut ss = s;
                let mut nn = n as i32;
                self.kop("q4_scale", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut ss, &mut nn))
            }
            O::BcastRows { src, dst, n, rows } => {
                let (mut sp, mut dp) = (self.fptr(src)?, self.fptr(dst)?);
                let (mut nn, mut rr) = (n as i32, rows as i32);
                self.kop("bcast_rows", (n as u32).div_ceil(128), rows as u32, 1, 128, &mut cargs!(&mut sp, &mut dp, &mut nn, &mut rr))
            }
            O::CopyRows { src, dst, src_off, dst_off, n } => {
                let (mut sp, mut dp) = (self.fptr(src)?, self.fptr(dst)?);
                let (mut so, mut dfo) = (src_off as i32, dst_off as i32);
                let mut nn = n as i32;
                self.kop("copy_rows", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut sp, &mut dp, &mut so, &mut dfo, &mut nn))
            }
            O::HcGateMean { xn, gate, out, hc, n } => {
                let (mut xp, mut gp, mut op_) = (self.fptr(xn)?, self.fptr(gate)?, self.fptr(out)?);
                let total = n * self.t_cur();
                let mut h = hc as i32;
                let mut nn = n as i32;
                let mut tt = total as i32;
                self.kop("q4_hc_gate_mean", (total as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut xp, &mut gp, &mut op_, &mut h, &mut nn, &mut tt))
            }
            O::HcCombine { res, out, inj, hc, n, total: _ } => {
                // 커널은 (토큰,차원)당 1스레드 — op의 total(=hc·n·t)을 범위로 쓰면
                // hc배만큼 범위 밖을 쓴다(실측: hc>1에서 폴트). n·t를 쓴다.
                let (mut rp, mut op_, mut ip) = (self.fptr(res)?, self.fptr(out)?, self.fptr(inj)?);
                let tn = n * self.t_cur();
                let mut h = hc as i32;
                let mut nn = n as i32;
                let mut tt = tn as i32;
                self.kop("q4_hc_combine", (tn as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut rp, &mut op_, &mut ip, &mut h, &mut nn, &mut tt))
            }
            O::Split3 { src, d0, d1, d2, n0, n1, n2 } => {
                let (mut sp, mut a0, mut a1, mut a2) = (
                    self.fptr(src)?, self.fptr(d0)?, self.fptr(d1)?, self.fptr(d2)?,
                );
                let (mut x0, mut x1, mut x2) = (n0 as i32, n1 as i32, n2 as i32);
                let total = ((n0 + n1 + n2) * self.t_cur()) as u32;
                self.kop("split3", total.div_ceil(128), 1, 1, 128, &mut cargs!(&mut sp, &mut a0, &mut a1, &mut a2, &mut x0, &mut x1, &mut x2))
            }
            O::GdnBetaG { b, a, dtb, sa, bg, n_h } => {
                let (mut bp, mut ap, mut dp, mut sp, mut gp) = (
                    self.fptr(b)?, self.fptr(a)?, self.fptr(dtb)?, self.fptr(sa)?, self.fptr(bg)?,
                );
                let mut nh = n_h as i32;
                // dt_rank = n_h / t (n_h = dt_rank·t) — t>1에서 n_h를 dt_rank로
                // 넘기면 dtb/sa(길이 dt_rank)를 넘겨 읽어 폴트 (실측 700).
                let mut dr = (n_h / self.t_cur().max(1)) as i32;
                self.kop("gdn_beta_g", (n_h as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut bp, &mut ap, &mut dp, &mut sp, &mut gp, &mut nh, &mut dr))
            }
            O::GdnConv { qkv, cw, state, out, ch, k, t_len } => {
                let (qp, cp, stp, op_) = (
                    self.fptr(qkv)?, self.fptr(cw)?, self.fptr(state)?, self.fptr(out)?,
                );
                if t_len == 1 {
                    let (mut q, mut c, mut s_, mut o_) = (qp, cp, stp, op_);
                    let mut chh = ch as i32;
                    let mut kk = k as i32;
                    return self.kop("gdn_conv", (ch as u32).div_ceil(64), 1, 1, 64, &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk));
                }
                if t_len >= k - 1 {
                    // 완전 병렬 청크판 (전제 t ≥ k-1) + 링 상태 갱신은 별도 커널
                    // (conv_t2는 상태를 갱신하지 않는다 — raw 디코더도 2런치)
                    {
                        let (mut q, mut c, mut s_, mut o_) = (qp, cp, stp, op_);
                        let mut chh = ch as i32;
                        let mut kk = k as i32;
                        let mut tt = t_len as i32;
                        self.kop("gdn_conv_t2", (ch as u32).div_ceil(64), t_len as u32, 1, 64, &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk, &mut tt))?;
                    }
                    let (mut q2, mut s2) = (qp, stp);
                    let mut ch2 = ch as i32;
                    let mut k2 = k as i32;
                    let mut t2 = t_len as i32;
                    return self.kop("gdn_conv_state", (k - 1) as u32, (ch as u32).div_ceil(64), 1, 64, &mut cargs!(&mut q2, &mut s2, &mut ch2, &mut k2, &mut t2));
                }
                // 짧은 꼬리(t < k-1): 토큰별 순차 (t=1 커널 반복, 포인터 전진)
                for ti in 0..t_len {
                    let (mut q, mut c, mut s_, mut o_) = (
                        unsafe { qp.add(ti * ch * 4) },
                        cp,
                        stp,
                        unsafe { op_.add(ti * ch * 4) },
                    );
                    let mut chh = ch as i32;
                    let mut kk = k as i32;
                    self.kop("gdn_conv", (ch as u32).div_ceil(64), 1, 1, 64, &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk))?;
                }
                Ok(())
            }
            O::MoeTop10 { route, ids, wt, n_exp, k_sel } => {
                // 라우팅이 새로 쓰였다 — 그룹화 캐시 무효화.
                self.moe_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (mut rp, mut ip, mut wp) = (self.fptr(route)?, self.fptr(ids)?, self.fptr(wt)?);
                let mut ne = n_exp as i32;
                let mut ks = k_sel as i32;
                let t = self.t_cur();
                // 워프 병렬판(2026-09-13) — 원판은 1스레드/토큰이라 디코드에서
                // 0.85ms/호출이었다(토큰당 48콜 = 41ms). 선택 로직은 동일해
                // 결과는 비트 동일.
                self.kop("q4_moe_top10_m", t as u32, 1, 1, 32, &mut cargs!(&mut rp, &mut ip, &mut wp, &mut ne, &mut ks))
            }
            O::MoeWeightedSum { ys, wt, out, k, n } => {
                let (mut yp, mut wp, mut op_) = (self.fptr(ys)?, self.fptr(wt)?, self.fptr(out)?);
                let mut kk = k as i32;
                let mut nn = (n * self.t_cur()) as i32;
                self.kop("q4_moe_weighted_sum", ((n * self.t_cur()) as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut yp, &mut wp, &mut op_, &mut kk, &mut nn))
            }
            O::AxpyScaled { y, x, s, n } => {
                let (mut yp, mut xp, mut sp) = (self.fptr(y)?, self.fptr(x)?, self.fptr(s)?);
                let mut nn = n as i32;
                let t = self.t_cur();
                if t <= 1 {
                    self.kop("axpy_scaled", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut yp, &mut xp, &mut sp, &mut nn))
                } else {
                    // 토큰 배치: s[t] — per = 토큰당 원소 수
                    let mut pp = (n / t) as i32;
                    self.kop("q4_axpy_scaled_t", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut yp, &mut xp, &mut sp, &mut nn, &mut pp))
                }
            }
            ref other => Err(format!("q4acc: 프레임 op 미지원 {other:?}")),
        }
    }
}
