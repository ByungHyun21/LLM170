//! Q4Acc FrameState 구현 — topk·GDN AR·MoE 그래프 GEMM·이벤트 원장 (plans/129 R7 이동).
use super::*;
use crate::rawhip::env_on;

impl llm170_core::matmul::FrameState for Q4Acc {
    fn frame_topk_cands(
        &self,
        logits: u64,
        t: usize,
        vocab: usize,
    ) -> Result<Vec<(f32, u32)>, String> {
        const NB: u32 = 64; // 후보 블록 수 → 64×8워프 = 512 후보
        const NWARP: usize = 8; // 256스레드 / 32
        let cv = {
            let mut g = self.tk_cand_v.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, t * (NB as usize) * NWARP * 4)?
        };
        let ci = {
            let mut g = self.tk_cand_i.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, t * (NB as usize) * NWARP * 4)?
        };
        let lp = self.fptr(logits)?;
        let (mut l_p, mut cv_p, mut ci_p) = (
            lp as *mut std::ffi::c_void,
            cv as *mut std::ffi::c_void,
            ci as *mut std::ffi::c_void,
        );
        let (mut vv, mut nb) = (vocab as i32, NB as i32);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut l_p) as *mut _ as *mut std::ffi::c_void,
            (&mut cv_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ci_p) as *mut _ as *mut std::ffi::c_void,
            (&mut vv) as *mut _ as *mut std::ffi::c_void,
            (&mut nb) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx
            .launch3("q4_logits_topk_cand", NB, t as u32, 1, 256, &mut args)?;
        let n_cand = t * (NB as usize) * NWARP;
        let mut hv = vec![0.0f32; n_cand];
        let mut hi = vec![0i32; n_cand];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv), cv)?;
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hi), ci)?;
        Ok(hv
            .into_iter()
            .zip(hi)
            .filter(|&(v, i)| i >= 0 && v > -1e29f32)
            .map(|(v, i)| (v, i as u32))
            .collect())
    }

    fn gdn_split_l2_scale(
        &self,
        gconv: u64,
        gq: u64,
        gk: u64,
        gv: u64,
        conv_ch: usize,
        k_len: usize,
        v_len: usize,
        d_state: usize,
        n_group: usize,
        t: usize,
        eps: f32,
    ) -> Result<(), String> {
        let qs = 1.0f32 / (d_state as f32).sqrt();
        let (mut gc, mut q, mut k, mut vv) = (
            self.fptr(gconv)? as *mut std::ffi::c_void,
            self.fptr(gq)? as *mut std::ffi::c_void,
            self.fptr(gk)? as *mut std::ffi::c_void,
            self.fptr(gv)? as *mut std::ffi::c_void,
        );
        let (mut cc, mut kl, mut vl, mut ds, mut ng, mut tt, mut e, mut sc) = (
            conv_ch as i32,
            k_len as i32,
            v_len as i32,
            d_state as i32,
            n_group as i32,
            t as i32,
            eps,
            qs,
        );
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut gc) as *mut _ as *mut std::ffi::c_void,
            (&mut q) as *mut _ as *mut std::ffi::c_void,
            (&mut k) as *mut _ as *mut std::ffi::c_void,
            (&mut vv) as *mut _ as *mut std::ffi::c_void,
            (&mut cc) as *mut _ as *mut std::ffi::c_void,
            (&mut kl) as *mut _ as *mut std::ffi::c_void,
            (&mut vl) as *mut _ as *mut std::ffi::c_void,
            (&mut ds) as *mut _ as *mut std::ffi::c_void,
            (&mut e) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut ng) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "gdn_split_l2_scale",
            n_group as u32,
            t as u32,
            1,
            32,
            &mut args,
        )
    }

    /// plans/115 P1-3 — 상태 D2D 복사(메인 스트림 비동기, 순서 보장).
    /// 접두 체크포인트 캡처/복원.
    fn frame_copy_states(&self, pairs: &[(u64, u64, usize)]) -> Result<(), String> {
        for &(dst, src, bytes) in pairs {
            let d = self.fptr(dst)?;
            let s = self.fptr(src)? as *const u8;
            self.ctx.d2d(d, s, bytes)?;
        }
        Ok(())
    }

    fn frame_begin_np(&self, t: usize) {
        self.cur_t
            .store(t.max(1), std::sync::atomic::Ordering::Relaxed);
        self.np_mode
            .store(true, std::sync::atomic::Ordering::Relaxed);
        prefill_pin(t, true);
    }

    fn frame_end_np(&self) {
        self.np_mode
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn frame_ev_mark(&self, tag: u8) {
        // plans/115 D: 섹션 GPU 벽 계측 — 이벤트 풀 고갈 방지 위해 보고 시 파괴.
        let mut ev: crate::rawhip::hip::hipEvent_t = std::ptr::null_mut();
        unsafe {
            if crate::rawhip::hip::hipEventCreateWithFlags(&mut ev, 0)
                != crate::rawhip::hip::hipError_t_hipSuccess
            {
                return;
            }
            if crate::rawhip::hip::hipEventRecord(ev, self.ctx.stream) != 0 {
                return;
            }
        }
        if let Ok(mut g) = self.ev_marks.lock() {
            g.push((tag, ev as usize));
        }
    }

    fn moe_graph_capable(&self) -> bool {
        !MOE_FALLBACK_USED.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn frame_ev_report(&self) {
        let marks = {
            let Ok(mut g) = self.ev_marks.lock() else {
                return;
            };
            std::mem::take(&mut *g)
        };
        if marks.len() < 2 {
            for (_, e) in &marks {
                unsafe { crate::rawhip::hip::hipEventDestroy(*e as *mut _) };
            }
            return;
        }
        let mut acc = [0.0f64; 5];
        for w in marks.windows(2) {
            let mut dt = 0f32;
            let ok = unsafe {
                crate::rawhip::hip::hipEventElapsedTime(&mut dt, w[0].1 as *mut _, w[1].1 as *mut _)
                    == crate::rawhip::hip::hipError_t_hipSuccess
            };
            if ok {
                acc[w[0].0 as usize] += dt as f64;
            }
        }
        for (_, e) in &marks {
            unsafe { crate::rawhip::hip::hipEventDestroy(*e as *mut _) };
        }
        let tot: f64 = acc.iter().sum();
        eprintln!(
            "[pfev] gdn {:.0} qsa {:.0} moe {:.0} head {:.0} 기타 {:.0} | GPU 벽 {:.0}",
            acc[1], acc[2], acc[3], acc[4], acc[0], tot
        );
    }

    fn set_ctx_len(&self, n: usize) {
        self.ctx_len.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn frame_begin(&self, t: usize) {
        self.cur_t
            .store(t.max(1), std::sync::atomic::Ordering::Relaxed);
        // plans/115 P5: np 디코드는 언핀 — t>1이어도 행별 t=1 패밀리(dmmv/
        // w16, 원장 124 VERIFY_ROW_PIN 산술)로 디스패치된다. 타일 핀은
        // 프리필 청크 불변성(원장 18)을 위한 것 — np 스텝 내부의 fs_begin(t)
        // 재호출도 np_mode 동안 언핀을 유지한다.
        let np = self.np_mode();
        prefill_pin(t, np);
        // plans/84 A/E.2 — 프리필(t>1) 패밀리 핀. 두 결함을 묶는다: (1) hc_attn
        // down(q8_0)의 t=16 GEMV ↔ t>64 j128 타일 갈림(원장 (12)), (2) MoE
        // 폴백의 전문가별 행수 r이 t 키로 쓰여 r>=16 타일/r<16 GEMV로 갈라
        // 같은 (토큰,전문가) 계산이 청킹별로 다른 패밀리 산술을 쓰는 것(원장
        // (18), value.rs launch_gemm 참조). 핀 + 타일 강제로 청크 16..512 전부
        // 비트 동일(chunk-check 3종 PASS, 2026-09-21). 핀은 무조건(원장 18
        // 기본 ON 승격 — =0 복원 경로는 plans/109 P6 삭제).
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
        // gdn_ar_w_swap: 전치 상태 레이아웃(s[dv*d+kdim]) + d=128 고정(레인당
        // kdim 4개). 구 q4_gdn_ar_w의 열 단위 접근은 512B 스트라이드였다.
        self.ctx.launch3(
            "gdn_ar_w_swap",
            d as u32,
            h_v as u32,
            1,
            32,
            &mut cargs!(
                &mut sp, &mut qp, &mut kp, &mut vp, &mut bp, &mut op_, &mut dd, &mut ks, &mut vs,
                &mut hv, &mut hk, &mut sc, &mut tt
            ),
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
        self.kop(
            "q4_moe_gather",
            total.div_ceil(128),
            1,
            1,
            128,
            &mut cargs!(&mut a, &mut b, &mut nn, &mut ks, &mut tt),
        )
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
        self.kop(
            "q4_moe_scatter",
            ((t * n) as u32).div_ceil(128),
            1,
            1,
            128,
            &mut cargs!(&mut a, &mut b, &mut c, &mut ks, &mut nn, &mut tt),
        )
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
        let n_in = ws.n_in as usize;
        let n_out = ws.n_out as usize / n_expert_stack.max(1);
        // 행 수 = t·k_sel — 버퍼는 t_max 크기라 길이에서 유도할 수 없다.
        let rows = self.t_cur() * k_sel.max(1);
        let xp = self.fptr(x)?;
        let op_ = self.fptr(out)?;
        let (wd, f32w) = self.dev_weight(ws)?;
        let per_expert = ws.data.len() / n_expert_stack.max(1);
        let gen_q = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
        // plans/108 P7 (메인 레버): MoE dmmv — f32 활성 직소비 direct-ids GEMM
        // (vk fn_moe_ids2/fn_moe_ids51 이식, q4_K·q5_1). 활성 quant(quant_cache/
        // frame_quant)과 K-분할 reduce·카운팅 정렬 기계를 통째로 건너뛰고
        // 전문가 그룹 GEMM을 런치 1회로 마친다. 킬스위치 LLM170_HIP_DMMV_OFF
        // (기본 ON=사용 — 끄면 종전 ge_ids/w_ids direct-ids 경로로 복귀).
        if crate::common::moe::ids2_takes(rows, self.t_cur(), ws.ty) && !f32w {
            let idp = self.fptr(ids)?;
            let kern: &'static str = if ws.ty == GgmlType::Q4K {
                "q4_gemm_q4k_dmmv_ids"
            } else if ws.ty == GgmlType::Q5K {
                "q5k_gemm_dmmv_ids"
            } else {
                "q5_1_gemm_dmmv_ids"
            };
            let mut x_p = xp as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut o_p = op_ as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let mut tt = rows as i32;
            let mut eb = per_expert as i32;
            let wgs = n_out.div_ceil(2);
            return self.ctx.launch3(
                kern,
                rows as u32,
                wgs.min(65535) as u32,
                wgs.div_ceil(65535) as u32,
                64,
                &mut cargs!(
                    &mut x_p, &mut w_p, &mut o_p, &mut ip, &mut ni, &mut no, &mut tt, &mut eb
                ),
            );
        }
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
                    Some((xp0, r0, g0, _, _)) => {
                        ((*xp0, *r0, *g0) == key, format!("({xp0:#x},{r0},{g0})"))
                    }
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
        // direct-ids(t=1, LLM170_MOE_DIRECT=1): 그룹화 테이블·gather·scatter를
        // 전부 건너뛰고 커널이 ids[row]를 직접 읽는다. 행 순서가 곧 ids 순서라
        // 가중합(ys[e*n+i])이 그대로 맞고, 호스트 왕복(ids d2h+빌드+h2d)도 없다.
        // t=1에서는 k_sel행이 같은 벡터이므로 스트라이드 0으로 0번 행을 읽는다.
        // plans/74: direct-ids 는 행 수만 보면 된다 — np t=4(k_sel×4=40행)도
        // 이 빠른 경로로(그룹화 경로는 프리필 대량 행 전용). rows<=64 게이트로
        if (self.t_cur() == 1 || rows <= 64) && ws.ty == GgmlType::Q4K && !f32w {
            let idp = self.fptr(ids)?;
            // K-분할: 타일 40블록(=1/CU)이던 점유율을 ksplit배로. 부분합은 part에
            // 남기고 reduce가 k 오름차순 합산(결정적, 순서 재결합만 다른 미세 드리프트).
            let ksplit: u32 = 4; // 측정 기본(원장 98 계열) — 노브는 plans/109 P6 삭제
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let part_buf = self.ctx.scratch(rows * n_out * ksplit as usize * 8)?;
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
        //
        // plans/141: `n_in/32 <= 32` 상한을 n_in ≤ 4096으로 완화했다. 커널 인덱스는
        // 이미 그 범위를 안전하게 덮는다 — xq 워드 로드는 `xw = sb*8`로 서브블록
        // 하나당 8 unsigned를 읽어 최대 (n_sub-1)*8+7 = n_in/4-1이므로 32 서브블록
        // 제한이 없어도 q8 영역을 넘지 않는다. 스케일 `xr[n_in/4 + sb]`도
        // xq_words = n/4+n/32+n/16 안이다. 레인 위임은 `sb = lane; sb += 32`로
        // 이미 처리되어 산술 순서는 불변이다(f32 부분합 + 32레인 shfl 트리).
        //
        // 상한 때문에 MTP 드래프트 Q8_0 스택(n_in=2560)이 전부 걸러져 512-전문가
        // 그룹 타일(q4_gemm_q8_gm)로 내려갔다. 그 타일은 **전문가당 격리 블록**을
        // gy=513으로 발사해 드래프트 63스텝에 126회·1058ms를 태웠다(라우팅된
        // 전문가가 10개여도 512 전체를 순회). 직접 ids 판은 rows 블록만 쓴다.
        if (self.t_cur() == 1 || rows <= 64)
            && ws.ty == GgmlType::Q8_0
            && !f32w
            && rows > 0
            && n_in <= 4096
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
            if llm170_diag::dump::opts().key("q8ids_dbg") {
                eprintln!(
                    "# q8ids launch n_in={n_in} n_out={n_out} rows={rows} per_expert={per_expert}"
                );
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
                .filter(|g| {
                    crate::common::moe::cache_hit(g.generation, generation, g.rows, rows, true)
                })
                .map(|g| {
                    (
                        g.perm_d,
                        g.inv_d,
                        g.rowexp_d,
                        g.perm_pad_d,
                        g.inv_pad_d,
                        g.tilexp_d,
                        g.rows_pad,
                        g.rows_pad_d,
                        g.off.clone(),
                        g.off_d,
                        g.pinned_off,
                    )
                })
        };
        let (
            perm_d,
            inv_d,
            rowexp_d,
            perm_pad_d,
            inv_pad_d,
            tilexp_d,
            rows_pad,
            rows_pad_d,
            off,
            off_d,
            pinned_off,
        ) = match hit {
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
                // (plans/115 P5 시도: np 디바이스 그룹화 — np는 dmmv_ids가
                // 그룹화를 통째로 우회해 무의료 했다. 게이트는 t=1 원복.)
                // (plans/115 D — t>1 디바이스 그룹화: 3차 시도까지 진행 후 주차.
                // 배리어 2종은 제거됐으나(moed2h·QSA 항등 업로드) 패딩 도메인
                // gm/ge가 호스트 테이블 대비 5× 느림(타일 수 동일·원인 미상 —
                // 원장 151). t=1로 복귀, 상세 기록.)
                // plans/115 D(원장 153): t>1 디바이스 그룹화 — Q5K 그룹 타일
                // (q4_gemm_q5k_gm) 추가로 전 투영이 디바이스 테이블 경로 →
                // 그래프 캡처 호환. bound 그리드 + 센티널.
                // plans/136 폐기 후 135 재개(2026-10-06): t>1 프리필도 디바이스
                // 그룹핑 활성화 — 라우팅 ids d2h(4KB)가 층당 어텐션+라우터 전체를
                // 드레인(52.2ms×768 = FN pp16384 호스트 바운드의 근원). 소비는
                // j128m WMMA 타일(plans/133, 구형 gm/ge 5× 원장 151 극복) —
                // pad_layout 플래그(plans/115 D 유산)로 패딩 도메인 소비.
                // 게이트(토큰 스트림)+pp16384 A/B로 검증. 실패 시 t==1 복귀.
                // 2026-10-07 plans/141: 실패 확인 — j128m을 꺼도 이 t>1 활성화
                // 상태에서는 FN 게이트가 퇴화했다(수리 후 실측 `0 18 18 ...`).
                // t==1 전용으로 원복. 재활성화는 패딩 도메인 gm/ge 정합 증명 후에만.
                if self.t_cur() == 1 && ne <= 512 && rows > 0 {
                    // **단일 상한**: Σ_e ceil(r_e/16)*16 ≤ rows + 16*ne
                    // (전문가당 ≤15행 패딩). 종전 rows*16+16은 16배 과대였고,
                    // 그 값으로 커널 zero-fill·호스트 버퍼가 어긋나 OOB가 났다.
                    let bound = crate::common::moe::grp_bound(rows, ne);
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
                    self.moe_group_dev(
                        ids, ne, rows, offd, pd, ivd, rxd, ppd, ipd, txd, rpd, bound,
                    )?;
                    if env_on("LLM170_MOE_GCHECK") {
                        // 진단: 디바이스 테이블과 호스트 재계산을 비교(첫 불일치 지점 출력).
                        self.ctx.sync().map_err(|e| e.to_string())?;
                        let mut dev_off = vec![0i32; ne + 2];
                        self.ctx
                            .d2h(bytemuck::cast_slice_mut(&mut dev_off), offd as *const u8)?;
                        let mut dev_perm = vec![0u32; rows];
                        self.ctx
                            .d2h(bytemuck::cast_slice_mut(&mut dev_perm), pd as *const u8)?;
                        let mut dev_rowexp = vec![0u32; bound];
                        self.ctx
                            .d2h(bytemuck::cast_slice_mut(&mut dev_rowexp), rxd as *const u8)?;
                        let idp = self.fptr(ids)?;
                        let mut idv = vec![0u32; rows];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
                        // 호스트 재계산 — common 판(cnt+누적과 동일 값, P13 공용화).
                        let hoff = crate::common::moe::grp_offsets(&idv, ne);
                        let mut bad = 0;
                        for e in 0..=ne {
                            if dev_off[e] as usize != hoff[e] {
                                eprintln!("# gcheck off[{e}] dev={} host={}", dev_off[e], hoff[e]);
                                bad += 1;
                                if bad > 4 {
                                    break;
                                }
                            }
                        }
                        if bad == 0 {
                            let mut cur = hoff[..ne].to_vec();
                            for (i, &e) in idv.iter().enumerate() {
                                let e2 = (e as usize).min(ne - 1);
                                let ppos = cur[e2];
                                cur[e2] += 1;
                                if dev_perm[ppos] as usize != i {
                                    eprintln!(
                                        "# gcheck perm@{ppos} dev={} host={i}",
                                        dev_perm[ppos]
                                    );
                                    bad += 1;
                                    if bad > 4 {
                                        break;
                                    }
                                }
                                if dev_rowexp[ppos] as usize != e2 {
                                    eprintln!(
                                        "# gcheck rowexp@{ppos} dev={} host={e2}",
                                        dev_rowexp[ppos]
                                    );
                                    bad += 1;
                                    if bad > 4 {
                                        break;
                                    }
                                }
                            }
                        }
                        eprintln!(
                            "# gcheck rows={rows} rows_pad_dev={} bad={bad}",
                            dev_off[ne + 1]
                        );
                    }
                    // 폴백(비 Q4K/Q5_1 타입)용 오프셋. 기본은 비동기로 미리 걸어
                    // 소비 시점(층 하단)까지 gate/up GEMM이 지연을 덮는다.
                    // LLM170_MOE_GROUP_SYNC=1이면 즉시 동기(스트림 드레인) —
                    // 호스트 경로와 같은 순서 조건을 만들어 순서 효과를 검정한다.
                    // 이분법: 비동기 예약 자체를 건너뛴다(폴백은 동기 d2h로).
                    // +4B: 오프셋 뒤에 디바이스가 계산한 rows_pad가 붙어 있다(가드용).
                    // (plans/115 env 정리: MOE_GROUP_SYNC/NOD2H 폐기 — 비동기 issue
                    // 기본 경로 승격. 소비 시점 d2h_wait이 순서를 보장한다.)
                    // plans/115 D(원장 152): rp 그리드 — 층 단위 동기 폐지.
                    // 종전 파이프라인도 d2h_wait가 GPU ~1층분(35ms)을 기다려
                    // 28층×35ms=980ms 직렬화. 청크 경계 학습 + 발행만 유지.
                    let (rp_grid, pinned_off_slot) = {
                        // 캡처 중 d2h_issue(정적 이벤트 재기록)는 그래프 교착 —
                        // 스킵(그래프 경로는 rp 학습 불요).
                        if self
                            .ctx
                            .capturing
                            .load(std::sync::atomic::Ordering::Relaxed)
                        {
                            (bound, std::ptr::null_mut())
                        } else {
                            let pinned = self.ctx.d2h_issue((ne + 2) * 4, offd as *const u8)?;
                            let mut pend = self.moe_rp_pending.lock().map_err(|e| e.to_string())?;
                            if !pinned.is_null() {
                                *pend = Some((pinned, ne));
                            }
                            let _ = &self.moe_rp_est;
                            (bound, pinned)
                        }
                    };
                    let mut c = self.moe_group.lock().map_err(|e| e.to_string())?;
                    *c = Some(MoeGroup {
                        generation,
                        rows,
                        perm_d: pd,
                        inv_d: ivd,
                        rowexp_d: rxd,
                        perm_pad_d: ppd,
                        inv_pad_d: ipd,
                        tilexp_d: txd,
                        rows_pad: rp_grid,
                        rows_pad_d: rpd,
                        off_d: offd,
                        pinned_off: pinned_off_slot,
                        off: Vec::new(),
                    });
                    (
                        pd,
                        ivd,
                        rxd,
                        ppd,
                        ipd,
                        txd,
                        rp_grid,
                        rpd,
                        Vec::new(),
                        offd,
                        pinned_off_slot,
                    )
                } else {
                    // 그래프 캡처 경계 — 이 블록은 d2h(라우팅 판독)+호스트 정렬+h2d를
                    // 하므로 캡처 밖이어야 한다(세그먼트 분할점).
                    // plans/115 D(원장 148): 업로드 6종은 h2d_async_m — 종전 각각 풀
                    // sync 종료라 층당 6회 드레인(pf_stage 배당 ~570ms/청크의 본체).
                    // pageable 소스는 호출 시점 스테이징이라 스코프 탈출 안전,
                    // 소비 커널은 같은 스트림 뒤에 발행돼 순서 보장.
                    let idp = self.fptr(ids)?;
                    let mut idv = vec![0u32; rows];
                    // plans/115 D: 그룹화 d2h 대기 직접 계측(원장 149 귀속).
                    let gdt = std::time::Instant::now();
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
                    if llm170_diag::dump::opts().key("moe_time") {
                        eprintln!(
                            "[moed2h] rows={rows} {:.2}ms",
                            gdt.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    // 카운팅 정렬 테이블 — common 공용판(vk 폴백과 바이트 동일, P13).
                    let off = crate::common::moe::grp_offsets(&idv, ne);
                    let perm = crate::common::moe::grp_perm(&idv, ne, &off);
                    let inv = crate::common::moe::grp_inv(&perm);
                    let (pd, ivd, rxd) = {
                        let mut a = self.rperm.lock().map_err(|e| e.to_string())?;
                        let pd = a.ensure(&self.ctx, rows * 4)? as u64;
                        let mut b = self.rperm2.lock().map_err(|e| e.to_string())?;
                        let ivd = b.ensure(&self.ctx, rows * 4)? as u64;
                        let mut c = self.rexp.lock().map_err(|e| e.to_string())?;
                        let rxd = c.ensure(&self.ctx, rows * 4)? as u64;
                        (pd, ivd, rxd)
                    };
                    self.ctx
                        .h2d_async_m(pd as *mut u8, bytemuck::cast_slice(&perm))?;
                    self.ctx
                        .h2d_async_m(ivd as *mut u8, bytemuck::cast_slice(&inv))?;
                    // rowexp: 순열 후 행 p의 전문가 = idv[perm[p]]
                    let mut rowexp = vec![0u32; rows];
                    for p in 0..rows {
                        rowexp[p] = idv[(perm[p] as usize).min(rows - 1)].min((ne - 1) as u32);
                    }
                    self.ctx
                        .h2d_async_m(rxd as *mut u8, bytemuck::cast_slice(&rowexp))?;
                    let (off_pad, rows_pad) = crate::common::moe::grp_padded(&off, ne, 16);
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
                    if llm170_diag::dump::opts().key("moe_pad") {
                        // plans/133 A 진단: 실 라우팅의 패딩 폐기 분포 — 캐시 미스
                        // (층당 첫 투영) 시에만 도달한다. rows_pad/rows 비가
                        // 컴팩트 배치의 회복 상한을 정산한다(원장 1.158 vs
                        // 라우터 텐서 형상 [2560,512] 모순 판별용).
                        let mut under16 = 0usize;
                        let mut seg_max = 0usize;
                        let mut seg_sum = 0usize;
                        for e in 0..ne {
                            let r = off[e + 1] - off[e];
                            if r < 16 {
                                under16 += 1;
                            }
                            seg_max = seg_max.max(r);
                            seg_sum += r;
                        }
                        eprintln!(
                            "# moepad rows={rows} ne={ne} rows_pad={rows_pad} ratio={:.3} under16={under16} seg_max={seg_max} seg_avg={:.1}",
                            rows_pad as f64 / rows.max(1) as f64,
                            seg_sum as f64 / ne.max(1) as f64
                        );
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
                    self.ctx
                        .h2d_async_m(ppd as *mut u8, bytemuck::cast_slice(&perm_pad))?;
                    self.ctx
                        .h2d_async_m(ipd as *mut u8, bytemuck::cast_slice(&inv_pad))?;
                    self.ctx
                        .h2d_async_m(txd as *mut u8, bytemuck::cast_slice(&tilexp))?;
                    let mut c = self.moe_group.lock().map_err(|e| e.to_string())?;
                    *c = Some(MoeGroup {
                        generation,
                        rows,
                        perm_d: pd,
                        inv_d: ivd,
                        rowexp_d: rxd,
                        perm_pad_d: ppd,
                        inv_pad_d: ipd,
                        tilexp_d: txd,
                        rows_pad,
                        rows_pad_d: 0,
                        off_d: 0,
                        pinned_off: std::ptr::null_mut(),
                        off: off.clone(),
                    });
                    (
                        pd,
                        ivd,
                        rxd,
                        ppd,
                        ipd,
                        txd,
                        rows_pad,
                        0u64,
                        off,
                        0u64,
                        std::ptr::null_mut(),
                    )
                }
            }
        };
        let row_u32 = if f32w { n_in } else { xq_w };
        // plans/68 레이아웃 실험 플래그 — t=1의 기존(검증된) 동작은 그대로 두고
        // 프리필 디바이스 그룹화 실험에서만 패딩 도메인 레이아웃을 쓴다.
        let pad_layout = rows_pad_d != 0 && self.t_cur() > 1;
        // 디바이스 그룹화 경로의 GEMM은 t = rows_pad로 x를 읽는다(패딩 행의 출력은
        // scatter가 버리므로 값은 무관, 크기만 rows_pad까지 필요).
        let xbuf_rows = if rows_pad_d != 0 {
            crate::common::moe::grp_bound(rows, ne)
        } else {
            rows
        };
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
            let ybuf_rows = if rows_pad_d != 0 {
                crate::common::moe::grp_bound(rows, ne)
            } else {
                rows
            };
            g.ensure(&self.ctx, ybuf_rows * n_out * 4)?
        };
        let xsrc0 = if f32w { xp } else { xq };
        // plans/116-2: 초기 gather를 지연으로 — gm 계열(q5_1/q5k/q8)은 자체
        // 패딩 gather를 쓰므로 여기서 모아도 버려진다(permute 331회 중 1/3이
        // 이 낭비). ge/fallback만 필요한 시점에 모은다.
        // plans/68 레이아웃 일관화: 디바이스 그룹화(rows_pad_d≠0)는 GEMM이
        // **패딩 도메인**(r < *rows_pad, rowexp=패딩 인덱스)으로 읽는다 — gather도
        // perm_pad/bound행으로. 호스트 경로는 종전대로 비패딩 perm_d/rows.
        let gather_xg = |me: &Self| -> Result<(), String> {
            if me.t_cur() == 0 {
                return Ok(());
            }
            if pad_layout {
                me.rows_permute_dev(
                    xsrc0,
                    perm_pad_d as *mut u8,
                    xg,
                    row_u32,
                    crate::common::moe::grp_bound(rows, ne),
                )
            } else {
                me.rows_permute_dev(xsrc0, perm_d as *mut u8, xg, row_u32, rows)
            }
        };
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
            if self.t_cur() == 1 {
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
            // 2026-10-07 plans/141: 이 자리에 j128m WMMA 그룹 타일(6800ce02)이
            // 있었고, 그 경로만 켜면 FN 출력이 6800ce02가 커밋한 열화 기준선과
            // 16토큰 정확히 일치했다(실측). 규칙 9(ADR-0019)에 따라 커널 ·
            // CO_W32M 등록 · .co 자산을 같은 변경에서 삭제하고 종전 gm 경로를
            // 무조건 실행으로 원복한다.
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
            return Ok(());
        }
        if ws.ty == GgmlType::Q4K && !f32w && rows > 0 {
            gather_xg(self)?;
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
                self.ctx
                    .d2h(bytemuck::cast_slice_mut(&mut xqf), xsrc0 as *const u8)?;
                let mut xf = vec![0u32; rows.min(160) * n_in];
                self.ctx
                    .d2h(bytemuck::cast_slice_mut(&mut xf), xp as *const u8)?;
                let mut rxf = vec![0u32; rows];
                self.ctx
                    .d2h(bytemuck::cast_slice_mut(&mut rxf), rowexp_d as *const u8)?;
                let mut pmf = vec![0u32; rows];
                self.ctx
                    .d2h(bytemuck::cast_slice_mut(&mut pmf), perm_d as *const u8)?;
                let mut ivf = vec![0u32; rows];
                self.ctx
                    .d2h(bytemuck::cast_slice_mut(&mut ivf), inv_d as *const u8)?;
                eprintln!(
                    "# moedump rows={rows} x_h={:016x} xq_h={:016x} rx_h={:016x} pm_h={:016x} iv_h={:016x}",
                    fnv(&xf),
                    fnv(&xqf),
                    fnv(&rxf),
                    fnv(&pmf),
                    fnv(&ivf)
                );
            }
            // 2026-10-07 plans/141: 여기 j128m WMMA 그룹 타일(6800ce02) 분기가
            // 있었고 그 경로가 FN 출력을 퇴화시켰다 — 같은 변경에서 커널 ·
            // CO_W32M 등록 · .co 자산까지 삭제(규칙 9/ADR-0019). 아래 ge 경로가
            // 이 타입의 유일한 소비자다.
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
            self.moe_hash_check("ge", op_, rows, n_out)?;
            return Ok(());
        }
        // 디바이스 그룹화 경로(off 비어 있음): 폴백(비 Q4K/Q5_1 타입, 예: down의
        // q8_0)은 전문가별 런치를 위해 오프셋만 읽는다 — 그룹화·테이블 빌드·업로드
        // 왕복은 GPU가 이미 끝냈으므로 여기서는 (ne+1)개 int만 받는다.
        let mut off_d2h;
        // plans/68: 디바이스 그룹화 경로의 xg는 **패딩 도메인** — 폴백(전문가별
        // 런치)도 패딩 오프셋(off_pad)에서 구간을 읽어야 행이 맞는다.
        // plans/115 D(원장 153): Q5_K 그룹 타일 — q4_gemm_q5_1_gm과 동일 패딩
        // 도메인 구조. 폴백(호스트 오프셋) 대신 디바이스 tilexp를 써 그래프
        // 캡처 호환(FN의 up_exps가 Q5K — 원장 153의 마지막 관문).
        // plans/116-2: q5_K·q8_0 전문가도 그룹 타일로 — 종전 rows_pad_d≠0(디바이스
        // 그룹화 t=1 전용) 게이트 때문에 프리필(호스트 테이블)은 전문가별 폴백
        // (j128/v4, pp512에서 336+43ms)로 돌았다. 호스트 테이블(perm_pad·tilexp·
        // rows_pad)은 동일 패딩 도메인을 제공하므로 그대로 소비한다.
        if ws.ty == GgmlType::Q5K
            && !f32w
            && rows > 0
            && (rows_pad_d != 0 || (tilexp_d != 0 && rows_pad > 0))
        {
            let ygp = {
                let mut g = self.gyp.lock().map_err(|e| e.to_string())?;
                g.ensure(
                    &self.ctx,
                    rows_pad.max(crate::common::moe::grp_bound(rows, ne)) * n_out * 4,
                )?
            };
            // 입력: 디바이스 경로는 xg가 이미 패딩 도메인. 호스트 경로는
            // perm_pad으로 자체 패딩 gather(gxp).
            let xin = if rows_pad_d != 0 {
                xg
            } else {
                let xgp = {
                    let mut g = self.gxp.lock().map_err(|e| e.to_string())?;
                    g.ensure(&self.ctx, rows_pad * row_u32 * 4)?
                };
                self.rows_permute_dev(xsrc0, perm_pad_d as *mut u8, xgp, row_u32, rows_pad)?;
                xgp
            };
            let per_expert_w = n_out * ((n_in >> 8) * 44); // 전문가 전체 워드
            let smem = (16 * ((n_in >> 8) * 44) * 4) as u32; // 16행 × wrow × 4B
            let (mut x_p, mut w_p, mut o_p, mut tx_p) = (
                xin as *mut std::ffi::c_void,
                wd as *mut std::ffi::c_void,
                ygp as *mut std::ffi::c_void,
                tilexp_d as *mut std::ffi::c_void,
            );
            let (mut ni, mut no, mut xw, mut tt, mut ew) = (
                n_in as i32,
                n_out as i32,
                row_u32 as i32,
                rows_pad as i32,
                per_expert_w as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                self.ctx.scratch(4)? as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut tx_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ew) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3_dyn(
                "q4_gemm_q5k_gm",
                n_out.div_ceil(16).min(65535) as u32,
                rows_pad.div_ceil(16) as u32,
                1,
                256,
                smem,
                &mut args,
            )?;
            let scat = inv_pad_d;
            self.rows_permute_dev(ygp, scat as *mut u8, op_, n_out, rows)?;
            return Ok(());
        }
        // plans/115 D: 폴백 도달 기록 — 그래프 캡처 호환성 판정(웜업 청크 설정).
        // plans/115 D(원장 154): Q8_0 그룹 타일 — 그래프 캡처 호환(폴백 회피).
        if ws.ty == GgmlType::Q8_0
            && !f32w
            && rows > 0
            && (rows_pad_d != 0 || (tilexp_d != 0 && rows_pad > 0))
        {
            let ygp = {
                let mut g = self.gyp.lock().map_err(|e| e.to_string())?;
                g.ensure(
                    &self.ctx,
                    rows_pad.max(crate::common::moe::grp_bound(rows, ne)) * n_out * 4,
                )?
            };
            let xin = if rows_pad_d != 0 {
                xg
            } else {
                let xgp = {
                    let mut g = self.gxp.lock().map_err(|e| e.to_string())?;
                    g.ensure(&self.ctx, rows_pad * row_u32 * 4)?
                };
                self.rows_permute_dev(xsrc0, perm_pad_d as *mut u8, xgp, row_u32, rows_pad)?;
                xgp
            };
            let per_expert_b = n_out * (n_in >> 5) * 34; // 바이트/전문가
            let (mut x_p, mut w_p, mut o_p, mut tx_p) = (
                xin as *mut std::ffi::c_void,
                wd as *mut std::ffi::c_void,
                ygp as *mut std::ffi::c_void,
                tilexp_d as *mut std::ffi::c_void,
            );
            let (mut ni, mut no, mut xw, mut tt, mut eb) = (
                n_in as i32,
                n_out as i32,
                row_u32 as i32,
                rows_pad as i32,
                per_expert_b as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                self.ctx.scratch(4)? as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut tx_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut eb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_gemm_q8_gm",
                n_out.div_ceil(16).min(65535) as u32,
                rows_pad.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            )?;
            let scat = inv_pad_d;
            self.rows_permute_dev(ygp, scat as *mut u8, op_, n_out, rows)?;
            return Ok(());
        }
        gather_xg(self)?;
        MOE_FALLBACK_USED.store(true, std::sync::atomic::Ordering::Relaxed);
        if llm170_diag::flag::on("LLM170_PF_GRAPH_DEBUG") {
            eprintln!(
                "# pfgraph fallback: ty={:?} f32w={f32w} rows={rows} rpd={}",
                ws.ty,
                rows_pad_d != 0
            );
        }
        let off: &[usize] = if off.is_empty() && rows_pad_d != 0 {
            self.ctx.d2h_wait()?;
            // 오프셋 + 그 뒤 4B(디바이스가 계산한 rows_pad)를 함께 읽어 **상한을
            // 검증**한다. 초과하면 크래시(h2d 700) 대신 진단 메시지로 실패시킨다 —
            // 2026-09-14에 상한 가정이 5곳에 흩어져 있어 디버깅이 오래 걸렸다.
            let mut b = vec![0i32; ne + 2];
            if !pinned_off.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pinned_off as *const u8,
                        b.as_mut_ptr() as *mut u8,
                        (ne + 2) * 4,
                    );
                }
            } else {
                self.ctx
                    .d2h(bytemuck::cast_slice_mut(&mut b), off_d as *const u8)?;
            }
            let _ = &b;
            let rows_pad_dev = b[ne + 1].max(0) as usize;
            let bound = crate::common::moe::grp_bound(rows, ne);
            if llm170_diag::dump::opts().key("moe_bcheck") {
                // b(pinned off) 무결성 — r 오염(gemm_q5k gx=1.04억)의 원본 관찰.
                let mut mono_ok = true;
                for i in 0..ne {
                    if b[i] > b[i + 1] {
                        mono_ok = false;
                        break;
                    }
                }
                let total = b[ne];
                if !mono_ok
                    || total < 0
                    || total as usize > bound
                    || b[..ne.min(8)].iter().any(|&x| x < 0)
                {
                    eprintln!(
                        "# bcheck BAD rows={rows} ne={ne} mono={mono_ok} total={total} bound={bound} b0..7={:?} rp={}",
                        &b[..8.min(ne)],
                        b[ne + 1]
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
                    off_d2h[e + 1] =
                        starts[e] + (b[e + 1].max(0) as usize).saturating_sub(b[e].max(0) as usize);
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
        let scat2 = if pad_layout { inv_pad_d } else { inv_d };
        self.rows_permute_dev(yg, scat2 as *mut u8, op_, n_out, rows)?;
        self.moe_hash_check("fb", op_, rows, n_out)?;
        Ok(())
    }

    /// 명시 no-op(107 P0-3): hip frame_moe_gemm은 quant_cache로 GEMM
    /// 시점 자체 정량 — 사전 팩 버퍼를 소비하지 않는다. 트레이트 기본이
    /// Err이므로 이 오버라이드는 "불필요함을 확인한 응답"이지 조용한
    /// 성공 보고가 아니다.
    fn frame_quant_pack(&self, _x: u64, _rows: usize, _n_in: usize) -> Result<(), String> {
        Ok(())
    }
}
