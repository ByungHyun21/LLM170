//! vkacc::matmul — FrameHost·MatmulHost — 호스트 스테이징 GEMM. (plans/90 B1b: gemv.rs 순수 이동)

use super::*;

impl llm170_core::matmul::FrameHost for VkAcc {
    fn ktrace_tick(&self) {
        self.ts_tick();
    }
    /// plans/84 B: 프레임 op군이 부분 구현(엘리먼트와이스+MoE) — 완성 전에는
    /// 옵트인(LLM170_VK_FRAME=1)일 때만 엔진이 프레임 경로에 들어온다.
    /// plans/86 §8 — 프레임 경로 완성(§1 정확성·§2 QSA 디바이스화·§5 성능) 후
    fn frame_capable(&self) -> bool {
        true // 원장 30 기본 ON — =0 킬스위치는 plans/109 P6 삭제
    }
    /// plans/85 §2 — 프레임 로짓 행별 argmax: fn_argmax_rows 2단 판.
    /// 동률 최저 인덱스 — CPU greedy_from과 동일 의미. 미구현이면 greedy
    /// 디코드 전체가 값경로 재연산으로 폴백했다(np/forward/multi 공통).
    fn frame_argmax_rows(&self, logits: u64, t: usize, vocab: usize) -> Result<Vec<u32>, String> {
        let lb = self.fbuf(logits)?;
        // WG당 256스레드×8원소 = 2048. stage1은 1워크그룹(256) 축소 — n_wg ≤ 256.
        let n_wg = vocab.div_ceil(2048);
        if n_wg > 256 || vocab == 0 || t == 0 {
            return Err(format!(
                "vk frame_argmax_rows: 형상 초과 n_wg={n_wg} vocab={vocab} t={t}"
            ));
        }
        let sc_bytes = 2 * n_wg * t * 4;
        let out_bytes = t * 4;
        let mut ctx = self.ctx.lock();
        // plans/88 P1 — 스텝 배치 플러시: 아래 2런치는 비배치 동기 실행 후
        // 호스트가 ob.ptr 을 직접 판독한다. 녹화만 된 커맨드를 먼저 실행.
        if ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            ctx.end_batch_wait()?;
        }
        {
            let mut g = self.argmax_bufs.lock();
            let need = match g.as_ref() {
                None => true,
                Some(b) => b.0.bytes < sc_bytes || b.1.bytes < out_bytes,
            };
            if need {
                *g = Some(crate::rawvk::context::site::scope("argmax", || {
                    Ok::<_, String>((
                        ctx.alloc_host(sc_bytes.max(1 << 16))?,
                        ctx.alloc_host(out_bytes.max(4096))?,
                    ))
                })?);
            }
        }
        let (scb, ob) = {
            let g = self.argmax_bufs.lock();
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone())
        };
        let p = self.pipeline(&mut ctx, Slot::FnArgmaxRows)?;
        let ds = ctx.bind_ds(&p, &[lb, scb.buf, ob.buf])?;
        let push0 = push_u32s(&[vocab as u32, 0u32, n_wg as u32]);
        ctx.run(p.pl, ds, p.pipe, &push0, n_wg as u32, t as u32, 1)?;
        let push1 = push_u32s(&[0u32, 1u32, n_wg as u32]);
        ctx.run(p.pl, ds, p.pipe, &push1, 1, t as u32, 1)?;
        // 비배치 run은 동기 — 안전한 직접 판독.
        let mut out = vec![0u32; t];
        // SAFETY (107 W8): ob 매핑 판독 — 비배치 run은 동기; t원소는 argmax 결과 버퍼 크기와 일치.
        unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const u32, out.as_mut_ptr(), t) };
        Ok(out)
    }

    /// plans/86 §2 — QSA q/k norm+rope in-place (hip qk_norm_rope 동일열).
    /// 상수(qn/kn 헤드 타일, cs 테이블)는 (ptr,len) 키로 1회 업로드 상주 —
    /// 프레임이 타일 Vec을 스텝 간 유지하므로 포인터가 곧 신원이다(hip 교훈).
    /// kq_scale=1.0(QSA 무척도 k 규약). hd ≤ 256(공유 스테이징 폭).
    #[allow(clippy::too_many_arguments)]
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
        if hd > 256 {
            return Err(format!("vk frame_qk_norm_rope: hd={hd} (≤256 전용)"));
        }
        let mut ctx = self.ctx.lock();
        let qnb = self.qk_const(&mut ctx, q_norm)?;
        let knb = self.qk_const(&mut ctx, k_norm)?;
        let csb = self.qk_const(&mut ctx, cs)?;
        let (qb, kb) = (self.fbuf(q)?, self.fbuf(k)?);
        let p = self.pipeline(&mut ctx, Slot::FnQkNormRope)?;
        let ds2 = ctx.bind_ds(&p, &[qb, kb, qnb.buf, knb.buf, csb.buf])?;
        // PC 선언순: eps, kqs, pos, n_head, n_kv, hd, n_rot.
        let mut push = eps.to_le_bytes().to_vec();
        push.extend_from_slice(&1.0f32.to_le_bytes());
        push.extend_from_slice(&push_u32s(&[
            pos0 as u32,
            n_head as u32,
            n_kv as u32,
            hd as u32,
            n_rot as u32,
        ]));
        ctx.run(
            p.pl,
            ds2,
            p.pipe,
            &push,
            (n_head + n_kv) as u32,
            t as u32,
            1,
        )
    }

    /// 프레임 버퍼 — host-visible(alloc_host)로 직접 읽기/쓰기.
    /// 값경로 버퍼와 동일 정책(plans/29).
    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        // (비풀 직할당 경로는 plans/115 env 정리로 삭제 — 풀 재활용이 기본)
        let need = len * 4;
        // 풀에서 최소 적합 버퍼 재활용 (할당 syscall·vk 객체 회피).
        let recycled = {
            let mut pool = self.frame_pool.lock();
            pool.iter()
                .enumerate()
                .filter(|(_, b)| b.bytes >= need)
                .min_by_key(|(_, b)| b.bytes)
                .map(|(i, _)| i)
                .map(|i| pool.swap_remove(i))
        };
        let b = match recycled {
            Some(b) => b,
            None => {
                let mut ctx = self.ctx.lock();
                crate::rawvk::context::site::scope("frame", || ctx.alloc_host(need))?
            }
        };
        let h = self
            .frame_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.framebufs.lock().insert(h, b);
        Ok(h)
    }
    fn frame_free(&self, h: u64) -> Result<(), String> {
        let b = self.framebufs.lock().remove(&h);
        if let Some(b) = &b {
            // plans/87 §4 — 풀 반납 기록(사이트별 재사용 재고).
            llm170_diag::alloc::recycle("frame", b.bytes);
        }
        if let Some(b) = b {
            let mut pool = self.frame_pool.lock();
            let total: usize = pool.iter().map(|b| b.bytes).sum();
            if total + b.bytes <= FRAME_POOL_CAP {
                pool.push(b);
            }
            // 상한 초과분은 종전대로 보류(파괴 없음) — 풀이 총량을 막는다.
        }
        Ok(())
    }
    fn frame_write(&self, h: u64, data: &[f32]) -> Result<(), String> {
        let (ptr, _) = {
            let g = self.framebufs.lock();
            let b = g.get(&h).ok_or("vk frame_write: 핸들 없음")?;
            (b.ptr, std::marker::PhantomData::<()>)
        };
        // SAFETY (107 W8): frame_write — 핸들로 조회한 버퍼가 살아 있고 data.len()은 호출부 계약상 그 크기 이내; 프레임 기록(제출 전) 구간.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len()) };
        Ok(())
    }
    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        let ptr = self
            .framebufs
            .lock()
            .get(&h)
            .ok_or("vk frame_write_u32: 핸들 없음")?
            .ptr;
        // SAFETY (107 W8): frame_write_u32 — 동일 계약: 핸들 버퍼 생존 + 길이 일치, 쓰기 전용.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u32, data.len()) };
        Ok(())
    }
    fn frame_sync(&self) {
        // 비배치 run()은 호출마다 제출+펜스 대기(동기) — 추가 대기 불필요.
        // 배치 모드(값경로 begin_batch)에서만 플러시한다.
        let mut ctx = self.ctx.lock();
        if ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = ctx.end_batch_wait();
        }
        // 107 W1.5-1: 이중버퍼 모드는 end_batch_wait이 즉시 반환 —
        // 매핑 판독(frame_read) 전 보류 제출 완료가 필수.
        let _ = ctx.wait_pending();
    }
    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        self.frame_sync();
        let ptr = self
            .framebufs
            .lock()
            .get(&h)
            .ok_or("vk frame_read: 핸들 없음")?
            .ptr;
        // SAFETY (107 W8): frame_read — frame_sync가 wait_pending으로 보류 제출 완료 보장; out.len()은 버퍼 크기 이내.
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), out.len()) };
        Ok(())
    }
    /// 상주 GEMM: 프레임 f32 버퍼 → (디바이스) quant → gemv. 업/다운 없음.
    fn frame_mm(&self, x: u64, w: &Weight, out: u64, t: usize) -> Result<(), String> {
        self.frame_mm_group_ex(x, std::slice::from_ref(w), &[out], t, false, false)
    }
    fn frame_mm_group(&self, x: u64, ws: &[Weight], outs: &[u64], t: usize) -> Result<(), String> {
        self.frame_mm_group_ex(x, ws, outs, t, false, false)
    }
    /// plans/104 — 격리 quant 판(공유전문가): xq2 전용 버퍼로 라우팅 xq 와
    /// 독립 병행.
    fn frame_mm_group_sep(
        &self,
        x: u64,
        ws: &[Weight],
        outs: &[u64],
        t: usize,
    ) -> Result<(), String> {
        self.frame_mm_group_ex(x, ws, outs, t, false, true)
    }
    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp as O;
        let t_cur = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        match *op {
            O::RmsRows {
                x,
                w,
                out,
                eps,
                n,
                w_reps,
            } => {
                let (xb, wb, ob) = (self.fbuf(x)?, self.fbuf(w)?, self.fbuf(out)?);
                let rows = w_reps * t_cur;
                // plans/92 P4.1: 대형 t는 256스레드 판(rms_wide) — t=1 디코드는
                // 32스레드 원판(산술 그대로, 실측 우위).
                let resf16 = llm170_core::qwen4exp::frame::res_f16_on();
                let slot = if rows >= 2 {
                    if resf16 {
                        Slot::RmsWideF16
                    } else {
                        Slot::RmsWide
                    }
                } else {
                    Slot::Rms
                };
                // plans/101 P2 잔여 — f16 out16 판은 HCF16과 함께 폐기
                // (107 W1: 이득 실측 없음·승격 미실증). 셰이더 푸시 레이아웃
                // 호환을 위해 플래그 자리는 유지(항상 0).
                let p = self.pipeline(&mut ctx, slot)?;
                let ds2 = ctx.bind_ds(&p, &[xb, wb, ob])?;
                let mut push = push_u32s(&[n as u32, rows as u32, w_reps as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                push.extend_from_slice(&0u32.to_le_bytes());
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    rows as u32,
                    1,
                    1,
                    &[xb, wb],
                    &[ob],
                )?;
            }
            O::SiluDiv { t, div, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::SiluDiv)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let mut push = push_u32s(&[n as u32]);
                push.extend_from_slice(&div.to_le_bytes());
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                    &[tb],
                    &[tb],
                )?;
            }
            O::SiluMul { g, u, out, n } => {
                self.last_silu_out
                    .store(out, std::sync::atomic::Ordering::Relaxed);
                let (gb, ub, ob) = (self.fbuf(g)?, self.fbuf(u)?, self.fbuf(out)?);
                // plans/96 G3 — down 직결 융합: glu로 학습된 출력이면 silu+quant를
                // xq에 직접 기록(쌍 슬롯 fill) — f32 mglu 왕복·down quant 폐지.
                // 디코드(ids2 f32 직결)는 f32 mglu 필요 → t≥2만.
                let t_now = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
                let glu = *self.moe_glu.lock();
                // silu+quant 융합 승격 — =0 복원 plans/109 P6 삭제
                let fused = t_now >= 2
                    && glu.is_some_and(|(h, n_in)| h == out && n % n_in == 0 && n / n_in >= 2);
                if fused {
                    let (_, n_in) = glu.unwrap();
                    let rows = n / n_in;
                    let xq_w = xq_words(n_in);
                    let p = self.pipeline(&mut ctx, Slot::SiluMulQ8)?;
                    let push = push_u32s(&[n_in as u32, rows as u32, xq_w as u32]);
                    let tgt = {
                        let pair_gen = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
                        let mut sl = self.moe_xq_pair.lock();
                        let need = rows * xq_w * 4;
                        let ok = sl.as_ref().is_some_and(|v| v.4.bytes >= need);
                        if !ok {
                            *sl = Some((0, 0, 0, pair_gen, ctx.alloc(need)?));
                        }
                        let v = sl.as_mut().unwrap();
                        v.0 = out;
                        v.1 = n_in;
                        v.2 = rows;
                        v.3 = pair_gen;
                        v.4.buf
                    };
                    let ds2 = ctx.bind_ds(&p, &[gb, ub, tgt])?;
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        ((n_in / 32) + 63) as u32 / 64,
                        rows as u32,
                        1,
                        &[gb, ub],
                        &[tgt],
                    )?;
                } else {
                    let (p, ds2, push) = {
                        let p = self.pipeline(&mut ctx, Slot::Silu)?;
                        let ds2 = ctx.bind_ds(&p, &[gb, ub, ob])?;
                        (p, ds2, push_u32s(&[n as u32]))
                    };
                    let rds: Vec<vk::Buffer> = vec![gb, ub];
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        (n.div_ceil(2) as u32)
                            .div_ceil(256)
                            .max((n as u32).div_ceil(256)),
                        1,
                        1,
                        &rds,
                        &[ob],
                    )?;
                }
            }
            O::Scale { t, s, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::Scale)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let mut push = push_u32s(&[n as u32]);
                push.extend_from_slice(&s.to_le_bytes());
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                    &[tb],
                    &[tb],
                )?;
            }
            O::CopyRows {
                src,
                dst,
                src_off,
                dst_off,
                n,
            } => {
                let (sb, db) = (self.fbuf(src)?, self.fbuf(dst)?);
                let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
                let ds2 = ctx.bind_ds(&p, &[sb, db])?;
                let push = push_u32s(&[n as u32, src_off as u32, dst_off as u32]);
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                    &[sb],
                    &[db],
                )?;
            }
            O::BcastRows { src, dst, n, rows } => {
                let (sb, db) = (self.fbuf(src)?, self.fbuf(dst)?);
                let p = self.pipeline(&mut ctx, Slot::BcastRows)?;
                let ds2 = ctx.bind_ds(&p, &[sb, db])?;
                let push = push_u32s(&[n as u32, rows as u32]);
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                    &[sb],
                    &[db],
                )?;
            }
            O::AxpyScaled { y, x, s, n } => {
                let (yb, xb, sb) = (self.fbuf(y)?, self.fbuf(x)?, self.fbuf(s)?);
                if t_cur <= 1 {
                    let p = self.pipeline(&mut ctx, Slot::AxpyT)?; // pp=n → s[0]와 동일
                    let ds2 = ctx.bind_ds(&p, &[yb, xb, sb])?;
                    let push = push_u32s(&[n as u32, n as u32]);
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        (n as u32).div_ceil(256),
                        1,
                        1,
                        &[yb, xb, sb],
                        &[yb],
                    )?;
                } else {
                    let pp = n / t_cur;
                    let p = self.pipeline(&mut ctx, Slot::AxpyT)?;
                    let ds2 = ctx.bind_ds(&p, &[yb, xb, sb])?;
                    let push = push_u32s(&[n as u32, pp as u32]);
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        (n as u32).div_ceil(256),
                        1,
                        1,
                        &[yb, xb, sb],
                        &[yb],
                    )?;
                }
            }
            O::HcGateMean {
                xn,
                gate,
                out,
                hc,
                n,
            } => {
                let (xb, gb, ob) = (self.fbuf(xn)?, self.fbuf(gate)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::HcGateMean)?;
                let ds2 = ctx.bind_ds(&p, &[xb, gb, ob])?;
                let total = n * t_cur;
                // 107 W1: f16 플래그(bit0 h16·bit1 xn16)는 HCF16/MOEH16
                // 폐기로 항상 0 — 셰이더 푸시 레이아웃 호환 유지.
                let flags = 0u32;
                let push = push_u32s(&[hc as u32, n as u32, total as u32, flags]);
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (total as u32).div_ceil(128),
                    1,
                    1,
                    &[xb, gb],
                    &[ob],
                )?;
            }
            O::HcCombine {
                res,
                out,
                inj,
                hc,
                n,
                total: _,
            } => {
                let (rb, ob, ib) = (self.fbuf(res)?, self.fbuf(out)?, self.fbuf(inj)?);
                let tn = n * t_cur;
                // plans/103: res_hc f16 버스 — RMW 변형 슬롯(페어 소유).
                let resf16 = llm170_core::qwen4exp::frame::res_f16_on();
                let slot = if resf16 {
                    Slot::HcCombineF16
                } else {
                    Slot::HcCombine
                };
                let p2 = self.pipeline(&mut ctx, slot)?;
                let ds3 = ctx.bind_ds(&p2, &[rb, ob, ib])?;
                let push = push_u32s(&[hc as u32, n as u32, tn as u32]);
                ctx.run_rw(
                    p2.pl,
                    ds3,
                    p2.pipe,
                    &push,
                    (tn as u32).div_ceil(128),
                    1,
                    1,
                    &[rb, ob, ib],
                    &[rb],
                )?;
            }
            O::NormGated {
                o,
                z,
                w,
                out,
                eps,
                d,
                n_h,
            } => {
                let (ob, zb, wb, ub) =
                    (self.fbuf(o)?, self.fbuf(z)?, self.fbuf(w)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::NormGatedSig)?;
                let ds2 = ctx.bind_ds(&p, &[ob, zb, wb, ub])?;
                let mut push = push_u32s(&[d as u32, n_h as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n_h * t_cur) as u32,
                    1,
                    1,
                    &[ob, zb, wb],
                    &[ub],
                )?;
            }
            O::GdnBetaG {
                b,
                a,
                dtb,
                sa,
                bg,
                n_h,
            } => {
                let (bb, ab, db, sb, gb) = (
                    self.fbuf(b)?,
                    self.fbuf(a)?,
                    self.fbuf(dtb)?,
                    self.fbuf(sa)?,
                    self.fbuf(bg)?,
                );
                let p = self.pipeline(&mut ctx, Slot::GdnBetaG)?;
                let ds2 = ctx.bind_ds(&p, &[bb, ab, db, sb, gb])?;
                let dr = n_h / t_cur.max(1);
                let push = push_u32s(&[n_h as u32, dr as u32]);
                // 판은 64스레드 — 128로 나누면 절반이 미기입된다(청크 크기별
                // 커버리지가 달라져 청크 불변성 위반의 원인이었다).
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n_h as u32).div_ceil(64),
                    1,
                    1,
                    &[bb, ab, db, sb],
                    &[gb],
                )?;
            }
            O::Sigmoid { t, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::EwSigmoid)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let push = push_u32s(&[n as u32]);
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                    &[tb],
                    &[tb],
                )?;
            }
            O::Split3 {
                src,
                d0,
                d1,
                d2,
                n0,
                n1,
                n2,
            } => {
                let (sb, a0, a1, a2) = (
                    self.fbuf(src)?,
                    self.fbuf(d0)?,
                    self.fbuf(d1)?,
                    self.fbuf(d2)?,
                );
                let p = self.pipeline(&mut ctx, Slot::Split3)?;
                let ds2 = ctx.bind_ds(&p, &[sb, a0, a1, a2])?;
                let push = push_u32s(&[n0 as u32, n1 as u32, n2 as u32]);
                let total = (n0 + n1 + n2) * t_cur;
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (total as u32).div_ceil(64),
                    1,
                    1,
                    &[sb],
                    &[a0, a1, a2],
                )?;
            }
            O::L2Rows { x, eps, d, n } => {
                let xb = self.fbuf(x)?;
                let p = self.pipeline(&mut ctx, Slot::L2Rows)?;
                let ds2 = ctx.bind_ds(&p, &[xb])?;
                let mut push = push_u32s(&[d as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                let rows = (n / d).max(1);
                ctx.run_rw(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1, &[xb], &[xb])?;
            }
            O::L2Rows2Scale {
                q,
                k,
                eps,
                scale,
                d,
                n_group,
            } => {
                let (qb, kb) = (self.fbuf(q)?, self.fbuf(k)?);
                let p = self.pipeline(&mut ctx, Slot::L2Rows2Scale)?;
                let ds2 = ctx.bind_ds(&p, &[qb, kb])?;
                let mut push = push_u32s(&[d as u32, n_group as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                push.extend_from_slice(&scale.to_le_bytes());
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    n_group as u32,
                    1,
                    1,
                    &[qb, kb],
                    &[qb, kb],
                )?;
            }
            O::GdnConv {
                qkv,
                cw,
                state,
                out,
                ch,
                k,
                t_len,
            } => {
                let (qb, cb, sb, ob) = (
                    self.fbuf(qkv)?,
                    self.fbuf(cw)?,
                    self.fbuf(state)?,
                    self.fbuf(out)?,
                );
                let binds3 = [qb, cb, sb, ob];
                if t_len >= k - 1 {
                    // 병렬 청크판 + 상태 갱신 2런치 (hip과 동일 구조)
                    let p = self.pipeline(&mut ctx, Slot::GdnConvT2)?;
                    let ds2 = ctx.bind_ds(&p, &binds3)?;
                    let push = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        (ch as u32).div_ceil(64),
                        t_len as u32,
                        1,
                        &[qb, cb, sb],
                        &[ob],
                    )?;
                    let p2 = self.pipeline(&mut ctx, Slot::GdnConvState)?;
                    let ds3 = ctx.bind_ds(&p2, &[qb, sb])?;
                    let push2 = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run_rw(
                        p2.pl,
                        ds3,
                        p2.pipe,
                        &push2,
                        (k - 1) as u32,
                        (ch as u32).div_ceil(64),
                        1,
                        &[qb],
                        &[sb],
                    )?;
                } else {
                    // 짧은 꼬리: 순차판 (상태 회전 포함)
                    let p = self.pipeline(&mut ctx, Slot::GdnConvSeq)?;
                    let ds2 = ctx.bind_ds(&p, &binds3)?;
                    let push = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        (ch as u32).div_ceil(64),
                        1,
                        1,
                        &[qb, cb, sb],
                        &[ob, sb],
                    )?;
                }
            }
            O::MoeTop10 {
                route,
                ids,
                wt,
                n_exp,
                k_sel,
            } => {
                let (rb, ib, wb) = (self.fbuf(route)?, self.fbuf(ids)?, self.fbuf(wt)?);
                let p = self.pipeline(&mut ctx, Slot::MoeTop10)?;
                let ds2 = ctx.bind_ds(&p, &[rb, ib, wb])?;
                let push = push_u32s(&[n_exp as u32, k_sel as u32]);
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    t_cur as u32,
                    1,
                    1,
                    &[rb],
                    &[ib, wb],
                )?;
                // plans/88 P2 — 라우팅 세대 증가: 그룹화 캐시 무효화 키.
                self.moe_gen
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            O::MoeWeightedSum { ys, wt, out, k, n } => {
                let (yb, wb, ob) = (self.fbuf(ys)?, self.fbuf(wt)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::MoeWsum)?;
                let ds2 = ctx.bind_ds(&p, &[yb, wb, ob])?;
                let total = n * t_cur;
                let push = push_u32s(&[n as u32, k as u32, total as u32]);
                ctx.run_rw(
                    p.pl,
                    ds2,
                    p.pipe,
                    &push,
                    (total as u32).div_ceil(256),
                    1,
                    1,
                    &[yb, wb],
                    &[ob],
                )?;
            }
            ref other => return Err(format!("vk frame_op: 미지원 {other:?}")),
        }
        Ok(())
    }
}

impl llm170_core::matmul::MatmulHost for VkAcc {
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let ty = match vk_ty(w.ty) {
            Some(t) => t,
            None => {
                llm170_core::matmul::matmul_batch(xs, w, outs);
                return Ok(());
            }
        };
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let t = xs.len();
        if llm170_diag::flag::on("LLM170_VK_MBDBG") {
            eprintln!(
                "[mb] ty={:?} n_in={} n_out={} t={}",
                w.ty, w.n_in, w.n_out, t
            );
        }
        let xq_w = xq_words(n_in);
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        let ob = self.value_buf(&mut ctx, &self.obuf, t * n_out * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        // plans/89 — t≥2 q8_0/q4_K는 밀집 coopmat 타일로(dense_mm와 동일
        // 판·동일 수치 클래스). gemv3 t-루프는 512토큰에서 ~50ms 직렬.
        if t >= 2 && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K) && wbufs.len() == 1 {
            // MBTILE+CM 승격 — =0 복원 plans/109 P6 삭제
            let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xq);
            binds.push(ob);
            let big = t >= 128;
            let slot = match (w.ty, big) {
                (GgmlType::Q8_0, true) => Slot::TileQ8128Cm,
                (GgmlType::Q8_0, false) => Slot::TileQ8msCm,
                (_, true) => Slot::TileQ4k128Cm,
                (_, false) => Slot::TileQ4kmsCm,
            };
            let p = self.pipeline(&mut ctx, slot)?;
            let ds2 = ctx.bind_ds(&p, &binds)?;
            let gx = (n_out as u32).div_ceil(64);
            if big {
                // plans/92 P1: 단일 디스패치(슬래브 x, 행 y) — 커널 유도 tok_base.
                let gys = (t as u32).div_ceil(128);
                let push =
                    push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, 0u32, 0u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, gys, gx, 1)?;
            } else {
                for tb in (0..t).step_by(64) {
                    let nt = (t - tb).min(64) as u32;
                    let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, tb as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, gx, 1, 1)?;
                }
            }
            self.download_out(outs, n_out, t);
            return Ok(());
        }
        self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
        self.download_out(outs, n_out, t);
        Ok(())
    }

    /// 같은 입력 그룹: 업로드+양자화 1회 → GEMV 각각 → 개별 다운로드.
    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        if ws.iter().any(|w| vk_ty(w.ty).is_none()) || ws.iter().any(|w| w.n_in != ws[0].n_in) {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                self.matmul_batch(xs, w, out)?;
            }
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        let t = xs.len();
        let xq_w = xq_words(n_in);
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        // 배치: 모든 가중 GEMV 녹화 → 단일 제출 → 일괄 다운로드 (plans/19)
        let do_batch = std::env::var_os("LLM170_VK_NOBATCH").is_none();
        if do_batch {
            ctx.begin_batch()?;
        }
        let mut hosts: Vec<(*mut u8, usize, usize)> = Vec::with_capacity(ws.len()); // (ptr, n_out, ti스트라이드)
        for (wi, w) in ws.iter().enumerate() {
            let ty = vk_ty(w.ty).unwrap();
            let n_out = w.n_out as usize;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            // 가중별 독립 출력 버퍼 (그룹 세션 슬롯)
            let ob = {
                let mut g = self.gobufs.lock();
                while g.len() <= wi {
                    g.push(None);
                }
                if g[wi]
                    .as_ref()
                    .map(|b| b.bytes >= t * n_out * 4)
                    .unwrap_or(false)
                {
                    g[wi].as_ref().unwrap().buf
                } else {
                    let b = ctx.alloc_host(t * n_out * 4)?;
                    let buf = b.buf;
                    g[wi] = Some(b);
                    buf
                }
            };
            self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
            let ptr = self.gobufs.lock()[wi].as_ref().unwrap().ptr;
            hosts.push((ptr, n_out, wi));
        }
        if do_batch {
            ctx.end_batch_wait()?;
        }
        for (ptr, n_out, wi) in hosts {
            // SAFETY (107 W8): 판독 — do_batch면 직전 end_batch_wait, 아니면 동기 run; t*n_out 원소.
            let host = unsafe { std::slice::from_raw_parts(ptr as *const f32, t * n_out) };
            for ti in 0..t {
                outs[wi][ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
            }
        }
        Ok(())
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        let xs = vec![x.to_vec()];
        let mut tmp = vec![vec![0.0f32; w.n_out as usize]];
        self.matmul_batch(&xs, w, &mut tmp)?;
        out.copy_from_slice(&tmp[0]);
        Ok(())
    }
}

impl VkAcc {
    /// plans/101 P1 — hout 플래그 확장 그룹 GEMM(트레이트 외부).
    fn frame_mm_group_ex(
        &self,
        x: u64,
        ws: &[Weight],
        outs: &[u64],
        t: usize,
        hout: bool,
        xq_sep: bool,
    ) -> Result<(), String> {
        let n_in = ws[0].n_in as usize;
        let xq_w = xq_words(n_in);
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        let xb = self.fbuf(x)?;
        // plans/88 P1 — F32·BF16 멤버는 fn_mm_f32 가 프레임 f32 를 직접 소비.
        // 양자 멤버가 있을 때만 quant 를 돌린다(순수 f32 그룹의 스텝 낭비 제거).
        let dense_ty = |ty: llm170_gguf::GgmlType| -> Option<u32> {
            match ty {
                llm170_gguf::GgmlType::F32 => Some(0u32),
                llm170_gguf::GgmlType::Bf16 => Some(1u32),
                _ => None,
            }
        };
        let has_quant = ws.iter().any(|w| vk_ty(w.ty).is_some());
        let xq = if has_quant {
            let xq = if xq_sep {
                self.xq2_dev_buf(&mut ctx, t * xq_w * 4)?
            } else {
                self.xq_dev_buf(&mut ctx, t * xq_w * 4)?
            };
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xb, xq])?;
            let push = push_u32s(&[n_in as u32, t as u32, xq_w as u32]);
            ctx.run_rw(
                p.pl,
                ds2,
                p.pipe,
                &push,
                ((n_in / 32) + 63) as u32 / 64,
                t as u32,
                1,
                &[xb],
                &[xq],
            )?;
            xq
        } else {
            vk::Buffer::null()
        };
        let mut mmgrp_skip: Vec<bool> = vec![false; ws.len()];
        if t < 16 && !ws.is_empty() && ws.len() <= 8 {
            let dty0 = dense_ty(ws[0].ty);
            let single_chunk: Vec<bool> = ws
                .iter()
                .map(|w| {
                    self.weight_bufs(&mut ctx, w)
                        .map(|b| b.len() == 1)
                        .unwrap_or(false)
                })
                .collect();
            let groupable = dty0.is_some()
                && ws.iter().enumerate().all(|(wi, w)| {
                    dense_ty(w.ty) == dty0 && w.n_in as usize == n_in && single_chunk[wi]
                });
            // 혼합 그룹: f32 멤버만 부분 그룹화(양자 멤버는 기존 경로).
            let (use_group, gw_idx): (bool, Vec<usize>) = if groupable {
                (true, (0..ws.len()).collect())
            } else if dty0.is_some() {
                let sub: Vec<usize> = ws
                    .iter()
                    .enumerate()
                    .filter(|(wi, w)| {
                        dense_ty(w.ty) == dty0 && w.n_in as usize == n_in && single_chunk[*wi]
                    })
                    .map(|(wi, _)| wi)
                    .collect();
                (sub.len() >= 2, sub)
            } else {
                (false, Vec::new())
            };
            if use_group {
                for &wi in &gw_idx {
                    mmgrp_skip[wi] = true;
                }
                let ibuf = self.ensure_grp_info(&mut ctx)?;
                let mut info = [0u32; 32];
                let mut binds: Vec<vk::Buffer> = Vec::with_capacity(10);
                let mut total_rows = 0u32;
                for (gi, &wi) in gw_idx.iter().enumerate() {
                    let w = &ws[wi];
                    let ob = self.fbuf(outs[wi])?;
                    let a = ctx.buffer_va(ob);
                    info[2 * gi] = a as u32;
                    info[2 * gi + 1] = (a >> 32) as u32;
                    info[16 + gi] = w.n_out as u32;
                    info[24 + gi] = if dty0 == Some(0) { n_in } else { n_in / 2 } as u32;
                    total_rows += w.n_out as u32;
                    let wbufs = self.weight_bufs(&mut ctx, w)?;
                    binds.push(wbufs[0]);
                }
                while binds.len() < 8 {
                    binds.push(binds[0]);
                }
                binds.push(xb);
                binds.push(ibuf.buf);
                unsafe {
                    let ip = ibuf.ptr as *mut u32;
                    std::ptr::copy_nonoverlapping(info.as_ptr(), ip, 32);
                }
                let _ = &ibuf;
                let p = self.pipeline(&mut ctx, Slot::MmF32bGrp)?;
                let ds2 = ctx.bind_ds(&p, &binds)?;
                let push = push_u32s(&[n_in as u32, t as u32, dty0.unwrap(), gw_idx.len() as u32]);
                let mut grp_ob: Vec<vk::Buffer> = Vec::new();
                for &wi in &gw_idx {
                    grp_ob.push(self.fbuf(outs[wi])?);
                }
                ctx.run_rw(
                    p.pl, ds2, p.pipe, &push, total_rows, t as u32, 1, &binds, &grp_ob,
                )?;
            }
        }
        let mut xs: Vec<Vec<f32>> = Vec::new();
        let mut need_pullback = false;
        for w in ws {
            if vk_ty(w.ty).is_none() && dense_ty(w.ty).is_none() {
                need_pullback = true;
                break;
            }
        }
        if need_pullback {
            if llm170_diag::flag::on("LLM170_VK_PULLDBG") {
                eprintln!(
                    "[pull] n_in={n_in} tys={:?}",
                    ws.iter().map(|w| format!("{:?}", w.ty)).collect::<Vec<_>>()
                );
            }
            // plans/84 B: 미지원 타입(f32 inject 등)은 값경로 폴백 — 프레임 f32를
            // 판독해 MatmulHost(CPU 포함)로 계산하고 out에 기록한다.
            drop(ctx);
            let mut flat = vec![0f32; t * n_in];
            self.frame_read(x, &mut flat)?;
            for ti in 0..t {
                xs.push(flat[ti * n_in..(ti + 1) * n_in].to_vec());
            }
            for (wi, w) in ws.iter().enumerate() {
                let mut outs_v = vec![vec![0f32; w.n_out as usize]; t];
                if vk_ty(w.ty).is_some() {
                    // 지원 타입도 여기선 일괄 값경로(호모지니어스 경로 유지)
                    self.matmul_batch(&xs, w, &mut outs_v)?;
                } else {
                    llm170_core::matmul::matmul_batch(&xs, w, &mut outs_v);
                }
                let mut flat_out = Vec::with_capacity(t * w.n_out as usize);
                for row in &outs_v {
                    flat_out.extend_from_slice(row);
                }
                self.frame_write(outs[wi], &flat_out)?;
            }
            return Ok(());
        }
        for (wi, w) in ws.iter().enumerate() {
            if mmgrp_skip[wi] {
                continue;
            }
            let n_out = w.n_out as usize;
            let ob = self.fbuf(outs[wi])?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            match vk_ty(w.ty) {
                Some(ty) => {
                    if llm170_diag::flag::on("LLM170_VK_MMDBG") {
                        eprintln!(
                            "[mm] ty={ty} n_in={n_in} n_out={n_out} t={t} bytes={}",
                            w.data.len()
                        );
                    }
                    // plans/88 P2 — 프리필(t≥2) 밀집 타일: gemv3 t-루프는
                    // 실측 ~5GB/s(gemv 6.6s/208tok). 타일(K-슬라이스 스테이징)로
                    // 대체 — 산술 클래스는 동일 표현식·스레드 직렬 누산.
                    // plans/89 P0.2 — 디코드(t<16) 밀집 GEMV를 llama dmmv
                    // 포트(q8b/q4b)로: f32 활성 직결(quant 불필요), 64스레드
                    // 2행 WG. [ts] 기준선 gemv 77ms/step — 272-329GB/s급으로
                    if t < 16 && self.gemv8_dense(&mut ctx, &wbufs, n_in, n_out, t, ty, xb, ob)? {
                        continue;
                    }
                    let dense_tile = t >= 2
                        && match w.ty {
                            GgmlType::Q8_0 => true,
                            GgmlType::Q4K => n_in <= 4096,
                            GgmlType::Q5_1 => n_in <= 2048,
                            _ => false,
                        };
                    if dense_tile {
                        let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
                        let chunk_words = (ctx.max_ssbo / 4) as u32;
                        let mut binds: Vec<vk::Buffer> = wbufs.clone();
                        while binds.len() < 8 {
                            binds.push(dbuf);
                        }
                        binds.push(xq);
                        binds.push(ob);
                        // plans/89 P1.1 — coopmat 타일 우선(q8_0/q4_K 밀집):
                        // decoder ms/128 패밀리(f16 coopMatMulAdd) 직접 재사용.
                        // 스칼라 K-슬라이스 타일은 ALU 바운드([ts] tile_q8
                        if matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K) && wbufs.len() == 1 {
                            let big = t >= 128;
                            let slot = match (w.ty, big) {
                                (GgmlType::Q8_0, true) => Slot::TileQ8128Cm,
                                (GgmlType::Q8_0, false) => Slot::TileQ8msCm,
                                (_, true) => Slot::TileQ4k128Cm,
                                (_, false) => Slot::TileQ4kmsCm,
                            };
                            let p = self.pipeline(&mut ctx, slot)?;
                            let ds2 = ctx.bind_ds(&p, &binds)?;
                            let gx = (n_out as u32).div_ceil(64);
                            if llm170_diag::flag::on("LLM170_T8_LOG") {
                                use std::sync::atomic::{AtomicU64, Ordering};
                                static N: AtomicU64 = AtomicU64::new(0);
                                eprintln!(
                                    "[t8log] #{:?} ty={:?} n_in={n_in} n_out={n_out} t={t}",
                                    N.fetch_add(1, Ordering::Relaxed),
                                    w.ty
                                );
                            }
                            if big {
                                // plans/92 P1: 128 패밀리 단일 디스패치(슬래브 x,
                                // 행 y) — 커널이 tok_base=wg.x*BN·꼬리 nt 유도.
                                // 종전 순차 t/128 디스패치의 슬래브별 전 가중
                                // 재판독 폐지.
                                // plans/105 P1: 스키니 coopmat K-분할판 — 원판
                                // 타일 형상 유지 + z=K슬라이스(점유 치유).
                                // 부분합 f32 → FnKsred 결정론 축소. 승격 전
                                // 드리프트 게이트 전례 적용.
                                if w.ty == GgmlType::Q8_0
                                    && t >= 128
                                    && !hout
                                    && n_out
                                        <= std::env::var("LLM170_VK_Q8KS_MAX")
                                            .ok()
                                            .and_then(|v| v.parse::<usize>().ok())
                                            // plans/105: 기본 512 — 0.28nat 드리프트
                                            // (f32s 밴드)·skinny −19%·스킵 20스텝 불변.
                                            // 킬스위치 =0.
                                            .unwrap_or(512)
                                {
                                    let ks: u32 = 8;
                                    let gys8 = (t as u32).div_ceil(128);
                                    let need = ks as usize * t * n_out * 4;
                                    let scr = {
                                        let mut g = self.ks_scratch.lock();
                                        if g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
                                            g.as_ref().unwrap().buf
                                        } else {
                                            let b = ctx.alloc(need)?;
                                            *g = Some(b);
                                            g.as_ref().unwrap().buf
                                        }
                                    };
                                    binds.push(scr);
                                    let pk = self.pipeline(&mut ctx, Slot::TileQ8128Ks)?;
                                    let dsk = ctx.bind_ds(&pk, &binds)?;
                                    let pushk = push_u32s(&[
                                        n_in as u32,
                                        n_out as u32,
                                        xq_w as u32,
                                        t as u32,
                                        0u32,
                                        ks,
                                    ]);
                                    ctx.run_rw(
                                        pk.pl,
                                        dsk,
                                        pk.pipe,
                                        &pushk,
                                        gys8,
                                        gx,
                                        ks,
                                        &binds,
                                        &[scr],
                                    )?;
                                    let pr = self.pipeline(&mut ctx, Slot::FnKsred)?;
                                    let dsr = ctx.bind_ds(&pr, &[ob, scr])?;
                                    let n_tot = (t * n_out) as u32;
                                    let pushr = push_u32s(&[n_tot, ks]);
                                    ctx.run_rw(
                                        pr.pl,
                                        dsr,
                                        pr.pipe,
                                        &pushr,
                                        n_tot.div_ceil(128),
                                        1,
                                        1,
                                        &[scr],
                                        &[ob],
                                    )?;
                                    continue;
                                }
                                let gys = (t as u32).div_ceil(128);
                                // plans/101 P1: hout=1 → outv packed f16(HC gate 축).
                                let push = push_u32s(&[
                                    n_in as u32,
                                    n_out as u32,
                                    xq_w as u32,
                                    t as u32,
                                    0u32,
                                    u32::from(hout),
                                ]);
                                ctx.run_rw(p.pl, ds2, p.pipe, &push, gys, gx, 1, &binds, &[ob])?;
                            } else {
                                for tb in (0..t).step_by(64) {
                                    let nt = (t - tb).min(64) as u32;
                                    let push = push_u32s(&[
                                        n_in as u32,
                                        n_out as u32,
                                        xq_w as u32,
                                        nt,
                                        tb as u32,
                                    ]);
                                    ctx.run_rw(p.pl, ds2, p.pipe, &push, gx, 1, 1, &binds, &[ob])?;
                                }
                            }
                            // plans/95 P3 계측: q8128 형상 수집(1회성).
                            if llm170_diag::flag::on("LLM170_Q8_TRACE") {
                                static N8: std::sync::atomic::AtomicUsize =
                                    std::sync::atomic::AtomicUsize::new(0);
                                let n = N8.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if n < 12 {
                                    eprintln!(
                                        "[q8128] #{n} n_in={n_in} n_out={n_out} t={t} big={big}"
                                    );
                                }
                            }
                            continue;
                        }
                        match w.ty {
                            GgmlType::Q8_0 => {
                                let p = self.pipeline(&mut ctx, Slot::FnTileQ8)?;
                                let ds2 = ctx.bind_ds(&p, &binds)?;
                                // PC: n_in, n_out, chunk_words, xq_w, t.
                                let push = push_u32s(&[
                                    n_in as u32,
                                    n_out as u32,
                                    chunk_words,
                                    xq_w as u32,
                                    t as u32,
                                ]);
                                ctx.run_rw(
                                    p.pl,
                                    ds2,
                                    p.pipe,
                                    &push,
                                    n_out.div_ceil(16) as u32,
                                    t.div_ceil(16) as u32,
                                    1,
                                    &binds,
                                    &[ob],
                                )?;
                            }
                            _ => {
                                let slot = if w.ty == GgmlType::Q4K {
                                    Slot::FnMoeTileQ4K
                                } else {
                                    Slot::FnMoeTileQ51
                                };
                                let p = self.pipeline(&mut ctx, slot)?;
                                let mut b2 = binds.clone();
                                b2.push(dbuf); // rowexp
                                b2.push(dbuf); // rows_pad
                                b2.push(dbuf); // perm_pad
                                let ds2 = ctx.bind_ds(&p, &b2)?;
                                // PC: n_in, n_out, per_expert(0), chunk_words, xq_w, mode=1, t.
                                let push = push_u32s(&[
                                    n_in as u32,
                                    n_out as u32,
                                    0u32,
                                    chunk_words,
                                    xq_w as u32,
                                    1u32,
                                    t as u32,
                                ]);
                                ctx.run_rw(
                                    p.pl,
                                    ds2,
                                    p.pipe,
                                    &push,
                                    n_out.div_ceil(16) as u32,
                                    t.div_ceil(16) as u32,
                                    1,
                                    &binds,
                                    &[ob],
                                )?;
                            }
                        }
                        continue;
                    }
                    self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
                }
                None => {
                    // plans/88 P1 — f32/BF16 밀식 GEMV(값폴백 소거).
                    let dty = dense_ty(w.ty).unwrap();
                    let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
                    // plans/89 P0.2 — 디코드(t<16)는 64스레드 판(mm_f32b):
                    // fn_mm_f32 256스레드 f64 트리는 512WG 지연바운드
                    // ([ts] 12ms/step = 0.4GB/s급). 킬스위치 LLM170_VK_MMB=0.
                    // plans/89 P1.2 — f32/BF16 프리필(t≥2) 타일: fn_mm_f32 그리드
                    // (n_out, t)의 가중 t-재판독(라우터 2.6GB/청크) 소거.
                    // 킬스위치 LLM170_VK_FT32=0.
                    if t >= 2 && wbufs.len() == 1 {
                        // plans/93: tile_f32_w는 실측 역행(558ms vs 352ms) — 원판 유지.
                        // plans/95 P1 — 스키니 f32(n_out ≤ 512): tile_f32는
                        // n_out=4(hc down)에서 WG 32개·활성 128스레드로 점유
                        // 붕괴 → K-분할판 f32s pp512 290 t/s(+27%).
                        // §8 판정(원장 40): 로짓 0.2-0.35nat 드리프트(근접타이
                        // 아님) — 사용자 승인으로 기준 스트림 재기록 후 기본
                        // 승격(2026-09-25, f64 참조에 더 근사·llama.cpp와 동일
                        // 클래스의 실행치 양자화). 킬스위치 =0.
                        if (dty == 0 || dty == 1)   // plans/105 잔여: bf16 f16입력 분기 폐지
                            && n_out <= std::env::var("LLM170_VK_FT32S_MAX")
                                .ok()
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(512)
                        {
                            let p = self.pipeline(&mut ctx, Slot::FnTileF32s)?;
                            let mut binds: Vec<vk::Buffer> = wbufs.clone();
                            // W0u(slot1)에도 동일 버퍼 — BF16 uint 뷰.
                            while binds.len() < 8 {
                                binds.push(if binds.len() == 1 { wbufs[0] } else { dbuf });
                            }
                            binds.push(xb);
                            binds.push(ob);
                            let ds2 = ctx.bind_ds(&p, &binds)?;
                            let push =
                                push_u32s(&[n_in as u32, n_out as u32, t as u32, dty, n_in as u32]);
                            ctx.run_rw(
                                p.pl,
                                ds2,
                                p.pipe,
                                &push,
                                (n_out as u32).div_ceil(16),
                                (t as u32).div_ceil(16),
                                1,
                                &binds,
                                &[ob],
                            )?;
                            continue;
                        }
                        let p = self.pipeline(&mut ctx, Slot::FnTileF32)?;
                        let mut binds: Vec<vk::Buffer> = wbufs.clone();
                        while binds.len() < 8 {
                            binds.push(dbuf);
                        }
                        binds.push(xb);
                        binds.push(ob);
                        let ds2 = ctx.bind_ds(&p, &binds)?;
                        let wpr = if dty == 0 { n_in } else { n_in / 2 };
                        let push =
                            push_u32s(&[n_in as u32, n_out as u32, t as u32, dty, wpr as u32]);
                        ctx.run_rw(
                            p.pl,
                            ds2,
                            p.pipe,
                            &push,
                            (n_out as u32).div_ceil(16),
                            (t as u32).div_ceil(16),
                            1,
                            &binds,
                            &[ob],
                        )?;
                        // plans/95 계측(1회성): tile_f32 형상 수집 — 76MiB f32·bf16
                        // 텐서에 359.6ms/청크의 원인 국소화.
                        continue;
                    }
                    let slot = if t < 16 && wbufs.len() == 1 {
                        Slot::MmF32b
                    } else {
                        Slot::FnMmf32
                    };
                    let p = self.pipeline(&mut ctx, slot)?;
                    let mut binds: Vec<vk::Buffer> = wbufs.clone();
                    while binds.len() < 8 {
                        binds.push(dbuf);
                    }
                    binds.push(xb);
                    binds.push(ob);
                    let ds2 = ctx.bind_ds(&p, &binds)?;
                    let push = push_u32s(&[
                        n_in as u32,
                        n_out as u32,
                        t as u32,
                        dty,
                        (ctx.max_ssbo / 4) as u32,
                    ]);
                    ctx.run_rw(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        n_out as u32,
                        t as u32,
                        1,
                        &binds,
                        &[ob],
                    )?;
                }
            }
        }
        Ok(())
    }
}
