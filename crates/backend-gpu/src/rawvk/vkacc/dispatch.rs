//! vkacc::dispatch — 값경로 디스패치 — gemv/quant/mmv 계열. (plans/90 B1b: gemv.rs 순수 이동)

use super::*;

impl VkAcc {
    /// rms_norm 오프로드 — f32 세그먼트+f64 결합 (CPU sq_sum 미러와 동일 순서).
    pub fn rms_norm_gpu(
        &self,
        xs: &[Vec<f32>],
        w: &[f32],
        eps: f32,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        let n = w.len();
        let mut ctx = self.ctx.lock();
        {
            let mut b = self.rbufs.lock();
            if b.is_none() {
                let xb = ctx.alloc_host((t * n * 4).max(1 << 21))?;
                let wb = ctx.alloc_host(n * 4)?;
                let ob = ctx.alloc_host((t * n * 4).max(1 << 21))?;
                *b = Some((xb, wb, ob));
            }
        }
        {
            let b = self.rbufs.lock();
            let (xv, wv, _) = b.as_ref().unwrap();
            for (ti, row) in xs.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
            unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), wv.ptr as *mut f32, n) };
        }
        let (xb, wb, ob) = {
            let b = self.rbufs.lock();
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf)
        };
        let p = self.pipeline(&mut ctx, Slot::Rms)?;
        let ds2 = ctx.bind_ds(&p, &[xb, wb, ob])?;
        let mut push = push_u32s(&[n as u32, t as u32, 1u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, 1, 1)?;
        let host = {
            let b = self.rbufs.lock();
            unsafe { std::slice::from_raw_parts(b.as_ref().unwrap().2.ptr as *const f32, t * n) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
        }
        Ok(())
    }

    /// silu_mul 오프로드 — exp_cr f64 호너 GLSL 비트 재현.
    pub fn silu_mul_gpu(
        &self,
        gs: &[Vec<f32>],
        us: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = gs.len();
        let n = gs[0].len();
        let total = t * n;
        let mut ctx = self.ctx.lock();
        {
            let mut b = self.sbufs.lock();
            if !b.as_ref().map(|(g, _, _)| g.bytes >= total * 4).unwrap_or(false) {
                let g = ctx.alloc_host((total * 4).max(1 << 21))?;
                let u = ctx.alloc_host((total * 4).max(1 << 21))?;
                let o = ctx.alloc_host((total * 4).max(1 << 21))?;
                *b = Some((g, u, o));
            }
        }
        {
            let b = self.sbufs.lock();
            let (gv, uv, _) = b.as_ref().unwrap();
            for (ti, row) in gs.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), gv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
            for (ti, row) in us.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), uv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
        }
        let (gb, ub, ob) = {
            let b = self.sbufs.lock();
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf)
        };
        let p = self.pipeline(&mut ctx, Slot::Silu)?;
        let ds2 = ctx.bind_ds(&p, &[gb, ub, ob])?;
        let total_u = total as u32;
        ctx.run(p.pl, ds2, p.pipe, &total_u.to_le_bytes(), total_u.div_ceil(256), 1, 1)?;
        let host = {
            let b = self.sbufs.lock();
            unsafe { std::slice::from_raw_parts(b.as_ref().unwrap().2.ptr as *const f32, total) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
        }
        Ok(())
    }

    /// FFN 상주 체인 — 업로드 1회(xs)·다운로드 1회(xs), gate/up/silu/glu/down 전부 GPU 상주.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain_gpu(
        &self,
        xs: &[Vec<f32>],
        gate_w: &Weight,
        up_w: &Weight,
        down_w: &Weight,
        xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        let n0 = gate_w.n_in as usize; // n_embd
        let n_ff = gate_w.n_out as usize;
        let xq0_w = xq_words(n0);
        let xq1_w = xq_words(n_ff);
        let mut ctx = self.ctx.lock();
        // 체인 버퍼 (고정 용량 — 모델 최대 기준)
        let (xbf, bq0, bfg, bfu, bglu, bq1, bob, xf_ptr, ob_ptr) = {
            let mut b = self.ffnbufs.lock();
            if b.is_none() {
                let xf = ctx.alloc_host(1 << 23)?;
                let xq0 = ctx.alloc_host(1 << 22)?;
                let fg = ctx.alloc_host(1 << 24)?;
                let fu = ctx.alloc_host(1 << 24)?;
                let glu = ctx.alloc_host(1 << 24)?;
                let xq1 = ctx.alloc_host(1 << 24)?;
                let ob = ctx.alloc_host(1 << 23)?;
                *b = Some((xf, xq0, fg, fu, glu, xq1, ob));
            }
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf, r.3.buf, r.4.buf, r.5.buf, r.6.buf, r.0.ptr, r.6.ptr)
        };
        // 배치 모드 — 6연산 단일 제출 (plans/19: sync ~0.9ms×5 절감)
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            ctx.begin_batch()?;
        }
        // 1) xs 업로드 → quant(n0)
        for (ti, row) in xs.iter().enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xf_ptr.add(ti * n0 * 4) as *mut f32, n0) };
        }
        {
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xbf, bq0])?;
            let push = push_u32s(&[n0 as u32, t as u32, xq0_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n0 / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 2) gate/up GEMV (같은 xq0) — 상주 출력.
        // plans/89 — t≥2 q8_0/q4_K는 밀집 coopmat 타일로: gemv3 t-루프는
        // 512토큰 프리필에서 ~50ms/디스패치(직렬 t). 레이아웃 동일
        // (outv[tok*n_out+row]). 킬스위치 LLM170_VK_FFNCH=0.
        for (w, obuf) in [(gate_w, bfg), (up_w, bfu)] {
            self.ffn_tile_or_gemv(&mut ctx, w, n0, xq0_w, t, bq0, obuf)?;
        }
        // 3) silu_mul 상주 (bfg, bfu → bglu)
        {
            let p = self.pipeline(&mut ctx, Slot::Silu)?;
            let ds2 = ctx.bind_ds(&p, &[bfg, bfu, bglu])?;
            let total = (t * n_ff) as u32;
            ctx.run(p.pl, ds2, p.pipe, &total.to_le_bytes(), total.div_ceil(256), 1, 1)?;
        }
        // 4) glu quant(n_ff)
        {
            // bglu는 f32가 아니라 f32→q8 변환 입력 — quant 셰이더에 직접.
            // (bglu는 silu 출력 f32 → quant가 읽는다)
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[bglu, bq1])?;
            let push = push_u32s(&[n_ff as u32, t as u32, xq1_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_ff / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 5) down GEMV
        {
            self.ffn_tile_or_gemv(&mut ctx, down_w, n_ff, xq1_w, t, bq1, bob)?;
        }
        // 6) 일괄 제출·대기 → 다운로드 1회
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            ctx.end_batch_wait()?;
        }
        let host = unsafe { std::slice::from_raw_parts(ob_ptr as *const f32, t * n0) };
        for ti in 0..t {
            xs_out[ti].copy_from_slice(&host[ti * n0..(ti + 1) * n0]);
        }
        Ok(())
    }

    /// ffn_chain GEMV — t≥2 q8_0/q4_K는 coopmat 타일(tile_q8128/q4k128 계열)
    /// 로, 그 외는 종전 gemv3. true=타일 경로 사용.
    fn ffn_tile_or_gemv(
        &self,
        ctx: &mut VkCtx,
        w: &Weight,
        n_in: usize,
        xq_w: usize,
        t: usize,
        xq: vk::Buffer,
        ob: vk::Buffer,
    ) -> Result<bool, String> {
        let n_out = w.n_out as usize;
        let wbufs = self.weight_bufs(ctx, w)?;
        let use_tile = t >= 2
            && std::env::var_os("LLM170_VK_FFNCH").map(|v| v != "0").unwrap_or(true)
            && std::env::var("LLM170_VK_CM").map(|v| v != "0").unwrap_or(true)
            && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K)
            && wbufs.len() == 1
            && vk_ty(w.ty).is_some();
        if !use_tile {
            let ty = vk_ty(w.ty).ok_or("ffn 타입 미지원")?;
            self.gemv_run(ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
            return Ok(false);
        }
        let (_, _, dbuf) = self.ensure_shared(ctx)?;
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
        let p = self.pipeline(ctx, slot)?;
        let ds2 = ctx.bind_ds(&p, &binds)?;
        let gx = (n_out as u32).div_ceil(64);
        if big {
            // plans/92 P1: 단일 디스패치(슬래브 x, 행 y) — 커널 유도 tok_base.
            let gys = (t as u32).div_ceil(128);
            let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, 0u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, gys, gx, 1)?;
        } else {
            for tb in (0..t).step_by(64) {
                let nt = (t - tb).min(64) as u32;
                let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, tb as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, gx, 1, 1)?;
            }
        }
        Ok(true)
    }
}

// 미지원 capability — 모든 메서드가 기본(Err) 구현이라 빈 impl 로 충분하다.
