//! vkacc::matmul — FrameHost·MatmulHost — 호스트 스테이징 GEMM. (plans/90 B1b: gemv.rs 순수 이동)

use super::*;


/// plans/93 — F32 바이트 → GGUF Q8_0 (f16 스케일 + 32×i8, 34B/블록).
/// tile_q8128 커널이 기대하는 표준 레이아웃. 꼬리 블록 남은 원소는 0 패딩.
fn f32_to_q8_0_bytes(data: &[u8], n_elem: usize) -> Vec<u8> {
    let xf: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_elem) };
    let blocks = n_elem.div_ceil(32);
    let mut out = Vec::with_capacity(blocks * 34);
    for b in 0..blocks {
        let lo = b * 32;
        let hi = (lo + 32).min(n_elem);
        let mut amax = 0.0f32;
        for &v in &xf[lo..hi] {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        // f16 인코딩 (ggml 규약: round-to-nearest-even 비트 절단)
        let dbits = d.to_bits();
        let f16 = (((dbits >> 16) as u32 & 0x8000)
            | (((dbits >> 23) as u32 & 0xFF).saturating_sub(112) as u32) << 10
            | ((dbits >> 13) as u32 & 0x3FF)) as u16;
        out.extend_from_slice(&f16.to_le_bytes());
        let mut pad = [0i8; 32];
        for (j, &v) in xf[lo..hi].iter().enumerate() {
            pad[j] = ((v * id).round()).clamp(-127.0, 127.0) as i8;
        }
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(pad.as_ptr() as *const u8, 32) });
    }
    out
}

impl llm170_core::matmul::FrameHost for VkAcc {
    fn ktrace_tick(&self) {
        self.ts_tick();
    }
    /// plans/84 B: 프레임 op군이 부분 구현(엘리먼트와이스+MoE) — 완성 전에는
    /// 옵트인(LLM170_VK_FRAME=1)일 때만 엔진이 프레임 경로에 들어온다.
    /// plans/86 §8 — 프레임 경로 완성(§1 정확성·§2 QSA 디바이스화·§5 성능) 후
    /// 기본 ON. 킬스위치 LLM170_VK_FRAME=0.
    fn frame_capable(&self) -> bool {
        std::env::var("LLM170_VK_FRAME").map(|v| v != "0").unwrap_or(true)
    }
    /// plans/85 §2 — 프레임 로짓 행별 argmax: fn_argmax_rows 2단 판.
    /// 동률 최저 인덱스 — CPU greedy_from과 동일 의미. 미구현이면 greedy
    /// 디코드 전체가 값경로 재연산으로 폴백했다(np/forward/multi 공통).
    fn frame_argmax_rows(&self, logits: u64, t: usize, vocab: usize) -> Result<Vec<u32>, String> {
        let lb = self.fbuf(logits)?;
        // WG당 256스레드×8원소 = 2048. stage1은 1워크그룹(256) 축소 — n_wg ≤ 256.
        let n_wg = vocab.div_ceil(2048);
        if n_wg > 256 || vocab == 0 || t == 0 {
            return Err(format!("vk frame_argmax_rows: 형상 초과 n_wg={n_wg} vocab={vocab} t={t}"));
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
                *g = Some(crate::rawvk::context::site::scope("argmax", || Ok::<_, String>((
                    ctx.alloc_host(sc_bytes.max(1 << 16))?,
                    ctx.alloc_host(out_bytes.max(4096))?,
                )))?);
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
            pos0 as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32,
        ]));
        ctx.run(p.pl, ds2, p.pipe, &push, (n_head + n_kv) as u32, t as u32, 1)
    }

    /// 프레임 버퍼 — host-visible(alloc_host)로 직접 읽기/쓰기.
    /// 값경로 버퍼와 동일 정책(plans/29).
    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        if std::env::var_os("LLM170_VK_POOL").is_some_and(|v| v == "0") {
            let mut ctx = self.ctx.lock();
            let b = crate::rawvk::context::site::scope("frame", || ctx.alloc_host(len * 4))?;
            let h = self.frame_next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.framebufs.lock().insert(h, b);
            return Ok(h);
        }
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
        let h = self.frame_next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len()) };
        Ok(())
    }
    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        let ptr = self.framebufs.lock().get(&h).ok_or("vk frame_write_u32: 핸들 없음")?.ptr;
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
    }
    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        self.frame_sync();
        let ptr = self.framebufs.lock().get(&h).ok_or("vk frame_read: 핸들 없음")?.ptr;
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), out.len()) };
        Ok(())
    }
    /// 상주 GEMM: 프레임 f32 버퍼 → (디바이스) quant → gemv. 업/다운 없음.
    fn frame_mm(&self, x: u64, w: &Weight, out: u64, t: usize) -> Result<(), String> {
        self.frame_mm_group(x, std::slice::from_ref(w), &[out], t)
    }
    fn frame_mm_group(&self, x: u64, ws: &[Weight], outs: &[u64], t: usize) -> Result<(), String> {
        // plans/93: F32 가중 → Q8_0 로드 시 변환(env 게이트 LLM170_F32Q8=1).
        // tile_f32 347ms → tile_q8128 경로(~115ms): 가중 판독 4× 절감.
        // conv_owned가 변환 바이트를 소유 — rebuilt는 이를 빌린다(스코프 내 생존).
        let do_f32q8 = std::env::var("LLM170_F32Q8").map(|v| v == "1").unwrap_or(false)
            && ws.iter().any(|w| w.ty == llm170_gguf::GgmlType::F32);
        // plans/93: 변환 캐시 — 가중(ptr,len)마다 1회 변환, 이후 Arc 클론.
        // 소유권: Arc<Vec<u8>>가 살아있는 동안 슬라이스 유효 (conv_arcs가 보유).
        let conv_arcs: Vec<std::sync::Arc<Vec<u8>>> = if do_f32q8 {
            let mut cache = self.f32q8_cache.lock();
            ws.iter()
                .map(|w| {
                    if w.ty != llm170_gguf::GgmlType::F32 {
                        return std::sync::Arc::new(Vec::new());
                    }
                    let key = (w.data.as_ptr() as usize, w.data.len());
                    cache
                        .entry(key)
                        .or_insert_with(|| {
                            std::sync::Arc::new(f32_to_q8_0_bytes(
                                w.data,
                                (w.n_in * w.n_out) as usize,
                            ))
                        })
                        .clone()
                })
                .collect()
        } else {
            Vec::new()
        };
        let rebuilt: Vec<Weight> = conv_arcs
            .iter()
            .zip(ws.iter())
            .map(|(c, w)| {
                if c.is_empty() {
                    Weight { data: w.data, ty: w.ty, n_in: w.n_in, n_out: w.n_out }
                } else {
                    Weight { data: c.as_slice(), ty: llm170_gguf::GgmlType::Q8_0, n_in: w.n_in, n_out: w.n_out }
                }
            })
            .collect();
        let ws: &[Weight] = if do_f32q8 { &rebuilt } else { ws };
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
            let xq = self.xq_dev_buf(&mut ctx, t * xq_w * 4)?;
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xb, xq])?;
            let push = push_u32s(&[n_in as u32, t as u32, xq_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_in / 32) + 63) as u32 / 64, t as u32, 1)?;
            xq
        } else {
            vk::Buffer::null()
        };
        let mut mmgrp_skip: Vec<bool> = vec![false; ws.len()];
        if t < 16
            && std::env::var("LLM170_VK_MMBGRP").map(|v| v != "0").unwrap_or(true)
            && !ws.is_empty()
            && ws.len() <= 8
        {
            let dty0 = dense_ty(ws[0].ty);
            let single_chunk: Vec<bool> = ws
                .iter()
                .map(|w| self.weight_bufs(&mut ctx, w).map(|b| b.len() == 1).unwrap_or(false))
                .collect();
            let groupable = dty0.is_some()
                && ws.iter().enumerate().all(|(wi, w)| {
                    dense_ty(w.ty) == dty0
                        && w.n_in as usize == n_in
                        && single_chunk[wi]
                });
            // 혼합 그룹: f32 멤버만 부분 그룹화(양자 멤버는 기존 경로).
            let (use_group, gw_idx): (bool, Vec<usize>) = if groupable {
                (true, (0..ws.len()).collect())
            } else if dty0.is_some() {
                let sub: Vec<usize> = ws
                    .iter()
                    .enumerate()
                    .filter(|(wi, w)| dense_ty(w.ty) == dty0 && w.n_in as usize == n_in && single_chunk[*wi])
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
                ctx.run(p.pl, ds2, p.pipe, &push, total_rows, t as u32, 1)?;
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
            if std::env::var_os("LLM170_VK_PULLDBG").is_some() {
                eprintln!("[pull] n_in={n_in} tys={:?}", ws.iter().map(|w| format!("{:?}", w.ty)).collect::<Vec<_>>());
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
                    if std::env::var_os("LLM170_VK_MMDBG").is_some() {
                        eprintln!("[mm] ty={ty} n_in={n_in} n_out={n_out} t={t} bytes={}", w.data.len());
                    }
                    // plans/88 P2 — 프리필(t≥2) 밀집 타일: gemv3 t-루프는
                    // 실측 ~5GB/s(gemv 6.6s/208tok). 타일(K-슬라이스 스테이징)로
                    // 대체 — 산술 클래스는 동일 표현식·스레드 직렬 누산.
                    // 스위치: LLM170_VK_DTILE=0 이면 종전 gemv.
                    // plans/89 P0.2 — 디코드(t<16) 밀집 GEMV를 llama dmmv
                    // 포트(q8b/q4b)로: f32 활성 직결(quant 불필요), 64스레드
                    // 2행 WG. [ts] 기준선 gemv 77ms/step — 272-329GB/s급으로
                    // 기대. 킬스위치 LLM170_VK_G8=0(종전 quant+gemv3).
                    if t < 16
                        && std::env::var("LLM170_VK_G8").map(|v| v != "0").unwrap_or(true)
                        && self.gemv8_dense(&mut ctx, &wbufs, n_in, n_out, t, ty, xb, ob)?
                    {
                        continue;
                    }
                    let dense_tile = t >= 2
                        && std::env::var_os("LLM170_VK_DTILE").map(|v| v != "0").unwrap_or(true)
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
                        // 2818ms/청크). 킬스위치 LLM170_VK_CM=0.
                        if std::env::var("LLM170_VK_CM").map(|v| v != "0").unwrap_or(true)
                            && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K)
                            && wbufs.len() == 1
                        {
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
                                // plans/92 P1: 128 패밀리 단일 디스패치(슬래브 x,
                                // 행 y) — 커널이 tok_base=wg.x*BN·꼬리 nt 유도.
                                // 종전 순차 t/128 디스패치의 슬래브별 전 가중
                                // 재판독 폐지.
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
                            continue;
                        }
                        match w.ty {
                            GgmlType::Q8_0 => {
                                let p = self.pipeline(&mut ctx, Slot::FnTileQ8)?;
                                let ds2 = ctx.bind_ds(&p, &binds)?;
                                // PC: n_in, n_out, chunk_words, xq_w, t.
                                let push = push_u32s(&[
                                    n_in as u32, n_out as u32, chunk_words, xq_w as u32, t as u32,
                                ]);
                                ctx.run(p.pl, ds2, p.pipe, &push, n_out.div_ceil(16) as u32, t.div_ceil(16) as u32, 1)?;
                            }
                            _ => {
                                let slot = if w.ty == GgmlType::Q4K { Slot::FnMoeTileQ4K } else { Slot::FnMoeTileQ51 };
                                let p = self.pipeline(&mut ctx, slot)?;
                                let mut b2 = binds.clone();
                                b2.push(dbuf); // rowexp
                                b2.push(dbuf); // rows_pad
                                b2.push(dbuf); // perm_pad
                                let ds2 = ctx.bind_ds(&p, &b2)?;
                                // PC: n_in, n_out, per_expert(0), chunk_words, xq_w, mode=1, t.
                                let push = push_u32s(&[
                                    n_in as u32, n_out as u32, 0u32, chunk_words, xq_w as u32, 1u32, t as u32,
                                ]);
                                ctx.run(p.pl, ds2, p.pipe, &push, n_out.div_ceil(16) as u32, t.div_ceil(16) as u32, 1)?;
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
                    if t >= 2
                        && wbufs.len() == 1
                        && std::env::var("LLM170_VK_FT32").map(|v| v != "0").unwrap_or(true)
                    {
                        // plans/93: tile_f32_w는 실측 역행(558ms vs 352ms) — 원판 유지.
                        let p = self.pipeline(&mut ctx, Slot::FnTileF32)?;
                        let mut binds: Vec<vk::Buffer> = wbufs.clone();
                        while binds.len() < 8 {
                            binds.push(dbuf);
                        }
                        binds.push(xb);
                        binds.push(ob);
                        let ds2 = ctx.bind_ds(&p, &binds)?;
                        let wpr = if dty == 0 { n_in } else { n_in / 2 };
                        let push = push_u32s(&[
                            n_in as u32, n_out as u32, t as u32, dty, wpr as u32,
                        ]);
                        ctx.run(p.pl, ds2, p.pipe, &push, (n_out as u32).div_ceil(16), (t as u32).div_ceil(16), 1)?;
                        continue;
                    }
                    let slot = if t < 16 && wbufs.len() == 1
                        && std::env::var("LLM170_VK_MMB").map(|v| v != "0").unwrap_or(true)
                    {
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
                        n_in as u32, n_out as u32, t as u32, dty,
                        (ctx.max_ssbo / 4) as u32,
                    ]);
                    ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, t as u32, 1)?;
                }
            }
        }
        Ok(())
    }
    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp as O;
        let t_cur = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        match *op {
            O::RmsRows { x, w, out, eps, n, w_reps } => {
                let (xb, wb, ob) = (self.fbuf(x)?, self.fbuf(w)?, self.fbuf(out)?);
                let rows = w_reps * t_cur;
                // plans/92 P4.1: 대형 t는 256스레드 판(rms_wide) — t=1 디코드는
                // 32스레드 원판(산술 그대로, 실측 우위).
                let slot = if rows >= 2 { Slot::RmsWide } else { Slot::Rms };
                let p = self.pipeline(&mut ctx, slot)?;
                let ds2 = ctx.bind_ds(&p, &[xb, wb, ob])?;
                let mut push = push_u32s(&[n as u32, rows as u32, w_reps as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
            }
            O::SiluDiv { t, div, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::SiluDiv)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let mut push = push_u32s(&[n as u32]);
                push.extend_from_slice(&div.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::SiluMul { g, u, out, n } => {
                let (gb, ub, ob) = (self.fbuf(g)?, self.fbuf(u)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::Silu)?;
                let ds2 = ctx.bind_ds(&p, &[gb, ub, ob])?;
                let push = push_u32s(&[n as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::Scale { t, s, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::Scale)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let mut push = push_u32s(&[n as u32]);
                push.extend_from_slice(&s.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::CopyRows { src, dst, src_off, dst_off, n } => {
                let (sb, db) = (self.fbuf(src)?, self.fbuf(dst)?);
                let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
                let ds2 = ctx.bind_ds(&p, &[sb, db])?;
                let push = push_u32s(&[n as u32, src_off as u32, dst_off as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::BcastRows { src, dst, n, rows } => {
                let (sb, db) = (self.fbuf(src)?, self.fbuf(dst)?);
                let p = self.pipeline(&mut ctx, Slot::BcastRows)?;
                let ds2 = ctx.bind_ds(&p, &[sb, db])?;
                let push = push_u32s(&[n as u32, rows as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::AxpyScaled { y, x, s, n } => {
                let (yb, xb, sb) = (self.fbuf(y)?, self.fbuf(x)?, self.fbuf(s)?);
                if t_cur <= 1 {
                    let p = self.pipeline(&mut ctx, Slot::AxpyT)?; // pp=n → s[0]와 동일
                    let ds2 = ctx.bind_ds(&p, &[yb, xb, sb])?;
                    let push = push_u32s(&[n as u32, n as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
                } else {
                    let pp = n / t_cur;
                    let p = self.pipeline(&mut ctx, Slot::AxpyT)?;
                    let ds2 = ctx.bind_ds(&p, &[yb, xb, sb])?;
                    let push = push_u32s(&[n as u32, pp as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
                }
            }
            O::HcGateMean { xn, gate, out, hc, n } => {
                let (xb, gb, ob) = (self.fbuf(xn)?, self.fbuf(gate)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::HcGateMean)?;
                let ds2 = ctx.bind_ds(&p, &[xb, gb, ob])?;
                let total = n * t_cur;
                let push = push_u32s(&[hc as u32, n as u32, total as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (total as u32).div_ceil(128), 1, 1)?;
            }
            O::HcCombine { res, out, inj, hc, n, total: _ } => {
                let (rb, ob, ib) = (self.fbuf(res)?, self.fbuf(out)?, self.fbuf(inj)?);
                let p = self.pipeline(&mut ctx, Slot::HcCombine)?;
                let ds2 = ctx.bind_ds(&p, &[rb, ob, ib])?;
                let tn = n * t_cur;
                let push = push_u32s(&[hc as u32, n as u32, tn as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (tn as u32).div_ceil(128), 1, 1)?;
            }
            O::NormGated { o, z, w, out, eps, d, n_h } => {
                let (ob, zb, wb, ub) = (self.fbuf(o)?, self.fbuf(z)?, self.fbuf(w)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::NormGatedSig)?;
                let ds2 = ctx.bind_ds(&p, &[ob, zb, wb, ub])?;
                let mut push = push_u32s(&[d as u32, n_h as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, (n_h * t_cur) as u32, 1, 1)?;
            }
            O::GdnBetaG { b, a, dtb, sa, bg, n_h } => {
                let (bb, ab, db, sb, gb) = (self.fbuf(b)?, self.fbuf(a)?, self.fbuf(dtb)?, self.fbuf(sa)?, self.fbuf(bg)?);
                let p = self.pipeline(&mut ctx, Slot::GdnBetaG)?;
                let ds2 = ctx.bind_ds(&p, &[bb, ab, db, sb, gb])?;
                let dr = n_h / t_cur.max(1);
                let push = push_u32s(&[n_h as u32, dr as u32]);
                // 판은 64스레드 — 128로 나누면 절반이 미기입된다(청크 크기별
                // 커버리지가 달라져 청크 불변성 위반의 원인이었다).
                ctx.run(p.pl, ds2, p.pipe, &push, (n_h as u32).div_ceil(64), 1, 1)?;
            }
            O::Sigmoid { t, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::EwSigmoid)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let push = push_u32s(&[n as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::Split3 { src, d0, d1, d2, n0, n1, n2 } => {
                let (sb, a0, a1, a2) = (self.fbuf(src)?, self.fbuf(d0)?, self.fbuf(d1)?, self.fbuf(d2)?);
                let p = self.pipeline(&mut ctx, Slot::Split3)?;
                let ds2 = ctx.bind_ds(&p, &[sb, a0, a1, a2])?;
                let push = push_u32s(&[n0 as u32, n1 as u32, n2 as u32]);
                let total = (n0 + n1 + n2) * t_cur;
                ctx.run(p.pl, ds2, p.pipe, &push, (total as u32).div_ceil(64), 1, 1)?;
            }
            O::L2Rows { x, eps, d, n } => {
                let xb = self.fbuf(x)?;
                let p = self.pipeline(&mut ctx, Slot::L2Rows)?;
                let ds2 = ctx.bind_ds(&p, &[xb])?;
                let mut push = push_u32s(&[d as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                let rows = (n / d).max(1);
                ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
            }
            O::L2Rows2Scale { q, k, eps, scale, d, n_group } => {
                let (qb, kb) = (self.fbuf(q)?, self.fbuf(k)?);
                let p = self.pipeline(&mut ctx, Slot::L2Rows2Scale)?;
                let ds2 = ctx.bind_ds(&p, &[qb, kb])?;
                let mut push = push_u32s(&[d as u32, n_group as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                push.extend_from_slice(&scale.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, n_group as u32, 1, 1)?;
            }
            O::GdnConv { qkv, cw, state, out, ch, k, t_len } => {
                let (qb, cb, sb, ob) = (self.fbuf(qkv)?, self.fbuf(cw)?, self.fbuf(state)?, self.fbuf(out)?);
                let binds3 = [qb, cb, sb, ob];
                if t_len >= k - 1 {
                    // 병렬 청크판 + 상태 갱신 2런치 (hip과 동일 구조)
                    let p = self.pipeline(&mut ctx, Slot::GdnConvT2)?;
                    let ds2 = ctx.bind_ds(&p, &binds3)?;
                    let push = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (ch as u32).div_ceil(64), t_len as u32, 1)?;
                    let p2 = self.pipeline(&mut ctx, Slot::GdnConvState)?;
                    let ds3 = ctx.bind_ds(&p2, &[qb, sb])?;
                    let push2 = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run(p2.pl, ds3, p2.pipe, &push2, (k - 1) as u32, (ch as u32).div_ceil(64), 1)?;
                } else {
                    // 짧은 꼬리: 순차판 (상태 회전 포함)
                    let p = self.pipeline(&mut ctx, Slot::GdnConvSeq)?;
                    let ds2 = ctx.bind_ds(&p, &binds3)?;
                    let push = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (ch as u32).div_ceil(64), 1, 1)?;
                }
            }
            O::MoeTop10 { route, ids, wt, n_exp, k_sel } => {
                let (rb, ib, wb) = (self.fbuf(route)?, self.fbuf(ids)?, self.fbuf(wt)?);
                let p = self.pipeline(&mut ctx, Slot::MoeTop10)?;
                let ds2 = ctx.bind_ds(&p, &[rb, ib, wb])?;
                let push = push_u32s(&[n_exp as u32, k_sel as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, t_cur as u32, 1, 1)?;
                // plans/88 P2 — 라우팅 세대 증가: 그룹화 캐시 무효화 키.
                self.moe_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            O::MoeWeightedSum { ys, wt, out, k, n } => {
                let (yb, wb, ob) = (self.fbuf(ys)?, self.fbuf(wt)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::MoeWsum)?;
                let ds2 = ctx.bind_ds(&p, &[yb, wb, ob])?;
                let total = n * t_cur;
                let push = push_u32s(&[n as u32, k as u32, total as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (total as u32).div_ceil(256), 1, 1)?;
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
        if std::env::var_os("LLM170_VK_MBDBG").is_some() {
            eprintln!("[mb] ty={:?} n_in={} n_out={} t={}", w.ty, w.n_in, w.n_out, t);
        }
        let xq_w = xq_words(n_in);
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        let ob = self.value_buf(&mut ctx, &self.obuf, t * n_out * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        // plans/89 — t≥2 q8_0/q4_K는 밀집 coopmat 타일로(dense_mm와 동일
        // 판·동일 수치 클래스). gemv3 t-루프는 512토큰에서 ~50ms 직렬.
        // 킬스위치 LLM170_VK_MBTILE=0.
        if t >= 2
            && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K)
            && wbufs.len() == 1
            && std::env::var_os("LLM170_VK_MBTILE").map(|v| v != "0").unwrap_or(true)
            && std::env::var("LLM170_VK_CM").map(|v| v != "0").unwrap_or(true)
        {
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
                let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, 0u32]);
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
        if ws.iter().any(|w| vk_ty(w.ty).is_none())
            || ws.iter().any(|w| w.n_in != ws[0].n_in)
        {
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
                if g[wi].as_ref().map(|b| b.bytes >= t * n_out * 4).unwrap_or(false) {
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
