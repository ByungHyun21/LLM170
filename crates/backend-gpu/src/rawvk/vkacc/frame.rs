//! vkacc::frame — FrameState — 프레임 버퍼/op 디스패치. (plans/90 B1b: gemv.rs 순수 이동)

thread_local! {
    static MOE_US: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static MOE_N: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static EW_US: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static EW_N: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

use super::*;

impl llm170_core::matmul::FrameState for VkAcc {
    /// plans/115 P1-3 — 상태 D2D 복사 묶음(접두 체크포인트 캡처/복원).
    /// copy_dev 원샷 제출(GDN 스냅샷·복원과 동일 경로).
    fn frame_copy_states(&self, pairs: &[(u64, u64, usize)]) -> Result<(), String> {
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        let g = self.framebufs.lock();
        let mut copies = Vec::with_capacity(pairs.len());
        for &(dst, src, bytes) in pairs {
            let d = g
                .get(&dst)
                .ok_or_else(|| format!("vk frame_copy: dst 핸들 없음: {dst}"))?;
            let s = g
                .get(&src)
                .ok_or_else(|| format!("vk frame_copy: src 핸들 없음: {src}"))?;
            if d.bytes < bytes || s.bytes < bytes {
                return Err(format!(
                    "vk frame_copy: 크기 부족 dst={} src={} need={bytes}",
                    d.bytes, s.bytes
                ));
            }
            copies.push((d.buf, 0, s.buf, 0, bytes as u64));
        }
        drop(g);
        ctx.copy_dev(&copies)
    }

    fn frame_begin(&self, t: usize) {
        self.frame_t
            .store(t.max(1), std::sync::atomic::Ordering::Relaxed);
        // plans/88 P1 — 스텝 수준 배치: 패스 전체를 세그먼트 최소 제출로 묶는다.
        // 비배치 run은 발사마다 제출+펜스 대기라 디코드 스텝(~2500발사)이
        // 호스트 간극에 지배됐다(실측 제출 2548/스텝). 스텝 도중 브리지의
        // frame_read가 플러시하면 프레임 op 진입마다 재개(frame_resume_batch).
        // 값경로는 이 게이트를 보지 않아 배치 상태가 새지 않는다.
        if !llm170_diag::flag::on("LLM170_VK_NOBATCH") {
            self.frame_step_batch
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = self.ctx.lock().begin_batch();
        }
    }

    fn set_ctx_len(&self, n: usize) {
        self.qsa_ctx.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// GDN AR (프레임) — q35 판(gdn_ar.spv) 재사용. q는 1/√d 스케일 완료
    /// 가정(scale=1.0), 순차 t 토큰 — 청크 불변(이전 입력만 의존).
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
            return Err("vk frame_gdn_ar: np 미지원".into());
        }
        let t = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        let (sb, qb, kb, vb, bb, ob) = (
            self.fbuf(states)?,
            self.fbuf(q_scaled)?,
            self.fbuf(k)?,
            self.fbuf(v)?,
            self.fbuf(beta_ge)?,
            self.fbuf(out)?,
        );
        let p = self.pipeline(&mut ctx, Slot::FnGdnArSwap)?;
        let ds2 = ctx.bind_ds(&p, &[sb, qb, kb, vb, bb, ob])?;
        let mut push = push_u32s(&[
            d as u32,
            (h_k * d) as u32,
            (h_v * d) as u32,
            h_v as u32,
            h_k as u32,
        ]);
        push.extend_from_slice(&1.0f32.to_le_bytes());
        push.extend_from_slice(&(t as u32).to_le_bytes());
        // plans/99 u폴딩×2 — WG당 상태행 2개(로드 절반·ILP 2배).
        ctx.run(p.pl, ds2, p.pipe, &push, (d / 2) as u32, h_v as u32, 1)
    }

    /// MoE 게더 — (토큰,전문가) 페어: xsel[(ti·k+s)·n] = mix[ti·n]
    /// (토큰 스트라이드 게더 — BcastRows 재사용은 1행 방송이라 틀렸다).
    fn frame_moe_gather(
        &self,
        mix: u64,
        xsel: u64,
        n: usize,
        k_sel: usize,
        t: usize,
    ) -> Result<(), String> {
        let mut ctx = self.ctx.lock();
        let (sb, db) = (self.fbuf(mix)?, self.fbuf(xsel)?);
        let p = self.pipeline(&mut ctx, Slot::MoeGatherRows)?;
        let ds2 = ctx.bind_ds(&p, &[sb, db])?;
        let push = push_u32s(&[n as u32, k_sel as u32, t as u32]);
        ctx.run_rw(
            p.pl,
            ds2,
            p.pipe,
            &push,
            (n as u32).div_ceil(128),
            ((t * k_sel) as u32).div_ceil(4),
            1,
            &[sb],
            &[db],
        )?;
        Ok(())
    }

    /// MoE 스캐터(가중합) — MoeWeightedSum 판 재사용(산술 동일).
    fn frame_moe_scatter(
        &self,
        ys: u64,
        wt: u64,
        out: u64,
        k_sel: usize,
        n: usize,
        _t: usize,
    ) -> Result<(), String> {
        use llm170_core::matmul::FrameHost;
        self.frame_op(&llm170_core::matmul::FrameOp::MoeWeightedSum {
            ys,
            wt,
            out,
            k: k_sel,
            n,
        })
    }

    /// plans/105(원장 80) — mxsel 팩 정량(생산 시점 1회).
    fn frame_quant_pack(&self, x: u64, rows: usize, n_in: usize) -> Result<(), String> {
        if rows == 0 || n_in == 0 {
            return Ok(());
        }
        let row_words = (n_in >> 5) * 10;
        let need = rows * row_words * 4;
        let buf = {
            let mut g = self.packbufs.lock();
            if let Some((b, bytes)) = g.0.get(&x) {
                if *bytes >= need {
                    *b
                } else {
                    let nb = {
                        let mut ctx = self.ctx.lock();
                        ctx.alloc(need)?
                    };
                    let raw = nb.buf;
                    let bytes2 = nb.bytes;
                    g.1.push(nb); // 구버퍼 보유
                    g.0.insert(x, (raw, bytes2));
                    raw
                }
            } else {
                let nb = {
                    let mut ctx = self.ctx.lock();
                    ctx.alloc(need)?
                };
                let raw = nb.buf;
                let bytes2 = nb.bytes;
                g.1.push(nb);
                g.0.insert(x, (raw, bytes2));
                raw
            }
        };
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        let xb = self.fbuf(x)?;
        let pq = self.pipeline(&mut ctx, Slot::FnQuantQ8p)?;
        let dsq = ctx.bind_ds(&pq, &[xb, buf])?;
        let nblk = n_in >> 5;
        let pushq = push_u32s(&[n_in as u32, rows as u32, row_words as u32]);
        ctx.run_rw(
            pq.pl,
            dsq,
            pq.pipe,
            &pushq,
            (nblk as u32).div_ceil(64),
            rows as u32,
            1,
            &[xb],
            &[buf],
        )?;
        Ok(())
    }

    /// 상주 MoE GEMM — plans/84 B 슬라이스: 호스트 그룹화(hip 폴백과 동일
    /// 구조) + 디바이스 게더/전문가별 GEMV/스캐터. vk GEMV는 단일 판이라
    /// 전문가별 행수가 패밀리를 갈라놓지 않는다(청크 불변성 안전).
    fn frame_moe_gemm(
        &self,
        x: u64,
        w: &Weight,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
        k_sel: usize,
    ) -> Result<(), String> {
        self.moe_gemm_impl(x, w, ids, out, n_expert_stack, k_sel)
    }
}

impl VkAcc {
    fn moe_gemm_impl(
        &self,
        x: u64,
        w: &Weight,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
        k_sel: usize,
    ) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let ne = n_expert_stack.max(1);
        // 스택 텐서: w.n_out = 전문가당 n_out × ne — GEMM은 전문가당 폭만 쓴다.
        let n_out = w.n_out as usize / ne;
        let t = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let rows = t * k_sel;
        let ty = vk_ty(w.ty).ok_or("vk frame_moe_gemm: 타입 미지원")?;
        let mut ctx = self.ctx.lock();
        let xb = self.fbuf(x)?;
        let ob = self.fbuf(out)?;
        let xq_w = xq_words(n_in);
        // plans/141: HIP 전용 Q5K dmmv 승격을 공유하면 VK fn_moe_ids(Q5K)가
        // xq=null을 읽어 L2.moe_sc를 전부 0으로 만든다. 단일 SSBO의 Q4K/Q5_1
        // f32 직독 셰이더에만 양자화를 생략한다(청크 폴백도 xq 필요).
        let ids2_takes =
            crate::common::moe::vk_ids2_takes(rows, t, w.ty, w.data.len(), ctx.max_ssbo);
        // plans/105(원장 80): 팩 등록 히트 — 상위 정량 스킵(llmmq가 팩 소비).
        let pack_skip_quant = w.ty == GgmlType::Q4K && self.packbufs.lock().0.contains_key(&x);
        let xq = if ids2_takes || pack_skip_quant {
            vk::Buffer::null()
        } else {
            // plans/96 G3 — gate+up 연속 쌍: 같은 (x,n_in,rows)의 재양자화를
            // 전용 버퍼 슬롯으로 회수. 두 엔진 호출 사이 어떤 op/quant도
            // 없고 전용 버퍼는 타 quant가 덮어쓰지 못함.
            // [2026-10-04 moech 결함] "세대 불필요" 전제는 같은 핸들·같은 형상에
            // **내용만 다른 재투입**(청크 프리필가 같은 mx 버퍼 재사용)에서
            // 깨진다 — 청크 2+가 청크 1의 xq로 계산된다(실측 moech 청크 불변
            // max|D|=1.0e0). MoeTop10가 라우팅마다 올리는 moe_gen 을 키에
            // 추가: gate→up 사이(세대 불변)만 히트, 청크/층 경계(세대 증가)는
            // 재양자화.
            // plans/96: 프리필(t≥2) 전용 — 전층 체크섬으로 프리필 정합 실측 확정.
            // 디코드 q8스택 경로(t=1)에서의 미세 발산(원장 종결 기록)을 원천 차단.
            let pair_on = t >= 2;
            let pair_gen = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
            let mut hit: Option<vk::Buffer> = None;
            if pair_on {
                let sl = self.moe_xq_pair.lock();
                if let Some((hx, hn, hr, hgen, b)) = sl.as_ref()
                    && *hx == x
                    && *hn == n_in
                    && *hr == rows
                    && *hgen == pair_gen
                    && b.bytes >= rows * xq_w * 4
                {
                    hit = Some(b.buf);
                }
            }
            match hit {
                Some(b) => b,
                None => {
                    // 107 W1: QuantS8 변형 폐기(+11% 지연 — 부정 주석이 근거).
                    let p = self.pipeline(&mut ctx, Slot::Quant)?;
                    let push = push_u32s(&[n_in as u32, rows as u32, xq_w as u32]);
                    let tgt = if pair_on {
                        let mut sl = self.moe_xq_pair.lock();
                        let need = rows * xq_w * 4;
                        let ok = sl.as_ref().is_some_and(|v| v.4.bytes >= need);
                        if !ok {
                            *sl = Some((x, n_in, rows, pair_gen, ctx.alloc(need)?));
                        } else if let Some(v) = sl.as_mut() {
                            v.0 = x;
                            v.1 = n_in;
                            v.2 = rows;
                            v.3 = pair_gen;
                        }
                        sl.as_ref().unwrap().4.buf
                    } else {
                        self.xq_dev_buf(&mut ctx, rows * xq_w * 4)?
                    };
                    let ds2 = ctx.bind_ds(&p, &[xb, tgt])?;
                    ctx.run(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        ((n_in / 32) + 63) as u32 / 64,
                        rows as u32,
                        1,
                    )?;
                    // plans/96 G3 — 직전 SiluMul 출력이 이 x면 glu로 학습
                    // (다음 층부터 silu+quant 융합).
                    if self
                        .last_silu_out
                        .load(std::sync::atomic::Ordering::Relaxed)
                        == x
                    {
                        *self.moe_glu.lock() = Some((x, n_in));
                    }
                    tgt
                }
            }
        };
        // 2b) direct-ids (plans/88 P1) — t=1·rows≤64: fn_moe_ids(gemv3 파생)
        // 그리드 (n_out, rows), 워크그룹=행 — 커널이 ids[r]을 직접 판독해 가중
        // 베이스 = ids[r]·per_expert 를 산출한다. ids d2h(동기 드레인)·호스트
        // 그룹화·perm/inv 업로드·게더·전문가 루프·스캐터 전부 소거(B1).
        // 산술은 종전 전문가별 gemv3 경로와 출력 요소당 비트 동일(레인 부담·
        // 감축 동일) — 토큰 스트림 불변 계약. (구 호스트 그룹화 강제 스위치
        // LLM170_MOE_GROUPED 는 90-B5 폐기 — 기본경로와 수치 동일 확인.)
        // 초판의 hip 16×16타일 직역은 이 vk에서 점유율 부족(160WG, 실측
        // 10GB/s vs hip 180GB/s)으로 폐기 — K-분할은 레인 분할(256)이 담당.
        if rows > 0
            && (t == 1 || rows <= 64)
            && matches!(
                w.ty,
                GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q5_1 | GgmlType::Q8_0
            )
        {
            // plans/89 P0.3 — ids dmmv 판 우선: llama dmmv 기하(64스레드·2행·
            // 서브그룹Add) + ids 간접, f32 활성 직결(MoE quant 불필요).
            // [ts] 기준선 moe_ids 30ms/step(43GB/s) — q8b급 150GB/s 기대.
            // plans/89 P0.3 — ids dmmv 판 우선(승격 기본 — 킬스위치 폐지, 107 W1).
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            if wbufs.len() == 1 {
                let (slot, blk) = match w.ty {
                    GgmlType::Q4K => (Slot::FnMoeIds2, 144usize),
                    GgmlType::Q5_1 => (Slot::FnMoeIds51, 24),
                    _ => (Slot::FnMoeIds, 0),
                };
                if blk != 0 {
                    let idb = self.fbuf(ids)?;
                    // ne 는 호출부에서 max(1) 보장 — 0이면 계약 위반 데이터.
                    let per_expert = w.data.len().checked_div(ne).expect("moe: ne=0");
                    let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
                    let p = self.pipeline(&mut ctx, slot)?;
                    let mut binds: Vec<vk::Buffer> = wbufs.clone();
                    while binds.len() < 8 {
                        binds.push(dbuf);
                    }
                    binds.push(xb);
                    binds.push(ob);
                    binds.push(idb);
                    let ds2 = ctx.bind_ds(&p, &binds)?;
                    // PC: n_in, n_out, rows, per_expert_blks, cw(0), rpf(2).
                    let push = push_u32s(&[
                        n_in as u32,
                        n_out as u32,
                        rows as u32,
                        per_expert.checked_div(blk).expect("moe: blk=0") as u32,
                        0,
                        2,
                    ]);
                    ctx.run(
                        p.pl,
                        ds2,
                        p.pipe,
                        &push,
                        1,
                        n_out.div_ceil(2) as u32,
                        rows as u32,
                    )?;
                    return Ok(());
                }
            }
            let idb = self.fbuf(ids)?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            let per_expert = w.data.len() / ne;
            let (kb, gb, dbuf) = self.ensure_shared(&mut ctx)?;
            let chunk_words = (ctx.max_ssbo / 4) as u32;
            let p = self.pipeline(&mut ctx, Slot::FnMoeIds)?;
            // PC 선언순: n_in, n_out, xq_w, ty, rows, per_expert, chunk_words.
            let push = push_u32s(&[
                n_in as u32,
                n_out as u32,
                xq_w as u32,
                ty,
                rows as u32,
                per_expert as u32,
                chunk_words,
            ]);
            // 바인딩순: W0..7, xq, out, ktab, grid3s, ids.
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xq);
            binds.push(ob);
            binds.push(kb);
            binds.push(gb);
            binds.push(idb);
            let ds2 = ctx.bind_ds(&p, &binds)?;
            ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, rows as u32, 1)?;
            return Ok(());
        }
        // 2c) 그룹 타일 (plans/88 P2) — 프리필 대량행: 디바이스 그룹화(세대
        // 캐시 — 같은 라우팅의 3개 GEMM이 테이블 공유) + 16×16 타일(패딩
        // 도메인, 타일=전문가, x는 perm_pad 간접 판독 — 게더 패스 불필요) +
        // inv_pad 산란. 호스트 ids 왕복·512 전문가 루프 전부 소거(B2).
        // 산술: hip ge/w_ids 열과 동일 표현식 — 프리필 클래스 재기록 대상.
        // fn_moe_group 셰이더 공유 배열 한계(ne ≤ 512) — 초과 모델은 조기복귀로
        // 테이블 미기록 상태가 되므로 호스트에서 차단(현행 512-전문가 무영향).
        let tile_ok = rows > 0
            && ne <= 512
            && match w.ty {
                GgmlType::Q4K => n_in <= 4096,
                GgmlType::Q5_1 => n_in <= 2048,
                // plans/89 P1.1c — q8_0/q5_K MoE 역할(UD-Q4_K_XL 혼합)도 타일로:
                // 레거시 512-전문가 gemv3 루프([ts] gemv 1101ms/청크) 소거.
                GgmlType::Q8_0 => n_in <= 4096,
                GgmlType::Q5K => n_in <= 4096,
                _ => false,
            };
        if tile_ok {
            let idb = self.fbuf(ids)?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            let per_expert = w.data.len() / ne;
            let bound = crate::common::moe::grp_bound(rows, ne);
            let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
            let chunk_words = (ctx.max_ssbo / 4) as u32;
            let generation = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
            let hit = {
                let g = self.moe_grp.lock();
                g.as_ref().is_some_and(|g| {
                    crate::common::moe::cache_hit(
                        g.generation,
                        generation,
                        g.rows,
                        rows,
                        g.ids_h == ids,
                    )
                })
            };
            if !hit {
                // 테이블 성장(단일 상한 bound — 그룹 커널이 [rp,bound)를 0채움).
                crate::rawvk::context::site::scope("moe_grp", || -> Result<(), String> {
                    let mut g = self.moe_grp.lock();
                    let e = g.get_or_insert_with(|| MoeGrp {
                        generation: 0,
                        rows: 0,
                        ids_h: 0,
                        bound: 0,
                        off_n: 0,
                        off: vkbuf_null(),
                        rows_pad: vkbuf_null(),
                        tilexp: vkbuf_null(),
                        perm: vkbuf_null(),
                        inv: vkbuf_null(),
                        inv_pad: vkbuf_null(),
                        rowexp: vkbuf_null(),
                        perm_pad: vkbuf_null(),
                        yg: vkbuf_null(),
                        yg_rows: 0,
                    });
                    if e.bound < bound {
                        e.rowexp = ctx.alloc_host(bound * 4)?;
                        e.perm_pad = ctx.alloc_host(bound * 4)?;
                        e.tilexp = ctx.alloc_host((bound / 16 + 1) * 4)?;
                    }
                    if e.rows < rows {
                        e.perm = ctx.alloc_host(rows * 4)?;
                        e.inv = ctx.alloc_host(rows * 4)?;
                        e.inv_pad = ctx.alloc_host(rows * 4)?;
                    }
                    if e.off_n < ne + 1 {
                        e.off = ctx.alloc_host((ne + 1) * 4)?;
                        e.off_n = ne + 1;
                    }
                    if e.rows_pad.bytes == 0 {
                        e.rows_pad = ctx.alloc_host(8)?;
                    }
                    Ok(())
                })?;
                let pg = self.pipeline(&mut ctx, Slot::FnMoeGroup)?;
                let (ob_, rpb, txb, pmb, ivb, ivpb, rxb, ppb) = {
                    let g = self.moe_grp.lock();
                    let g = g.as_ref().unwrap();
                    // 바인딩순 = 셰이더 선언순: off, rows_pad, tilexp, perm, inv,
                    // inv_pad, rowexp, perm_pad (초판이 inv 를 건너뛰어 전부 어긋남).
                    (
                        g.off.buf,
                        g.rows_pad.buf,
                        g.tilexp.buf,
                        g.perm.buf,
                        g.inv.buf,
                        g.inv_pad.buf,
                        g.rowexp.buf,
                        g.perm_pad.buf,
                    )
                };
                let dsg = ctx.bind_ds(&pg, &[idb, ob_, rpb, txb, pmb, ivb, ivpb, rxb, ppb])?;
                // plans/93 sg2: 32행/WG 판은 전문가 패딩도 32배수여야 경계 정렬.
                let padmul: u32 = 16; // plans/93 sg2 32배수 실험(PAD32)은 plans/109 P6 삭제
                let push = push_u32s(&[ne as u32, rows as u32, bound as u32, padmul]);
                ctx.run_rw(
                    pg.pl,
                    dsg,
                    pg.pipe,
                    &push,
                    1,
                    1,
                    1,
                    &[idb],
                    &[ob_, rpb, txb, pmb, ivb, ivpb, rxb, ppb],
                )?;
                {
                    let mut g = self.moe_grp.lock();
                    let gi = g.as_mut().unwrap();
                    gi.generation = generation;
                    gi.rows = rows;
                    gi.ids_h = ids;
                    gi.bound = gi.bound.max(bound);
                }
                if llm170_diag::flag::on("LLM170_MOE_GCHECK") {
                    // 진단: 그룹 테이블 불변식 검증(전문가 내 순서는 atomic이라
                    // 비결정 — 순서 무관 불변식으로 판정).
                    ctx.end_batch_wait()?;
                    let idv: Vec<u32> = {
                        let g = self.framebufs.lock();
                        let b = g.get(&ids).ok_or("ids 핸들 없음")?;
                        // SAFETY (107 W8): ids 매핑 판독 — 직전 end_batch_wait로 GPU 유휴; rows 원소.
                        unsafe { std::slice::from_raw_parts(b.ptr as *const u32, rows) }.to_vec()
                    };
                    let gg = self.moe_grp.lock();
                    let gg = gg.as_ref().unwrap();
                    // SAFETY (107 W8): moe_grp 매핑 판독 클로저 — end_batch_wait 후 유휴; n은 각 버퍼 원소수와 일치(호출부 지정).
                    let rd = |b: &VkBuf, n: usize| unsafe {
                        std::slice::from_raw_parts(b.ptr as *const u32, n)
                    };
                    let dev_perm_pad = rd(&gg.perm_pad, bound);
                    let dev_inv_pad = rd(&gg.inv_pad, rows);
                    let dev_rowexp = rd(&gg.rowexp, bound);
                    // 호스트 재계산 — common 공용판(hip 호스트 빌드와 동일 코드, P13).
                    let hoff = crate::common::moe::grp_offsets(&idv, ne);
                    // plans/93 sg2: 32행/WG 판은 전문가 경계가 32 배수여야 —
                    // WG가 두 전문가를 가로지르면 rowexp[0]의 가중치로 오계산.
                    let padmul = 16;
                    let (hpoff, rows_pad) = crate::common::moe::grp_padded(&hoff, ne, padmul);
                    let rp_dev =
                        // SAFETY (107 W8): rows_pad 매핑 판독 — 유휴 상태(end_batch_wait 후), 2원소.
                        unsafe { std::slice::from_raw_parts(gg.rows_pad.ptr as *const u32, 2) }[0]
                            as usize;
                    let rp_dbg =
                        unsafe { std::slice::from_raw_parts(gg.rows_pad.ptr as *const u32, 2) };
                    let mut bad = 0usize;
                    let mut seen = vec![false; rows];
                    if rp_dev != rows_pad {
                        eprintln!(
                            "[gcheck] rows_pad dev={rp_dev} host={rows_pad} dbg1={}",
                            rp_dbg[1]
                        );
                        bad += 1;
                    }
                    for pd in 0..rows_pad {
                        let e = dev_rowexp[pd] as usize;
                        if e >= ne || !(hpoff[e]..hpoff[e + 1]).contains(&pd) {
                            if bad < 6 {
                                eprintln!("[gcheck] rowexp[{pd}]={e} 세그 불일치");
                            }
                            bad += 1;
                            continue;
                        }
                        let i = pd - hpoff[e];
                        let r = hoff[e + 1] - hoff[e];
                        let src = dev_perm_pad[pd] as usize;
                        if i < r {
                            if src >= rows || idv[src] as usize != e {
                                if bad < 6 {
                                    eprintln!("[gcheck] perm_pad[{pd}]={src} 전문가 불일치(e={e})");
                                }
                                bad += 1;
                            } else if seen[src] {
                                if bad < 6 {
                                    eprintln!("[gcheck] perm_pad[{pd}]={src} 중복");
                                }
                                bad += 1;
                            } else {
                                seen[src] = true;
                            }
                            if dev_inv_pad[src] as usize != pd {
                                if bad < 10 {
                                    eprintln!(
                                        "[gcheck] inv_pad[{src}]={} != pd={pd}",
                                        dev_inv_pad[src]
                                    );
                                }
                                bad += 1;
                            }
                        } else if src != 0 {
                            if bad < 6 {
                                eprintln!("[gcheck] 패딩 perm_pad[{pd}]={src} != 0");
                            }
                            bad += 1;
                        }
                    }
                    let miss = seen.iter().filter(|s| !**s).count();
                    if miss > 0 && bad < 10 {
                        eprintln!("[gcheck] 커버 누락 {miss}행");
                    }
                    let dev_off2 = rd(&gg.off, ne + 1);
                    eprintln!(
                        "[gcheck] rows={rows} rows_pad={rows_pad} bad={bad} miss={miss} off[ne]={} off[0..3]={:?} rowexp[0..6]={:?} perm_pad[0..6]={:?}",
                        dev_off2[ne],
                        &dev_off2[..3],
                        &dev_rowexp[..6],
                        &dev_perm_pad[..6]
                    );
                }
            }
            let (rxb, rpb, ppb, ivb, ygb) = {
                let mut g = self.moe_grp.lock();
                let gi = g.as_mut().unwrap();
                let need_yg = gi.bound * n_out * 4;
                if gi.yg.bytes < need_yg {
                    gi.yg = crate::rawvk::context::site::scope("moe_grp", || ctx.alloc(need_yg))?;
                    gi.yg_rows = gi.bound;
                }
                (
                    gi.rowexp.buf,
                    gi.rows_pad.buf,
                    gi.perm_pad.buf,
                    gi.inv_pad.buf,
                    gi.yg.buf,
                )
            };
            // plans/96 G3 — q8_0 스택 down: 구 스칼라(24ms/회) 대체 MMQ.
            // A측 q8r 무손실 릴레이아웃(dense 자산 재사용) — per_expert는
            // q8r 행바이트 기준으로 교체. env LLM170_VK_Q8MOE(기본 on).
            let mut per_expert_push = per_expert;
            let mut w0_override: Option<vk::Buffer> = None;
            if w.ty == GgmlType::Q8_0 && wbufs.len() == 1 {
                let key = (w.data.as_ptr() as usize, w.data.len());
                let mut c = self.q8r_bufs.lock();
                let b = match c.get(&key) {
                    Some(b) => b.buf,
                    None => {
                        let bytes = q8_0_relayout(w.data, n_in, w.n_out as usize);
                        let b = ctx.alloc_host(bytes.len())?;
                        // SAFETY (107 W8): q8 relayout 캐시 기입 — b는 bytes.len()으로 방금 alloc_host; 매핑 유효, 제출 전.
                        unsafe {
                            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.ptr, bytes.len())
                        };
                        let buf = b.buf;
                        c.insert(key, b);
                        buf
                    }
                };
                w0_override = Some(b);
                per_expert_push = n_out * (n_in + (n_in / 32) * 4);
            }
            if w.ty == GgmlType::Q5K && wbufs.len() == 1 {
                w0_override = Some(vk::Buffer::null()); // 마커 — 실바인딩은 원본
            }
            if w.ty == GgmlType::Q5_1 && wbufs.len() == 1 {
                w0_override = Some(vk::Buffer::null()); // 마커 — 원본 바인딩
            }
            // plans/105 P2: llama mul_mmq 포트(옵트인 LLM170_VK_Q4KLL=1) —
            // BN64 워프타일·전문가당 단일 WG(가중 1회 판독). 근거: llama
            // 노드 타이밍 2897µs/콜 vs 원판 4690µs(원장 75).
            let pack_hit = w.ty == GgmlType::Q4K
                && wbufs.len() == 1
                && self.packbufs.lock().0.contains_key(&x);
            if pack_hit {
                let (offb, pmb) = {
                    let g = self.moe_grp.lock();
                    let g = g.as_ref().unwrap();
                    (g.off.buf, g.perm.buf)
                };
                let (xq_ll, _llbytes) = *self.packbufs.lock().0.get(&x).unwrap();
                // 107 W1: f16-dm 변형(LLMMQH16) 폐기 — 원장 83 중립 판정.
                let pk = self.pipeline(&mut ctx, Slot::FnMoeTileLlmmq)?;
                let mut pbinds: Vec<vk::Buffer> = vec![wbufs[0]];
                while pbinds.len() < 8 {
                    pbinds.push(dbuf);
                }
                pbinds.push(xq_ll); // binding 8 — 팩 버퍼
                pbinds.push(ob); // binding 9
                pbinds.push(dbuf); // binding 10 (커널 미사용 슬롯 패드)
                pbinds.push(pmb); // binding 11
                pbinds.push(offb); // binding 12
                let ds2 = ctx.bind_ds(&pk, &pbinds)?;
                let push = push_u32s(&[
                    n_in as u32,
                    n_out as u32,
                    per_expert as u32,
                    chunk_words,
                    xq_w as u32,
                    n_expert_stack as u32,
                    ((n_in >> 5) * 10) as u32,
                ]);
                let pkrds: Vec<vk::Buffer> = vec![wbufs[0], xq_ll, pmb, offb];
                ctx.run_rw(
                    pk.pl,
                    ds2,
                    pk.pipe,
                    &push,
                    (n_out as u32).div_ceil(64),
                    n_expert_stack as u32,
                    1,
                    &pkrds,
                    &[ob],
                )?;
                return Ok(());
            }
            let slot = if w.ty == GgmlType::Q5K && w0_override.is_some() {
                Slot::FnMoeTileQ5kmmq
            } else if w.ty == GgmlType::Q5_1 && w0_override.is_some() {
                Slot::FnMoeTileQ51mmq
            } else if w0_override.is_some() {
                Slot::FnMoeTileQ8mmq
            } else {
                // 107 W1: 부정 변형 슬롯 전량 폐기(원장 63-86) — 잔여는
                // 승격 기본(mmq 계열·q51_sg1)과 다중중량 스칼라 폴백뿐.
                match w.ty {
                    GgmlType::Q5_1 => Slot::FnMoeTileQ51Sg1,
                    GgmlType::Q4K if wbufs.len() == 1 => Slot::FnMoeTileQ4kMmq,
                    GgmlType::Q8_0 => Slot::FnMoeTileQ8,
                    _ => Slot::FnMoeTileQ5k,
                }
            };

            let p = self.pipeline(&mut ctx, slot)?;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            if let Some(b) = w0_override
                && b != vk::Buffer::null()
            {
                binds[0] = b;
            }
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            // plans/96 G2: 직접산란 타일(기본 4종+v3)은 출력을 ob에 직접 —
            // 스캐터(permute_f32) 폐지. 레거시 슬롯은 종전 yg+스캐터.
            let direct = matches!(slot, |Slot::FnMoeTileQ51Sg1| Slot::FnMoeTileQ8
                | Slot::FnMoeTileQ5k
                | Slot::FnMoeTileQ4kMmq
                | Slot::FnMoeTileQ8mmq
                | Slot::FnMoeTileQ5kmmq
                | Slot::FnMoeTileQ51mmq);
            binds.push(xq);
            binds.push(if direct { ob } else { ygb });
            binds.push(rxb);
            binds.push(rpb);
            binds.push(ppb);
            let ds2 = ctx.bind_ds(&p, &binds)?;
            // PC 선언순: n_in, n_out, per_expert_bytes, chunk_words, xq_w, mode, rows.
            let push = push_u32s(&[
                n_in as u32,
                n_out as u32,
                per_expert_push as u32,
                chunk_words,
                xq_w as u32,
                0u32,
                rows as u32,
            ]);
            // 107 W1: 변형 슬롯 폐기 후 잔여 — 와이드 64열판(mmq 계열)과
            // 16열판(sg1·스칼라 폴백) 두 가지.
            let wide = matches!(
                slot,
                Slot::FnMoeTileQ4kMmq
                    | Slot::FnMoeTileQ8mmq
                    | Slot::FnMoeTileQ5kmmq
                    | Slot::FnMoeTileQ51mmq
            );
            let (gx, gy) = if wide {
                (n_out.div_ceil(64) as u32, bound.div_ceil(16) as u32)
            } else {
                (n_out.div_ceil(16) as u32, bound.div_ceil(16) as u32)
            };
            let tile_out = if direct { ob } else { ygb };
            let mut tile_rds: Vec<vk::Buffer> = binds[..8].to_vec();
            tile_rds.push(xq);
            tile_rds.push(rxb);
            tile_rds.push(rpb);
            tile_rds.push(ppb);
            ctx.run_rw(p.pl, ds2, p.pipe, &push, gx, gy, 1, &tile_rds, &[tile_out])?;
            // plans/93: gate→up 독립 병렬화 — 이 타일이 gate이면 다음(up) 배리어 스킵.
            if self.moe_nobar.load(std::sync::atomic::Ordering::Relaxed) {
                // plans/105: gate‖up 강제 병행(승격 기본 — 킬스위치 폐지).
                ctx.nobar_next.set(true);
                self.moe_nobar
                    .store(false, std::sync::atomic::Ordering::Relaxed);
            }
            if direct {
                // plans/96 G2: 드레인이 perm 맵으로 원본 행에 직접 기록 — 폐지.
                return Ok(());
            }
            // 산란: out[i] = yg[inv_pad[i]] (행 순서 복원 — SiluMul/wsum 소비).
            let ps = self.pipeline(&mut ctx, Slot::PermuteF32)?;
            let dss = ctx.bind_ds(&ps, &[ygb, ivb, ob])?;
            let push = push_u32s(&[n_out as u32, rows as u32]);
            ctx.run_rw(
                ps.pl,
                dss,
                ps.pipe,
                &push,
                rows as u32,
                1,
                1,
                &[ygb, ivb],
                &[ob],
            )?;
            return Ok(());
        }
        // 1) ids 판독(호스트 그룹화) — direct-ids 가 걸러준 프리필 대량행만.
        // 가드를 내린 뒤 d2h 드레인(d2h 블록이 스스로 ctx 를 잡는다).
        drop(ctx);
        // 1) ids 판독(호스트 그룹화) — off/perm/inv 구축.
        let idv: Vec<u32> = {
            {
                let mut c = self.ctx.lock();
                if c.batching.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = c.end_batch_wait();
                }
            }
            let g = self.framebufs.lock();
            let b = g.get(&ids).ok_or("vk moe: ids 핸들 없음")?;
            // SAFETY (107 W8): ids 매핑 판독 — 직전 end_batch_wait로 GPU 유휴; rows 원소.
            unsafe { std::slice::from_raw_parts(b.ptr as *const u32, rows) }.to_vec()
        };
        // 카운팅 정렬 테이블 — common 공용판(hip 호스트 빌드와 동일 코드, P13).
        let off = crate::common::moe::grp_offsets(&idv, ne);
        let perm = crate::common::moe::grp_perm(&idv, ne, &off);
        let inv = crate::common::moe::grp_inv(&perm);
        let mut ctx = self.ctx.lock();
        // 3) MoE 스크래치 (perm u32, xg u32, iv u32, yg f32) — 필요시 성장.
        // xg 행 스트라이드는 16B 정렬로 패딩 — 전문가별 디스크립터 오프셋이
        // minStorageBufferOffsetAlignment를 만족해야 한다(down n_in=640의
        // 760B 행은 8 mod 16 → 미정렬 오프셋에서 오염 판독).
        let xq_w_pad = (xq_w + 3) & !3;
        let need_xg = rows * xq_w_pad * 4;
        let need_yg = rows * n_out * 4;
        {
            let mut g = self.moebufs.lock();
            // plans/86 §5 — 컴포넌트별 성장. 종전 전부-만족 검사는 gate(yg 52MB)와
            // down(yg 210MB)이 크기 계급을 달리해 매호출 4버퍼 재할당 → 48층×3gemm×
            // 청크마다 누적(pp4096 실측 33.6GiB, 카브아웃 오버플로→GTT 전이→OOM).
            let e =
                g.get_or_insert_with(|| (vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
            crate::rawvk::context::site::scope("moe_scratch", || -> Result<(), String> {
                if e.0.bytes < rows * 4 {
                    e.0 = ctx.alloc((rows * 4).max(1 << 16))?;
                }
                if e.1.bytes < need_xg {
                    e.1 = ctx.alloc(need_xg.max(1 << 16))?;
                }
                if e.2.bytes < rows * 4 {
                    e.2 = ctx.alloc((rows * 4).max(1 << 16))?;
                }
                if e.3.bytes < need_yg {
                    e.3 = ctx.alloc(need_yg.max(1 << 16))?;
                }
                Ok(())
            })?;
        }
        let (pmb, xgb, ivb, ygb) = {
            let g = self.moebufs.lock();
            let r = g.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf, r.3.buf)
        };
        // SAFETY (107 W8): moebufs 매핑 기입 — perm/inv 각 rows u32로 크기 일치; 기록 단계(제출 전).
        unsafe {
            let g = self.moebufs.lock();
            let r = g.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(perm.as_ptr(), r.0.ptr as *mut u32, rows);
            std::ptr::copy_nonoverlapping(inv.as_ptr(), r.2.ptr as *mut u32, rows);
        }
        // plans/84 B: 게더→전문가별 GEMV→스캐터를 배치 세션으로 — 비배치
        // run은 매 발사마다 제출+펜스 대기라 전문가 수만큼 동기가 걸린다
        // (프레임 경로 2.9배 열세의 주원인). 1회 제출로 묶는다.
        let batching = !llm170_diag::flag::on("LLM170_VK_NOBATCH");
        if batching {
            ctx.begin_batch()?;
        }
        // 4) 게더: xg[p] = xq[perm[p]] (u32 행, dst 스트라이드 = 패딩)
        {
            let p = self.pipeline(&mut ctx, Slot::PermuteU32)?;
            let ds2 = ctx.bind_ds(&p, &[xq, pmb, xgb])?;
            let push = push_u32s(&[xq_w as u32, xq_w_pad as u32, rows as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
        }
        // 5) 전문가별 GEMV — xg/yg 슬라이스 + 가중 전문가 오프셋.
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        let per_expert = w.data.len() / ne;
        // plans/86 §1b 진단 — 전문가 오프셋/청크 기하 (LLM170_MOE_SYNC=1).
        if llm170_diag::flag::on("LLM170_MOE_SYNC") {
            let sizes: Vec<usize> = {
                let wc = self.wcache.lock();
                wc.get(&(w.data.as_ptr() as usize, w.data.len()))
                    .map(|bs| bs.iter().map(|b| b.bytes).collect())
                    .unwrap_or_default()
            };
            eprintln!(
                "# moe-geom ty={ty} ne={ne} per_expert={per_expert} chunks={} sizes={sizes:?} max_ssbo={} ids={idv:?}",
                wbufs.len(),
                ctx.max_ssbo,
            );
        }
        for e in 0..ne {
            let r = off[e + 1] - off[e];
            if r == 0 {
                continue;
            }
            let xq_off = (off[e] * xq_w_pad * 4) as u64;
            let out_off = (off[e] * n_out * 4) as u64;
            let w_off = (e * per_expert) as u64;
            self.gemv_run_off(
                &mut ctx, &wbufs, n_in, n_out, xq_w_pad, ty, r, xgb, ygb, xq_off, out_off, w_off,
            )?;
        }
        // 6) 스캐터: out[inv^{-1}] — inv는 원본행→순열위치: out[i] = yg[inv[i]].
        {
            let p = self.pipeline(&mut ctx, Slot::PermuteF32)?;
            let ds2 = ctx.bind_ds(&p, &[ygb, ivb, ob])?;
            let push = push_u32s(&[n_out as u32, rows as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
        }
        if batching {
            ctx.end_batch_wait()?;
        }
        Ok(())
    }

    /// plans/88 P2 — 프레임 quant 출력용 디바이스 로컬 버퍼(성장 재할당).
    pub(super) fn xq_dev_buf(&self, ctx: &mut VkCtx, need: usize) -> Result<vk::Buffer, String> {
        let mut g = self.xq_dev.lock();
        if !g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
            *g = Some(crate::rawvk::context::site::scope("xq_dev", || {
                ctx.alloc(need.max(1 << 20))
            })?);
        }
        Ok(g.as_ref().unwrap().buf)
    }

    /// plans/104 — 격리 quant 버퍼(공유전문가 체인): xq_dev 와 독립 —
    /// RW 추적이 라우팅 GEMM 창과의 합법 병행을 증명할 수 있다.
    pub(super) fn xq2_dev_buf(&self, ctx: &mut VkCtx, need: usize) -> Result<vk::Buffer, String> {
        let mut g = self.xq2_dev.lock();
        if !g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
            *g = Some(crate::rawvk::context::site::scope("xq2_dev", || {
                ctx.alloc(need.max(1 << 20))
            })?);
        }
        Ok(g.as_ref().unwrap().buf)
    }

    /// plans/88 P1 — 프레임 op 진입 시 배치 재개(플러시 후 세그먼트 재결합).
    pub(super) fn frame_resume_batch(&self, ctx: &mut VkCtx) {
        if self
            .frame_step_batch
            .load(std::sync::atomic::Ordering::Relaxed)
            && !ctx.batching.load(std::sync::atomic::Ordering::Relaxed)
        {
            let _ = ctx.begin_batch();
        }
    }

    /// plans/87 §3 — 슬롯별 GPU 시간 집계 덤프(LLM170_VK_TS=1 로 풀 생성).
    /// 엔진의 ktrace 틱 지점(디코드 스텝/프리필 종료)에서 호출된다.
    pub fn ts_tick(&self) {
        let mut ctx = self.ctx.lock();
        ctx.ts_report();
    }
}
