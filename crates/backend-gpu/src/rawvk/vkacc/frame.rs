//! vkacc::frame — FrameState — 프레임 버퍼/op 디스패치. (plans/90 B1b: gemv.rs 순수 이동)

use super::*;


impl llm170_core::matmul::FrameState for VkAcc {
    fn frame_begin(&self, t: usize) {
        self.frame_t.store(t.max(1), std::sync::atomic::Ordering::Relaxed);
        // plans/88 P1 — 스텝 수준 배치: 패스 전체를 세그먼트 최소 제출로 묶는다.
        // 비배치 run은 발사마다 제출+펜스 대기라 디코드 스텝(~2500발사)이
        // 호스트 간극에 지배됐다(실측 제출 2548/스텝). 스텝 도중 브리지의
        // frame_read가 플러시하면 프레임 op 진입마다 재개(frame_resume_batch).
        // 값경로는 이 게이트를 보지 않아 배치 상태가 새지 않는다.
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            self.frame_step_batch.store(true, std::sync::atomic::Ordering::Relaxed);
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
            self.fbuf(states)?, self.fbuf(q_scaled)?, self.fbuf(k)?,
            self.fbuf(v)?, self.fbuf(beta_ge)?, self.fbuf(out)?,
        );
        // plans/84 B: FN 상태는 전치 레이아웃(hip gdn_ar_w_swap과 동일 규약) —
        // grid (d, h_v), 상태 s[pair·d·d + u·d + …].
        let p = self.pipeline(&mut ctx, Slot::FnGdnArSwap)?;
        let ds2 = ctx.bind_ds(&p, &[sb, qb, kb, vb, bb, ob])?;
        let mut push = push_u32s(&[d as u32, (h_k * d) as u32, (h_v * d) as u32, h_v as u32, h_k as u32]);
        push.extend_from_slice(&1.0f32.to_le_bytes());
        push.extend_from_slice(&(t as u32).to_le_bytes());
        ctx.run(p.pl, ds2, p.pipe, &push, d as u32, h_v as u32, 1)
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
        ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(128), ((t * k_sel) as u32).div_ceil(4), 1)?;
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
        self.frame_op(&llm170_core::matmul::FrameOp::MoeWeightedSum { ys, wt, out, k: k_sel, n })
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
        // plans/89 P1.2 — ids dmmv 판이 이 호출을 가져갈 거면 xq 양자화 자체가
        // 불필요(f32 직결). 아래 조건은 ids2 분기와 동일해야 한다.
        let ids2_takes = rows > 0
            && (t == 1 || rows <= 64)
            && std::env::var("LLM170_MOE_IDS2").map(|v| v != "0").unwrap_or(true)
            && matches!(w.ty, GgmlType::Q4K | GgmlType::Q5_1);
        let xq = if ids2_takes {
            vk::Buffer::null()
        } else {
            let xq = self.xq_dev_buf(&mut ctx, rows * xq_w * 4)?;
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xb, xq])?;
            let push = push_u32s(&[n_in as u32, rows as u32, xq_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_in / 32) + 63) as u32 / 64, rows as u32, 1)?;
            xq
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
        if std::env::var_os("LLM170_MOE_IDS_DBG").is_some() {
            eprintln!("[moeids] ty={ty} rows={rows} t={t} n_in={n_in} n_out={n_out}");
        }
            // plans/89 P0.3 — ids dmmv 판 우선: llama dmmv 기하(64스레드·2행·
            // 서브그룹Add) + ids 간접, f32 활성 직결(MoE quant 불필요).
            // [ts] 기준선 moe_ids 30ms/step(43GB/s) — q8b급 150GB/s 기대.
            // 킬스위치 LLM170_MOE_IDS2=0(종전 fn_moe_ids).
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            if std::env::var("LLM170_MOE_IDS2").map(|v| v != "0").unwrap_or(true)
                && wbufs.len() == 1
            {
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
                    ctx.run(p.pl, ds2, p.pipe, &push, 1, n_out.div_ceil(2) as u32, rows as u32)?;
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
            let bound = rows + 16 * ne;
            let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
            let chunk_words = (ctx.max_ssbo / 4) as u32;
            let generation = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
            let hit = {
                let g = self.moe_grp.lock();
                g.as_ref().is_some_and(|g| crate::common::moe::cache_hit(g.generation, generation, g.rows, rows, g.ids_h == ids))
            };
            if !hit {
                // 테이블 성장(단일 상한 bound — 그룹 커널이 [rp,bound)를 0채움).
                crate::rawvk::context::site::scope("moe_grp", || -> Result<(), String> {
                    let mut g = self.moe_grp.lock();
                    let e = g.get_or_insert_with(|| MoeGrp {
                        generation: 0, rows: 0, ids_h: 0, bound: 0, off_n: 0,
                        off: vkbuf_null(), rows_pad: vkbuf_null(), tilexp: vkbuf_null(),
                        perm: vkbuf_null(), inv: vkbuf_null(), inv_pad: vkbuf_null(),
                        rowexp: vkbuf_null(), perm_pad: vkbuf_null(),
                        yg: vkbuf_null(), yg_rows: 0,
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
                    // off/rows_pad 도 bound/rows 와 동일 성장 가드 — MoeTop10가
                    // 매 스텝 moe_gen 을 올려 !hit 이 항상 참이 되므로, 무가드
                    // 재할당은 세대마다 구 버퍼를 누수시킨다(90 A2 실측 누수).
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
                    (g.off.buf, g.rows_pad.buf, g.tilexp.buf, g.perm.buf, g.inv.buf, g.inv_pad.buf, g.rowexp.buf, g.perm_pad.buf)
                };
                let dsg = ctx.bind_ds(&pg, &[idb, ob_, rpb, txb, pmb, ivb, ivpb, rxb, ppb])?;
                let push = push_u32s(&[ne as u32, rows as u32, bound as u32]);
                ctx.run(pg.pl, dsg, pg.pipe, &push, 1, 1, 1)?;
                {
                    let mut g = self.moe_grp.lock();
                    let gi = g.as_mut().unwrap();
                    gi.generation = generation;
                    gi.rows = rows;
                    gi.ids_h = ids;
                    gi.bound = gi.bound.max(bound);
                }
                if std::env::var_os("LLM170_MOE_GCHECK").is_some() {
                    // 진단: 그룹 테이블 불변식 검증(전문가 내 순서는 atomic이라
                    // 비결정 — 순서 무관 불변식으로 판정).
                    ctx.end_batch_wait()?;
                    let idv: Vec<u32> = {
                        let g = self.framebufs.lock();
                        let b = g.get(&ids).ok_or("ids 핸들 없음")?;
                        unsafe { std::slice::from_raw_parts(b.ptr as *const u32, rows) }.to_vec()
                    };
                    let gg = self.moe_grp.lock();
                    let gg = gg.as_ref().unwrap();
                    let rd = |b: &VkBuf, n: usize| unsafe {
                        std::slice::from_raw_parts(b.ptr as *const u32, n)
                    };
                    let dev_perm_pad = rd(&gg.perm_pad, bound);
                    let dev_inv_pad = rd(&gg.inv_pad, rows);
                    let dev_rowexp = rd(&gg.rowexp, bound);
                    let mut hoff = vec![0usize; ne + 1];
                    for &e in &idv { hoff[(e as usize).min(ne - 1) + 1] += 1; }
                    for e in 0..ne { hoff[e + 1] += hoff[e]; }
                    let mut hpoff = vec![0usize; ne + 1];
                    for e in 0..ne { hpoff[e + 1] = hpoff[e] + (hoff[e + 1] - hoff[e]).div_ceil(16) * 16; }
                    let rows_pad = hpoff[ne].max(16);
                    let rp_dev = unsafe { std::slice::from_raw_parts(gg.rows_pad.ptr as *const u32, 2) }[0] as usize;
                    let rp_dbg = unsafe { std::slice::from_raw_parts(gg.rows_pad.ptr as *const u32, 2) };
                    let mut bad = 0usize;
                    let mut seen = vec![false; rows];
                    if rp_dev != rows_pad {
                        eprintln!("[gcheck] rows_pad dev={rp_dev} host={rows_pad} dbg1={}", rp_dbg[1]);
                        bad += 1;
                    }
                    for pd in 0..rows_pad {
                        let e = dev_rowexp[pd] as usize;
                        if e >= ne || !(hpoff[e]..hpoff[e + 1]).contains(&pd) {
                            if bad < 6 { eprintln!("[gcheck] rowexp[{pd}]={e} 세그 불일치"); }
                            bad += 1;
                            continue;
                        }
                        let i = pd - hpoff[e];
                        let r = hoff[e + 1] - hoff[e];
                        let src = dev_perm_pad[pd] as usize;
                        if i < r {
                            if src >= rows || idv[src] as usize != e {
                                if bad < 6 { eprintln!("[gcheck] perm_pad[{pd}]={src} 전문가 불일치(e={e})"); }
                                bad += 1;
                            } else if seen[src] {
                                if bad < 6 { eprintln!("[gcheck] perm_pad[{pd}]={src} 중복"); }
                                bad += 1;
                            } else {
                                seen[src] = true;
                            }
                            if dev_inv_pad[src] as usize != pd {
                                if bad < 10 { eprintln!("[gcheck] inv_pad[{src}]={} != pd={pd}", dev_inv_pad[src]); }
                                bad += 1;
                            }
                        } else if src != 0 {
                            if bad < 6 { eprintln!("[gcheck] 패딩 perm_pad[{pd}]={src} != 0"); }
                            bad += 1;
                        }
                    }
                    let miss = seen.iter().filter(|s| !**s).count();
                    if miss > 0 && bad < 10 {
                        eprintln!("[gcheck] 커버 누락 {miss}행");
                    }
                    let dev_off2 = rd(&gg.off, ne + 1);
                    eprintln!("[gcheck] rows={rows} rows_pad={rows_pad} bad={bad} miss={miss} off[ne]={} off[0..3]={:?} rowexp[0..6]={:?} perm_pad[0..6]={:?}",
                        dev_off2[ne], &dev_off2[..3], &dev_rowexp[..6], &dev_perm_pad[..6]);
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
                (gi.rowexp.buf, gi.rows_pad.buf, gi.perm_pad.buf, gi.inv_pad.buf, gi.yg.buf)
            };
            // plans/89 P1.1b/d — coopmat 타일 우선(q4_K/q5_1): 스칼라 16×16 판
            // 대신 f16 coopMatMulAdd 전문가-블록 판(WG() 25비트 함정 제거판,
            // 스케일 분리 + f32 드레인). 킬스위치 LLM170_VK_MOECM=0.
            let cm_on = wbufs.len() == 1
                && std::env::var("LLM170_VK_MOECM").map(|v| v == "1").unwrap_or(false);
            let q4k_cm = cm_on
                && std::env::var("LLM170_VK_Q4KCM").map(|v| v != "0").unwrap_or(true);
            let slot = match (w.ty, q4k_cm) {
                (GgmlType::Q4K, _k) if wbufs.len() == 1 && std::env::var("LLM170_VK_Q4KKP").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kKp,
                (GgmlType::Q4K, true) if std::env::var("LLM170_VK_Q4KCM").map(|v| v == "2").unwrap_or(false) => Slot::FnMoeTileQ4kCm2,
                (GgmlType::Q4K, true) if std::env::var("LLM170_VK_Q4KSG1").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kSg1,
                (GgmlType::Q4K, true) if std::env::var("LLM170_VK_Q4KSC").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kSc,
                (GgmlType::Q4K, true) => Slot::FnMoeTileQ4kCm,
                // plans/89 재개: q51_sg1은 엔진 결정적 실측(5회 4동일+타이 1) —
                // 기본 경로로 승격(종전 스칼라는 킬스위치 LLM170_VK_Q51SG1=0).
                // 8sg판(q51_cm)과 q4k_sg1은 엔진 비결정 — 옵트인만.
                (GgmlType::Q5_1, _sg) if wbufs.len() == 1 && std::env::var("LLM170_VK_Q51SG1").map(|v| v != "0").unwrap_or(true) => Slot::FnMoeTileQ51Sg1,
                (GgmlType::Q5_1, _q51cm) if cm_on && std::env::var("LLM170_VK_Q51CM").map(|v| v != "0").unwrap_or(true) => Slot::FnMoeTileQ51Cm,
                // plans/93: q4k_sg1(coopmat 1-sg, 224 t/s) 기본 승격 — 구 스칼라
                // (160 t/s) 대비 +40%. 라우팅 민감도는 기본 경로와 동일(원장 36).
                // 안전장치: bound > 8192(pp16384급)에서는 GPU 행업 관측 — 구 스칼라로.
                // 킬스위치 LLM170_VK_Q4KSG1=0.
                (GgmlType::Q4K, _) if wbufs.len() == 1 && rows <= 8192
                    && std::env::var("LLM170_VK_Q4KSG1").map(|v| v != "0").unwrap_or(true) => Slot::FnMoeTileQ4kSg1,
                (GgmlType::Q5_1, _) => Slot::FnMoeTileQ51,
                (GgmlType::Q8_0, _) => Slot::FnMoeTileQ8,
                _ => Slot::FnMoeTileQ5k,
            };
            let p = self.pipeline(&mut ctx, slot)?;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xq);
            binds.push(ygb);
            binds.push(rxb);
            binds.push(rpb);
            binds.push(ppb);
            let ds2 = ctx.bind_ds(&p, &binds)?;
            // PC 선언순: n_in, n_out, per_expert_bytes, chunk_words, xq_w, mode, rows.
            let push = push_u32s(&[
                n_in as u32, n_out as u32, per_expert as u32, chunk_words, xq_w as u32,
                0u32, // mode — 실험 파생 잔여(90 A2): 프로덕션 항상 0
                rows as u32,
            ]);
            let (gx, gy) = if matches!(slot, Slot::FnMoeTileQ4kKp) {
                (n_out.div_ceil(4) as u32, bound.div_ceil(16) as u32)
            } else if matches!(slot, Slot::FnMoeTileQ51Sg1 | Slot::FnMoeTileQ4kSg1 | Slot::FnMoeTileQ4kSc) {
                // plans/93: 16열/WG 판 — cm_on 그리드(n_out/128)와 별개.
                (n_out.div_ceil(16) as u32, bound.div_ceil(16) as u32)
            } else if cm_on {
                (n_out.div_ceil(128) as u32, bound.div_ceil(16) as u32)
            } else {
                (n_out.div_ceil(16) as u32, bound.div_ceil(16) as u32)
            };
            ctx.run(p.pl, ds2, p.pipe, &push, gx, gy, 1)?;
            // 산란: out[i] = yg[inv_pad[i]] (행 순서 복원 — SiluMul/wsum 소비).
            let ps = self.pipeline(&mut ctx, Slot::PermuteF32)?;
            let dss = ctx.bind_ds(&ps, &[ygb, ivb, ob])?;
            let push = push_u32s(&[n_out as u32, rows as u32]);
            ctx.run(ps.pl, dss, ps.pipe, &push, rows as u32, 1, 1)?;
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
            unsafe { std::slice::from_raw_parts(b.ptr as *const u32, rows) }.to_vec()
        };
        let mut off = vec![0usize; ne + 1];
        for &e in &idv {
            off[(e as usize).min(ne - 1) + 1] += 1;
        }
        for e in 0..ne {
            off[e + 1] += off[e];
        }
        let mut cur = off[..ne].to_vec();
        let mut perm = vec![0u32; rows];
        for (i, &e) in idv.iter().enumerate() {
            let e = (e as usize).min(ne - 1);
            perm[cur[e]] = i as u32;
            cur[e] += 1;
        }
        let mut inv = vec![0u32; rows];
        for (p_, &orig) in perm.iter().enumerate() {
            inv[orig as usize] = p_ as u32;
        }
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
            let e = g.get_or_insert_with(|| (vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
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
        unsafe {
            let g = self.moebufs.lock();
            let r = g.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(perm.as_ptr(), r.0.ptr as *mut u32, rows);
            std::ptr::copy_nonoverlapping(inv.as_ptr(), r.2.ptr as *mut u32, rows);
        }
        // plans/84 B: 게더→전문가별 GEMV→스캐터를 배치 세션으로 — 비배치
        // run은 매 발사마다 제출+펜스 대기라 전문가 수만큼 동기가 걸린다
        // (프레임 경로 2.9배 열세의 주원인). 1회 제출로 묶는다.
        let batching = std::env::var_os("LLM170_VK_NOBATCH").is_none();
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
        if std::env::var_os("LLM170_MOE_SYNC").is_some() {
            let sizes: Vec<usize> = {
                let wc = self.wcache.lock();
                wc.get(&(w.data.as_ptr() as usize, w.data.len()))
                    .map(|bs| bs.iter().map(|b| b.bytes).collect())
                    .unwrap_or_default()
            };
            eprintln!(
                "# moe-geom ty={ty} ne={ne} per_expert={per_expert} chunks={} sizes={sizes:?} max_ssbo={} ids={idv:?}",
                wbufs.len(), ctx.max_ssbo,
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
            self.gemv_run_off(&mut ctx, &wbufs, n_in, n_out, xq_w_pad, ty, r, xgb, ygb, xq_off, out_off, w_off)?;
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
}

impl VkAcc {
    /// plans/88 P2 — 프레임 quant 출력용 디바이스 로컬 버퍼(성장 재할당).
    pub(super) fn xq_dev_buf(&self, ctx: &mut VkCtx, need: usize) -> Result<vk::Buffer, String> {
        let mut g = self.xq_dev.lock();
        if !g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
            *g = Some(crate::rawvk::context::site::scope("xq_dev", || ctx.alloc(need.max(1 << 20)))?);
        }
        Ok(g.as_ref().unwrap().buf)
    }

    /// plans/88 P1 — 프레임 op 진입 시 배치 재개(플러시 후 세그먼트 재결합).
    pub(super) fn frame_resume_batch(&self, ctx: &mut VkCtx) {
        if self.frame_step_batch.load(std::sync::atomic::Ordering::Relaxed)
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

