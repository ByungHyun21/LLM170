//! vk decoder 무게 업로드·초기화 (decoder/mod.rs에서 이동, plans/79 B).

use super::*;

impl DecoderState {
    /// 초기화 — 가중치(carveout)+상수(GTT) 업로드, 상태 0.
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        mut ctx: VkCtx,
        weights: Vec<(&'a str, &'a [u8], u32, usize, usize)>,
        consts: Vec<(String, Vec<f32>)>,
        hp: &llm170_core::qwen35::hparams::Hparams,
        is_recr: Vec<bool>,
        n_seqs: usize,
        ctx_len: usize,
    ) -> Result<Self, String> {
        let n = hp.n_embd;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let conv_ch = hp.conv_ch();
        let conv_k = 4;
        let k_len = n_group_len(hp);
        let v_len = hp.dt_rank * hp.d_state;
        let kv_len = ctx_len * n_kv * hd;
        let gdn_len = hp.dt_rank * hp.d_state * hp.d_state;
        let conv_len = (conv_k - 1) * conv_ch;
        // plans/30 q3q8 옵트인: 소유 사본 재팩을 먼저 수행하고 모든 소비자는
        // 최종 뷰(weights_final)를 본다 (기본 경로는 mmap 빌림 그대로 — 클론 0).
        let q3q8 = llm170_diag::flag::eq1("LLM170_VK_Q3Q8");
        let mut weights_owned: Option<Vec<(String, Vec<u8>, u32, usize, usize)>> = if q3q8 {
            Some(
                weights
                    .iter()
                    .map(|(k, d, ty, ni, no)| (k.to_string(), d.to_vec(), *ty, *ni, *no))
                    .collect(),
            )
        } else {
            None
        };
        if let Some(wv) = weights_owned.as_mut() {
            for (_name, data, ty, _ni, _no) in wv.iter_mut() {
                if *ty != 11 {
                    continue;
                }
                let (rows, k) = (*_no, *_ni);
                let mut out = Vec::with_capacity(rows * (k / 32) * 34);
                let mut row = vec![0.0f32; k];
                for r in 0..rows {
                    llm170_core::quant::dequant_row(
                        llm170_gguf::GgmlType::Q3K,
                        data,
                        r as u64,
                        k as u64,
                        &mut row,
                    );
                    for blk in row.chunks(32) {
                        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                        let d = amax / 127.0;
                        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                        let h = f32_to_f16_bits(d);
                        out.extend_from_slice(&h.to_le_bytes());
                        for &v in blk {
                            out.push(((v * id).round()).clamp(-127.0, 127.0) as i8 as u8);
                        }
                    }
                }
                *data = out;
                *ty = 8;
            }
        }
        let weights_final: Vec<(&str, &[u8], u32, usize, usize)> = match &weights_owned {
            Some(v) => v
                .iter()
                .map(|(k, d, t, a, b)| (k.as_str(), d.as_slice(), *t, *a, *b))
                .collect(),
            None => weights.clone(),
        };
        // MTP 탑재·vocab — weights 이동 전 산출.
        let mtp_on = weights_final
            .iter()
            .any(|(k, ..)| *k == "blk.64.nextn.eh_proj.weight");
        let n_vocab = weights_final
            .iter()
            .find(|(k, ..)| *k == "output.weight")
            // plans/29: 튜플 (k, data, ty, n_in, n_out) — 5번째가 n_out.
            // 종래 4번째(n_in)를 읽어 헤드가 어휘 5120행만 봄 (발산 근원).
            .map(|(_, _, _, _, no)| *no)
            .unwrap_or(n);
        // q5_K 원본 캡처 (i8 언패용 — 루프가 weights를 소비하기 전)
        // plans/40: 빌림 유지 — 클론 제거 (구 d.clone()가 q5 전체 ~8GB 복제)
        let f16w_on = llm170_diag::flag::eq1("LLM170_VK_F16W");
        let f16w_max =
            llm170_diag::flag::val("LLM170_VK_F16W_MAX").and_then(|v| v.parse::<usize>().ok());
        let mut f16w: HashMap<String, VkBuf> = HashMap::new();
        if f16w_on {
            let e0 = std::time::Instant::now();
            let mut cand: Vec<&(&str, &[u8], u32, usize, usize)> = weights_final
                .iter()
                .filter(|(_, _, ty, _, _)| matches!(*ty, 8 | 11 | 12 | 13 | 14 | 20 | 21 | 23))
                .collect();
            if let Some(mx) = f16w_max {
                cand.truncate(mx);
            }
            for grp in cand.chunks(8) {
                let outs: std::sync::Mutex<Vec<(String, Vec<u16>)>> =
                    std::sync::Mutex::new(Vec::new());
                std::thread::scope(|sc| {
                    for (name, data, ty, ni, no) in grp {
                        let outs = &outs;
                        sc.spawn(move || {
                            let gty = llm170_gguf::GgmlType::from_u32(*ty)
                                .unwrap_or(llm170_gguf::GgmlType::Q5K);
                            let (ni, no) = (*ni, *no);
                            let mut buf16 = vec![0u16; ni * no];
                            let mut row = vec![0f32; ni];
                            for r in 0..no {
                                llm170_core::quant::dequant_row(
                                    gty, data, r as u64, ni as u64, &mut row,
                                );
                                for (k, &v) in row.iter().enumerate() {
                                    buf16[r * ni + k] = f32_to_f16_bits(v);
                                }
                            }
                            outs.lock().unwrap().push((name.to_string(), buf16));
                        });
                    }
                });
                for (name, buf16) in outs.into_inner().unwrap() {
                    let bytes = buf16.len() * 2;
                    let mut b = ctx.alloc(bytes)?;
                    // SAFETY (107 W8): b는 bytes=f16 결과 크기(=buf16.len()*2)로 직전 alloc — 매핑 ptr 쓰기 범위 내, 아직 제출되지 않아 GPU 접근 없음.
                    unsafe {
                        std::ptr::copy_nonoverlapping(buf16.as_ptr() as *const u8, b.ptr, bytes)
                    };
                    ctx.unmap(&mut b)?;
                    f16w.insert(name, b);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
            eprintln!(
                "[f16w] 디양자화+업로드 {} 텐서 {}s",
                f16w.len(),
                e0.elapsed().as_secs_f32()
            );
        }
        let mut w = HashMap::new();
        for &(name, data, ty, ni, no) in &weights_final {
            let mut bufs = Vec::new();
            let mut off = 0usize;
            // gemv4 WG() 시프트 산술 — 청크 크기 2의 거듭제곱. 마지막 청크는 실제 크기만
            // 할당: o = idx & mask 는 항상 청크 내 실데이터 오프셋이라 패딩 불필요.
            let ch_eff = data
                .len()
                .next_power_of_two()
                .min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
            while off < data.len() {
                let rem = data.len() - off;
                let sz = ch_eff.min(rem);
                let mut b = ctx.alloc(sz)?;
                // SAFETY (107 W8): b는 이번 청크 sz로 할당 — .add(off)+sz ≤ data.len()(루프 불변식 off<data.len()), 쓰기는 할당 크기 이내.
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(off), b.ptr, sz) };
                ctx.unmap(&mut b)?;
                bufs.push(b);
                off += sz;
            }
            w.insert(name.to_string(), (bufs, ty, ni, no));
        }
        // 상수 — GTT (읽기 전용). "one"은 axpy 계수 1.0.
        let mut consts_in = consts;
        consts_in.push(("one".to_string(), vec![1.0f32]));
        let consts = consts_in;
        // 상수 — GTT (읽기 전용, 매핑 유지 무방하나 언맵)
        let mut cmap = HashMap::new();
        for (name, vals) in consts {
            let b = ctx.alloc_host(vals.len() * 4)?;
            // SAFETY (107 W8): b는 vals.len()*4 바이트 alloc_host — f32 원소수×4와 복사 길이 일치.
            unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), b.ptr as *mut f32, vals.len()) };
            // HOST_CACHED 비결합 타입 — flush 없이는 CPU 캐시라인이 메모리에 도달 전
            // GPU가 스테일 바이트를 읽는다(런마다 드레인 타이밍 의존 = 확산형 비결정
            // 원장 87/90의 근원, plans/135 §21-3 14차). conv_w 등 모델 상수가 이 경로.
            ctx.flush_range(&b, vals.len() * 4);
            cmap.insert(name, b);
        }
        // gemv 공유 테이블
        let kv: Vec<u32> = llm170_core::ktab2_packed();
        let mut ktab = ctx.alloc(1024)?;
        // SAFETY (107 W8): ktab은 1024바이트(=u32 256개)로 할당 — kv 256원소 기입과 정확히 일치.
        unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr(), ktab.ptr as *mut u32, 256) };
        ctx.unmap(&mut ktab)?;
        let mut grid3s = ctx.alloc(2048)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                llm170_core::IQ3S_GRID.as_ptr() as *const u8,
                grid3s.ptr,
                2048,
            )
        }; // iq3s 512워드 진테이블 (VkAcc ensure_shared 대칭)
        ctx.unmap(&mut grid3s)?;
        let dummy = ctx.alloc(16)?;
        let z16 = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(z16.as_ptr(), dummy.ptr, 16) };

        let n_full = is_recr.iter().filter(|&&r| !r).count();
        let n_recr = is_recr.len() - n_full;
        let zeros_kv = vec![0u8; kv_len * 4];
        let mut kv_k = Vec::with_capacity(n_full);
        let mut kv_v = Vec::with_capacity(n_full);
        for _ in 0..n_full {
            let mut ck = Vec::with_capacity(n_seqs);
            let mut cv = Vec::with_capacity(n_seqs);
            for _ in 0..n_seqs {
                let k = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_kv.as_ptr(), k.ptr, zeros_kv.len()) };
                let v = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_kv.as_ptr(), v.ptr, zeros_kv.len()) };
                ck.push(k);
                cv.push(v);
            }
            kv_k.push(ck);
            kv_v.push(cv);
        }
        let zeros_g = vec![0u8; gdn_len * 4];
        let zeros_c = vec![0u8; conv_len * 4];
        let mut st_gdn = Vec::with_capacity(n_recr);
        let mut st_conv = Vec::with_capacity(n_recr);
        for _ in 0..n_recr {
            let mut gd = Vec::with_capacity(n_seqs);
            let mut cv = Vec::with_capacity(n_seqs);
            for _ in 0..n_seqs {
                let g = ctx.alloc(gdn_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_g.as_ptr(), g.ptr, zeros_g.len()) };
                let c = ctx.alloc(conv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_c.as_ptr(), c.ptr, zeros_c.len()) };
                gd.push(g);
                cv.push(c);
            }
            st_gdn.push(gd);
            st_conv.push(cv);
        }
        // plans/91 P0 — np 배치 상태 주소 테이블 ([그룹][슬롯] u64). 디바이스
        let mut np_mk_tbl = |rows: &[Vec<VkBuf>]| -> Result<VkBuf, String> {
            // GL_EXT_buffer_reference 로 행별 상태를 직접 주소 지정한다.
            let mut v = Vec::with_capacity(rows.len() * n_seqs);
            for row in rows {
                for b in row {
                    v.push(ctx.buffer_va(b.buf));
                }
            }
            let t = ctx.alloc_host(v.len() * 8)?;
            // SAFETY (107 W8): t는 v.len()*8 바이트 alloc_host — u64 테이블 바이트 재해석 기입, 길이 일치.
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, t.ptr, v.len() * 8) };
            // HOST_CACHED 비결합 — 상수 테이블도 flush 의무 (§21-3 14차).
            ctx.flush_range(&t, v.len() * 8);
            Ok(t)
        };
        let np_conv_tbl = np_mk_tbl(&st_conv)?;
        let np_gdn_tbl = np_mk_tbl(&st_gdn)?;
        let np_kvk_tbl = np_mk_tbl(&kv_k)?;
        let np_kv_v_tbl = np_mk_tbl(&kv_v)?;
        let np_pos = ctx.alloc_host(n_seqs.max(1) * 4)?;
        let np_slot = ctx.alloc_host(n_seqs.max(1) * 4)?;
        // np greedy 행별 argmax(fn_argmax_rows 2단계) 스크래치.
        let am_wg = n_vocab.div_ceil(256 * 8);
        let b_amsc = ctx.alloc_host(2 * am_wg * T_MAX * 4)?;
        let b_amr = ctx.alloc_host(T_MAX * 4)?;

        let xq_sn = crate::rawvk::vkacc::xq_words(n);
        let xq_sf = crate::rawvk::vkacc::xq_words(hp.n_ff);
        let xq_sg = crate::rawvk::vkacc::xq_words(hp.d_inner);
        let (
            b_xs,
            b_xn,
            b_xq_n,
            b_xq_f,
            b_xq_g,
            b_gqkv,
            b_gconv,
            b_gq,
            b_gk,
            b_gv,
            b_gb,
            b_ga,
            b_gbg,
            b_gz,
            b_go,
            b_ggated,
            b_aq,
            b_ak,
            b_av,
            b_aout,
            b_gout,
            b_fgate,
            b_fup,
            b_fglu,
            b_fdown,
            b_am,
        ) = {
            // plans/43: 활성 버퍼는 디바이스 힙(캐브아웃)에 — 종전 GTT(시스템 RAM)는
            // 타일이 K블록마다 활성 타일을 읽을 때 대역 병목(가중의 수 배 트래픽).
            // 캐브아웃도 HOST_VISIBLE|COHERENT라 CPU 업로드 경로는 그대로 동작.
            let mut a = |sz: usize| -> Result<VkBuf, String> {
                ctx.alloc(sz.max(1) * 4).map_err(|e| e.to_string())
            };
            (
                a(T_MAX * n)?,
                a(T_MAX * n)?,
                a(T_MAX * xq_sn)?,
                a(T_MAX * xq_sf)?,
                a(T_MAX * xq_sg)?,
                a(T_MAX * conv_ch)?,
                a(T_MAX * conv_ch)?,
                a(T_MAX * k_len)?,
                a(T_MAX * k_len)?,
                a(T_MAX * v_len)?,
                a(T_MAX * hp.dt_rank)?,
                a(T_MAX * hp.dt_rank)?,
                a(T_MAX * hp.dt_rank * 2)?,
                a(T_MAX * hp.d_inner)?,
                a(T_MAX * v_len)?,
                a(T_MAX * hp.d_inner)?,
                a(T_MAX * n_head * 2 * hd)?,
                a(T_MAX * n_kv * hd)?,
                a(T_MAX * n_kv * hd)?,
                a(T_MAX * n_head * hd)?,
                a(T_MAX * n)?,
                a(T_MAX * hp.n_ff)?,
                a(T_MAX * hp.n_ff)?,
                a(T_MAX * hp.n_ff)?,
                a(T_MAX * n)?,
                a(8)?,
            )
        };
        // ── MTP (blk.64) 상주 상태 — has_mtp 시에만.
        let (mut mkk, mut mvv) = (Vec::new(), Vec::new());
        if mtp_on {
            let zeros = vec![0u8; kv_len * 4];
            for _ in 0..n_seqs {
                let k = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros.as_ptr(), k.ptr, zeros.len()) };
                let v = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros.as_ptr(), v.ptr, zeros.len()) };
                mkk.push(k);
                mvv.push(v);
            }
        }
        let xq_2n = crate::rawvk::vkacc::xq_words(2 * n);
        let mut ah = |sz: usize| -> Result<VkBuf, String> {
            ctx.alloc_host(sz.max(1) * 4).map_err(|e| e.to_string())
        };
        let m_e = ah(n)?;
        let m_cat = ah(2 * n)?;
        let m_xq2 = ah(xq_2n)?;
        let m_cur = ah(n)?;
        let m_xq = ah(xq_sn)?;
        let m_h = ah(n)?;
        // plans/91 P2 — MTP 프리필 배치 버퍼 (T_MAX행).
        let m_be = ah(T_MAX * n)?;
        let m_bcur = ah(T_MAX * n)?;
        let m_bhs = ah(T_MAX * n)?;
        let m_bcat = ah(T_MAX * 2 * n)?;
        let m_bxq2 = ah(T_MAX * xq_2n)?;
        let m_bxqn = ah(T_MAX * xq_sn)?;
        let m_prefetched = std::sync::atomic::AtomicBool::new(false);
        let b_lg = ah(n_vocab)?;
        let b_ams = ah(512)?; // argmax 스테이지1 스크래치 (u32쌍 ×256WG)
        let b_lg_t = ah(T_MAX * n_vocab)?;
        Ok(Self {
            ctx,
            ktimes: std::collections::HashMap::new(),
            ktime: llm170_diag::dump::opts().key("vk_ktime"),
            dbg_drain_ms: 0.0,
            kkey: std::cell::RefCell::new(None),
            w,
            consts: cmap,
            is_recr,
            n_layer: hp.n_layer,
            n_embd: n,
            n_ff: hp.n_ff,
            dt_rank: hp.dt_rank,
            d_state: hp.d_state,
            d_inner: hp.d_inner,
            n_group: hp.n_group,
            n_head,
            n_kv,
            hd,
            n_rot,
            conv_k,
            conv_ch,
            k_len,
            v_len,
            eps: hp.eps,
            kq_scale: 1.0 / (hd as f32).sqrt(),
            kv_k,
            kv_v,
            st_gdn,
            st_conv,
            ktab,
            grid3s,
            dummy,
            b_xs,
            b_xn,
            b_xq_n,
            b_xq_f,
            b_xq_g,
            b_gqkv,
            b_gconv,
            b_gq,
            b_gk,
            b_gv,
            b_gb,
            b_ga,
            b_gbg,
            b_gz,
            b_go,
            b_ggated,
            b_aq,
            b_ak,
            b_av,
            b_aout,
            b_gout,
            b_fgate,
            b_fup,
            b_fglu,
            b_fdown,
            b_lg,
            b_ams,
            b_lg_t,
            b_am,
            pipes: HashMap::new(),
            split_ctr: 0,
            mtp_on,
            n_vocab,
            m_e,
            m_cat,
            m_xq2,
            m_cur,
            m_xq,
            m_h,
            m_be,
            m_bcur,
            m_bhs,
            m_bcat,
            m_bxq2,
            m_bxqn,
            m_prefetched,
            m_kv_k: mkk,
            m_kv_v: mvv,
            gdn_snap: None,
            f16w,
            np_conv_tbl,
            np_gdn_tbl,
            np_kvk_tbl,
            np_kv_v_tbl,
            np_pos,
            np_slot,
            b_amsc,
            b_amr,
        })
    }
}
