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
        let q3q8 = std::env::var("LLM170_VK_Q3Q8").map(|v| v == "1").unwrap_or(false);
        let mut weights_owned: Option<Vec<(String, Vec<u8>, u32, usize, usize)>> = if q3q8 {
            Some(weights.iter().map(|(k, d, ty, ni, no)| (k.to_string(), d.to_vec(), *ty, *ni, *no)).collect())
        } else { None };
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
                        data, r as u64, k as u64, &mut row);
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
            Some(v) => v.iter().map(|(k, d, t, a, b)| (k.as_str(), d.as_slice(), *t, *a, *b)).collect(),
            None => weights.clone(),
        };
        // MTP 탑재·vocab — weights 이동 전 산출.
        let mtp_on = weights_final.iter().any(|(k, ..)| *k == "blk.64.nextn.eh_proj.weight");
        let n_vocab = weights_final
            .iter()
            .find(|(k, ..)| *k == "output.weight")
            // plans/29: 튜플 (k, data, ty, n_in, n_out) — 5번째가 n_out.
            // 종래 4번째(n_in)를 읽어 헤드가 어휘 5120행만 봄 (발산 근원).
            .map(|(_, _, _, _, no)| *no)
            .unwrap_or(n);
        // q5_K 원본 캡처 (i8 언패용 — 루프가 weights를 소비하기 전)
        // plans/40: 빌림 유지 — 클론 제거 (구 d.clone()가 q5 전체 ~8GB 복제)
        let q5k_src: Vec<(&str, &[u8], usize, usize)> = weights_final
            .iter()
            .filter(|(_, _, ty, _, _)| *ty == 13)
            .map(|(k, d, _, ni, no)| (*k, *d, *ni, *no))
            .collect();

        // f16 사전 디양자화 캐시 (plans/39) — 데이터 복제 없음(대여만):
        // 디양자화를 가중 업로드 루프 앞에서 수행 (RCA: .cloned() 전체복제가
        // 30Gi 호스트 RAM을 초과해 OOM·세션 사망의 원인이었음).
        let f16w_on = std::env::var("LLM170_VK_F16W").map(|v| v == "1").unwrap_or(false);
        let f16w_max = std::env::var("LLM170_VK_F16W_MAX").ok().and_then(|v| v.parse::<usize>().ok());
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
                let outs: std::sync::Mutex<Vec<(String, Vec<u16>)>> = std::sync::Mutex::new(Vec::new());
                std::thread::scope(|sc| {
                    for (name, data, ty, ni, no) in grp {
                        let outs = &outs;
                        sc.spawn(move || {
                            let gty = llm170_gguf::GgmlType::from_u32(*ty).unwrap_or(llm170_gguf::GgmlType::Q5K);
                            let (ni, no) = (*ni, *no);
                            let mut buf16 = vec![0u16; ni * no];
                            let mut row = vec![0f32; ni];
                            for r in 0..no {
                                llm170_core::quant::dequant_row(gty, data, r as u64, ni as u64, &mut row);
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
                    unsafe { std::ptr::copy_nonoverlapping(buf16.as_ptr() as *const u8, b.ptr, bytes) };
                    ctx.unmap(&mut b)?;
                    f16w.insert(name, b);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
            eprintln!("[f16w] 디양자화+업로드 {} 텐서 {}s", f16w.len(), e0.elapsed().as_secs_f32());
        }
        let mut w = HashMap::new();
        for &(name, data, ty, ni, no) in &weights_final {
            let mut bufs = Vec::new();
            let mut off = 0usize;
            // gemv4 WG() 시프트 산술 — 청크 크기 2의 거듭제곱. 마지막 청크는 실제 크기만
            // 할당: o = idx & mask 는 항상 청크 내 실데이터 오프셋이라 패딩 불필요.
            let ch_eff = data.len().next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
            while off < data.len() {
                let rem = data.len() - off;
                let sz = ch_eff.min(rem);
                let mut b = ctx.alloc(sz)?;
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
            unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), b.ptr as *mut f32, vals.len()) };
            cmap.insert(name, b);
        }
        // gemv 공유 테이블
        let kv: Vec<u32> = llm170_core::ktab2_packed();
        let mut ktab = ctx.alloc(1024)?;
        unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr(), ktab.ptr as *mut u32, 256) };
        ctx.unmap(&mut ktab)?;
        let mut grid3s = ctx.alloc(2048)?;
        unsafe { std::ptr::copy_nonoverlapping(llm170_core::IQ3S_GRID.as_ptr() as *const u8, grid3s.ptr, 2048) }; // iq3s 512워드 진테이블 (VkAcc ensure_shared 대칭)
        ctx.unmap(&mut grid3s)?;
        let dummy = ctx.alloc(16)?;
        let z16 = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(z16.as_ptr(), dummy.ptr, 16) };

        let n_full = is_recr.iter().filter(|&&r| !r).count();
        let n_recr = is_recr.len() - n_full;
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
        let kv_store: usize = if kv8 { kv_len / 32 * 34 } else { kv_len * 4 };
        let zeros_kv = vec![0u8; kv_store];
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
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, t.ptr, v.len() * 8) };
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
        let max_ssbo0 = ctx.max_ssbo;
        let (b_xs, b_xn, b_xq_n, b_xq_f, b_xq_g, b_gqkv, b_gconv, b_gq, b_gk, b_gv,
             b_gb, b_ga, b_gbg, b_gz, b_go, b_ggated, b_aq, b_ak, b_av, b_aout,
             b_gout, b_fgate, b_fup, b_fglu, b_fdown, b_am) = {
            // plans/43: 활성 버퍼는 디바이스 힙(캐브아웃)에 — 종전 GTT(시스템 RAM)는
            // 타일이 K블록마다 활성 타일을 읽을 때 대역 병목(가중의 수 배 트래픽).
            // 캐브아웃도 HOST_VISIBLE|COHERENT라 CPU 업로드 경로는 그대로 동작.
            let mut a = |sz: usize| -> Result<VkBuf, String> {
                ctx.alloc(sz.max(1) * 4).map_err(|e| e.to_string())
            };
            (
                a(T_MAX * n)?, a(T_MAX * n)?, a(T_MAX * xq_sn)?, a(T_MAX * xq_sf)?,
                a(T_MAX * xq_sg)?, a(T_MAX * conv_ch)?, a(T_MAX * conv_ch)?,
                a(T_MAX * k_len)?, a(T_MAX * k_len)?, a(T_MAX * v_len)?,
                a(T_MAX * hp.dt_rank)?, a(T_MAX * hp.dt_rank)?,
                a(T_MAX * hp.dt_rank * 2)?, a(T_MAX * hp.d_inner)?, a(T_MAX * v_len)?,
                a(T_MAX * hp.d_inner)?, a(T_MAX * n_head * 2 * hd)?,
                a(T_MAX * n_kv * hd)?, a(T_MAX * n_kv * hd)?, a(T_MAX * n_head * hd)?,
                a(T_MAX * n)?, a(T_MAX * hp.n_ff)?, a(T_MAX * hp.n_ff)?,
                a(T_MAX * hp.n_ff)?, a(T_MAX * n)?, a(8)?,
            )
        };
        // ── MTP (blk.64) 상주 상태 — has_mtp 시에만.
        let (mut mkk, mut mvv) = (Vec::new(), Vec::new());
        if mtp_on {
            let zeros = vec![0u8; kv_store];
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
        let b_lg = ah(n_vocab)?;
        let b_ams = ah(512)?;   // argmax 스테이지1 스크래치 (u32쌍 ×256WG)
        let b_xf16 = ah(T_MAX * n * 2)?;   // f16-B 활성 (plans/46, 요소수 T_MAX*n)
        let b_lg_t = ah(T_MAX * n_vocab)?;
        // ── q5_K i8 언패 (plans/23, gemm_i8) — CPU 병렬, 업로드 1회.
        let mut i8w: HashMap<String, I8W> = HashMap::new();
        let mut wsr_map: HashMap<String, VkBuf> = HashMap::new();
        for &(name, data, ni, no) in &q5k_src {
            let n_sub = ni / 32;
            let nblk = ni / 256;
            let mut w8 = vec![0i8; no * ni];
            let mut wsp = vec![0f32; no * n_sub];
            let mut wsm = vec![0f32; no * n_sub];
            std::thread::scope(|s| {
                let rows_per = (no / 8).max(1);
                let mut hs = Vec::new();
                for r0 in (0..no).step_by(rows_per) {
                    let end = (r0 + rows_per).min(no);
                    let p8: *mut Vec<i8> = &mut w8;
                    let pp: *mut Vec<f32> = &mut wsp;
                    let pm: *mut Vec<f32> = &mut wsm;
                    let (w8s, wsps, wsms) = unsafe { (&mut *p8, &mut *pp, &mut *pm) };
                    let data = &data;
                    hs.push(s.spawn(move || {
                        for o in r0..end {
                            for bidx in 0..nblk {
                                let wb0 = (o * nblk + bidx) * 176; // 행 오프셋 — 블록은 행 우선
                                let wb = &data[wb0..wb0 + 176];
                                let d = llm170_core::quant::f16(wb, 0);
                                let dm = llm170_core::quant::f16(wb, 2);
                                for j in 0..8 {
                                    let (sc, m) = llm170_core::quant::scale_min_k4_local(wb, j);
                                    let it = j / 2;
                                    let half = j % 2;
                                    let u: u8 = if half == 0 { 1u8 << (2 * it) } else { 2u8 << (2 * it) };
                                    let sb = bidx * 8 + j;
                                    wsps[o * n_sub + sb] = d * sc as f32;
                                    wsms[o * n_sub + sb] = dm * m as f32;
                                    for e in 0..32 {
                                        let q = wb[48 + it * 32 + e];
                                        let nib = if half == 0 { q & 0xF } else { q >> 4 };
                                        let hi = if wb[16 + e] & u != 0 { 16i8 } else { 0i8 };
                                        w8s[o * ni + sb * 32 + e] = nib as i8 + hi;
                                    }
                                }
                            }
                        }
                    }));
                }
                for h in hs { let _ = h.join(); }
            });
            // carveout + 언맵 — alloc_host(대형)는 i8 coopmatLoad 경로에서
            // 데이터 붕괴 실측 (미니 재현: 소형 carveout ★, 대형 host ✗).
            let wspbuf = {
                let mut b = ctx.alloc(no * n_sub * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(wsp.as_ptr(), b.ptr as *mut f32, no * n_sub) };
                ctx.unmap(&mut b)?;
                b
            };
            let wsmbuf = {
                let mut b = ctx.alloc(no * n_sub * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(wsm.as_ptr(), b.ptr as *mut f32, no * n_sub) };
                ctx.unmap(&mut b)?;
                b
            };
            // v2 (mlx식): 정확 디양자화 후 행별 재양자 — w8 덮어씀 + wsr.
            // w_deq = d·sc·q − dm·m (q = w_int). v1 wsp/wsm은 이미 계산됨.
            {
                let mut wsr_v = vec![0f32; no];
                for o in 0..no {
                    let mut mx = 0f32;
                    for b in 0..n_sub {
                        let s = wsp[o * n_sub + b];
                        let mn = wsm[o * n_sub + b];
                        let mut _isum_min = 0i64;
                        for e in 0..32 {
                            _isum_min += w8[o * ni + b * 32 + e] as i64;
                        }
                        // 값 범위: max|d·sc·q − dm·m| 근사 — 실제 최댓값은 원소별 계산
                        let hi = (s * 47.0).abs() + mn.abs();
                        let lo = mn.abs();
                        mx = mx.max(hi.max(lo));
                    }
                    // 정확 최댓값: 원소별 (느려도 init 1회)
                    mx = 0f32;
                    for b in 0..n_sub {
                        let s = wsp[o * n_sub + b];
                        let mn = wsm[o * n_sub + b];
                        for e in 0..32 {
                            let v = s * w8[o * ni + b * 32 + e] as f32 - mn;
                            mx = mx.max(v.abs());
                        }
                    }
                    let d = mx / 127.0f32;
                    let id = if d > 0.0 { 1.0f32 / d } else { 0.0f32 };
                    wsr_v[o] = d;
                    for b in 0..n_sub {
                        let s = wsp[o * n_sub + b];
                        let mn = wsm[o * n_sub + b];
                        for e in 0..32 {
                            let v = s * w8[o * ni + b * 32 + e] as f32 - mn;
                            w8[o * ni + b * 32 + e] = (v * id).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                let mut b = ctx.alloc(no * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(wsr_v.as_ptr(), b.ptr as *mut f32, no) };
                ctx.unmap(&mut b)?;
                wsr_map.insert(name.to_string(), b);
            }
            let wbuf = {
                let mut b = ctx.alloc(no * ni)?;
                unsafe { std::ptr::copy_nonoverlapping(w8.as_ptr() as *const u8, b.ptr, no * ni) };
                ctx.unmap(&mut b)?;
                b
            };
            i8w.insert(name.to_string(), I8W { w: wbuf, wsp: wspbuf, wsm: wsmbuf, n_out: no, n_in: ni });
        }
        let n_max = hp.n_ff.max(n);
        let n_sub_max = n_max / 32;
        let b8 = ctx.alloc_host(T_MAX * n_max)?;
        let ydb = ctx.alloc_host(T_MAX * n_sub_max * 4)?;
        let qsb = ctx.alloc_host(T_MAX * n_sub_max * 4)?;
        let wg_max = i8_wg_max(&i8w).max(640);
        let ishs = ctx.alloc_host(wg_max * 256 * 4)?;
        let faccs = ctx.alloc_host(wg_max * 256 * 4)?;
        Ok(Self {
            ctx,
            max_ssbo: max_ssbo0,
            ktimes: std::collections::HashMap::new(),
            ktime: std::env::var_os("LLM170_VK_KTIME").is_some(),
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
            b_xf16,
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
            m_kv_k: mkk,
            m_kv_v: mvv,
            snap_gdn: vec![Vec::new(); n_recr * n_seqs],
            snap_conv: vec![Vec::new(); n_recr * n_seqs],
            f16w,
            i8w,
            wsr: wsr_map,
            b8,
            np_conv_tbl,
            np_gdn_tbl,
            np_kvk_tbl,
            np_kv_v_tbl,
            np_pos,
            np_slot,
            b_amsc,
            b_amr,
            ydb,
            qsb,
            ishs,
            faccs,
        })
    }
}
