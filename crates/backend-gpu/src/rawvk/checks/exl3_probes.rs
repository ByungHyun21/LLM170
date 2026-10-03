//! EXL3 모듈 격리 프로브(plans/121 F2) — 속도·산술 격리 작업장.
//! 사용자 확정 워크플로: 신규/변경 모듈은 본 프로브에서 합성/실캡처 입력으로
//! 검증 후 본체 부착. 프로브 반복 주기 수 초(전모델 corr 루프의 1/100).
//!
//! 측정 원장(2026-10-03):
//! - scan v3: kern-vs-f16mirror 2.6e-4(실캡처+랜덤S0), T512 9.3ms/디스패치
//!   (v1 53.4ms에서 5.7배). 부착 후 pp512 72.55→96.82 t/s(+33.5%), corr 0.999998.
//! - attn fwd3: maxdiff 5.3e-7, T512 8.2ms/층(CPU 38.5ms에서 4.7배). 부착 후
//!   pp512 96.82→109.33 t/s(+13%), corr 0.999999, greedy 8/8.
//! - gemm: 전 선형 형상 7.6-8.4 TF(피크 18%) — 트렐리스 디코드 25-30%·t-슬랩 중복.
//!   타일 기하 BM64/BN128/BK32 동결(변형 전부 드라이버 크래시).
//! - norm_resid: xn 2.6e-6·xo 0.0(얼라이어싱/행오프셋 2버그 수정 후).
//!
//! env: LLM170_EXL3_SCAN_CAP(캡처 경로)·LLM170_EXL3_SCAN_ST0(랜덤 초기상태).

use super::exl3_resident::TrellisResident;

// ── scan 모듈 독립 프로브(plans/121 F2) ──
// 모델 적재 없이 합성 입력으로 scan 커널만 검증: 속도·산술 격리 작업장.
// Rust f32 기준(커널 수식 미러)과 행별 출력·최종 상태를 직접 비교한다.
pub fn scan_check(t_len: usize, cap_path: &str) -> Result<String, String> {
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    let use_cap = !cap_path.is_empty();
    if use_cap {
        let raw = std::fs::read(cap_path).map_err(|e| format!("cap read: {e}"))?;
        let nf = raw.len() / 4;
        let fl: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, nf) }.to_vec();
        let n_q = t_len * 2048;
        let mut o = 0usize;
        let take = |o: &mut usize, n: usize| -> Vec<f32> {
            let v = fl[*o..*o + n].to_vec();
            *o += n;
            v
        };
        let q = take(&mut o, n_q);
        let k = take(&mut o, n_q);
        let v = take(&mut o, t_len * 6144);
        let bg = take(&mut o, t_len * 96);
        return scan_check_run(q, k, v, bg, t_len, true);
    }
    // 합성 입력(LCG) — q/k는 L2 정규화 후 스케일(≈1/√128), beta∈(0,1), g=음수.
    let mut seed: u32 = 0x1234_5678;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    let mut q = vec![0f32; t_len * HK * D];
    let mut k = vec![0f32; t_len * HK * D];
    let mut v = vec![0f32; t_len * HV * D];
    let mut bg = vec![0f32; t_len * 2 * HV];
    for e in q.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.09;
    }
    for e in k.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.09;
    }
    for e in v.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.5;
    }
    for t in 0..t_len {
        for h in 0..HV {
            bg[t * 2 * HV + h] = rnd();
            bg[t * 2 * HV + HV + h] = -rnd() * 2.0;
        }
    }

    scan_check_run(q, k, v, bg, t_len, false)
}

fn scan_check_run(
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    bg: Vec<f32>,
    t_len: usize,
    from_cap: bool,
) -> Result<String, String> {
    let mut ctx = crate::rawvk::context::VkCtx::new()?;
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    let gq = ctx.alloc_host_cached(t_len.max(64) * HK * D * 4)?;
    let gk = ctx.alloc_host_cached(t_len.max(64) * HK * D * 4)?;
    let gv = ctx.alloc_host_cached(t_len.max(64) * HV * D * 4)?;
    let gbg = ctx.alloc_host_cached(t_len.max(64) * 2 * HV * 4)?;
    let go = ctx.alloc_host_cached(t_len.max(64) * HV * D * 4)?;
    let gstate = ctx.alloc_host_cached(HV * D * D * 4)?; // 1층분
    let st0: Vec<f32> = if llm170_diag::flag::on("LLM170_EXL3_SCAN_ST0") {
        let mut sd: u32 = 0xC0FF_EE01;
        (0..HV * D * D)
            .map(|_| {
                sd = sd.wrapping_mul(1664525).wrapping_add(1013904223);
                ((sd >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
            })
            .collect()
    } else {
        vec![0f32; HV * D * D]
    };
    unsafe {
        std::ptr::copy_nonoverlapping(q.as_ptr(), gq.ptr as *mut f32, q.len());
        std::ptr::copy_nonoverlapping(k.as_ptr(), gk.ptr as *mut f32, k.len());
        std::ptr::copy_nonoverlapping(v.as_ptr(), gv.ptr as *mut f32, v.len());
        std::ptr::copy_nonoverlapping(bg.as_ptr(), gbg.ptr as *mut f32, bg.len());
        std::ptr::copy_nonoverlapping(st0.as_ptr(), gstate.ptr as *mut f32, st0.len());
        std::ptr::write_bytes(go.ptr, 0, t_len.max(64) * HV * D * 4);
    }
    ctx.flush_buf(&gq);
    ctx.flush_buf(&gk);
    ctx.flush_buf(&gv);
    ctx.flush_buf(&gbg);
    ctx.flush_buf(&gstate);
    let pgs = ctx.pipeline_pipes(include_bytes!("../spv/exl3_gdn_scan.spv"), 6, 20)?;
    let dispatch = |ctx: &mut crate::rawvk::context::VkCtx| -> Result<(), String> {
        ctx.begin_batch()?;
        let ds = ctx.fresh_ds_for(&pgs, 6)?;
        ctx.bind_bufs(ds, &[gq.buf, gk.buf, gv.buf, gbg.buf, gstate.buf, go.buf]);
        let push: Vec<u8> = [t_len as u32, HK as u32, HV as u32, D as u32, 0u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_scan_probe");
        ctx.run_rw(
            pgs.pl,
            ds,
            pgs.pipe,
            &push,
            HV as u32,
            1,
            1,
            &[gq.buf, gk.buf, gv.buf, gbg.buf, gstate.buf],
            &[go.buf, gstate.buf],
        )?;
        ctx.end_batch_wait()?;
        ctx.wait_pending()?;
        Ok(())
    };
    dispatch(&mut ctx)?;
    // 시간 측정(5회 중앙값)
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..5 {
        unsafe {
            std::ptr::copy_nonoverlapping(st0.as_ptr(), gstate.ptr as *mut f32, st0.len());
        }
        ctx.flush_buf(&gstate);
        let t0 = std::time::Instant::now();
        dispatch(&mut ctx)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 판독
    ctx.invalidate_buf(&go);
    ctx.invalidate_buf(&gstate);
    let out_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(go.ptr as *const f32, t_len * HV * D).to_vec() };
    let st_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(gstate.ptr as *const f32, HV * D * D).to_vec() };

    // core 기준(gdn_chunk_seq — CPU f32, v1과 동일 경로)
    let mut beta_v = vec![0f32; t_len * HV];
    let mut g_v = vec![0f32; t_len * HV];
    for t in 0..t_len {
        for h in 0..HV {
            beta_v[t * HV + h] = bg[t * 2 * HV + h];
            g_v[t * HV + h] = bg[t * 2 * HV + HV + h];
        }
    }
    let mut st_core = st0.clone();
    let mut out_core = vec![0f32; t_len * HV * D];
    llm170_core::gdn::gdn_chunk_seq(
        &q,
        &k,
        &v,
        &beta_v,
        &g_v,
        &mut st_core,
        &mut out_core,
        t_len,
        HK,
        HV,
    );
    let mut kern_vs_core = 0f32;
    for i in 0..out_gpu.len() {
        kern_vs_core = kern_vs_core.max((out_gpu[i] - out_core[i]).abs());
    }
    let mut stc_max = 0f32;
    for i in 0..st_gpu.len() {
        stc_max = stc_max.max((st_gpu[i] - st_core[i]).abs());
    }

    // Rust f32 기준 — 커널 수식 미러(CS=32)
    let (out_ref, st_ref) = scan_ref(&q, &k, &v, &bg, t_len, false, &st0);
    let (out_ref16, _) = scan_ref(&q, &k, &v, &bg, t_len, true, &st0);
    let mut kern_vs_f16ref = 0f32;
    for i in 0..out_gpu.len() {
        kern_vs_f16ref = kern_vs_f16ref.max((out_gpu[i] - out_ref16[i]).abs());
    }
    let mut mirror_vs_core = 0f32;
    for i in 0..out_ref.len() {
        mirror_vs_core = mirror_vs_core.max((out_ref[i] - out_core[i]).abs());
    }

    let _ = from_cap;
    let mut out_max = 0f32;
    let mut out_rel = 0f64;

    for i in 0..out_gpu.len() {
        let d = (out_gpu[i] - out_ref[i]).abs();
        out_max = out_max.max(d);
        let denom = out_ref[i].abs().max(1e-3);
        out_rel = out_rel.max(d as f64 / denom as f64);
    }
    let mut st_max = 0f32;
    for i in 0..st_gpu.len() {
        st_max = st_max.max((st_gpu[i] - st_ref[i]).abs());
    }
    Ok(format!(
        "scan-check T={t_len}: kern-vs-mirror={out_max:.3e} · kern-vs-f16mirror={kern_vs_f16ref:.3e} · mirror(CS32)-vs-core(CS64)={mirror_vs_core:.3e} · st(kern-vs-core)={stc_max:.3e} · kernel {:.2}ms (5회 중앙값)",
        times[2]
    ))
}

/// scan 커널의 f32 기준 미러 — A/KQ/sk/sv를 f32로 계산(커널의 f16과의 차이가
/// 판정 대상). CS 고정 32.
fn h16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn scan_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bg: &[f32],
    t_len: usize,
    f16_emul: bool,
    st0: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    const CS: usize = 32;
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    let qscale = 1.0f32 / (D as f32).sqrt();
    let mut st = st0.to_vec();
    let mut out = vec![0f32; t_len * HV * D];
    let n_chunks = t_len.div_ceil(CS);
    for c in 0..n_chunks {
        let t0 = c * CS;
        let n = (t_len - t0).min(CS);
        for h in 0..HV {
            let kh = h % HK;
            let mut sk = [[0f32; D]; CS];
            let mut sv = [[0f32; D]; CS];
            let mut bp = [0f32; CS];
            let mut gcs = [0f32; CS + 1];
            for i in 0..CS {
                let live = i < n;
                if live {
                    for s2 in 0..D {
                        let kv2 = k[(t0 + i) * HK * D + kh * D + s2];
                        let vv2 = v[(t0 + i) * HV * D + h * D + s2];
                        sk[i][s2] = if f16_emul { h16(kv2) } else { kv2 };
                        sv[i][s2] = if f16_emul { h16(vv2) } else { vv2 };
                    }
                    bp[i] = bg[(t0 + i) * 2 * HV + h];
                }
            }
            let mut acc = 0f32;
            for t in 0..CS {
                acc += if t < n {
                    bg[(t0 + t) * 2 * HV + HV + h]
                } else {
                    0.0
                };
                gcs[t] = acc;
            }
            gcs[CS] = acc;
            let mut a = [[0f32; CS]; CS];
            let mut kq = [[0f32; CS]; CS];
            for i in 0..n {
                for j in 0..=i {
                    let mut dk = 0f32;
                    let mut dq = 0f32;
                    for s2 in 0..D {
                        dk += sk[i][s2] * sk[j][s2];
                        dq += q[(t0 + i) * HK * D + kh * D + s2] * sk[j][s2];
                    }
                    if j < i {
                        let a2 = dk * bp[i] * (gcs[i] - gcs[j]).exp();
                        a[i][j] = if f16_emul { h16(a2) } else { a2 };
                    }
                    let kq2 = dq * qscale * (gcs[i] - gcs[j]).exp();
                    kq[i][j] = if f16_emul { h16(kq2) } else { kq2 };
                }
            }
            // ks/qs: [CS][D]
            let mut ks = [[0f32; D]; CS];
            let mut qs = [[0f32; D]; CS];
            for i in 0..n {
                for col in 0..D {
                    let mut ak = 0f32;
                    let mut aq = 0f32;
                    for s2 in 0..D {
                        let s_el = st[h * D * D + s2 * D + col];
                        ak += sk[i][s2] * s_el;
                        aq += q[(t0 + i) * HK * D + kh * D + s2] * s_el;
                    }
                    ks[i][col] = ak;
                    qs[i][col] = aq * qscale;
                }
            }
            let mut dc = [[0f32; D]; CS];
            for i in 0..n {
                for col in 0..D {
                    let mut rhs = bp[i] * (sv[i][col] - gcs[i].exp() * ks[i][col]);
                    for j in 0..i {
                        rhs -= a[i][j] * dc[j][col];
                    }
                    dc[i][col] = rhs;
                    let mut oi = gcs[i].exp() * qs[i][col];
                    for p in 0..=i {
                        oi += kq[i][p] * dc[p][col];
                    }
                    out[(t0 + i) * HV * D + h * D + col] = oi;
                }
            }
            let gt_exp = gcs[CS].exp();
            let mut wsm = [0f32; CS];
            for j in 0..CS {
                wsm[j] = if j < n { (gcs[CS] - gcs[j]).exp() } else { 0.0 };
            }
            for s2 in 0..D {
                for col in 0..D {
                    let base = h * D * D + s2 * D + col;
                    let mut a2 = st[base] * gt_exp;
                    for j in 0..n {
                        a2 += sk[j][s2] * wsm[j] * dc[j][col];
                    }
                    st[base] = a2;
                }
            }
        }
    }
    (out, st)
}

// ── 어텐션 모듈 독립 프로브(plans/121 F2b) ──
// 합성 q‖gate/k/v + 규격 노름으로 prep+fwd 2커널만 검증: 속도·산술 격리 작업장.
// Rust 미러는 core::ops::rope_head를 직접 재사용(수학 단일 진실 공급원).
pub fn attn_check(t_len: usize, pos0: usize) -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
    const NH: usize = 24;
    const NKV: usize = 4;
    const D: usize = 256;
    let mut ctx = VkCtx::new()?;

    let mut seed: u32 = 0xBEEF_5A17;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    let mut qg = vec![0f32; t_len * NH * D * 2];
    let mut kin = vec![0f32; t_len * NKV * D];
    let mut vin = vec![0f32; t_len * NKV * D];
    let mut qnw = vec![0f32; D];
    let mut knw = vec![0f32; D];
    for e in qg.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.5;
    }
    for e in kin.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.3;
    }
    for e in vin.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.8;
    }
    for e in qnw.iter_mut() {
        *e = 0.9 + rnd() * 0.2;
    }
    for e in knw.iter_mut() {
        *e = 0.9 + rnd() * 0.2;
    }

    let cap = 1024usize;
    let b_qg = ctx.alloc_host_cached(t_len.max(64) * NH * D * 2 * 4)?;
    let b_k = ctx.alloc_host_cached(t_len.max(64) * NKV * D * 4)?;
    let b_v = ctx.alloc_host_cached(t_len.max(64) * NKV * D * 4)?;
    let b_qnw = ctx.alloc_host_cached(D * 4)?;
    let b_knw = ctx.alloc_host_cached(D * 4)?;
    let b_qh = ctx.alloc_host_cached(t_len.max(64) * NH * D * 4)?;
    let b_kc = ctx.alloc_host_cached(cap * NKV * D * 4)?;
    let b_vc = ctx.alloc_host_cached(cap * NKV * D * 4)?;
    let b_out = ctx.alloc_host_cached(t_len.max(64) * NH * D * 4)?;
    unsafe {
        std::ptr::copy_nonoverlapping(qg.as_ptr(), b_qg.ptr as *mut f32, qg.len());
        std::ptr::copy_nonoverlapping(kin.as_ptr(), b_k.ptr as *mut f32, kin.len());
        std::ptr::copy_nonoverlapping(vin.as_ptr(), b_v.ptr as *mut f32, vin.len());
        std::ptr::copy_nonoverlapping(qnw.as_ptr(), b_qnw.ptr as *mut f32, D);
        std::ptr::copy_nonoverlapping(knw.as_ptr(), b_knw.ptr as *mut f32, D);
        std::ptr::write_bytes(b_kc.ptr, 0, cap * NKV * D * 4);
        std::ptr::write_bytes(b_vc.ptr, 0, cap * NKV * D * 4);
        std::ptr::write_bytes(b_out.ptr, 0, t_len.max(64) * NH * D * 4);
    }
    for b in [&b_qg, &b_k, &b_v, &b_qnw, &b_knw] {
        ctx.flush_buf(b);
    }
    let ppb = ctx.alloc_host_cached(16)?;
    unsafe {
        std::ptr::write_bytes(ppb.ptr, 0, 16);
    }
    ctx.flush_buf(&ppb);
    let pp = ctx.pipeline_pipes(include_bytes!("../spv/exl3_attn_prep.spv"), 9, 8)?;
    let pf = ctx.pipeline_pipes(include_bytes!("../spv/exl3_attn_fwd3.spv"), 6, 8)?;

    let run = |ctx: &mut VkCtx| -> Result<(), String> {
        ctx.begin_batch()?;
        let d1 = ctx.fresh_ds_for(&pp, 8)?;
        ctx.bind_bufs(
            d1,
            &[
                b_qg.buf, b_k.buf, b_v.buf, b_qnw.buf, b_knw.buf, b_qh.buf, b_kc.buf, b_vc.buf,
            ],
        );
        let push1: Vec<u8> = [t_len as u32, pos0 as u32, 0u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_attn_prep");
        ctx.run_rw(
            pp.pl,
            d1,
            pp.pipe,
            &push1,
            t_len as u32,
            28,
            1,
            &[b_qg.buf, b_k.buf, b_v.buf],
            &[b_qh.buf, b_kc.buf, b_vc.buf],
        )?;
        let d2 = ctx.fresh_ds_for(&pf, 5)?;
        ctx.bind_bufs(d2, &[b_qh.buf, b_kc.buf, b_vc.buf, b_qg.buf, b_out.buf]);
        let push2: Vec<u8> = [t_len as u32, pos0 as u32, 0u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_attn_fwd");
        ctx.run_rw(
            pf.pl,
            d2,
            pf.pipe,
            &push2,
            t_len as u32,
            24,
            1,
            &[b_qh.buf, b_kc.buf, b_vc.buf, b_qg.buf],
            &[b_out.buf],
        )?;
        ctx.end_batch_wait()?;
        ctx.wait_pending()?;
        Ok(())
    };
    run(&mut ctx)?;
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..5 {
        unsafe {
            std::ptr::write_bytes(b_kc.ptr, 0, cap * NKV * D * 4);
            std::ptr::write_bytes(b_vc.ptr, 0, cap * NKV * D * 4);
        }
        ctx.flush_buf(&b_kc);
        ctx.flush_buf(&b_vc);
        let t0 = std::time::Instant::now();
        run(&mut ctx)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ctx.invalidate_buf(&b_out);
    let out_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(b_out.ptr as *const f32, t_len * NH * D).to_vec() };

    ctx.invalidate_buf(&b_qh);
    ctx.invalidate_buf(&b_kc);
    let qh_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(b_qh.ptr as *const f32, t_len * NH * D).to_vec() };
    let kc_gpu: Vec<f32> = unsafe {
        std::slice::from_raw_parts(b_kc.ptr as *const f32, (pos0 + t_len) * NKV * D).to_vec()
    };
    let (qh_ref, kc_ref, out_ref) = attn_ref2(&qg, &kin, &vin, &qnw, &knw, t_len, pos0);
    let mut qh_max = 0f32;
    for i in 0..qh_gpu.len() {
        qh_max = qh_max.max((qh_gpu[i] - qh_ref[i]).abs());
    }
    let mut kc_max = 0f32;
    for i in 0..kc_gpu.len() {
        kc_max = kc_max.max((kc_gpu[i] - kc_ref[i]).abs());
    }
    eprintln!("  [attndbg] qh maxdiff={qh_max:.3e} kc maxdiff={kc_max:.3e}");
    let mut out_max = 0f32;
    let mut out_rel = 0f64;
    for i in 0..out_gpu.len() {
        let d = (out_gpu[i] - out_ref[i]).abs();
        out_max = out_max.max(d);
        let denom = out_ref[i].abs().max(1e-3);
        out_rel = out_rel.max(d as f64 / denom as f64);
    }
    Ok(format!(
        "attn-check T={t_len} pos0={pos0}: maxdiff={out_max:.3e} rel={out_rel:.3e} · {times:.2?}ms"
    ))
}

fn attn_ref2(
    qg: &[f32],
    kin: &[f32],
    vin: &[f32],
    qnw: &[f32],
    knw: &[f32],
    t_len: usize,
    pos0: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    const NH: usize = 24;
    const NKV: usize = 4;
    const D: usize = 256;
    let mut out = vec![0f32; t_len * NH * D];
    let mut qh_all = vec![0f32; t_len * NH * D];
    let mut kcache = vec![0f32; (pos0 + t_len) * NKV * D];
    let mut vcache = vec![0f32; (pos0 + t_len) * NKV * D];
    for t in 0..t_len {
        let pos = pos0 + t;
        for hh in 0..NKV {
            let src = t * NKV * D + hh * D;
            let mut head: Vec<f32> = kin[src..src + D].to_vec();
            let ss: f32 = head.iter().map(|x| x * x).sum();
            let inv = 1.0 / ((ss / D as f32 + 1e-6).sqrt());
            for d in 0..D {
                head[d] *= inv * knw[d];
            }
            llm170_core::ops::rope_head(&mut head, pos as u32, 64, 1e7);
            let kb = pos * NKV * D + hh * D;
            kcache[kb..kb + D].copy_from_slice(&head);
            vcache[kb..kb + D].copy_from_slice(&vin[src..src + D]);
        }
    }
    for t in 0..t_len {
        let kv_len = pos0 + t + 1;
        for hh in 0..NH {
            let kh = hh / 6;
            let src = t * NH * D * 2 + hh * D * 2;
            let mut q: Vec<f32> = qg[src..src + D].to_vec();
            let ss: f32 = q.iter().map(|x| x * x).sum();
            let inv = 1.0 / ((ss / D as f32 + 1e-6).sqrt());
            for d in 0..D {
                q[d] *= inv * qnw[d];
            }
            llm170_core::ops::rope_head(&mut q, (pos0 + t) as u32, 64, 1e7);
            qh_all[t * NH * D + hh * D..t * NH * D + hh * D + D].copy_from_slice(&q);
            let scale = 1.0f32 / (D as f32).sqrt();
            let mut scores = vec![0f32; kv_len];
            for (i, s) in scores.iter_mut().enumerate() {
                let kb = i * NKV * D + kh * D;
                *s = q
                    .iter()
                    .zip(&kcache[kb..kb + D])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut acc = vec![0f32; D];
            let mut wsum = 0f32;
            for i in 0..kv_len {
                let wgt = (scores[i] - mx).exp();
                wsum += wgt;
                let vb = i * NKV * D + kh * D;
                for d in 0..D {
                    acc[d] += wgt * vcache[vb + d];
                }
            }
            for d in 0..D {
                let g = qg[t * NH * D * 2 + hh * D * 2 + D + d];
                let sg = 1.0 / (1.0 + (-g).exp());
                out[t * NH * D + hh * D + d] = acc[d] / wsum * sg;
            }
        }
    }
    (qh_all, kcache, out)
}

// ── 전 모듈 격리 프로브(plans/121 F2c) ──
// 실모델의 모든 선형 형상에 대해 GEMM만 단독 측정 — 형상별 유효 TFLOPS로
// 숨은 타일 비효율을 노출한다(사용자 지시: 전 모듈 격리 점검).
pub fn gemm_check(dir: &str) -> Result<String, String> {
    let mut tr = TrellisResident::load(dir)?;
    let t_rows = 512usize;
    let stage = tr.stage_f32()?;
    // 입력: 균일 값(수치 무의미 — 속도 프로브)
    unsafe {
        std::ptr::write_bytes(stage, 0, t_rows * 6144 * 4);
        for t in 0..t_rows {
            let p = stage.add(t * 6144);
            for i in 0..6144usize {
                *p.add(i) = ((i % 17) as f32 - 8.0) * 0.01;
            }
        }
    }
    let mut report = Vec::new();
    let shapes: &[(&str, &str)] = &[
        (
            "GDN qkv",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
        ),
        (
            "GDN z",
            "model.language_model.layers.0.linear_attn.in_proj_z",
        ),
        (
            "GDN out",
            "model.language_model.layers.0.linear_attn.out_proj",
        ),
        ("ATTN q", "model.language_model.layers.3.self_attn.q_proj"),
        ("FFN gate", "model.language_model.layers.0.mlp.gate_proj"),
        ("FFN down", "model.language_model.layers.0.mlp.down_proj"),
        ("lm_head", "lm_head"),
    ];
    // 산술 검증 추가: T=8 배치 1행 vs 순차 GEMV 기준(BK=64 변형 판정용)
    {
        let t8 = 8usize;
        let st8 = tr.stage_f32()?;
        unsafe {
            for t in 0..t8 {
                let p8 = st8.add(t * 5120);
                for i in 0..5120usize {
                    *p8.add(i) = ((i % 31) as f32 - 15.0) * 0.013 + (t as f32) * 0.001;
                }
            }
        }
        let key = "model.language_model.layers.0.mlp.gate_proj";
        let _slots = tr.linear_batch_multi_gpu(&[key], t8)?;
        let li0 = tr.find_linear(key)?;
        let n0 = tr.linears[li0].1.n;
        let got = tr.read_yb_head(0, 8 * 4096);
        let xrow: Vec<f32> =
            unsafe { std::slice::from_raw_parts(st8 as *const f32, 5120).to_vec() };
        let want = tr.linear(key, &xrow)?;
        let mut md = 0f32;
        let mut nan_at: Vec<usize> = Vec::new();
        let mut nan_cnt = 0usize;
        for i in 0..n0.min(4096) {
            let g = got[i];
            if !g.is_finite() {
                nan_cnt += 1;
                if nan_at.len() < 6 {
                    nan_at.push(i);
                }
            }
            let d = (g - want[i]).abs();
            if d.is_finite() {
                md = md.max(d);
            }
        }
        let mut row_nan = vec![0usize; 8];
        for t in 0..8usize {
            let seg = &got[t * 4096..(t + 1) * 4096];
            row_nan[t] = seg.iter().filter(|v| !v.is_finite()).count();
        }
        eprintln!(
            "  [gemmdbg] gate T=8 row0 maxdiff={md:.3e} nan={nan_cnt} row_nan={row_nan:?} n0={n0}"
        );
    }
    for (name, key) in shapes {
        let li = tr.find_linear(key)?;
        let (k, n) = (tr.linears[li].1.k, tr.linears[li].1.n);
        // 5회 중앙값
        let mut ts: Vec<f64> = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            tr.linear_batch_multi_gpu(&[key], t_rows)?;
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = ts[2];
        let tf = 2.0 * t_rows as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
        report.push(format!(
            "{name:10} K={k:6} N={n:6}  {ms:7.2}ms  {tf:5.2} TF"
        ));
    }
    Ok(report.join("\n"))
}

// ── norm_resid 독립 프로브(plans/121 프레임) ──
pub fn nr_check() -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
    let mut ctx = VkCtx::new()?;
    let t_rows = 16usize;
    let mut seed: u32 = 0x5EED_1234;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    let xv: Vec<f32> = (0..t_rows * 5120)
        .map(|_| (rnd() * 2.0 - 1.0) * 0.5)
        .collect();
    let abv: Vec<f32> = (0..t_rows * 5120)
        .map(|_| (rnd() * 2.0 - 1.0) * 0.3)
        .collect();
    let wv: Vec<f32> = (0..5120).map(|_| 0.8 + rnd() * 0.4).collect();
    let bx = ctx.alloc_host_cached(t_rows * 5120 * 4)?;
    let bab = ctx.alloc_host_cached(t_rows * 5120 * 4)?;
    let bw = ctx.alloc_host_cached(5120 * 4)?;
    let bxn = ctx.alloc_host_cached(t_rows * 5120 * 4)?;
    unsafe {
        std::ptr::copy_nonoverlapping(xv.as_ptr(), bx.ptr as *mut f32, xv.len());
        std::ptr::copy_nonoverlapping(abv.as_ptr(), bab.ptr as *mut f32, abv.len());
        std::ptr::copy_nonoverlapping(wv.as_ptr(), bw.ptr as *mut f32, 5120);
        std::ptr::write_bytes(bxn.ptr, 0, t_rows * 5120 * 4);
    }
    ctx.flush_buf(&bx);
    ctx.flush_buf(&bab);
    ctx.flush_buf(&bw);
    let p = ctx.pipeline_pipes(include_bytes!("../spv/e3_norm_resid.spv"), 4, 8)?;
    ctx.begin_batch()?;
    let ds = ctx.fresh_ds_for(&p, 4)?;
    ctx.bind_bufs(ds, &[bx.buf, bw.buf, bab.buf, bxn.buf]);
    let push: Vec<u8> = [t_rows as u32, 0u32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    crate::rawvk::context::site::set_tag("e3_nr_probe");
    ctx.run_rw(
        p.pl,
        ds,
        p.pipe,
        &push,
        t_rows as u32,
        1,
        1,
        &[bw.buf, bab.buf],
        &[bxn.buf, bx.buf],
    )?;
    ctx.end_batch_wait()?;
    ctx.wait_pending()?;
    ctx.invalidate_buf(&bxn);
    ctx.invalidate_buf(&bx);
    let xn: Vec<f32> =
        unsafe { std::slice::from_raw_parts(bxn.ptr as *const f32, t_rows * 5120).to_vec() };
    let xo: Vec<f32> =
        unsafe { std::slice::from_raw_parts(bx.ptr as *const f32, t_rows * 5120).to_vec() };
    // CPU 기준
    let mut md_xn = 0f32;
    let mut md_xo = 0f32;
    for t in 0..t_rows {
        let base = t * 5120;
        let ss: f32 = (0..5120)
            .map(|i| {
                let v = xv[base + i] + abv[base + i];
                v * v
            })
            .sum();
        let inv = 1.0 / ((ss / 5120.0 + 1e-6).sqrt());
        for i in 0..5120 {
            let v = xv[base + i] + abv[base + i];
            md_xn = md_xn.max((xn[base + i] - v * inv * wv[i]).abs());
            md_xo = md_xo.max((xo[base + i] - v).abs());
        }
    }
    Ok(format!(
        "nr-check: xn maxdiff={md_xn:.3e} xo maxdiff={md_xo:.3e}"
    ))
}
// 마커: sqrt 프레임 판정용

pub fn ffn_check(dir: &str) -> Result<String, String> {
    let mut tr = TrellisResident::load(dir)?;
    let t_rows = 512usize;
    let stage = tr.stage_f32()?;
    unsafe {
        std::ptr::write_bytes(stage, 0, t_rows * 5120 * 4);
        for t in 0..t_rows {
            let p = stage.add(t * 5120);
            for i in 0..5120usize {
                *p.add(i) = ((i % 23) as f32 - 11.0) * 0.017 + (t as f32) * 0.0007;
            }
        }
    }
    let lp = "model.language_model.layers.0.mlp";
    let x0: Vec<f32> = unsafe { std::slice::from_raw_parts(stage as *const f32, 5120) }.to_vec();
    let g0 = tr.linear(&format!("{lp}.gate_proj"), &x0)?;
    let u0 = tr.linear(&format!("{lp}.up_proj"), &x0)?;
    let inter: Vec<f32> = g0
        .iter()
        .zip(u0.iter())
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let d0 = tr.linear(&format!("{lp}.down_proj"), &inter)?;
    tr.ffn_trio_batch(
        &format!("{lp}.gate_proj"),
        &format!("{lp}.up_proj"),
        &format!("{lp}.down_proj"),
        t_rows,
    )?;
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..5 {
        unsafe {
            let p = stage;
            for i in 0..5120usize {
                *p.add(i) = x0[i];
            }
        }
        let t0 = std::time::Instant::now();
        tr.ffn_trio_batch(
            &format!("{lp}.gate_proj"),
            &format!("{lp}.up_proj"),
            &format!("{lp}.down_proj"),
            t_rows,
        )?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let got = tr.read_yb_rows(2, 1, d0.len())?;
    let mut md = 0f32;
    let mut nan = 0usize;
    for i in 0..d0.len() {
        let d = (got[i] - d0[i]).abs();
        if !got[i].is_finite() {
            nan += 1;
        }
        if d.is_finite() {
            md = md.max(d);
        }
    }
    Ok(format!(
        "ffn-check T={t_rows}: row0 maxdiff={md:.3e} nan={nan} · trio median {:.1}ms",
        ts[2]
    ))
}
// 마커 ffn1

// ── GDN 비선형 체인 격리 프로브(plans/121 워크플로 — 마지막 간접 군) ──
// conv→l2perm→scan→gate 전체를 합성 입력으로 검증: CPU 미러(kernel 수학
// 직접 이식)와 행별 비교. 속도(4커널 dispatch 벽)도 보고.
pub fn chain_check(dir: &str) -> Result<String, String> {
    let t_rows = 32usize;
    let mut tr = TrellisResident::load(dir)?;
    let (cw, ab, alog, dtb, nw) = tr.gdn_chain_consts()?;
    let mut seed: u32 = 0x51DE_2718;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    let xn: Vec<f32> = (0..t_rows * 5120)
        .map(|_| (rnd() * 2.0 - 1.0) * 0.3)
        .collect();
    let qkv: Vec<f32> = (0..t_rows * 10240)
        .map(|_| (rnd() * 2.0 - 1.0) * 0.5)
        .collect();
    let z: Vec<f32> = (0..t_rows * 6144)
        .map(|_| (rnd() * 2.0 - 1.0) * 0.4)
        .collect();
    let got = tr.gdn_chain_run(t_rows, &xn, &qkv, &z)?;
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let _ = tr.gdn_chain_run(t_rows, &xn, &qkv, &z)?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // ── CPU 미러 ──
    let n_k = 16usize;
    let n_v = 48usize;
    let d_state = 128usize;
    let k_len = 2048usize;
    let d_inner = 6144usize;
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    // ① conv(FIR4, ring=0)
    let mut q_all = vec![0f32; t_rows * k_len];
    let mut k_all = vec![0f32; t_rows * k_len];
    let mut v_all = vec![0f32; t_rows * d_inner];
    for c in 0..10240usize {
        let (w0, w1, w2, w3) = (cw[c * 4], cw[c * 4 + 1], cw[c * 4 + 2], cw[c * 4 + 3]);
        let (mut h0, mut h1, mut h2) = (0f32, 0f32, 0f32);
        for t in 0..t_rows {
            let x = qkv[t * 10240 + c];
            let o = silu(w3 * x + w0 * h0 + w1 * h1 + w2 * h2);
            if c < k_len {
                q_all[t * k_len + c] = o;
            } else if c < 2 * k_len {
                k_all[t * k_len + (c - k_len)] = o;
            } else {
                v_all[t * d_inner + (c - 2 * k_len)] = o;
            }
            h0 = h1;
            h1 = h2;
            h2 = x;
        }
    }
    // ② l2perm: a/b 도트(xn·ab) + q/k L2 + v/beta|g lc 순열
    let mut q_l2 = vec![0f32; t_rows * k_len];
    let mut k_l2 = vec![0f32; t_rows * k_len];
    let mut v_lc = vec![0f32; t_rows * d_inner];
    let mut bg = vec![0f32; t_rows * 96];
    for t in 0..t_rows {
        // q/k L2(HF 헤드별)
        for kh in 0..n_k {
            let b0 = t * k_len + kh * d_state;
            let qn: f32 = (0..d_state).map(|i| q_all[b0 + i] * q_all[b0 + i]).sum();
            let kn: f32 = (0..d_state).map(|i| k_all[b0 + i] * k_all[b0 + i]).sum();
            let qi = 1.0 / (qn + 1e-6).sqrt();
            let ki = 1.0 / (kn + 1e-6).sqrt();
            for i in 0..d_state {
                q_l2[b0 + i] = q_all[b0 + i] * qi;
                k_l2[b0 + i] = k_all[b0 + i] * ki;
            }
        }
        // v-head별: a/b 도트 + 순열
        for h in 0..n_v {
            let xrow = &xn[t * 5120..(t + 1) * 5120];
            let a_row = &ab[h * 5120..(h + 1) * 5120];
            let b_row = &ab[(48 + h) * 5120..(48 + h + 1) * 5120];
            let a_v: f32 = xrow.iter().zip(a_row).map(|(x, w)| x * w).sum();
            let b_v: f32 = xrow.iter().zip(b_row).map(|(x, w)| x * w).sum();
            let g = softplus(a_v + dtb[h]) * (-alog[h].exp());
            let beta = 1.0 / (1.0 + (-b_v).exp());
            let p_inv = (h % 3) * 16 + h / 3;
            bg[t * 96 + p_inv] = beta;
            bg[t * 96 + 48 + p_inv] = g;
            // v lc 순열
            let src = t * d_inner + h * d_state;
            let dst = t * d_inner + p_inv * d_state;
            v_lc[dst..dst + d_state].copy_from_slice(&v_all[src..src + d_state]);
        }
    }
    // 중간 대조: l2perm gbg / conv gqr — 단계 격리(plans/121).
    let (gpu_bg, gpu_gq, gpu_gv, gpu_go) = tr.gdn_chain_mids(t_rows)?;
    let mut bg_md = 0f32;
    for i in 0..gpu_bg.len() {
        bg_md = bg_md.max((gpu_bg[i] - bg[i]).abs());
    }
    let mut q_md = 0f32;
    for i in 0..gpu_gq.len().min(q_l2.len()) {
        q_md = q_md.max((gpu_gq[i] - q_l2[i]).abs());
    }
    eprintln!("  [chainmid] conv q maxdiff={q_md:.3e} · l2perm bg maxdiff={bg_md:.3e}");

    // ③ scan(레지스터 리페런스 재사용 — scan_ref는 bg [T][96] 포맷)
    let (_, o_ref) = scan_ref(&q_l2, &k_l2, &v_lc, &bg, t_rows, false, &[0f32; 48 * 16384]);
    let (_, o_ref16) = scan_ref(&q_l2, &k_l2, &v_lc, &bg, t_rows, true, &[0f32; 48 * 16384]);
    let mut go16_md = 0f32;
    for i in 0..gpu_go.len().min(o_ref16.len()) {
        go16_md = go16_md.max((gpu_go[i] - o_ref16[i]).abs());
    }
    eprintln!("  [chainmid] scan go(f16mirror) maxdiff={go16_md:.3e}");
    let mut gq_md = 0f32;
    for i in 0..gpu_gq.len().min(q_l2.len()) {
        gq_md = gq_md.max((gpu_gq[i] - q_l2[i]).abs());
    }
    let mut gv_md = 0f32;
    for i in 0..gpu_gv.len().min(v_lc.len()) {
        gv_md = gv_md.max((gpu_gv[i] - v_lc[i]).abs());
    }
    eprintln!("  [chainmid] gq maxdiff={gq_md:.3e} · gv maxdiff={gv_md:.3e}");
    let mut go_md = 0f32;
    for i in 0..gpu_go.len().min(o_ref.len()) {
        go_md = go_md.max((gpu_go[i] - o_ref[i]).abs());
    }
    eprintln!("  [chainmid] scan go maxdiff={go_md:.3e}");
    // ④ gate: rms(o_lc)·nw·silu(z) → HF 역순열
    let mut want = vec![0f32; t_rows * d_inner];
    let eps = 1e-6f32;
    for t in 0..t_rows {
        for h in 0..n_v {
            let p_inv = (h % 3) * 16 + h / 3;
            let src = t * d_inner + p_inv * d_state;
            let dst = t * d_inner + h * d_state;
            let ss: f32 = (0..d_state).map(|i| o_ref[src + i] * o_ref[src + i]).sum();
            let inv = 1.0 / (ss / d_state as f32 + eps).sqrt();
            for i in 0..d_state {
                let zv = z[dst + i];
                want[dst + i] = o_ref[src + i] * inv * nw[i] * silu(zv);
            }
        }
    }
    let mut md = 0f32;
    let mut nan = 0usize;
    let mut rel_bad = 0usize;
    for i in 0..got.len() {
        if !got[i].is_finite() {
            nan += 1;
            continue;
        }
        let d = (got[i] - want[i]).abs();
        md = md.max(d);
        if d > 1e-3 && d / want[i].abs().max(1e-3) > 0.05 {
            rel_bad += 1;
        }
    }
    Ok(format!(
        "chain-check T={t_rows}: maxdiff={md:.3e} nan={nan} rel>5%={rel_bad}/{} · 4커널 {:.1}ms(중앙)",
        got.len(),
        ts[1]
    ))
}
// 마커 chain1
// 마커 f16ab
// 마커 gq2
