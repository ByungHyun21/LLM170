//! 어텐션/GEMV 계열 체커 — gemv·ft32·gemv8·gdn_chunk(plans/107 W4: checks.rs에서 분할).

use super::harness::{CheckHarness, load_ref};
use crate::rawvk::vkacc::{VkAcc, push_u32s, xq_words};
use ash::vk;
use llm170_core::matmul::MatmulHost as _;

/// vk-gemv-check — VkAcc matmul vs CPU W4A8 미러 단일 텐서 검증 + 타이밍.
pub fn gemv_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let (h, model) = CheckHarness::with_ref(path)?;
    let w = model.w(tname)?;
    let wref = &w;
    let n_in = w.n_in as usize;
    let acc = &h.acc;
    // ── quant 비트 검증: GPU xq vs CPU quantize_row_q8_ref ──
    {
        let mut seed2 = 0x1234abcdu64;
        let mut lcg2 = || {
            seed2 = seed2
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed2 >> 33) as f32 / 2147483648.0 - 0.5
        };
        let xrow: Vec<f32> = (0..n_in).map(|_| lcg2()).collect();
        let mut ctxg = acc.ctx.lock();
        let xq_w = xq_words(n_in);
        let xqb = ctxg.alloc_host(xq_w * 4)?;
        acc.quant_upload(&mut ctxg, std::slice::from_ref(&xrow), n_in, xqb.buf)?;
        let gpu: &[u32] = unsafe { std::slice::from_raw_parts(xqb.ptr as *const u32, xq_w) };
        let yref = llm170_core::quant::quantize_row_q8_ref(&xrow);
        // CPU 재구성: qs 워드 + d 비트 + s0/s1
        let mut qdiff = 0usize;
        let mut ddiff = 0usize;
        let mut sdiff = 0usize;
        let nwords = n_in / 4;
        let nblk = n_in / 32;
        for b in 0..nblk {
            let d_cpu = yref[b].d.to_bits();
            let d_gpu = gpu[nwords + b];
            if d_cpu != d_gpu {
                ddiff += 1;
                if ddiff <= 3 {
                    let mut amax = 0.0f32;
                    for &v in &xrow[b * 32..b * 32 + 32] {
                        amax = amax.max(v.abs());
                    }
                    let (cpu_d, gpu_d, rust_d, f64d) = (
                        d_cpu,
                        d_gpu,
                        (amax / 127.0f32).to_bits(),
                        (amax as f64 / 127.0).to_bits() as u32,
                    );
                    eprintln!(
                        "dblk{b}: amax={amax:e} cpu_d={cpu_d:08x} gpu_d={gpu_d:08x} rust_d={rust_d:08x} f64lo={f64d:08x}"
                    );
                }
            }
            let mut s0 = 0u32;
            let mut s1 = 0u32;
            for wi in 0..8 {
                let mut word = 0u32;
                for k in 0..4 {
                    let qv = yref[b].qs[wi * 4 + k] as i8 as i32 as u32;
                    word |= (qv & 0xFF) << (8 * k);
                }
                if word != gpu[b * 8 + wi] {
                    qdiff += 1;
                }
                // sd(서브바이트 차분 카운트)는 진단 전용으로 제거됨(2026-09-17).
                let bytes: i32 = (0..4)
                    .map(|k| ((gpu[b * 8 + wi] >> (8 * k)) & 0xFF) as i32)
                    .fold(0i32, |a, v| a + ((v << 24) >> 24));
                if wi < 4 {
                    s0 = s0.wrapping_add(bytes as u32);
                } else {
                    s1 = s1.wrapping_add(bytes as u32);
                }
            }
            let qsb = nwords + nblk;
            if s0 != gpu[qsb + b * 2] || s1 != gpu[qsb + b * 2 + 1] {
                sdiff += 1;
            }
        }
        eprintln!("quant-bits: qs워드 {qdiff}/{nwords} d {ddiff}/{nblk} s {sdiff}/{nblk} 상이");
    }
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let mut outs = vec![vec![0.0f32; w.n_out as usize]; t];
    acc.matmul_batch(&xs, wref, &mut outs)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        acc.matmul_batch(&xs, wref, &mut outs)?;
    }
    let dt = t0.elapsed().as_secs_f64() / 10.0;
    eprintln!(
        "vk-gemv-time: {} {:.2}ms → {:.1}GB/s ({}B 가중)",
        tname,
        dt * 1e3,
        wref.data.len() as f64 / dt / 1e9,
        wref.data.len()
    );
    let mut ref_outs = vec![vec![0.0f32; w.n_out as usize]; t];
    llm170_core::matmul::matmul_batch(&xs, wref, &mut ref_outs);
    let mut mx = 0f64;
    let mut rel = 0f64;
    let mut ndiff = 0usize;
    let mut ulp_hist = std::collections::HashMap::<i64, usize>::new();
    for (a, b) in outs.iter().zip(ref_outs.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            if x.to_bits() != y.to_bits() {
                ndiff += 1;
                let ulp = (x.to_bits() as i64 - y.to_bits() as i64).abs();
                *ulp_hist.entry(ulp).or_insert(0) += 1;
            }
            let d = (x - y).abs() as f64;
            if d > mx {
                mx = d;
            }
            let r = d / y.abs().max(1.0) as f64;
            if r > rel {
                rel = r;
            }
        }
    }
    let hist: Vec<String> = {
        let mut v: Vec<(i64, usize)> = ulp_hist.into_iter().collect();
        v.sort_by_key(|b| std::cmp::Reverse(b.1));
        v.iter()
            .take(4)
            .map(|(u, c)| format!("{c}x{u}ulp"))
            .collect()
    };
    let ia = outs[0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    let ib = ref_outs[0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    Ok(format!(
        "vk-gemv {tname} t={t}: max|D|={mx:.3e} maxrel={rel:.2e} argmax {ia:?}=={ib:?} {} | bits {ndiff}/{} differ, top {hist:?}",
        if ia == ib { "★" } else { "MISMATCH" },
        outs.len() * w.n_out as usize
    ))
}

/// vk-ft32-check (plans/89 P1.2) — fn_tile_f32(f32/BF16 밀집 프리필 타일)의
/// 실 텐서 CPU 대조. 라우터(ffn_gate_inp, f32)형상으로 게이트 발산 원인 특정.
pub fn ft32_check(path: &str) -> Result<String, String> {
    #[allow(unused_imports)]
    use llm170_core::matmul::{FrameHost as _FH, FrameState as _FS};
    let model = load_ref(path)?;
    let m4 = model.q4()?;
    let w = m4
        .w4("blk.0.ffn_gate_inp.weight")
        .map_err(|e| e.to_string())?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let h = CheckHarness::new()?;
    let acc = &h.acc;
    let t = 64usize;
    let mut lcg = 123456789u64;
    let mut lcgf = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t)
        .map(|_| (0..n_in).map(|_| lcgf()).collect())
        .collect();
    let mut flat = Vec::with_capacity(t * n_in);
    for r in &xs {
        flat.extend_from_slice(r);
    }
    let xh = acc.frame_alloc(t * n_in)?;
    let oh = acc.frame_alloc(t * n_out)?;
    acc.frame_write(xh, &flat)?;
    acc.frame_begin(t);
    acc.frame_mm_group(xh, std::slice::from_ref(&w), std::slice::from_ref(&oh), t)?;

    let mut got = vec![0f32; t * n_out];
    acc.frame_read(oh, &mut got)?;
    let _ = acc.frame_free(xh);
    let _ = acc.frame_free(oh);
    // CPU 참조 — w 는 f32 그대로.
    let wf = w.data.as_ptr() as *const f32;
    let mut mx = 0f64;
    let mut bad = 0usize;
    for r in 0..t {
        for j in 0..n_out {
            let mut s = 0f64;
            for k in 0..n_in {
                s += unsafe { *wf.add(j * n_in + k) } as f64 * xs[r][k] as f64;
            }
            let d = (got[r * n_out + j] as f64 - s).abs();
            if d > 1e-3 {
                bad += 1;
            }
            mx = mx.max(d);
        }
    }
    // 혼합 그룹(q8 down + f32 inject, n_out=4 극단 shape) — 실엔진 hc 믹스.
    let wd = m4
        .w4("blk.0.hc_attn_down.weight")
        .map_err(|e| e.to_string())?;
    let n2 = wd.n_in as usize;
    let wi = m4
        .w4("blk.0.hc_attn_inject.weight")
        .map_err(|e| e.to_string())?;
    let xs2: Vec<Vec<f32>> = (0..t).map(|_| (0..n2).map(|_| lcgf()).collect()).collect();
    let mut flat2 = Vec::with_capacity(t * n2);
    for r in &xs2 {
        flat2.extend_from_slice(r);
    }
    let xh2 = acc.frame_alloc(t * n2)?;
    let od = acc.frame_alloc(t * wd.n_out as usize)?;
    let oi = acc.frame_alloc(t * wi.n_out as usize)?;
    acc.frame_write(xh2, &flat2)?;
    acc.frame_begin(t);
    acc.frame_mm_group(xh2, &[wd, wi], &[od, oi], t)?;
    let mut gi = vec![0f32; t * wi.n_out as usize];
    acc.frame_read(oi, &mut gi)?;
    // plans/95: q8 down(q8mmq 경로) 판독 — 해제 전에 읽는다.
    let mut gd = vec![0f32; t * wd.n_out as usize];
    acc.frame_read(od, &mut gd)?;
    let _ = (acc.frame_free(xh2), acc.frame_free(od), acc.frame_free(oi));
    let wi_f = wi.data.as_ptr() as *const f32;
    let nin_i = wi.n_in as usize;
    let mut mx2 = 0f64;
    let mut bad2 = 0usize;
    for r in 0..t {
        for j in 0..wi.n_out as usize {
            let mut s = 0f64;
            for k in 0..nin_i {
                s += unsafe { *wi_f.add(j * nin_i + k) } as f64 * xs2[r][k] as f64;
            }
            let d = (gi[r * wi.n_out as usize + j] as f64 - s).abs();
            if d > 1e-3 {
                bad2 += 1;
            }
            mx2 = mx2.max(d);
        }
    }
    // q8_0 down 경로 검증.
    let mut mx3 = 0f64;
    let mut bad3 = 0usize;
    if llm170_diag::dump::opts().key("q8_dbg") {
        for j in 0..4usize {
            let mut rr = vec![0f32; n2];
            llm170_core::quant::dequant_row(wd.ty, wd.data, j as u64, n2 as u64, &mut rr);
            let dot: f32 = rr.iter().zip(xs2[0].iter()).map(|(a, b)| a * b).sum();
            eprintln!("[q8dbg] r=0 j={j} got={:.5} ref={:.5}", gd[j], dot);
        }
    }
    for r in 0..t {
        for j in 0..wd.n_out as usize {
            let mut rr = vec![0f32; n2];
            llm170_core::quant::dequant_row(wd.ty, wd.data, (j) as u64, n2 as u64, &mut rr);
            let dot: f32 = rr.iter().zip(xs2[r].iter()).map(|(a, b)| a * b).sum();
            let d = (gd[r * wd.n_out as usize + j] as f64 - dot as f64).abs();
            if d > 2e-2 {
                bad3 += 1;
            }
            mx3 = mx3.max(d);
        }
    }
    // 요약 행에 q8 결과 추가.
    Ok(format!(
        "ft32-check: router max|D|={mx:.3e} bad={bad} {} | inject(f32 {}x{}) max|D|={mx2:.3e} bad={bad2} {} | q8down({}x{}) max|D|={mx3:.3e} bad={bad3} {}",
        if bad == 0 { "★" } else { "✗" },
        wi.n_out,
        wi.n_in,
        if bad2 == 0 { "★" } else { "✗" },
        wd.n_out,
        wd.n_in,
        if bad3 == 0 { "★" } else { "✗" }
    ))
}

/// vk-gemv8-check — gemv8 패밀리(llama mul_mat_vec 포트, f32 직결) 검증+타이밍.
pub fn gemv8_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let is_xs = w.ty == llm170_gguf::GgmlType::Iq4Xs;
    let is_nl = w.ty == llm170_gguf::GgmlType::Iq4Nl;
    let is_q5 = w.ty == llm170_gguf::GgmlType::Q5K;
    let is_q6 = w.ty == llm170_gguf::GgmlType::Q6K;
    let is_q4 = w.ty == llm170_gguf::GgmlType::Q4K;
    let is_q3 = w.ty == llm170_gguf::GgmlType::Q3K;
    let is_q8 = w.ty == llm170_gguf::GgmlType::Q8_0;
    if !is_xs && !is_nl && !is_q5 && !is_q6 && !is_q4 && !is_q3 && !is_q8 {
        return Err("gemv8 검증은 q3_K/q4_K/q5_K/q6_K/iq4_xs/iq4_nl만".into());
    }
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut seed = 0x1234u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let xa = ctx.alloc_host(t * n_in * 4)?;
    for (j, x) in xs.iter().enumerate() {
        unsafe {
            // 행 스트라이드는 바이트 — n_in f32 = n_in*4바이트 (A2: 이 오타가
            // t≥2 하니스 오염의 전부였음 — 행1이 행0의 1/4 지점을 덮어씀)
            std::ptr::copy_nonoverlapping(x.as_ptr(), xa.ptr.add(j * n_in * 4) as *mut f32, n_in);
        }
    }
    let ob = ctx.alloc_host(t * n_out * 4)?; // 매핑 유지 — 판독용
    // 가중 업로드 — gemv3와 동일한 균일 청크
    let ch = ctx.max_ssbo;
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    let total = w.data.len();
    // 청크 크기 2의 거듭제곱 (WG 시프트 산술) — 마지막 청크는 실제 크기만 할당:
    // o = idx & mask 는 항상 청크 내 실데이터 오프셋만 생성하므로 패딩 불필요.
    let ch = total
        .next_power_of_two()
        .min(1usize << (63 - ch.leading_zeros()));
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let dummy = ctx.alloc_host(16)?;
    {
        let z = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(z.as_ptr(), dummy.ptr, 16) };
    }
    while wbufs.len() < 8 {
        wbufs.push(dummy.buf);
    }
    let chunk_words = (ch / 4) as u32;
    let spv_path = match w.ty {
        llm170_gguf::GgmlType::Q3K
            if std::env::var("LLM170_Q3B")
                .map(|v| v != "0")
                .unwrap_or(true) =>
        {
            "crates/backend-gpu/src/rawvk/spv/gemv8_q3b.spv"
        }
        llm170_gguf::GgmlType::Q3K => "crates/backend-gpu/src/rawvk/spv/gemv8_q3.spv",
        llm170_gguf::GgmlType::Q4K
            if std::env::var("LLM170_Q4B")
                .map(|v| v != "0")
                .unwrap_or(true) =>
        {
            "crates/backend-gpu/src/rawvk/spv/gemv8_q4b.spv"
        }
        llm170_gguf::GgmlType::Q4K => "crates/backend-gpu/src/rawvk/spv/gemv8_q4.spv",
        llm170_gguf::GgmlType::Q5K
            if std::env::var("LLM170_Q5B")
                .map(|v| v != "0")
                .unwrap_or(true) =>
        {
            "crates/backend-gpu/src/rawvk/spv/gemv8_q5b.spv"
        }
        llm170_gguf::GgmlType::Iq4Nl => "crates/backend-gpu/src/rawvk/spv/gemv8_nlb.spv",
        llm170_gguf::GgmlType::Q5K => "crates/backend-gpu/src/rawvk/spv/gemv8_q5.spv",
        llm170_gguf::GgmlType::Q6K
            if std::env::var("LLM170_Q6B")
                .map(|v| v != "0")
                .unwrap_or(true) =>
        {
            "crates/backend-gpu/src/rawvk/spv/gemv8_q6b.spv"
        }
        llm170_gguf::GgmlType::Q6K => "crates/backend-gpu/src/rawvk/spv/gemv8_q6.spv",
        llm170_gguf::GgmlType::Iq4Xs
            if std::env::var("LLM170_XSB")
                .map(|v| v != "0")
                .unwrap_or(true) =>
        {
            "crates/backend-gpu/src/rawvk/spv/gemv8_xsb.spv"
        }
        llm170_gguf::GgmlType::Iq4Xs => "crates/backend-gpu/src/rawvk/spv/gemv8_xs.spv",
        llm170_gguf::GgmlType::Q8_0
            if std::env::var("LLM170_Q8B")
                .map(|v| v != "0")
                .unwrap_or(true) =>
        {
            "crates/backend-gpu/src/rawvk/spv/gemv8_q8b.spv"
        }
        llm170_gguf::GgmlType::Q8_0 => "crates/backend-gpu/src/rawvk/spv/gemv8_q8.spv",
        _ => return Err("gemv8: 미지원 타입".into()),
    };
    let spv = std::fs::read(spv_path).map_err(|e| e.to_string())?;
    let (kb, _gb, _db) = acc.ensure_shared(&mut ctx)?;
    let n_kb_h = if is_xs { 12 } else { 10 };
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, n_kb_h, 24)?;
    let _ = (dsl, pool);
    let mut binds: Vec<vk::Buffer> = wbufs.clone();
    binds.push(xa.buf);
    binds.push(ob.buf);
    if is_xs {
        binds.push(kb);
    }
    ctx.bind_bufs(ds, &binds);
    let q5b = w.ty == llm170_gguf::GgmlType::Q5K
        && std::env::var("LLM170_Q5B")
            .map(|v| v != "0")
            .unwrap_or(true);
    let q4b = w.ty == llm170_gguf::GgmlType::Q4K
        && std::env::var("LLM170_Q4B")
            .map(|v| v != "0")
            .unwrap_or(true);
    let q6b = w.ty == llm170_gguf::GgmlType::Q6K
        && std::env::var("LLM170_Q6B")
            .map(|v| v != "0")
            .unwrap_or(true);
    let q8b = w.ty == llm170_gguf::GgmlType::Q8_0
        && std::env::var("LLM170_Q8B")
            .map(|v| v != "0")
            .unwrap_or(true);
    let q3b = w.ty == llm170_gguf::GgmlType::Q3K
        && std::env::var("LLM170_Q3B")
            .map(|v| v != "0")
            .unwrap_or(true);
    let xsb = w.ty == llm170_gguf::GgmlType::Iq4Xs
        && std::env::var("LLM170_XSB")
            .map(|v| v != "0")
            .unwrap_or(true);
    let rpf: u32 = if q5b || q4b || q6b || q8b || xsb || q3b {
        2
    } else if n_out < 4096 {
        1
    } else {
        2
    }; // llama NUM_ROWS=2
    let cw_log2 = 31u32 - chunk_words.leading_zeros();
    let cw_mask = (1u32 << cw_log2) - 1u32;
    // cw 단위: q5/q6(u16 typed 뷰)만 u16 단위, 나머지 u32
    let (cwpl, cwpm) = if is_q5 || is_q6 {
        (
            31u32 - (chunk_words * 2).leading_zeros(),
            (chunk_words * 2) - 1,
        )
    } else {
        (cw_log2, cw_mask)
    };
    let push = push_u32s(&[n_in as u32, n_out as u32, t as u32, cwpl, cwpm, rpf]);
    ctx.run(
        pl,
        ds,
        pipe,
        &push,
        1,
        n_out.div_ceil(rpf as usize) as u32,
        t as u32,
    )?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; t * n_out];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
        v
    };
    // CPU 기준: 디양자화 내적
    let mut mx = 0f64;
    let mut ref_row = vec![0.0f32; n_in];
    for (j, x) in xs.iter().enumerate() {
        for r in 0..n_out.min(64) {
            llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            mx = mx.max((dot - outs[j * n_out + r]).abs() as f64);
        }
    }
    let solo_t0 = Instant::now();
    for _ in 0..10 {
        ctx.run(
            pl,
            ds,
            pipe,
            &push,
            1,
            n_out.div_ceil(rpf as usize) as u32,
            t as u32,
        )?;
    }
    let solo_dt = solo_t0.elapsed().as_secs_f64() / 10.0;
    // L2 플러시 타이밍 — 반복 사이 자기 자신을 12회 연속 돌린 뒤
    // '매 반복 직전 타 텐서 1회' 교차 판독으로 캐시 몰아내기 (L2FLUSH=1).
    let flushed_dt: f64 = if std::env::var_os("LLM170_L2FLUSH").is_some() {
        // MULTI로 등록한 첫 extra 텐서를 플러시용으로 재사용: 그 weights로
        // 동일 커널 1회 (다른 ds/push 필요) — 여기선 간단히 xa를 8MB 재기록 후
        // 측정 대상 run 직전 xa 전체 재업로드 (호스트 memcpy가 L2 오염)
        let t4 = Instant::now();
        let xa2 = xs[0].clone();
        for _ in 0..10 {
            unsafe {
                std::ptr::copy_nonoverlapping(xa2.as_ptr(), xa.ptr as *mut f32, n_in);
            }
            ctx.run(
                pl,
                ds,
                pipe,
                &push,
                1,
                n_out.div_ceil(rpf as usize) as u32,
                t as u32,
            )?;
        }
        t4.elapsed().as_secs_f64() / 10.0
    } else {
        0.0
    };
    if flushed_dt > 0.0 {
        return Ok(format!(
            "gemv8-l2flush({tname}): {:.3}ms → {:.1}GB/s (웜 {})",
            flushed_dt * 1e3,
            w.data.len() as f64 / flushed_dt / 1e9,
            w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    if let Ok(list) = std::env::var("LLM170_MULTI") {
        // TLB/할당수 가설: 추가 텐서들을 같은 컨텍스트에 로드(상주)시킨 뒤
        // 이 텐서의 타이밍 재측정 — 속도 붕괴 시 가설 확인.
        for extra in list.split(',').filter(|x| !x.is_empty()) {
            if extra == tname {
                continue;
            }
            let w2 = match model.w(extra) {
                Some(w) => w,
                None => continue,
            };
            let mut off2 = 0usize;
            let tot2 = w2.data.len();
            while off2 < tot2 {
                let sz2 = ch.min(tot2 - off2);
                let mut b2 = ctx.alloc(sz2)?;
                unsafe { std::ptr::copy_nonoverlapping(w2.data.as_ptr().add(off2), b2.ptr, sz2) };
                ctx.unmap(&mut b2)?;
                off2 += sz2;
            }
        }
        let t2 = Instant::now();
        for _ in 0..10 {
            ctx.run(
                pl,
                ds,
                pipe,
                &push,
                1,
                n_out.div_ceil(rpf as usize) as u32,
                t as u32,
            )?;
        }
        let dt2 = t2.elapsed().as_secs_f64() / 10.0;
        let _ = &solo_dt;
        return Ok(format!(
            "gemv8-multi({tname}): {:.3}ms → {:.1}GB/s (단독 {:.1})",
            dt2 * 1e3,
            w.data.len() as f64 / dt2 / 1e9,
            w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    Ok(format!(
        "gemv8({tname}) t={t}: {:.3}ms → {:.1}GB/s · max|D|={mx:.4}",
        solo_dt * 1e3,
        w.data.len() as f64 / solo_dt / 1e9
    ))
}

/// vk-gdn-chunk-check (plans/100) — 청크 병렬 GDN vs 순차 스캔 대조.
/// 난수 q/k/v/bg로 단일 (pair, u블록) 수학 검증: 상대오차 <1e-3 판정.
pub fn gdn_chunk_check() -> Result<String, String> {
    eprintln!("[gdnc] enter");
    #[allow(unused_imports)]
    use llm170_core::matmul::{FrameHost as _FH, FrameState as _FS};
    let acc = VkAcc::new()?;
    eprintln!("[gdnc] acc ok");
    let d = 128usize;
    let hv = 4usize;
    let hk = 2usize;
    let t = 192usize; // 3청크(64×3) — 청크 경계 상태 전파 검증
    // 버퍼: s(전치), q, k, v, bg, o 두 세트(스캔/청크).
    let sh = acc.frame_alloc(hv * d * d)?;
    let qh = acc.frame_alloc(t * hk * d)?;
    let kh = acc.frame_alloc(t * hk * d)?;
    let vh = acc.frame_alloc(t * hv * d)?;
    let bh = acc.frame_alloc(t * hv * 2)?;
    let o1 = acc.frame_alloc(t * hv * d)?;
    let o2 = acc.frame_alloc(t * hv * d)?;
    let mut lcg = 20260927u64;
    let mut lcgf = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let mut s0 = vec![0f32; hv * d * d];
    for v in s0.iter_mut() {
        *v = lcgf() * 0.3;
    }
    let qv: Vec<f32> = (0..t * hk * d).map(|_| lcgf()).collect();
    let kv: Vec<f32> = (0..t * hk * d).map(|_| lcgf() * 0.1).collect(); // |K|↓ — 안정화
    let vv: Vec<f32> = (0..t * hv * d).map(|_| lcgf()).collect();
    let bv: Vec<f32> = (0..t * hv * 2)
        .map(|i| {
            if i % 2 == 0 {
                lcgf().abs() * 0.8 + 0.1
            } else {
                0.8 + lcgf().abs() * 0.19
            } // β<1, Ge∈(0.8,0.99)
        })
        .collect();
    acc.frame_write(sh, &s0)?;
    acc.frame_write(qh, &qv)?;
    acc.frame_write(kh, &kv)?;
    acc.frame_write(vh, &vv)?;
    acc.frame_write(bh, &bv)?;
    let fb = |h: u64| {
        acc.framebufs
            .lock()
            .get(&h)
            .map(|b| b.buf)
            .ok_or::<String>("핸들 없음".into())
    };
    eprintln!("[gdnc] buffers ok, scan...");
    #[allow(unused_mut)]
    let mut ctx = acc.ctx.lock(); // 버퍼 준비 후 락(재진입 교착 방지).
    // (a) 순차 스캔 참조.
    {
        let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/fn_gdn_ar_swap.spv")
            .map_err(|e| e.to_string())?;
        let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 6, 28)?;
        let _ = (dsl, pool);
        let (sb, qb, kb, vb, bb, ob) = (fb(sh)?, fb(qh)?, fb(kh)?, fb(vh)?, fb(bh)?, fb(o1)?);
        ctx.bind_bufs(ds, &[sb, qb, kb, vb, bb, ob]);
        let mut push = push_u32s(&[
            d as u32,
            (hk * d) as u32,
            (hv * d) as u32,
            hv as u32,
            hk as u32,
        ]);
        push.extend_from_slice(&1.0f32.to_le_bytes());
        push.extend_from_slice(&(t as u32).to_le_bytes());
        ctx.run(pl, ds, pipe, &push, d as u32, hv as u32, 1)?;
    }
    eprintln!("[gdnc] scan ok, chunk...");
    // (b) 청크 판 — 상태 리셋 후 3청크 순차 디스패치(외부 순차).
    acc.frame_write(sh, &s0)?;
    {
        let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/fn_gdn_chunk.spv")
            .map_err(|e| e.to_string())?;
        let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 6, 40)?;
        let _ = (dsl, pool);
        let (sb, qb, kb, vb, bb, ob) = (fb(sh)?, fb(qh)?, fb(kh)?, fb(vh)?, fb(bh)?, fb(o2)?);
        ctx.bind_bufs(ds, &[sb, qb, kb, vb, bb, ob]);
        let nchunks = t.div_ceil(64);
        eprintln!("[gdnc] chunk pipe ok, {}개", nchunks);
        for c in 0..nchunks {
            let csize = 64.min(t - c * 64);
            let mut push = push_u32s(&[
                d as u32,
                (hk * d) as u32,
                (hv * d) as u32,
                hv as u32,
                hk as u32,
            ]);
            push.extend_from_slice(&1.0f32.to_le_bytes());
            push.extend_from_slice(&(t as u32).to_le_bytes());
            push.extend_from_slice(&((c * 64) as u32).to_le_bytes());
            push.extend_from_slice(&(csize as u32).to_le_bytes());
            ctx.run(pl, ds, pipe, &push, 1, hv as u32, (d / 64) as u32)?;
        }
    }
    eprintln!("[gdnc] chunks ok, compare...");
    drop(ctx); // frame_read가 ctx 재잠금 — guard 해제(교착 2차 방지).
    // 대조.
    eprintln!("[gdnc] read1...");
    let mut got1 = vec![0f32; t * hv * d];
    acc.frame_read(o1, &mut got1)?;
    eprintln!("[gdnc] read2...");
    let mut got2 = vec![0f32; t * hv * d];
    acc.frame_read(o2, &mut got2)?;
    eprintln!("[gdnc] reads ok");
    let mut mx = 0f64;
    let mut bad = 0usize;
    for i in 0..got1.len() {
        let rel = ((got2[i] - got1[i]).abs() as f64) / got1[i].abs().max(1e-2) as f64;
        if rel > mx {
            mx = rel;
        }
        if rel > 1e-3 {
            bad += 1;
            if bad <= 3 {
                eprintln!(
                    "[gdnc] bad#{} idx={} tok={} scan={:.6e} chunk={:.6e}",
                    bad,
                    i,
                    i / (hv * d),
                    got1[i],
                    got2[i]
                );
            }
        }
    }
    // 상태도 대조.
    let mut s1 = vec![0f32; hv * d * d];
    let _s2 = vec![0f32; hv * d * d];
    acc.frame_read(sh, &mut s1)?;
    // 주의: sh는 청크판이 갱신했음 — 스캔 상태는 재실행 필요. 간이: o만.
    for h in [sh, qh, kh, vh, bh, o1, o2] {
        let _ = acc.frame_free(h);
    }
    Ok(format!(
        "gdn-chunk(t={t}, 3청크): max|relD|={mx:.3e} bad={bad}/{} {}",
        got1.len(),
        if bad == 0 { "★" } else { "✗" }
    ))
}
