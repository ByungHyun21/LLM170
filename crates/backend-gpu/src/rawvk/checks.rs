//! rawvk/checks — 진단 체커(plans/90 B1: gemv.rs에서 순수 이동).
//! `llm170 vk-*-check` 계열 CLI가 호출하는 CPU 대조 검증 — 프로덕션 경로와 무관.
//! 이동 원칙: 본문 불변(gemv.rs 시절 바이트 동일).

use super::vkacc::{push_u32s, Slot, VkAcc, xq_words};
use ash::vk;
use llm170_core::matmul::MatmulHost as _;

/// vk-gemv-check — VkAcc matmul vs CPU W4A8 미러 단일 텐서 검증 + 타이밍.
pub fn gemv_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    // plans/84 B: qwen4exp(Flash-Next, 멀티파트) 폴백 — arch 판별 후 단일 로드.
    // 진단 전용 값 semantic — 박싱 없이 값 소유(체커 1회 로드).
    #[allow(clippy::large_enum_variant)]
    enum AnyModel {
        Q35(llm170_core::qwen35::Model),
        Q4(llm170_core::qwen4exp::Model4),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    let model = if is_q4 {
        AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    } else {
        AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    };
    let w = match &model {
        AnyModel::Q35(m) => m.w(tname).ok_or("텐서 없음")?,
        AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string())?,
    };
    let wref = &w;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    // ── quant 비트 검증: GPU xq vs CPU quantize_row_q8_ref ──
    {
        let mut seed2 = 0x1234abcdu64;
        let mut lcg2 = || {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed2 >> 33) as f32 / 2147483648.0 - 0.5
        };
        let xrow: Vec<f32> = (0..n_in).map(|_| lcg2()).collect();
        let mut ctxg = acc.ctx.lock();
        let xq_w = xq_words(n_in);
        let xqb = ctxg.alloc_host(xq_w * 4)?;
        acc.quant_upload(&mut ctxg, std::slice::from_ref(&xrow), n_in, xqb.buf)?;
        let gpu: &[u32] =
            unsafe { std::slice::from_raw_parts(xqb.ptr as *const u32, xq_w) };
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
                    let (cpu_d, gpu_d, rust_d, f64d) = (d_cpu, d_gpu, (amax / 127.0f32).to_bits(), (amax as f64 / 127.0).to_bits() as u32);
                    eprintln!("dblk{b}: amax={amax:e} cpu_d={cpu_d:08x} gpu_d={gpu_d:08x} rust_d={rust_d:08x} f64lo={f64d:08x}");
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
        eprintln!(
            "quant-bits: qs워드 {qdiff}/{nwords} d {ddiff}/{nblk} s {sdiff}/{nblk} 상이"
        );
    }
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        v.iter().take(4).map(|(u, c)| format!("{c}x{u}ulp")).collect()
    };
    let ia = outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    let ib = ref_outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    Ok(format!(
        "vk-gemv {tname} t={t}: max|D|={mx:.3e} maxrel={rel:.2e} argmax {ia:?}=={ib:?} {} | bits {ndiff}/{} differ, top {hist:?}",
        if ia == ib { "★" } else { "MISMATCH" },
        outs.len() * w.n_out as usize
    ))
}

/// 부록87: llama matmul_q5_k_f16.spv 직접 로드 격리 측정 (t≥2 프리필 GEMM).
pub fn vk_mmq_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    if std::env::var_os("VK_DUMP_W").is_some() {
        let _ = std::fs::write("/tmp/q5k_w.bin", w.data);
        eprintln!("W 더프: {}B n_in={} n_out={}", w.data.len(), n_in, n_out);
    }
    let spv_name = std::env::var("VKMMQ_SPV").unwrap_or_else(|_| "matmul_q5_k_f16".into());
    let b_is_f32 = spv_name.ends_with("_f32") || spv_name.contains("_f32_");
    let spv = if spv_name.contains('/') {
        std::fs::read(&spv_name).map_err(|e| e.to_string())?
    } else {
        std::fs::read(format!("/home/yoon/local_llm/llama.cpp-master/build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/{}.spv", spv_name))
            .map_err(|e| e.to_string())?
    };
    let acc = VkAcc::new()?;
    let mut ctxg = acc.ctx.lock();
    // 버퍼: A=가중(호스트맵→h2d는 run 전 복사), B=f16 y, D=f32 out
    let ab_vram = std::env::var("VKMMQ_VRAM").map(|v| v=="1").unwrap_or(false);
    let ab = if ab_vram {
        let mut b = ctxg.alloc(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()); }
        ctxg.unmap(&mut b)?;
        b
    } else {
        let mut b = ctxg.alloc_host(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()); }
        ctxg.unmap(&mut b)?;
        b
    };
    let mut ybuf: Vec<u16> = Vec::with_capacity(n_in * t);
    let mut seed = 0x1234u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    // y [K][N] f16 (stride_b = K) — CPU 참조와 동일값 사용
    let mut yf: Vec<f32> = Vec::with_capacity(n_in * t);
    for _ in 0..n_in * t { let v = lcg(); yf.push(v); ybuf.push(hf(v)); }
    let b_bytes = if b_is_f32 { n_in * t * 4 } else { n_in * t * 2 };
    let mut bb = ctxg.alloc_host(b_bytes)?;
    if b_is_f32 {
        unsafe { std::ptr::copy_nonoverlapping(yf.as_ptr() as *const u8, bb.ptr, b_bytes); }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ybuf.as_ptr() as *const u8, bb.ptr, b_bytes); }
    }
    ctxg.unmap(&mut bb)?;
    eprintln!("bc: allocs ok");
    let db = ctxg.alloc_host(n_out * t * 4)?;
    // l 파이프라인 (non-cm, subgroup=64, gfx1151): ids 0..10 + ALIGNED=0
    let sp = std::env::var("VKMMQ_SPEC").unwrap_or_else(|_| "l".into());
    let is_cm1 = spv_name.contains("_cm1");
    let spec: Vec<u32> = match sp.as_str() {
        "m" => vec![128, 64, 64, 32, 64, 32, 2, 4, 2, 1, 64, 0],
        "s" => vec![64, 32, 32, 32, 32, 32, 2, 2, 2, 1, 64, 0],
        "c" => vec![128, 128, 32, 32, 64, 32, 2, 4, 4, 1, 64, 0],
        "m32" => vec![128, 64, 64, 32, 32, 32, 2, 4, 2, 1, 32, 0],
        "l32" => vec![128, 128, 128, 32, 64, 64, 2, 4, 4, 1, 32, 0],
        "cc" => vec![128, 64, 32, 32, 64, 32, 2, 4, 4, 1, 64, 0],
        "mini" => vec![32, 32, 16, 32, 32, 16, 1, 4, 4, 1, 32, 0],
        "ls" => vec![256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64, 1],  // AMD RADV l-warptile_mmq
        "ms" => vec![128, 64, 64, 32, 64, 32, 2, 16, 16, 16, 64, 1],    // m-warptile_mmq
        "ss" => vec![64, 32, 32, 32, 32, 32, 2, 16, 16, 16, 64, 1],     // s-warptile_mmq
        "def" => vec![64, 64, 64, 16, 32, 32, 2, 4, 2, 1, 32, 0],  // spv 기본값 (부록87 해독)
        "cm1" => vec![128, 128, 128, 16, 128, 64, 2, 16, 16, 16, 64, 0],
        _ => vec![128, 128, 128, 32, 128, 64, 2, 4, 4, 1, 64, 0],
    };
    eprintln!("bc: y/ab 채움");
    let (_dsl, pl, _dp, ds, pipe) = ctxg.pipeline_spec_fg(&spv, 3, 17 * 4, &spec, is_cm1)?;
    eprintln!("bc: 파이프라인 ok");
    let bufs = [ab.buf, bb.buf, db.buf];
    ctxg.bind_bufs(ds, &bufs);
    // push: M,N,K,stride_a=K,stride_b=K,stride_d=M,batch 0들 + k_split=1 등
    let mut pc: Vec<u32> = vec![
        n_out as u32, t as u32, n_in as u32,      // M, N, K
        n_in as u32, n_in as u32, n_out as u32,   // stride_a=K, stride_b=K, stride_d=M
        0, 0, 0,                                  // batch strides
        0, 1, n_in as u32,                        // base_wg_z, num_batches, k_split=K (split_k=1 규약)
        1, 1, 1, 1,                               // ne02, ne12, broadcast2, broadcast3
        t as u32,                                 // padded_n (f16 B — 비양자화 경로)
    ];
    let pcb: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
    // 그리드 분모 = 스펙의 BM/BN에 정합 (부록87 그리드-스펙 매칭)
    let (dx, dy) = match sp.as_str() {
        "ls" => (128u32, 128),
        "ms" => (64, 64),
        "ss" => (32, 32),
        "m" | "m32" => (64u32, 64),
        "s" => (32, 32),
        "c" | "cc" => (64, 32),
        "mini" => (32, 16),
        "def" => (64, 64),
        _ => (128, 128),
    };
    let gx = (n_out as u32).div_ceil(dx);
    let gy = (t as u32).div_ceil(dy);
    ctxg.begin_batch()?;
    ctxg.run(pl, ds, pipe, &pcb, gx, gy, 1)?;
    ctxg.end_batch_wait()?;
    // CPU 참조 대조 (처음 8값) + 타이밍
    let out: &[f32] = unsafe { std::slice::from_raw_parts(db.ptr as *const f32, n_out * t) };
    // CPU 참조: 몇 개 (m, n) 지점 대조 — y는 f16 반올림값 사용
    let yh: Vec<f32> = ybuf.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let rb = w.data.len() / n_out;
    let mut dq = vec![0f32; n_in];
    let mut ok_nm = 0usize; let mut ok_mn = 0usize; let mut tot = 0usize;
    for &m in &[0usize, 100, 3000, 6143] {
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        for n in 0..t {
            let yrow = &yh[n * n_in..(n + 1) * n_in];
            let mut acc = 0f32;
            for k in 0..n_in { acc += dq[k] * yrow[k]; }
            let g_nm = out[n * n_out + m];
            let g_mn = out[m * t + n];
            let r = |g: f32| (g - acc).abs() / acc.abs().max(1e-3);
            if r(g_nm) < 0.01 { ok_nm += 1; }
            if r(g_mn) < 0.01 { ok_mn += 1; }
            tot += 1;
        }
    }
    eprintln!("레이아웃 판별: [N][M]={}/{} · [M][N]={}/{}", ok_nm, tot, ok_mn, tot);
    {
        let m = 0usize;
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        let mut goods = vec![];
        for n in 0..t {
            let yrow = &yh[n * n_in..(n + 1) * n_in];
            let mut acc = 0f32;
            for k in 0..n_in { acc += dq[k] * yrow[k]; }
            if ((out[n * n_out + m] - acc).abs() / acc.abs().max(1e-3)) < 0.01 { goods.push(n); }
        }
        eprintln!("m=0 정답 n ({}개): {:?}", goods.len(), &goods[..goods.len().min(20)]);
    }
    let mut maxrel = 0f32;
    for &(m, n) in &[(0, 0), (1, 0), (63, 0), (64, 0), (100, 0), (127, 0), (128, 0), (0, 1), (0, 63), (0, 64), (0, 100), (0, 127), (0, 128), (100, 7), (200, 100)] {
        if m >= n_out || n >= t { continue; }
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        let yrow = &yh[n * n_in..(n + 1) * n_in];
        let mut acc = 0f64;
        for k in 0..n_in { acc += dq[k] as f64 * yrow[k] as f64; }
        let got = out[n * n_out + m];
        eprintln!("  ck m={m} n={n}: got={got:.5} ref={:.5}", acc);
        let rel = if acc.abs() > 1e-6 { ((got - acc as f32) / acc as f32).abs() } else { got.abs() };
        maxrel = maxrel.max(rel);
    }
    // 타이밍 20회
    let nrep: u32 = std::env::var("VKMMQ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let nb: u32 = std::env::var("VKMMQ_B").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut per_ms = 0f64;
    for b in 0..nb {
        ctxg.begin_batch()?;
        let t0 = std::time::Instant::now();
        for _ in 0..nrep { ctxg.run(pl, ds, pipe, &pcb, gx, gy, 1)?; }
        ctxg.end_batch_wait()?;
        let el = t0.elapsed().as_secs_f64() / nrep as f64;
        eprintln!("  배치 {b}: {el:.4}ms/회");
        per_ms = el; // 마지막 배치
    }
    let dt = per_ms;
    let _ = &mut pc;
    Ok(format!(
        "vk-mmq({tname}/{spv_name} spec={sp}) t={t}: {:.4}ms/회 · maxrel={maxrel:.4} · {:.1}GB/s",
        dt * 1e3, w.data.len() as f64 / dt / 1e9
    ))
}

/// vk-ft32-check (plans/89 P1.2) — fn_tile_f32(f32/BF16 밀집 프리필 타일)의
/// 실 텐서 CPU 대조. 라우터(ffn_gate_inp, f32)형상으로 게이트 발산 원인 특정.
pub fn ft32_check(path: &str) -> Result<String, String> {
    use llm170_core::matmul::FrameState as _FS;
    use llm170_core::matmul::FrameHost as _FH;
    let model = llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w4("blk.0.ffn_gate_inp.weight").map_err(|e| e.to_string())?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let t = 64usize;
    let mut lcg = 123456789u64;
    let mut lcgf = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcgf()).collect()).collect();
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
    let wd = model.w4("blk.0.hc_attn_down.weight").map_err(|e| e.to_string())?;
    let n2 = wd.n_in as usize;
    let wi = model.w4("blk.0.hc_attn_inject.weight").map_err(|e| e.to_string())?;
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
    Ok(format!(
        "ft32-check: router max|D|={mx:.3e} bad={bad} {} | inject(f32 {}x{}) max|D|={mx2:.3e} bad={bad2} {}",
        if bad == 0 { "★" } else { "✗" },
        wi.n_out, wi.n_in,
        if bad2 == 0 { "★" } else { "✗" }
    ))
}

/// vk-moe-tile-check <mode> (plans/89 P1.1c) — q8_0/q5_K MoE 타일의 CPU 대조.
/// 모드("q8_0"|"q5_K")에 해당하는 첫 레이어의 down/gate 스택 텐서로 검증.
pub fn moe_tile_type_check(mode: &str) -> Result<String, String> {
    use llm170_core::matmul::{FrameHost as _FH, FrameState as _FS};
    let path = "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
    let model = llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let want = match mode {
        "q8_0" => llm170_gguf::GgmlType::Q8_0,
        "q5_K" => llm170_gguf::GgmlType::Q5K,
        "q4_K" => llm170_gguf::GgmlType::Q4K,
        "q5_1" => llm170_gguf::GgmlType::Q5_1,
        _ => return Ok("moe-tile-check: 모드 q8_0|q5_K|q4_K|q5_1".into()),
    };
    // 해당 타입의 첫 스택 탐색(down 우선, q5_K는 gate/up에만 존재).
    let mut found = None;
    let names = if mode == "q5_K" {
        vec!["ffn_gate_exps", "ffn_up_exps"]
    } else {
        vec!["ffn_down_exps", "ffn_gate_exps"]
    };
    let il_only = std::env::var("LLM170_MTC_IL").ok().and_then(|v| v.parse::<usize>().ok());
    for il in 0..48 {
        if let Some(want_il) = il_only
            && il != want_il
        {
            continue;
        }
        for nm in &names {
            if let Ok(w) = model.w4(&format!("blk.{il}.{nm}.weight"))
                && w.ty == want
            {
                found = Some((il, w));
                break;
            }
        }
        if found.is_some() {
            break;
        }
    }
    let (il, wd) = found.ok_or("해당 타입 스택 없음")?;
    let ne = 512usize;
    let n_in_d = wd.n_in as usize;
    let n_out_d = wd.n_out as usize / ne;
    let acc = VkAcc::new()?;
    let t = std::env::var("LLM170_MTC_T").ok().and_then(|v| v.parse().ok()).unwrap_or(130usize);
    let k = 10usize;
    let mut lcg = 987654321u64;
    let mut lcgf = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let route0: Vec<f32> = (0..ne).map(|_| lcgf() * 4.0).collect();
    let route: Vec<f32> = (0..t).flat_map(|_| route0.iter().copied()).collect();
    let xs: Vec<Vec<f32>> = (0..t * k).map(|_| (0..n_in_d).map(|_| lcgf()).collect()).collect();
    let rh = acc.frame_alloc(t * ne)?;
    let idh = acc.frame_alloc(t * k)?;
    let wth = acc.frame_alloc(t * k)?;
    let mxh = acc.frame_alloc(t * k * n_in_d)?;
    let mgh = acc.frame_alloc(t * k * n_out_d)?;
    acc.frame_write(rh, &route)?;
    let mut flat = Vec::with_capacity(t * k * n_in_d);
    for row in &xs {
        flat.extend_from_slice(row);
    }
    acc.frame_write(mxh, &flat)?;
    acc.frame_begin(t);
    acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k })?;
    let mut ids_g = vec![0u32; t * k];
    {
        acc.frame_sync();
        let g = acc_frame_ptr(&acc, idh);
        unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t * k) };
    }
    acc.frame_moe_gemm(mxh, &wd, idh, mgh, ne, k)?;
    acc.frame_begin(t);
    let mut got = vec![0f32; t * k * n_out_d];
    acc.frame_read(mgh, &mut got)?;
    for h in [rh, idh, wth, mxh, mgh] {
        let _ = acc.frame_free(h);
    }
    // CPU 참조 — ids 순행, 전expert 행 디양자화 내적.
    let m = route.iter().cloned().fold(f32::MIN, f32::max);
    let ps: Vec<f32> = route.iter().map(|&v| (v - m).exp()).collect();
    let mut idx: Vec<usize> = (0..ne).collect();
    idx.sort_by(|&a, &b| ps[b].partial_cmp(&ps[a]).unwrap().then(a.cmp(&b)));
    let sel: Vec<usize> = idx[..k].to_vec();
    let mut mx = 0f64;
    let mut bad = 0usize;
    let mut ref_row = vec![0f32; n_in_d];
    let mut checked = 0usize;
    for (r, &e) in sel.iter().enumerate() {
        if ids_g[r] as usize != e {
            mx = mx.max(1.0);
        }
        for j in 0..n_out_d.min(6) {
            llm170_core::quant::dequant_row(wd.ty, wd.data, (e * n_out_d + j) as u64, n_in_d as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(xs[r].iter()).map(|(a, b)| a * b).sum();
            let d = (got[r * n_out_d + j] as f64 - dot as f64).abs();
            if d > 2e-2 {
                bad += 1;
            }
            mx = mx.max(d);
            checked += 1;
        }
    }
        if std::env::var_os("LLM170_MTC_DBG").is_some() {
            for rr in 0..sel.len().min(12) {
                let ee = sel[rr];
                let mut s2 = 0f64;
                let mut rr_row = vec![0f32; n_in_d];
                llm170_core::quant::dequant_row(wd.ty, wd.data, (ee * n_out_d) as u64, n_in_d as u64, &mut rr_row);
                for (a, b) in rr_row.iter().zip(xs[rr].iter()) {
                    s2 += *a as f64 * *b as f64;
                }
                eprintln!("[mtc] row={rr} e={ee} got={:.5} ref={:.5}", got[rr * n_out_d], s2);
            }
        }
        if std::env::var_os("LLM170_MTC_DBG2").is_some() {
            for (r2, &e2) in sel.iter().enumerate() {
                for j2 in 0..n_out_d.min(6) {
                    let mut rr2 = vec![0f32; n_in_d];
                    llm170_core::quant::dequant_row(wd.ty, wd.data, (e2 * n_out_d + j2) as u64, n_in_d as u64, &mut rr2);
                    let dot2: f32 = rr2.iter().zip(xs[r2].iter()).map(|(a, b)| a * b).sum();
                    let d2 = (got[r2 * n_out_d + j2] as f64 - dot2 as f64).abs();
                    if d2 > 2e-2 {
                        eprintln!("[mtc2] row={r2} e={e2} j={j2} got={:.5} ref={:.5}", got[r2 * n_out_d + j2], dot2);
                    }
                }
            }
        }
        if std::env::var_os("LLM170_MTC_DBG").is_some() {
            for rr in 0..sel.len().min(10) {
                let ee = sel[rr];
                let per = wd.data.len() / 512;
                let off = ee * per;
                let d_bits = u16::from_le_bytes([wd.data[off], wd.data[off + 1]]);
                let e10 = ((d_bits >> 10) & 0x1F) as i32;
                let m10 = (d_bits & 0x3FF) as f32;
                let dv = if e10 == 0 {
                    m10 * 2f32.powi(-24)
                } else {
                    (1024.0 + m10) * 2f32.powi(e10 - 25)
                } * if d_bits & 0x8000 != 0 { -1.0 } else { 1.0 };
                eprintln!("[mtcD] row={rr} e={ee} dBits={d_bits:#06x} d={dv:.3e}");
                for jj in 1..6usize {
                    let row_b = (n_in_d / 256) * 144;
                    let off2 = off + jj * row_b;
                    let db2 = u16::from_le_bytes([wd.data[off2], wd.data[off2 + 1]]);
                    eprintln!("[mtcD]   j={jj} lo={:#04x}", db2 & 0xFF);
                }
            }
        }
    Ok(format!(
        "moe-tile-check({mode} blk.{il} down {n_out_d}x{n_in_d}, rows={}): max|D|={mx:.3e} bad={bad}/{checked} {}",
        t * k,
        if bad == 0 { "★" } else { "✗" }
    ))
}

/// vk-moe-cm-race (plans/89 재개) — 엔진 패턴 재현: t=512 청크 내 N"레이어"
/// × (MoeTop10 → ids 공유 GEMM 3회[게이트/업/다운]) + 첫 레이어 배치 중간
/// frame_read 플러시(PLE 브리지 모방). 게이트 출력을 CPU와 대조해 체크 판
/// (1회 GEMM 결정적)과 엔진(간헐 발산)의 차이를 국소화한다.
pub fn moe_cm_race_check() -> Result<String, String> {
    use llm170_core::matmul::{FrameHost as _FH, FrameState as _FS};
    let path = "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
    let model = llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let wg = model.w4("blk.0.ffn_gate_exps.weight").map_err(|e| e.to_string())?;
    let wu = model.w4("blk.0.ffn_up_exps.weight").map_err(|e| e.to_string())?;
    let wdd = model.w4("blk.0.ffn_down_exps.weight").map_err(|e| e.to_string())?;
    let ne = 512usize;
    let k = 10usize;
    let n_in = wg.n_in as usize;
    let n_out = wg.n_out as usize / ne;
    let acc = VkAcc::new()?;
    let t = std::env::var("LLM170_RACE_T").ok().and_then(|v| v.parse().ok()).unwrap_or(512usize);
    let layers = std::env::var("LLM170_RACE_L").ok().and_then(|v| v.parse().ok()).unwrap_or(12usize);
    let mut lcg = 0xC0FFEEu64;
    let mut lcgf = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let xg: Vec<Vec<f32>> = (0..t * k).map(|_| (0..n_in).map(|_| lcgf()).collect()).collect();
    let rh = acc.frame_alloc(t * ne)?;
    let idh = acc.frame_alloc(t * k)?;
    let wth = acc.frame_alloc(t * k)?;
    let mxh = acc.frame_alloc(t * k * n_in)?;
    let mgh = acc.frame_alloc(t * k * n_out)?;
    let muh = acc.frame_alloc(t * k * n_out)?;
    let mdh = acc.frame_alloc(t * k * (wdd.n_out as usize / ne))?;
    let mut flat = Vec::with_capacity(t * k * n_in);
    for row in &xg {
        flat.extend_from_slice(row);
    }
    acc.frame_write(mxh, &flat)?;
    let mut worst = 0f64;
    let mut bad_layers = 0usize;
    let mut ref_row = vec![0f32; n_in];
    for l in 0..layers {
        let route: Vec<f32> = (0..t * ne).map(|_| lcgf() * 4.0).collect();
        acc.frame_write(rh, &route)?;
        acc.frame_begin(t);
        acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k })?;
        acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
        acc.frame_moe_gemm(mxh, &wu, idh, muh, ne, k)?;
        acc.frame_moe_gemm(mxh, &wdd, idh, mdh, ne, k)?;
        if l == 0 {
            let mut probe = vec![0f32; 64];
            acc.frame_read(mgh, &mut probe)?;
        }
        acc.frame_begin(t);
        let mut got = vec![0f32; t * k * n_out];
        acc.frame_read(mgh, &mut got)?;
        let ids_ref = {
            let mut probe = vec![0u32; t * k];
            acc.frame_sync();
            let g = acc_frame_ptr(&acc, idh);
            unsafe { std::ptr::copy_nonoverlapping(g as *const u32, probe.as_mut_ptr(), t * k) };
            probe
        };
        let mut lmx = 0f64;
        for r in 0..(t * k).min(64) {
            let e = ids_ref[r] as usize;
            for j in 0..n_out.min(4) {
                llm170_core::quant::dequant_row(wg.ty, wg.data, (e * n_out + j) as u64, n_in as u64, &mut ref_row);
                let dot: f32 = ref_row.iter().zip(xg[r].iter()).map(|(a, b)| a * b).sum();
                let d = (got[r * n_out + j] as f64 - dot as f64).abs();
                lmx = lmx.max(d);
            }
        }
        if lmx > 2e-2 {
            bad_layers += 1;
        }
        worst = worst.max(lmx);
        if std::env::var_os("LLM170_RACE_DBG").is_some() {
            eprintln!("[race] L{l} max|D|={lmx:.3e}");
        }
    }
    for h in [rh, idh, wth, mxh, mgh, muh, mdh] {
        let _ = acc.frame_free(h);
    }
    Ok(format!(
        "moe-cm-race(engine-pattern t={t} x{layers}): worst={worst:.3e} bad_layers={bad_layers} {}",
        if bad_layers == 0 { "★" } else { "✗" }
    ))
}

fn hf(v: f32) -> u16 {
    // f32→f16 변환 (반올림)
    half::f16::from_f32(v).to_bits()
}


/// vk-sdot-probe — OpSDot(정수 dot) 장치 지원 검증+타이밍. plans/33.
pub fn sdot_probe() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let buf = ctx.alloc_host(16)?;
    unsafe {
        let p = buf.ptr as *mut u32;
        *p.add(0) = 0x0182_0304;      // a (부호 혼합 i8x4)
        *p.add(1) = 0xF0FF_7F01;      // b
        *p.add(2) = 0;
        *p.add(3) = 0;
    }
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/sdot_probe.spv")
        .map_err(|e| e.to_string())?;
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 1, 4)?;
    let _ = (dsl, pool);
    ctx.bind_bufs(ds, &[buf.buf]);
    let t0 = Instant::now();
    ctx.run(pl, ds, pipe, &1_000_000u32.to_le_bytes(), 1, 1, 1)?;
    let dt = t0.elapsed().as_secs_f32();
    let r = unsafe { *(buf.ptr as *const u32).add(2) };
    // CPU 기준: acc = a; 1M회 acc = sdot(acc, b) — i32 감쇠/순환값
    let mut cacc: i32 = 0x0182_0304u32 as i32;
    let b4: i32 = 0xF0FF_7F01u32 as i32;
    let bx = |v: i32, i: u32| -> i32 {
        let byte = (v >> (i * 8)) & 0xFF;
        if byte >= 128 { byte - 256 } else { byte }
    };
    for _ in 0..1_000_000 {
        let mut s = 0i32;
        for i in 0..4 { s += bx(cacc, i) * bx(b4, i); }
        cacc = s;
    }
    let expect = cacc as u32;
    Ok(format!(
        "sdot-probe: gpu={r:#010x} cpu={expect:#010x} {} · {dt:.1}ms (1M 의존 dot)",
        if r == expect { "일치" } else { "불일치" }
    ))
}

/// vk-idot-probe (plans/89 P0.1) — OpSDot(PackedVectorFormat4x8Bit) 검증+타이밍.
/// sdot_probe(plans/33)의 어셈블리 패치는 커널 문맥에서 0을 반환했다. 이번 판의
/// 차이: (a) VkCtx가 Vulkan13Features.shader_integer_dot_product를 활성화,
/// (b) spirv-as 산출물을 val 통과 구조로 직접 인코딩(.spvasm 참조).
/// mode 0=OpSDot / 1=스칼라 에뮬레이션(gemv3 dot4 동일 산술) — 동일 커널 A/B.
pub fn idot_probe() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    if !acc.ctx.lock().idot {
        return Ok("idot-probe: 장치가 shader_integer_dot_product 미지원".into());
    }
    let mut ctx = acc.ctx.lock();
    let buf = ctx.alloc_host(32)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/idot_probe.spv")
        .map_err(|e| e.to_string())?;
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 1, 8)?;
    let _ = (dsl, pool);
    ctx.bind_bufs(ds, &[buf.buf]);
    // CPU 기준 — 단일 dot(비영 검증) + 1M 의존 루프 종값.
    let bx = |v: i32, i: u32| -> i32 {
        let b = (v >> (i * 8)) & 0xFF;
        if b >= 128 { b - 256 } else { b }
    };
    let (ai, bi): (i32, i32) = (0x0182_0304u32 as i32, 0xF0FF_7F01u32 as i32);
    let single: i32 = (0..4).map(|i| bx(ai, i) * bx(bi, i)).sum();
    let mut cacc = ai;
    for _ in 0..1_000_000 {
        let mut s = 0i32;
        for i in 0..4 {
            s += bx(cacc, i) * bx(bi, i);
        }
        cacc = s;
    }
    let mut lines = String::new();
    for mode in 0..2u32 {
        unsafe {
            let p = buf.ptr as *mut u32;
            *p.add(0) = 0x0182_0304;
            *p.add(1) = 0xF0FF_7F01;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let t0 = Instant::now();
        ctx.run(pl, ds, pipe, &push_u32s(&[mode, 1_000_000]), 1024, 1, 1)?;
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        let (r2, r3) = unsafe {
            (
                *(buf.ptr as *const u32).add(2),
                *(buf.ptr as *const u32).add(3) as i32,
            )
        };
        let ok_loop = r2 == cacc as u32;
        let ok_single = r3 == single;
        lines.push_str(&format!(
            "  mode{mode}({}): 루프 {r2:#010x} {} · 단일 dot {r3} (cpu {single}) {} · {dt:.1}ms/1M\n",
            if mode == 0 { "OpSDot" } else { "스칼라" },
            if ok_loop { "★" } else { "✗" },
            if ok_single { "★" } else { "✗" },
        ));
    }
    unsafe {
        ctx.device.destroy_pipeline(pipe, None);
        ctx.device.destroy_pipeline_layout(pl, None);
    }
    Ok(format!("idot-probe (packed i8x4 dot, plans/89 P0.1):\n{lines}"))
}


/// vk-gemv8-check — gemv8 패밀리(llama mul_mat_vec 포트, f32 직결) 검증+타이밍.
pub fn gemv8_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
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
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let xa = ctx.alloc_host(t * n_in * 4)?;
    for (j, x) in xs.iter().enumerate() {
        unsafe {
            // 행 스트라이드는 바이트 — n_in f32 = n_in*4바이트 (A2: 이 오타가
            // t≥2 하니스 오염의 전부였음 — 행1이 행0의 1/4 지점을 덮어씀)
            std::ptr::copy_nonoverlapping(
                x.as_ptr(), xa.ptr.add(j * n_in * 4) as *mut f32, n_in);
        }
    }
    let ob = ctx.alloc_host(t * n_out * 4)?;  // 매핑 유지 — 판독용
    // 가중 업로드 — gemv3와 동일한 균일 청크
    let ch = ctx.max_ssbo;
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    let total = w.data.len();
    // 청크 크기 2의 거듭제곱 (WG 시프트 산술) — 마지막 청크는 실제 크기만 할당:
    // o = idx & mask 는 항상 청크 내 실데이터 오프셋만 생성하므로 패딩 불필요.
    let ch = total.next_power_of_two().min(1usize << (63 - ch.leading_zeros()));
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
        llm170_gguf::GgmlType::Q3K if std::env::var("LLM170_Q3B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q3b.spv",
        llm170_gguf::GgmlType::Q3K => "crates/backend-gpu/src/rawvk/spv/gemv8_q3.spv",
        llm170_gguf::GgmlType::Q4K if std::env::var("LLM170_Q4B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q4b.spv",
        llm170_gguf::GgmlType::Q4K => "crates/backend-gpu/src/rawvk/spv/gemv8_q4.spv",
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_Q5B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q5b.spv",
        llm170_gguf::GgmlType::Iq4Nl =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_nlb.spv",
        llm170_gguf::GgmlType::Q5K => "crates/backend-gpu/src/rawvk/spv/gemv8_q5.spv",
        llm170_gguf::GgmlType::Q6K if std::env::var("LLM170_Q6B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q6b.spv",
        llm170_gguf::GgmlType::Q6K => "crates/backend-gpu/src/rawvk/spv/gemv8_q6.spv",
        llm170_gguf::GgmlType::Iq4Xs if std::env::var("LLM170_XSB").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_xsb.spv",
        llm170_gguf::GgmlType::Iq4Xs => "crates/backend-gpu/src/rawvk/spv/gemv8_xs.spv",
        llm170_gguf::GgmlType::Q8_0 if std::env::var("LLM170_Q8B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q8b.spv",
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
    let q5b = w.ty == llm170_gguf::GgmlType::Q5K && std::env::var("LLM170_Q5B").map(|v| v != "0").unwrap_or(true);
    let q4b = w.ty == llm170_gguf::GgmlType::Q4K && std::env::var("LLM170_Q4B").map(|v| v != "0").unwrap_or(true);
    let q6b = w.ty == llm170_gguf::GgmlType::Q6K && std::env::var("LLM170_Q6B").map(|v| v != "0").unwrap_or(true);
    let q8b = w.ty == llm170_gguf::GgmlType::Q8_0 && std::env::var("LLM170_Q8B").map(|v| v != "0").unwrap_or(true);
    let q3b = w.ty == llm170_gguf::GgmlType::Q3K && std::env::var("LLM170_Q3B").map(|v| v != "0").unwrap_or(true);
    let xsb = w.ty == llm170_gguf::GgmlType::Iq4Xs && std::env::var("LLM170_XSB").map(|v| v != "0").unwrap_or(true);
    let rpf: u32 = if q5b || q4b || q6b || q8b || xsb || q3b { 2 } else if n_out < 4096 { 1 } else { 2 };   // llama NUM_ROWS=2
    let cw_log2 = 31u32 - chunk_words.leading_zeros();
    let cw_mask = (1u32 << cw_log2) - 1u32;
    // cw 단위: q5/q6(u16 typed 뷰)만 u16 단위, 나머지 u32
    let (cwpl, cwpm) = if is_q5 || is_q6 {
        (31u32 - (chunk_words * 2).leading_zeros(), (chunk_words * 2) - 1)
    } else { (cw_log2, cw_mask) };
    let push = push_u32s(&[n_in as u32, n_out as u32, t as u32, cwpl, cwpm, rpf]);
    ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
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
            llm170_core::quant::dequant_row(
                w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            mx = mx.max((dot - outs[j * n_out + r]).abs() as f64);
        }
    }
    let solo_t0 = Instant::now();
    for _ in 0..10 {
        ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
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
            ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
        }
        t4.elapsed().as_secs_f64() / 10.0
    } else { 0.0 };
    if flushed_dt > 0.0 {
        return Ok(format!(
            "gemv8-l2flush({tname}): {:.3}ms → {:.1}GB/s (웜 {})",
            flushed_dt * 1e3, w.data.len() as f64 / flushed_dt / 1e9,
            w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    if let Ok(list) = std::env::var("LLM170_MULTI") {
        // TLB/할당수 가설: 추가 텐서들을 같은 컨텍스트에 로드(상주)시킨 뒤
        // 이 텐서의 타이밍 재측정 — 속도 붕괴 시 가설 확인.
        for extra in list.split(',').filter(|x| !x.is_empty()) {
            if extra == tname { continue; }
            let w2 = match model.w(extra) { Some(w) => w, None => continue };
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
            ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
        }
        let dt2 = t2.elapsed().as_secs_f64() / 10.0;
        let _ = &solo_dt;
        return Ok(format!(
            "gemv8-multi({tname}): {:.3}ms → {:.1}GB/s (단독 {:.1})",
            dt2 * 1e3, w.data.len() as f64 / dt2 / 1e9, w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    Ok(format!(
        "gemv8({tname}) t={t}: {:.3}ms → {:.1}GB/s · max|D|={mx:.4}",
        solo_dt * 1e3,
        w.data.len() as f64 / solo_dt / 1e9
    ))
}

/// vk-tile-check — 타일(coopmat f16) 커널 vs CPU 디양자화 GEMM 검증 (plans/38 A2).
/// f16 스테이징 품질계약: maxrel 허용치 ~2e-2 (근접 아닌 구조 오류 검출 목적).
#[allow(clippy::if_same_then_else)] // 진단 A/B: 커널(spv)은 분기마다 다르고 gx 산식만 우연히 동일
pub fn tile_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    // plans/84 B: arch 판별 후 단일 로드(vk-gemv-check와 동일 패턴) — FN 멀티파트 지원.
    // 진단 전용 값 semantic — 박싱 없이 값 소유(체커 1회 로드).
    #[allow(clippy::large_enum_variant)]
    enum AnyModel {
        Q35(llm170_core::qwen35::Model),
        Q4(llm170_core::qwen4exp::Model4),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    let model = if is_q4 {
        AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    } else {
        AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    };
    let w = match &model {
        AnyModel::Q35(m) => m.w(tname).ok_or("텐서 없음")?,
        AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string())?,
    };
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let ms4gy = std::env::var("LLM170_TILE_MS4GY").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K;
    let msall = std::env::var("LLM170_TILE_MSALL").map(|v| v=="1").unwrap_or(false);
    let gy2 = std::env::var("LLM170_TILE_GY2").map(|v| v=="1").unwrap_or(false) && msall && w.ty != llm170_gguf::GgmlType::Q5K;
    let bn128 = std::env::var("LLM170_TILE_BN128").map(|v| v=="1").unwrap_or(false) && msall && w.ty != llm170_gguf::GgmlType::Q5K;
    if t < 1 || (t > 128 && !ms4gy && !gy2 && !msall) {
        return Err("tile 검증 t는 1..=128 (MS4GY/GY2/MSALL는 512까지)".into());
    }
    let (spv_name, n_kb, extra) = match w.ty {
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false) => ("tile_ms128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if msall => ("tile_ms4.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q4K if bn128 => ("tile_q4k128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q6K if bn128 => ("tile_q6k128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q3K if bn128 => ("tile_q3k128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q8_0 if bn128 => ("tile_q8128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Iq4Xs if bn128 => ("tile_xs128.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Iq4Nl if bn128 => ("tile_nl128.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Q4K if msall => ("tile_q4kms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q6K if msall => ("tile_q6kms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q3K if msall => ("tile_q3kms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q8_0 if msall => ("tile_q8ms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Iq4Xs if msall => ("tile_xsms.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Iq4Nl if msall => ("tile_nlms.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS4GY").map(|v| v=="1").unwrap_or(false) => ("tile_ms4gy.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS4").map(|v| v=="1").unwrap_or(false) => ("tile_ms4.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_OCC").map(|v| v=="1").unwrap_or(false) => ("tile128o.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K => ("tile128_q5k.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q4K => ("tile_q4k.spv", 10, 0),
        llm170_gguf::GgmlType::Q6K => ("tile_q6k.spv", 10, 0),
        llm170_gguf::GgmlType::Q8_0 => ("tile_q8.spv", 10, 0),
        llm170_gguf::GgmlType::Iq4Xs => ("tile_xs.spv", 11, 1),   // ktab
        llm170_gguf::GgmlType::Iq4Nl => ("tile_nl.spv", 11, 1),    // ktab
        llm170_gguf::GgmlType::Iq3S => ("tile_iq3s.spv", 11, 2),   // grid3s
        _ => return Err("tile 검증 불가 타입".into()),
    };
    let is_128 = w.ty == llm170_gguf::GgmlType::Q5K || msall
        || std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false);
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut seed = 0x5deece66u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    // xq 양자화 (GPU quant — 비트 검증 완료 경로)
    let xq_w = xq_words(n_in);
    let xqb = if std::env::var_os("LLM170_TILE_BDEV").is_some() {
        ctx.alloc((t * xq_w * 4).max(t * n_in * 2))?
    } else {
        ctx.alloc_host((t * xq_w * 4).max(t * n_in * 2))?
    };
    acc.quant_upload(&mut ctx, &xs, n_in, xqb.buf)?;
    let ob = ctx.alloc_host(t * n_out * 4)?;
    // 가중 업로드
    let total = w.data.len();
    let ch = total.next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let (ktab, grid, dummy) = acc.ensure_shared(&mut ctx)?;
    while wbufs.len() < 8 {
        wbufs.push(dummy);
    }
    let spv = std::fs::read(format!("crates/backend-gpu/src/rawvk/spv/{spv_name}"))
        .map_err(|e| e.to_string())?;
    // plans/41: ms 패밀리는 push 5필드 [n_in,n_out,xq_w,t,tok_base] (pb=20)
    let bn128spv = spv_name.ends_with("128.spv");
    let is_msfam = spv_name.ends_with("ms.spv") || spv_name.ends_with("mgy.spv")
        || spv_name == "tile_ms4.spv" || bn128spv;
    let slab: usize = if bn128spv { 128 } else { 64 };
    let ms128fam_any = std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false);
    let q51fam = w.ty == llm170_gguf::GgmlType::Q5_1;   // plans/84 B: 6필드 push(24B)
    let pb: u32 = if q51fam { 24 } else if bn128spv || ms128fam_any { 24 } else if is_msfam { 20 } else if is_128 { 16 } else { 24 };
    let mpush = |tt: u32, base: u32| push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, tt, base]);
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, n_kb, pb)?;
    let _ = (dsl, pool);
    let mut binds: Vec<vk::Buffer> = wbufs.clone();
    binds.push(xqb.buf);
    binds.push(ob.buf);
    if extra == 1 {
        binds.push(ktab);
    } else if extra == 2 {
        binds.push(grid);
    }
    ctx.bind_bufs(ds, &binds);
    let cw = (ch / 4) as u32;
    let cw = cw.next_power_of_two();
    let cw_log2 = 31u32 - cw.leading_zeros();
    let cw_mask = cw - 1;
    let gx = if std::env::var("LLM170_TILE_MS4GY").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)   // tile_ms4gy: WG당 64행
    } else if spv_name.starts_with("tile_ms128") && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)   // tile_ms128 계열: WG당 64행 × 128토큰
    } else if msall || (std::env::var("LLM170_TILE_MS4").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K) {
        (n_out as u32).div_ceil(64)   // tile_ms4(msall): WG당 64행
    } else {
        (n_out as u32).div_ceil(128)
    };
    let t0 = Instant::now();
    if ms4gy {
        // gy 병렬: 단일 디스패치, gy=t/64, push t=64 (커널은 슬래브당 64토큰)
        let gy = (t as u32).div_ceil(64);
        let push = mpush(64, 0);
        ctx.run(pl, ds, pipe, &push, gy, gx, 1)?;   // plans/41 zs: 슬래브 x, 행 y
        let outs: Vec<f32> = unsafe {
            let mut v = vec![0f32; t * n_out];
            std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
            v
        };
        let _ = &outs;
        // (검증·벤치 공용 경로로 흐르게 outs 사용은 아래와 동일 — 여기선 run만 대체)
        // 아래 기존 로직이 outs를 다시 읽으므로 여기서 반환하지 않고 흐름 유지:
        // → 실제로는 아래 outs 재판독이 이 run 결과를 본다.
        let _ = t;
    }
    if q51fam {
        // plans/84 B: q5_1 타일판 — 128토큰 슬래브 + tok_base + 청크 wsh(엔진과 동일 규약).
        for tb in (0..t).step_by(128) {
            let nt = (t - tb).min(128) as u32;
            let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, tb as u32, cw_log2]);
            ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
        }
    } else if gy2 {
        let gy = (t as u32).div_ceil(64);
        let push = mpush(64, 0);
        ctx.run(pl, ds, pipe, &push, gy, gx, 1)?;
    } else if is_128 && !ms4gy {
        let ms128fam = std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false);
        if is_msfam {
            // ms 패밀리: t>슬래브는 분할 (tok_base로 전 토큰 커버; ms256 = 128토큰 슬래브)
            for tb in (0..t).step_by(slab) {
                let nt = (t - tb).min(slab) as u32;
                let push = if bn128spv {
                    push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, 0u32, tb as u32])
                } else {
                    mpush(nt, tb as u32)
                };
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            }
        } else {
            let push = if ms128fam {
                // ms128: [n_in,n_out,xq_w,t,row_off,tok_base] (pb=24)
                push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, 0u32, 0u32])
            } else {
                push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32])
            };
            ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
        }
    } else {
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, cw_log2, cw_mask]);
        ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    }
    ctx.flush2()?;   // plans/46: 판독 전 GPU 완료 — 종전 dt는 비동기 제출만 잼(실측 허수)
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; t * n_out];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
        v
    };
    let dt = t0.elapsed().as_secs_f64();
    // 배치 타이밍 (신뢰): N회 녹화 → 1회 제출·대기 — 단독 submit 계측 결함 회피
    if std::env::var_os("LLM170_TILE_BENCH").is_some() {
        let n = std::env::var("LLM170_TILE_BENCH").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(100);
        let t1 = std::time::Instant::now();
        ctx.begin_batch()?;
        for _ in 0..n {
            if gy2 {
                // gy 단일 디스패치 (엔진 gy와 동일): 슬래브 x, 행 y
                let gy = (t as u32).div_ceil(64);
                let push = mpush(64, 0);
                ctx.run(pl, ds, pipe, &push, gy, gx, 1)?;
            } else if is_msfam && t > slab {
                // 순차 슬래브 (엔진 비-gy 경로와 동일 형태): tok_base=tb로 전 토큰 커버
                for tb in (0..t).step_by(slab) {
                    let nt = (t - tb).min(slab) as u32;
                    let push = mpush(nt, tb as u32);
                    ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
                }
            } else if is_msfam {
                let push = mpush(t as u32, 0);
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            } else if is_128 {
                let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32]);
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            } else {
                let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, cw_log2, cw_mask]);
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            }
        }
        ctx.end_batch_wait()?;
        let per = t1.elapsed().as_secs_f64() / n as f64;
        return Ok(format!(
            "tile-bench({tname}/{spv_name}) t={t}: {:.4}ms/회 × {n} → {:.1}GB/s",
            per * 1e3, w.data.len() as f64 / per / 1e9
        ));
    }
    // CPU 기준: 디양자화 · f64 내적 — 행 0..64 + WG 경계/꼬리 샘플 (plans/40:
    // 행 64+ 미검증이 ms 패밀리 매핑 버그 은폐 — 전 WG 경계 커버)
    let mut rows: Vec<usize> = (0..n_out.min(64)).collect();
    for r in [63usize, 64, 65, 127, 128, 129, 191, 192, n_out.saturating_sub(2), n_out - 1] {
        if r < n_out && !rows.contains(&r) {
            rows.push(r);
        }
    }
    let mut ref_row = vec![0.0f32; n_in];
    let mut maxrel = 0f64;
    let mut worst = (0usize, 0usize, 0f64, 0f64);
    let mut bad_rows = 0usize;
    let mut bucket_bad = std::collections::BTreeMap::<u64, usize>::new();
    for (j, x) in xs.iter().enumerate() {
        let mut row_bad = false;
        for &r in &rows {
            llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f64 = ref_row.iter().zip(x.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let g = outs[j * n_out + r] as f64;
            let rel = (g - dot).abs() / dot.abs().max(1.0);
            if rel > maxrel {
                maxrel = rel;
                worst = (j, r, dot, g);
            }
            if rel > 2e-2 {
                row_bad = true;
                *bucket_bad.entry((r / 64) as u64).or_default() += 1;
            }
        }
        if row_bad {
            bad_rows += 1;
        }
    }
    eprintln!("[bucket] 2%초과 행(64행 버킷): {:?}", bucket_bad);
    if std::env::var_os("LLM170_TILE_DUMP").is_some() {
        eprintln!("[dump] outs[0][0..4] = {:?}", &outs[0..4]);
        if std::env::var_os("LLM170_TILE_DUMP").is_some() && t >= 1 {
            let mut zr = None;
            for (i, v) in outs[0..n_out].iter().enumerate() { if v.abs() < 1e-30 { zr = Some(i); break; } }
            let mut last_nz = 0;
            for (i, v) in outs[0..n_out].iter().enumerate() { if v.abs() > 1e-30 { last_nz = i; } }
            eprintln!("[dump] tok0 첫0행={:?} 마지막비0행={last_nz}", zr);
        }
        if n_out >= 6144 {
            eprintln!("[dump] tok0 rows 6078..6082 = {:?}", &outs[6078..6082]);
            eprintln!("[dump] tok0 rows 6126..6130 = {:?}", &outs[6126..6130]);
        }
        if n_out > 6144 {
            eprintln!("[dump] tok0 rows 6140..6144 = {:?}", &outs[6140..6144]);
        } else {
            eprintln!("[dump] tok0 rows {}..{} = {:?}", n_out-4, n_out, &outs[n_out-4..n_out]);
        }
        eprintln!("[dump] xs[0][0..6] = {:?}", &xs[0][0..6]);
    }
    Ok(format!(
        "tile({tname}/{spv_name}) t={t}: {dt:.3}ms · maxrel={maxrel:.4} (worst j={} r={} ref={:.4} gpu={:.4}) · 2%초과 토큰 {bad_rows}/{t}",
        worst.0, worst.1, worst.2, worst.3
    ))
}



/// dbg-q3 (plans/40) — tile_q3kms 디코드를 행 0 전원소 덤프해 CPU 진실과 대조.
pub fn q3_dbg(path: &str, tname: &str) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let total = w.data.len();
    let ch = total.next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let ob = ctx.alloc_host(n_in * 4 + 4096)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/dbg_q3.spv").map_err(|e| e.to_string())?;
    let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(&spv, 2, 4)?;
    ctx.bind_bufs(ds, &[wbufs[0], ob.buf]);
    let _gx = (n_in as u32) / 64 / 32 * 64;  // sb 수/64
    let n_sb = (n_in / 32) as u32;
    let gx = n_sb.div_ceil(64);
    let push = push_u32s(&[n_in as u32]);
    ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; n_in + 1024];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), n_in + 1024);
        v
    };
    // CPU 진실
    let mut ref_row = vec![0f32; n_in];
    llm170_core::quant::dequant_row(w.ty, w.data, 0, n_in as u64, &mut ref_row);
    let mut bad = 0usize;
    let mut first = vec![];
    for k in 0..n_in {
        let rel = (outs[k] - ref_row[k]).abs() / ref_row[k].abs().max(1e-3);
        if rel > 1e-3 {
            bad += 1;
            if first.len() < 10 { first.push(format!("k={k} gpu={:.5} ref={:.5}", outs[k], ref_row[k])); }
        }
    }
    if std::env::var_os("LLM170_Q3_WIDE").is_some() {
        let isv: Vec<u32> = outs[n_in..n_in+1024].iter().map(|f| *f as u32).collect();
        let _ = &isv;
        eprintln!("[is] k352/368 (sb11 hf0/hf1의 is_i×0.001): {:.4} {:.4}", outs[352]*1000.0, outs[368]*1000.0);
        eprintln!("[is] k0/16 (sb0): {:.4} {:.4}", outs[0]*1000.0, outs[16]*1000.0);
        eprintln!("[is] sb8..15: {:?}", &isv[16..32]);
        eprintln!("[wide] k48..63 gpu: {:?}", &outs[48..64]);
eprintln!("[wide] k352..383 gpu: {:?}", &outs[352..384]);
        // 불일치 k의 (sb&3, kc>>4) 히스토그램
        let mut hh = std::collections::BTreeMap::<(usize, usize), usize>::new();
        for k in 0..n_in {
            let rel = (outs[k] - ref_row[k]).abs() / ref_row[k].abs().max(1e-3);
            if rel > 1e-3 {
                *hh.entry(((k >> 5) & 3, (k & 16) >> 4)).or_default() += 1;
            }
        }
        eprintln!("[hist] (j, hf) → 불일치 수: {:?}", hh);
    }
    Ok(format!("dbg-q3: 불일치 {bad}/{n_in} | {}", first.join(" · ")))
}



/// dbg-q3b (plans/40) — gemv8_q3b 디코드 원소 덤프 ↔ CPU 진실.
pub fn q3b_dbg(path: &str, tname: &str) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut b = ctx.alloc(w.data.len())?;
    unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()) };
    ctx.unmap(&mut b)?;
    let ob = ctx.alloc_host(n_in * 4)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/dbg_q3b.spv").map_err(|e| e.to_string())?;
    let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(&spv, 2, 4)?;
    ctx.bind_bufs(ds, &[b.buf, ob.buf]);
    let gx = (n_in as u32).div_ceil(512);
    let push = push_u32s(&[n_in as u32]);
    ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; n_in];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), n_in);
        v
    };
    let mut ref_row = vec![0f32; n_in];
    llm170_core::quant::dequant_row(w.ty, w.data, 0, n_in as u64, &mut ref_row);
    let mut bad = 0usize;
    let mut first = vec![];
    for k in 0..n_in {
        let rel = (outs[k] - ref_row[k]).abs() / ref_row[k].abs().max(1e-3);
        if rel > 1e-3 {
            bad += 1;
            if first.len() < 10 { first.push(format!("k={k} gpu={:.5} ref={:.5}", outs[k], ref_row[k])); }
        }
    }
    Ok(format!("dbg-q3b: 불일치 {bad}/{n_in} | {}", first.join(" · ")))
}

/// mmv-check (plans/40) — llama mul_mat_vec_q5_k 직접 구동 격리 측정 (t≥1 dmmv).
/// 스펙 {BLOCK 64, NUM_ROWS 2, COLS 1} + full_subgroups(강제 wave64) —
/// llama RADV 설정 직역. B=f32 [t][K], D=f32 [t][M].
pub fn mmv_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/mmv_llm.spv").map_err(|e| e.to_string())?;
    let acc = VkAcc::new()?;
    let mut ctxg = acc.ctx.lock();
    let ab_vram = std::env::var("VKMMQ_VRAM").map(|v| v == "1").unwrap_or(true);
    let ab = if ab_vram {
        let mut b = ctxg.alloc(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()) };
        ctxg.unmap(&mut b)?;
        b
    } else {
        let mut b = ctxg.alloc_host(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()) };
        ctxg.unmap(&mut b)?;
        b
    };
    // y f32 [t][K]
    let mut seed = 0x1234u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let mut yf: Vec<f32> = Vec::with_capacity(n_in * t);
    let y0 = std::env::var("VKMMQ_Y0").map(|v| v == "1").unwrap_or(false);
    for _ in 0..n_in * t { yf.push(if y0 { 0.0 } else { lcg() }); }
    let mut bb = ctxg.alloc_host(n_in * t * 4)?;
    unsafe { std::ptr::copy_nonoverlapping(yf.as_ptr() as *const u8, bb.ptr, n_in * t * 4) };
    ctxg.unmap(&mut bb)?;
    let db = ctxg.alloc_host(n_out * t * 4)?;
    unsafe { std::ptr::write_bytes(db.ptr, 0, n_out * t * 4) };
    // F0/F1 dummy
    let mut fb = ctxg.alloc_host(64)?;
    ctxg.unmap(&mut fb)?;
    let spec = vec![64u32, 2, 1];
    let (_dsl, pl, _dp, ds, pipe) = ctxg.pipeline_spec_fg(&spv, 5, 13 * 4, &spec, true)?;
    ctxg.bind_bufs(ds, &[ab.buf, bb.buf, db.buf, fb.buf, fb.buf]);
    let mut pc: Vec<u32> = vec![
        n_in as u32, n_in as u32, n_in as u32, n_out as u32,   // ncols, stride_a, stride_b, stride_d
        0, 0, 0,            // batch strides
        0,                  // fusion_flags
        0, t as u32, 1, 1, 1,  // base_wg_y, ne02, ne12, b2, b3
    ];
    let _ = &mut pc;
    let pcb: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gx = (n_out as u32).div_ceil(2);
    ctxg.begin_batch()?;
    ctxg.run(pl, ds, pipe, &pcb, gx, t as u32, 1)?;
    ctxg.end_batch_wait()?;
    let out: &[f32] = unsafe { std::slice::from_raw_parts(db.ptr as *const f32, n_out * t) };
    eprintln!("mmv dbg: out[0..16]={:?} (y0={})", &out[0..16], y0);
    // 근사 검증: 첫 토큰 첫 4행
    let mut dq = vec![0f32; n_in];
    let mut maxrel = 0f64;
    for &(m, n) in &[(0usize, 0usize), (100, 0), (2000, 0), (6143, 0)] {
        if m >= n_out || n >= t { continue; }
        llm170_core::quant::dequant_row(w.ty, w.data, m as u64, n_in as u64, &mut dq);
        let yrow = &yf[n * n_in..(n + 1) * n_in];
        let acc: f64 = dq.iter().zip(yrow).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let got = out[n * n_out + m] as f64;
        let rel = if acc.abs() > 1e-6 { ((got - acc) / acc).abs() } else { got.abs() };
        maxrel = maxrel.max(rel);
    }
    let nrep: u32 = std::env::var("VKMMQ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    ctxg.begin_batch()?;
    let t0 = std::time::Instant::now();
    for _ in 0..nrep { ctxg.run(pl, ds, pipe, &pcb, gx, t as u32, 1)?; }
    ctxg.end_batch_wait()?;
    let dt = t0.elapsed().as_secs_f64() / nrep as f64;
    Ok(format!(
        "mmv({tname}) t={t}: {dt:.4}ms · maxrel={maxrel:.4} · {:.1}GB/s",
        w.data.len() as f64 / dt / 1e9
    ))
}

/// vk-frame-check — plans/84 B: 프레임 코어(버퍼 레지스트리+엘리먼트와이스+
/// 상주 GEMM)의 CPU 대조 검증. 각 op를 LCG 데이터로 실행해 판독 비교.
pub fn frame_check(path: &str, tname: &str) -> Result<String, String> {
    use std::time::Instant;
    // 진단 전용 값 semantic — 박싱 없이 값 소유(체커 1회 로드).
    #[allow(clippy::large_enum_variant)]
    enum AnyModel {
        Q35(llm170_core::qwen35::Model),
        Q4(llm170_core::qwen4exp::Model4),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    let model = if is_q4 {
        AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    } else {
        AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    };
    let w = match &model {
        AnyModel::Q35(m) => m.w(tname).ok_or("텐서 없음")?,
        AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string())?,
    };
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let t = 3usize;
    let mut seed = 0x5deece66u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let mut fails = 0usize;
    let mut report = String::new();
    let t0 = Instant::now();
    use llm170_core::matmul::FrameState;
    acc.frame_begin(t);

    // ── 1) RmsRows (w_reps=2) ──
    {
        let n = 64usize;
        let reps = 2usize;
        let xh = acc.frame_alloc(n * reps * t)?;
        let wh = acc.frame_alloc(n * reps)?;
        let oh = acc.frame_alloc(n * reps * t)?;
        // hip 규약: 입력 x도 reps*t행 (res_hc는 hc 반복 레이아웃).
        let mut x = Vec::with_capacity(n * reps * t);
        for _ in 0..reps * t {
            x.extend((0..n).map(|_| lcg()));
        }
        let wv: Vec<f32> = (0..n * reps).map(|_| lcg()).collect();
        acc.frame_write(xh, &x)?;
        acc.frame_write(wh, &wv)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_op(&llm170_core::matmul::FrameOp::RmsRows {
            x: xh, w: wh, out: oh, eps: 1e-5, n, w_reps: reps,
        })?;
        let mut got = vec![0f32; n * reps * t];
        acc.frame_read(oh, &mut got)?;
        let mut mx = 0f64;
        for row in 0..reps * t {
            let s: f64 = (0..n).map(|i| (x[row * n + i] as f64).powi(2)).sum();
            let inv = 1.0 / (s / n as f64 + 1e-5).sqrt();
            for i in 0..n {
                let exp = (x[row * n + i] as f64) * inv * wv[(row % reps) * n + i] as f64;
                mx = mx.max((got[row * n + i] as f64 - exp).abs());
            }
        }
        let ok = mx < 5e-5;
        if !ok { fails += 1; }
        report.push_str(&format!("RmsRows(w_reps={reps}) max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));
        acc.frame_free(xh)?; acc.frame_free(wh)?; acc.frame_free(oh)?;
    }
    // ── 2) SiluDiv / 3) SiluMul / 4) Scale ──
    {
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let b: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let ah = acc.frame_alloc(n)?;
        let bh = acc.frame_alloc(n)?;
        let oh = acc.frame_alloc(n)?;
        acc.frame_write(ah, &a)?;
        acc.frame_write(bh, &b)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_op(&llm170_core::matmul::FrameOp::SiluDiv { t: ah, div: 320.0, n })?;
        let mut got = vec![0f32; n];
        acc.frame_read(ah, &mut got)?;
        let mut mx = 0f64;
        // CPU 참조(stages/hc.rs)와 동일: silu(x/div). (구 검증식 silu(x)/div 는
        // 셰이더와 같은 잘못을 새겨넣고 있었다 — plans/86 §1.)
        for i in 0..n {
            let x = a[i] / 320.0f32;
            let exp = (x / (1.0 + (-x).exp())) as f64;
            mx = mx.max((got[i] as f64 - exp).abs());
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("SiluDiv max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));

        acc.frame_write(ah, &a)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::SiluMul { g: ah, u: bh, out: oh, n })?;
        acc.frame_read(oh, &mut got)?;
        mx = 0.0;
        for i in 0..n {
            let exp = (a[i] / (1.0 + (-a[i]).exp())) as f64 * b[i] as f64;
            mx = mx.max((got[i] as f64 - exp).abs());
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("SiluMul max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));

        acc.frame_write(ah, &a)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::Scale { t: ah, s: 0.5, n })?;
        acc.frame_read(ah, &mut got)?;
        mx = 0.0;
        for i in 0..n {
            mx = mx.max((got[i] as f64 - a[i] as f64 * 0.5).abs());
        }
        let ok = mx < 1e-7;
        if !ok { fails += 1; }
        report.push_str(&format!("Scale max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));
        acc.frame_free(ah)?; acc.frame_free(bh)?; acc.frame_free(oh)?;
    }
    // ── 5) CopyRows / 6) BcastRows / 7) AxpyScaled(t) ──
    {
        let n = 100usize;
        let src: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let sh = acc.frame_alloc(n)?;
        let dh = acc.frame_alloc(2 * n)?;
        acc.frame_write(sh, &src)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_op(&llm170_core::matmul::FrameOp::CopyRows { src: sh, dst: dh, src_off: 7, dst_off: n + 3, n: n - 10 })?;
        let mut got = vec![0f32; 2 * n];
        acc.frame_read(dh, &mut got)?;
        let mut ok = true;
        for i in 0..(n - 10) {
            if (got[n + 3 + i] - src[7 + i]).abs() > 1e-7 { ok = false; break; }
        }
        if !ok { fails += 1; }
        report.push_str(&format!("CopyRows {} | ", if ok { "OK" } else { "FAIL" }));

        let bh2 = acc.frame_alloc(n * t)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::BcastRows { src: sh, dst: bh2, n, rows: t })?;
        acc.frame_read(bh2, &mut got)?;
        let _ = &mut got;
        let mut got2 = vec![0f32; n * t];
        acc.frame_read(bh2, &mut got2)?;
        ok = true;
        for r in 0..t {
            for i in 0..n {
                if (got2[r * n + i] - src[i]).abs() > 1e-7 { ok = false; }
            }
        }
        if !ok { fails += 1; }
        report.push_str(&format!("BcastRows {} | ", if ok { "OK" } else { "FAIL" }));

        let per = 32usize;
        let y: Vec<f32> = (0..t * per).map(|_| lcg()).collect();
        let xx: Vec<f32> = (0..t * per).map(|_| lcg()).collect();
        let ss: Vec<f32> = (0..t).map(|_| lcg()).collect();
        let yh = acc.frame_alloc(t * per)?;
        let xh2 = acc.frame_alloc(t * per)?;
        let ssh = acc.frame_alloc(t)?;
        acc.frame_write(yh, &y)?;
        acc.frame_write(xh2, &xx)?;
        acc.frame_write(ssh, &ss)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::AxpyScaled { y: yh, x: xh2, s: ssh, n: t * per })?;
        let mut got3 = vec![0f32; t * per];
        acc.frame_read(yh, &mut got3)?;
        let mut mx = 0f64;
        for j in 0..t * per {
            let exp = y[j] as f64 + xx[j] as f64 * ss[j / per] as f64;
            mx = mx.max((got3[j] as f64 - exp).abs());
        }
        let aok = mx < 1e-6;
        if !aok { fails += 1; }
        report.push_str(&format!("AxpyScaled(t={t}) max|D|={mx:.2e} {}", if aok { "OK" } else { "FAIL" }));
        acc.frame_free(sh)?; acc.frame_free(dh)?; acc.frame_free(bh2)?;
        acc.frame_free(yh)?; acc.frame_free(xh2)?; acc.frame_free(ssh)?;
    }
    // ── 8) frame_mm — 상주 quant+GEMM vs CPU 디양자화 내적 ──
    {
        let xh = acc.frame_alloc(n_in * t)?;
        let oh = acc.frame_alloc(n_out * t)?;
        let mut flat = Vec::with_capacity(n_in * t);
        for row in &xs { flat.extend_from_slice(row); }
        acc.frame_write(xh, &flat)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_mm(xh, &w, oh, t)?;
        let mut got = vec![0f32; n_out * t];
        acc.frame_read(oh, &mut got)?;
        let mut mx = 0f64;
        let mut ref_row = vec![0f32; n_in];
        for (j, x) in xs.iter().enumerate() {
            for r in 0..n_out.min(16) {
                llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
                let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                if std::env::var_os("LLM170_DBG_8").is_some() && j == 0 && r < 6 {
                    eprintln!("[8] r={r} got={:.6} ref={:.6}", got[j * n_out + r], dot);
                }
                mx = mx.max((dot as f64 - got[j * n_out + r] as f64).abs());
            }
        }
        let ok = mx < 5e-3;
        if !ok { fails += 1; }
        report.push_str(&format!("frame_mm max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
        acc.frame_free(xh)?; acc.frame_free(oh)?;
    }
    // ── 8b) frame_mm q4_K 밀집 (plans/88 P2): mode-1 타일 산술 분리 검증 —
    //    그룹화(perm/rowexp) 없이 타일 커널 자체의 CPU 대조. ──
    {
        use llm170_core::matmul::FrameHost;
        if let AnyModel::Q4(m) = &model
            && let Ok(w4k) = m.w4("blk.0.ffn_gate_shexp.weight")
        {
                let ni = w4k.n_in as usize;
                let no = w4k.n_out as usize;
                let xs4: Vec<Vec<f32>> = (0..t).map(|_| (0..ni).map(|_| lcg()).collect()).collect();
                let xh = acc.frame_alloc(ni * t)?;
                let oh = acc.frame_alloc(no * t)?;
                let mut flat = Vec::with_capacity(ni * t);
                for row in &xs4 { flat.extend_from_slice(row); }
                acc.frame_write(xh, &flat)?;
                acc.frame_mm(xh, &w4k, oh, t)?;
                let mut got = vec![0f32; no * t];
                acc.frame_read(oh, &mut got)?;
                let mut mx = 0f64;
                let mut ref_row = vec![0f32; ni];
                for (j, x) in xs4.iter().enumerate() {
                    for r in 0..no.min(12) {
                        llm170_core::quant::dequant_row(w4k.ty, w4k.data, r as u64, ni as u64, &mut ref_row);
                        let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                        if std::env::var_os("LLM170_DBG_8B").is_some() && j == 0 && r < 4 {
                            eprintln!("[8b] r={r} got={:.6} ref={:.6}", got[j * no + r], dot);
                        }
                        mx = mx.max((dot as f64 - got[j * no + r] as f64).abs());
                    }
                }
                let ok = mx < 5e-3;
                if !ok { fails += 1; }
                report.push_str(&format!("| frame_mm-q4k max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
                acc.frame_free(xh)?; acc.frame_free(oh)?;
            }
    }
    // ── 9) MoE: top10 → 그룹 GEMM → 가중합 (게이트 가중, k=10) ──
    {
        use llm170_core::matmul::{FrameHost, FrameState};
        let k = 10usize;
        // FN 게이트 가중은 q4_K 스택(대부분 층) — 스택 텐서 하나로 검증.
        let wg = match &model {
            AnyModel::Q4(m) => m.w4("blk.0.ffn_gate_exps.weight").map_err(|e| e.to_string())?,
            AnyModel::Q35(m) => m.w("blk.0.ffn_gate.weight").ok_or("텐서 없음")?,
        };
        let n_in_m = wg.n_in as usize;
        let ne = match &model {
            AnyModel::Q4(_) => 512usize,
            AnyModel::Q35(_) => 1usize,
        };
        // 스택 텐서: 전문가당 폭만 출력에 쓴다(frame_moe_gemm 규약).
        let n_out_m = wg.n_out as usize / ne;
        if ne == 512 {
            let route: Vec<f32> = (0..t * ne).map(|_| lcg() * 4.0).collect();
            let mxs: Vec<Vec<f32>> = (0..t * k).map(|_| (0..n_in_m).map(|_| lcg()).collect()).collect();
            let rh = acc.frame_alloc(t * ne)?;
            let idh = acc.frame_alloc(t * k)?;
            let wth = acc.frame_alloc(t * k)?;
            let mxh = acc.frame_alloc(t * k * n_in_m)?;
            let mgh = acc.frame_alloc(t * k * n_out_m)?;
            let outh = acc.frame_alloc(t * n_out_m)?;
            acc.frame_write(rh, &route)?;
            let mut flat2 = Vec::with_capacity(t * k * n_in_m);
            for row in &mxs { flat2.extend_from_slice(row); }
            acc.frame_write(mxh, &flat2)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k })?;
            let mut ids_g = vec![0u32; t * k];
            {
                acc.frame_sync();
                let g = acc_frame_ptr(&acc, idh);
                unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t * k) };
            }
            acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeWeightedSum { ys: mgh, wt: wth, out: outh, k, n: n_out_m })?;
            let mut got = vec![0f32; t * n_out_m];
            acc.frame_read(outh, &mut got)?;
            // CPU 기준: softmax top-k + 디양자화 내적 + 가중합
            let mut mx = 0f64;
            for tok in 0..t {
                let r = &route[tok * ne..(tok + 1) * ne];
                let m = r.iter().cloned().fold(f32::MIN, f32::max);
                let ps: Vec<f32> = r.iter().map(|&v| (v - m).exp()).collect();
                let zs: f32 = ps.iter().sum();
                let mut idx: Vec<usize> = (0..ne).collect();
                idx.sort_by(|&a, &b| ps[b].partial_cmp(&ps[a]).unwrap().then(a.cmp(&b)));
                let sel: Vec<usize> = idx[..k].to_vec();
                let wsel: Vec<f32> = sel.iter().map(|&e| ps[e] / zs).collect();
                let wsum: f32 = wsel.iter().sum::<f32>().max(6.103515625e-5);
                for (j, &e) in sel.iter().enumerate() {
                    if ids_g[tok * k + j] as usize != e { mx = mx.max(1.0); }
                }
                let per_exp = n_out_m;
                let mut ref_row = vec![0f32; n_in_m];
                for j in 0..per_exp.min(8) {
                    let mut acc2 = 0f64;
                    for (ki, &e) in sel.iter().enumerate() {
                        llm170_core::quant::dequant_row(wg.ty, wg.data, (e * per_exp + j) as u64, n_in_m as u64, &mut ref_row);
                        let dot: f32 = ref_row.iter().zip(mxs[tok * k + ki].iter()).map(|(a, b)| a * b).sum();
                        acc2 += dot as f64 * (wsel[ki] / wsum) as f64;
                    }
                    let d = (got[tok * n_out_m + j] as f64 - acc2).abs();
                    mx = mx.max(d);
                    if std::env::var_os("LLM170_DBG_9A").is_some() && tok == 0 && j < 4 {
                        eprintln!("[9a] tok={tok} j={j} got={:.6} ref={:.6}", got[tok * n_out_m + j], acc2);
                    }
                }
            }
            let ok = mx < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| MoE(k={k}) max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [rh, idh, wth, mxh, mgh, outh] { acc.frame_free(h)?; }
        }
    }
    // ── 9b) MoE down direct-ids (plans/88 P1): q5_1 스택 t=1 — ids 직판독
    //    경로의 CPU 대조. 행마다 전문가가 다르고 x 행도 행마다 독립이다
    //    (게이트/up의 브로드캐스트 입력과 달리 행 r 을 정확히 읽어야 한다). ──
    {
        use llm170_core::matmul::{FrameHost, FrameState};
        if let AnyModel::Q4(_) = &model {
            let k = 10usize;
            let ne = 512usize;
            let wd = match &model {
                AnyModel::Q4(m) => m.w4("blk.0.ffn_down_exps.weight").map_err(|e| e.to_string())?,
                AnyModel::Q35(_) => unreachable!(),
            };
            let n_in_d = wd.n_in as usize;
            let n_out_d = wd.n_out as usize / ne;
            let route: Vec<f32> = (0..ne).map(|_| lcg() * 4.0).collect();
            let xs: Vec<Vec<f32>> = (0..k).map(|_| (0..n_in_d).map(|_| lcg()).collect()).collect();
            let rh = acc.frame_alloc(ne)?;
            let idh = acc.frame_alloc(k)?;
            let wth = acc.frame_alloc(k)?;
            let mxh = acc.frame_alloc(k * n_in_d)?;
            let mgh = acc.frame_alloc(k * n_out_d)?;
            acc.frame_write(rh, &route)?;
            let mut flat = Vec::with_capacity(k * n_in_d);
            for row in &xs {
                flat.extend_from_slice(row);
            }
            acc.frame_write(mxh, &flat)?;
            // t=1 — MoeTop10도 t토큰을 찍는다(route ne·ids k 크기 버퍼).
            // direct-ids 경로 강제(§9의 t=3 rows=30 도 지나가지만 q4_K만
            // 거친다. down 은 여기서 t=1 판을 본다).
            acc.frame_begin(1);
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
                route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k,
            })?;
            let mut ids_g = vec![0u32; k];
            {
                acc.frame_sync();
                let g = acc_frame_ptr(&acc, idh);
                unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), k) };
            }
            acc.frame_moe_gemm(mxh, &wd, idh, mgh, ne, k)?;
            acc.frame_begin(t);
            let mut got = vec![0f32; k * n_out_d];
            acc.frame_read(mgh, &mut got)?;
            let m = route.iter().cloned().fold(f32::MIN, f32::max);
            let ps: Vec<f32> = route.iter().map(|&v| (v - m).exp()).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| ps[b].partial_cmp(&ps[a]).unwrap().then(a.cmp(&b)));
            let sel: Vec<usize> = idx[..k].to_vec();
            let mut mx = 0f64;
            let mut ref_row = vec![0f32; n_in_d];
            for (r, &e) in sel.iter().enumerate() {
                if ids_g[r] as usize != e {
                    mx = mx.max(1.0);
                }
                for j in 0..n_out_d.min(8) {
                    llm170_core::quant::dequant_row(
                        wd.ty, wd.data, (e * n_out_d + j) as u64, n_in_d as u64, &mut ref_row,
                    );
                    let dot: f32 = ref_row.iter().zip(xs[r].iter()).map(|(a, b)| a * b).sum();
                    if std::env::var_os("LLM170_DBG_9B").is_some() && r == 0 {
                        eprintln!(
                            "[9b] r={r} e={e} j={j} got={:.6} ref={:.6}",
                            got[r * n_out_d + j],
                            dot
                        );
                    }
                    let d = (got[r * n_out_d + j] as f64 - dot as f64).abs();
                    mx = mx.max(d);
                }
            }
            let ok = mx < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| MoE-down-ids(k={k}) max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [rh, idh, wth, mxh, mgh] { acc.frame_free(h)?; }
        }
    }
    // ── 9c) MoE 타일 대량행 (plans/88 P2): t=210·k=10 → rows=2100 — 디바이스
    //    그룹화+타일 경로의 CPU 대조. 소형(§9 t=3)은 direct-ids만 지나가
    //    않으므로 대량 행이 필요하다. ──
    {
        use llm170_core::matmul::{FrameHost, FrameState};
        if let AnyModel::Q4(_) = &model {
            let k = 10usize;
            let ne = 512usize;
            let t2 = std::env::var("LLM170_T2").ok().and_then(|v| v.parse().ok()).unwrap_or(210usize);
            let wg = match &model {
                AnyModel::Q4(m) => m.w4("blk.0.ffn_gate_exps.weight").map_err(|e| e.to_string())?,
                AnyModel::Q35(_) => unreachable!(),
            };
            let n_in_m = wg.n_in as usize;
            let n_out_m = wg.n_out as usize / ne;
            // t2토큰 × ne 라우트 — 균등 랜덤이면 대부분의 전문가가 비게 되어
            // rows=2100이 희소 행을 만든다(실측 결함 재현 조건).
            let route: Vec<f32> = (0..t2 * ne).map(|_| lcg() * 4.0).collect();
            let mxs: Vec<Vec<f32>> = (0..t2 * k).map(|_| (0..n_in_m).map(|_| lcg()).collect()).collect();
            let rh = acc.frame_alloc(t2 * ne)?;
            let idh = acc.frame_alloc(t2 * k)?;
            let wth = acc.frame_alloc(t2 * k)?;
            let mxh = acc.frame_alloc(t2 * k * n_in_m)?;
            let mgh = acc.frame_alloc(t2 * k * n_out_m)?;
            acc.frame_write(rh, &route)?;
            let mut flat = Vec::with_capacity(t2 * k * n_in_m);
            for row in &mxs { flat.extend_from_slice(row); }
            acc.frame_write(mxh, &flat)?;
            acc.frame_begin(t2);
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
                route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k,
            })?;
            acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
            acc.frame_begin(t);
            let mut got = vec![0f32; t2 * k * n_out_m];
            acc.frame_read(mgh, &mut got)?;
            let mut ids_g = vec![0u32; t2 * k];
            {
                acc.frame_sync();
                let g = acc_frame_ptr(&acc, idh);
                unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t2 * k) };
            }
            // CPU 참조: (토큰,슬롯) 행별 디양자 내적 — 순열과 무관하게 행 자체가 맞는지.
            let mut mx = 0f64;
            let mut ref_row = vec![0f32; n_in_m];
            for row in 0..t2 * k {
                let e = ids_g[row] as usize;
                for j in 0..n_out_m.min(4) {
                    llm170_core::quant::dequant_row(wg.ty, wg.data, (e * n_out_m + j) as u64, n_in_m as u64, &mut ref_row);
                    let dot: f32 = ref_row.iter().zip(mxs[row].iter()).map(|(a, b)| a * b).sum();
                    let d = (got[row * n_out_m + j] as f64 - dot as f64).abs();
                    if std::env::var_os("LLM170_DBG_9C3").is_some() && d > 5e-3 && row < 40 {
                        eprintln!("[9c3] row={row} e={e} j={j} got={:.6} ref={:.6} d={d:.4}", got[row * n_out_m + j], dot);
                    }
                    if std::env::var_os("LLM170_DBG_9C2").is_some() && row < 210 {
                        eprintln!("[9c] row={row} e={e} j={j} got={:.6} ref={:.6}", got[row * n_out_m + j], dot);
                    }
                    mx = mx.max(d);
                }
            }
            let ok = mx < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| MoE-tile-2100 max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [rh, idh, wth, mxh, mgh] { acc.frame_free(h)?; }
        }
    }
    // ── 10) 어텐션 반쪽: HcGateMean/HcCombine/NormGated/GdnBetaG/Sigmoid/Split3 ──
    {
        use llm170_core::matmul::FrameHost;
        let n = 48usize;
        let hc = 4usize;
        // HcGateMean
        let xn: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
        let gate: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
        let xnh = acc.frame_alloc(t * hc * n)?;
        let gth = acc.frame_alloc(t * hc * n)?;
        let mkh = acc.frame_alloc(t * n)?;
        acc.frame_write(xnh, &xn)?;
        acc.frame_write(gth, &gate)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::HcGateMean { xn: xnh, gate: gth, out: mkh, hc, n })?;
        let mut got = vec![0f32; t * n];
        acc.frame_read(mkh, &mut got)?;
        let mut mx = 0f64;
        for ti in 0..t {
            for i in 0..n {
                let mut exp = 0f64;
                for s in 0..hc {
                    let k = (ti * hc + s) * n + i;
                    exp += xn[k] as f64 * (1.0 / (1.0 + (-gate[k] as f64).exp()));
                }
                mx = mx.max((got[ti * n + i] as f64 - exp / hc as f64).abs());
            }
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| HcGateMean {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // HcCombine — res 초기화 후 += 검증
        let res0: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
        let resh = acc.frame_alloc(t * hc * n)?;
        let inj: Vec<f32> = (0..t * hc).map(|_| lcg() * 2.0).collect();
        let ijh = acc.frame_alloc(t * hc)?;
        acc.frame_write(resh, &res0)?;
        acc.frame_write(ijh, &inj)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::HcCombine { res: resh, out: mkh, inj: ijh, hc, n, total: 0 })?;
        let mut resg = vec![0f32; t * hc * n];
        acc.frame_read(resh, &mut resg)?;
        mx = 0.0;
        for ti in 0..t {
            for i in 0..n {
                for s in 0..hc {
                    let g = 2.0 / (1.0 + (-(inj[ti * hc + s] as f64) / hc as f64).exp());
                    let exp = res0[(ti * hc + s) * n + i] as f64 + got[ti * n + i] as f64 * g;
                    mx = mx.max((resg[(ti * hc + s) * n + i] as f64 - exp).abs());
                }
            }
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| HcCombine {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // NormGated(sigmoid) — d=32, n_h=3
        let d = 32usize;
        let nh = 3usize;
        let o3: Vec<f32> = (0..t * nh * d).map(|_| lcg()).collect();
        let z3: Vec<f32> = (0..t * nh * d).map(|_| lcg()).collect();
        let w3: Vec<f32> = (0..nh * d).map(|_| lcg()).collect();
        let o3h = acc.frame_alloc(t * nh * d)?;
        let z3h = acc.frame_alloc(t * nh * d)?;
        let w3h = acc.frame_alloc(nh * d)?;
        let n3h = acc.frame_alloc(t * nh * d)?;
        acc.frame_write(o3h, &o3)?;
        acc.frame_write(z3h, &z3)?;
        acc.frame_write(w3h, &w3)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::NormGated { o: o3h, z: z3h, w: w3h, out: n3h, eps: 1e-5, d, n_h: nh })?;
        let mut ng = vec![0f32; t * nh * d];
        acc.frame_read(n3h, &mut ng)?;
        mx = 0.0;
        for row in 0..t * nh {
            let s: f64 = (0..d).map(|i| (o3[row * d + i] as f64).powi(2)).sum();
            let inv = 1.0 / (s / d as f64 + 1e-5).sqrt();
            for i in 0..d {
                let exp = o3[row * d + i] as f64 * inv * w3[(row % nh) * d + i] as f64 * (1.0 / (1.0 + (-z3[row * d + i] as f64).exp()));
                mx = mx.max((ng[row * d + i] as f64 - exp).abs());
            }
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| NormGated {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // GdnBetaG — dt_rank=6, n_h=6·t
        let dr = 6usize;
        let nh2 = dr * t;
        let b2: Vec<f32> = (0..nh2).map(|_| lcg()).collect();
        let a2: Vec<f32> = (0..nh2).map(|_| lcg()).collect();
        let dtb: Vec<f32> = (0..dr).map(|_| lcg()).collect();
        let sa: Vec<f32> = (0..dr).map(|_| lcg()).collect();
        let (b2h, a2h, dth, sah, bgh) = (acc.frame_alloc(nh2)?, acc.frame_alloc(nh2)?, acc.frame_alloc(dr)?, acc.frame_alloc(dr)?, acc.frame_alloc(nh2 * 2)?);
        acc.frame_write(b2h, &b2)?;
        acc.frame_write(a2h, &a2)?;
        acc.frame_write(dth, &dtb)?;
        acc.frame_write(sah, &sa)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::GdnBetaG { b: b2h, a: a2h, dtb: dth, sa: sah, bg: bgh, n_h: nh2 })?;
        let mut bgv = vec![0f32; nh2 * 2];
        acc.frame_read(bgh, &mut bgv)?;
        mx = 0.0;
        for h in 0..nh2 {
            let h0 = h % dr;
            let e0 = 1.0 / (1.0 + (-b2[h] as f64).exp());
            let x = (a2[h] as f64 + dtb[h0] as f64).min(80.0);
            let sp = (1.0 + x.exp()).ln();
            let e1 = (sp * sa[h0] as f64).exp();
            mx = mx.max((bgv[h * 2] as f64 - e0).abs() + (bgv[h * 2 + 1] as f64 - e1).abs());
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| GdnBetaG {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // Sigmoid + Split3
        let v4: Vec<f32> = (0..128).map(|_| lcg() * 3.0).collect();
        let v4h = acc.frame_alloc(128)?;
        acc.frame_write(v4h, &v4)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::Sigmoid { t: v4h, n: 128 })?;
        let mut sv = vec![0f32; 128];
        acc.frame_read(v4h, &mut sv)?;
        mx = 0.0;
        for j in 0..128 { mx = mx.max((sv[j] as f64 - (1.0 / (1.0 + (-v4[j] as f64).exp()))).abs()); }
        let ok = mx < 5e-7;
        if !ok { fails += 1; }
        report.push_str(&format!("| Sigmoid {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        let (n0, n1, n2) = (10usize, 6usize, 8usize);
        let tot = n0 + n1 + n2;
        let s3: Vec<f32> = (0..t * tot).map(|_| lcg()).collect();
        let s3h = acc.frame_alloc(t * tot)?;
        let (d0h, d1h, d2h) = (acc.frame_alloc(t * n0)?, acc.frame_alloc(t * n1)?, acc.frame_alloc(t * n2)?);
        acc.frame_write(s3h, &s3)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::Split3 { src: s3h, d0: d0h, d1: d1h, d2: d2h, n0, n1, n2 })?;
        let mut g0 = vec![0f32; t * n0];
        let mut g1 = vec![0f32; t * n1];
        let mut g2 = vec![0f32; t * n2];
        acc.frame_read(d0h, &mut g0)?;
        acc.frame_read(d1h, &mut g1)?;
        acc.frame_read(d2h, &mut g2)?;
        let mut ok = true;
        for ti in 0..t {
            for j in 0..n0 { if (g0[ti * n0 + j] - s3[ti * tot + j]).abs() > 1e-7 { ok = false; } }
            for j in 0..n1 { if (g1[ti * n1 + j] - s3[ti * tot + n0 + j]).abs() > 1e-7 { ok = false; } }
            for j in 0..n2 { if (g2[ti * n2 + j] - s3[ti * tot + n0 + n1 + j]).abs() > 1e-7 { ok = false; } }
        }
        if !ok { fails += 1; }
        report.push_str(&format!("| Split3 {}", if ok { "OK" } else { "FAIL" }));
        for h in [xnh, gth, mkh, resh, ijh, o3h, z3h, w3h, n3h, b2h, a2h, dth, sah, bgh, v4h, s3h, d0h, d1h, d2h] { acc.frame_free(h)?; }
    }
    // ── 11) GDN AR 청크 불변성 — 단일 t=8 대 2×t=4, 최종 상태·출력 비교 ──
    {
        use llm170_core::matmul::FrameHost as _FH;
        use llm170_core::matmul::FrameState as _FS;
        let hv = 4usize;
        let hk = 2usize;
        // 커널 레이아웃: 상태 u행 = kdim 128(32레인×4) — d≥128 필수(hip 규약).
        let d = 128usize;
        let (ks, vs) = (hk * d, hv * d);
        let full = 8usize;
        let mut s2 = 0xabcdu64;
        let mut lc2 = move || {
            s2 = s2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s2 >> 33) as f32 / 2147483648.0 - 0.5
        };
        let q8: Vec<f32> = (0..full * ks).map(|_| lc2()).collect();
        let k8: Vec<f32> = (0..full * ks).map(|_| lc2()).collect();
        let v8: Vec<f32> = (0..full * vs).map(|_| lc2()).collect();
        let bg8: Vec<f32> = (0..full * hv * 2).map(|_| lc2()).collect();
        let st0: Vec<f32> = (0..hv * d * d).map(|_| lc2() * 0.1).collect();
        // limit: 처리할 프리픽스 토큰 수. chunk: 청크 크기.
        let run_case = |chunk: usize, limit: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
            let acc2 = VkAcc::new()?;
            acc2.set_ctx_len(64);
            let qh = acc2.frame_alloc(full * ks)?;
            let kh = acc2.frame_alloc(full * ks)?;
            let vh = acc2.frame_alloc(full * vs)?;
            let bh = acc2.frame_alloc(full * hv * 2)?;
            let sh = acc2.frame_alloc(hv * d * d)?;
            acc2.frame_write(qh, &q8)?;
            acc2.frame_write(kh, &k8)?;
            acc2.frame_write(vh, &v8)?;
            acc2.frame_write(bh, &bg8)?;
            acc2.frame_write(sh, &st0)?;
            let mut got = vec![0f32; limit * vs];
            for p0 in (0..limit).step_by(chunk) {
                let tt = chunk.min(limit - p0);
                // 슬라이스 입력 버퍼 (q/k/v/bg의 [p0, p0+tt))
                let qs = acc2.frame_alloc(tt * ks)?;
                let ksb = acc2.frame_alloc(tt * ks)?;
                let vsb = acc2.frame_alloc(tt * vs)?;
                let bsb = acc2.frame_alloc(tt * hv * 2)?;
                let osb = acc2.frame_alloc(tt * vs)?;
                acc2.frame_write(qs, &q8[p0 * ks..(p0 + tt) * ks])?;
                acc2.frame_write(ksb, &k8[p0 * ks..(p0 + tt) * ks])?;
                acc2.frame_write(vsb, &v8[p0 * vs..(p0 + tt) * vs])?;
                acc2.frame_write(bsb, &bg8[p0 * hv * 2..(p0 + tt) * hv * 2])?;
                acc2.frame_begin(tt);
                acc2.frame_gdn_ar(qs, ksb, vsb, bsb, sh, osb, 1, hk, hv, d)?;
                let mut part = vec![0f32; tt * vs];
                acc2.frame_read(osb, &mut part)?;
                got[p0 * vs..(p0 + tt) * vs].copy_from_slice(&part);
                for h in [qs, ksb, vsb, bsb, osb] { acc2.frame_free(h)?; }
            }
            let mut stf = vec![0f32; hv * d * d];
            acc2.frame_read(sh, &mut stf)?;
            Ok((got, stf))
        };
        // t=1 결정론
        let (_a, s0a) = run_case(1, 1).map_err(|e| e.to_string())?;
        let (_b, s0b) = run_case(1, 1).map_err(|e| e.to_string())?;
        let mut m0 = 0f64;
        for i in 0..s0a.len() { m0 = m0.max((s0a[i] as f64 - s0b[i] as f64).abs()); }
        eprintln!("[gnar] t=1 상태 결정론={m0:.1e}");
        // t=8 결정론 + 청크 불변
        // ── 12b) MoE 청크 불변성 — 동일 64토큰, 1호출 vs 4×16 호출 ──
        {
            use llm170_core::matmul::FrameHost as _FH2;
            use llm170_core::matmul::FrameState as _FS2;
            let k10 = 10usize;
            let tt = 64usize;
            let route64: Vec<f32> = {
                let mut s3 = 0xc0deu64;
                (0..tt * 512).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s3 >> 33) as f32 / 2147483648.0 - 0.5) * 4.0 }).collect()
            };
            // down 경로도 같은 시험 — n_in=640(행 760B, 패딩 대상).
            let wg2 = match &model {
                AnyModel::Q4(m) => m.w4("blk.0.ffn_down_exps.weight").map_err(|e| e.to_string())?,
                AnyModel::Q35(m) => m.w("blk.0.ffn_down.weight").ok_or("텐서 없음")?,
            };
            let n_g = wg2.n_in as usize;
            let per_out = wg2.n_out as usize / 512;
            let mx_rows: Vec<f32> = {
                let mut s3 = 0x5a5au64;
                (0..tt * n_g).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
            };
            let wd_rows: Vec<f32> = Vec::new();
            let run_moe = |chunk: usize| -> Result<Vec<f32>, String> {
                let acc5 = VkAcc::new()?;
                let mxh = acc5.frame_alloc(tt * k10 * n_g)?;
                let rh = acc5.frame_alloc(tt * 512)?;
                let idh = acc5.frame_alloc(tt * k10)?;
                let wth = acc5.frame_alloc(tt * k10)?;
                let outh = acc5.frame_alloc(tt * k10 * per_out)?;
                // mxsel: 토큰별 10슬롯 동일 입력(토큰 t의 행 = mx_rows[t])
                acc5.frame_write(rh, &route64)?;
                let mut out = vec![0f32; tt * per_out];
                for p0 in (0..tt).step_by(chunk) {
                    let c = chunk.min(tt - p0);
                    // 청크 라우트를 [0, c·512)에 적립 — 엔진이 mroute를 청크행으로
                    // 다시 쓰는 것과 동일 규약.
                    let mut rbuf = vec![0f32; c * 512];
                    rbuf.copy_from_slice(&route64[p0 * 512..(p0 + c) * 512]);
                    // rh는 프레임 버퍼 — 청크 라우트를 앞 c행에 기록(호스트 직접)
                    {
                        let g = acc5.framebufs.lock();
                        let b = g.get(&rh).unwrap();
                        unsafe { std::ptr::copy_nonoverlapping(rbuf.as_ptr(), b.ptr as *mut f32, rbuf.len()) };
                    }
                    // 청크 mx를 [0, c·k10·n_g)에 적립(엔진의 mxsel 규약).
                    {
                        let g = acc5.framebufs.lock();
                        let b = g.get(&mxh).unwrap();
                        unsafe {
                            for t2 in 0..c {
                                for s in 0..k10 {
                                    std::ptr::copy_nonoverlapping(
                                        mx_rows[(p0 + t2) * n_g..].as_ptr(),
                                        b.ptr.add((t2 * k10 + s) * n_g * 4) as *mut f32,
                                        n_g);
                                }
                            }
                        }
                    }
                    acc5.frame_begin(c);
                    acc5.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: 512, k_sel: k10 })?;
                    acc5.frame_moe_gemm(mxh, &wg2, idh, outh, 512, k10)?;
                    // 게이트 출력만 비교(가중합/스캐터 생략 — gemm 자체 검증).
                    // 취하는 것: 각 토큰의 슬롯0 행(k10행 중 첫 행) — 토큰별 대표.
                    let mut part = vec![0f32; c * k10 * per_out];
                    acc5.frame_read(outh, &mut part)?;
                    for t2 in 0..c {
                        let src = t2 * k10 * per_out;
                        out[(p0 + t2) * per_out..(p0 + t2 + 1) * per_out]
                            .copy_from_slice(&part[src..src + per_out]);
                    }
                }
                let _ = wd_rows;
                Ok(out)
            };
            let m1 = run_moe(64).map_err(|e| e.to_string())?;
            let m2 = run_moe(16).map_err(|e| e.to_string())?;
            let mut mm = 0f64;
            for i in 0..m1.len() { mm = mm.max((m1[i] as f64 - m2[i] as f64).abs()); }
            eprintln!("[moech] 게이트 GEMM 청크 불변 max|D|={mm:.1e}");
            // ── 12c) f32 라우터 폴백 그룹 청크 불변성 — 실제 ffn_gate_inp ──
            {
                use llm170_core::matmul::FrameHost as _FH3;
                let wr = match &model {
                    AnyModel::Q4(m) => m.w4("blk.0.ffn_gate_inp.weight").map_err(|e| e.to_string())?,
                    AnyModel::Q35(m) => m.w("blk.0.ffn_gate.weight").ok_or("텐서 없음")?,
                };
                let nr = wr.n_in as usize;
                let nout_r = wr.n_out as usize;
                let mixr: Vec<f32> = {
                    let mut s3 = 0xfeedfaceu64;
                    (0..64 * nr).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
                };
                let run_r = |chunk: usize| -> Result<Vec<f32>, String> {
                    let acc6 = VkAcc::new()?;
                    let xh = acc6.frame_alloc(64 * nr)?;
                    let oh = acc6.frame_alloc(64 * nout_r)?;
                    acc6.frame_write(xh, &mixr)?;
                    let mut out = vec![0f32; 64 * nout_r];
                    for p0 in (0..64).step_by(chunk) {
                        let c = chunk.min(64 - p0);
                        // 청크 입력을 [0, c·nr)에 적립
                        {
                            let g = acc6.framebufs.lock();
                            let b = g.get(&xh).unwrap();
                            unsafe { std::ptr::copy_nonoverlapping(mixr[p0 * nr..].as_ptr(), b.ptr as *mut f32, c * nr) };
                        }
                        acc6.frame_begin(c);
                        acc6.frame_mm_group(xh, std::slice::from_ref(&wr), std::slice::from_ref(&oh), c)?;
                        let mut part = vec![0f32; c * nout_r];
                        acc6.frame_read(oh, &mut part)?;
                        out[p0 * nout_r..(p0 + c) * nout_r].copy_from_slice(&part);
                    }
                    Ok(out)
                };
                let r1 = run_r(64).map_err(|e| e.to_string())?;
                let r2 = run_r(16).map_err(|e| e.to_string())?;
                let mut mr = 0f64;
                for i in 0..r1.len() { mr = mr.max((r1[i] as f64 - r2[i] as f64).abs()); }
                eprintln!("[rtech] f32 라우터 폴백 청크 불변 max|D|={mr:.1e}");
            }
        }
        let (o1, s1) = run_case(8, 8).map_err(|e| e.to_string())?;
        let (o1b, s1b) = run_case(8, 8).map_err(|e| e.to_string())?;
        let (o2, s2v) = run_case(4, 8).map_err(|e| e.to_string())?;
        let mut mo = 0f64;
        for i in 0..o1.len() { mo = mo.max((o1[i] as f64 - o2[i] as f64).abs()); }
        let mut ms = 0f64;
        for i in 0..s1.len() { ms = ms.max((s1[i] as f64 - s2v[i] as f64).abs()); }
        let mut md = 0f64;
        for i in 0..o1.len() { md = md.max((o1[i] as f64 - o1b[i] as f64).abs()); }
        let mut mds = 0f64;
        for i in 0..s1.len() { mds = mds.max((s1[i] as f64 - s1b[i] as f64).abs()); }
        eprintln!("[gnar] t=8 결정론 out={md:.1e} st={mds:.1e}");
        let mut first_diff = None;
        let mut ndiff = 0usize;
        for i in 0..s1.len() {
            if s1[i] != s1b[i] { ndiff += 1; if first_diff.is_none() { first_diff = Some(i); } }
        }
        eprintln!("[gnar] 첫 상이 idx={:?} 상이={ndiff}/{}", first_diff, s1.len());
        let ok = mo < 1e-5 && ms < 1e-5;
        if !ok { fails += 1; }
        report.push_str(&format!("| GdnARchunk(t1={m0:.0e}) out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
        // ── 12) GdnConv 청크 불변성 — conv 링 상태 + 출력 ──
        {
            let ch = 48usize;
            let ck = 4usize;
            let conv_total = 8usize;
            let qkv8: Vec<f32> = (0..conv_total * ch).map(|_| {
                s2 = 0u64.wrapping_add(0); // (클로저 이동으로 새 랜덤은 불가 — 상수 시드 재사용)
                0.0
            }).collect();
            let _ = qkv8;
            let conv_src: Vec<f32> = {
                let mut s3 = 0xfeedu64;
                (0..conv_total * ch).map(|_| {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                }).collect()
            };
            let cw: Vec<f32> = {
                let mut s3 = 0xbeefu64;
                (0..ch * ck).map(|_| {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                }).collect()
            };
            let st0c: Vec<f32> = {
                let mut s3 = 0x1234u64;
                (0..(ck - 1) * ch).map(|_| {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                }).collect()
            };
            let run_conv = |chunk: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
                let acc3 = VkAcc::new()?;
                let src = acc3.frame_alloc(conv_total * ch)?;
                let cwb = acc3.frame_alloc(ch * ck)?;
                let stb = acc3.frame_alloc((ck - 1) * ch)?;
                acc3.frame_write(src, &conv_src)?;
                acc3.frame_write(cwb, &cw)?;
                acc3.frame_write(stb, &st0c)?;
                let mut out = vec![0f32; conv_total * ch];
                for p0 in (0..conv_total).step_by(chunk) {
                    let tt = chunk.min(conv_total - p0);
                    let inb = acc3.frame_alloc(tt * ch)?;
                    let ob = acc3.frame_alloc(tt * ch)?;
                    acc3.frame_write(inb, &conv_src[p0 * ch..(p0 + tt) * ch])?;
                    acc3.frame_begin(tt);
                    acc3.frame_op(&llm170_core::matmul::FrameOp::GdnConv {
                        qkv: inb, cw: cwb, state: stb, out: ob, ch, k: ck, t_len: tt,
                    })?;
                    let mut part = vec![0f32; tt * ch];
                    acc3.frame_read(ob, &mut part)?;
                    out[p0 * ch..(p0 + tt) * ch].copy_from_slice(&part);
                    acc3.frame_free(inb)?;
                    acc3.frame_free(ob)?;
                }
                let mut stf = vec![0f32; (ck - 1) * ch];
                acc3.frame_read(stb, &mut stf)?;
                Ok((out, stf))
            };
            let (c1, k1) = run_conv(8).map_err(|e| e.to_string())?;
            let (c2, k2) = run_conv(4).map_err(|e| e.to_string())?;
            let (_c3, k3) = run_conv(8).map_err(|e| e.to_string())?;
            let mut mo = 0f64;
            for i in 0..c1.len() { mo = mo.max((c1[i] as f64 - c2[i] as f64).abs()); }
            let mut ms = 0f64;
            for i in 0..k1.len() { ms = ms.max((k1[i] as f64 - k2[i] as f64).abs()); }
            let mut mdet = 0f64;
            for i in 0..k1.len() { mdet = mdet.max((k1[i] as f64 - k3[i] as f64).abs()); }
            eprintln!("[gcv] conv out={mo:.1e} st={ms:.1e} 결정론 st={mdet:.1e}");
            let ok = mo < 1e-6 && ms < 1e-6;
            if !ok { fails += 1; }
            report.push_str(&format!("| GdnConvChunk out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
            // ── 13) GdnBetaG 청크 불변성 (실측 형상 dr=48) ──
            {
                let dr = 48usize;
                let tot = 64usize;
                let (bsrc, asrc): (Vec<f32>, Vec<f32>) = {
                    let mut s3 = 0x9999u64;
                    let mut f1 = || { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 };
                    ((0..dr * tot).map(|_| f1()).collect(), (0..dr * tot).map(|_| f1()).collect())
                };
                let dtbv: Vec<f32> = {
                    let mut s3 = 0x8888u64;
                    (0..dr).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
                };
                let sav: Vec<f32> = {
                    let mut s3 = 0x7777u64;
                    (0..dr).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
                };
                let run_bg = |chunk: usize| -> Result<Vec<f32>, String> {
                    let acc4 = VkAcc::new()?;
                    let mut out = vec![0f32; dr * 2 * tot];
                    // 공유 dtb/sa 상수
                    let dth = acc4.frame_alloc(dr)?;
                    let sah = acc4.frame_alloc(dr)?;
                    acc4.frame_write(dth, &dtbv)?;
                    acc4.frame_write(sah, &sav)?;
                    for p0 in (0..tot).step_by(chunk) {
                        let tt = chunk.min(tot - p0);
                        let bh = acc4.frame_alloc(dr * tt)?;
                        let ah = acc4.frame_alloc(dr * tt)?;
                        let gh = acc4.frame_alloc(dr * tt * 2)?;
                        acc4.frame_write(bh, &bsrc[p0 * dr..(p0 + tt) * dr])?;
                        acc4.frame_write(ah, &asrc[p0 * dr..(p0 + tt) * dr])?;
                        acc4.frame_begin(tt);
                        acc4.frame_op(&llm170_core::matmul::FrameOp::GdnBetaG { b: bh, a: ah, dtb: dth, sa: sah, bg: gh, n_h: dr * tt })?;
                        let mut part = vec![0f32; dr * 2 * tt];
                        acc4.frame_read(gh, &mut part)?;
                        out[p0 * dr * 2..(p0 + tt) * dr * 2].copy_from_slice(&part);
                        acc4.frame_free(bh)?; acc4.frame_free(ah)?; acc4.frame_free(gh)?;
                    }
                    Ok(out)
                };
                let g1 = run_bg(64).map_err(|e| e.to_string())?;
                let g2 = run_bg(16).map_err(|e| e.to_string())?;
                let mut mb = 0f64;
                for i in 0..g1.len() { mb = mb.max((g1[i] as f64 - g2[i] as f64).abs()); }
                eprintln!("[gbg] chunk64 vs chunk16 max|D|={mb:.1e}");
                let ok = mb < 1e-6;
                if !ok { fails += 1; }
                report.push_str(&format!("| GdnBetaGChunk {mb:.1e} {}", if ok { "OK" } else { "FAIL" }));
            }
            // ── 14) QSA 디코드 선택(qsa_sel_dev) — 호스트 top-k 비트 일치 ──
            // 풀 사전 적립(append) + 디코드 1토큰 선택. 호스트 참조는 셰이더와
            // 동일 산술열(f64 순차 rms, f64 회전, 4누산 도트, 정수 순위).
            {
                use llm170_core::matmul::{FrameState as _, QsaOps as _};
                let (ih, dm, r, top_k) = (16usize, 128usize, 128usize, 512usize);
                let n_bulk = 1024usize;
                let n_past = n_bulk + 1;
                let eps = 1e-5f32;
                let full = 900usize;
                let acc5 = VkAcc::new()?;
                acc5.set_ctx_len(n_past);
                let mut s3 = 0x5a5au64;
                let mut lcg = || {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                };
                let ik_all: Vec<f32> = (0..n_past * dm).map(|_| lcg()).collect();
                let iq1: Vec<f32> = (0..ih * dm).map(|_| lcg()).collect();
                let iqw: Vec<f32> = (0..dm).map(|_| 0.5 + lcg().abs()).collect();
                let ikw: Vec<f32> = (0..dm).map(|_| 0.5 + lcg().abs()).collect();
                let cs: Vec<f32> = (0..n_past * dm).map(|_| lcg()).collect();
                // (a) 풀 사전 적립 — 행 0..n_bulk.
                {
                    let ikh = acc5.frame_alloc(n_bulk * dm)?;
                    acc5.frame_write(ikh, &ik_all[..n_bulk * dm])?;
                    acc5.qsa_idx_append_dev(full, 0, ikh, n_bulk, 0, dm, r, &ikw, &cs, eps)
                        .map_err(|e| e.to_string())?;
                    acc5.frame_free(ikh)?;
                }
                // (b) 디코드 토큰 — qsa_sel_dev가 마지막 행 적립+블록키까지.
                let (sd, od, list_len) = {
                    let ikh = acc5.frame_alloc(dm)?;
                    acc5.frame_write(ikh, &ik_all[n_bulk * dm..])?;
                    let iqh = acc5.frame_alloc(ih * dm)?;
                    acc5.frame_write(iqh, &iq1)?;
                    let out = acc5
                        .qsa_sel_dev(full, 0, iqh, ikh, 1, n_bulk, ih, dm, r, top_k, &iqw, &ikw, &cs, eps)
                        .map_err(|e| e.to_string())?;
                    acc5.frame_free(ikh)?;
                    acc5.frame_free(iqh)?;
                    out
                };
                // (c) 호스트 참조.
                let n_blocks = n_past / r;
                let rms_scale = |parts: &[f32; 32]| -> f32 {
                    let mut sum = 0f64;
                    for uu in 0..32 {
                        sum += parts[uu] as f64;
                    }
                    1.0f32 / (sum / dm as f64 + eps as f64).sqrt() as f32
                };
                let rope = |v: &mut [f32], csrow: &[f32]| {
                    let half = dm / 2;
                    for p in 0..half {
                        let c = csrow[p * 2] as f64;
                        let sf = csrow[p * 2 + 1] as f64;
                        let (x0, x1) = (v[p] as f64, v[p + half] as f64);
                        v[p] = (x0 * c - x1 * sf) as f32;
                        v[p + half] = (x0 * sf + x1 * c) as f32;
                    }
                };
                let mut bk = vec![0f32; n_blocks * dm];
                for b in 0..n_blocks {
                    let mut pvs = [[0f32; 4]; 32];
                    let mut parts = [0f32; 32];
                    for u in 0..32 {
                        for j in 0..r {
                            let row = &ik_all[(b * r + j) * dm..][..dm];
                            for k in 0..4 {
                                pvs[u][k] += row[u * 4 + k];
                            }
                        }
                        for k in 0..4 {
                            pvs[u][k] /= r as f32;
                        }
                        parts[u] = pvs[u][0] * pvs[u][0]
                            + pvs[u][1] * pvs[u][1]
                            + pvs[u][2] * pvs[u][2]
                            + pvs[u][3] * pvs[u][3];
                    }
                    let scale = rms_scale(&parts);
                    let out = &mut bk[b * dm..][..dm];
                    for u in 0..32 {
                        for k in 0..4 {
                            out[u * 4 + k] = pvs[u][k] * scale * ikw[u * 4 + k];
                        }
                    }
                    rope(out, &cs[(b * r) * dm..]);
                }
                let mut iqr = vec![0f32; ih * dm];
                {
                    for h in 0..ih {
                        let mut parts = [0f32; 32];
                        let row = &iq1[h * dm..(h + 1) * dm];
                        for u in 0..32 {
                            let mut mp = 0f32;
                            for k in 0..4 {
                                let dv = row[u * 4 + k];
                                mp += dv * dv;
                            }
                            parts[u] = mp;
                        }
                        let scale = rms_scale(&parts);
                        for u in 0..32 {
                            for k in 0..4 {
                                iqr[h * dm + u * 4 + k] = row[u * 4 + k] * scale * iqw[u * 4 + k];
                            }
                        }
                        rope(&mut iqr[h * dm..][..dm], &cs[n_bulk * dm..]);
                    }
                }
                let mut scores = vec![0f32; n_blocks];
                for b in 0..n_blocks {
                    let mut sc = 0f32;
                    for h in 0..ih {
                        let (mut d0, mut d1, mut d2, mut d3) = (0f32, 0f32, 0f32, 0f32);
                        let qh = &iqr[h * dm..(h + 1) * dm];
                        let pk = &bk[b * dm..(b + 1) * dm];
                        let mut i2 = 0usize;
                        while i2 + 4 <= dm {
                            d0 += qh[i2] * pk[i2];
                            d1 += qh[i2 + 1] * pk[i2 + 1];
                            d2 += qh[i2 + 2] * pk[i2 + 2];
                            d3 += qh[i2 + 3] * pk[i2 + 3];
                            i2 += 4;
                        }
                        let dot = (d0 + d1) + (d2 + d3);
                        if dot > 0.0 {
                            sc += dot;
                        }
                    }
                    scores[b] = sc;
                }
                let tail_start = n_blocks * r;
                let tail_cnt = n_past - tail_start;
                let width = n_past.min(top_k + r - 1);
                let n_sel = ((width - tail_cnt) / r).min(n_blocks);
                let mut h_idx = Vec::with_capacity(n_sel * r + tail_cnt);
                let mut sel: Vec<usize> = (0..n_blocks)
                    .filter(|&b| {
                        let sb = scores[b];
                        (0..n_blocks)
                            .filter(|&b2| {
                                let s2 = scores[b2];
                                s2 > sb || (s2 == sb && b2 < b)
                            })
                            .count()
                            < n_sel
                    })
                    .collect();
                sel.sort_unstable();
                for &b in &sel {
                    for j in 0..r {
                        h_idx.push((b * r + j) as u32);
                    }
                }
                for j in 0..tail_cnt {
                    h_idx.push((tail_start + j) as u32);
                }
                let h_off = vec![0u32, (n_sel * r + tail_cnt) as u32];
                let dev_scores: Vec<f32> = {
                    let g = acc5.qsa_sel_bufs.lock();
                    let b = g.as_ref().unwrap();
                    (0..n_blocks)
                        .map(|i| unsafe { *(b.1.ptr.add(i * 4) as *const f32) })
                        .collect()
                };
                eprintln!("[qsel] host_scores={:?}", &scores[..n_blocks.min(8)]);
                eprintln!("[qsel] dev_scores={:?}", &dev_scores[..n_blocks.min(8)]);
                // (d) 대조 — 목록 전체 비트 일치.
                let (d_idx, d_off) = acc5.qsa_sel_readback(sd, od, list_len).map_err(|e| e.to_string())?;
                let ok = list_len == h_idx.len() && d_idx == h_idx && d_off == h_off;
                eprintln!(
                    "[qsel] n_blocks={n_blocks} n_sel={n_sel} list={list_len} host_sel={:?}",
                    sel
                );
                if !ok {
                    fails += 1;
                }
                report.push_str(&format!(
                    "| QsaSelDev list={list_len} {}",
                    if ok { "OK" } else { "FAIL" }
                ));
            }
            // ── 15) shexp_gu/shexp_da — 디코드 t=1 융합 vs CPU 디양자화 참조 ──
            // (qwen4exp 전용 — q35는 SKIP)
            if let AnyModel::Q4(m4) = &model {
                use llm170_core::matmul::EwOps as _;
                let il: usize = tname
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let wg = m4.w4(&format!("blk.{il}.ffn_gate_shexp.weight")).map_err(|e| e.to_string())?;
                let wu = m4.w4(&format!("blk.{il}.ffn_up_shexp.weight")).map_err(|e| e.to_string())?;
                let wd = m4.w4(&format!("blk.{il}.ffn_down_shexp.weight")).map_err(|e| e.to_string())?;
                let (n1, n2) = (wg.n_in as usize, wg.n_out as usize);
                let x: Vec<f32> = (0..n1).map(|_| lcg()).collect();
                let m0: Vec<f32> = (0..n1).map(|_| lcg()).collect();
                let s_val = 0.7f32;
                let xh = acc.frame_alloc(n1)?;
                let hh = acc.frame_alloc(n2)?;
                let mh = acc.frame_alloc(n1)?;
                let sh = acc.frame_alloc(1)?;
                acc.frame_write(xh, &x)?;
                acc.frame_write(mh, &m0)?;
                acc.frame_write(sh, &[s_val])?;
                acc.frame_begin(1); // AxpyT t=1 판
                acc.shexp_gu(xh, &wg, &wu, hh, n1, n2).map_err(|e| e.to_string())?;
                acc.shexp_da(hh, &wd, sh, mh, n1, n2).map_err(|e| e.to_string())?;
                let mut hgot = vec![0f32; n2];
                acc.frame_read(hh, &mut hgot)?;
                let mut mgot = vec![0f32; n1];
                acc.frame_read(mh, &mut mgot)?;
                // CPU 참조 — 디양자화 내적 + silu + sigmoid·axpy.
                let mut grow = vec![0f32; n1];
                let mut urow = vec![0f32; n1];
                let mut href = vec![0f32; n2];
                for m in 0..n2 {
                    llm170_core::quant::dequant_row(wg.ty, wg.data, m as u64, n1 as u64, &mut grow);
                    llm170_core::quant::dequant_row(wu.ty, wu.data, m as u64, n1 as u64, &mut urow);
                    let g: f32 = grow.iter().zip(&x).map(|(a, b)| a * b).sum();
                    let u: f32 = urow.iter().zip(&x).map(|(a, b)| a * b).sum();
                    href[m] = (g / (1.0 + (-g).exp())) * u;
                }
                let mut hmx = 0f64;
                for m in 0..n2.min(256) {
                    hmx = hmx.max((href[m] as f64 - hgot[m] as f64).abs());
                }
                let mut drow = vec![0f32; n2];
                let mut mmx = 0f64;
                for i in 0..n1.min(256) {
                    llm170_core::quant::dequant_row(wd.ty, wd.data, i as u64, n2 as u64, &mut drow);
                    let dh: f32 = drow.iter().zip(&href).map(|(a, b)| a * b).sum();
                    let mref = m0[i] + s_val * dh;
                    mmx = mmx.max((mref as f64 - mgot[i] as f64).abs());
                }
                eprintln!("[shexp] h max|D|={hmx:.3e} mout max|D|={mmx:.3e}");
                let ok = hmx < 5e-2 && mmx < 1e-1;
                if !ok {
                    fails += 1;
                }
                report.push_str(&format!(
                    "| Shexp h={hmx:.1e} mout={mmx:.1e} {}",
                    if ok { "OK" } else { "FAIL" }
                ));
                acc.frame_free(xh)?;
                acc.frame_free(hh)?;
                acc.frame_free(mh)?;
                acc.frame_free(sh)?;
            }
        }
        // ── 16) GdnConv 절대 대조(t=1 순차판) — CPU 링 산술과 직접 비교 ──
        // (plans/86 §1: §12는 청크 불변성만 — t<k-1 순차 커널은 커버 밖이었다)
        {
            use llm170_core::matmul::FrameHost as _FH3;
            let (ch, ck, steps) = (48usize, 4usize, 3usize);
            let mut s4 = 0x51ceu64;
            let mut lc4 = move || {
                s4 = s4.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s4 >> 33) as f32 / 2147483648.0 - 0.5
            };
            let cw: Vec<f32> = (0..ch * ck).map(|_| lc4()).collect();
            let mut st: Vec<f32> = (0..(ck - 1) * ch).map(|_| lc4()).collect();
            let qkv: Vec<f32> = (0..steps * ch).map(|_| lc4()).collect();
            let src = acc.frame_alloc(steps * ch)?;
            let cwb = acc.frame_alloc(ch * ck)?;
            let stb = acc.frame_alloc((ck - 1) * ch)?;
            let inb = acc.frame_alloc(ch)?;
            let ob = acc.frame_alloc(ch)?;
            acc.frame_write(src, &qkv)?;
            acc.frame_write(cwb, &cw)?;
            acc.frame_write(stb, &st)?;
            acc.frame_begin(1);
            let mut mo = 0f64;
            for t in 0..steps {
                acc.frame_op(&llm170_core::matmul::FrameOp::CopyRows {
                    src, dst: inb, src_off: t * ch, dst_off: 0, n: ch,
                })?;
                acc.frame_op(&llm170_core::matmul::FrameOp::GdnConv {
                    qkv: inb, cw: cwb, state: stb, out: ob, ch, k: ck, t_len: 1,
                })?;
                let mut got = vec![0f32; ch];
                acc.frame_read(ob, &mut got)?;
                // CPU 참조 — stages/gdn.rs conv 산술 동일열(상태도 진화).
                for c in 0..ch {
                    let mut sum = cw[c * ck + (ck - 1)] * qkv[t * ch + c];
                    for j in 0..ck - 1 {
                        sum += cw[c * ck + j] * st[j * ch + c];
                    }
                    let out_c = sum / (1.0 + (-sum).exp());
                    for j in 0..ck - 2 {
                        st[j * ch + c] = st[(j + 1) * ch + c];
                    }
                    st[(ck - 2) * ch + c] = qkv[t * ch + c];
                    mo = mo.max((got[c] as f64 - out_c as f64).abs());
                }
            }
            let mut stf = vec![0f32; (ck - 1) * ch];
            acc.frame_read(stb, &mut stf)?;
            let mut ms = 0f64;
            for i in 0..st.len() {
                ms = ms.max((stf[i] as f64 - st[i] as f64).abs());
            }
            eprintln!("[gcvabs] out={mo:.1e} st={ms:.1e}");
            let ok = mo < 1e-6 && ms < 1e-6;
            if !ok { fails += 1; }
            report.push_str(&format!("| GdnConvT1 out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [src, cwb, stb, inb, ob] { acc.frame_free(h)?; }
        }
        // ── 17) GDN AR 절대 대조(t=1) — 전치 상태 + CPU 미러 ──
        {
            use llm170_core::matmul::FrameState as _FS3;
            let (hk, hv, d) = (2usize, 4usize, 128usize);
            let (ks, vs) = (hk * d, hv * d);
            let mut s5 = 0x600du64;
            let mut lc5 = move || {
                s5 = s5.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s5 >> 33) as f32 / 2147483648.0 - 0.5
            };
            let scale = 1.0f32 / (d as f32).sqrt();
            let qs: Vec<f32> = (0..ks).map(|_| lc5() * scale).collect();
            let kk: Vec<f32> = (0..ks).map(|_| lc5()).collect();
            let vv: Vec<f32> = (0..vs).map(|_| lc5()).collect();
            let beta: Vec<f32> = (0..hv).map(|_| 0.4 + 0.4 * lc5()).collect();
            let g: Vec<f32> = (0..hv).map(|_| lc5() * 0.2).collect();
            let mut bg = vec![0f32; hv * 2];
            for h in 0..hv {
                bg[h * 2] = beta[h];
                bg[h * 2 + 1] = g[h].exp();
            }
            let mut st = vec![0f32; hv * d * d];
            for x in st.iter_mut() { *x = lc5() * 0.05; }
            // CPU 미러(사전 스케일 q) — gdn.rs gdn_ar_batch 산술 동일열.
            let st_in = st.clone();
            let mut st_cpu = st_in.clone();
            let mut o_cpu = vec![0f32; vs];
            for h in 0..hv {
                let kh = h % hk;
                let (qb, kb, vb) = (&qs[kh * d..kh * d + d], &kk[kh * d..kh * d + d], &vv[h * d..h * d + d]);
                let s = &mut st_cpu[h * d * d..(h + 1) * d * d];
                let mut sk = vec![0f32; d];
                for kdim in 0..d {
                    for dv in 0..d {
                        let e = &mut s[kdim * d + dv];
                        *e *= bg[h * 2 + 1];
                        sk[dv] += *e * kb[kdim];
                    }
                }
                for dv in 0..d {
                    let delta = (vb[dv] - sk[dv]) * beta[h];
                    for kdim in 0..d {
                        s[kdim * d + dv] += kb[kdim] * delta;
                    }
                }
                for dv in 0..d {
                    let mut o = 0f32;
                    for kdim in 0..d {
                        o += s[kdim * d + dv] * qb[kdim];
                    }
                    o_cpu[h * d + dv] = o;
                }
            }
            // 디바이스: 전치 상태 업로드 → AR → 판독 역전치.
            let tr = |v: &[f32]| -> Vec<f32> {
                let mut o = vec![0f32; v.len()];
                for (cb, b) in v.chunks(d * d).enumerate() {
                    let base = cb * d * d;
                    for kd in 0..d {
                        for dv in 0..d {
                            o[base + dv * d + kd] = b[kd * d + dv];
                        }
                    }
                }
                o
            };
            let qh = acc.frame_alloc(ks)?;
            let kh2 = acc.frame_alloc(ks)?;
            let vh = acc.frame_alloc(vs)?;
            let bh = acc.frame_alloc(hv * 2)?;
            let sh = acc.frame_alloc(hv * d * d)?;
            let oh = acc.frame_alloc(vs)?;
            acc.frame_write(qh, &qs)?;
            acc.frame_write(kh2, &kk)?;
            acc.frame_write(vh, &vv)?;
            acc.frame_write(bh, &bg)?;
            acc.frame_write(sh, &tr(&st_in))?;
            acc.frame_begin(1);
            acc.frame_gdn_ar(qh, kh2, vh, bh, sh, oh, 1, hk, hv, d)?;
            let mut og = vec![0f32; vs];
            acc.frame_read(oh, &mut og)?;
            let mut sg = vec![0f32; hv * d * d];
            acc.frame_read(sh, &mut sg)?;
            let sg = tr(&sg); // 역전치 — CPU 레이아웃으로
            let mut mo = 0f64;
            let mut ms = 0f64;
            for i in 0..vs { mo = mo.max((og[i] as f64 - o_cpu[i] as f64).abs()); }
            for i in 0..st_cpu.len() { ms = ms.max((sg[i] as f64 - st_cpu[i] as f64).abs()); }
            eprintln!("[gnarabs] out={mo:.1e} st={ms:.1e}");
            let ok = mo < 1e-4 && ms < 1e-4;
            if !ok { fails += 1; }
            report.push_str(&format!("| GdnART1 out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [qh, kh2, vh, bh, sh, oh] { acc.frame_free(h)?; }
        }
        // ── 18) 헤드 체인 절대 대조(t=1) — 실가중 output_hc + output GEMM ──
        if let AnyModel::Q4(m4) = &model {
            use llm170_core::matmul::FrameHost as _FH4;
            let hp = &m4.hp;
            let (n, hc) = (hp.n_embd, hp.hc);
            let w_norm = m4.f32_vec4("output_hc_norm.weight").map_err(|e| e.to_string())?;
            let w_down = m4.w4("output_hc_down.weight").map_err(|e| e.to_string())?;
            let w_up = m4.w4("output_hc_up.weight").map_err(|e| e.to_string())?;
            let w_out = m4.w4("output.weight").map_err(|e| e.to_string())?;
            let mut s6 = 0x7a11u64;
            let mut lc6 = move || {
                s6 = s6.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s6 >> 33) as f32 / 2147483648.0 - 0.5
            };
            // res: 모든 스트림 동일(엔진 임베딩 방송을 미러) — norm 검증에 충분.
            let row: Vec<f32> = (0..n).map(|_| lc6()).collect();
            let mut res = vec![0f32; hc * n];
            for s in 0..hc {
                res[s * n..(s + 1) * n].copy_from_slice(&row);
            }
            let wn = acc.frame_alloc(hc * n)?;
            acc.frame_write(wn, &w_norm)?;
            let rh = acc.frame_alloc(hc * n)?;
            let xnh = acc.frame_alloc(hc * n)?;
            let loh = acc.frame_alloc(w_down.n_out as usize)?;
            let gah = acc.frame_alloc(hc * n)?;
            let hih = acc.frame_alloc(n)?;
            let lgh = acc.frame_alloc(16)?;
            acc.frame_write(rh, &res)?;
            acc.frame_begin(1);
            acc.frame_op(&llm170_core::matmul::FrameOp::RmsRows {
                x: rh, w: wn, out: xnh, eps: hp.eps, n, w_reps: hc,
            })?;
            acc.frame_mm(xnh, &w_down, loh, 1)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::SiluDiv {
                t: loh, div: hc as f32, n: w_down.n_out as usize,
            })?;
            acc.frame_mm(loh, &w_up, gah, 1)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::HcGateMean {
                xn: xnh, gate: gah, out: hih, hc, n,
            })?;
            acc.frame_mm(hih, &w_out, lgh, 1)?;
            let mut lg = vec![0f32; 16];
            acc.frame_read(lgh, &mut lg)?;
            // CPU 참조 — forward.rs 헤드 산술 동일열.
            let mut hxn = vec![0f32; hc * n];
            for s in 0..hc {
                let nn = llm170_core::ops::rms_norm(&row, &w_norm[s * n..(s + 1) * n], hp.eps);
                hxn[s * n..(s + 1) * n].copy_from_slice(&nn);
            }
            let mut hlo = vec![0f32; w_down.n_out as usize];
            llm170_core::matmul::matmul(&hxn, &w_down, &mut hlo);
            for v in hlo.iter_mut() { *v = llm170_core::ops::silu(*v / hc as f32); }
            let mut hgate = vec![0f32; hc * n];
            llm170_core::matmul::matmul(&hlo, &w_up, &mut hgate);
            let mut hin = vec![0f32; n];
            for i in 0..n {
                let mut m = 0f32;
                for s in 0..hc {
                    let k = s * n + i;
                    m += hxn[k] * (1.0 / (1.0 + (-hgate[k]).exp()));
                }
                hin[i] = m / hc as f32;
            }
            let mut hlg = vec![0f32; 16];
            llm170_core::matmul::matmul(&hin, &w_out, &mut hlg);
            let mut mx = 0f64;
            for i in 0..16 { mx = mx.max((lg[i] as f64 - hlg[i] as f64).abs()); }
            let scale = hlg.iter().fold(0f32, |a, &v| a.max(v.abs())) as f64;
            eprintln!("[headabs] max|D|={mx:.3e} (scale={scale:.1})");
            // 3연속 W4A8 GEMM + silu 증폭 — logit-diff.sh 의 MMA 클래스(maxrel<3e-2)와 동일 기준.
            let ok = mx / scale.max(1.0) < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| HeadChain {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [wn, rh, xnh, loh, gah, hih, lgh] { acc.frame_free(h)?; }
        }
    }
    let _ = t0;
    Ok(format!(
        "vk-frame-check({tname}, t={t}): {} — {} ({} 실패)",
        if fails == 0 { "PASS" } else { "FAIL" },
        report,
        fails
    ))
}

/// plans/87 §1 — 의도적 GPUVM 폴트 프로브: 실제 결함 패턴(디스크립터
/// 오프셋이 버퍼 끝 너머 — pipeline robustness가 주소 자체를 못 구한다)으로
/// 폴트를 유발해 RADV 주소 → va-lookup 체인을 검증한다. DEVICE_LOST가 정상.
pub fn fault_probe() -> Result<String, String> {
    let acc = VkAcc::new()?;
    llm170_diag::alloc::set_on(true);
    llm170_diag::alloc::set_vaddr(true);
    let mut ctx = acc.ctx.lock();
    let b = ctx.alloc_host(4096)?; // 원장 기록(VA 포함)
    let p = acc.pipeline(&mut ctx, Slot::Scale)?;
    let ds = ctx.fresh_ds_for(&p, 1)?;
    // 실효 패턴: 12-바인딩 gemv 파이프라인에 1개만 바인딩 — 미바인딩
    // 디스크립터(3..11)를 커널이 읽는다. 오프셋 초과는 RADV가 빈 범위로
    // 클램프해 폴트가 안 나는 것을 실측했다(정렬 무관).
    let _ = ds;
    let ds2 = ctx.bind_ds(&p, &[b.buf])?;
    let push = push_u32s(&[32u32, 32u32, 8u32, 8u32, 1u32, 1024u32]);
    let r = ctx.run(p.pl, ds2, p.pipe, &push, 1, 1, 1);
    let tsv = llm170_diag::alloc::tsv_path().unwrap_or_else(|| "(없음)".into());
    Ok(format!(
        "발사 결과: {r:?} (Err=DEVICE_LOST 정상) — tsv: {tsv} 에서 RADV 폴트 주소를 va-lookup 하라"
    ))
}

/// 프레임 버퍼 원시 포인터(프로브 내부용).
fn acc_frame_ptr(acc: &VkAcc, h: u64) -> *mut u8 {
    acc.framebufs.lock().get(&h).map(|b| b.ptr).unwrap_or(std::ptr::null_mut())
}
