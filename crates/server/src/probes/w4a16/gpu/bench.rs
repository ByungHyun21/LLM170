
pub(super) fn run_h2d_bench(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    h2d_mb: usize,
) -> Result<String, String> {
    let bytes = h2d_mb << 20;
    let src = vec![0x5Au8; bytes];
    let dst = dec.alloc_scratch(bytes)?;
    let pinned = dec.alloc_pinned_scratch(bytes)?;
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), pinned as *mut u8, bytes);
    }
    let t = std::time::Instant::now();
    dec.h2d_scratch(dst, &src)?;
    dec.sync_bench()?;
    let pageable_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let ps = unsafe { std::slice::from_raw_parts(pinned as *const u8, bytes) };
    dec.h2d_scratch(dst, ps)?;
    dec.sync_bench()?;
    let pinned_ms = t.elapsed().as_secs_f64() * 1e3;
    dec.free_pinned_scratch(pinned)?;
    dec.free_scratch(dst)?;
    Ok(format!(
        "h2d-bench {h2d_mb}MiB: 페이지러블 {pageable_ms:.0}ms ({:.1}GB/s) · 핀드 {pinned_ms:.0}ms ({:.1}GB/s)",
        bytes as f64 * 1e-9 / (pageable_ms * 1e-3),
        bytes as f64 * 1e-9 / (pinned_ms * 1e-3).max(1e-9)
    ))
}

pub(super) fn run_mma_diff(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    name: &str,
    t: usize,
) -> Result<String, String> {
    dec.mma_diff_check(name, t)
}

pub(super) fn run_marlin_diff(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    t: usize,
) -> Result<String, String> {
    let w = model
        .w_raw(name)
        .ok_or_else(|| format!("--marlin-diff {name}: 무게 없음"))?;
    if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
        return Err(format!(
            "--marlin-diff {name}: g128·f16 split 전용(ty={:?} group={} bf16={})",
            w.ty, w.group, w.scale_bf16
        ));
    }
    let q: Vec<u32> = w
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let qm = llm170_core::marlin::marlin_repack_q(&q, w.n_out as usize, w.n_in as usize)?;
    dec.marlin_diff_check(name, t, &qm)
}

pub(super) fn run_decode_bench(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    bench: &[(String, usize)],
    cold: bool,
) -> Result<String, String> {
    let mut out = Vec::new();
    for (name, reps) in bench {
        out.push(dec.decode_bench(name, *reps, cold)?);
    }
    Ok(out.join("\n"))
}

pub(super) fn run_marlin_selfcheck(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    (t, k, n, m16): (usize, usize, usize, bool),
) -> Result<String, String> {
    let nib_at = |kk: usize, nn: usize| ((kk * 7 + nn * 3) % 16) as u32;
    let mut q = vec![0u32; n * (k / 8)];
    for nn in 0..n {
        for kk in 0..k {
            q[nn * (k / 8) + kk / 8] |= nib_at(kk, nn) << ((kk % 8) * 4);
        }
    }
    let mut s = vec![0u16; n * (k / 128)];
    for nn in 0..n {
        for gg in 0..(k / 128) {
            s[nn * (k / 128) + gg] = ((15 - ((nn + gg) % 5)) as u16) << 10;
        }
    }
    let x16: Vec<u16> = (0..t * k)
        .map(|i| ((15 - ((i / k + (i % k) / 8) % 4)) as u16) << 10)
        .collect();
    let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
    let sf = |h: u16| -> f64 {
        let e = ((h >> 10) & 0x1F) as i32;
        let m = (h & 0x3FF) as f64 / 1024.0;
        (1.0 + m) * 2f64.powi(e - 15)
    };
    let mut expect = vec![0f32; t * n];
    for r in 0..t {
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let sc = sf(s[nn * (k / 128) + kk / 128]);
                let xv = sf(x16[r * k + kk]);
                acc += (nib_at(kk, nn) as f64 - 8.0) * sc * xv;
            }
            expect[r * n + nn] = acc as f32;
        }
    }
    let (sym, bm, bn, thr) = if m16 {
        ("w4a16_gemm_g128_marlin_m16", 16usize, 256usize, 128u32)
    } else {
        ("w4a16_gemm_g128_marlin", 64, 128, 256)
    };
    dec.marlin_selfcheck(sym, bm, bn, thr, &qm, &s, &x16, &expect, t, n, k)
}

pub(super) fn run_marlin_selfcheck32(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    (t, k, n): (usize, usize, usize),
) -> Result<String, String> {
    let nib_at = |kk: usize, nn: usize| ((kk * 7 + nn * 3) % 16) as u32;
    let mut q = vec![0u32; n * (k / 8)];
    for nn in 0..n {
        for kk in 0..k {
            q[nn * (k / 8) + kk / 8] |= nib_at(kk, nn) << ((kk % 8) * 4);
        }
    }
    let mut s = vec![0u16; n * (k / 32)];
    for nn in 0..n {
        for gg in 0..(k / 32) {
            s[nn * (k / 32) + gg] = ((127 - ((nn + gg) % 5)) as u16) << 7;
        }
    }
    let x16 = vec![0x3C00u16; t * k];
    let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
    let mut expect = vec![0f32; t * n];
    for r in 0..t {
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let j = ((nn + kk / 32) % 5) as i32;
                acc += (nib_at(kk, nn) as f64 - 8.0) * 2f64.powi(-j);
            }
            expect[r * n + nn] = acc as f32;
        }
    }
    dec.marlin_selfcheck(
        "w4a16_gemm_g32_marlin",
        64,
        128,
        256,
        &qm,
        &s,
        &x16,
        &expect,
        t,
        n,
        k,
    )
}

pub(super) fn run_attn_fa2_bench(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    bench: &[(usize, usize, usize, u32)],
) -> Result<String, String> {
    let mut out = Vec::new();
    for (t, lay, reps, pos0) in bench {
        out.push(dec.attn_fa2_bench(0, *lay, *t, *reps, *pos0)?);
    }
    Ok(out.join("\n"))
}

pub(super) fn run_attn_fa2(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    diffs: &[(usize, usize, u32)],
) -> Result<String, String> {
    let mut out = Vec::new();
    for (t, lay, p0) in diffs {
        out.push(dec.attn_fa2_diff(0, *lay, *t, *p0)?);
    }
    Ok(out.join("\n"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_marlin_ref(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    t: usize,
    mode: u8,
    kcap: usize,
) -> Result<String, String> {
    let w = model
        .w_raw(name)
        .ok_or_else(|| format!("--marlin-ref {name}: 무게 없음"))?;
    if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
        return Err(format!("--marlin-ref {name}: g128·f16 split 전용"));
    }
    let (n, k) = (64usize, kcap.min(w.n_in as usize));
    let q: Vec<u32> = w.data[..n * (k / 8) * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let s16: Vec<u16> = w.aux.ok_or("aux 부재")?[..n * (k / 128) * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
    let x16: Vec<u16> = (0..t * k)
        .map(|i| {
            let v = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
            llm170_backend_gpu::f32_to_f16(v)
        })
        .collect();
    let mut s_use = s16.clone();
    let mut x_use = x16.clone();
    if mode == 1 || mode == 3 {
        s_use.fill(0x3C00);
    }
    if mode == 2 || mode == 3 {
        x_use.fill(0x3C00);
    }
    let mut expect = vec![0f32; t * n];
    for r in 0..t {
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let nib = ((q[nn * (k / 8) + kk / 8] >> ((kk % 8) * 4)) & 0xF) as i32 - 8;
                let sc = llm170_core::quant::half_to_f32(s_use[nn * (k / 128) + kk / 128]);
                let wf = llm170_backend_gpu::h2f(llm170_backend_gpu::f32_to_f16(nib as f32 * sc));
                acc += (wf as f64) * (llm170_backend_gpu::h2f(x_use[r * k + kk]) as f64);
            }
            expect[r * n + nn] = acc as f32;
        }
    }
    dec.marlin_selfcheck(
        "w4a16_gemm_g128_marlin",
        64,
        128,
        256,
        &qm,
        &s_use,
        &x_use,
        &expect,
        t,
        n,
        k,
    )
}

pub(super) fn run_marlin_bench(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    bench: &[(String, usize, usize, String)],
    cold: bool,
) -> Result<String, String> {
    let mut out = Vec::new();
    for (name, t, reps, kern) in bench {
        let Some(w) = model.w_raw(name) else {
            out.push(format!("marlin-bench {name}: 무게 없음(건너뜀)"));
            continue;
        };
        if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
            out.push(format!(
                "marlin-bench {name}: g128·f16 split 아님(건너뜀: ty={:?} group={} bf16={})",
                w.ty, w.group, w.scale_bf16
            ));
            continue;
        }
        let q: Vec<u32> = w
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let qm = llm170_core::marlin::marlin_repack_q(&q, w.n_out as usize, w.n_in as usize)?;
        let (msym, mbm, mbn, mthr, vs_gemv) = match kern.as_str() {
            "m16v" => (
                "w4a16_gemm_g128_marlin_m16",
                16usize,
                256usize,
                128u32,
                true,
            ),
            "m64v" => ("w4a16_gemm_g128_marlin", 64usize, 128usize, 256u32, true),
            "m16" => (
                "w4a16_gemm_g128_marlin_m16",
                16usize,
                256usize,
                128u32,
                false,
            ),
            _ => ("w4a16_gemm_g128_marlin", 64, 128, 256, false),
        };
        out.push(dec.marlin_bench(name, *t, *reps, &qm, msym, mbm, mbn, mthr, vs_gemv, cold)?);
    }
    Ok(out.join("\n"))
}

pub(super) fn run_bench_plain(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    bt: usize,
    reps: usize,
) -> Result<String, String> {
    let (n, k) = model
        .w_raw(name)
        .map(|w| (w.n_out as usize, w.n_in as usize))
        .ok_or_else(|| format!("--bench-plain {name}: 무게 없음"))?;
    let xf: Vec<f32> = vec![1.0f32; bt * k];
    let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
    let dx = dec.alloc_scratch(xf.len() * 4)?;
    dec.h2d_scratch(dx, xb)?;
    let dy = dec.alloc_scratch(bt * n * 4)?;
    dec.plain_bench_launch(name, dx, dy, bt)?;
    dec.sync_bench()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dec.plain_bench_launch(name, dx, dy, bt)?;
    }
    dec.sync_bench()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let flop = 2.0 * (n as f64) * (k as f64) * (bt as f64);
    dec.free_scratch(dx)?;
    dec.free_scratch(dy)?;
    Ok(format!(
        "bench-plain {name} n={n} k={k} t={bt}: {ms:.3} ms/회 · {:.2} TF",
        flop * 1e-9 / (ms * 1e-3)
    ))
}

pub(super) fn run_bench_gemv_t(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    name: &str,
    t: usize,
    reps: usize,
) -> Result<String, String> {
    let ms = dec.bench_gemv_t(name, t, reps)?;
    Ok(format!("bench-gemv-t {name} t={t}: {ms:.3} ms/회"))
}

pub(super) fn run_bench_gemv_all(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    reps: usize,
) -> Result<String, String> {
    let (ms, wb) = dec.gemv_walk_bench(reps)?;
    let gbs = wb as f64 * 1e-9 / (ms * 1e-3);
    Ok(format!(
        "bench-gemv-all: {ms:.2} ms/회 · 가중치 {gbs:.0} GB/s ({:.2} GB)",
        wb as f64 / 1e9
    ))
}

pub(super) fn run_bench_gemv(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    reps: usize,
) -> Result<String, String> {
    let (n, k) = model
        .w_raw(name)
        .map(|w| (w.n_out as usize, w.n_in as usize))
        .ok_or_else(|| format!("--bench-gemv {name}: 무게 없음"))?;
    let xf: Vec<f32> = vec![1.0f32; k];
    let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
    let dx = dec.alloc_scratch(xf.len() * 4)?;
    dec.h2d_scratch(dx, xb)?;
    let dy = dec.alloc_scratch(n * 4)?;
    dec.gemv_bench_launch(name, dx, dy)?;
    dec.sync_bench()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dec.gemv_bench_launch(name, dx, dy)?;
    }
    dec.sync_bench()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let wb = if model.w_raw(name).map(|w| w.ty) == Some(llm170_core::wtype::WType::W4a16Split) {
        (n * k) as f64 / 2.0
    } else {
        (n * k) as f64 * 2.0
    };
    let gbs = wb * 1e-9 / (ms * 1e-3);
    dec.free_scratch(dx)?;
    dec.free_scratch(dy)?;
    Ok(format!(
        "bench-gemv {name} n={n} k={k}: {ms:.3} ms/회 · 가중치 {gbs:.0} GB/s"
    ))
}

pub(super) fn run_bench_gemm(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    bt: usize,
    reps: usize,
) -> Result<String, String> {
    let (n, k) = model
        .w_raw(name)
        .map(|w| (w.n_out as usize, w.n_in as usize))
        .ok_or_else(|| format!("--bench-gemm {name}: 무게 없음"))?;
    let xf: Vec<f32> = vec![1.0f32; bt * k];
    let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
    let dxh = dec.alloc_scratch(xf.len() * 4)?;
    dec.h2d_scratch(dxh, xb)?;
    let dout = dec.alloc_scratch(bt * n * 4)?;
    dec.gemm_bench_launch(name, dxh, dout, bt)?;
    dec.sync_bench()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dec.gemm_bench_launch(name, dxh, dout, bt)?;
    }
    dec.sync_bench()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let wb = (n * k) as f64 / 2.0;
    let gbs = wb * 1e-9 / (ms * 1e-3);
    dec.free_scratch(dxh)?;
    dec.free_scratch(dout)?;
    Ok(format!(
        "bench-gemm {name} n={n} k={k} t={bt} reps={reps}: {ms:.2} ms/회 · 가중치 {gbs:.0} GB/s"
    ))
}
