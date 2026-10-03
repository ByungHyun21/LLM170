use crate::rawhip::ctx::RawCtx as HipCtx;

// ── EXL3 hip GEMV 체인 프로브(plans/121 CMP 포팅 · todo 2/4) ──
// vk 가중치를 그대로 투입해 hipRTC 컴파일 exl3_had_in→gemv→had_out 체인을
// 실행, vk 트레이리던트 참조(tr.linear)와 대조 — 8060S hipRTC로 검증.
pub fn hip_gemv_check(dir: &str) -> Result<String, String> {
    let mut tr = TrellisResident::load(dir)?;
    let key = "lm_head";
    let (k, n, krate, suh, tre, svh) = tr.linear_raw(key)?;
    let x: Vec<f32> = {
        let mut seed: u32 = 0x5EED_00F1;
        (0..k)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2
            })
            .collect()
    };
    let want = tr.linear(key, &x)?;
    drop(tr);

    let hc = HipCtx::new()?;
    let nseg = 16usize;
    let dx = hc.alloc(k * 4)?;
    let dsuh = hc.alloc(suh.len())?;
    let dtre = hc.alloc(tre.len())?;
    let dsvh = hc.alloc(svh.len())?;
    let dah = hc.alloc(k * 2)?;
    let dsb = hc.alloc(nseg * n * 4)?;
    let dy = hc.alloc(n * 4)?;
    let xb: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
    hc.h2d(dx, xb)?;
    hc.h2d(dsuh, &suh)?;
    hc.h2d(dtre, &tre)?;
    hc.h2d(dsvh, &svh)?;

    let (mut kc, mut ks) = ((k / 128) as i32, k as i32);
    let (mut a0, mut a1, mut a2) = (dx, dsuh, dah);
    hc.launch(
        "exl3_had_in",
        (k / 128) as u32,
        1,
        128,
        &mut [
            &mut a0 as *mut *mut u8 as *mut _,
            &mut a1 as *mut *mut u8 as *mut _,
            &mut a2 as *mut *mut u8 as *mut _,
            &mut kc as *mut i32 as *mut _,
            &mut ks as *mut i32 as *mut _,
        ],
    )?;
    // had_in은 그리드 (kchunks, T=1) — gy=1 행.
    let (mut kt, mut nt, mut kk) = ((k / 16) as i32, (n / 16) as i32, krate as i32);
    let (mut b0, mut b1, mut b2) = (dah, dtre, dsb);
    hc.launch(
        "exl3_gemv",
        ((n / 16) / 8) as u32,
        nseg as u32,
        128,
        &mut [
            &mut b0 as *mut *mut u8 as *mut _,
            &mut b1 as *mut *mut u8 as *mut _,
            &mut b2 as *mut *mut u8 as *mut _,
            &mut kt as *mut i32 as *mut _,
            &mut nt as *mut i32 as *mut _,
            &mut kk as *mut i32 as *mut _,
        ],
    )?;
    let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, nseg as i32, n as i32);
    let (mut c0, mut c1, mut c2) = (dsb, dsvh, dy);
    hc.launch(
        "exl3_had_out",
        (n / 128) as u32,
        1,
        128,
        &mut [
            &mut c0 as *mut *mut u8 as *mut _,
            &mut c1 as *mut *mut u8 as *mut _,
            &mut c2 as *mut *mut u8 as *mut _,
            &mut nch as *mut i32 as *mut _,
            &mut nsg as *mut i32 as *mut _,
            &mut nst as *mut i32 as *mut _,
        ],
    )?;
    let mut yb = vec![0u8; n * 4];
    hc.d2h(&mut yb, dy)?;
    hc.sync()?;
    // had_in 산출 대조 — Rust WHT 미러(f16 RTNE, nrh_check 동일 산술).
    {
        let suh_f: Vec<f32> = (0..k)
            .map(|kk| {
                let b = (kk / 2) * 4;
                let w32 = u32::from_le_bytes([suh[b], suh[b + 1], suh[b + 2], suh[b + 3]]);
                let h = ((w32 >> ((kk & 1) * 16)) & 0xFFFF) as u16;
                half::f16::from_bits(h).to_f32()
            })
            .collect();
        let mut want = vec![0u8; k * 2];
        for ch in 0..k / 128 {
            let mut sm = [0f32; 128];
            for i in 0..128 {
                let pre = half::f16::from_f32(x[ch * 128 + i] * suh_f[ch * 128 + i]);
                sm[i] = pre.to_f32();
            }
            let mut w = 1usize;
            while w < 128 {
                let mut i = 0;
                while i < 128 {
                    let blk = (i / (2 * w)) * (2 * w);
                    for j in 0..w {
                        let a = sm[blk + j];
                        let b = sm[blk + j + w];
                        sm[blk + j] = a + b;
                        sm[blk + j + w] = a - b;
                    }
                    i += 2 * w;
                }
                w *= 2;
            }
            for i in (0..128).step_by(2) {
                let lo = half::f16::from_f32(sm[i] * 0.08838834764831845);
                let hi = half::f16::from_f32(sm[i + 1] * 0.08838834764831845);
                let pack = (lo.to_bits() as u32) | ((hi.to_bits() as u32) << 16);
                want[(ch * 128 + i) * 2..(ch * 128 + i) * 2 + 4]
                    .copy_from_slice(&pack.to_le_bytes());
            }
        }
        let mut ahb = vec![0u8; k * 2];
        hc.d2h(&mut ahb, dah)?;
        hc.sync()?;
        let mut bitdiff = 0usize;
        for i in 0..k * 2 {
            if ahb[i] != want[i] {
                bitdiff += 1;
            }
        }
        eprintln!("  [hipdbg] had_in 바이트 불일치 {bitdiff}/{}", k * 2);
    }
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] = unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, n) };
    let mut md = 0f32;
    let mut nan = 0usize;
    for i in 0..n {
        if !got[i].is_finite() {
            nan += 1;
            continue;
        }
        md = md.max((got[i] - want[i]).abs());
    }
    // 속도(5회 중앙): 체인 재실행(입력 동일)
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        hc.launch(
            "exl3_gemv",
            ((n / 16) / 8) as u32,
            nseg as u32,
            128,
            &mut [
                &mut b0 as *mut *mut u8 as *mut _,
                &mut b1 as *mut *mut u8 as *mut _,
                &mut b2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
            ],
        )?;
        hc.sync()?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bytes = ((k / 16) * (n / 16) * (8 * krate as usize) * 4 + k * 2) as f64 / 1e9;
    Ok(format!(
        "hip-gemv lm_head k={k} n={n}: maxdiff={md:.3e} nan={nan} · gemv {bytes:.3}GB → {:.2}ms = {:.0}GB/s",
        ts[2],
        bytes / (ts[2] / 1000.0)
    ))
}
// 마커 hippr1

// ── EXL3 hip norm_resid 격리 프로브(모듈 2/4) ── vk nr 산술과 동일 입력 대조.
pub fn hip_nr_check(dir: &str) -> Result<String, String> {
    use crate::rawvk::checks::TrellisResident;
    let t_rows = 4usize;
    let mut tr = TrellisResident::load(dir)?;
    tr.fframe_init()?;
    let nw = tr.nw128_dump()?;
    drop(tr);
    let mut seed: u32 = 0x1234_ABCD;
    let rnd = |s: &mut u32| {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
    };
    let (x, ab): (Vec<f32>, Vec<f32>) = (
        (0..t_rows * 5120).map(|_| rnd(&mut seed)).collect(),
        (0..t_rows * 5120).map(|_| rnd(&mut seed)).collect(),
    );
    let hc = HipCtx::new()?;
    let dx = hc.alloc(t_rows * 5120 * 4)?;
    let dab = hc.alloc(t_rows * 5120 * 4)?;
    let dnw = hc.alloc(5120 * 4)?;
    let dxn = hc.alloc(t_rows * 5120 * 4)?;
    let xb: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, t_rows * 5120 * 4) };
    let abb: &[u8] =
        unsafe { std::slice::from_raw_parts(ab.as_ptr() as *const u8, t_rows * 5120 * 4) };
    let nwb: &[u8] = unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, 5120 * 4) };
    hc.h2d(dx, xb)?;
    hc.h2d(dab, abb)?;
    hc.h2d(dnw, nwb)?;
    // 시그니처 순서: (x, nw, ab, xn, t_len, w_off)
    let (mut tl, mut wo) = (t_rows as i32, 0i32);
    let (mut p_x, mut p_nw, mut p_ab, mut p_xn) = (dx, dnw, dab, dxn);
    hc.launch(
        "exl3_norm_resid",
        t_rows as u32,
        1,
        1024,
        &mut [
            &mut p_x as *mut *mut u8 as *mut _,
            &mut p_nw as *mut *mut u8 as *mut _,
            &mut p_ab as *mut *mut u8 as *mut _,
            &mut p_xn as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut wo as *mut i32 as *mut _,
        ],
    )?;
    let mut outb = vec![0u8; t_rows * 5120 * 4];
    hc.d2h(&mut outb, dxn)?;
    hc.sync()?;
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] =
        unsafe { std::slice::from_raw_parts(outb.as_ptr() as *const f32, t_rows * 5120) };
    let mut want_xn = vec![0f32; t_rows * 5120];
    for t in 0..t_rows {
        let ss: f32 = (0..5120)
            .map(|i| {
                let v = x[t * 5120 + i] + ab[t * 5120 + i];
                v * v
            })
            .sum();
        let inv = 1.0 / (ss / 5120.0 + 1e-6).sqrt();
        for i in 0..5120 {
            want_xn[t * 5120 + i] = (x[t * 5120 + i] + ab[t * 5120 + i]) * inv * nw[i];
        }
    }
    let mut md = 0f32;
    for i in 0..t_rows * 5120 {
        md = md.max((got[i] - want_xn[i]).abs());
    }
    Ok(format!("hip-nr T={t_rows}: xn maxdiff={md:.3e}"))
}
// 마커 nrhip

// ── EXL3 hip 실선형 체인 프로브(모듈 3/4) ── 실제 가중치로 had_in→gemv→had_out.
pub fn hip_linear_check(dir: &str, key: &str) -> Result<String, String> {
    use crate::rawvk::checks::TrellisResident;
    let mut tr = TrellisResident::load(dir)?;
    let (k, n, krate, suh, tre, svh) = tr.linear_raw(key)?;
    let x: Vec<f32> = {
        let mut seed: u32 = 0x77AA_0011;
        (0..k)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.3
            })
            .collect()
    };
    let want = tr.linear(key, &x)?;
    drop(tr);
    let hc = HipCtx::new()?;
    let nseg = 16usize;
    let dx = hc.alloc(k * 4)?;
    let dsuh = hc.alloc(suh.len())?;
    let dtre = hc.alloc(tre.len())?;
    let dsvh = hc.alloc(svh.len())?;
    let dah = hc.alloc(k * 2)?;
    let dsb = hc.alloc(nseg * n * 4)?;
    let dy = hc.alloc(n * 4)?;
    let xb: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
    hc.h2d(dx, xb)?;
    hc.h2d(dsuh, &suh)?;
    hc.h2d(dtre, &tre)?;
    hc.h2d(dsvh, &svh)?;
    let (mut kc, mut ks) = ((k / 128) as i32, k as i32);
    let (mut p0, mut p1, mut p2) = (dx, dsuh, dah);
    hc.launch(
        "exl3_had_in",
        (k / 128) as u32,
        1,
        128,
        &mut [
            &mut p0 as *mut *mut u8 as *mut _,
            &mut p1 as *mut *mut u8 as *mut _,
            &mut p2 as *mut *mut u8 as *mut _,
            &mut kc as *mut i32 as *mut _,
            &mut ks as *mut i32 as *mut _,
        ],
    )?;
    let (mut kt, mut nt, mut kk) = ((k / 16) as i32, (n / 16) as i32, krate as i32);
    let (mut g0, mut g1, mut g2) = (dah, dtre, dsb);
    hc.launch(
        "exl3_gemv",
        ((n / 16) / 8) as u32,
        nseg as u32,
        128,
        &mut [
            &mut g0 as *mut *mut u8 as *mut _,
            &mut g1 as *mut *mut u8 as *mut _,
            &mut g2 as *mut *mut u8 as *mut _,
            &mut kt as *mut i32 as *mut _,
            &mut nt as *mut i32 as *mut _,
            &mut kk as *mut i32 as *mut _,
        ],
    )?;
    let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, nseg as i32, n as i32);
    let (mut c0, mut c1, mut c2) = (dsb, dsvh, dy);
    hc.launch(
        "exl3_had_out",
        (n / 128) as u32,
        1,
        128,
        &mut [
            &mut c0 as *mut *mut u8 as *mut _,
            &mut c1 as *mut *mut u8 as *mut _,
            &mut c2 as *mut *mut u8 as *mut _,
            &mut nch as *mut i32 as *mut _,
            &mut nsg as *mut i32 as *mut _,
            &mut nst as *mut i32 as *mut _,
        ],
    )?;
    let mut yb = vec![0u8; n * 4];
    hc.d2h(&mut yb, dy)?;
    hc.sync()?;
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] = unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, n) };
    let mut md = 0f32;
    let mut nan = 0usize;
    for i in 0..n {
        if !got[i].is_finite() {
            nan += 1;
            continue;
        }
        md = md.max((got[i] - want[i]).abs());
    }
    Ok(format!(
        "hip-linear {key} k={k} n={n}: maxdiff={md:.3e} nan={nan}"
    ))
}
// 마커 lin1

// ── EXL3 hip 배치 gemm2 프로브(모듈 4/4) ── T행 체인: had_in→gemm2→had_out.
pub fn hip_gemm_check(dir: &str, t_arg: usize) -> Result<String, String> {
    use crate::rawvk::checks::TrellisResident;
    let key = "model.language_model.layers.0.mlp.gate_proj";
    let t_rows = t_arg;
    let mut tr = TrellisResident::load(dir)?;
    let (k, n, krate, suh, tre, svh) = tr.linear_raw(key)?;
    let x: Vec<f32> = {
        let mut seed: u32 = 0x33CC_0F0F;
        (0..t_rows * k)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.3
            })
            .collect()
    };
    // vk 참조: 표본 4행
    let samp: Vec<usize> = (0..4).map(|i| i * (t_rows - 1) / 3).collect();
    let want: Vec<Vec<f32>> = samp
        .iter()
        .map(|&r| tr.linear(key, &x[r * k..(r + 1) * k]))
        .collect::<Result<_, _>>()?;
    drop(tr);
    let hc = HipCtx::new()?;
    let nseg = 1usize; // gemm2는 nseg=1(sb [T][n])
    let dx = hc.alloc(t_rows * k * 4)?;
    let dsuh = hc.alloc(suh.len())?;
    let dtre = hc.alloc(tre.len())?;
    let dsvh = hc.alloc(svh.len())?;
    let dah = hc.alloc(t_rows * k * 2)?;
    let dsb = hc.alloc(t_rows * n * 4)?;
    let dy = hc.alloc(t_rows * n * 4)?;
    let xb: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, t_rows * k * 4) };
    hc.h2d(dx, xb)?;
    hc.h2d(dsuh, &suh)?;
    hc.h2d(dtre, &tre)?;
    hc.h2d(dsvh, &svh)?;
    // had_in: 그리드 (k/128, T)
    let (mut kc, mut ks) = ((k / 128) as i32, k as i32);
    let (mut p0, mut p1, mut p2) = (dx, dsuh, dah);
    hc.launch3(
        "exl3_had_in",
        (k / 128) as u32,
        t_rows as u32,
        1,
        128,
        &mut [
            &mut p0 as *mut *mut u8 as *mut _,
            &mut p1 as *mut *mut u8 as *mut _,
            &mut p2 as *mut *mut u8 as *mut _,
            &mut kc as *mut i32 as *mut _,
            &mut ks as *mut i32 as *mut _,
        ],
    )?;
    // had_in 다중행 대조 — 행 0·중간행 WHT 미러(러닝: gemv 프로브 미러 행 오프셋판).
    {
        let suh_f: Vec<f32> = (0..k)
            .map(|kk| {
                let b = (kk / 2) * 4;
                let w32 = u32::from_le_bytes([suh[b], suh[b + 1], suh[b + 2], suh[b + 3]]);
                half::f16::from_bits(((w32 >> ((kk & 1) * 16)) & 0xFFFF) as u16).to_f32()
            })
            .collect();
        let mut ahb = vec![0u8; t_rows * k * 2];
        hc.d2h(&mut ahb, dah)?;
        hc.sync()?;
        for &r in &[0usize, t_rows / 2] {
            let mut bad = 0usize;
            let ah_row: &[u32] = unsafe {
                std::slice::from_raw_parts(ahb[r * k * 2..].as_ptr() as *const u32, k / 2)
            };
            for ch in 0..k / 128 {
                let mut sm = [0f32; 128];
                for i in 0..128 {
                    let pre = half::f16::from_f32(x[r * k + ch * 128 + i] * suh_f[ch * 128 + i]);
                    sm[i] = pre.to_f32();
                }
                let mut w = 1usize;
                while w < 128 {
                    let mut i = 0;
                    while i < 128 {
                        let blk = (i / (2 * w)) * (2 * w);
                        for j in 0..w {
                            let a = sm[blk + j];
                            let b = sm[blk + j + w];
                            sm[blk + j] = a + b;
                            sm[blk + j + w] = a - b;
                        }
                        i += 2 * w;
                    }
                    w *= 2;
                }
                for i in (0..128).step_by(2) {
                    let lo = half::f16::from_f32(sm[i] * 0.08838834764831845);
                    let hi = half::f16::from_f32(sm[i + 1] * 0.08838834764831845);
                    let want = (lo.to_bits() as u32) | ((hi.to_bits() as u32) << 16);
                    if ah_row[ch * 64 + i / 2] != want {
                        bad += 1;
                    }
                }
            }
            eprintln!("  [gemmdbg] had_in 행 {r} 워드 불일치 {bad}/{}", k / 2);
        }
    }
    let (mut kt, mut nt, mut kk, mut tt) = (
        (k / 16) as i32,
        (n / 16) as i32,
        krate as i32,
        t_rows as i32,
    );
    let (mut g0, mut g1, mut g2) = (dah, dtre, dsb);
    hc.launch3(
        "exl3_gemm2",
        (n / 64) as u32,
        t_rows.div_ceil(128) as u32,
        1,
        128,
        &mut [
            &mut g0 as *mut *mut u8 as *mut _,
            &mut g1 as *mut *mut u8 as *mut _,
            &mut g2 as *mut *mut u8 as *mut _,
            &mut kt as *mut i32 as *mut _,
            &mut nt as *mut i32 as *mut _,
            &mut kk as *mut i32 as *mut _,
            &mut tt as *mut i32 as *mut _,
        ],
    )?;
    let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, nseg as i32, n as i32);
    let (mut c0, mut c1, mut c2) = (dsb, dsvh, dy);
    hc.launch3(
        "exl3_had_out",
        (n / 128) as u32,
        t_rows as u32,
        1,
        128,
        &mut [
            &mut c0 as *mut *mut u8 as *mut _,
            &mut c1 as *mut *mut u8 as *mut _,
            &mut c2 as *mut *mut u8 as *mut _,
            &mut nch as *mut i32 as *mut _,
            &mut nsg as *mut i32 as *mut _,
            &mut nst as *mut i32 as *mut _,
        ],
    )?;
    let mut yb = vec![0u8; t_rows * n * 4];
    hc.d2h(&mut yb, dy)?;
    hc.sync()?;
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] = unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, t_rows * n) };
    // sb 덤프 — gemm2의 s[행0/행1][0..3] vs gemv 세그합(행1).
    {
        let mut sbb = vec![0u8; t_rows * n * 4];
        hc.d2h(&mut sbb, dsb)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let sf: &[f32] =
            unsafe { std::slice::from_raw_parts(sbb.as_ptr() as *const f32, t_rows * n) };
        eprintln!(
            "  [sbdbg] gemm s[0][0..4]={:?} s[1][0..4]={:?} s[5][0..4]={:?}",
            &sf[0..4],
            &sf[n..n + 4],
            &sf[5 * n..5 * n + 4]
        );
    }
    // 3-way: 검증된 T=1 GEMV 체인으로 행 5 재계산 → gemm 행5·vk 참조 삼각 대조.
    {
        let r5 = samp[1];
        let mut gv_nsg = 16i32;
        let x5: Vec<f32> = x[r5 * k..(r5 + 1) * k].to_vec();
        let x5b: &[u8] = unsafe { std::slice::from_raw_parts(x5.as_ptr() as *const u8, k * 4) };
        hc.h2d(dx, x5b)?;
        hc.launch3(
            "exl3_had_in",
            (k / 128) as u32,
            1,
            1,
            128,
            &mut [
                &mut p0 as *mut *mut u8 as *mut _,
                &mut p1 as *mut *mut u8 as *mut _,
                &mut p2 as *mut *mut u8 as *mut _,
                &mut kc as *mut i32 as *mut _,
                &mut ks as *mut i32 as *mut _,
            ],
        )?;
        hc.launch3(
            "exl3_gemv",
            ((n / 16) / 8) as u32,
            16,
            1,
            128,
            &mut [
                &mut g0 as *mut *mut u8 as *mut _,
                &mut g1 as *mut *mut u8 as *mut _,
                &mut g2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
            ],
        )?;
        hc.launch3(
            "exl3_had_out",
            (n / 128) as u32,
            1,
            1,
            128,
            &mut [
                &mut c0 as *mut *mut u8 as *mut _,
                &mut c1 as *mut *mut u8 as *mut _,
                &mut c2 as *mut *mut u8 as *mut _,
                &mut nch as *mut i32 as *mut _,
                &mut gv_nsg as *mut i32 as *mut _,
                &mut nst as *mut i32 as *mut _,
            ],
        )?;
        let mut y5 = vec![0u8; n * 4];
        hc.d2h(&mut y5, dy)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let y5f: &[f32] = unsafe { std::slice::from_raw_parts(y5.as_ptr() as *const f32, n) };
        let mut m_gv_gemm = 0f32;
        let mut m_gv_vk = 0f32;
        for i in 0..n {
            m_gv_gemm = m_gv_gemm.max((got[r5 * n + i] - y5f[i]).abs());
            m_gv_vk = m_gv_vk.max((y5f[i] - want[1][i]).abs());
        }
        eprintln!("  [gemmdbg] 행5: gemv-vs-gemm={m_gv_gemm:.3e} gemv-vs-vk={m_gv_vk:.3e}");
        {
            let mut seg = vec![0u8; 16 * n * 4];
            hc.d2h(&mut seg, dsb)?;
            hc.sync()?;
            // SAFETY: d2h 완료 후 재해석 — [16세그][n] 부분합.
            let segf: &[f32] =
                unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const f32, 16 * n) };
            let sums: Vec<f32> = (0..4)
                .map(|c| (0..16).map(|g| segf[g * n + c]).sum())
                .collect();
            eprintln!("  [sbdbg] gemv(행5) 세그합[0..4]={sums:?}");
        }
    }
    let mut worst = 0f32;
    for (si, &r) in samp.iter().enumerate() {
        let mut md = 0f32;
        for i in 0..n {
            md = md.max((got[r * n + i] - want[si][i]).abs());
        }
        eprintln!("  [gemmdbg] 행 {r} maxdiff={md:.3e}");
        worst = worst.max(md);
    }
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        hc.launch3(
            "exl3_gemm2",
            (n / 64) as u32,
            t_rows.div_ceil(128) as u32,
            1,
            128,
            &mut [
                &mut g0 as *mut *mut u8 as *mut _,
                &mut g1 as *mut *mut u8 as *mut _,
                &mut g2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
                &mut tt as *mut i32 as *mut _,
            ],
        )?;
        hc.sync()?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tf = 2.0 * k as f64 * n as f64 * t_rows as f64 / 1e12;
    Ok(format!(
        "hip-gemm T={t_rows} gate_proj: 샘플 maxdiff={worst:.3e} · gemm2 {:.1}ms = {:.1} TF",
        ts[1],
        tf / (ts[1] / 1000.0)
    ))
}
// 마커 hgm
// 마커 hb1
// 마커 ps1
// 마커 3w
// 마커 3wb
// 마커 sb1

// ── EXL3 hip GDN 체인 프로브(모듈 5) ── conv→l2perm→scan→gate 4커널 종단.
// vk에서 비트검증된 Rust 미러(체인 프로브와 동일 산술)와 대조.
pub fn hip_gdn_check(dir: &str, layer_arg: usize) -> Result<String, String> {
    use crate::rawvk::checks::TrellisResident;
    let t_rows = 32usize;
    let n_gdn = 48usize;
    let mut tr = TrellisResident::load(dir)?;
    // 디코드 완전 미러: 전 깊이 업로드 + lay 인덱싱(슬라이스 검증은 선행 완료).
    let (cw, ab, alog, dtb, nw) = tr.gdn_chain_consts()?;
    let cw_m = cw[layer_arg * 10240 * 4..(layer_arg + 1) * 10240 * 4].to_vec();
    let ab_m = ab[layer_arg * 2 * 48 * 5120..(layer_arg + 1) * 2 * 48 * 5120].to_vec();
    let alog_m = alog[layer_arg * 48..(layer_arg + 1) * 48].to_vec();
    let dtb_m = dtb[layer_arg * 48..(layer_arg + 1) * 48].to_vec();
    let nw_m = nw[layer_arg * 128..(layer_arg + 1) * 128].to_vec();
    let _ = layer_arg;
    drop(tr);

    let mut seed: u32 = 0x6E0D_1234;
    let rnd = |s: &mut u32| {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
    };
    let xn: Vec<f32> = (0..t_rows * 5120).map(|_| rnd(&mut seed) * 0.3).collect();
    let qkv: Vec<f32> = (0..t_rows * 10240).map(|_| rnd(&mut seed) * 0.5).collect();
    let z: Vec<f32> = (0..t_rows * 6144).map(|_| rnd(&mut seed) * 0.4).collect();

    // ── Rust 미러(체인 프로브 산술) ──
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let k_len = 2048usize;
    let d_inner = 6144usize;
    let mut q_all = vec![0f32; t_rows * k_len];
    let mut k_all = vec![0f32; t_rows * k_len];
    let mut v_all = vec![0f32; t_rows * d_inner];
    for c in 0..10240usize {
        let (w0, w1, w2, w3) = (
            cw_m[c * 4],
            cw_m[c * 4 + 1],
            cw_m[c * 4 + 2],
            cw_m[c * 4 + 3],
        );
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
    let mut q_l2 = vec![0f32; t_rows * k_len];
    let mut k_l2 = vec![0f32; t_rows * k_len];
    let mut v_lc = vec![0f32; t_rows * d_inner];
    let mut bg = vec![0f32; t_rows * 96];
    for t in 0..t_rows {
        for kh in 0..16usize {
            let b0 = t * k_len + kh * 128;
            let qn: f32 = (0..128).map(|i| q_all[b0 + i] * q_all[b0 + i]).sum();
            let kn: f32 = (0..128).map(|i| k_all[b0 + i] * k_all[b0 + i]).sum();
            let qi = 1.0 / (qn + 1e-6).sqrt();
            let ki = 1.0 / (kn + 1e-6).sqrt();
            for i in 0..128 {
                q_l2[b0 + i] = q_all[b0 + i] * qi;
                k_l2[b0 + i] = k_all[b0 + i] * ki;
            }
        }
        for h in 0..48usize {
            let xrow = &xn[t * 5120..(t + 1) * 5120];
            let a_row = &ab_m[h * 5120..(h + 1) * 5120];
            let b_row = &ab_m[(48 + h) * 5120..(48 + h + 1) * 5120];
            let a_v: f32 = xrow.iter().zip(a_row).map(|(x, w)| x * w).sum();
            let b_v: f32 = xrow.iter().zip(b_row).map(|(x, w)| x * w).sum();
            let g = softplus(a_v + dtb_m[h]) * (-alog_m[h].exp());
            let beta = 1.0 / (1.0 + (-b_v).exp());
            let p_inv = (h % 3) * 16 + h / 3;
            bg[t * 96 + p_inv] = beta;
            bg[t * 96 + 48 + p_inv] = g;
            let src = t * d_inner + h * 128;
            let dst = t * d_inner + p_inv * 128;
            v_lc[dst..dst + 128].copy_from_slice(&v_all[src..src + 128]);
        }
    }
    let (o_ref, _) = crate::rawvk::checks::exl3_probes::scan_ref(
        &q_l2,
        &k_l2,
        &v_lc,
        &bg,
        t_rows,
        false,
        &[0f32; 48 * 16384],
    );
    let mut want = vec![0f32; t_rows * d_inner];
    for t in 0..t_rows {
        for h in 0..48usize {
            let p_inv = (h % 3) * 16 + h / 3;
            let src = t * d_inner + p_inv * 128;
            let dst = t * d_inner + h * 128;
            let ss: f32 = (0..128).map(|i| o_ref[src + i] * o_ref[src + i]).sum();
            let inv = 1.0 / (ss / 128.0 + 1e-6).sqrt();
            for i in 0..128 {
                let zv = z[dst + i];
                want[dst + i] = o_ref[src + i] * inv * nw_m[i] * silu(zv);
            }
        }
    }

    // ── hip 체인 ──
    let hc = HipCtx::new()?;
    let f32b =
        |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let dqkv = hc.alloc(t_rows * 10240 * 4)?;
    let dcw = hc.alloc(cw.len() * 4)?;
    let dring = hc.alloc(n_gdn * 3 * 10240 * 4)?;
    let dgq = hc.alloc(t_rows * 2048 * 4)?;
    let dgk = hc.alloc(t_rows * 2048 * 4)?;
    let dgv = hc.alloc(t_rows * 6144 * 4)?;
    let dxn = hc.alloc(t_rows * 5120 * 4)?;
    let dab = hc.alloc(ab.len() * 4)?;
    let dal = hc.alloc(alog.len() * 4)?;
    let ddt = hc.alloc(dtb.len() * 4)?;
    let dq2 = hc.alloc(t_rows * 2048 * 4)?;
    let dk2 = hc.alloc(t_rows * 2048 * 4)?;
    let dv2 = hc.alloc(t_rows * 6144 * 4)?;
    let mut dbg = hc.alloc(t_rows * 96 * 4)?;
    let dst = hc.alloc(n_gdn * 48 * 16384 * 4)?;
    let dgo = hc.alloc(t_rows * 6144 * 4)?;
    let dz = hc.alloc(t_rows * 6144 * 4)?;
    let dnw = hc.alloc(nw.len() * 4)?;
    let dgt = hc.alloc(t_rows * 6144 * 4)?;
    hc.h2d(dqkv, f32b(&qkv))?;
    hc.h2d(dcw, f32b(&cw))?;
    hc.h2d(dxn, f32b(&xn))?;
    hc.h2d(dab, f32b(&ab))?;
    hc.h2d(dal, f32b(&alog))?;
    hc.h2d(ddt, f32b(&dtb))?;
    hc.h2d(dz, f32b(&z))?;
    hc.h2d(dnw, f32b(&nw))?;
    {
        let zero = vec![0u8; n_gdn * 3 * 10240 * 4];
        hc.h2d(dring, &zero)?;
        let zs = vec![0u8; n_gdn * 48 * 16384 * 4];
        hc.h2d(dst, &zs)?;
    }
    let mut tl = t_rows as i32;
    let mut lay = layer_arg as i32;
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (dqkv, dcw, dring, dgq, dgk, dgv);
    hc.launch(
        "exl3_gdn_conv",
        10240 / 128,
        1,
        128,
        &mut [
            &mut a0 as *mut *mut u8 as *mut _,
            &mut a1 as *mut *mut u8 as *mut _,
            &mut a2 as *mut *mut u8 as *mut _,
            &mut a3 as *mut *mut u8 as *mut _,
            &mut a4 as *mut *mut u8 as *mut _,
            &mut a5 as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut lay as *mut i32 as *mut _,
        ],
    )?;
    let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6, mut b7, mut b8, mut b9) =
        (dgq, dgk, dgv, dxn, dab, dal, ddt, dq2, dk2, dv2);
    let mut hk = 16i32;
    let mut hv = 48i32;
    let mut dd = 128i32;
    hc.launch3(
        "exl3_gdn_l2perm",
        48,
        t_rows as u32,
        1,
        128,
        &mut [
            &mut b0 as *mut *mut u8 as *mut _,
            &mut b1 as *mut *mut u8 as *mut _,
            &mut b2 as *mut *mut u8 as *mut _,
            &mut b3 as *mut *mut u8 as *mut _,
            &mut b4 as *mut *mut u8 as *mut _,
            &mut b5 as *mut *mut u8 as *mut _,
            &mut b6 as *mut *mut u8 as *mut _,
            &mut b7 as *mut *mut u8 as *mut _,
            &mut b8 as *mut *mut u8 as *mut _,
            &mut b9 as *mut *mut u8 as *mut _,
            &mut dbg as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut lay as *mut i32 as *mut _,
        ],
    )?;
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (dq2, dk2, dv2, dbg, dst, dgo);
    hc.launch3(
        "exl3_gdn_scan",
        48,
        1,
        1,
        128,
        &mut [
            &mut c0 as *mut *mut u8 as *mut _,
            &mut c1 as *mut *mut u8 as *mut _,
            &mut c2 as *mut *mut u8 as *mut _,
            &mut c3 as *mut *mut u8 as *mut _,
            &mut c4 as *mut *mut u8 as *mut _,
            &mut c5 as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut hk as *mut i32 as *mut _,
            &mut hv as *mut i32 as *mut _,
            &mut dd as *mut i32 as *mut _,
            &mut lay as *mut i32 as *mut _,
        ],
    )?;
    let (mut e0, mut e1, mut e2, mut e3) = (dgo, dz, dnw, dgt);
    hc.launch3(
        "exl3_gdn_gate",
        48,
        t_rows as u32,
        1,
        128,
        &mut [
            &mut e0 as *mut *mut u8 as *mut _,
            &mut e1 as *mut *mut u8 as *mut _,
            &mut e2 as *mut *mut u8 as *mut _,
            &mut e3 as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut lay as *mut i32 as *mut _,
        ],
    )?;
    let mut outb = vec![0u8; t_rows * 6144 * 4];
    hc.d2h(&mut outb, dgt)?;
    hc.sync()?;
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] =
        unsafe { std::slice::from_raw_parts(outb.as_ptr() as *const f32, t_rows * 6144) };
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
        "hip-gdn T={t_rows}: maxdiff={md:.3e} nan={nan} rel>5%={rel_bad}/{}",
        got.len()
    ))
}
// 마커 gdn1

// ── EXL3 hip 어텐션 체인 프로브(모듈 6-7) ── prep→fwd3, Rust 미러 대조.
pub fn hip_attn_check(dir: &str) -> Result<String, String> {
    use crate::rawvk::checks::TrellisResident;
    let t_rows = 8usize;
    let pos0 = 0usize;
    let layer = 0usize;
    let mut tr = TrellisResident::load(dir)?;
    let (qnw, knw) = tr.attn_norms_dump()?;
    drop(tr);
    let mut seed: u32 = 0xA77E_2024;
    let rnd = |s: &mut u32| {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
    };
    let qg: Vec<f32> = (0..t_rows * 12288).map(|_| rnd(&mut seed) * 0.5).collect();
    let kin: Vec<f32> = (0..t_rows * 1024).map(|_| rnd(&mut seed) * 0.4).collect();
    let vin: Vec<f32> = (0..t_rows * 1024).map(|_| rnd(&mut seed) * 0.4).collect();

    let rope = |hd: &mut [f32; 256], pos: usize| {
        for tid in 0..32usize {
            let theta = 1e7f32.powf(-(2.0 * tid as f32) / 64.0);
            let ang = pos as f32 * theta;
            let (c, s2) = (ang.cos(), ang.sin());
            let (x0, x1) = (hd[tid], hd[tid + 32]);
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
    };
    let mut qh = vec![0f32; t_rows * 6144];
    let mut kc = vec![0f32; 16 * 1024 * 1024];
    let mut vc = vec![0f32; 16 * 1024 * 1024];
    for t in 0..t_rows {
        let pos = pos0 + t;
        for j in 0..24usize {
            let mut hd = [0f32; 256];
            let src = t * 12288 + j * 512;
            hd.copy_from_slice(&qg[src..src + 256]);
            let ss: f32 = hd.iter().map(|v| v * v).sum::<f32>() / 256.0;
            let inv = 1.0 / (ss + 1e-6).sqrt();
            for i in 0..256 {
                hd[i] *= inv * qnw[layer * 256 + i];
            }
            rope(&mut hd, pos);
            for i in 0..256 {
                qh[t * 6144 + j * 256 + i] = hd[i];
            }
        }
        for m in 0..4usize {
            let mut hd = [0f32; 256];
            let src = t * 1024 + m * 256;
            hd.copy_from_slice(&kin[src..src + 256]);
            let ss: f32 = hd.iter().map(|v| v * v).sum::<f32>() / 256.0;
            let inv = 1.0 / (ss + 1e-6).sqrt();
            for i in 0..256 {
                hd[i] *= inv * knw[layer * 256 + i];
            }
            rope(&mut hd, pos);
            let dst = (layer * 1024 + pos) * 1024 + m * 256;
            kc[dst..dst + 256].copy_from_slice(&hd);
            vc[dst..dst + 256].copy_from_slice(&vin[src..src + 256]);
        }
    }
    let mut want = vec![0f32; t_rows * 6144];
    for t in 0..t_rows {
        let lim = pos0 + t + 1;
        for h in 0..24usize {
            let kh = h / 6;
            let mut sc = vec![0f32; lim];
            let mut mx = -1e30f32;
            for row in 0..lim {
                let mut p = 0f32;
                for d in 0..256 {
                    p +=
                        qh[t * 6144 + h * 256 + d] * kc[(layer * 1024 + row) * 1024 + kh * 256 + d];
                }
                sc[row] = p * 0.0625;
                mx = mx.max(sc[row]);
            }
            let mut ws = 0f32;
            for row in 0..lim {
                sc[row] = (sc[row] - mx).exp();
                ws += sc[row];
            }
            let mut acc = vec![0f32; 256];
            for row in 0..lim {
                let w = sc[row];
                for d in 0..256 {
                    acc[d] += w * vc[(layer * 1024 + row) * 1024 + kh * 256 + d];
                }
            }
            for d in 0..256 {
                let g = qg[t * 12288 + h * 512 + 256 + d];
                let sg2 = 1.0 / (1.0 + (-g).exp());
                want[t * 6144 + h * 256 + d] = (acc[d] / ws) * sg2;
            }
        }
    }

    let hc = HipCtx::new()?;
    let f32b =
        |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let dqg = hc.alloc(t_rows * 12288 * 4)?;
    let dkin = hc.alloc(t_rows * 1024 * 4)?;
    let dvin = hc.alloc(t_rows * 1024 * 4)?;
    let dqnw = hc.alloc(qnw.len() * 4)?;
    let dknw = hc.alloc(knw.len() * 4)?;
    let dqh = hc.alloc(t_rows * 6144 * 4)?;
    let dkc = hc.alloc(16 * 1024 * 1024 * 4)?;
    let dvc = hc.alloc(16 * 1024 * 1024 * 4)?;
    let dou = hc.alloc(t_rows * 6144 * 4)?;
    let dpp = hc.alloc(4)?;
    let mut ppv = pos0 as u32;
    hc.h2d(dqg, f32b(&qg))?;
    hc.h2d(dkin, f32b(&kin))?;
    hc.h2d(dvin, f32b(&vin))?;
    hc.h2d(dqnw, f32b(&qnw))?;
    hc.h2d(dknw, f32b(&knw))?;
    hc.h2d(dpp, &ppv.to_le_bytes())?;
    let z = vec![0u8; 16 * 1024 * 1024 * 4];
    hc.h2d(dkc, &z)?;
    hc.h2d(dvc, &z)?;
    let mut tl = t_rows as i32;
    let mut p0 = pos0 as i32;
    let mut lay = layer as i32;
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
        (dqg, dkin, dvin, dqnw, dknw, dqh, dkc, dvc, dpp);
    hc.launch3(
        "exl3_attn_prep",
        t_rows as u32,
        28,
        1,
        128,
        &mut [
            &mut a0 as *mut *mut u8 as *mut _,
            &mut a1 as *mut *mut u8 as *mut _,
            &mut a2 as *mut *mut u8 as *mut _,
            &mut a3 as *mut *mut u8 as *mut _,
            &mut a4 as *mut *mut u8 as *mut _,
            &mut a5 as *mut *mut u8 as *mut _,
            &mut a6 as *mut *mut u8 as *mut _,
            &mut a7 as *mut *mut u8 as *mut _,
            &mut a8 as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut p0 as *mut i32 as *mut _,
            &mut lay as *mut i32 as *mut _,
        ],
    )?;
    // prep 산출 대조 — qh/kc 미러와 직접(국소화: prep vs fwd3).
    {
        let mut qhb = vec![0u8; t_rows * 6144 * 4];
        hc.d2h(&mut qhb, dqh)?;
        let mut kcb = vec![0u8; 16 * 1024 * 1024 * 4];
        hc.d2h(&mut kcb, dkc)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let gh: &[f32] =
            unsafe { std::slice::from_raw_parts(qhb.as_ptr() as *const f32, t_rows * 6144) };
        let gk: &[f32] =
            unsafe { std::slice::from_raw_parts(kcb.as_ptr() as *const f32, 16 * 1024 * 1024) };
        let mut mq = 0f32;
        for i in 0..gh.len() {
            mq = mq.max((gh[i] - qh[i]).abs());
        }
        let mut mk = 0f32;
        let kvn = (pos0 + t_rows) * 1024;
        for i in 0..kvn.min(gk.len()) {
            mk = mk.max((gk[i] - kc[i]).abs());
        }
        eprintln!("  [attndbg] qh maxdiff={mq:.3e} · kc(적립분) maxdiff={mk:.3e}");
    }
    let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5) = (dqh, dkc, dvc, dqg, dou, dpp);
    hc.launch3(
        "exl3_attn_fwd3s",
        t_rows as u32,
        24,
        1,
        256,
        &mut [
            &mut b0 as *mut *mut u8 as *mut _,
            &mut b1 as *mut *mut u8 as *mut _,
            &mut b2 as *mut *mut u8 as *mut _,
            &mut b3 as *mut *mut u8 as *mut _,
            &mut b4 as *mut *mut u8 as *mut _,
            &mut b5 as *mut *mut u8 as *mut _,
            &mut tl as *mut i32 as *mut _,
            &mut p0 as *mut i32 as *mut _,
            &mut lay as *mut i32 as *mut _,
        ],
    )?;
    let mut outb = vec![0u8; t_rows * 6144 * 4];
    hc.d2h(&mut outb, dou)?;
    hc.sync()?;
    let _ = &mut ppv;
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] =
        unsafe { std::slice::from_raw_parts(outb.as_ptr() as *const f32, t_rows * 6144) };
    let mut md = 0f32;
    let mut nan = 0usize;
    let mut bad_t = [0usize; 8];
    let mut bad_h = [0usize; 24];
    for i in 0..got.len() {
        if !got[i].is_finite() {
            nan += 1;
            continue;
        }
        let d = (got[i] - want[i]).abs();
        if d > 1e-3 {
            bad_t[i / 6144] += 1;
            bad_h[(i % 6144) / 256] += 1;
        }
        md = md.max(d);
    }
    eprintln!(
        "  [attndbg] 불일치 t분포={:?} h분포={:?}",
        &bad_t[..t_rows],
        &bad_h[..]
    );
    Ok(format!("hip-attn T={t_rows}: maxdiff={md:.3e} nan={nan}"))
}
// 마커 at1
// 마커 ab1
// 마커 ab2
// 마커 l1f

use crate::rawvk::checks::TrellisResident;

// ── EXL3 hip 디코드 루프(정확성 우선 조립) ── vk decode_step 로짓 대조.
pub fn hip_decode_check(dir: &str, tok0: u32, lim_layers: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    use crate::rawvk::checks::TrellisResident;
    let mut dec = Exl3HipDecoder::load(dir, lim_layers)?;
    let mut tr = TrellisResident::load(dir)?;
    let embed: Vec<f32> = tr.embed_row(tok0).to_vec();
    // greedy 4스텝(첫 로짓이 대조 기준 — 상태는 자연 갱신).
    let mut tok = tok0;
    let mut hip_toks = Vec::new();
    let mut first: Option<Vec<f32>> = None;
    for _ in 0..4 {
        let (lg, _) = dec.forward(&tr.embed_row(tok).to_vec())?;
        if first.is_none() {
            first = Some(lg.clone());
        }
        let am = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        hip_toks.push(am as u32);
        tok = am as u32;
    }
    let got = first.unwrap_or_default();
    let hid: Vec<f32> = Vec::new();
    let mut seq = crate::rawvk::checks::exl3_decode::new_seq_state(tr.n_layers, 512);
    let want = crate::rawvk::checks::exl3_decode::decode_step(&mut tr, &mut seq, tok0)?;
    let whid = seq.last_h.clone();
    let mut vt = Vec::new();
    let mut vtok = want
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0) as u32;
    for _ in 0..3 {
        vt.push(vtok);
        let lg2 = crate::rawvk::checks::exl3_decode::decode_step(&mut tr, &mut seq, vtok)?;
        vtok = lg2
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as u32;
    }
    vt.push(vtok);
    eprintln!("  [vk-greedy] 4토큰 {vt:?}");
    drop(tr);
    if !hid.is_empty() {
        let mut hmd = 0f32;
        for i in 0..hid.len().min(whid.len()) {
            hmd = hmd.max((hid[i] - whid[i]).abs());
        }
        eprintln!("  [hiddbg] hidden maxdiff={hmd:.3e}");
    }
    eprintln!("  [hip-greedy] 4토큰 {hip_toks:?}");
    let mut md = 0f32;
    let (mut ga, mut wa) = (0usize, 0usize);
    for i in 0..got.len() {
        md = md.max((got[i] - want[i]).abs());
        if got[i] > got[ga] {
            ga = i;
        }
        if want[i] > want[wa] {
            wa = i;
        }
    }
    Ok(format!(
        "hip-decode tok{tok0}: 로짓 maxdiff={md:.3e} argmax hip={ga} vk={wa} {}",
        if ga == wa { "일치" } else { "불일치" }
    ))
}
// 마커 gp1
// 마커 fm1
// 마커 md1
// 마커 g4
