use crate::rawhip::ctx::RawCtx as HipCtx;

// ── EXL3 hip GEMV 체인 프로브(plans/121 CMP 포팅 · todo 2/4) ──
// vk 가중치를 그대로 투입해 hipRTC 컴파일 exl3_had_in→gemv→had_out 체인을
// 실행, vk 트레이리던트 참조(tr.linear)와 대조 — 8060S hipRTC로 검증.
pub fn hip_gemv_check(dir: &str) -> Result<String, String> {
    use crate::rawvk::checks::TrellisResident;
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
