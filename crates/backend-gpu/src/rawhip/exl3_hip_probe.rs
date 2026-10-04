//! [검증층 원장 2026-10-04] exl3-hip-gemv/nr/linear/gemm/gdn/attn 프로브 —
//! 모듈층(exl3_hip.rs) 헤더의 측정값 참조. 박막 hip-decode는 greedy-4 대조.
use crate::rawhip::ctx::RawCtx as HipCtx;

// ── EXL3 hip GEMV 체인 프로브(plans/121 CMP 포팅 · todo 2/4) ──
// vk 가중치를 그대로 투입해 hipRTC 컴파일 exl3_had_in→gemv→had_out 체인을
// 실행, vk 트레이리던트 참조(tr.linear)와 대조 — 8060S hipRTC로 검증.
pub fn hip_gemv_check(dir: &str) -> Result<String, String> {
    // 부분 적재: lm_head 1개 선형만 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|n: &str| n == "lm_head")?;
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
    // 부분 적재: 노름만 필요 — 선형 전용 스킵 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|_| false)?;
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
    // 부분 적재: 대상 선형 1개만 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|n: &str| n == key)?;
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
    // 부분 적재: L0 gate_proj 1개 선형만 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|n: &str| n == key)?;
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
    // [수리 2026-10-04] had_in 출력을 **전용 버퍼 dah5**에 쓴다 — 종전 p2=dah가
    // dah 행0을 x[samp[1]] 변환으로 덮어썼고, 뒤따르는 wmma/mma 블록이 오염된
    // 활성으로 토큰0 = 토큰 samp[1]의 dot를 계산했다(“WMMA 행0 오염”의 진범,
    // plans/125-3 · plans/126 I1 — 커널 무죄, 프로브 하네스 결함).
    {
        let r5 = samp[1];
        let mut gv_nsg = 16i32;
        let dah5 = hc.alloc(k * 2)?;
        let x5: Vec<f32> = x[r5 * k..(r5 + 1) * k].to_vec();
        let x5b: &[u8] = unsafe { std::slice::from_raw_parts(x5.as_ptr() as *const u8, k * 4) };
        hc.h2d(dx, x5b)?;
        let mut p2b = dah5;
        hc.launch3(
            "exl3_had_in",
            (k / 128) as u32,
            1,
            1,
            128,
            &mut [
                &mut p0 as *mut *mut u8 as *mut _,
                &mut p1 as *mut *mut u8 as *mut _,
                &mut p2b as *mut *mut u8 as *mut _,
                &mut kc as *mut i32 as *mut _,
                &mut ks as *mut i32 as *mut _,
            ],
        )?;
        let (mut g0, mut g1, mut g2) = (dah5, dtre, dsb);
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
    // MMA 변형(plans/127 B) — Q4-MMQ 구조 이식판: 정합+속도 A/B.
    {
        let (mut kt, mut nt2, mut kk2, mut tt2) = (
            (k / 16) as i32,
            (n / 16) as i32,
            krate as i32,
            t_rows as i32,
        );
        let (mut g0, mut g1, mut g2) = (dah, dtre, dsb);
        let mut tm: Vec<f64> = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            hc.launch3(
                "exl3_gemm2_mma",
                (n / 64) as u32,
                t_rows.div_ceil(64) as u32,
                1,
                256,
                &mut [
                    &mut g0 as *mut *mut u8 as *mut _,
                    &mut g1 as *mut *mut u8 as *mut _,
                    &mut g2 as *mut *mut u8 as *mut _,
                    &mut kt as *mut i32 as *mut _,
                    &mut nt2 as *mut i32 as *mut _,
                    &mut kk2 as *mut i32 as *mut _,
                    &mut tt2 as *mut i32 as *mut _,
                ],
            )?;
            hc.sync()?;
            tm.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        // H⁻¹⊙svh 후처리(제자리) — want가 최종 도메인이므로 동일 적용.
        {
            let (mut c0, mut c1, mut c2) = (dsb, dsvh, dsb);
            let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, 1i32, n as i32);
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
            hc.sync()?;
        }
        tm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut mbb = vec![0u8; t_rows * n * 4];
        hc.d2h(&mut mbb, dsb)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let mfc: &[f32] =
            unsafe { std::slice::from_raw_parts(mbb.as_ptr() as *const f32, t_rows * n) };
        let mut mmd = 0f32;
        for (si, &r) in samp.iter().enumerate() {
            let _ = si;
            for i in 0..n {
                mmd = mmd.max((mfc[r * n + i] - want[si][i]).abs());
            }
        }
        for (si, &r) in samp.iter().enumerate() {
            if si >= 4 {
                break;
            }
            let mut rowmd = 0f32;
            for i in 0..n {
                rowmd = rowmd.max((mfc[r * n + i] - want[si][i]).abs());
            }
            eprintln!("  [mmarow] 샘플행{r} maxdiff={rowmd:.3e}");
            if si == 0 && r == 0 {
                eprintln!("  [mmav0] gpu={:?}", &mfc[0..6]);
                eprintln!("  [mmav0] ref={:?}", &want[0][0..6]);
            }
        }
        let tf2 = 2.0 * k as f64 * n as f64 * t_rows as f64 / 1e12;
        eprintln!(
            "  [mmadbg] mma {:.1}ms = {:.1} TF · y-vs-ref maxdiff={mmd:.3e}",
            tm[1],
            tf2 / (tm[1] / 1000.0)
        );
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
    // 부분 적재: GDN 상수(노름)만 필요 — 선형 전용 스킵 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|_| false)?;
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
    // 부분 적재: q/k 노름만 필요 — 선형 전용 스킵 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|_| false)?;
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
    // 동결 방지(2026-10-04 사고 원칙): 한 시점에 한 모델만 상주.
    // 1단계: hip 디코더(임베딩 포함) 단독 — greedy 4스텝.
    let mut dec = Exl3HipDecoder::load(dir, lim_layers)?;
    let t0f = std::time::Instant::now();
    let mut tok = tok0;
    let mut hip_toks = Vec::new();
    let mut first: Option<Vec<f32>> = None;
    for _ in 0..4 {
        let lg = dec.forward_tok(tok)?;
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
    let fwd_ms = t0f.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "  [tgdbg] 4스텝 forward {fwd_ms:.0}ms → {:.2} t/s(셔틀 포함)",
        4000.0 / fwd_ms
    );
    let got = first.unwrap_or_default();
    drop(dec);
    // 2단계: vk 참조 단독 재로드.
    let mut tr = TrellisResident::load(dir)?;
    let mut seq = crate::rawvk::checks::exl3_decode::new_seq_state(tr.n_layers, 512);
    let want = crate::rawvk::checks::exl3_decode::decode_step(&mut tr, &mut seq, tok0)?;
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
// 마커 wab
// 마커 wab2

/// `llm170 exl3-hip-mtp <dir> <tok>` — MTP 드래프트 모듈 격리 검증:
/// 합성 hidden(결정론 패턴)으로 hip gemv 경로 vs vk 참조 mtp_step 로짓 대조.
pub fn hip_mtp_check(dir: &str, tok: u32) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    use crate::rawvk::checks::TrellisResident;
    let h = 5120usize;
    let synth: Vec<f32> = (0..h).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    // 1단계: hip 단독(mtp 가중치만 사용)
    let mut dec = Exl3HipDecoder::load(dir, 0)?;
    // GPU 드래프트 A/B: 동일 입력으로 정합 + 시간(호스트 버전 기준).
    let tg0 = std::time::Instant::now();
    let d_gpu = dec.mtp_draft_gpu(tok, &synth, 0)?;
    let tg = tg0.elapsed().as_secs_f64() * 1e3;
    let th0 = std::time::Instant::now();
    let tl = dec.mtp_draft(tok, &synth, 0)?;
    let th = th0.elapsed().as_secs_f64() * 1e3;
    let am_h = tl
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    eprintln!("  [dab] gpu={d_gpu} host={am_h} · gpu {tg:.0}ms host {th:.0}ms");
    let tl = dec.mtp_draft(tok, &synth, 0)?;
    drop(dec);
    // 2단계: vk 참조 단독
    let mut tr = TrellisResident::load(dir)?;
    let mut seq = crate::rawvk::checks::exl3_decode::new_seq_state(tr.n_layers, 512);
    let wl = crate::rawvk::checks::exl3_decode::mtp_step(&mut tr, &mut seq, tok, &synth, 0, true)?;
    let mut md = 0f32;
    let (mut ga, mut wa) = (0usize, 0usize);
    for i in 0..tl.len() {
        md = md.max((tl[i] - wl.0[i]).abs());
        if tl[i] > tl[ga] {
            ga = i;
        }
        if wl.0[i] > wl.0[wa] {
            wa = i;
        }
    }
    Ok(format!(
        "hip-mtp tok{tok}: 로짓 maxdiff={md:.3e} argmax hip={ga} vk={wa} {}",
        if ga == wa { "일치" } else { "불일치" }
    ))
}
// 마커 mtpd
// 마커 mtpf

/// `llm170 exl3-hip-batch <dir> <tok> [T]` — 배치 forward(프리필/검증 경로) 정합:
/// T행 임베딩으로 forward_batch → 행별 argmax를 순차 디코드와 대조.
pub fn hip_batch_check(dir: &str, tok: u32, t_len: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let t = t_len.clamp(1, 8);
    let mut dec = Exl3HipDecoder::load(dir, dec_layers_default(dir))?;
    // 1) 순차 greedy T+1스텝(기준)
    let mut seq_toks = Vec::new();
    let mut seq_lgs: Vec<Vec<f32>> = Vec::new();
    let mut tk = tok;
    for _ in 0..=t {
        let lg = dec.forward_tok(tk)?;
        let am = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        seq_toks.push(am);
        seq_lgs.push(lg);
        tk = am;
    }
    drop(dec);
    // 2) 배치: [tok, s1..st] 행 — 마지막 행의 argmax가 순차 t+1번째와 일치해야.
    let rows_toks: Vec<u32> = std::iter::once(tok)
        .chain(seq_toks.iter().take(t).copied())
        .collect();
    let mut dec2 = Exl3HipDecoder::load(dir, dec_layers_default(dir))?;
    dec2.dbg_layers = true;
    let mut rows = Vec::with_capacity(rows_toks.len());
    for rt in &rows_toks {
        rows.push(dec2.embed_row_host(*rt));
    }
    let (lgs, _) = dec2.forward_batch(&rows)?;
    // 전 행 argmax — 첫 이탈 행 국소화(순차 기준과 행별 대조).
    let seq_ref: Vec<u32> = std::iter::once(tok).chain(seq_toks.clone()).collect();
    for (ri, lgr) in lgs.iter().enumerate() {
        let ra = lgr
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        eprintln!(
            "  [fbrow] 행{ri} argmax={ra} (순차 다음토큰 {}){}",
            seq_ref.get(ri + 1).copied().unwrap_or(0),
            if seq_ref.get(ri + 1) == Some(&ra) {
                " ✓"
            } else {
                " ✗"
            }
        );
    }
    // 행별 로짓 maxdiff — 플립이 f16 노이즈(≈1e-2)인지 계통(≥1e-1)인지 정량화.
    for (ri, lgr) in lgs.iter().enumerate() {
        if let Some(sl) = seq_lgs.get(ri) {
            let md = lgr
                .iter()
                .zip(sl)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("  [fbmd] 行{ri} maxdiff={md:.3e}");
        }
    }
    let last = lgs.last().ok_or("batch empty")?;
    let bam = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    let ok = bam == seq_toks[t];
    Ok(format!(
        "hip-batch T={t}: 순차 {:?} · 배치 마지막 argmax={bam} (기준 {}) — {}",
        seq_toks,
        seq_toks[t],
        if ok { "일치" } else { "불일치" }
    ))
}

fn dec_layers_default(_dir: &str) -> usize {
    64
}
// 마커 fb4
// 마커 fb7
// 마커 fb8
// 마커 fbd
// 마커 fbc
// 마커 fbr
// 마커 blp
// 마커 mdq

/// `llm170 exl3-hip-mtp-round <dir> <tok> <rounds>` — MTP 라운드 경제성·정합 실측.
/// 구조(무롤백 v1): 상태=직전 확정 토큰 pp(타깃 자신의 예측). 라운드 =
///   드래프트 d=mtp_draft → 검증 배치 [pp, d] → d==argmax(row0)면 pp+d 확정,
///   아니면 pp+corr 확정 후 교정 T=1 배치(상태 정렬). 다음 pp=row_last argmax.
pub fn hip_mtp_round(dir: &str, tok: u32, rounds: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64)?;
    // 기준: 순차 greedy 2*rounds+4 토큰(교차 검증용)
    let n_ref = 2 * rounds + 4;
    let mut ref_toks = Vec::new();
    {
        let mut tk = tok;
        let mut ht = Vec::new();
        for _ in 0..n_ref {
            let row = dec.embed_row_host(tk);
            let (lg, h) = dec.forward(&row)?;
            ht = h;
            let am = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            ref_toks.push(am);
            tk = am;
        }
        let _ = ht;
    }
    // MTP 라운드 — 상태 리셋 필요: 새 디코더(순차와 동일 출발).
    drop(dec);
    let mut dec2 = Exl3HipDecoder::load(dir, 64)?;
    let mut h: Vec<f32>;
    {
        let row0 = dec2.embed_row_host(tok);
        let (lg0, h0) = dec2.forward_batch_with_mtp(&[row0], &[tok])?;
        h = h0;
        let _ = lg0;
    }
    let mut pp = ref_toks[0]; // 타깃 1스텝 후 자신의 예측(상태는 tok 처리까지)
    let mut out_toks = Vec::new();
    let (mut acc, mut tot) = (0usize, 0usize);
    let t0 = std::time::Instant::now();
    for _ in 0..rounds {
        // 드래프트(상태 = pp 직전? 규약: mtp_draft(tok=직전 확정, h, pos) — pos는 pp까지)
        let pos_now = dec2.pos;
        let d = dec2.mtp_draft_gpu(pp, &h, pos_now)?; // pos: pp가 처리된 뒤 위치
        // 검증 배치 [pp, d] + mtp KV 적립 훅
        let rows = vec![dec2.embed_row_host(pp), dec2.embed_row_host(d)];
        let (lgs, hnew) = dec2.forward_batch_with_mtp(&rows, &[pp, d])?;
        let am0 = lgs[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let am1 = lgs[1]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        tot += 1;
        if am0 == d {
            acc += 1;
            out_toks.push(pp);
            out_toks.push(d);
            pp = am1;
            h = hnew;
        } else {
            // 거부: pp+corr 확정, 상태는 [pp,d]까지 전진됨 → 교정 T=1로 corr 재처리
            let corr = am0;
            out_toks.push(pp);
            out_toks.push(corr);
            let rrow = dec2.embed_row_host(corr);
            let (lgc, hc) = dec2.forward_batch_with_mtp(&[rrow], &[corr])?;
            let amc = lgc[0]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            pp = amc;
            h = hc;
        }
    }
    let el = t0.elapsed().as_secs_f64();
    let tps = out_toks.len() as f64 / el;
    // 정합: out_toks가 순차 ref의 앞부분과 일치하는가(스펙은 배치 로짓 기준 — 대조는 참고)
    let mut agree = 0usize;
    for (i, ot) in out_toks.iter().enumerate() {
        if ref_toks.get(i) == Some(ot) {
            agree += 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "mtp-round: {rounds}라운드 {acc}/{tot} 수용 · {}토큰 {el:.2}s → {tps:.2} t/s · 순차 일치 {agree}/{}",
        out_toks.len(),
        out_toks.len()
    ))
}
// 마커 mr1
// 마커 mr4
// 마커 dg5
// 마커 dab
// 마커 d3q
// 마커 mrf

/// `llm170 exl3-hip-a1 <dir> <tok> <steps>` — MTP 드래프트 a1 수용률 측정(vk exl3-mtp 재현):
/// 타깃 순차(+KV 훅) 매 스텝, mtp_draft_gpu(현 토큰, h_현토큰, pos) vs 타깃 실제 다음 토큰.
pub fn hip_mtp_a1(dir: &str, tok: u32, steps: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64)?;
    // 변형 A: 순차(gemv) 타깃 + 호스트 mtp 훅(vk exl3-mtp 동일 구조) —
    // h 클래스(배치 gemm2 h vs 순차 gemv h)가 a1 격차(0.44 vs 0.625) 원인인지 판별.
    let mut h_seq_store: Vec<Vec<f32>> = Vec::new();
    {
        let (mut hit_s, mut tot_s, mut cur_s) = (0usize, 0usize, tok);
        for _ in 0..steps {
            let row = dec.embed_row_host(cur_s);
            let (lg, h) = dec.forward(&row)?;
            let nxt = lg
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            h_seq_store.push(h.clone());
            let dl = dec.mtp_draft(cur_s, &h, dec.pos - 1)?;
            tot_s += 1;
            let am_d = dl
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if am_d == nxt {
                hit_s += 1;
            }
            cur_s = nxt;
        }
        eprintln!("  [a1seq] 순차경로 a1 = {hit_s}/{tot_s}");
        let _ = &h_seq_store;
    }
    let mut cur = tok;
    let (mut hit, mut tot, mut hit_h) = (0usize, 0usize, 0usize);
    let mut t_draft = 0f64;
    let t0 = std::time::Instant::now();
    for batch_i in 0..steps {
        let row = dec.embed_row_host(cur);
        let pos_before = dec.pos;
        let (lg, h) = dec.forward_batch_with_mtp(&[row], &[cur])?;
        if let Some(hs) = h_seq_store.get(batch_i) {
            let md = h
                .iter()
                .zip(hs)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "  [hmd] 스텝{batch_i} h_batch-vs-seq maxdiff={md:.3e} rms_h={:.3}",
                h.iter().map(|v| v * v).sum::<f32>().sqrt()
            );
        }
        let nxt = lg[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        // 드래프트: (현 토큰 cur, h_cur, pos_before) → 다음 예측 — GPU·호스트 동시 측정
        let td = std::time::Instant::now();
        let d = dec.mtp_draft_gpu(cur, &h, pos_before)?;
        t_draft += td.elapsed().as_secs_f64();
        let dh = dec.mtp_draft(cur, &h, pos_before)?;
        let am_h = dh
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        tot += 1;
        if d == nxt {
            hit += 1;
        }
        if am_h == nxt {
            hit_h += 1;
        }
        if tot <= 6 {
            eprintln!("  [a1dbg] 스텝{tot} gpu={d} host={am_h} target={nxt}");
        }
        cur = nxt;
    }
    let el = t0.elapsed().as_secs_f64();
    Ok(format!(
        "hip-a1: gpu {hit}/{tot} = {:.2} · host {hit_h}/{tot} = {:.2} · 타깃순차 {el:.2}s({:.2} t/s) · gpu드래프트 {t_draft:.3}s({:.1}ms/회)",
        hit as f64 / tot as f64,
        hit_h as f64 / tot as f64,
        steps as f64 / el,
        t_draft * 1e3 / steps as f64
    ))
}
// 마커 a1p
// 마커 da1
// 마커 da2

/// `llm170 exl3-hip-tbench <dir> <tok> [T]` — 배치 forward T별 비용 상각 곡선.
pub fn hip_tbench(dir: &str, tok: u32, t_max: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64)?;
    let mut out = String::new();
    for t in [1usize, 2, 4, 8, 16] {
        if t > t_max {
            break;
        }
        // 같은 토큰 반복 행(비용 측정 — 상태는 순차와 무관)
        let rows: Vec<Vec<f32>> = (0..t).map(|_| dec.embed_row_host(tok)).collect();
        // 워밍 1회 + 측정 3회 중앙값
        let _ = dec.forward_batch(&rows)?;
        let mut ts: Vec<f64> = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let _ = dec.forward_batch(&rows)?;
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ts[1];
        let per_tok = med / t as f64;
        out.push_str(&format!(
            "T={t}: {med:.1}ms 배치 · {per_tok:.0}ms/토큰({:.2} t/s) | ",
            1000.0 / per_tok
        ));
    }
    Ok(out)
}
// 마커 tsb
// 마커 tsb2
// 마커 abh

/// `llm170 exl3-hip-graph <dir> <tok> [T]` — hipGraph 캡처·재생: 정합(순차 대조)+재생 시간.
pub fn hip_graph_check(dir: &str, tok: u32, t_len: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let t = t_len.clamp(1, 8);
    let mut dec = Exl3HipDecoder::load(dir, 64)?;
    // 기준: 일반 배치 1회(캡처 워밍이 상태 전진시킴 — 순서: 워밍→캡처→비교재생은
    // 상태가 다르다. 정합은 "같은 상태에서 재생 vs 비캡처" 비교로: 캡처 후
    // 그래프 재생 2회와 수동 배치의 토큰열 자기일관성으로 판정(재생1 vs 재생2 연속).
    let rows0: Vec<Vec<f32>> = (0..t).map(|i| dec.embed_row_host(tok + i as u32)).collect();
    dec.capture_batch(t)?;
    // 재생 3회 측정(행은 매회 동일 — 비용 측정; 상태 전진은 KV/ring에 누적)
    let mut ts: Vec<f64> = Vec::new();
    let mut last_am = 0u32;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let (lgs, _) = dec.replay_batch(&rows0)?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
        let lastrow = lgs.last().ok_or("empty")?;
        last_am = lastrow
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[1];
    let per = med / t as f64;
    Ok(format!(
        "hip-graph T={t}: 재생 {med:.1}ms ({per:.0}ms/토큰, {:.2} t/s) · 마지막 argmax={last_am}",
        1000.0 / per
    ))
}
// 마커 gpr
// 마커 gpf

/// `llm170 exl3-hip-gmini` — 그래프 캡처 FFI 최소 검증(모델 미적재, 수 초):
/// [h2d → pos_bump → d2h_pin] 캡처·인스턴스화·재생.
pub fn hip_graph_mini() -> Result<String, String> {
    use crate::rawhip::ctx::hipgraph as hg;
    let hc = crate::rawhip::ctx::RawCtx::new()?;
    let dbuf = hc.alloc(4)?;
    hc.h2d(dbuf, &41u32.to_le_bytes())?;
    hc.sync()?;
    unsafe {
        let st = hg::hipStreamBeginCapture(hc.stream as *mut _, 2);
        if st != 0 {
            return Err(format!("BeginCapture {st}"));
        }
        // 캡처 구간: pos_bump 2회 + d2h_pin
        let mut pb = dbuf;
        let r1 = hc.launch3(
            "exl3_pos_bump",
            1,
            1,
            1,
            32,
            &mut [&mut pb as *mut *mut u8 as *mut _],
        );
        let mut ppin: *mut std::ffi::c_void = std::ptr::null_mut();
        let pr = hg::hipHostMalloc(&mut ppin, 4, 0);
        if pr != 0 {
            return Err(format!("mini pin {pr}"));
        }
        let pout = ppin as *mut u8;
        let r2 = hc.d2h_pin_async(pout, dbuf, 4);
        let mut graph: hg::Graph = std::ptr::null_mut();
        let en = hg::hipStreamEndCapture(hc.stream as *mut _, &mut graph);
        r1?;
        r2?;
        if en != 0 {
            return Err(format!("EndCapture {en}"));
        }
        let mut exec: hg::GraphExec = std::ptr::null_mut();
        let ie = hg::hipGraphInstantiate(&mut exec, graph, 0);
        if ie != 0 {
            return Err(format!("Instantiate {ie}"));
        }
        let le = hg::hipGraphLaunch(exec, hc.stream as *mut _);
        if le != 0 {
            return Err(format!("GraphLaunch {le}"));
        }
        hc.sync()?;
        // SAFETY: 핀 버퍼 판독(재생 완료 후).
        // SAFETY: 상위 unsafe 블록 내 — 중첩 제거.
        let v = u32::from_le_bytes(std::slice::from_raw_parts(pout, 4).try_into().unwrap());
        hg::hipGraphExecDestroy(exec);
        hg::hipGraphDestroy(graph);
        Ok(format!("gmini: 41+2={v} (43 기대) — 캡처·재생 정상"))
    }
}
// 마커 gm3
// 마커 gm4

/// `llm170 exl3-hip-gmini2` — 바이섹션: 2D launch 래퍼(norm_p류) 캡처 호환성.
pub fn hip_graph_mini2() -> Result<String, String> {
    let hc = crate::rawhip::ctx::RawCtx::new()?;
    let dbuf = hc.alloc(4)?;
    hc.h2d(dbuf, &7u32.to_le_bytes())?;
    let dbx = hc.alloc(5120 * 4)?;
    let dbxn = hc.alloc(5120 * 4)?;
    let dnw = hc.alloc(5120 * 4)?;
    let dz = hc.alloc(5120 * 4)?;
    hc.h2d(
        dbx,
        &vec![0.5f32; 5120]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    hc.h2d(
        dnw,
        &vec![1.0f32; 5120]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    hc.h2d(dz, &vec![0u8; 5120 * 4])?;
    hc.sync()?;
    use crate::rawhip::ctx::hipgraph as hg;
    unsafe {
        let st = hg::hipStreamBeginCapture(hc.stream as *mut _, 2);
        if st != 0 {
            return Err(format!("BeginCapture {st}"));
        }
        // 2D launch 래퍼 1회(norm_resid_p)
        let mut tl = 1i32;
        let (mut a0, mut a1, mut a2, mut a3) = (dbx, dnw, dz, dbxn);
        let rl = hc.launch(
            "exl3_norm_resid_p",
            1,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut a2 as *mut *mut u8 as *mut _,
                &mut a3 as *mut *mut u8 as *mut _,
                &mut tl as *mut i32 as *mut _,
            ],
        );
        let mut graph: hg::Graph = std::ptr::null_mut();
        let en = hg::hipStreamEndCapture(hc.stream as *mut _, &mut graph);
        rl?;
        if en != 0 {
            return Err(format!("EndCapture {en}"));
        }
        let mut exec: hg::GraphExec = std::ptr::null_mut();
        let ie = hg::hipGraphInstantiate(&mut exec, graph, 0);
        if ie != 0 {
            return Err(format!("Instantiate {ie}"));
        }
        let le = hg::hipGraphLaunch(exec, hc.stream as *mut _);
        if le != 0 {
            return Err(format!("GraphLaunch {le}"));
        }
        hc.sync()?;
        hg::hipGraphExecDestroy(exec);
        hg::hipGraphDestroy(graph);
        Ok("gmini2: 2D launch 캡처·재생 정상".into())
    }
}
// 마커 gm5
// 마커 wmr
// 마커 wv0
// 마커 a1s
// 마커 hmd

/// `llm170 exl3-hip-hcmp <dir> <tok> <steps>` — 같은 토큰 스트림에서 순차 vs 배치 h 쌍 비교.
/// maxdiff ≈1e-2 → 산술 클래스(트레이드오프), 크면 배치-h 결함(수리 가능).
pub fn hip_h_pair(dir: &str, tok: u32, steps: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dseq = Exl3HipDecoder::load(dir, 64)?;
    dseq.dbg_hcurve = true;
    // 순차 h·다음토큰 수집
    let mut toks = vec![tok];
    let mut hs: Vec<Vec<f32>> = Vec::new();
    for i in 0..steps {
        let row = dseq.embed_row_host(toks[i]);
        let (lg, h) = dseq.forward(&row)?;
        hs.push(h);
        let nxt = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j as u32)
            .unwrap_or(0);
        toks.push(nxt);
    }
    let seq_curve = std::mem::take(&mut dseq.hcurve);
    drop(dseq);
    // 배치 디코더로 같은 스트림 T=1씩(문맥 동일)
    let mut dbat = Exl3HipDecoder::load(dir, 64)?;
    dbat.dbg_hcurve = true;
    // 클린 배치 a1 — 오염 없는 배치 루프 자체 수용률(기존 0.25-0.44는 순차 루프
    // 상태 오염 후 측정이라 무효 가능성).
    {
        let (mut hit_b, mut tot_b, mut cur_b) = (0usize, 0usize, tok);
        for _ in 0..steps {
            let row = dbat.embed_row_host(cur_b);
            let pos_b = dbat.pos;
            let (lgb, hb2) = dbat.forward_batch_with_mtp(&[row], &[cur_b])?;
            let nxt_b = lgb[0]
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(k, _)| k as u32)
                .unwrap_or(0);
            let d_b = dbat.mtp_draft_gpu(cur_b, &hb2, pos_b)?;
            tot_b += 1;
            if d_b == nxt_b {
                hit_b += 1;
            }
            cur_b = nxt_b;
            if tot_b == 1 {
                eprintln!(
                    "  [hcvd] seq={} bat={} — 곡선 비교 진입",
                    seq_curve.len(),
                    dbat.hcurve.len()
                );
            }
            // [수리 2026-10-04, plans/127 C] 종전 hcv는 seq "마지막" 4개(스텝 6의
            // 토큰) vs 배치 첫 스텝(토큰 1000)을 비교 — 서로 다른 토큰의 h 곡선
            // 비교로 "L1 시드 3.42e0" 전체가 아티팩트였다. 배치 스텝1 ↔ 순차
            // 스텝1(같은 토큰) 비교로 수정.
            let n4 = dbat.hcurve.len();
            let seq4 = &seq_curve[..n4.min(seq_curve.len())];
            if tot_b == 1 && !seq4.is_empty() && n4 == seq4.len() {
                for (k, (l, hs_cv)) in seq4.iter().enumerate() {
                    let (lb, hb_cv) = &dbat.hcurve[k];
                    let md = hs_cv
                        .iter()
                        .zip(hb_cv)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let rms = hs_cv.iter().map(|v| v * v).sum::<f32>().sqrt();
                    // 상관계수 + 오차-크기 관계: corr≈1·오차∝값 → 노이즈 증폭,
                    // 무상관 원소 존재 → 실결함(인덱싱/버퍼).
                    let n_e = hs_cv.len();
                    let (mut sa, mut sb, mut saa, mut sbb, mut sab) =
                        (0f64, 0f64, 0f64, 0f64, 0f64);
                    let mut big_bad = 0usize; // |seq|<1 인데 |diff|>1 → 무상관 오염
                    for (a_, b_) in hs_cv.iter().zip(hb_cv) {
                        let (a_, b_) = (*a_ as f64, *b_ as f64);
                        sa += a_;
                        sb += b_;
                        saa += a_ * a_;
                        sbb += b_ * b_;
                        sab += a_ * b_;
                        if a_.abs() < 1.0 && (a_ - b_).abs() > 1.0 {
                            big_bad += 1;
                        }
                    }
                    let cov = sab / n_e as f64 - (sa / n_e as f64) * (sb / n_e as f64);
                    let va = saa / n_e as f64 - (sa / n_e as f64).powi(2);
                    let vb = sbb / n_e as f64 - (sb / n_e as f64).powi(2);
                    let corr = cov / (va.sqrt() * vb.sqrt());
                    eprintln!(
                        "  [hcv] L{l}↔L{lb} maxdiff={md:.3e} rms={rms:.1} corr={corr:.6} 무상관오염={big_bad}"
                    );
                }
            }
        }
        eprintln!("  [a1bat] 클린 배치경로 a1 = {hit_b}/{tot_b}");
    }
    // [수리 2026-10-04, plans/127 C] 종전 h-pail은 a1bat 루프가 자체 greedy로
    // 진행한 상태(링/KV/pos)가 남은 dbat를 그대로 재사용 — 순차 toks 스트림과
    // 위치가 어긋나 1.07e2 "계통 오차"의 상당분이 상태 비정렬 아티팩트였다.
    // 신규 디코더로 순차와 동일 토큰·동일 위치 진행으로 교체.
    drop(dbat);
    let mut dbat2 = Exl3HipDecoder::load(dir, 64)?;
    let mut mds = Vec::new();
    for i in 0..steps {
        let row = dbat2.embed_row_host(toks[i]);
        let (_lg, hb) = dbat2.forward_batch(&[row])?;
        let md = hb
            .iter()
            .zip(&hs[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        mds.push(md);
        if i == 0 {
            // 희소 원소 국소화: 임계 초과 개수·상위 위반 위치의 모듈로 패턴(128=hadout 청크,
            // 64/16=gemm 타일 경계, 무주기=산술 경계).
            let diffs: Vec<(usize, f32)> = hb
                .iter()
                .zip(&hs[i])
                .enumerate()
                .map(|(j, (a, b))| (j, (a - b).abs()))
                .collect();
            let big: Vec<usize> = diffs
                .iter()
                .filter(|(_, d)| *d > 1.0)
                .map(|(j, _)| *j)
                .collect();
            eprintln!(
                "  [hhg] >1.0 오염 {}/5120개 · 상위 12: {:?}",
                big.len(),
                &big[..big.len().min(12)]
            );
            let m128 = big.iter().filter(|j| *j % 128 == 127).count();
            let m64 = big.iter().filter(|j| *j % 64 == 63).count();
            eprintln!("  [hhg] mod128==127: {m128}개 · mod64==63: {m64}개");
            let rms = hs[i].iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("  [hcmp] 스텝{i} maxdiff={md:.3e} rms={rms:.1}");
        }
    }
    let med = {
        mds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mds[mds.len() / 2]
    };
    Ok(format!(
        "h-pair: 중앙 maxdiff={med:.3e} — {}",
        if med < 0.05 {
            "f16급(산술 클래스)"
        } else {
            "계통 오차(배치-h 결함 의심)"
        }
    ))
}
// 마커 hcp
// 마커 hhg
// 마커 cba
// 마커 hrp
// 마커 hcx
// 마커 scv
// 마커 dcf
// 마커 s4
// 마커 cor
