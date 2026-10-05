//! EXL3 hip 프로브 linear (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;

pub fn hip_linear_check(dir: &str, key: &str) -> Result<String, String> {
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
