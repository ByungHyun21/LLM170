//! EXL3 hip 프로브 nr (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;

pub fn hip_nr_check(dir: &str) -> Result<String, String> {
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
