//! EXL3 hip 프로브 attn (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;

pub fn hip_attn_check(
    dir: &str,
    t_arg: usize,
    pos0_arg: usize,
    layer_arg: usize,
) -> Result<String, String> {
    let t_rows = t_arg;
    let pos0 = pos0_arg;
    let layer = layer_arg;
    // 프로브 KV 버퍼는 64MiB(=kvcap 1024×16층) 고정 — pos 상한 가드.
    if t_rows == 0 || t_rows > 1024 || pos0 + t_rows > 1024 || layer > 15 {
        return Err(format!(
            "hip_attn_check 인자 범위 외: t={t_rows}(1..1024) pos0={pos0}(pos0+t≤1024) layer={layer}(≤15)"
        ));
    }
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
    let mut kvc = 1024i32; // 프로브 KV 버퍼 64MiB(=cap 1024) — 계약 일치
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
            &mut kvc as *mut i32 as *mut _,
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
        // 대상 층 영역 비교(2026-10-05 수정) — 이전 버전은 항상 layer 0
        // 영역을 비교해 layer≠0에서 무의미(양쪽 0)했다.
        let lofs = layer * 1024 * 1024;
        let kvn = (pos0 + t_rows) * 1024;
        for i in 0..kvn {
            mk = mk.max((gk[lofs + i] - kc[lofs + i]).abs());
        }
        eprintln!("  [attndbg] qh maxdiff={mq:.3e} · kc(L{layer} 적립분) maxdiff={mk:.3e}");
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
            &mut kvc as *mut i32 as *mut _,
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
    let mut bad_t = vec![0usize; t_rows];
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

// ── EXL3 hip 디코드 루프(정확성 우선 조립) ── vk decode_step 로짓 대조.
