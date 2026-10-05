//! EXL3 hip 프로브 gdn (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;

pub fn hip_gdn_check(dir: &str, layer_arg: usize) -> Result<String, String> {
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
