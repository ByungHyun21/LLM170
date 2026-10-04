//! vk-frame-check 본체(plans/107 W4: checks.rs에서 분할, 본문 불변).

use super::harness::{AnyModel, CheckHarness};
use crate::rawvk::vkacc::VkAcc;
use llm170_core::matmul::{FrameHost as _, FrameState as _};

/// vk-frame-check — plans/84 B: 프레임 코어(버퍼 레지스트리+엘리먼트와이즈+
/// 상주 GEMM)의 CPU 대조 검증. 각 op를 LCG 데이터로 실행해 판독 비교.
/// 각 §N 절 검증은 secN_* 함수로 분리되어 있다(호출 순서 = 보고 순서).
pub fn frame_check(path: &str, tname: &str) -> Result<String, String> {
    use std::time::Instant;
    let (h, model) = CheckHarness::with_ref(path)?;
    let w = model.w(tname)?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = h.acc;
    let t = 3usize;
    let mut seed = 0x5deece66u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let mut fails = 0usize;
    let mut report = String::new();
    let t0 = Instant::now();
    acc.frame_begin(t);

    let (r, f) = sec1_rms_rows(&acc, t, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec2_silu_scale(&acc, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec5_copy_bcast_axpy(&acc, t, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec8_frame_mm(&acc, t, &w, n_in, n_out, &xs)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec8b_frame_mm_q4k(&acc, t, &model, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec9_moe_gate(&acc, t, &model, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec9b_moe_down_ids(&acc, t, &model, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec9c_moe_tile(&acc, t, &model, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec10_attention_half(&acc, t, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    // 107 P0-6: §11-14 불변성 체커는 각자 VkAcc를 만들어 쓴다 — 외부 acc를
    // 여기서 명시 파기해 동시 디바이스 오픈을 줄인다(7 → 최대 2).
    drop(acc);
    let (r, f) = sec11_gdn_ar_chunk(&model, tname, &mut lcg)?;
    report.push_str(&r);
    fails += f;
    let _ = t0;
    Ok(format!(
        "vk-frame-check({tname}, t={t}): {} — {} ({} 실패)",
        if fails == 0 { "PASS" } else { "FAIL" },
        report,
        fails
    ))
}

/// §1 RmsRows(w_reps=2) 검증.
fn sec1_rms_rows(
    acc: &VkAcc,
    t: usize,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

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
    acc.frame_op(&llm170_core::matmul::FrameOp::RmsRows {
        x: xh,
        w: wh,
        out: oh,
        eps: 1e-5,
        n,
        w_reps: reps,
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
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "RmsRows(w_reps={reps}) max|D|={mx:.2e} {} | ",
        if ok { "OK" } else { "FAIL" }
    ));
    acc.frame_free(xh)?;
    acc.frame_free(wh)?;
    acc.frame_free(oh)?;
    Ok((report, fails))
}

/// §2 SiluDiv / §3 SiluMul / §4 Scale 검증.
fn sec2_silu_scale(acc: &VkAcc, lcg: &mut impl FnMut() -> f32) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    let n = 256usize;
    let a: Vec<f32> = (0..n).map(|_| lcg()).collect();
    let b: Vec<f32> = (0..n).map(|_| lcg()).collect();
    let ah = acc.frame_alloc(n)?;
    let bh = acc.frame_alloc(n)?;
    let oh = acc.frame_alloc(n)?;
    acc.frame_write(ah, &a)?;
    acc.frame_write(bh, &b)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::SiluDiv {
        t: ah,
        div: 320.0,
        n,
    })?;
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
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "SiluDiv max|D|={mx:.2e} {} | ",
        if ok { "OK" } else { "FAIL" }
    ));

    acc.frame_write(ah, &a)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::SiluMul {
        g: ah,
        u: bh,
        out: oh,
        n,
    })?;
    acc.frame_read(oh, &mut got)?;
    mx = 0.0;
    for i in 0..n {
        let exp = (a[i] / (1.0 + (-a[i]).exp())) as f64 * b[i] as f64;
        mx = mx.max((got[i] as f64 - exp).abs());
    }
    let ok = mx < 5e-6;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "SiluMul max|D|={mx:.2e} {} | ",
        if ok { "OK" } else { "FAIL" }
    ));

    acc.frame_write(ah, &a)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::Scale { t: ah, s: 0.5, n })?;
    acc.frame_read(ah, &mut got)?;
    mx = 0.0;
    for i in 0..n {
        mx = mx.max((got[i] as f64 - a[i] as f64 * 0.5).abs());
    }
    let ok = mx < 1e-7;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "Scale max|D|={mx:.2e} {} | ",
        if ok { "OK" } else { "FAIL" }
    ));
    acc.frame_free(ah)?;
    acc.frame_free(bh)?;
    acc.frame_free(oh)?;
    Ok((report, fails))
}

/// §5 CopyRows / §6 BcastRows / §7 AxpyScaled(t) 검증.
fn sec5_copy_bcast_axpy(
    acc: &VkAcc,
    t: usize,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    let n = 100usize;
    let src: Vec<f32> = (0..n).map(|_| lcg()).collect();
    let sh = acc.frame_alloc(n)?;
    let dh = acc.frame_alloc(2 * n)?;
    acc.frame_write(sh, &src)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::CopyRows {
        src: sh,
        dst: dh,
        src_off: 7,
        dst_off: n + 3,
        n: n - 10,
    })?;
    let mut got = vec![0f32; 2 * n];
    acc.frame_read(dh, &mut got)?;
    let mut ok = true;
    for i in 0..(n - 10) {
        if (got[n + 3 + i] - src[7 + i]).abs() > 1e-7 {
            ok = false;
            break;
        }
    }
    if !ok {
        fails += 1;
    }
    report.push_str(&format!("CopyRows {} | ", if ok { "OK" } else { "FAIL" }));

    let bh2 = acc.frame_alloc(n * t)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::BcastRows {
        src: sh,
        dst: bh2,
        n,
        rows: t,
    })?;
    acc.frame_read(bh2, &mut got)?;
    let _ = &mut got;
    let mut got2 = vec![0f32; n * t];
    acc.frame_read(bh2, &mut got2)?;
    ok = true;
    for r in 0..t {
        for i in 0..n {
            if (got2[r * n + i] - src[i]).abs() > 1e-7 {
                ok = false;
            }
        }
    }
    if !ok {
        fails += 1;
    }
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
    acc.frame_op(&llm170_core::matmul::FrameOp::AxpyScaled {
        y: yh,
        x: xh2,
        s: ssh,
        n: t * per,
    })?;
    let mut got3 = vec![0f32; t * per];
    acc.frame_read(yh, &mut got3)?;
    let mut mx = 0f64;
    for j in 0..t * per {
        let exp = y[j] as f64 + xx[j] as f64 * ss[j / per] as f64;
        mx = mx.max((got3[j] as f64 - exp).abs());
    }
    let aok = mx < 1e-6;
    if !aok {
        fails += 1;
    }
    report.push_str(&format!(
        "AxpyScaled(t={t}) max|D|={mx:.2e} {}",
        if aok { "OK" } else { "FAIL" }
    ));
    acc.frame_free(sh)?;
    acc.frame_free(dh)?;
    acc.frame_free(bh2)?;
    acc.frame_free(yh)?;
    acc.frame_free(xh2)?;
    acc.frame_free(ssh)?;
    Ok((report, fails))
}

/// §8 frame_mm — 상주 quant+GEMM vs CPU 디양자화 내적.
fn sec8_frame_mm(
    acc: &VkAcc,
    t: usize,
    w: &llm170_core::matmul::Weight<'_>,
    n_in: usize,
    n_out: usize,
    xs: &[Vec<f32>],
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    let xh = acc.frame_alloc(n_in * t)?;
    let oh = acc.frame_alloc(n_out * t)?;
    let mut flat = Vec::with_capacity(n_in * t);
    for row in xs {
        flat.extend_from_slice(row);
    }
    acc.frame_write(xh, &flat)?;
    acc.frame_mm(xh, w, oh, t)?;
    let mut got = vec![0f32; n_out * t];
    acc.frame_read(oh, &mut got)?;
    let mut mx = 0f64;
    let mut ref_row = vec![0f32; n_in];
    for (j, x) in xs.iter().enumerate() {
        for r in 0..n_out.min(16) {
            llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            mx = mx.max((dot as f64 - got[j * n_out + r] as f64).abs());
        }
    }
    let ok = mx < 5e-3;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "frame_mm max|D|={mx:.2e} {}",
        if ok { "OK" } else { "FAIL" }
    ));
    acc.frame_free(xh)?;
    acc.frame_free(oh)?;
    Ok((report, fails))
}

/// §8b frame_mm q4_K 밀집 (plans/88 P2): mode-1 타일 산술 분리 검증 —
/// 그룹화(perm/rowexp) 없이 타일 커널 자체의 CPU 대조.
fn sec8b_frame_mm_q4k(
    acc: &VkAcc,
    t: usize,
    model: &AnyModel,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    if let AnyModel::Q4(m) = model
        && let Ok(w4k) = m.w4("blk.0.ffn_gate_shexp.weight")
    {
        let ni = w4k.n_in as usize;
        let no = w4k.n_out as usize;
        let xs4: Vec<Vec<f32>> = (0..t).map(|_| (0..ni).map(|_| lcg()).collect()).collect();
        let xh = acc.frame_alloc(ni * t)?;
        let oh = acc.frame_alloc(no * t)?;
        let mut flat = Vec::with_capacity(ni * t);
        for row in &xs4 {
            flat.extend_from_slice(row);
        }
        acc.frame_write(xh, &flat)?;
        acc.frame_mm(xh, &w4k, oh, t)?;
        let mut got = vec![0f32; no * t];
        acc.frame_read(oh, &mut got)?;
        let mut mx = 0f64;
        let mut ref_row = vec![0f32; ni];
        for (j, x) in xs4.iter().enumerate() {
            for r in 0..no.min(12) {
                llm170_core::quant::dequant_row(
                    w4k.ty,
                    w4k.data,
                    r as u64,
                    ni as u64,
                    &mut ref_row,
                );
                let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                mx = mx.max((dot as f64 - got[j * no + r] as f64).abs());
            }
        }
        let ok = mx < 5e-3;
        if !ok {
            fails += 1;
        }
        report.push_str(&format!(
            "| frame_mm-q4k max|D|={mx:.2e} {}",
            if ok { "OK" } else { "FAIL" }
        ));
        acc.frame_free(xh)?;
        acc.frame_free(oh)?;
    }
    Ok((report, fails))
}

/// §9 MoE: top10 → 그룹 GEMM → 가중합 (게이트 가중, k=10).
fn sec9_moe_gate(
    acc: &VkAcc,
    t: usize,
    model: &AnyModel,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    let k = 10usize;
    // FN 게이트 가중은 q4_K 스택(대부분 층) — 스택 텐서 하나로 검증.
    let wg = match model {
        AnyModel::Q4(m) => m
            .w4("blk.0.ffn_gate_exps.weight")
            .map_err(|e| e.to_string())?,
        AnyModel::Q35(m) => m.w("blk.0.ffn_gate.weight").ok_or("텐서 없음")?,
    };
    let n_in_m = wg.n_in as usize;
    let ne = match model {
        AnyModel::Q4(_) => 512usize,
        AnyModel::Q35(_) => 1usize,
    };
    // 스택 텐서: 전문가당 폭만 출력에 쓴다(frame_moe_gemm 규약).
    let n_out_m = wg.n_out as usize / ne;
    if ne == 512 {
        let route: Vec<f32> = (0..t * ne).map(|_| lcg() * 4.0).collect();
        let mxs: Vec<Vec<f32>> = (0..t * k)
            .map(|_| (0..n_in_m).map(|_| lcg()).collect())
            .collect();
        let rh = acc.frame_alloc(t * ne)?;
        let idh = acc.frame_alloc(t * k)?;
        let wth = acc.frame_alloc(t * k)?;
        let mxh = acc.frame_alloc(t * k * n_in_m)?;
        let mgh = acc.frame_alloc(t * k * n_out_m)?;
        let outh = acc.frame_alloc(t * n_out_m)?;
        acc.frame_write(rh, &route)?;
        let mut flat2 = Vec::with_capacity(t * k * n_in_m);
        for row in &mxs {
            flat2.extend_from_slice(row);
        }
        acc.frame_write(mxh, &flat2)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
            route: rh,
            ids: idh,
            wt: wth,
            n_exp: ne,
            k_sel: k,
        })?;
        let mut ids_g = vec![0u32; t * k];
        {
            acc.frame_sync();
            let g = acc_frame_ptr(acc, idh);
            unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t * k) };
        }
        acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::MoeWeightedSum {
            ys: mgh,
            wt: wth,
            out: outh,
            k,
            n: n_out_m,
        })?;
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
                if ids_g[tok * k + j] as usize != e {
                    mx = mx.max(1.0);
                }
            }
            let per_exp = n_out_m;
            let mut ref_row = vec![0f32; n_in_m];
            for j in 0..per_exp.min(8) {
                let mut acc2 = 0f64;
                for (ki, &e) in sel.iter().enumerate() {
                    llm170_core::quant::dequant_row(
                        wg.ty,
                        wg.data,
                        (e * per_exp + j) as u64,
                        n_in_m as u64,
                        &mut ref_row,
                    );
                    let dot: f32 = ref_row
                        .iter()
                        .zip(mxs[tok * k + ki].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    acc2 += dot as f64 * (wsel[ki] / wsum) as f64;
                }
                let d = (got[tok * n_out_m + j] as f64 - acc2).abs();
                mx = mx.max(d);
            }
        }
        let ok = mx < 3e-2;
        if !ok {
            fails += 1;
        }
        report.push_str(&format!(
            "| MoE(k={k}) max|D|={mx:.2e} {}",
            if ok { "OK" } else { "FAIL" }
        ));
        for h in [rh, idh, wth, mxh, mgh, outh] {
            acc.frame_free(h)?;
        }
    }
    Ok((report, fails))
}

/// §9b MoE down direct-ids (plans/88 P1): q5_1 스택 t=1 — ids 직판독 경로의
/// CPU 대조. 행마다 전문가가 다르고 x 행도 행마다 독립이다
/// (게이트/up의 브로드캐스트 입력과 달리 행 r 을 정확히 읽어야 한다).
fn sec9b_moe_down_ids(
    acc: &VkAcc,
    t: usize,
    model: &AnyModel,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    if let AnyModel::Q4(_) = model {
        let k = 10usize;
        let ne = 512usize;
        let wd = match model {
            AnyModel::Q4(m) => m
                .w4("blk.0.ffn_down_exps.weight")
                .map_err(|e| e.to_string())?,
            AnyModel::Q35(_) => unreachable!(),
        };
        let n_in_d = wd.n_in as usize;
        let n_out_d = wd.n_out as usize / ne;
        let route: Vec<f32> = (0..ne).map(|_| lcg() * 4.0).collect();
        let xs: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..n_in_d).map(|_| lcg()).collect())
            .collect();
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
            route: rh,
            ids: idh,
            wt: wth,
            n_exp: ne,
            k_sel: k,
        })?;
        let mut ids_g = vec![0u32; k];
        {
            acc.frame_sync();
            let g = acc_frame_ptr(acc, idh);
            unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), k) };
        }
        acc.frame_moe_gemm(mxh, &wd, idh, mgh, ne, k)?;
        // plans/98 통제 비교: 동일 형상 웜 5회 타이밍(llama 체커와 대칭).
        {
            let n = 5u32;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = acc.frame_moe_gemm(mxh, &wd, idh, mgh, ne, k);
            }
            acc.frame_sync();
            eprintln!(
                "[mtc-timing] q4_K t={t}: {:.2}ms/회",
                t0.elapsed().as_secs_f64() * 1e3 / f64::from(n)
            );
        }
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
                    wd.ty,
                    wd.data,
                    (e * n_out_d + j) as u64,
                    n_in_d as u64,
                    &mut ref_row,
                );
                let dot: f32 = ref_row.iter().zip(xs[r].iter()).map(|(a, b)| a * b).sum();
                let d = (got[r * n_out_d + j] as f64 - dot as f64).abs();
                mx = mx.max(d);
            }
        }
        let ok = mx < 3e-2;
        if !ok {
            fails += 1;
        }
        report.push_str(&format!(
            "| MoE-down-ids(k={k}) max|D|={mx:.2e} {}",
            if ok { "OK" } else { "FAIL" }
        ));
        for h in [rh, idh, wth, mxh, mgh] {
            acc.frame_free(h)?;
        }
    }
    Ok((report, fails))
}

/// §9c MoE 타일 대량행 (plans/88 P2): t=210·k=10 → rows=2100 — 디바이스
/// 그룹화+타일 경로의 CPU 대조. 소형(§9 t=3)은 direct-ids만 지나가
/// 않으므로 대량 행이 필요하다.
fn sec9c_moe_tile(
    acc: &VkAcc,
    _t: usize,
    model: &AnyModel,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    if let AnyModel::Q4(_) = model {
        let k = 10usize;
        let ne = 512usize;
        let t2 = std::env::var("LLM170_T2")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(210usize);
        let wg = match model {
            AnyModel::Q4(m) => m
                .w4("blk.0.ffn_gate_exps.weight")
                .map_err(|e| e.to_string())?,
            AnyModel::Q35(_) => unreachable!(),
        };
        let n_in_m = wg.n_in as usize;
        let n_out_m = wg.n_out as usize / ne;
        // t2토큰 × ne 라우트 — 균등 랜덤이면 대부분의 전문가가 비게 되어
        // rows=2100이 희소 행을 만든다(실측 결함 재현 조건).
        let route: Vec<f32> = (0..t2 * ne).map(|_| lcg() * 4.0).collect();
        let mxs: Vec<Vec<f32>> = (0..t2 * k)
            .map(|_| (0..n_in_m).map(|_| lcg()).collect())
            .collect();
        let rh = acc.frame_alloc(t2 * ne)?;
        let idh = acc.frame_alloc(t2 * k)?;
        let wth = acc.frame_alloc(t2 * k)?;
        let mxh = acc.frame_alloc(t2 * k * n_in_m)?;
        let mgh = acc.frame_alloc(t2 * k * n_out_m)?;
        acc.frame_write(rh, &route)?;
        let mut flat = Vec::with_capacity(t2 * k * n_in_m);
        for row in &mxs {
            flat.extend_from_slice(row);
        }
        acc.frame_write(mxh, &flat)?;
        acc.frame_begin(t2);
        acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
            route: rh,
            ids: idh,
            wt: wth,
            n_exp: ne,
            k_sel: k,
        })?;
        acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
        // [I5 수리 2026-10-04] 여기의 두 번째 frame_begin(t)은 §9b 템플릿 잔존이다.
        // frame_read는 frame_t를 쓰지 않고, begin_batch의 site depth만 +1해
        // 첫 frame_read의 end_batch_wait를 no-op로 만든다(depth>0 가드) — 즉
        // 미제출 타일 출력을 스테일 판독해 MoE-tile-2100가 8.07e-1(=|dot|)
        // FAIL로 결정적으로 오염됐다. §9b는 begin 사이 frame_sync로 균형을
        // 맞추는데 이 섹션은 그 동기화가 없었다. 삭제 — 단일 begin이면
        // frame_read의 드레인(depth 1→0)이 실제로 실행된다.
        let mut got = vec![0f32; t2 * k * n_out_m];
        acc.frame_read(mgh, &mut got)?;
        let mut ids_g = vec![0u32; t2 * k];
        {
            acc.frame_sync();
            let g = acc_frame_ptr(acc, idh);
            unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t2 * k) };
        }
        // CPU 참조: (토큰,슬롯) 행별 디양자 내적 — 순열과 무관하게 행 자체가 맞는지.
        let mut mx = 0f64;
        let mut ref_row = vec![0f32; n_in_m];
        for row in 0..t2 * k {
            let e = ids_g[row] as usize;
            for j in 0..n_out_m.min(4) {
                llm170_core::quant::dequant_row(
                    wg.ty,
                    wg.data,
                    (e * n_out_m + j) as u64,
                    n_in_m as u64,
                    &mut ref_row,
                );
                let dot: f32 = ref_row
                    .iter()
                    .zip(mxs[row].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let d = (got[row * n_out_m + j] as f64 - dot as f64).abs();
                mx = mx.max(d);
            }
        }
        let ok = mx < 3e-2;
        if !ok {
            fails += 1;
        }
        report.push_str(&format!(
            "| MoE-tile-2100 max|D|={mx:.2e} {}",
            if ok { "OK" } else { "FAIL" }
        ));
        for h in [rh, idh, wth, mxh, mgh] {
            acc.frame_free(h)?;
        }
    }
    Ok((report, fails))
}

/// §10 어텐션 반쪽: HcGateMean/HcCombine/NormGated/GdnBetaG/Sigmoid/Split3.
fn sec10_attention_half(
    acc: &VkAcc,
    t: usize,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    // [I5 수리 2026-10-04] 이 섹션의 커널들(GdnBetaG dr=n_h/t_cur 등)은
    // frame_t를 사용한다. 과거에는 sec9c의 두 번째 frame_begin(t)이 우연히
    // frame_t=3을 누출해 통과했던 숨은 결합이었다(sec9c 수리로 노출) —
    // 섹션 스스로 begin으로 frame_t를 명시한다. 산술 불변(동일 t=3).
    acc.frame_begin(t);

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
    acc.frame_op(&llm170_core::matmul::FrameOp::HcGateMean {
        xn: xnh,
        gate: gth,
        out: mkh,
        hc,
        n,
    })?;
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
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| HcGateMean {mx:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

    // HcCombine — res 초기화 후 += 검증
    let res0: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let resh = acc.frame_alloc(t * hc * n)?;
    let inj: Vec<f32> = (0..t * hc).map(|_| lcg() * 2.0).collect();
    let ijh = acc.frame_alloc(t * hc)?;
    acc.frame_write(resh, &res0)?;
    acc.frame_write(ijh, &inj)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::HcCombine {
        res: resh,
        out: mkh,
        inj: ijh,
        hc,
        n,
        total: 0,
    })?;
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
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| HcCombine {mx:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

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
    acc.frame_op(&llm170_core::matmul::FrameOp::NormGated {
        o: o3h,
        z: z3h,
        w: w3h,
        out: n3h,
        eps: 1e-5,
        d,
        n_h: nh,
    })?;
    let mut ng = vec![0f32; t * nh * d];
    acc.frame_read(n3h, &mut ng)?;
    mx = 0.0;
    for row in 0..t * nh {
        let s: f64 = (0..d).map(|i| (o3[row * d + i] as f64).powi(2)).sum();
        let inv = 1.0 / (s / d as f64 + 1e-5).sqrt();
        for i in 0..d {
            let exp = o3[row * d + i] as f64
                * inv
                * w3[(row % nh) * d + i] as f64
                * (1.0 / (1.0 + (-z3[row * d + i] as f64).exp()));
            mx = mx.max((ng[row * d + i] as f64 - exp).abs());
        }
    }
    let ok = mx < 5e-6;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| NormGated {mx:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

    // GdnBetaG — dt_rank=6, n_h=6·t
    let dr = 6usize;
    let nh2 = dr * t;
    let b2: Vec<f32> = (0..nh2).map(|_| lcg()).collect();
    let a2: Vec<f32> = (0..nh2).map(|_| lcg()).collect();
    let dtb: Vec<f32> = (0..dr).map(|_| lcg()).collect();
    let sa: Vec<f32> = (0..dr).map(|_| lcg()).collect();
    let (b2h, a2h, dth, sah, bgh) = (
        acc.frame_alloc(nh2)?,
        acc.frame_alloc(nh2)?,
        acc.frame_alloc(dr)?,
        acc.frame_alloc(dr)?,
        acc.frame_alloc(nh2 * 2)?,
    );
    acc.frame_write(b2h, &b2)?;
    acc.frame_write(a2h, &a2)?;
    acc.frame_write(dth, &dtb)?;
    acc.frame_write(sah, &sa)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::GdnBetaG {
        b: b2h,
        a: a2h,
        dtb: dth,
        sa: sah,
        bg: bgh,
        n_h: nh2,
    })?;
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
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| GdnBetaG {mx:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

    // Sigmoid + Split3
    let v4: Vec<f32> = (0..128).map(|_| lcg() * 3.0).collect();
    let v4h = acc.frame_alloc(128)?;
    acc.frame_write(v4h, &v4)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::Sigmoid { t: v4h, n: 128 })?;
    let mut sv = vec![0f32; 128];
    acc.frame_read(v4h, &mut sv)?;
    mx = 0.0;
    for j in 0..128 {
        mx = mx.max((sv[j] as f64 - (1.0 / (1.0 + (-v4[j] as f64).exp()))).abs());
    }
    let ok = mx < 5e-7;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| Sigmoid {mx:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

    let (n0, n1, n2) = (10usize, 6usize, 8usize);
    let tot = n0 + n1 + n2;
    let s3: Vec<f32> = (0..t * tot).map(|_| lcg()).collect();
    let s3h = acc.frame_alloc(t * tot)?;
    let (d0h, d1h, d2h) = (
        acc.frame_alloc(t * n0)?,
        acc.frame_alloc(t * n1)?,
        acc.frame_alloc(t * n2)?,
    );
    acc.frame_write(s3h, &s3)?;
    acc.frame_op(&llm170_core::matmul::FrameOp::Split3 {
        src: s3h,
        d0: d0h,
        d1: d1h,
        d2: d2h,
        n0,
        n1,
        n2,
    })?;
    let mut g0 = vec![0f32; t * n0];
    let mut g1 = vec![0f32; t * n1];
    let mut g2 = vec![0f32; t * n2];
    acc.frame_read(d0h, &mut g0)?;
    acc.frame_read(d1h, &mut g1)?;
    acc.frame_read(d2h, &mut g2)?;
    let mut ok = true;
    for ti in 0..t {
        for j in 0..n0 {
            if (g0[ti * n0 + j] - s3[ti * tot + j]).abs() > 1e-7 {
                ok = false;
            }
        }
        for j in 0..n1 {
            if (g1[ti * n1 + j] - s3[ti * tot + n0 + j]).abs() > 1e-7 {
                ok = false;
            }
        }
        for j in 0..n2 {
            if (g2[ti * n2 + j] - s3[ti * tot + n0 + n1 + j]).abs() > 1e-7 {
                ok = false;
            }
        }
    }
    if !ok {
        fails += 1;
    }
    report.push_str(&format!("| Split3 {}", if ok { "OK" } else { "FAIL" }));
    for h in [
        xnh, gth, mkh, resh, ijh, o3h, z3h, w3h, n3h, b2h, a2h, dth, sah, bgh, v4h, s3h, d0h, d1h,
        d2h,
    ] {
        acc.frame_free(h)?;
    }
    Ok((report, fails))
}

/// §12b MoE 청크 불변성 — 동일 64토큰, 1호출 vs 4×16 호출.
fn sec12b_moe_chunk(model: &AnyModel) -> Result<(String, usize), String> {
    let k10 = 10usize;
    let tt = 64usize;
    let route64: Vec<f32> = {
        let mut s3 = 0xc0deu64;
        (0..tt * 512)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s3 >> 33) as f32 / 2147483648.0 - 0.5) * 4.0
            })
            .collect()
    };
    // down 경로도 같은 시험 — n_in=640(행 760B, 패딩 대상).
    let wg2 = match model {
        AnyModel::Q4(m) => m
            .w4("blk.0.ffn_down_exps.weight")
            .map_err(|e| e.to_string())?,
        AnyModel::Q35(m) => m.w("blk.0.ffn_down.weight").ok_or("텐서 없음")?,
    };
    let n_g = wg2.n_in as usize;
    let per_out = wg2.n_out as usize / 512;
    let mx_rows: Vec<f32> = {
        let mut s3 = 0x5a5au64;
        (0..tt * n_g)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
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
                unsafe {
                    std::ptr::copy_nonoverlapping(rbuf.as_ptr(), b.ptr as *mut f32, rbuf.len())
                };
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
                                n_g,
                            );
                        }
                    }
                }
            }
            acc5.frame_begin(c);
            acc5.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
                route: rh,
                ids: idh,
                wt: wth,
                n_exp: 512,
                k_sel: k10,
            })?;
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
    for i in 0..m1.len() {
        mm = mm.max((m1[i] as f64 - m2[i] as f64).abs());
    }
    eprintln!("[moech] 게이트 GEMM 청크 불변 max|D|={mm:.1e}");
    sec12c_router_fallback_chunk(model)?;
    Ok((String::new(), 0))
}

/// §12c f32 라우터 폴백 그룹 청크 불변성 — 실제 ffn_gate_inp.
fn sec12c_router_fallback_chunk(model: &AnyModel) -> Result<(String, usize), String> {
    let wr = match model {
        AnyModel::Q4(m) => m
            .w4("blk.0.ffn_gate_inp.weight")
            .map_err(|e| e.to_string())?,
        AnyModel::Q35(m) => m.w("blk.0.ffn_gate.weight").ok_or("텐서 없음")?,
    };
    let nr = wr.n_in as usize;
    let nout_r = wr.n_out as usize;
    let mixr: Vec<f32> = {
        let mut s3 = 0xfeedfaceu64;
        (0..64 * nr)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
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
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mixr[p0 * nr..].as_ptr(),
                        b.ptr as *mut f32,
                        c * nr,
                    )
                };
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
    for i in 0..r1.len() {
        mr = mr.max((r1[i] as f64 - r2[i] as f64).abs());
    }
    eprintln!("[rtech] f32 라우터 폴백 청크 불변 max|D|={mr:.1e}");
    Ok((String::new(), 0))
}

/// §13 GdnBetaG 청크 불변성 (실측 형상 dr=48).
fn sec13_gdn_beta_g_chunk() -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
    let dr = 48usize;
    let tot = 64usize;
    let (bsrc, asrc): (Vec<f32>, Vec<f32>) = {
        let mut s3 = 0x9999u64;
        let mut f1 = || {
            s3 = s3
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s3 >> 33) as f32 / 2147483648.0 - 0.5
        };
        (
            (0..dr * tot).map(|_| f1()).collect(),
            (0..dr * tot).map(|_| f1()).collect(),
        )
    };
    let dtbv: Vec<f32> = {
        let mut s3 = 0x8888u64;
        (0..dr)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
    };
    let sav: Vec<f32> = {
        let mut s3 = 0x7777u64;
        (0..dr)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
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
            acc4.frame_op(&llm170_core::matmul::FrameOp::GdnBetaG {
                b: bh,
                a: ah,
                dtb: dth,
                sa: sah,
                bg: gh,
                n_h: dr * tt,
            })?;
            let mut part = vec![0f32; dr * 2 * tt];
            acc4.frame_read(gh, &mut part)?;
            out[p0 * dr * 2..(p0 + tt) * dr * 2].copy_from_slice(&part);
            acc4.frame_free(bh)?;
            acc4.frame_free(ah)?;
            acc4.frame_free(gh)?;
        }
        Ok(out)
    };
    let g1 = run_bg(64).map_err(|e| e.to_string())?;
    let g2 = run_bg(16).map_err(|e| e.to_string())?;
    let mut mb = 0f64;
    for i in 0..g1.len() {
        mb = mb.max((g1[i] as f64 - g2[i] as f64).abs());
    }
    eprintln!("[gbg] chunk64 vs chunk16 max|D|={mb:.1e}");
    let ok = mb < 1e-6;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| GdnBetaGChunk {mb:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));
    Ok((report, fails))
}

/// §14 QSA 디코드 선택(qsa_sel_dev) — 호스트 top-k 비트 일치.
/// 풀 사전 적립(append) + 디코드 1토큰 선택. 호스트 참조는 셰이더와
/// 동일 산술열(f64 순차 rms, f64 회전, 4누산 도트, 정수 순위).
fn sec14_qsa_sel() -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
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
        s3 = s3
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            .qsa_sel_dev(
                full, 0, iqh, ikh, 1, n_bulk, ih, dm, r, top_k, &iqw, &ikw, &cs, eps,
            )
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
    let (d_idx, d_off) = acc5
        .qsa_sel_readback(sd, od, list_len)
        .map_err(|e| e.to_string())?;
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
    Ok((report, fails))
}

/// §15 shexp_gu/shexp_da — 디코드 t=1 융합 vs CPU 디양자화 참조.
/// (qwen4exp 전용 — q35는 SKIP)
fn sec15_shexp(
    acc: &VkAcc,
    model: &AnyModel,
    tname: &str,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
    if let AnyModel::Q4(m4) = model {
        use llm170_core::matmul::EwOps as _;
        let il: usize = tname
            .strip_prefix("blk.")
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let wg = m4
            .w4(&format!("blk.{il}.ffn_gate_shexp.weight"))
            .map_err(|e| e.to_string())?;
        let wu = m4
            .w4(&format!("blk.{il}.ffn_up_shexp.weight"))
            .map_err(|e| e.to_string())?;
        let wd = m4
            .w4(&format!("blk.{il}.ffn_down_shexp.weight"))
            .map_err(|e| e.to_string())?;
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
        acc.shexp_gu(xh, &wg, &wu, hh, n1, n2)
            .map_err(|e| e.to_string())?;
        acc.shexp_da(hh, &wd, sh, mh, n1, n2)
            .map_err(|e| e.to_string())?;
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
    Ok((report, fails))
}

/// §12 GdnConv 청크 불변성 — conv 링 상태 + 출력 (§13-15 포함).
fn sec12_gdn_conv_chunk(
    model: &AnyModel,
    tname: &str,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
    let mut s2 = 0u64; // 원문은 §11의 lc2 클로저 이동 후 잔여 시드 — 더미 기록 재현
    let ch = 48usize;
    let ck = 4usize;
    let conv_total = 8usize;
    let qkv8: Vec<f32> = (0..conv_total * ch)
        .map(|_| {
            s2 = 0u64.wrapping_add(0); // (클로저 이동으로 새 랜덤은 불가 — 상수 시드 재사용)
            0.0
        })
        .collect();
    let _ = (qkv8, s2);
    let conv_src: Vec<f32> = {
        let mut s3 = 0xfeedu64;
        (0..conv_total * ch)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
    };
    let cw: Vec<f32> = {
        let mut s3 = 0xbeefu64;
        (0..ch * ck)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
    };
    let st0c: Vec<f32> = {
        let mut s3 = 0x1234u64;
        (0..(ck - 1) * ch)
            .map(|_| {
                s3 = s3
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s3 >> 33) as f32 / 2147483648.0 - 0.5
            })
            .collect()
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
                qkv: inb,
                cw: cwb,
                state: stb,
                out: ob,
                ch,
                k: ck,
                t_len: tt,
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
    for i in 0..c1.len() {
        mo = mo.max((c1[i] as f64 - c2[i] as f64).abs());
    }
    let mut ms = 0f64;
    for i in 0..k1.len() {
        ms = ms.max((k1[i] as f64 - k2[i] as f64).abs());
    }
    let mut mdet = 0f64;
    for i in 0..k1.len() {
        mdet = mdet.max((k1[i] as f64 - k3[i] as f64).abs());
    }
    eprintln!("[gcv] conv out={mo:.1e} st={ms:.1e} 결정론 st={mdet:.1e}");
    let ok = mo < 1e-6 && ms < 1e-6;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| GdnConvChunk out={mo:.1e} st={ms:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

    let (r, f) = sec13_gdn_beta_g_chunk()?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec14_qsa_sel()?;
    report.push_str(&r);
    fails += f;
    // 107 P0-6: §15+ 는 §1-10 acc와 무관한 신규 버퍼만 쓴다 —
    // 불변성 체커 디바이스가 닫힌 뒤 재생성.
    let acc = VkAcc::new()?;
    let (r, f) = sec15_shexp(&acc, model, tname, lcg)?;
    report.push_str(&r);
    fails += f;
    Ok((report, fails))
}

/// §16 GdnConv 절대 대조(t=1 순차판) — CPU 링 산술과 직접 비교.
/// (plans/86 §1: §12는 청크 불변성만 — t<k-1 순차 커널은 커버 밖이었다)
fn sec16_gdn_conv_t1(acc: &VkAcc) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
    let (ch, ck, steps) = (48usize, 4usize, 3usize);
    let mut s4 = 0x51ceu64;
    let mut lc4 = move || {
        s4 = s4
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            src,
            dst: inb,
            src_off: t * ch,
            dst_off: 0,
            n: ch,
        })?;
        acc.frame_op(&llm170_core::matmul::FrameOp::GdnConv {
            qkv: inb,
            cw: cwb,
            state: stb,
            out: ob,
            ch,
            k: ck,
            t_len: 1,
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
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| GdnConvT1 out={mo:.1e} st={ms:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));
    for h in [src, cwb, stb, inb, ob] {
        acc.frame_free(h)?;
    }
    Ok((report, fails))
}

/// §17 GDN AR 절대 대조(t=1) — 전치 상태 + CPU 미러.
fn sec17_gdn_ar_t1(acc: &VkAcc) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
    let (hk, hv, d) = (2usize, 4usize, 128usize);
    let (ks, vs) = (hk * d, hv * d);
    let mut s5 = 0x600du64;
    let mut lc5 = move || {
        s5 = s5
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    for x in st.iter_mut() {
        *x = lc5() * 0.05;
    }
    // CPU 미러(사전 스케일 q) — gdn.rs gdn_ar_batch 산술 동일열.
    let st_in = st.clone();
    let mut st_cpu = st_in.clone();
    let mut o_cpu = vec![0f32; vs];
    for h in 0..hv {
        let kh = h % hk;
        let (qb, kb, vb) = (
            &qs[kh * d..kh * d + d],
            &kk[kh * d..kh * d + d],
            &vv[h * d..h * d + d],
        );
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
    for i in 0..vs {
        mo = mo.max((og[i] as f64 - o_cpu[i] as f64).abs());
    }
    for i in 0..st_cpu.len() {
        ms = ms.max((sg[i] as f64 - st_cpu[i] as f64).abs());
    }
    eprintln!("[gnarabs] out={mo:.1e} st={ms:.1e}");
    let ok = mo < 1e-4 && ms < 1e-4;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| GdnART1 out={mo:.1e} st={ms:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));
    for h in [qh, kh2, vh, bh, sh, oh] {
        acc.frame_free(h)?;
    }
    Ok((report, fails))
}

/// §18 헤드 체인 절대 대조(t=1) — 실가중 output_hc + output GEMM.
fn sec18_head_chain(acc: &VkAcc, model: &AnyModel) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();
    if let AnyModel::Q4(m4) = model {
        let hp = &m4.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let w_norm = m4
            .f32_vec4("output_hc_norm.weight")
            .map_err(|e| e.to_string())?;
        let w_down = m4.w4("output_hc_down.weight").map_err(|e| e.to_string())?;
        let w_up = m4.w4("output_hc_up.weight").map_err(|e| e.to_string())?;
        let w_out = m4.w4("output.weight").map_err(|e| e.to_string())?;
        let mut s6 = 0x7a11u64;
        let mut lc6 = move || {
            s6 = s6
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
            x: rh,
            w: wn,
            out: xnh,
            eps: hp.eps,
            n,
            w_reps: hc,
        })?;
        acc.frame_mm(xnh, &w_down, loh, 1)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::SiluDiv {
            t: loh,
            div: hc as f32,
            n: w_down.n_out as usize,
        })?;
        acc.frame_mm(loh, &w_up, gah, 1)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::HcGateMean {
            xn: xnh,
            gate: gah,
            out: hih,
            hc,
            n,
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
        for v in hlo.iter_mut() {
            *v = llm170_core::ops::silu(*v / hc as f32);
        }
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
        for i in 0..16 {
            mx = mx.max((lg[i] as f64 - hlg[i] as f64).abs());
        }
        let scale = hlg.iter().fold(0f32, |a, &v| a.max(v.abs())) as f64;
        eprintln!("[headabs] max|D|={mx:.3e} (scale={scale:.1})");
        // 3연속 W4A8 GEMM + silu 증폭 — logit-diff.sh 의 MMA 클래스(maxrel<3e-2)와 동일 기준.
        let ok = mx / scale.max(1.0) < 3e-2;
        if !ok {
            fails += 1;
        }
        report.push_str(&format!(
            "| HeadChain {mx:.1e} {}",
            if ok { "OK" } else { "FAIL" }
        ));
        for h in [wn, rh, xnh, loh, gah, hih, lgh] {
            acc.frame_free(h)?;
        }
    }
    Ok((report, fails))
}

/// §11 GDN AR 청크 불변성 — 단일 t=8 대 2×t=4, 최종 상태·출력 비교 (§12-18 포함).
fn sec11_gdn_ar_chunk(
    model: &AnyModel,
    tname: &str,
    lcg: &mut impl FnMut() -> f32,
) -> Result<(String, usize), String> {
    let mut fails = 0usize;
    let mut report = String::new();

    let hv = 4usize;
    let hk = 2usize;
    // 커널 레이아웃: 상태 u행 = kdim 128(32레인×4) — d≥128 필수(hip 규약).
    let d = 128usize;
    let (ks, vs) = (hk * d, hv * d);
    let full = 8usize;
    let mut s2 = 0xabcdu64;
    let mut lc2 = move || {
        s2 = s2
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            for h in [qs, ksb, vsb, bsb, osb] {
                acc2.frame_free(h)?;
            }
        }
        let mut stf = vec![0f32; hv * d * d];
        acc2.frame_read(sh, &mut stf)?;
        Ok((got, stf))
    };
    // t=1 결정론
    let (_a, s0a) = run_case(1, 1).map_err(|e| e.to_string())?;
    let (_b, s0b) = run_case(1, 1).map_err(|e| e.to_string())?;
    let mut m0 = 0f64;
    for i in 0..s0a.len() {
        m0 = m0.max((s0a[i] as f64 - s0b[i] as f64).abs());
    }
    eprintln!("[gnar] t=1 상태 결정론={m0:.1e}");
    // t=8 결정론 + 청크 불변

    let (r, f) = sec12b_moe_chunk(model)?;
    report.push_str(&r);
    fails += f;
    let (o1, s1) = run_case(8, 8).map_err(|e| e.to_string())?;
    let (o1b, s1b) = run_case(8, 8).map_err(|e| e.to_string())?;
    let (o2, s2v) = run_case(4, 8).map_err(|e| e.to_string())?;
    let mut mo = 0f64;
    for i in 0..o1.len() {
        mo = mo.max((o1[i] as f64 - o2[i] as f64).abs());
    }
    let mut ms = 0f64;
    for i in 0..s1.len() {
        ms = ms.max((s1[i] as f64 - s2v[i] as f64).abs());
    }
    let mut md = 0f64;
    for i in 0..o1.len() {
        md = md.max((o1[i] as f64 - o1b[i] as f64).abs());
    }
    let mut mds = 0f64;
    for i in 0..s1.len() {
        mds = mds.max((s1[i] as f64 - s1b[i] as f64).abs());
    }
    eprintln!("[gnar] t=8 결정론 out={md:.1e} st={mds:.1e}");
    let mut first_diff = None;
    let mut ndiff = 0usize;
    for i in 0..s1.len() {
        if s1[i] != s1b[i] {
            ndiff += 1;
            if first_diff.is_none() {
                first_diff = Some(i);
            }
        }
    }
    eprintln!(
        "[gnar] 첫 상이 idx={:?} 상이={ndiff}/{}",
        first_diff,
        s1.len()
    );
    let ok = mo < 1e-5 && ms < 1e-5;
    if !ok {
        fails += 1;
    }
    report.push_str(&format!(
        "| GdnARchunk(t1={m0:.0e}) out={mo:.1e} st={ms:.1e} {}",
        if ok { "OK" } else { "FAIL" }
    ));

    let (r, f) = sec12_gdn_conv_chunk(model, tname, lcg)?;
    report.push_str(&r);
    fails += f;
    // 107 P0-6: §15+ 는 §1-10 acc와 무관한 신규 버퍼만 쓴다 — 불변성
    // 체커 디바이스가 전부 닫힌 지금 재생성(동시 1대 유지).
    let acc = VkAcc::new()?;
    let (r, f) = sec16_gdn_conv_t1(&acc)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec17_gdn_ar_t1(&acc)?;
    report.push_str(&r);
    fails += f;
    let (r, f) = sec18_head_chain(&acc, model)?;
    report.push_str(&r);
    fails += f;
    Ok((report, fails))
}

/// 프레임 버퍼 원시 포인터(프로브 내부용).
fn acc_frame_ptr(acc: &VkAcc, h: u64) -> *mut u8 {
    acc.framebufs
        .lock()
        .get(&h)
        .map(|b| b.ptr)
        .unwrap_or(std::ptr::null_mut())
}
