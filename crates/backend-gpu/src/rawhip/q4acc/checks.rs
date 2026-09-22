//! q4acc 검증 — GPU↔CPU 미러 체크 서브커맨드 (plans/78 R1).

use super::*;
use crate::rawhip::env_on;

/// q5_1 커널 마이크로 검증 — 합성 블록 1개(d=1.0, m=-0.5, q=i%32)로
/// GPU ↔ CPU 레인 미러를 원소 수준에서 대조한다 (`q4-acc-check micro`).
pub fn micro_check() -> Result<String, String> {
    use llm170_core::matmul::MatmulHost;
    let n = 32usize;
    let mut bytes = vec![0u8; 24];
    bytes[0] = 0x00;
    bytes[1] = 0x3C; // d = 1.0
    bytes[2] = 0x00;
    bytes[3] = 0xB8; // m = -0.5
    let mut qh = 0u32;
    for i in 0..n {
        let q = (i % 32) as u32;
        if (q >> 4) & 1 == 1 {
            qh |= 1 << i;
        }
    }
    bytes[4..8].copy_from_slice(&qh.to_le_bytes());
    for j in 0..16usize {
        let lo = ((j) % 32) as u8 & 0xF;
        let hi = ((16 + j) % 32) as u8 & 0xF;
        bytes[8 + j] = lo | (hi << 4);
    }
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.15).collect();
    let w = llm170_core::matmul::Weight { data: &bytes, ty: GgmlType::Q5_1, n_in: n as u64, n_out: 1 };
    let acc = Q4Acc::new()?;
    let mut gpu = vec![vec![0.0f32; 1]];
    acc.matmul_batch(std::slice::from_ref(&x), &w, &mut gpu)?;
    let y = llm170_core::quant::quantize_row_q8_ref(&x);
    let cpu = llm170_core::quant::dot_row_w4a8_q5_1_lane(&bytes, n as u64, &y);
    Ok(format!(
        "micro q5_1: gpu={:?} cpu={cpu:.6} qh={qh:#010x} block={:02x?}",
        gpu[0][0], bytes
    ))
}

/// `q4-ar-check` — 프레임 AR 커널(q4_gdn_ar_w) ↔ core `gdn_ar_batch` 대조.
/// 합성 입력(결정적 LCG)으로 수치 계약을 직접 확인한다.
#[allow(clippy::many_single_char_names)]
pub fn ar_check() -> Result<String, String> {
    ar_check_t(1)
}

/// t토큰 AR 대조 — t>1은 커널 내부 순차 재귀 경로.
/// `q4-ple-check` — q4_ple_gate 커널 ↔ 호스트 산술 미러 대조(합성 입력).
pub fn ple_gate_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_embd, hc) = (2560usize, 4usize);
    let hc_dim = hc * n_embd;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let res: Vec<f32> = (0..hc_dim).map(|_| lcg()).collect();
    let key: Vec<f32> = (0..hc_dim).map(|_| lcg()).collect();
    let val: Vec<f32> = (0..n_embd).map(|_| lcg()).collect();
    let nk: Vec<f32> = (0..hc_dim).map(|_| 0.8 + lcg().abs()).collect();
    let nq: Vec<f32> = (0..hc_dim).map(|_| 0.8 + lcg().abs()).collect();
    let nc: Vec<f32> = (0..hc_dim).map(|_| 0.8 + lcg().abs()).collect();
    let rd = ctx.alloc(hc_dim * 4)?;
    let kd = ctx.alloc(hc_dim * 4)?;
    let vd = ctx.alloc(n_embd * 4)?;
    let nkd = ctx.alloc(hc_dim * 4)?;
    let nqd = ctx.alloc(hc_dim * 4)?;
    let ncd = ctx.alloc(hc_dim * 4)?;
    let gd = ctx.alloc(hc_dim * 4)?;
    let god = ctx.alloc(hc * 4)?;
    ctx.h2d(rd, bytemuck::cast_slice(&res))?;
    ctx.h2d(kd, bytemuck::cast_slice(&key))?;
    ctx.h2d(vd, bytemuck::cast_slice(&val))?;
    ctx.h2d(nkd, bytemuck::cast_slice(&nk))?;
    ctx.h2d(nqd, bytemuck::cast_slice(&nq))?;
    ctx.h2d(ncd, bytemuck::cast_slice(&nc))?;
    let (mut rp, mut kp, mut vp, mut nk_, mut nq_, mut nc_, mut gp_, mut gop_) = (
        rd as *mut c_void, kd as *mut c_void, vd as *mut c_void,
        nkd as *mut c_void, nqd as *mut c_void, ncd as *mut c_void,
        gd as *mut c_void, god as *mut c_void,
    );
    let (mut e, mut ne, mut hcc, mut tt) = (1e-6f32, n_embd as i32, hc as i32, 1i32);
    let mut args: Vec<*mut c_void> = vec![
        &mut rp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
        &mut vp as *mut _ as *mut c_void, &mut nk_ as *mut _ as *mut c_void,
        &mut nq_ as *mut _ as *mut c_void, &mut nc_ as *mut _ as *mut c_void,
        &mut gp_ as *mut _ as *mut c_void, &mut gop_ as *mut _ as *mut c_void,
        &mut e as *mut _ as *mut c_void, &mut ne as *mut _ as *mut c_void,
        &mut hcc as *mut _ as *mut c_void, &mut tt as *mut _ as *mut c_void,
    ];
    ctx.launch3("q4_ple_gate", hc.div_ceil(8) as u32, 1, 1, 256, &mut args)?;
    ctx.sync()?;
    let mut dgate = vec![0f32; hc];
    let mut dgated = vec![0f32; hc_dim];
    ctx.d2h(bytemuck::cast_slice_mut(&mut dgate).as_mut(), god)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut dgated).as_mut(), gd)?;
    // 호스트 미러(ple_block 산술)
    let eps = 1e-6f32;
    let mut out = String::new();
    for s in 0..hc {
        let kn = llm170_core::ops::rms_norm(&key[s * n_embd..(s + 1) * n_embd], &nk[s * n_embd..(s + 1) * n_embd], eps);
        let qn = llm170_core::ops::rms_norm(&res[s * n_embd..(s + 1) * n_embd], &nq[s * n_embd..(s + 1) * n_embd], eps);
        let mut dot = 0.0f32;
        for i in 0..n_embd { dot += kn[i] * qn[i]; }
        dot /= (n_embd as f32).sqrt();
        let mag = dot.abs().max(1e-6).sqrt();
        let g = llm170_core::ops::sigmoid(if dot >= 0.0 { mag } else { -mag });
        let mut gated: Vec<f32> = (0..n_embd).map(|i| val[i] * g).collect();
        let sg = {
            let sum = llm170_core::ops::sq_sum(&gated);
            1.0 / ((sum / n_embd as f64 + eps as f64).sqrt() as f32)
        };
        for i in 0..n_embd { gated[i] = gated[i] * sg * nc[s * n_embd + i]; }
        let gmax = gated.iter().zip(dgated[s * n_embd..(s + 1) * n_embd].iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        out += &format!("s{s}: gate dev={:.6} host={:.6} (dot={:.4}) gated max|d-h|={gmax:.2e}\n", dgate[s], g, dot);
    }
    Ok(out)
}

pub fn ar_check_t(t: usize) -> Result<String, String> {
    use llm170_core::matmul::{FrameHost, FrameState};
    let (n_group, dt_rank, d) = (16usize, 48usize, 128usize);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let k_len = n_group * d;
    let v_len = dt_rank * d;
    let q: Vec<f32> = (0..k_len * t).map(|_| lcg()).collect();
    let k: Vec<f32> = (0..k_len * t).map(|_| lcg()).collect();
    let v: Vec<f32> = (0..v_len * t).map(|_| lcg()).collect();
    let beta: Vec<f32> = (0..dt_rank * t).map(|_| lcg()).collect();
    let g: Vec<f32> = (0..dt_rank * t).map(|_| lcg()).collect();
    let st0: Vec<f32> = (0..dt_rank * d * d).map(|_| lcg() * 0.1).collect();

    // CPU 기준 (프레임 규약: β = σ(b), g는 원값 — AR이 exp)
    let beta_sig: Vec<f32> = beta
        .iter()
        .map(|&b| 1.0 / (1.0 + llm170_core::ops::exp_cr(-b)))
        .collect();
    let mut st_cpu = st0.clone();
    let mut o_cpu = vec![0.0f32; v_len * t];
    llm170_core::gdn::gdn_ar_batch(
        &q, &k, &v, &beta_sig, &g, &mut st_cpu, &mut o_cpu, t, n_group, dt_rank,
    );

    // GPU 프레임 (q는 1/√d 선스케일, bg는 인터리브 [σ(b), e^g])
    let acc = Q4Acc::new()?;
    let hq = acc.frame_alloc(k_len * t)?;
    let hk = acc.frame_alloc(k_len * t)?;
    let hv = acc.frame_alloc(v_len * t)?;
    let hbg = acc.frame_alloc(dt_rank * 2 * t)?;
    let hst = acc.frame_alloc(st0.len())?;
    let ho = acc.frame_alloc(v_len * t)?;
    let mut bg = vec![0.0f32; dt_rank * 2 * t];
    for h in 0..dt_rank * t {
        bg[h * 2] = beta_sig[h];
        bg[h * 2 + 1] = llm170_core::ops::exp_cr(g[h]);
    }
    let qs: Vec<f32> = q.iter().map(|x| x / (d as f32).sqrt()).collect();
    acc.frame_write(hq, &qs)?;
    acc.frame_write(hk, &k)?;
    acc.frame_write(hv, &v)?;
    acc.frame_write(hbg, &bg)?;
    // 프레임 AR은 전치 상태 레이아웃(gdn_ar_w_swap) — 업로드 전치, 판독 후 복원.
    let st0_t = llm170_core::qwen4exp::frame::Frame4::transpose_pairs(&st0, d);
    acc.frame_write(hst, &st0_t)?;
    acc.frame_gdn_ar(hq, hk, hv, hbg, hst, ho, 1, n_group, dt_rank, d)?;
    let mut o_gpu = vec![0.0f32; v_len * t];
    acc.frame_read(ho, &mut o_gpu)?;
    let mut st_gpu_t = vec![0.0f32; st0.len()];
    acc.frame_read(hst, &mut st_gpu_t)?;
    let st_gpu = llm170_core::qwen4exp::frame::Frame4::transpose_pairs(&st_gpu_t, d);
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y).abs() as f64) / (y.abs().max(1e-3) as f64))
            .fold(0.0f64, f64::max)
    };
    Ok(format!(
        "q4-ar-check t={t} n_group={n_group} dt_rank={dt_rank} d={d}: out rel={:.3e} (cpu[0]={:+.6} gpu[0]={:+.6}), state rel={:.3e}",
        rel(&o_gpu, &o_cpu),
        o_cpu[0],
        o_gpu[0],
        rel(&st_gpu, &st_cpu)
    ))
}

/// `q4-qsa-check [t] [n_past]` — q4_qsa_attn GPU ↔ CPU 미러(합성 Q/K/V·마스크).
/// t>128 결함(오답)의 원인을 좁히기 위한 격리 하네스.
pub fn qsa_check(t: usize, n_past: usize) -> Result<String, String> {
    use llm170_core::ops::exp_cr;
    let (n_head, n_kv, hd) = (24usize, 2usize, 256usize);
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    // 인과 + 블록 스파스 마스크: 위치 p는 (tok, p)가 허용될 때만 1.
    let total = n_past;
    let q: Vec<f32> = (0..t * n_head * 2 * hd).map(|_| lcg()).collect();
    let ck: Vec<f32> = (0..total * n_kv * hd).map(|_| lcg()).collect();
    let cv: Vec<f32> = (0..total * n_kv * hd).map(|_| lcg()).collect();
    let mut mask = vec![0u32; t * n_past];
    let base = n_past - t; // 이 배치 이전 위치 수
    for tok in 0..t {
        for p in 0..n_past {
            // 인과: p <= base+tok. 스파스: 4토큰 블록당 최근 2블록만 남기는 흉내.
            let causal = p <= base + tok;
            let blk = p / 4;
            let cur = (base + tok) / 4;
            let keep = blk + 2 > cur;
            mask[tok * n_past + p] = if causal && keep { 1 } else { 0 };
        }
    }
    let kq_scale = 1.0f32;
    let acc = Q4Acc::new()?;
    let gpu = acc.qsa_attn_raw(&q, &ck, &cv, &mask, kq_scale, n_past, n_head, n_kv, hd, t)?;
    // CPU 미러 (core stages::qsa_attention과 같은 산술 구조).
    let mut cpu = vec![0.0f32; t * n_head * hd];
    for tok in 0..t {
        for h in 0..n_head {
            let kvh = h / (n_head / n_kv);
            let qh = &q[(tok * n_head + h) * 2 * hd..(tok * n_head + h) * 2 * hd + hd];
            let gate = &q[(tok * n_head + h) * 2 * hd + hd..(tok * n_head + h) * 2 * hd + 2 * hd];
            let mut m = f32::NEG_INFINITY;
            let mut sc = vec![f32::NEG_INFINITY; n_past];
            for p in 0..n_past {
                if mask[tok * n_past + p] == 0 {
                    continue;
                }
                let k = &ck[p * n_kv * hd + kvh * hd..p * n_kv * hd + kvh * hd + hd];
                let mut s = 0.0f32;
                for i in 0..hd {
                    s += qh[i] * k[i];
                }
                s *= kq_scale;
                sc[p] = s;
                m = m.max(s);
            }
            let mut l = 0.0f32;
            let mut a = [0.0f32; 256];
            for p in 0..n_past {
                if sc[p] == f32::NEG_INFINITY {
                    continue;
                }
                let e = exp_cr(sc[p] - m);
                l += e;
                let v = &cv[p * n_kv * hd + kvh * hd..p * n_kv * hd + kvh * hd + hd];
                for i in 0..hd {
                    a[i] += e * v[i];
                }
            }
            for i in 0..hd {
                let o = if l > 0.0 { a[i] / l } else { 0.0 };
                cpu[(tok * n_head + h) * hd + i] = o * (1.0 / (1.0 + exp_cr(-gate[i])));
            }
        }
    }
    let mut nz = 0usize;
    let mut maxrel = 0.0f64;
    let mut nonfinite = 0usize;
    for (i, (&a, &b)) in gpu.iter().zip(&cpu).enumerate() {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = ((a - b).abs() as f64) / (b.abs().max(1e-3) as f64);
        if d > 1e-3 {
            nz += 1;
            if nz <= 3 {
                let tok = i / (n_head * hd);
                let h = (i / hd) % n_head;
                let dim = i % hd;
                eprintln!("# qsa diff #{nz} tok={tok} h={h} dim={dim} gpu={a} cpu={b}");
            }
        }
        maxrel = maxrel.max(d);
    }
    // 선택-목록판 — 같은 마스크에서 오름차순 목록을 만들어 마스크판과 대조한다.
    // 산술 순서가 같으므로 **비트 동일**이 기대값이다(다르면 목록/커널 버그).
    let mut sel_idx: Vec<u32> = Vec::new();
    let mut sel_off: Vec<u32> = vec![0];
    for tok in 0..t {
        for p in 0..n_past {
            if mask[tok * n_past + p] != 0 {
                sel_idx.push(p as u32);
            }
        }
        sel_off.push(sel_idx.len() as u32);
    }
    let gpu_sel = acc.qsa_attn_sel_raw(
        &q, &ck, &cv, &sel_idx, &sel_off, kq_scale, n_head, n_kv, hd, t,
    )?;
    let mut bit_diff = 0usize;
    let mut maxrel_sel = 0.0f64;
    let mut maxrel_cpu = 0.0f64;
    for ((&a, &b), &c) in gpu.iter().zip(&gpu_sel).zip(&cpu) {
        if a != b {
            bit_diff += 1;
        }
        let d = ((a - c).abs() as f64) / (c.abs().max(1e-3) as f64);
        maxrel_sel = maxrel_sel.max(d);
        let d2 = ((b - c).abs() as f64) / (c.abs().max(1e-3) as f64);
        maxrel_cpu = maxrel_cpu.max(d2);
    }
    let gpu_sel4 = acc.qsa_attn_sel4_raw(
        &q, &ck, &cv, &sel_idx, &sel_off, kq_scale, n_head, n_kv, hd, t,
    )?;
    let mut bit_diff4 = 0usize;
    let mut maxrel_sel4 = 0.0f64;
    for (&b, &c) in gpu_sel.iter().zip(&gpu_sel4) {
        if b != c {
            bit_diff4 += 1;
        }
        let d = ((c - b).abs() as f64) / (b.abs().max(1e-3) as f64);
        maxrel_sel4 = maxrel_sel4.max(d);
    }
    Ok(format!(
        "q4-qsa-check t={t} n_past={n_past}: nonfinite={nonfinite} mismatch={nz}/{} maxrel={maxrel:.3e} | sel: bit_diff={bit_diff}/{} sel4_bit_diff={bit_diff4}/{} maxrel_sel4_vs_sel={maxrel_sel4:.3e} maxrel_sel_vs_cpu={maxrel_sel:.3e} maxrel_mask_vs_cpu={maxrel_cpu:.3e} sel_keys={} (스캔 {}키 대비 {:.1}배 적음)",
        gpu.len(),
        gpu_sel.len(),
        gpu_sel4.len(),
        sel_idx.len(),
        t * n_past,
        (t * n_past) as f64 / (sel_idx.len().max(1)) as f64,
    ))
}

/// `q4-hc-check [t] [n] [hc]` — 프레임 HC op(HcGateMean/HcCombine) 격리 검증.
/// 합성 입력으로 GPU ↔ CPU 미러를 대조하고, 폴트 여부를 직접 보고한다.
pub fn hc_check(t: usize, n: usize, hc: usize) -> Result<String, String> {
    use llm170_core::matmul::{FrameHost, FrameOp, FrameState};
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let acc = Q4Acc::new()?;
    let hxn = acc.frame_alloc(t * hc * n)?;
    let hgate = acc.frame_alloc(t * hc * n)?;
    let hmix = acc.frame_alloc(t * n)?;
    let hres = acc.frame_alloc(t * hc * n)?;
    let hout = acc.frame_alloc(t * n)?;
    let hinj = acc.frame_alloc(t * hc)?;
    let xn: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let gate: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let res0: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let out: Vec<f32> = (0..t * n).map(|_| lcg()).collect();
    let inj: Vec<f32> = (0..t * hc).map(|_| lcg()).collect();
    acc.frame_write(hxn, &xn)?;
    acc.frame_write(hgate, &gate)?;
    acc.frame_write(hres, &res0)?;
    acc.frame_write(hout, &out)?;
    acc.frame_write(hinj, &inj)?;
    acc.frame_begin(t);
    acc.frame_op(&FrameOp::HcGateMean { xn: hxn, gate: hgate, out: hmix, hc, n })?;
    acc.frame_op(&FrameOp::HcCombine { res: hres, out: hout, inj: hinj, hc, n, total: hc * n * t })?;
    let mut mix_gpu = vec![0.0f32; t * n];
    acc.frame_read(hmix, &mut mix_gpu)?;
    let mut res_gpu = vec![0.0f32; t * hc * n];
    acc.frame_read(hres, &mut res_gpu)?;
    // CPU 미러
    let sig = |x: f32| 1.0f32 / (1.0 + llm170_core::ops::exp_cr(-x));
    let mut mix_cpu = vec![0.0f32; t * n];
    for ti in 0..t {
        for i in 0..n {
            let mut a = 0.0f32;
            for s in 0..hc {
                let k = ti * hc * n + s * n + i;
                a += xn[k] * sig(gate[k]);
            }
            mix_cpu[ti * n + i] = a / hc as f32;
        }
    }
    let mut res_cpu = res0.clone();
    for ti in 0..t {
        for i in 0..n {
            for s in 0..hc {
                res_cpu[ti * hc * n + s * n + i] += out[ti * n + i] * 2.0 * sig(inj[ti * hc + s] / hc as f32);
            }
        }
    }
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y).abs() as f64) / (y.abs().max(1e-3) as f64))
            .fold(0.0f64, f64::max)
    };
    Ok(format!(
        "q4-hc-check t={t} n={n} hc={hc}: mix rel={:.3e} res rel={:.3e}",
        rel(&mix_gpu, &mix_cpu),
        rel(&res_gpu, &res_cpu)
    ))
}

/// `q4-acc-check <model> <tensor> [t] [rows]` — GPU(가속기 값 경로) ↔ CPU
/// W4A8 레인 미러 대조. 계약: 같은 산술 계열이므로 ≤1e-6(비트 일치 기대).
pub fn check_tensor(
    model: &std::path::Path,
    tensor: &str,
    t: usize,
    rows_max: usize,
) -> Result<String, String> {
    use llm170_core::matmul::MatmulHost;
    let m = llm170_core::qwen4exp::Model4::load(model).map_err(|e| e.to_string())?;
    let w = m
        .w(tensor)
        .ok_or_else(|| format!("텐서 없음: {tensor}"))?;
    let n_in = w.n_in as usize;
    let (blck, bsize) = w.ty.block_info();
    let row_bytes = (n_in / blck as usize) * bsize as usize;
    let n_out = (w.n_out as usize).min(rows_max.max(1));
    let ws = llm170_core::matmul::Weight {
        data: &w.data[..n_out * row_bytes],
        ty: w.ty,
        n_in: w.n_in,
        n_out: n_out as u64,
    };
    // 결정적 입력 (LCG, ±0.5)
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let acc = Q4Acc::new()?;
    let mut gpu = vec![vec![0.0f32; n_out]; t];
    acc.matmul_batch(&xs, &ws, &mut gpu)?;
    // CPU W4A8 레인 미러
    let mut cpu = vec![vec![0.0f32; n_out]; t];
    for (ti, x) in xs.iter().enumerate() {
        if matches!(ws.ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16) {
            // f32 계열 — 가속기도 f32로 전개해 f32 커널을 쓴다. 환원 순서만
            // 다르므로 f64 기준 대비 상대오차로 판정한다.
            let mut wrow = vec![0.0f32; n_in];
            for (o, out) in cpu[ti].iter_mut().enumerate() {
                llm170_core::quant::dequant_row(
                    ws.ty,
                    &ws.data[o * row_bytes..],
                    0,
                    n_in as u64,
                    &mut wrow,
                );
                let mut s = 0.0f64;
                for i in 0..n_in {
                    s += x[i] as f64 * wrow[i] as f64;
                }
                *out = s as f32;
            }
            continue;
        }
        let y = llm170_core::quant::quantize_row_q8_ref(x);
        for (o, out) in cpu[ti].iter_mut().enumerate() {
            let row = &ws.data[o * row_bytes..(o + 1) * row_bytes];
            *out = match ws.ty {
                GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, ws.n_in, &y),
                GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, ws.n_in, &y),
                GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, ws.n_in, &y),
                GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, ws.n_in, &y),
                GgmlType::Q8_0 => llm170_core::quant::dot_row_w4a8_q8_0_lane(row, ws.n_in, &y),
                GgmlType::Q5_1 => llm170_core::quant::dot_row_w4a8_q5_1_lane(row, ws.n_in, &y),
                GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, ws.n_in, &y),
                GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, ws.n_in, &y),
                GgmlType::Iq4Xs => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, ws.n_in, &y),
                other => return Err(format!("q4-acc-check: 미지원 타입 {other:?} — /dev/null")),
            };
        }
    }
    let (mut max_abs, mut max_rel, mut bit_eq, mut n) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        for (a, b) in g.iter().zip(c.iter()) {
            let d = (*a - *b).abs() as f64;
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / b.abs().max(1e-3) as f64);
            bit_eq += (a.to_bits() == b.to_bits()) as usize;
            n += 1;
        }
    }
    if env_on("LLM170_Q4ACC_ROWDBG") {
        for &ri in &[0usize, 1, 2, 127, 128, 129, 130, 199, 200, 201, 255] {
            if ri >= gpu.len() {
                continue;
            }
            let m = gpu[ri]
                .iter()
                .zip(&cpu[ri])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("# row {ri}: maxerr={m:.5}");
        }
    }
    if env_on("LLM170_Q4ACC_DBG") {
        eprintln!("# gpu[0][..8] = {:?}", &gpu[0][..8]);
        eprintln!("# cpu[0][..8] = {:?}", &cpu[0][..8]);
    }
    Ok(format!(
        "q4-acc-check {tensor} [{n_out}x{n_in}] ty={:?} t={t}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} bit_eq={}/{} ({:.1}%)",
        ws.ty,
        bit_eq,
        n,
        100.0 * bit_eq as f64 / n as f64
    ))
}

/// `moe-row-check <model> <tensor> [t_a] [t_b]` — 전문가 스택 GEMM(frame_moe_gemm)
/// 의 행 수 불변성 프로브. 동일 x·ids 앞부분(rows_a = t_a·k_sel)을 t_a와 t_b
/// 두 호출로 계산해 공유 행의 출력을 비트 대조한다. 청크 불변성 결함(plans/80)
/// 의 발원지 특정용 — GEMM이 행 수에 무관하면 bit_eq=전부.
pub fn moe_row_check(
    model: &std::path::Path,
    tensor: &str,
    t_a: usize,
    t_b: usize,
) -> Result<String, String> {
    use llm170_core::matmul::{FrameHost, FrameState};
    let m = llm170_core::qwen4exp::Model4::load(model).map_err(|e| e.to_string())?;
    let hp = &m.hp;
    let w = m.w(tensor).ok_or_else(|| format!("텐서 없음: {tensor}"))?;
    let n_in = w.n_in as usize;
    let k_sel = hp.n_expert_used;
    let ne = hp.n_expert;
    let n_out = (w.n_out as usize) / ne;
    let (rows_a, rows_b) = (t_a * k_sel, t_b * k_sel);
    if rows_a == 0 || rows_b < rows_a {
        return Err(format!("moe-row-check: t_b·k_sel({rows_b}) >= t_a·k_sel({rows_a}) 필요"));
    }
    // 결정적 입력 (LCG, ±0.5) — ids는 [0, ne) 균등.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let x: Vec<f32> = (0..rows_b * n_in).map(|_| lcg()).collect();
    // 충돌-중심 ids — 실제 라우팅(토큰×k_sel 픽)은 같은 전문가가 여러 번 뽑힌다.
    // 초기판(준정렬 ids)은 그룹화 충돌 경로를 못 덮었다(2026-09-18 재현).
    let ids: Vec<u32> = (0..rows_b).map(|i| ((i % 37) as u32) % ne as u32).collect();
    let acc = Q4Acc::new()?;
    let hx = acc.frame_alloc(rows_b * n_in)?;
    let hids = acc.frame_alloc(rows_b)?;
    let hoa = acc.frame_alloc(rows_a * n_out)?;
    let hob = acc.frame_alloc(rows_b * n_out)?;
    acc.frame_write(hx, &x)?;
    acc.frame_write_u32(hids, &ids)?;
    let ws = llm170_core::matmul::Weight { data: w.data, ty: w.ty, n_in: w.n_in, n_out: w.n_out };
    acc.frame_begin(t_a);
    acc.frame_moe_gemm(hx, &ws, hids, hoa, ne, k_sel)?;
    acc.frame_begin(t_b);
    acc.frame_moe_gemm(hx, &ws, hids, hob, ne, k_sel)?;
    let mut oa = vec![0.0f32; rows_a * n_out];
    let mut ob = vec![0.0f32; rows_b * n_out];
    acc.frame_read(hoa, &mut oa)?;
    acc.frame_read(hob, &mut ob)?;
    let (mut max_abs, mut bit_eq, mut first) = (0.0f64, 0usize, None);
    for r in 0..rows_a {
        for c in 0..n_out {
            let i = r * n_out + c;
            let (a, b) = (oa[i], ob[i]);
            if a.to_bits() == b.to_bits() {
                bit_eq += 1;
            } else if first.is_none() {
                first = Some((r, c, a, b));
            }
            max_abs = max_abs.max((a - b).abs() as f64);
        }
    }
    let n = rows_a * n_out;
    let mut s = format!(
        "moe-row-check {tensor} ty={:?} rows {rows_a} vs {rows_b} (t {t_a} vs {t_b}): bit_eq={bit_eq}/{n}",
        w.ty
    );
    if let Some((r, c, a, b)) = first {
        s.push_str(&format!(" max_abs={max_abs:.3e} FIRST-DIFF row={r} col={c} a={a:.6e} b={b:.6e}"));
    } else {
        s.push_str(" — 완전 비트 동일");
    }
    Ok(s)
}

/// `mm-row-check <model> <tensor> [t_a] [t_b]` — 밀도 GEMM(frame_mm)의 행 수
/// 불변성 프로브. 동일 x 앞 t_a행을 t_a/t_b 두 호출로 계산해 비트 대조.
/// route/qkv 등 frame_mm_group이 쓰는 계열(plans/80 청크 불변성).
pub fn mm_row_check(
    model: &std::path::Path,
    tensor: &str,
    t_a: usize,
    t_b: usize,
) -> Result<String, String> {
    use llm170_core::matmul::{FrameHost, FrameState};
    let m = llm170_core::qwen4exp::Model4::load(model).map_err(|e| e.to_string())?;
    let w = m.w(tensor).ok_or_else(|| format!("텐서 없음: {tensor}"))?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    if t_b <= t_a || t_a == 0 {
        return Err(format!("mm-row-check: t_b({t_b}) > t_a({t_a}) > 0 필요"));
    }
    let mut seed = 0x0D1B_54A3_2D19_2ED5u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let x: Vec<f32> = (0..t_b * n_in).map(|_| lcg()).collect();
    let acc = Q4Acc::new()?;
    let hx = acc.frame_alloc(t_b * n_in)?;
    let hoa = acc.frame_alloc(t_a * n_out)?;
    let hob = acc.frame_alloc(t_b * n_out)?;
    acc.frame_write(hx, &x)?;
    let ws = llm170_core::matmul::Weight { data: w.data, ty: w.ty, n_in: w.n_in, n_out: w.n_out };
    acc.frame_begin(t_a);
    acc.frame_mm(hx, &ws, hoa, t_a)?;
    acc.frame_begin(t_b);
    acc.frame_mm(hx, &ws, hob, t_b)?;
    let mut oa = vec![0.0f32; t_a * n_out];
    let mut ob = vec![0.0f32; t_b * n_out];
    acc.frame_read(hoa, &mut oa)?;
    acc.frame_read(hob, &mut ob)?;
    let (mut max_ulp, mut bit_eq, mut first) = (0i64, 0usize, None);
    for r in 0..t_a {
        for c in 0..n_out {
            let i = r * n_out + c;
            let (a, b) = (oa[i], ob[i]);
            if a.to_bits() == b.to_bits() {
                bit_eq += 1;
            } else {
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).abs();
                max_ulp = max_ulp.max(ulp);
                if first.is_none() {
                    first = Some((r, c, a, b, ulp));
                }
            }
        }
    }
    let n = t_a * n_out;
    let mut s = format!(
        "mm-row-check {tensor} ty={:?} [{n_out}x{n_in}] t {t_a} vs {t_b}: bit_eq={bit_eq}/{n}",
        w.ty
    );
    if let Some((r, c, a, b, ulp)) = first {
        s.push_str(&format!(
            " max_ulp={max_ulp} FIRST-DIFF row={r} col={c} a={a:.7e} b={b:.7e} ulp={ulp}"
        ));
    } else {
        s.push_str(" — 완전 비트 동일");
    }
    Ok(s)
}
