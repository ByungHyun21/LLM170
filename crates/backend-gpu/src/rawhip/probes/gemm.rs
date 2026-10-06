//! probes/gemm — GEMM 벤치·타일 검증 (probes.rs에서 이동, plans/78 R3).

use super::*;

/// 진단 기본 경로(인자 우선) — server/probes.rs d_q35 패턴(107 W4).
const D_Q35: &str = "/home/yoon/models/qwen3.8-27b/q35work.gguf";

/// 배치 mm 타이밍 — gy=1 대비 gy=t 배율.
pub fn mm_batch_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| D_Q35.into());
    let tname = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let x: Vec<f32> = (0..n_in * 64).map(|_| lcg()).collect();
    let xd = ctx.alloc(n_in * 64 * 4)?;
    ctx.h2d(xd, bytemuck::cast_slice(&x))?;
    let xq = ctx.alloc(xq_w * 4 * 64)?;
    ctx.quant_q8_b(xd, xq, n_in, xq_w, 64)?;
    let out = ctx.alloc(n_out * 4 * 64)?;
    let mut msg = String::new();
    for &t in &[1usize, 8, 64] {
        ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, t)?;
        ctx.sync()?;
        let reps = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, t)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        msg += &format!(
            "t={}: {:.3}ms ({:.0} GB/s-equiv)\n",
            t,
            dt * 1e3,
            w.data.len() as f64 / dt / 1e9
        );
    }
    Ok(msg)
}

/// 타일 커널 검증+타이밍 — gemm_q5k_bt vs 미러.
pub fn mm_tile_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| D_Q35.into());
    let tname = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let t = std::env::var("LLM170_TILE_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16usize);
    let mut xs = Vec::new();
    let mut q8s = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        q8s.push(llm170_core::quant::quantize_row_q8_ref(&x));
        xs.push(x);
    }
    let mut xq_h: Vec<u32> = Vec::new();
    for tok in &q8s {
        for blk in tok {
            for c in 0..8 {
                let base = c * 4;
                xq_h.push(
                    (blk.qs[base] as u32 & 0xFF)
                        | ((blk.qs[base + 1] as u32 & 0xFF) << 8)
                        | ((blk.qs[base + 2] as u32 & 0xFF) << 16)
                        | ((blk.qs[base + 3] as u32 & 0xFF) << 24),
                );
            }
        }
        for blk in tok {
            xq_h.push(blk.d.to_bits());
        }
        for blk in tok {
            let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
            let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
            xq_h.push(s0 as u32);
            xq_h.push(s1 as u32);
        }
    }
    let xq = ctx.alloc(xq_h.len() * 4)?;
    if let Some(f) = std::env::var_os("LLM170_XQN_FILE") {
        let bytes = std::fs::read(&f).unwrap();
        assert_eq!(
            bytes.len(),
            xq_h.len() * 4,
            "덤프 크기 불일치: {} vs {}",
            bytes.len(),
            xq_h.len() * 4
        );
        ctx.h2d(xq, &bytes)?;
        eprintln!("xq 리플레이: {}", f.to_string_lossy());
    } else {
        ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    }
    let out = ctx.alloc(n_out * 4 * t)?;
    let _wp = wd as *mut std::ffi::c_void;
    let _op = out as *mut std::ffi::c_void;
    let _xp = xq as *mut std::ffi::c_void;
    let _ni = n_in as i32;
    let _no = n_out as i32;
    let _xw = xq_w as i32;
    let _tt = t as i32;
    ctx.gemm_tile(xq, wd, kt_d, w.ty as u32, n_in, n_out, xq_w, t, out)?;
    ctx.sync()?;
    let mut o = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o).as_mut(), out)?;
    // 미러 검증
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut mism = 0;
    let mut first_dbg = String::new();
    for ti in 0..t {
        for oo in 0..n_out.min(256) {
            let row = &w.data[oo * rb..];
            let c = match w.ty {
                llm170_gguf::GgmlType::Q5K => {
                    llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Q4K => {
                    llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Q6K => {
                    llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Iq4Xs => {
                    llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &q8s[ti])
                }
                other => {
                    return Err(format!("mm-tile 미지원 타입 {other:?} — 미러 오계산 방지"));
                }
            };
            if c.to_bits() != o[ti * n_out + oo].to_bits() {
                mism += 1;
                if mism == 1 {
                    first_dbg =
                        format!("ti={ti} o={oo}: cpu={c:.7e} gpu={:.7e}", o[ti * n_out + oo]);
                }
            }
        }
    }
    eprintln!("dbg: {first_dbg}");
    // 타이밍
    let reps = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        ctx.gemm_tile(xq, wd, kt_d, w.ty as u32, n_in, n_out, xq_w, t, out)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    Ok(format!(
        "tile t={t}: 불일치 {mism}/{} (첫 256행×t) — {:.3}ms → {:.0} GB/s-equiv, 토큰당 {:.1}µs",
        n_out.min(256) * t,
        dt * 1e3,
        w.data.len() as f64 / dt / 1e9,
        dt * 1e6 / t as f64
    ))
}

/// `f16-bench [rows] [n_in] [n_out] [reps]` — 기존 `.co` f16 GEMM(`gemm_f16_v4`)을
/// 직접 측정한다. q4k-bench와 같은 형상으로 재면 "텐서코어 경로의 상한"이 나온다
/// (roof-test mfma1 L1-fed 24.9 TFLOPS). 새 커널 없이 경로 가치를 판정하는 용도.
pub fn f16_bench(rows: usize, n_in: usize, n_out: usize, reps: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let xq_w = crate::rawhip::q4acc::xq_words(n_in);
    let xdev = ctx.alloc(xq_w * rows * 4)?;
    let wdev = ctx.alloc(n_out * n_in * 2)?;
    let odev = ctx.alloc(n_out * rows * 4)?;
    let x = vec![0x11u8; xq_w * rows * 4];
    let w = vec![0x22u8; n_out * n_in * 2];
    ctx.h2d(xdev, &x)?;
    ctx.h2d(wdev, &w)?;
    let fns = &ctx.fns;
    let fm = *fns
        .get("gemm_f16_v4")
        .ok_or("gemm_f16_v4 없음(co/mmq2.co 미로드)")?;
    let launch = || -> Result<(), String> {
        unsafe {
            let mut a1 = xdev as *mut std::ffi::c_void;
            let mut a2 = wdev as *mut std::ffi::c_void;
            let mut a3 = odev as *mut std::ffi::c_void;
            let (mut ni, mut no, mut xw, mut tt) =
                (n_in as i32, n_out as i32, xq_w as i32, rows.min(128) as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut a1) as *mut _ as *mut std::ffi::c_void,
                (&mut a2) as *mut _ as *mut std::ffi::c_void,
                (&mut a3) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            let e = hip::hipModuleLaunchKernel(
                fm,
                n_out.div_ceil(128) as u32,
                1,
                rows.div_ceil(128) as u32,
                256,
                1,
                1,
                0,
                ctx.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if e != hip::hipError_t_hipSuccess {
                return Err(format!("gemm_f16_v4 launch {e:?}"));
            }
        }
        Ok(())
    };
    launch()?;
    ctx.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        launch()?;
    }
    ctx.sync()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let gb = (n_out * n_in * 2) as f64 / (ms / 1e3) / 1e9;
    let flops = 2.0 * (n_out * n_in * rows) as f64 / (ms / 1e3) / 1e12;
    Ok(format!(
        "# f16-bench t={rows} {n_in}x{n_out}: {ms:.3}ms/호출 ({gb:.1}GB/s, {flops:.1} TFLOPS)"
    ))
}

/// `q4k-bench [rows] [n_in] [n_out] [reps]` — q4_K GEMM 형상 격리 계측.
/// 합성 q4_K 텐서로 커널 변형별 실효 대역을 잰다(plans/65 하한 분석의 입력).
pub fn q4k_bench(rows: usize, n_in: usize, n_out: usize, reps: usize) -> Result<String, String> {
    use crate::rawhip::q4acc::Q4Acc;
    use llm170_core::matmul::MatmulHost;
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let n_super = n_in / 256;
    let nblk = n_out * n_super;
    let mut w = vec![0u8; nblk * 144];
    for b in 0..nblk {
        let o = &mut w[b * 144..(b + 1) * 144];
        o[0] = 0x00;
        o[1] = 0x38; // d = 0.5
        o[2] = 0x00;
        o[3] = 0x30; // dmin = 0.25
        for j in 0..12 {
            o[4 + j] = ((b * 7 + j * 13) & 0x3F) as u8;
        }
        for i in 0..128 {
            o[16 + i] = ((b * 31 + i * 37) & 0xFF) as u8;
        }
    }
    let xs: Vec<Vec<f32>> = (0..rows)
        .map(|_| (0..n_in).map(|_| lcg()).collect())
        .collect();
    let weight = llm170_core::matmul::Weight {
        data: &w,
        ty: llm170_gguf::GgmlType::Q4K,
        n_in: n_in as u64,
        n_out: n_out as u64,
    };
    let acc = Q4Acc::new()?;
    let mut out = vec![vec![0.0f32; n_out]; rows];
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        acc.matmul_batch(&xs, &weight, &mut out)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let gb = (nblk * 144) as f64 / (ms / 1e3) / 1e9;
    Ok(format!(
        "# q4k-bench t={rows} {n_in}x{n_out}: {ms:.3}ms/호출 ({gb:.1}GB/s 가중치, {}MB)",
        nblk * 144 / 1_000_000
    ))
}

/// `q4k-micro` — q4_K MMQ 타일을 **단일 256원소 슈퍼블록**에서 CPU 미러
/// (`dot_q4k_q8`)와 직접 대조한다. 인덱스 매핑 버그를 값 수준에서 드러낸다.
pub fn q4k_micro() -> Result<String, String> {
    use crate::rawhip::q4acc::Q4Acc;
    use llm170_core::matmul::MatmulHost;
    let (n_out, n_in, t) = (16usize, 256usize, 16usize);
    // 합성 q4_K 블록: d=1.0, dmin=0.5, 6비트 스케일 패턴, 결정적 니블
    let mut blk = vec![0u8; 144];
    blk[0] = 0x00;
    blk[1] = 0x3C; // d = 1.0
    blk[2] = 0x00;
    blk[3] = 0x38; // dmin = 0.5
    for j in 0..12 {
        blk[4 + j] = (0x15u8.wrapping_mul(j as u8 + 1)) & 0x3F;
    }
    for i in 0..128 {
        blk[16 + i] = ((i * 37 + 11) & 0xFF) as u8;
    }
    let w: Vec<u8> = (0..n_out)
        .flat_map(|o| {
            let mut b = blk.clone();
            b[4] = (b[4].wrapping_add(o as u8)) & 0x3F;
            b
        })
        .collect();
    let xs: Vec<Vec<f32>> = (0..t)
        .map(|r| {
            (0..n_in)
                .map(|i| (((r * 31 + i) as u64 * 2654435761u64) % 1000) as f32 / 500.0 - 1.0)
                .collect()
        })
        .collect();
    let weight = llm170_core::matmul::Weight {
        data: &w,
        ty: llm170_gguf::GgmlType::Q4K,
        n_in: n_in as u64,
        n_out: n_out as u64,
    };
    let acc = Q4Acc::new()?;
    let mut out = vec![vec![0.0f32; n_out]; t];
    acc.matmul_batch(&xs, &weight, &mut out)?;
    let mut max_abs = 0.0f32;
    let mut first = String::new();
    for r in 0..t {
        let y = llm170_core::quant::quantize_row_q8_ref(&xs[r]);
        for o in 0..n_out {
            let cpu = llm170_core::quant::dot_q4k_q8(&w[o * 144..(o + 1) * 144], &y);
            let gpu = out[r][o];
            let d = (cpu - gpu).abs();
            if d > max_abs {
                max_abs = d;
            }
            if first.is_empty() && d > 1e-4 {
                first = format!(" 첫 불일치 r={r} o={o} gpu={gpu:.6} cpu={cpu:.6}");
            }
        }
    }
    Ok(format!(
        "q4k-micro {n_out}x{n_in} t={t}: max_abs={max_abs:.3e} ({}){first}",
        if max_abs < 1e-4 {
            "일치"
        } else {
            "불일치"
        }
    ))
}

/// `q5-1-bench [rows] [n_in] [n_out] [reps]` — q5_1 GEMM 격리 계측.
/// 실모델 MoE expert-down 형상(20행 × 640 × 2560)을 합성 데이터로 돌려 커널
/// 자체의 시간을 잰다 — KTRACE가 0.24ms/런치를 보고한 그 값과 대조하면
/// 문제가 커널인지 컨텍스트(L2 상태·주변 런치)인지 갈린다.
pub fn q5_1_bench(rows: usize, n_in: usize, n_out: usize, reps: usize) -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    // q5_1 블록 = 32원소(24B). 행당 n_in/32 블록.
    let n_sub = n_in / 32;
    let wrow = n_sub * 24;
    let wbytes = n_out * wrow;
    let xq_w = crate::rawhip::q4acc::xq_words(n_in);
    let xbytes = rows * xq_w * 4;
    let obytes = rows * n_out * 4;
    let wdev = ctx.alloc(wbytes.max(4))?;
    let xdev = ctx.alloc(xbytes.max(4))?;
    let odev = ctx.alloc(obytes.max(4))?;
    let part = ctx.scratch(n_out * 64 * 8)?;
    // 합성: 가중치는 0x3c 패턴(d/m f16 = 1.0/0.0 근사), x는 1
    let w = vec![0x3cu8; wbytes];
    let x = vec![0x01u8; xbytes];
    ctx.h2d(wdev, &w)?;
    ctx.h2d(xdev, &x)?;
    let launch = |kern: &'static str, gx: u32, block: u32| -> Result<(), String> {
        let (mut xp, mut wp, mut pp, mut op) = (xdev, wdev, part, odev);
        let (mut ni, mut no, mut xw, mut tt) =
            (n_in as i32, n_out as i32, xq_w as i32, rows as i32);
        let mut args: Vec<*mut c_void> = vec![
            (&mut xp) as *mut _ as *mut c_void,
            (&mut wp) as *mut _ as *mut c_void,
            (&mut pp) as *mut _ as *mut c_void,
            (&mut op) as *mut _ as *mut c_void,
            (&mut ni) as *mut _ as *mut c_void,
            (&mut no) as *mut _ as *mut c_void,
            (&mut xw) as *mut _ as *mut c_void,
            (&mut tt) as *mut _ as *mut c_void,
        ];
        let (gy, gz) = if kern.ends_with("_t") {
            let nb = n_out.div_ceil(4);
            (nb.min(65535) as u32, nb.div_ceil(65535) as u32)
        } else {
            (n_out.min(65535) as u32, n_out.div_ceil(65535) as u32)
        };
        ctx.launch3(kern, gx, gy, gz, block, &mut args)
    };
    // 워밍업 + 시간
    let mut msg = String::new();
    for (kern, gx, blk) in [
        ("q4_gemm_q5_1", rows as u32, 64u32),
        ("q4_gemm_q5_1_t", rows.div_ceil(16) as u32, 256),
    ] {
        for _ in 0..2 {
            launch(kern, gx, blk)?;
        }
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            launch(kern, gx, blk)?;
        }
        ctx.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let gb = wbytes as f64 / (ms / 1e3) / 1e9;
        let attrs = ctx
            .kern_attrs(kern)
            .map(|(regs, loc, mx)| format!("regs={regs} local={loc}B maxthr={mx}"))
            .unwrap_or_else(|| "attrs 없음".into());
        msg += &format!(
            "# {kern}: {ms:.3}ms/런치 ({gb:.1}GB/s 가중치) rows={rows} {n_in}x{n_out} grid=({gx},{n_out}) blk={blk} {attrs}\n"
        );
    }
    Ok(msg)
}

/// `q4-d2h-bench` — 소형 d2h 비용 격리(프레임 MoE가 ids 20KB를 읽는 데 15.5ms를
/// 쓰고 있었다). 크기별·경로별로 잰다.
pub fn d2h_bench() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let mut out = String::new();
    for &n in &[20 << 10usize, 1 << 20, 8 << 20] {
        let d = ctx.alloc(n)?;
        let mut dst = vec![0u8; n];
        // 워밍업 + 5회 평균
        for _ in 0..2 {
            ctx.d2h(&mut dst, d as *const u8)?;
        }
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            ctx.d2h(&mut dst, d as *const u8)?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
        // 순수 커널 런치 1회 비용(동기 없음) 대조
        let t1 = std::time::Instant::now();
        for _ in 0..5 {
            let _ = ctx.scratch(4);
        }
        let lms = t1.elapsed().as_secs_f64() * 1e3 / 5.0;
        out += &format!("# d2h {}KB: {:.3}ms (scratch {:.3}ms)\n", n >> 10, ms, lms);
    }
    Ok(out)
}

/// MMQ 포트 A/B — bt vs mm (각 미러).
#[allow(unused_assignments)] // na0/sg4 는 런치 인자로 넘긴 **주소**가 읽는 값 (raw 포인터 경유)
pub fn launch_probe() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    // 디코드 소형 커널의 실제 런치 비용 (트레이스 페어링 무관, 직접 계측).
    {
        let n = 5120usize;
        let xb = ctx.alloc(n * 4)?;
        let wb = ctx.alloc(n * 4)?;
        let qb = ctx.alloc(n / 4 + n / 32 + n / 16 + 64)?;
        let mut xp0 = xb as *mut std::ffi::c_void;
        let mut wp0 = wb as *mut std::ffi::c_void;
        let mut qp0 = qb as *mut std::ffi::c_void;
        let mut eps0 = 1e-6f32;
        let mut na0 = n as i32;
        let mut a0: Vec<*mut std::ffi::c_void> = vec![
            &mut xp0 as *mut _ as *mut std::ffi::c_void,
            &mut wp0 as *mut _ as *mut std::ffi::c_void,
            &mut qp0 as *mut _ as *mut std::ffi::c_void,
            &mut eps0 as *mut _ as *mut std::ffi::c_void,
            &mut na0 as *mut _ as *mut std::ffi::c_void,
        ];
        let mut s0 = String::new();
        for (label, blk) in [
            ("rmsq n=512", 160u32),
            ("rmsq n=5120", 512u32),
            ("rmsq n=20480", 640u32),
        ] {
            let nv: i32 = match label {
                "rmsq n=512" => 512,
                "rmsq n=5120" => 5120,
                _ => 20480,
            };
            na0 = nv;
            for _ in 0..20 {
                let _ = ctx.launch("rmsq", 1, 1, blk, &mut a0);
            }
            ctx.sync()?;
            let n2 = 2000usize;
            let t0 = std::time::Instant::now();
            for _ in 0..n2 {
                let _ = ctx.launch("rmsq", 1, 1, blk, &mut a0);
            }
            ctx.sync()?;
            s0.push_str(&format!(
                "{label}: {:.2}us  ",
                t0.elapsed().as_secs_f64() * 1e6 / n2 as f64
            ));
        }
        eprintln!("{s0}");
        // 기준선: 자명한 커널(axpy_scaled)의 런치 비용 — n 크기별
        let ab = ctx.alloc(5120 * 4)?;
        let bb = ctx.alloc(5120 * 4)?;
        let cb = ctx.alloc(5120 * 4)?;
        let mut ap = ab as *mut std::ffi::c_void;
        let mut bp = bb as *mut std::ffi::c_void;
        let mut cp = cb as *mut std::ffi::c_void;
        let mut nn = 64i32;
        let mut a2: Vec<*mut std::ffi::c_void> = vec![
            &mut ap as *mut _ as *mut std::ffi::c_void,
            &mut bp as *mut _ as *mut std::ffi::c_void,
            &mut cp as *mut _ as *mut std::ffi::c_void,
            &mut nn as *mut _ as *mut std::ffi::c_void,
        ];
        let mut s1 = String::new();
        for (label, nval, gx) in [
            ("axpy n=64 1blk", 64i32, 1u32),
            ("axpy n=5120 80blk", 5120, 80),
        ] {
            nn = nval;
            for _ in 0..20 {
                let _ = ctx.launch3("axpy_scaled", gx, 1, 1, 64, &mut a2);
            }
            ctx.sync()?;
            let n2 = 2000usize;
            let t0 = std::time::Instant::now();
            for _ in 0..n2 {
                let _ = ctx.launch3("axpy_scaled", gx, 1, 1, 64, &mut a2);
            }
            ctx.sync()?;
            s1.push_str(&format!(
                "{label}: {:.2}us  ",
                t0.elapsed().as_secs_f64() * 1e6 / n2 as f64
            ));
        }
        eprintln!("{s1}");
        // gatedq 직접 계측: (o, z, w, xq, eps, d, n_h, n_tot)
        {
            let d = 128usize;
            let n_tot = 32 * d;
            let ob = ctx.alloc(n_tot * 4)?;
            let zb = ctx.alloc(n_tot * 4)?;
            let wb2 = ctx.alloc(n_tot * 4)?;
            let qb2 = ctx.alloc(n_tot / 4 + n_tot / 32 + n_tot / 16 + 64)?;
            let mut op = ob as *mut std::ffi::c_void;
            let mut zp = zb as *mut std::ffi::c_void;
            let mut wp2 = wb2 as *mut std::ffi::c_void;
            let mut qp2 = qb2 as *mut std::ffi::c_void;
            let mut eps2 = 1e-6f32;
            let mut dd = d as i32;
            let mut nh3 = 32i32;
            let mut nt3 = n_tot as i32;
            let mut a3: Vec<*mut std::ffi::c_void> = vec![
                &mut op as *mut _ as *mut std::ffi::c_void,
                &mut zp as *mut _ as *mut std::ffi::c_void,
                &mut wp2 as *mut _ as *mut std::ffi::c_void,
                &mut qp2 as *mut _ as *mut std::ffi::c_void,
                &mut eps2 as *mut _ as *mut std::ffi::c_void,
                &mut dd as *mut _ as *mut std::ffi::c_void,
                &mut nh3 as *mut _ as *mut std::ffi::c_void,
                &mut nt3 as *mut _ as *mut std::ffi::c_void,
            ];
            let mut res = String::new();
            for (nb, thr) in [(1u32, 32u32), (8, 32), (32, 32), (32, 128)] {
                for _ in 0..20 {
                    let _ = ctx.launch3("gatedq", nb, 1, 1, thr, &mut a3);
                }
                ctx.sync()?;
                let n2 = 2000usize;
                let t0 = std::time::Instant::now();
                for _ in 0..n2 {
                    let _ = ctx.launch3("gatedq", nb, 1, 1, thr, &mut a3);
                }
                ctx.sync()?;
                res.push_str(&format!(
                    "{nb}blk x{thr}thr: {:.2}us  ",
                    t0.elapsed().as_secs_f64() * 1e6 / n2 as f64
                ));
            }
            eprintln!("{res}");
        }
        // qsa_flash_gqa 직접 계측 (13 인자)
        {
            let hd = 256usize;
            let nh = 24usize;
            let nkv = 4usize;
            let npast = 128i32;
            let nseg = 4usize;
            let qb = ctx.alloc(nh * 2 * hd * 4)?;
            let kb = ctx.alloc(nkv * 4096 * hd * 4)?;
            let vb = ctx.alloc(nkv * 4096 * hd * 4)?;
            let mb = ctx.alloc(4096 * 4)?;
            let pb = ctx.alloc(nseg * nh * (hd + 2) * 4)?;
            let mut a4: Vec<*mut std::ffi::c_void> = Vec::new();
            let mut qp4 = qb as *mut std::ffi::c_void;
            let mut kp4 = kb as *mut std::ffi::c_void;
            let mut vp4 = vb as *mut std::ffi::c_void;
            let mut mp4 = mb as *mut std::ffi::c_void;
            let mut pp4 = pb as *mut std::ffi::c_void;
            let mut np_ = npast;
            let mut nh4 = nh as i32;
            let mut nk4 = nkv as i32;
            let mut h4 = hd as i32;
            let mut tl4 = 1i32;
            let mut ss4 = 4096i32;
            let mut p04 = 0i32;
            let mut sg4 = 32i32;
            a4.push(&mut qp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut kp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut vp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut mp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut pp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut np_ as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut nh4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut nk4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut h4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut tl4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut ss4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut p04 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut sg4 as *mut _ as *mut std::ffi::c_void);
            let mut r2 = String::new();
            for (lab, nb, thr) in [
                ("sg4  ", 1u32, 256u32),
                ("sg8  ", 1, 256),
                ("sg32 ", 1, 256),
                ("sg32x4", 4, 256),
            ] {
                sg4 = match lab {
                    "sg4  " => 4,
                    "sg8  " => 8,
                    _ => 32,
                };
                for _ in 0..20 {
                    let _ = ctx.launch3("qsa_flash_gqa", 1, 4, nb, thr, &mut a4);
                }
                ctx.sync()?;
                let n2 = 2000usize;
                let t0 = std::time::Instant::now();
                for _ in 0..n2 {
                    let _ = ctx.launch3("qsa_flash_gqa", 1, 4, nb, thr, &mut a4);
                }
                ctx.sync()?;
                r2.push_str(&format!(
                    "{lab}: {:.2}us  ",
                    t0.elapsed().as_secs_f64() * 1e6 / n2 as f64
                ));
                let _ = nseg;
            }
            eprintln!("{r2}");
        }
    }
    let mut xp = ctx.alloc(256)?;
    let mut op = ctx.alloc(256)?;
    let mut sp = ctx.alloc(256)?;
    let mut nn = 64i32;
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        (&mut op) as *mut _ as *mut std::ffi::c_void,
        (&mut xp) as *mut _ as *mut std::ffi::c_void,
        (&mut sp) as *mut _ as *mut std::ffi::c_void,
        (&mut nn) as *mut _ as *mut std::ffi::c_void,
    ];
    for _ in 0..10 {
        ctx.launch3("axpy_scaled", 1, 1, 1, 64, &mut args)?;
    }
    ctx.sync()?;
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        ctx.launch3("gemm_xs", 1, 1, 1, 64, &mut args)?;
    }
    let cpu = t0.elapsed();
    ctx.sync()?;
    let wall = t0.elapsed();
    Ok(format!(
        "launch-probe: {n}회 런치 cpu={:.3}ms/회 (동기 포함 wall={:.3}ms/회)",
        cpu.as_secs_f64() * 1e3 / n as f64,
        wall.as_secs_f64() * 1e3 / n as f64
    ))
}

/// plans/84 A — MMQ(gemm_mmq) 교차-t 행 불변 검증: 동일 활성 앞 t1행을
/// t=t1과 t=t2 두 번 계산해 공유 행을 비트 비교한다. 청크 불변성의
/// GEMM 축 펜스 (chunk-check의 커널 레벨 격리용).
pub fn mmq_row_check(path: &str, tname: &str, t1: usize, t2: usize) -> Result<String, String> {
    if t1 == 0 || t2 < t1 {
        return Err("mmq-row-check: 0 < t1 <= t2 필요".into());
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("tensor 없음")?;
    let ty = w.ty as u32;
    if !matches!(ty, 8 | 12 | 13 | 14 | 23) {
        return Err(format!("mmq-row-check: MMQ 타입 아님 (ty={ty})"));
    }
    let ctx = RawCtx::new()?;
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xf: Vec<f32> = (0..t2 * n_in).map(|_| lcg()).collect();
    let xfd = ctx.alloc(xf.len() * 4)?;
    ctx.h2d(xfd, bytemuck::cast_slice(&xf))?;
    let o1 = ctx.alloc(t1 * n_out * 4)?;
    let o2 = ctx.alloc(t2 * n_out * 4)?;
    // 워밍(1회) 후 순차 2회 — 캐시·스트림 상태 차단
    ctx.gemm_mmq(ty, xfd as *const u8, wd, n_in, n_out, t1, o1)?;
    ctx.sync()?;
    ctx.gemm_mmq(ty, xfd as *const u8, wd, n_in, n_out, t1, o1)?;
    ctx.gemm_mmq(ty, xfd as *const u8, wd, n_in, n_out, t2, o2)?;
    ctx.sync()?;
    let mut v1 = vec![0f32; t1 * n_out];
    let mut v2 = vec![0f32; t2 * n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut v1), o1)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut v2), o2)?;
    let mut mism = 0usize;
    let mut maxd = 0f32;
    let mut first: Option<(usize, usize, u32, u32)> = None;
    for r in 0..t1 {
        for c in 0..n_out {
            let (a, b) = (v1[r * n_out + c], v2[r * n_out + c]);
            if a.to_bits() != b.to_bits() {
                mism += 1;
                maxd = maxd.max((a - b).abs());
                if first.is_none() {
                    first = Some((r, c, a.to_bits(), b.to_bits()));
                }
            }
        }
    }
    let verdict = if mism == 0 { "PASS" } else { "FAIL" };
    let mut s = format!(
        "mmq-row-check {tname} ty={ty} t={t1} vs {t2}: {verdict} — {mism}/{} 원소 상이, max|Δ|={maxd:.3e}",
        t1 * n_out
    );
    if let Some((r, c, a, b)) = first {
        s.push_str(&format!(
            "\n  첫 불일치 [{r}][{c}]: {a:#010x} vs {b:#010x} ({:+.6} vs {:+.6})",
            v1[r * n_out + c],
            v2[r * n_out + c]
        ));
    }
    Ok(s)
}

/// plans/84 A — 타일 핀 패밀리(gemm_tile_pin: j128/v4/wm)의 교차-t 행 불변 검증.
/// 프리필 핀 경로(ssm_out q8_0 j128, ffn_down iq4_nl v4 등)의 펜스.
pub fn tile_row_check(path: &str, tname: &str, t1: usize, t2: usize) -> Result<String, String> {
    if t1 == 0 || t2 < t1 {
        return Err("tile-row-check: 0 < t1 <= t2 필요".into());
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("tensor 없음")?;
    let ty = w.ty as u32;
    let ctx = RawCtx::new()?;
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let kt: Vec<u32> = llm170_core::ktab2_packed();
    let ktd = ctx.alloc(kt.len() * 4)?;
    ctx.h2d(ktd, bytemuck::cast_slice(&kt))?;
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xf: Vec<f32> = (0..t2 * n_in).map(|_| lcg()).collect();
    let xfd = ctx.alloc(xf.len() * 4)?;
    ctx.h2d(xfd, bytemuck::cast_slice(&xf))?;
    // 활성 q8 인코딩 — quant_q8 배치(행 단위)
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16; // 엔진 quant_q8_b 배치 스트라이드(정렬 여유 포함)
    // LLM170_TRC_XQ: 엔진 덤프(tr_xqg.f32)를 활성 버퍼로 직접 사용 —
    // 엔진 맥락 재현 (plans/84 A).
    let dump_path = std::env::var_os("LLM170_TRC_XQ");
    let (xq, t2) = if let Some(p) = dump_path {
        let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
        let rows = bytes.len() / (xq_w * 4);
        if rows < t2 {
            return Err(format!("dump 행 부족: {rows} < {t2}"));
        }
        let buf = ctx.alloc(bytes.len())?;
        ctx.h2d(buf, &bytes)?;
        (buf, t2)
    } else {
        let xq = ctx.alloc(t2 * xq_w * 4)?;
        for ti in 0..t2 {
            let row = unsafe { xfd.add(ti * n_in * 4) };
            let dst = unsafe { xq.add(ti * xq_w * 4) };
            ctx.quant_q8(row, dst, n_in)?;
        }
        (xq, t2)
    };
    let _ = &xfd;
    let o1 = ctx.alloc(t1 * n_out * 4)?;
    let o2 = ctx.alloc(t2 * n_out * 4)?;
    // 센티넬: t2>128이면 128 넘은 행이 실제로 기록되는지 검증 (plans/84 A).
    let nan_fill = vec![f32::NAN.to_bits().to_le_bytes(); (t2 * n_out).max(1)];
    let flat: Vec<u8> = nan_fill.iter().flat_map(|b| b.iter().copied()).collect();
    ctx.h2d(o2, &flat)?;
    ctx.gemm_tile_pin(xq as *const u8, wd, ktd, ty, n_in, n_out, xq_w, t1, o1)?;
    // 엔진 조건 재현: t1 넘은 xq 행을 호출 이력마다 다른 값(스크래치 잔존)으로
    // 덮어쓴 뒤 t2 런치 — 부분 타일의 스테일 판독이 유효 행을 오염시키는지.
    if t2 > t1 {
        let junk: Vec<u8> = (0..(t2 - t1) * xq_w * 4)
            .map(|i| (i as u8).wrapping_mul(31))
            .collect();
        ctx.h2d(unsafe { xq.add(t1 * xq_w * 4) }, &junk)?;
    }
    ctx.gemm_tile_pin(xq as *const u8, wd, ktd, ty, n_in, n_out, xq_w, t2, o2)?;
    ctx.sync()?;
    let mut v1 = vec![0f32; t1 * n_out];
    let mut v2 = vec![0f32; t2 * n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut v1), o1)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut v2), o2)?;
    let mut mism = 0usize;
    let mut maxd = 0f32;
    let mut first: Option<(usize, usize)> = None;
    for r in 0..t1 {
        for c in 0..n_out {
            let (a, b) = (v1[r * n_out + c], v2[r * n_out + c]);
            if a.to_bits() != b.to_bits() {
                mism += 1;
                maxd = maxd.max((a - b).abs());
                if first.is_none() {
                    first = Some((r, c));
                }
            }
        }
    }
    let verdict = if mism == 0 { "PASS" } else { "FAIL" };
    let mut s = format!(
        "tile-row-check(pin) {tname} ty={ty} t={t1} vs {t2}: {verdict} — {mism}/{} 상이, max|Δ|={maxd:.3e}",
        t1 * n_out
    );
    if t2 > 128 {
        let unwritten = v2[128 * n_out..].iter().filter(|v| v.is_nan()).count();
        s.push_str(&format!(
            "\n  센티넬: t2>128 중 미기록(NaN 잔존) {unwritten}/{} 원소",
            (t2 - 128) * n_out
        ));
    }
    Ok(s)
}

pub fn mm_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| D_Q35.into());
    let tname = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let t = std::env::var("LLM170_MM_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16usize);
    let mut q8s = Vec::new();
    let mut xq_h: Vec<u32> = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        let blocks = llm170_core::quant::quantize_row_q8_ref(&x);
        for blk in &blocks {
            for c in 0..8 {
                let b = c * 4;
                xq_h.push(
                    (blk.qs[b] as u32 & 0xFF)
                        | ((blk.qs[b + 1] as u32 & 0xFF) << 8)
                        | ((blk.qs[b + 2] as u32 & 0xFF) << 16)
                        | ((blk.qs[b + 3] as u32 & 0xFF) << 24),
                );
            }
        }
        for blk in &blocks {
            xq_h.push(blk.d.to_bits());
        }
        for blk in &blocks {
            let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
            let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
            xq_h.push(s0 as u32);
            xq_h.push(s1 as u32);
        }
        q8s.push(blocks);
    }
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let xq = ctx.alloc(xq_h.len() * 4)?;
    ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    let out = ctx.alloc(n_out * 4 * t)?;
    // wm 상한 64 + bench GEMV 그리드 부적합: 128-커널 계열 미로드면 에러
    let (v4, j128f, odd) = (
        ctx.co_loaded(CO_V4),
        ctx.co_loaded(CO_J128),
        ctx.co_loaded(CO_ODD),
    );
    let big_ok = match w.ty {
        llm170_gguf::GgmlType::Q5K | llm170_gguf::GgmlType::Q4K | llm170_gguf::GgmlType::Iq4Xs => {
            v4 || j128f
        }
        llm170_gguf::GgmlType::Q6K | llm170_gguf::GgmlType::Q8_0 => j128f,
        llm170_gguf::GgmlType::Iq4Nl | llm170_gguf::GgmlType::Q3K | llm170_gguf::GgmlType::Iq3S => {
            odd
        }
        _ => true,
    };
    if t > 64 && !big_ok {
        return Err(format!("mm-bench 미지원: t={t}는 타입별 128-커널 필요"));
    }
    let kern_name = match w.ty {
        llm170_gguf::GgmlType::Q5K => {
            if std::env::var_os("LLM170_MM_WM8").is_some() {
                "gemm_q5k_wm8"
            } else if std::env::var_os("LLM170_EXACT").is_some() {
                // EXACT 우선(v4/j128 CO 상재 무관 mm 강제 — 베이스라인 측정용)
                "gemm_q5k_mm"
            } else if v4 {
                "gemm_q5k_v4"
            } else if j128f {
                "gemm_q5k_j128"
            } else if std::env::var_os("LLM170_EXACT").is_none() {
                "gemm_q5k_wm"
            } else {
                "gemm_q5k_mm"
            }
        }
        llm170_gguf::GgmlType::Q4K => {
            if std::env::var_os("LLM170_MM_WM8").is_some() {
                "gemm_q4k_wm8"
            } else if std::env::var_os("LLM170_EXACT").is_some() {
                "gemm_q4k_mm"
            } else if v4 {
                "gemm_q4k_v4"
            } else if j128f {
                "gemm_q4k_j128"
            } else if std::env::var_os("LLM170_EXACT").is_none() {
                "gemm_q4k_wm"
            } else {
                "gemm_q4k_mm"
            }
        }
        llm170_gguf::GgmlType::Q6K => {
            if std::env::var_os("LLM170_EXACT").is_some() {
                "gemm_q6k_mm"
            } else if j128f {
                "gemm_q6k_j128"
            } else if std::env::var_os("LLM170_EXACT").is_none() {
                "gemm_q6k_wm"
            } else {
                "gemm_q6k_mm"
            }
        }
        llm170_gguf::GgmlType::Q8_0 => {
            if j128f {
                "gemm_q8_j128"
            } else {
                return Err("mm-bench 미지원: q8_0은 j128 커널 필요".into());
            }
        }
        llm170_gguf::GgmlType::Iq4Xs => {
            if std::env::var_os("LLM170_MM_WM8").is_some() {
                "gemm_xs_wm8"
            } else if std::env::var_os("LLM170_EXACT").is_some() {
                "gemm_xs_mm"
            } else if v4 {
                "gemm_xs_v4"
            } else if j128f {
                "gemm_xs_j128"
            } else if std::env::var_os("LLM170_EXACT").is_none() {
                "gemm_xs_wm"
            } else {
                "gemm_xs_mm"
            }
        }
        llm170_gguf::GgmlType::Iq4Nl => {
            if odd {
                "gemm_nl_v4"
            } else {
                return Err("mm-bench 미지원: iq4_nl 타일은 odd CO 필요".into());
            }
        }
        llm170_gguf::GgmlType::Q3K => {
            if odd {
                "gemm_q3k_v4"
            } else {
                return Err("mm-bench 미지원: q3_K 타일은 odd CO 필요".into());
            }
        }
        llm170_gguf::GgmlType::Iq3S => {
            if odd {
                "gemm_iq3s_v4"
            } else {
                return Err("mm-bench 미지원: iq3_s 타일은 odd CO 필요".into());
            }
        }
        _ => "gemm_xs_mm",
    };

    let launch = |ctx: &RawCtx| -> Result<(), String> {
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ktp = kt_d as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
        ];
        if kern_name == "gemm_xs_mm"
            || kern_name == "gemm_xs_wm"
            || kern_name == "gemm_xs_wm8"
            || kern_name == "gemm_xs_j128"
            || kern_name == "gemm_xs_v4"
            || kern_name == "gemm_nl_v4"
        {
            args.push((&mut ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut tt) as *mut _ as *mut std::ffi::c_void);
        let rpb = if kern_name.ends_with("_j128") || kern_name.ends_with("_v4") {
            128
        } else {
            64
        };
        let gx = n_out.div_ceil(rpb).min(65535) as u32;
        let _gz = n_out.div_ceil(rpb).div_ceil(65535) as u32;
        let gz = if kern_name.ends_with("_wm8") {
            t.div_ceil(16) as u32
        } else {
            n_out.div_ceil(64).div_ceil(65535) as u32
        };
        let thr = if kern_name.ends_with("_wm8") { 128 } else { 256 };
        ctx.launch3(kern_name, gx, 1, gz, thr, &mut args)
    };
    launch(&ctx)?;
    ctx.sync()?;
    let mut o2 = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o2).as_mut(), out)?;
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        launch(&ctx)?;
    }
    ctx.sync()?;
    let dt2 = t0.elapsed().as_secs_f64() / reps as f64;
    // 순수 런치 CPU 비용: 그리드 1x1 소형 발사 (GPU 즉시 완료) 100회
    // xs/nl 계열은 ktab2 8인자 — 인자수 불일치 발사가 args 배열 초과 독해로
    // 세그폴트(기존 결함 — mm-bench2가 라우팅 누락 죽은 코드라 미노출, 2026-10-06 수리).
    let (mut sxa, mut swa, mut soa) = (xq, wd, out);
    let (mut sni, mut sno, mut sxw, mut stt) = (n_in as i32, n_out as i32, xq_w as i32, t as i32);
    let mut sktp = kt_d;
    let small_ktab = kern_name == "gemm_xs_mm"
        || kern_name == "gemm_xs_wm"
        || kern_name == "gemm_xs_wm8"
        || kern_name == "gemm_xs_j128"
        || kern_name == "gemm_xs_v4"
        || kern_name == "gemm_nl_v4";
    let mut sargs: Vec<*mut std::ffi::c_void> = vec![
        (&mut sxa) as *mut _ as *mut std::ffi::c_void,
        (&mut swa) as *mut _ as *mut std::ffi::c_void,
        (&mut soa) as *mut _ as *mut std::ffi::c_void,
    ];
    if small_ktab {
        sargs.push((&mut sktp) as *mut _ as *mut std::ffi::c_void);
    }
    sargs.push((&mut sni) as *mut _ as *mut std::ffi::c_void);
    sargs.push((&mut sno) as *mut _ as *mut std::ffi::c_void);
    sargs.push((&mut sxw) as *mut _ as *mut std::ffi::c_void);
    sargs.push((&mut stt) as *mut _ as *mut std::ffi::c_void);
    let tl0 = std::time::Instant::now();
    for _ in 0..100 {
        let _ = ctx.launch3(kern_name, 1, 1, 1, 64, &mut sargs);
    }
    let lcpu = tl0.elapsed().as_secs_f64() * 1e3 / 100.0;
    ctx.sync()?;
    eprintln!("launch-cpu: {:.3}ms/회 (1x1x64 소형)", lcpu);
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut m2 = 0usize;
    let mut maxrel = 0f32;
    for ti in 0..t {
        for oo in 0..n_out.min(256) {
            let row = &w.data[oo * rb..];
            let c2 = match w.ty {
                llm170_gguf::GgmlType::Q5K => {
                    llm170_core::quant::dot_row_w4a8_q5k_mm(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Q4K => {
                    llm170_core::quant::dot_row_w4a8_q4k_mm(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Q6K => {
                    llm170_core::quant::dot_row_w4a8_q6k_mm(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Iq4Nl => {
                    llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Q3K => {
                    llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Iq3S => {
                    llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &q8s[ti])
                }
                llm170_gguf::GgmlType::Q8_0 => {
                    let nblk = n_in as usize / 32;
                    let mut acc = 0.0f32;
                    for b in 0..nblk {
                        let wb = &row[b * 34..b * 34 + 34];
                        let h = ((wb[1] as u16) << 8) | wb[0] as u16;
                        let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
                        let exp = ((h >> 10) & 0x1F) as i32;
                        let man = (h & 0x3FF) as f32;
                        let d = if exp == 0 {
                            sign * man * 2f32.powi(-24)
                        } else {
                            sign * (man / 1024.0 + 1.0) * 2f32.powi(exp - 15)
                        };
                        let mut isum = 0i64;
                        for j in 0..32 {
                            let wv = wb[2 + j] as i8 as i64;
                            let yv = q8s[ti][b].qs[j] as i64;
                            isum += wv * yv;
                        }
                        let yd = q8s[ti][b].d;
                        acc += yd * d * isum as f32;
                    }
                    acc
                }
                _ => llm170_core::quant::dot_row_w4a8_iq4xs_mm(row, n_in as u64, &q8s[ti]),
            };
            if kern_name.ends_with("_wm")
                || kern_name.ends_with("_w32")
                || kern_name.ends_with("_j128")
                || kern_name.ends_with("_v4")
            {
                let g = o2[ti * n_out + oo];
                let denom = c2.abs().max(1.0);
                let rel = (g - c2).abs() / denom;
                if rel > maxrel {
                    maxrel = rel;
                }
                if rel > 5e-3 {
                    m2 += 1;
                }
            } else if c2.to_bits() != o2[ti * n_out + oo].to_bits() {
                m2 += 1;
            }
        }
    }
    Ok(format!(
        "mm({kern_name}): {:.3}ms ({:.1}us/tok) mism {m2} maxrel {maxrel:.2e}",
        dt2 * 1e3,
        dt2 * 1e6 / t as f64
    ))
}

/// plans/108 P7 — t=1 q8_0 dmmv(gemm_q8_0_dmmv, f32 활성 직소비) 검증:
/// 동일 활성으로 구경로(quant_q8 + gemm_q8_0_w)와 dmmv를 계산해 f64 CPU
/// 기준과 비교한다. 판정: dmmv 오차 ≤ 2× 구경로 오차 && ≤ 1e-2.
pub fn hip_dmmv_check(path: &str, tname: &str) -> Result<String, String> {
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("tensor 없음")?;
    let ty = w.ty as u32;
    if ty != 8 {
        return Err(format!("hip-dmmv-check: q8_0 전용 (ty={ty})"));
    }
    let ctx = RawCtx::new()?;
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xf: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let xfd = ctx.alloc(n_in * 4)?;
    ctx.h2d(xfd, bytemuck::cast_slice(&xf))?;
    // ── 구경로: quant_q8 + gemm_q8_0_w (t=1 q8_0 종전 산술과 동일 열) ──
    let xq_w = crate::rawhip::q4acc::xq_words(n_in);
    let xq = ctx.alloc(xq_w * 4)?;
    ctx.quant_q8_b(xfd, xq, n_in, xq_w, 1)?;
    let out_old = ctx.alloc(n_out * 4)?;
    {
        let part = ctx.alloc(n_out * 64 * 8)?;
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut pp = part as *mut std::ffi::c_void;
        let mut op = out_old as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut args = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut op as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void,
            &mut no as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
        ];
        ctx.launch3(
            "gemm_q8_0_w",
            n_out.div_ceil(8) as u32,
            1,
            1,
            256,
            &mut args,
        )?;
    }
    // ── 신경로: dmmv (f32 직소비 — 활성 quant 없음) ──
    let out_dm = ctx.alloc(n_out * 4)?;
    ctx.gemv_q8_dmmv_out(xfd, wd, n_in, n_out, out_dm)?;
    ctx.sync()?;
    let mut vo = vec![0f32; n_out];
    let mut vd = vec![0f32; n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut vo), out_old)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut vd), out_dm)?;
    // ── f64 CPU 기준: 블록 34B [d f16][32×i8] dequant 내적 ──
    let nblk = n_in / 32;
    let mut refr = vec![0f64; n_out];
    for o in 0..n_out {
        let mut acc = 0f64;
        let mut bo = o * nblk * 34;
        for b in 0..nblk {
            let d = llm170_core::quant::deq::f16(w.data, bo) as f64;
            let qs = &w.data[bo + 2..bo + 34];
            let mut s = 0f64;
            for (c, &qb) in qs.iter().enumerate() {
                s += (qb as i8 as f64) * xf[b * 32 + c] as f64;
            }
            acc += d * s;
            bo += 34;
        }
        refr[o] = acc;
    }
    let mut eo = 0f64;
    let mut ed = 0f64;
    for o in 0..n_out {
        eo = eo.max((vo[o] as f64 - refr[o]).abs());
        ed = ed.max((vd[o] as f64 - refr[o]).abs());
    }
    let pass = ed <= 2.0 * eo && ed <= 1e-2;
    let verdict = if pass { "PASS" } else { "FAIL" };
    Ok(format!(
        "hip-dmmv-check {tname} ty={ty} n_in={n_in} n_out={n_out}: {verdict} — old_err={eo:.3e} dmmv_err={ed:.3e} (f64 기준)\n  old[0..4]={:?}\n  dmmv[0..4]={:?}\n  ref[0..4]={:?}",
        &vo[..4.min(n_out)],
        &vd[..4.min(n_out)],
        &refr[..4.min(n_out)]
    ))
}

/// plans/108 P7 — MoE direct-ids dmmv(q4_gemm_q4k_dmmv_ids/q5_1_gemm_dmmv_ids,
/// f32 활성 직소비) 검증: 동일 활성·ids로 구경로(t=1 direct-ids: quant_q8 +
/// q4_gemm_q4k_ge_ids / q4_gemm_q5_1_w_ids)와 dmmv를 계산해 f64 CPU 기준과
/// 비교한다. 활성은 t=1 시맨틱(전 행 동일 벡터 — 구경로 q4k_ge_ids가 전 행
/// 0번 활성을 읽는다). 판정: dmmv 오차 ≤ 2× 구경로 오차 && ≤ 1e-2.
pub fn hip_moe_dmmv_check(path: &str, tname: &str) -> Result<String, String> {
    let m = llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = m.w(tname).ok_or("tensor 없음")?;
    let ty = w.ty;
    if !matches!(ty, llm170_gguf::GgmlType::Q4K | llm170_gguf::GgmlType::Q5_1) {
        return Err(format!("hip-moe-dmmv-check: q4_K/q5_1 전용 (ty={ty:?})"));
    }
    let ne = m.hp.n_expert.max(1);
    let (n_in, n_out) = (w.n_in as usize, (w.n_out as usize) / ne);
    let per_expert = w.data.len() / ne;
    let ctx = RawCtx::new()?;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let rows = m.hp.n_expert_used.clamp(1, 16);
    let x0: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let xf: Vec<f32> = x0.repeat(rows);
    let ids: Vec<u32> = (0..rows)
        .map(|_| ((lcg() + 0.5) * ne as f32) as u32 % ne as u32)
        .collect();
    let xfd = ctx.alloc(xf.len() * 4)?;
    ctx.h2d(xfd, bytemuck::cast_slice(&xf))?;
    let ids_d = ctx.alloc(rows * 4)?;
    ctx.h2d(ids_d, bytemuck::cast_slice(&ids))?;
    // ── 구경로: quant_q8 + t=1 direct-ids 커널 (frame_moe_gemm 종전 배치) ──
    let xq_w = crate::rawhip::q4acc::xq_words(n_in);
    let xq = ctx.alloc(xq_w * 4 * rows)?;
    ctx.quant_q8_b(xfd, xq, n_in, xq_w, rows)?;
    let out_old = ctx.alloc(rows * n_out * 4)?;
    let part = ctx.scratch(4)?;
    if ty == llm170_gguf::GgmlType::Q4K {
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut pp = part as *mut std::ffi::c_void;
        let mut op = out_old as *mut std::ffi::c_void;
        let mut ip = ids_d as *mut std::ffi::c_void;
        let (mut ni, mut no) = (n_in as i32, n_out as i32);
        let (mut xw, mut tt, mut eb) = (0i32, rows as i32, per_expert as i32);
        let mut rp: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut args = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut op as *mut _ as *mut std::ffi::c_void,
            &mut ip as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void,
            &mut no as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
            &mut tt as *mut _ as *mut std::ffi::c_void,
            &mut eb as *mut _ as *mut std::ffi::c_void,
            (&mut rp) as *mut _ as *mut std::ffi::c_void,
        ];
        ctx.launch3(
            "q4_gemm_q4k_ge_ids",
            n_out.div_ceil(16) as u32,
            rows.div_ceil(16) as u32,
            1,
            256,
            &mut args,
        )?;
    } else {
        if n_in / 32 > 32 {
            return Err(format!(
                "hip-moe-dmmv-check: q5_1 구경로는 n_sub≤32 필요 (n_in={n_in})"
            ));
        }
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut pp = part as *mut std::ffi::c_void;
        let mut op = out_old as *mut std::ffi::c_void;
        let mut ip = ids_d as *mut std::ffi::c_void;
        let (mut ni, mut no) = (n_in as i32, n_out as i32);
        let (mut xw, mut tt) = (xq_w as i32, rows as i32);
        let mut ew = (per_expert / 4) as i32;
        let mut args = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut op as *mut _ as *mut std::ffi::c_void,
            &mut ip as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void,
            &mut no as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
            &mut tt as *mut _ as *mut std::ffi::c_void,
            &mut ew as *mut _ as *mut std::ffi::c_void,
        ];
        ctx.launch3(
            "q4_gemm_q5_1_w_ids",
            n_out.div_ceil(8) as u32,
            rows as u32,
            1,
            256,
            &mut args,
        )?;
    }
    // ── 신경로: dmmv (f32 직소비 — 활성 quant 없음, 런치 1회) ──
    let out_dm = ctx.alloc(rows * n_out * 4)?;
    let kern = if ty == llm170_gguf::GgmlType::Q4K {
        "q4_gemm_q4k_dmmv_ids"
    } else {
        "q5_1_gemm_dmmv_ids"
    };
    {
        let mut xp = xfd as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut op = out_dm as *mut std::ffi::c_void;
        let mut ip = ids_d as *mut std::ffi::c_void;
        let (mut ni, mut no) = (n_in as i32, n_out as i32);
        let (mut tt, mut eb) = (rows as i32, per_expert as i32);
        let mut args = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut op as *mut _ as *mut std::ffi::c_void,
            &mut ip as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void,
            &mut no as *mut _ as *mut std::ffi::c_void,
            &mut tt as *mut _ as *mut std::ffi::c_void,
            &mut eb as *mut _ as *mut std::ffi::c_void,
        ];
        let wgs = n_out.div_ceil(2);
        ctx.launch3(
            kern,
            rows as u32,
            wgs.min(65535) as u32,
            wgs.div_ceil(65535) as u32,
            64,
            &mut args,
        )?;
    }
    ctx.sync()?;
    let mut vo = vec![0f32; rows * n_out];
    let mut vd = vec![0f32; rows * n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut vo), out_old)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut vd), out_dm)?;
    // ── f64 CPU 기준: dequant_row × 동일 활성 내적 ──
    let (blck, bsize) = ty.block_info();
    let row_bytes = (n_in / blck as usize) * bsize as usize;
    let mut wrow = vec![0f32; n_in];
    let mut eo = 0f64;
    let mut ed = 0f64;
    for r in 0..rows {
        let ebase = ids[r] as usize * per_expert;
        for o in 0..n_out {
            llm170_core::quant::dequant_row(
                ty,
                &w.data[ebase + o * row_bytes..],
                0,
                n_in as u64,
                &mut wrow,
            );
            let mut s = 0f64;
            for i in 0..n_in {
                s += x0[i] as f64 * wrow[i] as f64;
            }
            eo = eo.max((vo[r * n_out + o] as f64 - s).abs());
            ed = ed.max((vd[r * n_out + o] as f64 - s).abs());
        }
    }
    let pass = ed <= 2.0 * eo && ed <= 1e-2;
    let verdict = if pass { "PASS" } else { "FAIL" };
    Ok(format!(
        "hip-moe-dmmv-check {tname} ty={ty:?} n_in={n_in} n_out={n_out}/expert rows={rows} ne={ne}: {verdict} — old_err={eo:.3e} dmmv_err={ed:.3e} (f64 기준)\n  old[0..4]={:?}\n  dmmv[0..4]={:?}",
        &vo[..4.min(vo.len())],
        &vd[..4.min(vd.len())]
    ))
}
