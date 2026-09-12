//! 커널·프리미티브 검증 프로브 모음 — CLI(`llm170 <probe>`)와 게이트 스크립트가 호출한다.
//! 본체(rawhip/mod.rs)의 비공개 항목을 그대로 쓰기 위해 `use super::*` 로 가져온다.

use super::*;


/// 스파이크 벤치 — 개별 원시 런치 오버헤드 측정 (quant_q8 커널 재사용).
pub fn raw_probe(iters: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n = 1usize << 16;
    let x = ctx.alloc(n * 4)?;
    let xh: Vec<f32> = vec![0.5; n];
    ctx.h2d(x, bytemuck::cast_slice(&xh))?;
    let nblk = n / 32;
    let gx = nblk.div_ceil(64) as u32;
    let mut xq = ctx.alloc((n / 4 + n / 32 + n / 16) * 4)?;
    let mut xp = x as *mut std::ffi::c_void;
    let mut xqp = xq as *mut std::ffi::c_void;
    let mut na = n as i32;
    let mut args = vec![
        &mut xp as *mut _ as *mut std::ffi::c_void,
        &mut xqp as *mut _ as *mut std::ffi::c_void,
        &mut na as *mut _ as *mut std::ffi::c_void,
    ];
    // 워밍
    ctx.launch("quant_q8", gx, 1, 64, &mut args)?;
    ctx.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        ctx.launch("quant_q8", gx, 1, 64, &mut args)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed();
    Ok(format!(
        "원시 런치 {}회 = {:.2}µs/런치 (cubecl 경로 ~10.7µs — 실행기 정상)",
        iters,
        dt.as_secs_f64() * 1e6 / iters as f64
    ))
}

/// qk_norm_rope 단독 검증 — 디코드와 동일 파라미터.
pub fn qk_check() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n_head = 24usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let n_rot = 64usize;
    let half = n_rot / 2;
    let pos = 2usize;
    let rows = n_head + n_kv;
    let aq = ctx.alloc(n_head * 2 * hd * 4)?;
    let ak = ctx.alloc(n_kv * hd * 4)?;
    let qw: Vec<f32> = (0..n_head * hd).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let kw: Vec<f32> = (0..n_kv * hd).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect();
    let qw_d = ctx.alloc(qw.len() * 4)?;
    let kw_d = ctx.alloc(kw.len() * 4)?;
    ctx.h2d(qw_d, bytemuck::cast_slice(&qw))?;
    ctx.h2d(kw_d, bytemuck::cast_slice(&kw))?;
    let hq: Vec<f32> = (0..n_head * 2 * hd).map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.1).collect();
    let hk: Vec<f32> = (0..n_kv * hd).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.1).collect();
    ctx.h2d(aq, bytemuck::cast_slice(&hq))?;
    ctx.h2d(ak, bytemuck::cast_slice(&hk))?;
    let cs: Vec<f32> = (0..2048 * half * 2).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let cs_d = ctx.alloc(cs.len() * 4)?;
    ctx.h2d(cs_d, bytemuck::cast_slice(&cs))?;
    let eps = 1e-5f32;
    let kqs = 0.05f32;
    let mut qp = aq as *mut std::ffi::c_void;
    let mut kp = ak as *mut std::ffi::c_void;
    let mut qwp = qw_d as *mut std::ffi::c_void;
    let mut kwp = kw_d as *mut std::ffi::c_void;
    let mut csp = cs_d as *mut std::ffi::c_void;
    let mut e = eps;
    let mut k = kqs;
    let mut posv = pos as i32;
    let mut nh = n_head as i32;
    let mut nk = n_kv as i32;
    let mut h = hd as i32;
    let mut nr = n_rot as i32;
    fn pp<T>(v: &mut T) -> *mut std::ffi::c_void {
        v as *mut T as *mut std::ffi::c_void
    }
    let mut args = vec![
        pp(&mut qp), pp(&mut kp), pp(&mut qwp), pp(&mut kwp), pp(&mut csp), pp(&mut e),
        pp(&mut k), pp(&mut posv), pp(&mut nh), pp(&mut nk), pp(&mut h), pp(&mut nr),
    ];
    ctx.launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
    ctx.sync()?;
    let mut oq = vec![0f32; n_head * 2 * hd];
    ctx.d2h(bytemuck::cast_slice_mut(&mut oq).as_mut(), aq)?;
    Ok(format!("qk_check ok: q[0]={:.4} q[511]={:.4} q[512]={:.4} k[0]={:.4}", oq[0], oq[511], oq[512], {
        let mut ok = vec![0f32; n_kv * hd];
        ctx.d2h(bytemuck::cast_slice_mut(&mut ok).as_mut(), ak)?;
        ok[0]
    }))
}

/// iq3s 1블록 프로브 — 커널 part[64] vs 미러 lane[64] 레인별 비교.
pub fn iq3s_probe() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let k = 256usize * 68;
    let nblk = k / 256;
    // 결정적 데이터 — 블록별 변화 + 음수 d 포함
    let mut wbytes = vec![0u8; 110 * nblk];
    for b in 0..nblk {
        for i in 0..110 {
            wbytes[b * 110 + i] = ((i * 37 + 11 + b * 13) % 251) as u8;
        }
        if b % 3 == 1 { wbytes[b * 110 + 1] |= 0x80; } // 음수 d
    }
    wbytes[0] = 0x38; wbytes[1] = 0x53;
    let w_d = ctx.alloc(110 * nblk)?;
    ctx.h2d(w_d, &wbytes)?;
    // x 양자화
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 17) as f32 - 8.0) * 0.25).collect();
    let xq = ctx.alloc((k / 4 + k / 32) * 4)?;
    let blocks = llm170_core::quant::quantize_row_q8_ref(&x);
    let mut xq_host: Vec<u32> = Vec::with_capacity(k / 4 + k / 32);
    for b in &blocks {
        for c in 0..8 {
            let w = (b.qs[c * 4] as u32 & 0xFF)
                | ((b.qs[c * 4 + 1] as u32 & 0xFF) << 8)
                | ((b.qs[c * 4 + 2] as u32 & 0xFF) << 16)
                | ((b.qs[c * 4 + 3] as u32 & 0xFF) << 24);
            xq_host.push(w);
        }
    }
    for b in &blocks {
        xq_host.push(b.d.to_bits());
    }
    ctx.h2d(xq, bytemuck::cast_slice(&xq_host))?;
    let part = ctx.alloc(64 * 8)?;
    let mut w_p = w_d as *mut std::ffi::c_void;
    let mut xq_p = xq as *mut std::ffi::c_void;
    let mut part_p = part as *mut std::ffi::c_void;
    let mut ni = k as i32;
    let mut no = 1i32;
    let _ = &no;
    let mut args = vec![
        (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
        (&mut w_p) as *mut _ as *mut std::ffi::c_void,
        (&mut part_p) as *mut _ as *mut std::ffi::c_void,
        (&mut ni) as *mut _ as *mut std::ffi::c_void,
        (&mut no) as *mut _ as *mut std::ffi::c_void,
    ];
    ctx.launch("gemm_iq3s", 1, 1, 64, &mut args)?;
    ctx.sync()?;
    let mut p64 = vec![0f64; 64];
    ctx.d2h(bytemuck::cast_slice_mut(&mut p64).as_mut(), part)?;
    // 미러 — y를 Q8Block으로 재구성
    let y = blocks;
    let lane = llm170_core::quant::dot_row_w4a8_iq3s_lane_parts(&wbytes, k as u64, &y);
    let mut bad = 0;
    let mut msg = String::new();
    for l in 0..64 {
        if p64[l].to_bits() != lane[l].to_bits() {
            bad += 1;
            if bad <= 4 {
                msg += &format!("lane {l}: gpu={:.6e} cpu={:.6e}\n", p64[l], lane[l]);
            }
        }
    }
    Ok(format!("iq3s_probe: {bad}/64 lanes differ\n{msg}"))
}



/// dp4a 가용성 테스트.
pub fn dp4a_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let x: Vec<u32> = vec![0x12, 0x11, 0x00010000, 0x00050000];
    let xd = ctx.alloc(16)?;
    let od = ctx.alloc(32)?;
    ctx.h2d(xd, bytemuck::cast_slice(&x))?;
    let mut xp = xd as *mut std::ffi::c_void;
    let mut op = od as *mut std::ffi::c_void;
    let mut args = vec![
        (&mut xp) as *mut _ as *mut std::ffi::c_void,
        (&mut op) as *mut _ as *mut std::ffi::c_void,
    ];
    ctx.launch("dp4a_probe", 1, 1, 1, &mut args)?;
    ctx.sync()?;
    let mut r = [0i32; 8];
    ctx.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), od)?;
    Ok(format!("sdot8={} (3) udot8={} (5) sdot2={} (204) sdot4={} (204) neg={} (-4) lit_mix={} (-538) bc_mix={} (-538) bc_acc={} (462)", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
}

/// 대역폭 상한 프로브 — q5_K ffn_gate 형상 [5120→17408, 176B] 재현.
pub fn bw_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let (n_in, n_out, bsize) = (5120usize, 17408usize, 176usize);
    let bytes = n_out * (n_in / 256) * bsize;
    let w = ctx.alloc(bytes)?;
    let part = ctx.scratch(n_out * 64 * 8)?;
    let mut wp = w as *mut std::ffi::c_void;
    let mut pp = part as *mut std::ffi::c_void;
    let mut ni = n_in as i32;
    let mut no = n_out as i32;
    let mut bs = bsize as i32;
    let mut args = vec![
        (&mut wp) as *mut _ as *mut std::ffi::c_void,
        (&mut pp) as *mut _ as *mut std::ffi::c_void,
        (&mut ni) as *mut _ as *mut std::ffi::c_void,
        (&mut no) as *mut _ as *mut std::ffi::c_void,
        (&mut bs) as *mut _ as *mut std::ffi::c_void,
    ];
    // 워밍
    ctx.launch("bw_probe", 17408, 1, 64, &mut args)?;
    ctx.sync()?;
    let reps = 30;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        ctx.launch("bw_probe", 17408, 1, 64, &mut args)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let mut r = vec![0f64; 64];
    ctx.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), part)?;
    let _ = r[0];
    Ok(format!("bw_probe: {:.1}us -> {:.0} GB/s (checksum={})", dt * 1e6, bytes as f64 / dt / 1e9, r[63] as u32))
}




/// 배치 mm 타이밍 — gy=1 대비 gy=t 배율.
pub fn mm_batch_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
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
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
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
        msg += &format!("t={}: {:.3}ms ({:.0} GB/s-equiv)\n", t, dt * 1e3, w.data.len() as f64 / dt / 1e9);
    }
    Ok(msg)
}

/// 타일 커널 검증+타이밍 — gemm_q5k_bt vs 미러.
pub fn mm_tile_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
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
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let t = std::env::var("LLM170_TILE_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
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
                xq_h.push((blk.qs[base] as u32 & 0xFF) | ((blk.qs[base+1] as u32 & 0xFF) << 8) | ((blk.qs[base+2] as u32 & 0xFF) << 16) | ((blk.qs[base+3] as u32 & 0xFF) << 24));
            }
        }
        for blk in tok { xq_h.push(blk.d.to_bits()); }
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
        assert_eq!(bytes.len(), xq_h.len() * 4, "덤프 크기 불일치: {} vs {}", bytes.len(), xq_h.len() * 4);
        ctx.h2d(xq, &bytes)?;
        eprintln!("xq 리플레이: {}", f.to_string_lossy());
    } else {
        ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    }
    let out = ctx.alloc(n_out * 4 * t)?;
    let mut wp = wd as *mut std::ffi::c_void;
    let mut op = out as *mut std::ffi::c_void;
    let mut xp = xq as *mut std::ffi::c_void;
    let mut ni = n_in as i32;
    let mut no = n_out as i32;
    let mut xw = xq_w as i32;
    let mut tt = t as i32;
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
                llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &q8s[ti]),
                _ => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &q8s[ti]),
            };
            if c.to_bits() != o[ti * n_out + oo].to_bits() {
                mism += 1;
                if mism == 1 {
                    first_dbg = format!("ti={ti} o={oo}: cpu={c:.7e} gpu={:.7e}", o[ti * n_out + oo]);
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
    Ok(format!("tile t={t}: 불일치 {mism}/{} (첫 256행×t) — {:.3}ms → {:.0} GB/s-equiv, 토큰당 {:.1}µs", n_out.min(256) * t, dt * 1e3, w.data.len() as f64 / dt / 1e9, dt * 1e6 / t as f64))
}

/// dot4 루프-오버헤드 루프 프로브 — 모드별 유효 TIOPS.
/// rocwmma 16x16x16 프래그먼트 레이아웃 검증 — C 레이아웃(idx=lane+32*sl, row=idx>>4,
/// col=idx&15)과 A/B 레이아웃 가정을 정수 데이터로 정확히 확인한다(plans/47).
pub fn wmma_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let a: Vec<f32> = (0..256).map(|i| (((i / 16) * 3 + (i % 16)) % 9) as f32 - 4.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (((i / 16) * 5 + (i % 16)) % 7) as f32 - 3.0).collect();
    let ad = ctx.alloc(256 * 4)?;
    let bd = ctx.alloc(256 * 4)?;
    let cd = ctx.alloc(256 * 4)?;
    ctx.h2d(ad, bytemuck::cast_slice(&a))?;
    ctx.h2d(bd, bytemuck::cast_slice(&b))?;
    let mut msg = String::new();
    for mode in [0i32, 1] {
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut m = mode;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
            (&mut m) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma_probe", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        let mut maxerr = 0f32;
        let mut first = String::new();
        for i in 0..16usize {
            for jj in 0..16usize {
                let mut sum = 0f32;
                for k2 in 0..16usize {
                    let av = a[i * 16 + k2];
                    let bv = if mode == 0 { b[jj * 16 + k2] } else { b[k2 * 16 + jj] };
                    sum += av * bv;
                }
                let idx = i * 16 + jj;
                let d = (c[idx] - sum).abs();
                if d > 1e-3 && first.is_empty() {
                    first = format!(" 첫 불일치 (i={i},j={jj},idx={idx}) 기대 {sum} 실제 {}", c[idx]);
                }
                maxerr = maxerr.max(d);
            }
        }
        msg += &format!("mode{mode}: max|delta| = {maxerr:.6}{first}\n");
    }
    Ok(msg)
}

/// 합성 어텐션 검증: qsa_flash_wmma 를 작은 단일 케이스로 돌려 **CPU 기준**과 비교한다.
/// 디코드(t=1) GQA 어텐션 v2 검증·계측: 기존 qsa_flash_gqa 와 출력을 대조하고
/// n_past 별로 두 커널의 런치 시간을 잰다. 모델 구성(n_head=24, n_kv=4, hd=256)을 쓴다.
pub fn gqa_bench() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (24usize, 4usize, 256usize);
    let sstride = 4096usize;
    let seg = 32usize;
    let n_max = 3314usize;
    let qv: Vec<f32> = (0..n_head * 2 * hd)
        .map(|i| (((i * 1103515245 + 12345) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let kk: Vec<f32> = (0..n_max * n_kv * hd)
        .map(|i| (((i * 214013 + 2531011) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let vv: Vec<f32> = (0..n_max * n_kv * hd)
        .map(|i| (((i * 1260231 + 999983) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let mask: Vec<u32> = vec![1u32; sstride];
    let qd = ctx.alloc(qv.len() * 4)?;
    let kd = ctx.alloc(kk.len() * 4)?;
    let vd = ctx.alloc(vv.len() * 4)?;
    let md = ctx.alloc(mask.len() * 4)?;
    let nseg_max = n_max.div_ceil(seg);
    let npart = n_head * nseg_max * (hd + 2);
    let p1d = ctx.alloc(npart * 4)?;
    let p2d = ctx.alloc(npart * 4)?;
    ctx.h2d(qd, bytemuck::cast_slice(&qv))?;
    ctx.h2d(kd, bytemuck::cast_slice(&kk))?;
    ctx.h2d(vd, bytemuck::cast_slice(&vv))?;
    ctx.h2d(md, bytemuck::cast_slice(&mask))?;
    let mut out = String::new();
    let (mut us1, mut us2) = (0f64, 0f64);
    for n_past in [512usize, 1024, 2048, 3314] {
        let nseg = n_past.div_ceil(seg);
    let p3d = ctx.alloc(npart * 4)?;
    let p4d = ctx.alloc(npart * 4)?;
    let k16 = ctx.alloc(kk.len() * 2)?;
    let v16 = ctx.alloc(vv.len() * 2)?;
    {
        let mut a: Vec<*mut c_void> = Vec::new();
        let mut sp = kd as *mut c_void;
        let mut dp = k16 as *mut c_void;
        let mut nn = kk.len() as i32;
        a.push(&mut sp as *mut _ as *mut c_void);
        a.push(&mut dp as *mut _ as *mut c_void);
        a.push(&mut nn as *mut _ as *mut c_void);
        let nblk = ((kk.len() + 1023) / 1024) as u32;
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a)?;
        let mut sp2 = vd as *mut c_void;
        let mut dp2 = v16 as *mut c_void;
        let mut a2: Vec<*mut c_void> = Vec::new();
        a2.push(&mut sp2 as *mut _ as *mut c_void);
        a2.push(&mut dp2 as *mut _ as *mut c_void);
        a2.push(&mut nn as *mut _ as *mut c_void);
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a2)?;
        ctx.sync()?;
        // 변환 검증: 앞 8개 half 를 되읽어 f32 원본과 비교
        let mut hb = vec![0u16; 8];
        ctx.d2h(bytemuck::cast_slice_mut(&mut hb).as_mut(), k16)?;
        let f0: Vec<f32> = (0..8)
            .map(|i| {
                let h = hb[i] as u16;
                let s = (h >> 15) & 1;
                let e = (h >> 10) & 0x1f;
                let m = h & 0x3ff;
                let v = if e == 0 {
                    (m as f32) * 2f32.powi(-24)
                } else {
                    (1.0 + (m as f32) / 1024.0) * 2f32.powi(e as i32 - 15)
                };
                if s == 1 { -v } else { v }
            })
            .collect();
        eprintln!("# kv_f16 앞 8개: f16={:?}", f0.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
        eprintln!("# kv_f16 원본  : {:?}", kk.iter().take(8).map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    }
        for (lab, pd) in [("v1", p1d), ("v2", p2d), ("v2h", p3d), ("v2d", p4d)] {
            let mut qp = qd as *mut c_void;
            let mut kp = kd as *mut c_void;
            let mut vp = vd as *mut c_void;
            let mut mp = md as *mut c_void;
            let mut pp = pd as *mut c_void;
            let mut np_ = n_past as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut tl = 1i32;
            let mut ss = sstride as i32;
            let mut p0 = 0i32;
            let mut sg = seg as i32;
            let mut args: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void, &mut mp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void, &mut np_ as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void, &mut nk as *mut _ as *mut c_void,
                &mut h as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
                &mut ss as *mut _ as *mut c_void, &mut p0 as *mut _ as *mut c_void,
                &mut sg as *mut _ as *mut c_void,
            ];
            let name = match lab {
                "v1" => "qsa_flash_gqa",
                "v2" => "qsa_flash_gqa2",
                "v2h" => "qsa_flash_gqa2h",
                _ => "qsa_flash_gqa2d",
            };
            if lab == "v2h" || lab == "v2d" {
                // f16 KV 를 읽는 판: ck/cv 자리에 f16 버퍼를 넘긴다(q 는 그대로 f32)
                kp = k16 as *mut c_void;
                vp = v16 as *mut c_void;
            }
            for _ in 0..20 { let _ = ctx.launch3(name, 1, n_kv as u32, nseg as u32, 256, &mut args); }
            ctx.sync()?;
            let iters = 200usize;
            let t0 = std::time::Instant::now();
            for _ in 0..iters { let _ = ctx.launch3(name, 1, n_kv as u32, nseg as u32, 256, &mut args); }
            ctx.sync()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            if lab == "v1" {
                us1 = us;
            } else if lab == "v2h" {
                let mut b = vec![0f32; npart];
                let mut c = vec![0f32; npart];
                ctx.d2h(bytemuck::cast_slice_mut(&mut b).as_mut(), p2d)?;
                ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), p3d)?;
                let cmp_len = n_head * nseg * (hd + 2);
                let mut worst = 0f32;
                let mut bad = 0usize;
                for i in 0..cmp_len {
                    let d = (b[i] - c[i]).abs();
                    let rel = d / (1.0f32 + b[i].abs());
                    if rel > worst { worst = rel; }
                    if rel > 1e-3 { bad += 1; }
                }
                out.push_str(&format!(
                    "n_past={n_past:5}  v1 {us1:8.2}us  v2 {us2:8.2}us  v2h {us:8.2}us  v2h/v2={:.2}x  v2h vs v2 최대상대차 {worst:.2e} (>1e-3 {bad})\n",
                    us2 / us));
            } else if lab == "v2d" {
                let mut b = vec![0f32; npart];
                let mut c = vec![0f32; npart];
                ctx.d2h(bytemuck::cast_slice_mut(&mut b).as_mut(), p2d)?;
                ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), p4d)?;
                let cmp_len = n_head * nseg * (hd + 2);
                let mut worst = 0f32;
                let mut bad = 0usize;
                for i in 0..cmp_len {
                    let d = (b[i] - c[i]).abs();
                    let rel = d / (1.0f32 + b[i].abs());
                    if rel > worst { worst = rel; }
                    if rel > 1e-3 { bad += 1; }
                }
                out.push_str(&format!(
                    "n_past={n_past:5}  v2 {us2:8.2}us  v2d {us:8.2}us  v2d/v2={:.2}x  vs v2 최대상대차 {worst:.2e} (>1e-3 {bad})\n",
                    us2 / us));
            } else {
                us2 = us;
                let mut a = vec![0f32; npart];
                let mut b = vec![0f32; npart];
                ctx.d2h(bytemuck::cast_slice_mut(&mut a).as_mut(), p1d)?;
                ctx.d2h(bytemuck::cast_slice_mut(&mut b).as_mut(), p2d)?;
                let mut worst = 0f32;
                let mut bad = 0usize;
                let cmp_len = n_head * nseg * (hd + 2);   // 기록된 구간만 비교
                for i in 0..cmp_len {
                    let d = (a[i] - b[i]).abs();
                    let rel = d / (1.0f32 + a[i].abs());
                    if rel > worst { worst = rel; }
                    if rel > 1e-4 { bad += 1; }
                }
                out.push_str(&format!(
                    "n_past={n_past:5}  v1 {us1:8.2}us  v2 {us2:8.2}us  v1/v2={:.2}x  최대상대차 {worst:.2e}  불일치 {bad}\n",
                    us1 / us2));
            }
        }
    }
    Ok(out)
}

/// part 규약: seg 별 acc=Σ e_d·v (m,s 는 러닝 최대/합). 첫 불일치 위치를 보고한다.
pub fn wmma_attn_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (4usize, 2usize, 256usize);
    // 후반 청크 재현: pos0>0, n_past>t (실제 모델이 NaN 을 낸 구성)
    let (t, pos0, seg, sstride, ctx_len) = (64usize, 32usize, 16usize, 256usize, 256usize);
    let n_past = 96usize;
    let nseg = (pos0 + t + seg - 1) / seg;   // 4
    let qv: Vec<f32> = (0..t * n_head * 2 * hd)
        .map(|i| (((i * 1103515245 + 12345) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let kk: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 214013 + 2531011) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let vv: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 1260231 + 999983) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let mut mask: Vec<u32> = vec![0u32; (pos0 + t) * sstride];
    for r in 0..(pos0 + t) {
        for k in 0..=r.min(ctx_len - 1) { mask[r * sstride + k] = 1; }
    }
    let qd = ctx.alloc(qv.len() * 4)?;
    let kd = ctx.alloc(kk.len() * 4)?;
    let vd = ctx.alloc(vv.len() * 4)?;
    let md = ctx.alloc(mask.len() * 4)?;
    let pd = ctx.alloc(t * n_head * nseg * (hd + 2) * 4)?;
    ctx.h2d(qd, bytemuck::cast_slice(&qv))?;
    ctx.h2d(kd, bytemuck::cast_slice(&kk))?;
    ctx.h2d(vd, bytemuck::cast_slice(&vv))?;
    ctx.h2d(md, bytemuck::cast_slice(&mask))?;
    let mut qp = qd as *mut c_void;
    let mut kp = kd as *mut c_void;
    let mut vp = vd as *mut c_void;
    let mut mp = md as *mut c_void;
    let mut pp = pd as *mut c_void;
    let mut np_ = n_past as i32;
    let mut nh = n_head as i32;
    let mut nk = n_kv as i32;
    let mut h = hd as i32;
    let mut tl = t as i32;
    let mut ss = sstride as i32;
    let mut p0 = pos0 as i32;
    let mut sg = seg as i32;
    let mut args = vec![
        (&mut qp) as *mut _ as *mut c_void, (&mut kp) as *mut _ as *mut c_void,
        (&mut vp) as *mut _ as *mut c_void, (&mut mp) as *mut _ as *mut c_void,
        (&mut pp) as *mut _ as *mut c_void, (&mut np_) as *mut _ as *mut c_void,
        (&mut nh) as *mut _ as *mut c_void, (&mut nk) as *mut _ as *mut c_void,
        (&mut h) as *mut _ as *mut c_void, (&mut tl) as *mut _ as *mut c_void,
        (&mut ss) as *mut _ as *mut c_void, (&mut p0) as *mut _ as *mut c_void,
        (&mut sg) as *mut _ as *mut c_void,
    ];
    let smem = (4 * 16 * 256 * 2) as u32;   // Q 32768B — K/V·S·P 는 이 버퍼를 재사용한다
    ctx.launch3_dyn("qsa_flash_wmma", (t / 64) as u32, n_head as u32, nseg as u32, 256, smem, &mut args)?;
    ctx.sync()?;
    let mut got = vec![0f32; t * n_head * nseg * (hd + 2)];
    ctx.d2h(bytemuck::cast_slice_mut(&mut got).as_mut(), pd)?;
    // CPU 기준 (f32, 같은 소프트맥스 규약)
    let mut maxerr = 0f32;
    let mut nbad = 0usize;
    let mut first = String::new();
    for row in 0..t {
        for hh in 0..n_head {
            let kvh = hh / (n_head / n_kv);
            let sgc = row / seg;
            let lo = sgc * seg;
            let hi = (lo + seg).min(n_past);
            let mut m = f32::NEG_INFINITY;
            let mut sc: Vec<f32> = Vec::new();
            let mut ks: Vec<usize> = Vec::new();
            for k in lo..hi {
                if mask[(pos0 + row) * sstride + k] == 0 { continue; }   // 인과 마스크 반영
                let mut s2 = 0f32;
                for d in 0..hd {
                    s2 += qv[(row * n_head + hh) * 2 * hd + d] * kk[(k * n_kv + kvh) * hd + d];
                }
                sc.push(s2);
                ks.push(k);
                if s2 > m { m = s2; }
            }
            let mut ssum = 0f32;
            let mut acc = vec![0f32; hd];
            for (i, &s2) in sc.iter().enumerate() {
                let e = (s2 - m).exp();
                ssum += e;
                let k = ks[i];
                for d in 0..hd { acc[d] += e * vv[(k * n_kv + kvh) * hd + d]; }
            }
            let base = ((row * n_head + hh) * nseg + sgc) * (hd + 2);
            for d in 0..hd {
                let dv = (got[base + d] - acc[d]).abs();
                if dv.is_nan() || dv > 2.0 {
                    nbad += 1;
                    if first.is_empty() {
                        first = format!("행{row} 헤드{hh} 세그{sgc} dim{d}: ours {:.4} ref {:.4}", got[base + d], acc[d]);
                    }
                }
                if dv.is_finite() && dv > maxerr { maxerr = dv; }
            }
            let (gm, gs) = (got[base + hd], got[base + hd + 1]);
            if (gm - m).abs() > 1.0 || (gs - ssum).abs() > 1.0 {
                if first.is_empty() {
                    first = format!("행{row} 헤드{hh} 세그{sgc}: m {gm:.4}/{m:.4} s {gs:.4}/{ssum:.4}");
                }
                nbad += 1;
            }
        }
    }
    // 진단 상세: 행0 헤드0 세그0 의 acc 앞 4개 / m / s (ours vs ref)
    let b0 = 0usize;
    let (gm, gs) = (got[b0 + hd], got[b0 + hd + 1]);
    let mut det = String::new();
    {
        let row = 0usize;
        let hh = 0usize;
        let kvh = 0usize;
        let lo = 0usize;
        let hi = seg.min(n_past);
        let mut m = f32::NEG_INFINITY;
        let mut sc: Vec<f32> = Vec::new();
        for k in lo..hi {
            let mut s2 = 0f32;
            for d in 0..hd { s2 += qv[(row * n_head + hh) * 2 * hd + d] * kk[(k * n_kv + kvh) * hd + d]; }
            sc.push(s2);
            if s2 > m { m = s2; }
        }
        let mut ssum = 0f32;
        let mut acc = vec![0f32; hd];
        for (i, &s2) in sc.iter().enumerate() {
            let e = (s2 - m).exp();
            ssum += e;
            for d in 0..4 { acc[d] += e * vv[((lo + i) * n_kv + kvh) * hd + d]; }
        }
        det = format!(" | 행0h0세그0: ours acc {:?} m {:.4} s {:.4} / ref acc {:?} m {:.4} s {:.4}",
            [got[0], got[1], got[2], got[3]], gm, gs, [acc[0], acc[1], acc[2], acc[3]], m, ssum);
    }
    Ok(format!("합성 어텐션: acc 불일치 {nbad}개, max|delta| {maxerr:.4}  {}{}", if first.is_empty() { "전부 일치 ✓".to_string() } else { first }, det))
}

/// PV 경로 프로브: A=P(16x16 ldm=16) x B=V(16x256 **row_major** ldm=256) — 어텐션 PV 와 동일.
pub fn wmma_check_pv() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let pv: Vec<f32> = (0..16 * 16).map(|i| (((i * 7) % 5) as f32) * 0.25).collect();
    let vv: Vec<f32> = (0..16 * 256).map(|i| (((i / 256) * 5 + (i % 256)) % 7) as f32 - 3.0).collect();
    let pd = ctx.alloc(16 * 16 * 4)?;
    let vd = ctx.alloc(16 * 256 * 4)?;
    let cd = ctx.alloc(256 * 4)?;
    ctx.h2d(pd, bytemuck::cast_slice(&pv))?;
    ctx.h2d(vd, bytemuck::cast_slice(&vv))?;
    let mut pp = pd as *mut c_void;
    let mut vp = vd as *mut c_void;
    let mut cp = cd as *mut c_void;
    let mut args = vec![
        (&mut pp) as *mut _ as *mut c_void,
        (&mut vp) as *mut _ as *mut c_void,
        (&mut cp) as *mut _ as *mut c_void,
    ];
    ctx.launch3("wmma_probe_pv", 1, 1, 1, 32, &mut args)?;
    ctx.sync()?;
    let mut c = vec![0f32; 256];
    ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
    let mut maxerr = 0f32;
    let mut nnan = 0usize;
    for row in 0..16usize {
        for dim in 0..16usize {
            let mut sum = 0f32;
            for key in 0..16usize { sum += pv[row * 16 + key] * vv[key * 256 + dim]; }
            let got = c[row * 16 + dim];
            if got.is_nan() { nnan += 1; }
            let d = (got - sum).abs();
            if d > maxerr { maxerr = d; }
        }
    }
    Ok(format!("PV 경로 (B row_major ldm=256): max|delta| = {maxerr:.4}, NaN {nnan}/256"))
}

/// mode2 프로브: 16x256 타일을 ldm=256 으로 적재했을 때 프래그먼트 레이아웃이 맞는지.
pub fn wmma_check_ldm() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let a2: Vec<f32> = (0..16 * 256).map(|i| (((i / 256) * 3 + (i % 256)) % 9) as f32 - 4.0).collect();
    let b2: Vec<f32> = (0..16 * 256).map(|i| (((i / 256) * 5 + (i % 256)) % 7) as f32 - 3.0).collect();
    let ad2 = ctx.alloc(16 * 256 * 4)?;
    let bd2 = ctx.alloc(16 * 256 * 4)?;
    let cd2 = ctx.alloc(256 * 4)?;
    ctx.h2d(ad2, bytemuck::cast_slice(&a2))?;
    ctx.h2d(bd2, bytemuck::cast_slice(&b2))?;
    let mut ap = ad2 as *mut c_void;
    let mut bp = bd2 as *mut c_void;
    let mut cp = cd2 as *mut c_void;
    let mut args = vec![
        (&mut ap) as *mut _ as *mut c_void,
        (&mut bp) as *mut _ as *mut c_void,
        (&mut cp) as *mut _ as *mut c_void,
    ];
    ctx.launch3("wmma_probe_ldm", 1, 1, 1, 256, &mut args)?;
    ctx.sync()?;
    let mut c = vec![0f32; 256];
    ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd2)?;
    let mut maxerr = 0f32;
    let mut nnan = 0usize;
    for i in 0..16usize {
        for jj in 0..16usize {
            let mut sum = 0f32;
            for k2 in 0..256usize { sum += a2[i * 256 + k2] * b2[jj * 256 + k2]; }
            let d = (c[i * 16 + jj] - sum).abs();
            if c[i * 16 + jj].is_nan() { nnan += 1; }
            if d > maxerr { maxerr = d; }
        }
    }
    Ok(format!("mode2 (ldm=256): max|delta| = {maxerr:.4}, NaN {nnan}/256"))
}

/// 두 prefill 어텐션 커널(wk16 vs wk8)에 **동일한** Q/K/V/마스크를 넣고 part 버퍼를 비교한다.
/// 같은 입력에서 part 가 갈리면 커널 버그, 일치하면(또는 반올림 수준이면) 긴 문맥 발산은
/// 재귀 층을 통한 증폭이다. plans/47 의 판별 하네스.
pub fn attn_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (24usize, 4usize, 256usize);
    let (t, pos0, seg, sstride, ctx_len) = (512usize, 1536usize, 128usize, 2048usize, 2048usize);
    let n_past = pos0 + t;
    let nseg = (n_past + seg - 1) / seg;
    // 입력: 결정적 의사난수(양 커널에 동일)
    let qv: Vec<f32> = (0..t * n_head * 2 * hd)
        .map(|i| (((i * 1103515245 + 12345) % 2000) as f32 - 1000.0) * 1e-3)
        .collect();
    let kv_k: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 214013 + 2531011) % 2000) as f32 - 1000.0) * 1e-3)
        .collect();
    let kv_v: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 1260231 + 999983) % 2000) as f32 - 1000.0) * 1e-3)
        .collect();
    let mut mask: Vec<u32> = vec![0u32; (pos0 + t) * sstride];
    for r in 0..(pos0 + t) {
        for k in 0..=r.min(ctx_len - 1) {
            mask[r * sstride + k] = 1;
        }
    }
    let qd = ctx.alloc(qv.len() * 4)?;
    let kd = ctx.alloc(kv_k.len() * 4)?;
    let vd = ctx.alloc(kv_v.len() * 4)?;
    let md = ctx.alloc(mask.len() * 4)?;
    let pd = ctx.alloc(t * n_head * nseg * (hd + 2) * 4)?;
    ctx.h2d(qd, bytemuck::cast_slice(&qv))?;
    ctx.h2d(kd, bytemuck::cast_slice(&kv_k))?;
    ctx.h2d(vd, bytemuck::cast_slice(&kv_v))?;
    ctx.h2d(md, bytemuck::cast_slice(&mask))?;
    let mut out = String::new();
    for (name, gx) in [("qsa_flash_wk16", (t + 15) / 16), ("qsa_flash_wk8", (t + 31) / 32)] {
        let mut qp = qd as *mut c_void;
        let mut kp = kd as *mut c_void;
        let mut vp = vd as *mut c_void;
        let mut mp = md as *mut c_void;
        let mut pp = pd as *mut c_void;
        let mut np_ = n_past as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tl = t as i32;
        let mut ss = sstride as i32;
        let mut p0 = pos0 as i32;
        let mut sg = seg as i32;
        let mut args = vec![
            (&mut qp) as *mut _ as *mut c_void, (&mut kp) as *mut _ as *mut c_void,
            (&mut vp) as *mut _ as *mut c_void, (&mut mp) as *mut _ as *mut c_void,
            (&mut pp) as *mut _ as *mut c_void, (&mut np_) as *mut _ as *mut c_void,
            (&mut nh) as *mut _ as *mut c_void, (&mut nk) as *mut _ as *mut c_void,
            (&mut h) as *mut _ as *mut c_void, (&mut tl) as *mut _ as *mut c_void,
            (&mut ss) as *mut _ as *mut c_void, (&mut p0) as *mut _ as *mut c_void,
            (&mut sg) as *mut _ as *mut c_void,
        ];
        ctx.launch3(name, gx as u32, n_head as u32, nseg as u32, 256, &mut args)?;
        ctx.sync()?;
        let mut v = vec![0f32; t * n_head * nseg * (hd + 2)];
        ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), pd)?;
        if name == "qsa_flash_wk16" {
            std::fs::write("/tmp/attn_wk16.f32", bytemuck::cast_slice(&v)).map_err(|e| e.to_string())?;
        } else {
            std::fs::write("/tmp/attn_wk8.f32", bytemuck::cast_slice(&v)).map_err(|e| e.to_string())?;
        }
        out += &format!("{name}: {gx} blocks, {nseg} segs\n");
    }
    // 비교
    let a = std::fs::read("/tmp/attn_wk16.f32").map_err(|e| e.to_string())?;
    let b = std::fs::read("/tmp/attn_wk8.f32").map_err(|e| e.to_string())?;
    let a: &[f32] = bytemuck::cast_slice(&a);
    let b: &[f32] = bytemuck::cast_slice(&b);
    let mut maxd = 0f32;
    let mut nbad = 0usize;
    let pr = hd + 2;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > maxd { maxd = d; }
        if (i % pr) < hd && d > 1e-4 { nbad += 1; }
    }
    out += &format!("max|delta| = {maxd:.6}, acc 원소(>1e-4) 불일치 {nbad} / {}\n", a.len() / pr * hd);
    Ok(out)
}

pub fn roof_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n_in = 5120usize;
    let xq = ctx.alloc(n_in * 4)?;
    let w = ctx.alloc(n_in * 4)?;
    let out = ctx.alloc(16)?;
    let data: Vec<u32> = (0..n_in).map(|i| (i as u32).wrapping_mul(2654435761)).collect();
    ctx.h2d(xq, bytemuck::cast_slice(&data))?;
    ctx.h2d(w, bytemuck::cast_slice(&data))?;
    let mut msg = String::new();
    for &mode in &[0usize, 1, 2] {
        let iters = 20000usize;
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut m = mode as i32;
        let mut it = iters as i32;
        let mut ni = n_in as i32;
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
            (&mut m) as *mut _ as *mut std::ffi::c_void,
            (&mut it) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
        ];
        // grid: 40CU 채우도록 640블록×64스레드
        ctx.launch3("dot_roof", 640, 1, 1, 64, &mut args)?;
        ctx.sync()?;
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.launch3("dot_roof", 640, 1, 1, 64, &mut args)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let total_dots = iters as f64 * 640.0 * 64.0;
        let tips = total_dots * 4.0 / dt / 1e12; // MAC 4개/dot
        msg += &format!("mode{mode} ({}): {:.2}ms → {:.2} TIOPS MAC\n",
            ["reg-chain", "same-addr load", "stride load"][mode], dt * 1e3, tips);
    }
    // mfma 발행률 — rocwmma 16x16x16f32 (8192 FLOP/wave/mma)
    {
        let ntiles = 64usize;
        let ah = vec![0x3c00u16; ntiles * 256];
        let ad = ctx.alloc(ah.len() * 2)?;
        let bd = ctx.alloc(ah.len() * 2)?;
        ctx.h2d(ad, bytemuck::cast_slice(&ah))?;
        ctx.h2d(bd, bytemuck::cast_slice(&ah))?;
        let om = ctx.alloc(24)?;
        for &mode in &[0usize, 1] {
            let iters = 20000usize;
            let mut ap = ad as *mut std::ffi::c_void;
            let mut bp = bd as *mut std::ffi::c_void;
            let mut op = om as *mut std::ffi::c_void;
            let mut m = mode as i32;
            let mut it = iters as i32;
            let mut nn = ntiles as i32;
            let mut args = vec![
                (&mut ap) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut m) as *mut _ as *mut std::ffi::c_void,
                (&mut it) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("mfma_roof", 640, 1, 1, 64, &mut args)?;
            ctx.sync()?;
            let reps = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                ctx.launch3("mfma_roof", 640, 1, 1, 64, &mut args)?;
            }
            ctx.sync()?;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            let mut o3 = [0f64; 3];
            ctx.d2h(bytemuck::cast_slice_mut(&mut o3).as_mut(), om)?;
            let wavesize = o3[2];
            let waves = 640.0 * 64.0 / wavesize;
            let tflops = waves * iters as f64 * 8192.0 / dt / 1e12;
            msg += &format!("mfma{mode} ({} wave{}): {:.2}ms → {:.2} TFLOPS f32\n",
                ["reg-resident", "L1-fed"][mode], wavesize, dt * 1e3, tflops);
        }
    }
    Ok(msg)
}



/// MMQ 포트 A/B — bt vs mm (각 미러).
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
            &mut xp0 as *mut _ as *mut std::ffi::c_void, &mut wp0 as *mut _ as *mut std::ffi::c_void,
            &mut qp0 as *mut _ as *mut std::ffi::c_void, &mut eps0 as *mut _ as *mut std::ffi::c_void,
            &mut na0 as *mut _ as *mut std::ffi::c_void,
        ];
        let mut s0 = String::new();
        for (label, blk) in [("rmsq n=512", 160u32), ("rmsq n=5120", 512u32), ("rmsq n=20480", 640u32)] {
            let nv: i32 = match label { "rmsq n=512" => 512, "rmsq n=5120" => 5120, _ => 20480 };
            na0 = nv;
            for _ in 0..20 { let _ = ctx.launch("rmsq", 1, 1, blk, &mut a0); }
            ctx.sync()?;
            let n2 = 2000usize;
            let t0 = std::time::Instant::now();
            for _ in 0..n2 { let _ = ctx.launch("rmsq", 1, 1, blk, &mut a0); }
            ctx.sync()?;
            s0.push_str(&format!("{label}: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
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
            &mut ap as *mut _ as *mut std::ffi::c_void, &mut bp as *mut _ as *mut std::ffi::c_void,
            &mut cp as *mut _ as *mut std::ffi::c_void, &mut nn as *mut _ as *mut std::ffi::c_void,
        ];
        let mut s1 = String::new();
        for (label, nval, gx) in [("axpy n=64 1blk", 64i32, 1u32), ("axpy n=5120 80blk", 5120, 80)] {
            nn = nval;
            for _ in 0..20 { let _ = ctx.launch3("axpy_scaled", gx, 1, 1, 64, &mut a2); }
            ctx.sync()?;
            let n2 = 2000usize;
            let t0 = std::time::Instant::now();
            for _ in 0..n2 { let _ = ctx.launch3("axpy_scaled", gx, 1, 1, 64, &mut a2); }
            ctx.sync()?;
            s1.push_str(&format!("{label}: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
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
                &mut op as *mut _ as *mut std::ffi::c_void, &mut zp as *mut _ as *mut std::ffi::c_void,
                &mut wp2 as *mut _ as *mut std::ffi::c_void, &mut qp2 as *mut _ as *mut std::ffi::c_void,
                &mut eps2 as *mut _ as *mut std::ffi::c_void, &mut dd as *mut _ as *mut std::ffi::c_void,
                &mut nh3 as *mut _ as *mut std::ffi::c_void, &mut nt3 as *mut _ as *mut std::ffi::c_void,
            ];
            let mut res = String::new();
            for (nb, thr) in [(1u32, 32u32), (8, 32), (32, 32), (32, 128)] {
                for _ in 0..20 { let _ = ctx.launch3("gatedq", nb, 1, 1, thr, &mut a3); }
                ctx.sync()?;
                let n2 = 2000usize;
                let t0 = std::time::Instant::now();
                for _ in 0..n2 { let _ = ctx.launch3("gatedq", nb, 1, 1, thr, &mut a3); }
                ctx.sync()?;
                res.push_str(&format!("{nb}blk x{thr}thr: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
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
            let mut np_ = npast; let mut nh4 = nh as i32; let mut nk4 = nkv as i32;
            let mut h4 = hd as i32; let mut tl4 = 1i32; let mut ss4 = 4096i32; let mut p04 = 0i32; let mut sg4 = 32i32;
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
            for (lab, nb, thr) in [("sg4  ", 1u32, 256u32), ("sg8  ", 1, 256), ("sg32 ", 1, 256), ("sg32x4", 4, 256)] {
                sg4 = match lab { "sg4  " => 4, "sg8  " => 8, _ => 32 };
                for _ in 0..20 { let _ = ctx.launch3("qsa_flash_gqa", 1, 4, nb, thr, &mut a4); }
                ctx.sync()?;
                let n2 = 2000usize;
                let t0 = std::time::Instant::now();
                for _ in 0..n2 { let _ = ctx.launch3("qsa_flash_gqa", 1, 4, nb, thr, &mut a4); }
                ctx.sync()?;
                r2.push_str(&format!("{lab}: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
                let _ = nseg;
            }
            eprintln!("{r2}");
        }
    }
    let mut xp = ctx.alloc(256)?; let mut op = ctx.alloc(256)?; let mut sp = ctx.alloc(256)?;
    let mut nn = 64i32;
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        (&mut op) as *mut _ as *mut std::ffi::c_void,
        (&mut xp) as *mut _ as *mut std::ffi::c_void,
        (&mut sp) as *mut _ as *mut std::ffi::c_void,
        (&mut nn) as *mut _ as *mut std::ffi::c_void,
    ];
    for _ in 0..10 { ctx.launch3("axpy_scaled", 1, 1, 1, 64, &mut args)?; }
    ctx.sync()?;
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n { ctx.launch3("gemm_xs", 1, 1, 1, 64, &mut args)?; }
    let cpu = t0.elapsed();
    ctx.sync()?;
    let wall = t0.elapsed();
    Ok(format!("launch-probe: {n}회 런치 cpu={:.3}ms/회 (동기 포함 wall={:.3}ms/회)", cpu.as_secs_f64()*1e3/n as f64, wall.as_secs_f64()*1e3/n as f64))
}

pub fn mm_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
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
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let t = std::env::var("LLM170_MM_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let mut q8s = Vec::new();
    let mut xq_h: Vec<u32> = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        let blocks = llm170_core::quant::quantize_row_q8_ref(&x);
        for blk in &blocks {
            for c in 0..8 {
                let b = c * 4;
                xq_h.push((blk.qs[b] as u32 & 0xFF) | ((blk.qs[b+1] as u32 & 0xFF) << 8) | ((blk.qs[b+2] as u32 & 0xFF) << 16) | ((blk.qs[b+3] as u32 & 0xFF) << 24));
            }
        }
        for blk in &blocks { xq_h.push(blk.d.to_bits()); }
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
    let (v4, j128f, odd) = (co_loaded(CO_V4), co_loaded(CO_J128), co_loaded(CO_ODD));
    let big_ok = match w.ty {
        llm170_gguf::GgmlType::Q5K | llm170_gguf::GgmlType::Q4K | llm170_gguf::GgmlType::Iq4Xs
            => v4 || j128f,
        llm170_gguf::GgmlType::Q6K | llm170_gguf::GgmlType::Q8_0 => j128f,
        llm170_gguf::GgmlType::Iq4Nl | llm170_gguf::GgmlType::Q3K | llm170_gguf::GgmlType::Iq3S => odd,
        _ => true,
    };
    if t > 64 && !big_ok {
        return Err(format!("mm-bench 미지원: t={t}는 타입별 128-커널 필요"));
    }
    let kern_name = match w.ty {
        llm170_gguf::GgmlType::Q5K => if v4 { "gemm_q5k_v4" }
            else if j128f { "gemm_q5k_j128" }
            else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q5k_wm" } else { "gemm_q5k_mm" },
        llm170_gguf::GgmlType::Q4K => if v4 { "gemm_q4k_v4" }
            else if j128f { "gemm_q4k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q4k_wm" } else { "gemm_q4k_mm" },
        llm170_gguf::GgmlType::Q6K => if j128f { "gemm_q6k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q6k_wm" } else { "gemm_q6k_mm" },
        llm170_gguf::GgmlType::Q8_0 => if j128f { "gemm_q8_j128" } else { return Err("mm-bench 미지원: q8_0은 j128 커널 필요".into()) },
        llm170_gguf::GgmlType::Iq4Xs => if v4 { "gemm_xs_v4" }
            else if j128f { "gemm_xs_j128" }
            else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_xs_wm" } else { "gemm_xs_mm" },
        llm170_gguf::GgmlType::Iq4Nl => if odd { "gemm_nl_v4" }
            else { return Err("mm-bench 미지원: iq4_nl 타일은 odd CO 필요".into()) },
        llm170_gguf::GgmlType::Q3K => if odd { "gemm_q3k_v4" }
            else { return Err("mm-bench 미지원: q3_K 타일은 odd CO 필요".into()) },
        llm170_gguf::GgmlType::Iq3S => if odd { "gemm_iq3s_v4" }
            else { return Err("mm-bench 미지원: iq3_s 타일은 odd CO 필요".into()) },
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
        if kern_name == "gemm_xs_mm" || kern_name == "gemm_xs_wm" || kern_name == "gemm_xs_j128" || kern_name == "gemm_xs_v4" || kern_name == "gemm_nl_v4" {
            args.push((&mut ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut tt) as *mut _ as *mut std::ffi::c_void);
        let rpb = if kern_name.ends_with("_j128") || kern_name.ends_with("_v4") { 128 } else { 64 };
        let gx = n_out.div_ceil(rpb).min(65535) as u32;
        let gz = n_out.div_ceil(rpb).div_ceil(65535) as u32;
        let gz = n_out.div_ceil(64).div_ceil(65535) as u32;
        ctx.launch3(kern_name, gx, 1, gz, 256, &mut args)
    };
    launch(&ctx)?;
    ctx.sync()?;
    let mut o2 = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o2).as_mut(), out)?;
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { launch(&ctx)?; }
    ctx.sync()?;
    let dt2 = t0.elapsed().as_secs_f64() / reps as f64;
    // 순수 런치 CPU 비용: 그리드 1x1 소형 발사 (GPU 즉시 완료) 100회
    let (mut sxa, mut swa, mut soa) = (xq as *mut u8, wd as *mut u8, out as *mut u8);
    let (mut sni, mut sno, mut sxw, mut stt) = (n_in as i32, n_out as i32, xq_w as i32, t as i32);
    let mut sargs: Vec<*mut std::ffi::c_void> = vec![
        (&mut sxa) as *mut _ as *mut std::ffi::c_void,
        (&mut swa) as *mut _ as *mut std::ffi::c_void,
        (&mut soa) as *mut _ as *mut std::ffi::c_void,
        (&mut sni) as *mut _ as *mut std::ffi::c_void,
        (&mut sno) as *mut _ as *mut std::ffi::c_void,
        (&mut sxw) as *mut _ as *mut std::ffi::c_void,
        (&mut stt) as *mut _ as *mut std::ffi::c_void,
    ];
    let tl0 = std::time::Instant::now();
    for _ in 0..100 { let _ = ctx.launch3(kern_name, 1, 1, 1, 64, &mut sargs); }
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
                llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q8_0 => {
                    let nblk = n_in as usize / 32;
                    let mut acc = 0.0f32;
                    for b in 0..nblk {
                        let wb = &row[b * 34..b * 34 + 34];
                        let h = ((wb[1] as u16) << 8) | wb[0] as u16;
                        let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
                        let exp = ((h >> 10) & 0x1F) as i32;
                        let man = (h & 0x3FF) as f32;
                        let d = if exp == 0 { sign * man * 2f32.powi(-24) } else { sign * (man / 1024.0 + 1.0) * 2f32.powi(exp - 15) };
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
            if kern_name.ends_with("_wm") || kern_name.ends_with("_w32") || kern_name.ends_with("_j128") || kern_name.ends_with("_v4") {
                let g = o2[ti * n_out + oo];
                let denom = c2.abs().max(1.0);
                let rel = (g - c2).abs() / denom;
                if rel > maxrel { maxrel = rel; }
                if rel > 5e-3 { m2 += 1; }
            } else if c2.to_bits() != o2[ti * n_out + oo].to_bits() { m2 += 1; }
        }
    }
    Ok(format!("mm({kern_name}): {:.3}ms ({:.1}us/tok) mism {m2} maxrel {maxrel:.2e}", dt2 * 1e3, dt2 * 1e6 / t as f64))
}

/// 텐서 차원 출력 (디버그 보조)
pub fn dims_of(path: &str, names: &[&str]) -> String {
    let g = match llm170_gguf::GgufFile::open(std::path::Path::new(path)) {
        Ok(g) => g, Err(e) => return e.to_string(),
    };
    let mut s = String::new();
    for n in names {
        if let Some(t) = g.tensors.iter().find(|t| t.name == *n) {
            s += &format!("{n} ne={:?} ty={:?}\n", t.ne, t.ty);
        }
    }
    s
}

/// 진단: q6_K GEMV ↔ GPU 스칼라 기준 대조. 두 커널이 같은 가중 버퍼·같은 활성을
/// 서로 다른 코드로 소비한다 — 커널 인덱싱 오류와 호스트 인자 문제를 분리한다.
pub fn q6k_ref_probe(path: &str, tname: &str) -> Result<String, String> {
    let model = llm170_core::model::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("tensor 없음")?;
    if w.ty != llm170_gguf::GgmlType::Q6K {
        return Err(format!("q6k-ref: q6_K 전용 (ty={:?})", w.ty));
    }
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    // LLM170_Q6K_EK=k: y = e_k (단위 벡터) → out[o]가 곧 복호된 가중치 W[o][k]
    let x: Vec<f32> = if let Ok(spec) = std::env::var("LLM170_Q6K_EK") {
        let ks: Vec<usize> = spec.split(',').filter_map(|v| v.trim().parse::<usize>().ok()).collect();
        if ks.is_empty() || ks.iter().any(|&k| k >= n_in) {
            return Err("LLM170_Q6K_EK: 인덱스 범위 밖".into());
        }
        let mut v = vec![0f32; n_in];
        for &k in ks.iter() { v[k] = 1.0; }
        v
    } else {
        (0..n_in).map(|_| lcg()).collect()
    };
    // y는 엔진과 같은 커널(quant_q8)로 만든다 — 호스트 인코딩 차이를 배제.
    let xf = ctx.alloc(n_in * 4)?;
    ctx.h2d(xf, bytemuck::cast_slice(&x))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let xq = ctx.alloc(xq_w * 4)?;
    ctx.quant_q8(xf, xq, n_in)?;
    ctx.sync()?;
    // 장치 y를 되읽어 f32로 복원 (스칼라 기준 입력)
    let mut yw = vec![0u32; xq_w];
    ctx.d2h(bytemuck::cast_slice_mut(&mut yw).as_mut(), xq)?;
    let mut y_deq = vec![0f32; n_in];
    for blk in 0..(n_in / 32) {
        let d = f32::from_bits(yw[n_in / 4 + blk]);
        for c in 0..32 {
            let w_ = yw[(blk * 32 + c) / 4];
            let byte = ((w_ >> (((blk * 32 + c) % 4) * 8)) & 0xFF) as u8 as i8;
            y_deq[blk * 32 + c] = byte as f32 * d;
        }
    }
    let yf = ctx.alloc(n_in * 4)?;
    ctx.h2d(yf, bytemuck::cast_slice(&y_deq))?;
    let out_a = ctx.alloc(n_out * 4)?;
    let out_b = ctx.alloc(n_out * 4)?;
    let part = ctx.alloc(n_out * 64 * 8)?;
    // ① 엔진 GEMV (mm_direct와 동일 인자·그리드)
    {
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut pp = part as *mut std::ffi::c_void;
        let mut op = out_a as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut args = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void, &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut pp as *mut _ as *mut std::ffi::c_void, &mut op as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void, &mut no as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
        ];
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        ctx.launch3("gemm_q6k", 1, gy, gz, 64, &mut args)?;
    }
    // ② GPU 스칼라 기준
    {
        let mut yp = yf as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut op = out_b as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut args = vec![
            &mut yp as *mut _ as *mut std::ffi::c_void, &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut op as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void, &mut no as *mut _ as *mut std::ffi::c_void,
        ];
        let gx = (n_out as u32).div_ceil(64);
        ctx.launch("q6k_ref_scalar", gx, 1, 64, &mut args)?;
    }
    ctx.sync()?;
    let mut va = vec![0f32; n_out];
    let mut vb = vec![0f32; n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut va).as_mut(), out_a)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut vb).as_mut(), out_b)?;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut nbad = 0usize;
    for i in 0..n_out {
        let e = (va[i] - vb[i]).abs();
        let r = e / vb[i].abs().max(1e-6);
        if r > 1e-3 { nbad += 1; }
        max_abs = max_abs.max(e);
        max_rel = max_rel.max(r);
    }
    Ok(format!(
        "[q6k-ref] {tname} n_in={n_in} n_out={n_out}: 불일치 {nbad}/{n_out} max_abs={max_abs:.6} max_rel={max_rel:.3e}\n  엔진[0..4]={:?}\n  기준[0..4]={:?}",
        &va[..4.min(n_out)], &vb[..4.min(n_out)]
    ))
}
