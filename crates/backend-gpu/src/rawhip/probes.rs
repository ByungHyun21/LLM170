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
    let xq = ctx.alloc((n / 4 + n / 32 + n / 16) * 4)?;
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

/// 런치율(호스트) — 원소별 커널(q4_scale)을 N회 비동기 런치하고 µs/런치를 잰다.
/// 프레임 op의 런치당 비용(프로파일 실측 ≈0.285ms)이 **런치 API 자체**인지
/// 프레임 디스패치인지 가른다: 이 값이 작으면 범인은 디스패치 쪽이다.
pub fn launch_rate(iters: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n = 256usize;
    let buf = ctx.alloc(n * 4)?;
    let data: Vec<f32> = vec![1.0; n];
    ctx.h2d(buf, bytemuck::cast_slice(&data))?;
    ctx.sync()?;
    let mut p = buf as *mut std::ffi::c_void;
    let mut s = 1.0f32;
    let mut nn = n as i32;
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        (&mut p) as *mut _ as *mut std::ffi::c_void,
        (&mut s) as *mut _ as *mut std::ffi::c_void,
        (&mut nn) as *mut _ as *mut std::ffi::c_void,
    ];
    for _ in 0..64 {
        ctx.launch3("q4_scale", 1, 1, 1, 128, &mut args)?;
    }
    ctx.sync()?;
    // 그리드 크기를 바꿔가며 — 프레임의 GEMM은 (64,2560)~164k 블록이다.
    // q4_scale은 j<n 가드가 있어 초과 블록은 즉시 반환한다(안전).
    let mut out = String::new();
    for (gx, gy, gz) in [(1u32, 1u32, 1u32), (64, 2560, 1), (2048, 1, 1), (65535, 1, 1)] {
        for _ in 0..32 {
            ctx.launch3("q4_scale", gx, gy, gz, 128, &mut args)?;
        }
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            ctx.launch3("q4_scale", gx, gy, gz, 128, &mut args)?;
        }
        let host = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        ctx.sync()?;
        let tail = t1.elapsed().as_secs_f64();
        out += &format!(
            "grid=({gx},{gy},{gz}): 호스트 {:.1}µs/런치, GPU 꼬리 {:.1}µs/런치\n",
            host * 1e6 / iters as f64,
            tail * 1e6 / iters as f64
        );
    }
    Ok(out)
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
/// WMMA 가용성 게이트 — 기동 1회 측정 후 캐시(플래그 대신 실측).
/// 불가/오차 초과면 어텐션은 스칼라 판(wk8)으로 간다.
pub fn wmma_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| matches!(wmma_probe_both(), Ok((ok, _)) if ok))
}

/// 기기 실측 리포트 — 이름·가용/전체 메모리·호스트↔디바이스 대역폭.
/// 라우트 선택의 근거(기동 1회). UMA면 h2d/d2h가 메모리 대역폭급으로 높고,
/// PCIe 디스크리트면 수 GB/s 수준 — 같은 코드가 이 값으로 상주 정책을 정한다.
/// f32 → f16 비트(호스트측, 프로브 전용 근사).
fn half_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (x >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | frac as u16
}

/// `f16-map` — `.co` f16 GEMM(`gemm_f16_v4`, 래퍼 `gemm_f16_deq`)의 k-축 매핑을
/// **블록별로 분리 측정**한다. 가중치를 한 32원소 블록으로 제한(b)하고 원-핫
/// 활성(k)을 넣으면, 출력의 1 위치가 곧 "x의 k가 어느 열과 곱해지는가"다.
/// (전체 항등으로 한 번에 재면 블록 간 간섭이 섞여 전단사가 깨진다 — plans/65 §12)
pub fn f16_map(n_in_arg: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let (n_out, n_in) = (256usize, n_in_arg.max(128));
    let wd = ctx.alloc(n_out * (n_in / 32) * 34)?;
    let xd = ctx.alloc(n_in * 4 * 8)?;   // t ≤ 8 여유
    let od = ctx.alloc(n_out * 4 * 512)?;   // 커널은 t를 128 사분면 경계까지 쓴다(§29)
    let mut out = String::new();
    for b in 0..8usize {
        // 가중치: 블록 b에만 항등(그 블록의 요소 j가 행 o=j에)
        let mut w = vec![0u8; n_out * (n_in / 32) * 34];
        for o in 0..n_out {
            let blk = &mut w[(o * (n_in / 32) + b) * 34..][..34];
            blk[0] = 0x00;
            blk[1] = 0x3C;
            for l in 0..32 {
                blk[2 + l] = if o == b * 32 + l { 1 } else { 0 };
            }
        }
        ctx.h2d(wd, &w)?;
        let mut pairs = Vec::new();
        for k in 0..n_in {
            let mut x = vec![0.0f32; n_in];
            x[k] = 1.0;
            ctx.h2d(xd, bytemuck::cast_slice(&x))?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut o = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut o), od as *const u8)?;
            let nz: Vec<usize> = o
                .iter()
                .enumerate()
                .filter(|(_, v)| v.abs() > 0.25)
                .map(|(i, _)| i)
                .collect();
            match nz.len() {
                1 => pairs.push((k, nz[0])),
                0 => pairs.push((k, usize::MAX)),
                _ => pairs.push((k, 1000 + nz[0])),
            }
        }
        let distinct = pairs.iter().filter(|(_, o)| *o < 1000).count();
        let holes = pairs.iter().filter(|(_, o)| *o == usize::MAX).count();
        let multi = pairs.iter().filter(|(_, o)| *o >= 1000).count();
        // 값 검증: 같은 블록 구성에서 전 원소 = 1.0 (d=1, q=1) → x=전부 1이면 출력 = n_in
        {
            let mut wv = vec![0u8; n_out * (n_in / 32) * 34];
            for o in 0..n_out {
                for sb in 0..n_in / 32 {
                    let blk2 = &mut wv[(o * (n_in / 32) + sb) * 34..][..34];
                    blk2[0] = 0x00;
                    blk2[1] = 0x3C; // d = 1.0
                    for l in 0..32 {
                        blk2[2 + l] = 1;
                    }
                }
            }
            let wd1 = ctx.alloc(wv.len())?;
            ctx.h2d(wd1, &wv)?;
            let xones = vec![1.0f32; n_in];
            ctx.h2d(xd, bytemuck::cast_slice(&xones))?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd1 as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut ov = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut ov), od as *const u8)?;
            // 두 번째: d = 0.5(f16 0x3800), q = 2 → 같은 1.0 (스케일 경로 검증)
            let mut wv2 = wv.clone();
            for o in 0..n_out {
                for sb in 0..n_in / 32 {
                    let blk2 = &mut wv2[(o * (n_in / 32) + sb) * 34..][..34];
                    blk2[0] = 0x00;
                    blk2[1] = 0x38;
                    for l in 0..32 {
                        blk2[2 + l] = 2;
                    }
                }
            }
            let wd2 = ctx.alloc(wv2.len())?;
            ctx.h2d(wd2, &wv2)?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd2 as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut ov2 = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut ov2), od as *const u8)?;
            // 실효 k-범위: 전원소 1 가중치 + 원-핫 x(j) → out[0]=1이면 j는 기여, 0이면 범위 밖
            let mut contrib = Vec::new();
            let mut missing = Vec::new();
            for j in 0..n_in {
                let mut xo = vec![0.0f32; n_in];
                xo[j] = 1.0;
                ctx.h2d(xd, bytemuck::cast_slice(&xo))?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut oo = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut oo), od as *const u8)?;
                if oo[0].abs() > 0.25 {
                    contrib.push(j);
                } else {
                    missing.push(j);
                }
            }
            // 부호/블록별 스케일 검증: 블록마다 q = l-16 (합 -16), d_블록 = (sb+1)/16
            //  → 기대 out[0] = Σ_블록 d_블록 × (-16)
            let mut wv3 = vec![0u8; n_out * (n_in / 32) * 34];
            let mut expect = 0.0f32;
            for sb in 0..n_in / 32 {
                let dv = ((sb % 16) + 1) as f32 / 16.0;
                for o in 0..n_out {
                    let blk2 = &mut wv3[(o * (n_in / 32) + sb) * 34..][..34];
                    let h = half_bits(dv);
                    blk2[0] = (h & 0xFF) as u8;
                    blk2[1] = (h >> 8) as u8;
                    let mut ssum = 0i32;
                    for l in 0..32 {
                        let q = l as i32 - 16;
                        blk2[2 + l] = q as u8;
                        ssum += q;
                    }
                    if o == 0 {
                        expect += dv * ssum as f32;
                    }
                }
            }
            // 비영 블록 수를 늘려가며 관측(인덱싱 버그의 패턴을 드러낸다)
            let mut per_nb = String::new();
            for nb in [1usize, 2, 8] {
                let mut wv4 = vec![0u8; n_out * (n_in / 32) * 34];
                let mut exp4 = 0.0f32;
                for sb in 0..n_in / 32 {
                    let on = sb < nb;
                    // d는 모든 블록 동일(1/16) — q 패턴만 불균일하게 두어 "블록별 d"와
                    // "블록 0 이상"을 분리한다.
                    let dv = if on { 0.0625f32 } else { 0.0 };
                    for o in 0..n_out {
                        let blk2 = &mut wv4[(o * (n_in / 32) + sb) * 34..][..34];
                        let h = half_bits(dv);
                        blk2[0] = (h & 0xFF) as u8;
                        blk2[1] = (h >> 8) as u8;
                        let mut ssum = 0i32;
                        for l in 0..32 {
                            let q = if on { l as i32 - 16 } else { 0 };
                            blk2[2 + l] = q as u8;
                            ssum += q;
                        }
                        if o == 0 {
                            exp4 += dv * ssum as f32;
                        }
                    }
                }
                let wd4 = ctx.alloc(wv4.len())?;
                ctx.h2d(wd4, &wv4)?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd4 as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut ov4 = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut ov4), od as *const u8)?;
                per_nb += &format!(" nb={nb}: {:.3}/{exp4:.3}", ov4[0]);
            }
            let wd3 = ctx.alloc(wv3.len())?;
            ctx.h2d(wd3, &wv3)?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd3 as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut ov3 = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut ov3), od as *const u8)?;
            out += &format!("# 블록별 기여(실측/기대):{per_nb}\n");
            // 덤프용 마지막 호출: nb=1(블록 0만 비영) 패턴. LLM170_DEQ_DUMP=1 이면
            // 이 호출의 wf16이 /tmp/deq_wf16.f16 에 남는다.
            {
                let mut wv5 = vec![0u8; n_out * (n_in / 32) * 34];
                for sb in 0..n_in / 32 {
                    let on = sb == 0;
                    let dv = if on { 0.0625f32 } else { 0.0 };
                    for o in 0..n_out {
                        let blk2 = &mut wv5[(o * (n_in / 32) + sb) * 34..][..34];
                        let h = half_bits(dv);
                        blk2[0] = (h & 0xFF) as u8;
                        blk2[1] = (h >> 8) as u8;
                        for l in 0..32 {
                            blk2[2 + l] = if on { (l as i32 - 16) as u8 } else { 0 };
                        }
                    }
                }
                let wd5 = ctx.alloc(wv5.len())?;
                ctx.h2d(wd5, &wv5)?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd5 as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut ov5 = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut ov5), od as *const u8)?;
                out += &format!("# 덤프용 nb=1 재호출: out[0]={:.3} (기대 -1.000)\n", ov5[0]);
            }
            // 원소 커버리지 스캔: (행 0, 열 j) 한 원소만 1.0, x=전부 1 → out[0]=1이면
            // 그 j가 GEMM의 읽기 범위 안. j별로 **다른 포인터**를 써야 f16 캐시(포인터 키)를
            // 피한다 — 257행 버퍼에서 j번째 행을 가중치 시작으로 넘긴다.
            {
                // n_out=1 스캔: 가중치 1행 = 8블록 = 272B. j마다 **별도 슬롯(272B)** 을
                // 포인터로 넘겨 캐시 키(포인터)를 회피하고, 1은 그 슬롯 안의 (j/32, j%32)에 둔다.
                let row_bytes = (n_in / 32) * 34;
                let big = ctx.alloc((n_in + 1) * row_bytes)?;
                let zeros = vec![0u8; (n_in + 1) * row_bytes];
                ctx.h2d(big, &zeros)?;
                let mut covered = Vec::new();
                let mut holes = Vec::new();
                for j in 0..n_in {
                    let mut one = vec![0u8; 34];
                    one[0] = 0x00;
                    one[1] = 0x3C;
                    one[2 + (j % 32)] = 1;
                    let wptr = unsafe { big.add(j * row_bytes + (j / 32) * 34) };
                    ctx.h2d(wptr, &one)?;
                    let wp = unsafe { big.add(j * row_bytes) };
                    ctx.gemm_f16_deq(8, xd as *const u8, wp as *const u8, n_in, 1, 1, od)?;
                    ctx.sync()?;
                    let mut oo = vec![0.0f32; 1];
                    ctx.d2h(bytemuck::cast_slice_mut(&mut oo), od as *const u8)?;
                    if oo[0].abs() > 0.25 {
                        covered.push(j);
                    } else {
                        holes.push(j);
                    }
                }
                out += &format!(
                    "# 원소 커버리지: {}/{} (구멍 앞 12: {:?}, covered: {:?})\n",
                    covered.len(),
                    n_in,
                    &holes[..12.min(holes.len())],
                    covered
                );
            }
            out += &format!(
                "# 값검증: 전원소1 → {} (기대 {n_in}) / d=0.5,q=2 → {} (기대 {n_in}) / 부호·스케일 → {} (기대 {expect:.3})\n",
                ov[0], ov2[0], ov3[0]
            );
            out += &format!(
                "# 실효 k범위: 기여 {}개 (앞 12: {:?}) / 누락 {}개 (앞 12: {:?})\n",
                contrib.len(),
                &contrib[..12.min(contrib.len())],
                missing.len(),
                &missing[..12.min(missing.len())]
            );
            // x-측 매핑: 행 0의 각 열 j에 라벨 (j mod 127)+1 을 심고(q8_0 d=1/127),
            // 원-핫 x(j)의 출력값 × 127 = 짝지어진 열 → x가 어디로 가는지 값으로 읽힌다.
            let mut wl = vec![0u8; n_out * (n_in / 32) * 34];
            for sb in 0..n_in / 32 {
                let blk2 = &mut wl[(0 * (n_in / 32) + sb) * 34..][..34];
                blk2[0] = 0x00;
                // d = 1/127 ≈ 0x1C04? → 대신 d=1 로 두고 q 값 자체를 라벨로 쓴다(출력=q).
                blk2[1] = 0x3C;
                for l in 0..32 {
                    let j = sb * 32 + l;
                    blk2[2 + l] = ((j % 127) + 1) as u8;
                }
            }
            ctx.h2d(wd, &wl)?;
            let mut xmap = Vec::new();
            for j in [0usize, 1, 2, 5, 16, 31, 32, 33, 63, 64, 127, 128, 200, 255] {
                if j >= n_in {
                    continue;
                }
                let mut xo = vec![0.0f32; n_in];
                xo[j] = 1.0;
                ctx.h2d(xd, bytemuck::cast_slice(&xo))?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut oo = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut oo), od as *const u8)?;
                // out[0]만 보면 모순이 생긴다(plans/65 §18) — 전체 비영 분포를 찍는다.
                let nz2: Vec<(usize, f32)> = oo
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.abs() > 0.25)
                    .map(|(i, &v)| (i, v))
                    .collect();
                xmap.push((j, nz2));
            }
            out += &format!("# x-측 매핑(j → out[0]): {xmap:?}\n");
            // t>1 검증: 전부-1 가중치, x는 t행 — 행 r의 one-hot j가 행 r로 나와야 한다.
            {
                let tt = 256usize;   // 128 경계를 넘겨 행 블록 z>0까지 검증
                let mut xt = vec![0.0f32; tt * n_in];
                let rows = [0usize, 1, 127, 128, 129, 255];
                for &r in rows.iter() {
                    xt[r * n_in + (r % 7 + 3)] = 1.0;
                }
                let xd2 = ctx.alloc(tt * n_in * 4)?;
                ctx.h2d(xd2, bytemuck::cast_slice(&xt))?;
                let odt = ctx.alloc(tt * n_out * 4)?;
                let _ = &xd;
                ctx.gemm_f16_deq(8, xd2 as *const u8, wd1 as *const u8, n_in, n_out, tt, odt)?;
                ctx.sync()?;
                let mut ot = vec![0.0f32; tt * n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut ot), odt as *const u8)?;
                let mut info = String::new();
                for &r in rows.iter() {
                    let nz: Vec<(usize, f32)> = ot[r * n_out..(r + 1) * n_out]
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.abs() > 0.25)
                        .map(|(i, &v)| (i, v))
                        .collect();
                    info += &format!(
                        " r{r}: nz={} first={:?} val={:.2}",
                        nz.len(),
                        nz.first().map(|(i, _)| *i),
                        nz.first().map(|(_, v)| *v).unwrap_or(0.0)
                    );
                }
                out += &format!("# t>1 검증(t={tt}):{info}\n");
            }
        }
        out += &format!("# blk={b}: 1:1={distinct} 빈칸={holes} 다중={multi}\n");
        if b == 0 {
            for (k, o) in pairs.iter() {
                out += &format!("{k}:{o}\n");
            }
        }
    }
    Ok(out)
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
    let fm = *fns.get("gemm_f16_v4").ok_or("gemm_f16_v4 없음(co/mmq2.co 미로드)")?;
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
                ((n_out + 127) / 128) as u32,
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
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let n_super = n_in / 256;
    let nblk = n_out * n_super;
    let mut w = vec![0u8; nblk * 144];
    for b in 0..nblk {
        let o = &mut w[b * 144..(b + 1) * 144];
        o[0] = 0x00; o[1] = 0x38;   // d = 0.5
        o[2] = 0x00; o[3] = 0x30;   // dmin = 0.25
        for j in 0..12 { o[4 + j] = ((b * 7 + j * 13) & 0x3F) as u8; }
        for i in 0..128 { o[16 + i] = ((b * 31 + i * 37) & 0xFF) as u8; }
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
    blk[0] = 0x00; blk[1] = 0x3C;   // d = 1.0
    blk[2] = 0x00; blk[3] = 0x38;   // dmin = 0.5
    for j in 0..12 { blk[4 + j] = (0x15u8.wrapping_mul(j as u8 + 1)) & 0x3F; }
    for i in 0..128 { blk[16 + i] = ((i * 37 + 11) & 0xFF) as u8; }
    let w: Vec<u8> = (0..n_out).flat_map(|o| {
        let mut b = blk.clone();
        b[4] = (b[4].wrapping_add(o as u8)) & 0x3F;
        b
    }).collect();
    let xs: Vec<Vec<f32>> = (0..t)
        .map(|r| (0..n_in).map(|i| (((r * 31 + i) as u64 * 2654435761u64) % 1000) as f32 / 500.0 - 1.0).collect())
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
            if d > max_abs { max_abs = d; }
            if first.is_empty() && d > 1e-4 {
                first = format!(" 첫 불일치 r={r} o={o} gpu={gpu:.6} cpu={cpu:.6}");
            }
        }
    }
    Ok(format!(
        "q4k-micro {n_out}x{n_in} t={t}: max_abs={max_abs:.3e} ({}){first}",
        if max_abs < 1e-4 { "일치" } else { "불일치" }
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
        let (mut xp, mut wp, mut pp, mut op) =
            (xdev, wdev, part, odev);
        let (mut ni, mut no, mut xw, mut tt) = (n_in as i32, n_out as i32, xq_w as i32, rows as i32);
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

/// 가드용 최소 VRAM 조회 (2026-09-16) — 컨텍스트 없이 런타임 질의만.
/// 성공 시 (free, total) 바이트.
pub fn gpu_mem_free() -> Option<(u64, u64)> {
    unsafe {
        let (mut f, mut t) = (0usize, 0usize);
        if hip::hipMemGetInfo(&mut f, &mut t) == hip::hipError_t_hipSuccess {
            Some((f as u64, t as u64))
        } else {
            None
        }
    }
}

pub fn device_report(ctx: &RawCtx) -> String {
    let name = unsafe {
        let mut buf = vec![0i8; 256];
        if hip::hipDeviceGetName(buf.as_mut_ptr(), 256, 0) == hip::hipError_t_hipSuccess {
            std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
        } else {
            "unknown".to_string()
        }
    };
    let (free, total) = unsafe {
        let (mut f, mut t) = (0usize, 0usize);
        let e = hip::hipMemGetInfo(&mut f, &mut t);
        if e == hip::hipError_t_hipSuccess { (f as u64, t as u64) } else { (0, 0) }
    };
    // 호스트↔디바이스 왕복 64 MiB (pageable) — 오프로딩 비용 신호.
    let n = 64usize << 20;
    let mut h2d = 0.0f64;
    let mut d2h = 0.0f64;
    if let Ok(d) = ctx.scratch(n) {
        let src = vec![0x5au8; n];
        let mut dst = vec![0u8; n];
        for rep in 0..3 {
            let t0 = std::time::Instant::now();
            let _ = ctx.h2d(d, &src);
            let dt = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let _ = ctx.d2h(&mut dst, d as *const u8);
            let dt2 = t1.elapsed().as_secs_f64();
            if rep == 2 {
                h2d = n as f64 / dt / 1e9;
                d2h = n as f64 / dt2 / 1e9;
            }
        }
    }
    format!(
        "# device: {name} | mem free={:.1}GiB total={:.1}GiB | h2d={h2d:.1}GB/s d2h={d2h:.1}GB/s | wmma={}",
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        if wmma_ok() { "ok" } else { "none" }
    )
}

fn half_f32(bits: u16) -> f32 {
    let s = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((bits >> 10) & 0x1F) as i32 - 15;
    let m = (bits & 0x3FF) as f32;
    if e == -15 { s * m * 2f32.powi(-24) } else { s * (1.0 + m / 1024.0) * 2f32.powi(e) }
}

/// plans/70 P1 검증 — f16 dequant-cache GEMM(gemm_q5k_wc)이 인라인 디퀀트판
/// (gemm_q5k_wm)과 **비트 동일** 출력을 내는지 + 처리량 비교. t ≤ 64(wm B16 한계).
pub fn wc_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::ffi::c_void;
    let t = t.clamp(1, 64);
    let model = llm170_core::model::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("tensor 없음")?;
    let is_xs = w.ty == llm170_gguf::GgmlType::Iq4Xs;
    if w.ty != llm170_gguf::GgmlType::Q5K && !is_xs {
        return Err(format!("wc-check: q5_K/iq4_xs 전용 (ty={:?})", w.ty));
    }
    let ctx = RawCtx::new()?;
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    if n_in % 256 != 0 {
        return Err("wc-check: n_in이 256의 배수가 아님".into());
    }
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    // f16 캐시 생성(1회) — xs는 ktab2 LUT 필요
    let ac = ctx.alloc(n_out * n_in * 2)?;
    // 진단: 디퀀트 커널 값 검증 — CPU 대조 (xs만, o=0 앞 8원소)
    if is_xs && std::env::var_os("LLM170_WC_CPUCHK").is_some() {
        ctx.sync()?;
        let mut ac_host = vec![0u16; n_in.min(64) as usize];
        ctx.d2h(unsafe { std::slice::from_raw_parts_mut(ac_host.as_mut_ptr() as *mut u8, ac_host.len() * 2) }, ac)?;
        let f16v = |bits: u16| half_f32(bits);
        let wq = w.data;
        let blocks = n_in >> 8;
        let mut cpu = vec![0f32; 8];
        for (sb2, cv) in cpu.iter_mut().enumerate() {
            let ib = sb2 & 7;
            let wb = 0usize * blocks * 136 + (sb2 >> 3) * 136;
            let wqf = wb >> 2;
            let w0 = u32::from_le_bytes([wq[wqf*4], wq[wqf*4+1], wq[wqf*4+2], wq[wqf*4+3]]);
            let d = f16v((w0 & 0xFFFF) as u16);
            let w1 = u32::from_le_bytes([wq[(wqf+1)*4], wq[(wqf+1)*4+1], wq[(wqf+1)*4+2], wq[(wqf+1)*4+3]]);
            let ls = ((w1 >> ((ib >> 1) * 8 + (ib & 1) * 4)) & 0xF) as i32
                  | ((((w0 >> 16) >> (2 * ib)) & 3) as i32) << 4;
            let ds0 = d * (ls - 32) as f32;
            let qw = (wb + 8 + ib * 16) >> 2;
            let k = sb2 * 4; // sb2=0..7 → k=0,4,8..28 (원소 8개 샘플)
            let qv = u32::from_le_bytes([wq[(qw + ((k & 15) >> 2))*4], wq[(qw + ((k & 15) >> 2))*4+1], wq[(qw + ((k & 15) >> 2))*4+2], wq[(qw + ((k & 15) >> 2))*4+3]]);
            let byte_v = ((qv >> ((k & 3) * 8)) & 0xFF) as u8;
            let kt = llm170_core::ktab2_packed();
            let tt2 = kt[byte_v as usize];
            let val = if k < 16 { ((tt2 & 0xFF) as i8) as i32 } else { ((tt2 >> 8) as i8) as i32 };
            *cv = val as f32 * ds0;
        }
        eprintln!("# wc-cpuchk o=0 k=0..28(4씩): cpu={:?}", &cpu);
        eprintln!("# wc-cpuchk        ac(f16)={:?}",
            (0..8usize).map(|i| f16v(ac_host[i*4])).collect::<Vec<_>>());
        // 전체 체크섬 — 어디든 썼는지
        let mut all16 = vec![0u16; n_out * n_in];
        let _ = ctx.d2h(unsafe { std::slice::from_raw_parts_mut(all16.as_mut_ptr() as *mut u8, all16.len() * 2) }, ac);
        let nz = all16.iter().filter(|&&v| v != 0).count();
        let mut sum = 0f64;
        for &v in all16.iter() { sum += f16v(v) as f64; }
        eprintln!("# wc-cpuchk 전체: nonzero {nz}/{} sum={sum:.3}", all16.len());
    }
    let (dq_kern, wm_kern, wc_kern, warg) = if is_xs {
        ("dequant_f16_xs", "gemm_xs_wm", "gemm_xs_wc", {
            let kt: Vec<u32> = llm170_core::ktab2_packed();
            let ktd = ctx.alloc(kt.len() * 4)?;
            ctx.h2d(ktd, bytemuck::cast_slice(&kt))?;
            ktd
        })
    } else {
        ("dequant_f16_q5k", "gemm_q5k_wm", "gemm_q5k_wc", std::ptr::null_mut())
    };
    if std::env::var_os("LLM170_WC_DBG").is_some() {
        let mut acp2 = ac as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut a2 = vec![(&mut acp2) as *mut _ as *mut c_void, (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void];
        ctx.launch3("dequant_f16_xs_dbg", n_out as u32, 1, 1, 256, &mut a2)?;
        ctx.sync()?;
        let mut probe16 = vec![0u16; 256];
        let _ = ctx.d2h(unsafe { std::slice::from_raw_parts_mut(probe16.as_mut_ptr() as *mut u8, 512) }, ac);
        let nz2 = probe16.iter().filter(|&&v| v != 0).count();
        eprintln!("# wc-dbg 상수쓰기: 첫 256 중 nonzero={nz2} (0,1,2..여야)");
        return Err("디버그 종료".into());
    }
    {
        let mut wp = wd as *mut c_void;
        let mut ktp = warg as *mut c_void;
        let mut acp = ac as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut args = if is_xs {
            vec![(&mut wp) as *mut _ as *mut c_void, (&mut ktp) as *mut _ as *mut c_void, (&mut acp) as *mut _ as *mut c_void,
                 (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void]
        } else {
            vec![(&mut wp) as *mut _ as *mut c_void, (&mut acp) as *mut _ as *mut c_void,
                 (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void]
        };
        ctx.launch3(dq_kern, n_out as u32, 1, 1, 256, &mut args)?;
    }
    // 합성 활성 t행 — quant_q8로 장치 인코딩
    let mut seed = 0x1234_5678u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let xf: Vec<f32> = (0..t * n_in).map(|_| lcg()).collect();
    let xfd = ctx.alloc(xf.len() * 4)?;
    ctx.h2d(xfd, bytemuck::cast_slice(&xf))?;
    let xq_w = n_in / 4 + n_in / 32;
    let xq = ctx.alloc(t * xq_w * 4)?;
    for ti in 0..t {
        let row = unsafe { xfd.add(ti * n_in * 4) };
        let dst = unsafe { xq.add(ti * xq_w * 4) };
        ctx.quant_q8(row, dst, n_in)?;
    }
    let o1 = ctx.alloc(t * n_out * 4)?;
    let o2 = ctx.alloc(t * n_out * 4)?;
    let gx = n_out.div_ceil(64) as u32;
    let launch = |kern: &'static str, wp2: *mut u8, out: *mut u8| -> Result<(), String> {
        let mut xqp = xq as *mut c_void;
        let mut w2 = wp2 as *mut c_void;
        let mut op = out as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args = vec![(&mut xqp) as *mut _ as *mut c_void, (&mut w2) as *mut _ as *mut c_void,
                            (&mut op) as *mut _ as *mut c_void, (&mut ni) as *mut _ as *mut c_void,
                            (&mut no) as *mut _ as *mut c_void, (&mut xw) as *mut _ as *mut c_void,
                            (&mut tt) as *mut _ as *mut c_void];
        ctx.launch3(kern, gx, 1, 1, 256, &mut args)
    };
    // xs 원판은 ktab2를 추가 인자로 받는다 — 런처 분기
    if is_xs {
        let mut xqp = xq as *mut c_void;
        let mut w2 = wd as *mut c_void;
        let mut op = o1 as *mut c_void;
        let mut ktp = warg as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args = vec![(&mut xqp) as *mut _ as *mut c_void, (&mut w2) as *mut _ as *mut c_void,
                            (&mut op) as *mut _ as *mut c_void, (&mut ktp) as *mut _ as *mut c_void,
                            (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void,
                            (&mut xw) as *mut _ as *mut c_void, (&mut tt) as *mut _ as *mut c_void];
        ctx.launch3(wm_kern, gx, 1, 1, 256, &mut args)?;
    } else {
        launch(wm_kern, wd, o1)?;
    }
    launch(wc_kern, ac, o2)?;
    ctx.sync()?;
    let mut b1 = vec![0f32; t * n_out];
    let mut b2 = vec![0f32; t * n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut b1).as_mut(), o1)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut b2).as_mut(), o2)?;
    let (mut bit_same, mut maxd) = (0usize, 0f32);
    let mut first = String::new();
    for i in 0..t * n_out {
        if b1[i].to_bits() == b2[i].to_bits() { bit_same += 1; }
        let d = (b1[i] - b2[i]).abs();
        if d > maxd { maxd = d; }
        if first.is_empty() && d > 1e-4 {
            first = format!(" 첫 불일치 i={i} wm={:e} wc={:e}", b1[i], b2[i]);
        }
    }
    // 처리량 — 각 20회
    let bench = |kern: &'static str, wp2: *mut u8, out: *mut u8| -> Result<f64, String> {
        let reps = 20;
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            launch(kern, wp2, out)?;
        }
        ctx.sync()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64 * 1e3)
    };
    let ms_wm = if is_xs {
        // xs 원판 벤치 (ktab2 인자 포함 런치 20회)
        let reps = 20;
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let mut xqp = xq as *mut c_void;
            let mut w2 = wd as *mut c_void;
            let mut op = o1 as *mut c_void;
            let mut ktp = warg as *mut c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut tt = t as i32;
            let mut args = vec![(&mut xqp) as *mut _ as *mut c_void, (&mut w2) as *mut _ as *mut c_void,
                                (&mut op) as *mut _ as *mut c_void, (&mut ktp) as *mut _ as *mut c_void,
                                (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void,
                                (&mut xw) as *mut _ as *mut c_void, (&mut tt) as *mut _ as *mut c_void];
            ctx.launch3(wm_kern, gx, 1, 1, 256, &mut args)?;
        }
        ctx.sync()?;
        t0.elapsed().as_secs_f64() / reps as f64 * 1e3
    } else {
        bench(wm_kern, wd, o1)?
    };
    let ms_wc = bench(wc_kern, ac, o2)?;
    let tf = |ms: f64| 2.0 * t as f64 * n_in as f64 * n_out as f64 / (ms * 1e-3) / 1e12;
    Ok(format!(
        "wc-check {tname} [{n_out}x{n_in}] t={t}: 비트동일 {bit_same}/{} max|Δ|={maxd:.2e}{first}\n처리량: wm(인라인 디퀀트) {ms_wm:.3}ms={:.1} TFLOPS · wc(f16 캐시) {ms_wc:.3}ms={:.1} TFLOPS ({:.2}x)\n캐시 {:.1}MB (1회 dequant)",
        t * n_out, tf(ms_wm), tf(ms_wc), ms_wm / ms_wc, n_out * n_in * 2 / 1048576,
    ))
}

pub fn wmma_check() -> Result<String, String> {
    wmma_probe_both().map(|(_, m)| m)
}

/// plans/74 N4: raw WMMA(w32) 프래그먼트 ABI 확정 — 4가지 레이아웃 조합을
/// CPU 행렬곱과 대조해 매핑을 고른다.
pub fn wmma2_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let a: Vec<f32> = (0..256).map(|i| (((i / 16) * 3 + (i % 16) * 7) % 11) as f32 - 5.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (((i / 16) * 5 + (i % 16) * 2) % 13) as f32 - 6.0).collect();
    let ah: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    let bh: Vec<u16> = b.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    let ad = ctx.alloc(512)?;
    let bd = ctx.alloc(512)?;
    let cd = ctx.alloc(1024)?;
    ctx.h2d(ad, bytemuck::cast_slice(&ah))?;
    ctx.h2d(bd, bytemuck::cast_slice(&bh))?;
    let mut msg = String::new();
    let mut any_ok = false;
    for mode in 0..4i32 {
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
        ctx.launch3("wmma2_probe", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        let mut maxerr = 0f32;
        for i in 0..16usize {
            for j in 0..16usize {
                let mut sum = 0f32;
                for k in 0..16usize {
                    sum += a[i * 16 + k] * b[k * 16 + j];
                }
                maxerr = maxerr.max((c[i * 16 + j] - sum).abs());
            }
        }
        let ok = maxerr <= 2e-2;
        any_ok |= ok;
        msg += &format!("mode{mode} (D={}, AB={}): max|Δ|={maxerr:.4} {}\n",
            if mode & 1 == 0 { "2l+g" } else { "l+8g" },
            if mode & 2 == 0 { "contig" } else { "stride" },
            if ok { "★ 일치" } else { "" });
    }
    Ok(format!("wmma2(raw builtin w32) ABI 프로브:\n{msg}{}", if any_ok { "" } else { "전 불일치 — 매핑 재역추론 필요" }))
}

/// plans/74 N4: (lane,l)→(i,k) 매핑 역추론 — 단일원소 행렬 512조합 덤프.
pub fn wmma2_map() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let ad = ctx.alloc(512)?;
    let bd = ctx.alloc(512)?;
    let cd = ctx.alloc(32 * 8 * 4)?;
    let mut amap = vec![(-1i32, -1i32); 32 * 8]; // (row, col=contraction)
    let mut bmap = vec![(-1i32, -1i32); 32 * 8]; // (row=contraction, col)
    let mut ident = vec![0u16; 256];
    for i in 0..16 {
        ident[i * 16 + i] = half::f16::from_f32(1.0).to_bits();
    }
    let mut e = vec![0u16; 256];
    let _ = &e;
    let dump = |ad: usize, bd: usize, ah: &[u16], bh: &[u16]| -> Result<Vec<f32>, String> {
        ctx.h2d(ad as *mut u8, unsafe { std::slice::from_raw_parts(ah.as_ptr() as *const u8, 512) })?;
        ctx.h2d(bd as *mut u8, unsafe { std::slice::from_raw_parts(bh.as_ptr() as *const u8, 512) })?;
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_dump", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        Ok(c)
    };
    // A 매핑: B=I → D=A. A=e_{r,k} 의 1이 어느 (lane,l) 에 나오나.
    for r in 0..16usize {
        for k in 0..16usize {
            for v in e.iter_mut() { *v = 0; }
            e[r * 16 + k] = half::f16::from_f32(1.0).to_bits();
            let c = dump(ad as usize, bd as usize, &e, &ident)?;
            for idx in 0..256usize {
                if c[idx] == 1.0 {
                    if amap[idx].0 == -1 || amap[idx] == (r as i32, k as i32) {
                        amap[idx] = (r as i32, k as i32);
                    }
                }
            }
        }
    }
    // B 매핑: A=I → D=B. B=e_{k,j}.
    for kk in 0..16usize {
        for j in 0..16usize {
            for v in e.iter_mut() { *v = 0; }
            e[kk * 16 + j] = half::f16::from_f32(1.0).to_bits();
            let c = dump(ad as usize, bd as usize, &ident, &e)?;
            for idx in 0..256usize {
                if c[idx] == 1.0 {
                    if bmap[idx].0 == -1 || bmap[idx] == (kk as i32, j as i32) {
                        bmap[idx] = (kk as i32, j as i32);
                    }
                }
            }
        }
    }
    let fmt = |m: &[ (i32, i32) ]| -> String {
        let mut s = String::new();
        for lane in 0..32 {
            s += &format!("lane{lane:2}: ");
            for l in 0..8 {
                let (i, j) = m[lane * 8 + l];
                s += &format!("({i:2},{j:2})");
            }
            s += "\n";
        }
        s
    };
    Ok(format!("A(lane,l)→(row,col):\n{}\nB(lane,l)→(row,col):\n{}", fmt(&amap), fmt(&bmap)))
}

/// plans/74 N4: 랜덤 다중시행 교집합으로 D 레지스터 (lane,l)→(i,j) 확정.
pub fn wmma2_map2() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let ad = ctx.alloc(512)?;
    let bd = ctx.alloc(512)?;
    let cd = ctx.alloc(32 * 8 * 4)?;
    let mut ident = vec![0u16; 256];
    for i in 0..16 {
        ident[i * 16 + i] = half::f16::from_f32(1.0).to_bits();
    }
    let mut cand: Vec<Vec<(u8, u8)>> = vec![vec![]; 256];
    let mut first = true;
    for trial in 0..12u64 {
        let seed = 0x9E3779B97F4A7C15u64.wrapping_mul(trial + 1);
        let a: Vec<f32> = (0..256)
            .map(|i| {
                let h = seed.wrapping_mul(i as u64 + 1);
                ((h >> 33) % 13) as f32 - 6.0
            })
            .collect();
        let ah: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        ctx.h2d(ad, unsafe { std::slice::from_raw_parts(ah.as_ptr() as *const u8, 512) })?;
        ctx.h2d(bd, unsafe { std::slice::from_raw_parts(ident.as_ptr() as *const u8, 512) })?;
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_dump", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        for idx in 0..256usize {
            let mut hits = vec![];
            for i in 0..16usize {
                for j in 0..16usize {
                    if (a[i * 16 + j] - c[idx]).abs() < 1e-3 {
                        hits.push((i as u8, j as u8));
                    }
                }
            }
            if first {
                cand[idx] = hits;
            } else {
                cand[idx].retain(|&h| hits.contains(&h));
            }
        }
        first = false;
    }
    let mut msg = String::new();
    for lane in 0..32 {
        msg += &format!("lane{lane:2}:");
        for l in 0..8 {
            let cs = &cand[lane * 8 + l];
            let s = if cs.len() == 1 {
                format!(" ({},{})", cs[0].0, cs[0].1)
            } else if cs.is_empty() {
                " (?,?)".into()
            } else {
                format!(" {}안", cs.len())
            };
            msg += &s;
        }
        msg += "\n";
    }
    // 진단: lanes 16-31 원시값과 홀수행 기대값 비교(1시행)
    {
        let a: Vec<f32> = (0..256).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect();
        let ah: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        ctx.h2d(ad, unsafe { std::slice::from_raw_parts(ah.as_ptr() as *const u8, 512) })?;
        ctx.h2d(bd, unsafe { std::slice::from_raw_parts(ident.as_ptr() as *const u8, 512) })?;
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_dump", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        msg += "raw lanes16-31: ";
        for lane in 16..32 { for l in 0..8 { msg += &format!("{:.0},", c[lane*8+l]); } }
        msg += "\nA odd rows:    ";
        for i in (1..16).step_by(2) { for j in 0..16 { msg += &format!("{:.0},", a[i*16+j]); } }
        msg += "\n";
    }
    Ok(format!("D(lane,l)→(i,j) 확률적 확정(B=I, D=A):\n{msg}"))
}

fn wmma_probe_both() -> Result<(bool, String), String> {
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
    let mut ok = true;
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
        ok &= maxerr <= 1e-3;
    }
    Ok((ok, msg))
}

/// 합성 어텐션 검증: qsa_flash_wmma 를 작은 단일 케이스로 돌려 **CPU 기준**과 비교한다.
/// 디코드(t=1) GQA 어텐션 v2 검증·계측: 기존 qsa_flash_gqa 와 출력을 대조하고
/// n_past 별로 두 커널의 런치 시간을 잰다. 모델 구성(n_head=24, n_kv=4, hd=256)을 쓴다.
#[allow(unused_assignments)] // kp/vp 는 런치 인자로 넘긴 **주소**가 읽는 값 (raw 포인터 경유)
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
    // 커널은 f16 KV 미러를 읽는다 — 합성 입력도 f16 사본을 만들어 넘긴다.
    let kh = ctx.alloc(kk.len() * 2)?;
    let vh = ctx.alloc(vv.len() * 2)?;
    {
        let mut sp = kd as *mut c_void;
        let mut dp = kh as *mut c_void;
        let mut nn = kk.len() as i32;
        let mut a: Vec<*mut c_void> = vec![&mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void, &mut nn as *mut _ as *mut c_void];
        let nblk = ((kk.len() + 1023) / 1024) as u32;
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a)?;
        let mut sp2 = vd as *mut c_void;
        let mut dp2 = vh as *mut c_void;
        let mut a2: Vec<*mut c_void> = vec![&mut sp2 as *mut _ as *mut c_void,
            &mut dp2 as *mut _ as *mut c_void, &mut nn as *mut _ as *mut c_void];
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a2)?;
        ctx.sync()?;
    }
    let mut qp = qd as *mut c_void;
    let mut kp = kh as *mut c_void;
    let mut vp = vh as *mut c_void;
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
    let det;
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

/// plans/74 N4: qsa_flash_wmma2 검증 — 전체 인과 어텐션 CPU 기준 + 호스트 merge.
pub fn wmma2_attn_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (24usize, 8usize, 256usize);
    let mini = std::env::var_os("LLM170_WMMA2_MINI").is_some();
    let prod = std::env::var_os("LLM170_WMMA2_PROD").is_some();
    let (t, pos0, seg, sstride, n_past) = if mini {
        (16usize, 0usize, 16usize, 256usize, 16usize)
    } else if prod {
        (512usize, 1024usize, 1024usize, 2048usize, 1536usize)
    } else {
        (64usize, 32usize, 16usize, 256usize, 96usize)
    };
    let nseg = (n_past + seg - 1) / seg;
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
        for k in 0..=r.min(n_past - 1) { mask[r * sstride + k] = 1; }
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
    let kh = ctx.alloc(kk.len() * 2)?;
    let vh = ctx.alloc(vv.len() * 2)?;
    for (src, dst) in [(kd, kh), (vd, vh)] {
        let mut sp = src as *mut c_void;
        let mut dp = dst as *mut c_void;
        let mut nn = kk.len() as i32;
        let mut a: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let nblk = ((kk.len() + 1023) / 1024) as u32;
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a)?;
    }
    let mut qp = qd as *mut c_void;
    let mut kp = kh as *mut c_void;
    let mut vp = vh as *mut c_void;
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
    let v2 = std::env::var_os("LLM170_WMMA2V2").is_some();
    ctx.launch3(if v2 { "qsa_flash_wmma2v2" } else { "qsa_flash_wmma2" }, ((t + 15) / 16) as u32, n_head as u32, nseg as u32, 64, &mut args)?;
    ctx.sync()?;
    let mut got = vec![0f32; t * n_head * nseg * (hd + 2)];
    ctx.d2h(bytemuck::cast_slice_mut(&mut got).as_mut(), pd)?;
    // 검증: 세그별 부분 → 호스트 merge → CPU 전체 어텐션(+gate) 대조.
    let mut maxerr = 0f32;
    let mut first = String::new();
    for row in 0..t {
        for hh in 0..n_head {
            let kvh = hh / (n_head / n_kv);
            let mut m = f32::NEG_INFINITY;
            let mut s2v: Vec<f32> = vec![];
            let mut ks: Vec<usize> = vec![];
            for k in 0..n_past {
                if mask[(pos0 + row) * sstride + k] == 0 { continue; }
                let mut s2 = 0f32;
                for d in 0..hd {
                    s2 += qv[(row * n_head + hh) * 2 * hd + d] * kk[(k * n_kv + kvh) * hd + d];
                }
                s2v.push(s2);
                ks.push(k);
                m = m.max(s2);
            }
            let mut ssum = 0f32;
            let mut acc = vec![0f32; hd];
            for (i, &s2) in s2v.iter().enumerate() {
                let e = (s2 - m).exp();
                ssum += e;
                let k = ks[i];
                for d in 0..hd { acc[d] += e * vv[(k * n_kv + kvh) * hd + d]; }
            }
            // 호스트 merge
            let mut m_all = f32::NEG_INFINITY;
            for sgi in 0..nseg {
                let b = ((row * n_head + hh) * nseg + sgi) * (hd + 2);
                m_all = m_all.max(got[b + hd]);
            }
            let mut num = vec![0f32; hd];
            let mut den = 0f32;
            for sgi in 0..nseg {
                let b = ((row * n_head + hh) * nseg + sgi) * (hd + 2);
                let w = (got[b + hd] - m_all).exp();
                den += got[b + hd + 1] * w;
                for d in 0..hd { num[d] += got[b + d] * w; }
            }
            let gate = 1.0f32 / (1.0f32 + (-(qv[(row * n_head + hh) * 2 * hd + hd])).exp());
            for d in 0..hd {
                let refv = if ssum > 0.0 { acc[d] / ssum * gate } else { 0.0 };
                let ourv = if den > 0.0 { num[d] / den * gate } else { 0.0 };
                let dv = (ourv - refv).abs();
                if dv > maxerr { maxerr = dv; }
                if dv > 0.02 && first.is_empty() {
                    first = format!("행{row} 헤드{hh} d{d}: ours {ourv:.5} ref {refv:.5}");
                }
            }
        }
    }
    let mut dbg = String::new();
    if mini {
        let b = 0usize;
        dbg += &format!(" | part0: m={:.4} s={:.4} vk[..4]={:?}", got[b + hd], got[b + hd + 1], &got[b..b + 4]);
    }
    Ok(format!("wmma2-attn-check: max|Δ|={maxerr:.5} {}{}", if maxerr <= 0.02 { "★ PASS".to_string() } else { format!("FAIL {first}") }, dbg))
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
        let _gz = n_out.div_ceil(rpb).div_ceil(65535) as u32;
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
