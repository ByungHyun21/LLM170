//! probes/misc — 기기·잔여 진단 (probes.rs에서 이동, plans/78 R3).

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

/// 배치별 읽기 대역폭 프로브 (plans/83 D2) — 같은 bw_probe 커널로
/// ① VRAM 커브아웃 여유 상태, ② hipMallocHost(GTT 핀) 버퍼,
/// ③ VRAM을 채운 뒤의 hipMalloc(GTT 스펠) 버퍰를 각각 읽는다.
/// APU에서 무게 초과분(103GB 중 ~45GB)이 어느 속도로 읽히는지 직접 측정.
pub fn bw_place_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    // 스트리밍 판독(배치 무관 DRAM BW): 행 121KB × 17408행 = 2.1GB — L2(32MB)를
    // 65배 초과해 재독 캐시 효과를 제거한다. bsize=6056(=176×34.4→정수).
    let (n_in, n_out, bsize) = (5120usize, 17408usize, 6056usize);
    let bytes = n_out * (n_in / 256) * bsize; // ≈ 2.1 GiB
    let part = ctx.scratch(n_out * 64 * 8)?;
    let mut out = String::new();

    let n_q = (bytes / 8) as u64;
    let run = |w: *mut u8, label: &str, out: &mut String| -> Result<(), String> {
        let mut wp = w as *mut std::ffi::c_void;
        let mut pp = part as *mut std::ffi::c_void;
        let mut nq = n_q;
        let mut args = vec![
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut nq) as *mut u64 as *mut std::ffi::c_void,
        ];
        // 페이지 커밋 + 영-페이지 중복 제거(ROCm 지연 커밋 회피)
        unsafe { crate::rawhip::ck(hip::hipMemset(w as *mut std::ffi::c_void, 0x5A, bytes), "memset")?; }
        ctx.launch("bw_stream", 4096, 1, 256, &mut args)?;
        ctx.sync()?;
        let reps = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.launch("bw_stream", 4096, 1, 256, &mut args)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        out.push_str(&format!(
            "  {label}: {:.0}us → {:.0} GB/s\n",
            dt * 1e6,
            bytes as f64 / dt / 1e9
        ));
        Ok(())
    };

    // ① 여유 VRAM 내 hipMalloc
    let w1 = ctx.alloc(bytes)?;
    // 더미 패턴 기록 (읽기 최적화 방해 없음 — XOR 체크섬만 소비)
    run(w1, "hipMalloc (VRAM 여유)", &mut out)?;

    // ② hipMallocHost — GTT 핀 (호스트 매핑, coherent)
    let mut wh: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        let r = hip::hipMallocHost(&mut wh, bytes);
        if r != hip::hipError_t_hipSuccess {
            out.push_str(&format!("  hipMallocHost 실패: {r:?}\n"));
        } else {
            run(wh as *mut u8, "hipMallocHost (GTT 핀)", &mut out)?;
        }
    }

    // ③ VRAM을 거의 채운 뒤 hipMalloc — 스펠 판정
    let (free, _total) = gpu_mem_free().unwrap_or((0, 0));
    let chunk = 1usize << 30; // 1 GiB
    let mut _guard = 0;
    while free as usize > (_guard + 2) * chunk + (2 << 30) && _guard < 64 {
        match ctx.alloc(chunk) {
            Ok(_p) => _guard += 1,
            Err(_) => break,
        }
    }
    let w3 = ctx.alloc(bytes)?;
    run(w3, "hipMalloc (VRAM 포화 후)", &mut out)?;
    Ok(format!("bw-place ({bytes}B 버퍼, 프로브 30회 평균):\n{out}"))
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

pub(super) fn half_f32(bits: u16) -> f32 {
    let s = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((bits >> 10) & 0x1F) as i32 - 15;
    let m = (bits & 0x3FF) as f32;
    if e == -15 { s * m * 2f32.powi(-24) } else { s * (1.0 + m / 1024.0) * 2f32.powi(e) }
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
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
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

/// plans/84 C — q4_hc_a 폴트 최소 재현: 합성 버퍼로 커널 A 형상을 단독 실행.
/// 폴트 재현 시 코드젠/커널 문제, 무폴트 시 엔진 맥락(프레임 버퍼/B 후속) 문제.
pub fn hca_repro() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let (n, hc, r): (usize, usize, usize) = (2560, 4, 320);
    let total = hc * n;
    let n_sub = total / 32;
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / 2147483648.0) - 1.0
    };
    let res: Vec<f32> = (0..total).map(|_| lcg()).collect();
    let wnorm: Vec<f32> = (0..total).map(|_| 1.0 + lcg() * 0.1).collect();
    let wdown: Vec<u8> = (0..r * n_sub * 34).map(|i| (i as u8).wrapping_mul(7)).collect();
    let winj: Vec<f32> = (0..hc * total).map(|_| lcg()).collect();
    let (mut rp, mut np_, mut wd, mut wi) = (ctx.alloc(total * 4)?, ctx.alloc(total * 4)?, ctx.alloc(wdown.len() + 4096)?, ctx.alloc(winj.len() * 4)?);  // +4KB 슬랙: 꼬리 초과판독 이론
    let (mut lp, mut jp, mut xp) = (ctx.alloc(r * 4)?, ctx.alloc(hc * 4)?, ctx.alloc(total * 4)?);
    ctx.h2d(rp, bytemuck::cast_slice(&res))?;
    ctx.h2d(np_, bytemuck::cast_slice(&wnorm))?;
    ctx.h2d(wd, &wdown)?;
    ctx.h2d(wi, bytemuck::cast_slice(&winj))?;
    let (mut n_a, mut hc_a, mut r_a) = (n as i32, hc as i32, r as i32);
    let mut eps = 1e-5f32;
    let grid = (r + hc + total / 64) as u32;
    for it in 0..8 {
        let mut a: Vec<*mut std::ffi::c_void> = vec![
            &mut rp as *mut _ as *mut std::ffi::c_void,
            &mut np_ as *mut _ as *mut std::ffi::c_void,
            &mut wd as *mut _ as *mut std::ffi::c_void,
            &mut wi as *mut _ as *mut std::ffi::c_void,
            &mut lp as *mut _ as *mut std::ffi::c_void,
            &mut jp as *mut _ as *mut std::ffi::c_void,
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut n_a as *mut _ as *mut std::ffi::c_void,
            &mut hc_a as *mut _ as *mut std::ffi::c_void,
            &mut r_a as *mut _ as *mut std::ffi::c_void,
            &mut eps as *mut _ as *mut std::ffi::c_void,
        ];
        ctx.launch3("q4_hca_repro", grid, 1, 1, 64, &mut a)?;
        ctx.sync()?;   // 매 이터레이션 동기 — 폴트 즉시 노출
        eprintln!("[hca-repro] iter {it} ok");
    }
    let mut lo_out = vec![0f32; r];
    ctx.d2h(bytemuck::cast_slice_mut(&mut lo_out), lp)?;
    let nonz = lo_out.iter().filter(|v| **v != 0.0).count();
    Ok(format!("hca-repro: 8회 실행 무폴트, lo nonzero {nonz}/{r}"))
}
