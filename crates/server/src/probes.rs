//! 원오프 GPU 프로브/체크 서브커맨드 — main.rs에서 이관(plans/35 P4).
//! 본체는 backend-gpu(rawhip 프로브 fn, rawvk check fn)에 있고 여기는
//! 인자 파싱+호출만. 결론난 A/B 하니스(batch-abtest·tree-test·q6k-abtest·
//! exp-ab)는 2026-09-08 폐기.

use std::process::ExitCode;

/// 프로브 커맨드이면 실행해 Some(코드) 반환, 아니면 None.
pub fn run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let r: Result<String, String> = match cmd {
        "gpu-raw-probe" => {
            let iters: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
            llm170_backend_gpu::rawhip::raw_probe(iters)
        }
        "mm-bench2" => llm170_backend_gpu::rawhip::mm_bench(),
        "q6k-ref" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.64.nextn.eh_proj.weight".into());
            llm170_backend_gpu::rawhip::q6k_ref_probe(&path, &tn)
        }
        "launch-probe" => llm170_backend_gpu::rawhip::launch_probe(),
        "vk-mmq-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/tmp/model_link.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(512usize);
            llm170_backend_gpu::rawvk::gemv::vk_mmq_check(&path, &tn, t)
        }
        "vk-gemv-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
            llm170_backend_gpu::rawvk::gemv::gemv_check(&path, &tn, t)
        }
        "vk-sdot-probe" => llm170_backend_gpu::rawvk::gemv::sdot_probe(),
        "vk-gemv8-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ssm_out.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
            llm170_backend_gpu::rawvk::gemv::gemv8_check(&path, &tn, t)
        }
        "mmv-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.1.attn_qkv.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1usize);
            llm170_backend_gpu::rawvk::gemv::mmv_check(&path, &tn, t)
        }
        "dbg-q3b" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_up.weight".into());
            llm170_backend_gpu::rawvk::gemv::q3b_dbg(&path, &tn)
        }
        "dbg-q3" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_up.weight".into());
            llm170_backend_gpu::rawvk::gemv::q3_dbg(&path, &tn)
        }
        "vk-tile-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_down.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(32);
            llm170_backend_gpu::rawvk::gemv::tile_check(&path, &tn, t)
        }
        "subsum-check" => llm170_backend_gpu::rawvk::subsum_check(),
        "gdn-check" => llm170_backend_gpu::rawvk::gdn_check(),
        "vk-check" => llm170_backend_gpu::rawvk::smoke_test(),
        "roof-test" => llm170_backend_gpu::rawhip::roof_test(),
        "wmma-check" => llm170_backend_gpu::rawhip::wmma_check(),
        "attn-check" => llm170_backend_gpu::rawhip::attn_check(),
        "mm-tile" => llm170_backend_gpu::rawhip::mm_tile_bench(),
        "mm-bench" => llm170_backend_gpu::rawhip::mm_batch_bench(),
        "bw-test" => llm170_backend_gpu::rawhip::bw_test(),
        "dp4a-test" => llm170_backend_gpu::rawhip::dp4a_test(),
        "iq3s-probe" => llm170_backend_gpu::rawhip::iq3s_probe(),
        "qk-check" => llm170_backend_gpu::rawhip::qk_check(),
        _ => return special(cmd, args),
    };
    Some(match r {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    })
}

/// 프로브 중 출력 형태가 특수한 것들.
fn special(cmd: &str, args: &[String]) -> Option<ExitCode> {
    match cmd {
        "dims" => {
            let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            print!("{}", llm170_backend_gpu::rawhip::dims_of(a[0], &a[1..]));
            Some(ExitCode::SUCCESS)
        }
        "tty-probe" => {
            let path = args.first().cloned().unwrap_or_else(|| "/tmp/model_link.gguf".into());
            match llm170_gguf::GgufFile::open(std::path::Path::new(&path)) {
                Ok(g) => {
                    use std::collections::BTreeMap;
                    let mut cnt: BTreeMap<u32, usize> = BTreeMap::new();
                    let mut bytes: BTreeMap<u32, u64> = BTreeMap::new();
                    for t in &g.tensors {
                        *cnt.entry(t.ty as u32).or_insert(0) += 1;
                        *bytes.entry(t.ty as u32).or_insert(0) += t.nbytes().unwrap_or(0);
                    }
                    for (k, c) in cnt {
                        println!("ty{k}: {c} tensors {:.1}MB", bytes[&k] as f64 / 1e6);
                    }
                    Some(ExitCode::SUCCESS)
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    Some(ExitCode::FAILURE)
                }
            }
        }
        "rawhip-check" => Some(cmd_rawhip_check(args)),
        _ => None,
    }
}

/// llm170 rawhip-check <file> <tensor> — 원시 HIP GEMV(quant·gemm·reduce)
/// 대 CPU 레인 미러 to_bits 전행 검증 + 속도.
fn cmd_rawhip_check(args: &[String]) -> ExitCode {
    use llm170_backend_gpu::rawhip::RawCtx;
    if args.len() < 2 {
        eprintln!("usage: llm170 rawhip-check <file> <tensor>");
        return ExitCode::from(2);
    }
    let model = match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => { eprintln!("tensor not found: {}", args[1]); return ExitCode::FAILURE; }
    };
    let raw_ok = llm170_core::matmul::w4a8_ty(w.ty) || w.ty == llm170_gguf::GgmlType::Iq3S;
    if !raw_ok {
        eprintln!("rawhip-check: 미지원 타입");
        return ExitCode::FAILURE;
    }
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let ctx = match RawCtx::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
    };
    let y = llm170_core::quant::quantize_row_q8_ref(&x);
    // GPU 양자화 비트 미러 검증 (quant_q8 커널)
    let mut xq_gpu: Option<*mut u8> = None;
    {
        let mut inner = || -> Result<(), String> {
            let xd_buf = ctx.alloc(n_in * 4)?;
            let xq_buf = ctx.alloc((n_in / 4 + n_in / 32) * 4)?; // 워드 + d 비트
            xq_gpu = Some(xq_buf);
            ctx.h2d(xd_buf, bytemuck::cast_slice(&x))?;
            ctx.quant_q8(xd_buf as *const u8, xq_buf, n_in)?;
            let mut gq = vec![0u8; (n_in / 4 + n_in / 32) * 4];
            ctx.d2h(&mut gq, xq_buf)?;
            let gw: Vec<u32> = bytemuck::cast_slice(&gq[..n_in / 4 * 4]).to_vec();
            let mut qm = 0usize;
            let cpu_w: Vec<u32> = {
                let mut v = Vec::new();
                for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
                    let mut word = 0u32;
                    for (i, b) in c.iter().enumerate() { word |= (**b as u8 as u32) << (8 * i); }
                    v.push(word);
                }
                v
            };
            for (i, (a, b)) in gw.iter().zip(cpu_w.iter()).enumerate() {
                if a != b { qm += 1; if qm == 1 { println!("  ✗ quant 워드[{i}] gpu={a:#x} cpu={b:#x}"); } }
            }
            let gdbits: Vec<u32> = bytemuck::cast_slice(&gq[n_in / 4 * 4..]).to_vec();
            for (i, (a, b)) in gdbits.iter().zip(y.iter().map(|b| b.d.to_bits())).enumerate() {
                if *a != b { qm += 1; if qm <= 3 { println!("  ✗ quant d[{i}] gpu_bits={a:#x} cpu_bits={b:#x}"); } }
            }
            if qm == 0 { println!("  ★ quant_q8 원시 ≡ CPU 비트 일치"); }
            Ok(())
        };
        if let Err(e) = inner() { eprintln!("quant 검증: {e}"); }
    }
    let mut qs_words = Vec::with_capacity(n_in / 4);
    for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
        let mut word = 0u32;
        for (i, b) in c.iter().enumerate() {
            word |= (**b as u8 as u32) << (8 * i);
        }
        qs_words.push(word);
    }
    // ktab2
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    // GPU quant 사용 시: xq 버퍼 = 워드+d 통합 (gemv가 직접 판독)
    let xq_d = match xq_gpu {
        Some(p) => p,
        None => {
            // CPU 경로: 워드 + d 비트 통합 패킹
            let buf = ctx.alloc((n_in / 4 + n_in / 32) * 4).expect("alloc");
            let mut packed = qs_words.clone();
            packed.extend(y.iter().map(|b| b.d.to_bits()));
            ctx.h2d(buf, bytemuck::cast_slice(&packed)).expect("pack upload");
            buf
        }
    };
    let w_d = match ctx.alloc(w.data.len()) { Ok(p) => p, Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; } };
    let kt_d = match ctx.alloc(1024) { Ok(p) => p, Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; } };
    // GPU quant 출력 재사용 시 xq/xd 업로드 생략 (종단 검증 — d가 GPU 생산값)
    let up = ctx.h2d(w_d, w.data).and_then(|_| ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2)));
    if let Err(e) = up {
        eprintln!("upload: {e}"); return ExitCode::FAILURE;
    }
    // 워밍 + 측정
    let ty = w.ty as u32;
    let _ = match ctx.gemv_q8(xq_d as *const u8, w_d as *const u8, kt_d as *const u8, ty, n_in, n_out) {
        Ok(v) => v,
        Err(e) => { eprintln!("gemv: {e}"); return ExitCode::FAILURE; }
    };
    let reps = 30;
    let t0 = std::time::Instant::now();
    let mut g = Vec::new();
    for _ in 0..reps {
        g = match ctx.gemv_q8(xq_d as *const u8, w_d as *const u8, kt_d as *const u8, ty, n_in, n_out) {
            Ok(v) => v,
            Err(e) => { eprintln!("gemv: {e}"); return ExitCode::FAILURE; }
        };
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    // to_bits 전행 비교
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut mism = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for o in 0..n_out {
        let row = &w.data[o * rb..];
        let c = match w.ty {
            llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q8_0 => llm170_core::quant::dot_row_w4a8_q8_0_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &y),
            _ => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &y),
        };
        if c.to_bits() != g[o].to_bits() {
            mism += 1;
            if first.is_none() { first = Some((o, c, g[o])); }
        }
    }
    println!("[{}] {}: 원시 GEMV 불일치 {mism}/{n_out} — {:.0}µs/op {:.0}GB/s", w.ty.name(), args[1], dt * 1e6, w.data.len() as f64 / dt / 1e9);
    if let Some((o, c, gv)) = first {
        println!("  첫 불일치 [{o}]: cpu={c:.7e} gpu={gv:.7e}");
    }
    if mism > 0 { ExitCode::FAILURE } else { println!("  ★ 원시 HIP ≡ CPU 비트 일치"); ExitCode::SUCCESS }
}

/// llm170 check <model.gguf> [--quick] [--backend cpu|gpu]
/// debug 빌드 검증 경로 — ① 텐서 디양자화 스캔(NaN/Inf) ② GPU↔CPU GEMM
/// 상호검증 ③ 장문 청크 스모크(NaN 가드). RCA 도구 통합 (2026-09-01).
pub fn run_check(args: &[String]) -> ExitCode {
    use llm170_core::matmul::Accelerator;
    let mut path: Option<&str> = None;
    let mut quick = false;
    let mut backend = "gpu".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--backend" => backend = it.next().cloned().unwrap_or_else(|| "gpu".into()),
            p if !p.starts_with("--") => path = Some(p),
            _ => {}
        }
    }
    let Some(path) = path else {
        eprintln!("usage: llm170 check <model.gguf> [--quick] [--backend cpu|gpu]");
        return ExitCode::from(2);
    };
    let model_path = std::path::PathBuf::from(path);
    eprintln!("# check: {path} backend={backend} quick={quick}");

    // ① 텐서 스캔 — 각 텐서 첫 행 디양자화해 NaN/Inf 검출
    let mut n_scan = 0usize;
    let scan = std::thread::spawn({
        let p = model_path.clone();
        move || -> Result<(usize, usize), String> {
            let g = llm170_gguf::GgufFile::open(&p).map_err(|e| e.to_string())?;
            let file = std::fs::File::open(&p).map_err(|e| e.to_string())?;
            // SAFETY: 읽기 전용 매핑
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
            let mut bad = 0usize;
            let mut n = 0usize;
            for t in g.tensors.iter().take(if quick { 64 } else { usize::MAX }) {
                let (start, end) = match t.file_range(g.data_offset) {
                    Some(r) => r,
                    None => continue,
                };
                let data = &mmap[start as usize..end as usize];
                let n_in = t.ne[0] as usize;
                let mut row = vec![0.0f32; n_in.min(4096)];
                llm170_core::quant::dequant_row(t.ty, data, 0, row.len() as u64, &mut row);
                n += 1;
                if row.iter().any(|v| !v.is_finite()) {
                    eprintln!("# 텐서 비정상: {} ({})", t.name, t.ty.name());
                    bad += 1;
                }
            }
            Ok((n, bad))
        }
    });
    match scan.join() {
        Ok(Ok((n, bad))) => {
            n_scan = n;
            eprintln!("# ① 텐서 스캔: {n}개 중 비정상 {bad}");
            if bad > 0 {
                return ExitCode::FAILURE;
            }
        }
        Ok(Err(e)) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        Err(_) => return ExitCode::FAILURE,
    }
    let _ = n_scan;

    // ② GPU↔CPU GEMM 상호검증 (gpu 경로만) — 대표 텐서 t∈{1,64,1024}
    // (② GPU↔CPU GEMM 검증 — cubecl 제거로 rawhip-check가 대체)
    

    // ③ 장문 청크 스모크 — 1,024토큰 무작위 prefill (NaN 가드는 LLM170_Q4_TRACE)
    let arch = llm170_gguf::GgufFile::open(&model_path)
        .ok()
        .and_then(|g| g.arch().map(str::to_string));
    if arch.as_deref() == Some("qwen4exp") {
        let toks: Vec<String> = (0..1024).map(|i| (100 + (i * 7919) % 200000).to_string()).collect();
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap_or_default());
        cmd.args(["infer", "--model", path, "--prompt-tokens", &toks.join(","), "--n-predict", "2", "--ctx", "2048", "--backend", &backend])
            .env("LLM170_Q4_TRACE", "1")
            .env("LLM170_W_CAP_GB", "16")
            .stdout(std::process::Stdio::null());
        let st = cmd.status();
        match st {
            Ok(s) if s.success() => eprintln!("# ③ 청크 스모크(1024토큰): 통과"),
            Ok(s) => {
                eprintln!("# ③ 청크 스모크: 실패 ({s})");
                return ExitCode::FAILURE;
            }
            Err(e) => {
                eprintln!("# ③ 청크 스모크 실행 실패: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    eprintln!("# check 전체 통과");
    ExitCode::SUCCESS
}
