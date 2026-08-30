//! llm170 CLI.
//!
//! - gguf-dump: 모델 구조·양자화 믹스 덤프 (무게 미로딩)
//! - infer: qwen35 CPU 참조 추론 (greedy). 토큰 id 입력 — 토크나이저는 후속 단계.

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = r#"
llm170 — CMP 170HX 타깃 순수 Rust 추론 엔진 (개발 중)

사용법:
  llm170 gguf-dump [--meta-only] [--limit N] <file.gguf>
      GGUF 메타데이터·텐서 구성 덤프 (무게 미로딩)
  llm170 infer --model <file.gguf> --prompt-tokens <ids> [--prompt-tokens <ids> ...]
              [--n-predict N] [--ctx N] [--backend cpu|gpu] [--gpu-runtime hip|vulkan]
      greedy 추론. --prompt-tokens 반복 = 병렬 시퀀스(np), 콤마 구분 토큰 id.
      --backend gpu: matmul을 GPU(cubecl)로 오프로드. --gpu-runtime 기본 hip.
      출력: JSONL {"seq","pos","token","text"}
  llm170 help
"#;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gguf-dump") => cmd_gguf_dump(&args[1..]),
        Some("infer") => cmd_infer(&args[1..]),
        Some("gpu-probe-vk") => match llm170_backend_gpu::probe_vulkan() {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-probe") => match llm170_backend_gpu::probe() {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-smoke") => {
            let t = std::time::Instant::now();
            match llm170_backend_gpu::smoke_gemv(5120, 1024) {
                Ok(_) => {
                    println!("GPU GEMV 5120x1024 OK ({:.1?})", t.elapsed());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("gpu-mm") => cmd_gpu_mm(&args[1..]),
        Some("gpu-de") => cmd_gpu_de(&args[1..]),
        Some("gpu-de-bytes") => cmd_gpu_de_bytes(&args[1..]),
        Some("gpu-q3dbg") => cmd_gpu_q3dbg(&args[1..]),
        Some("dequant") => cmd_dequant(&args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn cmd_gguf_dump(args: &[String]) -> ExitCode {
    let mut meta_only = false;
    let mut limit = None;
    let mut path: Option<PathBuf> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--meta-only" => meta_only = true,
            "--limit" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => limit = Some(n),
                None => {
                    eprintln!("--limit requires a number");
                    return ExitCode::from(2);
                }
            },
            other if !other.starts_with("--") => {
                if path.is_some() {
                    eprintln!("multiple input files given");
                    return ExitCode::from(2);
                }
                path = Some(PathBuf::from(other));
            }
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(path) = path else {
        eprintln!("gguf-dump: input file required\n\n{USAGE}");
        return ExitCode::from(2);
    };

    llm170_profiler::reset();
    let f = {
        llm170_profiler::profile_span!("cli::gguf-dump::total");
        let f = match llm170_gguf::GgufFile::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        llm170_gguf::write_dump(&f, limit, meta_only, &mut std::io::stdout()).ok();
        f
    };
    drop(f);

    if let Some(rep) = llm170_profiler::report() {
        eprint!("\n{rep}");
    }
    ExitCode::SUCCESS
}

/// llm170 dequant <file> <tensor> <row> <n> — 디양자화 값 프로브 (검증용)
fn cmd_dequant(args: &[String]) -> ExitCode {
    if args.len() != 4 {
        eprintln!("usage: llm170 dequant <file> <tensor> <row> <n>");
        return ExitCode::from(2);
    }
    let f = match llm170_gguf::GgufFile::open(std::path::Path::new(&args[0])) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let t = match f.find_tensor(&args[1]) {
        Some(t) => t,
        None => {
            eprintln!("tensor not found: {}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let row: u64 = args[2].parse().unwrap();
    let n: usize = args[3].parse().unwrap();
    use std::os::unix::fs::FileExt;
    let file = std::fs::File::open(&args[0]).unwrap();
    let k = t.ne[0];
    let (blck, bsize) = t.ty.block_info();
    let row_bytes = (k / blck * bsize) as usize;
    let (start, _) = t.file_range(f.data_offset).unwrap();
    let mut buf = vec![0u8; row_bytes];
    file.read_exact_at(&mut buf, start + row * row_bytes as u64)
        .unwrap();
    let mut out = vec![0.0f32; k as usize];
    llm170_core::quant::dequant_row(t.ty, &buf, 0, k, &mut out);
    let vals: Vec<String> = out[..n].iter().map(|v| format!("{v:.6}")).collect();
    println!("[{}] row {row}: {}", t.ty.name(), vals.join(", "));
    ExitCode::SUCCESS
}

/// llm170 gpu-mm <file> <tensor> [t] [rows] — CPU matmul 대비 GPU GEMM 상호검증.
/// 결정적 LCG 입력(seed 0x1234_5678)으로 양쪽 누산, 최대 상대오차 보고.
/// llm170 gpu-de <file> <tensor> — 블록 0 디양자화 값 GPU 덤프 (검증용).
fn cmd_gpu_de(args: &[String]) -> ExitCode {
    if args.len() != 2 {
        eprintln!("usage: llm170 gpu-de <file> <tensor>");
        return ExitCode::from(2);
    }
    let model = match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => {
            eprintln!("tensor not found");
            return ExitCode::FAILURE;
        }
    };
    let gpu = match llm170_backend_gpu::GpuMatmul::new_hip() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let vals = gpu.debug_dequant_block(&w);
    match vals {
        Ok(v) => {
            let strs: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
            println!("{}", strs.join(","));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// llm170 gpu-de-bytes <file> <tensor> — 블록 0 원시 바이트 덤프.
fn cmd_gpu_de_bytes(args: &[String]) -> ExitCode {
    if args.len() != 2 {
        eprintln!("usage: llm170 gpu-de-bytes <file> <tensor>");
        return ExitCode::from(2);
    }
    let model = match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => {
            eprintln!("tensor not found");
            return ExitCode::FAILURE;
        }
    };
    let gpu = match llm170_backend_gpu::GpuMatmul::new_hip() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    match gpu.debug_block_bytes(&w) {
        Ok(v) => {
            let strs: Vec<String> = v.iter().map(|x| format!("{}", *x as u32)).collect();
            println!("{}", strs.join(","));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// llm170 gpu-q3dbg <file> <tensor> <mode 0-3> — q3_K 중간값 덤프.
fn cmd_gpu_q3dbg(args: &[String]) -> ExitCode {
    if args.len() != 3 {
        eprintln!("usage: llm170 gpu-q3dbg <file> <tensor> <mode>");
        return ExitCode::from(2);
    }
    let model = match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => {
            eprintln!("tensor not found");
            return ExitCode::FAILURE;
        }
    };
    let mode: usize = args[2].parse().unwrap();
    let gpu = match llm170_backend_gpu::GpuMatmul::new_hip() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    match gpu.debug_q3(&w, mode) {
        Ok(v) => {
            let strs: Vec<String> = v.iter().map(|x| format!("{}", *x as i64)).collect();
            println!("{}", strs.join(","));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
}

fn cmd_gpu_mm(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: llm170 gpu-mm <file> <tensor> [t] [rows]");
        return ExitCode::from(2);
    }
    let model = match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => {
            eprintln!("tensor not found: {}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let t: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let rows: usize = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(w.n_out as usize);
    let rows = rows.min(w.n_out as usize);
    let n_in = w.n_in as usize;

    // 결정적 입력
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let mut xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    // LLM170_UNIT=<i>: 단위 벡터 e_i → out[o] = W[o,i] 디양자화 값 프로브
    if let Ok(u) = std::env::var("LLM170_UNIT") {
        let u: usize = u.parse().unwrap();
        for (ti, row) in xs.iter_mut().enumerate() {
            for r in row.iter_mut() {
                *r = 0.0;
            }
            row[u + ti] = 1.0;
        }
    }

    // GPU
    let gpu: std::sync::Arc<dyn llm170_core::matmul::Accelerator> =
        match std::env::var("LLM170_GPU_RUNTIME").as_deref() {
            Ok("vulkan") => match llm170_backend_gpu::GpuMatmul::new_vulkan() {
                Ok(g) => std::sync::Arc::new(g),
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            },
            _ => match llm170_backend_gpu::GpuMatmul::new_hip() {
                Ok(g) => std::sync::Arc::new(g),
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            },
        };
    let mut gouts = vec![vec![0.0f32; w.n_out as usize]; t];
    if let Err(e) =
        llm170_core::matmul::Accelerator::matmul_batch(gpu.as_ref(), &xs, &w, &mut gouts)
    {
        eprintln!("gpu error: {e}");
        return ExitCode::FAILURE;
    }

    // 마이크로벤치: 동일 matmul N회 반복 (가중치 캐시 후) — op당 비용
    if let Some(n) = std::env::var("LLM170_GPU_BENCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ =
                llm170_core::matmul::Accelerator::matmul_batch(gpu.as_ref(), &xs, &w, &mut gouts);
        }
        let dt = t0.elapsed();
        eprintln!(
            "# bench: {n}회 × {dt:.2?} = {:.3}ms/op",
            dt.as_secs_f64() * 1000.0 / n as f64
        );
    }

    // CPU (동일 텐서 슬라이스 — 검증 대상 행만)
    let wsub = llm170_core::matmul::Weight {
        data: &w.data[..rows * (n_in / w.ty.blck_size() as usize) * w.ty.type_size() as usize],
        ty: w.ty,
        n_in: w.n_in,
        n_out: rows as u64,
    };
    let mut couts = vec![vec![0.0f32; rows]; t];
    llm170_core::matmul::matmul_batch(&xs, &wsub, &mut couts);

    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    for ti in 0..t {
        for o in 0..rows {
            let (g, c) = (gouts[ti][o], couts[ti][o]);
            let abs = (g - c).abs();
            let rel = abs / c.abs().max(1e-3);
            max_abs = max_abs.max(abs as f64);
            max_rel = max_rel.max(rel as f64);
        }
    }
    println!(
        "[{}] {} t={t} rows={rows}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
        w.ty.name(),
        args[1]
    );
    if max_rel > 1e-3 {
        eprintln!("MISMATCH");
        let mut shown = 0;
        for ti in 0..t {
            for o in 0..rows {
                let (g, c) = (gouts[ti][o], couts[ti][o]);
                if (g - c).abs() / c.abs().max(1e-3) > 1e-3 && shown < 8 {
                    eprintln!("  [{ti}][{o}] gpu={g:.6} cpu={c:.6}");
                    shown += 1;
                }
            }
        }
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn cmd_infer(args: &[String]) -> ExitCode {
    let mut model: Option<PathBuf> = None;
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let mut backend = "cpu".to_string();
    let mut gpu_runtime = "hip".to_string();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => match it.next() {
                Some(v) => model = Some(PathBuf::from(v)),
                None => return usage_err("--model requires a path"),
            },
            "--prompt-tokens" => match it.next() {
                Some(v) => match parse_ids(v) {
                    Ok(ids) if !ids.is_empty() => prompts.push(ids),
                    Ok(_) => return usage_err("empty prompt"),
                    Err(e) => return usage_err(&format!("bad tokens: {e}")),
                },
                None => return usage_err("--prompt-tokens requires ids"),
            },
            "--n-predict" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => n_predict = n,
                None => return usage_err("--n-predict requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => ctx = n,
                None => return usage_err("--ctx requires a number"),
            },
            "--backend" => match it.next() {
                Some(v) if v == "cpu" || v == "gpu" => backend = v.clone(),
                Some(v) => return usage_err(&format!("--backend: cpu|gpu (got {v})")),
                None => return usage_err("--backend requires cpu|gpu"),
            },
            "--gpu-runtime" => match it.next() {
                Some(v) if v == "hip" || v == "vulkan" => gpu_runtime = v.clone(),
                Some(v) => return usage_err(&format!("--gpu-runtime: hip|vulkan (got {v})")),
                None => return usage_err("--gpu-runtime requires hip|vulkan"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }

    let Some(model_path) = model else {
        return usage_err("--model required");
    };
    if prompts.is_empty() {
        return usage_err("at least one --prompt-tokens required");
    }
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    if max_prompt + n_predict + 8 >= ctx {
        return usage_err(&format!(
            "ctx({ctx}) too small for prompt({max_prompt})+n_predict({n_predict})"
        ));
    }

    llm170_profiler::reset();
    let t_start = std::time::Instant::now();
    // 아키텍처 판별 → qwen4exp 전용 엔진 분기
    let arch = llm170_gguf::GgufFile::open(&model_path)
        .ok()
        .and_then(|g| g.arch().map(|s| s.to_string()));
    if arch.as_deref() == Some("qwen4exp") {
        return run_q4_infer(&model_path, &prompts, n_predict, ctx, &backend);
    }
    let engine_res = llm170_core::model::Model::load(&model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::model::Engine::new(m, n, ctx);
            if backend == "gpu" {
                let acc: std::sync::Arc<dyn llm170_core::matmul::Accelerator> = match gpu_runtime
                    .as_str()
                {
                    "hip" => std::sync::Arc::new(
                        llm170_backend_gpu::GpuMatmul::new_hip().map_err(|e| e.to_string())?,
                    ),
                    "vulkan" => std::sync::Arc::new(
                        llm170_backend_gpu::GpuMatmul::new_vulkan().map_err(|e| e.to_string())?,
                    ),
                    other => {
                        return Err(format!("--gpu-runtime: hip|vulkan (got {other})"));
                    }
                };
                eng = eng.with_acc(acc);
                eprintln!("# backend: gpu ({gpu_runtime})");
            }
            let eos = 248044u32;
            // prefill (시퀀스별 — GDN chunked 경로)
            let mut last_logits = Vec::with_capacity(n);
            let dbg_topk = std::env::var("LLM170_DEBUG_TOPK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok());
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                if let Some(k) = dbg_topk {
                    let mut idx: Vec<usize> = (0..l.len()).collect();
                    idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                    let top: Vec<String> = idx[..k.min(l.len())]
                        .iter()
                        .map(|&i| format!("{}:{:.4}", i, l[i]))
                        .collect();
                    eprintln!("topk seq{s}: {}", top.join(" "));
                }
                last_logits.push(l);
            }
            let mut finished = vec![false; n];
            let mut gen_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut next: Vec<u32> = last_logits
                .iter()
                .map(|l| llm170_core::model::greedy(l))
                .collect();
            for s in 0..n {
                emit(s, prompts[s].len() as u32, next[s], &eng);
                gen_tokens[s].push(next[s]);
                if next[s] == eos {
                    finished[s] = true;
                }
            }
            // 배치 디코드 — 활성 시퀀스 묶어 1스텝 (np 상호검증 대상 경로)
            let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            for _step in 0..n_predict {
                let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                let toks: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                let seq_ids: Vec<usize> = active.clone();
                let logits = eng.decode(&seq_ids, &toks).map_err(|e| e.to_string())?;
                for (i, &s) in active.iter().enumerate() {
                    let t = llm170_core::model::greedy(&logits[i]);
                    next[s] = t;
                    pos[s] += 1;
                    emit(s, pos[s], t, &eng);
                    gen_tokens[s].push(t);
                    if t == eos {
                        finished[s] = true;
                    }
                }
            }
            let dt = t_start.elapsed();
            eprintln!(
                "# done: {} seqs, prompt max {}, gen per seq: {} (elapsed {dt:.1?})",
                n,
                max_prompt,
                gen_tokens.iter().map(|g| g.len()).min().unwrap_or(0)
            );
            Ok(())
        });
    if let Err(e) = engine_res {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    llm170_backend_gpu::timing_report();
    if let Some(rep) = llm170_profiler::report() {
        eprint!("\n{rep}");
    }
    ExitCode::SUCCESS
}

/// qwen4exp 추론 — Engine4 (시퀀스별 prefill/decode1).
fn run_q4_infer(
    model_path: &PathBuf,
    prompts: &[Vec<u32>],
    n_predict: usize,
    ctx: usize,
    backend: &str,
) -> ExitCode {
    let t_start = std::time::Instant::now();
    let res = llm170_core::qwen4exp::Model4::load(model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, n, ctx);
            if backend == "gpu" {
                let acc = llm170_backend_gpu::GpuMatmul::new_hip().map_err(|e| e.to_string())?;
                eng = eng.with_acc(std::sync::Arc::new(acc));
                eprintln!("# backend: gpu (cubecl) — qwen4exp");
            }
            let eos = eng.model.eos;
            let mut finished = vec![false; n];
            let mut next: Vec<u32> = Vec::with_capacity(n);
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                let t = llm170_core::model::greedy(&l);
                println!(
                    "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                    p.len(),
                    json_escape(&eng.piece(t))
                );
                next.push(t);
                finished[s] = t == eos;
            }
            let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            for _step in 0..n_predict {
                let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                for &s in &active {
                    let l = eng.decode1(s, next[s]).map_err(|e| e.to_string())?;
                    let t = llm170_core::model::greedy(&l);
                    next[s] = t;
                    pos[s] += 1;
                    println!(
                        "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                        pos[s],
                        json_escape(&eng.piece(t))
                    );
                    finished[s] = t == eos;
                }
            }
            eprintln!(
                "# done(q4): {n} seqs, gen per seq: {n_predict} (elapsed {:.1?})",
                t_start.elapsed()
            );
            Ok(())
        });
    if let Err(e) = res {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn emit(seq: usize, pos: u32, token: u32, eng: &llm170_core::model::Engine) {
    // 이 시점 eng는 &Engine 차입 — piece는 model 접근
    println!(
        "{{\"seq\":{},\"pos\":{},\"token\":{},\"text\":{}}}",
        seq,
        pos,
        token,
        json_escape(&eng.piece(token))
    );
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn parse_ids(s: &str) -> Result<Vec<u32>, std::num::ParseIntError> {
    s.split(',').map(|t| t.trim().parse::<u32>()).collect()
}

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}
