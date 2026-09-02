//! llm170 CLI.
//!
//! - gguf-dump: 모델 구조·양자화 믹스 덤프 (무게 미로딩)
//! - infer: qwen35 CPU 참조 추론 (greedy). 토큰 id 입력 — 토크나이저는 후속 단계.

mod bench;
mod engine;
mod http;
mod tokenize;

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
  llm170 gpu-ew-check
      ew 커널 전종 GPU↔CPU 상호검증 (norm 비트일치·활성화 abs<1e-5)
  llm170 gdn-ar-check [n_group dt_rank d]
      GDN AR 커널 GPU↔CPU 상호검증. 기본 8/48/128 (16·32·64·128 회귀 권장)
  llm170 moe-down-check
  llm170 bench --model <file.gguf> [--pp N] [--tg N] [--reps N] [--ctx N]
              [--backend cpu|gpu] [--gpu-runtime hip|vulkan] [--spec k]
      llama-bench 규격 PP/TG 측정 (t/s). --spec: MTP 스펙 디코드 유효 t/s.
  llm170 bench-streams <file.gguf> <tensor> [t] [max_n] [iters]
      동시 스트림 GEMV 집계 대역폭 (n=1..max_n 스윕) — P2-a 관문 측정.
  llm170 help
"#;

/// HIP 동기 런치 워크어라운드를 초기화 이전에 적용하기 위한 자기 재실행.
/// env는 HIP 런타임 init 시 1회 판독 — with_client에서 set_var해도
/// 무효였다(2026-09-01 실측: libamdhip64 할당·런치 경합 GPF 재발).
/// 성공 시 exec가 프로세스를 치환해 이 함수로 돌아오지 않는다.
#[cfg(target_os = "linux")]
fn reexec_hip_blocking() {
    if std::env::var_os("LLM170_NO_REEXEC").is_some()
        || std::env::var_os("HIP_LAUNCH_BLOCKING").is_some()
        || std::env::var_os("LLM170_HIP_ASYNC").is_some()
    {
        return;
    }
    use std::os::unix::process::CommandExt;
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if argv.is_empty() {
        return;
    }
    let err = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .env("HIP_LAUNCH_BLOCKING", "1")
        .env("LLM170_NO_REEXEC", "1")
        .exec();
    eprintln!("# re-exec 실패({err}) — HIP_LAUNCH_BLOCKING 미적용으로 계속");
}

fn main() -> ExitCode {
    // cubecl 커널 컴파일 오류 등 log 패싯 메시지 노출 — stderr 간이 로거.
    struct EL;
    impl log::Log for EL {
        fn enabled(&self, _: &log::Metadata) -> bool { true }
        fn log(&self, r: &log::Record) { eprintln!("[{}] {}", r.level(), r.args()); }
        fn flush(&self) {}
    }
    let _ = log::set_logger(&EL);
    log::set_max_level(log::LevelFilter::Error);
    reexec_hip_blocking();
    // OOM 킬러 지정 희생자 (실측 2026-09-01): 초대형 mmap(total-vm 150GB+)이
    // badness 최상위로 뽑혀 런·세션이 함께 죽는다. 스스로 adj=1000을 걸어
    // 런만 희생되게 한다 (무권한으로는 보호 불가 — 우선순위 이동만 가능).
    // LLM170_NO_OOM_ADJ=1이면 해제.
    if std::env::var_os("LLM170_NO_OOM_ADJ").is_none() {
        let _ = std::fs::write("/proc/self/oom_score_adj", b"1000");
    }
    // 프레임(활성 상주 디코드) 기본 ON — 게이트는 qwen4exp layers에 있어
    // qwen35·CPU는 무영향. 상주 불가(작은 GTT 등)면 decode1이 value 경로로
    // 자동 폴백. LLM170_FRAME=0으로 명시적 해제.
    if std::env::var_os("LLM170_FRAME").is_none() {
        // SAFETY: main 스레드 초기화 경로 — 다른 스레드 시작 전
        unsafe { std::env::set_var("LLM170_FRAME", "1") };
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gguf-dump") => cmd_gguf_dump(&args[1..]),
        Some("infer") => cmd_infer(&args[1..]),
        Some("serve") => return cmd_serve(&args[1..]),
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
        Some("bench-streams") => cmd_bench_streams(&args[1..]),
        Some("w4a8i-check") => cmd_w4a8i_check(&args[1..]),
        Some("gpu-ew-check") => cmd_gpu_ew_check(),
        Some("gdn-ar-check") => cmd_gdn_ar_check(),
        Some("gdn-chunk-check") => cmd_gdn_chunk_check(),
        Some("bench") => return bench::cmd_bench(&args[1..]),
        Some("moe-down-check") => cmd_moe_down_check(),
        Some("check") => cmd_check(&args[1..]),
        Some("w4a8-check") => cmd_w4a8_check(&args[1..]),
        Some("w4a8-gpu") => cmd_w4a8_gpu(&args[1..]),
        Some("gpu-de") => cmd_gpu_de(&args[1..]),
        Some("mem-profile") => cmd_mem_profile(),
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

/// --mode 파싱·적용 — env 기본값으로 반영 (기존 env 관례의 단일 소스 유지).
/// LLM170_W_CAP_GB·LLM170_Q4_CHUNK가 이미 있으면 사용자 명시로 존중.
fn apply_mode(m: llm170_core::mode::Mode) {
    // 프레임 기본 ON(2026-09-02): 전문가 스택 상주(~88GiB)가 성립 조건이라
    // 모드 프리셋 W_CAP(72GiB)를 세우면 프레임이 원천 불능이 된다. 프레임이
    // 켜져 있으면 프리셋을 생략해 WeightStore가 실측 총량의 95%로 유도하게
    // 한다(작은 기기는 상주 실패 → value 폴백). 사용자 명시는 존중.
    let frame_on = std::env::var("LLM170_FRAME").is_ok_and(|v| v != "0");
    if std::env::var_os("LLM170_W_CAP_GB").is_none() && !frame_on {
        // SAFETY: main 스레드 초기화 경로 — 다른 스레드 시작 전
        unsafe { std::env::set_var("LLM170_W_CAP_GB", m.w_cap_gb().to_string()) };
    }
    if std::env::var_os("LLM170_Q4_CHUNK").is_none() {
        // SAFETY: 위와 동일
        unsafe { std::env::set_var("LLM170_Q4_CHUNK", m.prefill_chunk().to_string()) };
    }
    eprintln!("# mode: {m:?} (w_cap={}GiB chunk={})", m.w_cap_gb(), m.prefill_chunk());
}

/// llm170 serve --model <file> [--port N] [--ctx N] [--backend cpu|gpu] [--mode M]
fn cmd_serve(args: &[String]) -> ExitCode {
    let mut model: Option<PathBuf> = None;
    let mut port = 8080u16;
    let mut ctx = 4096usize;
    let mut backend = "cpu".to_string();
    let mut gpu_runtime = String::new();
    let mut mode: Option<llm170_core::mode::Mode> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => match it.next() {
                Some(v) => model = Some(PathBuf::from(v)),
                None => return usage_err("--model requires a path"),
            },
            "--port" => match it.next().and_then(|v| v.parse().ok()) {
                Some(p) => port = p,
                None => return usage_err("--port requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse().ok()) {
                Some(c) => ctx = c,
                None => return usage_err("--ctx requires a number"),
            },
            "--backend" => match it.next() {
                Some(v) if v == "cpu" || v == "gpu" => backend = v.clone(),
                Some(v) => return usage_err(&format!("--backend: cpu|gpu (got {v})")),
                None => return usage_err("--backend requires cpu|gpu"),
            },
            "--mode" => match it.next().map(String::as_str).and_then(llm170_core::mode::Mode::from_str) {
                Some(m) => mode = Some(m),
                None => return usage_err("--mode requires universal|cmp-stock|cmp-unlocked"),
            },
            "--gpu-runtime" => match it.next().map(String::as_str) {
                Some(v) if v == "hip" || v == "vulkan" => gpu_runtime = v.to_string(),
                Some(v) => return usage_err(&format!("--gpu-runtime: hip|vulkan (got {v})")),
                None => return usage_err("--gpu-runtime requires hip|vulkan"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = model else { return usage_err("--model required") };
    if let Some(m) = mode {
        apply_mode(m);
    }
    // 토크나이저 적재 (part1 메타 → 실패시 part2)
    let part2 = {
        let stem = model_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if stem.contains("-00001-of-") {
            Some(model_path.with_file_name(stem.replace("-00001-of-", "-00002-of-")))
        } else {
            None
        }
    };
    // 간헐 ENOPT(transient ENOENT) 재시도 — 2026-09-01 실측 회복 패턴.
    let mut tok = None;
    for i in 0..5 {
        match tokenize::Tokenizer::load(&model_path, part2.as_deref()) {
            Ok(t) => {
                tok = Some(t);
                break;
            }
            Err(e) => {
                eprintln!("# tokenizer load 재시도 {}/5: {e}", i + 1);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    if tok.is_none() {
        eprintln!("# tokenizer load 실패 (토큰 id 모드만 동작)");
    }
    let _ = engine::TOKENIZER.set(tok.unwrap_or_else(|| tokenize::Tokenizer::empty()));
    let req = engine::InferRequest { model: model_path, ctx };
    let sel = if backend == "gpu" {
        if gpu_runtime.is_empty() {
            engine::BackendSel::Gpu
        } else {
            engine::BackendSel::GpuRuntime(gpu_runtime)
        }
    } else {
        engine::BackendSel::Cpu
    };
    match http::serve(&format!("127.0.0.1:{port}"), req, sel) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
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

/// llm170 w4a8-check <file> <tensor> [t] [rows] — W4A8 변형 ↔ f32 기준 상호검증.
fn cmd_w4a8_check(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: llm170 w4a8-check <file> <tensor> [t] [rows]");
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
    let rows: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(256);
    let rows = rows.min(w.n_out as usize);
    let n_in = w.n_in as usize;
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let wsub = llm170_core::matmul::Weight {
        data: &w.data[..rows * (n_in / w.ty.blck_size() as usize) * w.ty.type_size() as usize],
        ty: w.ty,
        n_in: w.n_in,
        n_out: rows as u64,
    };
    let mut couts = vec![vec![0.0f32; rows]; t];
    llm170_core::matmul::matmul_batch(&xs, &wsub, &mut couts);
    let mut wouts = vec![vec![0.0f32; rows]; t];
    for (xi, wo) in xs.iter().zip(wouts.iter_mut()) {
        llm170_core::matmul::matmul_w4a8(xi, &wsub, wo);
    }
    let (mut max_abs, mut max_mag) = (0.0f64, 0.0f64);
    for ti in 0..t {
        for o in 0..rows {
            let (g, c) = (wouts[ti][o], couts[ti][o]);
            max_abs = max_abs.max((g - c).abs() as f64);
            max_mag = max_mag.max(c.abs() as f64);
        }
    }
    let rel = max_abs / max_mag;
    println!(
        "[{}] {} t={t} rows={rows}: max_abs={max_abs:.3e} rel(vs max|y|)={rel:.3e}",
        w.ty.name(),
        args[1]
    );
    if rel > 2e-2 {
        eprintln!("MISMATCH (rel > 2e-2)");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// llm170 w4a8-gpu <file> <tensor> [rows] — GPU W4A8 커널 ↔ f32 CPU 기준 상호검증.
fn cmd_w4a8_gpu(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: llm170 w4a8-gpu <file> <tensor> [rows]");
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
    let rows: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    let rows = rows.min(w.n_out as usize);
    let n_in = w.n_in as usize;
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let gpu = match llm170_backend_gpu::GpuMatmul::new_hip() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let g = match gpu.matmul_w4a8_gpu(&x, &w) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gpu error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let wsub = llm170_core::matmul::Weight {
        data: &w.data[..rows * (n_in / w.ty.blck_size() as usize) * w.ty.type_size() as usize],
        ty: w.ty,
        n_in: w.n_in,
        n_out: rows as u64,
    };
    let mut c = vec![0.0f32; rows];
    llm170_core::matmul::matmul(&x, &wsub, &mut c);
    let (mut max_abs, mut max_mag) = (0.0f64, 0.0f64);
    for o in 0..rows {
        max_abs = max_abs.max((g[o] - c[o]).abs() as f64);
        max_mag = max_mag.max(c[o].abs() as f64);
    }
    let rel = max_abs / max_mag;
    println!(
        "[{}] {} rows={rows}: max_abs={max_abs:.3e} rel(vs max|y|)={rel:.3e}",
        w.ty.name(),
        args[1]
    );
    if rel > 2e-2 {
        eprintln!("MISMATCH (rel > 2e-2)");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// llm170 w4a8i-check <file> <tensor> [iters] — W4A8 정수 커널(gemm_q8i)
/// 비트 일치 검증(전 행, CPU 레인 미러 대비 to_bits) + 순수 속도 측정.
fn cmd_w4a8i_check(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: llm170 w4a8i-check <file> <tensor> [iters]");
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
    let iters: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50);
    let rows: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let blck0 = w.ty.blck_size() as usize;
    let bsize0 = w.ty.type_size() as usize;
    let w = if rows > 0 && rows < w.n_out as usize {
        println!("# rows 제한 {}/{}", rows, w.n_out);
        llm170_core::matmul::Weight {
            data: &w.data[..rows * (w.n_in as usize / blck0) * bsize0],
            ty: w.ty,
            n_in: w.n_in,
            n_out: rows as u64,
        }
    } else {
        w
    };
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let gpu = match llm170_backend_gpu::GpuMatmul::new_hip() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let _ = gpu.matmul_w4a8_int_gpu(&x, &w, 1); // 웜업 (hipRTC JIT 제거)
    let (g, dt) = match gpu.matmul_w4a8_int_gpu(&x, &w, iters) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gpu error: {e}");
            return ExitCode::FAILURE;
        }
    };
    // CPU 레인 미러 — 전 행 비트 비교 + 0행 레인 부분합 대조
    let y = llm170_core::quant::quantize_row_q8_ref(&x);
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let is_q3k = w.ty == llm170_gguf::GgmlType::Q3K;
    let row0d = &w.data[..(n_in / blck) * bsize];
    let lane0 = if is_q3k {
        llm170_core::quant::dot_row_w4a8_q3k_lane_parts(row0d, n_in as u64, &y)
            .to_vec()
    } else if w.ty == llm170_gguf::GgmlType::Q5K {
        llm170_core::quant::dot_row_w4a8_q5k_lane_parts(row0d, n_in as u64, &y)
            .to_vec()
    } else if w.ty == llm170_gguf::GgmlType::Q4K {
        llm170_core::quant::dot_row_w4a8_q4k_lane_parts(row0d, n_in as u64, &y).to_vec()
    } else if w.ty == llm170_gguf::GgmlType::Q8_0 {
        llm170_core::quant::dot_row_w4a8_q8_0_lane_parts(row0d, n_in as u64, &y).to_vec()
    } else if w.ty == llm170_gguf::GgmlType::Iq4Nl {
        llm170_core::quant::dot_row_w4a8_iq4nl_lane_parts(row0d, n_in as u64, &y).to_vec()
    } else if w.ty == llm170_gguf::GgmlType::Q6K {
        llm170_core::quant::dot_row_w4a8_q6k_lane_parts(row0d, n_in as u64, &y).to_vec()
    } else {
        llm170_core::quant::dot_row_w4a8_iq4xs_lane_parts(row0d, n_in as u64, &y).to_vec()
    };
    if lane0.len() >= 4 {
        println!("  cpu lane0[0..4] = {:?}", &lane0[..4]);
    }
    {
        let row0 = &w.data[..(n_in / blck) * bsize];
        let old_dot = llm170_core::quant::dot_row_w4a8(w.ty, row0, n_in as u64, &y);
        println!("  y[0].d={:e} qs[0..8]={:?}", y[0].d, &y[0].qs[..8]);
        // f32 디양자화 기준 (진짜 정확성 게이트) — q8 양자화 오차 ~1e-2
        let mut f32row = vec![0.0f32; n_in];
        llm170_core::quant::dequant_row(w.ty, row0, 0, w.n_in, &mut f32row);
        let mut fsum = 0.0f32;
        for (i, wv) in f32row.iter().enumerate() {
            fsum += wv * x[i];
        }
        let mirror_row = {
            let mut s = 0.0f64;
            for l in 0..64 {
                s += lane0[l];
            }
            s as f32
        };
        let rel = ((mirror_row - fsum) / fsum.abs().max(1e-3)).abs();
        println!(
            "  row0: f32={fsum:e} 미러={mirror_row:e} rel={rel:.3e} (구 dot={old_dot:e})"
        );
        if rel > 2e-2 {
            println!("  ✗ 미러·f32 상대오차 과대");
        }
    }
    // GPU 양자화 커널 비트 검증 (rust quantize_row_q8_ref 대비)
    let (gq, gd) = gpu.quant_q8_gpu(&x).expect("quant_q8_gpu");
    let mut qm = 0usize;
    {
        let mut cpu_words = Vec::with_capacity(n_in / 4);
        for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
            let mut word = 0u32;
            for (i, b) in c.iter().enumerate() {
                word |= (**b as u8 as u32) << (8 * i);
            }
            cpu_words.push(word);
        }
        for (i, (a, b)) in gq.iter().zip(cpu_words.iter()).enumerate() {
            if a != b {
                qm += 1;
                if qm == 1 {
                    println!("  ✗ quant 워드 불일치 [{i}]: gpu={a:#x} cpu={b:#x}");
                }
            }
        }
        for (i, (a, b)) in gd.iter().zip(y.iter().map(|b| b.d)).enumerate() {
            if a.to_bits() != b.to_bits() {
                qm += 1;
                if qm == 1 {
                    println!("  ✗ quant d 불일치 [{i}]: gpu={a:e} cpu={b:e}");
                }
            }
        }
        if qm == 0 {
            println!("  ★ quant_q8 GPU≡CPU 비트 일치 (워드 {} + d {})", n_in / 4, n_in / 32);
        }
    }
    let mut first: Option<(usize, f32, f32)> = None;
    let mut mismatch = 0usize;
    for o in 0..n_out {
        let row = &w.data[o * (n_in / blck) * bsize..];
        let c = if is_q3k {
            llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &y)
        } else if w.ty == llm170_gguf::GgmlType::Q5K {
            llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &y)
        } else if w.ty == llm170_gguf::GgmlType::Q4K {
            llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &y)
        } else if w.ty == llm170_gguf::GgmlType::Q8_0 {
            llm170_core::quant::dot_row_w4a8_q8_0_lane(row, n_in as u64, &y)
        } else if w.ty == llm170_gguf::GgmlType::Iq4Nl {
            llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &y)
        } else if w.ty == llm170_gguf::GgmlType::Q6K {
            llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &y)
        } else {
            llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &y)
        };
        if c.to_bits() != g[o].to_bits() {
            mismatch += 1;
            if first.is_none() {
                println!("  첫 불일치 [{o}]: cpu={c:.7e} gpu={:.7e}", g[o]);
                first = Some((o, c, g[o]));
            }
        }
    }
    let ms = dt.as_secs_f64() * 1000.0 / iters as f64;
    let gbps = w.data.len() as f64 / (dt.as_secs_f64() / iters as f64) / 1e9;
    println!(
        "[{}] {} n={n_in}x{n_out}: 비트 불일치 {mismatch}/{n_out} — {ms:.3}ms/op, {gbps:.0}GB/s (iters={iters})",
        w.ty.name(),
        args[1]
    );
    if let Some((o, c, gv)) = first {
        println!("  첫 불일치 [{o}]: cpu={c:.7e} gpu={gv:.7e}");
    }
    if mismatch > 0 {
        ExitCode::FAILURE
    } else {
        println!("  ★ GPU≡CPU 비트 일치 (레인 f64 미러)");
        ExitCode::SUCCESS
    }
}

/// llm170 check <model.gguf> [--quick] [--backend cpu|gpu]
/// debug 빌드 검증 경로 — ① 텐서 디양자화 스캔(NaN/Inf) ② GPU↔CPU GEMM
/// 상호검증 ③ 장문 청크 스모크(NaN 가드). RCA 도구 통합 (2026-09-01).
fn cmd_check(args: &[String]) -> ExitCode {
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
    if backend == "gpu" {
        let ok = (|| -> Result<(), String> {
            let g = llm170_gguf::GgufFile::open(&model_path).map_err(|e| e.to_string())?;
            let file = std::fs::File::open(&model_path).map_err(|e| e.to_string())?;
            // SAFETY: 읽기 전용 매핑
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
            let gpu: std::sync::Arc<dyn Accelerator> = match std::env::var("LLM170_GPU_RUNTIME").as_deref() {
                Ok("vulkan") => std::sync::Arc::new(
                    llm170_backend_gpu::GpuMatmul::new_vulkan().map_err(|e| e)?,
                ),
                _ => std::sync::Arc::new(llm170_backend_gpu::GpuMatmul::new_hip().map_err(|e| e)?),
            };
            // 대표 텐서: ffn_down류 (행렬形状)
            let cand = g
                .tensors
                .iter()
                .find(|t| t.name.contains("ffn_down") || t.name.contains("attn_qkv") || t.name.contains("attn_q."));
            let Some(t) = cand else { return Ok(()) };
            let (start, end) = t.file_range(g.data_offset).ok_or("range")?;
            let w = llm170_core::matmul::Weight {
                data: &mmap[start as usize..end as usize],
                ty: t.ty,
                n_in: t.ne[0],
                n_out: (t.ne[1] * t.ne[2] * t.ne[3]).min(256),
            };
            let n_in = w.n_in as usize;
            let mut seed = 0x1234_5678u64;
            let mut lcg = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
            };
            for t_len in [1usize, 64, if quick { 256 } else { 1024 }] {
                let xs: Vec<Vec<f32>> = (0..t_len).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
                let mut gouts = vec![vec![0.0f32; w.n_out as usize]; t_len];
                gpu.matmul_batch(&xs, &w, &mut gouts).map_err(|e| e)?;
                let mut couts = vec![vec![0.0f32; w.n_out as usize]; t_len];
                llm170_core::matmul::matmul_batch(&xs, &w, &mut couts);
                let mut max_rel = 0.0f64;
                for (g, c) in gouts.iter().zip(couts.iter()) {
                    for (a, b) in g.iter().zip(c.iter()) {
                        let rel = (a - b).abs() as f64 / b.abs().max(1e-2) as f64;
                        max_rel = max_rel.max(rel);
                    }
                }
                let nonfinite = gouts.iter().flatten().filter(|v| !v.is_finite()).count();
                eprintln!("# ② GEMM t={t_len}: max_rel={max_rel:.2e} 비finite={nonfinite} ({})", t.name);
                if nonfinite > 0 || max_rel > 5e-2 {
                    return Err("GEMM 상호검증 실패".into());
                }
            }
            Ok(())
        })();
        if let Err(e) = ok {
            eprintln!("# ② FAIL: {e}");
            return ExitCode::FAILURE;
        }
    }

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

/// gpu-ew-check — ew 커널 전종 GPU↔CPU 상호검증 (층 GPU 상주 P2-4 1단계).
/// 판정: norm류 max_rel < 1e-6 (f64 경로 — 비트일치 기대), 활성화류 < 1e-5
/// (libm 구현차), moe ids 완전일치.

/// mem-profile — device memory measurement report. Prints what the adaptive
/// weight budget derives from (per backend placement region), so machine
/// tuning stays data-driven instead of hardcoded. Later this grows into a
/// calibration tool that emits a profile file for injection.
fn cmd_mem_profile() -> ExitCode {
    let read = |kind: &str, field: &str| -> Option<usize> {
        let dir = std::fs::read_dir("/sys/class/drm").ok()?;
        for entry in dir.flatten() {
            let p = entry
                .path()
                .join("device")
                .join(format!("mem_info_{kind}_{field}"));
            if let Ok(s) = std::fs::read_to_string(&p) {
                if let Ok(v) = s.trim().parse::<usize>() {
                    if v > 0 {
                        return Some(v);
                    }
                }
            }
        }
        None
    };
    let gib = |v: usize| v as f64 / (1u64 << 30) as f64;
    for (kind, backend) in [("vram", "hip"), ("gtt", "vulkan")] {
        let total = read(kind, "total");
        let used = read(kind, "used");
        match (total, used) {
            (Some(t), Some(u)) => {
                let budget = if std::env::var("LLM170_W_CAP_GB").is_ok() {
                    "env override".to_string()
                } else {
                    format!("{:.0} GiB (40% adaptive)", gib(t / 5 * 2))
                };
                println!(
                    "[{backend}] {kind}: total {:.0} GiB, used {:.0} GiB, free {:.0} GiB — weight budget: {budget}",
                    gib(t),
                    gib(u),
                    gib(t.saturating_sub(u)),
                );
            }
            _ => println!("[{backend}] {kind}: not available (non-amdgpu?)"),
        }
    }
    ExitCode::SUCCESS
}
fn cmd_gpu_ew_check() -> ExitCode {
    let run = |report: Result<Vec<(&'static str, f64, f64, f64)>, String>| -> (bool, String) {
        let rels = match report {
            Ok(r) => r,
            Err(e) => return (false, format!("error: {e}")),
        };
        let mut ok = true;
        let mut out = String::new();
        for (name, mr, ma, beq) in &rels {
            let is_norm = name.contains("rms") || name.contains("l2");
            // norm류: 비트일치(f64 경로). 활성화류: libm σ 편차 — abs < 1e-5
            // (소폭 출력의 rel 증폭은 abs로 판별) 또는 rel < 1e-5.
            let pass = if is_norm {
                *mr < 1e-6
            } else {
                *ma < 1e-5 || *mr < 1e-5
            };
            ok &= pass;
            out.push_str(&format!(
                "[ew] {name}: max_rel={mr:.3e} max_abs={ma:.3e} bit_eq={beq:.4} {}\n",
                if pass { "ok" } else { "MISMATCH" }
            ));
        }
        (ok, out)
    };
    let (ok, out) = match std::env::var("LLM170_GPU_RUNTIME").as_deref() {
        Ok("vulkan") => run(llm170_backend_gpu::GpuMatmul::new_vulkan().and_then(|g| g.ew_check())),
        _ => run(llm170_backend_gpu::GpuMatmul::new_hip().and_then(|g| g.ew_check())),
    };
    print!("{out}");
    if ok {
        println!("gpu-ew-check PASS");
        ExitCode::SUCCESS
    } else {
        eprintln!("gpu-ew-check MISMATCH");
        ExitCode::FAILURE
    }
}

/// gdn-ar-check — GDN AR 커널 GPU↔CPU 상호검증 (합성 텐서, 수제 LCG).
fn cmd_gdn_ar_check() -> ExitCode {
    use llm170_core::matmul::Accelerator;
    // 인수: [n_group dt_rank d] — 소형 차원 회귀 검증 (2026-09-01 실측)
    let a: Vec<usize> = std::env::args()
        .skip(2)
        .filter_map(|v| v.parse().ok())
        .collect();
    let (n_group, dt_rank, d) = if a.len() == 3 {
        (a[0], a[1], a[2])
    } else {
        (8usize, 48usize, 128usize)
    };
    let k_len = n_group * d;
    let v_len = dt_rank * d;
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let q: Vec<f32> = (0..k_len).map(|_| lcg()).collect();
    let k: Vec<f32> = (0..k_len).map(|_| lcg()).collect();
    let v: Vec<f32> = (0..v_len).map(|_| lcg()).collect();
    let beta: Vec<f32> = (0..dt_rank).map(|_| lcg() + 0.5).collect();
    let g: Vec<f32> = (0..dt_rank).map(|_| lcg() * 4.0).collect();
    let scale = 1.0f32 / (d as f32).sqrt();

    // CPU 기준
    let mut st_c: Vec<f32> = (0..dt_rank * d * d).map(|_| lcg()).collect();
    let st_orig = st_c.clone();
    let mut out_c = vec![0.0f32; v_len];
    llm170_core::gdn::gdn_ar_batch(&q, &k, &v, &beta, &g, &mut st_c, &mut out_c, 1, n_group, dt_rank);
    if std::env::var_os("LLM170_GDN_CPU").is_some() {
        println!("[skip] LLM170_GDN_CPU");
        return ExitCode::SUCCESS;
    }
    let gpu: std::sync::Arc<dyn Accelerator> = match std::env::var("LLM170_GPU_RUNTIME").as_deref() {
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
    let mut st_g = st_orig.clone();
    let mut out_g = vec![0.0f32; v_len];
    let qs: Vec<f32> = q.iter().map(|x| x * scale).collect();
    let mut beta_ge = vec![0.0f32; dt_rank * 2];
    for h in 0..dt_rank {
        beta_ge[h * 2] = beta[h];
        beta_ge[h * 2 + 1] = g[h].exp();
    }
    if let Err(e) = gpu.gdn_ar(&qs, &k, &v, &beta_ge, &mut st_g, &mut out_g, 1, n_group, dt_rank, d) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
            let dd = (x - y).abs() as f64;
            m.max(dd / y.abs().max(1e-3) as f64)
        })
    };
    let (ro, rs) = (rel(&out_g, &out_c), rel(&st_g, &st_c));
    println!(
        "[gdn_ar] n_group={n_group} dt_rank={dt_rank} d={d}: out max_rel={ro:.3e} state max_rel={rs:.3e}"
    );
    if ro < 2e-3 && rs < 2e-3 {
        println!("gdn-ar PASS");
        ExitCode::SUCCESS
    } else {
        eprintln!("gdn-ar MISMATCH");
        ExitCode::FAILURE
    }
}

/// gdn-chunk-check — GDN 청크(t>1) 커널 GPU↔CPU 상호검증 (합성, 03 §3.1).
fn cmd_gdn_chunk_check() -> ExitCode {
    use llm170_core::matmul::Accelerator;
    // 인수: [t n_group dt_rank d]
    let a: Vec<usize> = std::env::args()
        .skip(2)
        .filter_map(|v| v.parse().ok())
        .collect();
    let (t_len, n_group, dt_rank, d) = if a.len() == 4 {
        (a[0], a[1], a[2], a[3])
    } else {
        (200usize, 8usize, 48usize, 128usize)
    };
    let k_len = n_group * d;
    let v_len = dt_rank * d;
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let q: Vec<f32> = (0..t_len * k_len).map(|_| lcg()).collect();
    let k: Vec<f32> = (0..t_len * k_len).map(|_| lcg()).collect();
    let v: Vec<f32> = (0..t_len * v_len).map(|_| lcg()).collect();
    let beta: Vec<f32> = (0..t_len * dt_rank).map(|_| lcg() + 0.5).collect();
    // g는 실모델처럼 음수(decay) — 양수 g는 gcs 양향 누적으로 (I+A)⁻¹
    // 병렬조건 악화 → ulp급 exp 편차 증폭 (AR 관례의 양수 대칭은 비현실).
    let g: Vec<f32> = (0..t_len * dt_rank).map(|_| lcg() * 4.0 - 4.5).collect();

    let mut st_c: Vec<f32> = (0..dt_rank * d * d).map(|_| lcg()).collect();
    let st_orig = st_c.clone();
    let mut out_c = vec![0.0f32; t_len * v_len];
    llm170_core::gdn::gdn_chunk_seq(&q, &k, &v, &beta, &g, &mut st_c, &mut out_c, t_len, n_group, dt_rank);

    if std::env::var_os("LLM170_GDN_CPU").is_some() {
        println!("[skip] LLM170_GDN_CPU");
        return ExitCode::SUCCESS;
    }
    let gpu: std::sync::Arc<dyn Accelerator> = match std::env::var("LLM170_GPU_RUNTIME").as_deref() {
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
    let mut st_g = st_orig.clone();
    let mut out_g = vec![0.0f32; t_len * v_len];
    if let Err(e) = gpu.gdn_chunk(&q, &k, &v, &beta, &g, &mut st_g, &mut out_g, t_len, n_group, dt_rank, d) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let rel = |x: &[f32], y: &[f32]| -> f64 {
        x.iter().zip(y).fold(0.0f64, |m, (a, b)| {
            let dd = (a - b).abs() as f64;
            m.max(dd / b.abs().max(1e-3) as f64)
        })
    };
    let (ro, rs) = (rel(&out_g, &out_c), rel(&st_g, &st_c));
    println!(
        "[gdn_chunk] t={t_len} n_group={n_group} dt_rank={dt_rank} d={d}: out max_rel={ro:.3e} state max_rel={rs:.3e}"
    );
    // 청크 커널은 CPU와 동일 순서 설계 — 엄밀 일치 기대 (f32 라이브러리 exp 편차 허용치)
    // 실측: 합성 음수-decay g에서 out ≤ 8.7e-6 (exp 라이브러리 ulp 편차).
    // e2e 게이트가 토큰 일치로 최종 보증.
    if ro < 2e-5 && rs < 2e-5 {
        println!("gdn-chunk PASS");
        ExitCode::SUCCESS
    } else {
        eprintln!("gdn-chunk MISMATCH");
        ExitCode::FAILURE
    }
}

/// moe-down-check — 배치 down 커널 GPU↔CPU 상호검증 (합성).
/// 합성 q8_0 전문가 스택 [K=13][rows=32][n_in=256] + 무작위 x, 짝(개별) 대비.
fn cmd_moe_down_check() -> ExitCode {
    use llm170_core::matmul::{Accelerator, Weight};
    let (k_stack, n_out, n_in) = (13usize, 32usize, 256usize);
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    // q8_0 스택 합성: 블록당 f16 d + 32 i8
    let n_blocks = n_in / 32;
    let per_expert_bytes = n_out * n_blocks * 34;
    let mut data = vec![0u8; k_stack * per_expert_bytes];
    for b in data.chunks_exact_mut(34) {
        let dh = half::f16::from_f32(lcg() * 0.01 + 0.005);
        b[..2].copy_from_slice(&dh.to_le_bytes());
        for q in &mut b[2..] {
            *q = (lcg() * 120.0) as i8 as u8;
        }
    }
    // 선택 전문가 (비순차) + 행 x
    let ids: [u32; 5] = [7, 0, 12, 3, 9];
    let ks = ids.len();
    let xs: Vec<Vec<f32>> = (0..ks).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();

    // CPU 기준 — 전문가별 뷰로 matmul
    let mut couts = vec![vec![0.0f32; n_out]; ks];
    for (ri, &e) in ids.iter().enumerate() {
        let off = e as usize * per_expert_bytes;
        let w = Weight {
            data: &data[off..off + per_expert_bytes],
            ty: llm170_gguf::GgmlType::Q8_0,
            n_in: n_in as u64,
            n_out: n_out as u64,
        };
        llm170_core::matmul::matmul(&xs[ri], &w, &mut couts[ri]);
    }

    // GPU — 스택 전체 뷰
    let gpu: std::sync::Arc<dyn Accelerator> = match std::env::var("LLM170_GPU_RUNTIME").as_deref() {
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
    let stack = Weight {
        data: &data,
        ty: llm170_gguf::GgmlType::Q8_0,
        n_in: n_in as u64,
        n_out: (n_out * k_stack) as u64,
    };
    let mut gouts = vec![vec![0.0f32; n_out]; ks];
    if let Err(e) = gpu.moe_down(&xs, &stack, &ids, k_stack, &mut gouts) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let mut max_rel = 0.0f64;
    for (g, c) in gouts.iter().zip(couts.iter()) {
        for (a, b) in g.iter().zip(c.iter()) {
            let rel = (a - b).abs() as f64 / b.abs().max(1e-2) as f64;
            max_rel = max_rel.max(rel);
        }
    }
    println!(
        "[moe_down] stack={k_stack} sel={ks} n_out={n_out} n_in={n_in}: max_rel={max_rel:.3e}"
    );
    if max_rel < 5e-3 {
        println!("moe-down PASS");
        ExitCode::SUCCESS
    } else {
        eprintln!("moe-down MISMATCH");
        ExitCode::FAILURE
    }
}

fn cmd_gpu_mm(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: llm170 gpu-mm <file> <tensor> [t] [rows]");
        return ExitCode::from(2);
    }
    // 아키텍처 자동: 모델 소유 홀더가 mmap 수명 유지 — Weight는 여기서 차입.
    enum Holder {
        Q35(Box<llm170_core::model::Model>),
        Q4(Box<llm170_core::qwen4exp::Model4>),
        /// split 파트 등 hparams 없는 GGUF — mmap만 유지 (raw 텐서 뷰).
        Raw(memmap2::Mmap, u64),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(&args[0]))
        .ok()
        .map(|g| {
            g.arch().map(|s| s == "qwen4exp").unwrap_or(false)
                // split 파트는 general.architecture 없음 — qwen4exp 전용 텐서명으로 판별
                || g.tensors.iter().any(|t| t.name.contains("shexp") || t.name.contains("indexer"))
        })
        .unwrap_or(false);
    let holder = if is_q4 {
        match llm170_core::qwen4exp::Model4::load(std::path::Path::new(&args[0])) {
            Ok(m) => Holder::Q4(Box::new(m)),
            Err(_) => {
                // 파트 파일 폴백 — 데이터 영역만 mmap
                let g = match llm170_gguf::GgufFile::open(std::path::Path::new(&args[0])) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                let file = match std::fs::File::open(&args[0]) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                // SAFETY: 읽기 전용 매핑 — 수정하지 않는다
                let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("error: mmap: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                Holder::Raw(mmap, g.data_offset)
            }
        }
    } else {
        match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
            Ok(m) => Holder::Q35(Box::new(m)),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    let raw_w = match &holder {
        Holder::Raw(mmap, off) => {
            let g = llm170_gguf::GgufFile::open(std::path::Path::new(&args[0])).ok();
            let ti = g.as_ref().and_then(|g| g.tensors.iter().position(|t| t.name == args[1]));
            ti.map(|ti| {
                let t = &g.as_ref().unwrap().tensors[ti];
                let (start, end) = t.file_range(*off).unwrap();
                llm170_core::matmul::Weight {
                    data: &mmap[start as usize..end as usize],
                    ty: t.ty,
                    n_in: t.ne[0],
                    n_out: t.ne[1] * t.ne[2] * t.ne[3],
                }
            })
        }
        _ => None,
    };
    let w = match &holder {
        Holder::Q35(m) => m.w(&args[1]),
        Holder::Q4(m) => m.w(&args[1]),
        Holder::Raw(..) => raw_w,
    };
    let Some(w) = w else {
        eprintln!("tensor not found: {}", args[1]);
        return ExitCode::FAILURE;
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
    // LLM170_X_FILE: 엔진 실패 지점에서 덤프한 실제 활성값으로 재생 (비트 단위 재현)
    if let Ok(xf) = std::env::var("LLM170_X_FILE") {
        let raw = std::fs::read(&xf).unwrap_or_default();
        if raw.len() >= 16 {
            let tt = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
            let nn = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
            eprintln!("# x 재생: t={tt} n_in={nn} (요청 t={t} n_in={n_in})");
            xs = (0..tt)
                .map(|ti| {
                    let b = 16 + ti * nn * 4;
                    (0..nn)
                        .map(|i| {
                            f32::from_le_bytes(raw[b + i * 4..b + i * 4 + 4].try_into().unwrap())
                        })
                        .collect()
                })
                .collect();
        }
    }
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
        // 위상 귀속(진단): up/launch/read 누적 — op당 고정오버헤드 분해.
        llm170_backend_gpu::timing_report();
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

/// llm170 bench-streams <file> <tensor> [t] [max_n] [iters] — 동시 스트림
/// GEMV 집계 대역폭 (P2-a S0 관문). 스트림 i(값 100+i)에서 동일 GEMV를
/// 독립 x/out 버퍼로 실행: 스트림 간 의존이 없어(가중치 읽기 공유 — 첫
/// 접근 자동 동기화 후 무写) 배리어 없이 정확하다. n=1..max_n 스윕으로
/// 배율 보고. 출력 일치(동일 입력 → 비트 동일)로 기계 정합도 검증.
fn cmd_bench_streams(args: &[String]) -> ExitCode {
    use llm170_core::matmul::Accelerator;
    use std::sync::Arc;
    if args.len() < 2 {
        eprintln!("usage: llm170 bench-streams <file> <tensor> [t] [max_n] [iters]");
        return ExitCode::from(2);
    }
    enum H {
        Q35(Box<llm170_core::model::Model>),
        Q4(Box<llm170_core::qwen4exp::Model4>),
    }
    let holder = match llm170_core::model::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => H::Q35(Box::new(m)),
        Err(e1) => match llm170_core::qwen4exp::Model4::load(std::path::Path::new(&args[0])) {
            Ok(m) => H::Q4(Box::new(m)),
            Err(e2) => {
                eprintln!("error: q35 {e1} / q4 {e2}");
                return ExitCode::FAILURE;
            }
        },
    };
    let w = match &holder {
        H::Q35(m) => m.w(&args[1]),
        H::Q4(m) => m.w(&args[1]),
    };
    let Some(w) = w else {
        eprintln!("tensor not found: {}", args[1]);
        return ExitCode::FAILURE;
    };
    let t: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let max_n: usize = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .clamp(1, 8);
    let iters: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(20);

    let gpu: Arc<dyn Accelerator> = match llm170_backend_gpu::GpuMatmul::new_hip() {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let wbytes = w.data.len() as f64;
    // 결정적 동일 입력 — 전 스트림 동일값 → 출력 비트 일치 검증 가능.
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let xrow: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let xdata: Vec<f32> = (0..t).flat_map(|_| xrow.iter().copied()).collect();

    eprintln!(
        "# bench-streams: {} [{}] n_in={n_in} n_out={n_out} 가중치 {:.1}MB t={t} iters={iters}",
        args[1],
        w.ty.name(),
        wbytes / 1e6
    );
    // LLM170_ALT_TENSOR=<tensor>: 스트림 1+가 다른 가중치를 사용 — 동일
    // 텐서 읽기 경합(같은 페이지/L2 라인)과 진짜 메모리 상한을 구분.
    // 게이트+업 동시 실행(실제 FFN 포크 형상) 재현.
    let alt = std::env::var("LLM170_ALT_TENSOR")
        .ok()
        .and_then(|name| match &holder {
            H::Q35(m) => m.w(&name).map(|w| (w.data.len() as f64, w)),
            H::Q4(m) => m.w(&name).map(|w| (w.data.len() as f64, w)),
        });
    if let Some((abytes, _)) = &alt {
        eprintln!("# alt tensor: {:.1}MB (스트림 1+ 적용)", abytes / 1e6);
    }
    let wfor = |i: usize| -> (f64, llm170_core::matmul::Weight<'_>) {
        match (&alt, i) {
            (Some((abytes, aw)), 1..) => (*abytes, *aw),
            _ => (wbytes, w),
        }
    };
    let mut base_gbps = 0.0f64;
    for n in 1..=max_n {
        let mut bufs = Vec::with_capacity(n);
        let total_bytes = |i: usize| iters as f64 * wfor(i).0;
        for i in 0..n {
            let (wb, wi) = wfor(i);
            let n_out_i = wi.n_out as usize;
            let x = gpu.frame_alloc(t * n_in).expect("frame_alloc");
            gpu.frame_write(x, &xdata).expect("frame_write");
            let o = gpu.frame_alloc(t * n_out_i).expect("frame_alloc");
            bufs.push((x, o, wb, n_out_i));
        }
        // 웜업: 스트림별 1회 (가중치 업로드 + 첫 접근 자동 동기화) + 판독.
        for (i, (x, o, _, n_out_i)) in bufs.iter().enumerate() {
            llm170_backend_gpu::on_stream(100 + i as u64, || {
                let (_, wi) = wfor(i);
                let _ = gpu.frame_mm(*x, &wi, *o, t).expect("warmup mm");
                let mut buf = vec![0.0f32; t * n_out_i];
                let _ = gpu.frame_read(*o, &mut buf).expect("warmup read");
            });
        }
        // 측정: 전 스트림 인큐(비동기) → 각 스트림 fence로 합류.
        let t0 = std::time::Instant::now();
        for (i, (x, o, _, _)) in bufs.iter().enumerate() {
            llm170_backend_gpu::on_stream(100 + i as u64, || {
                let (_, wi) = wfor(i);
                for _ in 0..iters {
                    let _ = gpu.frame_mm(*x, &wi, *o, t).expect("mm");
                }
            });
        }
        for i in 0..n {
            let (_, o, _, n_out_i) = &bufs[i];
            let mut last = vec![0.0f32; t * n_out_i];
            llm170_backend_gpu::on_stream(100 + i as u64, || {
                let _ = gpu.frame_read(*o, &mut last).expect("join read");
            });
        }
        let dt = t0.elapsed().as_secs_f64();
        // 정합도: 동일 입력·가중치 → 스트림 0과 alt가 다른 가중치면 스킵.
        let mut mismatch = false;
        if alt.is_none() {
            let mut refbits: Option<Vec<u32>> = None;
            for (i, (_, o, _, n_out_i)) in bufs.iter().enumerate() {
                let mut buf = vec![0.0f32; t * n_out_i];
                llm170_backend_gpu::on_stream(100 + i as u64, || {
                    let _ = gpu.frame_read(*o, &mut buf).expect("verify read");
                });
                let bits = buf.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
                match &refbits {
                    None => refbits = Some(bits),
                    Some(r) if r == &bits => {}
                    _ => mismatch = true,
                }
            }
        }
        let agg: f64 = (0..n).map(total_bytes).sum::<f64>() / dt;
        if n == 1 {
            base_gbps = agg;
        }
        eprintln!(
            "n={n}: {:.1}ms → 집계 {:.0}GB/s (배율 {:.2}, 스트림당 {:.0}GB/s){}",
            dt * 1000.0,
            agg / 1e9,
            agg / base_gbps.max(1.0),
            agg / n as f64 / 1e9,
            if mismatch { " ✗ 스트림 간 출력 불일치!" } else { "" }
        );
        for (x, o, _, _) in &bufs {
            let _ = gpu.frame_free(*o);
            let _ = gpu.frame_free(*x);
        }
    }
    if max_n >= 1 {
        eprintln!("# 판정: 배율 ≥1.2 → P2-a 진행 (S1 stream_barrier). ≈1.0 → 버스 상한, P4 전환.");
    }
    ExitCode::SUCCESS
}

fn cmd_infer(args: &[String]) -> ExitCode {
    let mut model: Option<PathBuf> = None;
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let mut backend = "cpu".to_string();
    let mut gpu_runtime = std::env::var("LLM170_GPU_RUNTIME").unwrap_or_else(|_| "hip".into());
    let mut mode: Option<llm170_core::mode::Mode> = None;
    let mut spec_k: Option<usize> = None;

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
            "--mode" => match it.next().map(String::as_str).and_then(llm170_core::mode::Mode::from_str) {
                Some(m) => mode = Some(m),
                None => return usage_err("--mode requires universal|cmp-stock|cmp-unlocked"),
            },
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) if k >= 1 && k <= 8 => spec_k = Some(k),
                _ => return usage_err("--spec requires k in 1..=8"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }

    let Some(model_path) = model else {
        return usage_err("--model required");
    };
    if let Some(m) = mode {
        apply_mode(m);
    }
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
    // 아키텍처 판별 → qwen4exp 전용 엔진 분기.
    // ENOENT 윈도우 대기 (LLM170_OPEN_WAIT_SECS) — 판별 실패시 재시도.
    let wait_secs: u64 = std::env::var("LLM170_OPEN_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut arch: Option<String> = None;
    for _ in 0..=wait_secs {
        arch = llm170_gguf::GgufFile::open(&model_path)
            .ok()
            .and_then(|g| g.arch().map(|s| s.to_string()));
        if arch.is_some() {
            break;
        }
        if wait_secs == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    if arch.as_deref() == Some("qwen4exp") {
        return run_q4_infer(&model_path, &prompts, n_predict, ctx, &backend, &gpu_runtime);
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
            // 스펙 디코드 (06) — --spec k 지정 시 MTP 체인 draft·연쇄 수용.
            let spec_k: usize = spec_k.unwrap_or(0);
            let has_mtp = eng.has_mtp();
            let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            if spec_k > 0 && has_mtp && n == 1 {
                let s = 0usize;
                let mut accepted_total = 0usize;
                let mut target_forwards = 0usize;
                let mut cycles = 0usize;
                while gen_tokens[s].len() <= n_predict && !finished[s] {
                    let (acc_toks, tf) = eng.spec_step(s, next[s], spec_k).map_err(|e| e.to_string())?;
                    cycles += 1;
                    target_forwards += tf;
                    for &t in &acc_toks {
                        if gen_tokens[s].len() > n_predict {
                            break;
                        }
                        pos[s] += 1;
                        emit(s, pos[s], t, &eng);
                        gen_tokens[s].push(t);
                        next[s] = t;
                        accepted_total += 1;
                        if t == eos {
                            finished[s] = true;
                        }
                    }
                }
                eprintln!(
                    "# spec(k={spec_k}): {cycles}사이클, 수용 {accepted_total}토큰, 타깃 forward {target_forwards}회 — 수용률/forward {:.2}",
                    accepted_total as f64 / target_forwards.max(1) as f64
                );
            } else {
                if spec_k > 0 && !has_mtp {
                    eprintln!("# --spec 무시: MTP(nextn) 텐서 없음");
                }
            // 배치 디코드 — 활성 시퀀스 묶어 1스텝 (np 상호검증 대상 경로)
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
    gpu_runtime: &str,
) -> ExitCode {
    let t_start = std::time::Instant::now();
    let res = llm170_core::qwen4exp::Model4::load(model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, n, ctx);
            if backend == "gpu" {
                // gpu_runtime 플래그/env 존중 — 종전 new_hip() 하드코딩은
                // qwen4exp에서 Vulkan 선택이 무시되는 결함 (2026-09-01 발견).
                let acc: std::sync::Arc<dyn llm170_core::matmul::Accelerator> =
                    if gpu_runtime == "vulkan" {
                        std::sync::Arc::new(
                            llm170_backend_gpu::GpuMatmul::new_vulkan().map_err(|e| e.to_string())?,
                        )
                    } else {
                        std::sync::Arc::new(
                            llm170_backend_gpu::GpuMatmul::new_hip().map_err(|e| e.to_string())?,
                        )
                    };
                eng = eng.with_acc(acc);
                eprintln!("# backend: gpu ({gpu_runtime}) — qwen4exp");
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
