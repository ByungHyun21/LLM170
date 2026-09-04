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
        Some("rawhip-check") => cmd_rawhip_check(&args[1..]),
        Some("vl") => return cmd_vl(&args[1..]),
        Some("gpu-raw-probe") => {
            let iters: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
            match llm170_backend_gpu::rawhip::raw_probe(iters) {
                Ok(msg) => {
                    println!("{msg}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("dims") => {
            let a: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
            print!("{}", llm170_backend_gpu::rawhip::dims_of(a[0], &a[1..]));
            ExitCode::SUCCESS
        }
        Some("mm-bench2") => match llm170_backend_gpu::rawhip::mm_bench() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("vk-gemv-check") => {
            let args2: Vec<String> = std::env::args().collect();
            let path = args2.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args2.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
            let t = args2.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
            match llm170_backend_gpu::rawvk::gemv::gemv_check(&path, &tn, t) {
                Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
                Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
            }
        }
        Some("subsum-check") => match llm170_backend_gpu::rawvk::subsum_check() {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("subsum: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gdn-check") => match llm170_backend_gpu::rawvk::gdn_check() {
            Ok(msg) => {
                for l in msg.split('\n') {
                    println!("{l}");
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("gdn-check: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-check") => match llm170_backend_gpu::rawvk::smoke_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("roof-test") => match llm170_backend_gpu::rawhip::roof_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("mm-tile") => match llm170_backend_gpu::rawhip::mm_tile_bench() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("mm-bench") => match llm170_backend_gpu::rawhip::mm_batch_bench() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("batch-abtest") => match llm170_backend_gpu::rawhip::batch_ab_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("tree-test") => match llm170_backend_gpu::rawhip::tree_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("q6k-abtest") => match llm170_backend_gpu::rawhip::q6k_ab_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("bw-test") => match llm170_backend_gpu::rawhip::bw_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("dp4a-test") => match llm170_backend_gpu::rawhip::dp4a_test() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("exp-ab") => match llm170_backend_gpu::rawhip::exp_ab() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("iq3s-probe") => match llm170_backend_gpu::rawhip::iq3s_probe() {
            Ok(msg) => { println!("{msg}"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        Some("qk-check") => match llm170_backend_gpu::rawhip::qk_check() {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("bench") => return bench::cmd_bench(&args[1..]),
        Some("check") => cmd_check(&args[1..]),
        Some("w4a8-check") => cmd_w4a8_check(&args[1..]),
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
    let mut spec_k = 0usize;
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
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) => spec_k = k.min(8),
                None => return usage_err("--spec requires k in 1..=8"),
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
    if spec_k > 0 {
        // GPU 스펙 경로 강제 (스레드 기동 전 단일 스레드 시점 env 설정).
        // 안전성: 이 시점은 단일 스레드 (엔진/슬롯 스레드 기동 전).
        unsafe { std::env::set_var("LLM170_SPEC_GPU", "1") };
        let _ = crate::engine::SPEC_K.set(spec_k);
        eprintln!("# spec: k={spec_k} (MTP 스펙 디코드)");
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

/// gpu-ew-check — ew 커널 전종 GPU↔CPU 상호검증 (층 GPU 상주 P2-4 1단계).
/// 판정: norm류 max_rel < 1e-6 (f64 경로 — 비트일치 기대), 활성화류 < 1e-5
/// (libm 구현차), moe ids 완전일치.







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
            if spec_k.is_some() {
                eng.mtp_wanted = true; // 스펙 의도 — prefill 훅 활성 (plans/22)
            }
            if gpu_runtime == "vulkan" {
                if std::env::var_os("LLM170_VK_DECODER").is_some() {
                    // GPU 상주 디코드 (plans/19 2단계) — 커널 8종 gdn-check ★
                    match crate::inject_rawvk(&mut eng) {
                        Ok(()) => eprintln!("# backend: gpu (vulkan VkDecoder)"),
                        Err(e) => eprintln!("vk-decoder: {e} (VkAcc로 진행)"),
                    }
                } else {
                    // Vulkan 경로 (plans/12): VkAcc로 matmul만 가속 — GDN·EW는 CPU.
                    match llm170_backend_gpu::rawvk::gemv::VkAcc::new() {
                        Ok(acc) => {
                            eng = eng.with_acc(std::sync::Arc::new(acc));
                            eprintln!("# backend: gpu (vulkan VkAcc)");
                        }
                        Err(e) => eprintln!("vk-acc: {e} (CPU로 진행)"),
                    }
                }
            } else if std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true) {
                crate::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
            }
            if backend == "gpu" && gpu_runtime != "vulkan" {
                eprintln!("# backend: gpu (raw hip)");
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
            if spec_k > 0 && has_mtp && n > 1 && std::env::var_os("LLM170_SPEC_GPU").is_some() {
                // np×spec 병합 (plans/18)
                let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
                let mut accepted_total = 0usize;
                let mut cycles = 0usize;
                let mut min_gen = gen_tokens[0].len();
                for g in gen_tokens.iter() {
                    min_gen = min_gen.min(g.len());
                }
                while min_gen <= n_predict {
                    let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                    if active.is_empty() {
                        break;
                    }
                    let nexts: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                    let acc = eng
                        .spec_step_multi(&active, &nexts, spec_k)
                        .map_err(|e| e.to_string())?;
                    cycles += 1;
                    let mut any = false;
                    for (i, &s) in active.iter().enumerate() {
                        for &t in &acc[i] {
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
                            any = true;
                        }
                    }
                    if !any {
                        break;
                    }
                    min_gen = usize::MAX;
                    for (s, g) in gen_tokens.iter().enumerate() {
                        if !finished[s] {
                            min_gen = min_gen.min(g.len());
                        }
                    }
                }
                eprintln!(
                    "# spec-multi(k={spec_k}, n={n}): {cycles}사이클, 수용 {accepted_total}토큰"
                );
            } else if spec_k > 0 && has_mtp && n == 1 {
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
                eprintln!("# backend: gpu — qwen4exp (cubecl 제거: CPU 폴백)");
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
    let mut xd_gpu: Option<*mut u8> = None;
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
    let ds: Vec<f32> = y.iter().map(|b| b.d).collect();
    // ktab2
    let ktab2: Vec<u32> = (0..256u32)
        .map(|b| {
            let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
            let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
            lo | (hi << 8)
        })
        .collect();
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

/// Engine에 원시 HIP 디코더 주입 — 필요 가중치·상수 전체를 백엔드로.
/// rawhip/VkDecoder 공용 상수 페치 (이름 리맵·타일 포함).
fn raw_consts(
    eng: &llm170_core::model::Engine,
    cnames: &[String],
) -> Vec<(String, Vec<f32>)> {
    let hp = &eng.model.hp;
    let ctx_n = eng.ctx_len();
    cnames
        .iter()
        .filter_map(|k| {
            let v = if k == "cs" {
                let half = hp.n_rot >> 1;
                let mut cs = vec![0.0f32; ctx_n * half * 2];
                for pos in 0..ctx_n {
                    for pp in 0..half {
                        let theta = (hp.rope_base as f32).powf(-(2.0 * pp as f32) / hp.n_rot as f32);
                        let angle = pos as f32 * theta;
                        cs[pos * half * 2 + pp * 2] = angle.cos();
                        cs[pos * half * 2 + pp * 2 + 1] = angle.sin();
                    }
                }
                Some(cs)
            } else if k == "mask" {
                // 인과 마스크 [pos][p]: p<=pos 만 1 — qsa 배치용 (원본 의미 복원)
                let mut m = vec![0.0f32; ctx_n * ctx_n];
                for pos in 0..ctx_n {
                    for pp in 0..=pos {
                        m[pos * ctx_n + pp] = 1.0;
                    }
                }
                Some(m)
            } else if k.ends_with("conv_w") {
                eng.model.f32_vec(&format!("blk.{}.ssm_conv1d.weight", k.split('.').nth(1).unwrap_or("0"))).ok()
            } else {
                let il = k.split('.').nth(1).unwrap_or("0").to_string();
                let (tn, tiled) = if k.ends_with("dt_bias") {
                    (format!("blk.{il}.ssm_dt.bias"), 1)
                } else if k.ends_with("ssm_a") {
                    (k.clone(), 1)
                } else if k.ends_with("ssm_norm") {
                    (format!("blk.{il}.ssm_norm.weight"), hp.dt_rank)
                } else if k.ends_with("post_attention_norm") {
                    (format!("blk.{il}.post_attention_norm.weight"), 1)
                } else if k.ends_with("attn_norm") {
                    (format!("blk.{il}.attn_norm.weight"), 1)
                } else if k.ends_with("post_norm") {
                    (format!("blk.{il}.post_attention_norm.weight"), 1)
                } else if k == "output_norm" {
                    ("output_norm.weight".to_string(), 1)
                } else if k.ends_with("attn_q_norm") {
                    (format!("blk.{il}.attn_q_norm.weight"), hp.n_head)
                } else if k.ends_with("attn_k_norm") {
                    (format!("blk.{il}.attn_k_norm.weight"), hp.n_kv)
                } else {
                    (format!("{k}.weight"), 1)
                };
                eng.model.f32_vec(&tn).ok().map(|v| {
                    if tiled > 1 && (v.len() == hp.d_state || v.len() == hp.head_dim) {
                        v.iter().copied().cycle().take(v.len() * tiled).collect()
                    } else {
                        v
                    }
                })
            };
            v.map(|v| (k.clone(), v))
        })
        .collect()
}

/// VkDecoder 주입 — rawhip과 동일 가중치·상수 목록 (plans/19).
fn inject_rawvk(eng: &mut llm170_core::model::Engine) -> Result<(), String> {
    use llm170_core::matmul::RawDecode;
    let hp = eng.model.hp.clone();
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let (wnames, cnames) = crate::raw_names(eng);
    let mut weights: Vec<(String, llm170_core::matmul::Weight<'_>)> = Vec::new();
    for n in &wnames {
        let w = eng.model.wchk(n).map_err(|e| e.to_string())?;
        weights.push((n.clone(), w));
    }
    let mut consts = crate::raw_consts(eng, &cnames);
    // VkDecoder 마스크: u32 all-ones (t=1 디코드 행은 p<=pos 전부 활성).
    {
        let cl = eng.ctx_len();
        if let Some(m) = consts.iter_mut().find(|(k, _)| k == "mask") {
            m.1 = (0..cl * cl).map(|_| f32::from_bits(1)).collect();
        }
    }
    let rd: std::sync::Arc<llm170_backend_gpu::rawvk::decoder::VkDecoder> =
        std::sync::Arc::new(llm170_backend_gpu::rawvk::decoder::VkDecoder::new());
    rd.raw_init(&hp, &weights, &consts, eng.seqs.len(), eng.ctx_len(), is_recr)
        .map_err(|e| format!("raw_init(vk): {e}"))?;
    eng.raw_decode = Some(rd);
    Ok(())
}

fn raw_names(eng: &llm170_core::model::Engine) -> (Vec<String>, Vec<String>) {
    let hp = &eng.model.hp;
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let mut wnames: Vec<String> = Vec::new();
    let mut cnames: Vec<String> = Vec::new();
    for il in 0..hp.n_layer {
        cnames.push(format!("blk.{il}.attn_norm"));
        cnames.push(format!("blk.{il}.post_norm"));
        if is_recr[il] {
            for w in ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out"] {
                wnames.push(format!("blk.{il}.{w}.weight"));
            }
            cnames.push(format!("blk.{il}.conv_w"));
            cnames.push(format!("blk.{il}.dt_bias"));
            cnames.push(format!("blk.{il}.ssm_a"));
            cnames.push(format!("blk.{il}.ssm_norm"));
        } else {
            for w in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                wnames.push(format!("blk.{il}.{w}.weight"));
            }
            cnames.push(format!("blk.{il}.attn_q_norm"));
            cnames.push(format!("blk.{il}.attn_k_norm"));
        }
        for w in ["ffn_gate", "ffn_up", "ffn_down"] {
            wnames.push(format!("blk.{il}.{w}.weight"));
        }
    }
    wnames.push("output.weight".into());
    // MTP층 (blk.64) — spec decode용 (has_mtp 시)
    if eng.has_mtp() {
        let mtp = 64usize;
        for w in ["attn_q", "attn_k", "attn_v", "attn_output",
                  "ffn_gate", "ffn_up", "ffn_down", "nextn.eh_proj"] {
            wnames.push(format!("blk.{mtp}.{w}.weight"));
        }
        for c in ["attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm",
                  "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"] {
            cnames.push(format!("blk.{mtp}.{c}"));
        }
    }
    cnames.push("output_norm".into());
    cnames.push("cs".into());
    cnames.push("mask".into());
    (wnames, cnames)
}

fn inject_rawhip(eng: &mut llm170_core::model::Engine) -> Result<(), String> {
    let hp = eng.model.hp.clone();
    let (wnames, cnames): (Vec<String>, Vec<String>) = crate::raw_names(eng);
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let weights: Vec<(String, llm170_core::matmul::Weight<'_>)> = wnames
        .iter()
        .filter_map(|k| eng.model.wchk(k).ok().map(|w| (k.clone(), w)))
        .collect();
    if weights.len() != wnames.len() {
        return Err(format!("rawhip: 가중치 누락 {}/{}", weights.len(), wnames.len()));
    }
    let consts = crate::raw_consts(eng, &cnames);
    let rd: std::sync::Arc<llm170_backend_gpu::rawhip::decode::RawDecoder> =
        std::sync::Arc::new(llm170_backend_gpu::rawhip::decode::RawDecoder::new());
    use llm170_core::matmul::RawDecode;
    rd.raw_init(&hp, &weights, &consts, eng.seqs.len(), eng.ctx_len(), is_recr)
        .map_err(|e| format!("raw_init: {e}"))?;
    eng.raw_decode = Some(rd);
    Ok(())
}

/// vl — mmproj 비전 인코딩 + LLM 스플라이스 추론 (plans/16).
fn cmd_vl(args: &[String]) -> ExitCode {
    let mut model: Option<PathBuf> = None;
    let mut mmproj: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut n_predict = 48usize;
    let mut ctx = 4096usize;
    let mut backend = "gpu".to_string();
    let mut spec_k = 0usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = it.next().map(PathBuf::from),
            "--mmproj" => mmproj = it.next().map(PathBuf::from),
            "--image" => {
                if let Some(p) = it.next() {
                    images.push(PathBuf::from(p));
                }
            }
            "--n-predict" => n_predict = it.next().and_then(|v| v.parse().ok()).unwrap_or(48),
            "--ctx" => ctx = it.next().and_then(|v| v.parse().ok()).unwrap_or(4096),
            "--backend" => backend = it.next().cloned().unwrap_or_else(|| "gpu".into()),
            "--spec" => spec_k = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            _ => {}
        }
    }
    let (model, mmproj) = match (model, mmproj) {
        (Some(m), Some(p)) => (m, p),
        _ => {
            eprintln!("usage: llm170 vl --model <llm.gguf> --mmproj <mmproj.gguf> --image <img> [--image <img>...] [--spec k] [--n-predict N]");
            return ExitCode::from(2);
        }
    };
    if images.is_empty() {
        eprintln!("usage: at least one --image required");
        return ExitCode::from(2);
    }
    // qwen3.8 VL 템플릿 토큰 (서버 /tokenize 확정): <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in one short sentence.<|im_end|>\n<|im_start|>assistant\n
    let prompt: Vec<u32> = vec![248045, 846, 198, 248053, 248056, 248054, 72240, 411, 2099, 303, 799, 2716, 11316, 13, 248046, 198, 248045, 74455, 198];
    // 1) 이미지 → 스마트리사이즈·정규화 → CLIP 인코딩 (이미지별 = 시퀀스별)
    let t0 = std::time::Instant::now();
    let mut clip = match llm170_core::clip::Clip::load(&mmproj) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("clip load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_img = images.len();
    let mut all_vis: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_img);
    let mut vit_cache: Option<(std::sync::Arc<llm170_backend_gpu::rawhip::RawCtx>, std::sync::Arc<llm170_backend_gpu::rawhip::vit::Vit>, usize)> = None;
    for (si, ipath) in images.iter().enumerate() {
        let img = match image::open(ipath) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("image: {e}");
                return ExitCode::FAILURE;
            }
        };
        let (iw, ih) = (img.width() as i64, img.height() as i64);
        // qwen3vl smart_resize (align 32, 토큰 8..4096) + Pillow bicubic
        let (tw, th) = llm170_core::clip_preproc::smart_resize(iw, ih, 16, 2, 8, 4096);
        let raw = img.as_raw();
        let rgb8 = llm170_core::clip_preproc::resize_pillow(raw, iw as usize, ih as usize, tw as usize, th as usize, true);
        eprintln!("# resize[{si}] {iw}x{ih} -> {tw}x{th}");
        let (tw, th) = (tw as usize, th as usize);
        let mut px = vec![0f32; tw * th * 3];
        for (i, v) in rgb8.iter().enumerate() {
            px[i] = ((*v as f32) / 255.0 - 0.5) / 0.5;
        }
        let vis = if backend != "cpu" {
            // GPU 경로 (plans/17): CPU conv+pos → ViT 27블록·merger GPU
            let (n_embd, n_head, n_blk, eps) = (1152usize, 16usize, 27usize, 1e-6f32);
            let tmax = (tw / 16) * (th / 16);
            let v = (|| -> Result<Vec<Vec<f32>>, String> {
                if vit_cache.is_none() {
                    let weights = clip.vit_weights()?;
                    let n_ff = clip.n_ff();
                    let tw0 = std::time::Instant::now();
                    let ctx = std::sync::Arc::new(llm170_backend_gpu::rawhip::RawCtx::new()?);
                    let vit = llm170_backend_gpu::rawhip::vit::Vit::new(
                        ctx.clone(), weights, n_embd, n_head, n_ff, n_blk, eps, 16, tmax,
                    )?;
                    eprintln!("# vit weights+upload {:.1}s", tw0.elapsed().as_secs_f64());
                    vit_cache = Some((ctx, std::sync::Arc::new(vit), tmax));
                }
                let (_, vit, tmax0) = vit_cache.as_ref().unwrap();
                if tmax > *tmax0 {
                    return Err(format!("tmax {tmax} > 초기화 {tmax0} — 이미지 해상도 초과"));
                }
                let tp0 = std::time::Instant::now();
                let (toks, yx, pw, ph) = clip.prep_tokens(&px, tw, th)?;
                eprintln!("# vit prep(conv) {:.1}s", tp0.elapsed().as_secs_f64());
                let tf0 = std::time::Instant::now();
                let flat = vit.forward(&toks, &yx, pw, ph)?;
                eprintln!("# vit forward {:.1}s", tf0.elapsed().as_secs_f64());
                let n_out = flat.len() / 5120;
                Ok((0..n_out).map(|i| flat[i * 5120..(i + 1) * 5120].to_vec()).collect())
            })();
            match v {
                Ok(rows) => rows,
                Err(e) => {
                    eprintln!("vit gpu: {e} — CPU 폴백");
                    match clip.encode(&px, tw, th) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("clip encode: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
        } else {
            match clip.encode(&px, tw, th) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("clip encode: {e}");
                    return ExitCode::FAILURE;
                }
            }
        };
        eprintln!("# clip[{si}]: {} tokens (총 {:.1}s)", vis.len(), t0.elapsed().as_secs_f64());
        all_vis.push(vis);
    }
    // 3) LLM
    let m = match llm170_core::model::Model::load(&model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut eng = llm170_core::model::Engine::new(m, n_img, ctx);
    if spec_k > 0 {
        eng.mtp_wanted = true; // 스펙 의도 — prefill 훅 활성
    }
    if backend != "cpu" {
        let rt = std::env::var("LLM170_GPU_RUNTIME").unwrap_or_else(|_| "hip".into());
        if rt == "vulkan" {
            if std::env::var_os("LLM170_VK_DECODER").is_some() {
                match crate::inject_rawvk(&mut eng) {
                    Ok(()) => eprintln!("# backend: gpu (vulkan VkDecoder)"),
                    Err(e) => eprintln!("vk-decoder: {e} — CPU 진행"),
                }
            } else {
                eprintln!("# backend: vulkan VK_DECODER 미지정 — LLM은 CPU 진행");
            }
        } else {
            crate::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
        }
    }
    let eos = 248044u32;
    let t1 = std::time::Instant::now();
    let mut last_logits = Vec::with_capacity(n_img);
    for s in 0..n_img {
        let l = match eng.prefill_vision(s, &prompt, 248056, &all_vis[s]) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("prefill_vision: {e}");
                return ExitCode::FAILURE;
            }
        };
        last_logits.push(l);
    }
    eprintln!("# prefill({n_img} seqs) {:.1}s", t1.elapsed().as_secs_f64());
    let mut finished = vec![false; n_img];
    let mut gen_toks: Vec<Vec<u32>> = vec![Vec::new(); n_img];
    let mut texts: Vec<String> = vec![String::new(); n_img];
    let mut next: Vec<u32> = last_logits.iter().map(|l| llm170_core::model::greedy(l)).collect();
    for s in 0..n_img {
        texts[s].push_str(&eng.piece(next[s]));
        if next[s] == eos {
            finished[s] = true;
        }
        gen_toks[s].push(next[s]);
    }
    let emit = |s: usize, t: u32, eng: &llm170_core::model::Engine, texts: &mut Vec<String>| {
        if t != eos {
            texts[s].push_str(&eng.piece(t));
        }
    };
    let spec_on = spec_k > 0
        && eng.has_mtp()
        && std::env::var_os("LLM170_SPEC_GPU").is_some();
    if spec_k > 0 && !eng.has_mtp() {
        eprintln!("# --spec 무시: MTP(nextn) 텐서 없음");
    }
    let gen_res = (|| -> Result<(), String> {
        if spec_on && n_img > 1 {
            while gen_toks.iter().filter(|g| !g.is_empty()).min_by_key(|g| g.len()).map(|g| g.len()).unwrap_or(0) <= n_predict {
                let active: Vec<usize> = (0..n_img).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                let nexts: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                let acc = eng.spec_step_multi(&active, &nexts, spec_k).map_err(|e| e.to_string())?;
                let mut any = false;
                for (i, &s) in active.iter().enumerate() {
                    for &t in &acc[i] {
                        if gen_toks[s].len() > n_predict {
                            break;
                        }
                        emit(s, t, &eng, &mut texts);
                        gen_toks[s].push(t);
                        next[s] = t;
                        if t == eos {
                            finished[s] = true;
                        }
                        any = true;
                    }
                }
                if !any {
                    break;
                }
            }
        } else if spec_on {
            let s = 0usize;
            while gen_toks[s].len() <= n_predict && !finished[s] {
                let (acc_toks, _tf) = eng.spec_step(s, next[s], spec_k).map_err(|e| e.to_string())?;
                for &t in &acc_toks {
                    if gen_toks[s].len() > n_predict {
                        break;
                    }
                    emit(s, t, &eng, &mut texts);
                    gen_toks[s].push(t);
                    next[s] = t;
                    if t == eos {
                        finished[s] = true;
                    }
                }
            }
        } else {
            for _step in 0..n_predict {
                let active: Vec<usize> = (0..n_img).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                let toks: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                let logits = eng.decode(&active, &toks).map_err(|e| e.to_string())?;
                for (i, &s) in active.iter().enumerate() {
                    let t = llm170_core::model::greedy(&logits[i]);
                    next[s] = t;
                    emit(s, t, &eng, &mut texts);
                    gen_toks[s].push(t);
                    if t == eos {
                        finished[s] = true;
                    }
                }
            }
        }
        Ok(())
    })();
    if let Err(e) = gen_res {
        eprintln!("decode: {e}");
    }
    for s in 0..n_img {
        if n_img > 1 {
            println!("seq{s}: {}", texts[s]);
        } else {
            println!("{}", texts[s]);
        }
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!("# done");
    ExitCode::SUCCESS
}
