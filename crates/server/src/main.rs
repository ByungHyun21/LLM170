//! llm170 CLI.
//!
//! - gguf-dump: 모델 구조·양자화 믹스 덤프 (무게 미로딩)
//! - infer: qwen35 CPU 참조 추론 (greedy). 토큰 id 입력 — 토크나이저는 후속 단계.

mod bench;
mod engine;
mod infer;
mod probes;
mod http;
mod tokenize;
mod vl;

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = r#"
llm170 — AMD APU 타깃 순수 Rust 추론 엔진 (CPU·HIP·Vulkan)

주요 커맨드:
  llm170 gguf-dump [--meta-only] [--limit N] <file.gguf>
      GGUF 메타데이터·텐서 구성 덤프 (무게 미로딩)
  llm170 infer --model <file.gguf> --prompt-tokens <ids> [--prompt-tokens <ids> ...]
              [--n-predict N] [--ctx N] [--backend cpu|gpu] [--gpu-runtime hip|vulkan] [--spec k]
      greedy 추론 (JSONL {"seq","pos","token","text"}).
      --prompt-tokens 반복 = 병렬 시퀀스(np). --backend gpu: 원시 디코더 상주 디코드.
  llm170 serve --model <file.gguf> [--port N] [--ctx N] [--backend cpu|gpu] [--mode M]
      OpenAI/Anthropic 호환 HTTP 서버.
  llm170 vl --model <llm.gguf> --mmproj <mmproj.gguf> --image <img> [--image <img>...]
            [--spec k] [--n-predict N] [--prefix-tokens ids] [--question-tokens ids]
      비전 인코딩 + LLM 스플라이스 추론.
  llm170 bench --model <file.gguf> [--pp N] [--tg N] [--reps N] [--ctx N]
              [--backend cpu|gpu] [--gpu-runtime hip|vulkan] [--spec k]
      llama-bench 규격 PP/TG 측정 (t/s).
  llm170 check <model.gguf> [--quick] [--backend cpu|gpu]
      텐서 스캔(NaN/Inf) + GPU↔CPU GEMM 상호검증 + 장문 청크 스모크.
  llm170 w4a8-check <file> <tensor> [t] [rows]
      W4A8 변형 ↔ f32 기준 상호검증.
  llm170 dequant <file> <tensor> <row> <n>
      디양자화 값 프로브.

개발 프로브 (backend-gpu 검증·타이밍):
  rawhip-check <file> <tensor>   HIP GEMV ↔ CPU 미러 to_bits 검증
  gpu-raw-probe [iters]          원시 런치 오버헤드
  dims <file> [tensor...]        텐서 차원 조회
  mm-bench2 | mm-bench | mm-tile | launch-probe | roof-test | bw-test | dp4a-test
  tty-probe [file]               타입별 텐서 수·용량 집계
  vk-check                       Vulkan 장치·coopmat·axpy 스모크
  vk-gemv-check <file> <tensor> [t]   엔진 경로(quant+gemv3) GEMV 검증
  vk-gemv8-check <file> <tensor> [t]  gemv8 패밀리 검증+타이밍
  vk-tile-check <file> <tensor> [t]    coopmat 타일 패밀리 검증 (t≤64)
  vk-mmq-check <file> <tensor> [t]    i8 GEMM(plans/23) 검증
  vk-sdot-probe                  OpSDot 장치 지원 프로브
  gdn-check | subsum-check       GDN/서브그룹 축소 커널 검증
  qk-check | iq3s-probe          qk_rope/iq3_s 커널 검증
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
    if let Some(cmd) = args.first().map(String::as_str) {
        if let Some(code) = probes::run(cmd, &args[1..]) {
            return code;
        }
    }
    match args.first().map(String::as_str) {
        Some("gguf-dump") => cmd_gguf_dump(&args[1..]),
        Some("infer") => infer::cmd_infer(&args[1..]),
        Some("serve") => return cmd_serve(&args[1..]),
        Some("rawhip-check") => return probes::run("rawhip-check", &args[1..]).unwrap(),
        Some("vl") => return vl::cmd_vl(&args[1..]),
        Some("bench") => return bench::cmd_bench(&args[1..]),
        Some("check") => return probes::run_check(&args[1..]),
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

/// &str → Option<Vec<u32>> (vl 플래그 파싱용).
fn parse_ids_ref(s: &str) -> Option<Vec<u32>> {
    parse_ids(s).ok()
}

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}





