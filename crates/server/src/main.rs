//! llm170 CLI.
//!
//! - gguf-dump: 모델 구조·양자화 믹스 덤프 (무게 미로딩)
//! - infer: qwen35 CPU 참조 추론 (greedy). 토큰 id 입력 — 토크나이저는 후속 단계.

mod bench;
mod engine;
mod exl3_engine;
mod exl3_hip_engine;
mod http;
mod infer;
mod json;
mod modcheck;
mod oai;
mod perplexity;
mod probes;
mod resource;
mod sched;
mod tokenize;
mod unicode_data;
mod vl;

/// 4분할 모델의 part2 경로 유도 — part1 메타(토크나이저) 실패 시 대안.
/// serve·tokenize가 같은 규칙을 썼다(plans/109 P5 단일화).
fn part2_path(model: &std::path::Path) -> Option<std::path::PathBuf> {
    let stem = model.file_name().and_then(|s| s.to_str()).unwrap_or("");
    stem.contains("-00001-of-")
        .then(|| model.with_file_name(stem.replace("-00001-of-", "-00002-of-")))
}

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = r#"
llm170 — AMD APU 타깃 순수 Rust 추론 엔진 (CPU·HIP·Vulkan)

주요 커맨드:
  llm170 gguf-dump [--meta-only] [--limit N] <file.gguf>
      GGUF 메타데이터·텐서 구성 덤프 (무게 미로딩)
  llm170 infer --model <file.gguf> --prompt-tokens <ids> [--prompt-tokens <ids> ...]
              [--n-predict N] [--ctx N] [--backend cpu|hip|vulkan] [--spec k]
      greedy 추론 (JSONL {"seq","pos","token","text"}).
      --prompt-tokens 반복 = 병렬 시퀀스(np). --backend hip|vulkan: 원시 디코더 상주 디코드.
  llm170 serve --model <file.gguf|exl3_dir> [--port N] [--ctx N] [--slots N] [--queue N] [--backend cpu|hip|vulkan] [--spec k] [--ple-table auto|ram|ssd] [--ple-cache MiB]
      OpenAI/Anthropic 호환 HTTP 서버. --slots N: 동시 요청 배치 디코드 슬롯(기본 1).
  llm170 vl --model <llm.gguf> --mmproj <mmproj.gguf> --image <img> [--image <img>...]
            [--spec k] [--n-predict N] [--prefix-tokens ids] [--question-tokens ids]
      비전 인코딩 + LLM 스플라이스 추론.
  llm170 bench --model <file.gguf> [--pp N] [--tg N] [--reps N] [--ctx N]
              [--backend cpu|hip|vulkan] [--spec k]
      llama-bench 규격 PP/TG 측정 (t/s).
  llm170 check <model.gguf> [--quick] [--backend cpu|hip|vulkan]
      텐서 스캔(NaN/Inf) + GPU↔CPU GEMM 상호검증 + 장문 청크 스모크.
  llm170 w4a8-check <file> <tensor> [t] [rows]
      W4A8 변형 ↔ f32 기준 상호검증.
  llm170 dequant <file> <tensor> <row> <n>
      디양자화 값 프로브.
  llm170 perplexity --model <file.gguf> --prompt-tokens <ids> [--ctx N]
      NLL·perplexity 산출 (품질 게이트, CPU 전용).
  llm170 exl3-load [exl3_dir]
      EXL3 모델 레지스트리 구축 + §7.1 완전성 검증 (mmap).
  llm170 exl3-check [exl3_dir] [q8.gguf]
      EXL3 트렐리스 디코드 ↔ GGUF Q8 대조 (K=3/4/5 corr 기준 0.97).
  llm170 exl3-vk-check [exl3_dir] [tensor-key]
      EXL3 vk 3커널(had_in/gemv/had_out)+FFN ew(silu·mul) ↔ CPU 미러 비트/FMA/허용치 검증 + 속도.
  llm170 exl3-pp [exl3_dir] [token_ids] [n_predict]
      EXL3 T-배치 프리필(하다마드/GEMM/GDN 청크) — 순차 대비 로짓·토큰 검증 + pp t/s (plans/121).

개발 프로브 (backend-gpu 검증·타이밍):
  rawhip-check <file> <tensor>   HIP GEMV ↔ CPU 미러 to_bits 검증
  gpu-raw-probe [iters]          원시 런치 오버헤드
  dims <file> [tensor...]        텐서 차원 조회
  mm-bench2 | mm-bench | mm-tile | launch-probe | bw-test | dp4a-test
  tty-probe [file]               타입별 텐서 수·용량 집계
  vk-check                       Vulkan 장치·coopmat·axpy 스모크
  vk-gemv-check <file> <tensor> [t]   엔진 경로(quant+gemv3) GEMV 검증
  vk-gemv8-check <file> <tensor> [t]  gemv8 패밀리 검증+타이밍
  vk-frame-check <file> <tensor>       vk 프레임 코어(버퍼/EW/GEMM) CPU 대조
  gdn-check | subsum-check       GDN/서브그룹 축소 커널 검증
  qk-check | iq3s-probe          qk_rope/iq3_s 커널 검증
  llm170 help
"#;

/// 모델 적재 서브커맨드 공용 인자 (plans/78 R5) — main에서 1회 파싱해
/// 사전 리소스 가드와 serve/infer/vl/bench가 같은 값을 본다(이중 파싱 제거).
/// `--flag value`와 `--flag=value` 양형 지원. `rest`는 공용 플래그(값 포함)를
/// 제외한 나머지 인자 — trio 서브커맨드의 개별 플래그 파싱에 그대로 쓴다.
/// probes/check는従来대로 원본 args를 받는다(자체 파싱 보존).
pub(crate) struct ModelArgs {
    pub model: Option<String>,
    pub backend: Option<String>,
    pub gpu_runtime: Option<String>,
    /// 외장 MTP(nextn) 모듈 경로 (plans/109 P15⑤) — "--mtp <path>".
    pub mtp: Option<String>,
    /// PLE 테이블 오프로드 모드(plans/111 W4c) — "--ple-table auto|ram|ssd".
    pub ple_table: Option<String>,
    /// SSD 블록 캐시 예산 MiB(plans/111 W4c) — "--ple-cache <MiB>".
    pub ple_cache_mib: Option<usize>,
    pub rest: Vec<String>,
}

fn common_value(args: &[String], i: &mut usize, inline: &Option<String>) -> String {
    match inline {
        Some(v) => v.clone(),
        None => {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        }
    }
}

pub(crate) fn parse_model_args(args: &[String]) -> Result<ModelArgs, String> {
    let mut ma = ModelArgs {
        model: None,
        backend: None,
        gpu_runtime: None,
        mtp: None,
        ple_table: None,
        ple_cache_mib: None,
        rest: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (a, None),
        };
        match name {
            "--model" => ma.model = Some(common_value(args, &mut i, &inline)),
            "--backend" => {
                let v = common_value(args, &mut i, &inline);
                // 통합 백엔드 1택(사용자 지시 2026-09-30): cpu|hip|vulkan|cuda.
                // 파싱층에서 (backend, runtime) 쌍으로 정규화 — 엔진 코드는 무변경.
                match v.as_str() {
                    "cpu" => {
                        ma.backend = Some("cpu".into());
                        ma.gpu_runtime = None;
                    }
                    "hip" | "vulkan" => {
                        ma.backend = Some("gpu".into());
                        ma.gpu_runtime = Some(v.clone());
                    }
                    "cuda" => {
                        return Err("--backend cuda: 미구현 (hip|vulkan 사용)".into());
                    }
                    // EXL3 백엔드값 폐지(사용자 지시 2026-10-05): --backend는
                    // 런타임만(cpu|hip|vulkan|cuda). EXL3는 --model이 디렉터리면
                    // 포맷 자동 판별로 라우팅된다.
                    "exl3" | "exl3-hip" => {
                        return Err("--backend exl3* 폐지: EXL3는 --model <EXL3 디렉터리>로 자동 판별 — --backend hip|vulkan|cpu".into());
                    }
                    // 하위호준 별칭 — 종전 2층(--backend gpu --gpu-runtime X) 폐지.
                    "gpu" => {
                        return Err("--backend gpu 폐지: --backend hip|vulkan|cpu 로 지정".into());
                    }
                    _ => return Err(format!("--backend: cpu|hip|vulkan|cuda (got {v})")),
                }
            }
            "--gpu-runtime" => {
                // 통합 폐지(2026-09-30): --backend hip|vulkan 이 단일 선택지다.
                return Err("--gpu-runtime 폐지: --backend hip|vulkan 사용".into());
            }
            "--mtp" => ma.mtp = Some(common_value(args, &mut i, &inline)),
            "--ple-table" => {
                let v = common_value(args, &mut i, &inline);
                if v != "auto" && v != "ram" && v != "ssd" {
                    return Err(format!("--ple-table: auto|ram|ssd (got {v})"));
                }
                ma.ple_table = Some(v);
            }
            "--ple-cache" => {
                let v = common_value(args, &mut i, &inline);
                match v.parse::<usize>() {
                    Ok(n) if n >= 16 => ma.ple_cache_mib = Some(n),
                    _ => return Err(format!("--ple-cache: MiB ≥ 16 (got {v})")),
                }
            }
            _ => ma.rest.push(a.to_string()),
        }
        i += 1;
    }
    Ok(ma)
}

fn main() -> ExitCode {
    let code = run_main();
    // A5(plans/129): 폴백 누계 종료 출력 — 카운터는 프로세스 로컬이라
    // `llm170 diag fb`(신규 프로세스)는 향상 0건이다. 폴백이 일어난 바로 그
    // 프로세스(infer·bench·프로브 등)가 자기 누계를 stderr에 남긴다.
    // serve는 Ctrl-C로 즉사해 이 출력을 건너뜀 — serve 관측은 ONCE 로그가 담당.
    let r = llm170_diag::fb::report();
    if !r.is_empty() {
        eprint!("[fb] 폴백 누계:\n{r}");
    }
    code
}

fn run_main() -> ExitCode {
    // plans/87 §2 — 와치독(스텔 보고·옵션 FAIL 자결).
    if let Some(v) = std::env::var("LLM170_WATCHDOG")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        llm170_diag::watchdog::spawn(v, std::env::var_os("LLM170_WATCHDOG_FAIL").is_some());
    }
    llm170_diag::fp::init_from_env();
    let args: Vec<String> = std::env::args().skip(1).collect();
    // 107 W12 (README 재검증 포착): 무인자 실행이 args[1..]로 패닉 —
    // 빈 인자는 USAGE 안내로.
    if args.is_empty() {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    // 공용 인자 1회 파싱 (plans/78 R5) — 아래 가드와 trio 디스패치가 공유.
    let ma = match parse_model_args(&args[1..]) {
        Ok(ma) => ma,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    // 사전 리소스 가드(2026-09-16): 이중 적재로 호스트가 먹통되는 사고 방지.
    // 대상 판정은 resource::guard_target 순수함수(A2/R1 추출, plans/129) —
    // 서브커맨드×인자 형태 계약은 표 테스트(guard_target_cases)가 고정하고
    // 무가드 적재 프로브 폐쇄(A13)도 같은 표가 담당한다.
    if !matches!(
        args.first().map(String::as_str),
        Some("gguf-dump") | Some("tokenize")
    ) {
        // 가드 대상 판정은 resource::guard_target 순수함수(A2/R1 추출, plans/129) —
        // 표 테이블 테스트가 계약을 고정한다(무가드 프로브 폐쇄 A13 포함).
        if let Some(gt) = resource::guard_target(
            args.first().map(String::as_str).unwrap_or(""),
            ma.model.as_deref(),
            ma.backend.as_deref(),
            ma.gpu_runtime.as_deref(),
            &ma.rest,
        ) && let Err(e) = resource::preflight(&gt.path, gt.gpu)
        {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
    // cubecl 커널 컴파일 오류 등 log 패싯 메시지 노출 — stderr 간이 로거.
    struct EL;
    impl log::Log for EL {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, r: &log::Record) {
            eprintln!("[{}] {}", r.level(), r.args());
        }
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
    if let Some(cmd) = args.first().map(String::as_str)
        && let Some(code) = probes::run(cmd, &args[1..])
    {
        return code;
    }
    match args.first().map(String::as_str) {
        Some("gguf-dump") => cmd_gguf_dump(&args[1..]),
        Some("infer") => infer::cmd_infer(&ma.rest, &ma),
        Some("serve") => cmd_serve(&ma.rest, &ma),
        Some("vl") => vl::cmd_vl(&ma.rest, &ma),
        Some("bench") => bench::cmd_bench(&ma.rest, &ma),
        Some("perplexity") => perplexity::cmd_perplexity(&ma.rest, &ma),
        Some("check") => probes::run_check(&args[1..]),
        Some("mod-check") => modcheck::cmd_mod_check(&args[1..]),
        Some("tokenize") => cmd_tokenize(&ma),
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

/// llm170 serve --model <file> [--port N] [--ctx N] [--slots N] [--backend cpu|hip|vulkan] [--spec k]
fn cmd_serve(args: &[String], ma: &ModelArgs) -> ExitCode {
    let mut port = 8080u16;
    let mut queue: Option<usize> = None;
    let mut slots: Option<usize> = None;
    let mut spec_k = 0usize;
    let mut ctx = 4096usize;
    let backend = ma.backend.clone().unwrap_or_else(|| "cpu".into());
    let gpu_runtime = ma.gpu_runtime.clone().unwrap_or_default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => match it.next().and_then(|v| v.parse().ok()) {
                Some(p) => port = p,
                None => return usage_err("--port requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse().ok()) {
                Some(c) => ctx = c,
                None => return usage_err("--ctx requires a number"),
            },
            "--slots" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(s) => slots = Some(s.clamp(1, 16)),
                None => return usage_err("--slots requires a number in 1..=16"),
            },
            "--queue" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(q) => queue = Some(q.max(1)),
                None => return usage_err("--queue requires a number"),
            },
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) => spec_k = k.min(8),
                None => return usage_err("--spec requires k in 1..=8"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = ma.model.clone().map(PathBuf::from) else {
        return usage_err("--model required");
    };
    // A12(plans/129): exl3-hip 엔진은 단일 슬롯 — --slots>1이 슬롯 생성 시점의
    // 점유 슬롯 reset으로 교묘하게 상태를 파괴했다(엔진 코드는 대응하지만
    // 진입에서 거부하는 게 계약상 정확). vk 엔진은 다중 슬롯 지원 — 제외.
    if model_path.is_dir() && gpu_runtime != "vulkan" && slots.unwrap_or(1) > 1 {
        return usage_err("EXL3 hip 백엔드는 단일 슬롯만 지원 — --slots 1");
    }
    if spec_k > 0 {
        // GPU 스펙 경로 강제 (스레드 기동 전 단일 스레드 시점 env 설정).
        // 안전성: 이 시점은 단일 스레드 (엔진/슬롯 스레드 기동 전).
        unsafe { std::env::set_var("LLM170_SPEC_GPU", "1") };
        let _ = crate::engine::SPEC_K.set(spec_k);
        eprintln!("# spec: k={spec_k} (MTP 스펙 디코드)");
    }
    // 토크나이저 적재 (part1 메타 → 실패시 part2)
    let part2 = part2_path(&model_path);
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
    // QA-12(plans/114): serve의 텍스트 엔드포인트(/v1/chat·completions·
    // messages)는 빈 인코딩 ids=[] 잡을 엔진에 투입해 쓰레기 스트림(또는 엔진
    // Err)을 뿜었다 — 조용한 서락 대신 기동 치명 오류. (infer·vl은 ids 인터
    // 페이스라 빈 토크나이저로도 동작 — 이곳 serve에만 적용.)
    let Some(tok) = tok else {
        eprintln!("error: tokenizer load 실패(5회 재시도) — serve 텍스트 요청에 필수");
        return ExitCode::FAILURE;
    };
    // A19(plans/129): load는 어느 파트에도 토크나이저가 없으면 Ok(empty)를
    // 돌려준다 — Some(empty) 통과가 쓰레기 스트림을 뿜었다. 치명 오류로.
    if tok.is_empty() {
        eprintln!("error: 토크나이저 비음(어느 파트에도 없음) — serve 텍스트 요청에 필수");
        return ExitCode::FAILURE;
    }
    let _ = engine::TOKENIZER.set(tok);
    let req = engine::InferRequest {
        model: model_path.clone(),
        ctx,
        mtp: ma.mtp.clone().map(PathBuf::from),
        ple_table: ma.ple_table.clone(),
        ple_cache_mib: ma.ple_cache_mib,
    };
    // 포맷 자동 판별(사용자 계약 2026-10-05): 모델 경로가 디렉터리(EXL3
    // 아카이브)면 --backend 런타임(hip|vulkan)으로 EXL3 엔진을 고른다.
    // cpu+디렉터리는 명확한 에러(무음 Q4 로드 실패 방지).
    if model_path.is_dir() && backend != "gpu" {
        eprintln!("error: EXL3(디렉터리)는 GPU 런타임 필요 — --backend hip|vulkan");
        return ExitCode::FAILURE;
    }
    let sel = if model_path.is_dir() {
        if gpu_runtime == "vulkan" {
            engine::BackendSel::Exl3
        } else {
            engine::BackendSel::Exl3Hip
        }
    } else if backend == "gpu" {
        if gpu_runtime.is_empty() {
            engine::BackendSel::Gpu
        } else {
            engine::BackendSel::GpuRuntime(gpu_runtime)
        }
    } else {
        engine::BackendSel::Cpu
    };
    match http::serve(&format!("127.0.0.1:{port}"), req, sel, slots, queue) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `llm170 tokenize --model <gguf> [--no-special] (--text <s> | --file <f> | --stdin)`
/// llama-tokenize 대응 출력 `[id, ...]` — plans/83 A 검증·디버깅용.
fn cmd_tokenize(ma: &ModelArgs) -> ExitCode {
    let Some(model) = ma.model.clone() else {
        eprintln!("error: --model required");
        return ExitCode::from(2);
    };
    let model_path = PathBuf::from(model);
    // part1 메타 → 실패시 part2 (serve와 동일 규칙)
    let part2 = part2_path(&model_path);
    let tok = match tokenize::Tokenizer::load(&model_path, part2.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: tokenizer load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let no_special = ma.rest.iter().any(|a| a == "--no-special");
    // A20(plans/129): 위치인자만 준 사용자에게 stdin 판독 무응답처럼 보였다
    // (원장 기록 ⑧) — usage 에러로. --text/--file 값 부재(마지막 인자)도
    // 빈 문자열 조용 인코딩 대신 에러.
    let has_text = ma.rest.iter().any(|a| a == "--text");
    let has_file = ma.rest.iter().any(|a| a == "--file");
    if !has_text && !has_file && ma.rest.iter().any(|a| !a.starts_with("--")) {
        eprintln!(
            "error: 텍스트는 --text <문자열> 또는 --file <경로>로 전달 (위치인자는 무시됩니다)"
        );
        return ExitCode::FAILURE;
    }
    // 플래그도 위치인자도 없으면 stdin 합법 사용 — 계속 진행.
    let text = if let Some(i) = ma.rest.iter().position(|a| a == "--text") {
        let Some(v) = ma.rest.get(i + 1) else {
            eprintln!("error: --text requires a value");
            return ExitCode::FAILURE;
        };
        v.clone()
    } else if let Some(i) = ma.rest.iter().position(|a| a == "--file") {
        let Some(p) = ma.rest.get(i + 1) else {
            eprintln!("error: --file requires a path");
            return ExitCode::FAILURE;
        };
        let p = p.clone();
        match std::fs::read_to_string(&p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: read {p}: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        use std::io::Read;
        let mut buf = String::new();
        if std::io::stdin().read_to_string(&mut buf).is_err() {
            eprintln!("error: stdin read");
            return ExitCode::FAILURE;
        }
        buf
    };
    let ids = tok.encode_opts(&text, !no_special);
    println!(
        "[{}]",
        ids.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    ExitCode::SUCCESS
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

    llm170_diag::span::reset();
    let f = {
        llm170_diag::profile_span!("cli::gguf-dump::total");
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

    if let Some(rep) = llm170_diag::span::report() {
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
    let (row, n): (u64, usize) = match (args[2].parse(), args[3].parse()) {
        (Ok(r), Ok(nn)) => (r, nn),
        _ => {
            eprintln!("error: <row>/<n> must be integers");
            return ExitCode::FAILURE;
        }
    };
    use std::os::unix::fs::FileExt;
    let file = match std::fs::File::open(&args[0]) {
        Ok(fl) => fl,
        Err(e) => {
            eprintln!("error: open {}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };
    let k = t.ne[0];
    let (blck, bsize) = t.ty.block_info();
    let row_bytes = (k / blck * bsize) as usize;
    let Some((start, _)) = t.file_range(f.data_offset) else {
        eprintln!("error: tensor file range 없음");
        return ExitCode::FAILURE;
    };
    let mut buf = vec![0u8; row_bytes];
    if let Err(e) = file.read_exact_at(&mut buf, start + row * row_bytes as u64) {
        eprintln!("error: read row {row}: {e}");
        return ExitCode::FAILURE;
    }
    let mut out = vec![0.0f32; k as usize];
    llm170_core::quant::dequant_row(t.ty, &buf, 0, k, &mut out);
    // A21b(plans/129): n>k 슬라이스 패닉 — 클램프(k가 실제 상한).
    let show = n.min(k as usize);
    let vals: Vec<String> = out[..show].iter().map(|v| format!("{v:.6}")).collect();
    println!("[{}] row {row}: {}", t.ty.name(), vals.join(", "));
    ExitCode::SUCCESS
}

/// llm170 w4a8-check <file> <tensor> [t] [rows] — W4A8 변형 ↔ f32 기준 상호검증.
fn cmd_w4a8_check(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: llm170 w4a8-check <file> <tensor> [t] [rows]");
        return ExitCode::from(2);
    }
    let model = match llm170_core::qwen35::Model::load(std::path::Path::new(&args[0])) {
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
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
// 마커 gpx
// 마커 ehi
