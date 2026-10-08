//! llm170 CLI.
//!
//! - infer: qwen35 CPU 참조 추론 (greedy). 토큰 id 입력 — 토크나이저는 후속 단계.

mod engine;
mod gpu_engine;
mod http;
mod infer;
mod json;
mod metrics;
mod oai;
mod probes;
mod resource;
mod sched;
mod tokenize;
mod unicode_data;

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = r#"
llm170 — 순수 Rust 추론 엔진 (현행 트랙: CUDA + W4A16 단일)

주요 커맨드:
  llm170 infer --model <w4a16_dir> --prompt-tokens <ids> [--prompt-tokens <ids> ...]
              [--n-predict N] [--ctx N]
      greedy 추론 (JSONL {"seq","pos","token","text"}).
  llm170 serve --model <w4a16_dir> [--port N] [--ctx N] [--slots N] [--queue N]
      OpenAI/Anthropic 호환 HTTP 서버. --slots N: 동시 요청 배치 디코드 슬롯.
  llm170 w4a16-load <dir>
      W4A16(compressed-tensors int4 sym g128) 로더 완전성 검증 — 트리플·커버리지.
  llm170 w4a16-ref <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N]
      참조(CPU) greedy 토큰열 — 커널/서빙 판정 오라클·디버깅 전용.
  llm170 tokenize --model <dir> (--text <s> | --file <f> | --stdin)
      토크나이저 인코딩 [id, ...] 출력.
  llm170 help

단일 트랙(2026-10-08 — 사용자 지시): CUDA W4A16만. 기본 경로는 CUDA(가속
커널 W2/W3 개발 중 — 착륙 전 기본 실행은 안내 에러). 경로는 CUDA 고정이며,
참조(CPU) 실행은 프로브(w4a16-ref — 오라클·커널 판정 기준)로만 가능하다.
"#;

/// 모델 적재 서브커맨드 공용 인자 — main에서 1회 파싱해
/// 사전 리소스 가드와 serve/infer/vl/bench가 같은 값을 본다(이중 파싱 제거).
/// `--flag value`와 `--flag=value` 양형 지원. `rest`는 공용 플래그(값 포함)를
/// 제외한 나머지 인자 — trio 서브커맨드의 개별 플래그 파싱에 그대로 쓴다.
/// probes/check는従来대로 원본 args를 받는다(자체 파싱 보존).
pub(crate) struct ModelArgs {
    pub model: Option<String>,
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
            // --backend/--cpu 폐지(2026-10-08): 경로는 CUDA 고정.
            // 참조·디버깅은 프로브(`w4a16-ref`)로만.
            "--backend" | "--cpu" => {
                return Err(format!(
                    "{name} 폐지: 경로는 CUDA 고정 — 참조·디버깅은 w4a16-ref 프로브"
                ));
            }
            // 단일 트랙에서 제거된 플래그 — 명시 안내(무음 무시 금지).
            "--gpu-runtime" | "--mtp" | "--ple-table" | "--ple-cache" => {
                return Err(format!("{name} 미지원(단일 트랙 W4A16)"));
            }
            _ => ma.rest.push(a.to_string()),
        }
        i += 1;
    }
    Ok(ma)
}

fn main() -> ExitCode {
    let code = run_main();
    // A5: 폴백 누계 종료 출력 — 카운터는 프로세스 로컬이라
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
    // 와치독(스텔 보고·옵션 FAIL 자결).
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
    // 공용 인자 1회 파싱 — 아래 가드와 trio 디스패치가 공유.
    let ma = match parse_model_args(&args[1..]) {
        Ok(ma) => ma,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    // 사전 리소스 가드(2026-09-16): 이중 적재로 호스트가 먹통되는 사고 방지.
    // 대상 판정은 resource::guard_target 순수함수(A2/R1 추출) —
    // 서브커맨드×인자 형태 계약은 표 테스트(guard_target_cases)가 고정하고
    // 무가드 적재 프로브 폐쇄(A13)도 같은 표가 담당한다.
    if !matches!(args.first().map(String::as_str), Some("tokenize")) {
        // 가드 대상 판정은 resource::guard_target 순수함수(A2/R1 추출) —
        // 표 테이블 테스트가 계약을 고정한다(무가드 프로브 폐쇄 A13 포함).
        if let Some(gt) = resource::guard_target(
            args.first().map(String::as_str).unwrap_or(""),
            ma.model.as_deref(),
            &ma.rest,
        ) {
            // B20: 전역 적재 락 획득 → **락 후
            // 재판정**(preflight) — 동시 기동 check-then-act 레이스 직렬화.
            // 해제는 적재 완료 지점(build_slots 반환 직후 등) — 여기서 실패
            // 시엔 즉시 반납한다.
            if let Err(e) = resource::acquire_load_lock() {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = resource::preflight(&gt.path, gt.gpu) {
                resource::release_load_lock();
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // 런타임 log 패싯 메시지 노출 — stderr 간이 로거.
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
        Some("infer") => infer::cmd_infer(&ma.rest, &ma),
        Some("serve") => cmd_serve(&ma.rest, &ma),
        Some("tokenize") => cmd_tokenize(&ma),
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

/// llm170 serve --model <dir> [--port N] [--ctx N] [--slots N]
fn cmd_serve(args: &[String], ma: &ModelArgs) -> ExitCode {
    let mut port = 8080u16;
    let mut queue: Option<usize> = None;
    let mut slots: Option<usize> = None;
    let mut ctx = 4096usize;
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
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = ma.model.clone().map(PathBuf::from) else {
        return usage_err("--model required");
    };
    // 단일 트랙(2026-10-08): 수용 모델은 W4A16
    // 디렉터리 단일 — 그 외 포맷은 스니핑 단계에서 명시 에러.
    let fmt = match engine::sniff_format(&model_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let _ = fmt; // W4A16 단일(현재)
    // 토크나이저 적재 (W4A16 디렉터리)
    // 간헐 ENOPT(transient ENOENT) 재시도 — 2026-09-01 실측 회복 패턴.
    let mut tok = None;
    for i in 0..5 {
        match tokenize::Tokenizer::load(&model_path) {
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
    // QA-12: serve의 텍스트 엔드포인트(/v1/chat·completions·
    // messages)는 빈 인코딩 ids=[] 잡을 엔진에 투입해 쓰레기 스트림(또는 엔진
    // Err)을 뿜었다 — 조용한 서락 대신 기동 치명 오류. (infer·vl은 ids 인터
    // 페이스라 빈 토크나이저로도 동작 — 이곳 serve에만 적용.)
    let Some(tok) = tok else {
        eprintln!("error: tokenizer load 실패(5회 재시도) — serve 텍스트 요청에 필수");
        return ExitCode::FAILURE;
    };
    // A19: load는 어느 파트에도 토크나이저가 없으면 Ok(empty)를
    // 돌려준다 — Some(empty) 통과가 쓰레기 스트림을 뿜었다. 치명 오류로.
    if tok.is_empty() {
        eprintln!("error: 토크나이저 비음(어느 파트에도 없음) — serve 텍스트 요청에 필수");
        return ExitCode::FAILURE;
    }
    let _ = engine::TOKENIZER.set(tok);
    let req = engine::InferRequest {
        model: model_path.clone(),
        ctx,
    };
    // 모니터링 — 정적 정보 등록 + 샘플러 스레드(1Hz; 무시 가능 비용,
    // 최신 스냅샷만 유지 — 시계열 누적은 외부 폴러 몫).
    metrics::init(metrics::Info {
        instance: llm170_diag::flag::val("LLM170_INSTANCE")
            .map(str::to_string)
            .unwrap_or_default(),
        model: model_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ctx,
        n_slots: slots.unwrap_or(1),
    });
    // 라우팅: W4A16 = qwen35 CPU 경로 단일(가속은 W2 커널 이후).
    let sel = engine::BackendSel::Cpu;
    match http::serve(&format!("127.0.0.1:{port}"), req, sel, slots, queue) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `llm170 tokenize --model <dir> [--no-special] (--text <s> | --file <f> | --stdin)`
/// llama-tokenize 대응 출력 `[id, ...]` — 검증·디버깅용.
fn cmd_tokenize(ma: &ModelArgs) -> ExitCode {
    let Some(model) = ma.model.clone() else {
        eprintln!("error: --model required");
        return ExitCode::from(2);
    };
    let model_path = PathBuf::from(model);
    let tok = match tokenize::Tokenizer::load(&model_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: tokenizer load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let no_special = ma.rest.iter().any(|a| a == "--no-special");
    // A20: 위치인자만 준 사용자에게 stdin 판독 무응답처럼 보였다
    // usage 에러로. --text/--file 값 부재(마지막 인자)도
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

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}
// 마커 gpx
// 마커 ehi
