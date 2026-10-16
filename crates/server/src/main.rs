//! llm170 CLI.
//!
//! - infer: qwen35 CPU 참조 추론 (greedy). 토큰 id 입력 — 토크나이저는 후속 단계.

mod bench;
mod engine;
mod gpu_engine;
mod http;
mod infer;
mod json;
mod memctl;
mod metrics;
mod oai;
mod probes;
mod resource;
mod sched;
mod tokenize;
mod tree;
mod unicode_data;

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = r#"
llm170 — 순수 Rust 추론 엔진 (현행 트랙: CUDA + W4A16 단일)

주요 커맨드:
  llm170 infer --model <w4a16_dir> --prompt-tokens <ids> [--prompt-tokens <ids> ...]
              [--n-predict N] [--ctx N]
      greedy 추론 (JSONL {"seq","pos","token","text"}).
  llm170 bench --model <w4a16_dir> [--pp N[,N..]] [--tg N[,N..]] [--np N[,N..]]
              [--reps N] [--ctx N]
      llama-bench 규격 PP/TG/NP 측정(엔진 직접 — HTTP 무경유, 합성 LCG 토큰).
  llm170 serve --model <w4a16_dir> --kv-mem <총량> [--port N] [--queue N]
              [--ple ssd|ram|vram] [--spec auto|mtp|ngram|off] [--moe auto|resident|streaming|hybrid]
      OpenAI/Anthropic 호환 HTTP 서버. KV 총량 = --kv-mem(필수·무 auto — 권고는 `llm170 plan`).
      요청 상한 = min(모델 최대, 풀) — 모델 ctx 초과분은 출력 클램프(vLLM식·기동 로그에 기록),
      풀 초과는 거부, 잔여 부족은 대기(드롭 없음).
  llm170 w4a16-load <dir>
      W4A16(compressed-tensors int4 sym g128) 로더 완전성 검증 — 트리플·커버리지.
  llm170 w4a16-ref <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N]
      참조(CPU) greedy 토큰열 — 커널/서빙 판정 오라클·디버깅 전용.
  llm170 w4a16-eval ppl <dir> --corpus <txt> [--ctx N] [--limit N] [--out dump.tsv] [--json]
  llm170 w4a16-eval agree <a.tsv> <b.tsv>
      통계 eval — PPL·teacher-forced argmax 일치율(연산 경로 편차 계측기).
  llm170 tokenize --model <dir> (--text <s> | --file <f> | --stdin) [--no-special] [--chat]
      토크나이저 인코딩 [id, ...] 출력. --chat: messages JSON을 아치별
      최소 턴 렌더로 변환해 토큰화(렌더·BOS·EOS 검증).
  llm170 plan --model <w4a16_dir> [--cards N] [--vram-gb X] [--host-gb Y] [--ctx N] [--ssd-gb Z]
      배치 플래너 드라이런(무적재) — 매니페스트 범주 분류 + 사다리 배치안(§2.M).
  llm170 help

단일 트랙(2026-10-08 — 사용자 지시): CUDA W4A16만. 경로는 CUDA 고정이며,
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
    if !matches!(
        args.first().map(String::as_str),
        Some("tokenize") | Some("plan")
    ) {
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
    // OOM 킬러 지정 희생자 (실측 2026-09-01): 초대형 mmap(total-vm 150GB+)이
    // badness 최상위로 뽑혀 런·세션이 함께 죽는다. 스스로 adj=1000을 걸어
    // 런만 희생되게 한다 (무권한으로는 보호 불가 — 우선순위 이동만 가능).
    // LLM170_NO_OOM_ADJ=1이면 해제.
    if std::env::var_os("LLM170_NO_OOM_ADJ").is_none() {
        let _ = std::fs::write("/proc/self/oom_score_adj", b"1000");
    }
    // [H 2026-10-09] 로그 레벨 개방 — 기본 Error(종전 고정값 동일),
    // LLM170_LOG=off|error|warn|info|debug|trace로 상향(진단 로그 확인 수단).
    // 주의: flag 스냅샷을 FRAME 기본값 set_var 이후로 미룬다 — 여기서 처음
    // flag를 읽으면 스냅샷이 FRAME 기본값을 못 보고 envcheck가 불일치한다.
    let level = match llm170_diag::flag::val("LLM170_LOG").map(str::to_ascii_lowercase) {
        Some(ref v) if v == "off" => log::LevelFilter::Off,
        Some(ref v) if v == "warn" => log::LevelFilter::Warn,
        Some(ref v) if v == "info" => log::LevelFilter::Info,
        Some(ref v) if v == "debug" => log::LevelFilter::Debug,
        Some(ref v) if v == "trace" => log::LevelFilter::Trace,
        Some(ref v) if v == "error" => log::LevelFilter::Error,
        Some(v) => {
            eprintln!("warning: LLM170_LOG={v} 미지 레벨 — error 유지");
            log::LevelFilter::Error
        }
        None => log::LevelFilter::Error,
    };
    log::set_max_level(level);
    if let Some(cmd) = args.first().map(String::as_str)
        && let Some(code) = probes::run(cmd, &args[1..])
    {
        return code;
    }
    match args.first().map(String::as_str) {
        Some("bench") => bench::cmd_bench(&ma.rest, &ma),
        Some("infer") => infer::cmd_infer(&ma.rest, &ma),
        Some("serve") => cmd_serve(&ma.rest, &ma),
        Some("tokenize") => cmd_tokenize(&ma),
        Some("plan") => cmd_plan(&ma.rest, &ma),
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

/// llm170 serve --model <dir> --kv-mem <총량> [--port N] [--queue N] [--ple ..] [--spec ..] [--moe ..]
fn cmd_serve(args: &[String], ma: &ModelArgs) -> ExitCode {
    let mut port = 8080u16;
    let mut queue: Option<usize> = None;
    // [K6-3 2026-10-13] --slots 폐지 — 동시성 = min(문맥 수, 풀 입장).
    // 문맥 수는 내부 상수(상태 메모리 = 문맥×gdn_state/슬롯 — 예산 회계 반영).
    const SERVE_CONTEXTS: usize = 4;
    // [2026-10-16] serve 입력 = --kv-mem(필수·무 auto) — --ctx(serve)는 폐지.
    // 배분 절충(KV vs 전문가 vs PLE)은 운영자 명시 결정, 플래너는 검증·권고.
    let mut kv_mem: Option<u64> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => match it.next().and_then(|v| v.parse().ok()) {
                Some(p) => port = p,
                None => return usage_err("--port requires a number"),
            },
            "--kv-mem" => match it.next().and_then(|v| parse_mem_bytes(v)) {
                Some(b) => kv_mem = Some(b),
                None => {
                    return usage_err(
                        "--kv-mem requires a size (e.g. 24G / 4096M / bytes — auto 없음)",
                    );
                }
            },
            "--ctx" => {
                return usage_err(
                    "serve: --ctx는 폐지 — KV 총량은 --kv-mem <총량>으로 지정(bench --ctx는 유지)",
                );
            }
            "--queue" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(q) => queue = Some(q.max(1)),
                None => return usage_err("--queue requires a number"),
            },
            // PLE 티어 오버라이드(기본 ssd) — CLI > env 진단.
            "--ple" => match it.next().map(|v| crate::engine::parse_ple_policy(v)) {
                Some(Some(p)) => {
                    crate::engine::PLE_POLICY.store(p as u64, std::sync::atomic::Ordering::Relaxed)
                }
                _ => return usage_err("--ple requires ssd|ram|vram"),
            },
            // [2.J 2026-10-13] 런치 정책 2종 (CLI > env 진단).
            "--spec" => match it.next().map(|v| crate::engine::parse_spec_policy(v)) {
                Some(Some(p)) => {
                    crate::engine::SPEC_POLICY.store(p as u64, std::sync::atomic::Ordering::Relaxed)
                }
                _ => return usage_err("--spec requires auto|mtp|ngram|off"),
            },
            "--moe" => match it.next().map(|v| crate::engine::parse_moe_policy(v)) {
                Some(Some(p)) => {
                    crate::engine::MOE_POLICY.store(p as u64, std::sync::atomic::Ordering::Relaxed)
                }
                _ => return usage_err("--moe requires auto|resident|streaming|hybrid"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = ma.model.clone().map(PathBuf::from) else {
        return usage_err("--model required");
    };
    let Some(kv_mem) = kv_mem else {
        return usage_err(
            "--kv-mem <총량> required — 배분 절충은 운영자 명시 결정(무 auto; 권고 = `llm170 plan`)",
        );
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
    // [2026-10-16] kv-mem 검증·파생 — 요청 상한 = min(모델 최대, 풀 총량).
    let (model_max_ctx, cap, kv_budget, kv_reserve, plan_txt, kv_vram_cap) = match model_ctx_info(
        &model_path,
    ) {
        Ok(ci) => {
            let model_max_ctx = ci.ctx;
            let mut plan_txt = String::new();
            // [2026-10-16] 2계층 분할 결과(VRAM 상한) — 아래 검증에서 확정.
            let mut vram_cap_opt: Option<u64> = None;
            // [2026-10-16] kv-mem 파생 — 풀 = kv-mem/KV-바이트-토큰(512 블록
            // 정렬), 요청 상한 = min(모델 최대, 풀). 클램프 없음(초과 = 거부).
            let pool_tokens = kv_layout(kv_mem, ci.kv_bytes_per_token, ci.ctx);
            if pool_tokens < 512 {
                eprintln!(
                    "# kv-mem 가드 거부: {} ≈ {pool_tokens} tokens < 512(블록 1개) — KV {:.1}KiB/토큰",
                    fmt_mem(kv_mem),
                    ci.kv_bytes_per_token.max(1) as f64 / 1024.0
                );
                return ExitCode::FAILURE;
            }
            if let Some((free, total)) = llm170_backend_gpu::cuda_mem_free() {
                // [예약 재산정 2026-10-14] 워크스페이스·스크래치 + 스펙 VRAM
                // (WY·링 스냅샷 ≈0.2GiB + MTP 아레나 등 — 체크포인트 링은
                // 핀드 호스트 RAM 이관으로 VRAM 몫 아님). 종전 2.5GiB는
                // 체크포인트(1.26GiB)를 이 예약의 여유로 먹던 시절 값.
                // [dsave 대체 2026-10-14] 스펙 정밀 스냅샷(dsave — 27B/4슬롯
                // 315.0MiB)이 프로브 전용 지연 할당으로 빠지고 dring0
                // (5.6MiB)이 대신 들어와 순감 309.4MiB — 그 몫만 환원.
                const RESERVE: u64 = resource::VRAM_RESERVE;
                // 폴백 상한(매니페스트 판독 불가 시): 가중치 = dense 전량(디렉터리).
                let wts = if ci.dense { dir_bytes(&model_path) } else { 0 };
                // [K6-3] 문맥 추가분 = (문맥−1)×GDN 상태 — KV 상한에서 차감.
                let ctx_state = (SERVE_CONTEXTS as u64 - 1) * ci.gdn_state_bytes_per_slot;
                let mut max_kv = free
                    .saturating_sub(wts)
                    .saturating_sub(RESERVE)
                    .saturating_sub(ctx_state);
                // [2.M 2026-10-13] 배치 플래너 = 상한·권고의 단일 산출원(매니페스트
                // 판독 시). dense 범주(실상주 — embed는 호스트 mmap)로 가중치를
                // 계상해 embed 바이트를 KV로 회수한다(구 공식은 dir 전량 — 27B에서
                // 2.4GiB 과소). 판독 실패·거부 시엔 폴백 공식 유지.
                match llm170_core::plan::Manifest::from_dir(&model_path) {
                    Ok(man) => {
                        let hw = llm170_core::plan::Hardware {
                            vram_total: total,
                            vram_free: Some(free),
                            host_ram: resource::host_mem_available().unwrap_or(0),
                            ssd: None,
                            reserve: RESERVE,
                            cards: 1,
                        };
                        let wl = llm170_core::plan::Workload {
                            model_max_ctx: ci.ctx,
                            want_ctx: ci.ctx,
                            kv_bytes_per_token: ci.kv_bytes_per_token,
                            state_bytes_per_ctx: ci.gdn_state_bytes_per_slot,
                            contexts: SERVE_CONTEXTS,
                            // [2026-10-16] embed 상주 정책(실행 경로) — qwen 계열
                            // true(VRAM 업로드), gemma4 false(호스트 embed_row).
                            embed_resident: embed_resident(&model_path),
                        };
                        let p = llm170_core::plan::plan(&man, &hw, &wl);
                        plan_txt = p.summary();
                        if p.fits() {
                            max_kv = p.kv_budget;
                        }
                    }
                    Err(e) => eprintln!("# plan 건너뜀: {e}"),
                }
                // [2026-10-16] 2계층 검증 — VRAM부 상한 초과분은 **호스트 확장**
                // (요청 우선 — 거부하지 않는다). 호스트부는 가용 RAM 검증.
                let (vram_part, host_part) = kv_split(kv_mem, max_kv);
                if host_part > 0 {
                    const HOST_RESERVE: u64 = 8 << 30; // 데스크톱 여유(운영 조정 대상)
                    let avail = resource::host_mem_available().unwrap_or(0);
                    if host_part + HOST_RESERVE > avail {
                        eprintln!(
                            "# kv-mem 가드 거부: 호스트부 {} > 가용 RAM {} − 예약 {} (VRAM부 {}·상한 {})",
                            fmt_mem(host_part),
                            fmt_mem(avail),
                            fmt_mem(HOST_RESERVE),
                            fmt_mem(vram_part),
                            fmt_mem(max_kv)
                        );
                        return ExitCode::FAILURE;
                    }
                    eprintln!(
                        "# kv-mem 계층: VRAM {} + 호스트 {} (초과분 = 호스트 아레나 — 오프로드율은 /stats)",
                        fmt_mem(vram_part),
                        fmt_mem(host_part)
                    );
                    // 핀드 = 페이지 고정(스왑 회수 불가) — 호스트부만큼 상시 점유.
                    eprintln!(
                        "#   주의: 호스트부는 핀드(스왑 회수 불가) — 데스크톱 압박 시 kv-mem 축소 권고"
                    );
                }
                vram_cap_opt = Some(vram_part);
                // 권고 = 플래너/폴백 상한(운영자 조정 참고 — 강제 아님).
                eprintln!("# kv-mem 권고(플래너 상한): {}", fmt_mem(max_kv));
            } else {
                eprintln!("# kv-mem 검증 건너뜀: VRAM 조회 불가(가드가 거부할 것)");
            }
            eprintln!(
                "# kv-mem: {} = {pool_tokens} tokens(512 블록) · 요청 상한 = min(model {model_max_ctx}, pool {pool_tokens}) = {pool_tokens}",
                fmt_mem(kv_mem)
            );
            eprintln!(
                "# policy: spec={}(CLI) moe={}(CLI) ple={}(CLI/기본) — env 진단(LLM170_SPEC_DRAFT_SOURCE/MOE_RESIDENT)보다 우선",
                crate::engine::spec_name(),
                crate::engine::moe_name(),
                crate::engine::ple_name()
            );
            if !plan_txt.is_empty() {
                eprintln!("# plan: {plan_txt}");
            }
            (
                model_max_ctx,
                pool_tokens,
                // [2.M 사다리] KV 예산·예약(전문가 상주 판정) = 운영자 명시 몫.
                Some(kv_mem),
                Some(kv_mem),
                plan_txt,
                vram_cap_opt,
            )
        }
        Err(e) => {
            eprintln!("error: kv-mem 검증 불가 — config 판독 실패: {e}");
            return ExitCode::FAILURE;
        }
    };
    let req = engine::InferRequest {
        model: model_path.clone(),
        ctx: cap,
        model_max_ctx,
        kv_budget,
        kv_reserve,
        kv_vram_cap,
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
        ctx: cap,
        model_max_ctx,
        plan: plan_txt,
        n_slots: SERVE_CONTEXTS,
        model_bytes: dir_bytes(&model_path),
        kv_mem,
        pool_cap_tokens: cap,
    });
    eprintln!(
        "# serve: 문맥 {SERVE_CONTEXTS}(내부 상수 — --slots 폐지) · 동시성 = min(문맥, 풀 입장)"
    );
    match http::serve(&format!("127.0.0.1:{port}"), req, SERVE_CONTEXTS, queue) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// llm170 plan --model <dir> [--cards N] [--vram-gb X] [--host-gb Y] [--ctx N] [--ssd-gb Z]
/// 배치 플래너 드라이런(무적재 — safetensors 헤더 + config만 판독): 범주 분류 +
/// 사다리 배치안. VRAM/호스트 미지정 시 현재 기기(단일 카드·MemTotal) 기준(§2.M).
fn cmd_plan(args: &[String], ma: &ModelArgs) -> ExitCode {
    let Some(model_path) = ma.model.clone().map(std::path::PathBuf::from) else {
        return usage_err("--model required");
    };
    let mut cards = 1usize;
    let mut vram_gb: Option<f64> = None;
    let mut host_gb: Option<f64> = None;
    let mut ssd_gb: Option<f64> = None;
    let mut ctx_opt: Option<usize> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cards" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(c) if c >= 1 => cards = c,
                _ => return usage_err("--cards requires a number >= 1"),
            },
            "--vram-gb" => match it.next().and_then(|v| v.parse::<f64>().ok()) {
                Some(v) if v > 0.0 => vram_gb = Some(v),
                _ => return usage_err("--vram-gb requires a positive number"),
            },
            "--host-gb" => match it.next().and_then(|v| v.parse::<f64>().ok()) {
                Some(v) if v > 0.0 => host_gb = Some(v),
                _ => return usage_err("--host-gb requires a positive number"),
            },
            "--ssd-gb" => match it.next().and_then(|v| v.parse::<f64>().ok()) {
                Some(v) if v > 0.0 => ssd_gb = Some(v),
                _ => return usage_err("--ssd-gb requires a positive number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(c) => ctx_opt = Some(c),
                None => return usage_err("--ctx requires a number"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }
    let to_gib = |g: f64| (g * (1u64 << 30) as f64) as u64;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let man = match llm170_core::plan::Manifest::from_dir(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    // 워크로드 — config 파생(ctx·KV/토큰·상태). 실패 시 표준 폴백(무적재 원칙).
    let (max_ctx, kv_bpt, state_bpc) = match model_ctx_info(&model_path) {
        Ok(ci) => (ci.ctx, ci.kv_bytes_per_token, ci.gdn_state_bytes_per_slot),
        Err(e) => {
            eprintln!("# plan: config 파생 실패({e}) — 폴백(256K·33KiB/tok·151MiB/ctx)");
            (262_144, 33 * 1024, 151 * (1 << 20))
        }
    };
    // 하드웨어 — 미지정은 실측(단일 카드 조회·MemTotal).
    let dev_total = llm170_backend_gpu::cuda_mem_free()
        .map(|(_, t)| t)
        .unwrap_or(0);
    let per_card = vram_gb.map(to_gib).unwrap_or(dev_total);
    if per_card == 0 {
        eprintln!("error: VRAM 조회 불가 — --vram-gb로 카드당 용량을 지정하라");
        return ExitCode::FAILURE;
    }
    let hw = llm170_core::plan::Hardware {
        vram_total: per_card.saturating_mul(cards as u64),
        vram_free: None,
        host_ram: host_gb
            .map(to_gib)
            .unwrap_or_else(|| resource::host_mem_total().unwrap_or(0)),
        ssd: ssd_gb.map(to_gib),
        reserve: 5 * (1u64 << 30) / 2,
        cards,
    };
    // 문맥 수 — serve 내부 상수(동시성 = min(문맥, 입장)).
    const SERVE_CONTEXTS: usize = 4;
    let wl = llm170_core::plan::Workload {
        model_max_ctx: max_ctx,
        want_ctx: ctx_opt.unwrap_or(max_ctx),
        kv_bytes_per_token: kv_bpt,
        state_bytes_per_ctx: state_bpc,
        contexts: SERVE_CONTEXTS,
        // [2026-10-16] embed 상주 정책(실행 경로 — 아치별).
        embed_resident: embed_resident(&model_path),
    };
    let p = llm170_core::plan::plan(&man, &hw, &wl);
    let s = p.sizes;
    println!(
        "tensors={} dense={:.1}G experts={:.1}G row_sparse={:.1}G embed={:.1}G total={:.1}G",
        man.len(),
        gib(s.dense),
        gib(s.experts),
        gib(s.row_sparse),
        gib(s.embed),
        gib(s.total())
    );
    println!(
        "vram={:.1}G × {cards} = {:.1}G · host={:.1}G · reserve={:.1}G · ctx 요건={} (KV {:.0}B/tok·상태 {:.0}M/ctx)",
        gib(per_card),
        gib(p.vram),
        gib(hw.host_ram),
        gib(hw.reserve),
        wl.want_ctx,
        wl.kv_bytes_per_token as f64,
        wl.state_bytes_per_ctx as f64 / (1u64 << 20) as f64
    );
    println!("plan: {}", p.summary());
    // [2026-10-16] serve 권고 — 플래너 KV 예산이 곧 kv-mem 권고값(운영자 명시).
    if p.fits() {
        println!(
            "serve 권고: --kv-mem {} — 풀 ctx {}/{} ({}) · experts={} · 호스트부 여유 최대 {}",
            fmt_mem(p.kv_budget),
            p.effective_ctx,
            p.goal_ctx,
            if p.ctx_capped {
                "풀 클램프"
            } else {
                "전량"
            },
            p.experts_tier.name(),
            fmt_mem(p.kv_host_budget),
        );
    } else {
        println!("serve 권고: 없음(수용 거부) — 상한/호스트 재배분 필요");
    }
    // [2.M 샤드 플랜] 다중 카드 요청 시 — 레이어 파이프라인 배분·수용 판정.
    if cards > 1 {
        let split = man.layer_split();
        let sp = llm170_core::plan::shard_plan(&split.layers, &split, &hw, &wl);
        println!(
            "shard: cards={} blocks={:?} fits={} · 최악 카드={} need {:.1}G ≤ cap {:.1}G",
            sp.cards,
            sp.blocks,
            sp.fits,
            sp.worst_card,
            gib(sp.per_card_need[sp.worst_card]),
            gib(sp.per_card_cap)
        );
        for (c, ((s, e), need)) in sp.blocks.iter().zip(&sp.per_card_need).enumerate() {
            println!(
                "  card{c}: layers {s}..{e} · w {:.1}G · kv {:.1}G · state {:.1}G · need {:.1}G",
                gib(sp.per_card_weights[c]),
                gib(sp.per_card_kv[c]),
                gib(sp.per_card_state[c]),
                gib(*need)
            );
        }
    }
    ExitCode::SUCCESS
}

/// 디렉터리 파일 크기 합(모니터링 표시용 — 실패는 0).
fn dir_bytes(p: &std::path::Path) -> u64 {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// [2026-10-16] 크기 표기 — 로그용(바이너리 단위, 1자리).
fn fmt_mem(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1}G", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1}M", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1}K", b as f64 / 1024.0)
    } else {
        format!("{b}B")
    }
}

/// [2026-10-16] `--kv-mem` 파싱 — 접미 K/M/G/T(2진) 또는 KB/MB/GB/TB·
/// KiB/MiB/GiB/TiB(대소문자 무시), 단위 없으면 바이트. 0·불명·소수점 = None
/// (무 auto — 운영자 명시값만).
fn parse_mem_bytes(s: &str) -> Option<u64> {
    let low = s.trim().to_ascii_lowercase();
    if low.is_empty() {
        return None;
    }
    for (sfx, mult) in [
        ("kib", 1u64 << 10),
        ("kb", 1u64 << 10),
        ("k", 1u64 << 10),
        ("mib", 1u64 << 20),
        ("mb", 1u64 << 20),
        ("m", 1u64 << 20),
        ("gib", 1u64 << 30),
        ("gb", 1u64 << 30),
        ("g", 1u64 << 30),
        ("tib", 1u64 << 40),
        ("tb", 1u64 << 40),
        ("t", 1u64 << 40),
        ("b", 1),
    ] {
        if let Some(n) = low.strip_suffix(sfx) {
            let v: u64 = n.parse().ok()?;
            return v.checked_mul(mult).filter(|b| *b > 0);
        }
    }
    let v: u64 = low.parse().ok()?;
    (v > 0).then_some(v)
}

/// [2026-10-16] kv-mem → 실효 요청 상한(토큰) = min(모델 최대,
/// kv-mem/KV-토큰), 512 블록 정렬. 값 자체가 풀 총량(초과 요청은 거부).
fn kv_layout(kv_mem: u64, kv_bytes_per_token: u64, model_max: usize) -> usize {
    let tokens = (kv_mem / kv_bytes_per_token.max(1)) as usize;
    (tokens.min(model_max)) / 512 * 512
}

/// [2026-10-16] kv-mem 2계층 분할 — (VRAM부, 호스트부). VRAM부 = min(총량,
/// VRAM 상한 = 여유 VRAM − 가중치·예약·상태), 초과분은 호스트 확장(요청 우선).
fn kv_split(kv_mem: u64, max_vram_kv: u64) -> (u64, u64) {
    let vram = kv_mem.min(max_vram_kv);
    (vram, kv_mem - vram)
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
    // [P1 gemma4 2026-10-14] --chat: messages JSON을 아치별 최소 턴 렌더로
    // 바꿔 토큰화 — 렌더·BOS·EOS 레지스트리 검증 경로.
    if ma.rest.iter().any(|a| a == "--chat") {
        let arch = llm170_core::qwen35::bind::dir_arch(&model_path).unwrap_or_default();
        let Some(spec) = crate::oai::chat_spec(&arch) else {
            eprintln!("error: 아치 챗 스펙 없음({arch})");
            return ExitCode::FAILURE;
        };
        let rendered = (spec.render)(&text);
        let ids = tok.encode_opts(&rendered, true);
        let csv = |v: &[u32]| {
            v.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut dec = Vec::new();
        for id in &ids {
            tok.piece_bytes_into(*id, &mut dec);
        }
        println!("arch: {arch}");
        println!("render: {rendered:?}");
        println!("ids: [{}]", csv(&ids));
        println!("decode: {:?}", String::from_utf8_lossy(&dec));
        println!("eos: [{}]", csv(spec.eos));
        return ExitCode::SUCCESS;
    }
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

/// [2026-10-16] embed 상주 정책(엔진 실측) — qwen 계열 로더는 F16/BF16 embed를
/// VRAM에 올린다(prefill gather 가속 — upload_embed). gemma4는 호스트
/// embed_row 유지(청크 체인). dtype만으로는 구분 불가(둘 다 BF16)이므로 실행
/// 경로(아치 로더)가 플래너에 정책을 넘긴다 — 플래너 회계 실정합.
/// [검토 2026-10-16] qwen4exp(FN)는 현재 "gemma4 아님 = 상주"로 계상된다 —
/// FN 로더 브링업에서 embed 실행 경로(PLE n-gram 임베딩 = 대형·호스트 예상)를
/// 확정한 뒤 이 판정을 재검할 것. 미검 상태로 두면 플래너가 VRAM을 과대계상해
/// 정당한 구성을 오거부할 수 있다(가중치 로드 불가라 현재 실해 없음).
fn embed_resident(dir: &std::path::Path) -> bool {
    let arch = llm170_core::qwen35::bind::dir_arch(dir).unwrap_or_default();
    !llm170_core::gemma4::bind::arch_ok(&arch)
}

/// [P3 gemma4 2026-10-14] 아치별 계획 파생 — ctx_info 레지스트리 디스패치.
fn model_ctx_info(p: &std::path::Path) -> Result<llm170_core::qwen35::CtxInfo, String> {
    let arch = llm170_core::qwen35::bind::dir_arch(p).unwrap_or_default();
    if llm170_core::gemma4::bind::arch_ok(&arch) {
        llm170_core::gemma4::ctx_info(p)
    } else if llm170_core::qwen4exp::arch_ok(&arch) {
        // [P7 2026-10-15] FN(qwen4_exp) 골격 — ctx 파생·가드 입력만 등록
        // (엔진 브링업은 §2.M 트랙 — 가중치 로드 불가 상태 그대로).
        llm170_core::qwen4exp::ctx_info(p)
    } else {
        llm170_core::qwen35::ctx_info(p)
    }
}

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// --kv-mem 파싱 — 단위 4계열·바이트·불명·0·소수점.
    #[test]
    fn parse_mem_bytes_units() {
        assert_eq!(parse_mem_bytes("24G"), Some(24 << 30));
        assert_eq!(parse_mem_bytes("24g"), Some(24 << 30));
        assert_eq!(parse_mem_bytes("24GiB"), Some(24 << 30));
        assert_eq!(parse_mem_bytes("4096M"), Some(4096 << 20));
        assert_eq!(parse_mem_bytes("4096MiB"), Some(4096 << 20));
        assert_eq!(parse_mem_bytes("512K"), Some(512 << 10));
        assert_eq!(parse_mem_bytes("1T"), Some(1 << 40));
        assert_eq!(parse_mem_bytes("12345"), Some(12345));
        assert_eq!(parse_mem_bytes("4096B"), Some(4096));
        // 거부: auto·0·소수·불명·공백.
        assert_eq!(parse_mem_bytes("auto"), None);
        assert_eq!(parse_mem_bytes("0"), None);
        assert_eq!(parse_mem_bytes("0G"), None);
        assert_eq!(parse_mem_bytes("4.5G"), None);
        assert_eq!(parse_mem_bytes("G"), None);
        assert_eq!(parse_mem_bytes(""), None);
        assert_eq!(parse_mem_bytes("12X"), None);
    }

    /// [2026-10-16] kv-mem → (VRAM부, 호스트부) 2계층 분할 — VRAM 상한
    /// 이하는 전부 VRAM, 초과분만 호스트. 총량 이하/경계 보존.
    #[test]
    fn kv_split_tiers() {
        assert_eq!(
            kv_split(4 << 30, 6 << 30),
            (4 << 30, 0),
            "VRAM 이하 = 전량 VRAM"
        );
        assert_eq!(
            kv_split(6 << 30, 4 << 30),
            (4 << 30, 2 << 30),
            "초과분 = 호스트"
        );
        assert_eq!(kv_split(4 << 30, 4 << 30), (4 << 30, 0), "경계 = 호스트 0");
        assert_eq!(kv_split(1, 0), (0, 1), "VRAM 상한 0 = 전량 호스트");
    }

    /// kv_layout — 512 정렬·모델 클램프·최소(블록 미만).
    #[test]
    fn kv_layout_align_and_clamp() {
        // 27B 실측급: KV 33280B/토큰.
        let bpt = 33_280u64;
        // 1 GiB → 32263 tokens → 512 정렬 31744.
        let t = kv_layout(1 << 30, bpt, 262_144);
        assert_eq!(t, ((1u64 << 30) / bpt) as usize / 512 * 512);
        assert_eq!(t % 512, 0);
        // 모델 최대 클램프(풀 262144 초과분은 모델 상한).
        assert_eq!(kv_layout(100 << 30, bpt, 16_384), 16_384);
        // 최소 미만 = 0(호출부가 512 가드로 거부).
        assert_eq!(kv_layout(1000, bpt, 262_144), 0);
        // kv_bpt 0(비정상 config) — 1로 방어(패닉 금지).
        assert_eq!(kv_layout(1 << 20, 0, 262_144) % 512, 0);
    }

    /// fmt_mem — 로그 표기 경계.
    #[test]
    fn fmt_mem_units() {
        assert_eq!(fmt_mem(24 << 30), "24.0G");
        assert_eq!(fmt_mem(128 << 20), "128.0M");
        assert_eq!(fmt_mem(512 << 10), "512.0K");
        assert_eq!(fmt_mem(999), "999B");
    }
}
