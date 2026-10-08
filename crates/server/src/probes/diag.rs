//! diag 진단 하위커맨드(diff·watchdog-selftest·fb·envcheck) + 로컬 하네스.
use std::process::ExitCode;

pub(super) fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let r: Result<String, String> = match cmd {
        "diag" => {
            // 지문 비교 — `llm170 diag diff <A> <B>`
            if args.first().map(String::as_str) == Some("diff") {
                let (pa, pb) = match (args.get(1), args.get(2)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => {
                        eprintln!("사용법: llm170 diag diff <A> <B>");
                        return Some(ExitCode::FAILURE);
                    }
                };
                match llm170_diag::fp_diff(pa, pb) {
                    Ok(r) if r.mismatch_count == 0 => Ok(format!("{r}")),
                    // QA-20: 발산은 실패로 — 종전 Ok→exit 0이 스크립트 체인에서
                    // 수치 발산을 성공으로 오판하게 했다(실패-성공 전환).
                    Ok(r) => Err(format!("{r}\nDIVERGENCE DETECTED")),
                    Err(e) => Err(e),
                }
            }
            // 와치독 자가 시험(진동 정지 후 스폰).
            else if args.first().map(String::as_str) == Some("watchdog-selftest") {
                return Some(cmd_watchdog_selftest());
            }
            // 폴백 카운터 관측. A5부터 diag 공유 원장
            // (실측 프로세스 종료 [fb] 출력과 같은 소스).
            else if args.first().map(String::as_str) == Some("fb") {
                let r = llm170_diag::fb::report();
                if r.is_empty() {
                    Ok("폴백 0건 (전 경로 GPU)".into())
                } else {
                    Ok(r)
                }
            }
            // env 스냅샷↔라이브 동치 검사.
            else if args.first().map(String::as_str) == Some("envcheck") {
                let bad = llm170_diag::flag::env_check();
                if bad.is_empty() {
                    Ok("env 동치 정상 (LLM170_ 키 전수)".into())
                } else {
                    Err(format!(
                        "env 동치 불일치 {}건:\n{}",
                        bad.len(),
                        bad.join("\n")
                    ))
                }
            } else {
                Err("diag: 하위커맨드 diff | watchdog-selftest | fb | envcheck".into())
            }
        }
        _ => return None,
    };
    Some(super::finish(r))
}

/// 진동을 멈추고 와치독 보고를 기다린다(FAIL 모드면 137).
fn cmd_watchdog_selftest() -> ExitCode {
    // QA-22: 자가시험은 실패 가능해야 한다 — 종전엔 와치독 미기동·미보고
    // 어느 쪽이든 SUCCESS 고정(자가시험 실패 불가 구조)이었다.
    if !llm170_diag::watchdog::on() {
        eprintln!("# watchdog-selftest: 와치독 미기동(LLM170_WATCHDOG 미설정?) — 시험 불가");
        return ExitCode::FAILURE;
    }
    let sec: u64 = std::env::var("LLM170_WATCHDOG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    println!("# watchdog-selftest: {sec}s 무진동 후 보고 대기");
    llm170_diag::watchdog::record_op("selftest_stall");
    // 진행 없이 대기 — 와치독이 보고한다. FAIL 모드면 여기서 종료(137).
    let before = llm170_diag::watchdog::reports();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(sec + 30);
    while std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    let n = llm170_diag::watchdog::reports() - before;
    if n == 0 {
        eprintln!("# watchdog-selftest: 보고 0건 — 와치독 스레드 미작동 의심 (FAIL)");
        return ExitCode::FAILURE;
    }
    println!("# watchdog-selftest: 와치독 보고 {n}건 확인");
    ExitCode::SUCCESS
}
