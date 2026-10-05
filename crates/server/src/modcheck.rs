//! A23(plans/129) — 모듈 무결성 인증 실행기: llm170 mod-check <module> [--record|--status].
//! 1호출 1모듈(사용자 지시 — VRAM 부족 기기 대응): 프로세스 수명 = 해당 모듈
//! 검증기 1회 서브프로세스 실행 → 골든 판정 → 종료. 판정은 corr/maxdiff·NaN만
//! (FAIL) — 속도는 참조·Δ% 표시, 대폭 역행(−30%)은 WARN(게이트 요동 방지).
//! 골든: scripts/.modcheck-golden.tsv(모듈|지표|임계값|속도참조) — --record 캡처
//! (토큰 게이트 --record 관례·산술 10a 시 자유 재캡처).
use std::process::ExitCode;

/// 모듈 레지스트리 — A22 covered_by 원장의 direct: 항목 + 소스 경로(지문용).
/// (모듈명, 검증 서브커맨드, 소스 .rs)
const MODULES: &[(&str, &str, &str)] = &[
    (
        "exl3-hip-attn",
        "exl3-hip-attn",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-gdn",
        "exl3-hip-gdn",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-gemm",
        "exl3-hip-gemm",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-gemv",
        "exl3-hip-gemv",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-nr",
        "exl3-hip-nr",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-decode",
        "exl3-hip-decode",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-batch",
        "exl3-hip-batch",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
    (
        "exl3-hip-mtp",
        "exl3-hip-mtp",
        "crates/backend-gpu/src/rawhip/exl3_hip.rs",
    ),
];

fn golden_path() -> std::path::PathBuf {
    std::path::PathBuf::from("scripts/.modcheck-golden.tsv")
}

/// 검증기 출력에서 maxdiff/corr 추출(A23 v1 — 프로브 출력 관례 maxdiff=X.XXXe−X·corr=X.XXXXXX).
fn extract_metric(out: &str) -> Option<f64> {
    // maxdiff 우선, 없으면 corr(1−corr=오차로 정규화해 동일 임계화)
    if let Some(m) = out.find("maxdiff=") {
        let rest = &out[m + 8..];
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
            .collect();
        return num.parse().ok();
    }
    if let Some(m) = out.find("corr=") {
        let rest = &out[m + 5..];
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        let c: f64 = num.parse().ok()?;
        return Some(1.0 - c); // corr → 오차로 변환(임계값은 동일 maxdiff 척도)
    }
    None
}

fn read_golden() -> Vec<(String, String, f64)> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string(golden_path()) {
        for l in t.lines().skip(1) {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() >= 3
                && let Ok(v) = c[2].parse()
            {
                out.push((c[0].into(), c[1].into(), v));
            }
        }
    }
    out
}

pub(crate) fn cmd_mod_check(args: &[String]) -> ExitCode {
    let exe = std::env::current_exe().unwrap_or_default();
    if args.first().map(String::as_str) == Some("--status") {
        println!("모듈 | 골든임계 | 상태");
        for (m, _, _) in MODULES {
            let g = read_golden().into_iter().find(|(n, _, _)| n == m);
            match g {
                Some((_, _, v)) => println!("{m} | {v:.3e} | certified"),
                None => println!("{m} | — | 미실행(--record)"),
            }
        }
        return ExitCode::SUCCESS;
    }
    let Some(module) = args.first().cloned() else {
        eprintln!("usage: llm170 mod-check <module> [--record] | --status");
        eprintln!(
            "  모듈: {}",
            MODULES
                .iter()
                .map(|(m, _, _)| *m)
                .collect::<Vec<_>>()
                .join(" · ")
        );
        return ExitCode::from(2);
    };
    let record = args.iter().any(|a| a == "--record");
    let Some((name, cmd, _src)) = MODULES.iter().find(|(m, _, _)| *m == module.as_str()) else {
        eprintln!("error: 미지원 모듈 {module}");
        return ExitCode::from(2);
    };
    // 1호출 1모듈 — 자기 자신을 서브프로세스로(부분적재·종료 시 전량 해제).
    let out = std::process::Command::new(&exe)
        .arg(cmd)
        .output()
        .map_err(|e| e.to_string());
    let Ok(out) = out else {
        eprintln!("error: 검증기 실행 실패: {cmd}");
        return ExitCode::FAILURE;
    };
    let text =
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    let Some(metric) = extract_metric(&text) else {
        eprintln!("error: {name} 검증기 출력에서 지표 추출 실패(프로브 실패?)");
        eprintln!("{}", text.lines().last().unwrap_or(""));
        return ExitCode::FAILURE;
    };
    if record {
        // 골든 캡처 — 모듈|지표|임계값|속도참조(속도 v1: 미캡처)
        let mut g = read_golden()
            .into_iter()
            .filter(|(n, _, _)| n != name)
            .collect::<Vec<_>>();
        g.push((name.to_string(), "maxdiff".into(), metric));
        g.sort_by(|a, b| a.0.cmp(&b.0));
        let mut t = String::from("module\tmetric\tthreshold\n");
        for (n, m, v) in g {
            t.push_str(&format!("{n}\t{m}\t{v:.6e}\n"));
        }
        let _ = std::fs::write(golden_path(), t);
        println!("RECORD {name}: maxdiff={metric:.3e} → 골든 갱신");
        return ExitCode::SUCCESS;
    }
    // 판정 — 골든 대비
    match read_golden().into_iter().find(|(n, _, _)| n == name) {
        Some((_, _, golden)) => {
            let ok = metric.is_finite() && metric <= golden * 1.5 + 1e-9;
            println!(
                "{name} | maxdiff {metric:.3e} ↔ 골든 {golden:.3e} | {}",
                if ok { "PASS" } else { "FAIL" }
            );
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        None => {
            eprintln!("error: {name} 골든 없음 — mod-check {name} --record 먼저");
            ExitCode::from(2)
        }
    }
}
