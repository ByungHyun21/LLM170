//! diag 진단 하위커맨드(diff·chunk-check·va-lookup·fb·envcheck·ckdiff…) + 그 로컬 하네스 (plans/129 R2③ — probes/ 분리, arm 본문 무변경 이동).
use std::process::ExitCode;

pub(super) fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let r: Result<String, String> = match cmd {
        "diag" => {
            // plans/82: 지문 비교 — `llm170 diag diff <A> <B>`
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
            // plans/87 §1 — tsv 원장에서 폴트 주소 매칭.
            else if args.first().map(String::as_str) == Some("va-lookup") {
                let Some(tsv) = args.get(1).cloned() else {
                    eprintln!("error: va-lookup <tsv> <addr-hex>");
                    return Some(ExitCode::FAILURE);
                };
                let Some(addr) = args.get(2).cloned() else {
                    eprintln!("error: va-lookup <tsv> <addr-hex>");
                    return Some(ExitCode::FAILURE);
                };
                return Some(cmd_va_lookup(&tsv, &addr));
            }
            // plans/87 §2 — 와치독 자가 시험(진동 정지 후 스폰).
            else if args.first().map(String::as_str) == Some("watchdog-selftest") {
                return Some(cmd_watchdog_selftest());
            }
            // plans/107 W8 — 폴백 카운터 관측. A5(plans/129)부터 diag 공유 원장
            // (실측 프로세스 종료 [fb] 출력과 같은 소스).
            else if args.first().map(String::as_str) == Some("fb") {
                let r = llm170_diag::fb::report();
                if r.is_empty() {
                    Ok("폴백 0건 (전 경로 GPU)".into())
                } else {
                    Ok(r)
                }
            }
            // plans/108 P1 — env 스냅샷↔라이브 동치 검사.
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
            }
            // plans/87 §5 — [npck] 로그 크로스 diff.
            else if args.first().map(String::as_str) == Some("ckdiff") {
                let Some(a) = args.get(1).cloned() else {
                    eprintln!("error: ckdiff <A.log> <B.log> [rel]");
                    return Some(ExitCode::FAILURE);
                };
                let Some(b) = args.get(2).cloned() else {
                    eprintln!("error: ckdiff <A.log> <B.log> [rel]");
                    return Some(ExitCode::FAILURE);
                };
                let rel = args
                    .get(3)
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(0.02);
                return Some(cmd_ckdiff(&a, &b, rel));
            } else {
                Err("diag: 하위커맨드 diff | va-lookup | watchdog-selftest | ckdiff | fb | envcheck".into())
            }
        }
        _ => return None,
    };
    Some(super::finish(r))
}

fn cmd_va_lookup(tsv: &str, addr: &str) -> ExitCode {
    // BDA는 canonical 부호확장(0xffff8001..), RADV 폴트는 48비트 절단형
    // (0x8001..) — 하위 48비트로 정규화해 비교한다(실측, plans/87 §1).
    const M: u64 = 0x0000_ffff_ffff_ffff;
    let Ok(a) = u64::from_str_radix(addr.trim_start_matches("0x"), 16) else {
        eprintln!("error: 주소 파싱 실패: {addr}");
        return ExitCode::FAILURE;
    };
    let a = a & M;
    let Ok(text) = std::fs::read_to_string(tsv) else {
        eprintln!("error: tsv 없음: {tsv}");
        return ExitCode::FAILURE;
    };
    let mut hits = 0usize;
    for (ln, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 4 {
            continue;
        }
        let (Ok(lo), Ok(hi)) = (
            u64::from_str_radix(f[2].trim_start_matches("0x"), 16),
            u64::from_str_radix(f[3].trim_start_matches("0x"), 16),
        ) else {
            continue;
        };
        let (lo, hi) = (lo & M, hi & M);
        if a >= lo && a < hi {
            println!(
                "HIT line {} site={} bytes={} range={:#x}..{:#x} seq={}",
                ln + 1,
                f[0],
                f[1],
                lo,
                hi,
                f.get(4).unwrap_or(&"?")
            );
            hits += 1;
        }
    }
    if hits == 0 {
        // OOB 접근은 대상 버퍼 **끝 너머**에 착지한다(원 사건: 921600B
        // 1-전문가 버퍼 +929792) — ±4MB 근접 항목을 인접 보고한다.
        const WINDOW: i64 = 4 << 20;
        let mut cands: Vec<(i64, i64, &str)> = Vec::new(); // (d, lo, line)
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 {
                continue;
            }
            let Ok(lo) = u64::from_str_radix(f[2].trim_start_matches("0x"), 16) else {
                continue;
            };
            let lo = lo & M;
            let d = a as i64 - lo as i64;
            if d.abs() <= WINDOW {
                cands.push((d, lo as i64, line));
            }
        }
        // OOB는 버퍼 베이스에서 앞으로 걷는다 — 양수(이후) 방향 우선, 그 후 근접.
        cands.sort_by_key(|(d, _, _)| (*d < 0, d.abs()));
        let best = cands.first().map(|(d, _, line)| (*d, *line));
        match best {
            Some((d, line)) => {
                let f: Vec<&str> = line.split('\t').collect();
                println!(
                    "NEAR addr={a:#x} — {} {}B 범위 {:#x}..{:#x} 에서 {d:+} 바이트 (OOB 착지로 추정)",
                    f[0],
                    f[1],
                    u64::from_str_radix(f[2].trim_start_matches("0x"), 16).unwrap_or(0) & M,
                    u64::from_str_radix(f[3].trim_start_matches("0x"), 16).unwrap_or(0) & M
                );
                ExitCode::SUCCESS
            }
            None => {
                println!("NO-HIT addr={a:#x} ({} entries)", text.lines().count());
                ExitCode::FAILURE
            }
        }
    } else {
        ExitCode::SUCCESS
    }
}

/// plans/87 §2 — 진동을 멈추고 와치독 보고를 기다린다(FAIL 모드면 137).
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

/// plans/87 §5 — [npck] 체크섬 로그 교차 diff: (tag, t, 등장순) 정렬 맞춤,
/// 첫 상이 태그 + 상위 상이 표. 무상이 exit 0.
fn cmd_ckdiff(a_path: &str, b_path: &str, rel_lim: f64) -> ExitCode {
    let parse = |p: &str| -> Result<Vec<(String, usize, f64)>, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        let mut out = Vec::new();
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("[npck] ") else {
                continue;
            };
            // {tag} t={t} sum={s} v0=.. mid0=.. last0=..
            let mut it = rest.split_whitespace();
            let Some(tag) = it.next() else { continue };
            let Some(t) = it
                .next()
                .and_then(|s| s.strip_prefix("t="))
                .and_then(|v| v.parse().ok())
            else {
                continue;
            };
            let Some(sum) = it
                .next()
                .and_then(|s| s.strip_prefix("sum="))
                .and_then(|v| v.parse().ok())
            else {
                continue;
            };
            out.push((tag.to_string(), t, sum));
        }
        Ok(out)
    };
    let (a, b) = match (parse(a_path), parse(b_path)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if a.is_empty() || b.is_empty() {
        eprintln!("error: 한쪽 로그에 [npck] 행 없음");
        return ExitCode::FAILURE;
    }
    let mut diffs = Vec::new();
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut seen = std::collections::HashSet::new();
    while ia < a.len() && ib < b.len() {
        let (ta, tta, sa) = &a[ia];
        let (tb, ttb, sb) = &b[ib];
        if ta == tb && tta == ttb {
            let denom = sa.abs().max(1e-9);
            let rel = (sa - sb).abs() / denom;
            if rel > rel_lim && seen.insert((ta.clone(), *tta)) {
                diffs.push((ta.clone(), *tta, *sa, *sb, rel));
            }
            ia += 1;
            ib += 1;
        } else if ta < tb {
            eprintln!("[ckdiff] A에만: {ta} t={tta}");
            ia += 1;
        } else {
            eprintln!("[ckdiff] B에만: {tb} t={ttb}");
            ib += 1;
        }
    }
    // QA-22: 한쪽 로그 절단(크래시 전형) 잔여 — 종전엔 공통 접두만 비교하고
    // 꼬리 항목을 무보고 버려 "무상이" exit 0이었다.
    let (tail_a, tail_b) = (a.len() - ia, b.len() - ib);
    if tail_a + tail_b > 0 {
        eprintln!(
            "ckdiff: 짧은 쪽 종료 후 잔여 — A에만 {tail_a}·B에만 {tail_b}항 (한쪽 절단 의심)"
        );
        return ExitCode::FAILURE;
    }
    if diffs.is_empty() {
        println!(
            "ckdiff: 무상이 ({} 마커, rel<{rel_lim})",
            a.len().min(b.len())
        );
        return ExitCode::SUCCESS;
    }
    println!(
        "ckdiff: {}개 상이 태그 (rel>{rel_lim}) — 첫 상이:",
        diffs.len()
    );
    for (tag, t, sa, sb, rel) in diffs.iter().take(10) {
        println!("  {tag:<16} t={t:<4} A={sa:14.4} B={sb:14.4} rel={rel:.3e}");
    }
    ExitCode::FAILURE
}
