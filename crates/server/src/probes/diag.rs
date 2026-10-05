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
            // plans/83 C3: 청크 불변성 자동 검증 — `llm170 diag chunk-check <model> <prompt> [sizes...]`
            else if args.first().map(String::as_str) == Some("chunk-check") {
                return Some(cmd_chunk_check(&args[1..]));
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
            // plans/87 §1 — 의도적 디스크립터-오프셋 OOB 폴트 유발.
            else if args.first().map(String::as_str) == Some("vk-fault-probe") {
                return Some(cmd_vk_fault_probe());
            }
            // plans/87 §2 — 와치독 자가 시험(진동 정지 후 스폰).
            else if args.first().map(String::as_str) == Some("watchdog-selftest") {
                return Some(cmd_watchdog_selftest());
            }
            // plans/107 W8 — 폴백 카운터 관측. A5(plans/129)부터 diag 공유 원장
            // (EXL3·vl 포함 — 실측 프로세스 종료 [fb] 출력과 같은 소스).
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
                Err("diag: 하위커맨드 diff | chunk-check | va-lookup | vk-fault-probe | watchdog-selftest | ckdiff | fb | envcheck".into())
            }
        }
        _ => return None,
    };
    Some(super::finish(r))
}


/// `llm170 diag chunk-check <model> <prompt> [sizes...] [--backend cpu]`
/// 청크 불변성 자동 검증 (plans/83 C3, docs/chunk-invariance.md 계약).
///
/// 프롬프트를 단일 호출(기준)과 각 청크 크기로 프리필해 최종 logits를 비교:
/// - bits 동일 → PASS
/// - argmax 동일 && max|Δ| < 1e-3 → PASS(near-tie, GPU 행 수 의존 잔여 축)
/// - 그 외 → FAIL
///
/// prompt: "1,2,3" 형태면 토큰 id, 아니면 텍스트(BPE 인코딩 — plans/83 A).
fn cmd_chunk_check(args: &[String]) -> ExitCode {
    let usage = "사용법: llm170 diag chunk-check <model> <prompt> [sizes...] [--backend cpu]";
    let Some(model) = args.first() else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    let Some(prompt) = args.get(1) else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    let backend_cpu = args
        .iter()
        .any(|a| a == "--backend" && args.iter().any(|b| b == "cpu"))
        || args.iter().any(|a| a == "--backend=cpu");
    let sizes: Vec<usize> = args[2..]
        .iter()
        .filter_map(|a| a.parse::<usize>().ok())
        .filter(|&s| s > 0)
        .collect();
    let sizes = if sizes.is_empty() {
        vec![16, 63, 64, 512]
    } else {
        sizes
    };

    // 프롬프트 파싱 — 숫자/콤마 전용이면 토큰 id, 아니면 텍스트
    let ids: Vec<u32> = if prompt
        .bytes()
        .all(|b| b.is_ascii_digit() || b == b',' || b == b' ')
        && prompt.contains(',')
    {
        prompt
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect()
    } else {
        let p = std::path::PathBuf::from(model);
        let stem = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let part2 = if stem.contains("-00001-of-") {
            Some(p.with_file_name(stem.replace("-00001-of-", "-00002-of-")))
        } else {
            None
        };
        match crate::tokenize::Tokenizer::load(&p, part2.as_deref()) {
            Ok(t) => t.encode(prompt),
            Err(e) => {
                eprintln!("error: tokenizer: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    if ids.is_empty() {
        eprintln!("error: 빈 프롬프트");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "# chunk-check: {}토큰, sizes={:?}, backend={}",
        ids.len(),
        sizes,
        if backend_cpu { "cpu" } else { "gpu" }
    );

    let path = std::path::PathBuf::from(model);
    let arch = llm170_gguf::GgufFile::open(&path)
        .ok()
        .and_then(|g| g.arch().map(|s| s.to_string()));
    let ctx = ids.len() * 2 + 64;
    let (ref_l, runs): (Vec<f32>, Vec<(usize, Vec<f32>)>) = match arch.as_deref() {
        Some("qwen4exp") => {
            let res = llm170_core::qwen4exp::Model4::load(&path)
                .map_err(|e| e.to_string())
                .and_then(|m| {
                    let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, 1, ctx);
                    if !backend_cpu && !crate::engine::q4_gpu_env_off() {
                        let sources = eng.model.part_sources();
                        // plans/88 — 런타임 존중: LLM170_GPU_RUNTIME=vulkan 이면 vk
                        // 프레임 경로의 청크 펜스를 잴 수 있다(종전 hip 고정이라
                        // vk 변경의 펜스 검증이 불가했다).
                        let vk = std::env::var("LLM170_GPU_RUNTIME")
                            .map(|v| v == "vulkan")
                            .unwrap_or(false);
                        let r = if vk {
                            llm170_backend_gpu::new_q4_acc_vk_with_sources(sources)
                        } else {
                            llm170_backend_gpu::new_q4_acc_with_sources(sources)
                        };
                        match r {
                            Ok(acc) => {
                                eng = eng.with_acc(acc);
                            }
                            Err(e) => {
                                return Err(format!(
                                    "GPU 가속기 생성 실패 — {e} (--backend cpu 로 회피)"
                                ));
                            }
                        }
                    }
                    // 기준: 단일 청크(프롬프트 전체) — 청크 크기는 env 스냅샷이
                    // 아니라 하네스 오버라이드 API로(A6: set_var는 스냅샷 이후 무효).
                    llm170_core::qwen4exp::layers::set_q4_chunk(ids.len().max(1));
                    let r = eng.prefill(0, &ids).map_err(|e| e.to_string())?;
                    eng.reset_seq(0);
                    let mut runs = Vec::new();
                    for &sz in &sizes {
                        llm170_core::qwen4exp::layers::set_q4_chunk(sz);
                        let l = eng.prefill(0, &ids).map_err(|e| e.to_string())?;
                        eng.reset_states();
                        runs.push((sz, l));
                    }
                    Ok((r, runs))
                });
            match res {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        _ => {
            // qwen35 (및 기본) — 호출부 청킹
            let res = llm170_core::qwen35::Model::load(&path)
                .map_err(|e| e.to_string())
                .and_then(|m| {
                    let cc_seq: usize = std::env::var("LLM170_CC_SEQ")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let mut eng = llm170_core::qwen35::Engine::new(m, (cc_seq + 1).max(1), ctx);
                    if !backend_cpu
                        && std::env::var("LLM170_RAWHIP")
                            .map(|v| v != "0")
                            .unwrap_or(true)
                    {
                        let _ = llm170_backend_gpu::inject_rawhip(&mut eng);
                    }
                    let r = eng.prefill(0, &ids).map_err(|e| e.to_string())?;
                    eng.reset_seq(cc_seq);
                    let mut runs = Vec::new();
                    for &sz in &sizes {
                        let mut last = None;
                        for ch in ids.chunks(sz) {
                            last = Some(eng.prefill(cc_seq, ch).map_err(|e| e.to_string())?);
                        }
                        eng.reset_states();
                        runs.push((sz, last.expect("청크 1개 이상")));
                    }
                    Ok((r, runs))
                });
            match res {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };

    let ref_tok = llm170_core::qwen35::greedy(&ref_l);
    println!("reference: {} logits, argmax={ref_tok}", ref_l.len());
    let mut all_pass = true;
    for (sz, l) in &runs {
        // QA-19: NaN 불감 수리 — fold(0.0, f32::max)는 NaN을 무시(반대편
        // 반환)해 diff 전체가 NaN이어도 maxd=0 → "PASS bits-identical".
        let bad_nan = l.iter().any(|v| v.is_nan()) || ref_l.iter().any(|v| v.is_nan());
        let maxd = l
            .iter()
            .zip(ref_l.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let tok = llm170_core::qwen35::greedy(l);
        let (verdict, why) = if bad_nan {
            all_pass = false;
            ("FAIL", "NaN in logits".to_string())
        } else if l.len() != ref_l.len() {
            // QA-19: 길이 불일치 — 종전엔 교집합 zip만 비교해 짧은 logits가 통과.
            all_pass = false;
            ("FAIL", format!("len {} != ref {}", l.len(), ref_l.len()))
        } else if maxd == 0.0 {
            ("PASS", "bits-identical".to_string())
        } else if tok == ref_tok && maxd < 1e-3 {
            (
                "PASS",
                format!(
                    "near-tie max|Δ|={maxd:.3e} (행 수 의존 잔여축 — plans/archive/chunk-invariance.md)"
                ),
            )
        } else {
            all_pass = false;
            ("FAIL", format!("max|Δ|={maxd:.3e} argmax {tok}≠{ref_tok}"))
        };
        println!("  chunk {sz:5}: {verdict} — {why}");
    }
    if all_pass {
        ExitCode::SUCCESS
    } else {
        println!("chunk-check: FAIL — 청크 불변성 위반 (plans/archive/chunk-invariance.md)");
        ExitCode::FAILURE
    }
}

/// plans/87 §1 — tsv 원장(site, bytes, va, va_end, seq)에서 폴트 주소 매칭.
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

/// plans/87 §1 — 의도적 GPUVM 폴트(디스크립터 오프셋 OOB) 유발.
fn cmd_vk_fault_probe() -> ExitCode {
    match llm170_backend_gpu::rawvk::checks::fault_probe() {
        Ok(msg) => {
            println!("# fault-probe: {msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
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
