//! A23(plans/129) — 모듈 무결성 인증 실행기: llm170 mod-check <module> [--record|--if-stale|--status|--list].
//! 1호출 1모듈(사용자 지시 — VRAM 부족 기기 대응): 프로세스 수명 = 해당 모듈
//! 가중치만 부분적재 → 격리 검증기 실행(subprocess) → 골든 판정 → 종료.
//! 판정 FAIL 기준은 corr/maxdiff·NaN만 — 속도는 런간 ±1.7% 노이즈 때문에
//! 참조·Δ% 표시, 대폭 역행(참조 대비 +30% 이상 느림)만 WARN(게이트 요동 방지).
//!
//! v2(2026-10-05, 플랜 본문 ⑤·② 완수): 골든 4열(속도참조)·출력 표
//! `모듈|검증기|corr/maxdiff(현재↔골든↔Δ)|속도(현재↔참조↔Δ%)|판정`·
//! 소스 지문(.modcheck-last.tsv — 모듈 .rs+커널 .hip/.comp 해시)·지문 동일
//! PASS 스킵(--if-stale, 전체 순회 scripts/mod-check-all.sh가 사용 — 재실행
//! 되는 것은 "코드가 바뀐 모듈"뿐, 세션당 전체 인증 1회 수렴)·--status
//! certified/stale/미골든/미실행 무GPU 표.
//!
//! 골든: scripts/.modcheck-golden.tsv `module|metric|threshold|speed_ms`
//! (--record 캡처, 산술 10a 변경 시 자유 재캡처 — 토큰 게이트 --record 관례).

use std::path::PathBuf;
use std::process::ExitCode;

/// 저장소 루트(빌드 시점 — 골든·last tsv·지문 소스 경로 기준).
fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn golden_path() -> PathBuf {
    root().join("scripts/.modcheck-golden.tsv")
}

fn last_path() -> PathBuf {
    root().join("scripts/.modcheck-last.tsv")
}

/// 모듈 레지스트리 — A22 covered_by 원장의 direct: 항목 + 소스 경로(지문용).
/// v2: EXL3 vk·Q4(q4acc) 확장, R6 분해 경로 갱신. 커널 매핑이 명확한 모듈은
/// .hip/.comp까지 지문에 포함; Q4처럼 커널이 다발에 흩어진 모듈은 검증기·
/// 디스패치 .rs로 지문(커널 변경은 NAMES 정합 테스트·charhash가 담당).
const MODULES: &[(&str, &str, &[&str])] = &[
    // ── EXL3 hip(R6 분해 후 경로) ──
    (
        "exl3-hip-attn",
        "exl3-hip-attn",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/mod.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip/batch.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/attn.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_attn.hip",
        ],
    ),
    (
        "exl3-hip-gdn",
        "exl3-hip-gdn",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/batch.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/gdn.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gdn.hip",
        ],
    ),
    (
        "exl3-hip-gemm",
        "exl3-hip-gemm",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/gemm.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/gemm.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemm2.hip",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemm2_mma.hip",
        ],
    ),
    (
        "exl3-hip-gemv",
        "exl3-hip-gemv",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/gemm.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/gemv.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemv.hip",
        ],
    ),
    (
        "exl3-hip-nr",
        "exl3-hip-nr",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/gemm.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/nr.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemm2.hip",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_ew.hip",
        ],
    ),
    (
        "exl3-hip-decode",
        "exl3-hip-decode",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/mod.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip/batch.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip/forward.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/decode.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_ew.hip",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_misc.hip",
        ],
    ),
    (
        "exl3-hip-batch",
        "exl3-hip-batch",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/batch.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/batch.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemm2.hip",
        ],
    ),
    (
        "exl3-hip-mtp",
        "exl3-hip-mtp",
        &[
            "crates/backend-gpu/src/rawhip/exl3_hip/mtp.rs",
            "crates/backend-gpu/src/rawhip/exl3_hip_probe/mtp.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemm2.hip",
            "crates/backend-gpu/src/rawhip/kernels/src_exl3_gemv.hip",
        ],
    ),
    // ── EXL3 vk(직접 수치 검증기 — 종단 token 게이트는 gate-exl3 소관) ──
    (
        "exl3-vk-attn",
        "exl3-attn-check",
        &[
            "crates/backend-gpu/src/rawvk/exl3/attn.rs",
            "crates/backend-gpu/src/rawvk/checks/exl3_probes.rs",
            "crates/backend-gpu/src/rawvk/spv/exl3_attn_prep.comp",
            "crates/backend-gpu/src/rawvk/spv/exl3_attn_fwd3.comp",
        ],
    ),
    (
        "exl3-vk-nr",
        "exl3-nr-check",
        &[
            "crates/backend-gpu/src/rawvk/exl3/decode.rs",
            "crates/backend-gpu/src/rawvk/checks/exl3_probes.rs",
            "crates/backend-gpu/src/rawvk/spv/e3_norm_resid.comp",
        ],
    ),
    (
        "exl3-vk-nrh",
        "exl3-nrh-check",
        &[
            "crates/backend-gpu/src/rawvk/exl3/util.rs",
            "crates/backend-gpu/src/rawvk/checks/exl3_probes.rs",
            "crates/backend-gpu/src/rawvk/spv/e3_norm_resid_had.comp",
        ],
    ),
    (
        "exl3-vk-ffn",
        "exl3-ffn-check",
        &[
            "crates/backend-gpu/src/rawvk/exl3/staging.rs",
            "crates/backend-gpu/src/rawvk/checks/exl3_probes.rs",
            "crates/backend-gpu/src/rawvk/spv/exl3_ffn_ew.comp",
            "crates/backend-gpu/src/rawvk/spv/exl3_gemm2.comp",
        ],
    ),
    (
        "exl3-vk-gemmd",
        "exl3-gemmd-check",
        &[
            "crates/backend-gpu/src/rawvk/exl3/staging.rs",
            "crates/backend-gpu/src/rawvk/checks/exl3_probes.rs",
            "crates/backend-gpu/src/rawvk/spv/exl3_gemm2d.comp",
        ],
    ),
    (
        "exl3-vk-scan",
        "exl3-scan-check",
        &[
            "crates/backend-gpu/src/rawvk/exl3/gdn.rs",
            "crates/backend-gpu/src/rawvk/checks/exl3_probes.rs",
            "crates/backend-gpu/src/rawvk/spv/exl3_gdn_scan.comp",
        ],
    ),
    // ── Q4 vk 값경로(A22② 직접 검증기) · q4acc ──
    (
        "q4-vk-addrms",
        "addrms-check",
        &[
            "crates/backend-gpu/src/rawvk/checks/addrms.rs",
            "crates/backend-gpu/src/rawvk/spv/addrms.comp",
        ],
    ),
    // ── Q4(q4acc 직접 검증기 — 핫패스 정합) ──
    (
        "q4-ar",
        "q4-ar-check",
        &[
            "crates/backend-gpu/src/rawhip/q4acc/checks.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_gdn.hip",
        ],
    ),
    (
        "q4-qsa",
        "q4-qsa-check",
        &[
            "crates/backend-gpu/src/rawhip/q4acc/checks.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_qsa.hip",
        ],
    ),
    (
        "q4-ple",
        "q4-ple-check",
        &[
            "crates/backend-gpu/src/rawhip/q4acc/checks.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_q4.hip",
        ],
    ),
    (
        "q4-hc",
        "q4-hc-check",
        &[
            "crates/backend-gpu/src/rawhip/q4acc/checks.rs",
            "crates/backend-gpu/src/rawhip/kernels/src_q4.hip",
        ],
    ),
];

/// 검증기 출력에서 수치 지표 추출 — 토큰 우선순위: maxdiff → max_abs → corr →
/// maxrel → md → " rel=".(프롭 출력 관례가 계열별로 갈림: hip maxdiff=·vk
/// max_abs=·q4acc maxrel=/rel=. corr은 1−corr로 오차 정규화.)
fn extract_metric(out: &str) -> Option<f64> {
    let take_num = |rest: &str| -> Option<f64> {
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
            .collect();
        num.parse().ok()
    };
    for tok in [
        "maxdiff=",
        "max_abs=",
        "corr=",
        "maxrel=",
        "kern-vs-mirror=",
        "max|d-h|=",
        "md=",
    ] {
        if let Some(m) = out.find(tok) {
            let v = take_num(&out[m + tok.len()..])?;
            return if tok == "corr=" {
                Some(1.0 - v)
            } else {
                Some(v)
            };
        }
    }
    if let Some(m) = out.find(" rel=") {
        return take_num(&out[m + 5..]);
    }
    None
}

/// 소스 지문 — 파일 내용 FNV-1a(파일명+바이트 연쇄). 파일 부재 시 None
/// (지문 불능 = 스킵 불가·항상 stale 취급).
fn fingerprint(sources: &[&str]) -> Option<u64> {
    let mut h: u64 = 0xcbf2_9ce4_8222_2325;
    for s in sources {
        let path = root().join(s);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                if llm170_diag::flag::on("FPDBG") {
                    eprintln!("FPDBG 읽기실패 {} ({e})", path.display());
                }
                return None;
            }
        };
        h ^= s.as_bytes()[0] as u64;
        for b in &bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    Some(h)
}

/// 골든 행: (module, threshold, speed_ms 참조).
fn read_golden() -> Vec<(String, f64, Option<f64>)> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string(golden_path()) {
        for l in t.lines().skip(1) {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() >= 3
                && let Ok(v) = c[2].parse()
            {
                let sp = c.get(3).and_then(|s| s.parse().ok());
                out.push((c[0].into(), v, sp));
            }
        }
    }
    out
}

fn write_golden(rows: &[(String, f64, Option<f64>)]) {
    let mut t = String::from("module\tmetric\tthreshold\tspeed_ms\n");
    for (n, v, sp) in rows {
        match sp {
            Some(s) => t.push_str(&format!("{n}\tmaxdiff\t{v:.6e}\t{s:.0}\n")),
            None => t.push_str(&format!("{n}\tmaxdiff\t{v:.6e}\t\n")),
        }
    }
    let _ = std::fs::write(golden_path(), t);
}

/// last.tsv 행: (module, verdict, metric, fingerprint, 시각).
fn read_last() -> Vec<(String, String, f64, Option<u64>, String)> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string(last_path()) {
        for l in t.lines().skip(1) {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() >= 5
                && let (Ok(m), Ok(f)) = (c[2].parse(), c[3].parse::<u64>())
            {
                out.push((c[0].into(), c[1].into(), m, Some(f), c[4].into()));
            }
        }
    }
    out
}

fn write_last_row(module: &str, verdict: &str, metric: f64, fp: Option<u64>) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut rows: Vec<_> = read_last()
        .into_iter()
        .filter(|(n, _, _, _, _)| n != module)
        .collect();
    rows.push((
        module.to_string(),
        verdict.to_string(),
        metric,
        fp,
        ts.to_string(),
    ));
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let mut t = String::from("module\tverdict\tmetric\tfingerprint\tunix_ts\n");
    for (n, v, m, f, ts) in rows {
        let f = f.map(|x| x.to_string()).unwrap_or_default();
        t.push_str(&format!("{n}\t{v}\t{m:.6e}\t{f}\t{ts}\n"));
    }
    let _ = std::fs::write(last_path(), t);
}

pub(crate) fn cmd_mod_check(args: &[String]) -> ExitCode {
    if args.first().map(String::as_str) == Some("--list") {
        for (m, _, _) in MODULES {
            println!("{m}");
        }
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("--status") {
        println!("모듈 | 골든임계 | 상태(지문 기준)");
        for (m, _, srcs) in MODULES {
            let g = read_golden().into_iter().find(|(n, _, _)| n == m);
            let Some((_, thr, _)) = g else {
                println!("{m} | — | 미골든(--record) — 순회 시 FAIL");
                continue;
            };
            let cur = fingerprint(srcs);
            let last = read_last().into_iter().find(|(n, _, _, _, _)| n == m);
            match (last, cur) {
                (Some((_, v, _, Some(lf), _)), Some(c)) if v == "PASS" => {
                    let st = if lf == c {
                        "certified"
                    } else {
                        "stale(코드 변경)"
                    };
                    println!("{m} | {thr:.3e} | {st}");
                }
                (Some(_), _) => println!("{m} | {thr:.3e} | stale(재인증 필요)"),
                (None, _) => println!("{m} | {thr:.3e} | 미실행"),
            }
        }
        return ExitCode::SUCCESS;
    }
    let Some(module) = args.first().cloned() else {
        eprintln!("usage: llm170 mod-check <module> [--record|--if-stale] | --status | --list");
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
    let if_stale = args.iter().any(|a| a == "--if-stale");
    let Some((name, cmd, srcs)) = MODULES.iter().find(|(m, _, _)| *m == module.as_str()) else {
        eprintln!("error: 미지원 모듈 {module} (--list 참조)");
        return ExitCode::from(2);
    };
    let fp = fingerprint(srcs);
    // 지문 스킵(⑤) — 마지막 PASS와 소스 지문이 동일하면 재검증 생략.
    // 전체 순회(mod-check-all.sh)가 이 경로를 써서 "코드가 바뀐 모듈만" 돈다.
    if if_stale
        && !record
        && let Some((_, v, m, Some(lf), _)) =
            read_last().into_iter().find(|(n, _, _, _, _)| n == name)
        && v == "PASS"
        && Some(lf) == fp
    {
        println!("{name} | — | — | — | SKIP(지문 동일 — 마지막 PASS 재사용, metric {m:.3e})");
        return ExitCode::SUCCESS;
    }
    // 1호출 1모듈 — 자기 자신을 서브프로세스로(부분적재·종료 시 전량 해제).
    let exe = std::env::current_exe().unwrap_or_default();
    let t0 = std::time::Instant::now();
    let Ok(out) = std::process::Command::new(&exe).arg(cmd).output() else {
        eprintln!("error: 검증기 실행 실패: {cmd}");
        return ExitCode::FAILURE;
    };
    let dt_ms = t0.elapsed().as_secs_f64() * 1e3;
    let text =
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    let Some(metric) = extract_metric(&text) else {
        eprintln!("error: {name} 검증기 출력에서 지표 추출 실패(프로브 실패/포맷 미지원)");
        eprintln!("{}", text.lines().last().unwrap_or(""));
        return ExitCode::FAILURE;
    };
    if record {
        let mut g: Vec<_> = read_golden()
            .into_iter()
            .filter(|(n, _, _)| n != name)
            .collect();
        g.push((name.to_string(), metric, Some(dt_ms)));
        g.sort_by(|a, b| a.0.cmp(&b.0));
        write_golden(&g);
        write_last_row(name, "PASS", metric, fp);
        println!(
            "RECORD {name} | 검증기 {cmd} | maxdiff {metric:.3e} | 속도 {dt_ms:.0}ms → 골든 갱신"
        );
        return ExitCode::SUCCESS;
    }
    // 판정 — 골든 대비(현재↔골든↔Δ · 속도 현재↔참조↔Δ%, +30% 이상 느림 WARN).
    let Some((_, golden, spd_ref)) = read_golden().into_iter().find(|(n, _, _)| n == name) else {
        eprintln!("error: {name} 골든 없음 — mod-check {name} --record 먼저");
        return ExitCode::from(2);
    };
    let ok = metric.is_finite() && metric <= golden * 1.5 + 1e-9;
    let d_metric = metric - golden;
    let (spd_s, d_spd, warn) = match spd_ref {
        Some(r) if r > 0.0 => {
            let pct = (dt_ms - r) / r * 100.0;
            (
                format!("{dt_ms:.0}ms↔{r:.0}ms(Δ{pct:+.0}%)"),
                format!("{pct:+.0}%"),
                pct > 30.0,
            )
        }
        _ => (format!("{dt_ms:.0}ms↔—"), "—".into(), false),
    };
    let verdict = if ok { "PASS" } else { "FAIL" };
    println!(
        "{name} | 검증기 {cmd} | maxdiff {metric:.3e}↔{golden:.3e}(Δ{d_metric:+.1e}) | 속도 {spd_s} | {verdict}"
    );
    if warn {
        println!("  WARN: {name} 속도 대폭 역행(참조 대비 {d_spd} 느림 — ±1.7% 런 노이즈 밖)");
    }
    write_last_row(name, verdict, metric, fp);
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
