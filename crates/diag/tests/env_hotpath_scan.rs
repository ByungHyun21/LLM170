//! env 직접 판독 금지 계약의 정적 검사 (A6).
//!
//! 프로덕션 핫패스는 diag 스냅샷(flag::on/eq1/ne0/on_nonzero/val) 또는
//! dump::opts()/key_arg()로만 env를 읽는다(원장 89·104). 이 테스트는
//! 프로덕션 소스 트리를 스캔해 `std::env::var(_os)` 직접 호출을 찾으면
//! 실패한다 — 새 코드가 계약을 위반하면 컴파일은 되도 cargo test가 잡는다.
//!
//! 허용 목록(위반 아니고, 사유와 함께):
//! - `crates/diag/src/**` — 스냅샷·덤프 메커니즘 자체(flag VALUES·dump OPTS·
//!   fp·watchdog가 env를 읽는 유일한 곳이어야 한다).
//! - `crates/server/src/main.rs` — 부트스트랩: LLM170_FRAME 기본값 set_var(스냅샷
//!   이전에 env를 *써야* 한다)·watchdog 초 파싱·OOM adj. 1회 판독 비핫패스.
//! - `crates/server/src/probes/` — 검증층(3층 분리: 하네스는 프로덕션 계약 밖).
//! - `crates/backend-gpu/src/rawcuda/**` — fatbin 자산 경로 오버라이드 env
//!   (`LLM170_CUDA_*_FATBIN_PATH` — 계산 경로 분기 아님, 로드 경로 전용).
//! - server의 tests/·examples — 이 스캔은 src 트리만 본다.

/// (루트, 허용 파일 접두사 목록) — 루트가 없으면 테스트 실패(이동 누락 방지).
const SCAN: &[(&str, &[&str])] = &[
    ("crates/core/src", &[]),
    ("crates/server/src", &["main.rs", "probes/"]),
];

fn repo_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = crates/diag → 저장소 루트는 2단 위.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf()
}

fn scan_dir(
    repo: &std::path::Path,
    dir: &std::path::Path,
    base: &std::path::Path,
    exempt: &[&str],
    out: &mut Vec<String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        out.push(format!("스캔 루트 부재: {}", dir.display()));
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            scan_dir(repo, &p, base, exempt, out);
            continue;
        }
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".rs") {
            continue;
        }
        let rel = p
            .strip_prefix(repo)
            .unwrap_or(&p)
            .to_string_lossy()
            .into_owned();
        let rel_from_root = p
            .strip_prefix(base)
            .unwrap_or(&p)
            .to_string_lossy()
            .into_owned();
        // 허용 판정은 루트 기준 상대경로 접두사(파일명·하위 디렉터리 공용).
        if exempt.iter().any(|x| rel_from_root.starts_with(x)) {
            continue;
        }
        let Ok(txt) = std::fs::read_to_string(&p) else {
            out.push(format!("판독 실패: {rel}"));
            continue;
        };
        for (i, line) in txt.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with("//!") {
                continue;
            }
            if line.contains("std::env::var") {
                out.push(format!("{rel}:{}: {}", i + 1, t.trim()));
            }
        }
    }
}

#[test]
fn production_sources_have_no_direct_env_reads() {
    let root = repo_root();
    let mut bad = Vec::new();
    for (sub, exempt) in SCAN {
        let r = root.join(sub);
        scan_dir(&root, &r, &r, exempt, &mut bad);
    }
    assert!(
        bad.is_empty(),
        "프로덕션 직접 env 판독 발견 (diag flag/val·dump 키로 전환할 것):\n{}",
        bad.join("\n")
    );
}

/// B18: 가드 우회 env 재도입 방지 — 폐지된
/// `LLM170_NO_RSRC_GUARD`가 프로덕션 소스에 다시 등장하면 실패한다
/// (bfb30654 폐지. 동결 사고 2026-09-16 재발 경로 차단 — resource.rs는
/// env를 읽지 않는다는 계약과 세트).
#[test]
fn guard_killswitch_env_stays_banned() {
    fn walk(d: &std::path::Path, bad: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, bad);
            } else if let Ok(s) = std::fs::read_to_string(&p)
                && s.contains("LLM170_NO_RSRC_GUARD")
            {
                bad.push(p.display().to_string());
            }
        }
    }
    let root = repo_root();
    let mut bad = Vec::new();
    for dir in ["crates/core/src", "crates/server/src"] {
        walk(&root.join(dir), &mut bad);
    }
    assert!(
        bad.is_empty(),
        "가드 우회 env 재도입 감지: {bad:?} — 폐지 계약 위반(B18)"
    );
}
