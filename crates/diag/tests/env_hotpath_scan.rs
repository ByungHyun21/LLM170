
const SCAN: &[(&str, &[&str])] = &[
    ("crates/core/src", &[]),
    ("crates/server/src", &["main.rs", "probes/"]),
    ("crates/backend-gpu/src", &[]),
];

fn repo_root() -> std::path::PathBuf {
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
