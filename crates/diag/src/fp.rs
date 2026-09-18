//! fp — 스테이지 지문(fingerprint) 기록·비교 (plans/82 §1).
//!
//! 각 스테이지의 출력 버퍼를 FNV-1a 64 해시 + max-abs + 첫 비순수로 요약해
//! 파일에 기록한다. 두 실행의 파일을 `diag diff`로 비교해 최초 발산
//! 스테이지를 자동 특정한다.
//!
//! 게이트: `LLM170_FP_FILE=경로` — 미설정 시 제로 코스트(원자 1회).

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::sync::OnceLock;

static ON: AtomicBool = AtomicBool::new(false);

fn writer() -> &'static Mutex<Option<std::fs::File>> {
    static W: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    W.get_or_init(|| Mutex::new(None))
}

/// 초기화 — `LLM170_FP_FILE` 경로로 출력 파일을 열고 게이트 on.
pub fn init_from_env() {
    let path = match std::env::var("LLM170_FP_FILE") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    let f = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("diag::fp: 파일 생성 실패 {path}: {e}");
            return;
        }
    };
    if let Ok(mut w) = writer().lock() {
        *w = Some(f);
    }
    ON.store(true, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// 스테이지 지문 기록 — FNV-1a 64 + max-abs + 첫 비순수 인덱스.
pub fn fp_record(stage: &str, data: &[f32]) {
    if !enabled() {
        return;
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut max_abs: f32 = 0.0;
    let mut first_nonfinite: Option<usize> = None;
    for (i, &v) in data.iter().enumerate() {
        hash = hash.wrapping_mul(0x100000001b3) ^ (v.to_bits() as u64);
        let a = v.abs();
        if a > max_abs && a.is_finite() {
            max_abs = a;
        }
        if first_nonfinite.is_none() && !v.is_finite() {
            first_nonfinite = Some(i);
        }
    }
    if let Ok(mut guard) = writer().lock() {
        if let Some(f) = guard.as_mut() {
            let _ = writeln!(
                f,
                "{stage}\t{hash:016x}\t{max_abs:.6e}\t{}",
                first_nonfinite.map_or(-1, |i| i as i64)
            );
            // NaN guard: 첫 비순수 발견 시 경고 (LLM170_FP_NAN=1이면 exit).
            if let Some(idx) = first_nonfinite {
                eprintln!(
                    "diag::fp: NaN/Inf at stage '{stage}' index {idx}"
                );
                if std::env::var_os("LLM170_FP_NAN").is_some() {
                    std::process::exit(101);
                }
            }
        }
    }
}

/// 두 지문 파일 비교 — 최초 불일치 스테이지 반환.
/// 반환: (최초 불일치 스테이지, 전체 불일치 목록, 상세).
pub fn fp_diff(path_a: &str, path_b: &str) -> Result<DiffReport, String> {
    let parse = |p: &str| -> Result<Vec<(String, u64, f64, i64)>, String> {
        let text = std::fs::read_to_string(p)
            .map_err(|e| format!("읽기 실패 {p}: {e}"))?;
        let mut out = Vec::new();
        for ln in text.lines() {
            let parts: Vec<&str> = ln.split('\t').collect();
            if parts.len() == 4 {
                let hash = u64::from_str_radix(parts[1], 16)
                    .map_err(|e| format!("해시 파싱 실패: {e}"))?;
                let maxabs: f64 = parts[2].parse()
                    .map_err(|e| format!("maxabs 파싱 실패: {e}"))?;
                let nf: i64 = parts[3].parse()
                    .map_err(|e| format!("nonfinite 파싱 실패: {e}"))?;
                out.push((parts[0].to_string(), hash, maxabs, nf));
            }
        }
        Ok(out)
    };
    let a = parse(path_a)?;
    let b = parse(path_b)?;
    let mut first_mismatch: Option<(usize, String)> = None;
    let mut all_mismatches = Vec::new();
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i].1 != b[i].1 {
            let entry = format!(
                "  [{i}] {}:\tA={:016x} max={:.3e} nf={}\tB={:016x} max={:.3e} nf={}",
                a[i].0, a[i].1, a[i].2, a[i].3, b[i].1, b[i].2, b[i].3
            );
            all_mismatches.push(entry);
            if first_mismatch.is_none() {
                first_mismatch = Some((i, a[i].0.clone()));
            }
        }
    }
    if a.len() != b.len() {
        all_mismatches.push(format!(
            "  (행 수 불일치: A={} B={})",
            a.len(),
            b.len()
        ));
    }
    Ok(DiffReport {
        first_mismatch,
        total_stages: n,
        mismatch_count: all_mismatches.len(),
        details: all_mismatches,
    })
}

/// 지문 비교 보고서.
pub struct DiffReport {
    pub first_mismatch: Option<(usize, String)>,
    pub total_stages: usize,
    pub mismatch_count: usize,
    pub details: Vec<String>,
}

impl std::fmt::Display for DiffReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.first_mismatch {
            Some((idx, name)) => {
                writeln!(f, "최초 발산: 스테이지 [{idx}] {name}")?;
                writeln!(
                    f,
                    "전체: {} 스테이지 중 {} 불일치",
                    self.total_stages, self.mismatch_count
                )?;
            }
            None => {
                writeln!(f, "완전 일치 ({} 스테이지)", self.total_stages)?;
            }
        }
        for d in &self.details {
            writeln!(f, "{d}")?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_deterministic() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let c = vec![1.0f32, 2.0, 3.1];
        let ha = {
            let mut h: u64 = 0xcbf29ce484222325;
            for v in &a { h = h.wrapping_mul(0x100000001b3) ^ (v.to_bits() as u64); }
            h
        };
        let hb = {
            let mut h: u64 = 0xcbf29ce484222325;
            for v in &b { h = h.wrapping_mul(0x100000001b3) ^ (v.to_bits() as u64); }
            h
        };
        let hc = {
            let mut h: u64 = 0xcbf29ce484222325;
            for v in &c { h = h.wrapping_mul(0x100000001b3) ^ (v.to_bits() as u64); }
            h
        };
        assert_eq!(ha, hb);
        assert_ne!(ha, hc);
        let _ = (a, b, c);
    }

    #[test]
    fn diff_detects_mismatch() {
        let dir = std::env::temp_dir();
        let pa = dir.join("diag_fp_test_a.txt");
        let pb = dir.join("diag_fp_test_b.txt");
        std::fs::write(&pa, "s1\t0000000000000001\t1.000000e+00\t-1\ns2\t0000000000000002\t2.000000e+00\t-1\ns3\t0000000000000003\t3.000000e+00\t-1\n").unwrap();
        std::fs::write(&pb, "s1\t0000000000000001\t1.000000e+00\t-1\ns2\t00000000000000ff\t2.000000e+00\t-1\ns3\t0000000000000003\t3.000000e+00\t-1\n").unwrap();
        let r = fp_diff(pa.to_str().unwrap(), pb.to_str().unwrap()).unwrap();
        assert_eq!(r.mismatch_count, 1);
        assert_eq!(r.first_mismatch.as_ref().unwrap().1, "s2");
        let _ = std::fs::remove_file(&pa);
        let _ = std::fs::remove_file(&pb);
    }
}
