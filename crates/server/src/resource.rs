
use std::path::Path;

const REQ_SLACK: f64 = 1.10;
const HOST_USABLE: f64 = 0.85;

pub fn check(
    model_bytes: u64,
    vram_free: Option<u64>,
    host_avail: Option<u64>,
    gpu_load: bool,
) -> Result<(), String> {
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
    if gpu_load && vram_free.is_none() {
        return Err(format!(
            "VRAM 측정 불능 — gpu 적재는 VRAM 조회가 필수다(B17, 안전 우선 정책). \
             model ~{} (with slack). 런타임/드라이버 상태 확인 후 재시도",
            gib((model_bytes as f64 * REQ_SLACK) as u64),
        ));
    }
    let required = (model_bytes as f64 * REQ_SLACK) as u64;
    let mut capacity = 0u64;
    let mut known = false;
    if let Some(v) = vram_free {
        capacity = capacity.saturating_add(v);
        known = true;
    }
    if let Some(h) = host_avail {
        capacity = capacity.saturating_add((h as f64 * HOST_USABLE) as u64);
        known = true;
    }
    if !known {
        return Ok(());
    }
    if required > capacity {
        return Err(format!(
            "insufficient resources: model needs ~{} (with slack) but only {} available (VRAM {}, host {}). \
             Another inference process may be resident - free GPU/RAM and retry.",
            gib(required),
            gib(capacity),
            vram_free.map(gib).unwrap_or_else(|| "unknown".into()),
            host_avail.map(gib).unwrap_or_else(|| "unknown".into()),
        ));
    }
    Ok(())
}

fn model_bytes(p: &Path) -> u64 {
    if p.is_dir() {
        let mut total = 0u64;
        fn walk(d: &Path, acc: &mut u64) {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let md = e.metadata();
                    if md.as_ref().is_ok_and(|m| m.is_dir()) {
                        walk(&e.path(), acc);
                    } else if let Ok(m) = md {
                        *acc += m.len();
                    }
                }
            }
        }
        walk(p, &mut total);
        return total.saturating_add(runtime_scratch(total));
    }
    p.metadata().map(|m| m.len()).unwrap_or(0)
}

fn runtime_scratch(model_bytes: u64) -> u64 {
    (model_bytes / 8).max(4 << 30)
}

fn meminfo_kb(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let kb: u64 = rest.trim().trim_end_matches(" kB").parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

pub fn host_mem_total() -> Option<u64> {
    meminfo_kb("MemTotal:")
}

pub fn process_rss() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

pub(crate) fn host_mem_available() -> Option<u64> {
    meminfo_kb("MemAvailable:")
}

pub const VRAM_RESERVE: u64 = 9 * (1 << 30) / 4 - 309 * (1 << 20);

pub struct GuardTarget {
    pub path: std::path::PathBuf,
    pub gpu: bool,
}

pub fn guard_target(sub: &str, model: Option<&str>, rest: &[String]) -> Option<GuardTarget> {
    if matches!(sub, "tokenize" | "plan" | "diag") {
        return None;
    }
    let path = model.map(std::path::PathBuf::from).or_else(|| {
        rest.iter()
            .find(|a| !a.starts_with("--"))
            .map(std::path::PathBuf::from)
    });
    if matches!(sub, "w4a16-load" | "w4a16-ref" | "w4a16-gpu")
        && path
            .as_deref()
            .and_then(llm170_core::qwen35::bind::dir_arch)
            .is_some_and(|a| llm170_core::qwen4exp::arch_ok(&a))
    {
        return None;
    }
    let gpu = !matches!(
        sub,
        "w4a16-load" | "w4a16-ref" | "w4a16-gemv" | "w4a16-gemm"
    );
    path.map(|path| GuardTarget { path, gpu })
}

pub fn fn_layout_bytes(dir: &std::path::Path) -> Result<(u64, u64), String> {
    let ar = llm170_core::st::StArchive::open(dir).map_err(|e| format!("FN 레이아웃: {e}"))?;
    let (mut dense, mut experts, mut ple) = (0u64, 0u64, 0u64);
    for (name, e) in ar.entries() {
        let b = e.nbytes();
        if name.contains(".mlp.experts.") {
            experts += b;
        } else if name.contains(".ple_embedding.") {
            ple += b;
        } else {
            dense += b;
        }
    }
    Ok((dense, experts + ple / 4))
}

pub fn check_dense_vram(dense: u64, vram_free: u64) -> Result<(), String> {
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
    let need = dense.saturating_add(VRAM_RESERVE);
    if need > vram_free {
        return Err(format!(
            "insufficient VRAM: dense(매-토큰 — VRAM 필수) ~{} + 예약 {} > 가용 {}. \
             호스트가 넉넉해도 dense는 상주해야 한다(오프로드 허용 = 전문가·KV 한정) — \
             다른 추론 프로세스의 VRAM을 확보한 뒤 재시도.",
            gib(dense),
            gib(VRAM_RESERVE),
            gib(vram_free)
        ));
    }
    Ok(())
}

pub fn preflight(model: &Path, gpu: bool) -> Result<(), String> {
    let bytes = model_bytes(model);
    if bytes == 0 {
        return Ok(());
    }
    let bytes = match fn_layout_bytes(model) {
        Ok((_dense, host)) if host > 0 => host,
        _ => bytes,
    };
    let vram = if gpu {
        let probe = llm170_backend_gpu::cuda_mem_free();
        match probe {
            Some((free, _total)) => Some(free),
            None => {
                eprintln!("# rsrc-guard: VRAM 조회 실패 — gpu 적재는 거부된다(B17)");
                None
            }
        }
    } else {
        None
    };
    let host = host_mem_available();
    if host.is_none() {
        eprintln!("# rsrc-guard: MemAvailable 조회 불가 - 호스트 항목 없이 판정한다");
    }
    if gpu
        && let Some(vf) = vram
        && let Ok(man) = llm170_core::plan::Manifest::from_dir(model)
    {
        check_dense_vram(man.sums().dense, vf)?;
    }
    check(bytes, vram, host, gpu)
}


#[cfg(unix)]
unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
    fn getuid() -> u32;
}
#[cfg(unix)]
const LOCK_EX: i32 = 2;
#[cfg(unix)]
const LOCK_NB: i32 = 4;

static LOAD_LOCK: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

pub fn acquire_load_lock() -> Result<(), String> {
    #[cfg(not(unix))]
    return Ok(());
    #[cfg(unix)]
    {
        let mut g = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_some() {
            return Ok(());
        }
        let uid = unsafe { getuid() };
        let path = format!("/tmp/llm170-load-{uid}.lock");
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| format!("적재 락 파일 열기 실패({path}): {e}"))?;
        use std::os::unix::io::AsRawFd;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
        let mut reported = 0u32;
        loop {
            let r = unsafe { flock(f.as_raw_fd(), LOCK_EX | LOCK_NB) };
            if r == 0 {
                break;
            }
            let holder = std::fs::read_to_string(&path).unwrap_or_default();
            if reported.is_multiple_of(40) {
                eprintln!("# rsrc-guard: 다른 llm170 적재 진행 중(보유 pid {holder}) — 대기(B20)");
            }
            reported += 1;
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "적재 락 대기 시간 초과(600s) — 보유 pid {holder}. 동시 기동 금지(B20)"
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        let _ = f.set_len(0);
        use std::io::{Seek, Write};
        let mut f = f;
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let _ = write!(f, "{}", std::process::id());
        *g = Some(f);
        Ok(())
    }
}

pub fn release_load_lock() {
    let mut g = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut f) = g.take() {
        let _ = f.set_len(0);
        use std::io::{Seek, Write};
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let _ = write!(f, "0");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn passes_flash_next_standalone() {
        assert!(
            check(
                103_700 * (1 << 20),
                Some(95 * GIB + 768 * (1 << 20)),
                Some(26 * GIB),
                true,
            )
            .is_ok()
        );
    }

    #[test]
    fn refuses_double_resident() {
        assert!(check(103_700 * (1 << 20), Some(GIB), Some(5 * GIB), true).is_err());
    }

    #[test]
    fn passes_27b_alongside_resident() {
        assert!(check(17 * GIB, Some(80 * GIB), Some(22 * GIB), true).is_ok());
    }

    #[test]
    fn cpu_unknown_measurements_pass_with_warning() {
        assert!(check(103_700 * (1 << 20), None, None, false).is_ok());
        assert!(check(103_700 * (1 << 20), Some(90 * GIB), None, true).is_err());
    }

    #[test]
    fn runtime_scratch_scales_for_large_models() {
        assert_eq!(runtime_scratch(13_600 * (1 << 20)), 4 << 30);
        assert_eq!(runtime_scratch(136 * GIB), 17 * GIB);
    }

    #[test]
    fn b17_gpu_load_requires_vram_measurement() {
        assert!(check(17 * GIB, None, Some(200 * GIB), true).is_err());
        assert!(check(17 * GIB, None, None, true).is_err());
        assert!(check(17 * GIB, Some(20 * GIB), Some(30 * GIB), true).is_ok());
    }

    #[test]
    fn dense_vram_gate() {
        assert!(check_dense_vram(15_700 * (1 << 20), 20 * GIB).is_ok());
        let e = check_dense_vram(15_700 * (1 << 20), 10 * GIB).unwrap_err();
        assert!(e.contains("dense"), "{e}");
        assert!(check_dense_vram(10 * GIB, 10 * GIB + VRAM_RESERVE).is_ok());
        assert!(check_dense_vram(10 * GIB, 10 * GIB + VRAM_RESERVE - 1).is_err());
    }

    #[test]
    fn guard_target_cases() {
        use super::guard_target;
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let g = guard_target("serve", Some("/m/w4a16"), &s(&[]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/w4a16".into(), true))
        );
        let g = guard_target("infer", Some("/m/w4a16"), &s(&[]));
        assert_eq!(g.map(|g| g.gpu), Some(true));
        let g = guard_target("w4a16-load", None, &s(&["/m/w4a16"]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/w4a16".into(), false))
        );
        let g = guard_target("w4a16-ref", None, &s(&["/m/w4a16"]));
        assert_eq!(g.map(|g| g.gpu), Some(false));
        assert!(guard_target("tokenize", Some("/m/w4a16"), &s(&[])).is_none());
        assert!(guard_target("plan", Some("/m/w4a16"), &s(&[])).is_none());
        assert!(guard_target("infer", None, &s(&[])).is_none());
        assert!(guard_target("serve", None, &s(&[])).is_none());
    }
}
