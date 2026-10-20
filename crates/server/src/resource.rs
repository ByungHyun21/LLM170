//! 모델 적재 전 사전 리소스 가드 (2026-09-16).
//!
//! 사고 경위: 다른 추론 서버(Flash-Next ~104 GiB 상주)가 떠 있는 상태에서
//! 이 엔진이 같은 모델을 한 벌 더 적재하려다 RAM+VRAM이 소진되어 호스트가
//! 먹통됐다(2026-09-16 실사고, 재부팅으로 복구). OOM 킬이나 스왑 폭주로
//! 시스템이 죽는 대신, 적재 **시작 전에** 필요량·가용량을 비교해 즉시
//! 명확한 에러로 거부한다.
//!
//! 정책(보수적 상한):
//! - 필요량  = 모델 전체 바이트 x 1.10 (KV/활성/업로드 스테이징 여유)
//! - 가용량  = VRAM 가용 + 0.85 x 호스트 가용(MemAvailable)
//!   (호스트 15%는 프로세스 자체/페이지캐시 변동을 위한 상한)
//! - **gpu 적재에서 VRAM 미측정이면 거부**(B17, 2026-10-08 사용자 승인 —
//!   '오탐 차단 우선'을 '안전 우선'으로 변경. 탈출구 없음). CPU 적재의
//!   호스트 미측정(비리눅스 등)은 기존대로 경고 후 판정.
//! - 킬스위치 폐지(2026-10-03, 사용자 지시) 유지 — 가드는 상시 동작.
//! - B20(2026-10-08): 전역 적재 락(flock) —
//!   동시 기동 check-then-act 레이스 직렬화. 락 획득 **후** 판정(재판정),
//!   적재 완료 지점에서 해제(장기 상주 락 아님).

use std::path::Path;

const REQ_SLACK: f64 = 1.10;
const HOST_USABLE: f64 = 0.85;

/// 순수 판정 함수 - 단위테스트 대상.
/// B17: gpu 적재(gpu_load=true)에서 VRAM 미측정(vram_free=None)은 즉시 거부 —
/// "측정 불능→통과"가 사실상 가드 해제였다(CUDA 기기 "VRAM unknown" 실측 2회).
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
        // 측정 불능(CPU 적재·비리눅스) - 오탐 방지 위해 통과 (호출부에서 경고).
        // gpu 적재는 위 B17 게이트에서 이미 거부됐다.
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

/// 모델 회계 — W4A16 디렉터리(전 샤드 재귀 합산 + 런타임 스크래치 가산).
/// (2026-10-03 사고: 디렉터리 metadata≈0 통과로 시스템 동결 — 근본 가드 결함.)
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

/// 런타임 스크래치 추정(H 2026-10-09) — KV·활성·업로드 스테이징 몫.
/// 종전 고정 4GiB는 대형 모델에서 과소평가였다(가드 취지 = 초대형 이중 적재
/// 동결 방지): 모델 크기/8을 하한 4GiB와 함께 가산한다(소형은 종전과 동일).
fn runtime_scratch(model_bytes: u64) -> u64 {
    (model_bytes / 8).max(4 << 30)
}

/// /proc/meminfo 한 줄(kB) 파싱 — MemTotal/MemAvailable 공용.
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

/// 호스트 전체 메모리 (/proc/meminfo MemTotal) — 모니터링용.
pub fn host_mem_total() -> Option<u64> {
    meminfo_kb("MemTotal:")
}

/// 프로세스 RSS (/proc/self/status VmRSS) — 모니터링용.
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

/// 호스트 가용 메모리 (/proc/meminfo MemAvailable).
pub(crate) fn host_mem_available() -> Option<u64> {
    meminfo_kb("MemAvailable:")
}

/// [2026-10-16] VRAM 예약(워크스페이스·스크래치 + 스펙 VRAM) — 서버 예산과
/// 자원 가드가 공유하는 단일 출처. (재산정: WY·링 ≈0.2GiB + MTP 아레나 등;
/// 체크포인트 링은 핀드 호스트 RAM 이관으로 VRAM 몫 아님. dsave→dring0 순감
/// 309MiB 환원 — 2026-10-14 실측.)
pub const VRAM_RESERVE: u64 = 9 * (1 << 30) / 4 - 309 * (1 << 20); // 1995MiB — 실측 마진

/// 가드 대상(A2/R1) — 판정 결과. 판정 계약은 guard_target_cases
/// 표 테스트가 고정한다(`gpu`=VRAM 계정 여부, B17: 조회 실패=거부).
pub struct GuardTarget {
    pub path: std::path::PathBuf,
    pub gpu: bool,
}

/// 가드 대상 판정 — main() 인라인의 순수함수(A2/R1).
/// 입력: 서브커맨드, --model, 위치인자. None = 가드 스킵
/// (메타 서브커맨드 또는 경로 부재 — 로더 에러가 더 정확).
pub fn guard_target(sub: &str, model: Option<&str>, rest: &[String]) -> Option<GuardTarget> {
    // 토크나이저 파일만 판독 — 무게 미적재. plan은 매니페스트 헤더만(무적재).
    // [2026-10-19] diag(envcheck·diff 등)도 무적재 — preflight가 envcheck를
    // 부르는데 다른 llm170(예: CPU 오라클)이 락을 쥐면 600s 대기 후 preflight
    // 자체가 실패했다(실측). 진단 계열은 가드 제외.
    if matches!(sub, "tokenize" | "plan" | "diag") {
        return None;
    }
    let path = model.map(std::path::PathBuf::from).or_else(|| {
        rest.iter()
            .find(|a| !a.starts_with("--"))
            .map(std::path::PathBuf::from)
    });
    // [FN 1단계 2026-10-17] qwen4_exp(FN) w4a16-load = 식별 전용 프로브
    // (config.json + index.json 판독, 가중치 mmap 없음) — 121GiB 체크포인트
    // 용량으로 호스트 예산 판정하면 식별 자체가 막힌다. 무가드 스킵.
    // [FN D2 2026-10-20] w4a16-ref 동일 스킵 — 오라클은 스토어 직접
    // mmap 스트리밍(RSS = 페이지 캐시 분량)이라 dense 적재 예산 추정
    // (148GiB)이 실측과 무관하다.
    if matches!(sub, "w4a16-load" | "w4a16-ref")
        && path
            .as_deref()
            .and_then(llm170_core::qwen35::bind::dir_arch)
            .is_some_and(|a| llm170_core::qwen4exp::arch_ok(&a))
    {
        return None;
    }
    // 서빙 경로는 CUDA 고정 → VRAM 계정. CPU 전용 프로브는 제외.
    let gpu = !matches!(
        sub,
        "w4a16-load" | "w4a16-ref" | "w4a16-gemv" | "w4a16-gemm"
    );
    path.map(|path| GuardTarget { path, gpu })
}

/// [2026-10-16] 범주 인지 dense 게이트(순수 — 테스트 표면) — ① 매-토큰
/// dense(+head)는 VRAM 필수. 합산 가드는 dense가 호스트 용량에 묻혀 통과할 수
/// 있다(소형 VRAM + 대형 호스트) → 로드 중 VRAM 실패/스래시. 예약 공유.
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
        return Ok(()); // 경로 오류는 로더의 에러가 더 정확하다 - 여기서는 통과
    }
    let vram = if gpu {
        // B6: CUDA 단일(2026-10-08 백엔드 탈락 반영) — cuMemGetInfo(rawcuda ffi).
        // 실패는 None → check의 B17 게이트가 거부한다.
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
    // [2026-10-16] 범주 인지 게이트 — 매니페스트 판독 시 ① dense는 VRAM 필수.
    if gpu
        && let Some(vf) = vram
        && let Ok(man) = llm170_core::plan::Manifest::from_dir(model)
    {
        check_dense_vram(man.sums().dense, vf)?;
    }
    check(bytes, vram, host, gpu)
}

// ── B20: 전역 적재 락(flock) ──────────────────────────────────────────────
// 동시 기동 레이스: 가드는 기동 시점 1회 판정이라 두 프로세스가 동시에 뜨면
// 양쪽 다 통과한 뒤 둘 다 적재한다(2026-09-16 동결 사고 계급). 전역 락으로
// 적재 창구를 직렬화하고, 락 획득 후 재판정한다(main이 acquire → preflight
// 순서로 호출). 해제는 적재 완료 지점(build_slots/각 로더 반환 후) — 장기
// 상주 락이 아니다. 모델 무관: 다른 모델끼리도 동시 적재 금지(사고 보고).

// std 외 크레이트 금지 계약(rawcuda loader 관례) — libc는 std가 이미 링크.
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

/// 적재 락 획득 — 배타(flock). 선점 중이면 보유자 pid를 보고하며 최대 600초
/// 대기(대형 적재 창구), 초과 시 거부. 같은 프로세스 재획득은 no-op.
pub fn acquire_load_lock() -> Result<(), String> {
    #[cfg(not(unix))]
    return Ok(()); // 비유닉스: flock 부재 — 가드 판정만으로 동작
    #[cfg(unix)]
    {
        let mut g = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_some() {
            return Ok(()); // 이미 보유(같은 프로세스 재진입)
        }
        // SAFETY: getuid는 부작용 없는 조회.
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
            // SAFETY: f의 raw fd에 저수준 잠금 — fd 소유는 f가 유지한다.
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
        // 보유자 기록(해제 시 지운다 — stale pid 최소화).
        let _ = f.set_len(0);
        use std::io::{Seek, Write};
        let mut f = f;
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let _ = write!(f, "{}", std::process::id());
        *g = Some(f);
        Ok(())
    }
}

/// 적재 락 해제 — 적재 완료 지점(build_slots 반환 직후 등)에서 호출.
/// 프로세스 exit도 fd close로 해제된다(단명 CLI 경로).
pub fn release_load_lock() {
    let mut g = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut f) = g.take() {
        let _ = f.set_len(0);
        use std::io::{Seek, Write};
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let _ = write!(f, "0");
        // drop(f) — flock 해제(fd close)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn passes_flash_next_standalone() {
        // 실측 조건(2026-09-16): 모델 103.7GiB, VRAM 가용 95.7, MemAvailable 26.
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
        // 다른 서버가 상주해 VRAM~=0, 호스트~=5GiB 남은 이중 적재 사고 조건.
        assert!(check(103_700 * (1 << 20), Some(GIB), Some(5 * GIB), true).is_err());
    }

    #[test]
    fn passes_27b_alongside_resident() {
        assert!(check(17 * GIB, Some(80 * GIB), Some(22 * GIB), true).is_ok());
    }

    #[test]
    fn cpu_unknown_measurements_pass_with_warning() {
        // CPU 적재: 측정 불능 통과 유지(오탐 방지 — 비리눅스 등).
        assert!(check(103_700 * (1 << 20), None, None, false).is_ok());
        // 호스트 불능 + VRAM 만으로 모델×슬랙을 못 덮으면 거부 — 가드의 목적
        // (이중 적재 동결 방지)상 이것이 옳다(2026-09-16: 통과 기대는 산식과
        // 모순되어 수정).
        assert!(check(103_700 * (1 << 20), Some(90 * GIB), None, true).is_err());
    }

    #[test]
    fn runtime_scratch_scales_for_large_models() {
        // 소형/중형(27B급 13.6GiB)은 종전과 동일 4GiB.
        assert_eq!(runtime_scratch(13_600 * (1 << 20)), 4 << 30);
        // 대형(136GiB)은 1/8 = 17GiB — 고정 4GiB 과소평가 교정.
        assert_eq!(runtime_scratch(136 * GIB), 17 * GIB);
    }

    #[test]
    fn b17_gpu_load_requires_vram_measurement() {
        // B17 승인 정책(2026-10-08): gpu 적재에서 VRAM 미측정 = 거부.
        // 호스트가 넉넉해도(CUDA 기기 "VRAM unknown" 사고 재현 조건) 거부.
        assert!(check(17 * GIB, None, Some(200 * GIB), true).is_err());
        assert!(check(17 * GIB, None, None, true).is_err());
        // 측정되면 통과(용량 충분).
        assert!(check(17 * GIB, Some(20 * GIB), Some(30 * GIB), true).is_ok());
    }

    /// [2026-10-16] 범주 인지 dense 게이트 — 합산 가드의 사각(호스트가 dense를
    /// 묻어 통과)을 차단. dense+예약 ≤ VRAM 통과 / 초과 거부.
    #[test]
    fn dense_vram_gate() {
        // 27B급: dense 15.7G + 예약 ~2G ≤ 20G → 통과.
        assert!(check_dense_vram(15_700 * (1 << 20), 20 * GIB).is_ok());
        // dense 15.7G vs VRAM 10G(호스트가 커도) → 거부.
        let e = check_dense_vram(15_700 * (1 << 20), 10 * GIB).unwrap_err();
        assert!(e.contains("dense"), "{e}");
        // 경계: dense + 예약 == 가용 → 통과(초과만 거부).
        assert!(check_dense_vram(10 * GIB, 10 * GIB + VRAM_RESERVE).is_ok());
        assert!(check_dense_vram(10 * GIB, 10 * GIB + VRAM_RESERVE - 1).is_err());
    }

    /// guard_target 판정 표 — 서브커맨드×인자 형태 계약(B19 확대 유지).
    #[test]
    fn guard_target_cases() {
        use super::guard_target;
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        // serve/infer류: --model (CUDA 고정 → gpu=true)
        let g = guard_target("serve", Some("/m/w4a16"), &s(&[]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/w4a16".into(), true))
        );
        let g = guard_target("infer", Some("/m/w4a16"), &s(&[]));
        assert_eq!(g.map(|g| g.gpu), Some(true));
        // CPU 전용 프로브 — 위치인자 폴백(gpu=false).
        let g = guard_target("w4a16-load", None, &s(&["/m/w4a16"]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/w4a16".into(), false))
        );
        let g = guard_target("w4a16-ref", None, &s(&["/m/w4a16"]));
        assert_eq!(g.map(|g| g.gpu), Some(false));
        // 메타 서브커맨드 → None (plan = 매니페스트 헤더만 — 무적재)
        assert!(guard_target("tokenize", Some("/m/w4a16"), &s(&[])).is_none());
        assert!(guard_target("plan", Some("/m/w4a16"), &s(&[])).is_none());
        // 무모델 로딩 창구 → None(로더/CLI 에러가 더 정확)
        assert!(guard_target("infer", None, &s(&[])).is_none());
        assert!(guard_target("serve", None, &s(&[])).is_none());
    }
}
