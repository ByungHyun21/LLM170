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
//! - B20(plans/cuda-models.md §4·§5, 2026-10-08): 전역 적재 락(flock) —
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

/// B7(plans/cuda-models.md §5): 스플릿 GGUF 파트 경로 정규화 — `-NNNNN-of-MMMMM.gguf`
/// 어떤 파트로 지정해도 파트1 기준 경로로(Model4::load가 파트를 전개하는 기준).
/// 종전 model_bytes의 rfind("-00001-of-")는 part2/3 입력을 단일 파일로 과소
/// 계상했다. 파트1이 없으면 원본 경로 그대로.
fn split_part1(p: &Path) -> std::path::PathBuf {
    let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
        return p.to_path_buf();
    };
    let b = name.as_bytes();
    let digits5 = |s: &[u8]| s.len() == 5 && s.iter().all(u8::is_ascii_digit);
    // 뒤쪽 `-MMMMM-of-NNNNN.gguf` 접미 확인(첫 `-of-`가 아니다 — 모델명에
    // `-of-`가 포함될 수 있다).
    if let Some(of) = name.rfind("-of-")
        && b.len() >= of + 9 + 5
        && b[of + 4..of + 9].starts_with(&b[of + 4..])
        && digits5(&b[of + 4..of + 9])
        && b[of + 9..].starts_with(b".gguf")
        && of >= 6
        && b[of - 6] == b'-'
        && digits5(&b[of - 5..of])
    {
        let mut n2 = String::with_capacity(name.len());
        n2.push_str(&name[..of - 6]);
        n2.push_str("-00001");
        n2.push_str(&name[of..]);
        let p1 = p.with_file_name(n2);
        if p1.exists() {
            return p1;
        }
    }
    p.to_path_buf()
}

/// 스플릿 GGUF 전체 파트 크기 합 - `-00001-of-00004.gguf` 패턴(Model4::load와 동일 규약).
/// EXL3 디렉터리 경로는 재귀 합산 + 런타임 스크래치 가산(2026-10-03:
/// 디렉터리 metadata≈0으로 통과하던 구멍 — gsnap/배치 스크래치 할당이
/// 시스템 동결로 폭발한 사고의 근본 가드 결함).
fn model_bytes(p: &Path) -> u64 {
    // B7: 파트 정규화 후 회계 — part2/3 직접 지정도 전체 파트 합산.
    let p = &split_part1(p);
    // EXL3 디렉터리: 샤드 전체 합 + GPU 스크래치(yb×3 1.5GB + xtb/ah 1.3GB
    // + gframe 0.53GB + aframe 0.27GB + fframe/gsnap 0.18GB ≈ 3.7GB → 4GB 가산).
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
        return total.saturating_add(4u64 << 30);
    }
    let name = match p.file_name().and_then(|s| s.to_str()) {
        Some(n) => n.to_string(),
        None => return 0,
    };
    let mut total = p.metadata().map(|m| m.len()).unwrap_or(0);
    // 부분 파일이면 같은 접두의 모든 파트를 합산한다.
    if let Some(idx) = name.rfind("-00001-of-") {
        let dir = p.parent().map(Path::new).unwrap_or_else(|| Path::new("."));
        let prefix = &name[..idx];
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let n = e.file_name();
                let Some(n) = n.to_str() else { continue };
                if n.starts_with(prefix) && n.ends_with(".gguf") && n != name {
                    total += e.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
    }
    // PLE 테이블(per_layer_token_embd) 스트리밍 제외 — 4-split qwen4exp GGUF의
    // PLE 테이블(26.8GiB)은 mmap+ssd pread로 스트리밍(ple_table auto/ssd,
    // plans/111 W4c·§21-4)되어 상주 불요. ram 모드(28.8GB pin)는 30GB 체제에서
    // 선택 불가 — 상수 차감이 무해. 파일을 못 읽으면 0(보수적으로 과대 가드).
    // (2026-10-07: host 19.5GB에서 FN 기동 거부 — max-ctx 실사용 장벽 수리.)
    total = total.saturating_sub(ple_stream_bytes(p));
    total
}

/// GGUF 파트들에서 PLE 테이블(per_layer_token_embd) 바이트 합 — mmap 스트리밍
/// 되어 상주 불요한 테이블의 가드 차감용. GGUF 헤더 파싱 실패 시 0.
fn ple_stream_bytes(p: &Path) -> u64 {
    let p = &split_part1(p);
    let name = match p.file_name().and_then(|s| s.to_str()) {
        Some(n) => n.to_string(),
        None => return 0,
    };
    let Some(idx) = name.find("-of-") else {
        return 0;
    };
    let dir = p.parent().map(Path::new).unwrap_or_else(|| Path::new("."));
    let prefix = &name[..idx];
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let n = e.file_name();
            let Some(n) = n.to_str() else { continue };
            if !n.starts_with(prefix) || !n.ends_with(".gguf") {
                continue;
            }
            if let Ok(g) = llm170_gguf::GgufFile::open(&dir.join(n))
                && let Some(t) = g.find_tensor("per_layer_token_embd.weight")
                && let Some(nb) = t.nbytes()
            {
                total += nb;
            }
        }
    }
    total
}

/// 호스트 가용 메모리 (/proc/meminfo MemAvailable).
fn host_mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// 적재 시작 전 가드 - 서브커맨드 진입부에서 호출.
/// `gpu`가 참이면 VRAM 가용을 조회해 산식에 포함한다(B17: 조회 실패=거부).
/// `runtime`: 요청 런타임(Some("cuda")면 cuMemGetInfo 프로브, 그 외 hip).
/// 진단용 기본 모델 경로(단일 소스 — probes.rs·가드 표가 공유,
/// plans/129 A2/R1·A13). **[2026-10-08 임시 대체]** 구 GGUF/EXL3 자산이
/// 사용자 정리(GGUF·EXL3 탈락)로 삭제되어 W4A16 자산으로 임시 지정 —
/// hip/vk/exl3 프로브 표면은 후속 배치(plans/w4a16-cuda.md §5 B1/B2)에서
/// 삭제 예정이라 이 표도 함께 소멸한다.
pub const DEFAULT_FN_MODEL: &str =
    "/home/harsper/Desktop/workspace/models/Qwen3.8-Flash-Next-W4A16-FP8PLE";
pub const DEFAULT_27_MODEL: &str =
    "/home/harsper/Desktop/workspace/models/Qwen3.8-27B-W4A16-AutoRound";
pub const DEFAULT_Q35_MODEL: &str =
    "/home/harsper/Desktop/workspace/models/Qwen3.8-27B-W4A16-AutoRound";
pub const DEFAULT_EXL3_MODEL: &str =
    "/home/harsper/Desktop/workspace/models/Qwen3.8-27B-W4A16-AutoRound";

/// 무인자 실행 시 모델을 적재하는 프로브(가드 우회 폐쇄 — plans/129 A13).
/// (서브커맨드, 기본 경로) — --model/위치인자 부재 시 기본 경로로 가드한다.
pub const PROBE_DEFAULT_MODELS: &[(&str, &str)] = &[
    ("mmq-row-check", DEFAULT_27_MODEL),
    ("hip-dmmv-check", DEFAULT_27_MODEL),
    ("tile-row-check", DEFAULT_27_MODEL),
    ("hip-moe-dmmv-check", DEFAULT_FN_MODEL),
    ("mtp-load-check", DEFAULT_FN_MODEL),
    ("vk-frame-check", DEFAULT_FN_MODEL),
];

/// B19: 위치인자로 모델을 적재하는 GPU 프로브 접두 — 일반 위치인자 폴백에서
/// gpu=true(가드가 VRAM까지 계정). CPU 프로브(mod-check 등)는 폴백 gpu=false.
const POSITIONAL_GPU_PROBES: &[&str] = &["vk-", "hip", "exl3-", "cuda", "mmq-", "tile-", "rawhip"];

/// 가드 대상(plans/129 A2/R1) — 판정 결과.
pub struct GuardTarget {
    pub path: std::path::PathBuf,
    pub gpu: bool,
    /// 요청 런타임("cuda"|"hip"|"vulkan" — B6 프로브 선택용).
    pub runtime: Option<String>,
}

/// 가드 대상 판정 — main() 인라인에서 추출한 순수함수(plans/129 A2/R1).
/// 입력: 서브커맨드, --model 값, 정규화 백엔드(hip|vulkan→"gpu"), 런타임,
/// 위치인자. None = 가드 스킵(메타데이터 서브커맨드 또는 경로 부재 — 로더
/// 에러가 더 정확). 계약은 테이블 테스트(guard_target_cases)가 고정한다.
pub fn guard_target(
    sub: &str,
    model: Option<&str>,
    backend: Option<&str>,
    gpu_runtime: Option<&str>,
    rest: &[String],
) -> Option<GuardTarget> {
    // 메타데이터/행 판독만 읽는 서브커맨드 — 무게 미적재. dequant는 무게
    // 텐서 1행(≤수백KB)만 판독하므로 적재 계정 대상이 아니다(B2 검증 워크플로
    // 가드 우회가 아니라 계약 — 모델 상주 불가 기기에서도 판독 가능해야 한다).
    if matches!(sub, "tokenize") {
        return None;
    }
    let mut path = model.map(std::path::PathBuf::from);
    let mut gpu = backend == Some("gpu") || gpu_runtime.is_some();
    let first_pos = || {
        rest.iter()
            .find(|a| !a.starts_with("--"))
            .map(std::path::PathBuf::from)
    };
    if sub == "check" {
        gpu = true; // run_check의 백엔드 기본값이 gpu다.
        if path.is_none() {
            path = first_pos();
        }
    } else if sub == "w4a8-check" && path.is_none() {
        path = first_pos(); // args[0] 필수 — 로더 적재.
    }
    // exl3-* 프로브도 모델을 적재한다 — 첫 비플래그 인자(A13: 상대경로 우회를
    // 닫기 위해 과거 슬래시 조건 폐지, 2026-10-04 사고 재발 방지). 무인자면
    // EXL3 기본 아카이브로 가드(안전 방향 — 오탐은 가드 에러가 안내).
    if sub.starts_with("exl3-") {
        gpu = true;
        if path.is_none() {
            path = first_pos().or_else(|| Some(DEFAULT_EXL3_MODEL.into()));
        }
    }
    // A13: 기본 경로로 적재하는 무인자 프로브 — 경로표로 가드.
    if path.is_none()
        && let Some((_, def)) = PROBE_DEFAULT_MODELS.iter().find(|(c, _)| *c == sub)
    {
        path = Some(std::path::PathBuf::from(*def));
        gpu = true;
    }
    // B19: 나머지 위치인자 프로브(rawhip-check·vk-gemv-check 등)도 모델을
    // 적재한다 — 일반 폴백으로 가드. 라우트 추가·프로브 확장 시 무가드
    // 실적재 경로가 새로 생기는 것을 막는다. GPU 프로브 접두는 gpu=true.
    if path.is_none()
        && let Some(p) = first_pos()
    {
        path = Some(p);
        if POSITIONAL_GPU_PROBES.iter().any(|pre| sub.starts_with(pre)) {
            gpu = true;
        }
    }
    path.map(|path| GuardTarget {
        path,
        gpu,
        runtime: gpu_runtime.map(String::from),
    })
}

/// PLE 테이블(per_layer_token_embd) SSD 스테이징 차감 — 상주 계정에서 제외.
/// ple-ssd 모드(8GiB+ 테이블은 auto 정책상 SSD 행선)에선 테이블이 RAM/VRAM
/// 비상주(pread 요청 시 판독)이므로 model_bytes에 포함하면 과대계상 — 버퍼드
/// PLE(plans/135 long-ctx)가 페이지캐시를 쓰며 MemAvailable이 오르내리는
/// 지금은 가드 오탐의 직접 원인. 메타데이터만 저비용 판독(GGUF 헤더+텐서 표).
fn ple_ssd_deduction(model: &Path) -> u64 {
    const TENSOR: &str = "per_layer_token_embd.weight";
    const SSD_MIN: u64 = 8u64 << 30; // 8GiB+ 테이블만 SSD 행선으로 간주
    let dir = model
        .parent()
        .map(Path::new)
        .unwrap_or_else(|| Path::new("."));
    let name = model.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    if let Some(idx) = name.rfind("-00001-of-") {
        let prefix = &name[..idx];
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let n = e.file_name();
                let Some(n) = n.to_str() else { continue };
                if n.starts_with(prefix) && n.ends_with(".gguf") {
                    shards.push(dir.join(n));
                }
            }
            shards.sort();
        }
    } else {
        shards.push(model.to_path_buf());
    }
    for sh in &shards {
        let Ok(f) = llm170_gguf::GgufFile::open(sh) else {
            continue;
        };
        if let Some(sz) = f.find_tensor(TENSOR).and_then(|t| t.nbytes()) {
            // 스플릿 텐서는 단일 샤드에 온전히 존재 (GGUF v3 배치 규약).
            return if sz >= SSD_MIN { sz } else { 0 };
        }
    }
    0
}

pub fn preflight(model: &Path, gpu: bool, runtime: Option<&str>) -> Result<(), String> {
    let bytes = model_bytes(model).saturating_sub(ple_ssd_deduction(model));
    if bytes == 0 {
        return Ok(()); // 경로 오류는 로더의 에러가 더 정확하다 - 여기서는 통과
    }
    let vram = if gpu {
        // B6: 런타임별 프로브 — cuda는 cuMemGetInfo(rawcuda ffi), 그 외는
        // 기존 hip 프로브. 실패는 None → check의 B17 게이트가 거부한다.
        let probe = if runtime == Some("cuda") {
            llm170_backend_gpu::cuda_mem_free()
        } else {
            llm170_backend_gpu::gpu_mem_free()
        };
        match probe {
            Some((free, _total)) => Some(free),
            None => {
                eprintln!(
                    "# rsrc-guard: VRAM 조회 실패(runtime={}) — gpu 적재는 거부된다(B17)",
                    runtime.unwrap_or("default")
                );
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
    fn b17_gpu_load_requires_vram_measurement() {
        // B17 승인 정책(2026-10-08): gpu 적재에서 VRAM 미측정 = 거부.
        // 호스트가 넉넉해도(CUDA 기기 "VRAM unknown" 사고 재현 조건) 거부.
        assert!(check(17 * GIB, None, Some(200 * GIB), true).is_err());
        assert!(check(17 * GIB, None, None, true).is_err());
        // 측정되면 통과(용량 충분).
        assert!(check(17 * GIB, Some(20 * GIB), Some(30 * GIB), true).is_ok());
    }

    #[test]
    fn b7_split_part1_normalization() {
        // part2/3 경로 → 파트1 유도. 존재하지 않는 파트면 원본 유지.
        let p2 = Path::new("/nonexistent/model-00002-of-00003.gguf");
        assert_eq!(split_part1(p2), p2); // 파트1 부재 — 원본
        // 실존 스플릿(FN 3파트)으로 검증: part2 입력도 part1로 정규화.
        let fn_dir = "/home/yoon/models/qwen3.8-Flash-Next/UD-Q3_K_XL";
        let base = "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q3_K_XL";
        if std::path::Path::new(&format!(
            "{fn_dir}/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf"
        ))
        .exists()
        {
            let p = split_part1(Path::new(&format!(
                "{fn_dir}/Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf"
            )));
            assert!(p.to_string_lossy().contains("-00001-of-00003"));
        }
        // 단일 파일·비-GGUF 접미는 무변환.
        assert_eq!(split_part1(Path::new("/m/a.gguf")), Path::new("/m/a.gguf"));
        let _ = base;
    }

    /// guard_target 판정 표(plans/129 A2/R1 + B19 확대) — 서브커맨드×인자 형태
    /// 계약을 고정한다. 무가드 적재 프로브 폐쇄(A13)·exl3 상대경로 우회 폐쇄·
    /// 위치인자 프로브 일반 폴백(B19) 포함.
    #[test]
    fn guard_target_cases() {
        use super::{DEFAULT_27_MODEL, DEFAULT_EXL3_MODEL, PROBE_DEFAULT_MODELS, guard_target};
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        // serve/infer류: --model + 백엔드 → (경로, gpu, 런타임)
        let g = guard_target(
            "serve",
            Some("/m/a.gguf"),
            Some("gpu"),
            Some("hip"),
            &s(&[]),
        );
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu, g.runtime)),
            Some(("/m/a.gguf".into(), true, Some("hip".into())))
        );
        // cpu 백엔드 → gpu=false
        let g = guard_target("infer", Some("/m/a.gguf"), Some("cpu"), None, &s(&[]));
        assert_eq!(g.map(|g| g.gpu), Some(false));
        // check: 첫 위치인자 + gpu 강제
        let g = guard_target("check", None, None, None, &s(&["/m/a.gguf", "--quick"]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/a.gguf".into(), true))
        );
        // w4a8-check: 위치인자 (gpu 강제 없음)
        let g = guard_target("w4a8-check", None, None, None, &s(&["/m/a.gguf"]));
        assert_eq!(
            g.map(|g| g.path.to_str().unwrap().to_string()),
            Some("/m/a.gguf".to_string())
        );
        // exl3 프로브: 상대경로(무슬래시)도 우회 없이 가드(A13)
        let g = guard_target("exl3-hip-decode", None, None, None, &s(&["relmodel"]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("relmodel".into(), true))
        );
        // exl3 무인자: 기본 EXL3 아카이브로 가드
        let g = guard_target("exl3-hip-attn", None, None, None, &s(&[]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some((DEFAULT_EXL3_MODEL.to_string(), true))
        );
        // 무인자 적재 프로브: 기본 경로표(A13)
        let g = guard_target("mmq-row-check", None, None, None, &s(&[]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some((DEFAULT_27_MODEL.to_string(), true))
        );
        let g = guard_target("vk-frame-check", None, None, None, &s(&[]));
        assert_eq!(
            g.map(|g| g.path.to_str().unwrap().to_string()),
            Some(super::DEFAULT_FN_MODEL.to_string())
        );
        // B19: 위치인자 GPU 프로브 일반 폴백 — rawhip-check/vk-gemv-check 무가드 폐쇄.
        let g = guard_target("rawhip-check", None, None, None, &s(&["/m/27b.gguf"]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/27b.gguf".into(), true))
        );
        let g = guard_target("vk-gemv-check", None, None, None, &s(&["/m/27b.gguf"]));
        assert_eq!(g.map(|g| g.gpu), Some(true));
        // CPU 프로브(mod-check) 위치인자 — gpu=false 폴백.
        let g = guard_target("mod-check", None, None, None, &s(&["/m/a.gguf"]));
        assert_eq!(g.map(|g| g.gpu), Some(false));
        // W4A16 로더 프로브(plans/w4a16-cuda.md §1) — 위치인자 폴백(gpu=false).
        let g = guard_target("w4a16-load", None, None, None, &s(&["/m/w4a16"]));
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/w4a16".into(), false))
        );
        // 메타·행 판독 서브커맨드 → None
        assert!(guard_target("tokenize", Some("/m/a.gguf"), None, None, &s(&[])).is_none());
        // 무모델 로딩 창구 → None(로더/CLI 에러가 더 정확 — bench는 --model required)
        assert!(guard_target("bench", None, Some("gpu"), None, &s(&[])).is_none());
        assert!(guard_target("infer", None, None, None, &s(&[])).is_none());
        assert!(guard_target("serve", None, None, None, &s(&[])).is_none());
        // perplexity --model → 가드(전 모델 적재)
        let g = guard_target("perplexity", Some("/m/a.gguf"), Some("cpu"), None, &s(&[]));
        assert_eq!(g.map(|g| g.gpu), Some(false));
        // 표 무결성: 기본 경로 전부 실존(부서진 기본 경로 = 무가드보다 못한 오탐)
        for (_, def) in PROBE_DEFAULT_MODELS {
            assert!(std::path::Path::new(def).exists(), "기본 경로 부재: {def}");
        }
    }
}
