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
//! - 측정 불능(비리눅스/프로브 실패)인 항목은 거부하지 않고 경고만 낸다
//!   - 가드가 오탐으로 정상 실행을 막는 일이 없어야 하기 때문.
//!
//! 킬스위치 폐지(2026-10-03, 사용자 지시): MemAvailable 기반 계정이 회수
//! 가능 캐시를 이미 반영하므로 오탐 근원이 아니며, 우회 env는 실질 무장
//! 해제(시스템 동결 사고 재발 위험) — 가드는 상시 동작.

use std::path::Path;

const REQ_SLACK: f64 = 1.10;
const HOST_USABLE: f64 = 0.85;

/// 순수 판정 함수 - 단위테스트 대상.
pub fn check(
    model_bytes: u64,
    vram_free: Option<u64>,
    host_avail: Option<u64>,
) -> Result<(), String> {
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
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
        // 측정 불능 - 오탐 방지 위해 통과 (호출부에서 경고).
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

/// 스플릿 GGUF 전체 파트 크기 합 - `-00001-of-00004.gguf` 패턴(Model4::load와 동일 규약).
/// EXL3 디렉터리 경로는 재귀 합산 + 런타임 스크래치 가산(2026-10-03:
/// 디렉터리 metadata≈0으로 통과하던 구멍 — gsnap/배치 스크래치 할당이
/// 시스템 동결로 폭발한 사고의 근본 가드 결함).
fn model_bytes(p: &Path) -> u64 {
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
/// `gpu`가 참이면 VRAM 가용을 조회해 산식에 포함한다.
/// 진단용 기본 모델 경로(단일 소스 — probes.rs·가드 표가 공유,
/// plans/129 A2/R1·A13).
pub const DEFAULT_FN_MODEL: &str =
    "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
pub const DEFAULT_27_MODEL: &str = "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf";
pub const DEFAULT_Q35_MODEL: &str = "/home/yoon/models/qwen3.8-27b/q35work.gguf";
pub const DEFAULT_EXL3_MODEL: &str = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";

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

/// 가드 대상(plans/129 A2/R1) — 판정 결과.
pub struct GuardTarget {
    pub path: std::path::PathBuf,
    pub gpu: bool,
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
    // 메타데이터만 읽는 서브커맨드 — 무게 미적재.
    if matches!(sub, "gguf-dump" | "tokenize") {
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
    path.map(|path| GuardTarget { path, gpu })
}

/// PLE 테이블(per_layer_token_embd) SSD 스테이징 차감 — 상주 계정에서 제외.
/// ple-ssd 모드(8GiB+ 테이블은 auto 정책상 SSD 행선)에선 테이블이 RAM/VRAM
/// 비상주(pread 요구 시 판독)이므로 model_bytes에 포함하면 과대계상 — 버퍼드
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

pub fn preflight(model: &Path, gpu: bool) -> Result<(), String> {
    let bytes = model_bytes(model).saturating_sub(ple_ssd_deduction(model));
    if bytes == 0 {
        return Ok(()); // 경로 오류는 로더의 에러가 더 정확하다 - 여기서는 통과
    }
    let vram = if gpu {
        match llm170_backend_gpu::gpu_mem_free() {
            Some((free, _total)) => Some(free),
            None => {
                eprintln!("# rsrc-guard: VRAM 조회 실패 - VRAM 항목 없이 판정한다");
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
    check(bytes, vram, host)
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
                Some(26 * GIB)
            )
            .is_ok()
        );
    }

    #[test]
    fn refuses_double_resident() {
        // 다른 서버가 상주해 VRAM~=0, 호스트~=5GiB 남은 이중 적재 사고 조건.
        assert!(check(103_700 * (1 << 20), Some(GIB), Some(5 * GIB)).is_err());
    }

    #[test]
    fn passes_27b_alongside_resident() {
        assert!(check(17 * GIB, Some(80 * GIB), Some(22 * GIB)).is_ok());
    }

    #[test]
    fn unknown_measurements_pass_with_warning() {
        assert!(check(103_700 * (1 << 20), None, None).is_ok());
        // 호스트 불능 + VRAM 만으로 모델×슬랙을 못 덮으면 거부 — 가드의 목적
        // (이중 적재 동결 방지)상 이것이 옳다(2026-09-16: 통과 기대는 산식과
        // 모순되어 수정).
        assert!(check(103_700 * (1 << 20), Some(90 * GIB), None).is_err());
    }

    /// guard_target 판정 표(plans/129 A2/R1) — 서브커맨드×인자 형태 계약을
    /// 고정한다. 무가드 적재 프로브 폐쇄(A13)·exl3 상대경로 우회 폐쇄 포함.
    #[test]
    fn guard_target_cases() {
        use super::{DEFAULT_27_MODEL, DEFAULT_EXL3_MODEL, PROBE_DEFAULT_MODELS, guard_target};
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        // serve/infer류: --model + 백엔드 → (경로, gpu)
        let g = guard_target(
            "serve",
            Some("/m/a.gguf"),
            Some("gpu"),
            Some("hip"),
            &s(&[]),
        );
        assert_eq!(
            g.map(|g| (g.path.to_str().unwrap().to_string(), g.gpu)),
            Some(("/m/a.gguf".into(), true))
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
        // 메타데이터 서브커맨드·경로 부재 → None
        assert!(guard_target("gguf-dump", None, None, None, &s(&[])).is_none());
        assert!(guard_target("tokenize", Some("/m/a.gguf"), None, None, &s(&[])).is_none());
        assert!(guard_target("bench", None, Some("gpu"), None, &s(&[])).is_none());
        // 표 무결성: 기본 경로 전부 실존(부서진 기본 경로 = 무가드보다 못한 오탐)
        for (_, def) in PROBE_DEFAULT_MODELS {
            assert!(std::path::Path::new(def).exists(), "기본 경로 부재: {def}");
        }
    }
}
