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
//! 킬스위치: LLM170_NO_RSRC_GUARD=1.

use std::path::Path;

const REQ_SLACK: f64 = 1.10;
const HOST_USABLE: f64 = 0.85;

/// 순수 판정 함수 - 단위테스트 대상.
pub fn check(model_bytes: u64, vram_free: Option<u64>, host_avail: Option<u64>) -> Result<(), String> {
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
             Another inference process may be resident - free GPU/RAM and retry. \
             (override: LLM170_NO_RSRC_GUARD=1)",
            gib(required),
            gib(capacity),
            vram_free.map(gib).unwrap_or_else(|| "unknown".into()),
            host_avail.map(gib).unwrap_or_else(|| "unknown".into()),
        ));
    }
    Ok(())
}

/// 스플릿 GGUF 전체 파트 크기 합 - `-00001-of-00004.gguf` 패턴(Model4::load와 동일 규약).
fn model_bytes(p: &Path) -> u64 {
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
pub fn preflight(model: &Path, gpu: bool) -> Result<(), String> {
    if std::env::var_os("LLM170_NO_RSRC_GUARD").is_some() {
        return Ok(());
    }
    let bytes = model_bytes(model);
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
        assert!(check(103_700 * (1 << 20), Some(95 * GIB + 768 * (1 << 20)), Some(26 * GIB)).is_ok());
    }

    #[test]
    fn refuses_double_resident() {
        // 다른 서버가 상주해 VRAM~=0, 호스트~=5GiB 남은 이중 적재 사고 조건.
        assert!(check(103_700 * (1 << 20), Some(1 * GIB), Some(5 * GIB)).is_err());
    }

    #[test]
    fn passes_27b_alongside_resident() {
        assert!(check(17 * GIB, Some(80 * GIB), Some(22 * GIB)).is_ok());
    }

    #[test]
    fn unknown_measurements_pass_with_warning() {
        assert!(check(103_700 * (1 << 20), None, None).is_ok());
        assert!(check(103_700 * (1 << 20), Some(90 * GIB), None).is_ok());
    }
}
