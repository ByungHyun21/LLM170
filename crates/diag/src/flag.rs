//! flag — 환경변수 캐시형 판독 (plans/82 §3).
//!
//! rawhip::env_on의 진행형: 이 크레이트가 소유하는 변수는 여기 등록·캐시하고,
//! backend-gpu 변수는 위임으로 자동 등록한다.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

/// 캐시 엔트리 — (값, 설명).
pub struct FlagInfo {
    pub value: bool,
    pub desc: &'static str,
}

fn registry() -> &'static Mutex<HashMap<String, FlagInfo>> {
    static R: OnceLock<Mutex<HashMap<String, FlagInfo>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 환경변수 존재 여부 판독 + 캐시 + 레지스트리 등록.
/// 캐시 키는 환경변수명 그대로. 첫 호출 시 1회 판독 후 고정.
pub fn env_on(name: &str) -> bool {
    // 캐시 조회
    if let Ok(r) = registry().lock() {
        if let Some(info) = r.get(name) {
            return info.value;
        }
    }
    // 첫 판독
    let v = std::env::var_os(name).is_some();
    // 등록 (기존 등록이 있으면 덮어쓰지 않음 — 진단용)
    if let Ok(mut r) = registry().lock() {
        r.entry(name.to_string())
            .or_insert(FlagInfo { value: v, desc: "" });
    }
    v
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_on_caches() {
        // 실제 환경변수가 없는 이름으로 테스트
        let name = "_DIAG_TEST_NONEXISTENT_";
        assert!(!env_on(name));
        assert!(!env_on(name)); // 캐시에서 반환
    }
}
