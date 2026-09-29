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
/// 값 맵 — 1회 스냅샷(plans/107 W2). 핫패스 판독 잠금·조회 원자화.
static VALUES: std::sync::LazyLock<HashMap<String, String>> = std::sync::LazyLock::new(|| {
    std::env::vars_os()
        .filter_map(|(k, v)| Some((k.to_str()?.to_string(), v.to_str()?.to_string())))
        .collect()
});

/// 이름 존재 여부 — `var_os(name).is_some()` 대응(캐시형).
pub fn on(name: &str) -> bool {
    VALUES.contains_key(name)
}

/// `== "1"` 엄격 옵트인 — `var(name).map(|v| v == "1").unwrap_or(false)` 대응.
pub fn eq1(name: &str) -> bool {
    VALUES.get(name).is_some_and(|v| v == "1")
}

/// `!= "0"` 기본 ON — `var(name).map(|v| v != "0").unwrap_or(true)` 대응.
/// 키 부재 = 기본 ON(true). is_some_and는 부재 시 false를 돌려 기본
/// ON 게이트 전체를 뒤집는 결함이었다(107 회귀, 원장 104).
pub fn ne0(name: &str) -> bool {
    VALUES.get(name).is_none_or(|v| v != "0")
}

/// 원시 값 — 수치 파싱 등 특수 호출부용.
pub fn val(name: &str) -> Option<&str> {
    VALUES.get(name).map(|v| v.as_str())
}

/// 환경변수 존재 여부 판독 + 캐시 + 레지스트리 등록.
/// 캐시 키는 환경변수명 그대로. 첫 호출 시 1회 판독 후 고정.
pub fn env_on(name: &str) -> bool {
    // 캐시 조회
    if let Ok(r) = registry().lock()
        && let Some(info) = r.get(name)
    {
        return info.value;
    }
    // 첫 판독
    let v = on(name);
    // 등록 (기존 등록이 있으면 덮어쓰지 않음 — 진단용)
    if let Ok(mut r) = registry().lock() {
        r.entry(name.to_string())
            .or_insert(FlagInfo { value: v, desc: "" });
    }
    v
}

/// 스냅샷↔라이브 동치 검사 (plans/108 P1) — 현재 환경의 LLM170_ 키 전수에
/// 대해 4개 의미론(on/eq1/ne0/val)을 라이브 getenv 판정과 독립 대조하고
/// 레지스트리 캐시값도 재검한다. 불일치 목록(빈 벡터 = 정상).
/// 원장 104(ne0 부재키 결함)류 회귀를 게이트 전에 포착한다.
pub fn env_check() -> Vec<String> {
    let mut bad = Vec::new();
    for (ko, vo) in std::env::vars_os() {
        let (Some(k), Some(v)) = (ko.to_str(), vo.to_str()) else {
            continue;
        };
        if !k.starts_with("LLM170_") {
            continue;
        }
        let live_some = std::env::var_os(k).is_some();
        if on(k) != live_some {
            bad.push(format!("{k}: on()={} live={live_some}", on(k)));
        }
        let live_eq1 = v == "1";
        if eq1(k) != live_eq1 {
            bad.push(format!("{k}: eq1()={} live={live_eq1}", eq1(k)));
        }
        let live_ne0 = v != "0";
        if ne0(k) != live_ne0 {
            bad.push(format!("{k}: ne0()={} live={live_ne0}", ne0(k)));
        }
        if val(k) != Some(v) {
            bad.push(format!("{k}: val()={:?} live={v:?}", val(k)));
        }
    }
    if let Ok(r) = registry().lock() {
        for (k, info) in r.iter() {
            if info.value != on(k) {
                bad.push(format!("{k}: registry={} snapshot={}", info.value, on(k)));
            }
        }
    }
    bad
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

    #[test]
    fn absent_key_contracts() {
        // 원장 104: 부재키 의미론 — ne0는 true(기본 ON), on/eq1는 false,
        // val은 None. is_some_and로 되돌리면 이 테스트가 즉시 잡는다
        // (108 P1 결함주입 검증 완료).
        const K: &str = "_LLM170_ABSENT_PROBE_";
        assert!(ne0(K), "ne0 absent-key must default ON");
        assert!(!on(K));
        assert!(!eq1(K));
        assert!(val(K).is_none());
    }
}
