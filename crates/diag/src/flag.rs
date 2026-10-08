//! flag — 환경변수 캐시형 판독 (plans/82 §3).
//!
//! 프로세스 기동 시 1회 스냅샷(VALUES) 후 불변 — 핫패스 판독의 잠금·조회
//! 원자화. 등록 레지스트리(env_on)는 프로덕션 호출 0으로 plans/109 P1에서
//! 삭제했다(구 백엔드는 자체 캐시 env_on 사용).
use std::collections::HashMap;

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

/// `존재 && 값 != "0"` — `var(name).is_ok_and(|v| v != "0")` 대응.
/// 부재 시 false(ne0와 반대 — 옵트인 값 게이트: LLM170_FRAME35·
/// LLM170_FRAME 등 "설정돼 있고 0이 아니면 ON" 관례). A6(plans/129).
pub fn on_nonzero(name: &str) -> bool {
    VALUES.get(name).is_some_and(|v| v != "0")
}

/// 스냅샷↔라이브 동치 검사 (plans/108 P1) — 현재 환경의 LLM170_ 키 전수에
/// 대해 4개 의미론(on/eq1/ne0/val)을 라이브 getenv 판정과 독립 대조한다.
/// 불일치 목록(빈 벡터 = 정상).
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
        let live_nz = v != "0";
        if on_nonzero(k) != live_nz {
            bad.push(format!(
                "{k}: on_nonzero()={} live={live_nz}",
                on_nonzero(k)
            ));
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

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
