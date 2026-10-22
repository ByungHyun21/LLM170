use std::collections::HashMap;

static VALUES: std::sync::LazyLock<HashMap<String, String>> = std::sync::LazyLock::new(|| {
    std::env::vars_os()
        .filter_map(|(k, v)| Some((k.to_str()?.to_string(), v.to_str()?.to_string())))
        .collect()
});

pub fn on(name: &str) -> bool {
    VALUES.contains_key(name)
}

pub fn eq1(name: &str) -> bool {
    VALUES.get(name).is_some_and(|v| v == "1")
}

pub fn ne0(name: &str) -> bool {
    VALUES.get(name).is_none_or(|v| v != "0")
}

pub fn val(name: &str) -> Option<&str> {
    VALUES.get(name).map(|v| v.as_str())
}

pub fn on_nonzero(name: &str) -> bool {
    VALUES.get(name).is_some_and(|v| v != "0")
}

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
        const K: &str = "_LLM170_ABSENT_PROBE_";
        assert!(ne0(K), "ne0 absent-key must default ON");
        assert!(!on(K));
        assert!(!eq1(K));
        assert!(val(K).is_none());
    }
}
