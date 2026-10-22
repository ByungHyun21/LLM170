
use std::sync::atomic::{AtomicUsize, Ordering};

const NAMES: [&str; 0] = [];

static COUNTS: [AtomicUsize; NAMES.len()] = [const { AtomicUsize::new(0) }; NAMES.len()];

fn idx(name: &str) -> Option<usize> {
    NAMES.iter().position(|n| *n == name)
}

pub fn incr(name: &str) -> usize {
    match idx(name) {
        Some(i) => COUNTS[i].fetch_add(1, Ordering::Relaxed) + 1,
        None => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!("diag fb: 미등록 카운터 '{name}' — NAMES에 등록할 것");
            });
            0
        }
    }
}

pub fn count(name: &str) -> usize {
    idx(name).map_or(0, |i| COUNTS[i].load(Ordering::Relaxed))
}

pub fn report() -> String {
    let mut out = String::new();
    for (i, n) in NAMES.iter().enumerate() {
        let c = COUNTS[i].load(Ordering::Relaxed);
        if c > 0 {
            out.push_str(&format!("{n} {c}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_name_warns_and_stays_zero() {
        assert_eq!(incr("no-such-counter"), 0);
        assert_eq!(count("no-such-counter"), 0);
    }

    #[test]
    fn empty_registry_reports_nothing() {
        assert!(report().is_empty());
    }
}
