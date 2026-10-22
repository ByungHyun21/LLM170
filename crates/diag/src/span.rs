
#[cfg(any(debug_assertions, feature = "profile"))]
mod imp {
    use std::sync::Mutex;
    use std::time::Instant;

    const MAX_EVENTS: usize = 1 << 22;

    #[derive(Debug)]
    struct Aggregate {
        count: u64,
        total_ns: u128,
        max_ns: u128,
    }

    fn registry() -> &'static Mutex<Vec<(&'static str, u128)>> {
        static REG: std::sync::LazyLock<Mutex<Vec<(&'static str, u128)>>> =
            std::sync::LazyLock::new(|| Mutex::new(Vec::new()));
        &REG
    }

    pub struct SpanGuard {
        name: &'static str,
        start: Instant,
    }

    impl Drop for SpanGuard {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed().as_nanos();
            if let Ok(mut reg) = registry().lock()
                && reg.len() < MAX_EVENTS
            {
                reg.push((self.name, elapsed));
            }
        }
    }

    pub fn span(name: &'static str) -> SpanGuard {
        SpanGuard {
            name,
            start: Instant::now(),
        }
    }

    pub fn report() -> Option<String> {
        let reg = registry().lock().ok()?;
        if reg.is_empty() {
            return None;
        }
        let mut agg: std::collections::HashMap<&'static str, Aggregate> =
            std::collections::HashMap::new();
        for (name, ns) in reg.iter() {
            let a = agg.entry(name).or_insert(Aggregate {
                count: 0,
                total_ns: 0,
                max_ns: 0,
            });
            a.count += 1;
            a.total_ns += ns;
            a.max_ns = a.max_ns.max(*ns);
        }
        let mut rows: Vec<(&'static str, &Aggregate)> = agg.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by_key(|b| std::cmp::Reverse(b.1.total_ns));

        let mut out = String::from("=== llm170 profile ===\n");
        out.push_str(&format!(
            "{:<44} {:>9} {:>12} {:>12} {:>12}\n",
            "span", "count", "total", "mean", "max"
        ));
        for (name, a) in rows {
            out.push_str(&format!(
                "{:<44} {:>9} {:>12} {:>12} {:>12}\n",
                name,
                a.count,
                fmt_ns(a.total_ns),
                fmt_ns(a.total_ns / a.count as u128),
                fmt_ns(a.max_ns)
            ));
        }
        out.push_str(&format!("events: {}\n", reg.len()));
        Some(out)
    }

    pub fn reset() {
        if let Ok(mut reg) = registry().lock() {
            reg.clear();
        }
    }

    fn fmt_ns(ns: u128) -> String {
        if ns >= 1_000_000_000 {
            format!("{:.3}s", ns as f64 / 1e9)
        } else if ns >= 1_000_000 {
            format!("{:.3}ms", ns as f64 / 1e6)
        } else if ns >= 1_000 {
            format!("{:.3}us", ns as f64 / 1e3)
        } else {
            format!("{}ns", ns)
        }
    }
}

#[cfg(not(any(debug_assertions, feature = "profile")))]
mod imp {
    pub struct SpanGuard;
    pub fn span(_name: &'static str) -> SpanGuard {
        SpanGuard
    }
    pub fn report() -> Option<String> {
        None
    }
    pub fn reset() {}
}

pub use imp::{report, reset, span};

#[macro_export]
macro_rules! profile_span {
    ($name:literal) => {
        #[allow(unused_variables)]
        let _llm170_span = $crate::span::span($name);
    };
}

#[cfg(all(test, any(debug_assertions, feature = "profile")))]
mod tests {
    use super::*;

    #[test]
    fn span_accumulates() {
        reset();
        {
            let _g = span("t");
        }
        let rep = report().unwrap();
        assert!(rep.contains("t"));
        assert!(rep.contains("llm170 profile"));
        reset();
        assert!(report().is_none());
    }
}
