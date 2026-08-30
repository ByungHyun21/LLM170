//! LLM170 자체 경량 프로파일러.
//!
//! 설계 요구사항 (AGENTS.md): debug 빌드에서 모든 구간을 세세히 계측,
//! release에서는 zero-cost. `profile` feature로 release 계측도 가능.
//!
//! v0: 이름 기반 span 누적(횟수/합/평균/최대) + 리포트 출력.
//! 중첩 구조·스레드별 분리·커널 단위 확장은 엔진 코어가 자리잡은 뒤 추가.
//!
//! 사용 규칙: `profile_span!`은 **스코프당 1개**. 같은 함수에서 순차 구간을 잴 때는
//! 블록 `{ ... }`으로 스코프를 분리한다 (가드가 스코프 끝에 Drop되므로).

/// 계측 활성 조건: debug 빌드 또는 `profile` feature.
#[cfg(any(debug_assertions, feature = "profile"))]
mod imp {
    use std::sync::Mutex;
    use std::time::Instant;

    const MAX_EVENTS: usize = 1 << 22; // 4M span — 초과 시 무음 누락 (OOM 방지)

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
            if let Ok(mut reg) = registry().lock() {
                if reg.len() < MAX_EVENTS {
                    reg.push((self.name, elapsed));
                }
            }
        }
    }

    /// span 시작. 반환값을 스코프 끝까지 유지할 것.
    pub fn span(name: &'static str) -> SpanGuard {
        SpanGuard { name, start: Instant::now() }
    }

    /// 누적 리포트: 총 소요 기준 내림차순. 계측된 게 없으면 None.
    pub fn report() -> Option<String> {
        let reg = registry().lock().ok()?;
        if reg.is_empty() {
            return None;
        }
        let mut agg: std::collections::HashMap<&'static str, Aggregate> =
            std::collections::HashMap::new();
        for (name, ns) in reg.iter() {
            let a = agg.entry(name).or_insert(Aggregate { count: 0, total_ns: 0, max_ns: 0 });
            a.count += 1;
            a.total_ns += ns;
            a.max_ns = a.max_ns.max(*ns);
        }
        let mut rows: Vec<(&'static str, &Aggregate)> = agg.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

        let mut out = String::from("=== llm170 profile ===\n");
        out.push_str(&format!("{:<44} {:>9} {:>12} {:>12} {:>12}\n",
            "span", "count", "total", "mean", "max"));
        for (name, a) in rows {
            out.push_str(&format!("{:<44} {:>9} {:>12} {:>12} {:>12}\n",
                name, a.count, fmt_ns(a.total_ns), fmt_ns(a.total_ns / a.count as u128), fmt_ns(a.max_ns)));
        }
        out.push_str(&format!("events: {}\n", reg.len()));
        Some(out)
    }

    /// 계측 버퍼 초기화 (재측정 시작 시).
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
    pub fn span(_name: &'static str) -> SpanGuard { SpanGuard }
    pub fn report() -> Option<String> { None }
    pub fn reset() {}
}

pub use imp::{report, reset, span};

#[macro_export]
macro_rules! profile_span {
    ($name:literal) => {
        #[allow(unused_variables)]
        let _llm170_span = $crate::span($name);
    };
}
