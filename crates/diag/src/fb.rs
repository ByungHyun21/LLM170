//! fb — 폴백 카운터 공유 원장 (plans/107 W8 → plans/129 A5).
//!
//! GPU→CPU/경로 폴백이 조용히 발산 원인을 가리는 일을 막는다. eprintln
//! 로그는 ONCE라 반복 폴백이 보이지 않는다; 카운터는 전수를 센다.
//!
//! [2026-10-08 단일 트랙] 등록 표는 비어 있다 — 구 GPU 경로·구 아키텍처
//! 삭제로 발화 지점이 전부 사라졌다. W2/W3(W4A16 CUDA
//! 디코더)에서 폴백 지점이 생기면 여기에 이름을 등록한다. 이름 공간은
//! dump.rs 키와 동일 방식(단일 고정 표)으로 여기가 단일 진실 공급원이다.
//!
//! 관측 경로 2개:
//! - `llm170 diag fb` — 독립 프로세스(원장·0건 확인용).
//! - 실측 프로세스(infer·bench·프로브 등) 종료 시 main이 [fb] 누계를
//!   stderr에 출력 — 카운터는 프로세스 로컬이라 폴백이 일어난 바로 그
//!   프로세스의 출력이 유일한 전수 관측면이다(A5).

use std::sync::atomic::{AtomicUsize, Ordering};

/// 등록 카운터 이름 — 여기 없는 이름 incr은 프로그래밍 에러(경고 로그).
const NAMES: [&str; 0] = [];

static COUNTS: [AtomicUsize; NAMES.len()] = [const { AtomicUsize::new(0) }; NAMES.len()];

fn idx(name: &str) -> Option<usize> {
    NAMES.iter().position(|n| *n == name)
}

/// 카운터 증가 — 증가 후 값을 반환("누적 n회" 로그용).
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

/// 현재 값(관측용).
pub fn count(name: &str) -> usize {
    idx(name).map_or(0, |i| COUNTS[i].load(Ordering::Relaxed))
}

/// 0이 아닌 카운터만 `name count` 한 줄씩. 전부 0이면 빈 문자열.
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
