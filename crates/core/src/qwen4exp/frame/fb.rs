//! 폴백 카운터 (plans/107 W8) — GPU→CPU/값경로 폴백이 조용히 발산 원인을
//! 가리는 일을 막는다. eprintln 로그는 ONCE라 반복 폴백이 보이지 않는다;
//! 카운터는 전수를 센다. `llm170 diag fb`로 관측.

use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(usize)]
pub enum Id {
    EmbQ8g = 0,
    PleGgpu = 1,
    QsaDevsel = 2,
    QsaDevselMt = 3,
    QsaIdxpool = 4,
    QsaAttn = 5,
    FrameCreate = 6,
    FrameCreateNp = 7,
    MtpDraft = 8,
}

const NAMES: [&str; 9] = [
    "emb-q8g",
    "ple-ggpu",
    "qsa-devsel",
    "qsa-devsel-mt",
    "qsa-idxpool",
    "qsa-attn",
    "frame-create",
    "frame-create-np",
    "mtp-draft",
];

static COUNTS: [AtomicUsize; 9] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

pub fn incr(id: Id) {
    COUNTS[id as usize].fetch_add(1, Ordering::Relaxed);
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
    fn incr_bumps_named_counter() {
        let before = COUNTS[Id::QsaAttn as usize].load(Ordering::Relaxed);
        incr(Id::QsaAttn);
        incr(Id::QsaAttn);
        assert_eq!(
            COUNTS[Id::QsaAttn as usize].load(Ordering::Relaxed),
            before + 2
        );
        assert!(report().contains("qsa-attn"));
    }
}
