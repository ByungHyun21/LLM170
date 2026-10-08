//! dump — `LLM170_DUMP` 통합 진단 키 프론트엔드.
//!
//! 콤마 키 목록 하나로 진단 스위치를 판정한다:
//! `LLM170_DUMP="debug_layers,top2,srv_time,wall_time,..."`.
//! 개별 필드 없는 키는 `dump::key(k)`로 조회(자유 확장).
//! 1회 파싱(LazyLock) — 판독 비용은 원자 조회 1회.

/// 파싱된 덤프 옵션 — 정적 싱글턴.
#[derive(Debug, Default)]
pub struct DumpOpts {
    /// top2 — greedy 스텝 상위2 토큰·마진 덤프(근접타이 실증용).
    pub top2: bool,
    /// 통합 진단 키 집합 — LLM170_DUMP CSV 멤버 전체.
    keys: std::collections::HashSet<String>,
}

impl DumpOpts {
    /// 통합 진단 키 조회 — `LLM170_DUMP=key,...` 멤버 판정.
    pub fn key(&self, k: &str) -> bool {
        self.keys.contains(k)
    }
}

static OPTS: std::sync::LazyLock<DumpOpts> = std::sync::LazyLock::new(|| {
    let mut o = DumpOpts::default();
    let Some(v) = std::env::var("LLM170_DUMP").ok() else {
        return o;
    };
    for key in v.split(',') {
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        if key == "top2" {
            o.top2 = true;
        }
        o.keys.insert(key.to_string());
    }
    o
});

/// 통합 옵션 접근.
pub fn opts() -> &'static DumpOpts {
    &OPTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_off() {
        // LLM170_DUMP 미설정 환경에서는 top2가 꺼져야 한다.
        if std::env::var_os("LLM170_DUMP").is_none() {
            assert!(!opts().top2);
        }
    }
}
