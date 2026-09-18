//! dump — `LLM170_DUMP` 통합 진단 덤프 프론트엔드 (plans/83 C4).
//!
//! 콤마 키 목록 하나로 산재한 덤프 플래그를 대체한다:
//!
//! ```text
//! LLM170_DUMP="checksum,rows:L1.gdn_ar;L2.moe_sc,row0full,bufhash,moe"
//! ```
//!
//! (`rows:` 태그 구분은 `;` — 키 구분 `,` 와 충돌 회피)
//! - `checksum`  — 프레임 체크섬 라인(`[npck]`, 구 LLM170_NP_CHECKSUM)
//! - `rows:<tags>` — 지정 태그의 행별 비트 표본(`[nprd]`, 구 LLM170_NP_ROWS)
//! - `row0full` — 태그 버퍼 첫 행 전체 헥스(`[npr0]`, 구 LLM170_NP_ROW0FULL)
//! - `bufhash`  — 버퍼 FNV 해시(`[npbh]`, 구 LLM170_NP_BUFHASH)
//! - `moe`      — MoE 그룹 GEMM 입력 해시·덤프(구 LLM170_MOE_DUMP)
//!
//! 1회 파싱(LazyLock) — 런치패스 비용은 원자 판독 1회.

/// 파싱된 덤프 옵션 — 정적 싱글턴.
#[derive(Debug, Default)]
pub struct DumpOpts {
    pub checksum: bool,
    pub rows: Vec<String>,
    pub row0full: bool,
    pub bufhash: bool,
    pub moe: bool,
}

impl DumpOpts {
    /// 태그가 rows 지정에 포함되는지.
    pub fn row_on(&self, tag: &str) -> bool {
        self.rows.iter().any(|x| x == tag)
    }
}

static OPTS: std::sync::LazyLock<DumpOpts> = std::sync::LazyLock::new(|| {
    let mut o = DumpOpts::default();
    let Some(v) = std::env::var("LLM170_DUMP").ok() else {
        return o;
    };
    for key in v.split(',') {
        let key = key.trim();
        if let Some(tags) = key.strip_prefix("rows:") {
            o.rows = tags.split(';').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
        } else if key == "row0full" {
            o.row0full = true;
        } else if key == "bufhash" {
            o.bufhash = true;
        } else if key == "moe" {
            o.moe = true;
        } else if key == "checksum" {
            o.checksum = true;
        }
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
        // LLM170_DUMP 미설정 환경에서는 전부 꺼져야 한다
        // (테스트 프로세스가 이 값을 설정하지 않는 한)
        if std::env::var_os("LLM170_DUMP").is_none() {
            let o = opts();
            assert!(!o.checksum && !o.row0full && !o.bufhash && !o.moe);
            assert!(o.rows.is_empty());
        }
    }
}
