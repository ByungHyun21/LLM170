//! json — JSON 문자열 이스케이프 단일 구현.
//!
//! 종전 http.rs(무따옴표·제어문자 미처리)와 infer.rs(따옴표 포함·완전판)가
//! 각자 구현해 제어문자에서 출력이 발산했다 — http 쪽은 invalid JSON을
//! 뿜을 수 있었다(B2). 여기가 유일 구현.

/// 이스케이프 본체 — 따옴표 미포함. 이미 `"..."` 틀 안에 끼워 넣는
/// 호출부(http.rs 응답 조립)용.
pub(crate) fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
