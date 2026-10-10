use super::*;
// A3: 핸드롤 JSON 파서군·경계 유틸·렌더의 유닛테스트 — 전부
// CPU 순수(무 GPU). 변형 JSON·UTF-8 절단·stop 오버랩·블록 content가
// 종전 무검증이었다.

#[test]
fn jstr_shapes() {
    assert_eq!(jstr(r#"{"a":"x"}"#, "a").as_deref(), Some("x"));
    assert_eq!(jstr(r#"{"a":  "spaced" }"#, "a").as_deref(), Some("spaced"));
    // 이스케이프 보존(unescape 여부는 계약상 원문 — 소비자 esc가 왕복)
    assert!(jstr(r#"{"a":"he said "hi""}"#, "a").is_some());
    assert_eq!(jstr(r#"{"a":123}"#, "a"), None);
    assert_eq!(jstr(r#"{"b":"y"}"#, "a"), None);
}

/// 2026-10-07: `\uXXXX`를 실제로 디코딩해야 한다. 종전 구현은 `\u`를
/// `other` 갈래로 흘려 `u`만 남겼다(`\uc778` → `uc778`) — 비ASCII를
/// 이스케이프해 보내는 클라이언트(파이썬 json.dumps 기본값)의 한국어가
/// 조용히 깨졌다. 게이트는 토큰 ID를 직접 넘겨 이 경로를 걷지 않는다.
#[test]
fn jstr_unicode_escape() {
    // BMP 한국자
    assert_eq!(jstr(r#"{"a":"서울"}"#, "a").as_deref(), Some("서울"));
    // BMP 밖 문자(서로게이트 페어) — 😀
    assert_eq!(jstr(r#"{"a":"😀"}"#, "a").as_deref(), Some("😀"));
    // 한글 + 이모지 혼합
    assert_eq!(
        jstr(r#"{"a":"한국어 😀 mix"}"#, "a").as_deref(),
        Some("한국어 😀 mix")
    );
    // 기존 단순 이스케이프는 그대로
    assert_eq!(jstr(r#"{"a":"x\ny"}"#, "a").as_deref(), Some("x\ny"));
    // 형식이 깨진 이스케이프는 원문 보존(무손실) — 조용히 지우지 않는다
    assert_eq!(jstr(r#"{"a":"\uZZZZ"}"#, "a").as_deref(), Some("\\uZZZZ"));
}

/// S1: 중첩 객체/문자열 안의 같은 키가 최상위 값을 오염시키지 않는다.
#[test]
fn jmember_ignores_nested_keys() {
    let b =
        r#"{"messages":[{"role":"user","content":"say \"temperature\": 9"}],"temperature":0.5}"#;
    assert_eq!(jnum(b, "temperature"), Some(0.5));
    let b2 = r#"{"tools":[{"parameters":{"top_p":0.1}}],"top_p":0.9}"#;
    assert_eq!(jnum(b2, "top_p"), Some(0.9));
    let b3 = r#"{"messages":[{"content":"\"stream\": true"}],"stream":false}"#;
    assert!(!jbool(b3, "stream"));
    let b4 = r#"{"messages":[{"content":"\"model\": \"evil\""}],"model":"real"}"#;
    assert_eq!(jstr(b4, "model").as_deref(), Some("real"));
    let b5 = r#"{"messages":[{"content":"\"stop\": [\"WRONG\"]"}],"stop":["END"]}"#;
    assert_eq!(jarr_str(b5, "stop"), Some(vec!["END".to_string()]));
}

/// S1: stop 배열 원소 안의 `]`가 절단을 만들지 않는다(종전 find(']')).
#[test]
fn jarr_str_keeps_brackets_in_strings() {
    assert_eq!(
        jarr_str(r#"{"stop":["a]b","c"]}"#, "stop"),
        Some(vec!["a]b".to_string(), "c".to_string()])
    );
    assert_eq!(
        jarr_str(r#"{"stop_sequences":["x","y"]}"#, "stop_sequences"),
        Some(vec!["x".to_string(), "y".to_string()])
    );
    assert_eq!(jarr_str(r#"{"stop":"z"}"#, "stop"), None);
    assert_eq!(jstop(r#"{"stop":["a]b"]}"#), vec!["a]b".to_string()]);
}

/// 값 슬라이스 경계 — 중첩 배열/객체 끝·이스케이프 따옴표.
#[test]
fn jmember_value_bounds() {
    let b = r#"{"a":[1,[2,3],{"x":"}"}],"b":7}"#;
    assert_eq!(jmember(b, "a"), Some(r#"[1,[2,3],{"x":"}"}]"#));
    assert_eq!(jnum(b, "b"), Some(7.0));
    assert_eq!(
        jarr_u32(r#"{"prompt":[1,2,3]}"#, "prompt"),
        Some(vec![1, 2, 3])
    );
    let c = r#"{"s":"a\"}b","t":1}"#;
    assert_eq!(jstr(c, "s").as_deref(), Some("a\"}b"));
    assert_eq!(jnum(c, "t"), Some(1.0));
}

#[test]
fn jnum_jbool() {
    assert_eq!(jnum(r#"{"n":42.5}"#, "n"), Some(42.5));
    assert_eq!(jnum(r#"{"n":"str"}"#, "n"), None);
    assert!(jbool(r#"{"f":true}"#, "f"));
    assert!(!jbool(r#"{"f":false}"#, "f"));
    assert!(!jbool(r#"{"f":"true"}"#, "f"));
}

#[test]
fn jstop_variants() {
    assert_eq!(jstop(r#"{"stop":"END"}"#), vec!["END".to_string()]);
    assert_eq!(
        jstop(r#"{"stop":["A","B"]}"#),
        vec!["A".to_string(), "B".to_string()]
    );
    // Anthropic stop_sequences(A14 경로)
    assert_eq!(
        jstop(r#"{"stop_sequences":["\n\n"]}"#),
        vec!["\n\n".to_string()]
    );
    assert!(jstop("{}").is_empty());
}

#[test]
fn earliest_stop_boundaries() {
    let stops = vec!["AB".to_string()];
    assert_eq!(earliest_stop("xxAByy", &stops, 0), Some((2, 2)));
    assert_eq!(earliest_stop("", &stops, 0), None);
    // 빈 stop은 무시(무한 절단 방지 계약)
    assert_eq!(earliest_stop("any", &[String::new()], 0), None);
    // 가장 이른 등장 선택
    let two = vec!["YY".to_string(), "XX".to_string()];
    assert_eq!(earliest_stop("aXXbYY", &two, 0), Some((1, 2)));
    // D4 창 경계 — from 이후에서도 절대 위치를 돌려준다.
    assert_eq!(earliest_stop("aXXbYY", &two, 2), Some((4, 2)));
}

#[test]
fn floor_char_boundary_multibyte() {
    let s = "한글abc"; // '한' 3바이트
    assert_eq!(floor_char_boundary(s, 0), 0);
    // 2바이트 지점은 경계 아님 → 0으로 보정
    assert_eq!(floor_char_boundary(s, 2), 0);
    assert_eq!(floor_char_boundary(s, 3), 3);
    assert_eq!(floor_char_boundary(s, 9), 9);
}

/// H(2026-10-09): 빈 messages 표면 — chat/anthropic 400 판정.
#[test]
fn empty_messages_surface() {
    assert!(!has_message_content(r#"{"messages":[]}"#));
    assert!(!has_message_content(
        r#"{"messages":[{"role":"user","content":""}]}"#
    ));
    assert!(!has_message_content(r#"{"max_tokens":4}"#));
    assert!(has_message_content(
        r#"{"messages":[{"role":"user","content":"hi"}]}"#
    ));
    // Anthropic 블록 배열 content.
    assert!(has_message_content(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#
    ));
}

#[test]
fn jblocks_nested_and_strings() {
    let arr = r#"[{"r":"a","c":"{x}"},{"r":"b"}]"#;
    let objs = jblocks(arr);
    assert_eq!(objs.len(), 2);
    assert!(objs[0].contains(r#""c":"{x}""#)); // 문자열 내 중괄호 무시
    // 빈 배열
    assert!(jblocks("[]").is_empty());
    // 비객체 원시 배열
    assert!(jblocks("[1,2]").is_empty());
}

#[test]
fn jcontent_string_and_blocks() {
    // 문자열 content — 종전 호환
    assert_eq!(jcontent(r#"{"content":"hi"}"#).as_deref(), Some("hi"));
    // 블록 배열(A14) — text 블록 연결, 비-text 건너뜀
    let o = r#"{"content":[{"type":"text","text":"a"},{"type":"image","src":"x"},{"type":"text","text":"b"}]}"#;
    assert_eq!(jcontent(o).as_deref(), Some("ab"));
    // content 없음
    assert_eq!(jcontent(r#"{"role":"user"}"#), None);
}

#[test]
fn jmessages_render_system_and_blocks() {
    // 최상위 system 필드(A14) — system 턴 선행
    let b = r#"{"system":"be brief","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}"#;
    let r = jmessages_render(b);
    assert!(r.contains(
        "<|im_start|>system
be brief<|im_end|>"
    ));
    assert!(r.contains(
        "<|im_start|>user
hello<|im_end|>"
    ));
    assert!(r.ends_with(
        "<|im_start|>assistant
"
    ));
    // 블록 배열이 없던 종전 형태(문자열 content) 동작 유지
    let b2 = r#"{"messages":[{"role":"user","content":"plain"}]}"#;
    assert!(jmessages_render(b2).contains(
        "<|im_start|>user
plain"
    ));
    // 원문 템플릿 통과 경로
    let b3 = r#"{"messages":[{"role":"user","content":"<|im_start|>raw"}]}"#;
    assert!(jmessages_render(b3).contains("<|im_start|>raw"));
}

#[test]
fn esc_roundtrip_control_chars() {
    assert_eq!(crate::json::esc("a\"b"), "a\\\"b");
    assert_eq!(crate::json::esc("nl\n"), "nl\\n");
    assert_eq!(crate::json::esc("tab\t"), "tab\\t");
}

/// [R3] TextStream — 홀드백(부분 stop 유출 방지)·stop 절단·잔여 플러시.
#[test]
fn text_stream_holdback_and_stop() {
    // stop 없음: 홀드백 0 → 즉시 방출.
    let mut ts = TextStream::new(vec![], false);
    assert_eq!(ts.push_text("abc").as_deref(), Some("abc"));
    assert!(ts.flush().is_none());
    // stop "END": 홀드백 2 — "abcEN"까지는 "abc"만 방출(EN 보류).
    let mut ts = TextStream::new(vec!["END".into()], false);
    assert_eq!(ts.push_text("abcEN").as_deref(), Some("abc"));
    // "D"로 stop 완성 — stop 본문 미포함(include_stop=false), stopped=true.
    assert!(ts.push_text("D").is_none());
    assert!(ts.stopped && ts.trunc);
    assert!(ts.flush().is_none());
    // include_stop=true — stop 본문 포함 방출.
    let mut ts = TextStream::new(vec!["END".into()], true);
    assert_eq!(ts.push_text("abcEN").as_deref(), Some("abc"));
    assert_eq!(ts.push_text("D").as_deref(), Some("END"));
    // 잔여 플러시(stop 미도달) — 홀드백(최장 stop-1 = 2B)은 보류된다.
    let mut ts = TextStream::new(vec!["END".into()], false);
    assert!(ts.push_text("hi").is_none(), "2B는 홀드백 보류");
    assert_eq!(ts.flush().as_deref(), Some("hi"));
    let mut ts = TextStream::new(vec!["END".into()], false);
    assert!(ts.push_text("hi").is_none());
    assert_eq!(
        ts.push_text("zz").as_deref(),
        Some("hi"),
        "4B-홀드백=2B 방출"
    );
    assert_eq!(ts.flush().as_deref(), Some("zz"));
}

/// [R3-fix] 멀티바이트 stop — 스캔 창(from)이 문자 중간에 떨어져도 패닉 금지
/// (종전 잠복 버그: 한글 본문 + 한글 stop에서 acc[base..] 패닉).
#[test]
fn text_stream_multibyte_stop_no_panic() {
    let mut ts = TextStream::new(vec!["。".into()], false);
    // " 의"(4B) → "미"(3B) — scan_from=2가 '의' 중간이 되는 시퀀스.
    assert_eq!(
        ts.push_text(" 의").as_deref(),
        Some(" "),
        "홀드백 2B → '의' 보류"
    );
    assert_eq!(ts.push_text("미").as_deref(), Some("의"));
    assert_eq!(ts.push_text("는").as_deref(), Some("미"));
    // stop 도달 — 직전 텍스트는 방출되고 stop 본문은 미포함.
    assert_eq!(ts.push_text("。").as_deref(), Some("는"));
    assert!(ts.stopped && ts.trunc);
}
