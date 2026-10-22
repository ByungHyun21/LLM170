use super::*;

#[test]
fn jstr_shapes() {
    assert_eq!(jstr(r#"{"a":"x"}"#, "a").as_deref(), Some("x"));
    assert_eq!(jstr(r#"{"a":  "spaced" }"#, "a").as_deref(), Some("spaced"));
    assert!(jstr(r#"{"a":"he said "hi""}"#, "a").is_some());
    assert_eq!(jstr(r#"{"a":123}"#, "a"), None);
    assert_eq!(jstr(r#"{"b":"y"}"#, "a"), None);
}

#[test]
fn jstr_unicode_escape() {
    assert_eq!(jstr(r#"{"a":"서울"}"#, "a").as_deref(), Some("서울"));
    assert_eq!(jstr(r#"{"a":"😀"}"#, "a").as_deref(), Some("😀"));
    assert_eq!(
        jstr(r#"{"a":"한국어 😀 mix"}"#, "a").as_deref(),
        Some("한국어 😀 mix")
    );
    assert_eq!(jstr(r#"{"a":"x\ny"}"#, "a").as_deref(), Some("x\ny"));
    assert_eq!(jstr(r#"{"a":"\uZZZZ"}"#, "a").as_deref(), Some("\\uZZZZ"));
}

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
    assert_eq!(earliest_stop("any", &[String::new()], 0), None);
    let two = vec!["YY".to_string(), "XX".to_string()];
    assert_eq!(earliest_stop("aXXbYY", &two, 0), Some((1, 2)));
    assert_eq!(earliest_stop("aXXbYY", &two, 2), Some((4, 2)));
}

#[test]
fn floor_char_boundary_multibyte() {
    let s = "한글abc";
    assert_eq!(floor_char_boundary(s, 0), 0);
    assert_eq!(floor_char_boundary(s, 2), 0);
    assert_eq!(floor_char_boundary(s, 3), 3);
    assert_eq!(floor_char_boundary(s, 9), 9);
}

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
    assert!(has_message_content(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#
    ));
}

#[test]
fn jblocks_nested_and_strings() {
    let arr = r#"[{"r":"a","c":"{x}"},{"r":"b"}]"#;
    let objs = jblocks(arr);
    assert_eq!(objs.len(), 2);
    assert!(objs[0].contains(r#""c":"{x}""#));
    assert!(jblocks("[]").is_empty());
    assert!(jblocks("[1,2]").is_empty());
}

#[test]
fn jcontent_string_and_blocks() {
    assert_eq!(jcontent(r#"{"content":"hi"}"#).as_deref(), Some("hi"));
    let o = r#"{"content":[{"type":"text","text":"a"},{"type":"image","src":"x"},{"type":"text","text":"b"}]}"#;
    assert_eq!(jcontent(o).as_deref(), Some("ab"));
    assert_eq!(jcontent(r#"{"role":"user"}"#), None);
}

#[test]
fn jmessages_render_system_and_blocks() {
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
    let b2 = r#"{"messages":[{"role":"user","content":"plain"}]}"#;
    assert!(jmessages_render(b2).contains(
        "<|im_start|>user
plain"
    ));
    let b3 = r#"{"messages":[{"role":"user","content":"<|im_start|>raw"}]}"#;
    assert!(jmessages_render(b3).contains("<|im_start|>raw"));
}

#[test]
fn esc_roundtrip_control_chars() {
    assert_eq!(crate::json::esc("a\"b"), "a\\\"b");
    assert_eq!(crate::json::esc("nl\n"), "nl\\n");
    assert_eq!(crate::json::esc("tab\t"), "tab\\t");
}

#[test]
fn text_stream_holdback_and_stop() {
    let mut ts = TextStream::new(vec![], false);
    assert_eq!(ts.push_text("abc").as_deref(), Some("abc"));
    assert!(ts.flush().is_none());
    let mut ts = TextStream::new(vec!["END".into()], false);
    assert_eq!(ts.push_text("abcEN").as_deref(), Some("abc"));
    assert!(ts.push_text("D").is_none());
    assert!(ts.stopped && ts.trunc);
    assert!(ts.flush().is_none());
    let mut ts = TextStream::new(vec!["END".into()], true);
    assert_eq!(ts.push_text("abcEN").as_deref(), Some("abc"));
    assert_eq!(ts.push_text("D").as_deref(), Some("END"));
    let mut ts = TextStream::new(vec!["END".into()], false);
    assert!(ts.push_text("hi").is_none(), "2B는 홀드백 보류");
    assert_eq!(ts.flush(), Some("hi"));
    let mut ts = TextStream::new(vec!["END".into()], false);
    assert!(ts.push_text("hi").is_none());
    assert_eq!(
        ts.push_text("zz").as_deref(),
        Some("hi"),
        "4B-홀드백=2B 방출"
    );
    assert_eq!(ts.flush(), Some("zz"));
}

#[test]
fn text_stream_multibyte_stop_no_panic() {
    let mut ts = TextStream::new(vec!["。".into()], false);
    assert_eq!(
        ts.push_text(" 의").as_deref(),
        Some(" "),
        "홀드백 2B → '의' 보류"
    );
    assert_eq!(ts.push_text("미").as_deref(), Some("의"));
    assert_eq!(ts.push_text("는").as_deref(), Some("미"));
    assert_eq!(ts.push_text("。").as_deref(), Some("는"));
    assert!(ts.stopped && ts.trunc);
}

#[test]
fn cap_output_policies() {
    let s = crate::sched::admit::RESERVE_SLACK;
    let max = 262_144usize;
    let c = cap_output(200_000, 100_000, max, 1 << 40, s);
    assert_eq!(c.tg, max - 200_000 - s, "모델 초과분은 출력 클램프(vLLM식)");
    assert_eq!(c.verdict, OutputVerdict::Ok);
    let c = cap_output(3, 8, max, 1 << 40, s);
    assert_eq!((c.tg, c.verdict), (8, OutputVerdict::Ok));
    let pool = 3 + 8 + s;
    let c = cap_output(3, 8, max, pool, s);
    assert_eq!((c.tg, c.verdict), (8, OutputVerdict::Ok));
    let c = cap_output(3, 9, max, pool, s);
    assert_eq!((c.tg, c.verdict), (9, OutputVerdict::PoolExceeded));
    let c = cap_output(200_000, 100_000, max, 150_000, s);
    assert_eq!(c.tg, max - 200_000 - s);
    assert_eq!(c.verdict, OutputVerdict::PoolExceeded);
    let c = cap_output(max - s, 1, max, 1 << 40, s);
    assert_eq!(c.verdict, OutputVerdict::PromptTooLong);
    let c = cap_output(1_000, 5_000, max, 2_000, s);
    assert_eq!((c.tg, c.verdict), (5_000, OutputVerdict::PoolExceeded));
}
