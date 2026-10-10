//! oai JSON 파서·메시지 렌더 — 순수 문자열 헬퍼(R3 분할).
// --- 최소 JSON 파싱 (중첩 없는 평탄 필드 추출) ---
/// 단순 이스케이프 1글자 해석(`n`,`t`,`u` 이외).
pub(super) fn push_simple(out: &mut String, c: char) {
    match c {
        'n' => out.push('\n'),
        't' => out.push('\t'),
        'r' => out.push('\r'),
        'b' => out.push('\u{8}'),
        'f' => out.push('\u{c}'),
        '/' => out.push('/'),
        '"' => out.push('"'),
        '\\' => out.push('\\'),
        // 미지의 이스케이프는 관례대로 문자를 그대로 통과시킨다.
        other => out.push(other),
    }
}

/// `tail` 앞 4자리 16진수를 잘라내 `(값, 소비 후 나머지)`.
/// 4자리가 모자라면 `None`이고 나머지는 **소비한 만큼만** 잘라낸 위치다 —
/// 호출부가 원문 보존 후 그 지점부터 파싱을 재개해야 닫는 따옴표를
/// 올바르게 처리할 수 있다.
pub(super) fn take_hex4(tail: &str) -> (Option<u32>, &str) {
    let mut v: u32 = 0;
    let mut rest = tail;
    let mut n = 0usize;
    while n < 4 {
        let Some(c) = rest.chars().next() else {
            break;
        };
        let Some(d) = c.to_digit(16) else {
            break;
        };
        v = v * 16 + d;
        n += 1;
        rest = &rest[c.len_utf8()..];
    }
    (if n == 4 { Some(v) } else { None }, rest)
}

/// 이스케이프 하나를 해석해 `(결과, 소비 후 나머지)`를 돌려준다.
/// 입력 `rest`는 `\` 다음부터 시작한다.
pub(super) fn take_escape(rest: &str) -> (String, &str) {
    let Some(first) = rest.chars().next() else {
        return (String::new(), rest);
    };
    if first != 'u' {
        let mut s = String::new();
        push_simple(&mut s, first);
        let n = first.len_utf8();
        return (s, &rest[n..]);
    }
    let (hi, after) = take_hex4(&rest[1..]);
    let Some(hi) = hi else {
        // 형식 이상 — `\u`만 원문 보존하고 소비한 16진수 뒤에서 재개한다.
        // 남은 문자열을 통째로 삼키면 닫는 따옴표를 놓쳐 파싱이 무너진다.
        return ("\\u".to_string(), after);
    };
    // 상위 서로게이트면 뒤따르는 `\uXXXX` 하위와 짝을 이룬다(BMP 밖 문자).
    if (0xD800..0xDC00).contains(&hi) {
        let low_src = after.strip_prefix("\\u").unwrap_or("");
        let (lo, after2) = take_hex4(low_src);
        if let Some(lo) = lo
            && (0xDC00..0xE000).contains(&lo)
            && let Some(c) = char::from_u32(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))
        {
            return (c.to_string(), after2);
        }
        // 짝이 안 맞으면 상위만 보존(무손실).
        return (format!("\\u{hi:04x}"), after);
    }
    match char::from_u32(hi) {
        Some(c) => (c.to_string(), after),
        None => (format!("\\u{hi:04x}"), after),
    }
}

/// 값 1개(문자열/숫자/리터럴/배열/객체)의 끝 오프셋 — 문자열·깊이 추적.
pub(super) fn value_end(body: &str, start: usize) -> Option<usize> {
    let b = body.as_bytes();
    match b.get(start)? {
        b'"' => jparse_string(&body[start..]).map(|(_, used)| start + used),
        &open @ (b'[' | b'{') => {
            let close = if open == b'[' { b']' } else { b'}' };
            let mut depth = 0usize;
            let mut i = start;
            while i < b.len() {
                match b[i] {
                    b'"' => {
                        i += jparse_string(&body[i..])?.1;
                        continue;
                    }
                    c if c == open => depth += 1,
                    c if c == close => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            // 숫자·true/false/null — 구분자 전까지.
            let mut i = start;
            while i < b.len() && !matches!(b[i], b',' | b'}' | b']' | b' ' | b'\n' | b'\t' | b'\r')
            {
                i += 1;
            }
            Some(i)
        }
    }
}

/// 최상위 객체에서 `key` 직계 멤버의 **값 슬라이스**(S1).
/// 종전 `find("\"key\":")`는 중첩 객체(예: messages 내 content 문자열)의
/// 같은 키를 먼저 잡아 샘플러·stop·model을 오염시킬 수 있었다 — 스캔은
/// 문자열·이스케이프·중첩 깊이를 추적해 직계 멤버만 본다.
pub(super) fn jmember<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let b = body.as_bytes();
    let mut i = body.find('{')? + 1;
    loop {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b',') {
            i += 1;
        }
        if i >= b.len() || b[i] != b'"' {
            return None; // '}' 또는 비정상 — 종료(호출부 기본값).
        }
        let (k, used) = jparse_string(&body[i..])?;
        i += used;
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() || b[i] != b':' {
            return None;
        }
        i += 1;
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let vend = value_end(body, i)?;
        if k == key {
            return Some(&body[i..vend]);
        }
        i = vend;
    }
}

pub(super) fn jstr(body: &str, key: &str) -> Option<String> {
    let (s, _) = jparse_string(jmember(body, key)?)?;
    Some(s)
}

pub(super) fn jnum(body: &str, key: &str) -> Option<f64> {
    let v = jmember(body, key)?;
    let end = v
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
        })
        .unwrap_or(v.len());
    v[..end].parse().ok()
}

pub(super) fn jbool(body: &str, key: &str) -> bool {
    jmember(body, key).is_some_and(|v| v.starts_with("true"))
}

pub(super) fn jarr_u32(body: &str, key: &str) -> Option<Vec<u32>> {
    let inner = jmember(body, key)?.strip_prefix('[')?.strip_suffix(']')?;
    Some(
        inner
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect(),
    )
}

/// 따옴표로 시작하는 JSON 문자열 1개 파싱 — (값, 소비 바이트 수).
/// jstr·jarr_str·jmember 공용. 이스케이프는 take_escape 재사용
/// (`\uXXXX`·서로게이트 페어를 jstr과 동일 규약으로 디코딩).
pub(super) fn jparse_string(s: &str) -> Option<(String, usize)> {
    if s.as_bytes().first() != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = 1usize;
    loop {
        let c = s[i..].chars().next()?;
        if c == '"' {
            return Some((out, i + 1));
        }
        if c == '\\' {
            let (dec, tail) = take_escape(&s[i + 1..]);
            out.push_str(&dec);
            i += 1 + (s[i + 1..].len() - tail.len());
        } else {
            out.push(c);
            i += c.len_utf8();
        }
    }
}

/// 문자열 배열 필드 파싱 (예: "stop": ["a", "b"]) — 최소 파서.
/// 원소는 jparse_string로 소비 — 문자열 안의 `]`에서 절단하지 않는다(S1).
pub(super) fn jarr_str(body: &str, key: &str) -> Option<Vec<String>> {
    let inner = jmember(body, key)?.strip_prefix('[')?.strip_suffix(']')?;
    let mut out = Vec::new();
    let mut seg = inner;
    loop {
        seg = seg.trim_start();
        if seg.is_empty() {
            break;
        }
        match jparse_string(seg) {
            Some((v, used)) => {
                out.push(v);
                seg = &seg[used..];
            }
            None => break,
        }
        seg = seg.trim_start();
        if let Some(rest) = seg.strip_prefix(',') {
            seg = rest;
        }
    }
    Some(out)
}

/// 요청 stop 파싱 — OpenAI "stop"(문자열|배열)과 Anthropic
/// "stop_sequences"(배열) 통합. 반환: (stop 문자열 목록, stop 문자열 포함 여부).
pub(super) fn jstop(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(s) = jstr(body, "stop") {
        out.push(s);
    } else if let Some(v) = jarr_str(body, "stop") {
        out = v;
    } else if let Some(v) = jarr_str(body, "stop_sequences") {
        out = v;
    }
    out
}

/// JSON 배열 내 최상위 객체 조각들 추출 — 중괄호 균형(문자열 리터럴 내부
/// { } 무시). jmessages_render·jcontent가 공유(A14 — 원본은
/// jmessages_render 인라인이었다).
pub(super) fn jblocks(arr: &str) -> Vec<&str> {
    let mut objs = Vec::new();
    let bytes = arr.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                let start = i;
                let mut depth = 0usize;
                let mut in_s = false;
                let mut esc = false;
                while i < bytes.len() {
                    let c = bytes[i];
                    if in_s {
                        if esc {
                            esc = false;
                        } else if c == b'\\' {
                            esc = true;
                        } else if c == b'"' {
                            in_s = false;
                        }
                    } else {
                        match c {
                            b'"' => in_s = true,
                            b'{' => depth += 1,
                            b'}' => {
                                depth -= 1;
                                if depth == 0 {
                                    i += 1;
                                    objs.push(&arr[start..i]);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    i += 1;
                }
            }
            b']' => break,
            _ => i += 1,
        }
    }
    objs
}

/// content 필드 추출(A14) — Anthropic 표준은 블록 배열
/// ([{"type":"text","text":"..."}])이라 문자열 전용 jstr은 조용히 빈 값을
/// 돌려줬다(빈 프롬프트 붕괴). text 블록을 연결하고 비-text 블록은 건너뛴다.
pub(super) fn jcontent(o: &str) -> Option<String> {
    if let Some(s) = jstr(o, "content") {
        return Some(s);
    }
    let k = o.find("\"content\"")?;
    let seg = &o[k..];
    let ob = seg.find('[')?;
    let mut out = String::new();
    for b in jblocks(&seg[ob..]) {
        if let Some(t) = jstr(b, "text") {
            out.push_str(&t);
        }
    }
    (!out.is_empty()).then_some(out)
}

/// role 인지 멀티턴 렌더링 — system/user/assistant 턴별
/// im_start 블록 + 마지막 generation prompt(마지막 턴이 assistant면 생략).
/// 정합 제약: 단일 user 메시지(시스템 없음)는 종전 단일 턴 출력과 바이트 동일
/// (verify 토큰 대면 표면 — 회귀 금지).
pub(super) fn jmessages_render(body: &str) -> String {
    let Some(mpos) = body.find("\"messages\"") else {
        return String::new();
    };
    let seg = &body[mpos..];
    let Some(ob) = seg.find('[') else {
        return String::new();
    };
    let arr = &seg[ob..];
    let objs = jblocks(arr);
    // 원문 연결(종전 출력) — content에 이미 템플릿이 있으면 원문 통과(종전 동작).
    let mut raw = String::new();
    for o in &objs {
        if let Some(c) = jcontent(o) {
            raw.push_str(&c);
            raw.push('\n');
        }
    }
    if raw.contains("<|im_start|>") {
        return raw;
    }
    let mut out = String::new();
    // A14: 최상위 system 필드 — Anthropic 표준은 messages 밖에 있다.
    if let Some(sys) = jstr(body, "system")
        && !sys.is_empty()
    {
        out.push_str("<|im_start|>system\n");
        out.push_str(&sys);
        out.push_str("<|im_end|>\n");
    }
    let mut last_assistant = false;
    for o in &objs {
        let role = jstr(o, "role").unwrap_or_else(|| "user".into());
        let content = jcontent(o).unwrap_or_default();
        out.push_str(match role.as_str() {
            "system" => "<|im_start|>system\n",
            "assistant" => "<|im_start|>assistant\n",
            _ => "<|im_start|>user\n",
        });
        out.push_str(&content);
        out.push_str("<|im_end|>\n");
        last_assistant = role == "assistant";
    }
    if !last_assistant {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// messages 배열에 내용 있는 메시지가 하나라도 있는가(H 2026-10-09).
/// 종전엔 빈 messages가 렌더 후 assistant 접두만 남아 200으로 진행됐다 —
/// completions의 빈 prompt 400과 표면 불일치(빈 프롬프트 붕괴).
pub(super) fn has_message_content(body: &str) -> bool {
    let Some(mpos) = body.find("\"messages\"") else {
        return false;
    };
    let seg = &body[mpos..];
    let Some(ob) = seg.find('[') else {
        return false;
    };
    jblocks(&seg[ob..])
        .iter()
        .any(|o| jcontent(o).is_some_and(|c| !c.is_empty()))
}

/// 요청 본문에서 샘플링 파라미터 추출 — 미지정시 None (greedy, 종전 동작).
/// OpenAI 파라미터 명칭: temperature·top_k·top_p·min_p·repeat_penalty·seed.
pub(super) fn parse_sampler(body: &str) -> Option<llm170_core::sampler::SamplerParams> {
    let temperature = jnum(body, "temperature").unwrap_or(0.0) as f32;
    let top_k = jnum(body, "top_k").unwrap_or(0.0).max(0.0) as usize;
    let top_p = jnum(body, "top_p").unwrap_or(1.0) as f32;
    let min_p = jnum(body, "min_p").unwrap_or(0.0) as f32;
    let repeat_penalty = jnum(body, "repeat_penalty").unwrap_or(1.0) as f32;
    let seed = jnum(body, "seed").unwrap_or(0.0) as u64;
    let p = llm170_core::sampler::SamplerParams {
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        seed,
        ..Default::default()
    };
    if !p.is_greedy() { Some(p) } else { None }
}
