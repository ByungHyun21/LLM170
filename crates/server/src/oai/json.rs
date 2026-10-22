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
        other => out.push(other),
    }
}

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
        return ("\\u".to_string(), after);
    };
    if (0xD800..0xDC00).contains(&hi) {
        let low_src = after.strip_prefix("\\u").unwrap_or("");
        let (lo, after2) = take_hex4(low_src);
        if let Some(lo) = lo
            && (0xDC00..0xE000).contains(&lo)
            && let Some(c) = char::from_u32(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))
        {
            return (c.to_string(), after2);
        }
        return (format!("\\u{hi:04x}"), after);
    }
    match char::from_u32(hi) {
        Some(c) => (c.to_string(), after),
        None => (format!("\\u{hi:04x}"), after),
    }
}

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
            let mut i = start;
            while i < b.len() && !matches!(b[i], b',' | b'}' | b']' | b' ' | b'\n' | b'\t' | b'\r')
            {
                i += 1;
            }
            Some(i)
        }
    }
}

pub(super) fn jmember<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let b = body.as_bytes();
    let mut i = body.find('{')? + 1;
    loop {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b',') {
            i += 1;
        }
        if i >= b.len() || b[i] != b'"' {
            return None;
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

pub(crate) fn gmessages_render(body: &str) -> String {
    let Some(mpos) = body.find("\"messages\"") else {
        return String::new();
    };
    let seg = &body[mpos..];
    let Some(ob) = seg.find('[') else {
        return String::new();
    };
    let objs = jblocks(&seg[ob..]);
    let mut out = String::from("<bos>");
    let mut rest: &[&str] = &objs;
    let mut sys: Option<String> = None;
    if let Some(s) = jstr(body, "system") {
        let t = s.trim();
        if !t.is_empty() {
            sys = Some(t.to_string());
        }
    }
    if sys.is_none()
        && let Some(first) = objs.first()
    {
        let r = jstr(first, "role").unwrap_or_default();
        if r == "system" || r == "developer" {
            let c = jcontent(first).unwrap_or_default();
            let t = c.trim();
            if !t.is_empty() {
                sys = Some(t.to_string());
            }
            rest = &objs[1..];
        }
    }
    if let Some(s) = sys {
        out.push_str("<|turn>system\n");
        out.push_str(&s);
        out.push_str("<turn|>\n");
    }
    for o in rest {
        let role = jstr(o, "role").unwrap_or_else(|| "user".into());
        let role = if role == "assistant" {
            "model".to_string()
        } else {
            role
        };
        let content = jcontent(o).unwrap_or_default();
        out.push_str(&format!("<|turn>{role}\n"));
        out.push_str(content.trim());
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    out
}

pub(crate) struct ChatSpec {
    pub render: fn(&str) -> String,
    pub eos: &'static [u32],
}

pub(crate) fn chat_spec(arch: &str) -> Option<ChatSpec> {
    if llm170_core::gemma4::bind::arch_ok(arch) {
        Some(ChatSpec {
            render: gmessages_render,
            eos: &[1, 106, 50],
        })
    } else if llm170_core::qwen35::bind::arch_supported(arch) {
        Some(ChatSpec {
            render: jmessages_render,
            eos: &[llm170_core::qwen35::EOS_EOT],
        })
    } else {
        None
    }
}

pub(crate) fn messages_render(body: &str) -> String {
    let arch = crate::engine::arch();
    match chat_spec(&arch) {
        Some(spec) => (spec.render)(body),
        None => jmessages_render(body),
    }
}

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

pub(super) fn jblock_at(s: &str, at: usize) -> Option<&str> {
    let b = s.as_bytes();
    if at >= b.len() || b[at] != b'{' {
        return None;
    }
    let mut depth = 0usize;
    let mut in_s = false;
    let mut esc = false;
    for i in at..b.len() {
        if esc {
            esc = false;
        } else if b[i] == b'\\' && in_s {
            esc = true;
        } else if b[i] == b'"' {
            in_s = !in_s;
        } else if !in_s {
            if b[i] == b'{' {
                depth += 1;
            } else if b[i] == b'}' {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[at..=i]);
                }
            }
        }
    }
    None
}

pub(super) fn jarr_span(s: &str, open_idx: usize) -> Option<usize> {
    let b = s.as_bytes();
    if open_idx >= b.len() || b[open_idx] != b'[' {
        return None;
    }
    let mut depth = 0usize;
    let mut in_s = false;
    let mut esc = false;
    for i in open_idx..b.len() {
        if esc {
            esc = false;
        } else if b[i] == b'\\' && in_s {
            esc = true;
        } else if b[i] == b'"' {
            in_s = !in_s;
        } else if !in_s {
            if b[i] == b'[' {
                depth += 1;
            } else if b[i] == b']' {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
    }
    None
}

pub(super) fn jstrs_span(b: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let kp = b.find(&pat)? + pat.len();
    let ob = b[kp..].find('[')? + kp;
    let cl = jarr_span(b, ob)?;
    Some(b[ob..=cl].to_string())
}

pub(super) fn jarr_pure_strings(s: &str) -> Option<Vec<String>> {
    let close = jarr_span(s, 0)?;
    let inner = &s[1..close];
    if inner.trim().is_empty() {
        return None;
    }
    let mut out = Vec::new();
    let b = inner.as_bytes();
    let mut i = 0usize;
    let mut saw_any = false;
    while i < b.len() {
        if b[i] == b'"' {
            saw_any = true;
            let st = i + 1;
            let mut j = st;
            let mut esc = false;
            while j < b.len() {
                if esc {
                    esc = false;
                } else if b[j] == b'\\' {
                    esc = true;
                } else if b[j] == b'"' {
                    break;
                }
                j += 1;
            }
            if j >= b.len() {
                return None;
            }
            out.push(inner[st..j].to_string());
            i = j + 1;
        } else if b[i] == b',' || (b[i] as char).is_whitespace() {
            i += 1;
        } else {
            return None;
        }
    }
    saw_any.then_some(out)
}

pub(super) fn jobject_props(b: &str) -> Option<Vec<(String, String)>> {
    let pp = b.find("\"properties\"")?;
    let ob = b[pp..].find('{')? + pp;
    let blk = jblock_at(b, ob)?;
    let mut out = Vec::new();
    let bytes = blk.as_bytes();
    let mut i = 1usize;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let ks = i + 1;
            let mut j = ks;
            while j < bytes.len() && bytes[j] != b'"' {
                if bytes[j] == b'\\' {
                    j += 1;
                }
                j += 1;
            }
            let key = blk[ks..j].to_string();
            let mut k = j + 1;
            while k < bytes.len() && (bytes[k] as char).is_whitespace() {
                k += 1;
            }
            if k < bytes.len() && bytes[k] == b':' {
                k += 1;
                while k < bytes.len() && (bytes[k] as char).is_whitespace() {
                    k += 1;
                }
                if k < bytes.len()
                    && bytes[k] == b'{'
                    && let Some(node) = jblock_at(blk, k)
                {
                    out.push((key, node.to_string()));
                    i = k + node.len();
                    continue;
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    if out.is_empty() { None } else { Some(out) }
}
