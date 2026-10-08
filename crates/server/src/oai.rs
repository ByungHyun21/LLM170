//! OAI/Anthropic 페이로드 계층 (http.rs에서 순수 이동).
//! JSON 파서군(j*)·메시지 렌더·샘플러·stop 유틸·요청 라우팅(handle)과
//! OAI/Anthropic 방출 러너가 산다. 전송(resp/sse/read_request/serve)은
//! http.rs — 이 층은 바이트 해석과 프로토콜 스키만 담는다.
use crate::engine::SlotJob;
use crate::http::{ENGINE_DEAD, READY, SERVER_CTX, TokOut, read_request, resp, resp_sse_open, sse};
use std::net::TcpStream;
use std::sync::atomic::Ordering;

// --- 최소 JSON 파싱 (중첩 없는 평탄 필드 추출) ---
/// 단순 이스케이프 1글자 해석(`n`,`t`,`u` 이외).
fn push_simple(out: &mut String, c: char) {
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
fn take_hex4(tail: &str) -> (Option<u32>, &str) {
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
fn take_escape(rest: &str) -> (String, &str) {
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

fn jstr(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)? + pat.len();
    let b = body[i..].trim_start();
    if !b.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut rest = &b[1..];
    loop {
        match rest.chars().next() {
            None => break,
            Some('\\') => {
                let (s, tail) = take_escape(&rest[1..]);
                out.push_str(&s);
                rest = tail;
            }
            Some('"') => break,
            Some(c) => {
                let n = c.len_utf8();
                out.push(c);
                rest = &rest[n..];
            }
        }
    }
    Some(out)
}

fn jnum(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)? + pat.len();
    let b = body[i..].trim_start();
    let end = b
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
        })
        .unwrap_or(b.len());
    b[..end].parse().ok()
}

fn jbool(body: &str, key: &str) -> bool {
    let pat = format!("\"{key}\":");
    body.find(&pat)
        .map(|i| body[i + pat.len()..].trim_start().starts_with("true"))
        .unwrap_or(false)
}

fn jarr_u32(body: &str, key: &str) -> Option<Vec<u32>> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)? + pat.len();
    let b = body[i..].trim_start();
    if !b.starts_with('[') {
        return None;
    }
    let end = b.find(']')?;
    Some(
        b[1..end]
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect(),
    )
}

/// 따옴표로 시작하는 JSON 문자열 1개 파싱 — (값, 소비 바이트 수).
/// jstr·jarr_str 공용.
fn jparse_string(s: &str) -> Option<(String, usize)> {
    let b = s.as_bytes();
    if b.first() != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut esc = false;
    for (n, c) in s[1..].char_indices() {
        if esc {
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                other => other,
            });
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return Some((out, n + 2));
        } else {
            out.push(c);
        }
    }
    None
}

/// 문자열 배열 필드 파싱 (예: "stop": ["a", "b"]) — 최소 파서.
fn jarr_str(body: &str, key: &str) -> Option<Vec<String>> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)? + pat.len();
    let b = body[i..].trim_start();
    if !b.starts_with('[') {
        return None;
    }
    let end = b.find(']')?;
    let mut out = Vec::new();
    let mut seg = &b[1..end];
    while let Some(q) = seg.find('"') {
        seg = &seg[q..];
        match jparse_string(seg) {
            Some((v, used)) => {
                out.push(v);
                seg = &seg[used..];
            }
            None => break,
        }
    }
    Some(out)
}

/// 요청 stop 파싱 — OpenAI "stop"(문자열|배열)과 Anthropic
/// "stop_sequences"(배열) 통합. 반환: (stop 문자열 목록, stop 문자열 포함 여부).
fn jstop(body: &str) -> Vec<String> {
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

/// 유닉스 초.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// OAI 응답 id — 프로세스 단조 카운터로 충돌 없는 유일값.
fn oai_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("cmpl-{}-{n}", std::process::id())
}

/// 문자 경계로 내림 보정 (UTF-8 안전 절단) — 스트림 holdback 계산용.
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// 응답 포맷 컨텍스트 — 엔드포인트별 조립에 필요한 최소값.
struct EmitFmt {
    chat: bool,
    model: String,
    stop_strs: Vec<String>,
    include_stop: bool,
}

/// role 인지 멀티턴 렌더링 — system/user/assistant 턴별
/// im_start 블록 + 마지막 generation prompt(마지막 턴이 assistant면 생략).
/// 정합 제약: 단일 user 메시지(시스템 없음)는 종전 단일 턴 출력과 바이트 동일
/// (verify 토큰 대면 표면 — 회귀 금지).
fn jmessages_render(body: &str) -> String {
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

/// 요청 본문에서 샘플링 파라미터 추출 — 미지정시 None (greedy, 종전 동작).
/// OpenAI 파라미터 명칭: temperature·top_k·top_p·min_p·repeat_penalty·seed.
fn parse_sampler(body: &str) -> Option<llm170_core::sampler::SamplerParams> {
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

/// qwen 어휘 <end_of_turn> — chat/anthropic 조기 정지 토큰 (EOS 248044와 구분).
const STOP_EOT: u32 = 248046;

pub(crate) fn handle(
    mut stream: TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
) -> Result<(), String> {
    loop {
        let req = match read_request(&mut stream) {
            Ok(r) => r,
            Err(_) => return Ok(()), // 연결 종료
        };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => {
                if READY.load(Ordering::Acquire) {
                    resp(&mut stream, 200, "application/json", "{\"status\":\"ok\"}")
                } else {
                    resp(
                        &mut stream,
                        503,
                        "application/json",
                        "{\"status\":\"loading\"}",
                    )
                }
            }
            ("GET", "/v1/models") => resp(
                &mut stream,
                200,
                "application/json",
                "{\"object\":\"list\",\"data\":[{\"id\":\"llm170\",\"object\":\"model\",\"owned_by\":\"local\"}]}",
            ),
            ("POST", "/tokenize") => {
                let Some(content) = jstr(&req.body, "content") else {
                    resp(
                        &mut stream,
                        400,
                        "application/json",
                        "{\"error\":\"content required\"}",
                    );
                    continue;
                };
                let toks = crate::engine::greedy_encode(&content);
                let ids: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
                resp(
                    &mut stream,
                    200,
                    "application/json",
                    &format!("{{\"tokens\":[{}]}}", ids.join(",")),
                );
            }
            ("POST", "/v1/completions") | ("POST", "/completion") => {
                let n_predict = jnum(&req.body, "max_tokens")
                    .unwrap_or(jnum(&req.body, "n_predict").unwrap_or(24.0))
                    .max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let prompt_ids = jarr_u32(&req.body, "prompt");
                let prompt_txt = jstr(&req.body, "prompt");
                let ids = match (prompt_ids, prompt_txt) {
                    (Some(v), _) if !v.is_empty() => v,
                    (_, Some(t)) => crate::engine::greedy_encode(&t),
                    _ => {
                        resp(
                            &mut stream,
                            400,
                            "application/json",
                            "{\"error\":\"prompt required\"}",
                        );
                        continue;
                    }
                };
                run_and_emit(
                    &mut stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    // completions는 text_completion 포맷.
                    &EmitFmt {
                        chat: false,
                        model: jstr(&req.body, "model").unwrap_or_else(|| "local_llm".into()),
                        stop_strs: jstop(&req.body),
                        include_stop: jbool(&req.body, "include_stop_str_in_output"),
                    },
                    Vec::new(),
                    parse_sampler(&req.body),
                );
            }
            ("POST", "/v1/chat/completions") => {
                let n_predict = jnum(&req.body, "max_tokens")
                    .unwrap_or(jnum(&req.body, "n_predict").unwrap_or(24.0))
                    .max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                // role 인지 멀티턴 렌더링(시스템 프롬프트 보존).
                let text = jmessages_render(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit(
                    &mut stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    &EmitFmt {
                        chat: true,
                        model: jstr(&req.body, "model").unwrap_or_else(|| "local_llm".into()),
                        stop_strs: jstop(&req.body),
                        include_stop: jbool(&req.body, "include_stop_str_in_output"),
                    },
                    vec![STOP_EOT],
                    parse_sampler(&req.body),
                );
            }
            ("POST", "/v1/messages") => {
                let n_predict = jnum(&req.body, "max_tokens").unwrap_or(24.0).max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let text = jmessages_render(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit_anthropic(
                    &mut stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    parse_sampler(&req.body),
                    jstop(&req.body), // A14: stop_sequences(jstop이 배열 파싱)
                );
            }
            _ => resp(
                &mut stream,
                404,
                "application/json",
                "{\"error\":\"not found\"}",
            ),
        }
    }
}

/// 슬롯 잡 enqueue 공통 — 채널 쌍 생성·SlotJob 조립·큐 송신.
/// Err면 이미 503(queue full) 응답을 썼다. 반환: (최종 결과 수신기, 스트림
/// 토큰 수신기 — 비스트림 모드는 진행 채널이 그대로 닫힌다).
#[allow(clippy::type_complexity)]
fn enqueue_job(
    stream: &mut TcpStream,
    tx: &std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stops: Vec<u32>,
    sampler: Option<llm170_core::sampler::SamplerParams>,
) -> Result<
    (
        std::sync::mpsc::Receiver<TokOut>,
        std::sync::mpsc::Receiver<u32>,
    ),
    (),
> {
    if ENGINE_DEAD.load(Ordering::Acquire) {
        // QA-11: 엔진 스레드 사망 — "queue full" 오보 방지.
        resp(
            stream,
            503,
            "application/json",
            "{\"error\":\"engine dead (slot loop panicked)\"}",
        );
        return Err(());
    }
    let (otx, orx) = std::sync::mpsc::channel::<TokOut>();
    let (ptx, prx) = std::sync::mpsc::channel::<u32>();
    let job = SlotJob {
        tokens: ids,
        n_predict,
        spec_k: crate::engine::SPEC_K.get().copied().unwrap_or(0),
        sampler,
        stops,
        // QA-3: 비스트림도 progress 채널 부여 — 핸들러가 prx를 잡고 폴링
        // 대기하며 절단 시 drop → slot_emit 송신 실패 → cancelled(스트림과
        // 동일 메커니즘). 종전 비스트림은 절단 감지 자체가 없었다.
        progress: Some(ptx),
        out: otx,
        queued: std::time::Instant::now(),
    };
    if tx.try_send(job).is_err() {
        resp(
            stream,
            503,
            "application/json",
            "{\"error\":\"queue full\"}",
        );
        return Err(());
    }
    Ok((orx, prx))
}
/// QA-3: 비차단 peek로 클라이언트 절단 판정 — EOF(0) 또는 커널 접속 에러만
/// 절단. 데이터 도착(파이프라인 후속 요청)은 생존으로 본다.
fn peer_gone(stream: &mut TcpStream) -> bool {
    let _ = stream.set_nonblocking(true);
    let mut b = [0u8; 1];
    let r = stream.peek(&mut b);
    let _ = stream.set_nonblocking(false);
    match r {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => matches!(
            e.kind(),
            std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::UnexpectedEof
        ),
    }
}

fn run_and_emit(
    stream: &mut TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stream_mode: bool,
    fmt: &EmitFmt,
    stops: Vec<u32>,
    sampler: Option<llm170_core::sampler::SamplerParams>,
) {
    // ctx 검증 — 프롬프트+생성이 컨텍스트를 넘으면 400 (context-shift v1:
    // 슬롯 무상태라 이동 없이 거절 — 이동 재배치는 접두 캐시 도입 시).
    let ctx = *SERVER_CTX.get().unwrap_or(&4096);
    if ids.len() + n_predict + 8 >= ctx {
        resp(
            stream,
            400,
            "application/json",
            &format!(
                "{{\"error\":\"context too small: prompt {} + n_predict {} >= ctx {}\"}}",
                ids.len(),
                n_predict,
                ctx
            ),
        );
        return;
    }
    let prompt_len = ids.len();
    let Ok((orx, prx)) = enqueue_job(stream, &tx, ids, n_predict, stops, sampler) else {
        return;
    };
    if !stream_mode {
        // QA-3: 폴링 대기 — 절단 감지 시 prx가 이 스코프를 벗어나 drop 되고
        // 다음 slot_emit부터 cancelled. 종전엔 n_predict 전량을 GPU에서 실행.
        let mut toks = Vec::new();
        let err: Option<String>;
        loop {
            match orx.recv_timeout(std::time::Duration::from_millis(500)) {
                Ok(r) => {
                    toks = r.tokens;
                    err = r.error;
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if peer_gone(stream) {
                        return; // 절단 — 응답 불필요, 잡 정리만
                    }
                }
                // QA-9: 송신측 소멸(QA-1 수리 전 무한루프가 만들던 상황 등) —
                // 빈 200 대신 명시적 오류.
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    err = Some("engine result channel closed".into());
                    break;
                }
            }
        }
        if let Some(e) = err {
            resp(
                stream,
                500,
                "application/json",
                &format!("{{\"error\":\"{}\"}}", crate::json::esc(&e)),
            );
            return;
        }
        // stop 문자열 응답 절단 — 토큰 순회하며 누적 텍스트에
        // 최초 등장 시 절단. 서버 측 절단은 과도 방안(엔진 조기 정지는 후속 항목).
        let mut det = crate::engine::Detok::new();
        let mut acc = String::new();
        let mut ntok = 0usize;
        let mut trunc = false;
        'tok: for &t in &toks {
            acc.push_str(&det.push(t));
            ntok += 1;
            for s in &fmt.stop_strs {
                if !s.is_empty()
                    && let Some(p) = acc.find(s.as_str())
                {
                    let end = if fmt.include_stop { p + s.len() } else { p };
                    acc.truncate(end);
                    trunc = true;
                    break 'tok;
                }
            }
        }
        // OAI 표준 비스트림 응답(id/created/model/object/
        // choices/usage) — 종전 소비자를 위한 tokens·text는 유지.
        let finish = if trunc {
            "stop"
        } else if toks.len() >= n_predict {
            "length"
        } else {
            "stop"
        };
        let esc = crate::json::esc(&acc);
        let arr: Vec<String> = toks[..ntok].iter().map(|t| t.to_string()).collect();
        let (id, created) = (oai_id(), now_secs());
        let body = if fmt.chat {
            format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{}\",\"object\":\"chat.completion\",\"\
                 choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{esc}\"}},\"finish_reason\":\"{finish}\"}}],\
                 \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},\"tokens\":[{}]}}",
                crate::json::esc(&fmt.model),
                prompt_len,
                ntok,
                prompt_len + ntok,
                arr.join(",")
            )
        } else {
            format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{}\",\"object\":\"text_completion\",\"\
                 choices\":[{{\"index\":0,\"text\":\"{esc}\",\"finish_reason\":\"{finish}\"}}],\
                 \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},\
                 \"text\":\"{esc}\",\"tokens\":[{}]}}",
                crate::json::esc(&fmt.model),
                prompt_len,
                ntok,
                prompt_len + ntok,
                arr.join(",")
            )
        };
        resp(stream, 200, "application/json", &body);
        return;
    }
    resp_sse_open(stream);
    let (id, created) = (oai_id(), now_secs());
    let model_esc = crate::json::esc(&fmt.model);
    if fmt.chat {
        // 첫 청크 — role delta(id/created/model 포함 표준 계약).
        let frame = sse(
            stream,
            "message",
            &format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{model_esc}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}"
            ),
        );
        if frame.is_err() {
            return;
        }
    }
    // 토큰 생성 즉시 SSE — 장문 요청이 완료까지 굳지 않게 (2026-09-01).
    let mut det = crate::engine::Detok::new();
    // 스트림 stop 처리 — 누적 텍스트 기준 판정 + holdback으로
    // stop 문자열 부분 유출 방지(마지막 max(stop)-1바이트는 미송신 보류).
    let mut acc = String::new();
    let mut emitted = 0usize;
    let mut ntok = 0usize;
    let mut trunc = false;
    let holdback = fmt
        .stop_strs
        .iter()
        .map(|s| s.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    let mut stopped = false;
    for t in prx {
        ntok += 1;
        acc.push_str(&det.push(t));
        if let Some((p, l)) = earliest_stop(&acc, &fmt.stop_strs) {
            let end = if fmt.include_stop { p + l } else { p };
            if end > emitted {
                let piece = crate::json::esc(&acc[emitted..end]);
                emitted = end;
                let frame = emit_delta(stream, fmt, &id, created, &model_esc, &piece);
                if frame.is_err() {
                    return;
                }
            }
            trunc = true;
            stopped = true;
            break; // prx drop → 기존 cancelled 경로로 슬롯 회수
        }
        let safe = floor_char_boundary(&acc, acc.len().saturating_sub(holdback));
        if safe > emitted {
            let piece = crate::json::esc(&acc[emitted..safe]);
            emitted = safe;
            let frame = emit_delta(stream, fmt, &id, created, &model_esc, &piece);
            // 쓰기 실패(클라 절단) 시 즉시 탈출 — 이 스코프를
            // 벗어나며 prx가 drop되고 다음 slot_emit부터 기존 cancelled 경로가
            // 슬롯을 회수한다(비스트림 QA-3 폴링과 동일 메커니즘).
            if frame.is_err() {
                return;
            }
        }
    }
    let _ = stopped;
    // 잔여 보류분 플러시(stop 없이 종료 시).
    if !trunc && acc.len() > emitted {
        let piece = crate::json::esc(&acc[emitted..]);
        let _ = emit_delta(stream, fmt, &id, created, &model_esc, &piece);
    }
    let final_res = orx.recv(); // 최종 결과 수령 (종료 정리)
    let n_gen = final_res.as_ref().map(|r| r.tokens.len()).unwrap_or(ntok);
    let finish = if trunc {
        "stop"
    } else if n_gen >= n_predict {
        "length"
    } else {
        "stop"
    };
    if fmt.chat {
        // 종료 청크 — finish_reason 후 [DONE](기존 계약 유지).
        let _ = sse(
            stream,
            "message",
            &format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{model_esc}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}]}}"
            ),
        );
    }
    let _ = sse(stream, "done", "[DONE]");
    // QA-2 연계 수리: SSE 완료 후 연결 종료. curl류 클라이언트는
    // [DONE]을 인지하지 못해 서버의 keep-alive 대기에 묶였고 — 무타임아웃
    // 시대엔 무한 대기, read_timeout(120s) 도입 후엔 요청마다 +120s 꼬리가
    // 붙었다(실측: 6s 생성 + 120s 꼬리 = 126.4s). 스트림은 완료 즉시 FIN.
    let _ = stream.shutdown(std::net::Shutdown::Write);
}

/// JSON 배열 내 최상위 객체 조각들 추출 — 중괄호 균형(문자열 리터럴 내부
/// { } 무시). jmessages_render·jcontent가 공유(A14 — 원본은
/// jmessages_render 인라인이었다).
fn jblocks(arr: &str) -> Vec<&str> {
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
fn jcontent(o: &str) -> Option<String> {
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

/// 스트림 누적 텍스트에서 stop 문자열 최초 등장 — (바이트 위치, 길이).
fn earliest_stop(acc: &str, stops: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(p) = acc.find(s.as_str())
            && best.is_none_or(|(bp, _)| p < bp)
        {
            best = Some((p, s.len()));
        }
    }
    best
}

/// 스트림 델타 청크 송신 — chat은 P0-4 표준 청크, completions는 종전 text 프레임.
fn emit_delta(
    stream: &mut TcpStream,
    fmt: &EmitFmt,
    id: &str,
    created: u64,
    model_esc: &str,
    piece: &str,
) -> std::io::Result<()> {
    if fmt.chat {
        sse(
            stream,
            "message",
            &format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{model_esc}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{piece}\"}},\"finish_reason\":null}}]}}"
            ),
        )
    } else {
        sse(stream, "message", &format!("{{\"text\":\"{piece}\"}}"))
    }
}

fn run_and_emit_anthropic(
    stream: &mut TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stream_mode: bool,
    sampler: Option<llm170_core::sampler::SamplerParams>,
    // A14: stop_sequences — jstop이 문자열·배열 모두 파싱.
    stop_strs: Vec<String>,
) {
    // A14: ctx 사전 검증 — run_and_emit과 동일(엔진 Err→500보다 400이 정확).
    let ctx = *SERVER_CTX.get().unwrap_or(&4096);
    if ids.len() + n_predict + 8 >= ctx {
        resp(
            stream,
            400,
            "application/json",
            &format!(
                "{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":\"context too small: prompt {} + max_tokens {} >= ctx {}\"}}}}",
                ids.len(),
                n_predict,
                ctx
            ),
        );
        return;
    }
    let Ok((orx, prx)) = enqueue_job(stream, &tx, ids, n_predict, vec![STOP_EOT], sampler) else {
        return;
    };
    if stream_mode {
        resp_sse_open(stream);
        let _ = sse(
            stream,
            "message_start",
            "{\"type\":\"message_start\",\"message\":{\"role\":\"assistant\"}}",
        );
        let mut det = crate::engine::Detok::new();
        // A14: stop_sequences holdback — run_and_emit과 동일 원리(누적 텍스트에서
        // stop 최초 등장 직전까지만 방출, 잠재 멀티바이트 경계는 floor_char_boundary).
        let mut acc = String::new();
        let mut sent = 0usize;
        let mut stopped = false;
        for t in prx {
            acc.push_str(&det.push(t));
            if let Some((sp, _)) = earliest_stop(&acc, &stop_strs) {
                if sp > sent {
                    let esc = crate::json::esc(&acc[sent..sp]);
                    let _ = sse(
                        stream,
                        "content_block_delta",
                        &format!(
                            "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                        ),
                    );
                }
                stopped = true;
                break;
            }
            let hold = stop_strs
                .iter()
                .map(|s| s.len().saturating_sub(1))
                .max()
                .unwrap_or(0);
            let mut safe = acc.len().saturating_sub(hold);
            safe = floor_char_boundary(&acc, safe);
            if safe > sent {
                let esc = crate::json::esc(&acc[sent..safe]);
                let _ = sse(
                    stream,
                    "content_block_delta",
                    &format!(
                        "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                    ),
                );
                sent = safe;
            }
        }
        if !stopped && acc.len() > sent {
            let esc = crate::json::esc(&acc[sent..]);
            let _ = sse(
                stream,
                "content_block_delta",
                &format!(
                    "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                ),
            );
        }
        let _ = orx.recv();
        let reason = if stopped { "stop_sequence" } else { "end_turn" };
        let _ = sse(
            stream,
            "message_delta",
            &format!("{{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{reason}\"}}}}"),
        );
        let _ = sse(stream, "message_stop", "{\"type\":\"message_stop\"}");
        let _ = stream.shutdown(std::net::Shutdown::Write);
        return;
    }
    // QA-3/9: run_and_emit 비스트림과 동일 — 폴링 대기로 절단 감지 + 에러 전파.
    let mut all = Vec::new();
    let err: Option<String>;
    loop {
        match orx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(r) => {
                all.extend(r.tokens);
                err = r.error;
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if peer_gone(stream) {
                    return;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                err = Some("engine result channel closed".into());
                break;
            }
        }
    }
    if let Some(e) = err {
        resp(
            stream,
            500,
            "application/json",
            &format!(
                "{{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":\"{}\"}}}}",
                crate::json::esc(&e)
            ),
        );
        return;
    }
    let mut det = crate::engine::Detok::new();
    let mut text: String = all.iter().map(|&t| det.push(t)).collect();
    // A14: stop_sequences 절단(비스트림) — stop 본문 미포함이 Anthropic 규약.
    let stopped = earliest_stop(&text, &stop_strs);
    if let Some((sp, _)) = stopped {
        text.truncate(sp);
    }
    let esc = crate::json::esc(&text);
    resp(
        stream,
        200,
        "application/json",
        &format!(
            "{{\"id\":\"msg_llm170\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"{esc}\"}}],\"stop_reason\":\"{}\"}}",
            if stopped.is_some() {
                "stop_sequence"
            } else {
                "end_turn"
            }
        ),
    );
}

#[cfg(test)]
mod http_tests {
    //! A3: 핸드롤 JSON 파서군·경계 유틸·렌더의 유닛테스트 — 전부
    //! CPU 순수(무 GPU). 변형 JSON·UTF-8 절단·stop 오버랩·블록 content가
    //! 종전 무검증이었다.

    use super::*;

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
        assert_eq!(earliest_stop("xxAByy", &stops), Some((2, 2)));
        assert_eq!(earliest_stop("", &stops), None);
        // 빈 stop은 무시(무한 절단 방지 계약)
        assert_eq!(earliest_stop("any", &[String::new()]), None);
        // 가장 이른 등장 선택
        let two = vec!["YY".to_string(), "XX".to_string()];
        assert_eq!(earliest_stop("aXXbYY", &two), Some((1, 2)));
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
}
