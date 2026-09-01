//! HTTP/1.1 서버 — OpenAI·Anthropic 호환 엔드포인트.
//!
//! 의존 0 (std::net) — 수동 파싱, 단일 스레드 요청 직렬 처리
//! (LLM 디코드가 병목이라 동시성 불필요).
//! 엔진은 백그라운드 스레드 1개가 요청 큐를 소비 — 생성 중 소켓은 SSE로 스트리밍.
//!
//! 엔드포인트:
//!   GET  /health                      — {"status":"ok"}
//!   GET  /v1/models                   — 모델 목록
//!   POST /tokenize                    — {"content"} → {"tokens":[...]} (탐욕 최장일치)
//!   POST /v1/completions              — prompt: 토큰 id 배열|텍스트, greedy
//!   POST /v1/chat/completions         — messages[].content (단순 연결), SSE 스트림
//!   POST /v1/messages (Anthropic)     — messages[].content, SSE 스트리밍
//!
//! 토크나이저는 탐욕 최장일치 근사 — 자기일관(self-consistent) 검증용.
//! llama.cpp 토큰 경계와 완전 일치하지 않음 (주석 참조).


use crate::engine::{BackendSel, InferRequest, InferResult, SlotJob};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

pub fn serve(addr: &str, req: InferRequest, backend: BackendSel) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| e.to_string())?;
    eprintln!("# llm170-server listening on http://{addr}");
    let slots = std::env::var("LLM170_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 16);
    let qcap = std::env::var("LLM170_QUEUE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64);
    let (tx, rx) = std::sync::mpsc::sync_channel::<SlotJob>(qcap);
    let eng = crate::engine::build_slots(req.clone(), backend, slots);
    std::thread::spawn(move || crate::engine::slot_loop(eng, rx, slots));
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let tx = tx.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, tx);
        });
    }
    Ok(())
}

pub struct Job {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
    /// true면 상태 리셋 후 prefill (무상태 HTTP 요청의 기본).
    pub fresh: bool,
    /// 토큰별 진행 채널 — 스트리밍 모드에서 생성 즉시 SSE 전송 (2026-09-01).
    pub progress: Option<std::sync::mpsc::Sender<u32>>,
}

pub type TokOut = InferResult;

struct HttpReq {
    method: String,
    path: String,
    body: String,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpReq, String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).map_err(|e| e.to_string())?;
        if h.trim().is_empty() {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    Ok(HttpReq {
        method,
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn resp(stream: &mut TcpStream, code: u16, ct: &str, body: &str) {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
        body.len()
    );
}

fn resp_sse_open(stream: &mut TcpStream) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n"
    );
}

fn sse(stream: &mut TcpStream, event: &str, data: &str) {
    let _ = write!(stream, "event: {event}\ndata: {data}\n\n");
    let _ = stream.flush();
}

// --- 최소 JSON 파싱 (중첩 없는 평탄 필드 추출) ---
fn jstr(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)? + pat.len();
    let b = body[i..].trim_start();
    if !b.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut esc = false;
    for c in b[1..].chars() {
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
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn jnum(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)? + pat.len();
    let b = body[i..].trim_start();
    let end = b
        .find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'))
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

/// messages[].content 전개 — 다중 문자열 연결 (최소 파서).
fn jmessages_content(body: &str) -> String {
    let mut out = String::new();
    if let Some(start) = body.find("\"messages\"") {
        let seg = &body[start..];
        let mut idx = 0;
        while let Some(cpos) = seg[idx..].find("\"content\"") {
            let abs = idx + cpos;
            let after = &seg[abs..];
            if let Some(c) = jstr(after, "content") {
                out.push_str(&c);
                out.push('\n');
            }
            idx = abs + 9;
        }
    }
    out
}

fn handle(mut stream: TcpStream, tx: std::sync::mpsc::SyncSender<SlotJob>) -> Result<(), String> {
    loop {
        let req = match read_request(&mut stream) {
            Ok(r) => r,
            Err(_) => return Ok(()), // 연결 종료
        };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => resp(&mut stream, 200, "application/json", "{\"status\":\"ok\"}"),
            ("GET", "/v1/models") => resp(
                &mut stream,
                200,
                "application/json",
                "{\"object\":\"list\",\"data\":[{\"id\":\"llm170\",\"object\":\"model\",\"owned_by\":\"local\"}]}",
            ),
            ("POST", "/tokenize") => {
                let Some(content) = jstr(&req.body, "content") else {
                    resp(&mut stream, 400, "application/json", "{\"error\":\"content required\"}");
                    continue;
                };
                let toks = crate::engine::greedy_encode(&content);
                let ids: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
                resp(&mut stream, 200, "application/json", &format!("{{\"tokens\":[{}]}}", ids.join(",")));
            }
            ("POST", "/v1/completions") | ("POST", "/completion") => {
                let n_predict = jnum(&req.body, "n_predict").unwrap_or(24.0).max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let prompt_ids = jarr_u32(&req.body, "prompt");
                let prompt_txt = jstr(&req.body, "prompt");
                let ids = match (prompt_ids, prompt_txt) {
                    (Some(v), _) if !v.is_empty() => v,
                    (_, Some(t)) => crate::engine::greedy_encode(&t),
                    _ => {
                        resp(&mut stream, 400, "application/json", "{\"error\":\"prompt required\"}");
                        continue;
                    }
                };
                run_and_emit(&mut stream, tx.clone(), ids, n_predict, stream_mode, req.path.contains("chat"));
            }
            ("POST", "/v1/chat/completions") => {
                let n_predict = jnum(&req.body, "max_tokens").unwrap_or(jnum(&req.body, "n_predict").unwrap_or(24.0)).max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let text = jmessages_content(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit(&mut stream, tx.clone(), ids, n_predict, stream_mode, true);
            }
            ("POST", "/v1/messages") => {
                let n_predict = jnum(&req.body, "max_tokens").unwrap_or(24.0).max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let text = jmessages_content(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit_anthropic(&mut stream, tx.clone(), ids, n_predict, stream_mode);
            }
            _ => resp(&mut stream, 404, "application/json", "{\"error\":\"not found\"}"),
        }
    }
}

fn run_and_emit(
    stream: &mut TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stream_mode: bool,
    chat: bool,
) {
    // ctx 검증 — 프롬프트+생성이 컨텍스트를 넘으면 400 (context-shift v1:
    // 슬롯 무상태라 이동 없이 거절 — 이동 재배치는 접두 캐시 도입 시).
    let ctx = std::env::var("LLM170_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    if ids.len() + n_predict + 8 >= ctx {
        resp(stream, 400, "application/json", &format!(
            "{{\"error\":\"context too small: prompt {} + n_predict {} >= ctx {}\"}}",
            ids.len(), n_predict, ctx));
        return;
    }
    let (otx, orx) = std::sync::mpsc::channel::<TokOut>();
    let (ptx, prx) = std::sync::mpsc::channel::<u32>();
    let job = SlotJob {
        tokens: ids,
        n_predict,
        progress: stream_mode.then_some(ptx),
        out: otx,
    };
    if tx.try_send(job).is_err() {
        resp(stream, 503, "application/json", "{\"error\":\"queue full\"}");
        return;
    }
    if !stream_mode {
        let mut toks = Vec::new();
        if let Ok(r) = orx.recv() {
            toks = r.tokens;
        }
        let arr: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
        resp(
            stream,
            200,
            "application/json",
            &format!("{{\"tokens\":[{}],\"object\":\"completion\"}}", arr.join(",")),
        );
        return;
    }
    resp_sse_open(stream);
    // 토큰 생성 즉시 SSE — 장문 요청이 완료까지 굳지 않게 (2026-09-01).
    for t in prx {
        let piece = crate::engine::piece_escaped(t);
        if chat {
            sse(
                stream,
                "message",
                &format!("{{\"choices\":[{{\"delta\":{{\"content\":\"{piece}\"}}}}]}}"),
            );
        } else {
            sse(stream, "message", &format!("{{\"text\":\"{piece}\"}}"));
        }
    }
    let _ = orx.recv(); // 최종 결과 수령 (종료 정리)
    sse(stream, "done", "[DONE]");
}

fn run_and_emit_anthropic(
    stream: &mut TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stream_mode: bool,
) {
    let (otx, orx) = std::sync::mpsc::channel::<TokOut>();
    let (ptx, prx) = std::sync::mpsc::channel::<u32>();
    let job = SlotJob {
        tokens: ids,
        n_predict,
        progress: stream_mode.then_some(ptx),
        out: otx,
    };
    if tx.try_send(job).is_err() {
        resp(stream, 503, "application/json", "{\"error\":\"queue full\"}");
        return;
    }
    if stream_mode {
        resp_sse_open(stream);
        sse(stream, "message_start", "{\"type\":\"message_start\",\"message\":{\"role\":\"assistant\"}}");
        for t in prx {
            let p = crate::engine::piece_plain(t);
            let esc = p.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
            sse(
                stream,
                "content_block_delta",
                &format!("{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"),
            );
        }
        let _ = orx.recv();
        sse(stream, "message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}");
        sse(stream, "message_stop", "{\"type\":\"message_stop\"}");
        return;
    }
    let mut all = Vec::new();
    while let Ok(r) = orx.recv() {
        all.extend(r.tokens);
    }
    if !stream_mode {
        let mut text = String::new();
        for t in &all {
            text.push_str(&crate::engine::piece_plain(*t));
        }
        let esc = text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        resp(
            stream,
            200,
            "application/json",
            &format!("{{\"id\":\"msg_llm170\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"{esc}\"}}],\"stop_reason\":\"end_turn\"}}"),
        );
        return;
    }
    resp_sse_open(stream);
    sse(stream, "message_start", "{\"type\":\"message_start\",\"message\":{\"role\":\"assistant\"}}");
    for t in &all {
        let p = crate::engine::piece_plain(*t);
        let esc = p.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        sse(
            stream,
            "content_block_delta",
            &format!("{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"),
        );
    }
    sse(stream, "message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}");
    sse(stream, "message_stop", "{\"type\":\"message_stop\"}");
}
