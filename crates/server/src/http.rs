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
use std::sync::atomic::{AtomicBool, Ordering};

/// 기동 준비 완료 플래그 — 기동 워밍업(slot_loop 진입 시) 전에는 /health가 503.
/// llama-server의 /health가 모델 로드·슬롯 초기화 후에야 200을 주는 것과 같은
/// 계약이다(2026-09-17: 워밍업 없는 첫 요청이 지연 초기화 raw_init을 뒤집어써
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
pub static READY: AtomicBool = AtomicBool::new(false);
/// QA-11: slot_loop 스레드 사망(패닉) — 신규 요청을 "queue full" 오보 503 대신
/// 명시적 사유로 거부하고 진행 중 요청의 채널 단절을 감지 가능하게.
pub static ENGINE_DEAD: AtomicBool = AtomicBool::new(false);
/// 서버 ctx 상한 — serve --ctx 값을 핸들러에 전달(107 W2: 종전
/// LLM170_CTX env 기본 4096이 --ctx 8192 엔진과 불일치해 조용히 거절).
pub static SERVER_CTX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub fn serve(
    addr: &str,
    req: InferRequest,
    backend: BackendSel,
    slots_flag: Option<usize>,
    queue_flag: Option<usize>,
) -> Result<(), String> {
    let _ = SERVER_CTX.set(req.ctx);
    // 107 P0-9: 슬롯 수 소스 계통 가시화 — 플래그 > env > 기본 1.
    // 이전엔 env 기본 1이 조용히 직렬 서버를 만들었다(np4 10.5 t/s 정체).
    let (slots, src) = if let Some(n) = slots_flag {
        (n.clamp(1, 16), "flag")
    } else if let Some(n) = std::env::var("LLM170_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        (n.clamp(1, 16), "env LLM170_SLOTS")
    } else {
        (1, "default")
    };
    eprintln!("# serve: slots={slots} (source: {src}) — 복수 동시 요청 배치 디코드에는 --slots N");
    // 대기열 기본 512(2026-09-16, 사용자 지시): 대기 작업은 토큰 배열+채널뿐인
    // 호스트 객체(건당 수백 바이트)라 넉넉해도 비용이 없고, 동시 요청 폭주 시
    let qcap = match queue_flag {
        Some(q) => q,
        None => std::env::var("LLM170_QUEUE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<SlotJob>(qcap);
    let eng = crate::engine::build_slots(req.clone(), backend, slots);
    // QA-11: 엔진 스레드 패닉 포착 — 종전엔 스레드가 죽어도 큐가 살아
    // try_send 성공 → 요청이 영구 행업, 원인은 stderr 1회뿐이었다.
    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::engine::slot_loop(eng, rx, slots)
        }));
        if r.is_err() {
            eprintln!("FATAL: slot_loop 패닉 — 엔진 사망. /health 503, 신규 요청 거부.");
            ENGINE_DEAD.store(true, Ordering::Release);
            READY.store(false, Ordering::Release);
        }
    });
    // QA-10: 바인딩은 적재 후 — 종전 역순(bind→적재)은 커널 백로그가 TCP만
    // 받아주는 무응답 창을 만들었다(llama-server은 적재 완료 후 바인딩).
    // READY 503("loading") 계약이 실제로 관측 가능해진다(워밍업 창).
    let listener = match TcpListener::bind(addr) {
        Ok(l) => l,
        Err(e) => return Err(e.to_string()),
    };
    eprintln!("# llm170-server listening on http://{addr}");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let tx = tx.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, tx);
        });
    }
    Ok(())
}

pub type TokOut = InferResult;

struct HttpReq {
    method: String,
    path: String,
    body: String,
}

/// QA-2: 요청 본문 상한 — Content-Length 무상한 vec![0; len]이 가상 메모리
/// +16.8GB 점유(2026-09-30 실측) 후 read_exact 영구 블록. 프롬프트 JSON 여유.
const MAX_BODY: usize = 64 << 20;

fn read_request(stream: &mut TcpStream) -> Result<HttpReq, String> {
    // QA-2: 읽기 타임아웃 — 헤더/바디 미완 송신(절단·slow-loris) 영구 블록 방지.
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(120)));
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    // EOF(keep-alive 연결이 끊긴 경우) — 종전엔 빈 요청으로 파싱돼 404 응답을
    // 무한히 재전송하는 스핀이 됐다(닫힌 소켓 read 는 즉시 0 반환): 유휴 서버가
    // 코어 하나를 태우고 system time 이 2/3 를 차지했다(2026-09-17 실측).
    if line.is_empty() {
        return Err("eof".into());
    }
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
    if len > MAX_BODY {
        // QA-2: 413 응답 후 절단 — 상한 초과 본문은 읽지도 않는다.
        // (stream은 reader로 이동됐으므로 get_mut 재차용)
        let _ = write!(
            reader.get_mut(),
            "HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        return Err(format!("content-length {len} exceeds limit {MAX_BODY}"));
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
        413 => "Payload Too Large", // QA-8: 폴백 "OK"로 "413 OK"가 나가던 결함
        500 => "Internal Server Error",
        503 => "Service Unavailable", // QA-8: 워밍업 /health가 "503 OK"였다
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

/// plans/113(sglang P0-1): 쓰기 오류를 반환한다 — 종전 `let _ =`가 절단된
/// 클라이언트로의 쓰기 실패를 삼켜, 잔여 n_predict를 GPU가 끝까지 계산했다.
fn sse(stream: &mut TcpStream, event: &str, data: &str) -> std::io::Result<()> {
    write!(stream, "event: {event}\ndata: {data}\n\n")?;
    stream.flush()
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
/// jstr·jarr_str 공용 (plans/123 P0-2·P0-3).
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

/// plans/123 P0-3: 요청 stop 파싱 — OpenAI "stop"(문자열|배열)과 Anthropic
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

/// plans/123 P0-1: 유닉스 초.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// plans/123 P0-1: OAI 응답 id — 프로세스 단조 카운터로 충돌 없는 유일값.
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

/// plans/123 P0-1/3/4: 응답 포맷 컨텍스트 — 엔드포인트별 조립에 필요한 최소값.
struct EmitFmt {
    chat: bool,
    model: String,
    stop_strs: Vec<String>,
    include_stop: bool,
}

/// plans/123 P0-2: role 인지 멀티턴 렌더링 — system/user/assistant 턴별
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
    // 객체 순회 — 중괄호 균형(문자열 리터럴 내부 { } 는 건너뜸).
    let mut objs: Vec<&str> = Vec::new();
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
    // 원문 연결(종전 출력) — content에 이미 템플릿이 있으면 원문 통과(종전 동작).
    let mut raw = String::new();
    for o in &objs {
        if let Some(c) = jstr(o, "content") {
            raw.push_str(&c);
            raw.push('\n');
        }
    }
    if raw.contains("<|im_start|>") {
        return raw;
    }
    let mut out = String::new();
    let mut last_assistant = false;
    for o in &objs {
        let role = jstr(o, "role").unwrap_or_else(|| "user".into());
        let content = jstr(o, "content").unwrap_or_default();
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

fn handle(mut stream: TcpStream, tx: std::sync::mpsc::SyncSender<SlotJob>) -> Result<(), String> {
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
                    // plans/123 P0-1: completions는 text_completion 포맷.
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
                // plans/123 P0-2: role 인지 멀티턴 렌더링(시스템 프롬프트 보존).
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

/// 슬롯 잡 enqueue 공통 (plans/109 P5) — 채널 쌍 생성·SlotJob 조립·큐 송신.
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
        // plans/123 P0-3: stop 문자열 응답 절단 — 토큰 순회하며 누적 텍스트에
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
        // plans/123 P0-1: OAI 표준 비스트림 응답(id/created/model/object/
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
        // plans/123 P0-4: 첫 청크 — role delta(id/created/model 포함 표준 계약).
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
    // plans/123 P0-3: 스트림 stop 처리 — 누적 텍스트 기준 판정 + holdback으로
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
            // plans/113(sglang P0-1): 쓰기 실패(클라 절단) 시 즉시 탈출 — 이 스코프를
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
        // plans/123 P0-4: 종료 청크 — finish_reason 후 [DONE](기존 계약 유지).
        let _ = sse(
            stream,
            "message",
            &format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{model_esc}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}]}}"
            ),
        );
    }
    let _ = sse(stream, "done", "[DONE]");
    // plans/114 QA-2 연계 수리: SSE 완료 후 연결 종료. curl류 클라이언트는
    // [DONE]을 인지하지 못해 서버의 keep-alive 대기에 묶였고 — 무타임아웃
    // 시대엔 무한 대기, read_timeout(120s) 도입 후엔 요청마다 +120s 꼬리가
    // 붙었다(실측: 6s 생성 + 120s 꼬리 = 126.4s). 스트림은 완료 즉시 FIN.
    let _ = stream.shutdown(std::net::Shutdown::Write);
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
) {
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
        for t in prx {
            let esc = crate::json::esc(&det.push(t));
            let frame = sse(
                stream,
                "content_block_delta",
                &format!(
                    "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                ),
            );
            // plans/113(sglang P0-1): run_and_emit과 동일 — 절단 시 즉시 취소.
            if frame.is_err() {
                return;
            }
        }
        let _ = orx.recv();
        let _ = sse(
            stream,
            "message_delta",
            "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}",
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
    let text: String = all.iter().map(|&t| det.push(t)).collect();
    let esc = crate::json::esc(&text);
    resp(
        stream,
        200,
        "application/json",
        &format!(
            "{{\"id\":\"msg_llm170\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"{esc}\"}}],\"stop_reason\":\"end_turn\"}}"
        ),
    );
}
