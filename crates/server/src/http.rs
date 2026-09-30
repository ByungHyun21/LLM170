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

/// messages[].content 전개 — 다중 문자열 연결 (최소 파서).
/// qwen3.8 챗 템플릿 — 사용자 단일 턴 래핑 (이미 템플릿 포함 시 원문).
fn chat_template(content: &str) -> String {
    if content.contains("<|im_start|>") {
        return content.to_string();
    }
    format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
}

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
                    req.path.contains("chat"),
                    Vec::new(),
                    parse_sampler(&req.body),
                );
            }
            ("POST", "/v1/chat/completions") => {
                let n_predict = jnum(&req.body, "max_tokens")
                    .unwrap_or(jnum(&req.body, "n_predict").unwrap_or(24.0))
                    .max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let text = chat_template(&jmessages_content(&req.body));
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit(
                    &mut stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    true,
                    vec![STOP_EOT],
                    parse_sampler(&req.body),
                );
            }
            ("POST", "/v1/messages") => {
                let n_predict = jnum(&req.body, "max_tokens").unwrap_or(24.0).max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                let text = chat_template(&jmessages_content(&req.body));
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
    chat: bool,
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
        let mut det = crate::engine::Detok::new();
        let text: String = toks.iter().map(|&t| det.push(t)).collect();
        let arr: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
        let esc = crate::json::esc(&text);
        resp(
            stream,
            200,
            "application/json",
            &format!(
                "{{\"tokens\":[{}],\"text\":\"{esc}\",\"object\":\"completion\"}}",
                arr.join(",")
            ),
        );
        return;
    }
    resp_sse_open(stream);
    // 토큰 생성 즉시 SSE — 장문 요청이 완료까지 굳지 않게 (2026-09-01).
    let mut det = crate::engine::Detok::new();
    for t in prx {
        let piece = crate::json::esc(&det.push(t));
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
    // plans/114 QA-2 연계 수리: SSE 완료 후 연결 종료. curl류 클라이언트는
    // [DONE]을 인지하지 못해 서버의 keep-alive 대기에 묶였고 — 무타임아웃
    // 시대엔 무한 대기, read_timeout(120s) 도입 후엔 요청마다 +120s 꼬리가
    // 붙었다(실측: 6s 생성 + 120s 꼬리 = 126.4s). 스트림은 완료 즉시 FIN.
    let _ = stream.shutdown(std::net::Shutdown::Write);
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
        sse(
            stream,
            "message_start",
            "{\"type\":\"message_start\",\"message\":{\"role\":\"assistant\"}}",
        );
        let mut det = crate::engine::Detok::new();
        for t in prx {
            let esc = crate::json::esc(&det.push(t));
            sse(
                stream,
                "content_block_delta",
                &format!(
                    "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                ),
            );
        }
        let _ = orx.recv();
        sse(
            stream,
            "message_delta",
            "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}",
        );
        sse(stream, "message_stop", "{\"type\":\"message_stop\"}");
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
