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

use crate::engine::{InferRequest, InferResult, SlotJob};
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
    slots_flag: Option<usize>,
    queue_flag: Option<usize>,
) -> Result<(), String> {
    let _ = SERVER_CTX.set(req.ctx);
    // 107 P0-9: 슬롯 수 소스 계통 가시화 — 플래그 > env > 기본 1.
    // 이전엔 env 기본 1이 조용히 직렬 서버를 만들었다(np4 10.5 t/s 정체).
    let (slots, src) = if let Some(n) = slots_flag {
        (n.clamp(1, 16), "flag")
    } else if let Some(n) =
        llm170_diag::flag::val("LLM170_SLOTS").and_then(|v| v.parse::<usize>().ok())
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
        None => llm170_diag::flag::val("LLM170_QUEUE")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<SlotJob>(qcap);
    let eng = crate::engine::build_slots(req.clone(), slots);
    // B20: 적재 완료 — 전역 적재 락 해제(다음 기동의 재판정이 이 상주분을 본다).
    crate::resource::release_load_lock();
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
    // [H 2026-10-09] 연결 상한 — 종전엔 수락마다 스레드를 무제한 생성했다.
    // 초과분은 503 후 즉시 종료(스레드·소켓 자원 보호).
    let conns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        // [2026-10-09 P5] 소형 SSE 프레임 × Nagle 지연 제거 — 연결당 1회.
        let _ = stream.set_nodelay(true);
        // [H] 쓰기 타임아웃 — 소켓 버퍼가 찬 slow consumer에 write가 영구
        // 블록되어 워커 스레드가 매달리는 것 방지(SSE는 write 실패로 조기 종료).
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(120)));
        let n = conns.fetch_add(1, Ordering::AcqRel);
        if n >= MAX_CONNS {
            conns.fetch_sub(1, Ordering::AcqRel);
            let _ = write!(
                stream,
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            continue;
        }
        let guard = ConnGuard(conns.clone());
        let tx = tx.clone();
        std::thread::spawn(move || {
            let _ = crate::oai::handle(stream, tx);
            drop(guard);
        });
    }
    Ok(())
}

/// 동시 연결 상한(수락 스레드 폭주 방지) — 슬롯·큐와 무관한 전송 계층 가드.
const MAX_CONNS: usize = 256;

/// 연결 카운터 감소 가드 — accept 실패/패닉 경로에서도 누수 없이 감소.
struct ConnGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub type TokOut = InferResult;

pub(crate) struct HttpReq {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: String,
}

/// QA-2: 요청 본문 상한 — Content-Length 무상한 vec![0; len]이 가상 메모리
/// +16.8GB 점유(2026-09-30 실측) 후 read_exact 영구 블록. 프롬프트 JSON 여유.
const MAX_BODY: usize = 64 << 20;

/// 라인 상한 — 요청 라인·헤더 라인 각각 8KB(S2). `read_line`은 무상한이라
/// `\n` 미송신 클라가 타임아웃(120s)까지 버퍼를 키울 수 있었다.
const MAX_LINE: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_HEADER_LINES: usize = 128;

/// 라인 리더 오류 — TooLong은 상태코드 응답(431/414) 대상.
#[derive(Debug)]
enum LineErr {
    Io(std::io::Error),
    TooLong,
}

/// `\n`까지 최대 max 바이트 라인 1개 — EOF면 None(S2).
fn read_line_bounded<R: BufRead>(r: &mut R, max: usize) -> Result<Option<Vec<u8>>, LineErr> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        let (nl, used) = {
            let avail = r.fill_buf().map_err(LineErr::Io)?;
            if avail.is_empty() {
                return if out.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(out))
                };
            }
            let used = match avail.iter().position(|&b| b == b'\n') {
                Some(i) => i + 1,
                None => avail.len(),
            };
            out.extend_from_slice(&avail[..used]);
            (avail[used - 1] == b'\n', used)
        };
        r.consume(used);
        if out.len() > max {
            return Err(LineErr::TooLong);
        }
        if nl {
            return Ok(Some(out));
        }
    }
}

/// 요청 파싱 오류 — 상태 응답이 필요한 경우와 연결 종료를 구분(S2 테스트 표면).
#[derive(Debug)]
enum ReqErr {
    /// (상태코드, 사유문구, 내부 메시지) — 호출부가 응답을 송신.
    Status(u16, &'static str, String),
    /// 응답 없이 연결 종료(EOF 등).
    Plain(String),
}

/// 요청 1개 파싱 본체 — BufReader 인자(유닛테스트: Cursor 주입).
/// 리더를 밖에서 유지하면 파이프라인 잔여 바이트가 보존된다(소비는 정확히
/// 요청 1개분 — 프로덕션 keep-alive 재사용은 P2 항목).
fn read_request_from<R: Read>(reader: &mut BufReader<R>) -> Result<HttpReq, ReqErr> {
    let req_line = match read_line_bounded(reader, MAX_LINE) {
        Ok(Some(v)) => v,
        // EOF(keep-alive 연결이 끊긴 경우) — 종전엔 빈 요청으로 파싱돼 404 응답을
        // 무한히 재전송하는 스핀이 됐다(닫힌 소켓 read 는 즉시 0 반환): 유휴 서버가
        // 코어 하나를 태우고 system time 이 2/3 를 차지했다(2026-09-17 실측).
        Ok(None) => return Err(ReqErr::Plain("eof".into())),
        Err(LineErr::TooLong) => {
            return Err(ReqErr::Status(
                414,
                "URI Too Long",
                "request line too long".into(),
            ));
        }
        Err(LineErr::Io(e)) => return Err(ReqErr::Plain(e.to_string())),
    };
    let line = String::from_utf8_lossy(&req_line).into_owned();
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut len = 0usize;
    // A18: 헤더 라인 상한·개수 상한 — 무상한 push로 악성/고장
    // 클라가 수십 GB를 소비한 사고 재발 방지. 변형 CL·chunked는 즉시 400
    // (len=0로 조용히 빈 바디 처리하던 종전 동작은 오해 백롭).
    let mut hdr_lines = 0usize;
    let mut hdr_bytes = 0usize;
    let mut bad_cl = false;
    let mut chunked = false;
    loop {
        let h = match read_line_bounded(reader, MAX_LINE) {
            Ok(Some(v)) => v,
            Ok(None) => break, // 헤더 중 EOF — 아래 빈 헤더 취급(바디 0).
            Err(LineErr::TooLong) => {
                return Err(ReqErr::Status(
                    431,
                    "Request Header Fields Too Large",
                    "header line too long".into(),
                ));
            }
            Err(LineErr::Io(e)) => return Err(ReqErr::Plain(e.to_string())),
        };
        hdr_lines += 1;
        hdr_bytes += h.len();
        let h = String::from_utf8_lossy(&h).into_owned();
        if h.trim().is_empty() {
            break;
        }
        if hdr_lines > MAX_HEADER_LINES || hdr_bytes > MAX_HEADER_BYTES {
            return Err(ReqErr::Status(
                431,
                "Request Header Fields Too Large",
                "header too large".into(),
            ));
        }
        let hl = h.to_ascii_lowercase();
        if let Some(v) = hl.strip_prefix("content-length:") {
            match v.trim().parse::<usize>() {
                Ok(n) => len = n,
                Err(_) => bad_cl = true,
            }
        }
        if hl.starts_with("transfer-encoding:") && hl.contains("chunked") {
            chunked = true;
        }
    }
    if bad_cl {
        return Err(ReqErr::Status(
            400,
            "Bad Request",
            "malformed content-length".into(),
        ));
    }
    if chunked {
        // [A13] chunked 지원(2026-10-09 사용자 결정) — 크기줄(hex)[;ext] CRLF +
        // 데이터 CRLF 반복, 크기 0에서 종료(트레일러 헤더 소비). 본문 상한은
        // Content-Length 경로와 동일(MAX_BODY) — 초과 시 413(읽지 않고 거부).
        let mut body: Vec<u8> = Vec::new();
        loop {
            let szline = match read_line_bounded(reader, MAX_LINE) {
                Ok(Some(v)) => v,
                Ok(None) => return Err(ReqErr::Plain("chunked: EOF".into())),
                Err(LineErr::TooLong) => {
                    return Err(ReqErr::Status(
                        400,
                        "Bad Request",
                        "chunk size too long".into(),
                    ));
                }
                Err(LineErr::Io(e)) => return Err(ReqErr::Plain(e.to_string())),
            };
            let sztxt = String::from_utf8_lossy(&szline);
            let tok = sztxt.trim().split(';').next().unwrap_or("").trim();
            let n = match usize::from_str_radix(tok, 16) {
                Ok(n) => n,
                Err(_) => {
                    return Err(ReqErr::Status(
                        400,
                        "Bad Request",
                        "malformed chunk size".into(),
                    ));
                }
            };
            if n == 0 {
                // 트레일러: 빈 줄까지 소비(있으면).
                loop {
                    match read_line_bounded(reader, MAX_LINE) {
                        Ok(Some(v)) if !v.iter().all(u8::is_ascii_whitespace) => continue,
                        _ => break,
                    }
                }
                break;
            }
            if body.len() + n > MAX_BODY {
                return Err(ReqErr::Status(
                    413,
                    "Payload Too Large",
                    format!("chunked body exceeds limit {MAX_BODY}"),
                ));
            }
            let mut chunk = vec![0u8; n];
            reader
                .read_exact(&mut chunk)
                .map_err(|e| ReqErr::Plain(e.to_string()))?;
            body.extend_from_slice(&chunk);
            let mut crlf = [0u8; 2];
            reader
                .read_exact(&mut crlf)
                .map_err(|e| ReqErr::Plain(e.to_string()))?;
            if &crlf != b"\r\n" {
                return Err(ReqErr::Status(
                    400,
                    "Bad Request",
                    "malformed chunk terminator".into(),
                ));
            }
        }
        return Ok(HttpReq {
            method,
            path,
            body: String::from_utf8_lossy(&body).into_owned(),
        });
    }
    if len > MAX_BODY {
        // QA-2: 413 응답 후 절단 — 상한 초과 본문은 읽지도 않는다.
        return Err(ReqErr::Status(
            413,
            "Payload Too Large",
            format!("content-length {len} exceeds limit {MAX_BODY}"),
        ));
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| ReqErr::Plain(e.to_string()))?;
    }
    Ok(HttpReq {
        method,
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

/// 연결당 리더 1개를 받는다(H 2026-10-09) — 종전엔 호출마다 BufReader를
/// 새로 만들어, 선행 판독분(파이프라인 잔여 바이트)이 리더와 함께 폐기됐다.
pub(crate) fn read_request(reader: &mut BufReader<TcpStream>) -> Result<HttpReq, String> {
    // QA-2: 읽기 타임아웃 — 헤더/바디 미완 송신(절단·slow-loris) 영구 블록 방지.
    let _ = reader
        .get_ref()
        .set_read_timeout(Some(std::time::Duration::from_secs(120)));
    match read_request_from(reader) {
        Ok(r) => Ok(r),
        Err(ReqErr::Status(code, reason, msg)) => {
            let _ = write!(
                reader.get_mut(),
                "HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            Err(msg)
        }
        Err(ReqErr::Plain(m)) => Err(m),
    }
}

pub(crate) fn resp(stream: &mut TcpStream, code: u16, ct: &str, body: &str) {
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

pub(crate) fn resp_sse_open(stream: &mut TcpStream) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n"
    );
}

/// [I 2026-10-10] SSE 주석 프레임 — ':'으로 시작하는 줄은 클라이언트가 무시.
/// 토큰 공백(장문 프리필 TTFT 수십 초 등)에 보내 프록시·클라 타임아웃을 막는다.
pub(crate) fn sse_comment_frame(text: &str) -> String {
    format!(": {text}\n\n")
}

pub(crate) fn sse_comment(stream: &mut TcpStream, text: &str) -> std::io::Result<()> {
    stream.write_all(sse_comment_frame(text).as_bytes())?;
    stream.flush()
}

/// [D 2026-10-10] 어드미션 게이트 응답 — 429 Too Many Requests + Retry-After.
/// 큐 포화(하드게이트) 전용 — 로딩/엔진 사망은 503(호출부 판단).
pub(crate) fn resp_429(stream: &mut TcpStream, body: &str) {
    let _ = stream.write_all(admission_429(body).as_bytes());
}

/// 429 응답 조립 — 순수 함수(전송 계약 테스트 표면).
pub(crate) fn admission_429(body: &str) -> String {
    format!(
        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nRetry-After: 1\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
        body.len()
    )
}

/// 쓰기 오류를 반환한다 — 종전 `let _ =`가 절단된
/// 클라이언트로의 쓰기 실패를 삼켜, 잔여 n_predict를 GPU가 끝까지 계산했다.
/// [2026-10-09 P5] 프레임을 문자열로 조립해 단일 write_all — 종전 write!는
/// 포맷 조각마다 write(2)(프레임당 ~5회) + Nagle과 겹쳐 스톨 소지.
/// 값/의미 불변(전송 바이트 동일).
pub(crate) fn sse(stream: &mut TcpStream, event: &str, data: &str) -> std::io::Result<()> {
    let mut frame = String::with_capacity(event.len() + data.len() + 16);
    frame.push_str("event: ");
    frame.push_str(event);
    frame.push_str("\ndata: ");
    frame.push_str(data);
    frame.push_str("\n\n");
    stream.write_all(frame.as_bytes())?;
    stream.flush()
}

#[cfg(test)]
mod req_tests {
    //! S2: 요청 파싱 매트릭스 — 431/414/400/413/EOF/파이프라인 소비.
    //! `read_request_from`이 impl Read를 받으므로 Cursor로 전수 검증한다.

    use super::*;
    use std::io::{BufReader, Cursor};
    fn parse(bytes: &[u8]) -> Result<HttpReq, ReqErr> {
        let mut r = BufReader::new(Cursor::new(bytes.to_vec()));
        read_request_from(&mut r)
    }

    #[test]
    fn get_and_post_ok() {
        let r = parse(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").expect("GET");
        assert_eq!((r.method.as_str(), r.path.as_str()), ("GET", "/health"));
        assert!(r.body.is_empty());

        let body = "{\"prompt\":\"hi\"}";
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let r = parse(req.as_bytes()).expect("POST");
        assert_eq!(r.body, body);
    }

    #[test]
    fn chunked_body_ok() {
        // [A13] chunked 요청 — 조각 2개 + 종료(0) + 트레일러.
        let body = "{\"prompt\":\"hi\"}";
        let (a, b) = body.split_at(5);
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\nX-T: 1\r\n\r\n",
            a.len(),
            b.len()
        );
        let r = parse(req.as_bytes()).expect("chunked");
        assert_eq!(r.body, body);
        // 잘못된 크기줄 → 400.
        match parse(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n") {
            Err(ReqErr::Status(400, _, _)) => {}
            _ => panic!("400 기대(chunk size)"),
        }
    }

    #[test]
    fn long_lines_are_bounded() {
        // 요청 라인 8KB 초과(개행 없음) → 414.
        let mut line = b"GET /".to_vec();
        line.extend(std::iter::repeat_n(b'a', MAX_LINE + 16));
        match parse(&line) {
            Err(ReqErr::Status(414, _, _)) => {}
            other => panic!("414 기대: {:?}", other.err().map(|e| format!("{e:?}"))),
        }
        // 헤더 라인 8KB 초과 → 431.
        let mut req = b"GET / HTTP/1.1\r\nX-Big: ".to_vec();
        req.extend(std::iter::repeat_n(b'b', MAX_LINE + 16));
        req.extend_from_slice(b"\r\n\r\n");
        match parse(&req) {
            Err(ReqErr::Status(431, _, _)) => {}
            _ => panic!("431 기대(헤더 라인)"),
        }
        // 헤더 라인 수 128 초과 → 431.
        let mut req = b"GET / HTTP/1.1\r\n".to_vec();
        for i in 0..(MAX_HEADER_LINES + 4) {
            req.extend_from_slice(format!("X-{i}: 1\r\n").as_bytes());
        }
        req.extend_from_slice(b"\r\n");
        match parse(&req) {
            Err(ReqErr::Status(431, _, _)) => {}
            _ => panic!("431 기대(헤더 수)"),
        }
    }

    #[test]
    fn cl_and_te_guards() {
        // 변형 CL → 400.
        match parse(b"POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n") {
            Err(ReqErr::Status(400, _, _)) => {}
            _ => panic!("400 기대(CL)"),
        }
        // chunked는 A13으로 지원 — 정상 종료(0)까지 오면 빈 본문 수용,
        // 미완(EOF)은 Plain 오류(400 아님).
        match parse(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n") {
            Ok(r) => assert!(r.body.is_empty()),
            Err(e) => panic!("chunked 수용 기대: {e:?}"),
        }
        // 상한 초과 CL → 413.
        let req = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        );
        match parse(req.as_bytes()) {
            Err(ReqErr::Status(413, _, _)) => {}
            _ => panic!("413 기대"),
        }
    }

    #[test]
    fn eof_and_pipeline_consumption() {
        // 빈 입력 → EOF(응답 없음).
        match parse(b"") {
            Err(ReqErr::Plain(m)) => assert_eq!(m, "eof"),
            _ => panic!("eof 기대"),
        }
        // 파이프라인: 리더를 유지하면 두 번째 요청이 그대로 남아 있다.
        let two = b"GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(two.to_vec()));
        let a = read_request_from(&mut r).expect("1st");
        let b = read_request_from(&mut r).expect("2nd");
        assert_eq!(a.path, "/a");
        assert_eq!(b.path, "/b");
    }

    /// [D] 어드미션 게이트 429 — 상태줄·Retry-After·Content-Length 계약.
    #[test]
    fn admission_429_contract() {
        let body = "{\"error\":\"queue full — admission gate\"}";
        let r = admission_429(body);
        assert!(r.starts_with("HTTP/1.1 429 Too Many Requests\r\n"), "{r}");
        assert!(r.contains("\r\nRetry-After: 1\r\n"), "{r}");
        assert!(
            r.contains(&format!("\r\nContent-Length: {}\r\n", body.len())),
            "{r}"
        );
        assert!(r.ends_with(&format!("\r\n\r\n{body}")), "{r}");
    }

    /// [I] SSE keep-alive 주석 프레임 — ':' 줄 + 빈 줄(클라 무시 계약).
    #[test]
    fn sse_comment_frame_contract() {
        assert_eq!(sse_comment_frame("keep-alive"), ": keep-alive\n\n");
    }
}
