
use crate::engine::{InferRequest, InferResult, SlotJob};
use std::sync::atomic::{AtomicBool, Ordering};

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
pub static READY: AtomicBool = AtomicBool::new(false);
pub static ENGINE_DEAD: AtomicBool = AtomicBool::new(false);
pub static SERVER_CTX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
pub static SERVER_MODEL_MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub fn serve(
    addr: &str,
    req: InferRequest,
    slots: usize,
    queue_flag: Option<usize>,
) -> Result<(), String> {
    let qcap = match queue_flag {
        Some(q) => q,
        None => llm170_diag::flag::val("LLM170_QUEUE")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<SlotJob>(qcap);
    let eng = crate::engine::build_slots(req.clone(), slots);
    let pool_cap = eng.kv_cap_tokens();
    let cap = if pool_cap >= 512 {
        req.ctx.min(pool_cap)
    } else {
        req.ctx
    };
    let _ = SERVER_CTX.set(cap);
    let _ = SERVER_MODEL_MAX.set(req.model_max_ctx);
    eprintln!(
        "# effective cap: {cap} tokens = min(model max {}, kv-mem 풀 실블록 {pool_cap})",
        req.model_max_ctx
    );
    let (kv_vram, kv_host, host_blocks, total_blocks) = eng.kv_tier_bytes();
    let (free_vram, free_host) = eng.kv_tier_free();
    if host_blocks > 0 {
        eprintln!(
            "# kv-tier: VRAM {:.2}GiB({}블록) + 호스트 {:.2}GiB({host_blocks}블록) / 전체 {total_blocks}블록 · 오프로드율 {:.1}% · 자유 VRAM {free_vram}/호스트 {free_host}블록",
            kv_vram as f64 / (1u64 << 30) as f64,
            total_blocks - host_blocks,
            kv_host as f64 / (1u64 << 30) as f64,
            100.0 * host_blocks as f64 / total_blocks.max(1) as f64
        );
    }
    crate::resource::release_load_lock();
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
    let t_warm = std::time::Instant::now();
    while !READY.load(Ordering::Acquire) && !ENGINE_DEAD.load(Ordering::Acquire) {
        if t_warm.elapsed() > std::time::Duration::from_secs(300) {
            eprintln!("# 경고: 워밍업 300s 미종료 — 바인딩 강행(READY=false 유지)");
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let listener = match TcpListener::bind(addr) {
        Ok(l) => l,
        Err(e) => return Err(e.to_string()),
    };
    eprintln!("# llm170-server listening on http://{addr}");
    let conns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let _ = stream.set_nodelay(true);
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

const MAX_CONNS: usize = 256;

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

const MAX_BODY: usize = 64 << 20;

const MAX_LINE: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_HEADER_LINES: usize = 128;

#[derive(Debug)]
enum LineErr {
    Io(std::io::Error),
    TooLong,
}

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

#[derive(Debug)]
enum ReqErr {
    Status(u16, &'static str, String),
    Plain(String),
}

fn read_request_from<R: Read>(reader: &mut BufReader<R>) -> Result<HttpReq, ReqErr> {
    let req_line = match read_line_bounded(reader, MAX_LINE) {
        Ok(Some(v)) => v,
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
    let mut hdr_lines = 0usize;
    let mut hdr_bytes = 0usize;
    let mut bad_cl = false;
    let mut chunked = false;
    loop {
        let h = match read_line_bounded(reader, MAX_LINE) {
            Ok(Some(v)) => v,
            Ok(None) => break,
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

pub(crate) fn read_request(reader: &mut BufReader<TcpStream>) -> Result<HttpReq, String> {
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
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
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

pub(crate) fn sse_comment_frame(text: &str) -> String {
    format!(": {text}\n\n")
}

pub(crate) fn sse_comment(stream: &mut TcpStream, text: &str) -> std::io::Result<()> {
    stream.write_all(sse_comment_frame(text).as_bytes())?;
    stream.flush()
}

pub(crate) fn peer_alive(stream: &mut TcpStream) -> bool {
    use std::io::Read;
    if stream.set_nonblocking(true).is_err() {
        return true;
    }
    let mut b = [0u8; 1];
    let mut alive = match stream.read(&mut b) {
        Ok(0) => false,
        Ok(_) => true,
        Err(e) => !matches!(
            e.kind(),
            std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::UnexpectedEof
        ),
    };
    if stream.set_nonblocking(false).is_err() {
        alive = false;
    }
    alive
}

pub(crate) fn resp_429(stream: &mut TcpStream, body: &str) {
    let _ = stream.write_all(admission_429(body).as_bytes());
}

pub(crate) fn admission_429(body: &str) -> String {
    format!(
        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nRetry-After: 1\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
        body.len()
    )
}

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
        let body = "{\"prompt\":\"hi\"}";
        let (a, b) = body.split_at(5);
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\nX-T: 1\r\n\r\n",
            a.len(),
            b.len()
        );
        let r = parse(req.as_bytes()).expect("chunked");
        assert_eq!(r.body, body);
        match parse(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n") {
            Err(ReqErr::Status(400, _, _)) => {}
            _ => panic!("400 기대(chunk size)"),
        }
    }

    #[test]
    fn long_lines_are_bounded() {
        let mut line = b"GET /".to_vec();
        line.extend(std::iter::repeat_n(b'a', MAX_LINE + 16));
        match parse(&line) {
            Err(ReqErr::Status(414, _, _)) => {}
            other => panic!("414 기대: {:?}", other.err().map(|e| format!("{e:?}"))),
        }
        let mut req = b"GET / HTTP/1.1\r\nX-Big: ".to_vec();
        req.extend(std::iter::repeat_n(b'b', MAX_LINE + 16));
        req.extend_from_slice(b"\r\n\r\n");
        match parse(&req) {
            Err(ReqErr::Status(431, _, _)) => {}
            _ => panic!("431 기대(헤더 라인)"),
        }
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
        match parse(b"POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n") {
            Err(ReqErr::Status(400, _, _)) => {}
            _ => panic!("400 기대(CL)"),
        }
        match parse(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n") {
            Ok(r) => assert!(r.body.is_empty()),
            Err(e) => panic!("chunked 수용 기대: {e:?}"),
        }
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
        match parse(b"") {
            Err(ReqErr::Plain(m)) => assert_eq!(m, "eof"),
            _ => panic!("eof 기대"),
        }
        let two = b"GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(two.to_vec()));
        let a = read_request_from(&mut r).expect("1st");
        let b = read_request_from(&mut r).expect("2nd");
        assert_eq!(a.path, "/a");
        assert_eq!(b.path, "/b");
    }

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

    #[test]
    fn sse_comment_frame_contract() {
        assert_eq!(sse_comment_frame("keep-alive"), ": keep-alive\n\n");
    }
}
