//! OAI/Anthropic 페이로드 계층 (http.rs에서 순수 이동).
//! JSON 파서군(j*)·메시지 렌더·샘플러·stop 유틸·요청 라우팅(handle)과
//! OAI/Anthropic 방출 러너가 산다. 전송(resp/sse/read_request/serve)은
//! http.rs — 이 층은 바이트 해석과 프로토콜 스키만 담는다.
use crate::engine::SlotJob;
use crate::http::{ENGINE_DEAD, READY, SERVER_CTX, TokOut, read_request, resp, resp_sse_open, sse};
use std::net::TcpStream;
use std::sync::atomic::Ordering;

mod json;
mod stream;
#[cfg(test)]
mod tests;

use self::json::*;
use self::stream::*;

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

/// Anthropic 메시지 id — 별도 카운터(OAI id와 네임스페이스 분리).
fn anthropic_msg_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("msg_llm170-{n}")
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

/// qwen 어휘 <end_of_turn> — chat/anthropic 조기 정지 토큰 (EOS 248044와 구분).
const STOP_EOT: u32 = 248046;

pub(crate) fn handle(
    stream: TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
) -> Result<(), String> {
    // [H 2026-10-09] 연결당 BufReader 1개 — 요청마다 재생성하면 선행 판독분
    // (파이프라인 잔여 바이트)이 리더와 함께 폐기돼 keep-alive 파이프라이닝이
    // 조용히 깨졌다.
    let mut reader = std::io::BufReader::new(stream);
    loop {
        let req = match read_request(&mut reader) {
            Ok(r) => r,
            Err(_) => return Ok(()), // 연결 종료
        };
        let stream = reader.get_mut();
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => {
                if READY.load(Ordering::Acquire) {
                    resp(stream, 200, "application/json", "{\"status\":\"ok\"}")
                } else {
                    resp(stream, 503, "application/json", "{\"status\":\"loading\"}")
                }
            }
            ("GET", "/v1/models") => resp(
                stream,
                200,
                "application/json",
                "{\"object\":\"list\",\"data\":[{\"id\":\"llm170\",\"object\":\"model\",\"owned_by\":\"local\"}]}",
            ),
            // 모니터링 — 최신 스냅샷 1장(시계열 누적은 외부 폴러 몫).
            // /stats = JSON(주 타깃), /metrics = Prometheus 텍스트(표준 호환).
            ("GET", "/stats") => resp(stream, 200, "application/json", &crate::metrics::json()),
            ("GET", "/metrics") => resp(
                stream,
                200,
                "text/plain; version=0.0.4",
                &crate::metrics::prometheus(),
            ),
            ("POST", "/tokenize") => {
                let Some(content) = jstr(&req.body, "content") else {
                    resp(
                        stream,
                        400,
                        "application/json",
                        "{\"error\":\"content required\"}",
                    );
                    continue;
                };
                let toks = crate::engine::greedy_encode(&content);
                let ids: Vec<String> = toks.iter().map(|t| t.to_string()).collect();
                resp(
                    stream,
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
                            stream,
                            400,
                            "application/json",
                            "{\"error\":\"prompt required\"}",
                        );
                        continue;
                    }
                };
                run_and_emit(
                    stream,
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
                // [H 2026-10-09] 빈 messages 거부 — 렌더가 assistant 접두만
                // 남겨 빈 디코드로 200이 나가던 표면 불일치.
                if !has_message_content(&req.body) {
                    resp(
                        stream,
                        400,
                        "application/json",
                        "{\"error\":\"messages required\"}",
                    );
                    continue;
                }
                // role 인지 멀티턴 렌더링(시스템 프롬프트 보존).
                let text = jmessages_render(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit(
                    stream,
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
                // [H 2026-10-09] 빈 messages 거부 — chat과 동일 표면.
                if !has_message_content(&req.body) {
                    resp(
                        stream,
                        400,
                        "application/json",
                        "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"messages required\"}}",
                    );
                    continue;
                }
                let text = jmessages_render(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit_anthropic(
                    stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    parse_sampler(&req.body),
                    jstop(&req.body), // A14: stop_sequences(jstop이 배열 파싱)
                );
            }
            _ => resp(stream, 404, "application/json", "{\"error\":\"not found\"}"),
        }
    }
}

/// 슬롯 잡 enqueue 공통 — 채널 쌍 생성·SlotJob 조립·큐 송신.
/// Err면 이미 응답을 썼다(큐 포화 429 / 엔진 사망 503). 반환: (최종 결과
/// 수신기, 스트림 토큰 수신기 — 비스트림 모드는 진행 채널이 그대로 닫힌다).
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
        sampler,
        stops,
        // QA-3: 비스트림도 progress 채널 부여 — 핸들러가 prx를 잡고 폴링
        // 대기하며 절단 시 drop → slot_emit 송신 실패 → cancelled(스트림과
        // 동일 메커니즘). 종전 비스트림은 절단 감지 자체가 없었다.
        progress: Some(ptx),
        out: otx,
        queued: std::time::Instant::now(),
    };
    // [D 2026-10-10] 어드미션 하드게이트 — 큐 포화는 429(재시도 가능)로 구분.
    // 종전엔 사망/포화가 같은 503 "queue full"이라 클라이언트가 재시도 여부를
    // 판단할 수 없었다(Retry-After 부재).
    match tx.try_send(job) {
        Ok(()) => {
            // [B5/I] 큐 깊이 게이지 — 스케줄러 수신 시 -1(sched::slot_loop).
            crate::sched::SCHED
                .queue_depth
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Err(std::sync::mpsc::TrySendError::Full(_)) => {
            crate::http::resp_429(stream, "{\"error\":\"queue full — admission gate\"}");
            return Err(());
        }
        Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
            resp(
                stream,
                503,
                "application/json",
                "{\"error\":\"engine channel closed\"}",
            );
            return Err(());
        }
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
    // [H 2026-10-09] 빈 프롬프트는 전 엔드포인트 동일 400 — 종전엔 빈 배열은
    // 400인데 빈 문자열 prompt·빈 chat 렌더는 통과해 스케줄러가 0번 토큰에서
    // 조용히 디코드했다(표면 불일치).
    if ids.is_empty() {
        resp(
            stream,
            400,
            "application/json",
            "{\"error\":\"empty prompt\"}",
        );
        return;
    }
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
    // [R3] 디톡·홀드백·stop 스캔은 TextStream(Anthropic 러너와 공유).
    let mut ts = TextStream::new(fmt.stop_strs.clone(), fmt.include_stop);
    let mut ntok = 0usize;
    // [I 2026-10-10] keep-alive — 토큰 공백 5s마다 SSE 주석 프레임(장문
    // 프리필 TTFT 수십 초 동안 프록시·클라 타임아웃 방지). 쓰기 실패는
    // 즉시 탈출(prx drop → cancelled 경로로 슬롯 회수).
    loop {
        let t = match prx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(t) => t,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if crate::http::sse_comment(stream, "keep-alive").is_err() {
                    return;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        // 정지 토큰은 텍스트로 방출하지 않는다(비스트림은 finish_slot이
        // 트림하지만 스트림 델타는 여기서 걸러야 새어나가지 않는다).
        if t == llm170_core::qwen35::EOS_EOT || t == STOP_EOT {
            break;
        }
        ntok += 1;
        if let Some(piece) = ts.push(t) {
            let piece = crate::json::esc(piece);
            // 쓰기 실패(클라 절단) 시 즉시 탈출 — prx drop → cancelled 경로.
            if emit_delta(stream, fmt, &id, created, &model_esc, &piece).is_err() {
                return;
            }
        }
        if ts.stopped {
            break; // prx drop → 기존 cancelled 경로로 슬롯 회수
        }
    }
    // 잔여 보류분 플러시(stop 없이 종료 시).
    if let Some(piece) = ts.flush() {
        let piece = crate::json::esc(piece);
        let _ = emit_delta(stream, fmt, &id, created, &model_esc, &piece);
    }
    let trunc = ts.trunc;
    let final_res = orx.recv(); // 최종 결과 수령 (종료 정리)
    // [D 2026-10-10] 엔진 확정 실패의 스트림 직렬화 — 종전엔 실패해도 정상
    // finish("stop")+[DONE]으로 뭉개 클라이언트가 잘린 응답을 성공으로 오인했다.
    let err: Option<String> = match &final_res {
        Ok(r) => r.error.clone(),
        Err(_) => Some("engine result channel closed".into()),
    };
    if let Some(e) = err {
        let _ = sse(
            stream,
            "message",
            &format!(
                "{{\"error\":{{\"message\":\"{}\",\"type\":\"engine_error\"}}}}",
                crate::json::esc(&e)
            ),
        );
        let _ = sse(stream, "done", "[DONE]");
        let _ = stream.shutdown(std::net::Shutdown::Write);
        return;
    }
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
    // [H 2026-10-09] 빈 프롬프트 — OAI·Anthropic 동일 400(이중 표면 제거).
    if ids.is_empty() {
        resp(
            stream,
            400,
            "application/json",
            "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"empty prompt\"}}",
        );
        return;
    }
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
    let prompt_len = ids.len();
    let Ok((orx, prx)) = enqueue_job(stream, &tx, ids, n_predict, vec![STOP_EOT], sampler) else {
        return;
    };
    if stream_mode {
        resp_sse_open(stream);
        // [H 2026-10-09] message_start 스텁 보강 — SDK가 요구하는
        // id/type/model/content/usage 골격(종전 role 하나뿐).
        let msg_id = anthropic_msg_id();
        let _ = sse(
            stream,
            "message_start",
            &format!(
                "{{\"type\":\"message_start\",\"message\":{{\"id\":\"{msg_id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"llm170\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{prompt_len},\"output_tokens\":0}}}}}}"
            ),
        );
        // [R3] 디톡·홀드백·stop 스캔은 TextStream(OAI 러너와 공유).
        let mut ts = TextStream::new(stop_strs.clone(), false);
        // [H] stop_reason 판정용 생성 수 — 정지 토큰(미방출)은 제외.
        let mut n_out = 0usize;
        // [I 2026-10-10] keep-alive — OAI 경로와 동일(5s 주석 프레임).
        loop {
            let t = match prx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(t) => t,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if crate::http::sse_comment(stream, "keep-alive").is_err() {
                        return;
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            if t == llm170_core::qwen35::EOS_EOT || t == STOP_EOT {
                break; // 정지 토큰 미방출(스트림 델타)
            }
            n_out += 1;
            if let Some(piece) = ts.push(t) {
                let esc = crate::json::esc(piece);
                let _ = sse(
                    stream,
                    "content_block_delta",
                    &format!(
                        "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                    ),
                );
            }
            if ts.stopped {
                break;
            }
        }
        // 잔여 보류분 플러시(stop 없이 종료 시).
        if let Some(piece) = ts.flush() {
            let esc = crate::json::esc(piece);
            let _ = sse(
                stream,
                "content_block_delta",
                &format!(
                    "{{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"{esc}\"}}}}"
                ),
            );
        }
        let stopped = ts.stopped;
        let final_res = orx.recv();
        // [D 2026-10-10] 확정 실패 — Anthropic error 이벤트 직렬화
        // (정상 message_delta/message_stop 위장 금지).
        let err: Option<String> = match &final_res {
            Ok(r) => r.error.clone(),
            Err(_) => Some("engine result channel closed".into()),
        };
        if let Some(e) = err {
            let _ = sse(
                stream,
                "error",
                &format!(
                    "{{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":\"{}\"}}}}",
                    crate::json::esc(&e)
                ),
            );
            let _ = stream.shutdown(std::net::Shutdown::Write);
            return;
        }
        // [H 2026-10-09] max_tokens 방출 — 종전엔 length 정지도 end_turn으로
        // 뭉갰다(Anthropic 규약 위반).
        let reason = if stopped {
            "stop_sequence"
        } else if n_out >= n_predict {
            "max_tokens"
        } else {
            "end_turn"
        };
        let _ = sse(
            stream,
            "message_delta",
            &format!(
                "{{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{reason}\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":{n_out}}}}}"
            ),
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
    let stopped = earliest_stop(&text, &stop_strs, 0);
    if let Some((sp, _)) = stopped {
        text.truncate(sp);
    }
    let esc = crate::json::esc(&text);
    // [H 2026-10-09] stop_reason 우선순위(stop_sequence > max_tokens > end_turn)
    // + usage 골격 — 종전엔 length 정지도 end_turn으로 뭉갰다.
    let reason = if stopped.is_some() {
        "stop_sequence"
    } else if all.len() >= n_predict {
        "max_tokens"
    } else {
        "end_turn"
    };
    resp(
        stream,
        200,
        "application/json",
        &format!(
            "{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"{esc}\"}}],\"stop_reason\":\"{reason}\",\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{prompt_len},\"output_tokens\":{}}}}}",
            anthropic_msg_id(),
            all.len()
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

    /// S1: 중첩 객체/문자열 안의 같은 키가 최상위 값을 오염시키지 않는다.
    #[test]
    fn jmember_ignores_nested_keys() {
        let b = r#"{"messages":[{"role":"user","content":"say \"temperature\": 9"}],"temperature":0.5}"#;
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
}
