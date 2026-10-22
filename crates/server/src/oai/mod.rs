use crate::engine::SlotJob;
use crate::http::{
    ENGINE_DEAD, READY, SERVER_CTX, SERVER_MODEL_MAX, TokOut, read_request, resp, resp_sse_open,
    sse,
};
use std::net::TcpStream;
use std::sync::atomic::Ordering;

mod json;
pub(crate) mod schema;
mod stream;
#[cfg(test)]
mod tests;

pub(crate) use self::json::chat_spec;
use self::json::*;
use self::stream::*;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct OutputCap {
    pub tg: usize,
    pub verdict: OutputVerdict,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OutputVerdict {
    Ok,
    PromptTooLong,
    PoolExceeded,
}

pub(crate) fn cap_output(
    prompt: usize,
    n_predict: usize,
    model_max: usize,
    pool_cap: usize,
    slack: usize,
) -> OutputCap {
    if prompt + slack >= model_max {
        return OutputCap {
            tg: n_predict,
            verdict: OutputVerdict::PromptTooLong,
        };
    }
    let tg = n_predict.min(model_max - prompt - slack).max(1);
    let verdict = if prompt + tg + slack > pool_cap {
        OutputVerdict::PoolExceeded
    } else {
        OutputVerdict::Ok
    };
    OutputCap { tg, verdict }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn oai_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("cmpl-{}-{n}", std::process::id())
}

fn anthropic_msg_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("msg_llm170-{n}")
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

struct EmitFmt {
    tool_names: Option<Vec<String>>,
    chat: bool,
    model: String,
    stop_strs: Vec<String>,
    include_stop: bool,
}

const STOP_EOT: u32 = 248046;

fn stops_for_arch() -> Vec<u32> {
    use llm170_core::qwen35::bind as q;
    let arch = crate::engine::arch();
    let mut v: Vec<u32> = json::chat_spec(&arch)
        .map(|s| s.eos.to_vec())
        .unwrap_or_default();
    if q::arch_supported(&arch) && !v.contains(&STOP_EOT) {
        v.push(STOP_EOT);
    }
    if v.is_empty() {
        v.push(STOP_EOT);
    }
    v
}

pub(crate) fn handle(
    stream: TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
) -> Result<(), String> {
    let mut reader = std::io::BufReader::new(stream);
    loop {
        let req = match read_request(&mut reader) {
            Ok(r) => r,
            Err(_) => return Ok(()),
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
            ("GET", "/v1/models") => {
                let eff = crate::metrics::effective_ctx();
                let mmax = crate::metrics::model_max_ctx();
                resp(
                    stream,
                    200,
                    "application/json",
                    &format!(
                        "{{\"object\":\"list\",\"data\":[{{\"id\":\"llm170\",\"object\":\"model\",\"owned_by\":\"local\",\"max_model_len\":{eff},\"model_max_len\":{mmax}}}]}}"
                    ),
                )
            }
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
                    &EmitFmt {
                        tool_names: None,
                        chat: false,
                        model: jstr(&req.body, "model").unwrap_or_else(|| "local_llm".into()),
                        stop_strs: jstop(&req.body),
                        include_stop: jbool(&req.body, "include_stop_str_in_output"),
                    },
                    Vec::new(),
                    parse_sampler(&req.body),
                    schema_constraint(&req.body),
                );
            }
            ("POST", "/v1/chat/completions") => {
                let n_predict = jnum(&req.body, "max_tokens")
                    .unwrap_or(jnum(&req.body, "n_predict").unwrap_or(24.0))
                    .max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                if !has_message_content(&req.body) {
                    resp(
                        stream,
                        400,
                        "application/json",
                        "{\"error\":\"messages required\"}",
                    );
                    continue;
                }
                let text = json::messages_render(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit(
                    stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    &EmitFmt {
                        tool_names: parse_tool_names(&req.body),
                        chat: true,
                        model: jstr(&req.body, "model").unwrap_or_else(|| "local_llm".into()),
                        stop_strs: jstop(&req.body),
                        include_stop: jbool(&req.body, "include_stop_str_in_output"),
                    },
                    stops_for_arch(),
                    parse_sampler(&req.body),
                    schema_constraint(&req.body),
                );
            }
            ("POST", "/v1/messages") => {
                let n_predict = jnum(&req.body, "max_tokens").unwrap_or(24.0).max(1.0) as usize;
                let stream_mode = jbool(&req.body, "stream");
                if !has_message_content(&req.body) {
                    resp(
                        stream,
                        400,
                        "application/json",
                        "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"messages required\"}}",
                    );
                    continue;
                }
                let text = json::messages_render(&req.body);
                let ids = crate::engine::greedy_encode(&text);
                run_and_emit_anthropic(
                    stream,
                    tx.clone(),
                    ids,
                    n_predict,
                    stream_mode,
                    parse_sampler(&req.body),
                    jstop(&req.body),
                );
            }
            _ => resp(stream, 404, "application/json", "{\"error\":\"not found\"}"),
        }
    }
}

#[allow(clippy::type_complexity)]
fn enqueue_job(
    stream: &mut TcpStream,
    tx: &std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stops: Vec<u32>,
    sampler: Option<llm170_core::sampler::SamplerParams>,
    constraint: Option<crate::oai::schema::Schema>,
) -> Result<
    (
        std::sync::mpsc::Receiver<TokOut>,
        std::sync::mpsc::Receiver<crate::sched::ProgressFrame>,
    ),
    (),
> {
    if ENGINE_DEAD.load(Ordering::Acquire) {
        resp(
            stream,
            503,
            "application/json",
            "{\"error\":\"engine dead (slot loop panicked)\"}",
        );
        return Err(());
    }
    let (otx, orx) = std::sync::mpsc::channel::<TokOut>();
    let (ptx, prx) = std::sync::mpsc::channel::<crate::sched::ProgressFrame>();
    let job = SlotJob {
        tokens: ids,
        n_predict,
        constraint,
        sampler,
        stops,
        progress: Some(ptx),
        out: otx,
        queued: std::time::Instant::now(),
        aging_blocked: false,
    };
    match tx.try_send(job) {
        Ok(()) => {
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

pub(crate) fn parse_tool_names(body: &str) -> Option<Vec<String>> {
    let tp = body.find("\"tools\"")?;
    let ob = body[tp..].find('[')? + tp;
    let close = crate::oai::json::jarr_span(body, ob)?;
    let names: Vec<String> = crate::oai::json::jblocks(&body[ob..=close])
        .iter()
        .filter_map(|blk| {
            let fval = crate::oai::json::jmember(blk, "function")?;
            let fobj = fval.trim_start().strip_prefix('{')?;
            crate::oai::json::jstr(&format!("{{{fobj}"), "name")
        })
        .collect();
    (!names.is_empty()).then_some(names)
}

pub(crate) fn wrap_tool_calls(names: &[String], acc: &str) -> Option<String> {
    let t = acc.trim();
    let objs: Vec<&str> = if t.starts_with('[') {
        crate::oai::json::jblocks(t)
    } else if t.starts_with('{') {
        vec![t]
    } else {
        return None;
    };
    let mut calls = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        let name = crate::oai::json::jstr(o, "name")?;
        if !names.iter().any(|n| n == &name) {
            return None;
        }
        let args = raw_value(o, "arguments").unwrap_or_else(|| "{}".into());
        calls.push(format!(
            "{{\"id\":\"call_{i}\",\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}",
            crate::json::esc(&name),
            crate::json::esc(&args)
        ));
    }
    Some(calls.join(","))
}

fn raw_value(o: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let kp = o.find(&pat)? + pat.len();
    let b = o.as_bytes();
    let mut i = kp;
    while i < b.len() && (b[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= b.len() {
        return None;
    }
    match b[i] {
        b'{' | b'[' => {
            let open = b[i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            let mut in_s = false;
            let mut esc = false;
            let mut j = i;
            while j < b.len() {
                if esc {
                    esc = false;
                } else if b[j] == b'\\' && in_s {
                    esc = true;
                } else if b[j] == b'"' {
                    in_s = !in_s;
                } else if !in_s {
                    if b[j] == open {
                        depth += 1;
                    } else if b[j] == close {
                        depth -= 1;
                        if depth == 0 {
                            return Some(o[i..=j].to_string());
                        }
                    }
                }
                j += 1;
            }
            None
        }
        b'"' => crate::oai::json::jstr(o, key).map(|s| s.to_string()),
        _ => {
            let mut j = i;
            while j < b.len() && b[j] != b',' && b[j] != b'}' {
                j += 1;
            }
            Some(o[i..j].trim().to_string())
        }
    }
}

fn schema_constraint(body: &str) -> Option<schema::Schema> {
    let sch = schema::schema_from_request(body);
    if sch.is_none() && body.contains("\"json_schema\"") {
        eprintln!("# json_schema: 미지원/파싱 실패 — 무제약 폴백");
    }
    sch
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
    constraint: Option<crate::oai::schema::Schema>,
) {
    if ids.is_empty() {
        resp(
            stream,
            400,
            "application/json",
            "{\"error\":\"empty prompt\"}",
        );
        return;
    }
    let pool_cap = *SERVER_CTX.get().unwrap_or(&4096);
    let model_max = *SERVER_MODEL_MAX.get().unwrap_or(&usize::MAX);
    let slack = crate::sched::admit::RESERVE_SLACK;
    let cap = cap_output(ids.len(), n_predict, model_max, pool_cap, slack);
    if cap.tg < n_predict {
        eprintln!(
            "# max_tokens 클램프: {} → {} (prompt {} + slack {slack} vs 모델 최대 {model_max})",
            n_predict,
            cap.tg,
            ids.len()
        );
    }
    let n_predict = match cap.verdict {
        OutputVerdict::Ok => cap.tg,
        OutputVerdict::PromptTooLong => {
            resp(
                stream,
                400,
                "application/json",
                &format!(
                    "{{\"error\":\"context_length_exceeded: prompt {} + slack {slack} >= model max {model_max}\"}}",
                    ids.len()
                ),
            );
            return;
        }
        OutputVerdict::PoolExceeded => {
            resp(
                stream,
                400,
                "application/json",
                &format!(
                    "{{\"error\":\"request exceeds the KV pool: prompt {} + max_tokens {} + slack {slack} > pool cap {pool_cap} (kv-mem)\"}}",
                    ids.len(),
                    cap.tg
                ),
            );
            return;
        }
    };
    let prompt_len = ids.len();
    let constraint_was_active = constraint.is_some();
    let mut stops = stops;
    if constraint_was_active && stops.is_empty() {
        stops = stops_for_arch();
    }
    let Ok((orx, prx)) = enqueue_job(stream, &tx, ids, n_predict, stops, sampler, constraint)
    else {
        return;
    };
    if !stream_mode {
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
                &format!("{{\"error\":\"{}\"}}", crate::json::esc(&e)),
            );
            return;
        }
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
        let tool_wrap = fmt
            .tool_names
            .as_deref()
            .filter(|_| constraint_was_active)
            .and_then(|names| wrap_tool_calls(names, &acc));
        let body = if let Some(tc) = tool_wrap {
            format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{}\",\"object\":\"chat.completion\",\"\
             choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{tc}]}},\"finish_reason\":\"tool_calls\"}}],\
             \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},\"tokens\":[{}]}}",
                crate::json::esc(&fmt.model),
                prompt_len,
                ntok,
                prompt_len + ntok,
                arr.join(",")
            )
        } else if fmt.chat {
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
    let mut ts = TextStream::new(fmt.stop_strs.clone(), fmt.include_stop);
    let mut ntok = 0usize;
    loop {
        let t = match prx.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(crate::sched::ProgressFrame::Tok(t)) => t,
            Ok(crate::sched::ProgressFrame::Ping) => {
                if !crate::http::peer_alive(stream) {
                    return;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !crate::http::peer_alive(stream)
                    || crate::http::sse_comment(stream, "keep-alive").is_err()
                {
                    return;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if t == llm170_core::qwen35::EOS_EOT || t == STOP_EOT {
            break;
        }
        ntok += 1;
        if let Some(piece) = ts.push(t) {
            let piece = crate::json::esc(piece);
            if emit_delta(stream, fmt, &id, created, &model_esc, &piece).is_err() {
                return;
            }
        }
        if ts.stopped {
            break;
        }
    }
    if let Some(piece) = ts.flush() {
        let piece = crate::json::esc(piece);
        let _ = emit_delta(stream, fmt, &id, created, &model_esc, &piece);
    }
    let trunc = ts.trunc;
    let final_res = orx.recv();
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
        let _ = sse(
            stream,
            "message",
            &format!(
                "{{\"id\":\"{id}\",\"created\":{created},\"model\":\"{model_esc}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}]}}"
            ),
        );
    }
    let _ = sse(stream, "done", "[DONE]");
    let _ = stream.shutdown(std::net::Shutdown::Write);
}

fn run_and_emit_anthropic(
    stream: &mut TcpStream,
    tx: std::sync::mpsc::SyncSender<SlotJob>,
    ids: Vec<u32>,
    n_predict: usize,
    stream_mode: bool,
    sampler: Option<llm170_core::sampler::SamplerParams>,
    stop_strs: Vec<String>,
) {
    if ids.is_empty() {
        resp(
            stream,
            400,
            "application/json",
            "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"empty prompt\"}}",
        );
        return;
    }
    let pool_cap = *SERVER_CTX.get().unwrap_or(&4096);
    let model_max = *SERVER_MODEL_MAX.get().unwrap_or(&usize::MAX);
    let slack = crate::sched::admit::RESERVE_SLACK;
    let cap = cap_output(ids.len(), n_predict, model_max, pool_cap, slack);
    if cap.tg < n_predict {
        eprintln!(
            "# max_tokens 클램프(anthropic): {} → {} (prompt {} + slack {slack} vs 모델 최대 {model_max})",
            n_predict,
            cap.tg,
            ids.len()
        );
    }
    let n_predict = match cap.verdict {
        OutputVerdict::Ok => cap.tg,
        OutputVerdict::PromptTooLong => {
            resp(
                stream,
                400,
                "application/json",
                &format!(
                    "{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":\"context_length_exceeded: prompt {} + slack {slack} >= model max {model_max}\"}}}}",
                    ids.len()
                ),
            );
            return;
        }
        OutputVerdict::PoolExceeded => {
            resp(
                stream,
                400,
                "application/json",
                &format!(
                    "{{\"type\":\"error\",\"error\":{{\"type\":\"invalid_request_error\",\"message\":\"request exceeds the KV pool: prompt {} + max_tokens {} + slack {slack} > pool cap {pool_cap} (kv-mem)\"}}}}",
                    ids.len(),
                    cap.tg
                ),
            );
            return;
        }
    };
    let prompt_len = ids.len();
    let Ok((orx, prx)) = enqueue_job(
        stream,
        &tx,
        ids,
        n_predict,
        stops_for_arch(),
        sampler,
        None,
    ) else {
        return;
    };
    if stream_mode {
        resp_sse_open(stream);
        let msg_id = anthropic_msg_id();
        let _ = sse(
            stream,
            "message_start",
            &format!(
                "{{\"type\":\"message_start\",\"message\":{{\"id\":\"{msg_id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"llm170\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{prompt_len},\"output_tokens\":0}}}}}}"
            ),
        );
        let mut ts = TextStream::new(stop_strs.clone(), false);
        let mut n_out = 0usize;
        loop {
            let t = match prx.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(crate::sched::ProgressFrame::Tok(t)) => t,
                Ok(crate::sched::ProgressFrame::Ping) => {
                    if !crate::http::peer_alive(stream) {
                        return;
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if !crate::http::peer_alive(stream)
                        || crate::http::sse_comment(stream, "keep-alive").is_err()
                    {
                        return;
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            if t == llm170_core::qwen35::EOS_EOT || t == STOP_EOT {
                break;
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
    let stopped = earliest_stop(&text, &stop_strs, 0);
    if let Some((sp, _)) = stopped {
        text.truncate(sp);
    }
    let esc = crate::json::esc(&text);
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
