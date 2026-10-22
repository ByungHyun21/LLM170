use super::*;

pub(crate) fn earliest_stop(acc: &str, stops: &[String], from: usize) -> Option<(usize, usize)> {
    let mut base = from.min(acc.len());
    while base > 0 && !acc.is_char_boundary(base) {
        base -= 1;
    }
    let mut best: Option<(usize, usize)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(p) = acc[base..].find(s.as_str()) {
            let p = base + p;
            if best.is_none_or(|(bp, _)| p < bp) {
                best = Some((p, s.len()));
            }
        }
    }
    best
}

pub(crate) fn emit_delta(
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

pub(crate) struct TextStream {
    det: crate::engine::Detok,
    acc: String,
    sent: usize,
    scan_from: usize,
    holdback: usize,
    stops: Vec<String>,
    include_stop: bool,
    pub(crate) stopped: bool,
    pub(crate) trunc: bool,
}

impl TextStream {
    pub(crate) fn new(stops: Vec<String>, include_stop: bool) -> Self {
        let holdback = stops
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        Self {
            det: crate::engine::Detok::new(),
            acc: String::new(),
            sent: 0,
            scan_from: 0,
            holdback,
            stops,
            include_stop,
            stopped: false,
            trunc: false,
        }
    }

    pub(crate) fn push(&mut self, t: u32) -> Option<&str> {
        let before = self.acc.len();
        let piece = self.det.push(t);
        self.acc.push_str(&piece);
        if let Some((p, l)) = earliest_stop(&self.acc, &self.stops, self.scan_from) {
            let end = if self.include_stop { p + l } else { p };
            self.trunc = true;
            self.stopped = true;
            if end > self.sent {
                let s = self.sent;
                self.sent = end;
                return self.acc.get(s..end);
            }
            return None;
        }
        self.scan_from = before.saturating_sub(self.holdback);
        let safe = floor_char_boundary(&self.acc, self.acc.len().saturating_sub(self.holdback));
        if safe > self.sent {
            let s = self.sent;
            self.sent = safe;
            return self.acc.get(s..safe);
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn push_text(&mut self, text: &str) -> Option<String> {
        let before = self.acc.len();
        self.acc.push_str(text);
        if let Some((p, l)) = earliest_stop(&self.acc, &self.stops, self.scan_from) {
            let end = if self.include_stop { p + l } else { p };
            self.trunc = true;
            self.stopped = true;
            if end > self.sent {
                let out = self.acc[self.sent..end].to_string();
                self.sent = end;
                return Some(out);
            }
            return None;
        }
        self.scan_from = before.saturating_sub(self.holdback);
        let safe = floor_char_boundary(&self.acc, self.acc.len().saturating_sub(self.holdback));
        if safe > self.sent {
            let out = self.acc[self.sent..safe].to_string();
            self.sent = safe;
            return Some(out);
        }
        None
    }

    pub(crate) fn flush(&mut self) -> Option<&str> {
        if !self.trunc && self.acc.len() > self.sent {
            let s = self.sent;
            self.sent = self.acc.len();
            return self.acc.get(s..);
        }
        None
    }
}
