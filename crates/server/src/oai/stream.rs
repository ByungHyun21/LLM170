//! oai 스트림 공통 — stop 스캔·델타 프레임(R3 분할).
use super::*;

/// 스트림 누적 텍스트에서 stop 문자열 최초 등장 — (바이트 위치, 길이).
/// [2026-10-09 D4] from = 이번 라운드 이전에 이미 스캔한 접두(의미상 확정된
/// 부분). 이전 라운드에서 매치가 없었다면 새 매치는 `old_len - (최장 stop-1)`
/// 이후에서만 시작할 수 있다(그보다 앞이면 이전 스캔이 이미 찾았어야 함) —
/// 호출자가 그 값을 넘긴다. 종전엔 토큰마다 누적 전체를 재스캔했다(O(N²)).
pub(crate) fn earliest_stop(acc: &str, stops: &[String], from: usize) -> Option<(usize, usize)> {
    // [R3-fix] from은 "이전 스캔 접두 - (최장 stop-1)"이라 멀티바이트 문자
    // 중간에 떨어질 수 있다 — acc[base..] 슬라이스 패닉(한글 stop 실측).
    // 이전 경계로 후퇴(몇 바이트 재스캔 — D4 절약 의도 유지).
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

/// 스트림 델타 청크 송신 — chat은 P0-4 표준 청크, completions는 종전 text 프레임.
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

/// [R3] 토큰 텍스트 스트림 공통 상태 — 디톡·홀드백·stop 스캔(OAI/Anthropic
/// 공유). 방출 조각만 반환하고 프레임 포맷은 러너가 담당한다. 동작은 종전
/// 두 러너의 인라인 루프와 동일(홀드백 = 최장 stop-1, 스캔 창 전진).
pub(crate) struct TextStream {
    det: crate::engine::Detok,
    acc: String,
    sent: usize,
    scan_from: usize,
    holdback: usize,
    stops: Vec<String>,
    include_stop: bool,
    /// stop 문자열 도달(방출 완료).
    pub(crate) stopped: bool,
    /// stop 절단 발생(stopped와 동치 — finish_reason 판정용 별칭).
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

    /// 토큰 1개 투입 → 방출할 조각(없으면 None). stop 도달 시 stopped=true.
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

    /// [테스트 전용] 디톡 없이 텍스트 직접 주입 — 스캔·홀드백 로직 표면.
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

    /// 종료 시 잔여 방출(stop 미도달분).
    pub(crate) fn flush(&mut self) -> Option<&str> {
        if !self.trunc && self.acc.len() > self.sent {
            let s = self.sent;
            self.sent = self.acc.len();
            return self.acc.get(s..);
        }
        None
    }
}
