//! safetensors 헤더·index.json·tokenizer/config용 최소 JSON 파서.
//! (구 llm170-exl3에서 core로 이관 — 2026-10-08 EXL3 탈락·W4A16 단일 트랙.)
//!
//! 표준 일반 JSON은 아님 — 헤더 스키마에 필요한 부분집합:
//! 객체·배열·문자열·정수/실수·bool·null. 깊이/길이 상한으로
//! 손상 헤더 방어.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(s: &str) -> Result<Json, String> {
        let b = s.as_bytes();
        let mut p = P { b, i: 0, depth: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != b.len() {
            return Err(format!("trailing bytes at {}", p.i));
        }
        Ok(v)
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Vec<(String, Json)>> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// 정수 배열(u64) — shape/data_offsets용.
    pub fn as_num_array(&self) -> Option<Vec<u64>> {
        match self {
            Json::Arr(a) => a
                .iter()
                .map(|v| match v {
                    Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// 객체를 HashMap 뷰로(중복 키는 마지막 우선 — JSON 관례).
    pub fn map(&self) -> HashMap<&str, &Json> {
        self.as_object()
            .map(|o| o.iter().map(|(k, v)| (k.as_str(), v)).collect())
            .unwrap_or_default()
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

const MAX_DEPTH: u32 = 32;
const MAX_LEN: usize = 1 << 26;

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        if self.depth >= MAX_DEPTH {
            return Err("nesting too deep".into());
        }
        match self.b.get(self.i) {
            Some(b'{') => {
                self.depth += 1;
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Obj(out));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(format!("expected ':' at {}", self.i));
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value()?;
                    out.push((k, v));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            self.depth -= 1;
                            return Ok(Json::Obj(out));
                        }
                        _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.depth += 1;
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Arr(out));
                }
                loop {
                    self.ws();
                    out.push(self.value()?);
                    if out.len() > MAX_LEN {
                        return Err("array too long".into());
                    }
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            self.depth -= 1;
                            return Ok(Json::Arr(out));
                        }
                        _ => return Err(format!("expected ',' or ']' at {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(c) if c.is_ascii_digit() || *c == b'-' => self.number(),
            _ => Err(format!("unexpected byte at {}", self.i)),
        }
    }

    fn lit(&mut self, s: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let s = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        while self
            .b
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[s..self.i])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .map(Json::Num)
            .ok_or_else(|| format!("bad number at {s}"))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(format!("expected string at {}", self.i));
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            match self.b.get(self.i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.b.get(self.i) {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'n') => out.push('\n'),
                        Some(b't') => out.push('\t'),
                        Some(b'r') => out.push('\r'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'u') => {
                            if self.i + 4 >= self.b.len() {
                                return Err("bad \\u escape".into());
                            }
                            let h = std::str::from_utf8(&self.b[self.i + 1..self.i + 5])
                                .ok()
                                .and_then(|h| u32::from_str_radix(h, 16).ok())
                                .ok_or("bad \\u escape")?;
                            out.push(char::from_u32(h).ok_or("bad \\u codepoint")?);
                            self.i += 4;
                        }
                        _ => return Err("bad escape".into()),
                    }
                    self.i += 1;
                }
                Some(&c) if c < 0x80 => {
                    out.push(c as char);
                    self.i += 1;
                }
                Some(_) => {
                    // 멀티바이트 UTF-8 — 그대로 통과.
                    let s = self.i;
                    while self.b.get(self.i).is_some_and(|c| *c >= 0x80) {
                        self.i += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.b[s..self.i]).map_err(|_| "bad utf8")?);
                }
            }
            if out.len() > MAX_LEN {
                return Err("string too long".into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_shape() {
        let h = r#"{"__metadata__":{"format":"pt"},"a.trellis":{"dtype":"I16","shape":[320,1088,48],"data_offsets":[0,100]}}"#;
        let v = Json::parse(h).unwrap();
        let t = v.get("a.trellis").unwrap();
        assert_eq!(t.get("dtype").unwrap().as_str(), Some("I16"));
        assert_eq!(
            t.get("shape").unwrap().as_num_array().unwrap(),
            vec![320, 1088, 48]
        );
        assert_eq!(
            t.get("data_offsets").unwrap().as_num_array().unwrap(),
            vec![0, 100]
        );
    }

    #[test]
    fn index_json() {
        let h = r#"{"metadata":{"total_size":123},"weight_map":{"a.suh":"model-00001-of-00002.safetensors"}}"#;
        let v = Json::parse(h).unwrap();
        let m = v.get("weight_map").unwrap().map();
        assert_eq!(
            m["a.suh"].as_str(),
            Some("model-00001-of-00002.safetensors")
        );
    }

    #[test]
    fn escapes_and_errors() {
        assert_eq!(Json::parse(r#""a\"b""#).unwrap(), Json::Str("a\"b".into()));
        assert!(Json::parse("{").is_err());
        assert!(Json::parse("[1,]").is_err());
    }
}
