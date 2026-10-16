//! [P8 2026-10-16] XGrammar-2 서브셋 — json_schema 제약 디코딩.
//! 스키마 하위집합: `type: object`(properties: string|integer|number|boolean|
//! enum)·선언 프로퍼티 전부 필수·키 고정 순서(서브셋 계약 — 승람 시 순서
//! 자유화). 컴파일 = 문자 NFA(상태-집합 스레딩으로 교대 표현) → 디코드
//! 스텝마다 허용 토큰 = NFA 경로 접두사. 토큰 허용집합은 상태집합별 캐시.
//! 계측: 마스크 스텝 수.
//!
//! 전제(운영 제약): 제약 슬롯은 스펙 라운드 비활성(초안이 마스크를 모른다)
//! ·greedy 전용.

use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Edge {
    lo: char,
    hi: char,
    to: usize,
}

#[derive(Clone, Debug, Default)]
struct Nfa {
    edges: Vec<Vec<Edge>>,
    accept: Vec<bool>,
}

impl Nfa {
    fn add_state(&mut self) -> usize {
        self.edges.push(Vec::new());
        self.accept.push(false);
        self.edges.len() - 1
    }
    fn chr(&mut self, froms: &[usize], ch: char) -> Vec<usize> {
        let to = self.add_state();
        for &f in froms {
            self.edges[f].push(Edge { lo: ch, hi: ch, to });
        }
        vec![to]
    }
    fn lit(&mut self, froms: &[usize], s: &str) -> Vec<usize> {
        let mut cur = froms.to_vec();
        for ch in s.chars() {
            cur = self.chr(&cur, ch);
        }
        cur
    }
    fn cls(&mut self, froms: &[usize], lo: char, hi: char) -> usize {
        let to = self.add_state();
        for &f in froms {
            self.edges[f].push(Edge { lo, hi, to });
        }
        to
    }
}

/// json 값 스키마 노드(서브셋).
#[derive(Clone, Debug)]
pub enum Schema {
    Str,
    Int,
    Num,
    Bool,
    Enum(Vec<String>),
    Obj(Vec<(String, Schema)>),
}

fn compile(nfa: &mut Nfa, froms: &[usize], sch: &Schema) -> Vec<usize> {
    match sch {
        Schema::Str => {
            // "…" — 몸통: printable(단 ", \ 제외) + \escape 2글자.
            let body = nfa.chr(froms, '"')[0];
            let esc = nfa.add_state();
            for c in (32u8..=126u8).map(char::from) {
                if c != '"' && c != '\\' {
                    nfa.edges[body].push(Edge {
                        lo: c,
                        hi: c,
                        to: body,
                    });
                }
            }
            let close = nfa.chr(&[body], '"');
            nfa.edges[body].push(Edge {
                lo: '\\',
                hi: '\\',
                to: esc,
            });
            // escape 다음 printable 1글자 → body로 복귀.
            let back = nfa.cls(&[esc], char::from(32), char::from(126));
            nfa.edges[back].push(Edge {
                lo: char::from(32),
                hi: char::from(126),
                to: body,
            });
            close
        }
        Schema::Int | Schema::Num => {
            // -? (0|[1-9][0-9]*)(.frac)?(e[+-]?d+)? — Num만 frac/exp.
            let neg = nfa.add_state();
            for &f in froms {
                nfa.edges[f].push(Edge {
                    lo: '-',
                    hi: '-',
                    to: neg,
                });
            }
            let d0 = nfa.cls(froms, '0', '0');
            let d19 = nfa.cls(froms, '1', '9');
            nfa.edges[d19].push(Edge {
                lo: '0',
                hi: '9',
                to: d19,
            });
            let n0 = nfa.cls(&[neg], '0', '0');
            let n19 = nfa.cls(&[neg], '1', '9');
            nfa.edges[n19].push(Edge {
                lo: '0',
                hi: '9',
                to: n19,
            });
            let mut ends = vec![d0, d19, n0, n19];
            // '-'만 있고 숫자 없음은 불허여야 하나 서브셋: 다음 스텝 마스크가
            // 숫자 외 차단(ends에 neg 포함이면 '-' 직후 허용집합이 숫자만).
            // → neg 제거(빈 값 방지는 호출부).
            if matches!(sch, Schema::Num) {
                let mut fends = Vec::new();
                for &e in &ends {
                    let dot = nfa.cls(&[e], '.', '.');
                    let fd = nfa.cls(&[dot], '0', '9');
                    nfa.edges[fd].push(Edge {
                        lo: '0',
                        hi: '9',
                        to: fd,
                    });
                    fends.push(e);
                    fends.push(fd);
                }
                let mut eends = Vec::new();
                for &e in &fends {
                    let ex = nfa.cls(&[e], 'e', 'e');
                    let sgn = nfa.cls(&[ex], '+', '+');
                    let m = nfa.cls(&[ex], '-', '-');
                    let ed = nfa.cls(&[sgn], '0', '9');
                    let ed2 = nfa.cls(&[m], '0', '9');
                    nfa.edges[ex].push(Edge {
                        lo: '0',
                        hi: '9',
                        to: ed,
                    });
                    nfa.edges[ed].push(Edge {
                        lo: '0',
                        hi: '9',
                        to: ed,
                    });
                    nfa.edges[ed2].push(Edge {
                        lo: '0',
                        hi: '9',
                        to: ed2,
                    });
                    eends.push(e);
                    eends.push(ed);
                    eends.push(ed2);
                }
                ends = eends;
            }
            ends
        }
        Schema::Bool => {
            let t = nfa.lit(froms, "true");
            let f = nfa.lit(froms, "false");
            [t, f].concat()
        }
        Schema::Enum(vs) => {
            // JSON 출력은 문자열 리터럴 — 값 인용 포함 컴파일.
            let mut ends = Vec::new();
            for v in vs {
                ends.extend(nfa.lit(froms, &format!("\"{v}\"")));
            }
            ends
        }
        Schema::Obj(props) => {
            let mut cur = nfa.lit(froms, "{");
            for (i, (k, sub)) in props.iter().enumerate() {
                if i > 0 {
                    cur = nfa.lit(&cur, ",");
                }
                cur = nfa.lit(&cur, &format!("\"{k}\":"));
                cur = compile(nfa, &cur, sub);
            }
            cur = nfa.lit(&cur, "}");
            cur
        }
    }
}

/// [P8] 요청 파싱 — `response_format:{"type":"json_schema","json_schema":
/// {"schema":{...}}}` → Schema 서브셋. 미지원 타입은 None(무제약 폴백 +
/// 경고 로그). 서브셋: object(string|integer|number|boolean|enum).
pub fn schema_from_request(body: &str) -> Option<Schema> {
    let rf = body.find("\"response_format\"")?;
    let seg = &body[rf..];
    if !seg.contains("\"json_schema\"") {
        return None; // json_object/미지정 — 무제약
    }
    let ks = seg.find("\"schema\"")?;
    let ob = seg[ks..].find('{')? + ks;
    let block = crate::oai::json::jblock_at(seg, ob)?;
    parse_schema_node(block)
}

fn parse_schema_node(b: &str) -> Option<Schema> {
    let ty = crate::oai::json::jstr(b, "type")?;
    match ty.as_str() {
        "string" => Some(Schema::Str),
        "integer" => Some(Schema::Int),
        "number" => Some(Schema::Num),
        "boolean" => Some(Schema::Bool),
        "object" => {
            // properties — jblocks로 (키, 노드) 재귀. enum은 string 내부.
            let props = crate::oai::json::jobject_props(b)?;
            let mut out = Vec::new();
            for (k, v) in props {
                if let Some(e) = crate::oai::json::jstrs(&v, "enum")
                    && !e.is_empty()
                {
                    out.push((k, Schema::Enum(e)));
                } else {
                    // [P0-D2] 미지원 프로퍼티 = 전체 폴백(None 전파).
                    let sub = parse_schema_node(&v)?;
                    out.push((k, sub));
                }
            }
            if out.is_empty() {
                return None;
            }
            Some(Schema::Obj(out))
        }
        _ => None,
    }
}

/// 디코드 스텝 제약기 — 상태 집합 시뮬레이션 + 토큰 허용 캐시.
pub struct Constraint {
    nfa: Nfa,
    states: Vec<usize>,
    cache: HashMap<Vec<usize>, Vec<u32>>,
    /// 계측: 마스크 스텝 수.
    pub steps: u64,
}

impl Constraint {
    pub fn new(sch: &Schema) -> Self {
        let mut nfa = Nfa::default();
        let ends = compile(&mut nfa, &[0], sch);
        for e in ends {
            nfa.accept[e] = true;
        }
        Constraint {
            nfa,
            states: vec![0],
            cache: HashMap::new(),
            steps: 0,
        }
    }

    fn step_char(&self, ch: char) -> Vec<usize> {
        let mut out = Vec::new();
        for &st in &self.states {
            for e in &self.nfa.edges[st] {
                if ch >= e.lo && ch <= e.hi && !out.contains(&e.to) {
                    out.push(e.to);
                }
            }
        }
        out
    }

    fn accepts(&self, text: &str) -> bool {
        let mut cur = self.states.clone();
        for ch in text.chars() {
            let mut next = Vec::new();
            for &st in &cur {
                for e in &self.nfa.edges[st] {
                    if ch >= e.lo && ch <= e.hi && !next.contains(&e.to) {
                        next.push(e.to);
                    }
                }
            }
            if next.is_empty() {
                return false;
            }
            cur = next;
        }
        true
    }

    /// 토큰 허용 집합(상태집합 캐시) — decode 클로저로 토큰 텍스트 주입.
    pub fn allowed<F: Fn(u32) -> String>(&mut self, decode: F, vocab: u32) -> &[u32] {
        let key = self.states.clone();
        if !self.cache.contains_key(&key) {
            let set: Vec<u32> = (0..vocab)
                .filter(|&t| {
                    let s = decode(t);
                    !s.is_empty() && self.accepts(&s)
                })
                .collect();
            self.cache.insert(key.clone(), set);
        }
        self.cache.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 토큰 확정 — 상태 전이.
    pub fn commit(&mut self, text: &str) {
        for ch in text.chars() {
            self.states = self.step_char(ch);
        }
        self.steps += 1;
    }

    /// 수용 상태 도달(테스트·계측 전용 — 운용은 allowed 빈집합→종결 토큰).
    #[cfg(test)]
    pub fn done(&self) -> bool {
        self.states.iter().any(|&s| self.nfa.accept[s])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tok(t: u32) -> String {
        match t {
            0 => "{\"a\":".into(),
            1 => "1".into(),
            2 => "}".into(),
            3 => "x".into(),
            4 => "2".into(),
            5 => "\"lo\"".into(),
            _ => String::new(),
        }
    }

    #[test]
    fn object_schema_flow() {
        let sch = Schema::Obj(vec![("a".into(), Schema::Int)]);
        let mut c = Constraint::new(&sch);
        let allowed: Vec<u32> = c.allowed(toy_tok, 6).to_vec();
        assert!(
            allowed.contains(&0),
            "시작은 {{\"a\": 접두사만: {allowed:?}"
        );
        assert!(!allowed.contains(&2), "}} 불허");
        c.commit("{\"a\":");
        let allowed: Vec<u32> = c.allowed(toy_tok, 6).to_vec();
        assert!(
            allowed.contains(&1) && allowed.contains(&4),
            "정수 허용: {allowed:?}"
        );
        assert!(!allowed.contains(&3), "x 불허");
        c.commit("1");
        let allowed: Vec<u32> = c.allowed(toy_tok, 6).to_vec();
        assert!(allowed.contains(&2), "닫기 허용: {allowed:?}");
        c.commit("}");
        assert!(c.done());
    }

    #[test]
    fn string_and_enum() {
        let sch = Schema::Obj(vec![
            ("m".into(), Schema::Enum(vec!["lo".into(), "hi".into()])),
            ("s".into(), Schema::Str),
        ]);
        let mut c = Constraint::new(&sch);
        c.commit("{\"m\":");
        assert!(c.accepts("\"lo\"") && c.accepts("\"hi\""));
        assert!(!c.accepts("\"zz\""));
        c.commit("\"lo\"");
        c.commit(",");
        c.commit("\"s\":");
        assert!(c.accepts("\"abc\"") && !c.accepts("abc"));
        c.commit("\"v\"");
        c.commit("}");
        assert!(c.done());
    }

    #[test]
    fn d2_unsupported_property_falls_back_whole() {
        // array 프로퍼티 1개 = 전체 폴백(조용한 필드 탈락 금지).
        let body = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array"}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(body).is_none());
        // 지원 범위 내 = 컴파일 성공.
        let ok = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"name":{"type":"string"}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(ok).is_some());
    }

    #[test]
    fn d4_double_backslash_block_scan() {
        // jblock_at: 값 내 이중 백슬래시 뒤 '"'가 문자열 종결로 인식되어야.
        let s = r#"{"a":"x\\d","b":1}"#;
        let blk = crate::oai::json::jblock_at(s, 0).unwrap();
        assert_eq!(blk.len(), s.len(), "블록 전체 절단(구 결함은 조기 절단)");
    }

    #[test]
    fn d3_tool_membership() {
        // 선언되지 않은 name = 래핑 없음.
        let acc = r#"{"name":"evil","arguments":{}}"#;
        let names = vec!["good".to_string()];
        assert!(crate::oai::wrap_tool_calls(&names, acc).is_none());
        let names2 = vec!["good".to_string(), "evil".to_string()];
        assert!(crate::oai::wrap_tool_calls(&names2, acc).is_some());
    }

    #[test]
    fn bool_num_branches() {
        let sch = Schema::Obj(vec![("b".into(), Schema::Bool), ("n".into(), Schema::Num)]);
        let mut c = Constraint::new(&sch);
        c.commit("{\"b\":");
        assert!(c.accepts("true") && c.accepts("false") && !c.accepts("truex"));
        c.commit("true");
        c.commit(",");
        c.commit("\"n\":");
        assert!(c.accepts("-1.5e3") && c.accepts("42") && !c.accepts("4x"));
        c.commit("42");
        c.commit("}");
        assert!(c.done());
    }
}
