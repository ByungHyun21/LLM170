
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
            let back = nfa.cls(&[esc], char::from(32), char::from(126));
            nfa.edges[back].push(Edge {
                lo: char::from(32),
                hi: char::from(126),
                to: body,
            });
            close
        }
        Schema::Int | Schema::Num => {
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

pub fn schema_from_request(body: &str) -> Option<Schema> {
    let rf = body.find("\"response_format\"")?;
    let seg = &body[rf..];
    if !seg.contains("\"json_schema\"") {
        return None;
    }
    let ks = seg.find("\"schema\"")?;
    let ob = seg[ks..].find('{')? + ks;
    let block = crate::oai::json::jblock_at(seg, ob)?;
    parse_schema_node(block, 0)
}

const MAX_SCHEMA_DEPTH: usize = 64;

fn parse_schema_node(b: &str, depth: usize) -> Option<Schema> {
    if depth > MAX_SCHEMA_DEPTH {
        return None;
    }
    let ty = crate::oai::json::jstr(b, "type")?;
    match ty.as_str() {
        "string" => Some(Schema::Str),
        "integer" => Some(Schema::Int),
        "number" => Some(Schema::Num),
        "boolean" => Some(Schema::Bool),
        "object" => {
            let props = crate::oai::json::jobject_props(b)?;
            let mut out = Vec::new();
            for (k, v) in props {
                if v.contains("\"enum\"") {
                    let arr = crate::oai::json::jstrs_span(&v, "enum");
                    let e = match arr.as_deref().and_then(crate::oai::json::jarr_pure_strings) {
                        Some(e) if !e.is_empty() => e,
                        _ => return None,
                    };
                    out.push((k, Schema::Enum(e)));
                } else {
                    let sub = parse_schema_node(&v, depth + 1)?;
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

pub struct Constraint {
    nfa: Nfa,
    states: Vec<usize>,
    cache: HashMap<Vec<usize>, Vec<u32>>,
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

    pub fn commit(&mut self, text: &str) {
        for ch in text.chars() {
            self.states = self.step_char(ch);
        }
        self.steps += 1;
    }

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
        let body = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array"}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(body).is_none());
        let ok = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"name":{"type":"string"}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(ok).is_some());
    }

    #[test]
    fn d4_double_backslash_block_scan() {
        let s = r#"{"a":"x\\d","b":1}"#;
        let blk = crate::oai::json::jblock_at(s, 0).unwrap();
        assert_eq!(blk.len(), s.len(), "블록 전체 절단(구 결함은 조기 절단)");
    }

    #[test]
    fn d3_tool_membership() {
        let acc = r#"{"name":"evil","arguments":{}}"#;
        let names = vec!["good".to_string()];
        assert!(crate::oai::wrap_tool_calls(&names, acc).is_none());
        let names2 = vec!["good".to_string(), "evil".to_string()];
        assert!(crate::oai::wrap_tool_calls(&names2, acc).is_some());
    }

    #[test]
    fn d6_integer_and_mixed_enum_fallback() {
        let body = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"lv":{"type":"integer","enum":[1,2]}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(body).is_none());
        let mixed = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"string","enum":["a",1]}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(mixed).is_none());
        let pure = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"string","enum":["a","b"]}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(pure).is_some());
    }

    #[test]
    fn d6_escaped_backslash_enum_scans() {
        let body = r#"{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"string","enum":["a\\","b"]}}}}}}"#;
        assert!(crate::oai::schema::schema_from_request(body).is_some());
    }

    #[test]
    fn d7_nested_array_tools_parse() {
        let body = r#"{"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string","enum":["KST","UTC"]}}}}}]}"#;
        let names = crate::oai::parse_tool_names(body).unwrap();
        assert_eq!(names, vec!["get_weather", "get_time"]);
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

    #[test]
    fn depth_cap_nested_schema() {
        let nest = |n: usize| {
            let mut s = String::from("{\"type\":\"integer\"}");
            for _ in 0..n {
                s = format!("{{\"type\":\"object\",\"properties\":{{\"k\":{s}}}}}");
            }
            s
        };
        assert!(parse_schema_node(&nest(8), 0).is_some());
        assert!(parse_schema_node(&nest(200), 0).is_none());
    }
}
