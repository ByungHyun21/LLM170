
use crate::unicode_data as udata;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

const F_NUMBER: u16 = 0x0002;
const F_LETTER: u16 = 0x0004;
const F_ACCENT: u16 = 0x0010;
const F_WHITESPACE: u16 = 0x0100;

#[derive(Default)]
struct MulHasher(u64);

impl std::hash::Hasher for MulHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0 ^ 0x9E37_79B9_7F4A_7C15;
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks {
            let v = u64::from_le_bytes(*c);
            h = (h ^ v).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h = h.rotate_left(27);
        }
        let mut tail = 0u64;
        for (i, &b) in rest.iter().enumerate() {
            tail |= (b as u64) << (8 * i);
        }
        h = (h ^ tail).wrapping_mul(0x94D0_49BB_1331_11EB);
        self.0 = h ^ (h >> 31);
    }
}

type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<MulHasher>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pre {
    Qwen35,
    Qwen2,
    Gemma,
    Other,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    text_to_id: FastMap<Box<[u8]>, u32>,
    bpe_ranks: FastMap<Vec<u8>, u32>,
    special: Vec<(String, u32)>,
    special_user: Vec<(String, u32)>,
    pre: Pre,
    ignore_merges: bool,
    c2b: FastMap<char, u8>,
    b2c: FastMap<u8, char>,
    greedy_index: FastMap<Vec<u8>, u32>,
    encode_cache: std::sync::Mutex<EncodeCache>,
}

struct EncodeCache {
    map: HashMap<String, std::sync::Arc<[u32]>>,
    order: std::collections::VecDeque<String>,
    cap: usize,
}

impl EncodeCache {
    fn get(&self, text: &str) -> Option<Vec<u32>> {
        self.map.get(text).map(|a| a.to_vec())
    }
    fn put(&mut self, text: &str, ids: &[u32]) {
        if text.len() < 32 || self.map.contains_key(text) {
            return;
        }
        if self.order.len() >= self.cap
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
        self.order.push_back(text.to_string());
        self.map.insert(text.to_string(), ids.into());
    }
}

impl Tokenizer {
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            return Self::from_hf_dir(path);
        }
        Err(format!(
            "토크나이저: 디렉터리 모델(W4A16)만 지원(2026-10-08): {}",
            path.display()
        ))
    }

    fn from_parts(
        vocab: &[String],
        bpe_ranks: FastMap<Vec<u8>, u32>,
        special_ctl: Vec<(String, u32)>,
        mut special_user: Vec<(String, u32)>,
        pre: Pre,
        ignore_merges: bool,
    ) -> Result<Self, String> {
        let mut special = special_ctl;
        special.extend(special_user.iter().cloned());
        special.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        special_user.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        let mut c2b: FastMap<char, u8> = FastMap::default();
        let mut b2c: FastMap<u8, char> = FastMap::default();
        {
            let mut n = 0u32;
            for b in 0u32..256 {
                let printable = (0x21..=0x7E).contains(&b)
                    || (0xA1..=0xAC).contains(&b)
                    || (0xAE..=0xFF).contains(&b);
                let c = if printable { b } else { 256 + n };
                if !printable {
                    n += 1;
                }
                let c = char::from_u32(c).unwrap();
                c2b.insert(c, b as u8);
                b2c.insert(b as u8, c);
            }
        }

        let mut text_to_id: FastMap<Box<[u8]>, u32> = FastMap::default();
        let mut greedy_index: FastMap<Vec<u8>, u32> = FastMap::default();
        for (i, t) in vocab.iter().enumerate() {
            let bytes: Vec<u8> = t
                .chars()
                .flat_map(|c| match c2b.get(&c) {
                    Some(&b) => vec![b],
                    None => c.to_string().into_bytes(),
                })
                .collect();
            if !bytes.is_empty() {
                greedy_index.entry(bytes).or_insert(i as u32);
            }
            text_to_id.insert(t.as_bytes().into(), i as u32);
        }
        Ok(Tokenizer {
            vocab: vocab.to_vec(),
            text_to_id,
            bpe_ranks,
            special,
            special_user,
            pre,
            ignore_merges,
            c2b,
            b2c,
            greedy_index,
            encode_cache: std::sync::Mutex::new(EncodeCache {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
                cap: 64,
            }),
        })
    }

    fn from_hf_dir(dir: &Path) -> Result<Self, String> {
        let tok_json: Option<llm170_core::json::Json> = if dir.join("vocab.json").is_file() {
            None
        } else {
            let txt = std::fs::read_to_string(dir.join("tokenizer.json"))
                .map_err(|e| format!("tokenizer.json: {e}"))?;
            Some(
                llm170_core::json::Json::parse(&txt)
                    .map_err(|e| format!("tokenizer.json 파싱: {e}"))?,
            )
        };
        let mut max_id = 0usize;
        let mut pairs: Vec<(String, u32)> = Vec::new();
        if let Some(tj) = &tok_json {
            let vocab = tj
                .get("model")
                .and_then(|m| m.get("vocab"))
                .and_then(|v| v.as_object())
                .ok_or("tokenizer.json: model.vocab 부재")?;
            for (piece, id) in vocab {
                let f = id
                    .as_f64()
                    .ok_or_else(|| format!("tokenizer.json: '{piece}' id가 숫자 아님"))?;
                let id = f as i64;
                if id < 0 || f != id as f64 {
                    return Err(format!("tokenizer.json: '{piece}' id가 정수 아님({f})"));
                }
                max_id = max_id.max(id as usize);
                pairs.push((piece.clone(), id as u32));
            }
            if let Some(llm170_core::json::Json::Arr(items)) = tj.get("added_tokens") {
                for it in items {
                    let id = it.get("id").and_then(llm170_core::json::Json::as_f64);
                    let content = it.get("content").and_then(llm170_core::json::Json::as_str);
                    if let (Some(id), Some(c)) = (id, content) {
                        let id = id as i64;
                        if id >= 0 {
                            max_id = max_id.max(id as usize);
                            pairs.push((c.to_string(), id as u32));
                        }
                    }
                }
            }
        } else {
            let vj = llm170_core::json::Json::parse(
                &std::fs::read_to_string(dir.join("vocab.json"))
                    .map_err(|e| format!("vocab.json: {e}"))?,
            )
            .map_err(|e| format!("vocab.json 파싱: {e}"))?;
            let obj = vj.as_object().ok_or("vocab.json: 객체 아님")?;
            pairs.reserve(obj.len());
            for (k, v) in obj {
                let f = v
                    .as_f64()
                    .ok_or_else(|| format!("vocab.json: '{k}' id가 숫자 아님"))?;
                let id = f as i64;
                if id < 0 || f != id as f64 {
                    return Err(format!("vocab.json: '{k}' id가 정수 아님({f})"));
                }
                max_id = max_id.max(id as usize);
                pairs.push((k.clone(), id as u32));
            }
        }
        let mut vocab = vec![String::new(); max_id + 1];
        for (piece, id) in &pairs {
            vocab[*id as usize] = piece.clone();
        }

        let mut bpe_ranks: FastMap<Vec<u8>, u32> = FastMap::default();
        let mut rank = 0u32;
        let push_merge =
            |first: &[u8], second: &[u8], bpe_ranks: &mut FastMap<Vec<u8>, u32>, rank: &mut u32| {
                let mut key = Vec::with_capacity(4 + first.len() + second.len());
                key.extend_from_slice(&(first.len() as u32).to_le_bytes());
                key.extend_from_slice(first);
                key.extend_from_slice(second);
                bpe_ranks.entry(key).or_insert(*rank);
                *rank += 1;
            };
        if let Some(tj) = &tok_json {
            if let Some(llm170_core::json::Json::Arr(ms)) =
                tj.get("model").and_then(|m| m.get("merges"))
            {
                for m in ms {
                    match m {
                        llm170_core::json::Json::Arr(pair) if pair.len() == 2 => {
                            if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                                push_merge(a.as_bytes(), b.as_bytes(), &mut bpe_ranks, &mut rank);
                            }
                        }
                        _ => {
                            if let Some(s) = m.as_str()
                                && let Some((a, b)) = s.split_once(' ')
                            {
                                push_merge(a.as_bytes(), b.as_bytes(), &mut bpe_ranks, &mut rank);
                            }
                        }
                    }
                }
            }
        } else {
            let merges_txt = std::fs::read_to_string(dir.join("merges.txt"))
                .map_err(|e| format!("merges.txt: {e}"))?;
            for line in merges_txt.lines() {
                if line.is_empty() || line.starts_with("#version") {
                    continue;
                }
                let b = line.as_bytes();
                let Some(p) = b[1..].iter().position(|&c| c == b' ') else {
                    continue;
                };
                let (first, second) = (&b[..p + 1], &b[p + 2..]);
                push_merge(first, second, &mut bpe_ranks, &mut rank);
            }
        }

        let mut special: Vec<(String, u32)> = Vec::new();
        let mut ignore_merges = false;
        if let Ok(tc) = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            && let Ok(tj) = llm170_core::json::Json::parse(&tc)
            && let Some(tobj) = tj.as_object()
        {
            for (k, v) in tobj {
                if k == "ignore_merges"
                    && let Some(bv) = v.as_bool()
                {
                    ignore_merges = bv;
                }
                if k != "added_tokens_decoder" {
                    continue;
                }
                if let Some(entries) = v.as_object() {
                    for (id_str, ent) in entries {
                        let Ok(id) = id_str.parse::<u32>() else {
                            continue;
                        };
                        let Some(fields) = ent.as_object() else {
                            continue;
                        };
                        let content = fields
                            .iter()
                            .find(|(fk, _)| fk == "content")
                            .and_then(|(_, fv)| fv.as_str().map(str::to_string));
                        let is_special = fields
                            .iter()
                            .find(|(fk, _)| fk == "special")
                            .and_then(|(_, fv)| fv.as_bool())
                            .unwrap_or(true);
                        if let Some(c) = content
                            && is_special
                        {
                            special.push((c, id));
                        }
                    }
                }
            }
        }
        let mut special_user: Vec<(String, u32)> = Vec::new();
        {
            let mut seen: std::collections::HashSet<u32> =
                special.iter().map(|(_, id)| *id).collect();
            if let Some(tj) = &tok_json
                && let Some(llm170_core::json::Json::Arr(items)) = tj.get("added_tokens")
            {
                for it in items {
                    let Some(id) = it
                        .get("id")
                        .and_then(llm170_core::json::Json::as_f64)
                        .map(|x| x as u32)
                    else {
                        continue;
                    };
                    let Some(content) = it.get("content").and_then(llm170_core::json::Json::as_str)
                    else {
                        continue;
                    };
                    if !seen.insert(id) {
                        continue;
                    }
                    let is_special = it
                        .get("special")
                        .and_then(llm170_core::json::Json::as_bool)
                        .unwrap_or(false);
                    if is_special {
                        special.push((content.to_string(), id));
                    } else {
                        special_user.push((content.to_string(), id));
                    }
                }
            }
        }
        special.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        special_user.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let mut pre = Pre::Other;
        if let Ok(cfg) = std::fs::read_to_string(dir.join("config.json"))
            && let Ok(cj) = llm170_core::json::Json::parse(&cfg)
            && let Some(cobj) = cj.as_object()
            && let Some((_, mt)) = cobj.iter().find(|(k, _)| k == "model_type")
            && let Some(mt) = mt.as_str()
        {
            pre = if mt.starts_with("qwen3_5") || mt.starts_with("qwen4") {
                Pre::Qwen35
            } else if mt.starts_with("qwen2") || mt.starts_with("qwen3") {
                Pre::Qwen2
            } else if mt.starts_with("gemma4") {
                Pre::Gemma
            } else {
                Pre::Other
            };
        }

        Self::from_parts(&vocab, bpe_ranks, special, special_user, pre, ignore_merges)
    }
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    pub fn piece_bytes_into(&self, tok: u32, out: &mut Vec<u8>) {
        let Some(p) = self.vocab.get(tok as usize) else {
            return;
        };
        if self.pre == Pre::Gemma {
            self.piece_bytes_gemma(p, out);
            return;
        }
        for c in p.chars() {
            match self.c2b.get(&c) {
                Some(&b) => out.push(b),
                None => {
                    let mut tmp = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                }
            }
        }
    }

    fn piece_bytes_gemma(&self, p: &str, out: &mut Vec<u8>) {
        if p.len() == 6
            && p.starts_with("<0x")
            && p.ends_with('>')
            && let Ok(b) = u8::from_str_radix(&p[3..5], 16)
        {
            out.push(b);
            return;
        }
        for c in p.chars() {
            if c == '▁' {
                out.push(b' ');
            } else {
                let mut tmp = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Ok(c) = self.encode_cache.lock()
            && let Some(ids) = c.get(text)
        {
            return ids;
        }
        let ids = self.encode_opts(text, true);
        if let Ok(mut c) = self.encode_cache.lock() {
            c.put(text, &ids);
        }
        ids
    }

    pub fn encode_opts(&self, text: &str, parse_special: bool) -> Vec<u32> {
        if self.pre == Pre::Other || text.is_empty() {
            return self.encode_greedy(text);
        }
        let mut out = Vec::new();
        for f in self.partition_special(text, parse_special) {
            match f {
                Frag::Tok(id) => out.push(id),
                Frag::Text(s) => {
                    if self.pre == Pre::Gemma {
                        self.bpe_fragment_gemma(s, &mut out);
                    } else {
                        self.bpe_fragment(s, &mut out);
                    }
                }
            }
        }
        out
    }

    pub fn encode_greedy(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let mut best = 0usize;
            let mut best_id = None;
            let max = (i + 64).min(bytes.len());
            for e in (i + 1..=max).rev() {
                if let Some(&id) = self.greedy_index.get(&bytes[i..e]) {
                    best = e - i;
                    best_id = Some(id);
                    break;
                }
            }
            match best_id {
                Some(id) if best > 0 => {
                    out.push(id);
                    i += best;
                }
                _ => {
                    if i < bytes.len()
                        && let Some(&id) = self.greedy_index.get(&bytes[i..i + 1])
                    {
                        out.push(id);
                    }
                    i += 1;
                }
            }
        }
        out
    }


    fn partition_special<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Frag<'a>> {
        let mut frags = vec![Frag::Text(text)];
        let ordered: &[(String, u32)] = if parse_special {
            &self.special
        } else {
            &self.special_user
        };
        for (stext, sid) in ordered {
            if stext.is_empty() {
                continue;
            }
            let mut next: Vec<Frag<'a>> = Vec::with_capacity(frags.len());
            for f in frags.drain(..) {
                let Frag::Text(seg) = f else {
                    next.push(f);
                    continue;
                };
                let mut rest = seg;
                loop {
                    match rest.find(stext.as_str()) {
                        Some(at) => {
                            if at > 0 {
                                next.push(Frag::Text(&rest[..at]));
                            }
                            next.push(Frag::Tok(*sid));
                            rest = &rest[at + stext.len()..];
                        }
                        None => {
                            if !rest.is_empty() {
                                next.push(Frag::Text(rest));
                            }
                            break;
                        }
                    }
                }
            }
            frags = next;
        }
        frags
    }


    fn bpe_fragment(&self, text: &str, out: &mut Vec<u32>) {
        let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let mut boff = Vec::with_capacity(cpts.len() + 1);
        let mut bo = 0usize;
        for c in text.chars() {
            boff.push(bo);
            bo += c.len_utf8();
        }
        boff.push(bo);
        let bytes = text.as_bytes();
        let accent = self.pre == Pre::Qwen35;
        for (s, len) in split_pre(&cpts, accent) {
            let wb = &bytes[boff[s]..boff[s + len]];
            let mut enc = String::with_capacity(wb.len() * 2);
            let mut soff: Vec<usize> = Vec::with_capacity(wb.len() + 1);
            for &b in wb {
                soff.push(enc.len());
                enc.push(self.b2c[&b]);
            }
            soff.push(enc.len());
            let eb = enc.as_bytes();
            if self.ignore_merges
                && let Some(&id) = self.text_to_id.get(eb)
            {
                out.push(id);
                continue;
            }
            let mut syms: Vec<(usize, usize)> = (0..wb.len())
                .map(|i| (soff[i], soff[i + 1] - soff[i]))
                .collect();
            let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
            let mut key = Vec::new();
            for i in 1..syms.len() {
                self.push_bigram(&mut heap, &syms, i - 1, i, eb, &mut key);
            }
            while let Some(b) = heap.pop() {
                if syms[b.li].1 != b.ln || syms[b.ri].1 != b.rn {
                    continue;
                }
                syms[b.li].1 += syms[b.ri].1;
                syms[b.ri].1 = 0;
                let mut p = b.li;
                while p > 0 && syms[p - 1].1 == 0 {
                    p -= 1;
                }
                if p > 0 {
                    self.push_bigram(&mut heap, &syms, p - 1, b.li, eb, &mut key);
                }
                let mut q = b.ri + 1;
                while q < syms.len() && syms[q].1 == 0 {
                    q += 1;
                }
                if q < syms.len() {
                    self.push_bigram(&mut heap, &syms, b.li, q, eb, &mut key);
                }
            }
            for &(st, n) in &syms {
                if n == 0 {
                    continue;
                }
                let piece = &eb[st..st + n];
                if let Some(&id) = self.text_to_id.get(piece) {
                    out.push(id);
                } else {
                    for j in 0..piece.len() {
                        if let Some(&id) = self.text_to_id.get(&piece[j..j + 1]) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    fn bpe_fragment_gemma(&self, text: &str, out: &mut Vec<u32>) {
        if text.contains(' ') {
            let norm = text.replace(' ', "▁");
            self.bpe_runs_gemma(&norm, out);
        } else {
            self.bpe_runs_gemma(text, out);
        }
    }

    fn bpe_runs_gemma(&self, s: &str, out: &mut Vec<u32>) {
        let mut run_lo = 0usize;
        for (b, ch) in s.char_indices() {
            let mut buf = [0u8; 4];
            let piece = ch.encode_utf8(&mut buf);
            if self.text_to_id.contains_key(piece.as_bytes()) {
                continue;
            }
            if b > run_lo {
                self.bpe_run_gemma(&s[run_lo..b], out);
            }
            for byte in piece.as_bytes() {
                let name = format!("<0x{byte:02X}>");
                if let Some(&id) = self.text_to_id.get(name.as_bytes()) {
                    out.push(id);
                }
            }
            run_lo = b + ch.len_utf8();
        }
        if run_lo < s.len() {
            self.bpe_run_gemma(&s[run_lo..], out);
        }
    }

    fn bpe_run_gemma(&self, run: &str, out: &mut Vec<u32>) {
        let eb = run.as_bytes();
        if self.ignore_merges
            && let Some(&id) = self.text_to_id.get(eb)
        {
            out.push(id);
            return;
        }
        let mut syms: Vec<(usize, usize)> = Vec::new();
        for (b, ch) in run.char_indices() {
            syms.push((b, ch.len_utf8()));
        }
        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
        let mut key = Vec::new();
        for i in 1..syms.len() {
            self.push_bigram(&mut heap, &syms, i - 1, i, eb, &mut key);
        }
        while let Some(b) = heap.pop() {
            if syms[b.li].1 != b.ln || syms[b.ri].1 != b.rn {
                continue;
            }
            syms[b.li].1 += syms[b.ri].1;
            syms[b.ri].1 = 0;
            let mut p = b.li;
            while p > 0 && syms[p - 1].1 == 0 {
                p -= 1;
            }
            if p > 0 {
                self.push_bigram(&mut heap, &syms, p - 1, b.li, eb, &mut key);
            }
            let mut q = b.ri + 1;
            while q < syms.len() && syms[q].1 == 0 {
                q += 1;
            }
            if q < syms.len() {
                self.push_bigram(&mut heap, &syms, b.li, q, eb, &mut key);
            }
        }
        for &(st, n) in &syms {
            if n == 0 {
                continue;
            }
            let piece = &eb[st..st + n];
            if let Some(&id) = self.text_to_id.get(piece) {
                out.push(id);
            } else {
                for &byte in piece {
                    let name = format!("<0x{byte:02X}>");
                    if let Some(&id) = self.text_to_id.get(name.as_bytes()) {
                        out.push(id);
                    }
                }
            }
        }
    }

    fn push_bigram(
        &self,
        heap: &mut BinaryHeap<Bigram>,
        syms: &[(usize, usize)],
        l: usize,
        r: usize,
        eb: &[u8],
        key: &mut Vec<u8>,
    ) {
        let (lst, lln) = syms[l];
        let (rst, rln) = syms[r];
        let left = &eb[lst..lst + lln];
        let right = &eb[rst..rst + rln];
        key.clear();
        key.extend_from_slice(&(left.len() as u32).to_le_bytes());
        key.extend_from_slice(left);
        key.extend_from_slice(right);
        if let Some(&rank) = self.bpe_ranks.get(key) {
            heap.push(Bigram {
                rank,
                li: l,
                ri: r,
                ln: lln,
                rn: rln,
            });
        }
    }
}

enum Frag<'a> {
    Text(&'a str),
    Tok(u32),
}

#[derive(Clone, Copy)]
struct Bigram {
    rank: u32,
    li: usize,
    ri: usize,
    ln: usize,
    rn: usize,
}
impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.rank == o.rank && self.li == o.li
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (o.rank, o.li).cmp(&(self.rank, self.li))
    }
}


fn flags_bits(cpt: u32) -> u16 {
    let i = match udata::RANGES_FLAGS.binary_search_by(|&(s, _)| s.cmp(&cpt)) {
        Ok(i) => i,
        Err(i) => i - 1,
    };
    let mut f = udata::RANGES_FLAGS[i].1;
    if udata::SET_WHITESPACE.binary_search(&cpt).is_ok() {
        f |= F_WHITESPACE;
    }
    f
}

fn tolower(cpt: u32) -> u32 {
    match udata::MAP_LOWERCASE.binary_search_by(|&(k, _)| k.cmp(&cpt)) {
        Ok(i) => udata::MAP_LOWERCASE[i].1,
        Err(_) => cpt,
    }
}

fn split_pre(cpts: &[u32], accent: bool) -> Vec<(usize, usize)> {
    let end = cpts.len();
    let mut segs = Vec::new();
    let mut prev_end = 0usize;
    let add = |segs: &mut Vec<(usize, usize)>, prev_end: &mut usize, e: usize| {
        if e > *prev_end {
            segs.push((*prev_end, e - *prev_end));
        }
        *prev_end = e;
    };
    let bad_symbol = |p: usize| -> bool {
        if p >= end {
            return true;
        }
        let f = flags_bits(cpts[p]);
        f & (F_WHITESPACE | F_LETTER | F_NUMBER) != 0 || (accent && f & F_ACCENT != 0)
    };
    let is_lm = |p: usize| -> bool {
        if p >= end {
            return false;
        }
        let f = flags_bits(cpts[p]);
        f & F_LETTER != 0 || (accent && f & F_ACCENT != 0)
    };

    let mut pos = 0usize;
    while pos < end {
        let cpt = cpts[pos];
        let flags = flags_bits(cpt);

        if cpt == b'\'' as u32 && pos + 1 < end {
            let nxt = tolower(cpts[pos + 1]);
            if matches!(nxt, 0x73 | 0x74 | 0x6D | 0x64) {
                add(&mut segs, &mut prev_end, pos + 2);
                pos += 2;
                continue;
            }
            if pos + 2 < end {
                let nn = tolower(cpts[pos + 2]);
                if (matches!(nxt, 0x72 | 0x76) && nn == 0x65) || (nxt == 0x6C && nn == 0x6C) {
                    add(&mut segs, &mut prev_end, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }

        if cpt != 0x0D && cpt != 0x0A && flags & F_NUMBER == 0 && (is_lm(pos) || is_lm(pos + 1)) {
            pos += 1;
            while is_lm(pos) {
                pos += 1;
            }
            add(&mut segs, &mut prev_end, pos);
            continue;
        }

        if flags & F_NUMBER != 0 {
            pos += 1;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }

        if !bad_symbol(if cpt == 0x20 { pos + 1 } else { pos }) {
            pos += usize::from(cpt == 0x20);
            while !bad_symbol(pos) {
                pos += 1;
            }
            while pos < end && (cpts[pos] == 0x0D || cpts[pos] == 0x0A) {
                pos += 1;
            }
            add(&mut segs, &mut prev_end, pos);
            continue;
        }

        let mut nw = 0usize;
        let mut last_rn = 0usize;
        while pos + nw < end && flags_bits(cpts[pos + nw]) & F_WHITESPACE != 0 {
            if cpts[pos + nw] == 0x0D || cpts[pos + nw] == 0x0A {
                last_rn = pos + nw + 1;
            }
            nw += 1;
        }

        if last_rn > 0 {
            pos = last_rn;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }
        if nw > 1 && pos + nw < end {
            pos += nw - 1;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }
        if nw > 0 {
            pos += nw;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }
        pos += 1;
        add(&mut segs, &mut prev_end, pos);
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Tokenizer {
        let vocab: Vec<String> = ["a", "b", "c", "ab"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut ranks: FastMap<Vec<u8>, u32> = FastMap::default();
        let mut key = vec![1u8, 0, 0, 0];
        key.push(b'a');
        key.push(b'b');
        ranks.insert(key, 0);
        Tokenizer::from_parts(&vocab, ranks, Vec::new(), Vec::new(), Pre::Other, false).unwrap()
    }

    #[test]
    fn encode_cache_is_used_and_consistent() {
        let t = tiny();
        let text = "abcabcabcabcabcabcabcabcabcabcabcabc";
        let uncached = t.encode_opts(text, true);
        let first = t.encode(text);
        assert_eq!(first, uncached, "캐시 미스 결과 = 비캐시");
        assert!(
            t.encode_cache.lock().unwrap().map.contains_key(text),
            "첫 encode 후 캐시 등록"
        );
        let second = t.encode(text);
        assert_eq!(second, uncached, "캐시 히트 결과 = 비캐시");
    }

    #[test]
    fn special_lists_pre_sorted() {
        let mut ranks: FastMap<Vec<u8>, u32> = FastMap::default();
        ranks.insert(vec![0u8, 0, 0, 0], 0);
        let vocab: Vec<String> = ["x"].iter().map(|s| s.to_string()).collect();
        let special = vec![("A".to_string(), 1u32), ("LONG".to_string(), 2u32)];
        let t =
            Tokenizer::from_parts(&vocab, ranks, special, Vec::new(), Pre::Qwen2, false).unwrap();
        assert_eq!(
            t.special
                .iter()
                .map(|(s, _)| s.as_str())
                .collect::<Vec<_>>(),
            vec!["LONG", "A"],
            "길이 내림차순 1회 정렬"
        );
    }

    #[test]
    fn gemma_mode_normalize_merge_fallback_decode() {
        let pieces = [
            "▁", "a", "b", "▁a", "▁b", "ab", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>",
        ];
        let vocab: Vec<String> = pieces.iter().map(|s| s.to_string()).collect();
        let mut ranks: FastMap<Vec<u8>, u32> = FastMap::default();
        for (rank, (f, s)) in [("▁", "a"), ("▁", "b"), ("a", "b")].into_iter().enumerate() {
            let mut key = Vec::new();
            key.extend_from_slice(&(f.len() as u32).to_le_bytes());
            key.extend_from_slice(f.as_bytes());
            key.extend_from_slice(s.as_bytes());
            ranks.insert(key, rank as u32);
        }
        let t = Tokenizer::from_parts(&vocab, ranks, Vec::new(), Vec::new(), Pre::Gemma, false)
            .unwrap();
        assert_eq!(t.encode_opts("a b", false), vec![1, 4]);
        assert_eq!(t.encode_opts("  ", false), vec![0, 0]);
        assert_eq!(t.encode_opts("😀", false), vec![6, 7, 8, 9]);
        let mut bytes = Vec::new();
        for id in t.encode_opts("a b", false) {
            t.piece_bytes_into(id, &mut bytes);
        }
        assert_eq!(String::from_utf8(bytes).unwrap(), "a b");
        let mut one = Vec::new();
        t.piece_bytes_into(6, &mut one);
        assert_eq!(one, vec![0xF0], "<0x..> → 원바이트");
        let mut sp = Vec::new();
        t.piece_bytes_into(0, &mut sp);
        assert_eq!(sp, vec![b' '], "▁ → 공백");
    }
}
