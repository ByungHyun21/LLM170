//! BPE 토크나이저 — llama.cpp `llama-vocab.cpp`·`unicode.cpp` 미러 (plans/83 A).
//!
//! 대상 pre: `qwen35`·`qwen2` (양대 게이트 모델). GPT-2 바이트 수준 BPE 전체 경로:
//! 특수 토큰 최장 분할 → pre-tokenizer 분할(qwen35/qwen2 커스텀 스플리터) →
//! bytes_to_unicode 인코딩 → 병합 랭크 우선순위 큐 루프.
//! 미지원 pre는 기존 탐욕 최장일치(`encode_greedy`)로 폴백 — 기동은 유지.
//! 디코딩(`piece_bytes`)은 기존 c2b 역표를 그대로 사용.

use crate::unicode_data as udata;
use llm170_gguf::GgufFile;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

// unicode_cpt_flags 비트 (llama.cpp unicode.h 동일 값)
const F_NUMBER: u16 = 0x0002;
const F_LETTER: u16 = 0x0004;
const F_ACCENT: u16 = 0x0010;
const F_WHITESPACE: u16 = 0x0100;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pre {
    Qwen35,
    Qwen2,
    Other,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    /// 토큰 원문 바이트 → id (llama.cpp text_to_token — 바이트 정확 매칭)
    text_to_id: HashMap<Box<[u8]>, u32>,
    /// 병합 키 = len_le32 ++ left ++ right → 랭크 (llama.cpp find_bpe_rank)
    bpe_ranks: HashMap<Vec<u8>, u32>,
    /// 특수 토큰 (본문 바이트 길이 내림차순) — 파티션용. CONTROL/USER_DEFINED/UNKNOWN
    special: Vec<(String, u32)>,
    /// USER_DEFINED 특수 토큰 — parse_special=false에서도 분할 (llama.cpp 규칙)
    special_user: Vec<(String, u32)>,
    pre: Pre,
    ignore_merges: bool,
    /// GPT-2 bytes_to_unicode 역표 (조각 문자 → 원바이트) — 디코딩용.
    c2b: HashMap<char, u8>,
    /// 원바이트 → 조각 문자 — 인코딩용.
    b2c: HashMap<u8, char>,
    /// 탐욕 폴백용 바이트열 → id (기존 index와 동일 규칙).
    greedy_index: HashMap<Vec<u8>, u32>,
}

/// 토큰 KV 존재 판정 (part1/part2 선택용).
fn has_tokens(g: &GgufFile) -> bool {
    g.kv("tokenizer.ggml.tokens")
        .and_then(llm170_gguf::Value::as_array)
        .map(|(_, v)| !v.is_empty())
        .unwrap_or(false)
}

impl Tokenizer {
    /// 빈 토크나이저 (로드 최종 실패시 — 토큰 id 모드만 동작).
    pub fn empty() -> Self {
        Tokenizer {
            vocab: Vec::new(),
            text_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            special: Vec::new(),
            special_user: Vec::new(),
            pre: Pre::Other,
            ignore_merges: false,
            c2b: HashMap::new(),
            b2c: HashMap::new(),
            greedy_index: HashMap::new(),
        }
    }

    pub fn load(path: &Path, part2: Option<&Path>) -> Result<Self, String> {
        let g = GgufFile::open(path).map_err(|e| e.to_string())?;
        if has_tokens(&g) {
            return Self::from_gguf(&g);
        }
        // part1(메타 전용)에 토크나이저가 없음 — part2 시도
        if let Some(g2) = part2
            .map(GgufFile::open)
            .transpose()
            .map_err(|e| e.to_string())?
        {
            if has_tokens(&g2) {
                return Self::from_gguf(&g2);
            }
        }
        Ok(Tokenizer::empty())
    }

    fn from_gguf(g: &GgufFile) -> Result<Self, String> {
        let toks = g
            .kv("tokenizer.ggml.tokens")
            .and_then(llm170_gguf::Value::as_array)
            .map(|(_, v)| v.to_vec())
            .unwrap_or_default();
        let mut vocab = Vec::with_capacity(toks.len());
        for t in &toks {
            vocab.push(t.as_str().unwrap_or("").to_string());
        }

        // pre 타입
        let pre_str = g
            .kv("tokenizer.ggml.pre")
            .and_then(llm170_gguf::Value::as_str)
            .unwrap_or("");
        let pre = match pre_str {
            "qwen35" => Pre::Qwen35,
            "qwen2" => Pre::Qwen2,
            _ => Pre::Other,
        };
        let ignore_merges = g
            .kv("tokenizer.ggml.ignore_merges")
            .and_then(llm170_gguf::Value::as_u64)
            .map(|v| v != 0)
            .unwrap_or(false);

        // 병합 랭크 — llama.cpp: 위치 1부터 첫 ' ' 분할, 중복 emplace(선발 우선)
        let mut bpe_ranks = HashMap::new();
        if let Some(merges) = g
            .kv("tokenizer.ggml.merges")
            .and_then(llm170_gguf::Value::as_array)
            .map(|(_, v)| v.to_vec())
        {
            for (i, m) in merges.iter().enumerate() {
                let w = m.as_str().unwrap_or("");
                let b = w.as_bytes();
                let (first, second) = match b[1..].iter().position(|&c| c == b' ') {
                    Some(p) => (&b[..p + 1], &b[p + 2..]),
                    None => continue, // 스페이스 없음 — llama.cpp의 ("","") 등재는 미사용
                };
                let mut key = Vec::with_capacity(4 + first.len() + second.len());
                key.extend_from_slice(&(first.len() as u32).to_le_bytes());
                key.extend_from_slice(first);
                key.extend_from_slice(second);
                bpe_ranks.entry(key).or_insert(i as u32);
            }
        }

        // 특수 토큰: token_type ∈ {UNKNOWN=2, CONTROL=3, USER_DEFINED=4}
        let mut special: Vec<(String, u32)> = Vec::new();
        let mut special_user: Vec<(String, u32)> = Vec::new();
        if let Some(types) = g
            .kv("tokenizer.ggml.token_type")
            .and_then(llm170_gguf::Value::as_array)
            .map(|(_, v)| v.to_vec())
        {
            for (i, t) in types.iter().take(vocab.len()).enumerate() {
                let ty = t.as_u64().unwrap_or(1) as i32;
                if ty == 4 {
                    special_user.push((vocab[i].clone(), i as u32));
                } else if matches!(ty, 2 | 3) {
                    special.push((vocab[i].clone(), i as u32));
                }
            }
        }
        // 파티션 순서 = 본문 길이 내림차순 (llama.cpp cache_special_tokens 정렬)
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        special_user.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // GPT-2 bytes_to_unicode 정/역표 (기존 구현과 동일)
        let mut c2b: HashMap<char, u8> = HashMap::new();
        let mut b2c: HashMap<u8, char> = HashMap::new();
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

        let mut text_to_id: HashMap<Box<[u8]>, u32> = HashMap::new();
        let mut greedy_index: HashMap<Vec<u8>, u32> = HashMap::new();
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
            // llama.cpp token_to_id: 후발 덮어씀(중복시 마지막 승)
            text_to_id.insert(t.as_bytes().into(), i as u32);
        }
        Ok(Tokenizer {
            vocab,
            text_to_id,
            bpe_ranks,
            special,
            special_user,
            pre,
            ignore_merges,
            c2b,
            b2c,
            greedy_index,
        })
    }
    /// 토큰 조각의 원 바이트열 (바이트 수준 BPE 역매핑).
    pub fn piece_bytes(&self, tok: u32) -> Vec<u8> {
        self.vocab
            .get(tok as usize)
            .map(|p| {
                p.chars()
                    .flat_map(|c| match self.c2b.get(&c) {
                        Some(&b) => vec![b],
                        None => c.to_string().into_bytes(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }


    /// 텍스트 → 토큰 (특수 토큰 해석 포함 — llama-server 채팅 경로와 동일).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_opts(text, true)
    }

    /// 특수 토큰 미해석 판 (llama-tokenize 기본값과 동일 — 검증용).
    pub fn encode_opts(&self, text: &str, parse_special: bool) -> Vec<u32> {
        if self.pre == Pre::Other || text.is_empty() {
            return self.encode_greedy(text);
        }
        let mut out = Vec::new();
        for f in self.partition_special(text, parse_special) {
            match f {
                Frag::Tok(id) => out.push(id),
                Frag::Text(s) => self.bpe_fragment(s, &mut out),
            }
        }
        out
    }

    /// 기존 탐욕 최장일치 (Pre::Other 폴백).
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
                    if i < bytes.len() {
                        if let Some(&id) = self.greedy_index.get(&bytes[i..i + 1]) {
                            out.push(id);
                        }
                    }
                    i += 1;
                }
            }
        }
        out
    }

    // ── 특수 토큰 파티션 (llama.cpp tokenizer_st_partition 미러) ──

    fn partition_special<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Frag<'a>> {
        let mut frags = vec![Frag::Text(text)];
        // USER_DEFINED는 항상 분할 — parse_special=false에서도 (llama.cpp 규칙).
        // 분할 순서: 길이 내림차순 우선 — 전체 목록을 통합 정렬한다.
        let all: Vec<&(String, u32)> = if parse_special {
            self.special.iter().chain(self.special_user.iter()).collect()
        } else {
            self.special_user.iter().collect()
        };
        let mut ordered: Vec<&(String, u32)> = all;
        ordered.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
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

    // ── BPE 조각 토큰화 (llm_tokenizer_bpe_session::tokenize 미러) ──

    fn bpe_fragment(&self, text: &str, out: &mut Vec<u32>) {
        let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
        // 코드포인트 → 원 바이트 오프셋
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
            // 바이트 인코딩 문자열 + 심볼별 시작 오프셋(1심볼 = 원바이트 1개)
            let mut enc = String::with_capacity(wb.len() * 2);
            let mut soff: Vec<usize> = Vec::with_capacity(wb.len() + 1);
            for &b in wb {
                soff.push(enc.len());
                enc.push(self.b2c[&b]);
            }
            soff.push(enc.len());
            let eb = enc.as_bytes();
            if self.ignore_merges {
                if let Some(&id) = self.text_to_id.get(eb) {
                    out.push(id);
                    continue;
                }
            }
            // 심볼 (start, len) — 초기 1심볼 = 인코딩 문자 1개
            let mut syms: Vec<(usize, usize)> =
                (0..wb.len()).map(|i| (soff[i], soff[i + 1] - soff[i])).collect();
            let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
            let mut key = Vec::new();
            for i in 1..syms.len() {
                self.push_bigram(&mut heap, &syms, i - 1, i, eb, &mut key);
            }
            while let Some(b) = heap.pop() {
                if syms[b.li].1 != b.ln || syms[b.ri].1 != b.rn {
                    continue; // 갱신된 bigram
                }
                syms[b.li].1 += syms[b.ri].1; // 좌측 흡수
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
                    // 바이트 폴백 — 미매칭 바이트는 조용히 스킵 (llama.cpp 동일)
                    for j in 0..piece.len() {
                        if let Some(&id) = self.text_to_id.get(&piece[j..j + 1]) {
                            out.push(id);
                        }
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
        // 팝 순서 = (rank, left) 오름차순 (llama.cpp comparator 미러) —
        // BinaryHeap은 최대 우선 팝이므로 순서 역전.
        (o.rank, o.li).cmp(&(self.rank, self.li))
    }
}

// ── 유니코드 분류 (llama.cpp unicode-data 테이블 조회) ──

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

/// qwen35/qwen2 pre-tokenizer 분할 (unicode_regex_split_custom_qwen{35,2} 미러).
/// 반환: (start, len) — 코드포인트 좌표. `accent` = qwen35 (\p{M} 포함).
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
    // [^\s\p{L}(\p{M})\p{N}] 판정 — qwen2는 accent 제외
    let bad_symbol = |p: usize| -> bool {
        if p >= end {
            return true; // 범위 밖 = 루프 정지 의미
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

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == b'\'' as u32 && pos + 1 < end {
            let nxt = tolower(cpts[pos + 1]);
            if matches!(nxt, 0x73 | 0x74 | 0x6D | 0x64) {
                add(&mut segs, &mut prev_end, pos + 2);
                pos += 2;
                continue;
            }
            if pos + 2 < end {
                let nn = tolower(cpts[pos + 2]);
                if (nxt == 0x72 && nn == 0x65)
                    || (nxt == 0x76 && nn == 0x65)
                    || (nxt == 0x6C && nn == 0x6C)
                {
                    add(&mut segs, &mut prev_end, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  (qwen2: \p{L}+)
        if cpt != 0x0D && cpt != 0x0A && flags & F_NUMBER == 0 {
            if is_lm(pos) || is_lm(pos + 1) {
                pos += 1;
                while is_lm(pos) {
                    pos += 1;
                }
                add(&mut segs, &mut prev_end, pos);
                continue;
            }
        }

        // \p{N}
        if flags & F_NUMBER != 0 {
            pos += 1;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }

        // " "?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
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

        // \s*[\r\n]+
        if last_rn > 0 {
            pos = last_rn;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }
        // \s+(?!\S)
        if nw > 1 && pos + nw < end {
            pos += nw - 1;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }
        // \s+
        if nw > 0 {
            pos += nw;
            add(&mut segs, &mut prev_end, pos);
            continue;
        }
        // no match
        pos += 1;
        add(&mut segs, &mut prev_end, pos);
    }
    segs
}
