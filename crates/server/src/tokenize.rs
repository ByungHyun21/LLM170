//! BPE 토크나이저 — llama.cpp `llama-vocab.cpp`·`unicode.cpp` 미러.
//!
//! 대상 pre: `qwen35`·`qwen2` (양대 게이트 모델). GPT-2 바이트 수준 BPE 전체 경로:
//! 특수 토큰 최장 분할 → pre-tokenizer 분할(qwen35/qwen2 커스텀 스플리터) →
//! bytes_to_unicode 인코딩 → 병합 랭크 우선순위 큐 루프.
//! 미지원 pre는 기존 탐욕 최장일치(`encode_greedy`)로 폴백 — 기동은 유지.
//! 디코딩(`piece_bytes`)은 기존 c2b 역표를 그대로 사용.

use crate::unicode_data as udata;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

// unicode_cpt_flags 비트 (llama.cpp unicode.h 동일 값)
const F_NUMBER: u16 = 0x0002;
const F_LETTER: u16 = 0x0004;
const F_ACCENT: u16 = 0x0010;
const F_WHITESPACE: u16 = 0x0100;

/// [D5 2026-10-09] 자작 곱셈 해시 — 짧은 바이트 키(토큰 조각·병합 키)에서
/// SipHash는 과하다(로컬 단일 사용자 서버 — DoS 저항 불필요).
/// splitmix64 계열 상수 곱 + 회전, 8바이트 청크 + 꼬리.
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

/// 빠른 해시 맵(바이트·문자 키 전용 맵들).
type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<MulHasher>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pre {
    Qwen35,
    Qwen2,
    Other,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    /// 토큰 원문 바이트 → id (llama.cpp text_to_token — 바이트 정확 매칭)
    text_to_id: FastMap<Box<[u8]>, u32>,
    /// 병합 키 = len_le32 ++ left ++ right → 랭크 (llama.cpp find_bpe_rank)
    bpe_ranks: FastMap<Vec<u8>, u32>,
    /// 특수 토큰 — parse_special=true 파티션용(special ∪ special_user,
    /// 본문 바이트 길이 내림차순 **로드 시 1회 정렬** — D5).
    special: Vec<(String, u32)>,
    /// USER_DEFINED 특수 토큰 — parse_special=false에서도 분할 (llama.cpp 규칙).
    special_user: Vec<(String, u32)>,
    pre: Pre,
    ignore_merges: bool,
    /// GPT-2 bytes_to_unicode 역표 (조각 문자 → 원바이트) — 디코딩용.
    c2b: FastMap<char, u8>,
    /// 원바이트 → 조각 문자 — 인코딩용.
    b2c: FastMap<u8, char>,
    /// 탐욕 폴백용 바이트열 → id (기존 index와 동일 규칙).
    greedy_index: FastMap<Vec<u8>, u32>,
    /// [D5] 완전 일치 텍스트 캐시 — 동일 요청 재인코딩 제거(유계 FIFO).
    encode_cache: std::sync::Mutex<EncodeCache>,
}

/// [D5 2026-10-09] 동일 텍스트 재인코딩 캐시 — 재시도·중복 요청에서 BPE
/// 재계산 제거. **완전 일치만** — 접두 재개는 BPE 병합이 조각 경계를 넘을 수
/// 있어 시임 안전이 보장되지 않는다(필요 시 조각 단위 캐시 + 시임 검증).
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
        // 짧은 텍스트는 비용 대비 이득 없음(해시·복사 > BPE).
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
    pub fn load(path: &Path) -> Result<Self, String> {
        // 모델 = W4A16 디렉터리(2026-10-08 단일 트랙).
        if path.is_dir() {
            return Self::from_hf_dir(path);
        }
        Err(format!(
            "토크나이저: 디렉터리 모델(W4A16)만 지원(2026-10-08): {}",
            path.display()
        ))
    }

    /// 공통 꼬리 — 바이트 표·인덱스 조립(from_hf_dir 공유).
    fn from_parts(
        vocab: &[String],
        bpe_ranks: FastMap<Vec<u8>, u32>,
        special_ctl: Vec<(String, u32)>,
        mut special_user: Vec<(String, u32)>,
        pre: Pre,
        ignore_merges: bool,
    ) -> Result<Self, String> {
        // [D5] 파티션용 통합 목록을 로드 시 1회 정렬 — 종전엔 encode 호출마다
        // chain+collect+sort를 했다. special_ctl(CONTROL/UNKNOWN계) ∪ user.
        let mut special = special_ctl;
        special.extend(special_user.iter().cloned());
        special.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        special_user.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        // GPT-2 bytes_to_unicode 정/역표 (기존 구현과 동일)
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
            // llama.cpp token_to_id: 후발 덮어씀(중복시 마지막 승)
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

    /// W4A16 디렉터리 — vocab.json(조각→id) + merges.txt(BPE 순위) +
    /// tokenizer_config.json(added_tokens_decoder 특수 토큰) + config.json
    /// (model_type → pre 스플리터).
    /// 파서는 llm170_core::json 재사용.
    fn from_hf_dir(dir: &Path) -> Result<Self, String> {
        // 토큰 조각표 — vocab.json(HF 벌크) 우선, 없으면 tokenizer.json.
        // (W4A16 HF 배포는 vocab.json/merges.txt 부재 실측 — tokenizer.json의
        // model.vocab + added_tokens 병합으로 대체.)
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
            // added_tokens(특수 토큰 — EOS 포함) 병합.
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

        // 병합 순위 — 표준 키(첫 ' ' 분할, 선발 우선). # 헤더는
        // 순위 소모 없이 스킵(HF 변환기 배열 순서와 정렬).
        // merges.txt 우선, tokenizer.json이면 model.merges(문자열 배열) 파생.
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
                    // 두 형식: ["a","b"] 배열쌍(HF 최신 — 실측) · "a b" 문자열.
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
                if line.is_empty() || line.starts_with('#') {
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

        // 특수 토큰 — added_tokens_decoder {id: {content, special}}.
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
        // 폴백: tokenizer_config의 added_tokens_decoder가 비어 있으면
        // tokenizer.json added_tokens에서 수집(HF 최신 배포 — 실측:
        // W4A16 AutoRound는 decoder 부재·added_tokens 33종에 special 플래그).
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

        // pre 스플리터 — config.json model_type 계열.
        let mut pre = Pre::Other;
        if let Ok(cfg) = std::fs::read_to_string(dir.join("config.json"))
            && let Ok(cj) = llm170_core::json::Json::parse(&cfg)
            && let Some(cobj) = cj.as_object()
            && let Some((_, mt)) = cobj.iter().find(|(k, _)| k == "model_type")
            && let Some(mt) = mt.as_str()
        {
            pre = match mt {
                "qwen3_5" => Pre::Qwen35,
                "qwen2" | "qwen3" => Pre::Qwen2,
                _ => Pre::Other,
            };
        }

        Self::from_parts(&vocab, bpe_ranks, special, special_user, pre, ignore_merges)
    }
    /// 토큰 조각의 원 바이트열 (바이트 수준 BPE 역매핑).
    /// 어휘 비었는지(A19) — part1/part2 어느 쪽에도 토크나이저가
    /// 없으면 load가 Ok(empty)를 돌려주므로 serve 텍스트 요청에 치명 여부를
    /// 호출자가 판정해야 한다.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// [2026-10-09 D4] 문자당 Vec 할당 없이 out에 append — 종전 flat_map(vec![b])
    /// 체인이 문자마다 힙 할당이었다(토큰마다).
    pub fn piece_bytes_into(&self, tok: u32, out: &mut Vec<u8>) {
        let Some(p) = self.vocab.get(tok as usize) else {
            return;
        };
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

    /// 텍스트 → 토큰 (특수 토큰 해석 포함 — llama-server 채팅 경로와 동일).
    /// [D5] 완전 일치 재인코딩 캐시 경유(값은 비캐시 경로와 동일).
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

    // ── 특수 토큰 파티션 (llama.cpp tokenizer_st_partition 미러) ──

    fn partition_special<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Frag<'a>> {
        let mut frags = vec![Frag::Text(text)];
        // [D5] 정렬은 로드 시 1회 — 종전엔 호출마다 chain+collect+sort였다.
        // special = 통합(parse=true), special_user = USER_DEFINED(항상 분할,
        // llama.cpp 규칙) — 양쪽 모두 길이 내림차순.
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
            if self.ignore_merges
                && let Some(&id) = self.text_to_id.get(eb)
            {
                out.push(id);
                continue;
            }
            // 심볼 (start, len) — 초기 1심볼 = 인코딩 문자 1개
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
                if (matches!(nxt, 0x72 | 0x76) && nn == 0x65) || (nxt == 0x6C && nn == 0x6C) {
                    add(&mut segs, &mut prev_end, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  (qwen2: \p{L}+)
        if cpt != 0x0D && cpt != 0x0A && flags & F_NUMBER == 0 && (is_lm(pos) || is_lm(pos + 1)) {
            pos += 1;
            while is_lm(pos) {
                pos += 1;
            }
            add(&mut segs, &mut prev_end, pos);
            continue;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 최소 토크나이저 — 조각 a/b/c + 병합 (a,b). Pre::Other(탐욕 경로).
    fn tiny() -> Tokenizer {
        let vocab: Vec<String> = ["a", "b", "c", "ab"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut ranks: FastMap<Vec<u8>, u32> = FastMap::default();
        // 키 = len_le32 ++ left ++ right.
        let mut key = vec![1u8, 0, 0, 0];
        key.push(b'a');
        key.push(b'b');
        ranks.insert(key, 0);
        Tokenizer::from_parts(&vocab, ranks, Vec::new(), Vec::new(), Pre::Other, false).unwrap()
    }

    #[test]
    fn encode_cache_is_used_and_consistent() {
        let t = tiny();
        let text = "abcabcabcabcabcabcabcabcabcabcabcabc"; // 36바이트 ≥ 32(캐시 대상)
        let uncached = t.encode_opts(text, true);
        let first = t.encode(text);
        assert_eq!(first, uncached, "캐시 미스 결과 = 비캐시");
        assert!(
            t.encode_cache.lock().unwrap().map.contains_key(text),
            "첫 encode 후 캐시 등록"
        );
        let second = t.encode(text); // 캐시 히트 경로
        assert_eq!(second, uncached, "캐시 히트 결과 = 비캐시");
    }

    #[test]
    fn special_lists_pre_sorted() {
        // 로드 시 정렬 계약 — partition 결과의 토큰 순서가 길이 우선과 일치.
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
}
