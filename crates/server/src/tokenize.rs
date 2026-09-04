//! 탐욕 최장일치 토크나이저 (근사 — 자기일관 검증용).
//! llama.cpp의 BPE pre-tokenizer·병합 순서와 일치하지 않음:
//! 서버 자체의 /tokenize→생성→판정 루프가 자기일관이면 매트릭스 유효.

use llm170_gguf::GgufFile;
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    vocab: Vec<String>,
    /// 바이트 문자열 → 토큰 id (탐욕 매칭용)
    index: HashMap<Vec<u8>, u32>,
    /// GPT-2 bytes_to_unicode 역표 (조각 문자 → 원바이트) — 디코딩용.
    c2b: HashMap<char, u8>,
}

impl Tokenizer {
    /// 빈 토크나이저 (로드 최종 실패시 — 토큰 id 모드만 동작).
    pub fn empty() -> Self {
        Tokenizer {
            vocab: Vec::new(),
            index: HashMap::new(),
            c2b: HashMap::new(),
        }
    }

    /// GPT-2 bytes_to_unicode 역표 (llama.cpp ggml과 동일 방식).
    fn byte_table() -> HashMap<char, u8> {
        let mut c2b = HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let printable =
                (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
            let c = if printable { b } else { 256 + n };
            if !printable {
                n += 1;
            }
            c2b.insert(char::from_u32(c).unwrap(), b as u8);
        }
        c2b
    }

    pub fn load(path: &Path, part2: Option<&Path>) -> Result<Self, String> {
        let g = GgufFile::open(path).map_err(|e| e.to_string())?;
        let mut vocab = Vec::new();
        // part1(메타 전용)에 토크나이저가 있음 — 실패시 part2 시도
        let toks = g
            .kv("tokenizer.ggml.tokens")
            .and_then(llm170_gguf::Value::as_array)
            .map(|(_, v)| v.to_vec());
        let toks = match toks {
            Some(t) if !t.is_empty() => t,
            _ => {
                let g2 = part2
                    .map(GgufFile::open)
                    .transpose()
                    .map_err(|e| e.to_string())?;
                g2.and_then(|g| {
                    g.kv("tokenizer.ggml.tokens")
                        .and_then(llm170_gguf::Value::as_array)
                        .map(|(_, v)| v.to_vec())
                })
                .unwrap_or_default()
            }
        };
        for t in &toks {
            vocab.push(t.as_str().unwrap_or("").to_string());
        }
        let mut index = HashMap::new();
        // GPT-2 bytes_to_unicode 역표 (llama.cpp ggml 동일) — qwen 바이트 수준 BPE:
        // 인쇄 가능 latin 범위는 자기 자신, 나머지 바이트(제어·0x7F-0xA0·0xAD)는
        // U+0100+ 순차 매핑. Ġ(0x120)=space, Ċ(0x10A)=\n도 이 표에 포함.
        let mut c2b: HashMap<char, u8> = HashMap::new();
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
                c2b.insert(char::from_u32(c).unwrap(), b as u8);
            }
        }
        for (i, t) in vocab.iter().enumerate() {
            let bytes: Vec<u8> = t
                .chars()
                .flat_map(|c| match c2b.get(&c) {
                    Some(&b) => vec![b],
                    None => c.to_string().into_bytes(),
                })
                .collect();
            if !bytes.is_empty() {
                index.entry(bytes).or_insert(i as u32);
            }
        }
        Ok(Tokenizer { vocab, index, c2b })
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

    pub fn piece(&self, tok: u32) -> String {
        String::from_utf8_lossy(&self.piece_bytes(tok)).into_owned()
    }

    /// 탐욕 최장일치 — 매칭 실패 바이트는 1바이트 토큰으로 (있으면) 매핑.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let mut best = 0usize;
            let mut best_id = None;
            let max = (i + 64).min(bytes.len());
            for e in (i + 1..=max).rev() {
                if let Some(&id) = self.index.get(&bytes[i..e]) {
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
                    if let Some(&id) = self.index.get(&bytes[i..i + 1]) {
                        out.push(id);
                    }
                    i += 1;
                }
            }
        }
        out
    }
}
