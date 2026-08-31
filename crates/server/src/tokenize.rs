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
}

impl Tokenizer {
    /// 빈 토크나이저 (로드 최종 실패시 — 토큰 id 모드만 동작).
    pub fn empty() -> Self {
        Tokenizer { vocab: Vec::new(), index: HashMap::new() }
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
        for (i, t) in vocab.iter().enumerate() {
            // Ġ류 공백 마커 복원 (qwen BPE 관례)
            let bytes: Vec<u8> = t
                .chars()
                .flat_map(|c| {
                    if c == 'Ġ' {
                        vec![b' ']
                    } else if c == 'Ċ' {
                        vec![b'\n']
                    } else if (c as u32) < 0x100 {
                        vec![c as u8]
                    } else {
                        c.to_string().into_bytes()
                    }
                })
                .collect();
            if !bytes.is_empty() {
                index.entry(bytes).or_insert(i as u32);
            }
        }
        Ok(Tokenizer { vocab, index })
    }

    pub fn piece(&self, tok: u32) -> String {
        self.vocab
            .get(tok as usize)
            .cloned()
            .unwrap_or_default()
            .replace('Ġ', " ")
            .replace('Ċ', "\n")
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
