//! GGUF v3 출력기 — EXL3→F16 GGUF 변환용(plans/118 §7-3②).
//!
//! 형식: `ggml/include/gguf.h` v3 — 매직 GGUF·버전 3·정렬 32.
//! kv는 llama.cpp 값 타입(u32=4, f32=6, 문자열=8, 배열=9, u64=10).

use crate::Result;
use std::io::Write;

const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");
const ALIGN: u64 = 32;

/// 쓰기 큐에 쌓는 kv 엔트리.
pub enum Kv {
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Str(String),
    StrArray(Vec<String>),
}

/// 텐서 정보 — 데이터는 순차 append.
struct TensorOut {
    name: String,
    ne: Vec<u64>,
    ty: u32, // 0=F32, 1=F16
    len: u64,
    offset: u64,
}

pub struct GgufWriter {
    kvs: Vec<(String, Kv)>,
    tensors: Vec<TensorOut>,
    data_len: u64,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            kvs: Vec::new(),
            tensors: Vec::new(),
            data_len: 0,
        }
    }

    pub fn kv(&mut self, key: &str, v: Kv) {
        self.kvs.push((key.to_string(), v));
    }

    /// 텐서 등록 — 이후 push_data로 정확히 len 바이트 순차 기입.
    pub fn tensor_f16(&mut self, name: &str, ne: &[u64]) -> u64 {
        self.tensor_ty(name, ne, 1, 2)
    }

    fn tensor_ty(&mut self, name: &str, ne: &[u64], ty: u32, elem: u64) -> u64 {
        let len: u64 = ne.iter().product::<u64>() * elem;
        let offset = align_up(self.data_len, ALIGN);
        self.tensors.push(TensorOut {
            name: name.to_string(),
            ne: ne.to_vec(),
            ty,
            len,
            offset,
        });
        self.data_len = offset + len;
        offset
    }

    /// 임의 원소 크기 텐서 등록 — 비원소 정렬 타입(W4A16G128=66B/128 등
    /// llm170 dialect)용. len은 호출자 계약(바이트).
    pub fn tensor_raw(&mut self, name: &str, ne: &[u64], ty: u32, len: u64) -> u64 {
        let offset = align_up(self.data_len, ALIGN);
        self.tensors.push(TensorOut {
            name: name.to_string(),
            ne: ne.to_vec(),
            ty,
            len,
            offset,
        });
        self.data_len = offset + len;
        offset
    }

    /// BF16 텐서 등록(ty=30).
    pub fn tensor_bf16(&mut self, name: &str, ne: &[u64]) -> u64 {
        self.tensor_ty(name, ne, 30, 2)
    }

    pub fn tensor_f32(&mut self, name: &str, ne: &[u64]) -> u64 {
        self.tensor_ty(name, ne, 0, 4)
    }

    /// 전체 기입 — kv → 헤더 → 텐서 정보 → 정렬 패딩 → 데이터 콜백.
    /// 데이터는 등록 순서대로 `sink(name, offset, len)`이 기입한다.
    pub fn write<W: Write, F: FnMut(&str, u64, u64, &mut W) -> Result<()>>(
        self,
        w: &mut W,
        mut sink: F,
    ) -> Result<()> {
        let mut head: Vec<u8> = Vec::new();
        put_u32(&mut head, GGUF_MAGIC);
        put_u32(&mut head, 3);
        put_u64(&mut head, self.tensors.len() as u64);
        put_u64(&mut head, self.kvs.len() as u64);
        for (k, v) in &self.kvs {
            put_str(&mut head, k);
            match v {
                Kv::U32(x) => {
                    put_u32(&mut head, 4);
                    put_u32(&mut head, *x);
                }
                Kv::U64(x) => {
                    put_u32(&mut head, 10);
                    put_u64(&mut head, *x);
                }
                Kv::F32(x) => {
                    put_u32(&mut head, 6);
                    head.extend_from_slice(&x.to_le_bytes());
                }
                Kv::F64(x) => {
                    put_u32(&mut head, 12);
                    head.extend_from_slice(&x.to_le_bytes());
                }
                Kv::Str(s) => {
                    put_u32(&mut head, 8);
                    put_str(&mut head, s);
                }
                Kv::StrArray(a) => {
                    put_u32(&mut head, 9);
                    put_u32(&mut head, 8); // 요소 타입 = 문자열
                    put_u64(&mut head, a.len() as u64);
                    for s in a {
                        put_str(&mut head, s);
                    }
                }
            }
        }
        for t in &self.tensors {
            put_str(&mut head, &t.name);
            put_u32(&mut head, t.ne.len() as u32);
            for d in &t.ne {
                put_u64(&mut head, *d);
            }
            put_u32(&mut head, t.ty);
            put_u64(&mut head, t.offset);
        }
        let data_start = align_up(head.len() as u64, ALIGN);
        // 헤더 뒤 패딩 — data 섹션을 data_start에 맞춘다.
        while head.len() < data_start as usize {
            head.push(0);
        }
        w.write_all(&head)?;
        let mut cursor = 0u64;
        for t in &self.tensors {
            // 텐서 간 정렬 패딩.
            while cursor < t.offset {
                w.write_all(&[0u8])?;
                cursor += 1;
            }
            sink(&t.name, t.offset, t.len, w)?;
            cursor += t.len;
        }
        Ok(())
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_via_parser() {
        // 최소 GGUF을 쓰고 llm170-gguf 파서로 재판독.
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Kv::Str("qwen35".into()));
        w.kv("qwen35.embedding_length", Kv::U32(8));
        w.kv(
            "tokenizer.ggml.tokens",
            Kv::StrArray(vec!["a".into(), "b".into()]),
        );
        w.tensor_f32("norm.weight", &[4]);
        w.tensor_f16("t.weight", &[2, 3]);
        let mut out = Vec::new();
        let f32v: Vec<u8> = [1f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.write(&mut out, |name, _off, len, w| {
            if name == "norm.weight" {
                w.write_all(&f32v[..len as usize])?;
            } else {
                let zeros = vec![0u8; len as usize];
                w.write_all(&zeros)?;
            }
            Ok(())
        })
        .unwrap();
        std::fs::write("/tmp/_exl3_gguf_test.gguf", &out).unwrap();
        let g = llm170_gguf::GgufFile::open(std::path::Path::new("/tmp/_exl3_gguf_test.gguf"))
            .map_err(|e| format!("{e}"))
            .unwrap();
        assert_eq!(g.arch(), Some("qwen35"));
        assert_eq!(g.arch_kv_u64("embedding_length"), Some(8));
        assert_eq!(g.find_tensor("t.weight").unwrap().ne, [2, 3, 1, 1]);
        std::fs::remove_file("/tmp/_exl3_gguf_test.gguf").ok();
    }
}
