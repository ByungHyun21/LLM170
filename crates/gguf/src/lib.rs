//! GGUF v3 파서 — 헤더·메타데이터·텐서 정보(무게 미로딩).
//!
//! 근거 원본: `source/llama.cpp/ggml/src/gguf.cpp` `gguf_init_from_reader` (2026-08-30 판),
//! `ggml/include/gguf.h` (GGUF_MAGIC "GGUF", GGUF_VERSION 3, GGUF_DEFAULT_ALIGNMENT 32).
//! v3만 지원 (v1 상류에서도 거부, v2는 현행 파일에서 존재하지 않음).

mod dump;
mod error;
mod types;
mod value;

pub use dump::write_dump;
pub use error::{GgufError, Result};
pub use types::GgmlType;
pub use value::{Value, ValueType};

use std::collections::HashSet;
use std::io::{BufReader, Read};
use std::path::Path;

// 손상 파일 방어 상한 — 정상 GGUF 범위보다 넉넉하게.
const MAX_STRING_BYTES: u64 = 1 << 26; // 64 MiB (개별 문자열)
const MAX_ARRAY_LEN: u64 = 1 << 26; // 토크나이저 ~25만 보다 훨씬 큼
const MAX_TENSORS: u64 = 1 << 20;
const MAX_TENSOR_NAME: u64 = 1 << 12;
const MAX_DIM: u64 = 1 << 40;

/// 단일 텐서의 파일 내 정보. `ne` 는 GGML 순서 (ne[0] 가 최속 축).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub ne: [u64; 4],
    pub ty: GgmlType,
    /// 데이터 섹션 시작 기준 오프셋 (파일 절대 아님)
    pub offset: u64,
}

impl TensorInfo {
    /// 이 텐서의 바이트 크기. ne[0] % blck != 0 (손상)면 None.
    pub fn nbytes(&self) -> Option<u64> {
        self.ty.nbytes(&self.ne)
    }

    /// 파일 절대 바이트 범위 (data_offset 포함)
    pub fn file_range(&self, data_offset: u64) -> Option<(u64, u64)> {
        let n = self.nbytes()?;
        Some((data_offset + self.offset, data_offset + self.offset + n))
    }
}

/// split.* 메타데이터 (첫 파일 기준).
#[derive(Debug, Clone, Copy)]
pub struct SplitInfo {
    pub no: u32,
    pub count: u32,
    /// 이 파일에 담긴 텐서 수
    pub tensors_count: u64,
}

/// 파싱된 GGUF 파일. 무게 데이터는 읽지 않는다 — 위치 계산만.
#[derive(Debug)]
pub struct GgufFile {
    pub path: std::path::PathBuf,
    pub file_size: u64,
    pub version: u32,
    pub alignment: u32,
    /// 텐서 데이터 섹션의 파일 절대 오프셋
    pub data_offset: u64,
    pub kv: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
}

struct Reader<'a> {
    r: &'a mut dyn Read,
    pos: u64,
}

impl<'a> Reader<'a> {
    fn new(r: &'a mut dyn Read) -> Self {
        Reader { r, pos: 0 }
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.r.read_exact(&mut buf)?;
        self.pos += n as u64;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// GGUF v3 문자열: u64 길이 + UTF-8 바이트.
    fn string(&mut self, what: &'static str, max: u64) -> Result<String> {
        let len = self.u64()?;
        if len > max {
            return Err(GgufError::LengthOverflow { what, len, max });
        }
        let b = self.bytes(len as usize)?;
        String::from_utf8(b).map_err(|e| GgufError::BadUtf8(e.utf8_error()))
    }

    fn scalar(&mut self, t: ValueType) -> Result<Value> {
        Ok(match t {
            ValueType::U8 => Value::U8(self.u8()?),
            ValueType::I8 => Value::I8(self.u8()? as i8),
            ValueType::U16 => Value::U16(self.u16()?),
            ValueType::I16 => Value::I16(self.u16()? as i16),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(self.u32()? as i32),
            ValueType::F32 => Value::F32(self.f32()?),
            ValueType::Bool => Value::Bool(self.u8()? != 0),
            ValueType::Str => Value::Str(self.string("value", MAX_STRING_BYTES)?),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(self.i64()?),
            ValueType::F64 => Value::F64(self.f64()?),
            ValueType::Array => unreachable!("array handled by caller"),
        })
    }

    fn value(&mut self) -> Result<Value> {
        let tag = self.u32()?;
        let vt = ValueType::from_u32(tag).ok_or(GgufError::BadMetadataType(tag))?;
        if vt != ValueType::Array {
            return self.scalar(vt);
        }
        // array: 요소 타입 u32 + 개수 u64 + 요소들 (gguf.cpp:575-579)
        let etag = self.u32()?;
        let et = ValueType::from_u32(etag).ok_or(GgufError::BadArrayType(etag))?;
        if et == ValueType::Array {
            // 상류도 배열-중첩 거부 (gguf.cpp:597-602)
            return Err(GgufError::BadArrayType(etag));
        }
        let n = self.u64()?;
        if n > MAX_ARRAY_LEN {
            return Err(GgufError::LengthOverflow {
                what: "array",
                len: n,
                max: MAX_ARRAY_LEN,
            });
        }
        let mut items = Vec::with_capacity(n as usize);
        for _ in 0..n {
            items.push(self.scalar(et)?);
        }
        Ok(Value::Array(et, items))
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    v.div_ceil(align) * align
}

impl GgufFile {
    /// 파일을 열어 헤더+메타+텐서 정보까지만 파싱 (무게 미로딩).
    pub fn open(path: &Path) -> Result<Self> {
        llm170_profiler::profile_span!("gguf::open");
        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut r = BufReader::new(file);
        let mut rd = Reader::new(&mut r);

        // magic
        let magic = rd.bytes(4)?;
        if magic != b"GGUF" {
            return Err(GgufError::NotGguf([magic[0], magic[1], magic[2], magic[3]]));
        }

        // header
        let version = rd.u32()?;
        if version == 0 || (version & 0x0000FFFF) == 0 {
            return Err(GgufError::EndiannessSuspect(version));
        }
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = rd.u64()?;
        if n_tensors > MAX_TENSORS {
            return Err(GgufError::TensorCountTooLarge(n_tensors));
        }
        let n_kv = rd.u64()?;

        // kv pairs (gguf.cpp:543-611)
        let mut kv: Vec<(String, Value)> = Vec::with_capacity(n_kv as usize);
        let mut seen: HashSet<String> = HashSet::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = rd.string("key", MAX_STRING_BYTES)?;
            if key.is_empty() {
                return Err(GgufError::EmptyKey);
            }
            if !seen.insert(key.clone()) {
                return Err(GgufError::DuplicateKey(key));
            }
            let value = rd.value()?;
            kv.push((key, value));
        }

        // alignment: general.alignment(u32, pow2) 없으면 32 (gguf.cpp:613-627)
        let alignment = match kv.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, v)) => {
                let a = v.as_u64().ok_or(GgufError::BadAlignment(0))? as u32;
                if a == 0 || (a & (a - 1)) != 0 {
                    return Err(GgufError::BadAlignment(a));
                }
                a
            }
            None => 32,
        };

        // tensor infos (gguf.cpp:630-)
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = rd.string("tensor-name", MAX_TENSOR_NAME)?;
            let n_dims = rd.u32()?;
            if !(1..=4).contains(&n_dims) {
                return Err(GgufError::BadDims(n_dims));
            }
            let mut ne = [1u64; 4];
            for d in ne.iter_mut().take(n_dims as usize) {
                let v = rd.u64()?;
                if v == 0 || v > MAX_DIM {
                    return Err(GgufError::DimTooLarge(v));
                }
                *d = v;
            }
            let tag = rd.u32()?;
            let ty = GgmlType::from_u32(tag).ok_or(GgufError::UnknownTensorType(tag))?;
            let offset = rd.u64()?;
            tensors.push(TensorInfo {
                name,
                n_dims,
                ne,
                ty,
                offset,
            });
        }

        let data_offset = align_up(rd.pos, alignment as u64);

        Ok(GgufFile {
            path: path.to_path_buf(),
            file_size,
            version,
            alignment,
            data_offset,
            kv,
            tensors,
        })
    }

    pub fn kv(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn kv_u64(&self, key: &str) -> Option<u64> {
        self.kv(key).and_then(Value::as_u64)
    }

    pub fn kv_str(&self, key: &str) -> Option<&str> {
        self.kv(key).and_then(Value::as_str)
    }

    /// general.architecture
    pub fn arch(&self) -> Option<&str> {
        self.kv_str("general.architecture")
    }

    /// split.* 정보 — split 메타가 없는 단일 파일이면 None.
    pub fn split(&self) -> Option<SplitInfo> {
        Some(SplitInfo {
            no: self.kv_u64("split.no")? as u32,
            count: self.kv_u64("split.count")? as u32,
            tensors_count: self.kv_u64("split.tensors.count")?,
        })
    }

    /// 아키텍처 접두 키 조회: `{arch}.{suffix}`.
    pub fn arch_kv(&self, suffix: &str) -> Option<&Value> {
        let arch = self.arch()?;
        self.kv(&format!("{arch}.{suffix}"))
    }

    pub fn arch_kv_u64(&self, suffix: &str) -> Option<u64> {
        self.arch_kv(suffix).and_then(Value::as_u64)
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 무게 데이터 총 바이트 (data_offset 이후 예상 크기).
    /// nbytes 계산 불가(손상) 텐서가 있으면 None.
    pub fn tensor_bytes_total(&self) -> Option<u64> {
        self.tensors
            .iter()
            .map(|t| t.nbytes())
            .collect::<Option<Vec<_>>>()
            .map(|v| v.iter().sum())
    }
}
