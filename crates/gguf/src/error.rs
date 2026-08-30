//! GGUF 오류. 외부 의존 없는 typed Result 정책 (AGENTS.md).

use std::fmt;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    NotGguf([u8; 4]),
    UnsupportedVersion(u32),
    EndiannessSuspect(u32),
    BadMetadataType(u32),
    BadArrayType(u32),
    BadUtf8(std::str::Utf8Error),
    EmptyKey,
    DuplicateKey(String),
    BadAlignment(u32),
    /// 텐서 GGML 타입 id가 범위 밖이거나 미지원
    UnknownTensorType(u32),
    BadDims(u32),
    DimTooLarge(u64),
    /// 문자열/배열 길이가 상한 초과 — 파일 손상 가능성
    LengthOverflow { what: &'static str, len: u64, max: u64 },
    TensorCountTooLarge(u64),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::NotGguf(m) => write!(f, "not a GGUF file (magic {:02x?})", m),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v} (지원: v3)"),
            Self::EndiannessSuspect(v) => write!(f, "version {v} 비정상 — 호스트/파일 엔디언 불일치 의심"),
            Self::BadMetadataType(t) => write!(f, "invalid metadata value type {t}"),
            Self::BadArrayType(t) => write!(f, "array element type {t} invalid"),
            Self::BadUtf8(e) => write!(f, "invalid utf-8 string: {e}"),
            Self::EmptyKey => write!(f, "empty metadata key"),
            Self::DuplicateKey(k) => write!(f, "duplicate metadata key '{k}'"),
            Self::BadAlignment(a) => write!(f, "alignment {a} not a power of 2 / zero"),
            Self::UnknownTensorType(t) => write!(f, "unknown tensor ggml type id {t}"),
            Self::BadDims(n) => write!(f, "tensor n_dims {n} out of range 1..=4"),
            Self::DimTooLarge(d) => write!(f, "tensor dim {d} exceeds sanity limit"),
            Self::LengthOverflow { what, len, max } => {
                write!(f, "{what} length {len} exceeds sanity max {max}")
            }
            Self::TensorCountTooLarge(n) => write!(f, "tensor count {n} exceeds sanity limit"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::BadUtf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<std::str::Utf8Error> for GgufError {
    fn from(e: std::str::Utf8Error) -> Self {
        Self::BadUtf8(e)
    }
}

pub type Result<T, E = GgufError> = std::result::Result<T, E>;
