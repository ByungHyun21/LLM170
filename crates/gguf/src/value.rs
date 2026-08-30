//! GGUF 메타데이터 값.

use std::fmt;

/// gguf.h `enum gguf_metadata_value_type` 과 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    Str = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub fn from_u32(id: u32) -> Option<Self> {
        use ValueType::*;
        Some(match id {
            0 => U8, 1 => I8, 2 => U16, 3 => I16, 4 => U32, 5 => I32, 6 => F32,
            7 => Bool, 8 => Str, 9 => Array, 10 => U64, 11 => I64, 12 => F64,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use ValueType::*;
        match self {
            U8 => "u8", I8 => "i8", U16 => "u16", I16 => "i16", U32 => "u32",
            I32 => "i32", F32 => "f32", Bool => "bool", Str => "str",
            Array => "arr", U64 => "u64", I64 => "i64", F64 => "f64",
        }
    }
}

/// 배열 요소는 균일 타입 — Value 로 재귀 저장하되 Array 내부는 단일 타입만 담는다.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(ValueType, Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn type_tag(&self) -> ValueType {
        use ValueType::*;
        match self {
            Value::U8(_) => U8, Value::I8(_) => I8, Value::U16(_) => U16,
            Value::I16(_) => I16, Value::U32(_) => U32, Value::I32(_) => I32,
            Value::F32(_) => F32, Value::Bool(_) => Bool, Value::Str(_) => Str,
            Value::Array(_, _) => Array, Value::U64(_) => U64,
            Value::I64(_) => I64, Value::F64(_) => F64,
        }
    }

    /// 정수 계열 통일 접근 (split.no / context_length 등 횡단 조회용)
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::I8(v) => u64::try_from(*v).ok(),
            Value::U16(v) => Some(*v as u64),
            Value::I16(v) => u64::try_from(*v).ok(),
            Value::U32(v) => Some(*v as u64),
            Value::I32(v) => u64::try_from(*v).ok(),
            Value::U64(v) => Some(*v),
            Value::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            other => other.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<(&ValueType, &[Value])> {
        match self {
            Value::Array(t, v) => Some((t, v)),
            _ => None,
        }
    }
}

/// 사람이 읽는 덤프용. 배열은 앞부분 + 길이 요약.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Array(t, items) => {
                const PREVIEW: usize = 4;
                write!(f, "[{}; {}] ", t.name(), items.len())?;
                for (i, v) in items.iter().take(PREVIEW).enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{v}")?;
                }
                if items.len() > PREVIEW {
                    write!(f, ", …")?;
                }
                Ok(())
            }
            other => write!(f, "{}", other.as_u64().map_or_else(
                || other.as_f64().map_or_else(|| "?".into(), |v| v.to_string()),
                |v| v.to_string())),
        }
    }
}
