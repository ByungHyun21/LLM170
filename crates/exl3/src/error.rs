//! EXL3 오류형 — thiserror 없이 Display 직구현(AGENTS.md 규약).

use std::fmt;

#[derive(Debug)]
pub enum Exl3Error {
    Io(std::io::Error),
    BadHeader(String),
    TensorNotFound(String),
    BadTensor(String),
    Utf8(std::str::Utf8Error),
}

impl fmt::Display for Exl3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exl3Error::Io(e) => write!(f, "io: {e}"),
            Exl3Error::BadHeader(s) => write!(f, "bad safetensors header: {s}"),
            Exl3Error::TensorNotFound(n) => write!(f, "tensor not found: {n}"),
            Exl3Error::BadTensor(s) => write!(f, "bad exl3 tensor: {s}"),
            Exl3Error::Utf8(e) => write!(f, "utf8: {e}"),
        }
    }
}

impl std::error::Error for Exl3Error {}

impl From<std::io::Error> for Exl3Error {
    fn from(e: std::io::Error) -> Self {
        Exl3Error::Io(e)
    }
}

impl From<std::str::Utf8Error> for Exl3Error {
    fn from(e: std::str::Utf8Error) -> Self {
        Exl3Error::Utf8(e)
    }
}

impl From<String> for Exl3Error {
    fn from(s: String) -> Self {
        Exl3Error::BadHeader(s)
    }
}

pub type Result<T> = std::result::Result<T, Exl3Error>;
