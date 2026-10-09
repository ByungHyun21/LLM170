//! safetensors 아카이브 리더 — 헤더 파서 + 샤드 인덱스 + 직독.
//! (2026-10-08 단일 트랙 재편에서 core로 이관 — W4A16 로더 공용 자산.)
//!
//! API 계약은 이관 전과 동일: entries()/entry()/read()/read_into()/
//! shard_paths()/shard_data_base(). 오류는 `StError`.

use crate::json::Json;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// safetensors 계열 오류 — thiserror 없이 Display 직구현(규약).
#[derive(Debug)]
pub enum StError {
    Io(std::io::Error),
    BadHeader(String),
    TensorNotFound(String),
    BadTensor(String),
    Utf8(std::str::Utf8Error),
}

impl std::fmt::Display for StError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StError::Io(e) => write!(f, "io: {e}"),
            StError::BadHeader(s) => write!(f, "bad safetensors header: {s}"),
            StError::TensorNotFound(n) => write!(f, "tensor not found: {n}"),
            StError::BadTensor(s) => write!(f, "bad tensor: {s}"),
            StError::Utf8(e) => write!(f, "utf8: {e}"),
        }
    }
}

impl std::error::Error for StError {}

impl From<std::io::Error> for StError {
    fn from(e: std::io::Error) -> Self {
        StError::Io(e)
    }
}

impl From<std::str::Utf8Error> for StError {
    fn from(e: std::str::Utf8Error) -> Self {
        StError::Utf8(e)
    }
}

impl From<String> for StError {
    fn from(s: String) -> Self {
        StError::BadHeader(s)
    }
}

pub type StResult<T> = std::result::Result<T, StError>;

/// safetensors dtype — 모델 배포에 실제 등장하는 종류만.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    F16,
    Bf16,
    F32,
    I16,
    I32,
    I64,
    U8,
}

impl StDtype {
    fn parse(s: &str) -> StResult<Self> {
        Ok(match s {
            "F16" => StDtype::F16,
            "BF16" => StDtype::Bf16,
            "F32" => StDtype::F32,
            "I16" => StDtype::I16,
            "I32" => StDtype::I32,
            "I64" => StDtype::I64,
            "U8" => StDtype::U8,
            other => return Err(StError::BadHeader(format!("unknown dtype {other}"))),
        })
    }

    pub fn nbytes(&self) -> u64 {
        match self {
            StDtype::F16 | StDtype::Bf16 | StDtype::I16 => 2,
            StDtype::F32 | StDtype::I32 => 4,
            StDtype::I64 => 8,
            StDtype::U8 => 1,
        }
    }
}

/// 단일 텐서 엔트리. `begin/end`는 샤드 데이터 섹션 기준 오프셋.
#[derive(Debug, Clone)]
pub struct StEntry {
    pub dtype: StDtype,
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
    /// 데이터가 있는 샤드(0-base).
    pub shard: usize,
}

impl StEntry {
    pub fn nbytes(&self) -> u64 {
        self.end - self.begin
    }
}

#[derive(Debug)]
struct Shard {
    path: PathBuf,
    /// 데이터 섹션 시작(= 8 + 헤더길이).
    data_base: u64,
    data_len: u64,
}

/// safetensors 아카이브 — 단일 파일 또는 dir(index.json 샤드 묶음).
#[derive(Debug)]
pub struct StArchive {
    shards: Vec<Shard>,
    entries: HashMap<String, StEntry>,
}

impl StArchive {
    /// `.safetensors` 파일 하나 또는 샤드 디렉터리를 연다.
    pub fn open(path: &Path) -> StResult<Self> {
        llm170_diag::profile_span!("st::open");
        if path.is_dir() {
            Self::open_dir(path)
        } else {
            let mut shards = Vec::new();
            let entries = Self::parse_shard(path, 0, &mut shards)?;
            Ok(Self { shards, entries })
        }
    }

    fn open_dir(dir: &Path) -> StResult<Self> {
        let idx_path = dir.join("model.safetensors.index.json");
        let mut entries = HashMap::new();
        let mut shard_names: Vec<String> = Vec::new();
        if idx_path.exists() {
            let mut raw = String::new();
            std::fs::File::open(&idx_path)?.read_to_string(&mut raw)?;
            let v = Json::parse(&raw)?;
            let map = v
                .get("weight_map")
                .and_then(Json::as_object)
                .ok_or_else(|| StError::BadHeader("index.json: weight_map missing".into()))?;
            for (name, file) in map {
                let file = file
                    .as_str()
                    .ok_or_else(|| StError::BadHeader("weight_map value not a string".into()))?;
                // H(2026-10-09): 중복 이름은 종전 마지막 우선(조용) — 거부.
                if entries.insert(name.clone(), file.to_string()).is_some() {
                    return Err(StError::BadHeader(format!(
                        "index.json: weight_map 중복 이름 {name}"
                    )));
                }
                if !shard_names.iter().any(|s| s == file) {
                    shard_names.push(file.to_string());
                }
            }
        } else {
            // index 없음: 디렉터리 내 model*.safetensors 전부(단일 파일 규약).
            for e in std::fs::read_dir(dir)? {
                let e = e?;
                let name = e.file_name();
                let name = name.to_string_lossy();
                if name.ends_with(".safetensors") && name.starts_with("model") {
                    shard_names.push(name.into_owned());
                }
            }
            shard_names.sort();
            if shard_names.is_empty() {
                return Err(StError::BadHeader(format!(
                    "no model*.safetensors in {}",
                    dir.display()
                )));
            }
        }
        shard_names.sort();
        let mut shards = Vec::new();
        let mut out = HashMap::new();
        for (si, fname) in shard_names.iter().enumerate() {
            let path = dir.join(fname);
            let shard_entries = Self::parse_shard(&path, si, &mut shards)?;
            if idx_path.exists() {
                // index가 있으면 엔트리는 weight_map이 지정한 샤드에만 귀속.
                for name in entries
                    .iter()
                    .filter(|(_, f)| f.as_str() == fname)
                    .map(|(n, _)| n)
                {
                    if let Some(mut e) = shard_entries.get(name).cloned() {
                        e.shard = si;
                        out.insert(name.clone(), e);
                    }
                }
            } else {
                for (name, mut e) in shard_entries {
                    e.shard = si;
                    out.insert(name, e);
                }
            }
        }
        if idx_path.exists() {
            // H(2026-10-09): weight_map이 지정했는데 샤드 헤더에 없는 텐서는
            // 종전 조용히 누락 — 로드 시점에 즉시 거부(원인 추적 가능하게).
            let mut missing: Vec<&str> = entries
                .keys()
                .filter(|n| !out.contains_key(*n))
                .map(String::as_str)
                .collect();
            if !missing.is_empty() {
                missing.sort_unstable();
                return Err(StError::BadHeader(format!(
                    "index.json: 샤드 헤더에 없는 텐서 {}건 {:?}",
                    missing.len(),
                    &missing[..missing.len().min(4)]
                )));
            }
        }
        Ok(Self {
            shards,
            entries: out,
        })
    }

    /// 단일 샤드 헤더 파싱 — (이름 → 엔트리). shard 벡터에 파일 정보 추가.
    fn parse_shard(
        path: &Path,
        shard: usize,
        shards: &mut Vec<Shard>,
    ) -> StResult<HashMap<String, StEntry>> {
        let mut f = std::fs::File::open(path)?;
        let mut lenb = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut lenb)?;
        let hlen = u64::from_le_bytes(lenb);
        // 상한: 16GiB 텐서 × 수천 개보다 넉넉히.
        if hlen == 0 || hlen > (1 << 30) {
            return Err(StError::BadHeader(format!(
                "{}: header len {hlen}",
                path.display()
            )));
        }
        let mut hb = vec![0u8; hlen as usize];
        std::io::Read::read_exact(&mut f, &mut hb)?;
        let data_base = 8 + hlen;
        let data_len = f.metadata()?.len().saturating_sub(data_base);
        let v = Json::parse(std::str::from_utf8(&hb)?)?;
        let obj = v
            .as_object()
            .ok_or_else(|| StError::BadHeader("header not an object".into()))?;
        let mut out = HashMap::new();
        for (name, tv) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = StDtype::parse(
                tv.get("dtype")
                    .and_then(Json::as_str)
                    .ok_or_else(|| StError::BadHeader(format!("{name}: dtype")))?,
            )?;
            let shape = tv
                .get("shape")
                .and_then(Json::as_num_array)
                .ok_or_else(|| StError::BadHeader(format!("{name}: shape")))?;
            let offs = tv
                .get("data_offsets")
                .and_then(Json::as_num_array)
                .ok_or_else(|| StError::BadHeader(format!("{name}: data_offsets")))?;
            if offs.len() != 2 {
                return Err(StError::BadHeader(format!("{name}: offsets len")));
            }
            let numel: u64 = shape.iter().product();
            let Some(expect) = numel.checked_mul(dtype.nbytes()) else {
                return Err(StError::BadHeader(format!("{name}: numel overflow")));
            };
            // H(2026-10-09): 반전 offsets는 u64 감산 언더플로(디버그 패닉) —
            // 여기서 즉시 거부.
            if offs[1] < offs[0] {
                return Err(StError::BadHeader(format!(
                    "{name}: data_offsets 반전 {}..{}",
                    offs[0], offs[1]
                )));
            }
            let nbytes = offs[1] - offs[0];
            if nbytes != expect {
                return Err(StError::BadHeader(format!(
                    "{name}: bytes {nbytes} != numel {numel} × {:?}",
                    dtype.nbytes()
                )));
            }
            out.insert(
                name.clone(),
                StEntry {
                    dtype,
                    shape,
                    begin: offs[0],
                    end: offs[1],
                    shard,
                },
            );
        }
        shards.push(Shard {
            path: path.into(),
            data_base,
            data_len,
        });
        Ok(out)
    }

    pub fn entries(&self) -> &HashMap<String, StEntry> {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&StEntry> {
        self.entries.get(name)
    }

    /// 텐서 원시 바이트를 호출자 버퍼로 직독(대형 상주 적재 — Vec 회피).
    ///
    /// # Safety
    /// `out`은 `cap >= nbytes(name)` 바이트의 쓰기 가능 영역이어야 한다.
    pub unsafe fn read_into(&self, name: &str, out: *mut u8, cap: usize) -> StResult<()> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| StError::TensorNotFound(name.into()))?;
        let n = e.nbytes() as usize;
        if n > cap {
            return Err(StError::BadTensor(format!("{name}: cap {cap} < {n}")));
        }
        let sh = &self.shards[e.shard];
        if e.end > sh.data_len {
            return Err(StError::BadHeader(format!(
                "{name}: offset {} beyond shard data {}/{}",
                e.end,
                sh.path.display(),
                sh.data_len
            )));
        }
        let mut f = std::fs::File::open(&sh.path)?;
        std::io::Seek::seek(&mut f, std::io::SeekFrom::Start(sh.data_base + e.begin))?;
        let mut off = 0usize;
        while off < n {
            // SAFETY: 상단 계약 — cap 내 영역만 기입.
            let w = unsafe {
                std::io::Read::read(
                    &mut f,
                    std::slice::from_raw_parts_mut(out.add(off), (n - off).min(1 << 20)),
                )?
            };
            if w == 0 {
                return Err(StError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "read_into: EOF",
                )));
            }
            off += w;
        }
        Ok(())
    }

    /// 샤드 파일 경로(엔진 mmap용).
    pub fn shard_paths(&self) -> Vec<&Path> {
        self.shards.iter().map(|s| s.path.as_path()).collect()
    }

    /// 샤드 i의 데이터 섹션 절대 오프셋.
    pub fn shard_data_base(&self, i: usize) -> Option<u64> {
        self.shards.get(i).map(|s| s.data_base)
    }

    /// 텐서 원시 바이트를 읽는다(검증·참조 경로 — 엔진 적재는 mmap 별도).
    pub fn read(&self, name: &str) -> StResult<Vec<u8>> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| StError::TensorNotFound(name.into()))?;
        let sh = &self.shards[e.shard];
        let n = e.nbytes() as usize;
        if e.end > sh.data_len {
            return Err(StError::BadHeader(format!(
                "{name}: offset {} beyond shard data {}/{}",
                e.end,
                sh.path.display(),
                sh.data_len
            )));
        }
        let mut f = std::fs::File::open(&sh.path)?;
        std::io::Seek::seek(&mut f, std::io::SeekFrom::Start(sh.data_base + e.begin))?;
        let mut buf = vec![0u8; n];
        std::io::Read::read_exact(&mut f, &mut buf)?;
        Ok(buf)
    }
}
