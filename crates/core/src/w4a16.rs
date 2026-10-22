
use crate::json::Json;
use crate::quant::deq::{bf16_to_f32, half_to_f32 as f16_to_f32};
use crate::st::{StArchive, StDtype};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub enum W4a16Error {
    Missing(String),
    BadTensor(String),
    Quant(String),
    Arch(String),
    Config(String),
    St(crate::st::StError),
    Io(std::io::Error),
}

impl std::fmt::Display for W4a16Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            W4a16Error::Missing(n) => write!(f, "w4a16 missing tensor: {n}"),
            W4a16Error::BadTensor(s) => write!(f, "w4a16 bad tensor: {s}"),
            W4a16Error::Quant(s) => write!(f, "w4a16 unsupported quant: {s}"),
            W4a16Error::Arch(s) => write!(f, "w4a16 unsupported arch: {s}"),
            W4a16Error::Config(s) => write!(f, "w4a16 config: {s}"),
            W4a16Error::St(e) => write!(f, "w4a16 archive: {e}"),
            W4a16Error::Io(e) => write!(f, "w4a16 io: {e}"),
        }
    }
}

impl std::error::Error for W4a16Error {}

impl From<crate::st::StError> for W4a16Error {
    fn from(e: crate::st::StError) -> Self {
        W4a16Error::St(e)
    }
}

impl From<std::io::Error> for W4a16Error {
    fn from(e: std::io::Error) -> Self {
        W4a16Error::Io(e)
    }
}

pub type R<T> = Result<T, W4a16Error>;

#[derive(Debug, Clone, Copy)]
pub struct QuantSpec {
    pub bits: usize,
    pub group: usize,
}

#[derive(Debug, Default)]
pub struct Report {
    pub triples: usize,
    pub plain_text: usize,
    pub visual: usize,
    pub mtp: usize,
    pub missing_quant: Vec<String>,
    pub missing_plain: Vec<String>,
    pub bad_shape_value: Vec<String>,
    pub unknown_text: Vec<String>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.missing_quant.is_empty()
            && self.missing_plain.is_empty()
            && self.bad_shape_value.is_empty()
            && self.unknown_text.is_empty()
    }

    pub fn summary(&self) -> String {
        let mut s = format!(
            "  트리플 {}(전수 정합) · 비양자화 텍스트 {} · visual {} · mtp {}\n",
            self.triples, self.plain_text, self.visual, self.mtp
        );
        for (label, v) in [
            ("missing_quant", &self.missing_quant),
            ("missing_plain", &self.missing_plain),
            ("bad_shape_value", &self.bad_shape_value),
            ("unknown_text", &self.unknown_text),
        ] {
            if !v.is_empty() {
                s.push_str(&format!(
                    "  {label} {}건: {:?}\n",
                    v.len(),
                    &v[..v.len().min(4)]
                ));
            }
        }
        s
    }
}

pub struct W4a16Model {
    ar: StArchive,
    mmaps: Vec<memmap2::Mmap>,
    pub dir: std::path::PathBuf,
    lins: HashMap<String, (usize, usize)>,
    quant: QuantSpec,
}

impl W4a16Model {
    pub fn open(dir: &Path) -> R<Self> {
        let q_txt = match ["quantization_config.json", "quant_config.json"]
            .iter()
            .find_map(|f| std::fs::read_to_string(dir.join(f)).ok())
        {
            Some(t) => t,
            None => {
                let cfg = std::fs::read_to_string(dir.join("config.json"))
                    .map_err(|e| W4a16Error::Missing(format!("config.json: {e}")))?;
                extract_embedded_quant(&cfg)?
            }
        };
        let quant = Self::check_quant(&q_txt)?;
        let ar = StArchive::open(dir)?;

        let mut lins = HashMap::new();
        for name in ar.entries().keys() {
            let Some(base) = name.strip_suffix(".weight_packed") else {
                continue;
            };
            let pk = ar.entry(name).expect("순회 중 엔트리");
            let sc = ar
                .entry(&format!("{base}.weight_scale"))
                .ok_or_else(|| W4a16Error::BadTensor(format!("{base}: weight_scale 부재")))?;
            let shp = ar
                .entry(&format!("{base}.weight_shape"))
                .ok_or_else(|| W4a16Error::BadTensor(format!("{base}: weight_shape 부재")))?;
            if pk.dtype != StDtype::I32 {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: packed {:?}",
                    pk.dtype
                )));
            }
            if sc.dtype != StDtype::F16 && sc.dtype != StDtype::Bf16 {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: scale {:?}",
                    sc.dtype
                )));
            }
            if shp.dtype != StDtype::I64 {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: shape {:?}",
                    shp.dtype
                )));
            }
            if pk.shape.len() != 2 || sc.shape.len() != 2 || shp.shape.as_slice() != [2] {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: 랭크 계약 위반 packed={:?} scale={:?} shape={:?}",
                    pk.shape, sc.shape, shp.shape
                )));
            }
            let n = pk.shape[0] as usize;
            let k = pk.shape[1] as usize * 8;
            if !k.is_multiple_of(quant.group) || n == 0 {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: n={n} k={k} — g{} 정렬 위반",
                    quant.group
                )));
            }
            if sc.shape[0] as usize != n || sc.shape[1] as usize != k / quant.group {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: scale {:?} != [{n}, {}]",
                    sc.shape,
                    k / quant.group
                )));
            }
            lins.insert(base.to_string(), (n, k));
        }
        if lins.is_empty() {
            return Err(W4a16Error::Quant("weight_packed 0건 — W4A16 아님".into()));
        }
        let mut mmaps = Vec::new();
        for p in ar.shard_paths() {
            let f = std::fs::File::open(p)?;
            mmaps.push(unsafe { memmap2::Mmap::map(&f)? });
        }
        Ok(W4a16Model {
            ar,
            mmaps,
            dir: dir.to_path_buf(),
            lins,
            quant,
        })
    }

    fn check_quant(text: &str) -> R<QuantSpec> {
        let v = Json::parse(text)
            .map_err(|e| W4a16Error::Quant(format!("quantization_config.json: {e}")))?;
        let method = v.get("quant_method").and_then(Json::as_str).unwrap_or("");
        if method != "compressed-tensors" {
            return Err(W4a16Error::Quant(format!("quant_method={method:?}")));
        }
        let cgs = v
            .get("config_groups")
            .and_then(Json::as_object)
            .ok_or_else(|| W4a16Error::Quant("config_groups 부재".into()))?;
        if cgs.len() != 1 {
            return Err(W4a16Error::Quant(format!(
                "config_groups {}개 — 단일 스킴만 지원",
                cgs.len()
            )));
        }
        let g0 = &cgs.first().expect("len == 1").1;
        let fmt = g0
            .get("format")
            .and_then(Json::as_str)
            .or_else(|| v.get("format").and_then(Json::as_str))
            .unwrap_or("");
        if fmt != "pack-quantized" {
            return Err(W4a16Error::Quant(format!("format={fmt:?}")));
        }
        let w = g0
            .get("weights")
            .ok_or_else(|| W4a16Error::Quant("group_0.weights 부재".into()))?;
        let bits = w.get("num_bits").and_then(Json::as_f64).unwrap_or(0.0) as usize;
        let group = w.get("group_size").and_then(Json::as_f64).unwrap_or(0.0) as usize;
        let sym = w.get("symmetric").and_then(Json::as_bool).unwrap_or(false);
        if bits != 4 || !sym {
            return Err(W4a16Error::Quant(format!(
                "bits={bits} sym={sym} — int4 sym 전용"
            )));
        }
        if group != 128 && group != 32 && group != 64 {
            return Err(W4a16Error::Quant(format!(
                "group={group} — g32/g64/g128만 지원"
            )));
        }
        Ok(QuantSpec { bits, group })
    }

    pub fn scale_is_bf16(&self, base: &str) -> bool {
        self.entry(&format!("{base}.weight_scale"))
            .is_some_and(|e| e.dtype == crate::st::StDtype::Bf16)
    }

    pub fn expert_at(&self, base: &str) -> Option<(&[u8], &[u8], usize, usize)> {
        let (n, k) = *self.lins.get(base)?;
        let q = self.tensor_slice(&format!("{base}.weight_packed"))?;
        let s = self.tensor_slice(&format!("{base}.weight_scale"))?;
        Some((q, s, n, k))
    }

    pub fn entry(&self, name: &str) -> Option<&crate::st::StEntry> {
        self.ar.entry(name)
    }

    pub fn entries(&self) -> &HashMap<String, crate::st::StEntry> {
        self.ar.entries()
    }

    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let e = self.ar.entry(name)?;
        let base = self.ar.shard_data_base(e.shard)?;
        let m = self.mmaps.get(e.shard)?;
        let s = (base + e.begin) as usize;
        let t = (base + e.end) as usize;
        m.get(s..t)
    }

    pub fn tensor_range(&self, name: &str) -> Option<(usize, u64, u64)> {
        let e = self.ar.entry(name)?;
        let base = self.ar.shard_data_base(e.shard)?;
        Some((e.shard, base + e.begin, e.end - e.begin))
    }

    pub fn shard_count(&self) -> usize {
        self.mmaps.len()
    }

    pub fn shard_paths(&self) -> Vec<&std::path::Path> {
        self.ar.shard_paths()
    }

    pub fn lin_shape(&self, base: &str) -> Option<(usize, usize)> {
        self.lins.get(base).copied()
    }

    pub fn n_lins(&self) -> usize {
        self.lins.len()
    }

    pub fn lin_shapes(&self) -> Vec<(String, usize, usize)> {
        let mut v: Vec<(String, usize, usize)> = self
            .lins
            .iter()
            .map(|(b, &(n, k))| (b.clone(), n, k))
            .collect();
        v.sort();
        v
    }

    pub fn group(&self) -> usize {
        self.quant.group
    }

    pub fn check_shape_values(&self) -> R<Vec<String>> {
        let mut bad = Vec::new();
        for (base, &(n, k)) in &self.lins {
            let raw = self.read_raw(&format!("{base}.weight_shape"), 16, 0, 1)?;
            let a = i64::from_le_bytes(raw[0..8].try_into().expect("8B"));
            let b = i64::from_le_bytes(raw[8..16].try_into().expect("8B"));
            if a != n as i64 || b != k as i64 {
                bad.push(format!("{base}: ({a},{b}) != ({n},{k})"));
            }
        }
        Ok(bad)
    }

    fn read_raw(&self, name: &str, row_bytes: usize, lo: u64, hi: u64) -> R<Vec<u8>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| W4a16Error::Missing(name.into()))?;
        let rows = *e.shape.first().unwrap_or(&0);
        if lo >= hi || hi > rows {
            return Err(W4a16Error::BadTensor(format!(
                "{name}: 행 범위 {lo}..{hi} (rows={rows})"
            )));
        }
        let sh_path = self
            .ar
            .shard_paths()
            .get(e.shard)
            .ok_or_else(|| W4a16Error::BadTensor(format!("{name}: 샤드 {} 부재", e.shard)))?
            .to_path_buf();
        let base = self
            .ar
            .shard_data_base(e.shard)
            .ok_or_else(|| W4a16Error::BadTensor(format!("{name}: 샤드 베이스 부재")))?;
        let mut f = std::fs::File::open(&sh_path)?;
        f.seek(SeekFrom::Start(base + e.begin + lo * row_bytes as u64))?;
        let mut raw = vec![0u8; (hi - lo) as usize * row_bytes];
        f.read_exact(&mut raw)?;
        Ok(raw)
    }

    pub fn plain_rows_f32(&self, name: &str, lo: u64, hi: u64) -> R<Vec<f32>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| W4a16Error::Missing(name.into()))?;
        if e.shape.len() != 2 {
            return Err(W4a16Error::BadTensor(format!("{name}: 2D 아님")));
        }
        let cols = e.shape[1] as usize;
        let nb = e.dtype.nbytes() as usize;
        let raw = self.read_raw(name, cols * nb, lo, hi)?;
        decode_f32(&raw, e.dtype, name)
    }

    pub fn raw_rows(&self, name: &str, lo: u64, hi: u64) -> R<Vec<u8>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| W4a16Error::Missing(name.into()))?;
        let row_bytes: u64 = e.shape[1..].iter().product::<u64>() * e.dtype.nbytes();
        self.read_raw(name, row_bytes as usize, lo, hi)
    }

    pub fn plain_vec_f32(&self, name: &str) -> R<Vec<f32>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| W4a16Error::Missing(name.into()))?;
        if e.shape.len() != 1 {
            return Err(W4a16Error::BadTensor(format!("{name}: 1D 아님")));
        }
        let nb = e.dtype.nbytes() as usize;
        let raw = self.read_raw(name, nb, 0, e.shape[0])?;
        decode_f32(&raw, e.dtype, name)
    }
}

pub(crate) fn decode_f32(raw: &[u8], dt: StDtype, name: &str) -> R<Vec<f32>> {
    Ok(match dt {
        StDtype::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        StDtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        StDtype::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        other => {
            return Err(W4a16Error::BadTensor(format!(
                "{name}: f32 변환 불가 dtype {other:?}"
            )));
        }
    })
}

fn vocab_id(v: &Json, src: &str) -> R<u32> {
    let n = v
        .as_f64()
        .ok_or_else(|| W4a16Error::BadTensor(format!("{src}: id 숫자 아님")))?;
    if !(0.0..=u32::MAX as f64).contains(&n) || n.fract() != 0.0 {
        return Err(W4a16Error::BadTensor(format!(
            "{src}: id 범위/정수 위반 {n}"
        )));
    }
    Ok(n as u32)
}

fn piece_put(map: &mut HashMap<u32, String>, id: u32, piece: &str, src: &str) -> R<()> {
    if let Some(prev) = map.get(&id) {
        if prev != piece {
            return Err(W4a16Error::BadTensor(format!(
                "{src}: id {id} 충돌 — {prev:?} vs {piece:?}"
            )));
        }
        return Ok(());
    }
    map.insert(id, piece.to_string());
    Ok(())
}

pub fn load_pieces(dir: &Path) -> R<Vec<String>> {
    let mut map: HashMap<u32, String> = HashMap::new();
    if dir.join("vocab.json").is_file() {
        let raw = std::fs::read_to_string(dir.join("vocab.json"))
            .map_err(|e| W4a16Error::Missing(format!("vocab.json: {e}")))?;
        let v = Json::parse(&raw).map_err(|e| W4a16Error::BadTensor(format!("vocab.json: {e}")))?;
        let obj = v
            .as_object()
            .ok_or_else(|| W4a16Error::BadTensor("vocab.json: 객체 아님".into()))?;
        for (piece, id) in obj {
            let id = vocab_id(id, "vocab.json")?;
            piece_put(&mut map, id, piece, "vocab.json")?;
        }
    } else {
        let txt = std::fs::read_to_string(dir.join("tokenizer.json"))
            .map_err(|e| W4a16Error::Missing(format!("tokenizer.json: {e}")))?;
        let v =
            Json::parse(&txt).map_err(|e| W4a16Error::BadTensor(format!("tokenizer.json: {e}")))?;
        let vocab = v
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(Json::as_object)
            .ok_or_else(|| W4a16Error::BadTensor("tokenizer.json: model.vocab 부재".into()))?;
        for (piece, id) in vocab {
            let id = vocab_id(id, "tokenizer.json model.vocab")?;
            piece_put(&mut map, id, piece, "tokenizer.json model.vocab")?;
        }
        if let Some(Json::Arr(items)) = v.get("added_tokens") {
            for it in items {
                let id = vocab_id(
                    it.get("id")
                        .ok_or_else(|| W4a16Error::BadTensor("added_tokens: id 부재".into()))?,
                    "added_tokens",
                )?;
                let content = it
                    .get("content")
                    .and_then(Json::as_str)
                    .ok_or_else(|| W4a16Error::BadTensor("added_tokens: content 부재".into()))?;
                piece_put(&mut map, id, content, "added_tokens")?;
            }
        }
    }
    let max = map
        .keys()
        .copied()
        .max()
        .ok_or_else(|| W4a16Error::BadTensor("조각표 비었음".into()))?;
    let mut slots: Vec<Option<String>> = vec![None; max as usize + 1];
    for (i, p) in map {
        slots[i as usize] = Some(p);
    }
    if let Some(i) = slots.iter().position(Option::is_none) {
        return Err(W4a16Error::BadTensor(format!(
            "조각표 스파스 — id {i} 누락"
        )));
    }
    Ok(slots.into_iter().map(Option::unwrap).collect())
}

fn extract_embedded_quant(cfg: &str) -> R<String> {
    let key = "\"quantization_config\"";
    let i = cfg
        .find(key)
        .ok_or_else(|| W4a16Error::Quant("quantization_config 부재(사이드카·내장 모두)".into()))?;
    let rest = &cfg[i + key.len()..];
    let start = rest
        .find('{')
        .ok_or_else(|| W4a16Error::Quant("quantization_config 값이 객체 아님".into()))?;
    let b = rest.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    let mut j = start;
    while j < b.len() {
        let c = b[j];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b'{' {
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                return Ok(rest[start..=j].to_string());
            }
        }
        j += 1;
    }
    Err(W4a16Error::Quant("quantization_config 객체 미종결".into()))
}


struct AlignedBuf {
    buf: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let buf = vec![0u8; len + align];
        let rem = buf.as_ptr() as usize % align;
        let off = if rem == 0 { align } else { align - rem };
        Self { buf, off, len }
    }

    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.off..self.off + self.len]
    }
}

const O_DIRECT: i32 = 0o40000;

static FALLBACK_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub struct DirectReader {
    file: std::fs::File,
    align: usize,
    win: AlignedBuf,
    file_size: u64,
    pub direct: bool,
    pub bytes_read: u64,
}

impl DirectReader {
    pub fn open(path: &std::path::Path) -> R<Self> {
        Self::open_with_window(path, 8 << 20)
    }

    pub fn open_with_window(path: &std::path::Path, win: usize) -> R<Self> {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        let meta = std::fs::metadata(path)?;
        let align = (meta.blksize() as usize)
            .next_power_of_two()
            .clamp(4096, 1 << 20);
        let win_len = win.max(align).div_ceil(align) * align;
        let oo = |flags: i32| {
            let mut o = std::fs::OpenOptions::new();
            o.read(true).custom_flags(flags);
            o.open(path)
        };
        let (file, direct) = match oo(O_DIRECT) {
            Ok(f) => (f, true),
            Err(e) => {
                if !FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "# load: O_DIRECT 불가({e}) — 버퍼 읽기 폴백(페이지 캐시 잔류): {} (이후 생략)",
                        path.display()
                    );
                }
                (oo(0)?, false)
            }
        };
        Ok(Self {
            file,
            align,
            win: AlignedBuf::new(win_len, align),
            file_size: meta.len(),
            direct,
            bytes_read: 0,
        })
    }

    pub fn align(&self) -> usize {
        self.align
    }

    pub fn read_into(&mut self, off: u64, out: &mut [u8]) -> R<()> {
        use std::os::unix::fs::FileExt;
        let len = out.len();
        if len == 0 {
            return Ok(());
        }
        let align = self.align as u64;
        let win_len = self.win.len as u64;
        let mut done = 0usize;
        while done < len {
            let cur = off + done as u64;
            let wstart = cur / align * align;
            let remaining = (len - done) as u64;
            let wend = (cur + remaining)
                .next_multiple_of(align)
                .min(wstart + win_len);
            let take = wend - wstart;
            let to = (cur + remaining).min(wend);
            let need = (to - cur) as usize;
            let from = (cur - wstart) as usize;
            let buf = self.win.slice_mut();
            let want = take as usize;
            let mut got = 0usize;
            while got < from + need {
                if got > 0 && self.direct && !got.is_multiple_of(self.align) {
                    break;
                }
                let g = self
                    .file
                    .read_at(&mut buf[got..want], wstart + got as u64)
                    .map_err(W4a16Error::Io)?;
                if g == 0 {
                    break;
                }
                got += g;
            }
            if got < from + need {
                return Err(W4a16Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "O_DIRECT 읽기 부족: off={wstart} want={want} got={got} \
                         need={} (파일 {})",
                        from + need,
                        self.file_size
                    ),
                )));
            }
            out[done..done + need].copy_from_slice(&buf[from..from + need]);
            done += need;
        }
        self.bytes_read += len as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qtext(bits: usize, sym: bool, group: usize) -> String {
        format!(
            r#"{{"quant_method":"compressed-tensors","config_groups":{{"group_0":{{"format":"pack-quantized","weights":{{"num_bits":{bits},"symmetric":{sym},"group_size":{group}}}}}}}}}"#
        )
    }

    #[test]
    fn quant_g128_accept() {
        let q = W4a16Model::check_quant(&qtext(4, true, 128)).expect("g128 계약");
        assert_eq!(q.group, 128);
    }

    #[test]
    fn quant_g32_g64_accept_other_groups_reject() {
        let q = W4a16Model::check_quant(&qtext(4, true, 32)).expect("g32 계약");
        assert_eq!(q.group, 32);
        let q = W4a16Model::check_quant(&qtext(4, true, 64)).expect("g64 계약");
        assert_eq!(q.group, 64);
        let e = W4a16Model::check_quant(&qtext(4, true, 16))
            .expect_err("g16 거부")
            .to_string();
        assert!(e.contains("g32/g64/g128"), "{e}");
    }

    #[test]
    fn quant_group_key_name_agnostic_single_scheme() {
        let alt = r#"{"quant_method":"compressed-tensors","config_groups":{"config_group_0":{"format":"pack-quantized","weights":{"num_bits":4,"symmetric":true,"group_size":32}}}}"#;
        let q = W4a16Model::check_quant(alt).expect("config_group_0 수용");
        assert_eq!(q.group, 32);
        let multi = r#"{"quant_method":"compressed-tensors","config_groups":{"a":{"format":"pack-quantized","weights":{"num_bits":4,"symmetric":true,"group_size":32}},"b":{"format":"pack-quantized","weights":{"num_bits":4,"symmetric":true,"group_size":32}}}}"#;
        let e = W4a16Model::check_quant(multi)
            .expect_err("다중 스킴 거부")
            .to_string();
        assert!(e.contains("단일 스킴"), "{e}");
    }

    #[test]
    fn quant_bits_sym_reject() {
        assert!(W4a16Model::check_quant(&qtext(8, true, 128)).is_err());
        assert!(W4a16Model::check_quant(&qtext(4, false, 128)).is_err());
    }

    #[test]
    fn quant_malformed_rejects() {
        assert!(W4a16Model::check_quant("{}").is_err());
        assert!(W4a16Model::check_quant(r#"{"quant_method":"rtn"}"#).is_err());
        assert!(W4a16Model::check_quant("not json").is_err());
    }

    #[test]
    fn load_pieces_rejects_dup_and_sparse() {
        let dir = std::env::temp_dir().join(format!("llm170_pieces_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":0}"#).unwrap();
        assert!(load_pieces(&dir).is_err(), "중복 id 거부");
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"c":2}"#).unwrap();
        assert!(load_pieces(&dir).is_err(), "스파스 거부");
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":1}"#).unwrap();
        let p = load_pieces(&dir).unwrap();
        assert_eq!(p, vec!["a".to_string(), "b".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn direct_test_path(tag: &str) -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        if std::fs::create_dir_all(&dir).is_ok() {
            return dir.join(format!("{tag}_{}", std::process::id()));
        }
        std::env::temp_dir().join(format!("{tag}_{}", std::process::id()))
    }

    #[test]
    fn direct_reader_boundaries() {
        let path = direct_test_path("llm170_direct");
        let size = 300 * 1024 + 123;
        let data: Vec<u8> = (0..size).map(|i| (i as u64 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();

        let mut r = DirectReader::open_with_window(&path, 64 << 10).unwrap();
        assert_eq!(r.align() % 4096, 0, "정렬 하한 4096");
        assert!(r.align().is_power_of_two());
        if !r.direct {
            eprintln!(
                " O_DIRECT 미지원 FS — 폴백(버퍼) 경로만 검증: {}",
                path.display()
            );
        }

        let check = |r: &mut DirectReader, off: usize, len: usize| {
            let mut out = vec![0u8; len];
            r.read_into(off as u64, &mut out).unwrap();
            assert_eq!(&out[..], &data[off..off + len], "off={off} len={len}");
        };
        check(&mut r, 0, 64 << 10);
        check(&mut r, 1234, 5000);
        check(&mut r, 7, 200_000);
        check(&mut r, size - 1000, 1000);
        check(&mut r, size, 0);
        let mut out = vec![0u8; 200];
        let e = r.read_into((size - 100) as u64, &mut out).unwrap_err();
        assert!(format!("{e}").contains("읽기 부족"), "{e}");
        assert_eq!(r.bytes_read, (64 << 10) + 5000 + 200_000 + 1000);

        let empty = direct_test_path("llm170_direct_empty");
        std::fs::write(&empty, b"").unwrap();
        let mut r2 = DirectReader::open(&empty).unwrap();
        let mut o = [0u8; 16];
        assert!(r2.read_into(0, &mut o).is_err());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&empty);
    }
}
