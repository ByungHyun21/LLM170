//! W4A16 safetensors 스토어 — compressed-tensors pack-quantized(int4 sym).
//! (2026-10-08 R4: **아키텍처 무지** — 이름맵·synth·V헤드 순열·기대 커버리지는
//! 아키텍처 바인딩 모듈(`qwen35/bind.rs`) 소관. 이 파일은 형식만 안다.)
//!
//! [스키마] 양자화 선형 1개 = 3조: `.weight_packed` I32[n, k/8] ·
//! `.weight_scale` F16[n, k/group] · `.weight_shape` I64[2]=(n,k). 행=출력(n),
//! 열=입력(k). 대칭(sym)이라 zero-point 미저장 — zp=8 고정(4bit 중심).
//! 니블 순서 lsb-first **확정**(2026-10-08 — lane.rs §3.6).
//! group은 quantization_config에서 읽는다 — **현행 g128 전용**(C3: 스토어·lane·
//! 커널·bind 순열이 전부 g128 가정이라 g32는 check_quant가 명시 거부. g32 정식
//! 지원은 W4-1에서 네 지점 동시 일반화).
//!
//! [계약] 헤더 인덱스(StArchive) + 행 단위 pread — 전체 적재 금지.
//! 트리플 구조 정합은 open이, 기대 집합 대조는 바인딩의 validate가 판정한다.

use crate::json::Json;
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
    /// config.json 시맨틱 위반(교차필드·정렬 계약) — BadTensor 남용 분리.
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

/// 양자화 파라미터(quantization_config 실측 — compressed-tensors pack-quantized).
#[derive(Debug, Clone, Copy)]
pub struct QuantSpec {
    pub bits: usize,
    /// 그룹 크기 — 현행 128 고정(g32는 W4-1 — check_quant 게이트).
    pub group: usize,
}

/// 전수 검증 리포트 — 바인딩 validate가 채우고 ok()가 판정.
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

/// f16(Half) → f32 — core는 half 크레이트에 의존하지 않는다.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let e = ((bits >> 10) & 0x1F) as u32;
    let m = (bits & 0x3FF) as u32;
    if e == 0 {
        // 비정규: m/1024·2^-14.
        return ((m as f32) * (2.0f64).powi(-24) as f32).copysign(f32::from_bits(sign));
    }
    if e == 31 {
        return f32::NAN;
    }
    f32::from_bits(sign | ((e - 15 + 127) << 23) | (m << 13))
}

/// bf16 → f32 (비트 확장, 정확).
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// W4A16 스토어 — 헤더 인덱스 + 양자화 선형 사전 + 샤드 mmap.
pub struct W4a16Model {
    ar: StArchive,
    /// 샤드 mmap — 엔진 직접 로드의 무카피 Weight 슬라이스용.
    mmaps: Vec<memmap2::Mmap>,
    /// 원본 디렉터리(vocab·tokenizer·config — 바인딩/토크나이저 공용).
    pub dir: std::path::PathBuf,
    /// HF 텐서명(접미사 제외) → (n, k).
    lins: HashMap<String, (usize, usize)>,
    quant: QuantSpec,
}

impl W4a16Model {
    /// 디렉터리 열기 — quantization_config 검증 + 트리플 구조 검사 + mmap.
    pub fn open(dir: &Path) -> R<Self> {
        let q_txt = std::fs::read_to_string(dir.join("quantization_config.json"))
            .map_err(|e| W4a16Error::Missing(format!("quantization_config.json: {e}")))?;
        let quant = Self::check_quant(&q_txt)?;
        let ar = StArchive::open(dir)?;

        // 트리플 구조 검사: packed 각각에 scale/shape가 있고 형상 계약이 맞는가.
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
            if sc.dtype != StDtype::F16 {
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
        // 엔진 직접 로드용 샤드 mmap(가상 매핑 — 물리 페이지는 접근 시).
        let mut mmaps = Vec::new();
        for p in ar.shard_paths() {
            let f = std::fs::File::open(p)?;
            // SAFETY: 읽기 전용 매핑 — 수정하지 않는다(qwen35 mmap 동일 계약).
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

    /// quantization_config 계약: compressed-tensors pack-quantized int4 sym g128.
    /// g32는 스토어·lane·커널·bind가 전부 g128 가정이라 명시 거부(패닉 예방 —
    /// W4-1에서 일반화 예정).
    fn check_quant(text: &str) -> R<QuantSpec> {
        let v = Json::parse(text)
            .map_err(|e| W4a16Error::Quant(format!("quantization_config.json: {e}")))?;
        let method = v.get("quant_method").and_then(Json::as_str).unwrap_or("");
        if method != "compressed-tensors" {
            return Err(W4a16Error::Quant(format!("quant_method={method:?}")));
        }
        let g0 = v
            .get("config_groups")
            .and_then(|x| x.get("group_0"))
            .ok_or_else(|| W4a16Error::Quant("config_groups.group_0 부재".into()))?;
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
        if group != 128 {
            return Err(W4a16Error::Quant(format!(
                "group={group} — 현행 g128 전용(g32는 W4-1에서 스토어·lane·커널·bind 동시 일반화)"
            )));
        }
        Ok(QuantSpec { bits, group })
    }

    /// 헤더 엔트리 조회(서빙 경로 — Weight 슬라이스 형상 판독).
    pub fn entry(&self, name: &str) -> Option<&crate::st::StEntry> {
        self.ar.entry(name)
    }

    /// 전체 텐서 이름 → 엔트리(커버리지 분류 — 바인딩 validate가 소비).
    pub fn entries(&self) -> &HashMap<String, crate::st::StEntry> {
        self.ar.entries()
    }

    /// 텐서 원시 슬라이스(mmap) — 엔진 직접 로드 무카피 경로.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let e = self.ar.entry(name)?;
        let base = self.ar.shard_data_base(e.shard)?;
        let m = self.mmaps.get(e.shard)?;
        let s = (base + e.begin) as usize;
        let t = (base + e.end) as usize;
        m.get(s..t)
    }

    /// 양자화 선형 형상 (n, k) — base는 HF 텐서명(접미사 제외).
    pub fn lin_shape(&self, base: &str) -> Option<(usize, usize)> {
        self.lins.get(base).copied()
    }

    /// 양자화 선형 수.
    pub fn n_lins(&self) -> usize {
        self.lins.len()
    }

    /// 양자화 선형 전수 (HF base, n, k) — 프로브 형상 자동 열거용.
    pub fn lin_shapes(&self) -> Vec<(String, usize, usize)> {
        let mut v: Vec<(String, usize, usize)> = self
            .lins
            .iter()
            .map(|(b, &(n, k))| (b.clone(), n, k))
            .collect();
        v.sort();
        v
    }

    /// 그룹 크기(현행 g128) — 커널 계약의 입력.
    pub fn group(&self) -> usize {
        self.quant.group
    }

    /// weight_shape 실값 대조(트리플 전수 — 행 1회 16B pread) → 불일치 목록.
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

    /// 원시 행 범위 pread — row_bytes·행 번호 계약은 호출자 검증.
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

    /// 플레인 행 범위 → f32(2D 텐서, F32/BF16/F16).
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

    /// 플레인 임의 행 범위 원시 바이트 — 행 = shape[0], 나머지 축이 한 행.
    pub fn raw_rows(&self, name: &str, lo: u64, hi: u64) -> R<Vec<u8>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| W4a16Error::Missing(name.into()))?;
        let row_bytes: u64 = e.shape[1..].iter().product::<u64>() * e.dtype.nbytes();
        self.read_raw(name, row_bytes as usize, lo, hi)
    }

    /// 1D 플레인 → f32(F32/BF16/F16).
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

/// 원시 바이트 → f32(F32/BF16/F16).
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

/// 토큰 조각표(id 순) — vocab.json(디렉터리 규약) 우선, 없으면 tokenizer.json
/// (model.vocab + added_tokens 병합 — W4A16 HF 배포는 vocab.json 부재 실측).
/// 아키텍처 무관 — 로더/바인딩 공용.
pub fn load_pieces(dir: &Path) -> R<Vec<String>> {
    if dir.join("vocab.json").is_file() {
        // vocab.json {"piece": id} — id 순 조각표(HF 벌크 배포 규약).
        let raw = std::fs::read_to_string(dir.join("vocab.json"))
            .map_err(|e| W4a16Error::Missing(format!("vocab.json: {e}")))?;
        let v = Json::parse(&raw).map_err(|e| W4a16Error::BadTensor(format!("vocab.json: {e}")))?;
        let obj = v
            .as_object()
            .ok_or_else(|| W4a16Error::BadTensor("vocab.json: 객체 아님".into()))?;
        let mut pairs: Vec<(u32, String)> = obj
            .iter()
            .filter_map(|(piece, id)| {
                id.as_f64()
                    .filter(|n| *n >= 0.0 && n.fract() == 0.0)
                    .map(|n| (n as u32, piece.clone()))
            })
            .collect();
        pairs.sort_by_key(|(id, _)| *id);
        return Ok(pairs.into_iter().map(|(_, p)| p).collect());
    }
    let txt = std::fs::read_to_string(dir.join("tokenizer.json"))
        .map_err(|e| W4a16Error::Missing(format!("tokenizer.json: {e}")))?;
    let v = Json::parse(&txt).map_err(|e| W4a16Error::BadTensor(format!("tokenizer.json: {e}")))?;
    let mut map: HashMap<u32, String> = HashMap::new();
    let vocab = v
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(Json::as_object)
        .ok_or_else(|| W4a16Error::BadTensor("tokenizer.json: model.vocab 부재".into()))?;
    for (piece, id) in vocab {
        if let Some(n) = id.as_f64() {
            map.insert(n as u32, piece.clone());
        }
    }
    if let Some(Json::Arr(items)) = v.get("added_tokens") {
        for it in items {
            let id = it.get("id").and_then(Json::as_f64).map(|x| x as u32);
            let content = it.get("content").and_then(Json::as_str).map(String::from);
            if let (Some(i), Some(c)) = (id, content) {
                map.insert(i, c);
            }
        }
    }
    let max = map.keys().copied().max().unwrap_or(0);
    let mut out = vec![String::new(); max as usize + 1];
    for (i, p) in map {
        out[i as usize] = p;
    }
    Ok(out)
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
    fn quant_g32_reject_until_w4() {
        // C3: g32는 로드 중 패닉(커널/lane 가정)을 내므로 파싱 시점 명시 거부.
        let e = W4a16Model::check_quant(&qtext(4, true, 32))
            .expect_err("g32 거부")
            .to_string();
        assert!(e.contains("g128 전용"), "{e}");
    }

    #[test]
    fn quant_bits_sym_reject() {
        assert!(W4a16Model::check_quant(&qtext(8, true, 128)).is_err());
        assert!(W4a16Model::check_quant(&qtext(4, false, 128)).is_err());
    }
}
