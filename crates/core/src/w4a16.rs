//! W4A16 safetensors 로더 — compressed-tensors pack-quantized(int4 sym g128).
//! (plans/cuda-models.md §3.5 W4A16 계열, 2026-10-08 — "3계열 모두 로드" 목표)
//!
//! [실측 스키마 — ../models/Qwen3.8-27B-W4A16-AutoRound]
//! - 양자화 선형 1개 = 3조: `.weight_packed` I32[n, k/8] · `.weight_scale`
//!   F16[n, k/128] · `.weight_shape` I64[2]=(n,k). 행=출력(n), 열=입력(k).
//! - 대칭(sym)이라 zero-point 미저장 → zp=8 고정(4bit 중심 — lane 미러 계약).
//! - 니블 순서 lsb-first **확정**(2026-10-08, w4a16-xcheck — 동일 기저
//!   27B GGUF 대조 corr(lsb) 0.991~0.994 vs corr(msb) ≈0.01; lane.rs §3.6).
//! - 비양자화: BF16 — embed_tokens·lm_head·norm·conv1d·A_log·dt_bias·
//!   in_proj_a/b, mtp 15종, visual 333종(텍스트 서빙 무사용 — 커버리지에서만 집계).
//! - 텐서명 `model.language_model.layers.{il}.*`(HF 원본), 전역은
//!   `model.language_model.{embed_tokens,norm}.weight`·`lm_head.weight`.
//!
//! [로더 계약] 헤더 인덱스(StArchive) + 행 단위 pread — 전체 적재 금지
//! (deepseek4 loader 선례). validate()가 전수 커버리지를 판정한다: 기대
//! 집합 = 64층 × 종류별(linear 6종/full 7종 양자화 + 8/4종 비양자화) +
//! 전역 3 — 실측 400 트리플·텍스트 451 플레인과 정확히 일치.
//!
//! [미지원 변이] 본 로더는 g128 sym pack-quantized 전용이다. g32(35B 전문가)·
//! auto-gptq 네이티브(qweight/qzeros — FN FP8PLE)·비대칭은 별도 확장 과제
//! (§3.5 패킹 2계열 — 로더 확장 시 이 파일에 계약을 추가한다).

use llm170_exl3::{Json, StArchive, StDtype};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// 그룹 크기 계약(quantization_config 실측 — g128).
pub const GROUP: usize = 128;
/// 대칭 4bit 중심 zero-point(미저장 — 상수 공급).
pub const ZP_SYM: u32 = 8;

#[derive(Debug)]
pub enum W4a16Error {
    Missing(String),
    BadTensor(String),
    Quant(String),
    Exl3(llm170_exl3::Exl3Error),
    Io(std::io::Error),
}

impl std::fmt::Display for W4a16Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            W4a16Error::Missing(n) => write!(f, "w4a16 missing tensor: {n}"),
            W4a16Error::BadTensor(s) => write!(f, "w4a16 bad tensor: {s}"),
            W4a16Error::Quant(s) => write!(f, "w4a16 unsupported quant: {s}"),
            W4a16Error::Exl3(e) => write!(f, "w4a16 archive: {e}"),
            W4a16Error::Io(e) => write!(f, "w4a16 io: {e}"),
        }
    }
}

impl std::error::Error for W4a16Error {}

impl From<llm170_exl3::Exl3Error> for W4a16Error {
    fn from(e: llm170_exl3::Exl3Error) -> Self {
        W4a16Error::Exl3(e)
    }
}

impl From<std::io::Error> for W4a16Error {
    fn from(e: std::io::Error) -> Self {
        W4a16Error::Io(e)
    }
}

pub type R<T> = Result<T, W4a16Error>;

/// 아키텍처 메타(config.json 발췌 — 로더 검증용 최소 집합).
#[derive(Debug, Clone)]
pub struct W4a16Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// full-attention 간격(4 → full il = {3,7,…,63}).
    pub full_interval: usize,
    pub group_size: usize,
    pub bits: usize,
}

/// 전수 검증 리포트 — ok()가 커버리지·정합 판정.
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

/// f16(Half) → f32 — core는 half 크레이트에 의존하지 않는다(deepseek4 동형).
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

/// W4A16 모델 — 헤더 인덱스 + 양자화 선형 사전.
pub struct W4a16Model {
    ar: StArchive,
    pub cfg: W4a16Config,
    /// `model.language_model.layers.{il}.mlp.gate_proj` → (n, k).
    lins: HashMap<String, (usize, usize)>,
}

impl W4a16Model {
    /// 디렉터리 열기 — config/quantization_config 검증 + 트리플 구조 검사.
    pub fn open(dir: &Path) -> R<Self> {
        let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| W4a16Error::Missing(format!("config.json: {e}")))?;
        let q_txt = std::fs::read_to_string(dir.join("quantization_config.json"))
            .map_err(|e| W4a16Error::Missing(format!("quantization_config.json: {e}")))?;
        let cfg = Self::parse_config(&cfg_txt)?;
        Self::check_quant(&q_txt)?;
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
            if !k.is_multiple_of(GROUP) || n == 0 {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: n={n} k={k} — g{GROUP} 정렬 위반"
                )));
            }
            if sc.shape[0] as usize != n || sc.shape[1] as usize != k / GROUP {
                return Err(W4a16Error::BadTensor(format!(
                    "{base}: scale {:?} != [{n}, {}]",
                    sc.shape,
                    k / GROUP
                )));
            }
            lins.insert(base.to_string(), (n, k));
        }
        if lins.is_empty() {
            return Err(W4a16Error::Quant("weight_packed 0건 — W4A16 아님".into()));
        }
        Ok(W4a16Model { ar, cfg, lins })
    }

    fn parse_config(text: &str) -> R<W4a16Config> {
        let v =
            Json::parse(text).map_err(|e| W4a16Error::BadTensor(format!("config.json: {e}")))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| -> Option<usize> { tc.get(k).and_then(Json::as_f64).map(|x| x as usize) };
        let bad = |k: &str| W4a16Error::BadTensor(format!("config.json: {k} 부재"));
        Ok(W4a16Config {
            hidden: u("hidden_size").ok_or_else(|| bad("hidden_size"))?,
            layers: u("num_hidden_layers").ok_or_else(|| bad("num_hidden_layers"))?,
            heads: u("num_attention_heads").ok_or_else(|| bad("num_attention_heads"))?,
            kv_heads: u("num_key_value_heads").ok_or_else(|| bad("num_key_value_heads"))?,
            head_dim: u("head_dim").ok_or_else(|| bad("head_dim"))?,
            ffn: u("intermediate_size").ok_or_else(|| bad("intermediate_size"))?,
            vocab: u("vocab_size").ok_or_else(|| bad("vocab_size"))?,
            full_interval: u("full_attention_interval").unwrap_or(4).max(1),
            group_size: GROUP,
            bits: 4,
        })
    }

    /// quantization_config 계약: compressed-tensors pack-quantized int4 sym g128.
    fn check_quant(text: &str) -> R<()> {
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
        let gs = w.get("group_size").and_then(Json::as_f64).unwrap_or(0.0) as usize;
        let sym = w.get("symmetric").and_then(Json::as_bool).unwrap_or(false);
        if bits != 4 || gs != GROUP || !sym {
            return Err(W4a16Error::Quant(format!(
                "bits={bits} group={gs} sym={sym} — int4 sym g{GROUP} 전용"
            )));
        }
        Ok(())
    }

    /// 양자화 선형 형상 (n, k) — base는 HF 텐서명(접미사 제외).
    pub fn lin_shape(&self, base: &str) -> Option<(usize, usize)> {
        self.lins.get(base).copied()
    }

    /// 양자화 선형 수.
    pub fn n_lins(&self) -> usize {
        self.lins.len()
    }

    /// 전수 검증 — weight_shape 값 대조 + 커버리지(기대 집합 전수).
    pub fn validate(&self) -> R<Report> {
        let mut rep = Report {
            triples: self.lins.len(),
            ..Default::default()
        };
        // weight_shape 실값 = (n, k) 대조(트리플 전수 — 행 1회 16B pread).
        for (base, &(n, k)) in &self.lins {
            let raw = self.read_raw(&format!("{base}.weight_shape"), 16, 0, 1)?;
            let a = i64::from_le_bytes(raw[0..8].try_into().expect("8B"));
            let b = i64::from_le_bytes(raw[8..16].try_into().expect("8B"));
            if a != n as i64 || b != k as i64 {
                rep.bad_shape_value
                    .push(format!("{base}: ({a},{b}) != ({n},{k})"));
            }
        }
        // 커버리지: 기대 양자화/플레인 집합 vs 실제.
        let (exp_q, exp_p) = self.expected_names();
        for base in &exp_q {
            if !self.lins.contains_key(base) {
                rep.missing_quant.push(base.clone());
            }
        }
        for name in &exp_p {
            if self.ar.entry(name).is_none() {
                rep.missing_plain.push(name.clone());
            }
        }
        // 분류: visual/mtp/텍스트 플레인/트리플 파트.
        let triple_part = |n: &str| {
            n.ends_with(".weight_packed")
                || n.ends_with(".weight_scale")
                || n.ends_with(".weight_shape")
        };
        for name in self.ar.entries().keys() {
            if name.starts_with("model.visual.") {
                // visual은 무양자화 BF16/F32(BF16 실측) — 텍스트 경로 무사용.
                rep.visual += 1;
                continue;
            }
            if name.starts_with("mtp.") {
                rep.mtp += 1;
                continue;
            }
            if triple_part(name) {
                continue;
            }
            if exp_p.contains(name) {
                rep.plain_text += 1;
            } else {
                // 기대 밖 텍스트 텐서 — 정체불명(텍스트 경로에 bias 부재 실측).
                rep.unknown_text.push(name.clone());
            }
        }
        Ok(rep)
    }

    /// 기대 텐서 집합 — (양자화 base, 플레인 이름). §3.5 실측 스키마의 전수.
    fn expected_names(&self) -> (Vec<String>, Vec<String>) {
        let c = &self.cfg;
        let (mut q, mut p) = (Vec::new(), Vec::new());
        let lp = "model.language_model.layers.";
        for il in 0..c.layers {
            let full = (il + 1).is_multiple_of(c.full_interval);
            let pre = format!("{lp}{il}.");
            if full {
                for m in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ] {
                    q.push(format!("{pre}{m}"));
                }
                for m in [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                ] {
                    p.push(format!("{pre}{m}"));
                }
            } else {
                for m in [
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.out_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ] {
                    q.push(format!("{pre}{m}"));
                }
                for m in [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "linear_attn.norm.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.conv1d.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ] {
                    p.push(format!("{pre}{m}"));
                }
            }
        }
        for g in [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        ] {
            p.push(g.to_string());
        }
        (q, p)
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

    /// 양자화 행 범위 → packed u32(행당 k/8 워드).
    pub fn packed_rows_u32(&self, base: &str, lo: u64, hi: u64) -> R<Vec<u32>> {
        let (_, k) = *self
            .lins
            .get(base)
            .ok_or_else(|| W4a16Error::Missing(format!("{base}.weight_packed")))?;
        let raw = self.read_raw(&format!("{base}.weight_packed"), (k / 8) * 4, lo, hi)?;
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }

    /// 양자화 행 범위 → scale u16(행당 k/128 그룹).
    pub fn scale_rows_u16(&self, base: &str, lo: u64, hi: u64) -> R<Vec<u16>> {
        let (_, k) = *self
            .lins
            .get(base)
            .ok_or_else(|| W4a16Error::Missing(format!("{base}.weight_scale")))?;
        let raw = self.read_raw(&format!("{base}.weight_scale"), (k / GROUP) * 2, lo, hi)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
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
        Ok(match e.dtype {
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
                    "{name}: 플레인 f32 변환 불가 dtype {other:?}"
                )));
            }
        })
    }

    /// 행 1개 W4A16 내적 — 활성 x는 f16 비트(u16, lane 미러 계약).
    /// zp=8 상수 공급(sym), scale은 F16 그룹값.
    pub fn dot_row_f16x(&self, base: &str, row: u64, x: &[u16]) -> R<f32> {
        let (_, k) = *self
            .lins
            .get(base)
            .ok_or_else(|| W4a16Error::Missing(base.into()))?;
        if x.len() != k {
            return Err(W4a16Error::BadTensor(format!(
                "{base}: x.len={} != k={k}",
                x.len()
            )));
        }
        let q = self.packed_rows_u32(base, row, row + 1)?;
        let s = self.scale_rows_u16(base, row, row + 1)?;
        let z = vec![ZP_SYM; k / GROUP];
        Ok(crate::quant::dot_row_w4a16_lane(&q, &z, &s, x))
    }
}
