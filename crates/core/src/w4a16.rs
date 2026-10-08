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

/// 아키텍처 메타(config.json 발췌 — 로더·변환기 공용).
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
    /// GDN — state_size·key_heads·value_heads·conv 커널(변환기 KV용).
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
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
    /// 원본 디렉터리(vocab.json·generation_config.json — 변환기용).
    pub dir: std::path::PathBuf,
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
        Ok(W4a16Model {
            ar,
            cfg,
            dir: dir.to_path_buf(),
            lins,
        })
    }

    fn parse_config(text: &str) -> R<W4a16Config> {
        let v =
            Json::parse(text).map_err(|e| W4a16Error::BadTensor(format!("config.json: {e}")))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| -> Option<usize> { tc.get(k).and_then(Json::as_f64).map(|x| x as usize) };
        let bad = |k: &str| W4a16Error::BadTensor(format!("config.json: {k} 부재"));
        // rope 파라미터는 text_config.rope_parameters에 중첩(실측) — 평면 키도 허용.
        let rp = tc.get("rope_parameters").unwrap_or(tc);
        let f = |k: &str| -> Option<f64> { rp.get(k).and_then(Json::as_f64) };
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
            linear_key_head_dim: u("linear_key_head_dim")
                .ok_or_else(|| bad("linear_key_head_dim"))?,
            linear_num_key_heads: u("linear_num_key_heads")
                .ok_or_else(|| bad("linear_num_key_heads"))?,
            linear_num_value_heads: u("linear_num_value_heads")
                .ok_or_else(|| bad("linear_num_value_heads"))?,
            linear_conv_kernel: u("linear_conv_kernel_dim").unwrap_or(4),
            partial_rotary_factor: f("partial_rotary_factor")
                .or_else(|| tc.get("partial_rotary_factor").and_then(Json::as_f64))
                .unwrap_or(0.25),
            rope_theta: f("rope_theta")
                .or_else(|| tc.get("rope_theta").and_then(Json::as_f64))
                .unwrap_or(1e7),
            rms_norm_eps: tc
                .get("rms_norm_eps")
                .and_then(Json::as_f64)
                .unwrap_or(1e-6),
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

    /// GDN V헤드 순열 — llama.cpp(subhead-major) ↔ HF(group-major):
    /// gguf 블록 i ← 원본 블록 ratio·(i%nk) + i/nk (exl3 convert.rs 실측 확정
    /// — beta 지문 corr 1.000·ssm_out 블록 corr 0.999, 동일 규약 미러).
    fn vperm(&self, i: usize) -> usize {
        let nk = self.cfg.linear_num_key_heads;
        let ratio = self.cfg.linear_num_value_heads / nk;
        ratio * (i % nk) + i / nk
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
fn decode_f32(raw: &[u8], dt: StDtype, name: &str) -> R<Vec<f32>> {
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

/// 변환 통계 — w4a16-to-gguf.
pub struct ConvertStats {
    pub tensors: usize,
    pub bytes: u64,
    pub elapsed_s: f64,
}

/// 변환 플랜 — exl3 convert.rs의 qwen35 계약 미러(HF→GGUF). W4A16은 원본
/// HF 레이아웃이므로 EXL3의 norm w−1 보정은 없다(무보정 F32).
enum WPlan {
    /// 양자화 선형 → W4A16G128(66B 블록). n_base/k_base: V순열 시작 축
    /// (usize::MAX=무순열) — 순열은 128(=1헤드) 단위.
    Quant {
        base: String,
        n: usize,
        k: usize,
        n_base: usize,
        k_base: usize,
    },
    /// 2D BF16 직접(embed/lm_head — 행우선 그대로, ne만 [cols, rows]).
    Bf16Direct { name: String, rows: u64, cols: u64 },
    /// 2D BF16 행 V순열(alpha/beta).
    Bf16Rows { name: String, rows: u64, cols: u64 },
    /// conv1d — 채널행 V순열(vbase 채널 이후 128채널 블록).
    Bf16Ch {
        name: String,
        ch: u64,
        kk: u64,
        vbase: usize,
    },
    /// 1D F32 — plus1=HF zero-centered norm(Qwen3.5 계열: HF는 w−1 저장,
    /// GGUF는 w 저장 — llama.cpp convert의 +1 규약, exl3 NormPlus1 실측).
    F32Direct { name: String, n: usize, plus1: bool },
    /// 1D F32 V순열(dt_bias).
    F32V { name: String, n: usize },
    /// ssm_a = −exp(A_log[V순열]).
    SsmA { name: String, n: usize },
}

impl W4a16Model {
    /// W4A16 디렉터리 → llm170 dialect GGUF(`w4a16-to-gguf`).
    /// 매핑은 exl3 convert.rs의 qwen35 계약 미러 — V헤드 순열·ssm_a −exp,
    /// MTP 15종은 미기입(스펙 디코드는 별도 과제). 양자화 선형만 dialect
    /// 타입, 플레인은 BF16/F32 원본 비트 보존.
    pub fn to_gguf(&self, out_path: &Path) -> R<ConvertStats> {
        use llm170_exl3::gguf_out::{GgufWriter, Kv};
        let t0 = std::time::Instant::now();
        let c = &self.cfg;
        let h = c.hidden as u64;
        let v = c.vocab as u64;
        let nv = c.linear_num_value_heads;
        let nk = c.linear_num_key_heads;
        let hd = c.linear_key_head_dim;
        let ahd = c.head_dim;
        let qkv = (2 * nk + nv) * hd;

        // ── KV ──
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Kv::Str("qwen35".into()));
        w.kv("qwen35.embedding_length", Kv::U32(c.hidden as u32));
        w.kv("qwen35.block_count", Kv::U32(c.layers as u32));
        w.kv("qwen35.attention.head_count", Kv::U32(c.heads as u32));
        w.kv("qwen35.attention.head_count_kv", Kv::U32(c.kv_heads as u32));
        w.kv("qwen35.attention.key_length", Kv::U32(c.head_dim as u32));
        w.kv("qwen35.ssm.state_size", Kv::U32(hd as u32));
        w.kv("qwen35.ssm.group_count", Kv::U32(nk as u32));
        w.kv("qwen35.ssm.time_step_rank", Kv::U32(nv as u32));
        w.kv("qwen35.ssm.inner_size", Kv::U32((nv * hd) as u32));
        w.kv("qwen35.feed_forward_length", Kv::U32(c.ffn as u32));
        w.kv(
            "qwen35.rope.dimension_count",
            Kv::U32((c.head_dim as f64 * c.partial_rotary_factor) as u32),
        );
        w.kv("qwen35.rope.freq_base", Kv::F64(c.rope_theta));
        w.kv(
            "qwen35.attention.layer_norm_rms_epsilon",
            Kv::F32(c.rms_norm_eps as f32),
        );
        w.kv(
            "qwen35.full_attention_interval",
            Kv::U32(c.full_interval as u32),
        );
        w.kv(
            "qwen35.ssm.conv_kernel",
            Kv::U32(c.linear_conv_kernel as u32),
        );
        w.kv("qwen35.context_length", Kv::U64(262144));
        let pieces = load_pieces(&self.dir)?;
        w.kv("tokenizer.ggml.tokens", Kv::StrArray(pieces));
        if let Some(e) = eos_of(&self.dir) {
            w.kv("tokenizer.ggml.eos_token_id", Kv::U32(e));
        }

        // ── 플랜 ──
        let mut plans: Vec<(String, WPlan)> = Vec::new();
        plans.push((
            "token_embd.weight".into(),
            WPlan::Bf16Direct {
                name: "model.language_model.embed_tokens.weight".into(),
                rows: v,
                cols: h,
            },
        ));
        plans.push((
            "output_norm.weight".into(),
            WPlan::F32Direct {
                name: "model.language_model.norm.weight".into(),
                n: c.hidden,
                plus1: true,
            },
        ));
        plans.push((
            "output.weight".into(),
            WPlan::Bf16Direct {
                name: "lm_head.weight".into(),
                rows: v,
                cols: h,
            },
        ));
        let vbase = 2 * nk * hd;
        for il in 0..c.layers {
            let l = format!("model.language_model.layers.{il}");
            let g = format!("blk.{il}");
            plans.push((
                format!("{g}.attn_norm.weight"),
                WPlan::F32Direct {
                    name: format!("{l}.input_layernorm.weight"),
                    n: c.hidden,
                    plus1: true,
                },
            ));
            let full = (il + 1).is_multiple_of(c.full_interval);
            if full {
                for (gn, sn, n_out) in [
                    ("attn_q", "self_attn.q_proj", c.heads * ahd * 2),
                    ("attn_k", "self_attn.k_proj", c.kv_heads * ahd),
                    ("attn_v", "self_attn.v_proj", c.kv_heads * ahd),
                ] {
                    plans.push((
                        format!("{g}.{gn}.weight"),
                        WPlan::Quant {
                            base: format!("{l}.{sn}"),
                            n: n_out,
                            k: c.hidden,
                            n_base: usize::MAX,
                            k_base: usize::MAX,
                        },
                    ));
                }
                plans.push((
                    format!("{g}.attn_q_norm.weight"),
                    WPlan::F32Direct {
                        name: format!("{l}.self_attn.q_norm.weight"),
                        n: ahd,
                        plus1: true,
                    },
                ));
                plans.push((
                    format!("{g}.attn_k_norm.weight"),
                    WPlan::F32Direct {
                        name: format!("{l}.self_attn.k_norm.weight"),
                        n: ahd,
                        plus1: true,
                    },
                ));
                plans.push((
                    format!("{g}.attn_output.weight"),
                    WPlan::Quant {
                        base: format!("{l}.self_attn.o_proj"),
                        n: c.hidden,
                        k: c.heads * ahd,
                        n_base: usize::MAX,
                        k_base: usize::MAX,
                    },
                ));
            } else {
                plans.push((
                    format!("{g}.attn_qkv.weight"),
                    WPlan::Quant {
                        base: format!("{l}.linear_attn.in_proj_qkv"),
                        n: qkv,
                        k: c.hidden,
                        n_base: vbase,
                        k_base: usize::MAX,
                    },
                ));
                plans.push((
                    format!("{g}.attn_gate.weight"),
                    WPlan::Quant {
                        base: format!("{l}.linear_attn.in_proj_z"),
                        n: nv * hd,
                        k: c.hidden,
                        n_base: 0,
                        k_base: usize::MAX,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_conv1d.weight"),
                    WPlan::Bf16Ch {
                        name: format!("{l}.linear_attn.conv1d.weight"),
                        ch: qkv as u64,
                        kk: c.linear_conv_kernel as u64,
                        vbase,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_dt.bias"),
                    WPlan::F32V {
                        name: format!("{l}.linear_attn.dt_bias"),
                        n: nv,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_a"),
                    WPlan::SsmA {
                        name: format!("{l}.linear_attn.A_log"),
                        n: nv,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_alpha.weight"),
                    WPlan::Bf16Rows {
                        name: format!("{l}.linear_attn.in_proj_a.weight"),
                        rows: nv as u64,
                        cols: h,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_beta.weight"),
                    WPlan::Bf16Rows {
                        name: format!("{l}.linear_attn.in_proj_b.weight"),
                        rows: nv as u64,
                        cols: h,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_norm.weight"),
                    WPlan::F32Direct {
                        name: format!("{l}.linear_attn.norm.weight"),
                        n: hd,
                        plus1: false,
                    },
                ));
                plans.push((
                    format!("{g}.ssm_out.weight"),
                    WPlan::Quant {
                        base: format!("{l}.linear_attn.out_proj"),
                        n: c.hidden,
                        k: nv * hd,
                        n_base: usize::MAX,
                        k_base: 0,
                    },
                ));
            }
            plans.push((
                format!("{g}.post_attention_norm.weight"),
                WPlan::F32Direct {
                    name: format!("{l}.post_attention_layernorm.weight"),
                    n: c.hidden,
                    plus1: true,
                },
            ));
            for (gn, sn) in [("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj")] {
                plans.push((
                    format!("{g}.{gn}.weight"),
                    WPlan::Quant {
                        base: format!("{l}.{sn}"),
                        n: c.ffn,
                        k: c.hidden,
                        n_base: usize::MAX,
                        k_base: usize::MAX,
                    },
                ));
            }
            plans.push((
                format!("{g}.ffn_down.weight"),
                WPlan::Quant {
                    base: format!("{l}.mlp.down_proj"),
                    n: c.hidden,
                    k: c.ffn,
                    n_base: usize::MAX,
                    k_base: usize::MAX,
                },
            ));
        }

        // ── 등록(플랜 순서 = 데이터 순서) ──
        let mut total = 0u64;
        for (name, p) in &plans {
            let len = match p {
                WPlan::Quant { n, k, .. } => (n * (k / 2 + k / GROUP * 2)) as u64,
                WPlan::Bf16Direct { rows, cols, .. } | WPlan::Bf16Rows { rows, cols, .. } => {
                    rows * cols * 2
                }
                WPlan::Bf16Ch { ch, kk, .. } => ch * kk * 2,
                WPlan::F32Direct { n, .. } | WPlan::F32V { n, .. } | WPlan::SsmA { n, .. } => {
                    (*n * 4) as u64
                }
            };
            match p {
                WPlan::Quant { n, k, .. } => {
                    w.tensor_raw(name, &[*k as u64, *n as u64], 100, len);
                }
                WPlan::Bf16Direct { rows, cols, .. } | WPlan::Bf16Rows { rows, cols, .. } => {
                    w.tensor_bf16(name, &[*cols, *rows]);
                }
                WPlan::Bf16Ch { ch, kk, .. } => {
                    w.tensor_bf16(name, &[*kk, *ch]);
                }
                WPlan::F32Direct { n, .. } | WPlan::F32V { n, .. } | WPlan::SsmA { n, .. } => {
                    w.tensor_f32(name, &[*n as u64]);
                }
            }
            total += len;
        }

        // ── 기입 ──
        let file = std::fs::File::create(out_path)?;
        let mut bw = std::io::BufWriter::with_capacity(16 << 20, file);
        let mut written = 0u64;
        let plans_ref = &plans;
        w.write(&mut bw, |name, _off, len, out| {
            let (_, plan) = plans_ref.iter().find(|(n, _)| n == name).expect("플랜");
            let buf = self.materialize(plan)?;
            if buf.len() as u64 != len {
                return Err(llm170_exl3::Exl3Error::BadTensor(format!(
                    "{name}: {} != {len} 바이트",
                    buf.len()
                )));
            }
            std::io::Write::write_all(out, &buf)?;
            written += len;
            if written / (1 << 30) != (written - len) / (1 << 30) {
                eprintln!(
                    "  [w4a16-conv] {:.1}/{:.1} GB",
                    written as f64 / 1e9,
                    total as f64 / 1e9
                );
            }
            Ok(())
        })?;
        std::io::Write::flush(&mut bw)?;
        Ok(ConvertStats {
            tensors: plans.len(),
            bytes: total,
            elapsed_s: t0.elapsed().as_secs_f64(),
        })
    }

    /// 플랜 → 바이트(변환기 본체).
    fn materialize(&self, p: &WPlan) -> Result<Vec<u8>, llm170_exl3::Exl3Error> {
        let err = |e: W4a16Error| llm170_exl3::Exl3Error::BadTensor(e.to_string());
        match p {
            WPlan::Quant {
                base,
                n,
                k,
                n_base,
                k_base,
            } => self
                .quant_bytes(base, *n, *k, *n_base, *k_base)
                .map_err(err),
            WPlan::Bf16Direct { name, rows, .. } => self.raw_rows(name, 0, *rows).map_err(err),
            WPlan::Bf16Rows { name, rows, cols } => {
                let raw = self.raw_rows(name, 0, *rows).map_err(err)?;
                let rb = (*cols * 2) as usize;
                let mut out = vec![0u8; raw.len()];
                for i in 0..*rows as usize {
                    out[i * rb..(i + 1) * rb]
                        .copy_from_slice(&raw[self.vperm(i) * rb..(self.vperm(i) + 1) * rb]);
                }
                Ok(out)
            }
            WPlan::Bf16Ch {
                name,
                ch,
                kk,
                vbase,
            } => {
                let raw = self.raw_rows(name, 0, *ch).map_err(err)?;
                let rb = (*kk * 2) as usize;
                let mut out = vec![0u8; raw.len()];
                for r in 0..*vbase {
                    out[r * rb..(r + 1) * rb].copy_from_slice(&raw[r * rb..(r + 1) * rb]);
                }
                let nb = (*ch as usize - *vbase) / GROUP;
                for blk in 0..nb {
                    let src = *vbase + self.vperm(blk) * GROUP;
                    let dst = *vbase + blk * GROUP;
                    out[dst * rb..(dst + GROUP) * rb]
                        .copy_from_slice(&raw[src * rb..(src + GROUP) * rb]);
                }
                Ok(out)
            }
            WPlan::F32Direct { name, plus1, .. } => {
                let v = self.plain_vec_f32(name).map_err(err)?;
                let mut out = Vec::with_capacity(v.len() * 4);
                for x in &v {
                    out.extend_from_slice(&(if *plus1 { x + 1.0 } else { *x }).to_le_bytes());
                }
                Ok(out)
            }
            WPlan::F32V { name, n } => {
                let v = self.plain_vec_f32(name).map_err(err)?;
                let mut out = Vec::with_capacity(*n * 4);
                for i in 0..*n {
                    out.extend_from_slice(&v[self.vperm(i)].to_le_bytes());
                }
                Ok(out)
            }
            WPlan::SsmA { name, n } => {
                let v = self.plain_vec_f32(name).map_err(err)?;
                let mut out = Vec::with_capacity(*n * 4);
                for i in 0..*n {
                    out.extend_from_slice(&(-v[self.vperm(i)].exp()).to_le_bytes());
                }
                Ok(out)
            }
        }
    }

    /// 양자화 선형 → W4A16G128 인터리브 블록(행 V순열·열 V순열).
    fn quant_bytes(
        &self,
        base: &str,
        n: usize,
        k: usize,
        n_base: usize,
        k_base: usize,
    ) -> R<Vec<u8>> {
        let packed = self.packed_rows_u32(base, 0, n as u64)?;
        let scales = self.scale_rows_u16(base, 0, n as u64)?;
        let nblk = k / GROUP;
        let wprow = k / 8;
        let span_n = if n_base == usize::MAX {
            0
        } else {
            (n - n_base) / GROUP
        };
        let span_k = if k_base == usize::MAX {
            0
        } else {
            (k - k_base) / GROUP
        };
        let row_map = |no: usize| -> usize {
            if span_n == 0 || no < n_base {
                return no;
            }
            let off = no - n_base;
            if off / GROUP >= span_n {
                return no;
            }
            n_base + self.vperm(off / GROUP) * GROUP + off % GROUP
        };
        let col_map = |g: usize| -> usize {
            if span_k == 0 {
                return g;
            }
            let off = g * GROUP;
            if off < k_base {
                return g;
            }
            let ob = (off - k_base) / GROUP;
            if ob >= span_k {
                return g;
            }
            (k_base + self.vperm(ob) * GROUP) / GROUP
        };
        let row_bytes = k / 2 + nblk * 2;
        let mut out = vec![0u8; n * row_bytes];
        for no in 0..n {
            let sn = row_map(no);
            let dst_row = no * row_bytes;
            for g in 0..nblk {
                let sg = col_map(g);
                let dst = dst_row + g * 66;
                for wi in 0..16 {
                    let word = packed[sn * wprow + sg * 16 + wi];
                    out[dst + wi * 4..dst + wi * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                out[dst + 64..dst + 66].copy_from_slice(&scales[sn * nblk + sg].to_le_bytes());
            }
        }
        Ok(out)
    }
}

/// 토큰 조각표(id 순) — vocab.json(exl3 규약) 우선, 없으면 tokenizer.json
/// (model.vocab + added_tokens 병합 — W4A16 HF 배포는 vocab.json 부재 실측,
/// 특수 토큰 33종은 added_tokens에만 있다).
fn load_pieces(dir: &Path) -> R<Vec<String>> {
    if dir.join("vocab.json").is_file() {
        return llm170_exl3::convert::load_token_pieces(dir).map_err(W4a16Error::Exl3);
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
    if let Some(llm170_exl3::Json::Arr(items)) = v.get("added_tokens") {
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

/// eos — generation_config.json 우선(tokenizer_config.json 폴백).
fn eos_of(dir: &Path) -> Option<u32> {
    for f in ["generation_config.json", "tokenizer_config.json"] {
        if let Ok(t) = std::fs::read_to_string(dir.join(f))
            && let Ok(j) = Json::parse(&t)
        {
            if let Some(e) = j.get("eos_token_id").and_then(Json::as_f64) {
                return Some(e as u32);
            }
            if let Some(e) = j.get("eos_token").and_then(Json::as_f64) {
                return Some(e as u32);
            }
        }
    }
    None
}
