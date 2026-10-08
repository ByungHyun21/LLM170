//! W4A16 safetensors 로더 — compressed-tensors pack-quantized(int4 sym g128).
//! (2026-10-08 — "3계열 모두 로드" 목표)
//!
//! [실측 스키마 — ../models/Qwen3.8-27B-W4A16-AutoRound]
//! - 양자화 선형 1개 = 3조: `.weight_packed` I32[n, k/8] · `.weight_scale`
//!   F16[n, k/128] · `.weight_shape` I64[2]=(n,k). 행=출력(n), 열=입력(k).
//! - 대칭(sym)이라 zero-point 미저장 → zp=8 고정(4bit 중심 — lane 미러 계약).
//! - 니블 순서 lsb-first **확정**(2026-10-08, w4a16-xcheck — 동일 기저
//!   27B 원본 대조 corr(lsb) 0.991~0.994 vs corr(msb) ≈0.01; lane.rs §3.6).
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

use crate::json::Json;
use crate::st::{StArchive, StDtype};
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
    St(crate::st::StError),
    Io(std::io::Error),
}

impl std::fmt::Display for W4a16Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            W4a16Error::Missing(n) => write!(f, "w4a16 missing tensor: {n}"),
            W4a16Error::BadTensor(s) => write!(f, "w4a16 bad tensor: {s}"),
            W4a16Error::Quant(s) => write!(f, "w4a16 unsupported quant: {s}"),
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
    /// 샤드 mmap — 엔진 직접 로드(§3.5 A안)의 무카피 Weight 슬라이스용.
    /// 프로브 경로는 pread 접근자를 그대로 쓴다.
    mmaps: Vec<memmap2::Mmap>,
    pub cfg: W4a16Config,
    /// 원본 디렉터리(vocab.json·generation_config.json — 변환기용).
    pub dir: std::path::PathBuf,
    /// `model.language_model.layers.{il}.mlp.gate_proj` → (n, k).
    lins: HashMap<String, (usize, usize)>,
    /// V축 순열 사본(엔진 접점 전용 — 첫 접근 시 1회 구축, ~2.9GB).
    perm: std::sync::OnceLock<PermStore>,
}

/// 엔진(블록) 텐서명 해석 — qwen35 스테이지가 요구하는 이름을 소스로 매핑.
/// **[정정]** V헤드 순열은 **필수**다 — 엔진은 subhead-major 계약
/// (V헤드 h ↔ K헤드 h % nk, gdn.rs `ik1 = iv1 % nek1` 미러)이라, HF 원본
/// (group-major)을 그대로 주면 k/v 짝이 어긋난다(직접 로드 실측: 출력 붕괴).
/// 따라서 v-축 텐서는 순열 사본(perm store)으로 subhead-major를 공급한다.
pub enum Eng {
    /// 양자화 선형 — 분리 버퍼 Weight(packed+scale). vperm 적용 축 명시.
    Quant { base: String, vperm: PV },
    /// BF16 플레인 — rows_perm=true면 행(V헤드) 순열(alpha/beta).
    Plain { name: String, rows_perm: bool },
    /// f32 합성 — perm: 0 없음(norm)·1 헤드 인덱스(dt_bias/A_log)·2 conv 채널.
    Synth {
        name: String,
        plus1: bool,
        neg_exp: bool,
        perm: u8,
    },
}

/// V축 순열 스펙 — 엔진(subhead-major) 계약.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PV {
    /// 순열 없음.
    None,
    /// n축·vbase(=2·nk·hd) 이후 128행 블록(attn_qkv v부).
    VPart,
    /// n축 전 행 128블록(in_proj_z).
    AllN,
    /// k축 전 열 128블록(out_proj).
    AllK,
}

/// `blk.{il}.*`·전역 이름 → 소스 해석. (스테이지 44개 접점 전수가 지나는 단일 계약)
pub fn eng(name: &str) -> Option<Eng> {
    if name == "token_embd.weight" {
        return Some(Eng::Plain {
            name: "model.language_model.embed_tokens.weight".into(),
            rows_perm: false,
        });
    }
    if name == "output.weight" {
        return Some(Eng::Plain {
            name: "lm_head.weight".into(),
            rows_perm: false,
        });
    }
    if name == "output_norm.weight" {
        return Some(Eng::Synth {
            name: "model.language_model.norm.weight".into(),
            plus1: true,
            neg_exp: false,
            perm: 0,
        });
    }
    let rest = name.strip_prefix("blk.")?;
    let (il, suf) = rest.split_once('.')?;
    let l = format!("model.language_model.layers.{il}");
    let q = |m: &str, vperm: PV| {
        Some(Eng::Quant {
            base: format!("{l}.{m}"),
            vperm,
        })
    };
    let sy = |m: &str, plus1: bool, neg_exp: bool, perm: u8| {
        Some(Eng::Synth {
            name: format!("{l}.{m}"),
            plus1,
            neg_exp,
            perm,
        })
    };
    match suf {
        "attn_norm.weight" => sy("input_layernorm.weight", true, false, 0),
        "post_attention_norm.weight" => sy("post_attention_layernorm.weight", true, false, 0),
        "attn_q.weight" => q("self_attn.q_proj", PV::None),
        "attn_k.weight" => q("self_attn.k_proj", PV::None),
        "attn_v.weight" => q("self_attn.v_proj", PV::None),
        "attn_output.weight" => q("self_attn.o_proj", PV::None),
        "attn_q_norm.weight" => sy("self_attn.q_norm.weight", true, false, 0),
        "attn_k_norm.weight" => sy("self_attn.k_norm.weight", true, false, 0),
        "attn_qkv.weight" => q("linear_attn.in_proj_qkv", PV::VPart),
        "attn_gate.weight" => q("linear_attn.in_proj_z", PV::AllN),
        "ssm_conv1d.weight" => sy("linear_attn.conv1d.weight", false, false, 2),
        "ssm_dt.bias" => sy("linear_attn.dt_bias", false, false, 1),
        "ssm_a" => sy("linear_attn.A_log", false, true, 1),
        "ssm_alpha.weight" => Some(Eng::Plain {
            name: format!("{l}.linear_attn.in_proj_a.weight"),
            rows_perm: true,
        }),
        "ssm_beta.weight" => Some(Eng::Plain {
            name: format!("{l}.linear_attn.in_proj_b.weight"),
            rows_perm: true,
        }),
        "ssm_norm.weight" => sy("linear_attn.norm.weight", false, false, 0),
        "ssm_out.weight" => q("linear_attn.out_proj", PV::AllK),
        "ffn_gate.weight" => q("mlp.gate_proj", PV::None),
        "ffn_up.weight" => q("mlp.up_proj", PV::None),
        "ffn_down.weight" => q("mlp.down_proj", PV::None),
        _ => None,
    }
}

/// 순열 사본 저장소 — 엔진 접점의 V축 텐서(HF→subhead-major).
struct PermStore {
    /// quant base → (packed, scale) 순열 사본.
    q: HashMap<String, (Vec<u8>, Vec<u8>)>,
    /// 플레인(HF 이름) → BF16 행 순열 사본(alpha/beta).
    p: HashMap<String, Vec<u8>>,
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
            cfg,
            dir: dir.to_path_buf(),
            lins,
            perm: std::sync::OnceLock::new(),
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

    /// 헤더 엔트리 조회(서빙 경로 — Weight 슬라이스 형상 판독).
    pub fn entry(&self, name: &str) -> Option<&crate::st::StEntry> {
        self.ar.entry(name)
    }

    /// 텐서 원시 슬라이스(mmap) — 엔진 직접 로드(§3.5 A안) 무카피 경로.
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

    /// 순열 사본 저장소(지연 1회 구축 — 48 선형층 qkv-v/z/out + alpha/beta).
    fn perm_store(&self) -> &PermStore {
        self.perm.get_or_init(|| self.build_perm())
    }

    /// V헤드 순열 사본 구축 — 엔진(subhead-major) 계약 공급용.
    /// 규약은 vperm 미러(F→3·(i%16)+i/16 실측 확정).
    fn build_perm(&self) -> PermStore {
        let c = &self.cfg;
        let nk = c.linear_num_key_heads;
        let vbase = 2 * nk * c.linear_key_head_dim;
        let mut q = HashMap::new();
        let mut p = HashMap::new();
        let lp = "model.language_model.layers.";
        for il in 0..c.layers {
            if (il + 1).is_multiple_of(c.full_interval) {
                continue; // full-attn 층은 V축 순열 없음(직접 대응 확인됨).
            }
            let l = format!("{lp}{il}.");
            for (suf, pv) in [
                ("linear_attn.in_proj_qkv", PV::VPart),
                ("linear_attn.in_proj_z", PV::AllN),
                ("linear_attn.out_proj", PV::AllK),
            ] {
                let base = format!("{l}{suf}");
                let Some((n, k)) = self.lin_shape(&base) else {
                    continue;
                };
                let (Some(pk), Some(sc)) = (
                    self.tensor_slice(&format!("{base}.weight_packed")),
                    self.tensor_slice(&format!("{base}.weight_scale")),
                ) else {
                    continue;
                };
                let (rb, sb) = (k / 2, k / 64);
                let mut d = vec![0u8; n * rb];
                let mut ds = vec![0u8; n * sb];
                match pv {
                    PV::VPart | PV::AllN => {
                        let nb0 = if pv == PV::VPart { vbase / 128 } else { 0 };
                        let nblk = n / 128;
                        d[..nb0 * 128 * rb].copy_from_slice(&pk[..nb0 * 128 * rb]);
                        ds[..nb0 * 128 * sb].copy_from_slice(&sc[..nb0 * 128 * sb]);
                        for b in 0..nblk - nb0 {
                            let s = (nb0 + self.vperm(b)) * 128 * rb;
                            let t = (nb0 + b) * 128 * rb;
                            d[t..t + 128 * rb].copy_from_slice(&pk[s..s + 128 * rb]);
                            let s2 = (nb0 + self.vperm(b)) * 128 * sb;
                            let t2 = (nb0 + b) * 128 * sb;
                            ds[t2..t2 + 128 * sb].copy_from_slice(&sc[s2..s2 + 128 * sb]);
                        }
                    }
                    PV::AllK => {
                        // 행 내 k-블록(128열) 순열 — 원본 행 사본 후 셔플.
                        let nblk = k / 128;
                        let mut scratch = vec![0u8; rb];
                        for r in 0..n {
                            scratch.copy_from_slice(&pk[r * rb..(r + 1) * rb]);
                            for g in 0..nblk {
                                let s = g * 64; // 128원소 = u32×16 = 64B
                                let sg = self.vperm(g) * 64;
                                d[r * rb + s..r * rb + s + 64]
                                    .copy_from_slice(&scratch[sg..sg + 64]);
                            }
                            let srow = &mut ds[r * sb..(r + 1) * sb];
                            let temp: Vec<u8> = sc[r * sb..(r + 1) * sb].to_vec();
                            for g in 0..nblk {
                                let sg = self.vperm(g) * 2;
                                srow[g * 2..g * 2 + 2].copy_from_slice(&temp[sg..sg + 2]);
                            }
                        }
                    }
                    PV::None => {}
                }
                q.insert(base, (d, ds));
            }
            // alpha/beta — 행(V헤드) 순열(BF16 바이트).
            for suf in [
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
            ] {
                let name = format!("{l}{suf}");
                let Some(src) = self.tensor_slice(&name) else {
                    continue;
                };
                let rows = c.linear_num_value_heads;
                let rb = src.len() / rows;
                let mut out = vec![0u8; src.len()];
                for i in 0..rows {
                    out[i * rb..(i + 1) * rb]
                        .copy_from_slice(&src[self.vperm(i) * rb..(self.vperm(i) + 1) * rb]);
                }
                p.insert(name, out);
            }
        }
        PermStore { q, p }
    }

    /// 양자화 순열 사본 슬라이스 — PV::None이면 호출하지 않는다(원본 슬라이스 사용).
    pub fn perm_quant(&self, base: &str, pv: PV) -> Option<(&[u8], &[u8])> {
        debug_assert!(pv != PV::None);
        let _ = pv;
        let s = self.perm_store().q.get(base)?;
        Some((&s.0, &s.1))
    }

    /// 플레인 행 순열 사본 슬라이스(alpha/beta).
    pub fn perm_plain(&self, name: &str) -> Option<&[u8]> {
        self.perm_store().p.get(name).map(|v| v.as_slice())
    }

    /// 헤드 인덱스 순열(1D — dt_bias·A_log).
    pub fn permute_heads_f32(&self, v: &[f32]) -> Vec<f32> {
        (0..v.len()).map(|i| v[self.vperm(i)]).collect()
    }

    /// conv 채널 행 순열 f32 — [ch][kk] 평탄, vbase 이후 128채널 블록 순열.
    pub fn conv_rows_f32_permuted(&self, name: &str) -> R<Vec<f32>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| W4a16Error::Missing(name.into()))?;
        let chk: u64 = e.shape[1..].iter().product();
        let ch = e.shape[0];
        let raw = self.raw_rows(name, 0, ch)?;
        let v = decode_f32(&raw, e.dtype, name)?;
        let kk = chk as usize;
        let c = &self.cfg;
        let vbase = 2 * c.linear_num_key_heads * c.linear_key_head_dim;
        let mut out = vec![0f32; v.len()];
        out[..vbase * kk].copy_from_slice(&v[..vbase * kk]);
        let nb = (ch as usize - vbase) / 128;
        for b in 0..nb {
            let s = (vbase + self.vperm(b) * 128) * kk;
            let t = (vbase + b * 128) * kk;
            out[t..t + 128 * kk].copy_from_slice(&v[s..s + 128 * kk]);
        }
        Ok(out)
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
    /// 블록 i ← 원본 블록 ratio·(i%nk) + i/nk (실측 확정 — beta 지문
    /// corr 1.000·ssm_out 블록 corr 0.999, 동일 규약 미러).
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

/// 토큰 조각표(id 순) — vocab.json(디렉터리 규약) 우선, 없으면 tokenizer.json
/// (model.vocab + added_tokens 병합 — W4A16 HF 배포는 vocab.json 부재 실측,
/// 특수 토큰 33종은 added_tokens에만 있다). Model 로드 공용.
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
