//! qwen35 아키텍처 바인딩 — HF 이름맵·synth·V헤드 순열·기대 커버리지.
//! (2026-10-08 R4: `w4a16` 스토어에서 분리. 새 모델군은 이 모듈에 준하는
//! 바인딩을 추가한다 — 스토어는 형식만, 바인딩이 아키텍처를 안다.)

use crate::json::Json;
use crate::w4a16::{R, Report, W4a16Error, W4a16Model, decode_f32};
use std::collections::HashMap;
use std::path::Path;

/// 지원 아키텍처 식별자(config.json `architectures`/`model_type`).
pub const ARCHES: &[&str] = &[
    "Qwen3_5ForConditionalGeneration",
    // W4-1: 35B-A3B MoE(g32) — dense와 같은 qwen3_5 계열, FFN만 MoE.
    "Qwen3_5MoeForConditionalGeneration",
];

pub fn arch_supported(arch: &str) -> bool {
    ARCHES.contains(&arch)
}

/// 디렉터리 config.json에서 아키텍처 식별자 추출(architectures[0] → model_type).
pub fn dir_arch(dir: &Path) -> Option<String> {
    let txt = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v = Json::parse(&txt).ok()?;
    if let Some(Json::Arr(items)) = v.get("architectures")
        && let Some(Json::Str(s)) = items.first()
    {
        return Some(s.clone());
    }
    v.get("model_type").and_then(Json::as_str).map(String::from)
}

/// qwen3_5 계열 하이퍼파라미터(config.json 발췌).
#[derive(Debug, Clone)]
pub struct QwenCfg {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// full-attention 간격(4 → full il = {3,7,…,63}).
    pub full_interval: usize,
    /// GDN — state_size·key_heads·value_heads·conv 커널.
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    // ── MoE(0이면 dense FFN) ──
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_ffn: usize,
    pub shared_ffn: usize,
}

impl QwenCfg {
    /// config.json 로드 + 아키텍처 게이트.
    pub fn load(dir: &Path) -> R<Self> {
        if let Some(arch) = dir_arch(dir)
            && !arch_supported(&arch)
        {
            return Err(W4a16Error::Arch(format!("{arch} — 지원: {ARCHES:?}")));
        }
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| W4a16Error::Missing(format!("config.json: {e}")))?;
        Self::parse(&text)
    }

    fn parse(text: &str) -> R<Self> {
        let v =
            Json::parse(text).map_err(|e| W4a16Error::BadTensor(format!("config.json: {e}")))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| -> Option<usize> { tc.get(k).and_then(Json::as_f64).map(|x| x as usize) };
        let bad = |k: &str| W4a16Error::BadTensor(format!("config.json: {k} 부재"));
        // rope 파라미터는 text_config.rope_parameters에 중첩(실측) — 평면 키도 허용.
        let rp = tc.get("rope_parameters").unwrap_or(tc);
        let f = |k: &str| -> Option<f64> { rp.get(k).and_then(Json::as_f64) };
        let cfg = QwenCfg {
            hidden: u("hidden_size").ok_or_else(|| bad("hidden_size"))?,
            layers: u("num_hidden_layers").ok_or_else(|| bad("num_hidden_layers"))?,
            heads: u("num_attention_heads").ok_or_else(|| bad("num_attention_heads"))?,
            kv_heads: u("num_key_value_heads").ok_or_else(|| bad("num_key_value_heads"))?,
            head_dim: u("head_dim").ok_or_else(|| bad("head_dim"))?,
            // MoE는 dense intermediate_size가 없다 — moe_intermediate_size로
            // 대체(엔진 ffn 필드는 dense 전용, MoE 경로는 moe_ffn을 쓴다).
            ffn: u("intermediate_size")
                .or_else(|| u("moe_intermediate_size"))
                .ok_or_else(|| bad("intermediate_size"))?,
            vocab: u("vocab_size").ok_or_else(|| bad("vocab_size"))?,
            full_interval: u("full_attention_interval").unwrap_or(4).max(1),
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
            n_experts: u("num_experts").unwrap_or(0),
            top_k: u("num_experts_per_tok").unwrap_or(0),
            moe_ffn: u("moe_intermediate_size").unwrap_or(0),
            shared_ffn: u("shared_expert_intermediate_size").unwrap_or(0),
        };
        // 교차필드 계약(C4) — 위반은 로드 중 0나누기(attn)·언더플로(conv)·조용한
        // GQA 오매핑이 된다. 파싱 시점에 명시 거부(게이트 B/C 계약 보호).
        let pos = |v: usize, k: &str| -> R<()> {
            if v == 0 {
                Err(W4a16Error::Config(format!("config.json: {k}=0")))
            } else {
                Ok(())
            }
        };
        pos(cfg.hidden, "hidden_size")?;
        pos(cfg.layers, "num_hidden_layers")?;
        pos(cfg.heads, "num_attention_heads")?;
        pos(cfg.kv_heads, "num_key_value_heads")?;
        pos(cfg.head_dim, "head_dim")?;
        pos(cfg.ffn, "intermediate_size")?;
        pos(cfg.vocab, "vocab_size")?;
        pos(cfg.linear_key_head_dim, "linear_key_head_dim")?;
        pos(cfg.linear_num_key_heads, "linear_num_key_heads")?;
        pos(cfg.linear_num_value_heads, "linear_num_value_heads")?;
        pos(cfg.linear_conv_kernel, "linear_conv_kernel_dim")?;
        if !cfg.heads.is_multiple_of(cfg.kv_heads) {
            return Err(W4a16Error::Config(format!(
                "num_attention_heads={} % num_key_value_heads={} != 0",
                cfg.heads, cfg.kv_heads
            )));
        }
        if !cfg
            .linear_num_value_heads
            .is_multiple_of(cfg.linear_num_key_heads)
        {
            return Err(W4a16Error::Config(format!(
                "linear_num_value_heads={} % linear_num_key_heads={} != 0 (vperm ratio)",
                cfg.linear_num_value_heads, cfg.linear_num_key_heads
            )));
        }
        // MoE 교차필드 — n_experts>0이면 top_k/FFN 폭 전부 양수·상한 내.
        if cfg.n_experts > 0
            && (cfg.top_k == 0
                || cfg.top_k > cfg.n_experts
                || cfg.moe_ffn == 0
                || cfg.shared_ffn == 0)
        {
            return Err(W4a16Error::Config(format!(
                "MoE 구성 위반 — experts={} top_k={} moe_ffn={} shared_ffn={}",
                cfg.n_experts, cfg.top_k, cfg.moe_ffn, cfg.shared_ffn
            )));
        }
        Ok(cfg)
    }
}

/// 순열 사본 정렬 계약(C2) — build_perm/conv_rows_f32_permuted는 128행 블록
/// 단위로만 순열하고 꼬리(n%128)를 0으로 방치한다(오류·패닉 없이 잘못된
/// 가중치 = 조용한 오염). validate가 구축 전에 전수 검사한다.
/// name 규약: `in_proj_z`는 전 행(base 0), 그 외(qkv·conv1d)는 vbase 이후.
pub fn perm_align_check(cfg: &QwenCfg, items: &[(String, usize)]) -> R<()> {
    let vbase = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    for (name, v) in items {
        let base = if name.ends_with("in_proj_z") {
            0
        } else {
            vbase
        };
        if *v < base || !(*v - base).is_multiple_of(128) {
            return Err(W4a16Error::Config(format!(
                "{name}: 순열 정렬 위반 — n-base={} (128배수 아님, 순열 사본 0 방치)",
                *v as i64 - base as i64
            )));
        }
    }
    Ok(())
}

/// 엔진(블록) 텐서명 해석 — qwen35 스테이지가 요구하는 이름을 소스로 매핑.
/// V헤드 순열은 **필수**다 — 엔진은 subhead-major 계약
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

/// `blk.{il}.*`·전역 이름 → 소스 해석. (스테이지 접점 전수가 지나는 단일 계약)
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
        // MoE(35B) — 라우터·shared_expert는 플레인. 전문가는 스토어 접근자
        // (Model::expert_slice)로 인덱스 접근 — 이름맵에 30,720개를 넣지 않는다.
        "moe_gate.weight" => Some(Eng::Plain {
            name: format!("{l}.mlp.gate.weight"),
            rows_perm: false,
        }),
        "moe_shared_gate.weight" => Some(Eng::Plain {
            name: format!("{l}.mlp.shared_expert.gate_proj.weight"),
            rows_perm: false,
        }),
        "moe_shared_up.weight" => Some(Eng::Plain {
            name: format!("{l}.mlp.shared_expert.up_proj.weight"),
            rows_perm: false,
        }),
        "moe_shared_down.weight" => Some(Eng::Plain {
            name: format!("{l}.mlp.shared_expert.down_proj.weight"),
            rows_perm: false,
        }),
        "moe_shared_sgate.weight" => Some(Eng::Plain {
            name: format!("{l}.mlp.shared_expert_gate.weight"),
            rows_perm: false,
        }),
        _ => None,
    }
}

/// 엔진 접점 이름 전수(스테이지가 요구하는 이름) — 상주 로더/W3 배선용.
/// 층 유형과 무관하게 전 접미사를 열거한다(비실재는 w()가 None으로 걸러냄).
pub fn engine_names(cfg: &QwenCfg) -> Vec<String> {
    let mut v = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "output.weight".to_string(),
    ];
    for il in 0..cfg.layers {
        for suf in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_conv1d.weight",
            "ssm_dt.bias",
            "ssm_a",
            "ssm_alpha.weight",
            "ssm_beta.weight",
            "ssm_norm.weight",
            "ssm_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            v.push(format!("blk.{il}.{suf}"));
        }
    }
    v
}

/// 순열 사본 저장소 — 엔진 접점의 V축 텐서(HF→subhead-major).
pub struct PermStore {
    /// quant base → (packed, scale) 순열 사본.
    q: HashMap<String, (Vec<u8>, Vec<u8>)>,
    /// 플레인(HF 이름) → BF16 행 순열 사본(alpha/beta).
    p: HashMap<String, Vec<u8>>,
}

impl PermStore {
    /// 양자화 순열 사본 슬라이스 — PV::None이면 호출하지 않는다(원본 사용).
    pub fn quant(&self, base: &str) -> Option<(&[u8], &[u8])> {
        let s = self.q.get(base)?;
        Some((&s.0, &s.1))
    }

    /// 플레인 행 순열 사본 슬라이스(alpha/beta).
    pub fn plain(&self, name: &str) -> Option<&[u8]> {
        self.p.get(name).map(|v| v.as_slice())
    }
}

/// GDN V헤드 순열 — llama.cpp(subhead-major) ↔ HF(group-major):
/// 블록 i ← 원본 블록 ratio·(i%nk) + i/nk (실측 확정 — beta 지문
/// corr 1.000·ssm_out 블록 corr 0.999, 동일 규약 미러).
fn vperm(cfg: &QwenCfg, i: usize) -> usize {
    let nk = cfg.linear_num_key_heads;
    let ratio = cfg.linear_num_value_heads / nk;
    ratio * (i % nk) + i / nk
}

/// V헤드 순열 사본 구축 — 엔진(subhead-major) 계약 공급용.
/// 128블록 = 헤드차원(linear_key_head_dim) 단위라 그룹 무관 — 스케일 행/블록
/// 바이트만 group으로 환산한다(g128 27B · g32 35B).
pub fn build_perm(store: &W4a16Model, cfg: &QwenCfg) -> PermStore {
    let group = store.group();
    let nk = cfg.linear_num_key_heads;
    let vbase = 2 * nk * cfg.linear_key_head_dim;
    let mut q = HashMap::new();
    let mut p = HashMap::new();
    let lp = "model.language_model.layers.";
    for il in 0..cfg.layers {
        if (il + 1).is_multiple_of(cfg.full_interval) {
            continue; // full-attn 층은 V축 순열 없음(직접 대응 확인됨).
        }
        let l = format!("{lp}{il}.");
        for (suf, pv) in [
            ("linear_attn.in_proj_qkv", PV::VPart),
            ("linear_attn.in_proj_z", PV::AllN),
            ("linear_attn.out_proj", PV::AllK),
        ] {
            let base = format!("{l}{suf}");
            let Some((n, k)) = store.lin_shape(&base) else {
                continue;
            };
            let (Some(pk), Some(sc)) = (
                store.tensor_slice(&format!("{base}.weight_packed")),
                store.tensor_slice(&format!("{base}.weight_scale")),
            ) else {
                continue;
            };
            let (rb, sb) = (k / 2, k * 2 / group);
            let mut d = vec![0u8; n * rb];
            let mut ds = vec![0u8; n * sb];
            match pv {
                PV::VPart | PV::AllN => {
                    let nb0 = if pv == PV::VPart { vbase / 128 } else { 0 };
                    let nblk = n / 128;
                    d[..nb0 * 128 * rb].copy_from_slice(&pk[..nb0 * 128 * rb]);
                    ds[..nb0 * 128 * sb].copy_from_slice(&sc[..nb0 * 128 * sb]);
                    for b in 0..nblk - nb0 {
                        let s = (nb0 + vperm(cfg, b)) * 128 * rb;
                        let t = (nb0 + b) * 128 * rb;
                        d[t..t + 128 * rb].copy_from_slice(&pk[s..s + 128 * rb]);
                        let s2 = (nb0 + vperm(cfg, b)) * 128 * sb;
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
                            let sg = vperm(cfg, g) * 64;
                            d[r * rb + s..r * rb + s + 64].copy_from_slice(&scratch[sg..sg + 64]);
                        }
                        // 스케일 블록 = 128열 = 128/group개 스케일(×2B).
                        let sblk = 128 * 2 / group;
                        let srow = &mut ds[r * sb..(r + 1) * sb];
                        let temp: Vec<u8> = sc[r * sb..(r + 1) * sb].to_vec();
                        for g in 0..nblk {
                            let sg = vperm(cfg, g) * sblk;
                            srow[g * sblk..(g + 1) * sblk].copy_from_slice(&temp[sg..sg + sblk]);
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
            let Some(src) = store.tensor_slice(&name) else {
                continue;
            };
            let rows = cfg.linear_num_value_heads;
            let rb = src.len() / rows;
            let mut out = vec![0u8; src.len()];
            for i in 0..rows {
                out[i * rb..(i + 1) * rb]
                    .copy_from_slice(&src[vperm(cfg, i) * rb..(vperm(cfg, i) + 1) * rb]);
            }
            p.insert(name, out);
        }
    }
    PermStore { q, p }
}

/// 헤드 인덱스 순열(1D — dt_bias·A_log).
pub fn permute_heads_f32(cfg: &QwenCfg, v: &[f32]) -> Vec<f32> {
    (0..v.len()).map(|i| v[vperm(cfg, i)]).collect()
}

/// conv 채널 행 순열 f32 — [ch][kk] 평탄, vbase 이후 128채널(헤드차원) 블록
/// 순열 — 그룹 무관(스케일 무접촉).
pub fn conv_rows_f32_permuted(store: &W4a16Model, cfg: &QwenCfg, name: &str) -> R<Vec<f32>> {
    let e = store
        .entry(name)
        .ok_or_else(|| W4a16Error::Missing(name.into()))?;
    let chk: u64 = e.shape[1..].iter().product();
    let ch = e.shape[0];
    let raw = store.raw_rows(name, 0, ch)?;
    let v = decode_f32(&raw, e.dtype, name)?;
    let kk = chk as usize;
    let vbase = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let mut out = vec![0f32; v.len()];
    out[..vbase * kk].copy_from_slice(&v[..vbase * kk]);
    let nb = (ch as usize - vbase) / 128;
    for b in 0..nb {
        let s = (vbase + vperm(cfg, b) * 128) * kk;
        let t = (vbase + b * 128) * kk;
        out[t..t + 128 * kk].copy_from_slice(&v[s..s + 128 * kk]);
    }
    Ok(out)
}

/// 전수 검증 — weight_shape 값 대조(스토어) + 커버리지(기대 집합 전수).
pub fn validate(store: &W4a16Model, cfg: &QwenCfg) -> R<Report> {
    let mut rep = Report {
        triples: store.n_lins(),
        ..Default::default()
    };
    rep.bad_shape_value = store.check_shape_values()?;
    // MoE 전문가 트리플 구조(W4-1) — 층당 n_experts×3, 형상 계약 포함.
    // 이름맵·커버리지에 30,720개를 싣지 않고 여기서 구조로 검증한다.
    if cfg.n_experts > 0 {
        let lp = "model.language_model.layers.";
        for il in 0..cfg.layers {
            for e in 0..cfg.n_experts {
                for (proj, n, k) in [
                    ("gate_proj", cfg.moe_ffn, cfg.hidden),
                    ("up_proj", cfg.moe_ffn, cfg.hidden),
                    ("down_proj", cfg.hidden, cfg.moe_ffn),
                ] {
                    let base = format!("{lp}{il}.mlp.experts.{e}.{proj}");
                    match store.lin_shape(&base) {
                        Some((nn, kk)) if nn == n && kk == k => {}
                        Some((nn, kk)) => {
                            return Err(W4a16Error::Config(format!(
                                "{base}: 형상 ({nn},{kk}) != ({n},{k})"
                            )));
                        }
                        None => {
                            return Err(W4a16Error::Config(format!("{base}: 전문가 트리플 부재")));
                        }
                    }
                }
            }
        }
    }
    // 순열 정렬 계약(C2) — 순열 사본 구축(build_perm) 전에 전수 거부.
    {
        let lp = "model.language_model.layers.";
        let mut items: Vec<(String, usize)> = Vec::new();
        for il in 0..cfg.layers {
            if (il + 1).is_multiple_of(cfg.full_interval) {
                continue; // full-attn 층은 V축 순열 없음.
            }
            let l = format!("{lp}{il}.");
            for (suf, nm) in [
                ("linear_attn.in_proj_qkv", "in_proj_qkv"),
                ("linear_attn.in_proj_z", "in_proj_z"),
            ] {
                if let Some((n, _k)) = store.lin_shape(&format!("{l}{suf}")) {
                    items.push((format!("{l}{nm}"), n));
                }
            }
            if let Some(e) = store.entry(&format!("{l}linear_attn.conv1d.weight")) {
                items.push((format!("{l}conv1d"), e.shape[0] as usize));
            }
        }
        perm_align_check(cfg, &items)?;
    }
    // 커버리지: 기대 양자화/플레인 집합 vs 실제.
    let (exp_q, exp_p) = expected_names(cfg);
    for base in &exp_q {
        if store.lin_shape(base).is_none() {
            rep.missing_quant.push(base.clone());
        }
    }
    for name in &exp_p {
        if store.entry(name).is_none() {
            rep.missing_plain.push(name.clone());
        }
    }
    // 분류: visual/mtp/텍스트 플레인/트리플 파트.
    let triple_part = |n: &str| {
        n.ends_with(".weight_packed")
            || n.ends_with(".weight_scale")
            || n.ends_with(".weight_shape")
    };
    for name in store.entries().keys() {
        if name.starts_with("model.visual.") {
            // visual은 무양자화 BF16/F32 — 텍스트 경로 무사용.
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

/// 기대 텐서 집합 — (양자화 base, 플레인 이름). qwen3_5 실측 스키마의 전수.
fn expected_names(cfg: &QwenCfg) -> (Vec<String>, Vec<String>) {
    let (mut q, mut p) = (Vec::new(), Vec::new());
    let lp = "model.language_model.layers.";
    // MoE(35B): dense mlp.{gate,up,down}_proj 부재 — 라우터·shared_expert가
    // 플레인으로, 전문가 트리플은 validate의 구조 검사(층당 experts×3)가 맡는다.
    let ffn_q: &[&str] = if cfg.n_experts > 0 {
        &[]
    } else {
        &["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    };
    let ffn_p: &[&str] = if cfg.n_experts > 0 {
        &[
            "mlp.gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        ]
    } else {
        &[]
    };
    // MoE는 어텐션 투영도 BF16 플레인(실측 — 전문가만 양자화).
    let attn_q: &[&str] = if cfg.n_experts > 0 {
        &[]
    } else {
        &[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
    };
    let attn_qg: &[&str] = if cfg.n_experts > 0 {
        &[]
    } else {
        &[
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ]
    };
    let attn_p: &[&str] = if cfg.n_experts > 0 {
        &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]
    } else {
        &[]
    };
    let attn_pg: &[&str] = if cfg.n_experts > 0 {
        &[
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.out_proj.weight",
        ]
    } else {
        &[]
    };
    for il in 0..cfg.layers {
        let full = (il + 1).is_multiple_of(cfg.full_interval);
        let pre = format!("{lp}{il}.");
        if full {
            for m in attn_q.iter().chain(ffn_q) {
                q.push(format!("{pre}{m}"));
            }
            for m in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]
            .into_iter()
            .chain(attn_p.iter().copied())
            .chain(ffn_p.iter().copied())
            {
                p.push(format!("{pre}{m}"));
            }
        } else {
            for m in attn_qg.iter().chain(ffn_q) {
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
            ]
            .into_iter()
            .chain(attn_pg.iter().copied())
            .chain(ffn_p.iter().copied())
            {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_text(heads: usize, kv: usize, nk: usize, nv: usize, conv_k: usize) -> String {
        format!(
            r#"{{"hidden_size":5120,"num_hidden_layers":4,"num_attention_heads":{heads},"num_key_value_heads":{kv},"head_dim":128,"intermediate_size":17408,"vocab_size":248320,"linear_key_head_dim":128,"linear_num_key_heads":{nk},"linear_num_value_heads":{nv},"linear_conv_kernel_dim":{conv_k}}}"#
        )
    }

    #[test]
    fn cfg_accept_and_reject_cross_fields() {
        assert!(QwenCfg::parse(&cfg_text(40, 8, 16, 64, 4)).is_ok());
        // C4: 0 필드·GQA 불일치·vperm ratio 위반은 명시 거부.
        let e = QwenCfg::parse(&cfg_text(40, 0, 16, 64, 4))
            .unwrap_err()
            .to_string();
        assert!(e.contains("num_key_value_heads=0"), "{e}");
        let e = QwenCfg::parse(&cfg_text(41, 8, 16, 64, 4))
            .unwrap_err()
            .to_string();
        assert!(e.contains("num_key_value_heads"), "{e}");
        let e = QwenCfg::parse(&cfg_text(40, 8, 16, 65, 4))
            .unwrap_err()
            .to_string();
        assert!(e.contains("vperm"), "{e}");
        let e = QwenCfg::parse(&cfg_text(40, 8, 16, 64, 0))
            .unwrap_err()
            .to_string();
        assert!(e.contains("linear_conv_kernel_dim=0"), "{e}");
    }

    fn cfg_fixture() -> QwenCfg {
        QwenCfg {
            hidden: 5120,
            layers: 4,
            heads: 40,
            kv_heads: 8,
            head_dim: 128,
            ffn: 17408,
            vocab: 248320,
            full_interval: 4,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 64,
            linear_conv_kernel: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rms_norm_eps: 1e-6,
            n_experts: 0,
            top_k: 0,
            moe_ffn: 0,
            shared_ffn: 0,
        }
    }

    #[test]
    fn perm_align_rules() {
        let cfg = cfg_fixture();
        let vbase = 2 * 16 * 128; // 4096
        // 정렬 OK — qkv v부(6144)·z(5120)·conv(vbase+2560).
        let ok = vec![
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv".to_string(),
                vbase + 6144,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_z".to_string(),
                5120,
            ),
            (
                "model.language_model.layers.0.conv1d".to_string(),
                vbase + 2560,
            ),
        ];
        assert!(perm_align_check(&cfg, &ok).is_ok());
        // C2: 128 꼬리행은 거부(순열 사본이 0으로 방치되는 오염).
        let bad = vec![(
            "model.language_model.layers.0.linear_attn.in_proj_qkv".to_string(),
            vbase + 6144 + 64,
        )];
        let e = perm_align_check(&cfg, &bad).unwrap_err().to_string();
        assert!(e.contains("순열 정렬 위반"), "{e}");
        // vbase 미만도 거부.
        let bad2 = vec![(
            "model.language_model.layers.0.conv1d".to_string(),
            vbase - 128,
        )];
        assert!(perm_align_check(&cfg, &bad2).is_err());
    }
}
