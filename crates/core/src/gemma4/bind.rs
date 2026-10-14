//! gemma4 아키텍처 바인딩 — HF 이름맵·기대 커버리지·전문가 경로.
//! (qwen35/bind.rs 미러·병렬 — trait 추출 없이 아치별 독립 바인딩.)
//!
//! 계약(2026-10-14 HF `modeling_gemma4.py`/`configuration_gemma4.py` 대조):
//! - 층별 dims: 슬라이딩 `head_dim`/`num_key_value_heads`(256/8),
//!   풀 `global_head_dim`/`num_global_key_value_heads`(512/2 — k_eq_v 게이트).
//! - rope 2종: 슬라이딩 기본(θ=1e4·전차원 256), 풀 proportional
//!   (θ=1e6·partial 0.25 → 512 중 128 — 앞 64+뒤 64 블록, 페어 (p, p+256)).
//! - RMSNorm은 **직접 승수**(+1 합성 금지 — zero-centered 아님).
//! - 전문가 경로 `.experts.{e}.`(`.mlp.` 없음) — 스토어 `expert_at(base)`.

use crate::json::Json;
use crate::w4a16::{R, Report, W4a16Error, W4a16Model};
use std::path::Path;

/// 지원 아키텍처 식별자(config.json `architectures`).
pub const ARCHES: &[&str] = &["Gemma4ForConditionalGeneration"];

pub fn arch_ok(arch: &str) -> bool {
    ARCHES.contains(&arch)
}

/// 전문가 텐서 베이스 이름 — gemma4(`...layers.{l}.experts.{e}.{proj}`).
pub fn expert_base(layer: usize, e: usize, proj: &str) -> String {
    format!("model.language_model.layers.{layer}.experts.{e}.{proj}")
}

const LP: &str = "model.language_model.layers.";

/// gemma4 텍스트 설정(config.json text_config 발췌).
#[derive(Debug, Clone)]
pub struct GemmaCfg {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// 슬라이딩 rope θ(1e4).
    pub rope_theta_local: f64,
    /// 풀 rope θ(1e6).
    pub rope_theta_global: f64,
    /// 풀 rope 부분 회전율(0.25).
    pub partial_rot: f64,
    pub eps: f64,
    pub max_pos: usize,
    /// 최종 로짓 softcap(30.0 — 0이면 미적용).
    pub softcap: f64,
    /// 층별 풀 어텐션 여부(30 = 25 sliding + 5 full).
    pub layer_full: Vec<bool>,
    /// 슬라이딩 KV 헤드·head_dim(8/256).
    pub kv_local: usize,
    pub head_dim_local: usize,
    /// 풀 KV 헤드·head_dim(2/512) — k_eq_v면 v_proj 없음.
    pub kv_global: usize,
    pub head_dim_global: usize,
    pub k_eq_v: bool,
    /// MoE — enable_moe_block·num_experts.
    pub moe: bool,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_ffn: usize,
    /// 슬라이딩 창(1024 — 자기 포함).
    pub sw: usize,
}

impl GemmaCfg {
    /// config.json 로드 + 아치 게이트.
    pub fn load(dir: &Path) -> R<Self> {
        let txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| W4a16Error::Missing(format!("config.json: {e}")))?;
        let v =
            Json::parse(&txt).map_err(|e| W4a16Error::BadTensor(format!("config.json: {e}")))?;
        // 아치 게이트 — architectures[0] 또는 model_type.
        let arch = match v.get("architectures") {
            Some(Json::Arr(items)) => items.first().and_then(Json::as_str).map(String::from),
            _ => v.get("model_type").and_then(Json::as_str).map(String::from),
        };
        if let Some(a) = &arch
            && !arch_ok(a)
            && !a.starts_with("gemma4")
        {
            return Err(W4a16Error::Arch(format!("{a} — 지원: {ARCHES:?}")));
        }
        Self::parse(&txt)
    }

    fn parse(text: &str) -> R<Self> {
        let v =
            Json::parse(text).map_err(|e| W4a16Error::BadTensor(format!("config.json: {e}")))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| -> Option<usize> { tc.get(k).and_then(Json::as_f64).map(|x| x as usize) };
        let f = |k: &str| -> Option<f64> { tc.get(k).and_then(Json::as_f64) };
        let bad = |k: &str| W4a16Error::BadTensor(format!("config.json: {k} 부재"));
        let layers = u("num_hidden_layers").ok_or_else(|| bad("num_hidden_layers"))?;
        // layer_types — 필수(없으면 6주기 기본 복원은 하지 않는다: 오매핑 위험).
        let lt_arr = match tc.get("layer_types") {
            Some(Json::Arr(a)) => a,
            _ => return Err(bad("layer_types")),
        };
        if lt_arr.len() != layers {
            return Err(W4a16Error::Config(format!(
                "layer_types {} != num_hidden_layers {layers}",
                lt_arr.len()
            )));
        }
        let mut layer_full = Vec::with_capacity(layers);
        for it in lt_arr {
            match it.as_str() {
                Some("full_attention") => layer_full.push(true),
                Some("sliding_attention") => layer_full.push(false),
                other => {
                    return Err(W4a16Error::Config(format!(
                        "layer_types: {other:?} — full/sliding만"
                    )));
                }
            }
        }
        if !layer_full.last().copied().unwrap_or(false) {
            return Err(W4a16Error::Config("마지막 층이 full_attention 아님".into()));
        }
        // rope 2종 — rope_parameters.{sliding_attention,full_attention}.
        let rp = tc
            .get("rope_parameters")
            .ok_or_else(|| bad("rope_parameters"))?;
        let rp_s = rp
            .get("sliding_attention")
            .ok_or_else(|| bad("rope_parameters.sliding_attention"))?;
        let rp_f = rp
            .get("full_attention")
            .ok_or_else(|| bad("rope_parameters.full_attention"))?;
        let rope_theta_local = rp_s.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e4);
        let rope_theta_global = rp_f.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6);
        let partial_rot = rp_f
            .get("partial_rotary_factor")
            .and_then(Json::as_f64)
            .unwrap_or(1.0);
        let k_eq_v = tc
            .get("attention_k_eq_v")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let kv_local = u("num_key_value_heads").ok_or_else(|| bad("num_key_value_heads"))?;
        let head_dim_local = u("head_dim").ok_or_else(|| bad("head_dim"))?;
        let head_dim_global = u("global_head_dim").unwrap_or(head_dim_local);
        let kv_global = if k_eq_v {
            u("num_global_key_value_heads").ok_or_else(|| bad("num_global_key_value_heads"))?
        } else {
            kv_local
        };
        let moe = tc
            .get("enable_moe_block")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let cfg = GemmaCfg {
            hidden: u("hidden_size").ok_or_else(|| bad("hidden_size"))?,
            layers,
            heads: u("num_attention_heads").ok_or_else(|| bad("num_attention_heads"))?,
            ffn: u("intermediate_size").ok_or_else(|| bad("intermediate_size"))?,
            vocab: u("vocab_size").ok_or_else(|| bad("vocab_size"))?,
            rope_theta_local,
            rope_theta_global,
            partial_rot,
            eps: f("rms_norm_eps").unwrap_or(1e-6),
            max_pos: u("max_position_embeddings").ok_or_else(|| bad("max_position_embeddings"))?,
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            layer_full,
            kv_local,
            head_dim_local,
            kv_global,
            head_dim_global,
            k_eq_v,
            moe,
            n_experts: u("num_experts").unwrap_or(0),
            top_k: u("top_k_experts").unwrap_or(0),
            moe_ffn: u("moe_intermediate_size").unwrap_or(0),
            sw: u("sliding_window").unwrap_or(0),
        };
        // 교차필드 계약 — 위반은 0나누기·조용한 GQA 오매핑이 된다.
        let pos = |v: usize, k: &str| -> R<()> {
            if v == 0 {
                Err(W4a16Error::Config(format!("config.json: {k}=0")))
            } else {
                Ok(())
            }
        };
        pos(cfg.hidden, "hidden_size")?;
        pos(cfg.heads, "num_attention_heads")?;
        pos(cfg.ffn, "intermediate_size")?;
        pos(cfg.vocab, "vocab_size")?;
        pos(cfg.kv_local, "num_key_value_heads")?;
        pos(cfg.head_dim_local, "head_dim")?;
        pos(cfg.head_dim_global, "global_head_dim")?;
        pos(cfg.kv_global, "num_global_key_value_heads")?;
        pos(cfg.sw, "sliding_window")?;
        if !cfg.heads.is_multiple_of(cfg.kv_local) || !cfg.heads.is_multiple_of(cfg.kv_global) {
            return Err(W4a16Error::Config(format!(
                "num_attention_heads={} % kv({}/{}) != 0",
                cfg.heads, cfg.kv_local, cfg.kv_global
            )));
        }
        if !(0.0 < cfg.partial_rot && cfg.partial_rot <= 1.0) {
            return Err(W4a16Error::Config(format!(
                "partial_rotary_factor={} — (0,1]",
                cfg.partial_rot
            )));
        }
        if cfg.moe && (cfg.top_k == 0 || cfg.top_k > cfg.n_experts || cfg.moe_ffn == 0) {
            return Err(W4a16Error::Config(format!(
                "MoE 구성 위반 — experts={} top_k={} moe_ffn={}",
                cfg.n_experts, cfg.top_k, cfg.moe_ffn
            )));
        }
        Ok(cfg)
    }

    /// 층별 — 풀 층이면 global dims(kv_global·head_dim_global).
    pub fn head_dim(&self, il: usize) -> usize {
        if self.layer_full[il] {
            self.head_dim_global
        } else {
            self.head_dim_local
        }
    }

    pub fn n_kv(&self, il: usize) -> usize {
        if self.layer_full[il] {
            self.kv_global
        } else {
            self.kv_local
        }
    }

    /// 층별 rope 회전 차원 수 — 슬라이딩 전차원·풀 partial(짝수 정렬).
    pub fn n_rot(&self, il: usize) -> usize {
        if self.layer_full[il] {
            let r = (self.head_dim_global as f64 * self.partial_rot) as usize;
            r - (r % 2)
        } else {
            self.head_dim_local
        }
    }

    pub fn rope_base(&self, il: usize) -> f32 {
        if self.layer_full[il] {
            self.rope_theta_global as f32
        } else {
            self.rope_theta_local as f32
        }
    }

    /// 풀 층이 v_proj 없이 k 재사용(k_eq_v).
    pub fn kv_eq_v(&self, il: usize) -> bool {
        self.k_eq_v && self.layer_full[il]
    }
}

/// 엔진 접점 이름 → 소스 해석. (스테이지 접점 전수가 지나는 단일 계약)
pub enum Eng {
    /// 양자화 선형 — 분리 버퍼(packed+scale).
    Quant { base: String },
    /// 플레인(BF16) — 1D(노름·스칼라)·2D(라우터).
    Plain { name: String },
}

/// `blk.{il}.*`·전역 이름 → 소스 해석.
pub fn eng(name: &str) -> Option<Eng> {
    let plain = |n: String| Some(Eng::Plain { name: n });
    if name == "token_embd.weight" || name == "output.weight" {
        // tied — 같은 임베딩(입력은 √h 베이크, head는 원본 — Model 계약).
        return plain("model.language_model.embed_tokens.weight".into());
    }
    if name == "output_norm.weight" {
        return plain("model.language_model.norm.weight".into());
    }
    let rest = name.strip_prefix("blk.")?;
    let (il, suf) = rest.split_once('.')?;
    let l = format!("{LP}{il}");
    let qn = |m: &str| {
        Some(Eng::Quant {
            base: format!("{l}.{m}"),
        })
    };
    match suf {
        "attn_norm.weight" => plain(format!("{l}.input_layernorm.weight")),
        "post_attn_norm.weight" => plain(format!("{l}.post_attention_layernorm.weight")),
        "attn_q.weight" => qn("self_attn.q_proj"),
        "attn_k.weight" => qn("self_attn.k_proj"),
        "attn_v.weight" => qn("self_attn.v_proj"),
        "attn_output.weight" => qn("self_attn.o_proj"),
        "attn_q_norm.weight" => plain(format!("{l}.self_attn.q_norm.weight")),
        "attn_k_norm.weight" => plain(format!("{l}.self_attn.k_norm.weight")),
        "pre_ff_norm.weight" => plain(format!("{l}.pre_feedforward_layernorm.weight")),
        "pre_ff_norm2.weight" => plain(format!("{l}.pre_feedforward_layernorm_2.weight")),
        "post_ff_norm.weight" => plain(format!("{l}.post_feedforward_layernorm.weight")),
        "post_ff_norm1.weight" => plain(format!("{l}.post_feedforward_layernorm_1.weight")),
        "post_ff_norm2.weight" => plain(format!("{l}.post_feedforward_layernorm_2.weight")),
        "ffn_gate.weight" => qn("mlp.gate_proj"),
        "ffn_up.weight" => qn("mlp.up_proj"),
        "ffn_down.weight" => qn("mlp.down_proj"),
        "moe_gate.weight" => plain(format!("{l}.router.proj.weight")),
        "moe_scale" => plain(format!("{l}.router.scale")),
        "moe_per_expert_scale" => plain(format!("{l}.router.per_expert_scale")),
        "layer_scalar" => plain(format!("{l}.layer_scalar")),
        _ => None,
    }
}

/// 기대 텐서 집합 — (양자화 base, 플레인 이름). gemma4 텍스트 스키마 전수.
pub fn expected_names(cfg: &GemmaCfg) -> (Vec<String>, Vec<String>) {
    let (mut q, mut p) = (Vec::new(), Vec::new());
    for il in 0..cfg.layers {
        let pre = format!("{LP}{il}.");
        // 양자화 — 어텐션(q·k·o, 슬라이딩만 v)·dense MLP.
        for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"] {
            q.push(format!("{pre}{m}"));
        }
        if !cfg.kv_eq_v(il) {
            q.push(format!("{pre}self_attn.v_proj"));
        }
        for m in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            q.push(format!("{pre}{m}"));
        }
        // 플레인 — 노름 7종·QK 노름·라우터 3종·layer_scalar.
        for m in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "router.proj.weight",
            "router.scale",
            "router.per_expert_scale",
            "layer_scalar",
        ] {
            p.push(format!("{pre}{m}"));
        }
    }
    // 전역 — 임베딩(BF16)·최종 노름. lm_head는 tied(텐서 없음).
    p.push("model.language_model.embed_tokens.weight".into());
    p.push("model.language_model.norm.weight".into());
    (q, p)
}

/// 전수 검증 — 트리플 구조(스토어) + 층별 형상 계약 + 커버리지.
pub fn validate(store: &W4a16Model, cfg: &GemmaCfg) -> R<Report> {
    let mut rep = Report {
        triples: store.n_lins(),
        ..Default::default()
    };
    rep.bad_shape_value = store.check_shape_values()?;
    // 층별 양자화 형상 — (base, n, k).
    for il in 0..cfg.layers {
        let pre = format!("{LP}{il}.");
        let hd = cfg.head_dim(il);
        let kv = cfg.n_kv(il);
        let mut want: Vec<(String, usize, usize)> = vec![
            ("self_attn.q_proj".into(), cfg.heads * hd, cfg.hidden),
            ("self_attn.k_proj".into(), kv * hd, cfg.hidden),
            ("self_attn.o_proj".into(), cfg.hidden, cfg.heads * hd),
            ("mlp.gate_proj".into(), cfg.ffn, cfg.hidden),
            ("mlp.up_proj".into(), cfg.ffn, cfg.hidden),
            ("mlp.down_proj".into(), cfg.hidden, cfg.ffn),
        ];
        if !cfg.kv_eq_v(il) {
            want.push(("self_attn.v_proj".into(), kv * hd, cfg.hidden));
        }
        for (m, n, k) in want {
            let base = format!("{pre}{m}");
            match store.lin_shape(&base) {
                Some((nn, kk)) if nn == n && kk == k => {}
                Some((nn, kk)) => {
                    return Err(W4a16Error::Config(format!(
                        "{base}: 형상 ({nn},{kk}) != ({n},{k})"
                    )));
                }
                None => {
                    return Err(W4a16Error::Config(format!("{base}: 트리플 부재")));
                }
            }
        }
        if cfg.moe {
            for e in 0..cfg.n_experts {
                for (proj, n, k) in [
                    ("gate_proj", cfg.moe_ffn, cfg.hidden),
                    ("up_proj", cfg.moe_ffn, cfg.hidden),
                    ("down_proj", cfg.hidden, cfg.moe_ffn),
                ] {
                    let base = expert_base(il, e, proj);
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
    // 커버리지 — 기대 양자화/플레인 + 분류(비전 타워 = visual).
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
    let triple_part = |n: &str| {
        n.ends_with(".weight_packed")
            || n.ends_with(".weight_scale")
            || n.ends_with(".weight_shape")
    };
    for name in store.entries().keys() {
        if name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.") {
            rep.visual += 1;
            continue;
        }
        if triple_part(name) {
            continue;
        }
        if exp_p.contains(name) {
            rep.plain_text += 1;
        } else {
            rep.unknown_text.push(name.clone());
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_text(full_last: bool, k_eq_v: bool) -> String {
        let lt = if full_last {
            r#"["sliding_attention","sliding_attention","full_attention"]"#
        } else {
            r#"["sliding_attention","sliding_attention","sliding_attention"]"#
        };
        format!(
            r#"{{"text_config":{{"hidden_size":2816,"num_hidden_layers":3,"num_attention_heads":16,
            "num_key_value_heads":8,"head_dim":256,"global_head_dim":512,"num_global_key_value_heads":2,
            "intermediate_size":2112,"vocab_size":262144,"rms_norm_eps":1e-6,
            "max_position_embeddings":262144,"final_logit_softcapping":30.0,
            "sliding_window":1024,"attention_k_eq_v":{k_eq_v},"enable_moe_block":true,
            "num_experts":128,"top_k_experts":8,"moe_intermediate_size":704,
            "layer_types":{lt},"rope_parameters":{{"sliding_attention":{{"rope_theta":10000.0}},
            "full_attention":{{"rope_theta":1000000.0,"partial_rotary_factor":0.25,"rope_type":"proportional"}}}}}}}}"#
        )
    }

    #[test]
    fn cfg_parse_layer_dims_and_rope() {
        let c = GemmaCfg::parse(&cfg_text(true, true)).expect("계약");
        assert_eq!(c.layers, 3);
        assert!(!c.layer_full[0] && c.layer_full[2]);
        // 슬라이딩 = 256/8kv·전차원 rope.
        assert_eq!(c.head_dim(0), 256);
        assert_eq!(c.n_kv(0), 8);
        assert_eq!(c.n_rot(0), 256);
        assert_eq!(c.rope_base(0), 10_000.0);
        // 풀 = 512/2kv·partial 0.25 → 128 회전·k_eq_v면 v_proj 없음.
        assert_eq!(c.head_dim(2), 512);
        assert_eq!(c.n_kv(2), 2);
        assert_eq!(c.n_rot(2), 128);
        assert_eq!(c.rope_base(2), 1_000_000.0);
        assert!(c.kv_eq_v(2));
        assert!(!c.kv_eq_v(0));
    }

    #[test]
    fn cfg_reject_cross_fields() {
        // 마지막 층 sliding — 거부(HF도 강제 full로 교정하지만 우리는 명시 거부).
        let e = GemmaCfg::parse(&cfg_text(false, true))
            .unwrap_err()
            .to_string();
        assert!(e.contains("full_attention"), "{e}");
        // k_eq_v=false면 풀층도 v_proj 기대.
        let c = GemmaCfg::parse(&cfg_text(true, false)).expect("k_eq_v off");
        assert!(!c.kv_eq_v(2));
        let (q, p) = expected_names(&c);
        assert!(q.contains(&format!("{LP}2.self_attn.v_proj")));
        assert!(p.contains(&"model.language_model.embed_tokens.weight".to_string()));
    }
}
