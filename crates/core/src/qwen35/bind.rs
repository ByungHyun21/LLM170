
use crate::json::Json;
use crate::w4a16::{R, Report, W4a16Error, W4a16Model, decode_f32};
use std::collections::HashMap;
use std::path::Path;

pub const ARCHES: &[&str] = &[
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
];

pub fn arch_supported(arch: &str) -> bool {
    ARCHES.contains(&arch)
}

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

pub fn final_logit_softcap(dir: &Path) -> f32 {
    let Ok(txt) = std::fs::read_to_string(dir.join("config.json")) else {
        return 0.0;
    };
    let Ok(v) = Json::parse(&txt) else {
        return 0.0;
    };
    let tc = v.get("text_config").unwrap_or(&v);
    tc.get("final_logit_softcapping")
        .and_then(Json::as_f64)
        .unwrap_or(0.0) as f32
}

pub fn expert_base(layer: usize, e: usize, proj: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{e}.{proj}")
}

#[derive(Debug, Clone)]
pub struct QwenCfg {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub full_interval: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_pos: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_ffn: usize,
    pub shared_ffn: usize,
}

impl QwenCfg {
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
        let rp = tc.get("rope_parameters").unwrap_or(tc);
        let f = |k: &str| -> Option<f64> { rp.get(k).and_then(Json::as_f64) };
        let cfg = QwenCfg {
            hidden: u("hidden_size").ok_or_else(|| bad("hidden_size"))?,
            layers: u("num_hidden_layers").ok_or_else(|| bad("num_hidden_layers"))?,
            heads: u("num_attention_heads").ok_or_else(|| bad("num_attention_heads"))?,
            kv_heads: u("num_key_value_heads").ok_or_else(|| bad("num_key_value_heads"))?,
            head_dim: u("head_dim").ok_or_else(|| bad("head_dim"))?,
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
            max_pos: u("max_position_embeddings")
                .or_else(|| {
                    v.get("max_position_embeddings")
                        .and_then(Json::as_f64)
                        .map(|x| x as usize)
                })
                .ok_or_else(|| bad("max_position_embeddings"))?,
            n_experts: u("num_experts").unwrap_or(0),
            top_k: u("num_experts_per_tok").unwrap_or(0),
            moe_ffn: u("moe_intermediate_size").unwrap_or(0),
            shared_ffn: u("shared_expert_intermediate_size").unwrap_or(0),
        };
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
        pos(cfg.max_pos, "max_position_embeddings")?;
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

pub enum Eng {
    Quant { base: String, vperm: PV },
    Plain { name: String, rows_perm: bool },
    Synth {
        name: String,
        plus1: bool,
        neg_exp: bool,
        perm: u8,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PV {
    None,
    VPart,
    AllN,
    AllK,
}

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
    if name.starts_with("mtp.") {
        if name.ends_with(".weight") && name.contains("norm") {
            return Some(Eng::Synth {
                name: name.into(),
                plus1: true,
                neg_exp: false,
                perm: 0,
            });
        }
        if name.ends_with(".weight") {
            return Some(Eng::Plain {
                name: name.into(),
                rows_perm: false,
            });
        }
        return None;
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
        if cfg.n_experts > 0 {
            for suf in [
                "moe_gate.weight",
                "moe_shared_gate.weight",
                "moe_shared_up.weight",
                "moe_shared_down.weight",
                "moe_shared_sgate.weight",
            ] {
                v.push(format!("blk.{il}.{suf}"));
            }
        }
    }
    v
}

pub struct PermStore {
    q: HashMap<String, (Vec<u8>, Vec<u8>)>,
    p: HashMap<String, Vec<u8>>,
}

impl PermStore {
    pub fn quant(&self, base: &str) -> Option<(&[u8], &[u8])> {
        let s = self.q.get(base)?;
        Some((&s.0, &s.1))
    }

    pub fn plain(&self, name: &str) -> Option<&[u8]> {
        self.p.get(name).map(|v| v.as_slice())
    }
}

fn vperm(cfg: &QwenCfg, i: usize) -> usize {
    let nk = cfg.linear_num_key_heads;
    let ratio = cfg.linear_num_value_heads / nk;
    ratio * (i % nk) + i / nk
}

fn plain_perm(store: &W4a16Model, cfg: &QwenCfg, name: &str, pv: PV) -> Option<Vec<u8>> {
    let e = store.entry(name)?;
    let es = match e.dtype {
        crate::st::StDtype::Bf16 | crate::st::StDtype::F16 => 2usize,
        crate::st::StDtype::F32 => 4,
        _ => return None,
    };
    let n = *e.shape.first()? as usize;
    let k: usize = e.shape[1..].iter().product::<u64>() as usize;
    let src = store.tensor_slice(name)?;
    let rb = k * es;
    if src.len() != n * rb || !n.is_multiple_of(128) || !k.is_multiple_of(128) {
        return None;
    }
    let mut d = vec![0u8; src.len()];
    match pv {
        PV::VPart | PV::AllN => {
            let vbase = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            let nb0 = if pv == PV::VPart { vbase / 128 } else { 0 };
            let nblk = n / 128;
            d[..nb0 * 128 * rb].copy_from_slice(&src[..nb0 * 128 * rb]);
            for b in 0..nblk - nb0 {
                let s = (nb0 + vperm(cfg, b)) * 128 * rb;
                let t = (nb0 + b) * 128 * rb;
                d[t..t + 128 * rb].copy_from_slice(&src[s..s + 128 * rb]);
            }
        }
        PV::AllK => {
            let nblk = k / 128;
            let blk = 128 * es;
            for r in 0..n {
                let row = &src[r * rb..(r + 1) * rb];
                let dst = &mut d[r * rb..(r + 1) * rb];
                for g in 0..nblk {
                    let s = vperm(cfg, g) * blk;
                    dst[g * blk..(g + 1) * blk].copy_from_slice(&row[s..s + blk]);
                }
            }
        }
        PV::None => {}
    }
    Some(d)
}

pub fn build_perm(store: &W4a16Model, cfg: &QwenCfg) -> PermStore {
    let group = store.group();
    let nk = cfg.linear_num_key_heads;
    let vbase = 2 * nk * cfg.linear_key_head_dim;
    let mut q = HashMap::new();
    let mut p = HashMap::new();
    let lp = "model.language_model.layers.";
    for il in 0..cfg.layers {
        if (il + 1).is_multiple_of(cfg.full_interval) {
            continue;
        }
        let l = format!("{lp}{il}.");
        for (suf, pv) in [
            ("linear_attn.in_proj_qkv", PV::VPart),
            ("linear_attn.in_proj_z", PV::AllN),
            ("linear_attn.out_proj", PV::AllK),
        ] {
            let base = format!("{l}{suf}");
            let Some((n, k)) = store.lin_shape(&base) else {
                let name = format!("{base}.weight");
                if let Some(out) = plain_perm(store, cfg, &name, pv) {
                    p.insert(name, out);
                }
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
                    let nblk = k / 128;
                    let mut scratch = vec![0u8; rb];
                    for r in 0..n {
                        scratch.copy_from_slice(&pk[r * rb..(r + 1) * rb]);
                        for g in 0..nblk {
                            let s = g * 64;
                            let sg = vperm(cfg, g) * 64;
                            d[r * rb + s..r * rb + s + 64].copy_from_slice(&scratch[sg..sg + 64]);
                        }
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

pub fn permute_heads_f32(cfg: &QwenCfg, v: &[f32]) -> Vec<f32> {
    (0..v.len()).map(|i| v[vperm(cfg, i)]).collect()
}

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

pub fn validate(store: &W4a16Model, cfg: &QwenCfg) -> R<Report> {
    let mut rep = Report {
        triples: store.n_lins(),
        ..Default::default()
    };
    rep.bad_shape_value = store.check_shape_values()?;
    if cfg.n_experts > 0 {
        for il in 0..cfg.layers {
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
    {
        let lp = "model.language_model.layers.";
        let mut items: Vec<(String, usize)> = Vec::new();
        for il in 0..cfg.layers {
            if (il + 1).is_multiple_of(cfg.full_interval) {
                continue;
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
        if name.starts_with("model.visual.") {
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
            rep.unknown_text.push(name.clone());
        }
    }
    Ok(rep)
}

fn expected_names(cfg: &QwenCfg) -> (Vec<String>, Vec<String>) {
    let (mut q, mut p) = (Vec::new(), Vec::new());
    let lp = "model.language_model.layers.";
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
            r#"{{"hidden_size":5120,"num_hidden_layers":4,"num_attention_heads":{heads},"num_key_value_heads":{kv},"head_dim":128,"intermediate_size":17408,"vocab_size":248320,"max_position_embeddings":262144,"linear_key_head_dim":128,"linear_num_key_heads":{nk},"linear_num_value_heads":{nv},"linear_conv_kernel_dim":{conv_k}}}"#
        )
    }

    #[test]
    fn cfg_accept_and_reject_cross_fields() {
        assert!(QwenCfg::parse(&cfg_text(40, 8, 16, 64, 4)).is_ok());
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

    #[test]
    fn final_logit_softcap_from_config() {
        let dir = std::env::temp_dir().join(format!("llm170_softcap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"text_config":{"final_logit_softcapping":30.0}}"#,
        )
        .unwrap();
        assert_eq!(final_logit_softcap(&dir), 30.0, "중첩(text_config)");
        std::fs::write(
            dir.join("config.json"),
            r#"{"final_logit_softcapping":12.5}"#,
        )
        .unwrap();
        assert_eq!(final_logit_softcap(&dir), 12.5, "평면 키");
        std::fs::write(dir.join("config.json"), r#"{"hidden_size":1}"#).unwrap();
        assert_eq!(final_logit_softcap(&dir), 0.0, "부재 = 미적용");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cfg_max_pos_parse() {
        assert_eq!(
            QwenCfg::parse(&cfg_text(40, 8, 16, 64, 4)).unwrap().max_pos,
            262144
        );
        let nested = r#"{"text_config":{"hidden_size":5120,"num_hidden_layers":4,
            "num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,
            "intermediate_size":17408,"vocab_size":248320,
            "max_position_embeddings":131072,"linear_key_head_dim":128,
            "linear_num_key_heads":16,"linear_num_value_heads":64,
            "linear_conv_kernel_dim":4}}"#;
        assert_eq!(QwenCfg::parse(nested).unwrap().max_pos, 131072);
        let miss = cfg_text(40, 8, 16, 64, 4).replace(r#","max_position_embeddings":262144"#, "");
        assert!(
            QwenCfg::parse(&miss)
                .unwrap_err()
                .to_string()
                .contains("max_position_embeddings")
        );
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
            max_pos: 262144,
            n_experts: 0,
            top_k: 0,
            moe_ffn: 0,
            shared_ffn: 0,
        }
    }

    #[test]
    fn perm_align_rules() {
        let cfg = cfg_fixture();
        let vbase = 2 * 16 * 128;
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
        let bad = vec![(
            "model.language_model.layers.0.linear_attn.in_proj_qkv".to_string(),
            vbase + 6144 + 64,
        )];
        let e = perm_align_check(&cfg, &bad).unwrap_err().to_string();
        assert!(e.contains("순열 정렬 위반"), "{e}");
        let bad2 = vec![(
            "model.language_model.layers.0.conv1d".to_string(),
            vbase - 128,
        )];
        assert!(perm_align_check(&cfg, &bad2).is_err());
    }

    #[test]
    fn vperm_bijection_and_interleave() {
        let mk = |nk: usize, nv: usize| {
            let mut c = cfg_fixture();
            c.linear_num_key_heads = nk;
            c.linear_num_value_heads = nv;
            c
        };
        for (nk, nv) in [(16usize, 48usize), (16, 32), (8, 16), (16, 16)] {
            let c = mk(nk, nv);
            let mut img: Vec<usize> = (0..nv).map(|i| vperm(&c, i)).collect();
            img.sort_unstable();
            assert_eq!(img, (0..nv).collect::<Vec<_>>(), "nk={nk} nv={nv}");
        }
        let c1 = mk(16, 16);
        assert!((0..16).all(|i| vperm(&c1, i) == i));
        let c27 = mk(16, 48);
        assert_eq!(
            (vperm(&c27, 0), vperm(&c27, 16), vperm(&c27, 32)),
            (0, 1, 2)
        );
        let v: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let out = permute_heads_f32(&c27, &v);
        assert_eq!(out.len(), 48);
        for i in 0..48 {
            assert_eq!(out[i], vperm(&c27, i) as f32);
        }
    }
}
