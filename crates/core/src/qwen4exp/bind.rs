
use crate::json::Json;
use std::path::Path;

pub const ARCHES: &[&str] = &["Qwen4ExpForConditionalGeneration", "qwen4_exp"];

pub fn arch_ok(arch: &str) -> bool {
    ARCHES.contains(&arch)
}

pub fn dir_arch_ok(arch: &str) -> bool {
    arch_ok(arch)
}

#[derive(Clone)]
pub struct Qwen4ExpCfg {
    pub num_hidden_layers: usize,
    pub full_interval: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub heads: usize,
    pub hidden_size: usize,
    pub vocab: usize,
    pub linear_value_heads: usize,
    pub linear_head_dim: usize,
    pub n_linear: usize,
    pub n_full: usize,
    pub max_pos: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub output_gate_sigmoid: bool,
    pub shared_ffn: usize,
    pub moe_ffn: usize,
    pub hc_lowrank: usize,
    pub rms_norm_eps: f32,
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ngram_vocab_base: u64,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub split_ngram_parts: usize,
    pub mtp_cfg_layers: usize,
    pub eos_token_id: u32,
    pub hc_count: usize,
    pub indexer_n_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
}

impl Qwen4ExpCfg {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let v = Json::parse(&txt).map_err(|e| format!("config.json 파싱: {e}"))?;
        let t = v
            .get("text_config")
            .ok_or("text_config 부재 — qwen4_exp 계약 위반")?;
        let num = |k: &str| -> Result<usize, String> {
            t.get(k)
                .and_then(Json::as_f64)
                .map(|x| x as usize)
                .ok_or(format!("text_config.{k} 부재/비수"))
        };
        let num_opt = |k: &str, dflt: usize| -> usize {
            t.get(k).and_then(Json::as_f64).map_or(dflt, |x| x as usize)
        };
        let layers = num("num_hidden_layers")?;
        let full_interval = num("full_attention_interval")?;
        let mut n_linear = layers - layers / full_interval.max(1);
        let mut n_full = layers / full_interval.max(1);
        if let Some(Json::Arr(items)) = t.get("layer_types")
            && items.len() == layers
        {
            let lf = items
                .iter()
                .filter(|j| j.as_str() == Some("linear_attention"))
                .count();
            let ff = items
                .iter()
                .filter(|j| j.as_str() == Some("full_attention"))
                .count();
            if lf + ff == layers {
                n_linear = lf;
                n_full = ff;
            }
        }
        let ple_layer_ids = match t.get("ple_layer_ids") {
            Some(Json::Arr(items)) => items
                .iter()
                .filter_map(|j| j.as_f64().map(|x| x as usize))
                .collect(),
            _ => Vec::new(),
        };
        Ok(Qwen4ExpCfg {
            num_hidden_layers: layers,
            full_interval,
            kv_heads: num("num_key_value_heads")?,
            head_dim: num("head_dim")?,
            heads: num("num_attention_heads")?,
            hidden_size: num("hidden_size")?,
            vocab: num("vocab_size")?,
            linear_value_heads: num("linear_num_value_heads")?,
            linear_head_dim: num("linear_value_head_dim")?,
            n_linear,
            n_full,
            max_pos: num("max_position_embeddings")?,
            n_experts: num("num_experts")?,
            top_k: num("num_experts_per_tok")?,
            shared_ffn: num_opt("shared_expert_intermediate_size", 0),
            moe_ffn: num_opt("moe_intermediate_size", 0),
            hc_lowrank: num_opt("hc_lowrank", 0),
            rms_norm_eps: t.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            ple_layer_ids,
            ple_embed_dim: num_opt("ple_embed_dim", 0),
            ngram_vocab_base: t
                .get("ngram_vocab_size_base")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as u64,
            ngram_size: num_opt("ngram_size", 0),
            heads_per_ngram: num_opt("heads_per_ngram", 0),
            split_ngram_parts: num_opt("split_ngram_parts", 0),
            eos_token_id: t
                .get("eos_token_id")
                .and_then(Json::as_f64)
                .map_or(0, |x| x as u32),
            hc_count: num_opt("hc_count", 4),
            indexer_n_heads: num_opt("indexer_n_heads", 0),
            indexer_head_dim: num_opt("indexer_head_dim", 0),
            indexer_budget: num_opt("indexer_budget", 0),
            indexer_compress_ratio: num_opt("indexer_compress_ratio", 0),
            output_gate_sigmoid: t
                .get("output_gate_type")
                .and_then(Json::as_str)
                .is_some_and(|s| s == "sigmoid"),
            mtp_cfg_layers: num_opt("mtp_num_hidden_layers", 0),
        })
    }
}

pub fn ctx_info(dir: &Path) -> Result<crate::qwen35::CtxInfo, String> {
    let c = Qwen4ExpCfg::load(dir)?;
    let fi = c.full_interval.max(1) as u64;
    let layers = c.num_hidden_layers as u64;
    let full = layers.div_ceil(fi);
    let kv = full * c.kv_heads as u64 * c.head_dim as u64 * 2;
    let sc = full * c.kv_heads as u64 * 2 * 4;
    let n_gdn = layers - layers / fi;
    let hv = c.linear_value_heads as u64;
    let hd = c.linear_head_dim as u64;
    Ok(crate::qwen35::CtxInfo {
        ctx: c.max_pos,
        gdn_state_bytes_per_slot: n_gdn * hv * hd * hd * 2 + n_gdn * hv * hd * 4,
        kv_bytes_per_token: kv + sc,
        dense: c.n_experts == 0,
    })
}

struct Skeleton {
    ple_layers: Vec<usize>,
    ngram_shards: usize,
    mtp_tensors: usize,
}

fn skeleton(dir: &Path) -> Result<Skeleton, String> {
    let txt = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .map_err(|e| format!("model.safetensors.index.json: {e}"))?;
    let v = Json::parse(&txt).map_err(|e| format!("index 파싱: {e}"))?;
    let wm = v.get("weight_map").ok_or("weight_map 부재")?;
    let mut ple_layers = Vec::new();
    let mut ngram_shards = 0;
    let mut mtp_tensors = 0;
    if let Json::Obj(items) = wm {
        for (name, _) in items {
            if name.contains(".ple.") {
                if let Some(idx) = name
                    .strip_prefix("model.language_model.layers.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
                    && !ple_layers.contains(&idx)
                {
                    ple_layers.push(idx);
                }
            }
            if name.contains("ngram_embedding.shard_") {
                ngram_shards += 1;
            }
            if name.contains(".mtp.") {
                mtp_tensors += 1;
            }
        }
    }
    ple_layers.sort_unstable();
    Ok(Skeleton {
        ple_layers,
        ngram_shards,
        mtp_tensors,
    })
}

pub fn report(dir: &Path) -> Result<String, String> {
    let c = Qwen4ExpCfg::load(dir)?;
    let ci = ctx_info(dir)?;
    let sk = skeleton(dir)?;
    let ple = if c.ple_layer_ids.is_empty() {
        "없음".to_string()
    } else {
        format!(
            "배치 config={} · 텐서 층(0-based)={:?} · ngram_embedding shard {}/{}",
            c.ple_layer_ids.len(),
            sk.ple_layers,
            sk.ngram_shards,
            c.split_ngram_parts
        )
    };
    let mtp = if sk.mtp_tensors == 0 {
        format!(
            "텐서 미포함(0개) — 엔진 n-gram 초안 폴백 (config 기술 {}층)",
            c.mtp_cfg_layers
        )
    } else {
        format!("텐서 {}개 포함", sk.mtp_tensors)
    };
    Ok(format!(
        "w4a16-load {}\n  arch=qwen4_exp(FN) hidden={} layers={} heads={}/{} head_dim={} vocab={} ctx={}\n  하이브리드: linear_attn {} / full_attn {} (interval={}) · linear상태={}×{}×{} · kv_bytes/tok={} · gdn_state/slot={}\n  MoE: {}전문가 top-{} · shared ffn={} \n  PLE: {} (embed={} · ngram vocab={} size={} heads={} shards={})\n  MTP: {}\n  판정: FN 1단계 식별 통과 — packed W4A16 검증·엔진 경로는 본 트랙",
        dir.display(),
        c.hidden_size,
        c.num_hidden_layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.vocab,
        c.max_pos,
        c.n_linear,
        c.n_full,
        c.full_interval,
        c.linear_value_heads,
        c.linear_head_dim,
        c.linear_head_dim,
        ci.kv_bytes_per_token,
        ci.gdn_state_bytes_per_slot,
        c.n_experts,
        c.top_k,
        c.shared_ffn,
        ple,
        c.ple_embed_dim,
        c.ngram_vocab_base,
        c.ngram_size,
        c.heads_per_ngram,
        c.split_ngram_parts,
        mtp,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctx_info_from_synthetic_config() {
        let d = std::env::temp_dir().join(format!("q4e-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("config.json"),
            r#"{"architectures":["Qwen4ExpForConditionalGeneration"],
               "text_config":{"num_hidden_layers":48,"full_attention_interval":4,
               "num_key_value_heads":2,"head_dim":256,"num_attention_heads":24,
               "output_gate_type":"sigmoid",
               "hidden_size":2560,"vocab_size":248320,
               "linear_num_value_heads":48,"linear_value_head_dim":128,
               "max_position_embeddings":262144,"num_experts":512,
               "num_experts_per_tok":10,"shared_expert_intermediate_size":640,
               "ple_layer_ids":[2],"ple_embed_dim":2560,
               "ngram_vocab_size_base":20000000,"ngram_size":3,
               "heads_per_ngram":8,"split_ngram_parts":128,
               "mtp_num_hidden_layers":1,
               "layer_types":["linear_attention","linear_attention",
                 "linear_attention","full_attention"]}}"#,
        )
        .unwrap();
        let ci = ctx_info(&d).unwrap();
        assert_eq!(ci.ctx, 262144);
        assert_eq!(ci.kv_bytes_per_token, 12_288 + 192);
        let st = 36 * 48u64 * 128 * 128 * 2 + 36 * 48u64 * 128 * 4;
        assert_eq!(ci.gdn_state_bytes_per_slot, st);
        assert!(!ci.dense, "512 전문가 = MoE");
        let c = Qwen4ExpCfg::load(&d).unwrap();
        assert_eq!((c.n_linear, c.n_full), (36, 12));
        assert_eq!(c.ple_layer_ids, vec![2]);
        assert_eq!(c.split_ngram_parts, 128);
        assert_eq!(c.mtp_cfg_layers, 1);
        assert!(c.output_gate_sigmoid);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn arch_detection() {
        assert!(arch_ok("Qwen4ExpForConditionalGeneration"));
        assert!(arch_ok("qwen4_exp"));
        assert!(!arch_ok("qwen3_5"));
    }
}
