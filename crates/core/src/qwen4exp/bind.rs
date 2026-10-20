//! qwen4_exp config 판독 — `text_config` 중첩 hparams만 (전체 로드 없음).
//! [2.J] serve의 ctx 파생·기동 가드 입력 계약(qwen35::CtxInfo 재사용).

use crate::json::Json;
use std::path::Path;

/// arch 식별자 — `architectures[0]`(정식)와 `model_type`(폴백) 양쪽.
pub const ARCHES: &[&str] = &["Qwen4ExpForConditionalGeneration", "qwen4_exp"];

pub fn arch_ok(arch: &str) -> bool {
    ARCHES.contains(&arch)
}

/// dir_arch 결과가 qwen4_exp 계열인지(호출부 디스패치용).
pub fn dir_arch_ok(arch: &str) -> bool {
    arch_ok(arch)
}

/// text_config 산출물 — ctx_info + w4a16-load 브링업 리포트 전항.
/// [FN 1단계 2026-10-17] hidden·heads·하이브리드 층 분포·PLE/ngram hparams
/// 추가 판독(2026-10-17 config 실측 계약 반영).
pub struct Qwen4ExpCfg {
    pub num_hidden_layers: usize,
    /// full_attention_interval — layer_types [linear×(n-1), full×1] 반복.
    pub full_interval: usize,
    /// full층 KV 헤드 수(num_key_value_heads).
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Q 헤드 수(num_attention_heads).
    pub heads: usize,
    pub hidden_size: usize,
    pub vocab: usize,
    /// linear(GDN류) 상태 축 — 값 헤드 × (키차원 × 값차원).
    pub linear_value_heads: usize,
    pub linear_head_dim: usize,
    /// linear_attn / full_attn 층 수(layer_types 실측, 부재 시 interval 산식).
    pub n_linear: usize,
    pub n_full: usize,
    pub max_pos: usize,
    pub n_experts: usize,
    pub top_k: usize,
    /// GDN 출력게이트 활성화(config output_gate_type — "sigmoid"면 참,
    /// vLLM getattr 기본값 "silu"). qwen4exp(FN) 실측 = sigmoid(논문 Eq.11).
    pub output_gate_sigmoid: bool,
    /// shared_expert_intermediate_size.
    pub shared_ffn: usize,
    /// PLE 배치 층(config ple_layer_ids — 체크포인트 0-based 인덱스와 1 차이).
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    /// n-gram PLE 임베딩 hparams — 샤드 수 대조 판별용.
    pub ngram_vocab_base: u64,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub split_ngram_parts: usize,
    /// config MTP 층 수(기록만 — 체크포인트 미포함, 엔진 n-gram 초안 폴백).
    pub mtp_cfg_layers: usize,
}

impl Qwen4ExpCfg {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let v = Json::parse(&txt).map_err(|e| format!("config.json 파싱: {e}"))?;
        // hparams는 text_config 중첩(gemma4 동일 패턴) — 평면 키 폴백 없음
        // (FN 체크포인트 실측 계약). vision_config는 스킵(텍스트 엔진 전용).
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
        // 층 분포 = layer_types 실측 우선(부재/불일치 시 interval 산식 폴백).
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
            ple_layer_ids,
            ple_embed_dim: num_opt("ple_embed_dim", 0),
            ngram_vocab_base: t
                .get("ngram_vocab_size_base")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as u64,
            ngram_size: num_opt("ngram_size", 0),
            heads_per_ngram: num_opt("heads_per_ngram", 0),
            split_ngram_parts: num_opt("split_ngram_parts", 0),
            output_gate_sigmoid: t
                .get("output_gate_type")
                .and_then(Json::as_str)
                .map_or(false, |s| s == "sigmoid"),
            mtp_cfg_layers: num_opt("mtp_num_hidden_layers", 0),
        })
    }
}

/// serve ctx 파생·기동 가드 입력 — qwen35::CtxInfo와 동일 항 산식.
/// KV = full층 × kv헤드 × head_dim × 2B(int8 K+V) + 행 스케일(4B×2/헤드).
/// GDN류 상태 = linear층 × 값헤드 × (head_dim)² × 2B + 행 스케일 — qwen35
/// 산식 미러(hv×128×128 — FN도 128로 동일하나 cfg 차원으로 일반화).
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

/// 모델 계층 스켈레톤 — model.safetensors.index.json weight_map 키 스캔.
/// [FN 1단계 2026-10-17] PLE/ngram 임베딩·MTP 헤드 실물 존재 판별
/// (AGENTS.md 'MTP 초안: 등록 게이트' — config 기술과 체크포인트 실물 대조).
struct Skeleton {
    /// ple 텐서를 가진 층(0-based 인덱스 — config ple_layer_ids와 1 차이).
    ple_layers: Vec<usize>,
    /// ngram_embedding.shard_N 텐서 수(split_ngram_parts 대조).
    ngram_shards: usize,
    /// mtp 텐서 수(FN 실측 = 0 — config는 1층 기술하나 미탑재).
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
                // model.language_model.layers.<N>.ple.* — N 추출(0-based).
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

/// w4a16-load FN 리포트 — 1단계(로더 식별) 출력. hparams 전항 + ctx_info
/// 산출 + 스켈레톤(PLE/ngram/MTP) 판별. packed W4A16 검증·엔진 진입은
/// 본 트랙(FN은 auto_round:auto_gptq 팩 — 트리플 구조 아님).
/// [FN 1단계 2026-10-17] qwen35/gemma4 w4a16-load 패턴 준수.
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

    /// 최소 합성 config → ctx_info 산식 검증(FN 실측 차원 대입).
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
        // full = ceil(48/4) = 12 · KV = 12·2·256·2 = 12,288 + 스케일 192.
        assert_eq!(ci.ctx, 262144);
        assert_eq!(ci.kv_bytes_per_token, 12_288 + 192);
        // linear = 48 − 12 = 36층 · 상태 = 36·48·128·128·2 + 36·48·128·4.
        let st = 36 * 48u64 * 128 * 128 * 2 + 36 * 48u64 * 128 * 4;
        assert_eq!(ci.gdn_state_bytes_per_slot, st);
        assert!(!ci.dense, "512 전문가 = MoE");
        // [FN 1단계] 확장 판독 — layer_types 길이 불일치(4≠48)면 interval
        // 산식 폴백 36/12, ple/ngram/mtp hparams 그대로 통과.
        let c = Qwen4ExpCfg::load(&d).unwrap();
        assert_eq!((c.n_linear, c.n_full), (36, 12));
        assert_eq!(c.ple_layer_ids, vec![2]);
        assert_eq!(c.split_ngram_parts, 128);
        assert_eq!(c.mtp_cfg_layers, 1);
        // [D0] GDN 출력게이트 판독 — FN 실측 sigmoid(기본 silu 폴백 별도).
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
