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

/// text_config 산출물(브링업 트랙이 확장 — 여기선 ctx_info에 필요한 최소).
pub struct Qwen4ExpCfg {
    pub num_hidden_layers: usize,
    /// full_attention_interval — layer_types [linear×(n-1), full×1] 반복.
    pub full_interval: usize,
    /// full층 KV 헤드 수(num_key_value_heads).
    pub kv_heads: usize,
    pub head_dim: usize,
    /// linear(GDN류) 상태 축 — 값 헤드 × (키차원 × 값차원).
    pub linear_value_heads: usize,
    pub linear_head_dim: usize,
    pub max_pos: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

impl Qwen4ExpCfg {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let v = Json::parse(&txt).map_err(|e| format!("config.json 파싱: {e}"))?;
        // hparams는 text_config 중첩(gemma4 동일 패턴) — 평면 키 폴백 없음
        // (FN 체크포인트 실측 계약).
        let t = v
            .get("text_config")
            .ok_or("text_config 부재 — qwen4_exp 계약 위반")?;
        let num = |k: &str| -> Result<usize, String> {
            t.get(k)
                .and_then(Json::as_f64)
                .map(|x| x as usize)
                .ok_or(format!("text_config.{k} 부재/비수"))
        };
        Ok(Qwen4ExpCfg {
            num_hidden_layers: num("num_hidden_layers")?,
            full_interval: num("full_attention_interval")?,
            kv_heads: num("num_key_value_heads")?,
            head_dim: num("head_dim")?,
            linear_value_heads: num("linear_num_value_heads")?,
            linear_head_dim: num("linear_value_head_dim")?,
            max_pos: num("max_position_embeddings")?,
            n_experts: num("num_experts")?,
            top_k: num("num_experts_per_tok")?,
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
               "num_key_value_heads":2,"head_dim":256,
               "linear_num_value_heads":48,"linear_value_head_dim":128,
               "max_position_embeddings":262144,"num_experts":512,
               "num_experts_per_tok":10}}"#,
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
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn arch_detection() {
        assert!(arch_ok("Qwen4ExpForConditionalGeneration"));
        assert!(arch_ok("qwen4_exp"));
        assert!(!arch_ok("qwen3_5"));
    }
}
