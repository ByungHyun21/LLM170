//! DeepSeek-V4-Flash 설정 — config.json → `Deepseek4Config`.
//!
//! 변형별 상수(보고서 §8): `rms_norm_eps`는 **적재한 config 값 그대로** 쓴다.
//! EXL3 Vision-Exp LLM 타워 = 1e-20, 0731-GGUF = 1e-6. hc_eps(싱크홀른·
//! 시그모이드 eps, 1e-6)는 별개 상수다.
//!
//! 층 지도(보고서 §9.7): `compress_ratios[il]` 0=SWA(윈도우 128, base 1e4,
//! YaRN 없음) / 4=CSA(겹침 압축+인덱서, base 160000+YaRN) / 128=HCA(비겹침
//! 밀도). 그 외 값은 로더가 거부한다(0731 규격 고정). 해시 층 = L0..n_hash-1.

use llm170_exl3::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// ratio 0 — 순수 슬라이딩 윈도우(128), rope base 10000, YaRN 없음.
    Swa,
    /// ratio 4 — 겹침 압축(CSA) + 인덱서 top-512, rope base 160000 + YaRN.
    Csa,
    /// ratio 128 — 비겹침 압축(HCA), 압축 엔트리 전체 밀도 어텐션.
    Hca,
}

/// config.json 하이퍼파라미터 — 필드명은 config 키와 대응.
#[derive(Debug, Clone)]
pub struct Deepseek4Config {
    pub n_layers: usize,
    pub dim: usize,
    pub vocab: usize,
    /// 적재한 config의 rms_norm_eps (Vision-Exp LLM 타워 1e-20).
    pub rms_eps: f32,
    /// hc 싱크홀른/시그모이드 eps (1e-6, RMSNorm eps와 별개).
    pub hc_eps: f32,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window: usize,
    /// 길이 ≥ n_layers + n_mtp_layers (본 모델 46 = 43 + 3).
    pub compress_ratios: Vec<i64>,
    pub compress_rope_theta: f64,
    pub rope_theta: f64,
    // YaRN (rope_scaling.type == "yarn"일 때만 활성)
    pub yarn_factor: f64,
    pub yarn_orig_len: usize,
    pub yarn_beta_fast: f64,
    pub yarn_beta_slow: f64,
    // 인덱서 (CSA 층만)
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    // MoE
    pub n_routed: usize,
    pub n_shared: usize,
    pub n_activated: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub n_hash_layers: usize,
    // DSpark MTP
    pub n_mtp_layers: usize,
    pub dspark_block: usize,
    pub dspark_noise_token: u32,
    pub dspark_targets: Vec<usize>,
    pub dspark_markov_rank: usize,
}

impl Deepseek4Config {
    /// 층 il의 압축 비율 (본체 0..n_layers, MTP 스테이지는 n_layers+stage).
    pub fn ratio(&self, il: usize) -> usize {
        self.compress_ratios[il] as usize
    }

    /// 층 유형 — 0/4/128 외 거부 (0731 규격 고정).
    pub fn kind(&self, il: usize) -> LayerKind {
        match self.ratio(il) {
            0 => LayerKind::Swa,
            4 => LayerKind::Csa,
            128 => LayerKind::Hca,
            other => panic!("compress_ratios[{il}]={other}: 0/4/128 외 미지원"),
        }
    }

    /// L0..n_hash_layers-1 — tid2eid 표 라우팅 MoE (보고서 §3).
    pub fn is_hash(&self, il: usize) -> bool {
        il < self.n_hash_layers
    }

    /// YaRN 활성 여부 — 압축 층(CSA/HCA)만. SWA는 base 10000 단독(보고서 §6).
    pub fn yarn(&self, il: usize) -> bool {
        self.ratio(il) != 0
    }

    /// 층별 rope base — 압축 160000 / 윈도우 10000.
    pub fn rope_base(&self, il: usize) -> f64 {
        if self.ratio(il) != 0 {
            self.compress_rope_theta
        } else {
            self.rope_theta
        }
    }

    /// config.json 본문에서 구축. exl3 커스텀 Json 파서 사용(serde 금지 규율).
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v = Json::parse(text).map_err(|e| format!("config.json 파싱: {e}"))?;
        let g = |k: &str| -> Option<f64> { v.get(k).and_then(Json::as_f64) };
        let u = |k: &str| -> Result<usize, String> {
            g(k).map(|x| x as usize)
                .filter(|&x| x as f64 == g(k).unwrap_or(f64::NAN))
                .ok_or_else(|| format!("config: {k} 없음/비정수"))
        };
        let ratios = v
            .get("compress_ratios")
            .and_then(Json::as_num_array)
            .ok_or("config: compress_ratios 없음")?
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let rope_scaling = v.get("rope_scaling");
        let (yarn_factor, yarn_orig_len, beta_fast, beta_slow) = match rope_scaling
            .and_then(|r| r.get("type"))
            .and_then(Json::as_str)
        {
            Some("yarn") => (
                rope_scaling
                    .and_then(|r| r.get("factor"))
                    .and_then(Json::as_f64)
                    .unwrap_or(1.0),
                rope_scaling
                    .and_then(|r| r.get("original_max_position_embeddings"))
                    .and_then(Json::as_f64)
                    .unwrap_or(0.0) as usize,
                rope_scaling
                    .and_then(|r| r.get("beta_fast"))
                    .and_then(Json::as_f64)
                    .unwrap_or(32.0),
                rope_scaling
                    .and_then(|r| r.get("beta_slow"))
                    .and_then(Json::as_f64)
                    .unwrap_or(1.0),
            ),
            _ => (1.0, 0, 32.0, 1.0),
        };
        let targets = v
            .get("dspark_target_layer_ids")
            .and_then(Json::as_num_array)
            .map(|a| a.iter().map(|&x| x as usize).collect())
            .unwrap_or_default();
        let cfg = Deepseek4Config {
            n_layers: u("num_hidden_layers")?,
            dim: u("hidden_size")?,
            vocab: u("vocab_size")?,
            rms_eps: g("rms_norm_eps").unwrap_or(1e-6) as f32,
            hc_eps: g("hc_eps").unwrap_or(1e-6) as f32,
            hc_mult: u("hc_mult").unwrap_or(1),
            hc_sinkhorn_iters: u("hc_sinkhorn_iters").unwrap_or(20),
            n_heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            rope_head_dim: g("qk_rope_head_dim")
                .map(|x| x as usize)
                .ok_or("config: qk_rope_head_dim 없음")?,
            q_lora_rank: u("q_lora_rank")?,
            o_lora_rank: g("o_lora_rank")
                .map(|x| x as usize)
                .ok_or("config: o_lora_rank 없음")?,
            o_groups: g("o_groups")
                .map(|x| x as usize)
                .ok_or("config: o_groups 없음")?,
            window: g("sliding_window")
                .map(|x| x as usize)
                .ok_or("config: sliding_window 없음")?,
            compress_ratios: ratios,
            compress_rope_theta: g("compress_rope_theta").unwrap_or(160000.0),
            rope_theta: g("rope_theta").unwrap_or(10000.0),
            yarn_factor,
            yarn_orig_len,
            yarn_beta_fast: beta_fast,
            yarn_beta_slow: beta_slow,
            index_n_heads: u("index_n_heads")?,
            index_head_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
            n_routed: g("n_routed_experts")
                .map(|x| x as usize)
                .ok_or("config: n_routed_experts 없음")?,
            n_shared: g("n_shared_experts")
                .map(|x| x as usize)
                .ok_or("config: n_shared_experts 없음")?,
            n_activated: g("num_experts_per_tok")
                .map(|x| x as usize)
                .ok_or("config: num_experts_per_tok 없음")?,
            moe_inter: g("moe_intermediate_size")
                .map(|x| x as usize)
                .ok_or("config: moe_intermediate_size 없음")?,
            route_scale: g("routed_scaling_factor").unwrap_or(1.0) as f32,
            swiglu_limit: g("swiglu_limit").unwrap_or(0.0) as f32,
            n_hash_layers: g("num_hash_layers").map(|x| x as usize).unwrap_or(0),
            n_mtp_layers: g("num_nextn_predict_layers")
                .map(|x| x as usize)
                .unwrap_or(0),
            dspark_block: g("dspark_block_size").map(|x| x as usize).unwrap_or(0),
            dspark_noise_token: g("dspark_noise_token_id").map(|x| x as u32).unwrap_or(0),
            dspark_targets: targets,
            dspark_markov_rank: g("dspark_markov_rank").map(|x| x as usize).unwrap_or(0),
        };
        // compress_ratios는 본체+MTP 스테이지 길이 이상 (0731 꼬리 0 — 보고서 §9.7).
        if cfg.compress_ratios.len() < cfg.n_layers + cfg.n_mtp_layers {
            return Err(format!(
                "compress_ratios 길이 {} < {} (본체+MTP)",
                cfg.compress_ratios.len(),
                cfg.n_layers + cfg.n_mtp_layers
            ));
        }
        for (il, &r) in cfg.compress_ratios.iter().enumerate() {
            if !matches!(r, 0 | 4 | 128) {
                return Err(format!("compress_ratios[{il}]={r}: 0/4/128 외 거부"));
            }
        }
        Ok(cfg)
    }
}

/// 테스트 공용 소형 cfg — 실측 비율(4/128)은 유지, 차원만 축소.
#[cfg(test)]
pub(crate) fn test_cfg() -> Deepseek4Config {
    Deepseek4Config {
        n_layers: 3,
        dim: 8,
        vocab: 16,
        rms_eps: 1e-6,
        hc_eps: 1e-6,
        hc_mult: 2,
        hc_sinkhorn_iters: 20,
        n_heads: 2,
        head_dim: 4,
        rope_head_dim: 2,
        q_lora_rank: 4,
        o_lora_rank: 4,
        o_groups: 2,
        window: 4,
        compress_ratios: vec![0, 4, 128],
        compress_rope_theta: 160000.0,
        rope_theta: 10000.0,
        yarn_factor: 16.0,
        yarn_orig_len: 65536,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        index_n_heads: 2,
        index_head_dim: 2,
        index_topk: 2,
        n_routed: 4,
        n_shared: 1,
        n_activated: 3,
        moe_inter: 8,
        route_scale: 1.5,
        swiglu_limit: 10.0,
        n_hash_layers: 1,
        n_mtp_layers: 0,
        dspark_block: 0,
        dspark_noise_token: 0,
        dspark_targets: vec![],
        dspark_markov_rank: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 실측 config(Vision-Exp EXL3)의 필드를 재현한 표본 — 층 지도·변형 상수 검증.
    #[test]
    fn layer_map_from_ratios() {
        let mut ratios = vec![0i64, 0];
        for i in 0..41 {
            ratios.push(if i % 2 == 0 { 4 } else { 128 });
        }
        ratios.extend([0, 0, 0]); // MTP 꼬리
        assert_eq!(ratios.len(), 46);
        let cfg = Deepseek4Config {
            n_layers: 43,
            dim: 4096,
            vocab: 129280,
            rms_eps: 1e-20,
            hc_eps: 1e-6,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            n_heads: 64,
            head_dim: 512,
            rope_head_dim: 64,
            q_lora_rank: 1024,
            o_lora_rank: 1024,
            o_groups: 8,
            window: 128,
            compress_ratios: ratios,
            compress_rope_theta: 160000.0,
            rope_theta: 10000.0,
            yarn_factor: 16.0,
            yarn_orig_len: 65536,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            index_n_heads: 64,
            index_head_dim: 128,
            index_topk: 512,
            n_routed: 256,
            n_shared: 1,
            n_activated: 6,
            moe_inter: 2048,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            n_hash_layers: 3,
            n_mtp_layers: 3,
            dspark_block: 5,
            dspark_noise_token: 128799,
            dspark_targets: vec![40, 41, 42],
            dspark_markov_rank: 256,
        };
        // 층 지도: L0-1 SWA+해시, L2 CSA+해시, L3 HCA, L42 CSA(짝수).
        assert_eq!(cfg.kind(0), LayerKind::Swa);
        assert_eq!(cfg.kind(1), LayerKind::Swa);
        assert_eq!(cfg.kind(2), LayerKind::Csa);
        assert_eq!(cfg.kind(3), LayerKind::Hca);
        assert_eq!(cfg.kind(42), LayerKind::Csa);
        assert!(cfg.is_hash(0) && cfg.is_hash(2) && !cfg.is_hash(3));
        // rope base 선택 (보고서 §6): 압축층 160000+YaRN, L0-1은 10000 단독.
        assert_eq!(cfg.rope_base(0), 10000.0);
        assert_eq!(cfg.rope_base(2), 160000.0);
        assert!(!cfg.yarn(1) && cfg.yarn(2) && cfg.yarn(3));
        // MTP 스테이지 ratio 0 (DSpark 윈도우 강제).
        assert_eq!(cfg.kind(43), LayerKind::Swa);
        // 인덱서 유무는 kind(CSA)로 판정 — 홀수 층·SWA 없음.
        assert_ne!(cfg.kind(5), LayerKind::Csa);
    }

    /// from_json — 실 config.json 문자열(요약) 파싱 + 거부 규칙.
    #[test]
    fn from_json_parses_real_shape() {
        let mut ratios = String::from("[0,0");
        for i in 0..41 {
            ratios.push_str(if i % 2 == 0 { ",4" } else { ",128" });
        }
        ratios.push_str(",0,0,0]");
        let text = format!(
            r#"{{
            "hidden_size": 4096, "vocab_size": 129280, "num_hidden_layers": 43,
            "rms_norm_eps": 1e-20, "hc_eps": 1e-06, "hc_mult": 4,
            "hc_sinkhorn_iters": 20, "num_attention_heads": 64, "head_dim": 512,
            "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_lora_rank": 1024,
            "o_groups": 8, "sliding_window": 128, "compress_ratios": {ratios},
            "compress_rope_theta": 160000.0, "rope_theta": 10000,
            "rope_scaling": {{ "beta_fast": 32, "beta_slow": 1, "factor": 16,
                "original_max_position_embeddings": 65536, "type": "yarn" }},
            "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
            "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
            "routed_scaling_factor": 1.5, "swiglu_limit": 10.0,
            "num_hash_layers": 3, "num_nextn_predict_layers": 3,
            "dspark_block_size": 5, "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40,41,42], "dspark_markov_rank": 256
        }}"#
        );
        let cfg = Deepseek4Config::from_json(&text).expect("parse");
        assert_eq!(cfg.rms_eps, 1e-20); // 변형 상수: 적재값 그대로 (보고서 §8)
        assert_eq!(cfg.hc_eps, 1e-6);
        assert_eq!((cfg.n_layers, cfg.n_mtp_layers), (43, 3));
        assert_eq!(cfg.kind(2), LayerKind::Csa);
        assert_eq!((cfg.yarn_factor, cfg.yarn_orig_len), (16.0, 65536));
        // 0/4/128 외 거부
        let bad = text.replace("[0,0,", "[0,3,");
        assert!(Deepseek4Config::from_json(&bad).is_err());
        // 길이 부족 거부
        let short = text.replace(",0,0,0]", "]");
        assert!(Deepseek4Config::from_json(&short).is_err());
    }
}
