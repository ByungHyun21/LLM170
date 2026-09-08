//! 원시 디코더(rawhip/VkDecoder) 주입용 가중치·상수 명세 — 모델 지식의
//! 단일 소스. server/main.rs에서 이관(plans/35 P4): 백엔드 주입
//! (backend-gpu::{rawhip,rawvk}::inject)과 bench·engine이 공유한다.

use crate::model::Engine;

/// 원시 디코더가 필요한 가중치·상수 이름 목록 (recr/full층별 + MTP).
pub fn raw_names(eng: &Engine) -> (Vec<String>, Vec<String>) {
    let hp = &eng.model.hp;
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let mut wnames: Vec<String> = Vec::new();
    let mut cnames: Vec<String> = Vec::new();
    for il in 0..hp.n_layer {
        cnames.push(format!("blk.{il}.attn_norm"));
        cnames.push(format!("blk.{il}.post_norm"));
        if is_recr[il] {
            for w in ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out"] {
                wnames.push(format!("blk.{il}.{w}.weight"));
            }
            cnames.push(format!("blk.{il}.conv_w"));
            cnames.push(format!("blk.{il}.dt_bias"));
            cnames.push(format!("blk.{il}.ssm_a"));
            cnames.push(format!("blk.{il}.ssm_norm"));
        } else {
            for w in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                wnames.push(format!("blk.{il}.{w}.weight"));
            }
            cnames.push(format!("blk.{il}.attn_q_norm"));
            cnames.push(format!("blk.{il}.attn_k_norm"));
        }
        for w in ["ffn_gate", "ffn_up", "ffn_down"] {
            wnames.push(format!("blk.{il}.{w}.weight"));
        }
    }
    wnames.push("output.weight".into());
    // MTP층 (blk.64) — spec decode용 (has_mtp 시)
    if eng.has_mtp() {
        let mtp = 64usize;
        for w in ["attn_q", "attn_k", "attn_v", "attn_output",
                  "ffn_gate", "ffn_up", "ffn_down", "nextn.eh_proj"] {
            wnames.push(format!("blk.{mtp}.{w}.weight"));
        }
        for c in ["attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm",
                  "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"] {
            cnames.push(format!("blk.{mtp}.{c}"));
        }
    }
    cnames.push("output_norm".into());
    cnames.push("cs".into());
    cnames.push("mask".into());
    (wnames, cnames)
}

/// rawhip/VkDecoder 공용 상수 페치 (이름 리맵·타일 포함).
pub fn raw_consts(
    eng: &Engine,
    cnames: &[String],
) -> Vec<(String, Vec<f32>)> {
    let hp = &eng.model.hp;
    let ctx_n = eng.ctx_len();
    cnames
        .iter()
        .filter_map(|k| {
            let v = if k == "cs" {
                Some(hp.rope_cs(ctx_n))
            } else if k == "mask" {
                // 인과 마스크 [pos][p]: p<=pos 만 1 — qsa 배치용 (원본 의미 복원)
                let mut m = vec![0.0f32; ctx_n * ctx_n];
                for pos in 0..ctx_n {
                    for pp in 0..=pos {
                        m[pos * ctx_n + pp] = 1.0;
                    }
                }
                Some(m)
            } else if k.ends_with("conv_w") {
                eng.model.f32_vec(&format!("blk.{}.ssm_conv1d.weight", k.split('.').nth(1).unwrap_or("0"))).ok()
            } else {
                let il = k.split('.').nth(1).unwrap_or("0").to_string();
                let (tn, tiled) = if k.ends_with("dt_bias") {
                    (format!("blk.{il}.ssm_dt.bias"), 1)
                } else if k.ends_with("ssm_a") {
                    (k.clone(), 1)
                } else if k.ends_with("ssm_norm") {
                    (format!("blk.{il}.ssm_norm.weight"), hp.dt_rank)
                } else if k.ends_with("post_attention_norm") {
                    (format!("blk.{il}.post_attention_norm.weight"), 1)
                } else if k.ends_with("attn_norm") {
                    (format!("blk.{il}.attn_norm.weight"), 1)
                } else if k.ends_with("post_norm") {
                    (format!("blk.{il}.post_attention_norm.weight"), 1)
                } else if k == "output_norm" {
                    ("output_norm.weight".to_string(), 1)
                } else if k.ends_with("attn_q_norm") {
                    (format!("blk.{il}.attn_q_norm.weight"), hp.n_head)
                } else if k.ends_with("attn_k_norm") {
                    (format!("blk.{il}.attn_k_norm.weight"), hp.n_kv)
                } else {
                    (format!("{k}.weight"), 1)
                };
                eng.model.f32_vec(&tn).ok().map(|v| {
                    if tiled > 1 && (v.len() == hp.d_state || v.len() == hp.head_dim) {
                        v.iter().copied().cycle().take(v.len() * tiled).collect()
                    } else {
                        v
                    }
                })
            };
            v.map(|v| (k.clone(), v))
        })
        .collect()
}
