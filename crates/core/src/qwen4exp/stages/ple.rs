//! PLE(previous-layer embedding) 스테이지 — **판독 검증 + 명시적 Err**.
//!
//! 텐서 계약(FN 실측, 배치 층 il=1): conv1d [10240,1,4] · key_proj
//! [10240,2560] · value_proj [2560,2560] · norm_conv/norm_key/norm_query
//! [10240] · ple_embedding: ngram_embedding.shard_{0..127} F8_E4M3
//! [2500012,160] + weight_scale + ngram_heads_offsets/vocab_sizes I64[16]
//! + layer_multipliers I64[3]. FP8 디코드·ngram 조회 그래프는 3차.
//! [D0] norm_{conv,key,query} = Gemma식 zero-centered(1+w) — vLLM
//! Qwen4ExpPLEGroupedNorm `normalized*(1+w)`(ple_layer.py:62). 3차 forward
//! 구현 시 가중치는 반드시 `Model::norm_vec`(+1 합성 단일 지점)으로.

use super::super::engine::Model;

/// PLE 텐서 전수 판독·형상 검증 → 요약 리포트(디코드 없음).
pub fn verify(m: &Model, il: usize) -> Result<String, String> {
    let l = format!("model.language_model.layers.{il}.ple");
    let mut rep = Vec::new();
    for p in ["conv1d.weight", "key_proj.weight", "value_proj.weight"] {
        let e = m.ent(&format!("{l}.{p}"))?;
        rep.push(format!("{p}={:?}", e.shape));
    }
    for p in ["norm_conv.weight", "norm_key.weight", "norm_query.weight"] {
        let e = m.ent(&format!("{l}.{p}"))?;
        rep.push(format!("{p}=[{}]", e.shape.first().unwrap_or(&0)));
    }
    let eb = format!("{l}.ple_embedding");
    let e0 = m.ent(&format!("{eb}.ngram_embedding.shard_0.weight"))?;
    let n_shards = (0..m.cfg.split_ngram_parts)
        .filter(|&s| {
            m.ent(&format!("{eb}.ngram_embedding.shard_{s}.weight"))
                .is_ok()
        })
        .count();
    let offs = i64_vec(m, &format!("{eb}.ngram_heads_offsets"))?;
    let vocabs = i64_vec(m, &format!("{eb}.ngram_heads_vocab_sizes"))?;
    let mult = i64_vec(m, &format!("{eb}.layer_multipliers"))?;
    m.ent(&format!("{eb}.ngram_embedding.weight_scale"))?;
    if n_shards != m.cfg.split_ngram_parts {
        return Err(format!(
            "ngram shard {n_shards}/{}",
            m.cfg.split_ngram_parts
        ));
    }
    if offs.len() != m.cfg.heads_per_ngram * 2 || vocabs.len() != m.cfg.heads_per_ngram * 2 {
        return Err(format!(
            "ngram offsets {}·vocabs {} — heads_per_ngram {}×2 계약 위반",
            offs.len(),
            vocabs.len(),
            m.cfg.heads_per_ngram
        ));
    }
    rep.push(format!(
        "shards {n_shards}×{:?} {dt:?} · offsets {} · vocab합 {} · multipliers {:?}",
        e0.shape,
        offs.len(),
        vocabs.iter().sum::<i64>(),
        mult,
        dt = e0.dtype,
    ));
    Ok(rep.join(" · "))
}

/// PLE forward — **미구현(3차)**: FP8_E4M3 임베딩 디코드(weight_scale 곱)
/// + ngram(3-gram) 인덱스 산출 + key/value_proj 경로 해부 필요.
pub fn forward(_m: &Model, _il: usize, _xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    Err("PLE forward 미구현 — FN 3차 (FP8_E4M3 ngram 임베딩 디코드·weight_scale·ngram 인덱스 산출 그래프 해부 필요)".into())
}

fn i64_vec(m: &Model, name: &str) -> Result<Vec<i64>, String> {
    let d = m.slice(name)?;
    Ok(d.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}
