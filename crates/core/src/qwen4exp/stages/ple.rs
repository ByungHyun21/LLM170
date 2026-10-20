//! PLE(previous-layer embedding) 스테이지 — 판독 검증 + [D2 2026-10-20]
//! forward 전수(CPU 오라클).
//!
//! 텐서 계약(FN 실측, 배치 층 il=1): conv1d [10240,1,4] · key_proj
//! [10240,2560] · value_proj [2560,2560] · norm_conv/norm_key/norm_query
//! [10240] · ple_embedding = ngram_embedding.shard_{0..127} F8_E4M3
//! [2500012,160] · weight_scale(BF16 스칼라, 글로벌) ·
//! ngram_heads_offsets/vocab_sizes I64[16] · layer_multipliers I64[3].
//!
//! forward(vLLM ple_layer.py + common/ops/ple.py 전수):
//! 1. n-gram id: 16헤드 = (order 2,3)×8. mixed = tok[t]·mult[0] ⊕
//!    (shift 1..2: order>shift만 tok[t−shift]·mult[shift]; 선행 EOS 한번
//!    경과면 그 이전 토큰 전부 EOS 치환; 시퀀스 시작 전은 EOS — 신규
//!    시퀀스 ngram_context가 eos로 채워짐). id = (mixed rem_euclid
//!    size[h]) + offset[h]. 곱셈은 i64 wrapping(트리토닉 오버플로
//!    정합)·XOR 결합.
//! 2. 임베딩 16헤드×160=2560 concat → FP8_E4M3 디코드 ×weight_scale.
//! 3. kv_proj: key[10240]·value[2560].
//! 4. 게이트(스트림 s별): k_n=RMSN(key_s)·(1+nk)·q_n=RMSN(R_s)·(1+nq)
//!    ·d=⟨⟩/√2560·g=σ(sign(d)·√max(|d|,1e-6))·gated=g·value
//!    ·conv_in=RMSN(gated_s)·(1+nc).
//! 5. depthwise conv k4·dilation 3(수신 오프셋 −9,−6,−3,0) + silu +
//!    잔차: R += gated + silu(conv). **상태 = 최근 9토큰×10240**(
//!    norm_conv 입력 윈도우 — prefill은 구상태 읽기 후 일괄 갱신).
//!
//! D0 계약 — norm_{conv,key,query} = Gemma식 zero-centered(1+w)
//! (vLLM Qwen4ExpPLEGroupedNorm `normalized*(1+w)` ple_layer.py:62)
//! — 가중치는 `Model::norm_vec`(+1 합성 단일 지점)으로.

use super::super::engine::Model;
use crate::ops::{rms_norm, sigmoid, silu};

/// conv 상태 — 최근 9토큰((k−1)·dilation)×10240 norm_conv 입력 윈도우.
pub struct PleState {
    /// [9][10240] — state[i] = x[T−9+i] (T = 시퀀스 길이).
    pub conv: Vec<Vec<f32>>,
    /// 초기(프리필 전) 상태 부재 — vLLM has_initial_states=false 등가.
    pub has_init: bool,
}

impl PleState {
    pub fn new(ch: usize) -> Self {
        PleState {
            conv: vec![vec![0.0f32; ch]; conv_state_len()],
            has_init: false,
        }
    }
}

/// conv 창 길이 = (kernel 4 − 1)·dilation 3 = 9.
pub fn conv_state_len() -> usize {
    9
}

/// PLE forward — R(멀티스트림 [T, hc·H])에 PLE 출력 직접 가산(원위치).
/// `base_pos` = 청크 첫 토큰의 절대 위치, `hist` = 전체 시퀀스 토큰열
/// (hist[base_pos..]가 이번 청크). 상태(conv) 갱신 포함.
pub fn forward(
    m: &Model,
    il: usize,
    hist: &[u32],
    base_pos: usize,
    r: &mut [Vec<f32>],
    st: &mut PleState,
) -> Result<(), String> {
    let cfg = &m.cfg;
    let (hc, hidden) = (cfg.hc_count, cfg.hidden_size);
    let l = format!("model.language_model.layers.{il}.ple");
    let n_tok = r.len();
    let emb_dim = cfg.ple_embed_dim;
    let ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    let head_dim = emb_dim / ngram_heads;

    // 버퍼 판독 + 검증.
    let eb = format!("{l}.ple_embedding");
    let mult = i64_vec(m, &format!("{eb}.layer_multipliers"))?;
    let sizes = i64_vec(m, &format!("{eb}.ngram_heads_vocab_sizes"))?;
    let offsets = i64_vec(m, &format!("{eb}.ngram_heads_offsets"))?;
    if mult.len() != cfg.ngram_size || sizes.len() != ngram_heads || offsets.len() != ngram_heads {
        return Err(format!(
            "PLE[{il}] ngram 버퍼 위반 mult={} sizes={} offsets={} (ngram_size {}·heads {ngram_heads})",
            mult.len(),
            sizes.len(),
            offsets.len(),
            cfg.ngram_size
        ));
    }
    let scale = m.f32_vec(&format!("{eb}.ngram_embedding.weight_scale"))?[0];
    let shard_rows = m
        .ent(&format!("{eb}.ngram_embedding.shard_0.weight"))?
        .shape[0] as usize;
    let conv_w = m.f32_vec(&format!("{l}.conv1d.weight"))?; // [ch][1][k] 행 우선
    let nk = m.norm_vec(&format!("{l}.norm_key.weight"))?;
    let nq = m.norm_vec(&format!("{l}.norm_query.weight"))?;
    let nc = m.norm_vec(&format!("{l}.norm_conv.weight"))?;

    let eos = cfg.eos_token_id;
    let mut gated_all = vec![vec![0.0f32; hc * hidden]; n_tok];
    let mut normed_all = vec![vec![0.0f32; hc * hidden]; n_tok];
    for t in 0..n_tok {
        // ① n-gram id → FP8 임베딩 디코드.
        let ids = ngram_ids(hist, base_pos + t, &mult, &sizes, &offsets, eos, cfg.heads_per_ngram);
        let mut emb = vec![0.0f32; emb_dim];
        for (h, &id) in ids.iter().enumerate() {
            ple_row(m, &eb, shard_rows, id, scale, &mut emb[h * head_dim..(h + 1) * head_dim])?;
        }
        // ② kv_proj.
        let key = m.matvec_bf16(&format!("{l}.key_proj.weight"), &emb)?;
        let value = m.matvec_bf16(&format!("{l}.value_proj.weight"), &emb)?;
        // ③ 게이트 — 스트림별(ple_gate 커널 전수).
        for s in 0..hc {
            let b = s * hidden;
            let k_n = rms_norm(&key[b..b + hidden], &nk[b..b + hidden], m.eps);
            let q_n = rms_norm(&r[t][b..b + hidden], &nq[b..b + hidden], m.eps);
            let mut dot = 0.0f64;
            for i in 0..hidden {
                dot += k_n[i] as f64 * q_n[i] as f64;
            }
            let d = (dot as f32) / (hidden as f32).sqrt();
            let sign = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            let magnitude = d.abs().max(1e-6).sqrt();
            let g = sigmoid(sign * magnitude);
            let gated = &mut gated_all[t][b..b + hidden];
            for i in 0..hidden {
                gated[i] = g * value[i];
            }
            // norm_conv — 게이트 출력의 스트림별 RMSN(1+nc).
            let normed = rms_norm(gated, &nc[b..b + hidden], m.eps);
            normed_all[t][b..b + hidden].copy_from_slice(&normed);
        }
    }

    // ④ depthwise conv k4·dilation3 — tap 오프셋 (+3k−9), 구상태 먼저 읽기.
    let ch = hc * hidden;
    let wlen = conv_state_len();
    let conv_y = |t: usize, c: usize| -> f32 {
        let mut acc = 0.0f32;
        for k in 0..4usize {
            // 절대 위치 q = (base_pos+t) + 3k − 9.
            let q = (base_pos + t + 3 * k) as isize - 9;
            let tap: f32 = if q < 0 {
                0.0
            } else if (q as usize) < base_pos {
                // 청크 이전 — conv 상태(state[i] = x[base_pos−9+i] → i = q−base_pos+9).
                if st.has_init {
                    st.conv[q as usize + wlen - base_pos][c]
                } else {
                    0.0
                }
            } else {
                normed_all[q as usize - base_pos][c]
            };
            acc += conv_w[c * 4 + k] * tap;
        }
        acc
    };
    let mut y = vec![vec![0.0f32; ch]; n_tok];
    for t in 0..n_tok {
        for c in 0..ch {
            y[t][c] = conv_y(t, c);
        }
    }
    // ⑤ 잔차 가산 — R += gated + silu(conv).
    for t in 0..n_tok {
        for c in 0..ch {
            r[t][c] += gated_all[t][c] + silu(y[t][c]);
        }
    }
    // ⑥ 상태 갱신(ple_conv_writeback 전수): new[i] = x[T−9+i], T = 청크
    // 끝 절대 위치 — 청크 내 값, 부족분은 구상태(초기화 전 0)·시퀀스
    // 시작 전 0.
    let t_end = base_pos + n_tok;
    let mut new_st = vec![vec![0.0f32; ch]; wlen];
    for i in 0..wlen {
        let q = t_end as isize + i as isize - 9;
        if q < 0 {
            continue;
        }
        let q = q as usize;
        if q >= base_pos {
            new_st[i].copy_from_slice(&normed_all[q - base_pos]);
        } else if st.has_init {
            new_st[i].copy_from_slice(&st.conv[q + wlen - base_pos]);
        }
    }
    st.conv = new_st;
    st.has_init = true;
    Ok(())
}

/// n-gram id 16개(_ple_ngram_ids_kernel 전수 — 단일 시퀀스).
/// `pos` = 현재 토큰 절대 위치. 선행 토큰은 hist에서, 시퀀스 시작
/// 전(t−shift < 0)은 EOS(신규 시퀀스 ngram_context = eos 채움).
pub fn ngram_ids(
    hist: &[u32],
    pos: usize,
    mult: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    eos: u32,
    heads_per: usize,
) -> Vec<i64> {
    let ngram_ctx = mult.len() - 1; // shift 범위(= ngram_size−1)
    let n_heads = sizes.len();
    let tok = hist[pos] as i64;
    let mut mixed = vec![tok.wrapping_mul(mult[0]); n_heads];
    let mut crossed = false;
    for shift in 1..=ngram_ctx {
        let mut cand = if pos >= shift {
            hist[pos - shift] as i64
        } else {
            eos as i64
        };
        if crossed {
            cand = eos as i64;
        }
        crossed = crossed || cand == eos as i64;
        let term = cand.wrapping_mul(mult[shift]);
        for (h, m) in mixed.iter_mut().enumerate() {
            let order = h / heads_per + 2;
            if order > shift {
                *m ^= term;
            }
        }
    }
    mixed
        .iter()
        .enumerate()
        .map(|(h, &m)| m.rem_euclid(sizes[h]) + offsets[h])
        .collect()
}

/// FP8 E4M3FN(u8) → f32. sign 1·exp 4(bias 7)·mant 3. 무한 없음
/// (S.1111.111만 NaN) — 최대 유한 448.
pub fn e4m3_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    match e {
        0 => s * (m / 8.0) * 2f32.powi(-6),
        15 if m == 7.0 => f32::NAN * s,
        _ => s * (1.0 + m / 8.0) * 2f32.powi(e - 7),
    }
}

/// 임베딩 행 조회 — 논리 샤드 128개가 big-table[id // rows]를 소유
/// (vLLM loader: checkpoint_start = shard_index·shard_size).
fn ple_row(
    m: &Model,
    eb: &str,
    shard_rows: usize,
    id: i64,
    scale: f32,
    out: &mut [f32],
) -> Result<(), String> {
    let si = (id / shard_rows as i64) as usize;
    let row = (id % shard_rows as i64) as u64;
    let name = format!("{eb}.ngram_embedding.shard_{si}.weight");
    let e = m.ent(&name)?;
    if row >= e.shape[0] {
        return Err(format!("PLE id {id} — 샤드 {si} 행 {row} ≥ {}", e.shape[0]));
    }
    let d = m.slice(&name)?;
    let base = (row as usize) * out.len();
    for (j, o) in out.iter_mut().enumerate() {
        *o = e4m3_f32(d[base + j]) * scale;
    }
    Ok(())
}

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

fn i64_vec(m: &Model, name: &str) -> Result<Vec<i64>, String> {
    let d = m.slice(name)?;
    Ok(d.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E4M3FN 디코드 — 규격값 표 검증(OCP FP8, 무한 없음).
    #[test]
    fn e4m3_decode() {
        assert_eq!(e4m3_f32(0x00), 0.0);
        assert_eq!(e4m3_f32(0x80), -0.0);
        assert_eq!(e4m3_f32(0x40), 2.0); // exp 8 → 2^1
        assert_eq!(e4m3_f32(0xC0), -2.0);
        assert_eq!(e4m3_f32(0x30), 0.5); // exp 6 → 2^−1
        assert_eq!(e4m3_f32(0x38), 1.0); // exp 7 → 2^0
        assert_eq!(e4m3_f32(0x3C), 1.5); // mant 4/8
        assert_eq!(e4m3_f32(0x7E), 448.0); // 최대 유한
        assert_eq!(e4m3_f32(0xFE), -448.0);
        assert!(e4m3_f32(0x7F).is_nan());
        assert!(e4m3_f32(0xFF).is_nan());
        // 비정규: exp 0 → (m/8)·2^−6.
        assert_eq!(e4m3_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_f32(0x07), 0.875 * 2f32.powi(-6));
        // 최소 정규.
        assert_eq!(e4m3_f32(0x08), 2f32.powi(-6));
    }

    /// n-gram id — 커널 수식 손계산 대조. 다중플라이어 임의 수
    /// (정수 해시 산술 — rand 금지 계약 준수), size/offset 소형.
    #[test]
    fn ngram_id_math() {
        // 멀티플라이어: [m0, m1, m2] = [7, 11, 13](테스트 전용 값).
        let mult = vec![7i64, 11, 13];
        // 헤드 16 = (order2, order3)×8: sizes 임의, offsets 홀수 헤드만 +100.
        let sizes: Vec<i64> = (0..16).map(|h| 1_000_003 + h as i64).collect();
        let offsets: Vec<i64> = (0..16).map(|h| h as i64 * 100).collect();
        let eos = 999u32;
        let hist = vec![5u32, 11, 3, 999, 8, 2];
        // pos=2: mixed_h = 3·7 ⊕ (order>1: 11·11) ⊕ (order>2: 5·13).
        let ids = ngram_ids(&hist, 2, &mult, &sizes, &offsets, eos, 8);
        let base2 = 3i64.wrapping_mul(7) ^ 11i64.wrapping_mul(11) ^ 5i64.wrapping_mul(13);
        let tri2 = 3i64.wrapping_mul(7) ^ 11i64.wrapping_mul(11);
        for h in 0..16 {
            let expect = if h < 8 { tri2 } else { base2 };
            assert_eq!(
                ids[h],
                expect.rem_euclid(sizes[h]) + offsets[h],
                "헤드 {h}"
            );
        }
        // pos=4: 선행 [3, 999] — shift1=999(자연 EOS) → crossed, shift2=EOS 치환.
        let ids4 = ngram_ids(&hist, 4, &mult, &sizes, &offsets, eos, 8);
        let tri4 = 8i64.wrapping_mul(7) ^ 999i64.wrapping_mul(11);
        let base4 = tri4 ^ 999i64.wrapping_mul(13); // 치환된 shift2도 EOS
        for h in 0..16 {
            let expect = if h < 8 { tri4 } else { base4 };
            assert_eq!(ids4[h], expect.rem_euclid(sizes[h]) + offsets[h]);
        }
        // pos=0: 선행 둘 다 컨텍스트 EOS.
        let ids0 = ngram_ids(&hist, 0, &mult, &sizes, &offsets, eos, 8);
        let tri0 = 5i64.wrapping_mul(7) ^ 999i64.wrapping_mul(11);
        let base0 = tri0 ^ 999i64.wrapping_mul(13);
        for h in 0..16 {
            let expect = if h < 8 { tri0 } else { base0 };
            assert_eq!(ids0[h], expect.rem_euclid(sizes[h]) + offsets[h]);
        }
        // 음수 rem_euclid 경계: 큰 토큰·mult 곱이 i64 음수가 되어도
        // id는 [0, size) 보장(torch.remainder 정합).
        let big = vec![248319u32, 1];
        let bm = vec![0x9E3779B97F4A7C15u64 as i64, 3, 5];
        let ids_b = ngram_ids(&big, 1, &bm, &sizes, &offsets, eos, 8);
        for (h, &v) in ids_b.iter().enumerate() {
            let rel = v - offsets[h];
            assert!((0..sizes[h]).contains(&rel), "헤드 {h} id {v}");
        }
    }
}
