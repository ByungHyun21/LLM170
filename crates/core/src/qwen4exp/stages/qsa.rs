//! QSA 인덱서 + 선택 집합 full어텐션 스테이지 — [D2 2026-10-20].
//!
//! vLLM amd/indexer_qsa.py + amd/ops/qsa.py 전수(단일 시퀀스 CPU 오라클):
//! - 인덱서: index_qk_proj [640,2560] → Q 4헤드×128(RMSN+1 q_layernorm
//!   → 부분 RoPE n_rot 64·θ1e7 — 메인 rotary_dim 계약 동일) + 원시 K
//!   1헤드×128(무정규). 압축: r=4 완성 블록마다 원시 K 산술평균 →
//!   RMSN+1(k_layernorm) → 블록 첫 위치 p_b RoPE → comp_k 캐시.
//! - 점수(_qsa_mqa_paged_kernel): s_j = Σ_{h<4} ReLU(⟨q_h, k̄_j⟩)/√128
//!   ·보이는 블록 j < visible = (p+1)//4. top-512블록
//!   (budget 2048/4) + 불완전 블록 꼬리(visible·4..p) 전개.
//! - 본 어텐션(_qsa_sparse_paged_gqa_splitk_kernel): 선택 토큰 집합에만
//!   softmax(scale = 256^−0.5, GQA 24Q/2KV) — 추가 인과 마스크 불필요
//!   (선택 블록·꼬리 전부 ≤ p). Q‖gate 인터리브·σ 게이트는 stages/full
//!   D0 계약 승계.
//!
//! 상태: 원시 K(전토큰)·압축 키(블록수)·본 KV(전토큰) — 접두 체크포인트
//! 축. 성능 무관 — 정확성이 목적.

use super::super::engine::Model;
use crate::ops::{rms_norm, rope_head, sigmoid};

/// 인덱서 캐시 — 원시 K·압축 키(full층별).
pub struct QsaCache {
    /// [T][128] 원시 인덱서 K(index_qk_proj 후반, 무정규·무RoPE).
    pub raw_k: Vec<Vec<f32>>,
    /// [T/4][128] 압축 키(평균 → RMSN+1 → p_b RoPE).
    pub comp_k: Vec<Vec<f32>>,
}

impl QsaCache {
    pub fn new() -> Self {
        QsaCache {
            raw_k: Vec::new(),
            comp_k: Vec::new(),
        }
    }
}

impl Default for QsaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 본 KV 캐시(full층별) — k는 RoPE 적용 후.
pub struct KvCache {
    /// [T][n_kv·256]
    pub k: Vec<Vec<f32>>,
    /// [T][n_kv·256]
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new() -> Self {
        KvCache {
            k: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 인덱서 투영 — 토큰 1개: (Q 4×128 정규+RoPE, 원시 K 128).
fn indexer_project(
    m: &Model,
    l: &str,
    x: &[f32],
    pos: u32,
    q_ln: &[f32],
) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
    let (n_ih, ihd, n_rot) = (
        m.cfg.indexer_n_heads,
        m.cfg.indexer_head_dim,
        m.cfg.indexer_head_dim / 2,
    );
    let qk = m.matvec_bf16(&format!("{l}.indexer.index_qk_proj.weight"), x)?;
    let mut q = Vec::with_capacity(n_ih);
    for h in 0..n_ih {
        let b = h * ihd;
        // q_layernorm[128]은 헤드 공유(vLLM GemmaRMSNorm(index_head_dim)).
        let mut head = rms_norm(&qk[b..b + ihd], q_ln, m.eps);
        rope_head(&mut head, pos, n_rot, 1e7);
        q.push(head);
    }
    let k = qk[n_ih * ihd..n_ih * ihd + ihd].to_vec();
    Ok((q, k))
}

/// 선택 토큰 집합 — 쿼리 절대 위치 p. top-512 블록 전개 + 불완전 블록
/// 꼬리(expand_qsa_block_indices 전수). 동점은 블록 인덱스 오름차순
/// (결정적 오라클 규약 — GPU topk 동점 순서와 다를 수 있음[INFERENCE]).
pub fn select(
    idx: &QsaCache,
    q: &[Vec<f32>],
    p: usize,
    ratio: usize,
    block_topk: usize,
) -> Vec<usize> {
    let visible = (p + 1) / ratio;
    if visible == 0 {
        return (0..=p).collect();
    }
    // 점수 — f32 곱·f64 누산(√128 나눈 뒤 f32).
    let ihd = q[0].len();
    let mut scores = Vec::with_capacity(visible);
    for j in 0..visible {
        let kb = &idx.comp_k[j];
        let mut s = 0.0f32;
        for qh in q {
            let mut dot = 0.0f64;
            for i in 0..ihd {
                dot += qh[i] as f64 * kb[i] as f64;
            }
            s += (dot as f32).max(0.0);
        }
        scores.push(s / (ihd as f32).sqrt());
    }
    let mut order: Vec<usize> = (0..visible).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(block_topk);
    // 전개 순서 = 참조 커널과 동일(랭크순 블록 · 블록 내 오름차순 · 꼬리).
    let mut toks: Vec<usize> = Vec::with_capacity(block_topk * ratio + ratio);
    for &b in &order {
        toks.extend(b * ratio..b * ratio + ratio);
    }
    toks.extend(visible * ratio..=p);
    toks
}

/// 상태ful full어텐션 forward — 청크 [base_pos..base_pos+len]. KV·인덱서
/// 캐시 갱신. 반환: o_proj 출력 [T][2560].
pub fn full_forward(
    m: &Model,
    il: usize,
    xs: &[Vec<f32>],
    base_pos: usize,
    kv: &mut KvCache,
    idx: &mut QsaCache,
) -> Result<Vec<Vec<f32>>, String> {
    let cfg = &m.cfg;
    let l = format!("model.language_model.layers.{il}.self_attn");
    let (n_head, n_kv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let ratio = cfg.indexer_compress_ratio.max(1);
    let block_topk = cfg.indexer_budget / ratio;
    let q_norm_w = m.norm_vec(&format!("{l}.q_norm.weight"))?;
    let k_norm_w = m.norm_vec(&format!("{l}.k_norm.weight"))?;
    let q_ln = m.norm_vec(&format!("{l}.indexer.q_layernorm.weight"))?;
    let k_ln = m.norm_vec(&format!("{l}.indexer.k_layernorm.weight"))?;
    let i_n_rot = cfg.indexer_head_dim / 2; // 인덱서 부분 RoPE 64/128
    let n_rot = hd / 4; // partial_rotary_factor 0.25
    let kq_scale = 1.0f32 / (hd as f32).sqrt();
    let n_tok = xs.len();

    let mut out = vec![vec![0.0f32; n_head * hd]; n_tok];
    for t in 0..n_tok {
        let p = base_pos + t;
        // 본 KV — k는 헤드별 RMSN(+1)+RoPE 후 적재.
        let kk = m.matvec_bf16(&format!("{l}.k_proj.weight"), &xs[t])?;
        let vv = m.matvec_bf16(&format!("{l}.v_proj.weight"), &xs[t])?;
        let mut krow = vec![0.0f32; n_kv * hd];
        for h in 0..n_kv {
            let mut head = rms_norm(&kk[h * hd..h * hd + hd], &k_norm_w, m.eps);
            rope_head(&mut head, p as u32, n_rot, 1e7);
            krow[h * hd..h * hd + hd].copy_from_slice(&head);
        }
        kv.k.push(krow);
        kv.v.push(vv);
        // 인덱서 — 원시 K 적재 + 블록 완성 시 압축.
        let (iq, ik) = indexer_project(m, &l, &xs[t], p as u32, &q_ln)?;
        idx.raw_k.push(ik);
        if (p + 1).is_multiple_of(ratio) {
            let ihd = cfg.indexer_head_dim;
            let mut pooled = vec![0.0f32; ihd];
            // 오래된 토큰부터 누산(참조 커널 group_offset 순서 정합).
            for off in 0..ratio {
                let src = &idx.raw_k[idx.raw_k.len() - ratio + off];
                for (d, v) in pooled.iter_mut().enumerate() {
                    *v += src[d];
                }
            }
            for v in pooled.iter_mut() {
                *v /= ratio as f32;
            }
            let mut nk = rms_norm(&pooled, &k_ln, m.eps);
            rope_head(&mut nk, (p + 1 - ratio) as u32, i_n_rot, 1e7);
            idx.comp_k.push(nk);
        }
        // 선택 집합 → 본 어텐션.
        let sel = select(idx, &iq, p, ratio, block_topk);
        let qg = m.matvec_bf16(&format!("{l}.q_proj.weight"), &xs[t])?;
        for h in 0..n_head {
            let qb = h * 2 * hd;
            let mut qh = rms_norm(&qg[qb..qb + hd], &q_norm_w, m.eps);
            rope_head(&mut qh, p as u32, n_rot, 1e7);
            let kvh = h / (n_head / n_kv);
            let mut scores = Vec::with_capacity(sel.len());
            let mut mx = f32::NEG_INFINITY;
            for &s in &sel {
                let kb = kvh * hd;
                let mut dp = 0.0f64;
                for i in 0..hd {
                    dp += qh[i] as f64 * kv.k[s][kb + i] as f64;
                }
                let sc = (dp as f32) * kq_scale;
                scores.push(sc);
                mx = mx.max(sc);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let ob = h * hd;
            for i in 0..hd {
                let mut acc = 0.0f64;
                for (j, &s) in scores.iter().enumerate() {
                    acc += s as f64 * kv.v[sel[j]][kvh * hd + i] as f64;
                }
                out[t][ob + i] = (acc / sum as f64) as f32;
            }
            // 출력 게이트 — Q 헤드 뒤 256차원 ⊙σ(D0 인터리브 계약).
            for i in 0..hd {
                out[t][ob + i] *= sigmoid(qg[qb + hd + i]);
            }
        }
    }
    for t in 0..n_tok {
        out[t] = m.matvec_bf16(&format!("{l}.o_proj.weight"), &out[t])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 선택 전개 — 블록+꼬리 집합·정렬·중복 제거·동점 순서 검증(순수 산술).
    #[test]
    fn select_math() {
        let ihd = 4usize;
        // comp_k: 블록 j의 키 = j 방향 단위벡터(j<3) — q 헤드 1개가 블록 2와
        // 평행(점수 최대), 나머지 0(동점) — 동점은 인덱스 오름차순.
        let mut idx = QsaCache::new();
        for j in 0..3usize {
            let mut k = vec![0.0f32; ihd];
            k[j % ihd] = 1.0;
            idx.comp_k.push(k);
        }
        let q = vec![vec![0.0f32, 0.0, 1.0, 0.0]]; // 블록 2와 평행
        let ratio = 4;
        // p=11 → visible=3(블록 0·1·2) — q는 블록 2와 평행(점수 최대)
        // → top1 = 블록 2, 꼬리 없음(12토큰 = 3 완성 블록).
        let sel = select(&idx, &q, 11, ratio, 1);
        assert_eq!(sel, vec![8, 9, 10, 11], "visible 3·top1=블록2");
        // p=10 → visible=2(블록 0·1만 점수 대상·점수 동점 0) — 동점은
        // 인덱스 오름차순 → 블록 0 + 꼬리 8..10.
        let selb = select(&idx, &q, 10, ratio, 1);
        assert_eq!(selb, vec![0, 1, 2, 3, 8, 9, 10], "동점→블록0·꼬리 8..10");
        // p=7 → visible=2·top2 → 전체 블록.
        let sel2 = select(&idx, &q, 7, ratio, 2);
        assert_eq!(sel2, vec![0, 1, 2, 3, 4, 5, 6, 7], "visible 2·top2 전체");
        // p=3 → visible 0 → 밀집 전체.
        let sel3 = select(&idx, &q, 3, ratio, 1);
        assert_eq!(sel3, vec![0, 1, 2, 3]);
    }
}
