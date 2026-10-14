//! gemma4 어텐션 스테이지 — CPU 참조 (HF `Gemma4TextAttention` 미러).
//!
//! 계약(2026-10-14 HF 대조):
//! - q = rope(q_norm(q_proj(h))) — QK 노름 = **학습가중** per-head RMS(+1 아님).
//! - k = rope(k_norm(k_proj(h))) · v = **가중 없는** RMS(v_proj(h));
//!   풀층(k_eq_v) = v_proj 없음 → v = v_norm(k_proj(h) **원시**).
//! - 스케일 **1.0**(1/√d 없음)·출력 게이트 없음·q/k/v 캐시는 노름·rope 후 값.
//! - 슬라이딩 층 마스크 = 자기 포함 창(`kv > q − sw`), 풀층 전역 causal.

use super::super::{Model, ModelError, SeqState};
use crate::matmul::cpu::matmul_batch;
use crate::ops::{rms_norm, rms_norm_plain, rope_head_dim};

/// 단일 헤드 어텐션 — q·k 내적(스케일 1.0) → exp_cr 소프트맥스 → Σp·v.
/// `start` = 창 시작(포함) — 슬라이딩 마스크.
#[allow(clippy::too_many_arguments)]
fn attn_head(
    qh: &[f32],
    cache_k: &[f32],
    cache_v: &[f32],
    start: usize,
    pos: usize,
    kvh: usize,
    n_kv: usize,
    hd: usize,
    out: &mut [f32],
) {
    let n = pos + 1 - start;
    let mut scores = vec![0.0f32; n];
    let mut maxv = f32::NEG_INFINITY;
    for (i, sc) in scores.iter_mut().enumerate() {
        let b = (start + i) * n_kv * hd + kvh * hd;
        let mut d = 0.0f32;
        for j in 0..hd {
            d += qh[j] * cache_k[b + j];
        }
        *sc = d;
        maxv = maxv.max(d);
    }
    let mut sum = 0.0f32;
    for sc in scores.iter_mut() {
        *sc = crate::ops::exp_cr(*sc - maxv);
        sum += *sc;
    }
    for sc in scores.iter_mut() {
        *sc /= sum;
    }
    for (i, &w) in scores.iter().enumerate() {
        if w == 0.0 {
            continue;
        }
        let b = (start + i) * n_kv * hd + kvh * hd;
        for j in 0..hd {
            out[j] += w * cache_v[b + j];
        }
    }
}

pub(crate) fn attn_layer(
    model: &Model,
    seqs: &mut [SeqState],
    il: usize,
    xs: &[Vec<f32>],
    seq_ids: &[usize],
    t_len: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let hp = &model.hp;
    let n_seqs = seq_ids.len();
    let n_tok = n_seqs * t_len;
    let n_head = hp.n_head;
    let hd = hp.head_dim[il];
    let n_kv = hp.n_kv[il];
    let n_rot = hp.n_rot[il];
    let base = hp.rope_base[il];
    let sw = if hp.full[il] { 0 } else { hp.sw };
    let k_eq_v = model.kv_eq_v(il);
    let wq = model.wchk(&format!("blk.{il}.attn_q.weight"))?;
    let wk = model.wchk(&format!("blk.{il}.attn_k.weight"))?;
    let wv = if k_eq_v {
        None
    } else {
        Some(model.wchk(&format!("blk.{il}.attn_v.weight"))?)
    };
    let wo = model.wchk(&format!("blk.{il}.attn_output.weight"))?;
    let q_norm_w = model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
    let k_norm_w = model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;

    // q·k·(v) 투영 — 전 토큰 배치.
    let mut q_all = vec![vec![0.0f32; n_head * hd]; n_tok];
    let mut k_all = vec![vec![0.0f32; n_kv * hd]; n_tok];
    matmul_batch(xs, &wq, &mut q_all);
    matmul_batch(xs, &wk, &mut k_all);
    let v_all = match &wv {
        Some(w) => {
            let mut v = vec![vec![0.0f32; n_kv * hd]; n_tok];
            matmul_batch(xs, w, &mut v);
            v
        }
        None => Vec::new(),
    };

    let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
    let mut attn_all = vec![vec![0.0f32; n_head * hd]; n_tok];
    for s in 0..n_seqs {
        let pos0 = seqs[seq_ids[s]].pos;
        let seq = &mut seqs[seq_ids[s]];
        let (cache_k, cache_v) = (seq.kv_k[il].as_mut_slice(), seq.kv_v[il].as_mut_slice());
        for t in 0..t_len {
            let row = s * t_len + t;
            let pos = pos0 + t as u32;
            // k·v 캐시 적립 — k는 노름+rope, v는 가중 없는 노름만.
            for h in 0..n_kv {
                let src = &k_all[row][h * hd..h * hd + hd];
                let mut kh = rms_norm(src, &k_norm_w, hp.eps);
                rope_head_dim(&mut kh, pos, n_rot, hd, base);
                let b = pos as usize * n_kv * hd + h * hd;
                cache_k[b..b + hd].copy_from_slice(&kh);
                let vraw: &[f32] = if k_eq_v {
                    src
                } else {
                    &v_all[row][h * hd..h * hd + hd]
                };
                let vh = rms_norm_plain(vraw, hp.eps);
                cache_v[b..b + hd].copy_from_slice(&vh);
            }
            // q — 헤드별 노름+rope.
            let mut qh = vec![0.0f32; hd];
            let mut attn_out = std::mem::take(&mut attn_all[row]);
            let start = if sw == 0 {
                0
            } else {
                (pos as usize + 1).saturating_sub(sw)
            };
            for h in 0..n_head {
                qh.copy_from_slice(&q_all[row][h * hd..h * hd + hd]);
                let mut qn = rms_norm(&qh, &q_norm_w, hp.eps);
                rope_head_dim(&mut qn, pos, n_rot, hd, base);
                let kvh = h / (n_head / n_kv);
                attn_head(
                    &qn,
                    cache_k,
                    cache_v,
                    start,
                    pos as usize,
                    kvh,
                    n_kv,
                    hd,
                    &mut attn_out[h * hd..h * hd + hd],
                );
            }
            attn_all[row] = attn_out;
        }
    }
    // wo 프로젝션 — 전 토큰 배치 1회.
    matmul_batch(&attn_all, &wo, &mut out);
    Ok(out)
}
