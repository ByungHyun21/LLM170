//! Full-attention층 스테이지 — layers.rs에서 이동(qwen4exp P1 패턴, plans/90 B4).

use super::super::{ModelError, SeqState, span_block};
use super::Ctx;
use crate::matmul::{mm_batch, mm_group};
use crate::ops::{rms_norm, rope_head, sigmoid};
use llm170_diag::profile_span;


/// qwen35 단일 헤드 어텐션 — 점수→exp_cr 소프트맥스→가중합→게이트(plans/90 B4 D2).
/// 본체 t-루프와 GPU 실패 폴백 재계산이 공유 — 연산 순서·exp 선택(exp_cr) 불변.
#[allow(clippy::too_many_arguments)]
fn attn_head(
    qh: &[f32],
    gate: &[f32],
    cache_k: &[f32],
    cache_v: &[f32],
    n_past: usize,
    kvh: usize,
    n_kv: usize,
    hd: usize,
    kq_scale: f32,
    out: &mut [f32],
) {
    let mut scores = vec![0.0f32; n_past];
    let mut maxv = f32::NEG_INFINITY;
    for (p, sc) in scores.iter_mut().enumerate() {
        let b = p * n_kv * hd + kvh * hd;
        let mut d = 0.0f32;
        for i in 0..hd {
            d += qh[i] * cache_k[b + i];
        }
        *sc = d * kq_scale;
        maxv = maxv.max(*sc);
    }
    let mut sum = 0.0f32;
    for sc in scores.iter_mut() {
        *sc = crate::ops::exp_cr(*sc - maxv);
        sum += *sc;
    }
    for sc in scores.iter_mut() {
        *sc /= sum;
    }
    for p in 0..n_past {
        let w = scores[p];
        if w == 0.0 {
            continue;
        }
        let b = p * n_kv * hd + kvh * hd;
        for i in 0..hd {
            out[i] += w * cache_v[b + i];
        }
    }
    for i in 0..hd {
        out[i] *= sigmoid(gate[i]);
    }
}

pub(crate) fn attn_layer(
    ctx: &Ctx,
    seqs: &mut [SeqState],
        il: usize,
        xs: &[Vec<f32>],
        seq_ids: &[usize],
        t_len: usize,
        full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        profile_span!("cpu::layer_attn");
        let acc = ctx.acc.clone();
        let hp = &ctx.model.hp;
        let n_seqs = seq_ids.len();
        let n_tok = n_seqs * t_len;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = ctx.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
        let wk = ctx.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
        let wv = ctx.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
        let wo = ctx.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = ctx
            .model
            .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = ctx
            .model
            .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;

        if il == 3 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
            crate::qwen35::diag::a3_normed(&xs[0]);
        }
        // q·k·v 동일 입력 xs — 1그룹 배치
        let mut group: [Vec<Vec<f32>>; 3] = [
            vec![vec![0.0f32; wq.n_out as usize]; n_tok],
            vec![vec![0.0f32; wk.n_out as usize]; n_tok],
            vec![vec![0.0f32; wv.n_out as usize]; n_tok],
        ];
        {
            span_block!("cpu::attn_qkv", {
                mm_group(&acc, xs, &[wq, wk, wv], &mut group)?;
            });
        }
        let [qg, kk, vv] = group;

        let kq_scale = hp.kq_scale();
        // GPU 연결 (02-3): t=1 디코드 score/softmax/V-mix를 qsa_attention 커널
        // 재사용(마스크=prefix 전체). norm·rope·캐시 기록은 CPU 유지(저렴).
        // LLM170_ATTN_CPU=1 또는 실패 시 CPU 루프.
        let attn_gpu = acc.is_some() && t_len == 1 && n_seqs == 1
            && std::env::var_os("LLM170_ATTN_CPU").is_none();
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        let mut attn_all = vec![vec![0.0f32; n_head * hd]; n_tok];
        let mut gpu_qrow: Option<Vec<f32>> = None;
        let mut gpu_done = false;
        for s in 0..n_seqs {
            let pos0 = seqs[seq_ids[s]].pos;
            let seq = &mut seqs[seq_ids[s]];
            let (cache_k, cache_v) = (
                seq.kv_k[full_idx].as_mut_slice(),
                seq.kv_v[full_idx].as_mut_slice(),
            );

            for t in 0..t_len {
                let row = s * t_len + t;
                let pos = pos0 + t as u32;
                for h in 0..n_kv {
                    let src = kk[row][h * hd..h * hd + hd].to_vec();
                    let mut head = rms_norm(&src, &k_norm_w, hp.eps);
                    rope_head(&mut head, pos, n_rot, hp.rope_base);
                    let b = (pos as usize) * n_kv * hd + h * hd;
                    cache_k[b..b + hd].copy_from_slice(&head);
                    cache_v[b..b + hd].copy_from_slice(&vv[row][h * hd..h * hd + hd]);
                }
                if attn_gpu {
                    // q norm+rope → q‖gate 인터리브 플랫 [n_head·2·hd]
                    // (qsa_score/mix 커널 계약 — 게이트는 커널이 sigmoid 적용)
                    let mut qrow = vec![0.0f32; n_head * 2 * hd];
                    for h in 0..n_head {
                        let src = qg[row][h * 2 * hd..h * 2 * hd + hd].to_vec();
                        let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                        rope_head(&mut qh, pos, n_rot, hp.rope_base);
                        qrow[h * 2 * hd..h * 2 * hd + hd].copy_from_slice(&qh);
                        let gb = h * 2 * hd + hd;
                        qrow[gb..(hd + gb)].copy_from_slice(&qg[row][gb..(hd + gb)]);
                    }
                    gpu_qrow = Some(qrow);
                    continue;
                }
                let mut attn_out = std::mem::take(&mut attn_all[row]);
                let dbg3 = il == 3 && t == 0 && std::env::var_os("LLM170_DEBUG_LAYERS").is_some();
                if dbg3 {
                    crate::qwen35::diag::a3_cache(pos as usize, pos as usize * n_kv * hd, &cache_k, &cache_v, &qg[row], hd);
                }
                for h in 0..n_head {
                    let src = qg[row][h * 2 * hd..h * 2 * hd + hd].to_vec();
                    let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                    rope_head(&mut qh, pos, n_rot, hp.rope_base);
                    let kvh = h / (n_head / n_kv);
                    let ob = h * hd;
                    let gb = h * 2 * hd + hd;
                    attn_head(&qh, &qg[row][gb..gb + hd], &cache_k, &cache_v, pos as usize + 1, kvh, n_kv, hd, kq_scale, &mut attn_out[ob..ob + hd]);
                    if dbg3 && h == 0 {
                        eprintln!("  A3dbg h0 attn_out[0..4]={:?}", &attn_out[0..4]);
                    }
                }
                attn_all[row] = attn_out;
            }
        }
        if let Some(qrow) = gpu_qrow.as_ref()
            && let Some(acc_ref) = acc.as_deref() {
                let seq = &seqs[seq_ids[0]];
                let n_past = seq.pos as usize + 1;
                let ck = &seq.kv_k[full_idx][..n_past * n_kv * hd];
                let cv = &seq.kv_v[full_idx][..n_past * n_kv * hd];
                let mask: Vec<u32> = (0..n_past).map(|p| (p < n_past) as u32).collect();
                if let Ok(res) = acc_ref.qsa_attention(
                    qrow, ck, cv, &mask, kq_scale, n_past, n_head, n_kv, hd, 1,
                ) {
                    // 커널 출력에 게이트 이미 적용 (qsa_mix)
                    attn_all[0].copy_from_slice(&res[..n_head * hd]);
                    gpu_done = true;
                }
            }
        if gpu_qrow.is_some() && !gpu_done {
            // GPU 시도 실패 → CPU 재계산 (row 0, t=1)
            let pos = seqs[seq_ids[0]].pos;
            let seq = &seqs[seq_ids[0]];
            let (cache_k, cache_v) = (
                seq.kv_k[full_idx].as_slice(),
                seq.kv_v[full_idx].as_slice(),
            );
            let mut attn_out = std::mem::take(&mut attn_all[0]);
            for h in 0..n_head {
                let src = qg[0][h * 2 * hd..h * 2 * hd + hd].to_vec();
                let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                rope_head(&mut qh, pos, n_rot, hp.rope_base);
                let kvh = h / (n_head / n_kv);
                let ob = h * hd;
                let gb = h * 2 * hd + hd;
                attn_head(&qh, &qg[0][gb..gb + hd], cache_k, cache_v, pos as usize + 1, kvh, n_kv, hd, kq_scale, &mut attn_out[ob..ob + hd]);
            }
            attn_all[0] = attn_out;
        }
        // wo 프로젝션 — 전 토큰 배치 1회
        {
            span_block!("cpu::attn_wo", {
                mm_batch(&acc, &attn_all, &wo, &mut out)?;
            });
        }
        Ok(out)
    }
