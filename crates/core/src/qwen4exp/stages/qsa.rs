//! qsa(인덩서 top-k 게이트드 GQA) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use super::super::layers::SeqState4;
use crate::matmul::Accelerator;
use crate::ops::{rms_norm, rope_head, sigmoid};
use llm170_profiler::profile_span;

    /// QSA층 — 인덱서 top-k 마스크 게이트드 GQA.
    pub fn qsa_layer(
        ctx: &Ctx,
        seq: &mut SeqState4,
        il: usize,
        xs: &[Vec<f32>],
        t_len: usize,
        full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::layer_qsa");
        let hp = ctx.model.hp.clone();
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = ctx.model.w4(&format!("blk.{il}.attn_q.weight"))?;
        let wk = ctx.model.w4(&format!("blk.{il}.attn_k.weight"))?;
        let wv = ctx.model.w4(&format!("blk.{il}.attn_v.weight"))?;
        let wo = ctx.model.w4(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = ctx.model.f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = ctx.model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
        let iq_w = ctx.model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
        let ik_w = ctx.model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        let w_iq = ctx.model.w4(&format!("blk.{il}.indexer.q_proj.weight"))?;
        let w_ik = ctx.model.w4(&format!("blk.{il}.indexer.k_proj.weight"))?;

        let n_tok = t_len;
        // q/k/v/iq/ik는 동일 입력 xs — 그룹 1호출 (왕복 5→1).
        let mut qg = vec![vec![0.0f32; wq.n_out as usize]; n_tok];
        let mut kk = vec![vec![0.0f32; wk.n_out as usize]; n_tok];
        let mut vv = vec![vec![0.0f32; wv.n_out as usize]; n_tok];
        let mut iq = vec![vec![0.0f32; w_iq.n_out as usize]; n_tok];
        let mut ik = vec![vec![0.0f32; w_ik.n_out as usize]; n_tok];
        {
            let mut gi = vec![
                std::mem::take(&mut qg),
                std::mem::take(&mut kk),
                std::mem::take(&mut vv),
                std::mem::take(&mut iq),
                std::mem::take(&mut ik),
            ];
            ctx.mm_group(xs, &[wq, wk, wv, w_iq, w_ik], &mut gi)?;
            qg = std::mem::take(&mut gi[0]);
            kk = std::mem::take(&mut gi[1]);
            vv = std::mem::take(&mut gi[2]);
            iq = std::mem::take(&mut gi[3]);
            ik = std::mem::take(&mut gi[4]);
        }

        let kq_scale = hp.kq_scale();
        let pos0 = seq.pos;
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        let mut attn_all = vec![vec![0.0f32; n_head * hd]; n_tok];
        let n_past_max = (pos0 as usize) + t_len;
        let gpu_attn = ctx.acc.is_some();
        let mut mask_all: Vec<Vec<bool>> = vec![vec![false; n_past_max]; n_tok];

        let seq_state = &mut *seq;
        for t in 0..t_len {
            let pos = pos0 + t as u32;
            let (cache_k, cache_v, idx_cache) = {
                let st = &mut seq_state.kv_k[full_idx];
                let st2 = &mut seq_state.kv_v[full_idx];
                let st3 = &mut seq_state.idx_k[full_idx];
                // 안전 분할: 세 벡터는 서로 다른 필드 — std::split_at_mut 불필요
                (st.as_mut_slice(), st2.as_mut_slice(), st3.as_mut_slice())
            };
            let n_past = pos as usize + 1;

            // K/V 캐시 적립 + 인덱서 raw k 캐시
            for h in 0..n_kv {
                let src = kk[t][h * hd..h * hd + hd].to_vec();
                let mut head = rms_norm(&src, &k_norm_w, hp.eps);
                rope_head(&mut head, pos, n_rot, hp.rope_base);
                let b = pos as usize * n_kv * hd + h * hd;
                cache_k[b..b + hd].copy_from_slice(&head);
                cache_v[b..b + hd].copy_from_slice(&vv[t][h * hd..h * hd + hd]);
            }
            idx_cache[pos as usize * hp.idx_dim..(pos as usize + 1) * hp.idx_dim]
                .copy_from_slice(&ik[t]);

            // 인덱서 스코어: 완전 블록(4토큰) mean-pool → rms → rope(b*4) → ReLU 헤드합
            let r = hp.compress[il] as usize;
            let n_blocks = n_past / r;
            let tail_start = n_blocks * r;
            let mut q_rope: Vec<Vec<f32>> = Vec::with_capacity(hp.idx_heads);
            for h in 0..hp.idx_heads {
                let mut qh = rms_norm(
                    &iq[t][h * hp.idx_dim..(h + 1) * hp.idx_dim].to_vec(),
                    &iq_w,
                    hp.eps,
                );
                rope_head(&mut qh, pos, hp.idx_dim, hp.rope_base);
                q_rope.push(qh);
            }
            let mut block_score = vec![0.0f32; n_blocks];
            for b in 0..n_blocks {
                let mut pooled = vec![0.0f32; hp.idx_dim];
                for j in 0..r {
                    let base = (b * r + j) * hp.idx_dim;
                    for i in 0..hp.idx_dim {
                        pooled[i] += idx_cache[base + i];
                    }
                }
                for v in pooled.iter_mut() {
                    *v /= r as f32;
                }
                let mut pk = rms_norm(&pooled, &ik_w, hp.eps);
                rope_head(&mut pk, (b * r) as u32, hp.idx_dim, hp.rope_base);
                for qh in &q_rope {
                    let mut dot = 0.0f32;
                    for i in 0..hp.idx_dim {
                        dot += qh[i] * pk[i];
                    }
                    if dot > 0.0 {
                        block_score[b] += dot;
                    }
                }
            }

            // 선택: 테일(강제) + 상위 B개 완전블록 — 폭 = min(n_past, top_k + r − 1)
            let width = n_past.min(hp.idx_top_k + r - 1);
            let tail_cnt = n_past - tail_start;
            let mut sel_blocks: Vec<usize> = (0..n_blocks).collect();
            sel_blocks.sort_by(|&a, &b| block_score[b].partial_cmp(&block_score[a]).unwrap());
            let n_sel_blocks = ((width - tail_cnt) / r).min(n_blocks);
            let mut mask = vec![false; n_past];
            for j in tail_start..n_past {
                mask[j] = true;
            }
            for &b in &sel_blocks[..n_sel_blocks] {
                for j in b * r..(b + 1) * r {
                    mask[j] = true;
                }
            }

            // q norm·rope를 qg에 즉시 적용 (attention은 루프 후 일괄)
            for h in 0..n_head {
                let src = qg[t][h * 2 * hd..h * 2 * hd + hd].to_vec();
                let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                rope_head(&mut qh, pos, n_rot, hp.rope_base);
                for (a, b) in qh.iter().zip(qg[t][h * 2 * hd..h * 2 * hd + hd].iter_mut()) {
                    *b = *a;
                }
            }
            mask_all[t] = mask;

            // 마스크 밀집 GQA + 게이트 — acc 있으면 루프 후 GPU 일괄, 없으면 즉시 CPU
            if gpu_attn {
                continue;
            }
            let mut attn_out = std::mem::take(&mut attn_all[t]);
            for h in 0..n_head {
                let kvh = h / (n_head / n_kv);
                let mut maxv = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_past];
                for (p, sc) in scores.iter_mut().enumerate() {
                    if !mask_all[t][p] {
                        *sc = f32::NEG_INFINITY;
                        continue;
                    }
                    let b = p * n_kv * hd + kvh * hd;
                    let mut d = 0.0f32;
                    for i in 0..hd {
                        d += qg[t][h * 2 * hd + i] * cache_k[b + i];
                    }
                    *sc = d * kq_scale;
                    maxv = maxv.max(*sc);
                }
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - maxv).exp();
                    sum += *sc;
                }
                let ob = h * hd;
                for (p, sc) in scores.iter().enumerate() {
                    let w = sc / sum;
                    if w == 0.0 {
                        continue;
                    }
                    let b = p * n_kv * hd + kvh * hd;
                    for i in 0..hd {
                        attn_out[ob + i] += w * cache_v[b + i];
                    }
                }
                let gb = h * 2 * hd + hd;
                for i in 0..hd {
                    attn_out[ob + i] *= sigmoid(qg[t][gb + i]);
                }
            }
            attn_all[t] = attn_out;
        }
        // GPU 일괄 마스크 GQA — 캐시 전체(≤n_past_max)와 토큰별 마스크 전달.
        // 미래 위치는 mask 0으로 차단 (토큰 t는 pos_t+1까지만 참석).
        if gpu_attn {
            if let Some(acc) = ctx.acc.as_deref() {
                let qflat: Vec<f32> = qg.iter().flatten().copied().collect();
                let mut masku32: Vec<u32> = Vec::with_capacity(n_tok * n_past_max);
                for t in 0..t_len {
                    let n_past = (pos0 as usize) + t + 1;
                    for p in 0..n_past_max {
                        masku32.push((p < n_past && mask_all[t][p]) as u32);
                    }
                }
                // 전체 ctx clone은 디코드 스텝당 ~800MB 복사 — 사용 prefix만.
                let kn = n_past_max * n_kv * hd;
                let ck = seq.kv_k[full_idx][..kn].to_vec();
                let cv = seq.kv_v[full_idx][..kn].to_vec();
                let res = acc
                    .qsa_attention(
                        &qflat, &ck[..n_past_max * n_kv * hd], &cv[..n_past_max * n_kv * hd],
                        &masku32, kq_scale, n_past_max, n_head, n_kv, hd, n_tok,
                    )
                    .map_err(Q4Error::Io)?;
                for (t, row) in attn_all.iter_mut().enumerate() {
                    row.copy_from_slice(&res[t * n_head * hd..(t + 1) * n_head * hd]);
                }
            }
        }
        ctx.mm_batch(&attn_all, &wo, &mut out)?;
        Ok(out)
    }
