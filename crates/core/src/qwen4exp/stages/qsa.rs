//! qsa(인덩서 top-k 게이트드 GQA) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use super::super::layers::SeqState4;
use crate::ops::{rms_norm, rope_head, sigmoid};
use llm170_profiler::profile_span;

    /// 토큰 1개의 CPU 마스크드 GQA + 게이트 — 루프 본문과 GPU 실패 폴백이 공용.
    /// 수치 경로는 원문 그대로(이동만).
    #[allow(clippy::too_many_arguments)]
    fn cpu_attn_row(
        attn_out: &mut [f32],
        q_t: &[f32],
        mask_t: &[bool],
        n_past: usize,
        cache_k: &[f32],
        cache_v: &[f32],
        n_head: usize,
        n_kv: usize,
        hd: usize,
        kq_scale: f32,
    ) {
        for h in 0..n_head {
            let kvh = h / (n_head / n_kv);
            let mut maxv = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; n_past];
            for (p, sc) in scores.iter_mut().enumerate() {
                if !mask_t[p] {
                    *sc = f32::NEG_INFINITY;
                    continue;
                }
                let b = p * n_kv * hd + kvh * hd;
                let mut d = 0.0f32;
                for i in 0..hd {
                    d += q_t[h * 2 * hd + i] * cache_k[b + i];
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
                attn_out[ob + i] *= sigmoid(q_t[gb + i]);
            }
        }
    }

/// 선택 목록에서 마스크를 복원한다 — `mask_all[t]`이 비어 있을 때만(즉
/// 마스크를 만들지 않은 GPU 경로에서 CPU 폴백이 걸릴 때) 호출된다.
fn mask_from_list(
    mask_all: &[Vec<bool>],
    sel_blk: &[u32],
    sel_cnt: &[u32],
    sel_stride: usize,
    r: usize,
    t: usize,
    n_past: usize,
) -> Vec<bool> {
    if !mask_all[t].is_empty() {
        return mask_all[t].clone();
    }
    let mut m = vec![false; n_past];
    let tail_start = (n_past / r) * r;
    for k2 in 0..sel_cnt[t] as usize {
        let b = sel_blk[t * sel_stride + k2] as usize;
        for j in b * r..(b + 1) * r {
            m[j] = true;
        }
    }
    for j in tail_start..n_past {
        m[j] = true;
    }
    m
}

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
        let w_t0 = std::time::Instant::now();
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
        let tm = std::env::var_os("LLM170_Q4_TIME").is_some();
        if tm {
            eprintln!(
                "# qsa-stage wlookup={:.1}ms",
                w_t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_all = std::time::Instant::now();
        let mut t_lap = t_all;
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
            if tm {
                eprintln!("# qsa-stage mm_group={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
                t_lap = std::time::Instant::now();
            }
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
        let mut gpu_attn = ctx.acc.is_some();
        // 마스크는 CPU 어텐션 경로에서만 쓴다 — GPU 경로는 선택 목록을 쓴다.
        // (행당 n_past bool을 2048행 만들면 24MB 할당 + 4.2M 채우기가 층마다 든다.)
        // GPU 경로에서 폴백이 걸리면 목록에서 그때 만든다(mask_of).
        let need_mask = !gpu_attn;
        let mut mask_all: Vec<Vec<bool>> = if need_mask {
            vec![vec![false; n_past_max]; n_tok]
        } else {
            vec![Vec::new(); n_tok]
        };
        let mut cpu_attn = false;   // GPU 어텐션 실패 시 CPU 폴백 (q4acc t>128 결함)

        let seq_state = &mut *seq;
        // 블록 키 캐시는 행마다 clone하지 않고 지역 버퍼로 승격한다(핫 루프 복사 제거).
        let mut bk_local: Vec<f32> = std::mem::take(&mut seq_state.idx_bk[full_idx]);
        let r = hp.compress[il] as usize;
        // 패스 A(직렬): KV·인덱서 캐시 적립 + q_rope(q norm·rope 포함).
        // 패스 B(병렬): 블록 점수 + 선택 + 마스크. 행이 서로 독립이라 층 단위로
        // 스레드를 한 번만 띄운다(행마다 spawn하면 오버헤드가 이득을 넘는다 — 실측).
        let mut q_rows: Vec<Vec<Vec<f32>>> = Vec::with_capacity(t_len);
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
            let n_blocks = n_past / r;
            // 블록 키 캐시 — 이 토큰 시점까지의 완전 블록만 유효 (증분).
            // 이전 청크가 계산한 키는 재사용, 신규 블록만 계산 (수치 동일).
            if bk_local.len() < n_blocks * hp.idx_dim {
                let mut b = bk_local.len() / hp.idx_dim;
                while b < n_blocks {
                    let mut pooled = vec![0.0f32; hp.idx_dim];
                    for j in 0..r {
                        let base = (b * r + j) * hp.idx_dim;
                        for i2 in 0..hp.idx_dim {
                            pooled[i2] += idx_cache[base + i2];
                        }
                    }
                    for v in pooled.iter_mut() {
                        *v /= r as f32;
                    }
                    let mut pk = rms_norm(&pooled, &ik_w, hp.eps);
                    rope_head(&mut pk, (b * r) as u32, hp.idx_dim, hp.rope_base);
                    bk_local.extend_from_slice(&pk);
                    b += 1;
                }
            }
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
            q_rows.push(q_rope);

            // q norm·rope를 qg에 즉시 적용 (attention은 패스 뒤 일괄)
            for h in 0..n_head {
                let src = qg[t][h * 2 * hd..h * 2 * hd + hd].to_vec();
                let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                rope_head(&mut qh, pos, n_rot, hp.rope_base);
                for (a, b) in qh.iter().zip(qg[t][h * 2 * hd..h * 2 * hd + hd].iter_mut()) {
                    *b = *a;
                }
            }
        }

        if tm {
            eprintln!("# qsa-stage passA={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
            t_lap = std::time::Instant::now();
        }
        // 패스 B — 행 단위 독립: 블록 점수 + 선택 + 마스크를 병렬로.
        // 선택된 블록은 (정렬해) 고정 보폭 배열에 적재한다 — 이어서 패스 C가
        // 오름차순 위치 목록으로 압축하고, GPU는 그 목록만 순회한다.
        let sel_stride = hp.idx_top_k / r + 2;
        let mut sel_blk: Vec<u32> = vec![0u32; t_len * sel_stride];
        let mut sel_cnt: Vec<u32> = vec![0u32; t_len];
        // 패스 B — 행 단위 독립: 블록 점수 + 선택 + 마스크를 병렬로.
        // 선택된 블록은 (정렬해) 고정 보폭 배열에 적재한다 — 이어서 패스 C가
        // 오름차순 위치 목록으로 압축하고, GPU는 그 목록만 순회한다.
        let sel_stride = hp.idx_top_k / r + 2;
        let mut sel_blk: Vec<u32> = vec![0u32; t_len * sel_stride];
        let mut sel_cnt: Vec<u32> = vec![0u32; t_len];
        // 선택 목록에서 마스크를 복원한다 (GPU 경로에서 CPU 폴백이 걸릴 때만).
        let mask_of = |t: usize, n_past: usize| -> Vec<bool> {
            if !mask_all[t].is_empty() {
                return mask_all[t].clone();
            }
            let mut m = vec![false; n_past];
            let tail_start = (n_past / r) * r;
            for k2 in 0..sel_cnt[t] as usize {
                let b = sel_blk[t * sel_stride + k2] as usize;
                for j in b * r..(b + 1) * r {
                    m[j] = true;
                }
            }
            for j in tail_start..n_past {
                m[j] = true;
            }
            m
        };
        {
            let bkl: &[f32] = &bk_local;
            let qr: &[Vec<Vec<f32>>] = &q_rows;
            let idx_dim = hp.idx_dim;
            let idx_top_k = hp.idx_top_k;
            // 16코어 × SMT = 32 논리 CPU — 캡을 두면 CPU 스테이지가 노는 동안
            // GPU가 굶는다(프로파일: 창의 ~25% 유휴). 토큰별 산술은 그대로라
            // 스레드 수는 수치에 영향이 없다.
            let nthreads = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4)
                .min(32);
            let per = t_len.div_ceil(nthreads.max(1)).max(1);
            let pos0u = pos0 as usize;
            std::thread::scope(|sc| {
                for (ci, (chunk, (blk_chunk, cnt_chunk))) in mask_all
                    .chunks_mut(per)
                    .zip(sel_blk.chunks_mut(per * sel_stride).zip(sel_cnt.chunks_mut(per)))
                    .enumerate()
                {
                    let base = ci * per;
                    sc.spawn(move || {
                        for ((i, slot), (bslot, cslot)) in chunk
                            .iter_mut()
                            .enumerate()
                            .zip(blk_chunk.chunks_mut(sel_stride).zip(cnt_chunk.iter_mut()))
                        {
                            let t = base + i;
                            let n_past = pos0u + t + 1;
                            let n_blocks = n_past / r;
                            let tail_start = n_blocks * r;
                            let bk = &bkl[..n_blocks * idx_dim];
                            let mut block_score = vec![0.0f32; n_blocks];
                            for b in 0..n_blocks {
                                let pk = &bk[b * idx_dim..(b + 1) * idx_dim];
                                for qh in &qr[t] {
                                    // 4-누산기로 펼쳐 의존 사슬을 끊는다.
                                    let (mut d0, mut d1, mut d2, mut d3) =
                                        (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                                    let mut i2 = 0usize;
                                    while i2 + 4 <= idx_dim {
                                        d0 += qh[i2] * pk[i2];
                                        d1 += qh[i2 + 1] * pk[i2 + 1];
                                        d2 += qh[i2 + 2] * pk[i2 + 2];
                                        d3 += qh[i2 + 3] * pk[i2 + 3];
                                        i2 += 4;
                                    }
                                    while i2 < idx_dim {
                                        d0 += qh[i2] * pk[i2];
                                        i2 += 1;
                                    }
                                    let dot = (d0 + d1) + (d2 + d3);
                                    if dot > 0.0 {
                                        block_score[b] += dot;
                                    }
                                }
                            }
                            // 선택: 테일(강제) + 상위 B개 완전블록 — 폭 = min(n_past, top_k + r − 1)
                            let width = n_past.min(idx_top_k + r - 1);
                            let tail_cnt = n_past - tail_start;
                            let n_sel_blocks = ((width - tail_cnt) / r).min(n_blocks);
                            let mut sel_blocks: Vec<usize> = (0..n_blocks).collect();
                            if n_sel_blocks < n_blocks {
                                // 상위 n_sel_blocks개만 필요 — 전체 정렬 대신 부분 선택(평균 O(n)).
                                sel_blocks.select_nth_unstable_by(n_sel_blocks, |&a, &b| {
                                    block_score[b]
                                        .partial_cmp(&block_score[a])
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                            }
                            let mut mask = if need_mask { vec![false; n_past] } else { Vec::new() };
                            if need_mask {
                                for j in tail_start..n_past {
                                    mask[j] = true;
                                }
                                for &b in &sel_blocks[..n_sel_blocks] {
                                    for j in b * r..(b + 1) * r {
                                        mask[j] = true;
                                    }
                                }
                            }
                            // 목록용: 선택 블록을 오름차순으로 고정 보폭 배열에.
                            let mut sb: Vec<usize> = sel_blocks[..n_sel_blocks].to_vec();
                            sb.sort_unstable();
                            for (k2, &b) in sb.iter().enumerate() {
                                bslot[k2] = b as u32;
                            }
                            *cslot = n_sel_blocks as u32;
                            *slot = mask;
                        }
                    });
                }
            });
        }
        // 패스 C — GPU 어텐션 미사용 시 CPU 어텐션(폴백 경로).
        if !(gpu_attn && !cpu_attn) {
            let (ckv, cvv) = (&seq_state.kv_k[full_idx], &seq_state.kv_v[full_idx]);
            for t in 0..t_len {
                let n_past = (pos0 as usize) + t + 1;
                let mut attn_out = std::mem::take(&mut attn_all[t]);
                let m_t = mask_from_list(&mask_all, &sel_blk, &sel_cnt, sel_stride, r, t, n_past);
                cpu_attn_row(
                    &mut attn_out, &qg[t], &m_t, n_past, ckv, cvv,
                    n_head, n_kv, hd, kq_scale,
                );
                attn_all[t] = attn_out;
            }
        }
        seq_state.idx_bk[full_idx] = bk_local;
        if tm {
            eprintln!("# qsa-stage sel+proj={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
            t_lap = std::time::Instant::now();
        }
        // GPU 일괄 마스크 GQA — 캐시 전체(≤n_past_max)와 토큰별 마스크 전달.
        // 미래 위치는 mask 0으로 차단 (토큰 t는 pos_t+1까지만 참석).
        if gpu_attn {
            if let Some(acc) = ctx.acc.as_deref() {
                if tm {
                    eprintln!("# qsa-stage passB={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
                    t_lap = std::time::Instant::now();
                }
                let qflat: Vec<f32> = qg.iter().flatten().copied().collect();
                // 선택 목록 압축 — 블록(오름차순) + 테일. 위치는 오름차순이므로
                // 마스크 스캔과 산술 순서가 같다(프로브에서 비트 동일 확인).
                let mut sel_off: Vec<u32> = vec![0u32; n_tok + 1];
                for t2 in 0..n_tok {
                    let n_past = pos0 as usize + t2 + 1;
                    let tail_cnt = n_past - (n_past / r) * r;
                    sel_off[t2 + 1] = sel_off[t2] + sel_cnt[t2] * r as u32 + tail_cnt as u32;
                }
                let mut sel_idx: Vec<u32> = vec![0u32; sel_off[n_tok] as usize];
                for t2 in 0..n_tok {
                    let n_past = pos0 as usize + t2 + 1;
                    let tail_start = (n_past / r) * r;
                    let mut o = sel_off[t2] as usize;
                    for k2 in 0..sel_cnt[t2] as usize {
                        let b = sel_blk[t2 * sel_stride + k2] as usize;
                        for j in 0..r {
                            sel_idx[o] = (b * r + j) as u32;
                            o += 1;
                        }
                    }
                    for j in tail_start..n_past {
                        sel_idx[o] = j as u32;
                        o += 1;
                    }
                }
                // 전체 ctx clone은 디코드 스텝당 ~800MB 복사 — 사용 prefix만.
                let kn = n_past_max * n_kv * hd;
                let ck = seq.kv_k[full_idx][..kn].to_vec();
                let cv = seq.kv_v[full_idx][..kn].to_vec();
                // 미지원이면 CPU 어텐션 폴백 — gdn_ar과 같은 규약.
                match acc.qsa_attention_sel(
                    &qflat, &ck[..n_past_max * n_kv * hd], &cv[..n_past_max * n_kv * hd],
                    &sel_idx, &sel_off, kq_scale, n_head, n_kv, hd, n_tok,
                ) {
                    Ok(res) => {
                        for (t, row) in attn_all.iter_mut().enumerate() {
                            row.copy_from_slice(&res[t * n_head * hd..(t + 1) * n_head * hd]);
                        }
                    }
                    Err(e) => {
                        // 12개 QSA 층 × 매 호출로 반복되므로 프로세스당 1회만 알린다.
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        if std::env::var_os("LLM170_Q4_NOFAST").is_none() {
                            ONCE.call_once(|| eprintln!("# qsa: GPU 어텐션 폴백 — CPU 재계산 ({e})"));
                        }
                        // 폴백은 실제로 CPU 재계산을 해야 한다 — 이전 구현은
                        // 행을 빈 채로 두어 12개 QSA 층의 어텐션이 조용히
                        // 누락됐다(2026-09-13 발견: 프레임==값 자가일치 통과).
                        cpu_attn = true;
                        for (t, row) in attn_all.iter_mut().enumerate() {
                            let n_past = (pos0 as usize) + t + 1;
                            let mask_t =
                                mask_from_list(&mask_all, &sel_blk, &sel_cnt, sel_stride, r, t, n_past);
                            let q_t = qg[t].clone();
                            let mut attn_out = std::mem::take(row);
                            cpu_attn_row(
                                &mut attn_out, &q_t, &mask_t, n_past, &ck, &cv,
                                n_head, n_kv, hd, kq_scale,
                            );
                            *row = attn_out;
                        }
                    }
                }
            }
        }
        ctx.mm_batch(&attn_all, &wo, &mut out)?;
        if tm {
            eprintln!(
                "# qsa-stage t={t_len} attn={:.1}ms out_mm={:.1}ms total={:.1}ms",
                t_lap.elapsed().as_secs_f64() * 1e3,
                0.0,
                t_all.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(out)
    }
