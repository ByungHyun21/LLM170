//! qsa(인덩서 top-k 게이트드 GQA) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).
#![allow(dead_code)] // 프론트 정리(2026-09-14): 레거시·진단 경로 보존

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
    /// plans/67 2b: 선택부만 추출 — 투영 결과를 받아 (1) 인덱서 norm·rope와
    /// KV/idx 캐시 적립, (2) 블록 키 풀링, (3) 블록 점수·top-k 선택을 수행하고
    /// 선택 블록 목록을 반환한다. **q/k의 attention norm·rope는 포함하지 않는다**
    /// (새 경로는 디바이스가 수행 — frame_qk_norm_rope). `qsa_layer`가 이 함수를
    /// 호출하므로 동작 불변(diverse 게이트로 확인).
    pub fn qsa_select(
        ctx: &Ctx,
        seq_state: &mut SeqState4,
        il: usize,
        kk: &[Vec<f32>],
        vv: &[Vec<f32>],
        iq: &[Vec<f32>],
        ik: &[Vec<f32>],
        t_len: usize,
        full_idx: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, usize), Q4Error> {
        let hp = &ctx.model.hp;
        let (n_kv, hd) = (hp.n_kv, hp.head_dim);
        let (n_rot, idx_dim, idx_heads) = (hp.n_rot, hp.idx_dim, hp.idx_heads);
        let (rope_base, eps) = (hp.rope_base, hp.eps);
        let r = hp.compress[il] as usize;
        let knw = &ctx.model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
        let iqw = &ctx.model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
        let ikw = &ctx.model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        let pos0 = seq_state.pos;
        let mut bk_local: Vec<f32> = std::mem::take(&mut seq_state.idx_bk[full_idx]);
        // 패스 A: KV·인덱서 캐시 적립 + 인덱서 q_rope(norm·rope).
        let mut q_rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); t_len];
        {
            let nthreads_a = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4)
                .min(32);
            let per_a = t_len.div_ceil(nthreads_a.max(1)).max(1);
            let pos0u = pos0 as usize;
            let skip_kv = pos0u * n_kv * hd;
            let skip_idx = pos0u * idx_dim;
            let st = &mut *seq_state;
            let (kv_k, kv_v, idx_k) = (
                st.kv_k[full_idx].as_mut_slice(),
                st.kv_v[full_idx].as_mut_slice(),
                st.idx_k[full_idx].as_mut_slice(),
            );
            std::thread::scope(|sc| {
                for (ci, (((kc, vc), ic), qc)) in kv_k[skip_kv..]
                    .chunks_mut(per_a * n_kv * hd)
                    .zip(kv_v[skip_kv..].chunks_mut(per_a * n_kv * hd))
                    .zip(idx_k[skip_idx..].chunks_mut(per_a * idx_dim))
                    .zip(q_rows.chunks_mut(per_a))
                    .enumerate()
                {
                    let base = ci * per_a;
                    sc.spawn(move || {
                        for (i, (((kch, vch), ich), qslot)) in kc
                            .chunks_mut(n_kv * hd)
                            .zip(vc.chunks_mut(n_kv * hd))
                            .zip(ic.chunks_mut(idx_dim))
                            .zip(qc.iter_mut())
                            .enumerate()
                        {
                            let t = base + i;
                            if t >= t_len { break; }
                            let pos = pos0u as u32 + t as u32;
                            for h in 0..n_kv {
                                let lo = h * hd;
                                let mut head = rms_norm(&kk[t][lo..lo + hd], knw, eps);
                                rope_head(&mut head, pos, n_rot, rope_base);
                                kch[lo..lo + hd].copy_from_slice(&head);
                                vch[lo..lo + hd].copy_from_slice(&vv[t][lo..lo + hd]);
                            }
                            ich.copy_from_slice(&ik[t][..idx_dim]);
                            let mut qr: Vec<Vec<f32>> = Vec::with_capacity(idx_heads);
                            for h in 0..idx_heads {
                                let lo = h * idx_dim;
                                let mut qh = rms_norm(&iq[t][lo..lo + idx_dim], iqw, eps);
                                rope_head(&mut qh, pos, idx_dim, rope_base);
                                qr.push(qh);
                            }
                            *qslot = qr;
                        }
                    });
                }
            });
        }
        // 블록 키 캐시 — 청크 끝까지의 완전 블록을 병렬로(증분, 원본과 동일 산술:
        // 블록 내 r개 인덱서 k의 mean-pool → rms_norm → rope(pos = 블록 시작)).
        {
            let n_blocks_max = (pos0 as usize + t_len) / r;
            if bk_local.len() < n_blocks_max * idx_dim {
                let dim = idx_dim;
                let b0 = bk_local.len() / dim;
                bk_local.resize(n_blocks_max * dim, 0.0);
                let idx_all: &[f32] = seq_state.idx_k[full_idx].as_slice();
                let nthreads_b = std::thread::available_parallelism()
                    .map(|v| v.get())
                    .unwrap_or(4)
                    .min(32);
                let per_b = (n_blocks_max - b0).div_ceil(nthreads_b.max(1)).max(1);
                std::thread::scope(|sc| {
                    for (ci, chunk) in bk_local[b0 * dim..].chunks_mut(per_b * dim).enumerate() {
                        let base = b0 + ci * per_b;
                        sc.spawn(move || {
                            for (i, slot) in chunk.chunks_mut(dim).enumerate() {
                                let b = base + i;
                                if b >= n_blocks_max {
                                    break;
                                }
                                let mut pooled = vec![0.0f32; dim];
                                for j in 0..r {
                                    let src = (b * r + j) * dim;
                                    for i2 in 0..dim {
                                        pooled[i2] += idx_all[src + i2];
                                    }
                                }
                                for v in pooled.iter_mut() {
                                    *v /= r as f32;
                                }
                                let mut pk = rms_norm(&pooled, ikw, eps);
                                rope_head(&mut pk, (b * r) as u32, dim, rope_base);
                                slot.copy_from_slice(&pk);
                            }
                        });
                    }
                });
            }
        }
        seq_state.idx_bk[full_idx] = bk_local;
        // 패스 B: 블록 점수 + 선택(top-k).
        let sel_stride = hp.idx_top_k / r + 2;
        let mut sel_blk: Vec<u32> = vec![0u32; t_len * sel_stride];
        let mut sel_cnt: Vec<u32> = vec![0u32; t_len];
        {
            let bkl: &[f32] = &seq_state.idx_bk[full_idx];
            let qr: &[Vec<Vec<f32>>] = &q_rows;
            let idx_top_k = hp.idx_top_k;
            let nthreads = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4)
                .min(32);
            let per = t_len.div_ceil(nthreads.max(1)).max(1);
            let pos0u = pos0 as usize;
            std::thread::scope(|sc| {
                for (ci, (blk_chunk, cnt_chunk)) in sel_blk
                    .chunks_mut(per * sel_stride)
                    .zip(sel_cnt.chunks_mut(per))
                    .enumerate()
                {
                    let base = ci * per;
                    sc.spawn(move || {
                        for (i, (bslot, cslot)) in blk_chunk
                            .chunks_mut(sel_stride)
                            .zip(cnt_chunk.iter_mut())
                            .enumerate()
                        {
                            let t = base + i;
                            if t >= t_len { break; }
                            let n_past = pos0u + t + 1;
                            let n_blocks = n_past / r;
                            let tail_start = n_blocks * r;
                            let bk = &bkl[..n_blocks * idx_dim];
                            let mut block_score = vec![0.0f32; n_blocks];
                            for b in 0..n_blocks {
                                let pk = &bk[b * idx_dim..(b + 1) * idx_dim];
                                for qh in &qr[t] {
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
                                    if dot > 0.0 { block_score[b] += dot; }
                                }
                            }
                            let width = n_past.min(idx_top_k + r - 1);
                            let tail_cnt = n_past - tail_start;
                            let n_sel_blocks = ((width - tail_cnt) / r).min(n_blocks);
                            let mut sel_blocks: Vec<usize> = (0..n_blocks).collect();
                            if n_sel_blocks < n_blocks {
                                sel_blocks.select_nth_unstable_by(n_sel_blocks, |&a, &b| {
                                    block_score[b]
                                        .partial_cmp(&block_score[a])
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                            }
                            let mut sb: Vec<usize> = sel_blocks[..n_sel_blocks].to_vec();
                            sb.sort_unstable();
                            for (k2, &b) in sb.iter().enumerate() {
                                bslot[k2] = b as u32;
                            }
                            *cslot = n_sel_blocks as u32;
                        }
                    });
                }
            });
        }
        Ok((sel_blk, sel_cnt, sel_stride))
    }

    #[allow(unused_assignments)] // 진단 코드의 중간 변수
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
        let _k_norm_w = ctx.model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
        let _iq_w = ctx.model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
        let _ik_w = ctx.model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
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
            // 실측(2026-09-14, pp8192, t 라벨로 분리한 콜당 값):
            //   t=1   : attn 2.55 · sel_build 1.39 · sel+proj 0.69 · mm_group 0.40 = 5.03ms
            //           -> 스텝(12층) 60.4ms = 장문맥 디코드 프레임 144.2ms의 42%
            //   t=2048: attn 100.97 · mm_group 72.76 · sel+proj 41.89 · sel_build 32.66
            //           = 248.3ms/콜 -> 청크 12층 2.98s = 청크 8.9s의 33%
            // 라벨 없이 합산하면 두 체제가 섞여 오염된 평균이 나온다(과거 5.0s/57%,
            // 58% 주장의 원인). 최대 항목은 양쪽 모두 **attn**이고, KTRACE로 본
            // 어텐션 커널 자체는 1.425ms/콜 = 17.1ms/스텝(장문맥 디코드).
            // 확인된 것(pp2048 청크 전용): QSA 호스트 스테이지는 5.0s이고
            // mm_group 38% + attn 37%(둘 다 가속기 작업) + sel_build 14% +
            // sel+proj 10%다. 전송(d2h/h2d)은 0.07s로 무죄였다.
            // sel_build의 버퍼 재사용은 중립(0.70 vs 0.71s) — 비용은 평탄화된
            // 선택목록 물질화 자체라 커널이 블록 목록을 직접 순회해야 줄어든다.
            ctx.mm_group(xs, &[wq, wk, wv, w_iq, w_ik], &mut gi)?;
            if tm {
                eprintln!("# qsa-stage t={t_len} mm_group={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
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
        let gpu_attn = ctx.acc.is_some();
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

        // 블록 키 캐시는 행마다 clone하지 않고 지역 버퍼로 승격한다(핫 루프 복사 제거).
        // plans/67 2b(후반): 선택부를 추출본 qsa_select에 위임 — 캐시 적립(Pass A)·
        // 블록 키 풀링·top-k 선택(Pass B)이 거기에 있고 산술은 원본과 동일하다.
        // 여기는 q rope + CPU 폴백 마스크 복원 + 패스 C만 남는다.
        let (sel_blk, sel_cnt, sel_stride) = qsa_select(
            ctx, seq, il, &kk, &vv, &iq, &ik, t_len, full_idx,
        )?;
        let r = hp.compress[il] as usize;
        // q norm·rope를 qg에 적용 — 원본과 동일 산술(어텐션은 패스 뒤 일괄).
        // 패스 A 병렬화 때 이 루프가 누락되어 어텐션이 비정규화 q를 쓰는 회귀가 있었다
        // (diverse 프롬프트 감사로 발견: out 해시가 갈렸다).
        {
            let nthreads_q = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4)
                .min(32);
            let per_q = t_len.div_ceil(nthreads_q.max(1)).max(1);
            let pos0u = pos0 as usize;
            let qn: &[f32] = &q_norm_w;
            let (n_rot_l, rope_base, eps) = (n_rot, hp.rope_base, hp.eps);
            std::thread::scope(|sc| {
                for (ci, chunk) in qg.chunks_mut(per_q).enumerate() {
                    let base = ci * per_q;
                    sc.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            let t = base + i;
                            if t >= t_len {
                                break;
                            }
                            let pos = pos0u as u32 + t as u32;
                            for h in 0..n_head {
                                let lo = h * 2 * hd;
                                let mut qh = rms_norm(&row[lo..lo + hd], qn, eps);
                                rope_head(&mut qh, pos, n_rot_l, rope_base);
                                row[lo..lo + hd].copy_from_slice(&qh);
                            }
                        }
                    });
                }
            });
        }
        // CPU 폴백용 마스크 복원(GPU 경로에서 폴백이 걸릴 때만 필요 — 선택 목록에서 만든다).
        let mask_of = |t: usize, n_past: usize| -> Vec<bool> {
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
        // CPU 어텐션 경로(need_mask)에서는 mask_all을 선택 목록에서 복원해 채운다 —
        // 원본 Pass B가 마스크를 같이 채웠지만 추출본에는 목록만 있다(2026-09-14).
        if need_mask {
            for t in 0..n_tok {
                let n_past = pos0 as usize + t + 1;
                mask_all[t] = mask_of(t, n_past);
            }
        }
        // 패스 C — GPU 어텐션 미사용 시 CPU 어텐션(폴백 경로).
        if !(gpu_attn && !cpu_attn) {
            let (ckv, cvv) = (&seq.kv_k[full_idx], &seq.kv_v[full_idx]);
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
        if tm {
            eprintln!("# qsa-stage t={t_len} sel+proj={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
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
                // 사용 prefix만 — 그리고 **복사하지 않는다**: 과거에는 여기서
                // .to_vec()으로 33.6MB/층(n_past 8192 기준)을 매 호출 복사했고,
                // 그 memcpy가 sel_build 타이머의 실체였다(실측 1.39ms/층 = 24GB/s).
                // 어댑터는 &[f32]를 받으므로 대여로 충분하다.
                // 실측 효과(pp8192 tg8): 1,019.9 -> 738.4ms = **-27.6%**(스텝 92.3ms).
                // 예상(16.7ms/스텝)보다 큰 것은 복사가 L2/L3를 밀어내 뒤따르는 커널까지
                // 느리게 했기 때문이다. 출력 토큰은 완전히 동일(순수 리팩터).
                let kn = n_past_max * n_kv * hd;
                let ck = &seq.kv_k[full_idx][..kn];
                let cv = &seq.kv_v[full_idx][..kn];
                if tm {
                    eprintln!("# qsa-stage t={t_len} sel_build={:.1}ms", t_lap.elapsed().as_secs_f64() * 1e3);
                    t_lap = std::time::Instant::now();
                }
                // 미지원이면 CPU 어텐션 폴백 — gdn_ar과 같은 규약.
                match acc.qsa_attention_sel(
                    &qflat, ck, cv,
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
        if std::env::var_os("LLM170_QSA_HASH").is_some() {
            let used_kv = (((pos0 as usize) + t_len) * n_kv * hd).min(seq.kv_k[full_idx].len());
            let used_idx = (((pos0 as usize) + t_len) * hp.idx_dim).min(seq.idx_k[full_idx].len());
            let h = |v: &[f32]| -> u64 {
                let mut x = 0xcbf29ce484222325u64;
                for f in v.iter() {
                    x ^= f.to_bits() as u64;
                    x = x.wrapping_mul(0x100000001b3);
                }
                x
            };
            let qflat: Vec<f32> = qg.iter().flatten().copied().collect();
            let oflat: Vec<f32> = out.iter().flatten().copied().collect();
            eprintln!(
                "# qsa-hash il={il} kv={:016x} idx={:016x} bk={:016x} q={:016x} out={:016x}",
                h(&seq.kv_k[full_idx][..used_kv]),
                h(&seq.idx_k[full_idx][..used_idx]),
                h(&seq.idx_bk[full_idx]),
                h(&qflat),
                h(&oflat),
            );
        }
        Ok(out)
    }
