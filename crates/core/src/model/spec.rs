//! Engine 스펙/MTP 오케스트레이션 — mod.rs에서 분리(plans/35 P5).
//! draft·검증·GPU 스펙 경로와 접두 캐시 플러시.

use super::*;

impl Engine {
    pub fn mtp_step(
        &mut self,
        seq: usize,
        token: u32,
        h_in: &[f32],
        pos: u32,
        with_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        profile_span!("cpu::mtp_forward");
        let hp = self.model.hp.clone();
        let n_embd = hp.n_embd;
        let il = 64; // blk.64 — MTP층
        let w_eh = self.model.wchk(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let enorm = self.model.f32_vec(&format!("blk.{il}.nextn.enorm.weight"))?;
        let hnorm = self.model.f32_vec(&format!("blk.{il}.nextn.hnorm.weight"))?;

        // 1) embd(tok) 디양자화 → enorm / h_t → hnorm, concat → eh_proj
        let embd = self.model.wchk("token_embd.weight")?;
        let mut tok_row = vec![0.0f32; n_embd];
        crate::quant::dequant_row(embd.ty, embd.data, token as u64, n_embd as u64, &mut tok_row);
        let e_n = crate::ops::rms_norm(&tok_row, &enorm, hp.eps);
        let h_n = crate::ops::rms_norm(h_in, &hnorm, hp.eps);
        let mut cat = vec![0.0f32; 2 * n_embd];
        cat[..n_embd].copy_from_slice(&e_n);
        cat[n_embd..].copy_from_slice(&h_n);
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] cat e0={:.6} e1={:.6} esum={:.4} | h0={:.6} h1={:.6} hsum={:.4}", e_n[0], e_n[1], e_n.iter().map(|&x| x as f64).sum::<f64>(), h_n[0], h_n[1], h_n.iter().map(|&x| x as f64).sum::<f64>());
        }
        let acc = self.acc.clone();
        let mut cur = vec![0.0f32; n_embd];
        crate::matmul::mm(&acc, &cat, &w_eh, &mut cur)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] eh sum={:.5} x0={:.5} x1={:.5}", cur.iter().map(|&x| x as f64).sum::<f64>(), cur[0], cur[1]);
        }

        // 2) 게이티드 어텐션 — attn_layer와 동일 구조, 자체 KV(mtp_kv_*) 사용
        let attn_out = self.mtp_attn(seq, il, &cur, pos)?;

        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] wo sum={:.5} x0={:.5}", attn_out.iter().map(|&x| x as f64).sum::<f64>(), attn_out[0]);
        }
        // 3) 잔차 + post_attention_norm + FFN
        for i in 0..n_embd {
            cur[i] += attn_out[i];
        }
        let ffn_res = cur.clone();
        let post_w = self.model.f32_vec(&format!("blk.{il}.post_attention_norm.weight"))?;
        let gate_w = self.model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
        let up_w = self.model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
        let down_w = self.model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
        let normed = rms_norm(&cur, &post_w, hp.eps);
        let mut gu: [Vec<Vec<f32>>; 2] =
            [vec![vec![0.0f32; hp.n_ff]; 1], vec![vec![0.0f32; hp.n_ff]; 1]];
        crate::matmul::mm_group(&acc, &[normed], &[gate_w, up_w], &mut gu)?;
        let [mut g, u] = gu;
        for i in 0..hp.n_ff {
            g[0][i] = crate::ops::silu(g[0][i]) * u[0][i];
        }
        let mut ffn_out = vec![vec![0.0f32; n_embd]; 1];
        crate::matmul::mm_batch(&acc, &g, &down_w, &mut ffn_out)?;
        for i in 0..n_embd {
            cur[i] = ffn_out[0][i] + ffn_res[i];
        }
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] ff sum={:.5} x0={:.5}", cur.iter().map(|&x| x as f64).sum::<f64>(), cur[0]);
        }

        // 4) shared head — with_logits에만 (output.weight GEMV는 고가)
        if !with_logits {
            return Ok((Vec::new(), cur));
        }
        let sh_norm = self.model.f32_vec(&format!("blk.{il}.nextn.shared_head_norm.weight"))?;
        let head = self.model.wchk("output.weight")?;
        let h = rms_norm(&cur, &sh_norm, hp.eps);
        let mut logits = vec![0.0f32; head.n_out as usize];
        crate::matmul::mm(&acc, &h, &head, &mut logits)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            eprintln!("[c] head L0..7={:?} hnorm0..3={:?}", &logits[0..8], &h[0..4]);
        }
        Ok((logits, cur))
    }

    /// MTP draft forward — 로짓 포함 (spec 체인용).
    pub fn mtp_forward(
        &mut self,
        seq: usize,
        token: u32,
        h_in: &[f32],
        pos: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        self.mtp_step(seq, token, h_in, pos, true)
    }

    fn mtp_attn(
        &mut self,
        seq: usize,
        il: usize,
        x: &[f32],
        pos: u32,
    ) -> Result<Vec<f32>, ModelError> {
        let hp = self.model.hp.clone();
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = self.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;
        let kq_scale = hp.kq_scale();
        let acc = self.acc.clone();

        // llama.cpp: eh_proj 출력 → attn_norm → q/k/v (누락이 0-수용률 주벅)
        let attn_norm_w = self.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?;
        let xn = crate::ops::rms_norm(x, &attn_norm_w, hp.eps);
        let mut group: [Vec<Vec<f32>>; 3] = [
            vec![vec![0.0f32; wq.n_out as usize]; 1],
            vec![vec![0.0f32; wk.n_out as usize]; 1],
            vec![vec![0.0f32; wv.n_out as usize]; 1],
        ];
        {
            let xs = vec![xn];
            crate::matmul::mm_group(&acc, &xs, &[wq, wk, wv], &mut group)?;
        }
        let [qg, kk, vv] = group;

        // k norm+rope → 자체 KV 캐시 적립
        {
            let st = &mut self.seqs[seq];
            for h in 0..n_kv {
                let src = kk[0][h * hd..h * hd + hd].to_vec();
                let mut head = crate::ops::rms_norm(&src, &k_norm_w, hp.eps);
                crate::ops::rope_head(&mut head, pos, n_rot, hp.rope_base);
                let b = pos as usize * n_kv * hd + h * hd;
                st.mtp_kv_k[b..b + hd].copy_from_slice(&head);
                st.mtp_kv_v[b..b + hd].copy_from_slice(&vv[0][h * hd..h * hd + hd]);
            }
        }
        let st = &self.seqs[seq];
        let cache_k = &st.mtp_kv_k;
        let cache_v = &st.mtp_kv_v;

        let mut attn_out = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            let src = qg[0][h * 2 * hd..h * 2 * hd + hd].to_vec();
            let mut qh = crate::ops::rms_norm(&src, &q_norm_w, hp.eps);
            crate::ops::rope_head(&mut qh, pos, n_rot, hp.rope_base);
            let kvh = h / (n_head / n_kv);
            let n_past = pos as usize + 1;
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
            let ob = h * hd;
            for p in 0..n_past {
                let w = scores[p] / sum;
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
                attn_out[ob + i] *= crate::ops::sigmoid(qg[0][gb + i]);
            }
        }
        // wo 프로젝션
        let mut out = vec![vec![0.0f32; wo.n_out as usize]; 1];
        crate::matmul::mm_batch(&acc, &[attn_out], &wo, &mut out)?;
        Ok(out.into_iter().next().unwrap())
    }

    /// 스펙 디코드 1사이클 (06) — k draft 생성 → 타깃 forward 연쇄 검증.
    /// v1 단순화: 검증은 타깃 t=1 순차 forward (정확성 동일 — 06 §4.3
    /// "1차는 슬롯별 순차도 허용"). 반환: (수용 토큰들, 타깃 forward 수).
    #[allow(clippy::type_complexity)]
    pub fn spec_step(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), ModelError> {
        // GPU 검증 경로 (rawhip): draft 체인(CPU MTP층 + GPU head) → 1배치 검증.
        if self.raw_decode.is_some()
            && !self.seqs[seq].mtp_h.is_empty()
            && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
            && std::env::var_os("LLM170_SPEC_GPU").is_some()
        {
            return self.spec_step_gpu(seq, last_token, k);
        }
        let eos = 248044u32;
        // 교차 검증: 타깃 decode(토큰) → (hook이 계산한 (토큰,h) 쌍의 draft 로짓) 비교.
        // draft 체인은 직전 draft 토큰 쌍으로 순차 — target decode가 h를 갱신하는 즉시.
        let base_pos = self.seqs[seq].pos; // 슬롯 0..base_pos-1 처리됨; last_token = 위치 base_pos 토큰(미처리)
        let mut accepted: Vec<u32> = Vec::new();
        let mut total = 0usize;
        let mut cur = last_token;
        // 체인 상태: (draft 토큰, 그 pair의 h_next) — j=0은 hook 저장분 사용
        let mut chain_tok: Option<u32> = None;
        let mut chain_h: Vec<f32> = Vec::new();
        let mut chain_pos = base_pos; // 다음 mtp_forward가 쓸 슬롯
        for j in 0..=k {
            let logits = self.decode(&[seq], &[cur])?;
            total += 1;
            let t = greedy(&logits[0]);
            // draft 예측
            let d = if j == 0 {
                // raw 경로는 mtp_draft_logits를 채우지 않음(GPU MTP 헤드 argmax만 저장) — 폴백.
                if self.seqs[seq].mtp_draft_logits.is_empty() {
                    self.seqs[seq].mtp_draft_tok
                } else {
                    greedy(&self.seqs[seq].mtp_draft_logits)
                }
            } else {
                // 직전 루프에서 준비한 체인 로짓
                let (lgt, nh) = self.mtp_forward(seq, chain_tok.unwrap(), &chain_h, chain_pos)?;
                chain_h = nh;
                chain_pos += 1;
                greedy(&lgt)
            };
            accepted.push(t);
            if std::env::var_os("LLM170_SPEC_DBG").is_some() {
                eprintln!("  verify j={j} target={t} draft={d} {}", if t == d { "OK" } else { "MISS" });
            }
            if t != d || t == eos {
                break;
            }
            // 수용: 다음 비교용 체인 준비 — (d, h_next) pair는 다음 루프에서 forward
            chain_tok = Some(d);
            if chain_h.is_empty() {
                chain_h = self.seqs[seq].mtp_h_next.clone();
            }
            cur = t;
        }
        Ok((accepted, total))
    }

    /// np×spec 병합 스펙 (plans/18) — 배치 원자 의미론:
    /// verify 배치 = 시퀀스별 [carried ++ next ++ drafts]. 전 시퀀스 전체수용 시에만
    /// GDN 유지; 하나라도 부분수용이면 전체 restore + 수용 접두 carried로 재실행.
    /// 반환: [seq][accepted].
    pub fn spec_step_multi(
        &mut self,
        seqs: &[usize],
        nexts: &[u32],
        k: usize,
    ) -> Result<Vec<Vec<u32>>, ModelError> {
        // 시퀀스 청크 — verify_batch_ms 행 상한 32 (np·(k+1) > 28이면 분할)
        let per = 60usize / (1 + k);
        if seqs.len() > per.max(1) {
            let mut out: Vec<Vec<u32>> = Vec::with_capacity(seqs.len());
            for c in seqs.chunks(per.max(1)) {
                let i0 = out.len();
                let ns: Vec<u32> = {
                    let base = seqs.iter().position(|&x| x == c[0]).unwrap_or(0);
                    (0..c.len()).map(|i| nexts[base + i]).collect()
                };
                out.extend(self.spec_step_multi(c, &ns, k)?);
                let _ = i0;
            }
            return Ok(out);
        }
        let eos = 248044u32;
        let rd = self.raw_decode.clone().ok_or(ModelError::Accel("raw 없음".into()))?;
        let n_seq = seqs.len();
        let n_e = self.model.hp.n_embd;
        if self.embd_cache.is_none() {
            let t = self.model.wchk("token_embd.weight")?;
            self.embd_cache = Some((t.ty, std::sync::Arc::new(t.data.to_vec())));
        }
        let (embd_ty, embd_arc) = self.embd_cache.as_ref().unwrap().clone();

        // ── 시퀀스별 draft 체인
        let mut all_drafts: Vec<Vec<u32>> = Vec::with_capacity(n_seq);
        {
            let mut trow = vec![0.0f32; n_e];
            for si in 0..n_seq {
                let seq = seqs[si];
                let mut drafts = Vec::with_capacity(k);
                let pending = std::mem::take(&mut self.seqs[seq].mtp_pending_h);
                crate::quant::dequant_row(embd_ty, &embd_arc, nexts[si] as u64, n_e as u64, &mut trow);
                let (d0, _) = rd
                    .mtp_step_gpu(seq, &trow, &pending, self.seqs[seq].pos as usize)
                    .map_err(ModelError::Accel)?;
                self.seqs[seq].mtp_pending_h = pending;
                drafts.push(d0);
                let mut tok = d0;
                for _ in 1..k {
                    let dpos = self.seqs[seq].pos as usize + drafts.len() - 1;
                    crate::quant::dequant_row(embd_ty, &embd_arc, tok as u64, n_e as u64, &mut trow);
                    let d = rd.mtp_step_chain(seq, &trow, dpos).map_err(ModelError::Accel)?;
                    drafts.push(d);
                    tok = d;
                    if d == eos {
                        break;
                    }
                }
                all_drafts.push(drafts);
            }
        }

        // ── 배치 조립: [s0: carried+next+drafts | ...] — carried 포함 그룹 위치 = pos - carried
        let mut carried: Vec<Vec<u32>> = Vec::with_capacity(n_seq);
        for si in 0..n_seq {
            carried.push(std::mem::take(&mut self.seqs[seqs[si]].gdn_carried));
        }
        // carried 상한 — 총 행수 > 28이면 커밋 배치로 소화 (verify_batch_ms t≤32 상한).
        // 커밋은 seq 청크로 분할 (np8×k4 = 40행 → 2×20행).
        {
            let total: usize = carried
                .iter()
                .zip(all_drafts.iter())
                .map(|(c, d)| c.len() + 1 + d.len())
                .sum::<usize>();
            if total > 60 {
                let mut crows: Vec<f32> = Vec::new();
                let mut starts = Vec::new();
                let mut sposs = Vec::new();
                let mut sseqs = Vec::new();
                for si in 0..n_seq {
                    if carried[si].is_empty() {
                        continue;
                    }
                    starts.push(crows.len() / n_e);
                    sposs.push(self.seqs[seqs[si]].pos as usize - carried[si].len());
                    sseqs.push(seqs[si]);
                    for &tk in &carried[si] {
                        let mut r = vec![0.0f32; n_e];
                        crate::quant::dequant_row(embd_ty, &embd_arc, tk as u64, n_e as u64, &mut r);
                        crows.extend(r);
                    }
                }
                if !crows.is_empty() {
                    // 시퀀스 청크 분할 커밋 (행수 ≤ 28)
                    let mut ci = 0usize;
                    while ci < sseqs.len() {
                        let mut cj = ci;
                        let mut rows_n = 0usize;
                        while cj < sseqs.len() {
                            let add = starts[cj + 1..].first().copied().unwrap_or(crows.len() / n_e) - starts[cj];
                            if rows_n + add > 60 && cj > ci {
                                break;
                            }
                            rows_n += add;
                            cj += 1;
                        }
                        let r0 = starts[ci] * n_e;
                        let r1 = if cj < starts.len() { starts[cj] * n_e } else { crows.len() };
                        let sub_starts: Vec<usize> = starts[ci..cj].iter().map(|&x| x - starts[ci]).collect();
                        let mut cam: Vec<u32> = Vec::new();
                        let mut ch_all: Vec<f32> = Vec::new();
                        rd.verify_batch_ms(
                            &sseqs[ci..cj],
                            &sposs[ci..cj],
                            &sub_starts,
                            &crows[r0..r1],
                            &mut cam,
                            &mut ch_all,
                        )
                        .map_err(ModelError::Accel)?;
                        ci = cj;
                    }
                    for si in 0..n_seq {
                        self.seqs[seqs[si]].gdn_carried = Vec::new();
                    }
                    carried = vec![Vec::new(); n_seq];
                }
            }
        }
        let mut rows: Vec<f32> = Vec::new();
        let mut group_starts = Vec::with_capacity(n_seq);
        let mut group_pos: Vec<usize> = Vec::with_capacity(n_seq); // 그룹 첫 행 위치
        for si in 0..n_seq {
            group_starts.push(rows.len() / n_e);
            let pos0 = self.seqs[seqs[si]].pos as usize - carried[si].len();
            group_pos.push(pos0);
            for &tk in carried[si].iter().chain(std::iter::once(&nexts[si])).chain(all_drafts[si].iter()) {
                let mut r = vec![0.0f32; n_e];
                crate::quant::dequant_row(embd_ty, &embd_arc, tk as u64, n_e as u64, &mut r);
                rows.extend(r);
            }
        }
        rd.gdn_snapshot().map_err(ModelError::Accel)?;
        let mut am: Vec<u32> = Vec::new();
        let mut h_all: Vec<f32> = Vec::new();
        if std::env::var_os("LLM170_MS_SEQ").is_some() {
            // 폴백/바이섹트 (plans/28): 그룹별 단일-시퀀스 verify — 시퀀스 상태가
            // 분리라 병합과 의미동치 (커널 버그 회피용; 성능 하락).
            for si in 0..n_seq {
                let g0 = group_starts[si];
                let g1 = if si + 1 < n_seq { group_starts[si + 1] } else { rows.len() / n_e };
                let mut am2 = Vec::new();
                let mut h2 = Vec::new();
                rd.raw_verify(seqs[si], group_pos[si], &rows[g0 * n_e..g1 * n_e],
                              &mut am2, &mut h2)
                    .map_err(ModelError::Accel)?;
                am.extend_from_slice(&am2);
                h_all.extend_from_slice(&h2);
            }
        } else {
            rd.verify_batch_ms(seqs, &group_pos, &group_starts, &rows, &mut am, &mut h_all)
                .map_err(ModelError::Accel)?;
        }
        if std::env::var_os("LLM170_SPEC_DBG").is_some() {
            eprintln!("  [msV] groups={group_starts:?}");
            eprintln!("  [msV] am={am:?}");
            eprintln!("  [msV] drafts={all_drafts:?}");
        }
        if std::env::var_os("LLM170_MS_AB").is_some() {
            // A/B: 스냅샷으로 상태 복원 후 각 그룹을 단일-verify로 재계산·비교
            // (all_full이면 재검증이 상태를 동일하게 재진행 — 본류 불변.
            //  부분수용이면 본류에서 어차피 restore.)
            rd.gdn_restore().map_err(ModelError::Accel)?;
            for si in 0..n_seq {
                let g0 = group_starts[si];
                let g1 = if si + 1 < n_seq { group_starts[si + 1] } else { rows.len() / n_e };
                let sub: Vec<f32> = rows[g0 * n_e..g1 * n_e].to_vec();
                let mut am2: Vec<u32> = Vec::new();
                let mut h2: Vec<f32> = Vec::new();
                rd.raw_verify(seqs[si], group_pos[si], &sub, &mut am2, &mut h2)
                    .map_err(ModelError::Accel)?;
                eprintln!("[AB] seq={} merged={:?} single={:?}", seqs[si],
                    &am[g0..g1], &am2);
            }
        }
        // ── 시퀀스별 수용 판정 (신규 세그먼트: next+drafts)
        let mut all_full = true;
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(n_seq);
        let mut new_kept: Vec<usize> = Vec::with_capacity(n_seq); // next+matched drafts 수
        for si in 0..n_seq {
            let g0 = group_starts[si];
            let next_off = carried[si].len(); // 신규 세그먼트 내 next 위치
            let drafts = &all_drafts[si];
            let mut accepted: Vec<u32> = Vec::new();
            for j in 0..drafts.len() {
                accepted.push(am[g0 + next_off + j]);
                if am[g0 + next_off + j] != drafts[j] || am[g0 + next_off + j] == eos {
                    break;
                }
            }
            let full = accepted.len() == drafts.len()
                && drafts.iter().zip(accepted.iter()).all(|(d, a)| d == a);
            if full {
                let g1 = if si + 1 < n_seq { group_starts[si + 1] } else { am.len() };
                accepted.push(am[g1 - 1]); // 보너스 (마지막 행)
            } else {
                all_full = false;
            }
            // kept new rows = next + matched drafts (보너스 제외)
            let matched = if full { drafts.len() } else { accepted.len().saturating_sub(1) };
            new_kept.push(1 + matched);
            results.push(accepted);
        }

        // ── 상태 갱신
        if all_full {
            for si in 0..n_seq {
                self.seqs[seqs[si]].gdn_carried = Vec::new();
            }
        } else {
            rd.gdn_restore().map_err(ModelError::Accel)?;
            for si in 0..n_seq {
                let seq = seqs[si];
                let mut c = carried[si].clone();
                // 유지 신규 행 토큰: next + matched drafts
                let matched = new_kept[si] - 1;
                c.push(nexts[si]);
                for j in 0..matched {
                    c.push(all_drafts[si][j]);
                }
                self.seqs[seq].gdn_carried = c;
            }
        }
        for si in 0..n_seq {
            self.seqs[seqs[si]].pos += new_kept[si] as u32;
            // MTP 상태 진행 (shift 페어링): 신규 kept 행 — carried 재적립 멱등
            let g0 = group_starts[si];
            let kept = carried[si].len() + new_kept[si];
            let base = group_pos[si];
            let mut prev_h = self.seqs[seqs[si]].mtp_pending_h.clone();
            let mut trow = vec![0.0f32; n_e];
            for r in carried[si].len()..kept {
                // 행 r 토큰 = carried면 carried[r], else next/drafts
                let tok = if r < carried[si].len() {
                    carried[si][r]
                } else if r == carried[si].len() {
                    nexts[si]
                } else {
                    all_drafts[si][r - carried[si].len() - 1]
                };
                let h_prev = prev_h.clone();
                let row = &rows[(g0 + r) * n_e..(g0 + r + 1) * n_e];
                let _ = tok;
                // draft0 행(next)은 이미 mtp_step_gpu가 처리 — r == carried.len() 스킵
                if r == carried[si].len() {
                    prev_h.copy_from_slice(&h_all[(g0 + r) * n_e..(g0 + r + 1) * n_e]);
                    continue;
                }
                rd.mtp_step_adv(seqs[si], row, &h_prev, base + r)
                    .map_err(ModelError::Accel)?;
                prev_h.copy_from_slice(&h_all[(g0 + r) * n_e..(g0 + r + 1) * n_e]);
            }
            self.seqs[seqs[si]].mtp_pending_h = prev_h;
        }
        Ok(results)
    }


    pub fn flush_carried(&mut self, seq: usize) -> Result<(), ModelError> {
        let carried = std::mem::take(&mut self.seqs[seq].gdn_carried);
        if carried.is_empty() {
            return Ok(());
        }
        let rd = match self.raw_decode.as_ref() {
            Some(r) => r.clone(),
            None => return Ok(()),
        };
        let n = self.model.hp.n_embd;
        if self.embd_cache.is_none() {
            let t = self.model.wchk("token_embd.weight")?;
            self.embd_cache = Some((t.ty, std::sync::Arc::new(t.data.to_vec())));
        }
        let (ty, data) = self.embd_cache.as_ref().unwrap().clone();
        let mut rows = Vec::with_capacity(carried.len() * n);
        for &tk in &carried {
            let mut r = vec![0.0f32; n];
            crate::quant::dequant_row(ty, &data, tk as u64, n as u64, &mut r);
            rows.extend(r);
        }
        let pos0 = self.seqs[seq].pos as usize - carried.len();
        let mut am = Vec::new();
        let mut h = Vec::new();
        rd.raw_verify(seq, pos0, &rows, &mut am, &mut h).map_err(ModelError::Accel)?;
        Ok(())
    }

    fn spec_step_gpu(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), ModelError> {
        let eos = 248044u32;
        let sp_t0 = std::time::Instant::now();
        let rd = self.raw_decode.clone().ok_or(ModelError::Accel("raw 없음".into()))?;
        let base_pos = self.seqs[seq].pos; // 슬롯 0..base_pos-1 처리됨
        let t_draft0 = std::time::Instant::now();
        // ── draft: step-0 = (last_token, pending_h) 시프트 페어링; j≥1 = 체인 자가 h
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let n_e = self.model.hp.n_embd;
        {
            if self.embd_cache.is_none() {
                let t = self.model.wchk("token_embd.weight")?;
                self.embd_cache = Some((t.ty, std::sync::Arc::new(t.data.to_vec())));
            }
            let (embd_ty, embd_arc) = self.embd_cache.as_ref().unwrap().clone();
            let embd_data: &Vec<u8> = &embd_arc;
            let mut trow = vec![0.0f32; n_e];
            crate::quant::dequant_row(embd_ty, &embd_data, last_token as u64, n_e as u64, &mut trow);
            let pending = std::mem::take(&mut self.seqs[seq].mtp_pending_h);
            let t_d0 = std::time::Instant::now();
            let (d0, _) = rd
                .mtp_step_gpu(seq, &trow, &pending, base_pos as usize)
                .map_err(ModelError::Accel)?;
            if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
                eprintln!("[d0] draft0={:.1}ms", t_d0.elapsed().as_secs_f64() * 1e3);
            }
            // pending은 다시 저장 (verify 후 마지막 행 hidden으로 갱신)
            self.seqs[seq].mtp_pending_h = pending;
            drafts.push(d0);
            let mut tok = d0;
            for j in 1..k {
                let dpos = (base_pos + drafts.len() as u32 - 1) as usize;
                let tc = std::time::Instant::now();
                let mut trow = vec![0.0f32; n_e];
                crate::quant::dequant_row(embd_ty, &embd_data, tok as u64, n_e as u64, &mut trow);
                let td = std::time::Instant::now();
                let d = rd.mtp_step_chain(seq, &trow, dpos).map_err(ModelError::Accel)?;
                if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
                    eprintln!("[ch] j={j} deq={:.2}ms step={:.2}ms", td.duration_since(tc).as_secs_f64()*1e3, td.elapsed().as_secs_f64()*1e3);
                }
                drafts.push(d);
                tok = d;
                if d == eos {
                    break;
                }
            }
        }
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[sp] draft chain={:.1}ms k={}", t_draft0.elapsed().as_secs_f64() * 1e3, drafts.len());
        }
        let t_v0 = std::time::Instant::now();
        // ── verify: [carried..., last_token, d0, d1, ...] 1배치 — 행별 argmax = 다음 토큰 정답
        // carried = 직전 부분수용에서 GDN이 미확정인 행 — 같은 토큰·같은 위치 재실행
        // (결정론적 커널 → 동일 결과, KV는 동일값 재기입). 재실행 배치를 대체한다.
        let mut carried: Vec<u32> = std::mem::take(&mut self.seqs[seq].gdn_carried);
        // carried 상한 — 초과 시 GDN 커밋 배치(헤드 무의미하지만 간단)로 소화 후 청소.
        // 전체수용이 드문 높은 k에서 배치 무한 증식 방지 (롤백 재실행의 분할 상환).
        if carried.len() + 1 + k > 16 {
            let n_c = self.model.hp.n_embd;
            let (embd_ty_c, embd_arc_c) = self.embd_cache.as_ref().unwrap().clone();
            let mut crows: Vec<f32> = Vec::with_capacity(carried.len() * n_c);
            for &tk in carried.iter() {
                let mut r = vec![0.0f32; n_c];
                crate::quant::dequant_row(embd_ty_c, &embd_arc_c, tk as u64, n_c as u64, &mut r);
                crows.extend(r);
            }
            let mut cam: Vec<u32> = Vec::new();
            let mut ch_all: Vec<f32> = Vec::new();
            rd.raw_verify(seq, (base_pos - carried.len() as u32) as usize, &crows, &mut cam, &mut ch_all)
                .map_err(ModelError::Accel)?;
            // 커밋 후 mtp 진행도 보강 (멱등 — 신규 행만)
            for i in 1..carried.len() {
                let h_prev = ch_all[(i - 1) * n_c..i * n_c].to_vec();
                rd.mtp_step_adv(seq, &crows[i * n_c..(i + 1) * n_c], &h_prev, (base_pos as usize) - carried.len() + i)
                    .map_err(ModelError::Accel)?;
            }
            carried = Vec::new();
        }
        let carried_n = carried.len();
        let pos0 = base_pos - carried_n as u32;
        let t = 1 + drafts.len() + carried_n;
        let n = self.model.hp.n_embd;
        if self.embd_cache.is_none() {
            let tw = self.model.wchk("token_embd.weight")?;
            self.embd_cache = Some((tw.ty, std::sync::Arc::new(tw.data.to_vec())));
        }
        let (embd_ty, embd_arc) = self.embd_cache.as_ref().unwrap().clone();
        let embd_data: &Vec<u8> = &embd_arc;
        let mut rows: Vec<f32> = Vec::with_capacity(t * n);
        let mut row_toks: Vec<u32> = Vec::with_capacity(t);
        for &tk in carried
            .iter()
            .chain(std::iter::once(&last_token))
            .chain(drafts.iter())
        {
            let mut r = vec![0.0f32; n];
            crate::quant::dequant_row(embd_ty, embd_data, tk as u64, n as u64, &mut r);
            rows.extend(r);
            row_toks.push(tk);
        }
        // 부분수용 대비 GDN/conv 스냅샷 (KV는 위치 색인이라 자가치유)
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[vv] rows+dequant={:.1}ms", t_v0.elapsed().as_secs_f64() * 1e3);
        }
        rd.gdn_snapshot().map_err(ModelError::Accel)?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[vv] snapshot={:.1}ms", t_v0.elapsed().as_secs_f64() * 1e3);
        }
        let mut am: Vec<u32> = Vec::new();
        let mut h_all: Vec<f32> = Vec::new();
        rd.raw_verify(seq, pos0 as usize, &rows, &mut am, &mut h_all)
            .map_err(ModelError::Accel)?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[vv] raw_verify={:.1}ms", t_v0.elapsed().as_secs_f64() * 1e3);
        }
        // 수용 (신규 세그먼트만): am[carried_n + j] vs drafts[j]
        let mut accepted: Vec<u32> = Vec::new();
        for j in 0..drafts.len() {
            let i = carried_n + j;
            accepted.push(am[i]);
            if am[i] != drafts[j] || am[i] == eos {
                break;
            }
        }
        if std::env::var_os("LLM170_SPEC_DBG").is_some() {
            eprintln!(
                "  gpu-verify pos={base_pos} carried={carried_n} drafts={drafts:?} am={am:?} acc_n={}",
                if accepted.len() == drafts.len()
                    && drafts.iter().zip(accepted.iter()).all(|(d, a)| d == a)
                { accepted.len() + 1 } else { accepted.len().max(1) }
            );
        }
        let all_acc = accepted.len() == drafts.len()
            && drafts.iter().zip(accepted.iter()).all(|(d, a)| d == a);
        let kept_new = if all_acc {
            accepted.push(am[t - 1]);
            1 + drafts.len() // last + 전부
        } else {
            accepted.len() // last + 매칭 draft 수
        };
        if all_acc {
            // 전부 수용 — 상태 그대로 유효, carried 소멸.
            self.seqs[seq].gdn_carried = Vec::new();
        } else {
            // 부분수용 — GDN 복원 후 유지 행을 carried로 (다음 배치에서 재실행).
            rd.gdn_restore().map_err(ModelError::Accel)?;
            self.seqs[seq].gdn_carried = row_toks[..carried_n + kept_new].to_vec();
        }
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[sp] verify+decide={:.1}ms", t_v0.elapsed().as_secs_f64() * 1e3);
        }
        let t_adv0 = std::time::Instant::now();
        // ── MTP 상태 진행 (시프트 페어링): 행 0은 draft step-0이 이미 처리.
        // carried 구간은 직전 스텝이 이미 적립(멱등) — 신규 행부터만.
        {
            let kept = carried_n + kept_new;
            for i in (carried_n.max(1))..kept {
                let h_prev = h_all[(i - 1) * n..i * n].to_vec();
                rd.mtp_step_adv(seq, &rows[i * n..(i + 1) * n], &h_prev, pos0 as usize + i)
                    .map_err(ModelError::Accel)?;
            }
            // pending = 마지막 유지 행의 trunk hidden (다음 draft step-0 페어링)
            self.seqs[seq].mtp_pending_h = h_all[(kept - 1) * n..kept * n].to_vec();
        }
        // 시퀀스 pos 동기 — 유지 신규 행 수만 반영
        self.seqs[seq].pos = base_pos + (kept_new as u32);
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[sp] advance={:.1}ms | step total={:.1}ms acc={}", t_adv0.elapsed().as_secs_f64() * 1e3, sp_t0.elapsed().as_secs_f64() * 1e3, accepted.len());
        }
        let n = accepted.len().max(1);
        Ok((accepted, n))
    }
}
