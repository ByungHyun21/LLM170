//! Engine 프리필 — 청킹·비전 스플라이스·행 단주 프리필 (mod.rs에서 분리, plans/35 P5).

use super::*;

impl Engine {
    pub fn prefill_vision(
        &mut self,
        seq: usize,
        tokens: &[u32],
        marker: u32,
        vis: &[Vec<f32>],
    ) -> Result<Vec<f32>, ModelError> {
        // marker → vis 행 치환 후 내부 prefill 재사용: 임시 토큰열 + 스플라이스 맵은
        // 복잡하므로 여기서 캐시 행을 직접 조립한다 (prefill과 동일 절차).
        let n = self.model.hp.n_embd;
        let embd = self.model.wchk("token_embd.weight")?;
        let mut cache: Vec<Vec<f32>> = Vec::with_capacity(tokens.len() + vis.len());
        let mut tok_seq: Vec<u32> = Vec::with_capacity(tokens.len() + vis.len());
        let mut spliced = 0usize;
        for &t in tokens {
            if t == marker && !vis.is_empty() {
                for row in vis {
                    cache.push(row.clone());
                    tok_seq.push(marker);
                }
                spliced += 1;
            } else {
                let mut r = vec![0.0f32; n];
                crate::quant::dequant_row(embd.ty, embd.data, t as u64, n as u64, &mut r);
                cache.push(r);
                tok_seq.push(t);
            }
        }
        if spliced != 1 {
            return Err(ModelError::Accel(format!(
                "prefill_vision: marker {marker} {spliced}회 (1회 기대)"
            )));
        }
        self.prefill_rows(seq, &tok_seq, &cache)
    }

    /// 시퀀스 prefill: 전체 토큰 적립 + 마지막 logits.

    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let n = self.model.hp.n_embd as usize;
        // 제자리 borrow. 과거(훅 루프 &mut self 공존 회피)에는 token_embd 전체
        // to_vec(636MB)을 매 호출마다 복사해 pp에 고정 ~170ms를 더했음.
        // 훅은 prefill_rows(embd 스코프 밖)에 있으므로 borrow 충돌 없음.
        let cache: Vec<Vec<f32>> = {
            let embd = self.model.wchk("token_embd.weight")?;
            // 512행 × 5120값 q4_K 디양자화를 단일 스레드로 돌리면 pp512에서
            // ~20ms가 GPU 패스 밖(호스트)에 붙는다 — 행 단위로 병렬화.
            let mut rows: Vec<Vec<f32>> = tokens.iter().map(|_| vec![0.0f32; n]).collect();
            let nt = crate::matmul::n_threads().max(1).min(rows.len().max(1));
            let per = rows.len().div_ceil(nt).max(1);
            std::thread::scope(|s| {
                for (lo, ch) in rows.chunks_mut(per).enumerate() {
                    let t0 = lo * per;
                    let toks = &tokens[t0..t0 + ch.len()];
                    let embd = &embd;
                    s.spawn(move || {
                        for (row, &tok) in ch.iter_mut().zip(toks.iter()) {
                            crate::quant::dequant_row(embd.ty, embd.data, tok as u64, n as u64, row);
                        }
                    });
                }
            });
            rows
        };
        let _ = n;
        self.prefill_rows(seq, tokens, &cache)
    }

    /// 행이 미리 조립된 prefill (비전 스플라이스 재사용).
    pub fn prefill_rows(
        &mut self,
        seq: usize,
        tokens: &[u32],
        cache: &[Vec<f32>],
    ) -> Result<Vec<f32>, ModelError> {
        // 1024토큰 청크 — qwen4exp와 동일 근거: 단일 초대형 forward는 GPU
        // 스크래치·상태 크기를 폭주시킨다 (qwen4exp GPF 실측, 2026-08-31).
        // 청킹은 수치 불변 (GDN chunked·attention 캐시는 순차 적립).
        let chunk: usize = std::env::var("LLM170_Q35_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
            .clamp(16, 1024);
        let mut last = None;
        // 원시 HIP 활성 시 프리필도 t=1 raw 스텝으로 — 상태 동기화 불필요
        // (KV/GDN/conv 링이 raw 디코더에 직접 적립).
        if std::env::var("LLM170_T1_PREFILL").is_ok()
            || (self.raw_decode.is_some()
                && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true))
        {
            let use_batch = std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
                && std::env::var_os("LLM170_T1_PREFILL").is_none()
                && (tokens.len() > 1 || std::env::var_os("LLM170_FORCE_BATCH").is_some());
            if use_batch {
                let rd = self.raw_decode.clone().unwrap();
                let n = self.model.hp.n_embd as usize;
                let mut pos = self.seqs[seq].pos as usize;
                // 청크 128은 128-행 타일(j128/v4 CO) 로드 시에만 유효
                // z-그리드 사분면 CO: t>128 프리필 상각 (2026-09-05, +1.5%,
                // 장문600 게이트 chunk128과 비트동일 검증)
                let ch_sz = std::env::var("LLM170_CHUNK").ok().and_then(|v| v.parse().ok())
                    .unwrap_or(if rd.tile_big_chunk() && std::env::var_os("LLM170_EXACT").is_none() { 512 } else { 64 });
                let n_chunks = cache.len().div_ceil(ch_sz).max(1);
                for (ci, ch) in cache.chunks(ch_sz).enumerate() {
                    let flat: Vec<f32> = ch.iter().flatten().copied().collect();
                    let logits = if !self.seqs[seq].mtp_h.is_empty() && self.mtp_wanted {
                        let n_e = self.model.hp.n_embd;
                        // 임베딩 선반입: 사이드 스트림 async h2d를 메인 프리필과 중첩
                        // (청크당 10.5MB 블로킹 업로드 제거).
                        let mut tok_flat: Vec<f32> = Vec::with_capacity(ch.len() * n_e);
                        for row in ch.iter() { tok_flat.extend_from_slice(row); }
                        rd.mtp_upload_tok_emb(&tok_flat).map_err(ModelError::Accel)?;
                        // MTP KV 적립: 마지막 행 hidden(carry)만 회수
                        let (lg, h_last) = rd
                            .raw_prefill_h(seq, pos, &flat)
                            .map_err(ModelError::Accel)?;
                        // 배치 MTP 프리필: blk.64를 청크 전체(t행) 한 번에 — t=1 스텝
                        // ×토큰수 대체(헤드는 마지막 행만). tok/h 시프트 페어링은
                        // llama.cpp와 동일: MTP(tok_p, h_{p-1}), h_{-1}=pending.

                        // h_shift는 GPU에서 조립(디바이스 행 시프트) — carry만 호스트에서 전달.
                        let mut carry: Vec<f32> = Vec::with_capacity(n_e);
                        if self.seqs[seq].mtp_pending_h.len() == n_e {
                            carry.extend_from_slice(&self.seqs[seq].mtp_pending_h);
                        } else {
                            carry.resize(n_e, 0.0);
                        }
                        // 헤드는 프롬프트 종료 청크에서만 (초안은 그때만 쓰인다).
                        let with_head = ci + 1 == n_chunks;
                        let draft = rd
                            .mtp_prefill_batch(seq, &tok_flat, &carry, ch.len(), pos, with_head)
                            .map_err(ModelError::Accel)?;
                        {
                            let st = &mut self.seqs[seq];
                            if with_head {
                                st.mtp_draft_tok = draft;
                            }
                            st.mtp_pending_h = h_last;
                        }
                        lg
                    } else {
                        rd.raw_prefill(seq, pos, &flat).map_err(ModelError::Accel)?
                    };
                    if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                        let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        eprintln!("logits(batch): max={m:.4} argmax={}", greedy(&logits));
                    }
                    pos += ch.len();
                    last = Some(logits);
                }
                self.seqs[seq].pos = pos as u32;
                return Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]));
            }
            for (ti, &t) in tokens.iter().enumerate() {
                let _ = ti;
                let logits = self.decode(&[seq], &[t])?;
                last = Some(logits.into_iter().next().unwrap());
            }
            return Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]));
        }
        for (ch_t, ch_r) in tokens.chunks(chunk).zip(cache.chunks(chunk)) {
            let logits = self.forward_emb(&[seq], &[ch_t.to_vec()], Some(ch_r))?;
            self.seqs[seq].pos += ch_t.len() as u32;
            last = Some(logits.into_iter().next().unwrap());
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

}
