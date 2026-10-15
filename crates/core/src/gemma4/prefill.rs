//! gemma4 프리필 — 임베딩 행 조립(병렬) + 1024 청크 forward.

use super::*;

impl Engine {
    /// 시퀀스 prefill: 전체 토큰 적립 + 마지막 위치 logits.
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        // 임베딩 행 병렬 디코드(√h 베이크 포함) — Model(RefCell) 비경유:
        // 원시 테이블 뷰 + 스케일만 캡처한다(스레드 안전).
        let (buf, cols) = self.model.embed_table()?;
        let scale = self.model.embed_scale();
        let mut rows: Vec<Vec<f32>> = tokens.iter().map(|_| Vec::new()).collect();
        let nt = crate::matmul::n_threads().max(1).min(rows.len().max(1));
        let per = rows.len().div_ceil(nt).max(1);
        std::thread::scope(|s| {
            for (lo, ch) in rows.chunks_mut(per).enumerate() {
                let t0 = lo * per;
                let toks = &tokens[t0..t0 + ch.len()];
                s.spawn(move || {
                    let mut v = Vec::with_capacity(cols);
                    for (row, &tok) in ch.iter_mut().zip(toks.iter()) {
                        bake_embed_row(buf, cols, tok, scale, &mut v);
                        *row = v.clone();
                    }
                });
            }
        });
        // 1024토큰 청크 — 청킹은 수치 불변(캐시 순차 적립).
        let chunk: usize = 1024;
        let mut last = None;
        for (ch_t, ch_r) in tokens.chunks(chunk).zip(rows.chunks(chunk)) {
            let logits = self.forward_emb(&[seq], &[ch_t.to_vec()], Some(ch_r))?;
            self.seqs[seq].pos += ch_t.len() as u32;
            last = Some(logits.into_iter().next().expect("행 1개"));
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

    /// 체크포인트 저장 — 전 시퀀스 (pos + 전층 KV f32). 벡터 순서:
    /// kv_k[0..L) 뒤 kv_v[0..L). 코덱 계약 = `crate::ckpt` 문서.
    pub fn ckpt_save(&self) -> Result<Vec<u8>, ModelError> {
        let no = || ModelError::W4a16("ckpt: 빈 엔진".into());
        let s0 = self.seqs.first().ok_or_else(no)?;
        let row = (self.model.hp.n_kv[0] * self.model.hp.head_dim[0]).max(1);
        let ctx = (s0.kv_k.first().map(|k| k.len() / row).unwrap_or(0) as u32).max(1);
        let mut seqs = Vec::with_capacity(self.seqs.len());
        for s in &self.seqs {
            let mut vecs: Vec<&[f32]> = Vec::with_capacity(s.kv_k.len() * 2);
            for k in &s.kv_k {
                vecs.push(k.as_slice());
            }
            for v in &s.kv_v {
                vecs.push(v.as_slice());
            }
            seqs.push((s.pos, vecs));
        }
        Ok(crate::ckpt::encode(crate::ckpt::ARCH_GEMMA, ctx, seqs))
    }

    /// 체크포인트 복원 — 매직·아치·시퀀스 수·벡터 수·길이 전부 대조(불일치 거부).
    pub fn ckpt_load(&mut self, blob: &[u8]) -> Result<(), ModelError> {
        let bad = |m: String| ModelError::W4a16(format!("ckpt: {m}"));
        let (arch, _ctx, seqs) = crate::ckpt::decode(blob).map_err(bad)?;
        if arch != crate::ckpt::ARCH_GEMMA {
            return Err(bad(format!("아치 {arch} ≠ gemma4")));
        }
        if seqs.len() != self.seqs.len() {
            return Err(bad(format!(
                "시퀀스 수 {} ≠ {}",
                seqs.len(),
                self.seqs.len()
            )));
        }
        for (i, (pos, vecs)) in seqs.into_iter().enumerate() {
            let live = &mut self.seqs[i];
            let l = live.kv_k.len();
            if vecs.len() != 2 * l {
                return Err(bad(format!("벡터 수 {} ≠ {}", vecs.len(), 2 * l)));
            }
            for (j, v) in vecs.into_iter().enumerate() {
                let dst = if j < l {
                    &mut live.kv_k[j]
                } else {
                    &mut live.kv_v[j - l]
                };
                if v.len() != dst.len() {
                    return Err(bad(format!("층 {j} 길이 {} ≠ {}", v.len(), dst.len())));
                }
                dst.copy_from_slice(&v);
            }
            live.pos = pos;
        }
        Ok(())
    }
}
