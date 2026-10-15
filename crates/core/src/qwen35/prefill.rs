//! Engine 프리필 — 행 조립(스레드) + 1024 청크 forward.

use super::*;

impl Engine {
    /// 시퀀스 prefill: 전체 토큰 적립 + 마지막 logits.
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let n = self.model.hp.n_embd;
        // 제자리 borrow — token_embd 전체 to_vec(636MB) 복사를 회피한다.
        let cache: Vec<Vec<f32>> = {
            let embd = self.model.wchk("token_embd.weight")?;
            // 행 단위 병렬 디양자화 (단일 스레드면 pp512에서 ~20ms가 붙는다).
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
                            crate::quant::dequant_row(
                                embd.ty, embd.data, tok as u64, n as u64, row,
                            );
                        }
                    });
                }
            });
            rows
        };
        // 1024토큰 청크 — 청킹은 수치 불변(GDN chunked·attention 캐시 순차 적립).
        let chunk: usize = 1024;
        let mut last = None;
        for (ch_t, ch_r) in tokens.chunks(chunk).zip(cache.chunks(chunk)) {
            let logits = self.forward_emb(&[seq], &[ch_t.to_vec()], Some(ch_r))?;
            self.seqs[seq].pos += ch_t.len() as u32;
            last = Some(logits.into_iter().next().unwrap());
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

    /// 체크포인트 저장 — 전 시퀀스 (pos + 상태 전부). 벡터 순서:
    /// kv_k[0..F)·kv_v[0..F)·gdn_s[0..R)·conv[0..R). 코덱 계약 = `crate::ckpt` 문서.
    pub fn ckpt_save(&self) -> Result<Vec<u8>, ModelError> {
        let no = || ModelError::W4a16("ckpt: 빈 엔진".into());
        let s0 = self.seqs.first().ok_or_else(no)?;
        let row = (self.model.hp.n_kv * self.model.hp.head_dim).max(1);
        let ctx = (s0.kv_k.first().map(|k| k.len() / row).unwrap_or(0) as u32).max(1);
        let mut seqs = Vec::with_capacity(self.seqs.len());
        for s in &self.seqs {
            let mut vecs: Vec<&[f32]> =
                Vec::with_capacity(s.kv_k.len() * 2 + s.gdn_s.len() + s.conv.len());
            for k in &s.kv_k {
                vecs.push(k.as_slice());
            }
            for v in &s.kv_v {
                vecs.push(v.as_slice());
            }
            for g in &s.gdn_s {
                vecs.push(g.as_slice());
            }
            for c in &s.conv {
                vecs.push(c.as_slice());
            }
            seqs.push((s.pos, vecs));
        }
        Ok(crate::ckpt::encode(crate::ckpt::ARCH_QWEN, ctx, seqs))
    }

    /// 체크포인트 복원 — 매직·아치·시퀀스 수·벡터 수·길이 전부 대조(불일치 거부).
    pub fn ckpt_load(&mut self, blob: &[u8]) -> Result<(), ModelError> {
        let bad = |m: String| ModelError::W4a16(format!("ckpt: {m}"));
        let (arch, _ctx, seqs) = crate::ckpt::decode(blob).map_err(bad)?;
        if arch != crate::ckpt::ARCH_QWEN {
            return Err(bad(format!("아치 {arch} ≠ qwen35")));
        }
        if seqs.len() != self.seqs.len() {
            return Err(bad(format!(
                "시퀀스 수 {} ≠ {}",
                seqs.len(),
                self.seqs.len()
            )));
        }
        let n_full = (0..self.model.hp.n_layer)
            .filter(|&il| !self.model.is_recr(il))
            .count();
        let n_recr = self.model.hp.n_layer - n_full;
        for (i, (pos, vecs)) in seqs.into_iter().enumerate() {
            let live = &mut self.seqs[i];
            if vecs.len() != 2 * n_full + 2 * n_recr {
                return Err(bad(format!("벡터 수 {} 비정상", vecs.len())));
            }
            let mut it = vecs.into_iter();
            for dst in live
                .kv_k
                .iter_mut()
                .chain(live.kv_v.iter_mut())
                .chain(live.gdn_s.iter_mut())
                .chain(live.conv.iter_mut())
            {
                let v = it.next().ok_or_else(|| bad("벡터 부족".into()))?;
                if v.len() != dst.len() {
                    return Err(bad(format!("길이 {} ≠ {}", v.len(), dst.len())));
                }
                dst.copy_from_slice(&v);
            }
            live.pos = pos;
        }
        Ok(())
    }
}
