
use super::*;

impl Engine {
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let n = self.model.hp.n_embd;
        let cache: Vec<Vec<f32>> = {
            let embd = self.model.wchk("token_embd.weight")?;
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
        let chunk: usize = 1024;
        let mut last = None;
        for (ch_t, ch_r) in tokens.chunks(chunk).zip(cache.chunks(chunk)) {
            let logits = self.forward_emb(&[seq], &[ch_t.to_vec()], Some(ch_r))?;
            self.seqs[seq].pos += ch_t.len() as u32;
            last = Some(logits.into_iter().next().unwrap());
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

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
