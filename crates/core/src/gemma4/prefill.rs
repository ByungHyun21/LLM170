
use super::*;

impl Engine {
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
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
        let chunk: usize = 1024;
        let mut last = None;
        for (ch_t, ch_r) in tokens.chunks(chunk).zip(rows.chunks(chunk)) {
            let logits = self.forward_emb(&[seq], &[ch_t.to_vec()], Some(ch_r))?;
            self.seqs[seq].pos += ch_t.len() as u32;
            last = Some(logits.into_iter().next().expect("행 1개"));
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

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
