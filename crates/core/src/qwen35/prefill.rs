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
}
