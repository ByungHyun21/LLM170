use super::QwenEngine;
use super::mtp::upload_mtp;
use super::prefill_tmax;
use super::upload::upload_model;
use llm170_backend_gpu::W4a16Dec;
use std::path::Path;

enum PrefillOut {
    Logits(Vec<f32>),
    Token(u32),
}

impl QwenEngine {
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
        kv_vram_cap: Option<u64>,
    ) -> Result<Self, String> {
        Self::load_dev(
            dir,
            n_slots,
            ctx,
            kv_budget,
            kv_reserve,
            kv_vram_cap,
            0,
            None,
        )
    }

    pub fn load_dev(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
        kv_vram_cap: Option<u64>,
        device: i32,
        window: Option<(usize, usize)>,
    ) -> Result<Self, String> {
        let model = llm170_core::qwen35::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new_dev(n_slots, hp.n_embd, hp.n_layer, device)?;
        if let Some((ws, we)) = window {
            dec.set_layer_window(ws, we)?;
        }
        dec.set_kv_budget(kv_budget);
        dec.set_kv_vram_cap(kv_vram_cap);
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        match Self::fill(&mut dec, &model, ctx, kv_reserve.unwrap_or(0), window) {
            Ok((head_gpu, mtp)) => Ok(QwenEngine {
                model,
                dec,
                n_slots: n_slots.max(1),
                head_gpu,
                softcap: llm170_core::qwen35::bind::final_logit_softcap(dir),
                mtp,
                tuning: super::Tuning::default(),
            }),
            Err(e) => {
                dec.free_all_device();
                Err(e)
            }
        }
    }

    pub fn apply_tuning(&mut self, t: super::Tuning) -> Result<(), String> {
        self.tuning = t;
        self.dec.set_lru_tuning(t.lru_cap, t.lru_frac)
    }

    pub fn ngram_knobs(&self) -> (usize, usize) {
        (
            self.tuning.ngram_n.unwrap_or(0),
            self.tuning.ngram_m.unwrap_or(48),
        )
    }

    fn fill(
        dec: &mut W4a16Dec,
        model: &llm170_core::qwen35::Model,
        ctx: usize,
        kv_reserve: u64,
        window: Option<(usize, usize)>,
    ) -> Result<(bool, bool), String> {
        upload_model(dec, model, ctx, kv_reserve, window)?;
        let mut ld = super::upload::DirectLoad::new(model.store());
        let mut head_gpu = false;
        if let Some(w) = model.w("output.weight")
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            let (tn, _) = model
                .w_raw_names("output.weight")
                .ok_or_else(|| "output.weight: 원시 텐서 이름 부재".to_string())?;
            let len = ld.tensor_len(&tn)? as usize;
            let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
            dec.upload_head_stream(w.n_out as usize, w.n_in as usize, len, &mut f)?;
            head_gpu = true;
        }
        if let Some(w) = model.w("token_embd.weight")
            && matches!(
                w.ty,
                llm170_core::wtype::WType::F16 | llm170_core::wtype::WType::Bf16
            )
        {
            let (tn, _) = model
                .w_raw_names("token_embd.weight")
                .ok_or_else(|| "token_embd.weight: 원시 텐서 이름 부재".to_string())?;
            let len = ld.tensor_len(&tn)? as usize;
            let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
            dec.upload_embed_stream(
                w.n_out as usize,
                w.n_in as usize,
                w.ty == llm170_core::wtype::WType::Bf16,
                len,
                &mut f,
            )?;
        }
        ld.log_summary();
        let mtp = if window.is_none() {
            upload_mtp(dec, model)?
        } else {
            false
        };
        Ok((head_gpu && dec.owns_final_layer(), mtp))
    }

    pub fn layer_window(&self) -> llm170_backend_gpu::LayerWindow {
        self.dec.layer_window()
    }

    pub fn owns_final_layer(&self) -> bool {
        self.dec.owns_final_layer()
    }

    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        self.dec.mem_stats()
    }

    pub fn active_weight_bytes(&self) -> u64 {
        self.dec.active_weight_bytes()
    }

    pub fn moe_mode(&self) -> &'static str {
        self.dec.moe_mode()
    }

    pub fn mcache_stats(&self) -> (u64, u64, usize) {
        self.dec.mcache_stats()
    }

    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.dec.copy_stats()
    }

    fn forward(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        self.dec.forward_device(slot, row)
    }

    fn token_greedy(&mut self, slot: usize, row: &[f32]) -> Result<u32, String> {
        if self.head_gpu {
            return self.dec.forward_device_argmax(slot, row);
        }
        let lg = self.logits(slot, row)?;
        Ok(llm170_core::matmul::greedy_from(&lg))
    }

    fn logits(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_gpu {
            let mut lg = self.dec.forward_device_head(slot, row)?;
            self.softcap_row(&mut lg);
            return Ok(lg);
        }
        let xn = self.forward(slot, row)?;
        let mut lg = self.head_logits(&xn)?;
        self.softcap_row(&mut lg);
        Ok(lg)
    }

    pub(super) fn softcap_row(&self, lg: &mut [f32]) {
        if self.softcap > 0.0 {
            for v in lg.iter_mut() {
                *v = llm170_core::ops::logit_softcap(*v, self.softcap);
            }
        }
    }

    pub(super) fn head_logits(&self, xn: &[f32]) -> Result<Vec<f32>, String> {
        let head = self
            .model
            .w("output.weight")
            .ok_or_else(|| "output.weight 부재 — head 계약 위반".to_string())?;
        let mut lg = vec![0.0f32; head.n_out as usize];
        llm170_core::matmul::matmul(xn, &head, &mut lg);
        Ok(lg)
    }

    pub(super) fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.model.hp.n_embd;
        let mut rows = Vec::with_capacity(tokens.len() * h);
        for tok in tokens {
            rows.extend_from_slice(&self.model.embed_row(*tok).map_err(|e| e.to_string())?);
        }
        Ok(rows)
    }

    fn np_batch_argmax(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
        label: &str,
    ) -> Result<Option<Vec<u32>>, String> {
        if (2..=llm170_backend_gpu::BATCH_DEC_MAX).contains(&seq_ids.len()) {
            let rows = self.embed_rows(tokens)?;
            match self.dec.forward_device_argmax_batch(seq_ids, &rows) {
                Ok(toks) => return Ok(Some(toks)),
                Err(e) => eprintln!("# {label} 폴백(직렬): {e}"),
            }
        }
        Ok(None)
    }

    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        match self.prefill_impl(seq, tokens, true, false)? {
            Some(PrefillOut::Logits(v)) => Ok(v),
            _ => Err("prefill: 로짓 없음".into()),
        }
    }

    pub fn prefill_greedy(&mut self, seq: usize, tokens: &[u32]) -> Result<u32, String> {
        match self.prefill_impl(seq, tokens, true, true)? {
            Some(PrefillOut::Token(t)) => Ok(t),
            _ => Err("prefill_greedy: 토큰 없음".into()),
        }
    }

    pub fn prefill_mixed_cap(&self) -> usize {
        if self.dec.is_moe() && self.dec.moe_experts_resident() {
            512
        } else {
            128
        }
    }

    pub fn prefill_state(&mut self, seq: usize, tokens: &[u32]) -> Result<(), String> {
        self.prefill_impl(seq, tokens, false, false).map(|_| ())
    }

    fn prefill_impl(
        &mut self,
        seq: usize,
        tokens: &[u32],
        want: bool,
        greedy: bool,
    ) -> Result<Option<PrefillOut>, String> {
        let n = tokens.len();
        if n == 0 {
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = None;
        let tmax = match self.tuning.chunk_t {
            Some(v) => prefill_tmax(
                self.dec.is_moe(),
                self.dec.moe_experts_resident(),
                Some(&v.to_string()),
            ),
            None => prefill_tmax(
                self.dec.is_moe(),
                self.dec.moe_experts_resident(),
                llm170_diag::flag::val("LLM170_PREFILL_T"),
            ),
        };
        let mut i = 0usize;
        while i < n {
            let t = (n - i).min(tmax);
            let is_last = i + t == n;
            let part = &tokens[i..i + t];
            let mtp_pos = if self.mtp_drafts_on() {
                Some(self.dec.slot_pos_of(seq))
            } else {
                None
            };
            if t == 1 {
                let row = self.model.embed_row(part[0]).map_err(|e| e.to_string())?;
                if is_last && want && self.head_gpu {
                    last = Some(if greedy && self.softcap <= 0.0 {
                        PrefillOut::Token(self.dec.forward_device_argmax(seq, &row)?)
                    } else {
                        PrefillOut::Logits(self.dec.forward_prefill(seq, &row, 1, true)?)
                    });
                } else if is_last && want {
                    let xn = self.dec.forward_prefill(seq, &row, 1, false)?;
                    last = Some(PrefillOut::Logits(self.head_logits(&xn)?));
                } else {
                    self.dec.forward_prefill(seq, &row, 1, false)?;
                }
            } else if self.dec.embed_ready() {
                if is_last && want && self.head_gpu {
                    last = Some(if greedy && self.softcap <= 0.0 {
                        PrefillOut::Token(self.dec.forward_prefill_toks_greedy(seq, part, t)?)
                    } else {
                        PrefillOut::Logits(self.dec.forward_prefill_toks(seq, part, t, true)?)
                    });
                } else if is_last && want {
                    let xn = self.dec.forward_prefill_toks(seq, part, t, false)?;
                    last = Some(PrefillOut::Logits(self.head_logits(&xn)?));
                } else {
                    self.dec.forward_prefill_toks(seq, part, t, false)?;
                }
            } else {
                let rows = self.embed_rows(part)?;
                if is_last && want && self.head_gpu {
                    let l = self.dec.forward_prefill(seq, &rows, t, true)?;
                    last = Some(if greedy {
                        PrefillOut::Token(llm170_core::qwen35::greedy(&l))
                    } else {
                        PrefillOut::Logits(l)
                    });
                } else if is_last && want {
                    let xn = self.dec.forward_prefill(seq, &rows, t, false)?;
                    last = Some(PrefillOut::Logits(self.head_logits(&xn)?));
                } else {
                    self.dec.forward_prefill(seq, &rows, t, false)?;
                }
            }
            if let Some(ps) = mtp_pos {
                let (calls, save_row): (&[u32], Option<usize>) = if is_last {
                    (&tokens[i + 1..i + t], Some(t - 1))
                } else {
                    (&tokens[i + 1..=i + t], None)
                };
                let emb = self.embed_rows(calls)?;
                if let Err(err) = self.dec.mtp_prefill_chunk(seq, calls, &emb, ps, save_row) {
                    eprintln!("# mtp(무시): {err}");
                }
            }
            i += t;
        }
        Ok(if want {
            let o = last.ok_or_else(|| "prefill: 빈 프롼프트 — 토큰 1개 이상 필요".to_string())?;
            Some(match o {
                PrefillOut::Logits(l) if greedy => {
                    PrefillOut::Token(llm170_core::qwen35::greedy(&l))
                }
                o => o,
            })
        } else {
            None
        })
    }

    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let rows = self.embed_rows(tokens)?;
        let h = self.model.hp.n_embd;
        let mut out = Vec::with_capacity(seq_ids.len());
        for (i, &s) in seq_ids.iter().enumerate() {
            out.push(self.logits(s, &rows[i * h..(i + 1) * h])?);
        }
        Ok(out)
    }

    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        if let Some(v) = self.np_batch_argmax(seq_ids, tokens, "np-batch")? {
            return Ok(v);
        }
        let mut out = Vec::with_capacity(seq_ids.len());
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.push(self.token_greedy(*s, &row)?);
        }
        Ok(out)
    }

    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        let row = self.model.embed_row(token).map_err(|e| e.to_string())?;
        self.token_greedy(seq, &row)
    }

    pub fn decode_greedy_k(
        &mut self,
        seq: usize,
        token: u32,
        k: usize,
        eos: u32,
    ) -> Result<Vec<u32>, String> {
        if !self.head_gpu {
            return Err("kstep: GPU head 미상주".into());
        }
        self.dec.decode_k_greedy(seq, token, k, eos)
    }

    pub fn decode_np_mixed(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        if !(2..=llm170_backend_gpu::BATCH_DEC_MAX).contains(&seq_ids.len()) {
            return Err("mixed: t 2..=BATCH_DEC_MAX 전용".into());
        }
        let rows = self.embed_rows(tokens)?;
        let toks = self.dec.forward_device_argmax_batch(seq_ids, &rows)?;
        let lg = self.dec.batch_read_logits(seq_ids.len())?;
        Ok((toks, lg))
    }

    pub fn decode_np_logits(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        if self
            .np_batch_argmax(seq_ids, tokens, "np-logits 배치")?
            .is_some()
        {
            match self.dec.batch_read_logits(seq_ids.len()) {
                Ok(mut lg) => {
                    self.softcap_row(&mut lg);
                    return Ok(lg);
                }
                Err(e) => eprintln!("# np-logits 판독 폴백(직렬): {e}"),
            }
        }
        let mut out = Vec::new();
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.extend_from_slice(&self.logits(*s, &row)?);
        }
        Ok(out)
    }

    pub fn prof_report(&mut self, label: &str) -> Result<String, String> {
        self.dec.prof_report(label)
    }

    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        self.dec.kv_release(seq);
        self.dec.reset_state(seq)
    }

    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        self.dec.kv_grow(slot, n_tokens)
    }

    pub fn kv_free_blocks(&self) -> usize {
        self.dec.kv_free_blocks()
    }

    pub fn kv_total_blocks(&self) -> usize {
        self.dec.kv_total_blocks()
    }

    pub fn kv_release_slot(&mut self, slot: usize) -> usize {
        self.dec.kv_release(slot)
    }

    pub fn kv_blocks_needed(&self, n_tokens: usize) -> usize {
        self.dec.kv_blocks_needed(n_tokens)
    }

    pub fn kv_cap_tokens(&self) -> usize {
        self.dec.kv_cap_tokens()
    }

    pub fn kv_used_blocks(&self, slot: usize) -> usize {
        self.dec.kv_used_blocks(slot)
    }

    pub fn kv_tier_bytes(&self) -> (u64, u64, usize, usize) {
        self.dec.kv_tier_bytes()
    }

    pub fn kv_tier_free(&self) -> (usize, usize) {
        self.dec.kv_tier_free()
    }

    pub fn eos(&self) -> u32 {
        llm170_core::qwen35::EOS_EOT
    }
    pub fn vocab(&self) -> usize {
        self.model.hp.vocab
    }
    pub fn seq_pos(&self, slot: usize) -> u32 {
        self.dec.slot_pos_of(slot)
    }
    pub fn restore_prefix(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        self.dec.restore_gdn(slot, upto)
    }
    pub fn reuse_valid(&self, _slot: usize, _l: usize, _prompt_len: usize) -> bool {
        true
    }
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, String> {
        self.model.embed_row(tok).map_err(|e| e.to_string())
    }
    pub fn n_embd(&self) -> usize {
        self.model.hp.n_embd
    }

    #[allow(dead_code)]
    pub fn reset_states(&mut self) -> Result<(), String> {
        for s in 0..self.n_slots {
            self.dec.reset_state(s)?;
        }
        self.dec.mtp_kv_reset()?;
        Ok(())
    }

    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(String::as_str)
            .unwrap_or("")
            .replace('Ġ', " ")
            .replace('Ċ', "\n")
    }
}
