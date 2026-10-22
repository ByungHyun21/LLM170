
use super::QwenEngine;
use llm170_backend_gpu::{forward_2shard, forward_2shard_argmax, forward_2shard_toks};
use std::path::Path;

pub struct PpEngine {
    a: Box<QwenEngine>,
    b: Box<QwenEngine>,
}

impl PpEngine {
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
        kv_vram_cap: Option<u64>,
    ) -> Result<Self, String> {
        let model = llm170_core::qwen35::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mtp_attn = model
            .w_raw("mtp.layers.0.self_attn.q_proj.weight")
            .is_some()
            && crate::engine::mtp_upload_on(true);
        let n_layer = hp.n_layer;
        if n_layer % 2 != 0 {
            return Err(format!("PP 2샤드: 홀수 층({n_layer}) — 균등 분할 불가"));
        }
        let k = n_layer / 2;
        let interval = hp.full_attn_interval.max(1);
        let owned = |range: std::ops::Range<usize>| {
            range.filter(|&il| (il + 1).is_multiple_of(interval)).count()
        };
        let a_owned = owned(0..k);
        let b_owned = owned(k..n_layer) + usize::from(mtp_attn);
        let total = (a_owned + b_owned).max(1);
        let split =
            |v: Option<u64>, n: usize| v.map(|b| (b as u128 * n as u128 / total as u128) as u64);
        let mut a = Box::new(QwenEngine::load_dev(
            dir,
            n_slots,
            ctx,
            split(kv_budget, a_owned),
            split(kv_reserve, a_owned),
            split(kv_vram_cap, a_owned),
            0,
            Some((0, k)),
        )?);
        let b = match QwenEngine::load_dev(
            dir,
            n_slots,
            ctx,
            split(kv_budget, b_owned),
            split(kv_reserve, b_owned),
            split(kv_vram_cap, b_owned),
            1,
            Some((k, n_layer)),
        ) {
            Ok(b) => Box::new(b),
            Err(e) => {
                a.dec.free_all_device();
                return Err(e);
            }
        };
        Ok(PpEngine { a, b })
    }

    fn logits_1(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.b.head_gpu {
            let mut lg = forward_2shard(&mut self.a.dec, &mut self.b.dec, slot, row, 1, true)?;
            self.b.softcap_row(&mut lg);
            return Ok(lg);
        }
        let xn = forward_2shard(&mut self.a.dec, &mut self.b.dec, slot, row, 1, false)?;
        let mut lg = self.b.head_logits(&xn)?;
        self.b.softcap_row(&mut lg);
        Ok(lg)
    }

    fn token_greedy(&mut self, slot: usize, row: &[f32]) -> Result<u32, String> {
        if self.b.head_gpu {
            return forward_2shard_argmax(&mut self.a.dec, &mut self.b.dec, slot, row);
        }
        let lg = self.logits_1(slot, row)?;
        Ok(llm170_core::matmul::greedy_from(&lg))
    }

    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.prefill_impl(seq, tokens, true)
            .map(|o| o.unwrap_or_default())
    }

    pub fn prefill_greedy(&mut self, seq: usize, tokens: &[u32]) -> Result<u32, String> {
        let l = self.prefill(seq, tokens)?;
        Ok(llm170_core::qwen35::greedy(&l))
    }

    pub fn prefill_mixed_cap(&self) -> usize {
        128
    }

    pub fn prefill_state(&mut self, seq: usize, tokens: &[u32]) -> Result<(), String> {
        self.prefill_impl(seq, tokens, false).map(|_| ())
    }

    fn prefill_impl(
        &mut self,
        seq: usize,
        tokens: &[u32],
        want: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        let n = tokens.len();
        if n == 0 {
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let tmax = match self.a.tuning.chunk_t {
            Some(v) => super::prefill_tmax(
                self.a.dec.is_moe(),
                self.a.dec.moe_experts_resident(),
                Some(&v.to_string()),
            ),
            None => super::prefill_tmax(
                self.a.dec.is_moe(),
                self.a.dec.moe_experts_resident(),
                llm170_diag::flag::val("LLM170_PREFILL_T"),
            ),
        };
        let mut last = None;
        let mut i = 0usize;
        let mtp_on = self.b.mtp_drafts_on();
        while i < n {
            let t = (n - i).min(tmax);
            let is_last = i + t == n;
            let part = &tokens[i..i + t];
            let mtp_pos = if mtp_on {
                Some(self.b.dec.slot_pos_of(seq))
            } else {
                None
            };
            if t == 1 {
                let row = self.a.model.embed_row(part[0]).map_err(|e| e.to_string())?;
                if is_last && want && self.b.head_gpu {
                    last = Some(forward_2shard(
                        &mut self.a.dec,
                        &mut self.b.dec,
                        seq,
                        &row,
                        1,
                        true,
                    )?);
                } else if is_last && want {
                    let xn = forward_2shard(&mut self.a.dec, &mut self.b.dec, seq, &row, 1, false)?;
                    last = Some(self.b.head_logits(&xn)?);
                } else {
                    forward_2shard(&mut self.a.dec, &mut self.b.dec, seq, &row, 1, false)?;
                }
            } else if self.a.dec.embed_ready() {
                if is_last && want && self.b.head_gpu {
                    last = Some(forward_2shard_toks(
                        &mut self.a.dec,
                        &mut self.b.dec,
                        seq,
                        part,
                        t,
                        true,
                    )?);
                } else if is_last && want {
                    let xn = forward_2shard_toks(
                        &mut self.a.dec,
                        &mut self.b.dec,
                        seq,
                        part,
                        t,
                        false,
                    )?;
                    last = Some(self.b.head_logits(&xn)?);
                } else {
                    forward_2shard_toks(
                        &mut self.a.dec,
                        &mut self.b.dec,
                        seq,
                        part,
                        t,
                        false,
                    )?;
                }
            } else {
                let rows = self.a.embed_rows(part)?;
                if is_last && want && self.b.head_gpu {
                    last = Some(forward_2shard(
                        &mut self.a.dec,
                        &mut self.b.dec,
                        seq,
                        &rows,
                        t,
                        true,
                    )?);
                } else if is_last && want {
                    let xn =
                        forward_2shard(&mut self.a.dec, &mut self.b.dec, seq, &rows, t, false)?;
                    last = Some(self.b.head_logits(&xn)?);
                } else {
                    forward_2shard(&mut self.a.dec, &mut self.b.dec, seq, &rows, t, false)?;
                }
            }
            if let Some(ps) = mtp_pos {
                let (calls, save_row): (&[u32], Option<usize>) = if is_last {
                    (&tokens[i + 1..i + t], Some(t - 1))
                } else {
                    (&tokens[i + 1..=i + t], None)
                };
                let emb = self.b.embed_rows(calls)?;
                if let Err(err) = self
                    .b
                    .dec
                    .mtp_prefill_chunk(seq, calls, &emb, ps, save_row)
                {
                    eprintln!("# mtp(무시): {err}");
                }
            }
            i += t;
        }
        Ok(if want {
            Some(last.ok_or_else(|| "prefill: 빈 프롬프트 — 토큰 1개 이상 필요".to_string())?)
        } else {
            None
        })
    }

    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let rows = self.a.embed_rows(tokens)?;
        let h = self.a.model.hp.n_embd;
        let mut out = Vec::with_capacity(seq_ids.len());
        for (i, &s) in seq_ids.iter().enumerate() {
            out.push(self.logits_1(s, &rows[i * h..(i + 1) * h])?);
        }
        Ok(out)
    }

    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        let mut out = Vec::with_capacity(seq_ids.len());
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.a.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.push(self.token_greedy(*s, &row)?);
        }
        Ok(out)
    }

    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        let row = self.a.model.embed_row(token).map_err(|e| e.to_string())?;
        self.token_greedy(seq, &row)
    }

    pub fn decode_greedy_k(
        &mut self,
        _seq: usize,
        _token: u32,
        _k: usize,
        _eos: u32,
    ) -> Result<Vec<u32>, String> {
        Err("pp 1차: kstep 미지원 — 토큰당 경로 폴백".into())
    }

    pub fn decode_np_mixed(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let mut toks = Vec::with_capacity(seq_ids.len());
        let mut flat = Vec::new();
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.a.model.embed_row(*tok).map_err(|e| e.to_string())?;
            if self.b.head_gpu {
                toks.push(forward_2shard_argmax(
                    &mut self.a.dec,
                    &mut self.b.dec,
                    *s,
                    &row,
                )?);
                flat.extend(forward_2shard(
                    &mut self.a.dec,
                    &mut self.b.dec,
                    *s,
                    &row,
                    1,
                    true,
                )?);
            } else {
                let lg = self.logits_1(*s, &row)?;
                toks.push(llm170_core::matmul::greedy_from(&lg));
                flat.extend(lg);
            }
        }
        Ok((toks, flat))
    }

    pub fn decode_np_logits(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        let rows = self.a.embed_rows(tokens)?;
        let h = self.a.model.hp.n_embd;
        let mut out = Vec::new();
        for (i, &s) in seq_ids.iter().enumerate() {
            out.extend_from_slice(&self.logits_1(s, &rows[i * h..(i + 1) * h])?);
        }
        Ok(out)
    }


    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        let na = self.a.dec.kv_grow(slot, n_tokens)?;
        let nb = match self.b.dec.kv_grow(slot, n_tokens) {
            Ok(v) => v,
            Err(e) => {
                let _ = self.a.dec.kv_release(slot);
                return Err(e);
            }
        };
        Ok(na.max(nb))
    }

    pub fn kv_release_slot(&mut self, slot: usize) -> usize {
        self.a.dec.kv_release(slot) + self.b.dec.kv_release(slot)
    }

    pub fn kv_total_blocks(&self) -> usize {
        self.a
            .dec
            .kv_total_blocks()
            .min(self.b.dec.kv_total_blocks())
    }

    pub fn kv_free_blocks(&self) -> usize {
        self.a.dec.kv_free_blocks().min(self.b.dec.kv_free_blocks())
    }

    pub fn kv_blocks_needed(&self, n_tokens: usize) -> usize {
        self.a
            .dec
            .kv_blocks_needed(n_tokens)
            .min(self.b.dec.kv_blocks_needed(n_tokens))
    }

    pub fn kv_cap_tokens(&self) -> usize {
        self.a.dec.kv_cap_tokens().min(self.b.dec.kv_cap_tokens())
    }

    pub fn kv_used_blocks(&self, slot: usize) -> usize {
        self.a
            .dec
            .kv_used_blocks(slot)
            .max(self.b.dec.kv_used_blocks(slot))
    }

    pub fn kv_tier_bytes(&self) -> (u64, u64, usize, usize) {
        let a = self.a.dec.kv_tier_bytes();
        let b = self.b.dec.kv_tier_bytes();
        (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
    }

    pub fn kv_tier_free(&self) -> (usize, usize) {
        let a = self.a.dec.kv_tier_free();
        let b = self.b.dec.kv_tier_free();
        (a.0.min(b.0), a.1.min(b.1))
    }

    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        self.a.reset_seq(seq)?;
        self.b.reset_seq(seq)
    }

    pub fn restore_prefix(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        let pa = self.a.dec.restore_gdn(slot, upto)?;
        let pb = self.b.dec.restore_gdn(slot, upto)?;
        if pa != pb {
            return Err(format!("pp restore_prefix: 샤드 pos 불일치 {pa}/{pb}"));
        }
        Ok(pa)
    }

    pub fn seq_pos(&self, slot: usize) -> u32 {
        self.a.dec.slot_pos_of(slot)
    }

    pub fn reuse_valid(&self, slot: usize, l: usize, prompt_len: usize) -> bool {
        self.a.reuse_valid(slot, l, prompt_len)
    }


    pub fn enable_spec(&mut self) -> Result<(), String> {
        if llm170_diag::flag::val("LLM170_PP_SPEC") == Some("0") {
            return Err("pp spec disabled(env)".into());
        }
        self.a.dec.enable_spec()?;
        self.b.dec.enable_spec()
    }
    pub fn spec_on(&self) -> bool {
        self.a.dec.spec_on() && self.b.dec.spec_on()
    }
    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        self.a.dec.spec_rollback(slot, keep)?;
        self.b.dec.spec_rollback(slot, keep)
    }
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        self.a.dec.spec_rewind_pos(slot, n)?;
        self.b.dec.spec_rewind_pos(slot, n)
    }
    pub fn spec_verify(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<u32>, String> {
        let h = self.a.n_embd();
        if rows.len() != t * h {
            return Err(format!(
                "pp spec_verify: rows={} != t{t}×h{h}",
                rows.len()
            ));
        }
        let mid = self.a.dec.forward_prefill(slot, rows, t, false)?;
        if mid.len() != t * h {
            return Err(format!(
                "pp spec_verify: 경계 잔차 {} != t{t}×h{h}",
                mid.len()
            ));
        }
        self.b.dec.spec_verify(slot, &mid, t)
    }
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.a.n_embd();
        if rows.len() != t * h {
            return Err(format!(
                "pp spec_verify_logits: rows={} != t{t}×h{h}",
                rows.len()
            ));
        }
        let mid = self.a.dec.forward_prefill(slot, rows, t, false)?;
        if mid.len() != t * h {
            return Err(format!(
                "pp spec_verify_logits: 경계 잔차 {} != t{t}×h{h}",
                mid.len()
            ));
        }
        self.b.spec_verify_logits(slot, &mid, t)
    }
    pub fn mtp_first(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        self.b.mtp_first(slot, tok)
    }
    pub fn mtp_single(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        self.b.mtp_single(slot, tok)
    }
    pub fn mtp_pre_round(&mut self, acc: usize) -> Result<(), String> {
        self.b.mtp_pre_round(acc)
    }
    pub fn mtp_post_round(
        &mut self,
        slot: usize,
        accepted: &[u32],
        corrected: u32,
    ) -> Result<(), String> {
        self.b.mtp_post_round(slot, accepted, corrected)
    }
    pub fn mtp_slot_kv_reset(&mut self) -> Result<(), String> {
        self.b.mtp_slot_kv_reset()
    }
    pub fn mtp_slot_fresh(&mut self, slot: usize) {
        self.b.mtp_slot_fresh(slot)
    }
    pub fn mtp_slot_reused(&mut self, slot: usize) {
        self.b.mtp_slot_reused(slot)
    }
    pub fn mtp_take_draft(&mut self, slot: usize) -> Option<Vec<u32>> {
        self.b.mtp_take_draft(slot)
    }
    pub fn mtp_drafts_on(&self) -> bool {
        self.b.spec_on() && self.b.mtp_drafts_on()
    }

    pub fn warm_spec_graphs(&mut self, _n_slots: usize, _eos: u32) -> Result<(), String> {
        Ok(())
    }


    pub fn apply_tuning(&mut self, t: super::Tuning) -> Result<(), String> {
        self.a.apply_tuning(t)?;
        self.b.apply_tuning(t)
    }
    pub fn ngram_knobs(&self) -> (usize, usize) {
        self.a.ngram_knobs()
    }
    pub fn prof_report(&mut self, label: &str) -> Result<String, String> {
        let ra = self.a.prof_report(label)?;
        let rb = self.b.prof_report(label)?;
        Ok(format!("{ra}\n{rb}"))
    }
    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        let a = self.a.mem_stats();
        let b = self.b.mem_stats();
        (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
    }
    pub fn active_weight_bytes(&self) -> u64 {
        self.a.active_weight_bytes() + self.b.active_weight_bytes()
    }
    pub fn moe_mode(&self) -> &'static str {
        self.a.moe_mode()
    }
    pub fn mcache_stats(&self) -> (u64, u64, usize) {
        let a = self.a.mcache_stats();
        let b = self.b.mcache_stats();
        (a.0 + b.0, a.1 + b.1, a.2 + b.2)
    }
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        let a = self.a.copy_stats();
        let b = self.b.copy_stats();
        [
            (a[0].0 + b[0].0, a[0].1 + b[0].1, a[0].2 + b[0].2),
            (a[1].0 + b[1].0, a[1].1 + b[1].1, a[1].2 + b[1].2),
            (a[2].0 + b[2].0, a[2].1 + b[2].1, a[2].2 + b[2].2),
        ]
    }
    pub fn eos(&self) -> u32 {
        self.a.eos()
    }
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, String> {
        self.a.embed_row(tok)
    }
    pub fn vocab(&self) -> usize {
        self.a.vocab()
    }
    pub fn n_embd(&self) -> usize {
        self.a.n_embd()
    }
    pub fn piece(&self, tok: u32) -> String {
        self.a.piece(tok)
    }
    pub fn layer_window(&self) -> llm170_backend_gpu::LayerWindow {
        self.a.layer_window()
    }
    pub fn owns_final_layer(&self) -> bool {
        self.b.owns_final_layer()
    }
}
