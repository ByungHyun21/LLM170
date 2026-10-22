use super::g4::{g4_chunk, g4_step};
use super::upload_g4::{G4HostCfg, upload_model_g4};
use llm170_backend_gpu::W4a16Dec;
use std::path::Path;

pub struct GemmaEngine {
    model: llm170_core::gemma4::Model,
    cfg: G4HostCfg,
    dec: W4a16Dec,
    pub tuning: super::Tuning,
    #[allow(dead_code)]
    n_slots: usize,
}

impl GemmaEngine {
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
        kv_vram_cap: Option<u64>,
    ) -> Result<Self, String> {
        let _ = kv_reserve;
        let model = llm170_core::gemma4::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        dec.set_kv_budget(kv_budget);
        dec.set_kv_vram_cap(kv_vram_cap);
        match Self::fill(&mut dec, &model, ctx) {
            Ok(cfg) => Ok(GemmaEngine {
                model,
                cfg,
                dec,
                tuning: super::Tuning::default(),
                n_slots,
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
        model: &llm170_core::gemma4::Model,
        ctx: usize,
    ) -> Result<G4HostCfg, String> {
        let cfg = upload_model_g4(dec, model, ctx)?;
        for il in 0..model.hp.n_layer {
            dec.set_g4_pes(il, cfg.pes_row(il, model.hp.n_experts))?;
        }
        Ok(cfg)
    }

    fn step(&mut self, seq: usize, tok: u32, softcap: bool) -> Result<Vec<f32>, String> {
        let pos = self.dec.slot_pos[seq];
        self.dec.kv_grow(seq, pos as usize + 1)?;
        if !self.dec.moe_experts_resident() {
            let lg = g4_step(
                &mut self.dec,
                &self.model,
                &self.cfg,
                seq,
                tok,
                pos,
                softcap,
            )?;
            self.dec.slot_pos[seq] = pos + 1;
            self.maybe_trim(seq)?;
            return Ok(lg);
        }
        let rows = self.model.embed_row(tok).map_err(|e| e.to_string())?;
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            &[seq],
            &[pos],
            &rows,
            1,
            false,
            softcap,
        )?;
        self.dec.slot_pos[seq] = pos + 1;
        self.maybe_trim(seq)?;
        Ok(lgs.into_iter().next().unwrap_or_default())
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
        if tokens.is_empty() {
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = Vec::new();
        let t_chunk = if self.dec.moe_experts_resident() {
            self.tuning
                .chunk_t
                .unwrap_or(llm170_backend_gpu::G4_CHUNK_MAX)
                .min(llm170_backend_gpu::G4_CHUNK_MAX)
        } else {
            1
        };
        for chunk in tokens.chunks(t_chunk) {
            let pos = self.dec.slot_pos[seq];
            self.dec.kv_grow(seq, pos as usize + chunk.len())?;
            if chunk.len() == 1 {
                last = self.step(seq, chunk[0], true)?;
            } else {
                let mut rows: Vec<f32> = Vec::with_capacity(chunk.len() * self.model.hp.n_embd);
                for &tk in chunk {
                    rows.extend_from_slice(&self.model.embed_row(tk).map_err(|e| e.to_string())?);
                }
                let csv: Vec<usize> = vec![seq; chunk.len()];
                let cpos: Vec<u32> = (0..chunk.len()).map(|i| pos + i as u32).collect();
                let lgs = g4_chunk(
                    &mut self.dec,
                    &self.model,
                    &self.cfg,
                    &csv,
                    &cpos,
                    &rows,
                    chunk.len(),
                    want,
                    true,
                )?;
                self.dec.slot_pos[seq] = pos + chunk.len() as u32;
                if want {
                    last = lgs.into_iter().last().unwrap_or_default();
                }
            }
        }
        self.maybe_trim(seq)?;
        Ok(if want { Some(last) } else { None })
    }

    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        self.decode_sc(seq_ids, tokens, true)
    }

    fn decode_sc(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
        softcap: bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        if seq_ids.len() != tokens.len() {
            return Err(format!(
                "decode: seqs={} toks={} 불일치",
                seq_ids.len(),
                tokens.len()
            ));
        }
        if seq_ids.is_empty() {
            return Err("decode: 빈 배치".into());
        }
        let t = seq_ids.len();
        if t == 1 {
            return Ok(vec![self.step(seq_ids[0], tokens[0], softcap)?]);
        }
        if t > 16 || !self.dec.moe_experts_resident() {
            let mut out = Vec::with_capacity(t);
            for (&s, &tok) in seq_ids.iter().zip(tokens.iter()) {
                out.push(self.step(s, tok, softcap)?);
            }
            return Ok(out);
        }
        let mut rows = Vec::with_capacity(t * self.model.hp.n_embd);
        let mut pos: Vec<u32> = Vec::with_capacity(t);
        for (&s, &tok) in seq_ids.iter().zip(tokens.iter()) {
            rows.extend_from_slice(&self.model.embed_row(tok).map_err(|e| e.to_string())?);
            let p = self.dec.slot_pos[s];
            self.dec.kv_grow(s, p as usize + 1)?;
            pos.push(p);
        }
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            seq_ids,
            &pos,
            &rows,
            t,
            false,
            true,
        )?;
        for (i, &s) in seq_ids.iter().enumerate() {
            self.dec.slot_pos[s] = pos[i] + 1;
        }
        for &s in seq_ids {
            self.maybe_trim(s)?;
        }
        Ok(lgs)
    }

    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        Ok(self
            .decode_sc(seq_ids, tokens, false)?
            .iter()
            .map(|l| llm170_core::matmul::greedy_from(l))
            .collect())
    }

    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        let lg = self.step(seq, token, false)?;
        Ok(llm170_core::matmul::greedy_from(&lg))
    }

    pub fn decode_np_mixed(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let rows = self.decode(seq_ids, tokens)?;
        let toks = rows
            .iter()
            .map(|l| llm170_core::matmul::greedy_from(l))
            .collect();
        let mut flat = Vec::with_capacity(rows.len() * self.model.hp.vocab);
        for l in &rows {
            flat.extend_from_slice(l);
        }
        Ok((toks, flat))
    }

    pub fn decode_np_logits(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        let rows = self.decode(seq_ids, tokens)?;
        let mut flat = Vec::with_capacity(rows.len() * self.model.hp.vocab);
        for l in &rows {
            flat.extend_from_slice(l);
        }
        Ok(flat)
    }

    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        let _ = self.dec.kv_release(seq);
        if seq < self.dec.slot_pos.len() {
            self.dec.slot_pos[seq] = 0;
        }
        Ok(())
    }

    pub fn enable_spec(&mut self) -> Result<(), String> {
        Ok(())
    }
    pub fn spec_on(&self) -> bool {
        true
    }
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        let lgs = self.spec_run(slot, rows, t)?;
        Ok(lgs.iter().map(|l| llm170_core::gemma4::greedy(l)).collect())
    }
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let lgs = self.spec_run(slot, rows, t)?;
        let mut flat = Vec::with_capacity(t * self.model.hp.vocab);
        for l in &lgs {
            flat.extend_from_slice(l);
        }
        Ok(flat)
    }
    fn spec_run(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<Vec<f32>>, String> {
        if !(2..=8).contains(&t) {
            return Err(format!("g4 spec: t={t} (2..=8 계약)"));
        }
        let pos0 = self.dec.slot_pos[slot];
        self.dec.kv_grow(slot, pos0 as usize + t)?;
        let csv: Vec<usize> = vec![slot; t];
        let cpos: Vec<u32> = (0..t).map(|i| pos0 + i as u32).collect();
        let lgs = g4_chunk(
            &mut self.dec,
            &self.model,
            &self.cfg,
            &csv,
            &cpos,
            rows,
            t,
            false,
            true,
        )?;
        self.dec.slot_pos[slot] = pos0 + t as u32;
        Ok(lgs)
    }
    pub fn decode_greedy_k(
        &mut self,
        _seq: usize,
        _token: u32,
        _k: usize,
        _eos: u32,
    ) -> Result<Vec<u32>, String> {
        Err("gemma4: kstep 미지원(v0)".into())
    }
    pub fn mtp_drafts_on(&self) -> bool {
        false
    }

    pub fn warm_spec_graphs(&mut self, _n_slots: usize, _eos: u32) -> Result<(), String> {
        Ok(())
    }

    pub fn seq_pos(&self, slot: usize) -> u32 {
        self.dec.slot_pos.get(slot).copied().unwrap_or(0)
    }
    pub fn restore_prefix(&mut self, slot: usize, upto: u32) -> Result<u32, String> {
        if slot >= self.n_slots {
            return Err(format!(
                "gemma4: slot {slot} 범위 밖(문맥 {})",
                self.n_slots
            ));
        }
        let p = self.dec.slot_pos[slot];
        if upto == 0 || upto > p {
            return Err(format!("gemma4: restore_prefix {upto} > pos {p}"));
        }
        let t_grid = if self.dec.moe_experts_resident() {
            self.tuning
                .chunk_t
                .unwrap_or(llm170_backend_gpu::G4_CHUNK_MAX)
                .min(llm170_backend_gpu::G4_CHUNK_MAX)
        } else {
            1
        };
        let q = if t_grid > 1 {
            (upto / t_grid as u32) * t_grid as u32
        } else {
            upto
        };
        if q == 0 {
            return Err("gemma4: restore_prefix 그리드 양자화 = 0 — 콜드".into());
        }
        if self.dec.kv_floor_of(slot) > q {
            return Err(format!(
                "gemma4: restore_prefix {q} < floor {}",
                self.dec.kv_floor_of(slot)
            ));
        }
        self.dec.slot_pos[slot] = q;
        Ok(q)
    }
    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        self.dec.kv_grow(slot, n_tokens)
    }
    pub fn kv_total_blocks(&self) -> usize {
        self.dec.kv_total_blocks()
    }
    pub fn kv_free_blocks(&self) -> usize {
        self.dec.kv_free_blocks()
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
    pub fn kv_release_slot(&mut self, slot: usize) -> usize {
        self.dec.kv_release(slot)
    }
    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(|s| s.replace('▁', " "))
            .unwrap_or_default()
    }
    pub fn prof_report(&mut self, _label: &str) -> Result<String, String> {
        Err("gemma4: prof_report 미지원(v0)".into())
    }
    pub fn spec_rollback(&mut self, _slot: usize, _keep: usize) -> Result<(), String> {
        Ok(())
    }
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        let p = self.dec.slot_pos[slot];
        self.dec.slot_pos[slot] = p.saturating_sub(n);
        Ok(())
    }
    pub fn mtp_first(&mut self, _slot: usize, _tok: u32) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_single(&mut self, _slot: usize, _tok: u32) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_pre_round(&mut self, _acc: usize) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_post_round(
        &mut self,
        _slot: usize,
        _accepted: &[u32],
        _corrected: u32,
    ) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_slot_kv_reset(&mut self) -> Result<(), String> {
        Err("gemma4: MTP 미지원".into())
    }
    pub fn mtp_slot_fresh(&mut self, _slot: usize) {}
    pub fn mtp_slot_reused(&mut self, _slot: usize) {}
    pub fn mtp_take_draft(&mut self, _slot: usize) -> Option<Vec<u32>> {
        None
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
    pub fn eos(&self) -> u32 {
        1
    }
    pub fn embed_row(&self, tok: u32) -> Result<Vec<f32>, String> {
        self.model.embed_row(tok).map_err(|e| e.to_string())
    }
    pub fn vocab(&self) -> usize {
        self.model.hp.vocab
    }
    pub fn n_embd(&self) -> usize {
        self.model.hp.n_embd
    }

    pub fn layer_window(&self) -> llm170_backend_gpu::LayerWindow {
        self.dec.layer_window()
    }

    pub fn owns_final_layer(&self) -> bool {
        self.dec.owns_final_layer()
    }

    pub fn reuse_valid(&self, slot: usize, l: usize, prompt_len: usize) -> bool {
        reuse_valid(
            self.dec.kv_floor_of(slot),
            l as u32,
            prompt_len as u32,
            self.model.hp.sw as u32,
        )
    }

    fn maybe_trim(&mut self, slot: usize) -> Result<(), String> {
        let pos = self.dec.slot_pos.get(slot).copied().unwrap_or(0);
        let f = floor_for_pos(pos, self.model.hp.sw as u32);
        if f > self.dec.kv_floor_of(slot) {
            let n = self.dec.kv_release_below(slot, f)?;
            eprintln!("# g4-trim: slot{slot} floor {f}행(해제 {n}블록)");
        }
        Ok(())
    }
}

fn floor_for_pos(pos: u32, sw: u32) -> u32 {
    const SLACK: u32 = 512;
    if pos > sw + SLACK {
        ((pos - sw - SLACK) / 512) * 512
    } else {
        0
    }
}

fn reuse_valid(floor: u32, l: u32, prompt_len: u32, sw: u32) -> bool {
    floor <= l && floor <= prompt_len.saturating_sub(sw)
}

#[cfg(test)]
mod tests {
    use super::{floor_for_pos, reuse_valid};

    #[test]
    fn floor_quantized_monotone() {
        let sw = 1024;
        assert_eq!(floor_for_pos(0, sw), 0);
        assert_eq!(floor_for_pos(1536, sw), 0, "sw+여유 이하는 미트림");
        assert_eq!(floor_for_pos(1537, sw), 0, "512 미만 내림");
        assert_eq!(floor_for_pos(2048, sw), 512);
        assert_eq!(floor_for_pos(8192, sw), 6656, "(8192−1536)/512 내림");
        assert!(floor_for_pos(9000, sw) >= floor_for_pos(8192, sw), "단조");
    }

    #[test]
    fn reuse_guard_needs_floor_below_l_and_window() {
        let sw = 1024;
        assert!(reuse_valid(0, 0, 100, sw), "미트림은 항상 유효");
        assert!(reuse_valid(0, 512, 600, sw), "미트림 단문 유효");
        assert!(reuse_valid(6500, 8000, 8500, sw), "장문 계속 유효");
        assert!(!reuse_valid(6500, 512, 8000, sw), "l이 floor 미만 → 콜드");
        assert!(
            !reuse_valid(6500, 7000, 7000, sw),
            "prompt−sw가 floor 미만 → 콜드"
        );
        assert!(reuse_valid(6500, 7000, 8000, sw), "둘 다 만족 → 유효");
    }
}
