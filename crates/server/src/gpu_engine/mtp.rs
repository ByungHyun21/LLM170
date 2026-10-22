use super::QwenEngine;

impl QwenEngine {
    pub fn mtp_drafts_on(&self) -> bool {
        match crate::engine::SPEC_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
            1 => self.mtp,
            2 => false,
            _ => self.mtp && llm170_diag::flag::val("LLM170_SPEC_DRAFT_SOURCE") != Some("ngram"),
        }
    }

    pub fn warm_spec_graphs(&mut self, n_slots: usize, eos: u32) -> Result<(), String> {
        self.dec.warm_spec_graphs(n_slots, eos)
    }

    pub fn mtp_pre_round(&mut self, acc: usize) -> Result<(), String> {
        self.dec.mtp_reset_hidden();
        for r in 0..acc {
            self.dec.mtp_push_hidden(r)?;
        }
        Ok(())
    }

    const MTP_K: usize = 4;

    fn mtp_chain_round(&mut self, slot: usize, d1: u32, base_pos: u32) -> Result<Vec<u32>, String> {
        self.dec.mtp_chain_seed(slot, d1, base_pos)?;
        self.dec.mtp_chain_drafts(slot, Self::MTP_K, base_pos)
    }

    pub fn mtp_post_round(
        &mut self,
        slot: usize,
        accepted: &[u32],
        corrected: u32,
    ) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        self.dec.mtp_push_hidden(0)?;
        let mut toks: Vec<u32> = accepted.to_vec();
        toks.push(corrected);
        let mut embeds: Vec<f32> = Vec::with_capacity(toks.len() * self.model.hp.n_embd);
        for &tk in &toks {
            embeds.extend_from_slice(&self.model.embed_row(tk).map_err(|e| e.to_string())?);
        }
        let t = toks.len() as u32;
        let pos_start = self.dec.slot_pos_of(slot).saturating_sub(t);
        let draft = self.dec.mtp_run(slot, &toks, &embeds, pos_start)?;
        self.dec.mtp_fin_seed(t as usize - 1)?;
        let base_pos = pos_start + t - 1;
        let drafts = match self.mtp_chain_round(slot, draft, base_pos) {
            Ok(ds) => ds,
            Err(err) => {
                eprintln!("# mtp-chain(폴백 순차): {err}");
                let mut ds = vec![draft];
                let mut tok = draft;
                for i in 0..(Self::MTP_K.saturating_sub(1)) {
                    let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
                    let d = self
                        .dec
                        .mtp_recur_step(slot, tok, &emb, pos_start + t + i as u32)?;
                    ds.push(d);
                    tok = d;
                }
                ds
            }
        };
        if let Err(err) = self
            .dec
            .mtp_zero_row(self.dec.slot_pos_of(slot) as usize + 1)
        {
            eprintln!("# mtp(무시): {err}");
        }
        self.dec.mtp_store_drafts(slot, &drafts);
        Ok(())
    }

    pub fn mtp_single(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        self.mtp_pre_round(0)?;
        self.mtp_post_round(slot, &[], tok)
    }

    pub fn mtp_first(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
        let d = self.dec.mtp_run_prev(slot, tok, &emb)?;
        let base_pos = self.dec.slot_pos_of(slot).saturating_sub(1);
        let drafts = match self.mtp_chain_round(slot, d, base_pos) {
            Ok(ds) => ds,
            Err(err) => {
                eprintln!("# mtp-chain(폴백 순차): {err}");
                let mut ds = vec![d];
                let mut tk = d;
                for i in 0..(Self::MTP_K.saturating_sub(1)) {
                    let eb = self.model.embed_row(tk).map_err(|e| e.to_string())?;
                    let nd = self
                        .dec
                        .mtp_recur_step(slot, tk, &eb, base_pos + 1 + i as u32)?;
                    ds.push(nd);
                    tk = nd;
                }
                ds
            }
        };
        self.dec.mtp_store_drafts(slot, &drafts);
        Ok(())
    }

    pub fn mtp_slot_reused(&mut self, slot: usize) {
        self.dec.mtp_slot_off(slot);
    }

    pub fn mtp_slot_fresh(&mut self, slot: usize) {
        self.dec.mtp_slot_on(slot);
    }

    pub fn mtp_slot_kv_reset(&mut self) -> Result<(), String> {
        self.dec.mtp_kv_reset()
    }

    pub fn mtp_take_draft(&mut self, slot: usize) -> Option<Vec<u32>> {
        self.dec.mtp_take_draft(slot)
    }
}

pub(crate) fn upload_mtp(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
) -> Result<bool, String> {
    const MTP_ENABLED: bool = true;
    if !MTP_ENABLED || model.w_raw("mtp.fc.weight").is_none() {
        return Ok(false);
    }
    if model.w_raw("mtp.layers.0.mlp.gate_proj.weight").is_none() {
        eprintln!("# mtp 스킵: dense MTP 아님(MoE MTP는 차기) — n-gram 폴백");
        return Ok(false);
    }
    let mut q_out = 0usize;
    for name in [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ] {
        let w = model
            .w_raw(name)
            .ok_or_else(|| format!("mtp: {name} 부재"))?;
        if name.ends_with("attn.q_proj.weight") {
            q_out = w.n_out as usize;
        }
        dec.upload_plain(name, w.data, w.n_out as usize, w.n_in as usize)?;
    }
    let mut norms: Vec<f32> = Vec::new();
    for name in [
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ] {
        norms.extend(model.f32_vec(name).map_err(|e| e.to_string())?);
    }
    let hp = &model.hp;
    let ffn = model
        .w_raw("mtp.layers.0.mlp.gate_proj.weight")
        .map(|w| w.n_out as usize)
        .unwrap_or(0);
    dec.set_mtp(&norms, q_out, hp.n_kv * hp.head_dim, ffn)?;
    let _ = hp;
    Ok(true)
}
