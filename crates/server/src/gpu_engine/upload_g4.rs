
pub(crate) struct G4HostCfg {
    pub softcap: f32,
    pub nw: Vec<f32>,
    pub layer_scalar: Vec<f32>,
    pub pes: Vec<f32>,
    pub qnw: Vec<f32>,
    pub knw: Vec<f32>,
    pub d_max: usize,
}

impl G4HostCfg {
    pub fn nw_row(&self, il: usize, k: usize, hidden: usize) -> &[f32] {
        &self.nw[(il * 8 + k) * hidden..(il * 8 + k + 1) * hidden]
    }

    pub fn out_norm_row(&self, n_layer: usize, hidden: usize) -> &[f32] {
        &self.nw[n_layer * 8 * hidden..(n_layer * 8 + 1) * hidden]
    }

    pub fn qk_rows(&self, il: usize, d: usize) -> (&[f32], &[f32]) {
        let b = il * self.d_max;
        (&self.qnw[b..b + d], &self.knw[b..b + d])
    }

    pub fn pes_row(&self, il: usize, n_exp: usize) -> &[f32] {
        &self.pes[il * n_exp..(il + 1) * n_exp]
    }
}

pub(crate) fn upload_model_g4(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    ctx: usize,
) -> Result<G4HostCfg, String> {
    use llm170_core::wtype::WType;
    let hp = model.hp.clone();
    let h = hp.n_embd;
    let t0 = std::time::Instant::now();
    let mut ld = super::upload::DirectLoad::new(model.store());
    for il in 0..hp.n_layer {
        for suf in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "moe_gate.weight",
        ] {
            let name = format!("blk.{il}.{suf}");
            let Some(w) = model.w(&name) else {
                continue;
            };
            let (tn, sn) = model
                .w_names(&name)
                .ok_or_else(|| format!("{name}: 원시 텐서 이름 부재"))?;
            match w.ty {
                WType::W4a16Split => {
                    let sn = sn.ok_or_else(|| format!("{name}: scale 이름 부재"))?;
                    let (ql, sl) = (ld.tensor_len(&tn)? as usize, ld.tensor_len(&sn)? as usize);
                    let mut f = |which: usize, off: u64, buf: &mut [u8]| {
                        let n = if which == 0 { tn.as_str() } else { sn.as_str() };
                        ld.read_into(n, off, buf)
                    };
                    dec.upload_lin_g_stream(
                        &name,
                        w.n_out as usize,
                        w.n_in as usize,
                        w.group,
                        w.scale_bf16,
                        ql,
                        sl,
                        &mut f,
                    )?;
                }
                WType::Bf16 => {
                    let len = ld.tensor_len(&tn)? as usize;
                    let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
                    dec.upload_plain_stream(&name, w.n_out as usize, w.n_in as usize, len, &mut f)?;
                }
                _ => {}
            }
        }
    }
    let inv_h = 1.0f32 / (h as f32).sqrt();
    let mut nw: Vec<f32> = Vec::with_capacity((8 * hp.n_layer + 1) * h);
    for il in 0..hp.n_layer {
        for suf in [
            "attn_norm.weight",
            "post_attn_norm.weight",
            "pre_ff_norm.weight",
            "post_ff_norm1.weight",
            "pre_ff_norm2.weight",
            "post_ff_norm2.weight",
            "post_ff_norm.weight",
        ] {
            let v = model
                .f32_vec(&format!("blk.{il}.{suf}"))
                .map_err(|e| e.to_string())?;
            if v.len() != h {
                return Err(format!("g4 노름 형상 위반 blk.{il}.{suf}: {}", v.len()));
            }
            nw.extend(v);
        }
        let sc = model
            .f32_vec(&format!("blk.{il}.moe_scale"))
            .map_err(|e| e.to_string())?;
        if sc.len() != h {
            return Err(format!("g4 moe_scale 형상 위반 L{il}: {}", sc.len()));
        }
        nw.extend(sc.iter().map(|v| v * inv_h));
    }
    nw.extend(
        model
            .f32_vec("output_norm.weight")
            .map_err(|e| e.to_string())?,
    );
    dec.set_norm_weights(&nw, 8 * hp.n_layer + 1)?;
    let d_max = hp.head_dim.iter().copied().max().unwrap_or(0);
    let mut qnw = vec![0f32; hp.n_layer * d_max];
    let mut knw = vec![0f32; hp.n_layer * d_max];
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        let q = model
            .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
            .map_err(|e| e.to_string())?;
        let k = model
            .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
            .map_err(|e| e.to_string())?;
        if q.len() != d || k.len() != d {
            return Err(format!(
                "g4 QK 노름 형상 L{il}: q={} k={} d={d}",
                q.len(),
                k.len()
            ));
        }
        qnw[il * d_max..il * d_max + d].copy_from_slice(&q);
        knw[il * d_max..il * d_max + d].copy_from_slice(&k);
    }
    let layers: Vec<llm170_backend_gpu::AttnLayer> = (0..hp.n_layer)
        .map(|il| llm170_backend_gpu::AttnLayer {
            kv_heads: hp.n_kv[il],
            d: hp.head_dim[il],
            sliding: !hp.full[il],
            k_eq_v: model.kv_eq_v(il),
            kq_scale: 1.0,
        })
        .collect();
    dec.set_attn(
        llm170_backend_gpu::AttnDims {
            n_attn: hp.n_layer,
            q_heads: hp.n_head,
            kv_heads: hp.n_kv[0],
            d: hp.head_dim[0],
            cap: ctx,
            interval: 1,
            sliding_window: hp.sw,
        },
        layers,
        &qnw,
        &knw,
    )?;
    let attn_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    if hp.n_experts > 0 {
        let (group, scale_bf16) = model.expert_quant();
        dec.set_moe(
            hp.n_experts,
            hp.top_k,
            hp.moe_ffn,
            0,
            group,
            scale_bf16,
            true,
        );
        dec.ensure_moe_stage_chain()?;
        let mut tab: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(hp.n_layer * hp.n_experts * 3);
        for il in 0..hp.n_layer {
            for e in 0..hp.n_experts {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    let (q, s, _, _) = model
                        .expert_slice(il, e, proj)
                        .ok_or_else(|| format!("전문가 슬라이스 부재: L{il} e{e} {proj}"))?;
                    tab.push((
                        q.as_ptr() as u64,
                        q.len() as u64,
                        s.as_ptr() as u64,
                        s.len() as u64,
                    ));
                }
            }
        }
        let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
        let free = dec.free_vram().map(|(f, _)| f).unwrap_or(0);
        let resident = match crate::engine::MOE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
                Some("0") => false,
                Some("1") => true,
                _ => free > expert_bytes + (1 << 30),
            },
        };
        if resident {
            let mut lens: Vec<(usize, usize)> = Vec::with_capacity(tab.len());
            let mut names: Vec<(String, String)> = Vec::with_capacity(tab.len());
            for il in 0..hp.n_layer {
                for e in 0..hp.n_experts {
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        let (qn, sn) = model.expert_names(il, e, proj);
                        lens.push((ld.tensor_len(&qn)? as usize, ld.tensor_len(&sn)? as usize));
                        names.push((qn, sn));
                    }
                }
            }
            let mut f = |cell: usize, which: usize, off: u64, buf: &mut [u8]| {
                let n = if which == 0 {
                    &names[cell].0
                } else {
                    &names[cell].1
                };
                ld.read_into(n, off, buf)
            };
            dec.upload_experts_resident_streamed(&lens, &mut f)?;
        } else {
            dec.set_expert_table(tab);
        }
        eprintln!(
            "[g4-moe] 전문가 {:.2}GiB · 여유 {:.2}GiB → {} (g{group}{})",
            expert_bytes as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
            if resident {
                "전량 상주"
            } else {
                "호스트 스트리밍"
            },
            if scale_bf16 { "/bf16" } else { "" }
        );
    }
    let moe_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    {
        let hw = model.wchk("output.weight").map_err(|e| e.to_string())?;
        let (tn, _) = model
            .w_names("output.weight")
            .ok_or_else(|| "output.weight: 원시 텐서 이름 부재".to_string())?;
        let len = ld.tensor_len(&tn)? as usize;
        let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
        dec.upload_head_stream(hw.n_out as usize, hw.n_in as usize, len, &mut f)?;
    }
    ld.log_summary();
    let head_ms = t2.elapsed().as_secs_f64() * 1e3;
    let mut layer_scalar = Vec::with_capacity(hp.n_layer);
    let mut pes: Vec<f32> = Vec::with_capacity(hp.n_layer * hp.n_experts);
    let mut rp: Vec<f32> = Vec::with_capacity(hp.n_layer * 7);
    for il in 0..hp.n_layer {
        let ls = model
            .f32_vec(&format!("blk.{il}.layer_scalar"))
            .map_err(|e| e.to_string())?;
        layer_scalar.push(
            *ls.first()
                .ok_or_else(|| format!("layer_scalar 부재 L{il}"))?,
        );
        pes.extend(
            model
                .f32_vec(&format!("blk.{il}.moe_per_expert_scale"))
                .map_err(|e| e.to_string())?,
        );
        let d = hp.head_dim[il];
        rp.extend([
            d as f32,
            (hp.n_rot[il] / 2) as f32,
            (d / 2) as f32,
            d as f32,
            hp.rope_base[il],
            1.0,
            if model.kv_eq_v(il) { 1.0 } else { 0.0 },
        ]);
    }
    dec.set_g4_rope(&rp)?;
    dec.set_g4_layer_scalar(&layer_scalar)?;
    dec.build_multi_tables_g4()?;
    eprintln!("[g4] 업로드 — 선형·노름 {attn_ms:.0}ms · MoE {moe_ms:.0}ms · head {head_ms:.0}ms");
    Ok(G4HostCfg {
        softcap: hp.softcap,
        nw,
        layer_scalar,
        pes,
        qnw,
        knw,
        d_max,
    })
}
