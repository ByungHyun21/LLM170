use super::upload_g4::G4HostCfg;

pub(crate) fn g4_chunk(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &G4HostCfg,
    slots: &[usize],
    pos: &[u32],
    rows: &[f32],
    t: usize,
    head_last: bool,
    softcap: bool,
) -> Result<Vec<Vec<f32>>, String> {
    let hp = &model.hp;
    let rows_per_layer = 8usize;
    let mut out = dec.g4_chunk_dev(slots, pos, rows, t, head_last, rows_per_layer, hp.eps)?;
    if softcap && cfg.softcap > 0.0 {
        for r in out.iter_mut() {
            for l in r.iter_mut() {
                *l = cfg.softcap * (*l / cfg.softcap).tanh();
            }
        }
    }
    Ok(out)
}

pub(crate) fn g4_step(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &crate::gpu_engine::G4HostCfg,
    slot: usize,
    tok: u32,
    pos: u32,
    softcap: bool,
) -> Result<Vec<f32>, String> {
    let hp = &model.hp;
    let h = hp.n_embd;
    let mut x = model.embed_row(tok).map_err(|e| e.to_string())?;
    let stat = |v: &[f32]| -> String {
        let (mut mx, mut nan) = (f32::NEG_INFINITY, false);
        for &t in v {
            if t.is_nan() {
                nan = true;
            }
            mx = mx.max(t);
        }
        format!("max={mx:.4} nan={nan} len={}", v.len())
    };
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        let dbg = dec.debug_layers && hp.full[il] && il <= 5;
        let xn = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 0, h), hp.eps);
        let q = dec.gemv_f32_host(&format!("blk.{il}.attn_q.weight"), &xn)?;
        let k = dec.gemv_f32_host(&format!("blk.{il}.attn_k.weight"), &xn)?;
        let v = if model.kv_eq_v(il) {
            k.clone()
        } else {
            dec.gemv_f32_host(&format!("blk.{il}.attn_v.weight"), &xn)?
        };
        if dbg {
            eprintln!("  L{il} xn {} / q {} / k {}", stat(&xn), stat(&q), stat(&k));
        }
        let (qnw, knw) = cfg.qk_rows(il, d);
        let sw = if hp.full[il] { 0 } else { hp.sw };
        let attn_out = dec.attn_g4_host(
            slot,
            &q,
            &k,
            &v,
            qnw,
            knw,
            il,
            1,
            hp.n_head,
            hp.n_kv[il],
            d,
            sw,
            1.0,
            2,
            pos,
        )?;
        if dbg {
            eprintln!(
                "  L{il} attn_out {} / qnw {} / knw {}",
                stat(&attn_out),
                stat(qnw),
                stat(knw)
            );
        }
        let ao = dec.gemv_f32_host(&format!("blk.{il}.attn_output.weight"), &attn_out)?;
        let hh = llm170_core::ops::rms_norm(&ao, cfg.nw_row(il, 1, h), hp.eps);
        if dbg {
            eprintln!(
                "  L{il} ao {} / hh {} / wo {}",
                stat(&ao),
                stat(&hh),
                stat(cfg.nw_row(il, 1, h))
            );
        }
        for i in 0..h {
            x[i] += hh[i];
        }
        let nd = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 2, h), hp.eps);
        let g = dec.gemv_f32_host(&format!("blk.{il}.ffn_gate.weight"), &nd)?;
        let u = dec.gemv_f32_host(&format!("blk.{il}.ffn_up.weight"), &nd)?;
        let act = dec.ew_gelu_host(&g, &u)?;
        let m = dec.gemv_f32_host(&format!("blk.{il}.ffn_down.weight"), &act)?;
        let h1 = llm170_core::ops::rms_norm(&m, cfg.nw_row(il, 3, h), hp.eps);
        let hn = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 7, h), hp.eps);
        let nm = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 4, h), hp.eps);
        let e_out = dec.g4_moe_host(il, &hn, &nm, cfg.pes_row(il, hp.n_experts))?;
        let h2 = llm170_core::ops::rms_norm(&e_out, cfg.nw_row(il, 5, h), hp.eps);
        let mut merged = h1;
        for i in 0..h {
            merged[i] += h2[i];
        }
        let hf = llm170_core::ops::rms_norm(&merged, cfg.nw_row(il, 6, h), hp.eps);
        let ls = cfg.layer_scalar[il];
        for i in 0..h {
            x[i] = (x[i] + hf[i]) * ls;
        }
        if dec.debug_layers {
            let mut mx = f32::NEG_INFINITY;
            let mut nan = false;
            for &v in x.iter() {
                if v.is_nan() {
                    nan = true;
                }
                mx = mx.max(v);
            }
            eprintln!("gemma4[gpu] layer {il:>2} max|x|={mx:.4} nan={nan}");
        }
    }
    let xf = llm170_core::ops::rms_norm(&x, cfg.out_norm_row(hp.n_layer, h), hp.eps);
    if dec.debug_layers {
        let mut mx = f32::NEG_INFINITY;
        let mut nan = false;
        for &v in xf.iter() {
            if v.is_nan() {
                nan = true;
            }
            mx = mx.max(v);
        }
        eprintln!("gemma4[gpu] final max|x|={mx:.4} nan={nan}");
    }
    let mut lg = dec.head_logits_host(&xf)?;
    if softcap && cfg.softcap > 0.0 {
        for l in lg.iter_mut() {
            *l = cfg.softcap * (*l / cfg.softcap).tanh();
        }
    }
    Ok(lg)
}
