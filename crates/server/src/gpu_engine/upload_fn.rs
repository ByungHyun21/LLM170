
use llm170_backend_gpu::{AttnDims, AttnLayer, GdnDims, W4a16Dec};
use llm170_core::w4a16::W4a16Model;
use std::path::Path;

fn up(
    dec: &mut W4a16Dec,
    store: &W4a16Model,
    name: &str,
    tn: &str,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let b = store.tensor_slice(tn).ok_or(format!("{tn}: 부재"))?;
    dec.upload_plain(name, b, n, k)
}

#[allow(dead_code)]
pub(crate) fn upload_fn(
    dec: &mut W4a16Dec,
    cfg: &llm170_core::qwen4exp::Qwen4ExpCfg,
    dir: &Path,
    ctx: usize,
    stage_host: bool,
) -> Result<W4a16Model, String> {
    upload_fn_win(dec, cfg, dir, ctx, None, stage_host)
}

pub(crate) fn upload_fn_win(
    dec: &mut W4a16Dec,
    cfg: &llm170_core::qwen4exp::Qwen4ExpCfg,
    dir: &Path,
    ctx: usize,
    window: Option<(usize, usize)>,
    stage_host: bool,
) -> Result<W4a16Model, String> {
    if let Some((ws, we)) = window {
        dec.set_layer_window(ws, we)?;
    }
    let owns_final = window.is_none_or(|(_, we)| we == cfg.num_hidden_layers);
    let store = W4a16Model::open(dir).map_err(|e| format!("FN CT 스토어: {e}"))?;
    let lp = |il: usize| format!("model.language_model.layers.{il}");
    let (h, hh, lr) = (
        cfg.hidden_size,
        cfg.hc_count * cfg.hidden_size,
        cfg.hc_lowrank,
    );
    let v_len = cfg.linear_value_heads * cfg.linear_head_dim;
    let qkv_n = store
        .tensor_slice("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        .ok_or("qkv 폭 판독")?
        .len()
        / (2 * h);
    let h_k = (qkv_n - v_len) / (2 * cfg.linear_head_dim);
    for il in 0..cfg.num_hidden_layers {
        if !window.is_none_or(|(ws, we)| il >= ws && il < we) {
            continue;
        }
        let is_full = (il + 1) % cfg.full_interval.max(1) == 0;
        if is_full {
            up(
                dec,
                &store,
                &format!("blk.{il}.attn_q.weight"),
                &format!("{}.self_attn.q_proj.weight", lp(il)),
                cfg.heads * cfg.head_dim * 2,
                h,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.idx_qk.weight"),
                &format!("{}.self_attn.indexer.index_qk_proj.weight", lp(il)),
                (cfg.indexer_n_heads + 1) * cfg.indexer_head_dim,
                h,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.attn_k.weight"),
                &format!("{}.self_attn.k_proj.weight", lp(il)),
                cfg.kv_heads * cfg.head_dim,
                h,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.attn_v.weight"),
                &format!("{}.self_attn.v_proj.weight", lp(il)),
                cfg.kv_heads * cfg.head_dim,
                h,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.attn_o.weight"),
                &format!("{}.self_attn.o_proj.weight", lp(il)),
                h,
                cfg.heads * cfg.head_dim,
            )?;
        } else {
            up(
                dec,
                &store,
                &format!("blk.{il}.attn_qkv.weight"),
                &format!("{}.linear_attn.in_proj_qkv.weight", lp(il)),
                qkv_n,
                h,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.attn_gate.weight"),
                &format!("{}.linear_attn.in_proj_z.weight", lp(il)),
                v_len,
                h,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.ssm_out.weight"),
                &format!("{}.linear_attn.out_proj.weight", lp(il)),
                h,
                v_len,
            )?;
        }
        for (tag, hc) in [
            ("a", "attn_hyper_connection"),
            ("m", "mlp_hyper_connection"),
        ] {
            up(
                dec,
                &store,
                &format!("blk.{il}.hc_dn_{tag}.weight"),
                &format!("{}.{hc}.input_mix_weight_down.weight", lp(il)),
                lr,
                hh,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.hc_up_{tag}.weight"),
                &format!("{}.{hc}.input_mix_weight_up.weight", lp(il)),
                hh,
                lr,
            )?;
            up(
                dec,
                &store,
                &format!("blk.{il}.hc_inj_{tag}.weight"),
                &format!("{}.{hc}.block_inject_weight.weight", lp(il)),
                cfg.hc_count,
                hh,
            )?;
        }
        up(
            dec,
            &store,
            &format!("blk.{il}.moe_gate.weight"),
            &format!("{}.mlp.gate.weight", lp(il)),
            cfg.n_experts,
            h,
        )?;
        up(
            dec,
            &store,
            &format!("blk.{il}.moe_shared_gate.weight"),
            &format!("{}.mlp.shared_expert.gate_proj.weight", lp(il)),
            cfg.moe_ffn,
            h,
        )?;
        up(
            dec,
            &store,
            &format!("blk.{il}.moe_shared_up.weight"),
            &format!("{}.mlp.shared_expert.up_proj.weight", lp(il)),
            cfg.moe_ffn,
            h,
        )?;
        up(
            dec,
            &store,
            &format!("blk.{il}.moe_shared_down.weight"),
            &format!("{}.mlp.shared_expert.down_proj.weight", lp(il)),
            h,
            cfg.moe_ffn,
        )?;
        up(
            dec,
            &store,
            &format!("blk.{il}.moe_shared_sgate.weight"),
            &format!("{}.mlp.shared_expert_gate.weight", lp(il)),
            1,
            h,
        )?;
    }
    if owns_final {
        up(
            dec,
            &store,
            "blk.final_hc_dn.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            lr,
            hh,
        )?;
        up(
            dec,
            &store,
            "blk.final_hc_up.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
            hh,
            lr,
        )?;
        dec.upload_plain(
            "blk.final_hc_inj.weight",
            &vec![0u8; cfg.hc_count * hh * 2],
            cfg.hc_count,
            hh,
        )?;
    }
    if let Some(&p) = cfg.ple_layer_ids.first()
        && !window.is_none_or(|(ws, we)| p >= ws && p < we)
    {
    } else {
        let kb = store
            .tensor_slice("model.language_model.layers.1.ple.key_proj.weight")
            .ok_or("ple.key_proj 부재")?;
        let vb = store
            .tensor_slice("model.language_model.layers.1.ple.value_proj.weight")
            .ok_or("ple.value_proj 부재")?;
        let mut pk = Vec::with_capacity(kb.len() + vb.len());
        pk.extend_from_slice(kb);
        pk.extend_from_slice(vb);
        dec.upload_plain("blk.1.ple_kv.weight", &pk, hh + h, h)?;
    }
    if owns_final {
        up(
            dec,
            &store,
            "blk.head.weight",
            "lm_head.weight",
            cfg.vocab,
            h,
        )?;
    }
    dec.set_plain_mode(true);
    dec.set_moe(
        cfg.n_experts,
        cfg.top_k,
        cfg.moe_ffn,
        cfg.shared_ffn,
        32,
        false,
        false,
    );
    let mut tab: Vec<(u64, u64, u64, u64)> =
        Vec::with_capacity(cfg.num_hidden_layers * cfg.n_experts * 3);
    for il in 0..cfg.num_hidden_layers {
        if !window.is_none_or(|(ws, we)| il >= ws && il < we) {
            continue;
        }
        for e in 0..cfg.n_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let base = format!("{}.mlp.experts.{e}.{proj}", lp(il));
                let (q, s, _, _) = store.expert_at(&base).ok_or("전문가 트리플 부재")?;
                tab.push((
                    q.as_ptr() as u64,
                    q.len() as u64,
                    s.as_ptr() as u64,
                    s.len() as u64,
                ));
            }
        }
    }
    let mut marlin_only = false;
    if let Some((ws, we)) = window {
        let n_win = we - ws;
        let ne = cfg.n_experts;
        let mut lens_win: Vec<(usize, usize)> = Vec::with_capacity(n_win * ne * 3);
        let mut fills: Vec<(u64, u64)> = Vec::with_capacity(n_win * ne * 3);
        for il in ws..we {
            for e in 0..ne {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    let base = format!("{}.mlp.experts.{e}.{proj}", lp(il));
                    let (q, s, _, _) = store.expert_at(&base).ok_or("전문가 트리플 부재")?;
                    lens_win.push((q.len(), s.len()));
                    fills.push((q.as_ptr() as u64, s.as_ptr() as u64));
                }
            }
        }
        let mut fill = |i: usize, which: usize, off: u64, buf: &mut [u8]| -> Result<(), String> {
            let (qp, sp) = fills[i];
            let src = if which == 0 { qp } else { sp };
            let sl = unsafe { std::slice::from_raw_parts(src as *const u8, buf.len()) };
            let off = off as usize;
            if off + buf.len() > sl.len() {
                return Err(format!(
                    "marlin_only fill: i={i} off {off}+{} > {}",
                    buf.len(),
                    sl.len()
                ));
            }
            buf.copy_from_slice(&sl[off..off + buf.len()]);
            Ok(())
        };
        match dec.upload_experts_marlin_only_streamed(&lens_win, &mut fill, ws) {
            Ok(()) => {
                marlin_only = true;
                eprintln!("[fn-up] 마린-단독 상주: {n_win}층 × {ne}전문가(프리필 전용 계약)");
            }
            Err(e) => eprintln!("[fn-up] 마린-단독 실패({e}) — 스트리밍 폴백"),
        }
    }
    if !marlin_only {
        if let Some((ws, we)) = window {
            let mut full =
                vec![(0u64, 0u64, 0u64, 0u64); cfg.num_hidden_layers * cfg.n_experts * 3];
            let stride = cfg.n_experts * 3;
            full[ws * stride..we * stride].copy_from_slice(&tab);
            if stage_host {
                dec.set_expert_table(full);
            } else {
                dec.set_expert_table_nostage(full);
            }
        } else if stage_host {
            dec.set_expert_table(tab);
        } else {
            dec.set_expert_table_nostage(tab);
        }
    }
    dec.set_lru_tuning(Some(320), Some(1))?;
    let n_gdn = cfg.num_hidden_layers - cfg.num_hidden_layers / cfg.full_interval.max(1);
    let dims = GdnDims {
        n_gdn,
        hidden: h,
        h_k,
        h_v: cfg.linear_value_heads,
        d: cfg.linear_head_dim,
    };
    let (cch, hv) = (dims.conv_ch(), dims.h_v);
    let mut cw: Vec<f32> = Vec::with_capacity(n_gdn * cch * 4);
    let mut ab: Vec<f32> = Vec::with_capacity(n_gdn * 2 * hv * h);
    let mut alog: Vec<f32> = Vec::with_capacity(n_gdn * hv);
    let mut dtb: Vec<f32> = Vec::with_capacity(n_gdn * hv);
    let mut nw: Vec<f32> = Vec::with_capacity(n_gdn * 128);
    let f32_of = |b: &[u8]| -> Vec<f32> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect()
    };
    let ts = |nm: String| -> Result<Vec<u8>, String> {
        Ok(store
            .tensor_slice(&nm)
            .ok_or(format!("{nm} 부재"))?
            .to_vec())
    };
    for il in 0..cfg.num_hidden_layers {
        if (il + 1) % cfg.full_interval.max(1) == 0 {
            continue;
        }
        cw.extend(f32_of(&ts(format!(
            "{}.linear_attn.conv1d.weight",
            lp(il)
        ))?));
        ab.extend(f32_of(&ts(format!(
            "{}.linear_attn.in_proj_a.weight",
            lp(il)
        ))?));
        ab.extend(f32_of(&ts(format!(
            "{}.linear_attn.in_proj_b.weight",
            lp(il)
        ))?));
        alog.extend(f32_of(&ts(format!("{}.linear_attn.A_log", lp(il)))?));
        dtb.extend(f32_of(&ts(format!("{}.linear_attn.dt_bias", lp(il)))?));
        nw.extend(f32_of(&ts(format!("{}.linear_attn.norm.weight", lp(il)))?));
    }
    eprintln!(
        "[fn-up] GDN 상수: cw={} ab={} alog={} dtb={} nw={} (기대 cw={} ab={} alog={} dtb={} nw={})",
        cw.len(),
        ab.len(),
        alog.len(),
        dtb.len(),
        nw.len(),
        n_gdn * cch * 4,
        n_gdn * 2 * hv * h,
        n_gdn * hv,
        n_gdn * hv,
        n_gdn * 128
    );
    dec.set_gdn(dims, &cw, &ab, &alog, &dtb, &nw)?;
    let n_attn = cfg.num_hidden_layers / cfg.full_interval.max(1);
    let adims = AttnDims {
        q_heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        d: cfg.head_dim,
        interval: cfg.full_interval.max(1),
        cap: ctx,
        n_attn,
        sliding_window: 0,
    };
    let mut layers: Vec<AttnLayer> = Vec::with_capacity(n_attn);
    let mut qnw: Vec<f32> = Vec::with_capacity(n_attn * cfg.head_dim);
    let mut knw: Vec<f32> = Vec::with_capacity(n_attn * cfg.head_dim);
    for fi in 0..n_attn {
        let il = (fi + 1) * cfg.full_interval.max(1) - 1;
        layers.push(AttnLayer {
            kv_heads: cfg.kv_heads,
            d: cfg.head_dim,
            sliding: false,
            k_eq_v: false,
            kq_scale: 0.0,
        });
        for nm in ["q_norm", "k_norm"] {
            let mut v = f32_of(&ts(format!("{}.self_attn.{nm}.weight", lp(il)))?);
            for x in v.iter_mut() {
                *x += 1.0;
            }
            if nm == "q_norm" {
                qnw.extend(v)
            } else {
                knw.extend(v)
            }
        }
    }
    dec.set_attn(adims, layers, &qnw, &knw)?;
    let rows = 2 * cfg.num_hidden_layers + 1;
    let mut hcn: Vec<f32> = Vec::with_capacity(rows * hh);
    for il in 0..cfg.num_hidden_layers {
        for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
            let mut v = f32_of(&ts(format!("{}.{hc}.hc_norm.weight", lp(il)))?);
            for x in v.iter_mut() {
                *x += 1.0;
            }
            hcn.extend(v);
        }
    }
    {
        let mut v = f32_of(&ts(
            "model.language_model.hyper_connection_mixer.hc_norm.weight".to_string(),
        )?);
        for x in v.iter_mut() {
            *x += 1.0;
        }
        hcn.extend(v);
    }
    let ple_w: Vec<f32> = f32_of(&ts(
        "model.language_model.layers.1.ple.conv1d.weight".to_string()
    )?);
    let mut n1 = f32_of(&ts(
        "model.language_model.layers.1.ple.norm_key.weight".to_string()
    )?);
    let mut n2 = f32_of(&ts(
        "model.language_model.layers.1.ple.norm_query.weight".to_string()
    )?);
    let mut n3 = f32_of(&ts(
        "model.language_model.layers.1.ple.norm_conv.weight".to_string()
    )?);
    for x in n1.iter_mut() {
        *x += 1.0;
    }
    for x in n2.iter_mut() {
        *x += 1.0;
    }
    for x in n3.iter_mut() {
        *x += 1.0;
    }
    if ple_w.len() != hh * 4 || n1.len() != hh {
        return Err(format!(
            "upload_fn: PLE 형상({}·{}) 위반",
            ple_w.len(),
            n1.len()
        ));
    }
    dec.set_fn(
        cfg.hc_count,
        cfg.hc_lowrank,
        cfg.rms_norm_eps,
        cfg.num_hidden_layers,
        &hcn,
        Some((1, &ple_w, &n1, &n2, &n3)),
        true,
        cfg.indexer_n_heads,
    )?;
    Ok(store)
}
