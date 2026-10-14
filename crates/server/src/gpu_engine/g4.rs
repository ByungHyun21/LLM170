//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use super::upload_g4::G4HostCfg;

/// [gemma4 P3 v2 2026-10-14] t행 청크(≤8) — 프리필·검증·디코드 공용.
/// **전면 디바이스 상주**: 잔차 스트림 x·노름·gelu·잔차 가산·layer_scalar·
/// MoE·head 전부 디바이스(층당 호스트 왕복 0). 계약 출처 = 호스트 체인
/// (`g4_step`)과 동일 — 노름은 CPU 미러 비트(`rms_rows`), 잔차는
/// `w4a16_axpy(1.0)`(mul·add 분리 = 호스트 `x += h`와 비트동일), 스칼라는
/// `w4a16_scale`(1회 반올림 = 호스트 `(x+hf)*ls`).
/// 반환 = 행별 로짓(softcap 적용).
pub(crate) fn g4_chunk(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &G4HostCfg,
    slots: &[usize],
    pos: &[u32],
    rows: &[f32],
    t: usize,
    head_last: bool,
) -> Result<Vec<Vec<f32>>, String> {
    let hp = &model.hp;
    // 노름 행 계약 = 8행/층 + 최종 1행(upload_model_g4 조립 순서).
    let rows_per_layer = 8usize;
    let mut out = dec.g4_chunk_dev(slots, pos, rows, t, head_last, rows_per_layer, hp.eps)?;
    if cfg.softcap > 0.0 {
        for r in out.iter_mut() {
            for l in r.iter_mut() {
                *l = cfg.softcap * (*l / cfg.softcap).tanh();
            }
        }
    }
    Ok(out)
}

/// [gemma4 P2-3 2026-10-14] gemma4 v0 체인 1토큰 스텝(프로브·GemmaEngine 공용).
/// 층 산술(노름·잔차·layer_scalar)은 CPU 코어 계약(`llm170_core::ops`)으로
/// 조립하고, GPU가 담당하는 것은 GEMV(그룹 인식)·g4 어텐션·gelu·MoE(라우터
/// topk·pes·전문가)·head다. 계약 출처 = `crates/core/src/gemma4/`(CPU 오라클).
pub(crate) fn g4_step(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &crate::gpu_engine::G4HostCfg,
    slot: usize,
    tok: u32,
    pos: u32,
) -> Result<Vec<f32>, String> {
    let hp = &model.hp;
    let h = hp.n_embd;
    let mut x = model.embed_row(tok).map_err(|e| e.to_string())?;
    // 첫 풀층(층별 어텐션 도입부) 중간값 지문 — NaN/발산 위치 특정용.
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
        // ① 어텐션 — input_layernorm → q/k/(v) → prep_g4+f3s_g4 → post_attn+잔차.
        let xn = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 0, h), hp.eps);
        let q = dec.gemv_f32_host(&format!("blk.{il}.attn_q.weight"), &xn)?;
        let k = dec.gemv_f32_host(&format!("blk.{il}.attn_k.weight"), &xn)?;
        let v = if model.kv_eq_v(il) {
            k.clone() // v_norm(k 원시) — prep_g4의 vn_on이 적용(HF 계약)
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
        // o_proj — 헤드 결합(16×256=4096) → hidden(2816).
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
        // ② dense MLP — pre_ff → gate/up → gelu_tanh → down → post_ff1.
        let nd = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 2, h), hp.eps);
        let g = dec.gemv_f32_host(&format!("blk.{il}.ffn_gate.weight"), &nd)?;
        let u = dec.gemv_f32_host(&format!("blk.{il}.ffn_up.weight"), &nd)?;
        let act = dec.ew_gelu_host(&g, &u)?;
        let m = dec.gemv_f32_host(&format!("blk.{il}.ffn_down.weight"), &act)?;
        let h1 = llm170_core::ops::rms_norm(&m, cfg.nw_row(il, 3, h), hp.eps);
        // ③ MoE — 라우터 입력(hn)과 전문가 입력(nm)이 **별개** 노름(gemma4 계약).
        let hn = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 7, h), hp.eps);
        let nm = llm170_core::ops::rms_norm(&x, cfg.nw_row(il, 4, h), hp.eps);
        let e_out = dec.g4_moe_host(il, &hn, &nm, cfg.pes_row(il, hp.n_experts))?;
        let h2 = llm170_core::ops::rms_norm(&e_out, cfg.nw_row(il, 5, h), hp.eps);
        // ④ 병합 — post_ff_norm(h1+h2) → 잔차 → ×layer_scalar.
        let mut merged = h1;
        for i in 0..h {
            merged[i] += h2[i];
        }
        let hf = llm170_core::ops::rms_norm(&merged, cfg.nw_row(il, 6, h), hp.eps);
        let ls = cfg.layer_scalar[il];
        for i in 0..h {
            x[i] = (x[i] + hf[i]) * ls;
        }
        // 층별 지문 — CPU 오라클(debug_layers)과 같은 형식(최초 발산 층 추적).
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
    // ⑤ 최종 노름 + GPU head + softcap(호스트 — argmax 단조 불변).
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
    if cfg.softcap > 0.0 {
        for l in lg.iter_mut() {
            *l = cfg.softcap * (*l / cfg.softcap).tanh();
        }
    }
    Ok(lg)
}
