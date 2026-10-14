//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use super::upload_g4::G4HostCfg;

/// [gemma4 P3 2026-10-14] t행 청크(≤8) — 프리필·검증 스텝.
/// 하이브리드: 노름·gelu·잔차·layer_scalar는 호스트(f32 = CPU 오라클 계약),
/// GEMM(t행)·어텐션(g4 호스트 경로 — 토큰별)·MoE(t행 디바이스)·head(t행 TR)는
/// GPU(전부 기존 검증 커널). KV = t=1 경로와 동일 슬롯·연속위치.
/// 반환 = 행별 로짓(t개, softcap 적용).
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
    let h = hp.n_embd;
    if t == 0 || t > 8 || rows.len() != t * h || slots.len() != t || pos.len() != t {
        return Err(format!(
            "g4_chunk: t={t} rows={} slots={} pos={} (1..=8 · t*h 계약)",
            rows.len(),
            slots.len(),
            pos.len()
        ));
    }
    let mut x: Vec<f32> = rows.to_vec();
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        // ① input_layernorm → t행 q/k/(v) GEMM.
        let xn = g4_norm_rows(&x, cfg.nw_row(il, 0, h), hp.eps, t);
        // ② q/k/v GEMM + 어텐션 + o_proj 전부 **디바이스**(중간 d2h 0 —
        // 토큰별 호스트 경로의 왕복을 청크당 3런치로). KV 누적 = t=1과 동일.
        let sw = if hp.full[il] { 0 } else { hp.sw };
        let qd = hp.n_head * d;
        let dxn = dec.g4_dev_in(0, &xn)?;
        let q = dec.g4_dev_gemm(&format!("blk.{il}.attn_q.weight"), dxn, t, h, 1)?;
        let k = dec.g4_dev_gemm(&format!("blk.{il}.attn_k.weight"), dxn, t, h, 2)?;
        let v = if model.kv_eq_v(il) {
            k // v_norm(k 원시) — prep_g4 vn_on 계약
        } else {
            dec.g4_dev_gemm(&format!("blk.{il}.attn_v.weight"), dxn, t, h, 3)?
        };
        let outv = dec.attn_g4_dev(
            slots,
            pos,
            il,
            q,
            k,
            v,
            hp.n_head,
            hp.n_kv[il],
            d,
            sw,
            1.0,
            2,
            6,
            7,
            8,
        )?;
        // ③ o_proj(t행 — 디바이스) → d2h → post_attn_norm + 잔차.
        let o = dec.g4_dev_gemm(&format!("blk.{il}.attn_output.weight"), outv, t, qd, 9)?;
        let _ = o;
        let aop = dec.g4_dev_out(9, t * h)?;
        let hh = g4_norm_rows(&aop, cfg.nw_row(il, 1, h), hp.eps, t);
        for i in 0..t * h {
            x[i] += hh[i];
        }
        // ④ dense MLP — 디바이스 체인(pre_ff → gate/up GEMM → ew_gelu 커널 →
        // down GEMM) + d2h 1회. gelu = t=1 경로와 동일 커널 심볼(수치 정합).
        let nd = g4_norm_rows(&x, cfg.nw_row(il, 2, h), hp.eps, t);
        let dnd = dec.g4_dev_in(0, &nd)?;
        let g = dec.g4_dev_gemm(&format!("blk.{il}.ffn_gate.weight"), dnd, t, h, 1)?;
        let u = dec.g4_dev_gemm(&format!("blk.{il}.ffn_up.weight"), dnd, t, h, 2)?;
        let act = dec.g4_dev_gelu(g, u, 3, t * hp.n_ff)?;
        let _ = act;
        let dm_ = dec.g4_dev_gemm(&format!("blk.{il}.ffn_down.weight"), act, t, hp.n_ff, 4)?;
        let _ = dm_;
        let m = dec.g4_dev_out(4, t * h)?;
        let h1 = g4_norm_rows(&m, cfg.nw_row(il, 3, h), hp.eps, t);
        // ⑤ MoE — 라우터 입력(hn)·전문가 입력(nm) 분리 · pes 디바이스.
        let hn = g4_norm_rows(&x, cfg.nw_row(il, 7, h), hp.eps, t);
        let nm = g4_norm_rows(&x, cfg.nw_row(il, 4, h), hp.eps, t);
        let pes_dev = dec.g4_pes_dev(il);
        if pes_dev == 0 {
            return Err(format!("g4_chunk: L{il} pes 미등록(set_g4_pes 선행)"));
        }
        let e_out = dec.g4_moe_host_t(il, &hn, &nm, cfg.pes_row(il, hp.n_experts), t, pes_dev)?;
        let h2 = g4_norm_rows(&e_out, cfg.nw_row(il, 5, h), hp.eps, t);
        // ⑥ 병합 → post_ff_norm → 잔차 → layer_scalar.
        let mut merged = h1;
        for i in 0..t * h {
            merged[i] += h2[i];
        }
        let hf = g4_norm_rows(&merged, cfg.nw_row(il, 6, h), hp.eps, t);
        let ls = cfg.layer_scalar[il];
        for i in 0..t * h {
            x[i] = (x[i] + hf[i]) * ls;
        }
    }
    // ⑦ 최종 노름 + head + softcap. head_last(프리필) = **마지막 행만**
    // (nsys: t행 head가 청크당 1.6ms = 7.7% — 프리필은 마지막 로짓만 소비).
    if head_last {
        let last = &x[(t - 1) * h..t * h];
        let xf = llm170_core::ops::rms_norm(last, cfg.out_norm_row(hp.n_layer, h), hp.eps);
        let mut lg = dec.g4_head_host_t(&xf, 1)?;
        if cfg.softcap > 0.0 {
            for l in lg.iter_mut() {
                *l = cfg.softcap * (*l / cfg.softcap).tanh();
            }
        }
        return Ok(vec![lg]);
    }
    let xf = g4_norm_rows(&x, cfg.out_norm_row(hp.n_layer, h), hp.eps, t);
    let flat = dec.g4_head_host_t(&xf, t)?;
    let n_vocab = hp.vocab;
    let mut out = Vec::with_capacity(t);
    for kk in 0..t {
        let mut lg = flat[kk * n_vocab..(kk + 1) * n_vocab].to_vec();
        if cfg.softcap > 0.0 {
            for l in lg.iter_mut() {
                *l = cfg.softcap * (*l / cfg.softcap).tanh();
            }
        }
        out.push(lg);
    }
    Ok(out)
}

/// [t][h] 행렬 행별 가중 RMS — CPU 계약(ops::rms_norm)과 동일 함수.
fn g4_norm_rows(x: &[f32], w: &[f32], eps: f32, t: usize) -> Vec<f32> {
    let h = w.len();
    let mut out = Vec::with_capacity(t * h);
    for r in 0..t {
        out.extend_from_slice(&llm170_core::ops::rms_norm(&x[r * h..(r + 1) * h], w, eps));
    }
    out
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
