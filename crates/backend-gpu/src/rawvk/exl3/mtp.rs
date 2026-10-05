//! EXL3 MTP 훅 (R4 — 순수 이동). mtp_step·스펙 검증(prefill_batch_spec·
//! frame_spec_forward·exl3_spec_step)·mtp2 — 드래프트/검증 하네스.
use super::cpu::SeqState;
use super::cpu::*;
use super::decode::*;
use super::resident::TrellisResident;
use super::wire::*;

/// MTP 1스텝: (token, h_in) → (logits, mtp_hidden). pos는 이 토큰의 위치.
pub(crate) fn mtp_step(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    token: u32,
    h_in: &[f32],
    pos: u32,
    with_logits: bool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let h = tr.hidden;
    let eps = 1e-6f32;
    let enorm = tr
        .norm("mtp.pre_fc_norm_embedding.weight")
        .ok_or("mtp enorm")?;
    let hnorm = tr
        .norm("mtp.pre_fc_norm_hidden.weight")
        .ok_or("mtp hnorm")?;
    let e = tr.embed_row(token);
    let e_n = rms_norm(e, enorm, eps);
    let h_n = rms_norm(h_in, hnorm, eps);
    let mut cat = Vec::with_capacity(2 * h);
    cat.extend_from_slice(&e_n);
    cat.extend_from_slice(&h_n);
    let mut cur = tr.linear("mtp.fc", &cat)?;

    // gated attention — input_layernorm → q/k/v → norm+rope → 자체 KV → o_proj
    let lp = "mtp.layers.0.self_attn";
    let attn_norm_w = tr
        .norm("mtp.layers.0.input_layernorm.weight")
        .ok_or("mtp attn_norm")?;
    let xn = rms_norm(&cur, attn_norm_w, eps);
    let (q_gate, k, v) = tr.linear_triple(
        &format!("{lp}.q_proj"),
        &format!("{lp}.k_proj"),
        &format!("{lp}.v_proj"),
        &xn,
    )?;
    let q_norm_w = tr
        .norm(&format!("{lp}.q_norm.weight"))
        .ok_or("mtp q_norm")?;
    let k_norm_w = tr
        .norm(&format!("{lp}.k_norm.weight"))
        .ok_or("mtp k_norm")?;
    let (n_head, n_kv, head_dim, n_rot) = (24usize, 4usize, 256usize, 64usize);
    let rope_base = 1e7f32;
    let mut q_heads = vec![0f32; n_head * head_dim];
    let mut gate_heads = vec![0f32; n_head * head_dim];
    {
        let kv = &mut seq.mtp_kv[0];
        let kv_cap = kv.k.len() / (n_kv * head_dim);
        if pos as usize >= kv_cap {
            return Err(format!("mtp_step: kv 용량 초과(pos {pos})"));
        }
        for hh in 0..n_head {
            let src = hh * head_dim * 2;
            q_heads[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_gate[src..src + head_dim]);
            gate_heads[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_gate[src + head_dim..src + head_dim * 2]);
        }
        for hh in 0..n_head {
            let b0 = hh * head_dim;
            let head: Vec<f32> = q_heads[b0..b0 + head_dim].to_vec();
            let n = rms_norm(&head, q_norm_w, eps);
            let mut h_rot = n;
            rope(&mut h_rot, pos, n_rot, rope_base);
            q_heads[b0..b0 + head_dim].copy_from_slice(&h_rot);
        }
        let k_base = pos as usize * n_kv * head_dim;
        for hh in 0..n_kv {
            let b0 = hh * head_dim;
            let head: Vec<f32> = k[b0..b0 + head_dim].to_vec();
            let n = rms_norm(&head, k_norm_w, eps);
            let mut h_rot = n;
            rope(&mut h_rot, pos, n_rot, rope_base);
            kv.k[k_base + hh * head_dim..k_base + (hh + 1) * head_dim].copy_from_slice(&h_rot);
            kv.v[k_base + hh * head_dim..k_base + (hh + 1) * head_dim]
                .copy_from_slice(&v[b0..b0 + head_dim]);
        }
        if kv.len < pos as usize + 1 {
            kv.len = pos as usize + 1;
        }
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_head / n_kv;
    let mut attn_out = vec![0f32; n_head * head_dim];
    {
        let kv = &seq.mtp_kv[0];
        let kv_len = (pos as usize + 1).min(kv.len);
        for hh in 0..n_head {
            let kv_h = hh / n_rep;
            let b0 = hh * head_dim;
            let mut scores = vec![0f32; kv_len];
            for (tt, sc) in scores.iter_mut().enumerate() {
                let kb = tt * n_kv * head_dim + kv_h * head_dim;
                let mut d = 0f32;
                for i in 0..head_dim {
                    d += q_heads[b0 + i] * kv.k[kb + i];
                }
                *sc = d * scale;
            }
            let maxv = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - maxv).exp();
                sum += *sc;
            }
            for tt in 0..kv_len {
                let w = scores[tt] / sum;
                let vb = tt * n_kv * head_dim + kv_h * head_dim;
                for i in 0..head_dim {
                    attn_out[b0 + i] += w * kv.v[vb + i];
                }
            }
            for i in 0..head_dim {
                attn_out[b0 + i] *= sigmoid(gate_heads[b0 + i]);
            }
        }
    }
    let o = tr.linear(&format!("{lp}.o_proj"), &attn_out)?;
    for i in 0..h {
        cur[i] += o[i];
    }

    // FFN — ffn_triple(GEMV 배치 3커널)
    let ffn_norm_w = tr
        .norm("mtp.layers.0.post_attention_layernorm.weight")
        .ok_or("mtp ffn_norm")?;
    let xf = rms_norm(&cur, ffn_norm_w, eps);
    let ffn_out = tr.ffn_triple(
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
        "mtp.layers.0.mlp.down_proj",
        &xf,
    )?;
    for i in 0..h {
        cur[i] += ffn_out[i];
    }

    if !with_logits {
        return Ok((Vec::new(), cur));
    }
    let sh_norm = tr.norm("mtp.norm.weight").ok_or("mtp shared norm")?;
    let hn = rms_norm(&cur, sh_norm, eps);
    let logits = tr.linear("lm_head", &hn)?;
    Ok((logits, cur))
}

/// \`llm170 exl3-mtp <dir> <token_ids> [n_predict]\` — MTP 드래프트 수용률·
/// 비용 실측(plans/121 A2 판정 근거): 타깃 순차 기준 생성과 교차 검증해
/// a1/a2(조건부) 수용률·mtp 스텝 벽시간을 보고한다.
pub fn exl3_mtp(dir: &str, tokens_str: &str, n_predict: usize) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    eprintln!("  [exl3-mtp] 상주 적재 중...");
    let t0 = std::time::Instant::now();
    let mut tr = TrellisResident::load(dir)?;
    eprintln!("  [exl3-mtp] 적재 완료 {:.1}s", t0.elapsed().as_secs_f64());
    let ctx_len = (prompt.len() + n_predict + 64).max(512);
    let mut seq = new_seq_state(tr.n_layers, ctx_len);

    // 프리필 — 타깃 순차 + mtp 훅(KV 적립·마지막 토큰만 로짓)
    let mut draft_logits = Vec::new();
    let mut mtp_hook_ms = 0f64;
    let t1 = std::time::Instant::now();
    for (i, &tok) in prompt.iter().enumerate() {
        let _ = decode_step(&mut tr, &mut seq, tok)?;
        let last = i + 1 == prompt.len();
        let t = std::time::Instant::now();
        let h_snap = seq.last_h.clone();
        let (dl, _) = mtp_step(&mut tr, &mut seq, tok, &h_snap, i as u32, last)?;
        mtp_hook_ms += t.elapsed().as_secs_f64() * 1e3;
        if last {
            draft_logits = dl;
        }
    }
    let pf_s = t1.elapsed().as_secs_f64();

    // 생성 — 매 스텝: d0(훅 드래프트) vs 타깃 greedy; d1 체인(조건부)
    let (mut a1n, mut a1d, mut a2n, mut a2d) = (0u64, 0u64, 0u64, 0u64);
    let (mut corr_acc, mut corr_n) = (0f64, 0u64);
    let (mut dec_ms, mut mtp2_ms) = (0f64, 0f64);
    let t2 = std::time::Instant::now();
    let mut out_tokens = Vec::new();
    let mut prev_d0: Option<u32> = None;
    let mut prev_d1: Option<u32> = None;
    for step in 0..n_predict {
        let t0_ = greedy_ref(&seq.last_logits);
        let d0 = argmax32(&draft_logits);
        if d0 == t0_ {
            a1n += 1;
        }
        a1d += 1;
        // k=2 체인 드래프트(이전 스텝 수용 조건부 측정)
        if prev_d0.is_some() && prev_d0 == Some(t0_) {
            // 직전 d0이 수용됨 — prev_d1 비교는 이전 루프에서 이미 예약됨
        }
        out_tokens.push(t0_);
        // 디버그: 드래프트↔타깃 로짓 상관(정렬/산술 판별)
        {
            let n = draft_logits.len().min(seq.last_logits.len());
            let (mut sxy, mut sx, mut sy, mut sxx, mut syy) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..n {
                let (a, b) = (draft_logits[i] as f64, seq.last_logits[i] as f64);
                sxy += a * b;
                sx += a;
                sy += b;
                sxx += a * a;
                syy += b * b;
            }
            let c = (n as f64 * sxy - sx * sy)
                / ((n as f64 * sxx - sx * sx) * (n as f64 * syy - sy * sy)).sqrt();
            corr_acc += c;
            corr_n += 1;
            if step < 3 {
                eprintln!(
                    "  [mtp-dbg] step{step}: d0={d0} t0={t0_} corr={c:.4} dl0={:.3} tl0={:.3}",
                    draft_logits[0], seq.last_logits[0]
                );
            }
        }
        if step + 1 < n_predict {
            let t = std::time::Instant::now();
            let _ = decode_step(&mut tr, &mut seq, t0_)?;
            dec_ms += t.elapsed().as_secs_f64() * 1e3;
            // 훅: 수용 토큰 KV 재기입 + 다음 드래프트 + (수용 시) d1 체인
            let t = std::time::Instant::now();
            let h_snap = seq.last_h.clone();
            let pos_snap = seq.pos - 1;
            let (dl, hm) = mtp_step(&mut tr, &mut seq, t0_, &h_snap, pos_snap, true)?;
            mtp2_ms += t.elapsed().as_secs_f64() * 1e3;
            draft_logits = dl;
            prev_d0 = Some(d0);
            prev_d1 = None;
            let hm_kept = hm;
            if d0 == t0_ && step + 2 < n_predict {
                // d1: 수용된 d0으로 체인 1스텝 더
                let pos_next = seq.pos;
                let (l1, _) = mtp_step(&mut tr, &mut seq, d0, &hm_kept, pos_next, true)?;
                prev_d1 = Some(argmax32(&l1));
            }
            // 다음 스텝에서 a2 판정: prev_d1 == 다음 타깃 greedy && prev_d0 수용
            if prev_d1.is_some() && prev_d0 == Some(d0) {
                // a2 판정 예약 — 실제 비교는 루프 끝 a2 블록에서.
            }
        }
        // 간이 a2 측정: 이전 스텝 d0 수용 && d1 존재 → 현재 t0_와 비교
        if let Some(pd1) = prev_d1.take() {
            a2d += 1;
            if pd1 == t0_ {
                a2n += 1;
            }
        }
    }
    let gen_s = t2.elapsed().as_secs_f64();

    let a1 = a1n as f64 / a1d.max(1) as f64;
    let a2 = a2n as f64 / a2d.max(1) as f64;
    let corr = corr_acc / corr_n.max(1) as f64;
    let dec_avg = dec_ms / n_predict.max(1) as f64;
    let mtp_avg = mtp2_ms / n_predict.max(1) as f64;
    let base_tps = n_predict as f64 / gen_s;
    Ok(format!(
        "exl3-mtp: 프롬프트 {}({:.1}s, mtp훅 {:.1}ms/tok) + {}스텝 | a1={:.3}({}/{}) a2={:.3}({}/{}) corr={:.4} | 타깃 {:.1}ms/스텝 mtp {:.1}ms/스텝 | 순차기준 {:.2} t/s",
        prompt.len(),
        pf_s,
        mtp_hook_ms / prompt.len().max(1) as f64,
        n_predict,
        a1,
        a1n,
        a1d,
        a2,
        a2n,
        a2d,
        corr,
        dec_avg,
        mtp_avg,
        base_tps,
    ))
}

/// 스펙 프리필 변형 — 행별 argmax 반환(검증용). 마지막 hidden/logits도
/// 갱신(prefill_batch와 동일 계약).
pub fn prefill_batch_spec(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    tokens: &[u32],
) -> Result<Vec<u32>, String> {
    let h = tr.hidden;
    let eps = 1e-6f32;
    let mut argmaxes = Vec::new();
    for chunk_toks in tokens.chunks(super::resident::BATCH_TMAX) {
        let t_rows = chunk_toks.len();
        let _e0 = ph("pp:embed");
        let mut x = vec![0f32; t_rows * h];
        for (t, &tok) in chunk_toks.iter().enumerate() {
            x[t * h..(t + 1) * h].copy_from_slice(tr.embed_row(tok));
        }
        drop(_e0);
        let stage = tr.stage_f32()?;
        let mut attn_count = 0;
        for il in 0..tr.n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let full = il % 4 == 3;
            let norm_w = tr
                .norm(&format!("{lp}.input_layernorm.weight"))
                .ok_or("norm missing")?
                .to_vec();
            let _n0 = ph("pp:norm_x");
            let mut xn = vec![0f32; t_rows * h];
            {
                let (xp, np, op, sp) = (
                    PP(x.as_ptr() as usize),
                    PP(norm_w.as_ptr() as usize),
                    PP(xn.as_mut_ptr() as usize),
                    PP(stage as usize),
                );
                par_rows(t_rows, move |t| unsafe {
                    let xr = std::slice::from_raw_parts((xp.0 as *const f32).add(t * h), h);
                    let ss: f32 = xr.iter().map(|&v| v * v).sum();
                    let inv = 1.0 / ((ss / h as f32 + eps).sqrt());
                    let ob = (op.0 as *mut f32).add(t * h);
                    let sb = (sp.0 as *mut f32).add(t * h);
                    let nw = np.0 as *const f32;
                    for i in 0..h {
                        let v = *xr.get_unchecked(i) * inv * *nw.add(i);
                        *ob.add(i) = v;
                        *sb.add(i) = v;
                    }
                });
            }
            drop(_n0);
            let attn_out = if full {
                let _g = ph("pp:attn");
                let r = attn_batch(tr, seq, il, attn_count, t_rows, 24, 4, 256)?;
                drop(_g);
                attn_count += 1;
                r
            } else {
                let _g = ph("pp:gdn");
                let r = gdn_batch(tr, seq, il, &xn, t_rows)?;
                drop(_g);
                r
            };
            let _r0 = ph("pp:resid");
            for (a, b) in x.iter_mut().zip(attn_out.iter()) {
                *a += b;
            }
            drop(_r0);
            let ffn_norm_w = tr
                .norm(&format!("{lp}.post_attention_layernorm.weight"))
                .ok_or("ffn norm missing")?
                .to_vec();
            let _n1 = ph("pp:norm_f");
            {
                let (xp, np, sp) = (
                    PP(x.as_ptr() as usize),
                    PP(ffn_norm_w.as_ptr() as usize),
                    PP(stage as usize),
                );
                par_rows(t_rows, move |t| unsafe {
                    let xr = std::slice::from_raw_parts((xp.0 as *const f32).add(t * h), h);
                    let ss: f32 = xr.iter().map(|&v| v * v).sum();
                    let inv = 1.0 / ((ss / h as f32 + eps).sqrt());
                    let sb = (sp.0 as *mut f32).add(t * h);
                    let nw = np.0 as *const f32;
                    for i in 0..h {
                        *sb.add(i) = *xr.get_unchecked(i) * inv * *nw.add(i);
                    }
                });
            }
            drop(_n1);
            let _gf = ph("pp:ffn");
            let ffn_out = tr.ffn_trio_batch(
                &format!("{lp}.mlp.gate_proj"),
                &format!("{lp}.mlp.up_proj"),
                &format!("{lp}.mlp.down_proj"),
                t_rows,
            )?;
            drop(_gf);
            let _r1 = ph("pp:resid2");
            for (a, b) in x.iter_mut().zip(ffn_out.iter()) {
                *a += b;
            }
            drop(_r1);
        }
        seq.pos += t_rows as u32;
        seq.last_h.clear();
        seq.last_h
            .extend_from_slice(&x[(t_rows - 1) * h..t_rows * h]);
        // 행별 head — output_norm 행별 후 lm_head 배치.
        let out_norm_w = tr
            .norm("model.language_model.norm.weight")
            .ok_or("output norm missing")?
            .to_vec();
        {
            let (xp, np, sp) = (
                PP(x.as_ptr() as usize),
                PP(out_norm_w.as_ptr() as usize),
                PP(stage as usize),
            );
            par_rows(t_rows, move |t| unsafe {
                let xr = std::slice::from_raw_parts((xp.0 as *const f32).add(t * h), h);
                let ss: f32 = xr.iter().map(|&v| v * v).sum();
                let inv = 1.0 / ((ss / h as f32 + eps).sqrt());
                let sb = (sp.0 as *mut f32).add(t * h);
                let nw = np.0 as *const f32;
                for i in 0..h {
                    *sb.add(i) = *xr.get_unchecked(i) * inv * *nw.add(i);
                }
            });
        }
        let ys = tr.linear_batch_multi_staged(&["lm_head"], t_rows)?;
        let vocab = ys[0].len() / t_rows;
        for t in 0..t_rows {
            argmaxes.push(argmax32(&ys[0][t * vocab..(t + 1) * vocab]));
        }
        seq.last_logits.clear();
        let last_row = t_rows - 1;
        seq.last_logits
            .extend_from_slice(&ys[0][last_row * vocab..(last_row + 1) * vocab]);
    }
    Ok(argmaxes)
}

/// 스펙 1라운드 doc(위) — 프레임 검증 forward가 실체 대체(plans/121 tg).
///
/// 프레임 스펙 검증 forward(plans/121 tg 경로): toks(k행)을 원-서브밋 프레임으로
/// 처리하고 행별 lm_head 로짓을 반환. 상태는 GPU 권위 그대로(동기 없음).
/// seq 갱신: pos/last_tok/last_h/last_logits/kv.len(=pos 설정 의미).
pub(super) fn frame_spec_forward(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    toks: &[u32],
) -> Result<Vec<f32>, String> {
    let t_rows = toks.len();
    let h = 5120usize;
    let pos0 = seq.pos;
    tr.fframe_init()?;
    tr.gdn_frame_init()?;
    let xp = tr.frame_x_ptr()?;
    for (t, &tok) in toks.iter().enumerate() {
        // SAFETY: xbuf 프레임 소유 — 행별 분리 기록.
        unsafe {
            let src: &[f32] = tr.embed_row(tok);
            std::ptr::copy_nonoverlapping(src.as_ptr(), xp.add(t * h), h);
        }
    }
    tr.frame_x_flush(t_rows)?;
    // 재생 통합(plans/121 tg): 녹화됨 → 호스트 입력(xbuf/pbuf)만 갱신해 재제출.
    // 미녹화 → 이번 실행으로 녹화(기록 비용 최초 1회).
    let fkey = t_rows as u32;
    // 재생 옵트인(plans/121 tg): 라운드2+ 재생에서 NaN 발생(원장 계류) —
    // 정합 경로 보호를 위해 LLM170_EXL3_REPLAY=1까지만 활성.
    let replaying = false; // 재생 NaN 미결 — ENV 계약으로 경로 삭제(녹화만 유지)
    if replaying {
        tr.attn_set_pos(pos0)?;
        tr.ctx.frame_replay(fkey)?;
        if llm170_diag::dump::opts().key("exl3_rpdbg") {
            let xh = tr.read_xtb_rows(1)?;
            eprintln!("  [rpdbg] pos0={pos0} xn0={:?}", &xh[..4]);
        }
        let xn_all = tr.read_xtb_rows(t_rows)?;
        let mut logits = Vec::with_capacity(t_rows * 248320);
        for t in 0..t_rows {
            let row = tr.linear("lm_head", &xn_all[t * h..(t + 1) * h])?;
            logits.extend_from_slice(&row);
        }
        seq.pos += t_rows as u32;
        seq.last_tok = *toks.last().ok_or("빈 스펙 입력")?;
        seq.last_h = tr.frame_read_x_last(t_rows)?;
        let last_row = &logits[(t_rows - 1) * 248320..t_rows * 248320];
        seq.last_logits.clear();
        seq.last_logits.extend_from_slice(last_row);
        return Ok(logits);
    }
    let recording = llm170_diag::flag::eq1("LLM170_EXL3_REPLAY");
    let vf_dbg = llm170_diag::dump::opts().key("exl3_specphase");
    let vf_t0 = std::time::Instant::now();
    if recording {
        tr.ctx.frame_record_begin(fkey)?;
    }
    tr.ctx.begin_outer()?;
    let zb = tr.frame_zeros_buf()?;
    // 메가융합 1호: L0 qkv/z had 융합(비트 검증 nrh-check).
    {
        let (s1, s2) = (
            tr.suh_of("model.language_model.layers.0.linear_attn.in_proj_qkv")?,
            tr.suh_of("model.language_model.layers.0.linear_attn.in_proj_z")?,
        );
        tr.frame_norm_resid_had(0, t_rows, zb, s1, s2)?;
    }
    for il in 0..tr.n_layers {
        let lp = format!("model.language_model.layers.{il}");
        let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
        let ab = if il % 4 == 3 {
            let ai = il / 4;
            let yq = tr.linear_chain(&format!("{lp}.self_attn.q_proj"), t_rows, 0)?;
            let yk = tr.linear_chain(&format!("{lp}.self_attn.k_proj"), t_rows, 1)?;
            let yv = tr.linear_chain(&format!("{lp}.self_attn.v_proj"), t_rows, 2)?;
            tr.attn_layer_gpu(ai, t_rows, pos0, yq.0, yk.0, yv.0)?;
            seq.kv[ai].len = (pos0 + t_rows as u32) as usize;
            tr.linear_chain(&format!("{lp}.self_attn.o_proj"), t_rows, 0)?
                .0
        } else {
            {
                let g = &seq.gdn[il];
                tr.gdn_state_upload(gdn_il, &g.states, &g.conv)?;
            }
            // 듀얼 gemm2d(메가융합 3호) — 비트검증 gemmd-check.
            let dslots = tr.linear_pair_dual(
                [
                    &format!("{lp}.linear_attn.in_proj_qkv"),
                    &format!("{lp}.linear_attn.in_proj_z"),
                ],
                t_rows,
            )?;
            let (yq, yz) = (dslots[0], dslots[1]);
            tr.gdn_layer_gpu(gdn_il, t_rows, std::ptr::null_mut(), yq.0, yz.0)?;
            tr.linear_chain(&format!("{lp}.linear_attn.out_proj"), t_rows, 0)?
                .0
        };
        // 메가융합 2호: post_ln에 gate/up had 융합.
        {
            let (sg, su) = (
                tr.suh_of(&format!("{lp}.mlp.gate_proj"))?,
                tr.suh_of(&format!("{lp}.mlp.up_proj"))?,
            );
            tr.frame_norm_resid_had(2 * il + 1, t_rows, ab, sg, su)?;
        }
        let yf = tr.ffn_trio_preah(
            &format!("{lp}.mlp.gate_proj"),
            &format!("{lp}.mlp.up_proj"),
            &format!("{lp}.mlp.down_proj"),
            t_rows,
        )?;
        let w_next = if il + 1 == tr.n_layers {
            128
        } else {
            2 * (il + 1)
        };
        // 다음 층이 GDN이면 qkv/z had 융합(마지막 층 제외).
        if il + 1 < tr.n_layers && (il + 1) % 4 != 3 {
            let lp2 = format!("model.language_model.layers.{}", il + 1);
            let (s1, s2) = (
                tr.suh_of(&format!("{lp2}.linear_attn.in_proj_qkv"))?,
                tr.suh_of(&format!("{lp2}.linear_attn.in_proj_z"))?,
            );
            tr.frame_norm_resid_had(w_next, t_rows, yf.0, s1, s2)?;
        } else {
            tr.frame_norm_resid(w_next, t_rows, yf.0)?;
        }
    }
    // 프레임 종료 후 행별 lm_head GEMV — xtb 행은 마지막 norm_resid가 이미
    // output_norm(행 128) 적용: 추가 rms_norm 금지(이중 노름 버그 — 삼각
    // 비교 적발: decode/prefillT1=198 vs spec=1195).
    tr.ctx.end_outer()?;
    let vf_t1 = std::time::Instant::now(); // 층 루프+기록 완료
    if vf_dbg {
        eprintln!(
            "[vphase] record+loop={:.0}ms",
            (vf_t1 - vf_t0).as_secs_f64() * 1e3
        );
    }
    if recording {
        tr.ctx.frame_record_end(fkey)?;
        // 녹화 패스는 제출 없이 종료됐다(record_only) — 여기서 1회 재생해
        // 이번 라운드의 실행으로 삼는다(설계 결함 수정 2026-10-03:
        // 미실행 녹화의 스테일 xtb 판독이 재생 NaN의 근원).
        tr.ctx.frame_replay(fkey)?;
    }
    let xn_all = tr.read_xtb_rows(t_rows)?;
    let mut logits = Vec::with_capacity(t_rows * 248320);
    for t in 0..t_rows {
        let row = tr.linear("lm_head", &xn_all[t * h..(t + 1) * h])?;
        logits.extend_from_slice(&row);
    }
    // seq 갱신
    seq.pos += t_rows as u32;
    seq.last_tok = *toks.last().ok_or("빈 스펙 입력")?;
    seq.last_h = tr.frame_read_x_last(t_rows)?;
    let last_row = &logits[(t_rows - 1) * 248320..t_rows * 248320];
    seq.last_logits.clear();
    seq.last_logits.extend_from_slice(last_row);
    Ok(logits)
}

pub fn exl3_spec_step(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    k: usize,
) -> Result<(Vec<u32>, usize), String> {
    let _sp_ph = llm170_diag::dump::opts().key("exl3_specphase");
    let _sp_t0 = std::time::Instant::now();
    let k = k.clamp(1, 4);
    // ① 드래프트 체인 — 시드 (last_tok, last_h), mtp KV 슬롯 = pos+i.
    let mut drafts = Vec::with_capacity(k);
    let mut tok = seq.last_tok;
    let mut h = seq.last_h.clone();
    for i in 0..k {
        let pos = seq.pos + i as u32;
        let (lgt, hm) = mtp_step(tr, seq, tok, &h, pos, true)?;
        drafts.push(argmax32(&lgt));
        tok = drafts[i];
        h = hm;
    }
    let _sp_t1 = std::time::Instant::now(); // 드래프트 완료
    // ② 검증 기준 g0 + GPU 상태 스냅샷(프레임 경로 — plans/121 tg).
    let g0 = argmax32(&seq.last_logits);
    tr.fframe_init()?;
    tr.gdn_frame_init()?;
    tr.attn_frame_init()?;
    // ③ 검증 — 기본 프레임 / A/B: CPU 검증(LLM170_EXL3_SPECCPU=1, 진단).
    let _sp_t2 = std::time::Instant::now(); // 스냅샷 완료
    let row_am: Vec<u32> = if false && llm170_diag::flag::eq1("LLM170_EXL3_SPECCPU") {
        let snap2 = spec_snap(seq);
        let mut row = Vec::with_capacity(k);
        for &d in &drafts {
            let lg = decode_step(tr, seq, d)?;
            row.push(argmax32(&lg));
        }
        spec_restore(seq, &snap2);
        row
    } else {
        // A/B 진단(2026-10-03): 스냅샷 생략 — copy_dev가 verify 오염시키는지 판정.
        if !llm170_diag::flag::eq1("LLM170_EXL3_NOSNAP") {
            tr.gdn_state_snapshot()?;
        }
        let spec_logits = frame_spec_forward(tr, seq, &drafts)?;
        (0..k)
            .map(|i| argmax32(&spec_logits[i * 248320..(i + 1) * 248320]))
            .collect()
    };
    // ④ 수용 보행.
    let mut accepted: Vec<u32> = Vec::with_capacity(k + 1);
    let mut diverged = false;
    for i in 0..k {
        let gi = if i == 0 { g0 } else { row_am[i - 1] };
        if drafts[i] == gi {
            accepted.push(drafts[i]);
        } else {
            accepted.push(gi);
            diverged = true;
            break;
        }
    }
    let _sp_t3 = std::time::Instant::now(); // 검증 완료
    if _sp_ph {
        let (_d1, _d2, _d3) = (
            (_sp_t1 - _sp_t0).as_secs_f64() * 1e3,
            (_sp_t2 - _sp_t1).as_secs_f64() * 1e3,
            (_sp_t3 - _sp_t2).as_secs_f64() * 1e3,
        );
        eprintln!("[specphase] draft={_d1:.0}ms snap={_d2:.0}ms verify={_d3:.0}ms");
    }
    if llm170_diag::dump::opts().key("exl3_specdbg") {
        eprintln!(
            "  [specdbg] pos={} drafts={drafts:?} row_am={row_am:?} g0={g0} div={diverged} acc={accepted:?}",
            seq.pos
        );
    }
    let forwards = if diverged {
        // ⑤ 롤백 + 수용 접두 재실행 — 프레임 기본 / CPU A/B.
        if false && llm170_diag::flag::eq1("LLM170_EXL3_SPECCPU") {
            let re_accepted = accepted.clone();
            for &t in &re_accepted {
                let _ = decode_step(tr, seq, t)?;
            }
        } else {
            // kvc는 재실행이 정확히 pos0.. 행을 덮으므로 복원 불요.
            if !llm170_diag::flag::eq1("LLM170_EXL3_NOSNAP") {
                tr.gdn_state_restore()?;
                if llm170_diag::dump::opts().key("exl3_rpdbg") {
                    eprintln!("  [rpdbg] restore후 gstate={:?}", tr.debug_gstate_head()?);
                }
            }
            seq.pos -= k as u32; // frame_spec_forward가 다시 증가
            let re_accepted = accepted.clone();
            frame_spec_forward(tr, seq, &re_accepted)?;
        }
        if llm170_diag::dump::opts().key("exl3_rpdbg") {
            eprintln!("  [rpdbg] 재실행후 gstate={:?}", tr.debug_gstate_head()?);
        }
        2
    } else {
        accepted.push(row_am[k - 1]); // 전 수용 — 선행 보너스 토큰
        1
    };
    // ⑥ MTP 훅 — 마지막 수용 토큰만 (본체 h) 재기입 + 다음 드래프트 로짓.
    //    중간 슬롯은 드래프트 시점 기입 (tok==수용 토큰) 상태로 둔다 —
    //    k/v는 (토큰, mtp_h) 쌍이고 수용 토큰이 같으므로 근사 오차는
    //    다음 라운드 검증이 흡수한다(원장 128 수용률 수준과 동일 논리).
    {
        // 훅 입력은 '마지막 처리 토큰' — 전수용 경로의 accepted 마지막은
        // 선행 보너스(미처리)일 수 있어 last_tok을 쓴다(슬롯=seq.pos).
        let h_last = seq.last_h.clone();
        let _ = mtp_step(tr, seq, seq.last_tok, &h_last, seq.pos, true)?;
    }
    Ok((accepted, forwards))
}

/// `llm170 exl3-mtp2 <dir> <token_ids> [n_predict] [k]` — 스펙 라운드 실측:
/// 유효 t/s(스냅샷·롤백·훅 비용 포함) + 수용률 + forward 수. 그리디 기준
/// 토큰열과의 동일성도 보고(스펙 경로 정합 게이트).
pub fn exl3_mtp2(
    dir: &str,
    tokens_str: &str,
    n_predict: usize,
    k: usize,
) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    eprintln!("  [exl3-mtp2] 상주 적재 중...");
    let t0 = std::time::Instant::now();
    let mut tr = TrellisResident::load(dir)?;
    eprintln!("  [exl3-mtp2] 적재 완료 {:.1}s", t0.elapsed().as_secs_f64());
    let ctx_len = (prompt.len() + n_predict + 64).max(512);
    let mut seq = new_seq_state(tr.n_layers, ctx_len);

    // 프리필(프레임) — GPU kvc/gstate 적립 + 벌크 동기(스펙 시드 정합).
    let am = prefill_batch(&mut tr, &mut seq, &prompt)?;
    {
        let h_last = seq.last_h.clone();
        let pos = seq.pos - 1;
        let tok = seq.last_tok;
        let _ = mtp_step(&mut tr, &mut seq, tok, &h_last, pos, true)?;
        let _ = am;
    }

    // 삼각 비교 진단(plans/121 tg): 동일 토큰(=plain[0] 예상 전이)을
    // ① decode_step ② prefill_batch(T=1) ③ frame_spec_forward 로 각각 처리해
    // argmax 대조 — verify 오염의 소속(코드 diff vs 상태 반입)을 가른다.
    if llm170_diag::dump::opts().key("exl3_tri") {
        let snap3 = spec_snap(&seq);
        let t_probe = argmax32(&seq.last_logits); // = plain[0] 후보
        let lg1 = {
            let l = decode_step(&mut tr, &mut seq, t_probe)?;
            spec_restore(&mut seq, &snap3);
            l
        };
        let lg2 = {
            let l = prefill_batch(&mut tr, &mut seq, &[t_probe])?;
            spec_restore(&mut seq, &snap3);
            l
        };
        let lg3 = {
            let l = frame_spec_forward(&mut tr, &mut seq, &[t_probe])?;
            spec_restore(&mut seq, &snap3);
            l
        };
        eprintln!(
            "  [tri] tok={t_probe} decode={} prefillT1={} specFWD={}",
            argmax32(&lg1),
            argmax32(&lg2),
            argmax32(&lg3)
        );
    }
    // 스펙 루프
    let t2 = std::time::Instant::now();
    let (mut toks, mut fwds, mut rounds) = (0u64, 0u64, 0u64);
    let mut spec_tokens = Vec::new();
    while spec_tokens.len() < n_predict {
        let (acc, fw) = exl3_spec_step(&mut tr, &mut seq, k)?;
        toks += acc.len() as u64;
        fwds += fw as u64;
        rounds += 1;
        for t in acc {
            if spec_tokens.len() < n_predict {
                spec_tokens.push(t);
            }
        }
    }
    let spec_s = t2.elapsed().as_secs_f64();
    // 진단 덤프 — exl3_phase 위상 분해(exl3_pp와 동일 원장 89 키).
    phase_report();
    let tps = toks as f64 / spec_s;

    // 정합 — 같은 상태에서 순차 greedy 재현(토큰 동일성 게이트).
    let mut seq2 = new_seq_state(tr.n_layers, ctx_len);
    let mut logits = prefill_batch(&mut tr, &mut seq2, &prompt)?;
    let mut plain_tokens = Vec::new();
    for _ in 0..n_predict {
        let t = argmax32(&logits);
        plain_tokens.push(t);
        logits = decode_step(&mut tr, &mut seq2, t)?;
    }
    let same = spec_tokens == plain_tokens;
    let per_round = toks as f64 / rounds as f64;
    Ok(format!(
        "exl3-mtp2(k={k}): {}라운드 {}토큰(평균 {per_round:.2}/라운드) {}forward | 유효 {:.2} t/s | 토큰 정합 {} ({}/{} 일치)
  spec {:?}
  plain {:?}
  초 forward {:.0}ms/forward",
        rounds,
        toks,
        fwds,
        tps,
        if same { "통과" } else { "불일치" },
        spec_tokens
            .iter()
            .zip(plain_tokens.iter())
            .filter(|(a, b)| a == b)
            .count(),
        n_predict,
        &spec_tokens[..12.min(spec_tokens.len())],
        &plain_tokens[..12.min(plain_tokens.len())],
        (spec_s / fwds.max(1) as f64) * 1e3,
    ))
}
// 마커 r1diff
// 마커 dblnorm
// 마커 ls2
// 마커 att1
// 마커 specatt
// 마커 mf2
// 마커 dualatt
// 마커 fb1
// 마커 tadj
// 마커 cb1
// 마커 dcl2
