//! EXL3 레이어 스트리밍 단일 토큰 디코드 (§3-2) — 전방향 실구현.
//! TrellisResident 위에 qwen35 전방향: 선형=vk GEMV, 비선형=CPU.
//!
//! 기존 core의 gdn_ar_batch·gdn_norm_gated·rope_head 함수를 재사용 —
//! 비선형 로직을 재발명하지 않고 검증된 구현을 호출한다.

use super::cpu::*;
use super::wire::*;

use super::resident::TrellisResident;

// ── 위상 프로파일러 (LLM170_DUMP=exl3_phase, 원장 89: dump 키로만) ──
// profile_span은 release no-op이라 exl3 프로브 전용 경량 계측.
// 단일 스레드 오케스트레이션 전제(thread_local) — gdn_ar_batch 내부 스레드는 미계측.

pub fn decode_step(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    token: u32,
) -> Result<Vec<f32>, String> {
    let h = tr.hidden;
    let eps = 1e-6f32;

    // 임베딩
    let mut x = tr.embed_row(token).to_vec();
    if seq.pos == 0 {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
        eprintln!("  [dbg] embed rms={rms:.4}");
        // 첫 층 norm 가중치 확인
        if let Some(nw) = tr.norm("model.language_model.layers.0.input_layernorm.weight") {
            eprintln!("  [dbg] attn_norm[0..3] = {:?}", &nw[..3.min(nw.len())]);
        }
    }

    let mut attn_count = 0;
    for il in 0..tr.n_layers {
        let lp = format!("model.language_model.layers.{il}");
        let full = il % 4 == 3;

        // attn_norm
        let norm_w = tr
            .norm(&format!("{lp}.input_layernorm.weight"))
            .ok_or("norm missing")?;
        let xn = rms_norm(&x, norm_w, eps);
        if std::env::var_os("LLM170_EXL3_DBG")
            .map(|v| v == "layerdump")
            .unwrap_or(false)
            && il <= 1
        {
            let r = (xn.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
            eprintln!(
                "  [vkl] L{il} xn rms={r:.5} xn[0]={:.6} nw[0..3]={:?}",
                xn[0],
                &norm_w[..3.min(norm_w.len())]
            );
        }

        let mut attn_out = if full {
            let _g = ph("attn_fwd");
            let out = attn_forward(tr, seq, il, attn_count, &xn, 24, 4, 256)?;
            drop(_g);
            attn_count += 1;
            out
        } else {
            let _g = ph("gdn_fwd");
            let r = gdn_forward(tr, seq, il, &xn)?;
            drop(_g);
            r
        };
        // 디버그: attention/GDN 출력 제거 — FFN만 남겨 격리.
        if std::env::var_os("LLM170_EXL3_DBG")
            .map(|v| v == "attn_skip")
            .unwrap_or(false)
        {
            for v in attn_out.iter_mut() {
                *v = 0.0;
            }
        }

        // 잔차
        for i in 0..h {
            x[i] += attn_out.get(i).copied().unwrap_or(0.0);
        }
        if std::env::var_os("LLM170_EXL3_DBG")
            .map(|v| v == "layerdump")
            .unwrap_or(false)
        {
            let r = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
            eprintln!(
                "  [vkl] L{il} post-attn rms={r:.5} gdn[0..2]={:?}",
                &attn_out[..2.min(attn_out.len())]
            );
        }

        // FFN (post_attention_norm)
        let ffn_norm_w = tr
            .norm(&format!("{lp}.post_attention_layernorm.weight"))
            .ok_or("ffn norm missing")?;
        let xf = rms_norm(&x, ffn_norm_w, eps);
        // plans/120 A1: FFN 3선형 + GPU ew(silu·mul) 단일 배치 — 게이트/업
        // 판독·CPU 활성화·업로드 제거(간극 감소). ew GPU exp는 10a.
        let _gf = ph("ffn_all");
        let ffn_out = tr.ffn_triple(
            &format!("{lp}.mlp.gate_proj"),
            &format!("{lp}.mlp.up_proj"),
            &format!("{lp}.mlp.down_proj"),
            &xf,
        )?;
        drop(_gf);
        for i in 0..h {
            x[i] += ffn_out.get(i).copied().unwrap_or(0.0);
        }
        if std::env::var_os("LLM170_EXL3_DBG")
            .map(|v| v == "layerdump")
            .unwrap_or(false)
        {
            let r = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
            eprintln!(
                "  [vkl] L{il} post-ffn rms={r:.5} ffn_out[0..2]={:?}",
                &ffn_out[..2.min(ffn_out.len())]
            );
        }
    }

    // 활성화 크기 디버그 (첫 스텝만)
    if seq.pos == 0 {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
        eprintln!(
            "  [dbg] 최종 x rms={rms:.4} max={:.4}",
            x.iter().cloned().fold(f32::MIN, f32::max)
        );
    }
    seq.pos += 1;
    seq.last_tok = token;

    // MTP 스냅샷(plans/121 A2) — h_in은 output_norm 전 잔차(qwen35 mtp_h 관례).
    seq.last_h.clear();
    seq.last_h.extend_from_slice(&x);

    // output_norm + lm_head
    let out_norm_w = tr
        .norm("model.language_model.norm.weight")
        .ok_or("output norm missing")?;
    let xn = rms_norm(&x, out_norm_w, eps);
    let _gh = ph("head");
    let r = tr.linear("lm_head", &xn)?;
    drop(_gh);
    seq.last_logits.clear();
    seq.last_logits.extend_from_slice(&r);
    Ok(r)
}

/// `llm170 exl3-decode <dir> <token_ids_comma> [n_predict]`
pub fn exl3_decode(dir: &str, tokens_str: &str, n_predict: usize) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    let t0 = std::time::Instant::now();
    eprintln!("  [exl3-decode] 상주 적재 중...");
    let mut tr = TrellisResident::load(dir)?;
    eprintln!(
        "  [exl3-decode] 적재 완료 {}선형 {:.1}s — 디코드 시작",
        tr.linears.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut seq = new_seq_state(tr.n_layers, 512);
    let t1 = std::time::Instant::now();
    let mut logits = Vec::new();
    // 프리필 (순차 디코드)
    for (i, &tok) in prompt.iter().enumerate() {
        logits = decode_step(&mut tr, &mut seq, tok)?;
        if i == 0 {
            let rms = (0..1).map(|_| 0f32).sum::<f32>(); // suppress warning
            let _ = rms;
        }
    }
    // 생성
    let mut out_tokens = Vec::new();
    for step in 0..n_predict {
        let (best, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));
        out_tokens.push(best as u32);
        if step < 6 {
            eprintln!(
                "  [exl3-decode] step {step}: token={}",
                out_tokens.last().copied().unwrap_or(0)
            );
        }
        if step + 1 < n_predict {
            logits = decode_step(&mut tr, &mut seq, out_tokens[out_tokens.len() - 1])?;
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    // 진단 덤프 — VK_TS: GPU 디스패치 집계(프레임 경로와 동일 ts 원장),
    // exl3_phase: CPU 위상 분해. decode_s 측정 후 호출(쿼리 WAIT 제외).
    if llm170_diag::flag::on("LLM170_VK_TS") {
        tr.ctx.ts_report();
    }
    phase_report();
    let total = prompt.len() + n_predict;
    let tps = total as f64 / decode_s;
    Ok(format!(
        "exl3-decode: prompt {} + gen {} in {:.1}s ({:.2} t/s) — gen {:?}",
        prompt.len(),
        n_predict,
        decode_s,
        tps,
        out_tokens.as_slice()
    ))
}

// ── T-배치 프리필 (plans/121 A1-pp) ─────────────────────────────────────
//
// 순차 디코드 대비: 선형은 T-배치 GEMM(연산강도 32배 — 메모리 벽 탈출),
// 비선형은 core 자산 재사용 — GDN 청크 스캔(core::gdn::gdn_chunk_seq,
// AR 등가 검증됨), 어텐션/활성화는 행 병렬. 수치 클래스는 순차 경로와
// 동일(f16 쌍 누산 + FOLD=4 케이던스 — gemm 커널 주석 참조).

/// 배치 프리필: 전체 프롬프트를 T-배치로 통과(>512 청크 분할) — 마지막
/// 토큰 logits 반환. 상태는 decode_step 연속 가능 형식으로 적립.
/// 산술: 10a(환원 순서·청크 분해 — 골든/토큰 기준 갱신 대상).
pub fn prefill_batch(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    let h = tr.hidden;
    if tokens.is_empty() {
        return Err("prefill_batch: 빈 프롬프트".into());
    }
    let mut logits = Vec::new();
    for chunk_toks in tokens.chunks(super::resident::BATCH_TMAX) {
        let t_rows = chunk_toks.len();
        // ── 프레임 경로(plans/121 원-서브밋): 잔차 GPU 상주, 층간 판독 0 ──
        // 기본 경로(2026-10-03 승격, 2026-10-04 ENV 계약으로 옵트아웃 삭제):
        // 원-서브밋 프레임 — pp512 123.02 t/s(+12.7%), corr 0.999999.
        {
            // T 전 범위 — 스펙 라운드 포함
            if seq.pos == 0 {
                // fresh 시퀀스 — GPU 상태가 타 시퀀스 잔류일 수 있다(슬롯
                // 재사용·기준 재생): 강제 재업로드(2026-10-03 사고).
                tr.gdn_st_invalidate_all();
            }
            let _g0 = ph("ppf:frame");
            tr.fframe_init()?;
            tr.gdn_frame_init()?;
            let mut attn_count_f = 0usize;
            let xp = tr.frame_x_ptr()?;
            for (t, &tok) in chunk_toks.iter().enumerate() {
                // SAFETY: xbuf [T][5120] 프레임 소유 — 행별 분리 기록.
                unsafe {
                    let src: &[f32] = tr.embed_row(tok);
                    std::ptr::copy_nonoverlapping(src.as_ptr(), xp.add(t * h), h);
                }
            }
            tr.frame_x_flush(t_rows)?;
            tr.ctx.begin_outer()?;
            let zb = tr.frame_zeros_buf()?;
            // 메가융합 1호(T-적응): 융합은 소형-T 전용 — T>64 프리필은
            // 인라인 WHT가 전용 had_in보다 느림(123→112 역행, 2026-10-04 측정).
            if t_rows <= 64 {
                let (s1, s2) = (
                    tr.suh_of("model.language_model.layers.0.linear_attn.in_proj_qkv")?,
                    tr.suh_of("model.language_model.layers.0.linear_attn.in_proj_z")?,
                );
                tr.frame_norm_resid_had(0, t_rows, zb, s1, s2)?;
            } else {
                tr.frame_norm_resid(0, t_rows, zb)?;
            }
            if llm170_diag::dump::opts().key("exl3_framedbg") {
                let got = tr.frame_read_xtb_row(1)?;
                let w0 = tr
                    .norm("model.language_model.layers.0.input_layernorm.weight")
                    .ok_or("ln0")?;
                let em: Vec<f32> = tr.embed_row(chunk_toks[0]).to_vec();
                let ss: f32 = em.iter().map(|v| v * v).sum();
                let inv = 1.0 / ((ss / em.len() as f32 + 1e-6).sqrt());
                let want: Vec<f32> = em
                    .iter()
                    .zip(w0)
                    .map(|(v, w)| v * inv * w)
                    .take(6)
                    .collect();
                eprintln!("  [framedbg] norm0 got={:?} want={:?}", &got[..6], want);
            }
            for il in 0..tr.n_layers {
                let lp = format!("model.language_model.layers.{il}");
                let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
                let ab = if il % 4 == 3 {
                    let _a = ph("ppf:attn");
                    let yq = tr.linear_chain(&format!("{lp}.self_attn.q_proj"), t_rows, 0)?;
                    let yk = tr.linear_chain(&format!("{lp}.self_attn.k_proj"), t_rows, 1)?;
                    let yv = tr.linear_chain(&format!("{lp}.self_attn.v_proj"), t_rows, 2)?;
                    tr.attn_layer_gpu(attn_count_f, t_rows, seq.pos, yq.0, yk.0, yv.0)?;
                    seq.kv[attn_count_f].len += t_rows;
                    attn_count_f += 1;
                    drop(_a);
                    tr.linear_chain(&format!("{lp}.self_attn.o_proj"), t_rows, 0)?
                        .0
                } else {
                    let _g = ph("ppf:gdn");
                    {
                        let g = &seq.gdn[il];
                        tr.gdn_state_upload(gdn_il, &g.states, &g.conv)?;
                    }
                    let (yq, yz) = if t_rows <= 64 {
                        let dslots = tr.linear_pair_dual(
                            [
                                &format!("{lp}.linear_attn.in_proj_qkv"),
                                &format!("{lp}.linear_attn.in_proj_z"),
                            ],
                            t_rows,
                        )?;
                        (dslots[0], dslots[1])
                    } else {
                        let slots = tr.linear_batch_multi_gpu(
                            &[
                                &format!("{lp}.linear_attn.in_proj_qkv"),
                                &format!("{lp}.linear_attn.in_proj_z"),
                            ],
                            t_rows,
                        )?;
                        (slots[0], slots[1])
                    };
                    tr.gdn_layer_gpu(gdn_il, t_rows, std::ptr::null_mut(), yq.0, yz.0)?;
                    drop(_g);
                    tr.linear_chain(&format!("{lp}.linear_attn.out_proj"), t_rows, 0)?
                        .0
                };
                if llm170_diag::dump::opts().key("exl3_framedbg") {
                    // 판독을 위해 잠시 외부 배치 해제(추가 대기 — 디버그 전용)
                    tr.ctx.end_outer()?;
                    let b = tr.debug_yb0_row(t_rows)?;
                    let kind = if il % 4 == 3 { "attn" } else { "gdn" };
                    eprintln!("  [framedbg] L{il} {kind} out={:?}", &b[..4]);
                    tr.ctx.begin_outer()?;
                }
                if t_rows <= 64 {
                    let (sg, su) = (
                        tr.suh_of(&format!("{lp}.mlp.gate_proj"))?,
                        tr.suh_of(&format!("{lp}.mlp.up_proj"))?,
                    );
                    tr.frame_norm_resid_had(2 * il + 1, t_rows, ab, sg, su)?;
                } else {
                    tr.frame_norm_resid(2 * il + 1, t_rows, ab)?;
                }
                if llm170_diag::dump::opts().key("exl3_framedbg") {
                    tr.ctx.end_outer()?;
                    let b = tr.debug_xtb_row()?;
                    let xr = tr.debug_xbuf_row()?;
                    let em: Vec<f32> = tr.embed_row(chunk_toks[0]).to_vec();
                    eprintln!(
                        "  [framedbg] L{il} ffn-in xn={:?} xbuf={:?} embed={:?}",
                        &b[..4],
                        &xr[..4],
                        &em[..4]
                    );
                    tr.ctx.begin_outer()?;
                }
                let yf = if t_rows <= 64 {
                    tr.ffn_trio_preah(
                        &format!("{lp}.mlp.gate_proj"),
                        &format!("{lp}.mlp.up_proj"),
                        &format!("{lp}.mlp.down_proj"),
                        t_rows,
                    )?
                } else {
                    tr.ffn_trio_chain(
                        &format!("{lp}.mlp.gate_proj"),
                        &format!("{lp}.mlp.up_proj"),
                        &format!("{lp}.mlp.down_proj"),
                        t_rows,
                    )?
                };
                if llm170_diag::dump::opts().key("exl3_framedbg") {
                    tr.ctx.end_outer()?;
                    let b = tr.debug_yb2_row()?;
                    eprintln!("  [framedbg] L{il} ffn out={:?}", &b[..4]);
                    tr.ctx.begin_outer()?;
                }
                let w_next = if il + 1 == tr.n_layers {
                    128 // output_norm(행 127≠L63 post_ln — 충돌 버그 수정)
                } else {
                    2 * (il + 1)
                };
                // 다음 층이 GDN이면 그 qkv/z의 had까지 융합(마지막 층 제외).
                let nxt_gdn = il + 1 < tr.n_layers && (il + 1) % 4 != 3 && t_rows <= 64;
                if nxt_gdn {
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
            tr.ctx.end_outer()?;
            drop(_g0);
            let xn_last = tr.frame_read_xtb_row(t_rows)?;
            logits = tr.linear("lm_head", &xn_last)?;
            seq.last_logits.clear();
            seq.last_logits.extend_from_slice(&logits);
            if tr.gpu_frames_active() {
                let n_gdn = tr.n_layers - tr.n_layers / 4;
                let mut gi = 0usize;
                let mut ai2 = 0usize;
                for il in 0..tr.n_layers {
                    if il % 4 != 3 {
                        let (states, conv) = {
                            let g = &mut seq.gdn[il];
                            (&mut g.states, &mut g.conv)
                        };
                        tr.gdn_state_sync(gi, states, conv)?;
                        gi += 1;
                        if gi >= n_gdn {
                            break;
                        }
                    } else {
                        let len = seq.kv[ai2].len;
                        let rows = (len * 1024).min(seq.kv[ai2].k.len());
                        let [k, v] = tr.attn_kv_sync(ai2, rows / 1024)?;
                        seq.kv[ai2].k[..rows].copy_from_slice(&k[..rows]);
                        seq.kv[ai2].v[..rows].copy_from_slice(&v[..rows]);
                        ai2 += 1;
                    }
                }
            }
            seq.pos += t_rows as u32;
            seq.last_tok = *chunk_toks.last().ok_or("빈 청크")?;
            seq.last_h.clear();
            {
                let xp = tr.frame_x_ptr()?;
                // SAFETY: 호스트가 기록한 마지막 임베딩/잔차 경로 — end_outer 후
                // xbuf 마지막 행은 GPU가 갱신했을 수 있어 판독 전 flush 역방향
                // (GPU→호스트) 동기가 필요하다 — frame_x_flush는 호스트→GPU라
                // 여기선 invalidate 경로를 쓴다(아래 frame_read_x_last).
                let row = tr.frame_read_x_last(t_rows)?.to_vec();
                let _ = xp;
                seq.last_h.extend_from_slice(&row);
            }
            continue;
        }
        // 임베딩 행 조립
    }
    Ok(logits)
}

/// `llm170 exl3-pp <dir> <token_ids> [n_predict]` — 배치 프리필 검증+벤치:
/// ① 순차 프리필 기준 로짓 확보 ② 배치 프리필 로짓 비교(상관·argmax)
/// ③ 이어서 greedy 생성 토큰 비교(배치→순차 디코드 전환 정합)
/// ④ 배치 pp 3회 중앙값 t/s.
pub fn exl3_pp(dir: &str, tokens_str: &str, n_predict: usize) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    eprintln!("  [exl3-pp] 상주 적재 중...");
    let t0 = std::time::Instant::now();
    let mut tr = TrellisResident::load(dir)?;
    eprintln!(
        "  [exl3-pp] 적재 완료 {:.1}s — 프롬프트 {} 토큰",
        t0.elapsed().as_secs_f64(),
        prompt.len()
    );
    let ctx_len = (prompt.len() + n_predict + 64).max(512);
    // 벤치 전용 모드 — 순차 기준·비교 생략(진단 루프용, 검증은 전체 모드로)
    let bench_only = llm170_diag::flag::on("LLM170_PP_BENCH_ONLY");

    // ① 순차 기준
    let mut seq1 = new_seq_state(tr.n_layers, ctx_len);
    let mut logits_seq = Vec::new();
    let mut seq_s = 0f64;
    if !bench_only {
        let t1 = std::time::Instant::now();
        for &tok in &prompt {
            logits_seq = decode_step(&mut tr, &mut seq1, tok)?;
        }
        seq_s = t1.elapsed().as_secs_f64();
    }

    // ② 배치 프리필
    let mut seq2 = new_seq_state(tr.n_layers, ctx_len);
    let t2 = std::time::Instant::now();
    let logits_b = prefill_batch(&mut tr, &mut seq2, &prompt)?;
    let bat_s = t2.elapsed().as_secs_f64();

    // 비교: 상관·최대차·argmax
    let n = logits_seq.len().min(logits_b.len());
    let (mut sxy, mut sx, mut sy, mut sxx, mut syy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut maxd = 0f32;
    for i in 0..n {
        let (a, b) = (logits_seq[i] as f64, logits_b[i] as f64);
        sxy += a * b;
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        maxd = maxd.max((logits_seq[i] - logits_b[i]).abs());
    }
    let corr = (n as f64 * sxy - sx * sy)
        / ((n as f64 * sxx - sx * sx) * (n as f64 * syy - sy * sy)).sqrt();
    let am_seq = logits_seq
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let am_b = logits_b
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // ③ greedy 생성 비교(각 상태에서 n_predict)
    let gen_from = |tr: &mut TrellisResident, seq: &mut SeqState, lg: &[f32]| -> Vec<u32> {
        let mut out = Vec::new();
        let mut logits = lg.to_vec();
        for step in 0..n_predict {
            let (best, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap_or((0, &0.0));
            out.push(best as u32);
            if step + 1 < n_predict {
                logits = match decode_step(tr, seq, best as u32) {
                    Ok(v) => v,
                    Err(_) => break,
                };
            }
        }
        out
    };
    let gen_seq = if bench_only {
        Vec::new()
    } else {
        gen_from(&mut tr, &mut seq1, &logits_seq)
    };
    let gen_b = gen_from(&mut tr, &mut seq2, &logits_b);

    // ④ 배치 pp 타이밍 3회 중앙값(상태 할당 제외)
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let mut s = new_seq_state(tr.n_layers, ctx_len);
        let t = std::time::Instant::now();
        prefill_batch(&mut tr, &mut s, &prompt)?;
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let med = times[times.len() / 2];

    // 진단 덤프 — VK_TS: GPU 디스패치 집계, exl3_phase: CPU 위상 분해.
    if llm170_diag::flag::on("LLM170_VK_TS") {
        tr.ctx.ts_report();
    }
    phase_report();

    let seq_tps = prompt.len() as f64 / seq_s;
    let bat_tps = prompt.len() as f64 / med;
    Ok(format!(
        "exl3-pp: 프롬프트 {} — 순차 {seq_tps:.2} t/s vs 배치 {bat_tps:.2} t/s (3회 중앙값, 배치 최초 {bat_s:.2}s)\n  로짓 corr={corr:.6} maxdiff={maxd:.4} argmax {}=={}{}\n  생성 {}토큰: 순차 {:?} 배치 {:?} {}",
        prompt.len(),
        am_seq,
        am_b,
        if am_seq == am_b {
            "일치"
        } else {
            "불일치"
        },
        n_predict,
        gen_seq,
        gen_b,
        if gen_seq == gen_b {
            "일치"
        } else {
            "불일치"
        },
    ))
}

// ── MTP 드래프트 층 (plans/121 A2) ─────────────────────────────────────
// qwen35 spec.rs 수학의 트렐리스 판: eh_proj=mtp.fc, enorm/hnorm=pre_fc_norm_*,
// attn_norm=input_layernorm, shared_head_norm=mtp.norm, head=lm_head 공유.
// KV 규약 "슬롯=pos"(QA-27): 드래프트가 쓴 슬롯은 수용 후 타깃 훅(h=본체
// 잔차)이 다시 쓴다 — 훅이 드래프트 로짓도 함께 낸다(Q35 GPU 경로 패턴).
