//! frame/forward — 단일 시퀀스 포워드(디코드·프리필) (plans/79 A).

use super::*;

/// 프레임 forward — t토큰 (t=1 디코드도 이 경로; decode_frame이 래퍼).
/// 포워드 종료 방식 — 비동기 프리필은 head 커널까지만 발행하고 리드백을 미룬다.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FwdMode {
    /// logits 전사(샘플링용).
    Full,
    /// GPU argmax 만 회수.
    Greedy,
    /// head 커널까지만 — 리드백 없음(호출부가 이벤트 확인 후 argmax).
    NoReadback,
}

pub fn frame_forward(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq: usize,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<Vec<f32>, Q4Error> {
    frame_forward_ex(acc, model, ctx, seq, seq_st, f, tokens, FwdMode::Full).map(|(l, _)| l)
}

/// greedy 판 — head 후 전사 대신 GPU argmax 로 토큰만 회수(plans/74).
pub fn frame_forward_greedy(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq: usize,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<u32, Q4Error> {
    frame_forward_ex(acc, model, ctx, seq, seq_st, f, tokens, FwdMode::Greedy).map(|(_, t)| t.expect("greedy token"))
}

#[allow(clippy::too_many_lines)]
pub(super) fn frame_forward_ex(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq: usize,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    tokens: &[u32],
    mode: FwdMode,
) -> Result<(Vec<f32>, Option<u32>), Q4Error> {
    let hp: &Hparams4 = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let k_len = hp.n_group * hp.d_state;
    let v_len = hp.dt_rank * hp.d_state;
    let conv_ch = 2 * k_len + v_len;
    let eps = hp.eps;
    let t = tokens.len();
    fs_begin(acc, t);

    // 0) 임베딩 — t행 → hc 스트림 방송 ([t][hc][n])
    {
        let embd = model
            .w("token_embd.weight")
            .ok_or(Q4Error::MissingTensor("token_embd".into()))?;
        let mut row = vec![0.0f32; n];
        let mut r = vec![0.0f32; t * hc * n];
        for (ti, &tok) in tokens.iter().enumerate() {
            dequant_row(embd.ty, embd.data, tok as u64, n as u64, &mut row);
            for s in 0..hc {
                r[ti * hc * n + s * n..ti * hc * n + (s + 1) * n].copy_from_slice(&row);
            }
        }
        acc.frame_write(f.res_hc, &r).map_err(Q4Error::Io)?;
        acc.capture_mark("emb_out").map_err(Q4Error::Io)?;
    }

    // PLE n-gram 행 (호스트 해시)
    let ple_rows = if hp.is_ple(1) {
        stages::ple_hash(ctx, seq_st, tokens)
    } else {
        Vec::new()
    };

    let trace = std::env::var_os("LLM170_Q4_TRACE").is_some();
    let t_call = std::time::Instant::now();
    let mut recr_idx = 0usize;
    let mut full_idx = 0usize;
    for il in 0..hp.n_layer {
        if trace {
            eprintln!("# frame layer {il} t={t} (ple={} recr={})", hp.is_ple(il), hp.is_recr(il));
        }
        if il < 4 {
            frame_ck(acc, f.res_hc, hc * n, t, &format!("L{il}.res_in"));
        }
        if llm170_diag::dump::opts().bufhash {
            // 앞 min(t,16)행만 해시 — 서로 다른 t 실행에서 공유 접두 행을
            // 맞대기 위한 캡(plans/80 §A).
            let rows16 = t.min(16);
            let (k_sel, n_ff) = (hp.n_expert_used, hp.n_ff_exp);
            buf_hash(acc, f.res_hc, hc * n * rows16, &format!("L{il}B.res_hc"));
            buf_hash(acc, f.mix, n * rows16, &format!("L{il}B.mix"));
            buf_hash(acc, f.gqkv, conv_ch * rows16, &format!("L{il}B.gqkv"));
            buf_hash(acc, f.gbg, hp.dt_rank * 2 * rows16, &format!("L{il}B.gbg"));
            buf_hash(acc, f.gconv, conv_ch * rows16, &format!("L{il}B.gconv"));
            buf_hash(acc, f.go, v_len * rows16, &format!("L{il}B.go"));
            buf_hash(acc, f.ffn_out, n * rows16, &format!("L{il}B.ffn_out"));
            buf_hash(acc, f.mroute, hp.n_expert * rows16, &format!("L{il}B.mroute"));
            buf_hash(acc, f.mids, k_sel * rows16, &format!("L{il}B.mids"));
            buf_hash(acc, f.mwt, k_sel * rows16, &format!("L{il}B.mwt"));
            buf_hash(acc, f.mxsel, n * k_sel * rows16, &format!("L{il}B.mxsel"));
            buf_hash(acc, f.mgu, n_ff * k_sel * rows16, &format!("L{il}B.mgu"));
            buf_hash(acc, f.my, n_ff * k_sel * rows16, &format!("L{il}B.my"));
            buf_hash(acc, f.mout, n * rows16, &format!("L{il}B.mout"));
            // plans/84 E.2: hc 중간체·PLE 버퍼 — 발산 국소화용.
            buf_hash(acc, f.lo, f.lo_len * rows16, &format!("L{il}B.lo"));
            buf_hash(acc, f.inj, hc * rows16, &format!("L{il}B.inj"));
            buf_hash(acc, f.gate, hc * n * rows16, &format!("L{il}B.gate"));
            if hp.is_ple(il) {
                buf_hash(acc, f.ple_key, hc * n * rows16, &format!("L{il}B.ple_key"));
                buf_hash(acc, f.ple_value, n * rows16, &format!("L{il}B.ple_value"));
                buf_hash(acc, f.ple_gated, hc * n * rows16, &format!("L{il}B.ple_gated"));
                buf_hash(acc, f.ple_conv_out, hc * n * rows16, &format!("L{il}B.ple_conv_out"));
                buf_hash(acc, f.ple_gate, hc * rows16, &format!("L{il}B.ple_gate"));
            }
        }
        // 1) PLE (blk.1) — plans/73: 디코드(t=1)는 디바이스 경로. 해시/gather는
        //    스텝 초에 호스트가 끝냈고(GPU 무의존), key/value 투영은 프레임 GEMM,
        //    gate/conv/잔차는 ple_math_dev 의 3커널 — 동기 d2h/h2d 왕복과
        //    CPU mm_batch 투영 2회([2560→10240])가 사라진다(4.5-11ms/step).
        //    폴백/프리필(t>1)은 기존 호스트 브리지. LLM170_PLE_HOST=1 강제.
        if hp.is_ple(il) && !stage_skipped("ple") {
            let mut ple_dev_done = false;
            if t == 1 && std::env::var_os("LLM170_PLE_HOST").is_none() {
                let heads = hp.ple_heads_per_ngram * 2;
                let emb_w = heads * hp.ple_head_dim;
                let mut emb = vec![0.0f32; emb_w];
                if ple_rows.len() == heads {
                    if let Err(e) = ctx.model.ple_gather(&ple_rows, &mut emb) {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| eprintln!("# ple-frame: gather 실패 — 호스트 브리지 ({e})"));
                    } else {
                    let mut pre_capture = Vec::new();
                    if std::env::var_os("LLM170_PLE_CHECK").is_some() {
                        // 그림자용 PLE 직전 res_hc(레이어 0 출력) 판독 — 동기 1회.
                        pre_capture = vec![0.0f32; hc * n];
                        acc.frame_read(f.res_hc, &mut pre_capture).map_err(Q4Error::Io)?;
                    }
                let w_key = model.w4(&format!("blk.{il}.ple_key.weight"))?;
                let w_value = model.w4(&format!("blk.{il}.ple_value.weight"))?;
                let nk = model.f32_vec4(&format!("blk.{il}.ple_norm_key.weight"))?;
                let nq = model.f32_vec4(&format!("blk.{il}.ple_norm_query.weight"))?;
                let nc = model.f32_vec4(&format!("blk.{il}.ple_norm_conv.weight"))?;
                let cw = model.f32_vec4(&format!("blk.{il}.ple_conv1d.weight"))?;
                let r = acc
                    .frame_write(f.ple_emb, &emb)
                    .map_err(|e| e.to_string())
                    .and_then(|_| {
                        acc.frame_mm_group(
                            f.ple_emb, &[w_key, w_value], &[f.ple_key, f.ple_value], t,
                        )
                    })
                    .and_then(|_| {
                        acc.ple_math_dev(
                            f.res_hc, f.ple_key, f.ple_value, &nk, &nq, &nc, &cw,
                            f.ple_gated, f.ple_conv_out, f.ple_gate, seq, t, hp.eps,
                            n, hc, hp.ple_conv_k, hp.ple_ngram,
                            (hp.ple_conv_k - 1) * hp.ple_ngram, &seq_st.ple_conv,
                        )
                    });
                match r {
                    Ok(()) => {
                        ple_dev_done = true;
                        let check = std::env::var_os("LLM170_PLE_CHECK").is_some();
                        if check {
                            // 그림자: PLE 이전 값(토큰 임베딩 방송)에서 호스트 재계산해
                            // 디바이스 결과와 비교. 호스트 링도 갱신(스텝 흐름 유지).
                            let pre_capture_ref = &pre_capture;
                            let mut rows2: Vec<Vec<f32>> = vec![pre_capture.clone()];
                            stages::ple_block(ctx, seq_st, il, &mut rows2, &ple_rows, Some(vec![emb.clone()]))?;
                            let host: Vec<f32> = rows2.concat();
                            let mut r2 = vec![0.0f32; hc * n];
                            let mut dkey = vec![0.0f32; hc * n];
                            let mut dval = vec![0.0f32; n];
                            acc.frame_read(f.ple_key, &mut dkey).map_err(Q4Error::Io)?;
                            acc.frame_read(f.ple_value, &mut dval).map_err(Q4Error::Io)?;
                            let mut hkey = vec![vec![0.0f32; hc * n]; 1];
                            let w_key2 = model.w4(&format!("blk.{il}.ple_key.weight"))?;
                            let w_value2 = model.w4(&format!("blk.{il}.ple_value.weight"))?;
                            ctx.mm_batch(&[emb.clone()], &w_key2, &mut hkey)?;
                            let mut hval = vec![vec![0.0f32; n]; 1];
                            ctx.mm_batch(&[emb.clone()], &w_value2, &mut hval)?;
                            let mk = dkey.iter().zip(hkey[0].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                            let mv = dval.iter().zip(hval[0].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                            let mut dgate = vec![0.0f32; hc];
                            let mut dgated = vec![0.0f32; hc * n];
                            acc.frame_read(f.ple_gate, &mut dgate).map_err(Q4Error::Io)?;
                            acc.frame_read(f.ple_gated, &mut dgated).map_err(Q4Error::Io)?;
                            // 호스트 게이트 재계산(ple_block 잔차부와 동일식)
                            let mut hgate = vec![0.0f32; hc];
                            for s in 0..hc {
                                let kk = &hkey[0][s * n..(s + 1) * n];
                                let kn = crate::ops::rms_norm(kk, &nk[s * n..(s + 1) * n], hp.eps);
                                let qq = &pre_capture_ref[s * n..(s + 1) * n];
                                let qn = crate::ops::rms_norm(qq, &nq[s * n..(s + 1) * n], hp.eps);
                                let mut dot = 0.0f32;
                                for i in 0..n { dot += kn[i] * qn[i]; }
                                dot /= (n as f32).sqrt();
                                let mag = dot.abs().max(1e-6).sqrt();
                                hgate[s] = crate::ops::sigmoid(if dot >= 0.0 { mag } else { -mag });
                            }
                            eprintln!("# ple-check lens nk={} nq={} nc={} pre.len={} key.len={}", nk.len(), nq.len(), nc.len(), pre_capture_ref.len(), hkey[0].len());
                            let mg = dgate.iter().zip(hgate.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                            eprintln!("# ple-check gate dev={:?} host={:?} max={mg:.3e}", dgate.iter().map(|x| (x*1e4).round()/1e4).collect::<Vec<_>>(), hgate.iter().map(|x| (x*1e4).round()/1e4).collect::<Vec<_>>());
                            eprintln!("# ple-check key max|d-h|={mk:.3e} value max|d-h|={mv:.3e}");
                            acc.frame_read(f.res_hc, &mut r2).map_err(Q4Error::Io)?;
                            let mut md = 0.0f32;
                            let mut at = 0usize;
                            for (i, (a, b)) in r2.iter().zip(host.iter()).enumerate() {
                                let d = (a - b).abs();
                                if d > md { md = d; at = i; }
                            }
                            eprintln!(
                                "# ple-check pos={} max|dev-host|={md:.3e} at={at} (dev={:.4} host={:.4})",
                                seq_st.pos, r2[at.min(r2.len() - 1)], host[at.min(host.len() - 1)]
                            );
                            seq_st.qsa_host_stale = false;
                        }
                    }
                    Err(e) => {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            eprintln!("# ple-frame: 디바이스 경로 폴백 — 호스트 브리지 ({e})")
                        });
                    }
                }
                }
            }
            // (t==1 블록 종료 — 폴백은 바깥에서)
            }
            if !ple_dev_done {
                let mut r = vec![0.0f32; t * hc * n];
                acc.capture_mark("ple_in").map_err(Q4Error::Io)?;
                acc.frame_read(f.res_hc, &mut r).map_err(Q4Error::Io)?;
                let mut rows: Vec<Vec<f32>> = r.chunks_exact(hc * n).map(|c| c.to_vec()).collect();
                stages::ple_block(ctx, seq_st, il, &mut rows, &ple_rows, None)?;
                let flat: Vec<f32> = rows.concat();
                acc.frame_write(f.res_hc, &flat).map_err(Q4Error::Io)?;
                acc.capture_mark("ple_out").map_err(Q4Error::Io)?;
                sync_mark(acc, "hc.ple_bridge", f.res_hc)?;
            }
        }

        // 2) hc attn mix
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc, t)?;
        sync_mark(acc, &format!("L{il}.hc_attn"), f.mix)?;
        if il < 4 {
            frame_ck(acc, f.mix, n, t, &format!("L{il}.mix"));
        }
        if il == 0 {
            dbg("res_hc", acc, f.res_hc, hc * n * t);
        }

        // 3) attention — GDN 프레임 / QSA 값 브리지
        if hp.is_recr(il) {
            if stage_skipped("gdn") {
                // 진단용: GDN 단계 생략(출력 무효) — 디코드 스텝 비용 분해.
            } else {
            gdn_frame(acc, model, f, il, seq, recr_idx, conv_ch, k_len, v_len, eps, t)?;
            if il < 4 {
                frame_ck(acc, f.ffn_out, n, t, &format!("L{il}.gdn"));
            }
            }
            recr_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
            if il <= 4 && llm170_diag::dump::opts().bufhash {
                buf_hash(acc, f.res_hc, hc * n * t.min(16), &format!("L{il}A.res_attn"));
            }
            sync_mark(acc, &format!("L{il}.gdn_combine"), f.res_hc)?;
        } else {
            // QSA — plans/67 2c: 디바이스 상주 경로 우선. 투영·norm·rope·어텐션·
            // wo가 전부 GPU에 있고 d2h는 iq/ik/k/v(캐시 적립)뿐이다. 초기 3단계
            // (mm_group/qk_norm_rope/판독) 실패 시에만 구값 브리지로 폴백 — 그
            // 시점엔 캐시 미변경이라 이중 적립이 없다.
            if stage_skipped("qsa") {
                // 진단용: QSA 브리지 생략(출력 무효).
            } else if qsa_frame(
                acc, model, ctx, seq_st, f, il, t, full_idx, seq, &QsaBufs::whole(f),
            )
            .is_ok()
            {
                acc.capture_mark("recr_out").map_err(Q4Error::Io)?;
            } else {
            let qtm = std::env::var_os("LLM170_Q4_TIME").is_some();
            let mut ql = std::time::Instant::now();
            let mut mix_v = vec![0.0f32; t * n];
            acc.capture_mark("recr_in").map_err(Q4Error::Io)?;
            acc.frame_read(f.mix, &mut mix_v).map_err(Q4Error::Io)?;
            let read_ms = ql.elapsed().as_secs_f64() * 1e3;
            ql = std::time::Instant::now();
            let xs: Vec<Vec<f32>> = mix_v.chunks_exact(n).map(|c| c.to_vec()).collect();
            let out = stages::qsa_layer(ctx, seq_st, il, &xs, t, full_idx)?;
            let stage_ms = ql.elapsed().as_secs_f64() * 1e3;
            ql = std::time::Instant::now();
            let flat: Vec<f32> = out.concat();
            acc.frame_write(f.ffn_out, &flat).map_err(Q4Error::Io)?;
            acc.capture_mark("recr_out").map_err(Q4Error::Io)?;
            if qtm {
                eprintln!(
                    "# qsa-bridge L{il} t={t} read(d2h+드레인)={read_ms:.1}ms stage={stage_ms:.1}ms write(h2d)={:.1}ms",
                    ql.elapsed().as_secs_f64() * 1e3
                );
            }
            }
            full_idx += 1;
            sync_mark(acc, &format!("L{il}.qsa_bridge"), f.ffn_out)?;
            if il < 4 {
                frame_ck(acc, f.ffn_out, n, t, &format!("L{il}.qsa"));
            }
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
        }

        // 4) hc ffn mix + MoE
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc, t)?;
        sync_mark(acc, &format!("L{il}.hc_ffn"), f.mix)?;
        if il < 4 {
            frame_ck(acc, f.mix, n, t, &format!("L{il}.mixf"));
        }
        if il == 0 {
            dbg("mix2", acc, f.mix, n * t);
        }
        moe_frame(acc, model, f, il, n, t)?;
        sync_mark(acc, &format!("L{il}.moe"), f.mout)?;
        if il < 4 {
            frame_ck(acc, f.mout, n, t, &format!("L{il}.moe"));
        }
        
        if il <= 4 && llm170_diag::dump::opts().bufhash {
            // plans/84 E.2: ffn combine 직전 3입력 + res 스냅샷
            buf_hash(acc, f.res_hc, hc * n * t.min(16), &format!("L{il}A.res_pre_ffn"));
            buf_hash(acc, f.mout, n * t.min(16), &format!("L{il}A.mout_site"));
            buf_hash(acc, f.inj, hc * t.min(16), &format!("L{il}A.inj_site"));
            // 결정적 판별: 같은 순간 2회 판독 + 결합 직후 재판독 — 커널이 mout을
            // 쓰는지/판독이 비결정인지/진짜 값 차이인지 분리.
            buf_hash(acc, f.mout, n * t.min(16), &format!("L{il}A.mout_reread"));
        }
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
        if il <= 4 && llm170_diag::dump::opts().bufhash {
            // plans/84 E.2: ffn combine 직후 + 행 반분(0-7/8-15) — 경계 행 패턴 식별
            buf_hash(acc, f.mout, n * t.min(16), &format!("L{il}A.mout_postcombine"));
            buf_hash(acc, f.res_hc, hc * n * t.min(16), &format!("L{il}A.res_ffn"));
            if t >= 16 {
                buf_hash(acc, f.res_hc, hc * n * 8, &format!("L{il}A.res_ffn_lo"));
            }
        }
        sync_mark(acc, &format!("L{il}.ffn_combine"), f.res_hc)?;
        if il == 0 {
        }
    }
    frame_ck(acc, f.res_hc, hc * n, t, "head.res");

    // 5) head — output hc mix(전 토큰) → 마지막 행만 GEMM → 판독
    {
        let w_norm = f.consts["output_hc_norm"];
        op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.hxn, eps, n, w_reps: hc })?;
        let w_down = model.w4("output_hc_down.weight")?;
        acc.frame_mm(f.hxn, &w_down, f.hlo, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluDiv { t: f.hlo, div: hc as f32, n: f.hlo_len * t })?;
        let w_up = model.w4("output_hc_up.weight")?;
        acc.frame_mm(f.hlo, &w_up, f.hgate, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::HcGateMean { xn: f.hxn, gate: f.hgate, out: f.hin, hc, n })?;
        if t > 1 {
            op(acc, FrameOp::CopyRows { src: f.hin, dst: f.hin_last, src_off: (t - 1) * n, dst_off: 0, n })?;
        }
        let hin = if t > 1 { f.hin_last } else { f.hin };
        let wout = model.w("output.weight").ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        acc.frame_mm(hin, &wout, f.logits, 1).map_err(Q4Error::Io)?;
        if mode == FwdMode::NoReadback {
            ftime_report(t);
            return Ok((Vec::new(), None));
        }
        if mode == FwdMode::Greedy {
            // GPU argmax — vocab×4B 전사·CPU 스캔 회피(plans/74).
            let toks = acc.frame_argmax_rows(f.logits, 1, hp.vocab).map_err(Q4Error::Io)?;
            ftime_report(t);
            return Ok((Vec::new(), Some(toks[0])));
        }
        let mut logits = vec![0.0f32; hp.vocab];
        acc.capture_mark("logits_in").map_err(Q4Error::Io)?;
        acc.frame_read(f.logits, &mut logits).map_err(Q4Error::Io)?;
        ftime_report(t);
        if ftime_on() {
            eprintln!("# frame-total t={t} {:.1}ms", t_call.elapsed().as_secs_f64() * 1e3);
        }
        if std::env::var_os("LLM170_Q4_DBG").is_some() {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            eprintln!(
                "# fdbg logits t={t}: top5 {:?}",
                idx[..5].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>()
            );
        }
        Ok((logits, None))
    }
}


pub fn decode_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq: usize,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    token: u32,
) -> Result<Vec<f32>, Q4Error> {
    frame_forward(acc, model, ctx, seq, seq_st, f, &[token])
}

/// decode_frame 의 greedy 판 — head 후 로짓 전사 대신 GPU argmax(plans/74).
pub fn decode_frame_greedy(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq: usize,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    token: u32,
) -> Result<u32, Q4Error> {
    frame_forward_greedy(acc, model, ctx, seq, seq_st, f, &[token])
}

/// QSA 프레임이 소비하는 프레임 버퍼 핸들 — 단일 시퀀스 경로는 Frame4의
/// 전폭 버퍼(whole), 다중 시퀀스 청크 프리필은 시퀀스별 행 대역(pre_views)을
/// 넘긴다. 커널·산술 순서는 동일 — 배치 결과가 순차 프리필과 같은 이유다.
pub(super) struct QsaBufs {
    pub(super) mix: u64,
    pub(super) q: u64,
    pub(super) k: u64,
    pub(super) v: u64,
    pub(super) iq: u64,
    pub(super) ik: u64,
    pub(super) attn: u64,
    pub(super) out: u64,
}

impl QsaBufs {
    fn whole(f: &Frame4) -> Self {
        Self {
            mix: f.mix,
            q: f.qsa_q,
            k: f.qsa_k,
            v: f.qsa_v,
            iq: f.qsa_iq,
            ik: f.qsa_ik,
            attn: f.qsa_attn,
            out: f.ffn_out,
        }
    }
}

/// QSA 프레임 (plans/67 2c) — 투영·norm·rope·어텐션·wo 전부 디바이스 상주.
/// d2h는 캐시 적립용 iq/ik/k/v(t×(idx_heads·idx_dim+idx_dim+2·n_kv·hd) ≈ t×3,840
/// floats)뿐 — 기존 값 브리지는 mix+wq+wo 왕복 t×~10,880 floats를 나르던 것과
/// 비교해 첫 3단계 실패 시에만 호출부가 값 브리지로 폴백한다(그 시점엔 아직
/// 캐시를 건드리지 않는다 — 이중 적립 없음).
#[allow(clippy::too_many_arguments)]
pub(super) fn qsa_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq_st: &mut SeqState4,
    f: &Frame4,
    il: usize,
    t: usize,
    full_idx: usize,
    seq: usize,
    b: &QsaBufs,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let qtm = std::env::var_os("LLM170_Q4_TIME").is_some();
    let t_qsa = std::time::Instant::now();
    let mut lap = t_qsa;
    let (n_head, n_kv, hd) = (hp.n_head, hp.n_kv, hp.head_dim);
    let (n_rot, idx_dim) = (hp.n_rot, hp.idx_dim);
    let wq = model.w4(&format!("blk.{il}.attn_q.weight"))?;
    let wk = model.w4(&format!("blk.{il}.attn_k.weight"))?;
    let wv = model.w4(&format!("blk.{il}.attn_v.weight"))?;
    let wo = model.w4(&format!("blk.{il}.attn_output.weight"))?;
    let w_iq = model.w4(&format!("blk.{il}.indexer.q_proj.weight"))?;
    let w_ik = model.w4(&format!("blk.{il}.indexer.k_proj.weight"))?;
    // 1) 5투영 — 디바이스 그룹 1호출(왕복 0). wq 출력 [t][n_head·2hd]는 어텐션
    //    커널의 q 레이아웃(q‖게이트 인터리브)과 정확히 일치(plans/67 위험 항 해소).
    let t_mm = std::time::Instant::now();
    acc.frame_mm_group(
        b.mix,
        &[wq, wk, wv, w_iq, w_ik],
        &[b.q, b.k, b.v, b.iq, b.ik],
        t,
    )
    .map_err(Q4Error::Io)?;
    if qtm { eprintln!("# qsa-frame L{il} t={t} proj-mm={:.2}ms", t_mm.elapsed().as_secs_f64()*1e3); }
    let t_rp = std::time::Instant::now();
    sync_mark(acc, "qsa.mm_group", b.q)?;
    if il == 3 && llm170_diag::dump::opts().bufhash {
        // plans/84 E.2: QSA 내부 이분 — 첫 상이 서브옵을 노출한다.
        buf_hash(acc, b.q, (n_head * 2 * hd) * t.min(16), "L3Q.proj_q");
        buf_hash(acc, b.iq, idx_dim * hp.idx_heads * t.min(16), "L3Q.proj_iq");
    }
    // 2) q/k norm+rope in-place — 커널 산술은 호스트 rms_norm(sq_sum 32세그먼트
    //    f64)+rope_head(f64 회전)와 동일열(비트 동일 기대).
    let pos0 = seq_st.pos;
    // qk_norm_rope 커널은 norm 가중치를 **헤드별 타일**(qw[r0·hd..])로 읽는다
    // (decode 경로는 rawinject가 타일해 업로드 — ssm_norm 타일링과 같은 규약).
    // 공유 [hd] 원본을 그대로 올리면 24헤드 분량(6144)을 256원소 버퍼에서 읽어
    // illegal address(700)로 폭주한다 — plans/67 2c 연결 시 실측 발견(2026-09-14).
    let (qn, kn) = (&f.qsa_qn_t[full_idx], &f.qsa_kn_t[full_idx]);
    acc.frame_qk_norm_rope(
        b.q, b.k, qn, kn, &f.qsa_cs, hp.eps, pos0 as usize,
        n_head, n_kv, hd, n_rot, t,
    )
    .map_err(Q4Error::Io)?;
    if qtm { eprintln!("# qsa-frame L{il} t={t} rope={:.2}ms", t_rp.elapsed().as_secs_f64()*1e3); }
    sync_mark(acc, "qsa.qkrope", b.k)?;
    if il == 3 && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, b.q, (n_head * 2 * hd) * t.min(16), "L3Q.rope_q");
        buf_hash(acc, b.k, (n_kv * hd) * t.min(16), "L3Q.rope_k");
    }
    if qtm {
        eprintln!("# qsa-frame L{il} t={t} mm+rope={:.2}ms", t_qsa.elapsed().as_secs_f64() * 1e3);
        lap = std::time::Instant::now();
    }
    // ─── plans/73: 디코드(t=1) 디바이스 선택 ───
    // iq/ik/k/v의 d2h 4회(각각 동기식 드레인) + 호스트 선택(0.8-1.5ms/층)이
    // 스텝의 최대 단일 유휴였다(KTRACE 16k: "after qk_norm_rope" 40ms/step).
    // 선택 전 과정을 커널로 옮기고 어텐션이 목록을 디바이스에서 직접 읽는다.
    // 호스트 kv/idx 캐시는 이 경로에서 갱신하지 않는다(→ qsa_host_stale;
    // 프리필 진입 시 풀에서 1회 재구축). LLM170_QSA_HOSTSEL=1이면 구경로.
    let kq_scale = hp.kq_scale();
    let r = hp.compress[il] as usize;
    if t == 1 && std::env::var_os("LLM170_QSA_HOSTSEL").is_none() {
        let t_w = std::time::Instant::now();
        let iqw = model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
        let ikw = model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        if qtm { eprintln!("# qsa-frame L{il} w-extract={:.2}ms", t_w.elapsed().as_secs_f64()*1e3); }
        let t_s = std::time::Instant::now();
        let dev = acc
            .qsa_sel_dev(
                full_idx, seq, b.iq, b.ik, t, pos0 as usize,
                hp.idx_heads, hp.idx_dim, r, hp.idx_top_k,
                &iqw, &ikw, &f.qsa_cs_idx, hp.eps,
            )
            .and_then(|(sd, od, list_len)| {
                if qtm { eprintln!("# qsa-frame L{il} sel_dev={:.2}ms", t_s.elapsed().as_secs_f64()*1e3); }
                let t_kv = std::time::Instant::now();
                acc.qsa_kv_dev(full_idx, seq, b.k, b.v, t, pos0 as usize, n_kv, hd)
                    .and_then(|(kc, vc)| {
                        if qtm { eprintln!("# qsa-frame L{il} kv_dev={:.2}ms", t_kv.elapsed().as_secs_f64()*1e3); }
                        let t_attn = std::time::Instant::now();
                        let r = acc.qsa_attention_dev_sel(
                            b.q, kc, vc, sd, od, list_len, kq_scale,
                            n_head, n_kv, hd, t, b.attn,
                        );
                        if qtm { eprintln!("# qsa-frame L{il} attn_sel={:.2}ms", t_attn.elapsed().as_secs_f64()*1e3); }
                        r.map(|_| (sd, od, list_len))
                    })
            });
        match dev {
            Ok((sd, od, list_len)) => {
                if std::env::var_os("LLM170_QSA_SELCHECK").is_some() {
                    // 검증 그림자: 동일 입력으로 호스트 선택을 재계산해 목록을
                    // 대조한다. 이 경로는 호스트 캐시도 갱신하므로 stale가 유지
                    // 되지 않는다(프리필 재구축 불필요 — 검증 모드의 부수 효과).
                    let (h_idx, h_off) =
                        qsa_selcheck_host(ctx, acc, seq_st, f, il, t, full_idx, pos0 as usize, r)?;
                    match acc.qsa_sel_readback(sd, od, list_len) {
                        Ok((d_idx, d_off)) => {
                            if h_idx != d_idx
                                || h_off.first() != d_off.first()
                                || h_off.get(1) != d_off.get(1)
                            {
                                eprintln!(
                                    "# qsa-selcheck L{il} pos={pos0} MISMATCH host({} entries) dev({} entries)",
                                    h_idx.len(),
                                    d_idx.len()
                                );
                            }
                        }
                        Err(e) => eprintln!("# qsa-selcheck L{il} readback 실패: {e}"),
                    }
                } else {
                    seq_st.qsa_host_stale = true;
                }
                acc.frame_mm_group(b.attn, &[wo], &[b.out], t)
                    .map_err(Q4Error::Io)?;
                return Ok(());
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("# qsa-frame: 디바이스 선택 폴백 — 호스트 경로 ({e})"));
            }
        }
    }
    // ─── plans/74: 프리필 항등 선택 단축(비트 동일) ───
    // n_past ≤ idx_top_k + r - 1 이면 호스트 선택도 **전체 블록을 오름차순**으로
    // 고른다(qsa_select 패스 B: sel_blocks=(0..n_blocks) 그대로 → sort_unstable).
    // 즉 점수·순위·top-k 가 모두 항등이라 목록을 직접 만들어도 결과가 같다 —
    // d2h 4회(동기 드레인) + 호스트 점수/정렬(0.8-1.5ms/층)을 건너뛰고
    // 디바이스 풀(KV·idx)만 적립한다. 실패하면 종전 호스트 경로로 폴백.
    // 킬스위치 LLM170_QSA_NOID=1.
    if t > 1
        && pos0 as usize + t < hp.idx_top_k + r
        && std::env::var_os("LLM170_QSA_NOID").is_none()
    {
        let pos0u = pos0 as usize;
        let ikw = model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        let dev = acc
            .qsa_kv_dev(full_idx, seq, b.k, b.v, t, pos0u, n_kv, hd)
            .and_then(|(kc, vc)| {
                acc.qsa_idx_append_dev(
                    full_idx, seq, b.ik, t, pos0u, idx_dim, r, &ikw, &f.qsa_cs_idx, hp.eps,
                )?;
                // 항등 목록 — 행 i 의 선택 = [0, pos0+i] (오름차순 전체).
                let mut sel_off: Vec<u32> = vec![0u32; t + 1];
                for t2 in 0..t {
                    sel_off[t2 + 1] = sel_off[t2] + (pos0u + t2 + 1) as u32;
                }
                let mut sel_idx: Vec<u32> = vec![0u32; sel_off[t] as usize];
                let mut o = 0usize;
                for t2 in 0..t {
                    for j in 0..(pos0u + t2 + 1) {
                        sel_idx[o] = j as u32;
                        o += 1;
                    }
                }
                acc.qsa_attention_dev_res(
                    b.q, kc, vc, &sel_idx, &sel_off, kq_scale,
                    n_head, n_kv, hd, t, b.attn,
                )
            });
        match dev {
            Ok(()) => {
                seq_st.qsa_host_stale = true;
                if qtm {
                    eprintln!("# qsa-frame L{il} t={t} identity-select={:.2}ms", lap.elapsed().as_secs_f64() * 1e3);
                }
                acc.frame_mm_group(b.attn, &[wo], &[b.out], t)
                    .map_err(Q4Error::Io)?;
                return Ok(());
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("# qsa-frame: 항등 선택 단축 실패 — 호스트 경로 ({e})"));
            }
        }
    }
    // ─── 프리필(t>1) 진입: 호스트 캐시 재구축(디코드가 갱신을 건너뛴 경우) ───
    if t > 1 && seq_st.qsa_host_stale {
        let pos = pos0 as usize;
        let nb = pos / r.max(1);
        let mut bk = vec![0.0f32; nb * idx_dim];
        acc.qsa_host_rebuild(
            full_idx, seq, pos, n_kv * hd,
            &mut seq_st.kv_k[full_idx], &mut seq_st.kv_v[full_idx],
            &mut seq_st.idx_k[full_idx], &mut bk, r, idx_dim,
        )
        .map_err(|e| {
            Q4Error::Io(format!("L{il} t={t} 풀→호스트 재구축 실패: {e}"))
        })?;
        seq_st.idx_bk[full_idx] = bk;
        seq_st.qsa_host_stale = false;
    }
    // 3) 캐시 적립용 최소 d2h — iq/ik(선택 로직 입력) + k(이미 norm·rope됨)/v.
    let (iq_len, ik_len, kv_len) = (
        hp.idx_heads * idx_dim,
        idx_dim,
        n_kv * hd,
    );
    let mut iq_v = vec![0.0f32; t * iq_len];
    let mut ik_v = vec![0.0f32; t * ik_len];
    let mut k_v = vec![0.0f32; t * kv_len];
    let mut v_v = vec![0.0f32; t * kv_len];
    acc.frame_read(b.iq, &mut iq_v).map_err(Q4Error::Io)?;
    acc.frame_read(b.ik, &mut ik_v).map_err(Q4Error::Io)?;
    acc.frame_read(b.k, &mut k_v).map_err(Q4Error::Io)?;
    acc.frame_read(b.v, &mut v_v).map_err(Q4Error::Io)?;
    sync_mark(acc, "qsa.d2h", b.v)?;
    let rows = |flat: &[f32], w: usize| -> Vec<Vec<f32>> {
        flat.chunks_exact(w).map(|c| c.to_vec()).collect()
    };
    if qtm {
        eprintln!("# qsa-frame L{il} t={t} d2h={:.2}ms", lap.elapsed().as_secs_f64() * 1e3);
        lap = std::time::Instant::now();
    }
    let kk = rows(&k_v, kv_len);
    let vv = rows(&v_v, kv_len);
    let iq = rows(&iq_v, iq_len);
    let ik = rows(&ik_v, ik_len);
    // 4) 선택(호스트) — k_prenormed=true: 디바이스가 norm·rope를 마친 k를
    //    그대로 적립. 이후 단계는 캐시가 갱신된 뒤라 폴백 없이 진행한다.
    let (sel_blk, sel_cnt, sel_stride) = stages::qsa_select(
        ctx, seq_st, il, &kk, &vv, &iq, &ik, t, full_idx, true,
    )?;
    let (sel_idx, sel_off) =
        stages::qsa_sel_list(&sel_blk, &sel_cnt, sel_stride, r, pos0 as usize, t);
    if t > 1 {
        // plans/73: 프리필도 디바이스 idx 풀을 갱신 — 이후 디코드의 qsa_sel_dev가
        // 풀을 이어 쓴다(호스트 선택 결과와 무관하게 풀은 항상 최신).
        let ikw2 = model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        if let Err(e) = acc.qsa_idx_append_host(
            full_idx, seq, &ik_v, t, pos0 as usize, hp.idx_dim, r,
            &ikw2, &f.qsa_cs_idx, hp.eps,
        ) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("# qsa-frame: idx 풀 적립 실패(디코드 폴백 예정) — {e}"));
        }
    }
    if qtm {
        eprintln!("# qsa-frame L{il} t={t} select+list={:.2}ms", lap.elapsed().as_secs_f64() * 1e3);
        lap = std::time::Instant::now();
    }
    // 5) 어텐션 — q를 디바이스 버퍼에서 직접. 실패 시에만 d2h q + CPU 재계산.
    let kn_max = (pos0 as usize + t) * n_kv * hd;
    // plans/67 3단계: KV 상주 풀 우선 — k/v를 D2D append하고 어텐션이 풀을
    // 직접 읽는다(매 층 매 스텝의 캐시 재업로드 8k 문맥 32MB 제거).
    // 미지원/실측 실패 시 기존 업로드 경로(qsa_attention_dev)로, 그것도
    // 실패하면 CPU 재계산으로 — 3단 폴백.
    let res = if std::env::var_os("LLM170_QSA_NORES").is_some() {
        Err("진단: 상주 풀 비활성".to_string())
    } else {
        acc.qsa_kv_dev(full_idx, seq, b.k, b.v, t, pos0 as usize, n_kv, hd)
            .and_then(|(kc, vc)| {
                acc.qsa_attention_dev_res(
                    b.q, kc, vc, &sel_idx, &sel_off, kq_scale,
                    n_head, n_kv, hd, t, b.attn,
                )
            })
    };
    let ck = &seq_st.kv_k[full_idx][..kn_max];
    let cv = &seq_st.kv_v[full_idx][..kn_max];
    if std::env::var_os("LLM170_QSA_RESCHECK").is_some() && !seq_st.qsa_host_stale
        && let Err(e) = acc.qsa_kv_check(full_idx, seq, ck, cv) {
            eprintln!("# qsa-rescheck L{il} t={t} pos0={pos0}: {e}");
        }
    let attn = res.or_else(|e2| {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| eprintln!("# qsa-frame: 상주 풀 미사용 — 업로드 경로 ({e2})"));
        acc.qsa_attention_dev(
            b.q, ck, cv, &sel_idx, &sel_off, kq_scale,
            n_head, n_kv, hd, t, b.attn,
        )
        .map_err(|e| format!("{e2}; {e}"))
    });
    if let Err(e) = attn {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!("# qsa-frame: GPU 어텐션 폴백 — CPU 재계산 ({e})")
        });
        let mut q_v = vec![0.0f32; t * n_head * 2 * hd];
        acc.frame_read(b.q, &mut q_v).map_err(Q4Error::Io)?;
        let qg = rows(&q_v, n_head * 2 * hd);
        let attn = stages::qsa_cpu_attn_rows(
            &qg, seq_st, full_idx, &sel_blk, &sel_cnt, sel_stride, r, pos0 as usize, t,
            n_head, n_kv, hd, kq_scale,
        );
        let flat: Vec<f32> = attn.concat();
        acc.frame_write(b.attn, &flat).map_err(Q4Error::Io)?;
    }
    // 6) wo 투영 — 어텐션 출력을 디바이스에서 ffn_out으로(왕복 0).
    if qtm {
        eprintln!("# qsa-frame L{il} t={t} attn={:.2}ms", lap.elapsed().as_secs_f64() * 1e3);
        lap = std::time::Instant::now();
    }
    acc.frame_mm_group(b.attn, &[wo], &[b.out], t)
        .map_err(Q4Error::Io)?;
    let _ = &mut lap;
    Ok(())
}


/// SELCHECK 검증 그림자(plans/73) — 디바이스 선택과 동일 입력으로 호스트
/// 선택을 재계산해 목록을 돌려준다. 기존 d2h+qsa_select 경로를 그대로 쓰므로
/// 호스트 kv/idx 캐시도 함께 갱신된다(검증 모드에선 stale가 유지되지 않음).
#[allow(clippy::too_many_arguments)]
pub(super) fn qsa_selcheck_host(
    ctx: &Ctx,
    acc: &dyn Accelerator,
    seq_st: &mut SeqState4,
    f: &Frame4,
    il: usize,
    t: usize,
    full_idx: usize,
    pos0: usize,
    r: usize,
) -> Result<(Vec<u32>, Vec<u32>), Q4Error> {
    let hp = &ctx.model.hp;
    let (n_kv, hd) = (hp.n_kv, hp.head_dim);
    let (iq_len, ik_len, kv_len) = (hp.idx_heads * hp.idx_dim, hp.idx_dim, n_kv * hd);
    let mut iq_v = vec![0.0f32; t * iq_len];
    let mut ik_v = vec![0.0f32; t * ik_len];
    let mut k_v = vec![0.0f32; t * kv_len];
    let mut v_v = vec![0.0f32; t * kv_len];
    acc.frame_read(f.qsa_iq, &mut iq_v).map_err(Q4Error::Io)?;
    acc.frame_read(f.qsa_ik, &mut ik_v).map_err(Q4Error::Io)?;
    acc.frame_read(f.qsa_k, &mut k_v).map_err(Q4Error::Io)?;
    acc.frame_read(f.qsa_v, &mut v_v).map_err(Q4Error::Io)?;
    let rows = |flat: &[f32], w: usize| -> Vec<Vec<f32>> {
        flat.chunks_exact(w).map(|c| c.to_vec()).collect()
    };
    let (kk, vv, iq, ik) = (
        rows(&k_v, kv_len),
        rows(&v_v, kv_len),
        rows(&iq_v, iq_len),
        rows(&ik_v, ik_len),
    );
    let (sel_blk, sel_cnt, sel_stride) =
        stages::qsa_select(ctx, seq_st, il, &kk, &vv, &iq, &ik, t, full_idx, true)?;
    Ok(stages::qsa_sel_list(&sel_blk, &sel_cnt, sel_stride, r, pos0, t))
}

/// hc_mix 프레임 — CPU stages/hc.rs hc_mix와 동일 순서 (inject 반환 포함).
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_mix_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    kind: &str,
    eps: f32,
    n: usize,
    hc: usize,
    t: usize,
) -> Result<(), Q4Error> {
    let w_norm = f.consts[&format!("blk.{il}.hc_{kind}_norm")];
    // plans/84 E.2: 어텐션 반쪽(lo/inj/gate)은 ffn 반쪽이 덮어써 계측 사각지대 —
    // il==0 attn에서 즉시 해시해 첫 상이 GEMM 출력을 직접 노출한다.
    let mark_attn = il <= 3 && kind == "attn";
    op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.xn, eps, n, w_reps: hc })?;
    sync_mark(acc, "hc.rms", f.xn)?;
    let w_down = model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
    let w_inject = model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;
    acc.frame_mm_group(f.xn, &[w_down, w_inject], &[f.lo, f.inj], t)
        .map_err(Q4Error::Io)?;
    if mark_attn && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, f.xn, hc * n * t.min(16), &format!("L{il}C.attn_xn"));
        buf_hash(acc, f.lo, f.lo_len * t.min(16), &format!("L{il}C.attn_lo"));
        buf_hash(acc, f.inj, hc * t.min(16), &format!("L{il}C.attn_inj"));
    }
    sync_mark(acc, "hc.down", f.lo)?;
    op(acc, FrameOp::SiluDiv { t: f.lo, div: hc as f32, n: f.lo_len * t })?;
    sync_mark(acc, "hc.silu", f.lo)?;
    let w_up = model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?;
    acc.frame_mm(f.lo, &w_up, f.gate, t).map_err(Q4Error::Io)?;
    if mark_attn && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, f.gate, hc * n * t.min(16), &format!("L{il}C.attn_gate"));
    }
    sync_mark(acc, "hc.up", f.gate)?;
    op(acc, FrameOp::HcGateMean { xn: f.xn, gate: f.gate, out: f.mix, hc, n })?;
    if mark_attn && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, f.mix, n * t.min(16), &format!("L{il}C.attn_mix"));
    }
    sync_mark(acc, "hc.gate", f.mix)?;
    Ok(())
}

/// hc_combine 프레임 — layers.rs hc_combine과 동일 수식.
pub(super) fn hc_combine_frame(
    acc: &dyn Accelerator,
    f: &Frame4,
    out: u64,
    inj: u64,
    n: usize,
    hc: usize,
    t: usize,
) -> Result<(), Q4Error> {
    op(acc, FrameOp::HcCombine { res: f.res_hc, out, inj, hc, n, total: hc * n * t })
}

/// GDN 프레임 — stages/gdn.rs와 동일 순서 (t=1).
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    seq: usize,
    ri: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    eps: f32,
    t: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    // qkv/z/b/a 그룹 — 동일 입력 mix
    let wqkv = model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
    let wz = model.w4(&format!("blk.{il}.attn_gate.weight"))?;
    let wb = model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
    let wa = model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
    if !stage_skipped("gdn.mm") {
        acc.frame_mm_group(f.mix, &[wqkv, wz, wb, wa], &[f.gqkv, f.gz, f.gb, f.ga], t)
            .map_err(Q4Error::Io)?;
    }
    sync_mark(acc, "gdn.mm_group", f.gqkv)?;
    if il < 4 {
        frame_ck(acc, f.gqkv, conv_ch, t, &format!("L{il}.gqkv"));
    }
    // β/e^g
    let dtb = f.consts[&format!("blk.{il}.dt_bias")];
    let ssa = f.consts[&format!("blk.{il}.ssm_a")];
    if !stage_skipped("gdn.betag") {
        op(acc, FrameOp::GdnBetaG { b: f.gb, a: f.ga, dtb, sa: ssa, bg: f.gbg, n_h: hp.dt_rank * t })?;
    }
    sync_mark(acc, "gdn.betag", f.gbg)?;
    if il < 4 {
        frame_ck(acc, f.gbg, hp.dt_rank * 2, t, &format!("L{il}.gbg"));
    }
    // conv + ring
    let cw = f.consts[&format!("blk.{il}.conv_w")];
    if !stage_skipped("gdn.conv") {
        op(acc, FrameOp::GdnConv { qkv: f.gqkv, cw, state: f.st_conv[seq][ri], out: f.gconv, ch: conv_ch, k: hp.conv_k, t_len: t })?;
        if il == 0 && llm170_diag::dump::opts().bufhash {
            buf_hash(acc, f.gconv, conv_ch * t.min(16), "G0.conv");
        }
        if il < 4 {
            frame_ck(acc, f.gconv, conv_ch, t, &format!("L{il}.gdn_conv"));
        }
    }
    sync_mark(acc, "gdn.conv", f.gconv)?;
    // q/k/v 분할 (토큰 배치 = split3) + l2 + q·scale
    if !stage_skipped("gdn.l2") {
        op(acc, FrameOp::Split3 { src: f.gconv, d0: f.gq, d1: f.gk, d2: f.gv, n0: k_len, n1: k_len, n2: v_len })?;
        op(acc, FrameOp::L2Rows { x: f.gq, eps, d: hp.d_state, n: k_len * t })?;
        op(acc, FrameOp::L2Rows { x: f.gk, eps, d: hp.d_state, n: k_len * t })?;
        let scale = 1.0f32 / (hp.d_state as f32).sqrt();
        op(acc, FrameOp::Scale { t: f.gq, s: scale, n: k_len * t })?;
    }
    sync_mark(acc, "gdn.l2scale", f.gq)?;
    if il == 0 && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, f.gq, k_len * t.min(16), "G0.gq");
        buf_hash(acc, f.gk, k_len * t.min(16), "G0.gk");
        buf_hash(acc, f.gbg, hp.dt_rank * 2 * t.min(16), "G0.bgb");
    }
    // AR 상태 갱신 — 상태 GPU 상주, 판독 없음
    let fs: &dyn FrameState = acc;
    if il == 0 {
        dbg("gbg_post", acc, f.gbg, hp.dt_rank * 2 * t);
    }
    if !stage_skipped("gdn.ar") {
        fs.frame_gdn_ar(f.gq, f.gk, f.gv, f.gbg, f.st_gdn[seq][ri], f.go, 1, hp.n_group, hp.dt_rank, hp.d_state)
            .map_err(Q4Error::Io)?;
        if il < 4 {
            frame_ck(acc, f.go, v_len, t, &format!("L{il}.gdn_ar"));
            // 이월 상태(carry) — conv 링과 AR 상태가 청크 간 동일하게 유지되는지.
            // 입력이 모두 비트 동일한데 AR 출력이 갈리는 경우 이 둘이 유일한 미지수다.
            frame_ck(acc, f.st_conv[seq][ri], (hp.conv_k - 1) * conv_ch, 1, &format!("L{il}.st_conv"));
            frame_ck(acc, f.st_gdn[seq][ri], hp.dt_rank * hp.d_state * hp.d_state, 1, &format!("L{il}.st_gdn"));
        }
    }
    sync_mark(acc, "gdn.ar", f.go)?;
    if il == 0 && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, f.go, v_len * t.min(16), "G0.go");
    }
    if std::env::var_os("LLM170_NP_DBG").is_some() && il == 0 {
        let mut v = vec![0.0f32; v_len];
        if acc.frame_read(f.go, &mut v).is_ok() {
            eprintln!("# npdbg(ar_seq): sum={:.6}", v.iter().map(|&x| x as f64).sum::<f64>());
        }
    }
    // norm_gated + out proj
    let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
    if !stage_skipped("gdn.ng") {
        op(acc, FrameOp::NormGated { o: f.go, z: f.gz, w: snorm, out: f.ggated, eps, d: hp.d_state, n_h: hp.dt_rank })?;
    }
    sync_mark(acc, "gdn.normgated", f.ggated)?;
    if il == 0 && llm170_diag::dump::opts().bufhash {
        buf_hash(acc, f.ggated, hp.d_state * hp.dt_rank * t.min(16), "G0.ng");
    }
    let wout = model.w4(&format!("blk.{il}.ssm_out.weight"))?;
    if !stage_skipped("gdn.out") {
        acc.frame_mm(f.ggated, &wout, f.ffn_out, t).map_err(Q4Error::Io)?;
    }
    sync_mark(acc, "gdn.out", f.ffn_out)?;
    Ok(())
}



/// MoE 프레임 — stages/moe.rs t=1 경로와 동일 수식.
/// 합산 순서 차이: 전문가 가중합을 ids 순(확률 내림차순)으로 누산 —
/// CPU는 전문가 id 오름차순. f32 10항 합의 순서 차이 (~1e-7) — 기존
/// GPU GEMM 재정렬 편차(5e-3)보다 4자리 작아 매트릭스로 검증.
pub(super) fn moe_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    n: usize,
    t: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let k_sel = hp.n_expert_used;
    let n_ff = hp.n_ff_exp;
    // route + shared 게이트
    let w_route = model.w4(&format!("blk.{il}.ffn_gate_inp.weight"))?;
    let w_route_sh = model.w4(&format!("blk.{il}.ffn_gate_inp_shexp.weight"))?;
    acc.frame_mm_group(f.mix, &[w_route, w_route_sh], &[f.mroute, f.msgate], t)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "moe.route", f.mroute)?;
    let _ = stage_skipped("moe.route");
    if !stage_skipped("moe.top10") {
        op(acc, FrameOp::MoeTop10 { route: f.mroute, ids: f.mids, wt: f.mwt, n_exp: hp.n_expert, k_sel })?;
    }
    sync_mark(acc, "moe.top10", f.mids)?;
    if il < 4 {
        frame_ck(acc, f.mids, k_sel, t, &format!("L{il}.mids"));
        frame_ck(acc, f.mwt, k_sel, t, &format!("L{il}.mwt"));
    }
    let fs: &dyn FrameState = acc;
    let w_gate = model.w4(&format!("blk.{il}.ffn_gate_exps.weight"))?;
    let w_up = model.w4(&format!("blk.{il}.ffn_up_exps.weight"))?;
    let w_down = model.w4(&format!("blk.{il}.ffn_down_exps.weight"))?;
    if t == 1 {
        // 디코드: mix를 k_sel행 브로드캐스트 — 전용 커널 1런치(기존 k_sel런치).
        op(acc, FrameOp::BcastRows { src: f.mix, dst: f.mxsel, n, rows: k_sel })?;
        fs.frame_moe_gemm(f.mxsel, &w_gate, f.mids, f.mgu, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        fs.frame_moe_gemm(f.mxsel, &w_up, f.mids, f.mup, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluMul { g: f.mgu, u: f.mup, out: f.mglu, n: k_sel * n_ff })?;
        fs.frame_moe_gemm(f.mglu, &w_down, f.mids, f.my, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(acc, FrameOp::MoeWeightedSum { ys: f.my, wt: f.mwt, out: f.mout, k: k_sel, n })?;
    } else {
        // 프리필: (토큰,전문가) 페어 행 gather → 3회 스택 GEMM → scatter
        fs.frame_moe_gather(f.mix, f.mxsel, n, k_sel, t).map_err(Q4Error::Io)?;
        if il <= 3 && llm170_diag::dump::opts().bufhash {
            // plans/84 E.2: gather 시점 mix/mxsel — 과도 현상의 소스 분리.
            buf_hash(acc, f.mix, n * t.min(16), &format!("L{il}D.mix_at_gather"));
            buf_hash(acc, f.mxsel, n * k_sel * t.min(16), &format!("L{il}D.mxsel_after_gather"));
        }
        if std::env::var_os("LLM170_MOE_GATHER2").is_some() {
            // 진단(plans/80): gather 2회 — 멱등 쓰기라 결과 불변이어야 한다.
            // 2회째에 x가 바르게 되면 첫 쓰기가 찢어진 것, 그대로면 이웃 오염.
            fs.frame_moe_gather(f.mix, f.mxsel, n, k_sel, t).map_err(Q4Error::Io)?;
        }
        fs.frame_moe_gemm(f.mxsel, &w_gate, f.mids, f.mgu, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        if il < 4 {
            frame_ck(acc, f.mgu, n_ff, t * k_sel, &format!("L{il}.mgu"));
        }
        fs.frame_moe_gemm(f.mxsel, &w_up, f.mids, f.mup, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluMul { g: f.mgu, u: f.mup, out: f.mglu, n: t * k_sel * n_ff })?;
        fs.frame_moe_gemm(f.mglu, &w_down, f.mids, f.my, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        sync_mark(acc, "moe.gemm3", f.my)?;
        fs.frame_moe_scatter(f.my, f.mwt, f.mout, k_sel, n, t).map_err(Q4Error::Io)?;
        sync_mark(acc, "moe.scatter", f.mout)?;
    }
    // shared 전문가 — σ(sgate)·shout 가산
        if il < 4 {
            frame_ck(acc, f.mout, n, t, &format!("L{il}.moe_sc"));
        }
    if !stage_skipped("moe.shared") {
        let shg_w = model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
        let shu_w = model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
        let shd_w = model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
        // plans/72: t=1은 융합 2런치(gate+up+silu → down+sigmoid·axpy).
        // 기존 8런치(quant×2+gemv×3+sigmoid+silu+axpy)가 19.4ms/step의
        // 지배 항이었다 — 런치 오버헤프 지배(유효 대역폭 1.5GB/s).
        if t == 1 {
            op(acc, FrameOp::Sigmoid { t: f.msgate, n: t })?;
            acc.shexp_gu(f.mix, &shg_w, &shu_w, f.shglu, n, n_ff)
                .map_err(Q4Error::Io)?;
            acc.shexp_da(f.shglu, &shd_w, f.msgate, f.mout, n, n_ff)
                .map_err(Q4Error::Io)?;
        } else if let Some(vv2) = f.np_views.as_ref().filter(|v| t <= v.mix.len()) {
            // plans/74: np 배치판도 공유전문가는 **행별 융합 2런치** — 일반
            // GEMM+SiluMul 경로와 융합 커널의 산술이 미세히 달라 토큰이 갈라
            // 진다(실측). 행별 융합으로 per-row 경로와 비트동일 유지.
            // 프리필(np_views 없음)은 일반 배치 경로 유지.
            // (2026-09-17) t > 뷰 행수(프리필 청크)면 배치 경로로 내린다 —
            // 종전엔 앞 8행만 공유전문가를 받고 나머지 행이 조용히 누락됐다.
            let rows_avail = vv2.mix.len().min(t);
            op(acc, FrameOp::Sigmoid { t: f.msgate, n: rows_avail })?;
            for row in 0..rows_avail {
                acc.shexp_gu(vv2.mix[row], &shg_w, &shu_w, f.shglu, n, n_ff)
                    .map_err(Q4Error::Io)?;
                let sg_view = acc
                    .frame_slice(f.msgate, row, 1)
                    .map_err(Q4Error::Io)?;
                acc.shexp_da(f.shglu, &shd_w, sg_view, vv2.mout[row], n, n_ff)
                    .map_err(Q4Error::Io)?;
            }
        } else {
            op(acc, FrameOp::Sigmoid { t: f.msgate, n: t })?;
            acc.frame_mm_group(f.mix, &[shg_w, shu_w], &[f.shg, f.shu], t)
                .map_err(Q4Error::Io)?;
            op(acc, FrameOp::SiluMul { g: f.shg, u: f.shu, out: f.shglu, n: n_ff * t })?;
            acc.frame_mm(f.shglu, &shd_w, f.shout, t).map_err(Q4Error::Io)?;
            if il <= 2 && llm170_diag::dump::opts().bufhash {
                buf_hash(acc, f.shg, n_ff * t.min(16), &format!("L{il}A.shg"));
                buf_hash(acc, f.shout, n_ff * t.min(16), &format!("L{il}A.shout"));
                buf_hash(acc, f.msgate, t.min(16), &format!("L{il}A.msgate"));
            }
            op(acc, FrameOp::AxpyScaled { y: f.mout, x: f.shout, s: f.msgate, n: n * t })?;
        }
    }
    sync_mark(acc, "moe.shared", f.mout)?;
    Ok(())
}
