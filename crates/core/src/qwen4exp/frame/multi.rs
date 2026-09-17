//! frame/multi — 다중 시퀀스 청크 프리필 (plans/76, plans/79 A).

use super::*;


/// 청크 프리필 행 대역 뷰 — 기하 (n_seq, per_seq)가 바뀔 때만 재생성한다.
/// frame_slice 핸들은 반납되지 않으므로(ADR-0014) 스텝마다 만들면 테이블이
/// 무한히 큰다 — 청크 크기 종류 수만큼만 늘어나게 고정한다.
#[allow(clippy::too_many_arguments)]
pub(super) fn ensure_pre_views(
    acc: &dyn Accelerator,
    f: &mut Frame4,
    slots: usize,
    per_seq: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    n: usize,
    hc: usize,
    hp: &Hparams4,
) -> Result<(), Q4Error> {
    if f.pre_views
        .as_ref()
        .is_some_and(|v| v.slots == slots && v.rows == per_seq)
    {
        return Ok(());
    }
    let dt2 = hp.dt_rank * 2;
    let qrow = hp.n_head * 2 * hp.head_dim;
    let kvrow = hp.n_kv * hp.head_dim;
    let iqrow = hp.idx_heads * hp.idx_dim;
    let arow = hp.n_head * hp.head_dim;
    let mk = |acc: &dyn Accelerator, tag: &str, h: u64, row_len: usize| -> Result<Vec<u64>, Q4Error> {
        (0..slots)
            .map(|s| {
                acc.frame_slice(h, s * per_seq * row_len, per_seq * row_len)
                    .map_err(|e| {
                        Q4Error::Io(format!("pre-view {tag} s={s}/{slots} rows={per_seq}: {e}"))
                    })
            })
            .collect()
    };
    let v = PreViews {
        slots,
        rows: per_seq,
        res_hc: mk(acc, "res_hc", f.res_hc, hc * n)?,
        gqkv: mk(acc, "gqkv", f.gqkv, conv_ch)?,
        gconv: mk(acc, "gconv", f.gconv, conv_ch)?,
        gq: mk(acc, "gq", f.gq, k_len)?,
        gk: mk(acc, "gk", f.gk, k_len)?,
        gv: mk(acc, "gv", f.gv, v_len)?,
        gbg: mk(acc, "gbg", f.gbg, dt2)?,
        go: mk(acc, "go", f.go, v_len)?,
        qsa_q: mk(acc, "qsa_q", f.qsa_q, qrow)?,
        qsa_k: mk(acc, "qsa_k", f.qsa_k, kvrow)?,
        qsa_v: mk(acc, "qsa_v", f.qsa_v, kvrow)?,
        qsa_iq: mk(acc, "qsa_iq", f.qsa_iq, iqrow)?,
        qsa_ik: mk(acc, "qsa_ik", f.qsa_ik, hp.idx_dim)?,
        qsa_attn: mk(acc, "qsa_attn", f.qsa_attn, arow)?,
        mix: mk(acc, "mix", f.mix, n)?,
        ffn_out: mk(acc, "ffn_out", f.ffn_out, n)?,
        hin_last: (0..slots)
            .map(|s| {
                acc.frame_slice(f.hin, (s * per_seq + per_seq - 1) * n, n)
                    .map_err(Q4Error::Io)
            })
            .collect::<Result<Vec<u64>, Q4Error>>()?,
        logits: (0..slots)
            .map(|s| {
                acc.frame_slice(f.logits_t, s * hp.vocab, hp.vocab)
                    .map_err(Q4Error::Io)
            })
            .collect::<Result<Vec<u64>, Q4Error>>()?,
    };
    f.pre_views = Some(Box::new(v));
    Ok(())
}


// ─────────────────────────────────────────────────────────────────────────────
// plans/76(청크 프리필 배치): 다중 시퀀스 청크 프리필 (2026-09-17)
//
// 문제: 프리필은 청크마다 **전체 무게 1회 읽기**가 고정비다(FN 104GiB ≈ 0.4s
// 하한). 슬롯 4개가 함께 도착하면 같은 읽기를 4번 하게 되고, 이것이 np4 셀
// 0.86x 의 주범이었다(docs/benchmarks.md "Why np4 loses" 2항).
//
// 구조: dense op(mm_group / hc / MoE / head GEMM)는 t_total = n_seq·per_seq
// 행을 **1회**로 처리해 무게 읽기를 공유한다. 시퀀스 소유 상태(GDN conv 링·AR,
// QSA rope·선택·KV·어텐션, PLE 해시·게이트·conv)는 seq별 행 대역(pre_views)으로
// per_seq행씩 실행한다 — 산술 순서가 단일 시퀀스 프리필과 동일하므로 배치
// 결과는 시퀀스별 독립 프리필과 토큰 일치해야 한다
// (검증: qwen4exp::tests::prefill_multi_matches_sequential).
// ─────────────────────────────────────────────────────────────────────────────

/// GDN 프레임 — 다중 시퀀스 청크 프리필. 무게 공유 구간(투영·β/e^g·분할·L2·
/// scale·norm_gated·out)은 t_total행 **1회**, 상태 구간(conv 링·AR)은 seq별
/// 행 대역 + per_seq행 **사슬 1호출**이다 — 같은 시퀀스의 행을 행별 독립
/// 호출로 쪼개면 순환 상태가 경합해 순차 결과와 갈라진다(프리필 커널을
/// t_len=per_seq로 그대로 쓴다).
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_frame_pre(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    seqs: &[usize],
    ri: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    eps: f32,
    t: usize,
    per_seq: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let wqkv = model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
    let wz = model.w4(&format!("blk.{il}.attn_gate.weight"))?;
    let wb = model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
    let wa = model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
    if !stage_skipped("gdn.mm") {
        acc.frame_mm_group(f.mix, &[wqkv, wz, wb, wa], &[f.gqkv, f.gz, f.gb, f.ga], t)
            .map_err(Q4Error::Io)?;
    }
    // β/e^g — n_h = dt_rank·t 이고 커널은 dr = n_h/cur_t 로 dtb/sa 길이를 유추한다.
    let dtb = f.consts[&format!("blk.{il}.dt_bias")];
    let ssa = f.consts[&format!("blk.{il}.ssm_a")];
    if !stage_skipped("gdn.betag") {
        op(acc, FrameOp::GdnBetaG { b: f.gb, a: f.ga, dtb, sa: ssa, bg: f.gbg, n_h: hp.dt_rank * t })?;
    }
    // 상태 구간 ①: conv 링 + silu — 시퀀스별 사슬(공유 구간은 아래에서 복귀).
    let cw = f.consts[&format!("blk.{il}.conv_w")];
    fs_begin(acc, per_seq);
    {
        let pv = f.pre_views.as_ref().unwrap();
        for (si, &sq) in seqs.iter().enumerate() {
            if !stage_skipped("gdn.conv") {
                op(acc, FrameOp::GdnConv {
                    qkv: pv.gqkv[si],
                    cw,
                    state: f.st_conv[sq][ri],
                    out: pv.gconv[si],
                    ch: conv_ch,
                    k: hp.conv_k,
                    t_len: per_seq,
                })?;
            }
        }
    }
    fs_begin(acc, t);
    if !stage_skipped("gdn.l2") {
        op(acc, FrameOp::Split3 { src: f.gconv, d0: f.gq, d1: f.gk, d2: f.gv, n0: k_len, n1: k_len, n2: v_len })?;
        op(acc, FrameOp::L2Rows { x: f.gq, eps, d: hp.d_state, n: k_len * t })?;
        op(acc, FrameOp::L2Rows { x: f.gk, eps, d: hp.d_state, n: k_len * t })?;
        let scale = 1.0f32 / (hp.d_state as f32).sqrt();
        op(acc, FrameOp::Scale { t: f.gq, s: scale, n: k_len * t })?;
    }
    // 상태 구간 ②: AR — 시퀀스별 사슬(행 수는 frame_begin이 정한다).
    let fs: &dyn FrameState = acc;
    fs_begin(acc, per_seq);
    {
        let pv = f.pre_views.as_ref().unwrap();
        for (si, &sq) in seqs.iter().enumerate() {
            if !stage_skipped("gdn.ar") {
                fs.frame_gdn_ar(
                    pv.gq[si], pv.gk[si], pv.gv[si], pv.gbg[si], f.st_gdn[sq][ri], pv.go[si],
                    1, hp.n_group, hp.dt_rank, hp.d_state,
                )
                .map_err(Q4Error::Io)?;
            }
        }
    }
    fs_begin(acc, t);
    let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
    if !stage_skipped("gdn.ng") {
        op(acc, FrameOp::NormGated { o: f.go, z: f.gz, w: snorm, out: f.ggated, eps, d: hp.d_state, n_h: hp.dt_rank })?;
    }
    let wout = model.w4(&format!("blk.{il}.ssm_out.weight"))?;
    if !stage_skipped("gdn.out") {
        acc.frame_mm(f.ggated, &wout, f.ffn_out, t).map_err(Q4Error::Io)?;
    }
    Ok(())
}

/// 다중 시퀀스 청크 프리필 — 각 seq의 per_seq 토큰을 한 forward로 처리한다.
/// 행 배치는 seq-major(seq0의 per_seq행, seq1의 ... ) — dense op는 t_total = n_seq*per_seq
/// 로 한 번에, 상태 op(conv/AR/rope/KV/attention/PLE)는 seq별 슬라이스 핸들로 per_seq행씩.
///
/// 호출부 계약(단일 시퀀스 프리필과 동일): 포워드는 `pos`를 만지지 않는다 —
/// 호출 전 `f.dirty[seq]`면 `sync_states`, 반환 후 `seq_sts[seq].pos += per_seq`.
/// QSA 디바이스 풀은 (층, seq) 키라 같은 seq 인덱스를 다른 슬롯과 공유하면 안 된다.
///
/// 게이트: **기본 off** — `LLM170_PREFILL_MULTI=1`일 때만 돈다. 배치는 dense op의
/// 행 수가 n_seq배가 되므로 백엔드의 행 수 의존 커널 선택·환원 순서가 단일
/// 프리필과 달라진다(2026-09-17 실측: 같은 프롬프트로도 logit 최대차 O(0.5),
/// 근접 타이는 뒤집힘 — 4×128에서 4행 중 1행). 이는 기존 단일 경로가 청크
/// 크기만 바꿔도 겪는 것과 같은 계열이고 그쪽이 더 크다(1×256 tok=271 vs
/// 4×64 tok=1375, 최대차 9.1). 서비스 채택 전에 드리프트 정책(재베이스라인
/// 또는 백엔드 커널 정합)이 필요해 옵트인으로 둔다. 구조 자체는 비트 동일이다:
/// n_seq=1이면 단일 프리필과 전 층 활성 해시가 일치한다(검증:
/// crates/core/tests/prefill_multi.rs).
pub fn frame_forward_prefill_multi(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seqs: &[usize],
    seq_sts: &mut [SeqState4],
    f: &mut Frame4,
    tokens: &[u32],
    per_seq: usize,
) -> Result<Vec<u32>, Q4Error> {
    if std::env::var("LLM170_PREFILL_MULTI").map(|v| v == "0").unwrap_or(true) {
        return Err(Q4Error::Io(
            "frame_forward_prefill_multi: 게이트 off (LLM170_PREFILL_MULTI=1 로 켠다)".into(),
        ));
    }
    let hp: &Hparams4 = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let k_len = hp.n_group * hp.d_state;
    let v_len = hp.dt_rank * hp.d_state;
    let conv_ch = 2 * k_len + v_len;
    let eps = hp.eps;
    let n_seq = seqs.len();
    let t = n_seq * per_seq;
    if per_seq == 0 || tokens.len() != t {
        return Err(Q4Error::Io(format!(
            "frame_forward_prefill_multi: 토큰 계약 위반 n_seq={n_seq} per_seq={per_seq} tokens={}",
            tokens.len()
        )));
    }
    if n_seq > 8 || t > f.t_max {
        return Err(Q4Error::Io(format!(
            "frame_forward_prefill_multi: 용량 초과 n_seq={n_seq}(≤8) t={t} > t_max={}",
            f.t_max
        )));
    }
    if seqs.iter().enumerate().any(|(i, &s)| seqs[..i].contains(&s)) {
        return Err(Q4Error::Io("frame_forward_prefill_multi: seq 중복(상태 핸들 겹침)".into()));
    }
    let t_call = std::time::Instant::now();
    fs_begin(acc, t);
    ensure_pre_views(acc, f, n_seq, per_seq, conv_ch, k_len, v_len, n, hc, hp)?;

    // 0) 임베딩 — seq-major t행 → hc 스트림 방송 (단일 시퀀스와 동일 산술)
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

    // PLE n-gram 행 — 시퀀스별 호스트 해시(각자 pos·이력·링을 소유)
    let ple_rows: Vec<Vec<u32>> = if hp.is_ple(1) {
        seqs.iter()
            .enumerate()
            .map(|(si, &sq)| {
                stages::ple_hash(ctx, &mut seq_sts[sq], &tokens[si * per_seq..(si + 1) * per_seq])
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut recr_idx = 0usize;
    let mut full_idx = 0usize;
    for il in 0..hp.n_layer {
        // 1) PLE — 호스트 브리지 시퀀스별(해시는 위에서 끝냈고 key/value 투영·
        //    게이트·conv·잔차는 시퀀스 상태라 per_seq행씩). 단일 시퀀스 프리필
        //    (t>1)과 같은 브리지·같은 산술 순서.
        if hp.is_ple(il) {
            let mut r = vec![0.0f32; per_seq * hc * n];
            for (si, &sq) in seqs.iter().enumerate() {
                let h = f.pre_views.as_ref().unwrap().res_hc[si];
                acc.capture_mark("ple_in").map_err(Q4Error::Io)?;
                acc.frame_read(h, &mut r).map_err(Q4Error::Io)?;
                let mut rows: Vec<Vec<f32>> = r.chunks_exact(hc * n).map(|c| c.to_vec()).collect();
                stages::ple_block(ctx, &mut seq_sts[sq], il, &mut rows, &ple_rows[si], None)?;
                let flat: Vec<f32> = rows.concat();
                acc.frame_write(h, &flat).map_err(Q4Error::Io)?;
                acc.capture_mark("ple_out").map_err(Q4Error::Io)?;
                sync_mark(acc, "hc.ple_bridge", h)?;
            }
        }

        // 2) hc attn mix — t_total 공유
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc, t)?;
        sync_mark(acc, &format!("pre{il}.hc_attn"), f.mix)?;

        // 3) attention — GDN(공유 1회 + 상태만 seq별) / QSA(seq별 디바이스 경로)
        if hp.is_recr(il) {
            if !stage_skipped("gdn") {
                gdn_frame_pre(acc, model, f, il, seqs, recr_idx, conv_ch, k_len, v_len, eps, t, per_seq)?;
            }
            recr_idx += 1;
            sync_mark(acc, &format!("pre{il}.gdn"), f.ffn_out)?;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
        } else {
            if !stage_skipped("qsa") {
                for (si, &sq) in seqs.iter().enumerate() {
                    let b = {
                        let pv = f.pre_views.as_ref().unwrap();
                        QsaBufs {
                            mix: pv.mix[si],
                            q: pv.qsa_q[si],
                            k: pv.qsa_k[si],
                            v: pv.qsa_v[si],
                            iq: pv.qsa_iq[si],
                            ik: pv.qsa_ik[si],
                            attn: pv.qsa_attn[si],
                            out: pv.ffn_out[si],
                        }
                    };
                    qsa_frame(acc, model, ctx, &mut seq_sts[sq], f, il, per_seq, full_idx, sq, &b)?;
                }
                acc.capture_mark("recr_out").map_err(Q4Error::Io)?;
            }
            full_idx += 1;
            sync_mark(acc, &format!("pre{il}.qsa"), f.ffn_out)?;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
        }

        // 4) hc ffn mix(공유) + MoE(공유 배치 — (토큰,전문가) gather 1회. 행별
        //    t=1 판은 무게 재사용이 없어 청크 프리필의 취지와 반대다)
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc, t)?;
        sync_mark(acc, &format!("pre{il}.hc_ffn"), f.mix)?;
        moe_frame(acc, model, f, il, n, t)?;
        sync_mark(acc, &format!("pre{il}.moe"), f.mout)?;
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
    }

    // 5) head — output hc mix(전 행) → 시퀀스별 **마지막 행** GEMM → 행별 argmax
    {
        fs_begin(acc, t);
        let w_norm = f.consts["output_hc_norm"];
        op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.hxn, eps, n, w_reps: hc })?;
        let w_down = model.w4("output_hc_down.weight")?;
        acc.frame_mm(f.hxn, &w_down, f.hlo, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluDiv { t: f.hlo, div: hc as f32, n: f.hlo_len * t })?;
        let w_up = model.w4("output_hc_up.weight")?;
        acc.frame_mm(f.hlo, &w_up, f.hgate, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::HcGateMean { xn: f.hxn, gate: f.hgate, out: f.hin, hc, n })?;
        let wout = model.w("output.weight").ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        // 마지막 행 판정 — 단일 시퀀스(프리필 t>1)와 같은 t=1 GEMM 경로를 쓴다
        // (np 배치 head의 t행 GEMM과 산술이 다르다 — 프리필 등가성은 이쪽).
        for si in 0..n_seq {
            let (hs, ls) = {
                let pv = f.pre_views.as_ref().unwrap();
                (pv.hin_last[si], pv.logits[si])
            };
            acc.frame_mm(hs, &wout, ls, 1).map_err(Q4Error::Io)?;
        }
        let toks = acc.frame_argmax_rows(f.logits_t, n_seq, hp.vocab).map_err(Q4Error::Io)?;
        ftime_report(t);
        if ftime_on() {
            eprintln!(
                "# pre-frame-total n_seq={n_seq} per_seq={per_seq} {:.1}ms",
                t_call.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(toks)
    }
}
