//! frame/np — np 배치 디코드 포워드 (plans/73, plans/79 A).

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// plans/73(np): 다중 시퀀스 배치 디코드 (2026-09-16)
//
// 구조: 무게 스트리밍(mm_group/hc/MoE/head GEMM)은 t=n_seqs 행이 **공유**하고,
// 시퀀스 소유 상태(GDN conv 링·AR, QSA rope·선택·KV·어텐션, PLE)는 행 뷰
// (frame_slice)로 per-seq t=1 실행한다. 상태 op는 기존 t=1 커널·기존 산술
// 순서 그대로(행별 독립)라 시퀀스 격리 불변식이 성립한다 — 배치 결과는 순차
// decode1과 토큰 일치해야 한다(검증: infer 다중 프롬프트).
// ─────────────────────────────────────────────────────────────────────────────

/// 행 뷰를 처음 한 번만 만든다(이후 스텝 재사용 — frames 테이블 무한 증가 방지).
#[allow(clippy::too_many_arguments)]
pub(super) fn ensure_np_views(
    acc: &dyn Accelerator,
    f: &mut Frame4,
    rows: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    n: usize,
    hc: usize,
    hp: &Hparams4,
) -> Result<(), Q4Error> {
    if f.np_views.is_some() {
        return Ok(());
    }
    // 뷰는 항상 최대 슬롯으로 만든다 — 첫 배치가 2슬롯이어도 이후 4슬롯
    // 스텝이 같은 뷰 테이블을 쓴다(패닉 방지, frames 테이블 무한 증가 방지).
    // plans/113(sglang P0-2): 8→16 — serve --slots가 1..16인데 t>8 에러가
    // decode_batch의 프레임 파괴+매 스텝 직렬 폴백을 유발했다. 뷰는 핸들만
    // 증가(공유 버퍼는 t_max≥512 예산 내)·산술은 행별 독립이라 비트 불변.
    const NP_MAX: usize = 16;
    let rows = rows.max(NP_MAX);
    let dt2 = hp.dt_rank * 2;
    let qrow = hp.n_head * 2 * hp.head_dim;
    let kvrow = hp.n_kv * hp.head_dim;
    let iqrow = hp.idx_heads * hp.idx_dim;
    let arow = hp.n_head * hp.head_dim;
    let mk = |acc: &dyn Accelerator, h: u64, row_len: usize| -> Result<Vec<u64>, Q4Error> {
        (0..rows)
            .map(|r| {
                acc.frame_slice(h, r * row_len, row_len)
                    .map_err(Q4Error::Io)
            })
            .collect()
    };
    let v = NpViews {
        res_hc: mk(acc, f.res_hc, hc * n)?,
        gqkv: mk(acc, f.gqkv, conv_ch)?,
        gconv: mk(acc, f.gconv, conv_ch)?,
        gq: mk(acc, f.gq, k_len)?,
        gk: mk(acc, f.gk, k_len)?,
        gv: mk(acc, f.gv, v_len)?,
        gbg: mk(acc, f.gbg, dt2)?,
        go: mk(acc, f.go, v_len)?,
        qsa_q: mk(acc, f.qsa_q, qrow)?,
        qsa_k: mk(acc, f.qsa_k, kvrow)?,
        qsa_v: mk(acc, f.qsa_v, kvrow)?,
        qsa_iq: mk(acc, f.qsa_iq, iqrow)?,
        qsa_ik: mk(acc, f.qsa_ik, hp.idx_dim)?,
        qsa_attn: mk(acc, f.qsa_attn, arow)?,
        mix: mk(acc, f.mix, n)?,
        mout: mk(acc, f.mout, n)?,
        ple_key: mk(acc, f.ple_key, hc * n)?,
        ple_value: mk(acc, f.ple_value, n)?,
    };
    f.np_views = Some(Box::new(v));
    Ok(())
}

/// GDN 프레임(np) — mm_group/betag/split/l2/scale/normgated/out은 t=rows 공유,
/// conv와 AR만 per-seq(행 뷰 + 해당 seq 상태).
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_frame_np(
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
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    // 행이 서로 다른 seq일 때만 1런치 conv/AR 경로 — 같은 seq 행(W2 검증
    // 배치)은 상태를 행 간 체인해야 해서 per-row t=1 순차 폴백만이 정확하다
    // (1런치 경로는 행별 독립 상태 전제).
    let mut uniq = seqs.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let distinct = uniq.len() == seqs.len();
    let wqkv = model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
    let wz = model.w4(&format!("blk.{il}.attn_gate.weight"))?;
    let wb = model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
    let wa = model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
    acc.frame_mm_group(f.mix, &[wqkv, wz, wb, wa], &[f.gqkv, f.gz, f.gb, f.ga], t)
        .map_err(Q4Error::Io)?;
    let dtb = f.consts[&format!("blk.{il}.dt_bias")];
    let ssa = f.consts[&format!("blk.{il}.ssm_a")];
    op(
        acc,
        FrameOp::GdnBetaG {
            b: f.gb,
            a: f.ga,
            dtb,
            sa: ssa,
            bg: f.gbg,
            n_h: hp.dt_rank * t,
        },
    )?;
    let cw = f.consts[&format!("blk.{il}.conv_w")];
    let vv = f.np_views.as_ref().unwrap();
    // NP 디버그 프로브(2026-09-16): conv/AR 직후 행0 합계
    let npdbg = llm170_diag::dump::opts().key("np_dbg") && il == 0;
    let psum = |acc: &dyn Accelerator, h: u64, n2: usize, tag: &str| {
        if npdbg {
            let mut v = vec![0.0f32; n2];
            if acc.frame_read(h, &mut v).is_ok() {
                eprintln!(
                    "# npdbg {tag}: sum={:.6} v0={:.6}",
                    v.iter().map(|&x| x as f64).sum::<f64>(),
                    v[0]
                );
            }
        }
    };
    // conv(링) — plans/74 N2: 행별 상태를 포인터 테이블로 1런치. 실패 시
    // 종전 행별 t=1 루프(상태 커널이 t_cur 로 행 수를 유추해 t_cur=1로 내린다).
    // qkv/gconv는 [t][ch] 연속 프레임 버퍼라 정본 핸들 직접.
    let conv_states: Vec<u64> = seqs.iter().map(|&sq| f.st_conv[sq][ri]).collect();
    if distinct
        && acc
            .frame_gdn_conv_np(f.gqkv, f.gconv, &conv_states, cw, conv_ch, hp.conv_k)
            .is_ok()
    {
        fs_begin_np(acc, t);
    } else {
        fs_begin(acc, 1);
        for (row, &sq) in seqs.iter().enumerate() {
            op(
                acc,
                FrameOp::GdnConv {
                    qkv: vv.gqkv[row],
                    cw,
                    state: f.st_conv[sq][ri],
                    out: vv.gconv[row],
                    ch: conv_ch,
                    k: hp.conv_k,
                    t_len: 1,
                },
            )?;
        }
        fs_begin_np(acc, t);
    }
    fs_begin(acc, t); // split/l2/scale는 전 행 배치
    psum(acc, vv.gconv[0], conv_ch, "conv_row0");
    if seqs.len() > 1 {
        psum(acc, vv.gconv[1], conv_ch, "conv_row1");
    }
    // split/l2/scale — 행별 독립 원소연산, t 배치 그대로
    op(
        acc,
        FrameOp::Split3 {
            src: f.gconv,
            d0: f.gq,
            d1: f.gk,
            d2: f.gv,
            n0: k_len,
            n1: k_len,
            n2: v_len,
        },
    )?;
    op(
        acc,
        FrameOp::L2Rows {
            x: f.gq,
            eps,
            d: hp.d_state,
            n: k_len * t,
        },
    )?;
    op(
        acc,
        FrameOp::L2Rows {
            x: f.gk,
            eps,
            d: hp.d_state,
            n: k_len * t,
        },
    )?;
    let scale = 1.0f32 / (hp.d_state as f32).sqrt();
    op(
        acc,
        FrameOp::Scale {
            t: f.gq,
            s: scale,
            n: k_len * t,
        },
    )?;
    // AR(상태) — per-seq t=1 (다시 내림)
    // AR(상태) — plans/74 N2: 행별 상태 테이블 1런치. 실패 시 종전 행별 t=1.
    let ar_states: Vec<u64> = seqs.iter().map(|&sq| f.st_gdn[sq][ri]).collect();
    let fs: &dyn FrameState = acc;
    if distinct
        && acc
            .frame_gdn_ar_np(
                f.gq, f.gk, f.gv, f.gbg, f.go, &ar_states, hp.n_group, hp.dt_rank, hp.d_state,
            )
            .is_ok()
    {
        // 1런치 경로 사용
    } else {
        fs_begin(acc, 1);
        for (row, &sq) in seqs.iter().enumerate() {
            fs.frame_gdn_ar(
                vv.gq[row],
                vv.gk[row],
                vv.gv[row],
                vv.gbg[row],
                f.st_gdn[sq][ri],
                vv.go[row],
                1,
                hp.n_group,
                hp.dt_rank,
                hp.d_state,
            )
            .map_err(Q4Error::Io)?;
        }
    }
    if seqs.len() > 1 {
        psum(acc, vv.gq[0], k_len, "ar_in_q0");
        psum(acc, vv.gq[1], k_len, "ar_in_q1");
        psum(acc, vv.gbg[0], hp.dt_rank * 2, "ar_in_bg0");
        psum(acc, vv.gbg[1], hp.dt_rank * 2, "ar_in_bg1");
    }
    psum(acc, vv.go[0], v_len, "ar_row0");
    if seqs.len() > 1 {
        psum(acc, vv.go[1], v_len, "ar_row1");
    }
    fs_begin(acc, t); // 공유 구간 복귀
    let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
    op(
        acc,
        FrameOp::NormGated {
            o: f.go,
            z: f.gz,
            w: snorm,
            out: f.ggated,
            eps,
            d: hp.d_state,
            n_h: hp.dt_rank,
        },
    )?;
    let wout = model.w4(&format!("blk.{il}.ssm_out.weight"))?;
    acc.frame_mm(f.ggated, &wout, f.ffn_out, t)
        .map_err(Q4Error::Io)?;
    psum(acc, f.ffn_out, 64, "ffnout_head");
    Ok(())
}

/// QSA 프레임(np) — 5투영·wo는 t=rows 공유, rope/선택/KV/어텐션은 per-seq.
/// (호스트 선택 폴백은 np 경로에서 지원하지 않는다 — 디바이스 경로 실패 시 Err.)
#[allow(clippy::too_many_arguments)]
pub(super) fn qsa_frame_np(
    acc: &dyn Accelerator,
    model: &Model4,
    seq_sts: &mut [SeqState4],
    seqs: &[usize],
    f: &mut Frame4,
    il: usize,
    t: usize,
    full_idx: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
    let idx_dim = hp.idx_dim;
    let wq = model.w4(&format!("blk.{il}.attn_q.weight"))?;
    let wk = model.w4(&format!("blk.{il}.attn_k.weight"))?;
    let wv = model.w4(&format!("blk.{il}.attn_v.weight"))?;
    let wo = model.w4(&format!("blk.{il}.attn_output.weight"))?;
    let w_iq = model.w4(&format!("blk.{il}.indexer.q_proj.weight"))?;
    let w_ik = model.w4(&format!("blk.{il}.indexer.k_proj.weight"))?;
    acc.frame_mm_group(
        f.mix,
        &[wq, wk, wv, w_iq, w_ik],
        &[f.qsa_q, f.qsa_k, f.qsa_v, f.qsa_iq, f.qsa_ik],
        t,
    )
    .map_err(Q4Error::Io)?;
    let qn_raw = model.f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
    let kn_raw = model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
    let qn: Vec<f32> = qn_raw
        .iter()
        .copied()
        .cycle()
        .take(qn_raw.len() * n_head)
        .collect();
    let kn: Vec<f32> = kn_raw
        .iter()
        .copied()
        .cycle()
        .take(kn_raw.len() * n_kv)
        .collect();
    let iqw = model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
    let ikw = model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
    let kq_scale = hp.kq_scale();
    let r = hp.compress[il] as usize;
    let vv = f.np_views.as_ref().unwrap();
    fs_begin(acc, 1); // per-seq 구간
    // 같은 seq가 여러 행이면(W2 검증 배치) 행 순서대로 위치를 진행 —
    // 서로 다른 seq(np)는 행당 1회라 발생 수가 0이어서 종전과 동일.
    let mut occ: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (row, &sq) in seqs.iter().enumerate() {
        let base = seq_sts[sq].pos as usize;
        let seen = *occ.entry(sq).or_insert(0);
        occ.insert(sq, seen + 1);
        let pos0 = base + seen;
        acc.frame_qk_norm_rope(
            vv.qsa_q[row],
            vv.qsa_k[row],
            &qn,
            &kn,
            &f.qsa_cs,
            hp.eps,
            pos0,
            n_head,
            n_kv,
            hd,
            n_rot,
            1,
        )
        .map_err(Q4Error::Io)?;
        let (sd, od, list_len) = acc
            .qsa_sel_dev(
                full_idx,
                sq,
                vv.qsa_iq[row],
                vv.qsa_ik[row],
                1,
                pos0,
                hp.idx_heads,
                idx_dim,
                r,
                hp.idx_top_k,
                &iqw,
                &ikw,
                &f.qsa_cs_idx,
                hp.eps,
            )
            .map_err(Q4Error::Io)?;
        let (kc, vc) = acc
            .qsa_kv_dev(
                full_idx,
                sq,
                vv.qsa_k[row],
                vv.qsa_v[row],
                1,
                pos0,
                n_kv,
                hd,
            )
            .map_err(Q4Error::Io)?;
        acc.qsa_attention_dev_sel(
            vv.qsa_q[row],
            kc,
            vc,
            sd,
            od,
            list_len,
            kq_scale,
            n_head,
            n_kv,
            hd,
            1,
            vv.qsa_attn[row],
        )
        .map_err(Q4Error::Io)?;
        seq_sts[sq].qsa_host_stale = true;
    }
    fs_begin(acc, t); // 공유 구간 복귀
    acc.frame_mm_group(f.qsa_attn, &[wo], &[f.ffn_out], t)
        .map_err(Q4Error::Io)?;
    Ok(())
}

/// np 배치 디코드 포워드 — seqs/tokens는 1:1, 반환은 seq별 로짓.
#[allow(clippy::too_many_lines)]
pub fn frame_forward_np(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seqs: &[usize],
    seq_sts: &mut [SeqState4],
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<Vec<Vec<f32>>, Q4Error> {
    frame_forward_np_ex(acc, model, ctx, seqs, seq_sts, f, tokens, false, false).map(|(l, _)| l)
}

/// np greedy판 — head 후 전사 대신 GPU argmax, 토큰만 회수 (plans/74 N1).
pub fn frame_forward_np_greedy(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seqs: &[usize],
    seq_sts: &mut [SeqState4],
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<Vec<u32>, Q4Error> {
    frame_forward_np_ex(acc, model, ctx, seqs, seq_sts, f, tokens, true, false).map(|(_, t)| t)
}

/// plans/110 W5 — 다중 슬롯 스펙 라운드 시작용: greedy + pre-mixer res_hc 행
/// export(드래프트 h 입력). 검증 행핀(VERIFY_ROW_PIN)으로 decode1 비트 동일.
pub fn frame_forward_np_greedy_h(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seqs: &[usize],
    seq_sts: &mut [SeqState4],
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<Vec<u32>, Q4Error> {
    frame_forward_np_ex(acc, model, ctx, seqs, seq_sts, f, tokens, true, true).map(|(_, t)| t)
}

#[allow(clippy::too_many_lines)]
pub(super) fn frame_forward_np_ex(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seqs: &[usize],
    seq_sts: &mut [SeqState4],
    f: &mut Frame4,
    tokens: &[u32],
    greedy: bool,
    h_export: bool,
) -> Result<(Vec<Vec<f32>>, Vec<u32>), Q4Error> {
    let hp: &Hparams4 = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let k_len = hp.n_group * hp.d_state;
    let v_len = hp.dt_rank * hp.d_state;
    let conv_ch = 2 * k_len + v_len;
    let eps = hp.eps;
    let t = seqs.len();
    if t > 16 {
        return Err(Q4Error::Io("frame_forward_np: t>16 미지원".into()));
    }
    let t_call = std::time::Instant::now();
    // plans/115 P1 계측 — 스테이지별 적립(스텝 타이머는 np_time이 담당).
    let stage_on = llm170_diag::dump::opts().key("np_stage");
    let mut st = [0.0f64; 4]; // [gdn, qsa, moe, head]
    // plans/115 P5: np 스텝 가드 — 언핀 디스패치 표식(np_mode)을 스텝 전체
    // (조기 return 포함)에서 유지하고 종료 시 해제한다.
    struct NpPinGuard<'a>(&'a dyn Accelerator);
    impl Drop for NpPinGuard<'_> {
        fn drop(&mut self) {
            let fs: &dyn FrameState = self.0;
            fs.frame_end_np();
        }
    }
    let _np_pin = NpPinGuard(acc);
    fs_begin_np(acc, t);
    ensure_np_views(acc, f, t, conv_ch, k_len, v_len, n, hc, hp)?;

    // 0) 임베딩 — 각 seq 토큰 → res_hc [t][hc·n]
    {
        let embd = model
            .w("token_embd.weight")
            .ok_or(Q4Error::MissingTensor("token_embd".into()))?;
        super::emb_broadcast_write(acc, &embd, tokens, f.res_hc, n, hc)?;
    }

    // PLE n-gram 행(per-seq 호스트 해시)
    let ple_rows: Vec<Vec<u32>> = if hp.is_ple(1) {
        seqs.iter()
            .enumerate()
            .map(|(row, &sq)| stages::ple_hash(ctx, &mut seq_sts[sq], &tokens[row..row + 1]))
            .collect()
    } else {
        Vec::new()
    };

    let ck_on = llm170_diag::dump::opts().checksum;
    let ck = |acc: &dyn Accelerator, h: u64, n2: usize, tag: &str| {
        if !ck_on {
            return;
        }
        let mut v = vec![0.0f32; n2];
        if acc.frame_read(h, &mut v).is_ok() {
            let s2: f64 = v.iter().map(|&x| x as f64).sum();
            eprintln!(
                "[npck] {tag} sum={s2:.6} v0={:.6} v1={:.6}",
                v[0],
                v.get(1).copied().unwrap_or(0.0)
            );
        }
    };
    let mut recr_idx = 0usize;
    let mut full_idx = 0usize;
    for il in 0..hp.n_layer {
        // 1) PLE — 배치 gather·투영 + 행별 math (plans/115 P0-4)
        //
        // 종전 per-row 루프(호스트 mmap gather+h2d+투영 1행 × t행)는 np4 스텝에
        // ~87ms(48층 × 행 직렬, 26.8GiB mmap 페이지폴트 포함)를 냈다 — np4
        // 24-27 t/s의 주된 갭. gather(ple_gather_dev 블록 캐시)·투영(mm_group
        // t행)은 행별 독립이라 배치해도 산술 불변; math(디바이스 링 체인)만
        // 행별 t=1을 유지한다. t=1 디코드(forward.rs)·프리필과 동일 구조.
        if hp.is_ple(il) {
            let heads = hp.ple_heads_per_ngram * 2;
            let emb_w = heads * hp.ple_head_dim;
            let w_key = model.w4(&format!("blk.{il}.ple_key.weight"))?;
            let w_value = model.w4(&format!("blk.{il}.ple_value.weight"))?;
            let nk = model.f32_vec4(&format!("blk.{il}.ple_norm_key.weight"))?;
            let nq = model.f32_vec4(&format!("blk.{il}.ple_norm_query.weight"))?;
            let nc = model.f32_vec4(&format!("blk.{il}.ple_norm_conv.weight"))?;
            let cw = model.f32_vec4(&format!("blk.{il}.ple_conv1d.weight"))?;
            let vv = f.np_views.as_ref().unwrap();
            fs_begin_np(acc, t); // 배치 투영 구간
            let all_valid = ple_rows.iter().all(|r| r.len() == heads);
            if all_valid {
                let flat: Vec<u32> = ple_rows.concat();
                let gpu = (|| -> Result<(), Q4Error> {
                    let (tptr, tlen, _tty, thd) = ctx.model.ple_table_view()?;
                    // SAFETY (107 W8): ple_table_view 계약 — tptr..tptr+tlen은 PLE
                    // 테이블 mmap 유효 범위(읽기 전용, 동시 쓰기 없음).
                    let tdata: &[u8] =
                        unsafe { std::slice::from_raw_parts(tptr as *const u8, tlen) };
                    acc.ple_gather_dev(tptr, tdata, &flat, f.ple_emb, thd)
                        .map_err(Q4Error::Io)
                })()
                .is_ok();
                if !gpu {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::PleGgpu);
                    ONCE.call_once(|| eprintln!("# np-ple-ggpu: 실패 — 호스트 폴백"));
                    let mut emb = vec![0.0f32; emb_w * t];
                    for (row, r) in ple_rows.iter().enumerate() {
                        ctx.model
                            .ple_gather(r, &mut emb[row * emb_w..(row + 1) * emb_w])?;
                    }
                    acc.frame_write(f.ple_emb, &emb).map_err(Q4Error::Io)?;
                }
                acc.frame_mm_group(f.ple_emb, &[w_key, w_value], &[f.ple_key, f.ple_value], t)
                    .map_err(Q4Error::Io)?;
            } else {
                // 히스토리 미확정 행(시퀀스 초반) — 종전 의미론: 0행 투영 포함.
                let mut emb = vec![0.0f32; emb_w * t];
                for (row, r) in ple_rows.iter().enumerate() {
                    if r.len() == heads {
                        ctx.model
                            .ple_gather(r, &mut emb[row * emb_w..(row + 1) * emb_w])?;
                    }
                }
                acc.frame_write(f.ple_emb, &emb).map_err(Q4Error::Io)?;
                acc.frame_mm_group(f.ple_emb, &[w_key, w_value], &[f.ple_key, f.ple_value], t)
                    .map_err(Q4Error::Io)?;
            }
            for (row, &sq) in seqs.iter().enumerate() {
                fs_begin(acc, 1); // 행별 math 구간(링 체인)
                acc.ple_math_dev(
                    vv.res_hc[row],
                    vv.ple_key[row],
                    vv.ple_value[row],
                    &nk,
                    &nq,
                    &nc,
                    &cw,
                    f.ple_gated,
                    f.ple_conv_out,
                    f.ple_gate,
                    sq,
                    seq_sts[sq].pos as usize,
                    1,
                    hp.eps,
                    n,
                    hc,
                    hp.ple_conv_k,
                    hp.ple_ngram,
                    (hp.ple_conv_k - 1) * hp.ple_ngram,
                    &seq_sts[sq].ple_conv,
                )
                .map_err(Q4Error::Io)?;
            }
        }

        fs_begin_np(acc, t); // 공유 구간
        // 2) hc attn mix (t 공유)
        if il < 4 {
            ck(acc, f.res_hc, 64, &format!("L{il}.res_in"));
        }
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc, t)?;
        if il < 4 {
            ck(acc, f.mix, 64, &format!("L{il}.mix"));
        }
        sync_mark(acc, &format!("np{il}.hc_attn"), f.mix)?;

        // 3) GDN / QSA
        if hp.is_recr(il) {
            let s0 = std::time::Instant::now();
            gdn_frame_np(
                acc, model, f, il, seqs, recr_idx, conv_ch, k_len, v_len, eps, t,
            )?;
            if il < 4 {
                ck(acc, f.ffn_out, 64, &format!("L{il}.gdn"));
            }
            sync_mark(acc, &format!("np{il}.gdn"), f.ffn_out)?;
            recr_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
            if stage_on {
                st[0] += s0.elapsed().as_secs_f64() * 1e3;
            }
        } else {
            let s0 = std::time::Instant::now();
            qsa_frame_np(acc, model, seq_sts, seqs, f, il, t, full_idx)?;
            if il < 4 {
                ck(acc, f.ffn_out, 64, &format!("L{il}.qsa"));
            }
            sync_mark(acc, &format!("np{il}.qsa"), f.ffn_out)?;
            full_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
            if stage_on {
                st[1] += s0.elapsed().as_secs_f64() * 1e3;
            }
        }

        // 4) hc ffn mix(t 공유) + MoE(행별 t=1 — 산술 불변) + combine(t 공유)
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc, t)?;
        let s2 = std::time::Instant::now();
        // MoE: 기본 행별 t=1(모멘텀 유지 — t 배치 gather 판이 t=4 에서 10ms 느림,
        // 2026-09-16 실측). LLM170_NP_MOE_BATCH=1이면 t 배치(gather — 전문가
        // 가중합 순서 차이로 근접 평탄점 플립 가능, 문서화 tie 등급).
        // plans/74: np 기본 배치 MoE — direct-ids 커널이 rows<=64 에서도
        // 돌아가므로 t·k_sel=40행 1회 GEMM(행별 루프 대비 런치 1/4, 점유율 4배,
        // 산술 비트동일 — q4_moe_scatter 합산순서 = q4_moe_weighted_sum).
        // plans/79 C: NO_MOE_NPB 폐기 — t>1 배치 MoE 확정(행별 t=1은 아래 폴백).
        if t > 1 {
            moe_frame(acc, model, f, il, n, t)?;
        } else {
            moe_frame_np(acc, model, f, il, n, seqs)?;
            // moe_frame_np는 per-seq 구간에서 t_cur를 1로 내리고 seqs.len()으로
            // 되돌린다(자기 안에서 t를 모른다). 헤드의 RmsRows/HcGateMean은
            // t_cur로 행 수를 유추하므로 여기서 실제 행 수 t로 복원해야 한다.
            fs_begin_np(acc, t);
        }
        sync_mark(acc, &format!("np{il}.moe"), f.mout)?;
        if il < 4 {
            ck(acc, f.mout, 64, &format!("L{il}.moe"));
        }
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
        if stage_on {
            st[2] += s2.elapsed().as_secs_f64() * 1e3;
        }
    }
    if ck_on {
        ck(acc, f.res_hc, 64, "head.res");
    }

    if stage_on {
        st[3] = t_call.elapsed().as_secs_f64() * 1e3 - st[0] - st[1] - st[2];
        eprintln!(
            "[npstage] t={t} gdn {:.1} qsa {:.1} moe {:.1} head+기타 {:.1}",
            st[0], st[1], st[2], st[3]
        );
    }
    // 5a) plans/110 W5: pre-mixer res_hc 행 export — 다중 슬롯 스펙의 라운드
    // 시작을 np로 배칭할 때 드래프트 h 입력이 필요하다(mtp_h_export 시만).
    if h_export && f.mtp_h_export {
        let mut rows = vec![0.0f32; t * hc * n];
        acc.frame_read(f.res_hc, &mut rows).map_err(Q4Error::Io)?;
        f.last_res_hc_rows = rows.chunks(hc * n).map(|c| c.to_vec()).collect();
    }

    // 5) head — 전 행 GEMM 1회 → [t][vocab] 판독
    {
        let w_norm = f.consts["output_hc_norm"];
        op(
            acc,
            FrameOp::RmsRows {
                x: f.res_hc,
                w: w_norm,
                out: f.hxn,
                eps,
                n,
                w_reps: hc,
            },
        )?;
        let w_down = model.w4("output_hc_down.weight")?;
        acc.frame_mm(f.hxn, &w_down, f.hlo, t)
            .map_err(Q4Error::Io)?;
        op(
            acc,
            FrameOp::SiluDiv {
                t: f.hlo,
                div: hc as f32,
                n: f.hlo_len * t,
            },
        )?;
        let w_up = model.w4("output_hc_up.weight")?;
        acc.frame_mm(f.hlo, &w_up, f.hgate, t)
            .map_err(Q4Error::Io)?;
        op(
            acc,
            FrameOp::HcGateMean {
                xn: f.hxn,
                gate: f.hgate,
                out: f.hin,
                hc,
                n,
            },
        )?;
        let wout = model
            .w("output.weight")
            .ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        acc.frame_mm(f.hin, &wout, f.logits_t, t)
            .map_err(Q4Error::Io)?;
        let (logits, toks) = if greedy {
            // GPU argmax — vocab×t 플로트 전사·CPU 스캔 회피 (plans/74 N1).
            (
                Vec::new(),
                acc.frame_argmax_rows(f.logits_t, t, hp.vocab)
                    .map_err(Q4Error::Io)?,
            )
        } else {
            let mut all = vec![0.0f32; hp.vocab * t];
            acc.frame_read(f.logits_t, &mut all).map_err(Q4Error::Io)?;
            if llm170_diag::dump::opts().key("np_dbg") {
                for r in 0..t {
                    let row = &all[r * hp.vocab..(r + 1) * hp.vocab];
                    let (i1, v1) = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    let (i2, v2) = row
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != i1)
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    eprintln!(
                        "# npdbg logits row{r}: top2 ({i1},{v1:.4}) ({i2},{v2:.4}) gap={:.4}",
                        v1 - v2
                    );
                }
            }
            (
                (0..t)
                    .map(|r| all[r * hp.vocab..(r + 1) * hp.vocab].to_vec())
                    .collect(),
                Vec::new(),
            )
        };
        ftime_report(t);
        if ftime_on() {
            eprintln!(
                "# np-frame-total t={t} greedy={greedy} {:.1}ms",
                t_call.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok((logits, toks))
    }
}

/// MoE 프레임(np) — **행별 t=1 경로**. MoE는 행마다 전문가가 달라 무게 공유가
/// 없고, t>1 gather/scatter 경로는 가중합 순서가 t=1과 달라(~1e-7, 문서화)
/// 근접 평탄점을 플립한다. 배치 불변식(== 순차 decode1)을 위해 t=1 산술을
/// 그대로 행마다 실행한다(전문가 읽기 총량은 동일).
pub(super) fn moe_frame_np(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    n: usize,
    seqs: &[usize],
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let k_sel = hp.n_expert_used;
    let n_ff = hp.n_ff_exp;
    let w_route = model.w4(&format!("blk.{il}.ffn_gate_inp.weight"))?;
    let w_route_sh = model.w4(&format!("blk.{il}.ffn_gate_inp_shexp.weight"))?;
    let w_gate = model.w4(&format!("blk.{il}.ffn_gate_exps.weight"))?;
    let w_up = model.w4(&format!("blk.{il}.ffn_up_exps.weight"))?;
    let w_down = model.w4(&format!("blk.{il}.ffn_down_exps.weight"))?;
    let fs: &dyn FrameState = acc;
    let vv = f.np_views.as_ref().unwrap();
    fs_begin(acc, 1); // per-seq 구간 — frame_moe_gemm이 t_cur×k_sel 행을 유추
    for (row, _sq) in seqs.iter().enumerate() {
        let mix_row = vv.mix[row];
        let mout_row = vv.mout[row];
        acc.frame_mm_group(mix_row, &[w_route, w_route_sh], &[f.mroute, f.msgate], 1)
            .map_err(Q4Error::Io)?;
        op(
            acc,
            FrameOp::MoeTop10 {
                route: f.mroute,
                ids: f.mids,
                wt: f.mwt,
                n_exp: hp.n_expert,
                k_sel,
            },
        )?;
        op(
            acc,
            FrameOp::BcastRows {
                src: mix_row,
                dst: f.mxsel,
                n,
                rows: k_sel,
            },
        )?;
        fs.frame_moe_gemm(f.mxsel, &w_gate, f.mids, f.mgu, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        fs.frame_moe_gemm(f.mxsel, &w_up, f.mids, f.mup, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(
            acc,
            FrameOp::SiluMul {
                g: f.mgu,
                u: f.mup,
                out: f.mglu,
                n: k_sel * n_ff,
            },
        )?;
        fs.frame_moe_gemm(f.mglu, &w_down, f.mids, f.my, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(
            acc,
            FrameOp::MoeWeightedSum {
                ys: f.my,
                wt: f.mwt,
                out: mout_row,
                k: k_sel,
                n,
            },
        )?;
        // shared 전문가 — t=1 융합 2런치(순차 경로와 동일)
        let shg_w = model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
        let shu_w = model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
        let shd_w = model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
        op(acc, FrameOp::Sigmoid { t: f.msgate, n: 1 })?;
        acc.shexp_gu(mix_row, &shg_w, &shu_w, f.shglu, n, n_ff)
            .map_err(Q4Error::Io)?;
        acc.shexp_da(f.shglu, &shd_w, f.msgate, mout_row, n, n_ff)
            .map_err(Q4Error::Io)?;
    }
    // per-seq 구간을 벗어났다는 표시 — 정확한 행 수는 호출자만 알므로
    // (프레임 진입 t) 이 함수는 1을 남기고 호출자가 t로 다시 세운다.
    // 종전 seqs.len()은 행 수가 아니라 시퀀스 수라 헤드를 오도할 수 있었다.
    fs_begin(acc, 1);
    Ok(())
}
