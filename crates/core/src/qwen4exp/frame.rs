//! 프레임 디코드 — 활성화 전 층 GPU 상주 (P2-4, plans/gpu-frame.md).
//!
//! 값 경로(matmul/ops Vec 기반)와 병행 — CPU golden 대조가 항상 가능하다.
//! v1 범위 (2026-09-01): 디코드 t=1. hc·GDN·MoE·head는 전부 프레임,
//! PLE는 CPU 브리지(호스트 해시), QSA는 값 경로 브리지(캐시 업로드 유지).
//! 게이트: LLM170_FRAME=1 (기본 off — 검증 통과 후 전환).
//!
//! 동기화 예산: QSA 브리지 12 + PLE 판독 1 + head 판독 1 ≈ 14회/스텝
//! (값 경로 ~600회). 나머지는 HIP_LAUNCH_BLOCKING 런치 (~350회).

use super::layers::SeqState4;
use super::stages::{self, Ctx};
use super::{Hparams4, Model4, Q4Error};
use crate::matmul::{Accelerator, FrameOp, FrameState};
use crate::quant::dequant_row;
use std::collections::HashMap;

/// 프레임 버퍼 집합 — 활성화(스텝 공용) + 상태(시퀀스별) + 상수 가중치.
pub struct Frame4 {
    // ── 스텝 활성 (t=1) ──
    pub res_hc: u64, // [hc·n_embd] — hc 스트림 잔차
    pub xn: u64,     // [hc·n_embd] — hc rms 출력
    pub lo: u64,     // [hc_down.n_out] — 저랭크
    pub gate: u64,   // [hc·n_embd] — hc up 출력(게이트)
    pub inj: u64,    // [hc] — inject
    pub mix: u64,    // [n_embd]
    pub ffn_out: u64, // [n_embd] — attn/moe 출력 (combine 입력)
    // GDN
    pub gqkv: u64, // [conv_ch]
    pub gz: u64,   // [d_inner]
    pub gb: u64,   // [dt_rank]
    pub ga: u64,   // [dt_rank]
    pub gbg: u64,  // [dt_rank·2] — β, e^g
    pub gconv: u64, // [conv_ch] — silu 적용 출력
    pub gq: u64,   // [k_len]
    pub gk: u64,   // [k_len]
    pub gv: u64,   // [v_len]
    pub go: u64,   // [v_len]
    pub ggated: u64, // [d_inner]
    // MoE
    pub mroute: u64, // [n_expert]
    pub msgate: u64, // [1]
    pub mids: u64,   // [k_sel] u32
    pub mwt: u64,    // [k_sel]
    pub mxsel: u64,  // [k_sel·n_embd] — x 브로드캐스트
    pub mgu: u64,    // [k_sel·n_ff]
    pub mup: u64,    // [k_sel·n_ff]
    pub mglu: u64,   // [k_sel·n_ff]
    pub my: u64,     // [k_sel·n_embd]
    pub mout: u64,   // [n_embd]
    pub shg: u64,    // [n_ff] shared gate
    pub shu: u64,    // [n_ff] shared up
    pub shglu: u64,  // [n_ff]
    pub shout: u64,  // [n_embd]
    // head
    pub hxn: u64,  // [hc·n_embd]
    pub hlo: u64,
    pub hgate: u64,
    pub hin: u64, // [n_embd]
    pub logits: u64, // [vocab]
    // 저랭크 버퍼 길이 (hc_down/output_hc_down n_out — 생성 시 고정)
    pub lo_len: usize,
    pub hlo_len: usize,
    // ── 시퀀스별 상주 상태 (순환 idx / full idx 순) ──
    pub st_gdn: Vec<u64>,  // [n_recr][dt_rank·d·d]
    pub st_conv: Vec<u64>, // [n_recr][(k-1)·conv_ch]
    // ── 상수 (이름 → 핸들) — f32 norm류 등 스텝 프레임에 미리 상주 ──
    pub consts: HashMap<String, u64>,
    /// 값 경로 prefill 이후 상태 재동기 필요 플래그.
    pub dirty: bool,
}

fn alloc(acc: &dyn Accelerator, len: usize) -> Result<u64, Q4Error> {
    acc.frame_alloc(len).map_err(Q4Error::Io)
}

fn op(acc: &dyn Accelerator, o: FrameOp) -> Result<(), Q4Error> {
    acc.frame_op(&o).map_err(Q4Error::Io)
}

impl Frame4 {
    /// 최초 프레임 디코드 시 생성 — CPU SeqState4에서 상태 업로드.
    pub fn new(
        acc: &dyn Accelerator,
        model: &Model4,
        seqs: &[SeqState4],
    ) -> Result<Self, Q4Error> {
        let hp = &model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let k_len = hp.n_group * hp.d_state;
        let v_len = hp.dt_rank * hp.d_state;
        let conv_ch = 2 * k_len + v_len;
        let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recr(il)).count();
        let a = |len: usize| alloc(acc, len);
        let lo_n = model.w4("blk.0.hc_attn_down.weight")?.n_out as usize;
        let hlo_n = model.w4("output_hc_down.weight")?.n_out as usize;

        let mut f = Frame4 {
            res_hc: a(hc * n)?,
            xn: a(hc * n)?,
            lo: a(lo_n)?,
            gate: a(hc * n)?,
            inj: a(hc)?,
            mix: a(n)?,
            ffn_out: a(n)?,
            gqkv: a(conv_ch)?,
            gz: a(hp.d_inner)?,
            gb: a(hp.dt_rank)?,
            ga: a(hp.dt_rank)?,
            gbg: a(hp.dt_rank * 2)?,
            gconv: a(conv_ch)?,
            gq: a(k_len)?,
            gk: a(k_len)?,
            gv: a(v_len)?,
            go: a(v_len)?,
            ggated: a(hp.d_inner)?,
            mroute: a(hp.n_expert)?,
            msgate: a(1)?,
            mids: a(hp.n_expert_used)?,
            mwt: a(hp.n_expert_used)?,
            mxsel: a(hp.n_expert_used * n)?,
            mgu: a(hp.n_expert_used * hp.n_ff_exp)?,
            mup: a(hp.n_expert_used * hp.n_ff_exp)?,
            mglu: a(hp.n_expert_used * hp.n_ff_exp)?,
            my: a(hp.n_expert_used * n)?,
            mout: a(n)?,
            shg: a(hp.n_ff_exp)?,
            shu: a(hp.n_ff_exp)?,
            shglu: a(hp.n_ff_exp)?,
            shout: a(n)?,
            hxn: a(hc * n)?,
            hlo: a(hlo_n)?,
            hgate: a(hc * n)?,
            hin: a(n)?,
            logits: a(hp.vocab)?,
            lo_len: lo_n,
            hlo_len: hlo_n,
            st_gdn: Vec::with_capacity(n_recr),
            st_conv: Vec::with_capacity(n_recr),
            consts: HashMap::new(),
            dirty: true,
        };
        // 시퀀스별 GDN 상태 — 시퀀스 0 기준 할당, 이후 시퀀스는 별도 풀.
        // v1: 디코드 단일 시퀀스 가정 (np 디코드는 seq별 순차 처리 —
        // 스테이트 스왑은 v2에서 st_gdn을 [n_seqs][n_recr]로 확장).
        for ri in 0..n_recr {
            f.st_gdn.push(a(hp.dt_rank * hp.d_state * hp.d_state)?);
            f.st_conv.push(a((hp.conv_k - 1) * conv_ch)?);
            let _ = ri;
        }
        // 상수 가중치 업로드 — 층별 norm류 + GDN 스칼라 계수.
        let mut put = |name: &str, v: &[f32]| -> Result<(), Q4Error> {
            let h = a(v.len())?;
            acc.frame_write(h, v).map_err(Q4Error::Io)?;
            f.consts.insert(name.into(), h);
            Ok(())
        };
        for il in 0..hp.n_layer {
            put(&format!("blk.{il}.hc_attn_norm"), &model.f32_vec4(&format!("blk.{il}.hc_attn_norm.weight"))?)?;
            put(&format!("blk.{il}.hc_ffn_norm"), &model.f32_vec4(&format!("blk.{il}.hc_ffn_norm.weight"))?)?;
            if hp.is_recr(il) {
                // ssm_norm.weight는 [d_state] 전헤드 공유 — norm_gated_rows 커널이
                // 헤드별 슬라이스 인덱싱(w[(row%n_h)·d+i])하므로 dt_rank 타일로
                // 업로드. 미타일 업로드는 OOB 읽기로 v1 발산의 근원 (2026-09-01).
                let sn = model.f32_vec4(&format!("blk.{il}.ssm_norm.weight"))?;
                let sn_tiled: Vec<f32> = sn.iter().copied().cycle().take(sn.len() * hp.dt_rank).collect();
                put(&format!("blk.{il}.ssm_norm"), &sn_tiled)?;
                put(&format!("blk.{il}.dt_bias"), &model.f32_vec4(&format!("blk.{il}.ssm_dt.bias"))?)?;
                put(&format!("blk.{il}.ssm_a"), &model.f32_vec4(&format!("blk.{il}.ssm_a"))?)?;
                put(&format!("blk.{il}.conv_w"), &model.f32_vec4(&format!("blk.{il}.ssm_conv1d.weight"))?)?;
            }
        }
        put("output_hc_norm", &model.f32_vec4("output_hc_norm.weight")?)?;
        // 시퀀스 0의 현재 CPU 상태를 초기값으로 (dirty 해소)
        f.sync_states(acc, &seqs[0])?;
        Ok(f)
    }

    /// CPU SeqState4의 GDN 상태를 GPU로 재동기 (prefill 직후).
    pub fn sync_states(&mut self, acc: &dyn Accelerator, st: &SeqState4) -> Result<(), Q4Error> {
        for (ri, h) in self.st_gdn.iter().enumerate() {
            acc.frame_write(*h, &st.gdn_s[ri]).map_err(Q4Error::Io)?;
        }
        for (ri, h) in self.st_conv.iter().enumerate() {
            acc.frame_write(*h, &st.conv[ri]).map_err(Q4Error::Io)?;
        }
        self.dirty = false;
        Ok(())
    }
}

/// 프레임 디코드 1스텝 — Engine4::decode1에서 호출 (t=1 전용).
pub fn decode_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    token: u32,
) -> Result<Vec<f32>, Q4Error> {
    let hp: &Hparams4 = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let k_len = hp.n_group * hp.d_state;
    let v_len = hp.dt_rank * hp.d_state;
    let conv_ch = 2 * k_len + v_len;
    let eps = hp.eps;

    // 0) 임베딩 — CPU dequant → 4스트림 방송 기록
    {
        let embd = model
            .w("token_embd.weight")
            .ok_or(Q4Error::MissingTensor("token_embd".into()))?;
        let mut row = vec![0.0f32; n];
        dequant_row(embd.ty, embd.data, token as u64, n as u64, &mut row);
        let mut r = vec![0.0f32; hc * n];
        for s in 0..hc {
            r[s * n..(s + 1) * n].copy_from_slice(&row);
        }
        acc.frame_write(f.res_hc, &r).map_err(Q4Error::Io)?;
    }

    // PLE n-gram 행 (호스트 해시) — PLE층 직전에 CPU에서 필요
    let ple_rows = if hp.is_ple(1) {
        stages::ple_hash(ctx, seq_st, &[token])
    } else {
        Vec::new()
    };

    let mut recr_idx = 0usize;
    let mut full_idx = 0usize;
    for il in 0..hp.n_layer {
        // 1) PLE (blk.1) — CPU 브리지: res_hc 판독 → CPU → 기록
        if hp.is_ple(il) {
            let mut r = vec![0.0f32; hc * n];
            acc.frame_read(f.res_hc, &mut r).map_err(Q4Error::Io)?;
            let mut rows = vec![r];
            stages::ple_block(ctx, seq_st, il, &mut rows, &ple_rows)?;
            acc.frame_write(f.res_hc, &rows[0]).map_err(Q4Error::Io)?;
        }

        // 2) hc attn mix — rms → down/inject 그룹 → silu(hc 나눗셈) → up → 게이트 평균
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc)?;

        // 3) attention — GDN 프레임 / QSA 값 브리지
        if hp.is_recr(il) {
            gdn_frame(acc, model, f, il, recr_idx, conv_ch, k_len, v_len, eps)?;
            recr_idx += 1;
            let o = f.ffn_out;
            let inj = f.inj;
            hc_combine_frame(acc, f, o, inj, n, hc)?;
        } else {
            // QSA 값 경로 브리지 — mix 판독 → qsa_layer → 출력 기록 (1 sync)
            let mut mix_v = vec![0.0f32; n];
            acc.frame_read(f.mix, &mut mix_v).map_err(Q4Error::Io)?;
            let xs = vec![mix_v];
            let out = stages::qsa_layer(ctx, seq_st, il, &xs, 1, full_idx)?;
            acc.frame_write(f.ffn_out, &out[0]).map_err(Q4Error::Io)?;
            full_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc)?;
        }

        // 4) hc ffn mix + MoE
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc)?;
        moe_frame(acc, model, f, il, n)?;
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc)?;
    }

    // 5) head — output hc mix + output GEMM + 판독 (최종 1 sync)
    {
        let w_norm = f.consts["output_hc_norm"];
        op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.hxn, eps, n, w_reps: hc })?;
        let w_down = model.w4("output_hc_down.weight")?;
        acc.frame_mm(f.hxn, &w_down, f.hlo, 1).map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluDiv { t: f.hlo, div: hc as f32, n: f.hlo_len })?;
        let w_up = model.w4("output_hc_up.weight")?;
        acc.frame_mm(f.hlo, &w_up, f.hgate, 1).map_err(Q4Error::Io)?;
        op(acc, FrameOp::HcGateMean { xn: f.hxn, gate: f.hgate, out: f.hin, hc, n })?;
        let wout = model.w("output.weight").ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        acc.frame_mm(f.hin, &wout, f.logits, 1).map_err(Q4Error::Io)?;
        let mut logits = vec![0.0f32; hp.vocab];
        acc.frame_read(f.logits, &mut logits).map_err(Q4Error::Io)?;
        Ok(logits)
    }
}

/// hc_mix 프레임 — CPU stages/hc.rs hc_mix와 동일 순서 (inject 반환 포함).
fn hc_mix_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    kind: &str,
    eps: f32,
    n: usize,
    hc: usize,
) -> Result<(), Q4Error> {
    let w_norm = f.consts[&format!("blk.{il}.hc_{kind}_norm")];
    op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.xn, eps, n, w_reps: hc })?;
    let w_down = model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
    let w_inject = model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;
    acc.frame_mm_group(f.xn, &[w_down, w_inject], &[f.lo, f.inj], 1)
        .map_err(Q4Error::Io)?;
    op(acc, FrameOp::SiluDiv { t: f.lo, div: hc as f32, n: f.lo_len })?;
    let w_up = model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?;
    acc.frame_mm(f.lo, &w_up, f.gate, 1).map_err(Q4Error::Io)?;
    op(acc, FrameOp::HcGateMean { xn: f.xn, gate: f.gate, out: f.mix, hc, n })?;
    Ok(())
}

/// hc_combine 프레임 — layers.rs hc_combine과 동일 수식.
fn hc_combine_frame(
    acc: &dyn Accelerator,
    f: &Frame4,
    out: u64,
    inj: u64,
    n: usize,
    hc: usize,
) -> Result<(), Q4Error> {
    op(acc, FrameOp::HcCombine { res: f.res_hc, out, inj, hc, n, total: hc * n })
}

/// GDN 프레임 — stages/gdn.rs와 동일 순서 (t=1).
#[allow(clippy::too_many_arguments)]
fn gdn_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    ri: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    eps: f32,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    // qkv/z/b/a 그룹 — 동일 입력 mix
    let wqkv = model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
    let wz = model.w4(&format!("blk.{il}.attn_gate.weight"))?;
    let wb = model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
    let wa = model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
    acc.frame_mm_group(f.mix, &[wqkv, wz, wb, wa], &[f.gqkv, f.gz, f.gb, f.ga], 1)
        .map_err(Q4Error::Io)?;
    // β/e^g
    let dtb = f.consts[&format!("blk.{il}.dt_bias")];
    let ssa = f.consts[&format!("blk.{il}.ssm_a")];
    op(acc, FrameOp::GdnBetaG { b: f.gb, a: f.ga, dtb, sa: ssa, bg: f.gbg, n_h: hp.dt_rank })?;
    // conv + ring
    let cw = f.consts[&format!("blk.{il}.conv_w")];
    op(acc, FrameOp::GdnConv { qkv: f.gqkv, cw, state: f.st_conv[ri], out: f.gconv, ch: conv_ch, k: hp.conv_k, t_len: 1 })?;
    // q/k/v 분할 + l2 + q·scale
    op(acc, FrameOp::CopyRows { src: f.gconv, dst: f.gq, src_off: 0, dst_off: 0, n: k_len })?;
    op(acc, FrameOp::CopyRows { src: f.gconv, dst: f.gk, src_off: k_len, dst_off: 0, n: k_len })?;
    op(acc, FrameOp::CopyRows { src: f.gconv, dst: f.gv, src_off: 2 * k_len, dst_off: 0, n: v_len })?;
    op(acc, FrameOp::L2Rows { x: f.gq, eps, d: hp.d_state })?;
    op(acc, FrameOp::L2Rows { x: f.gk, eps, d: hp.d_state })?;
    let scale = 1.0f32 / (hp.d_state as f32).sqrt();
    op(acc, FrameOp::Scale { t: f.gq, s: scale, n: k_len })?;
    // AR 상태 갱신 — 상태 GPU 상주, 판독 없음
    let fs: &dyn FrameState = acc;
    fs.frame_gdn_ar(f.gq, f.gk, f.gv, f.gbg, f.st_gdn[ri], f.go, 1, hp.n_group, hp.dt_rank, hp.d_state)
        .map_err(Q4Error::Io)?;
    // norm_gated + out proj
    let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
    op(acc, FrameOp::NormGated { o: f.go, z: f.gz, w: snorm, out: f.ggated, eps, d: hp.d_state, n_h: hp.dt_rank })?;
    let wout = model.w4(&format!("blk.{il}.ssm_out.weight"))?;
    acc.frame_mm(f.ggated, &wout, f.ffn_out, 1).map_err(Q4Error::Io)?;
    Ok(())
}

/// MoE 프레임 — stages/moe.rs t=1 경로와 동일 수식.
/// 합산 순서 차이: 전문가 가중합을 ids 순(확률 내림차순)으로 누산 —
/// CPU는 전문가 id 오름차순. f32 10항 합의 순서 차이 (~1e-7) — 기존
/// GPU GEMM 재정렬 편차(5e-3)보다 4자리 작아 매트릭스로 검증.
fn moe_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    il: usize,
    n: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let k_sel = hp.n_expert_used;
    let n_ff = hp.n_ff_exp;
    // route + shared 게이트
    let w_route = model.w4(&format!("blk.{il}.ffn_gate_inp.weight"))?;
    let w_route_sh = model.w4(&format!("blk.{il}.ffn_gate_inp_shexp.weight"))?;
    acc.frame_mm_group(f.mix, &[w_route, w_route_sh], &[f.mroute, f.msgate], 1)
        .map_err(Q4Error::Io)?;
    op(acc, FrameOp::MoeTop10 { route: f.mroute, ids: f.mids, wt: f.mwt, n_exp: hp.n_expert, k_sel })?;
    // x 브로드캐스트 (전문가당 동일 입력)
    for e in 0..k_sel {
        op(acc, FrameOp::CopyRows { src: f.mix, dst: f.mxsel, src_off: 0, dst_off: e * n, n })?;
    }
    let fs: &dyn FrameState = acc;
    let w_gate = model.w4(&format!("blk.{il}.ffn_gate_exps.weight"))?;
    let w_up = model.w4(&format!("blk.{il}.ffn_up_exps.weight"))?;
    fs.frame_moe_gemm(f.mxsel, &w_gate, f.mids, f.mgu, hp.n_expert)
        .map_err(Q4Error::Io)?;
    fs.frame_moe_gemm(f.mxsel, &w_up, f.mids, f.mup, hp.n_expert)
        .map_err(Q4Error::Io)?;
    op(acc, FrameOp::SiluMul { g: f.mgu, u: f.mup, out: f.mglu, n: k_sel * n_ff })?;
    let w_down = model.w4(&format!("blk.{il}.ffn_down_exps.weight"))?;
    fs.frame_moe_gemm(f.mglu, &w_down, f.mids, f.my, hp.n_expert)
        .map_err(Q4Error::Io)?;
    op(acc, FrameOp::MoeWeightedSum { ys: f.my, wt: f.mwt, out: f.mout, k: k_sel, n })?;
    // shared 전문가 — σ(sgate)·shout 가산
    op(acc, FrameOp::Sigmoid { t: f.msgate, n: 1 })?;
    let shg_w = model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
    let shu_w = model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
    acc.frame_mm_group(f.mix, &[shg_w, shu_w], &[f.shg, f.shu], 1)
        .map_err(Q4Error::Io)?;
    op(acc, FrameOp::SiluMul { g: f.shg, u: f.shu, out: f.shglu, n: n_ff })?;
    let shd_w = model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
    acc.frame_mm(f.shglu, &shd_w, f.shout, 1).map_err(Q4Error::Io)?;
    op(acc, FrameOp::AxpyScaled { y: f.mout, x: f.shout, s: f.msgate, n })?;
    Ok(())
}

