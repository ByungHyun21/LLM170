//! frame/mtp — MTP 드래프트 프레임 경로 (plans/110 W1).
//!
//! 외장 nextn 블록(il=n_layer)의 GEMV를 Frame4 상주 버퍼 + FrameOp 체인으로
//! 실행한다. 종전 값경로(`Ctx::mm`→run_prepared)는 GEMV 호출마다 h2d→런치→
//! **d2h 동기**를 했다(~5.2ms × ~25호출 ≈ 125ms/드래프트 스텝, 원장 122).
//! 프레임판은 활성을 디바이스에 둔 채 체인을 이어 d2h를 스텝당 5회
//! (q/k/v 판독·argmax·chain_h 반출)로 축소한다.
//!
//! 산술 클래스 불변: GEMV는 값경로와 동일 kernel 패밀리(launch_gemm +
//! quant_q8_b — q4acc value.rs run_prepared와 frame_gemm이 같은 기계를 쓴다).
//! elementwise는 trunk 프레임과 동일 커널(rms_part/finish·q4_silu_div·
//! q4_hc_gate_mean·q4_hc_combine·q4_moe_top10_m 등). dense 어텐션의
//! norm/rope/KV/softmax는 CPU 유지(경량 — layers.rs mtp_attn_cpu_row 공유,
//! GPU화는 W1b).
//!
//! 버퍼는 트렁크 Frame4 활성과 완전 분리(자체 MtpFrame) — 트렁크 스텝 사이
//! 임시 실행이므로 재사용도 가능하지만, lo/hlo 길이가 nextn 가중치
//! n_out에 의존해 별도 할당이 사이즈 계약도 함께 해소한다.

use super::diag::sync_mark;
use super::{Frame4, op};
use crate::matmul::{Accelerator, FrameOp, FrameState};
use crate::qwen4exp::layers::{SeqState4, mtp_attn_cpu_row};
use crate::qwen4exp::{Model4, Q4Error};

/// 백엔드 GEMV 미지원 타입(Q5_0 — MTP 모듈 hc_up 3종)의 f32 디양자화
/// 상주판. 디양자화는 비트 보존이라 산술 클래스 불변 — GEMV는 트렁크
/// MoE 라우터(ffn_gate_inp F32)가 매 스텝 쓰는 q4_gemm_f32 경로.
pub struct F32Weight {
    pub data: Vec<u8>, // f32 비트열(네이티브)
    pub n_in: u64,
    pub n_out: u64,
}

impl F32Weight {
    fn empty() -> Self {
        F32Weight {
            data: Vec::new(),
            n_in: 0,
            n_out: 0,
        }
    }
    fn w(&self) -> crate::matmul::Weight<'_> {
        crate::matmul::Weight {
            data: &self.data,
            ty: llm170_gguf::GgmlType::F32,
            n_in: self.n_in,
            n_out: self.n_out,
        }
    }
    /// Q5_0이면 f32 디양자화 상주판 생성, 아니면 empty(모델 원본 사용).
    fn deq_q5(model: &Model4, name: &str) -> Result<Self, Q4Error> {
        let w = model.w4(name)?;
        if w.ty != llm170_gguf::GgmlType::Q5_0 {
            return Ok(Self::empty());
        }
        let v = w.dequant_f32_vec();
        Ok(F32Weight {
            data: v
                .iter()
                .flat_map(|f: &f32| f.to_bits().to_ne_bytes())
                .collect(),
            n_in: w.n_in,
            n_out: w.n_out,
        })
    }
}

/// 드래프트 전용 상주 버퍼(t=1 단일 행) — Frame4::new에서 has_mtp 시 할당.
pub struct MtpFrame {
    /// eh_proj 입력 [hc][2n] — 호스트 조립(en 방송 ‖ hn_s) 후 1회 기입.
    pub cat: u64,
    /// 드래프트 프리-믹서 잔차 [hc·n] — eh_proj 출력이자 체인 반출원.
    pub res: u64,
    pub xn: u64,   // [hc·n] hc rms 출력
    pub lo: u64,   // [lo_len] 저랭크(attn/ffn/head 공유, 순차 사용)
    pub inj: u64,  // [hc]
    pub gate: u64, // [hc·n]
    pub mix: u64,  // [n]
    // dense 어텐션 투영·출력
    pub q: u64,    // [n_head·2hd] q‖게이트 인터리브
    pub k: u64,    // [n_kv·hd]
    pub v: u64,    // [n_kv·hd]
    pub attn: u64, // [n_head·hd] CPU softmax 결과 기입(wo 입력)
    pub ao: u64,   // [n] wo 출력
    // MoE
    pub mroute: u64, // [n_expert]
    pub msgate: u64, // [1]
    pub mids: u64,   // [k_sel] u32
    pub mwt: u64,    // [k_sel]
    pub mxsel: u64,  // [k_sel·n]
    pub mgu: u64,    // [k_sel·n_ff]
    pub mup: u64,
    pub mglu: u64,
    pub my: u64,   // [k_sel·n]
    pub mout: u64, // [n]
    // shared 전문가 — 융합 커널(q4_shexp_*)은 Q8_0 전용 레이아웃이라 드래프트
    // shexp(q4_K/q8_0 혼합)엔 일반 경로(trunk 프리필 판)를 쓴다.
    pub shg: u64,   // [n_ff] shared gate
    pub shu: u64,   // [n_ff] shared up
    pub shglu: u64, // [n_ff]
    pub shout: u64, // [n]
    // 헤드(nextn.hc_head_*)
    pub hxn: u64,    // [hc·n]
    pub hgate: u64,  // [hc·n]
    pub hin: u64,    // [n]
    pub logits: u64, // [vocab]
    /// lo 버퍼 실제 길이(attn/ffn/head down n_out 최대).
    pub lo_len: usize,
    /// Q5_0 hc_up 디양자화 상주판(미지원 타입 폴백 — 비어 있으면 원본 사용).
    pub up_attn: F32Weight,
    pub up_ffn: F32Weight,
    pub up_head: F32Weight,
}

impl MtpFrame {
    pub(super) fn new(acc: &dyn Accelerator, model: &Model4) -> Result<Self, Q4Error> {
        let hp = &model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let il = hp.n_layer;
        let a = |len: usize| super::alloc(acc, len);
        // attn/ffn/head 저랭크 down n_out은 nextn 가중치마다 다를 수 있다 —
        // 최대값으로 단일 버퍼(SiluDiv는 매 시점 정확 길이를 받는다).
        let lo_len = [
            model.w4(&format!("blk.{il}.hc_attn_down.weight"))?.n_out,
            model.w4(&format!("blk.{il}.hc_ffn_down.weight"))?.n_out,
            model
                .w4(&format!("blk.{il}.nextn.hc_head_down.weight"))?
                .n_out,
        ]
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as usize;
        let (k_sel, n_ff) = (hp.n_expert_used, hp.n_ff_exp);
        Ok(MtpFrame {
            cat: a(hc * 2 * n)?,
            res: a(hc * n)?,
            xn: a(hc * n)?,
            lo: a(lo_len)?,
            inj: a(hc)?,
            gate: a(hc * n)?,
            mix: a(n)?,
            q: a(hp.n_head * 2 * hp.head_dim)?,
            k: a(hp.n_kv * hp.head_dim)?,
            v: a(hp.n_kv * hp.head_dim)?,
            attn: a(hp.n_head * hp.head_dim)?,
            ao: a(n)?,
            mroute: a(hp.n_expert)?,
            msgate: a(1)?,
            mids: a(k_sel)?,
            mwt: a(k_sel)?,
            mxsel: a(k_sel * n)?,
            mgu: a(k_sel * n_ff)?,
            mup: a(k_sel * n_ff)?,
            mglu: a(k_sel * n_ff)?,
            my: a(k_sel * n)?,
            mout: a(n)?,
            shg: a(n_ff)?,
            shu: a(n_ff)?,
            shglu: a(n_ff)?,
            shout: a(n)?,
            hxn: a(hc * n)?,
            hgate: a(hc * n)?,
            hin: a(n)?,
            logits: a(hp.vocab)?,
            lo_len,
            up_attn: F32Weight::deq_q5(model, &format!("blk.{il}.hc_attn_up.weight"))?,
            up_ffn: F32Weight::deq_q5(model, &format!("blk.{il}.hc_ffn_up.weight"))?,
            up_head: F32Weight::deq_q5(model, &format!("blk.{il}.nextn.hc_head_up.weight"))?,
        })
    }
}

/// hc_mix 드래프트판 — trunk hc_mix_frame(forward.rs)과 동일 op열.
/// 가중치는 blk.{il}.hc_{kind}_{norm,down,inject,up}(il=n_layer, nextn 블록).
#[allow(clippy::too_many_arguments)]
fn hc_mix_draft(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &Frame4,
    mf: &MtpFrame,
    il: usize,
    kind: &str,
    eps: f32,
    n: usize,
    hc: usize,
) -> Result<(), Q4Error> {
    let w_norm = f.consts[&format!("blk.{il}.hc_{kind}_norm")];
    op(
        acc,
        FrameOp::RmsRows {
            x: mf.res,
            w: w_norm,
            out: mf.xn,
            eps,
            n,
            w_reps: hc,
        },
    )?;
    let w_down = model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
    let w_inject = model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;
    acc.frame_mm_group(mf.xn, &[w_down, w_inject], &[mf.lo, mf.inj], 1)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dft_hc_down", mf.lo)?;
    op(
        acc,
        FrameOp::SiluDiv {
            t: mf.lo,
            div: hc as f32,
            n: mf.lo_len,
        },
    )?;
    // Q5_0 up(MTP 모듈)은 f32 디양자화 상주판 — 비어 있으면 모델 원본.
    let up_fw = match kind {
        "attn" => &mf.up_attn,
        _ => &mf.up_ffn,
    };
    let w_up = if up_fw.data.is_empty() {
        model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?
    } else {
        up_fw.w()
    };
    acc.frame_mm(mf.lo, &w_up, mf.gate, 1)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::HcGateMean {
            xn: mf.xn,
            gate: mf.gate,
            out: mf.mix,
            hc,
            n,
        },
    )?;
    sync_mark(acc, "dft_hc_mix", mf.mix)?;
    Ok(())
}

/// hc_combine 드래프트판 — hc_combine_frame과 동일 수식(res=드래프트 잔차).
fn hc_combine_draft(
    acc: &dyn Accelerator,
    mf: &MtpFrame,
    out: u64,
    n: usize,
    hc: usize,
) -> Result<(), Q4Error> {
    op(
        acc,
        FrameOp::HcCombine {
            res: mf.res,
            out,
            inj: mf.inj,
            hc,
            n,
            total: hc * n,
        },
    )
}

/// MoE 드래프트판 — moe_frame t=1 체인과 동일(top10·direct-ids·shared 융합).
fn moe_draft(
    acc: &dyn Accelerator,
    model: &Model4,
    mf: &MtpFrame,
    il: usize,
    n: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let (k_sel, n_ff) = (hp.n_expert_used, hp.n_ff_exp);
    let w_route = model.w4(&format!("blk.{il}.ffn_gate_inp.weight"))?;
    let w_route_sh = model.w4(&format!("blk.{il}.ffn_gate_inp_shexp.weight"))?;
    acc.frame_mm_group(mf.mix, &[w_route, w_route_sh], &[mf.mroute, mf.msgate], 1)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::MoeTop10 {
            route: mf.mroute,
            ids: mf.mids,
            wt: mf.mwt,
            n_exp: hp.n_expert,
            k_sel,
        },
    )?;
    op(
        acc,
        FrameOp::BcastRows {
            src: mf.mix,
            dst: mf.mxsel,
            n,
            rows: k_sel,
        },
    )?;
    let fs: &dyn FrameState = acc;
    sync_mark(acc, "dftpreroute", mf.mroute)?;
    let w_gate = model.w4(&format!("blk.{il}.ffn_gate_exps.weight"))?;
    let w_up = model.w4(&format!("blk.{il}.ffn_up_exps.weight"))?;
    let w_down = model.w4(&format!("blk.{il}.ffn_down_exps.weight"))?;
    fs.frame_moe_gemm(mf.mxsel, &w_gate, mf.mids, mf.mgu, hp.n_expert, k_sel)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dftxgate", mf.mgu)?;
    fs.frame_moe_gemm(mf.mxsel, &w_up, mf.mids, mf.mup, hp.n_expert, k_sel)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::SiluMul {
            g: mf.mgu,
            u: mf.mup,
            out: mf.mglu,
            n: k_sel * n_ff,
        },
    )?;
    fs.frame_moe_gemm(mf.mglu, &w_down, mf.mids, mf.my, hp.n_expert, k_sel)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dftxdown", mf.my)?;
    op(
        acc,
        FrameOp::MoeWeightedSum {
            ys: mf.my,
            wt: mf.mwt,
            out: mf.mout,
            k: k_sel,
            n,
        },
    )?;
    sync_mark(acc, "dftmoewsum", mf.mout)?;
    // shared 전문가 — 일반 경로(Sigmoid→gate/up 그룹→SiluMul→down→Axpy).
    // 융합 커널(q4_shexp_gu/da)은 Q8_0 전용 레이아웃인데 드래프트 shexp
    // gate/up은 q4_K이라 조용한 오염이 된다 — trunk 프리필 판 산술을 쓴다.
    let shg_w = model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
    let shu_w = model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
    let shd_w = model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
    op(acc, FrameOp::Sigmoid { t: mf.msgate, n: 1 })?;
    acc.frame_mm_group(mf.mix, &[shg_w, shu_w], &[mf.shg, mf.shu], 1)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::SiluMul {
            g: mf.shg,
            u: mf.shu,
            out: mf.shglu,
            n: n_ff,
        },
    )?;
    acc.frame_mm(mf.shglu, &shd_w, mf.shout, 1)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::AxpyScaled {
            y: mf.mout,
            x: mf.shout,
            s: mf.msgate,
            n,
        },
    )?;
    sync_mark(acc, "dftmoeshared", mf.mout)?;
    Ok(())
}

/// MTP 드래프트 스텝(프레임 경로) — layers.rs mtp_draft_step_h 값경로와
/// 동일 산술 순서. 입력 en[n]·hn[hc·n]은 호출부가 CPU에서 정규화해 전달
/// (값경로와 공유). 반환: (greedy 토큰, 체인 h=프리-믹서 잔차[hc·n]).
/// mtp_st.pos 진행은 호출부가 담당(KV 기입은 내부 pos 기준).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mtp_draft_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    f: &mut Frame4,
    mtp_st: &mut SeqState4,
    en: &[f32],
    hn: &[f32],
) -> Result<(u32, Vec<f32>), Q4Error> {
    let hp = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let il = hp.n_layer;
    let eps = hp.eps;
    let mf = f
        .mtp
        .as_ref()
        .ok_or(Q4Error::Io("mtp 프레임 버퍼 미할당".into()))?;
    // t=1 op 계약 — RmsRows/HcGateMean/MoeTop10 등이 t_cur에서 행 수를 유도.
    super::fs_begin(acc, 1);

    // 1) eh_proj — cat[s]=[en ‖ hn_s] 조립 후 t=hc 배치 GEMM 1호출.
    //    값경로는 스트림별 ctx.mm 호출(hc회)이었다.
    let mut cat = vec![0.0f32; hc * 2 * n];
    for s in 0..hc {
        let b = s * 2 * n;
        cat[b..b + n].copy_from_slice(en);
        cat[b + n..b + 2 * n].copy_from_slice(&hn[s * n..(s + 1) * n]);
    }
    acc.frame_write(mf.cat, &cat).map_err(Q4Error::Io)?;
    let weh = model.w4(&format!("blk.{il}.nextn.eh_proj.weight"))?;
    acc.frame_mm(mf.cat, &weh, mf.res, hc)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dfteh", mf.res)?;

    // 2) attn 반쪽 hc_mix
    hc_mix_draft(acc, model, f, mf, il, "attn", eps, n, hc)?;
    sync_mark(acc, "dftattnmix", mf.mix)?;

    // 3) dense 어텐션 — 투영/wo GPU, norm·rope·KV·softmax·게이트 CPU
    //    (mtp_dense_attn과 동일 산술 — layers.rs mtp_attn_cpu_row 공유).
    sync_mark(acc, "dftpreattn", mf.mix)?;
    let wq = model.w4(&format!("blk.{il}.attn_q.weight"))?;
    let wk = model.w4(&format!("blk.{il}.attn_k.weight"))?;
    let wv = model.w4(&format!("blk.{il}.attn_v.weight"))?;
    let wo = model.w4(&format!("blk.{il}.attn_output.weight"))?;
    acc.frame_mm_group(mf.mix, &[wq, wk, wv], &[mf.q, mf.k, mf.v], 1)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dftqkv", mf.q)?;
    let qn = model.f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
    let kn = model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
    let mut q = vec![0.0f32; hp.n_head * 2 * hp.head_dim];
    let mut k = vec![0.0f32; hp.n_kv * hp.head_dim];
    let mut v = vec![0.0f32; hp.n_kv * hp.head_dim];
    acc.frame_read(mf.q, &mut q).map_err(Q4Error::Io)?;
    acc.frame_read(mf.k, &mut k).map_err(Q4Error::Io)?;
    acc.frame_read(mf.v, &mut v).map_err(Q4Error::Io)?;
    let attn = mtp_attn_cpu_row(hp, &mut q, &mut k, &v, mtp_st, &qn, &kn);
    acc.frame_write(mf.attn, &attn).map_err(Q4Error::Io)?;
    acc.frame_mm_group(mf.attn, &[wo], &[mf.ao], 1)
        .map_err(Q4Error::Io)?;
    hc_combine_draft(acc, mf, mf.ao, n, hc)?;
    sync_mark(acc, "dftattn", mf.ao)?;

    // 4) ffn 반쪽 — hc_mix + MoE + combine
    hc_mix_draft(acc, model, f, mf, il, "ffn", eps, n, hc)?;
    sync_mark(acc, "dftffnmix", mf.mix)?;
    moe_draft(acc, model, mf, il, n)?;
    sync_mark(acc, "dftmoe", mf.mout)?;
    hc_combine_draft(acc, mf, mf.mout, n, hc)?;
    sync_mark(acc, "dftffncomb", mf.mout)?;

    // 5) 헤드 — nextn.hc_head_{norm,down,up}(trunk output_hc_*와 동일 구조)
    let w_norm = f.consts[&format!("blk.{il}.nextn.hc_head_norm")];
    sync_mark(acc, "dftprehead", mf.hxn)?;
    op(
        acc,
        FrameOp::RmsRows {
            x: mf.res,
            w: w_norm,
            out: mf.hxn,
            eps,
            n,
            w_reps: hc,
        },
    )?;
    let w_down = model.w4(&format!("blk.{il}.nextn.hc_head_down.weight"))?;
    acc.frame_mm(mf.hxn, &w_down, mf.lo, 1)
        .map_err(Q4Error::Io)?;
    let d_len = w_down.n_out as usize;
    op(
        acc,
        FrameOp::SiluDiv {
            t: mf.lo,
            div: hc as f32,
            n: d_len,
        },
    )?;
    let w_up = if mf.up_head.data.is_empty() {
        model.w4(&format!("blk.{il}.nextn.hc_head_up.weight"))?
    } else {
        mf.up_head.w()
    };
    acc.frame_mm(mf.lo, &w_up, mf.hgate, 1)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::HcGateMean {
            xn: mf.hxn,
            gate: mf.hgate,
            out: mf.hin,
            hc,
            n,
        },
    )?;

    // 6) output(본체 공유) GEMV + GPU argmax — 로짓 전사 대신 토큰 1개 회수.
    let wout = model
        .w("output.weight")
        .ok_or(Q4Error::MissingTensor("output.weight".into()))?;
    acc.frame_mm(mf.hin, &wout, mf.logits, 1)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dftout", mf.logits)?;
    let toks = acc
        .frame_argmax_rows(mf.logits, 1, hp.vocab)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "dftargmax", mf.logits)?;

    // 7) 체인 h 반출 — 다음 드래프트 스텝의 hnorm 입력(프리-믹서 잔차).
    let mut chain = vec![0.0f32; hc * n];
    acc.frame_read(mf.res, &mut chain).map_err(Q4Error::Io)?;
    Ok((toks[0], chain))
}
