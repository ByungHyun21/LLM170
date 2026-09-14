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
    pub hin: u64, // [t·n_embd]
    pub hin_last: u64, // [n_embd] — 헤드 입력(마지막 토큰)
    pub logits: u64, // [vocab]
    // 저랭크 버퍼 길이 (hc_down/output_hc_down n_out — 생성 시 고정)
    pub lo_len: usize,
    pub hlo_len: usize,
    // ── 시퀀스별 상주 상태 (순환 idx / full idx 순) ──
    pub st_gdn: Vec<Vec<u64>>,  // [n_seqs][n_recr][dt_rank·d·d]
    pub st_conv: Vec<Vec<u64>>, // [n_seqs][n_recr][(k-1)·conv_ch]
    // ── 상수 (이름 → 핸들) — f32 norm류 등 스텝 프레임에 미리 상주 ──
    pub consts: HashMap<String, u64>,
    /// 시퀀스별 값 경로 prefill 이후 상태 재동기 필요 플래그.
    pub dirty: Vec<bool>,
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
        t_max: usize,
    ) -> Result<Self, Q4Error> {
        let hp = &model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let k_len = hp.n_group * hp.d_state;
        let v_len = hp.dt_rank * hp.d_state;
        let conv_ch = 2 * k_len + v_len;
        let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recr(il)).count();
        let t_max = t_max.max(1);
        let a = |len: usize| alloc(acc, len);
        let at = |len: usize| alloc(acc, len * t_max);
        let lo_n = model.w4("blk.0.hc_attn_down.weight")?.n_out as usize;
        let hlo_n = model.w4("output_hc_down.weight")?.n_out as usize;

        let mut f = Frame4 {
            res_hc: at(hc * n)?,
            xn: at(hc * n)?,
            lo: at(lo_n)?,
            gate: at(hc * n)?,
            inj: at(hc)?,
            mix: at(n)?,
            ffn_out: at(n)?,
            gqkv: at(conv_ch)?,
            gz: at(hp.d_inner)?,
            gb: at(hp.dt_rank)?,
            ga: at(hp.dt_rank)?,
            gbg: at(hp.dt_rank * 2)?,
            gconv: at(conv_ch)?,
            gq: at(k_len)?,
            gk: at(k_len)?,
            gv: at(v_len)?,
            go: at(v_len)?,
            ggated: at(hp.d_inner)?,
            mroute: at(hp.n_expert)?,
            msgate: at(1)?,
            mids: at(hp.n_expert_used)?,
            mwt: at(hp.n_expert_used)?,
            mxsel: at(hp.n_expert_used * n)?,
            mgu: at(hp.n_expert_used * hp.n_ff_exp)?,
            mup: at(hp.n_expert_used * hp.n_ff_exp)?,
            mglu: at(hp.n_expert_used * hp.n_ff_exp)?,
            my: at(hp.n_expert_used * n)?,
            mout: at(n)?,
            shg: at(hp.n_ff_exp)?,
            shu: at(hp.n_ff_exp)?,
            shglu: at(hp.n_ff_exp)?,
            shout: at(n)?,
            hxn: at(hc * n)?,
            hlo: at(hlo_n)?,
            hgate: at(hc * n)?,
            hin: at(n)?,
            hin_last: a(n)?,
            logits: a(hp.vocab)?,
            lo_len: lo_n,
            hlo_len: hlo_n,
            st_gdn: Vec::with_capacity(seqs.len()),
            st_conv: Vec::with_capacity(seqs.len()),
            consts: HashMap::new(),
            dirty: vec![true; seqs.len()],
        };
        // 시퀀스별 GDN 상태 핸들 세트 — np 디코드 지원 (스테이트 스왑 없이
        // 시퀀스 고유 핸들 세트를 소유; 활성화 버퍼는 스텝마다 재사용).
        for _ in 0..seqs.len() {
            let mut gdn = Vec::with_capacity(n_recr);
            let mut conv = Vec::with_capacity(n_recr);
            for _ in 0..n_recr {
                gdn.push(a(hp.dt_rank * hp.d_state * hp.d_state)?);
                conv.push(a((hp.conv_k - 1) * conv_ch)?);
            }
            f.st_gdn.push(gdn);
            f.st_conv.push(conv);
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
        // 전 시퀀스의 현재 CPU 상태를 초기값으로 (dirty 해소)
        for (si, st) in seqs.iter().enumerate() {
            f.sync_states(acc, si, st, hp.d_state)?;
        }
        Ok(f)
    }

    /// (dv,kdim) 전치 — AR 커널(gdn_ar_w_swap)이 kdim 연속 레이아웃을 쓴다.
    /// 이전 열 단위 접근은 d=128에서 512B 스트라이드로 대역폭을 32배 증폭시켰다
    /// (실측 AR 3.97s/청크 = 프리필 38%). 경계(h2d/d2h)에서만 전치한다.
    pub fn transpose_pairs(s: &[f32], d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s.len()];
        if d == 0 {
            return out;
        }
        let pair = d * d;
        for b in (0..s.len()).step_by(pair) {
            for kd in 0..d {
                for dv in 0..d {
                    out[b + dv * d + kd] = s[b + kd * d + dv];
                }
            }
        }
        out
    }

    /// CPU SeqState4의 GDN 상태를 GPU로 재동기 (prefill 직후) — 시퀀스 지정.
    pub fn sync_states(
        &mut self,
        acc: &dyn Accelerator,
        seq: usize,
        st: &SeqState4,
        d_state: usize,
    ) -> Result<(), Q4Error> {
        for (ri, h) in self.st_gdn[seq].iter().enumerate() {
            let t = Self::transpose_pairs(&st.gdn_s[ri], d_state);
            acc.frame_write(*h, &t).map_err(Q4Error::Io)?;
        }
        for (ri, h) in self.st_conv[seq].iter().enumerate() {
            acc.frame_write(*h, &st.conv[ri]).map_err(Q4Error::Io)?;
        }
        self.dirty[seq] = false;
        Ok(())
    }
}

/// 프레임 스텝 시작 알림 — 버퍼는 t_max 크기이므로 op 커널이 토큰 수를
/// 버퍼 길이에서 유도할 수 없다. 명시적으로 전달한다.
fn fs_begin(acc: &dyn Accelerator, t: usize) {
    let fs: &dyn FrameState = acc;
    fs.frame_begin(t);
}

// 스테이지 동기 마커 (LLM170_FRAME_SYNC=1) — 스티키 폴트의 발생 지점을
// 즉시 드러낸다(폴트는 다음 API 호출에서야 보고된다).
thread_local! {
    /// 스테이지 누적 시간 — (마지막 경계 시각, [(접미사, us, 호출수)]).
    static FT: std::cell::RefCell<(std::time::Instant, Vec<(String, u64, u64)>)> =
        std::cell::RefCell::new((std::time::Instant::now(), Vec::new()));
}

/// 프레임 스테이지 시간 계측 (LLM170_FRAME_TIME=1). sync_mark가 만드는 경계
/// 에서만 측정한다 — 프레임 op는 비동기라 호출 시간만으로는 GPU 시간이 안 나온다.
fn ftime_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LLM170_FRAME_TIME").is_some())
}

/// 진단용 스테이지 스킵(LLM170_STAGE_SKIP="qsa,gdn,moe") — 비용 분해 전용.
pub fn stage_skipped(name: &str) -> bool {
    std::env::var("LLM170_STAGE_SKIP")
        .map(|v| v.split(',').any(|x| x.trim() == name))
        .unwrap_or(false)
}

fn sync_mark(acc: &dyn Accelerator, tag: &str, h: u64) -> Result<(), Q4Error> {
    match (ftime_on(), std::env::var_os("LLM170_FRAME_SYNC").is_some()) {
        (false, false) => return Ok(()),
        (ft, sync) => {
            // 1원소 판독 = 동기 + 폴트 보고 (barrier는 오류를 삼킨다).
            let mut v = [0.0f32; 1];
            acc.frame_read(h, &mut v)
                .map_err(|e| Q4Error::Io(format!("fsync {tag}: {e}")))?;
            if ft {
                FT.with(|s| {
                    let mut s = s.borrow_mut();
                    let dt = s.0.elapsed().as_micros() as u64;
                    s.0 = std::time::Instant::now();
                    let key = tag.rsplit('.').next().unwrap_or(tag).to_string();
                    match s.1.iter_mut().find(|e| e.0 == key) {
                        Some(e) => {
                            e.1 += dt;
                            e.2 += 1;
                        }
                        None => s.1.push((key, dt, 1)),
                    }
                });
            }
            if sync {
                eprintln!("# fsync {tag}");
            }
        }
    }
    Ok(())
}

/// 청크/스텝 단위 리포트 — 누적이 있으면 한 줄 출력 후 초기화.
/// (디코드 t=1도 찍는다: 프리필과 달리 동기 지점이 많아 누적-지연 왜곡이 없다.)
fn ftime_report(_t: usize) {
    if !ftime_on() {
        return;
    }
    FT.with(|s| {
        let mut s = s.borrow_mut();
        if !s.1.is_empty() {
            s.1.sort_by(|a, b| b.1.cmp(&a.1));
            let mut line = String::from("# frame-time(t) ");
            for (k, us, n) in s.1.iter() {
                line.push_str(&format!("{k}={:.1}ms×{n} ", *us as f64 / 1e3));
            }
            eprintln!("{line}");
            s.1.clear();
        }
        s.0 = std::time::Instant::now();
    });
}

/// 단계 덤프 (LLM170_Q4_DBG=1) — 값 경로와 같은 양을 찍어 대조한다.
fn dbg(tag: &str, acc: &dyn Accelerator, h: u64, n: usize) {
    if std::env::var_os("LLM170_Q4_DBG").is_none() {
        return;
    }
    let mut v = vec![0.0f32; n];
    if acc.frame_read(h, &mut v).is_err() {
        return;
    }
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    let mx = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    eprintln!("# fdbg {tag}: sum={s:.6} max={mx:.6} v0..3={:?}", &v[..3.min(n)]);
}

/// 프레임 forward — t토큰 (t=1 디코드도 이 경로; decode_frame이 래퍼).
pub fn frame_forward(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq: usize,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<Vec<f32>, Q4Error> {
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
        // 1) PLE (blk.1) — CPU 브리지: res_hc 판독 → CPU → 기록
        if hp.is_ple(il) {
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

        // 2) hc attn mix
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc, t)?;
        sync_mark(acc, &format!("L{il}.hc_attn"), f.mix)?;
        if il == 0 {
            dbg("res_hc", acc, f.res_hc, hc * n * t);
        }

        // 3) attention — GDN 프레임 / QSA 값 브리지
        if hp.is_recr(il) {
            if stage_skipped("gdn") {
                // 진단용: GDN 단계 생략(출력 무효) — 디코드 스텝 비용 분해.
            } else {
            gdn_frame(acc, model, f, il, seq, recr_idx, conv_ch, k_len, v_len, eps, t)?;
            }
            recr_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
            sync_mark(acc, &format!("L{il}.gdn_combine"), f.res_hc)?;
        } else {
            // QSA 값 경로 브리지 — mix 판독 → qsa_layer(t행) → 출력 기록
            if stage_skipped("qsa") {
                // 진단용: QSA 브리지 생략(출력 무효).
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
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
        }

        // 4) hc ffn mix + MoE
        if il == 0 {
        }
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc, t)?;
        sync_mark(acc, &format!("L{il}.hc_ffn"), f.mix)?;
        if il == 0 {
            dbg("mix2", acc, f.mix, n * t);
        }
        moe_frame(acc, model, f, il, n, t)?;
        sync_mark(acc, &format!("L{il}.moe"), f.mout)?;
        if il == 0 {
        }
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
        sync_mark(acc, &format!("L{il}.ffn_combine"), f.res_hc)?;
        if il == 0 {
        }
    }

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
        Ok(logits)
    }
}

/// 프레임 디코드 1스텝 — Engine4::decode1에서 호출.
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

/// hc_mix 프레임 — CPU stages/hc.rs hc_mix와 동일 순서 (inject 반환 포함).
#[allow(clippy::too_many_arguments)]
fn hc_mix_frame(
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
    op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.xn, eps, n, w_reps: hc })?;
    sync_mark(acc, "hc.rms", f.xn)?;
    let w_down = model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
    let w_inject = model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;
    acc.frame_mm_group(f.xn, &[w_down, w_inject], &[f.lo, f.inj], t)
        .map_err(Q4Error::Io)?;
    sync_mark(acc, "hc.down", f.lo)?;
    op(acc, FrameOp::SiluDiv { t: f.lo, div: hc as f32, n: f.lo_len * t })?;
    sync_mark(acc, "hc.silu", f.lo)?;
    let w_up = model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?;
    acc.frame_mm(f.lo, &w_up, f.gate, t).map_err(Q4Error::Io)?;
    sync_mark(acc, "hc.up", f.gate)?;
    op(acc, FrameOp::HcGateMean { xn: f.xn, gate: f.gate, out: f.mix, hc, n })?;
    sync_mark(acc, "hc.gate", f.mix)?;
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
    t: usize,
) -> Result<(), Q4Error> {
    op(acc, FrameOp::HcCombine { res: f.res_hc, out, inj, hc, n, total: hc * n * t })
}

/// GDN 프레임 — stages/gdn.rs와 동일 순서 (t=1).
#[allow(clippy::too_many_arguments)]
fn gdn_frame(
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
    // β/e^g
    let dtb = f.consts[&format!("blk.{il}.dt_bias")];
    let ssa = f.consts[&format!("blk.{il}.ssm_a")];
    if !stage_skipped("gdn.betag") {
        op(acc, FrameOp::GdnBetaG { b: f.gb, a: f.ga, dtb, sa: ssa, bg: f.gbg, n_h: hp.dt_rank * t })?;
    }
    sync_mark(acc, "gdn.betag", f.gbg)?;
    // conv + ring
    let cw = f.consts[&format!("blk.{il}.conv_w")];
    if !stage_skipped("gdn.conv") {
        op(acc, FrameOp::GdnConv { qkv: f.gqkv, cw, state: f.st_conv[seq][ri], out: f.gconv, ch: conv_ch, k: hp.conv_k, t_len: t })?;
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
    // AR 상태 갱신 — 상태 GPU 상주, 판독 없음
    let fs: &dyn FrameState = acc;
    if il == 0 {
        dbg("gbg_post", acc, f.gbg, hp.dt_rank * 2 * t);
    }
    if !stage_skipped("gdn.ar") {
        fs.frame_gdn_ar(f.gq, f.gk, f.gv, f.gbg, f.st_gdn[seq][ri], f.go, 1, hp.n_group, hp.dt_rank, hp.d_state)
            .map_err(Q4Error::Io)?;
    }
    sync_mark(acc, "gdn.ar", f.go)?;
    // norm_gated + out proj
    let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
    if !stage_skipped("gdn.ng") {
        op(acc, FrameOp::NormGated { o: f.go, z: f.gz, w: snorm, out: f.ggated, eps, d: hp.d_state, n_h: hp.dt_rank })?;
    }
    sync_mark(acc, "gdn.normgated", f.ggated)?;
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
fn moe_frame(
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
        sync_mark(acc, "moe.gather", f.mxsel)?;
        fs.frame_moe_gemm(f.mxsel, &w_gate, f.mids, f.mgu, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
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
    if !stage_skipped("moe.shared") {
        op(acc, FrameOp::Sigmoid { t: f.msgate, n: t })?;
        let shg_w = model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
        let shu_w = model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
        acc.frame_mm_group(f.mix, &[shg_w, shu_w], &[f.shg, f.shu], t)
            .map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluMul { g: f.shg, u: f.shu, out: f.shglu, n: n_ff * t })?;
        let shd_w = model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
        acc.frame_mm(f.shglu, &shd_w, f.shout, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::AxpyScaled { y: f.mout, x: f.shout, s: f.msgate, n: n * t })?;
    }
    sync_mark(acc, "moe.shared", f.mout)?;
    Ok(())
}

