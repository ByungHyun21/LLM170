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
    // QSA (plans/67 2c — 디바이스 상주 투영·어텐션)
    pub qsa_q: u64,    // [t][n_head·2hd] — wq 출력(q‖게이트 인터리브)
    pub qsa_k: u64,    // [t][n_kv·hd] — norm+rope 후 k
    pub qsa_v: u64,    // [t][n_kv·hd] — v
    pub qsa_iq: u64,   // [t][idx_heads·idx_dim]
    pub qsa_ik: u64,   // [t][idx_dim]
    pub qsa_attn: u64, // [t][n_head·hd] — 어텐션 출력(wo 입력)
    // PLE (plans/73 — 디바이스 수학)
    pub ple_emb: u64,     // [ple_heads*ple_head_dim] 게이트된 n-gram 임베딩
    pub ple_key: u64,     // [hc·n] w_key 출력
    pub ple_value: u64,   // [n] w_value 출력(스트림 공유)
    pub ple_gated: u64,   // [hc·n] norm된 게이트 방송(conv 입력)
    pub ple_conv_out: u64,// [hc·n]
    pub ple_gate: u64,    // [hc]
    /// rope cos/sin 테이블 호스트 사본 — frame_qk_norm_rope가 받아 올린다.
    pub qsa_cs: Vec<f32>,
    /// 인덱서 로프 cos/sin 테이블(π n_rot=idx_dim) — 디바이스 선택(plans/73)이
    /// 받아 올린다. qsa_cs와 동일 산술, 다른 n_rot.
    pub qsa_cs_idx: Vec<f32>,
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
            ple_emb: at(hp.ple_heads_per_ngram * 2 * hp.ple_head_dim)?,
            ple_key: at(hc * n)?,
            ple_value: at(n)?,
            ple_gated: at(hc * n)?,
            ple_conv_out: at(hc * n)?,
            ple_gate: at(hc)?,
            shout: at(n)?,
            qsa_q: at(hp.n_head * 2 * hp.head_dim)?,
            qsa_k: at(hp.n_kv * hp.head_dim)?,
            qsa_v: at(hp.n_kv * hp.head_dim)?,
            qsa_iq: at(hp.idx_heads * hp.idx_dim)?,
            qsa_ik: at(hp.idx_dim)?,
            qsa_attn: at(hp.n_head * hp.head_dim)?,
            qsa_cs: {
                let ctx_n = seqs
                    .first()
                    .and_then(|s| s.kv_k.first())
                    .map(|k| k.len() / (hp.n_kv.max(1) * hp.head_dim.max(1)))
                    .unwrap_or(8192);
                // Hparams4에 rope_cs 헬퍼가 없어 로컬 빌드 — 산술은
                // model/hparams.rs rope_cs(=ops::rope_head)와 동일 값을 쓴다.
                let (half, base) = (hp.n_rot / 2, hp.rope_base);
                let mut cs = vec![0.0f32; ctx_n * half * 2];
                for pos in 0..ctx_n {
                    for pp in 0..half {
                        let theta = base.powf(-(2.0 * pp as f32) / hp.n_rot as f32);
                        let angle = pos as f32 * theta;
                        cs[pos * half * 2 + pp * 2] = angle.cos();
                        cs[pos * half * 2 + pp * 2 + 1] = angle.sin();
                    }
                }
                cs
            },
            qsa_cs_idx: {
                // 인덱서 로프 — rope_head(pos, n_rot=idx_dim, base)와 동일 값을
                // 쓴다(디바이스 q4_idx_q_rope/bk_update가 소비, plans/73).
                let ctx_n = seqs
                    .first()
                    .and_then(|s| s.idx_k.first())
                    .map(|k| k.len() / hp.idx_dim.max(1))
                    .unwrap_or(8192);
                let (half, base) = (hp.idx_dim / 2, hp.rope_base);
                let mut cs = vec![0.0f32; ctx_n * half * 2];
                for pos in 0..ctx_n {
                    for pp in 0..half {
                        let theta = base.powf(-(2.0 * pp as f32) / hp.idx_dim as f32);
                        let angle = pos as f32 * theta;
                        cs[pos * half * 2 + pp * 2] = angle.cos();
                        cs[pos * half * 2 + pp * 2 + 1] = angle.sin();
                    }
                }
                cs
            },
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
        // 1) PLE (blk.1) — plans/73: 디코드(t=1)는 디바이스 경로. 해시/gather는
        //    스텝 초에 호스트가 끝냈고(GPU 무의존), key/value 투영은 프레임 GEMM,
        //    gate/conv/잔차는 ple_math_dev 의 3커널 — 동기 d2h/h2d 왕복과
        //    CPU mm_batch 투영 2회([2560→10240])가 사라진다(4.5-11ms/step).
        //    폴백/프리필(t>1)은 기존 호스트 브리지. LLM170_PLE_HOST=1 강제.
        if hp.is_ple(il) {
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
            // QSA — plans/67 2c: 디바이스 상주 경로 우선. 투영·norm·rope·어텐션·
            // wo가 전부 GPU에 있고 d2h는 iq/ik/k/v(캐시 적립)뿐이다. 초기 3단계
            // (mm_group/qk_norm_rope/판독) 실패 시에만 구값 브리지로 폴백 — 그
            // 시점엔 캐시 미변경이라 이중 적립이 없다.
            if stage_skipped("qsa") {
                // 진단용: QSA 브리지 생략(출력 무효).
            } else if qsa_frame(acc, model, ctx, seq_st, f, il, t, full_idx, seq).is_ok() {
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

/// QSA 프레임 (plans/67 2c) — 투영·norm·rope·어텐션·wo 전부 디바이스 상주.
/// d2h는 캐시 적립용 iq/ik/k/v(t×(idx_heads·idx_dim+idx_dim+2·n_kv·hd) ≈ t×3,840
/// floats)뿐 — 기존 값 브리지는 mix+wq+wo 왕복 t×~10,880 floats를 나르던 것과
/// 비교해 첫 3단계 실패 시에만 호출부가 값 브리지로 폴백한다(그 시점엔 아직
/// 캐시를 건드리지 않는다 — 이중 적립 없음).
#[allow(clippy::too_many_arguments)]
fn qsa_frame(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq_st: &mut SeqState4,
    f: &mut Frame4,
    il: usize,
    t: usize,
    full_idx: usize,
    seq: usize,
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
    acc.frame_mm_group(
        f.mix,
        &[wq, wk, wv, w_iq, w_ik],
        &[f.qsa_q, f.qsa_k, f.qsa_v, f.qsa_iq, f.qsa_ik],
        t,
    )
    .map_err(Q4Error::Io)?;
    sync_mark(acc, "qsa.mm_group", f.qsa_q)?;
    // 2) q/k norm+rope in-place — 커널 산술은 호스트 rms_norm(sq_sum 32세그먼트
    //    f64)+rope_head(f64 회전)와 동일열(비트 동일 기대).
    let pos0 = seq_st.pos;
    // qk_norm_rope 커널은 norm 가중치를 **헤드별 타일**(qw[r0·hd..])로 읽는다
    // (decode 경로는 rawinject가 타일해 업로드 — ssm_norm 타일링과 같은 규약).
    // 공유 [hd] 원본을 그대로 올리면 24헤드 분량(6144)을 256원소 버퍼에서 읽어
    // illegal address(700)로 폭주한다 — plans/67 2c 연결 시 실측 발견(2026-09-14).
    let qn_raw = model.f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
    let kn_raw = model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
    let qn: Vec<f32> = qn_raw.iter().copied().cycle().take(qn_raw.len() * n_head).collect();
    let kn: Vec<f32> = kn_raw.iter().copied().cycle().take(kn_raw.len() * n_kv).collect();
    acc.frame_qk_norm_rope(
        f.qsa_q, f.qsa_k, &qn, &kn, &f.qsa_cs, hp.eps, pos0 as usize,
        n_head, n_kv, hd, n_rot, t,
    )
    .map_err(Q4Error::Io)?;
    sync_mark(acc, "qsa.qkrope", f.qsa_k)?;
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
        let iqw = model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
        let ikw = model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        let dev = acc
            .qsa_sel_dev(
                full_idx, seq, f.qsa_iq, f.qsa_ik, t, pos0 as usize,
                hp.idx_heads, hp.idx_dim, r, hp.idx_top_k,
                &iqw, &ikw, &f.qsa_cs_idx, hp.eps,
            )
            .and_then(|(sd, od, list_len)| {
                acc.qsa_kv_dev(full_idx, seq, f.qsa_k, f.qsa_v, t, pos0 as usize, n_kv, hd)
                    .and_then(|(kc, vc)| {
                        acc.qsa_attention_dev_sel(
                            f.qsa_q, kc, vc, sd, od, list_len, kq_scale,
                            n_head, n_kv, hd, t, f.qsa_attn,
                        )
                        .map(|_| (sd, od, list_len))
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
                acc.frame_mm_group(f.qsa_attn, &[wo], &[f.ffn_out], t)
                    .map_err(Q4Error::Io)?;
                return Ok(());
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("# qsa-frame: 디바이스 선택 폴백 — 호스트 경로 ({e})"));
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
    acc.frame_read(f.qsa_iq, &mut iq_v).map_err(Q4Error::Io)?;
    acc.frame_read(f.qsa_ik, &mut ik_v).map_err(Q4Error::Io)?;
    acc.frame_read(f.qsa_k, &mut k_v).map_err(Q4Error::Io)?;
    acc.frame_read(f.qsa_v, &mut v_v).map_err(Q4Error::Io)?;
    sync_mark(acc, "qsa.d2h", f.qsa_v)?;
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
        acc.qsa_kv_dev(full_idx, seq, f.qsa_k, f.qsa_v, t, pos0 as usize, n_kv, hd)
            .and_then(|(kc, vc)| {
                acc.qsa_attention_dev_res(
                    f.qsa_q, kc, vc, &sel_idx, &sel_off, kq_scale,
                    n_head, n_kv, hd, t, f.qsa_attn,
                )
            })
    };
    let ck = &seq_st.kv_k[full_idx][..kn_max];
    let cv = &seq_st.kv_v[full_idx][..kn_max];
    if std::env::var_os("LLM170_QSA_RESCHECK").is_some() && !seq_st.qsa_host_stale {
        if let Err(e) = acc.qsa_kv_check(full_idx, seq, ck, cv) {
            eprintln!("# qsa-rescheck L{il} t={t} pos0={pos0}: {e}");
        }
    }
    let attn = res.or_else(|e2| {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| eprintln!("# qsa-frame: 상주 풀 미사용 — 업로드 경로 ({e2})"));
        acc.qsa_attention_dev(
            f.qsa_q, ck, cv, &sel_idx, &sel_off, kq_scale,
            n_head, n_kv, hd, t, f.qsa_attn,
        )
        .map_err(|e| format!("{e2}; {e}"))
    });
    if let Err(e) = attn {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!("# qsa-frame: GPU 어텐션 폴백 — CPU 재계산 ({e})")
        });
        let mut q_v = vec![0.0f32; t * n_head * 2 * hd];
        acc.frame_read(f.qsa_q, &mut q_v).map_err(Q4Error::Io)?;
        let qg = rows(&q_v, n_head * 2 * hd);
        let attn = stages::qsa_cpu_attn_rows(
            &qg, seq_st, full_idx, &sel_blk, &sel_cnt, sel_stride, r, pos0 as usize, t,
            n_head, n_kv, hd, kq_scale,
        );
        let flat: Vec<f32> = attn.concat();
        acc.frame_write(f.qsa_attn, &flat).map_err(Q4Error::Io)?;
    }
    // 6) wo 투영 — 어텐션 출력을 디바이스에서 ffn_out으로(왕복 0).
    if qtm {
        eprintln!("# qsa-frame L{il} t={t} attn={:.2}ms", lap.elapsed().as_secs_f64() * 1e3);
        lap = std::time::Instant::now();
    }
    acc.frame_mm_group(f.qsa_attn, &[wo], &[f.ffn_out], t)
        .map_err(Q4Error::Io)?;
    let _ = &mut lap;
    Ok(())
}


/// SELCHECK 검증 그림자(plans/73) — 디바이스 선택과 동일 입력으로 호스트
/// 선택을 재계산해 목록을 돌려준다. 기존 d2h+qsa_select 경로를 그대로 쓰므로
/// 호스트 kv/idx 캐시도 함께 갱신된다(검증 모드에선 stale가 유지되지 않음).
#[allow(clippy::too_many_arguments)]
fn qsa_selcheck_host(
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
        let shg_w = model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
        let shu_w = model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
        let shd_w = model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
        // plans/72: t=1은 융합 2런치(gate+up+silu → down+sigmoid·axpy).
        // 기존 8런치(quant×2+gemv×3+sigmoid+silu+axpy)가 19.4ms/step의
        // 지배 항이었다 — 런치 오버헤프 지배(유효 대역폭 1.5GB/s).
        if t == 1 && std::env::var_os("LLM170_NO_SHEXP_FUSED").is_none() {
            op(acc, FrameOp::Sigmoid { t: f.msgate, n: t })?;
            acc.shexp_gu(f.mix, &shg_w, &shu_w, f.shglu, n, n_ff)
                .map_err(Q4Error::Io)?;
            acc.shexp_da(f.shglu, &shd_w, f.msgate, f.mout, n, n_ff)
                .map_err(Q4Error::Io)?;
        } else {
            op(acc, FrameOp::Sigmoid { t: f.msgate, n: t })?;
            acc.frame_mm_group(f.mix, &[shg_w, shu_w], &[f.shg, f.shu], t)
                .map_err(Q4Error::Io)?;
            op(acc, FrameOp::SiluMul { g: f.shg, u: f.shu, out: f.shglu, n: n_ff * t })?;
            acc.frame_mm(f.shglu, &shd_w, f.shout, t).map_err(Q4Error::Io)?;
            op(acc, FrameOp::AxpyScaled { y: f.mout, x: f.shout, s: f.msgate, n: n * t })?;
        }
    }
    sync_mark(acc, "moe.shared", f.mout)?;
    Ok(())
}

