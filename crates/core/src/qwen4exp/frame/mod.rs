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

mod diag;
mod forward;
mod multi;
mod np;

use diag::{buf_hash, dbg, ftime_on, ftime_report, frame_ck, sync_mark};

pub use diag::stage_skipped;
pub use forward::*;
pub use multi::*;
pub use np::*;

use super::stages::{self, Ctx};
use super::{Hparams4, Model4, Q4Error};
use crate::matmul::{Accelerator, FrameOp, FrameState};
use crate::quant::dequant_row;
use std::collections::HashMap;

/// 프레임 버퍼 집합 — 활성화(스텝 공용) + 상태(시퀀스별) + 상수 가중치.
/// np 배치 디코드용 행 뷰 핸들 — per-seq 상태 op에 넘긴다(초기화 1회).
pub struct NpViews {
    pub res_hc: Vec<u64>,   // [row][hc·n]
    pub gqkv: Vec<u64>,     // [row][conv_ch]
    pub gconv: Vec<u64>,    // [row][conv_ch]
    pub gq: Vec<u64>,       // [row][k_len]
    pub gk: Vec<u64>,
    pub gv: Vec<u64>,
    pub gbg: Vec<u64>,      // [row][dt_rank·2]
    pub go: Vec<u64>,       // [row][v_len]
    pub qsa_q: Vec<u64>,    // [row][n_head·2hd]
    pub qsa_k: Vec<u64>,    // [row][n_kv·hd]
    pub qsa_v: Vec<u64>,
    pub qsa_iq: Vec<u64>,   // [row][idx_heads·idx_dim]
    pub qsa_ik: Vec<u64>,   // [row][idx_dim]
    pub qsa_attn: Vec<u64>, // [row][n_head·hd]
    pub mix: Vec<u64>,      // [row][n_embd] — MoE per-seq t=1 경로 입력
    pub mout: Vec<u64>,     // [row][n_embd] — MoE 출력(가중합 목적지)
}

/// 다중 시퀀스 청크 프리필용 per-seq **행 대역** 핸들 — 시퀀스 si의 행
/// [si·per_seq, (si+1)·per_seq). np_views(행 1개)와 달리 per_seq행 연속
/// 뷰다 — GDN conv/AR은 같은 시퀀스의 행들이 **한 호출**에서 사슬로 이어져야
/// 한다(행별 독립 호출은 순환 상태가 경합해 순차 결과와 갈라진다).
/// 기하 (n_seq, per_seq)가 바뀔 때만 재생성 — frame_free가 no-op이라
/// 핸들 테이블은 청크 크기 종류 수만큼만 늘어난다.
pub struct PreViews {
    pub slots: usize, // 생성 시 n_seq
    pub rows: usize,  // 생성 시 per_seq
    pub res_hc: Vec<u64>, // [seq][rows·hc·n] — PLE 브리지 입출력
    pub gqkv: Vec<u64>,   // [seq][rows·conv_ch]
    pub gconv: Vec<u64>,  // [seq][rows·conv_ch]
    pub gq: Vec<u64>,     // [seq][rows·k_len]
    pub gk: Vec<u64>,
    pub gv: Vec<u64>,     // [seq][rows·v_len]
    pub gbg: Vec<u64>,    // [seq][rows·dt_rank·2]
    pub go: Vec<u64>,     // [seq][rows·v_len]
    pub qsa_q: Vec<u64>,  // [seq][rows·n_head·2hd]
    pub qsa_k: Vec<u64>,  // [seq][rows·n_kv·hd]
    pub qsa_v: Vec<u64>,
    pub qsa_iq: Vec<u64>,   // [seq][rows·idx_heads·idx_dim]
    pub qsa_ik: Vec<u64>,   // [seq][rows·idx_dim]
    pub qsa_attn: Vec<u64>, // [seq][rows·n_head·hd]
    pub mix: Vec<u64>,      // [seq][rows·n] — QSA 5투영 입력
    pub ffn_out: Vec<u64>,  // [seq][rows·n] — QSA wo 출력
    pub hin_last: Vec<u64>, // [seq][n] — 헤드 입력(시퀀스별 마지막 행)
    pub logits: Vec<u64>,   // [seq][vocab] — logits_t 행 뷰
}

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
    /// np 배치 디코드 헤드 출력 [8][vocab] — t=8 행까지 한 번의 head GEMM으로.
    pub logits_t: u64,
    /// np 배치 디코드 행 뷰 캐시(1회 생성) — None이면 아직 np 스텝 없음.
    pub np_views: Option<Box<NpViews>>,
    /// 다중 시퀀스 청크 프리필 행 뷰 캐시(기하별 1회) — None이면 아직 없음.
    pub pre_views: Option<Box<PreViews>>,
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
    /// QSA q/k norm의 헤드 타일 사본 (full_idx 순) — 스텝마다 새 Vec을 만들어
    /// 가속기 업로드 캐시가 **매 층 미스**하던 것(24KB 동기 복사 ×2/층)을 막는다.
    /// 포인터가 고정이라 가속기의 포인터 키 캐시가 상주한다.
    pub qsa_qn_t: Vec<Vec<f32>>,
    pub qsa_kn_t: Vec<Vec<f32>>,
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
    /// 활성 버퍼 행 상한(생성 인자 t_max) — 다중 프리필 용량 검사용.
    pub t_max: usize,
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
                // 산술은 ops::rope_cs_table 단일 소스(plans/90 A1 D5).
                crate::ops::rope_cs_table(hp.n_rot, hp.rope_base, ctx_n)
            },
            qsa_cs_idx: {
                // 인덱서 로프 — n_rot=idx_dim 판(디바이스 q4_idx_q_rope/
                // bk_update가 소비, plans/73).
                let ctx_n = seqs
                    .first()
                    .and_then(|s| s.idx_k.first())
                    .map(|k| k.len() / hp.idx_dim.max(1))
                    .unwrap_or(8192);
                crate::ops::rope_cs_table(hp.idx_dim, hp.rope_base, ctx_n)
            },
            qsa_qn_t: {
                // 헤드 타일: qk_norm_rope 커널이 qw[r0·hd..] 형태로 읽는다.
                let mut v = Vec::new();
                for il in 0..hp.n_layer {
                    if hp.is_recr(il) {
                        continue;
                    }
                    let src = model.f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
                    v.push(src.iter().copied().cycle().take(src.len() * hp.n_head).collect());
                }
                v
            },
            qsa_kn_t: {
                let mut v = Vec::new();
                for il in 0..hp.n_layer {
                    if hp.is_recr(il) {
                        continue;
                    }
                    let src = model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
                    v.push(src.iter().copied().cycle().take(src.len() * hp.n_kv).collect());
                }
                v
            },
            hxn: at(hc * n)?,
            hlo: at(hlo_n)?,
            hgate: at(hc * n)?,
            hin: at(n)?,
            hin_last: a(n)?,
            logits: a(hp.vocab)?,
            logits_t: a(hp.vocab * 8)?,
            np_views: None,
            pre_views: None,
            lo_len: lo_n,
            hlo_len: hlo_n,
            t_max,
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

/// plans/86 §3 — 프레임 시도 트랜잭션 스냅샷.
///
/// 프레임 디코드 시도는 실패 직전까지 호스트 상태를 부분 진화시킨다:
/// `ple_hash`는 hist/next_pos 를, `ple_block`은 conv 링을 진화시킨다. 실패 후
/// 값경로 폴백이 같은 스텝을 처음부터 재계산하면 이중 진화가 된다 — 특히
/// ple_hash 재실행은 `hist_valid`가 이미 깨져(eos 패딩 히스토리) 잘못된
/// n-gram 행을 낸다(85 실측: 중단 지점별 폴백 토큰 66/18/14078 분기).
///
/// pos 는 시도 중 불변(성공 후 호출부가 진행), QSA kv/idx 캐시는 pos 인덱스
/// 쓰기라 동일 스텝 재실행에 멱등 — 스냅샷 대상에서 제외. GPU 상주 상태는
/// 폴백과 함께 프레임이 폐기되므로 무관.
pub struct PleSnap {
    hist: Vec<u32>,
    next_pos: u32,
    conv: Vec<f32>,
}

pub fn ple_snap(st: &SeqState4) -> PleSnap {
    PleSnap {
        hist: st.ple_hist.clone(),
        next_pos: st.ple_next_pos,
        conv: st.ple_conv.clone(),
    }
}

pub fn ple_restore(st: &mut SeqState4, s: PleSnap) {
    st.ple_hist = s.hist;
    st.ple_next_pos = s.next_pos;
    st.ple_conv = s.conv;
}

/// 프레임 스텝 시작 알림 — 버퍼는 t_max 크기이므로 op 커널이 토큰 수를
/// 버퍼 길이에서 유도할 수 없다. 명시적으로 전달한다.
fn fs_begin(acc: &dyn Accelerator, t: usize) {
    let fs: &dyn FrameState = acc;
    fs.frame_begin(t);
}
