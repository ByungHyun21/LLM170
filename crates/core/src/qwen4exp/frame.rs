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

/// 진단용 프레임 체크섬 — `LLM170_NP_CHECKSUM=1`. np·단일·배치 경로 공용.
/// 버퍼 앞 t·n개를 전부 읽어 합과 행 표본(첫·중간·마지막 행의 첫 원소)을
/// 보고한다. 청크 크기가 다른 두 실행에서 "같은 층·같은 단계·같은 토큰 수"를
/// 맞대어 첫 발산 지점을 찾는 용도 — 합만으로는 상쇄로 가려질 수 있어 행
/// 표본을 함께 낸다. 기본 꺼짐(1회 판독).
fn frame_ck(acc: &dyn Accelerator, h: u64, n: usize, t: usize, tag: &str) {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("LLM170_NP_CHECKSUM").is_some());
    if !*ON {
        return;
    }
    let mut v = vec![0.0f32; n * t];
    if acc.frame_read(h, &mut v).is_ok() {
        let s: f64 = v.iter().map(|&x| x as f64).sum();
        let mid = v[(t / 2) * n];
        let last = v[(t - 1) * n];
        eprintln!(
            "[npck] {tag} t={t} sum={s:.6} v0={:.6} mid0={mid:.6} last0={last:.6}",
            v[0]
        );
    }
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
fn frame_forward_ex(
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
            frame_ck(acc, f.res_hc, n, t, &format!("L{il}.res_in"));
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
        if il == 0 {
            dbg("mix2", acc, f.mix, n * t);
        }
        moe_frame(acc, model, f, il, n, t)?;
        sync_mark(acc, &format!("L{il}.moe"), f.mout)?;
        if il < 4 {
            frame_ck(acc, f.mout, n, t, &format!("L{il}.moe"));
        }
        
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
        sync_mark(acc, &format!("L{il}.ffn_combine"), f.res_hc)?;
        if il == 0 {
        }
    }
    frame_ck(acc, f.res_hc, n, t, "head.res");

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
fn ensure_np_views(
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
    // 뷰는 항상 최대 슬롯(8)로 만든다 — 첫 배치가 2슬롯이어도 이후 4슬롯
    // 스텝이 같은 뷰 테이블을 쓴다(패닉 방지, frames 테이블 무한 증가 방지).
    const NP_MAX: usize = 8;
    let rows = rows.max(NP_MAX);
    let dt2 = hp.dt_rank * 2;
    let qrow = hp.n_head * 2 * hp.head_dim;
    let kvrow = hp.n_kv * hp.head_dim;
    let iqrow = hp.idx_heads * hp.idx_dim;
    let arow = hp.n_head * hp.head_dim;
    let mk = |acc: &dyn Accelerator, h: u64, row_len: usize| -> Result<Vec<u64>, Q4Error> {
        (0..rows)
            .map(|r| acc.frame_slice(h, r * row_len, row_len).map_err(Q4Error::Io))
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
    };
    f.np_views = Some(Box::new(v));
    Ok(())
}

/// 청크 프리필 행 대역 뷰 — 기하 (n_seq, per_seq)가 바뀔 때만 재생성한다.
/// frame_slice 핸들은 반납되지 않으므로(ADR-0014) 스텝마다 만들면 테이블이
/// 무한히 큰다 — 청크 크기 종류 수만큼만 늘어나게 고정한다.
#[allow(clippy::too_many_arguments)]
fn ensure_pre_views(
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

/// GDN 프레임(np) — mm_group/betag/split/l2/scale/normgated/out은 t=rows 공유,
/// conv와 AR만 per-seq(행 뷰 + 해당 seq 상태).
#[allow(clippy::too_many_arguments)]
fn gdn_frame_np(
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
    let wqkv = model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
    let wz = model.w4(&format!("blk.{il}.attn_gate.weight"))?;
    let wb = model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
    let wa = model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
    acc.frame_mm_group(f.mix, &[wqkv, wz, wb, wa], &[f.gqkv, f.gz, f.gb, f.ga], t)
        .map_err(Q4Error::Io)?;
    let dtb = f.consts[&format!("blk.{il}.dt_bias")];
    let ssa = f.consts[&format!("blk.{il}.ssm_a")];
    op(acc, FrameOp::GdnBetaG { b: f.gb, a: f.ga, dtb, sa: ssa, bg: f.gbg, n_h: hp.dt_rank * t })?;
    let cw = f.consts[&format!("blk.{il}.conv_w")];
    let vv = f.np_views.as_ref().unwrap();
    // NP 디버그 프로브(2026-09-16): conv/AR 직후 행0 합계
    let npdbg = std::env::var_os("LLM170_NP_DBG").is_some() && il == 0;
    let psum = |acc: &dyn Accelerator, h: u64, n2: usize, tag: &str| {
        if npdbg {
            let mut v = vec![0.0f32; n2];
            if acc.frame_read(h, &mut v).is_ok() {
                eprintln!("# npdbg {tag}: sum={:.6} v0={:.6}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0]);
            }
        }
    };
    // conv(링) — plans/74 N2: 행별 상태를 포인터 테이블로 1런치. 실패 시
    // 종전 행별 t=1 루프(상태 커널이 t_cur 로 행 수를 유추해 t_cur=1로 내린다).
    // qkv/gconv는 [t][ch] 연속 프레임 버퍼라 정본 핸들 직접.
    let conv_states: Vec<u64> = seqs.iter().map(|&sq| f.st_conv[sq][ri]).collect();
    if seqs.len() > 1
        && std::env::var_os("LLM170_NO_NPCONV").is_none()
        && acc
            .frame_gdn_conv_np(f.gqkv, f.gconv, &conv_states, cw, conv_ch, hp.conv_k)
            .is_ok()
    {
        fs_begin(acc, t);
    } else {
        fs_begin(acc, 1);
        for (row, &sq) in seqs.iter().enumerate() {
            op(acc, FrameOp::GdnConv {
                qkv: vv.gqkv[row], cw, state: f.st_conv[sq][ri], out: vv.gconv[row],
                ch: conv_ch, k: hp.conv_k, t_len: 1,
            })?;
        }
        fs_begin(acc, t);
    }
    fs_begin(acc, t); // split/l2/scale는 전 행 배치
    psum(acc, vv.gconv[0], conv_ch, "conv_row0");
    if seqs.len() > 1 { psum(acc, vv.gconv[1], conv_ch, "conv_row1"); }
    // split/l2/scale — 행별 독립 원소연산, t 배치 그대로
    op(acc, FrameOp::Split3 { src: f.gconv, d0: f.gq, d1: f.gk, d2: f.gv, n0: k_len, n1: k_len, n2: v_len })?;
    op(acc, FrameOp::L2Rows { x: f.gq, eps, d: hp.d_state, n: k_len * t })?;
    op(acc, FrameOp::L2Rows { x: f.gk, eps, d: hp.d_state, n: k_len * t })?;
    let scale = 1.0f32 / (hp.d_state as f32).sqrt();
    op(acc, FrameOp::Scale { t: f.gq, s: scale, n: k_len * t })?;
    // AR(상태) — per-seq t=1 (다시 내림)
    // AR(상태) — plans/74 N2: 행별 상태 테이블 1런치. 실패 시 종전 행별 t=1.
    let ar_states: Vec<u64> = seqs.iter().map(|&sq| f.st_gdn[sq][ri]).collect();
    let fs: &dyn FrameState = acc;
    if seqs.len() > 1
        && std::env::var_os("LLM170_NO_NPAR").is_none()
        && acc
            .frame_gdn_ar_np(f.gq, f.gk, f.gv, f.gbg, f.go, &ar_states, hp.n_group, hp.dt_rank, hp.d_state)
            .is_ok()
    {
        // 1런치 경로 사용
    } else {
        fs_begin(acc, 1);
        for (row, &sq) in seqs.iter().enumerate() {
            fs.frame_gdn_ar(vv.gq[row], vv.gk[row], vv.gv[row], vv.gbg[row], f.st_gdn[sq][ri], vv.go[row], 1, hp.n_group, hp.dt_rank, hp.d_state)
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
    if seqs.len() > 1 { psum(acc, vv.go[1], v_len, "ar_row1"); }
    fs_begin(acc, t); // 공유 구간 복귀
    let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
    op(acc, FrameOp::NormGated { o: f.go, z: f.gz, w: snorm, out: f.ggated, eps, d: hp.d_state, n_h: hp.dt_rank })?;
    let wout = model.w4(&format!("blk.{il}.ssm_out.weight"))?;
    acc.frame_mm(f.ggated, &wout, f.ffn_out, t).map_err(Q4Error::Io)?;
    psum(acc, f.ffn_out, 64, "ffnout_head");
    Ok(())
}

/// QSA 프레임(np) — 5투영·wo는 t=rows 공유, rope/선택/KV/어텐션은 per-seq.
/// (호스트 선택 폴백은 np 경로에서 지원하지 않는다 — 디바이스 경로 실패 시 Err.)
#[allow(clippy::too_many_arguments)]
fn qsa_frame_np(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
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
    let qn: Vec<f32> = qn_raw.iter().copied().cycle().take(qn_raw.len() * n_head).collect();
    let kn: Vec<f32> = kn_raw.iter().copied().cycle().take(kn_raw.len() * n_kv).collect();
    let iqw = model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
    let ikw = model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
    let kq_scale = hp.kq_scale();
    let r = hp.compress[il] as usize;
    let vv = f.np_views.as_ref().unwrap();
    fs_begin(acc, 1); // per-seq 구간
    for (row, &sq) in seqs.iter().enumerate() {
        let st = &mut seq_sts[sq];
        let pos0 = st.pos as usize;
        acc.frame_qk_norm_rope(
            vv.qsa_q[row], vv.qsa_k[row], &qn, &kn, &f.qsa_cs, hp.eps, pos0,
            n_head, n_kv, hd, n_rot, 1,
        )
        .map_err(Q4Error::Io)?;
        let (sd, od, list_len) = acc
            .qsa_sel_dev(
                full_idx, sq, vv.qsa_iq[row], vv.qsa_ik[row], 1, pos0,
                hp.idx_heads, idx_dim, r, hp.idx_top_k, &iqw, &ikw, &f.qsa_cs_idx, hp.eps,
            )
            .map_err(Q4Error::Io)?;
        let (kc, vc) = acc
            .qsa_kv_dev(full_idx, sq, vv.qsa_k[row], vv.qsa_v[row], 1, pos0, n_kv, hd)
            .map_err(Q4Error::Io)?;
        acc.qsa_attention_dev_sel(
            vv.qsa_q[row], kc, vc, sd, od, list_len, kq_scale,
            n_head, n_kv, hd, 1, vv.qsa_attn[row],
        )
        .map_err(Q4Error::Io)?;
        st.qsa_host_stale = true;
    }
    fs_begin(acc, t); // 공유 구간 복귀
    acc.frame_mm_group(f.qsa_attn, &[wo], &[f.ffn_out], t)
        .map_err(Q4Error::Io)?;
    let _ = ctx;
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
    frame_forward_np_ex(acc, model, ctx, seqs, seq_sts, f, tokens, false).map(|(l, _)| l)
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
    frame_forward_np_ex(acc, model, ctx, seqs, seq_sts, f, tokens, true).map(|(_, t)| t)
}

#[allow(clippy::too_many_lines)]
fn frame_forward_np_ex(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seqs: &[usize],
    seq_sts: &mut [SeqState4],
    f: &mut Frame4,
    tokens: &[u32],
    greedy: bool,
) -> Result<(Vec<Vec<f32>>, Vec<u32>), Q4Error> {
    let hp: &Hparams4 = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let k_len = hp.n_group * hp.d_state;
    let v_len = hp.dt_rank * hp.d_state;
    let conv_ch = 2 * k_len + v_len;
    let eps = hp.eps;
    let t = seqs.len();
    if t > 8 {
        return Err(Q4Error::Io("frame_forward_np: t>8 미지원".into()));
    }
    let t_call = std::time::Instant::now();
    fs_begin(acc, t);
    ensure_np_views(acc, f, t, conv_ch, k_len, v_len, n, hc, hp)?;

    // 0) 임베딩 — 각 seq 토큰 → res_hc [t][hc·n]
    {
        let embd = model
            .w("token_embd.weight")
            .ok_or(Q4Error::MissingTensor("token_embd".into()))?;
        let mut row = vec![0.0f32; n];
        let mut r = vec![0.0f32; t * hc * n];
        for (ti, &tok) in tokens.iter().enumerate() {
            dequant_row(embd.ty, embd.data, tok as u64, n as u64, &mut row);
            for s in 0..hc {
                let b = ti * hc * n + s * n;
                r[b..b + n].copy_from_slice(&row);
            }
        }
        acc.frame_write(f.res_hc, &r).map_err(Q4Error::Io)?;
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

    let ck_on = std::env::var_os("LLM170_NP_CHECKSUM").is_some();
    let ck = |acc: &dyn Accelerator, h: u64, n2: usize, tag: &str| {
        if !ck_on { return; }
        let mut v = vec![0.0f32; n2];
        if acc.frame_read(h, &mut v).is_ok() {
            let s2: f64 = v.iter().map(|&x| x as f64).sum();
            eprintln!("[npck] {tag} sum={s2:.6} v0={:.6} v1={:.6}", v[0], v.get(1).copied().unwrap_or(0.0));
        }
    };
    let mut recr_idx = 0usize;
    let mut full_idx = 0usize;
    for il in 0..hp.n_layer {
        // 1) PLE — per-seq t=1 디바이스 경로(행 뷰)
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
            fs_begin(acc, 1); // per-seq 구간
            for (row, &sq) in seqs.iter().enumerate() {
                let mut emb = vec![0.0f32; emb_w];
                if ple_rows[row].len() == heads {
                    ctx.model.ple_gather(&ple_rows[row], &mut emb)?;
                }
                acc.frame_write(f.ple_emb, &emb).map_err(Q4Error::Io)?;
                acc.frame_mm_group(f.ple_emb, &[w_key, w_value], &[f.ple_key, f.ple_value], 1)
                    .map_err(Q4Error::Io)?;
                acc.ple_math_dev(
                    vv.res_hc[row], f.ple_key, f.ple_value, &nk, &nq, &nc, &cw,
                    f.ple_gated, f.ple_conv_out, f.ple_gate, sq, 1, hp.eps,
                    n, hc, hp.ple_conv_k, hp.ple_ngram,
                    (hp.ple_conv_k - 1) * hp.ple_ngram, &seq_sts[sq].ple_conv,
                )
                .map_err(Q4Error::Io)?;
            }
        }

        fs_begin(acc, t); // 공유 구간
        // 2) hc attn mix (t 공유)
        if il < 4 { ck(acc, f.res_hc, 64, &format!("L{il}.res_in")); }
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc, t)?;
        if il < 4 { ck(acc, f.mix, 64, &format!("L{il}.mix")); }
        sync_mark(acc, &format!("np{il}.hc_attn"), f.mix)?;

        // 3) GDN / QSA
        if hp.is_recr(il) {
            gdn_frame_np(acc, model, f, il, seqs, recr_idx, conv_ch, k_len, v_len, eps, t)?;
            if il < 4 { ck(acc, f.ffn_out, 64, &format!("L{il}.gdn")); }
            sync_mark(acc, &format!("np{il}.gdn"), f.ffn_out)?;
            recr_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
        } else {
            qsa_frame_np(acc, model, ctx, seq_sts, seqs, f, il, t, full_idx)?;
            if il < 4 { ck(acc, f.ffn_out, 64, &format!("L{il}.qsa")); }
            sync_mark(acc, &format!("np{il}.qsa"), f.ffn_out)?;
            full_idx += 1;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
        }

        // 4) hc ffn mix(t 공유) + MoE(행별 t=1 — 산술 불변) + combine(t 공유)
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc, t)?;
        // MoE: 기본 행별 t=1(모멘텀 유지 — t 배치 gather 판이 t=4 에서 10ms 느림,
        // 2026-09-16 실측). LLM170_NP_MOE_BATCH=1이면 t 배치(gather — 전문가
        // 가중합 순서 차이로 근접 평탄점 플립 가능, 문서화 tie 등급).
        // plans/74: np 기본 배치 MoE — direct-ids 커널이 rows<=64 에서도
        // 돌아가므로 t·k_sel=40행 1회 GEMM(행별 루프 대비 런치 1/4, 점유율 4배,
        // 산술 비트동일 — q4_moe_scatter 합산순서 = q4_moe_weighted_sum).
        // LLM170_NO_MOE_NPB=1 이면 행별로 복귀.
        if t > 1 && std::env::var_os("LLM170_NO_MOE_NPB").is_none() {
            moe_frame(acc, model, f, il, n, t)?;
        } else {
            moe_frame_np(acc, model, f, il, n, seqs)?;
        }
        sync_mark(acc, &format!("np{il}.moe"), f.mout)?;
        if il < 4 { ck(acc, f.mout, 64, &format!("L{il}.moe")); }
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
    }
    if ck_on { ck(acc, f.res_hc, 64, "head.res"); }

    // 5) head — 전 행 GEMM 1회 → [t][vocab] 판독
    {
        let w_norm = f.consts["output_hc_norm"];
        op(acc, FrameOp::RmsRows { x: f.res_hc, w: w_norm, out: f.hxn, eps, n, w_reps: hc })?;
        let w_down = model.w4("output_hc_down.weight")?;
        acc.frame_mm(f.hxn, &w_down, f.hlo, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluDiv { t: f.hlo, div: hc as f32, n: f.hlo_len * t })?;
        let w_up = model.w4("output_hc_up.weight")?;
        acc.frame_mm(f.hlo, &w_up, f.hgate, t).map_err(Q4Error::Io)?;
        op(acc, FrameOp::HcGateMean { xn: f.hxn, gate: f.hgate, out: f.hin, hc, n })?;
        let wout = model.w("output.weight").ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        acc.frame_mm(f.hin, &wout, f.logits_t, t).map_err(Q4Error::Io)?;
        let (logits, toks) = if greedy {
            // GPU argmax — vocab×t 플로트 전사·CPU 스캔 회피 (plans/74 N1).
            (Vec::new(), acc.frame_argmax_rows(f.logits_t, t, hp.vocab).map_err(Q4Error::Io)?)
        } else {
            let mut all = vec![0.0f32; hp.vocab * t];
            acc.frame_read(f.logits_t, &mut all).map_err(Q4Error::Io)?;
            if std::env::var_os("LLM170_NP_DBG").is_some() {
                for r in 0..t {
                    let row = &all[r * hp.vocab..(r + 1) * hp.vocab];
                    let (i1, v1) = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                    let (i2, v2) = row.iter().enumerate().filter(|(i, _)| *i != i1).max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                    eprintln!("# npdbg logits row{r}: top2 ({i1},{v1:.4}) ({i2},{v2:.4}) gap={:.4}", v1 - v2);
                }
            }
            ((0..t).map(|r| all[r * hp.vocab..(r + 1) * hp.vocab].to_vec()).collect(), Vec::new())
        };
        ftime_report(t);
        if ftime_on() {
            eprintln!("# np-frame-total t={t} greedy={greedy} {:.1}ms", t_call.elapsed().as_secs_f64() * 1e3);
        }
        Ok((logits, toks))
    }
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
fn gdn_frame_pre(
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
struct QsaBufs {
    mix: u64,
    q: u64,
    k: u64,
    v: u64,
    iq: u64,
    ik: u64,
    attn: u64,
    out: u64,
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
fn qsa_frame(
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
        }
    }
    sync_mark(acc, "gdn.ar", f.go)?;
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
    let wout = model.w4(&format!("blk.{il}.ssm_out.weight"))?;
    if !stage_skipped("gdn.out") {
        acc.frame_mm(f.ggated, &wout, f.ffn_out, t).map_err(Q4Error::Io)?;
    }
    sync_mark(acc, "gdn.out", f.ffn_out)?;
    Ok(())
}


/// MoE 프레임(np) — **행별 t=1 경로**. MoE는 행마다 전문가가 달라 무게 공유가
/// 없고, t>1 gather/scatter 경로는 가중합 순서가 t=1과 달라(~1e-7, 문서화)
/// 근접 평탄점을 플립한다. 배치 불변식(== 순차 decode1)을 위해 t=1 산술을
/// 그대로 행마다 실행한다(전문가 읽기 총량은 동일).
fn moe_frame_np(
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
        op(acc, FrameOp::MoeTop10 { route: f.mroute, ids: f.mids, wt: f.mwt, n_exp: hp.n_expert, k_sel })?;
        op(acc, FrameOp::BcastRows { src: mix_row, dst: f.mxsel, n, rows: k_sel })?;
        fs.frame_moe_gemm(f.mxsel, &w_gate, f.mids, f.mgu, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        fs.frame_moe_gemm(f.mxsel, &w_up, f.mids, f.mup, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(acc, FrameOp::SiluMul { g: f.mgu, u: f.mup, out: f.mglu, n: k_sel * n_ff })?;
        fs.frame_moe_gemm(f.mglu, &w_down, f.mids, f.my, hp.n_expert, k_sel)
            .map_err(Q4Error::Io)?;
        op(acc, FrameOp::MoeWeightedSum { ys: f.my, wt: f.mwt, out: mout_row, k: k_sel, n })?;
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
    fs_begin(acc, seqs.len()); // 공유 구간 복귀
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
            op(acc, FrameOp::AxpyScaled { y: f.mout, x: f.shout, s: f.msgate, n: n * t })?;
        }
    }
    sync_mark(acc, "moe.shared", f.mout)?;
    Ok(())
}

