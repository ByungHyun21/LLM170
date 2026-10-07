//! qwen4exp 상태·forward — HC 잔차, GDN(z-gate sigmoid), QSA(인덱서 top-k),
//! MoE(512·10+shared), PLE(n-gram 해시·게이트·dilated conv).
//!
//! 배선 근거: qwen4exp.cpp build_hc_mix/combine/build_qsa_top_k/build_attn_qsa/
//! build_ple + llama-graph.cpp build_moe_ffn (2026-08-30 판). 수치는 f32 참조.

use super::stages::{self, Ctx};
use super::{Model4, Q4Error};
use crate::matmul::Accelerator;
use crate::quant::dequant_row;
use llm170_diag::profile_span;

/// plans/115 P1-3: 체크포인트 전역 RAM 예산(바이트) — 슬롯 수로 분할해
/// 슬롯당 보관 수(1..=4)를 산정한다. 4슬롯×4개×112MB ≈ 1.8GB 상한.
pub(super) const CKPT_BUDGET: usize = 2 << 30;

mod mtp;
mod state;

use mtp::hc_combine;
pub(crate) use mtp::mtp_attn_cpu_row;
pub use state::{SeqCkpt, SeqState4};

pub struct Engine4 {
    pub model: Model4,
    pub seqs: Vec<SeqState4>,
    pub acc: Option<std::sync::Arc<dyn Accelerator>>,
    /// 프레임(활성화 상주) 상태 — LLM170_FRAME=1 첫 디코드에서 생성.
    pub frame: Option<super::frame::Frame4>,
    /// 프레임 폴백 확정 — 상주 불가 등 오류 시 value 경로로 영구 전환.
    frame_broken: bool,
    /// PLE 프리페치 (05-2) — 토큰 t 확정 직후 t+1분 16행×ple_head_dim을
    /// 사이드 스레드에서 mmap 읽기+디양자화. 다음 decode의 ple_block이 소비.
    pub ple_next: Option<std::sync::Arc<std::sync::Mutex<PlePrefetched>>>,
    /// 소비 대기 emb (prefetch 히트분) — decode1/prefill이 채우고 forward가 take.
    ple_consume: Option<Vec<Vec<f32>>>,
    /// 프리페치 사이드 스레드 핸들 — 다음 스텝 시작부 조인 (mmap 수명 보장).
    ple_worker: Option<std::thread::JoinHandle<()>>,
    /// MTP 드래프트 상태 (plans/109 P15②) — mtp_seqs[s]는 블록 n_layer의
    /// KV/idx 슬롯을 추가 소유. 모델이 load_mtp된 경우에만 생성.
    pub mtp_seqs: Vec<SeqState4>,
    /// 직전 forward의 h_nextn(프리-헤드 hidden) — MTP 드래프트 입력용.
    pub last_h: Vec<f32>,
    /// 직전 forward의 h_nextn 행 전체[t][n] — 기존 호환.
    pub last_h_rows: Vec<Vec<f32>>,
    /// 직전 forward의 **프리-믹서 멀티 스트림 잔차** [t][hc·n](P15④) — MTP
    /// 드래프트 h 입력(hnorm[10240] 플랫 정규화 대상).
    pub last_res_hc_rows: Vec<Vec<f32>>,
    /// 마지막 행 프리-믹서 잔차 [hc·n].
    pub last_res_hc: Vec<f32>,
    /// plans/110 W5: 슬롯별 스펙 h_prev(직전 라운드 최종 export) — 다중 슬롯에서
    /// 전역 last_res_hc가 타 슬롯 export로 덮이는 것을 막는다(④′ 페어링 결함).
    pub spec_h_prev: Vec<Vec<f32>>,
    /// plans/115 P1-3: 슬롯별 접두 체크포인트 — 프리필 청크 경계(≥512토큰
    /// 간격) 캡처. 부분 접두 재사용(l < cached.len) 시 되감는다.
    pub ckpt: Vec<std::collections::VecDeque<SeqCkpt>>,
    /// 슬롯당 체크포인트 보관 수 — 전역 예산(CKPT_BUDGET)을 상태 크기로 나눠
    /// 산정(4슬롯=4, 16슬롯=1). 클론은 GDN/conv/PLE만(≈112MB) — kv/idx는
    /// pos 인덱스 쓰기·접두 불변이라 제외.
    ckpt_keep: usize,
}

/// 사이드 스레드가 채운 프리페치 결과 — token이 다음 입력과 일치할 때만 사용.
pub struct PlePrefetched {
    pub token: u32,
    pub emb: Vec<f32>,
}

/// 프레임 버퍼의 토큰 상한 — 프리필 청크와 동일(디코드 t=1 포함).
/// 512 상한: t_max 버퍼는 청크에 비례하고(≈0.8 GB @512), 1024는 실측
/// hipMalloc OOM이었다. 값 경로 청크(1024)와 독립.
const FRAME_T_MAX: usize = 512;

/// 프레임 토큰 상한 — 기본 512(8GB CMP에서 1024는 hipMalloc OOM 이력).
/// `LLM170_FRAME_TMAX`로 올릴 수 있다(대형 VRAM 기기: 전문가당 행 수가 늘어
/// MoE 가중치 재사용이 좋아진다).
fn frame_t_max_cap(acc: Option<&dyn crate::matmul::Accelerator>) -> usize {
    // QA-5(plans/114): 핫패스 env는 flag 스냅샷 판독(원장 104 계약).
    if let Some(v) =
        llm170_diag::flag::val("LLM170_FRAME_TMAX").and_then(|v| v.parse::<usize>().ok())
    {
        return v.clamp(16, 4096);
    }
    // 적응형: 청크가 크면 전문가당 행 수가 늘어 MoE 가중치 재사용이 좋아진다.
    // 11,750토큰 실측(프리필 elapsed, 로드 포함): 512→1024 −5.7%, 1024→2048 −5.1%
    // (누적 155.2→139.0s). 프레임 버퍼는 청크에 비례(≈0.8GB@512)하므로 VRAM 계층으로
    // 고른다. 8GB CMP는 512 유지(1024는 hipMalloc OOM 이력).
    const GB: u64 = 1024 * 1024 * 1024;
    match acc.map(|a| a.total_mem_bytes()).unwrap_or(0) {
        m if m >= 48 * GB => 2048,
        m if m >= 16 * GB => 1024,
        _ => FRAME_T_MAX,
    }
}

fn frame_t_max(acc: Option<&dyn crate::matmul::Accelerator>) -> usize {
    let cap = frame_t_max_cap(acc);
    // 기본값 = 적응형 상한(env는 "요청"이고 상한이 최종 결정 — VRAM이 작으면 내려간다).
    q4_chunk_env(cap).min(cap)
}

// 청크 크기 소스 — env 스냅샷(A6 계약: 기동 1회) + 하네스 오버라이드.
// chunk-check 프로브는 프로세스 중간 값 변경이 필요해서 set_var 대신
// 이 오버라이드를 쓴다(스냅샷 이후 set_var는 계약 밖 — prefill_multi 사고,
// plans/129 A6).
static CHUNK_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 프리필 청크 크기 오버라이드(하네스용) — 0이면 env 기본으로 복귀.
pub fn set_q4_chunk(n: usize) {
    CHUNK_OVERRIDE.store(n, std::sync::atomic::Ordering::Relaxed);
}

fn q4_chunk_env(default: usize) -> usize {
    let o = CHUNK_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if o > 0 {
        return o.clamp(16, 4096);
    }
    llm170_diag::flag::val("LLM170_Q4_CHUNK")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(16, 4096)
}

/// 프레임 환경 게이트(캐시) — LLM170_FRAME!=0 && {PREFILL,DECODE}!=0.
fn frame_env_on(decode: bool) -> bool {
    static PRE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static DEC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let lk = if decode { &DEC } else { &PRE };
    *lk.get_or_init(|| {
        llm170_diag::flag::on_nonzero("LLM170_FRAME")
            && if decode {
                llm170_diag::flag::ne0("LLM170_FRAME_DECODE")
            } else {
                llm170_diag::flag::ne0("LLM170_FRAME_PREFILL")
            }
    })
}

impl Engine4 {
    /// Frame4 지연 생성 스탠자 — 7중 복제 통합(plans/109 P7). 실패 시
    /// frame_broken + fb_incr(FrameCreate) + 경고 후 false(호출부 value 폴백).
    fn frame_ensure(&mut self) -> bool {
        if let Some(f) = self.frame.as_mut() {
            f.mtp_h_export = self.model.has_mtp();
            return true;
        }
        let Some(acc) = self.acc.as_deref() else {
            return false;
        };
        match super::frame::Frame4::new(acc, &self.model, &self.seqs, frame_t_max(Some(acc))) {
            Ok(mut f) => {
                f.mtp_h_export = self.model.has_mtp();
                self.frame = Some(f);
                true
            }
            Err(e) => {
                self.frame_broken = true;
                crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::FrameCreate);
                eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
                false
            }
        }
    }
    pub fn new(model: Model4, n_seqs: usize, ctx: usize) -> Self {
        let seqs = (0..n_seqs)
            .map(|_| SeqState4::new(&model.hp, ctx))
            .collect();
        let mtp_seqs = if model.has_mtp() {
            (0..n_seqs)
                .map(|_| SeqState4::new_mtp(&model.hp, ctx))
                .collect()
        } else {
            Vec::new()
        };
        let ckpt_keep = {
            // 상태 실측 크기로 예산 배분 — n_recr×(state+conv) 바이트.
            let hp = &model.hp;
            let state = hp.dt_rank * hp.d_state * hp.d_state;
            let conv_len =
                (hp.conv_k - 1) * (hp.n_group * hp.d_state * 2 + hp.dt_rank * hp.d_state);
            let est = ((state + conv_len) * 4).max(1);
            let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recr(il)).count();
            let per_state = (est * n_recr).max(1);
            ((CKPT_BUDGET / per_state) / n_seqs.max(1)).clamp(1, 4)
        };
        Engine4 {
            model,
            seqs,
            acc: None,
            frame: None,
            frame_broken: false,
            ple_next: None,
            ple_consume: None,
            ple_worker: None,
            mtp_seqs,
            last_h: Vec::new(),
            last_h_rows: Vec::new(),
            last_res_hc_rows: Vec::new(),
            last_res_hc: Vec::new(),
            spec_h_prev: vec![Vec::new(); n_seqs],
            ckpt: (0..n_seqs)
                .map(|_| std::collections::VecDeque::new())
                .collect(),
            ckpt_keep,
        }
    }

    pub fn hrows_reset(&mut self, seq: usize) {
        self.last_res_hc_rows.clear();
        self.last_res_hc.clear();
        if let Some(f) = &mut self.frame {
            f.last_res_hc_rows.clear();
        }
        let _ = seq; // 현재 엔진 전역 행 — 슬롯 인자는 계약 문서화용
    }

    pub fn with_acc(mut self, acc: std::sync::Arc<dyn Accelerator>) -> Self {
        // 컨텍스트 길이를 가속기에 주입한다 — KV 등 상한이 정해진 풀을 **선할당**하게
        // 하기 위함(호스트 KV는 SeqState4가 ctx로 이미 잡혀 있다, model/mod.rs:331 참조).
        let (n_kv, hd) = (self.model.hp.n_kv.max(1), self.model.hp.head_dim.max(1));
        if let Some(k) = self.seqs.first().and_then(|s| s.kv_k.first()) {
            acc.set_ctx_len(k.len() / (n_kv * hd));
        }
        self.acc = Some(acc);
        self
    }

    /// 프레임(디바이스 상주) 경로 활성 게이트 — 6벌 복제 통합(plans/90 A1 D10).
    /// decode=false → LLM170_FRAME_PREFILL, true → LLM170_FRAME_DECODE
    /// (둘 다 기본 on). 조건 순서·의미는 기존 인라인 판과 동일.
    /// 환경 판독은 기동 후 불변 전제로 1회 캐시(90 B5 — 스텝당 env::var 제거).
    fn frame_on(&self, decode: bool) -> bool {
        self.acc.is_some()
            && !self.frame_broken
            && self.acc.as_ref().is_some_and(|a| a.frame_capable())
            && frame_env_on(decode)
    }

    /// 프레임 GPU 상태 → CPU 사본 풀백(값 경로 진입 전 정합화, plans/90 A1 D12).
    /// gdn은 전치 레이아웃(AR 커널 규약) 역변환 포함.
    /// 프레임 없음·가속기 없음·dirty(CPU가 권위)면 no-op.
    fn frame_pullback_cpu(&mut self, seq: usize) -> Result<(), Q4Error> {
        let Some(acc) = self.acc.as_deref() else {
            return Ok(());
        };
        let (gdn, conv) = {
            let Some(f) = self.frame.as_ref() else {
                return Ok(());
            };
            if f.dirty[seq] {
                return Ok(());
            }
            (f.st_gdn[seq].clone(), f.st_conv[seq].clone())
        };
        let st = &mut self.seqs[seq];
        let d_state = self.model.hp.d_state;
        for (ri, h) in gdn.iter().enumerate() {
            let mut t = vec![0.0f32; st.gdn_s[ri].len()];
            acc.frame_read(*h, &mut t).map_err(Q4Error::Io)?;
            st.gdn_s[ri] = super::frame::Frame4::transpose_pairs(&t, d_state);
        }
        for (ri, h) in conv.iter().enumerate() {
            acc.frame_read(*h, &mut st.conv[ri]).map_err(Q4Error::Io)?;
        }
        Ok(())
    }

    fn forward_timed(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        profile_span!("q4::forward");
        macro_rules! stage {
            ($field:ident, $body:expr) => {
                $body
            };
        }

        let hp = self.model.hp.clone();
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t_len = tokens.len();
        // 스테이지 컨텍스트 — model 불변 차입 + seq 상태 가변 차입 (필드 분리)
        let ctx = Ctx {
            model: &self.model,
            acc: self.acc.as_deref(),
        };
        let seq_st = &mut self.seqs[seq];
        // 임베딩 디양자화를 먼저 끝내 borrow 분리
        let embd_rows: Vec<Vec<f32>> = {
            let embd = self
                .model
                .w("token_embd.weight")
                .ok_or(Q4Error::MissingTensor("token_embd".into()))?;
            tokens
                .iter()
                .map(|&tok| {
                    let mut row = vec![0.0f32; n_embd];
                    dequant_row(embd.ty, embd.data, tok as u64, n_embd as u64, &mut row);
                    row
                })
                .collect()
        };

        // PLE n-gram 행 (호스트 u64 해시) — 히스토리 스냅샷 후 갱신
        let ple_rows = if hp.is_ple(1) {
            stages::ple_hash(&ctx, seq_st, tokens)
        } else {
            Vec::new()
        };

        // 초기 상태 = 임베딩 ×4 스트림
        let mut res_hc: Vec<Vec<f32>> = Vec::with_capacity(t_len);
        for row in &embd_rows {
            let mut r = vec![0.0f32; hc_dim];
            for s in 0..hc {
                r[s * n_embd..(s + 1) * n_embd].copy_from_slice(row);
            }
            res_hc.push(r);
        }

        let mut full_idx = 0usize;
        let mut recr_idx = 0usize;
        let trace = llm170_diag::dump::opts().key("q4_trace");
        // NaN 조기 국소화 — 발산 층·스테이지를 즉시 보고 (LLM170_Q4_TRACE).
        let nan_guard = |v: &[Vec<f32>], tag: &str, il: usize| {
            for (ti, row) in v.iter().enumerate() {
                if row.iter().any(|x| !x.is_finite()) {
                    eprintln!("# NaN발견 layer={il} {tag} token={ti} t={}", row.len());
                    // 107 W11: exit(101) 제거 — 전 슬롯 사망 대신 보고.
                    return;
                }
            }
        };
        for il in 0..hp.n_layer {
            if trace {
                eprintln!("q4 layer {il} t={t_len}");
            }
            if hp.is_ple(il) && !super::frame::stage_skipped("ple") {
                // 05-2 프리페치 소비 — decode1이 stash한 emb (t=1 전용.
                // 프리필 전량 선적재(05-3)는 chunk 경계 행 불일치로 보류 — 주석 참조).
                let pre = if t_len == 1 {
                    self.ple_consume.take()
                } else {
                    None
                };
                stage!(
                    ple,
                    stages::ple_block(&ctx, seq_st, il, &mut res_hc, &ple_rows, pre)?
                );
            }
            let (mix, inject) = stage!(hc, stages::hc_mix(&ctx, il, "attn", &res_hc)?);
            let attn_out = if hp.is_recr(il) {
                if super::frame::stage_skipped("gdn") {
                    // 진단용: GDN 생략(출력 무효) — 청크 의존 축 분리(plans/80).
                    recr_idx += 1;
                    vec![vec![0.0f32; n_embd]; t_len]
                } else {
                    let o = stage!(
                        gdn,
                        stages::gdn_layer(&ctx, seq_st, il, &mix, t_len, recr_idx)?
                    );
                    recr_idx += 1;
                    o
                }
            } else if super::frame::stage_skipped("qsa") {
                full_idx += 1;
                vec![vec![0.0f32; n_embd]; t_len]
            } else {
                let o = stage!(
                    qsa,
                    stages::qsa_layer(&ctx, seq_st, il, &mix, t_len, full_idx)?
                );
                full_idx += 1;
                o
            };
            if trace {
                nan_guard(
                    &attn_out,
                    if hp.is_recr(il) { "gdn_out" } else { "qsa_out" },
                    il,
                );
            }
            hc_combine(&mut res_hc, &attn_out, &inject, hc);

            let (mix2, inject2) = stage!(hc, stages::hc_mix(&ctx, il, "ffn", &res_hc)?);
            if trace {
                nan_guard(&mix2, "hc_ffn_mix", il);
                // 값 폭발 추적 — max|x| (Inf 직전 값도 is_finite 통과)
                let _mx = mix2
                    .iter()
                    .flat_map(|r| r.iter())
                    .fold(0.0f32, |a, &v| if v.abs() > a { v.abs() } else { a });
            }
            let ffn_out = stage!(moe, stages::moe_ffn(&ctx, il, &mix2)?);
            if llm170_diag::fp::enabled() {
                llm170_diag::fp::fp_record(
                    &format!("L{il}.ffn_out"),
                    &ffn_out
                        .iter()
                        .flat_map(|r| r.iter())
                        .copied()
                        .collect::<Vec<_>>(),
                );
            }
            if trace {
                nan_guard(&ffn_out, "moe_out", il);
            }
            hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        }

        // plans/109 P15④(vLLM qwen4_exp mtp.py 준거): MTP 드래프트 h 입력은
        // **최종 믹서 이전**의 멀티 스트림 잔차 [hc·n] 평탄화(pre-hc_head
        // residual). 프리필 쌍·체인 모두 이 행을 쓴다.
        self.last_res_hc_rows = res_hc.clone();
        self.last_res_hc = res_hc.last().cloned().unwrap_or_default();
        // output HC mix → logits (inject 없음)
        let head_in = stage!(head, stages::hc_mix_head(&ctx, &res_hc)?);
        let last = head_in.last().ok_or(Q4Error::BadMeta("빈 배치"))?.clone();
        // 기존 호환 필드(다른 소비자 없음).
        self.last_h_rows = head_in.clone();
        self.last_h = last.clone();
        let wout = self
            .model
            .w("output.weight")
            .ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        let mut logits = vec![0.0f32; wout.n_out as usize];
        stage!(head, ctx.mm(&last, &wout, &mut logits)?);
        Ok(logits)
    }

    /// 시퀀스 상태 전체 초기화 (무상태 HTTP 서버용).
    pub fn reset_states(&mut self) {
        self.ckpt_clear(None);
        // plans/141: MTP 드래프트 h 행 누산 초기화. prefill은 청크마다
        // last_res_hc_rows에 **적립**하므로(멀티청크 드래프트 프리필 계약)
        // 초기화 없이 재사용하면 이전 잡의 행이 남아 새 프롬프트보다 길어진다.
        // 실측 그 결과 mtp_draft_prefill의 `행 수 == 토큰 수` 계약이 영구히
        // 깨져 드래프트 프리필이 생략되고, 드래프트 KV가 빈 문맥으로 시작해
        // 스펙 수용률이 0이 된다(벤치 tg 3 t/s). 서버는 슬롯 시작에
        // hrows_reset으로 같은 정리를 하므로 bench 경로가 남은 누락이었다.
        self.last_res_hc_rows.clear();
        self.last_res_hc.clear();
        if let Some(f) = &mut self.frame {
            f.last_res_hc_rows.clear();
        }
        // CPU 상태를 영점화했다 — 프레임 GPU 상태는 stale이므로 pull 금지
        // (dirty=true → 다음 prefill 후 decode에서 재동기).
        if let Some(f) = &mut self.frame {
            for d in f.dirty.iter_mut() {
                *d = true;
            }
        }
        let ctx = self
            .seqs
            .first()
            .map(|s| {
                s.kv_k
                    .first()
                    .map(|k| k.len() / (self.model.hp.n_kv * self.model.hp.head_dim))
                    .unwrap_or(4096)
            })
            .unwrap_or(4096);
        for i in 0..self.seqs.len() {
            self.seqs[i] = SeqState4::new(&self.model.hp, ctx);
        }
    }

    /// 슬롯 단위 상태 초기화 (연속 배칭 서버 — 04). dirty[seq]만 표시.
    pub fn reset_seq(&mut self, seq: usize) {
        self.ckpt_clear(Some(seq));
        if let Some(acc) = self.acc.as_deref() {
            acc.acc_reset_seq(seq);
        }
        if let Some(f) = &mut self.frame {
            f.dirty[seq] = true;
        }
        let ctx = self.seqs[seq]
            .kv_k
            .first()
            .map(|k| k.len() / (self.model.hp.n_kv * self.model.hp.head_dim))
            .unwrap_or(4096);
        self.spec_h_prev[seq] = Vec::new();
        self.seqs[seq] = SeqState4::new(&self.model.hp, ctx);
    }

    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        // LLM170_Q4_CHUNK: 프리필 청크 토큰 수 (기본 1024; 프레임 경로는 t_max 상한).
        let cap0 = frame_t_max_cap(self.acc.as_deref());
        let chunk: usize = q4_chunk_env(cap0);
        // 프레임 상태가 권위적이면(직전 디코드) CPU 사본을 GPU에서 갱신 —
        // 값 경로 prefill이 정합 상태에서 시작하기 위함. 프레임 프리필(기본)은
        // 디바이스 상태를 그대로 쓰므로 이 풀백이 데드 워크다 — 슬롯당 수십 회의
        // 소형 D2H(2026-09-17, np 서버 TTFT/프리필 간극 RCA). 아래 값 경로
        // 직전으로 이동했다.
        let frame_prefill_on = self.frame_on(false);
        let need_cpu_pullback = !frame_prefill_on;
        if need_cpu_pullback {
            self.frame_pullback_cpu(seq)?;
        }
        // 프레임(디바이스 상주) 프리필 — 기본 on (끄기: LLM170_FRAME_PREFILL=0).
        // 토큰 계약 검증: 230@512·300@128(3청크)·512 모두 값 경로와 일치.
        // pp512 36.8 t/s = 값 경로(11.2)의 3.3배.
        let frame_on = frame_prefill_on;
        // 프레임 버퍼(t_max)보다 큰 청크는 범위를 넘는다 — 프레임 경로는 청크를 묶는다.
        let chunk = if frame_on {
            chunk.min(frame_t_max_cap(self.acc.as_deref()))
        } else {
            chunk
        };
        if frame_on {
            self.frame_ensure();
        }
        if let Some(f) = self.frame.as_mut().filter(|_| frame_on) {
            let acc = self.acc.as_deref().unwrap();
            let mut last = None;
            let n_chunks = tokens.chunks(chunk).len();
            for (ci, ch) in tokens.chunks(chunk).enumerate() {
                let _sync_t0 = std::time::Instant::now();
                if f.dirty[seq] {
                    f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
                }
                if llm170_diag::dump::opts().key("frame_time") {
                    eprintln!("# pf-sync {:.1}ms", _sync_t0.elapsed().as_secs_f64() * 1e3);
                }
                let ctx = Ctx {
                    model: &self.model,
                    acc: Some(acc),
                };
                // 107 W1.5-4: 중간 청크는 head+로짓 전사 스킵(NoReadback) —
                // 최종 청크만 Full. 버려지던 152k GEMV·608KB d2h 제거.
                let is_last = ci + 1 == n_chunks;
                // P15④c: 청크별 pre-mixer 행 누적(MTP h export — 비스펙 0비용).
                if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                    self.last_res_hc_rows
                        .extend(f.last_res_hc_rows.iter().cloned());
                }
                let logits = if is_last {
                    super::frame::frame_forward(
                        acc,
                        &self.model,
                        &ctx,
                        seq,
                        &mut self.seqs[seq],
                        f,
                        ch,
                    )?
                } else {
                    super::frame::frame_forward_ex(
                        acc,
                        &self.model,
                        &ctx,
                        seq,
                        &mut self.seqs[seq],
                        f,
                        ch,
                        super::frame::FwdMode::NoReadback,
                    )
                    .map(|(l, _)| l)?
                };
                self.seqs[seq].pos += ch.len() as u32;
                f.dirty[seq] = false;
                if is_last {
                    // 최종 청크 export 누적 + 마지막 행 동기.
                    if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                        self.last_res_hc_rows
                            .extend(f.last_res_hc_rows.iter().cloned());
                        // plans/115 D2: 호출 간 이중 적립 방지 — f.rows는 이미
                        // self에 적립됐다. 남기면 다음 호출의 pre-extend가 같은
                        // 행을 두 번 쌓아 mtp_draft_prefill len 검사를 영구 깬다.
                        f.last_res_hc_rows.clear();
                    }
                    last = Some(logits);
                }
            }
            if !self.last_res_hc_rows.is_empty() {
                self.last_res_hc = self.last_res_hc_rows.last().cloned().unwrap_or_default();
            }
            return Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]));
        }
        // 값 경로 전용 풀백(프레임 경로 미사용 시에만).
        if !need_cpu_pullback {
            self.frame_pullback_cpu(seq)?;
        }
        let mut last = None;
        for ch in tokens.chunks(chunk) {
            let logits = self.forward_timed(seq, ch)?;
            self.seqs[seq].pos += ch.len() as u32;
            if let Some(f) = &mut self.frame {
                f.dirty[seq] = true; // 값 경로가 상태를 갱신 — 프레임 재동기 필요
            }
            last = Some(logits);
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

    /// greedy 프리필 — 마지막 토큰 로짓 전사(어휘 152k×4B pageable D2H,
    /// 슬로패스 수십 ms) 대신 GPU argmax 로 토큰만 회수(plans/74, np 서버).
    /// 프레임 경로 판만 갈리며 값 폴백은 종전 prefill+greedy 와 동일.
    pub fn prefill_greedy(&mut self, seq: usize, tokens: &[u32]) -> Result<u32, Q4Error> {
        let frame_on = self.frame_on(false);
        if !frame_on {
            let l = self.prefill(seq, tokens)?;
            return Ok(crate::qwen35::greedy(&l));
        }
        let cap0 = frame_t_max_cap(self.acc.as_deref());
        let chunk: usize = q4_chunk_env(cap0).min(frame_t_max_cap(self.acc.as_deref()));
        // 프레임 경로(이 함수의 주경로)는 CPU 상태 불필요 — 풀백 생략(데드 워크).
        if !self.frame_ensure() {
            let l = self.prefill(seq, tokens)?;
            return Ok(crate::qwen35::greedy(&l));
        }
        let acc = self.acc.as_deref().unwrap();
        let f = self.frame.as_mut().unwrap();
        let mut last = 0u32;
        for ch in tokens.chunks(chunk) {
            if f.dirty[seq] {
                f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
            }
            let ctx = Ctx {
                model: &self.model,
                acc: Some(acc),
            };
            last = super::frame::frame_forward_greedy(
                acc,
                &self.model,
                &ctx,
                seq,
                &mut self.seqs[seq],
                f,
                ch,
            )?;
            self.seqs[seq].pos += ch.len() as u32;
            f.dirty[seq] = false;
            // P15④c: 청크별 pre-mixer 행을 **누적** — 멀티청크 프리필에서
            // 전체 프롬프트의 h 행이 드래프트 프리필에 필요하다(④c-2).
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                if self.last_res_hc_rows.len() == tokens.len() - ch.len() {
                    self.last_res_hc_rows
                        .extend(f.last_res_hc_rows.iter().cloned());
                } else {
                    // 청크 경계 재시작 등 — 전체 재구성(마지막 청크만 오면 불완전).
                    self.last_res_hc_rows
                        .extend(f.last_res_hc_rows.iter().cloned());
                }
            }
        }
        // 마지막 행 동기.
        if !self.last_res_hc_rows.is_empty() {
            self.last_res_hc = self.last_res_hc_rows.last().cloned().unwrap_or_default();
        }
        Ok(last)
    }

    /// 다중 시퀀스 청크 프리필 — 대기 슬롯 N개의 같은 길이 청크를 한 forward 로
    /// 처리해 무게 패스를 공유한다(plans/74 np4; 슬롯별 프리필이 4회 무게를 읽던 것).
    /// 실패 시 호출부가 슬롯별 `prefill_greedy` 로 폴백한다.
    pub fn prefill_multi(
        &mut self,
        seqs: &[usize],
        tokens: &[u32],
        per_seq: usize,
    ) -> Result<Vec<u32>, Q4Error> {
        let acc = self
            .acc
            .clone()
            .ok_or(Q4Error::Io("prefill_multi: 가속기 없음".into()))?;
        if tokens.len() != seqs.len() * per_seq || seqs.len() < 2 {
            return Err(Q4Error::Io("prefill_multi: 계약 위반".into()));
        }
        if !self.frame_ensure() {
            // 프레임 생성 실패 — 호출부 트랜잭션 계약(plans/86 §3)대로 슬롯별
            // prefill_greedy 폴백 유도.
            return Err(Q4Error::Io("prefill_multi: frame 생성 실패".into()));
        }
        let f = self.frame.as_mut().unwrap();
        // 상태 동기화 — 슬롯별 prefill_greedy 와 동일 규칙.
        for &sq in seqs {
            if f.dirty[sq] {
                f.sync_states(acc.as_ref(), sq, &self.seqs[sq], self.model.hp.d_state)?;
            }
        }
        // plans/86 §3 — 트랜잭션: 호출부가 슬롯별 prefill_greedy 로 폴백하므로
        // 시도 전 상태를 스냅샷해 Err 시 복원한다.
        let ple0: Vec<_> = seqs
            .iter()
            .map(|&s| super::frame::ple_snap(&self.seqs[s]))
            .collect();
        let ctx = Ctx {
            model: &self.model,
            acc: Some(acc.as_ref()),
        };
        let toks = match super::frame::frame_forward_prefill_multi(
            acc.as_ref(),
            &self.model,
            &ctx,
            seqs,
            &mut self.seqs,
            f,
            tokens,
            per_seq,
        ) {
            Ok(t) => t,
            Err(e) => {
                for (snap, &sq) in ple0.into_iter().zip(seqs.iter()) {
                    super::frame::ple_restore(&mut self.seqs[sq], snap);
                }
                // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                if let Some(a) = self.acc.as_deref() {
                    a.frame_sync();
                }
                return Err(e);
            }
        };
        for &sq in seqs {
            f.dirty[sq] = false;
        }
        Ok(toks)
    }

    /// 디코드 1토큰 — LLM170_FRAME=1이면 프레임 경로 (활성화 GPU 상주).
    /// 시퀀스별 상태 핸들 세트로 np 디코드 지원 + PLE 프리페치 조인·소비.
    /// plans/73(np): 다중 시퀀스 배치 디코드 — 무게 스트리밍 공유(t=seqs.len()).
    /// 실패 시 프레임을 버리고 순차 decode1로 폴백해 서비스가 끊기지 않게 한다.
    /// plans/115 A-2: 샘플링용 top-k 디코드 — 전체 로짓 대신 후보만 반환.
    /// (val, idx) 쌍의 리스트. CPU 샘플러가 병합·필터링.
    pub fn decode_batch_topk(
        &mut self,
        seqs: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<Vec<(f32, u32)>>, Q4Error> {
        // 기존 decode_batch와 동일한 경로 (전체 로짓 계산)
        // 마지막에 로짓 대신 top-k 후보만 추출
        let _rows = self.decode_batch(seqs, tokens)?;
        // frame의 logits_t에서 top-k 추출
        if let Some(acc) = self.acc.as_deref() {
            let vocab = self.model.hp.vocab;
            let frame_logits = match &self.frame {
                Some(f) => f.logits_t,
                None => return Err(Q4Error::Io("frame 없음".into())),
            };
            match acc.frame_topk_cands(frame_logits, seqs.len(), vocab) {
                Ok(cands) => {
                    // 균등 분할
                    let per = cands.len() / seqs.len().max(1);
                    Ok((0..seqs.len())
                        .map(|r| cands[r * per..(r + 1) * per].to_vec())
                        .collect())
                }
                Err(_) => {
                    // 미지원 — 전체 로짓을 (val, idx) 쌍으로 반환
                    Ok(_rows
                        .into_iter()
                        .map(|row| {
                            row.iter()
                                .enumerate()
                                .map(|(i, &v)| (v, i as u32))
                                .collect::<Vec<_>>()
                        })
                        .collect())
                }
            }
        } else {
            Err(Q4Error::Io("가속기 없음".into()))
        }
    }

    pub fn decode_batch(
        &mut self,
        seqs: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        if seqs.len() < 2 || seqs.len() != tokens.len() {
            let mut out = Vec::with_capacity(seqs.len());
            for (&s, &tk) in seqs.iter().zip(tokens.iter()) {
                out.push(self.decode1(s, tk)?);
            }
            return Ok(out);
        }
        let frame_on = self.frame_on(true);
        if !frame_on {
            let mut out = Vec::with_capacity(seqs.len());
            for (&s, &tk) in seqs.iter().zip(tokens.iter()) {
                out.push(self.decode1(s, tk)?);
            }
            return Ok(out);
        }
        // PLE 프리페치 worker 조인(배치 경로는 프리페치 시작 안 함)
        if let Some(h) = self.ple_worker.take() {
            let _ = h.join();
        }
        self.ple_next = None;
        self.ple_consume = None;
        self.frame_ensure();
        // plans/86 §3 — 트랜잭션 스냅샷(전 시퀀스).
        let ple0: Vec<_> = seqs
            .iter()
            .map(|&s| super::frame::ple_snap(&self.seqs[s]))
            .collect();
        let r = (|| -> Result<Vec<Vec<f32>>, Q4Error> {
            let f = self
                .frame
                .as_mut()
                .ok_or_else(|| Q4Error::Io("frame 없음".into()))?;
            let acc = self.acc.as_deref().unwrap();
            for &s in seqs {
                if f.dirty[s] {
                    f.sync_states(acc, s, &self.seqs[s], self.model.hp.d_state)?;
                }
            }
            let ctx = Ctx {
                model: &self.model,
                acc: Some(acc),
            };
            super::frame::frame_forward_np(acc, &self.model, &ctx, seqs, &mut self.seqs, f, tokens)
        })();
        match r {
            Ok(ls) => {
                for &s in seqs {
                    self.seqs[s].pos += 1;
                }
                Ok(ls)
            }
            Err(e) => {
                self.frame = None;
                for (snap, &s) in ple0.into_iter().zip(seqs.iter()) {
                    super::frame::ple_restore(&mut self.seqs[s], snap);
                    // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                    if let Some(a) = self.acc.as_deref() {
                        a.frame_sync();
                    }
                }
                static ONCE: std::sync::Once = std::sync::Once::new();
                crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::FrameCreateNp);
                ONCE.call_once(|| eprintln!("# frame-np: 배치 디코드 실패 — 순차 폴백 ({e})"));
                let mut out = Vec::with_capacity(seqs.len());
                for (&s, &tk) in seqs.iter().zip(tokens.iter()) {
                    out.push(self.decode1(s, tk)?);
                }
                Ok(out)
            }
        }
    }

    /// np 배치 디코드 greedy — 토큰만 회수 (logits 전사·CPU greedy 회피, plans/74 N1).
    /// 구조·폴백 규칙은 decode_batch와 동일.
    pub fn decode_batch_greedy(
        &mut self,
        seqs: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, Q4Error> {
        let npg_t0 = std::time::Instant::now();
        if seqs.len() < 2 || seqs.len() != tokens.len() {
            let mut out = Vec::with_capacity(seqs.len());
            for (&s, &tk) in seqs.iter().zip(tokens.iter()) {
                let lg = self.decode1(s, tk)?;
                out.push(crate::qwen35::greedy(&lg));
            }
            return Ok(out);
        }
        // QA-5: 매 배치 호출 직독 → frame_env_on(OnceLock 캐시) 재사용.
        let frame_on = self.acc.is_some() && !self.frame_broken && frame_env_on(true);
        if !frame_on {
            let lg = self.decode_batch(seqs, tokens)?;
            return Ok(lg.iter().map(|l| crate::qwen35::greedy(l)).collect());
        }
        if let Some(h) = self.ple_worker.take() {
            let _ = h.join();
        }
        self.ple_next = None;
        self.ple_consume = None;
        self.frame_ensure();
        // plans/86 §3 — 트랜잭션 스냅샷(전 시퀀스).
        let ple0: Vec<_> = seqs
            .iter()
            .map(|&s| super::frame::ple_snap(&self.seqs[s]))
            .collect();
        let r = (|| -> Result<Vec<u32>, Q4Error> {
            let f = self
                .frame
                .as_mut()
                .ok_or_else(|| Q4Error::Io("frame 없음".into()))?;
            let acc = self.acc.as_deref().unwrap();
            for &s in seqs {
                if f.dirty[s] {
                    f.sync_states(acc, s, &self.seqs[s], self.model.hp.d_state)?;
                }
            }
            let ctx = Ctx {
                model: &self.model,
                acc: Some(acc),
            };
            super::frame::frame_forward_np_greedy(
                acc,
                &self.model,
                &ctx,
                seqs,
                &mut self.seqs,
                f,
                tokens,
            )
        })();
        if llm170_diag::dump::opts().key("np_time") {
            eprintln!(
                "[npstep4] t={} {:.1}ms",
                seqs.len(),
                npg_t0.elapsed().as_secs_f64() * 1e3
            );
        }
        match r {
            Ok(toks) => {
                for &s in seqs {
                    self.seqs[s].pos += 1;
                }
                Ok(toks)
            }
            Err(e) => {
                self.frame = None;
                for (snap, &s) in ple0.into_iter().zip(seqs.iter()) {
                    super::frame::ple_restore(&mut self.seqs[s], snap);
                    // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                    if let Some(a) = self.acc.as_deref() {
                        a.frame_sync();
                    }
                }
                static ONCE: std::sync::Once = std::sync::Once::new();
                crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::FrameCreateNp);
                ONCE.call_once(|| eprintln!("# frame-np-greedy: 배치 실패 — 순차 폴백 ({e})"));
                let mut out = Vec::with_capacity(seqs.len());
                for (&s, &tk) in seqs.iter().zip(tokens.iter()) {
                    let lg = self.decode1(s, tk)?;
                    out.push(crate::qwen35::greedy(&lg));
                }
                Ok(out)
            }
        }
    }

    /// greedy 디코드 — 로짓 전사 없이 GPU argmax 로 토큰만(plans/74).
    /// 구조는 decode1 과 동일, head 판만 갈린다.
    pub fn decode1_greedy(&mut self, seq: usize, token: u32) -> Result<u32, Q4Error> {
        if !self.frame_on(true) {
            let l = self.decode1(seq, token)?;
            return Ok(crate::qwen35::greedy(&l));
        }
        if let Some(h) = self.ple_worker.take() {
            let _ = h.join();
        }
        if let Some(slot) = self.ple_next.take()
            && let Ok(mut g) = slot.lock()
            && g.token == token
            && !g.emb.is_empty()
        {
            self.ple_consume = Some(vec![std::mem::take(&mut g.emb)]);
        }
        if !self.frame_ensure() {
            let l = self.decode1(seq, token)?;
            return Ok(crate::qwen35::greedy(&l));
        }
        // plans/86 §3 — 트랜잭션 스냅샷(폴백 시 복원).
        let ple0 = super::frame::ple_snap(&self.seqs[seq]);
        let r = (|| -> Result<u32, Q4Error> {
            let f = self
                .frame
                .as_mut()
                .ok_or_else(|| Q4Error::Io("frame 없음".into()))?;
            let acc = self.acc.as_deref().unwrap();
            if f.dirty[seq] {
                f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
            }
            let ctx = Ctx {
                model: &self.model,
                acc: Some(acc),
            };
            super::frame::decode_frame_greedy(
                acc,
                &self.model,
                &ctx,
                seq,
                &mut self.seqs[seq],
                f,
                token,
            )
        })();
        match r {
            Ok(tok) => {
                self.seqs[seq].pos += 1;
                // P15④c: frame pre-mixer 행 pull(greedy 판).
                if let Some(f) = self.frame.as_ref()
                    && f.mtp_h_export
                    && !f.last_res_hc_rows.is_empty()
                {
                    self.last_res_hc_rows = f.last_res_hc_rows.clone();
                    self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
                }
                Ok(tok)
            }
            Err(e) => {
                self.frame = None;
                self.frame_broken = true;
                super::frame::ple_restore(&mut self.seqs[seq], ple0);
                // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                if let Some(a) = self.acc.as_deref() {
                    a.frame_sync();
                }
                static ONCE: std::sync::Once = std::sync::Once::new();
                crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::FrameCreate);
                ONCE.call_once(|| eprintln!("# frame-greedy: 디코드 실패 — 폴백 ({e})"));
                let l = self.decode1(seq, token)?;
                Ok(crate::qwen35::greedy(&l))
            }
        }
    }

    pub fn decode1(&mut self, seq: usize, token: u32) -> Result<Vec<f32>, Q4Error> {
        // 05-2: 직전 스텝이 예측한 토큰의 프리페치 완료 대기 (조인)
        if let Some(h) = self.ple_worker.take() {
            let _ = h.join();
        }
        // 예측 토큰 == 실제 입력 토큰이면 소비 대기로 스태시
        if let Some(slot) = self.ple_next.take()
            && let Ok(mut g) = slot.lock()
            && g.token == token
            && !g.emb.is_empty()
        {
            self.ple_consume = Some(vec![std::mem::take(&mut g.emb)]);
        }
        // 프레임 기본 ON(2026-09-02) — 상주 불가 시 1회 재시도 후 value 경로로
        // 영구 폴백. 게이트 실패는 mm 오류(호스트 폴백 가중치)로 첫 스텝 초반에
        // 발생해 상태 오염 전에 중단된다.
        let frame_on = self.frame_on(true);
        let frame_try = if frame_on {
            if self.frame_ensure() { Some(()) } else { None }
        } else {
            None
        };
        let logits = if let (true, Some(())) = (
            frame_on,
            frame_try
                .as_ref()
                .filter(|_| self.frame.is_some())
                .map(|_| ()),
        ) {
            let acc = self.acc.as_deref().unwrap();
            let f = self.frame.as_mut().unwrap();
            // plans/86 §3 — 트랜잭션: 시도 전 PLE 상태 스냅샷, Err 시 복원 후
            // 값경로 폴백(이중 진화 방지).
            let ple0 = super::frame::ple_snap(&self.seqs[seq]);
            let mut run_step = || -> Result<Vec<f32>, Q4Error> {
                if f.dirty[seq] {
                    f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
                }
                let ctx = Ctx {
                    model: &self.model,
                    acc: Some(acc),
                };
                super::frame::decode_frame(
                    acc,
                    &self.model,
                    &ctx,
                    seq,
                    &mut self.seqs[seq],
                    f,
                    token,
                )
            };
            let r = run_step();
            match r {
                Ok(l) => {
                    // P15④c: frame pre-mixer res_hc 행을 엔진 export로 pull.
                    if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                        self.last_res_hc_rows = f.last_res_hc_rows.clone();
                        self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
                    }
                    l
                }
                Err(e) => {
                    self.frame = None;
                    self.frame_broken = true;
                    super::frame::ple_restore(&mut self.seqs[seq], ple0);
                    // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                    if let Some(a) = self.acc.as_deref() {
                        a.frame_sync();
                    }
                    crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::FrameCreate);
                    eprintln!("# frame: 디코드 실패 — value 경로 폴백 ({e})");
                    self.forward_timed(seq, &[token])?
                }
            }
        } else {
            self.forward_timed(seq, &[token])?
        };
        self.seqs[seq].pos += 1;
        self.spawn_ple_prefetch(seq, &logits);
        Ok(logits)
    }

    /// 토큰 t의 로짓이 확정된 순간 t+1(=argmax)의 PLE 행을 사이드 스레드로
    /// 선적재 — 해시는 과거 토큰만의 함수라 오차 없는 선(先)적재 (05 §3).
    /// (plans/115 env 정리: PLE_PREFETCH 킬스위치 폐기 — 항시 선적재.)
    fn spawn_ple_prefetch(&mut self, seq: usize, logits: &[f32]) {
        if !self.model.hp.is_ple(1) || logits.is_empty() {
            return;
        }
        let (ptr, len, ty, hd) = match self.model.ple_table_view() {
            Ok(v) => v,
            Err(_) => return,
        };
        let next_tok = crate::qwen35::greedy(logits);
        let hp_ngram = self.model.hp.ple_ngram;
        let hist = self.seqs[seq].ple_hist.clone();
        let hist_valid = self.seqs[seq].ple_next_pos == self.seqs[seq].pos;
        let mult: Vec<u64> = self.model.hp.ple_multipliers.iter().copied().collect();
        let offs: Vec<u64> = self.model.hp.ple_head_offsets.iter().copied().collect();
        let vs: Vec<u64> = self.model.hp.ple_head_vocab_sizes.iter().copied().collect();
        let hpng = self.model.hp.ple_heads_per_ngram;
        let eos = self.model.hp.ple_eos;
        let heads = hpng * 2;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(PlePrefetched {
            token: next_tok,
            emb: Vec::new(),
        }));
        self.ple_next = Some(slot.clone());
        // plans/111 4차 W-P: ssd 오프로드 활성 시 mmap gather(페이지캐시 오염
        // + 스래시 원천) 대신 블록 캐시 선예열 — 본경로 디바이스 gather가
        // 예열된 캐시에서 즉시 적중한다.
        let acc = self.acc.clone();
        self.ple_worker = Some(std::thread::spawn(move || {
            let rows = pure_hash(
                &hist,
                hist_valid,
                &[next_tok],
                hp_ngram,
                hpng,
                &mult,
                &offs,
                &vs,
                eos,
            );
            if let Some(acc) = &acc
                && acc.ple_table_ssd_active()
            {
                acc.ple_ssd_warm(&rows);
                return;
            }
            // SAFETY (107 W8): ple_table_view 계약 — ptr..ptr+len은 PLE 테이블 mmap 유효 범위; 모델 가중이 유지되는 동안만 참조한다.
            let data: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
            let mut emb = vec![0.0f32; heads * hd];
            super::ple_gather_parts(data, ty, hd, &rows, &mut emb);
            if let Ok(mut g) = slot.lock() {
                g.emb = emb;
            }
        }));
    }

    pub fn piece(&self, tok: u32) -> String {
        self.model.piece(tok)
    }
}

/// 프리페치 워커용 순수 n-gram 해시 — 공용 코어(ple_hash_rows) 위임.
/// 종전 복제판은 lookback을 진행 중 hist에서 읽어 청크 경계에서 ple_hash와
/// 행이 갈라졌다(프리페치 적중 저하만 있고 토큰 무영향) — 109 P7 통합.
#[allow(clippy::too_many_arguments)]
fn pure_hash(
    hist: &[u32],
    hist_valid: bool,
    tokens: &[u32],
    ngram: usize,
    hpng: usize,
    mult: &[u64],
    offs: &[u64],
    vs: &[u64],
    eos: u32,
) -> Vec<u32> {
    crate::qwen4exp::stages::ple_hash_rows(
        hist, hist_valid, tokens, ngram, hpng, mult, offs, vs, eos,
    )
    .0
}

#[cfg(test)]
mod forward_tests {
    use super::*;

    const MODEL: &str =
        "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

    /// prefill+디코드 2토큰: 유한 logits·결정성·greedy 후보 정상 범위.
    #[test]
    fn forward_smoke() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("skip: {MODEL} 없음");
            return;
        }
        let m = Model4::load(std::path::Path::new(MODEL)).expect("load");
        let hp = m.hp.clone();
        let mut eng = Engine4::new(m, 1, 128);
        let toks: Vec<u32> = vec![760, 6511, 314];
        let l1 = eng.prefill(0, &toks).expect("prefill");
        assert_eq!(l1.len(), hp.vocab);
        assert!(l1.iter().all(|v| v.is_finite()), "logits 비유한");
        let t1 = crate::qwen35::greedy(&l1);
        assert!(t1 < hp.vocab as u32);
        let l2 = eng.decode1(0, t1).expect("decode");
        assert!(l2.iter().all(|v| v.is_finite()));
        // 결정성: 동일 경로 재실행 (새 엔진)
        let m2 = Model4::load(std::path::Path::new(MODEL)).expect("load2");
        let mut e2 = Engine4::new(m2, 1, 128);
        let l1b = e2.prefill(0, &toks).expect("prefill2");
        let t1b = crate::qwen35::greedy(&l1b);
        assert_eq!(t1, t1b, "greedy 불일치");
        // 결정성: 같은 입력 → 같은 로짓(첫 값 기준). `|| true` 로 항상 통과하던
        // 죽은 단언을 살렸다(2026-09-17 clippy 발견).
        assert!(
            (l1[0] - l1b[0]).abs() < 1e-6,
            "logit 비결정성: {} vs {}",
            l1[0],
            l1b[0]
        );
    }
}
