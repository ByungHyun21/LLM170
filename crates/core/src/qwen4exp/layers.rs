//! qwen4exp 상태·forward — HC 잔차, GDN(z-gate sigmoid), QSA(인덱서 top-k),
//! MoE(512·10+shared), PLE(n-gram 해시·게이트·dilated conv).
//!
//! 배선 근거: qwen4exp.cpp build_hc_mix/combine/build_qsa_top_k/build_attn_qsa/
//! build_ple + llama-graph.cpp build_moe_ffn (2026-08-30 판). 수치는 f32 참조.

use super::stages::{self, Ctx};
use super::{Hparams4, Model4, Q4Error};
use crate::matmul::Accelerator;
use crate::ops::sigmoid;
use crate::quant::dequant_row;
use llm170_diag::profile_span;

/// plans/115 P1-3: 체크포인트 전역 RAM 예산(바이트) — 슬롯 수로 분할해
/// 슬롯당 보관 수(1..=4)를 산정한다. 4슬롯×4개×112MB ≈ 1.8GB 상한.
const CKPT_BUDGET: usize = 2 << 30;

/// 시퀀스 상태 — GDN S/conv, QSA KV+인덱서 캐시, PLE conv 히스토리·n-gram 히스토리.
#[derive(Clone)]
pub struct SeqState4 {
    pub pos: u32,
    /// GDN층 S 상태 [dt_rank×d_state×d_state] (순환층 순서)
    pub gdn_s: Vec<Vec<f32>>,
    /// GDN depthwise conv 링 [(conv_k-1)×conv_ch]
    pub conv: Vec<Vec<f32>>,
    /// QSA층 KV [ctx][n_kv×hd] ×2 (full_idx 순서)
    pub kv_k: Vec<Vec<f32>>,
    pub kv_v: Vec<Vec<f32>>,
    /// QSA 인덱서 raw k 캐시 [ctx][idx_dim]
    pub idx_k: Vec<Vec<f32>>,
    /// QSA 인덱서 블록 키 캐시 [QSA층][n_blocks·idx_dim] — pooled+norm+rope
    /// 된 블록 키를 1회 계산해 전 토큰 재사용 (O(T²)→O(T), 2026-09-01).
    pub idx_bk: Vec<Vec<f32>>,
    /// PLE dilated conv 히스토리 [(kern-1)*dil][hc_dim]
    pub ple_conv: Vec<f32>,
    /// PLE n-gram 직전 토큰 히스토리 (최대 ngram-1개, 오래된 것이 앞)
    pub ple_hist: Vec<u32>,
    pub ple_next_pos: u32,
    /// plans/73: 디코드가 QSA 선택을 디바이스에서 수행해 호스트 kv/idx 캐시
    /// 갱신을 건너뛰었음. 프리필(t>1) 진입 시 풀에서 1회 재구축한다.
    pub qsa_host_stale: bool,
}

impl SeqState4 {
    pub fn new(hp: &Hparams4, ctx: usize) -> Self {
        let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recr(il)).count();
        let n_full = hp.n_layer - n_recr;
        let state_size = hp.dt_rank * hp.d_state * hp.d_state;
        let conv_len = (hp.conv_k - 1) * (hp.n_group * hp.d_state * 2 + hp.dt_rank * hp.d_state);
        let ple_hist_len = (hp.ple_conv_k - 1) * hp.ple_ngram;
        let has_ple = hp.is_ple(1);
        SeqState4 {
            pos: 0,
            gdn_s: vec![vec![0.0; state_size]; n_recr],
            conv: vec![vec![0.0; conv_len]; n_recr],
            kv_k: vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full],
            kv_v: vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full],
            idx_k: vec![vec![0.0; ctx * hp.idx_dim]; n_full],
            idx_bk: vec![Vec::new(); n_full],
            ple_conv: vec![
                0.0;
                if has_ple {
                    ple_hist_len * hp.hc * hp.n_embd
                } else {
                    0
                }
            ],
            ple_hist: Vec::new(),
            ple_next_pos: 0,
            qsa_host_stale: false,
        }
    }
}

impl SeqState4 {
    /// MTP 드래프트 전용 상태 (plans/109 P15②) — KV/idx 슬롯을 풀어텐션층
    /// +1(블록 n_layer)로 잡는다. GDN/PLE 슬롯은 미사용(0 유지).
    pub fn new_mtp(hp: &Hparams4, ctx: usize) -> Self {
        let mut st = SeqState4::new(hp, ctx);
        let n_full_mtp = st.kv_k.len() + 1;
        st.kv_k = vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full_mtp];
        st.kv_v = vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full_mtp];
        st.idx_k = vec![vec![0.0; ctx * hp.idx_dim]; n_full_mtp];
        st.idx_bk = vec![Vec::new(); n_full_mtp];
        st
    }
}

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

/// plans/115 P1-3: 접두 체크포인트 — 순환 상태 스냅샷.
/// kv_k/kv_v/idx_k는 pos 인덱스 쓰기(되감기 재프리필이 동일 행을 다시 쓴다
/// — 디바이스 풀 워터마크 계약, common/qsa.rs wm_advance)라 제외했다.
/// idx_bk는 블록 완결 길이만 저장(복원 시 truncate → 재프리필이 이어 재계산).
pub struct SeqCkpt {
    pub pos: u32,
    /// GDN/conv 스냅샷의 디바이스 링 슬롯(frame.ckpt_dev[seq] 인덱스).
    pub dev: usize,
    pub ple_conv: Vec<f32>,
    pub ple_hist: Vec<u32>,
    pub ple_next_pos: u32,
    /// [n_full] idx_bk.len() — 복원 truncate 기준
    pub idx_bk_lens: Vec<usize>,
    pub qsa_host_stale: bool,
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
    llm170_diag::flag::val("LLM170_Q4_CHUNK")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(cap)
        .clamp(16, 4096)
        .min(cap)
}

/// 프레임 환경 게이트(캐시) — LLM170_FRAME!=0 && {PREFILL,DECODE}!=0.
fn frame_env_on(decode: bool) -> bool {
    static PRE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static DEC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let lk = if decode { &DEC } else { &PRE };
    *lk.get_or_init(|| {
        std::env::var_os("LLM170_FRAME").is_some_and(|v| v != "0")
            && std::env::var(if decode {
                "LLM170_FRAME_DECODE"
            } else {
                "LLM170_FRAME_PREFILL"
            })
            .map(|v| v != "0")
            .unwrap_or(true)
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

    /// MTP 드래프트 스텝 (CPU 참조, plans/109 P15②) — 외장 nextn 블록
    /// (blk.{n_layer}) 1회 포워드. 입력: 직전 타깃 hidden h(프리-헤드,
    /// hc_mix_head 출력) + 채택 토큰 x. 출력: 드래프트 로짓 [vocab].
    /// 산술: qwen3next MTP 패턴(qwen4exp HC 변형) —
    ///   e=emb(x) → enorm·hnorm → eh_proj([en;hn]) → x̃
    ///   res_hc = x̃ 방송 → hc_mix(attn) → QSA(블록 n_layer, 자체 KV) →
    ///   hc_combine → hc_mix(ffn) → MoE → hc_combine →
    ///   hc_mix(nextn.hc_head) → output(@본체 공유) → logits.
    /// 드래프트 상태 pos는 호출부가 1씩 진행(mtp_seqs[seq].pos).
    /// MTP 스펙 스텝 (plans/109 P15③, CPU 참조판) — 드래프트 k-1토oken 체인
    /// 제안 + 타깃 순차 디코드 검증. q35 spec_step과 동일 계약:
    /// 반환 (수용 토큰열[보너스 포함], 타깃 forward 수).
    /// 상태 안전: 검증 전 타깃+드래프트 상태를 스냅샷해 거부 시 복원
    /// (SeqState4는 Vec 필드라 clone이 완전 복사). 값경로 전제 — 프레임
    /// 경로는 ④에서 프레임 트랜잭션(plans/86 §3)으로 동일 보장.
    pub fn mtp_spec_step(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        if k == 0 || !self.model.has_mtp() {
            return Ok((Vec::new(), 0));
        }
        // ── plans/110 W2: 프레임 경로 — 배치 검증(1회 t=k-1 포워드) ──
        // 실패 시 fb 카운터 + 순차(값경로) 폴백.
        if k >= 2 {
            match self.mtp_spec_step_frame(seq, last_token, k) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    super::frame::fb_incr(super::frame::FbId::MtpSpec);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("# mtp-spec-frame: 실패 — 순차 경로 폴백 ({e})"));
                }
            }
        }
        // ① 직전의 pre-mixer 잔차 = last_token의 예측자 hidden(① 전에 확보 —
        // decode1이 덮어쓴다). 이것이 주기 시작 커밋 토큰의 드래프트 쌍 h다.
        let h_prev = if self.spec_h_prev[seq].is_empty() {
            self.last_res_hc.clone()
        } else {
            std::mem::take(&mut self.spec_h_prev[seq])
        };
        let l = self.decode1(seq, last_token)?;
        let t0 = crate::qwen35::greedy(&l);
        let mut forwards = 1usize;
        let h_after_first = self.last_res_hc.clone();
        // ④′ 주기 시작 커밋 토큰(last_token)의 드래프트 KV 행을 **진위치에
        // 기입** — 종전 미기입이 드래프트 문맥에서 가장 최근 행을 잃게 했다.
        {
            let hp_hc = self.model.hp.hc * self.model.hp.n_embd;
            let hp0 = self.mtp_seqs[seq].pos;
            let _ = hp_hc;
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let _ = hp0;
        }
        // ② 드래프트 체인 — h는 pre-mix 멀티[10240]로 연결(chain export).
        let snap_d = self.mtp_seqs[seq].clone();
        let snap_t = self.seqs[seq].clone();
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k.saturating_sub(1) {
            let _acc = self.acc.clone();
            let (next_d, dh) = self.mtp_draft_step_h(seq, next, &chain_h, _acc.as_deref())?;
            proposals.push(next);
            chain_h = dh;
            next = next_d;
        }
        // ③ 검증 — 제안 순차 타깃 디코드 후 수용 접두 판정.
        let mut tgt_out = Vec::new();
        for &p in &proposals {
            let l = self.decode1(seq, p)?;
            forwards += 1;
            tgt_out.push(crate::qwen35::greedy(&l));
        }
        let mut n_acc = proposals.len();
        for i in 0..proposals.len() {
            if let Some(e) = proposals.get(i + 1)
                && tgt_out[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if n_acc + 1 >= proposals.len() {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*tgt_out.last().unwrap_or(&t0));
        } else {
            // 거부 — ① 직후 스냅샷 복원 후 주기시작 행 재기입 + 수용분 재적립.
            self.seqs[seq] = snap_t;
            self.mtp_seqs[seq] = snap_d;
            let hp_hc = self.model.hp.hc * self.model.hp.n_embd;
            let _ = hp_hc;
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc.min(proposals.len() - 1)] {
                let l = self.decode1(seq, p)?;
                forwards += 1;
                let _acc = self.acc.clone();
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh, _acc.as_deref())?;
                dh = ndh;
                let _ = l;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc.min(proposals.len() - 1)]);
            accepted.push(tgt_out[n_acc]);
        }
        self.spec_h_prev[seq] = self.last_res_hc.clone();
        Ok((accepted, forwards))
    }

    /// plans/115 P12 (Strata P0-3 서픽스 드래프터): 히스토리 접미 n-gram
    /// (길이 2..=8, 긴 것 우선)의 가장 최근 선행 출현 이후 k 토큰.
    /// llama.cpp prompt-lookup과 동일 규칙 — 드래프트 비용 0.
    pub fn suffix_drafts(hist: &[u32], k: usize) -> Vec<u32> {
        let h = hist.len();
        if h < 4 || k == 0 {
            return Vec::new();
        }
        let max_n = 8.min(h - 1);
        for n in (2..=max_n).rev() {
            let suf = &hist[h - n..];
            for start in (0..h - n).rev() {
                if &hist[start..start + n] == suf {
                    let out: Vec<u32> = hist[start + n..].iter().take(k).copied().collect();
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
        Vec::new()
    }

    /// plans/115 P12: 서픽스 제안 스펙 라운드 — 검증/수용/기각 재실행은
    /// mtp_spec_step_frame과 동일 기계(배치 verify + 배치 재실행, 43/43 등가
    /// 계약). 드래프트 헤드가 없다(MTP 무관 — 모델 공통). 제안이 비면 호출부가
    /// MTP/plain으로 폴백한다(이 메서드는 제안 있는 경우만 담당).
    pub fn suffix_spec_step(
        &mut self,
        seq: usize,
        last_token: u32,
        drafts: &[u32],
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        if drafts.is_empty() {
            return Ok((Vec::new(), 0));
        }
        if self.frame_on(true) && self.frame_ensure() {
            return self.suffix_spec_step_frame(seq, last_token, drafts);
        }
        // 프레임 경로 불가 — 순차 검증(값경로). 수용 접두 판정은 동일식.
        let l = self.decode1(seq, last_token)?;
        let t0 = crate::qwen35::greedy(&l);
        let mut forwards = 1usize;
        let proposals: Vec<u32> = std::iter::once(t0).chain(drafts.iter().copied()).collect();
        let snap_t = self.seqs[seq].clone();
        let mut tgt_out = Vec::new();
        for &p in &proposals {
            let li = self.decode1(seq, p)?;
            forwards += 1;
            tgt_out.push(crate::qwen35::greedy(&li));
        }
        let n_all = proposals.len() - 1;
        let mut n_acc = n_all;
        for i in 0..n_all {
            if let Some(e) = proposals.get(i + 1)
                && tgt_out[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = vec![t0];
        if n_acc >= n_all {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*tgt_out.last().unwrap_or(&t0));
        } else {
            self.seqs[seq] = snap_t;
            for &p in &proposals[..=n_acc] {
                self.decode1(seq, p)?;
                forwards += 1;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(tgt_out[n_acc]);
        }
        Ok((accepted, forwards))
    }

    /// 서픽스 스펙 프레임 판 — 배치 검증 + (기각 시) 배치 재실행.
    fn suffix_spec_step_frame(
        &mut self,
        seq: usize,
        last_token: u32,
        drafts: &[u32],
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        // ① 주기 시작 — last_token 디코드 1회(t0 산출·상태 전진).
        let l = self.decode1(seq, last_token)?;
        let t0 = crate::qwen35::greedy(&l);
        let mut forwards = 1usize;
        let proposals: Vec<u32> = std::iter::once(t0).chain(drafts.iter().copied()).collect();
        let snap_t = self.seqs[seq].clone();
        // ② 배치 검증 — 스냅샷 → t=len(proposals) 포워드 → 행별 argmax.
        let y: Vec<u32>;
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("suffix-spec: 프레임 없음".into()));
            };
            super::frame::verify_snap_capture(a, f, seq)?;
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            y = super::frame::frame_forward_verify(
                a,
                model,
                &ctx,
                seqs.as_mut_slice(),
                seq,
                f,
                &proposals,
            )?;
        }
        forwards += 1;
        // ③ 수용 접두 판정 + 정착 — full은 배치 상태 그대로, 기각은 복원 후
        // 배치 재실행(수용분 1회 포워드, 등가 계약).
        let n_all = proposals.len() - 1;
        let mut n_acc = n_all;
        for i in 0..n_all {
            if let Some(e) = proposals.get(i + 1)
                && y[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if n_acc >= n_all {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*y.last().unwrap_or(&t0));
            self.seqs[seq].pos += proposals.len() as u32;
            Ok((accepted, forwards))
        } else {
            // 기각 — CPU 상태 복원 + GDN 스냅샷 복원 + 수용분 배치 재실행.
            self.seqs[seq] = snap_t;
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("suffix-spec: 프레임 없음".into()));
                };
                super::frame::verify_snap_restore(a, f, seq)?;
            }
            let win: Vec<u32> = proposals[..=n_acc].to_vec();
            {
                let Engine4 {
                    model,
                    frame,
                    seqs,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("suffix-spec: 프레임 없음".into()));
                };
                let ctx = crate::qwen4exp::stages::Ctx {
                    model,
                    acc: Some(a),
                };
                super::frame::frame_forward_verify(
                    a,
                    model,
                    &ctx,
                    seqs.as_mut_slice(),
                    seq,
                    f,
                    &win,
                )?;
            }
            forwards += 1;
            self.seqs[seq].pos += win.len() as u32;
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(y[n_acc]);
            Ok((accepted, forwards))
        }
    }

    /// MTP 스펙 스텝 프레임 판 (plans/110 W2) — 검증을 t=k-1 배치 포워드
    /// 1회로 통합(np 불변식: 배치 == 순차 decode1 비트 동일). 기각 시
    /// GDN 디바이스 상태 스냅샷 복원 + PLE 링 pos 되감기 + 수용분 재실행.
    /// 수용 산출식은 순차 판과 동일 — 프레임 경로 상태 부패(스펙≠비스펙
    /// 토큰 분기, 110 W2 발견)도 이 트랜잭션으로 해소된다.
    fn mtp_spec_step_frame(
        &mut self,
        seq: usize,
        last_token: u32,
        k: usize,
    ) -> Result<(Vec<u32>, usize), Q4Error> {
        // ① 주기 시작 커밋 토큰의 타깃 forward — greedy 판정만 회수.
        let h_prev = if self.spec_h_prev[seq].is_empty() {
            self.last_res_hc.clone()
        } else {
            std::mem::take(&mut self.spec_h_prev[seq])
        };
        let t0 = self.decode1_greedy(seq, last_token)?;
        let mut forwards = 1usize;
        let h_after_first = self.last_res_hc.clone();
        // ④′ 주기 시작 커밋 토큰의 드래프트 KV 행 진위치 기입.
        {
            let _acc = self.acc.clone();
            self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
        }
        // ② 드래프트 체인 — proposals = [t0, g1, .., g_{k-2}](k-1개).
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k.saturating_sub(1) {
            let _acc = self.acc.clone();
            let (next_d, dh) = self.mtp_draft_step_h(seq, next, &chain_h, _acc.as_deref())?;
            proposals.push(next);
            chain_h = dh;
            next = next_d;
        }
        // ── 배치 검증: t=k-1행 1회 포워드 + 행별 GPU argmax ──
        let snap_t = self.seqs[seq].clone();
        let snap_d = self.mtp_seqs[seq].clone();
        let y: Vec<u32>;
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("mtp-spec-frame: 프레임 없음".into()));
            };
            // 검증 직전 GDN 디바이스 상태 스냅샷(기각 복원용).
            super::frame::verify_snap_capture(a, f, seq)?;
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            y = super::frame::frame_forward_verify(
                a,
                model,
                &ctx,
                seqs.as_mut_slice(),
                seq,
                f,
                &proposals,
            )?;
            // 다음 라운드 h 입력 — 배치 export 행 풀(마지막 행 = 마지막 처리 행).
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                self.last_res_hc_rows = f.last_res_hc_rows.clone();
                self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
            }
        }
        forwards += 1;
        // y[i] = proposals[i] 처리 후 greedy — 수용 접두 판정(순차 판과 동일식).
        let mut n_acc = proposals.len();
        for i in 0..proposals.len() {
            if let Some(e) = proposals.get(i + 1)
                && y[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let full = n_acc + 1 >= proposals.len();
        // 그림자 진단(LLM170_DUMP=spec_check) — 배치 y·상태와 순차 decode1
        // 재현을 전수 대조. 그림자 종료 상태 = 순차 전이(배치가 도달해야 할
        // 상태)라 관측이 스트림을 오염시키지 않는다.
        let shadow = llm170_diag::dump::opts().key("spec_check");
        if shadow {
            // 배치가 남긴 GDN 디바이스 상태.
            let batch_gdn = {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    unreachable!("프레임 존재");
                };
                let mut snap = vec![Vec::new(); f.st_gdn[seq].len()];
                for (ri, &h) in f.st_gdn[seq].iter().enumerate() {
                    snap[ri] = vec![0.0f32; f.gdn_state_len];
                    a.frame_read(h, &mut snap[ri]).map_err(Q4Error::Io)?;
                }
                snap
            };
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    unreachable!("프레임 존재");
                };
                super::frame::verify_snap_restore(a, f, seq)?;
            }
            self.seqs[seq] = snap_t.clone();
            self.mtp_seqs[seq] = snap_d.clone();
            let seq_y: Vec<u32> = proposals
                .iter()
                .map(|&p| self.decode1_greedy(seq, p))
                .collect::<Result<_, _>>()?;
            let mism: Vec<String> = (0..proposals.len())
                .filter(|&i| y[i] != seq_y[i])
                .map(|i| format!("y[{i}]={} seq={}", y[i], seq_y[i]))
                .collect();
            let mut gdn_bad = 0usize;
            let mut first = String::new();
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    unreachable!("프레임 존재");
                };
                for (ri, &h) in f.st_gdn[seq].iter().enumerate() {
                    let mut v = vec![0.0f32; f.gdn_state_len];
                    a.frame_read(h, &mut v).map_err(Q4Error::Io)?;
                    let bad = v
                        .iter()
                        .zip(batch_gdn[ri].iter())
                        .filter(|(x, b)| x.to_bits() != b.to_bits())
                        .count();
                    if bad > 0 && first.is_empty() {
                        first = format!("gdn[ri={ri}] {bad}");
                    }
                    gdn_bad += bad;
                }
            }
            eprintln!(
                "# spec-check pos={} full={} mismatch {} gdn_bad={gdn_bad} {first}",
                snap_t.pos,
                full,
                mism.len(),
            );
            if full {
                // 전수용: 순차 상태가 곧 정답 — 이 상태로 계속.
                let mut acc_v = vec![t0];
                acc_v.extend_from_slice(&proposals[1..]);
                acc_v.push(*seq_y.last().unwrap_or(&t0));
                return Ok((acc_v, forwards));
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if full {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*y.last().unwrap_or(&t0));
            // 배치가 정확히 proposals행만큼 상태를 전진시켰다 — pos 정산.
            self.seqs[seq].pos += proposals.len() as u32;
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok((accepted, forwards))
        } else {
            // 기각 — 스냅샷 복원(GDN 디바이스 + CPU) 후 수용분 재실행.
            let snap_pos = snap_t.pos;
            self.seqs[seq] = snap_t;
            self.mtp_seqs[seq] = snap_d;
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("mtp-spec-frame: 프레임 없음".into()));
                };
                super::frame::verify_snap_restore(a, f, seq)?;
            }
            // PLE 링: CPU snap_t.ple_conv가 정합 — 이후 첫 PLE 디바이스 스텝이
            // pos 기반 워터마크 되감기로 호스트 링을 리프레시한다(백엔드 계약).
            //
            // plans/115 P0-1(Strata commit-replay 1단계): 수용분 재실행을 종전
            // m× 순차 decode1(스텝당 ~55ms×m — 기각 라운드의 지배 비용)에서
            // **배치 verify 1회**로. t=k-1 검증 배치는 순차 decode1과 완전
            // 등가(43/43, 원장 128) — 상태 전진과 y 모두 동일 산술이다.
            {
                let Engine4 {
                    model,
                    frame,
                    seqs,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("mtp-spec-frame: 프레임 없음".into()));
                };
                let ctx = crate::qwen4exp::stages::Ctx {
                    model,
                    acc: Some(a),
                };
                let win: Vec<u32> = proposals[..=n_acc].to_vec();
                let ry = super::frame::frame_forward_verify(
                    a,
                    model,
                    &ctx,
                    seqs.as_mut_slice(),
                    seq,
                    f,
                    &win,
                )?;
                forwards += 1;
                if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                    self.last_res_hc_rows = f.last_res_hc_rows.clone();
                    self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
                }
                // 진단(spec_check) — 재실행 ry[i]는 원 배치 y[i]와 일치해야 한다.
                if shadow {
                    let mm: Vec<String> = (0..=n_acc)
                        .filter(|&i| ry[i] != y[i])
                        .map(|i| format!("r[{i}]={} y={}", ry[i], y[i]))
                        .collect();
                    eprintln!(
                        "# spec-reject pos={snap_pos} n_acc={} replay_mismatch {}{}",
                        n_acc,
                        mm.len(),
                        if mm.is_empty() {
                            String::new()
                        } else {
                            format!(" first={}", mm[0])
                        }
                    );
                }
            }
            self.seqs[seq].pos += (n_acc + 1) as u32;
            // 드래프트 재체인 — 종전대로(수용 토큰별 mtp_draft_step_h).
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc] {
                let _acc = self.acc.clone();
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh, _acc.as_deref())?;
                dh = ndh;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(y[n_acc]);
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok((accepted, forwards))
        }
    }

    /// np×spec 병합 스펙 라운드 (plans/110 W5) — 다중 슬롯의 라운드 시작
    /// decode1을 **1회 np 배치 포워드**로 묶고(무게 패스 공유), 드래프트·
    /// 검증·롤백은 슬롯별(배치 검증 기계 재사용). 반환 [slot][accepted],
    /// forwards 총합. 큐35 spec_step_multi의 Q4판 — 검증 배치 병합(원자
    /// 의미론)은 후속, 여기선 라운드 시작 병합만.
    pub fn mtp_spec_step_multi(
        &mut self,
        slots: &[usize],
        last_tokens: &[u32],
        k: usize,
    ) -> Result<(Vec<Vec<u32>>, usize), Q4Error> {
        if k == 0 || !self.model.has_mtp() || slots.is_empty() {
            return Ok((Vec::new(), 0));
        }
        if !(self.frame_on(true) && self.frame_ensure()) || slots.len() < 2 {
            // 폴백: 단일 슬롯 순차판(기존 mtp_spec_step).
            let mut out = Vec::with_capacity(slots.len());
            let mut fw = 0usize;
            for (&s, &t) in slots.iter().zip(last_tokens.iter()) {
                let (acc, f) = self.mtp_spec_step(s, t, k)?;
                fw += f;
                out.push(acc);
            }
            return Ok((out, fw));
        }
        // ① 다중 슬롯 라운드 시작 — 1회 np 배치(행핀으로 decode1 비트 동일)
        // + pre-mixer res_hc 행 export(드래프트 h_after_first).
        let toks: Vec<u32>;
        let mut row_h: Vec<Vec<f32>> = Vec::new();
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("mtp-spec-multi: 프레임 없음".into()));
            };
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            // 행핀 없음 — serve 비스펙(np) 경로와 동일 산술·무게 상각 유지.
            // (근접 타이 플립은 수용률 저하로만 나타난다 — 스펙 고유 성질.)
            let r = super::frame::frame_forward_np_greedy_h(
                a,
                model,
                &ctx,
                slots,
                seqs.as_mut_slice(),
                f,
                last_tokens,
            );
            toks = r?;
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                row_h = f.last_res_hc_rows.clone();
            }
        }
        let mut forwards = 1usize;
        let mut out = Vec::with_capacity(slots.len());
        for (i, (&s, &lt)) in slots.iter().zip(last_tokens.iter()).enumerate() {
            let t0 = toks.get(i).copied().unwrap_or(0);
            // ② 이 슬롯의 h_after_first = np export 행 i.
            let h_after_first = row_h.get(i).cloned().unwrap_or_default();
            let accepted = self.mtp_spec_round_rest(s, lt, t0, h_after_first, k, &mut forwards)?;
            out.push(accepted);
        }
        Ok((out, forwards))
    }

    /// 스펙 라운드의 잔여(④′ 드래프트 + 체인 + 배치 검증 + 수용/롤백) —
    /// 라운드 시작이 외부(다중 병합)에서 처리된 경우의 공유 본체.
    fn mtp_spec_round_rest(
        &mut self,
        seq: usize,
        last_token: u32,
        t0: u32,
        h_after_first: Vec<f32>,
        k: usize,
        forwards: &mut usize,
    ) -> Result<Vec<u32>, Q4Error> {
        // ④′ + 체인 드래프트 — h_prev는 슬롯별 직전 라운드 최종 export
        // (spec_h_prev; 110 W5 — 전역 last_res_hc는 타 슬롯이 덮는다).
        let h_prev = if self.spec_h_prev[seq].is_empty() {
            return Err(Q4Error::Io("spec_h_prev 미시드".into()));
        } else {
            std::mem::take(&mut self.spec_h_prev[seq])
        };
        {
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
        }
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k.saturating_sub(1) {
            let _acc = self.acc.clone();
            let (next_d, dh) = self.mtp_draft_step_h(seq, next, &chain_h, _acc.as_deref())?;
            proposals.push(next);
            chain_h = dh;
            next = next_d;
        }
        // 배치 검증 + 수용/롤백 — mtp_spec_step_frame의 ②이후와 동일 기계.
        let snap_t = self.seqs[seq].clone();
        let snap_d = self.mtp_seqs[seq].clone();
        let y: Vec<u32>;
        {
            let Engine4 {
                model,
                frame,
                seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                return Err(Q4Error::Io("mtp-spec-round: 프레임 없음".into()));
            };
            super::frame::verify_snap_capture(a, f, seq)?;
            let ctx = crate::qwen4exp::stages::Ctx {
                model,
                acc: Some(a),
            };
            y = super::frame::frame_forward_verify(
                a,
                model,
                &ctx,
                seqs.as_mut_slice(),
                seq,
                f,
                &proposals,
            )?;
            if f.mtp_h_export && !f.last_res_hc_rows.is_empty() {
                self.last_res_hc_rows = f.last_res_hc_rows.clone();
                self.last_res_hc = f.last_res_hc_rows.last().cloned().unwrap_or_default();
            }
        }
        *forwards += 1;
        let mut n_acc = proposals.len();
        for i in 0..proposals.len() {
            if let Some(e) = proposals.get(i + 1)
                && y[i] != *e
            {
                n_acc = i;
                break;
            }
        }
        let mut accepted = Vec::with_capacity(n_acc + 2);
        accepted.push(t0);
        if n_acc + 1 >= proposals.len() {
            accepted.extend_from_slice(&proposals[1..]);
            accepted.push(*y.last().unwrap_or(&t0));
            self.seqs[seq].pos += proposals.len() as u32;
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok(accepted)
        } else {
            self.seqs[seq] = snap_t;
            self.mtp_seqs[seq] = snap_d;
            {
                let Engine4 {
                    frame,
                    acc: acc_field,
                    ..
                } = self;
                let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                    return Err(Q4Error::Io("mtp-spec-round: 프레임 없음".into()));
                };
                super::frame::verify_snap_restore(a, f, seq)?;
            }
            let _acc = self.acc.clone();
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev, _acc.as_deref())?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc] {
                self.decode1_greedy(seq, p)?;
                *forwards += 1;
                let _acc = self.acc.clone();
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh, _acc.as_deref())?;
                dh = ndh;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc]);
            accepted.push(y[n_acc]);
            self.spec_h_prev[seq] = self.last_res_hc.clone();
            Ok(accepted)
        }
    }
    /// MTP 드래프트 프리필 (P15④) — 타깃 프리필 직후 호출. 프롬프트 토큰
    /// c_1..c_{T-1}을 (c_{i+1}, h_i) 쌍으로 드래프트 계층에 적립해 드래프트
    /// KV가 전체 문맥을 갖게 한다(빈 문맥 시작이 수용률 붕괴 원인 — 실측).
    /// last_h_rows는 직전 값경로 prefill의 h 행 전체.
    pub fn mtp_draft_prefill(
        &mut self,
        seq: usize,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<(), Q4Error> {
        if !self.model.has_mtp() || self.mtp_seqs.is_empty() {
            return Ok(());
        }
        if self.last_res_hc_rows.len() != tokens.len() {
            return Err(Q4Error::Io(format!(
                "mtp prefill: pre-mix h행({}) ≠ 토큰({}) — 값경로 프리필 직후에 호출",
                self.last_res_hc_rows.len(),
                tokens.len()
            )));
        }
        // 쌍 규약: 위치 p의 드래프트 입력은 (c_p, h_{p-1}). h_rows[i]는
        // c_i 처리 후 h — 즉 위치 i+1의 쌍은 (c_{i+1}, h_rows[i]).
        // **드래프트 KV 위치는 타깃 위치와 1:1**(vLLM "cell for cell") —
        // c_1은 pos 1에 적립(위치 0의 h_{-1}은 없음). 종전 pos 0 시작은
        // 로프 위치 전체를 1 어긋나게 했다(수용률 억제 원인, P15④).
        // base_pos(P1-3): 접두 복원 잡은 [cp..)만 재생 — 드래프트 KV [0..cp)는
        // 접두 불변이라 이미 유효(체크포인트가 pos만 되감았다).
        self.mtp_seqs[seq].pos = (base_pos + 1) as u32;
        for i in 0..tokens.len().saturating_sub(1) {
            let x = tokens[i + 1];
            let hi = self.last_res_hc_rows[i].clone();
            let _acc = self.acc.clone();
            let (_lg, _h) = self.mtp_draft_step_h(seq, x, &hi, _acc.as_deref())?;
        }
        // 110 W5: 슬롯별 스펙 h 시드 — 첫 스펙 라운드의 h_prev(프리필 최종 h).
        self.spec_h_prev[seq] = self.last_res_hc.clone();
        Ok(())
    }

    /// plans/115 D2: 잡 h행 세션 시작 — 슬롯 프리필 첫 청크 직전 호출.
    /// 잔존 행(이전 잡/스펙 라운드)이 mtp_draft_prefill의 len 검사를
    /// 깨고 드래프트 프리필을 영구 생략시켰다(수용률 붕괴).
    pub fn hrows_reset(&mut self, seq: usize) {
        self.last_res_hc_rows.clear();
        self.last_res_hc.clear();
        if let Some(f) = &mut self.frame {
            f.last_res_hc_rows.clear();
        }
        let _ = seq; // 현재 엔진 전역 행 — 슬롯 인자는 계약 문서화용
    }

    /// MTP 드래프트 스텝(vLLM qwen4_exp mtp.py 준거, plans/109 P15④).
    /// h 계약: **프리-믹서 멀티 스트림 잔차 [hc·n] 평탄화** — 타깃은
    /// last_res_hc(최종 hc_mix_head 이전), 체인은 직전 드래프트 스텝의
    /// pre-mix 반출. 반환: (greedy 토큰, pre-mix 멀티 [hc·n]).
    /// plans/110 W1: 프레임 경로 우선 — Frame4 상주 버퍼 + FrameOp 체인
    /// (값경로 GEMV의 호출당 d2h 동기 제거). 실패 시 fb 카운터 + 값경로 폴백.
    fn mtp_draft_step_h(
        &mut self,
        seq: usize,
        x: u32,
        h_pre: &[f32],
        acc: Option<&dyn Accelerator>,
    ) -> Result<(u32, Vec<f32>), Q4Error> {
        if !self.model.has_mtp() || self.mtp_seqs.is_empty() {
            return Err(Q4Error::Io("mtp_draft_step: MTP 미적재".into()));
        }
        let hp = &self.model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let il = hp.n_layer;
        let hc_dim = hc * n;
        if h_pre.len() != hc_dim {
            return Err(Q4Error::Io(format!(
                "mtp h 계약 위반: {} ≠ hc·n {}",
                h_pre.len(),
                hc_dim
            )));
        }
        let embd = self.model.w4("token_embd.weight")?;
        let mut e = vec![0.0f32; n];
        dequant_row(embd.ty, embd.data, x as u64, n as u64, &mut e);
        // enorm: 임베딩 플랫 정규화 [n]. hnorm: **멀티 스트림 전체 플랫 RMS**
        // [hc·n](vLLM GemmaRMSNorm(hidden*hc_count) 평탄 1회 — 스트림별 아님).
        // 두 norm은 공통(값·프레임 경로 동일 입력).
        let en = crate::ops::rms_norm(
            &e,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.enorm.weight"))?,
            hp.eps,
        );
        let hn = crate::ops::rms_norm(
            h_pre,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.hnorm.weight"))?,
            hp.eps,
        );
        // ── plans/110 W1: 프레임 경로 — 상주 버퍼 GEMV 체인 ──
        let mtp_t = std::time::Instant::now();
        if self.frame_on(true) && self.frame_ensure() {
            let Engine4 {
                model,
                frame,
                mtp_seqs,
                acc: acc_field,
                ..
            } = self;
            let (Some(f), Some(a)) = (frame.as_mut(), acc_field.as_deref()) else {
                unreachable!("frame_ensure 성공 직후");
            };
            match super::frame::mtp_draft_frame(a, model, f, &mut mtp_seqs[seq], &en, &hn) {
                Ok((tok, chain_h)) => {
                    if llm170_diag::dump::opts().key("mtp_time") {
                        eprintln!(
                            "# mtp-draft-frame: {:.2}ms",
                            mtp_t.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    mtp_seqs[seq].pos += 1;
                    return Ok((tok, chain_h));
                }
                Err(e) => {
                    super::frame::fb_incr(super::frame::FbId::MtpDraft);
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("# mtp-draft-frame: 실패 — 값경로 폴백 ({e})"));
                }
            }
        }
        // ── 값경로(종전 판) ──
        // eh_proj [2n→n] = 융합 [fc_embedding | fc_hidden]: 스트림 s 초기값 =
        //   fc_hidden(hn_s) + fc_embedding(en)  (vLLM amd: emb.unsqueeze + hidden).
        // fc_embedding = 입력 반쪽 [..n], fc_hidden = 입력 뒤반쪽 [n..2n].
        let weh = self.model.w4(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let mut res_hc: Vec<Vec<f32>> = Vec::with_capacity(1);
        {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            let mut r = vec![0.0f32; hc_dim];
            let mut x_t = vec![0.0f32; n];
            for s_i in 0..hc {
                let mut cat = vec![0.0f32; 2 * n];
                cat[..n].clone_from_slice(&en);
                cat[n..].clone_from_slice(&hn[s_i * n..(s_i + 1) * n]);
                ctx.mm(&cat, &weh, &mut x_t)?;
                r[s_i * n..(s_i + 1) * n].copy_from_slice(&x_t);
            }
            res_hc.push(r);
        }
        let (mix, inject) = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::hc_mix(&ctx, il, "attn", &res_hc)?
        };
        let attn_out = self.mtp_dense_attn(seq, il, &mix, acc)?;
        hc_combine(&mut res_hc, &attn_out, &inject, hc);
        let (mix2, inject2) = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::hc_mix(&ctx, il, "ffn", &res_hc)?
        };
        let ffn_out = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::moe_ffn(&ctx, il, &mix2)?
        };
        hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        let head_rows = {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            stages::hc_mix_nextn_head(&ctx, il, &res_hc)?
        };
        let h1 = head_rows
            .last()
            .ok_or(Q4Error::BadMeta("mtp 빈 헤드"))?
            .clone();
        let wout = self
            .model
            .w4("output.weight")
            .map_err(|_| Q4Error::MissingTensor("output.weight".into()))?;
        let mut logits = vec![0.0f32; wout.n_out as usize];
        {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            ctx.mm(&h1, &wout, &mut logits)?;
        }
        self.mtp_seqs[seq].pos += 1;
        // 체인 반출 = pre-mix 멀티 스트림(마지막 행).
        let chain_h = res_hc.last().cloned().unwrap_or_default();
        Ok((crate::qwen35::greedy(&logits), chain_h))
    }

    /// MTP dense 게이트드 어텐션 (plans/109 P15②) — 값경로 판. 투영(q/k/v)은
    /// mm_group, norm·rope·KV·softmax·게이트는 mtp_attn_cpu_row(공유 코어),
    /// wo 투영 mm_group. 프레임 경로(110 W1)는 frame/mtp.rs가 동일 코어를
    /// 판독한 q/k/v 행으로 호출한다.
    fn mtp_dense_attn(
        &mut self,
        seq: usize,
        il: usize,
        xs: &[Vec<f32>],
        acc: Option<&dyn Accelerator>,
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        let hp = &self.model.hp;
        let wq = self.model.w4(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.w4(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.w4(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.w4(&format!("blk.{il}.attn_output.weight"))?;
        let qn = self
            .model
            .f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
        let kn = self
            .model
            .f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
        let n_tok = xs.len();
        let mut qg = vec![vec![0.0f32; wq.n_out as usize]; n_tok];
        let mut kk = vec![vec![0.0f32; wk.n_out as usize]; n_tok];
        let mut vv = vec![vec![0.0f32; wv.n_out as usize]; n_tok];
        {
            let ctx = Ctx {
                model: &self.model,
                acc,
            };
            let mut gi = vec![
                std::mem::take(&mut qg),
                std::mem::take(&mut kk),
                std::mem::take(&mut vv),
            ];
            ctx.mm_group(xs, &[wq, wk, wv], &mut gi)?;
            qg = std::mem::take(&mut gi[0]);
            kk = std::mem::take(&mut gi[1]);
            vv = std::mem::take(&mut gi[2]);
        }
        let mtp_st = &mut self.mtp_seqs[seq];
        let mut attn_all = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            attn_all.push(mtp_attn_cpu_row(
                hp, &mut qg[t], &mut kk[t], &vv[t], mtp_st, &qn, &kn,
            ));
        }
        // wo 투영 (대여 분리 — KV 적립 종료 후 새 Ctx).
        let ctx = Ctx {
            model: &self.model,
            acc,
        };
        let mut out_rows = vec![vec![vec![0.0f32; wo.n_out as usize]; n_tok]; 1];
        ctx.mm_group(&attn_all, std::slice::from_ref(&wo), &mut out_rows)?;
        Ok(std::mem::take(&mut out_rows[0]))
    }

    pub fn mtp_draft_step(&mut self, seq: usize, x: u32, h: &[f32]) -> Result<Vec<f32>, Q4Error> {
        if !self.model.has_mtp() || self.mtp_seqs.is_empty() {
            return Err(Q4Error::Io("mtp_draft_step: MTP 미적재".into()));
        }
        let hp = &self.model.hp;
        let (n, hc) = (hp.n_embd, hp.hc);
        let il = hp.n_layer; // 블록 48
        let hc_dim = hc * n;
        // 1) e = emb(x) — 본체 임베딩 공유
        let embd = self.model.w4("token_embd.weight")?;
        let mut e = vec![0.0f32; n];
        dequant_row(embd.ty, embd.data, x as u64, n as u64, &mut e);
        // 2) enorm/hnorm
        let en = crate::ops::rms_norm(
            &e,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.enorm.weight"))?,
            hp.eps,
        );
        let hn = crate::ops::rms_norm(
            h,
            &self
                .model
                .f32_vec4(&format!("blk.{il}.nextn.hnorm.weight"))?,
            hp.eps,
        );
        // 3) eh_proj: [en;hn](2n) → x̃(n)
        let mut cat = vec![0.0f32; 2 * n];
        cat[..n].clone_from_slice(&en);
        cat[n..].clone_from_slice(&hn);
        let weh = self.model.w4(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let mut x_t = vec![0.0f32; n];
        {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            ctx.mm(&cat, &weh, &mut x_t)?;
        }
        // 4) res_hc 방송 (트렁크 hc_init와 동일)
        let mut res_hc: Vec<Vec<f32>> = Vec::with_capacity(1);
        {
            let mut r = vec![0.0f32; hc_dim];
            for s in 0..hc {
                r[s * n..(s + 1) * n].copy_from_slice(&x_t);
            }
            res_hc.push(r);
        }
        // 5) 어텐션 반쪽 — MTP 헤드는 **dense** 게이트드 어텐션(llama.cpp
        // qwen3next MTP 패턴: 인덱서 미사용, compress[48]=0과 일관). 트렁크
        // qsa의 cpu_attn_row 산술(전체 위치 마스크)과 동일 열.
        // 대여 분리: 각 스테이지가 스코프 ctx로 self.model만 빌린다.
        let (mix, inject) = {
            let ctx = Ctx {
                model: &self.model,
                acc: None, // CPU 참조 — GPU 경로는 ④(frame)에서
            };
            stages::hc_mix(&ctx, il, "attn", &res_hc)?
        };
        let _acc = self.acc.clone();
        let attn_out = self.mtp_dense_attn(seq, il, &mix, _acc.as_deref())?;
        hc_combine(&mut res_hc, &attn_out, &inject, hc);
        // 6) FFN 반쪽 (MoE + shexp)
        let (mix2, inject2) = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            stages::hc_mix(&ctx, il, "ffn", &res_hc)?
        };
        let ffn_out = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            stages::moe_ffn(&ctx, il, &mix2)?
        };
        hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        // 7) 드래프트 헤드 — nextn.hc_head 믹서 → 본체 output 공유
        let head_rows = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            stages::hc_mix_nextn_head(&ctx, il, &res_hc)?
        };
        let h1 = head_rows.last().ok_or(Q4Error::BadMeta("mtp 빈 헤드"))?;
        let wout = self
            .model
            .w4("output.weight")
            .map_err(|_| Q4Error::MissingTensor("output.weight".into()))?;
        let mut logits = vec![0.0f32; wout.n_out as usize];
        {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
            };
            ctx.mm(h1, &wout, &mut logits)?;
        }
        self.mtp_seqs[seq].pos += 1;
        Ok(logits)
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

    /// plans/115 P1-3: 접두 체크포인트 캡처 — 프리필 청크 경계 호출.
    /// 간격 게이트(≥512토큰): 배치 프리필은 per=16..256 청크로 잘리므로 매
    /// 청크 클론(≈112MB)이면 TTFT를 갉아먹는다. 프레임 경로는 CPU gdn_s/conv
    /// 을 갱신하지 않으므로 캡처 직전 풀백(D2H)이 필수다(dirty면 CPU 권위 —
    /// 풀백 불요). 실패 시 캡처 생략(부분 재사용만 늦춰질 뿐 정확성 무영향).
    pub fn ckpt_capture(&mut self, seq: usize, pos: usize) {
        const MIN_SPACING: usize = 512;
        if pos < MIN_SPACING {
            return;
        }
        if let Some(c) = self.ckpt[seq].back() {
            let last = c.pos as usize;
            if pos <= last || pos - last < MIN_SPACING {
                return; // 후퇴/중복/과밀 캡처 무시
            }
        }
        // GDN/conv은 디바이스 D2D 스냅샷(스트림 순서 — 청크 직후 상태 그대로).
        // 프레임 없음(CPU 서빙)은 체크포인트 없음 — 부분 재사용 불가(폴백 reset).
        let t0 = std::time::Instant::now();
        let Some(acc) = self.acc.clone() else { return };
        let Some(f) = self.frame.as_mut() else { return };
        // 링 슬롯 확보 — free-list 우선, 부족하면 신규 할당, 상한이면 최旧
        // 체크포인트를 퇴출해 슬롯을 물려받는다.
        let slot = if let Some(g) = f.ckpt_free[seq].pop() {
            g
        } else if f.ckpt_dev[seq].len() < self.ckpt_keep {
            let g = f.ckpt_dev[seq].len();
            let mut gs = Vec::with_capacity(f.st_gdn[seq].len());
            let mut cs = Vec::with_capacity(f.st_conv[seq].len());
            for _ in 0..f.st_gdn[seq].len() {
                match acc.frame_alloc(f.gdn_state_len) {
                    Ok(h) => gs.push(h),
                    Err(e) => {
                        eprintln!("# ckpt: gdn 버퍼 할당 실패 — 캡처 중단({e})");
                        return;
                    }
                }
            }
            for _ in 0..f.st_conv[seq].len() {
                match acc.frame_alloc(f.conv_state_len) {
                    Ok(h) => cs.push(h),
                    Err(e) => {
                        eprintln!("# ckpt: conv 버퍼 할당 실패 — 캡처 중단({e})");
                        return;
                    }
                }
            }
            f.ckpt_dev[seq].push((gs, cs));
            g
        } else if let Some(old) = self.ckpt[seq].pop_front() {
            old.dev
        } else {
            return;
        };
        let (gd, cv) = f.ckpt_dev[seq][slot].clone();
        let mut pairs: Vec<(u64, u64, usize)> = Vec::with_capacity(gd.len() + cv.len());
        for (d, &src) in gd.iter().zip(f.st_gdn[seq].iter()) {
            pairs.push((*d, src, f.gdn_state_len * 4));
        }
        for (d, &src) in cv.iter().zip(f.st_conv[seq].iter()) {
            pairs.push((*d, src, f.conv_state_len * 4));
        }
        if let Err(e) = acc.frame_copy_states(&pairs) {
            eprintln!("# ckpt: D2D 캡처 실패 — 생략({e})");
            f.ckpt_free[seq].push(slot);
            return;
        }
        let st = &self.seqs[seq];
        self.ckpt[seq].push_back(SeqCkpt {
            pos: pos as u32,
            dev: slot,
            ple_conv: st.ple_conv.clone(),
            ple_hist: st.ple_hist.clone(),
            ple_next_pos: st.ple_next_pos,
            idx_bk_lens: st.idx_bk.iter().map(|v| v.len()).collect(),
            qsa_host_stale: st.qsa_host_stale,
        });
        if llm170_diag::dump::opts().key("ckpt_time") {
            eprintln!(
                "# ckpt capture seq{seq} pos{pos}: {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// plans/115 P1-3: l 이하 최신 체크포인트로 되감기 — 복원 위치 반환(없으면
    /// None, 상태 무변). kv/idx_k 호스트 행은 접두 불변(그대로), idx_bk는 블록
    /// 길이까지 truncate — 재프리필 [cp..)이 pos 인덱스로 동일 행을 다시 쓴다.
    /// GDN은 dirty=true → 다음 프리필 sync_states가 CPU 클론을 디바이스로 전사.
    /// PLE 링 캐시 무효(acc_reset_seq — pos 기반 되감기 계약).
    pub fn ckpt_restore_upto(&mut self, seq: usize, l: usize) -> Option<usize> {
        let idx = self.ckpt[seq].iter().rposition(|c| (c.pos as usize) <= l)?;
        let ck = self.ckpt[seq].remove(idx).unwrap();
        // 복원 시점 이후 체크포인트는 무효(이후 토큰열이 갈린다) — 슬롯 회수.
        for dropped in self.ckpt[seq].drain(idx..) {
            if let Some(f) = self.frame.as_mut() {
                f.ckpt_free[seq].push(dropped.dev);
            }
        }
        let acc = self.acc.clone()?;
        let f = self.frame.as_mut()?;
        // GDN/conv D2D 되감기 — 디바이스가 곧 권위다(dirty=false 유지).
        // CPU gdn_s는 stale가 되는데 이는 통상 디코드 체제와 동일 — 값경로
        // 진입 시 frame_pullback_cpu가 디바이스에서 지연 갱신한다.
        let (gd, cv) = f.ckpt_dev[seq][ck.dev].clone();
        let mut pairs: Vec<(u64, u64, usize)> = Vec::with_capacity(gd.len() + cv.len());
        for (&d, &src) in gd.iter().zip(f.st_gdn[seq].iter()) {
            pairs.push((src, d, f.gdn_state_len * 4));
        }
        for (&d, &src) in cv.iter().zip(f.st_conv[seq].iter()) {
            pairs.push((src, d, f.conv_state_len * 4));
        }
        let pos = ck.pos as usize;
        f.ckpt_free[seq].push(ck.dev);
        if acc.frame_copy_states(&pairs).is_err() {
            return None; // 상태 무변 — 호출부는 reset 폴백
        }
        acc.acc_reset_seq(seq);
        f.dirty[seq] = false; // 디바이스 GDN = 체크포인트 상태(권위)
        f.last_res_hc_rows = Vec::new(); // MTP h 행 — 새 프리필이 재적립
        self.spec_h_prev[seq] = Vec::new();
        self.last_res_hc_rows = Vec::new();
        // 드래프트 KV는 pos 인덱스 쓰기·접두 불변 — [0..cp)가 그대로 유효하므로
        // pos만 되감는다(신규 초기화보다 낫다: 문맥 보존 → 수용률 유지).
        // [cp..]의 재생은 mtp_draft_prefill(base_pos=cp)이 맡는다.
        if self.model.has_mtp() && !self.mtp_seqs.is_empty() {
            self.mtp_seqs[seq].pos = ck.pos;
        }
        let st = &mut self.seqs[seq];
        st.pos = ck.pos;
        st.ple_conv = ck.ple_conv;
        st.ple_hist = ck.ple_hist;
        st.ple_next_pos = ck.ple_next_pos;
        st.qsa_host_stale = ck.qsa_host_stale;
        for (v, &len) in st.idx_bk.iter_mut().zip(ck.idx_bk_lens.iter()) {
            v.truncate(len);
        }
        Some(pos)
    }

    /// 접두 체크포인트 전체 폐기 — reset_seq/reset_states에서 호출(새 대화의
    /// 체크포인트가 이전 토큰열을 역참조하지 않게).
    fn ckpt_clear(&mut self, seq: Option<usize>) {
        match seq {
            Some(s) => self.ckpt[s].clear(),
            None => {
                for q in &mut self.ckpt {
                    q.clear();
                }
            }
        }
    }

    /// prefill: 전체 토큰 적립 + 마지막 logits.
    /// 1024토큰 청크로 분할 — 단일 초대형 forward는 libamdhip64 GPF 트리거
    /// (t=2311 실측, llama-server -ub 512도 같은 이유로 청크).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        // LLM170_Q4_CHUNK: 프리필 청크 토큰 수 (기본 1024; 프레임 경로는 t_max 상한).
        let cap0 = frame_t_max_cap(self.acc.as_deref());
        let chunk: usize = llm170_diag::flag::val("LLM170_Q4_CHUNK")
            .and_then(|v| v.parse().ok())
            .unwrap_or(cap0)
            .clamp(16, 4096); // 상한은 frame_t_max_cap이 결정(적응형)
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
        let chunk: usize = llm170_diag::flag::val("LLM170_Q4_CHUNK")
            .and_then(|v| v.parse().ok())
            .unwrap_or(cap0)
            .clamp(16, 4096)
            .min(frame_t_max_cap(self.acc.as_deref()));
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

/// MTP dense 어텐션 CPU 코어(plans/110 W1 분리) — q/k/v 원시 행 1개에
/// norm·rope·KV 적립·softmax(cell 0 스킵)·게이트를 적용해 [n_head·hd] 반환.
/// 값경로(mtp_dense_attn)와 프레임 경로(frame/mtp.rs)가 공유 — 산술 단일 소스.
pub(crate) fn mtp_attn_cpu_row(
    hp: &Hparams4,
    q_row: &mut [f32],
    k_row: &mut [f32],
    v_row: &[f32],
    st: &mut SeqState4,
    qn: &[f32],
    kn: &[f32],
) -> Vec<f32> {
    let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
    let pos = st.pos;
    let kq_scale = hp.kq_scale();
    // q: 헤드별 norm+rope(전반 hd) — 게이트 후반은 미가공.
    for h in 0..n_head {
        let lo = h * 2 * hd;
        let mut qh = crate::ops::rms_norm(&q_row[lo..lo + hd], qn, hp.eps);
        crate::ops::rope_head(&mut qh, pos, n_rot, hp.rope_base);
        q_row[lo..lo + hd].copy_from_slice(&qh);
    }
    // k: kv헤드별 norm+rope → 캐시 적립. v: 원문 그대로.
    let kbase = pos as usize * n_kv * hd;
    for h in 0..n_kv {
        let lo = h * hd;
        let mut kh = crate::ops::rms_norm(&k_row[lo..lo + hd], kn, hp.eps);
        crate::ops::rope_head(&mut kh, pos, n_rot, hp.rope_base);
        st.kv_k[0][kbase + lo..kbase + lo + hd].copy_from_slice(&kh);
    }
    st.kv_v[0][kbase..kbase + n_kv * hd].copy_from_slice(v_row);
    // dense softmax 어텐션 + 게이트 — cpu_attn_row 열에서 **cell 0은
    // 스킵**(드래프트 KV는 위치 1부터 기입 — 팬텀 0키가 softmax 질량을
    // 훔치는 결함, P15④-5).
    let n_past = pos as usize;
    let (ck, cv) = (&st.kv_k[0], &st.kv_v[0]);
    let mut out = vec![0.0f32; n_head * hd];
    for h in 0..n_head {
        let kvh = h / (n_head / n_kv);
        let mut maxv = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; n_past];
        for (p, sc) in scores.iter_mut().enumerate() {
            let p = p + 1; // cell 0 스킵
            let b = p * n_kv * hd + kvh * hd;
            let mut d = 0.0f32;
            for i in 0..hd {
                d += q_row[h * 2 * hd + i] * ck[b + i];
            }
            *sc = d * kq_scale;
            maxv = maxv.max(*sc);
        }
        let mut sum = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - maxv).exp();
            sum += *sc;
        }
        let ob = h * hd;
        for (p0, sc) in scores.iter().enumerate() {
            let w = sc / sum;
            if w == 0.0 {
                continue;
            }
            let b = (p0 + 1) * n_kv * hd + kvh * hd;
            for i in 0..hd {
                out[ob + i] += w * cv[b + i];
            }
        }
        let gb = h * 2 * hd + hd;
        for i in 0..hd {
            out[ob + i] *= sigmoid(q_row[gb + i]);
        }
    }
    out
}
/// hc_combine: res[s] += out·(2·σ(inject_s/4)).
fn hc_combine(res_hc: &mut [Vec<f32>], out: &[Vec<f32>], inject: &[Vec<f32>], hc: usize) {
    for (t, o) in out.iter().enumerate() {
        for s in 0..hc {
            let w = 2.0 * sigmoid(inject[t][s] / hc as f32);
            let base = s * o.len();
            for (i, ov) in o.iter().enumerate() {
                res_hc[t][base + i] += ov * w;
            }
        }
    }
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
