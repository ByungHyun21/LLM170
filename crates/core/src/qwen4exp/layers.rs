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
    if let Some(v) = std::env::var("LLM170_FRAME_TMAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
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
    std::env::var("LLM170_Q4_CHUNK")
        .ok()
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
        // ① 직전의 pre-mixer 잔차 = last_token의 예측자 hidden(① 전에 확보 —
        // decode1이 덮어쓴다). 이것이 주기 시작 커밋 토큰의 드래프트 쌍 h다.
        let h_prev = self.last_res_hc.clone();
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
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev)?;
            let _ = hp0;
        }
        // ② 드래프트 체인 — h는 pre-mix 멀티[10240]로 연결(chain export).
        let snap_d = self.mtp_seqs[seq].clone();
        let snap_t = self.seqs[seq].clone();
        let mut proposals: Vec<u32> = Vec::new();
        let mut chain_h = h_after_first.clone();
        let mut next = t0;
        for _ in 0..k.saturating_sub(1) {
            let (lg, dh) = self.mtp_draft_step_h(seq, next, &chain_h)?;
            proposals.push(next);
            chain_h = dh;
            next = crate::qwen35::greedy(&lg);
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
            let (_dl, _dh) = self.mtp_draft_step_h(seq, last_token, &h_prev)?;
            let mut dh = h_after_first.clone();
            for &p in &proposals[..=n_acc.min(proposals.len() - 1)] {
                let l = self.decode1(seq, p)?;
                forwards += 1;
                let (_dl, ndh) = self.mtp_draft_step_h(seq, p, &dh)?;
                dh = ndh;
                let _ = l;
            }
            accepted.extend_from_slice(&proposals[1..=n_acc.min(proposals.len() - 1)]);
            accepted.push(tgt_out[n_acc]);
        }
        Ok((accepted, forwards))
    }

    /// MTP 드래프트 프리필 (P15④) — 타깃 프리필 직후 호출. 프롬프트 토큰
    /// c_1..c_{T-1}을 (c_{i+1}, h_i) 쌍으로 드래프트 계층에 적립해 드래프트
    /// KV가 전체 문맥을 갖게 한다(빈 문맥 시작이 수용률 붕괴 원인 — 실측).
    /// last_h_rows는 직전 값경로 prefill의 h 행 전체.
    pub fn mtp_draft_prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<(), Q4Error> {
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
        self.mtp_seqs[seq].pos = 1;
        for i in 0..tokens.len().saturating_sub(1) {
            let x = tokens[i + 1];
            let hi = self.last_res_hc_rows[i].clone();
            let (_lg, _h) = self.mtp_draft_step_h(seq, x, &hi)?;
        }
        Ok(())
    }

    /// MTP 드래프트 스텝(vLLM qwen4_exp mtp.py 준거, plans/109 P15④).
    /// h 계약: **프리-믹서 멀티 스트림 잔차 [hc·n] 평탄화** — 타깃은
    /// last_res_hc(최종 hc_mix_head 이전), 체인은 직전 드래프트 스텝의
    /// pre-mix 반출. 반환: (로짓, pre-mix 멀티 [hc·n]).
    fn mtp_draft_step_h(
        &mut self,
        seq: usize,
        x: u32,
        h_pre: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Q4Error> {
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
        // eh_proj [2n→n] = 융합 [fc_embedding | fc_hidden]: 스트림 s 초기값 =
        //   fc_hidden(hn_s) + fc_embedding(en)  (vLLM amd: emb.unsqueeze + hidden).
        // fc_embedding = 입력 반쪽 [..n], fc_hidden = 입력 뒤반쪽 [n..2n].
        let weh = self.model.w4(&format!("blk.{il}.nextn.eh_proj.weight"))?;
        let mut res_hc: Vec<Vec<f32>> = Vec::with_capacity(1);
        {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
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
                acc: None,
            };
            stages::hc_mix(&ctx, il, "attn", &res_hc)?
        };
        let attn_out = self.mtp_dense_attn(seq, il, &mix)?;
        hc_combine(&mut res_hc, &attn_out, &inject, hc);
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
        let head_rows = {
            let ctx = Ctx {
                model: &self.model,
                acc: None,
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
                acc: None,
            };
            ctx.mm(&h1, &wout, &mut logits)?;
        }
        self.mtp_seqs[seq].pos += 1;
        // 체인 반출 = pre-mix 멀티 스트림(마지막 행).
        let chain_h = res_hc.last().cloned().unwrap_or_default();
        Ok((logits, chain_h))
    }

    /// MTP dense 게이트드 어텐션 (plans/109 P15②) — 트렁크 cpu_attn_row와
    /// 동일 산술(마스크=전체 참석) + q/k norm·rope + KV 적립 + wo.
    fn mtp_dense_attn(
        &mut self,
        seq: usize,
        il: usize,
        xs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        let hp = &self.model.hp;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
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
                acc: None,
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
        let pos0 = mtp_st.pos;
        let kq_scale = hp.kq_scale();
        let mut attn_all = vec![vec![0.0f32; n_head * hd]; n_tok];
        for t in 0..n_tok {
            let pos = pos0 + t as u32;
            // q: 헤드별 norm+rope(전반 hd) — 게이트 후반은 미가공.
            for h in 0..n_head {
                let lo = h * 2 * hd;
                let mut qh = crate::ops::rms_norm(&qg[t][lo..lo + hd], &qn, hp.eps);
                crate::ops::rope_head(&mut qh, pos, n_rot, hp.rope_base);
                qg[t][lo..lo + hd].copy_from_slice(&qh);
            }
            // k: kv헤드별 norm+rope → 캐시 적립. v: 원문 그대로.
            let kbase = pos as usize * n_kv * hd;
            for h in 0..n_kv {
                let lo = h * hd;
                let mut kh = crate::ops::rms_norm(&kk[t][lo..lo + hd], &kn, hp.eps);
                crate::ops::rope_head(&mut kh, pos, n_rot, hp.rope_base);
                mtp_st.kv_k[0][kbase + lo..kbase + lo + hd].copy_from_slice(&kh);
            }
            mtp_st.kv_v[0][kbase..kbase + n_kv * hd].copy_from_slice(&vv[t]);
            // dense softmax 어텐션 + 게이트 — cpu_attn_row 열에서 **cell 0은
            // 스킵**(드래프트 KV는 위치 1부터 기입 — 팬텀 0키가 softmax 질량을
            // 훔치는 결함, P15④-5).
            let n_past = pos as usize;
            let (ck, cv) = (&mtp_st.kv_k[0], &mtp_st.kv_v[0]);
            let out = &mut attn_all[t];
            for h in 0..n_head {
                let kvh = h / (n_head / n_kv);
                let mut maxv = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_past];
                for (p, sc) in scores.iter_mut().enumerate() {
                    let p = p + 1; // cell 0 스킵
                    let b = p * n_kv * hd + kvh * hd;
                    let mut d = 0.0f32;
                    for i in 0..hd {
                        d += qg[t][h * 2 * hd + i] * ck[b + i];
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
                    out[ob + i] *= sigmoid(qg[t][gb + i]);
                }
            }
        }
        // wo 투영 (대여 분리 — KV 적립 종료 후 새 Ctx).
        let ctx = Ctx {
            model: &self.model,
            acc: None,
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
        let attn_out = self.mtp_dense_attn(seq, il, &mix)?;
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
        self.seqs[seq] = SeqState4::new(&self.model.hp, ctx);
    }

    /// prefill: 전체 토큰 적립 + 마지막 logits.
    /// 1024토큰 청크로 분할 — 단일 초대형 forward는 libamdhip64 GPF 트리거
    /// (t=2311 실측, llama-server -ub 512도 같은 이유로 청크).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        // LLM170_Q4_CHUNK: 프리필 청크 토큰 수 (기본 1024; 프레임 경로는 t_max 상한).
        let cap0 = frame_t_max_cap(self.acc.as_deref());
        let chunk: usize = std::env::var("LLM170_Q4_CHUNK")
            .ok()
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
        let chunk: usize = std::env::var("LLM170_Q4_CHUNK")
            .ok()
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
        let frame_on = self.acc.is_some()
            && !self.frame_broken
            && std::env::var_os("LLM170_FRAME").is_some_and(|v| v != "0")
            && std::env::var("LLM170_FRAME_DECODE")
                .map(|v| v != "0")
                .unwrap_or(true);
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
    /// LLM170_PLE_PREFETCH=1 게이트. np 디코드: 마지막 활성 시퀀스만.
    fn spawn_ple_prefetch(&mut self, seq: usize, logits: &[f32]) {
        if std::env::var_os("LLM170_PLE_PREFETCH").is_none()
            || !self.model.hp.is_ple(1)
            || logits.is_empty()
        {
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
        // SAFETY: mmap 데이터 포인터는 Engine4(나아가 프로세스 수명)와 함께
        // 살고, worker는 다음 decode1 시작부에서 반드시 조인한다 — 조인 전
        // 엔진 drop 경로 없음 (서버 슬롯 루프도 decode1 직렬 호출).
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
