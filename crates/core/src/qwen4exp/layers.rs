//! qwen4exp 상태·forward — HC 잔차, GDN(z-gate sigmoid), QSA(인덱서 top-k),
//! MoE(512·10+shared), PLE(n-gram 해시·게이트·dilated conv).
//!
//! 배선 근거: qwen4exp.cpp build_hc_mix/combine/build_qsa_top_k/build_attn_qsa/
//! build_ple + llama-graph.cpp build_moe_ffn (2026-08-30 판). 수치는 f32 참조.

use super::{Hparams4, Model4, Q4Error};
use super::stages::{self, Ctx};
use crate::matmul::Accelerator;
use crate::ops::sigmoid;
use crate::quant::dequant_row;
use llm170_diag::profile_span;

/// 시퀀스 상태 — GDN S/conv, QSA KV+인덱서 캐시, PLE conv 히스토리·n-gram 히스토리.
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
            ple_conv: vec![0.0; if has_ple { ple_hist_len * hp.hc * hp.n_embd } else { 0 }],
            ple_hist: Vec::new(),
            ple_next_pos: 0,
            qsa_host_stale: false,
        }
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
    /// 디코드 그래프 캡처(LLM170_GRAPH=1) — 0=워밍, 1=캡처, 2+=재생.
    graph_want: bool,
    graph_step: usize,
    /// PLE 프리페치 (05-2) — 토큰 t 확정 직후 t+1분 16행×ple_head_dim을
    /// 사이드 스레드에서 mmap 읽기+디양자화. 다음 decode의 ple_block이 소비.
    pub ple_next: Option<std::sync::Arc<std::sync::Mutex<PlePrefetched>>>,
    /// 소비 대기 emb (prefetch 히트분) — decode1/prefill이 채우고 forward가 take.
    ple_consume: Option<Vec<Vec<f32>>>,
    /// 프리페치 사이드 스레드 핸들 — 다음 스텝 시작부 조인 (mmap 수명 보장).
    ple_worker: Option<std::thread::JoinHandle<()>>,
}

/// 사이드 스레드가 채운 프리페치 결과 — token이 다음 입력과 일치할 때만 사용.
pub struct PlePrefetched {
    pub token: u32,
    pub emb: Vec<f32>,
}

/// 스테이지별 누적(µs) — LLM170_Q4_TIME=1일 때 prefill/decode 완료 후 보고.
#[derive(Default)]
pub struct Q4Timings {
    pub hc: u64,
    pub gdn: u64,
    pub qsa: u64,
    pub moe: u64,
    pub ple: u64,
    pub head: u64,
}

impl Q4Timings {
    fn report(&self, tag: &str) {
        eprintln!(
            "# q4-timing {tag}: hc={:.0}ms gdn={:.0}ms qsa={:.0}ms moe={:.0}ms ple={:.0}ms head={:.0}ms",
            self.hc as f64 / 1e3,
            self.gdn as f64 / 1e3,
            self.qsa as f64 / 1e3,
            self.moe as f64 / 1e3,
            self.ple as f64 / 1e3,
            self.head as f64 / 1e3
        );
    }
}

/// 프레임 버퍼의 토큰 상한 — 프리필 청크와 동일(디코드 t=1 포함).
/// 512 상한: t_max 버퍼는 청크에 비례하고(≈0.8 GB @512), 1024는 실측
/// hipMalloc OOM이었다. 값 경로 청크(1024)와 독립.
const FRAME_T_MAX: usize = 512;

/// 프레임 토큰 상한 — 기본 512(8GB CMP에서 1024는 hipMalloc OOM 이력).
/// `LLM170_FRAME_TMAX`로 올릴 수 있다(대형 VRAM 기기: 전문가당 행 수가 늘어
/// MoE 가중치 재사용이 좋아진다).
fn frame_t_max_cap(acc: Option<&dyn crate::matmul::Accelerator>) -> usize {
    if let Some(v) = std::env::var("LLM170_FRAME_TMAX").ok().and_then(|v| v.parse::<usize>().ok()) {
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

impl Engine4 {
    pub fn new(model: Model4, n_seqs: usize, ctx: usize) -> Self {
        let seqs = (0..n_seqs).map(|_| SeqState4::new(&model.hp, ctx)).collect();
        Engine4 {
            model, seqs, acc: None, frame: None, frame_broken: false,
            graph_want: std::env::var_os("LLM170_GRAPH").is_some(),
            graph_step: 0,
            ple_next: None, ple_consume: None, ple_worker: None,
        }
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
    fn frame_on(&self, decode: bool) -> bool {
        self.acc.is_some()
            && !self.frame_broken
            && self.acc.as_ref().is_some_and(|a| a.frame_capable())
            && std::env::var_os("LLM170_FRAME").is_some_and(|v| v != "0")
            && std::env::var(if decode { "LLM170_FRAME_DECODE" } else { "LLM170_FRAME_PREFILL" })
                .map(|v| v != "0")
                .unwrap_or(true)
    }

    /// 프레임 GPU 상태 → CPU 사본 풀백(값 경로 진입 전 정합화, plans/90 A1 D12).
    /// gdn은 전치 레이아웃(AR 커널 규약) 역변환 포함.
    /// 프레임 없음·가속기 없음·dirty(CPU가 권위)면 no-op.
    fn frame_pullback_cpu(&mut self, seq: usize) -> Result<(), Q4Error> {
        let Some(acc) = self.acc.as_deref() else { return Ok(()) };
        let (gdn, conv) = {
            let Some(f) = self.frame.as_ref() else { return Ok(()) };
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

    fn forward_timed(
        &mut self,
        seq: usize,
        tokens: &[u32],
        mut tm: Option<&mut Q4Timings>,
    ) -> Result<Vec<f32>, Q4Error> {
        profile_span!("q4::forward");
        macro_rules! stage {
            ($field:ident, $body:expr) => {
                match &mut tm {
                    Some(t) => {
                        let t0 = std::time::Instant::now();
                        let r = $body;
                        t.$field += t0.elapsed().as_micros() as u64;
                        r
                    }
                    None => $body,
                }
            };
        }

        let hp = self.model.hp.clone();
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t_len = tokens.len();
        // 스테이지 컨텍스트 — model 불변 차입 + seq 상태 가변 차입 (필드 분리)
        let ctx = Ctx { model: &self.model, acc: self.acc.as_deref() };
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
        let ple_rows = if hp.is_ple(1) { stages::ple_hash(&ctx, seq_st, tokens) } else { Vec::new() };

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
        let trace = std::env::var_os("LLM170_Q4_TRACE").is_some();
        // NaN 조기 국소화 — 발산 층·스테이지를 즉시 보고 (LLM170_Q4_TRACE).
        let nan_guard = |v: &[Vec<f32>], tag: &str, il: usize| {
            for (ti, row) in v.iter().enumerate() {
                if row.iter().any(|x| !x.is_finite()) {
                    eprintln!("# NaN발견 layer={il} {tag} token={ti} t={}", row.len());
                    std::process::exit(101);
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
                let pre = if t_len == 1 { self.ple_consume.take() } else { None };
                stage!(ple, stages::ple_block(&ctx, seq_st, il, &mut res_hc, &ple_rows, pre)?);
            }
        let (mix, inject) = stage!(hc, stages::hc_mix(&ctx, il, "attn", &res_hc)?);
        let attn_out = if hp.is_recr(il) {
            if super::frame::stage_skipped("gdn") {
                // 진단용: GDN 생략(출력 무효) — 청크 의존 축 분리(plans/80).
                recr_idx += 1;
                vec![vec![0.0f32; n_embd]; t_len]
            } else {
                let o = stage!(gdn, stages::gdn_layer(&ctx, seq_st, il, &mix, t_len, recr_idx)?);
                recr_idx += 1;
                o
            }
        } else if super::frame::stage_skipped("qsa") {
            full_idx += 1;
            vec![vec![0.0f32; n_embd]; t_len]
        } else {
            let o = stage!(qsa, stages::qsa_layer(&ctx, seq_st, il, &mix, t_len, full_idx)?);
            full_idx += 1;
            o
        };
            if trace {
                nan_guard(&attn_out, if hp.is_recr(il) { "gdn_out" } else { "qsa_out" }, il);
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
                    &ffn_out.iter().flat_map(|r| r.iter()).copied().collect::<Vec<_>>(),
                );
            }
            if trace {
                nan_guard(&ffn_out, "moe_out", il);
            }
            hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        }

        // output HC mix → logits (inject 없음)
        let head_in = stage!(head, stages::hc_mix_head(&ctx, &res_hc)?);
        let last = head_in.last().ok_or(Q4Error::BadMeta("빈 배치"))?.clone();
        let wout = self.model.w("output.weight").ok_or(Q4Error::MissingTensor("output.weight".into()))?;
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
        let ctx = self.seqs.first().map(|s| s.kv_k.first().map(|k| k.len() / (self.model.hp.n_kv * self.model.hp.head_dim)).unwrap_or(4096)).unwrap_or(4096);
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
        let ctx = self.seqs[seq].kv_k.first().map(|k| k.len() / (self.model.hp.n_kv * self.model.hp.head_dim)).unwrap_or(4096);
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
            .clamp(16, 4096);   // 상한은 frame_t_max_cap이 결정(적응형)
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
        let chunk = if frame_on { chunk.min(frame_t_max_cap(self.acc.as_deref())) } else { chunk };
        if frame_on {
            let acc = self.acc.as_deref().unwrap();
            if self.frame.is_none() {
                match super::frame::Frame4::new(acc, &self.model, &self.seqs, frame_t_max(Some(acc))) {
                    Ok(f) => self.frame = Some(f),
                    Err(e) => {
                        self.frame_broken = true;
                        eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
                    }
                }
            }
        }
        if let Some(f) = self.frame.as_mut().filter(|_| frame_on) {
            let acc = self.acc.as_deref().unwrap();
            let mut last = None;
            for ch in tokens.chunks(chunk) {
                if f.dirty[seq] {
                    f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
                }
                let ctx = Ctx { model: &self.model, acc: Some(acc) };
                let logits = super::frame::frame_forward(
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
                last = Some(logits);
            }
            return Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]));
        }
        // 값 경로 전용 풀백(프레임 경로 미사용 시에만).
        if !need_cpu_pullback {
            self.frame_pullback_cpu(seq)?;
        }
        let mut last = None;
        for ch in tokens.chunks(chunk) {
            let mut tm = init_timings();
            let logits = self.forward_timed(seq, ch, tm.as_mut())?;
            if let Some(t) = &tm {
                t.report(&format!("prefill {}tok", ch.len()));
            }
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
        if self.frame.is_none() {
            match super::frame::Frame4::new(self.acc.as_deref().unwrap(), &self.model, &self.seqs, frame_t_max(self.acc.as_deref())) {
                Ok(f) => self.frame = Some(f),
                Err(e) => {
                    self.frame_broken = true;
                    eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
                    let l = self.prefill(seq, tokens)?;
                    return Ok(crate::qwen35::greedy(&l));
                }
            }
        }
        let acc = self.acc.as_deref().unwrap();
        let f = self.frame.as_mut().unwrap();
        let mut last = 0u32;
        for ch in tokens.chunks(chunk) {
            if f.dirty[seq] {
                f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
            }
            let ctx = Ctx { model: &self.model, acc: Some(acc) };
            last = super::frame::frame_forward_greedy(
                acc, &self.model, &ctx, seq, &mut self.seqs[seq], f, ch,
            )?;
            self.seqs[seq].pos += ch.len() as u32;
            f.dirty[seq] = false;
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
        let acc = self.acc.clone().ok_or(Q4Error::Io("prefill_multi: 가속기 없음".into()))?;
        if tokens.len() != seqs.len() * per_seq || seqs.len() < 2 {
            return Err(Q4Error::Io("prefill_multi: 계약 위반".into()));
        }
        if self.frame.is_none() {
            let f = super::frame::Frame4::new(
                acc.as_ref(), &self.model, &self.seqs, frame_t_max(Some(acc.as_ref())),
            )?;
            self.frame = Some(f);
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
        let ple0: Vec<_> = seqs.iter().map(|&s| super::frame::ple_snap(&self.seqs[s])).collect();
        let ctx = Ctx { model: &self.model, acc: Some(acc.as_ref()) };
        let toks = match super::frame::frame_forward_prefill_multi(
            acc.as_ref(), &self.model, &ctx, seqs, &mut self.seqs, f, tokens, per_seq,
        ) {
            Ok(t) => t,
            Err(e) => {
                for (snap, &sq) in ple0.into_iter().zip(seqs.iter()) {
                    super::frame::ple_restore(&mut self.seqs[sq], snap);
                }
                // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                if let Some(a) = self.acc.as_deref() { a.frame_sync(); }
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
    pub fn decode_batch(&mut self, seqs: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, Q4Error> {
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
        let acc = self.acc.as_deref().unwrap();
        if self.frame.is_none() {
            match super::frame::Frame4::new(acc, &self.model, &self.seqs, frame_t_max(Some(acc))) {
                Ok(f) => self.frame = Some(f),
                Err(e) => {
                    self.frame_broken = true;
                    eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
                }
            }
        }
        // plans/86 §3 — 트랜잭션 스냅샷(전 시퀀스).
        let ple0: Vec<_> = seqs.iter().map(|&s| super::frame::ple_snap(&self.seqs[s])).collect();
        let r = (|| -> Result<Vec<Vec<f32>>, Q4Error> {
            let f = self.frame.as_mut().ok_or_else(|| Q4Error::Io("frame 없음".into()))?;
            let acc = self.acc.as_deref().unwrap();
            for &s in seqs {
                if f.dirty[s] {
                    f.sync_states(acc, s, &self.seqs[s], self.model.hp.d_state)?;
                }
            }
            let ctx = Ctx { model: &self.model, acc: Some(acc) };
            super::frame::frame_forward_np(
                acc, &self.model, &ctx, seqs, &mut self.seqs, f, tokens,
            )
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
                if let Some(a) = self.acc.as_deref() { a.frame_sync(); }
            }
                static ONCE: std::sync::Once = std::sync::Once::new();
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
pub fn decode_batch_greedy(&mut self, seqs: &[usize], tokens: &[u32]) -> Result<Vec<u32>, Q4Error> {
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
        && std::env::var("LLM170_FRAME_DECODE").map(|v| v != "0").unwrap_or(true)
        && std::env::var("LLM170_NP_GREEDY").map(|v| v != "0").unwrap_or(true);
    if !frame_on {
        let lg = self.decode_batch(seqs, tokens)?;
        return Ok(lg.iter().map(|l| crate::qwen35::greedy(l)).collect());
    }
    if let Some(h) = self.ple_worker.take() {
        let _ = h.join();
    }
    self.ple_next = None;
    self.ple_consume = None;
    let acc = self.acc.as_deref().unwrap();
    if self.frame.is_none() {
        match super::frame::Frame4::new(acc, &self.model, &self.seqs, frame_t_max(Some(acc))) {
            Ok(f) => self.frame = Some(f),
            Err(e) => {
                self.frame_broken = true;
                eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
            }
        }
    }
    // plans/86 §3 — 트랜잭션 스냅샷(전 시퀀스).
    let ple0: Vec<_> = seqs.iter().map(|&s| super::frame::ple_snap(&self.seqs[s])).collect();
    let r = (|| -> Result<Vec<u32>, Q4Error> {
        let f = self.frame.as_mut().ok_or_else(|| Q4Error::Io("frame 없음".into()))?;
        let acc = self.acc.as_deref().unwrap();
        for &s in seqs {
            if f.dirty[s] {
                f.sync_states(acc, s, &self.seqs[s], self.model.hp.d_state)?;
            }
        }
        let ctx = Ctx { model: &self.model, acc: Some(acc) };
        let r2 = super::frame::frame_forward_np_greedy(
            acc, &self.model, &ctx, seqs, &mut self.seqs, f, tokens,
        );
        if std::env::var_os("LLM170_KTRACE").is_some() {
            acc.ktrace_tick();
        }
        r2
    })();
    if std::env::var_os("LLM170_NP_TIME").is_some() {
        eprintln!("[npstep4] t={} {:.1}ms", seqs.len(), npg_t0.elapsed().as_secs_f64() * 1e3);
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
                if let Some(a) = self.acc.as_deref() { a.frame_sync(); }
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
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
                && g.token == token && !g.emb.is_empty() {
                    self.ple_consume = Some(vec![std::mem::take(&mut g.emb)]);
                }
        let acc = self.acc.as_deref().unwrap();
        if self.frame.is_none() {
            match super::frame::Frame4::new(acc, &self.model, &self.seqs, frame_t_max(Some(acc))) {
                Ok(f) => self.frame = Some(f),
                Err(e) => {
                    self.frame_broken = true;
                    eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
                    let l = self.decode1(seq, token)?;
                    return Ok(crate::qwen35::greedy(&l));
                }
            }
        }
        // plans/86 §3 — 트랜잭션 스냅샷(폴백 시 복원).
        let ple0 = super::frame::ple_snap(&self.seqs[seq]);
        let r = (|| -> Result<u32, Q4Error> {
            let f = self.frame.as_mut().ok_or_else(|| Q4Error::Io("frame 없음".into()))?;
            let acc = self.acc.as_deref().unwrap();
            if f.dirty[seq] {
                f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
            }
            let ctx = Ctx { model: &self.model, acc: Some(acc) };
            // 그래프 캡처(LLM170_GRAPH=1) — value 경로와 동일 규약: 스텝0 = 워밍,
            // 스텝1 = 캡처(결과 폐기 후 같은 토큰 즉시 재생), 스텝2+ = 재생.
            // 프레임 디코드는 커널 수가 많아(≈1000) 런치 간극이 스텝의 12-18%다 —
            // 그래프로 굳히면 그 간극이 사라진다(2026-09-17, value 경로 실측 +5.1%).
            let cap = self.graph_want;
            let cap_step = cap && self.graph_step == 1;
            let rep_step = cap && self.graph_step >= 2;
            if cap_step {
                if let Err(e) = acc.graph_capture_begin() {
                    eprintln!("# graph(frame): 캡처 시작 실패 — 정상 경로 ({e})");
                    self.graph_want = false;
                }
            } else if rep_step
                && let Err(e) = acc.graph_replay(true) {
                    eprintln!("# graph(frame): 재생 실패 — 정상 경로 ({e})");
                    self.graph_want = false;
                }
            let r0 = super::frame::decode_frame_greedy(
                acc, &self.model, &ctx, seq, &mut self.seqs[seq], f, token,
            );
            let r3 = if cap_step {
                match acc.graph_capture_end() {
                    Err(e) => {
                        eprintln!("# graph(frame): 캡처 실패 — 정상 경로 유지 ({e})");
                        self.graph_want = false;
                        r0
                    }
                    Ok(()) => match acc.graph_replay(true) {
                        Err(e) => {
                            eprintln!("# graph(frame): 재생 실패 — 정상 경로 ({e})");
                            self.graph_want = false;
                            r0
                        }
                        Ok(()) => {
                            let r1 = super::frame::decode_frame_greedy(
                                acc, &self.model, &ctx, seq, &mut self.seqs[seq], f, token,
                            );
                            let _ = acc.graph_replay(false);
                            self.graph_step += 1;
                            r1
                        }
                    },
                }
            } else {
                if rep_step {
                    let _ = acc.graph_replay(false);
                    self.graph_step += 1;
                }
                r0
            };
            if std::env::var_os("LLM170_KTRACE").is_some() {
                acc.ktrace_tick();
            }
            r3
        })();
        match r {
            Ok(tok) => {
                self.seqs[seq].pos += 1;
                Ok(tok)
            }
            Err(e) => {
                // 그래프 상태가 걸린 채 value 경로로 넘어가면 백엔드가 Replay 모드로
                // 남아 런치를 건너뛴다 — 폴백 시 그래프를 먼저 중단한다.
                if self.graph_step > 0
                    && let Some(a) = self.acc.as_deref() { a.graph_abort() }
                self.graph_want = false;
                self.frame = None;
                self.frame_broken = true;
                super::frame::ple_restore(&mut self.seqs[seq], ple0);
                // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                if let Some(a) = self.acc.as_deref() { a.frame_sync(); }
                static ONCE: std::sync::Once = std::sync::Once::new();
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
                && g.token == token && !g.emb.is_empty() {
                    self.ple_consume = Some(vec![std::mem::take(&mut g.emb)]);
                }
        // 프레임 기본 ON(2026-09-02) — 상주 불가 시 1회 재시도 후 value 경로로
        // 영구 폴백. 게이트 실패는 mm 오류(호스트 폴백 가중치)로 첫 스텝 초반에
        // 발생해 상태 오염 전에 중단된다.
        let frame_on = self.frame_on(true);
        let frame_try = if frame_on {
            let acc = self.acc.as_deref().unwrap();
            if self.frame.is_none() {
                match super::frame::Frame4::new(acc, &self.model, &self.seqs, frame_t_max(Some(acc))) {
                    Ok(f) => {
                        self.frame = Some(f);
                        Some(())
                    }
                    Err(e) => {
                        self.frame_broken = true;
                        eprintln!("# frame: 생성 실패 — value 경로 폴백 ({e})");
                        None
                    }
                }
            } else {
                Some(())
            }
        } else {
            None
        };
        let logits = if let (true, Some(())) = (frame_on, frame_try.as_ref().filter(|_| self.frame.is_some()).map(|_| ())) {
            let acc = self.acc.as_deref().unwrap();
            let f = self.frame.as_mut().unwrap();
            // 그래프 캡처(LLM170_GRAPH=1): 스텝1 = 캡처, 스텝2+ = 재생.
            // 캡처 중에는 커널이 *기록만* 되고 실행되지 않으므로(상태 미진행,
            // 호스트 판독은 직전 값) 캡처 스텝은 결과를 버리고 같은 토큰으로
            // 즉시 재생해 실제 진행·정답 로짓을 얻는다.
            let cap = self.graph_want;
            let cap_step = cap && self.graph_step == 1;
            let rep_step = cap && self.graph_step >= 2;
            if cap_step {
                if let Err(e) = acc.graph_capture_begin() {
                    eprintln!("# graph: 캡처 시작 실패 — 정상 경로 ({e})");
                    self.graph_want = false;
                }
            } else if rep_step
                && let Err(e) = acc.graph_replay(true) {
                    eprintln!("# graph: 재생 실패 — 정상 경로 ({e})");
                    self.graph_want = false;
                }
            // plans/86 §3 — 트랜잭션: 시도 전 PLE 상태 스냅샷, Err 시 복원 후
            // 값경로 폴백(이중 진화 방지).
            let ple0 = super::frame::ple_snap(&self.seqs[seq]);
            let mut run_step = || -> Result<Vec<f32>, Q4Error> {
                if f.dirty[seq] {
                    f.sync_states(acc, seq, &self.seqs[seq], self.model.hp.d_state)?;
                }
                let ctx = Ctx { model: &self.model, acc: Some(acc) };
                super::frame::decode_frame(
                    acc, &self.model, &ctx, seq, &mut self.seqs[seq], f, token,
                )
            };
            let r0 = run_step();
            let r = if cap_step {
                let close = acc.graph_capture_end();
                match close {
                    Err(e) => {
                        eprintln!("# graph: 캡처 실패 — 정상 경로 유지 ({e})");
                        self.graph_want = false;
                        r0
                    }
                    Ok(()) => {
                        if let Err(e) = acc.graph_replay(true) {
                            eprintln!("# graph: 재생 실패 — 정상 경로 ({e})");
                            self.graph_want = false;
                            r0
                        } else {
                            let r1 = run_step();
                            let _ = acc.graph_replay(false);
                            self.graph_step += 1;
                            r1
                        }
                    }
                }
            } else {
                if rep_step {
                    let _ = acc.graph_replay(false);
                    self.graph_step += 1;
                }
                r0
            };
            match r {
                Ok(l) => l,
                Err(e) => {
                    self.frame = None;
                    self.frame_broken = true;
                    super::frame::ple_restore(&mut self.seqs[seq], ple0);
                    // plans/88 P1 — 스텝 배치 잔류 플러시(녹화 커맨드 실행 보장).
                    if let Some(a) = self.acc.as_deref() { a.frame_sync(); }
                    eprintln!("# frame: 디코드 실패 — value 경로 폴백 ({e})");
                    let mut tm = init_timings();
                    self.forward_timed(seq, &[token], tm.as_mut())?
                }
            }
        } else {
            let mut tm = init_timings();
            let l = self.forward_timed(seq, &[token], tm.as_mut())?;
            if let Some(t) = &tm {
                t.report("decode1");
            }
            l
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
            let rows = pure_hash(&hist, hist_valid, &[next_tok], hp_ngram, hpng, &mult, &offs, &vs, eos);
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

/// 프리페치 워커용 순수 n-gram 해시 — ple_hash와 동일 수식 (파라미터만 전달).
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
    let heads = hpng * 2;
    let mut hist: Vec<u32> = if hist_valid { hist.to_vec() } else { vec![eos; ngram - 1] };
    let mut rows = Vec::with_capacity(tokens.len() * heads);
    for (i, &tok) in tokens.iter().enumerate() {
        let mut ctx = vec![tok as u64; ngram];
        let mut cut = false;
        for s in 1..ngram {
            let j = i as i64 - s as i64;
            let prev: u64 = if j >= 0 {
                tokens[j as usize] as u64
            } else {
                let back = s as i64 - i as i64;
                let k = hist.len() as i64 - back;
                if k >= 0 && (k as usize) < hist.len() {
                    hist[k as usize] as u64
                } else {
                    eos as u64
                }
            };
            ctx[s] = if cut { eos as u64 } else { prev };
            if ctx[s] == eos as u64 {
                cut = true;
            }
        }
        for n in 2..=ngram {
            let mut mixed = ctx[0].wrapping_mul(mult[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(mult[j]);
            }
            let base = (n - 2) * hpng;
            for g in 0..hpng {
                let h = base + g;
                rows.push((mixed % vs[h] + offs[h]) as u32);
            }
        }
        hist.push(tok);
        if hist.len() > ngram - 1 {
            let cutn = hist.len() - (ngram - 1);
            hist.drain(..cutn);
        }
    }
    rows
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

fn init_timings() -> Option<Q4Timings> {
    if std::env::var_os("LLM170_Q4_TIME").is_some() {
        Some(Q4Timings::default())
    } else {
        None
    }
}

#[cfg(test)]
mod forward_tests {
    use super::*;

    const MODEL: &str = "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

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
        assert!((l1[0] - l1b[0]).abs() < 1e-6, "logit 비결정성: {} vs {}", l1[0], l1b[0]);
    }
}
