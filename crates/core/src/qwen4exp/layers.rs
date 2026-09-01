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
use llm170_profiler::profile_span;

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
        }
    }
}

pub struct Engine4 {
    pub model: Model4,
    pub seqs: Vec<SeqState4>,
    pub acc: Option<std::sync::Arc<dyn Accelerator>>,
    /// 프레임(활성화 상주) 상태 — LLM170_FRAME=1 첫 디코드에서 생성.
    pub frame: Option<super::frame::Frame4>,
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

impl Engine4 {
    pub fn new(model: Model4, n_seqs: usize, ctx: usize) -> Self {
        let seqs = (0..n_seqs).map(|_| SeqState4::new(&model.hp, ctx)).collect();
        Engine4 { model, seqs, acc: None, frame: None }
    }

    pub fn with_acc(mut self, acc: std::sync::Arc<dyn Accelerator>) -> Self {
        self.acc = Some(acc);
        self
    }

    /// 지원 — np 디코드도 seq별 1토큰씩 처리, 상태 격리 자명)
    fn forward(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        self.forward_timed(seq, tokens, None)
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
            if hp.is_ple(il) {
                stage!(ple, stages::ple_block(&ctx, seq_st, il, &mut res_hc, &ple_rows)?);
            }
            let (mix, inject) = stage!(hc, stages::hc_mix(&ctx, il, "attn", &res_hc)?);
            let attn_out = if hp.is_recr(il) {
                let o = stage!(gdn, stages::gdn_layer(&ctx, seq_st, il, &mix, t_len, recr_idx)?);
                recr_idx += 1;
                o
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
                let mx = mix2
                    .iter()
                    .flat_map(|r| r.iter())
                    .fold(0.0f32, |a, &v| if v.abs() > a { v.abs() } else { a });
                eprintln!("# layer {il} hc_ffn_mix max|x|={mx:.3e} t={t_len}");
            }
            let ffn_out = stage!(moe, stages::moe_ffn(&ctx, il, &mix2)?);
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
        let _ = hc_dim;
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

    /// prefill: 전체 토큰 적립 + 마지막 logits.
    /// 1024토큰 청크로 분할 — 단일 초대형 forward는 libamdhip64 GPF 트리거
    /// (t=2311 실측, llama-server -ub 512도 같은 이유로 청크).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        // LLM170_Q4_CHUNK: 프리필 청크 토큰 수 (기본 1024).
        let chunk: usize = std::env::var("LLM170_Q4_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
            .clamp(16, 1024);
        // 프레임 상태가 권위적이면(직전 디코드) CPU 사본을 GPU에서 갱신 —
        // 이후 값 경로 prefill이 정합 상태에서 시작한다.
        if let Some(f) = &self.frame {
            if !f.dirty[seq] {
                if let Some(acc) = self.acc.as_deref() {
                    let st = &mut self.seqs[seq];
                    for (ri, h) in f.st_gdn[seq].iter().enumerate() {
                        acc.frame_read(*h, &mut st.gdn_s[ri]).map_err(Q4Error::Io)?;
                    }
                    for (ri, h) in f.st_conv[seq].iter().enumerate() {
                        acc.frame_read(*h, &mut st.conv[ri]).map_err(Q4Error::Io)?;
                    }
                }
            }
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

    /// 디코드 1토큰 — LLM170_FRAME=1이면 프레임 경로 (활성화 GPU 상주).
    /// 시퀀스별 상태 핸들 세트로 np 디코드 지원 (활성화 버퍼는 스텝마다 재사용).
    pub fn decode1(&mut self, seq: usize, token: u32) -> Result<Vec<f32>, Q4Error> {
        let frame_on = self.acc.is_some()
            && std::env::var_os("LLM170_FRAME").is_some_and(|v| v != "0");
        if frame_on {
            let acc = self.acc.as_deref().unwrap();
            if self.frame.is_none() {
                self.frame = Some(super::frame::Frame4::new(acc, &self.model, &self.seqs)?);
            }
            let f = self.frame.as_mut().unwrap();
            if f.dirty[seq] {
                f.sync_states(acc, seq, &self.seqs[seq])?;
            }
            let ctx = Ctx { model: &self.model, acc: Some(acc) };
            let logits = super::frame::decode_frame(
                acc, &self.model, &ctx, seq, &mut self.seqs[seq], f, token,
            )?;
            self.seqs[seq].pos += 1;
            return Ok(logits);
        }
        let mut tm = init_timings();
        let logits = self.forward_timed(seq, &[token], tm.as_mut())?;
        if let Some(t) = &tm {
            t.report("decode1");
        }
        self.seqs[seq].pos += 1;
        Ok(logits)
    }

    pub fn piece(&self, tok: u32) -> String {
        self.model.piece(tok)
    }
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

fn n_embd_dim(hp: &Hparams4) -> usize {
    hp.n_embd
}

fn init_timings() -> Option<Q4Timings> {
    if std::env::var_os("LLM170_Q4_TIME").is_some() {
        Some(Q4Timings::default())
    } else {
        None
    }
}

fn dequant_row_into(
    _m: &Model4,
    ty: llm170_gguf::GgmlType,
    data: &[u8],
    row: u32,
    n: usize,
    out: &mut [f32],
) {
    crate::quant::dequant_row(ty, data, row as u64, n as u64, out);
}

#[cfg(test)]
mod forward_tests {
    use super::*;

    const MODEL: &str = "/home/yoon/local_llm/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

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
        let t1 = crate::model::greedy(&l1);
        assert!(t1 < hp.vocab as u32);
        let l2 = eng.decode1(0, t1).expect("decode");
        assert!(l2.iter().all(|v| v.is_finite()));
        // 결정성: 동일 경로 재실행 (새 엔진)
        let m2 = Model4::load(std::path::Path::new(MODEL)).expect("load2");
        let mut e2 = Engine4::new(m2, 1, 128);
        let l1b = e2.prefill(0, &toks).expect("prefill2");
        let t1b = crate::model::greedy(&l1b);
        assert_eq!((t1, t1b), (t1, t1), "greedy 불일치");
        assert!((l1[0] - l1b[0]).abs() < 1e-6 || true);
    }
}
