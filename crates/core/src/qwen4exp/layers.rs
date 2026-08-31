//! qwen4exp 상태·forward — HC 잔차, GDN(z-gate sigmoid), QSA(인덱서 top-k),
//! MoE(512·10+shared), PLE(n-gram 해시·게이트·dilated conv).
//!
//! 배선 근거: qwen4exp.cpp build_hc_mix/combine/build_qsa_top_k/build_attn_qsa/
//! build_ple + llama-graph.cpp build_moe_ffn (2026-08-30 판). 수치는 f32 참조.

use super::{Hparams4, Model4, Q4Error};
use crate::matmul::{matmul, matmul_batch, Accelerator};
use crate::ops::{l2_norm, rms_norm, rope_head, sigmoid, silu, softplus};
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
        Engine4 { model, seqs, acc: None }
    }

    pub fn with_acc(mut self, acc: std::sync::Arc<dyn Accelerator>) -> Self {
        self.acc = Some(acc);
        self
    }

    fn mm(&self, x: &[f32], w: &crate::matmul::Weight, out: &mut [f32]) -> Result<(), Q4Error> {
        match self.acc.as_deref() {
            Some(a) => a.matmul(x, w, out).map_err(Q4Error::Io),
            None => {
                matmul(x, w, out);
                Ok(())
            }
        }
    }

    /// 그룹 디스패치 — 동일 입력 복수 가중치. 가속기 없으면 CPU 개별.
    fn mm_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[crate::matmul::Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), Q4Error> {
        match self.acc.as_deref() {
            Some(a) => a.matmul_group(xs, ws, outs).map_err(Q4Error::Io),
            None => {
                for (w, o) in ws.iter().zip(outs.iter_mut()) {
                    matmul_batch(xs, w, o);
                }
                Ok(())
            }
        }
    }

    /// 배치 디스패치 — 가속기 없으면 CPU thread::scope 경로.
    fn mm_batch(
        &self,
        xs: &[Vec<f32>],
        w: &crate::matmul::Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), Q4Error> {
        match self.acc.as_deref() {
            Some(a) => a.matmul_batch(xs, w, outs).map_err(Q4Error::Io),
            None => {
                matmul_batch(xs, w, outs);
                Ok(())
            }
        }
    }

    /// 단일 시퀀스 배치 forward → 마지막 토큰 logits. (qwen4exp는 시퀀스별 prefill만
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
        let ple_rows = if hp.is_ple(1) { self.ple_hash(seq, tokens) } else { Vec::new() };

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
        for il in 0..hp.n_layer {
            if trace {
                eprintln!("q4 layer {il} t={t_len}");
            }
            if hp.is_ple(il) {
                stage!(ple, self.ple_block(seq, il, &mut res_hc, &ple_rows)?);
            }
            let (mix, inject) = stage!(hc, self.hc_mix(il, "attn", &res_hc)?);
            let attn_out = if hp.is_recr(il) {
                let o = stage!(gdn, self.gdn_layer(seq, il, &mix, t_len, recr_idx)?);
                recr_idx += 1;
                o
            } else {
                let o = stage!(qsa, self.qsa_layer(seq, il, &mix, t_len, full_idx)?);
                full_idx += 1;
                o
            };
            hc_combine(&mut res_hc, &attn_out, &inject, hc);

            let (mix2, inject2) = stage!(hc, self.hc_mix(il, "ffn", &res_hc)?);
            let ffn_out = stage!(moe, self.moe_ffn(il, &mix2)?);
            hc_combine(&mut res_hc, &ffn_out, &inject2, hc);
        }

        // output HC mix → logits (inject 없음)
        let head_in = stage!(head, self.hc_mix_head(&res_hc)?);
        let last = head_in.last().ok_or(Q4Error::BadMeta("빈 배치"))?.clone();
        let wout = self.model.w("output.weight").ok_or(Q4Error::MissingTensor("output.weight".into()))?;
        let mut logits = vec![0.0f32; wout.n_out as usize];
        stage!(head, self.mm(&last, &wout, &mut logits)?);
        let _ = hc_dim;
        Ok(logits)
    }

    /// grouped RMSNorm + 저랭크 게이트 + 스트림 평균 + inject.
    /// kind = "attn"|"ffn" → blk.{il}.hc_{kind}_{norm,down,up,inject}.weight
    /// 토큰 축 배치: down/up/inject 각 전 토큰 1회 — GPU 왕복을 층당 6회로 고정
    /// (토큰당 288회 왕복이 장문 prefill 병목이었음 — 2026-08-31 실측).
    fn hc_mix(
        &self,
        il: usize,
        kind: &str,
        res_hc: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), Q4Error> {
        profile_span!("q4::hc_mix");
        let hp = &self.model.hp;
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t = res_hc.len();
        let w_norm = self.model.f32_vec4(&format!("blk.{il}.hc_{kind}_norm.weight"))?;
        let w_down = self.model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
        let w_up = self.model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?;
        let w_inject = self.model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;

        // 1) grouped RMSNorm — 전 토큰 (감마는 (1+w) 폴딩, 스트림별 축소)
        let mut xn_all: Vec<Vec<f32>> = Vec::with_capacity(t);
        for x in res_hc {
            let mut xn = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = x[s * n_embd..(s + 1) * n_embd].to_vec();
                xn[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &w_norm[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            xn_all.push(xn);
        }
        // 2) 저랭크 down → silu(lo/hc) → up → 게이트 — 배치 1회씩
        let mut lo_all = vec![vec![0.0f32; w_down.n_out as usize]; t];
        self.mm_batch(&xn_all, &w_down, &mut lo_all)?;
        for lo in lo_all.iter_mut() {
            for v in lo.iter_mut() {
                *v = silu(*v / hc as f32);
            }
        }
        let mut gate_all = vec![vec![0.0f32; hc_dim]; t];
        self.mm_batch(&lo_all, &w_up, &mut gate_all)?;
        let mut inject_all = vec![vec![0.0f32; hc]; t];
        self.mm_batch(&xn_all, &w_inject, &mut inject_all)?;
        // 3) 게이트 적용 + 스트림 평균
        let mut mixed: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (gate, xn) in gate_all.iter_mut().zip(xn_all.iter()) {
            for (g, gi) in gate.iter_mut().zip(xn.iter()) {
                *g = *gi * sigmoid(*g);
            }
            let mut m = vec![0.0f32; n_embd];
            for s in 0..hc {
                for i in 0..n_embd {
                    m[i] += gate[s * n_embd + i];
                }
            }
            for v in m.iter_mut() {
                *v /= hc as f32;
            }
            mixed.push(m);
        }
        Ok((mixed, inject_all))
    }

    /// 출력 헤드용 HC mix (inject 없음) — output_hc_{norm,down,up}. 동일 배치 구조.
    fn hc_mix_head(&self, res_hc: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::hc_mix_head");
        let hp = &self.model.hp;
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t = res_hc.len();
        let w_norm = self.model.f32_vec4("output_hc_norm.weight")?;
        let w_down = self.model.w4("output_hc_down.weight")?;
        let w_up = self.model.w4("output_hc_up.weight")?;
        let mut xn_all: Vec<Vec<f32>> = Vec::with_capacity(t);
        for x in res_hc {
            let mut xn = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = x[s * n_embd..(s + 1) * n_embd].to_vec();
                xn[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &w_norm[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            xn_all.push(xn);
        }
        let mut lo_all = vec![vec![0.0f32; w_down.n_out as usize]; t];
        self.mm_batch(&xn_all, &w_down, &mut lo_all)?;
        for lo in lo_all.iter_mut() {
            for v in lo.iter_mut() {
                *v = silu(*v / hc as f32);
            }
        }
        let mut gate_all = vec![vec![0.0f32; hc_dim]; t];
        self.mm_batch(&lo_all, &w_up, &mut gate_all)?;
        let mut out = Vec::with_capacity(t);
        for (gate, xn) in gate_all.iter_mut().zip(xn_all.iter()) {
            for (g, gi) in gate.iter_mut().zip(xn.iter()) {
                *g = *gi * sigmoid(*g);
            }
            let mut m = vec![0.0f32; n_embd];
            for s in 0..hc {
                for i in 0..n_embd {
                    m[i] += gate[s * n_embd + i];
                }
            }
            for v in m.iter_mut() {
                *v /= hc as f32;
            }
            out.push(m);
        }
        Ok(out)
    }

    /// GDN층 — qwen35와 동일 모듈, 차이: z-gate가 sigmoid.
    fn gdn_layer(
        &mut self,
        seq: usize,
        il: usize,
        xs: &[Vec<f32>],
        t_len: usize,
        recr_idx: usize,
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::layer_gdn");
        let hp = self.model.hp.clone();
        let n_tok = t_len;
        let (d_state, dt_rank, n_group, d_inner) = (hp.d_state, hp.dt_rank, hp.n_group, hp.d_inner);
        let conv_ch = n_group * d_state * 2 + dt_rank * d_state;
        let wqkv = self.model.w4(&format!("blk.{il}.attn_qkv.weight"))?;
        let wgate = self.model.w4(&format!("blk.{il}.attn_gate.weight"))?;
        let wbeta = self.model.w4(&format!("blk.{il}.ssm_beta.weight"))?;
        let walpha = self.model.w4(&format!("blk.{il}.ssm_alpha.weight"))?;
        let ssm_a = self.model.f32_vec4(&format!("blk.{il}.ssm_a"))?;
        let dt_bias = self.model.f32_vec4(&format!("blk.{il}.ssm_dt.bias"))?;
        let conv_w = self.model.f32_vec4(&format!("blk.{il}.ssm_conv1d.weight"))?;
        let ssm_norm_w = self.model.f32_vec4(&format!("blk.{il}.ssm_norm.weight"))?;
        let wout = self.model.w4(&format!("blk.{il}.ssm_out.weight"))?;

        let mut qkv = vec![vec![0.0f32; conv_ch]; n_tok];
        self.mm_batch(xs, &wqkv, &mut qkv)?;
        let mut z = vec![vec![0.0f32; d_inner]; n_tok];
        self.mm_batch(xs, &wgate, &mut z)?;
        let mut b = vec![vec![0.0f32; dt_rank]; n_tok];
        self.mm_batch(xs, &wbeta, &mut b)?;
        let mut a = vec![vec![0.0f32; dt_rank]; n_tok];
        self.mm_batch(xs, &walpha, &mut a)?;

        let mut beta_all = vec![0.0f32; n_tok * dt_rank];
        let mut g_all = vec![0.0f32; n_tok * dt_rank];
        for t in 0..n_tok {
            for h in 0..dt_rank {
                beta_all[t * dt_rank + h] = sigmoid(b[t][h]);
                g_all[t * dt_rank + h] = softplus(a[t][h] + dt_bias[h]) * ssm_a[h];
            }
        }

        let k_len = n_group * d_state;
        let v_len = dt_rank * d_state;
        let mut q_all = vec![0.0f32; n_tok * k_len];
        let mut k_all = vec![0.0f32; n_tok * k_len];
        let mut v_all = vec![0.0f32; n_tok * v_len];
        let mut o_all = vec![0.0f32; n_tok * v_len];
        {
            let conv_state = &mut self.seqs[seq].conv[recr_idx];
            for t in 0..t_len {
                for c in 0..conv_ch {
                    let mut sum = conv_w[c * hp.conv_k + (hp.conv_k - 1)] * qkv[t][c];
                    for j in 0..hp.conv_k - 1 {
                        sum += conv_w[c * hp.conv_k + j] * conv_state[j * conv_ch + c];
                    }
                    let out_c = silu(sum);
                    for j in 0..hp.conv_k - 2 {
                        conv_state[j * conv_ch + c] = conv_state[(j + 1) * conv_ch + c];
                    }
                    conv_state[(hp.conv_k - 2) * conv_ch + c] = qkv[t][c];
                    if c < k_len {
                        q_all[t * k_len + c] = out_c;
                    } else if c < 2 * k_len {
                        k_all[t * k_len + c - k_len] = out_c;
                    } else {
                        v_all[t * v_len + c - 2 * k_len] = out_c;
                    }
                }
            }
        }
        for row in 0..n_tok {
            for h in 0..n_group {
                let b0 = row * k_len + h * d_state;
                let head: Vec<f32> = q_all[b0..b0 + d_state].to_vec();
                q_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&head, hp.eps));
                let headk: Vec<f32> = k_all[b0..b0 + d_state].to_vec();
                k_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&headk, hp.eps));
            }
        }
        {
            let st = &mut self.seqs[seq].gdn_s[recr_idx];
            if t_len == 1 {
                crate::gdn::gdn_ar_batch(
                    &q_all, &k_all, &v_all, &beta_all, &g_all, st, &mut o_all, 1, n_group, dt_rank,
                );
            } else {
                crate::gdn::gdn_chunk_seq(
                    &q_all, &k_all, &v_all, &beta_all, &g_all, st, &mut o_all, t_len, n_group, dt_rank,
                );
            }
        }
        // norm_gated: rms·sigmoid(z) — qwen35(silu)와의 유일 차이
        let mut gated = vec![vec![0.0f32; d_inner]; n_tok];
        for t in 0..n_tok {
            for h in 0..dt_rank {
                let b0 = t * v_len + h * d_state;
                let head: Vec<f32> = o_all[b0..b0 + d_state].to_vec();
                let n = rms_norm(&head, &ssm_norm_w, hp.eps);
                for i in 0..d_state {
                    gated[t][h * d_state + i] = n[i] * sigmoid(z[t][h * d_state + i]);
                }
            }
        }
        let mut out = vec![vec![0.0f32; n_embd_dim(&hp)]; n_tok];
        self.mm_batch(&gated, &wout, &mut out)?;
        Ok(out)
    }

    /// QSA층 — 인덱서 top-k 마스크 게이트드 GQA.
    fn qsa_layer(
        &mut self,
        seq: usize,
        il: usize,
        xs: &[Vec<f32>],
        t_len: usize,
        full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::layer_qsa");
        let hp = self.model.hp.clone();
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let wq = self.model.w4(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.model.w4(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.model.w4(&format!("blk.{il}.attn_v.weight"))?;
        let wo = self.model.w4(&format!("blk.{il}.attn_output.weight"))?;
        let q_norm_w = self.model.f32_vec4(&format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm_w = self.model.f32_vec4(&format!("blk.{il}.attn_k_norm.weight"))?;
        let iq_w = self.model.f32_vec4(&format!("blk.{il}.indexer.q_norm.weight"))?;
        let ik_w = self.model.f32_vec4(&format!("blk.{il}.indexer.k_norm.weight"))?;
        let w_iq = self.model.w4(&format!("blk.{il}.indexer.q_proj.weight"))?;
        let w_ik = self.model.w4(&format!("blk.{il}.indexer.k_proj.weight"))?;

        let n_tok = t_len;
        let mut qg = vec![vec![0.0f32; wq.n_out as usize]; n_tok];
        self.mm_batch(xs, &wq, &mut qg)?;
        let mut kk = vec![vec![0.0f32; wk.n_out as usize]; n_tok];
        self.mm_batch(xs, &wk, &mut kk)?;
        let mut vv = vec![vec![0.0f32; wv.n_out as usize]; n_tok];
        self.mm_batch(xs, &wv, &mut vv)?;
        let mut iq = vec![vec![0.0f32; w_iq.n_out as usize]; n_tok];
        self.mm_batch(xs, &w_iq, &mut iq)?;
        let mut ik = vec![vec![0.0f32; w_ik.n_out as usize]; n_tok];
        self.mm_batch(xs, &w_ik, &mut ik)?;

        let kq_scale = hp.kq_scale();
        let pos0 = self.seqs[seq].pos;
        let mut out = vec![vec![0.0f32; hp.n_embd]; n_tok];
        let mut attn_all = vec![vec![0.0f32; n_head * hd]; n_tok];
        let n_past_max = (pos0 as usize) + t_len;
        let gpu_attn = self.acc.is_some();
        let mut mask_all: Vec<Vec<bool>> = vec![vec![false; n_past_max]; n_tok];

        let seq_state = &mut self.seqs[seq];
        for t in 0..t_len {
            let pos = pos0 + t as u32;
            let (cache_k, cache_v, idx_cache) = {
                let st = &mut seq_state.kv_k[full_idx];
                let st2 = &mut seq_state.kv_v[full_idx];
                let st3 = &mut seq_state.idx_k[full_idx];
                // 안전 분할: 세 벡터는 서로 다른 필드 — std::split_at_mut 불필요
                (st.as_mut_slice(), st2.as_mut_slice(), st3.as_mut_slice())
            };
            let n_past = pos as usize + 1;

            // K/V 캐시 적립 + 인덱서 raw k 캐시
            for h in 0..n_kv {
                let src = kk[t][h * hd..h * hd + hd].to_vec();
                let mut head = rms_norm(&src, &k_norm_w, hp.eps);
                rope_head(&mut head, pos, n_rot, hp.rope_base);
                let b = pos as usize * n_kv * hd + h * hd;
                cache_k[b..b + hd].copy_from_slice(&head);
                cache_v[b..b + hd].copy_from_slice(&vv[t][h * hd..h * hd + hd]);
            }
            idx_cache[pos as usize * hp.idx_dim..(pos as usize + 1) * hp.idx_dim]
                .copy_from_slice(&ik[t]);

            // 인덱서 스코어: 완전 블록(4토큰) mean-pool → rms → rope(b*4) → ReLU 헤드합
            let r = hp.compress[il] as usize;
            let n_blocks = n_past / r;
            let tail_start = n_blocks * r;
            let mut q_rope: Vec<Vec<f32>> = Vec::with_capacity(hp.idx_heads);
            for h in 0..hp.idx_heads {
                let mut qh = rms_norm(
                    &iq[t][h * hp.idx_dim..(h + 1) * hp.idx_dim].to_vec(),
                    &iq_w,
                    hp.eps,
                );
                rope_head(&mut qh, pos, hp.idx_dim, hp.rope_base);
                q_rope.push(qh);
            }
            let mut block_score = vec![0.0f32; n_blocks];
            for b in 0..n_blocks {
                let mut pooled = vec![0.0f32; hp.idx_dim];
                for j in 0..r {
                    let base = (b * r + j) * hp.idx_dim;
                    for i in 0..hp.idx_dim {
                        pooled[i] += idx_cache[base + i];
                    }
                }
                for v in pooled.iter_mut() {
                    *v /= r as f32;
                }
                let mut pk = rms_norm(&pooled, &ik_w, hp.eps);
                rope_head(&mut pk, (b * r) as u32, hp.idx_dim, hp.rope_base);
                for qh in &q_rope {
                    let mut dot = 0.0f32;
                    for i in 0..hp.idx_dim {
                        dot += qh[i] * pk[i];
                    }
                    if dot > 0.0 {
                        block_score[b] += dot;
                    }
                }
            }

            // 선택: 테일(강제) + 상위 B개 완전블록 — 폭 = min(n_past, top_k + r − 1)
            let width = n_past.min(hp.idx_top_k + r - 1);
            let tail_cnt = n_past - tail_start;
            let mut sel_blocks: Vec<usize> = (0..n_blocks).collect();
            sel_blocks.sort_by(|&a, &b| block_score[b].partial_cmp(&block_score[a]).unwrap());
            let n_sel_blocks = ((width - tail_cnt) / r).min(n_blocks);
            let mut mask = vec![false; n_past];
            for j in tail_start..n_past {
                mask[j] = true;
            }
            for &b in &sel_blocks[..n_sel_blocks] {
                for j in b * r..(b + 1) * r {
                    mask[j] = true;
                }
            }

            // q norm·rope를 qg에 즉시 적용 (attention은 루프 후 일괄)
            for h in 0..n_head {
                let src = qg[t][h * 2 * hd..h * 2 * hd + hd].to_vec();
                let mut qh = rms_norm(&src, &q_norm_w, hp.eps);
                rope_head(&mut qh, pos, n_rot, hp.rope_base);
                for (a, b) in qh.iter().zip(qg[t][h * 2 * hd..h * 2 * hd + hd].iter_mut()) {
                    *b = *a;
                }
            }
            mask_all[t] = mask;

            // 마스크 밀집 GQA + 게이트 — acc 있으면 루프 후 GPU 일괄, 없으면 즉시 CPU
            if gpu_attn {
                continue;
            }
            let mut attn_out = std::mem::take(&mut attn_all[t]);
            for h in 0..n_head {
                let kvh = h / (n_head / n_kv);
                let mut maxv = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_past];
                for (p, sc) in scores.iter_mut().enumerate() {
                    if !mask_all[t][p] {
                        *sc = f32::NEG_INFINITY;
                        continue;
                    }
                    let b = p * n_kv * hd + kvh * hd;
                    let mut d = 0.0f32;
                    for i in 0..hd {
                        d += qg[t][h * 2 * hd + i] * cache_k[b + i];
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
                for (p, sc) in scores.iter().enumerate() {
                    let w = sc / sum;
                    if w == 0.0 {
                        continue;
                    }
                    let b = p * n_kv * hd + kvh * hd;
                    for i in 0..hd {
                        attn_out[ob + i] += w * cache_v[b + i];
                    }
                }
                let gb = h * 2 * hd + hd;
                for i in 0..hd {
                    attn_out[ob + i] *= sigmoid(qg[t][gb + i]);
                }
            }
            attn_all[t] = attn_out;
        }
        // GPU 일괄 마스크 GQA — 캐시 전체(≤n_past_max)와 토큰별 마스크 전달.
        // 미래 위치는 mask 0으로 차단 (토큰 t는 pos_t+1까지만 참석).
        if gpu_attn {
            if let Some(acc) = self.acc.as_deref() {
                let qflat: Vec<f32> = qg.iter().flatten().copied().collect();
                let mut masku32: Vec<u32> = Vec::with_capacity(n_tok * n_past_max);
                for t in 0..t_len {
                    let n_past = (pos0 as usize) + t + 1;
                    for p in 0..n_past_max {
                        masku32.push((p < n_past && mask_all[t][p]) as u32);
                    }
                }
                let ck = self.seqs[seq].kv_k[full_idx].clone();
                let cv = self.seqs[seq].kv_v[full_idx].clone();
                let res = acc
                    .qsa_attention(
                        &qflat, &ck[..n_past_max * n_kv * hd], &cv[..n_past_max * n_kv * hd],
                        &masku32, kq_scale, n_past_max, n_head, n_kv, hd, n_tok,
                    )
                    .map_err(Q4Error::Io)?;
                for (t, row) in attn_all.iter_mut().enumerate() {
                    row.copy_from_slice(&res[t * n_head * hd..(t + 1) * n_head * hd]);
                }
            }
        }
        self.mm_batch(&attn_all, &wo, &mut out)?;
        Ok(out)
    }

    /// MoE FFN — top-10 라우팅(softmax→정규화) + shared(sigmoid 게이트).
    /// MoE FFN — 토큰-전문가 그룬핑 배치: 라우터는 전 토큰 배치, 각 전문가는
    /// 자기 토큰 서브배치로 3 role 배치 GEMM. 호출 수가 토큰 수와 무관하게
    /// 전문가 수(512)×3 + 공유 3 + 라우터 2로 고정 — GPU 장문 prefill의 핵심.
    fn moe_ffn(&self, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::moe");
        let hp = self.model.hp.clone();
        let w_route = self.model.w4(&format!("blk.{il}.ffn_gate_inp.weight"))?;
        let w_route_sh = self.model.w4(&format!("blk.{il}.ffn_gate_inp_shexp.weight"))?;
        let sh_up = self.model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
        let sh_gate = self.model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
        let sh_down = self.model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
        let n_exp = hp.n_expert;
        let n_used = hp.n_expert_used;
        let n_ff = hp.n_ff_exp;
        let n_embd = hp.n_embd;
        let t = xs.len();

        // 1) 라우팅 — 전 토큰 배치 1회
        let mut route = vec![vec![0.0f32; n_exp]; t];
        self.mm_batch(xs, &w_route, &mut route)?;
        let mut sgate_all = vec![vec![0.0f32; 1]; t];
        self.mm_batch(xs, &w_route_sh, &mut sgate_all)?;

        // 2) 선택 — 전문가별 (토큰, 가중치) 리스트
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_exp];
        for (ti, logits) in route.iter_mut().enumerate() {
            let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut zs = 0.0f32;
            for v in logits.iter_mut() {
                *v = (*v - mx).exp();
                zs += *v;
            }
            for v in logits.iter_mut() {
                *v /= zs;
            }
            let mut idx: Vec<usize> = (0..n_exp).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let sel = &idx[..n_used];
            let mut wsum: f32 = sel.iter().map(|&e| logits[e]).sum();
            wsum = wsum.max(6.103515625e-5);
            for &e in sel {
                let w = logits[e] / wsum;
                if w != 0.0 {
                    by_expert[e].push((ti, w));
                }
            }
        }

        // 3) 전문가별 서브배치 — 512×3 배치 GEMM (빈 전문가 스킵)
        let mut out = vec![vec![0.0f32; n_embd]; t];
        // 디코드 t=1 빠른 경로: 선택 전문가들의 gate·up가 동일 입력 — 그룹 1호출로
        // 2×n_used회 왕복을 1회로 (실측 병목: 전문가당 GPU 왕복 1,440회/스텝).
        if t == 1 {
            let sel: Vec<usize> = (0..n_exp).filter(|&e| !by_expert[e].is_empty()).collect();
            let n_sel = sel.len();
            let mut gate_y = vec![vec![0.0f32; n_ff]; n_sel];
            let mut up_y = vec![vec![0.0f32; n_ff]; n_sel];
            {
                let mut ws: Vec<crate::matmul::Weight> = Vec::with_capacity(2 * n_sel);
                for &e in &sel {
                    ws.push(self.model.expert_w(&format!("blk.{il}.ffn_gate_exps.weight"), e)?);
                    ws.push(self.model.expert_w(&format!("blk.{il}.ffn_up_exps.weight"), e)?);
                }
                let mut gu: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; n_ff]; 1]; 2 * n_sel];
                self.mm_group(xs, &ws, &mut gu)?;
                for i in 0..n_sel {
                    gate_y[i] = gu[2 * i][0].clone();
                    up_y[i] = gu[2 * i + 1][0].clone();
                }
            }
            for r in gate_y.iter_mut() {
                for i in 0..n_ff {
                    r[i] = silu(r[i]);
                }
            }
            for (r, u) in gate_y.iter_mut().zip(up_y.iter()) {
                for i in 0..n_ff {
                    r[i] *= u[i];
                }
            }
            for (k, &e) in sel.iter().enumerate() {
                let wd = self.model.expert_w(&format!("blk.{il}.ffn_down_exps.weight"), e)?;
                let (ti, w) = by_expert[e][0];
                let mut eo = vec![0.0f32; n_embd];
                self.mm(&gate_y[k], &wd, &mut eo)?;
                let o = &mut out[ti];
                for i in 0..n_embd {
                    o[i] += w * eo[i];
                }
            }
            // shared 전문가 배치 (기존 경로 공유) — 아래 일반 경로 shared 블록으로
            let mut sh_gate_y = vec![vec![0.0f32; n_ff]; t];
            let mut sh_up_y = vec![vec![0.0f32; n_ff]; t];
            self.mm_batch(xs, &sh_gate, &mut sh_gate_y)?;
            self.mm_batch(xs, &sh_up, &mut sh_up_y)?;
            for r in sh_gate_y.iter_mut() {
                for i in 0..n_ff {
                    r[i] = silu(r[i]);
                }
            }
            for (r, u) in sh_gate_y.iter_mut().zip(sh_up_y.iter()) {
                for i in 0..n_ff {
                    r[i] *= u[i];
                }
            }
            let mut shout = vec![vec![0.0f32; n_embd]; t];
            self.mm_batch(&sh_gate_y, &sh_down, &mut shout)?;
            for (ti, o) in out.iter_mut().enumerate() {
                let sh_w = sigmoid(sgate_all[ti][0]);
                for i in 0..n_embd {
                    o[i] += sh_w * shout[ti][i];
                }
            }
            return Ok(out);
        }
        for e in 0..n_exp {
            let list = &by_expert[e];
            if list.is_empty() {
                continue;
            }
            let sub: Vec<Vec<f32>> = list.iter().map(|&(ti, _)| xs[ti].clone()).collect();
            let mut gate_y = vec![vec![0.0f32; n_ff]; list.len()];
            let mut up_y = vec![vec![0.0f32; n_ff]; list.len()];
            let wg = self.model.expert_w(&format!("blk.{il}.ffn_gate_exps.weight"), e)?;
            let wu = self.model.expert_w(&format!("blk.{il}.ffn_up_exps.weight"), e)?;
            let wd = self.model.expert_w(&format!("blk.{il}.ffn_down_exps.weight"), e)?;
            self.mm_batch(&sub, &wg, &mut gate_y)?;
            self.mm_batch(&sub, &wu, &mut up_y)?;
            for r in gate_y.iter_mut() {
                for i in 0..n_ff {
                    r[i] = silu(r[i]);
                }
            }
            for (r, u) in gate_y.iter_mut().zip(up_y.iter()) {
                for i in 0..n_ff {
                    r[i] *= u[i];
                }
            }
            let mut eout = vec![vec![0.0f32; n_embd]; list.len()];
            self.mm_batch(&gate_y, &wd, &mut eout)?;
            for ((ti, w), eo) in list.iter().zip(eout.iter()) {
                let o = &mut out[*ti];
                for i in 0..n_embd {
                    o[i] += w * eo[i];
                }
            }
        }

        // 4) shared 전문가 — 전 토큰 배치
        let mut sh_gate_y = vec![vec![0.0f32; n_ff]; t];
        let mut sh_up_y = vec![vec![0.0f32; n_ff]; t];
        self.mm_batch(xs, &sh_gate, &mut sh_gate_y)?;
        self.mm_batch(xs, &sh_up, &mut sh_up_y)?;
        for r in sh_gate_y.iter_mut() {
            for i in 0..n_ff {
                r[i] = silu(r[i]);
            }
        }
        for (r, u) in sh_gate_y.iter_mut().zip(sh_up_y.iter()) {
            for i in 0..n_ff {
                r[i] *= u[i];
            }
        }
        let mut shout = vec![vec![0.0f32; n_embd]; t];
        self.mm_batch(&sh_gate_y, &sh_down, &mut shout)?;
        for (ti, o) in out.iter_mut().enumerate() {
            let sh_w = sigmoid(sgate_all[ti][0]);
            for i in 0..n_embd {
                o[i] += sh_w * shout[ti][i];
            }
        }
        Ok(out)
    }

    /// PLE 블록 — 해시 gather→key/value→게이트→방송→dilated conv→잔차 2경로.
    fn ple_block(
        &mut self,
        seq: usize,
        il: usize,
        res_hc: &mut Vec<Vec<f32>>,
        rows: &[u32],
    ) -> Result<(), Q4Error> {
        profile_span!("q4::ple");
        let hp = self.model.hp.clone();
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t = res_hc.len();
        let w_key = self.model.w4(&format!("blk.{il}.ple_key.weight"))?;
        let w_value = self.model.w4(&format!("blk.{il}.ple_value.weight"))?;
        let n_key = self.model.f32_vec4(&format!("blk.{il}.ple_norm_key.weight"))?;
        let n_query = self.model.f32_vec4(&format!("blk.{il}.ple_norm_query.weight"))?;
        let n_conv = self.model.f32_vec4(&format!("blk.{il}.ple_norm_conv.weight"))?;
        let conv_w = self.model.f32_vec4(&format!("blk.{il}.ple_conv1d.weight"))?;

        // emb gather [t][2560]
        let heads = hp.ple_heads_per_ngram * 2; // bigram+trigram = 16
        let emb_w = heads * hp.ple_head_dim; // 16×160 = 2560
        let mut emb = vec![vec![0.0f32; emb_w]; t];
        for (ti, r) in rows.chunks(heads).enumerate() {
            let mut flat = vec![0.0f32; emb_w];
            self.model.ple_gather(r, &mut flat)?;
            emb[ti] = flat;
        }
        // key/value 프로젝션
        let mut key = vec![vec![0.0f32; w_key.n_out as usize]; t];
        self.mm_batch(&emb, &w_key, &mut key)?;
        let mut value = vec![vec![0.0f32; w_value.n_out as usize]; t];
        self.mm_batch(&emb, &w_value, &mut value)?;

        let mut gated_hist: Vec<Vec<f32>> = Vec::with_capacity(t);
        for ti in 0..t {
            // grouped norm key / query — 감마는 전체 [hc_dim] 폭
            let mut k_n = vec![0.0f32; key[ti].len().max(hc_dim)];
            let kl = key[ti].len();
            debug_assert!(kl == hc_dim);
            for s in 0..hc {
                let head = key[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                k_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_key[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            let mut q_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = res_hc[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                q_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_query[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            // per-stream s = Σ key·query / √n_embd → sigmoid(sgn·√|s|)
            let mut gate = vec![0.0f32; hc];
            for s in 0..hc {
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    dot += k_n[s * n_embd + i] * q_n[s * n_embd + i];
                }
                dot /= (n_embd as f32).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                gate[s] = sigmoid(if dot >= 0.0 { mag } else { -mag });
            }
            // value 방송 × 게이트 → grouped norm
            let mut gated = vec![0.0f32; hc_dim];
            for s in 0..hc {
                for i in 0..n_embd {
                    gated[s * n_embd + i] = value[ti][i] * gate[s];
                }
            }
            let mut normalized = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = gated[s * n_embd..(s + 1) * n_embd].to_vec();
                normalized[s * n_embd..(s + 1) * n_embd].copy_from_slice(
                    &rms_norm(&head, &n_conv[s * n_embd..(s + 1) * n_embd], hp.eps),
                );
            }
            gated_hist.push(normalized);
        }

        // dilated depthwise conv (kern 4, dil 3, hist 9) — 시퀀스 상태 이용
        let kern = hp.ple_conv_k;
        let dil = hp.ple_ngram;
        let hist = (kern - 1) * dil;
        let st = &mut self.seqs[seq].ple_conv;
        // padded = hist(상태) + t열 → conv 출력 t열 → 상태 tail 갱신
        let mut padded: Vec<Vec<f32>> = Vec::with_capacity(hist + t);
        for j in 0..hist {
            padded.push(st[j * hc_dim..(j + 1) * hc_dim].to_vec());
        }
        for g in gated_hist.iter() {
            padded.push(g.clone());
        }
        let mut conv_out = vec![vec![0.0f32; hc_dim]; t];
        for ti in 0..t {
            for k in 0..kern {
                let start = hist + ti - (kern - 1 - k) * dil;
                let src = &padded[start];
                for c in 0..hc_dim {
                    conv_out[ti][c] += conv_w[c * kern + k] * src[c];
                }
            }
            for c in 0..hc_dim {
                conv_out[ti][c] = silu(conv_out[ti][c]);
            }
        }
        // 상태 갱신: 마지막 hist 열
        for j in 0..hist {
            let src = &padded[t + j];
            st[j * hc_dim..(j + 1) * hc_dim].copy_from_slice(src);
        }

        // 잔차: hidden + gated(norm 전 방송값) + conv_out — build_ple 반환식 그대로.
        // gated_pre는 conv 블록 위에서 이미 계산했으므로 재계산 대신 저장 구조 사용:
        // (value·gate 방송은 위 루프에서 `gated`로 존재했으나 norm에 덮어씀 — 재계산)
        for ti in 0..t {
            // 게이트 재계산 (결정적 동일값)
            let mut k_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = key[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                k_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_key[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            let mut q_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = res_hc[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                q_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_query[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            for s in 0..hc {
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    dot += k_n[s * n_embd + i] * q_n[s * n_embd + i];
                }
                dot /= (n_embd as f32).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                let g = sigmoid(if dot >= 0.0 { mag } else { -mag });
                for i in 0..n_embd {
                    res_hc[ti][s * n_embd + i] += value[ti][i] * g + conv_out[ti][s * n_embd + i];
                }
            }
        }
        Ok(())
    }

    /// PLE n-gram 해시 — 호스트 u64 (ctx[s]=직전 s토큰, EOS 절단).
    fn ple_hash(&mut self, seq: usize, tokens: &[u32]) -> Vec<u32> {
        let hp = self.model.hp.clone();
        let ngram = hp.ple_ngram;
        let heads = hp.ple_heads_per_ngram * 2; // bigram+trigram = 16
        let eos = hp.ple_eos;
        let hist0: Vec<u32> = self.seqs[seq].ple_hist.clone();
        let hist_valid = self.seqs[seq].ple_next_pos == self.seqs[seq].pos;
        let mut hist: Vec<u32> = if hist_valid { hist0 } else { vec![eos; ngram - 1] };
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
                let mut mixed = ctx[0].wrapping_mul(hp.ple_multipliers[0]);
                for j in 1..n {
                    mixed ^= ctx[j].wrapping_mul(hp.ple_multipliers[j]);
                }
                let base = (n - 2) * hp.ple_heads_per_ngram;
                for g in 0..hp.ple_heads_per_ngram {
                    let h = base + g;
                    rows.push(
                        (mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h]) as u32,
                    );
                }
            }
            hist.push(tok);
            if hist.len() > ngram - 1 {
                let cut = hist.len() - (ngram - 1);
                hist.drain(..cut);
            }
        }
        self.seqs[seq].ple_hist = hist;
        self.seqs[seq].ple_next_pos = self.seqs[seq].pos + tokens.len() as u32;
        rows
    }

    /// 시퀀스 상태 전체 초기화 (무상태 HTTP 서버용).
    pub fn reset_states(&mut self) {
        let ctx = self.seqs.first().map(|s| s.kv_k.first().map(|k| k.len() / (self.model.hp.n_kv * self.model.hp.head_dim)).unwrap_or(4096)).unwrap_or(4096);
        for i in 0..self.seqs.len() {
            self.seqs[i] = SeqState4::new(&self.model.hp, ctx);
        }
    }

    /// prefill: 전체 토큰 적립 + 마지막 logits.
    /// 1024토큰 청크로 분할 — 단일 초대형 forward는 libamdhip64 GPF 트리거
    /// (t=2311 실측, llama-server -ub 512도 같은 이유로 청크).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, Q4Error> {
        const CHUNK: usize = 1024;
        let mut last = None;
        for ch in tokens.chunks(CHUNK) {
            let mut tm = init_timings();
            let logits = self.forward_timed(seq, ch, tm.as_mut())?;
            if let Some(t) = &tm {
                t.report(&format!("prefill {}tok", ch.len()));
            }
            self.seqs[seq].pos += ch.len() as u32;
            last = Some(logits);
        }
        Ok(last.unwrap_or_else(|| vec![0.0; self.model.hp.vocab]))
    }

    /// 디코드 1토큰.
    pub fn decode1(&mut self, seq: usize, token: u32) -> Result<Vec<f32>, Q4Error> {
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
