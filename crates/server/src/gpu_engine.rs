//! GPU 서빙 엔진 — W4A16 CUDA 디바이스 체인 + GPU head.
//!
//! W3-3 착륙 후 성능 캠페인(2026-10-08): 활성은 디바이스 상주(왕복은
//! 토큰당 임베딩 업로드 + 로짓 판독뿐), head는 head_bf16 커널(bf16
//! →f32 시프트 + k 직렬 f32 내적 — CPU 참조와 동일 순서).
//! LLM170_STAGED=1이면 구 스테이징 경로(호스트 왕복)로 폴백한다.
//! 판정 기준은 토큰열(골든)이다.

use llm170_backend_gpu::{AttnDims, GdnDims, W4a16Dec};
use std::path::Path;

pub struct GpuEngine {
    model: llm170_core::qwen35::Model,
    dec: W4a16Dec,
    n_slots: usize,
    /// 스테이징 폴백(LLM170_STAGED=1 — 비교/디버그 전용).
    staged: bool,
    /// GPU head 상주 여부(bf16 output.weight 업로드 성공).
    head_gpu: bool,
}

impl GpuEngine {
    /// 모델 로드 + 가중치 상주 업로드(간헐 ENOENT 재시도는 호출부 소관).
    pub fn load(dir: &Path, n_slots: usize, ctx: usize) -> Result<Self, String> {
        let model = llm170_core::qwen35::Model::load(dir).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let interval = hp.full_attn_interval.max(1);
        if interval != 4 {
            return Err(format!(
                "디코더는 4층 주기(full_attention_interval) 전용 — interval={interval}"
            ));
        }
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        // 1) 선형 상주 — GPU 체인은 원본(HF) 무게(순열은 커널 내부 처리).
        for name in model.engine_names() {
            if let Some(w) = model.w_raw(&name)
                && w.ty == llm170_core::wtype::WType::W4a16G128Split
            {
                let s = w.aux.ok_or_else(|| format!("{name}: aux 부재"))?;
                dec.upload_lin(&name, w.data, s, w.n_out as usize, w.n_in as usize)?;
            }
        }
        // 2) 노름 nw [2L+1][hidden] (+1 보정 — f32_vec).
        let mut nw: Vec<f32> = Vec::new();
        for il in 0..hp.n_layer {
            nw.extend(
                model
                    .f32_vec(&format!("blk.{il}.attn_norm.weight"))
                    .map_err(|e| e.to_string())?,
            );
            nw.extend(
                model
                    .f32_vec(&format!("blk.{il}.post_attention_norm.weight"))
                    .map_err(|e| e.to_string())?,
            );
        }
        nw.extend(
            model
                .f32_vec("output_norm.weight")
                .map_err(|e| e.to_string())?,
        );
        dec.set_norm_weights(&nw, 2 * hp.n_layer + 1)?;
        // 3) GDN 상수 — 원본(HF) 순서(a 먼저 b 다음 — 커널 ab 색인).
        let n_gdn = hp.n_layer - hp.n_layer / interval;
        let gd = GdnDims {
            n_gdn,
            hidden: hp.n_embd,
            h_k: hp.n_group,
            h_v: hp.dt_rank,
            d: hp.d_state,
        };
        let (mut cw, mut ab, mut alog, mut dtb, mut gnw): (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for il in 0..hp.n_layer {
            if (il + 1) % interval == 0 {
                continue;
            }
            cw.extend(
                model
                    .raw_f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))
                    .map_err(|e| e.to_string())?,
            );
            ab.extend(
                model
                    .raw_f32_vec(&format!("blk.{il}.ssm_alpha.weight"))
                    .map_err(|e| e.to_string())?,
            );
            ab.extend(
                model
                    .raw_f32_vec(&format!("blk.{il}.ssm_beta.weight"))
                    .map_err(|e| e.to_string())?,
            );
            alog.extend(
                model
                    .raw_f32_vec(&format!("blk.{il}.ssm_a"))
                    .map_err(|e| e.to_string())?,
            );
            dtb.extend(
                model
                    .raw_f32_vec(&format!("blk.{il}.ssm_dt.bias"))
                    .map_err(|e| e.to_string())?,
            );
            gnw.extend(
                model
                    .raw_f32_vec(&format!("blk.{il}.ssm_norm.weight"))
                    .map_err(|e| e.to_string())?,
            );
        }
        dec.set_gdn(gd, &cw, &ab, &alog, &dtb, &gnw)?;
        // 4) 어텐션 q/k 노름(+1 — f32_vec 보정).
        let n_attn = hp.n_layer / interval;
        let ad = AttnDims {
            n_attn,
            q_heads: hp.n_head,
            kv_heads: hp.n_kv,
            d: hp.head_dim,
            cap: ctx,
        };
        let (mut qnw, mut knw): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        for ai in 0..n_attn {
            let il = ai * interval + interval - 1;
            qnw.extend(
                model
                    .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
                    .map_err(|e| e.to_string())?,
            );
            knw.extend(
                model
                    .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
                    .map_err(|e| e.to_string())?,
            );
        }
        dec.set_attn(ad, &qnw, &knw)?;
        // 5) head — bf16 output.weight는 GPU 커널 경로(아니면 CPU 참조).
        let mut head_gpu = false;
        if let Some(w) = model.w("output.weight")
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            dec.upload_head(w.data, w.n_out as usize, w.n_in as usize)?;
            head_gpu = true;
        }
        Ok(GpuEngine {
            model,
            dec,
            n_slots: n_slots.max(1),
            staged: llm170_diag::flag::on("LLM170_STAGED"),
            head_gpu,
        })
    }

    /// 1토큰 체인(로짓 없이 상태만 진행) — prefill 중간 토큰용.
    fn forward(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.staged {
            self.dec.forward(slot, row)
        } else {
            self.dec.forward_device(slot, row)
        }
    }

    /// 1토큰 다음 로짓 — GPU head 상주 시 로짓만 회수(xn 판독 생략),
    /// 아니면 디바이스/스테이징 체인 + CPU 참조 head.
    fn logits(&mut self, slot: usize, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_gpu && !self.staged {
            return self.dec.forward_device_head(slot, row);
        }
        let xn = self.forward(slot, row)?;
        self.head_logits(&xn)
    }

    /// head 로짓 — CPU 참조 경로(bf16 디퀀트 f32 내적, 골든과 동일 계급).
    /// A9: 로더가 보증하더라도 패닉 대신 Result — 서버 오류 응답으로 유도.
    fn head_logits(&self, xn: &[f32]) -> Result<Vec<f32>, String> {
        let head = self
            .model
            .w("output.weight")
            .ok_or_else(|| "output.weight 부재 — head 계약 위반".to_string())?;
        let mut lg = vec![0.0f32; head.n_out as usize];
        llm170_core::matmul::matmul(xn, &head, &mut lg);
        Ok(lg)
    }

    /// 시퀀스 prefill(토큰 순차 forward) → 마지막 로짓.
    /// head는 마지막 토큰만 계산한다(중간 토큰은 체인만 — head 비용 상각).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut last = None;
        let n = tokens.len();
        for (i, tok) in tokens.iter().enumerate() {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            if i + 1 == n {
                last = Some(self.logits(seq, &row)?);
            } else {
                self.forward(seq, &row)?;
            }
        }
        // A10: 빈 프롬프트는 조용한 제로 로짓 대신 명시 오류(조용한 오염 금지 —
        // infer.rs 거부 표면과 일치).
        last.ok_or_else(|| "prefill: 빈 프롬프트 — 토큰 1개 이상 필요".to_string())
    }

    /// 배치 디코드 — seq별 1토큰 forward → 로짓.
    pub fn decode(&mut self, seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(seq_ids.len());
        for (s, tok) in seq_ids.iter().zip(tokens.iter()) {
            let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
            out.push(self.logits(*s, &row)?);
        }
        Ok(out)
    }

    /// 배치 greedy — 토큰만 회수.
    pub fn decode_np_greedy(
        &mut self,
        seq_ids: &[usize],
        tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        let lg = self.decode(seq_ids, tokens)?;
        Ok(lg
            .iter()
            .map(|l| llm170_core::matmul::greedy_from(l))
            .collect())
    }

    /// 단일 greedy 디코드.
    pub fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        let lg = self.decode(&[seq], &[token])?;
        Ok(llm170_core::matmul::greedy_from(&lg[0]))
    }

    /// 슬롯 상태 리셋(GDN 링/스캔 + pp + pos).
    pub fn reset_seq(&mut self, seq: usize) -> Result<(), String> {
        self.dec.reset_state(seq)
    }

    /// 전 슬롯 리셋(서버 워밍업 종료 후).
    pub fn reset_states(&mut self) -> Result<(), String> {
        for s in 0..self.n_slots {
            self.dec.reset_state(s)?;
        }
        Ok(())
    }

    /// 표면형 근사 디토크(표시용 — 정식 BPE 디토크는 TOKENIZER 몫).
    pub fn piece(&self, tok: u32) -> String {
        self.model
            .token_pieces
            .get(tok as usize)
            .map(String::as_str)
            .unwrap_or("")
            .replace('Ġ', " ")
            .replace('Ċ', "\n")
    }
}
