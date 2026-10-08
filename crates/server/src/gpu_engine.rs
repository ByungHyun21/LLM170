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
        let mut dec = W4a16Dec::new(n_slots, hp.n_embd, hp.n_layer)?;
        dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
        upload_model(&mut dec, &model, ctx)?;
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

    /// 메모리 분류(모니터링) — (상주 가중치, KV, CPU 오프로드, PLE 오프로드).
    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        self.dec.mem_stats()
    }

    /// 토큰당 활성 가중치 바이트(실효 대역폭 계산용).
    pub fn active_weight_bytes(&self) -> u64 {
        self.dec.active_weight_bytes()
    }

    /// MoE 배치 모드 — "none" | "resident" | "streaming".
    pub fn moe_mode(&self) -> &'static str {
        self.dec.moe_mode()
    }

    /// 복사 계측 — [h2d, d2h, d2d] × (바이트, ns, 호출).
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.dec.copy_stats()
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

    /// 시퀀스 prefill → 마지막 로짓. t≤8 배치 청크(GEMM t≥2 경로)로 처리하고
    /// head는 마지막 토큰만 계산한다(중간 토큰 상각).
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.model.hp.n_embd;
        let n = tokens.len();
        if n == 0 {
            // A10: 빈 프롬프트는 조용한 제로 로짓 대신 명시 오류.
            return Err("prefill: 빈 프롬프트 — 토큰 1개 이상 필요".into());
        }
        let mut last = None;
        if self.staged {
            // 구 스테이징 경로는 t=1 전용 — 한 토큰씩.
            for (i, tok) in tokens.iter().enumerate() {
                let row = self.model.embed_row(*tok).map_err(|e| e.to_string())?;
                if i + 1 == n {
                    let xn = self.dec.forward(seq, &row)?;
                    last = Some(self.head_logits(&xn)?);
                } else {
                    self.dec.forward(seq, &row)?;
                }
            }
            return last.ok_or_else(|| "prefill: 빈 프롬프트".to_string());
        }
        // 청크 크기 오버라이드(진단/폴백): LLM170_PREFILL_T=1이면 토큰 순차.
        // MoE 프리필 배치(t≤8)는 전문가 상주 전제 — 스트리밍은 t=1 폴백.
        let tmax = if self.dec.is_moe() && !self.dec.moe_experts_resident() {
            1
        } else {
            llm170_diag::flag::val("LLM170_PREFILL_T")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8)
                .clamp(1, 8)
        };
        let mut i = 0usize;
        while i < n {
            let t = (n - i).min(tmax);
            let mut rows = Vec::with_capacity(t * h);
            for &tok in &tokens[i..i + t] {
                rows.extend_from_slice(&self.model.embed_row(tok).map_err(|e| e.to_string())?);
            }
            let is_last = i + t == n;
            if is_last && self.head_gpu {
                last = Some(self.dec.forward_prefill(seq, &rows, t, true)?);
            } else if is_last {
                let xn = self.dec.forward_prefill(seq, &rows, t, false)?;
                last = Some(self.head_logits(&xn)?);
            } else {
                self.dec.forward_prefill(seq, &rows, t, false)?;
            }
            i += t;
        }
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

/// MoE(35B-A3B) 상주 배선 — 플레인 bf16 업로드 + 전문가 슬라이스 테이블.
/// GpuEngine·w4a16-gpu 프로브 공용.
///
/// SAFETY(전문가 테이블): 항목은 모델 스토어의 mmap 슬라이스 주소다 —
/// 호출자가 모델을 디코더와 함께 보유하는 수명 계약(디코더가 먼저 drop).
pub(crate) fn upload_moe(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    for name in model.engine_names() {
        // token_embd(호스트 embed_row)·output(head 전용 업로드)는 체인이
        // 선형으로 쓰지 않는다 — VRAM 절약(각 ~970MiB @35B).
        if name == "token_embd.weight" || name == "output.weight" {
            continue;
        }
        if let Some(w) = model.w_raw(&name)
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            dec.upload_plain(&name, w.data, w.n_out as usize, w.n_in as usize)?;
        }
    }
    let plain_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    dec.set_plain_mode(true);
    let (group, scale_bf16) = model.expert_quant();
    dec.set_moe(
        hp.n_experts,
        hp.top_k,
        hp.moe_ffn,
        hp.shared_ffn,
        group,
        scale_bf16,
    );
    let mut tab: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(hp.n_layer * hp.n_experts * 3);
    for il in 0..hp.n_layer {
        for e in 0..hp.n_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let (q, s, _, _) = model
                    .expert_slice(il, e, proj)
                    .ok_or_else(|| format!("전문가 슬라이스 부재: L{il} e{e} {proj}"))?;
                tab.push((
                    q.as_ptr() as u64,
                    q.len() as u64,
                    s.as_ptr() as u64,
                    s.len() as u64,
                ));
            }
        }
    }
    // 상주 가능(170HX 64GB 등)이면 전문가 전량 VRAM — 토큰당 PCIe 스트리밍 제거.
    // 판정: 전문가 바이트 + 1GiB 여유. 불가면 스트리밍 테이블(호스트 mmap).
    let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
    let free = llm170_backend_gpu::cuda_mem_free()
        .map(|(f, _)| f)
        .unwrap_or(0);
    let resident = match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
        Some("0") => false,
        Some("1") => true,
        _ => free > expert_bytes + (1 << 30),
    };
    let tab_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    if resident {
        dec.upload_experts_resident(&tab)?;
    } else {
        dec.set_expert_table(tab);
    }
    eprintln!(
        "[moe] 플레인 {plain_ms:.0}ms · 테이블 {tab_ms:.0}ms · 전문가 {:.0}ms ({:.1}GB/s) — 전문가 {:.2}GiB · 여유 {:.2}GiB → {}",
        t2.elapsed().as_secs_f64() * 1e3,
        (expert_bytes as f64 * 1e-9) / t2.elapsed().as_secs_f64().max(1e-9),
        expert_bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        if resident {
            "전량 상주"
        } else {
            "호스트 스트리밍"
        }
    );
    Ok(())
}

/// 모델 상주 업로드(1~4단계) — GpuEngine·w4a16-gpu 프로브 공용.
/// head는 호출부 소관(프로브는 --no-head/스테이징 분기가 있다).
pub(crate) fn upload_model(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    ctx: usize,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let interval = hp.full_attn_interval.max(1);
    if interval != 4 {
        return Err(format!(
            "디코더는 4층 주기(full_attention_interval) 전용 — interval={interval}"
        ));
    }
    let moe = hp.n_experts > 0;
    // 1) 선형 상주 — GPU 체인은 원본(HF) 무게(순열은 커널 내부 처리).
    if moe {
        upload_moe(dec, model)?;
    } else {
        for name in model.engine_names() {
            if let Some(w) = model.w_raw(&name)
                && w.ty == llm170_core::wtype::WType::W4a16Split
            {
                let s = w.aux.ok_or_else(|| format!("{name}: aux 부재"))?;
                dec.upload_lin(&name, w.data, s, w.n_out as usize, w.n_in as usize)?;
            }
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
    Ok(())
}
