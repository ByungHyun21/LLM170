//! full어텐션층 스테이지 — Q‖gate 퓨전 q_proj + GQA + RoPE(qwen35 미러).
//!
//! 텐서 계약(FN 실측): q_proj [12288,2560] = 24헤드×(q 256‖gate 256)
//! · k/v_proj [512,2560](KV 2헤드×256) · o_proj [2560,6144] ·
//! q/k_norm [256]. RoPE: n_rot 64(partial_rotary_factor 0.25)·theta 1e7
//! — qwen35 rope_head 재사용. indexer top-k 실제 경로는 [D2]
//! `stages/qsa.rs`(full_forward — 상태ful 선택 집합 어텐션). 아래
//! forward는 dense 전체 attention 프로브(구조 검증 목적).
//!
//! [D0 2026-10-20 판별] q_proj Q‖gate 레이아웃 = **헤드별 인터리브 확정**:
//! vLLM qwen3_next.py `_project_qkv_gate` 이저 경로가
//! `q_gate.view(num_heads, -1)` 후 `torch.chunk(2, dim=-1)`로 각 헤드
//! (q‖gate)를 분할 — q_proj 행 h*512..h*512+256=Q·+256..+512=gate.
//! FN QSAAttention(nvidia/qsa.py:337)이 이 클래스를 상속하고
//! "checkpoints always pack a sigmoid output gate next to Q" 주석과 함께
//! 동일 qkv_proj 계약 사용. 샤드 노름 통계 교차검증은 q/gate 분포가
//! 유사해 판별 불능[INFERENCE] — 코드 증거가 준거.
//!
//! q/k_norm = Gemma식 zero-centered(1+w) — `Model::norm_vec` 합성(D0).

use super::super::engine::Model;
use crate::ops::{rms_norm, rope_head, sigmoid};

pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let l = format!("model.language_model.layers.{il}.self_attn");
    let cfg = &m.cfg;
    let (n_head, n_kv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let n_tok = xs.len();
    let q = m.ent(&format!("{l}.q_proj.weight"))?;
    let q_n = *q.shape.first().ok_or("q_proj shape 부재")? as usize;
    if q_n != n_head * 2 * hd {
        return Err(format!("q_proj n={q_n} ≠ Q‖gate {n_head}×2×{hd}"));
    }
    let q_norm_w = m.norm_vec(&format!("{l}.q_norm.weight"))?;
    let k_norm_w = m.norm_vec(&format!("{l}.k_norm.weight"))?;
    let n_rot = hd / 4; // partial_rotary_factor 0.25
    let kq_scale = 1.0f32 / (hd as f32).sqrt();

    let mut out = vec![vec![0.0f32; n_head * hd]; n_tok];
    // 프로브 청크 내 causal attention(위치 0..n_tok, KV 캐시 없음).
    let mut k_cache = vec![0.0f32; n_tok * n_kv * hd];
    let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let qg = m.matvec_bf16(&format!("{l}.q_proj.weight"), &xs[t])?;
        let kk = m.matvec_bf16(&format!("{l}.k_proj.weight"), &xs[t])?;
        let vv = m.matvec_bf16(&format!("{l}.v_proj.weight"), &xs[t])?;
        for h in 0..n_kv {
            let src = kk[h * hd..h * hd + hd].to_vec();
            let mut head = rms_norm(&src, &k_norm_w, m.eps);
            rope_head(&mut head, t as u32, n_rot, cfg_rope_base());
            let b = t * n_kv * hd + h * hd;
            k_cache[b..b + hd].copy_from_slice(&head);
        }
        v_cache.push(vv);
        let mut attn = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            let src = qg[h * 2 * hd..h * 2 * hd + hd].to_vec();
            let mut qh = rms_norm(&src, &q_norm_w, m.eps);
            rope_head(&mut qh, t as u32, n_rot, cfg_rope_base());
            let kvh = h / (n_head / n_kv);
            // 점수 → 소프트맥스 → 가중합 (exp_cr 안씀 — 프로브 단순 경로).
            let mut scores = vec![0.0f32; t + 1];
            let mut mx = f32::NEG_INFINITY;
            for p in 0..=t {
                let kb = p * n_kv * hd + kvh * hd;
                let mut dp = 0.0f64;
                for i in 0..hd {
                    dp += qh[i] as f64 * k_cache[kb + i] as f64;
                }
                scores[p] = (dp as f32) * kq_scale;
                mx = mx.max(scores[p]);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let ob = h * hd;
            for (i, o) in attn[ob..ob + hd].iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (p, &sv) in scores.iter().enumerate() {
                    acc += sv as f64 * v_cache[p][kvh * hd + i] as f64;
                }
                *o = (acc / sum as f64) as f32;
            }
            // 게이트 — q 헤드 뒤 256차원 ⊙sigmoid.
            let gb = h * 2 * hd + hd;
            for i in 0..hd {
                attn[ob + i] *= sigmoid(qg[gb + i]);
            }
        }
        out[t] = attn;
    }
    for t in 0..n_tok {
        out[t] = m.matvec_bf16(&format!("{l}.o_proj.weight"), &out[t])?;
    }
    Ok(out)
}

/// rope_theta(FN 실측 1e7 — config text_config.rope_parameters).
fn cfg_rope_base() -> f32 {
    1e7
}
