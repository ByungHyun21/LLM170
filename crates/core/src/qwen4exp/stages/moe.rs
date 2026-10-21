//! MoE FFN 스테이지 — 512전문가 top-10 + shared(gemma4/qwen35 MoE 미러).
//!
//! 텐서 계약(FN 실측): gate(라우터) BF16 [512,2560] · 전문가 gptq int4
//! 트리플(gate/up [640,2560] · down [2560,640], g128) · shared_expert
//! gate/up/down BF16 + shared_expert_gate [1,2560].
//! 라우팅 = 전체 softmax → top-k → 재정규화(qwen35 select_topk 재사용 —
//! vllm Qwen3NextSparseMoeBlock 시맨틱).
//!
//! [오라클 병렬화 2026-10-21] 3페이즈 재구성(engine::par_ranges — 비트동일):
//! A) (t,e) 유닛 병렬 — h[t][e] = silu(gw·x)⊙(uw·x). 전문가 게이트/업
//!    디양자화가 지배 비용(토큰×전문가마다 전량 재디양자화 — 종전 순차
//!    구조의 낭비 그대로, 분할만)이라 유닛 자기완결로 병렬화.
//! B) shared 사전계산(토큰 병렬) — sgate·s_out(BF16 matvec — 임계 미만
//!    순차, 중첩 없음).
//! C) (t,행밴드) 유닛 병렬 — out[t][r] = Σ_e(선택 순서) w·dot(dw_e[r], h_e)
//!    에 sgate·s_out[t][r]를 마지막 가산. 행별 전문가 가산 순서·shared
//!    마지막 가산 순서는 순차판과 동일(비트동일). hs·out은 평탄 버퍼
//!    (유닛별 상호배타 원소 기록 — 중첩 Vec raw 인덱싱 회피).

use super::super::engine::{Model, SendPtr, par_ranges};
use crate::ops::{sigmoid, silu};
use crate::qwen35::stages::moe::select_topk;

/// C페이즈 행밴드 폭(2560행 → 10밴드 — 유닛 그레인 ≈ 1.6M MAC).
const OUT_BAND: usize = 256;

pub fn forward(m: &Model, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let cfg = &m.cfg;
    let l = format!("model.language_model.layers.{il}.mlp");
    let n_exp = cfg.n_experts;
    let g = m.ent(&format!("{l}.gate.weight"))?;
    if g.shape != [n_exp as u64, cfg.hidden_size as u64] {
        return Err(format!(
            "mlp.gate {:?} ≠ [{n_exp},{}]",
            g.shape, cfg.hidden_size
        ));
    }
    let (hidden, top_k) = (cfg.hidden_size, cfg.top_k);
    let n_tok = xs.len();
    // 라우팅 — 토큰 순차(소형 matvec [512,2560]).
    let mut sels: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tok);
    for x in xs {
        let logits = m.matvec_bf16(&format!("{l}.gate.weight"), x)?;
        sels.push(select_topk(&logits, top_k));
    }
    // 선택 전문가 존재 사전검증(병렬 본체의 expect는 여기서 보증 —
    // gptq 트리플은 {p}.qweight 형태로 존재).
    for sel in &sels {
        for &(e, _) in sel {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                m.ent(&format!("{l}.experts.{e}.{p}.qweight"))?;
            }
        }
    }
    // A) h — 평탄 [n_tok×top_k], 유닛 (t,e) 병렬.
    // SAFETY: hs 기록 원소 u=(t,ei)는 유닛마다 상호배타 — 병렬 쓰기 레이스
    // 없음. 스코프 조인 전에 모든 기록이 완료된다(구조는 gdn.rs ar_pool
    // SendPtr 선례와 동일).
    let mut hs: Vec<Vec<f32>> = vec![Vec::new(); n_tok * top_k];
    {
        let hs_ptr = SendPtr::new(hs.as_mut_ptr());
        let m_r: &Model = m;
        let l_r = l.as_str();
        par_ranges(n_tok * top_k, |b0, b1| {
            let hsp = unsafe { hs_ptr.as_ptr::<Vec<f32>>() };
            for u in b0..b1 {
                let (t, ei) = (u / top_k, u % top_k);
                let (e, _w) = sels[t][ei];
                let x = &xs[t];
                let gw = m_r
                    .expert_mat(&format!("{l_r}.experts.{e}.gate_proj"))
                    .expect("사전검증됨");
                let uw = m_r
                    .expert_mat(&format!("{l_r}.experts.{e}.up_proj"))
                    .expect("사전검증됨");
                let mut h = vec![0.0f32; gw.1];
                for (r, row) in gw.0.iter().enumerate() {
                    h[r] = silu(dot(row, x)) * dot(&uw.0[r], x);
                }
                unsafe { *hsp.add(u) = h };
            }
        });
    }
    // B) shared 사전계산 — sgate[t]·s_out[t](토큰 병렬).
    // SAFETY: 기록 인덱스 t는 유닛별 상호배타.
    let mut sgate = vec![0.0f32; n_tok];
    let mut s_out: Vec<Vec<f32>> = vec![Vec::new(); n_tok];
    {
        let (sg_ptr, so_ptr) = (
            SendPtr::new(sgate.as_mut_ptr()),
            SendPtr::new(s_out.as_mut_ptr()),
        );
        par_ranges(n_tok, |b0, b1| {
            let sgp = unsafe { sg_ptr.as_ptr::<f32>() };
            let sop = unsafe { so_ptr.as_ptr::<Vec<f32>>() };
            for t in b0..b1 {
                let x = &xs[t];
                let sg = sigmoid(
                    m.matvec_bf16(&format!("{l}.shared_expert_gate.weight"), x)
                        .unwrap()[0],
                );
                let s_up = m
                    .matvec_bf16(&format!("{l}.shared_expert.gate_proj.weight"), x)
                    .unwrap();
                let s_val = m
                    .matvec_bf16(&format!("{l}.shared_expert.up_proj.weight"), x)
                    .unwrap();
                let mut h = vec![0.0f32; s_up.len()];
                for (r, (&a, &b)) in s_up.iter().zip(s_val.iter()).enumerate() {
                    h[r] = silu(a) * b;
                }
                let so = m
                    .matvec_bf16(&format!("{l}.shared_expert.down_proj.weight"), &h)
                    .unwrap();
                unsafe {
                    *sgp.add(t) = sg;
                    *sop.add(t) = so;
                }
            }
        });
    }
    // C) 출력 — 평탄 [n_tok×hidden], 유닛 (t,행밴드) 병렬. 전문가 순서·
    // shared 마지막 가산은 순차판과 동일(비트동일).
    // SAFETY: 기록 원소 t·hidden+r은 유닛별 상호배타.
    let mut flat = vec![0.0f32; n_tok * hidden];
    {
        let out_ptr = SendPtr::new(flat.as_mut_ptr());
        let m_r: &Model = m;
        let l_r = l.as_str();
        let bands = hidden.div_ceil(OUT_BAND);
        par_ranges(n_tok * bands, |b0, b1| {
            let op = unsafe { out_ptr.as_ptr::<f32>() };
            for u in b0..b1 {
                let (t, bi) = (u / bands, u % bands);
                let (r0, r1) = (bi * OUT_BAND, ((bi + 1) * OUT_BAND).min(hidden));
                for r in r0..r1 {
                    let mut acc = 0.0f32;
                    for (ei, &(e, w)) in sels[t].iter().enumerate() {
                        let drow = m_r
                            .expert_row(&format!("{l_r}.experts.{e}.down_proj"), r)
                            .expect("down_proj 행");
                        acc += w * dot(&drow, &hs[t * top_k + ei]);
                    }
                    acc += sgate[t] * s_out[t][r];
                    unsafe { *op.add(t * hidden + r) = acc };
                }
            }
        });
    }
    Ok(flat.chunks(hidden).map(<[f32]>::to_vec).collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x as f64 * y as f64;
    }
    acc as f32
}
