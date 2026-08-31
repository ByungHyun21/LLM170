//! hc(hyper-connection mix) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use crate::ops::{rms_norm, sigmoid, silu};
use llm170_profiler::profile_span;

    /// grouped RMSNorm + 저랭크 게이트 + 스트림 평균 + inject.
    /// kind = "attn"|"ffn" → blk.{il}.hc_{kind}_{norm,down,up,inject}.weight
    /// 토큰 축 배치: down/up/inject 각 전 토큰 1회 — GPU 왕복을 층당 6회로 고정
    /// (토큰당 288회 왕복이 장문 prefill 병목이었음 — 2026-08-31 실측).
    pub fn hc_mix(
        ctx: &Ctx,
        il: usize,
        kind: &str,
        res_hc: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), Q4Error> {
        profile_span!("q4::hc_mix");
        let hp = &ctx.model.hp;
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t = res_hc.len();
        let w_norm = ctx.model.f32_vec4(&format!("blk.{il}.hc_{kind}_norm.weight"))?;
        let w_down = ctx.model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
        let w_up = ctx.model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?;
        let w_inject = ctx.model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;

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
        // 2) 저랭크 down → silu(lo/hc) → up → 게이트.
        // down·inject는 동일 입력 xn_all — 그룹 1호출 (왕복 3→2).
        let mut lo_all = vec![vec![0.0f32; w_down.n_out as usize]; t];
        let mut inject_all = vec![vec![0.0f32; hc]; t];
        {
            let mut gi = vec![
                std::mem::take(&mut lo_all),
                std::mem::take(&mut inject_all),
            ];
            ctx.mm_group(&xn_all, &[w_down, w_inject], &mut gi)?;
            lo_all = std::mem::take(&mut gi[0]);
            inject_all = std::mem::take(&mut gi[1]);
        }
        for lo in lo_all.iter_mut() {
            for v in lo.iter_mut() {
                *v = silu(*v / hc as f32);
            }
        }
        let mut gate_all = vec![vec![0.0f32; hc_dim]; t];
        ctx.mm_batch(&lo_all, &w_up, &mut gate_all)?;
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
    pub fn hc_mix_head(ctx: &Ctx, res_hc: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::hc_mix_head");
        let hp = &ctx.model.hp;
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t = res_hc.len();
        let w_norm = ctx.model.f32_vec4("output_hc_norm.weight")?;
        let w_down = ctx.model.w4("output_hc_down.weight")?;
        let w_up = ctx.model.w4("output_hc_up.weight")?;
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
        ctx.mm_batch(&xn_all, &w_down, &mut lo_all)?;
        for lo in lo_all.iter_mut() {
            for v in lo.iter_mut() {
                *v = silu(*v / hc as f32);
            }
        }
        let mut gate_all = vec![vec![0.0f32; hc_dim]; t];
        ctx.mm_batch(&lo_all, &w_up, &mut gate_all)?;
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

