//! hc(hyper-connection mix) 스테이지 — Engine4에서 분리 (리팩터링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).
//! plans/90 B4 D4: hc_mix·hc_mix_head 공통골격 통합(inject Optional) —
//! f32 순차 순서는 통합 전과 행별로 동일.

use super::super::Q4Error;
use super::Ctx;
use crate::matmul::Weight;
use crate::ops::{rms_norm, sigmoid, silu};
use llm170_diag::profile_span;

/// 그룹 RMSNorm — [hc·n] 행을 스트림별로 잘라 각각 정규화(plans/90 B4 D7).
/// `for s in 0..hc` 인라인 판과 동일 산술(스트림 순서 불변).
pub(crate) fn grouped_rms(x: &[f32], w: &[f32], hc: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut xn = vec![0.0f32; hc * n];
    for s in 0..hc {
        let head = x[s * n..(s + 1) * n].to_vec();
        xn[s * n..(s + 1) * n].copy_from_slice(&rms_norm(&head, &w[s * n..(s + 1) * n], eps));
    }
    xn
}

/// hc_mix·hc_mix_head 공통 골격(D4) — inject 있으면 down·inject 그룹 1호출,
/// 없으면 down 단독. 게이트 적용·스트림 평균까지 공유.
fn hc_mix_ex(
    ctx: &Ctx,
    w_norm: &[f32],
    w_down: &Weight,
    w_up: &Weight,
    w_inject: Option<&Weight>,
    res_hc: &[Vec<f32>],
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), Q4Error> {
    let hp = &ctx.model.hp;
    let (n_embd, hc) = (hp.n_embd, hp.hc);
    let hc_dim = hc * n_embd;
    let t = res_hc.len();

    // 1) grouped RMSNorm — 전 토큰 (감마는 (1+w) 폴딩, 스트림별 축소)
    let xn_all: Vec<Vec<f32>> = res_hc
        .iter()
        .map(|x| grouped_rms(x, w_norm, hc, n_embd, hp.eps))
        .collect();
    // 2) 저랭크 down → silu(lo/hc) → up → 게이트.
    // down·inject는 동일 입력 xn_all — 그룹 1호출 (왕복 3→2).
    let mut lo_all = vec![vec![0.0f32; w_down.n_out as usize]; t];
    let mut inject_all = vec![vec![0.0f32; hc]; t];
    match w_inject {
        Some(wi) => {
            let mut gi = vec![
                std::mem::take(&mut lo_all),
                std::mem::take(&mut inject_all),
            ];
            ctx.mm_group(&xn_all, &[*w_down, *wi], &mut gi)?;
            lo_all = std::mem::take(&mut gi[0]);
            inject_all = std::mem::take(&mut gi[1]);
        }
        None => {
            ctx.mm_batch(&xn_all, w_down, &mut lo_all)?;
        }
    }
    for lo in lo_all.iter_mut() {
        for v in lo.iter_mut() {
            *v = silu(*v / hc as f32);
        }
    }
    let mut gate_all = vec![vec![0.0f32; hc_dim]; t];
    ctx.mm_batch(&lo_all, w_up, &mut gate_all)?;
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
    let w_norm = ctx.model.f32_vec4(&format!("blk.{il}.hc_{kind}_norm.weight"))?;
    let w_down = ctx.model.w4(&format!("blk.{il}.hc_{kind}_down.weight"))?;
    let w_up = ctx.model.w4(&format!("blk.{il}.hc_{kind}_up.weight"))?;
    let w_inject = ctx.model.w4(&format!("blk.{il}.hc_{kind}_inject.weight"))?;
    hc_mix_ex(ctx, &w_norm, &w_down, &w_up, Some(&w_inject), res_hc)
}

/// 출력 헤드용 HC mix (inject 없음) — output_hc_{norm,down,up}. 동일 배치 구조.
pub fn hc_mix_head(ctx: &Ctx, res_hc: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Q4Error> {
    profile_span!("q4::hc_mix_head");
    let w_norm = ctx.model.f32_vec4("output_hc_norm.weight")?;
    let w_down = ctx.model.w4("output_hc_down.weight")?;
    let w_up = ctx.model.w4("output_hc_up.weight")?;
    let (out, _) = hc_mix_ex(ctx, &w_norm, &w_down, &w_up, None, res_hc)?;
    Ok(out)
}
