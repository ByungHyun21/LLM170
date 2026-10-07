//! MoE 스테이지 — 해시 라우팅(L0-2) + noaux_tc 라우팅 + MXFP4 전문가 SwiGLU.
//!
//! 재현 대상(model.py `Gate`·`Expert`·`MoE`) — 보고서 §3·§5:
//! - 게이트: scores = x.float()·W^T (fp32, 무양자화) → sqrtsoftplus
//!   s_i = √softplus(g_i).
//! - 해시 층(L0-2): tid2eid [vocab,6] I64 고정 룩업 — top-k/bias 없이
//!   토큰 ID로 전문가 선택. 라우팅 가중치는 여전히 계산:
//!   w_i = s_{eid_i} gather → 6개 정규화(Σ로 나눔) → ×1.5.
//! - noaux_tc: sel = topk(s + bias)(bias는 선택에만), w = gather(s, sel)
//!   (bias 미포함) → w /= Σw → ×1.5. 공유 전문가는 이후 무스케일 가산.
//! - top-k 동점: 값 내림차순, 동점 낮은 인덱스 우선(안정 정렬 — 모듈 계약).
//! - 전문가 SwiGLU limit 10 (비대칭): up-proj [-10,10], gate-proj max 10.
//! - 전문가 QAT: w1/w3/w2 입력 FP8-sim(128블록) — fp4_gemm이 act를
//!   FP8로 양자화한다. 가중치는 EXL3 trellis 디양자치(원본 MXFP4:
//!   e2m1×2^(s-127) per 32 — `ops::mxfp4_dequant` 참조).
//! - 누산 순서: 전문가 인덱스 오름차순 y += w·E(x), 공유 마지막 —
//!   model.py 루프 순서와 동일(결정적).

use crate::deepseek4::ops::{
    bf16_round, bf16_round_slice, fp8_sim, gemm_nt, sqrtsoftplus, swiglu_limit,
};

/// 전문가 가중치 — w1/w3 [dim][moe_inter], w2 [moe_inter][dim] (k-major f32).
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
}

/// 게이트 파라미터 — weight [dim][n_routed] k-major(체크포인트 [n,dim] 전치),
/// bias [n_routed](해시 층은 None), tid2eid [vocab·6](해시 층만).
#[derive(Debug, Clone)]
pub struct GateWeights {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    /// vocab×6 행 우선 int (해시 층).
    pub tid2eid: Option<Vec<i64>>,
}

/// 게이트 스코어 — fp32 gemm(무양자화) + sqrtsoftplus.
pub fn gate_scores(x: &[f32], gw: &GateWeights, n_routed: usize, dim: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n_routed];
    gemm_nt(x, &gw.weight, dim, n_routed, &mut s);
    for v in s.iter_mut() {
        *v = sqrtsoftplus(*v);
    }
    s
}

/// 안정 top-k — 값 내림차순, 동점 낮은 인덱스 우선. 반환 (인덱스 오름차순 아님 —
/// 스코어 순서) — 라우팅 가중치와 함께 반환.
pub fn topk_stable(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.into_iter().map(|i| (i, scores[i])).collect()
}

/// 라우팅 결과 — 전문가 인덱스(오름차순 정렬됨)와 정규화 가중치.
/// 정규화: w = gather(s, sel); w /= Σw; w ×= route_scale (§5).
pub fn route(scores: &[f32], sel: Vec<usize>, route_scale: f32) -> Vec<(usize, f32)> {
    let raw: Vec<f32> = sel.iter().map(|&i| scores[i]).collect();
    let sum: f32 = raw.iter().sum();
    let mut out: Vec<(usize, f32)> = sel
        .into_iter()
        .zip(raw)
        .map(|(i, w)| (i, w / sum * route_scale))
        .collect();
    // 누산 순서 계약: 전문가 인덱스 오름차순.
    out.sort_by_key(|&(i, _)| i);
    out
}

/// 해시 라우팅 — tid2eid 행 룩업(6전문가 고정) + 동일 정규화.
pub fn route_hash(scores: &[f32], tid2eid_row: &[i64], route_scale: f32) -> Vec<(usize, f32)> {
    let sel: Vec<usize> = tid2eid_row.iter().map(|&e| e as usize).collect();
    route(scores, sel, route_scale)
}

/// noaux_tc 라우팅 — sel = topk(s + bias), w = gather(s, sel).
pub fn route_routed(scores: &[f32], bias: &[f32], k: usize, route_scale: f32) -> Vec<(usize, f32)> {
    let biased: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();
    let sel: Vec<usize> = topk_stable(&biased, k)
        .into_iter()
        .map(|(i, _)| i)
        .collect();
    route(scores, sel, route_scale)
}

/// 전문가 FFN 1토큰 — FP8-sim(128) 입력 → w1/w3 → 비대칭 클램프 SwiGLU →
/// ×w → bf16 → FP8-sim → w2. 출력 [dim] bf16값.
pub fn expert_ffn(
    x: &[f32],
    e: &ExpertWeights,
    weight: f32,
    dim: usize,
    inter: usize,
    limit: f32,
) -> Vec<f32> {
    let mut xq = x.to_vec();
    fp8_sim(&mut xq, dim, 128);
    let mut g = vec![0.0f32; inter];
    gemm_nt(&xq, &e.w1, dim, inter, &mut g);
    bf16_round_slice(&mut g);
    let mut u = vec![0.0f32; inter];
    gemm_nt(&xq, &e.w3, dim, inter, &mut u);
    bf16_round_slice(&mut u);
    // SwiGLU f32 — 비대칭 클램프 후 라우팅 가중치 곱.
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        h[i] = swiglu_limit(g[i], u[i], limit) * weight;
    }
    bf16_round_slice(&mut h);
    fp8_sim(&mut h, inter, 128);
    let mut y = vec![0.0f32; dim];
    gemm_nt(&h, &e.w2, inter, dim, &mut y);
    bf16_round_slice(&mut y);
    y
}

/// MoE 프리필 — 토큰별 라우팅 후 **전문가별 그룹 처리**(전문가 인덱스
/// 오름차순 — 토큰별 누산 순서가 model.py 루프와 동일, 계약).
/// 전문가 가중치는 한 번에 1개만 상주(피크 메모리) — `expert` 콜백은
/// 필요 전문가당 1회 호출(로더 지연 디양자화). 공유 전문가는 토큰순
/// 무가중 후행, 최종 bf16 경계.
#[allow(clippy::too_many_arguments)]
pub fn moe_forward(
    x: &[f32],
    input_ids: &[u32],
    gw: &GateWeights,
    shared: &ExpertWeights,
    cfg: &crate::deepseek4::Deepseek4Config,
    is_hash: bool,
    mut expert: impl FnMut(usize) -> ExpertWeights,
) -> Vec<f32> {
    let (dim, t) = (cfg.dim, x.len() / cfg.dim);
    // 1) 토큰별 라우팅(게이트만 필요 — 전문가 적재 전).
    let routes: Vec<Vec<(usize, f32)>> = (0..t)
        .map(|ti| {
            let xt = &x[ti * dim..(ti + 1) * dim];
            let scores = gate_scores(xt, gw, cfg.n_routed, dim);
            if is_hash {
                let base = input_ids[ti] as usize * cfg.n_activated;
                let row = &gw.tid2eid.as_ref().expect("해시 층에 tid2eid 필요")
                    [base..base + cfg.n_activated];
                route_hash(&scores, row, cfg.route_scale)
            } else {
                route_routed(
                    &scores,
                    gw.bias.as_ref().expect("라우티드 층에 bias 필요"),
                    cfg.n_activated,
                    cfg.route_scale,
                )
            }
        })
        .collect();
    // 2) 전문가 id 오름차순 — 각 토큰의 누산 순서가 참조와 동일.
    let mut by_expert: std::collections::BTreeMap<usize, Vec<(usize, f32)>> =
        std::collections::BTreeMap::new();
    for (ti, r) in routes.iter().enumerate() {
        for &(eid, w) in r {
            by_expert.entry(eid).or_default().push((ti, w));
        }
    }
    let mut y = vec![0.0f32; t * dim];
    for (eid, toks) in by_expert {
        let e = expert(eid);
        for (ti, w) in toks {
            let out = expert_ffn(
                &x[ti * dim..(ti + 1) * dim],
                &e,
                w,
                dim,
                cfg.moe_inter,
                cfg.swiglu_limit,
            );
            for (yi, &ov) in y[ti * dim..(ti + 1) * dim].iter_mut().zip(out.iter()) {
                *yi += ov;
            }
        }
    }
    // 3) 공유 전문가 — 무가중, 라우티드 이후(§5). 최종 bf16 경계.
    for ti in 0..t {
        let sh = expert_ffn(
            &x[ti * dim..(ti + 1) * dim],
            shared,
            1.0,
            dim,
            cfg.moe_inter,
            cfg.swiglu_limit,
        );
        for (yi, &sv) in y[ti * dim..(ti + 1) * dim].iter_mut().zip(sh.iter()) {
            *yi += sv;
        }
        for yi in y[ti * dim..(ti + 1) * dim].iter_mut() {
            *yi = bf16_round(*yi);
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> crate::deepseek4::Deepseek4Config {
        crate::deepseek4::config::test_cfg()
    }

    /// 해시 라우팅 — 표 룩업·정규화·×1.5 (보고서 §3).
    #[test]
    fn hash_route_lookup() {
        let scores = vec![0.1f32, 0.9, 0.3, 0.7, 0.5, 0.2, 0.4, 0.6];
        // tid 7 → 전문가 [3, 1, 5] (k=3로 축소 테스트).
        let row = [3i64, 1, 5];
        let r = route_hash(&scores, &row, 1.5);
        assert_eq!(r.len(), 3);
        let ids: Vec<usize> = r.iter().map(|&(i, _)| i).collect();
        assert_eq!(ids, vec![1, 3, 5], "인덱스 오름차순 정렬");
        let wsum: f32 = r.iter().map(|&(_, w)| w).sum();
        assert!((wsum - 1.5).abs() < 1e-6, "정규화+×1.5 → 합 1.5 ({wsum})");
        // 정렬 후 r[0]=eid1(w=0.9/Σ), r[1]=eid3(w=0.7/Σ), r[2]=eid5(0.2/Σ).
        assert!(
            (r[0].1 - 0.9 / (0.7 + 0.9 + 0.2) * 1.5).abs() < 1e-6,
            "{}",
            r[0].1
        );
        assert!(
            (r[1].1 - 0.7 / (0.7 + 0.9 + 0.2) * 1.5).abs() < 1e-6,
            "{}",
            r[1].1
        );
    }

    /// noaux_tc — bias는 선택에만 반영, 가중치는 bias 미포함.
    #[test]
    fn routed_bias_selection_only() {
        let scores = vec![1.0f32, 0.9, 0.1, 0.05];
        let bias = vec![-2.0f32, 5.0, 0.0, 0.0];
        // s+bias = [-1, 5.9, .1, .05] → top2 = [1, 2].
        let r = route_routed(&scores, &bias, 2, 1.5);
        let ids: Vec<usize> = r.iter().map(|&(i, _)| i).collect();
        assert_eq!(ids, vec![1, 2]);
        // w = s (bias 없음): 0.9/(0.9+0.1)·1.5, 0.1/(1.0)·1.5.
        assert!((r[0].1 - 0.9 / 1.0 * 1.5).abs() < 1e-6);
        assert!((r[1].1 - 0.1 / 1.0 * 1.5).abs() < 1e-6);
    }

    /// top-k 동점 — 낮은 인덱스 우선 (계약 문서화).
    #[test]
    fn topk_tie_lower_index_wins() {
        let scores = vec![1.0f32, 3.0, 3.0, 2.0, 3.0];
        let top = topk_stable(&scores, 3);
        let ids: Vec<usize> = top.iter().map(|&(i, _)| i).collect();
        assert_eq!(ids, vec![1, 2, 4], "동점 3.0 → 1,2,4 순");
        let top2 = topk_stable(&scores, 5);
        assert_eq!(top2.len(), 5);
    }

    /// 전문가 SwiGLU — 비대칭 클램프 적용 확인(작은 항등 가중치).
    #[test]
    fn expert_swiglu_clamp() {
        let c = cfg();
        let (dim, inter) = (c.dim, c.moe_inter);
        // w1: gate = 2·x (첫 행만 상수 11을 만들기보다 — 직접 값 주입 어려움:
        // x=1, w1 행 전부 11 → gate=11·dim… 클램프 10 초과 확인용).
        let x = vec![1.0f32 / dim as f32; dim];
        let w1 = vec![11.0f32; dim * inter];
        let w3 = vec![11.0f32; dim * inter];
        let w2 = vec![1.0f32; inter * dim]; // 항등 아님(전치) — 값만 확인
        let e = ExpertWeights { w1, w2, w3 };
        let y = expert_ffn(&x, &e, 1.0, dim, inter, 10.0);
        assert_eq!(y.len(), dim);
        assert!(y.iter().all(|v| v.is_finite()));
        // gate=11 → 10, up=11 → 10 → silu(10)·10 > 0 — w2 1.0 가중합 양수.
        assert!(y.iter().all(|&v| v > 0.0));
    }

    /// moe_forward — 해시 경로 형상·유한·결정성.
    #[test]
    fn moe_forward_hash_shape() {
        let c = cfg();
        let (dim, t) = (c.dim, 2);
        let gw = GateWeights {
            weight: vec![0.01f32; dim * c.n_routed],
            bias: None,
            tid2eid: Some(vec![1i64, 2, 3, 0, 2, 3]), // tid 0,1
        };
        let shared = ExpertWeights {
            w1: vec![0.02; dim * c.moe_inter],
            w2: vec![0.02; c.moe_inter * dim],
            w3: vec![0.02; dim * c.moe_inter],
        };
        let x = vec![0.3f32; t * dim];
        let ids = [0u32, 1];
        let y = moe_forward(&x, &ids, &gw, &shared, &c, true, |_| ExpertWeights {
            w1: vec![0.05; dim * c.moe_inter],
            w2: vec![0.05; c.moe_inter * dim],
            w3: vec![0.05; dim * c.moe_inter],
        });
        assert_eq!(y.len(), t * dim);
        assert!(y.iter().all(|v| v.is_finite()));
        let y2 = moe_forward(&x, &ids, &gw, &shared, &c, true, |_| ExpertWeights {
            w1: vec![0.05; dim * c.moe_inter],
            w2: vec![0.05; c.moe_inter * dim],
            w3: vec![0.05; dim * c.moe_inter],
        });
        assert_eq!(y, y2, "결정성");
    }
}
