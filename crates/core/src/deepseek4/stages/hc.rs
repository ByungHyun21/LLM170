//! mHC(멀티 하이퍼커넥션) + Sinkhorn 스테이지 — 보고서 §4, kernel.py
//! `hc_split_sinkhorn_kernel`·model.py `Block.hc_pre/hc_post/hc_head` 재현.
//!
//! 4스트림 X∈[b,s,4,4096]. 층 입력 믹스:
//! - X̂ = RMSNorm(vec(X)) (비가중, 16384디m, eps = **norm_eps**, f32)
//! - mix = X̂·hc_fn^T, fn [24,16384] → 24 = (2+4)·4
//! - pre[j]  = σ(mix[j]·scale[0] + base[j]) + hc_eps        (A_l, j<4)
//! - post[j] = 2σ(mix[4+j]·scale[1] + base[4+j])            (C_l = 2σ)
//! - raw[j,k] = mix[8+j·4+k]·scale[2] + base[8+j·4+k]       (B̃ 4×4 row-major)
//! - Sinkhorn(raw, 20iter): softmax_rows → +eps → /(col+eps) →
//!   19×{ /(row+eps); /(col+eps) } — 이중확률 근사 B
//! - 적용: layer_in = Σ_j pre_j·X[j] (X는 **비정규 원본**),
//!   X'[j] = post_j·F(layer_in) + Σ_k B[j,k]·X[k]
//!
//! hc 파라미터는 전부 fp32(체크포인트 F32). fp32 전용 스테이지(§9.4).

use crate::deepseek4::ops::{bf16_round, rms_scale};
use crate::ops::sigmoid;

/// hc 층 파라미터(소유형) — fn[24·d], base[24], scale[3]. 헤드 변형은
/// fn[4·d], base[4], scale[1].
#[derive(Debug, Clone)]
pub struct HcParams {
    pub fns: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: Vec<f32>,
}

/// Sinkhorn 분할 — kernel.py `hc_split_sinkhorn_kernel`와 동일 연산열.
/// 입력 mixes[24], 출력 (pre[4], post[4], comb[16] row-major).
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(mixes.len(), (2 + hc) * hc);
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    }
    for j in 0..hc {
        post[j] = 2.0 * sigmoid(mixes[hc + j] * scale[1] + base[hc + j]);
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
        }
    }
    // 1) comb = comb.softmax(-1) — 행 최대값 분산 안정화.
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = crate::ops::exp_cr(*v - m);
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    // 2) comb += eps; comb /= (colsum + eps).
    for v in comb.iter_mut() {
        *v += eps;
    }
    let mut col = vec![0.0f32; hc];
    for j in 0..hc {
        for k in 0..hc {
            col[k] += comb[j * hc + k];
        }
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] /= col[k] + eps;
        }
    }
    // 3) (iters-1)× { /(rowsum+eps); /(colsum+eps) }.
    let mut row = vec![0.0f32; hc];
    for _ in 1..iters {
        row.fill(0.0);
        for j in 0..hc {
            for k in 0..hc {
                row[j] += comb[j * hc + k];
            }
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] /= row[j] + eps;
            }
        }
        col.fill(0.0);
        for j in 0..hc {
            for k in 0..hc {
                col[k] += comb[j * hc + k];
            }
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] /= col[k] + eps;
            }
        }
    }
    (pre, post, comb)
}

/// hc_pre — 층 입력 믹스. `x`는 [hc·d] (스트림 j at x[j·d..(j+1)·d]).
/// 반환: (y[d] = Σ pre_j·X[j] bf16 경계, post, comb).
/// mixes = (X·fn^T)·rsqrt(mean(X²)+norm_eps) — 정규화는 **믹스에만** 곱하고
/// 가중합에는 원본 X를 쓴다(model.py `hc_pre` 그대로).
pub fn hc_pre(
    x: &[f32],
    d: usize,
    hc: usize,
    p: &HcParams,
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mix_hc = (2 + hc) * hc;
    let rsqrt = rms_scale(x, norm_eps);
    let mut mixes = vec![0.0f32; mix_hc];
    // F.linear(x, fn) — f32 dot, k 오름차순; 이후 rsqrt 곱.
    for (i, m) in mixes.iter_mut().enumerate() {
        let fr = &p.fns[i * (hc * d)..(i + 1) * (hc * d)];
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(fr.iter()) {
            acc += a * b;
        }
        *m = acc * rsqrt;
    }
    let scale3 = [p.scale[0], p.scale[1], p.scale[2]];
    let (pre, post, comb) = hc_split_sinkhorn(&mixes, &scale3, &p.base, hc, iters, hc_eps);
    // y = Σ_j pre_j·X_j — f32 누산 후 bf16 경계 반올림(y.to(dtype)).
    let mut y = vec![0.0f32; d];
    for j in 0..hc {
        let xj = &x[j * d..(j + 1) * d];
        let pj = pre[j];
        for (yi, &xv) in y.iter_mut().zip(xj.iter()) {
            *yi += pj * xv;
        }
    }
    for yi in y.iter_mut() {
        *yi = bf16_round(*yi);
    }
    (y, post, comb)
}

/// hc_post — 층 출력 재확장. X'[j] = post_j·f + Σ_k comb[j,k]·residual[k].
/// f32 산술, 출력 bf16 경계(type_as(x)).
pub fn hc_post(
    f: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    d: usize,
    hc: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; hc * d];
    for j in 0..hc {
        let pj = post[j];
        for i in 0..d {
            y[j * d + i] = pj * f[i];
        }
        for k in 0..hc {
            let c = comb[j * hc + k];
            let rk = &residual[k * d..(k + 1) * d];
            for i in 0..d {
                y[j * d + i] += c * rk[i];
            }
        }
        for i in 0..d {
            y[j * d + i] = bf16_round(y[j * d + i]);
        }
    }
    y
}

/// hc_head — 최종 축소 믹스(fn[4·d], base[4], scale[1]).
/// pre = σ(mix·scale + base) + hc_eps, y = Σ pre_j·X_j (bf16 경계).
pub fn hc_head(
    x: &[f32],
    d: usize,
    hc: usize,
    p: &HcParams,
    norm_eps: f32,
    hc_eps: f32,
) -> Vec<f32> {
    let rsqrt = rms_scale(x, norm_eps);
    let mut pre = vec![0.0f32; hc];
    for j in 0..hc {
        let fr = &p.fns[j * (hc * d)..(j + 1) * (hc * d)];
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(fr.iter()) {
            acc += a * b;
        }
        pre[j] = sigmoid(acc * rsqrt * p.scale[0] + p.base[j]) + hc_eps;
    }
    let mut y = vec![0.0f32; d];
    for j in 0..hc {
        let xj = &x[j * d..(j + 1) * d];
        for (yi, &xv) in y.iter_mut().zip(xj.iter()) {
            *yi += pre[j] * xv;
        }
    }
    for yi in y.iter_mut() {
        *yi = bf16_round(*yi);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sinkhorn → 이중확률 근사: 행·열 합 ≈ 1 (eps 1e-6 오차 허용).
    #[test]
    fn sinkhorn_doubly_stochastic() {
        let hc = 4;
        let raw = [
            0.9f32, -0.3, 0.1, 2.0, 0.2, -1.1, 0.05, 0.4, 1.3, 0.0, -0.7, 0.22, 0.31, 0.8, -0.15,
            0.6,
        ];
        let mixes = vec![0.5f32; 8] // pre/post 영역 (여기선 값 무관)
            .into_iter()
            .chain(raw.iter().copied())
            .collect::<Vec<_>>();
        let scale = [0.3f32, 0.2, 0.15];
        let base = vec![0.1f32; 24];
        let (_, _, comb) = hc_split_sinkhorn(&mixes, &scale, &base, hc, 20, 1e-6);
        for j in 0..hc {
            let rs: f32 = comb[j * hc..(j + 1) * hc].iter().sum();
            assert!((rs - 1.0).abs() < 1e-3, "행 {j} 합 {rs}");
        }
        for k in 0..hc {
            let cs: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
            assert!((cs - 1.0).abs() < 1e-3, "열 {k} 합 {cs}");
        }
        assert!(comb.iter().all(|&v| v > 0.0));
    }

    /// hc_pre/hc_post 공식 검증 — fn=0이면 pre=σ(base)+eps, post=2σ(base),
    /// comb는 softmax(+eps) 기반. 스트림 가중합·재확장 항등 확인.
    #[test]
    fn hc_pre_post_formulas() {
        let (d, hc) = (8usize, 4);
        let p = HcParams {
            fns: vec![0.0f32; 24 * hc * d],
            base: vec![0.25f32; 24],
            scale: vec![0.5f32; 3],
        };
        let x: Vec<f32> = (0..hc * d).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let (y, post, comb) = hc_pre(&x, d, hc, &p, 1e-6, 1e-6, 20);
        // pre = σ(0.25)+1e-6, y = Σ pre·X_j — f32 누산 후 bf16 경계.
        let pre = sigmoid(0.25) + 1e-6;
        for i in 0..d {
            let want: f32 = (0..hc).map(|j| pre * x[j * d + i]).sum();
            assert_eq!(y[i], bf16_round(want), "{} vs {}", y[i], want);
        }
        assert!(post.iter().all(|&v| (v - 2.0 * sigmoid(0.25)).abs() < 1e-6));
        // fn=0 → comb raw = base = 0.25 균등 → softmax 균등 1/4 → 행합 1 근사.
        for j in 0..hc {
            let rs: f32 = comb[j * hc..(j + 1) * hc].iter().sum();
            assert!((rs - 1.0).abs() < 1e-3);
        }
        // hc_post: comb 균일 c → y[j] = post·f + c·Σresidual (근사) — 형상·유한 확인.
        let f = vec![0.5f32; d];
        let yp = hc_post(&f, &x, &post, &comb, d, hc);
        assert_eq!(yp.len(), hc * d);
        assert!(yp.iter().all(|v| v.is_finite()));
        // pre/post/comb를 재계산해 결정성(동일 입력 → 동일 출력).
        let (y2, _, _) = hc_pre(&x, d, hc, &p, 1e-6, 1e-6, 20);
        assert_eq!(y, y2);
    }

    /// hc_head — fn=0 → pre_j 상수, y = Σ(σ(base_j)+eps)·X_j.
    #[test]
    fn hc_head_formula() {
        let (d, hc) = (4usize, 4);
        let p = HcParams {
            fns: vec![0.0f32; hc * hc * d],
            base: vec![-0.5f32; 4],
            scale: vec![2.0f32; 1],
        };
        let x: Vec<f32> = (0..hc * d).map(|i| i as f32 * 0.1).collect();
        let y = hc_head(&x, d, hc, &p, 1e-6, 1e-6);
        let c = sigmoid(-0.5) + 1e-6;
        for i in 0..d {
            let want: f32 = (0..hc).map(|j| c * x[j * d + i]).sum();
            assert_eq!(y[i], bf16_round(want), "{} vs {}", y[i], want);
        }
    }
}
