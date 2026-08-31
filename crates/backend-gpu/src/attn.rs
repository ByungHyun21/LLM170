//! QSA 마스크드 밀집 GQA 어텐션 커널 — GPU 상주 (2026-08-31).
//!
//! qsa_layer의 CPU 핫루프(헤드별 score→softmax→V 혼합→게이트)를 GPU로.
//! 호스트가 이미 계산한 것: q/k/v 프로젝션·per-head norm·rope·KV 캐시 적립·
//! 인덱서 스코어·top-k 마스크(토큰별 u32 0/1 배열).
//!
//! 2-커널 분해 (결정적 — 축 순차 누산):
//!  1) score_k: 유닛=(t,h,p) — d 순차 내적 → scores[t][h][p]
//!     (마스크 외 p는 -INFINITY)
//!  2) mix_k: 유닛=(t,h,d) — p 순차 max→Σexp→정규화 V 혼합 + sigmoid 게이트
//! 병렬도: score = t·24·n_past 유닛, mix = t·24·256 유닛 — 디코드 t=1이면
//! score ~49K·mix 6K 유닛 (작지만 CPU→GPU 이관으로 왕복 제거가 본 목적).

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn qsa_score(
    q: &Tensor<f32>,       // [t][n_head*2*hd] q‖gate 인터리브 (norm·rope 완료)
    ck: &Tensor<f32>,      // [n_past*n_kv*hd]
    mask: &Tensor<u32>,    // [t][n_past] 0/1
    scores: &mut Tensor<f32>, // [t][n_head][n_past]
    n_past: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    t_len: usize,
) {
    let p = ABSOLUTE_POS_X as usize;
    let h = ABSOLUTE_POS_Y as usize;
    let t = ABSOLUTE_POS_Z as usize;
    if p >= n_past || h >= n_head || t >= t_len {
        terminate!();
    }
    if mask[t * n_past + p] == 0 {
        scores[(t * n_head + h) * n_past + p] = -3.0e38;
        terminate!();
    }
    let kvh = h / (n_head / n_kv);
    let qb = t * n_head * 2 * hd + h * 2 * hd;
    let kb = p * n_kv * hd + kvh * hd;
    let mut d = 0.0f32;
    for i in 0..hd {
        d += q[qb + i] * ck[kb + i];
    }
    scores[(t * n_head + h) * n_past + p] = d;
}

#[cube(launch_unchecked)]
pub fn qsa_mix(
    q: &Tensor<f32>,        // [t][n_head*2*hd] — 게이트 접근용
    scores: &Tensor<f32>,   // [t][n_head][n_past]
    cv: &Tensor<f32>,       // [n_past*n_kv*hd]
    out: &mut Tensor<f32>,  // [t][n_head*hd]
    n_past: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    t_len: usize,
) {
    let d_i = ABSOLUTE_POS_X as usize;
    let h = ABSOLUTE_POS_Y as usize;
    let t = ABSOLUTE_POS_Z as usize;
    if d_i >= hd || h >= n_head || t >= t_len {
        terminate!();
    }
    let sbase = (t * n_head + h) * n_past;
    // max (순차 — 결정적) — 초기값을 첫 score로 (리터럴 타입 통일)
    let mut maxv = scores[sbase];
    for p in 0..n_past {
        let s = scores[sbase + p];
        if s > maxv {
            maxv = s;
        }
    }
    // Σexp (순차)
    let mut sum = (scores[sbase] - maxv).exp();
    for p in 1..n_past {
        sum += (scores[sbase + p] - maxv).exp();
    }
    // V 혼합 (p 순차 — 결정적) + 게이트 — continue 미지원이라 if 블록
    let kvh = h / (n_head / n_kv);
    let mut a = 0.0f32;
    for p in 0..n_past {
        let w = (scores[sbase + p] - maxv).exp() / sum;
        if w != 0.0 {
            let kb = p * n_kv * hd + kvh * hd;
            a += w * cv[kb + d_i];
        }
    }
    let gb = t * n_head * 2 * hd + h * 2 * hd + hd;
    let g = 1.0 / (1.0 + (-q[gb + d_i]).exp());
    out[t * n_head * hd + h * hd + d_i] = a * g;
}
