//! gdn_ar — GDN 단일 토큰(AR) 상태 갱신 GPU 커널.
//!
//! 유닛 = S 행렬의 dv 열 1개 (d=128). 각 유닛은 자기 열만 읽고 쓰므로
//! 크로스 유닛 통신 없음:
//!   1) sk[dv] = Σ_kdim (S[kdim,dv]·e^g)·k[kdim]   (S ← S·e^g 동시)
//!   2) delta = (v[dv] − sk[dv])·β; S[kdim,dv] += k[kdim]·delta
//!   3) o[dv] = Σ_kdim S'[kdim,dv]·(q[kdim]·scale)
//! e^g·β·(q·scale)는 호스트 사전 계산 (커널 f32 스칼라 인수·exp 미지원 회피).
//!
//! ADR-0005: mul_add 금지 — 순수 mul+add. cubecl HIP 코드젠 주의 규칙 준수
//! (순수 u32 산술 불필요 — 전 인수 f32 텐서).

use cubecl::prelude::*;

/// 그리드: (h_v, n_seqs) — 큐브당 1 (seq, v-head), 유닛당 dv 열.
#[cube(launch_unchecked)]
pub fn gdn_ar(
    s: &mut Tensor<f32>,          // [b][h_v][d*d] 상태 (in-place)
    q_scaled: &Tensor<f32>,       // [b][k_stride] (q·scale 완료)
    k: &Tensor<f32>,              // [b][k_stride]
    v: &Tensor<f32>,              // [b][v_stride]
    beta_ge: &Tensor<f32>,        // [b][h_v][2] — β, e^g 쌍
    out: &mut Tensor<f32>,        // [b][v_stride]
    d: usize,                     // 상태 차원 (128)
    k_stride: usize,
    v_stride: usize,
    h_v: usize,
    h_k: usize,
) {
    // 큐브 = (b,h) pair 1개, 유닛 = dv 열. CUBE_DIM 128 고정 — d≤128 커버,
    // 초과 유닛 terminate (CubeDim(d<32) 비정상 실측, 2026-09-01).
    let pair = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u >= d {
        terminate!();
    }
    let b = pair / h_v;
    let h = pair % h_v;
    let kh = h % h_k;
    let base_s = pair * d * d;
    let qk0 = b * k_stride + kh * d;
    let v0 = b * v_stride + h * d;
    let beta = beta_ge[pair * 2];
    let g_exp = beta_ge[pair * 2 + 1];

    // 1) S ← S·e^g; sk = Σ S·k
    let mut sk = 0.0f32;
    for kdim in 0..d {
        let sv = s[base_s + kdim * d + u] * g_exp;
        s[base_s + kdim * d + u] = sv;
        sk += sv * k[qk0 + kdim];
    }
    // 2) delta·k 외적 가산
    let delta = (v[v0 + u] - sk) * beta;
    for kdim in 0..d {
        let addr = base_s + kdim * d + u;
        s[addr] += k[qk0 + kdim] * delta;
    }
    // 3) o = Σ S'·q_scaled
    let mut o = 0.0f32;
    for kdim in 0..d {
        o += s[base_s + kdim * d + u] * q_scaled[qk0 + kdim];
    }
    out[v0 + u] = o;
}
