//! 기본 연산: norms, 활성화, RoPE — llama.cpp ggml-cpu 시맨틱스 대응.
//!
//! - rms_norm: f64 누적, scale = 1/sqrt(mean(x²)+eps), 감마는 GGUF 저장값(1+w) 그대로 곱.
//! - l2_norm: scale = 1/max(sqrt(Σx²), eps) (ops.cpp — eps 는 floor).
//! - rope: 인접 페어 (2i, 2i+1), θ_p = base^(−p/n_pairs). IMROPE의 텍스트 전용 퇴화형
//!   (모든 위치 성분이 동일하면 표준 RoPE와 동일 — ggml mrope_cache_init 참조).

/// 제곱합 표준 순서 (2026-09-02, P0): 32세그먼트 병렬 친화적 구조 —
/// 각 세그먼트 f64 순차 누산 후 세그먼트 합을 순차 결합.
/// GPU rms_rows_part/finish 커널과 동일 순서 → 병렬화해도 비트 일치.
pub fn sq_sum(x: &[f32]) -> f64 {
    const SEG: usize = 32;
    let n = x.len();
    let chunk = n.div_ceil(SEG);
    let mut sum = 0.0f64;
    for u in 0..SEG {
        let lo = u * chunk;
        if lo >= n { break; }
        let hi = (lo + chunk).min(n);
        let mut part = 0.0f64;
        for &v in &x[lo..hi] {
            part += (v as f64) * (v as f64);
        }
        sum += part;
    }
    sum
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

pub fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / (sum.sqrt() as f32).max(eps);
    x.iter().map(|&v| v * scale).collect()
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// 헤드 벡터의 첫 n_rot 차원에 RoPE 적용 (나머지 통과).
/// qwen35 = IMROPE → **NEOX 페어링**: 페어 p = (p, p + n_rot/2),
/// θ_p = base^(−2p/n_rot). 텍스트 전용에서는 모든 위치 성분이 같아 표준 NEOX와 동일.
/// (ggml.h: GGML_ROPE_TYPE_IMROPE 40 — "still NEOX ordering")
pub fn rope_head(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for p in 0..half {
        let theta = base.powf(-(2.0 * p as f32) / n_rot as f32);
        let angle = pos as f32 * theta;
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (head[p], head[p + half]);
        head[p] = x0 * c - x1 * s;
        head[p + half] = x0 * s + x1 * c;
    }
}
