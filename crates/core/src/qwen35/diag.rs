//! qwen35 진단 잔재 격리 (plans/90 B4 — qwen4exp/frame/diag.rs 패턴).
//! G0(GDN)/A3(어텐션) 계측 덤프 — 본체(layers.rs)와 분리해 조사 코드가
//! 값 경로를 오염하지 않게 한다. 전부 LLM170_DEBUG_LAYERS 게이트.

/// 비트 xor 체크섬(가중) — VkD 대조용.
fn wxor(v: &[f32]) -> u64 {
    v.iter().map(|&x| (x.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15)).fold(0u64, |a, b| a ^ b)
}

/// G0: GDN 배치 입력·출력 체크섬(레이어 0, t_len==1 경로).
#[allow(clippy::too_many_arguments)]
pub(super) fn g0_gdn(
    r0: usize,
    r1: usize,
    o_all: &[f32],
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    v_len: usize,
    k_len: usize,
    dt_rank: usize,
) {
    // t_len 무관 마지막 행 기준 (per-token VkD 대조용)
    let last = r1 - 1;
    let sumo: f64 = o_all[last * v_len..(last + 1) * v_len].iter().map(|&v| v as f64).sum();
    let xco = wxor(&o_all[r0 * v_len..r1 * v_len]);
    let xcq = wxor(&q_all[r0 * k_len..r1 * k_len]);
    eprintln!("  G0dbg o_all sum={sumo:.6} xor={xco:016x} q_all xor={xcq:016x}");
    let xck = wxor(&k_all[r0 * k_len..r1 * k_len]);
    let xcv = wxor(&v_all[r0 * v_len..r1 * v_len]);
    let xcb = wxor(&beta_all[r0 * dt_rank..r1 * dt_rank]);
    let xcg = wxor(&g_all[r0 * dt_rank..r1 * dt_rank]);
    eprintln!("  G0dbg k_all xor={xck:016x} v_all xor={xcv:016x} beta xor={xcb:016x} g_all xor={xcg:016x}");
    let xce = g_all[r0 * dt_rank..r1 * dt_rank]
        .iter()
        .map(|&v| (crate::ops::exp_cr(v).to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .fold(0u64, |a, b| a ^ b);
    eprintln!("  G0dbg exp_cr(g) xor={xce:016x}");
}

/// A3: 어텐션 층 norm 직후 q 판(word0·첫 6개 q8).
pub(super) fn a3_normed(xs0: &[f32]) {
    eprintln!("  A3dbg normed[0..6]={:?}", &xs0[0..6]);
    if let Some(qb) = crate::quant::quantize_row_q8_ref(xs0).first() {
        let mut word = 0u32;
        for (i, b) in qb.qs.iter().take(4).enumerate() {
            word |= (*b as u8 as u32) << (8 * i);
        }
        eprintln!(
            "  A3dbg cpu q word0={word:#010x} d={:e} q[0..6]={:?}",
            qb.d,
            qb.qs.iter().take(6).collect::<Vec<_>>()
        );
    }
}

/// A3: CPU 어텐션 직전 캐시·게이트 표본.
pub(super) fn a3_cache(pos: usize, b0: usize, cache_k: &[f32], cache_v: &[f32], qg_row: &[f32], hd: usize) {
    eprintln!("  A3dbg pos{pos} cache_k[b0..4]={:?} cache_k[0..4]={:?}", &cache_k[b0..b0 + 4], &cache_k[0..4]);
    eprintln!("  A3dbg cache_v[0..4]={:?}", &cache_v[b0..b0 + 4]);
    eprintln!("  A3dbg gate h0 [0..4]={:?}", &qg_row[hd..hd + 4]);
    eprintln!(
        "  A3dbg sigmoid(g)={:?}",
        (0..4).map(|i| crate::ops::sigmoid(qg_row[hd + i])).collect::<Vec<_>>()
    );
}
