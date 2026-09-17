//! probes/gdn — GDN t 불변성(청크 결함 진단) (probes.rs에서 이동, plans/78 R3).


/// GDN AR(순환 상태 갱신)의 **t 불변성** 프로브 — 무게·모델 불필요, 수 초.
///
/// 같은 가상 시퀀스를 두 방식으로 흘리고 상태·출력을 대조한다:
///   (가) 1회 t=T            (나) T/per회 t=per (행 밴드 슬라이스, 상태 이월)
/// 청크 크기에 따라 프리필 결과가 갈리는 결함(2026-09-17: Q4_CHUNK=64에서
/// Flash-Next 출력 붕괴)의 커널 축 판정용. 상태는 `t_cur`(=frame_begin)로
/// 행 수를 정하므로 밴드 실행은 호출마다 frame_begin(per)가 필요하다.
pub fn gdn_ar_invariance(t: usize, per: usize, h_k: usize, h_v: usize, d: usize) -> Result<String, String> {
    use super::q4acc::Q4Acc;
    use llm170_core::matmul::{FrameHost, FrameState};
    if per == 0 || t % per != 0 || t == per {
        return Err(format!("인자: t={t} per={per} (t % per == 0 && t != per)"));
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let acc = Q4Acc::new()?;
    let (k_len, v_len) = (h_k * d, h_v * d);
    let hn = acc.frame_alloc(t * k_len)?;
    let hk = acc.frame_alloc(t * k_len)?;
    let hv = acc.frame_alloc(t * v_len)?;
    let hbg = acc.frame_alloc(t * h_v * 2)?;
    let ha = acc.frame_alloc(h_v * d * d)?;
    let hb = acc.frame_alloc(h_v * d * d)?;
    let hoa = acc.frame_alloc(t * v_len)?;
    let hob = acc.frame_alloc(t * v_len)?;
    let (q, k, v, bg): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) = (
        (0..t * k_len).map(|_| lcg() * 0.1).collect(),
        (0..t * k_len).map(|_| lcg() * 0.1).collect(),
        (0..t * v_len).map(|_| lcg() * 0.1).collect(),
        // β>0, e^g∈(0,1) 근사 — 실분포 흉내(β=row0, g=row1 교차).
        (0..t * h_v * 2)
            .map(|i| if i % 2 == 0 { 0.05 + lcg().abs() * 0.05 } else { 0.9 - lcg().abs() * 0.05 })
            .collect(),
    );
    acc.frame_write(hn, &q)?;
    acc.frame_write(hk, &k)?;
    acc.frame_write(hv, &v)?;
    acc.frame_write(hbg, &bg)?;
    let zeros = vec![0.0f32; h_v * d * d];
    acc.frame_write(ha, &zeros)?;
    acc.frame_write(hb, &zeros)?;

    // (가) 1회 t
    acc.frame_begin(t);
    acc.frame_gdn_ar(hn, hk, hv, hbg, ha, hoa, 1, h_k, h_v, d)?;
    // (나) t/per회 per행 밴드
    for c in 0..(t / per) {
        let off = c * per;
        let (n2, k2, v2, b2, o2) = (
            acc.frame_slice(hn, off * k_len, per * k_len)?,
            acc.frame_slice(hk, off * k_len, per * k_len)?,
            acc.frame_slice(hv, off * v_len, per * v_len)?,
            acc.frame_slice(hbg, off * h_v * 2, per * h_v * 2)?,
            // ★ 출력 밴드는 hob 쪽에 — hoa에 쓰고 hob을 읽는 배선 오류를 피한다.
            acc.frame_slice(hob, off * v_len, per * v_len)?,
        );
        let _ = o2;
        acc.frame_begin(per);
        acc.frame_gdn_ar(n2, k2, v2, b2, hb, o2, 1, h_k, h_v, d)?;
    }

    let mut sa = vec![0.0f32; h_v * d * d];
    let mut sb = vec![0.0f32; h_v * d * d];
    let mut oa = vec![0.0f32; t * v_len];
    let mut ob = vec![0.0f32; t * v_len];
    acc.frame_read(ha, &mut sa)?;
    acc.frame_read(hb, &mut sb)?;
    acc.frame_read(hoa, &mut oa)?;
    acc.frame_read(hob, &mut ob)?;
    let mx = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    // 첫 불일치 요소 — 행·열로 환산해 밴드 경계 문제인지 난수 반올림인지 가른다.
    let first = oa
        .iter()
        .zip(&ob)
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, (x, y))| format!("첫 불일치 out[{i}] (행 {} 열 {}) {x:.6e} vs {y:.6e}", i / v_len, i % v_len));
    let nbad = oa.iter().zip(&ob).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    Ok(format!(
        "[gdn-ar-inv] t={t} per={per}: 상태 maxΔ={:.3e} · 출력 maxΔ={:.3e} ({nbad}/{}) {} → {}",
        mx(&sa, &sb),
        mx(&oa, &ob),
        oa.len(),
        first.unwrap_or_else(|| "-".into()),
        if mx(&sa, &sb) == 0.0 && mx(&oa, &ob) == 0.0 { "t 불변 ✓" } else { "t 의존 ✗" }
    ))
}

/// GDN conv(t=1 커널의 청크판 + 별도 링 갱신)의 **t 불변성** 프로브.
/// (가) 1회 t=T vs (나) T/per회 t=per 밴드 — 링 상태와 출력을 대조한다.
pub fn gdn_conv_invariance(t: usize, per: usize, ch: usize, k: usize) -> Result<String, String> {
    use super::q4acc::Q4Acc;
    use llm170_core::matmul::{FrameHost, FrameOp, FrameState};
    if per == 0 || t % per != 0 || t == per || per < k - 1 {
        return Err(format!("인자: t={t} per={per} k={k}"));
    }
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let acc = Q4Acc::new()?;
    let hq = acc.frame_alloc(t * ch)?;
    let hc = acc.frame_alloc(ch * k)?;
    let hs = acc.frame_alloc((k - 1) * ch)?;
    let hsa = acc.frame_alloc((k - 1) * ch)?;
    let hoa = acc.frame_alloc(t * ch)?;
    let hob = acc.frame_alloc(t * ch)?;
    let q: Vec<f32> = (0..t * ch).map(|_| lcg()).collect();
    let cw: Vec<f32> = (0..ch * k).map(|_| lcg()).collect();
    acc.frame_write(hq, &q)?;
    acc.frame_write(hc, &cw)?;
    let zeros = vec![0.0f32; (k - 1) * ch];
    acc.frame_write(hs, &zeros)?;
    acc.frame_write(hsa, &zeros)?;
    acc.frame_write(hob, &zeros.iter().cycle().take(t * ch).copied().collect::<Vec<f32>>())?;
    // (가) 1회 t
    acc.frame_begin(t);
    acc.frame_op(&FrameOp::GdnConv { qkv: hq, cw: hc, state: hs, out: hoa, ch, k, t_len: t })?;
    // (나) 밴드 — 상태는 hsa로 이월
    for c in 0..(t / per) {
        let off = c * per;
        let (q2, o2) = (
            acc.frame_slice(hq, off * ch, per * ch)?,
            acc.frame_slice(hob, off * ch, per * ch)?,
        );
        acc.frame_begin(per);
        acc.frame_op(&FrameOp::GdnConv { qkv: q2, cw: hc, state: hsa, out: o2, ch, k, t_len: per })?;
    }
    let mut oa = vec![0.0f32; t * ch];
    let mut ob = vec![0.0f32; t * ch];
    let mut sa = vec![0.0f32; (k - 1) * ch];
    let mut sb = vec![0.0f32; (k - 1) * ch];
    acc.frame_read(hoa, &mut oa)?;
    acc.frame_read(hob, &mut ob)?;
    acc.frame_read(hs, &mut sa)?;
    acc.frame_read(hsa, &mut sb)?;
    let mx = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let nbad = oa.iter().zip(&ob).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let first = oa
        .iter()
        .zip(&ob)
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, _)| format!("첫 불일치 out[{i}] (행 {} 열 {})", i / ch, i % ch));
    Ok(format!(
        "[gdn-conv-inv] t={t} per={per} ch={ch} k={k}: 링 maxΔ={:.3e} · 출력 maxΔ={:.3e} ({nbad}/{}) {} → {}",
        mx(&sa, &sb),
        mx(&oa, &ob),
        oa.len(),
        first.unwrap_or_else(|| "-".into()),
        if mx(&sa, &sb) == 0.0 && mx(&oa, &ob) == 0.0 { "t 불변 ✓" } else { "t 의존 ✗" }
    ))
}
