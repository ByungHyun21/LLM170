//! EXL3 hip 프로브 decode (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawvk::checks::TrellisResident;

pub fn hip_decode_check(dir: &str, tok0: u32, lim_layers: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    // 동결 방지(2026-10-04 사고 원칙): 한 시점에 한 모델만 상주.
    // 1단계: hip 디코더(임베딩 포함) 단독 — greedy 4스텝.
    let mut dec = Exl3HipDecoder::load(dir, lim_layers, 1024)?;
    let t0f = std::time::Instant::now();
    let mut tok = tok0;
    let mut hip_toks = Vec::new();
    let mut first: Option<Vec<f32>> = None;
    for i in 0..4 {
        // 1스텝 KTRACE(3번째) — 디코드 토큰의 커널별 GPU 시간 분해(plans/130).
        let tracing = i == 2;
        if tracing {
            crate::rawhip::ktrace::ktrace_on();
        }
        let lg = dec.forward_tok(tok)?;
        if tracing {
            eprintln!("[KTRACE decode 1tok]\n{}", crate::rawhip::ktrace::ktrace_dump());
        }
        if first.is_none() {
            first = Some(lg.clone());
        }
        let am = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        hip_toks.push(am as u32);
        tok = am as u32;
    }
    let fwd_ms = t0f.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "  [tgdbg] 4스텝 forward {fwd_ms:.0}ms → {:.2} t/s(셔틀 포함)",
        4000.0 / fwd_ms
    );
    let got = first.unwrap_or_default();
    drop(dec);
    // 2단계: vk 참조 단독 재로드.
    let mut tr = TrellisResident::load(dir)?;
    let mut seq = crate::rawvk::exl3::cpu::new_seq_state(tr.n_layers, 512);
    let want = crate::rawvk::exl3::decode::decode_step(&mut tr, &mut seq, tok0)?;
    let mut vt = Vec::new();
    let mut vtok = want
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0) as u32;
    for _ in 0..3 {
        vt.push(vtok);
        let lg2 = crate::rawvk::exl3::decode::decode_step(&mut tr, &mut seq, vtok)?;
        vtok = lg2
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as u32;
    }
    vt.push(vtok);
    eprintln!("  [vk-greedy] 4토큰 {vt:?}");
    eprintln!("  [hip-greedy] 4토큰 {hip_toks:?}");
    let mut md = 0f32;
    let (mut ga, mut wa) = (0usize, 0usize);
    for i in 0..got.len() {
        md = md.max((got[i] - want[i]).abs());
        if got[i] > got[ga] {
            ga = i;
        }
        if want[i] > want[wa] {
            wa = i;
        }
    }
    Ok(format!(
        "hip-decode tok{tok0}: 로짓 maxdiff={md:.3e} argmax hip={ga} vk={wa} {}",
        if ga == wa { "일치" } else { "불일치" }
    ))
}
// 마커 wab
// 마커 wab2
