//! EXL3 hip 프로브 mtp (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawvk::checks::TrellisResident;

/// `llm170 exl3-hip-mtp <dir> <tok>` — MTP 드래프트 모듈 격리 검증:
/// 합성 hidden(결정론 패턴)으로 hip gemv 경로 vs vk 참조 mtp_step 로짓 대조.
pub fn hip_mtp_check(dir: &str, tok: u32) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let h = 5120usize;
    let synth: Vec<f32> = (0..h).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    // 1단계: hip 단독(mtp 가중치만 사용)
    let mut dec = Exl3HipDecoder::load(dir, 0, 1024)?;
    // GPU 드래프트 A/B: 동일 입력으로 정합 + 시간(호스트 버전 기준).
    let tg0 = std::time::Instant::now();
    let d_gpu = dec.mtp_draft_gpu(tok, &synth, 0)?;
    let tg = tg0.elapsed().as_secs_f64() * 1e3;
    let th0 = std::time::Instant::now();
    let tl = dec.mtp_draft(tok, &synth, 0)?;
    let th = th0.elapsed().as_secs_f64() * 1e3;
    let am_h = tl
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    eprintln!("  [dab] gpu={d_gpu} host={am_h} · gpu {tg:.0}ms host {th:.0}ms");
    let tl = dec.mtp_draft(tok, &synth, 0)?;
    drop(dec);
    // 2단계: vk 참조 단독
    let mut tr = TrellisResident::load(dir)?;
    let mut seq = crate::rawvk::exl3::cpu::new_seq_state(tr.n_layers, 512);
    let wl = crate::rawvk::exl3::mtp::mtp_step(&mut tr, &mut seq, tok, &synth, 0, true)?;
    let mut md = 0f32;
    let (mut ga, mut wa) = (0usize, 0usize);
    for i in 0..tl.len() {
        md = md.max((tl[i] - wl.0[i]).abs());
        if tl[i] > tl[ga] {
            ga = i;
        }
        if wl.0[i] > wl.0[wa] {
            wa = i;
        }
    }
    Ok(format!(
        "hip-mtp tok{tok}: 로짓 maxdiff={md:.3e} argmax hip={ga} vk={wa} {}",
        if ga == wa { "일치" } else { "불일치" }
    ))
}
// 마커 mtpd
// 마커 mtpf
