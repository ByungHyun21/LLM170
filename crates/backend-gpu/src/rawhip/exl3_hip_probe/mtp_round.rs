//! EXL3 hip 프로브 mtp_round (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-mtp-round <dir> <tok> <rounds>` — MTP 라운드 경제성·정합 실측.
/// 구조(무롤백 v1): 상태=직전 확정 토큰 pp(타깃 자신의 예측). 라운드 =
///   드래프트 d=mtp_draft → 검증 배치 [pp, d] → d==argmax(row0)면 pp+d 확정,
///   아니면 pp+corr 확정 후 교정 T=1 배치(상태 정렬). 다음 pp=row_last argmax.
pub fn hip_mtp_round(dir: &str, tok: u32, rounds: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64, 1024)?;
    // 기준: 순차 greedy 2*rounds+4 토큰(교차 검증용)
    let n_ref = 2 * rounds + 4;
    let mut ref_toks = Vec::new();
    {
        let mut tk = tok;
        let mut ht = Vec::new();
        for _ in 0..n_ref {
            let row = dec.embed_row_host(tk);
            let (lg, h) = dec.forward(&row)?;
            ht = h;
            let am = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            ref_toks.push(am);
            tk = am;
        }
        let _ = ht;
    }
    // MTP 라운드 — 상태 리셋 필요: 새 디코더(순차와 동일 출발).
    drop(dec);
    let mut dec2 = Exl3HipDecoder::load(dir, 64, 1024)?;
    let mut h: Vec<f32>;
    {
        let row0 = dec2.embed_row_host(tok);
        let (lg0, h0) = dec2.forward_batch_with_mtp(&[row0], &[tok])?;
        h = h0;
        let _ = lg0;
    }
    let mut pp = ref_toks[0]; // 타깃 1스텝 후 자신의 예측(상태는 tok 처리까지)
    let mut out_toks = Vec::new();
    let (mut acc, mut tot) = (0usize, 0usize);
    let t0 = std::time::Instant::now();
    for _ in 0..rounds {
        // 드래프트(상태 = pp 직전? 규약: mtp_draft(tok=직전 확정, h, pos) — pos는 pp까지)
        let pos_now = dec2.pos;
        let d = dec2.mtp_draft_gpu(pp, &h, pos_now)?; // pos: pp가 처리된 뒤 위치
        // 검증 배치 [pp, d] + mtp KV 적립 훅
        let rows = vec![dec2.embed_row_host(pp), dec2.embed_row_host(d)];
        let (lgs, hnew) = dec2.forward_batch_with_mtp(&rows, &[pp, d])?;
        let am0 = lgs[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let am1 = lgs[1]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        tot += 1;
        if am0 == d {
            acc += 1;
            out_toks.push(pp);
            out_toks.push(d);
            pp = am1;
            h = hnew;
        } else {
            // 거부: pp+corr 확정, 상태는 [pp,d]까지 전진됨 → 교정 T=1로 corr 재처리
            let corr = am0;
            out_toks.push(pp);
            out_toks.push(corr);
            let rrow = dec2.embed_row_host(corr);
            let (lgc, hc) = dec2.forward_batch_with_mtp(&[rrow], &[corr])?;
            let amc = lgc[0]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            pp = amc;
            h = hc;
        }
    }
    let el = t0.elapsed().as_secs_f64();
    let tps = out_toks.len() as f64 / el;
    // 정합: out_toks가 순차 ref의 앞부분과 일치하는가(스펙은 배치 로짓 기준 — 대조는 참고)
    let mut agree = 0usize;
    for (i, ot) in out_toks.iter().enumerate() {
        if ref_toks.get(i) == Some(ot) {
            agree += 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "mtp-round: {rounds}라운드 {acc}/{tot} 수용 · {}토큰 {el:.2}s → {tps:.2} t/s · 순차 일치 {agree}/{}",
        out_toks.len(),
        out_toks.len()
    ))
}
// 마커 mr1
// 마커 mr4
// 마커 dg5
// 마커 dab
// 마커 d3q
// 마커 mrf
