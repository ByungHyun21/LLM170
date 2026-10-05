//! EXL3 hip 프로브 mtp_round (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-mtp-round <dir> <tok> <rounds> [k]` — MTP 라운드 경제성·정합 실측.
/// 구조(무롤백 v2, plans/130 D1 k확장): 라운드 = 드래프트 k체인(드래프트 자체
/// hidden을 연속 전달) → 검증 배치 [pp, d1..dk](T=k+1) → 수용 보행(첫 불일치
/// j에서 corr=argmax(row_j) 확정 후 교정 T=1 배치로 상태 정렬). k=1은 종전
/// v1과 동치. h 규약: **post-final-norm**(batch pgh — plans/121 V-3 종결,
/// exllamav3 qwen3_5_mtp 규약, 수용 0.375→0.458 실측으로 V1/pre-add 기각).
pub fn hip_mtp_round(dir: &str, tok: u32, rounds: usize, k: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64, 1024)?;
    // 기준: 순차 greedy rounds+k+4 토큰(교차 검증용)
    let n_ref = rounds * (k + 1) + 4;
    let mut ref_toks = Vec::new();
    {
        let mut tk = tok;
        for _ in 0..n_ref {
            let lg = dec.forward_tok(tk)?;
            let am = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0) as u32;
            ref_toks.push(am);
            tk = am;
        }
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
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0) as u32
    };
    let t0 = std::time::Instant::now();
    for _ in 0..rounds {
        // 드래프트 k체인 — 드래프트 자체 hidden(dbab 완전 잔차)을 연속 전달.
        let pos_now = dec2.pos;
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut hcur = h.clone();
        let mut prev = pp;
        for i in 0..k {
            let (d, hd) = dec2.mtp_draft_gpu(prev, &hcur, pos_now + i as u32)?;
            drafts.push(d);
            prev = d;
            hcur = hd;
        }
        // 검증 배치 [pp, d1..dk] + mtp KV 적립 훅
        let mut toks = vec![pp];
        toks.extend_from_slice(&drafts);
        let rows: Vec<Vec<f32>> = toks.iter().map(|&t| dec2.embed_row_host(t)).collect();
        let (lgs, hnew) = dec2.forward_batch_with_mtp(&rows, &toks)?;
        out_toks.push(pp);
        let mut rejected = false;
        for j in 0..k {
            tot += 1;
            let amj = argmax(&lgs[j]);
            if amj == drafts[j] {
                acc += 1;
                out_toks.push(drafts[j]);
            } else {
                // 거부: corr=argmax(row_j) 확정(그리디 정확) — 상태는 d_{j+1}까지
                // 전진됨 → 교정 T=1 재처리로 정렬 후 다음 pp.
                out_toks.push(amj);
                let rrow = dec2.embed_row_host(amj);
                let (lgc, hc) = dec2.forward_batch_with_mtp(&[rrow], &[amj])?;
                pp = argmax(&lgc[0]);
                h = hc;
                rejected = true;
                break;
            }
        }
        if !rejected {
            // 전부 수용 — row_k(마지막 드래프트 후)가 다음 pp.
            pp = argmax(&lgs[k]);
            h = hnew;
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
        "mtp-round k={k}: {rounds}라운드 드래프트 수용 {acc}/{tot} · {}토큰 {el:.2}s → {tps:.2} t/s · 순차 일치 {agree}/{}",
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
