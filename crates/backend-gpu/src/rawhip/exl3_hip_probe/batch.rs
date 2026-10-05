//! EXL3 hip 프로브 batch (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-batch <dir> <tok> [T]` — 배치 forward(프리필/검증 경로) 정합:
/// T행 임베딩으로 forward_batch → 행별 argmax를 순차 디코드와 대조.
pub fn hip_batch_check(dir: &str, tok: u32, t_len: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let t = t_len.clamp(1, 8);
    let mut dec = Exl3HipDecoder::load(dir, dec_layers_default(dir), 1024)?;
    // 1) 순차 greedy T+1스텝(기준)
    let mut seq_toks = Vec::new();
    let mut seq_lgs: Vec<Vec<f32>> = Vec::new();
    let mut tk = tok;
    for _ in 0..=t {
        let lg = dec.forward_tok(tk)?;
        let am = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        seq_toks.push(am);
        seq_lgs.push(lg);
        tk = am;
    }
    // htrace 캡처(drop 전) — 배치 대조용(plans/128 P1 선행 국소화).
    let seq_trace = std::mem::take(&mut dec.htrace);
    let seq_dou = std::mem::take(&mut dec.atrace_dou);
    let seq_dab = std::mem::take(&mut dec.atrace_dab);
    let seq_qh = std::mem::take(&mut dec.atrace_qh);
    let seq_k = std::mem::take(&mut dec.atrace_k);
    let seq_v = std::mem::take(&mut dec.atrace_v);
    let seq_g = std::mem::take(&mut dec.atrace_g);
    let seq_xn = std::mem::take(&mut dec.atrace_xn);
    drop(dec);
    // 2) 배치: [tok, s1..st] 행 — 마지막 행의 argmax가 순차 t+1번째와 일치해야.
    let rows_toks: Vec<u32> = std::iter::once(tok)
        .chain(seq_toks.iter().take(t).copied())
        .collect();
    let mut dec2 = Exl3HipDecoder::load(dir, dec_layers_default(dir), 1024)?;
    dec2.dbg_layers = true;
    let mut rows = Vec::with_capacity(rows_toks.len());
    for rt in &rows_toks {
        rows.push(dec2.embed_row_host(*rt));
    }
    let (lgs, _) = dec2.forward_batch(&rows)?;
    // htrace 전층 대조 — 순차[s*64+il][0] vs 배치[il][s], 첫 발산층 국소화.
    if !seq_trace.is_empty() && !dec2.htrace.is_empty() {
        let steps = seq_trace.len() / 64;
        eprintln!(
            "  [htr] 순차 스텝={steps} · 배치 층수={}",
            dec2.htrace.len()
        );
        let mut first: Option<(usize, f32)> = None;
        let mut per_il_max = vec![0f32; dec2.htrace.len()];
        for s in 0..steps {
            for (il, brows) in dec2.htrace.iter().enumerate() {
                let Some(srow) = seq_trace.get(s * 64 + il).and_then(|v| v.first()) else {
                    continue;
                };
                let Some(brow) = brows.get(s) else { continue };
                let mut md = 0f32;
                for (a, b) in srow.iter().zip(brow.iter()) {
                    md = md.max((a - b).abs());
                }
                per_il_max[il] = per_il_max[il].max(md);
                if first.is_none() && md > 5e-2 {
                    first = Some((il, md));
                }
            }
        }
        for (il, md) in per_il_max.iter().enumerate() {
            if *md > 1e-3 {
                eprintln!("  [htr] L{il} 행별 maxdiff 최대={md:.3e}");
            }
        }
        match first {
            Some((il, md)) => eprintln!(
                "  [htr] 첫 발산층 L{il} (md={md:.3e} > 5e-2) — 이 층의 입력은 일치, 출력부터 발산"
            ),
            None => eprintln!("  [htr] 전층 5e-2 내 — 잔차 스트림 무결"),
        }
    }
    // atrace(il==3) — dou(fwd3s출력)·dab(o_proj출력) 행별 대조:
    // dou가 이미 발산하면 prep/fwd3s 배치 하네스, dou 일치·dab 발산이면 had16/gemm2.
    if !seq_dou.is_empty() && !dec2.atrace_dou.is_empty() {
        let cmp = |name: &str, s: &[Vec<f32>], b: &[Vec<f32>]| {
            let mut worst = 0f32;
            let mut worst_r = 0usize;
            for (r, (sr, br)) in s.iter().zip(b.iter()).enumerate() {
                let mut md = 0f32;
                for (a, c) in sr.iter().zip(br.iter()) {
                    md = md.max((a - c).abs());
                }
                if md > worst {
                    worst = md;
                    worst_r = r;
                }
            }
            eprintln!("  [atr] {name} 최악 행={worst_r} maxdiff={worst:.3e}");
        };
        cmp("dou(fwd3s출력)", &seq_dou, &dec2.atrace_dou);
        cmp("dab(o_proj출력)", &seq_dab, &dec2.atrace_dab);
        cmp("qh(prep출력)", &seq_qh, &dec2.atrace_qh);
        cmp("k(KV행)", &seq_k, &dec2.atrace_k);
        cmp("v(KV행)", &seq_v, &dec2.atrace_v);
        cmp("g(게이트)", &seq_g, &dec2.atrace_g);
        cmp("xn(노름출력=점근입력)", &seq_xn, &dec2.atrace_xn);
    }
    // 전 행 argmax — 첫 이탈 행 국소화(순차 기준과 행별 대조).
    let seq_ref: Vec<u32> = std::iter::once(tok).chain(seq_toks.clone()).collect();
    for (ri, lgr) in lgs.iter().enumerate() {
        let ra = lgr
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        eprintln!(
            "  [fbrow] 행{ri} argmax={ra} (순차 다음토큰 {}){}",
            seq_ref.get(ri + 1).copied().unwrap_or(0),
            if seq_ref.get(ri + 1) == Some(&ra) {
                " ✓"
            } else {
                " ✗"
            }
        );
    }
    // 행별 로짓 maxdiff — 플립이 f16 노이즈(≈1e-2)인지 계통(≥1e-1)인지 정량화.
    for (ri, lgr) in lgs.iter().enumerate() {
        if let Some(sl) = seq_lgs.get(ri) {
            let md = lgr
                .iter()
                .zip(sl)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("  [fbmd] 行{ri} maxdiff={md:.3e}");
        }
    }
    let last = lgs.last().ok_or("batch empty")?;
    let bam = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    let ok = bam == seq_toks[t];
    Ok(format!(
        "hip-batch T={t}: 순차 {:?} · 배치 마지막 argmax={bam} (기준 {}) — {}",
        seq_toks,
        seq_toks[t],
        if ok { "일치" } else { "불일치" }
    ))
}

fn dec_layers_default(_dir: &str) -> usize {
    64
}
// 마커 fb4
// 마커 fb7
// 마커 fb8
// 마커 fbd
// 마커 fbc
// 마커 fbr
// 마커 blp
// 마커 mdq
