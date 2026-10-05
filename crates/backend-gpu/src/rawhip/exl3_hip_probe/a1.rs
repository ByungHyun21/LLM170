//! EXL3 hip 프로브 a1 (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-a1 <dir> <tok> <steps>` — MTP 드래프트 a1 수용률 측정(vk exl3-mtp 재현):
/// 타깃 순차(+KV 훅) 매 스텝, mtp_draft_gpu(현 토큰, h_현토큰, pos) vs 타깃 실제 다음 토큰.
pub fn hip_mtp_a1(dir: &str, tok: u32, steps: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64, 1024)?;
    // 변형 A: 순차(gemv) 타깃 + 호스트 mtp 훅(vk exl3-mtp 동일 구조) —
    // h 클래스(배치 gemm2 h vs 순차 gemv h)가 a1 격차(0.44 vs 0.625) 원인인지 판별.
    let mut h_seq_store: Vec<Vec<f32>> = Vec::new();
    {
        let (mut hit_s, mut tot_s, mut cur_s) = (0usize, 0usize, tok);
        for _ in 0..steps {
            let row = dec.embed_row_host(cur_s);
            let (lg, h) = dec.forward(&row)?;
            let nxt = lg
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            h_seq_store.push(h.clone());
            let dl = dec.mtp_draft(cur_s, &h, dec.pos - 1)?;
            tot_s += 1;
            let am_d = dl
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if am_d == nxt {
                hit_s += 1;
            }
            cur_s = nxt;
        }
        eprintln!("  [a1seq] 순차경로 a1 = {hit_s}/{tot_s}");
        let _ = &h_seq_store;
    }
    let mut cur = tok;
    let (mut hit, mut tot, mut hit_h) = (0usize, 0usize, 0usize);
    let mut t_draft = 0f64;
    let t0 = std::time::Instant::now();
    for batch_i in 0..steps {
        let row = dec.embed_row_host(cur);
        let pos_before = dec.pos;
        let (lg, h) = dec.forward_batch_with_mtp(&[row], &[cur])?;
        if let Some(hs) = h_seq_store.get(batch_i) {
            let md = h
                .iter()
                .zip(hs)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "  [hmd] 스텝{batch_i} h_batch-vs-seq maxdiff={md:.3e} rms_h={:.3}",
                h.iter().map(|v| v * v).sum::<f32>().sqrt()
            );
        }
        let nxt = lg[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        // 드래프트: (현 토큰 cur, h_cur, pos_before) → 다음 예측 — GPU·호스트 동시 측정
        let td = std::time::Instant::now();
        let d = dec.mtp_draft_gpu(cur, &h, pos_before)?;
        t_draft += td.elapsed().as_secs_f64();
        let dh = dec.mtp_draft(cur, &h, pos_before)?;
        let am_h = dh
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        tot += 1;
        if d == nxt {
            hit += 1;
        }
        if am_h == nxt {
            hit_h += 1;
        }
        if tot <= 6 {
            eprintln!("  [a1dbg] 스텝{tot} gpu={d} host={am_h} target={nxt}");
        }
        cur = nxt;
    }
    let el = t0.elapsed().as_secs_f64();
    Ok(format!(
        "hip-a1: gpu {hit}/{tot} = {:.2} · host {hit_h}/{tot} = {:.2} · 타깃순차 {el:.2}s({:.2} t/s) · gpu드래프트 {t_draft:.3}s({:.1}ms/회)",
        hit as f64 / tot as f64,
        hit_h as f64 / tot as f64,
        steps as f64 / el,
        t_draft * 1e3 / steps as f64
    ))
}
// 마커 a1p
// 마커 da1
// 마커 da2
