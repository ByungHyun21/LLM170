//! EXL3 hip 프로브 hcmp (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-hcmp <dir> <tok> <steps>` — 같은 토큰 스트림에서 순차 vs 배치 h 쌍 비교.
/// maxdiff ≈1e-2 → 산술 클래스(트레이드오프), 크면 배치-h 결함(수리 가능).
pub fn hip_h_pair(dir: &str, tok: u32, steps: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dseq = Exl3HipDecoder::load(dir, 64, 1024)?;
    dseq.dbg_hcurve = true;
    // 순차 h·다음토큰 수집
    let mut toks = vec![tok];
    let mut hs: Vec<Vec<f32>> = Vec::new();
    for i in 0..steps {
        let row = dseq.embed_row_host(toks[i]);
        let (lg, h) = dseq.forward(&row)?;
        hs.push(h);
        let nxt = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j as u32)
            .unwrap_or(0);
        toks.push(nxt);
    }
    let seq_curve = std::mem::take(&mut dseq.hcurve);
    drop(dseq);
    // 배치 디코더로 같은 스트림 T=1씩(문맥 동일)
    let mut dbat = Exl3HipDecoder::load(dir, 64, 1024)?;
    dbat.dbg_hcurve = true;
    // 클린 배치 a1 — 오염 없는 배치 루프 자체 수용률(기존 0.25-0.44는 순차 루프
    // 상태 오염 후 측정이라 무효 가능성).
    {
        let (mut hit_b, mut tot_b, mut cur_b) = (0usize, 0usize, tok);
        for _ in 0..steps {
            let row = dbat.embed_row_host(cur_b);
            let pos_b = dbat.pos;
            let (lgb, hb2) = dbat.forward_batch_with_mtp(&[row], &[cur_b])?;
            let nxt_b = lgb[0]
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(k, _)| k as u32)
                .unwrap_or(0);
            let d_b = dbat.mtp_draft_gpu(cur_b, &hb2, pos_b)?;
            tot_b += 1;
            if d_b == nxt_b {
                hit_b += 1;
            }
            cur_b = nxt_b;
            if tot_b == 1 {
                eprintln!(
                    "  [hcvd] seq={} bat={} — 곡선 비교 진입",
                    seq_curve.len(),
                    dbat.hcurve.len()
                );
            }
            // [수리 2026-10-04, plans/127 C] 종전 hcv는 seq "마지막" 4개(스텝 6의
            // 토큰) vs 배치 첫 스텝(토큰 1000)을 비교 — 서로 다른 토큰의 h 곡선
            // 비교로 "L1 시드 3.42e0" 전체가 아티팩트였다. 배치 스텝1 ↔ 순차
            // 스텝1(같은 토큰) 비교로 수정.
            let n4 = dbat.hcurve.len();
            let seq4 = &seq_curve[..n4.min(seq_curve.len())];
            if tot_b == 1 && !seq4.is_empty() && n4 == seq4.len() {
                for (k, (l, hs_cv)) in seq4.iter().enumerate() {
                    let (lb, hb_cv) = &dbat.hcurve[k];
                    let md = hs_cv
                        .iter()
                        .zip(hb_cv)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let rms = hs_cv.iter().map(|v| v * v).sum::<f32>().sqrt();
                    // 상관계수 + 오차-크기 관계: corr≈1·오차∝값 → 노이즈 증폭,
                    // 무상관 원소 존재 → 실결함(인덱싱/버퍼).
                    let n_e = hs_cv.len();
                    let (mut sa, mut sb, mut saa, mut sbb, mut sab) =
                        (0f64, 0f64, 0f64, 0f64, 0f64);
                    let mut big_bad = 0usize; // |seq|<1 인데 |diff|>1 → 무상관 오염
                    for (a_, b_) in hs_cv.iter().zip(hb_cv) {
                        let (a_, b_) = (*a_ as f64, *b_ as f64);
                        sa += a_;
                        sb += b_;
                        saa += a_ * a_;
                        sbb += b_ * b_;
                        sab += a_ * b_;
                        if a_.abs() < 1.0 && (a_ - b_).abs() > 1.0 {
                            big_bad += 1;
                        }
                    }
                    let cov = sab / n_e as f64 - (sa / n_e as f64) * (sb / n_e as f64);
                    let va = saa / n_e as f64 - (sa / n_e as f64).powi(2);
                    let vb = sbb / n_e as f64 - (sb / n_e as f64).powi(2);
                    let corr = cov / (va.sqrt() * vb.sqrt());
                    eprintln!(
                        "  [hcv] L{l}↔L{lb} maxdiff={md:.3e} rms={rms:.1} corr={corr:.6} 무상관오염={big_bad}"
                    );
                }
            }
        }
        eprintln!("  [a1bat] 클린 배치경로 a1 = {hit_b}/{tot_b}");
    }
    // [수리 2026-10-04, plans/127 C] 종전 h-pail은 a1bat 루프가 자체 greedy로
    // 진행한 상태(링/KV/pos)가 남은 dbat를 그대로 재사용 — 순차 toks 스트림과
    // 위치가 어긋나 1.07e2 "계통 오차"의 상당분이 상태 비정렬 아티팩트였다.
    // 신규 디코더로 순차와 동일 토큰·동일 위치 진행으로 교체.
    drop(dbat);
    let mut dbat2 = Exl3HipDecoder::load(dir, 64, 1024)?;
    let mut mds = Vec::new();
    for i in 0..steps {
        let row = dbat2.embed_row_host(toks[i]);
        let (_lg, hb) = dbat2.forward_batch(&[row])?;
        let md = hb
            .iter()
            .zip(&hs[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        mds.push(md);
        if i == 0 {
            // 희소 원소 국소화: 임계 초과 개수·상위 위반 위치의 모듈로 패턴(128=hadout 청크,
            // 64/16=gemm 타일 경계, 무주기=산술 경계).
            let diffs: Vec<(usize, f32)> = hb
                .iter()
                .zip(&hs[i])
                .enumerate()
                .map(|(j, (a, b))| (j, (a - b).abs()))
                .collect();
            let big: Vec<usize> = diffs
                .iter()
                .filter(|(_, d)| *d > 1.0)
                .map(|(j, _)| *j)
                .collect();
            eprintln!(
                "  [hhg] >1.0 오염 {}/5120개 · 상위 12: {:?}",
                big.len(),
                &big[..big.len().min(12)]
            );
            let m128 = big.iter().filter(|j| *j % 128 == 127).count();
            let m64 = big.iter().filter(|j| *j % 64 == 63).count();
            eprintln!("  [hhg] mod128==127: {m128}개 · mod64==63: {m64}개");
            let rms = hs[i].iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("  [hcmp] 스텝{i} maxdiff={md:.3e} rms={rms:.1}");
        }
    }
    let med = {
        mds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mds[mds.len() / 2]
    };
    Ok(format!(
        "h-pair: 중앙 maxdiff={med:.3e} — {}",
        if med < 0.05 {
            "f16급(산술 클래스)"
        } else {
            "계통 오차(배치-h 결함 의심)"
        }
    ))
}
// 마커 hcp
// 마커 hhg
// 마커 cba
// 마커 hrp
// 마커 hcx
// 마커 scv
// 마커 dcf
// 마커 s4
// 마커 cor
