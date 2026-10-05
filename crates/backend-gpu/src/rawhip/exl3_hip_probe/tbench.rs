//! EXL3 hip 프로브 tbench (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-tbench <dir> <tok> [T]` — 배치 forward T별 비용 상각 곡선.
pub fn hip_tbench(dir: &str, tok: u32, t_max: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64, 1024)?;
    let mut out = String::new();
    for t in [1usize, 2, 4, 8, 16] {
        if t > t_max {
            break;
        }
        // 같은 토큰 반복 행(비용 측정 — 상태는 순차와 무관)
        let rows: Vec<Vec<f32>> = (0..t).map(|_| dec.embed_row_host(tok)).collect();
        // 워밍 1회 + 측정 3회 중앙값
        let _ = dec.forward_batch(&rows)?;
        let mut ts: Vec<f64> = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let _ = dec.forward_batch(&rows)?;
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ts[1];
        let per_tok = med / t as f64;
        out.push_str(&format!(
            "T={t}: {med:.1}ms 배치 · {per_tok:.0}ms/토큰({:.2} t/s) | ",
            1000.0 / per_tok
        ));
    }
    Ok(out)
}
// 마커 tsb
// 마커 tsb2
// 마커 abh
