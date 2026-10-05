//! EXL3 hip 프로브 tbench (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-tbench <dir> <tok> [T]` — 배치 forward T별 비용 상각 곡선.
pub fn hip_tbench(dir: &str, tok: u32, t_max: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let mut dec = Exl3HipDecoder::load(dir, 64, 1024)?;
    let mut out = String::new();
    for t in [1usize, 2, 4, 8, 16, 32, 64] {
        if t > t_max {
            break;
        }
        // 같은 토큰 반복 행(비용 측정 — 상태는 순차와 무관)
        let rows: Vec<Vec<f32>> = (0..t).map(|_| dec.embed_row_host(tok)).collect();
        // 워밍 1회 + 측정 3회 중앙값
        let _ = dec.forward_batch(&rows)?;
        let mut ts: Vec<f64> = Vec::new();
        let mut kt_report = String::new();
        for _ in 0..3 {
            // T=64 측정 1회에만 KTRACE — 커널별 GPU 시간 분해(plans/130 C1 진단).
            let tracing = t == 64;
            if tracing {
                crate::rawhip::ktrace::ktrace_on();
            }
            let t0 = std::time::Instant::now();
            let _ = dec.forward_batch(&rows)?;
            let el = t0.elapsed().as_secs_f64() * 1e3;
            if tracing {
                kt_report = crate::rawhip::ktrace::ktrace_dump();
            }
            ts.push(el);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ts[1];
        let per_tok = med / t as f64;
        out.push_str(&format!(
            "T={t}: {med:.1}ms 배치 · {per_tok:.0}ms/토큰({:.2} t/s) | ",
            1000.0 / per_tok
        ));
        if !kt_report.is_empty() {
            out.push_str(&format!("\n[KTRACE T=64]\n{kt_report}\n"));
        }
    }
    Ok(out)
}
// 마커 tsb
// 마커 tsb2
// 마커 abh
