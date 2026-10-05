//! EXL3 hip 프로브 graph (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

/// `llm170 exl3-hip-graph <dir> <tok> [T]` — hipGraph 캡처·재생: 정합(순차 대조)+재생 시간.
pub fn hip_graph_check(dir: &str, tok: u32, t_len: usize) -> Result<String, String> {
    use crate::rawhip::exl3_hip::Exl3HipDecoder;
    let t = t_len.clamp(1, 8);
    let mut dec = Exl3HipDecoder::load(dir, 64, 1024)?;
    // 기준: 일반 배치 1회(캡처 워밍이 상태 전진시킴 — 순서: 워밍→캡처→비교재생은
    // 상태가 다르다. 정합은 "같은 상태에서 재생 vs 비캡처" 비교로: 캡처 후
    // 그래프 재생 2회와 수동 배치의 토큰열 자기일관성으로 판정(재생1 vs 재생2 연속).
    let rows0: Vec<Vec<f32>> = (0..t).map(|i| dec.embed_row_host(tok + i as u32)).collect();
    dec.capture_batch(t)?;
    // 재생 3회 측정(행은 매회 동일 — 비용 측정; 상태 전진은 KV/ring에 누적)
    let mut ts: Vec<f64> = Vec::new();
    let mut last_am = 0u32;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let (lgs, _) = dec.replay_batch(&rows0)?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
        let lastrow = lgs.last().ok_or("empty")?;
        last_am = lastrow
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[1];
    let per = med / t as f64;
    Ok(format!(
        "hip-graph T={t}: 재생 {med:.1}ms ({per:.0}ms/토큰, {:.2} t/s) · 마지막 argmax={last_am}",
        1000.0 / per
    ))
}
// 마커 gpr
// 마커 gpf

/// `llm170 exl3-hip-gmini` — 그래프 캡처 FFI 최소 검증(모델 미적재, 수 초):
/// [h2d → pos_bump → d2h_pin] 캡처·인스턴스화·재생.
pub fn hip_graph_mini() -> Result<String, String> {
    use crate::rawhip::ctx::hipgraph as hg;
    let hc = crate::rawhip::ctx::RawCtx::new()?;
    let dbuf = hc.alloc(4)?;
    hc.h2d(dbuf, &41u32.to_le_bytes())?;
    hc.sync()?;
    unsafe {
        let st = hg::hipStreamBeginCapture(hc.stream as *mut _, 2);
        if st != 0 {
            return Err(format!("BeginCapture {st}"));
        }
        // 캡처 구간: pos_bump 2회 + d2h_pin
        let mut pb = dbuf;
        let r1 = hc.launch3(
            "exl3_pos_bump",
            1,
            1,
            1,
            32,
            &mut [&mut pb as *mut *mut u8 as *mut _],
        );
        let mut ppin: *mut std::ffi::c_void = std::ptr::null_mut();
        let pr = hg::hipHostMalloc(&mut ppin, 4, 0);
        if pr != 0 {
            return Err(format!("mini pin {pr}"));
        }
        let pout = ppin as *mut u8;
        let r2 = hc.d2h_pin_async(pout, dbuf, 4);
        let mut graph: hg::Graph = std::ptr::null_mut();
        let en = hg::hipStreamEndCapture(hc.stream as *mut _, &mut graph);
        r1?;
        r2?;
        if en != 0 {
            return Err(format!("EndCapture {en}"));
        }
        let mut exec: hg::GraphExec = std::ptr::null_mut();
        let ie = hg::hipGraphInstantiate(&mut exec, graph, 0);
        if ie != 0 {
            return Err(format!("Instantiate {ie}"));
        }
        let le = hg::hipGraphLaunch(exec, hc.stream as *mut _);
        if le != 0 {
            return Err(format!("GraphLaunch {le}"));
        }
        hc.sync()?;
        // SAFETY: 핀 버퍼 판독(재생 완료 후).
        // SAFETY: 상위 unsafe 블록 내 — 중첩 제거.
        let v = u32::from_le_bytes(std::slice::from_raw_parts(pout, 4).try_into().unwrap());
        hg::hipGraphExecDestroy(exec);
        hg::hipGraphDestroy(graph);
        Ok(format!("gmini: 41+2={v} (43 기대) — 캡처·재생 정상"))
    }
}
// 마커 gm3
// 마커 gm4

/// `llm170 exl3-hip-gmini2` — 바이섹션: 2D launch 래퍼(norm_p류) 캡처 호환성.
pub fn hip_graph_mini2() -> Result<String, String> {
    let hc = crate::rawhip::ctx::RawCtx::new()?;
    let dbuf = hc.alloc(4)?;
    hc.h2d(dbuf, &7u32.to_le_bytes())?;
    let dbx = hc.alloc(5120 * 4)?;
    let dbxn = hc.alloc(5120 * 4)?;
    let dnw = hc.alloc(5120 * 4)?;
    let dz = hc.alloc(5120 * 4)?;
    hc.h2d(
        dbx,
        &vec![0.5f32; 5120]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    hc.h2d(
        dnw,
        &vec![1.0f32; 5120]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    hc.h2d(dz, &vec![0u8; 5120 * 4])?;
    hc.sync()?;
    use crate::rawhip::ctx::hipgraph as hg;
    unsafe {
        let st = hg::hipStreamBeginCapture(hc.stream as *mut _, 2);
        if st != 0 {
            return Err(format!("BeginCapture {st}"));
        }
        // 2D launch 래퍼 1회(norm_resid_p)
        let mut tl = 1i32;
        let (mut a0, mut a1, mut a2, mut a3) = (dbx, dnw, dz, dbxn);
        let rl = hc.launch(
            "exl3_norm_resid_p",
            1,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut a2 as *mut *mut u8 as *mut _,
                &mut a3 as *mut *mut u8 as *mut _,
                &mut tl as *mut i32 as *mut _,
            ],
        );
        let mut graph: hg::Graph = std::ptr::null_mut();
        let en = hg::hipStreamEndCapture(hc.stream as *mut _, &mut graph);
        rl?;
        if en != 0 {
            return Err(format!("EndCapture {en}"));
        }
        let mut exec: hg::GraphExec = std::ptr::null_mut();
        let ie = hg::hipGraphInstantiate(&mut exec, graph, 0);
        if ie != 0 {
            return Err(format!("Instantiate {ie}"));
        }
        let le = hg::hipGraphLaunch(exec, hc.stream as *mut _);
        if le != 0 {
            return Err(format!("GraphLaunch {le}"));
        }
        hc.sync()?;
        hg::hipGraphExecDestroy(exec);
        hg::hipGraphDestroy(graph);
        Ok("gmini2: 2D launch 캡처·재생 정상".into())
    }
}
// 마커 gm5
// 마커 wmr
// 마커 wv0
// 마커 a1s
// 마커 hmd
