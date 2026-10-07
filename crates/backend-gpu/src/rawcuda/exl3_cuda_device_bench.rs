//! S10 프로파일 — 디바이스 상주 forward의 시간 분해.
//! [질문] 토큰당 11 tok/s는 무엇이 지배하는가? (a) GPU 커널 실행 (b) 전성
//! (c) CPU/GEMV 스테이징 오버헤드. nsys 없이도 대략을 잡기 위해 전체
//! 스텝 시간과, 디바이스 체인만 돌린 시간(상태 리셋 후)을 비교한다.
//!
//! [방법] 두 디코더를 준비하고
//!  (1) N스텝 전체 디바이스 경로 시간
//!  (2) 같은 N스텝에서 d2h 호출을 제외한 GPU-only 구간을 sync로 측정
//! 를 나눠 본다. 정밀 분해는 후속(커널별 타이 이벤트) — 여기서는
//! "왕복이 남아 있는가"라는 이분 판단에 초점을 둔다.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use std::time::Instant;

/// N스텝 디바이스 경로 실측. 반환 (초, 초/토큰).
pub fn time_device_steps(dec: &mut Exl3CudaDecoder, toks: &[u32], n: usize) -> Result<f64, String> {
    let _g = dec.cc.guard()?;
    let t0 = Instant::now();
    for i in 0..n {
        let tok = toks[i % toks.len()];
        let row = dec.embed_row_host(tok);
        let _ = dec.forward_device(0, &row)?;
    }
    Ok(t0.elapsed().as_secs_f64() / n as f64)
}

/// N스텝 호스트 스테이징 경로 실측(비교 기준).
pub fn time_host_steps(dec: &mut Exl3CudaDecoder, toks: &[u32], n: usize) -> Result<f64, String> {
    let _g = dec.cc.guard()?;
    let t0 = Instant::now();
    for i in 0..n {
        let tok = toks[i % toks.len()];
        let row = dec.embed_row_host(tok);
        let _ = dec.forward(0, &row)?;
    }
    Ok(t0.elapsed().as_secs_f64() / n as f64)
}

/// 시간 비교 보고.
pub fn report(host_spt: f64, dev_spt: f64, dev_name: &str) -> String {
    let speedup = if dev_spt > 0.0 {
        host_spt / dev_spt
    } else {
        0.0
    };
    format!(
        "device: {dev_name} | exl3-cuda-s10 profile: host={host_spt:.3}s/tok ({:.2} tok/s) \
         device={dev_spt:.3}s/tok ({:.2} tok/s) speedup={speedup:.2}x",
        1.0 / host_spt,
        1.0 / dev_spt
    )
}
