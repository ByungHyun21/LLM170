//! S10 격리 프로브 — 동일 입력에 대한 단일 GEMV만 두 경로로 비교.
//!
//! [왜 이 파일이 따로 있나] forward 비교는 체인 전체가 얽혀 있어 "어느
//! 단계가 틀렸는지"를 알려주지 않는다(현상: 어텐션 입력에서 갈리는데 그
//! 원인은 3단계 앞 GDN일 수 있다). 그래서 **입력이 100% 같은 단일 GEMV
//! 2회**만 비교한다. 여기서 값이 다르면 GEMV 체인 자체의 문제이고,
//! 같으면 forward 배선 문제다.
//!
//! [주의] 두 디코더의 dx가 서로 다른 버퍼여야 한다. 같은 포인터를 쓰면
//! 두 호출이 같은 버퍼를 공유해 비교가 무의미해진다 — load 두 벌이 각자
//! 자기 버퍼를 갖는다.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;

/// 동일 x에 대해 host gemv_host와 device gemv_dev 결과 비교.
/// 반환 (maxdiff, shape (k,n)).
pub fn single_gemv_compare(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    key: &str,
    x: &[f32],
) -> Result<(f32, (usize, usize)), String> {
    let yh = host.gemv_host(key, x)?;
    let (k, n, _) = dev
        .lin_shape(key)
        .ok_or_else(|| format!("선형 없음: {key}"))?;
    // 디바이스 경로: x를 dx에 올린 뒤 gemv_dev → dyb를 읽는다.
    {
        let _g = dev.cc.guard()?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이 일치).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        dev.cc.h2d(dev.dx, xb)?;
        dev.prewarm_chain_bufs()?;
        let p = dev.gemv_dev(key, dev.dx)?;
        let mut buf = vec![0u8; n * 4];
        dev.cc.d2h(&mut buf, p)?;
        dev.cc.sync()?;
        let yd = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n) };
        let md = yh
            .iter()
            .zip(yd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        Ok((md, (k, n)))
    }
}

/// 여러 키를 순서대로 비교 — 첫 불일치 지점을 준다.
pub fn gemv_isolation(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    x: &[f32],
    keys: &[String],
) -> Result<String, String> {
    let dev_name = dev.device_name().to_string();
    let mut rows = String::new();
    let mut first_bad = None;
    for key in keys {
        let (md, (k, n)) = single_gemv_compare(host, dev, key, x)?;
        rows.push_str(&format!("{key}(k{k}n{n})={md:.1e} "));
        if md > 1e-3 && first_bad.is_none() {
            first_bad = Some(key.clone());
        }
    }
    Ok(format!(
        "device: {dev_name} | exl3-cuda-s10 gemv-isolation: {rows}| first_bad={:?}",
        first_bad
    ))
}
