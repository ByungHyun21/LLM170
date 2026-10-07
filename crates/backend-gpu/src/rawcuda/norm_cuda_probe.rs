//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: norm_run은 케이스마다 x·ab·nw를 재업로드한다(케이스 간 잔류 없음).
//! ② 형상은 전 선형 메타에서 자동 열거(t=1..tmax·혼합 krate·S0≠0):
//!    hidden 5120(27B)·2048(35B)·강분리 행 전 w + T=4 배치 — 전 선형 메타 열거는 실모델 krate 다양성까지 미확장(원장 기입).
//! ③ 캡처-재생 3방향: 합성 입력·실층 캡처·쌍 디코더 A/B.
//! ④ 종단 상태가 유일 불변량: 판정은 종단 값 maxdiff(중간 단계는 진단).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-04, 전체 원장은 exl3_cuda_probe.rs]
//! (i) 4.768e-7 · (ii) 0.000e0 · (iii) 0.000e0(코어 오라클 비트일치) ·
//! 음성대조 (a) L0 판독 6.613e0 · (b) eps 1e-5 6.719e-4 → NEG-DETECTED.
//! 임계 3e-6(NORM_THRESH).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::{Rng, gen_unif, maxdiff_nan};

/// 노름 값 maxdiff 임계(plans/124 §1 — hip 원장 2.861e-6 기준).
const NORM_THRESH: f32 = 3e-6;
// ── norm_resid 오라클(G3) — crates/core/src/ops.rs 줄 단위 미러 ──

/// 제곱합 — crates/core/src/ops.rs sq_sum(L11-31) 직이식: 32세그먼트
/// 병렬 친화 구조(세그먼트 f32 순차 누산 → 세그먼트 f64 결합).
/// 커널의 스레드별 적산+트리 리덕션과는 순서만 다르고 같은 계급 —
/// 차이가 임계 3e-6 내인 것이 norm_resid 정합 판정 자체다(plans/124 §5).
fn core_sq_sum(x: &[f32]) -> f64 {
    const SEG: usize = 32;
    let n = x.len();
    let chunk = n.div_ceil(SEG);
    let mut sum = 0.0f64;
    for u in 0..SEG {
        let lo = u * chunk;
        if lo >= n {
            break;
        }
        let hi = (lo + chunk).min(n);
        let mut part = 0.0f32;
        for &v in &x[lo..hi] {
            part += v * v;
        }
        sum += part as f64;
    }
    sum
}

/// rms_norm — crates/core/src/ops.rs rms_norm(L33-37) 직이식: f64
/// 평균+eps → sqrt → f32 역수 → v·scale·g. 노름 정합의 유일 기준
/// (plans/124 §6 수치 진실 계층: 커널 → core 참조 → 수학).
fn core_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = core_sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// norm_resid 오라클(§3.2 계약) — 행별: v = x+ab(원소순) →
/// xn = core_rms_norm(v, w_row, eps), 잔차 x' = v(제자리 가산 기록).
/// 반환 (x', xn) — 둘 다 값 maxdiff 판정한다(잔차는 비트동일 기대).
fn norm_resid_reference(
    x: &[f32],
    ab: &[f32],
    w_row: &[f32],
    eps: f32,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut vr = Vec::with_capacity(x.len());
    let mut xn = Vec::with_capacity(x.len());
    for (xr, ar) in x.chunks(hidden).zip(ab.chunks(hidden)) {
        let v: Vec<f32> = xr.iter().zip(ar).map(|(&a, &b)| a + b).collect();
        xn.extend_from_slice(&core_rms_norm(&v, w_row, eps));
        vr.extend_from_slice(&v);
    }
    (vr, xn)
}

// ── 노름 프로브 합성 자료(결정론 시드 — splitmix64 미러 재사용) ──

/// 노름 가중 행 결정론 생성 — lo..hi 균일. 행 강분리(결함 2호 가드)는
/// 호출측 lo/hi(부호 반전·진폭)로 만든다.
fn gen_w_row(hidden: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..hidden)
        .map(|_| (lo + rng.next_f64() * (hi - lo)) as f32)
        .collect()
}

/// nw 배열 바이트([rows][hidden] f32 LE — 모듈 set_norm_weights 규격).
fn nw_bytes(rows: &[Vec<f32>]) -> Vec<u8> {
    let mut b = Vec::with_capacity(rows.len() * rows[0].len() * 4);
    for r in rows {
        for v in r {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b
}

/// 1회 실행 + xn·잔차 양측 판정. 반환 (xn maxdiff, resid maxdiff, nan).
fn norm_run(
    dec: &mut Exl3CudaDecoder,
    w: usize,
    x: &[f32],
    ab: &[f32],
    w_rows: &[Vec<f32>],
) -> Result<(f32, f32, usize), String> {
    let (want_r, want_xn) = norm_resid_reference(x, ab, &w_rows[w], 1e-6, dec.hidden);
    let (got_r, got_xn) = dec.norm_resid_host(w, x, ab)?;
    let (md_xn, nan) = maxdiff_nan(&got_xn, &want_xn);
    let (md_r, nan_r) = maxdiff_nan(&got_r, &want_r);
    Ok((md_xn, md_r, nan + nan_r))
}

/// exl3-cuda-norm — plans/124 G3 §3.2 norm_resid 값 maxdiff 판정(임계
/// 3e-6). (i) 27B hidden=5120 다중행 w(말단 행 포함) · (ii) 35B
/// hidden=2048 · (iii) 행 강분리 nw 전 행 판정 + T=4 다중행 잔차.
/// 하나라도 FAIL이면 Err(→ CLI 비영).
pub fn cuda_norm_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (i) Qwen3.8-27B 형상 hidden=5120(config.json text_config 실측
    // 2026-10-04) · 130행 nw — w·5120 행 오프셋 계약, 말단 w=129 경계 포함.
    {
        dec.hidden = 5120;
        let rows: Vec<Vec<f32>> = (0..130)
            .map(|r| gen_w_row(5120, 0x5EED_0000_0000_1000u64 + r as u64, 0.8, 1.2))
            .collect();
        dec.set_norm_weights(&nw_bytes(&rows), rows.len())?;
        let x = gen_unif(5120, 0x5EED_0000_0000_1001, 0.2);
        let ab = gen_unif(5120, 0x5EED_0000_0000_1002, 0.2);
        for w in [1usize, 129] {
            let (md, mdr, nan) = norm_run(&mut dec, w, &x, &ab, &rows)?;
            let pass = md <= NORM_THRESH && mdr == 0.0 && nan == 0;
            println!(
                "device: {dev} | exl3-cuda-norm (i) 27B hidden=5120 rows=130 w={w}: xn maxdiff={md:.3e} resid maxdiff={mdr:.3e} nan={nan} | {}",
                if pass { "PASS" } else { "FAIL" }
            );
            report.push_str(&format!("(i,w{w}) {md:.3e}"));
            if !pass {
                fails.push(format!("(i,w{w}) xn={md:.3e} resid={mdr:.3e} nan={nan}"));
            }
        }
    }

    // (ii) Qwen3.6-35B-A3B 형상 hidden=2048(config.json text_config 실측
    // 2026-10-04 — G2 (ii)와 동일 출처) · 64행, w=41.
    {
        dec.hidden = 2048;
        let rows: Vec<Vec<f32>> = (0..64)
            .map(|r| gen_w_row(2048, 0x5EED_0000_0000_2000u64 + r as u64, 0.8, 1.2))
            .collect();
        dec.set_norm_weights(&nw_bytes(&rows), rows.len())?;
        let x = gen_unif(2048, 0x5EED_0000_0000_2001, 0.2);
        let ab = gen_unif(2048, 0x5EED_0000_0000_2002, 0.2);
        let (md, mdr, nan) = norm_run(&mut dec, 41, &x, &ab, &rows)?;
        let pass = md <= NORM_THRESH && mdr == 0.0 && nan == 0;
        println!(
            "device: {dev} | exl3-cuda-norm (ii) 35B hidden=2048 rows=64 w=41: xn maxdiff={md:.3e} resid maxdiff={mdr:.3e} nan={nan} | {}",
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (ii) {md:.3e}"));
        if !pass {
            fails.push(format!("(ii) xn={md:.3e} resid={mdr:.3e} nan={nan}"));
        }
    }

    // (iii) 결함 2호 가드 — 6행 nw, 행 2를 강분리(부호 반전+진폭 상향:
    // |w2-w0|≈2 → L0 판독 시 xn 오차 O(1) — 임계와 6자리수 이상 이격).
    // 전 행 w=0..5 판정(오프셋 w·5120이 행마다 정확히 걸리는지) +
    // T=4 다중행 잔차(그리드 t축 · 행별 독립 rms) w=2.
    {
        dec.hidden = 5120;
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|r| {
                if r == 2 {
                    gen_w_row(5120, 0x5EED_0000_0000_3002, -1.8, -1.2)
                } else {
                    gen_w_row(5120, 0x5EED_0000_0000_3000u64 + r as u64, 0.8, 1.2)
                }
            })
            .collect();
        dec.set_norm_weights(&nw_bytes(&rows), rows.len())?;
        let x = gen_unif(5120, 0x5EED_0000_0000_3010, 0.2);
        let ab = gen_unif(5120, 0x5EED_0000_0000_3011, 0.2);
        let mut worst = 0f32;
        for w in 0..6usize {
            let (md, mdr, nan) = norm_run(&mut dec, w, &x, &ab, &rows)?;
            worst = worst.max(md);
            let pass = md <= NORM_THRESH && mdr == 0.0 && nan == 0;
            println!(
                "device: {dev} | exl3-cuda-norm (iii) strong-row nw w={w}: xn maxdiff={md:.3e} resid maxdiff={mdr:.3e} nan={nan} | {}",
                if pass { "PASS" } else { "FAIL" }
            );
            if !pass {
                fails.push(format!("(iii,w{w}) xn={md:.3e} resid={mdr:.3e} nan={nan}"));
            }
        }
        report.push_str(&format!(" · (iii rows) worst={worst:.3e}"));
        let x4: Vec<f32> = (0..4)
            .flat_map(|t| gen_unif(5120, 0x5EED_0000_0000_3020u64 + t, 0.2))
            .collect();
        let ab4: Vec<f32> = (0..4)
            .flat_map(|t| gen_unif(5120, 0x5EED_0000_0000_3030u64 + t, 0.2))
            .collect();
        let (md, mdr, nan) = norm_run(&mut dec, 2, &x4, &ab4, &rows)?;
        let pass = md <= NORM_THRESH && mdr == 0.0 && nan == 0;
        println!(
            "device: {dev} | exl3-cuda-norm (iii) T=4 multi-row w=2: xn maxdiff={md:.3e} resid maxdiff={mdr:.3e} nan={nan} | {}",
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (iii T=4) {md:.3e}"));
        if !pass {
            fails.push(format!("(iii T=4) xn={md:.3e} resid={mdr:.3e} nan={nan}"));
        }
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | {report} | ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-norm 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// exl3-cuda-norm-neg — 음성대조 2종(원장 17호: 검증 계기도 스스로 검증).
/// (a) 결함 2호 재현: 모듈 w=0 호출(=전 행 L0 판독 출력)을 w=2 정답
///     오라클과 비교 — L0 고정 구현이면 (iii)의 w=2 판정이 정확히 이
///     maxdiff로 FAIL한다(잡아냄이 증명된다).
/// (b) eps 1e-5 오라클 주입: 모듈(eps=1e-6) w=2 출력과 비교 — eps
///     계약 위반 산출과의 maxdiff가 임계를 넘는다(탐지 됨이 증명된다).
/// 양쪽 모두 초과 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
/// (iii)과 동일 시드 재생성 — 결정론 재현성 유지.
pub fn cuda_norm_negative_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    dec.hidden = 5120;
    let rows: Vec<Vec<f32>> = (0..6)
        .map(|r| {
            if r == 2 {
                gen_w_row(5120, 0x5EED_0000_0000_3002, -1.8, -1.2)
            } else {
                gen_w_row(5120, 0x5EED_0000_0000_3000u64 + r as u64, 0.8, 1.2)
            }
        })
        .collect();
    dec.set_norm_weights(&nw_bytes(&rows), rows.len())?;
    let x = gen_unif(5120, 0x5EED_0000_0000_3010, 0.2);
    let ab = gen_unif(5120, 0x5EED_0000_0000_3011, 0.2);

    // (a) L0 판독 출력(모듈 실경로 w=0) vs w=2 정답 오라클.
    let (_, want_xn) = norm_resid_reference(&x, &ab, &rows[2], 1e-6, 5120);
    let (_, got0_xn) = dec.norm_resid_host(0, &x, &ab)?;
    let (md_a, nan_a) = maxdiff_nan(&got0_xn, &want_xn);
    println!(
        "device: {dev} | exl3-cuda-norm (iva) negative control row0-read vs w=2 oracle: maxdiff={md_a:.3e} nan={nan_a} | FAIL(expected)"
    );
    let det_a = md_a > NORM_THRESH;

    // (b) eps 1e-5 오라클 vs 모듈(eps=1e-6) w=2 출력.
    let (_, got2_xn) = dec.norm_resid_host(2, &x, &ab)?;
    let (_, want_xn_e5) = norm_resid_reference(&x, &ab, &rows[2], 1e-5, 5120);
    let (md_b, nan_b) = maxdiff_nan(&got2_xn, &want_xn_e5);
    println!(
        "device: {dev} | exl3-cuda-norm (ivb) negative control eps=1e-5 oracle vs kernel eps=1e-6: maxdiff={md_b:.3e} nan={nan_b} | FAIL(expected)"
    );
    let det_b = md_b > NORM_THRESH;

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) maxdiff={md_a:.3e} (b) maxdiff={md_b:.3e} > {NORM_THRESH:.0e} — 검증계기 정상(결함 2호·eps 편차 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={md_a:.3e} (b)={md_b:.3e} <= {NORM_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
