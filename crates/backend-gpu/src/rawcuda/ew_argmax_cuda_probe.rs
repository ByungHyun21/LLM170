//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: ew·argmax 케이스는 각각 독립 버퍼로 실행(공유 스테이징 오염 없음 — ew_host·argmax_host_n가 케이스마다 재업로드).
//! ② 형상은 전 선형 메타에서 자동 열거(t=1..tmax·혼합 krate·S0≠0):
//!    n=17408(27B)·512(35B)·존/타이 고정 케이스 — 전 로짓 길이 248320 스윕은 음성대조(n=5120·n=1)로 경계만 커버(원장 기입).
//! ③ 캡처-재생 3방향: 합성 입력·실층 캡처·쌍 디코더 A/B.
//! ④ 종단 상태가 유일 불변량: 판정은 종단 값 maxdiff(중간 단계는 진단).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-04, 전체 원장은 아래 ew 배너]
//! ew (i) 0.000e0 · (ii) 0.000e0(비트동일 — 임계 1e-6) · argmax 전 타이
//! 케이스 exact · 음성대조 n=5120→4321 · n=1→0 → NEG-DETECTED(결함 8호).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::{Rng, gdn_exp_d, gen_unif, maxdiff_nan};

// ── ew(silu·mul)·argmax 프로브(G7) — plans/124 §1 "ew(silu·mul), argmax" ──
// 임계: ew 값 maxdiff ≤1e-6(값 판정)·argmax 토큰 exact-match(토큰 무결
// — 원장 §6 "argmax 일치를 정합 판정으로 사용 금지"는 값 모듈에 대한
// 규정이고 argmax 모듈 자체의 판정은 토큰 무일치 검출이다).
// 근거 소스(전부 워크트리 줄번호, 2026-10-04):
// - rawhip/kernels/src_exl3.hip exl3_ew L898-908 · exl3_argmax L911-934
//   (산술 원본 — assets/exl3_ew.cu가 1:1 직이식; exp 트윈 치환이 유일
//   차이).
// - rawhip/exl3_hip.rs ew·argmax 발사 L786-796(그리드 ceil(n/128)·블록
//   128)·L811-824 및 step_tok L481-512(단일 블록 1024) — 모듈층 래퍼
//   형상 원본.
// - crates/core/src/ops.rs silu L127-130(x/(1.0+exp_cr(−x)) — CPU 참조
//   공식. exp 구현은 아래 트윈 노선 치환: plans/124 §6 "hip은 참조
//   구현일 뿐 진실 아님").
// 트랜센던트 트윈 계약(G5/G6 노선 계승): exp는 gdn_exp_d(장치
// assets/exl3_ew.cu ew_exp_d와 동일 DAG — 리터럴까지 동일)을 그대로
// 재사용, -fmad=false 빌드. f32 add/div/mul 양측 IEEE 정확 반올림 →
// maxdiff 0 기대(임계 1e-6은 여유). argmax 오라클은 커널 선택 규칙
// (스트라이드 1024 클래스별 상향 스캔 "초과" 갱신 → 공유 트리 환원
// 동일값 낮은 tid 우선)의 정수 미러 — exact-match. n은 로짓 길이
// 248320(결함 8호: 행수 아님 — 음성대조가 잘못된 n을 잡는다).
//
// [실측 원장 2026-10-04, RTX 4070 SUPER(sm_89) — 검증 호스트]
//   (i) 27B n=17408: maxdiff 0.000e0 nan=0(오라클과 비트동일 — 임계 1e-6의
//   무한대 여유, 미러 계약 실증. 존 |g|≥10 5804개 · |g|≤1e-3 2904개 —
//   포화·근영역 실커버) · (ii) 35B n=512: 0.000e0(존 172/85).
//   (iii-a) 최대@17 + 근접타이@999·65432·248319: token 17=17 exact ·
//   (iii-b) 최대@248318 + 근접타이@3·4096·100000: 248318 exact ·
//   (iii-c) 동일값 타이 1·1024: 1024(트리 규칙 고정 실증 — 전역 첫
//   등장이 아닌 "낮은 tid 클래스 내 첫 등장"이 우승).
//   음성대조(결함 8호): (a) n=5120 → token 4321(미스캔 구간 준최대)
//   · (b) n=1 → token 0 — 양측 want=200000와 불일치, NEG-DETECTED
//   (비영 exit + 마커).
//   sm_80 자원 증거(cuobjdump --dump-resource-usage, 커밋 fatbin):
//   exl3_ew REG:16 SHARED:0 → 블록 128스레드=4와프, GA100 와프 상한
//   기준 12블록/SM 여유(레지스터 16×128=2K도 소형) — 27B 그리드
//   136블록은 70SM 미만의 소형 런치, T=1 레이턴시 도미넌트(ew는 순수
//   스트리밍이라 점유 확산 무의미) · exl3_argmax REG:31 SHARED:8192 →
//   단일 블록 1024스레드=32와프(GA100 64와프/SM의 절반 — 자원상
//   2블록/SM 가능하나 원본 1:1 계약상 1블록 발사, 병합형 재판정은
//   실측 장비 도착 후). 개발기 4070(sm_89)은 정합 검증 전용 — CMP
//   170HX 실측은 도착 후(plans/124 §0).

/// ew 값 maxdiff 임계(plans/124 G7 계약 — §1 ew·argmax 행 "토큰 무결"
/// 의 값 판정 보강).
const EW_THRESH: f32 = 1e-6;

/// ew 오라클 — 커널 산술 미러(비트동일 계약): x = −v(f32 부호, 정확)
/// → e = 트윈 exp(x)(f64 DAG → f32 RTNE) → v/(1+e) → ·u(f32 IEEE
/// add/div/mul — 커널 assets/exl3_ew.cu exl3_ew 본체와 순서까지 동일).
/// 공식 참조: crates/core/src/ops.rs silu L127-130.
fn ew_reference(g: &[f32], u: &[f32]) -> Vec<f32> {
    g.iter()
        .zip(u)
        .map(|(&v, &w)| {
            let x = -v;
            let e = gdn_exp_d(x as f64) as f32;
            (v / (1.0f32 + e)) * w
        })
        .collect()
}

/// argmax 오라클 — 커널 선택 규칙의 정수 미러(exact-match 계약):
/// tid별 잔여 클래스(i ≡ tid mod 1024) 상향 스캔(초기 −1e30, "초과"
/// 갱신 — 클래스 내 첫 등장 유지) → 트리 환원(동일값 낮은 tid 우선).
/// 동일값 최대가 여러 개일 때 전역 첫 등장이 아닌 이 규칙의 위너를
/// 반환한다(커널 reduction 구조 1:1 — 아래 (iii-c)가 이 규칙을 고정).
fn argmax_reference(lg: &[f32]) -> u32 {
    let mut sv = [-1e30f32; 1024];
    let mut si = [0u32; 1024];
    for tid in 0..1024usize {
        let mut best = -1e30f32;
        let mut idx = 0u32;
        let mut i = tid;
        while i < lg.len() {
            let v = lg[i];
            if v > best {
                best = v;
                idx = i as u32;
            }
            i += 1024;
        }
        sv[tid] = best;
        si[tid] = idx;
    }
    let mut st = 512usize;
    while st > 0 {
        for tid in 0..st {
            if sv[tid + st] > sv[tid] {
                sv[tid] = sv[tid + st];
                si[tid] = si[tid + st];
            }
        }
        st >>= 1;
    }
    si[0]
}

/// ew 게이트 g 픽스처(결정론 시드) — sigmoid 전 역역 6존 혼합:
/// j%6 [0] 포화 음 [-30,-10] · [1] 포화 양 [+10,+30] · [2] 전이 ±10 ·
/// [3] 근영역(±1e-3, j%12==0이면 ±1e-7, j%96==0이면 정확 0) · [4] 전형
/// ±2 · [5] 중폭 ±6. 포화 양측(silu→±v, exp(∓30)≈1e±13 스트레칭)과
/// 근영역(silu→v/2)·정확 0을 전부 걸고 u는 ±1.2 균일(gen_unif 재사용).
fn gen_ew_g(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|j| {
            let r = rng.next_f64() * 2.0 - 1.0;
            let v = match j % 6 {
                0 => -(10.0 + 20.0 * (r * 0.5 + 0.5)),
                1 => 10.0 + 20.0 * (r * 0.5 + 0.5),
                2 => 10.0 * r,
                3 => {
                    if j % 96 == 0 {
                        0.0
                    } else if j % 12 == 0 {
                        1e-7 * r
                    } else {
                        1e-3 * r
                    }
                }
                4 => 2.0 * r,
                _ => 6.0 * r,
            };
            v as f32
        })
        .collect()
}

/// exl3-cuda-ew — G7 ew 값 maxdiff 판정(임계 1e-6).
/// (i) 27B FFN 중간 차원 n=17408(config.json text_config
/// intermediate_size 실측 2026-10-04 — G4 gemm2 (i)과 동일 출처) ·
/// (ii) 35B-A3B MoE 전문가 FFN n=512(moe_intermediate_size 실측 —
/// MoE 게이트 폭). 하나라도 FAIL이면 Err(→ CLI 비영).
pub fn cuda_ew_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    for (tag, n, seed) in [
        (
            "(i) 27B n=17408 (intermediate_size)",
            17408usize,
            0x5EED_0000_0000_4001u64,
        ),
        (
            "(ii) 35B n=512 (moe_intermediate_size)",
            512,
            0x5EED_0000_0000_4005,
        ),
    ] {
        let g = gen_ew_g(n, seed);
        let u = gen_unif(n, seed + 1, 1.2);
        let want = ew_reference(&g, &u);
        let got = dec.ew_host(&g, &u)?;
        let (md, nan) = maxdiff_nan(&got, &want);
        let sat = g.iter().filter(|v| v.abs() >= 10.0).count();
        let nz = g.iter().filter(|v| v.abs() <= 1e-3).count();
        let pass = md <= EW_THRESH && nan == 0;
        println!(
            "device: {dev} | exl3-cuda-ew {tag}: maxdiff={md:.3e} nan={nan} zones(|g|>=10:{sat}, |g|<=1e-3:{nz}) | {}",
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!("{tag} {md:.3e} · "));
        if !pass {
            fails.push(format!("{tag} maxdiff={md:.3e} nan={nan}"));
        }
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | {report}ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-ew 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// 단일 argmax 케이스 실행: 결정론 base 생성 → 최댓값 배치(top_at에
/// base+64ulp, near_at에 최대−1ulp 근접 타이) → 모듈 vs 오라클
/// exact-match + 픽스처 무결성(오라클 토큰 = 설계 기대치 expect).
#[allow(clippy::too_many_arguments)]
fn argmax_run_case(
    dec: &mut Exl3CudaDecoder,
    dev: &str,
    tag: &str,
    seed: u64,
    top_at: &[usize],
    near_at: &[usize],
    expect: usize,
) -> Result<bool, String> {
    let n = 248320usize;
    let mut lg = gen_unif(n, seed, 6.0);
    let base = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let top = f32::from_bits(base.to_bits() + 64);
    for &i in top_at {
        lg[i] = top;
    }
    let near = f32::from_bits(top.to_bits() - 1);
    for &i in near_at {
        lg[i] = near;
    }
    let want = argmax_reference(&lg);
    let got = dec.argmax_host(&lg)?;
    let pass = got == want && want as usize == expect;
    println!(
        "device: {dev} | exl3-cuda-argmax {tag} n=248320 top@{top_at:?} near(-1ulp)@{near_at:?}: token got={got} want={want} | {}",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// exl3-cuda-argmax — G7 토큰 무결 판정(exact-match). n=248320(어휘
/// 폭 — 27B·35B config.json vocab_size 공통 실측 2026-10-04).
/// (iii-a) 참 최대 저인덱스(17) + 근접 타이 고저산포(999·65432·248319)
/// · (iii-b) 참 최대 고인덱스(248318) + 근접 타이 저인덱스(3·4096·
/// 100000) · (iii-c) 동일값 타이 쌍(max@1·@1024 — 커널 트리 규칙상
/// 낮은 tid(0) 클래스의 1024가 우승: 전역 첫 등장(1)과 다른 선택
/// 규칙을 오라클이 고정한다).
pub fn cuda_argmax_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();

    let pa = argmax_run_case(
        &mut dec,
        &dev,
        "(iii-a) max@low",
        0x5EED_0000_0000_5001,
        &[17],
        &[999, 65432, 248319],
        17,
    )?;
    if !pa {
        fails.push("(iii-a) token mismatch".into());
    }
    let pb = argmax_run_case(
        &mut dec,
        &dev,
        "(iii-b) max@high",
        0x5EED_0000_0000_5003,
        &[248318],
        &[3, 4096, 100000],
        248318,
    )?;
    if !pb {
        fails.push("(iii-b) token mismatch".into());
    }
    let pc = argmax_run_case(
        &mut dec,
        &dev,
        "(iii-c) exact-tie 1,1024",
        0x5EED_0000_0000_5005,
        &[1, 1024],
        &[],
        1024,
    )?;
    if !pc {
        fails.push("(iii-c) token mismatch".into());
    }

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | (iii-a) exact (iii-b) exact (iii-c) exact(1024) | ALL PASS"
        ))
    } else {
        Err(format!(
            "exl3-cuda-argmax 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// exl3-cuda-argmax-neg — 결함 8호 음성대조(원장 17호: 계기 자체
/// 검증): argmax를 "잘못된 길이 n"으로 발사했을 때 토큰 불일치가
/// 검출됨을 증명. 픽스처: 참 최대 200000(고인덱스) + 5120 내 준최대
/// 4321(base+32ulp). (a) n=5120(hidden 폭 혼동 — 미스캔 구간의 준최대
/// 우승) · (b) n=1(행수 [T=1] 리터럴 오독 — 항상 토큰 0). 양쪽 모두
/// 불일치 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_argmax_negative_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let n = 248320usize;
    let mut lg = gen_unif(n, 0x5EED_0000_0000_6001, 6.0);
    let base = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    lg[4321] = f32::from_bits(base.to_bits() + 32);
    lg[200000] = f32::from_bits(base.to_bits() + 64);
    let want = argmax_reference(&lg);

    // (a) 폭 혼동: n=5120(27B hidden) — 5120 이후 미스캔.
    let got_a = dec.argmax_host_n(&lg, 5120)?;
    println!(
        "device: {dev} | exl3-cuda-argmax (iva) negative control n=5120 (width misread): token got={got_a} want={want} | FAIL(expected)"
    );
    let det_a = got_a != want;

    // (b) 행수 리터럴: n=1([T=1] 행 오독) — 항상 토큰 0.
    let got_b = dec.argmax_host_n(&lg, 1)?;
    println!(
        "device: {dev} | exl3-cuda-argmax (ivb) negative control n=1 (row-count misread): token got={got_b} want={want} | FAIL(expected)"
    );
    let det_b = got_b != want;

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) n=5120 token={got_a} (b) n=1 token={got_b} != want={want} — 검증계기 정상(결함 8호: 잘못된 길이 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={got_a} (b)={got_b} want={want} — 검증계기 결함: 잘못된 길이가 탐지되지 않음"
        ))
    }
}
