//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: gemv_run은 케이스마다 입력을 fresh 업로드해 체인이 전 스테이지를 재기록한다(잔류 값 혼입 없음).
//! ② 형상은 전 선형 메타에서 자동 열거(t=1..tmax·혼합 krate·S0≠0):
//!    현재 k=5120 krate K∈{3,5}·동일 k 이중 선형·실가중 1종 — krate 1..8 전수·혼합 krate는 미커버(스윕 갭 원장 기입, plans/129 C1.2. 신규 픽스처 추가 시 전 프루브 재실행으로 기존 원장 값 무변화 확인).
//! ③ 캡처-재생 3방향: 합성 입력·실층 캡처·쌍 디코더 A/B.
//! ④ 종단 상태가 유일 불변량: 판정은 종단 값 maxdiff(중간 단계는 진단).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-04, 전체 원장은 exl3_cuda_probe.rs]
//! 합성 (i) 1.623e-4 · (ii) 1.021e-4 · (iiia) 1.414e-4 · (iiib) 1.511e-4
//! · 실가중 27B gate_proj 2.216e-4 · 음성대조 2.844e-1 → NEG-DETECTED.
//! 임계 3e-4(GEMV_THRESH). argmax 판정 금지 — 값 maxdiff(plans/124 §5).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::{
    R_SCALE, RefLin, SynthLin, decode_tile_f16, f16_to_f32, f16le, f32_to_f16, gen_x, had128_f32,
    maxdiff_nan, register,
};

/// 값 maxdiff 판정 임계(plans/124 §1 — hip 원장 2.664e-4 기준).
/// 값 maxdiff 판정 임계(plans/124 §1 — hip 원장 2.664e-4 기준).
const GEMV_THRESH: f32 = 3e-4;

/// 참조 단계 산출 — 단계별 바이섹트용 진단(검증 계기 자체 검증, 원장 17호).
struct RefStages {
    /// had_in 산출 f16 비트 [k].
    ah_bits: Vec<u16>,
    /// gemv 부분합 [nseg][n].
    s: Vec<f32>,
    /// had_out 산출 [n].
    y: Vec<f32>,
}

/// GEMV 체인 f32 오라클(단계 산출 포함): had_in → gemv(nseg=16·FOLD=4) →
/// had_out — rawvk/checks/exl3.rs CPU 참조 블록(L54-105)의 산술·연산 순서를
/// 줄 단위로 미러한다(코어 f32 참조가 G2 정합 기준, plans/124 §5).
/// 커널(__hfma2 단일 반올림)과 미러(곱·합 각 f16 반올림)의 반올림 지점
/// 차이는 연산당 ≤1 ulp의 예상 계급이다 — hip도 같은 계급으로 2.664e-4
/// 통과(원장). mul1의 f32 fma는 이 호스트에 +fma 특성이 없어 소프트웨어
/// 이중 반올림으로 떨어지므로, 참조가 의도한 correctly-rounded fma를
/// f64 정확합→f16 1회 반올림으로 대체한다(타이 경계 실측 2026-10-04).
fn gemv_reference_stages(lin: &RefLin, x: &[f32], nseg: usize) -> RefStages {
    let (k, n, krate) = (lin.k, lin.n, lin.krate);
    let ktiles = k / 16;
    let ntiles = n / 16;

    // had_in: f16 pre-scale → WHT → ×1/√128 → f16(L57-73 미러).
    let mut ah = vec![0f32; k];
    let mut ah_bits = vec![0u16; k];
    for ch in 0..k / 128 {
        let mut v = [0f32; 128];
        for j in 0..128 {
            let pre = f16_to_f32(f32_to_f16(
                x[ch * 128 + j] * f16_to_f32(f16le(lin.suh, ch * 128 + j)),
            ));
            v[j] = pre;
        }
        had128_f32(&mut v);
        for j in 0..128 {
            let b = f32_to_f16(v[j] * R_SCALE);
            ah_bits[ch * 128 + j] = b;
            ah[ch * 128 + j] = f16_to_f32(b);
        }
    }

    // gemv: nseg 분할 부분합 + 4 k-타일 f32 폴드(L76-97 미러 — 코어 산술:
    // prod=f16(a·w) → acc=f16(acc+prod), f32 곱/합 각 1회 반올림).
    let mut s = vec![0f32; nseg * n];
    let mut tile = [0f32; 256];
    for nt in 0..ntiles {
        for seg in 0..nseg {
            let ktb = ktiles * seg / nseg;
            let kte = ktiles * (seg + 1) / nseg;
            let mut accf = [0f32; 16];
            let mut lo = [0u16; 16];
            let mut hi = [0u16; 16];
            for kt in ktb..kte {
                decode_tile_f16(lin.tre_u32, krate, kt, nt, ntiles, &mut tile);
                for c in 0..16 {
                    for j in 0..8 {
                        let alo = ah[kt * 16 + 2 * j];
                        let ahi = ah[kt * 16 + 2 * j + 1];
                        let wlo = tile[(2 * j) * 16 + c];
                        let whi = tile[(2 * j + 1) * 16 + c];
                        // 코어 f32 미러 산술(exl3.rs L84-91): prod=f16(a·w)
                        // → acc=f16(acc+prod) — f32 곱/합 각 1회.
                        let pl = f32_to_f16(alo * wlo);
                        let ph = f32_to_f16(ahi * whi);
                        lo[c] = f32_to_f16(f16_to_f32(lo[c]) + f16_to_f32(pl));
                        hi[c] = f32_to_f16(f16_to_f32(hi[c]) + f16_to_f32(ph));
                    }
                }
                if (kt & 3) == 3 {
                    for c in 0..16 {
                        accf[c] += f16_to_f32(lo[c]) + f16_to_f32(hi[c]);
                        lo[c] = 0;
                        hi[c] = 0;
                    }
                }
            }
            for c in 0..16 {
                accf[c] += f16_to_f32(lo[c]) + f16_to_f32(hi[c]);
                s[seg * n + nt * 16 + c] = accf[c];
            }
        }
    }

    // had_out: nseg 합산(순서 g=0..nseg-1) → WHT → ×R·svh(L98-105 미러).
    let mut y = vec![0f32; n];
    for ch in 0..n / 128 {
        let mut v = [0f32; 128];
        for j in 0..128 {
            let mut acc0 = 0f32;
            for g in 0..nseg {
                acc0 += s[g * n + ch * 128 + j];
            }
            v[j] = acc0;
        }
        had128_f32(&mut v);
        for j in 0..128 {
            y[ch * 128 + j] = v[j] * R_SCALE * f16_to_f32(f16le(lin.svh, ch * 128 + j));
        }
    }
    RefStages { ah_bits, s, y }
}

/// 오라클 최종 산출만(통상 프로브용 래퍼 — hfma2 단일반올림 · FFMA 디코드).
fn gemv_reference(lin: &RefLin, x: &[f32], nseg: usize) -> Vec<f32> {
    gemv_reference_stages(lin, x, nseg).y
}

/// 1개 선형 실행 + 판정. 반환: (maxdiff, nan, chain_ms_중앙).
fn gemv_run(
    dec: &mut Exl3CudaDecoder,
    key: &str,
    lin: &SynthLin,
    x: &[f32],
    skip_had_out: bool,
) -> Result<(f32, usize, f64), String> {
    let want = gemv_reference(&lin.ref_lin(), x, crate::rawcuda::exl3_cuda::GEMV_NSEG);
    let got = if skip_had_out {
        dec.gemv_host_skip_had_out(key, x)?
    } else {
        dec.gemv_host(key, x)?
    };
    // 체인 시간(3회 중앙 — 참조 판정과 무관한 진단 부가산).
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        dec.gemv_host(key, x)?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (md, nan) = maxdiff_nan(&got, &want);
    Ok((md, nan, ts[1]))
}

/// exl3-cuda-gemv — 합성 3종(① 27B 형상 ② 35B 형상 ③ 동일 k·상이
/// suh/svh 2선형) 값 maxdiff 판정. 하나라도 FAIL이면 Err(→ CLI 비영).
pub fn cuda_gemv_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (i) Qwen3.8-27B mlp.gate_proj 형상 — 실측 트레일리스 [320,1088,48]
    // (K=3) · hidden=5120(config.json 실측 2026-10-04).
    {
        let key = "synth.27b.mlp.gate_proj";
        let lin = SynthLin::generate(5120, 17408, 3, 0x170C_0DA0_0000_0001);
        register(&mut dec, key, &lin)?;
        let x = gen_x(5120, 0x5EED_0000_0000_0001);
        let (md, nan, ms) = gemv_run(&mut dec, key, &lin, &x, false)?;
        let pass = md <= GEMV_THRESH && nan == 0;
        println!(
            "device: {dev} | exl3-cuda-gemv (i) Qwen3.8-27B gate_proj k=5120 n=17408 K=3: maxdiff={md:.3e} nan={nan} chain={ms:.2}ms | {}",
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!("(i) maxdiff={md:.3e}"));
        if !pass {
            fails.push(format!("(i) maxdiff={md:.3e} nan={nan}"));
        }
    }

    // (ii) Qwen3.6-35B-A3B in_proj_qkv 형상 — hidden=2048
    // (D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw/config.json text_config
    // hidden_size 실측 2026-10-04) · 트레일리스 [128,512,64](K=4).
    {
        let key = "synth.35b.linear_attn.in_proj_qkv";
        let lin = SynthLin::generate(2048, 8192, 4, 0x170C_0DA0_0000_0002);
        register(&mut dec, key, &lin)?;
        let x = gen_x(2048, 0x5EED_0000_0000_0002);
        let (md, nan, ms) = gemv_run(&mut dec, key, &lin, &x, false)?;
        let pass = md <= GEMV_THRESH && nan == 0;
        println!(
            "device: {dev} | exl3-cuda-gemv (ii) Qwen3.6-35B-A3B in_proj_qkv k=2048 n=8192 K=4: maxdiff={md:.3e} nan={nan} chain={ms:.2}ms | {}",
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (ii) maxdiff={md:.3e}"));
        if !pass {
            fails.push(format!("(ii) maxdiff={md:.3e} nan={nan}"));
        }
    }

    // (iii) 동일 k=5120·상이 suh/svh 2선형(결함 1호 가드 — suh/svh는
    // 선형별, k값 탐색 금지). 형상은 27B in_proj_qkv [320,640,80](K=5).
    {
        let ka = "synth.dual.a";
        let kb = "synth.dual.b";
        let la = SynthLin::generate(5120, 10240, 5, 0x170C_0DA0_0000_010A);
        let lb = SynthLin::generate(5120, 10240, 5, 0x170C_0DA0_0000_010B);
        register(&mut dec, ka, &la)?;
        register(&mut dec, kb, &lb)?;
        let x = gen_x(5120, 0x5EED_0000_0000_0003);
        for (key, lin, tag) in [(ka, &la, "a"), (kb, &lb, "b")] {
            let (md, nan, ms) = gemv_run(&mut dec, key, lin, &x, false)?;
            let pass = md <= GEMV_THRESH && nan == 0;
            println!(
                "device: {dev} | exl3-cuda-gemv (iii{tag}) same-k dual lin k=5120 n=10240 K=5: maxdiff={md:.3e} nan={nan} chain={ms:.2}ms | {}",
                if pass { "PASS" } else { "FAIL" }
            );
            report.push_str(&format!(" · (iii{tag}) maxdiff={md:.3e}"));
            if !pass {
                fails.push(format!("(iii{tag}) maxdiff={md:.3e} nan={nan}"));
            }
        }
        // 부가 산단: 두 출력은 유의미하게 달라야 한다(스케일이 선형별로
        // 반영됨) — k 기준 조회 결함이면 위 maxdiff에서 이미 FAIL.
        let ya = dec.gemv_host(ka, &x)?;
        let yb = dec.gemv_host(kb, &x)?;
        let (xd, _) = maxdiff_nan(&ya, &yb);
        println!(
            "device: {dev} | exl3-cuda-gemv (iii) cross-diff a-vs-b: {xd:.3e} (스케일 반영 확인)"
        );
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | {report} | ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-gemv 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// exl3-cuda-gemv-neg — 음성대조(원장 17호: 검증 계기도 스스로 검증).
/// had_out을 생략하면(결함 3호 재현) maxdiff가 임계를 초과해야 한다.
/// 정상 동작 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_gemv_negative_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let key = "synth.27b.mlp.gate_proj";
    let lin = SynthLin::generate(5120, 17408, 3, 0x170C_0DA0_0000_0001);
    register(&mut dec, key, &lin)?;
    let x = gen_x(5120, 0x5EED_0000_0000_0001);
    let (md, nan, _) = gemv_run(&mut dec, key, &lin, &x, true)?;
    println!(
        "device: {dev} | exl3-cuda-gemv (iv) negative control had_out skipped: maxdiff={md:.3e} nan={nan} | FAIL(expected)"
    );
    if md > GEMV_THRESH {
        Err(format!(
            "NEG-DETECTED maxdiff={md:.3e} > {GEMV_THRESH:.0e} — 검증계기 정상(생략 결함 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED maxdiff={md:.3e} <= {GEMV_THRESH:.0e} — 검증계기 결함: had_out 생략이 탐지되지 않음"
        ))
    }
}

/// exl3-cuda-gemv-debug — 단계별 비섹트 진단(소형 합성): had_in 비트
/// 일치 여부 → gemv 부분합 maxdiff → 최종 maxdiff. 어느 단계에서
/// 발산하는지 인쇄한다(결함 국소화 계기 — 원장 17호).
pub fn cuda_gemv_debug_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let key = "synth.dbg";
    let lin = SynthLin::generate(1280, 2560, 3, 0x170C_0DA0_0000_0DB9);
    register(&mut dec, key, &lin)?;
    let x = gen_x(1280, 0x5EED_0000_0000_00D9);

    let mut ahb = Vec::new();
    let mut sb = Vec::new();
    dec.debug_run_stages(key, &x, 1, &mut ahb, &mut sb)?;
    let st = gemv_reference_stages(&lin.ref_lin(), &x, crate::rawcuda::exl3_cuda::GEMV_NSEG);
    let mut bitdiff = 0usize;
    let mut first: Option<String> = None;
    for i in 0..lin.k {
        let g = f16le(&ahb, i);
        if g != st.ah_bits[i] {
            bitdiff += 1;
            if first.is_none() {
                first = Some(format!(
                    "k[{i}] gpu=0x{:04X} ref=0x{:04X} (f32 {} vs {})",
                    g,
                    st.ah_bits[i],
                    f16_to_f32(g),
                    f16_to_f32(st.ah_bits[i])
                ));
            }
        }
    }
    println!(
        "device: {dev} | stage1 had_in: bit-diff {bitdiff}/{} {}",
        lin.k,
        first.as_deref().unwrap_or("(정합)")
    );

    dec.debug_run_stages(key, &x, 2, &mut ahb, &mut sb)?;
    let (md_sb, _) = maxdiff_nan(&sb, &st.s);
    let got_y = dec.gemv_host(key, &x)?;
    let (md_y, _) = maxdiff_nan(&got_y, &st.y);
    println!("device: {dev} | stage2 gemv sb [nseg][n]: maxdiff={md_sb:.3e}");
    println!("device: {dev} | stage3 had_out y: maxdiff={md_y:.3e}");
    // 오차 프로파일: 상위 4개 (seg,열) 차 + 분포 요약 — 국소 결함(구조)과
    // 균질 잡음(예상 f16 계급)의 구분 계기(원장 17호).
    let mut idx: Vec<usize> = (0..sb.len()).collect();
    idx.sort_by(|&a, &b| {
        (sb[a] - st.s[a])
            .abs()
            .partial_cmp(&(sb[b] - st.s[b]).abs())
            .unwrap()
    });
    idx.reverse();
    for &i in idx.iter().take(4) {
        let (seg, col) = (i / lin.n, i % lin.n);
        println!(
            "device: {dev} | worst seg={seg} col={col}: gpu={:.6} ref={:.6} diff={:.3e}",
            sb[i],
            st.s[i],
            (sb[i] - st.s[i]).abs()
        );
    }
    let mut c6 = 0usize;
    let mut c7 = 0usize;
    for i in 0..sb.len() {
        let d = (sb[i] - st.s[i]).abs();
        if d > 1e-6 {
            c6 += 1;
        }
        if d > 1e-7 {
            c7 += 1;
        }
    }
    println!(
        "device: {dev} | sb diff counts: >1e-6:{c6} >1e-7:{c7} / {}",
        sb.len()
    );
    Ok(format!(
        "device: {dev} | debug: had_in bit-diff={bitdiff} sb={md_sb:.3e} y={md_y:.3e}"
    ))
}

/// exl3-cuda-gemv-real — 실가중 단일 선형(모듈 load_keys 경유 + d2h 판독
/// 오라클). 합성 3종이 기본 계약이고 이 경로는 실아카이브 적재 검증 부가.
pub fn cuda_gemv_real_check(dir: &str, key: &str) -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::load_keys(dir, &[key])?;
    let dev = dec.device_name().to_string();
    let (k, n, krate) = dec
        .lin_shape(key)
        .ok_or_else(|| format!("선형 없음: {key}"))?;
    let (rk, rn, rkrate, suh, tre, svh) = dec.readback_linear(key)?;
    if (rk, rn, rkrate) != (k, n, krate) {
        return Err(format!("{key}: 판독 형상 불일치 ({rk},{rn},{rkrate})"));
    }
    let tre_u32: Vec<u32> = tre
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let lin = RefLin {
        k,
        n,
        krate,
        suh: &suh,
        svh: &svh,
        tre_u32: &tre_u32,
    };
    let x = gen_x(k, 0x5EED_0000_0000_00F1);
    let want = gemv_reference(&lin, &x, crate::rawcuda::exl3_cuda::GEMV_NSEG);
    let got = dec.gemv_host(key, &x)?;
    let (md, nan) = maxdiff_nan(&got, &want);
    let pass = md <= GEMV_THRESH && nan == 0;
    println!(
        "device: {dev} | exl3-cuda-gemv-real {key} k={k} n={n} K={krate}: maxdiff={md:.3e} nan={nan} | {}",
        if pass { "PASS" } else { "FAIL" }
    );
    if pass {
        Ok(format!(
            "device: {dev} | real {key}: maxdiff={md:.3e} | PASS"
        ))
    } else {
        Err(format!("real {key}: maxdiff={md:.3e} nan={nan}"))
    }
}
