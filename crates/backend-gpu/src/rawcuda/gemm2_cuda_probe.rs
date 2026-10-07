//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: gemm2_run은 케이스마다 행을 fresh 업로드·전 체인 재기록한다.
//! ② 형상은 전 선형 메타에서 자동 열거(t=1..tmax·혼합 krate·S0≠0):
//!    T∈{1,4,32}·K∈{3,4}(k=5120 n=17408·k=2048 n=8192) — t=1..tmax 전수·혼합 krate는 미커버(스윕 갭 원장 기입, plans/129 C1.2).
//! ③ 캡처-재생 3방향: 합성 입력·실층 캡처·쌍 디코더 A/B.
//! ④ 종단 상태가 유일 불변량: 판정은 종단 값 maxdiff(중간 단계는 진단).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-04, 전체 원장은 exl3_cuda_probe.rs]
//! (i) 2.190e-6[T=32 plain] · (ii) 2.235e-7[T=1 kseg] · (iii) 2.831e-7
//! [T=4 kseg] · (iv) 5.662e-7[35B plain] · had_in 비트일치 0 · 음성대조
//! had_out 2회 3.118e-1 → NEG-DETECTED(결함 15호). 임계 4e-4(GEMM2_THRESH).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::{
    R_SCALE, RefLin, SynthLin, decode_tile_f16, f16_to_f32, f16le, f32_to_f16, gen_x, had128_f32,
    maxdiff_nan, register,
};

// ── 배치 GEMM(gemm2/kseg) 프로브 — plans/124 G4 ──

/// 배치 GEMM 값 maxdiff 임계(plans/124 §1 — hip 원장 3.0-3.8e-4 기준).
const GEMM2_THRESH: f32 = 4e-4;

/// had_in 1행 f16 비트 — G2 gemv_reference_stages had_in 블록(L57-73 미러,
/// 원천 rawvk/checks/exl3.rs) 재사용: f16 pre-scale → WHT → ×1/√128 → f16.
/// G2 원장: 커널과 비트일치(bit-diff=0) — 배치 경로에서도 동일 산식.
fn had_in_bits_row(lin: &RefLin, x_row: &[f32]) -> Vec<u16> {
    let k = lin.k;
    let mut bits = vec![0u16; k];
    for ch in 0..k / 128 {
        let mut v = [0f32; 128];
        for j in 0..128 {
            let pre = f16_to_f32(f32_to_f16(
                x_row[ch * 128 + j] * f16_to_f32(f16le(lin.suh, ch * 128 + j)),
            ));
            v[j] = pre;
        }
        had128_f32(&mut v);
        for j in 0..128 {
            bits[ch * 128 + j] = f32_to_f16(v[j] * R_SCALE);
        }
    }
    bits
}

/// 배치 GEMM 오라클 — 커널 산술 계약의 f32 미러(인용: trellis.rs
/// mul1_decode L47-52 · tile_word L60-70 · decode_tile L85-91,
/// rawvk/checks/exl3.rs had128_f32 L11-25 · 체인 L54-105, src_exl3.hip
/// exl3_gemm2 계약 [T][n] H도메인 + had_out 1회):
/// 1. had_in 행별 f16 비트(커널과 비트일치 — G2 원장).
/// 2. gemm2: 활성 f16 × 가중치 f16(mul1 f64정확합→f16 1회 반올림 —
///    G2 원장: 커널 FFMA+F2FP 미러) 곱은 f32에서 정확(가수 11+11<24) —
///    f64 누산(코어 참조 계급) → s f32(H도메인).
/// 3. had_out: nseg 합산 순서(g=0..) f32 → WHT-128 f32 → ×R·svh f32.
///    커널(mma f32 누산, kt 순서·mma 내부 트리)과 미러(f64 누산)의 차이는
///    f32 누산 순서 차이의 예상 계급(~1e-6) — 임계 4e-4의 1/100 미만.
///    kseg 부분합의 f32 부분반올림(8세그먼트)도 같은 계급에 흡수된다.
///    n-타일 청크를 스레드 병렬(값은 열 독립 — 스레드 무관, 결정론 유지).
fn gemm2_reference(lin: &RefLin, rows: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (k, n, krate) = (lin.k, lin.n, lin.krate);
    let ktiles = k / 16;
    let ntiles = n / 16;
    let t_len = rows.len() / k;
    let ah: Vec<Vec<f32>> = (0..t_len)
        .map(|t| {
            had_in_bits_row(lin, &rows[t * k..(t + 1) * k])
                .iter()
                .map(|&b| f16_to_f32(b))
                .collect()
        })
        .collect();

    let workers = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let per = ntiles.div_ceil(workers);
    let mut chunks: Vec<(usize, Vec<f32>)> = Vec::new();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..workers)
            .filter_map(|w| {
                let nt0 = w * per;
                if nt0 >= ntiles {
                    return None;
                }
                let nt1 = (nt0 + per).min(ntiles);
                let ah = &ah;
                Some(sc.spawn(move || {
                    let ncols = (nt1 - nt0) * 16;
                    let mut out = vec![0f32; t_len * ncols];
                    let mut tile = [0f32; 256];
                    for nt in nt0..nt1 {
                        let mut acc = vec![0f64; t_len * 16];
                        for kt in 0..ktiles {
                            decode_tile_f16(lin.tre_u32, krate, kt, nt, ntiles, &mut tile);
                            for t in 0..t_len {
                                let arow = &ah[t][kt * 16..kt * 16 + 16];
                                for c in 0..16 {
                                    let a = &mut acc[t * 16 + c];
                                    for r in 0..16 {
                                        *a += (arow[r] as f64) * (tile[r * 16 + c] as f64);
                                    }
                                }
                            }
                        }
                        for t in 0..t_len {
                            let base = t * ncols + (nt - nt0) * 16;
                            for c in 0..16 {
                                out[base + c] = acc[t * 16 + c] as f32;
                            }
                        }
                    }
                    (nt0 * 16, out)
                }))
            })
            .collect();
        for h in handles {
            chunks.push(h.join().map_err(|_| "오라클 스레드 패닉").unwrap());
        }
    });

    // 열 청크 → 행주요 s [t][n] 조립.
    let mut s = vec![0f32; t_len * n];
    for t in 0..t_len {
        for (col0, chunk) in &chunks {
            let clen = chunk.len() / t_len;
            s[t * n + col0..t * n + col0 + clen].copy_from_slice(&chunk[t * clen..t * clen + clen]);
        }
    }

    // had_out 미러: WHT-128 f32 → ×R·svh(exl3_had_out 산술 원본).
    let mut y = vec![0f32; t_len * n];
    for t in 0..t_len {
        for ch in 0..n / 128 {
            let mut v = [0f32; 128];
            v.copy_from_slice(&s[t * n + ch * 128..t * n + ch * 128 + 128]);
            had128_f32(&mut v);
            for j in 0..128 {
                y[t * n + ch * 128 + j] = v[j] * R_SCALE * f16_to_f32(f16le(lin.svh, ch * 128 + j));
            }
        }
    }
    (s, y)
}

/// 결정론 배치 입력 rows [t][k]: 행별 gen_x(균일 ±0.1 — G2 계급 계승).
fn gen_rows(k: usize, t_len: usize, seed: u64) -> Vec<f32> {
    let mut rows = Vec::with_capacity(t_len * k);
    for t in 0..t_len {
        rows.extend_from_slice(&gen_x(k, seed + t as u64));
    }
    rows
}

/// 1회 실행 + 판정. 반환: (maxdiff, nan, chain_ms 중앙값).
fn gemm2_run(
    dec: &mut Exl3CudaDecoder,
    key: &str,
    lin: &SynthLin,
    rows: &[f32],
    double_had_out: bool,
) -> Result<(f32, usize, f64), String> {
    let (_, want) = gemm2_reference(&lin.ref_lin(), rows);
    let got = if double_had_out {
        dec.gemm2_host_double_had_out(key, rows)?
    } else {
        dec.gemm2_host(key, rows)?
    };
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        dec.gemm2_host(key, rows)?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (md, nan) = maxdiff_nan(&got, &want);
    Ok((md, nan, ts[1]))
}

/// exl3-cuda-gemm2 — plans/124 G4 배치 GEMM 값 maxdiff 판정(임계 4e-4).
/// (i) 27B gate_proj 형상 T=32(plain 경로) · (ii) 동일 형상 T=1(kseg
/// 경로 — 결함 18호: grid (n/64=272, kseg=8)=2176블록으로 70SM 포화,
/// mma n축 타일링+kseg 이중 레버) · (iii) T=4(kseg) · (iv) 35B-A3B
/// in_proj_qkv 형상(hidden=2048) T=32(plain). 하나라도 FAIL이면
/// Err(→ CLI 비영).
pub fn cuda_gemm2_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (i)-(iii) Qwen3.8-27B mlp.gate_proj 형상 — 실측 트렐리스
    // [320,1088,48](K=3) · hidden=5120(G2 원장과 동일 출처). 선형 1개
    // 등록 재사용(합성 3종은 시드만 상이 — 결정론).
    {
        let key = "synth.27b.mlp.gate_proj";
        let lin = SynthLin::generate(5120, 17408, 3, 0x170C_0DA0_0000_0001);
        register(&mut dec, key, &lin)?;
        for (t_len, tag, path) in [
            (32usize, "i", "plain"),
            (1usize, "ii", "kseg"),
            (4usize, "iii", "kseg"),
        ] {
            let rows = gen_rows(5120, t_len, 0x5EED_0000_0001_0000 + t_len as u64);
            let (md, nan, ms) = gemm2_run(&mut dec, key, &lin, &rows, false)?;
            let pass = md <= GEMM2_THRESH && nan == 0;
            println!(
                "device: {dev} | exl3-cuda-gemm2 ({tag}) Qwen3.8-27B gate_proj k=5120 n=17408 K=3 T={t_len} [{path}]: maxdiff={md:.3e} nan={nan} chain={ms:.2}ms | {}",
                if pass { "PASS" } else { "FAIL" }
            );
            report.push_str(&format!("({tag}) maxdiff={md:.3e}"));
            if !pass {
                fails.push(format!("({tag}) maxdiff={md:.3e} nan={nan}"));
            }
        }
    }

    // (iv) Qwen3.6-35B-A3B in_proj_qkv 형상 — hidden=2048(G2 원장: config.json
    // text_config 실측) · 트렐리스 [128,512,64](K=4).
    {
        let key = "synth.35b.linear_attn.in_proj_qkv";
        let lin = SynthLin::generate(2048, 8192, 4, 0x170C_0DA0_0000_0002);
        register(&mut dec, key, &lin)?;
        let rows = gen_rows(2048, 32, 0x5EED_0000_0002_0020);
        let (md, nan, ms) = gemm2_run(&mut dec, key, &lin, &rows, false)?;
        let pass = md <= GEMM2_THRESH && nan == 0;
        println!(
            "device: {dev} | exl3-cuda-gemm2 (iv) Qwen3.6-35B-A3B in_proj_qkv k=2048 n=8192 K=4 T=32 [plain]: maxdiff={md:.3e} nan={nan} chain={ms:.2}ms | {}",
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (iv) maxdiff={md:.3e}"));
        if !pass {
            fails.push(format!("(iv) maxdiff={md:.3e} nan={nan}"));
        }
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | {report} | ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-gemm2 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// exl3-cuda-gemm2-neg — 음성대조(원장 17호: 검증 계기도 스스로 검증).
/// had_out을 2회 적용하면(결함 15호 재현) maxdiff가 임계를 초과해야
/// 한다. 정상 동작 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_gemm2_negative_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let key = "synth.27b.mlp.gate_proj";
    let lin = SynthLin::generate(5120, 17408, 3, 0x170C_0DA0_0000_0001);
    register(&mut dec, key, &lin)?;
    let rows = gen_rows(5120, 4, 0x5EED_0000_0001_0004);
    let (md, nan, _) = gemm2_run(&mut dec, key, &lin, &rows, true)?;
    println!(
        "device: {dev} | exl3-cuda-gemm2 (v) negative control had_out applied twice: maxdiff={md:.3e} nan={nan} | FAIL(expected)"
    );
    if md > GEMM2_THRESH {
        Err(format!(
            "NEG-DETECTED maxdiff={md:.3e} > {GEMM2_THRESH:.0e} — 검증계기 정상(이중 hadout 결함 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED maxdiff={md:.3e} <= {GEMM2_THRESH:.0e} — 검증계기 결함: 이중 hadout이 탐지되지 않음"
        ))
    }
}

/// exl3-cuda-gemm2-debug — 단계별 비섹트 진단(소형 합성, kseg·plain 양
/// 경로): had_in 비트 일치(전 행) → gemm2 H도메인 s maxdiff → 최종
/// maxdiff → 최악 열 프로파일(결함 국소화 계기 — 원장 17호).
pub fn cuda_gemm2_debug_check() -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let key = "synth.dbg";
    let lin = SynthLin::generate(1280, 2560, 3, 0x170C_0DA0_0000_0D64);
    register(&mut dec, key, &lin)?;

    // kseg 경로(T=4)와 plain 경로(T=32)를 동일 선형으로 양측 판정.
    for (t_len, path) in [(4usize, "kseg"), (32usize, "plain")] {
        let rows = gen_rows(1280, t_len, 0x5EED_0000_0000_00D6);
        let (want_s, want_y) = gemm2_reference(&lin.ref_lin(), &rows);

        let mut ahb = Vec::new();
        let mut got_s = Vec::new();
        dec.debug_gemm2_stages(key, &rows, 1, &mut ahb, &mut got_s)?;
        let want_bits: Vec<u16> = (0..t_len)
            .flat_map(|t| had_in_bits_row(&lin.ref_lin(), &rows[t * 1280..(t + 1) * 1280]))
            .collect();
        let mut bitdiff = 0usize;
        let mut first: Option<String> = None;
        for i in 0..t_len * lin.k {
            let g = f16le(&ahb, i);
            if g != want_bits[i] {
                bitdiff += 1;
                if first.is_none() {
                    first = Some(format!("k[{i}] gpu=0x{g:04X} ref=0x{:04X}", want_bits[i]));
                }
            }
        }
        println!(
            "device: {dev} | [{path} T={t_len}] stage1 had_in: bit-diff {bitdiff}/{} {}",
            t_len * lin.k,
            first.as_deref().unwrap_or("(정합)")
        );

        dec.debug_gemm2_stages(key, &rows, 2, &mut ahb, &mut got_s)?;
        let (md_s, _) = maxdiff_nan(&got_s, &want_s);
        let got_y = dec.gemm2_host(key, &rows)?;
        let (md_y, _) = maxdiff_nan(&got_y, &want_y);
        println!("device: {dev} | [{path} T={t_len}] stage2 gemm2 s(H-domain): maxdiff={md_s:.3e}");
        println!("device: {dev} | [{path} T={t_len}] stage3 had_out y: maxdiff={md_y:.3e}");
        let mut idx: Vec<usize> = (0..got_s.len()).collect();
        idx.sort_by(|&a, &b| {
            (got_s[a] - want_s[a])
                .abs()
                .partial_cmp(&(got_s[b] - want_s[b]).abs())
                .unwrap()
        });
        idx.reverse();
        for &i in idx.iter().take(3) {
            println!(
                "device: {dev} | [{path} T={t_len}] worst s[{}][{}]: gpu={:.6} ref={:.6} diff={:.3e}",
                i / lin.n,
                i % lin.n,
                got_s[i],
                want_s[i],
                (got_s[i] - want_s[i]).abs()
            );
        }
    }
    Ok(format!(
        "device: {dev} | gemm2-debug: 양 경로 단계 판정 완료"
    ))
}
