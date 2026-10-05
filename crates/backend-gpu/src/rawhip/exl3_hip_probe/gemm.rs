//! EXL3 hip 프로브 gemm (plans/129 R6 — exl3_hip_probe.rs 순수 이동).

use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;

pub fn hip_gemm_check(dir: &str, key_sel: &str, t_arg: usize) -> Result<String, String> {
    // 형상 스윕(plans/128 P2): 단축명 → 전체 키. gemm2는 선형 무관 동일 커널이라
    // 대표 형상별 스윕이 전-선형 검증을 대행한다(n=1024~248320).
    let key = match key_sel {
        "g" => "model.language_model.layers.0.mlp.gate_proj",
        "g5" => "model.language_model.layers.5.mlp.gate_proj", // 혼합정밀 층(P2 krate 스윕)
        "u" => "model.language_model.layers.0.mlp.up_proj",
        "d" => "model.language_model.layers.0.mlp.down_proj",
        "qkv" => "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "z" => "model.language_model.layers.0.linear_attn.in_proj_z",
        "gop" => "model.language_model.layers.0.linear_attn.out_proj",
        "q" => "model.language_model.layers.3.self_attn.q_proj",
        "k" => "model.language_model.layers.3.self_attn.k_proj",
        "v" => "model.language_model.layers.3.self_attn.v_proj",
        "o" => "model.language_model.layers.3.self_attn.o_proj",
        "lh" => "lm_head",
        _ => {
            return Err(format!("미지 key {key_sel}: g,u,d,qkv,z,gop,q,k,v,o,lh,g5"));
        }
    };
    let t_rows = t_arg;
    // 부분 적재: L0 gate_proj 1개 선형만 (풀모델 상주 금지).
    let mut tr = TrellisResident::load_keep(dir, &|n: &str| n == key)?;
    let (k, n, krate, suh, tre, svh) = tr.linear_raw(key)?;
    let x: Vec<f32> = {
        let mut seed: u32 = 0x33CC_0F0F;
        (0..t_rows * k)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.3
            })
            .collect()
    };
    // vk 참조: 표본 4행
    let samp: Vec<usize> = (0..4).map(|i| i * (t_rows - 1) / 3).collect();
    let want: Vec<Vec<f32>> = samp
        .iter()
        .map(|&r| tr.linear(key, &x[r * k..(r + 1) * k]))
        .collect::<Result<_, _>>()?;
    drop(tr);
    let hc = HipCtx::new()?;
    let nseg = 1usize; // gemm2는 nseg=1(sb [T][n])
    let dx = hc.alloc(t_rows * k * 4)?;
    let dsuh = hc.alloc(suh.len())?;
    let dtre = hc.alloc(tre.len())?;
    let dsvh = hc.alloc(svh.len())?;
    let dah = hc.alloc(t_rows * k * 2)?;
    let dsb = hc.alloc(t_rows * n * 4)?;
    let dy = hc.alloc(t_rows * n * 4)?;
    let xb: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, t_rows * k * 4) };
    hc.h2d(dx, xb)?;
    hc.h2d(dsuh, &suh)?;
    hc.h2d(dtre, &tre)?;
    hc.h2d(dsvh, &svh)?;
    // had_in: 그리드 (k/128, T)
    let (mut kc, mut ks) = ((k / 128) as i32, k as i32);
    let (mut p0, mut p1, mut p2) = (dx, dsuh, dah);
    hc.launch3(
        "exl3_had_in",
        (k / 128) as u32,
        t_rows as u32,
        1,
        128,
        &mut [
            &mut p0 as *mut *mut u8 as *mut _,
            &mut p1 as *mut *mut u8 as *mut _,
            &mut p2 as *mut *mut u8 as *mut _,
            &mut kc as *mut i32 as *mut _,
            &mut ks as *mut i32 as *mut _,
        ],
    )?;
    // had_in 다중행 대조 — 행 0·중간행 WHT 미러(러닝: gemv 프로브 미러 행 오프셋판).
    {
        let suh_f: Vec<f32> = (0..k)
            .map(|kk| {
                let b = (kk / 2) * 4;
                let w32 = u32::from_le_bytes([suh[b], suh[b + 1], suh[b + 2], suh[b + 3]]);
                half::f16::from_bits(((w32 >> ((kk & 1) * 16)) & 0xFFFF) as u16).to_f32()
            })
            .collect();
        let mut ahb = vec![0u8; t_rows * k * 2];
        hc.d2h(&mut ahb, dah)?;
        hc.sync()?;
        for &r in &[0usize, t_rows / 2] {
            let mut bad = 0usize;
            let ah_row: &[u32] = unsafe {
                std::slice::from_raw_parts(ahb[r * k * 2..].as_ptr() as *const u32, k / 2)
            };
            for ch in 0..k / 128 {
                let mut sm = [0f32; 128];
                for i in 0..128 {
                    let pre = half::f16::from_f32(x[r * k + ch * 128 + i] * suh_f[ch * 128 + i]);
                    sm[i] = pre.to_f32();
                }
                let mut w = 1usize;
                while w < 128 {
                    let mut i = 0;
                    while i < 128 {
                        let blk = (i / (2 * w)) * (2 * w);
                        for j in 0..w {
                            let a = sm[blk + j];
                            let b = sm[blk + j + w];
                            sm[blk + j] = a + b;
                            sm[blk + j + w] = a - b;
                        }
                        i += 2 * w;
                    }
                    w *= 2;
                }
                for i in (0..128).step_by(2) {
                    let lo = half::f16::from_f32(sm[i] * 0.08838834764831845);
                    let hi = half::f16::from_f32(sm[i + 1] * 0.08838834764831845);
                    let want = (lo.to_bits() as u32) | ((hi.to_bits() as u32) << 16);
                    if ah_row[ch * 64 + i / 2] != want {
                        bad += 1;
                    }
                }
            }
            eprintln!("  [gemmdbg] had_in 행 {r} 워드 불일치 {bad}/{}", k / 2);
        }
    }
    let (mut kt, mut nt, mut kk, mut tt) = (
        (k / 16) as i32,
        (n / 16) as i32,
        krate as i32,
        t_rows as i32,
    );
    let (mut g0, mut g1, mut g2) = (dah, dtre, dsb);
    hc.launch3(
        "exl3_gemm2",
        (n / 64) as u32,
        t_rows.div_ceil(128) as u32,
        1,
        128,
        &mut [
            &mut g0 as *mut *mut u8 as *mut _,
            &mut g1 as *mut *mut u8 as *mut _,
            &mut g2 as *mut *mut u8 as *mut _,
            &mut kt as *mut i32 as *mut _,
            &mut nt as *mut i32 as *mut _,
            &mut kk as *mut i32 as *mut _,
            &mut tt as *mut i32 as *mut _,
        ],
    )?;
    let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, nseg as i32, n as i32);
    let (mut c0, mut c1, mut c2) = (dsb, dsvh, dy);
    hc.launch3(
        "exl3_had_out",
        (n / 128) as u32,
        t_rows as u32,
        1,
        128,
        &mut [
            &mut c0 as *mut *mut u8 as *mut _,
            &mut c1 as *mut *mut u8 as *mut _,
            &mut c2 as *mut *mut u8 as *mut _,
            &mut nch as *mut i32 as *mut _,
            &mut nsg as *mut i32 as *mut _,
            &mut nst as *mut i32 as *mut _,
        ],
    )?;
    let mut yb = vec![0u8; t_rows * n * 4];
    hc.d2h(&mut yb, dy)?;
    hc.sync()?;
    // SAFETY: d2h 완료 후 재해석.
    let got: &[f32] = unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, t_rows * n) };
    // sb 덤프 — gemm2의 s[행0/행1][0..3] vs gemv 세그합(행1).
    {
        let mut sbb = vec![0u8; t_rows * n * 4];
        hc.d2h(&mut sbb, dsb)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let sf: &[f32] =
            unsafe { std::slice::from_raw_parts(sbb.as_ptr() as *const f32, t_rows * n) };
        eprintln!(
            "  [sbdbg] gemm s[0][0..4]={:?}{}{}",
            &sf[0..4],
            if t_rows >= 2 {
                format!(" s[1][0..4]={:?}", &sf[n..n + 4])
            } else {
                String::new()
            },
            if t_rows > 5 {
                format!(" s[5][0..4]={:?}", &sf[5 * n..5 * n + 4])
            } else {
                String::new()
            }
        );
    }
    // 3-way: 검증된 T=1 GEMV 체인으로 행 5 재계산 → gemm 행5·vk 참조 삼각 대조.
    // [수리 2026-10-04] had_in 출력을 **전용 버퍼 dah5**에 쓴다 — 종전 p2=dah가
    // dah 행0을 x[samp[1]] 변환으로 덮어썼고, 뒤따르는 wmma/mma 블록이 오염된
    // 활성으로 토큰0 = 토큰 samp[1]의 dot를 계산했다(“WMMA 행0 오염”의 진범,
    // plans/125-3 · plans/126 I1 — 커널 무죄, 프로브 하네스 결함).
    {
        let r5 = samp[1];
        // T<16에서는 nsg=T로 줄인다 — dsb는 [t_rows][n] 할당이라 16세그 기록이
        // 버퍼를 초과한다(plans/130: T≤8 스윕 시 발견된 프로브 하네스 결함).
        let mut gv_nsg = 16i32.min(t_rows as i32);
        let dah5 = hc.alloc(k * 2)?;
        let x5: Vec<f32> = x[r5 * k..(r5 + 1) * k].to_vec();
        let x5b: &[u8] = unsafe { std::slice::from_raw_parts(x5.as_ptr() as *const u8, k * 4) };
        hc.h2d(dx, x5b)?;
        let mut p2b = dah5;
        hc.launch3(
            "exl3_had_in",
            (k / 128) as u32,
            1,
            1,
            128,
            &mut [
                &mut p0 as *mut *mut u8 as *mut _,
                &mut p1 as *mut *mut u8 as *mut _,
                &mut p2b as *mut *mut u8 as *mut _,
                &mut kc as *mut i32 as *mut _,
                &mut ks as *mut i32 as *mut _,
            ],
        )?;
        let (mut g0, mut g1, mut g2) = (dah5, dtre, dsb);
        hc.launch3(
            "exl3_gemv",
            ((n / 16) / 8) as u32,
            gv_nsg as u32,
            1,
            128,
            &mut [
                &mut g0 as *mut *mut u8 as *mut _,
                &mut g1 as *mut *mut u8 as *mut _,
                &mut g2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
            ],
        )?;
        hc.launch3(
            "exl3_had_out",
            (n / 128) as u32,
            1,
            1,
            128,
            &mut [
                &mut c0 as *mut *mut u8 as *mut _,
                &mut c1 as *mut *mut u8 as *mut _,
                &mut c2 as *mut *mut u8 as *mut _,
                &mut nch as *mut i32 as *mut _,
                &mut gv_nsg as *mut i32 as *mut _,
                &mut nst as *mut i32 as *mut _,
            ],
        )?;
        let mut y5 = vec![0u8; n * 4];
        hc.d2h(&mut y5, dy)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let y5f: &[f32] = unsafe { std::slice::from_raw_parts(y5.as_ptr() as *const f32, n) };
        let mut m_gv_gemm = 0f32;
        let mut m_gv_vk = 0f32;
        for i in 0..n {
            m_gv_gemm = m_gv_gemm.max((got[r5 * n + i] - y5f[i]).abs());
            m_gv_vk = m_gv_vk.max((y5f[i] - want[1][i]).abs());
        }
        eprintln!("  [gemmdbg] 행5: gemv-vs-gemm={m_gv_gemm:.3e} gemv-vs-vk={m_gv_vk:.3e}");
        {
            if t_rows >= 16 {
                let mut seg = vec![0u8; 16 * n * 4];
                hc.d2h(&mut seg, dsb)?;
                hc.sync()?;
                // SAFETY: d2h 완료 후 재해석 — [16세그][n] 부분합.
                let segf: &[f32] =
                    unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const f32, 16 * n) };
                let sums: Vec<f32> = (0..4)
                    .map(|c| (0..16).map(|g| segf[g * n + c]).sum())
                    .collect();
                eprintln!("  [sbdbg] gemv(행5) 세그합[0..4]={sums:?}");
            }
        }
    }
    let mut worst = 0f32;
    for (si, &r) in samp.iter().enumerate() {
        let mut md = 0f32;
        for i in 0..n {
            md = md.max((got[r * n + i] - want[si][i]).abs());
        }
        eprintln!("  [gemmdbg] 행 {r} maxdiff={md:.3e}");
        worst = worst.max(md);
    }
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        hc.launch3(
            "exl3_gemm2",
            (n / 64) as u32,
            t_rows.div_ceil(128) as u32,
            1,
            128,
            &mut [
                &mut g0 as *mut *mut u8 as *mut _,
                &mut g1 as *mut *mut u8 as *mut _,
                &mut g2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
                &mut tt as *mut i32 as *mut _,
            ],
        )?;
        hc.sync()?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    // MMA 변형(plans/127 B) — Q4-MMQ 구조 이식판: 정합+속도 A/B.
    {
        let (mut kt, mut nt2, mut kk2, mut tt2) = (
            (k / 16) as i32,
            (n / 16) as i32,
            krate as i32,
            t_rows as i32,
        );
        let (mut g0, mut g1, mut g2) = (dah, dtre, dsb);
        let dflush_m = hc.alloc(128 << 20)?;
        let mut tm: Vec<f64> = Vec::new();
        for _ in 0..3 {
            hc.l2_flush(dflush_m, 128 << 20)?;
            hc.sync()?;
            let t0 = std::time::Instant::now();
            hc.launch3(
                "exl3_gemm2_mma",
                (n / 64) as u32,
                t_rows.div_ceil(64) as u32,
                1,
                256,
                &mut [
                    &mut g0 as *mut *mut u8 as *mut _,
                    &mut g1 as *mut *mut u8 as *mut _,
                    &mut g2 as *mut *mut u8 as *mut _,
                    &mut kt as *mut i32 as *mut _,
                    &mut nt2 as *mut i32 as *mut _,
                    &mut kk2 as *mut i32 as *mut _,
                    &mut tt2 as *mut i32 as *mut _,
                ],
            )?;
            hc.sync()?;
            tm.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        // H⁻¹⊙svh 후처리(제자리) — want가 최종 도메인이므로 동일 적용.
        {
            let (mut c0, mut c1, mut c2) = (dsb, dsvh, dsb);
            let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, 1i32, n as i32);
            hc.launch3(
                "exl3_had_out",
                (n / 128) as u32,
                t_rows as u32,
                1,
                128,
                &mut [
                    &mut c0 as *mut *mut u8 as *mut _,
                    &mut c1 as *mut *mut u8 as *mut _,
                    &mut c2 as *mut *mut u8 as *mut _,
                    &mut nch as *mut i32 as *mut _,
                    &mut nsg as *mut i32 as *mut _,
                    &mut nst as *mut i32 as *mut _,
                ],
            )?;
            hc.sync()?;
        }
        tm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut mbb = vec![0u8; t_rows * n * 4];
        hc.d2h(&mut mbb, dsb)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let mfc: &[f32] =
            unsafe { std::slice::from_raw_parts(mbb.as_ptr() as *const f32, t_rows * n) };
        let mut mmd = 0f32;
        for (si, &r) in samp.iter().enumerate() {
            let _ = si;
            for i in 0..n {
                mmd = mmd.max((mfc[r * n + i] - want[si][i]).abs());
            }
        }
        for (si, &r) in samp.iter().enumerate() {
            if si >= 4 {
                break;
            }
            let mut rowmd = 0f32;
            for i in 0..n {
                rowmd = rowmd.max((mfc[r * n + i] - want[si][i]).abs());
            }
            eprintln!("  [mmarow] 샘플행{r} maxdiff={rowmd:.3e}");
            if si == 0 && r == 0 {
                eprintln!("  [mmav0] gpu={:?}", &mfc[0..6]);
                eprintln!("  [mmav0] ref={:?}", &want[0][0..6]);
            }
        }
        let tf2 = 2.0 * k as f64 * n as f64 * t_rows as f64 / 1e12;
        eprintln!(
            "  [mmadbg] mma {:.1}ms(콜드 L2플러시) = {:.1} TF · y-vs-ref maxdiff={mmd:.3e}",
            tm[1],
            tf2 / (tm[1] / 1000.0)
        );
    }
    // gemv_m(plans/130) — 소형-T(≤8) m행 GEMV: kseg 대체 후보. 정합+속도 A/B.
    if t_rows <= 8 {
        let dbat = hc.alloc(t_rows * 16 * n * 4)?;
        // L2 플러시 측정(plans/131 S10): 반복 사이 128MB memset으로 L2 교체 —
        // 고립 측정의 웜 편향을 걷어내고 콜드 스트리밍 레이트를 잰다.
        let dflush = hc.alloc(128 << 20)?;
        let (mut gkt, mut gnt, mut gkk, mut gtt) = (
            (k / 16) as i32,
            (n / 16) as i32,
            krate as i32,
            t_rows as i32,
        );
        let (mut a0, mut a1, mut a2) = (dah, dtre, dbat);
        let mut tg: Vec<f64> = Vec::new();
        for _ in 0..3 {
            hc.l2_flush(dflush, 128 << 20)?;
            hc.sync()?;
            let t0 = std::time::Instant::now();
            const GVM: [&str; 9] = [
                "",
                "exl3_gemv_m1",
                "exl3_gemv_m2",
                "exl3_gemv_m3",
                "exl3_gemv_m4",
                "exl3_gemv_m5",
                "exl3_gemv_m6",
                "exl3_gemv_m7",
                "exl3_gemv_m8",
            ];
            hc.launch3(
                GVM[t_rows],
                ((n / 16) / 8) as u32,
                16,
                1,
                128,
                &mut [
                    &mut a0 as *mut *mut u8 as *mut _,
                    &mut a1 as *mut *mut u8 as *mut _,
                    &mut a2 as *mut *mut u8 as *mut _,
                    &mut gkt as *mut i32 as *mut _,
                    &mut gnt as *mut i32 as *mut _,
                    &mut gkk as *mut i32 as *mut _,
                    &mut gtt as *mut i32 as *mut _,
                ],
            )?;
            let (mut c0, mut c1, mut c2) = (dbat, dsvh, dy);
            let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, 16i32, n as i32);
            hc.launch3(
                "exl3_had_out",
                (n / 128) as u32,
                t_rows as u32,
                1,
                128,
                &mut [
                    &mut c0 as *mut *mut u8 as *mut _,
                    &mut c1 as *mut *mut u8 as *mut _,
                    &mut c2 as *mut *mut u8 as *mut _,
                    &mut nch as *mut i32 as *mut _,
                    &mut nsg as *mut i32 as *mut _,
                    &mut nst as *mut i32 as *mut _,
                ],
            )?;
            hc.sync()?;
            tg.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let mut gb = vec![0u8; t_rows * n * 4];
        hc.d2h(&mut gb, dy)?;
        hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let gmf: &[f32] =
            unsafe { std::slice::from_raw_parts(gb.as_ptr() as *const f32, t_rows * n) };
        let mut gmd = 0f32;
        for (si, &r) in samp.iter().enumerate() {
            for i in 0..n {
                gmd = gmd.max((gmf[r * n + i] - want[si][i]).abs());
            }
        }
        // 가중치 스트리밍 관점(GB/s): tre는 체인 전체에서 1회 판독(콜드 — L2 플러시 후).
        let tre_gb = tre.len() as f64 / 1e9;
        eprintln!(
            "  [gvmdbg] gemv_m T={t_rows}: {:.1}ms(중앙값, 콜드 L2플러시) · tre {:.0}GB/s · y-vs-ref maxdiff={gmd:.3e}",
            tg[1],
            tre_gb / (tg[1] / 1000.0),
        );
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tf = 2.0 * k as f64 * n as f64 * t_rows as f64 / 1e12;
    Ok(format!(
        "hip-gemm T={t_rows} {key_sel}(k={k},n={n}): 샘플 maxdiff={worst:.3e} · gemm2 {:.1}ms = {:.1} TF",
        ts[1],
        tf / (ts[1] / 1000.0)
    ))
}
// 마커 hgm
// 마커 hb1
// 마커 ps1
// 마커 3w
// 마커 3wb
// 마커 sb1

// ── EXL3 hip GDN 체인 프로브(모듈 5) ── conv→l2perm→scan→gate 4커널 종단.
// vk에서 비트검증된 Rust 미러(체인 프로브와 동일 산술)와 대조.
