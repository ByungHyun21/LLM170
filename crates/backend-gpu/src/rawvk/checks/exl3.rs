//! EXL3 vk GEMV 체커 (plans/118 §3-1) — 트렐리스 디코드 커널 3종
//! (had_in → gemv → had_out)의 GPU↔CPU 상호검증 + 속도 측정.
//!
//! CPU 미러는 커널과 동일 반올림 지점(f16 pre-scale, f16 A_had 저장,
//! f32 누산) — 차이는 f32 FMA 수축 수준(rel ~1e-6)이어야 한다.

use crate::rawvk::context::VkCtx;
use half::f16;

/// f32 자연 순서 WHT-128 (커널 버터플라이와 동일 순서).
fn had128_f32(v: &mut [f32]) {
    let mut w = 1usize;
    while w < 128 {
        let mut blk = 0;
        while blk < 128 {
            for i in 0..w {
                let a = v[blk + i];
                let b = v[blk + w + i];
                v[blk + i] = a + b;
                v[blk + w + i] = a - b;
            }
            blk += 2 * w;
        }
        w *= 2;
    }
}

const R_SCALE: f32 = 0.08838834764831845;

/// `llm170 exl3-vk-check [exl3_dir] [tensor-key]` — 기본 27B gate_proj.
pub fn exl3_vk_check(exl3_dir: &str, key: &str) -> Result<String, String> {
    let ar =
        llm170_exl3::StArchive::open(std::path::Path::new(exl3_dir)).map_err(|e| e.to_string())?;
    let w = llm170_exl3::Exl3Linear::load(&ar, key).map_err(|e| e.to_string())?;
    if w.half_k {
        return Err("반정수 bpw 미지원 (27B 4.0bpw에 없음)".into());
    }
    let (k, n) = (w.k, w.n);
    let reps = 20usize;

    // ── 합성 입력 (LCG) ──
    let mut seed = 0x00e3_1700_0000_0001u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xf32: Vec<f32> = (0..k).map(|_| lcg()).collect();
    let x16: Vec<u16> = xf32.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let suh16: Vec<u16> = w.suh.iter().map(|v| v.to_bits()).collect();
    let svh16: Vec<u16> = w.svh.iter().map(|v| v.to_bits()).collect();

    // ── CPU 참조 (커널 미러) ──
    let t0 = std::time::Instant::now();
    // had_in: f16 pre-scale → H → ×1/√128 → f16.
    let mut ah = vec![0f32; k];
    for ch in 0..k / 128 {
        let mut v = [0f32; 128];
        for (j, vv) in v.iter_mut().enumerate() {
            let i = ch * 128 + j;
            let pre = f16::from_f32(f16::from_bits(x16[i]).to_f32() * w.suh[i].to_f32());
            *vv = pre.to_f32();
        }
        had128_f32(&mut v);
        for j in 0..128 {
            ah[ch * 128 + j] = f16::from_f32(v[j] * R_SCALE).to_f32();
        }
    }
    // gemv: 타일 디코드 f32 누산.
    let mut s = vec![0f32; n];
    let mut tile = [0f32; 256];
    for nt in 0..n / 16 {
        for c in 0..16 {
            let mut acc = 0f32;
            for kt in 0..k / 16 {
                w.tile(kt, nt, &mut tile);
                for r in 0..16 {
                    acc += ah[kt * 16 + r] * tile[r * 16 + c];
                }
            }
            s[nt * 16 + c] = acc;
        }
    }
    // had_out: H → ×1/√128 × svh (f32).
    let mut y_ref = vec![0f32; n];
    for ch in 0..n / 128 {
        let mut v = s[ch * 128..ch * 128 + 128].to_vec();
        had128_f32(&mut v);
        for j in 0..128 {
            y_ref[ch * 128 + j] = v[j] * R_SCALE * w.svh[ch * 128 + j].to_f32();
        }
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    // ── GPU ──
    let mut ctx = VkCtx::new()?;
    let mut up = |bytes: &[u8]| -> Result<crate::rawvk::context::VkBuf, String> {
        let b = ctx.alloc(bytes.len())?;
        // SAFETY: alloc 영구 매핑 ptr — 크기 일치, 업로드 후 동기 런치 전.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.ptr, bytes.len()) };
        Ok(b)
    };
    let xb = up(&pack_u16(&x16))?;
    let suhb = up(&pack_u16(&suh16))?;
    let svhb = up(&pack_u16(&svh16))?;
    let treb = up(&w.trellis)?;
    let ahb = ctx.alloc(k * 2)?;
    // 디버그 우회: had_in 스킵·CPU ah 직접 업로드 — 스테이지 격리.
    let had_cpu = llm170_diag::flag::eq1("LLM170_EXL3_HADCPU");
    if had_cpu {
        let mut ahp = vec![0u8; k * 2];
        for i in 0..k {
            ahp[2 * i..2 * i + 2].copy_from_slice(&f16::from_f32(ah[i]).to_bits().to_le_bytes());
        }
        // SAFETY: 매핑 업로드(크기 일치).
        unsafe { std::ptr::copy_nonoverlapping(ahp.as_ptr(), ahb.ptr, k * 2) };
    };
    let sb = ctx.alloc(n * 4)?;
    let yb = ctx.alloc(n * 4)?;

    let (_d1, pl1, _p1, ds1, pipe1) =
        ctx.pipeline(include_bytes!("../spv/exl3_had_in.spv"), 3, 4)?;
    ctx.bind_bufs(ds1, &[xb.buf, suhb.buf, ahb.buf]);
    let (_d2, pl2, _p2, ds2, pipe2) =
        ctx.pipeline(include_bytes!("../spv/exl3_gemv.spv"), 3, 12)?;
    ctx.bind_bufs(ds2, &[ahb.buf, treb.buf, sb.buf]);
    let (_d3, pl3, _p3, ds3, pipe3) =
        ctx.pipeline(include_bytes!("../spv/exl3_had_out.spv"), 3, 4)?;
    ctx.bind_bufs(ds3, &[sb.buf, svhb.buf, yb.buf]);

    let ktiles = (k / 16) as u32;
    let ntiles = (n / 16) as u32;
    let ngroups = ntiles.div_ceil(8);
    let push1 = push_u32s(&[k as u32 / 128]);
    let push2 = push_u32s(&[ktiles, ntiles, w.krate]);
    let push3 = push_u32s(&[n as u32 / 128]);

    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        if !had_cpu {
            ctx.run_rw(
                pl1,
                ds1,
                pipe1,
                &push1,
                (k / 128) as u32,
                1,
                1,
                &[xb.buf, suhb.buf],
                &[ahb.buf],
            )?;
        }
        ctx.run_rw(
            pl2,
            ds2,
            pipe2,
            &push2,
            ngroups,
            1,
            1,
            &[ahb.buf, treb.buf],
            &[sb.buf],
        )?;
        ctx.run_rw(
            pl3,
            ds3,
            pipe3,
            &push3,
            (n / 128) as u32,
            1,
            1,
            &[sb.buf, svhb.buf],
            &[yb.buf],
        )?;
    }
    ctx.end_batch_wait()?;
    let gpu_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;

    // 스테이지별 판독·비교 (진단: 어느 커널이 틀렸는지 국소화).
    // SAFETY: 영구 매핑 버퍼 — end_batch_wait 후 판독.
    let mut ah_gpu = vec![0u8; k * 2];
    unsafe { std::ptr::copy_nonoverlapping(ahb.ptr as *const u8, ah_gpu.as_mut_ptr(), k * 2) };
    let mut s_gpu = vec![0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(sb.ptr as *const f32, s_gpu.as_mut_ptr(), n) };
    let mut y_gpu = vec![0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(yb.ptr as *const f32, y_gpu.as_mut_ptr(), n) };

    let cmp_ah = {
        let (mut m, mut cnt) = (0f32, 0usize);
        for i in 0..k {
            let g = f16::from_bits(u16::from_le_bytes([ah_gpu[2 * i], ah_gpu[2 * i + 1]])).to_f32();
            m = m.max((g - ah[i]).abs());
            cnt += (g == ah[i]) as usize;
        }
        if llm170_diag::flag::on("LLM170_EXL3_HADDBG") {
            let mut dump = String::from("ah[0..8] gpu:");
            for i in 0..8 {
                let g =
                    f16::from_bits(u16::from_le_bytes([ah_gpu[2 * i], ah_gpu[2 * i + 1]])).to_f32();
                dump.push_str(&format!(" {g:.4}"));
            }
            dump.push_str("\n          cpu:");
            for i in 0..8 {
                dump.push_str(&format!(" {:.4}", ah[i]));
            }
            eprintln!("{dump}");
            // 상이 샘플 비트 패턴 (반올림 모드 진단).
            let mut shown = 0;
            for i in 0..k {
                let gb = u16::from_le_bytes([ah_gpu[2 * i], ah_gpu[2 * i + 1]]);
                let cb = f16::from_f32(ah[i]).to_bits();
                if gb != cb && shown < 6 {
                    eprintln!(
                        "  diff[{i}]: gpu={gb:#06x} ({}) cpu={cb:#06x} d={}",
                        f16::from_bits(gb).to_f32(),
                        (gb as i32 - cb as i32)
                    );
                    shown += 1;
                }
            }
        }
        (m, cnt)
    };
    let cmp_s = {
        let mut m = 0f32;
        for i in 0..n {
            m = m.max((s_gpu[i] - s[i]).abs());
        }
        m
    };
    let (mut max_abs, mut max_rel) = (0f32, 0f32);
    for i in 0..n {
        let d = (y_gpu[i] - y_ref[i]).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / y_ref[i].abs().max(1e-3));
    }
    // 이론 대역폭 기준 하한(tre 독점 가정) — 참고 정보.
    let bytes_per_call = w.trellis.len() as f64;
    let gbps = bytes_per_call / (gpu_us * 1e-6) / 1e9;
    // 판정: had_in 비트 동일·gemv/out FMA 수축 수준(abs ≤ 1e-5, y 스케일
    // 대비) — 상회 시 디코드 회귀. 1e-5는 f32 FMA·스테이지 순서차 상한.
    let ok = cmp_ah.0 == 0.0 && cmp_s <= 1e-5 && max_abs <= 1e-4;
    let report = format!(
        "exl3-vk-check {key}: k={k} n={n} K={}\n  had_in: max_abs={:.3e} (f16 일치 {}/{k})\n  gemv : max_abs={:.3e}\n  out  : max_abs={max_abs:.3e} max_rel={max_rel:.3e}\n  gpu {gpu_us:.0} µs/step (trellis {:.1} MB → {gbps:.0} GB/s)\n  cpu 참조 {cpu_ms:.0} ms",
        w.krate,
        cmp_ah.0,
        cmp_ah.1,
        cmp_s,
        bytes_per_call / 1e6,
    );
    if ok {
        Ok(report)
    } else {
        Err(format!(
            "{report}\nFAIL: 허용치 초과 (had_in 비트 동일·gemv≤1e-5·out≤1e-4)"
        ))
    }
}

fn pack_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u32 푸시 상수 조립 헬퍼.
fn push_u32s(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
