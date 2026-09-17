//! probes/gemm — GEMM 벤치·타일 검증 (probes.rs에서 이동, plans/78 R3).

use super::*;
use super::misc::half_f32;
use super::wmma::half_bits;

/// 배치 mm 타이밍 — gy=1 대비 gy=t 배율.
pub fn mm_batch_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let x: Vec<f32> = (0..n_in * 64).map(|_| lcg()).collect();
    let xd = ctx.alloc(n_in * 64 * 4)?;
    ctx.h2d(xd, bytemuck::cast_slice(&x))?;
    let xq = ctx.alloc(xq_w * 4 * 64)?;
    ctx.quant_q8_b(xd, xq, n_in, xq_w, 64)?;
    let out = ctx.alloc(n_out * 4 * 64)?;
    let mut msg = String::new();
    for &t in &[1usize, 8, 64] {
        ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, t)?;
        ctx.sync()?;
        let reps = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, t)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        msg += &format!("t={}: {:.3}ms ({:.0} GB/s-equiv)\n", t, dt * 1e3, w.data.len() as f64 / dt / 1e9);
    }
    Ok(msg)
}

/// 타일 커널 검증+타이밍 — gemm_q5k_bt vs 미러.
pub fn mm_tile_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let t = std::env::var("LLM170_TILE_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let mut xs = Vec::new();
    let mut q8s = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        q8s.push(llm170_core::quant::quantize_row_q8_ref(&x));
        xs.push(x);
    }
    let mut xq_h: Vec<u32> = Vec::new();
    for tok in &q8s {
        for blk in tok {
            for c in 0..8 {
                let base = c * 4;
                xq_h.push((blk.qs[base] as u32 & 0xFF) | ((blk.qs[base+1] as u32 & 0xFF) << 8) | ((blk.qs[base+2] as u32 & 0xFF) << 16) | ((blk.qs[base+3] as u32 & 0xFF) << 24));
            }
        }
        for blk in tok { xq_h.push(blk.d.to_bits()); }
        for blk in tok {
            let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
            let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
            xq_h.push(s0 as u32);
            xq_h.push(s1 as u32);
        }
    }
    let xq = ctx.alloc(xq_h.len() * 4)?;
    if let Some(f) = std::env::var_os("LLM170_XQN_FILE") {
        let bytes = std::fs::read(&f).unwrap();
        assert_eq!(bytes.len(), xq_h.len() * 4, "덤프 크기 불일치: {} vs {}", bytes.len(), xq_h.len() * 4);
        ctx.h2d(xq, &bytes)?;
        eprintln!("xq 리플레이: {}", f.to_string_lossy());
    } else {
        ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    }
    let out = ctx.alloc(n_out * 4 * t)?;
    let _wp = wd as *mut std::ffi::c_void;
    let _op = out as *mut std::ffi::c_void;
    let _xp = xq as *mut std::ffi::c_void;
    let _ni = n_in as i32;
    let _no = n_out as i32;
    let _xw = xq_w as i32;
    let _tt = t as i32;
    ctx.gemm_tile(xq, wd, kt_d, w.ty as u32, n_in, n_out, xq_w, t, out)?;
    ctx.sync()?;
    let mut o = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o).as_mut(), out)?;
    // 미러 검증
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut mism = 0;
    let mut first_dbg = String::new();
    for ti in 0..t {
        for oo in 0..n_out.min(256) {
            let row = &w.data[oo * rb..];
            let c = match w.ty {
                llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &q8s[ti]),
                _ => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &q8s[ti]),
            };
            if c.to_bits() != o[ti * n_out + oo].to_bits() {
                mism += 1;
                if mism == 1 {
                    first_dbg = format!("ti={ti} o={oo}: cpu={c:.7e} gpu={:.7e}", o[ti * n_out + oo]);
                }
            }
        }
    }
    eprintln!("dbg: {first_dbg}");
    // 타이밍
    let reps = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        ctx.gemm_tile(xq, wd, kt_d, w.ty as u32, n_in, n_out, xq_w, t, out)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    Ok(format!("tile t={t}: 불일치 {mism}/{} (첫 256행×t) — {:.3}ms → {:.0} GB/s-equiv, 토큰당 {:.1}µs", n_out.min(256) * t, dt * 1e3, w.data.len() as f64 / dt / 1e9, dt * 1e6 / t as f64))
}

/// `f16-map` — `.co` f16 GEMM(`gemm_f16_v4`, 래퍼 `gemm_f16_deq`)의 k-축 매핑을
/// **블록별로 분리 측정**한다. 가중치를 한 32원소 블록으로 제한(b)하고 원-핫
/// 활성(k)을 넣으면, 출력의 1 위치가 곧 "x의 k가 어느 열과 곱해지는가"다.
/// (전체 항등으로 한 번에 재면 블록 간 간섭이 섞여 전단사가 깨진다 — plans/65 §12)
pub fn f16_map(n_in_arg: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let (n_out, n_in) = (256usize, n_in_arg.max(128));
    let wd = ctx.alloc(n_out * (n_in / 32) * 34)?;
    let xd = ctx.alloc(n_in * 4 * 8)?;   // t ≤ 8 여유
    let od = ctx.alloc(n_out * 4 * 512)?;   // 커널은 t를 128 사분면 경계까지 쓴다(§29)
    let mut out = String::new();
    for b in 0..8usize {
        // 가중치: 블록 b에만 항등(그 블록의 요소 j가 행 o=j에)
        let mut w = vec![0u8; n_out * (n_in / 32) * 34];
        for o in 0..n_out {
            let blk = &mut w[(o * (n_in / 32) + b) * 34..][..34];
            blk[0] = 0x00;
            blk[1] = 0x3C;
            for l in 0..32 {
                blk[2 + l] = if o == b * 32 + l { 1 } else { 0 };
            }
        }
        ctx.h2d(wd, &w)?;
        let mut pairs = Vec::new();
        for k in 0..n_in {
            let mut x = vec![0.0f32; n_in];
            x[k] = 1.0;
            ctx.h2d(xd, bytemuck::cast_slice(&x))?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut o = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut o), od as *const u8)?;
            let nz: Vec<usize> = o
                .iter()
                .enumerate()
                .filter(|(_, v)| v.abs() > 0.25)
                .map(|(i, _)| i)
                .collect();
            match nz.len() {
                1 => pairs.push((k, nz[0])),
                0 => pairs.push((k, usize::MAX)),
                _ => pairs.push((k, 1000 + nz[0])),
            }
        }
        let distinct = pairs.iter().filter(|(_, o)| *o < 1000).count();
        let holes = pairs.iter().filter(|(_, o)| *o == usize::MAX).count();
        let multi = pairs.iter().filter(|(_, o)| *o >= 1000).count();
        // 값 검증: 같은 블록 구성에서 전 원소 = 1.0 (d=1, q=1) → x=전부 1이면 출력 = n_in
        {
            let mut wv = vec![0u8; n_out * (n_in / 32) * 34];
            for o in 0..n_out {
                for sb in 0..n_in / 32 {
                    let blk2 = &mut wv[(o * (n_in / 32) + sb) * 34..][..34];
                    blk2[0] = 0x00;
                    blk2[1] = 0x3C; // d = 1.0
                    for l in 0..32 {
                        blk2[2 + l] = 1;
                    }
                }
            }
            let wd1 = ctx.alloc(wv.len())?;
            ctx.h2d(wd1, &wv)?;
            let xones = vec![1.0f32; n_in];
            ctx.h2d(xd, bytemuck::cast_slice(&xones))?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd1 as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut ov = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut ov), od as *const u8)?;
            // 두 번째: d = 0.5(f16 0x3800), q = 2 → 같은 1.0 (스케일 경로 검증)
            let mut wv2 = wv.clone();
            for o in 0..n_out {
                for sb in 0..n_in / 32 {
                    let blk2 = &mut wv2[(o * (n_in / 32) + sb) * 34..][..34];
                    blk2[0] = 0x00;
                    blk2[1] = 0x38;
                    for l in 0..32 {
                        blk2[2 + l] = 2;
                    }
                }
            }
            let wd2 = ctx.alloc(wv2.len())?;
            ctx.h2d(wd2, &wv2)?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd2 as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut ov2 = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut ov2), od as *const u8)?;
            // 실효 k-범위: 전원소 1 가중치 + 원-핫 x(j) → out[0]=1이면 j는 기여, 0이면 범위 밖
            let mut contrib = Vec::new();
            let mut missing = Vec::new();
            for j in 0..n_in {
                let mut xo = vec![0.0f32; n_in];
                xo[j] = 1.0;
                ctx.h2d(xd, bytemuck::cast_slice(&xo))?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut oo = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut oo), od as *const u8)?;
                if oo[0].abs() > 0.25 {
                    contrib.push(j);
                } else {
                    missing.push(j);
                }
            }
            // 부호/블록별 스케일 검증: 블록마다 q = l-16 (합 -16), d_블록 = (sb+1)/16
            //  → 기대 out[0] = Σ_블록 d_블록 × (-16)
            let mut wv3 = vec![0u8; n_out * (n_in / 32) * 34];
            let mut expect = 0.0f32;
            for sb in 0..n_in / 32 {
                let dv = ((sb % 16) + 1) as f32 / 16.0;
                for o in 0..n_out {
                    let blk2 = &mut wv3[(o * (n_in / 32) + sb) * 34..][..34];
                    let h = half_bits(dv);
                    blk2[0] = (h & 0xFF) as u8;
                    blk2[1] = (h >> 8) as u8;
                    let mut ssum = 0i32;
                    for l in 0..32 {
                        let q = l as i32 - 16;
                        blk2[2 + l] = q as u8;
                        ssum += q;
                    }
                    if o == 0 {
                        expect += dv * ssum as f32;
                    }
                }
            }
            // 비영 블록 수를 늘려가며 관측(인덱싱 버그의 패턴을 드러낸다)
            let mut per_nb = String::new();
            for nb in [1usize, 2, 8] {
                let mut wv4 = vec![0u8; n_out * (n_in / 32) * 34];
                let mut exp4 = 0.0f32;
                for sb in 0..n_in / 32 {
                    let on = sb < nb;
                    // d는 모든 블록 동일(1/16) — q 패턴만 불균일하게 두어 "블록별 d"와
                    // "블록 0 이상"을 분리한다.
                    let dv = if on { 0.0625f32 } else { 0.0 };
                    for o in 0..n_out {
                        let blk2 = &mut wv4[(o * (n_in / 32) + sb) * 34..][..34];
                        let h = half_bits(dv);
                        blk2[0] = (h & 0xFF) as u8;
                        blk2[1] = (h >> 8) as u8;
                        let mut ssum = 0i32;
                        for l in 0..32 {
                            let q = if on { l as i32 - 16 } else { 0 };
                            blk2[2 + l] = q as u8;
                            ssum += q;
                        }
                        if o == 0 {
                            exp4 += dv * ssum as f32;
                        }
                    }
                }
                let wd4 = ctx.alloc(wv4.len())?;
                ctx.h2d(wd4, &wv4)?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd4 as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut ov4 = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut ov4), od as *const u8)?;
                per_nb += &format!(" nb={nb}: {:.3}/{exp4:.3}", ov4[0]);
            }
            let wd3 = ctx.alloc(wv3.len())?;
            ctx.h2d(wd3, &wv3)?;
            ctx.gemm_f16_deq(8, xd as *const u8, wd3 as *const u8, n_in, n_out, 1, od)?;
            ctx.sync()?;
            let mut ov3 = vec![0.0f32; n_out];
            ctx.d2h(bytemuck::cast_slice_mut(&mut ov3), od as *const u8)?;
            out += &format!("# 블록별 기여(실측/기대):{per_nb}\n");
            // 덤프용 마지막 호출: nb=1(블록 0만 비영) 패턴. LLM170_DEQ_DUMP=1 이면
            // 이 호출의 wf16이 /tmp/deq_wf16.f16 에 남는다.
            {
                let mut wv5 = vec![0u8; n_out * (n_in / 32) * 34];
                for sb in 0..n_in / 32 {
                    let on = sb == 0;
                    let dv = if on { 0.0625f32 } else { 0.0 };
                    for o in 0..n_out {
                        let blk2 = &mut wv5[(o * (n_in / 32) + sb) * 34..][..34];
                        let h = half_bits(dv);
                        blk2[0] = (h & 0xFF) as u8;
                        blk2[1] = (h >> 8) as u8;
                        for l in 0..32 {
                            blk2[2 + l] = if on { (l as i32 - 16) as u8 } else { 0 };
                        }
                    }
                }
                let wd5 = ctx.alloc(wv5.len())?;
                ctx.h2d(wd5, &wv5)?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd5 as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut ov5 = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut ov5), od as *const u8)?;
                out += &format!("# 덤프용 nb=1 재호출: out[0]={:.3} (기대 -1.000)\n", ov5[0]);
            }
            // 원소 커버리지 스캔: (행 0, 열 j) 한 원소만 1.0, x=전부 1 → out[0]=1이면
            // 그 j가 GEMM의 읽기 범위 안. j별로 **다른 포인터**를 써야 f16 캐시(포인터 키)를
            // 피한다 — 257행 버퍼에서 j번째 행을 가중치 시작으로 넘긴다.
            {
                // n_out=1 스캔: 가중치 1행 = 8블록 = 272B. j마다 **별도 슬롯(272B)** 을
                // 포인터로 넘겨 캐시 키(포인터)를 회피하고, 1은 그 슬롯 안의 (j/32, j%32)에 둔다.
                let row_bytes = (n_in / 32) * 34;
                let big = ctx.alloc((n_in + 1) * row_bytes)?;
                let zeros = vec![0u8; (n_in + 1) * row_bytes];
                ctx.h2d(big, &zeros)?;
                let mut covered = Vec::new();
                let mut holes = Vec::new();
                for j in 0..n_in {
                    let mut one = vec![0u8; 34];
                    one[0] = 0x00;
                    one[1] = 0x3C;
                    one[2 + (j % 32)] = 1;
                    let wptr = unsafe { big.add(j * row_bytes + (j / 32) * 34) };
                    ctx.h2d(wptr, &one)?;
                    let wp = unsafe { big.add(j * row_bytes) };
                    ctx.gemm_f16_deq(8, xd as *const u8, wp as *const u8, n_in, 1, 1, od)?;
                    ctx.sync()?;
                    let mut oo = vec![0.0f32; 1];
                    ctx.d2h(bytemuck::cast_slice_mut(&mut oo), od as *const u8)?;
                    if oo[0].abs() > 0.25 {
                        covered.push(j);
                    } else {
                        holes.push(j);
                    }
                }
                out += &format!(
                    "# 원소 커버리지: {}/{} (구멍 앞 12: {:?}, covered: {:?})\n",
                    covered.len(),
                    n_in,
                    &holes[..12.min(holes.len())],
                    covered
                );
            }
            out += &format!(
                "# 값검증: 전원소1 → {} (기대 {n_in}) / d=0.5,q=2 → {} (기대 {n_in}) / 부호·스케일 → {} (기대 {expect:.3})\n",
                ov[0], ov2[0], ov3[0]
            );
            out += &format!(
                "# 실효 k범위: 기여 {}개 (앞 12: {:?}) / 누락 {}개 (앞 12: {:?})\n",
                contrib.len(),
                &contrib[..12.min(contrib.len())],
                missing.len(),
                &missing[..12.min(missing.len())]
            );
            // x-측 매핑: 행 0의 각 열 j에 라벨 (j mod 127)+1 을 심고(q8_0 d=1/127),
            // 원-핫 x(j)의 출력값 × 127 = 짝지어진 열 → x가 어디로 가는지 값으로 읽힌다.
            let mut wl = vec![0u8; n_out * (n_in / 32) * 34];
            for sb in 0..n_in / 32 {
                let blk2 = &mut wl[sb * 34..][..34];
                blk2[0] = 0x00;
                // d = 1/127 ≈ 0x1C04? → 대신 d=1 로 두고 q 값 자체를 라벨로 쓴다(출력=q).
                blk2[1] = 0x3C;
                for l in 0..32 {
                    let j = sb * 32 + l;
                    blk2[2 + l] = ((j % 127) + 1) as u8;
                }
            }
            ctx.h2d(wd, &wl)?;
            let mut xmap = Vec::new();
            for j in [0usize, 1, 2, 5, 16, 31, 32, 33, 63, 64, 127, 128, 200, 255] {
                if j >= n_in {
                    continue;
                }
                let mut xo = vec![0.0f32; n_in];
                xo[j] = 1.0;
                ctx.h2d(xd, bytemuck::cast_slice(&xo))?;
                ctx.gemm_f16_deq(8, xd as *const u8, wd as *const u8, n_in, n_out, 1, od)?;
                ctx.sync()?;
                let mut oo = vec![0.0f32; n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut oo), od as *const u8)?;
                // out[0]만 보면 모순이 생긴다(plans/65 §18) — 전체 비영 분포를 찍는다.
                let nz2: Vec<(usize, f32)> = oo
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.abs() > 0.25)
                    .map(|(i, &v)| (i, v))
                    .collect();
                xmap.push((j, nz2));
            }
            out += &format!("# x-측 매핑(j → out[0]): {xmap:?}\n");
            // t>1 검증: 전부-1 가중치, x는 t행 — 행 r의 one-hot j가 행 r로 나와야 한다.
            {
                let tt = 256usize;   // 128 경계를 넘겨 행 블록 z>0까지 검증
                let mut xt = vec![0.0f32; tt * n_in];
                let rows = [0usize, 1, 127, 128, 129, 255];
                for &r in rows.iter() {
                    xt[r * n_in + (r % 7 + 3)] = 1.0;
                }
                let xd2 = ctx.alloc(tt * n_in * 4)?;
                ctx.h2d(xd2, bytemuck::cast_slice(&xt))?;
                let odt = ctx.alloc(tt * n_out * 4)?;
                let _ = &xd;
                ctx.gemm_f16_deq(8, xd2 as *const u8, wd1 as *const u8, n_in, n_out, tt, odt)?;
                ctx.sync()?;
                let mut ot = vec![0.0f32; tt * n_out];
                ctx.d2h(bytemuck::cast_slice_mut(&mut ot), odt as *const u8)?;
                let mut info = String::new();
                for &r in rows.iter() {
                    let nz: Vec<(usize, f32)> = ot[r * n_out..(r + 1) * n_out]
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.abs() > 0.25)
                        .map(|(i, &v)| (i, v))
                        .collect();
                    info += &format!(
                        " r{r}: nz={} first={:?} val={:.2}",
                        nz.len(),
                        nz.first().map(|(i, _)| *i),
                        nz.first().map(|(_, v)| *v).unwrap_or(0.0)
                    );
                }
                out += &format!("# t>1 검증(t={tt}):{info}\n");
            }
        }
        out += &format!("# blk={b}: 1:1={distinct} 빈칸={holes} 다중={multi}\n");
        if b == 0 {
            for (k, o) in pairs.iter() {
                out += &format!("{k}:{o}\n");
            }
        }
    }
    Ok(out)
}

/// `f16-bench [rows] [n_in] [n_out] [reps]` — 기존 `.co` f16 GEMM(`gemm_f16_v4`)을
/// 직접 측정한다. q4k-bench와 같은 형상으로 재면 "텐서코어 경로의 상한"이 나온다
/// (roof-test mfma1 L1-fed 24.9 TFLOPS). 새 커널 없이 경로 가치를 판정하는 용도.
pub fn f16_bench(rows: usize, n_in: usize, n_out: usize, reps: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let xq_w = crate::rawhip::q4acc::xq_words(n_in);
    let xdev = ctx.alloc(xq_w * rows * 4)?;
    let wdev = ctx.alloc(n_out * n_in * 2)?;
    let odev = ctx.alloc(n_out * rows * 4)?;
    let x = vec![0x11u8; xq_w * rows * 4];
    let w = vec![0x22u8; n_out * n_in * 2];
    ctx.h2d(xdev, &x)?;
    ctx.h2d(wdev, &w)?;
    let fns = &ctx.fns;
    let fm = *fns.get("gemm_f16_v4").ok_or("gemm_f16_v4 없음(co/mmq2.co 미로드)")?;
    let launch = || -> Result<(), String> {
        unsafe {
            let mut a1 = xdev as *mut std::ffi::c_void;
            let mut a2 = wdev as *mut std::ffi::c_void;
            let mut a3 = odev as *mut std::ffi::c_void;
            let (mut ni, mut no, mut xw, mut tt) =
                (n_in as i32, n_out as i32, xq_w as i32, rows.min(128) as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut a1) as *mut _ as *mut std::ffi::c_void,
                (&mut a2) as *mut _ as *mut std::ffi::c_void,
                (&mut a3) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            let e = hip::hipModuleLaunchKernel(
                fm,
                n_out.div_ceil(128) as u32,
                1,
                rows.div_ceil(128) as u32,
                256,
                1,
                1,
                0,
                ctx.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if e != hip::hipError_t_hipSuccess {
                return Err(format!("gemm_f16_v4 launch {e:?}"));
            }
        }
        Ok(())
    };
    launch()?;
    ctx.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        launch()?;
    }
    ctx.sync()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let gb = (n_out * n_in * 2) as f64 / (ms / 1e3) / 1e9;
    let flops = 2.0 * (n_out * n_in * rows) as f64 / (ms / 1e3) / 1e12;
    Ok(format!(
        "# f16-bench t={rows} {n_in}x{n_out}: {ms:.3}ms/호출 ({gb:.1}GB/s, {flops:.1} TFLOPS)"
    ))
}

/// `q4k-bench [rows] [n_in] [n_out] [reps]` — q4_K GEMM 형상 격리 계측.
/// 합성 q4_K 텐서로 커널 변형별 실효 대역을 잰다(plans/65 하한 분석의 입력).
pub fn q4k_bench(rows: usize, n_in: usize, n_out: usize, reps: usize) -> Result<String, String> {
    use crate::rawhip::q4acc::Q4Acc;
    use llm170_core::matmul::MatmulHost;
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let n_super = n_in / 256;
    let nblk = n_out * n_super;
    let mut w = vec![0u8; nblk * 144];
    for b in 0..nblk {
        let o = &mut w[b * 144..(b + 1) * 144];
        o[0] = 0x00; o[1] = 0x38;   // d = 0.5
        o[2] = 0x00; o[3] = 0x30;   // dmin = 0.25
        for j in 0..12 { o[4 + j] = ((b * 7 + j * 13) & 0x3F) as u8; }
        for i in 0..128 { o[16 + i] = ((b * 31 + i * 37) & 0xFF) as u8; }
    }
    let xs: Vec<Vec<f32>> = (0..rows)
        .map(|_| (0..n_in).map(|_| lcg()).collect())
        .collect();
    let weight = llm170_core::matmul::Weight {
        data: &w,
        ty: llm170_gguf::GgmlType::Q4K,
        n_in: n_in as u64,
        n_out: n_out as u64,
    };
    let acc = Q4Acc::new()?;
    let mut out = vec![vec![0.0f32; n_out]; rows];
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        acc.matmul_batch(&xs, &weight, &mut out)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let gb = (nblk * 144) as f64 / (ms / 1e3) / 1e9;
    Ok(format!(
        "# q4k-bench t={rows} {n_in}x{n_out}: {ms:.3}ms/호출 ({gb:.1}GB/s 가중치, {}MB)",
        nblk * 144 / 1_000_000
    ))
}

/// `q4k-micro` — q4_K MMQ 타일을 **단일 256원소 슈퍼블록**에서 CPU 미러
/// (`dot_q4k_q8`)와 직접 대조한다. 인덱스 매핑 버그를 값 수준에서 드러낸다.
pub fn q4k_micro() -> Result<String, String> {
    use crate::rawhip::q4acc::Q4Acc;
    use llm170_core::matmul::MatmulHost;
    let (n_out, n_in, t) = (16usize, 256usize, 16usize);
    // 합성 q4_K 블록: d=1.0, dmin=0.5, 6비트 스케일 패턴, 결정적 니블
    let mut blk = vec![0u8; 144];
    blk[0] = 0x00; blk[1] = 0x3C;   // d = 1.0
    blk[2] = 0x00; blk[3] = 0x38;   // dmin = 0.5
    for j in 0..12 { blk[4 + j] = (0x15u8.wrapping_mul(j as u8 + 1)) & 0x3F; }
    for i in 0..128 { blk[16 + i] = ((i * 37 + 11) & 0xFF) as u8; }
    let w: Vec<u8> = (0..n_out).flat_map(|o| {
        let mut b = blk.clone();
        b[4] = (b[4].wrapping_add(o as u8)) & 0x3F;
        b
    }).collect();
    let xs: Vec<Vec<f32>> = (0..t)
        .map(|r| (0..n_in).map(|i| (((r * 31 + i) as u64 * 2654435761u64) % 1000) as f32 / 500.0 - 1.0).collect())
        .collect();
    let weight = llm170_core::matmul::Weight {
        data: &w,
        ty: llm170_gguf::GgmlType::Q4K,
        n_in: n_in as u64,
        n_out: n_out as u64,
    };
    let acc = Q4Acc::new()?;
    let mut out = vec![vec![0.0f32; n_out]; t];
    acc.matmul_batch(&xs, &weight, &mut out)?;
    let mut max_abs = 0.0f32;
    let mut first = String::new();
    for r in 0..t {
        let y = llm170_core::quant::quantize_row_q8_ref(&xs[r]);
        for o in 0..n_out {
            let cpu = llm170_core::quant::dot_q4k_q8(&w[o * 144..(o + 1) * 144], &y);
            let gpu = out[r][o];
            let d = (cpu - gpu).abs();
            if d > max_abs { max_abs = d; }
            if first.is_empty() && d > 1e-4 {
                first = format!(" 첫 불일치 r={r} o={o} gpu={gpu:.6} cpu={cpu:.6}");
            }
        }
    }
    Ok(format!(
        "q4k-micro {n_out}x{n_in} t={t}: max_abs={max_abs:.3e} ({}){first}",
        if max_abs < 1e-4 { "일치" } else { "불일치" }
    ))
}

/// `q5-1-bench [rows] [n_in] [n_out] [reps]` — q5_1 GEMM 격리 계측.
/// 실모델 MoE expert-down 형상(20행 × 640 × 2560)을 합성 데이터로 돌려 커널
/// 자체의 시간을 잰다 — KTRACE가 0.24ms/런치를 보고한 그 값과 대조하면
/// 문제가 커널인지 컨텍스트(L2 상태·주변 런치)인지 갈린다.
pub fn q5_1_bench(rows: usize, n_in: usize, n_out: usize, reps: usize) -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    // q5_1 블록 = 32원소(24B). 행당 n_in/32 블록.
    let n_sub = n_in / 32;
    let wrow = n_sub * 24;
    let wbytes = n_out * wrow;
    let xq_w = crate::rawhip::q4acc::xq_words(n_in);
    let xbytes = rows * xq_w * 4;
    let obytes = rows * n_out * 4;
    let wdev = ctx.alloc(wbytes.max(4))?;
    let xdev = ctx.alloc(xbytes.max(4))?;
    let odev = ctx.alloc(obytes.max(4))?;
    let part = ctx.scratch(n_out * 64 * 8)?;
    // 합성: 가중치는 0x3c 패턴(d/m f16 = 1.0/0.0 근사), x는 1
    let w = vec![0x3cu8; wbytes];
    let x = vec![0x01u8; xbytes];
    ctx.h2d(wdev, &w)?;
    ctx.h2d(xdev, &x)?;
    let launch = |kern: &'static str, gx: u32, block: u32| -> Result<(), String> {
        let (mut xp, mut wp, mut pp, mut op) =
            (xdev, wdev, part, odev);
        let (mut ni, mut no, mut xw, mut tt) = (n_in as i32, n_out as i32, xq_w as i32, rows as i32);
        let mut args: Vec<*mut c_void> = vec![
            (&mut xp) as *mut _ as *mut c_void,
            (&mut wp) as *mut _ as *mut c_void,
            (&mut pp) as *mut _ as *mut c_void,
            (&mut op) as *mut _ as *mut c_void,
            (&mut ni) as *mut _ as *mut c_void,
            (&mut no) as *mut _ as *mut c_void,
            (&mut xw) as *mut _ as *mut c_void,
            (&mut tt) as *mut _ as *mut c_void,
        ];
        let (gy, gz) = if kern.ends_with("_t") {
            let nb = n_out.div_ceil(4);
            (nb.min(65535) as u32, nb.div_ceil(65535) as u32)
        } else {
            (n_out.min(65535) as u32, n_out.div_ceil(65535) as u32)
        };
        ctx.launch3(kern, gx, gy, gz, block, &mut args)
    };
    // 워밍업 + 시간
    let mut msg = String::new();
    for (kern, gx, blk) in [
        ("q4_gemm_q5_1", rows as u32, 64u32),
        ("q4_gemm_q5_1_t", rows.div_ceil(16) as u32, 256),
    ] {
        for _ in 0..2 {
            launch(kern, gx, blk)?;
        }
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            launch(kern, gx, blk)?;
        }
        ctx.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let gb = wbytes as f64 / (ms / 1e3) / 1e9;
        let attrs = ctx
            .kern_attrs(kern)
            .map(|(regs, loc, mx)| format!("regs={regs} local={loc}B maxthr={mx}"))
            .unwrap_or_else(|| "attrs 없음".into());
        msg += &format!(
            "# {kern}: {ms:.3}ms/런치 ({gb:.1}GB/s 가중치) rows={rows} {n_in}x{n_out} grid=({gx},{n_out}) blk={blk} {attrs}\n"
        );
    }
    Ok(msg)
}

/// `q4-d2h-bench` — 소형 d2h 비용 격리(프레임 MoE가 ids 20KB를 읽는 데 15.5ms를
/// 쓰고 있었다). 크기별·경로별로 잰다.
pub fn d2h_bench() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let mut out = String::new();
    for &n in &[20 << 10usize, 1 << 20, 8 << 20] {
        let d = ctx.alloc(n)?;
        let mut dst = vec![0u8; n];
        // 워밍업 + 5회 평균
        for _ in 0..2 {
            ctx.d2h(&mut dst, d as *const u8)?;
        }
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            ctx.d2h(&mut dst, d as *const u8)?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
        // 순수 커널 런치 1회 비용(동기 없음) 대조
        let t1 = std::time::Instant::now();
        for _ in 0..5 {
            let _ = ctx.scratch(4);
        }
        let lms = t1.elapsed().as_secs_f64() * 1e3 / 5.0;
        out += &format!("# d2h {}KB: {:.3}ms (scratch {:.3}ms)\n", n >> 10, ms, lms);
    }
    Ok(out)
}

/// plans/70 P1 검증 — f16 dequant-cache GEMM(gemm_q5k_wc)이 인라인 디퀀트판
/// (gemm_q5k_wm)과 **비트 동일** 출력을 내는지 + 처리량 비교. t ≤ 64(wm B16 한계).
pub fn wc_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::ffi::c_void;
    let t = t.clamp(1, 64);
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path)).map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("tensor 없음")?;
    let is_xs = w.ty == llm170_gguf::GgmlType::Iq4Xs;
    if w.ty != llm170_gguf::GgmlType::Q5K && !is_xs {
        return Err(format!("wc-check: q5_K/iq4_xs 전용 (ty={:?})", w.ty));
    }
    let ctx = RawCtx::new()?;
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    if n_in % 256 != 0 {
        return Err("wc-check: n_in이 256의 배수가 아님".into());
    }
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    // f16 캐시 생성(1회) — xs는 ktab2 LUT 필요
    let ac = ctx.alloc(n_out * n_in * 2)?;
    // 진단: 디퀀트 커널 값 검증 — CPU 대조 (xs만, o=0 앞 8원소)
    if is_xs && std::env::var_os("LLM170_WC_CPUCHK").is_some() {
        ctx.sync()?;
        let mut ac_host = vec![0u16; n_in.min(64)];
        ctx.d2h(unsafe { std::slice::from_raw_parts_mut(ac_host.as_mut_ptr() as *mut u8, ac_host.len() * 2) }, ac)?;
        let f16v = |bits: u16| half_f32(bits);
        let wq = w.data;
        let _blocks = n_in >> 8; // 진단용(현재 미사용) — 2026-09-17 clippy
        let mut cpu = vec![0f32; 8];
        for (sb2, cv) in cpu.iter_mut().enumerate() {
            let ib = sb2 & 7;
            let wb = (sb2 >> 3) * 136;
            let wqf = wb >> 2;
            let w0 = u32::from_le_bytes([wq[wqf*4], wq[wqf*4+1], wq[wqf*4+2], wq[wqf*4+3]]);
            let d = f16v((w0 & 0xFFFF) as u16);
            let w1 = u32::from_le_bytes([wq[(wqf+1)*4], wq[(wqf+1)*4+1], wq[(wqf+1)*4+2], wq[(wqf+1)*4+3]]);
            let ls = ((w1 >> ((ib >> 1) * 8 + (ib & 1) * 4)) & 0xF) as i32
                  | ((((w0 >> 16) >> (2 * ib)) & 3) as i32) << 4;
            let ds0 = d * (ls - 32) as f32;
            let qw = (wb + 8 + ib * 16) >> 2;
            let k = sb2 * 4; // sb2=0..7 → k=0,4,8..28 (원소 8개 샘플)
            let qv = u32::from_le_bytes([wq[(qw + ((k & 15) >> 2))*4], wq[(qw + ((k & 15) >> 2))*4+1], wq[(qw + ((k & 15) >> 2))*4+2], wq[(qw + ((k & 15) >> 2))*4+3]]);
            let byte_v = ((qv >> ((k & 3) * 8)) & 0xFF) as u8;
            let kt = llm170_core::ktab2_packed();
            let tt2 = kt[byte_v as usize];
            let val = if k < 16 { ((tt2 & 0xFF) as i8) as i32 } else { ((tt2 >> 8) as i8) as i32 };
            *cv = val as f32 * ds0;
        }
        eprintln!("# wc-cpuchk o=0 k=0..28(4씩): cpu={:?}", &cpu);
        eprintln!("# wc-cpuchk        ac(f16)={:?}",
            (0..8usize).map(|i| f16v(ac_host[i*4])).collect::<Vec<_>>());
        // 전체 체크섬 — 어디든 썼는지
        let mut all16 = vec![0u16; n_out * n_in];
        let _ = ctx.d2h(unsafe { std::slice::from_raw_parts_mut(all16.as_mut_ptr() as *mut u8, all16.len() * 2) }, ac);
        let nz = all16.iter().filter(|&&v| v != 0).count();
        let mut sum = 0f64;
        for &v in all16.iter() { sum += f16v(v) as f64; }
        eprintln!("# wc-cpuchk 전체: nonzero {nz}/{} sum={sum:.3}", all16.len());
    }
    let (dq_kern, wm_kern, wc_kern, warg) = if is_xs {
        ("dequant_f16_xs", "gemm_xs_wm", "gemm_xs_wc", {
            let kt: Vec<u32> = llm170_core::ktab2_packed();
            let ktd = ctx.alloc(kt.len() * 4)?;
            ctx.h2d(ktd, bytemuck::cast_slice(&kt))?;
            ktd
        })
    } else {
        ("dequant_f16_q5k", "gemm_q5k_wm", "gemm_q5k_wc", std::ptr::null_mut())
    };
    if std::env::var_os("LLM170_WC_DBG").is_some() {
        let mut acp2 = ac as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut a2 = vec![(&mut acp2) as *mut _ as *mut c_void, (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void];
        ctx.launch3("dequant_f16_xs_dbg", n_out as u32, 1, 1, 256, &mut a2)?;
        ctx.sync()?;
        let mut probe16 = vec![0u16; 256];
        let _ = ctx.d2h(unsafe { std::slice::from_raw_parts_mut(probe16.as_mut_ptr() as *mut u8, 512) }, ac);
        let nz2 = probe16.iter().filter(|&&v| v != 0).count();
        eprintln!("# wc-dbg 상수쓰기: 첫 256 중 nonzero={nz2} (0,1,2..여야)");
        return Err("디버그 종료".into());
    }
    {
        let mut wp = wd as *mut c_void;
        let mut ktp = warg as *mut c_void;
        let mut acp = ac as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut args = if is_xs {
            vec![(&mut wp) as *mut _ as *mut c_void, (&mut ktp) as *mut _ as *mut c_void, (&mut acp) as *mut _ as *mut c_void,
                 (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void]
        } else {
            vec![(&mut wp) as *mut _ as *mut c_void, (&mut acp) as *mut _ as *mut c_void,
                 (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void]
        };
        ctx.launch3(dq_kern, n_out as u32, 1, 1, 256, &mut args)?;
    }
    // 합성 활성 t행 — quant_q8로 장치 인코딩
    let mut seed = 0x1234_5678u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let xf: Vec<f32> = (0..t * n_in).map(|_| lcg()).collect();
    let xfd = ctx.alloc(xf.len() * 4)?;
    ctx.h2d(xfd, bytemuck::cast_slice(&xf))?;
    let xq_w = n_in / 4 + n_in / 32;
    let xq = ctx.alloc(t * xq_w * 4)?;
    for ti in 0..t {
        let row = unsafe { xfd.add(ti * n_in * 4) };
        let dst = unsafe { xq.add(ti * xq_w * 4) };
        ctx.quant_q8(row, dst, n_in)?;
    }
    let o1 = ctx.alloc(t * n_out * 4)?;
    let o2 = ctx.alloc(t * n_out * 4)?;
    let gx = n_out.div_ceil(64) as u32;
    let launch = |kern: &'static str, wp2: *mut u8, out: *mut u8| -> Result<(), String> {
        let mut xqp = xq as *mut c_void;
        let mut w2 = wp2 as *mut c_void;
        let mut op = out as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args = vec![(&mut xqp) as *mut _ as *mut c_void, (&mut w2) as *mut _ as *mut c_void,
                            (&mut op) as *mut _ as *mut c_void, (&mut ni) as *mut _ as *mut c_void,
                            (&mut no) as *mut _ as *mut c_void, (&mut xw) as *mut _ as *mut c_void,
                            (&mut tt) as *mut _ as *mut c_void];
        ctx.launch3(kern, gx, 1, 1, 256, &mut args)
    };
    // xs 원판은 ktab2를 추가 인자로 받는다 — 런처 분기
    if is_xs {
        let mut xqp = xq as *mut c_void;
        let mut w2 = wd as *mut c_void;
        let mut op = o1 as *mut c_void;
        let mut ktp = warg as *mut c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args = vec![(&mut xqp) as *mut _ as *mut c_void, (&mut w2) as *mut _ as *mut c_void,
                            (&mut op) as *mut _ as *mut c_void, (&mut ktp) as *mut _ as *mut c_void,
                            (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void,
                            (&mut xw) as *mut _ as *mut c_void, (&mut tt) as *mut _ as *mut c_void];
        ctx.launch3(wm_kern, gx, 1, 1, 256, &mut args)?;
    } else {
        launch(wm_kern, wd, o1)?;
    }
    launch(wc_kern, ac, o2)?;
    ctx.sync()?;
    let mut b1 = vec![0f32; t * n_out];
    let mut b2 = vec![0f32; t * n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut b1).as_mut(), o1)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut b2).as_mut(), o2)?;
    let (mut bit_same, mut maxd) = (0usize, 0f32);
    let mut first = String::new();
    for i in 0..t * n_out {
        if b1[i].to_bits() == b2[i].to_bits() { bit_same += 1; }
        let d = (b1[i] - b2[i]).abs();
        if d > maxd { maxd = d; }
        if first.is_empty() && d > 1e-4 {
            first = format!(" 첫 불일치 i={i} wm={:e} wc={:e}", b1[i], b2[i]);
        }
    }
    // 처리량 — 각 20회
    let bench = |kern: &'static str, wp2: *mut u8, out: *mut u8| -> Result<f64, String> {
        let reps = 20;
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            launch(kern, wp2, out)?;
        }
        ctx.sync()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64 * 1e3)
    };
    let ms_wm = if is_xs {
        // xs 원판 벤치 (ktab2 인자 포함 런치 20회)
        let reps = 20;
        ctx.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let mut xqp = xq as *mut c_void;
            let mut w2 = wd as *mut c_void;
            let mut op = o1 as *mut c_void;
            let mut ktp = warg as *mut c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut tt = t as i32;
            let mut args = vec![(&mut xqp) as *mut _ as *mut c_void, (&mut w2) as *mut _ as *mut c_void,
                                (&mut op) as *mut _ as *mut c_void, (&mut ktp) as *mut _ as *mut c_void,
                                (&mut ni) as *mut _ as *mut c_void, (&mut no) as *mut _ as *mut c_void,
                                (&mut xw) as *mut _ as *mut c_void, (&mut tt) as *mut _ as *mut c_void];
            ctx.launch3(wm_kern, gx, 1, 1, 256, &mut args)?;
        }
        ctx.sync()?;
        t0.elapsed().as_secs_f64() / reps as f64 * 1e3
    } else {
        bench(wm_kern, wd, o1)?
    };
    let ms_wc = bench(wc_kern, ac, o2)?;
    let tf = |ms: f64| 2.0 * t as f64 * n_in as f64 * n_out as f64 / (ms * 1e-3) / 1e12;
    Ok(format!(
        "wc-check {tname} [{n_out}x{n_in}] t={t}: 비트동일 {bit_same}/{} max|Δ|={maxd:.2e}{first}\n처리량: wm(인라인 디퀀트) {ms_wm:.3}ms={:.1} TFLOPS · wc(f16 캐시) {ms_wc:.3}ms={:.1} TFLOPS ({:.2}x)\n캐시 {:.1}MB (1회 dequant)",
        t * n_out, tf(ms_wm), tf(ms_wc), ms_wm / ms_wc, n_out * n_in * 2 / 1048576,
    ))
}

/// MMQ 포트 A/B — bt vs mm (각 미러).
#[allow(unused_assignments)] // na0/sg4 는 런치 인자로 넘긴 **주소**가 읽는 값 (raw 포인터 경유)
pub fn launch_probe() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    // 디코드 소형 커널의 실제 런치 비용 (트레이스 페어링 무관, 직접 계측).
    {
        let n = 5120usize;
        let xb = ctx.alloc(n * 4)?;
        let wb = ctx.alloc(n * 4)?;
        let qb = ctx.alloc(n / 4 + n / 32 + n / 16 + 64)?;
        let mut xp0 = xb as *mut std::ffi::c_void;
        let mut wp0 = wb as *mut std::ffi::c_void;
        let mut qp0 = qb as *mut std::ffi::c_void;
        let mut eps0 = 1e-6f32;
        let mut na0 = n as i32;
        let mut a0: Vec<*mut std::ffi::c_void> = vec![
            &mut xp0 as *mut _ as *mut std::ffi::c_void, &mut wp0 as *mut _ as *mut std::ffi::c_void,
            &mut qp0 as *mut _ as *mut std::ffi::c_void, &mut eps0 as *mut _ as *mut std::ffi::c_void,
            &mut na0 as *mut _ as *mut std::ffi::c_void,
        ];
        let mut s0 = String::new();
        for (label, blk) in [("rmsq n=512", 160u32), ("rmsq n=5120", 512u32), ("rmsq n=20480", 640u32)] {
            let nv: i32 = match label { "rmsq n=512" => 512, "rmsq n=5120" => 5120, _ => 20480 };
            na0 = nv;
            for _ in 0..20 { let _ = ctx.launch("rmsq", 1, 1, blk, &mut a0); }
            ctx.sync()?;
            let n2 = 2000usize;
            let t0 = std::time::Instant::now();
            for _ in 0..n2 { let _ = ctx.launch("rmsq", 1, 1, blk, &mut a0); }
            ctx.sync()?;
            s0.push_str(&format!("{label}: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
        }
        eprintln!("{s0}");
        // 기준선: 자명한 커널(axpy_scaled)의 런치 비용 — n 크기별
        let ab = ctx.alloc(5120 * 4)?;
        let bb = ctx.alloc(5120 * 4)?;
        let cb = ctx.alloc(5120 * 4)?;
        let mut ap = ab as *mut std::ffi::c_void;
        let mut bp = bb as *mut std::ffi::c_void;
        let mut cp = cb as *mut std::ffi::c_void;
        let mut nn = 64i32;
        let mut a2: Vec<*mut std::ffi::c_void> = vec![
            &mut ap as *mut _ as *mut std::ffi::c_void, &mut bp as *mut _ as *mut std::ffi::c_void,
            &mut cp as *mut _ as *mut std::ffi::c_void, &mut nn as *mut _ as *mut std::ffi::c_void,
        ];
        let mut s1 = String::new();
        for (label, nval, gx) in [("axpy n=64 1blk", 64i32, 1u32), ("axpy n=5120 80blk", 5120, 80)] {
            nn = nval;
            for _ in 0..20 { let _ = ctx.launch3("axpy_scaled", gx, 1, 1, 64, &mut a2); }
            ctx.sync()?;
            let n2 = 2000usize;
            let t0 = std::time::Instant::now();
            for _ in 0..n2 { let _ = ctx.launch3("axpy_scaled", gx, 1, 1, 64, &mut a2); }
            ctx.sync()?;
            s1.push_str(&format!("{label}: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
        }
        eprintln!("{s1}");
        // gatedq 직접 계측: (o, z, w, xq, eps, d, n_h, n_tot)
        {
            let d = 128usize;
            let n_tot = 32 * d;
            let ob = ctx.alloc(n_tot * 4)?;
            let zb = ctx.alloc(n_tot * 4)?;
            let wb2 = ctx.alloc(n_tot * 4)?;
            let qb2 = ctx.alloc(n_tot / 4 + n_tot / 32 + n_tot / 16 + 64)?;
            let mut op = ob as *mut std::ffi::c_void;
            let mut zp = zb as *mut std::ffi::c_void;
            let mut wp2 = wb2 as *mut std::ffi::c_void;
            let mut qp2 = qb2 as *mut std::ffi::c_void;
            let mut eps2 = 1e-6f32;
            let mut dd = d as i32;
            let mut nh3 = 32i32;
            let mut nt3 = n_tot as i32;
            let mut a3: Vec<*mut std::ffi::c_void> = vec![
                &mut op as *mut _ as *mut std::ffi::c_void, &mut zp as *mut _ as *mut std::ffi::c_void,
                &mut wp2 as *mut _ as *mut std::ffi::c_void, &mut qp2 as *mut _ as *mut std::ffi::c_void,
                &mut eps2 as *mut _ as *mut std::ffi::c_void, &mut dd as *mut _ as *mut std::ffi::c_void,
                &mut nh3 as *mut _ as *mut std::ffi::c_void, &mut nt3 as *mut _ as *mut std::ffi::c_void,
            ];
            let mut res = String::new();
            for (nb, thr) in [(1u32, 32u32), (8, 32), (32, 32), (32, 128)] {
                for _ in 0..20 { let _ = ctx.launch3("gatedq", nb, 1, 1, thr, &mut a3); }
                ctx.sync()?;
                let n2 = 2000usize;
                let t0 = std::time::Instant::now();
                for _ in 0..n2 { let _ = ctx.launch3("gatedq", nb, 1, 1, thr, &mut a3); }
                ctx.sync()?;
                res.push_str(&format!("{nb}blk x{thr}thr: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
            }
            eprintln!("{res}");
        }
        // qsa_flash_gqa 직접 계측 (13 인자)
        {
            let hd = 256usize;
            let nh = 24usize;
            let nkv = 4usize;
            let npast = 128i32;
            let nseg = 4usize;
            let qb = ctx.alloc(nh * 2 * hd * 4)?;
            let kb = ctx.alloc(nkv * 4096 * hd * 4)?;
            let vb = ctx.alloc(nkv * 4096 * hd * 4)?;
            let mb = ctx.alloc(4096 * 4)?;
            let pb = ctx.alloc(nseg * nh * (hd + 2) * 4)?;
            let mut a4: Vec<*mut std::ffi::c_void> = Vec::new();
            let mut qp4 = qb as *mut std::ffi::c_void;
            let mut kp4 = kb as *mut std::ffi::c_void;
            let mut vp4 = vb as *mut std::ffi::c_void;
            let mut mp4 = mb as *mut std::ffi::c_void;
            let mut pp4 = pb as *mut std::ffi::c_void;
            let mut np_ = npast; let mut nh4 = nh as i32; let mut nk4 = nkv as i32;
            let mut h4 = hd as i32; let mut tl4 = 1i32; let mut ss4 = 4096i32; let mut p04 = 0i32; let mut sg4 = 32i32;
            a4.push(&mut qp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut kp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut vp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut mp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut pp4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut np_ as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut nh4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut nk4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut h4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut tl4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut ss4 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut p04 as *mut _ as *mut std::ffi::c_void);
            a4.push(&mut sg4 as *mut _ as *mut std::ffi::c_void);
            let mut r2 = String::new();
            for (lab, nb, thr) in [("sg4  ", 1u32, 256u32), ("sg8  ", 1, 256), ("sg32 ", 1, 256), ("sg32x4", 4, 256)] {
                sg4 = match lab { "sg4  " => 4, "sg8  " => 8, _ => 32 };
                for _ in 0..20 { let _ = ctx.launch3("qsa_flash_gqa", 1, 4, nb, thr, &mut a4); }
                ctx.sync()?;
                let n2 = 2000usize;
                let t0 = std::time::Instant::now();
                for _ in 0..n2 { let _ = ctx.launch3("qsa_flash_gqa", 1, 4, nb, thr, &mut a4); }
                ctx.sync()?;
                r2.push_str(&format!("{lab}: {:.2}us  ", t0.elapsed().as_secs_f64() * 1e6 / n2 as f64));
                let _ = nseg;
            }
            eprintln!("{r2}");
        }
    }
    let mut xp = ctx.alloc(256)?; let mut op = ctx.alloc(256)?; let mut sp = ctx.alloc(256)?;
    let mut nn = 64i32;
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        (&mut op) as *mut _ as *mut std::ffi::c_void,
        (&mut xp) as *mut _ as *mut std::ffi::c_void,
        (&mut sp) as *mut _ as *mut std::ffi::c_void,
        (&mut nn) as *mut _ as *mut std::ffi::c_void,
    ];
    for _ in 0..10 { ctx.launch3("axpy_scaled", 1, 1, 1, 64, &mut args)?; }
    ctx.sync()?;
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n { ctx.launch3("gemm_xs", 1, 1, 1, 64, &mut args)?; }
    let cpu = t0.elapsed();
    ctx.sync()?;
    let wall = t0.elapsed();
    Ok(format!("launch-probe: {n}회 런치 cpu={:.3}ms/회 (동기 포함 wall={:.3}ms/회)", cpu.as_secs_f64()*1e3/n as f64, wall.as_secs_f64()*1e3/n as f64))
}

pub fn mm_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let t = std::env::var("LLM170_MM_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let mut q8s = Vec::new();
    let mut xq_h: Vec<u32> = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        let blocks = llm170_core::quant::quantize_row_q8_ref(&x);
        for blk in &blocks {
            for c in 0..8 {
                let b = c * 4;
                xq_h.push((blk.qs[b] as u32 & 0xFF) | ((blk.qs[b+1] as u32 & 0xFF) << 8) | ((blk.qs[b+2] as u32 & 0xFF) << 16) | ((blk.qs[b+3] as u32 & 0xFF) << 24));
            }
        }
        for blk in &blocks { xq_h.push(blk.d.to_bits()); }
        for blk in &blocks {
            let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
            let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
            xq_h.push(s0 as u32);
            xq_h.push(s1 as u32);
        }
        q8s.push(blocks);
    }
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let xq = ctx.alloc(xq_h.len() * 4)?;
    ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    let out = ctx.alloc(n_out * 4 * t)?;
    // wm 상한 64 + bench GEMV 그리드 부적합: 128-커널 계열 미로드면 에러
    let (v4, j128f, odd) = (ctx.co_loaded(CO_V4), ctx.co_loaded(CO_J128), ctx.co_loaded(CO_ODD));
    let big_ok = match w.ty {
        llm170_gguf::GgmlType::Q5K | llm170_gguf::GgmlType::Q4K | llm170_gguf::GgmlType::Iq4Xs
            => v4 || j128f,
        llm170_gguf::GgmlType::Q6K | llm170_gguf::GgmlType::Q8_0 => j128f,
        llm170_gguf::GgmlType::Iq4Nl | llm170_gguf::GgmlType::Q3K | llm170_gguf::GgmlType::Iq3S => odd,
        _ => true,
    };
    if t > 64 && !big_ok {
        return Err(format!("mm-bench 미지원: t={t}는 타입별 128-커널 필요"));
    }
    let kern_name = match w.ty {
        llm170_gguf::GgmlType::Q5K => if v4 { "gemm_q5k_v4" }
            else if j128f { "gemm_q5k_j128" }
            else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q5k_wm" } else { "gemm_q5k_mm" },
        llm170_gguf::GgmlType::Q4K => if v4 { "gemm_q4k_v4" }
            else if j128f { "gemm_q4k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q4k_wm" } else { "gemm_q4k_mm" },
        llm170_gguf::GgmlType::Q6K => if j128f { "gemm_q6k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q6k_wm" } else { "gemm_q6k_mm" },
        llm170_gguf::GgmlType::Q8_0 => if j128f { "gemm_q8_j128" } else { return Err("mm-bench 미지원: q8_0은 j128 커널 필요".into()) },
        llm170_gguf::GgmlType::Iq4Xs => if v4 { "gemm_xs_v4" }
            else if j128f { "gemm_xs_j128" }
            else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_xs_wm" } else { "gemm_xs_mm" },
        llm170_gguf::GgmlType::Iq4Nl => if odd { "gemm_nl_v4" }
            else { return Err("mm-bench 미지원: iq4_nl 타일은 odd CO 필요".into()) },
        llm170_gguf::GgmlType::Q3K => if odd { "gemm_q3k_v4" }
            else { return Err("mm-bench 미지원: q3_K 타일은 odd CO 필요".into()) },
        llm170_gguf::GgmlType::Iq3S => if odd { "gemm_iq3s_v4" }
            else { return Err("mm-bench 미지원: iq3_s 타일은 odd CO 필요".into()) },
        _ => "gemm_xs_mm",
    };

    let launch = |ctx: &RawCtx| -> Result<(), String> {
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ktp = kt_d as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
        ];
        if kern_name == "gemm_xs_mm" || kern_name == "gemm_xs_wm" || kern_name == "gemm_xs_j128" || kern_name == "gemm_xs_v4" || kern_name == "gemm_nl_v4" {
            args.push((&mut ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut tt) as *mut _ as *mut std::ffi::c_void);
        let rpb = if kern_name.ends_with("_j128") || kern_name.ends_with("_v4") { 128 } else { 64 };
        let gx = n_out.div_ceil(rpb).min(65535) as u32;
        let _gz = n_out.div_ceil(rpb).div_ceil(65535) as u32;
        let gz = n_out.div_ceil(64).div_ceil(65535) as u32;
        ctx.launch3(kern_name, gx, 1, gz, 256, &mut args)
    };
    launch(&ctx)?;
    ctx.sync()?;
    let mut o2 = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o2).as_mut(), out)?;
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { launch(&ctx)?; }
    ctx.sync()?;
    let dt2 = t0.elapsed().as_secs_f64() / reps as f64;
    // 순수 런치 CPU 비용: 그리드 1x1 소형 발사 (GPU 즉시 완료) 100회
    let (mut sxa, mut swa, mut soa) = (xq, wd, out);
    let (mut sni, mut sno, mut sxw, mut stt) = (n_in as i32, n_out as i32, xq_w as i32, t as i32);
    let mut sargs: Vec<*mut std::ffi::c_void> = vec![
        (&mut sxa) as *mut _ as *mut std::ffi::c_void,
        (&mut swa) as *mut _ as *mut std::ffi::c_void,
        (&mut soa) as *mut _ as *mut std::ffi::c_void,
        (&mut sni) as *mut _ as *mut std::ffi::c_void,
        (&mut sno) as *mut _ as *mut std::ffi::c_void,
        (&mut sxw) as *mut _ as *mut std::ffi::c_void,
        (&mut stt) as *mut _ as *mut std::ffi::c_void,
    ];
    let tl0 = std::time::Instant::now();
    for _ in 0..100 { let _ = ctx.launch3(kern_name, 1, 1, 1, 64, &mut sargs); }
    let lcpu = tl0.elapsed().as_secs_f64() * 1e3 / 100.0;
    ctx.sync()?;
    eprintln!("launch-cpu: {:.3}ms/회 (1x1x64 소형)", lcpu);
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut m2 = 0usize;
    let mut maxrel = 0f32;
    for ti in 0..t {
        for oo in 0..n_out.min(256) {
            let row = &w.data[oo * rb..];
            let c2 = match w.ty {
                llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q8_0 => {
                    let nblk = n_in as usize / 32;
                    let mut acc = 0.0f32;
                    for b in 0..nblk {
                        let wb = &row[b * 34..b * 34 + 34];
                        let h = ((wb[1] as u16) << 8) | wb[0] as u16;
                        let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
                        let exp = ((h >> 10) & 0x1F) as i32;
                        let man = (h & 0x3FF) as f32;
                        let d = if exp == 0 { sign * man * 2f32.powi(-24) } else { sign * (man / 1024.0 + 1.0) * 2f32.powi(exp - 15) };
                        let mut isum = 0i64;
                        for j in 0..32 {
                            let wv = wb[2 + j] as i8 as i64;
                            let yv = q8s[ti][b].qs[j] as i64;
                            isum += wv * yv;
                        }
                        let yd = q8s[ti][b].d;
                        acc += yd * d * isum as f32;
                    }
                    acc
                }
                _ => llm170_core::quant::dot_row_w4a8_iq4xs_mm(row, n_in as u64, &q8s[ti]),
            };
            if kern_name.ends_with("_wm") || kern_name.ends_with("_w32") || kern_name.ends_with("_j128") || kern_name.ends_with("_v4") {
                let g = o2[ti * n_out + oo];
                let denom = c2.abs().max(1.0);
                let rel = (g - c2).abs() / denom;
                if rel > maxrel { maxrel = rel; }
                if rel > 5e-3 { m2 += 1; }
            } else if c2.to_bits() != o2[ti * n_out + oo].to_bits() { m2 += 1; }
        }
    }
    Ok(format!("mm({kern_name}): {:.3}ms ({:.1}us/tok) mism {m2} maxrel {maxrel:.2e}", dt2 * 1e3, dt2 * 1e6 / t as f64))
}
