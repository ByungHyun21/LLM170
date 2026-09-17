//! probes/wmma — WMMA 프래그먼트·루프 (probes.rs에서 이동, plans/78 R3).

use super::*;

/// dot4 루프-오버헤드 루프 프로브 — 모드별 유효 TIOPS.
/// rocwmma 16x16x16 프래그먼트 레이아웃 검증 — C 레이아웃(idx=lane+32*sl, row=idx>>4,
/// col=idx&15)과 A/B 레이아웃 가정을 정수 데이터로 정확히 확인한다(plans/47).
/// WMMA 가용성 게이트 — 기동 1회 측정 후 캐시(플래그 대신 실측).
/// 불가/오차 초과면 어텐션은 스칼라 판(wk8)으로 간다.
pub fn wmma_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| matches!(wmma_probe_both(), Ok((ok, _)) if ok))
}

/// 기기 실측 리포트 — 이름·가용/전체 메모리·호스트↔디바이스 대역폭.
/// 라우트 선택의 근거(기동 1회). UMA면 h2d/d2h가 메모리 대역폭급으로 높고,
/// PCIe 디스크리트면 수 GB/s 수준 — 같은 코드가 이 값으로 상주 정책을 정한다.
/// f32 → f16 비트(호스트측, 프로브 전용 근사).
pub(super) fn half_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (x >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | frac as u16
}

pub fn wmma_check() -> Result<String, String> {
    wmma_probe_both().map(|(_, m)| m)
}

/// plans/74 N4: raw WMMA(w32) 프래그먼트 ABI 확정 — 4가지 레이아웃 조합을
/// CPU 행렬곱과 대조해 매핑을 고른다.
pub fn wmma2_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let a: Vec<f32> = (0..256).map(|i| (((i / 16) * 3 + (i % 16) * 7) % 11) as f32 - 5.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (((i / 16) * 5 + (i % 16) * 2) % 13) as f32 - 6.0).collect();
    let ah: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    let bh: Vec<u16> = b.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    let ad = ctx.alloc(512)?;
    let bd = ctx.alloc(512)?;
    let cd = ctx.alloc(1024)?;
    ctx.h2d(ad, bytemuck::cast_slice(&ah))?;
    ctx.h2d(bd, bytemuck::cast_slice(&bh))?;
    let mut msg = String::new();
    let mut any_ok = false;
    for mode in 0..4i32 {
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut m = mode;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
            (&mut m) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_probe", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        let mut maxerr = 0f32;
        for i in 0..16usize {
            for j in 0..16usize {
                let mut sum = 0f32;
                for k in 0..16usize {
                    sum += a[i * 16 + k] * b[k * 16 + j];
                }
                maxerr = maxerr.max((c[i * 16 + j] - sum).abs());
            }
        }
        let ok = maxerr <= 2e-2;
        any_ok |= ok;
        msg += &format!("mode{mode} (D={}, AB={}): max|Δ|={maxerr:.4} {}\n",
            if mode & 1 == 0 { "2l+g" } else { "l+8g" },
            if mode & 2 == 0 { "contig" } else { "stride" },
            if ok { "★ 일치" } else { "" });
    }
    Ok(format!("wmma2(raw builtin w32) ABI 프로브:\n{msg}{}", if any_ok { "" } else { "전 불일치 — 매핑 재역추론 필요" }))
}

/// plans/74 N4: (lane,l)→(i,k) 매핑 역추론 — 단일원소 행렬 512조합 덤프.
pub fn wmma2_map() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let ad = ctx.alloc(512)?;
    let bd = ctx.alloc(512)?;
    let cd = ctx.alloc(32 * 8 * 4)?;
    let mut amap = vec![(-1i32, -1i32); 32 * 8]; // (row, col=contraction)
    let mut bmap = vec![(-1i32, -1i32); 32 * 8]; // (row=contraction, col)
    let mut ident = vec![0u16; 256];
    for i in 0..16 {
        ident[i * 16 + i] = half::f16::from_f32(1.0).to_bits();
    }
    let mut e = vec![0u16; 256];
    let _ = &e;
    let dump = |ad: usize, bd: usize, ah: &[u16], bh: &[u16]| -> Result<Vec<f32>, String> {
        ctx.h2d(ad as *mut u8, unsafe { std::slice::from_raw_parts(ah.as_ptr() as *const u8, 512) })?;
        ctx.h2d(bd as *mut u8, unsafe { std::slice::from_raw_parts(bh.as_ptr() as *const u8, 512) })?;
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_dump", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        Ok(c)
    };
    // A 매핑: B=I → D=A. A=e_{r,k} 의 1이 어느 (lane,l) 에 나오나.
    for r in 0..16usize {
        for k in 0..16usize {
            for v in e.iter_mut() { *v = 0; }
            e[r * 16 + k] = half::f16::from_f32(1.0).to_bits();
            let c = dump(ad as usize, bd as usize, &e, &ident)?;
            for idx in 0..256usize {
                if c[idx] == 1.0
                    && (amap[idx].0 == -1 || amap[idx] == (r as i32, k as i32)) {
                        amap[idx] = (r as i32, k as i32);
                    }
            }
        }
    }
    // B 매핑: A=I → D=B. B=e_{k,j}.
    for kk in 0..16usize {
        for j in 0..16usize {
            for v in e.iter_mut() { *v = 0; }
            e[kk * 16 + j] = half::f16::from_f32(1.0).to_bits();
            let c = dump(ad as usize, bd as usize, &ident, &e)?;
            for idx in 0..256usize {
                if c[idx] == 1.0
                    && (bmap[idx].0 == -1 || bmap[idx] == (kk as i32, j as i32)) {
                        bmap[idx] = (kk as i32, j as i32);
                    }
            }
        }
    }
    let fmt = |m: &[ (i32, i32) ]| -> String {
        let mut s = String::new();
        for lane in 0..32 {
            s += &format!("lane{lane:2}: ");
            for l in 0..8 {
                let (i, j) = m[lane * 8 + l];
                s += &format!("({i:2},{j:2})");
            }
            s += "\n";
        }
        s
    };
    Ok(format!("A(lane,l)→(row,col):\n{}\nB(lane,l)→(row,col):\n{}", fmt(&amap), fmt(&bmap)))
}

/// plans/74 N4: 랜덤 다중시행 교집합으로 D 레지스터 (lane,l)→(i,j) 확정.
pub fn wmma2_map2() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let ad = ctx.alloc(512)?;
    let bd = ctx.alloc(512)?;
    let cd = ctx.alloc(32 * 8 * 4)?;
    let mut ident = vec![0u16; 256];
    for i in 0..16 {
        ident[i * 16 + i] = half::f16::from_f32(1.0).to_bits();
    }
    let mut cand: Vec<Vec<(u8, u8)>> = vec![vec![]; 256];
    let mut first = true;
    for trial in 0..12u64 {
        let seed = 0x9E3779B97F4A7C15u64.wrapping_mul(trial + 1);
        let a: Vec<f32> = (0..256)
            .map(|i| {
                let h = seed.wrapping_mul(i as u64 + 1);
                ((h >> 33) % 13) as f32 - 6.0
            })
            .collect();
        let ah: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        ctx.h2d(ad, unsafe { std::slice::from_raw_parts(ah.as_ptr() as *const u8, 512) })?;
        ctx.h2d(bd, unsafe { std::slice::from_raw_parts(ident.as_ptr() as *const u8, 512) })?;
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_dump", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        for idx in 0..256usize {
            let mut hits = vec![];
            for i in 0..16usize {
                for j in 0..16usize {
                    if (a[i * 16 + j] - c[idx]).abs() < 1e-3 {
                        hits.push((i as u8, j as u8));
                    }
                }
            }
            if first {
                cand[idx] = hits;
            } else {
                cand[idx].retain(|&h| hits.contains(&h));
            }
        }
        first = false;
    }
    let mut msg = String::new();
    for lane in 0..32 {
        msg += &format!("lane{lane:2}:");
        for l in 0..8 {
            let cs = &cand[lane * 8 + l];
            let s = if cs.len() == 1 {
                format!(" ({},{})", cs[0].0, cs[0].1)
            } else if cs.is_empty() {
                " (?,?)".into()
            } else {
                format!(" {}안", cs.len())
            };
            msg += &s;
        }
        msg += "\n";
    }
    // 진단: lanes 16-31 원시값과 홀수행 기대값 비교(1시행)
    {
        let a: Vec<f32> = (0..256).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect();
        let ah: Vec<u16> = a.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        ctx.h2d(ad, unsafe { std::slice::from_raw_parts(ah.as_ptr() as *const u8, 512) })?;
        ctx.h2d(bd, unsafe { std::slice::from_raw_parts(ident.as_ptr() as *const u8, 512) })?;
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma2_dump", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        msg += "raw lanes16-31: ";
        for lane in 16..32 { for l in 0..8 { msg += &format!("{:.0},", c[lane*8+l]); } }
        msg += "\nA odd rows:    ";
        for i in (1..16).step_by(2) { for j in 0..16 { msg += &format!("{:.0},", a[i*16+j]); } }
        msg += "\n";
    }
    Ok(format!("D(lane,l)→(i,j) 확률적 확정(B=I, D=A):\n{msg}"))
}

fn wmma_probe_both() -> Result<(bool, String), String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let a: Vec<f32> = (0..256).map(|i| (((i / 16) * 3 + (i % 16)) % 9) as f32 - 4.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (((i / 16) * 5 + (i % 16)) % 7) as f32 - 3.0).collect();
    let ad = ctx.alloc(256 * 4)?;
    let bd = ctx.alloc(256 * 4)?;
    let cd = ctx.alloc(256 * 4)?;
    ctx.h2d(ad, bytemuck::cast_slice(&a))?;
    ctx.h2d(bd, bytemuck::cast_slice(&b))?;
    let mut msg = String::new();
    let mut ok = true;
    for mode in [0i32, 1] {
        let mut ap = ad as *mut c_void;
        let mut bp = bd as *mut c_void;
        let mut cp = cd as *mut c_void;
        let mut m = mode;
        let mut args = vec![
            (&mut ap) as *mut _ as *mut c_void,
            (&mut bp) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
            (&mut m) as *mut _ as *mut c_void,
        ];
        ctx.launch3("wmma_probe", 1, 1, 1, 32, &mut args)?;
        ctx.sync()?;
        let mut c = vec![0f32; 256];
        ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
        let mut maxerr = 0f32;
        let mut first = String::new();
        for i in 0..16usize {
            for jj in 0..16usize {
                let mut sum = 0f32;
                for k2 in 0..16usize {
                    let av = a[i * 16 + k2];
                    let bv = if mode == 0 { b[jj * 16 + k2] } else { b[k2 * 16 + jj] };
                    sum += av * bv;
                }
                let idx = i * 16 + jj;
                let d = (c[idx] - sum).abs();
                if d > 1e-3 && first.is_empty() {
                    first = format!(" 첫 불일치 (i={i},j={jj},idx={idx}) 기대 {sum} 실제 {}", c[idx]);
                }
                maxerr = maxerr.max(d);
            }
        }
        msg += &format!("mode{mode}: max|delta| = {maxerr:.6}{first}\n");
        ok &= maxerr <= 1e-3;
    }
    Ok((ok, msg))
}

/// PV 경로 프로브: A=P(16x16 ldm=16) x B=V(16x256 **row_major** ldm=256) — 어텐션 PV 와 동일.
pub fn wmma_check_pv() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let pv: Vec<f32> = (0..16 * 16).map(|i| (((i * 7) % 5) as f32) * 0.25).collect();
    let vv: Vec<f32> = (0..16 * 256).map(|i| (((i / 256) * 5 + (i % 256)) % 7) as f32 - 3.0).collect();
    let pd = ctx.alloc(16 * 16 * 4)?;
    let vd = ctx.alloc(16 * 256 * 4)?;
    let cd = ctx.alloc(256 * 4)?;
    ctx.h2d(pd, bytemuck::cast_slice(&pv))?;
    ctx.h2d(vd, bytemuck::cast_slice(&vv))?;
    let mut pp = pd as *mut c_void;
    let mut vp = vd as *mut c_void;
    let mut cp = cd as *mut c_void;
    let mut args = vec![
        (&mut pp) as *mut _ as *mut c_void,
        (&mut vp) as *mut _ as *mut c_void,
        (&mut cp) as *mut _ as *mut c_void,
    ];
    ctx.launch3("wmma_probe_pv", 1, 1, 1, 32, &mut args)?;
    ctx.sync()?;
    let mut c = vec![0f32; 256];
    ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd)?;
    let mut maxerr = 0f32;
    let mut nnan = 0usize;
    for row in 0..16usize {
        for dim in 0..16usize {
            let mut sum = 0f32;
            for key in 0..16usize { sum += pv[row * 16 + key] * vv[key * 256 + dim]; }
            let got = c[row * 16 + dim];
            if got.is_nan() { nnan += 1; }
            let d = (got - sum).abs();
            if d > maxerr { maxerr = d; }
        }
    }
    Ok(format!("PV 경로 (B row_major ldm=256): max|delta| = {maxerr:.4}, NaN {nnan}/256"))
}

/// mode2 프로브: 16x256 타일을 ldm=256 으로 적재했을 때 프래그먼트 레이아웃이 맞는지.
pub fn wmma_check_ldm() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let a2: Vec<f32> = (0..16 * 256).map(|i| (((i / 256) * 3 + (i % 256)) % 9) as f32 - 4.0).collect();
    let b2: Vec<f32> = (0..16 * 256).map(|i| (((i / 256) * 5 + (i % 256)) % 7) as f32 - 3.0).collect();
    let ad2 = ctx.alloc(16 * 256 * 4)?;
    let bd2 = ctx.alloc(16 * 256 * 4)?;
    let cd2 = ctx.alloc(256 * 4)?;
    ctx.h2d(ad2, bytemuck::cast_slice(&a2))?;
    ctx.h2d(bd2, bytemuck::cast_slice(&b2))?;
    let mut ap = ad2 as *mut c_void;
    let mut bp = bd2 as *mut c_void;
    let mut cp = cd2 as *mut c_void;
    let mut args = vec![
        (&mut ap) as *mut _ as *mut c_void,
        (&mut bp) as *mut _ as *mut c_void,
        (&mut cp) as *mut _ as *mut c_void,
    ];
    ctx.launch3("wmma_probe_ldm", 1, 1, 1, 256, &mut args)?;
    ctx.sync()?;
    let mut c = vec![0f32; 256];
    ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), cd2)?;
    let mut maxerr = 0f32;
    let mut nnan = 0usize;
    for i in 0..16usize {
        for jj in 0..16usize {
            let mut sum = 0f32;
            for k2 in 0..256usize { sum += a2[i * 256 + k2] * b2[jj * 256 + k2]; }
            let d = (c[i * 16 + jj] - sum).abs();
            if c[i * 16 + jj].is_nan() { nnan += 1; }
            if d > maxerr { maxerr = d; }
        }
    }
    Ok(format!("mode2 (ldm=256): max|delta| = {maxerr:.4}, NaN {nnan}/256"))
}

pub fn roof_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n_in = 5120usize;
    let xq = ctx.alloc(n_in * 4)?;
    let w = ctx.alloc(n_in * 4)?;
    let out = ctx.alloc(16)?;
    let data: Vec<u32> = (0..n_in).map(|i| (i as u32).wrapping_mul(2654435761)).collect();
    ctx.h2d(xq, bytemuck::cast_slice(&data))?;
    ctx.h2d(w, bytemuck::cast_slice(&data))?;
    let mut msg = String::new();
    for &mode in &[0usize, 1, 2] {
        let iters = 20000usize;
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut m = mode as i32;
        let mut it = iters as i32;
        let mut ni = n_in as i32;
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
            (&mut m) as *mut _ as *mut std::ffi::c_void,
            (&mut it) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
        ];
        // grid: 40CU 채우도록 640블록×64스레드
        ctx.launch3("dot_roof", 640, 1, 1, 64, &mut args)?;
        ctx.sync()?;
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.launch3("dot_roof", 640, 1, 1, 64, &mut args)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let total_dots = iters as f64 * 640.0 * 64.0;
        let tips = total_dots * 4.0 / dt / 1e12; // MAC 4개/dot
        msg += &format!("mode{mode} ({}): {:.2}ms → {:.2} TIOPS MAC\n",
            ["reg-chain", "same-addr load", "stride load"][mode], dt * 1e3, tips);
    }
    // mfma 발행률 — rocwmma 16x16x16f32 (8192 FLOP/wave/mma)
    {
        let ntiles = 64usize;
        let ah = vec![0x3c00u16; ntiles * 256];
        let ad = ctx.alloc(ah.len() * 2)?;
        let bd = ctx.alloc(ah.len() * 2)?;
        ctx.h2d(ad, bytemuck::cast_slice(&ah))?;
        ctx.h2d(bd, bytemuck::cast_slice(&ah))?;
        let om = ctx.alloc(24)?;
        for &mode in &[0usize, 1] {
            let iters = 20000usize;
            let mut ap = ad as *mut std::ffi::c_void;
            let mut bp = bd as *mut std::ffi::c_void;
            let mut op = om as *mut std::ffi::c_void;
            let mut m = mode as i32;
            let mut it = iters as i32;
            let mut nn = ntiles as i32;
            let mut args = vec![
                (&mut ap) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut m) as *mut _ as *mut std::ffi::c_void,
                (&mut it) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("mfma_roof", 640, 1, 1, 64, &mut args)?;
            ctx.sync()?;
            let reps = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                ctx.launch3("mfma_roof", 640, 1, 1, 64, &mut args)?;
            }
            ctx.sync()?;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            let mut o3 = [0f64; 3];
            ctx.d2h(bytemuck::cast_slice_mut(&mut o3).as_mut(), om)?;
            let wavesize = o3[2];
            let waves = 640.0 * 64.0 / wavesize;
            let tflops = waves * iters as f64 * 8192.0 / dt / 1e12;
            msg += &format!("mfma{mode} ({} wave{}): {:.2}ms → {:.2} TFLOPS f32\n",
                ["reg-resident", "L1-fed"][mode], wavesize, dt * 1e3, tflops);
        }
    }
    Ok(msg)
}
