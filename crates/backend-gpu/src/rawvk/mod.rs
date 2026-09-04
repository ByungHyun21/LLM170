//! rawvk — Vulkan 백엔드 (plans/12). 모듈 루트.

pub mod context;
pub mod gemv;

use context::VkCtx;

const SMOKE_SPV: &[u8] = include_bytes!("spv/smoke.spv");
const COOPMAT_PROBE_SPV: &[u8] = include_bytes!("spv/coopmat_probe.spv");
const AXPY_SPV: &[u8] = include_bytes!("spv/axpy_scaled.spv");

/// vk-check — 디바이스 역량 + 트리비얼 컴퓨트 값 검증 (M1 게이트).
/// subsum-check — 서브그룹 리덕션 프로브 (xor 트리 / add / broadcast).
pub fn subsum_check() -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
    let mut ctx = VkCtx::new()?;
    let ob = ctx.alloc(16)?;
    let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(include_bytes!("spv/subsum.spv"), 1, 4)?;
    ctx.bind_bufs(ds, &[ob.buf]);
    for mode in 0..3 {
        let push = (mode as u32).to_le_bytes().to_vec();
        // gy=128 (gdn_ar과 동일 그리드 형상) — WG 수 의존성 검출
        ctx.run(pl, ds, pipe, &push, 1, 128, 1)?;
    }
    let mut r = vec![0f32; 3];
    unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const f32, r.as_mut_ptr(), 3) };
    Ok(format!(
        "subgroup: xor_tree={} (기대 496) add={} broadcast={}",
        r[0], r[1], r[2]
    ))
}

/// gdn-check — GDN 커널군 (plans/19) GPU↔CPU 상호검증.
/// split3·conv_t·beta_g·norm_gated·ar — 각 커널 독립 LCG 입력·CPU 미러 대조.
pub fn gdn_check() -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
    let mut ctx = VkCtx::new()?;
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let mut lines: Vec<String> = Vec::new();

    // ── split3: [t=3][n0+n1+n2=12]
    {
        let (n0, n1, n2, t) = (4usize, 5usize, 3usize, 3usize);
        let total = n0 + n1 + n2;
        let src: Vec<f32> = (0..t * total).map(|_| lcg()).collect();
        let mut sbuf = ctx.alloc(t * total * 4)?;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), sbuf.ptr as *mut f32, src.len()) };
        let d0 = ctx.alloc(t * n0 * 4)?;
        let d1 = ctx.alloc(t * n1 * 4)?;
        let d2 = ctx.alloc(t * n2 * 4)?;
        let (dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/split3.spv"), 4, 12,
        )?;
        let _ = dsl;
        ctx.bind_bufs(ds, &[sbuf.buf, d0.buf, d1.buf, d2.buf]);
        let push: Vec<u8> = [n0 as u32, n1 as u32, n2 as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.run(pl, ds, pipe, &push, (t * total).div_ceil(64) as u32, 1, 1)?;
        let (mut r0, mut r1, mut r2) = (vec![0f32; t * n0], vec![0f32; t * n1], vec![0f32; t * n2]);
        unsafe {
            std::ptr::copy_nonoverlapping(d0.ptr as *const f32, r0.as_mut_ptr(), r0.len());
            std::ptr::copy_nonoverlapping(d1.ptr as *const f32, r1.as_mut_ptr(), r1.len());
            std::ptr::copy_nonoverlapping(d2.ptr as *const f32, r2.as_mut_ptr(), r2.len());
        }
        // CPU 미러
        let (mut c0, mut c1, mut c2) = (vec![0f32; t * n0], vec![0f32; t * n1], vec![0f32; t * n2]);
        for ti in 0..t {
            for j in 0..total {
                let v = src[ti * total + j];
                if j < n0 { c0[ti * n0 + j] = v; }
                else if j < n0 + n1 { c1[ti * n1 + j - n0] = v; }
                else { c2[ti * n2 + j - n0 - n1] = v; }
            }
        }
        let ok = r0 == c0 && r1 == c1 && r2 == c2;
        lines.push(format!("split3: {}", if ok { "★" } else { "MISMATCH" }));
    }

    // ── gdn_conv_t: ch=128, k=4, t=8 (t≥k-1)
    {
        let (ch, k, t) = (128usize, 4usize, 8usize);
        let qkv: Vec<f32> = (0..t * ch).map(|_| lcg()).collect();
        let cw: Vec<f32> = (0..ch * k).map(|_| lcg()).collect();
        let st0: Vec<f32> = (0..(k - 1) * ch).map(|_| lcg()).collect();
        let qbuf = ctx.alloc(qkv.len() * 4)?;
        let cbuf = ctx.alloc(cw.len() * 4)?;
        let sbuf = ctx.alloc(st0.len() * 4)?;
        let obuf = ctx.alloc(t * ch * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(qkv.as_ptr(), qbuf.ptr as *mut f32, qkv.len());
            std::ptr::copy_nonoverlapping(cw.as_ptr(), cbuf.ptr as *mut f32, cw.len());
            std::ptr::copy_nonoverlapping(st0.as_ptr(), sbuf.ptr as *mut f32, st0.len());
        }
        let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/gdn_conv_t.spv"), 4, 12,
        )?;
        ctx.bind_bufs(ds, &[qbuf.buf, cbuf.buf, sbuf.buf, obuf.buf]);
        let push: Vec<u8> = [ch as u32, k as u32, t as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.run(pl, ds, pipe, &push, ch.div_ceil(64) as u32, t as u32, 1)?;
        let mut r = vec![0f32; t * ch];
        unsafe { std::ptr::copy_nonoverlapping(obuf.ptr as *const f32, r.as_mut_ptr(), r.len()) };
        // CPU 미라
        let mut c = vec![0f32; t * ch];
        for ti in 0..t {
            for cc in 0..ch {
                let mut sum = cw[cc * k + (k - 1)] * qkv[ti * ch + cc];
                for j in 0..k - 1 {
                    let pos = ti as isize - (k as isize - 1) + j as isize;
                    let xv = if pos >= 0 { qkv[pos as usize * ch + cc] } else { st0[((pos + k as isize - 1) as usize) * ch + cc] };
                    sum += cw[cc * k + j] * xv;
                }
                c[ti * ch + cc] = sum / (1.0 + (-sum).exp());
            }
        }
        let maxd = r.iter().zip(c.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        lines.push(format!("gdn_conv_t: max|D|={maxd:.2e} {}", if maxd < 1e-5 { "★" } else { "MISMATCH" }));
    }

    // ── gdn_beta_g
    {
        let (dr, t) = (48usize, 4usize);
        let nh = dr * t;
        let b: Vec<f32> = (0..nh).map(|_| lcg()).collect();
        let a: Vec<f32> = (0..nh).map(|_| lcg()).collect();
        let dtb: Vec<f32> = (0..dr).map(|_| lcg()).collect();
        let sa: Vec<f32> = (0..dr).map(|_| lcg()).collect();
        let bb = ctx.alloc(nh * 4)?;
        let ab = ctx.alloc(nh * 4)?;
        let db = ctx.alloc(dr * 4)?;
        let sb = ctx.alloc(dr * 4)?;
        let gb = ctx.alloc(nh * 2 * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), bb.ptr as *mut f32, nh);
            std::ptr::copy_nonoverlapping(a.as_ptr(), ab.ptr as *mut f32, nh);
            std::ptr::copy_nonoverlapping(dtb.as_ptr(), db.ptr as *mut f32, dr);
            std::ptr::copy_nonoverlapping(sa.as_ptr(), sb.ptr as *mut f32, dr);
        }
        let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/gdn_beta_g.spv"), 5, 8,
        )?;
        ctx.bind_bufs(ds, &[bb.buf, ab.buf, db.buf, sb.buf, gb.buf]);
        let push: Vec<u8> = [nh as u32, dr as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.run(pl, ds, pipe, &push, nh.div_ceil(64) as u32, 1, 1)?;
        let mut r = vec![0f32; nh * 2];
        unsafe { std::ptr::copy_nonoverlapping(gb.ptr as *const f32, r.as_mut_ptr(), r.len()) };
        let mut c = vec![0f32; nh * 2];
        for h in 0..nh {
            let h0 = h % dr;
            c[h * 2] = 1.0 / (1.0 + (-b[h]).exp());
            let x = (a[h] + dtb[h0]).min(80.0);
            let sp = (1.0 + x.exp()).ln();
            c[h * 2 + 1] = (sp * sa[h0]).exp();
        }
        let maxd = r.iter().zip(c.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let mrel = r.iter().zip(c.iter()).map(|(x, y)| ((x - y) / y.abs().max(1e-6) as f32).abs()).fold(0.0f32, f32::max);
        lines.push(format!("gdn_beta_g: max|D|={maxd:.2e} mrel={mrel:.2e} {}", if mrel < 1e-4 { "★" } else { "MISMATCH" }));
    }

    // ── gdn_ar (핵심 재귀) — d=128 (커널 32레인×4 kdim 전제), hv=8, hk=2, t=4
    {
        let (d, hv, hk, t) = (128usize, 8usize, 2usize, 4usize); // 실제 GDN 축 (dt_rank 48의 축소판)
        let ks = hk * d;
        let vs = hv * d;
        let mut s0: Vec<f32> = (0..hv * d * d).map(|_| lcg()).collect();
        let mut q: Vec<f32> = (0..t * ks).map(|_| lcg()).collect();
        let mut k: Vec<f32> = (0..t * ks).map(|_| lcg()).collect();
        let v: Vec<f32> = (0..t * vs).map(|_| lcg()).collect();
        let mut bg: Vec<f32> = (0..t * hv * 2).map(|_| lcg()).collect();
        // 결정적 프로브: s=1, q=e_0, k=0, beta=0, g=1 → out[u] = scale (전 원소 동일)
        let det = std::env::var_os("LLM170_GDN_DET").is_some();
        if det {
            // s[i][u] = i+1 (i-색인 감지), q = 전부 1, k=0, beta=0, g=1
            // → out[u] = scale·Σ_{i=0..127}(i+1) = scale·8128 (전 u 동일)
            for i in 0..d {
                for u2 in 0..d {
                    s0[i * d + u2] = (i + 1) as f32;
                }
            }
            for x in q.iter_mut() { *x = 1.0; }
            for x in k.iter_mut() { *x = 0.0; }
            for x in bg.iter_mut() { *x = 0.0; }
            for x in bg.iter_mut().skip(1).step_by(2) { *x = 1.0; }
        } else if std::env::var_os("LLM170_GDN_PROBE").is_some() {
            // beta=0, g=2: out = 2·scale·Σq·s
            for x in bg.iter_mut().step_by(2) { *x = 0.0; }
            for x in bg.iter_mut().skip(1).step_by(2) { *x = 2.0; }
        }
        let sb = ctx.alloc(s0.len() * 4)?;
        let qb = ctx.alloc(q.len() * 4)?;
        let kb = ctx.alloc(k.len() * 4)?;
        let vb = ctx.alloc(v.len() * 4)?;
        let bgb = ctx.alloc(bg.len() * 4)?;
        let ob = ctx.alloc(t * vs * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(s0.as_ptr(), sb.ptr as *mut f32, s0.len());
            std::ptr::copy_nonoverlapping(q.as_ptr(), qb.ptr as *mut f32, q.len());
            std::ptr::copy_nonoverlapping(k.as_ptr(), kb.ptr as *mut f32, k.len());
            std::ptr::copy_nonoverlapping(v.as_ptr(), vb.ptr as *mut f32, v.len());
            std::ptr::copy_nonoverlapping(bg.as_ptr(), bgb.ptr as *mut f32, bg.len());
        }
        let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/gdn_ar.spv"), 6, 28,
        )?;
        ctx.bind_bufs(ds, &[sb.buf, qb.buf, kb.buf, vb.buf, bgb.buf, ob.buf]);
        let scale = 1.0f32 / (d as f32).sqrt();
        let mut push: Vec<u8> = Vec::new();
        push.extend_from_slice(&[d as u32, ks as u32, vs as u32, hv as u32, hk as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        push.extend_from_slice(&scale.to_le_bytes());
        push.extend_from_slice(&(t as u32).to_le_bytes());
        ctx.run(pl, ds, pipe, &push, hv as u32, d as u32, 1)?;
        let mut r = vec![0f32; t * vs];
        let mut s_out = vec![0f32; s0.len()];
        unsafe {
            std::ptr::copy_nonoverlapping(ob.ptr as *const f32, r.as_mut_ptr(), r.len());
            std::ptr::copy_nonoverlapping(sb.ptr as *const f32, s_out.as_mut_ptr(), s_out.len());
        }
        // CPU 미러 (f64)
        let mut st = s0.clone();
        let mut c = vec![0f32; t * vs];
        for ti in 0..t {
            for pair in 0..hv {
                let h = pair % hv;
                let kh = h % hk;
                let base_s = pair * d * d;
                let beta = bg[ti * hv * 2 + pair * 2];
                let g = bg[ti * hv * 2 + pair * 2 + 1];
                let qk0 = ti * ks + kh * d;
                let v0 = ti * vs + h * d;
                // ssr *= g; sk = Σ k·s ; delta; s += k·delta; out = Σ q·s
                // g는 (pair, ti)당 1회 — 전 행 적용 후 u 순회
                for i in 0..d {
                    st[base_s + i * d + 0..base_s + i * d + d].iter_mut().for_each(|x| *x *= g);
                }
                let mut sk = 0.0f64;
                for u in 0..d {
                    let mut acc = 0.0f64;
                    for i in 0..d {
                        acc += st[base_s + i * d + u] as f64 * k[qk0 + i] as f64;
                    }
                    // 주의: rawhip 스레드 구조와 달리 여기 u=열, i=kdim 순으로 합 —
                    // 부동 합 순서 차이 허용 오차로 처리
                    sk = acc;
                    let delta = (v[v0 + u] as f64 - sk) * beta as f64;
                    for i in 0..d {
                        st[base_s + i * d + u] += (k[qk0 + i] * delta as f32) as f32;
                    }
                }
                let mut otot = 0.0f64;
                for u in 0..d {
                    let mut acc = 0.0f64;
                    for i in 0..d {
                        acc += st[base_s + i * d + u] as f64 * q[qk0 + i] as f64;
                    }
                    otot = acc;
                    c[v0 + u] = (otot * scale as f64) as f32;
                }
            }
        }
        if std::env::var_os("LLM170_GDN_PROBE").is_some() {
            // g=2·beta=0: s_out은 2·s0, out은 2·scale·Σq·s 여야
            let s_ratio = s_out[0] / s0[0];
            let o_ratio = if c[0].abs() > 1e-9 { r[0] / c[0] } else { f32::NAN };
            lines.push(format!("  probe: s_out/s0={s_ratio:.4} (기대 2.0) out/out_cpu={o_ratio:.4}"));
        }
        let maxd = r.iter().zip(c.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let fb = (0..r.len()).find(|&i| (r[i] - c[i]).abs() > 1e-3);
        let dbg = match fb {
            Some(i) => format!(" first_bad[{i}] gpu={:.6} cpu={:.6}", r[i], c[i]),
            None => String::new(),
        };
        lines.push(format!("gdn_ar: max|D|={maxd:.2e}{dbg} {}", if maxd < 1e-3 { "★" } else { "MISMATCH" }));
    }

    Ok(lines.join("\n"))
}

pub fn smoke_test() -> Result<String, String> {
    let mut ctx = VkCtx::new()?;
    unsafe {
        let props = ctx.instance.get_physical_device_properties(ctx.physical);
        let name = props.device_name_as_c_str().map_err(|_| "이름")?.to_string_lossy();
        let mut msg = format!(
            "vk: {name} (subgroup={}), coop_matrix={} coop_f16xf16_f32={}\n",
            props.limits.max_compute_work_group_size[0], ctx.coop_matrix, ctx.coop_f16_f32
        );
        // 트리비얼 컴퓨트: o[i] = i*scale + bias
        let n = 256usize;
        let ob = ctx.alloc(n * 4)?;
        let (dsl, pl, dp, ds, pipe) = ctx.pipeline(SMOKE_SPV, 1, 12)?;
        ctx.bind_bufs(ds, &[ob.buf]);
        let scale = 2.5f32;
        let bias = -1.0f32;
        let mut push = Vec::with_capacity(8);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&scale.to_le_bytes());
        push.extend_from_slice(&bias.to_le_bytes());
        let _ = dsl;
        ctx.run(pl, ds, pipe, &push, (n as u32).div_ceil(64), 1, 1)?;
        let host = std::slice::from_raw_parts(ob.ptr as *const f32, n);
        let mut bad = 0;
        for (i, &v) in host.iter().enumerate() {
            let want = i as f32 * scale + bias;
            if (v - want).abs() > 1e-6 {
                bad += 1;
            }
        }
        let _ = dp;
        if bad == 0 {
            msg += "smoke: ★ 256/256 값 일치\n";
        } else {
            msg += &format!("smoke: 불일치 {bad}/256\n");
        }
        ctx.device.destroy_pipeline(pipe, None);
        ctx.device.destroy_pipeline_layout(pl, None);
        ctx.device.destroy_descriptor_set_layout(dsl, None);
        ctx.device.destroy_descriptor_pool(dp, None);

        // 사용 가능 매트릭스 타입 열거
        {
            let cm = ash::khr::cooperative_matrix::Instance::new(&ctx.entry, &ctx.instance);
            let types = cm.get_physical_device_cooperative_matrix_properties(ctx.physical).map_err(|e| format!("{e:?}"))?;
            for t in &types {
                msg += &format!("cmtype: M={} N={} K={} A={:?} B={:?} R={:?} scope={:?} sat={}\n",
                    t.m_size, t.n_size, t.k_size, t.a_type, t.b_type, t.result_type, t.scope, t.saturating_accumulation);
            }
        }
        // coopmat 검증: A(0..255) × I = A
        let (dsl, pl, dp, ds, pipe) = ctx.pipeline(COOPMAT_PROBE_SPV, 3, 0)?;
        let ab = ctx.alloc(256 * 2)?;
        let bb = ctx.alloc(256 * 2)?;
        let cb = ctx.alloc(256 * 4)?;
        ctx.bind_bufs(ds, &[ab.buf, bb.buf, cb.buf]);
        let ah = std::slice::from_raw_parts_mut(ab.ptr as *mut u16, 256);
        for i in 0..256usize {
            ah[i] = f16_from_f32(i as f32);
        }
        let bh = std::slice::from_raw_parts_mut(bb.ptr as *mut u16, 256);
        for r in 0..16usize {
            for c in 0..16usize {
                bh[r * 16 + c] = if r == c { f16_from_f32(1.0) } else { 0 };
            }
        }
        ctx.run(pl, ds, pipe, &[], 1, 1, 1)?;
        let ch = std::slice::from_raw_parts(cb.ptr as *const f32, 256);
        let mut bad = 0;
        for i in 0..256usize {
            if (ch[i] - i as f32).abs() > 1e-3 {
                bad += 1;
                if bad <= 6 {
                    msg += &format!("  c[{i}]={:.3} want={:.3}\n", ch[i], i as f32);
                }
            }
        }
        if bad == 0 {
            msg += "coopmat16: ★ 256/256 (A×I=A)\n";
        } else {
            msg += &format!("coopmat16: 불일치 {bad}/256\n");
        }
        ctx.device.destroy_pipeline(pipe, None);
        ctx.device.destroy_pipeline_layout(pl, None);
        ctx.device.destroy_descriptor_set_layout(dsl, None);
        ctx.device.destroy_descriptor_pool(dp, None);

        // axpy_scaled: y += x*s — HIP 커널과 동일 산술, 16MB 규모
        let n = 4 * 1024 * 1024usize;
        let yb = ctx.alloc(n * 4)?;
        let xb = ctx.alloc(n * 4)?;
        let sb = ctx.alloc(4)?;
        let (dsl, pl, dp, ds, pipe) = ctx.pipeline(AXPY_SPV, 3, 4)?;
        let y = std::slice::from_raw_parts_mut(yb.ptr as *mut f32, n);
        let x = std::slice::from_raw_parts_mut(xb.ptr as *mut f32, n);
        let sc = std::slice::from_raw_parts_mut(sb.ptr as *mut f32, 1);
        let mut want = vec![0f64; n];
        for i in 0..n {
            y[i] = (i % 977) as f32 * 0.25 - 100.0;
            x[i] = ((i * 31) % 1231) as f32 * 0.5 - 300.0;
            want[i] = y[i] as f64 + x[i] as f64 * 1.75;
        }
        sc[0] = 1.75f32;
        ctx.bind_bufs(ds, &[yb.buf, xb.buf, sb.buf]);
        let push = (n as u32).to_le_bytes();
        let t0 = std::time::Instant::now();
        ctx.run(pl, ds, pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
        let dt = t0.elapsed().as_secs_f64();
        let mut bad = 0;
        for i in 0..n {
            if (y[i] as f64 - want[i]).abs() > 1e-3 {
                bad += 1;
            }
        }
        if bad == 0 {
            msg += &format!("axpy: ★ {n}/{n} 일치 ({:.2}ms, {:.1}GB/s)\n", dt * 1e3, (n as f64 * 4.0 * 3.0) / dt / 1e9);
        } else {
            msg += &format!("axpy: 불일치 {bad}/{n}\n");
        }
        ctx.device.destroy_pipeline(pipe, None);
        ctx.device.destroy_pipeline_layout(pl, None);
        ctx.device.destroy_descriptor_set_layout(dsl, None);
        ctx.device.destroy_descriptor_pool(dp, None);
        Ok(msg)
    }
}

/// f32→f16 비트 (절단 — 프로브 전용, 정밀도 무관).
fn f16_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign;
    }
    if exp >= 0x8F {
        return sign | 0x7C00;
    }
    let e = exp - 127 + 15;
    if e < 1 {
        return sign;
    }
    let m = mant >> 13;
    sign | ((e as u16) << 10) | m as u16
}
