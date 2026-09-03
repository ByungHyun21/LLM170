//! rawvk — Vulkan 백엔드 (plans/12). 모듈 루트.

pub mod context;

use context::VkCtx;

const SMOKE_SPV: &[u8] = include_bytes!("spv/smoke.spv");
const COOPMAT_PROBE_SPV: &[u8] = include_bytes!("spv/coopmat_probe.spv");
const AXPY_SPV: &[u8] = include_bytes!("spv/axpy_scaled.spv");

/// vk-check — 디바이스 역량 + 트리비얼 컴퓨트 값 검증 (M1 게이트).
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
