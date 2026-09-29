//! 기본 연산(op) 체커 — idot 프로브(plans/107 W4: checks.rs에서 분할).

use crate::rawvk::vkacc::{VkAcc, push_u32s};

/// vk-idot-probe (plans/89 P0.1) — OpSDot(PackedVectorFormat4x8Bit) 검증+타이밍.
/// sdot_probe(plans/33)의 어셈블리 패치는 커널 문맥에서 0을 반환했다. 이번 판의
/// 차이: (a) VkCtx가 Vulkan13Features.shader_integer_dot_product를 활성화,
/// (b) spirv-as 산출물을 val 통과 구조로 직접 인코딩(.spvasm 참조).
/// mode 0=OpSDot / 1=스칼라 에뮬레이션(gemv3 dot4 동일 산술) — 동일 커널 A/B.
pub fn idot_probe() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    if !acc.ctx.lock().idot {
        return Ok("idot-probe: 장치가 shader_integer_dot_product 미지원".into());
    }
    let mut ctx = acc.ctx.lock();
    let buf = ctx.alloc_host(32)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/idot_probe.spv")
        .map_err(|e| e.to_string())?;
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 1, 8)?;
    let _ = (dsl, pool);
    ctx.bind_bufs(ds, &[buf.buf]);
    // CPU 기준 — 단일 dot(비영 검증) + 1M 의존 루프 종값.
    let bx = |v: i32, i: u32| -> i32 {
        let b = (v >> (i * 8)) & 0xFF;
        if b >= 128 { b - 256 } else { b }
    };
    let (ai, bi): (i32, i32) = (0x0182_0304u32 as i32, 0xF0FF_7F01u32 as i32);
    let single: i32 = (0..4).map(|i| bx(ai, i) * bx(bi, i)).sum();
    let mut cacc = ai;
    for _ in 0..1_000_000 {
        let mut s = 0i32;
        for i in 0..4 {
            s += bx(cacc, i) * bx(bi, i);
        }
        cacc = s;
    }
    let mut lines = String::new();
    for mode in 0..2u32 {
        unsafe {
            let p = buf.ptr as *mut u32;
            *p.add(0) = 0x0182_0304;
            *p.add(1) = 0xF0FF_7F01;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let t0 = Instant::now();
        ctx.run(pl, ds, pipe, &push_u32s(&[mode, 1_000_000]), 1024, 1, 1)?;
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        let (r2, r3) = unsafe {
            (
                *(buf.ptr as *const u32).add(2),
                *(buf.ptr as *const u32).add(3) as i32,
            )
        };
        let ok_loop = r2 == cacc as u32;
        let ok_single = r3 == single;
        lines.push_str(&format!(
            "  mode{mode}({}): 루프 {r2:#010x} {} · 단일 dot {r3} (cpu {single}) {} · {dt:.1}ms/1M\n",
            if mode == 0 { "OpSDot" } else { "스칼라" },
            if ok_loop { "★" } else { "✗" },
            if ok_single { "★" } else { "✗" },
        ));
    }
    unsafe {
        ctx.device.destroy_pipeline(pipe, None);
        ctx.device.destroy_pipeline_layout(pl, None);
    }
    Ok(format!(
        "idot-probe (packed i8x4 dot, plans/89 P0.1):\n{lines}"
    ))
}
