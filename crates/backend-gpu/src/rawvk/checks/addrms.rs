//! addrms 직접 검증기 (plans/129 A22② — Q4 vk 값경로 융합커널 1:1).
//!
//! addrms.comp(잔차+rms_norm 융합, plans/36 G2→46)은 종전 assembly
//! (charhash+token 게이트)로만 덮여 있었다 — norm_resid xo-aliasing(corr
//! 0.637→0.945, 며칠) 클래스가 조립 corr에 가려지는 바로 그 빈자리.
//! 미러는 커널 산식을 정밀히 재현(세그먼트 256×chunk f32 연속 누산 →
//! f64 순차 결합(u=0..255) → sqrt f64→f32 캐스팅) — 비트 일치 기대.

use crate::rawvk::vkacc::VkAcc;

/// CPU 미러 — 커널 main()의 연산 순서 그대로(프로브 미러 규약, 원장).
fn addrms_mirror(y: &mut [f32], x: &[f32], wv: &[f32], out: &mut [f32], n: usize, eps: f32) {
    let chunk = (n + 255) >> 8;
    for row in 0..(y.len() / n) {
        let (ys, os) = (
            &mut y[row * n..(row + 1) * n],
            &mut out[row * n..(row + 1) * n],
        );
        let xs = &x[row * n..(row + 1) * n];
        let mut seg = [0f64; 256];
        for u in 0..256usize {
            let lo = u * chunk;
            if lo >= n {
                continue;
            }
            let hi = (lo + chunk).min(n);
            let mut acc = 0f32;
            for i in lo..hi {
                let v = ys[i] + xs[i] * 1.0;
                ys[i] = v;
                acc += v * v;
            }
            seg[u] = acc as f64;
        }
        let mut sum = 0f64;
        for u in 0..256usize {
            if u * chunk < n {
                sum += seg[u];
            }
        }
        let scale32 = ((sum / n as f64 + eps as f64).sqrt()) as f32;
        let invs = 1.0f32 / scale32;
        for i in 0..n {
            os[i] = ys[i] * invs * wv[i];
        }
    }
}

/// `llm170 addrms-check` — 합성 LCG 입력·형상 스윕(n=5120 정규 + n=100 홀수
/// 부분 청크 가드, t=1/4/16)으로 GPU↔미러 비트 대조.
pub fn addrms_check() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let spv =
        std::fs::read("crates/backend-gpu/src/rawvk/spv/addrms.spv").map_err(|e| e.to_string())?;
    let mut lines = String::new();
    for &(n, t) in &[(5120usize, 1usize), (5120, 4), (5120, 16), (100, 3)] {
        let mut seed = 0x0ad0_0770_u64 | ((n as u64) << 32) ^ (t as u64);
        let mut lcg = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / 2147483648.0 - 0.5
        };
        let y0: Vec<f32> = (0..n * t).map(|_| lcg()).collect();
        let x: Vec<f32> = (0..n * t).map(|_| lcg()).collect();
        let wv: Vec<f32> = (0..n).map(|_| 1.0 + lcg()).collect();
        let eps = 1e-5f32;

        let by = ctx.alloc(n * t * 4)?;
        let bx = ctx.alloc(n * t * 4)?;
        let bw = ctx.alloc(n * 4)?;
        let bo = ctx.alloc(n * t * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytemuck::cast_slice::<f32, u8>(&y0).as_ptr(),
                by.ptr,
                n * t * 4,
            );
            std::ptr::copy_nonoverlapping(
                bytemuck::cast_slice::<f32, u8>(&x).as_ptr(),
                bx.ptr,
                n * t * 4,
            );
            std::ptr::copy_nonoverlapping(
                bytemuck::cast_slice::<f32, u8>(&wv).as_ptr(),
                bw.ptr,
                n * 4,
            );
        }
        // 비결합 호스트 메모리 계약(원장 — alloc_host_cached): CPU 쓰기 후 flush,
        // GPU 쓰기 판독 전 invalidate. 누락시 GPU가 0을 읽어 y'=y+0로 침묵 통과한다.
        let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 4, 12)?;
        let _ = (dsl, pool);
        ctx.bind_bufs(ds, &[by.buf, bx.buf, bw.buf, bo.buf]);
        let push: Vec<u8> = (n as u32)
            .to_le_bytes()
            .iter()
            .chain((t as u32).to_le_bytes().iter())
            .chain(eps.to_le_bytes().iter())
            .copied()
            .collect();
        let t0 = Instant::now();
        ctx.run(pl, ds, pipe, &push, t as u32, 1, 1)?;
        ctx.wait_pending()?;
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        let (mut yc, mut oc) = (y0.clone(), vec![0f32; n * t]);
        addrms_mirror(&mut yc, &x, &wv, &mut oc, n, eps);
        // 판독(호스트 가시 버퍼) — SAFETY: run 동기 완료 후 재해석.
        let (yg, og): (&[f32], &[f32]) = unsafe {
            (
                std::slice::from_raw_parts(by.ptr as *const f32, n * t),
                std::slice::from_raw_parts(bo.ptr as *const f32, n * t),
            )
        };
        let md = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        let (mdy, mdo) = (md(yg, &yc), md(og, &oc));
        let bits = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count()
        };
        lines.push_str(&format!(
            "  n={n} t={t}: y maxdiff={mdy:.3e}({}불일치) out maxdiff={mdo:.3e}({}불일치) · {dt:.2}ms\n",
            bits(yg, &yc),
            bits(og, &oc),
        ));
        unsafe {
            ctx.device.destroy_pipeline(pipe, None);
            ctx.device.destroy_pipeline_layout(pl, None);
        }
    }
    Ok(format!(
        "addrms-check (잔차+rms 융합, A22② 직접 검증):\n{lines}"
    ))
}
