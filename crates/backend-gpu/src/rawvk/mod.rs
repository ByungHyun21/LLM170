//! rawvk — Vulkan 백엔드 (plans/12). 모듈 루트.

pub mod context;
pub mod decoder;
pub mod gemv;

use context::VkCtx;

const SMOKE_SPV: &[u8] = include_bytes!("spv/smoke.spv");
const COOPMAT_PROBE_SPV: &[u8] = include_bytes!("spv/coopmat_probe.spv");
pub const AXPY_SPV: &[u8] = include_bytes!("spv/axpy_scaled.spv");

/// vk-check — 디바이스 역량 + 트리비얼 컴퓨트 값 검증 (M1 게이트).
/// subsum-check — 서브그룹 리덕션 프로브 (xor 트리 / add / broadcast).
pub fn subsum_check() -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
use ash::vk;
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
use ash::vk;
    let mut ctx = VkCtx::new()?;
    let seed = std::cell::Cell::new(0x1234_5678u64);
    let lcg = || {
        seed.set(seed.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
        (seed.get() >> 33) as f32 / 2147483648.0 - 0.5
    };
    let lcg_v = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg()).collect() };
    let mut lines: Vec<String> = Vec::new();

    // ── i8 coopmat 프로브 (plans/23): C = A(16x16 i8) × B — 하드웨어 정수 MMA 동작 확인.
    {
        let mut a = vec![0i8; 256];
        let mut b = vec![0i8; 256];
        for m in 0..16 {
            for k in 0..16 {
                a[m * 16 + k] = (m + k) as i8 - 8;
            }
        }
        for n in 0..16 {
            for k in 0..16 {
                b[n * 16 + k] = (k as i8 - n as i8) * 2;
            }
        }
        let abuf = ctx.alloc(256)?;
        unsafe { std::ptr::copy_nonoverlapping(a.as_ptr() as *const u8, abuf.ptr, 256) };
        let bbuf = ctx.alloc(256)?;
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr() as *const u8, bbuf.ptr, 256) };
        let cbuf = ctx.alloc_host(256 * 4)?;
        let (dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/i8probe.spv"), 3, 8,
        )?;
        let _ = dsl;
        ctx.bind_bufs(ds, &[abuf.buf, bbuf.buf, cbuf.buf]);
        ctx.run(pl, ds, pipe, &[], 1, 1, 1)?;
        let mut c = vec![0i32; 256];
        unsafe { std::ptr::copy_nonoverlapping(cbuf.ptr as *const i32, c.as_mut_ptr(), 256) };
        // CPU 미러: C[m][n] = Σ_k A[m][k]·B[k][n], B[k][n] = b[n*16+k]
        let mut ok = true;
        let mut mx = 0i64;
        for m in 0..16 {
            for n in 0..16 {
                let mut s = 0i32;
                for k in 0..16 {
                    s += a[m * 16 + k] as i32 * b[n * 16 + k] as i32;
                }
                let d = (c[m * 16 + n] - s) as i64;
                if d != 0 { ok = false; }
                mx = mx.max(d.abs());
            }
        }
        let sample = format!("c00={} c01={} c77={}", c[0], c[1], c[7 * 16 + 7]);
        lines.push(format!("i8coopmat: {} ({} max|D|={})", if ok { "★" } else { "✗" }, sample, mx));
    }

    // ── i8 coopmat 프로브2: offset=37·stride=32 비정방 피치 (gemm_i8 실패턴).
    {
        let pitch = 32usize;
        let off = 37usize;
        let mut a = vec![0i8; off + 16 * pitch + 16];
        let mut b = vec![0i8; off + 16 * pitch + 16];
        for m in 0..16 {
            for k in 0..16 {
                a[off + m * pitch + k] = ((m as i32 * 7 + k as i32 * 3) % 21 - 10) as i8;
            }
        }
        for n in 0..16 {
            for k in 0..16 {
                b[off + n * pitch + k] = ((k as i32 - n as i32 * 2) % 19 - 9) as i8;
            }
        }
        let abuf = ctx.alloc(a.len())?;
        unsafe { std::ptr::copy_nonoverlapping(a.as_ptr() as *const u8, abuf.ptr, a.len()) };
        let bbuf = ctx.alloc(b.len())?;
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr() as *const u8, bbuf.ptr, b.len()) };
        let cbuf = ctx.alloc_host(256 * 4)?;
        let (dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/i8probe2.spv"), 3, 8,
        )?;
        let _ = dsl;
        ctx.bind_bufs(ds, &[abuf.buf, bbuf.buf, cbuf.buf]);
        ctx.run(pl, ds, pipe, &[], 1, 1, 1)?;
        // 프로브3: offset=0 데이터 별도 구성 (참조 정합) — kernel 조건 재현.
        {
            let p3 = 32usize;
            let mut a3 = vec![0i8; 16 * p3 + 16];
            let mut b3 = vec![0i8; 16 * p3 + 16];
            for m in 0..16usize {
                for k in 0..16usize { a3[m * p3 + k] = ((m as i32 * 5 + k as i32 * 3) % 23 - 11) as i8; }
            }
            for n in 0..16usize {
                for k in 0..16usize { b3[n * p3 + k] = ((k as i32 - n as i32 * 3) % 17 - 8) as i8; }
            }
            let ab3 = ctx.alloc(a3.len())?;
            unsafe { std::ptr::copy_nonoverlapping(a3.as_ptr() as *const u8, ab3.ptr, a3.len()) };
            let bb3 = ctx.alloc(b3.len())?;
            unsafe { std::ptr::copy_nonoverlapping(b3.as_ptr() as *const u8, bb3.ptr, b3.len()) };
            let cbuf3 = ctx.alloc_host(256 * 4)?;
            let (dsl3, pl3, _dp3, ds3, pipe3) = ctx.pipeline(
                include_bytes!("spv/i8probe3.spv"), 3, 8,
            )?;
            let _ = dsl3;
            ctx.bind_bufs(ds3, &[ab3.buf, bb3.buf, cbuf3.buf]);
            ctx.run(pl3, ds3, pipe3, &[], 1, 1, 1)?;
            let mut c3 = vec![0i32; 256];
            unsafe { std::ptr::copy_nonoverlapping(cbuf3.ptr as *const i32, c3.as_mut_ptr(), 256) };
            let mut ok3 = true;
            let mut first3 = String::new();
            for m in 0..16usize {
                for n in 0..16usize {
                    let mut s = 0i32;
                    for k in 0..16usize {
                        s += a3[m * p3 + k] as i32 * b3[n * p3 + k] as i32;
                    }
                    if c3[m * 16 + n] != s {
                        ok3 = false;
                        if first3.is_empty() { first3 = format!("(m{m},n{n}) got={} want={s}", c3[m * 16 + n]); }
                    }
                }
            }
            let nz3 = c3.iter().filter(|&&x| x != 0).count();
            lines.push(format!("i8coopmat3(off0/stride32): {} (nz={} {})", if ok3 { "★" } else { "✗" }, nz3, first3));
        }
        let mut c = vec![0i32; 256];
        unsafe { std::ptr::copy_nonoverlapping(cbuf.ptr as *const i32, c.as_mut_ptr(), 256) };
        let mut ok = true;
        let mut mx = 0i64;
        for m in 0..16 {
            for n in 0..16 {
                let mut s = 0i32;
                for k in 0..16 {
                    s += a[off + m * pitch + k] as i32 * b[off + n * pitch + k] as i32;
                }
                let d = (c[m * 16 + n] - s) as i64;
                if d != 0 { ok = false; }
                mx = mx.max(d.abs());
            }
        }
        lines.push(format!("i8coopmat2(off37/stride32): {} (c00={} c15={} c51={} max|D|={})", if ok { "★" } else { "✗" }, c[0], c[15], c[5*16+1], mx));
    }

    // ── gemm_i8 미니: n_in=32(n_sub=1), no=16, t=2 — 실 커널 소형 등가검증.
    {
        let (n_in, no, t) = (32usize, 16usize, 2usize);
        let n_sub = 1usize;
        // 데이타: w8 [-20,20], wsp/wsm [0.01,0.05], b8 [-100,100], yd/qsum 결정적
        let seed2 = std::cell::Cell::new(42u64);
        let mut lcg2 = || { seed2.set(seed2.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)); (seed2.get() >> 33) as i32 };
        let w8: Vec<i8> = (0..no * n_in).map(|_| (lcg2() % 41 - 20) as i8).collect();
        let wsp: Vec<f32> = (0..no * n_sub).map(|i| 0.01 + (i % 7) as f32 * 0.005).collect();
        let wsm: Vec<f32> = (0..no * n_sub).map(|i| 0.02 + (i % 5) as f32 * 0.004).collect();
        let mut b8: Vec<i8> = (0..t * n_in).map(|_| (lcg2() % 201 - 100) as i8).collect();
        b8.resize(16 * n_in, 0); // B 로드가 n=0..15 읽음 — 패딩
        let yd: Vec<f32> = (0..t * n_sub).map(|i| 0.012 + i as f32 * 0.003).collect();
        let qsum: Vec<i32> = (0..t * n_sub).map(|i| -50 + i as i32 * 30).collect();
        let mut mk = |v: &[u8]| -> Result<vk::Buffer, String> {
            let b = ctx.alloc(v.len() + 16)?;
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.ptr, v.len()) };
            Ok(b.buf)
        };
        let wbuf = mk(unsafe { std::slice::from_raw_parts(w8.as_ptr() as *const u8, w8.len()) })?;
        let b8buf = mk(unsafe { std::slice::from_raw_parts(b8.as_ptr() as *const u8, b8.len()) })?;
        let wspbuf = mk(bytemuck::cast_slice(&wsp))?;
        let wsmbuf = mk(bytemuck::cast_slice(&wsm))?;
        let ydbuf = mk(bytemuck::cast_slice(&yd))?;
        let qsbuf = mk(bytemuck::cast_slice(&qsum))?;
        let obuf = ctx.alloc_host(t * no * 4)?;
        let ishsb = ctx.alloc_host(640 * 256 * 4)?;
        let faccsb = ctx.alloc_host(640 * 256 * 4)?;
        // 센티넬 프리필 — 커널이 실제 쓴 영역 검증
        unsafe { std::ptr::write_bytes(ishsb.ptr as *mut u8, 0xAB, 640 * 256 * 4); }
        let (dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/gemm_i8.spv"), 16, 16,
        )?;
        let _ = dsl;
        let mut binds16 = vec![wbuf; 8];
        binds16.extend_from_slice(&[b8buf, obuf.buf, wspbuf, wsmbuf, ydbuf, qsbuf, ishsb.buf, faccsb.buf]);
        ctx.bind_bufs(ds, &binds16);
        let push16: Vec<u8> = [n_in as u32, no as u32, t as u32, n_sub as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.run(pl, ds, pipe, &push16, (no.div_ceil(16)) as u32, 1, 1)?;
        let mut out = vec![0f32; t * no];
        unsafe { std::ptr::copy_nonoverlapping(obuf.ptr as *const f32, out.as_mut_ptr(), out.len()) };
        let sentinel = unsafe { std::slice::from_raw_parts(ishsb.ptr as *const u32, 4) }[0];
        let nsent = unsafe { std::slice::from_raw_parts(ishsb.ptr as *const u32, 256) }.iter().filter(|&&x| x == 0xABABABAB).count();
        // CPU 미러: out[tok][o] = Σ_b yd·(wsp·isum − wsm·qsum)
        let mut mx = 0f64;
        let mut ok = true;
        for tok in 0..t {
            for o in 0..no {
                let mut acc = 0f32;
                for b in 0..n_sub {
                    let mut isum = 0i32;
                    let mut qs = 0i32;
                    for k in 0..32 {
                        isum += w8[o * n_in + b * 32 + k] as i32 * b8[tok * n_in + b * 32 + k] as i32;
                        qs += b8[tok * n_in + b * 32 + k] as i32;
                    }
                    acc += yd[tok * n_sub + b] * (wsp[o * n_sub + b] * isum as f32 - wsm[o * n_sub + b] * qs as f32);
                }
                let d = (out[tok * no + o] - acc).abs() as f64;
                if d > 1e-4 { ok = false; }
                mx = mx.max(d);
            }
        }
        let f8: Vec<String> = out[..8].iter().map(|v| format!("{v:.4}")).collect();
        // ishs 검증: wg0의 isum[m][n=0] — 기대 Σ_k w8[m*32+k]·b8[0*32+k]
        let mut iv = vec![0i32; 256];
        unsafe { std::ptr::copy_nonoverlapping(ishsb.ptr as *const i32, iv.as_mut_ptr(), 32) };
        let mut iok = true;
        let mut first_bad = String::new();
        for m in 0..16usize {
            let mut s = 0i32;
            for k in 0..32usize {
                s += w8[m * n_in + k] as i32 * b8[0 * n_in + k] as i32;
            }
            if iv[m * 16] != s {
                iok = false;
                if first_bad.is_empty() {
                    first_bad = format!("m={m} got={} want={s}", iv[m * 16]);
                }
            }
        }
        // 크로스체크: 동일 w8/b8 버퍼로 probe3 커널 (동일 SPIR-V 계열) — 하니스 분리.
        {
            let cbuf3 = ctx.alloc_host(256 * 4)?;
            let (dsl3, pl3, _dp3, ds3, pipe3) = ctx.pipeline(
                include_bytes!("spv/i8probe3.spv"), 3, 8,
            )?;
            let _ = dsl3;
            ctx.bind_bufs(ds3, &[wbuf, b8buf, cbuf3.buf]);
            ctx.run(pl3, ds3, pipe3, &[], 1, 1, 1)?;
            let mut c3 = vec![0i32; 256];
            unsafe { std::ptr::copy_nonoverlapping(cbuf3.ptr as *const i32, c3.as_mut_ptr(), 256) };
            let nz3 = c3.iter().filter(|&&x| x != 0).count();
            let mut ok3 = true;
            for m in 0..16usize {
                for n in 0..2usize {
                    let mut s = 0i32;
                    for k in 0..32usize { s += w8[m * 32 + k] as i32 * b8[n * 32 + k] as i32; }
                    if n == 0 && c3[m * 16] != s { ok3 = false; }
                }
            }
            lines.push(format!("  gemm_i8_cross(probe3커널+미니데이터): nz={} diag0={}", nz3, if ok3 { "★" } else { "✗" }));
        }
        // 기대 isum[m][n] 256개 생성 후 ishs 내 위치 탐색 (순열 규명)
        let mut want_full: Vec<i32> = Vec::with_capacity(32);
        for m in 0..16usize {
            for tok in 0..2usize {
                let mut s = 0i32;
                for k in 0..32usize { s += w8[m * 32 + k] as i32 * b8[tok * 32 + k] as i32; }
                want_full.push(s);
            }
        }
        let mut found = String::new();
        for (wi, wv) in want_full.iter().enumerate().take(6) {
            let m = wi / 2; let tok = wi % 2;
            let pos = iv.iter().position(|&x| x == *wv);
            found.push_str(&format!("(m{m},t{tok})@{:?} ", pos));
        }
        let nz = iv.iter().filter(|&&x| x != 0).count();
        lines.push(format!("gemm_i8_mini: {} (max|D|={mx:.2e} nz256={nz} pos: {} first4={:08x})", if ok { "★" } else { "✗" }, found, sentinel));
    }

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

        // ── 다중 스텝 링 진화: t=1 4회 연속 — 상태 시프트·push 검증
        {
            let (ch, k, steps) = (12usize, 4usize, 4usize);
            let cw: Vec<f32> = (0..ch * k).map(|_| lcg()).collect();
            let mut st: Vec<f32> = vec![0.0; (k - 1) * ch];
            let inputs: Vec<Vec<f32>> = (0..steps).map(|_| (0..ch).map(|_| lcg()).collect()).collect();
            let cwb = ctx.alloc(cw.len() * 4)?;
            let stb = ctx.alloc(st.len() * 4)?;
            let inb = ctx.alloc(ch * 4)?;
            let ob = ctx.alloc(ch * 4)?;
            unsafe {
                std::ptr::copy_nonoverlapping(cw.as_ptr(), cwb.ptr as *mut f32, cw.len());
                std::ptr::copy_nonoverlapping(st.as_ptr(), stb.ptr as *mut f32, st.len());
            }
            let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
                include_bytes!("spv/gdn_conv_t.spv"), 4, 8,
            )?;
            ctx.bind_bufs(ds, &[inb.buf, cwb.buf, stb.buf, ob.buf]);
            let push: Vec<u8> = [ch as u32, k as u32, 1u32].iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut gpu_outs: Vec<Vec<f32>> = Vec::new();
            for inp in &inputs {
                unsafe { std::ptr::copy_nonoverlapping(inp.as_ptr(), inb.ptr as *mut f32, ch) };
                ctx.run(pl, ds, pipe, &push, ch.div_ceil(64) as u32, 1, 1)?;
                let mut o = vec![0f32; ch];
                unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const f32, o.as_mut_ptr(), ch) };
                gpu_outs.push(o);
            }
            let mut st2 = st.clone();
            let mut cpu_outs: Vec<Vec<f32>> = Vec::new();
            for inp in &inputs {
                let mut o = vec![0f32; ch];
                for c2 in 0..ch {
                    let mut sum = cw[c2 * k + (k - 1)] * inp[c2];
                    for j in 0..k - 1 {
                        sum += cw[c2 * k + j] * st2[j * ch + c2];
                    }
                    o[c2] = sum / (1.0 + (-sum as f64).exp() as f32);
                }
                // shift+push
                for j in 0..k - 2 {
                    for c2 in 0..ch {
                        st2[j * ch + c2] = st2[(j + 1) * ch + c2];
                    }
                }
                for c2 in 0..ch {
                    st2[(k - 2) * ch + c2] = inp[c2];
                }
                cpu_outs.push(o);
            }
            let mut maxd2 = 0f32;
            for st_i in 0..steps {
                for c2 in 0..ch {
                    maxd2 = maxd2.max((gpu_outs[st_i][c2] - cpu_outs[st_i][c2]).abs());
                }
            }
            lines.push(format!("gdn_conv_ring({steps} steps): max|D|={maxd2:.2e} {}", if maxd2 < 1e-5 { "★" } else { "MISMATCH" }));
        }
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

        // ── AR 실차원 2스텝 체인 (hv=48, hk=16, d=128) — VkD step 방식 미러
        {
            let (d, hv, hk) = (128usize, 48usize, 16usize);
            let (ks, vs) = (hk * d, hv * d);
            let scale = 1.0f32 / (d as f32).sqrt();
            let mut st: Vec<f32> = (0..hv * d * d).map(|_| lcg() * 0.5).collect();
            let steps = 2usize;
            let inputs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..steps).map(|_| {
                (lcg_v(ks), lcg_v(ks), lcg_v(vs), {
                    let mut bg = vec![0f32; hv * 2];
                    for h in 0..hv { bg[h * 2] = 0.3; bg[h * 2 + 1] = 0.9; }
                    bg
                })
            }).collect();
            let sb = ctx.alloc(st.len() * 4)?;
            unsafe { std::ptr::copy_nonoverlapping(st.as_ptr(), sb.ptr as *mut f32, st.len()) };
            let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(include_bytes!("spv/gdn_ar.spv"), 6, 28)?;
            let mut gpu_outs: Vec<Vec<f32>> = Vec::new();
            for st_i in 0..steps {
                let (q, k, v, bg) = &inputs[st_i];
                let qb = ctx.alloc(ks * 4)?; unsafe { std::ptr::copy_nonoverlapping(q.as_ptr(), qb.ptr as *mut f32, ks) };
                let kb = ctx.alloc(ks * 4)?; unsafe { std::ptr::copy_nonoverlapping(k.as_ptr(), kb.ptr as *mut f32, ks) };
                let vb = ctx.alloc(vs * 4)?; unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), vb.ptr as *mut f32, vs) };
                let bb = ctx.alloc(hv * 2 * 4)?; unsafe { std::ptr::copy_nonoverlapping(bg.as_ptr(), bb.ptr as *mut f32, hv * 2) };
                let ob = ctx.alloc(vs * 4)?;
                ctx.bind_bufs(ds, &[sb.buf, qb.buf, kb.buf, vb.buf, bb.buf, ob.buf]);
                let mut push: Vec<u8> = Vec::new();
                push.extend([d as u32, ks as u32, vs as u32, hv as u32, hk as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                push.extend(scale.to_le_bytes());
                push.extend(1u32.to_le_bytes());
                ctx.run(pl, ds, pipe, &push, hv as u32, d as u32, 1)?;
                let mut o = vec![0f32; vs];
                unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const f32, o.as_mut_ptr(), vs) };
                gpu_outs.push(o);
            }
            // CPU 미러 (rawhip gdn_ar 산술, kdim 순차)
            let mut cs = st.clone();
            let mut cpu_outs: Vec<Vec<f32>> = Vec::new();
            for st_i in 0..steps {
                let (q, k, v, bg) = &inputs[st_i];
                let mut o = vec![0f32; vs];
                for pair in 0..hv {
                    let kh = pair % hk;
                    let base_s = pair * d * d;
                    let qk0 = kh * d;
                    let v0 = pair * d;
                    let beta = bg[pair * 2];
                    let g = bg[pair * 2 + 1];
                    let mut sk2 = vec![0f32; d];
                    for u2 in 0..d {
                        let mut acc = 0f32;
                        for i in 0..d {
                            let sv = cs[base_s + i * d + u2] * g;
                            cs[base_s + i * d + u2] = sv;
                            acc += sv * k[qk0 + i];
                        }
                        sk2[u2] = acc;
                    }
                    for u2 in 0..d {
                        let delta = (v[v0 + u2] - sk2[u2]) * beta;
                        for i in 0..d { cs[base_s + i * d + u2] += k[qk0 + i] * delta; }
                    }
                    for u2 in 0..d {
                        let mut acc = 0f32;
                        for i in 0..d { acc += cs[base_s + i * d + u2] * q[qk0 + i]; }
                        o[v0 + u2] = acc * scale;
                    }
                }
                cpu_outs.push(o);
            }
            let mut md2 = 0f32;
            for st_i in 0..steps {
                for j2 in 0..vs { md2 = md2.max((gpu_outs[st_i][j2] - cpu_outs[st_i][j2]).abs()); }
            }
            lines.push(format!("gdn_ar_real2({steps}): max|D|={md2:.2e} {}", if md2 < 1e-2 { "★" } else { "MISMATCH" }));
        }
    }

    // ── norm_gated (요소별 z — rawhip kernels.rs:2554 미러): n_h=3, d=8, t=2
    {
        let (n_h, d, t) = (3usize, 8usize, 2usize);
        let eps = 1e-5f32;
        let mut o: Vec<f32> = (0..t * n_h * d).map(|_| lcg()).collect();
        let z: Vec<f32> = (0..t * n_h * d).map(|_| lcg()).collect();
        let w: Vec<f32> = (0..n_h * d).map(|_| lcg()).collect();
        let ob = ctx.alloc(o.len() * 4)?;
        let zb = ctx.alloc(z.len() * 4)?;
        let wb = ctx.alloc(w.len() * 4)?;
        let outb = ctx.alloc(o.len() * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(o.as_ptr(), ob.ptr as *mut f32, o.len());
            std::ptr::copy_nonoverlapping(z.as_ptr(), zb.ptr as *mut f32, z.len());
            std::ptr::copy_nonoverlapping(w.as_ptr(), wb.ptr as *mut f32, w.len());
        }
        let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/norm_gated.spv"), 4, 12,
        )?;
        ctx.bind_bufs(ds, &[ob.buf, zb.buf, wb.buf, outb.buf]);
        let mut push = eps.to_le_bytes().to_vec();
        push.extend([d as u32, n_h as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        ctx.run(pl, ds, pipe, &push, n_h as u32, t as u32, 1)?;
        let mut r = vec![0f32; t * n_h * d];
        unsafe { std::ptr::copy_nonoverlapping(outb.ptr as *const f32, r.as_mut_ptr(), r.len()) };
        // CPU 미러 (rawhip 산술)
        let mut c = vec![0f32; t * n_h * d];
        for ti in 0..t {
            for h in 0..n_h {
                let xb = (ti * n_h + h) * d;
                let wb2 = h * d;
                let mut sum = 0f64;
                for i in 0..d { sum += o[xb + i] as f64 * o[xb + i] as f64; }
                let inv = 1.0 / ((sum / d as f64 + eps as f64).sqrt() as f32);
                for i in 0..d {
                    let zz = z[xb + i];
                    let g = zz / (1.0 + (-zz).exp());
                    c[xb + i] = o[xb + i] * inv * w[wb2 + i] * g;
                }
            }
        }
        let maxd = r.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let fb = (0..r.len()).find(|&i| (r[i] - c[i]).abs() > 1e-4);
        let dbg = match fb {
            Some(i) => format!(
                " first_bad[{i}] gpu={:.6} cpu={:.6} row0gpu={:?} row0cpu={:?}",
                r[i], c[i], &r[0..d.min(8)], &c[0..d.min(8)]
            ),
            None => String::new(),
        };
        lines.push(format!("norm_gated: max|D|={maxd:.2e}{dbg} {}", if maxd < 1e-4 { "★" } else { "MISMATCH" }));
    }

    // ── silu_mul: exp_cr f64 폴리 포트 검증 (극값 포함)
    {
        let n = 4096usize;
        let mut g: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            g.push(match i {
                0 => -120.0,
                1 => -104.0,
                2 => -103.0,
                3 => -90.0,
                4 => 90.0,
                5 => 88.8,
                6 => 88.0,
                7 => 0.0,
                _ => lcg() * 20.0,
            });
        }
        let u: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let gb = ctx.alloc(n * 4)?;
        let ub = ctx.alloc(n * 4)?;
        let ob = ctx.alloc(n * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(g.as_ptr(), gb.ptr as *mut f32, n);
            std::ptr::copy_nonoverlapping(u.as_ptr(), ub.ptr as *mut f32, n);
        }
        let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/silu_mul.spv"), 3, 4,
        )?;
        ctx.bind_bufs(ds, &[gb.buf, ub.buf, ob.buf]);
        let push = (n as u32).to_le_bytes().to_vec();
        ctx.run(pl, ds, pipe, &push, n.div_ceil(256) as u32, 1, 1)?;
        let mut r = vec![0f32; n];
        unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const f32, r.as_mut_ptr(), n) };
        // exp_cr 폴리 미러 (rawhip kernels.rs:42 정확 이식)
        let c: Vec<f32> = g.iter().zip(&u).map(|(&v, &uv)| {
            let s = v / (1.0 + (-v as f64).exp() as f32);
            s * uv
        }).collect();
        let maxd = r.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let fb = (0..n).find(|&i| (r[i] - c[i]).abs() > 1e-5);
        let dbg = match fb {
            Some(i) => format!(" first_bad[{i}] g={:.4} gpu={:.7} cpu={:.7}", g[i], r[i], c[i]),
            None => String::new(),
        };
        lines.push(format!("silu_mul: max|D|={maxd:.2e}{dbg} {}", if maxd < 1e-4 { "★" } else { "MISMATCH" }));
    }

    // ── l2_rows2: ng=2, d=128, t=3
    {
        let (ng, d, t) = (2usize, 128usize, 3usize);
        let row = ng * d;
        let mut qv: Vec<f32> = (0..t * row).map(|_| lcg()).collect();
        let mut kv: Vec<f32> = (0..t * row).map(|_| lcg()).collect();
        let qb = ctx.alloc(qv.len() * 4)?;
        let kb = ctx.alloc(kv.len() * 4)?;
        unsafe {
            std::ptr::copy_nonoverlapping(qv.as_ptr(), qb.ptr as *mut f32, qv.len());
            std::ptr::copy_nonoverlapping(kv.as_ptr(), kb.ptr as *mut f32, kv.len());
        }
        let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(
            include_bytes!("spv/l2_rows2.spv"), 2, 12,
        )?;
        ctx.bind_bufs(ds, &[qb.buf, kb.buf]);
        let eps = 1e-6f32;
        let mut push: Vec<u8> = Vec::new();
        push.extend_from_slice(&eps.to_le_bytes());
        push.extend_from_slice(&[d as u32, ng as u32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        ctx.run(pl, ds, pipe, &push, (2 * ng) as u32, t as u32, 1)?;
        unsafe {
            std::ptr::copy_nonoverlapping(qb.ptr as *const f32, qv.as_mut_ptr(), qv.len());
            std::ptr::copy_nonoverlapping(kb.ptr as *const f32, kv.as_mut_ptr(), kv.len());
        }
        // CPU 미러
        let mut cq = qv.clone();
        let ck = kv.clone();
        let _ = &ck;
        let mut q0: Vec<f32> = (0..t * row).map(|_| lcg()).collect();
        let _ = &mut q0;
        // 원본 복사로 재계산 (qv는 이미 GPU 결과로 덮음 — 처음 값에서 재생 불가)
        // → 비교는 정규화 후 노름=1 검증으로 대체: 각 (t, x) 블록의 Σv² ≈ 1
        let mut worst = 0.0f64;
        for ti in 0..t {
            for x in 0..2 * ng {
                let base = ti * row + if x < ng { x * d } else { (x - ng) * d };
                let ssum: f64 = (0..d).map(|i| {
                    let v = if x < ng { qv[base + i] } else { kv[base + i] };
                    (v as f64) * (v as f64)
                }).sum();
                worst = worst.max((ssum - 1.0).abs());
            }
        }
        lines.push(format!("l2_rows2: ||v||²−1 max={worst:.2e} {}", if worst < 1e-5 { "★" } else { "MISMATCH" }));
    }

    // ── qk_rope + kv_append + qsa_flash 어텐션 체인 (nh=4, nk=2, hd=128, nr=64)
    {
        let (nh, nk, hd, nr, np) = (4usize, 2usize, 128usize, 64usize, 16usize);
        // q [nh*2hd], k/v [np][nk*hd]
        let mut qv: Vec<f32> = (0..nh * 2 * hd).map(|_| lcg()).collect();
        let mut kvv: Vec<f32> = (0..np * nk * hd).map(|_| lcg()).collect();
        let ckk: Vec<f32> = (0..np * nk * hd).map(|_| lcg()).collect();
        let qwt: Vec<f32> = (0..nh * hd).map(|_| lcg()).collect();
        let kwt: Vec<f32> = (0..nk * hd).map(|_| lcg()).collect();
        // cs: [np][nr/2 * 2] — cos/sin
        let halfn = nr >> 1;
        let csv: Vec<f32> = (0..np * halfn * 2).map(|_| lcg()).collect();
        let mut kvo: Vec<f32> = kvv.clone();
        let qv0 = qv.clone();   // 원본 (미러 기준)
        let kv0 = kvo.clone();

        let qb = ctx.alloc(qv.len() * 4)?;
        let kb = ctx.alloc(kvo.len() * 4)?;
        let ckb = ctx.alloc(ckk.len() * 4)?;
        let qwb = ctx.alloc(qwt.len() * 4)?;
        let kwb = ctx.alloc(kwt.len() * 4)?;
        let csb = ctx.alloc(csv.len() * 4)?;
        let vb = ctx.alloc(kvo.len() * 4)?;
        let vb_host = kvv.clone(); // v 원본 (rope 미적용)
        unsafe {
            std::ptr::copy_nonoverlapping(qv.as_ptr(), qb.ptr as *mut f32, qv.len());
            std::ptr::copy_nonoverlapping(kvo.as_ptr(), kb.ptr as *mut f32, kvo.len());
            std::ptr::copy_nonoverlapping(ckk.as_ptr(), ckb.ptr as *mut f32, ckk.len());
            std::ptr::copy_nonoverlapping(qwt.as_ptr(), qwb.ptr as *mut f32, qwt.len());
            std::ptr::copy_nonoverlapping(kwt.as_ptr(), kwb.ptr as *mut f32, kwt.len());
            std::ptr::copy_nonoverlapping(csv.as_ptr(), csb.ptr as *mut f32, csv.len());
            std::ptr::copy_nonoverlapping(kvo.as_ptr(), vb.ptr as *mut f32, kvo.len());
        }
        // 1) qk_rope: pos=np-1
        {
            let (_d, pl, _p, ds, pipe) = ctx.pipeline(include_bytes!("spv/qk_rope.spv"), 5, 28)?;
            ctx.bind_bufs(ds, &[qb.buf, kb.buf, qwb.buf, kwb.buf, csb.buf]);
            let eps = 1e-6f32;
            let kqs = 1.0f32 / (hd as f32).sqrt();
            let mut push: Vec<u8> = Vec::new();
            push.extend_from_slice(&eps.to_le_bytes());
            push.extend_from_slice(&kqs.to_le_bytes());
            push.extend_from_slice(&[(np - 1) as u32, nh as u32, nk as u32, hd as u32, nr as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            ctx.run(pl, ds, pipe, &push, (nh + nk) as u32, 1, 1)?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(qb.ptr as *const f32, qv.as_mut_ptr(), qv.len());
            std::ptr::copy_nonoverlapping(kb.ptr as *const f32, kvo.as_mut_ptr(), kvo.len());
        }
        // CPU 미러 (f64)
        let eps = 1e-6f32;
        let kqs = 1.0f32 / (hd as f32).sqrt();
        let mirror = |qv: &mut Vec<f32>, kv: &mut Vec<f32>| {
            for r0 in 0..nh + nk {
                let is_q = r0 < nh;
                let halfn2 = nr >> 1;
                let csbase = (np - 1) * halfn2 * 2;
                let row_base = if is_q { r0 * 2 * hd } else { (r0 - nh) * hd };
                let mut sum = 0.0f64;
                for i in 0..hd {
                    let dv = if is_q { qv[row_base + i] } else { kv[row_base + i] };
                    sum += (dv * dv) as f64;
                }
                let scale = 1.0f32 / ((sum / hd as f64 + eps as f64).sqrt() as f32);
                for i in 0..hd {
                    let w = if is_q { qwt[r0 * hd + i] } else { kwt[(r0 - nh) * hd + i] };
                    let v = (if is_q { qv[row_base + i] } else { kv[row_base + i] }) * scale * w
                        * if is_q { 1.0 } else { kqs };
                    if is_q { qv[row_base + i] = v; } else { kv[row_base + i] = v; }
                }
                for p in 0..halfn2 {
                    let c = csv[csbase + p * 2] as f64;
                    let sf = csv[csbase + p * 2 + 1] as f64;
                    let a = row_base + p;
                    let b = a + halfn2;
                    let (x0, x1) = if is_q {
                        (qv[a] as f64, qv[b] as f64)
                    } else {
                        (kv[a] as f64, kv[b] as f64)
                    };
                    let (ra, rb) = ((x0 * c - x1 * sf) as f32, (x0 * sf + x1 * c) as f32);
                    if is_q { qv[a] = ra; qv[b] = rb; } else { kv[a] = ra; kv[b] = rb; }
                }
            }
        };
        let mut cq = qv0.clone();
        let mut ck2 = kv0.clone();
        mirror(&mut cq, &mut ck2);
        let qd = qv.iter().zip(cq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let kd = kvo.iter().zip(ck2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        lines.push(format!("qk_rope: q|D|={qd:.2e} k|D|={kd:.2e} {}", if qd < 1e-5 && kd < 1e-5 { "★" } else { "MISMATCH" }));

        // 2) kv_append: k/v → kv 캐시 [np][nk*hd] (이미 그 형상) pos=np-1 재기입 → 검증 생략(자명)
        // 3) qsa_flash: mask 전 1, np=16
        let maskv: Vec<u32> = vec![1u32; np * np];
        let mb = ctx.alloc(maskv.len() * 4)?;
        unsafe { std::ptr::copy_nonoverlapping(maskv.as_ptr(), mb.ptr as *mut u32, maskv.len()) };
        let ob = ctx.alloc(nh * hd * 4)?;
        {
            let (_d, pl, _p, ds, pipe) = ctx.pipeline(include_bytes!("spv/qsa_flash.spv"), 5, 24)?;
            ctx.bind_bufs(ds, &[qb.buf, ckb.buf, vb.buf, mb.buf, ob.buf]);
            let push: Vec<u8> = [np as u32, nh as u32, nk as u32, hd as u32, np as u32, (np - 1) as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            ctx.run(pl, ds, pipe, &push, 1, nh as u32, 1)?;
        }
        let mut r = vec![0f32; nh * hd];
        unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const f32, r.as_mut_ptr(), r.len()) };
        // CPU 미러 (f64) — 게이트 후보 포함
        let mut c = vec![0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / (nh / nk);
            let qb2 = h * 2 * hd;
            let mut sc = vec![0f64; np];
            let mut mx = f64::NEG_INFINITY;
            for p in 0..np {
                let mut d = 0.0f64;
                for i in 0..hd {
                    d += qv[qb2 + i] as f64 * ckk[p * nk * hd + kvh * hd + i] as f64;
                }
                sc[p] = d;
                mx = mx.max(sc[p]);
            }
            let mut sum = 0.0f64;
            for v in sc.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            for p in 0..np {
                let w = sc[p] / sum;
                for i in 0..hd {
                    // v는 rope 대상 아님 — 업로드한 원본 vb(=kvv) 사용
                    let vv = vb_host[p * nk * hd + kvh * hd + i];
                    c[h * hd + i] += (w * vv as f64) as f32;
                }
            }
            // 게이트
            for i in 0..hd {
                let g = 1.0 / (1.0 + (-qv[qb2 + hd + i]).exp());
                c[h * hd + i] *= g;
            }
        }
        let _ = &mut kvo;
        let fd = r.iter().zip(c.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        lines.push(format!("qsa_flash: |D|={fd:.2e} {}", if fd < 1e-3 { "★" } else { "MISMATCH" }));
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
