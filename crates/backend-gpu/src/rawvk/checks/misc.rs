//! 기타 체커 — ple_mt(plans/107 W4: checks.rs에서 분할).

use crate::rawvk::vkacc::{Slot, VkAcc, push_u32s};

/// vk-ple-mt-check (plans/94) — fn_ple_gate_mt·conv·res 3커널 체인 합성 검증.
/// 모델 파일 불필요(LCG 입력). 청크 2회(t=13→5, 디바이스 링 캐리 포함)를
/// 실전 그리드·푸시 상수로 발사해 CPU 미러(ple_block 산술)와 대조 + 지정 회수
/// 반복 비트 결정성 검사 — gate_mt 공유메모리 레이스(red[0] 소비-재사용) 검출용.
pub fn ple_mt_check(reps: usize) -> Result<String, String> {
    let acc = VkAcc::new()?;
    let n = 2560usize; // n_embd (FN 실값)
    let hc = 8usize;
    let (kern, dil) = (4usize, 3usize);
    let hist = (kern - 1) * dil; // 9
    let eps = 1e-5f32;
    let hc_dim = hc * n;
    let t1 = 13usize;
    let t2 = 5usize;
    let tt = t1 + t2;

    // ── 합성 입력 (LCG) ──
    let mut seed = 0x0094_feed_0000_0001u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let res0: Vec<f32> = (0..tt * hc_dim).map(|_| lcg()).collect();
    let key: Vec<f32> = (0..tt * hc_dim).map(|_| lcg()).collect();
    let value: Vec<f32> = (0..tt * n).map(|_| lcg()).collect();
    // norm 감마·conv_w는 [0.75,1.25) — 게이트 값 분포 확보
    let nk: Vec<f32> = (0..hc_dim).map(|_| 1.0 + lcg() * 0.5).collect();
    let nq: Vec<f32> = (0..hc_dim).map(|_| 1.0 + lcg() * 0.5).collect();
    let nc: Vec<f32> = (0..hc_dim).map(|_| 1.0 + lcg() * 0.5).collect();
    let cw: Vec<f32> = (0..hc_dim * kern).map(|_| 1.0 + lcg() * 0.5).collect();
    let rms = |v: &[f32]| -> f32 {
        let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        (1.0 / (s / n as f64 + eps as f64).sqrt()) as f32
    };
    let sig = |x: f32| 1.0f32 / (1.0 + (-x).exp());
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let mut res_m = res0.clone();
    let mut ring_m = vec![0f32; hist * hc_dim];
    let mut gate_m = vec![0f32; tt * hc];
    let mut gated_m = vec![0f32; tt * hc_dim];
    let mut conv_m = vec![0f32; tt * hc_dim];
    for (ci, &t) in [t1, t2].iter().enumerate() {
        let base = if ci == 0 { 0 } else { t1 };
        // gate + 방송 + grouped norm
        for ti in 0..t {
            let row = base + ti;
            for s in 0..hc {
                let (kb, rb) = (row * hc_dim + s * n, row * hc_dim + s * n);
                let kr = &key[kb..kb + n];
                let rr = &res_m[rb..rb + n];
                let (sk, sq) = (rms(kr), rms(rr));
                // CPU ple_block과 동일 결합: dot += (key·sk·nk)·(res·sq·nq) → /√n
                let mut dot = 0f32;
                for i in 0..n {
                    dot += (kr[i] * sk) * nk[s * n + i] * ((rr[i] * sq) * nq[s * n + i]);
                }
                let dot = dot / (n as f32).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                let g = sig(if dot >= 0.0 { mag } else { -mag });
                gate_m[row * hc + s] = g;
                let vg: Vec<f32> = (0..n).map(|i| value[row * n + i] * g).collect();
                let sg = rms(&vg);
                for i in 0..n {
                    gated_m[row * hc_dim + s * n + i] = vg[i] * sg * nc[s * n + i];
                }
            }
        }
        // dilated conv — V[j] = j<hist ? ring[j] : gated[base+j-hist] … 청크 로컬 ti
        for ti in 0..t {
            let row = base + ti;
            for c in 0..hc_dim {
                let mut acc = 0f32;
                for k in 0..kern {
                    let start = hist as i64 + ti as i64 - ((kern - 1 - k) * dil) as i64;
                    let v = if start >= hist as i64 {
                        gated_m[(base + (start - hist as i64) as usize) * hc_dim + c]
                    } else {
                        ring_m[start as usize * hc_dim + c]
                    };
                    acc += cw[c * kern + k] * v;
                }
                conv_m[row * hc_dim + c] = silu(acc);
            }
        }
        // 링 갱신: V = ring ++ gated(청크 t행) → 새 ring = V의 마지막 hist행
        let mut newring = vec![0f32; hist * hc_dim];
        for j in 0..hist {
            let src = t as i64 + j as i64;
            for c in 0..hc_dim {
                newring[j * hc_dim + c] = if src >= hist as i64 {
                    gated_m[(base + (src - hist as i64) as usize) * hc_dim + c]
                } else {
                    ring_m[src as usize * hc_dim + c]
                };
            }
        }
        ring_m = newring;
        // 잔차 — 청크1 완료 후 res_m이 갱신되므로 청크2 gate는 갱신값을 읽는다(GPU와 동일 순서)
        for ti in 0..t {
            let row = base + ti;
            for s in 0..hc {
                let g = gate_m[row * hc + s];
                for i in 0..n {
                    res_m[row * hc_dim + s * n + i] +=
                        value[row * n + i] * g + conv_m[row * hc_dim + s * n + i];
                }
            }
        }
    }

    // ── GPU 발사 — 청크별 버퍼(커널은 청크 로컬 ti 인덱싱), 링은 청크 간 캐리 ──
    let mut ctx = acc.ctx.lock();
    let up = |ctx: &mut crate::rawvk::context::VkCtx,
              v: &[f32]|
     -> Result<crate::rawvk::context::VkBuf, String> {
        let b = ctx.alloc_host(v.len() * 4)?;
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, b.ptr, v.len() * 4) };
        Ok(b)
    };
    let kb1 = up(&mut ctx, &key[..t1 * hc_dim])?;
    let vb1 = up(&mut ctx, &value[..t1 * n])?;
    let kb2 = up(&mut ctx, &key[t1 * hc_dim..])?;
    let vb2 = up(&mut ctx, &value[t1 * n..])?;
    let (nkb, nqb, ncb, cwb) = (
        up(&mut ctx, &nk)?,
        up(&mut ctx, &nq)?,
        up(&mut ctx, &nc)?,
        up(&mut ctx, &cw)?,
    );
    let res_b1 = ctx.alloc_host(t1 * hc_dim * 4)?;
    let res_b2 = ctx.alloc_host(t2 * hc_dim * 4)?;
    let gated_b1 = ctx.alloc_host(t1 * hc_dim * 4)?;
    let gated_b2 = ctx.alloc_host(t2 * hc_dim * 4)?;
    let gob1 = ctx.alloc_host(t1 * hc * 4)?;
    let gob2 = ctx.alloc_host(t2 * hc * 4)?;
    let cob1 = ctx.alloc_host(t1 * hc_dim * 4)?;
    let cob2 = ctx.alloc_host(t2 * hc_dim * 4)?;
    let ring_b = ctx.alloc_host(hist * hc_dim * 4)?;
    let p_gate = acc.pipeline(&mut ctx, Slot::FnPleGateMt)?;
    let p_conv = acc.pipeline(&mut ctx, Slot::FnPleConv)?;
    let p_res = acc.pipeline(&mut ctx, Slot::FnPleRes)?;
    drop(ctx);

    let zero = vec![0f32; hist * hc_dim];
    let mut first: Option<Vec<Vec<f32>>> = None;
    let mut mismatch_reps = 0usize;
    let mut rel_max_overall = 0f64;
    let mut stages_out: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
    for rep in 0..reps.max(1) {
        let mut ctx = acc.ctx.lock();
        // 상태 리셋: res0·링 0 (key/value/norm/conv_w는 불변)
        unsafe {
            std::ptr::copy_nonoverlapping(res0.as_ptr() as *const u8, res_b1.ptr, t1 * hc_dim * 4);
            std::ptr::copy_nonoverlapping(
                res0[t1 * hc_dim..].as_ptr() as *const u8,
                res_b2.ptr,
                t2 * hc_dim * 4,
            );
            std::ptr::copy_nonoverlapping(
                zero.as_ptr() as *const u8,
                ring_b.ptr,
                hist * hc_dim * 4,
            );
        }
        for (base, t, kb, vb, rb, gb, gob, cob) in [
            (0usize, t1, &kb1, &vb1, &res_b1, &gated_b1, &gob1, &cob1),
            (t1, t2, &kb2, &vb2, &res_b2, &gated_b2, &gob2, &cob2),
        ] {
            let _ = base;
            let ds = ctx.bind_ds(
                &p_gate,
                &[
                    rb.buf, kb.buf, vb.buf, nkb.buf, nqb.buf, ncb.buf, gb.buf, gob.buf,
                ],
            )?;
            let mut push = eps.to_le_bytes().to_vec();
            push.extend_from_slice(&push_u32s(&[n as u32, hc as u32, t as u32]));
            ctx.run(p_gate.pl, ds, p_gate.pipe, &push, hc as u32, t as u32, 1)?;
            let ds2 = ctx.bind_ds(&p_conv, &[gb.buf, cwb.buf, ring_b.buf, cob.buf])?;
            let push2 = push_u32s(&[
                hc_dim as u32,
                t as u32,
                kern as u32,
                dil as u32,
                hist as u32,
            ]);
            ctx.run(
                p_conv.pl,
                ds2,
                p_conv.pipe,
                &push2,
                hc_dim.div_ceil(256) as u32,
                1,
                1,
            )?;
            let ds3 = ctx.bind_ds(&p_res, &[rb.buf, vb.buf, gob.buf, cob.buf])?;
            let push3 = push_u32s(&[n as u32, hc as u32, t as u32]);
            ctx.run(
                p_res.pl,
                ds3,
                p_res.pipe,
                &push3,
                n.div_ceil(256) as u32,
                1,
                1,
            )?;
        }
        // 스테이지 산출물 스냅샷(마지막 rep) — 국소화용.
        if rep == reps - 1 || (reps == 1 && rep == 0) {
            let stage = |p: *mut u8, k: usize| -> Vec<f32> {
                let mut v = vec![0f32; k];
                unsafe { std::ptr::copy_nonoverlapping(p as *const f32, v.as_mut_ptr(), k) };
                v
            };
            let mut g1 = stage(gated_b1.ptr, t1 * hc_dim);
            let g2 = stage(gated_b2.ptr, t2 * hc_dim);
            g1.extend_from_slice(&g2);
            let mut go1 = stage(gob1.ptr, t1 * hc);
            let go2 = stage(gob2.ptr, t2 * hc);
            go1.extend_from_slice(&go2);
            let mut c1 = stage(cob1.ptr, t1 * hc_dim);
            let c2 = stage(cob2.ptr, t2 * hc_dim);
            c1.extend_from_slice(&c2);
            stages_out = Some((g1, go1, c1));
        }
        let mut snap = vec![vec![0f32; tt * hc_dim + hist * hc_dim]; 2];
        unsafe {
            std::ptr::copy_nonoverlapping(
                res_b1.ptr as *const f32,
                snap[0].as_mut_ptr(),
                t1 * hc_dim,
            );
            std::ptr::copy_nonoverlapping(
                res_b2.ptr as *const f32,
                snap[0].as_mut_ptr().add(t1 * hc_dim),
                t2 * hc_dim,
            );
            std::ptr::copy_nonoverlapping(
                ring_b.ptr as *const f32,
                snap[1].as_mut_ptr(),
                hist * hc_dim,
            );
        }
        drop(ctx);
        if let Some(f0) = &first {
            if snap.iter().zip(f0.iter()).any(|(a, b)| a != b) {
                mismatch_reps += 1;
            }
        } else {
            first = Some(snap.clone());
        }
        if rep == 0 || rep == reps - 1 {
            let mut rel_max = 0f64;
            for (a, b) in snap[0]
                .iter()
                .zip(res_m.iter())
                .chain(snap[1].iter().zip(ring_m.iter()))
            {
                let rel = (*a as f64 - *b as f64).abs() / (b.abs() as f64 + 1e-6);
                rel_max = rel_max.max(rel);
            }
            rel_max_overall = rel_max_overall.max(rel_max);
        }
    }
    let det = if mismatch_reps == 0 {
        "결정성 ✓".to_string()
    } else {
        format!("비결정 {mismatch_reps}/{} rep", reps.max(1) - 1)
    };
    // 국소화 보고: |a−b| > 1e-4·(1+|b|) 인 "실질 오류" 원소 수 + 최대 절대오류.
    // rel_max는 영근접 원소 부풀림이 있어 참고치로만 출력한다.
    let mut loc = String::new();
    let mut sig_total = 0usize;
    if let Some((g, go, c)) = stages_out {
        let stat = |name: &str, a: &[f32], b: &[f32], loc: &mut String| -> usize {
            let (mut sig, mut mx) = (0usize, 0f64);
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x as f64 - *y as f64).abs();
                if d > 1e-4 * (1.0 + y.abs() as f64) {
                    sig += 1;
                }
                mx = mx.max(d);
            }
            loc.push_str(&format!(" {name}:{sig}/{}(max|D|={mx:.1e})", a.len()));
            sig
        };
        sig_total += stat("gated", &g, &gated_m, &mut loc);
        sig_total += stat("gate", &go, &gate_m, &mut loc);
        sig_total += stat("conv", &c, &conv_m, &mut loc);
        let f0 = first.as_ref().unwrap();
        sig_total += stat("res", &f0[0], &res_m, &mut loc);
        sig_total += stat("ring", &f0[1], &ring_m, &mut loc);
    }
    let ok = mismatch_reps == 0 && sig_total == 0;
    Ok(format!(
        "ple-mt: {} rel_max={:.2e} (t={t1}→{t2}, 링 캐리, reps={}) — {}|{loc}",
        if ok { "★" } else { "✗" },
        rel_max_overall,
        reps.max(1),
        det
    ))
}
