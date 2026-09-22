//! probes/attn — 어텐션 커널 검증 (probes.rs에서 이동, plans/78 R3).

use super::*;

/// qk_norm_rope 단독 검증 — 디코드와 동일 파라미터.
pub fn qk_check() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n_head = 24usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let n_rot = 64usize;
    let half = n_rot / 2;
    let pos = 2usize;
    let rows = n_head + n_kv;
    let aq = ctx.alloc(n_head * 2 * hd * 4)?;
    let ak = ctx.alloc(n_kv * hd * 4)?;
    let qw: Vec<f32> = (0..n_head * hd).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let kw: Vec<f32> = (0..n_kv * hd).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect();
    let qw_d = ctx.alloc(qw.len() * 4)?;
    let kw_d = ctx.alloc(kw.len() * 4)?;
    ctx.h2d(qw_d, bytemuck::cast_slice(&qw))?;
    ctx.h2d(kw_d, bytemuck::cast_slice(&kw))?;
    let hq: Vec<f32> = (0..n_head * 2 * hd).map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.1).collect();
    let hk: Vec<f32> = (0..n_kv * hd).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.1).collect();
    ctx.h2d(aq, bytemuck::cast_slice(&hq))?;
    ctx.h2d(ak, bytemuck::cast_slice(&hk))?;
    let cs: Vec<f32> = (0..2048 * half * 2).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let cs_d = ctx.alloc(cs.len() * 4)?;
    ctx.h2d(cs_d, bytemuck::cast_slice(&cs))?;
    let eps = 1e-5f32;
    let kqs = 0.05f32;
    let mut qp = aq as *mut std::ffi::c_void;
    let mut kp = ak as *mut std::ffi::c_void;
    let mut qwp = qw_d as *mut std::ffi::c_void;
    let mut kwp = kw_d as *mut std::ffi::c_void;
    let mut csp = cs_d as *mut std::ffi::c_void;
    let mut e = eps;
    let mut k = kqs;
    let mut posv = pos as i32;
    let mut nh = n_head as i32;
    let mut nk = n_kv as i32;
    let mut h = hd as i32;
    let mut nr = n_rot as i32;
    fn pp<T>(v: &mut T) -> *mut std::ffi::c_void {
        v as *mut T as *mut std::ffi::c_void
    }
    let mut args = vec![
        pp(&mut qp), pp(&mut kp), pp(&mut qwp), pp(&mut kwp), pp(&mut csp), pp(&mut e),
        pp(&mut k), pp(&mut posv), pp(&mut nh), pp(&mut nk), pp(&mut h), pp(&mut nr),
    ];
    ctx.launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
    ctx.sync()?;
    let mut oq = vec![0f32; n_head * 2 * hd];
    ctx.d2h(bytemuck::cast_slice_mut(&mut oq).as_mut(), aq)?;
    Ok(format!("qk_check ok: q[0]={:.4} q[511]={:.4} q[512]={:.4} k[0]={:.4}", oq[0], oq[511], oq[512], {
        let mut ok = vec![0f32; n_kv * hd];
        ctx.d2h(bytemuck::cast_slice_mut(&mut ok).as_mut(), ak)?;
        ok[0]
    }))
}

/// 합성 어텐션 검증: qsa_flash_wmma 를 작은 단일 케이스로 돌려 **CPU 기준**과 비교한다.
/// 디코드(t=1) GQA 어텐션 v2 검증·계측: 기존 qsa_flash_gqa 와 출력을 대조하고
/// n_past 별로 두 커널의 런치 시간을 잰다. 모델 구성(n_head=24, n_kv=4, hd=256)을 쓴다.
#[allow(unused_assignments)] // kp/vp 는 런치 인자로 넘긴 **주소**가 읽는 값 (raw 포인터 경유)
pub fn gqa_bench() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (24usize, 4usize, 256usize);
    let sstride = 4096usize;
    let seg = 32usize;
    let n_max = 3314usize;
    let qv: Vec<f32> = (0..n_head * 2 * hd)
        .map(|i| (((i * 1103515245 + 12345) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let kk: Vec<f32> = (0..n_max * n_kv * hd)
        .map(|i| (((i * 214013 + 2531011) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let vv: Vec<f32> = (0..n_max * n_kv * hd)
        .map(|i| (((i * 1260231 + 999983) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let mask: Vec<u32> = vec![1u32; sstride];
    let qd = ctx.alloc(qv.len() * 4)?;
    let kd = ctx.alloc(kk.len() * 4)?;
    let vd = ctx.alloc(vv.len() * 4)?;
    let md = ctx.alloc(mask.len() * 4)?;
    let nseg_max = n_max.div_ceil(seg);
    let npart = n_head * nseg_max * (hd + 2);
    let p1d = ctx.alloc(npart * 4)?;
    let p2d = ctx.alloc(npart * 4)?;
    ctx.h2d(qd, bytemuck::cast_slice(&qv))?;
    ctx.h2d(kd, bytemuck::cast_slice(&kk))?;
    ctx.h2d(vd, bytemuck::cast_slice(&vv))?;
    ctx.h2d(md, bytemuck::cast_slice(&mask))?;
    let mut out = String::new();
    let (mut us1, mut us2) = (0f64, 0f64);
    for n_past in [512usize, 1024, 2048, 3314] {
        let nseg = n_past.div_ceil(seg);
    let p3d = ctx.alloc(npart * 4)?;
    let p4d = ctx.alloc(npart * 4)?;
    let k16 = ctx.alloc(kk.len() * 2)?;
    let v16 = ctx.alloc(vv.len() * 2)?;
    {
        let mut a: Vec<*mut c_void> = Vec::new();
        let mut sp = kd as *mut c_void;
        let mut dp = k16 as *mut c_void;
        let mut nn = kk.len() as i32;
        a.push(&mut sp as *mut _ as *mut c_void);
        a.push(&mut dp as *mut _ as *mut c_void);
        a.push(&mut nn as *mut _ as *mut c_void);
        let nblk = kk.len().div_ceil(1024) as u32;
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a)?;
        let mut sp2 = vd as *mut c_void;
        let mut dp2 = v16 as *mut c_void;
        let mut a2: Vec<*mut c_void> = Vec::new();
        a2.push(&mut sp2 as *mut _ as *mut c_void);
        a2.push(&mut dp2 as *mut _ as *mut c_void);
        a2.push(&mut nn as *mut _ as *mut c_void);
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a2)?;
        ctx.sync()?;
        // 변환 검증: 앞 8개 half 를 되읽어 f32 원본과 비교
        let mut hb = vec![0u16; 8];
        ctx.d2h(bytemuck::cast_slice_mut(&mut hb).as_mut(), k16)?;
        let f0: Vec<f32> = (0..8)
            .map(|i| {
                let h = hb[i] as u16;
                let s = (h >> 15) & 1;
                let e = (h >> 10) & 0x1f;
                let m = h & 0x3ff;
                let v = if e == 0 {
                    (m as f32) * 2f32.powi(-24)
                } else {
                    (1.0 + (m as f32) / 1024.0) * 2f32.powi(e as i32 - 15)
                };
                if s == 1 { -v } else { v }
            })
            .collect();
        eprintln!("# kv_f16 앞 8개: f16={:?}", f0.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
        eprintln!("# kv_f16 원본  : {:?}", kk.iter().take(8).map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    }
        for (lab, pd) in [("v1", p1d), ("v2", p2d), ("v2h", p3d), ("v2d", p4d)] {
            let mut qp = qd as *mut c_void;
            let mut kp = kd as *mut c_void;
            let mut vp = vd as *mut c_void;
            let mut mp = md as *mut c_void;
            let mut pp = pd as *mut c_void;
            let mut np_ = n_past as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut tl = 1i32;
            let mut ss = sstride as i32;
            let mut p0 = 0i32;
            let mut sg = seg as i32;
            let mut args: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void, &mut mp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void, &mut np_ as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void, &mut nk as *mut _ as *mut c_void,
                &mut h as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
                &mut ss as *mut _ as *mut c_void, &mut p0 as *mut _ as *mut c_void,
                &mut sg as *mut _ as *mut c_void,
            ];
            let name = match lab {
                "v1" => "qsa_flash_gqa",
                "v2" => "qsa_flash_gqa2",
                "v2h" => "qsa_flash_gqa2h",
                _ => "qsa_flash_gqa2d",
            };
            if lab == "v2h" || lab == "v2d" {
                // f16 KV 를 읽는 판: ck/cv 자리에 f16 버퍼를 넘긴다(q 는 그대로 f32)
                kp = k16 as *mut c_void;
                vp = v16 as *mut c_void;
            }
            for _ in 0..20 { let _ = ctx.launch3(name, 1, n_kv as u32, nseg as u32, 256, &mut args); }
            ctx.sync()?;
            let iters = 200usize;
            let t0 = std::time::Instant::now();
            for _ in 0..iters { let _ = ctx.launch3(name, 1, n_kv as u32, nseg as u32, 256, &mut args); }
            ctx.sync()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            if lab == "v1" {
                us1 = us;
            } else if lab == "v2h" {
                let mut b = vec![0f32; npart];
                let mut c = vec![0f32; npart];
                ctx.d2h(bytemuck::cast_slice_mut(&mut b).as_mut(), p2d)?;
                ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), p3d)?;
                let cmp_len = n_head * nseg * (hd + 2);
                let mut worst = 0f32;
                let mut bad = 0usize;
                for i in 0..cmp_len {
                    let d = (b[i] - c[i]).abs();
                    let rel = d / (1.0f32 + b[i].abs());
                    if rel > worst { worst = rel; }
                    if rel > 1e-3 { bad += 1; }
                }
                out.push_str(&format!(
                    "n_past={n_past:5}  v1 {us1:8.2}us  v2 {us2:8.2}us  v2h {us:8.2}us  v2h/v2={:.2}x  v2h vs v2 최대상대차 {worst:.2e} (>1e-3 {bad})\n",
                    us2 / us));
            } else if lab == "v2d" {
                let mut b = vec![0f32; npart];
                let mut c = vec![0f32; npart];
                ctx.d2h(bytemuck::cast_slice_mut(&mut b).as_mut(), p2d)?;
                ctx.d2h(bytemuck::cast_slice_mut(&mut c).as_mut(), p4d)?;
                let cmp_len = n_head * nseg * (hd + 2);
                let mut worst = 0f32;
                let mut bad = 0usize;
                for i in 0..cmp_len {
                    let d = (b[i] - c[i]).abs();
                    let rel = d / (1.0f32 + b[i].abs());
                    if rel > worst { worst = rel; }
                    if rel > 1e-3 { bad += 1; }
                }
                out.push_str(&format!(
                    "n_past={n_past:5}  v2 {us2:8.2}us  v2d {us:8.2}us  v2d/v2={:.2}x  vs v2 최대상대차 {worst:.2e} (>1e-3 {bad})\n",
                    us2 / us));
            } else {
                us2 = us;
                let mut a = vec![0f32; npart];
                let mut b = vec![0f32; npart];
                ctx.d2h(bytemuck::cast_slice_mut(&mut a).as_mut(), p1d)?;
                ctx.d2h(bytemuck::cast_slice_mut(&mut b).as_mut(), p2d)?;
                let mut worst = 0f32;
                let mut bad = 0usize;
                let cmp_len = n_head * nseg * (hd + 2);   // 기록된 구간만 비교
                for i in 0..cmp_len {
                    let d = (a[i] - b[i]).abs();
                    let rel = d / (1.0f32 + a[i].abs());
                    if rel > worst { worst = rel; }
                    if rel > 1e-4 { bad += 1; }
                }
                out.push_str(&format!(
                    "n_past={n_past:5}  v1 {us1:8.2}us  v2 {us2:8.2}us  v1/v2={:.2}x  최대상대차 {worst:.2e}  불일치 {bad}\n",
                    us1 / us2));
            }
        }
    }
    Ok(out)
}


/// plans/74 N4: qsa_flash_wmma2 검증 — 전체 인과 어텐션 CPU 기준 + 호스트 merge.
pub fn wmma2_attn_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (24usize, 8usize, 256usize);
    let mini = std::env::var_os("LLM170_WMMA2_MINI").is_some();
    let prod = std::env::var_os("LLM170_WMMA2_PROD").is_some();
    let (t, pos0, seg, sstride, n_past) = if mini {
        (16usize, 0usize, 16usize, 256usize, 16usize)
    } else if prod {
        (512usize, 1024usize, 1024usize, 2048usize, 1536usize)
    } else {
        (64usize, 32usize, 16usize, 256usize, 96usize)
    };
    let nseg = n_past.div_ceil(seg);
    let qv: Vec<f32> = (0..t * n_head * 2 * hd)
        .map(|i| (((i * 1103515245 + 12345) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let kk: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 214013 + 2531011) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let vv: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 1260231 + 999983) % 200) as f32 - 100.0) * 5e-3)
        .collect();
    let mut mask: Vec<u32> = vec![0u32; (pos0 + t) * sstride];
    for r in 0..(pos0 + t) {
        for k in 0..=r.min(n_past - 1) { mask[r * sstride + k] = 1; }
    }
    let qd = ctx.alloc(qv.len() * 4)?;
    let kd = ctx.alloc(kk.len() * 4)?;
    let vd = ctx.alloc(vv.len() * 4)?;
    let md = ctx.alloc(mask.len() * 4)?;
    let pd = ctx.alloc(t * n_head * nseg * (hd + 2) * 4)?;
    ctx.h2d(qd, bytemuck::cast_slice(&qv))?;
    ctx.h2d(kd, bytemuck::cast_slice(&kk))?;
    ctx.h2d(vd, bytemuck::cast_slice(&vv))?;
    ctx.h2d(md, bytemuck::cast_slice(&mask))?;
    let kh = ctx.alloc(kk.len() * 2)?;
    let vh = ctx.alloc(vv.len() * 2)?;
    for (src, dst) in [(kd, kh), (vd, vh)] {
        let mut sp = src as *mut c_void;
        let mut dp = dst as *mut c_void;
        let mut nn = kk.len() as i32;
        let mut a: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let nblk = kk.len().div_ceil(1024) as u32;
        ctx.launch3("kv_f16", nblk, 1, 1, 256, &mut a)?;
    }
    let mut qp = qd as *mut c_void;
    let mut kp = kh as *mut c_void;
    let mut vp = vh as *mut c_void;
    let mut mp = md as *mut c_void;
    let mut pp = pd as *mut c_void;
    let mut np_ = n_past as i32;
    let mut nh = n_head as i32;
    let mut nk = n_kv as i32;
    let mut h = hd as i32;
    let mut tl = t as i32;
    let mut ss = sstride as i32;
    let mut p0 = pos0 as i32;
    let mut sg = seg as i32;
    let mut args = vec![
        (&mut qp) as *mut _ as *mut c_void, (&mut kp) as *mut _ as *mut c_void,
        (&mut vp) as *mut _ as *mut c_void, (&mut mp) as *mut _ as *mut c_void,
        (&mut pp) as *mut _ as *mut c_void, (&mut np_) as *mut _ as *mut c_void,
        (&mut nh) as *mut _ as *mut c_void, (&mut nk) as *mut _ as *mut c_void,
        (&mut h) as *mut _ as *mut c_void, (&mut tl) as *mut _ as *mut c_void,
        (&mut ss) as *mut _ as *mut c_void, (&mut p0) as *mut _ as *mut c_void,
        (&mut sg) as *mut _ as *mut c_void,
    ];
    // v1(qsa_flash_wmma2)은 종결 제거(90 A3) — 프로덕션 체인 v2만 검증.
    ctx.launch3("qsa_flash_wmma2v2", t.div_ceil(16) as u32, n_head as u32, nseg as u32, 64, &mut args)?;
    ctx.sync()?;
    let mut got = vec![0f32; t * n_head * nseg * (hd + 2)];
    ctx.d2h(bytemuck::cast_slice_mut(&mut got).as_mut(), pd)?;
    // 검증: 세그별 부분 → 호스트 merge → CPU 전체 어텐션(+gate) 대조.
    let mut maxerr = 0f32;
    let mut first = String::new();
    for row in 0..t {
        for hh in 0..n_head {
            let kvh = hh / (n_head / n_kv);
            let mut m = f32::NEG_INFINITY;
            let mut s2v: Vec<f32> = vec![];
            let mut ks: Vec<usize> = vec![];
            for k in 0..n_past {
                if mask[(pos0 + row) * sstride + k] == 0 { continue; }
                let mut s2 = 0f32;
                for d in 0..hd {
                    s2 += qv[(row * n_head + hh) * 2 * hd + d] * kk[(k * n_kv + kvh) * hd + d];
                }
                s2v.push(s2);
                ks.push(k);
                m = m.max(s2);
            }
            let mut ssum = 0f32;
            let mut acc = vec![0f32; hd];
            for (i, &s2) in s2v.iter().enumerate() {
                let e = (s2 - m).exp();
                ssum += e;
                let k = ks[i];
                for d in 0..hd { acc[d] += e * vv[(k * n_kv + kvh) * hd + d]; }
            }
            // 호스트 merge
            let mut m_all = f32::NEG_INFINITY;
            for sgi in 0..nseg {
                let b = ((row * n_head + hh) * nseg + sgi) * (hd + 2);
                m_all = m_all.max(got[b + hd]);
            }
            let mut num = vec![0f32; hd];
            let mut den = 0f32;
            for sgi in 0..nseg {
                let b = ((row * n_head + hh) * nseg + sgi) * (hd + 2);
                let w = (got[b + hd] - m_all).exp();
                den += got[b + hd + 1] * w;
                for d in 0..hd { num[d] += got[b + d] * w; }
            }
            let gate = 1.0f32 / (1.0f32 + (-(qv[(row * n_head + hh) * 2 * hd + hd])).exp());
            for d in 0..hd {
                let refv = if ssum > 0.0 { acc[d] / ssum * gate } else { 0.0 };
                let ourv = if den > 0.0 { num[d] / den * gate } else { 0.0 };
                let dv = (ourv - refv).abs();
                if dv > maxerr { maxerr = dv; }
                if dv > 0.02 && first.is_empty() {
                    first = format!("행{row} 헤드{hh} d{d}: ours {ourv:.5} ref {refv:.5}");
                }
            }
        }
    }
    let mut dbg = String::new();
    if mini {
        let b = 0usize;
        dbg += &format!(" | part0: m={:.4} s={:.4} vk[..4]={:?}", got[b + hd], got[b + hd + 1], &got[b..b + 4]);
    }
    Ok(format!("wmma2-attn-check: max|Δ|={maxerr:.5} {}{}", if maxerr <= 0.02 { "★ PASS".to_string() } else { format!("FAIL {first}") }, dbg))
}

/// 두 prefill 어텐션 커널(wk16 vs wk8)에 **동일한** Q/K/V/마스크를 넣고 part 버퍼를 비교한다.
/// 같은 입력에서 part 가 갈리면 커널 버그, 일치하면(또는 반올림 수준이면) 긴 문맥 발산은
/// 재귀 층을 통한 증폭이다. plans/47 의 판별 하네스.
pub fn attn_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_head, n_kv, hd) = (24usize, 4usize, 256usize);
    let (t, pos0, seg, sstride, ctx_len) = (512usize, 1536usize, 128usize, 2048usize, 2048usize);
    let n_past = pos0 + t;
    let nseg = n_past.div_ceil(seg);
    // 입력: 결정적 의사난수(양 커널에 동일)
    let qv: Vec<f32> = (0..t * n_head * 2 * hd)
        .map(|i| (((i * 1103515245 + 12345) % 2000) as f32 - 1000.0) * 1e-3)
        .collect();
    let kv_k: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 214013 + 2531011) % 2000) as f32 - 1000.0) * 1e-3)
        .collect();
    let kv_v: Vec<f32> = (0..n_past * n_kv * hd)
        .map(|i| (((i * 1260231 + 999983) % 2000) as f32 - 1000.0) * 1e-3)
        .collect();
    let mut mask: Vec<u32> = vec![0u32; (pos0 + t) * sstride];
    for r in 0..(pos0 + t) {
        for k in 0..=r.min(ctx_len - 1) {
            mask[r * sstride + k] = 1;
        }
    }
    let qd = ctx.alloc(qv.len() * 4)?;
    let kd = ctx.alloc(kv_k.len() * 4)?;
    let vd = ctx.alloc(kv_v.len() * 4)?;
    let md = ctx.alloc(mask.len() * 4)?;
    let pd = ctx.alloc(t * n_head * nseg * (hd + 2) * 4)?;
    ctx.h2d(qd, bytemuck::cast_slice(&qv))?;
    ctx.h2d(kd, bytemuck::cast_slice(&kv_k))?;
    ctx.h2d(vd, bytemuck::cast_slice(&kv_v))?;
    ctx.h2d(md, bytemuck::cast_slice(&mask))?;
    let mut out = String::new();
    for (name, gx) in [("qsa_flash_wk16", t.div_ceil(16)), ("qsa_flash_wk8", t.div_ceil(32))] {
        let mut qp = qd as *mut c_void;
        let mut kp = kd as *mut c_void;
        let mut vp = vd as *mut c_void;
        let mut mp = md as *mut c_void;
        let mut pp = pd as *mut c_void;
        let mut np_ = n_past as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tl = t as i32;
        let mut ss = sstride as i32;
        let mut p0 = pos0 as i32;
        let mut sg = seg as i32;
        let mut args = vec![
            (&mut qp) as *mut _ as *mut c_void, (&mut kp) as *mut _ as *mut c_void,
            (&mut vp) as *mut _ as *mut c_void, (&mut mp) as *mut _ as *mut c_void,
            (&mut pp) as *mut _ as *mut c_void, (&mut np_) as *mut _ as *mut c_void,
            (&mut nh) as *mut _ as *mut c_void, (&mut nk) as *mut _ as *mut c_void,
            (&mut h) as *mut _ as *mut c_void, (&mut tl) as *mut _ as *mut c_void,
            (&mut ss) as *mut _ as *mut c_void, (&mut p0) as *mut _ as *mut c_void,
            (&mut sg) as *mut _ as *mut c_void,
        ];
        ctx.launch3(name, gx as u32, n_head as u32, nseg as u32, 256, &mut args)?;
        ctx.sync()?;
        let mut v = vec![0f32; t * n_head * nseg * (hd + 2)];
        ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), pd)?;
        if name == "qsa_flash_wk16" {
            std::fs::write("/tmp/attn_wk16.f32", bytemuck::cast_slice(&v)).map_err(|e| e.to_string())?;
        } else {
            std::fs::write("/tmp/attn_wk8.f32", bytemuck::cast_slice(&v)).map_err(|e| e.to_string())?;
        }
        out += &format!("{name}: {gx} blocks, {nseg} segs\n");
    }
    // 비교
    let a = std::fs::read("/tmp/attn_wk16.f32").map_err(|e| e.to_string())?;
    let b = std::fs::read("/tmp/attn_wk8.f32").map_err(|e| e.to_string())?;
    let a: &[f32] = bytemuck::cast_slice(&a);
    let b: &[f32] = bytemuck::cast_slice(&b);
    let mut maxd = 0f32;
    let mut nbad = 0usize;
    let pr = hd + 2;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > maxd { maxd = d; }
        if (i % pr) < hd && d > 1e-4 { nbad += 1; }
    }
    out += &format!("max|delta| = {maxd:.6}, acc 원소(>1e-4) 불일치 {nbad} / {}\n", a.len() / pr * hd);
    Ok(out)
}
