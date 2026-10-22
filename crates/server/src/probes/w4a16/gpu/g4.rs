use crate::gpu_engine::g4_step;

pub(super) fn run_attn_g4_check(dec: &mut llm170_backend_gpu::W4a16Dec) -> Result<String, String> {
    use llm170_backend_gpu::AttnDims;
    let configs: [(
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        f32,
        bool,
        usize,
    ); 2] = [
        (
            "슬라이딩(d=256 창 sw=8)",
            256,
            4,
            2,
            128,
            128,
            256,
            1e4,
            true,
            8,
        ),
        ("풀(d=512 전역)", 512, 4, 1, 64, 256, 512, 1e6, true, 0),
    ];
    let (t_len, pos0, splits) = (16usize, 0u32, 2usize);
    let mut out = String::from("attn_g4 통합(prep→part→merge) — CPU 미러 대조\n");
    let mut ok_all = true;
    for (name, d, q_heads, kv_heads, pairs, span, denom, base, vn, sw) in configs {
        let dims = AttnDims {
            n_attn: 1,
            q_heads,
            kv_heads,
            d,
            cap: 64,
            interval: 1,
            sliding_window: sw,
        };
        let layer = llm170_backend_gpu::AttnLayer {
            kv_heads,
            d,
            sliding: sw > 0,
            k_eq_v: false,
            kq_scale: 1.0,
        };
        dec.set_attn(dims, vec![layer], &vec![1.0f32; d], &vec![1.0f32; d])
            .map_err(|e| format!("set_attn: {e}"))?;
        dec.set_g4_rope(&[
            d as f32,
            pairs as f32,
            span as f32,
            denom as f32,
            base,
            if vn { 1.0 } else { 0.0 },
            0.0,
        ])
        .map_err(|e| format!("set_g4_rope: {e}"))?;
        dec.kv_grow(0, t_len).map_err(|e| format!("kv_grow: {e}"))?;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let pm1 = |r: u64| ((r >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        let q: Vec<f32> = (0..t_len * q_heads * d).map(|_| pm1(rnd())).collect();
        let kin: Vec<f32> = (0..t_len * kv_heads * d).map(|_| pm1(rnd())).collect();
        let vin: Vec<f32> = (0..t_len * kv_heads * d).map(|_| pm1(rnd())).collect();
        let qnw: Vec<f32> = (0..d).map(|_| 0.9 + pm1(rnd()) * 0.1).collect();
        let knw: Vec<f32> = (0..d).map(|_| 0.9 + pm1(rnd()) * 0.1).collect();
        let kq_scale = 1.0f32;
        let got = dec
            .attn_g4_host(
                0, &q, &kin, &vin, &qnw, &knw, 0, t_len, q_heads, kv_heads, d, sw, kq_scale,
                splits, pos0,
            )
            .map_err(|e| format!("attn_g4({name}): {e}"))?;
        let rope = |h: &mut [f32], pos: u32| {
            for p in 0..pairs {
                let theta = base.powf(-(2.0 * p as f32) / denom as f32);
                let ang = pos as f32 * theta;
                let (c, sn) = (ang.cos(), ang.sin());
                let (x0, x1) = (h[p] as f64, h[p + span] as f64);
                let (cf, sf) = (c as f64, sn as f64);
                h[p] = (x0 * cf - x1 * sf) as f32;
                h[p + span] = (x0 * sf + x1 * cf) as f32;
            }
        };
        let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let ss: f32 = x.iter().map(|v| v * v).sum();
            let inv = 1.0f32 / (ss / x.len() as f32 + 1e-6).sqrt();
            match w {
                Some(w) => x.iter().zip(w.iter()).map(|(v, g)| v * inv * g).collect(),
                None => x.iter().map(|v| v * inv).collect(),
            }
        };
        let mut qh = vec![0f32; t_len * q_heads * d];
        for t in 0..t_len {
            for h in 0..q_heads {
                let src = &q[(t * q_heads + h) * d..(t * q_heads + h) * d + d];
                let mut hd = rms(src, Some(&qnw));
                rope(&mut hd, pos0 + t as u32);
                qh[(t * q_heads + h) * d..(t * q_heads + h) * d + d].copy_from_slice(&hd);
            }
        }
        let quant = |hd: &[f32]| -> (Vec<i8>, f32) {
            let amax = hd.iter().fold(0f32, |m, v| m.max(v.abs()));
            let mut ks = amax * (1.0f32 / 127.0);
            if ks == 0.0 {
                ks = 1.0;
            }
            let iv = 1.0 / ks;
            (
                hd.iter()
                    .map(|v| {
                        let q = (v * iv).round_ties_even() as i32;
                        q.clamp(-127, 127) as i8
                    })
                    .collect(),
                ks,
            )
        };
        let mut k8 = vec![Vec::new(); t_len * kv_heads];
        let mut ks8 = vec![0f32; t_len * kv_heads];
        let mut v8 = vec![Vec::new(); t_len * kv_heads];
        let mut vs8 = vec![0f32; t_len * kv_heads];
        for t in 0..t_len {
            for kh in 0..kv_heads {
                let src = &kin[(t * kv_heads + kh) * d..(t * kv_heads + kh) * d + d];
                let mut hd = rms(src, Some(&knw));
                rope(&mut hd, pos0 + t as u32);
                let (qq, ss) = quant(&hd);
                k8[t * kv_heads + kh] = qq;
                ks8[t * kv_heads + kh] = ss;
                let vsrc = &vin[(t * kv_heads + kh) * d..(t * kv_heads + kh) * d + d];
                let hd2 = rms(vsrc, None);
                let (qq2, ss2) = quant(&hd2);
                v8[t * kv_heads + kh] = qq2;
                vs8[t * kv_heads + kh] = ss2;
            }
        }
        let mut maxd = 0f32;
        for t in 0..t_len {
            let lim = pos0 as usize + t + 1;
            let start = if sw > 0 && lim > sw { lim - sw } else { 0 };
            for h in 0..q_heads {
                let kvh = h / (q_heads / kv_heads);
                let qrow = &qh[(t * q_heads + h) * d..(t * q_heads + h) * d + d];
                let mut sc = Vec::new();
                for row in start..lim {
                    let kk = &k8[row * kv_heads + kvh];
                    let kss = ks8[row * kv_heads + kvh];
                    let mut acc = 0f32;
                    for j in 0..d {
                        acc += qrow[j] * (kk[j] as f32 * kss);
                    }
                    sc.push(acc * kq_scale);
                }
                let mx = sc.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let mut sum = 0f32;
                for v in sc.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                let mut o = vec![0f32; d];
                for (i, &p) in sc.iter().enumerate() {
                    let row = start + i;
                    let vv = &v8[row * kv_heads + kvh];
                    let vss = vs8[row * kv_heads + kvh];
                    for j in 0..d {
                        o[j] += p * (vv[j] as f32 * vss);
                    }
                }
                for j in 0..d {
                    o[j] /= sum;
                }
                let base_i = (t * q_heads + h) * d;
                for j in 0..d {
                    maxd = maxd.max((got[base_i + j] - o[j]).abs());
                }
            }
        }
        ok_all &= maxd < 2e-2;
        let q2 = &q[..2 * q_heads * d];
        let k2 = &kin[..2 * kv_heads * d];
        let v2 = &vin[..2 * kv_heads * d];
        let batch2 = dec
            .attn_g4_host(
                0, q2, k2, v2, &qnw, &knw, 0, 2, q_heads, kv_heads, d, sw, kq_scale, splits, pos0,
            )
            .map_err(|e| format!("attn_g4(분할·배치2 {name}): {e}"))?;
        let s1 = dec
            .attn_g4_host(
                0,
                &q2[..q_heads * d],
                &k2[..kv_heads * d],
                &v2[..kv_heads * d],
                &qnw,
                &knw,
                0,
                1,
                q_heads,
                kv_heads,
                d,
                sw,
                kq_scale,
                splits,
                pos0,
            )
            .map_err(|e| format!("attn_g4(분할1 {name}): {e}"))?;
        let s2 = dec
            .attn_g4_host(
                0,
                &q2[q_heads * d..],
                &k2[kv_heads * d..],
                &v2[kv_heads * d..],
                &qnw,
                &knw,
                0,
                1,
                q_heads,
                kv_heads,
                d,
                sw,
                kq_scale,
                splits,
                pos0 + 1,
            )
            .map_err(|e| format!("attn_g4(분할2 {name}): {e}"))?;
        let mut md = 0f32;
        for h in 0..q_heads {
            for j in 0..d {
                md = md.max((batch2[h * d + j] - s1[h * d + j]).abs());
                md = md.max((batch2[q_heads * d + h * d + j] - s2[h * d + j]).abs());
            }
        }
        ok_all &= md < 2e-2;
        out.push_str(&format!(
            "  {name}: 최대 |차| {maxd:.3e} · 분할(1×2) 대조 {md:.3e} — {}\n",
            if maxd < 2e-2 && md < 2e-2 {
                "통과"
            } else {
                "초과"
            }
        ));
    }
    out.push_str(&format!(
        "  판정: {}",
        if ok_all {
            "전 구성 허용(절대 ≤2e-2 — int8 KV 밴드)"
        } else {
            "초과"
        }
    ));
    Ok(out)
}

pub(super) fn run_prep_g4_check(dec: &mut llm170_backend_gpu::W4a16Dec) -> Result<String, String> {
    let (t_len, q_heads) = (2usize, 2usize);
    let pos0 = 3u32;
    let configs: [(&str, usize, usize, usize, usize, f32); 3] = [
        ("qwen 등가(256/32/32/64)", 256, 32, 32, 64, 1e7),
        ("gemma4 슬라이딩(256/128/128/256)", 256, 128, 128, 256, 1e4),
        ("gemma4 풀(512/64/256/512)", 512, 64, 256, 512, 1e6),
    ];
    let mut out = String::from("prep_g4 Q 경로 — CPU 미러(가중 RMS+rope 파라미터) 대조\n");
    let (mut ok_all, mut maxr_all) = (true, 0f32);
    for (name, d, pairs, span, denom, base) in configs {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let pm1 = |r: u64| ((r >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        let q: Vec<f32> = (0..t_len * q_heads * d).map(|_| pm1(rnd())).collect();
        let qnw: Vec<f32> = (0..d)
            .map(|_| 0.5 + (rnd() >> 40) as f32 / (1u32 << 25) as f32)
            .collect();
        let got = dec
            .attn_prep_g4_q_host(&q, &qnw, t_len, q_heads, d, pairs, span, denom, base, pos0)
            .map_err(|e| format!("prep_g4({name}): {e}"))?;
        let (mut maxd, mut maxr) = (0f32, 0f32);
        for t in 0..t_len {
            for j in 0..q_heads {
                let row = &q[(t * q_heads + j) * d..(t * q_heads + j) * d + d];
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let inv = 1.0f32 / (ss / d as f32 + 1e-6).sqrt();
                let mut h: Vec<f32> = row
                    .iter()
                    .zip(qnw.iter())
                    .map(|(v, w)| v * inv * w)
                    .collect();
                let pos = pos0 + t as u32;
                for p in 0..pairs {
                    let theta = base.powf(-(2.0 * p as f32) / denom as f32);
                    let ang = pos as f32 * theta;
                    let (c, s) = (ang.cos(), ang.sin());
                    let (x0, x1) = (h[p] as f64, h[p + span] as f64);
                    let (cf, sf) = (c as f64, s as f64);
                    h[p] = (x0 * cf - x1 * sf) as f32;
                    h[p + span] = (x0 * sf + x1 * cf) as f32;
                }
                let base_i = (t * q_heads + j) * d;
                for i in 0..d {
                    let dd = (got[base_i + i] - h[i]).abs();
                    maxd = maxd.max(dd);
                    if h[i].abs() > 1e-3 {
                        maxr = maxr.max(dd / h[i].abs());
                    }
                }
            }
        }
        ok_all &= maxd < 1e-5;
        maxr_all = maxr_all.max(maxd);
        out.push_str(&format!(
            "  {name}: max|diff| {maxd:.3e} 최대상대 {maxr:.3e} — {}\n",
            if maxd < 1e-5 { "통과" } else { "초과" }
        ));
    }
    out.push_str(&format!(
        "  판정: {} (최대절대 {maxr_all:.3e})",
        if ok_all {
            "전 구성 허용(≤1e-5)"
        } else {
            "초과"
        }
    ));
    Ok(out)
}

pub(super) fn run_moe_pes_check(dec: &mut llm170_backend_gpu::W4a16Dec) -> Result<String, String> {
    let n = 8usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let wt: Vec<f32> = (0..n)
        .map(|_| (rnd() >> 40) as f32 / (1u32 << 24) as f32)
        .collect();
    let idx: Vec<i32> = (0..n).map(|_| ((rnd() >> 50) as i32) % 16).collect();
    let pes: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.1).collect();
    let got = dec.moe_pes_scale_host(&wt, &idx, &pes)?;
    let mut bad = 0usize;
    for i in 0..n {
        let want = wt[i] * pes[idx[i] as usize];
        if got[i].to_bits() != want.to_bits() {
            bad += 1;
        }
    }
    Ok(format!(
        "moe_pes_scale vs wt·pes[idx] — n={n} 불일치 {bad}건 · 판정: {}",
        if bad == 0 {
            "비트일치"
        } else {
            "불일치"
        }
    ))
}

pub(super) fn run_ew_gelu_check(dec: &mut llm170_backend_gpu::W4a16Dec) -> Result<String, String> {
    let n = 4096usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mk = |rnd: &mut dyn FnMut() -> u64| -> Vec<f32> {
        (0..n)
            .map(|_| ((rnd() >> 40) as f32 / (1u32 << 24) as f32) * 6.0 - 3.0)
            .collect()
    };
    let g = mk(&mut rnd);
    let u = mk(&mut rnd);
    let got = dec.ew_gelu_host(&g, &u)?;
    let (mut maxd, mut rel) = (0f32, 0f32);
    for i in 0..n {
        let want = llm170_core::ops::gelu_tanh(g[i]) * u[i];
        let d = (got[i] - want).abs();
        maxd = maxd.max(d);
        if want.abs() > 1e-3 {
            rel = rel.max(d / want.abs());
        }
    }
    Ok(format!(
        "ew_gelu vs core::ops::gelu_tanh — n={n} max|diff|={maxd:.3e} 최대상대={rel:.3e} · 판정: {}",
        if maxd < 1e-5 {
            "일치(허용)"
        } else {
            "초과"
        }
    ))
}

pub(super) fn g4_gpu_chain(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    cfg: &crate::gpu_engine::G4HostCfg,
    prompt: &[u32],
    n_predict: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>), String> {
    let mut pos = 0u32;
    let mut lg = Vec::new();
    for &tok in prompt {
        dec.kv_grow(0, pos as usize + 1)?;
        lg = g4_step(dec, model, cfg, 0, tok, pos, true)?;
        pos += 1;
    }
    let mut step_logits: Vec<Vec<f32>> = vec![lg.clone()];
    let mut out = Vec::with_capacity(n_predict);
    for _ in 0..n_predict {
        let next = llm170_core::gemma4::greedy(&lg);
        out.push(next);
        dec.kv_grow(0, pos as usize + 1)?;
        lg = g4_step(dec, model, cfg, 0, next, pos, true)?;
        pos += 1;
        step_logits.push(lg.clone());
    }
    Ok((out, step_logits))
}

pub(super) fn g4_top5(l: &[f32]) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = l
        .iter()
        .copied()
        .enumerate()
        .map(|(i, x)| (i as u32, x))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(5);
    v
}
