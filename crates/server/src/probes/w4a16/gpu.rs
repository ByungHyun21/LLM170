//! [R2] w4a16-gpu — GPU 순차 디코드 프로브 + 캐스트 계약 테스트.

use super::super::{arg, arg_str};

/// GPU 순차 디코드 — 64층 체인을 CUDA(호스트 스테이징)로 돌리고
/// 임베딩 행·최종 head만 CPU 참조 경로(골든과 동일 계급).
///   w4a16-gpu <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N]
///              [--no-head] [--bench-gemm <lin> <t> <reps>] [--moe-check]
///              [--h2d-bench <MB>]
pub(super) fn gpu_run(args: &[String]) -> Result<String, String> {
    use llm170_backend_gpu::W4a16Dec;
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-gpu <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N] [--no-head] [--bench-gemm <lin> <t> <reps>] [--moe-check] [--h2d-bench <MB>] — 사용법: llm170 w4a16-gpu ../models/Qwen3.8-27B-W4A16-AutoRound --prompt-tokens 148678,65233,202419 --n-predict 8".into(),
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 8usize;
    let mut spec_k: usize = 0; // [A-1] 스페큘러티브 초안 상한(0=off).
    let mut spec_check = false; // [A-1 진단] 검증 vs t=1 디코드 토큰 대조.
    let mut ctx = 1024usize;
    let mut no_head = false;
    let mut moe_check = false;
    let mut plain_check = false;
    let mut bench_ew = false;
    let mut mma_smoke = false;
    let mut bench_gemv: Option<(String, usize)> = None;
    let mut bench_gemv_t: Option<(String, usize, usize)> = None;
    let mut bench_plain: Option<(String, usize, usize)> = None;
    let mut mma_diff: Option<(String, usize)> = None;
    let mut marlin_diff: Option<(String, usize)> = None; // [marlin S2] A/B 재패킹 게이트
    let mut marlin_selfcheck: Option<(usize, usize, usize, bool)> = None; // (t, k, n, m16)
    let mut marlin_selfcheck32: Option<(usize, usize, usize)> = None; // [marlin g32] bf16 자체점검
    let mut marlin_bench: Vec<(String, usize, usize, String)> = Vec::new(); // [marlin S2] mma 대비 벤치
    let mut marlin_ref: Option<(String, usize)> = None; // [marlin S2] 실무게 64행 참조 대조
    let mut attn_fa2: Vec<(usize, usize, u32)> = Vec::new(); // [FA2] A/B (t, layer, pos0)
    let mut attn_fa2_bench: Vec<(usize, usize, usize, u32)> = Vec::new(); // [FA2] (t, layer, reps, pos0)
    let mut mode_ref: u8 = 0; // 0 실값, 1 s→1, 2 x→1, 3 둘 다 1
    let mut kcap: usize = usize::MAX; // ref k 상한(이분탐색용)
    let mut moe_topk_check: Option<(usize, usize, usize)> = None;
    let mut h2d_mb = 0usize;
    let mut bench: Option<(String, usize, usize)> = None; // (lin, t, reps)
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prompt-tokens" => {
                let v = arg(&mut it, "--prompt-tokens requires ids")?;
                let ids: Vec<u32> = v
                    .split(',')
                    .map(|t| t.trim().parse::<u32>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("bad tokens: {e}"))?;
                if ids.is_empty() {
                    return Err("empty prompt".into());
                }
                prompts.push(ids);
            }
            "--n-predict" => {
                n_predict = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--n-predict requires a number")?;
            }
            "--spec" => {
                spec_k = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--spec requires a number")?;
            }
            "--spec-check" => {
                spec_check = true;
            }
            "--ctx" => {
                ctx = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ctx requires a number")?;
            }
            "--no-head" => no_head = true,
            "--moe-check" => moe_check = true,
            "--plain-gemm-check" => plain_check = true,
            "--bench-ew" => bench_ew = true,
            "--mma-smoke" => mma_smoke = true,
            "--h2d-bench" => {
                let mb = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--h2d-bench requires MB")?;
                h2d_mb = mb;
            }
            "--moe-topk-check" => {
                let t = it.next().and_then(|v| v.parse().ok()).unwrap_or(32);
                let n = it.next().and_then(|v| v.parse().ok()).unwrap_or(512);
                let k = it.next().and_then(|v| v.parse().ok()).unwrap_or(8);
                moe_topk_check = Some((t, n, k));
            }
            "--mma-diff" => {
                let name = arg(&mut it, "--mma-diff requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--mma-diff requires t")?;
                mma_diff = Some((name, t));
            }
            "--marlin-diff" => {
                let name = arg(&mut it, "--marlin-diff requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-diff requires t")?;
                marlin_diff = Some((name, t));
            }
            "--marlin-selfcheck32" => {
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-selfcheck32 requires t")?;
                let k = it.next().and_then(|v| v.parse().ok()).unwrap_or(128usize);
                let n = it.next().and_then(|v| v.parse().ok()).unwrap_or(128usize);
                marlin_selfcheck32 = Some((t, k, n));
            }
            "--attn-fa2-bench" => {
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--attn-fa2-bench requires t")?;
                let lay = it.next().and_then(|v| v.parse().ok()).unwrap_or(0usize);
                let reps = it.next().and_then(|v| v.parse().ok()).unwrap_or(8usize);
                let pos0 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0u32);
                attn_fa2_bench.push((t, lay, reps, pos0));
            }
            "--attn-fa2" => {
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--attn-fa2 requires t")?;
                let mut lay = 0usize;
                if let Some(nx) = it.clone().next()
                    && !nx.starts_with("--")
                {
                    lay = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                let mut p0 = 0u32;
                if let Some(nx) = it.clone().next()
                    && !nx.starts_with("--")
                {
                    p0 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                attn_fa2.push((t, lay, p0));
            }
            "--marlin-ref" => {
                let name = arg(&mut it, "--marlin-ref requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-ref requires t")?;
                mode_ref = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                kcap = it.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
                marlin_ref = Some((name, t));
            }
            "--marlin-selfcheck" => {
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-selfcheck requires t")?;
                let (mut k, mut n, mut m16) = (128usize, 128usize, false);
                if let Some(nx) = it.clone().next()
                    && !nx.starts_with("--")
                {
                    k = it.next().and_then(|v| v.parse().ok()).unwrap_or(128);
                }
                if let Some(nx) = it.clone().next()
                    && !nx.starts_with("--")
                {
                    n = it.next().and_then(|v| v.parse().ok()).unwrap_or(128);
                }
                if let Some(nx) = it.clone().next()
                    && nx == "m16"
                {
                    let _ = it.next();
                    m16 = true;
                }
                marlin_selfcheck = Some((t, k, n, m16));
            }
            "--marlin-bench" => {
                let name = arg(&mut it, "--marlin-bench requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-bench requires t")?;
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-bench requires reps")?;
                let mut kern = String::new();
                if let Some(nx) = it.clone().next()
                    && !nx.starts_with("--")
                {
                    kern = it.next().unwrap().to_string();
                }
                marlin_bench.push((name, t, reps, kern));
            }
            "--bench-plain" => {
                let name = arg(&mut it, "--bench-plain requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-plain requires t")?;
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-plain requires reps")?;
                bench_plain = Some((name, t, reps));
            }
            "--bench-gemv" => {
                let name = arg(&mut it, "--bench-gemv requires a name")?.to_string();
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemv requires reps")?;
                bench_gemv = Some((name, reps));
            }
            "--bench-gemv-all" => {
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemv-all requires reps")?;
                bench_gemv = Some(("ALL".to_string(), reps));
            }
            "--bench-gemm" => {
                let name = arg(&mut it, "--bench-gemm requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemm requires t")?;
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemm requires reps")?;
                bench = Some((name, t, reps));
            }
            "--bench-gemv-t" => {
                let name = arg(&mut it, "--bench-gemv-t requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemv-t requires t")?;
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemv-t requires reps")?;
                bench_gemv_t = Some((name, t, reps));
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if bench.is_none() && prompts.len() != 1 {
        return Err("w4a16-gpu: 단일 프롬프트 전용(v1)".into());
    }
    let prompt: Vec<u32> = prompts.first().cloned().unwrap_or_default();
    let prompt = &prompt;
    if bench.is_none() && prompt.len() + n_predict + 1 > ctx {
        return Err(format!("ctx({ctx}) too small for prompt+n_predict"));
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
    // 1~4) 모델 상주 업로드(GpuEngine 공용 — dense/MoE 분기 포함).
    crate::gpu_engine::upload_model(&mut dec, &model, ctx)?;
    // 보고용 split 선형 수(dense 한정 — MoE는 0).
    let n_lin = model
        .engine_names()
        .iter()
        .filter(|n| {
            model
                .w_raw(n)
                .map(|w| w.ty == llm170_core::wtype::WType::W4a16Split)
                .unwrap_or(false)
        })
        .count();
    if h2d_mb > 0 {
        // h2d 대역폭 실측 — (a) 핀드 소스, (b) 페이지러블 소스(드라이버 스테이징).
        let bytes = h2d_mb << 20;
        let src = vec![0x5Au8; bytes];
        let dst = dec.alloc_scratch(bytes)?;
        let pinned = dec.alloc_pinned_scratch(bytes)?;
        // SAFETY: pinned은 alloc_pinned_scratch가 돌려준 유효 핸들, bytes는 할당 크기.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), pinned as *mut u8, bytes);
        }
        let t = std::time::Instant::now();
        dec.h2d_scratch(dst, &src)?;
        dec.sync_bench()?;
        let pageable_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        // SAFETY: 핀드 버퍼 슬라이스(할당 크기).
        let ps = unsafe { std::slice::from_raw_parts(pinned as *const u8, bytes) };
        dec.h2d_scratch(dst, ps)?;
        dec.sync_bench()?;
        let pinned_ms = t.elapsed().as_secs_f64() * 1e3;
        dec.free_pinned_scratch(pinned)?;
        dec.free_scratch(dst)?;
        return Ok(format!(
            "h2d-bench {h2d_mb}MiB: 페이지러블 {pageable_ms:.0}ms ({:.1}GB/s) · 핀드 {pinned_ms:.0}ms ({:.1}GB/s)",
            bytes as f64 * 1e-9 / (pageable_ms * 1e-3),
            bytes as f64 * 1e-9 / (pinned_ms * 1e-3).max(1e-9)
        ));
    }
    if bench_ew {
        return dec.bench_ew();
    }
    if plain_check {
        return dec.plain_gemm_selfcheck();
    }
    if mma_smoke {
        return dec.mma_smoke();
    }
    if moe_check {
        return dec.moe_selfcheck();
    }
    let upload_ms = t0.elapsed().as_secs_f64() * 1e3;
    // 5) 프롬프트 순차 prefill + greedy 생성(head는 CPU 참조 경로).
    let head = model
        .w("output.weight")
        .ok_or_else(|| "output.weight 부재".to_string())?;
    let head_logits = |xn: &[f32]| -> Vec<f32> {
        if no_head {
            // head 비용 분리 계측 전용 — 체인 시간만 재기 위한 가짜 로짓.
            return xn[..5120].to_vec();
        }
        let mut lg = vec![0.0f32; head.n_out as usize];
        llm170_core::matmul::matmul(xn, &head, &mut lg);
        lg
    };
    if let Some((t, n, k)) = &moe_topk_check {
        return dec.moe_topk_check(*t, *n, *k);
    }
    if let Some((name, t)) = &mma_diff {
        return dec.mma_diff_check(name, *t);
    }
    // [marlin S2 2026-10-12] marlin B 배치 GEMM A/B — 원본 q를 core::marlin
    // (호스트 재패킹)으로 변환해 신규 커널 vs 기존 mma 커널 판정. g128·f16 전용.
    if let Some((name, t)) = &marlin_diff {
        let w = model
            .w_raw(name)
            .ok_or_else(|| format!("--marlin-diff {name}: 무게 없음"))?;
        if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
            return Err(format!(
                "--marlin-diff {name}: g128·f16 split 전용(ty={:?} group={} bf16={})",
                w.ty, w.group, w.scale_bf16
            ));
        }
        let q: Vec<u32> = w
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let qm = llm170_core::marlin::marlin_repack_q(&q, w.n_out as usize, w.n_in as usize)?;
        return dec.marlin_diff_check(name, *t, &qm);
    }
    // [marlin S2 디버그] 커널 격리 자체점검 — 합성 n=128·k=128·s=1·x=1.
    // 기대 = 정수 Σ(nib−8) (f32 누적 순서 무관 비트정확).
    if let Some((t, k, n, m16)) = marlin_selfcheck {
        let nib_at = |kk: usize, nn: usize| ((kk * 7 + nn * 3) % 16) as u32;
        let mut q = vec![0u32; n * (k / 8)];
        for nn in 0..n {
            for kk in 0..k {
                q[nn * (k / 8) + kk / 8] |= nib_at(kk, nn) << ((kk % 8) * 4);
            }
        }
        // 스케일·x = 2의 거듭제곱(정확 f16) — 지수 오배치 시 기대가 어긋난다.
        let mut s = vec![0u16; n * (k / 128)];
        for nn in 0..n {
            for gg in 0..(k / 128) {
                s[nn * (k / 128) + gg] = ((15 - ((nn + gg) % 5)) as u16) << 10;
            }
        }
        let x16: Vec<u16> = (0..t * k)
            .map(|i| ((15 - ((i / k + (i % k) / 8) % 4)) as u16) << 10)
            .collect();
        let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
        let sf = |h: u16| -> f64 {
            let e = ((h >> 10) & 0x1F) as i32;
            let m = (h & 0x3FF) as f64 / 1024.0;
            (1.0 + m) * 2f64.powi(e - 15)
        };
        let mut expect = vec![0f32; t * n];
        for r in 0..t {
            for nn in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    let sc = sf(s[nn * (k / 128) + kk / 128]);
                    let xv = sf(x16[r * k + kk]);
                    acc += (nib_at(kk, nn) as f64 - 8.0) * sc * xv;
                }
                expect[r * n + nn] = acc as f32;
            }
        }
        let (sym, bm, bn, thr) = if m16 {
            ("w4a16_gemm_g128_marlin_m16", 16usize, 256usize, 128u32)
        } else {
            ("w4a16_gemm_g128_marlin", 64, 128, 256)
        };
        return dec.marlin_selfcheck(sym, bm, bn, thr, &qm, &s, &x16, &expect, t, n, k);
    }
    // [marlin g32] 합성 자체점검 — bf16 2^-j 스케일(KC=32=그룹 1개), x=1.
    if let Some((t, k, n)) = marlin_selfcheck32 {
        let nib_at = |kk: usize, nn: usize| ((kk * 7 + nn * 3) % 16) as u32;
        let mut q = vec![0u32; n * (k / 8)];
        for nn in 0..n {
            for kk in 0..k {
                q[nn * (k / 8) + kk / 8] |= nib_at(kk, nn) << ((kk % 8) * 4);
            }
        }
        let mut s = vec![0u16; n * (k / 32)];
        for nn in 0..n {
            for gg in 0..(k / 32) {
                s[nn * (k / 32) + gg] = ((127 - ((nn + gg) % 5)) as u16) << 7;
            }
        }
        let x16 = vec![0x3C00u16; t * k]; // f16 1.0
        let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
        let mut expect = vec![0f32; t * n];
        for r in 0..t {
            for nn in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    let j = ((nn + kk / 32) % 5) as i32;
                    acc += (nib_at(kk, nn) as f64 - 8.0) * 2f64.powi(-j);
                }
                expect[r * n + nn] = acc as f32;
            }
        }
        return dec.marlin_selfcheck(
            "w4a16_gemm_g32_marlin",
            64,
            128,
            256,
            &qm,
            &s,
            &x16,
            &expect,
            t,
            n,
            k,
        );
    }
    // [FA2] 프리필 어텐션 벤치 — prep 포함 기존 vs FA2.
    if !attn_fa2_bench.is_empty() {
        let mut out = Vec::new();
        for (t, lay, reps, pos0) in &attn_fa2_bench {
            out.push(dec.attn_fa2_bench(0, *lay, *t, *reps, *pos0)?);
        }
        return Ok(out.join("\n"));
    }
    // [FA2] 프리필 어텐션 A/B — 기존 경로 vs attn_fa2_qg.
    if !attn_fa2.is_empty() {
        let mut out = Vec::new();
        for (t, lay, p0) in &attn_fa2 {
            out.push(dec.attn_fa2_diff(0, *lay, *t, *p0)?);
        }
        return Ok(out.join("\n"));
    }
    // [marlin S2 디버그] 실무게 64행 슬라이스 vs 호스트 기대값 — 커널 단독
    // 정오 판정(스케일·디퀀트 포함). 실패 시 (행,열,기대,실제)를 보여준다.
    if let Some((name, t)) = &marlin_ref {
        let t = *t;
        let w = model
            .w_raw(name)
            .ok_or_else(|| format!("--marlin-ref {name}: 무게 없음"))?;
        if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
            return Err(format!("--marlin-ref {name}: g128·f16 split 전용"));
        }
        let (n, k) = (64usize, kcap.min(w.n_in as usize));
        let q: Vec<u32> = w.data[..n * (k / 8) * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let s16: Vec<u16> = w.aux.ok_or("aux 부재")?[..n * (k / 128) * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
        let x16: Vec<u16> = (0..t * k)
            .map(|i| {
                let v =
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
                llm170_backend_gpu::f32_to_f16(v)
            })
            .collect();
        // 모드: 0=실스케일·실x · 1=s→1 · 2=x→1 · 3=둘 다 1(가중치만).
        let mode = mode_ref;
        let mut s_use = s16.clone();
        let mut x_use = x16.clone();
        if mode == 1 || mode == 3 {
            s_use.fill(0x3C00);
        }
        if mode == 2 || mode == 3 {
            x_use.fill(0x3C00);
        }
        // 기대값 — 디퀀트 후 f16(커널과 동일 RNE), 누적은 f64(순서 무관).
        let mut expect = vec![0f32; t * n];
        for r in 0..t {
            for nn in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    let nib = ((q[nn * (k / 8) + kk / 8] >> ((kk % 8) * 4)) & 0xF) as i32 - 8;
                    let sc = llm170_core::quant::half_to_f32(s_use[nn * (k / 128) + kk / 128]);
                    let wf =
                        llm170_backend_gpu::h2f(llm170_backend_gpu::f32_to_f16(nib as f32 * sc));
                    acc += (wf as f64) * (llm170_backend_gpu::h2f(x_use[r * k + kk]) as f64);
                }
                expect[r * n + nn] = acc as f32;
            }
        }
        return dec.marlin_selfcheck(
            "w4a16_gemm_g128_marlin",
            64,
            128,
            256,
            &qm,
            &s_use,
            &x_use,
            &expect,
            t,
            n,
            k,
        );
    }
    // [marlin S2] mma 대비 커널 벤치 — 같은 입력·grid·회차(인터리브).
    if !marlin_bench.is_empty() {
        let mut out = Vec::new();
        for (name, t, reps, kern) in &marlin_bench {
            let Some(w) = model.w_raw(name) else {
                out.push(format!("marlin-bench {name}: 무게 없음(건너뜀)"));
                continue;
            };
            if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
                out.push(format!(
                    "marlin-bench {name}: g128·f16 split 아님(건너뜀: ty={:?} group={} bf16={})",
                    w.ty, w.group, w.scale_bf16
                ));
                continue;
            }
            let q: Vec<u32> = w
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let qm = llm170_core::marlin::marlin_repack_q(&q, w.n_out as usize, w.n_in as usize)?;
            let (msym, mbm, mbn, mthr, vs_gemv) = match kern.as_str() {
                // [G4 2026-10-12] m16v = m16 마린 vs 행-GEMV(t=1) — 재개 조건 (a) 판정.
                "m16v" => (
                    "w4a16_gemm_g128_marlin_m16",
                    16usize,
                    256usize,
                    128u32,
                    true,
                ),
                "m16" => (
                    "w4a16_gemm_g128_marlin_m16",
                    16usize,
                    256usize,
                    128u32,
                    false,
                ),
                _ => ("w4a16_gemm_g128_marlin", 64, 128, 256, false),
            };
            out.push(dec.marlin_bench(name, *t, *reps, &qm, msym, mbm, mbn, mthr, vs_gemv)?);
        }
        return Ok(out.join("\n"));
    }
    // 마이크로벤치(진단): 플레인 GEMM(t) 반복 — v1/v3 vs mma.
    if let Some((name, bt, reps)) = &bench_plain {
        let (n, k) = model
            .w_raw(name)
            .map(|w| (w.n_out as usize, w.n_in as usize))
            .ok_or_else(|| format!("--bench-plain {name}: 무게 없음"))?;
        let xf: Vec<f32> = vec![1.0f32; bt * k];
        // SAFETY: xf는 f32 슬라이스 — 바이트 재해석 읽기(할당 범위 내).
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = dec.alloc_scratch(xf.len() * 4)?;
        dec.h2d_scratch(dx, xb)?;
        let dy = dec.alloc_scratch(bt * n * 4)?;
        dec.plain_bench_launch(name, dx, dy, *bt)?;
        dec.sync_bench()?;
        let t0 = std::time::Instant::now();
        for _ in 0..*reps {
            dec.plain_bench_launch(name, dx, dy, *bt)?;
        }
        dec.sync_bench()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / *reps as f64;
        let flop = 2.0 * (n as f64) * (k as f64) * (*bt as f64);
        dec.free_scratch(dx)?;
        dec.free_scratch(dy)?;
        return Ok(format!(
            "bench-plain {name} n={n} k={k} t={bt}: {ms:.3} ms/회 · {:.2} TF",
            flop * 1e-9 / (ms * 1e-3)
        ));
    }
    // 마이크로벤치(진단): t행 GEMV(배치 디코드 커널) 반복 — A9 판정.
    if let Some((name, t, reps)) = &bench_gemv_t {
        let ms = dec.bench_gemv_t(name, *t, *reps)?;
        return Ok(format!("bench-gemv-t {name} t={t}: {ms:.3} ms/회"));
    }
    // 마이크로벤치(진단): 지정 선형의 t=1 GEMV 반복 — 실효 가중치 대역.
    if let Some((name, reps)) = &bench_gemv
        && name == "ALL"
    {
        let (ms, wb) = dec.gemv_walk_bench(*reps)?;
        let gbs = wb as f64 * 1e-9 / (ms * 1e-3);
        return Ok(format!(
            "bench-gemv-all: {ms:.2} ms/회 · 가중치 {gbs:.0} GB/s ({:.2} GB)",
            wb as f64 / 1e9
        ));
    }
    if let Some((name, reps)) = &bench_gemv {
        let (n, k) = model
            .w_raw(name)
            .map(|w| (w.n_out as usize, w.n_in as usize))
            .ok_or_else(|| format!("--bench-gemv {name}: 무게 없음"))?;
        let xf: Vec<f32> = vec![1.0f32; k];
        // SAFETY: xf는 f32 슬라이스 — 바이트 재해석 읽기(할당 범위 내).
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = dec.alloc_scratch(xf.len() * 4)?;
        dec.h2d_scratch(dx, xb)?;
        let dy = dec.alloc_scratch(n * 4)?;
        dec.gemv_bench_launch(name, dx, dy)?;
        dec.sync_bench()?;
        let t0 = std::time::Instant::now();
        for _ in 0..*reps {
            dec.gemv_bench_launch(name, dx, dy)?;
        }
        dec.sync_bench()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / *reps as f64;
        let wb = if model.w_raw(name).map(|w| w.ty) == Some(llm170_core::wtype::WType::W4a16Split) {
            (n * k) as f64 / 2.0
        } else {
            (n * k) as f64 * 2.0
        };
        let gbs = wb * 1e-9 / (ms * 1e-3);
        dec.free_scratch(dx)?;
        dec.free_scratch(dy)?;
        return Ok(format!(
            "bench-gemv {name} n={n} k={k}: {ms:.3} ms/회 · 가중치 {gbs:.0} GB/s"
        ));
    }
    // 마이크로벤치(진단): 지정 선형의 t≥2 GEMM 반복 시간·실효 GB/s.
    if let Some((name, bt, reps)) = &bench {
        let (n, k) = model
            .w_raw(name)
            .map(|w| (w.n_out as usize, w.n_in as usize))
            .ok_or_else(|| format!("--bench-gemm {name}: 무게 없음"))?;
        let xf: Vec<f32> = vec![1.0f32; bt * k]; // f32 1.0 — 수치 무의미
        // SAFETY: xf는 f32 슬라이스 — 바이트 재해석 읽기(할당 범위 내).
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dxh = dec.alloc_scratch(xf.len() * 4)?;
        dec.h2d_scratch(dxh, xb)?;
        let dout = dec.alloc_scratch(bt * n * 4)?;
        // 워밍 1회.
        dec.gemm_bench_launch(name, dxh, dout, *bt)?;
        dec.sync_bench()?;
        let t0 = std::time::Instant::now();
        for _ in 0..*reps {
            dec.gemm_bench_launch(name, dxh, dout, *bt)?;
        }
        dec.sync_bench()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / *reps as f64;
        let wb = (n * k) as f64 / 2.0; // 4bit+scale ≈ 0.5625B/원소 근사
        let gbs = wb * 1e-9 / (ms * 1e-3);
        dec.free_scratch(dxh)?;
        dec.free_scratch(dout)?;
        return Ok(format!(
            "bench-gemm {name} n={n} k={k} t={bt} reps={reps}: {ms:.2} ms/회 · 가중치 {gbs:.0} GB/s"
        ));
    }
    // GPU head — bf16 output.weight를 상주 업로드(--no-head는 CPU 참조).
    let mut head_gpu = false;
    if !no_head && head.ty == llm170_core::wtype::WType::Bf16 {
        dec.upload_head(head.data, head.n_out as usize, head.n_in as usize)?;
        head_gpu = true;
    }
    let t1 = std::time::Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut next = 0u32;
    // 프롬프트는 t≤8 배치 청크로 — 골든 판정이 배치 경로(GEMM t≥2)를 지난다.
    {
        let h = hp.n_embd;
        let mut i = 0usize;
        while i < prompt.len() {
            // [2026-10-09] MoE 상주는 32(기본), 스트리밍은 1. [P10] 상한 128 —
            // LLM170_PREFILL_T로 실측 오버라이드(엔진 prefill과 동일 규칙).
            let cap = if hp.n_experts > 0 && !dec.moe_experts_resident() {
                1
            } else {
                llm170_diag::flag::val("LLM170_PREFILL_T")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(512) // 청크 확대
                    .clamp(1, 512)
            };
            let t = (prompt.len() - i).min(cap);
            let mut rows: Vec<f32> = Vec::with_capacity(t * h);
            for &tok in &prompt[i..i + t] {
                rows.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
            }
            let last = i + t == prompt.len();
            if head_gpu {
                let lg = dec.forward_prefill(0, &rows, t, last)?;
                if last {
                    next = llm170_core::matmul::greedy_from(&lg);
                }
            } else {
                let xn = dec.forward_prefill(0, &rows, t, false)?;
                if last {
                    next = llm170_core::matmul::greedy_from(&head_logits(&xn));
                }
            }
            i += t;
        }
    }
    out.push(next);
    if llm170_diag::flag::ne0("LLM170_TIME") {
        eprintln!("{}", dec.prof_report("prefill")?);
    }
    // [A-1] 스페큘러티브 디코딩 — n-gram 초안 + 배치 검증 + GDN 롤백.
    // 출력 토큰 스트림은 비스펙(그리디)과 **정확히 동일**해야 한다(골든 판정).
    let mut hist: Vec<u32> = prompt.to_vec();
    if spec_k >= 2 && head_gpu {
        dec.enable_spec()?;
    }
    if spec_check && head_gpu {
        // [A-1 진단] 검증(t=4 배치) vs t=1 디코드의 토큰 대조.
        // S0 저장 → t=1 4회(T1..T4) → S0 복원+pos 되감기 → 검증 [next,T1,T2,T3].
        dec.enable_spec()?;
        dec.spec_save_state(0)?;
        let mut t1_4: Vec<u32> = Vec::new();
        {
            let mut cur = next;
            for _ in 0..4 {
                let row = model.embed_row(cur).map_err(|e| e.to_string())?;
                cur = dec.forward_device_argmax(0, &row)?;
                t1_4.push(cur);
            }
        }
        // trio(4×t=1)의 최종 상태 저장 — spec scan 대조의 정답 참조.
        let s_trio = dec.spec_dump_state(0)?;
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        let mut rows: Vec<f32> = Vec::with_capacity(4 * hp.n_embd);
        rows.extend_from_slice(&model.embed_row(next).map_err(|e| e.to_string())?);
        for &tk in &t1_4[..3] {
            rows.extend_from_slice(&model.embed_row(tk).map_err(|e| e.to_string())?);
        }
        let toks = dec.spec_verify(0, &rows, 4)?;
        eprintln!("[spec-check] t1={t1_4:?} verify={toks:?}");
        // o_lc 대조 — spec scan on vs off(같은 입력·같은 상태).
        let nv = 4 * 32 * 128; // h_v=32 고정 가정(35B)
        let a = dec.spec_dump_outv(nv)?;
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        dec.spec_set_scan(false);
        let _ = dec.spec_verify(0, &rows, 4)?;
        let b = dec.spec_dump_outv(nv)?;
        dec.spec_set_scan(true);
        let mut mx = 0.0f32;
        let mut mi = 0usize;
        for i in 0..nv {
            let d = (a[i] - b[i]).abs();
            if d > mx {
                mx = d;
                mi = i;
            }
        }
        eprintln!(
            "[spec-check] o_lc maxdiff(on-off) = {mx:.6e} @i={mi}(t={} h={} d={}) on={:.6e} off={:.6e}",
            mi / (32 * 128),
            (mi / 128) % 32,
            mi % 128,
            a[mi],
            b[mi]
        );
        for t in 0..4 {
            let mut tm = 0.0f32;
            for i in t * 32 * 128..(t + 1) * 32 * 128 {
                tm = tm.max((a[i] - b[i]).abs());
            }
            eprintln!("[spec-check] t={t} maxdiff={tm:.6e}");
        }
        // [핵심 대조] spec scan(t=4) 최종 상태 vs trio(4×t=1) 최종 상태.
        // + 정상 scan(WY) 경로도 같은 비교 — 공식 차이 규모 판별.
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        dec.spec_set_scan(false);
        let _ = dec.spec_verify(0, &rows, 4)?;
        let s_wy = dec.spec_dump_state(0)?;
        dec.spec_set_scan(true);
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        let _ = dec.spec_verify(0, &rows, 4)?;
        let mut wmx = 0.0f32;
        for i in 0..s_trio.len().min(s_wy.len()) {
            wmx = wmx.max((s_trio[i] - s_wy[i]).abs());
        }
        eprintln!("[spec-check] state maxdiff(trio-WY) = {wmx:.6e}");
        let s_spec = dec.spec_dump_state(0)?;
        let mut smx = 0.0f32;
        let mut smi = 0usize;
        for i in 0..s_trio.len().min(s_spec.len()) {
            let d = (s_trio[i] - s_spec[i]).abs();
            if d > smx {
                smx = d;
                smi = i;
            }
        }
        eprintln!(
            "[spec-check] state maxdiff(trio-spec) = {smx:.6e} @i={smi} (층 {}) trio={:.6e} spec={:.6e}",
            smi / (32 * 128 * 128),
            s_trio[smi],
            s_spec[smi]
        );
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
    }
    let (mut spec_rounds, mut spec_drafted, mut spec_acc) = (0u64, 0u64, 0u64);
    let spec_t0 = std::time::Instant::now();
    // 원 루프(for _ in 0..n_predict) = n_predict회 **추가** — out은 프리필
    // 첫 토큰을 이미 담고 있다(오프바이원 실측: 35B 4000 골든 8토큰).
    while out.len() <= n_predict {
        if spec_k >= 2 && head_gpu {
            // 초안 = 최대 spec_k-1개(검증 배치 = 1+초안 ≤ spec_k).
            // n=4 — 12는 우리 산문 테스트에서 발화 0(실측). 짧은 패턴이
            // 반복 구조(리스트·코드)를 더 자주 잡는다(수용률은 검증이 판정).
            let draft = llm170_core::spec::ngram_draft(&hist, 4, 48, spec_k - 1);
            if !draft.is_empty() {
                let t = 1 + draft.len();
                let mut rows: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
                rows.extend_from_slice(&model.embed_row(next).map_err(|e| e.to_string())?);
                for &tk in &draft {
                    rows.extend_from_slice(&model.embed_row(tk).map_err(|e| e.to_string())?);
                }
                // [A-1 완화 ①] 검증 전 상태 저장 — 보정 토큰을 t=1로 재계산
                // (배치 검증의 근접 동률 argmax 플립이 오거부/오보정을 만들던
                // 구조적 한계 해소: 수용 판정만 배치, 방출 토큰은 정확 수치).
                dec.spec_save_state(0)?;
                let toks = dec.spec_verify(0, &rows, t)?;
                let mut acc = 0usize;
                while acc < draft.len() && toks[acc] == draft[acc] {
                    acc += 1;
                }
                spec_rounds += 1;
                spec_drafted += draft.len() as u64;
                spec_acc += acc as u64;
                // 상태·pos를 마지막 수용 토큰 **직전**으로 되감고 t=1 디코드.
                let last = if acc >= 1 { draft[acc - 1] } else { next };
                if acc >= 1 {
                    dec.spec_rollback(0, acc)?;
                    dec.spec_rewind_pos(0, (t - acc) as u32)?;
                } else {
                    dec.spec_load_state(0)?;
                    dec.spec_rewind_pos(0, t as u32)?;
                }
                let lrow = model.embed_row(last).map_err(|e| e.to_string())?;
                let corr = dec.forward_device_argmax(0, &lrow)?;
                for &tk in &draft[..acc] {
                    out.push(tk);
                    hist.push(tk);
                }
                out.push(corr);
                hist.push(corr);
                next = corr;
                continue;
            }
        }
        let row = model.embed_row(next).map_err(|e| e.to_string())?;
        if head_gpu {
            // [P3] 디코드는 디바이스 argmax(4B d2h) — 토큰 스트림은 종전
            // 로짓+CPU greedy_from과 동일해야 한다(골든이 동등성 판정).
            next = dec.forward_device_argmax(0, &row)?;
        } else {
            let xn = dec.forward_device(0, &row)?;
            next = llm170_core::matmul::greedy_from(&head_logits(&xn));
        }
        out.push(next);
        hist.push(next);
    }
    if llm170_diag::flag::ne0("LLM170_TIME") {
        // 진단(P8): 커널 범주별 소요 — 그래프 캡처 중에는 마킹이 꺼지므로
        // LLM170_GRAPH=0 직접 경로에서 의미가 있다.
        eprintln!("{}", dec.prof_report(&format!("decode {n_predict}토큰"))?);
    }
    if spec_rounds > 0 {
        eprintln!(
            "[spec] 라운드 {spec_rounds} · 초안 {spec_drafted} · 수용 {spec_acc} (수용률 {:.0}%) · {:.1}ms",
            spec_acc as f64 / spec_drafted.max(1) as f64 * 100.0,
            spec_t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;
    let csv = out
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    // MoE는 플레인(비양자화) 업로드 실측 — dense는 split 선형 수 기반 추정.
    let plain_gb = if hp.n_experts > 0 {
        model
            .engine_names()
            .iter()
            .filter(|n| n.as_str() != "token_embd.weight" && n.as_str() != "output.weight")
            .filter_map(|n| model.w_raw(n))
            .filter(|w| w.ty == llm170_core::wtype::WType::Bf16)
            .map(|w| w.data.len() as f64 * 1e-9)
            .sum::<f64>()
    } else {
        n_lin as f64 * 44.6e-3
    };
    Ok(format!(
        "w4a16-gpu {dir} — GPU 체인(디바이스 상주{} · {})\n  업로드 {upload_ms:.0}ms · 생성 {}토큰 {gen_ms:.0}ms ({:.1}ms/토큰)\n tokens: {csv}",
        if head_gpu {
            " + GPU head"
        } else {
            " + CPU head"
        },
        if hp.n_experts > 0 {
            format!("MoE 플레인 {plain_gb:.1}GB + 전문가 스트리밍/상주")
        } else {
            format!("선형 {n_lin}개 · {plain_gb:.1}GB급")
        },
        out.len(),
        gen_ms / out.len() as f64,
    ))
}

#[cfg(test)]
mod cast_contract {
    //! G3 — f16 캐스트 트윈 전수 대조(CPU 전용, GPU 불필요).
    //! 4중 수동 복사 중 호스트 3본(gptq4::h2f · w4a16_dec::f32_to_f16 ·
    //! core deq)을 65,536 패턴으로 묶는다. 커널(.cu f2h/h2f)은 골든 게이트 소관.
    use llm170_core::quant::half_to_f32;

    /// h2f(backend-gpu) ≡ core half_to_f32 — u16 전 패턴.
    #[test]
    fn h2f_matches_core_all_patterns() {
        for h in 0..=u16::MAX {
            assert_eq!(
                llm170_backend_gpu::h2f(h).to_bits(),
                half_to_f32(h).to_bits(),
                "h2f({h:#06x})"
            );
        }
    }

    /// f32_to_f16 왕복 항등 — f16 전 패턴(NaN 제외). core half_to_f32로
    /// 올린 뒤 되돌리면 원 비트(서브노멀·±inf 포함).
    #[test]
    fn f32_to_f16_roundtrips_all_non_nan_patterns() {
        for h in 0..=u16::MAX {
            let exp = (h >> 10) & 0x1F;
            let man = h & 0x3FF;
            if exp == 0x1F && man != 0 {
                continue; // NaN — 페이로드 정규화(0x200)라 항등 아님(별도 검사).
            }
            let v = half_to_f32(h);
            assert_eq!(llm170_backend_gpu::f32_to_f16(v), h, "roundtrip {h:#06x}");
        }
    }

    /// 경계·반올림(RN-even)·NaN 규약.
    #[test]
    fn f32_to_f16_edges() {
        let f = llm170_backend_gpu::f32_to_f16;
        assert_eq!(f(1.0), 0x3C00);
        assert_eq!(f(-2.0), 0xC000);
        assert_eq!(f(0.0), 0x0000);
        assert_eq!(f(-0.0), 0x8000);
        assert_eq!(f(65504.0), 0x7BFF); // f16 최대 정규수
        assert_eq!(f(65520.0), 0x7C00); // 오버플로 타이 → +inf
        assert_eq!(f(f32::INFINITY), 0x7C00);
        assert_eq!(f(f32::NEG_INFINITY), 0xFC00);
        assert_eq!(f(f32::NAN), 0x7E00); // qNaN 정규화
        assert_eq!(f(1e-8), 0x0000); // e < -10 → 0
        assert_eq!(f(5.960_464_5e-8), 0x0001); // 2^-24 = 최소 서브노멀
        // RN-even 타이: 2049.0(2048+1, ulp=2의 반) → 짝수 가수 2048.
        assert_eq!(f(2049.0), f(2048.0));
        assert_eq!(f(2050.0), 0x6801); // 정확값(반올림 없음)
    }
}
