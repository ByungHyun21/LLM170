//! W4A16 프로브 — `w4a16-load`(로더 완전성) · `w4a16-ref`(참조 러너).
//!
//! `w4a16-ref`: CPU 참조 greedy 토큰열 — 커널/서빙 판정의 오라클.
//! 모듈 단위 디버그(gemv/gemm/layer) 프로브는 W2/W3에서 이 파일에 얹는다.

use super::{arg_str, finish};
use std::process::ExitCode;

pub fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    match cmd {
        "w4a16-load" => Some(finish(load(args))),
        "w4a16-ref" => Some(finish(reference(args))),
        "w4a16-gemv" => Some(finish(gemm_gate(args, 1))),
        "w4a16-gemm" => Some(finish(gemm_gate(args, 8))),
        "w4a16-gpu" => Some(finish(gpu_run(args))),
        _ => None,
    }
}

fn load(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-load <dir> — 사용법: llm170 w4a16-load ../models/Qwen3.8-27B-W4A16-AutoRound"
                .into(),
        );
    }
    let store = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let cfg = llm170_core::qwen35::bind::QwenCfg::load(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let rep = llm170_core::qwen35::bind::validate(&store, &cfg).map_err(|e| e.to_string())?;
    let c = &cfg;
    let head = format!(
        "w4a16-load {dir}\n  hidden={} layers={} heads={}/{} head_dim={} ffn={} vocab={} interval={}\n  양자화=compressed-tensors pack-quantized int4 sym g{} (zp=8 상수) · 선형 {}개",
        c.hidden,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.ffn,
        c.vocab,
        c.full_interval,
        store.group(),
        store.n_lins()
    );
    let body = rep.summary();
    if rep.ok() {
        Ok(format!("{head}\n{body}  판정: 완전성 검증 통과"))
    } else {
        Err(format!("{head}\n{body}  판정: 검증 실패"))
    }
}

/// 참조 러너 — CPU qwen35 greedy. 프롬프트 prefill → 시드 1토큰 + `--n-predict`
/// 디코드(총 n_predict+1 토큰). 출력: `tokens: a,b,c`(단일 프롬프트).
fn reference(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-ref <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N] — 참조(CPU) greedy"
                .into(),
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prompt-tokens" => {
                let v = it.next().ok_or("--prompt-tokens requires ids")?;
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
            "--ctx" => {
                ctx = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ctx requires a number")?;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if prompts.is_empty() {
        return Err("at least one --prompt-tokens required".into());
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    let n = prompts.len();
    let mut eng = llm170_core::qwen35::Engine::new(model, n, ctx);
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (s, p) in prompts.iter().enumerate() {
        let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
        out[s].push(llm170_core::qwen35::greedy(&l));
    }
    for _ in 0..n_predict {
        for s in 0..n {
            let t = *out[s].last().expect("시드 토큰");
            let nt = eng.decode_greedy(s, t).map_err(|e| e.to_string())?;
            out[s].push(nt);
        }
    }
    let csv = |ts: &[u32]| {
        ts.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut r = format!("w4a16-ref {dir} — 참조(CPU) greedy\n");
    if n == 1 {
        r.push_str(&format!("tokens: {}\n", csv(&out[0])));
    } else {
        for (s, ts) in out.iter().enumerate() {
            r.push_str(&format!("tokens[{s}]: {}\n", csv(ts)));
        }
    }
    Ok(r)
}

/// 모듈 게이트 — 커널 vs core `dot_row_w4a16_lane` 비트 판정.
/// 형상은 스토어에서 자동 열거(distinct (n,k) → 첫 base), x는 결정적 생성.
///   w4a16-gemv <dir> [--rows N] [--seed X]          (t=1)
///   w4a16-gemm <dir> [--rows N] [--t N≤8] [--seed X]
fn gemm_gate(args: &[String], default_t: usize) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-gemv|w4a16-gemm <dir> [--rows N] [--t N] [--seed X] — 사용법: llm170 w4a16-gemv ../models/Qwen3.8-27B-W4A16-AutoRound".into(),
        );
    }
    let mut rows_limit = 4usize;
    let mut t = default_t;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut lin: Option<String> = None;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rows" => {
                rows_limit = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--rows requires a number")?;
            }
            "--t" => {
                t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--t requires a number")?;
            }
            "--seed" => {
                seed = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--seed requires a number")?;
            }
            "--lin" => {
                lin = Some(it.next().ok_or("--lin requires a name")?.clone());
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if t == 0 || t > 8 {
        return Err(format!("--t {t}: 1..=8"));
    }
    let g4 = llm170_backend_gpu::Gptq4::new()?;
    // 형상 열거 — --lin이면 model.w()(순열 사본 포함) 단일, 아니면 store 전수.
    let mut shapes: std::collections::BTreeMap<(usize, usize), String> =
        std::collections::BTreeMap::new();
    let mut lin_data: Option<(Vec<u8>, Vec<u8>, usize, bool)> = None;
    let store = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    if let Some(name) = &lin {
        let model = llm170_core::qwen35::Model::load(std::path::Path::new(&dir))
            .map_err(|e| e.to_string())?;
        let w = model
            .w(name)
            .ok_or_else(|| format!("--lin {name}: 무게 없음"))?;
        let s = w.aux.ok_or_else(|| format!("--lin {name}: split 아님"))?;
        shapes.insert((w.n_out as usize, w.n_in as usize), name.clone());
        lin_data = Some((w.data.to_vec(), s.to_vec(), w.group, w.scale_bf16));
    } else {
        for (base, n, k) in store.lin_shapes() {
            shapes.entry((n, k)).or_insert(base);
        }
    }
    let mut lines = Vec::new();
    let mut all_ok = true;
    let n_shape = shapes.len();
    for ((n, k), base) in &shapes {
        let (qb, sb) = if let Some((q, s, _, _)) = &lin_data {
            (q.as_slice(), s.as_slice())
        } else {
            (
                store
                    .tensor_slice(&format!("{base}.weight_packed"))
                    .ok_or_else(|| format!("{base}: weight_packed 슬라이스 부재"))?,
                store
                    .tensor_slice(&format!("{base}.weight_scale"))
                    .ok_or_else(|| format!("{base}: weight_scale 슬라이스 부재"))?,
            )
        };
        // 그룹·스케일 dtype — --lin이면 Weight 실측, 아니면 스토어 실측.
        let (group, bf16) = match &lin_data {
            Some((_, _, g, b)) => (*g, *b),
            None => (store.group(), store.scale_is_bf16(base)),
        };
        let r = rows_limit.min(*n);
        let q: Vec<u32> = qb[..r * (k / 8) * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let s: Vec<u16> = sb[..r * (k / group) * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        // 참조용 f32 스케일 — f16/bf16 디코드(둘 다 정확).
        let sf32: Vec<f32> = s
            .iter()
            .map(|&h| {
                if bf16 {
                    llm170_core::quant::deq::bf16_to_f32(h)
                } else {
                    llm170_core::quant::half_to_f32(h)
                }
            })
            .collect();
        // x — 결정적(splitmix64) f16 비트. 스케일이 작은 모델이라 ±1 균일.
        let mut rnd = SplitMix64::new(seed ^ ((*n as u64) << 32) ^ *k as u64);
        let x: Vec<u16> = (0..t * k)
            .map(|_| llm170_backend_gpu::f32_to_f16(rnd.next_pm1()))
            .collect();
        let got = g4.gemm(&x, t, &q, &s, r, *k, group, bf16)?;
        let z8 = vec![8u32; k / group];
        let mut mism = 0usize;
        let mut maxd = 0f64;
        for ti in 0..t {
            for o in 0..r {
                let qrow = &q[o * (k / 8)..(o + 1) * (k / 8)];
                let srow = &sf32[o * (k / group)..(o + 1) * (k / group)];
                let want = llm170_core::quant::dot_row_w4a16_lane_group(
                    qrow,
                    &z8,
                    srow,
                    &x[ti * k..(ti + 1) * k],
                    group,
                );
                let g = got[ti * r + o];
                if g.to_bits() != want.to_bits() {
                    mism += 1;
                    maxd = maxd.max((g as f64 - want as f64).abs());
                }
            }
        }
        let ok = mism == 0;
        all_ok &= ok;
        lines.push(format!(
            "  n={n:<6} k={k:<6} g{group}{} rows={r} t={t}  {}",
            if bf16 { "/bf16" } else { "/f16" },
            if ok {
                "PASS(비트일치)".to_string()
            } else {
                format!("FAIL mism={mism} maxdiff={maxd:.3e}")
            }
        ));
    }
    let name = if default_t == 1 {
        "w4a16-gemv"
    } else {
        "w4a16-gemm"
    };
    let head = format!(
        "{name} {dir}\n  형상 {n_shape}종 × rows≤{rows_limit} × t={t} — 커널 vs dot_row_w4a16_lane(비트 판정)"
    );
    let body = lines.join("\n");
    if all_ok {
        Ok(format!("{head}\n{body}\n  판정: 전 형상 비트일치"))
    } else {
        Err(format!("{head}\n{body}\n  판정: 불일치"))
    }
}

/// splitmix64 — 결정적 테스트 입력(rand 크레이트 금지 계약).
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_pm1(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// GPU 순차 디코드 — 64층 체인을 CUDA(호스트 스테이징)로 돌리고
/// 임베딩 행·최종 head만 CPU 참조 경로(골든과 동일 계급).
///   w4a16-gpu <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N]
///              [--no-head] [--bench-gemm <lin> <t> <reps>] [--moe-check]
///              [--h2d-bench <MB>]
fn gpu_run(args: &[String]) -> Result<String, String> {
    use llm170_backend_gpu::W4a16Dec;
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-gpu <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N] [--no-head] [--bench-gemm <lin> <t> <reps>] [--moe-check] [--h2d-bench <MB>] — 사용법: llm170 w4a16-gpu ../models/Qwen3.8-27B-W4A16-AutoRound --prompt-tokens 148678,65233,202419 --n-predict 8".into(),
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 8usize;
    let mut ctx = 1024usize;
    let mut no_head = false;
    let mut moe_check = false;
    let mut h2d_mb = 0usize;
    let mut bench: Option<(String, usize, usize)> = None; // (lin, t, reps)
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prompt-tokens" => {
                let v = it.next().ok_or("--prompt-tokens requires ids")?;
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
            "--ctx" => {
                ctx = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ctx requires a number")?;
            }
            "--no-head" => no_head = true,
            "--moe-check" => moe_check = true,
            "--h2d-bench" => {
                let mb = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--h2d-bench requires MB")?;
                h2d_mb = mb;
            }
            "--bench-gemm" => {
                let name = it.next().ok_or("--bench-gemm requires a name")?.clone();
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
    // 마이크로벤치(진단): 지정 선형의 t≥2 GEMM 반복 시간·실효 GB/s.
    if let Some((name, bt, reps)) = &bench {
        let (n, k) = model
            .w_raw(name)
            .map(|w| (w.n_out as usize, w.n_in as usize))
            .ok_or_else(|| format!("--bench-gemm {name}: 무게 없음"))?;
        let xf: Vec<f32> = vec![1.0f32; bt * k]; // f32 1.0 — 수치 무의미
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
    let staged = llm170_diag::flag::on("LLM170_STAGED");
    // GPU head — bf16 output.weight를 상주 업로드(--no-head·스테이징은 CPU 참조).
    let mut head_gpu = false;
    if !no_head && !staged && head.ty == llm170_core::wtype::WType::Bf16 {
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
            // [2026-10-09 개방 결함] MoE 배치 프리필 t≥8 NaN — MoE는 t=1 고정.
            let t = if hp.n_experts > 0 {
                1
            } else {
                (prompt.len() - i).min(8)
            };
            let mut rows: Vec<f32> = Vec::with_capacity(t * h);
            for &tok in &prompt[i..i + t] {
                rows.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
            }
            let last = i + t == prompt.len();
            if staged {
                for u in 0..t {
                    let row = &rows[u * h..(u + 1) * h];
                    let xn = dec.forward(0, row)?;
                    if last && u + 1 == t {
                        next = llm170_core::matmul::greedy_from(&head_logits(&xn));
                    }
                }
            } else if head_gpu {
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
    for _ in 0..n_predict {
        let row = model.embed_row(next).map_err(|e| e.to_string())?;
        if head_gpu {
            let lg = dec.forward_device_head(0, &row)?;
            next = llm170_core::matmul::greedy_from(&lg);
        } else {
            let xn = if staged {
                dec.forward(0, &row)?
            } else {
                dec.forward_device(0, &row)?
            };
            next = llm170_core::matmul::greedy_from(&head_logits(&xn));
        }
        out.push(next);
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
        "w4a16-gpu {dir} — GPU 체인({}{}) · {}\n  업로드 {upload_ms:.0}ms · 생성 {}토큰 {gen_ms:.0}ms ({:.1}ms/토큰)\n tokens: {csv}",
        if staged {
            "호스트 스테이징"
        } else {
            "디바이스 상주"
        },
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
