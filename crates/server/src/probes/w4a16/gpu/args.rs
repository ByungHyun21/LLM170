//! probes/w4a16/gpu.rs에서 분해(순수 이동 — 코드 무변).
use crate::probes::{arg, arg_str};

/// 파싱된 w4a16-gpu 옵션 묶음 — [R9] 순수 이동(기존 gpu_run 로컬 변수 그대로).
pub(super) struct GpuArgs {
    pub(super) prompts: Vec<Vec<u32>>,
    pub(super) n_predict: usize,
    pub(super) spec_k: usize,    // [A-1] 스페큘러티브 초안 상한(0=off).
    pub(super) spec_check: bool, // [A-1 진단] 검증 vs t=1 디코드 토큰 대조.
    pub(super) ctx: usize,
    pub(super) no_head: bool,
    pub(super) moe_check: bool,
    pub(super) plain_check: bool,
    pub(super) bench_ew: bool,
    pub(super) ew_gelu_check: bool,
    pub(super) moe_pes_check: bool,
    pub(super) prep_g4_check: bool,
    pub(super) attn_g4_check: bool,
    pub(super) mma_smoke: bool,
    pub(super) bench_gemv: Option<(String, usize)>,
    pub(super) bench_gemv_t: Option<(String, usize, usize)>,
    pub(super) bench_plain: Option<(String, usize, usize)>,
    pub(super) mma_diff: Option<(String, usize)>,
    pub(super) marlin_diff: Option<(String, usize)>, // [marlin S2] A/B 재패킹 게이트
    pub(super) marlin_selfcheck: Option<(usize, usize, usize, bool)>, // (t, k, n, m16)
    pub(super) marlin_selfcheck32: Option<(usize, usize, usize)>, // [marlin g32] bf16 자체점검
    pub(super) marlin_bench: Vec<(String, usize, usize, String)>, // [marlin S2] mma 대비 벤치
    /// [2.A-② 2026-10-13] marlin-bench 콜드 모드(L2 플러시 — HBM 조건 근사).
    pub(super) marlin_cold: bool,
    /// [split-k marlin 디코드 2026-10-17] t=1 sk 커널 벤치(이름, reps).
    pub(super) decode_bench: Vec<(String, usize)>,
    /// [sk] 콜드 모드(L2 플러시 — marlin_cold와 동일 조건).
    pub(super) decode_cold: bool,
    /// [sk] 커널 격리 자체점검(합성 — 모델 무게 불요).
    pub(super) decode_selfcheck: bool,
    pub(super) marlin_ref: Option<(String, usize)>, // [marlin S2] 실무게 64행 참조 대조
    pub(super) attn_fa2: Vec<(usize, usize, u32)>,  // [FA2] A/B (t, layer, pos0)
    pub(super) attn_fa2_bench: Vec<(usize, usize, usize, u32)>, // [FA2] (t, layer, reps, pos0)
    pub(super) mode_ref: u8,                        // 0 실값, 1 s→1, 2 x→1, 3 둘 다 1
    pub(super) kcap: usize,                         // ref k 상한(이분탐색용)
    pub(super) moe_topk_check: Option<(usize, usize, usize)>,
    pub(super) h2d_mb: usize,
    pub(super) bench: Option<(String, usize, usize)>, // (lin, t, reps)
}

/// 인자 파싱 — [R9] 순수 이동(gpu_run 앞부분).
pub(super) fn parse_args(args: &[String]) -> Result<GpuArgs, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-gpu <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N] [--no-head] [--bench-gemm <lin> <t> <reps>] [--moe-check] [--h2d-bench <MB>] [--device <0,1>] — 사용법: llm170 w4a16-gpu ../models/Qwen3.8-27B-W4A16-AutoRound --prompt-tokens 148678,65233,202419 --n-predict 8".into(),
        );
    }
    let mut a = GpuArgs {
        prompts: Vec::new(),
        n_predict: 8usize,
        spec_k: 0usize,
        spec_check: false,
        ctx: 1024usize,
        no_head: false,
        moe_check: false,
        plain_check: false,
        bench_ew: false,
        ew_gelu_check: false,
        moe_pes_check: false,
        prep_g4_check: false,
        attn_g4_check: false,
        mma_smoke: false,
        bench_gemv: None,
        bench_gemv_t: None,
        bench_plain: None,
        mma_diff: None,
        marlin_diff: None,
        marlin_selfcheck: None,
        marlin_selfcheck32: None,
        marlin_bench: Vec::new(),
        marlin_cold: false,
        marlin_ref: None,
        decode_bench: Vec::new(),
        decode_cold: false,
        decode_selfcheck: false,
        attn_fa2: Vec::new(),
        attn_fa2_bench: Vec::new(),
        mode_ref: 0,
        kcap: usize::MAX,
        moe_topk_check: None,
        h2d_mb: 0usize,
        bench: None,
    };
    let mut it = args.iter().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
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
                a.prompts.push(ids);
            }
            "--n-predict" => {
                a.n_predict = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--n-predict requires a number")?;
            }
            "--spec" => {
                a.spec_k = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--spec requires a number")?;
            }
            "--spec-check" => {
                a.spec_check = true;
            }
            "--ctx" => {
                a.ctx = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ctx requires a number")?;
            }
            "--no-head" => a.no_head = true,
            "--moe-check" => a.moe_check = true,
            "--plain-gemm-check" => a.plain_check = true,
            "--bench-ew" => a.bench_ew = true,
            "--ew-gelu-check" => a.ew_gelu_check = true,
            "--moe-pes-check" => a.moe_pes_check = true,
            "--prep-g4-check" => a.prep_g4_check = true,
            "--attn-g4-check" => a.attn_g4_check = true,
            "--mma-smoke" => a.mma_smoke = true,
            "--h2d-bench" => {
                let mb = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--h2d-bench requires MB")?;
                a.h2d_mb = mb;
            }
            "--moe-topk-check" => {
                let t = it.next().and_then(|v| v.parse().ok()).unwrap_or(32);
                let n = it.next().and_then(|v| v.parse().ok()).unwrap_or(512);
                let k = it.next().and_then(|v| v.parse().ok()).unwrap_or(8);
                a.moe_topk_check = Some((t, n, k));
            }
            "--mma-diff" => {
                let name = arg(&mut it, "--mma-diff requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--mma-diff requires t")?;
                a.mma_diff = Some((name, t));
            }
            "--marlin-diff" => {
                let name = arg(&mut it, "--marlin-diff requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-diff requires t")?;
                a.marlin_diff = Some((name, t));
            }
            "--marlin-selfcheck32" => {
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-selfcheck32 requires t")?;
                let k = it.next().and_then(|v| v.parse().ok()).unwrap_or(128usize);
                let n = it.next().and_then(|v| v.parse().ok()).unwrap_or(128usize);
                a.marlin_selfcheck32 = Some((t, k, n));
            }
            "--attn-fa2-bench" => {
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--attn-fa2-bench requires t")?;
                let lay = it.next().and_then(|v| v.parse().ok()).unwrap_or(0usize);
                let reps = it.next().and_then(|v| v.parse().ok()).unwrap_or(8usize);
                let pos0 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0u32);
                a.attn_fa2_bench.push((t, lay, reps, pos0));
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
                a.attn_fa2.push((t, lay, p0));
            }
            "--marlin-ref" => {
                let name = arg(&mut it, "--marlin-ref requires a name")?.to_string();
                let t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--marlin-ref requires t")?;
                a.mode_ref = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                a.kcap = it.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
                a.marlin_ref = Some((name, t));
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
                a.marlin_selfcheck = Some((t, k, n, m16));
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
                a.marlin_bench.push((name, t, reps, kern));
            }
            "--marlin-cold" => a.marlin_cold = true,
            "--decode-bench" => {
                let name = arg(&mut it, "--decode-bench requires a name")?.to_string();
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--decode-bench requires reps")?;
                a.decode_bench.push((name, reps));
            }
            "--decode-cold" => a.decode_cold = true,
            "--decode-selfcheck" => a.decode_selfcheck = true,
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
                a.bench_plain = Some((name, t, reps));
            }
            "--bench-gemv" => {
                let name = arg(&mut it, "--bench-gemv requires a name")?.to_string();
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemv requires reps")?;
                a.bench_gemv = Some((name, reps));
            }
            "--bench-gemv-all" => {
                let reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench-gemv-all requires reps")?;
                a.bench_gemv = Some(("ALL".to_string(), reps));
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
                a.bench = Some((name, t, reps));
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
                a.bench_gemv_t = Some((name, t, reps));
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    Ok(a)
}
