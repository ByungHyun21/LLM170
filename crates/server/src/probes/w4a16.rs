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
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if t == 0 || t > 8 {
        return Err(format!("--t {t}: 1..=8"));
    }
    let store = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let g4 = llm170_backend_gpu::Gptq4::new()?;
    // 형상 자동 열거 — distinct (n,k) → 첫 base.
    let mut shapes: std::collections::BTreeMap<(usize, usize), String> =
        std::collections::BTreeMap::new();
    for (base, n, k) in store.lin_shapes() {
        shapes.entry((n, k)).or_insert(base);
    }
    let mut lines = Vec::new();
    let mut all_ok = true;
    let n_shape = shapes.len();
    for ((n, k), base) in &shapes {
        let qb = store
            .tensor_slice(&format!("{base}.weight_packed"))
            .ok_or_else(|| format!("{base}: weight_packed 슬라이스 부재"))?;
        let sb = store
            .tensor_slice(&format!("{base}.weight_scale"))
            .ok_or_else(|| format!("{base}: weight_scale 슬라이스 부재"))?;
        let r = rows_limit.min(*n);
        let q: Vec<u32> = qb[..r * (k / 8) * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let s: Vec<u16> = sb[..r * (k / 128) * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        // x — 결정적(splitmix64) f16 비트. 스케일이 작은 모델이라 ±1 균일.
        let mut rnd = SplitMix64::new(seed ^ ((*n as u64) << 32) ^ *k as u64);
        let x: Vec<u16> = (0..t * k).map(|_| f32_to_f16(rnd.next_pm1())).collect();
        let got = g4.gemm(&x, t, &q, &s, r, *k)?;
        let z8 = vec![8u32; k / 128];
        let mut mism = 0usize;
        let mut maxd = 0f64;
        for ti in 0..t {
            for o in 0..r {
                let qrow = &q[o * (k / 8)..(o + 1) * (k / 8)];
                let srow = &s[o * (k / 128)..(o + 1) * (k / 128)];
                let want = llm170_core::quant::dot_row_w4a16_lane(
                    qrow,
                    &z8,
                    srow,
                    &x[ti * k..(ti + 1) * k],
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
            "  n={n:<6} k={k:<6} rows={r} t={t}  {}",
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

/// f32 → f16 비트 (정규수 round-to-nearest — 게이트 입력 전용).
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let man = b & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let half_man = man >> 13;
    let round = (man >> 12) & 1;
    let m = half_man + round;
    let (m, e) = if m & 0x400 != 0 {
        (m & 0x3FF, exp + 1)
    } else {
        (m, exp)
    };
    sign | ((e as u16) << 10) | (m as u16)
}
