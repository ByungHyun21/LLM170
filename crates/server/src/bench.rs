//! bench — llama-bench 규격 PP/TG/NP 측정 (2026-10-11 복원 — W4A16/CUDA 단일
//! 트랙용 재작성).
//!
//! 원형(구 트랙 bench CLI — 이미 삭제된 커밋)의 의미론 승계: 합성 LCG 토큰(seed 0x1234_5678 —
//! 크레이트 관례, rand 금지), 워밍업 1회 + reps회 측정, llama-bench 스타일
//! 출력. **엔진 직접**(GpuEngine — HTTP 무경유). PP = prefill 벽(로그짓 전),
//! TG = 프리필 후 greedy 디코드, NP = N슬롯 배치 디코드 합산(슬롯별 짧은
//! 프리필 후 배치 체인). eos 무시(ignore-eos 관례 — 조기 종료 방지).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use crate::gpu_engine::GpuEngine;

fn usage_err_bench(msg: &str) -> ExitCode {
    eprintln!(
        "error: {msg}\n사용법: llm170 bench --model <w4a16_dir> [--pp N[,N..]] [--tg N[,N..]] [--np N[,N..]] [--reps N] [--ctx N]"
    );
    ExitCode::from(2)
}

fn parse_list(v: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in v.split(',') {
        out.push(part.parse::<usize>().ok()?);
    }
    if out.is_empty() { None } else { Some(out) }
}

pub fn cmd_bench(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let model: Option<String> = ma.model.clone();
    let mut pps = vec![512usize];
    let mut tgs = vec![128usize];
    let mut nps: Vec<usize> = Vec::new();
    let mut reps = 1usize;
    let mut ctx = 4096usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pp" => match it.next().and_then(|v| parse_list(v)) {
                Some(v) if v.iter().all(|&n| (8..=65536).contains(&n)) => pps = v,
                _ => return usage_err_bench("--pp requires N[,N..] (8..=65536)"),
            },
            "--tg" => match it.next().and_then(|v| parse_list(v)) {
                Some(v) if v.iter().all(|&n| (1..=4096).contains(&n)) => tgs = v,
                _ => return usage_err_bench("--tg requires N[,N..] (1..=4096)"),
            },
            "--np" => match it.next().and_then(|v| parse_list(v)) {
                Some(v) if v.iter().all(|&n| (2..=8).contains(&n)) => nps = v,
                _ => return usage_err_bench("--np requires N[,N..] (2..=8)"),
            },
            "--reps" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => reps = v.clamp(1, 20),
                None => return usage_err_bench("--reps requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => ctx = v,
                None => return usage_err_bench("--ctx requires a number"),
            },
            other => return usage_err_bench(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = model.map(PathBuf::from) else {
        return usage_err_bench("--model required");
    };
    let max_pp = *pps.iter().max().unwrap_or(&512);
    let max_tg = *tgs.iter().max().unwrap_or(&128);
    let max_np = nps.iter().copied().max().unwrap_or(1).max(1);
    if max_pp + max_tg + 16 >= ctx {
        return usage_err_bench(&format!(
            "ctx({ctx}) too small for pp({max_pp})+tg({max_tg})"
        ));
    }

    // 수제 LCG 합성 토큰(관례 seed — rand 금지).
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % 200_000) as u32
    };
    let prompt: Vec<u32> = (0..max_pp + 8).map(|_| lcg()).collect();

    let res: Result<Vec<String>, String> = (|| {
        let mut lines = Vec::new();
        let mut eng =
            GpuEngine::load(&model_path, max_np, ctx, None, None).map_err(|e| e.to_string())?;
        // 워밍업 1회(짧은 프리필+디코드 — 스크래치·그래프 warm).
        {
            eng.reset_seq(0).map_err(|e| e.to_string())?;
            let _ = eng
                .prefill(0, &prompt[..64.min(max_pp)])
                .map_err(|e| e.to_string())?;
            let _ = eng.decode_greedy(0, prompt[0]).map_err(|e| e.to_string())?;
        }
        // pp별 워밍업 — 청크 스크래치(ensure) 재할당·그래프 캡처를 계측 밖으로.
        for &pp in &pps {
            eng.reset_seq(0).map_err(|e| e.to_string())?;
            let _ = eng.prefill(0, &prompt[..pp]).map_err(|e| e.to_string())?;
        }
        for &pp in &pps {
            for r in 0..reps {
                eng.reset_seq(0).map_err(|e| e.to_string())?;
                let t0 = Instant::now();
                let l = eng.prefill(0, &prompt[..pp]).map_err(|e| e.to_string())?;
                let pp_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut next = llm170_core::qwen35::greedy(&l);
                for &tg in &tgs {
                    let t1 = Instant::now();
                    let mut n = 0usize;
                    while n < tg {
                        next = eng.decode_greedy(0, next).map_err(|e| e.to_string())?;
                        n += 1;
                    }
                    let tg_ms = t1.elapsed().as_secs_f64() * 1e3;
                    lines.push(format!(
                        "pp{pp} | rep{r} | {pp_ms:9.1} ms | {:8.2} t/s",
                        pp as f64 / (pp_ms / 1e3)
                    ));
                    lines.push(format!(
                        "tg{tg}@pp{pp} | rep{r} | {tg_ms:9.1} ms | {:8.2} t/s",
                        n as f64 / (tg_ms / 1e3)
                    ));
                }
            }
        }
        // NP — 슬롯별 짧은 프리필(비계측) 후 배치 디코드 합산.
        for &np in &nps {
            let mut toks: Vec<u32> = Vec::with_capacity(np);
            for s in 0..np {
                eng.reset_seq(s).map_err(|e| e.to_string())?;
                let l = eng
                    .prefill(s, &prompt[..64.min(max_pp)])
                    .map_err(|e| e.to_string())?;
                toks.push(llm170_core::qwen35::greedy(&l));
            }
            let seqs: Vec<usize> = (0..np).collect();
            let t0 = Instant::now();
            let mut total = 0usize;
            while total < max_tg * np {
                toks = eng
                    .decode_np_greedy(&seqs, &toks)
                    .map_err(|e| e.to_string())?;
                total += np;
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            lines.push(format!(
                "np{np}(tg{max_tg}) | 합산 {total}토큰 | {ms:9.1} ms | {:8.2} t/s",
                total as f64 / (ms / 1e3)
            ));
        }
        Ok(lines)
    })();
    match res {
        Ok(lines) => {
            for l in lines {
                println!("{l}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
