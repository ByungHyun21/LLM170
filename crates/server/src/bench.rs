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

/// [검수 통합 2026-10-14] 토큰 스트림 지문(FNV-1a) — 빌드 간 np 스트림 대조용.
fn fnv1a_u32(v: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn cmd_bench(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let model: Option<String> = ma.model.clone();
    let mut pps = vec![512usize];
    let mut tgs = vec![128usize];
    let mut nps: Vec<usize> = Vec::new();
    let mut reps = 1usize;
    let mut ctx = 4096usize;
    // [검수 통합 2026-10-14] np 단일 경로 등가 검사 폭(K토큰) — 0=끔.
    // 기본 64: np 측정 런이 곧 배치 불변성 판정(해시는 상시 출력).
    let mut np_check = 64usize;
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
                Some(v) if v.iter().all(|&n| (2..=16).contains(&n)) => nps = v,
                _ => return usage_err_bench("--np requires N[,N..] (2..=16)"),
            },
            "--np-check" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => np_check = v.min(8192),
                None => return usage_err_bench("--np-check requires a number"),
            },
            "--reps" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => reps = v.clamp(1, 20),
                None => return usage_err_bench("--reps requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => ctx = v,
                None => return usage_err_bench("--ctx requires a number"),
            },
            // [P6 2026-10-15] --moe 강제(serve와 동일 파서) — C20: auto 상주
            // 판정이 환경으로 뒤집히지 않게 스트리밍 레짐 실험을 고정한다.
            "--moe" => match it.next().and_then(|v| crate::engine::parse_moe_policy(v)) {
                Some(p) => {
                    crate::engine::MOE_POLICY.store(p as u64, std::sync::atomic::Ordering::Relaxed)
                }
                None => return usage_err_bench("--moe requires auto|resident|streaming"),
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
        let mut eng = GpuEngine::load(&model_path, max_np, ctx, None, None, None)
            .map_err(|e| e.to_string())?;
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
        // [검수 통합 2026-10-14] 슬롯별 스트림 해시 상시 + 단일 경로 등가
        // 검사(--np-check K, 기본 64) — 측정 런이 곧 배치 불변성 판정. 해시는
        // 빌드 간 A/B 대조용(경로 변경이 토큰을 바꾸면 해시 불일치).
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
            let mut streams: Vec<Vec<u32>> = vec![Vec::new(); np];
            let t0 = Instant::now();
            let mut total = 0usize;
            while total < max_tg * np {
                toks = eng
                    .decode_np_greedy(&seqs, &toks)
                    .map_err(|e| e.to_string())?;
                for (s, &tk) in toks.iter().enumerate() {
                    streams[s].push(tk);
                }
                total += np;
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            lines.push(format!(
                "np{np}(tg{max_tg}) | 합산 {total}토큰 | {ms:9.1} ms | {:8.2} t/s",
                total as f64 / (ms / 1e3)
            ));
            let hh: Vec<String> = streams
                .iter()
                .enumerate()
                .map(|(s, v)| format!("s{s}=0x{:016x}", fnv1a_u32(v)))
                .collect();
            lines.push(format!("np{np}-hash | {}", hh.join(" ")));
            if np_check > 0 {
                let k = np_check.min(max_tg);
                let mut bad: Option<(usize, usize)> = None;
                'slots: for (s, stream) in streams.iter().enumerate() {
                    eng.reset_seq(s).map_err(|e| e.to_string())?;
                    let l = eng
                        .prefill(s, &prompt[..64.min(max_pp)])
                        .map_err(|e| e.to_string())?;
                    // 배치 스트림의 선행 토큰 = 프리필 그리디(루프 밖 소비) —
                    // 단일 경로도 같은 토큰을 먹인 뒤의 출력과 비교(오프바이원 주의).
                    let mut next = llm170_core::qwen35::greedy(&l);
                    for (i, &bt) in stream.iter().take(k).enumerate() {
                        let single = eng.decode_greedy(s, next).map_err(|e| e.to_string())?;
                        if bt != single {
                            bad = Some((s, i));
                            break 'slots;
                        }
                        next = single;
                    }
                }
                match bad {
                    None => lines.push(format!("np{np}-check | 단일 경로 등가 OK (K={k})")),
                    Some((s, i)) => lines.push(format!(
                        "np{np}-check | FAIL 불일치 slot{s} @idx{i} — 배치 불변성 위반"
                    )),
                }
            }
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
