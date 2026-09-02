//! bench — llama-bench 규격 PP/TG 측정 (2026-09-02).
//!
//! PP: 합성 pp 토큰 prefill 시간 → t/s. TG: prefill 후 tg 토큰 디코드 시간 → t/s.
//! 토큰은 수제 LCG(seed 0x1234_5678, 관례) — rand 금지. 워밍업 1회 + reps 회 측정.
//! qwen35(MTP --spec 포함)·qwen4exp(--frame 포함) 양쪽 대응.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

fn usage_err_bench(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n사용법: llm170 bench --model <gguf> [--pp N] [--tg N] [--reps N] [--ctx N] [--backend cpu|gpu] [--gpu-runtime hip|vulkan] [--spec k]");
    ExitCode::from(2)
}

pub fn cmd_bench(args: &[String]) -> ExitCode {
    let mut model: Option<PathBuf> = None;
    let mut pp = 512usize;
    let mut tg = 128usize;
    let mut reps = 1usize;
    let mut ctx = 4096usize;
    let mut backend = "cpu".to_string();
    let mut gpu_runtime = std::env::var("LLM170_GPU_RUNTIME").unwrap_or_else(|_| "hip".into());
    let mut spec_k = 0usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => match it.next() {
                Some(v) => model = Some(PathBuf::from(v)),
                None => return usage_err_bench("--model requires a path"),
            },
            "--pp" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => pp = v.clamp(8, 65536),
                None => return usage_err_bench("--pp requires a number"),
            },
            "--tg" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => tg = v.clamp(1, 4096),
                None => return usage_err_bench("--tg requires a number"),
            },
            "--reps" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => reps = v.clamp(1, 20),
                None => return usage_err_bench("--reps requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => ctx = v,
                None => return usage_err_bench("--ctx requires a number"),
            },
            "--backend" => match it.next() {
                Some(v) if v == "cpu" || v == "gpu" => backend = v.clone(),
                Some(v) => return usage_err_bench(&format!("--backend: cpu|gpu (got {v})")),
                None => return usage_err_bench("--backend requires cpu|gpu"),
            },
            "--gpu-runtime" => match it.next() {
                Some(v) if v == "hip" || v == "vulkan" => gpu_runtime = v.clone(),
                Some(v) => return usage_err_bench(&format!("--gpu-runtime: hip|vulkan (got {v})")),
                None => return usage_err_bench("--gpu-runtime requires hip|vulkan"),
            },
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) if k >= 1 && k <= 8 => spec_k = k,
                _ => return usage_err_bench("--spec requires k in 1..=8"),
            },
            other => return usage_err_bench(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = model else {
        return usage_err_bench("--model required");
    };
    if pp + tg + 16 >= ctx {
        return usage_err_bench(&format!("ctx({ctx}) too small for pp({pp})+tg({tg})"));
    }

    // 수제 LCG 합성 토큰
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % 200_000) as u32
    };
    let prompt: Vec<u32> = (0..pp).map(|_| lcg()).collect();

    // 아키텍처 판별 (ENOENT 재시도 관례)
    let mut arch: Option<String> = None;
    for _ in 0..5 {
        arch = llm170_gguf::GgufFile::open(&model_path)
            .ok()
            .and_then(|g| g.arch().map(|s| s.to_string()));
        if arch.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    let res: Result<Vec<String>, String> = (|| {
        let mut lines = Vec::new();
        if arch.as_deref() == Some("qwen4exp") {
            let m = llm170_core::qwen4exp::Model4::load(&model_path)
                .map_err(|e| e.to_string())?;
            let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, 1, ctx);
            let _ = (&backend, &gpu_runtime);
            let eos = eng.model.eos;
            // 워밍업 1회 (스크래치 풀·가속기 warm)
            {
                let _ = eng.prefill(0, &prompt[..64.min(pp)]).map_err(|e| e.to_string())?;
                let l = eng.decode1(0, 1u32).map_err(|e| e.to_string())?;
                let _ = llm170_core::model::greedy(&l);
            }
            let frame_on = std::env::var("LLM170_FRAME").is_ok_and(|v| v != "0");
            for r in 0..reps {
                eng.reset_states();
                let t0 = Instant::now();
                let l = eng.prefill(0, &prompt).map_err(|e| e.to_string())?;
                let pp_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut next = llm170_core::model::greedy(&l);
                // TG — 프레임 경로는 decode1 내부 분기
                let t1 = Instant::now();
                let mut n_gen = 0usize;
                let mut step = 0usize;
                while n_gen < tg {
                    let l = eng.decode1(0, next).map_err(|e| e.to_string())?;
                    next = llm170_core::model::greedy(&l);
                    n_gen += 1;
                    step += 1;
                    if next == eos {
                        break;
                    }
                }
                let tg_ms = t1.elapsed().as_secs_f64() * 1e3;
                let fr = if frame_on { " frame" } else { "" };
                lines.push(format!(
                    "pp{pp}{fr} | rep{r} | {pp_ms:8.1} ms | {:7.2} t/s",
                    pp as f64 / (pp_ms / 1e3)
                ));
                lines.push(format!(
                    "tg{tg}{fr} | rep{r} | {tg_ms:8.1} ms | {:7.2} t/s (steps {step}, gen {n_gen})",
                    n_gen as f64 / (tg_ms / 1e3)
                ));
            }
        } else {
            let m = llm170_core::model::Model::load(&model_path)
                .map_err(|e| e.to_string())?;
            let mut eng = llm170_core::model::Engine::new(m, 1, ctx);
            if std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true) {
                crate::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
            }
            let _ = (&backend, &gpu_runtime);
            let has_mtp = eng.has_mtp();
            let spec_desc = if spec_k > 0 && has_mtp {
                format!(" spec{spec_k}")
            } else {
                String::new()
            };
            // 워밍업
            {
                let _ = eng.prefill(0, &prompt[..64.min(pp)]).map_err(|e| e.to_string())?;
                let l = eng.decode(&[0], &[1u32]).map_err(|e| e.to_string())?;
                let _ = llm170_core::model::greedy(&l[0]);
            }
            for r in 0..reps {
                eng.reset_states();
                let t0 = Instant::now();
                let l = eng.prefill(0, &prompt).map_err(|e| e.to_string())?;
                let pp_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut next = llm170_core::model::greedy(&l);
                let t1 = Instant::now();
                let mut n_gen = 0usize;
                let mut fwd = 0usize;
                if spec_k > 0 && has_mtp {
                    while n_gen < tg {
                        let (toks, tf) = eng
                            .spec_step(0, next, spec_k)
                            .map_err(|e| e.to_string())?;
                        fwd += tf;
                        for &t in &toks {
                            if n_gen >= tg {
                                break;
                            }
                            next = t;
                            n_gen += 1;
                        }
                    }
                } else {
                    while n_gen < tg {
                        next = eng.decode_greedy(0, next).map_err(|e| e.to_string())?;
                        n_gen += 1;
                        fwd += 1;
                        if next == 248044 {
                            break;
                        }
                    }
                }
                let tg_ms = t1.elapsed().as_secs_f64() * 1e3;
                lines.push(format!(
                    "pp{pp}{spec_desc} | rep{r} | {pp_ms:8.1} ms | {:7.2} t/s",
                    pp as f64 / (pp_ms / 1e3)
                ));
                lines.push(format!(
                    "tg{tg}{spec_desc} | rep{r} | {tg_ms:8.1} ms | {:7.2} t/s (fwd {fwd}, gen {n_gen}, {:.2} tok/fwd)",
                    n_gen as f64 / (tg_ms / 1e3),
                    n_gen as f64 / fwd.max(1) as f64
                ));
            }
        }
        Ok(lines)
    })();

    match res {
        Ok(lines) => {
            println!("model            | test         |       time |        rate");
            println!("-----------------+--------------+-----------+------------");
            let short = model_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let bemark = if backend == "gpu" {
                format!("gpu:{gpu_runtime}")
            } else {
                "cpu".into()
            };
            for l in &lines {
                // "pp512 | rep0 | ..." → 앞부분 파싱해 정렬
                println!("{:16} | {}", format!("{short}[{bemark}]"), l);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
