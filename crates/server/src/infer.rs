//! infer — qwen35/qwen4exp greedy 추론 CLI (main.rs에서 이관, plans/35 P7).
//! JSONL {"seq","pos","token","text"} 스트림 출력.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::{apply_mode, parse_ids_ref, usage_err};

pub(crate) fn cmd_infer(args: &[String]) -> ExitCode {
    let mut model: Option<PathBuf> = None;
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let mut backend = "cpu".to_string();
    let mut gpu_runtime = std::env::var("LLM170_GPU_RUNTIME").unwrap_or_else(|_| "hip".into());
    let mut mode: Option<llm170_core::mode::Mode> = None;
    let mut spec_k: Option<usize> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => match it.next() {
                Some(v) => model = Some(PathBuf::from(v)),
                None => return usage_err("--model requires a path"),
            },
            "--prompt-tokens" => match it.next() {
                Some(v) => match parse_ids(v) {
                    Ok(ids) if !ids.is_empty() => prompts.push(ids),
                    Ok(_) => return usage_err("empty prompt"),
                    Err(e) => return usage_err(&format!("bad tokens: {e}")),
                },
                None => return usage_err("--prompt-tokens requires ids"),
            },
            "--n-predict" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => n_predict = n,
                None => return usage_err("--n-predict requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => ctx = n,
                None => return usage_err("--ctx requires a number"),
            },
            "--backend" => match it.next() {
                Some(v) if v == "cpu" || v == "gpu" => backend = v.clone(),
                Some(v) => return usage_err(&format!("--backend: cpu|gpu (got {v})")),
                None => return usage_err("--backend requires cpu|gpu"),
            },
            "--gpu-runtime" => match it.next() {
                Some(v) if v == "hip" || v == "vulkan" => gpu_runtime = v.clone(),
                Some(v) => return usage_err(&format!("--gpu-runtime: hip|vulkan (got {v})")),
                None => return usage_err("--gpu-runtime requires hip|vulkan"),
            },
            "--mode" => match it.next().map(String::as_str).and_then(llm170_core::mode::Mode::from_str) {
                Some(m) => mode = Some(m),
                None => return usage_err("--mode requires universal|cmp-stock|cmp-unlocked"),
            },
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) if k >= 1 && k <= 8 => spec_k = Some(k),
                _ => return usage_err("--spec requires k in 1..=8"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }

    let Some(model_path) = model else {
        return usage_err("--model required");
    };
    if let Some(m) = mode {
        apply_mode(m);
    }
    if prompts.is_empty() {
        return usage_err("at least one --prompt-tokens required");
    }
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    if max_prompt + n_predict + 8 >= ctx {
        return usage_err(&format!(
            "ctx({ctx}) too small for prompt({max_prompt})+n_predict({n_predict})"
        ));
    }

    llm170_profiler::reset();
    let t_start = std::time::Instant::now();
    // 아키텍처 판별 → qwen4exp 전용 엔진 분기.
    // ENOENT 윈도우 대기 (LLM170_OPEN_WAIT_SECS) — 판별 실패시 재시도.
    let wait_secs: u64 = std::env::var("LLM170_OPEN_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut arch: Option<String> = None;
    for _ in 0..=wait_secs {
        arch = llm170_gguf::GgufFile::open(&model_path)
            .ok()
            .and_then(|g| g.arch().map(|s| s.to_string()));
        if arch.is_some() {
            break;
        }
        if wait_secs == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    if arch.as_deref() == Some("qwen4exp") {
        return run_q4_infer(&model_path, &prompts, n_predict, ctx, &backend, &gpu_runtime);
    }
    let engine_res = llm170_core::model::Model::load(&model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::model::Engine::new(m, n, ctx);
            if spec_k.is_some() {
                eng.mtp_wanted = true; // 스펙 의도 — prefill 훅 활성 (plans/22)
            }
            if gpu_runtime == "vulkan" {
                // plans/29: VkDecoder(GPU 상주)가 vulkan 기본 — 헤드 n_vocab·
                // WG 청크 수정으로 llama 패리티 확보(19f68bc). VkAcc 복원:
                // LLM170_VK_ACC=1.
                if std::env::var_os("LLM170_VK_ACC").is_some() {
                    match llm170_backend_gpu::rawvk::gemv::VkAcc::new() {
                        Ok(acc) => {
                            eng = eng.with_acc(std::sync::Arc::new(acc));
                            eprintln!("# backend: gpu (vulkan VkAcc)");
                        }
                        Err(e) => eprintln!("vk-acc: {e} (CPU로 진행)"),
                    }
                } else {
                    match llm170_backend_gpu::inject_rawvk(&mut eng) {
                        Ok(()) => eprintln!("# backend: gpu (vulkan VkDecoder)"),
                        Err(e) => eprintln!("vk-decoder: {e} (VkAcc로 진행)"),
                    }
                }
            } else if std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true) {
                llm170_backend_gpu::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
            }
            if backend == "gpu" && gpu_runtime != "vulkan" {
                eprintln!("# backend: gpu (raw hip)");
            }
            let eos = 248044u32;
            // prefill (시퀀스별 — GDN chunked 경로)
            let mut last_logits = Vec::with_capacity(n);
            let dbg_topk = std::env::var("LLM170_DEBUG_TOPK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok());
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                if let Some(k) = dbg_topk {
                    let mut idx: Vec<usize> = (0..l.len()).collect();
                    idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                    let top: Vec<String> = idx[..k.min(l.len())]
                        .iter()
                        .map(|&i| format!("{}:{:.4}", i, l[i]))
                        .collect();
                    eprintln!("topk seq{s}: {}", top.join(" "));
                }
                last_logits.push(l);
            }
            let mut finished = vec![false; n];
            let mut gen_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut next: Vec<u32> = last_logits
                .iter()
                .map(|l| llm170_core::model::greedy(l))
                .collect();
            for s in 0..n {
                emit(s, prompts[s].len() as u32, next[s], &eng);
                gen_tokens[s].push(next[s]);
                if next[s] == eos {
                    finished[s] = true;
                }
            }
            // 스펙 디코드 (06) — --spec k 지정 시 MTP 체인 draft·연쇄 수용.
            let spec_k: usize = spec_k.unwrap_or(0);
            let has_mtp = eng.has_mtp();
            let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            if spec_k > 0 && has_mtp && n > 1 && std::env::var_os("LLM170_SPEC_GPU").is_some() {
                // np×spec 병합 (plans/18)
                let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
                let mut accepted_total = 0usize;
                let mut cycles = 0usize;
                let mut min_gen = gen_tokens[0].len();
                for g in gen_tokens.iter() {
                    min_gen = min_gen.min(g.len());
                }
                while min_gen <= n_predict {
                    let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                    if active.is_empty() {
                        break;
                    }
                    let nexts: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                    let acc = eng
                        .spec_step_multi(&active, &nexts, spec_k)
                        .map_err(|e| e.to_string())?;
                    cycles += 1;
                    let mut any = false;
                    for (i, &s) in active.iter().enumerate() {
                        for &t in &acc[i] {
                            if gen_tokens[s].len() > n_predict {
                                break;
                            }
                            pos[s] += 1;
                            emit(s, pos[s], t, &eng);
                            gen_tokens[s].push(t);
                            next[s] = t;
                            accepted_total += 1;
                            if t == eos {
                                finished[s] = true;
                            }
                            any = true;
                        }
                    }
                    if !any {
                        break;
                    }
                    min_gen = usize::MAX;
                    for (s, g) in gen_tokens.iter().enumerate() {
                        if !finished[s] {
                            min_gen = min_gen.min(g.len());
                        }
                    }
                }
                eprintln!(
                    "# spec-multi(k={spec_k}, n={n}): {cycles}사이클, 수용 {accepted_total}토큰"
                );
            } else if spec_k > 0 && has_mtp && n == 1 {
                let s = 0usize;
                let mut accepted_total = 0usize;
                let mut target_forwards = 0usize;
                let mut cycles = 0usize;
                while gen_tokens[s].len() <= n_predict && !finished[s] {
                    let (acc_toks, tf) = eng.spec_step(s, next[s], spec_k).map_err(|e| e.to_string())?;
                    cycles += 1;
                    target_forwards += tf;
                    for &t in &acc_toks {
                        if gen_tokens[s].len() > n_predict {
                            break;
                        }
                        pos[s] += 1;
                        emit(s, pos[s], t, &eng);
                        gen_tokens[s].push(t);
                        next[s] = t;
                        accepted_total += 1;
                        if t == eos {
                            finished[s] = true;
                        }
                    }
                }
                eprintln!(
                    "# spec(k={spec_k}): {cycles}사이클, 수용 {accepted_total}토큰, 타깃 forward {target_forwards}회 — 수용률/forward {:.2}",
                    accepted_total as f64 / target_forwards.max(1) as f64
                );
            } else {
                if spec_k > 0 && !has_mtp {
                    eprintln!("# --spec 무시: MTP(nextn) 텐서 없음");
                }
            // 배치 디코드 — 활성 시퀀스 묶어 1스텝 (np 상호검증 대상 경로)
            for _step in 0..n_predict {
                let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                let toks: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                let seq_ids: Vec<usize> = active.clone();
                let logits = eng.decode(&seq_ids, &toks).map_err(|e| e.to_string())?;
                for (i, &s) in active.iter().enumerate() {
                    let t = llm170_core::model::greedy(&logits[i]);
                    next[s] = t;
                    pos[s] += 1;
                    emit(s, pos[s], t, &eng);
                    gen_tokens[s].push(t);
                    if t == eos {
                        finished[s] = true;
                    }
                }
            }
            }
            let dt = t_start.elapsed();
            eprintln!(
                "# done: {} seqs, prompt max {}, gen per seq: {} (elapsed {dt:.1?})",
                n,
                max_prompt,
                gen_tokens.iter().map(|g| g.len()).min().unwrap_or(0)
            );
            Ok(())
        });
    if let Err(e) = engine_res {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    if let Some(rep) = llm170_profiler::report() {
        eprint!("\n{rep}");
    }
    ExitCode::SUCCESS
}

/// qwen4exp 추론 — Engine4 (시퀀스별 prefill/decode1).
fn run_q4_infer(
    model_path: &PathBuf,
    prompts: &[Vec<u32>],
    n_predict: usize,
    ctx: usize,
    backend: &str,
    gpu_runtime: &str,
) -> ExitCode {
    let t_start = std::time::Instant::now();
    let res = llm170_core::qwen4exp::Model4::load(model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, n, ctx);
            if backend == "gpu" {
                eprintln!("# backend: gpu — qwen4exp (cubecl 제거: CPU 폴백)");
            }
            let eos = eng.model.eos;
            let mut finished = vec![false; n];
            let mut next: Vec<u32> = Vec::with_capacity(n);
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                let t = llm170_core::model::greedy(&l);
                println!(
                    "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                    p.len(),
                    json_escape(&eng.piece(t))
                );
                next.push(t);
                finished[s] = t == eos;
            }
            let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            for _step in 0..n_predict {
                let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                for &s in &active {
                    let l = eng.decode1(s, next[s]).map_err(|e| e.to_string())?;
                    let t = llm170_core::model::greedy(&l);
                    next[s] = t;
                    pos[s] += 1;
                    println!(
                        "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                        pos[s],
                        json_escape(&eng.piece(t))
                    );
                    finished[s] = t == eos;
                }
            }
            eprintln!(
                "# done(q4): {n} seqs, gen per seq: {n_predict} (elapsed {:.1?})",
                t_start.elapsed()
            );
            Ok(())
        });
    if let Err(e) = res {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// 이 시점 emit — JSONL 1행 출력.
fn emit(seq: usize, pos: u32, token: u32, eng: &llm170_core::model::Engine) {
    // 이 시점 eng는 &Engine 차입 — piece는 model 접근
    println!(
        "{{\"seq\":{},\"pos\":{},\"token\":{},\"text\":{}}}",
        seq,
        pos,
        token,
        json_escape(&eng.piece(token))
    );
}
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn parse_ids(s: &str) -> Result<Vec<u32>, std::num::ParseIntError> {
    s.split(',').map(|t| t.trim().parse::<u32>()).collect()
}
