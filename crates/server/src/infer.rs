//! infer — qwen35/qwen4exp greedy 추론 CLI (main.rs에서 이관, plans/35 P7).
//! JSONL {"seq","pos","token","text"} 스트림 출력.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::usage_err;
use std::path::Path;

pub(crate) fn cmd_infer(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let backend = ma.backend.clone().unwrap_or_else(|| "cpu".into());
    let gpu_runtime = ma
        .gpu_runtime
        .clone()
        .or_else(|| std::env::var("LLM170_GPU_RUNTIME").ok())
        .unwrap_or_else(|| "hip".into());
    let mut spec_k: Option<usize> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
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
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) if (1..=8).contains(&k) => spec_k = Some(k),
                _ => return usage_err("--spec requires k in 1..=8"),
            },
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }

    let Some(model_path) = ma.model.clone().map(PathBuf::from) else {
        return usage_err("--model required");
    };
    if prompts.is_empty() {
        return usage_err("at least one --prompt-tokens required");
    }
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    if max_prompt + n_predict + 8 >= ctx {
        return usage_err(&format!(
            "ctx({ctx}) too small for prompt({max_prompt})+n_predict({n_predict})"
        ));
    }

    llm170_diag::span::reset();
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
        return run_q4_infer(
            &model_path,
            &prompts,
            n_predict,
            ctx,
            &backend,
            &gpu_runtime,
        );
    }
    let engine_res = llm170_core::qwen35::Model::load(&model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::qwen35::Engine::new(m, n, ctx);
            if spec_k.is_some() {
                eng.mtp_wanted = true; // 스펙 의도 — prefill 훅 활성 (plans/22)
            }
            // 백엔드 부착 — 단일 경로(attach_q35). LLM170_REQUIRE_GPU=1이면 폴백
            // 금지(2026-09-12: infer 검증이 폴백으로 통과한 사고 방지).
            let policy = if std::env::var_os("LLM170_REQUIRE_GPU").is_some() {
                crate::engine::AttachPolicy::Strict
            } else {
                crate::engine::AttachPolicy::Warn
            };
            eng = crate::engine::attach_q35(eng, gpu_runtime == "vulkan", policy)
                .map_err(|e| format!("GPU 백엔드 주입 실패(REQUIRE_GPU): {e}"))?;
            let eos = 248044u32;
            // prefill (시퀀스별 — GDN chunked 경로)
            let mut last_logits = Vec::with_capacity(n);
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                last_logits.push(l);
            }
            let mut finished = vec![false; n];
            let mut gen_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
            let next: Vec<u32> = last_logits
                .iter()
                .map(|l| llm170_core::qwen35::greedy(l))
                .collect();
            for s in 0..n {
                emit(s, prompts[s].len() as u32, next[s], &eng);
                gen_tokens[s].push(next[s]);
                if next[s] == eos {
                    finished[s] = true;
                }
            }
            // 생성 — 단일 루프(generate_q35): spec-multi/spec-single/batch.
            let spec_k: usize = spec_k.unwrap_or(0);
            let mut st = crate::engine::GenState {
                finished,
                gen_toks: gen_tokens,
                next,
                pos: prompts.iter().map(|p| p.len() as u32).collect(),
            };
            let (mode, stats) =
                crate::engine::generate_q35(&mut eng, &mut st, n_predict, spec_k, eos, &mut InferSink)?;
            let gen_tokens = st.gen_toks;
            match mode {
                "spec-multi" => eprintln!(
                    "# spec-multi(k={spec_k}, n={n}): {}사이클, 수용 {}토큰",
                    stats.cycles, stats.accepted
                ),
                "spec" => eprintln!(
                    "# spec(k={spec_k}): {}사이클, 수용 {}토큰, 타깃 forward {}회 — 수용률/forward {:.2}",
                    stats.cycles,
                    stats.accepted,
                    stats.target_forwards,
                    stats.accepted as f64 / stats.target_forwards.max(1) as f64
                ),
                _ => {}
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
    if let Some(rep) = llm170_diag::span::report() {
        eprint!("\n{rep}");
    }
    ExitCode::SUCCESS
}

/// infer JSONL 싱크 — 토큰마다 {"seq","pos","token","text"} 1행.
struct InferSink;
impl crate::engine::TokenSink for InferSink {
    fn on_token(
        &mut self,
        s: usize,
        pos: u32,
        t: u32,
        eng: &llm170_core::qwen35::Engine,
    ) {
        println!(
            "{{\"seq\":{s},\"pos\":{pos},\"token\":{t},\"text\":{}}}",
            crate::json::quoted(&eng.piece(t))
        );
    }
}

/// qwen4exp 추론 — Engine4 (시퀀스별 prefill/decode1).
fn run_q4_infer(
    model_path: &Path,
    prompts: &[Vec<u32>],
    n_predict: usize,
    ctx: usize,
    backend: &str,
    gpu_runtime: &str,
) -> ExitCode {
    let t_start = std::time::Instant::now();
    let want_gpu = crate::engine::q4_gpu_wanted_str(backend, gpu_runtime);
    let res = llm170_core::qwen4exp::Model4::load(model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let sources = m.part_sources();
            let eng = llm170_core::qwen4exp::layers::Engine4::new(m, n, ctx);
            // GPU 부착 — 단일 경로(attach_q4, Strict: infer 검증은 폴백 금지).
            let mut eng = crate::engine::attach_q4(
                eng,
                sources,
                want_gpu,
                crate::engine::q4_vk_runtime_str(gpu_runtime),
                false,
                crate::engine::AttachPolicy::Strict,
            )?;
            let eos = eng.model.eos;
            let mut finished = vec![false; n];
            let mut next: Vec<u32> = Vec::with_capacity(n);
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                let t = llm170_core::qwen35::greedy(&l);
                println!(
                    "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                    p.len(),
                    crate::json::quoted(&eng.piece(t))
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
                // plans/73(np): 활성 2+ 는 배치 디코드(무게 스트리밍 공유).
                if active.len() > 1 {
                    let toks: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                    let ls = eng
                        .decode_batch(&active, &toks)
                        .map_err(|e| e.to_string())?;
                    for (row, &s) in active.iter().enumerate() {
                        let t = llm170_core::qwen35::greedy(&ls[row]);
                        next[s] = t;
                        pos[s] += 1;
                        println!(
                            "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                            pos[s],
                            crate::json::quoted(&eng.piece(t))
                        );
                        finished[s] = t == eos;
                    }
                } else {
                    for &s in &active {
                        let d1g = std::env::var_os("LLM170_NO_D1G").is_none();
                        let t = if !d1g {
                            let l = eng.decode1(s, next[s]).map_err(|e| e.to_string())?;
                            llm170_core::qwen35::greedy(&l)
                        } else {
                            eng.decode1_greedy(s, next[s]).map_err(|e| e.to_string())?
                        };
                        next[s] = t;
                        pos[s] += 1;
                        println!(
                            "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                            pos[s],
                            crate::json::quoted(&eng.piece(t))
                        );
                        finished[s] = t == eos;
                    }
                }
            }
            if llm170_diag::dump::opts().alloc {
                llm170_diag::alloc::report();
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
fn emit(seq: usize, pos: u32, token: u32, eng: &llm170_core::qwen35::Engine) {
    // 이 시점 eng는 &Engine 차입 — piece는 model 접근
    println!(
        "{{\"seq\":{},\"pos\":{},\"token\":{},\"text\":{}}}",
        seq,
        pos,
        token,
        crate::json::quoted(&eng.piece(token))
    );
}
fn parse_ids(s: &str) -> Result<Vec<u32>, std::num::ParseIntError> {
    s.split(',').map(|t| t.trim().parse::<u32>()).collect()
}
