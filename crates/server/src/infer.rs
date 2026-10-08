//! infer — W4A16 CUDA 체인 greedy 추론 CLI (JSONL {"seq","pos","token","text"}).
//! 참조(CPU)는 `w4a16-ref` 프로브 — 여기는 GPU 단일 경로.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::usage_err;

pub(crate) fn cmd_infer(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
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
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path) = ma.model.clone().map(PathBuf::from) else {
        return usage_err("--model required");
    };
    if prompts.is_empty() {
        return usage_err("at least one --prompt-tokens required");
    }
    if let Err(e) = crate::engine::sniff_format(&model_path) {
        return usage_err(&e);
    }
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    if max_prompt + n_predict + 8 >= ctx {
        return usage_err(&format!(
            "ctx({ctx}) too small for prompt({max_prompt})+n_predict({n_predict})"
        ));
    }

    llm170_diag::span::reset();
    let t_start = std::time::Instant::now();
    let n = prompts.len();
    let mut eng = crate::sched::load_gpu_retry(&model_path, n, ctx);
    crate::resource::release_load_lock();
    let eos = llm170_core::qwen35::EOS_EOT;
    let mut last_logits = Vec::with_capacity(n);
    for (s, p) in prompts.iter().enumerate() {
        match eng.prefill(s, p) {
            Ok(l) => last_logits.push(l),
            Err(e) => {
                eprintln!("error: prefill seq{s}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let mut finished = vec![false; n];
    let mut gen_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
    let next: Vec<u32> = last_logits
        .iter()
        .map(|l| llm170_core::qwen35::greedy(l))
        .collect();
    let mut cur = next.clone();
    for s in 0..n {
        emit(s, prompts[s].len() as u32, cur[s], &eng);
        gen_tokens[s].push(cur[s]);
        if cur[s] == eos {
            finished[s] = true;
        }
    }
    for _ in 0..n_predict {
        for s in 0..n {
            if finished[s] {
                continue;
            }
            match eng.decode_greedy(s, cur[s]) {
                Ok(t) => {
                    cur[s] = t;
                    emit(
                        s,
                        prompts[s].len() as u32 + gen_tokens[s].len() as u32,
                        t,
                        &eng,
                    );
                    gen_tokens[s].push(t);
                    if t == eos {
                        finished[s] = true;
                    }
                }
                Err(e) => {
                    eprintln!("error: decode seq{s}: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if finished.iter().all(|&f| f) {
            break;
        }
    }
    let dt = t_start.elapsed();
    eprintln!(
        "# done: {} seqs, prompt max {}, gen per seq: {} (elapsed {dt:.1?})",
        n,
        max_prompt,
        gen_tokens.iter().map(|g| g.len()).min().unwrap_or(0)
    );
    if let Some(rep) = llm170_diag::span::report() {
        eprint!("\n{rep}");
    }
    ExitCode::SUCCESS
}

/// 이 시점 emit — JSONL 1행 출력.
fn emit(seq: usize, pos: u32, token: u32, eng: &crate::gpu_engine::GpuEngine) {
    println!(
        "{{\"seq\":{},\"pos\":{},\"token\":{},\"text\":{}}}",
        seq,
        pos,
        token,
        crate::json::esc(&eng.piece(token))
    );
}

fn parse_ids(s: &str) -> Result<Vec<u32>, std::num::ParseIntError> {
    s.split(',').map(|t| t.trim().parse::<u32>()).collect()
}
