//! infer — qwen35/qwen4exp greedy 추론 CLI (main.rs에서 이관, plans/35 P7).
//! JSONL {"seq","pos","token","text"} 스트림 출력.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::usage_err;

pub(crate) fn cmd_infer(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let backend = ma.backend.clone().unwrap_or_else(|| "cpu".into());
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
    // 단일 트랙(2026-10-08, plans/w4a16-cuda.md §5): 모델은 W4A16 디렉터리 단일 —
    // 그 외는 스니핑이 명시 에러로 안내.
    if let Err(e) = crate::engine::sniff_format(&model_path) {
        return usage_err(&e);
    }
    // W4A16은 CPU 전용(가속 커널 미구현 — W2): GPU 백엔드 지정은 명시 거부.
    if backend != "cpu" {
        return usage_err(
            "W4A16은 아직 CPU 전용(가속 커널 미구현 — plans/w4a16-cuda.md W2): --backend cpu",
        );
    }
    if spec_k.is_some() {
        return usage_err("W4A16은 --spec 미지원(MTP 미매핑 — plans/w4a16-cuda.md §2)");
    }
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    if max_prompt + n_predict + 8 >= ctx {
        return usage_err(&format!(
            "ctx({ctx}) too small for prompt({max_prompt})+n_predict({n_predict})"
        ));
    }

    llm170_diag::span::reset();
    let t_start = std::time::Instant::now();
    let engine_res = llm170_core::qwen35::Model::load(&model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::qwen35::Engine::new(m, n, ctx);
            // CPU 단일(가속은 W2 커널 이후 — 부착 없음).
            crate::resource::release_load_lock();
            let eos = llm170_core::qwen35::EOS_EOT;
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
    fn on_token(&mut self, s: usize, pos: u32, t: u32, eng: &llm170_core::qwen35::Engine) {
        println!(
            "{{\"seq\":{s},\"pos\":{pos},\"token\":{t},\"text\":{}}}",
            crate::json::quoted(&eng.piece(t))
        );
    }
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
