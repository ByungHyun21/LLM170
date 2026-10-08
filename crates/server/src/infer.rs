//! infer — CLI 게이트(기본 CUDA 경로). CPU 참조·디버깅은 `w4a16-ref` 프로브.
//! (W3에서 CUDA 실행부가 여기 들어온다 — 그 전까지는 미착륙 안내로 거부.)

use std::path::PathBuf;
use std::process::ExitCode;

use crate::usage_err;

pub(crate) fn cmd_infer(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut n_prompts = 0usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prompt-tokens" => match it.next() {
                Some(v) => match parse_ids(v) {
                    Ok(ids) if !ids.is_empty() => n_prompts += 1,
                    Ok(_) => return usage_err("empty prompt"),
                    Err(e) => return usage_err(&format!("bad tokens: {e}")),
                },
                None => return usage_err("--prompt-tokens requires ids"),
            },
            "--n-predict" => {
                if it.next().and_then(|v| v.parse::<usize>().ok()).is_none() {
                    return usage_err("--n-predict requires a number");
                }
            }
            "--ctx" => {
                if it.next().and_then(|v| v.parse::<usize>().ok()).is_none() {
                    return usage_err("--ctx requires a number");
                }
            }
            other => return usage_err(&format!("unknown flag: {other}")),
        }
    }

    let Some(model_path) = ma.model.clone().map(PathBuf::from) else {
        return usage_err("--model required");
    };
    if n_prompts == 0 {
        return usage_err("at least one --prompt-tokens required");
    }
    if let Err(e) = crate::engine::sniff_format(&model_path) {
        return usage_err(&e);
    }
    // 기본 = CUDA 단일(가속 커널 미착륙 — W2/W3): W3에서 실행부가 들어오면
    // 이 게이트를 제거한다. 참조·디버깅은 `w4a16-ref` 프로브.
    const CUDA_READY: bool = false;
    if !CUDA_READY {
        return usage_err(
            "W4A16 CUDA 경로는 W2/W3 개발 중(가속 커널 미착륙) — 참조·디버깅은 w4a16-ref 프로브",
        );
    }
    ExitCode::SUCCESS
}

fn parse_ids(s: &str) -> Result<Vec<u32>, std::num::ParseIntError> {
    s.split(',').map(|t| t.trim().parse::<u32>()).collect()
}
