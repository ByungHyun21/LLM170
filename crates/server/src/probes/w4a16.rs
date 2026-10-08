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
    let m = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let rep = m.validate().map_err(|e| e.to_string())?;
    let c = &m.cfg;
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
        c.group_size,
        m.n_lins()
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
