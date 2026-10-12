//! [R2] 참조 러너·로더 프로브 — w4a16-load · w4a16-ref.

use super::super::{arg, arg_str};

pub(super) fn load(args: &[String]) -> Result<String, String> {
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
pub(super) fn reference(args: &[String]) -> Result<String, String> {
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
    // [Q1 2026-10-12] 오라클 NLL 모드 — 코퍼스 순회 CPU teacher-forced NLL.
    let (mut corpus, mut limit) = (None::<String>, usize::MAX);
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prompt-tokens" => {
                let v = arg(&mut it, "--prompt-tokens requires ids")?;
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
            "--eval-corpus" => {
                corpus = Some(arg(&mut it, "--eval-corpus requires a path")?.to_string())
            }
            "--limit" => {
                limit = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--limit requires a number")?;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if prompts.is_empty() && corpus.is_none() {
        return Err("at least one --prompt-tokens 또는 --eval-corpus required".into());
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    if let Some(c) = &corpus {
        // [Q1] CPU 오라클 NLL — 전 위치 teacher-forced, 전 어휘 logsumexp(f64).
        let text = std::fs::read_to_string(c).map_err(|e| format!("corpus {c}: {e}"))?;
        let tk = crate::tokenize::Tokenizer::load(std::path::Path::new(&dir))?;
        let toks = tk.encode_opts(&text, false);
        let n = toks.len().min(limit);
        if n < 2 {
            return Err(format!("토큰 {n}개 — NLL에는 2개 이상 필요"));
        }
        let mut eng = llm170_core::qwen35::Engine::new(model, 1, ctx);
        let t0 = std::time::Instant::now();
        let mut l = eng.prefill(0, &toks[..1]).map_err(|e| e.to_string())?;
        let mut nll_sum = 0f64;
        for i in 1..n {
            nll_sum += crate::probes::eval::nll_of(&l, toks[i]);
            let mut ls = eng.decode(&[0], &[toks[i]]).map_err(|e| e.to_string())?;
            l = ls.pop().ok_or("decode logits 부재")?;
        }
        let mean = nll_sum / (n - 1) as f64;
        let ppl = mean.exp();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        return Ok(format!(
            "w4a16-ref {dir} — CPU 오라클 NLL (teacher-forced · logsumexp f64)\n  corpus {c} · 토큰 {n} (NLL {}위치) · {ms:.0}ms ({:.0}ms/토큰)\n  NLL평균 {mean:.9} · PPL {ppl:.9}\n",
            n - 1,
            ms / (n - 1) as f64
        ));
    }
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
