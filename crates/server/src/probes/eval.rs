//! 통계 eval 프로브 — `w4a16-eval ppl|agree` (2026-10-10 신설).
//!
//! 목적: 연산 경로 후보의 **품질 편차 계측기**. 방침(루트 AGENTS.md —
//! "연산 경로는 모듈당 하나, fast를 기본으로")상 수치를 바꾸는 후보는
//! 런타임 플래그로 병존시키지 않는다. 후보는 머지 시점에 이 계측으로
//! PPL·teacher-forced argmax 일치율(편차)을 확인해 단일 경로로 채택
//! (골든 재동결)하거나 삭제한다. 기본 경로 골든과 독립인 별도 계측기.
//!
//! `ppl`: 코퍼스(텍스트 또는 토큰 파일)를 t=1 디코드로 훑어 전 위치
//! NLL·PPL을 계산하고 per-position 덤프(토큰/argmax/NLL/top1-2 마진)를
//! 기록한다. NLL은 tokens[1..] 기준(첫 토큰은 문맥만), logsumexp f64.
//! `agree`: 두 덤프 비교 — argmax 일치율·최초 발산·NLL/PPL 델타.
//!
//! 사용법:
//!   llm170 w4a16-eval ppl <w4a16_dir> --corpus <txt> [--ctx N] [--limit N]
//!                           [--out dump.tsv] [--json]
//!   llm170 w4a16-eval ppl <w4a16_dir> --from-tokens <file> [...]
//!   llm170 w4a16-eval agree <a.tsv> <b.tsv>

use super::{arg, arg_str, finish};
use llm170_backend_gpu::W4a16Dec;
use std::path::Path;
use std::process::ExitCode;

pub fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    if cmd != "w4a16-eval" {
        return None;
    }
    Some(finish(run(args)))
}

fn run(args: &[String]) -> Result<String, String> {
    match args.first().map(String::as_str).unwrap_or("") {
        "ppl" => ppl(&args[1..]),
        "agree" => agree(&args[1..]),
        other => Err(format!(
            "w4a16-eval: 서브커맨드 '{other}' — 사용법: w4a16-eval ppl <dir> --corpus <txt> [--out dump] | w4a16-eval agree <a> <b>"
        )),
    }
}

/// 덤프 한 행 — (pos, target, argmax, nll, margin).
type Row = (u32, u32, u32, f64, f32);

/// ppl — 코퍼스 순회 + NLL/PPL + 덤프.
fn ppl(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-eval ppl <dir> --corpus <txt> [--out dump] — 예: w4a16-eval ppl ../models/Qwen3.8-27B-W4A16-AutoRound --corpus benchmark/eval/corpus-en.txt --out /tmp/eval-a.tsv".into(),
        );
    }
    let (mut corpus, mut from_tokens, mut out_path) =
        (None::<String>, None::<String>, None::<String>);
    let (mut ctx, mut limit, mut json) = (8192usize, usize::MAX, false);
    let mut chunk = 1usize;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--corpus" => corpus = Some(arg(&mut it, "--corpus requires a path")?.to_string()),
            "--chunk" => {
                chunk = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--chunk requires a number")?;
                if !(1..=32).contains(&chunk) {
                    return Err(format!("--chunk {chunk}: 1..=32(GEMM_FFMA_TMAX)"));
                }
            }
            "--from-tokens" => {
                from_tokens = Some(arg(&mut it, "--from-tokens requires a path")?.to_string())
            }
            "--out" => out_path = Some(arg(&mut it, "--out requires a path")?.to_string()),
            "--ctx" => {
                ctx = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ctx requires a number")?;
            }
            "--limit" => {
                limit = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--limit requires a number")?;
            }
            "--json" => json = true,
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if corpus.is_none() == from_tokens.is_none() {
        return Err("--corpus 또는 --from-tokens 중 정확히 하나 필요".into());
    }
    let toks: Vec<u32> = if let Some(c) = &corpus {
        let text = std::fs::read_to_string(c).map_err(|e| format!("corpus {c}: {e}"))?;
        let tk = crate::tokenize::Tokenizer::load(Path::new(&dir))?;
        // 특수 토큰 미해석 — 코퍼스 원문 그대로(결정적).
        tk.encode_opts(&text, false)
    } else {
        let f = from_tokens.as_deref().unwrap_or("");
        let s = std::fs::read_to_string(f).map_err(|e| format!("tokens {f}: {e}"))?;
        let mut v = Vec::new();
        for p in s.split(|c: char| c == ',' || c.is_whitespace()) {
            if p.is_empty() {
                continue;
            }
            v.push(
                p.parse::<u32>()
                    .map_err(|e| format!("토큰 파싱 {p}: {e}"))?,
            );
        }
        v
    };
    let n = toks.len().min(limit);
    if n < 2 {
        return Err(format!("토큰 {n}개 — NLL 계산에는 2개 이상 필요"));
    }
    if n - 1 > ctx {
        return Err(format!("토큰 {n} > ctx {ctx} — --ctx 확대 필요"));
    }
    let model = llm170_core::qwen35::Model::load(Path::new(&dir)).map_err(|e| e.to_string())?;
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    crate::gpu_engine::upload_model(&mut dec, &model, ctx)?;
    let head = model
        .w("output.weight")
        .ok_or_else(|| "output.weight 부재".to_string())?;
    let head_gpu = head.ty == llm170_core::wtype::WType::Bf16;
    if head_gpu {
        dec.upload_head(head.data, head.n_out as usize, head.n_in as usize)?;
    }
    dec.reset_state(0)?;
    let vocab = head.n_out as usize;
    let mut rows: Vec<Row> = Vec::with_capacity(n - 1);
    let mut nll_sum = 0f64;
    if chunk >= 2 {
        // [2026-10-10] 청크 경로 — t행 배치 프리필 + 전 위치 로짓 회수
        // (GEMM t≥2 — t=1 GEMV와 수치 계급이 다름: 방법 B, eval.md 참조).
        if !head_gpu {
            return Err("--chunk≥2는 GPU head 필요(CPU head 경로 미지원)".into());
        }
        dec.batch_tmax_min = chunk;
        let mut i = 0usize;
        while i + 1 < n {
            let t = (n - i).min(chunk);
            if t < 2 {
                break;
            }
            let mut rb: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
            for &tok in &toks[i..i + t] {
                rb.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
            }
            let lg = dec.forward_prefill_logits(0, &rb, t)?;
            for r in 0..t {
                let g = i + r;
                if g + 1 >= n {
                    break;
                }
                let lr = &lg[r * vocab..(r + 1) * vocab];
                let (am, margin) = argmax_margin(lr);
                let tgt = toks[g + 1];
                let nll = nll_of(lr, tgt);
                nll_sum += nll;
                rows.push((g as u32 + 1, tgt, am, nll, margin));
            }
            i += t;
        }
    } else {
        for i in 0..n - 1 {
            let row = model.embed_row(toks[i]).map_err(|e| e.to_string())?;
            let lg: Vec<f32> = if head_gpu {
                dec.forward_device_head(0, &row)?
            } else {
                let xn = dec.forward_device(0, &row)?;
                let mut lg = vec![0.0f32; vocab];
                llm170_core::matmul::matmul(&xn, &head, &mut lg);
                lg
            };
            let (am, margin) = argmax_margin(&lg);
            let tgt = toks[i + 1];
            let nll = nll_of(&lg, tgt);
            nll_sum += nll;
            rows.push((i as u32 + 1, tgt, am, nll, margin));
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mean = nll_sum / rows.len() as f64;
    let ppl = mean.exp();
    if let Some(p) = &out_path {
        let mut s = String::with_capacity(rows.len() * 40 + 256);
        s.push_str(&format!(
            "# w4a16-eval dump v1 model={dir} tokens={n} head={} chunk={chunk} nll_sum={nll_sum:.9} ppl={ppl:.9}\n",
            if head_gpu { "gpu" } else { "cpu" }
        ));
        s.push_str("# cols: pos target argmax nll margin\n");
        for &(pos, tgt, am, nll, margin) in &rows {
            s.push_str(&format!("{pos}\t{tgt}\t{am}\t{nll:.9}\t{margin:.9}\n"));
        }
        std::fs::write(p, s).map_err(|e| format!("dump {p}: {e}"))?;
    }
    if json {
        return Ok(format!(
            "{{\"model\":\"{dir}\",\"tokens\":{n},\"scored\":{},\"nll_mean\":{mean:.9},\"ppl\":{ppl:.9},\"head\":\"{}\",\"chunk\":{chunk},\"elapsed_ms\":{ms:.1},\"dump\":{}}}",
            rows.len(),
            if head_gpu { "gpu" } else { "cpu" },
            out_path
                .as_ref()
                .map(|p| format!("\"{p}\""))
                .unwrap_or_else(|| "null".into())
        ));
    }
    Ok(format!(
        "w4a16-eval ppl {dir}\n  토큰 {n} (NLL {}위치) · head {} · 청크 {chunk} · 업로드+순회 {ms:.0}ms ({:.1}ms/토큰)\n  NLL평균 {mean:.6} · PPL {ppl:.6}{}",
        rows.len(),
        if head_gpu { "GPU" } else { "CPU" },
        ms / rows.len() as f64,
        out_path
            .as_ref()
            .map(|p| format!("\n  덤프 {p}"))
            .unwrap_or_default()
    ))
}

/// 덤프 파싱 결과.
struct Dump {
    rows: Vec<Row>,
    nll_sum: Option<f64>,
}

fn read_dump(path: &str) -> Result<Dump, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("dump {path}: {e}"))?;
    let mut rows = Vec::new();
    let mut nll_sum = None;
    for (ln, line) in s.lines().enumerate() {
        if line.starts_with('#') {
            if let Some(v) = line.split("nll_sum=").nth(1)
                && let Some(tok) = v.split_whitespace().next()
            {
                nll_sum = tok.parse::<f64>().ok();
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 5 {
            return Err(format!("{path}:{}: 필드 5개 아님: {line}", ln + 1));
        }
        let parse = |x: &str| -> Result<f64, String> {
            x.parse::<f64>()
                .map_err(|e| format!("{path}:{}: {x}: {e}", ln + 1))
        };
        rows.push((
            parse(f[0])? as u32,
            parse(f[1])? as u32,
            parse(f[2])? as u32,
            parse(f[3])?,
            parse(f[4])? as f32,
        ));
    }
    if rows.is_empty() {
        return Err(format!("{path}: 유효 행 없음"));
    }
    Ok(Dump { rows, nll_sum })
}

/// agree — 두 덤프의 argmax 일치율·최초 발산·NLL/PPL 델타.
fn agree(args: &[String]) -> Result<String, String> {
    let a = arg_str(args, 0, "");
    let b = arg_str(args, 1, "");
    if a.is_empty() || b.is_empty() {
        return Err("w4a16-eval agree <a.tsv> <b.tsv>".into());
    }
    let da = read_dump(&a)?;
    let db = read_dump(&b)?;
    if da.rows.len() != db.rows.len() {
        return Err(format!(
            "행 수 불일치: {a}={} {b}={}",
            da.rows.len(),
            db.rows.len()
        ));
    }
    let mut matches = 0usize;
    let mut div = 0usize;
    let mut first: Option<(u32, u32, u32, u32)> = None;
    for (x, y) in da.rows.iter().zip(db.rows.iter()) {
        if x.0 != y.0 || x.1 != y.1 {
            return Err(format!(
                "pos/타깃 불일치(pos {} vs {}) — 같은 코퍼스 덤프가 아님",
                x.0, y.0
            ));
        }
        if x.2 == y.2 {
            matches += 1;
        } else {
            div += 1;
            if first.is_none() {
                first = Some((x.0, x.1, x.2, y.2));
            }
        }
    }
    let n = da.rows.len();
    let rate = matches as f64 / n as f64 * 100.0;
    let sum_a = da
        .nll_sum
        .unwrap_or_else(|| da.rows.iter().map(|r| r.3).sum());
    let sum_b = db
        .nll_sum
        .unwrap_or_else(|| db.rows.iter().map(|r| r.3).sum());
    let ppl_a = (sum_a / n as f64).exp();
    let ppl_b = (sum_b / n as f64).exp();
    let d_nll = (sum_b - sum_a) / sum_a.abs() * 100.0;
    let d_ppl = (ppl_b - ppl_a) / ppl_a * 100.0;
    let div_desc = match first {
        Some((p, t, x, y)) => format!("최초 발산 pos={p} target={t}: a={x} b={y}"),
        None => "발산 없음".into(),
    };
    Ok(format!(
        "w4a16-eval agree\n  {a}\n  {b}\n  위치 {n} · argmax 일치 {matches} ({rate:.3}%) · 발산 {div}개 · {div_desc}\n  NLL합 a={sum_a:.6} b={sum_b:.6} (델타 {d_nll:+.4}%)\n  PPL a={ppl_a:.6} b={ppl_b:.6} (델타 {d_ppl:+.4}%)"
    ))
}

/// argmax + top1-2 마진 — CPU `greedy_from` 규약(엄격 비교, 동률=최저 인덱스).
fn argmax_margin(lg: &[f32]) -> (u32, f32) {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0u32;
    for (i, &v) in lg.iter().enumerate() {
        if v > best {
            best = v;
            bi = i as u32;
        }
    }
    let mut second = f32::NEG_INFINITY;
    for (i, &v) in lg.iter().enumerate() {
        if i as u32 != bi && v > second {
            second = v;
        }
    }
    let margin = if second == f32::NEG_INFINITY {
        f32::INFINITY
    } else {
        best - second
    };
    (bi, margin)
}

/// -ln p(target) — logsumexp(전 어휘) f64 누적, max 차감.
fn nll_of(lg: &[f32], tgt: u32) -> f64 {
    let m = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut s = 0f64;
    for &v in lg {
        s += (v as f64 - m).exp();
    }
    let lp = lg[tgt as usize] as f64 - m - s.ln();
    -lp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_margin_tie_lowest_index() {
        let (i, m) = argmax_margin(&[1.0, 5.0, 5.0, 2.0]);
        assert_eq!(i, 1);
        assert_eq!(m, 0.0);
    }

    #[test]
    fn argmax_margin_basic() {
        let (i, m) = argmax_margin(&[0.0, 2.0, 5.0]);
        assert_eq!(i, 2);
        assert_eq!(m, 3.0);
    }

    #[test]
    fn nll_uniform_two() {
        // 로짓 [0,0] → p=1/2 → NLL=ln2.
        let v = nll_of(&[0.0, 0.0], 0);
        assert!((v - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn nll_dominant() {
        // 로짓 [10,0] → p(tgt=0)≈1 → NLL≈0.
        let v = nll_of(&[10.0, 0.0], 0);
        assert!(v < 1e-4, "nll={v}");
    }

    #[test]
    fn dump_roundtrip() {
        let p = std::env::temp_dir().join(format!("eval-dump-{}.tsv", std::process::id()));
        let s = "# w4a16-eval dump v1 model=x tokens=3 head=gpu nll_sum=2.5 ppl=2.11\n# cols: pos target argmax nll margin\n1\t5\t5\t1.0\t0.5\n2\t7\t8\t1.5\t0.1\n";
        std::fs::write(&p, s).unwrap();
        let d = read_dump(p.to_str().unwrap()).unwrap();
        assert_eq!(d.rows.len(), 2);
        assert_eq!(d.rows[0], (1, 5, 5, 1.0, 0.5));
        assert_eq!(d.nll_sum, Some(2.5));
        let _ = std::fs::remove_file(&p);
    }
}
