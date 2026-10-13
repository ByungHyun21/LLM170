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
//!   llm170 w4a16-eval needle <w4a16_dir> [--ctx N] [--lens N,N] [--sets N] [--json]

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
        "ci" => ci(&args[1..]),
        "needle" => needle(&args[1..]),
        "sample" => sample(&args[1..]),
        other => Err(format!(
            "w4a16-eval: 서브커맨드 '{other}' — 사용법: w4a16-eval ppl <dir> --corpus <txt> [--out dump] | w4a16-eval agree <a> <b> | w4a16-eval ci <base> <cand> [--reps N] [--block N]"
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
            "w4a16-eval ppl <dir> --corpus <txt> [--out dump] — 예: w4a16-eval ppl ../models/Qwen3.8-27B-W4A16-AutoRound --corpus docs/eval/corpus-en.txt --out /tmp/eval-a.tsv".into(),
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
    crate::gpu_engine::upload_model(&mut dec, &model, ctx, 0)?;
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

/// ci — 두 덤프의 **쌍대 ΔNLL 블록 부트스트랩 CI**(채택 게이트 계측기, §2.I).
/// 같은 코퍼스·같은 창의 teacher-forced NLL 두 벌을 토큰별로 짝지어 Δ(후보−기준)
/// 평균과 95% CI를 낸다. 인접 토큰 자기상관을 반영해 **이동블록 부트스트랩**
/// (기본 블록 16토큰) — i.i.d. 재표집은 CI를 과소평가한다. 난수는 splitmix64
/// (rand 크레이트 금지 계약 — seed 고정 시 결정적).
/// 밴드 판정은 ΔPPL%(= exp(ΔNLL평균)−1)의 CI로: 상한 < 밴드면 통과 등.
fn ci(args: &[String]) -> Result<String, String> {
    let a = arg_str(args, 0, "");
    let b = arg_str(args, 1, "");
    if a.is_empty() || b.is_empty() {
        return Err(
            "w4a16-eval ci <base.tsv> <cand.tsv> [--reps N] [--block N] [--seed S] [--json]".into(),
        );
    }
    let (mut reps, mut block, mut seed, mut json) =
        (10_000usize, 16usize, 0x1234_5678_9abc_def0u64, false);
    let mut it = args.iter().skip(2);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--reps" => {
                reps = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--reps requires a number")?
            }
            "--block" => {
                block = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--block requires a number")?
            }
            "--seed" => {
                seed = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--seed requires a number")?
            }
            "--json" => json = true,
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if block == 0 || reps == 0 {
        return Err("--block/--reps는 1 이상".into());
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
    let n = da.rows.len();
    let mut d: Vec<f64> = Vec::with_capacity(n);
    let (mut win, mut loss, mut tie) = (0usize, 0usize, 0usize);
    let mut worst = (0f64, 0u32);
    for (x, y) in da.rows.iter().zip(db.rows.iter()) {
        if x.0 != y.0 || x.1 != y.1 {
            return Err(format!(
                "pos/타깃 불일치(pos {} vs {}) — 같은 코퍼스 덤프가 아님",
                x.0, y.0
            ));
        }
        let delta = y.3 - x.3; // 후보 − 기준(음수 = 후보 우세)
        if delta < 0.0 {
            win += 1;
        } else if delta > 0.0 {
            loss += 1;
        } else {
            tie += 1;
        }
        if delta.abs() > worst.0.abs() {
            worst = (delta, x.0);
        }
        d.push(delta);
    }
    let mean = d.iter().sum::<f64>() / n as f64;
    let mean_a = da.rows.iter().map(|r| r.3).sum::<f64>() / n as f64;
    let mean_b = db.rows.iter().map(|r| r.3).sum::<f64>() / n as f64;
    // 이동블록 부트스트랩 — 시작 위치 균일 추출, 블록 내 순서 보존(순환).
    let blk = block.min(n);
    let nblk = n.div_ceil(blk);
    let mut rng = seed;
    let mut next = move || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut means: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut acc = 0f64;
        for _ in 0..nblk {
            let start = (next() % n as u64) as usize;
            for j in 0..blk {
                acc += d[(start + j) % n];
            }
        }
        means.push(acc / (nblk * blk) as f64);
    }
    means.sort_by(|p, q| p.total_cmp(q));
    let qp = |p: f64| means[((means.len() - 1) as f64 * p).round() as usize];
    let (lo, hi) = (qp(0.025), qp(0.975));
    let pos_frac = means.iter().filter(|&&m| m > 0.0).count() as f64 / means.len() as f64;
    // ΔNLL → ΔPPL% (PPL = exp(nll_mean)).
    let (ppl_a, ppl_b) = (mean_a.exp(), mean_b.exp());
    let pct = |x: f64| (x.exp() - 1.0) * 100.0;
    if json {
        return Ok(format!(
            "{{\"n\":{n},\"base_ppl\":{ppl_a:.9},\"cand_ppl\":{ppl_b:.9},\"d_nll_mean\":{mean:.9},\"ci_nll_lo\":{lo:.9},\"ci_nll_hi\":{hi:.9},\"d_ppl_pct\":{:.6},\"ci_ppl_lo\":{:.6},\"ci_ppl_hi\":{:.6},\"win\":{win},\"loss\":{loss},\"tie\":{tie},\"block\":{blk},\"reps\":{reps},\"p_pos\":{pos_frac:.4}}}",
            pct(mean),
            pct(lo),
            pct(hi)
        ));
    }
    Ok(format!(
        "w4a16-eval ci\n  base: {a} (ppl={ppl_a:.6})\n  cand: {b} (ppl={ppl_b:.6})\n  n={n} · 블록 {blk} · 재표집 {reps}\n  ΔNLL 평균 {mean:+.7} (95% CI [{lo:+.7}, {hi:+.7}])\n  ΔPPL {:+.4}% (95% CI [{:+.4}%, {:+.4}%])\n  토큰 승/패/동 {win}/{loss}/{tie} · 최악 {}{:+.4} @pos {} · P(Δ>0)={pos_frac:.3}",
        pct(mean),
        pct(lo),
        pct(hi),
        if worst.0 >= 0.0 { "+" } else { "" },
        worst.0,
        worst.1
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
    // [Q5 2026-10-12] 마진 컷 — 근접동률(margin<ε) 플립과 실질 발산을 구분.
    let mut margin_cut = 1.0f32;
    let mut it = args.iter().skip(2);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--margin-cut" => {
                margin_cut = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--margin-cut requires a number")?;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
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
    let mut real = 0usize;
    let mut m_min = f32::INFINITY;
    let mut m_max = 0f32;
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
            // 플립의 대표 마진 = 두 덤프 중 **작은** 쪽(양쪽 모두 확신해야 실질).
            let m = x.4.min(y.4);
            m_min = m_min.min(m);
            m_max = m_max.max(m);
            if m >= margin_cut {
                real += 1;
            }
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
    let flip_m = if div > 0 {
        format!("플립 마진 min={m_min:.3} max={m_max:.3}")
    } else {
        "플립 없음".into()
    };
    Ok(format!(
        "w4a16-eval agree\n  {a}\n  {b}\n  위치 {n} · argmax 일치 {matches} ({rate:.3}%) · 발산 {div}개(실질 {real}·근접 {}) · {div_desc}\n  {flip_m} (실질 기준 margin≥{margin_cut})\n  NLL합 a={sum_a:.6} b={sum_b:.6} (델타 {d_nll:+.4}%)\n  PPL a={ppl_a:.6} b={ppl_b:.6} (델타 {d_ppl:+.4}%)",
        div - real
    ))
}

/// argmax + top1-2 마진 — CPU `greedy_from` 규약(엄격 비교, 동률=최저 인덱스).
/// [Q3 2026-10-12] needle 하네스 — 장문 검색 정확도(플랜 §5 Q3).
/// 합성 haystack 내 고유 패스코드(결정적 LCG — rand 금지 계약)를 깊이
/// (10/50/90%)별로 삽입하고, greedy 답변 8토큰에 패스코드 토큰 부분열이
/// 있는지로 채점한다(토큰 id 기준 — 디코딩 무관·결정적). 문맥 프리필은
/// ppl 청크 경로와 동일(KV 누적 — 수치 계급 동일).
fn needle(args: &[String]) -> Result<String, String> {
    let mut it = args.iter();
    let Some(dir) = it.next().cloned() else {
        return Err("w4a16-eval needle <dir> [--ctx N] [--lens N,N] [--sets N] [--json] — 예: w4a16-eval needle ../models/Qwen3.8-27B-W4A16-AutoRound --lens 4096,16384".into());
    };
    let (mut ctx, mut lens, mut sets, mut json) =
        (16384usize, vec![4096usize, 16384], 1usize, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ctx" => {
                ctx = arg(&mut it, "--ctx requires a number")?
                    .parse()
                    .map_err(|_| "--ctx requires a number")?;
            }
            "--lens" => {
                lens = arg(&mut it, "--lens requires N,N")?
                    .split(',')
                    .map(|v| {
                        v.parse::<usize>()
                            .map_err(|_| "--lens requires N,N".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--sets" => {
                sets = arg(&mut it, "--sets requires a number")?
                    .parse()
                    .map_err(|_| "--sets requires a number")?;
            }
            "--json" => json = true,
            other => return Err(format!("w4a16-eval needle: unknown flag {other}")),
        }
    }
    let model = llm170_core::qwen35::Model::load(Path::new(&dir)).map_err(|e| e.to_string())?;
    let hp = model.hp.clone();
    let tk = crate::tokenize::Tokenizer::load(Path::new(&dir))?;
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    crate::gpu_engine::upload_model(&mut dec, &model, ctx, 0)?;
    let head = model
        .w("output.weight")
        .ok_or_else(|| "output.weight 부재".to_string())?;
    if head.ty != llm170_core::wtype::WType::Bf16 {
        return Err("needle은 GPU head 전제(Bf16 head 필요)".into());
    }
    dec.upload_head(head.data, head.n_out as usize, head.n_in as usize)?;

    // 합성 haystack — 중성 산책 문장 반복(코퍼스 의존 없음·결정적).
    const HAY: &str = "The narrow alpine trail bends past the old stone bridge and continues through the quiet valley below the snow line. ";
    let hay: Vec<u32> = tk.encode_opts(&HAY.repeat(1400), false);
    let mut seed = 0x5EED_C0DE_1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let alphabet = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let depths = [10usize, 50, 90];
    let mut rows: Vec<(usize, usize, usize, bool)> = Vec::new();
    for &l in &lens {
        if l + 32 > ctx {
            return Err(format!("len {l} + 질문 > ctx {ctx} — --ctx 확대 필요"));
        }
        for set in 0..sets {
            let word: String = (0..6)
                .map(|_| alphabet[(lcg() as usize) % alphabet.len()] as char)
                .collect();
            let n_id = tk.encode_opts(
                &format!(" The secret passcode for slot {set} is {word}. "),
                false,
            );
            let q_id = tk.encode_opts(
                &format!(
                    " What is the secret passcode for slot {set}? Reply with the passcode only:"
                ),
                false,
            );
            let word_id = tk.encode_opts(&word, false);
            let hay_need = l.saturating_sub(n_id.len() + q_id.len());
            if hay.len() < hay_need {
                return Err(format!("haystack 부족: {} < {hay_need}", hay.len()));
            }
            for &d in &depths {
                let cut = hay_need * d / 100;
                let mut toks: Vec<u32> = Vec::with_capacity(l + 16);
                toks.extend_from_slice(&hay[..cut]);
                toks.extend_from_slice(&n_id);
                toks.extend_from_slice(&hay[cut..hay_need]);
                toks.extend_from_slice(&q_id);
                dec.reset_state(0)?;
                // 문맥 n-1토큰까지 배치 체인 프리필(무로짓·t≥ 2) — 마지막
                // 토큰은 t=1 head로 예측(프리필/디코드 경계 동일).
                let n_ctx = toks.len() - 1;
                let mut i = 0usize;
                while i < n_ctx {
                    let t = (n_ctx - i).clamp(2, 512);
                    let mut rb: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
                    for &tok in &toks[i..i + t] {
                        rb.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
                    }
                    dec.forward_prefill(0, &rb, t, false)?;
                    i += t;
                }
                // greedy 8토큰 — 마지막 문맥 토큰에서 시작.
                let mut row: Vec<f32> = dec.forward_device_head(
                    0,
                    &model.embed_row(toks[n_ctx]).map_err(|e| e.to_string())?,
                )?;
                let mut picks: Vec<u32> = Vec::with_capacity(8);
                for _ in 0..8 {
                    let mut best = (0u32, f32::MIN);
                    for (j, &v) in row.iter().enumerate() {
                        if v > best.1 {
                            best = (j as u32, v);
                        }
                    }
                    picks.push(best.0);
                    let er = model.embed_row(best.0).map_err(|e| e.to_string())?;
                    row = dec.forward_device_head(0, &er)?;
                }
                let hit = picks.windows(word_id.len()).any(|w| w == &word_id[..]);
                rows.push((l, d, set, hit));
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let hits = rows.iter().filter(|r| r.3).count();
    if json {
        let cells: Vec<String> = rows
            .iter()
            .map(|(l, d, s, h)| {
                format!("{{\"len\":{l},\"depth_pct\":{d},\"set\":{s},\"hit\":{h}}}")
            })
            .collect();
        return Ok(format!(
            "{{\"model\":\"{dir}\",\"cases\":{},\"hits\":{hits},\"acc\":{:.4},\"elapsed_ms\":{ms:.0},\"rows\":[{}]}}",
            rows.len(),
            hits as f64 / rows.len().max(1) as f64,
            cells.join(",")
        ));
    }
    let mut s = format!(
        "needle {dir}: {hits}/{} hits ({:.1}%)\n",
        rows.len(),
        100.0 * hits as f64 / rows.len().max(1) as f64
    );
    for (l, d, set, h) in &rows {
        s.push_str(&format!("  len={l} depth={d}% set={set} hit={h}\n"));
    }
    s.push_str(&format!("  elapsed {ms:.0}ms\n"));
    Ok(s)
}

/// [Q4 2026-10-12] 샘플링 검증 하네스 — 고정 LCG 로짓에서 서빙 CPU
/// 샘플러(splitmix64 시드 — rand 금지 계약)로 N표본 뽑고 이론
/// softmax(temperature·top_k/top_p/min_p 비활성) 분포와 TV 거리·
/// 카이제곱을 대조한다(판정 기준은 실행 출력 참조 — docs/eval.md).
fn sample(args: &[String]) -> Result<String, String> {
    let mut it = args.iter();
    let Some(dir) = it.next().cloned() else {
        return Err("w4a16-eval sample <dir> [--vocab N] [--n N] [--temperature F] [--top-k K] — 예: w4a16-eval sample ../models/...".into());
    };
    let (mut nv, mut n, mut temp, mut topk) = (512usize, 50_000usize, 1.0f32, 0usize);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--vocab" => {
                nv = arg(&mut it, "--vocab requires a number")?
                    .parse()
                    .map_err(|_| "--vocab requires a number")?
            }
            "--n" => {
                n = arg(&mut it, "--n requires a number")?
                    .parse()
                    .map_err(|_| "--n requires a number")?
            }
            "--temperature" => {
                temp = arg(&mut it, "--temperature requires a number")?
                    .parse()
                    .map_err(|_| "--temperature requires a number")?
            }
            "--top-k" => {
                topk = arg(&mut it, "--top-k requires a number")?
                    .parse()
                    .map_err(|_| "--top-k requires a number")?
            }
            other => return Err(format!("w4a16-eval sample: unknown flag {other}")),
        }
    }
    let _ = std::path::Path::new(&dir)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let params = llm170_core::sampler::SamplerParams {
        temperature: temp,
        top_k: topk,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        seed: 0x5EED_C0DE,
    };
    let mut smp = llm170_core::sampler::Sampler::new(params.clone());
    // 고정 로짓 — LCG 결정적([-8, 8] 범위).
    let mut seed = 0xABCD_1234_5678_9DEFu64;
    let base_logits: Vec<f32> = (0..nv)
        .map(|i| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64);
            ((seed >> 33) as f32 / 2147483648.0) * 8.0
        })
        .collect();
    let mut counts = vec![0u64; nv];
    for _ in 0..n {
        let mut lg = base_logits.clone();
        let t = smp.sample(&mut lg);
        counts[t as usize] += 1;
    }
    // 이론 분포 — 비활성 필터(top_k만) 적용 softmax·코드 표본은 f64.
    let mx = base_logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut idx: Vec<usize> = (0..nv).collect();
    if topk > 0 && topk < nv {
        idx.sort_by(|&a, &b| base_logits[b].total_cmp(&base_logits[a]));
        idx.truncate(topk);
    }
    let zsum: f64 = idx
        .iter()
        .map(|&i| f64::from(((base_logits[i] - mx) / temp).exp()))
        .sum();
    let (mut tv, mut chi2) = (0.0f64, 0.0f64);
    for i in 0..nv {
        let p = if idx.contains(&i) {
            f64::from(((base_logits[i] - mx) / temp).exp()) / zsum
        } else {
            0.0
        };
        let q = counts[i] as f64 / n as f64;
        tv += (p - q).abs();
        if p > 0.0 {
            let e = p * n as f64;
            chi2 += (counts[i] as f64 - e).powi(2) / e;
        }
    }
    tv *= 0.5;
    // [Q4-fx] 상위 편차 프로파일 — 편향 방향 진단(토큰·기대·관측·비).
    let mut top: Vec<(usize, f64, u64)> = idx
        .iter()
        .map(|&i| {
            let pp = f64::from((base_logits[i] - mx).exp()) / zsum;
            (i, pp, counts[i])
        })
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    top.truncate(8);
    let dump: Vec<String> = top
        .iter()
        .map(|(i, pp, c)| {
            let e = pp * n as f64;
            format!("{i}:exp={e:.0},obs={c},r={:.2}", *c as f64 / e)
        })
        .collect();
    eprintln!("top8 {}", dump.join(" | "));
    let df = idx.len() - 1;
    let z = (chi2 / df as f64 - 1.0) * (df as f64 / 2.0).sqrt();
    Ok(format!(
        "sample: vocab={nv} n={n} temperature={temp} top_k={topk} — TV={tv:.5} chi2={chi2:.1} df={df} z={z:+.2} (|z|<3 = 밴드 내)"
    ))
}

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
pub(crate) fn nll_of(lg: &[f32], tgt: u32) -> f64 {
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
    fn agree_margin_cut_splits_real_vs_near() {
        // pos1: argmax 동일 · pos2: 근접 플립(마진 0.01) · pos3: 실질 플립(마진 3.0).
        let pa = std::env::temp_dir().join(format!("eval-ag-a-{}.tsv", std::process::id()));
        let pb = std::env::temp_dir().join(format!("eval-ag-b-{}.tsv", std::process::id()));
        let hdr = "# w4a16-eval dump v1 model=x tokens=4 head=gpu nll_sum=4.0 ppl=2.7\n# cols: pos target argmax nll margin\n";
        std::fs::write(
            &pa,
            format!("{hdr}1\t5\t5\t1.0\t2.0\n2\t7\t7\t1.0\t0.01\n3\t9\t9\t1.0\t3.0\n"),
        )
        .unwrap();
        std::fs::write(
            &pb,
            format!("{hdr}1\t5\t5\t1.0\t2.0\n2\t7\t8\t1.0\t0.02\n3\t9\t4\t1.0\t3.5\n"),
        )
        .unwrap();
        let out = agree(&[
            pa.to_str().unwrap().into(),
            pb.to_str().unwrap().into(),
            "--margin-cut".into(),
            "1.0".into(),
        ])
        .unwrap();
        assert!(out.contains("발산 2개(실질 1·근접 1)"), "{out}");
        assert!(out.contains("margin≥1"), "{out}");
        let _ = std::fs::remove_file(&pa);
        let _ = std::fs::remove_file(&pb);
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

#[cfg(test)]
mod ci_tests {
    use super::*;

    #[test]
    fn ci_identical_dumps_zero() {
        let p1 = write_tmp_dump(&[1.0, 1.2, 0.8, 2.0], "ci-a");
        let p2 = write_tmp_dump(&[1.0, 1.2, 0.8, 2.0], "ci-b");
        let out = ci(&[p1.clone(), p2.clone(), "--reps".into(), "1000".into()]).unwrap();
        assert!(out.contains("+0.0000%"), "{out}");
        assert!(out.contains("P(Δ>0)=0.000"), "{out}");
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn ci_uniform_delta_ppl() {
        // 균일 Δ=+0.1 NLL → ΔPPL = e^0.1−1 = +10.5171%.
        let base: Vec<f64> = (0..64).map(|i| 1.0 + 0.001 * i as f64).collect();
        let cand: Vec<f64> = base.iter().map(|x| x + 0.1).collect();
        let p1 = write_tmp_dump(&base, "ci-u1");
        let p2 = write_tmp_dump(&cand, "ci-u2");
        let out = ci(&[p1.clone(), p2.clone()]).unwrap();
        assert!(out.contains("+10.5171%"), "{out}");
        assert!(out.contains("1/0/63") || out.contains("0/64/0"), "{out}");
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn ci_block_bootstrap_orders() {
        // 혼합 부호 — CI가 평균을 감싼다.
        let base: Vec<f64> = (0..128).map(|i| 2.0).collect();
        let mut cand = base.clone();
        for i in (0..128).step_by(2) {
            cand[i] += 0.05;
            cand[i + 1] -= 0.02;
        }
        let p1 = write_tmp_dump(&base, "ci-m1");
        let p2 = write_tmp_dump(&cand, "ci-m2");
        let out = ci(&[p1.clone(), p2.clone(), "--json".into()]).unwrap();
        assert!(out.contains("\"ci_nll_lo\"") && out.contains("\"ci_nll_hi\""), "{out}");
        let lo: f64 = out.split("\"ci_nll_lo\":").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
        let hi: f64 = out.split("\"ci_nll_hi\":").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
        assert!(lo <= hi);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    fn write_tmp_dump(nlls: &[f64], tag: &str) -> String {
        let p = std::env::temp_dir().join(format!("eval-{tag}-{}.tsv", std::process::id()));
        let mut s = String::from("# w4a16-eval dump v1 model=x tokens=0 head=gpu nll_sum=0 ppl=0\n# cols: pos target argmax nll margin\n");
        for (i, &n) in nlls.iter().enumerate() {
            s.push_str(&format!("{}\t{}\t0\t{}\t0.5\n", i + 1, 7, n));
        }
        std::fs::write(&p, s).unwrap();
        p.to_string_lossy().into_owned()
    }
}
