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
    // 아치 바인딩이 등록된 모델만 커버리지(완전성)까지 검증한다 — 미등록
    // 아치는 포맷 계층(트리플 구조·형상 값·양자 파라미터)만.
    let arch = llm170_core::qwen35::bind::dir_arch(std::path::Path::new(&dir));
    if arch
        .as_deref()
        .is_some_and(llm170_core::gemma4::bind::arch_ok)
    {
        return load_gemma4(&dir, &store);
    }
    if !arch
        .as_deref()
        .is_some_and(llm170_core::qwen35::bind::arch_supported)
    {
        return format_store_level(&dir, &store, arch.as_deref());
    }
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

/// gemma4 로더 완전성 — 바인딩 validate(형상 계약·커버리지·전문가 구조).
fn load_gemma4(dir: &str, store: &llm170_core::w4a16::W4a16Model) -> Result<String, String> {
    let cfg = llm170_core::gemma4::bind::GemmaCfg::load(std::path::Path::new(dir))
        .map_err(|e| e.to_string())?;
    let rep = llm170_core::gemma4::bind::validate(store, &cfg).map_err(|e| e.to_string())?;
    let n_full = cfg.layer_full.iter().filter(|&&f| f).count();
    let head = format!(
        "w4a16-load {dir}\n  hidden={} layers={} heads={} (슬라이딩 {}×{}·풀 {}×{} — k_eq_v {}) ffn={} vocab={}\n  MoE: {}전문가×{} top-{} (dense {} 병렬) · SW {} · softcap {} · g{} (zp=8 상수)\n  슬라이딩 {}/{}층 · 선형 {}개",
        cfg.hidden,
        cfg.layers,
        cfg.heads,
        cfg.kv_local,
        cfg.head_dim_local,
        cfg.kv_global,
        cfg.head_dim_global,
        cfg.k_eq_v,
        cfg.ffn,
        cfg.vocab,
        cfg.n_experts,
        cfg.moe_ffn,
        cfg.top_k,
        cfg.ffn,
        cfg.sw,
        cfg.softcap,
        store.group(),
        cfg.layers - n_full,
        cfg.layers,
        store.n_lins()
    );
    let body = rep.summary();
    if rep.ok() {
        Ok(format!("{head}\n{body}  판정: 완전성 검증 통과"))
    } else {
        Err(format!("{head}\n{body}  판정: 검증 실패"))
    }
}

/// 포맷 계층 리포트 — 아치 바인딩 미등록 모델(신규 도입 트랙).
/// 검증 = 트리플 구조(open) + weight_shape 값 대조 + 양자 파라미터 표기.
fn format_store_level(
    dir: &str,
    store: &llm170_core::w4a16::W4a16Model,
    arch: Option<&str>,
) -> Result<String, String> {
    let head = format!(
        "w4a16-load {dir}\n  양자화=compressed-tensors pack-quantized int4 sym g{} (zp=8 상수) · 선형 {}개",
        store.group(),
        store.n_lins()
    );
    let bad = store.check_shape_values().map_err(|e| e.to_string())?;
    let body = format!(
        "  트리플 {}개 — 구조 계약 통과 · weight_shape 값 대조 불일치 {}건",
        store.n_lins(),
        bad.len()
    );
    if bad.is_empty() {
        Ok(format!(
            "{head}\n{body}\n  판정: 포맷 계층 통과 — 아치 바인딩 미등록({}): 커버리지(완전성)·엔진 검증은 도입 트랙",
            arch.unwrap_or("판별 불가")
        ))
    } else {
        Err(format!(
            "{head}\n{body}: {:?}\n  판정: 검증 실패",
            &bad[..bad.len().min(4)]
        ))
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
    // [gemma4 2026-10-14] --bos <id>: 코퍼스 선두에 BOS 부착. gemma4처럼
    // 채팅 템플릿·BOS 의존이 큰 모델은 raw 텍스트 NLL이 프로토콜 아티팩트로
    // 폭발한다(HF 대조로 확인) — 품질 게이트는 BOS 부착 프로토콜을 쓴다.
    let mut bos: Option<u32> = None;
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
            "--bos" => {
                bos = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or("--bos requires an id")?,
                );
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
    if let Some(c) = &corpus {
        // [Q1] CPU 오라클 NLL — 전 위치 teacher-forced, 전 어휘 logsumexp(f64).
        let text = std::fs::read_to_string(c).map_err(|e| format!("corpus {c}: {e}"))?;
        let tk = crate::tokenize::Tokenizer::load(std::path::Path::new(&dir))?;
        let mut toks = tk.encode_opts(&text, false);
        if let Some(b) = bos {
            toks.insert(0, b);
        }
        let n = toks.len().min(limit);
        if n < 2 {
            return Err(format!("토큰 {n}개 — NLL에는 2개 이상 필요"));
        }
        let mut eng = RefEngine::load(&dir, 1, ctx)?;
        let t0 = std::time::Instant::now();
        let mut l = eng.prefill(0, &toks[..1])?;
        let mut nll_sum = 0f64;
        for i in 1..n {
            nll_sum += crate::probes::eval::nll_of(&l, toks[i]);
            let mut ls = eng.decode(0, &[toks[i]])?;
            l = ls.pop().ok_or("decode logits 부재")?;
        }
        let mean = nll_sum / (n - 1) as f64;
        let ppl = mean.exp();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        return Ok(format!(
            "w4a16-ref {dir} — CPU 오라클 NLL (teacher-forced · logsumexp f64)\n  corpus {c}{} · 토큰 {n} (NLL {}위치) · {ms:.0}ms ({:.0}ms/토큰)\n  NLL평균 {mean:.9} · PPL {ppl:.9}\n",
            bos.map(|b| format!(" · BOS {b}")).unwrap_or_default(),
            n - 1,
            ms / (n - 1) as f64
        ));
    }
    let n = prompts.len();
    let mut eng = RefEngine::load(&dir, n, ctx)?;
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (s, p) in prompts.iter().enumerate() {
        let l = eng.prefill(s, p)?;
        out[s].push(llm170_core::qwen35::greedy(&l));
    }
    for _ in 0..n_predict {
        for s in 0..n {
            let t = *out[s].last().expect("시드 토큰");
            let nt = eng.decode_greedy(s, t)?;
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

/// 참조 엔진 — 아치별 CPU 엔진(qwen35·gemma4) 디스패치(병렬 미러·trait 없음).
/// [gemma4 P2-3 2026-10-14] GPU 프로브의 오라클 대조가 쓰도록 공개(super).
pub(super) enum RefEngine {
    Qwen(llm170_core::qwen35::Engine),
    Gemma(llm170_core::gemma4::Engine),
}

impl RefEngine {
    pub(super) fn load(dir: &str, n_seqs: usize, ctx: usize) -> Result<Self, String> {
        let path = std::path::Path::new(dir);
        let arch = llm170_core::qwen35::bind::dir_arch(path).unwrap_or_default();
        if llm170_core::gemma4::bind::arch_ok(&arch) {
            let m = llm170_core::gemma4::Model::load(path).map_err(|e| e.to_string())?;
            Ok(RefEngine::Gemma(llm170_core::gemma4::Engine::new(
                m, n_seqs, ctx,
            )))
        } else {
            let m = llm170_core::qwen35::Model::load(path).map_err(|e| e.to_string())?;
            Ok(RefEngine::Qwen(llm170_core::qwen35::Engine::new(
                m, n_seqs, ctx,
            )))
        }
    }

    pub(super) fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        match self {
            RefEngine::Qwen(e) => e.prefill(seq, tokens).map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.prefill(seq, tokens).map_err(|x| x.to_string()),
        }
    }

    pub(super) fn decode(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        match self {
            RefEngine::Qwen(e) => e.decode(&[seq], tokens).map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.decode(&[seq], tokens).map_err(|x| x.to_string()),
        }
    }

    pub(super) fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        match self {
            RefEngine::Qwen(e) => e.decode_greedy(seq, token).map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.decode_greedy(seq, token).map_err(|x| x.to_string()),
        }
    }
}
