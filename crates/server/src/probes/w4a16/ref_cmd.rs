
use super::super::{arg, arg_str};

pub(super) fn load(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-load <dir> — 사용법: llm170 w4a16-load ../models/Qwen3.8-27B-W4A16-AutoRound"
                .into(),
        );
    }
    let arch = llm170_core::qwen35::bind::dir_arch(std::path::Path::new(&dir));
    if arch.as_deref().is_some_and(llm170_core::qwen4exp::arch_ok) {
        return llm170_core::qwen4exp::report(std::path::Path::new(&dir));
    }
    let store = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
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

pub(super) fn reference(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-ref <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N] [--ckpt <p> [--ckpt-every K]] [--resume <p>] — 참조(CPU) greedy"
                .into(),
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let mut ckpt_path: Option<String> = None;
    let mut resume_path: Option<String> = None;
    let mut ckpt_every: usize = 512;
    let (mut corpus, mut limit) = (None::<String>, usize::MAX);
    let mut nll_out: Option<String> = None;
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
            "--hi-prec" => {
                llm170_core::quant::exact::set_hi_prec(true);
            }
            "--nll-out" => {
                nll_out = Some(arg(&mut it, "--nll-out requires a path")?.to_string());
            }
            "--ckpt" => {
                ckpt_path = Some(arg(&mut it, "--ckpt requires a path")?.to_string());
            }
            "--resume" => {
                resume_path = Some(arg(&mut it, "--resume requires a path")?.to_string());
            }
            "--ckpt-every" => {
                ckpt_every = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--ckpt-every requires a number")?;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    let path = std::path::Path::new(&dir);
    let is_fn = llm170_core::qwen35::bind::dir_arch(path)
        .map(|a| llm170_core::qwen4exp::arch_ok(&a))
        .unwrap_or(false);
    if is_fn && prompts.is_empty() && corpus.is_none() {
        return llm170_core::qwen4exp::probe(path);
    }
    if prompts.is_empty() && corpus.is_none() {
        return Err("at least one --prompt-tokens 또는 --eval-corpus required".into());
    }
    if let Some(c) = &corpus {
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
        crate::resource::release_load_lock();
        let t0 = std::time::Instant::now();
        let mut l = eng.prefill(0, &toks[..1])?;
        let mut nll_sum = 0f64;
        let mut per_pos: Vec<f64> = Vec::with_capacity(n - 1);
        let mut argm: Vec<Vec<f32>> = Vec::with_capacity(n - 1);
        for i in 1..n {
            let v = crate::probes::eval::nll_of(&l, toks[i]);
            nll_sum += v;
            per_pos.push(v);
            argm.push(l.clone());
            let mut ls = eng.decode(0, &[toks[i]])?;
            l = ls.pop().ok_or("decode logits 부재")?;
            if i % 128 == 0 {
                progress(&format!("# ref: nll {i}/{n}"));
            }
        }
        if let Some(p) = &nll_out {
            let mut buf =
                String::from("# w4a16-ref nll dump v2\n# cols: pos target argmax nll margin\n");
            for (i, v) in per_pos.iter().enumerate() {
                let tgt = toks[i + 1];
                let (am, mg) = crate::probes::eval::argmax_margin(&argm[i]);
                buf.push_str(&format!("{}\t{tgt}\t{am}\t{v:.9}\t{mg:.9}\n", i + 1));
            }
            std::fs::write(p, buf).map_err(|e| format!("nll-out: {e}"))?;
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
    crate::resource::release_load_lock();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    if let Some(rp) = &resume_path {
        let (saved_prompts, saved_out) = ckpt_read(rp, &mut eng)?;
        if saved_prompts != prompts {
            return Err(format!(
                "resume: 프롬프트 불일치(체크포인트 {}시퀀스)",
                saved_prompts.len()
            ));
        }
        out = saved_out;
        for s in 0..n {
            progress(&format!(
                "# ref: resume {rp} s{s} pos={} 대체출 {}토큰",
                eng.seq_pos(s),
                out[s].len()
            ));
        }
    }
    for (s, p) in prompts.iter().enumerate() {
        let mut pos = eng.seq_pos(s) as usize;
        if pos > p.len() && out[s].is_empty() {
            return Err(format!(
                "s{s}: pos {pos} > 프롬프트 {}(출력 없음 — 손상)",
                p.len()
            ));
        }
        let save_now = |eng: &RefEngine, out: &[Vec<u32>]| -> Result<(), String> {
            if let Some(cp) = &ckpt_path {
                ckpt_write(cp, &prompts, out, eng)?;
            }
            Ok(())
        };
        const PCH: usize = 128;
        let mut last: Option<Vec<f32>> = None;
        let mut consumed = 0usize;
        while pos < p.len() {
            let nx = (pos + PCH).min(p.len());
            last = Some(eng.prefill(s, &p[pos..nx])?);
            consumed += nx - pos;
            pos = nx;
            progress(&format!("# ref: prefill s{s} {pos}/{}", p.len()));
            if ckpt_path.is_some() && consumed >= ckpt_every.max(1) {
                save_now(&eng, &out)?;
                consumed = 0;
            }
        }
        if out[s].is_empty() {
            let l = last.ok_or_else(|| format!("s{s}: 시드 없음(빈 출력+잔여 프리필 없음)"))?;
            out[s].push(llm170_core::qwen35::greedy(&l));
            progress(&format!("# ref: seed s{s} tok={}", out[s][0]));
        }
        if ckpt_path.is_some() {
            save_now(&eng, &out)?;
        }
        let target = out[s].len() + n_predict;
        let mut made = 0usize;
        while out[s].len() < target {
            let t = *out[s].last().expect("시드 토큰");
            let nt = eng.decode_greedy(s, t)?;
            out[s].push(nt);
            made += 1;
            progress(&format!(
                "# ref: decode s{s} {}/{} tok={nt}",
                out[s].len(),
                target
            ));
            if ckpt_path.is_some() && made.is_multiple_of(8) {
                save_now(&eng, &out)?;
            }
        }
        if ckpt_path.is_some() {
            save_now(&eng, &out)?;
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

pub(super) enum RefEngine {
    Qwen(llm170_core::qwen35::Engine),
    Gemma(llm170_core::gemma4::Engine),
    Fn(Box<llm170_core::qwen4exp::oracle::Oracle>),
}

impl RefEngine {
    pub(super) fn load(dir: &str, n_seqs: usize, ctx: usize) -> Result<Self, String> {
        let path = std::path::Path::new(dir);
        let arch = llm170_core::qwen35::bind::dir_arch(path).unwrap_or_default();
        if llm170_core::qwen4exp::arch_ok(&arch) {
            if n_seqs > 1 {
                return Err("FN 참조 오라클은 단일 시퀀스만 지원".into());
            }
            let o = llm170_core::qwen4exp::oracle::Oracle::new(path).map_err(|e| e.to_string())?;
            return Ok(RefEngine::Fn(Box::new(o)));
        }
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
            RefEngine::Fn(e) => {
                if seq != 0 {
                    return Err("FN 오라클 단일 시퀀스 위반".into());
                }
                e.forward(tokens).map_err(|x| x.to_string())
            }
        }
    }

    pub(super) fn decode(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        match self {
            RefEngine::Qwen(e) => e.decode(&[seq], tokens).map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.decode(&[seq], tokens).map_err(|x| x.to_string()),
            RefEngine::Fn(e) => Ok(vec![e.forward(tokens).map_err(|x| x.to_string())?]),
        }
    }

    pub(super) fn decode_greedy(&mut self, seq: usize, token: u32) -> Result<u32, String> {
        match self {
            RefEngine::Qwen(e) => e.decode_greedy(seq, token).map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.decode_greedy(seq, token).map_err(|x| x.to_string()),
            RefEngine::Fn(e) => Ok(llm170_core::matmul::greedy_from(
                &e.forward(&[token]).map_err(|x| x.to_string())?,
            )),
        }
    }

    pub(super) fn seq_pos(&self, seq: usize) -> u32 {
        match self {
            RefEngine::Qwen(e) => e.seqs.get(seq).map(|s| s.pos).unwrap_or(0),
            RefEngine::Gemma(e) => e.seqs.get(seq).map(|s| s.pos).unwrap_or(0),
            RefEngine::Fn(e) => e.pos() as u32,
        }
    }

    pub(super) fn ckpt_save(&self) -> Result<Vec<u8>, String> {
        match self {
            RefEngine::Qwen(e) => e.ckpt_save().map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.ckpt_save().map_err(|x| x.to_string()),
            RefEngine::Fn(_) => Err("FN 참조 체크포인트 미지원(D3 접두 체크포인트 트랙)".into()),
        }
    }

    pub(super) fn ckpt_load(&mut self, blob: &[u8]) -> Result<(), String> {
        match self {
            RefEngine::Qwen(e) => e.ckpt_load(blob).map_err(|x| x.to_string()),
            RefEngine::Gemma(e) => e.ckpt_load(blob).map_err(|x| x.to_string()),
            RefEngine::Fn(_) => Err("FN 참조 체크포인트 미지원(D3 접두 체크포인트 트랙)".into()),
        }
    }
}

fn progress(msg: &str) {
    eprintln!("{msg}");
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

const CK_MAGIC: &[u8; 8] = b"REFCKP01";

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn get_u32(b: &[u8], p: &mut usize) -> Result<u32, String> {
    if *p + 4 > b.len() {
        return Err(format!("resume: 파일 잘림(offset {p})"));
    }
    let x = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4;
    Ok(x)
}

fn ckpt_write(
    path: &str,
    prompts: &[Vec<u32>],
    out: &[Vec<u32>],
    eng: &RefEngine,
) -> Result<(), String> {
    let core = eng.ckpt_save()?;
    let mut v = Vec::new();
    v.extend_from_slice(CK_MAGIC);
    put_u32(&mut v, prompts.len() as u32);
    for (p, o) in prompts.iter().zip(out.iter()) {
        put_u32(&mut v, p.len() as u32);
        for t in p {
            put_u32(&mut v, *t);
        }
        put_u32(&mut v, o.len() as u32);
        for t in o {
            put_u32(&mut v, *t);
        }
    }
    put_u32(&mut v, core.len() as u32);
    v.extend_from_slice(&core);
    let tmp = format!("{path}.tmp{}", std::process::id());
    std::fs::write(&tmp, &v).map_err(|e| format!("ckpt 기록 실패 {tmp}: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("ckpt 확정 실패 {path}: {e}"))?;
    progress(&format!("# ref: ckpt 저장 {path} ({}B)", v.len()));
    Ok(())
}

fn ckpt_read(path: &str, eng: &mut RefEngine) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), String> {
    let b = std::fs::read(path).map_err(|e| format!("resume 읽기 실패 {path}: {e}"))?;
    if b.len() < 8 || &b[..8] != CK_MAGIC {
        return Err(format!("resume: 매직 불일치 {path}"));
    }
    let mut p = 8usize;
    let n = get_u32(&b, &mut p)? as usize;
    if n == 0 || n > 1024 {
        return Err(format!("resume: 시퀀스 수 {n} 비정상"));
    }
    let mut prompts = Vec::with_capacity(n);
    let mut outs = Vec::with_capacity(n);
    for _ in 0..n {
        let pl = get_u32(&b, &mut p)? as usize;
        if pl > 1 << 24 || p + pl * 4 > b.len() {
            return Err("resume: 프롬프트 길이 비정상".into());
        }
        let mut pr = Vec::with_capacity(pl);
        for _ in 0..pl {
            pr.push(get_u32(&b, &mut p)?);
        }
        let ol = get_u32(&b, &mut p)? as usize;
        if ol > 1 << 24 || p + ol * 4 > b.len() {
            return Err("resume: 출력 길이 비정상".into());
        }
        let mut ov = Vec::with_capacity(ol);
        for _ in 0..ol {
            ov.push(get_u32(&b, &mut p)?);
        }
        prompts.push(pr);
        outs.push(ov);
    }
    let cl = get_u32(&b, &mut p)? as usize;
    if p + cl > b.len() {
        return Err("resume: 엔진 상태 잘림".into());
    }
    eng.ckpt_load(&b[p..p + cl])?;
    Ok((prompts, outs))
}
