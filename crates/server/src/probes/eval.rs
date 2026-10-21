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
//!                           [--chunk N] [--bos <id>] [--route] [--out dump.tsv] [--json]
//!   llm170 w4a16-eval ppl <w4a16_dir> --from-tokens <file> [...]
//!   llm170 w4a16-eval agree <a.tsv> <b.tsv>
//!   llm170 w4a16-eval needle <w4a16_dir> [--ctx N] [--lens N,N] [--sets N] [--json]

use super::{arg, arg_str, finish};
use llm170_backend_gpu::W4a16Dec;
// [품질 게이트 규약 2026-10-12/13 — 코드가 정본] 숫자 후보(KV·상태 dtype 등)의
// 채택/기각 판정(단일 경로 교체제 — 채택 시 골든 재동결+사유·델타 기록):
// ① 단문 PPL 쌍(창별 en/ko) + `ci` 블록 부트스트랩 CI — 밴드 |ΔPPL| ≤ 0.3%(초과 = 기각).
// ② 장문 PPL 쌍(mix 8K/16K/32K · chunk16) + `ci` — **단문 창은 장문 손상을 과소평가**
//    (실측: GDN 상태 bf16·35B — 128창 +0.72% vs 32K창 **+10.8%**, 15×; guide와 무관한
//    순환 누적형 손상은 장문에서만 드러남).
// ③ 장문 AND: `needle`(MRCR-lite 다중 바늘·--hard·confused 검출) — PPL이 못 보는 회수
//    실패 검출(역도 성립: needle이 못 잡는 누적 손상을 PPL 장문이 잡음 — 둘 다 필수).
// ④ 불변식: 골든(단문·장문)·`agree`(argmax 일치율).
// 업계 대비(2026-10-13 실코드 조사): vLLM = 태스크 정확도 0.88~0.95±0~8%p · KV는
// 옵트인 PPL 델타 공개(+1.17~2.71%) · SGLang = "reported GSM8K −5%p" 사전 등록 —
// 우리 밴드가 10~20× 타이트(폴백 없는 영구 교체의 대가 — 완화 논의는 장문 창+CI 증거로만).

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
            "w4a16-eval ppl <dir> --corpus <txt> [--out dump] — 예: w4a16-eval ppl ../models/Qwen3.8-27B-W4A16-AutoRound --corpus scripts/eval-data/corpus-en.txt --out /tmp/eval-a.tsv".into(),
        );
    }
    let (mut corpus, mut from_tokens, mut out_path) =
        (None::<String>, None::<String>, None::<String>);
    let (mut ctx, mut limit, mut json) = (8192usize, usize::MAX, false);
    let mut chunk = 1usize;
    let mut bos: Option<u32> = None;
    let mut route = false;
    let mut kld = false;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--corpus" => corpus = Some(arg(&mut it, "--corpus requires a path")?.to_string()),
            "--chunk" => {
                chunk = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--chunk requires a number")?;
                if !(1..=llm170_backend_gpu::G4_CHUNK_MAX).contains(&chunk) {
                    return Err(format!(
                        "--chunk {chunk}: 1..=G4_CHUNK_MAX({})",
                        llm170_backend_gpu::G4_CHUNK_MAX
                    ));
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
            // [gemma4 2026-10-14] --bos <id>: 코퍼스 선두 BOS 부착 — ref 명령
            // (`--bos`)과 동일 프로토콜(gemma4는 add_bos 강제 — HF 914/870 대조).
            "--bos" => {
                bos = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or("--bos requires an id")?,
                );
            }
            // [P4 2026-10-15] 라우팅 top-k 캡처(gemma4 전용 — MoE 진단·agree
            // Jaccard 소스). 그래프 캡처 비활성 직접 발사 경로(수치 불변).
            "--route" => route = true,
            // [§2.I 2026-10-16] top-k 로짓 덤프 — agree KLD 근사 소스.
            "--kld" => kld = true,
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
    let mut toks: Vec<u32> = if let Some(c) = &corpus {
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
    // [gemma4 2026-10-14] BOS 선두 부착(--bos) — ref 명령(`--bos`)과 동일 순서
    // (삽입 후 limit). gemma4는 add_bos 강제(프로토콜 정합 — 상대 편차가 게이트).
    if let Some(b) = bos {
        toks.insert(0, b);
    }
    let n = toks.len().min(limit);
    if n < 2 {
        return Err(format!("토큰 {n}개 — NLL 계산에는 2개 이상 필요"));
    }
    if n - 1 > ctx {
        return Err(format!("토큰 {n} > ctx {ctx} — --ctx 확대 필요"));
    }
    // [gemma4 2026-10-14] 아치 분기 — gemma4는 서빙 단일 경로(청크 t≤16·1..16
    // 동일 클래스)로 전 위치 로짓을 회수한다(경로 계측기 — 채택/기각 판정).
    let arch = llm170_core::qwen35::bind::dir_arch(Path::new(&dir)).unwrap_or_default();
    if kld && !route {
        return Err("--kld는 --route 동반 필수(덤프 열 순서 계약)".into());
    }
    if llm170_core::gemma4::bind::arch_ok(&arch) {
        return ppl_g4(&dir, &toks, n, ctx, chunk, &out_path, json, route, kld);
    }
    // [2026-10-21] FN(qwen4_exp) — 골든 정합 후 eval 인가 개통. 청크 128
    // 고정(전문가 LRU 320슬롯 distinct 가드 — 랜덤 코퍼스 512/256청크 초과
    // 실측), 전 위치 로짓 = 체인 + 헤드 GEMM(fn_chain_logits). route/kld는
    // FN 미지원(가드 유지).
    if llm170_core::qwen4exp::arch_ok(&arch) {
        if route || kld {
            return Err("FN은 --route/--kld 미지원(FN eval = ppl/agree 우선)".into());
        }
        return ppl_fn(&dir, &toks, n, ctx, chunk, &out_path, json);
    }
    if route {
        return Err("--route는 gemma4(MoE 라우팅) 전용".into());
    }
    if kld {
        return Err("--kld는 gemma4 전용(현 계측 축)".into());
    }
    let model = llm170_core::qwen35::Model::load(Path::new(&dir)).map_err(|e| e.to_string())?;
    // [B20 보완 2026-10-19] 적재 완료 = 락 해제.
    crate::resource::release_load_lock();
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    crate::gpu_engine::upload_model(&mut dec, &model, ctx, 0, None)?;
    let head = model
        .w("output.weight")
        .ok_or_else(|| "output.weight 부재".to_string())?;
    let head_gpu = head.ty == llm170_core::wtype::WType::Bf16;
    if head_gpu {
        dec.upload_head(head.data, head.n_out as usize, head.n_in as usize)?;
    }
    dec.reset_state(0)?;
    // [K6-3 결함수정 2026-10-13] 프로브 직접 경로는 스케줄러가 없어 KV 풀
    // 블록을 스스로 확보해야 한다 — 미확보 시 512토큰(블록 경계) 초과에서
    // 테이블 범위 밖 접근(illegal access, 실측). 창 전체(+슬랙)를 1회 보장.
    dec.kv_grow(0, n + 8)?;
    let vocab = head.n_out as usize;
    let mut rows: Vec<Row> = Vec::with_capacity(n - 1);
    let mut nll_sum = 0f64;
    if chunk >= 2 {
        // [A2 2026-10-15] qwen 배치 청크 = FFMA GEMM 경로(≤GEMM_FFMA_TMAX) —
        // gemma4(ppl_g4)만 G4_CHUNK_MAX(64)까지 허용한다.
        if chunk > llm170_backend_gpu::GEMM_FFMA_TMAX {
            return Err(format!(
                "--chunk {chunk}: qwen 배치 ≤ GEMM_FFMA_TMAX({})",
                llm170_backend_gpu::GEMM_FFMA_TMAX
            ));
        }
        // [2026-10-10] 청크 경로 — t행 배치 프리필 + 전 위치 로짓 회수
        // (GEMM t≥2 — t=1 GEMV와 수치 계급이 다름: 방법 B).
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
    eval_report(
        &dir, n, chunk, head_gpu, &rows, nll_sum, ms, &out_path, json, None, None,
    )
}

/// 덤프·요약 공용 — qwen·gemma4 경로가 같은 형식을 낸다(동결 — agree/ci 소비).
#[allow(clippy::too_many_arguments)]
fn eval_report(
    dir: &str,
    n: usize,
    chunk: usize,
    head_gpu: bool,
    rows: &[Row],
    nll_sum: f64,
    ms: f64,
    out_path: &Option<String>,
    json: bool,
    route: Option<(&[String], usize, usize)>,
    kld: Option<&[String]>,
) -> Result<String, String> {
    let mean = nll_sum / rows.len() as f64;
    let ppl = mean.exp();
    if let Some(p) = out_path {
        let mut s = String::with_capacity(rows.len() * 40 + 256);
        // [P4 2026-10-15] route 덤프 = 헤더 route=L{층}k{k} + 6열째(층순
        // top-k id 16진 — agree 자카드 소스). 형식 v1 호환(무route는 5열).
        if let Some((rs, nl, tk)) = route {
            s.push_str(&format!(
                "# w4a16-eval dump v1 model={dir} tokens={n} head={} chunk={chunk} nll_sum={nll_sum:.9} ppl={ppl:.9} route=L{nl}k{tk}{}\n",
                if head_gpu { "gpu" } else { "cpu" },
                if kld.is_some() { " kld=16" } else { "" }
            ));
            s.push_str(if kld.is_some() {
                "# cols: pos target argmax nll margin route kld\n"
            } else {
                "# cols: pos target argmax nll margin route\n"
            });
            match kld {
                Some(ks) if ks.len() == rows.len() => {
                    for (&(pos, tgt, am, nll, margin), (rt, kt)) in
                        rows.iter().zip(rs.iter().zip(ks.iter()))
                    {
                        s.push_str(&format!(
                            "{pos}\t{tgt}\t{am}\t{nll:.9}\t{margin:.9}\t{rt}\t{kt}\n"
                        ));
                    }
                }
                _ => {
                    for (&(pos, tgt, am, nll, margin), rt) in rows.iter().zip(rs) {
                        s.push_str(&format!(
                            "{pos}\t{tgt}\t{am}\t{nll:.9}\t{margin:.9}\t{rt}\n"
                        ));
                    }
                }
            }
        } else {
            s.push_str(&format!(
                "# w4a16-eval dump v1 model={dir} tokens={n} head={} chunk={chunk} nll_sum={nll_sum:.9} ppl={ppl:.9}\n",
                if head_gpu { "gpu" } else { "cpu" }
            ));
            s.push_str("# cols: pos target argmax nll margin\n");
            for &(pos, tgt, am, nll, margin) in rows {
                s.push_str(&format!("{pos}\t{tgt}\t{am}\t{nll:.9}\t{margin:.9}\n"));
            }
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

/// ppl — FN(qwen4_exp) 아치 분기 — 2026-10-21 골든 정합 후 개통.
/// 청크 = min(chunk.max(2),128) — 전문가 LRU 320슬롯 distinct 가드(랜덤
/// 코퍼스 512/256 초과 실측). 전 위치 로짓 = 체인 + 헤드 GEMM
/// (fn_chain_logits). 행 규약 = qwen35/ppl_g4와 동일(agree/ci 소비 동일).
fn ppl_fn(
    dir: &str,
    toks: &[u32],
    n: usize,
    ctx: usize,
    chunk: usize,
    out_path: &Option<String>,
    json: bool,
) -> Result<String, String> {
    use llm170_core::qwen4exp::stages::ple;
    let m = llm170_core::qwen4exp::engine::Model::load(Path::new(dir))?;
    crate::resource::release_load_lock();
    let cfg = m.cfg.clone();
    let ct_dir = format!("{dir}-CT");
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, cfg.hidden_size, cfg.num_hidden_layers)?;
    crate::gpu_engine::upload_fn(&mut dec, &cfg, Path::new(&ct_dir), ctx)?;
    // 기본 --chunk=1이면 FN은 128로 승격(전문가·KV 캐시 가드 상한). 명시
    // chunk≥2는 128로 클램프.
    let chunk_max = if chunk >= 2 { chunk.min(128) } else { 128 };
    let vocab = cfg.vocab;
    let mut rows: Vec<Row> = Vec::with_capacity(n - 1);
    let mut nll_sum = 0f64;
    let mut i = 0usize;
    while i + 1 < n {
        let t = (n - i).min(chunk_max);
        let emb = m.embed(&toks[i..i + t])?;
        let pl = ple::gather_emb(&m, toks, i)?;
        let emb_flat: Vec<f32> = emb.iter().flatten().copied().collect();
        let ple_flat: Vec<f32> = pl.iter().flatten().copied().collect();
        let lg = dec.fn_chain_logits(0, i as u32, t, &emb_flat, Some(&ple_flat))?;
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
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    eval_report(
        dir, n, chunk_max, true, &rows, nll_sum, ms, out_path, json, None, None,
    )
}

/// ppl — gemma4 아치 분기(서빙 청크 경로 = 단일 클래스). 헤드·softcap은 청크
/// 내부(원문 embed 행별 int8 W8A16) — 청크 t는 1..=G4_CHUNK_MAX로 클램프
/// ([2.N t≤16 2026-10-15] 승격 상한과 동기 — 청크 8/16 동등성 계측용).
fn ppl_g4(
    dir: &str,
    toks: &[u32],
    n: usize,
    ctx: usize,
    chunk: usize,
    out_path: &Option<String>,
    json: bool,
    route: bool,
    kld: bool,
) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    let model = llm170_core::gemma4::Model::load(Path::new(dir)).map_err(|e| e.to_string())?;
    // [B20 보완 2026-10-19] 적재 완료 = 락 해제.
    crate::resource::release_load_lock();
    let hp = model.hp.clone();
    let h = hp.n_embd;
    let mut dec = W4a16Dec::new(1, h, hp.n_layer)?;
    let cfg = crate::gpu_engine::upload_model_g4(&mut dec, &model, ctx)?;
    for il in 0..hp.n_layer {
        dec.set_g4_pes(il, cfg.pes_row(il, hp.n_experts))?;
    }
    dec.kv_grow(0, n + 8)?;
    // [P4 2026-10-15] 라우팅 캡처 — 그래프 캡처와 양립 불가(g4_graph_ok가
    // 차단 → 직접 발사). 수치 불변(디바이스 moe_idx 판독만).
    let (nl, tk) = (hp.n_layer, hp.top_k.max(1));
    let mut routes: Vec<String> = Vec::new();
    // [§2.I 2026-10-16] KLD 근사 — top-16 로짓(id:f32비트) 열. 전어휘 저장
    // 과다(248k×4B/토큰), 합집합 top-k KL로 agree가 근사 판정.
    let mut klds: Vec<String> = Vec::new();
    if route {
        if hp.n_experts > 256 {
            return Err(format!(
                "route: n_experts {} > 256 (16진 2자리 인코딩 상한)",
                hp.n_experts
            ));
        }
        dec.route_capture = true;
    }
    let chunk = chunk.clamp(1, llm170_backend_gpu::G4_CHUNK_MAX);
    let mut rows: Vec<Row> = Vec::with_capacity(n - 1);
    let mut nll_sum = 0f64;
    let mut i = 0usize;
    while i + 1 < n {
        let t = (n - i).min(chunk);
        let mut rb: Vec<f32> = Vec::with_capacity(t * h);
        for &tok in &toks[i..i + t] {
            rb.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
        }
        let csv: Vec<usize> = vec![0; t];
        let cpos: Vec<u32> = (0..t).map(|r| (i + r) as u32).collect();
        let lgs =
            crate::gpu_engine::g4_chunk(&mut dec, &model, &cfg, &csv, &cpos, &rb, t, false, true)?;
        if route {
            // 청크분 = 층 실행 순서대로 [t][k] 세그먼트 nl개(캡처 계약).
            // 행 정렬 = rows 루프와 동일 가드(마지막 위치는 스코어링 없음).
            let seg = std::mem::take(&mut dec.route_out);
            if seg.len() != nl * t * tk {
                return Err(format!(
                    "route: 세그먼트 {} != {}층×{}행×{}k",
                    seg.len(),
                    nl,
                    t,
                    tk
                ));
            }
            for r in 0..t {
                if i + r + 1 >= n {
                    break;
                }
                let mut s = String::with_capacity(nl * tk * 2);
                for l in 0..nl {
                    let b = (l * t + r) * tk;
                    for v in &seg[b..b + tk] {
                        s.push_str(&format!("{v:02x}"));
                    }
                }
                routes.push(s);
            }
        }
        for (r, lr) in lgs.iter().enumerate() {
            let g = i + r;
            if g + 1 >= n {
                break;
            }
            let (am, margin) = argmax_margin(lr);
            let tgt = toks[g + 1];
            let nll = nll_of(lr, tgt);
            nll_sum += nll;
            if kld {
                let mut s = String::with_capacity(16 * 14);
                for (id, v) in topk_logits(lr, 16) {
                    s.push_str(&format!("{id:06x}{:08x}", v.to_bits()));
                }
                klds.push(s);
            }
            rows.push((g as u32 + 1, tgt, am, nll, margin));
        }
        dec.slot_pos[0] = (i + t) as u32;
        i += t;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    eval_report(
        dir,
        n,
        chunk,
        true,
        &rows,
        nll_sum,
        ms,
        out_path,
        json,
        if route { Some((&routes, nl, tk)) } else { None },
        if kld { Some(&klds) } else { None },
    )
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
    /// [P4 2026-10-15] 라우팅 캡처 덤프 — (n_layer, top_k)와 행별 hex 열.
    route: Option<(usize, usize)>,
    routes: Vec<String>,
    /// [§2.I 2026-10-16] top-16 로짓 덤프(헤더 kld=16) — agree KLD 소스.
    kld: bool,
    klds: Vec<String>,
}

fn read_dump(path: &str) -> Result<Dump, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("dump {path}: {e}"))?;
    let mut rows = Vec::new();
    let mut nll_sum = None;
    let mut route = None;
    let mut routes = Vec::new();
    let mut kld = false;
    let mut klds = Vec::new();
    for (ln, line) in s.lines().enumerate() {
        if line.starts_with('#') {
            if let Some(v) = line.split("nll_sum=").nth(1)
                && let Some(tok) = v.split_whitespace().next()
            {
                nll_sum = tok.parse::<f64>().ok();
            }
            // [§2.I] kld=16 플래그(라우트 행과 동일 헤더 라인 끝).
            if line.contains("kld=16") {
                kld = true;
            }
            // [P4] route=L{층}k{k} — 마지막 헤더 토큰(층·k 메타).
            if let Some(v) = line.split("route=").nth(1)
                && let Some(tok) = v.split_whitespace().next()
            {
                let (l, k) = tok
                    .trim_start_matches('L')
                    .split_once('k')
                    .ok_or_else(|| format!("{path}:{}: route 헤더 {tok}", ln + 1))?;
                route = Some((
                    l.parse().map_err(|e| format!("{path}:{}: {e}", ln + 1))?,
                    k.parse().map_err(|e| format!("{path}:{}: {e}", ln + 1))?,
                ));
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let want = match (route.is_some(), kld) {
            (true, true) => 7,
            (true, false) => 6,
            (false, true) => 6,
            (false, false) => 5,
        };
        if f.len() != want {
            return Err(format!("{path}:{}: 필드 {want}개 아님: {line}", ln + 1));
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
        if route.is_some() {
            routes.push(f[5].to_string());
        }
        if kld {
            klds.push(f[if route.is_some() { 6 } else { 5 }].to_string());
        }
    }
    if rows.is_empty() {
        return Err(format!("{path}: 유효 행 없음"));
    }
    if kld && klds.len() != rows.len() {
        return Err(format!(
            "{path}: kld 열 {} != 행 {}",
            klds.len(),
            rows.len()
        ));
    }
    if route.is_some() && routes.len() != rows.len() {
        return Err(format!(
            "{path}: route 열 {} != 행 {}",
            routes.len(),
            rows.len()
        ));
    }
    Ok(Dump {
        rows,
        nll_sum,
        route,
        routes,
        kld,
        klds,
    })
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
    // [P4 2026-10-15] 라우팅 Jaccard — 두 덤프의 MoE top-k 선택 일치율(§2.I
    // baseline: 라우팅 민감도의 직접 계측 — dense mma 기각 기록과 연결).
    let mut route_desc = String::new();
    if da.route.is_some() || db.route.is_some() {
        let (nl, k) = match (da.route, db.route) {
            (Some(x), Some(y)) if x == y => x,
            _ => return Err("route: 두 덤프의 route 헤더 불일치(양쪽 --route 필요)".into()),
        };
        if da.routes.len() != db.routes.len() {
            return Err(format!(
                "route: 행 수 불일치 a={} b={}",
                da.routes.len(),
                db.routes.len()
            ));
        }
        let width = nl * k * 2;
        let (mut jsum, mut jn, mut jflip) = (0f64, 0usize, 0usize);
        let mut lsum = vec![0f64; nl];
        for (ra, rb) in da.routes.iter().zip(db.routes.iter()) {
            let (ha, hb) = (route_ids(ra)?, route_ids(rb)?);
            if ha.len() != nl * k || hb.len() != nl * k || ra.len() != width || rb.len() != width {
                return Err(format!(
                    "route: 열 폭 불일치(a {} / b {} — L{nl}k{k} 기대)",
                    ra.len(),
                    rb.len()
                ));
            }
            for l in 0..nl {
                let j = route_jaccard(&ha[l * k..(l + 1) * k], &hb[l * k..(l + 1) * k]);
                jsum += j;
                jn += 1;
                lsum[l] += j;
                jflip += (j < 1.0) as usize;
            }
        }
        let (wl, wj) = lsum
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, v)| (i, v / da.routes.len() as f64))
            .unwrap_or((0, 1.0));
        route_desc = format!(
            "\n  라우팅 Jaccard: 쌍 {jn} · 평균 {:.4} · 플립 {} ({:.3}%) · 최악 층 L{wl} {:.4}",
            jsum / jn as f64,
            jflip,
            jflip as f64 / jn as f64 * 100.0,
            wj
        );
        // [§2.I 2026-10-16] top-16 근사 KL — 로짓 분포 편차의 직접 계측.
        if da.kld && db.kld {
            let mut ksum = 0f64;
            let mut kn = 0usize;
            let mut kmax = 0f64;
            for (ra, rb) in da.klds.iter().zip(db.klds.iter()) {
                let (ta, tb) = (dec_topk(ra), dec_topk(rb));
                if let (Some(ta), Some(tb)) = (ta, tb) {
                    let k = kld_topk(&ta, &tb);
                    ksum += k;
                    kmax = kmax.max(k);
                    kn += 1;
                }
            }
            if kn > 0 {
                route_desc.push_str(&format!(
                    "\n  로짓 KL(top-16 근사): 쌍 {kn} · 평균 {:.4} · 최대 {kmax:.4}",
                    ksum / kn as f64
                ));
            }
        }
    }
    Ok(format!(
        "w4a16-eval agree\n  {a}\n  {b}\n  위치 {n} · argmax 일치 {matches} ({rate:.3}%) · 발산 {div}개(실질 {real}·근접 {}) · {div_desc}\n  {flip_m} (실질 기준 margin≥{margin_cut})\n  NLL합 a={sum_a:.6} b={sum_b:.6} (델타 {d_nll:+.4}%)\n  PPL a={ppl_a:.6} b={ppl_b:.6} (델타 {d_ppl:+.4}%){route_desc}",
        div - real
    ))
}
fn route_ids(hex: &str) -> Result<Vec<u32>, String> {
    let b = hex.as_bytes();
    if !b.len().is_multiple_of(2) {
        return Err(format!("route: 홀수 길이 hex {hex}"));
    }
    (0..b.len() / 2)
        .map(|i| {
            let p =
                std::str::from_utf8(&b[i * 2..i * 2 + 2]).map_err(|_| "route: hex".to_string())?;
            u32::from_str_radix(p, 16).map_err(|e| format!("route: {e}"))
        })
        .collect()
}

/// [§2.I] kld 열 디코드 — (id, f32비트) 16쌍.
fn dec_topk(h: &str) -> Option<Vec<(u32, f32)>> {
    let b = h.as_bytes();
    if b.len() != 16 * 14 {
        return None;
    }
    let mut out = Vec::with_capacity(16);
    for i in 0..16 {
        let id = u32::from_str_radix(std::str::from_utf8(&b[i * 14..i * 14 + 6]).ok()?, 16).ok()?;
        let bits =
            u32::from_str_radix(std::str::from_utf8(&b[i * 14 + 6..i * 14 + 14]).ok()?, 16).ok()?;
        out.push((id, f32::from_bits(bits)));
    }
    Some(out)
}

/// top-k 집합 자카드 = |∩| / |A∪B| (동일 k 전제). 플립 정의 = J < 1.
fn route_jaccard(a: &[u32], b: &[u32]) -> f64 {
    let c = a.iter().filter(|x| b.contains(x)).count();
    let u = a.len() + b.len() - c;
    if u == 0 { 1.0 } else { c as f64 / u as f64 }
}

/// argmax + top1-2 마진 — CPU `greedy_from` 규약(엄격 비교, 동률=최저 인덱스).
/// [Q3 2026-10-12] needle 하네스 — 장문 검색 정확도(플랜 §5 Q3).
/// 장문 검색 게이트 — 합성 haystack(중성 문장 반복)에 고유 패스코드를 심고
/// greedy 답변이 **그 패스코드를 정확히** 꺼내는지 채점한다(결정적 LCG·greedy).
/// ① 단일 바늘(기본): --depths(기본 10/50/90%)별 삽입·질의.
/// ② 다중 바늘 MRCR-lite(--needles N): N개를 균등 분산 삽입, 슬롯별 질의 —
///    다른 바늘 오답 검출 = confused. --hard면 접두 4자 공유(근접 디스턴트랙터,
///    접두 매칭으론 못 맞히게).
/// 판정은 토큰 id 부분열 **또는 디토큰화 문자열 포함** — 생성 토큰화는 문맥
/// 의존이라 id만 보면 정답을 놓친다(실측: 35B 오판). 문맥 프리필은 ppl 청크
/// 경로와 동일(KV 누적 — 수치 계급 동일). 판정 결과·프로토콜은 코드 옆이 정본.
fn needle(args: &[String]) -> Result<String, String> {
    let mut it = args.iter();
    let Some(dir) = it.next().cloned() else {
        return Err("w4a16-eval needle <dir> [--ctx N] [--lens N,N] [--sets N] [--needles N] [--hard] [--depths d,d,..] [--gen N] [--json] — 예: w4a16-eval needle ../models/Qwen3.8-27B-W4A16-AutoRound --lens 16384,24576 --needles 3 --hard".into());
    };
    let (mut ctx, mut lens, mut sets, mut json) =
        (16384usize, vec![4096usize, 16384], 1usize, false);
    let mut ngen = 8usize; // greedy 생성 토큰(패스코드 탐색 창)
    let (mut pf, mut picks_dbg) = (512usize, false); // 프리필 청크 캡·답변 덤프
    let (mut needles, mut hard) = (1usize, false); // 다중 바늘·근접 디스턴트랙터
    let mut bos: Option<u32> = None; // [gemma4] 선두 BOS(ref·ppl --bos와 동일 프로토콜)
    let mut depths = vec![10usize, 50, 90];
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
            "--pf" => {
                pf = arg(&mut it, "--pf requires a number")?
                    .parse()
                    .map_err(|_| "--pf requires a number".to_string())?;
            }
            "--picks" => picks_dbg = true,
            "--gen" => {
                ngen = arg(&mut it, "--gen requires a number")?
                    .parse()
                    .map_err(|_| "--gen requires a number".to_string())?;
            }
            "--sets" => {
                sets = arg(&mut it, "--sets requires a number")?
                    .parse()
                    .map_err(|_| "--sets requires a number")?;
            }
            "--needles" => {
                needles = arg(&mut it, "--needles requires a number")?
                    .parse()
                    .map_err(|_| "--needles requires a number".to_string())?;
                if needles == 0 {
                    return Err("--needles는 1 이상".into());
                }
            }
            "--hard" => hard = true,
            "--depths" => {
                depths = arg(&mut it, "--depths requires d,d,..")?
                    .split(',')
                    .map(|v| {
                        v.parse::<usize>()
                            .map_err(|_| "--depths requires d,d,..".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--bos" => {
                bos = Some(
                    arg(&mut it, "--bos requires an id")?
                        .parse()
                        .map_err(|_| "--bos requires an id")?,
                );
            }
            "--json" => json = true,
            other => return Err(format!("w4a16-eval needle: unknown flag {other}")),
        }
    }
    // [gemma4 2026-10-14] 아치 분기 — 케이스 생성(프로토콜)은 공용, 실행만
    // 분리(클로저). gemma4 = 서빙 청크 경로(프리필 ≤8·그리디 t=1).
    let arch = llm170_core::qwen35::bind::dir_arch(Path::new(&dir)).unwrap_or_default();
    let is_g4 = llm170_core::gemma4::bind::arch_ok(&arch);
    let tk = crate::tokenize::Tokenizer::load(Path::new(&dir))?;
    let t0 = std::time::Instant::now();
    let mut run_case: Box<dyn FnMut(&[u32]) -> Result<Vec<u32>, String>> = if is_g4 {
        let model = llm170_core::gemma4::Model::load(Path::new(&dir)).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
        let cfg = crate::gpu_engine::upload_model_g4(&mut dec, &model, ctx)?;
        for il in 0..hp.n_layer {
            dec.set_g4_pes(il, cfg.pes_row(il, hp.n_experts))?;
        }
        Box::new(move |toks: &[u32]| -> Result<Vec<u32>, String> {
            dec.reset_state(0)?;
            // [K6-3 결함수정] 문맥+답변 KV 블록 확보 — 프로브는 스케줄러 부재.
            dec.kv_grow(0, toks.len() + 16)?;
            // 청크 프리필 — [2026-10-19] --pf 존중(기본 512 = 서빙 프리필 동일
            // 경로·t>16 마린 dense). 8 고정 시대의 판정과의 비교는 --pf 8으로.
            let mut i = 0usize;
            let mut lg: Vec<f32> = Vec::new();
            while i < toks.len() {
                let t = (toks.len() - i).min(pf.max(2));
                let last = i + t == toks.len();
                let mut rb: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
                for &tok in &toks[i..i + t] {
                    rb.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
                }
                let csv: Vec<usize> = vec![0; t];
                let cpos: Vec<u32> = (0..t).map(|r| (i + r) as u32).collect();
                let out = crate::gpu_engine::g4_chunk(
                    &mut dec, &model, &cfg, &csv, &cpos, &rb, t, last, true,
                )?;
                if last {
                    lg = out.into_iter().next().unwrap_or_default();
                }
                i += t;
            }
            // 그리디 ngen — 마지막 문맥 위치에서 이어감.
            let mut picks: Vec<u32> = Vec::with_capacity(ngen);
            let mut row = lg;
            let base = toks.len() as u32;
            for step in 0..ngen {
                let mut best = (0u32, f32::MIN);
                for (j, &v) in row.iter().enumerate() {
                    if v > best.1 {
                        best = (j as u32, v);
                    }
                }
                picks.push(best.0);
                let er = model.embed_row(best.0).map_err(|e| e.to_string())?;
                let cpos = [base + step as u32];
                let out = crate::gpu_engine::g4_chunk(
                    &mut dec,
                    &model,
                    &cfg,
                    &[0],
                    &cpos,
                    &er,
                    1,
                    false,
                    true,
                )?;
                row = out.into_iter().next().unwrap_or_default();
            }
            Ok(picks)
        })
    } else {
        let model = llm170_core::qwen35::Model::load(Path::new(&dir)).map_err(|e| e.to_string())?;
        let hp = model.hp.clone();
        let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
        crate::gpu_engine::upload_model(&mut dec, &model, ctx, 0, None)?;
        let head = model
            .w("output.weight")
            .ok_or_else(|| "output.weight 부재".to_string())?;
        if head.ty != llm170_core::wtype::WType::Bf16 {
            return Err("needle은 GPU head 전제(Bf16 head 필요)".into());
        }
        dec.upload_head(head.data, head.n_out as usize, head.n_in as usize)?;
        Box::new(move |toks: &[u32]| -> Result<Vec<u32>, String> {
            dec.reset_state(0)?;
            let n_ctx = toks.len() - 1;
            dec.kv_grow(0, toks.len() + 16)?;
            // 문맥 n-1토큰까지 배치 체인 프리필(무로짓·t≥2) — 마지막 토큰은
            // t=1 head로 예측(프리필/디코드 경계 동일).
            let mut i = 0usize;
            while i < n_ctx {
                let t = (n_ctx - i).clamp(2, pf.max(2));
                let mut rb: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
                for &tok in &toks[i..i + t] {
                    rb.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
                }
                dec.forward_prefill(0, &rb, t, false)?;
                i += t;
            }
            let mut row: Vec<f32> = dec.forward_device_head(
                0,
                &model.embed_row(toks[n_ctx]).map_err(|e| e.to_string())?,
            )?;
            let mut picks: Vec<u32> = Vec::with_capacity(ngen);
            for _ in 0..ngen {
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
            Ok(picks)
        })
    };

    // 합성 haystack — 중성 산책 문장 반복(코퍼스 의존 없음·결정적).
    const HAY: &str = "The narrow alpine trail bends past the old stone bridge and continues through the quiet valley below the snow line. ";
    // [장거리 렌즈 2026-10-13] 1400→4000(≈88K토큰) — 32K/64K 렌즈 지원.
    // [200K 렌즈 2026-10-17] 고정 4000 → lens 최대값 적응형(문장≈22토큰·마진 4K) —
    // 204K 렌즈(GDN 200K 재검증) 상향. 코퍼스 무의존·결정성 유지.
    let hay_reps = (lens.iter().max().unwrap_or(&16384) + 4096) / 22 + 1;
    let hay: Vec<u32> = tk.encode_opts(&HAY.repeat(hay_reps.max(4000)), false);
    let mut seed = 0x5EED_C0DE_1234_5678u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let alphabet = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let mut rows: Vec<(usize, usize, usize, bool, bool)> = Vec::new(); // len·깊이%·슬롯·hit·confused
    for &l in &lens {
        if l + 64 * needles + usize::from(bos.is_some()) > ctx {
            return Err(format!("len {l} + 질문 > ctx {ctx} — --ctx 확대 필요"));
        }
        for _set in 0..sets {
            // [MRCR-lite 2026-10-13] 패스코드 needles개 — --hard면 접두 4자 공유.
            let shared: String = if hard && needles > 1 {
                (0..4)
                    .map(|_| alphabet[(lcg() as usize) % alphabet.len()] as char)
                    .collect()
            } else {
                String::new()
            };
            let mut pcs: Vec<String> = Vec::with_capacity(needles);
            for _ in 0..needles {
                let mut w = shared.clone();
                while w.len() < 6 {
                    w.push(alphabet[(lcg() as usize) % alphabet.len()] as char);
                }
                pcs.push(w);
            }
            let m_id: Vec<Vec<u32>> = pcs
                .iter()
                .enumerate()
                .map(|(j, w)| {
                    tk.encode_opts(
                        &format!(" The secret passcode for slot {j} is {w}. "),
                        false,
                    )
                })
                .collect();
            let q_id: Vec<Vec<u32>> = (0..needles)
                .map(|j| {
                    tk.encode_opts(
                        &format!(
                            " What is the secret passcode for slot {j}? Reply with the passcode only:"
                        ),
                        false,
                    )
                })
                .collect();
            let word_id: Vec<Vec<u32>> = pcs.iter().map(|w| tk.encode_opts(w, false)).collect();
            // 삽입 위치 % — needles=1이면 --depths(호환), N>1이면 균등 분산.
            let positions: Vec<usize> = if needles == 1 {
                depths.clone()
            } else {
                (1..=needles).map(|j| (j * 100) / (needles + 1)).collect()
            };
            let sum_m: usize = m_id.iter().map(|m| m.len()).sum();
            let max_q = q_id.iter().map(|q| q.len()).max().unwrap_or(0);
            let hay_need = l.saturating_sub(sum_m + max_q);
            if hay.len() < hay_need {
                return Err(format!("haystack 부족: {} < {hay_need}", hay.len()));
            }
            // 케이스 = (마커 배치 인덱스, 질의 슬롯) — 단일 바늘은 깊이별 슬롯0,
            // 다중 바늘은 문맥(마커 전부)을 조립한 뒤 슬롯별 질의.
            let cases: Vec<(usize, usize)> = if needles == 1 {
                (0..positions.len()).map(|i| (i, 0)).collect()
            } else {
                (0..needles).map(|j| (j, j)).collect()
            };
            for (pos_i, ask) in cases {
                let mut toks: Vec<u32> = Vec::with_capacity(l + 64);
                if needles == 1 {
                    let cut = hay_need * positions[pos_i] / 100;
                    toks.extend_from_slice(&hay[..cut]);
                    toks.extend_from_slice(&m_id[0]);
                    toks.extend_from_slice(&hay[cut..hay_need]);
                } else {
                    let mut prev = 0usize;
                    for (j, m) in m_id.iter().enumerate() {
                        let cut = hay_need * positions[j] / 100;
                        toks.extend_from_slice(&hay[prev..cut]);
                        toks.extend_from_slice(m);
                        prev = cut;
                    }
                    toks.extend_from_slice(&hay[prev..hay_need]);
                }
                toks.extend_from_slice(&q_id[ask]);
                // [gemma4 2026-10-14] --bos: 선두 BOS(ref·ppl과 동일 순서).
                if let Some(b) = bos {
                    toks.insert(0, b);
                }
                let picks = run_case(&toks)?;
                // [하네스 결함수정 2026-10-13] 토큰 id 부분열만으로는 생성-시점
                // 토큰화가 단독 인코딩과 다를 때 정답을 놓친다(실측: 35B가
                // "FAR39C"를 정확히 출력했는데도 0/6 — 병합 차이). 조각 문자열
                // 연결로 패스코드 원문(ASCII)도 함께 찾는다.
                let mut gen_text = String::new();
                for &id in &picks {
                    gen_text.push_str(tk.piece(id).unwrap_or(""));
                }
                let found: Vec<bool> = (0..needles)
                    .map(|j| {
                        picks
                            .windows(word_id[j].len())
                            .any(|w| w == &word_id[j][..])
                            || gen_text.contains(pcs[j].as_str())
                    })
                    .collect();
                let hit = found[ask];
                let confused = found.iter().enumerate().any(|(i, &h)| i != ask && h);
                let d = positions[pos_i];
                if picks_dbg {
                    eprintln!(
                        "[needle] len={l} d={d}% ask={ask} pcs={pcs:?} hit={hit} confused={confused} picks={picks:?}"
                    );
                }
                rows.push((l, d, ask, hit, confused));
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let hits = rows.iter().filter(|r| r.3).count();
    let conf = rows.iter().filter(|r| r.4).count();
    if json {
        let cells: Vec<String> = rows
            .iter()
            .map(|(l, d, s, h, c)| {
                format!(
                    "{{\"len\":{l},\"depth_pct\":{d},\"slot\":{s},\"hit\":{h},\"confused\":{c}}}"
                )
            })
            .collect();
        return Ok(format!(
            "{{\"model\":\"{dir}\",\"cases\":{},\"hits\":{hits},\"confused\":{conf},\"acc\":{:.4},\"elapsed_ms\":{ms:.0},\"rows\":[{}]}}",
            rows.len(),
            hits as f64 / rows.len().max(1) as f64,
            cells.join(",")
        ));
    }
    let mut s = format!(
        "needle {dir}: {hits}/{} hits ({:.1}%) · confused {conf}\n",
        rows.len(),
        100.0 * hits as f64 / rows.len().max(1) as f64
    );
    for (l, d, set, h, c) in &rows {
        s.push_str(&format!(
            "  len={l} depth={d}% slot={set} hit={h} confused={c}\n"
        ));
    }
    s.push_str(&format!("  elapsed {ms:.0}ms\n"));
    Ok(s)
}

/// [Q4 2026-10-12] 샘플링 검증 하네스 — 고정 LCG 로짓에서 서빙 CPU
/// 샘플러(splitmix64 시드 — rand 금지 계약)로 N표본 뽑고 이론
/// softmax(temperature·top_k/top_p/min_p 비활성) 분포와 TV 거리·
/// 카이제곱을 대조한다(판정 기준은 실행 출력 참조).
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

/// [§2.I 2026-10-16] top-k 로짓 쌍(id, 값) — KLD 근사 덤프용(전어휘 저장
/// 과다: 248k×4B/토큰). k=16: 질량 대부분 커버, agree에서 두 덤프의
/// 합집합 기준 근사 KL 계산.
fn topk_logits(lg: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..lg.len() as u32).collect();
    idx.sort_by(|&a, &b| lg[b as usize].total_cmp(&lg[a as usize]));
    idx.truncate(k);
    idx.into_iter().map(|i| (i, lg[i as usize])).collect()
}

/// 합집합 top-k 근사 KL(P‖Q) — 로그확도 정규화(전합 대신 top-k 합) 후
/// 누락 토큰은 하한(최소 로짓)으로 채움. 근사 계측(판정 보조).
fn kld_topk(a: &[(u32, f32)], b: &[(u32, f32)]) -> f64 {
    const FLOOR: f64 = -30.0;
    let la = |id: u32| {
        a.iter()
            .find(|x| x.0 == id)
            .map(|x| x.1 as f64)
            .unwrap_or(FLOOR)
    };
    let lb = |id: u32| {
        b.iter()
            .find(|x| x.0 == id)
            .map(|x| x.1 as f64)
            .unwrap_or(FLOOR)
    };
    // 정규화 상수(합집합).
    let ids: Vec<u32> = a.iter().map(|x| x.0).chain(b.iter().map(|x| x.0)).collect();
    let za = ids.iter().map(|&i| la(i).exp()).sum::<f64>().ln();
    let zb = ids.iter().map(|&i| lb(i).exp()).sum::<f64>().ln();
    ids.iter()
        .map(|&i| {
            let pa = (la(i) - za).exp();
            let qb = (lb(i) - zb).exp().max(1e-30);
            pa * ((pa / qb).ln())
        })
        .sum()
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

    #[test]
    fn route_ids_jaccard_roundtrip() {
        let ids = route_ids("00017ffe").unwrap();
        assert_eq!(ids, vec![0, 1, 127, 254]);
        assert!(route_ids("0").is_err());
        let a = [0u32, 1, 2, 3];
        let b = [0u32, 1, 2, 4];
        assert!((route_jaccard(&a, &a) - 1.0).abs() < 1e-12);
        assert!((route_jaccard(&a, &b) - 3.0 / 5.0).abs() < 1e-12);
        assert!(route_jaccard(&a, &[9u32, 10, 11, 12]).abs() < 1e-12);
    }

    #[test]
    fn dump_route_roundtrip() {
        let p = std::env::temp_dir().join(format!("eval-route-{}.tsv", std::process::id()));
        let s = "# w4a16-eval dump v1 model=x tokens=2 head=gpu chunk=16 nll_sum=2.5 ppl=2.11 route=L2k2\n# cols: pos target argmax nll margin route\n1\t5\t5\t1.0\t0.5\t00010203\n2\t7\t8\t1.5\t0.1\t04050607\n";
        std::fs::write(&p, s).unwrap();
        let d = read_dump(p.to_str().unwrap()).unwrap();
        assert_eq!(d.route, Some((2, 2)));
        assert_eq!(d.routes.len(), 2);
        assert_eq!(route_ids(&d.routes[0]).unwrap(), vec![0, 1, 2, 3]);
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
        let base: Vec<f64> = (0..128).map(|_| 2.0).collect();
        let mut cand = base.clone();
        for i in (0..128).step_by(2) {
            cand[i] += 0.05;
            cand[i + 1] -= 0.02;
        }
        let p1 = write_tmp_dump(&base, "ci-m1");
        let p2 = write_tmp_dump(&cand, "ci-m2");
        let out = ci(&[p1.clone(), p2.clone(), "--json".into()]).unwrap();
        assert!(
            out.contains("\"ci_nll_lo\"") && out.contains("\"ci_nll_hi\""),
            "{out}"
        );
        let lo: f64 = out
            .split("\"ci_nll_lo\":")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let hi: f64 = out
            .split("\"ci_nll_hi\":")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(lo <= hi);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    fn write_tmp_dump(nlls: &[f64], tag: &str) -> String {
        let p = std::env::temp_dir().join(format!("eval-{tag}-{}.tsv", std::process::id()));
        let mut s = String::from(
            "# w4a16-eval dump v1 model=x tokens=0 head=gpu nll_sum=0 ppl=0\n# cols: pos target argmax nll margin\n",
        );
        for (i, &n) in nlls.iter().enumerate() {
            s.push_str(&format!("{}\t{}\t0\t{}\t0.5\n", i + 1, 7, n));
        }
        std::fs::write(&p, s).unwrap();
        p.to_string_lossy().into_owned()
    }
}
