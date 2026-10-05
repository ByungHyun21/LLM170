//! bench — llama-bench 규격 PP/TG 측정 (2026-09-02).
//!
//! PP: 합성 pp 토큰 prefill 시간 → t/s. TG: prefill 후 tg 토큰 디코드 시간 → t/s.
//! 토큰은 수제 LCG(seed 0x1234_5678, 관례) — rand 금지. 워밍업 1회 + reps 회 측정.
//! qwen35(MTP --spec 포함)·qwen4exp(--frame 포함) 양쪽 대응.
//!
//! 구조 (plans/109 P5): cmd_bench = 인자 파싱·프리플라이트·프롬프트·아키텍처
//! 판별 → bench_q4 / bench_q35 측정 → median_summary → print_table.

use std::process::ExitCode;
use std::time::Instant;

fn usage_err_bench(msg: &str) -> ExitCode {
    eprintln!(
        "error: {msg}\n사용법: llm170 bench --model <gguf|exl3-dir> [--pp N] [--tg N] [--reps N] [--ctx N] [--backend cpu|hip|vulkan] [--spec k] [--np K]"
    );
    ExitCode::from(2)
}

/// 벤치 구성 — 측정 함수(bench_q4/bench_q35)에 한 번에 전달.
struct BenchCfg {
    model_path: std::path::PathBuf,
    backend: String,
    gpu_runtime: String,
    pp: usize,
    tg: usize,
    reps: usize,
    ctx: usize,
    spec_k: usize,
    np_slots: usize,
    prompt: Vec<u32>,
    /// --mtp 인자(plans/109 P15⑤) — None이면 자동 탐지.
    mtp: Option<String>,
}

/// LCG 합성 프롬프트 — np 측정에서 슬롯마다 **다른 시드**를 줘 프리픽스
/// 캐시 공유를 배제한다(같은 접두사면 캐시 적중으로 처리량이 부풀려진다).
fn lcg_prompt(len: usize, seed0: u64) -> Vec<u32> {
    let mut seed = seed0;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % 200_000) as u32
    };
    (0..len).map(|_| lcg()).collect()
}

pub fn cmd_bench(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut pp = 512usize;
    let mut tg = 128usize;
    let mut reps = 1usize;
    let mut ctx = 4096usize;
    let backend = ma.backend.clone().unwrap_or_else(|| "cpu".into());
    let gpu_runtime = ma
        .gpu_runtime
        .clone()
        .or_else(|| llm170_diag::flag::val("LLM170_GPU_RUNTIME").map(str::to_string))
        .unwrap_or_else(|| "hip".into());
    let mut spec_k = 0usize;
    let mut np_slots = 1usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pp" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => pp = v.clamp(8, 65536),
                None => return usage_err_bench("--pp requires a number"),
            },
            "--tg" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => tg = v.clamp(1, 4096),
                None => return usage_err_bench("--tg requires a number"),
            },
            "--reps" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => reps = v.clamp(1, 20),
                None => return usage_err_bench("--reps requires a number"),
            },
            "--ctx" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => ctx = v,
                None => return usage_err_bench("--ctx requires a number"),
            },
            "--spec" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) if (1..=8).contains(&k) => spec_k = k,
                _ => return usage_err_bench("--spec requires k in 1..=8"),
            },
            // np 슬롯 집계 — 서버 슬롯 루프와 **같은 엔진 API**로 직접 측정한다.
            // HTTP 계층의 워밍업·프리픽스 캐시·ctx 나눗셈을 배제한 기준선.
            "--np" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(k) if (1..=8).contains(&k) => np_slots = k,
                _ => return usage_err_bench("--np requires k in 1..=8"),
            },
            other => return usage_err_bench(&format!("unknown flag: {other}")),
        }
    }
    let Some(model_path): Option<std::path::PathBuf> =
        ma.model.clone().map(std::path::PathBuf::from)
    else {
        return usage_err_bench("--model required");
    };
    // plans/93: FS 프리플라이트 — inode 플래핑(대형 mmap 벤치마크 + GPU fault
    // 이력으로 유발된 무음 손상) 감지. 3회 연속 open 실패 시 즉시 중단해
    // 손상 상태에서의 추가 I/O를 막는다.
    {
        let mut flaky = 0;
        for _ in 0..3 {
            match std::fs::File::open(&model_path) {
                Ok(_) => {}
                Err(_) => flaky += 1,
            }
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        if flaky == 3 {
            eprintln!("오류: 모델 파일 열기 불안정(inode 손상 의심) — fs-preflight 실패.");
            eprintln!("  복구: sudo touch /forcefsck && sudo reboot");
            return ExitCode::FAILURE;
        }
    }
    if pp + tg + 16 >= ctx {
        return usage_err_bench(&format!("ctx({ctx}) too small for pp({pp})+tg({tg})"));
    }
    // 프롬프트: LLM170_BENCH_TEXT(자연어, Tokenizer 인코딩) 또는 수제 LCG 합성 토큰
    let prompt: Vec<u32> = match llm170_diag::flag::val("LLM170_BENCH_TEXT") {
        Some(txt) => {
            // A8(plans/129): panic → 오류 반환(usage_err_bench 패턴과 통일 —
            // 불완전 디렉터리 등 인위 오류 경로가 프로세스 패닉이었다).
            let tok = match crate::tokenize::Tokenizer::load(&model_path, None) {
                Ok(t) => t,
                Err(e) => return usage_err_bench(&format!("토크나이저 로드 실패: {e}")),
            };
            let mut ids = tok.encode(txt);
            // QA-23: 0토큰 인코딩 가드 — 빈 ids로 pp 패딩 루프가 무한 회전.
            if ids.is_empty() {
                return usage_err_bench("LLM170_BENCH_TEXT encoded to 0 tokens — refusing to pad");
            }
            // pp 길이에 맞게 자르기/반복
            ids.truncate(pp);
            while ids.len() < pp {
                let ext = ids.clone();
                ids.extend(ext);
                ids.truncate(pp);
            }
            ids
        }
        None => lcg_prompt(pp, 0x1234_5678),
    };

    // 아키텍처 판별 (ENOENT 재시도 관례)
    let mut arch: Option<String> = None;
    for _ in 0..5 {
        arch = llm170_gguf::GgufFile::open(&model_path)
            .ok()
            .and_then(|g| g.arch().map(|s| s.to_string()));
        if arch.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    let cfg = BenchCfg {
        model_path,
        backend,
        gpu_runtime,
        pp,
        tg,
        reps,
        ctx,
        spec_k,
        np_slots,
        prompt,
        mtp: ma.mtp.clone(),
    };
    let res_lines = if cfg.model_path.is_dir() {
        // EXL3 아카이브 디렉터리(plans/125-4) — GGUF 아키텍처 판별이 아닌
        // 디렉터리 여부로 판정. arch 변수는 GGUF 파일에만 유효하다.
        bench_exl3(&cfg)
    } else if arch.as_deref() == Some("qwen4exp") {
        bench_q4(&cfg)
    } else {
        bench_q35(&cfg)
    };
    let mut lines = match res_lines {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    median_summary(&mut lines);
    print_table(&lines, &cfg);
    ExitCode::SUCCESS
}

/// EXL3(아카이브 디렉터리) 측정 — plans/125-4: bench가 GGUF 아키텍처 판별에
/// 묶여 EXL3 dir을 거부하던 결함 수리. 프로토콜은 bench_q4와 동일(워밍업 1회 +
/// reps, pp=prefill, tg=순차 greedy decode1 — serve 단일슬롯 경로와 동일).
/// 백엔드: 포맷 자동 판별(2026-10-05) — dir→EXL3 엔진, 런타임은 gpu_runtime
/// (hip→Exl3Hip, vulkan→Exl3(vk)). 힙 수치 측정은 ROCm10 런타임으로.
fn bench_exl3(cfg: &BenchCfg) -> Result<Vec<String>, String> {
    let dir = cfg
        .model_path
        .to_str()
        .ok_or("exl3: 모델 경로가 utf8가 아님")?
        .to_string();
    enum Exl3 {
        Vk(Box<crate::exl3_engine::Exl3Engine>),
        Hip(Box<crate::exl3_hip_engine::Exl3HipEngine>),
    }
    impl Exl3 {
        fn prefill(&mut self, toks: &[u32]) -> Result<Vec<f32>, String> {
            match self {
                Exl3::Vk(e) => e.prefill(0, toks),
                Exl3::Hip(e) => e.prefill(toks),
            }
        }
        fn decode1(&mut self, tok: u32) -> Result<Vec<f32>, String> {
            match self {
                Exl3::Vk(e) => e.decode1(0, tok),
                Exl3::Hip(e) => e.decode1(tok),
            }
        }
        fn step_tok(&mut self, tok: u32) -> Result<u32, String> {
            match self {
                Exl3::Vk(e) => e.decode1(0, tok).map(|l| llm170_core::qwen35::greedy(&l)),
                Exl3::Hip(e) => e.step_tok(tok),
            }
        }
        fn spec_round(&mut self, k: usize) -> Result<Vec<u32>, String> {
            match self {
                Exl3::Vk(_) => Err("--spec은 hip 런타임만 지원(EXL3)".into()),
                Exl3::Hip(e) => e.spec_round(k),
            }
        }
        fn reset(&mut self) {
            match self {
                Exl3::Vk(e) => e.reset_states(),
                Exl3::Hip(e) => {
                    if let Err(err) = e.reset_seq() {
                        eprintln!("# hip reset_seq 실패: {err}");
                    }
                }
            }
        }
    }
    if cfg.backend == "cpu" {
        // 포맷 자동 판별 계약(2026-10-05): dir→EXL3는 GPU 런타임 필요.
        return Err("EXL3(디렉터리)는 GPU 런타임 필요 — --backend hip|vulkan".into());
    }
    let mut eng = if cfg.gpu_runtime == "vulkan" {
        Exl3::Vk(Box::new(crate::exl3_engine::Exl3Engine::load(
            &dir, 1, cfg.ctx,
        )?))
    } else {
        Exl3::Hip(Box::new(crate::exl3_hip_engine::Exl3HipEngine::load(
            &dir, 1, cfg.ctx,
        )?))
    };
    // 워밍업 1회 — 측정 형상과 동일(plans/79, llama-bench 정합).
    {
        let l = eng.prefill(&cfg.prompt)?;
        let t = llm170_core::qwen35::greedy(&l);
        let _ = eng.decode1(t)?;
    }
    eng.reset();
    let mut lines = Vec::new();
    for r in 0..cfg.reps {
        eng.reset();
        let t0 = std::time::Instant::now();
        let l = eng.prefill(&cfg.prompt)?;
        let pp_ms = t0.elapsed().as_secs_f64() * 1e3;
        lines.push(format!(
            "pp{} gpu | rep{r} | {pp_ms:8.1} ms | {:7.2} t/s",
            cfg.pp,
            cfg.pp as f64 / (pp_ms / 1e3)
        ));
        let mut next = llm170_core::qwen35::greedy(&l);
        let t1 = std::time::Instant::now();
        let mut n_gen = 0usize;
        // 스펙 경로(plans/130 D2): hip + --spec k — MTP 라운드(롤백 포함).
        if cfg.spec_k > 0 {
            let (mut n_round, mut n_emit) = (0usize, 0usize);
            while n_gen < cfg.tg {
                let toks = eng.spec_round(cfg.spec_k)?;
                if let Some(&t) = toks.last() {
                    next = t;
                }
                n_gen += toks.len();
                n_round += 1;
                n_emit += toks.len();
            }
            eprintln!(
                "  [spec] 라운드 {n_round} · 배출 {n_emit} ({:.2}/라운드, 드래프트 k={})",
                n_emit as f64 / n_round as f64,
                cfg.spec_k
            );
        } else {
            while n_gen < cfg.tg {
                next = eng.step_tok(next)?; // serve greedy 경로와 동일(plans/130 A2)
                n_gen += 1;
            }
        }
        let tg_ms = t1.elapsed().as_secs_f64() * 1e3;
        lines.push(format!(
            "tg{} gpu | rep{r} | {tg_ms:8.1} ms | {:7.2} t/s (steps {n_gen}, gen {n_gen})",
            cfg.tg,
            n_gen as f64 / (tg_ms / 1e3)
        ));
    }
    Ok(lines)
}

/// qwen4exp 측정 — Engine4 prefill/decode1 greedy + np 배치.
fn bench_q4(cfg: &BenchCfg) -> Result<Vec<String>, String> {
    let BenchCfg {
        model_path,
        backend,
        gpu_runtime,
        pp,
        tg,
        reps,
        ctx,
        spec_k,
        np_slots,
        prompt,
        mtp: _,
    } = cfg;
    let mut lines = Vec::new();
    let m = llm170_core::qwen4exp::Model4::load(model_path).map_err(|e| e.to_string())?;
    let mut m = m;
    crate::engine::apply_mtp(
        &mut m,
        model_path,
        cfg.mtp.as_deref().map(std::path::Path::new),
        *spec_k,
    )?;
    let sources = m.part_sources();
    let eng = llm170_core::qwen4exp::layers::Engine4::new(m, *np_slots, *ctx);
    // GPU 부착 — 단일 경로(attach_q4, Strict: bench는 CPU 폴백하지
    // 않는다 — 폴백 수치가 GPU로 오인된 이력).
    let want_gpu = crate::engine::q4_gpu_wanted_str(backend, gpu_runtime);
    let mut eng = crate::engine::attach_q4(
        eng,
        sources,
        want_gpu,
        crate::engine::q4_vk_runtime_str(gpu_runtime),
        false,
        crate::engine::AttachPolicy::Strict,
    )?;
    let eos = eng.model.eos;
    // QA-16: --spec 계약 — 스펙 의도면 측정도 스펙 경로로. 종전엔 MTP 가중치만
    // 적재하고 tg 루프는 순수 decode1_greedy(스펙 아님)였다.
    let has_mtp = *spec_k > 0 && eng.model.has_mtp();
    let spec_desc = if has_mtp {
        format!(" spec{spec_k}")
    } else {
        String::new()
    };
    // 워밍업 1회 — 측정 형상과 동일하게(plans/79, llama-bench 정합).
    // QA-24: 측정 판(decode1_greedy)과 동일 형상 + 스펙 의도면 드래프트
    // 프리필·스펙 스텝도 예열.
    {
        let _ = eng.prefill(0, prompt).map_err(|e| e.to_string())?;
        if has_mtp && let Err(err) = eng.mtp_draft_prefill(0, prompt, 0) {
            // 엔진 슬롯 루프와 동일 계약 — 값경로 h행 전제라 프레임 프리필에선
            // Err이 날 수 있고 생략은 품질 저하일 뿐 정확성 무영향(프레임 스펙
            // 경로는 자립).
            eprintln!("# mtp prefill 생략({err})");
        }
        let warm_next = eng.decode1_greedy(0, 1u32).map_err(|e| e.to_string())?;
        if has_mtp {
            let _ = eng
                .mtp_spec_step(0, warm_next, *spec_k)
                .map_err(|e| e.to_string())?;
        }
    }
    // 라벨은 백엔드를 그대로 반영한다 — 프레임(ADR-0017)은 cubecl 제거로
    // 사라졌고, env를 "frame"으로 표기해 GPU 수치로 오인된 이력이 있다.
    let dev = if want_gpu { " gpu" } else { " cpu" };
    for r in 0..*reps {
        eng.reset_states();
        // KTRACE — 프레임 op/커널의 GPU 시간을 t/s 옆에서 확정한다.
        let t0 = Instant::now();
        let l = eng.prefill(0, prompt).map_err(|e| e.to_string())?;
        if has_mtp && let Err(err) = eng.mtp_draft_prefill(0, prompt, 0) {
            eprintln!("# mtp prefill 생략({err})");
        }
        let pp_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut next = llm170_core::qwen35::greedy(&l);
        // TG — 프레임 경로는 decode1 내부 분기
        let t1 = Instant::now();
        let mut n_gen = 0usize;
        let mut step = 0usize;
        let mut fwd = 0usize;
        // 디코드 스텝 KTRACE — 첫 스텝 1회만 덤프(2026-09-14, +27ms 비교용).
        if has_mtp {
            while n_gen < *tg {
                let (toks, tf) = eng
                    .mtp_spec_step(0, next, *spec_k)
                    .map_err(|e| e.to_string())?;
                fwd += tf;
                step += 1;
                for &t in &toks {
                    if n_gen >= *tg {
                        break;
                    }
                    next = t;
                    n_gen += 1;
                    if t == eos {
                        break;
                    }
                }
                if next == eos {
                    break;
                }
            }
        } else {
            while n_gen < *tg {
                // plans/74: greedy 벤치는 GPU argmax 판(서빙 경로와 동일).
                next = eng.decode1_greedy(0, next).map_err(|e| e.to_string())?;
                // 덤프는 스텝 **이후** — 이전 판은 스텝 전에 덤프해 빈 트레이스를
                // 출력했다(2026-09-18 수정). 스텝 1회분이 그대로 찍힌다.
                n_gen += 1;
                step += 1;
                fwd += 1;
                if next == eos {
                    break;
                }
            }
        }
        let tg_ms = t1.elapsed().as_secs_f64() * 1e3;
        let fr = dev;
        lines.push(format!(
            "pp{pp}{fr}{spec_desc} | rep{r} | {pp_ms:8.1} ms | {:7.2} t/s",
            *pp as f64 / (pp_ms / 1e3)
        ));
        let tail = if has_mtp {
            format!("fwd {fwd}, gen {n_gen}")
        } else {
            format!("steps {step}, gen {n_gen}")
        };
        lines.push(format!(
            "tg{tg}{fr}{spec_desc} | rep{r} | {tg_ms:8.1} ms | {:7.2} t/s ({tail})",
            n_gen as f64 / (tg_ms / 1e3)
        ));
    }
    // np 슬롯 집계 (np ≥ 2) — 프로토콜: 슬롯별 분리 프롬프트, 전 슬롯
    // 워밍업 1회(계측 제외), 프리필은 슬롯 순차(서버 슬롯 루프와 동일 순서).
    if *np_slots >= 2 {
        let prompts: Vec<Vec<u32>> = (0..*np_slots)
            .map(|s| {
                lcg_prompt(
                    *pp,
                    0x9e37_79b9_u64.wrapping_add((s as u64 + 1) * 0x2545_f491),
                )
            })
            .collect();
        eng.reset_states();
        for s in 0..*np_slots {
            let _ = eng
                .prefill(s, &prompts[s][..16.min(*pp)])
                .map_err(|e| e.to_string())?;
            let _ = eng.decode1_greedy(s, 1).map_err(|e| e.to_string())?;
        }
        eng.reset_states();
        let t0 = Instant::now();
        for s in 0..*np_slots {
            let _ = eng.prefill(s, &prompts[s]).map_err(|e| e.to_string())?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let total = np_slots * pp;
        lines.push(format!(
            "np{np_slots}-pp{pp}{dev} | {ms:8.1} ms | {:7.2} t/s aggregate ({total} tok, disjoint prompts)",
            total as f64 / (ms / 1e3)
        ));
        let mut next: Vec<u32> = vec![1u32; *np_slots];
        let t1 = Instant::now();
        for _ in 0..*tg {
            // plans/80-B: 배치 프레임 디코드(decode_batch_greedy) —
            // t=n_slots 행의 단일 forward로 무게 패스 공유. 종전 순차
            // decode1_greedy는 슬롯당 53ms×4=212ms/step였다.
            next = eng
                .decode_batch_greedy(&(0..*np_slots).collect::<Vec<_>>(), &next)
                .map_err(|e| e.to_string())?;
        }
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        let n_tok = np_slots * tg;
        lines.push(format!(
            "np{np_slots}-tg{tg}{dev} | {ms:8.1} ms | {:7.2} t/s aggregate ({n_tok} tok, {tg} steps)",
            n_tok as f64 / (ms / 1e3)
        ));
        // 사용자 지시(README FN 모드 행): spec+np 결합 셀 — q35판(QA-15 수리)
        // 과 동일 프로토콜. 위 np 측정이 상태를 소진했으므로 재프리필 후
        // 슬롯별 자기 greedy로 시드, 라운드마다 mtp_spec_step_multi(배치 검증).
        if has_mtp {
            let mut seeds: Vec<u32> = Vec::with_capacity(*np_slots);
            eng.reset_states();
            for s in 0..*np_slots {
                let l = eng.prefill(s, &prompts[s]).map_err(|e| e.to_string())?;
                seeds.push(llm170_core::qwen35::greedy(&l));
            }
            let k = (*spec_k).clamp(1, 8);
            // 멀티 라운드는 직전 라운드의 spec_h_prev 전제(라운드 시작 병합
            // 계약, layers.rs mtp_spec_round_rest) — 서빙은 1라운드 순차
            // 폴백이 채운다. 벤치도 각 슬롯 1스텝 웜(앵컵 밖)으로 동일하게.
            for s2 in 0..*np_slots {
                let _ = eng.mtp_spec_step(s2, seeds[s2], k);
            }
            let mut nexts = seeds;
            let mut done = vec![0usize; *np_slots];
            let mut total_gen = 0usize;
            let t_sn = Instant::now();
            while total_gen < tg * np_slots {
                let active: Vec<usize> = (0..*np_slots).filter(|&s| done[s] < *tg).collect();
                if active.is_empty() {
                    break;
                }
                let ns: Vec<u32> = active.iter().map(|&s| nexts[s]).collect();
                let (accs, fw_total) = eng
                    .mtp_spec_step_multi(&active, &ns, k)
                    .map_err(|e| e.to_string())?;
                let _ = fw_total;
                for (row, &s) in active.iter().enumerate() {
                    for &t in &accs[row] {
                        if done[s] >= *tg {
                            break;
                        }
                        nexts[s] = t;
                        done[s] += 1;
                        total_gen += 1;
                    }
                }
            }
            let el = t_sn.elapsed().as_secs_f64() * 1e3;
            lines.push(format!(
                "np{np_slots}-tg{tg}{dev} spec{k} | {el:8.1} ms | {:7.2} t/s aggregate spec (gen {total_gen})",
                total_gen as f64 / (el / 1e3)
            ));
        }
    }
    Ok(lines)
}

/// qwen35 측정 — spec 단일/np·병합·per-seq 변형 + np 집계.
fn bench_q35(cfg: &BenchCfg) -> Result<Vec<String>, String> {
    let BenchCfg {
        model_path,
        backend,
        gpu_runtime,
        pp,
        tg,
        reps,
        ctx,
        spec_k,
        np_slots,
        prompt,
        mtp: _,
    } = cfg;
    let mut lines = Vec::new();
    let m = llm170_core::qwen35::Model::load(model_path).map_err(|e| e.to_string())?;
    // plans/79: --np 플래그가 qwen35 집계도 지휘하게 통일 — 종전엔
    // LLM170_BENCH_NP env만 읽어 --np 4가 무시됐다(측정 도구 결함).
    let bench_np0 = (*np_slots).max(
        llm170_diag::flag::val("LLM170_BENCH_NP")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
    );
    let mut eng = llm170_core::qwen35::Engine::new(m, bench_np0, *ctx);
    if *spec_k > 0 {
        eng.mtp_wanted = true;
    }
    // GPU 부착 — 단일 경로(attach_q35, Strict: 폴백 수치의 GPU 오인 방지).
    // QA-17: backend 문자열 반영 — 종전 --backend cpu가 무시돼 GPU 측정을
    // cpu로 미스라벨했다(vl의 `backend != "cpu"` 패턴과 동일 계약).
    if backend == "gpu" {
        eng = crate::engine::attach_q35(
            eng,
            gpu_runtime == "vulkan",
            crate::engine::AttachPolicy::Strict,
        )?;
    }
    let has_mtp = eng.has_mtp();
    let spec_desc = if *spec_k > 0 && has_mtp {
        format!(" spec{spec_k}")
    } else {
        String::new()
    };
    // 워밍업 — llama-bench 프로토콜 정합(plans/79): 측정 전 동일 형상을
    // 1회 흘린다(64토큰만 데우던 종전 방식은 콜드 상태에서 ours만 불리).
    // QA-24: 측정 판(decode_greedy)과 동일 형상으로 — 종전 decode+호스트
    // greedy는 측정 커널(GPU argmax)을 데우지 못했다. 스펙 의도면 스펙
    // 경로도 1회 예열.
    {
        let _ = eng.prefill(0, prompt).map_err(|e| e.to_string())?;
        let warm_next = eng.decode_greedy(0, 1u32).map_err(|e| e.to_string())?;
        if *spec_k > 0 && has_mtp {
            let _ = eng
                .spec_step(0, warm_next, *spec_k)
                .map_err(|e| e.to_string())?;
        }
    }
    for r in 0..*reps {
        eng.reset_states();
        let t0 = Instant::now();
        let l = eng.prefill(0, prompt).map_err(|e| e.to_string())?;
        let pp_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut next = llm170_core::qwen35::greedy(&l);
        let t1 = Instant::now();
        let mut n_gen = 0usize;
        let mut fwd = 0usize;
        let bench_np = bench_np0;
        // QA-15: spec+np 셀 시드 — np-pp 프리필의 슬롯별 마지막 greedy.
        let mut np_seeds: Vec<u32> = Vec::new();
        if bench_np > 1 {
            // np **프리필 집계** —슬롯별 분리 프롬프트(프리픽스 캐시
            // 공유 배제), 전 슬롯 워밍업 1회(계측 제외), 슬롯 순차
            // (서버 슬롯 루프와 동일 순서). 종전 np 셀은 프리필 집계가
            // 없어 HTTP 측정에 의존했고, 프로토콜 차이로 수치가 어긋났다
            // (README 374 = 단일 스트림 값). 여기서 같은 엔진 API로 잰다.
            let prompts: Vec<Vec<u32>> = (0..bench_np)
                .map(|s| {
                    lcg_prompt(
                        *pp,
                        0x9e37_79b9_u64.wrapping_add((s as u64 + 1) * 0x2545_f491),
                    )
                })
                .collect();
            eng.reset_states();
            for s in 0..bench_np {
                let _ = eng
                    .prefill(s, &prompts[s][..16.min(*pp)])
                    .map_err(|e| e.to_string())?;
                let _ = eng.decode_greedy(s, 1).map_err(|e| e.to_string())?;
            }
            eng.reset_states();
            let want_seeds = *spec_k > 0 && has_mtp;
            let mut last_logits: Vec<Vec<f32>> = Vec::with_capacity(bench_np);
            let t_pp = Instant::now();
            for s in 0..bench_np {
                let l = eng.prefill(s, &prompts[s]).map_err(|e| e.to_string())?;
                if want_seeds {
                    last_logits.push(l);
                }
            }
            let el = t_pp.elapsed().as_secs_f64() * 1e3;
            let total = bench_np * pp;
            lines.push(format!(
                "pp{pp} np{bench_np} | rep{r} | {el:8.1} ms | {:7.2} t/s agg ({total} tok, disjoint)",
                total as f64 / (el / 1e3)
            ));
            if want_seeds {
                np_seeds = last_logits
                    .iter()
                    .map(|l| llm170_core::qwen35::greedy(l))
                    .collect();
            }
        }
        if bench_np > 1 && *spec_k == 0 {
            // plans/79: np 디코드 집계(qwen4exp 판 미러) — 전 슬롯 프리필은
            // 계측 제외, 슬롯 순차 t=1 디코드로 tg·np 토큰 생성.
            let prompts: Vec<Vec<u32>> = (0..bench_np)
                .map(|s2| {
                    lcg_prompt(
                        *pp,
                        0x9e37_79b9_u64.wrapping_add((s2 as u64 + 1) * 0x9e37_79b9),
                    )
                })
                .collect();
            eng.reset_states();
            for s2 in 0..bench_np {
                let _ = eng.prefill(s2, &prompts[s2]).map_err(|e| e.to_string())?;
            }
            let seqs: Vec<usize> = (0..bench_np).collect();
            let mut next: Vec<u32> = vec![1u32; bench_np];
            let mut n_gen = 0usize;
            let t_np = Instant::now();
            for _ in 0..*tg {
                // 서버 슬롯 루프와 동일한 다중 시퀀스 배치 디코드(decode) —
                // 순차 decode1은 np 집계 프로토콜이 아니다(4배 느림).
                let lg = eng.decode(&seqs, &next).map_err(|e| e.to_string())?;
                for s2 in 0..bench_np {
                    next[s2] = llm170_core::qwen35::greedy(&lg[s2]);
                    n_gen += 1;
                }
            }
            let el = t_np.elapsed().as_secs_f64() * 1e3;
            lines.push(format!(
                "np{bench_np}-tg{tg} | rep{r} | {el:8.1} ms | {:7.2} t/s aggregate (gen {n_gen})",
                n_gen as f64 / (el / 1e3)
            ));
        }
        if *spec_k > 0 && has_mtp && bench_np > 1 {
            // np×spec 병합(spec_step_multi). per-seq 독립 변형(SPEC_PERSEQ)은
            // 원장 115 부정 판정으로 plans/109 P6 삭제.
            // QA-15: 슬롯별 자기 시드(np-pp 프리필의 마지막 greedy) — 종전엔
            // 슬롯0의 greedy를 전 슬롯에 복제(슬롯0 KV와도 불일치).
            let mut nexts: Vec<u32> = np_seeds.clone();
            let mut done: Vec<usize> = vec![0; bench_np];
            let mut total_gen = 0usize;
            // QA-15: 앵커는 이 셀 직전 — 종전 t1은 np-pp 프리필 집계 전체를
            // tg 시간에 포함해 t/s를 대폭 과소했다.
            let t_sn = Instant::now();
            while total_gen < tg * bench_np {
                let active: Vec<usize> = (0..bench_np).filter(|&s2| done[s2] < *tg).collect();
                if active.is_empty() {
                    break;
                }
                let ns: Vec<u32> = active.iter().map(|&s2| nexts[s2]).collect();
                let acc = eng
                    .spec_step_multi(&active, &ns, *spec_k)
                    .map_err(|e| e.to_string())?;
                for (i, &s2) in active.iter().enumerate() {
                    for &t2 in &acc[i] {
                        if done[s2] >= *tg {
                            break;
                        }
                        nexts[s2] = t2;
                        done[s2] += 1;
                        total_gen += 1;
                    }
                }
            }
            let el = t_sn.elapsed().as_secs_f64() * 1e3;
            lines.push(format!(
                "tg{tg} spec{spec_k} np{bench_np} | rep{r} | {el:8.1} ms | {:7.2} t/s agg (gen {total_gen})",
                total_gen as f64 / (el / 1e3)
            ));
            // QA-15: continue 누락 — 공통 꼬리 중복 실행으로 pp 중복행 +
            // "0.00 t/s (fwd 0, gen 0)" 가짜행이 찍혔다.
            continue;
        } else if *spec_k > 0 && has_mtp {
            while n_gen < *tg {
                let (toks, tf) = eng.spec_step(0, next, *spec_k).map_err(|e| e.to_string())?;
                fwd += tf;
                for &t in &toks {
                    if n_gen >= *tg {
                        break;
                    }
                    if llm170_diag::dump::opts().key("spec_dump") {
                        eprintln!("SPEC_TOK {t}");
                    }
                    next = t;
                    n_gen += 1;
                }
            }
        } else if bench_np > 1 {
            // np 집계(스펙 없음) — llama-server np4 슬롯과 동일 조건으로
            // 전 슬롯에 같은 프롬프트를 프리필한 뒤(집계 시간 제외)
            // 배치 디코드로 tg*bench_np 생성.
            // P7.1(plans/92): 셀 진입 전 전 슬롯 리셋+재프리필 — 종전엔
            // pp·np-tg 셀의 캐리오버 상태 위에 슬롯1..3만 append
            // 프리필하고 슬롯0 pp의 stale 토큰으로 시드해 슬롯0 스트림이
            // 갈림 → 부분 EOS → act.retain t=4→2 축소(측정 결함).
            eng.reset_states();
            let mut next_fresh = next;
            for s in 0..bench_np {
                let l = eng.prefill(s, prompt).map_err(|e| e.to_string())?;
                if s == 0 {
                    next_fresh = llm170_core::qwen35::greedy(&l);
                }
            }
            let t_np = Instant::now();
            let mut nexts: Vec<u32> = vec![next_fresh; bench_np];
            let mut act: Vec<usize> = (0..bench_np).collect();
            while n_gen < tg * bench_np {
                let ns: Vec<u32> = act.iter().map(|&s| nexts[s]).collect();
                // np greedy — logits 전사 회피 (plans/74 N1)
                let l = eng.decode_np_greedy(&act, &ns).map_err(|e| e.to_string())?;
                let mut eos: Vec<usize> = Vec::new();
                for (i, &s) in act.iter().enumerate() {
                    nexts[s] = l[i];
                    n_gen += 1;
                    if nexts[s] == llm170_core::qwen35::EOS_EOT {
                        eos.push(s);
                    }
                }
                // EOS 시퀀스 퇴출(집계 지속) — 남은 시퀀스만 다음 배치 참여
                if !eos.is_empty() && eos.len() < act.len() {
                    act.retain(|s| !eos.contains(s));
                }
            }
            let el = t_np.elapsed().as_secs_f64() * 1e3;
            lines.push(format!(
                "tg{tg} np{bench_np} | rep{r} | {el:8.1} ms | {:7.2} t/s agg (gen {n_gen})",
                n_gen as f64 / (el / 1e3)
            ));
            continue;
        } else {
            while n_gen < *tg {
                next = eng.decode_greedy(0, next).map_err(|e| e.to_string())?;
                if llm170_diag::dump::opts().key("spec_dump") {
                    eprintln!("SPEC_TOK {next}");
                }
                n_gen += 1;
                fwd += 1;
                if next == llm170_core::qwen35::EOS_EOT {
                    break;
                }
            }
        }
        let tg_ms = t1.elapsed().as_secs_f64() * 1e3;
        lines.push(format!(
            "pp{pp}{spec_desc} | rep{r} | {pp_ms:8.1} ms | {:7.2} t/s",
            *pp as f64 / (pp_ms / 1e3)
        ));
        lines.push(format!(
            "tg{tg}{spec_desc} | rep{r} | {tg_ms:8.1} ms | {:7.2} t/s (fwd {fwd}, gen {n_gen}, {:.2} tok/fwd)",
            n_gen as f64 / (tg_ms / 1e3),
            n_gen as f64 / fwd.max(1) as f64
        ));
    }
    Ok(lines)
}

/// 108 P2 — reps 중앙값 요약: 라벨(" | rep" 이전)별 t/s를 모아
/// 중앙값·스프레드 한 줄 추가. 런간 편차 ±1.7%(원장 98)가 +0.6%급
/// A/B 차이를 못 가리는 판별 프로토콜.
fn median_summary(lines: &mut Vec<String>) {
    use std::collections::BTreeMap;
    let mut by_label: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for l in lines.iter() {
        let Some((label, rest)) = l.split_once("| rep") else {
            continue;
        };
        // "0 | 412.3 ms | 412.33 t/s ..." → 마지막 t/s 직전 수치.
        let Some(v) = rest
            .split("t/s")
            .next()
            .and_then(|s| s.rsplit('|').next())
            .and_then(|s| s.split_whitespace().last())
            .and_then(|s| s.parse::<f64>().ok())
        else {
            continue;
        };
        by_label
            .entry(label.trim_end().to_string())
            .or_default()
            .push(v);
    }
    for (label, mut vals) in by_label {
        if vals.len() < 2 {
            continue;
        }
        vals.sort_by(|a, b| a.total_cmp(b));
        let mid = vals.len() / 2;
        let median = if vals.len() % 2 == 1 {
            vals[mid]
        } else {
            (vals[mid - 1] + vals[mid]) / 2.0
        };
        let spread = (vals[vals.len() - 1] - vals[0]).abs() / median * 100.0;
        lines.push(format!(
            "{label} | median x{} | {:7.2} t/s (spread {spread:.1}%)",
            vals.len(),
            median
        ));
    }
}

/// llama-bench형 표 출력.
fn print_table(lines: &[String], cfg: &BenchCfg) {
    println!("model            | test         |       time |        rate");
    println!("-----------------+--------------+-----------+------------");
    let short = cfg
        .model_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let bemark = match cfg.backend.as_str() {
        "gpu" => format!("gpu:{}", cfg.gpu_runtime),
        "cpu" => "cpu".into(),
        other => other.to_string(),
    };
    for l in lines {
        // "pp512 | rep0 | ..." → 앞부분 파싱해 정렬
        println!("{:16} | {}", format!("{short}[{bemark}]"), l);
    }
}
