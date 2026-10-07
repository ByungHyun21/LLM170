//! infer — qwen35/qwen4exp greedy 추론 CLI (main.rs에서 이관, plans/35 P7).
//! JSONL {"seq","pos","token","text"} 스트림 출력.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::usage_err;
use std::path::Path;

pub(crate) fn cmd_infer(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut n_predict = 32usize;
    let mut ctx = 4096usize;
    let backend = ma.backend.clone().unwrap_or_else(|| "cpu".into());
    let gpu_runtime = ma
        .gpu_runtime
        .clone()
        .or_else(|| llm170_diag::flag::val("LLM170_GPU_RUNTIME").map(str::to_string))
        .unwrap_or_else(|| "hip".into());
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
    // EXL3 아카이브(디렉터리) — 포맷 자동 판별(사용자 계약 2026-10-05):
    // --backend는 런타임만 받고 모델 포맷은 경로로 결정. 단일 프롬프트만
    // 지원(엔진이 단일 슬롯) — 게이트(gate-exl3.sh)의 고정 토큰 러너.
    if model_path.is_dir() {
        if backend == "cpu" {
            return usage_err("EXL3(디렉터리)는 GPU 런타임 필요 — --backend hip|vulkan|cuda");
        }
        if prompts.len() > 1 {
            return usage_err("EXL3 infer는 단일 --prompt-tokens만 지원");
        }
        return run_exl3_infer(&model_path, &prompts[0], n_predict, ctx, &gpu_runtime);
    }
    // plans/cuda-port.md §1.3 S6 — GGUF+cuda는 Q4AccCuda 값경로로 진행한다
    // (attach_q4 cuda 분기). W4A16(safetensors)은 여전히 미지원 — S7.
    //
    // [S7 착수 전제 실측 — 2026-10-08, plans/cuda-port.md §1.3 전제 정정]
    // 플랜의 "W4A16 커널 비트일치 인증 완료"는 성립하지 않는다 — rawcuda·
    // rawhip에 w4a16/gptq 커널이 없고, core/quant/lane.rs의
    // dot_row_w4a16_lane(GPU gemm_gptq4 64레인 미러, plans/137 §3.5)은
    // 호출부 0개 미검증 상태다. 대상 모델 실측
    // (../models/Qwen3.8-27B-W4A16-AutoRound, 7파트+extra 전부 존재 —
    // 1999 텐서): arch가 Qwen3_5ForConditionalGeneration(qwen4exp 아님 —
    // qwen35 엔진 계열), quantization_config는 compressed-tensors
    // pack-quantized(int4 sym g128, weight_packed/scale/shape 3조),
    // linear_attn in_proj_a/b와 lm_head·visual은 미양자화 ignore.
    // 즉 S7의 실제 남은 일은 "매핑"이 아니라 (1) gemm_gptq4 CUDA 커널
    // 신규 작성+비트계약 확립, (2) compressed-tensors safetensors 로더,
    // (3) qwen35 계열 Engine 가속기 매핑이다 — 커널 프로젝트 규모.
    if gpu_runtime == "cuda"
        && !model_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        return usage_err("CUDA+GGUF는 S6 값경로 지원 — W4A16(safetensors)은 S7 대기");
    }
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    if max_prompt + n_predict + 8 >= ctx {
        return usage_err(&format!(
            "ctx({ctx}) too small for prompt({max_prompt})+n_predict({n_predict})"
        ));
    }

    llm170_diag::span::reset();
    let t_start = std::time::Instant::now();
    // 아키텍처 판별 → qwen4exp 전용 엔진 분기.
    // ENOENT 윈도우 대기 (LLM170_OPEN_WAIT_SECS) — 판별 실패시 재시도.
    let wait_secs: u64 = llm170_diag::flag::val("LLM170_OPEN_WAIT_SECS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut arch: Option<String> = None;
    for _ in 0..=wait_secs {
        arch = llm170_gguf::GgufFile::open(&model_path)
            .ok()
            .and_then(|g| g.arch().map(|s| s.to_string()));
        if arch.is_some() {
            break;
        }
        if wait_secs == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    if arch.as_deref() == Some("qwen4exp") {
        return run_q4_infer(
            &model_path,
            &prompts,
            n_predict,
            ctx,
            &backend,
            &gpu_runtime,
            spec_k,
            ma.mtp.as_deref(),
        );
    }
    let engine_res = llm170_core::qwen35::Model::load(&model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let n = prompts.len();
            let mut eng = llm170_core::qwen35::Engine::new(m, n, ctx);
            if spec_k.is_some() {
                eng.mtp_wanted = true; // 스펙 의도 — prefill 훅 활성 (plans/22)
            }
            // 백엔드 부착 — 단일 경로(attach_q35). LLM170_REQUIRE_GPU=1이면 폴백
            // 금지(2026-09-12: infer 검증이 폴백으로 통과한 사고 방지).
            let policy = if llm170_diag::flag::on("LLM170_REQUIRE_GPU") {
                crate::engine::AttachPolicy::Strict
            } else {
                crate::engine::AttachPolicy::Warn
            };
            // QA-17: backend 문자열 반영 — 종전 --backend cpu가 무시돼 GPU
            // 부착 결과를 cpu로 취급했다(vl 패턴과 동일 계약).
            if backend != "cpu" {
                eng = crate::engine::attach_q35(eng, gpu_runtime == "vulkan", policy)
                    .map_err(|e| format!("GPU 백엔드 주입 실패(REQUIRE_GPU): {e}"))?;
            }
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

/// qwen4exp 추론 — Engine4 (시퀀스별 prefill/decode1).
fn run_q4_infer(
    model_path: &Path,
    prompts: &[Vec<u32>],
    n_predict: usize,
    ctx: usize,
    backend: &str,
    gpu_runtime: &str,
    spec_k: Option<usize>,
    mtp_arg: Option<&str>,
) -> ExitCode {
    let t_start = std::time::Instant::now();
    let want_gpu = crate::engine::q4_gpu_wanted_str(backend, gpu_runtime);
    let res = llm170_core::qwen4exp::Model4::load(model_path)
        .map_err(|e| e.to_string())
        .and_then(|m| {
            let mut m = m;
            crate::engine::apply_mtp(
                &mut m,
                model_path,
                mtp_arg.map(std::path::Path::new),
                spec_k.unwrap_or(0),
            )
            .map_err(|e| e.to_string())?;
            let n = prompts.len();
            let sources = m.part_sources();
            let eng = llm170_core::qwen4exp::layers::Engine4::new(m, n, ctx);
            // GPU 부착 — 단일 경로(attach_q4, Strict: infer 검증은 폴백 금지).
            let mut eng = crate::engine::attach_q4(
                eng,
                sources,
                want_gpu,
                crate::engine::q4_vk_runtime_str(gpu_runtime),
                crate::engine::q4_cuda_runtime_str(gpu_runtime),
                false,
                crate::engine::AttachPolicy::Strict,
            )?;
            let eos = eng.model.eos;
            let mut finished = vec![false; n];
            let mut next: Vec<u32> = Vec::with_capacity(n);
            let k_spec = spec_k.unwrap_or(0);
            for (s, p) in prompts.iter().enumerate() {
                let l = eng.prefill(s, p).map_err(|e| e.to_string())?;
                // P15④: 드래프트 프리필 — 타깃 h 행 전체로 드래프트 KV 적립
                // (스펙 의도일 때만; 값경로 last_h_rows 사용).
                if k_spec > 0 && eng.model.has_mtp() {
                    eng.mtp_draft_prefill(s, p, 0).map_err(|e| e.to_string())?;
                }
                let t = llm170_core::qwen35::greedy(&l);
                println!(
                    "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                    p.len(),
                    crate::json::quoted(&eng.piece(t))
                );
                next.push(t);
                finished[s] = t == eos;
            }
            let mut pos: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            // QA-18: 시퀀스별 생성 총량 — 스펙 수용 토큰(≤k+1)을 검사 없이
            // emit해 ≤(k+1)×n_predict 초과 생성하던 결함의 상한.
            let mut gen_count: Vec<u32> = vec![0; n];
            let mut spec_stats = (0usize, 0usize); // (수용, forward)
            for _step in 0..n_predict {
                let active: Vec<usize> = (0..n).filter(|&s| !finished[s]).collect();
                if active.is_empty() {
                    break;
                }
                // plans/109 P15⑤: MTP 스펙(단일 시퀀스+greedy) — mtp_spec_step.
                if k_spec > 0 && eng.model.has_mtp() && active.len() == 1 {
                    let s = active[0];
                    let (acc, fwd) = eng
                        .mtp_spec_step(s, next[s], k_spec)
                        .map_err(|e| e.to_string())?;
                    spec_stats.0 += acc.len();
                    spec_stats.1 += fwd;
                    let mut stop = false;
                    for &t in &acc {
                        if gen_count[s] >= n_predict as u32 {
                            break;
                        }
                        pos[s] += 1;
                        gen_count[s] += 1;
                        println!(
                            "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                            pos[s],
                            crate::json::quoted(&eng.piece(t))
                        );
                        next[s] = t;
                        if t == eos {
                            finished[s] = true;
                            stop = true;
                            break;
                        }
                    }
                    let _ = stop;
                    continue;
                }
                // plans/73(np): 활성 2+ 는 배치 디코드(무게 스트리밍 공유).
                if active.len() > 1 {
                    let toks: Vec<u32> = active.iter().map(|&s| next[s]).collect();
                    let ls = eng
                        .decode_batch(&active, &toks)
                        .map_err(|e| e.to_string())?;
                    for (row, &s) in active.iter().enumerate() {
                        let t = llm170_core::qwen35::greedy(&ls[row]);
                        next[s] = t;
                        pos[s] += 1;
                        println!(
                            "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                            pos[s],
                            crate::json::quoted(&eng.piece(t))
                        );
                        finished[s] = t == eos;
                    }
                } else {
                    for &s in &active {
                        let d1g = !llm170_diag::flag::on("LLM170_NO_D1G");
                        let t = if !d1g {
                            let l = eng.decode1(s, next[s]).map_err(|e| e.to_string())?;
                            llm170_core::qwen35::greedy(&l)
                        } else {
                            eng.decode1_greedy(s, next[s]).map_err(|e| e.to_string())?
                        };
                        next[s] = t;
                        pos[s] += 1;
                        println!(
                            "{{\"seq\":{s},\"pos\":{},\"token\":{t},\"text\":{}}}",
                            pos[s],
                            crate::json::quoted(&eng.piece(t))
                        );
                        finished[s] = t == eos;
                    }
                }
            }
            if llm170_diag::dump::opts().alloc {
                llm170_diag::alloc::report();
            }
            if k_spec > 0 && spec_stats.1 > 0 {
                eprintln!(
                    "# spec(q4, k={k_spec}): 수용 {}토큰 / {} forward = {:.2} tok/fwd",
                    spec_stats.0,
                    spec_stats.1,
                    spec_stats.0 as f64 / spec_stats.1 as f64
                );
            }
            eprintln!(
                "# done(q4): {n} seqs, gen per seq: {n_predict} (elapsed {:.1?})",
                t_start.elapsed()
            );
            Ok(())
        });
    if let Err(e) = res {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
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

/// EXL3 아카이브 infer — 포맷 자동 판별 경로(사용자 계약 2026-10-05).
/// --backend는 런타임만 받는다: hip→Exl3Hip, vulkan→Exl3(vk), cuda→Exl3Cuda.
/// gate-exl3.sh 고정 토큰 게이트의 러너 — JSONL 형식은 q35 emit과 동일
/// ({{"seq","pos","token","text"}})해 게이트 grep이 양쪽 공용이다.
fn run_exl3_infer(
    dir: &std::path::Path,
    prompt: &[u32],
    n_predict: usize,
    ctx: usize,
    gpu_runtime: &str,
) -> ExitCode {
    if prompt.len() + n_predict + 8 >= ctx {
        eprintln!(
            "error: ctx({ctx}) too small for prompt({})+n_predict({n_predict})",
            prompt.len()
        );
        return ExitCode::FAILURE;
    }
    let dir_s = dir.to_string_lossy().into_owned();
    let tok = crate::tokenize::Tokenizer::load(dir, None).ok();
    let piece = |t: u32| -> String {
        match &tok {
            Some(tk) => String::from_utf8_lossy(&tk.piece_bytes(t)).into_owned(),
            None => String::new(),
        }
    };
    enum E {
        Vk(Box<crate::exl3_engine::Exl3Engine>),
        Hip(Box<crate::exl3_hip_engine::Exl3HipEngine>),
        Cuda(Box<crate::exl3_cuda_engine::Exl3CudaEngine>),
    }
    impl E {
        fn prefill(&mut self, toks: &[u32]) -> Result<Vec<f32>, String> {
            match self {
                E::Vk(e) => e.prefill(0, toks),
                E::Hip(e) => e.prefill(toks),
                E::Cuda(e) => e.prefill(0, toks),
            }
        }
        fn decode1(&mut self, t: u32) -> Result<Vec<f32>, String> {
            match self {
                E::Vk(e) => e.decode1(0, t),
                E::Hip(e) => e.decode1(t),
                E::Cuda(e) => e.decode1(0, t),
            }
        }
    }
    let mut eng = if gpu_runtime == "vulkan" {
        match crate::exl3_engine::Exl3Engine::load(&dir_s, 1, ctx) {
            Ok(e) => E::Vk(Box::new(e)),
            Err(e) => {
                eprintln!("error: exl3(vk) 로드 실패: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else if gpu_runtime == "hip" {
        match crate::exl3_hip_engine::Exl3HipEngine::load(&dir_s, 1, ctx) {
            Ok(e) => E::Hip(Box::new(e)),
            Err(e) => {
                eprintln!("error: exl3-hip 로드 실패: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else if gpu_runtime == "cuda" {
        // plans/cuda-port.md S5: 명시적 CUDA 런타임만 디코더에 연결한다.
        match crate::exl3_cuda_engine::Exl3CudaEngine::load(&dir_s, 1, ctx) {
            Ok(e) => E::Cuda(Box::new(e)),
            Err(e) => {
                eprintln!("error: exl3-cuda 로드 실패: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        eprintln!("error: EXL3 GPU 런타임 미지원: {gpu_runtime} (hip|vulkan|cuda 필요)");
        return ExitCode::FAILURE;
    };
    let mut lg = match eng.prefill(prompt) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: prefill: {e}");
            return ExitCode::FAILURE;
        }
    };
    for pos in (prompt.len() as u32..).take(n_predict) {
        let t = llm170_core::qwen35::greedy(&lg);
        println!(
            "{{\"seq\":0,\"pos\":{pos},\"token\":{t},\"text\":{}}}",
            crate::json::quoted(&piece(t))
        );
        match eng.decode1(t) {
            Ok(l) => lg = l,
            Err(e) => {
                eprintln!("error: decode: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
