//! 원오프 GPU 프로브/체크 서브커맨드 — main.rs에서 이관(plans/35 P4).
//! 본체는 backend-gpu(rawhip 프로브 fn, rawvk check fn)에 있고 여기는
//! 인자 파싱+호출만. 결론난 A/B 하니스(batch-abtest·tree-test·q6k-abtest·
//! exp-ab)는 2026-09-08 폐기.
use std::process::ExitCode;

/// 위치 인자 규약 헬퍼 (plans/109 P5) — `args[i] | default` 파싱이
/// 디스패치 전체에 ~30번 손베껴져 있었다. 프로브 전용(경로/텐서명/수치).
fn arg_str(args: &[String], i: usize, d: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| d.into())
}
fn arg_num<T: std::str::FromStr>(args: &[String], i: usize, d: T) -> T {
    args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// 프로브 커맨드이면 실행해 Some(코드) 반환, 아니면 None.
pub fn run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    // 107 W4: 프로브 기본 모델 경로 통합(하드코딩 11곳 → 3상수).
    // 인자 우선 — 기본값은 진단 편의용.
    let d_fn =
        "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
    let d_27 = "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf";
    let d_q35 = "/home/yoon/models/qwen3.8-27b/q35work.gguf";
    let r: Result<String, String> = match cmd {
        "gpu-raw-probe" => llm170_backend_gpu::rawhip::raw_probe(arg_num(args, 0, 2000)),
        "launch-rate" => llm170_backend_gpu::rawhip::launch_rate(arg_num(args, 0, 20000)),
        "f16-bench" => llm170_backend_gpu::rawhip::f16_bench(
            arg_num(args, 0, 128),
            arg_num(args, 1, 2560),
            arg_num(args, 2, 640),
            arg_num(args, 3, 20),
        ),
        "q4k-bench" => llm170_backend_gpu::rawhip::q4k_bench(
            arg_num(args, 0, 128),
            arg_num(args, 1, 2560),
            arg_num(args, 2, 640),
            arg_num(args, 3, 20),
        ),
        "q4k-micro" => llm170_backend_gpu::rawhip::q4k_micro(),
        "q4-d2h-bench" => llm170_backend_gpu::rawhip::d2h_bench(),
        "q5-1-bench" => llm170_backend_gpu::rawhip::q5_1_bench(
            arg_num(args, 0, 20),
            arg_num(args, 1, 640),
            arg_num(args, 2, 2560),
            arg_num(args, 3, 50),
        ),
        "q4-qsa-check" => llm170_backend_gpu::rawhip::q4acc::qsa_check(
            arg_num(args, 0, 200usize),
            arg_num(args, 1, 200usize),
        ),
        "q4-hc-check" => llm170_backend_gpu::rawhip::q4acc::hc_check(
            arg_num(args, 0, 230usize),
            arg_num(args, 1, 2560usize),
            arg_num(args, 2, 4usize),
        ),
        "q4-ple-check" => llm170_backend_gpu::rawhip::q4acc::ple_gate_check(),
        "q4-ar-check" => llm170_backend_gpu::rawhip::q4acc::ar_check_t(arg_num(args, 0, 1usize)),
        "q4-acc-check" => {
            let path = arg_str(args, 0, d_fn);
            if args.first().map(String::as_str) == Some("micro") {
                return Some(match llm170_backend_gpu::rawhip::q4acc::micro_check() {
                    Ok(s) => {
                        println!("{s}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::FAILURE
                    }
                });
            }
            let tn = arg_str(args, 1, "blk.0.ffn_gate_exps.weight");
            let t = arg_num(args, 2, 2usize);
            let rows = arg_num(args, 3, 256usize);
            llm170_backend_gpu::rawhip::q4acc::check_tensor(
                std::path::Path::new(&path),
                &tn,
                t,
                rows,
            )
        }
        "moe-row-check" => {
            let path = arg_str(args, 0, d_fn);
            let tn = arg_str(args, 1, "blk.0.ffn_gate_exps.weight");
            let t_a = arg_num(args, 2, 16usize);
            let t_b = arg_num(args, 3, 64usize);
            llm170_backend_gpu::rawhip::q4acc::moe_row_check(
                std::path::Path::new(&path),
                &tn,
                t_a,
                t_b,
            )
        }
        "mm-row-check" => {
            let path = arg_str(args, 0, d_fn);
            let tn = arg_str(args, 1, "blk.0.attn_qkv.weight");
            let t_a = arg_num(args, 2, 16usize);
            let t_b = arg_num(args, 3, 64usize);
            llm170_backend_gpu::rawhip::q4acc::mm_row_check(
                std::path::Path::new(&path),
                &tn,
                t_a,
                t_b,
            )
        }
        "exl3-check" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            let d_q8 = "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf";
            cmd_exl3_check(&arg_str(args, 0, d_exl3), &arg_str(args, 1, d_q8))
        }
        "diag" => {
            // plans/82: 지문 비교 — `llm170 diag diff <A> <B>`
            if args.first().map(String::as_str) == Some("diff") {
                let (pa, pb) = match (args.get(1), args.get(2)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => {
                        eprintln!("사용법: llm170 diag diff <A> <B>");
                        return Some(ExitCode::FAILURE);
                    }
                };
                match llm170_diag::fp_diff(pa, pb) {
                    Ok(r) if r.mismatch_count == 0 => Ok(format!("{r}")),
                    // QA-20: 발산은 실패로 — 종전 Ok→exit 0이 스크립트 체인에서
                    // 수치 발산을 성공으로 오판하게 했다(실패-성공 전환).
                    Ok(r) => Err(format!("{r}\nDIVERGENCE DETECTED")),
                    Err(e) => Err(e),
                }
            }
            // plans/83 C3: 청크 불변성 자동 검증 — `llm170 diag chunk-check <model> <prompt> [sizes...]`
            else if args.first().map(String::as_str) == Some("chunk-check") {
                return Some(cmd_chunk_check(&args[1..]));
            }
            // plans/87 §1 — tsv 원장에서 폴트 주소 매칭.
            else if args.first().map(String::as_str) == Some("va-lookup") {
                let Some(tsv) = args.get(1).cloned() else {
                    eprintln!("error: va-lookup <tsv> <addr-hex>");
                    return Some(ExitCode::FAILURE);
                };
                let Some(addr) = args.get(2).cloned() else {
                    eprintln!("error: va-lookup <tsv> <addr-hex>");
                    return Some(ExitCode::FAILURE);
                };
                return Some(cmd_va_lookup(&tsv, &addr));
            }
            // plans/87 §1 — 의도적 디스크립터-오프셋 OOB 폴트 유발.
            else if args.first().map(String::as_str) == Some("vk-fault-probe") {
                return Some(cmd_vk_fault_probe());
            }
            // plans/87 §2 — 와치독 자가 시험(진동 정지 후 스폰).
            else if args.first().map(String::as_str) == Some("watchdog-selftest") {
                return Some(cmd_watchdog_selftest());
            }
            // plans/107 W8 — 폴백 카운터 관측.
            else if args.first().map(String::as_str) == Some("fb") {
                let r = llm170_core::qwen4exp::frame::fb_report();
                if r.is_empty() {
                    Ok("폴백 0건 (전 경로 GPU)".into())
                } else {
                    Ok(r)
                }
            }
            // plans/108 P1 — env 스냅샷↔라이브 동치 검사.
            else if args.first().map(String::as_str) == Some("envcheck") {
                let bad = llm170_diag::flag::env_check();
                if bad.is_empty() {
                    Ok("env 동치 정상 (LLM170_ 키 전수)".into())
                } else {
                    Err(format!(
                        "env 동치 불일치 {}건:\n{}",
                        bad.len(),
                        bad.join("\n")
                    ))
                }
            }
            // plans/87 §5 — [npck] 로그 크로스 diff.
            else if args.first().map(String::as_str) == Some("ckdiff") {
                let Some(a) = args.get(1).cloned() else {
                    eprintln!("error: ckdiff <A.log> <B.log> [rel]");
                    return Some(ExitCode::FAILURE);
                };
                let Some(b) = args.get(2).cloned() else {
                    eprintln!("error: ckdiff <A.log> <B.log> [rel]");
                    return Some(ExitCode::FAILURE);
                };
                let rel = args
                    .get(3)
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(0.02);
                return Some(cmd_ckdiff(&a, &b, rel));
            } else {
                Err("diag: 하위커맨드 diff | chunk-check | va-lookup | vk-fault-probe | watchdog-selftest | ckdiff | fb | envcheck".into())
            }
        }
        "mmq-row-check" => {
            let path = arg_str(args, 0, d_27);
            let tn = arg_str(args, 1, "blk.0.attn_gate.weight");
            let t1 = arg_num(args, 2, 16usize);
            let t2 = arg_num(args, 3, 208usize);
            llm170_backend_gpu::rawhip::mmq_row_check(&path, &tn, t1, t2)
        }
        "hip-dmmv-check" => {
            let path = arg_str(args, 0, d_27);
            let tn = arg_str(args, 1, "blk.0.ssm_out.weight");
            llm170_backend_gpu::rawhip::hip_dmmv_check(&path, &tn)
        }
        "hip-moe-dmmv-check" => {
            let path = arg_str(args, 0, d_fn);
            let tn = arg_str(args, 1, "blk.0.ffn_gate_exps.weight");
            llm170_backend_gpu::rawhip::hip_moe_dmmv_check(&path, &tn)
        }
        "tile-row-check" => {
            let path = arg_str(args, 0, d_27);
            let tn = arg_str(args, 1, "blk.0.ssm_out.weight");
            let t1 = arg_num(args, 2, 16usize);
            let t2 = arg_num(args, 3, 208usize);
            llm170_backend_gpu::rawhip::tile_row_check(&path, &tn, t1, t2)
        }
        "mtp-load-check" => {
            // plans/109 P15① 검증 — 외장 MTP 모듈 파트 병합·텐서 뷰 동작 확인.
            let main_p = arg_str(args, 0, d_fn);
            let mtp_p = arg_str(
                args,
                1,
                "/home/yoon/models/qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
            );
            (|| -> Result<String, String> {
                let mut m = llm170_core::qwen4exp::Model4::load(std::path::Path::new(&main_p))
                    .map_err(|e| e.to_string())?;
                m.load_mtp(std::path::Path::new(&mtp_p))
                    .map_err(|e| e.to_string())?;
                let mut out = format!(
                    "mtp-load-check: has_mtp={} n_layer={} 병합 텐서:\n",
                    m.has_mtp(),
                    m.hp.n_layer
                );
                for name in [
                    "blk.48.attn_q.weight",
                    "blk.48.nextn.eh_proj.weight",
                    "blk.48.nextn.enorm.weight",
                    "blk.48.nextn.hc_head_up.weight",
                    "token_embd.weight",
                ] {
                    let w = m.w(name).ok_or_else(|| format!("텐서 없음: {name}"))?;
                    out.push_str(&format!(
                        "  {name}: ty={} n_in={} n_out={} bytes={}\n",
                        w.ty.name(),
                        w.n_in,
                        w.n_out,
                        w.data.len()
                    ));
                }
                let enorm = m
                    .f32_vec4("blk.48.nextn.enorm.weight")
                    .map_err(|e| e.to_string())?;
                out.push_str(&format!("  enorm f32_vec4: {}원소\n", enorm.len()));
                out.push_str(&format!(
                    "  compress[48]={:?} (len={}) is_recr={}\n",
                    m.hp.compress.last(),
                    m.hp.compress.len(),
                    m.hp.is_recr(48)
                ));
                Ok(out)
            })()
        }
        "q6k-ref" => {
            let path = arg_str(args, 0, d_q35);
            let tn = arg_str(args, 1, "blk.64.nextn.eh_proj.weight");
            llm170_backend_gpu::rawhip::q6k_ref_probe(&path, &tn)
        }

        "mtp-draft-check" => {
            // plans/109 P15② — CPU 참조 드래프트 스텝 스모크: 프리필 → h →
            // mtp_draft_step → 로짓 유한성·top-5 토큰.
            let main_p = arg_str(args, 0, d_fn);
            let mtp_p = arg_str(
                args,
                1,
                "/home/yoon/models/qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
            );
            (|| -> Result<String, String> {
                let mut m = llm170_core::qwen4exp::Model4::load(std::path::Path::new(&main_p))
                    .map_err(|e| e.to_string())?;
                m.load_mtp(std::path::Path::new(&mtp_p))
                    .map_err(|e| e.to_string())?;
                let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, 1, 512);
                let p: Vec<u32> = [386, 18, 15, 15, 643, 20, 20].to_vec();
                let l = eng.prefill(0, &p).map_err(|e| e.to_string())?;
                let t0 = llm170_core::qwen35::greedy(&l);
                let h = eng.last_h.clone();
                let lg = eng.mtp_draft_step(0, t0, &h).map_err(|e| e.to_string())?;
                let finite = lg.iter().all(|v| v.is_finite());
                // QA-22: 로짓 NaN/Inf는 실패로 — 종전엔 finite=false를 출력에만
                // 실어 보내고 exit 0이었다(이 프로브가 잡으려는 결함 클래스).
                if !finite {
                    return Err(format!(
                        "mtp-draft-check: 로짓 비유한(NaN/Inf {}개)",
                        lg.iter().filter(|v| !v.is_finite()).count()
                    ));
                }
                let mut idx: Vec<usize> = (0..lg.len()).collect();
                idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]));
                let top: Vec<String> = idx[..5]
                    .iter()
                    .map(|&i| format!("{}:{:.2}", i, lg[i]))
                    .collect();
                // P15③ 스모크 — k=3 스펙 3스텝 수용률.
                let mut acc_total = 0usize;
                let mut fwd_total = 0usize;
                let mut last = t0;
                for _ in 0..3 {
                    let (acc, fwd) = eng.mtp_spec_step(0, last, 3).map_err(|e| e.to_string())?;
                    acc_total += acc.len();
                    fwd_total += fwd;
                    last = *acc.last().unwrap_or(&last);
                }
                Ok(format!(
                    "mtp-draft-check: 로짓 {}개 finite={} top5=[{}] (draft pos={}) | spec k=3×3: 수용 {}토큰/{} forward = {:.2} tok/fwd",
                    lg.len(),
                    finite,
                    top.join(" "),
                    eng.mtp_seqs[0].pos,
                    acc_total,
                    fwd_total,
                    acc_total as f64 / fwd_total.max(1) as f64
                ))
            })()
        }
        "gdn-check" => llm170_backend_gpu::rawvk::gdn_check(),
        "vk-check" => llm170_backend_gpu::rawvk::smoke_test(),
        // 109 P15-1a(7e06e90)에서 우발 삭제된 진입점 복원(110 P12c 검증용).
        "vk-frame-check" => {
            let path = arg_str(args, 0, d_fn);
            let tn = arg_str(args, 1, "blk.0.ffn_down_shexp.weight");
            llm170_backend_gpu::rawvk::checks::frame_check(&path, &tn)
        }
        "exl3-bench" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_bench(
                &arg_str(args, 0, d_exl3),
                arg_num(args, 1, 5usize),
            )
        }
        "exl3-vk-check" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_vk_check(
                &arg_str(args, 0, d_exl3),
                &arg_str(args, 1, "model.language_model.layers.0.mlp.gate_proj"),
            )
        }
        "gqa-bench" => llm170_backend_gpu::rawhip::gqa_bench(),
        "mm-tile" => llm170_backend_gpu::rawhip::mm_tile_bench(),
        "mm-bench" => llm170_backend_gpu::rawhip::mm_batch_bench(),
        "bw-test" => llm170_backend_gpu::rawhip::bw_test(),
        "dp4a-test" => llm170_backend_gpu::rawhip::dp4a_test(),
        "iq3s-probe" => llm170_backend_gpu::rawhip::iq3s_probe(),
        "qk-check" => llm170_backend_gpu::rawhip::qk_check(),
        _ => return special(cmd, args),
    };
    Some(match r {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    })
}

/// 프로브 중 출력 형태가 특수한 것들.
fn special(cmd: &str, args: &[String]) -> Option<ExitCode> {
    match cmd {
        "dims" => {
            let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            print!("{}", llm170_backend_gpu::rawhip::dims_of(a[0], &a[1..]));
            Some(ExitCode::SUCCESS)
        }
        "tty-probe" => {
            let path = args
                .first()
                .cloned()
                .unwrap_or_else(|| "/tmp/model_link.gguf".into());
            match llm170_gguf::GgufFile::open(std::path::Path::new(&path)) {
                Ok(g) => {
                    use std::collections::BTreeMap;
                    let mut cnt: BTreeMap<u32, usize> = BTreeMap::new();
                    let mut bytes: BTreeMap<u32, u64> = BTreeMap::new();
                    for t in &g.tensors {
                        *cnt.entry(t.ty as u32).or_insert(0) += 1;
                        *bytes.entry(t.ty as u32).or_insert(0) += t.nbytes().unwrap_or(0);
                    }
                    for (k, c) in cnt {
                        println!("ty{k}: {c} tensors {:.1}MB", bytes[&k] as f64 / 1e6);
                    }
                    Some(ExitCode::SUCCESS)
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    Some(ExitCode::FAILURE)
                }
            }
        }

        "rawhip-check" => Some(cmd_rawhip_check(args)),
        _ => None,
    }
}

/// `llm170 diag chunk-check <model> <prompt> [sizes...] [--backend cpu]`
/// 청크 불변성 자동 검증 (plans/83 C3, docs/chunk-invariance.md 계약).
///
/// 프롬프트를 단일 호출(기준)과 각 청크 크기로 프리필해 최종 logits를 비교:
/// - bits 동일 → PASS
/// - argmax 동일 && max|Δ| < 1e-3 → PASS(near-tie, GPU 행 수 의존 잔여 축)
/// - 그 외 → FAIL
///
/// prompt: "1,2,3" 형태면 토큰 id, 아니면 텍스트(BPE 인코딩 — plans/83 A).
fn cmd_chunk_check(args: &[String]) -> ExitCode {
    let usage = "사용법: llm170 diag chunk-check <model> <prompt> [sizes...] [--backend cpu]";
    let Some(model) = args.first() else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    let Some(prompt) = args.get(1) else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    let backend_cpu = args
        .iter()
        .any(|a| a == "--backend" && args.iter().any(|b| b == "cpu"))
        || args.iter().any(|a| a == "--backend=cpu");
    let sizes: Vec<usize> = args[2..]
        .iter()
        .filter_map(|a| a.parse::<usize>().ok())
        .filter(|&s| s > 0)
        .collect();
    let sizes = if sizes.is_empty() {
        vec![16, 63, 64, 512]
    } else {
        sizes
    };

    // 프롬프트 파싱 — 숫자/콤마 전용이면 토큰 id, 아니면 텍스트
    let ids: Vec<u32> = if prompt
        .bytes()
        .all(|b| b.is_ascii_digit() || b == b',' || b == b' ')
        && prompt.contains(',')
    {
        prompt
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect()
    } else {
        let p = std::path::PathBuf::from(model);
        let stem = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let part2 = if stem.contains("-00001-of-") {
            Some(p.with_file_name(stem.replace("-00001-of-", "-00002-of-")))
        } else {
            None
        };
        match crate::tokenize::Tokenizer::load(&p, part2.as_deref()) {
            Ok(t) => t.encode(prompt),
            Err(e) => {
                eprintln!("error: tokenizer: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    if ids.is_empty() {
        eprintln!("error: 빈 프롬프트");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "# chunk-check: {}토큰, sizes={:?}, backend={}",
        ids.len(),
        sizes,
        if backend_cpu { "cpu" } else { "gpu" }
    );

    let path = std::path::PathBuf::from(model);
    let arch = llm170_gguf::GgufFile::open(&path)
        .ok()
        .and_then(|g| g.arch().map(|s| s.to_string()));
    let ctx = ids.len() * 2 + 64;
    let (ref_l, runs): (Vec<f32>, Vec<(usize, Vec<f32>)>) = match arch.as_deref() {
        Some("qwen4exp") => {
            let res = llm170_core::qwen4exp::Model4::load(&path)
                .map_err(|e| e.to_string())
                .and_then(|m| {
                    let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, 1, ctx);
                    if !backend_cpu && !crate::engine::q4_gpu_env_off() {
                        let sources = eng.model.part_sources();
                        // plans/88 — 런타임 존중: LLM170_GPU_RUNTIME=vulkan 이면 vk
                        // 프레임 경로의 청크 펜스를 잴 수 있다(종전 hip 고정이라
                        // vk 변경의 펜스 검증이 불가했다).
                        let vk = std::env::var("LLM170_GPU_RUNTIME")
                            .map(|v| v == "vulkan")
                            .unwrap_or(false);
                        let r = if vk {
                            llm170_backend_gpu::new_q4_acc_vk_with_sources(sources)
                        } else {
                            llm170_backend_gpu::new_q4_acc_with_sources(sources)
                        };
                        match r {
                            Ok(acc) => {
                                eng = eng.with_acc(acc);
                            }
                            Err(e) => {
                                return Err(format!(
                                    "GPU 가속기 생성 실패 — {e} (--backend cpu 로 회피)"
                                ));
                            }
                        }
                    }
                    // 기준: 단일 청크(프롬프트 전체)
                    unsafe {
                        std::env::set_var("LLM170_Q4_CHUNK", format!("{}", ids.len().max(1)))
                    };
                    let r = eng.prefill(0, &ids).map_err(|e| e.to_string())?;
                    eng.reset_seq(0);
                    let mut runs = Vec::new();
                    for &sz in &sizes {
                        unsafe { std::env::set_var("LLM170_Q4_CHUNK", format!("{sz}")) };
                        let l = eng.prefill(0, &ids).map_err(|e| e.to_string())?;
                        eng.reset_states();
                        runs.push((sz, l));
                    }
                    Ok((r, runs))
                });
            match res {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        _ => {
            // qwen35 (및 기본) — 호출부 청킹
            let res = llm170_core::qwen35::Model::load(&path)
                .map_err(|e| e.to_string())
                .and_then(|m| {
                    let cc_seq: usize = std::env::var("LLM170_CC_SEQ")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let mut eng = llm170_core::qwen35::Engine::new(m, (cc_seq + 1).max(1), ctx);
                    if !backend_cpu
                        && std::env::var("LLM170_RAWHIP")
                            .map(|v| v != "0")
                            .unwrap_or(true)
                    {
                        let _ = llm170_backend_gpu::inject_rawhip(&mut eng);
                    }
                    let r = eng.prefill(0, &ids).map_err(|e| e.to_string())?;
                    eng.reset_seq(cc_seq);
                    let mut runs = Vec::new();
                    for &sz in &sizes {
                        let mut last = None;
                        for ch in ids.chunks(sz) {
                            last = Some(eng.prefill(cc_seq, ch).map_err(|e| e.to_string())?);
                        }
                        eng.reset_states();
                        runs.push((sz, last.expect("청크 1개 이상")));
                    }
                    Ok((r, runs))
                });
            match res {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };

    let ref_tok = llm170_core::qwen35::greedy(&ref_l);
    println!("reference: {} logits, argmax={ref_tok}", ref_l.len());
    let mut all_pass = true;
    for (sz, l) in &runs {
        // QA-19: NaN 불감 수리 — fold(0.0, f32::max)는 NaN을 무시(반대편
        // 반환)해 diff 전체가 NaN이어도 maxd=0 → "PASS bits-identical".
        let bad_nan = l.iter().any(|v| v.is_nan()) || ref_l.iter().any(|v| v.is_nan());
        let maxd = l
            .iter()
            .zip(ref_l.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let tok = llm170_core::qwen35::greedy(l);
        let (verdict, why) = if bad_nan {
            all_pass = false;
            ("FAIL", "NaN in logits".to_string())
        } else if l.len() != ref_l.len() {
            // QA-19: 길이 불일치 — 종전엔 교집합 zip만 비교해 짧은 logits가 통과.
            all_pass = false;
            ("FAIL", format!("len {} != ref {}", l.len(), ref_l.len()))
        } else if maxd == 0.0 {
            ("PASS", "bits-identical".to_string())
        } else if tok == ref_tok && maxd < 1e-3 {
            (
                "PASS",
                format!(
                    "near-tie max|Δ|={maxd:.3e} (행 수 의존 잔여축 — plans/archive/chunk-invariance.md)"
                ),
            )
        } else {
            all_pass = false;
            ("FAIL", format!("max|Δ|={maxd:.3e} argmax {tok}≠{ref_tok}"))
        };
        println!("  chunk {sz:5}: {verdict} — {why}");
    }
    if all_pass {
        ExitCode::SUCCESS
    } else {
        println!("chunk-check: FAIL — 청크 불변성 위반 (plans/archive/chunk-invariance.md)");
        ExitCode::FAILURE
    }
}

/// llm170 rawhip-check <file> <tensor> — 원시 HIP GEMV(quant·gemm·reduce)
/// 대 CPU 레인 미러 to_bits 전행 검증 + 속도.
fn cmd_rawhip_check(args: &[String]) -> ExitCode {
    use llm170_backend_gpu::rawhip::RawCtx;
    if args.len() < 2 {
        eprintln!("usage: llm170 rawhip-check <file> <tensor>");
        return ExitCode::from(2);
    }
    let model = match llm170_core::qwen35::Model::load(std::path::Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => {
            eprintln!("tensor not found: {}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let raw_ok = llm170_core::matmul::w4a8_ty(w.ty) || w.ty == llm170_gguf::GgmlType::Iq3S;
    if !raw_ok {
        eprintln!("rawhip-check: 미지원 타입");
        return ExitCode::FAILURE;
    }
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let ctx = match RawCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let y = llm170_core::quant::quantize_row_q8_ref(&x);
    // GPU 양자화 비트 미러 검증 (quant_q8 커널)
    let mut xq_gpu: Option<*mut u8> = None;
    {
        let mut inner = || -> Result<(), String> {
            let xd_buf = ctx.alloc(n_in * 4)?;
            let xq_buf = ctx.alloc((n_in / 4 + n_in / 32) * 4)?; // 워드 + d 비트
            xq_gpu = Some(xq_buf);
            ctx.h2d(xd_buf, bytemuck::cast_slice(&x))?;
            ctx.quant_q8(xd_buf as *const u8, xq_buf, n_in)?;
            let mut gq = vec![0u8; (n_in / 4 + n_in / 32) * 4];
            ctx.d2h(&mut gq, xq_buf)?;
            let gw: Vec<u32> = bytemuck::cast_slice(&gq[..n_in / 4 * 4]).to_vec();
            let mut qm = 0usize;
            let cpu_w: Vec<u32> = {
                let mut v = Vec::new();
                for c in y
                    .iter()
                    .flat_map(|b| b.qs.iter())
                    .collect::<Vec<_>>()
                    .chunks(4)
                {
                    let mut word = 0u32;
                    for (i, b) in c.iter().enumerate() {
                        word |= (**b as u8 as u32) << (8 * i);
                    }
                    v.push(word);
                }
                v
            };
            for (i, (a, b)) in gw.iter().zip(cpu_w.iter()).enumerate() {
                if a != b {
                    qm += 1;
                    if qm == 1 {
                        println!("  ✗ quant 워드[{i}] gpu={a:#x} cpu={b:#x}");
                    }
                }
            }
            let gdbits: Vec<u32> = bytemuck::cast_slice(&gq[n_in / 4 * 4..]).to_vec();
            for (i, (a, b)) in gdbits
                .iter()
                .zip(y.iter().map(|b| b.d.to_bits()))
                .enumerate()
            {
                if *a != b {
                    qm += 1;
                    if qm <= 3 {
                        println!("  ✗ quant d[{i}] gpu_bits={a:#x} cpu_bits={b:#x}");
                    }
                }
            }
            if qm > 0 {
                // QA-21: 불일치는 실패로 — 종전엔 ✗ 출력 후 Ok로 넘어가 최종
                // 통과 요약이 나갔다.
                return Err(format!("quant_q8 미러 불일치 {qm}워드/비트"));
            }
            println!("  ★ quant_q8 원시 ≡ CPU 비트 일치");
            Ok(())
        };
        // QA-21: quant 검증 실패(에러·불일치)는 GEMV 검증 없이 실패 종료 —
        // 종전엔 실패를 삼키고 CPU 패킹 폴백으로 GEMV만 검증한 뒤 통과 보고.
        if let Err(e) = inner() {
            eprintln!("quant 검증 실패: {e}");
            return ExitCode::FAILURE;
        }
    }
    let mut qs_words = Vec::with_capacity(n_in / 4);
    for c in y
        .iter()
        .flat_map(|b| b.qs.iter())
        .collect::<Vec<_>>()
        .chunks(4)
    {
        let mut word = 0u32;
        for (i, b) in c.iter().enumerate() {
            word |= (**b as u8 as u32) << (8 * i);
        }
        qs_words.push(word);
    }
    // ktab2
    let ktab2: Vec<u32> = llm170_core::ktab2_packed();
    // GPU quant 사용 시: xq 버퍼 = 워드+d 통합 (gemv가 직접 판독)
    let xq_d = match xq_gpu {
        Some(p) => p,
        None => {
            // CPU 경로: 워드 + d 비트 통합 패킹
            let buf = ctx.alloc((n_in / 4 + n_in / 32) * 4).expect("alloc");
            let mut packed = qs_words.clone();
            packed.extend(y.iter().map(|b| b.d.to_bits()));
            ctx.h2d(buf, bytemuck::cast_slice(&packed))
                .expect("pack upload");
            buf
        }
    };
    let w_d = match ctx.alloc(w.data.len()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let kt_d = match ctx.alloc(1024) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    // GPU quant 출력 재사용 시 xq/xd 업로드 생략 (종단 검증 — d가 GPU 생산값)
    let up = ctx
        .h2d(w_d, w.data)
        .and_then(|_| ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2)));
    if let Err(e) = up {
        eprintln!("upload: {e}");
        return ExitCode::FAILURE;
    }
    // 워밍 + 측정
    let ty = w.ty as u32;
    let _ = match ctx.gemv_q8(
        xq_d as *const u8,
        w_d as *const u8,
        kt_d as *const u8,
        ty,
        n_in,
        n_out,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gemv: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reps = 30;
    let t0 = std::time::Instant::now();
    let mut g = Vec::new();
    for _ in 0..reps {
        g = match ctx.gemv_q8(
            xq_d as *const u8,
            w_d as *const u8,
            kt_d as *const u8,
            ty,
            n_in,
            n_out,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("gemv: {e}");
                return ExitCode::FAILURE;
            }
        };
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    // to_bits 전행 비교
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut mism = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for o in 0..n_out {
        let row = &w.data[o * rb..];
        let c = match w.ty {
            llm170_gguf::GgmlType::Q5K => {
                llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Q4K => {
                llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Q8_0 => {
                llm170_core::quant::dot_row_w4a8_q8_0_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Q6K => {
                llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Iq4Nl => {
                llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Q3K => {
                llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Iq3S => {
                llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Q5_1 => {
                llm170_core::quant::dot_row_w4a8_q5_1_lane(row, n_in as u64, &y)
            }
            llm170_gguf::GgmlType::Iq4Xs => {
                llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &y)
            }
            other => {
                eprintln!("gemv 미지원 타입 {other:?} — 미러 오계산 방지");
                return ExitCode::FAILURE;
            }
        };
        if c.to_bits() != g[o].to_bits() {
            mism += 1;
            if first.is_none() {
                first = Some((o, c, g[o]));
            }
        }
    }
    println!(
        "[{}] {}: 원시 GEMV 불일치 {mism}/{n_out} — {:.0}µs/op {:.0}GB/s",
        w.ty.name(),
        args[1],
        dt * 1e6,
        w.data.len() as f64 / dt / 1e9
    );
    if let Some((o, c, gv)) = first {
        println!("  첫 불일치 [{o}]: cpu={c:.7e} gpu={gv:.7e}");
    }
    if mism > 0 {
        ExitCode::FAILURE
    } else {
        println!("  ★ 원시 HIP ≡ CPU 비트 일치");
        ExitCode::SUCCESS
    }
}

/// llm170 check <model.gguf> [--quick] [--backend cpu|gpu]
/// debug 빌드 검증 경로 — ① 텐서 디양자화 스캔(NaN/Inf) ② GPU↔CPU GEMM
/// 상호검증 ③ 장문 청크 스모크(NaN 가드). RCA 도구 통합 (2026-09-01).
pub fn run_check(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut quick = false;
    let mut backend = "gpu".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--backend" => backend = it.next().cloned().unwrap_or_else(|| "gpu".into()),
            p if !p.starts_with("--") => path = Some(p),
            _ => {}
        }
    }
    let Some(path) = path else {
        eprintln!("usage: llm170 check <model.gguf> [--quick] [--backend cpu|gpu]");
        return ExitCode::from(2);
    };
    let model_path = std::path::PathBuf::from(path);
    eprintln!("# check: {path} backend={backend} quick={quick}");

    // ① 텐서 스캔 — 각 텐서 첫 행 디양자화해 NaN/Inf 검출
    let scan = std::thread::spawn({
        let p = model_path.clone();
        move || -> Result<(usize, usize), String> {
            let g = llm170_gguf::GgufFile::open(&p).map_err(|e| e.to_string())?;
            let file = std::fs::File::open(&p).map_err(|e| e.to_string())?;
            // SAFETY: 읽기 전용 매핑
            let mmap =
                unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
            let mut bad = 0usize;
            let mut n = 0usize;
            for t in g.tensors.iter().take(if quick { 64 } else { usize::MAX }) {
                let (start, end) = match t.file_range(g.data_offset) {
                    Some(r) => r,
                    None => continue,
                };
                let data = &mmap[start as usize..end as usize];
                let n_in = t.ne[0] as usize;
                let mut row = vec![0.0f32; n_in.min(4096)];
                llm170_core::quant::dequant_row(t.ty, data, 0, row.len() as u64, &mut row);
                n += 1;
                if row.iter().any(|v| !v.is_finite()) {
                    eprintln!("# 텐서 비정상: {} ({})", t.name, t.ty.name());
                    bad += 1;
                }
            }
            Ok((n, bad))
        }
    });
    match scan.join() {
        Ok(Ok((n, bad))) => {
            eprintln!("# ① 텐서 스캔: {n}개 중 비정상 {bad}");
            if bad > 0 {
                return ExitCode::FAILURE;
            }
        }
        Ok(Err(e)) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        Err(_) => return ExitCode::FAILURE,
    }

    // ② GPU↔CPU GEMM 상호검증 (gpu 경로만) — 대표 텐서 t∈{1,64,1024}
    // (② GPU↔CPU GEMM 검증 — cubecl 제거로 rawhip-check가 대체)

    // ③ 장문 청크 스모크 — 1,024토큰 무작위 prefill (NaN 가드는 dump 키 q4_trace)
    let arch = llm170_gguf::GgufFile::open(&model_path)
        .ok()
        .and_then(|g| g.arch().map(str::to_string));
    if arch.as_deref() == Some("qwen4exp") {
        let toks: Vec<String> = (0..1024)
            .map(|i| (100 + (i * 7919) % 200000).to_string())
            .collect();
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap_or_default());
        cmd.args([
            "infer",
            "--model",
            path,
            "--prompt-tokens",
            &toks.join(","),
            "--n-predict",
            "2",
            "--ctx",
            "2048",
            "--backend",
            &backend,
        ])
        .env("LLM170_DUMP", "q4_trace")
        .stdout(std::process::Stdio::null());
        let st = cmd.status();
        match st {
            Ok(s) if s.success() => eprintln!("# ③ 청크 스모크(1024토큰): 통과"),
            Ok(s) => {
                eprintln!("# ③ 청크 스모크: 실패 ({s})");
                return ExitCode::FAILURE;
            }
            Err(e) => {
                eprintln!("# ③ 청크 스모크 실행 실패: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        // QA-22: q35 모델은 ③이 미수행 — "전체 통과"로 속이지 않는다.
        eprintln!("# ③ 청크 스모크: qwen4exp 전용 — 이 아키텍처는 스킵");
    }
    eprintln!("# check 통과 (① 텐서 스캔 수행, ② 는 rawhip-check 별도, ③ 은 qwen4exp 한정)");
    ExitCode::SUCCESS
}

/// plans/87 §1 — tsv 원장(site, bytes, va, va_end, seq)에서 폴트 주소 매칭.
fn cmd_va_lookup(tsv: &str, addr: &str) -> ExitCode {
    // BDA는 canonical 부호확장(0xffff8001..), RADV 폴트는 48비트 절단형
    // (0x8001..) — 하위 48비트로 정규화해 비교한다(실측, plans/87 §1).
    const M: u64 = 0x0000_ffff_ffff_ffff;
    let Ok(a) = u64::from_str_radix(addr.trim_start_matches("0x"), 16) else {
        eprintln!("error: 주소 파싱 실패: {addr}");
        return ExitCode::FAILURE;
    };
    let a = a & M;
    let Ok(text) = std::fs::read_to_string(tsv) else {
        eprintln!("error: tsv 없음: {tsv}");
        return ExitCode::FAILURE;
    };
    let mut hits = 0usize;
    for (ln, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 4 {
            continue;
        }
        let (Ok(lo), Ok(hi)) = (
            u64::from_str_radix(f[2].trim_start_matches("0x"), 16),
            u64::from_str_radix(f[3].trim_start_matches("0x"), 16),
        ) else {
            continue;
        };
        let (lo, hi) = (lo & M, hi & M);
        if a >= lo && a < hi {
            println!(
                "HIT line {} site={} bytes={} range={:#x}..{:#x} seq={}",
                ln + 1,
                f[0],
                f[1],
                lo,
                hi,
                f.get(4).unwrap_or(&"?")
            );
            hits += 1;
        }
    }
    if hits == 0 {
        // OOB 접근은 대상 버퍼 **끝 너머**에 착지한다(원 사건: 921600B
        // 1-전문가 버퍼 +929792) — ±4MB 근접 항목을 인접 보고한다.
        const WINDOW: i64 = 4 << 20;
        let mut cands: Vec<(i64, i64, &str)> = Vec::new(); // (d, lo, line)
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 {
                continue;
            }
            let Ok(lo) = u64::from_str_radix(f[2].trim_start_matches("0x"), 16) else {
                continue;
            };
            let lo = lo & M;
            let d = a as i64 - lo as i64;
            if d.abs() <= WINDOW {
                cands.push((d, lo as i64, line));
            }
        }
        // OOB는 버퍼 베이스에서 앞으로 걷는다 — 양수(이후) 방향 우선, 그 후 근접.
        cands.sort_by_key(|(d, _, _)| (*d < 0, d.abs()));
        let best = cands.first().map(|(d, _, line)| (*d, *line));
        match best {
            Some((d, line)) => {
                let f: Vec<&str> = line.split('\t').collect();
                println!(
                    "NEAR addr={a:#x} — {} {}B 범위 {:#x}..{:#x} 에서 {d:+} 바이트 (OOB 착지로 추정)",
                    f[0],
                    f[1],
                    u64::from_str_radix(f[2].trim_start_matches("0x"), 16).unwrap_or(0) & M,
                    u64::from_str_radix(f[3].trim_start_matches("0x"), 16).unwrap_or(0) & M
                );
                ExitCode::SUCCESS
            }
            None => {
                println!("NO-HIT addr={a:#x} ({} entries)", text.lines().count());
                ExitCode::FAILURE
            }
        }
    } else {
        ExitCode::SUCCESS
    }
}

/// plans/87 §1 — 의도적 GPUVM 폴트(디스크립터 오프셋 OOB) 유발.
fn cmd_vk_fault_probe() -> ExitCode {
    match llm170_backend_gpu::rawvk::checks::fault_probe() {
        Ok(msg) => {
            println!("# fault-probe: {msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// plans/87 §2 — 진동을 멈추고 와치독 보고를 기다린다(FAIL 모드면 137).
fn cmd_watchdog_selftest() -> ExitCode {
    // QA-22: 자가시험은 실패 가능해야 한다 — 종전엔 와치독 미기동·미보고
    // 어느 쪽이든 SUCCESS 고정(자가시험 실패 불가 구조)이었다.
    if !llm170_diag::watchdog::on() {
        eprintln!("# watchdog-selftest: 와치독 미기동(LLM170_WATCHDOG 미설정?) — 시험 불가");
        return ExitCode::FAILURE;
    }
    let sec: u64 = std::env::var("LLM170_WATCHDOG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    println!("# watchdog-selftest: {sec}s 무진동 후 보고 대기");
    llm170_diag::watchdog::record_op("selftest_stall");
    // 진행 없이 대기 — 와치독이 보고한다. FAIL 모드면 여기서 종료(137).
    let before = llm170_diag::watchdog::reports();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(sec + 30);
    while std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    let n = llm170_diag::watchdog::reports() - before;
    if n == 0 {
        eprintln!("# watchdog-selftest: 보고 0건 — 와치독 스레드 미작동 의심 (FAIL)");
        return ExitCode::FAILURE;
    }
    println!("# watchdog-selftest: 와치독 보고 {n}건 확인");
    ExitCode::SUCCESS
}

/// plans/87 §5 — [npck] 체크섬 로그 교차 diff: (tag, t, 등장순) 정렬 맞춤,
/// 첫 상이 태그 + 상위 상이 표. 무상이 exit 0.
fn cmd_ckdiff(a_path: &str, b_path: &str, rel_lim: f64) -> ExitCode {
    let parse = |p: &str| -> Result<Vec<(String, usize, f64)>, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        let mut out = Vec::new();
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("[npck] ") else {
                continue;
            };
            // {tag} t={t} sum={s} v0=.. mid0=.. last0=..
            let mut it = rest.split_whitespace();
            let Some(tag) = it.next() else { continue };
            let Some(t) = it
                .next()
                .and_then(|s| s.strip_prefix("t="))
                .and_then(|v| v.parse().ok())
            else {
                continue;
            };
            let Some(sum) = it
                .next()
                .and_then(|s| s.strip_prefix("sum="))
                .and_then(|v| v.parse().ok())
            else {
                continue;
            };
            out.push((tag.to_string(), t, sum));
        }
        Ok(out)
    };
    let (a, b) = match (parse(a_path), parse(b_path)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if a.is_empty() || b.is_empty() {
        eprintln!("error: 한쪽 로그에 [npck] 행 없음");
        return ExitCode::FAILURE;
    }
    let mut diffs = Vec::new();
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut seen = std::collections::HashSet::new();
    while ia < a.len() && ib < b.len() {
        let (ta, tta, sa) = &a[ia];
        let (tb, ttb, sb) = &b[ib];
        if ta == tb && tta == ttb {
            let denom = sa.abs().max(1e-9);
            let rel = (sa - sb).abs() / denom;
            if rel > rel_lim && seen.insert((ta.clone(), *tta)) {
                diffs.push((ta.clone(), *tta, *sa, *sb, rel));
            }
            ia += 1;
            ib += 1;
        } else if ta < tb {
            eprintln!("[ckdiff] A에만: {ta} t={tta}");
            ia += 1;
        } else {
            eprintln!("[ckdiff] B에만: {tb} t={ttb}");
            ib += 1;
        }
    }
    // QA-22: 한쪽 로그 절단(크래시 전형) 잔여 — 종전엔 공통 접두만 비교하고
    // 꼬리 항목을 무보고 버려 "무상이" exit 0이었다.
    let (tail_a, tail_b) = (a.len() - ia, b.len() - ib);
    if tail_a + tail_b > 0 {
        eprintln!(
            "ckdiff: 짧은 쪽 종료 후 잔여 — A에만 {tail_a}·B에만 {tail_b}항 (한쪽 절단 의심)"
        );
        return ExitCode::FAILURE;
    }
    if diffs.is_empty() {
        println!(
            "ckdiff: 무상이 ({} 마커, rel<{rel_lim})",
            a.len().min(b.len())
        );
        return ExitCode::SUCCESS;
    }
    println!(
        "ckdiff: {}개 상이 태그 (rel>{rel_lim}) — 첫 상이:",
        diffs.len()
    );
    for (tag, t, sa, sb, rel) in diffs.iter().take(10) {
        println!("  {tag:<16} t={t:<4} A={sa:14.4} B={sb:14.4} rel={rel:.3e}");
    }
    ExitCode::FAILURE
}

/// llm170 exl3-check <exl3_dir> <q8.gguf> — EXL3 참조 디코드 ↔ GGUF Q8 대조
/// (plans/118 §3-1). Python 검증기(scripts/exl3_validate.py)의 내부화:
/// K=3/4/5 혼재 텐서의 128×128 블록 상관계수. 기준: corr ≥ 0.97(K=3 양자화
/// 오차 수준) — 그 미만이면 디코드 회귀.
fn cmd_exl3_check(exl3_dir: &str, gguf_path: &str) -> Result<String, String> {
    let ar =
        llm170_exl3::StArchive::open(std::path::Path::new(exl3_dir)).map_err(|e| e.to_string())?;
    let g =
        llm170_gguf::GgufFile::open(std::path::Path::new(gguf_path)).map_err(|e| e.to_string())?;
    let cases: &[(&str, &str)] = &[
        (
            "model.language_model.layers.0.mlp.gate_proj",
            "blk.0.ffn_gate.weight",
        ),
        (
            "model.language_model.layers.0.mlp.down_proj",
            "blk.0.ffn_down.weight",
        ),
        (
            "model.language_model.layers.3.self_attn.o_proj",
            "blk.3.attn_output.weight",
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "blk.0.attn_qkv.weight",
        ),
    ];
    use std::io::{Read, Seek, SeekFrom};
    let mut report = String::new();
    let mut all_ok = true;
    for (key, gname) in cases {
        let (corr, krate) = (|| -> Result<(f64, u32), String> {
            let w = llm170_exl3::Exl3Linear::load(&ar, key).map_err(|e| e.to_string())?;
            let ti = g
                .find_tensor(gname)
                .ok_or_else(|| format!("gguf tensor not found: {gname}"))?;
            if ti.ty != llm170_gguf::GgmlType::Q8_0 {
                return Err(format!("{gname}: Q8_0 아님({:?})", ti.ty));
            }
            let (start, end) = ti
                .file_range(g.data_offset)
                .ok_or_else(|| format!("{gname}: 범위 계산 불가"))?;
            // 참조는 처음 128행(출력)만 — 텐서 전체 미로딩.
            let row_bytes = (ti.ne[0] as usize / 32) * 34;
            let nread = ((end - start) as usize).min(128 * row_bytes);
            let mut raw = vec![0u8; nread];
            let mut f = std::fs::File::open(&g.path).map_err(|e| e.to_string())?;
            f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
            f.read_exact(&mut raw).map_err(|e| e.to_string())?;
            let k = ti.ne[0] as usize;
            let mut wref = vec![0f64; 128 * 128]; // [k][n]
            let mut row = vec![0f32; k];
            for n in 0..128usize {
                llm170_core::quant::dequant_row(ti.ty, &raw, n as u64, k as u64, &mut row);
                for (i, v) in row.iter().take(128).enumerate() {
                    wref[i * 128 + n] = *v as f64;
                }
            }
            let wex = w.dequant_block_f64(0, 0, 128, 128);
            let (ma, mb) = (
                wex.iter().sum::<f64>() / 16384.0,
                wref.iter().sum::<f64>() / 16384.0,
            );
            let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
            for i in 0..16384 {
                let (a, b) = (wex[i] - ma, wref[i] - mb);
                sab += a * b;
                saa += a * a;
                sbb += b * b;
            }
            Ok((sab / (saa * sbb).sqrt(), w.krate))
        })()
        .inspect_err(|_| all_ok = false)?;
        let ok = corr >= 0.97;
        all_ok &= ok;
        let short: String = key.split('.').rev().take(2).collect::<Vec<_>>().join(".");
        report.push_str(&format!(
            "{short:44} K={krate} corr={corr:.5} {}\n",
            if ok { "ok" } else { "FAIL" }
        ));
    }
    if all_ok {
        report.push_str("exl3-check: 전 텐서 통과 (기준 corr ≥ 0.97)");
        Ok(report)
    } else {
        Err(report)
    }
}
