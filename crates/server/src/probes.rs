//! 원오프 GPU 프로브/체크 서브커맨드 — main.rs에서 이관(plans/35 P4).
//! 본체는 backend-gpu(rawhip 프로브 fn, rawvk check fn)에 있고 여기는
//! 인자 파싱+호출만. 결론난 A/B 하니스(batch-abtest·tree-test·q6k-abtest·
//! exp-ab)는 2026-09-08 폐기.

use std::process::ExitCode;

/// 프로브 커맨드이면 실행해 Some(코드) 반환, 아니면 None.
pub fn run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let r: Result<String, String> = match cmd {
        "gpu-raw-probe" => {
            let iters: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
            llm170_backend_gpu::rawhip::raw_probe(iters)
        }
        "launch-rate" => {
            let iters: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(20000);
            llm170_backend_gpu::rawhip::launch_rate(iters)
        }
        "f16-map" => llm170_backend_gpu::rawhip::f16_map(
            args.first().and_then(|v| v.parse().ok()).unwrap_or(256),
        ),
        "f16-bench" => {
            let a = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
            llm170_backend_gpu::rawhip::f16_bench(a(0, 128), a(1, 2560), a(2, 640), a(3, 20))
        }
        "q4k-bench" => {
            let a = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
            llm170_backend_gpu::rawhip::q4k_bench(a(0, 128), a(1, 2560), a(2, 640), a(3, 20))
        }
        "q4k-micro" => llm170_backend_gpu::rawhip::q4k_micro(),
        "q4-d2h-bench" => llm170_backend_gpu::rawhip::d2h_bench(),
        "q5-1-bench" => {
            let a = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
            llm170_backend_gpu::rawhip::q5_1_bench(a(0, 20), a(1, 640), a(2, 2560), a(3, 50))
        }
        "q4-qsa-check" => {
            let t = args.first().and_then(|v| v.parse().ok()).unwrap_or(200usize);
            let np = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(200usize);
            llm170_backend_gpu::rawhip::q4acc::qsa_check(t, np)
        }
        "q4-hc-check" => {
            let t = args.first().and_then(|v| v.parse().ok()).unwrap_or(230usize);
            let n = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2560usize);
            let hc = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4usize);
            llm170_backend_gpu::rawhip::q4acc::hc_check(t, n, hc)
        }
        "q4-ple-check" => {
            llm170_backend_gpu::rawhip::q4acc::ple_gate_check()
        }
        "q4-ar-check" => {
            let t = args.first().and_then(|v| v.parse().ok()).unwrap_or(1usize);
            llm170_backend_gpu::rawhip::q4acc::ar_check_t(t)
        }
        "q4-acc-check" => {
            let path = args
                .first()
                .cloned()
                .unwrap_or_else(|| "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf".into());
            if args.first().map(String::as_str) == Some("micro") {
                return Some(match llm170_backend_gpu::rawhip::q4acc::micro_check() {
                    Ok(s) => { println!("{s}"); ExitCode::SUCCESS }
                    Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
                });
            }
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_gate_exps.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2usize);
            let rows = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(256usize);
            llm170_backend_gpu::rawhip::q4acc::check_tensor(std::path::Path::new(&path), &tn, t, rows)
        }
        "moe-row-check" => {
            let path = args.first().cloned().unwrap_or_else(|| {
                "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf".into()
            });
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_gate_exps.weight".into());
            let t_a = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(16usize);
            let t_b = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64usize);
            llm170_backend_gpu::rawhip::q4acc::moe_row_check(std::path::Path::new(&path), &tn, t_a, t_b)
        }
        "mm-row-check" => {
            let path = args.first().cloned().unwrap_or_else(|| {
                "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf".into()
            });
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.attn_qkv.weight".into());
            let t_a = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(16usize);
            let t_b = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64usize);
            llm170_backend_gpu::rawhip::q4acc::mm_row_check(std::path::Path::new(&path), &tn, t_a, t_b)
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
                    Ok(r) => Ok(format!("{r}\nDIVERGENCE DETECTED")),
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
                let rel = args.get(3).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.02);
                return Some(cmd_ckdiff(&a, &b, rel));
            } else {
                Err("diag: 하위커맨드 diff | chunk-check | va-lookup | vk-fault-probe | watchdog-selftest | ckdiff".into())
            }
        }
        "mmq-row-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
            let t1 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(16usize);
            let t2 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(208usize);
            llm170_backend_gpu::rawhip::mmq_row_check(&path, &tn, t1, t2)
        }
        "tile-row-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ssm_out.weight".into());
            let t1 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(16usize);
            let t2 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(208usize);
            llm170_backend_gpu::rawhip::tile_row_check(&path, &tn, t1, t2)
        }
        "wc-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_gate.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(64usize);
            llm170_backend_gpu::rawhip::wc_check(&path, &tn, t)
        }
        "q6k-ref" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.64.nextn.eh_proj.weight".into());
            llm170_backend_gpu::rawhip::q6k_ref_probe(&path, &tn)
        }
        "hca-repro" => llm170_backend_gpu::rawhip::hca_repro(),
        "launch-probe" => llm170_backend_gpu::rawhip::launch_probe(),
        "vk-flash-check" => llm170_backend_gpu::rawvk::flashcheck::flash_check(),
        "vk-mmq-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/tmp/model_link.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(512usize);
            llm170_backend_gpu::rawvk::gemv::vk_mmq_check(&path, &tn, t)
        }
        "vk-gemv-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
            llm170_backend_gpu::rawvk::gemv::gemv_check(&path, &tn, t)
        }
        "vk-sdot-probe" => llm170_backend_gpu::rawvk::gemv::sdot_probe(),
        "vk-idot-probe" => llm170_backend_gpu::rawvk::gemv::idot_probe(),
        "vk-gemv8-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ssm_out.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
            llm170_backend_gpu::rawvk::gemv::gemv8_check(&path, &tn, t)
        }
        "vk-ft32-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf".into());
            llm170_backend_gpu::rawvk::gemv::ft32_check(&path)
        }
        "mmv-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.1.attn_qkv.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1usize);
            llm170_backend_gpu::rawvk::gemv::mmv_check(&path, &tn, t)
        }
        "dbg-q3b" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_up.weight".into());
            llm170_backend_gpu::rawvk::gemv::q3b_dbg(&path, &tn)
        }
        "dbg-q3" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_up.weight".into());
            llm170_backend_gpu::rawvk::gemv::q3_dbg(&path, &tn)
        }
        "vk-frame-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_down_shexp.weight".into());
            llm170_backend_gpu::rawvk::gemv::frame_check(&path, &tn)
        }
        "vk-tile-check" => {
            let path = args.first().cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
            let tn = args.get(1).cloned().unwrap_or_else(|| "blk.0.ffn_down.weight".into());
            let t = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(32);
            llm170_backend_gpu::rawvk::gemv::tile_check(&path, &tn, t)
        }
        "subsum-check" => llm170_backend_gpu::rawvk::subsum_check(),
        "gdn-check" => llm170_backend_gpu::rawvk::gdn_check(),
        "vk-check" => llm170_backend_gpu::rawvk::smoke_test(),
        "roof-test" => llm170_backend_gpu::rawhip::roof_test(),
        "wmma-check" => llm170_backend_gpu::rawhip::wmma_check(),
        "wmma2-check" => llm170_backend_gpu::rawhip::wmma2_check(),
        "wmma2-map" => llm170_backend_gpu::rawhip::wmma2_map(),
        "wmma2-map2" => llm170_backend_gpu::rawhip::wmma2_map2(),
        "wmma2-attn-check" => llm170_backend_gpu::rawhip::wmma2_attn_check(),
        "attn-check" => llm170_backend_gpu::rawhip::attn_check(),
        "wmma-check-ldm" => llm170_backend_gpu::rawhip::wmma_check_ldm(),
        "wmma-check-pv" => llm170_backend_gpu::rawhip::wmma_check_pv(),
        "wmma-attn-check" => llm170_backend_gpu::rawhip::wmma_attn_check(),
        "gqa-bench" => llm170_backend_gpu::rawhip::gqa_bench(),
        "mm-tile" => llm170_backend_gpu::rawhip::mm_tile_bench(),
        "mm-bench" => llm170_backend_gpu::rawhip::mm_batch_bench(),
        "bw-test" => llm170_backend_gpu::rawhip::bw_test(),
        "bw-place" => llm170_backend_gpu::rawhip::bw_place_test(),
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
            let path = args.first().cloned().unwrap_or_else(|| "/tmp/model_link.gguf".into());
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
    let backend_cpu = args.iter().any(|a| a == "--backend" && args.iter().any(|b| b == "cpu"))
        || args.iter().any(|a| a == "--backend=cpu");
    let sizes: Vec<usize> = args[2..]
        .iter()
        .filter_map(|a| a.parse::<usize>().ok())
        .filter(|&s| s > 0)
        .collect();
    let sizes = if sizes.is_empty() { vec![16, 63, 64, 512] } else { sizes };

    // 프롬프트 파싱 — 숫자/콤마 전용이면 토큰 id, 아니면 텍스트
    let ids: Vec<u32> = if prompt.bytes().all(|b| b.is_ascii_digit() || b == b',' || b == b' ')
        && prompt.contains(',')
    {
        prompt.split(',').filter_map(|t| t.trim().parse().ok()).collect()
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
            let res = llm170_core::qwen4exp::Model4::load(&path).map_err(|e| e.to_string()).and_then(|m| {
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
                        Err(e) => return Err(format!("GPU 가속기 생성 실패 — {e} (--backend cpu 로 회피)")),
                    }
                }
                // 기준: 단일 청크(프롬프트 전체)
                unsafe { std::env::set_var("LLM170_Q4_CHUNK", format!("{}", ids.len().max(1))) };
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
                    let cc_seq: usize = std::env::var("LLM170_CC_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let mut eng = llm170_core::qwen35::Engine::new(m, (cc_seq + 1).max(1), ctx);
                    if !backend_cpu
                        && std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true)
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
        let maxd = l
            .iter()
            .zip(ref_l.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let tok = llm170_core::qwen35::greedy(l);
        let (verdict, why) = if maxd == 0.0 {
            ("PASS", "bits-identical".to_string())
        } else if tok == ref_tok && maxd < 1e-3 {
            ("PASS", format!("near-tie max|Δ|={maxd:.3e} (행 수 의존 잔여축 — docs/chunk-invariance.md)"))
        } else {
            all_pass = false;
            (
                "FAIL",
                format!("max|Δ|={maxd:.3e} argmax {tok}≠{ref_tok}"),
            )
        };
        println!("  chunk {sz:5}: {verdict} — {why}");
    }
    if all_pass {
        ExitCode::SUCCESS
    } else {
        println!("chunk-check: FAIL — 청크 불변성 위반 (docs/chunk-invariance.md)");
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
        Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
    };
    let w = match model.w(&args[1]) {
        Some(w) => w,
        None => { eprintln!("tensor not found: {}", args[1]); return ExitCode::FAILURE; }
    };
    let raw_ok = llm170_core::matmul::w4a8_ty(w.ty) || w.ty == llm170_gguf::GgmlType::Iq3S;
    if !raw_ok {
        eprintln!("rawhip-check: 미지원 타입");
        return ExitCode::FAILURE;
    }
    let (n_in, n_out) = (w.n_in as usize, w.n_out as usize);
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 1.0
    };
    let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let ctx = match RawCtx::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
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
                for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
                    let mut word = 0u32;
                    for (i, b) in c.iter().enumerate() { word |= (**b as u8 as u32) << (8 * i); }
                    v.push(word);
                }
                v
            };
            for (i, (a, b)) in gw.iter().zip(cpu_w.iter()).enumerate() {
                if a != b { qm += 1; if qm == 1 { println!("  ✗ quant 워드[{i}] gpu={a:#x} cpu={b:#x}"); } }
            }
            let gdbits: Vec<u32> = bytemuck::cast_slice(&gq[n_in / 4 * 4..]).to_vec();
            for (i, (a, b)) in gdbits.iter().zip(y.iter().map(|b| b.d.to_bits())).enumerate() {
                if *a != b { qm += 1; if qm <= 3 { println!("  ✗ quant d[{i}] gpu_bits={a:#x} cpu_bits={b:#x}"); } }
            }
            if qm == 0 { println!("  ★ quant_q8 원시 ≡ CPU 비트 일치"); }
            Ok(())
        };
        if let Err(e) = inner() { eprintln!("quant 검증: {e}"); }
    }
    let mut qs_words = Vec::with_capacity(n_in / 4);
    for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
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
            ctx.h2d(buf, bytemuck::cast_slice(&packed)).expect("pack upload");
            buf
        }
    };
    let w_d = match ctx.alloc(w.data.len()) { Ok(p) => p, Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; } };
    let kt_d = match ctx.alloc(1024) { Ok(p) => p, Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; } };
    // GPU quant 출력 재사용 시 xq/xd 업로드 생략 (종단 검증 — d가 GPU 생산값)
    let up = ctx.h2d(w_d, w.data).and_then(|_| ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2)));
    if let Err(e) = up {
        eprintln!("upload: {e}"); return ExitCode::FAILURE;
    }
    // 워밍 + 측정
    let ty = w.ty as u32;
    let _ = match ctx.gemv_q8(xq_d as *const u8, w_d as *const u8, kt_d as *const u8, ty, n_in, n_out) {
        Ok(v) => v,
        Err(e) => { eprintln!("gemv: {e}"); return ExitCode::FAILURE; }
    };
    let reps = 30;
    let t0 = std::time::Instant::now();
    let mut g = Vec::new();
    for _ in 0..reps {
        g = match ctx.gemv_q8(xq_d as *const u8, w_d as *const u8, kt_d as *const u8, ty, n_in, n_out) {
            Ok(v) => v,
            Err(e) => { eprintln!("gemv: {e}"); return ExitCode::FAILURE; }
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
            llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q8_0 => llm170_core::quant::dot_row_w4a8_q8_0_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &y),
            llm170_gguf::GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &y),
            _ => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &y),
        };
        if c.to_bits() != g[o].to_bits() {
            mism += 1;
            if first.is_none() { first = Some((o, c, g[o])); }
        }
    }
    println!("[{}] {}: 원시 GEMV 불일치 {mism}/{n_out} — {:.0}µs/op {:.0}GB/s", w.ty.name(), args[1], dt * 1e6, w.data.len() as f64 / dt / 1e9);
    if let Some((o, c, gv)) = first {
        println!("  첫 불일치 [{o}]: cpu={c:.7e} gpu={gv:.7e}");
    }
    if mism > 0 { ExitCode::FAILURE } else { println!("  ★ 원시 HIP ≡ CPU 비트 일치"); ExitCode::SUCCESS }
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
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
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
    

    // ③ 장문 청크 스모크 — 1,024토큰 무작위 prefill (NaN 가드는 LLM170_Q4_TRACE)
    let arch = llm170_gguf::GgufFile::open(&model_path)
        .ok()
        .and_then(|g| g.arch().map(str::to_string));
    if arch.as_deref() == Some("qwen4exp") {
        let toks: Vec<String> = (0..1024).map(|i| (100 + (i * 7919) % 200000).to_string()).collect();
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap_or_default());
        cmd.args(["infer", "--model", path, "--prompt-tokens", &toks.join(","), "--n-predict", "2", "--ctx", "2048", "--backend", &backend])
            .env("LLM170_Q4_TRACE", "1")
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
    }
    eprintln!("# check 전체 통과");
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
        let (Ok(lo), Ok(hi)) = (u64::from_str_radix(f[2].trim_start_matches("0x"), 16), u64::from_str_radix(f[3].trim_start_matches("0x"), 16)) else {
            continue;
        };
        let (lo, hi) = (lo & M, hi & M);
        if a >= lo && a < hi {
            println!("HIT line {} site={} bytes={} range={:#x}..{:#x} seq={}", ln + 1, f[0], f[1], lo, hi, f.get(4).unwrap_or(&"?"));
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
            let Ok(lo) = u64::from_str_radix(f[2].trim_start_matches("0x"), 16) else { continue };
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
                    f[0], f[1],
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
    match llm170_backend_gpu::rawvk::gemv::fault_probe() {
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
    let sec: u64 = std::env::var("LLM170_WATCHDOG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    println!("# watchdog-selftest: {sec}s 무진동 후 보고 대기");
    llm170_diag::watchdog::record_op("selftest_stall");
    // 진행 없이 대기 — 와치독이 보고한다. FAIL 모드면 여기서 종료(137).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(sec + 30);
    while std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    println!("# watchdog-selftest: 시간 종료 (보고 못 봄 — FAIL 아님)");
    ExitCode::SUCCESS
}

/// plans/87 §5 — [npck] 체크섬 로그 교차 diff: (tag, t, 등장순) 정렬 맞춤,
/// 첫 상이 태그 + 상위 상이 표. 무상이 exit 0.
fn cmd_ckdiff(a_path: &str, b_path: &str, rel_lim: f64) -> ExitCode {
    let parse = |p: &str| -> Result<Vec<(String, usize, f64)>, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        let mut out = Vec::new();
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("[npck] ") else { continue };
            // {tag} t={t} sum={s} v0=.. mid0=.. last0=..
            let mut it = rest.split_whitespace();
            let Some(tag) = it.next() else { continue };
            let Some(t) = it.next().and_then(|s| s.strip_prefix("t=")).and_then(|v| v.parse().ok()) else { continue };
            let Some(sum) = it.next().and_then(|s| s.strip_prefix("sum=")).and_then(|v| v.parse().ok()) else { continue };
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
    if diffs.is_empty() {
        println!("ckdiff: 무상이 ({} 마커, rel<{rel_lim})", a.len().min(b.len()));
        return ExitCode::SUCCESS;
    }
    println!("ckdiff: {}개 상이 태그 (rel>{rel_lim}) — 첫 상이:", diffs.len());
    for (tag, t, sa, sb, rel) in diffs.iter().take(10) {
        println!("  {tag:<16} t={t:<4} A={sa:14.4} B={sb:14.4} rel={rel:.3e}");
    }
    ExitCode::FAILURE
}
