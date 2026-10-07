//! rawcuda 독립 검증 셔임 (plans/124 G2, 2026-10-04).
//! G1 패턴(단발 rustc 검증)의 제품화 — 이후 목표(G3+)가 같은 하니스에
//! 프로브 서브커맨드를 추가해 재사용한다. rustc --edition 2024로 이 파일과
//! rawcuda 전체(공유 글루 exl3_cuda + G10 분리 모듈 파일들)를 직접 컴파일한다
//! (전체 워크스페이스는 Windows에서 llm170-core mmap 결함으로 cargo 불가 —
//! G1 원장. rawcuda는 std 전용이라 단독 컴파일이 계약 위반이 아니다).
//! 실행은 반드시 워크트리 루트에서(상대 자산 경로 계약 —
//! scripts/verify_cuda.bat가 보장).

#![allow(dead_code)] // G3+ 미구현 필드·스텁 — 독립 빌드 dead_code 경고 억제

#[path = "../crates/backend-gpu/src/rawcuda/mod.rs"]
mod rawcuda;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let prog = args
        .first()
        .map(String::as_str)
        .unwrap_or("cuda_probe")
        .to_string();
    // FNF fn-gdn-* 서브커맨드 — 조기 인터셉트(파일 끝 fn_gdn_main으로
    // 위임 — q4_main과 동일 패턴, 2026-10-05). fn-gdn | fn-gdn-neg.
    if let Some(sub) = args.get(1).and_then(|a| a.strip_prefix("fn-gdn")) {
        return fn_gdn_main(sub);
    }
    // FNH fn-chain-* 서브커맨드(2026-10-05) — fn_chain_cuda_probe 위임.
    // fn-chain | fn-chain-neg.
    if let Some(sub) = args.get(1).and_then(|a| a.strip_prefix("fn-chain")) {
        let gguf = std::env::args()
            .nth(2)
            .unwrap_or_else(|| rawcuda::fn_support::FN_GGUF_MAIN.to_string());
        let r = match sub {
            "" | "-check" => rawcuda::fn_chain_cuda_probe::cuda_fn_chain_check(&gguf),
            "-neg" => rawcuda::fn_chain_cuda_probe::cuda_fn_chain_negative_check(&gguf),
            _ => Err(format!(
                "사용법: fn-chain [gguf_main] | fn-chain-neg [gguf_main]"
            )),
        };
        return match r {
            Ok(s) => {
                println!("{s}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("FAIL: {e}");
                ExitCode::FAILURE
            }
        };
    }
    // G8 q4-* 서브커맨드 — 조기 인터셉트(파일 끝 q4_main으로 위임 —
    // 별도 모듈 파일 q4_cuda_probe와 같은 독립 파일 구조 계약, 2026-10-04).
    if let Some(sub) = args.get(1).and_then(|a| a.strip_prefix("q4-")) {
        return q4_main(&prog, sub);
    }
    // FNG mtp-frame 서브커맨드(mtp-frame | mtp-frame-neg [dir [gguf]]) —
    // 조기 인터셉트(파일 끝 mtp_frame_main으로 위임 — hc 인터셉트와 동일
    // 계약, 2026-10-05). 기존 G9 mtp/mtp-neg(EXL3 27B)과 별개 서브커맨드.
    if let Some(a1) = args.get(1).map(String::as_str) {
        if a1 == "mtp-frame" || a1.starts_with("mtp-frame-") {
            return mtp_frame_main(&args);
        }
    }
    // FNC hc 서브커맨드(hc | hc-neg [dir]) — 조기 인터셉트(파일 끝
    // hc_main으로 위임 — q4 인터셉트와 동일 독립 파일 구조 계약, 2026-10-05).
    let hc_sub = args.get(1).map(String::as_str);
    if hc_sub == Some("hc") || hc_sub.map(|s| s.starts_with("hc-")).unwrap_or(false) {
        return hc_main(&prog, &args);
    }
    // plans/130 B3 ds4-hc 서브커맨드(ds4-hc | ds4-hc-neg [dir]) — 조기 인터셉트(파일 끝
    // ds4_hc_main 위임 — hc 인터셉트와 동일 계약, 2026-10-05).
    if let Some(a1) = args.get(1).map(String::as_str) {
        if a1 == "ds4-hc" || a1.starts_with("ds4-hc-") {
            return ds4_hc_main(&prog, &args);
        }
    }
    // plans/130 B4 ds4-mtp 서브커맨드(ds4-mtp | ds4-mtp-neg [dir]) — 조기
    // 인터셉트(파일 끝 ds4_mtp_main 위임 — ds4-hc 인터셉트와 동일 계약,
    // 2026-10-05). DSpark 드래프트 스테이지 체인 프로브.
    if let Some(a1) = args.get(1).map(String::as_str) {
        if a1 == "ds4-mtp" || a1.starts_with("ds4-mtp-") {
            return ds4_mtp_main(&prog, &args);
        }
    }
    let r = match args.get(1).map(String::as_str) {
        // S10 디바이스 상주 vs 호스트 스테이징 종단 대조(신규 파일 위임).
        Some("s10") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("");
            let ntok: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
            let lim: usize = args
                .get(4)
                .and_then(|s| s.parse().ok())
                .map(|v: usize| if v == 0 { usize::MAX } else { v })
                .unwrap_or(8);
            return match s10_check(dir, ntok, lim) {
                Ok(s) => {
                    println!("{s}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("FAIL: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        Some("smoke") => rawcuda::exl3_cuda_probe::cuda_smoke_check(),
        Some("gemv") => rawcuda::gemv_cuda_probe::cuda_gemv_check(),
        Some("gemv-neg") => rawcuda::gemv_cuda_probe::cuda_gemv_negative_check(),
        Some("norm") => rawcuda::norm_cuda_probe::cuda_norm_check(),
        Some("norm-neg") => rawcuda::norm_cuda_probe::cuda_norm_negative_check(),
        Some("gemm2") => rawcuda::gemm2_cuda_probe::cuda_gemm2_check(),
        Some("gemm2-neg") => rawcuda::gemm2_cuda_probe::cuda_gemm2_negative_check(),
        Some("gemm2-debug") => rawcuda::gemm2_cuda_probe::cuda_gemm2_debug_check(),
        Some("gemv-debug") => rawcuda::gemv_cuda_probe::cuda_gemv_debug_check(),
        Some("gdn") => match (args.get(2), args.get(3)) {
            (Some(a), Some(b)) => rawcuda::gdn_cuda_probe::cuda_gdn_check(a, b),
            (Some(a), None) => rawcuda::gdn_cuda_probe::cuda_gdn_check(
                a,
                "D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw",
            ),
            _ => rawcuda::gdn_cuda_probe::cuda_gdn_check(
                "D:/models/Qwen3.8-27B-exl3-4.00bpw",
                "D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw",
            ),
        },
        Some("gdn-neg") => match args.get(2) {
            Some(a) => rawcuda::gdn_cuda_probe::cuda_gdn_negative_check(a),
            None => rawcuda::gdn_cuda_probe::cuda_gdn_negative_check(
                "D:/models/Qwen3.8-27B-exl3-4.00bpw",
            ),
        },
        Some("attn") => match (args.get(2), args.get(3)) {
            (Some(a), Some(b)) => rawcuda::attn_cuda_probe::cuda_attn_check(a, b),
            (Some(a), None) => rawcuda::attn_cuda_probe::cuda_attn_check(
                a,
                "D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw",
            ),
            _ => rawcuda::attn_cuda_probe::cuda_attn_check(
                "D:/models/Qwen3.8-27B-exl3-4.00bpw",
                "D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw",
            ),
        },
        Some("attn-neg") => match args.get(2) {
            Some(a) => rawcuda::attn_cuda_probe::cuda_attn_negative_check(a),
            None => rawcuda::attn_cuda_probe::cuda_attn_negative_check(
                "D:/models/Qwen3.8-27B-exl3-4.00bpw",
            ),
        },
        Some("ew") => rawcuda::ew_argmax_cuda_probe::cuda_ew_check(),
        Some("argmax") => rawcuda::ew_argmax_cuda_probe::cuda_argmax_check(),
        Some("argmax-neg") => rawcuda::ew_argmax_cuda_probe::cuda_argmax_negative_check(),
        Some("gemv-real") => match (args.get(2), args.get(3)) {
            (Some(d), Some(k)) => rawcuda::gemv_cuda_probe::cuda_gemv_real_check(d, k),
            _ => Err("gemv-real <model_dir> <tensor-key>".into()),
        },
        Some("mtp") => match args.get(2) {
            Some(d) => rawcuda::mtp_cuda_probe::cuda_mtp_check(d),
            None => rawcuda::mtp_cuda_probe::cuda_mtp_check(
                "D:/models/Qwen3.8-27B-exl3-4.00bpw",
            ),
        },
        Some("mtp-neg") => match args.get(2) {
            Some(d) => rawcuda::mtp_cuda_probe::cuda_mtp_negative_check(d),
            None => rawcuda::mtp_cuda_probe::cuda_mtp_negative_check(
                "D:/models/Qwen3.8-27B-exl3-4.00bpw",
            ),
        },
        Some("fn") => rawcuda::fn_support::cuda_fn_smoke_check(),
        Some("fn-inv") => rawcuda::fn_support::cuda_fn_inventory_check(),
        Some("ple") => match args.get(2) {
            Some(d) => rawcuda::ple_cuda_probe::cuda_ple_check(d),
            None => rawcuda::ple_cuda_probe::cuda_ple_check(rawcuda::fn_support::FN_GGUF_MAIN),
        },
        Some("ple-neg") => match args.get(2) {
            Some(d) => rawcuda::ple_cuda_probe::cuda_ple_negative_check(d),
            None => rawcuda::ple_cuda_probe::cuda_ple_negative_check(rawcuda::fn_support::FN_GGUF_MAIN),
        },
        Some("moe") => rawcuda::moe_cuda_probe::cuda_moe_check(),
        Some("moe-neg") => rawcuda::moe_cuda_probe::cuda_moe_negative_check(),
        Some("qsa") => rawcuda::qsa_cuda_probe::cuda_qsa_check(args.get(2).map(String::as_str)),
        Some("qsa-neg") => {
            rawcuda::qsa_cuda_probe::cuda_qsa_negative_check(args.get(2).map(String::as_str))
        }
        // B3 ds4-attn 서브커맨드(2026-10-05) — ds4_attn_cuda_probe 위임.
        // ds4-attn [dir] | ds4-attn-neg [dir]. 실패는 전부 비영 exit.
        Some("ds4-attn") => match args.get(2) {
            Some(dd) => rawcuda::ds4_attn_cuda_probe::cuda_ds4_check(dd),
            None => rawcuda::ds4_attn_cuda_probe::cuda_ds4_check(
                rawcuda::ds4_attn_cuda_probe::DS4_EXL3_DIR,
            ),
        },
        Some("ds4-attn-neg") => match args.get(2) {
            Some(dd) => rawcuda::ds4_attn_cuda_probe::cuda_ds4_negative_check(dd),
            None => rawcuda::ds4_attn_cuda_probe::cuda_ds4_negative_check(
                rawcuda::ds4_attn_cuda_probe::DS4_EXL3_DIR,
            ),
        },
        Some("ds4-moe") => {
            rawcuda::ds4_moe_cuda_probe::cuda_ds4_moe_check(args.get(2).map(String::as_str))
        }
        Some("ds4-moe-neg") => rawcuda::ds4_moe_cuda_probe::cuda_ds4_moe_negative_check(
            args.get(2).map(String::as_str),
        ),
        _ => Err(format!(
            "사용법: {prog} smoke|gemv|gemv-neg|norm|norm-neg|gemm2|gemm2-neg|gemm2-debug|gemv-debug|gdn [dir27 [dir35]]|gdn-neg [dir27]|gemv-real <dir> <key>|attn [dir27 [dir35]]|attn-neg [dir27] [dir27 [dir35]]|attn-neg [dir27]|ew|argmax|argmax-neg|mtp [dir27]|mtp-neg [dir27]|fn|fn-inv|ple [gguf_main]|ple-neg [gguf_main]|moe|moe-neg|hc [dir]|hc-neg [dir]|mtp-frame [dir [gguf]]|mtp-frame-neg [dir]|qsa [gguf_main]|qsa-neg [gguf_main]|fn-chain [gguf_main]|fn-chain-neg [gguf_main]|ds4-moe [dir]|ds4-moe-neg [dir]|ds4-mtp [dir]|ds4-mtp-neg [dir]"
        )),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── S10 디바이스 상주 대조(plan:cuda-port.md) ──
// 실모델 EXL3 디코더를 두 벌 로드해(VRAM 2배 — 27B는 24GB에 불가하므로
// 작은 lim_layers를 쓴다) 같은 토큰열을 호스트 스테이징·디바이스 경로로
// 각각 그리디 디코드하고 종단 토큰열을 비교한다. 1스텝 로짓 대조로 놓치는
// 상태 오염(GDN 스캔·KV 누적)을 여러 스텝에 걸쳐 잡는다.
fn s10_check(dir: &str, ntok: usize, lim: usize) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    if dir.is_empty() {
        return Err("사용법: cuda_probe s10 <exl3_dir> [ntok]".into());
    }
    // S10 인자 확장: s10 <dir> [ntok] [layers] — layers가 주어지면 그만큼
    // 적재한다(0=전층). 2벌 로드이므로 VRAM 여유가 필요: 27B 전층 2벌은
    // 24GB에 불가하므로 기본은 8층이다.
    let mut host = Exl3CudaDecoder::load_slots(dir, lim, 256, 1)?;
    let mut dev = Exl3CudaDecoder::load_slots(dir, lim, 256, 1)?;
    // 결정론 프롬프트: 작은 정수 토큰열(어휘 앞쪽, 임베딩 범위 내).
    let toks: Vec<u32> = (0..ntok).map(|i| 100 + i as u32 * 7).collect();
    let line = rawcuda::exl3_cuda_device_probe::cuda_s10_stream_check(
        &mut host, &mut dev, &toks, 6,
    )?;
    // 진단 계기 — 결함 시 규위 좁히기용(정상 시 전부 0.000e0이어야 한다).
    let d_embed = rawcuda::exl3_cuda_device_probe::embed_resid_probe(
        &mut host, &mut dev, toks[0],
    )?;
    let d_norm = rawcuda::exl3_cuda_device_probe::first_norm_probe(
        &mut host, &mut dev, toks[0],
    )?;
    let d_mid = rawcuda::exl3_cuda_device_probe::mid_layer_probe(&mut host, &mut dev, toks[0])?;
    let d_attn = rawcuda::exl3_cuda_device_probe::attn_input_probe(&mut host, &mut dev)?;
    let d_step = rawcuda::exl3_cuda_device_probe::one_step_state_probe(
        &mut host, &mut dev, toks[0],
    )?;
    let d_trace = rawcuda::exl3_cuda_device_probe::layer_trace_probe(
        &mut host,
        &mut dev,
        toks[0],
        lim.min(8),
    )
    .unwrap_or_else(|e| format!("S10 layer-trace: 실행 실패({e})"));
    // layer_trace는 z를 제로 대입해 두 경로의 입력이 달라지는 계기라
    // 판정 근거가 아니라 진단 전용이다 — 결함 규위가 앞 단계에서 좁혀지지
    // 않을 때만 본다.
    let probe1 = rawcuda::exl3_cuda_device_probe::one_step_state_probe(
        &mut host, &mut dev, toks[0],
    )?;
    let probe2 = rawcuda::exl3_cuda_device_probe::mid_layer_probe(
        &mut host, &mut dev, toks[0],
    )?;
    let probe3 = rawcuda::exl3_cuda_device_probe::attn_input_probe(&mut host, &mut dev)?;
    // 격리: 동일 x에 대한 단일 GEMV 2회만 비교한다. 여기서 다르면 GEMV
    // 체인 문제, 같으면 forward 배선 문제로 갈린다. x 폭은 키마다 다르므로
    // 키별로 그 선형의 k에 맞춰 만든다(같은 **내용 패턴**을 폭만 맞춘다).
    let mut keys: Vec<String> = Vec::new();
    for il in 0..lim.min(8) {
        let lp = format!("model.language_model.layers.{il}");
        keys.push(format!("{lp}.linear_attn.in_proj_qkv"));
        keys.push(format!("{lp}.mlp.gate_proj"));
        keys.push(format!("{lp}.mlp.down_proj"));
    }
    let mut iso_rows: Vec<String> = Vec::new();
    for key in &keys {
        let k = match dev.lin_shape(key) {
            Some((k, _, _)) => k,
            None => continue,
        };
        let xg: Vec<f32> = (0..k).map(|i| ((i % 97) as f32) * 0.01).collect();
        let (md, _) = rawcuda::exl3_cuda_gemv_probe::single_gemv_compare(
            &mut host, &mut dev, key, &xg,
        )?;
        iso_rows.push(format!("{key}={md:.1e}"));
    }
    let iso = format!("S10 gemv-isolation: {}", iso_rows.join(" "));
    // 속도 비교 — 같은 8층으로 호스트/디바이스 경로 스텝당 시간을 잰다.
    // 위치축 한계(S9) 밖으로 나가지 않도록 소량만.
    const BENCH_STEPS: usize = 6;
    let h = rawcuda::exl3_cuda_device_bench::time_host_steps(&mut host, &toks, BENCH_STEPS)?;
    let d = rawcuda::exl3_cuda_device_bench::time_device_steps(&mut dev, &toks, BENCH_STEPS)?;
    let prof = rawcuda::exl3_cuda_device_bench::report(h, d, dev.device_name());
    Ok(format!(
    "{line}\n{iso}\n{d_embed}\n{d_norm}\n{d_mid}\n{d_attn}\n{d_step}\n{d_trace}\n\
     {prof} (layers={lim})"
))
}

// ── G8 Q4 모듈 서브커맨드(2026-10-04 — 독립 파일 q4_cuda_probe 위임) ──
// q4-dequant | q4-gemv | q4-gemm | q4-neg. 실패는 전부 비영 exit.
fn q4_main(prog: &str, sub: &str) -> ExitCode {
    use std::process::ExitCode as EC;
    let r = match sub {
        "dequant" => rawcuda::q4_cuda_probe::cuda_q4_dequant_check(),
        "gemv" => rawcuda::q4_cuda_probe::cuda_q4_gemv_check(),
        "gemm" => rawcuda::q4_cuda_probe::cuda_q4_gemm_check(),
        "neg" => rawcuda::q4_cuda_probe::cuda_q4_negative_check(),
        _ => Err(format!(
            "사용법: {prog} q4-dequant|q4-gemv|q4-gemm|q4-neg"
        )),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            EC::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            EC::FAILURE
        }
    }
}

// ── FNF Flash-Next GDN 서브커맨드(2026-10-05 — 독립 파일 fn_gdn_cuda_probe 위임) ──
// fn-gdn | fn-gdn-neg. 인자는 GGUF 메인 샤드 경로(기본 Flash-Next 실측
// 픽스처). 실패는 전부 비영 exit.
fn fn_gdn_main(sub: &str) -> ExitCode {
    let gguf = std::env::args().nth(2).unwrap_or_else(|| {
        rawcuda::fn_support::FN_GGUF_MAIN.to_string()
    });
    let r = match sub {
        "" | "-check" => rawcuda::fn_gdn_cuda_probe::cuda_fn_gdn_check(&gguf),
        "-neg" => rawcuda::fn_gdn_cuda_probe::cuda_fn_gdn_negative_check(&gguf),
        _ => Err(format!("사용법: fn-gdn [gguf_main] | fn-gdn-neg [gguf_main]")),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── FNC hc 모듈 서브커맨드(2026-10-05 — 독립 파일 hc_cuda_probe 위임) ──
// hc [dir] | hc-neg [dir]. 실패는 전부 비영 exit.
fn hc_main(prog: &str, args: &[String]) -> ExitCode {
    let dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or(rawcuda::hc_cuda_probe::FN_HC_EXL3_DIR);
    let r = match args.get(1).map(String::as_str) {
        Some("hc") => rawcuda::hc_cuda_probe::cuda_hc_check(dir),
        Some("hc-neg") => rawcuda::hc_cuda_probe::cuda_hc_negative_check(dir),
        _ => Err(format!("사용법: {prog} hc [dir]|hc-neg [dir]")),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── FNG Flash-Next MTP 드래프트 프레임 서브커맨드(2026-10-05 — 독립 파일
// mtp_fn_cuda_probe 위임). mtp-frame [dir [gguf]] | mtp-frame-neg [dir].
// 실패는 전부 비영 exit(음성대조 NEG-DETECTED 포함).
fn mtp_frame_main(args: &[String]) -> ExitCode {
    use std::process::ExitCode as EC;
    let dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or(rawcuda::mtp_fn_cuda_probe::FN_MTP_EXL3_DIR);
    let r = match args.get(1).map(String::as_str) {
        Some("mtp-frame") => {
            let gguf = args.get(3).map(String::as_str).unwrap_or("");
            rawcuda::mtp_fn_cuda_probe::cuda_mtp_frame_check(dir, gguf)
        }
        Some("mtp-frame-neg") => rawcuda::mtp_fn_cuda_probe::cuda_mtp_frame_negative_check(dir),
        _ => Err("사용법: mtp-frame [dir [gguf]] | mtp-frame-neg [dir]".into()),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            EC::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            EC::FAILURE
        }
    }
}

// ── plans/130 B3 DeepSeek-V4 mHC 모듈 서브커맨드(2026-10-05 — 독립 파일
// ds4_hc_cuda_probe 위임). ds4-hc [dir] | ds4-hc-neg [dir]. 실패는 전부 비영 exit(음성대조
// NEG-DETECTED 포함).
fn ds4_hc_main(prog: &str, args: &[String]) -> ExitCode {
    let dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or(rawcuda::ds4_hc_cuda_probe::DS4_HC_EXL3_DIR);
    let r = match args.get(1).map(String::as_str) {
        Some("ds4-hc") => rawcuda::ds4_hc_cuda_probe::cuda_ds4_hc_check(dir),
        Some("ds4-hc-neg") => rawcuda::ds4_hc_cuda_probe::cuda_ds4_hc_negative_check(dir),
        _ => Err(format!("사용법: {prog} ds4-hc [dir]|ds4-hc-neg [dir]")),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── plans/130 B4 DeepSeek-V4 DSpark MTP 드래프트 스테이지 서브커맨드
// (2026-10-05 — 독립 파일 ds4_mtp_cuda_probe 위임). ds4-mtp [dir] |
// ds4-mtp-neg [dir]. 실패는 전부 비영 exit(음성대조 NEG-DETECTED 포함).
fn ds4_mtp_main(prog: &str, args: &[String]) -> ExitCode {
    let dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or(rawcuda::ds4_mtp_cuda_probe::DS4_MTP_EXL3_DIR);
    let r = match args.get(1).map(String::as_str) {
        Some("ds4-mtp") => rawcuda::ds4_mtp_cuda_probe::cuda_ds4_mtp_check(dir),
        Some("ds4-mtp-neg") => rawcuda::ds4_mtp_cuda_probe::cuda_ds4_mtp_negative_check(dir),
        _ => Err(format!("사용법: {prog} ds4-mtp [dir]|ds4-mtp-neg [dir]")),
    };
    match r {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}
