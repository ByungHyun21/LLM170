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
        // S11 배치 동치성: T>1 배치가 T=1 순차와 같은 값을 내는지.
        // 프리필 배치화의 전제 조건(배치가 틀리면 배치화해서는 안 된다).
        Some("s11") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("");
            let nrows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
            let lim: usize = args
                .get(4)
                .and_then(|s| s.parse().ok())
                .map(|v: usize| if v == 0 { usize::MAX } else { v })
                .unwrap_or(8);
            return match s11_check(dir, nrows, lim) {
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
        // 슬롯 간 배치(T=2) 동치성 — 최초 발산층 이분 탐색.
        Some("ms") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("");
            let lim: usize = args
                .get(3)
                .and_then(|s| s.parse().ok())
                .map(|v: usize| if v == 0 { usize::MAX } else { v })
                .unwrap_or(8);
            return match ms_check(dir, lim) {
                Ok(s) => {
                    println!("{s}");
                    if s.contains("FAIL") {
                        ExitCode::FAILURE
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("FAIL: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        // 동일 프롬프트 순수 배치 — 행 상태 오염 vs 행별 산술 편차 분리.
        Some("ms4") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("");
            let pt: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(41);
            let lim: usize = args
                .get(4)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return match ms4_check(dir, pt, lim) {
                Ok(s) => {
                    println!("{s}");
                    if s.contains("FAIL") {
                        ExitCode::FAILURE
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("FAIL: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        // 1스텝 비트 판정 — argmax 마스킹 우회.
        Some("ms4b") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("");
            let pt: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(41);
            let lim: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            return match ms4b_check(dir, pt, lim) {
                Ok(s) => {
                    println!("{s}");
                    if s.contains("FAIL") {
                        ExitCode::FAILURE
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("FAIL: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        // 서버 타임라인 재현 — 슬롯 간 배치 배선 결함 국소화(2026-10-08).
        // msrv=배치 디코드, msrv-ser=동일 타임라인을 step_tok으로(요인 분리).
        Some("msrv") | Some("msrv-ser") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("");
            return match msrv_check(dir, args.get(1).map(String::as_str) == Some("msrv")) {
                Ok(s) => {
                    println!("{s}");
                    if s.contains("FAIL") {
                        ExitCode::FAILURE
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("FAIL: {e}");
                    ExitCode::FAILURE
                }
            };
        }
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
        Some("gemv-t") => rawcuda::gemv_cuda_probe::cuda_gemv_t_check(),
        Some("gemv-t-neg") => rawcuda::gemv_cuda_probe::cuda_gemv_t_negative_check(),
        Some("norm") => rawcuda::norm_cuda_probe::cuda_norm_check(),
        Some("norm-neg") => rawcuda::norm_cuda_probe::cuda_norm_negative_check(),
        Some("gemm2") => rawcuda::gemm2_cuda_probe::cuda_gemm2_check(),
        Some("gemm2-neg") => rawcuda::gemm2_cuda_probe::cuda_gemm2_negative_check(),
        Some("gemm2-debug") => rawcuda::gemm2_cuda_probe::cuda_gemm2_debug_check(),
        Some("gemv-debug") => rawcuda::gemv_cuda_probe::cuda_gemv_debug_check(),
        Some("gdn") => match (args.get(2), args.get(3)) {
            (Some(a), Some(b)) => rawcuda::gdn_cuda_probe::cuda_gdn_check(a, b),
            (Some(a), None) => {
                rawcuda::gdn_cuda_probe::cuda_gdn_check(a, "D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw")
            }
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
            None => rawcuda::mtp_cuda_probe::cuda_mtp_check("D:/models/Qwen3.8-27B-exl3-4.00bpw"),
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
            None => {
                rawcuda::ple_cuda_probe::cuda_ple_negative_check(rawcuda::fn_support::FN_GGUF_MAIN)
            }
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
            "사용법: {prog} smoke|gemv|gemv-neg|gemv-t|gemv-t-neg|norm|norm-neg|gemm2|gemm2-neg|gemm2-debug|gemv-debug|gdn [dir27 [dir35]]|gdn-neg [dir27]|gemv-real <dir> <key>|attn [dir27 [dir35]]|attn-neg [dir27] [dir27 [dir35]]|attn-neg [dir27]|ew|argmax|argmax-neg|mtp [dir27]|mtp-neg [dir27]|fn|fn-inv|ple [gguf_main]|ple-neg [gguf_main]|moe|moe-neg|hc [dir]|hc-neg [dir]|mtp-frame [dir [gguf]]|mtp-frame-neg [dir]|qsa [gguf_main]|qsa-neg [gguf_main]|fn-chain [gguf_main]|fn-chain-neg [gguf_main]|ds4-moe [dir]|ds4-moe-neg [dir]|ds4-mtp [dir]|ds4-mtp-neg [dir]"
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

// S11 동치성 검사 — GDN/어텐션이 T=1 순차와 같은 값을 내는지.
// 실모델 상수를 아카이브에서 읽어와 set_gdn에 넣어야 상태 리셋이 된다.
/// 슬롯 간 배치(T=2) 동치성 탐침 — 최초 발산층 이분 탐색용.
///
/// [왜 이 탐침이 필요한가] `decode_batch_slots`(슬롯 간 디코드 배치)는 T=1에서
/// 게이트 기준선 그대로 PASS하지만 T>1에서 행 간 상태 오염이 관측됐다 —
/// 서로 다른 두 프롬프트가 idx=9부터 완전히 동일한 토큰열을 냈다. 서버 배선은
/// 하지 않은 상태이고, 원인을 확정해야 배선 여부를 결정할 수 있다.
///
/// [방법] `lim_layers`로 층을 1·2·3·4·8… 줄여가며 같은 비교를 반복한다. 처음
/// 어긋나기 시작하는 lim이 곧 최초 발산층이다. lim=0은 전체 층.
///
/// [판정] 슬롯 0·1에 **길이가 다른** 프롬프트(5/9토큰)를 프리필한 뒤,
/// 같은 다음 토큰을 (a) T=2 한 번 vs (b) T=1 두 번으로 디코드해 슬롯별
/// argmax를 비교한다. 길이가 달라야 상태 공유가 드러난다(같은 길이는 우연히
/// 맞을 수 있다). 환원 순서 차이는 판정 대상이 아니다 — 같은 코드 경로에서
/// T만 다르므로 여기서 어긋나면 배선 오류다.
fn ms_check(dir: &str, lim: usize) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    if dir.is_empty() {
        return Err("사용법: cuda_probe ms <exl3_dir> [layers]".into());
    }
    // 두 프롬프트의 토큰열(게이트 한국어 문장의 앞부분에서 유효 어휘 id만).
    const P0: [u32; 5] = [148678, 65233, 202419, 220, 49849];
    const P1: [u32; 9] = [
        148678, 65233, 202419, 220, 49849, 155497, 220, 151314, 39504,
    ];
    const NEXT0: u32 = 149635;
    const NEXT1: u32 = 174675;

    // **디코더는 한 벌만** 쓴다 — 두 벌(각 64층 × 19.5GB)은 24GB 카드를 넘겨
    // OOM한다(같은 제약이 슬롯 격리 하네스에도 있다). 대신 (a) 배치 → 상태
    // 리셋 → (b) 직렬 을 **같은 디코더**에서 순서대로 돌린다. `reset_state`가
    // 슬롯의 GDN 링·스캔·KV·pos를 0으로 돌려주므로 두 실행의 상태가 같다.
    // (단 한 가지 예외 — §6의 '단일 배치 경로 T>1은 슬롯 0만 실측' 계열: 리셋이
    //  호출 사이의 상태만 지우므로 로직 결함은 그대로 드러난다.)
    let mut d = Exl3CudaDecoder::load_slots(dir, lim, 512, 2)?;
    let dev = d.device_name().to_string();

    let embed = |d: &mut Exl3CudaDecoder, toks: &[u32]| -> Result<Vec<f32>, String> {
        let mut rows = Vec::new();
        for &tk in toks {
            rows.extend_from_slice(&d.embed_row_host(tk));
        }
        Ok(rows)
    };
    // 프리필 — 슬롯 0은 5토큰, 슬롯 1은 9토큰(위치 다름). fwd3s 도메인(T≤8)에
    // 맞춰 8토큰 단위로 나눠 넣는다(S11 규약).
    let prefill = |d: &mut Exl3CudaDecoder| -> Result<(), String> {
        for (slot, toks) in [(0usize, &P0[..]), (1usize, &P1[..])] {
            let rows = embed(d, toks)?;
            for ch in rows.chunks(8 * d.hidden) {
                d.forward_batch_device(slot, ch)?;
            }
        }
        Ok(())
    };

    // (a) T=2 한 번.
    prefill(&mut d)?;
    let tb = d.decode_batch_slots(&[(0, NEXT0), (1, NEXT1)])?;
    // 리셋 후 (b) T=1 두 번 — 순서·상태를 (a)와 같게 만든다.
    d.reset_state(0)?;
    d.reset_state(1)?;
    prefill(&mut d)?;
    let s0 = d.decode_batch_slots(&[(0, NEXT0)])?;
    let s1 = d.decode_batch_slots(&[(1, NEXT1)])?;
    let ok0 = tb[0] == s0[0];
    let ok1 = tb[1] == s1[0];
    let lim_s = if lim == usize::MAX {
        "full".to_string()
    } else {
        lim.to_string()
    };
    Ok(format!(
        "device: {dev} | exl3-cuda-multislot (ms) lim_layers={lim_s} T=2 vs T=1: \
         slot0 {} (batch {} / serial {}) · slot1 {} (batch {} / serial {}) | {}",
        if ok0 { "OK" } else { "MISMATCH" },
        tb[0],
        s0[0],
        if ok1 { "OK" } else { "MISMATCH" },
        tb[1],
        s1[0],
        if ok0 && ok1 { "PASS" } else { "FAIL" }
    ))
}

// msrv 스텝 공용 — batch=true면 슬롯 간 배치 1스텝, false면 행별 step_tok.
// (A) 타임라인의 프리필 교차는 그대로 두고 **디코드 요인만** 분리한다:
// msrv-ser에서도 발산하면 프리필 교차·상태 쪽 결함, 발산하지 않으면
// 배치 디코드(gemv_t 체인) 쪽 결함이다.
fn msrv_step_group(
    d: &mut rawcuda::exl3_cuda::Exl3CudaDecoder,
    items: &[(usize, u32)],
    batch: bool,
) -> Result<Vec<u32>, String> {
    if batch {
        d.decode_batch_slots(items)
    } else {
        items.iter().map(|&(s, tok)| d.step_tok(s, tok)).collect()
    }
}

// ms4 — 동일 프롬프트 4슬롯 순수 배치: 프리필 교차 없이 슬롯 0..3에 같은
// 프롬프트를 프리필하고 T=4 배치로 24스텝. 행 i가 (a) 직렬 기준선과
// (b) 다른 행과 어긋나는지를 분리한다 — 행 간 어긋남 = 배치 내부 행 상태
// 오염(프롬프트 이질성 무관), 전 행 일치+직렬만 어긋남 = 행별 산술 편차.
// ptokens로 프리필 길이를 조절한다(기본 41 — msrv #0과 동일).
fn ms4_check(dir: &str, ptokens: usize, lim: usize) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    if dir.is_empty() {
        return Err("사용법: cuda_probe ms4 <exl3_dir> [ptokens] [layers]".into());
    }
    const BASE: [u32; 41] = [
        148678, 65233, 202419, 220, 49849, 155497, 220, 151314, 39504, 149635, 13, 220, 174675,
        30061, 220, 152055, 152065, 12434, 220, 154854, 149248, 80102, 20673, 220, 214009, 149789,
        11, 220, 60177, 148726, 22836, 220, 149965, 176289, 220, 12434, 160288, 220, 158201,
        149635, 13,
    ];
    let toks: Vec<u32> = BASE.iter().cycle().take(ptokens).copied().collect();
    const NGEN: usize = 24;
    let tmax = rawcuda::attn_cuda::ATTN_F3S_TMAX;
    let lim = if lim == 0 { usize::MAX } else { lim };
    let mut d = Exl3CudaDecoder::load_slots(dir, lim, 512, 4)?;
    let dev = d.device_name().to_string();
    let prefill = |d: &mut Exl3CudaDecoder, slot: usize, toks: &[u32]| -> Result<u32, String> {
        let mut last = Vec::new();
        for ch in toks.chunks(tmax) {
            let mut rows = Vec::with_capacity(ch.len() * d.hidden);
            for &tk in ch {
                rows.extend_from_slice(&d.embed_row_host(tk));
            }
            last = d.forward_batch_device(slot, &rows)?.0;
        }
        d.argmax_host(&last)
    };
    // 직렬 기준선 — 슬롯 0.
    let t0 = prefill(&mut d, 0, &toks)?;
    let mut ser = vec![t0];
    let mut nx = t0;
    let ts_start = std::time::Instant::now();
    while ser.len() < NGEN {
        let logits = d.forward_tok_device(0, nx)?;
        nx = d.argmax_host(&logits)?;
        ser.push(nx);
    }
    let serial_ms = ts_start.elapsed().as_secs_f64() * 1000.0 / (NGEN - 1) as f64;
    // 4슬롯 동일 프롬프트 재프리필.
    for s in 0..4 {
        d.reset_state(s)?;
    }
    let mut next = [0u32; 4];
    for s in 0..4 {
        next[s] = prefill(&mut d, s, &toks)?;
    }
    // 순수 T=4 배치 24스텝 — 스텝 시간 측정 포함(§1.2 launch/스텝 분해 기초).
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); 4];
    for s in 0..4 {
        rows[s].push(next[s]);
    }
    let t_start = std::time::Instant::now();
    let mut steps = 0usize;
    while rows.iter().all(|r| r.len() < NGEN) {
        let items: Vec<(usize, u32)> = (0..4).map(|i| (i, next[i])).collect();
        let tb = d.decode_batch_slots(&items)?;
        for (i, &t) in tb.iter().enumerate() {
            rows[i].push(t);
            next[i] = t;
        }
        steps += 1;
    }
    let batch_ms = t_start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let mut out = Vec::new();
    let mut fails = 0;
    for s in 0..4 {
        let first = (0..NGEN).find(|&j| rows[s][j] != ser[j]);
        let cross = (0..NGEN).any(|j| rows[s][j] != rows[0][j]);
        match first {
            None => out.push(format!("행{s}: 24 일치",)),
            Some(j) => {
                fails += 1;
                out.push(format!(
                    "행{s}: 최초 발산 idx={j} 배치={:?} 직렬={:?} 행간불일치={cross}",
                    &rows[s][j..(j + 4).min(NGEN)],
                    &ser[j..(j + 4).min(NGEN)]
                ));
            }
        }
    }
    if fails == 0 {
        Ok(format!(
            "device: {dev} | ms4 프롬프트 {ptokens}토큰 ×4슬롯 순수 T=4 (layers={lim}): 전 행 직렬과 24 일치 | PASS | 직렬 {serial_ms:.1}ms/스텝 · 배치 {batch_ms:.1}ms/스텝"
        ))
    } else {
        Ok(format!(
            "device: {dev} | ms4 FAIL({fails}/4) (layers={lim}) — {} | 직렬 {serial_ms:.1}ms · 배치 {batch_ms:.1}ms",
            out.join(" · ")
        ))
    }
}

/// 서버 타임라인 재현 탐침 — 슬롯 간 배치 배선 결함 국소화(2026-10-08,
/// plans/cuda-port.md §1). verify_cuda_slots.py에서 일부 슬롯이 idx=9부터
/// "빈 문맥 고정 출력"(25 198 16 13 …)으로 어긋나는 증상을 스케줄러 없이
/// 디코더 호출 순서만으로 재현한다 — 재현되면 디코더/타임라인 결함,
/// 재현되지 않으면 sched.rs·HTTP 계층 결함으로 좁혀진다.
///
/// [타임라인 — sched.rs ②디코드 우선·③프리필 1청크(512토큰 → 엔진이
/// 8토큰씩)의 실제 순서를 그대로 밟는다]
///   tick1: (활성 없음) 프리필 슬롯0 완료 → 첫 토큰 방출
///   tick2: 디코드 [0](단독 step_tok) → 프리필 슬롯1
///   tick3: 디코드 배치 [0,1] → 프리필 슬롯2
///   tick4: 디코드 배치 [0,1,2] → 프리필 슬롯3
///   tick5+: 디코드 배치 [0,1,2,3]
///
/// [판정] 슬롯 1 서버 직렬(모든 요청을 슬롯 0에서 reset→프리필→step_tok)
/// 의 토큰열과 행별 완전 일치해야 한다(gemv_t 비트계약 — 일치하지 않으면
/// 배치 타임라인이 상태를 오염시킨다).
fn msrv_check(dir: &str, batch: bool) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    if dir.is_empty() {
        return Err("사용법: cuda_probe msrv <exl3_dir>".into());
    }
    // verify_cuda_slots.py와 동일한 41토큰 기본 프롬프트 × k(1,2,3,5).
    const BASE: [u32; 41] = [
        148678, 65233, 202419, 220, 49849, 155497, 220, 151314, 39504, 149635, 13, 220, 174675,
        30061, 220, 152055, 152065, 12434, 220, 154854, 149248, 80102, 20673, 220, 214009, 149789,
        11, 220, 60177, 148726, 22836, 220, 149965, 176289, 220, 12434, 160288, 220, 158201,
        149635, 13,
    ];
    let prompts: Vec<Vec<u32>> = [1usize, 2, 3, 5]
        .iter()
        .map(|&k| BASE.iter().cycle().take(41 * k).copied().collect())
        .collect();
    const NGEN: usize = 24;
    let tmax = rawcuda::attn_cuda::ATTN_F3S_TMAX;

    let mut d = Exl3CudaDecoder::load_slots(dir, usize::MAX, 512, 4)?;
    let dev = d.device_name().to_string();

    // 프리필 — 엔진 prefill 규약(8토큰 청크 forward_batch_device) 그대로.
    // 반환 = 마지막 청크 로짓의 argmax(서버가 첫 토큰으로 방출하는 값).
    let prefill = |d: &mut Exl3CudaDecoder, slot: usize, toks: &[u32]| -> Result<u32, String> {
        let mut last = Vec::new();
        for ch in toks.chunks(tmax) {
            let mut rows = Vec::with_capacity(ch.len() * d.hidden);
            for &tk in ch {
                rows.extend_from_slice(&d.embed_row_host(tk));
            }
            last = d.forward_batch_device(slot, &rows)?.0;
        }
        d.argmax_host(&last)
    };

    // (A) 동시 타임라인 — 슬롯 i가 프롬프트 i를 담당(서버 배정 순서와
    // 동일 — 활성화 순서 0→1→2→3, 배치 구성 1→2→3→4로 단조 확장).
    let mut emit: Vec<Vec<u32>> = vec![Vec::new(); 4];
    let mut next = [0u32; 4];
    {
        // tick1: 프리필 슬롯0.
        let t = prefill(&mut d, 0, &prompts[0])?;
        emit[0].push(t);
        next[0] = t;
        // tick2: 디코드 [0] 단독 → 프리필 슬롯1.
        let t0 = d.step_tok(0, next[0])?;
        emit[0].push(t0);
        next[0] = t0;
        let t = prefill(&mut d, 1, &prompts[1])?;
        emit[1].push(t);
        next[1] = t;
        // tick3: 배치 [0,1] → 프리필 슬롯2.
        let tb = msrv_step_group(&mut d, &[(0, next[0]), (1, next[1])], batch)?;
        for (i, &t) in tb.iter().enumerate() {
            emit[i].push(t);
            next[i] = t;
        }
        let t = prefill(&mut d, 2, &prompts[2])?;
        emit[2].push(t);
        next[2] = t;
        // tick4: 배치 [0,1,2] → 프리필 슬롯3.
        let tb = msrv_step_group(
            &mut d,
            &[(0, next[0]), (1, next[1]), (2, next[2])],
            batch,
        )?;
        for (i, &t) in tb.iter().enumerate() {
            emit[i].push(t);
            next[i] = t;
        }
        let t = prefill(&mut d, 3, &prompts[3])?;
        emit[3].push(t);
        next[3] = t;
        // tick5+: 배치 [0,1,2,3] — 전원 24토큰까지.
        while emit.iter().any(|e| e.len() < NGEN) {
            let items: Vec<(usize, u32)> = (0..4).map(|i| (i, next[i])).collect();
            let tb = msrv_step_group(&mut d, &items, batch)?;
            for (i, &t) in tb.iter().enumerate() {
                if emit[i].len() < NGEN {
                    emit[i].push(t);
                    next[i] = t;
                }
            }
        }
    }

    // (B) 직렬 기준선 — 전 요청을 슬롯 0에서 reset→프리필→step_tok
    // (슬롯 1 서버 = 직렬 4요청의 재현).
    let mut ser: Vec<Vec<u32>> = Vec::new();
    for toks in &prompts {
        for s in 0..4 {
            d.reset_state(s)?;
        }
        let t0 = prefill(&mut d, 0, toks)?;
        let mut row = vec![t0];
        let mut nx = t0;
        while row.len() < NGEN {
            nx = d.step_tok(0, nx)?;
            row.push(nx);
        }
        ser.push(row);
    }

    let mut fails = Vec::new();
    for i in 0..4 {
        let (a, b) = (&emit[i], &ser[i]);
        let first = (0..NGEN).find(|&j| a[j] != b[j]);
        match first {
            None => println!(
                "device: {dev} | msrv 프롬프트#{i}({}토큰): 24토큰 일치",
                41 * (i + 1)
            ),
            Some(j) => {
                println!(
                    "device: {dev} | msrv 프롬프트#{i}({}토큰): 최초 발산 idx={j} \
                     동시={:?} 직렬={:?}",
                    41 * (i + 1),
                    &a[j..(j + 6).min(NGEN)],
                    &b[j..(j + 6).min(NGEN)]
                );
                fails.push(format!("#{i} idx={j}"));
            }
        }
    }
    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | msrv 서버 타임라인 재현: 전 슬롯 24토큰 일치 — \
             디코더·타임라인 무결(결함은 sched/http 계층)"
        ))
    } else {
        Ok(format!(
            "device: {dev} | msrv FAIL — 재현됨({}) — 디코더 배치 타임라인 결함",
            fails.join(", ")
        ))
    }
}

// ms4b — 1스텝 비트 판정: 배치 T=4 로짓 vs 직렬(S10 디바이스 경로) 로짓을
// to_bits로 행별 비교한다. argmax 다중스텝 판정의 마스킹(비단조 lim —
// lim=13 PASS·14 FAIL·15 PASS 실측)을 우회하는 자이그: 1스텝에서 이미
// rows≥1에 비트 편차가 있으면 전 lim에서 잡힌다.
fn ms4b_check(dir: &str, ptokens: usize, lim: usize) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    if dir.is_empty() {
        return Err("사용법: cuda_probe ms4b <exl3_dir> [ptokens] [layers]".into());
    }
    const BASE: [u32; 41] = [
        148678, 65233, 202419, 220, 49849, 155497, 220, 151314, 39504, 149635, 13, 220, 174675,
        30061, 220, 152055, 152065, 12434, 220, 154854, 149248, 80102, 20673, 220, 214009, 149789,
        11, 220, 60177, 148726, 22836, 220, 149965, 176289, 220, 12434, 160288, 220, 158201,
        149635, 13,
    ];
    let toks: Vec<u32> = BASE.iter().cycle().take(ptokens).copied().collect();
    let tmax = rawcuda::attn_cuda::ATTN_F3S_TMAX;
    let lim = if lim == 0 { usize::MAX } else { lim };
    let mut d = Exl3CudaDecoder::load_slots(dir, lim, 512, 4)?;
    let dev = d.device_name().to_string();
    let prefill = |d: &mut Exl3CudaDecoder, slot: usize, toks: &[u32]| -> Result<u32, String> {
        let mut last = Vec::new();
        for ch in toks.chunks(tmax) {
            let mut rows = Vec::with_capacity(ch.len() * d.hidden);
            for &tk in ch {
                rows.extend_from_slice(&d.embed_row_host(tk));
            }
            last = d.forward_batch_device(slot, &rows)?.0;
        }
        d.argmax_host(&last)
    };
    // 배치 1스텝 로짓 — 동일 프롬프트라 4슬롯 첫 토큰이 같다.
    let t0 = prefill(&mut d, 0, &toks)?;
    for s in 1..4 {
        let ts = prefill(&mut d, s, &toks)?;
        if ts != t0 {
            return Err(format!("ms4b: 슬롯 {s} 프리필 첫 토큰 {ts} ≠ 슬롯 0 {t0}"));
        }
    }
    let items: Vec<(usize, u32)> = (0..4).map(|i| (i, t0)).collect();
    let (bl, n_head) = d.decode_batch_slots_logits(&items)?;
    // 직렬 기준선 — 리셋 후 재프리필, S10 디바이스 경로 1스텝.
    for s in 0..4 {
        d.reset_state(s)?;
    }
    let mut ser: Vec<Vec<f32>> = Vec::new();
    for s in 0..4 {
        let ts = prefill(&mut d, s, &toks)?;
        ser.push(d.forward_tok_device(s, ts)?);
    }
    let mut out = Vec::new();
    let mut fails = 0;
    for s in 0..4 {
        let bd: usize = (0..n_head)
            .filter(|&j| bl[s * n_head + j].to_bits() != ser[s][j].to_bits())
            .count();
        if bd == 0 {
            out.push(format!("행{s}: 비트일치"));
        } else {
            fails += 1;
            let first = (0..n_head)
                .find(|&j| bl[s * n_head + j].to_bits() != ser[s][j].to_bits())
                .unwrap();
            let md = (0..n_head)
                .map(|j| (bl[s * n_head + j] - ser[s][j]).abs())
                .fold(0f32, f32::max);
            out.push(format!(
                "행{s}: bit-diff {bd}/{n_head} 첫={first} maxdiff={md:.3e}"
            ));
        }
    }
    if fails == 0 {
        Ok(format!(
            "device: {dev} | ms4b 1스텝 비트 (layers={lim}, 프롬프트 {ptokens}): 전 행 비트일치 | PASS"
        ))
    } else {
        Ok(format!(
            "device: {dev} | ms4b FAIL({fails}/4) (layers={lim}) — {}",
            out.join(" · ")
        ))
    }
}

fn s11_check(dir: &str, nrows: usize, lim: usize) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    if dir.is_empty() {
        return Err("사용법: cuda_probe s11 <exl3_dir> [nrows] [layers]".into());
    }
    let cfg = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| format!("config.json: {e}"))?;
    let dims = rawcuda::gdn_cuda::GdnDims::from_config(&cfg)?;
    let adims = rawcuda::attn_cuda::AttnDims::from_config(&cfg)?;
    let dec = Exl3CudaDecoder::load_slots(dir, lim, 512, 1)?;
    let dev = dec.device_name().to_string();
    // GDN 상수는 load가 이미 올렸지만 리셋을 위해 다시 읽는다(원장 S11:
    // 동치성 검사는 상태를 0으로 되돌려 두 번 돌려야 한다).
    let ar = rawcuda::exl3_cuda::StArchive::open(std::path::Path::new(dir))?;
    let rd = |name: &str| -> Result<Vec<f32>, String> {
        let dt = ar.dtype_of(name).ok_or_else(|| format!("{name} 없음"))?;
        let raw = ar.read(name)?;
        rawcuda::exl3_cuda_probe::st_to_f32(&raw, dt)
    };
    let (n, cch, hv, hd) = (dims.n_gdn, dims.conv_ch(), dims.h_v, dims.hidden);
    let mut cw = vec![0f32; n * cch * 4];
    let mut ab = vec![0f32; n * 2 * hv * hd];
    let mut alog = vec![0f32; n * hv];
    let mut dtb = vec![0f32; n * hv];
    let mut nw = vec![0f32; n * dims.d];
    for (gi, il) in (0..n).map(|g| (g, (g / 3) * 4 + g % 3)).collect::<Vec<_>>() {
        let lp = format!("model.language_model.layers.{il}.linear_attn");
        cw[gi * cch * 4..(gi + 1) * cch * 4].copy_from_slice(&rd(&format!("{lp}.conv1d.weight"))?);
        let a = rd(&format!("{lp}.in_proj_a.weight"))?;
        let b = rd(&format!("{lp}.in_proj_b.weight"))?;
        let base = gi * 2 * hv * hd;
        ab[base..base + hv * hd].copy_from_slice(&a);
        ab[base + hv * hd..(gi + 1) * 2 * hv * hd].copy_from_slice(&b);
        alog[gi * hv..(gi + 1) * hv].copy_from_slice(&rd(&format!("{lp}.A_log"))?);
        dtb[gi * hv..(gi + 1) * hv].copy_from_slice(&rd(&format!("{lp}.dt_bias"))?);
        nw[gi * dims.d..(gi + 1) * dims.d].copy_from_slice(&rd(&format!("{lp}.norm.weight"))?);
    }
    let consts = rawcuda::exl3_cuda_batch_probe::GdnConsts {
        cw,
        ab,
        alog,
        dtb,
        nw,
    };
    let mut dec = dec;
    let mut rows = String::new();
    let mut ok = true;
    // GDN 동치성 — T=2,4 (fwd3s와 scan 청크 경계 전).
    for t in [2usize, 4] {
        let (md, n_cmp) = rawcuda::exl3_cuda_batch_probe::gdn_t_equivalence(
            &mut dec,
            dims,
            &consts,
            0,
            t,
            0x5111_0000_0000_0001,
        )?;
        rows.push_str(&format!("gdn T={t} last-row maxdiff={md:.3e}/{n_cmp} "));
        if md > 1e-3 {
            ok = false;
        }
    }
    // 어텐션 동치성 — T=2,4 (fwd3s 상한 8 이내).
    for t in [2usize, 4] {
        if t <= rawcuda::attn_cuda::ATTN_F3S_TMAX {
            let (md, n_cmp) = rawcuda::exl3_cuda_batch_probe::attn_t_equivalence(
                &mut dec,
                0,
                0,
                t,
                0x5111_0000_0000_0002,
            )?;
            rows.push_str(&format!("attn T={t} last-row maxdiff={md:.3e}/{n_cmp} "));
            if md > 1e-3 {
                ok = false;
            }
        }
    }
    let fwd = s11_forward_check(dir, nrows, lim)?;
    let _ = adims;
    Ok(format!(
        "device: {dev} | exl3-cuda-s11 T-equivalence (nrows={nrows}): {rows}\n{fwd}| {}",
        if ok {
            "EQUIVALENT | PASS"
        } else {
            "NOT EQUIVALENT | FAIL"
        }
    ))
}

// S11 forward 배선 동치성 — 모듈 단위가 맞아도 forward가 틀릴 수 있다.
// 두 디코더(순차/배치)에 같은 임베딩 열을 넣어 마지막 로짓을 비교한다.
fn s11_forward_check(dir: &str, nrows: usize, lim: usize) -> Result<String, String> {
    use rawcuda::exl3_cuda::Exl3CudaDecoder;
    let mut seq = Exl3CudaDecoder::load_slots(dir, lim, 512, 1)?;
    let mut bat = Exl3CudaDecoder::load_slots(dir, lim, 512, 1)?;
    let dev = bat.device_name().to_string();
    let tmax = rawcuda::attn_cuda::ATTN_F3S_TMAX;
    let mut rows = String::new();
    let mut ok = true;
    for t in [2usize, 4, tmax]
        .into_iter()
        .filter(|t| *t <= tmax && *t <= nrows.max(2))
    {
        let mut emb = vec![0f32; t * seq.hidden];
        for (i, v) in emb.iter_mut().enumerate() {
            *v = ((i % 89) as f32) * 0.02 - 0.5;
        }
        let (md, n_cmp) =
            rawcuda::exl3_cuda_batch_probe::forward_batch_equivalence(&mut seq, &mut bat, &emb)?;
        // argmax 일치까지 본다 — 절대 오차보다 "같은 토큰을 고르는가"가
        // 실사용 판정이다. GEMM2는 mma f16 누적이라 GEMV와 환원 순서가
        // 달라 T>1에서 미세하게 다르다(구조적으로 0이 아니다).
        let (arg_same, ht, bt) =
            rawcuda::exl3_cuda_batch_probe::forward_batch_argmax(&mut seq, &mut bat, &emb)?;
        // [판정 기준] argmax 일치가 게이트다. logit maxdiff 1.6e-2는 f16 mma
        // 누산 차이로 T>1에서 구조적으로 0이 될 수 없다(환원 순서가
        // GEMV와 다르다). 실사용 판정은 "같은 토큰을 고르는가"이며
        // 게이트 스크립트도 같은 기준이다. 참고용으로 절댓값도 함께 찍되
        // FAIL 판정에는 쓰지 않는다.
        rows.push_str(&format!(
            "fwd T={t} argmax_same={arg_same}({ht}vs{bt}) logit maxdiff={md:.3e} "
        ));
        if !arg_same {
            ok = false;
        }
        let _ = n_cmp;
    }
    let mut emb2 = vec![0f32; 2 * seq.hidden];
    for (i, v) in emb2.iter_mut().enumerate() {
        *v = ((i % 89) as f32) * 0.02 - 0.5;
    }
    let mid = rawcuda::exl3_cuda_batch_probe::batch_mid_probe(&mut seq, &mut bat, &emb2)?;
    // 첫 층 선형의 순차 gemv_host vs 배치 gemm2_host — T>1 GEMM이 같은 값을
    // 내는지(모듈 자체는 이미 검증됐지만 T=1↔T=2 교차 확인).
    let kq = seq.hidden;
    let mut xq = vec![0f32; 2 * kq];
    for (i, v) in xq.iter_mut().enumerate() {
        *v = ((i % 89) as f32) * 0.02 - 0.5;
    }
    let key = "model.language_model.layers.0.linear_attn.in_proj_qkv";
    let (gmd, gn) = rawcuda::exl3_cuda_batch_probe::batch_gemv_probe(&mut seq, &mut bat, key, &xq)?;
    Ok(format!(
        "device: {dev} | exl3-cuda-s11 forward-equivalence: {rows}\n{mid}\n\
         S11 batch gemm2({key} T=2): vs sequential maxdiff={gmd:.3e}/{gn}| {}",
        if ok { "PASS" } else { "FAIL" }
    ))
}

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
    let line =
        rawcuda::exl3_cuda_device_probe::cuda_s10_stream_check(&mut host, &mut dev, &toks, 6)?;
    // 진단 계기 — 결함 시 규위 좁히기용(정상 시 전부 0.000e0이어야 한다).
    let d_embed = rawcuda::exl3_cuda_device_probe::embed_resid_probe(&mut host, &mut dev, toks[0])?;
    let d_norm = rawcuda::exl3_cuda_device_probe::first_norm_probe(&mut host, &mut dev, toks[0])?;
    let d_mid = rawcuda::exl3_cuda_device_probe::mid_layer_probe(&mut host, &mut dev, toks[0])?;
    let d_attn = rawcuda::exl3_cuda_device_probe::attn_input_probe(&mut host, &mut dev)?;
    let d_step =
        rawcuda::exl3_cuda_device_probe::one_step_state_probe(&mut host, &mut dev, toks[0])?;
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
        let (md, _) =
            rawcuda::exl3_cuda_gemv_probe::single_gemv_compare(&mut host, &mut dev, key, &xg)?;
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
        _ => Err(format!("사용법: {prog} q4-dequant|q4-gemv|q4-gemm|q4-neg")),
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
    let gguf = std::env::args()
        .nth(2)
        .unwrap_or_else(|| rawcuda::fn_support::FN_GGUF_MAIN.to_string());
    let r = match sub {
        "" | "-check" => rawcuda::fn_gdn_cuda_probe::cuda_fn_gdn_check(&gguf),
        "-neg" => rawcuda::fn_gdn_cuda_probe::cuda_fn_gdn_negative_check(&gguf),
        _ => Err(format!(
            "사용법: fn-gdn [gguf_main] | fn-gdn-neg [gguf_main]"
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
