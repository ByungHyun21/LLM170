//! Q4 게이트 프롭(rawhip q4acc·micro-bench·row-check·rawhip-check·check) (plans/129 R2③ — probes/ 분리, arm 본문 무변경 이동).
use std::process::ExitCode;

use super::{arg_num, arg_str};

pub(super) fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let d_fn = crate::resource::DEFAULT_FN_MODEL;
    let d_27 = crate::resource::DEFAULT_27_MODEL;
    let d_q35 = crate::resource::DEFAULT_Q35_MODEL;
    let r: Result<String, String> = match cmd {
        "rawhip-check" => return Some(cmd_rawhip_check(args)),
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
        "gqa-bench" => llm170_backend_gpu::rawhip::gqa_bench(),
        "mm-tile" => llm170_backend_gpu::rawhip::mm_tile_bench(),
        "mm-bench" => llm170_backend_gpu::rawhip::mm_batch_bench(),
        "bw-test" => llm170_backend_gpu::rawhip::bw_test(),
        "dp4a-test" => llm170_backend_gpu::rawhip::dp4a_test(),
        "iq3s-probe" => llm170_backend_gpu::rawhip::iq3s_probe(),
        "qk-check" => llm170_backend_gpu::rawhip::qk_check(),
        _ => return None,
    };
    Some(super::finish(r))
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
