//! [R2] w4a16-gpu — GPU 순차 디코드 프로브 + 캐스트 계약 테스트.

use super::super::arg_str;

mod args;
use self::args::*;
mod bench;
use self::bench::*;
mod g4;
use self::g4::*;
mod generate;
use self::generate::*;

/// GPU 순차 디코드 — 64층 체인을 CUDA(호스트 스테이징)로 돌리고
/// 임베딩 행·최종 head만 CPU 참조 경로(골든과 동일 계급).
///   w4a16-gpu <dir> --prompt-tokens <ids> [--n-predict N] [--ctx N]
///              [--no-head] [--bench-gemm <lin> <t> <reps>] [--moe-check]
///              [--h2d-bench <MB>]
pub(super) fn gpu_run(args: &[String]) -> Result<String, String> {
    use llm170_backend_gpu::W4a16Dec;
    let dir = arg_str(args, 0, "");
    // [gemma4 P2-3 2026-10-14] 아치 분기 — gemma4는 전용 v0 체인 경로
    // (호스트 스테이징 + g4 커널군 — P2 토큰 게이트).
    if llm170_core::qwen35::bind::dir_arch(std::path::Path::new(&dir))
        .as_deref()
        .is_some_and(llm170_core::gemma4::bind::arch_ok)
    {
        return gpu_run_g4(args);
    }
    let a = parse_args(args)?;
    if a.bench.is_none() && !a.decode_selfcheck && a.prompts.is_empty() {
        return Err("w4a16-gpu: 프롬프트 필요(--prompt-tokens)".into());
    }
    let prompt: Vec<u32> = a.prompts.first().cloned().unwrap_or_default();
    if a.bench.is_none() && prompt.len() + a.n_predict + 1 > a.ctx {
        return Err(format!("ctx({}) too small for prompt+n_predict", a.ctx));
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    // [B20 보완 2026-10-19] 적재 완료 = 락 해제.
    crate::resource::release_load_lock();
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
    // 1~4) 모델 상주 업로드(GpuEngine 공용 — dense/MoE 분기 포함).
    // kv_reserve 0 = 종전 상주 규칙(프로브 — 서버 사다리 판정과 분리).
    crate::gpu_engine::upload_model(&mut dec, &model, a.ctx, 0, None)?;
    // 보고용 split 선형 수(dense 한정 — MoE는 0).
    let n_lin = model
        .engine_names()
        .iter()
        .filter(|n| {
            model
                .w_raw(n)
                .map(|w| w.ty == llm170_core::wtype::WType::W4a16Split)
                .unwrap_or(false)
        })
        .count();
    if a.h2d_mb > 0 {
        return run_h2d_bench(&mut dec, a.h2d_mb);
    }
    if a.bench_ew {
        return dec.bench_ew();
    }
    if a.ew_gelu_check {
        return run_ew_gelu_check(&mut dec);
    }
    if a.moe_pes_check {
        return run_moe_pes_check(&mut dec);
    }
    if a.prep_g4_check {
        return run_prep_g4_check(&mut dec);
    }
    if a.attn_g4_check {
        return run_attn_g4_check(&mut dec);
    }
    if a.plain_check {
        return dec.plain_gemm_selfcheck();
    }
    if a.mma_smoke {
        return dec.mma_smoke();
    }
    if a.moe_check {
        return dec.moe_selfcheck();
    }
    let upload_ms = t0.elapsed().as_secs_f64() * 1e3;
    if let Some((t, n, k)) = a.moe_topk_check {
        return dec.moe_topk_check(t, n, k);
    }
    if let Some((name, t)) = &a.mma_diff {
        return run_mma_diff(&mut dec, name, *t);
    }
    if let Some((name, t)) = &a.marlin_diff {
        return run_marlin_diff(&mut dec, &model, name, *t);
    }
    if let Some(x) = a.marlin_selfcheck {
        return run_marlin_selfcheck(&mut dec, x);
    }
    if let Some(x) = a.marlin_selfcheck32 {
        return run_marlin_selfcheck32(&mut dec, x);
    }
    if a.decode_selfcheck {
        return dec.decode_selfcheck();
    }
    if !a.decode_bench.is_empty() {
        return run_decode_bench(&mut dec, &a.decode_bench, a.decode_cold);
    }
    if !a.attn_fa2_bench.is_empty() {
        return run_attn_fa2_bench(&mut dec, &a.attn_fa2_bench);
    }
    if !a.attn_fa2.is_empty() {
        return run_attn_fa2(&mut dec, &a.attn_fa2);
    }
    if let Some((name, t)) = &a.marlin_ref {
        return run_marlin_ref(&mut dec, &model, name, *t, a.mode_ref, a.kcap);
    }
    if !a.marlin_bench.is_empty() {
        return run_marlin_bench(&mut dec, &model, &a.marlin_bench, a.marlin_cold);
    }
    if let Some((name, bt, reps)) = &a.bench_plain {
        return run_bench_plain(&mut dec, &model, name, *bt, *reps);
    }
    if let Some((name, t, reps)) = &a.bench_gemv_t {
        return run_bench_gemv_t(&mut dec, name, *t, *reps);
    }
    if let Some((name, reps)) = &a.bench_gemv {
        if name == "ALL" {
            return run_bench_gemv_all(&mut dec, *reps);
        }
        return run_bench_gemv(&mut dec, &model, name, *reps);
    }
    if let Some((name, bt, reps)) = &a.bench {
        return run_bench_gemm(&mut dec, &model, name, *bt, *reps);
    }
    // [검수 통합 2026-10-14] 다중 프롬프트 = 시나리오 순차(모델 로드 1회) —
    // 골든·경계·장문 판정을 한 런에. 시나리오 사이 슬롯 리셋(= reset_seq 동형:
    // kv_release + reset_state). 출력: 시나리오별 `s{i}-tokens:` 줄.
    if a.prompts.len() > 1 {
        let mut rep = String::new();
        for (i, p) in a.prompts.iter().enumerate() {
            if p.len() + a.n_predict + 1 > a.ctx {
                return Err(format!("ctx({}) too small for scenario{i}", a.ctx));
            }
            if i > 0 {
                dec.kv_release(0);
                dec.reset_state(0)?;
            }
            let r = run_generate(
                &mut dec,
                &model,
                p,
                a.no_head,
                a.n_predict,
                a.spec_k,
                a.spec_check,
                upload_ms,
                &dir,
                n_lin,
                i > 0, // head 등록 1회(이후 재사용 — 재양자화/재할당 방지)
            )?;
            rep.push_str(&format!("# scenario{i} (프롬프트 {}토큰)\n", p.len()));
            rep.push_str(&r.replace("\n tokens: ", &format!("\n s{i}-tokens: ")));
            rep.push('\n');
        }
        return Ok(rep);
    }
    run_generate(
        &mut dec,
        &model,
        &prompt,
        a.no_head,
        a.n_predict,
        a.spec_k,
        a.spec_check,
        upload_ms,
        &dir,
        n_lin,
        false,
    )
}

/// [gemma4 P2-3 2026-10-14] gemma4 GPU v0 체인 프로브 — 업로드 + 토큰별 체인 +
/// **CPU 오라클(RefEngine) 토큰 대조**(단일 실행 판정 — P2 게이트).
///   llm170 w4a16-gpu <gemma4 dir> --prompt-tokens <ids> [--n-predict N] [--ctx N]
pub(super) fn gpu_run_g4(args: &[String]) -> Result<String, String> {
    use llm170_backend_gpu::W4a16Dec;
    let dir = arg_str(args, 0, "");
    let a = parse_args(args)?;
    if a.prompts.is_empty() {
        return Err("w4a16-gpu(gemma4): --prompt-tokens 필요".into());
    }
    let prompt: Vec<u32> = a.prompts.first().cloned().unwrap_or_default();
    if prompt.is_empty() {
        return Err("w4a16-gpu(gemma4): 빈 프롬프트".into());
    }
    if prompt.len() + a.n_predict + 1 > a.ctx {
        return Err(format!(
            "ctx({}) too small for prompt({})+n_predict({})",
            a.ctx,
            prompt.len(),
            a.n_predict
        ));
    }
    let model =
        llm170_core::gemma4::Model::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    // [B20 보완 2026-10-19] 적재 완료 = 락 해제.
    crate::resource::release_load_lock();
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
    let cfg = crate::gpu_engine::upload_model_g4(&mut dec, &model, a.ctx)?;
    // [P9-2] 형상별 t행 GEMV 벤치 게이트 — 업로드 직후·체인 전 단발 측정.
    if let Some((name, t, reps)) = &a.bench_gemv_t {
        return run_bench_gemv_t(&mut dec, name, *t, *reps);
    }
    let upload_ms = t0.elapsed().as_secs_f64() * 1e3;
    // GPU 체인 — 프롬프트 전파 + 그리디 생성.
    let tg = std::time::Instant::now();
    let (got, gpu_logits) = g4_gpu_chain(&mut dec, &model, &cfg, &prompt, a.n_predict)?;
    let gpu_ms = tg.elapsed().as_secs_f64() * 1e3;
    let n_tok = prompt.len() + a.n_predict;
    // CPU 오라클 — 같은 프롬프트·같은 그리디(RefEngine) + 스텝별 로짓 수집.
    let tr = std::time::Instant::now();
    let mut eng = super::ref_cmd::RefEngine::load(&dir, 1, a.ctx)?;
    let mut cl = eng.prefill(0, &prompt)?;
    let mut cpu_logits: Vec<Vec<f32>> = vec![cl.clone()];
    let mut want = vec![llm170_core::qwen35::greedy(&cl)];
    for _ in 1..a.n_predict {
        let t = *want.last().unwrap();
        let mut ls = eng.decode(0, &[t])?;
        cl = ls.pop().ok_or("decode logits 부재")?;
        cpu_logits.push(cl.clone());
        want.push(llm170_core::qwen35::greedy(&cl));
    }
    let ref_ms = tr.elapsed().as_secs_f64() * 1e3;
    let same = got == want;
    let _n_eq = got.iter().zip(want.iter()).filter(|(a, b)| a == b).count();
    let csv = |v: &[u32]| {
        v.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    // 스텝별 로짓 top-5(최초 3스텝) — 발산 정량 진단 + 마진 인지 판정.
    // [gemma4 P2-3] 이 체인의 수치 클래스(int8 KV·int8 head — 단일 경로 계약)는
    // CPU 오라클 f32와 ±θ만큼 다르다. θ = 스텝별 공통 상위 토큰의 |Δ| 최대
    // (자기 교정 측정). CPU top-2 마진 < θ인 위치의 토큰 플립은 근접동률(허용),
    // 마진 ≥ θ인 플립은 실결함(불일치).
    let mut diag = String::new();
    // [판정 보정 2026-10-15] 실결함 계수는 **첫 발산 위치만** 유효 — 발산 후
    // 스텝의 margin/θ는 문맥이 달라 비교 불가(아래 verdict 주석의 의도와 동기).
    // 종전 구현은 이후 스텝까지 margin≥θ를 세어 니어타이 케이스를 실결함으로
    // 오분류했다(실측: 9토큰 절단 프롬프트 — 첫 발산 마진 0.369 < θ 0.830인데
    // step1 문맥 상이 비교 15.363≥15.238로 실결함 오표기).
    let mut theta_max = 0f32;
    let n_step = gpu_logits.len().min(cpu_logits.len()).min(got.len());
    for i in 0..n_step {
        let cw =
            cpu_logits[i]
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (j, &v)| {
                    if v > a.1 { (j, v) } else { a }
                });
        // 스텝 편차 클래스 θ_i = CPU 상위 5토큰의 |Δ| 최대(자기 교정).
        // 발산 후 스텝은 문맥이 달라 비교 불가 — θ_class는 첫 발산 이전만.
        let mut idx: Vec<usize> = (0..cpu_logits[i].len()).collect();
        idx.sort_by(|&a, &b| cpu_logits[i][b].total_cmp(&cpu_logits[i][a]));
        let theta = idx
            .iter()
            .take(5)
            .map(|&j| (gpu_logits[i][j] - cpu_logits[i][j]).abs())
            .fold(0f32, f32::max);
        // 첫 발산 전까지만 클래스 누적(이후 문맥 상이).
        if got[..i.min(got.len())]
            .iter()
            .zip(&want[..i.min(want.len())])
            .all(|(a, b)| a == b)
        {
            theta_max = theta_max.max(theta);
        }
        let margin = cw.1
            - cpu_logits[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != cw.0)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
        if i < 3 {
            let gt = g4_top5(&gpu_logits[i]);
            let ct = g4_top5(&cpu_logits[i]);
            let grank = gpu_logits[i].iter().filter(|&&v| v > cw.1).count();
            diag.push_str(&format!(
                "  step{i} GPU top5: {} | CPU top5: {} | CPU선택 {}({:.3}) GPU순위 {} GPU값 {:.3} | 마진 {:.3} θ {:.3}\n",
                gt.iter().map(|(t, v)| format!("{t}:{v:.2}")).collect::<Vec<_>>().join(" "),
                ct.iter().map(|(t, v)| format!("{t}:{v:.2}")).collect::<Vec<_>>().join(" "),
                cw.0,
                cw.1,
                grank + 1,
                gpu_logits[i][cw.0],
                margin,
                theta,
            ));
        }
    }
    let lg = gpu_logits.last().cloned().unwrap_or_default();
    let (mx, am) = lg
        .iter()
        .enumerate()
        .fold((f32::NEG_INFINITY, 0usize), |a, (i, &v)| {
            if v > a.0 { (v, i) } else { a }
        });
    // 판정 = **첫 발산 위치만** — 이후는 문맥 상이(비교 불가). 마진 < θ_class면
    // 근접동률 플립(수치 클래스 내 — 허용), 이상이면 실결함.
    let (mut margin_first, mut theta_first) = (f32::NAN, f32::NAN);
    if let Some(k) = got.iter().zip(want.iter()).position(|(a, b)| a != b)
        && k < cpu_logits.len()
    {
        let cw0 =
            cpu_logits[k]
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (j, &v)| {
                    if v > a.1 { (j, v) } else { a }
                });
        margin_first = cw0.1
            - cpu_logits[k]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != cw0.0)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
        let mut idx: Vec<usize> = (0..cpu_logits[k].len()).collect();
        idx.sort_by(|&a, &b| cpu_logits[k][b].total_cmp(&cpu_logits[k][a]));
        theta_first = idx
            .iter()
            .take(5)
            .map(|&j| (gpu_logits[k][j] - cpu_logits[k][j]).abs())
            .fold(0f32, f32::max);
    }
    let moved = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    let verdict = if same {
        format!("일치 — P2 토큰 게이트 통과 (편차 클래스 θmax {theta_max:.3})")
    } else if margin_first < theta_first {
        format!(
            "근접동률 플립(허용) — 최초 발산 위치 {} · CPU 마진 {margin_first:.3} < 편차 클래스 θ {theta_first:.3} · 이전 스텝 θmax {theta_max:.3}",
            moved.unwrap_or(0)
        )
    } else {
        format!(
            "실결함 — 최초 발산 위치 {} · CPU 마진 {margin_first:.3} ≥ 편차 클래스 θ {theta_first:.3} (이전 스텝 θmax {theta_max:.3})",
            moved.unwrap_or(0)
        )
    };
    Ok(format!(
        "w4a16-gpu(gemma4 v0) {dir} — 호스트 스테이징 체인 + GPU head · qk_scale 1.0 · softcap {}\n  업로드 {upload_ms:.0}ms · GPU 체인 {n_tok}토큰 {gpu_ms:.0}ms ({:.0}ms/토큰) · CPU 오라클 {ref_ms:.0}ms\n  GPU tokens: {}\n  CPU tokens: {}\n{diag}  로짓 max {mx:.4} · argmax {am}\n  판정: {verdict}",
        hp.softcap,
        gpu_ms / n_tok as f64,
        csv(&got),
        csv(&want),
    ))
}

#[cfg(test)]
mod cast_contract {
    //! G3 — f16 캐스트 트윈 전수 대조(CPU 전용, GPU 불필요).
    //! 4중 수동 복사 중 호스트 3본(gptq4::h2f · w4a16_dec::f32_to_f16 ·
    //! core deq)을 65,536 패턴으로 묶는다. 커널(.cu f2h/h2f)은 골든 게이트 소관.
    use llm170_core::quant::half_to_f32;

    /// h2f(backend-gpu) ≡ core half_to_f32 — u16 전 패턴.
    #[test]
    fn h2f_matches_core_all_patterns() {
        for h in 0..=u16::MAX {
            assert_eq!(
                llm170_backend_gpu::h2f(h).to_bits(),
                half_to_f32(h).to_bits(),
                "h2f({h:#06x})"
            );
        }
    }

    /// f32_to_f16 왕복 항등 — f16 전 패턴(NaN 제외). core half_to_f32로
    /// 올린 뒤 되돌리면 원 비트(서브노멀·±inf 포함).
    #[test]
    fn f32_to_f16_roundtrips_all_non_nan_patterns() {
        for h in 0..=u16::MAX {
            let exp = (h >> 10) & 0x1F;
            let man = h & 0x3FF;
            if exp == 0x1F && man != 0 {
                continue; // NaN — 페이로드 정규화(0x200)라 항등 아님(별도 검사).
            }
            let v = half_to_f32(h);
            assert_eq!(llm170_backend_gpu::f32_to_f16(v), h, "roundtrip {h:#06x}");
        }
    }

    /// 경계·반올림(RN-even)·NaN 규약.
    #[test]
    fn f32_to_f16_edges() {
        let f = llm170_backend_gpu::f32_to_f16;
        assert_eq!(f(1.0), 0x3C00);
        assert_eq!(f(-2.0), 0xC000);
        assert_eq!(f(0.0), 0x0000);
        assert_eq!(f(-0.0), 0x8000);
        assert_eq!(f(65504.0), 0x7BFF); // f16 최대 정규수
        assert_eq!(f(65520.0), 0x7C00); // 오버플로 타이 → +inf
        assert_eq!(f(f32::INFINITY), 0x7C00);
        assert_eq!(f(f32::NEG_INFINITY), 0xFC00);
        assert_eq!(f(f32::NAN), 0x7E00); // qNaN 정규화
        assert_eq!(f(1e-8), 0x0000); // e < -10 → 0
        assert_eq!(f(5.960_464_5e-8), 0x0001); // 2^-24 = 최소 서브노멀
        // RN-even 타이: 2049.0(2048+1, ulp=2의 반) → 짝수 가수 2048.
        assert_eq!(f(2049.0), f(2048.0));
        assert_eq!(f(2050.0), 0x6801); // 정확값(반올림 없음)
    }
}
