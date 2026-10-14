//! [R2] w4a16-gpu — GPU 순차 디코드 프로브 + 캐스트 계약 테스트.

use super::super::arg_str;

mod args;
use self::args::*;
mod g4;
use self::g4::*;

fn run_h2d_bench(dec: &mut llm170_backend_gpu::W4a16Dec, h2d_mb: usize) -> Result<String, String> {
    let bytes = h2d_mb << 20;
    let src = vec![0x5Au8; bytes];
    let dst = dec.alloc_scratch(bytes)?;
    let pinned = dec.alloc_pinned_scratch(bytes)?;
    // SAFETY: pinned은 alloc_pinned_scratch가 돌려준 유효 핸들, bytes는 할당 크기.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), pinned as *mut u8, bytes);
    }
    let t = std::time::Instant::now();
    dec.h2d_scratch(dst, &src)?;
    dec.sync_bench()?;
    let pageable_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    // SAFETY: 핀드 버퍼 슬라이스(할당 크기).
    let ps = unsafe { std::slice::from_raw_parts(pinned as *const u8, bytes) };
    dec.h2d_scratch(dst, ps)?;
    dec.sync_bench()?;
    let pinned_ms = t.elapsed().as_secs_f64() * 1e3;
    dec.free_pinned_scratch(pinned)?;
    dec.free_scratch(dst)?;
    Ok(format!(
        "h2d-bench {h2d_mb}MiB: 페이지러블 {pageable_ms:.0}ms ({:.1}GB/s) · 핀드 {pinned_ms:.0}ms ({:.1}GB/s)",
        bytes as f64 * 1e-9 / (pageable_ms * 1e-3),
        bytes as f64 * 1e-9 / (pinned_ms * 1e-3).max(1e-9)
    ))
}

/// --mma-diff — mma 커널 정오 대조.
fn run_mma_diff(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    name: &str,
    t: usize,
) -> Result<String, String> {
    dec.mma_diff_check(name, t)
}

/// [marlin S2 2026-10-12] marlin B 배치 GEMM A/B — 원본 q를 core::marlin
/// (호스트 재패킹)으로 변환해 신규 커널 vs 기존 mma 커널 판정. g128·f16 전용.
fn run_marlin_diff(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    t: usize,
) -> Result<String, String> {
    let w = model
        .w_raw(name)
        .ok_or_else(|| format!("--marlin-diff {name}: 무게 없음"))?;
    if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
        return Err(format!(
            "--marlin-diff {name}: g128·f16 split 전용(ty={:?} group={} bf16={})",
            w.ty, w.group, w.scale_bf16
        ));
    }
    let q: Vec<u32> = w
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let qm = llm170_core::marlin::marlin_repack_q(&q, w.n_out as usize, w.n_in as usize)?;
    dec.marlin_diff_check(name, t, &qm)
}

/// [marlin S2 디버그] 커널 격리 자체점검 — 합성 n=128·k=128·s=1·x=1.
/// 기대 = 정수 Σ(nib−8) (f32 누적 순서 무관 비트정확).
fn run_marlin_selfcheck(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    (t, k, n, m16): (usize, usize, usize, bool),
) -> Result<String, String> {
    let nib_at = |kk: usize, nn: usize| ((kk * 7 + nn * 3) % 16) as u32;
    let mut q = vec![0u32; n * (k / 8)];
    for nn in 0..n {
        for kk in 0..k {
            q[nn * (k / 8) + kk / 8] |= nib_at(kk, nn) << ((kk % 8) * 4);
        }
    }
    // 스케일·x = 2의 거듭제곱(정확 f16) — 지수 오배치 시 기대가 어긋난다.
    let mut s = vec![0u16; n * (k / 128)];
    for nn in 0..n {
        for gg in 0..(k / 128) {
            s[nn * (k / 128) + gg] = ((15 - ((nn + gg) % 5)) as u16) << 10;
        }
    }
    let x16: Vec<u16> = (0..t * k)
        .map(|i| ((15 - ((i / k + (i % k) / 8) % 4)) as u16) << 10)
        .collect();
    let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
    let sf = |h: u16| -> f64 {
        let e = ((h >> 10) & 0x1F) as i32;
        let m = (h & 0x3FF) as f64 / 1024.0;
        (1.0 + m) * 2f64.powi(e - 15)
    };
    let mut expect = vec![0f32; t * n];
    for r in 0..t {
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let sc = sf(s[nn * (k / 128) + kk / 128]);
                let xv = sf(x16[r * k + kk]);
                acc += (nib_at(kk, nn) as f64 - 8.0) * sc * xv;
            }
            expect[r * n + nn] = acc as f32;
        }
    }
    let (sym, bm, bn, thr) = if m16 {
        ("w4a16_gemm_g128_marlin_m16", 16usize, 256usize, 128u32)
    } else {
        ("w4a16_gemm_g128_marlin", 64, 128, 256)
    };
    dec.marlin_selfcheck(sym, bm, bn, thr, &qm, &s, &x16, &expect, t, n, k)
}

/// [marlin g32] 합성 자체점검 — bf16 2^-j 스케일(KC=32=그룹 1개), x=1.
fn run_marlin_selfcheck32(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    (t, k, n): (usize, usize, usize),
) -> Result<String, String> {
    let nib_at = |kk: usize, nn: usize| ((kk * 7 + nn * 3) % 16) as u32;
    let mut q = vec![0u32; n * (k / 8)];
    for nn in 0..n {
        for kk in 0..k {
            q[nn * (k / 8) + kk / 8] |= nib_at(kk, nn) << ((kk % 8) * 4);
        }
    }
    let mut s = vec![0u16; n * (k / 32)];
    for nn in 0..n {
        for gg in 0..(k / 32) {
            s[nn * (k / 32) + gg] = ((127 - ((nn + gg) % 5)) as u16) << 7;
        }
    }
    let x16 = vec![0x3C00u16; t * k]; // f16 1.0
    let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
    let mut expect = vec![0f32; t * n];
    for r in 0..t {
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let j = ((nn + kk / 32) % 5) as i32;
                acc += (nib_at(kk, nn) as f64 - 8.0) * 2f64.powi(-j);
            }
            expect[r * n + nn] = acc as f32;
        }
    }
    dec.marlin_selfcheck(
        "w4a16_gemm_g32_marlin",
        64,
        128,
        256,
        &qm,
        &s,
        &x16,
        &expect,
        t,
        n,
        k,
    )
}

/// [FA2] 프리필 어텐션 벤치 — prep 포함 기존 vs FA2.
fn run_attn_fa2_bench(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    bench: &[(usize, usize, usize, u32)],
) -> Result<String, String> {
    let mut out = Vec::new();
    for (t, lay, reps, pos0) in bench {
        out.push(dec.attn_fa2_bench(0, *lay, *t, *reps, *pos0)?);
    }
    Ok(out.join("\n"))
}

/// [FA2] 프리필 어텐션 A/B — 기존 경로 vs attn_fa2_qg.
fn run_attn_fa2(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    diffs: &[(usize, usize, u32)],
) -> Result<String, String> {
    let mut out = Vec::new();
    for (t, lay, p0) in diffs {
        out.push(dec.attn_fa2_diff(0, *lay, *t, *p0)?);
    }
    Ok(out.join("\n"))
}

/// [marlin S2 디버그] 실무게 64행 슬라이스 vs 호스트 기대값 — 커널 단독
/// 정오 판정(스케일·디퀀트 포함). 실패 시 (행,열,기대,실제)를 보여준다.
#[allow(clippy::too_many_arguments)]
fn run_marlin_ref(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    t: usize,
    mode: u8,
    kcap: usize,
) -> Result<String, String> {
    let w = model
        .w_raw(name)
        .ok_or_else(|| format!("--marlin-ref {name}: 무게 없음"))?;
    if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
        return Err(format!("--marlin-ref {name}: g128·f16 split 전용"));
    }
    let (n, k) = (64usize, kcap.min(w.n_in as usize));
    let q: Vec<u32> = w.data[..n * (k / 8) * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let s16: Vec<u16> = w.aux.ok_or("aux 부재")?[..n * (k / 128) * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
    let x16: Vec<u16> = (0..t * k)
        .map(|i| {
            let v = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
            llm170_backend_gpu::f32_to_f16(v)
        })
        .collect();
    // 모드: 0=실스케일·실x · 1=s→1 · 2=x→1 · 3=둘 다 1(가중치만).
    let mut s_use = s16.clone();
    let mut x_use = x16.clone();
    if mode == 1 || mode == 3 {
        s_use.fill(0x3C00);
    }
    if mode == 2 || mode == 3 {
        x_use.fill(0x3C00);
    }
    // 기대값 — 디퀀트 후 f16(커널과 동일 RNE), 누적은 f64(순서 무관).
    let mut expect = vec![0f32; t * n];
    for r in 0..t {
        for nn in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let nib = ((q[nn * (k / 8) + kk / 8] >> ((kk % 8) * 4)) & 0xF) as i32 - 8;
                let sc = llm170_core::quant::half_to_f32(s_use[nn * (k / 128) + kk / 128]);
                let wf = llm170_backend_gpu::h2f(llm170_backend_gpu::f32_to_f16(nib as f32 * sc));
                acc += (wf as f64) * (llm170_backend_gpu::h2f(x_use[r * k + kk]) as f64);
            }
            expect[r * n + nn] = acc as f32;
        }
    }
    dec.marlin_selfcheck(
        "w4a16_gemm_g128_marlin",
        64,
        128,
        256,
        &qm,
        &s_use,
        &x_use,
        &expect,
        t,
        n,
        k,
    )
}

/// [marlin S2] mma 대비 커널 벤치 — 같은 입력·grid·회차(인터리브).
fn run_marlin_bench(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    bench: &[(String, usize, usize, String)],
    cold: bool,
) -> Result<String, String> {
    let mut out = Vec::new();
    for (name, t, reps, kern) in bench {
        let Some(w) = model.w_raw(name) else {
            out.push(format!("marlin-bench {name}: 무게 없음(건너뜀)"));
            continue;
        };
        if w.ty != llm170_core::wtype::WType::W4a16Split || w.group != 128 || w.scale_bf16 {
            out.push(format!(
                "marlin-bench {name}: g128·f16 split 아님(건너뜀: ty={:?} group={} bf16={})",
                w.ty, w.group, w.scale_bf16
            ));
            continue;
        }
        let q: Vec<u32> = w
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let qm = llm170_core::marlin::marlin_repack_q(&q, w.n_out as usize, w.n_in as usize)?;
        let (msym, mbm, mbn, mthr, vs_gemv) = match kern.as_str() {
            // [G4 2026-10-12] m16v = m16 마린 vs 행-GEMV(t=1) — 재개 조건 (a) 판정.
            "m16v" => (
                "w4a16_gemm_g128_marlin_m16",
                16usize,
                256usize,
                128u32,
                true,
            ),
            "m16" => (
                "w4a16_gemm_g128_marlin_m16",
                16usize,
                256usize,
                128u32,
                false,
            ),
            _ => ("w4a16_gemm_g128_marlin", 64, 128, 256, false),
        };
        out.push(dec.marlin_bench(name, *t, *reps, &qm, msym, mbm, mbn, mthr, vs_gemv, cold)?);
    }
    Ok(out.join("\n"))
}

/// 마이크로벤치(진단): 플레인 GEMM(t) 반복 — v1/v3 vs mma.
fn run_bench_plain(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    bt: usize,
    reps: usize,
) -> Result<String, String> {
    let (n, k) = model
        .w_raw(name)
        .map(|w| (w.n_out as usize, w.n_in as usize))
        .ok_or_else(|| format!("--bench-plain {name}: 무게 없음"))?;
    let xf: Vec<f32> = vec![1.0f32; bt * k];
    // SAFETY: xf는 f32 슬라이스 — 바이트 재해석 읽기(할당 범위 내).
    let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
    let dx = dec.alloc_scratch(xf.len() * 4)?;
    dec.h2d_scratch(dx, xb)?;
    let dy = dec.alloc_scratch(bt * n * 4)?;
    dec.plain_bench_launch(name, dx, dy, bt)?;
    dec.sync_bench()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dec.plain_bench_launch(name, dx, dy, bt)?;
    }
    dec.sync_bench()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let flop = 2.0 * (n as f64) * (k as f64) * (bt as f64);
    dec.free_scratch(dx)?;
    dec.free_scratch(dy)?;
    Ok(format!(
        "bench-plain {name} n={n} k={k} t={bt}: {ms:.3} ms/회 · {:.2} TF",
        flop * 1e-9 / (ms * 1e-3)
    ))
}

/// 마이크로벤치(진단): t행 GEMV(배치 디코드 커널) 반복 — A9 판정.
fn run_bench_gemv_t(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    name: &str,
    t: usize,
    reps: usize,
) -> Result<String, String> {
    let ms = dec.bench_gemv_t(name, t, reps)?;
    Ok(format!("bench-gemv-t {name} t={t}: {ms:.3} ms/회"))
}

/// 마이크로벤치(진단): 전 선형 t=1 GEMV 워크 — 실효 가중치 대역.
fn run_bench_gemv_all(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    reps: usize,
) -> Result<String, String> {
    let (ms, wb) = dec.gemv_walk_bench(reps)?;
    let gbs = wb as f64 * 1e-9 / (ms * 1e-3);
    Ok(format!(
        "bench-gemv-all: {ms:.2} ms/회 · 가중치 {gbs:.0} GB/s ({:.2} GB)",
        wb as f64 / 1e9
    ))
}

/// 마이크로벤치(진단): 지정 선형의 t=1 GEMV 반복 — 실효 가중치 대역.
fn run_bench_gemv(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    reps: usize,
) -> Result<String, String> {
    let (n, k) = model
        .w_raw(name)
        .map(|w| (w.n_out as usize, w.n_in as usize))
        .ok_or_else(|| format!("--bench-gemv {name}: 무게 없음"))?;
    let xf: Vec<f32> = vec![1.0f32; k];
    // SAFETY: xf는 f32 슬라이스 — 바이트 재해석 읽기(할당 범위 내).
    let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
    let dx = dec.alloc_scratch(xf.len() * 4)?;
    dec.h2d_scratch(dx, xb)?;
    let dy = dec.alloc_scratch(n * 4)?;
    dec.gemv_bench_launch(name, dx, dy)?;
    dec.sync_bench()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dec.gemv_bench_launch(name, dx, dy)?;
    }
    dec.sync_bench()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let wb = if model.w_raw(name).map(|w| w.ty) == Some(llm170_core::wtype::WType::W4a16Split) {
        (n * k) as f64 / 2.0
    } else {
        (n * k) as f64 * 2.0
    };
    let gbs = wb * 1e-9 / (ms * 1e-3);
    dec.free_scratch(dx)?;
    dec.free_scratch(dy)?;
    Ok(format!(
        "bench-gemv {name} n={n} k={k}: {ms:.3} ms/회 · 가중치 {gbs:.0} GB/s"
    ))
}

/// 마이크로벤치(진단): 지정 선형의 t≥2 GEMM 반복 시간·실효 GB/s.
fn run_bench_gemm(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    name: &str,
    bt: usize,
    reps: usize,
) -> Result<String, String> {
    let (n, k) = model
        .w_raw(name)
        .map(|w| (w.n_out as usize, w.n_in as usize))
        .ok_or_else(|| format!("--bench-gemm {name}: 무게 없음"))?;
    let xf: Vec<f32> = vec![1.0f32; bt * k]; // f32 1.0 — 수치 무의미
    // SAFETY: xf는 f32 슬라이스 — 바이트 재해석 읽기(할당 범위 내).
    let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
    let dxh = dec.alloc_scratch(xf.len() * 4)?;
    dec.h2d_scratch(dxh, xb)?;
    let dout = dec.alloc_scratch(bt * n * 4)?;
    // 워밍 1회.
    dec.gemm_bench_launch(name, dxh, dout, bt)?;
    dec.sync_bench()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dec.gemm_bench_launch(name, dxh, dout, bt)?;
    }
    dec.sync_bench()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let wb = (n * k) as f64 / 2.0; // 4bit+scale ≈ 0.5625B/원소 근사
    let gbs = wb * 1e-9 / (ms * 1e-3);
    dec.free_scratch(dxh)?;
    dec.free_scratch(dout)?;
    Ok(format!(
        "bench-gemm {name} n={n} k={k} t={bt} reps={reps}: {ms:.2} ms/회 · 가중치 {gbs:.0} GB/s"
    ))
}

/// 5) 프롬프트 순차 prefill + greedy 생성(head는 CPU 참조 경로) + [A-1]
///    스페큘러티브 디코딩 — [R9] 순수 이동(gpu_run 후반부).
#[allow(clippy::too_many_arguments)]
fn run_generate(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    prompt: &[u32],
    no_head: bool,
    n_predict: usize,
    spec_k: usize,
    spec_check: bool,
    upload_ms: f64,
    dir: &str,
    n_lin: usize,
    head_ready: bool,
) -> Result<String, String> {
    let hp = model.hp.clone();
    let head = model
        .w("output.weight")
        .ok_or_else(|| "output.weight 부재".to_string())?;
    let head_logits = |xn: &[f32]| -> Vec<f32> {
        if no_head {
            // head 비용 분리 계측 전용 — 체인 시간만 재기 위한 가짜 로짓.
            return xn[..5120].to_vec();
        }
        let mut lg = vec![0.0f32; head.n_out as usize];
        llm170_core::matmul::matmul(xn, &head, &mut lg);
        lg
    };
    // GPU head — bf16 output.weight를 상주 업로드(--no-head는 CPU 참조).
    // [검수 통합 2026-10-14] head_ready(다중 시나리오 2회차+)면 재업로드 생략.
    let mut head_gpu = head_ready;
    if !no_head && !head_ready && head.ty == llm170_core::wtype::WType::Bf16 {
        dec.upload_head(head.data, head.n_out as usize, head.n_in as usize)?;
        head_gpu = true;
    }
    let t1 = std::time::Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut next = 0u32;
    // 프롬프트는 t≤8 배치 청크로 — 골든 판정이 배치 경로(GEMM t≥2)를 지난다.
    {
        let h = hp.n_embd;
        let mut i = 0usize;
        while i < prompt.len() {
            // [2026-10-09] MoE 상주는 32(기본), 스트리밍은 1. [P10] 상한 128 —
            // LLM170_PREFILL_T로 실측 오버라이드(엔진 prefill과 동일 규칙).
            let cap = if hp.n_experts > 0 && !dec.moe_experts_resident() {
                1
            } else {
                llm170_diag::flag::val("LLM170_PREFILL_T")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(512) // 청크 확대
                    .clamp(1, 512)
            };
            let t = (prompt.len() - i).min(cap);
            let mut rows: Vec<f32> = Vec::with_capacity(t * h);
            for &tok in &prompt[i..i + t] {
                rows.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
            }
            let last = i + t == prompt.len();
            if head_gpu {
                let lg = dec.forward_prefill(0, &rows, t, last)?;
                if last {
                    next = llm170_core::matmul::greedy_from(&lg);
                }
            } else {
                let xn = dec.forward_prefill(0, &rows, t, false)?;
                if last {
                    next = llm170_core::matmul::greedy_from(&head_logits(&xn));
                }
            }
            i += t;
        }
    }
    out.push(next);
    if llm170_diag::flag::ne0("LLM170_TIME") {
        eprintln!("{}", dec.prof_report("prefill")?);
    }
    // [A-1] 스페큘러티브 디코딩 — n-gram 초안 + 배치 검증 + GDN 롤백.
    // 출력 토큰 스트림은 비스펙(그리디)과 **정확히 동일**해야 한다(골든 판정).
    let mut hist: Vec<u32> = prompt.to_vec();
    if spec_k >= 2 && head_gpu {
        dec.enable_spec()?;
    }
    if spec_check && head_gpu {
        // [A-1 진단] 검증(t=4 배치) vs t=1 디코드의 토큰 대조.
        // S0 저장 → t=1 4회(T1..T4) → S0 복원+pos 되감기 → 검증 [next,T1,T2,T3].
        dec.enable_spec()?;
        dec.spec_save_state(0)?;
        let mut t1_4: Vec<u32> = Vec::new();
        {
            let mut cur = next;
            for _ in 0..4 {
                let row = model.embed_row(cur).map_err(|e| e.to_string())?;
                cur = dec.forward_device_argmax(0, &row)?;
                t1_4.push(cur);
            }
        }
        // trio(4×t=1)의 최종 상태 저장 — spec scan 대조의 정답 참조.
        let s_trio = dec.spec_dump_state(0)?;
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        let mut rows: Vec<f32> = Vec::with_capacity(4 * hp.n_embd);
        rows.extend_from_slice(&model.embed_row(next).map_err(|e| e.to_string())?);
        for &tk in &t1_4[..3] {
            rows.extend_from_slice(&model.embed_row(tk).map_err(|e| e.to_string())?);
        }
        let toks = dec.spec_verify(0, &rows, 4)?;
        eprintln!("[spec-check] t1={t1_4:?} verify={toks:?}");
        // o_lc 대조 — spec scan on vs off(같은 입력·같은 상태).
        let nv = 4 * 32 * 128; // h_v=32 고정 가정(35B)
        let a = dec.spec_dump_outv(nv)?;
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        dec.spec_set_scan(false);
        let _ = dec.spec_verify(0, &rows, 4)?;
        let b = dec.spec_dump_outv(nv)?;
        dec.spec_set_scan(true);
        let mut mx = 0.0f32;
        let mut mi = 0usize;
        for i in 0..nv {
            let d = (a[i] - b[i]).abs();
            if d > mx {
                mx = d;
                mi = i;
            }
        }
        eprintln!(
            "[spec-check] o_lc maxdiff(on-off) = {mx:.6e} @i={mi}(t={} h={} d={}) on={:.6e} off={:.6e}",
            mi / (32 * 128),
            (mi / 128) % 32,
            mi % 128,
            a[mi],
            b[mi]
        );
        for t in 0..4 {
            let mut tm = 0.0f32;
            for i in t * 32 * 128..(t + 1) * 32 * 128 {
                tm = tm.max((a[i] - b[i]).abs());
            }
            eprintln!("[spec-check] t={t} maxdiff={tm:.6e}");
        }
        // [핵심 대조] spec scan(t=4) 최종 상태 vs trio(4×t=1) 최종 상태.
        // + 정상 scan(WY) 경로도 같은 비교 — 공식 차이 규모 판별.
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        dec.spec_set_scan(false);
        let _ = dec.spec_verify(0, &rows, 4)?;
        let s_wy = dec.spec_dump_state(0)?;
        dec.spec_set_scan(true);
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        let _ = dec.spec_verify(0, &rows, 4)?;
        let mut wmx = 0.0f32;
        for i in 0..s_trio.len().min(s_wy.len()) {
            wmx = wmx.max((s_trio[i] - s_wy[i]).abs());
        }
        eprintln!("[spec-check] state maxdiff(trio-WY) = {wmx:.6e}");
        let s_spec = dec.spec_dump_state(0)?;
        let mut smx = 0.0f32;
        let mut smi = 0usize;
        for i in 0..s_trio.len().min(s_spec.len()) {
            let d = (s_trio[i] - s_spec[i]).abs();
            if d > smx {
                smx = d;
                smi = i;
            }
        }
        eprintln!(
            "[spec-check] state maxdiff(trio-spec) = {smx:.6e} @i={smi} (층 {}) trio={:.6e} spec={:.6e}",
            smi / (32 * 128 * 128),
            s_trio[smi],
            s_spec[smi]
        );
        // [TreeWY 2.B 2026-10-12] WY 재구성 롤백 대조 — verify(4) 후
        // rollback(k) 상태 ≈ 순차 t=1 디코드 k회 상태(k=1..4 전 접두).
        // 참조 = S0 복원 → t=1 디코드마다 덤프(순차 스캔이 정답). 기대
        // 오차 ~1e-7급(PoC eval 클래스 — 골든 토큰열 판정은 H 섹션이 담당).
        let mut refs: Vec<Vec<f32>> = Vec::new();
        {
            dec.spec_load_state(0)?;
            dec.spec_rewind_pos(0, 4)?;
            let mut cur = next;
            for _ in 0..4 {
                let row = model.embed_row(cur).map_err(|e| e.to_string())?;
                cur = dec.forward_device_argmax(0, &row)?;
                refs.push(dec.spec_dump_state(0)?);
            }
        }
        let mut wy_mx = 0.0f32;
        let dm_hv = dec.spec_hv().max(1);
        for k in 1..=4usize {
            dec.spec_load_state(0)?;
            dec.spec_rewind_pos(0, 4)?;
            let _ = dec.spec_verify(0, &rows, 4)?;
            dec.spec_rollback(0, k)?;
            let s = dec.spec_dump_state(0)?;
            let mut mx = 0.0f32;
            let mut mi = 0usize;
            for i in 0..s.len().min(refs[k - 1].len()) {
                let dv = (s[i] - refs[k - 1][i]).abs();
                if dv > mx {
                    mx = dv;
                    mi = i;
                }
            }
            eprintln!(
                "[spec-check] WY rollback k={k} state maxdiff(seq-wy) = {mx:.6e} @i={mi} (층 {} 헤드{} 행{} 열{}) wy={:.6e} seq={:.6e}",
                mi / (dm_hv * 128 * 128),
                (mi / (128 * 128)) % dm_hv,
                (mi / 128) % 128,
                mi % 128,
                s[mi],
                refs[k - 1][mi]
            );
            wy_mx = wy_mx.max(mx);
        }
        eprintln!("[spec-check] WY rollback max(k=1..4) = {wy_mx:.6e}");
        // [TreeWY 진단] **같은 verify(4) 라운드 내** 대조 — 순차 스캔 최종
        // 상태(S_4) vs 같은 라운드의 wy 데이터로 재구성한 S_4. 입력·pos·
        // 참조 경로 교란 원천이 없는 순수 커널 대조(합성 단위 테스트
        // wy_reconstruct_matches_sequential의 실모델 버전 — 기대 ~1e-6).
        {
            dec.spec_load_state(0)?;
            dec.spec_rewind_pos(0, 4)?;
            let _ = dec.spec_verify(0, &rows, 4)?;
            let s_batch4 = dec.spec_dump_state(0)?;
            dec.spec_rollback(0, 4)?;
            let s_wy4b = dec.spec_dump_state(0)?;
            let mut mx = 0.0f32;
            let mut mi = 0usize;
            for i in 0..s_batch4.len().min(s_wy4b.len()) {
                let dv = (s_batch4[i] - s_wy4b[i]).abs();
                if dv > mx {
                    mx = dv;
                    mi = i;
                }
            }
            eprintln!(
                "[spec-check] same-round WY(4) vs scan-final maxdiff = {mx:.6e} @i={mi} wy={:.6e} scan={:.6e}",
                s_wy4b[mi], s_batch4[mi]
            );
        }
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
    }
    let (mut spec_rounds, mut spec_drafted, mut spec_acc) = (0u64, 0u64, 0u64);
    let spec_t0 = std::time::Instant::now();
    // 원 루프(for _ in 0..n_predict) = n_predict회 **추가** — out은 프리필
    // 첫 토큰을 이미 담고 있다(오프바이원 실측: 35B 4000 골든 8토큰).
    while out.len() <= n_predict {
        if spec_k >= 2 && head_gpu {
            // 초안 = 최대 spec_k-1개(검증 배치 = 1+초안 ≤ spec_k).
            // n=4 — 12는 우리 산문 테스트에서 발화 0(실측). 짧은 패턴이
            // 반복 구조(리스트·코드)를 더 자주 잡는다(수용률은 검증이 판정).
            let draft = llm170_core::spec::ngram_draft(&hist, 4, 48, spec_k - 1);
            if !draft.is_empty() {
                let t = 1 + draft.len();
                let mut rows: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
                rows.extend_from_slice(&model.embed_row(next).map_err(|e| e.to_string())?);
                for &tk in &draft {
                    rows.extend_from_slice(&model.embed_row(tk).map_err(|e| e.to_string())?);
                }
                // [dsave 대체 2026-10-14] 라운드 전 저장 폐지 — 롤백 재료는
                // WY solve + conv 링 스냅샷(프로덕션 sched와 동일 계약).
                let toks = dec.spec_verify(0, &rows, t)?;
                let mut acc = 0usize;
                while acc < draft.len() && toks[acc] == draft[acc] {
                    acc += 1;
                }
                spec_rounds += 1;
                spec_drafted += draft.len() as u64;
                spec_acc += acc as u64;
                // 상태·pos를 마지막 수용 토큰 **직전**으로 되감고 t=1 디코드.
                let last = if acc >= 1 { draft[acc - 1] } else { next };
                if acc >= 1 {
                    dec.spec_rollback(0, acc)?;
                    dec.spec_rewind_pos(0, (t - acc) as u32)?;
                } else {
                    dec.spec_rollback(0, 0)?;
                    dec.spec_rewind_pos(0, t as u32)?;
                }
                let lrow = model.embed_row(last).map_err(|e| e.to_string())?;
                let corr = dec.forward_device_argmax(0, &lrow)?;
                for &tk in &draft[..acc] {
                    out.push(tk);
                    hist.push(tk);
                }
                out.push(corr);
                hist.push(corr);
                next = corr;
                continue;
            }
        }
        let row = model.embed_row(next).map_err(|e| e.to_string())?;
        if head_gpu {
            // [P3] 디코드는 디바이스 argmax(4B d2h) — 토큰 스트림은 종전
            // 로짓+CPU greedy_from과 동일해야 한다(골든이 동등성 판정).
            next = dec.forward_device_argmax(0, &row)?;
        } else {
            let xn = dec.forward_device(0, &row)?;
            next = llm170_core::matmul::greedy_from(&head_logits(&xn));
        }
        out.push(next);
        hist.push(next);
    }
    if llm170_diag::flag::ne0("LLM170_TIME") {
        // 진단(P8): 커널 범주별 소요 — 그래프 캡처 중에는 마킹이 꺼지므로
        // LLM170_GRAPH=0 직접 경로에서 의미가 있다.
        eprintln!("{}", dec.prof_report(&format!("decode {n_predict}토큰"))?);
    }
    if spec_rounds > 0 {
        eprintln!(
            "[spec] 라운드 {spec_rounds} · 초안 {spec_drafted} · 수용 {spec_acc} (수용률 {:.0}%) · {:.1}ms",
            spec_acc as f64 / spec_drafted.max(1) as f64 * 100.0,
            spec_t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;
    let csv = out
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    // MoE는 플레인(비양자화) 업로드 실측 — dense는 split 선형 수 기반 추정.
    let plain_gb = if hp.n_experts > 0 {
        model
            .engine_names()
            .iter()
            .filter(|n| n.as_str() != "token_embd.weight" && n.as_str() != "output.weight")
            .filter_map(|n| model.w_raw(n))
            .filter(|w| w.ty == llm170_core::wtype::WType::Bf16)
            .map(|w| w.data.len() as f64 * 1e-9)
            .sum::<f64>()
    } else {
        n_lin as f64 * 44.6e-3
    };
    Ok(format!(
        "w4a16-gpu {dir} — GPU 체인(디바이스 상주{} · {})\n  업로드 {upload_ms:.0}ms · 생성 {}토큰 {gen_ms:.0}ms ({:.1}ms/토큰)\n tokens: {csv}",
        if head_gpu {
            " + GPU head"
        } else {
            " + CPU head"
        },
        if hp.n_experts > 0 {
            format!("MoE 플레인 {plain_gb:.1}GB + 전문가 스트리밍/상주")
        } else {
            format!("선형 {n_lin}개 · {plain_gb:.1}GB급")
        },
        out.len(),
        gen_ms / out.len() as f64,
    ))
}

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
    if a.bench.is_none() && a.prompts.is_empty() {
        return Err("w4a16-gpu: 프롬프트 필요(--prompt-tokens)".into());
    }
    let prompt: Vec<u32> = a.prompts.first().cloned().unwrap_or_default();
    if a.bench.is_none() && prompt.len() + a.n_predict + 1 > a.ctx {
        return Err(format!("ctx({}) too small for prompt+n_predict", a.ctx));
    }
    let model =
        llm170_core::qwen35::Model::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
    // 1~4) 모델 상주 업로드(GpuEngine 공용 — dense/MoE 분기 포함).
    // kv_reserve 0 = 종전 상주 규칙(프로브 — 서버 사다리 판정과 분리).
    crate::gpu_engine::upload_model(&mut dec, &model, a.ctx, 0)?;
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
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut dec = W4a16Dec::new(1, hp.n_embd, hp.n_layer)?;
    dec.debug_layers = llm170_diag::dump::opts().key("debug_layers");
    let cfg = crate::gpu_engine::upload_model_g4(&mut dec, &model, a.ctx)?;
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
    let (mut flips_bad, mut theta_max) = (0usize, 0f32);
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
        if got[i] != want[i] && margin >= theta {
            flips_bad += 1; // 마진 ≥ 클래스 = 실결함 후보(첫 발산 판정이 정본)
        }
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
    } else if flips_bad == 0 {
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
