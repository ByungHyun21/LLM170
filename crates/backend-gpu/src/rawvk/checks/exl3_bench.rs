//! EXL3 전 모델 선형 체인 벤치 (plans/118 §3-1) — 디코드 스텝 t/s 실측.
//!
//! 모든 EXL3 선형(trellis+suh+svh)을 vk 버퍼에 상주시키고 각 선형의
//! 3-커널 체인(had_in→gemv→had_out)을 실측 → 스텝 총 시간·유효 대역폭·
//! 예상 t/s. 어텐션/GDN 등 비선형 비용은 기존 Q4 실측치로 별도 가산.
//! 정확성 검증은 exl3-vk-check 담당 — 여기는 속도만.

use crate::rawvk::context::{Pipes, VkBuf, VkCtx};
use ash::vk::Buffer;
use half::f16;

struct BenchLinear {
    k: usize,
    n: usize,
    krate: u32,
    tre: Buffer,
    suh: Buffer,
    svh: Buffer,
}

/// `llm170 exl3-bench [exl3_dir] [reps]` — 기본 27B·5회.
pub fn exl3_bench(exl3_dir: &str, reps: usize) -> Result<String, String> {
    let ar =
        llm170_exl3::StArchive::open(std::path::Path::new(exl3_dir)).map_err(|e| e.to_string())?;
    // 디코드 스텝은 언어 모델 선형만 — 비전 타워(model.visual.*) 제외.
    // (비전 인코딩은 이미지당 1회, t/s 분모에 부당)
    let mut tre_keys: Vec<String> = ar
        .entries()
        .keys()
        .filter(|k| k.ends_with(".trellis") && !k.contains("model.visual."))
        .cloned()
        .collect();
    tre_keys.sort();
    if tre_keys.is_empty() {
        return Err("trellis 텐서 없음".into());
    }

    let mut ctx = VkCtx::new()?;
    let p1 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_had_in.spv"), 3, 4)?;
    let p2 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_gemv.spv"), 3, 12)?;
    let p3 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_had_out.spv"), 3, 8)?;

    // ── 상주 적재: 샤드 파일 → 매핑 ptr 직독(RAM 이중 상주 회피) ──
    let mut lins: Vec<BenchLinear> = Vec::new();
    let mut total_bytes = 0usize;
    let mut k_count: std::collections::BTreeMap<u32, usize> = Default::default();
    for key in &tre_keys {
        let base = key.trim_end_matches(".trellis");
        let shape = &ar.entry(key).ok_or("엔트리 유실")?.shape;
        if shape.len() != 3 || shape[2] % 16 != 0 {
            continue; // 반정수/비정규 — 현 타깃 없음(§6.1 실측)
        }
        let (krate, kt, nt) = ((shape[2] / 16) as u32, shape[0] as usize, shape[1] as usize);
        let (k, n) = (kt * 16, nt * 16);
        let tre_bytes = (ar.entry(key).unwrap().nbytes()) as usize;

        let treb = ctx.alloc(tre_bytes)?;
        let suhb = ctx.alloc(k * 2)?;
        let svhb = ctx.alloc(n * 2)?;
        // SAFETY: alloc 영구 매핑 영역 — 크기 일치 파일 직독.
        unsafe {
            ar.read_into(key, treb.ptr, tre_bytes)
                .map_err(|e| e.to_string())?;
            ar.read_into(&format!("{base}.suh"), suhb.ptr, k * 2)
                .map_err(|e| e.to_string())?;
            ar.read_into(&format!("{base}.svh"), svhb.ptr, n * 2)
                .map_err(|e| e.to_string())?;
        }
        total_bytes += tre_bytes + k * 2 + n * 2;
        *k_count.entry(krate).or_default() += 1;
        // 하다마드 커널 계약: k·n 128 배수 — 비전 타워(4304 등) 예외는 스킵.
        if k % 128 != 0 || n % 128 != 0 {
            eprintln!("  [bench] 스킵(128 미배수): {base} k={k} n={n}");
            lins.pop();
            *k_count.get_mut(&krate).unwrap() -= 1;
            continue;
        }
        lins.push(BenchLinear {
            k,
            n,
            krate,
            tre: treb.buf,
            suh: suhb.buf,
            svh: svhb.buf,
        });
    }
    if lins.is_empty() {
        return Err("적재 가능한 trellis 없음".into());
    }

    // k 값별 x(f16 랜덤)·공유 중간 버퍼.
    let mut seed = 0x1234_0000_0005u64;
    let mut lcg = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let mut xbufs: Vec<(usize, VkBuf)> = Vec::new();
    let nseg: u32 = 16; // [E3 실험] plans/120 A1 — k-분할 8→16
    let (mut max_ah, mut max_s, mut max_y) = (0usize, 0usize, 0usize);
    for l in &lins {
        max_ah = max_ah.max(l.k * 2);
        max_s = max_s.max(l.n * 4 * nseg as usize);
        max_y = max_y.max(l.n * 4);
    }
    let ahb = ctx.alloc(max_ah)?;
    let sb = ctx.alloc(max_s)?;
    let yb = ctx.alloc(max_y)?;

    let mut xb_for =
        |k: usize, xbufs: &mut Vec<(usize, VkBuf)>, ctx: &mut VkCtx| -> Result<Buffer, String> {
            if let Some((_, b)) = xbufs.iter().find(|(kk, _)| *kk == k) {
                return Ok(b.buf);
            }
            let b = ctx.alloc(k * 2)?;
            let mut v = vec![0u8; k * 2];
            for i in 0..k {
                let bits = f16::from_f32(lcg()).to_bits();
                v[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
            }
            // SAFETY: 매핑 업로드(런치 전).
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.ptr, k * 2) };
            let ret = b.buf;
            xbufs.push((k, b));
            Ok(ret)
        };

    let run_chain = |ctx: &mut VkCtx,
                     p1: &Pipes,
                     p2: &Pipes,
                     p3: &Pipes,
                     l: &BenchLinear,
                     xb: Buffer,
                     ahb: &VkBuf,
                     sb: &VkBuf,
                     yb: &VkBuf|
     -> Result<(), String> {
        let ds1 = ctx.fresh_ds_for(p1, 3)?;
        ctx.bind_bufs(ds1, &[xb, l.suh, ahb.buf]);
        ctx.run_rw(
            p1.pl,
            ds1,
            p1.pipe,
            &(l.k as u32 / 128).to_le_bytes(),
            (l.k / 128) as u32,
            1,
            1,
            &[xb, l.suh],
            &[ahb.buf],
        )?;
        let ds2 = ctx.fresh_ds_for(p2, 3)?;
        ctx.bind_bufs(ds2, &[ahb.buf, l.tre, sb.buf]);
        let push2: Vec<u8> = [(l.k / 16) as u32, (l.n / 16) as u32, l.krate]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ctx.run_rw(
            p2.pl,
            ds2,
            p2.pipe,
            &push2,
            ((l.n / 16) as u32).div_ceil(8),
            nseg,
            1,
            &[ahb.buf, l.tre],
            &[sb.buf],
        )?;
        let ds3 = ctx.fresh_ds_for(p3, 3)?;
        ctx.bind_bufs(ds3, &[sb.buf, l.svh, yb.buf]);
        let push3: Vec<u8> = [(l.n as u32 / 128), nseg]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ctx.run_rw(
            p3.pl,
            ds3,
            p3.pipe,
            &push3,
            (l.n / 128) as u32,
            1,
            1,
            &[sb.buf, l.svh],
            &[yb.buf],
        )?;
        Ok(())
    };

    // 워밍업 1 스텝 — 실측과 동일한 배치 경로로.
    {
        ctx.begin_batch()?;
        let xb = xb_for(lins[0].k, &mut xbufs, &mut ctx)?;
        run_chain(&mut ctx, &p1, &p2, &p3, &lins[0], xb, &ahb, &sb, &yb)?;
        ctx.end_batch_wait()?;
    }

    // 실측 — 배치 디스패치: rep당 begin_batch 1회·녹화 후 단일 제출/대기.
    // (초판 비배치 run은 디스패치마다 자체 동기 — 1719회/스텝 ≈ 40ms
    // 오버헤드로 4.25 t/s 측정. 엔진의 스텝 배치 아키텍처와 동일 구조.)
    let mut per_rep_us: Vec<f64> = Vec::new();
    for _ in 0..reps.max(1) {
        let t0 = std::time::Instant::now();
        ctx.begin_batch()?;
        for l in &lins {
            let xb = xb_for(l.k, &mut xbufs, &mut ctx)?;
            run_chain(&mut ctx, &p1, &p2, &p3, l, xb, &ahb, &sb, &yb)?;
        }
        ctx.end_batch_wait()?;
        per_rep_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    per_rep_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_us = per_rep_us[per_rep_us.len() / 2]; // 중앙값

    let gbps = total_bytes as f64 / (total_us * 1e-6) / 1e9;
    let tps = 1e6 / total_us;
    let mut report = format!(
        "exl3-bench {exl3_dir}: 선형 {}개 상주 {:.2} GB, K분포 {:?}\n  스텝 {total_us:.0} µs(중앙) → 예상 t/s(EXL3 선형만) {tps:.2} · 유효 {gbps:.0} GB/s\n  per-rep: {:?}",
        lins.len(),
        total_bytes as f64 / 1e9,
        k_count,
        per_rep_us
            .iter()
            .map(|u| format!("{u:.0}"))
            .collect::<Vec<_>>(),
    );
    report.push_str("\n  참고: Q4 기준선 27B decode ~14 t/s — 비선형 비용 별도 가산");
    Ok(report)
}
