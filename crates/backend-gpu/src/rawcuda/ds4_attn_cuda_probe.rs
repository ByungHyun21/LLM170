//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: 층 케이스마다 가중치 재등록·입력 재업로드,
//!    모듈 작업 버퍼는 현 t 행 전체를 매 체인마다 덮어쓴다.
//! ② 형상은 픽스처 config.json에서 자동 확정(Vision-Exp 실측 dim 4096·헤드
//!    64×512·rope 64·q_lora 1024·window 128·index 64×128 top-512·ratio
//!    [0,0,4,128,...]) — L0(SWA)/L2(CSA)/L3(HCA) 3층 유형 전수.
//! ③ 캡처-재생: 실측 EXL3 픽스처(D:/models/DeepSeek-V4-Flash-Vision-Exp-
//!    exl3-3.04bpw) — trellis 선형 디양자화 + BF16/F32 plain + embed 행
//!    전부 오프셋 직독(전체 적재 금지 — loader.rs 계약). 입력 x 는 embed 행
//!    (실측 bf16 격자값) t 토큰.
//! ④ 종단 값이 유일 불변량: 판정은 스테이지별 값 maxdiff+bitdiff(비트동일
//!    기준)·선택 리스트 동일성. argmax 판정 금지(plans/124 §5).
//!
//! [오라클 — core deepseek4 참조 직이식(값 판정의 유일 기준, plans/124 §6).
//! 인용은 전부 워크트리 c5da4b9a 기준 줄번호, 2026-10-05]
//! - crates/core/src/deepseek4/ops.rs — bf16_round L17 · rms_norm_weighted
//!   L33 · rms_scale L48 · gemm_nt L58 · pow2_ceil L77 · f32_to_e4m3 L88 ·
//!   e4m3_to_f32 L131 · f32_to_e2m1 L150 · fp8_sim L174 · fp4_sim L194 ·
//!   hadamard_rotate L243 · RopeTable L268-333 · rope_apply L335.
//! - crates/core/src/ops.rs — exp_cr L52-98(f64 fma 호너 13단 — 커널
//!   ds4_exp_cr 과 리터럴까지 동일).
//! - crates/core/src/deepseek4/stages/attn.rs — project_q L84 · rope_q L122 ·
//!   project_kv L137 · window_idx_prefill L160 · compress_idx_dense L175 ·
//!   compressor_pool_prefill L183 · pool_rows L231 · compressor_finish L256 ·
//!   indexer_q L394 · indexer_k L429 · indexer_weights L452 · indexer_scores
//!   L465 · indexer_topk L494 · sparse_attn_one L531 · attention_output L582 ·
//!   attention_forward L601.
//! - crates/exl3/src/trellis.rs — PERM_INV L24 · mul1_decode L47 · tile_word
//!   L60 · dequant_view L128 · had128(f64) L279 — 가중치 디양자화 미러.
//! - crates/core/src/deepseek4/loader.rs — linear L344(스트립 병렬 디양자화)
//!    · conv/plain_f32 L156 · plain_kmat L385 · embed_rows L437(오프셋 행 독).
//!
//! [정합 원장 요약 — 전 케이스 비트동일 목표(B3 판정 기준)] L0/L2/L3 전
//! 스테이지(c_Q·q·kv·압축 엔트리·qI·kI·인덱서 가중치·스코어·선택·어텐션
//! o·종단 y) bitdiff=0·nan=0 기대. 초월함수는 전부 트윈/테이블(exp_cr f64
//! 트윈·RoPE 호스트 표)이라 libm 발산 여지 없음 — 측정값은 커밋에 기록.
//!
//! [병렬 계약] 오라클 병렬화는 전부 "출력 영역 분할 소유"형(행/블록/토큰
//! 독립 — 항목별 k-오름차순 누산 등 연산 순서 불변, loader.rs 스트립
//! 병렬과 동일 결정성 계약).
//!
//! [음성대조 — 원장 17호(계기 자체 검증)] 3종 모두 NEG-DETECTED 필:
//! (a) indexer top-k 인과 경계 off-by-one(visible+1 — t ≤ (k−1)·ratio
//!     구간에서 선택 리스트 길이 자체가 어긋남: 구조 보장).
//! (b) 컴프레서 ape 행 미스얼라인(풀 가중치 이격).
//! (c) 싱크 로짓 누락(분모 exp(z'−m) 제거 — bf16 경계 플립으로 검출).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0. 개발기 RTX 4070
//! SUPER sm_89 타이밍은 판단 근거 아님 — plans/130 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::ds4_attn_cuda::{
    Ds4AttnCuda, Ds4AttnDims, Ds4CompF32, Ds4IndexerF32, Ds4LayerF32, Ds4Neg,
};
use crate::rawcuda::exl3_cuda::{JParser, JVal, StArchive};
use crate::rawcuda::exl3_cuda_probe::{f16_to_f32, maxdiff_nan};
use std::path::{Path, PathBuf};

/// 기본 픽스처 경로(과제 지정 — 프로브 인자로 대체 가능).
pub const DS4_EXL3_DIR: &str = "D:/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw";

/// 음성대조 "탐지됨" 판정 경계(정상 케이스 판정은 bitdiff=0 기준).
const DS4_THRESH: f32 = 1e-6;

// ── 병렬 유틸(출력 분할 소유 — 비트 불변) ──

fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .max(1)
}

/// [0,n) 항목(항목당 `elems` 원소)을 연속 스트립으로 분할 소유해 병렬
/// 실행. f(lo, hi, part) — part 는 v[lo·elems .. hi·elems) 슬라이스.
/// 파트별 출력 분할 소유(공유 &mut 금지 — 항목별 산출 순서 불변, 비트 불변).
fn par_ranges_mut<T: Send, F: Fn(usize, usize, &mut [T]) + Sync>(v: &mut [T], elems: usize, f: &F) {
    let n = v.len() / elems;
    if n == 0 {
        return;
    }
    let threads = n_threads().min(n);
    if threads <= 1 {
        f(0, n, v);
        return;
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|sc| {
        let mut rest: &mut [T] = v;
        let mut base = 0usize;
        let mut hs = Vec::new();
        while base < n {
            let take = per.min(n - base);
            let cnt = take * elems;
            let (part, tail) = rest.split_at_mut(cnt);
            rest = tail;
            let b0 = base;
            hs.push(sc.spawn(move || f(b0, b0 + take, part)));
            base += take;
        }
        for h in hs {
            h.join().expect("par_ranges_mut 스레드 패닉");
        }
    });
}

/// buf 를 row_len 행 단위 스트립으로 분할 소유해 행별 변환(제자리).
/// f(row_index, row_slice) — 행 인덱스는 전체 기준.
fn par_rows_mut<F: Fn(usize, &mut [f32]) + Sync>(v: &mut [f32], row_len: usize, f: &F) {
    par_ranges_mut(v, row_len, &|lo, _hi, part: &mut [f32]| {
        for (i, row) in part.chunks_mut(row_len).enumerate() {
            f(lo + i, row);
        }
    });
}

// ── core 미러 오라클: 스칼라/벡터 연산(ops.rs 직이식) ──

/// ops.rs bf16_round L17-23.
fn o_bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let hi = ((b >> 16) as u16) as u32 & 1;
    f32::from_bits(((b + 0x7FFF + hi) >> 16) << 16)
}

/// ops.rs rms_norm_weighted L33-46 — f32 순차 제곱합, 출력 bf16 경계.
fn o_rms_norm_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| o_bf16_round(v * scale * g))
        .collect()
}

/// ops.rs rms_scale L48-55.
fn o_rms_scale(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}

/// ops.rs gemm_nt L58-75 — 행 스트립 병렬((i,j)별 k-오름차순 누산 불변).
fn o_gemm_nt(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let m = x.len() / k;
    let mut y = vec![0.0f32; m * n];
    let body = |lo: usize, hi: usize, ys: &mut [f32]| {
        for i in lo..hi {
            let xr = &x[i * k..(i + 1) * k];
            let yr = &mut ys[(i - lo) * n..(i - lo + 1) * n];
            for (kk, &xv) in xr.iter().enumerate() {
                let wr = &w[kk * n..(kk + 1) * n];
                for (j, &wv) in wr.iter().enumerate() {
                    yr[j] += xv * wv;
                }
            }
        }
    };
    par_ranges_mut(&mut y, n, &body);
    y
}

/// gemm 1행(attention_output 그룹 슬라이스용 — k-오름차순 누산).
fn o_gemm_row(x: &[f32], w: &[f32], _k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for (kk, &xv) in x.iter().enumerate() {
        let wr = &w[kk * n..(kk + 1) * n];
        for (j, &wv) in wr.iter().enumerate() {
            y[j] += xv * wv;
        }
    }
    y
}

/// ops.rs pow2_ceil L77-85.
fn o_pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    let l2 = e - 127 + i32::from(b & 0x7F_FFFF != 0);
    f32::from_bits(((l2 + 127) as u32) << 23)
}

/// ops.rs f32_to_e4m3 L88-129.
fn o_f32_to_e4m3(x: f32) -> u8 {
    let sign = u8::from(x.is_sign_negative()) << 7;
    let a = x.abs();
    if a < 2.0f32.powi(-10) {
        return sign;
    }
    if a < 2.0f32.powi(-6) {
        let q = a * 512.0;
        let r = q.round_ties_even();
        if r >= 8.0 {
            return sign | (1 << 3);
        }
        return sign | r as u8;
    }
    let b = a.to_bits();
    let e = (((b >> 23) & 0xFF) as i32) - 127;
    let mant = b & 0x7F_FFFF;
    const SH: u32 = 20;
    let rem = mant & ((1 << SH) - 1);
    let half = 1u32 << (SH - 1);
    let mut mm = mant >> SH;
    let mut ee = e;
    if rem > half || (rem == half && (mm & 1) == 1) {
        mm += 1;
    }
    if mm == 8 {
        mm = 0;
        ee += 1;
    }
    let e4 = (ee + 7) as u8;
    if e4 > 15 || (e4 == 15 && mm > 7) {
        return sign | 0x7F;
    }
    sign | (e4 << 3) | mm as u8
}

/// ops.rs e4m3_to_f32 L131-148.
fn o_e4m3_to_f32(u: u8) -> f32 {
    let sign = f32::from_bits((((u & 0x80) as u32) << 24) | 0x3F80_0000);
    let e4 = ((u >> 3) & 0xF) as i32;
    let m = (u & 7) as u32;
    if e4 == 15 && m == 7 {
        return f32::NAN.copysign(sign);
    }
    if e4 == 0 {
        return sign * (m as f32) * 2.0f32.powi(-9);
    }
    let bits = (((e4 - 7 + 127) as u32) << 23) | (m << 20);
    sign * f32::from_bits(bits)
}

/// ops.rs f32_to_e2m1 L150-160.
fn o_f32_to_e2m1(x: f32) -> f32 {
    let a = x.abs();
    if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    }
}

/// ops.rs fp8_sim L174-192 — (행, 블록) 독립, 행 스트립 병렬.
fn o_fp8_sim(x: &mut [f32], cols: usize, block: usize) {
    let body = |_r: usize, row: &mut [f32]| {
        for c0 in (0..cols).step_by(block) {
            let blk = &mut row[c0..(c0 + block).min(cols)];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = o_pow2_ceil(amax * (1.0 / 448.0));
            for v in blk.iter_mut() {
                let q = (*v / s).clamp(-448.0, 448.0);
                *v = o_e4m3_to_f32(o_f32_to_e4m3(q)) * s;
            }
        }
    };
    par_rows_mut(x, cols, &body);
}

/// ops.rs fp4_sim L194-210 — 32원소 블록.
fn o_fp4_sim(x: &mut [f32]) {
    for blk in x.chunks_mut(32) {
        let mut amax = 0.0f32;
        for &v in blk.iter() {
            amax = amax.max(v.abs());
        }
        amax = amax.max(6.0 * 2.0f32.powi(-126));
        let s = o_pow2_ceil(amax * (1.0 / 6.0));
        for v in blk.iter_mut() {
            let q = (*v / s).clamp(-6.0, 6.0);
            *v = o_f32_to_e2m1(q).copysign(*v) * s;
        }
    }
}

/// core ops.rs exp_cr L52-98 — f64 fma 호너 13단(커널 ds4_exp_cr 과 동일 DAG).
fn o_exp_cr(x: f32) -> f32 {
    let xd = x as f64;
    if xd > 88.72 {
        return f32::INFINITY;
    }
    if xd < -103.97 {
        return 0.0;
    }
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let kd = (xd * INV_LN2).round_ties_even();
    let k = kd as i64;
    let mut r = (-kd).mul_add(LN2_HI, xd);
    r = (-kd).mul_add(LN2_LO, r);
    let mut p = 1.0f64 / 1307674368000.0;
    p = p.mul_add(r, 1.0 / 479001600.0);
    p = p.mul_add(r, 1.0 / 39916800.0);
    p = p.mul_add(r, 1.0 / 3628800.0);
    p = p.mul_add(r, 1.0 / 362880.0);
    p = p.mul_add(r, 1.0 / 40320.0);
    p = p.mul_add(r, 1.0 / 5040.0);
    p = p.mul_add(r, 1.0 / 720.0);
    p = p.mul_add(r, 1.0 / 120.0);
    p = p.mul_add(r, 1.0 / 24.0);
    p = p.mul_add(r, 1.0 / 6.0);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);
    if k > 127 {
        return f32::INFINITY;
    }
    let scale = f64::from_bits(((k + 1023) as u64) << 52);
    (p * scale) as f32
}

/// ops.rs hadamard_rotate L243-261.
fn o_hadamard_rotate(x: &mut [f32]) {
    let n = x.len();
    let mut width = 1;
    while width < n {
        let mut base = 0;
        while base < n {
            for i in 0..width {
                let (a, b) = (x[base + i], x[base + width + i]);
                x[base + i] = a + b;
                x[base + width + i] = a - b;
            }
            base += 2 * width;
        }
        width *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// ops.rs RopeTable L268-333 — f64 powf 호스트 표(단일 기준 — 모듈과 공유).
struct ORope {
    cs: Vec<f32>,
    half: usize,
}

fn o_find_correction_dim(num_rot: f64, dim: usize, base: f64, max_len: usize) -> f64 {
    dim as f64 * (max_len as f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
}

fn o_rope_build(
    dim: usize,
    len: usize,
    base: f64,
    yarn: bool,
    factor: f64,
    orig_len: usize,
    beta_fast: f64,
    beta_slow: f64,
) -> ORope {
    let half = dim / 2;
    let mut freqs = vec![0.0f32; half];
    for (p, f) in freqs.iter_mut().enumerate() {
        *f = (base.powf((2 * p) as f64 / dim as f64) as f32).recip();
    }
    if yarn && orig_len > 0 {
        let low = o_find_correction_dim(beta_fast, dim, base, orig_len)
            .floor()
            .max(0.0) as usize;
        let high = (o_find_correction_dim(beta_slow, dim, base, orig_len)
            .ceil()
            .min(dim as f64 - 1.0)) as usize;
        let (lo, hi) = (low as f32, high as f32);
        for (p, f) in freqs.iter_mut().enumerate() {
            let mut ramp = ((p as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
            if low == high {
                ramp = ((p as f32 - lo) / (hi - lo + 0.001)).clamp(0.0, 1.0);
            }
            let smooth = 1.0 - ramp;
            *f = (*f / factor as f32) * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut cs = vec![0.0f32; len * half * 2];
    for pos in 0..len {
        for (p, &f) in freqs.iter().enumerate() {
            let angle = pos as f32 * f;
            cs[pos * half * 2 + p * 2] = angle.cos();
            cs[pos * half * 2 + p * 2 + 1] = angle.sin();
        }
    }
    ORope { cs, half }
}

/// ops.rs rope_apply L335-350.
fn o_rope_apply(x: &mut [f32], cs: &[f32], inverse: bool) {
    let half = cs.len() / 2;
    for p in 0..half {
        let (c, s) = (
            cs[p * 2],
            if inverse {
                -cs[p * 2 + 1]
            } else {
                cs[p * 2 + 1]
            },
        );
        let (x0, x1) = (x[p * 2], x[p * 2 + 1]);
        x[p * 2] = x0 * c - x1 * s;
        x[p * 2 + 1] = x0 * s + x1 * c;
    }
}

// ── core 미러 오라클: 어텐션 스테이지(attn.rs 직이식) ──
// 가중치 자료는 모듈 Ds4LayerF32/Ds4CompF32/Ds4IndexerF32 를 그대로 공유
// (필드명 · 레이아웃이 attn.rs AttnWeights/CompressorWeights/IndexerWeights
// 와 동치 — 메모리 이중 적재 회피).

/// attn.rs project_q L84-120 — (c_Q, q[로프 전]).
fn o_project_q(lay: &Ds4LayerF32, x: &[f32], d: &Ds4AttnDims) -> (Vec<f32>, Vec<f32>) {
    let (dm, qrank, nh, hd) = (d.dim, d.q_lora_rank, d.n_heads, d.head_dim);
    let _ = (nh,);
    let t = x.len() / dm;
    let mut xq = x.to_vec();
    o_fp8_sim(&mut xq, dm, 128);
    let mut c = o_gemm_nt(&xq, &lay.wq_a, dm, qrank);
    for v in c.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        rows.push(o_rms_norm_weighted(
            &c[i * qrank..(i + 1) * qrank],
            &lay.q_norm,
            d.rms_eps,
        ));
    }
    c = rows.concat();
    let mut cq = c.clone();
    o_fp8_sim(&mut cq, qrank, 128);
    let mut q = o_gemm_nt(&cq, &lay.wq_b, qrank, d.n_heads * hd);
    for v in q.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let eps = d.rms_eps;
    let body = |_i: usize, head: &mut [f32]| {
        let s = o_rms_scale(head, eps);
        for v in head.iter_mut() {
            *v = o_bf16_round(*v * s);
        }
    };
    par_rows_mut(&mut q, hd, &body);
    (c, q)
}

/// attn.rs rope_q L122-135(헤드 행 병렬 — 위치 = 행/nh).
fn o_rope_q(q: &mut [f32], d: &Ds4AttnDims, rope: &ORope) {
    let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
    let csr = &rope.cs;
    let half = rope.half;
    let body = |i: usize, head: &mut [f32]| {
        let pos = i / nh;
        o_rope_apply(
            &mut head[hd - rd..],
            &csr[pos * half * 2..(pos + 1) * half * 2],
            false,
        );
        for v in head[hd - rd..].iter_mut() {
            *v = o_bf16_round(*v);
        }
    };
    par_rows_mut(q, hd, &body);
}

/// attn.rs project_kv L137-158(행 병렬).
fn o_project_kv(lay: &Ds4LayerF32, x: &[f32], d: &Ds4AttnDims, rope: &ORope) -> Vec<f32> {
    let (dm, hd, rd) = (d.dim, d.head_dim, d.rope_head_dim);
    let t = x.len() / dm;
    let mut xq = x.to_vec();
    o_fp8_sim(&mut xq, dm, 128);
    let mut kv = o_gemm_nt(&xq, &lay.wkv, dm, hd);
    for v in kv.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let mut out = vec![0.0f32; t * hd];
    let (eps, knorm) = (d.rms_eps, &lay.kv_norm);
    let kvr = &kv;
    let csr = &rope.cs;
    let half = rope.half;
    let body = |lo: usize, hi: usize, part: &mut [f32]| {
        for i in lo..hi {
            let mut row = o_rms_norm_weighted(&kvr[i * hd..(i + 1) * hd], knorm, eps);
            o_rope_apply(
                &mut row[hd - rd..],
                &csr[i * half * 2..(i + 1) * half * 2],
                false,
            );
            for v in row[hd - rd..].iter_mut() {
                *v = o_bf16_round(*v);
            }
            o_fp8_sim(&mut row[..hd - rd], hd - rd, 64);
            part[(i - lo) * hd..(i - lo + 1) * hd].copy_from_slice(&row);
        }
    };
    par_ranges_mut(&mut out, hd, &body);
    out
}

/// attn.rs window_idx_prefill L160-165.
fn o_window_idx_prefill(t: usize, win: usize) -> Vec<i32> {
    let start = t.saturating_sub(win - 1);
    (start..=t).map(|v| v as i32).collect()
}

/// attn.rs compress_idx_dense L175-179.
fn o_compress_idx_dense(t: usize, ratio: usize, offset: usize) -> Vec<i32> {
    (0..(t + 1) / ratio).map(|b| (b + offset) as i32).collect()
}

/// attn.rs pool_rows L231-254.
fn o_pool_rows(pkv: &[f32], psc: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    let mut w = vec![0.0f32; rows];
    for dd in 0..d {
        let mut m = f32::NEG_INFINITY;
        for r in 0..rows {
            m = m.max(psc[r * d + dd]);
        }
        let mut s = 0.0f32;
        for r in 0..rows {
            w[r] = o_exp_cr(psc[r * d + dd] - m);
            s += w[r];
        }
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += w[r] * pkv[r * d + dd];
        }
        out[dd] = acc / s;
    }
    out
}

/// attn.rs compressor_pool_prefill L183-229(블록 병렬 — 블록 독립).
fn o_pool_prefill(
    kv_c: &[f32],
    score_c: &[f32],
    ape: &[f32],
    head_dim: usize,
    ratio: usize,
) -> Vec<Vec<f32>> {
    let overlap = ratio == 4;
    let coff = 1 + usize::from(overlap);
    let t = kv_c.len() / (coff * head_dim);
    if t < ratio {
        return Vec::new();
    }
    let cd = coff * head_dim;
    let nb = (t - t % ratio) / ratio;
    let mut out = vec![Vec::new(); nb];
    let body = |lo: usize, hi: usize, part: &mut [Vec<f32>]| {
        for i in lo..hi {
            let mut pkv = vec![0.0f32; coff * ratio * head_dim];
            let mut psc = vec![f32::NEG_INFINITY; coff * ratio * head_dim];
            let row_off = if overlap { ratio } else { 0 };
            let src = if overlap { head_dim } else { 0 };
            for j in 0..ratio {
                let cur = i * ratio + j;
                for dd in 0..head_dim {
                    pkv[(row_off + j) * head_dim + dd] = kv_c[cur * cd + src + dd];
                    psc[(row_off + j) * head_dim + dd] =
                        score_c[cur * cd + src + dd] + ape[j * cd + src + dd];
                }
                if overlap && i > 0 {
                    let prev = cur - ratio;
                    for dd in 0..head_dim {
                        pkv[j * head_dim + dd] = kv_c[prev * cd + dd];
                        psc[j * head_dim + dd] = score_c[prev * cd + dd] + ape[j * cd + dd];
                    }
                }
            }
            part[i - lo] = o_pool_rows(&pkv, &psc, coff * ratio, head_dim);
        }
    };
    par_ranges_mut(&mut out, 1, &body);
    out
}

/// attn.rs compressor_finish L256-281.
fn o_comp_finish(
    pooled: &[f32],
    cw: &Ds4CompF32,
    d: &Ds4AttnDims,
    rope: &ORope,
    block_pos: usize,
) -> Vec<f32> {
    let (hd, rd) = (cw.head_dim, d.rope_head_dim);
    let mut kv = o_rms_norm_weighted(pooled, &cw.norm, d.rms_eps);
    o_rope_apply(
        &mut kv[hd - rd..],
        &rope.cs[block_pos * rope.half * 2..(block_pos + 1) * rope.half * 2],
        false,
    );
    for v in kv[hd - rd..].iter_mut() {
        *v = o_bf16_round(*v);
    }
    if cw.rotate {
        o_hadamard_rotate(&mut kv);
        for v in kv.iter_mut() {
            *v = o_bf16_round(*v);
        }
        o_fp4_sim(&mut kv);
    } else {
        o_fp8_sim(&mut kv[..hd - rd], hd - rd, 64);
    }
    kv
}

/// attn.rs indexer_q L394-427(헤드 행 병렬).
fn o_indexer_q(idx: &Ds4IndexerF32, c_q: &[f32], d: &Ds4AttnDims, rope: &ORope) -> Vec<f32> {
    let (qrank, ih, id, rd) = (
        d.q_lora_rank,
        d.index_n_heads,
        d.index_head_dim,
        d.rope_head_dim,
    );
    let t = c_q.len() / qrank;
    let mut xq = c_q.to_vec();
    o_fp8_sim(&mut xq, qrank, 128);
    let mut q = o_gemm_nt(&xq, &idx.wq_b, qrank, ih * id);
    for v in q.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let csr = &rope.cs;
    let half = rope.half;
    let body = |i: usize, head: &mut [f32]| {
        let pos = i / ih;
        o_rope_apply(
            &mut head[id - rd..],
            &csr[pos * half * 2..(pos + 1) * half * 2],
            false,
        );
        for v in head[id - rd..].iter_mut() {
            *v = o_bf16_round(*v);
        }
        o_hadamard_rotate(head);
        for v in head.iter_mut() {
            *v = o_bf16_round(*v);
        }
        o_fp4_sim(head);
    };
    par_rows_mut(&mut q, id, &body);
    let _ = t;
    q
}

/// attn.rs indexer_k L429-450.
fn o_indexer_k(idx: &Ds4IndexerF32, x: &[f32], d: &Ds4AttnDims, rope: &ORope) -> Vec<Vec<f32>> {
    let dm = d.dim;
    let cw = &idx.comp;
    let cd = cw.coff() * cw.head_dim;
    let kv_c = o_gemm_nt(x, &cw.wkv, dm, cd);
    let sc_c = o_gemm_nt(x, &cw.wgate, dm, cd);
    o_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio)
        .into_iter()
        .enumerate()
        .map(|(i, p)| o_comp_finish(&p, cw, d, rope, i * cw.ratio))
        .collect()
}

/// attn.rs indexer_weights L452-463.
fn o_indexer_weights(idx: &Ds4IndexerF32, x: &[f32], d: &Ds4AttnDims) -> Vec<f32> {
    let dm = d.dim;
    let mut w = o_gemm_nt(x, &idx.weights_proj, dm, d.index_n_heads);
    let c = 1.0 / (d.index_head_dim as f32).sqrt() / (d.index_n_heads as f32).sqrt();
    for v in w.iter_mut() {
        *v = o_bf16_round(o_bf16_round(*v) * c);
    }
    w
}

/// attn.rs indexer_scores L465-492(토큰 병렬).
fn o_indexer_scores(iq: &[f32], ki: &[Vec<f32>], w: &[f32], d: &Ds4AttnDims) -> Vec<Vec<f32>> {
    let (ih, id) = (d.index_n_heads, d.index_head_dim);
    let t = w.len() / ih;
    let nb = ki.len();
    let mut out = vec![vec![0.0f32; nb]; t];
    let body = |lo: usize, hi: usize, part: &mut [Vec<f32>]| {
        for ti in lo..hi {
            for (bi, kb) in ki.iter().enumerate() {
                let mut sum = 0.0f32;
                for h in 0..ih {
                    let qh = &iq[ti * ih * id + h * id..ti * ih * id + (h + 1) * id];
                    let mut dot = 0.0f32;
                    for dd in 0..id {
                        dot += qh[dd] * kb[dd];
                    }
                    sum += dot.max(0.0) * w[ti * ih + h];
                }
                part[ti - lo][bi] = o_bf16_round(sum);
            }
        }
    };
    par_ranges_mut(&mut out, 1, &body);
    out
}

/// attn.rs indexer_topk L494-529 — 값 내림차순·동점 낮은 인덱스(모듈 계약).
fn o_indexer_topk(
    scores: &[Vec<f32>],
    d: &Ds4AttnDims,
    ratio: usize,
    offset: usize,
) -> Vec<Vec<i32>> {
    let t = scores.len();
    let mut out = Vec::with_capacity(t);
    for (ti, row) in scores.iter().enumerate() {
        let nb = row.len();
        let k = d.index_topk.min(nb);
        let visible = (ti + 1) / ratio;
        let mut cand: Vec<(f32, usize)> = row
            .iter()
            .enumerate()
            .filter(|&(b, _)| b < visible)
            .map(|(b, &s)| (s, b))
            .collect();
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        out.push(
            cand.into_iter()
                .take(k)
                .map(|(_, b)| (b + offset) as i32)
                .collect(),
        );
    }
    out
}

/// attn.rs sparse_attn_one L531-580.
fn o_sparse_attn_one(
    q: &[f32],
    kv: &[f32],
    idxs: &[i32],
    sink: &[f32],
    d: &Ds4AttnDims,
) -> Vec<f32> {
    let (nh, hd) = (d.n_heads, d.head_dim);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0.0f32; nh * hd];
    let n_idx = idxs.len();
    for h in 0..nh {
        let qh = &q[h * hd..(h + 1) * hd];
        let mut s = vec![f32::NEG_INFINITY; n_idx];
        for (e, &ix) in idxs.iter().enumerate() {
            if ix < 0 {
                continue;
            }
            let krow = &kv[ix as usize * hd..(ix as usize + 1) * hd];
            let mut dot = 0.0f32;
            for dd in 0..hd {
                dot += qh[dd] * krow[dd];
            }
            s[e] = dot * scale;
        }
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = o_exp_cr(sink[h] - m);
        let mut acc = vec![0.0f32; hd];
        for (e, &se) in s.iter().enumerate() {
            if se == f32::NEG_INFINITY {
                continue;
            }
            let p = o_exp_cr(se - m);
            denom += p;
            let p16 = o_bf16_round(p);
            if idxs[e] < 0 {
                continue;
            }
            let krow = &kv[idxs[e] as usize * hd..(idxs[e] as usize + 1) * hd];
            for dd in 0..hd {
                acc[dd] += p16 * krow[dd];
            }
        }
        for dd in 0..hd {
            o[h * hd + dd] = o_bf16_round(acc[dd] / denom);
        }
    }
    o
}

/// attn.rs attention_output L582-599(전 t 묶음 — latents fp8 블록이 행=8192
/// 전체라 토큰 일괄 처리가 원본 형상).
fn o_attention_output(o: &[f32], lay: &Ds4LayerF32, d: &Ds4AttnDims) -> Vec<f32> {
    let (dm, g, r) = (d.dim, d.o_groups, d.o_lora_rank);
    let hd = d.head_dim;
    let gdim = d.n_heads * hd / g;
    let t = o.len() / (d.n_heads * hd);
    let mut latents = vec![0.0f32; t * g * r];
    let body = |lo: usize, hi: usize, part: &mut [f32]| {
        for gt in lo..hi {
            let (gi, ti) = (gt % g, gt / g);
            let base = ti * d.n_heads * hd;
            let og = &o[base + gi * gdim..base + (gi + 1) * gdim];
            let got = o_gemm_row(og, &lay.wo_a[gi], gdim, r);
            part[(gt - lo) * r..(gt - lo + 1) * r].copy_from_slice(&got);
        }
    };
    par_ranges_mut(&mut latents, r, &body);
    for v in latents.iter_mut() {
        *v = o_bf16_round(*v);
    }
    o_fp8_sim(&mut latents, g * r, 128);
    let mut y = o_gemm_nt(&latents, &lay.wo_b, g * r, dm);
    for v in y.iter_mut() {
        *v = o_bf16_round(*v);
    }
    y
}

/// attn.rs attention_forward L601-660 전체 미러(토큰 스트립 병렬).
fn o_attention_forward(
    lay: &Ds4LayerF32,
    x: &[f32],
    d: &Ds4AttnDims,
    rope: &ORope,
    ratio: usize,
) -> Vec<f32> {
    let (dm, t) = (d.dim, x.len() / d.dim);
    let (c_q, mut q) = o_project_q(lay, x, d);
    o_rope_q(&mut q, d, rope);
    let kv = o_project_kv(lay, x, d, rope);
    let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
    let mut kv_all = kv;
    let mut comp_sel: Vec<Vec<i32>> = vec![Vec::new(); t];
    if let Some(cw) = &lay.comp {
        let cd = cw.coff() * cw.head_dim;
        let kv_c = o_gemm_nt(x, &cw.wkv, dm, cd);
        let sc_c = o_gemm_nt(x, &cw.wgate, dm, cd);
        let pooled = o_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio);
        for (i, p) in pooled.into_iter().enumerate() {
            let e = o_comp_finish(&p, cw, d, rope, i * cw.ratio);
            kv_all.extend(e);
        }
        match &lay.idx {
            Some(idx) => {
                let iq = o_indexer_q(idx, &c_q, d, rope);
                let ki = o_indexer_k(idx, x, d, rope);
                let w = o_indexer_weights(idx, x, d);
                let scores = o_indexer_scores(&iq, &ki, &w, d);
                comp_sel = o_indexer_topk(&scores, d, ratio, t);
            }
            None => {
                for ti in 0..t {
                    comp_sel[ti] = o_compress_idx_dense(ti, ratio, t);
                }
            }
        }
    }
    let mut out = vec![0.0f32; t * dm];
    let win = d.window;
    let body = |lo: usize, hi: usize, part: &mut [f32]| {
        for ti in lo..hi {
            let mut idxs = o_window_idx_prefill(ti, win);
            idxs.extend(comp_sel[ti].iter().copied());
            let q_t = &q[ti * nh * hd..(ti + 1) * nh * hd];
            let mut o = o_sparse_attn_one(q_t, &kv_all, &idxs, &lay.sink, d);
            for h in 0..nh {
                let base = h * hd;
                o_rope_apply(
                    &mut o[base + hd - rd..base + hd],
                    &rope.cs[ti * rope.half * 2..(ti + 1) * rope.half * 2],
                    true,
                );
                for v in o[base + hd - rd..base + hd].iter_mut() {
                    *v = o_bf16_round(*v);
                }
            }
            let y = o_attn_output_tok(&o, lay, d);
            part[(ti - lo) * dm..(ti - lo + 1) * dm].copy_from_slice(&y);
        }
    };
    par_ranges_mut(&mut out, dm, &body);
    out
}

/// 토큰 1개 그룹 출력(attention_output 의 t=1 형 — 산술 동일, latents 행
/// 폭 g·r 도 동일해 fp8 블록 경계 불변).
fn o_attn_output_tok(o: &[f32], lay: &Ds4LayerF32, d: &Ds4AttnDims) -> Vec<f32> {
    let (dm, g, r) = (d.dim, d.o_groups, d.o_lora_rank);
    let gdim = d.n_heads * d.head_dim / g;
    let mut latents = vec![0.0f32; g * r];
    for gi in 0..g {
        let og = &o[gi * gdim..(gi + 1) * gdim];
        let got = o_gemm_row(og, &lay.wo_a[gi], gdim, r);
        latents[gi * r..(gi + 1) * r].copy_from_slice(&got);
    }
    for v in latents.iter_mut() {
        *v = o_bf16_round(*v);
    }
    o_fp8_sim(&mut latents, g * r, 128);
    let mut y = o_gemm_row(&latents, &lay.wo_b, g * r, dm);
    for v in y.iter_mut() {
        *v = o_bf16_round(*v);
    }
    y
}

// ── trellis 디양자화 미러(trellis.rs 직이식 — 오프셋 직독) ──

/// trellis.rs PERM_INV L24-42.
const PERM_INV: [u16; 256] = {
    let mut inv = [0u16; 256];
    let mut t = 0;
    while t < 32 {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let mut s = 0;
        while s < 8 {
            let r = r0 + [0, 1, 8, 9, 0, 1, 8, 9][s];
            let c = c0 + if s < 4 { 0 } else { 8 };
            inv[r * 16 + c] = (t * 8 + s) as u16;
            s += 1;
        }
        t += 1;
    }
    inv
};

/// trellis.rs mul1_decode L47-52.
fn o_mul1_decode(word: u16) -> f32 {
    const MUL1_MULT: u32 = 0x83DCD12D;
    const K_INV: f32 = 0.00676727294921875;
    const K_BIAS: f32 = -10.3828125;
    let x = (word as u32).wrapping_mul(MUL1_MULT);
    let sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    let f = 1024.0f32 + sum as f32;
    f.mul_add(K_INV, K_BIAS)
}

/// trellis.rs tile_word L60-69.
fn o_tile_word(u32s: &[u32], krate: u32, t: u32) -> u16 {
    let words32 = 8 * krate as usize;
    let b0 = (t * krate + (krate + 256 * krate - 16)) as usize;
    let b1 = b0 + 16;
    let i0 = (b0 / 32) % words32;
    let i1 = ((b1 - 1) / 32) % words32;
    let s = ((b1 - 1) / 32 + 1) * 32 - b1;
    let merged = ((u32s[i0] as u64) << 32) | u32s[i1] as u64;
    ((merged >> s) & 0xFFFF) as u16
}

/// trellis.rs had128(f64) L279-297.
fn o_had128_f64(v: &mut [f64]) {
    let mut width = 1;
    while width < 128 {
        let mut base = 0;
        while base < 128 {
            for i in 0..width {
                let a = v[base + i];
                let b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
            base += 2 * width;
        }
        width *= 2;
    }
}

/// trellis 원시 3중(u32 재해석 완료형).
struct LinearMirror {
    tre_u32: Vec<u32>,
    suh: Vec<u8>,
    svh: Vec<u8>,
    n: usize,
    krate: u32,
    ntiles: usize,
}

fn o_f16le(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]])
}

/// trellis.rs dequant_view L128-167 — 원 기저 128×128 블록(f64).
fn o_dequant_block(w: &LinearMirror, k0: usize, n0: usize) -> Vec<f64> {
    let (ck0, cn0) = ((k0 % 128) / 16, (n0 % 128) / 16);
    let mut full = vec![0f64; 128 * 128];
    let mut tile = [0f32; 256];
    let words32 = 8 * w.krate as usize;
    for kt in 0..8 {
        for nt in 0..8 {
            let base = ((k0 / 16 - ck0 + kt) * w.ntiles + (n0 / 16 - cn0 + nt)) * words32;
            let tw = &w.tre_u32[base..base + words32];
            for pos in 0..256usize {
                let t = PERM_INV[pos] as u32;
                tile[pos] = o_mul1_decode(o_tile_word(tw, w.krate, t));
            }
            for r in 0..16 {
                for c in 0..16 {
                    full[(kt * 16 + r) * 128 + nt * 16 + c] = tile[r * 16 + c] as f64;
                }
            }
        }
    }
    for row in full.chunks_mut(128) {
        o_had128_f64(row);
    }
    let mut col = [0f64; 128];
    for c in 0..128 {
        for (r, v) in col.iter_mut().enumerate() {
            *v = full[r * 128 + c];
        }
        o_had128_f64(&mut col);
        for (r, v) in col.iter().enumerate() {
            full[r * 128 + c] = v / 128.0;
        }
    }
    let mut out = vec![0f64; 128 * 128];
    for i in 0..128 {
        let si = f16_to_f32(o_f16le(&w.suh, k0 + i)) as f64;
        for j in 0..128 {
            let sj = f16_to_f32(o_f16le(&w.svh, n0 + j)) as f64;
            out[i * 128 + j] = si * sj * full[(ck0 * 16 + i) * 128 + cn0 * 16 + j];
        }
    }
    out
}

/// loader.rs linear L344-381 — 스트립 병렬 디양자화 → [k][n] f32 k-major.
fn dequant_linear(ar: &StArchive, key: &str) -> Result<Vec<f32>, String> {
    let shape = ar
        .shape_of(&format!("{key}.trellis"))
        .ok_or_else(|| format!("{key}.trellis 없음"))?
        .to_vec();
    if shape.len() != 3 {
        return Err(format!("{key}: trellis dim {}", shape.len()));
    }
    let (kt, nt, tw) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    if tw % 16 != 0 {
        return Err(format!("{key}: 반정수 bpw(tw={tw}) 미지원"));
    }
    let (k, n, krate) = (kt * 16, nt * 16, (tw / 16) as u32);
    let tre = ar.read(&format!("{key}.trellis"))?;
    let suh = ar.read(&format!("{key}.suh"))?;
    let svh = ar.read(&format!("{key}.svh"))?;
    let tre_u32: Vec<u32> = tre
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let lin = LinearMirror {
        tre_u32,
        suh,
        svh,
        n,
        krate,
        ntiles: nt,
    };
    let mut w = vec![0.0f32; k * n];
    let n_ = n;
    let body = |lo: usize, hi: usize, part: &mut [f32]| {
        for sc in lo..hi {
            let kc = sc * 128;
            for n0 in (0..n_).step_by(128) {
                let blk = o_dequant_block(&lin, kc, n0);
                for i in 0..128 {
                    let dst = &mut part
                        [((kc + i) - lo * 128) * n_ + n0..((kc + i) - lo * 128) * n_ + n0 + 128];
                    for (j, &vv) in blk[i * 128..(i + 1) * 128].iter().enumerate() {
                        dst[j] = vv as f32;
                    }
                }
            }
        }
    };
    par_ranges_mut(&mut w, 128 * n, &body);
    Ok(w)
}

/// loader.rs conv/plain_f32 L156-172 — plain 텐서 전체 → f32(dtype 분기).
fn plain_f32(ar: &StArchive, name: &str) -> Result<Vec<f32>, String> {
    let raw = ar.read(name)?;
    let dt = ar
        .dtype_of(name)
        .ok_or_else(|| format!("{name}: dtype 없음"))?;
    Ok(match dt {
        0 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        2 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        1 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        other => return Err(format!("{name}: dtype 코드 {other} 미지원")),
    })
}

/// loader.rs plain_kmat L385-400 — 행우선 [out][in] → k-major 전치.
fn plain_kmat(ar: &StArchive, name: &str) -> Result<Vec<f32>, String> {
    let shape = ar
        .shape_of(name)
        .ok_or_else(|| format!("{name} 없음"))?
        .to_vec();
    if shape.len() != 2 {
        return Err(format!("{name}: 2차원 아님"));
    }
    let (n, k) = (shape[0] as usize, shape[1] as usize);
    let src = plain_f32(ar, name)?;
    let mut w = vec![0.0f32; k * n];
    // k-행 병렬(연속 쓰기 — 원래 n-축 순회는 스트립 분할이 불가).
    let body = |lo: usize, hi: usize, part: &mut [f32]| {
        for kk in lo..hi {
            for o in 0..n {
                part[(kk - lo) * n + o] = src[o * k + kk];
            }
        }
    };
    par_ranges_mut(&mut w, n, &body);
    Ok(w)
}

/// loader.rs embed_rows L437-447 — 임베딩 행 슬라이스 오프셋 직독(index.json
/// → 샤드 헤더 → pread — 전체 적재 금지 계약).
fn embed_rows(dir: &Path, ids: &[u32]) -> Result<Vec<f32>, String> {
    let idx_path = dir.join("model.safetensors.index.json");
    let raw =
        std::fs::read_to_string(&idx_path).map_err(|e| format!("{}: {e}", idx_path.display()))?;
    let v = JParser {
        b: raw.as_bytes(),
        p: 0,
    }
    .parse()?;
    let shard_name = v
        .get("weight_map")
        .and_then(JVal::as_obj)
        .and_then(|m| {
            m.iter()
                .find(|(k, _)| k == "embed.weight")
                .map(|(_, val)| val.as_str().unwrap_or("").to_string())
        })
        .ok_or("index.json: embed.weight 매핑 없음")?;
    let path = dir.join(&shard_name);
    let mut f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    use std::io::{Read, Seek, SeekFrom};
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb).map_err(|e| e.to_string())?;
    let hlen = u64::from_le_bytes(lenb);
    if hlen == 0 || hlen > (1 << 30) {
        return Err(format!("embed 헤더 길이 {hlen}"));
    }
    let mut hb = vec![0u8; hlen as usize];
    f.read_exact(&mut hb).map_err(|e| e.to_string())?;
    let data_base = 8 + hlen;
    let hv = JParser { b: &hb, p: 0 }.parse()?;
    let tv = hv
        .as_obj()
        .and_then(|m| m.iter().find(|(k, _)| k == "embed.weight"))
        .map(|(_, v)| v)
        .ok_or("샤드 헤더: embed.weight 없음")?;
    let dt = tv
        .get("dtype")
        .and_then(JVal::as_str)
        .ok_or("embed.weight: dtype 없음")?;
    if dt != "BF16" {
        return Err(format!("embed.weight dtype {dt} — BF16 고정(픽스처 가드)"));
    }
    let shape = tv
        .get("shape")
        .and_then(JVal::as_arr)
        .ok_or("embed.weight: shape 없음")?;
    let cols = shape
        .get(1)
        .and_then(JVal::as_f64)
        .ok_or("embed.weight: shape[1] 없음")? as usize;
    let offs = tv
        .get("data_offsets")
        .and_then(JVal::as_arr)
        .ok_or("embed.weight: data_offsets 없음")?;
    let begin = offs
        .first()
        .and_then(JVal::as_f64)
        .ok_or("embed.weight: begin 없음")? as u64;
    let mut out = Vec::with_capacity(ids.len() * cols);
    for &id in ids {
        f.seek(SeekFrom::Start(
            data_base + begin + id as u64 * cols as u64 * 2,
        ))
        .map_err(|e| e.to_string())?;
        let mut rowb = vec![0u8; cols * 2];
        f.read_exact(&mut rowb).map_err(|e| e.to_string())?;
        out.extend(
            rowb.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16)),
        );
    }
    Ok(out)
}

/// 층 가중치 적재(loader.rs block L344-381 attn 부분 미러 — 모듈·오라클 공유).
fn load_layer(ar: &StArchive, d: &Ds4AttnDims, il: usize) -> Result<Ds4LayerF32, String> {
    let p = format!("layers.{il}.attn");
    let wo_a = (0..d.o_groups)
        .map(|g| dequant_linear(ar, &format!("{p}.wo_a.slice.{g}")))
        .collect::<Result<Vec<_>, _>>()?;
    let comp = match d.kind(il)? {
        crate::rawcuda::ds4_attn_cuda::Ds4Kind::Swa => None,
        _ => Some(Ds4CompF32 {
            wkv: dequant_linear(ar, &format!("{p}.compressor.wkv"))?,
            wgate: dequant_linear(ar, &format!("{p}.compressor.wgate"))?,
            ape: plain_f32(ar, &format!("{p}.compressor.ape"))?,
            norm: plain_f32(ar, &format!("{p}.compressor.norm.weight"))?,
            head_dim: d.head_dim,
            ratio: d.ratio(il),
            rotate: false,
        }),
    };
    let idx = if d.kind(il)? == crate::rawcuda::ds4_attn_cuda::Ds4Kind::Csa {
        Some(Ds4IndexerF32 {
            wq_b: dequant_linear(ar, &format!("{p}.indexer.wq_b"))?,
            weights_proj: plain_kmat(ar, &format!("{p}.indexer.weights_proj.weight"))?,
            comp: Ds4CompF32 {
                wkv: dequant_linear(ar, &format!("{p}.indexer.compressor.wkv"))?,
                wgate: dequant_linear(ar, &format!("{p}.indexer.compressor.wgate"))?,
                ape: plain_f32(ar, &format!("{p}.indexer.compressor.ape"))?,
                norm: plain_f32(ar, &format!("{p}.indexer.compressor.norm.weight"))?,
                head_dim: d.index_head_dim,
                ratio: d.ratio(il),
                rotate: true,
            },
        })
    } else {
        None
    };
    Ok(Ds4LayerF32 {
        wq_a: dequant_linear(ar, &format!("{p}.wq_a"))?,
        q_norm: plain_f32(ar, &format!("{p}.q_norm.weight"))?,
        wq_b: dequant_linear(ar, &format!("{p}.wq_b"))?,
        wkv: dequant_linear(ar, &format!("{p}.wkv"))?,
        kv_norm: plain_f32(ar, &format!("{p}.kv_norm.weight"))?,
        sink: plain_f32(ar, &format!("{p}.attn_sink"))?,
        wo_a,
        wo_b: dequant_linear(ar, &format!("{p}.wo_b"))?,
        comp,
        idx,
    })
}

/// rope 인자(config.json) — 압축층 160000+YaRN / L0-1 10000 단독(§6).
struct RopeArgs {
    base: f64,
    yarn: bool,
    factor: f64,
    orig_len: usize,
    beta_fast: f64,
    beta_slow: f64,
}

fn rope_args(cfg_text: &str, d: &Ds4AttnDims, il: usize) -> RopeArgs {
    let v = JParser {
        b: cfg_text.as_bytes(),
        p: 0,
    }
    .parse()
    .unwrap_or(JVal::Nul);
    let g = |k: &str| v.get(k).and_then(JVal::as_f64);
    let compress = g("compress_rope_theta").unwrap_or(160000.0);
    let plain = g("rope_theta").unwrap_or(10000.0);
    let rs = v.get("rope_scaling");
    let is_yarn = rs
        .and_then(|r| r.get("type"))
        .and_then(JVal::as_str)
        .is_some_and(|t| t == "yarn");
    if d.ratio(il) != 0 {
        RopeArgs {
            base: compress,
            yarn: is_yarn,
            factor: rs
                .and_then(|r| r.get("factor"))
                .and_then(JVal::as_f64)
                .unwrap_or(1.0),
            orig_len: rs
                .and_then(|r| r.get("original_max_position_embeddings"))
                .and_then(JVal::as_f64)
                .unwrap_or(0.0) as usize,
            beta_fast: rs
                .and_then(|r| r.get("beta_fast"))
                .and_then(JVal::as_f64)
                .unwrap_or(32.0),
            beta_slow: rs
                .and_then(|r| r.get("beta_slow"))
                .and_then(JVal::as_f64)
                .unwrap_or(1.0),
        }
    } else {
        RopeArgs {
            base: plain,
            yarn: false,
            factor: 1.0,
            orig_len: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// 비트 불일치 수(to_bits 동일 판정).
fn bitdiff(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count()
}

/// 선택 리스트 동일성(순서 포함 — 안정 정렬 계약).
fn sel_diff(got: &[i32], want: &[Vec<i32>], topk: usize) -> usize {
    let mut n = 0usize;
    for (ti, wrow) in want.iter().enumerate() {
        let grow: Vec<i32> = got[ti * topk..(ti + 1) * topk]
            .iter()
            .copied()
            .filter(|&v| v >= 0)
            .collect();
        if grow != *wrow {
            n += 1;
        }
    }
    n
}

/// 스테이지 판정 보고 1줄 + 실패 적립.
fn judge(
    dev: &str,
    tag: &str,
    got: &[f32],
    want: &[f32],
    fails: &mut Vec<String>,
    report: &mut String,
) -> bool {
    let (md, nan) = maxdiff_nan(got, want);
    let bd = bitdiff(got, want);
    let pass = bd == 0 && nan == 0;
    println!(
        "device: {dev} | ds4-attn {tag}: maxdiff={md:.3e} bitdiff={bd}/{} nan={nan} | {}",
        want.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · {tag} {md:.3e}/{bd}b"));
    if !pass {
        fails.push(format!("{tag} maxdiff={md:.3e} bitdiff={bd} nan={nan}"));
    }
    pass
}

/// ds4-attn — 3층 유형(SWA L0·CSA L2·HCA L3) 전 스테이지 값 판정.
/// 층별 T: L0=300(윈도우)·L2=2052(nb 513 ≥ topk 512 — 절단 실측)·
/// L3=1032(nb 8 밀도). bitdiff=0 기준.
pub fn cuda_ds4_check(dir: &str) -> Result<String, String> {
    let dir = PathBuf::from(dir);
    let cfg_path = dir.join("config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let d = Ds4AttnDims::from_config(&cfg_text)?;
    let ar = StArchive::open(&dir)?;
    let mut modl = Ds4AttnCuda::new(d.clone())?;
    let dev = modl.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    for (il, t) in [(0usize, 300usize), (2, 2052), (3, 1032)] {
        let kind = d.kind(il)?;
        let kind_tag = match kind {
            crate::rawcuda::ds4_attn_cuda::Ds4Kind::Swa => "SWA",
            crate::rawcuda::ds4_attn_cuda::Ds4Kind::Csa => "CSA",
            crate::rawcuda::ds4_attn_cuda::Ds4Kind::Hca => "HCA",
        };
        let lay = load_layer(&ar, &d, il)?;
        modl.register_layer(il, &lay)?;
        let ra = rope_args(&cfg_text, &d, il);
        let rope = o_rope_build(
            d.rope_head_dim,
            t,
            ra.base,
            ra.yarn,
            ra.factor,
            ra.orig_len,
            ra.beta_fast,
            ra.beta_slow,
        );
        modl.set_rope(il, &rope.cs, rope.half)?;
        let ids: Vec<u32> = (0..t as u32).collect();
        let x = embed_rows(&dir, &ids)?;

        // 오라클 전 스테이지.
        let (o_cq, mut o_q) = o_project_q(&lay, &x, &d);
        o_rope_q(&mut o_q, &d, &rope);
        let o_kv = o_project_kv(&lay, &x, &d, &rope);
        let ratio = d.ratio(il);
        let mut o_kv_all = o_kv.clone();
        let o_sel: Vec<Vec<i32>> = match kind {
            crate::rawcuda::ds4_attn_cuda::Ds4Kind::Swa => vec![Vec::new(); t],
            _ => {
                let cw = lay.comp.as_ref().unwrap();
                let cd = cw.coff() * cw.head_dim;
                let kv_c = o_gemm_nt(&x, &cw.wkv, d.dim, cd);
                let sc_c = o_gemm_nt(&x, &cw.wgate, d.dim, cd);
                let pooled = o_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio);
                for (i, p) in pooled.iter().enumerate() {
                    let e = o_comp_finish(p, cw, &d, &rope, i * cw.ratio);
                    o_kv_all.extend(e);
                }
                match &lay.idx {
                    Some(idx) => {
                        let iq = o_indexer_q(idx, &o_cq, &d, &rope);
                        let ki = o_indexer_k(idx, &x, &d, &rope);
                        let w = o_indexer_weights(idx, &x, &d);
                        let scores = o_indexer_scores(&iq, &ki, &w, &d);
                        o_indexer_topk(&scores, &d, ratio, t)
                    }
                    None => (0..t)
                        .map(|ti| o_compress_idx_dense(ti, ratio, t))
                        .collect(),
                }
            }
        };
        // 토큰별 스파스+비회전 o(오라클).
        let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
        let mut o_o = vec![0.0f32; t * nh * hd];
        {
            let sel = &o_sel;
            let body = |lo: usize, hi: usize, part: &mut [f32]| {
                for ti in lo..hi {
                    let mut idxs = o_window_idx_prefill(ti, d.window);
                    idxs.extend(sel[ti].iter().copied());
                    let q_t = &o_q[ti * nh * hd..(ti + 1) * nh * hd];
                    let mut o = o_sparse_attn_one(q_t, &o_kv_all, &idxs, &lay.sink, &d);
                    for h in 0..nh {
                        let base = h * hd;
                        o_rope_apply(
                            &mut o[base + hd - rd..base + hd],
                            &rope.cs[ti * rope.half * 2..(ti + 1) * rope.half * 2],
                            true,
                        );
                        for v in o[base + hd - rd..base + hd].iter_mut() {
                            *v = o_bf16_round(*v);
                        }
                    }
                    part[(ti - lo) * nh * hd..(ti - lo + 1) * nh * hd].copy_from_slice(&o);
                }
            };
            par_ranges_mut(&mut o_o, nh * hd, &body);
        }
        let o_y = o_attention_output(&o_o, &lay, &d);

        // 모듈 스테이지 판정.
        let (m_cq, m_q) = modl.stage_project_q(il, &x)?;
        judge(
            &dev,
            &format!("L{il} {kind_tag} c_Q"),
            &m_cq,
            &o_cq,
            &mut fails,
            &mut report,
        );
        judge(
            &dev,
            &format!("L{il} {kind_tag} q(roped)"),
            &m_q,
            &o_q,
            &mut fails,
            &mut report,
        );
        let m_kv = modl.stage_project_kv(il, &x)?;
        judge(
            &dev,
            &format!("L{il} {kind_tag} kv"),
            &m_kv,
            &o_kv,
            &mut fails,
            &mut report,
        );
        let mut m_kv_all = m_kv.clone();
        if kind != crate::rawcuda::ds4_attn_cuda::Ds4Kind::Swa {
            let m_ent = modl.stage_compress(il, &x, None)?;
            let cw = lay.comp.as_ref().unwrap();
            let mut o_ent = Vec::new();
            {
                let cd = cw.coff() * cw.head_dim;
                let kv_c = o_gemm_nt(&x, &cw.wkv, d.dim, cd);
                let sc_c = o_gemm_nt(&x, &cw.wgate, d.dim, cd);
                for (i, p) in o_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio)
                    .into_iter()
                    .enumerate()
                {
                    o_ent.extend(o_comp_finish(&p, cw, &d, &rope, i * cw.ratio));
                }
            }
            judge(
                &dev,
                &format!("L{il} {kind_tag} comp-entries"),
                &m_ent,
                &o_ent,
                &mut fails,
                &mut report,
            );
            m_kv_all.extend(m_ent);
        }
        if let Some(_idx) = &lay.idx {
            let (m_iq, m_ki, m_iw, m_sc, m_sel) = modl.stage_indexer(il, &m_cq, &x, None)?;
            let idx = lay.idx.as_ref().unwrap();
            let o_iq = o_indexer_q(idx, &o_cq, &d, &rope);
            let o_ki_v = o_indexer_k(idx, &x, &d, &rope);
            let o_ki: Vec<f32> = o_ki_v.concat();
            let o_iw = o_indexer_weights(idx, &x, &d);
            let o_scores = o_indexer_scores(&o_iq, &o_ki_v, &o_iw, &d);
            let o_scores_flat: Vec<f32> = o_scores.concat();
            judge(
                &dev,
                &format!("L{il} {kind_tag} indexer qI"),
                &m_iq,
                &o_iq,
                &mut fails,
                &mut report,
            );
            judge(
                &dev,
                &format!("L{il} {kind_tag} indexer kI"),
                &m_ki,
                &o_ki,
                &mut fails,
                &mut report,
            );
            judge(
                &dev,
                &format!("L{il} {kind_tag} indexer w"),
                &m_iw,
                &o_iw,
                &mut fails,
                &mut report,
            );
            judge(
                &dev,
                &format!("L{il} {kind_tag} indexer scores"),
                &m_sc,
                &o_scores_flat,
                &mut fails,
                &mut report,
            );
            let sdiff = sel_diff(&m_sel, &o_sel, d.index_topk);
            let spass = sdiff == 0;
            println!(
                "device: {dev} | ds4-attn L{il} {kind_tag} indexer sel: 토큰 불일치 {sdiff}/{t} | {}",
                if spass { "PASS" } else { "FAIL" }
            );
            report.push_str(&format!(" · L{il} sel {sdiff} mismatch"));
            if !spass {
                fails.push(format!("L{il} indexer sel mismatch {sdiff}/{t}"));
            }
        }
        let m_o = modl.stage_sparse(il, &m_q, &m_kv_all, &o_sel, None)?;
        judge(
            &dev,
            &format!("L{il} {kind_tag} sparse-o(derot)"),
            &m_o,
            &o_o,
            &mut fails,
            &mut report,
        );
        let m_y = modl.stage_output(il, &m_o)?;
        judge(
            &dev,
            &format!("L{il} {kind_tag} y(grouped)"),
            &m_y,
            &o_y,
            &mut fails,
            &mut report,
        );
        let m_yf = modl.attention_forward(il, &x)?;
        judge(
            &dev,
            &format!("L{il} {kind_tag} y(end-to-end)"),
            &m_yf,
            &o_y,
            &mut fails,
            &mut report,
        );
    }

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | dims {}L dim {} heads {}×{} win {} topk {} | {report} | ALL PASS (비트동일 기준)",
            d.n_layers, d.dim, d.n_heads, d.head_dim, d.window, d.index_topk
        ))
    } else {
        Err(format!(
            "ds4-attn 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// ds4-attn-neg — 음성대조 3종(원장 17호). L2(CSA) T=64:
/// (a) top-k 인과 off-by-one: sel 리스트 불일치 토큰 수 > 0(구조 보장 —
///     visible+1 이 k 미만인 토큰은 선택 길이 자체가 +1).
/// (b) ape 미스얼라인: comp 엔트리 maxdiff > DS4_THRESH.
/// (c) 싱크 누락: sparse o bitdiff > 0(bf16 경계 플립).
/// 셋 다 탐지되면 NEG-DETECTED(비영 exit).
pub fn cuda_ds4_negative_check(dir: &str) -> Result<String, String> {
    let dir = PathBuf::from(dir);
    let cfg_path = dir.join("config.json");
    let cfg_text =
        std::fs::read_to_string(&cfg_path).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
    let d = Ds4AttnDims::from_config(&cfg_text)?;
    let ar = StArchive::open(&dir)?;
    let mut modl = Ds4AttnCuda::new(d.clone())?;
    let dev = modl.device_name().to_string();
    let (il, t) = (2usize, 64usize);
    let lay = load_layer(&ar, &d, il)?;
    modl.register_layer(il, &lay)?;
    let ra = rope_args(&cfg_text, &d, il);
    let rope = o_rope_build(
        d.rope_head_dim,
        t,
        ra.base,
        ra.yarn,
        ra.factor,
        ra.orig_len,
        ra.beta_fast,
        ra.beta_slow,
    );
    modl.set_rope(il, &rope.cs, rope.half)?;
    let ids: Vec<u32> = (0..t as u32).collect();
    let x = embed_rows(&dir, &ids)?;
    let ratio = d.ratio(il);

    // 오라클.
    let (o_cq, mut o_q) = o_project_q(&lay, &x, &d);
    o_rope_q(&mut o_q, &d, &rope);
    let idx = lay.idx.as_ref().expect("L2 인덱서");
    let o_iq = o_indexer_q(idx, &o_cq, &d, &rope);
    let o_ki = o_indexer_k(idx, &x, &d, &rope);
    let o_iw = o_indexer_weights(idx, &x, &d);
    let o_scores = o_indexer_scores(&o_iq, &o_ki, &o_iw, &d);
    let o_sel = o_indexer_topk(&o_scores, &d, ratio, t);
    let o_kv = o_project_kv(&lay, &x, &d, &rope);
    let cw = lay.comp.as_ref().unwrap();
    let cd = cw.coff() * cw.head_dim;
    let kv_c = o_gemm_nt(&x, &cw.wkv, d.dim, cd);
    let sc_c = o_gemm_nt(&x, &cw.wgate, d.dim, cd);
    let mut o_ent = Vec::new();
    let mut o_kv_all = o_kv.clone();
    for (i, p) in o_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio)
        .into_iter()
        .enumerate()
    {
        let e = o_comp_finish(&p, cw, &d, &rope, i * cw.ratio);
        o_kv_all.extend(e.clone());
        o_ent.extend(e);
    }
    let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
    let mut o_o = vec![0.0f32; t * nh * hd];
    {
        let sel = &o_sel;
        let body = |lo: usize, hi: usize, part: &mut [f32]| {
            for ti in lo..hi {
                let mut idxs = o_window_idx_prefill(ti, d.window);
                idxs.extend(sel[ti].iter().copied());
                let q_t = &o_q[ti * nh * hd..(ti + 1) * nh * hd];
                let mut o = o_sparse_attn_one(q_t, &o_kv_all, &idxs, &lay.sink, &d);
                for h in 0..nh {
                    let base = h * hd;
                    o_rope_apply(
                        &mut o[base + hd - rd..base + hd],
                        &rope.cs[ti * rope.half * 2..(ti + 1) * rope.half * 2],
                        true,
                    );
                    for v in o[base + hd - rd..base + hd].iter_mut() {
                        *v = o_bf16_round(*v);
                    }
                }
                part[(ti - lo) * nh * hd..(ti - lo + 1) * nh * hd].copy_from_slice(&o);
            }
        };
        par_ranges_mut(&mut o_o, nh * hd, &body);
    }

    // (a) top-k off-by-one — 모듈 쌍둥이 sel vs 오라클 sel.
    let (_, _, _, _, m_sel) = modl.stage_indexer(il, &o_cq, &x, Some(Ds4Neg::TopkVis1))?;
    let sa = sel_diff(&m_sel, &o_sel, d.index_topk);
    println!(
        "device: {dev} | ds4-attn-neg (a) topk visible+1: sel mismatch {sa}/{t} 토큰 | FAIL(expected)"
    );
    let det_a = sa > 0;

    // (b) ape 미스얼라인 — 쌍둥이 comp 엔트리 vs 오라클.
    let m_ent = modl.stage_compress(il, &x, Some(Ds4Neg::ApeOff))?;
    let (mdb, nanb) = maxdiff_nan(&m_ent, &o_ent);
    let bdb = bitdiff(&m_ent, &o_ent);
    println!(
        "device: {dev} | ds4-attn-neg (b) ape misalign: comp maxdiff={mdb:.3e} bitdiff={bdb}/{} nan={nanb} | FAIL(expected)",
        o_ent.len()
    );
    let det_b = mdb > DS4_THRESH;

    // (c) 싱크 누락 — 쌍둥이 sparse o vs 오라클(bf16 경계 플립 판정).
    let m_o = modl.stage_sparse(il, &o_q, &o_kv_all, &o_sel, Some(Ds4Neg::SinkDropped))?;
    let (mdc, nanc) = maxdiff_nan(&m_o, &o_o);
    let bdc = bitdiff(&m_o, &o_o);
    println!(
        "device: {dev} | ds4-attn-neg (c) sink dropped: sparse-o maxdiff={mdc:.3e} bitdiff={bdc}/{} nan={nanc} | FAIL(expected)",
        o_o.len()
    );
    let det_c = bdc > 0 || mdc > DS4_THRESH;

    if det_a && det_b && det_c {
        Err(format!(
            "NEG-DETECTED (a) sel mismatch {sa}/{t} (b) comp maxdiff={mdb:.3e}>{DS4_THRESH:.0e} (c) sparse-o bitdiff={bdc} maxdiff={mdc:.3e} — 검증계기 정상(인과 경계·ape·싱크 편차 모두 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={sa} (b)={mdb:.3e} (c) bitdiff={bdc}/{mdc:.3e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
