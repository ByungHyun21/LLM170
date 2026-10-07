//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: ds4-mtp 케이스마다 잔류·가중치를
//!    재업로드한다(모듈 작업 버퍼는 현 행 전체를 매 체인마다 덮어쓴다).
//! ② 형상은 픽스처 config.json에서 자동 확정(Vision-Exp 실측 dim 4096 ·
//!    vocab 129280 · hc_mult 4 · window 128 · block_size 5 · noise 128799
//!    · markov_rank 256 · nextn 3)+ 트렁크 타깃 평균 t=8 토큰.
//! ③ 캡처-재생: 실측 픽스처 가중치(EXL3 샤드 safetensors — 헤더 판독 후
//!    data_offsets 오프셋 직독, 전량 적재 금지) + 합성 main_hidden
//!    (트렁크 40-42 스트림 평균 계급)으로 구동. mtp.* 텐서 실측 존재
//!    확인(2026-10-05): mtp.0.{main_proj,main_norm,attn.*,hc_*,ffn.*} ·
//!    mtp.2.{hc_head_*,norm,markov_head.*,confidence_head.proj}.
//! ④ 종단 값이 유일 불변량: 판정은 DSpark 체인 산출의 값 maxdiff+
//!    비트 불일치 수. main_x·링·디코드 어텐션·attn 서브블록·마르코프·
//!    신뢰도는 bitdiff=0 기준, MoE 이후(hmix·h·로짓)는 문서화 임계
//!    2e-3(ds4-moe 라우트 가중치 계급 승계 — ds4_moe ROUTE_W 1e-6 원장).
//!
//! [오라클 — core deepseek4 참조 직이식(값 판정의 유일 기준, plans/124
//! §6). 인용은 전부 워크트리 3f1ee12d 기준 줄번호, 2026-10-05]
//! - crates/core/src/deepseek4/frame.rs — dspark_main_x L208-230 ·
//!   dspark_draft_ids L233-239 · dspark_window_warm L242-263 ·
//!   dspark_decode_attn L267-345(project_kv_unroped L349-371 포함) ·
//!   markov_logits_bias L373-383 · confidence_score L386-397 ·
//!   head_gemv_stripwise L41-77.
//! - crates/core/src/deepseek4/layers.rs — block_forward L28-121(드래프트
//!   블록 배선 — 어텐션만 dspark_decode_attn 치환).
//! - crates/core/src/deepseek4/stages/attn.rs — project_q L84-120 ·
//!   rope_q L122-135 · project_kv L137-158 · sparse_attn_one L531-580 ·
//!   attention_output L582-599.
//! - crates/core/src/deepseek4/stages/hc.rs — hc_split_sinkhorn L31-112 ·
//!   hc_pre L115-152 · hc_post L155-181 · hc_head L185-214.
//! - crates/core/src/deepseek4/stages/moe.rs — gate_scores L42-50 ·
//!   topk_stable L53-65 · route L67-79 · route_routed L87-96 ·
//!   expert_ffn L98-131 · moe_forward L133-216.
//! - crates/core/src/deepseek4/ops.rs — rms_scale L48-55 · bf16_round
//!   L17-22 · fp8_sim L174-192 · rope_apply L335-350 · RopeTable::build
//!   L280-325(호스트 표 구축).
//! - crates/core/src/ops.rs — exp_cr L52-119(f64 fma 호너 13단).
//! - crates/exl3/src/trellis.rs — mul1_decode L47-58 · tile_word L60-70 ·
//!   had128 L279-297 · dequant_view L128-167(헤드 스트립 디양자화).
//! - crates/core/src/deepseek4/loader.rs — dspark_main L431-437 ·
//!   dspark_head_parts L439-449(mtp 텐서명 계약) · markov_w1_row L451-465.
//!
//! [DSpark 체인 계약 — 보고서 §7 + core frame.rs 헤더]
//! main_hidden = 트렁크 40-42(dspark_targets) 4스트림 평균 concat(bf16
//! 경계) 인터리브 [t][3·4096] → main_proj/main_norm(mtp.0) → main_x →
//! 윈도우 링 워밍 → 드래프트 입력 [token, noise×4](block 5, id 128799)
//! 공유 embed 4스트림 방송 → 풀 블록(mtp.0 — 윈도우 전용 어텐션
//! compress_ratio 0, mHC 배선, noaux_tc MoE) → hc_head(mtp.2) → norm
//! (mtp.2) → 공유 head 스트립 gemv → 로짓[5] → 마르코프 바이어스
//! w2(w1(out_ids)) 가산 → 6토큰(트렁크 1 + 드래프트 5 — §7 "6토큰").
//! 신뢰도 proj [4096+256→1] f32 on [h, w1(out_ids)].
//!
//! [정합 원장 요약] (i) main_x 비트동일 · (ii) 링 워밍 비트동일 ·
//! (iii) 디코드 어텐션(q/kv/스파스/비회전/그룹 출력) 비트동일 ·
//! (iv) attn 서브블록 X' 비트동일 · (v) MoE 이후 hmix/h/로짓 ≤ 2e-3 ·
//! (vi) 마르코프 바이어스 비트동일 · (vii) 신뢰도(동일 h 입력) 비트동일.
//!
//! [음성대조 — 원장 17호(계기 자체 검증)] 3종 NEG-DETECTED 필(정밀
//! 체인에서는 로짓 레벨, 축소 팔에서는 검출 최점 레벨):
//! (a) 트렁크 타깃 층 순열(main_hidden 평면 [40,41,42]→[41,40,42] —
//!     main_proj 열그룹 오배열) · (b) 마르코프 바이어스 누락(로짓 편차
//!     = |bias|) · (c) noise 토큰 id 오류(128799→128798 — 임베드 행
//!     교체). 임계 1e-3(실측 효과 원장 — 프로브 출력).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 개발기
//! (RTX 4070 SUPER) 타이밍 금지.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::ds4_attn_cuda::{Ds4AttnDims, Ds4LayerF32};
use crate::rawcuda::ds4_mtp_cuda::{Ds4MtpCuda, Ds4MtpDims};
use crate::rawcuda::exl3_cuda::{JParser, JVal, StArchive};
use crate::rawcuda::exl3_cuda_probe::{f16_to_f32, gen_unif, maxdiff_nan};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// 기본 픽스처 경로(과제 지정 — 프로브 인자로 대체 가능).
pub const DS4_MTP_EXL3_DIR: &str = "D:/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw";

/// MoE 이후 종단(hmix·h·로짓) 판정 임계 — ds4-moe 라우트 가중치 계급
/// (ROUTE_W 1e-6, ds4_moe_cuda_probe L64) 승계 + fp8 체인. 실측 대부분
/// bitdiff=0 기대(원장 갱신 커밋으로 문서화 — <=2e-3 허용 계약).
const DS4_MTP_LOGITS_THRESH: f32 = 2e-3;

/// 음성대조 탐지 임계(로짓/편차 스케일) — 실측 효과는 프로브 출력으로
/// 원장화(측정 전 1e-3 가정: |bias|·평면 순열·임베드 교체 편차 ≫ 임계).
const DS4_MTP_NEG_THRESH: f32 = 1e-3;

// ── 병렬 헬퍼(ds4_attn_cuda_probe 재이식 — 출력 분할 소유, 비트 불변) ──

fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .max(1)
}

/// [0,n) 항목(항목당 `elems` 원소)을 연속 스트립으로 분할 소유해 병렬
/// 실행. f(lo, hi, part) — part 는 v[lo·elems .. hi·elems) 슬라이스.
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
fn par_rows_mut<F: Fn(usize, &mut [f32]) + Sync>(v: &mut [f32], row_len: usize, f: &F) {
    par_ranges_mut(v, row_len, &|lo, _hi, part: &mut [f32]| {
        for (i, row) in part.chunks_mut(row_len).enumerate() {
            f(lo + i, row);
        }
    });
}

// ── core 미러: 비트 프리미티브(ops.rs 직이식) ──

fn o_bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let hi = ((b >> 16) as u16) as u32 & 1;
    f32::from_bits(((b + 0x7FFF + hi) >> 16) << 16)
}

fn o_rms_scale(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}

fn o_rms_norm_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let s = o_rms_scale(x, eps);
    x.iter()
        .zip(w.iter())
        .map(|(&v, &wv)| o_bf16_round(v * s * wv))
        .collect()
}

fn o_gemm_nt(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let t = x.len() / k;
    let mut y = vec![0.0f32; t * n];
    let body = |lo: usize, hi: usize, part: &mut [f32]| {
        for i in lo..hi {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x[i * k + kk] * w[kk * n + j];
                }
                part[(i - lo) * n + j] = acc;
            }
        }
    };
    par_ranges_mut(&mut y, n, &body);
    y
}

fn o_gemm_row(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += x[kk] * w[kk * n + j];
        }
        y[j] = acc;
    }
    y
}

fn o_pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    let l2 = e - 127 + i32::from(b & 0x7F_FFFF != 0);
    f32::from_bits(((l2 + 127) as u32) << 23)
}

fn o_f32_to_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a < 9.765625e-4 {
        return sign;
    }
    if a < 0.015625 {
        let q = a * 512.0;
        let r = q.round_ties_even();
        if r >= 8.0 {
            return sign | 0x08;
        }
        return sign | r as u8;
    }
    let b = a.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127;
    let mant = b & 0x7F_FFFF;
    let rem = mant & 0xF_FFFF;
    let mut mm = mant >> 20;
    let mut ee = e;
    if rem > 0x8_0000 || (rem == 0x8_0000 && (mm & 1) != 0) {
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

fn o_e4m3_to_f32(u: u8) -> f32 {
    let sv = if u & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e4 = ((u >> 3) & 0xF) as u32;
    let m = (u & 7) as u32;
    if e4 == 15 && m == 7 {
        return f32::NAN;
    }
    if e4 == 0 {
        return sv * m as f32 * 0.001953125;
    }
    f32::from_bits(((e4 - 7 + 127) << 23) | (m << 20)) * sv
}

fn o_fp8_sim(x: &mut [f32], cols: usize, block: usize) {
    let rows = x.len() / cols;
    for r in 0..rows {
        for c0 in (0..cols).step_by(block) {
            let hi = (c0 + block).min(cols);
            let blk = &mut x[r * cols + c0..r * cols + hi];
            let mut amax = 0.0f32;
            for v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = o_pow2_ceil(amax * (1.0 / 448.0));
            for v in blk.iter_mut() {
                let mut q = *v / s;
                q = q.clamp(-448.0, 448.0);
                *v = o_e4m3_to_f32(o_f32_to_e4m3(q)) * s;
            }
        }
    }
}

/// ops.rs exp_cr(L52-119) 직이식 — f64 fma 호너 13단 + 2^k 비트 재구성.
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

// ── core 미러: RoPE(ops.rs RopeTable::build/rope_apply 직이식) ──

struct ORope {
    cs: Vec<f32>,
    half: usize,
}

fn o_find_correction_dim(num_rot: f64, dim: usize, base: f64, max_len: usize) -> f64 {
    dim as f64 * (max_len as f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
}

#[allow(clippy::too_many_arguments)]
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

// ── core 미러: 어텐션 스테이지(attn.rs 직이식 — DSpark 경로 분) ──

/// attn.rs project_q L84-120 — (c_Q, q[로프 전]).
fn o_project_q(lay: &Ds4LayerF32, x: &[f32], d: &Ds4AttnDims) -> (Vec<f32>, Vec<f32>) {
    let (dm, qrank, nh, hd) = (d.dim, d.q_lora_rank, d.n_heads, d.head_dim);
    let _ = nh;
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
    par_rows_mut(&mut q, hd, &|i: usize, head: &mut [f32]| {
        let _ = i;
        let s = o_rms_scale(head, eps);
        for v in head.iter_mut() {
            *v = o_bf16_round(*v * s);
        }
    });
    (c, q)
}

/// attn.rs rope_q L122-135 — 토큰 i 위치 = 로프표 행 i(호출부 시프트표).
fn o_rope_q(q: &mut [f32], d: &Ds4AttnDims, rope: &ORope) {
    let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
    let csr = &rope.cs;
    let half = rope.half;
    par_rows_mut(q, hd, &|i: usize, head: &mut [f32]| {
        let pos = i / nh;
        o_rope_apply(
            &mut head[hd - rd..],
            &csr[pos * half * 2..(pos + 1) * half * 2],
            false,
        );
        for v in head[hd - rd..].iter_mut() {
            *v = o_bf16_round(*v);
        }
    });
}

/// attn.rs project_kv L137-158 — 1토큰/시프트표 재사용(로프 행 = 행 인덱스).
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

/// attn.rs attention_output L582-599(t=1 형 — o_attn_output_tok).
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

// ── core 미러: mHC 스테이지(hc.rs 직이식 — ds4_hc probe 재이식) ──

fn core_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + o_exp_cr(-x))
}

/// hc.rs hc_split_sinkhorn(L31-112) 직이식.
fn ref_split_sinkhorn(
    mixes: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(mixes.len(), (2 + hc) * hc);
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        pre[j] = core_sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    }
    for j in 0..hc {
        post[j] = 2.0 * core_sigmoid(mixes[hc + j] * scale[1] + base[hc + j]);
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
        }
    }
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = o_exp_cr(*v - m);
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    for v in comb.iter_mut() {
        *v += eps;
    }
    let mut col = vec![0.0f32; hc];
    for j in 0..hc {
        for k in 0..hc {
            col[k] += comb[j * hc + k];
        }
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] /= col[k] + eps;
        }
    }
    let mut row = vec![0.0f32; hc];
    for _ in 1..iters {
        row.fill(0.0);
        for j in 0..hc {
            for k in 0..hc {
                row[j] += comb[j * hc + k];
            }
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] /= row[j] + eps;
            }
        }
        col.fill(0.0);
        for j in 0..hc {
            for k in 0..hc {
                col[k] += comb[j * hc + k];
            }
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] /= col[k] + eps;
            }
        }
    }
    (pre, post, comb)
}

/// hc.rs hc_pre(L115-152) 직이식 — (y, post, comb).
fn hc_pre_ref(
    x: &[f32],
    d: usize,
    hc: usize,
    fns: &[f32],
    base: &[f32],
    scale: &[f32],
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mix_hc = (2 + hc) * hc;
    let rsqrt = o_rms_scale(x, norm_eps);
    let mut mixes = vec![0.0f32; mix_hc];
    for (i, m) in mixes.iter_mut().enumerate() {
        let fr = &fns[i * (hc * d)..(i + 1) * (hc * d)];
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(fr.iter()) {
            acc += a * b;
        }
        *m = acc * rsqrt;
    }
    let scale3 = [scale[0], scale[1], scale[2]];
    let (pre, post, comb) = ref_split_sinkhorn(&mixes, &scale3, base, hc, iters, hc_eps);
    let mut y = vec![0.0f32; d];
    for j in 0..hc {
        let xj = &x[j * d..(j + 1) * d];
        let pj = pre[j];
        for (yi, &xv) in y.iter_mut().zip(xj.iter()) {
            *yi += pj * xv;
        }
    }
    for yi in y.iter_mut() {
        *yi = o_bf16_round(*yi);
    }
    (y, post, comb)
}

/// hc.rs hc_post(L155-181) 직이식.
fn hc_post_ref(
    f: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    d: usize,
    hc: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; hc * d];
    for j in 0..hc {
        let pj = post[j];
        for i in 0..d {
            y[j * d + i] = pj * f[i];
        }
        for k in 0..hc {
            let c = comb[j * hc + k];
            let rk = &residual[k * d..(k + 1) * d];
            for i in 0..d {
                y[j * d + i] += c * rk[i];
            }
        }
        for i in 0..d {
            y[j * d + i] = o_bf16_round(y[j * d + i]);
        }
    }
    y
}

/// hc.rs hc_head(L185-214) 직이식 — fn[4·d]·base[4]·scale[1].
fn hc_head_ref(
    x: &[f32],
    d: usize,
    hc: usize,
    fns: &[f32],
    base: &[f32],
    scale: &[f32],
    norm_eps: f32,
    hc_eps: f32,
) -> Vec<f32> {
    let rsqrt = o_rms_scale(x, norm_eps);
    let mut pre = vec![0.0f32; hc];
    for j in 0..hc {
        let fr = &fns[j * (hc * d)..(j + 1) * (hc * d)];
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(fr.iter()) {
            acc += a * b;
        }
        pre[j] = core_sigmoid(acc * rsqrt * scale[0] + base[j]) + hc_eps;
    }
    let mut y = vec![0.0f32; d];
    for j in 0..hc {
        let xj = &x[j * d..(j + 1) * d];
        for (yi, &xv) in y.iter_mut().zip(xj.iter()) {
            *yi += pre[j] * xv;
        }
    }
    for yi in y.iter_mut() {
        *yi = o_bf16_round(*yi);
    }
    y
}

// ── core 미러: MoE 스테이지(moe.rs 직이식 — ds4_moe probe 재이식) ──

fn ln_cr_mirror(v: f64) -> f64 {
    let bits = v.to_bits();
    let k = (((bits >> 52) & 0x7ff) as i64) - 1023;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52));
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut q = 1.0f64 / 25.0;
    q = q.mul_add(t2, 1.0 / 23.0);
    q = q.mul_add(t2, 1.0 / 21.0);
    q = q.mul_add(t2, 1.0 / 19.0);
    q = q.mul_add(t2, 1.0 / 17.0);
    q = q.mul_add(t2, 1.0 / 15.0);
    q = q.mul_add(t2, 1.0 / 13.0);
    q = q.mul_add(t2, 1.0 / 11.0);
    q = q.mul_add(t2, 1.0 / 9.0);
    q = q.mul_add(t2, 1.0 / 7.0);
    q = q.mul_add(t2, 1.0 / 5.0);
    q = q.mul_add(t2, 1.0 / 3.0);
    q = q.mul_add(t2, 1.0);
    let lnm = 2.0 * t * q;
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let kh = (k as f64) * LN2_HI;
    let kl = (k as f64) * LN2_LO;
    let s1 = lnm + kh;
    let s2 = (lnm - s1) + kh;
    s1 + (s2 + kl)
}

fn softplus_mirror(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        ln_cr_mirror(o_exp_cr(x) as f64 + 1.0) as f32
    }
}

fn silu_mirror(x: f32) -> f32 {
    x / (1.0 + o_exp_cr(-x))
}

fn sqrtsoftplus_mirror(g: f32) -> f32 {
    softplus_mirror(g).sqrt()
}

fn swiglu_limit_mirror(gate: f32, up: f32, limit: f32) -> f32 {
    silu_mirror(gate.min(limit)) * up.clamp(-limit, limit)
}

fn gate_scores_mirror(x: &[f32], gw: &[f32], n_routed: usize, dim: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n_routed];
    for (j, sj) in s.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for i in 0..dim {
            acc += x[i] * gw[i * n_routed + j];
        }
        *sj = sqrtsoftplus_mirror(acc);
    }
    s
}

fn topk_stable_mirror(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.into_iter().map(|i| (i, scores[i])).collect()
}

fn route_mirror(scores: &[f32], sel: Vec<usize>, route_scale: f32) -> Vec<(usize, f32)> {
    let raw: Vec<f32> = sel.iter().map(|&i| scores[i]).collect();
    let sum: f32 = raw.iter().sum();
    let mut out: Vec<(usize, f32)> = sel
        .into_iter()
        .zip(raw)
        .map(|(i, w)| (i, w / sum * route_scale))
        .collect();
    out.sort_by_key(|&(i, _)| i);
    out
}

fn route_routed_mirror(
    scores: &[f32],
    bias: &[f32],
    k: usize,
    route_scale: f32,
) -> Vec<(usize, f32)> {
    let biased: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();
    let sel: Vec<usize> = topk_stable_mirror(&biased, k)
        .into_iter()
        .map(|(i, _)| i)
        .collect();
    route_mirror(scores, sel, route_scale)
}

#[derive(Clone)]
struct ExpertW {
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

struct GateMirror {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// moe.rs expert_ffn L98-131 미러.
fn expert_ffn_mirror(
    x: &[f32],
    e: &ExpertW,
    weight: f32,
    dim: usize,
    inter: usize,
    limit: f32,
) -> Vec<f32> {
    let mut xq = x.to_vec();
    o_fp8_sim(&mut xq, dim, 128);
    let mut g = o_gemm_row(&xq, &e.w1, dim, inter);
    for v in g.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let mut u = o_gemm_row(&xq, &e.w3, dim, inter);
    for v in u.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        h[i] = swiglu_limit_mirror(g[i], u[i], limit) * weight;
    }
    for v in h.iter_mut() {
        *v = o_bf16_round(*v);
    }
    o_fp8_sim(&mut h, inter, 128);
    let mut y = o_gemm_row(&h, &e.w2, inter, dim);
    for v in y.iter_mut() {
        *v = o_bf16_round(*v);
    }
    y
}

/// moe.rs moe_forward L133-216 미러(전문가 id 오름차순 누산·공유 무가중
/// 후행·최종 bf16 — by_expert BTreeMap 계약).
fn moe_forward_mirror(
    x: &[f32],
    gw: &GateMirror,
    shared: &ExpertW,
    experts: &HashMap<usize, ExpertW>,
    n_routed: usize,
    n_active: usize,
    route_scale: f32,
    swiglu_limit: f32,
) -> Vec<f32> {
    let dim = if x.is_empty() {
        0
    } else {
        gw.weight.len() / n_routed
    };
    let t = x.len() / dim;
    let inter = shared.w1.len() / dim;
    let routes: Vec<Vec<(usize, f32)>> = (0..t)
        .map(|ti| {
            let xt = &x[ti * dim..(ti + 1) * dim];
            let scores = gate_scores_mirror(xt, &gw.weight, n_routed, dim);
            route_routed_mirror(&scores, &gw.bias, n_active, route_scale)
        })
        .collect();
    let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (ti, r) in routes.iter().enumerate() {
        for &(eid, w) in r {
            by_expert.entry(eid).or_default().push((ti, w));
        }
    }
    let mut y = vec![0.0f32; t * dim];
    for (eid, toks) in by_expert {
        let e = experts
            .get(&eid)
            .unwrap_or_else(|| panic!("오라클: 전문가 e={eid} 미적재"));
        for (ti, w) in toks {
            let out =
                expert_ffn_mirror(&x[ti * dim..(ti + 1) * dim], e, w, dim, inter, swiglu_limit);
            for (yi, &ov) in y[ti * dim..(ti + 1) * dim].iter_mut().zip(out.iter()) {
                *yi += ov;
            }
        }
    }
    for ti in 0..t {
        let xt = &x[ti * dim..(ti + 1) * dim];
        let out = expert_ffn_mirror(xt, shared, 1.0, dim, inter, swiglu_limit);
        for (yi, &ov) in y[ti * dim..(ti + 1) * dim].iter_mut().zip(out.iter()) {
            *yi += ov;
        }
    }
    for v in y.iter_mut() {
        *v = o_bf16_round(*v);
    }
    y
}

// ── trellis 디양자화 미러(trellis.rs 직이식 — 오프셋 직독) ──

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

fn o_mul1_decode(word: u16) -> f32 {
    const MUL1_MULT: u32 = 0x83DCD12D;
    const K_INV: f32 = 0.00676727294921875;
    const K_BIAS: f32 = -10.3828125;
    let x = (word as u32).wrapping_mul(MUL1_MULT);
    let sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    let f = 1024.0f32 + sum as f32;
    f.mul_add(K_INV, K_BIAS)
}

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

/// trellis 원시 3중 보유(헤드 스트립 디양자화용 — 전행렬 미조립).
fn linear_mirror(ar: &StArchive, key: &str) -> Result<(LinearMirror, usize), String> {
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
    let tre_u32: Vec<u32> = tre
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((
        LinearMirror {
            tre_u32,
            suh: ar.read(&format!("{key}.suh"))?,
            svh: ar.read(&format!("{key}.svh"))?,
            n,
            krate,
            ntiles: nt,
        },
        k,
    ))
}

/// loader.rs plain_f32 L156-172 — plain 텐서 전체 → f32(dtype 분기).
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

/// loader.rs embed_rows L437-447 — 임베딩 행 슬라이스 오프셋 직독.
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

/// mtp.0.attn 적재(loader.rs block L344-381 attn 부분 미러 — SWA: comp/idx 없음).
fn load_mtp_attn_layer(ar: &StArchive, d: &Ds4AttnDims) -> Result<Ds4LayerF32, String> {
    let p = "mtp.0.attn";
    let wo_a = (0..d.o_groups)
        .map(|g| dequant_linear(ar, &format!("{p}.wo_a.slice.{g}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Ds4LayerF32 {
        wq_a: dequant_linear(ar, &format!("{p}.wq_a"))?,
        q_norm: plain_f32(ar, &format!("{p}.q_norm.weight"))?,
        wq_b: dequant_linear(ar, &format!("{p}.wq_b"))?,
        wkv: dequant_linear(ar, &format!("{p}.wkv"))?,
        kv_norm: plain_f32(ar, &format!("{p}.kv_norm.weight"))?,
        sink: plain_f32(ar, &format!("{p}.attn_sink"))?,
        wo_a,
        wo_b: dequant_linear(ar, &format!("{p}.wo_b"))?,
        comp: None,
        idx: None,
    })
}

// ── core 미러: DSpark 코어 fn(frame.rs 직이식) ──

/// frame.rs dspark_main_x L208-230 — main_x = main_norm(main_proj(mh)).
fn o_dspark_main_x(mh: &[f32], proj: &[f32], norm_w: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let t = mh.len() / (3 * dim);
    let mut xq = mh.to_vec();
    o_fp8_sim(&mut xq, 3 * dim, 128);
    let mut y = o_gemm_nt(&xq, proj, 3 * dim, dim);
    for v in y.iter_mut() {
        *v = o_bf16_round(*v);
    }
    let mut out = Vec::with_capacity(t * dim);
    for i in 0..t {
        out.extend_from_slice(&o_rms_norm_weighted(
            &y[i * dim..(i + 1) * dim],
            norm_w,
            eps,
        ));
    }
    out
}

/// frame.rs dspark_window_warm L242-263 — 링 [win][hd].
fn o_dspark_window_warm(
    lay: &Ds4LayerF32,
    main_x: &[f32],
    d: &Ds4AttnDims,
    rope: &ORope,
) -> Vec<f32> {
    let (hd, win) = (d.head_dim, d.window);
    let t = main_x.len() / d.dim;
    let kv = o_project_kv(lay, main_x, d, rope);
    let mut ring = vec![0.0f32; win * hd];
    if t <= win {
        ring[..t * hd].copy_from_slice(&kv);
    } else {
        let cutoff = t % win;
        let tail = &kv[(t - win) * hd..];
        ring[cutoff * hd..].copy_from_slice(&tail[..(win - cutoff) * hd]);
        ring[..cutoff * hd].copy_from_slice(&tail[(win - cutoff) * hd..]);
    }
    ring
}

/// frame.rs dspark_decode_attn L267-345 직이식 — 시프트 ORope(행 i =
/// 절대 위치 pos+1+i)로 project_q/kv 재사용(project_kv_unroped L349-371
/// 산술과 동일 비트 — 로프 딤/비로프 fp8 분리 불변).
fn o_dspark_decode_attn(
    lay: &Ds4LayerF32,
    x_draft: &[f32],
    main_x_tok: &[f32],
    ring: &mut [f32],
    pos: usize,
    d: &Ds4AttnDims,
    rope: &ORope,
) -> Vec<f32> {
    let (dm, hd, nh, win, rd) = (d.dim, d.head_dim, d.n_heads, d.window, d.rope_head_dim);
    let b = x_draft.len() / dm;
    let half = rope.half;
    // 메인 토큰 kv — 1토큰 project_kv(로프 행 0 계약) 후 링 기입.
    let mkv = o_project_kv(lay, main_x_tok, d, rope);
    ring[pos % win * hd..(pos % win + 1) * hd].copy_from_slice(&mkv);
    // 시프트 표(행 i = rope 행 pos+1+i).
    let mut shift = ORope {
        cs: vec![0.0f32; b * half * 2],
        half,
    };
    for i in 0..b {
        let src = (pos + 1 + i) * half * 2;
        shift.cs[i * half * 2..(i + 1) * half * 2].copy_from_slice(&rope.cs[src..src + half * 2]);
    }
    // 드래프트 q — project_q + 시프트 rope_q(수동 로프 pos+1+i 와 동일).
    let (_, mut q) = o_project_q(lay, x_draft, d);
    o_rope_q(&mut q, d, &shift);
    // 드래프트 kv — project_kv(시프트) = project_kv_unroped+수동 로프.
    let kvd = o_project_kv(lay, x_draft, d, &shift);
    let mut kv_all = ring.to_vec();
    kv_all.extend_from_slice(&kvd);
    // idxs = [0..min(win,pos+1)] ++ [win..win+b](L313-315).
    let mut idxs: Vec<i32> = (0..win.min(pos + 1)).map(|v| v as i32).collect();
    idxs.extend((0..b).map(|v| (win + v) as i32));
    let mut out = vec![0.0f32; b * dm];
    par_rows_mut(&mut out, dm, &|ti: usize, part: &mut [f32]| {
        let mut o = o_sparse_attn_one(
            &q[ti * nh * hd..(ti + 1) * nh * hd],
            &kv_all,
            &idxs,
            &lay.sink,
            d,
        );
        for h in 0..nh {
            let base = h * hd;
            o_rope_apply(
                &mut o[base + hd - rd..base + hd],
                &shift.cs[ti * half * 2..(ti + 1) * half * 2],
                true,
            );
            for v in o[base + hd - rd..base + hd].iter_mut() {
                *v = o_bf16_round(*v);
            }
        }
        let y = o_attn_output_tok(&o, lay, d);
        part.copy_from_slice(&y);
    });
    out
}

/// frame.rs markov_logits_bias L373-383 — w2 gemv(k 오름차순 f32).
fn o_markov_logits_bias(emb: &[f32], w2: &[f32], vocab: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; vocab];
    for (kk, &e) in emb.iter().enumerate() {
        let row = &w2[kk * vocab..(kk + 1) * vocab];
        for (yi, &wv) in y.iter_mut().zip(row.iter()) {
            *yi += e * wv;
        }
    }
    y
}

/// frame.rs confidence_score L386-397 — chain 순서 f32 순차 dot.
fn o_confidence(x: &[f32], m: &[f32], proj: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&v, &p) in x.iter().chain(m.iter()).zip(proj.iter()) {
        acc += v * p;
    }
    acc
}

/// frame.rs head_gemv_stripwise L41-77 — 스트립 gemv(k블록 부분합 순서,
/// 반올림 없음). xs: [b][k] 토큰 묶음(블록 디양자화 1회 공유).
fn o_head_strip(lin: &LinearMirror, xs: &[Vec<f32>], n0: usize, k: usize) -> Vec<Vec<f32>> {
    let b = xs.len();
    let mut ys = vec![vec![0.0f32; 128]; b];
    for k0 in (0..k).step_by(128) {
        let blk = o_dequant_block(lin, k0, n0);
        for j in 0..128 {
            for (ti, xs_row) in xs.iter().enumerate() {
                let mut acc = 0.0f32;
                for i in 0..128 {
                    acc += xs_row[k0 + i] * blk[i * 128 + j] as f32;
                }
                ys[ti][j] += acc;
            }
        }
    }
    ys
}

/// 비트 불일치 수(to_bits 동일 판정).
fn bitdiff(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count()
}

/// 판정 보고 1줄 + 실패 적립.
fn judge(
    dev: &str,
    tag: &str,
    got: &[f32],
    want: &[f32],
    bit_exact: bool,
    thresh: f32,
    fails: &mut Vec<String>,
    report: &mut String,
) -> bool {
    let (md, nan) = maxdiff_nan(got, want);
    let bd = bitdiff(got, want);
    let pass = if bit_exact {
        bd == 0 && nan == 0
    } else {
        md <= thresh && nan == 0
    };
    println!(
        "device: {dev} | ds4-mtp {tag}: maxdiff={md:.3e} bitdiff={bd}/{} nan={nan} | {}",
        want.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · {tag} {md:.3e}/{bd}b"));
    if !pass {
        fails.push(format!(
            "{tag} maxdiff={md:.3e} bitdiff={bd} nan={nan} (기준 {})",
            if bit_exact {
                "비트동일".to_string()
            } else {
                format!("≤{thresh:.0e}")
            }
        ));
    }
    pass
}

// ── 오라클 드래프트 체인(사전/사후 분할 — 전문가 적재 창구) ──

/// 어텐션 서브블록까지의 체인 상태(전문가 적재 전 정지점).
struct ChainPre {
    x2: Vec<Vec<f32>>,
    posts2: Vec<Vec<f32>>,
    combs2: Vec<Vec<f32>>,
    xn2: Vec<f32>,
}

/// 체인 사전부: mh → main_x → 링 워밍 → 드래프트 블록 attn 서브블록.
#[allow(clippy::too_many_arguments)]
fn chain_pre(
    mh: &[f32],
    lay: &Ds4LayerF32,
    proj: &[f32],
    main_norm: &[f32],
    attn_norm: &[f32],
    ffn_norm: &[f32],
    hc_attn: (&[f32], &[f32], &[f32]),
    hc_ffn: (&[f32], &[f32], &[f32]),
    ad: &Ds4AttnDims,
    rope: &ORope,
    x_in: &[Vec<f32>],
    pos: usize,
    dim: usize,
    hc: usize,
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
) -> Result<(Vec<f32>, Vec<f32>, ChainPre), String> {
    // main_x + 링 + 메인 토큰.
    let main_x = o_dspark_main_x(mh, proj, main_norm, dim, norm_eps);
    let ring = o_dspark_window_warm(lay, &main_x, ad, rope);
    let main_x_tok = main_x[(main_x.len() / dim - 1) * dim..].to_vec();
    // attn 서브블록.
    let t = x_in.len();
    let mut y = Vec::with_capacity(t);
    let mut posts = Vec::with_capacity(t);
    let mut combs = Vec::with_capacity(t);
    for r in x_in {
        let (yi, post, comb) = hc_pre_ref(
            r, dim, hc, hc_attn.0, hc_attn.1, hc_attn.2, norm_eps, hc_eps, iters,
        );
        y.push(yi);
        posts.push(post);
        combs.push(comb);
    }
    let mut xn = Vec::with_capacity(t * dim);
    for yi in &y {
        xn.extend_from_slice(&o_rms_norm_weighted(yi, attn_norm, norm_eps));
    }
    let mut ring_m = ring.clone();
    let a = o_dspark_decode_attn(lay, &xn, &main_x_tok, &mut ring_m, pos, ad, rope);
    let a_rows: Vec<Vec<f32>> = a.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let mut x2 = Vec::with_capacity(t);
    for (ti, a_row) in a_rows.iter().enumerate() {
        x2.push(hc_post_ref(
            a_row, &x_in[ti], &posts[ti], &combs[ti], dim, hc,
        ));
    }
    // ffn 서브블록 pre(전문가 적재 전 정지).
    let mut y2 = Vec::with_capacity(t);
    let mut posts2 = Vec::with_capacity(t);
    let mut combs2 = Vec::with_capacity(t);
    for r in &x2 {
        let (yi, post, comb) = hc_pre_ref(
            r, dim, hc, hc_ffn.0, hc_ffn.1, hc_ffn.2, norm_eps, hc_eps, iters,
        );
        y2.push(yi);
        posts2.push(post);
        combs2.push(comb);
    }
    let mut xn2 = Vec::with_capacity(t * dim);
    for yi in &y2 {
        xn2.extend_from_slice(&o_rms_norm_weighted(yi, ffn_norm, norm_eps));
    }
    Ok((
        main_x,
        ring,
        ChainPre {
            x2,
            posts2,
            combs2,
            xn2,
        },
    ))
}

/// 체인 사후부: MoE → hc_post → hc_head → 종결 norm → h [b][dim].
#[allow(clippy::too_many_arguments)]
fn chain_post(
    pre: &ChainPre,
    gw: &GateMirror,
    shared: &ExpertW,
    experts: &HashMap<usize, ExpertW>,
    dims: &Ds4MtpDims,
    hc_head: (&[f32], &[f32], &[f32]),
    final_norm: &[f32],
    norm_eps: f32,
    hc_eps: f32,
) -> Result<Vec<Vec<f32>>, String> {
    let (dim, hc) = (dims.dim, dims.hc);
    let t = pre.x2.len();
    let _xn2_rows: Vec<Vec<f32>> = pre.xn2.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let f = moe_forward_mirror(
        &pre.xn2,
        gw,
        shared,
        experts,
        dims.moe.n_routed,
        dims.moe.n_active,
        dims.moe.route_scale,
        dims.moe.swiglu_limit,
    );
    let f_rows: Vec<Vec<f32>> = f.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let mut x_out = Vec::with_capacity(t);
    for (ti, f_row) in f_rows.iter().enumerate() {
        x_out.push(hc_post_ref(
            f_row,
            &pre.x2[ti],
            &pre.posts2[ti],
            &pre.combs2[ti],
            dim,
            hc,
        ));
    }
    let mut hmix = Vec::with_capacity(t);
    for r in &x_out {
        hmix.push(hc_head_ref(
            r, dim, hc, hc_head.0, hc_head.1, hc_head.2, norm_eps, hc_eps,
        ));
    }
    let mut h = Vec::with_capacity(t);
    for r in &hmix {
        h.push(o_rms_norm_weighted(r, final_norm, norm_eps));
    }
    Ok(h)
}

/// 체인 라우팅 사전 판정(전문가 적재 목록 — xn2 기준).
fn chain_needed_experts(pre: &ChainPre, gw: &GateMirror, dims: &Ds4MtpDims) -> Vec<u32> {
    let dim = dims.dim;
    let mut need = std::collections::BTreeSet::new();
    for ti in 0..pre.xn2.len() / dim {
        let xt = &pre.xn2[ti * dim..(ti + 1) * dim];
        let scores = gate_scores_mirror(xt, &gw.weight, dims.moe.n_routed, dim);
        for (e, _) in
            route_routed_mirror(&scores, &gw.bias, dims.moe.n_active, dims.moe.route_scale)
        {
            need.insert(e as u32);
        }
    }
    need.into_iter().collect()
}

// ── 메인 프로브 ──

/// ds4-mtp — DSpark 드래프트 스테이지 전체 체인 값 판정. 사전 비트동일
/// 지점(main_x·링·디코드 어텐션·attn 서브블록)은 bitdiff=0, MoE 이후
/// (hmix·h·로짓)는 ≤ 2e-3(라우트 가중치 계급 승계 — 헤드 원장). 음성대조
/// 3종(트렁크 타깃 순열·마르코프 누락·noise id 오류)은 로짓 레벨 탐지.
pub fn cuda_ds4_mtp_check(dir: &str) -> Result<String, String> {
    let dir = Path::new(dir);
    let cfg_path = dir.join("config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = Ds4MtpDims::from_config(&cfg_text)?;
    let ad = Ds4AttnDims::from_config(&cfg_text)?;
    let (dim, hc, vocab, rank) = (dims.dim, dims.hc, dims.vocab, dims.markov_rank);
    let ar = StArchive::open(dir)?;
    let mut modl = Ds4MtpCuda::open(&cfg_text)?;
    let dev = modl.device_name().to_string();

    // ── 가중치 적재(오프셋 직독) + 등록 ──
    let lay = load_mtp_attn_layer(&ar, &ad)?;
    modl.register_draft_layer(&lay)?;
    modl.register_sink(&lay.sink)?;
    // 윈도우 계열 로프(base 10000, YaRN 없음 — ratio 0 계약).
    let rope = o_rope_build(ad.rope_head_dim, 16, 10000.0, false, 1.0, 0, 32.0, 1.0);
    modl.set_rope_main(&rope.cs, rope.half)?;
    let hc_attn = (
        plain_f32(&ar, "mtp.0.hc_attn_fn")?,
        plain_f32(&ar, "mtp.0.hc_attn_base")?,
        plain_f32(&ar, "mtp.0.hc_attn_scale")?,
    );
    let hc_ffn = (
        plain_f32(&ar, "mtp.0.hc_ffn_fn")?,
        plain_f32(&ar, "mtp.0.hc_ffn_base")?,
        plain_f32(&ar, "mtp.0.hc_ffn_scale")?,
    );
    modl.register_hc(
        &hc_attn.0, &hc_attn.1, &hc_attn.2, &hc_ffn.0, &hc_ffn.1, &hc_ffn.2,
    )?;
    let hc_head = (
        plain_f32(&ar, "mtp.2.hc_head_fn")?,
        plain_f32(&ar, "mtp.2.hc_head_base")?,
        plain_f32(&ar, "mtp.2.hc_head_scale")?,
    );
    modl.register_hc_head(&hc_head.0, &hc_head.1, &hc_head.2)?;
    let (main_norm, attn_norm, ffn_norm, final_norm) = (
        plain_f32(&ar, "mtp.0.main_norm.weight")?,
        plain_f32(&ar, "mtp.0.attn_norm.weight")?,
        plain_f32(&ar, "mtp.0.ffn_norm.weight")?,
        plain_f32(&ar, "mtp.2.norm.weight")?,
    );
    modl.register_norms(&main_norm, &attn_norm, &ffn_norm, &final_norm)?;
    let proj = dequant_linear(&ar, "mtp.0.main_proj")?;
    modl.register_main_proj(&proj)?;
    let (w1, w2, conf_proj) = (
        plain_f32(&ar, "mtp.2.markov_head.markov_w1.weight")?,
        plain_kmat(&ar, "mtp.2.markov_head.markov_w2.weight")?,
        plain_f32(&ar, "mtp.2.confidence_head.proj.weight")?,
    );
    modl.register_markov(&w1, &w2, &conf_proj)?;
    let gate = GateMirror {
        weight: plain_kmat(&ar, "mtp.0.ffn.gate.weight")?,
        bias: plain_f32(&ar, "mtp.0.ffn.gate.bias")?,
    };
    let shared = load_expert_w(&ar, "mtp.0.ffn.shared_experts")?;
    modl.moe
        .set_gate_f32(&gate.weight, Some(&gate.bias), None)?;
    modl.moe
        .set_shared_f32(&shared.w1, &shared.w2, &shared.w3)?;
    let (head_lin, head_k) = linear_mirror(&ar, "head")?;
    if head_k != dim || head_lin.n != vocab {
        return Err(format!(
            "ds4-mtp: head {head_k}×{} != {dim}×{vocab}(픽스처 가드)",
            head_lin.n
        ));
    }

    // ── 합성 자료 + 입력 ──
    let (t_main, pos, in_tok) = (8usize, 7usize, 1000u32);
    let b = dims.dspark_block;
    // main_hidden — 트렁크 타깃 3평면(40/41/42) 스트림 평균 계급 합성.
    let planes: Vec<Vec<f32>> = (0..3)
        .map(|li| gen_unif(t_main * dim, 0x5EED_BA0B_0000_0000 + li as u64, 0.05))
        .collect();
    let mut mh = vec![0.0f32; t_main * 3 * dim];
    for i in 0..t_main * dim {
        for li in 0..3 {
            mh[i * 3 + li] = planes[li][i];
        }
    }
    let draft_ids = modl.dspark_draft_ids(in_tok);
    let emb_all = embed_rows(dir, &draft_ids)?;
    let x_in: Vec<Vec<f32>> = (0..b)
        .map(|ti| {
            let mut row = Vec::with_capacity(hc * dim);
            for _ in 0..hc {
                row.extend_from_slice(&emb_all[ti * dim..(ti + 1) * dim]);
            }
            row
        })
        .collect();

    // ── 오라클 체인(정밀 + 음성 변형 a/c) ──
    let hc_eps = 1e-6f32;
    let iters = 20usize;
    let (o_main_x, o_ring, pre_o) = chain_pre(
        &mh,
        &lay,
        &proj,
        &main_norm,
        &attn_norm,
        &ffn_norm,
        (&hc_attn.0, &hc_attn.1, &hc_attn.2),
        (&hc_ffn.0, &hc_ffn.1, &hc_ffn.2),
        &ad,
        &rope,
        &x_in,
        pos,
        dim,
        hc,
        dims.rms_eps,
        hc_eps,
        iters,
    )?;
    // (a) 트렁크 타깃 순열 — 평면 [40,41,42] → [41,40,42].
    let mut mh_a = mh.clone();
    for i in 0..t_main * dim {
        mh_a[i * 3] = planes[1][i];
        mh_a[i * 3 + 1] = planes[0][i];
    }
    let (_, _, pre_a) = chain_pre(
        &mh_a,
        &lay,
        &proj,
        &main_norm,
        &attn_norm,
        &ffn_norm,
        (&hc_attn.0, &hc_attn.1, &hc_attn.2),
        (&hc_ffn.0, &hc_ffn.1, &hc_ffn.2),
        &ad,
        &rope,
        &x_in,
        pos,
        dim,
        hc,
        dims.rms_eps,
        hc_eps,
        iters,
    )?;
    // (c) noise id 오류 — 128799 → 128798(임베드 행 교체).
    let mut ids_c = draft_ids.clone();
    for id in ids_c.iter_mut().skip(1) {
        *id = dims.dspark_noise_token - 1;
    }
    let emb_c = embed_rows(dir, &ids_c)?;
    let x_c: Vec<Vec<f32>> = (0..b)
        .map(|ti| {
            let mut row = Vec::with_capacity(hc * dim);
            for _ in 0..hc {
                row.extend_from_slice(&emb_c[ti * dim..(ti + 1) * dim]);
            }
            row
        })
        .collect();
    let (_, _, pre_c) = chain_pre(
        &mh,
        &lay,
        &proj,
        &main_norm,
        &attn_norm,
        &ffn_norm,
        (&hc_attn.0, &hc_attn.1, &hc_attn.2),
        (&hc_ffn.0, &hc_ffn.1, &hc_ffn.2),
        &ad,
        &rope,
        &x_c,
        pos,
        dim,
        hc,
        dims.rms_eps,
        hc_eps,
        iters,
    )?;

    // ── 전문가 적재(정밀 체인 라우팅 — 모듈 상주 + 오라클 호스트) ──
    let need_o = chain_needed_experts(&pre_o, &gate, &dims);
    let mut experts: HashMap<usize, ExpertW> = HashMap::new();
    for e in &need_o {
        let w = load_expert_w(&ar, &format!("mtp.0.ffn.experts.{e}"))?;
        modl.moe.add_expert_f32(*e as usize, &w.w1, &w.w2, &w.w3)?;
        experts.insert(*e as usize, w);
    }

    // ── 모듈 체인(스텝별 + 종단 draft_block) ──
    let m_main_x = modl.dspark_main_x(&mh)?;
    let mut m_ring = modl.window_warm(&m_main_x)?;
    // (ii) 판정용 프리-디코드 스냅샷 — decode attn이 ring[pos%win]을 덮어쓰므로
    // window_warm 출력은 decode 전에 보존해 비교해야 한다(작성 버그 정정).
    let m_ring_predecode = m_ring.clone();
    // 중간 비교용 스텝(hc_pre → attn_norm → decode attn → hc_post).
    let (m_y, m_posts, m_combs) = modl.hc.hc_pre(modl.il_main, "attn", &x_in)?;
    let mut yflat = Vec::with_capacity(b * dim);
    for r in &m_y {
        yflat.extend_from_slice(r);
    }
    let m_xn = modl.norm_rows_pub(&yflat, b, "attn")?;
    let main_x_tok = &m_main_x[(t_main - 1) * dim..];
    let m_a = modl.dspark_decode_attn(&m_xn, main_x_tok, &mut m_ring, pos)?;
    let a_rows: Vec<Vec<f32>> = m_a.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let m_x2 = modl.hc.hc_post(&a_rows, &x_in, &m_posts, &m_combs)?;
    // 종단: 풀 블록(모듈 공개 API) → hc_head → norm → h.
    let mut m_ring2 = modl.window_warm(&m_main_x)?;
    let m_out = modl.draft_block(&x_in, main_x_tok, &mut m_ring2, pos)?;
    let m_hmix = modl.draft_head_hmix(&m_out)?;
    let m_h = modl.draft_final_norm(&m_hmix)?;

    // ── 오라클 사후부(정밀 + 변형 a/c — 전문가는 변형별 적재창) ──
    let h_o = chain_post(
        &pre_o,
        &gate,
        &shared,
        &experts,
        &dims,
        (&hc_head.0, &hc_head.1, &hc_head.2),
        &final_norm,
        dims.rms_eps,
        hc_eps,
    )?;
    let need_a = chain_needed_experts(&pre_a, &gate, &dims);
    let mut exp_a: HashMap<usize, ExpertW> = HashMap::new();
    for e in &need_a {
        if let Some(w) = experts.get(&(*e as usize)) {
            exp_a.insert(*e as usize, w.clone());
        } else {
            exp_a.insert(
                *e as usize,
                load_expert_w(&ar, &format!("mtp.0.ffn.experts.{e}"))?,
            );
        }
    }
    let h_a = chain_post(
        &pre_a,
        &gate,
        &shared,
        &exp_a,
        &dims,
        (&hc_head.0, &hc_head.1, &hc_head.2),
        &final_norm,
        dims.rms_eps,
        hc_eps,
    )?;
    drop(exp_a);
    let need_c = chain_needed_experts(&pre_c, &gate, &dims);
    let mut exp_c: HashMap<usize, ExpertW> = HashMap::new();
    for e in &need_c {
        if let Some(w) = experts.get(&(*e as usize)) {
            exp_c.insert(*e as usize, w.clone());
        } else {
            exp_c.insert(
                *e as usize,
                load_expert_w(&ar, &format!("mtp.0.ffn.experts.{e}"))?,
            );
        }
    }
    let h_c = chain_post(
        &pre_c,
        &gate,
        &shared,
        &exp_c,
        &dims,
        (&hc_head.0, &hc_head.1, &hc_head.2),
        &final_norm,
        dims.rms_eps,
        hc_eps,
    )?;
    drop(exp_c);

    // ── 판정(사전 비트동일 지점) ──
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();
    judge(
        &dev,
        "(i) main_x",
        &m_main_x,
        &o_main_x,
        true,
        0.0,
        &mut fails,
        &mut report,
    );
    judge(
        &dev,
        "(ii) ring",
        &m_ring_predecode,
        &o_ring,
        true,
        0.0,
        &mut fails,
        &mut report,
    );
    // 어텐션 중간: xn·a — 오라클 대응값(hc_pre posts/combs 1회 산출).
    let (o_y, o_posts, o_combs): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) = {
        let mut y = Vec::with_capacity(b);
        let mut p = Vec::with_capacity(b);
        let mut c = Vec::with_capacity(b);
        for r in &x_in {
            let (yi, post, comb) = hc_pre_ref(
                r,
                dim,
                hc,
                &hc_attn.0,
                &hc_attn.1,
                &hc_attn.2,
                dims.rms_eps,
                hc_eps,
                iters,
            );
            y.push(yi);
            p.push(post);
            c.push(comb);
        }
        (y, p, c)
    };
    let mut o_xn = Vec::with_capacity(b * dim);
    for yi in &o_y {
        o_xn.extend_from_slice(&o_rms_norm_weighted(yi, &attn_norm, dims.rms_eps));
    }
    judge(
        &dev,
        "(iii) attn_norm xn",
        &m_xn,
        &o_xn,
        true,
        0.0,
        &mut fails,
        &mut report,
    );
    let mut ring_o = o_ring.clone();
    let o_a = o_dspark_decode_attn(
        &lay,
        &o_xn,
        &o_main_x[(t_main - 1) * dim..],
        &mut ring_o,
        pos,
        &ad,
        &rope,
    );
    judge(
        &dev,
        "(iv) decode attn",
        &m_a,
        &o_a,
        true,
        0.0,
        &mut fails,
        &mut report,
    );
    let o_x2: Vec<f32> = {
        let a_rows: Vec<Vec<f32>> = o_a.chunks_exact(dim).map(|c| c.to_vec()).collect();
        let mut out = Vec::with_capacity(b * hc * dim);
        for ti in 0..b {
            out.extend_from_slice(&hc_post_ref(
                &a_rows[ti],
                &x_in[ti],
                &o_posts[ti],
                &o_combs[ti],
                dim,
                hc,
            ));
        }
        out
    };
    let m_x2_flat: Vec<f32> = m_x2.iter().flatten().copied().collect();
    judge(
        &dev,
        "(v) attn X'",
        &m_x2_flat,
        &o_x2,
        true,
        0.0,
        &mut fails,
        &mut report,
    );

    // ── 판정(MoE 이후 — 문서화 임계) ──
    let m_h_flat: Vec<f32> = m_h.iter().flatten().copied().collect();
    let h_o_flat: Vec<f32> = h_o.iter().flatten().copied().collect();
    judge(
        &dev,
        "(vi) final h",
        &m_h_flat,
        &h_o_flat,
        false,
        DS4_MTP_LOGITS_THRESH,
        &mut fails,
        &mut report,
    );

    // ── 헤드 스트립 루프(정밀 + 변형 a/c 오라클 공회) ──
    let mut logits_m = vec![0.0f32; b * vocab];
    let mut logits_o = vec![0.0f32; b * vocab];
    let mut logits_a = vec![0.0f32; b * vocab];
    let mut logits_c = vec![0.0f32; b * vocab];
    let h_a_flat: Vec<f32> = h_a.iter().flatten().copied().collect();
    let h_c_flat: Vec<f32> = h_c.iter().flatten().copied().collect();
    for n0 in (0..vocab).step_by(128) {
        // 스트립 디양자화 [k][128](k-major 재팩) — 블록당 1회.
        let mut w_strip = vec![0.0f32; dim * 128];
        {
            let body = |lo: usize, hi: usize, part: &mut [f32]| {
                for sc in lo..hi {
                    let k0 = sc * 128;
                    let blk = o_dequant_block(&head_lin, k0, n0);
                    for i in 0..128 {
                        let dst =
                            &mut part[(k0 + i - lo * 128) * 128..(k0 + i - lo * 128 + 1) * 128];
                        for (j, &vv) in blk[i * 128..(i + 1) * 128].iter().enumerate() {
                            dst[j] = vv as f32;
                        }
                    }
                }
            };
            par_ranges_mut(&mut w_strip, 128 * 128, &body);
        }
        let ys_o = o_head_strip(&head_lin, &h_o, n0, dim);
        let ys_a = o_head_strip(&head_lin, &h_a, n0, dim);
        let ys_c = o_head_strip(&head_lin, &h_c, n0, dim);
        let ys_m = modl.head_strip_gemv(&m_h_flat, &w_strip)?;
        for ti in 0..b {
            logits_m[ti * vocab + n0..ti * vocab + n0 + 128]
                .copy_from_slice(&ys_m[ti * 128..(ti + 1) * 128]);
            logits_o[ti * vocab + n0..ti * vocab + n0 + 128].copy_from_slice(&ys_o[ti]);
            logits_a[ti * vocab + n0..ti * vocab + n0 + 128].copy_from_slice(&ys_a[ti]);
            logits_c[ti * vocab + n0..ti * vocab + n0 + 128].copy_from_slice(&ys_c[ti]);
        }
    }

    // ── 마르코프 바이어스 + 신뢰도 ──
    let bias_m = modl.markov_logits_bias(&draft_ids)?;
    let mut bias_o = vec![0.0f32; b * vocab];
    let mut marks_o = Vec::with_capacity(b * rank);
    for (ti, &id) in draft_ids.iter().enumerate() {
        let row = &w1[id as usize * rank..(id as usize + 1) * rank];
        marks_o.extend_from_slice(row);
        let bias = o_markov_logits_bias(row, &w2, vocab);
        bias_o[ti * vocab..(ti + 1) * vocab].copy_from_slice(&bias);
    }
    judge(
        &dev,
        "(vii) markov bias",
        &bias_m,
        &bias_o,
        true,
        0.0,
        &mut fails,
        &mut report,
    );
    // 로짓 가산(코어: logits[:, i] += bias).
    for (lv, bv) in logits_m.iter_mut().zip(bias_m.iter()) {
        *lv += bv;
    }
    for (lv, bv) in logits_o.iter_mut().zip(bias_o.iter()) {
        *lv += bv;
    }
    judge(
        &dev,
        "(viii) logits",
        &logits_m,
        &logits_o,
        false,
        DS4_MTP_LOGITS_THRESH,
        &mut fails,
        &mut report,
    );
    // 신뢰도 — 동일 h(오라클 h) 입력으로 커널 비트 판정 + 모듈 체인 h 보고.
    let conf_m = modl.confidence(&h_o_flat, &marks_o)?;
    let conf_o: Vec<f32> = (0..b)
        .map(|ti| o_confidence(&h_o[ti], &marks_o[ti * rank..(ti + 1) * rank], &conf_proj))
        .collect();
    judge(
        &dev,
        "(ix) confidence",
        &conf_m,
        &conf_o,
        true,
        0.0,
        &mut fails,
        &mut report,
    );
    let conf_chain = modl.confidence(&m_h_flat, &marks_o)?;

    // ── 음성대조(로짓 레벨 — 원장 17호) ──
    let (md_a, _) = maxdiff_nan(&logits_m, &logits_a);
    let (md_b, _) = maxdiff_nan(
        &logits_m,
        &logits_o
            .iter()
            .zip(bias_o.iter())
            .map(|(l, b_)| l - b_)
            .collect::<Vec<_>>(),
    );
    let (md_c, _) = maxdiff_nan(&logits_m, &logits_c);
    let (bias_mag, _) = maxdiff_nan(&bias_o, &vec![0.0f32; bias_o.len()]);
    let det_a = md_a > DS4_MTP_NEG_THRESH;
    let det_b = md_b > DS4_MTP_NEG_THRESH;
    let det_c = md_c > DS4_MTP_NEG_THRESH;
    println!(
        "device: {dev} | ds4-mtp neg: (a) trunk-perm logits maxdiff={md_a:.3e} (b) bias-omitted maxdiff={md_b:.3e} (|bias|max={bias_mag:.3e}) (c) noise-id maxdiff={md_c:.3e} | {}",
        if det_a && det_b && det_c {
            "DETECTED"
        } else {
            "MISSED"
        }
    );
    if !(det_a && det_b && det_c) {
        fails.push(format!(
            "neg (a)={md_a:.3e} (b)={md_b:.3e} (c)={md_c:.3e} ≤ {DS4_MTP_NEG_THRESH:.0e} — 음성 미탐지"
        ));
    }
    report.push_str(&format!(
        " · neg a={md_a:.1e} b={md_b:.1e}(bias {bias_mag:.1e}) c={md_c:.1e} conf_chain={:.4}",
        conf_chain[0]
    ));

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | dims d={dim} vocab={vocab} hc={hc} block={} noise={} rank={} | {report} | ALL PASS",
            dims.dspark_block, dims.dspark_noise_token, rank
        ))
    } else {
        Err(format!(
            "ds4-mtp 실패 — {} (device: {dev}, dims d={dim} vocab={vocab})",
            fails.join(", ")
        ))
    }
}

/// ds4-mtp-neg — 음성대조 단독 팔(원장 17호 — 검증 계기 자체 검증).
/// 축소 체인(main_x·링·attn 서브블록 + 마르코프 — MoE/헤드 없이)에서
/// 3종 결함 클래스가 값 탐지되는지: (a) 트렁크 타깃 층 순열(main_x·X'
/// 에서) · (b) 마르코프 누락(|bias| 스케일 — 로짓 편차 하한) ·
/// (c) noise 토큰 id 오류(X' 에서). 전부 NEG-DETECTED.
pub fn cuda_ds4_mtp_negative_check(dir: &str) -> Result<String, String> {
    let dir = Path::new(dir);
    let cfg_path = dir.join("config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = Ds4MtpDims::from_config(&cfg_text)?;
    let ad = Ds4AttnDims::from_config(&cfg_text)?;
    let (dim, hc, vocab, rank) = (dims.dim, dims.hc, dims.vocab, dims.markov_rank);
    let ar = StArchive::open(dir)?;
    let mut modl = Ds4MtpCuda::open(&cfg_text)?;
    let dev = modl.device_name().to_string();

    let lay = load_mtp_attn_layer(&ar, &ad)?;
    modl.register_draft_layer(&lay)?;
    modl.register_sink(&lay.sink)?;
    let rope = o_rope_build(ad.rope_head_dim, 16, 10000.0, false, 1.0, 0, 32.0, 1.0);
    modl.set_rope_main(&rope.cs, rope.half)?;
    let hc_attn = (
        plain_f32(&ar, "mtp.0.hc_attn_fn")?,
        plain_f32(&ar, "mtp.0.hc_attn_base")?,
        plain_f32(&ar, "mtp.0.hc_attn_scale")?,
    );
    modl.register_hc(
        &hc_attn.0, &hc_attn.1, &hc_attn.2, &hc_attn.0, &hc_attn.1, &hc_attn.2,
    )?;
    let (main_norm, attn_norm, ffn_norm, final_norm) = (
        plain_f32(&ar, "mtp.0.main_norm.weight")?,
        plain_f32(&ar, "mtp.0.attn_norm.weight")?,
        plain_f32(&ar, "mtp.0.ffn_norm.weight")?,
        plain_f32(&ar, "mtp.2.norm.weight")?,
    );
    modl.register_norms(&main_norm, &attn_norm, &ffn_norm, &final_norm)?;
    let proj = dequant_linear(&ar, "mtp.0.main_proj")?;
    modl.register_main_proj(&proj)?;
    let (w1, w2, conf_proj) = (
        plain_f32(&ar, "mtp.2.markov_head.markov_w1.weight")?,
        plain_kmat(&ar, "mtp.2.markov_head.markov_w2.weight")?,
        plain_f32(&ar, "mtp.2.confidence_head.proj.weight")?,
    );
    modl.register_markov(&w1, &w2, &conf_proj)?;

    let (t_main, pos, in_tok) = (8usize, 7usize, 1000u32);
    let b = dims.dspark_block;
    let planes: Vec<Vec<f32>> = (0..3)
        .map(|li| gen_unif(t_main * dim, 0x5EED_BA0B_0000_0000 + li as u64, 0.05))
        .collect();
    let mut mh = vec![0.0f32; t_main * 3 * dim];
    for i in 0..t_main * dim {
        for li in 0..3 {
            mh[i * 3 + li] = planes[li][i];
        }
    }
    let draft_ids = modl.dspark_draft_ids(in_tok);
    let emb_all = embed_rows(dir, &draft_ids)?;
    let x_in: Vec<Vec<f32>> = (0..b)
        .map(|ti| {
            let mut row = Vec::with_capacity(hc * dim);
            for _ in 0..hc {
                row.extend_from_slice(&emb_all[ti * dim..(ti + 1) * dim]);
            }
            row
        })
        .collect();
    let hc_eps = 1e-6f32;
    let iters = 20usize;

    // 모듈 정밀 산출(main_x·링·attn 서브블록·바이어스).
    let m_main_x = modl.dspark_main_x(&mh)?;
    let mut m_ring = modl.window_warm(&m_main_x)?;
    let main_x_tok = m_main_x[(t_main - 1) * dim..].to_vec();
    let (m_y, m_posts, m_combs) = modl.hc.hc_pre(modl.il_main, "attn", &x_in)?;
    let mut yflat = Vec::with_capacity(b * dim);
    for r in &m_y {
        yflat.extend_from_slice(r);
    }
    let m_xn = modl.norm_rows_pub(&yflat, b, "attn")?;
    let m_a = modl.dspark_decode_attn(&m_xn, &main_x_tok, &mut m_ring, pos)?;
    let a_rows: Vec<Vec<f32>> = m_a.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let m_x2 = modl.hc.hc_post(&a_rows, &x_in, &m_posts, &m_combs)?;
    let m_bias = modl.markov_logits_bias(&draft_ids)?;

    // (a) 트렁크 타깃 순열 오라클 — main_x·X' 이격.
    let mut mh_a = mh.clone();
    for i in 0..t_main * dim {
        mh_a[i * 3] = planes[1][i];
        mh_a[i * 3 + 1] = planes[0][i];
    }
    let o_main_x_a = o_dspark_main_x(&mh_a, &proj, &main_norm, dim, dims.rms_eps);
    let (md_a1, _) = maxdiff_nan(&m_main_x, &o_main_x_a);
    // 순열 변형 체인 X'(attn 서브블록) — 오라클 미러.
    let mut ring_a = o_dspark_window_warm(&lay, &o_main_x_a, &ad, &rope);
    let mut o_xn_a = Vec::with_capacity(b * dim);
    for r in &x_in {
        let (yi, _, _) = hc_pre_ref(
            r,
            dim,
            hc,
            &hc_attn.0,
            &hc_attn.1,
            &hc_attn.2,
            dims.rms_eps,
            hc_eps,
            iters,
        );
        o_xn_a.extend_from_slice(&o_rms_norm_weighted(&yi, &attn_norm, dims.rms_eps));
    }
    let tok_a = &o_main_x_a[(t_main - 1) * dim..];
    let a_a = o_dspark_decode_attn(&lay, &o_xn_a, tok_a, &mut ring_a, pos, &ad, &rope);
    let a_a_rows: Vec<Vec<f32>> = a_a.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let mut x2_a = Vec::with_capacity(b * hc * dim);
    for ti in 0..b {
        let (_, post, comb) = hc_pre_ref(
            &x_in[ti],
            dim,
            hc,
            &hc_attn.0,
            &hc_attn.1,
            &hc_attn.2,
            dims.rms_eps,
            hc_eps,
            iters,
        );
        x2_a.extend_from_slice(&hc_post_ref(
            &a_a_rows[ti],
            &x_in[ti],
            &post,
            &comb,
            dim,
            hc,
        ));
    }
    let m_x2_flat: Vec<f32> = m_x2.iter().flatten().copied().collect();
    let (md_a2, _) = maxdiff_nan(&m_x2_flat, &x2_a);
    println!(
        "device: {dev} | ds4-mtp-neg (a) trunk-perm oracle: main_x maxdiff={md_a1:.3e} X' maxdiff={md_a2:.3e} | FAIL(expected)"
    );
    let det_a = md_a1 > DS4_MTP_NEG_THRESH && md_a2 > DS4_MTP_NEG_THRESH;

    // (b) 마르코프 누락 — 바이어스 크기(로짓 편차 하한) + out_ids 민감성.
    let (bias_mag, _) = maxdiff_nan(&m_bias, &vec![0.0f32; m_bias.len()]);
    let mut ids_b = draft_ids.clone();
    ids_b[0] = in_tok + 1;
    let m_bias_b = modl.markov_logits_bias(&ids_b)?;
    let (md_b_ids, _) = maxdiff_nan(&m_bias, &m_bias_b);
    println!(
        "device: {dev} | ds4-mtp-neg (b) bias-omitted scale |bias|max={bias_mag:.3e} out-ids sensitivity={md_b_ids:.3e} | FAIL(expected)"
    );
    let det_b = bias_mag > DS4_MTP_NEG_THRESH && md_b_ids > DS4_MTP_NEG_THRESH;

    // (c) noise id 오류 — 임베드 행 교체 → attn 서브블록 X' 이격.
    let mut ids_c = draft_ids.clone();
    for id in ids_c.iter_mut().skip(1) {
        *id = dims.dspark_noise_token - 1;
    }
    let emb_c = embed_rows(dir, &ids_c)?;
    let x_c: Vec<Vec<f32>> = (0..b)
        .map(|ti| {
            let mut row = Vec::with_capacity(hc * dim);
            for _ in 0..hc {
                row.extend_from_slice(&emb_c[ti * dim..(ti + 1) * dim]);
            }
            row
        })
        .collect();
    let mut o_xn_c = Vec::with_capacity(b * dim);
    for r in &x_c {
        let (yi, _, _) = hc_pre_ref(
            r,
            dim,
            hc,
            &hc_attn.0,
            &hc_attn.1,
            &hc_attn.2,
            dims.rms_eps,
            hc_eps,
            iters,
        );
        o_xn_c.extend_from_slice(&o_rms_norm_weighted(&yi, &attn_norm, dims.rms_eps));
    }
    let mut ring_c = o_dspark_window_warm(&lay, &m_main_x, &ad, &rope);
    let a_c = o_dspark_decode_attn(&lay, &o_xn_c, &main_x_tok, &mut ring_c, pos, &ad, &rope);
    let a_c_rows: Vec<Vec<f32>> = a_c.chunks_exact(dim).map(|c| c.to_vec()).collect();
    let mut x2_c = Vec::with_capacity(b * hc * dim);
    for ti in 0..b {
        let (_, post, comb) = hc_pre_ref(
            &x_c[ti],
            dim,
            hc,
            &hc_attn.0,
            &hc_attn.1,
            &hc_attn.2,
            dims.rms_eps,
            hc_eps,
            iters,
        );
        x2_c.extend_from_slice(&hc_post_ref(&a_c_rows[ti], &x_c[ti], &post, &comb, dim, hc));
    }
    let (md_c, _) = maxdiff_nan(&m_x2_flat, &x2_c);
    println!(
        "device: {dev} | ds4-mtp-neg (c) noise-id oracle (128799→{}) : X' maxdiff={md_c:.3e} | FAIL(expected)",
        dims.dspark_noise_token - 1
    );
    let det_c = md_c > DS4_MTP_NEG_THRESH;

    if det_a && det_b && det_c {
        Err(format!(
            "NEG-DETECTED (a) trunk-perm main_x={md_a1:.3e}/X'={md_a2:.3e} (b) bias-omitted |bias|={bias_mag:.3e}/ids={md_b_ids:.3e} (c) noise-id X'={md_c:.3e} > {DS4_MTP_NEG_THRESH:.0e} — 검증계기 정상(타깃 순열·바이어스 누락·noise id 편차 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={md_a1:.3e}/{md_a2:.3e} (b)={bias_mag:.3e}/{md_b_ids:.3e} (c)={md_c:.3e} ≤ {DS4_MTP_NEG_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}

/// mtp 전문가 1개 적재(loader.rs expert_at L401-409 미러).
fn load_expert_w(ar: &StArchive, base: &str) -> Result<ExpertW, String> {
    Ok(ExpertW {
        w1: dequant_linear(ar, &format!("{base}.w1"))?,
        w2: dequant_linear(ar, &format!("{base}.w2"))?,
        w3: dequant_linear(ar, &format!("{base}.w3"))?,
    })
}
