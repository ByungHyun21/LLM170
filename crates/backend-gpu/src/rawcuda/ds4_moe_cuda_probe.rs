//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: (i)/(ii)/(iii)/음성대조 케이스는 각각
//!    독립 Ds4MoeCuda(독립 가중치 상주)로 실행 — 케이스 간 디바이스 상태
//!    공유 없음(단일 상주 원칙도 케이스별 drop으로 준수).
//! ② 형상은 실측 config에서 자동 판독(D:/models/DeepSeek-V4-Flash-Vision-
//!    Exp-exl3-3.04bpw — 인자/LLM170_DS4_EXL3 오버라이드). 라우팅 존
//!    (해시 표 룩업·동률·바이어스 낙하·클램프 경계) 고정 케이스.
//! ③ 캡처-재생: 실측 EXL3 트렐리스 픽스처(오프셋 판독 — 아카이브 전량
//!    적재 금지 계약, loader.rs L1-4) — 모듈과 오라클이 동일 디양자화
//!    f32 가중치를 소비(트렐리스 참조 디코드는 본 파일 미러).
//! ④ 종단 값이 불변량: 판정은 moe_ffn 출력 값 maxdiff + 라우팅 이산
//!    선택(전문가 id 순열 — 순서 포함) exact-match(과제 계약:
//!    "routing ids EXACT — never argmax-only"). 라우팅 가중치도 비교.
//!
//! [오라클 — core deepseek4 미러(값 maxdiff 판정의 유일 기준)]
//! crates/core/src/deepseek4/stages/moe.rs(CPU 황금 기준)을 그대로 재생:
//! - 게이트: gemm_nt 순차 f32 누산(deepseek4/ops.rs L58-75: k 오름차순,
//!   곱·가산 각 1회 반올림) → sqrtsoftplus(moe.rs gate_scores L42-50 —
//!   softplus = core ops.rs L138-140: x>20 → x, else ln_cr(exp_cr(x)+1).
//!   exp_cr(core ops.rs L52-89)·ln_cr(L91-121)은 f64 FMA 호너 DAG의
//!   직이식 미러(커널 ds4_exp_cr/ds4_ln_cr과 리터럴까지 동일 — 비트동일).
//! - 해시 라우팅(L0-2): tid2eid 행 룩업(moe.rs route_hash L81-85 +
//!   moe_forward L152-159) — w = s[sel]/Σ(행 순서 순차)×1.5(route L67-79).
//! - noaux_tc(L3+): sel = topk(s + bias)(동점 낮은 인덱스 — topk_stable
//!   L53-65), w = gather(s, sel) bias 미포함(route_routed L87-96).
//! - 전문가 FFN(moe.rs expert_ffn L98-131): FP8-sim 128블록(deepseek4/
//!   ops.rs fp8_sim L174-190 — amax 하한 1e-4·pow2_ceil·e4m3 RNE 왕복)
//!   → w1/w3 → 각 bf16 경계 → 비대칭 클램프 SwiGLU(swiglu_limit L361-364:
//!   gate-proj max L·up-proj [-L,L]) ×라우팅 가중치 → bf16 → FP8-sim →
//!   w2 → bf16.
//! - moe_forward L133-216: by_expert BTreeMap(전문가 id 오름차순 누산) →
//!   공유 무가중 후행 → 최종 bf16 경계(L199-212).
//! - 전문가 가중치: EXL3 trellis 디양자화 f32 — loader.rs linear L203-247
//!   (= llm170_exl3 Exl3Linear dequant_block_f64: mul1 코드북 trellis.rs
//!   mul1_decode L47-58 → 128×128 블록 had128 f64 → suh·svh 스케일)의
//!   미러. 실측 전문가 K=3 링·공유 K=5 링(trellis [256,128,48]·
//!   [256,128,80] — 2026-10-05 실측).
//!
//! [검증층 원장 — 실측 2026-10-05, RTX 4070 SUPER(sm_89) — 검증 호스트]
//! (i) 해시 층 L0 실측(실 tid2eid [7,128000]·트렐리스 전문가 K=3·공유 K=5)
//! t=2 np=12: route ids EXACT(오름차 순서 포함) · route-w maxdiff 0.000e0 ·
//! out maxdiff 0.000e0 nan=0 — 순차 누산 미러·-fmad=false 계약의 비트동일
//! 실증(임계 ROUTE_W 1e-6·MOE_VAL 3e-4 무한대 여유).
//! (ii) 라우티드 층 L3 실측 noaux_tc(실 gate F16+bias) t=2 np=12: 동일 —
//! ids EXACT · w/out 0.000e0. (iii) 합성 존(동률 하위 슬롯 {10,200} — 낮은
//! id 10 획득·바이어스 낙하 e6 탈락·클램프 채널 g>10/g<-10/u>10/u<-10 전
//! 계급 도달) dim=128 inter=128: ids EXACT · 0.000e0.
//! 음성대조: (a) bias 포함 가중치식 ids_same=true 하 w_maxdiff 4.048e-2 ·
//! (b) 전치 tid2eid ids_differ=true · (c) 클램프 누락 maxdiff 7.875e0 > 3e-4
//! → 전계급 NEG-DETECTED(비영 exit + 마커).
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 개발기(RTX
//! 4070 SUPER, sm_89)은 정합 호스트일 뿐 — 타이밍 판단 근거 아님.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 — scripts/
//! cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::ds4_moe_cuda::{Ds4MoeCuda, Ds4MoeDims};
use crate::rawcuda::exl3_cuda::{JParser, JVal, StArchive};
use crate::rawcuda::exl3_cuda_probe::{PERM_INV, Rng, f16_to_f32, f16le, maxdiff_nan};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// 라우팅 가중치 maxdiff 임계(s/Σ×1.5 f32 — 비트동일 기대, 임계는 여유).
const ROUTE_W_THRESH: f32 = 1e-6;
/// moe_ffn 출력 값 maxdiff 임계 — plans/124 §1 GEMV 체인 등급(3e-4).
/// 순차 누산 미러·-fmad=false 계약상 비트동일(0.0) 기대.
const MOE_VAL_THRESH: f32 = 3e-4;
/// DS4 EXL3 픽스처 기본 경로(하네스 계약 — 인자·LLM170_DS4_EXL3 우선).
const DS4_EXL3_DEFAULT: &str = "D:/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw";

// ── core 트랜센던트 직이식(커널 ds4_exp_cr/ds4_ln_cr과 동일 DAG) ──
/// f64 FMA 호너 13차 exp — core ops.rs L52-89 리터럴 그대로(변경 금지).
fn exp_cr_mirror(x: f32) -> f32 {
    let xd = x as f64;
    if xd > 88.72 {
        return f32::INFINITY;
    }
    if xd < -103.97 {
        return 0.0;
    }
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    const INV_LN2: f64 = 1.442_695_040_888_963_4; // log2(e) — std 상수 비트동일
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

/// f64 자연로그 — atanh 급수 fma 호너. core ops.rs L91-121 리터럴 그대로.
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

/// softplus — core ops.rs L138-140 그대로(x>20 → x, else log1p_cr(exp_cr)).
fn softplus_mirror(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        // log1p_cr(core ops.rs L123-125): ln_cr(y as f64 + 1.0) as f32.
        ln_cr_mirror(exp_cr_mirror(x) as f64 + 1.0) as f32
    }
}

/// silu — core ops.rs L128-131 그대로.
fn silu_mirror(x: f32) -> f32 {
    x / (1.0 + exp_cr_mirror(-x))
}

/// sqrtsoftplus — deepseek4/ops.rs L355-358(softplus(g).sqrt()).
fn sqrtsoftplus_mirror(g: f32) -> f32 {
    softplus_mirror(g).sqrt()
}

/// SwiGLU limit 비대칭 클램프 — deepseek4/ops.rs L361-364.
fn swiglu_limit_mirror(gate: f32, up: f32, limit: f32) -> f32 {
    silu_mirror(gate.min(limit)) * up.clamp(-limit, limit)
}

/// bf16 RNE 경계 — deepseek4/ops.rs bf16_round L17-23.
fn bf16_round_mirror(x: f32) -> f32 {
    let b = x.to_bits();
    let hi = ((b >> 16) as u16) as u32 & 1;
    f32::from_bits(((b + 0x7FFF + hi) >> 16) << 16)
}

/// 2^ceil(log2(x)) — deepseek4/ops.rs pow2_ceil L77-86(x>0 정규수).
fn pow2_ceil_mirror(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    let l2 = e - 127 + i32::from(b & 0x7F_FFFF != 0);
    f32::from_bits(((l2 + 127) as u32) << 23)
}

/// f32 → e4m3fn RNE — deepseek4/ops.rs f32_to_e4m3 L88-129 직이식.
fn f32_to_e4m3_mirror(x: f32) -> u8 {
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

/// e4m3fn → f32(정확) — deepseek4/ops.rs e4m3_to_f32 L131-152 직이식.
fn e4m3_to_f32_mirror(u: u8) -> f32 {
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

/// FP8-sim — deepseek4/ops.rs fp8_sim L174-190 그대로(128블록).
fn fp8_sim_mirror(x: &mut [f32], cols: usize, block: usize) {
    let rows = x.len() / cols;
    for r in 0..rows {
        for c0 in (0..cols).step_by(block) {
            let blk = &mut x[r * cols + c0..r * cols + (c0 + block).min(cols)];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = pow2_ceil_mirror(amax * (1.0 / 448.0));
            for v in blk.iter_mut() {
                let q = (*v / s).clamp(-448.0, 448.0);
                *v = e4m3_to_f32_mirror(f32_to_e4m3_mirror(q)) * s;
            }
        }
    }
}

/// y[m,n] = Σ_k x[k]·w[k,n] — deepseek4/ops.rs gemm_nt L58-75(k 오름차순
/// 순차 누산, w k-major). 커널 ds4_gate/ds4_gemv_f32_ptr과 동일 계약.
fn gemm_nt_mirror(x: &[f32], w: &[f32], k: usize, n: usize, y: &mut [f32]) {
    let m = x.len() / k;
    y.fill(0.0);
    for i in 0..m {
        let xr = &x[i * k..(i + 1) * k];
        let yr = &mut y[i * n..(i + 1) * n];
        for (kk, &xv) in xr.iter().enumerate() {
            let wr = &w[kk * n..(kk + 1) * n];
            for (j, &wv) in wr.iter().enumerate() {
                yr[j] += xv * wv;
            }
        }
    }
}

// ── 라우팅 오라클 — moe.rs L42-96 그대로 ──
/// 게이트 스코어 — moe.rs gate_scores L42-50(gemm + sqrtsoftplus).
fn gate_scores_mirror(x: &[f32], gw: &[f32], n_routed: usize, dim: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n_routed];
    gemm_nt_mirror(x, gw, dim, n_routed, &mut s);
    for v in s.iter_mut() {
        *v = sqrtsoftplus_mirror(*v);
    }
    s
}

/// 안정 top-k — moe.rs topk_stable L53-65(값 내림차순·동점 낮은 인덱스).
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

/// 정규화 — moe.rs route L67-79(Σ 선택 순서 순차, w = s/Σ×scale, eid 정렬).
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

/// 해시 라우팅 — moe.rs route_hash L81-85.
fn route_hash_mirror(scores: &[f32], tid2eid_row: &[i64], route_scale: f32) -> Vec<(usize, f32)> {
    let sel: Vec<usize> = tid2eid_row.iter().map(|&e| e as usize).collect();
    route_mirror(scores, sel, route_scale)
}

/// noaux_tc 라우팅 — moe.rs route_routed L87-96(bias 선택 전용).
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

// ── 전문가 FFN 오라클 — moe.rs expert_ffn L98-131 그대로 ──
/// 전문가 가중치 f32 k-major 트리플(w1/w3 [dim][inter]·w2 [inter][dim]).
#[derive(Clone)]
struct ExpertW {
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

/// 게이트 오라클 — loader.rs block 게이트 L358-371 계약(해시 층 bias 무시).
struct GateMirror {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    tid2eid: Option<Vec<i64>>,
}

/// 전문가 FFN 1토큰 — moe.rs expert_ffn L98-131 미러(FP8-sim·bf16·클램프).
/// `limit` 오버라이드는 음성대조(c) 전용(None이면 cfg 값).
fn expert_ffn_mirror(
    x: &[f32],
    e: &ExpertW,
    weight: f32,
    dim: usize,
    inter: usize,
    limit: f32,
) -> Vec<f32> {
    let mut xq = x.to_vec();
    fp8_sim_mirror(&mut xq, dim, 128);
    let mut g = vec![0.0f32; inter];
    gemm_nt_mirror(&xq, &e.w1, dim, inter, &mut g);
    for v in g.iter_mut() {
        *v = bf16_round_mirror(*v);
    }
    let mut u = vec![0.0f32; inter];
    gemm_nt_mirror(&xq, &e.w3, dim, inter, &mut u);
    for v in u.iter_mut() {
        *v = bf16_round_mirror(*v);
    }
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        h[i] = swiglu_limit_mirror(g[i], u[i], limit) * weight;
    }
    for v in h.iter_mut() {
        *v = bf16_round_mirror(*v);
    }
    fp8_sim_mirror(&mut h, inter, 128);
    let mut y = vec![0.0f32; dim];
    gemm_nt_mirror(&h, &e.w2, inter, dim, &mut y);
    for v in y.iter_mut() {
        *v = bf16_round_mirror(*v);
    }
    y
}

/// moe_forward 오라클 — moe.rs L133-216 미러(by_expert 오름차순 누산·공유
/// 무가중 후행·최종 bf16). experts는 (id → 가중치) 맵(모듈과 동일 바이트).
fn moe_forward_mirror(
    x: &[f32],
    input_ids: &[u32],
    gw: &GateMirror,
    shared: &ExpertW,
    experts: &HashMap<usize, ExpertW>,
    d: &Ds4MoeDims,
    is_hash: bool,
    limit_override: Option<f32>,
) -> Vec<f32> {
    let (dim, t) = (d.dim, x.len() / d.dim);
    let inter = d.inter;
    let limit = limit_override.unwrap_or(d.swiglu_limit);
    // 1) 토큰별 라우팅(moe.rs L146-165).
    let routes: Vec<Vec<(usize, f32)>> = (0..t)
        .map(|ti| {
            let xt = &x[ti * dim..(ti + 1) * dim];
            let scores = gate_scores_mirror(xt, &gw.weight, d.n_routed, dim);
            if is_hash {
                let base = input_ids[ti] as usize * d.n_active;
                let row =
                    &gw.tid2eid.as_ref().expect("해시 층에 tid2eid 필요")[base..base + d.n_active];
                route_hash_mirror(&scores, row, d.route_scale)
            } else {
                route_routed_mirror(
                    &scores,
                    gw.bias.as_ref().expect("라우티드 층에 bias 필요"),
                    d.n_active,
                    d.route_scale,
                )
            }
        })
        .collect();
    // 2) 전문가 id 오름차순 누산(moe.rs L168-196 BTreeMap 계약).
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
            let out = expert_ffn_mirror(&x[ti * dim..(ti + 1) * dim], e, w, dim, inter, limit);
            for (yi, &ov) in y[ti * dim..(ti + 1) * dim].iter_mut().zip(out.iter()) {
                *yi += ov;
            }
        }
    }
    // 3) 공유 전문가 — 무가중 후행 + 최종 bf16(moe.rs L199-212).
    for ti in 0..t {
        let sh = expert_ffn_mirror(&x[ti * dim..(ti + 1) * dim], shared, 1.0, dim, inter, limit);
        for (yi, &sv) in y[ti * dim..(ti + 1) * dim].iter_mut().zip(sh.iter()) {
            *yi += sv;
        }
        for yi in y[ti * dim..(ti + 1) * dim].iter_mut() {
            *yi = bf16_round_mirror(*yi);
        }
    }
    y
}

// ── EXL3 trellis 디양자화 미러 — llm170_exl3 trellis.rs + loader.rs linear ──
/// mul1 코드북 — trellis.rs mul1_decode L47-58 원식(f32 mul_add).
fn mul1_decode_mirror(word: u16) -> f32 {
    let x = (word as u32).wrapping_mul(0x83DC_D12D);
    let sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    let f = 1024.0f32 + sum as f32;
    f.mul_add(0.00676727294921875, -10.3828125) // f16(0x1eee)·f16(0xc931) 정확값
}

/// 타일 비트링에서 워드 t — trellis.rs tile_word L60-70 직이식.
fn tile_word_mirror(u32s: &[u32], krate: u32, t: u32) -> u16 {
    let words32 = 8 * krate as usize;
    let b0 = (t * krate + (krate + 256 * krate - 16)) as usize;
    let b1 = b0 + 16;
    let i0 = (b0 / 32) % words32;
    let i1 = ((b1 - 1) / 32) % words32;
    let s = ((b1 - 1) / 32 + 1) * 32 - b1;
    let merged = ((u32s[i0] as u64) << 32) | u32s[i1] as u64;
    ((merged >> s) & 0xFFFF) as u16
}

/// f64 WHT-128 — trellis.rs had128 L279-297 직이식(스케일 없음).
fn had128_f64(v: &mut [f64]) {
    let mut width = 1usize;
    while width < 128 {
        let mut base = 0usize;
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

/// trellis 3중 + suh/svh → f32 k-major [k][n] — loader.rs linear L203-247
/// (= Exl3Linear::view().dequant_block_f64 128×128 블록: mul1 코드북 →
/// had128 f64 행/열 → /128 → suh·svh)의 전행렬 미러. 128행 스트립 스레드
/// 병렬(결정적 — 스트립별 독립, loader.rs L218-246과 동일 구조).
fn dequant_kmat(
    tre16: &[u16],
    suh: &[u16],
    svh: &[u16],
    k: usize,
    n: usize,
    krate: u32,
) -> Result<Vec<f32>, String> {
    let tw = 16 * krate as usize;
    if tre16.len() != (k / 16) * (n / 16) * tw || suh.len() != k || svh.len() != n {
        return Err(format!(
            "trellis 형상 불일치: tre={} suh={} svh={} k={k} n={n} K={krate}",
            tre16.len(),
            suh.len(),
            svh.len()
        ));
    }
    let ntiles = n / 16;
    let mut w = vec![0.0f32; k * n];
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .min(k / 128)
        .max(1);
    let per = (k / 128).div_ceil(threads) * 128;
    std::thread::scope(|sc| -> Result<(), String> {
        let mut off = 0usize;
        let mut rest = w.as_mut_slice();
        while off < k {
            let rows_here = per.min(k - off);
            let (part, tail) = rest.split_at_mut(rows_here * n);
            rest = tail;
            let k0 = off;
            let tre = &*tre16; // 읽기 공유(스레드별 스트립 독립)
            sc.spawn(move || -> Result<(), String> {
                for (ci, chunk) in part.chunks_mut(128 * n).enumerate() {
                    let kb = k0 + ci * 128;
                    for n0 in (0..n).step_by(128) {
                        // 블록 128×128 — trellis.rs dequant_view L128-158 미러.
                        let mut full = [0.0f64; 128 * 128];
                        let mut tile = [0.0f32; 256];
                        for kt in 0..8 {
                            for nt2 in 0..8 {
                                let base = ((kb / 16 + kt) * ntiles + (n0 / 16 + nt2)) * tw;
                                let u32s: Vec<u32> = (0..8 * krate as usize)
                                    .map(|i| {
                                        (tre[base + 2 * i] as u32)
                                            | ((tre[base + 2 * i + 1] as u32) << 16)
                                    })
                                    .collect();
                                for pos in 0..256usize {
                                    let t = PERM_INV[pos] as u32;
                                    tile[pos] =
                                        mul1_decode_mirror(tile_word_mirror(&u32s, krate, t));
                                }
                                for r in 0..16 {
                                    for c in 0..16 {
                                        full[(kt * 16 + r) * 128 + nt2 * 16 + c] =
                                            tile[r * 16 + c] as f64;
                                    }
                                }
                            }
                        }
                        for row in full.as_chunks_mut::<128>().0 {
                            had128_f64(row);
                        }
                        let mut col = [0.0f64; 128];
                        for c in 0..128 {
                            for (r, v) in col.iter_mut().enumerate() {
                                *v = full[r * 128 + c];
                            }
                            had128_f64(&mut col);
                            for (r, v) in col.iter().enumerate() {
                                full[r * 128 + c] = v / 128.0;
                            }
                        }
                        for i in 0..128 {
                            let si = f16_to_f32(suh[kb + i]) as f64;
                            for j in 0..128 {
                                let sj = f16_to_f32(svh[n0 + j]) as f64;
                                chunk[i * n + n0 + j] = (si * sj * full[i * 128 + j]) as f32;
                            }
                        }
                    }
                }
                Ok(())
            });
            off += rows_here;
        }
        Ok(())
    })?;
    Ok(w)
}

/// trellis 3중(u16 뷰) — raw 바이트 LE 변환(단일 소스).
fn u16_view(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// 아카이브에서 선형 1개 디양자화 — loader.rs linear(L203-247) 미러.
fn dequant_linear(ar: &StArchive, key: &str) -> Result<(usize, usize, Vec<f32>), String> {
    let shape = ar
        .shape_of(&format!("{key}.trellis"))
        .ok_or_else(|| format!("{key}.trellis 없음"))?
        .to_vec();
    if shape.len() != 3 || shape[2] % 16 != 0 {
        return Err(format!("{key}: trellis 형상 {shape:?} — 3차원·16배수 계약"));
    }
    let (k, n, krate) = (
        shape[0] as usize * 16,
        shape[1] as usize * 16,
        (shape[2] / 16) as u32,
    );
    let tre = u16_view(&ar.read(&format!("{key}.trellis"))?);
    let suh = u16_view(&ar.read(&format!("{key}.suh"))?);
    let svh = u16_view(&ar.read(&format!("{key}.svh"))?);
    let w = dequant_kmat(&tre, &suh, &svh, k, n, krate)?;
    Ok((k, n, w))
}

/// 전문가 3중 적재 — loader.rs expert_at(L401-409) 미러(w1/w3·w2 순서).
fn load_expert(ar: &StArchive, base: &str) -> Result<ExpertW, String> {
    let (k1, n1, w1) = dequant_linear(ar, &format!("{base}.w1"))?;
    let (k2, n2, w2) = dequant_linear(ar, &format!("{base}.w2"))?;
    let (k3, n3, w3) = dequant_linear(ar, &format!("{base}.w3"))?;
    if k1 != k3 || n1 != n3 || k1 != n2 || n1 != k2 {
        return Err(format!(
            "{base}: 형상 계약 위반 w1[{k1},{n1}] w2[{k2},{n2}] w3[{k3},{n3}]"
        ));
    }
    Ok(ExpertW { w1, w2, w3 })
}

// ── 픽스처 판독(config.json + 게이트 F16/I64) ──
/// config.json → (MoE 형상, n_hash, vocab) — JParser(exl3_cuda 최소 JSON).
fn ds4_cfg_from_dir(dir: &str) -> Result<(Ds4MoeDims, usize, usize), String> {
    let b = std::fs::read(format!("{dir}/config.json"))
        .map_err(|e| format!("{dir}/config.json: {e}"))?;
    let v = JParser { b: &b, p: 0 }.parse()?;
    let u = |k: &str| -> Result<usize, String> {
        v.get(k)
            .and_then(JVal::as_f64)
            .map(|x| x as usize)
            .ok_or_else(|| format!("config: {k} 없음"))
    };
    let g = |k: &str| -> Result<f64, String> {
        v.get(k)
            .and_then(JVal::as_f64)
            .ok_or_else(|| format!("config: {k} 없음"))
    };
    Ok((
        Ds4MoeDims {
            dim: u("hidden_size")?,
            n_routed: u("n_routed_experts")?,
            n_active: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            route_scale: g("routed_scaling_factor")? as f32,
            swiglu_limit: g("swiglu_limit")? as f32,
        },
        u("num_hash_layers")?,
        u("vocab_size")?,
    ))
}

/// 게이트 적재 — loader.rs 게이트 L358-371 계약: weight F16 [n,k] 행 우선 →
/// k-major 전치(plain_kmat L252-268)·bias F16(라우티드만)·tid2eid I64(해시만).
fn load_gate(
    ar: &StArchive,
    il: usize,
    d: &Ds4MoeDims,
    is_hash: bool,
) -> Result<GateMirror, String> {
    let p = format!("layers.{il}");
    let raw = ar.read(&format!("{p}.ffn.gate.weight"))?;
    if raw.len() != d.n_routed * d.dim * 2 {
        return Err(format!(
            "{p}.ffn.gate.weight: {}B != n_routed·dim·2={}",
            raw.len(),
            d.n_routed * d.dim * 2
        ));
    }
    // 행 우선 [n_routed][dim] → k-major [dim][n_routed](loader.rs L258-265).
    let mut weight = vec![0.0f32; d.dim * d.n_routed];
    for o in 0..d.n_routed {
        for kk in 0..d.dim {
            weight[kk * d.n_routed + o] = f16_to_f32(f16le(&raw, o * d.dim + kk));
        }
    }
    let bias = if is_hash {
        None // 해시 층 bias 무시 — loader.rs L362-366(model.py Gate 계약)
    } else {
        let braw = ar.read(&format!("{p}.ffn.gate.bias"))?;
        if braw.len() != d.n_routed * 2 {
            return Err(format!(
                "{p}.ffn.gate.bias: {}B != n_routed·2={}",
                braw.len(),
                d.n_routed * 2
            ));
        }
        Some(
            (0..d.n_routed)
                .map(|i| f16_to_f32(f16le(&braw, i)))
                .collect(),
        )
    };
    let tid2eid = if is_hash {
        let traw = ar.read(&format!("{p}.ffn.gate.tid2eid"))?;
        if traw.len() % 8 != 0 {
            return Err(format!("{p}.ffn.gate.tid2eid: i64 정렬 아님"));
        }
        Some(
            traw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        )
    } else {
        None
    };
    Ok(GateMirror {
        weight,
        bias,
        tid2eid,
    })
}

/// 결정론 입력 — 균일 ±0.5(잔차 스트림 계급).
fn gen_xs(dim: usize, t: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..t)
        .map(|_| {
            (0..dim)
                .map(|_| ((rng.next_f64() * 2.0 - 1.0) * 0.5) as f32)
                .collect()
        })
        .collect()
}

/// 해시 도메인 계기(계기의 계기): 표 값·행 길이·선택 eid 전개 검증.
fn hash_table_check(t2e: &[i64], d: &Ds4MoeDims, vocab: usize, tids: &[u32]) -> Result<(), String> {
    if t2e.len() != vocab * d.n_active {
        return Err(format!(
            "tid2eid len={} != vocab·k={}",
            t2e.len(),
            vocab * d.n_active
        ));
    }
    for &tid in tids {
        for j in 0..d.n_active {
            let e = t2e[tid as usize * d.n_active + j];
            if e < 0 || e as usize >= d.n_routed {
                return Err(format!("tid2eid[{tid}][{j}]={e} — 전문가 id 도메인 위반"));
            }
        }
    }
    Ok(())
}

/// 전문가 상주: 선택 전문가를 실측 픽스처에서 디양자화·등록(모듈·오라클
/// 동일 f32 바이트 공유). 반환 맵은 오라클 소비.
fn stage_real_experts(
    m: &mut Ds4MoeCuda,
    ar: &StArchive,
    il: usize,
    need: &[u32],
) -> Result<HashMap<usize, ExpertW>, String> {
    let mut experts = HashMap::new();
    for &e in need {
        let w = load_expert(ar, &format!("layers.{il}.ffn.experts.{e}"))?;
        m.add_expert_f32(e as usize, &w.w1, &w.w2, &w.w3)?;
        experts.insert(e as usize, w);
    }
    Ok(experts)
}

/// 선택 전개 → 정렬된 필요 전문가 리스트(중복 제거 — 오름차순).
fn need_experts(sel: &[Vec<(u32, f32)>]) -> Vec<u32> {
    let mut need: Vec<u32> = Vec::new();
    for row in sel {
        for &(e, _) in row {
            if !need.contains(&e) {
                need.push(e);
            }
        }
    }
    need.sort_unstable();
    need
}

/// 1개 실측 케이스: 모듈 vs 오라클 — 라우팅 이산 exact + 가중치 maxdiff +
/// moe_ffn 출력 값 maxdiff. (dev, 요약, 통과, 실패 상세) 반환.
#[allow(clippy::too_many_arguments)]
fn run_real_case(
    tag: &str,
    ar: &StArchive,
    d: &Ds4MoeDims,
    vocab: usize,
    il: usize,
    is_hash: bool,
    t: usize,
    seed: u64,
) -> Result<(String, String, bool, String), String> {
    let dim = d.dim;
    let gw = load_gate(ar, il, d, is_hash)?;
    let xs = gen_xs(dim, t, seed);
    let flat: Vec<f32> = xs.concat();
    let tids: Vec<u32> = if t > 1 { vec![7, 128000] } else { vec![7] };
    // 오라클 라우팅(먼저 — 필요 전문가 산출).
    let routes: Vec<Vec<(usize, f32)>> = (0..t)
        .map(|ti| {
            let scores = gate_scores_mirror(&xs[ti], &gw.weight, d.n_routed, dim);
            if is_hash {
                let base = tids[ti] as usize * d.n_active;
                let row = &gw.tid2eid.as_ref().unwrap()[base..base + d.n_active];
                route_hash_mirror(&scores, row, d.route_scale)
            } else {
                route_routed_mirror(
                    &scores,
                    gw.bias.as_ref().unwrap(),
                    d.n_active,
                    d.route_scale,
                )
            }
        })
        .collect();
    if is_hash {
        hash_table_check(gw.tid2eid.as_ref().unwrap(), d, vocab, &tids)?;
        // 계기의 계기: 해시 선택은 표 행과 정확히 일치(정렬 후).
        for (ti, r) in routes.iter().enumerate() {
            let row = &gw.tid2eid.as_ref().unwrap()
                [tids[ti] as usize * d.n_active..(tids[ti] as usize + 1) * d.n_active];
            let mut got: Vec<usize> = r.iter().map(|&(e, _)| e).collect();
            let mut want: Vec<usize> = row.iter().map(|&e| e as usize).collect();
            got.sort_unstable();
            want.sort_unstable();
            if got != want {
                return Err(format!(
                    "zones: 해시 토큰 {ti} 선택 {got:?} ≠ 표 행 {want:?} — 오라클 결함"
                ));
            }
        }
    }
    let need = need_experts(
        &routes
            .iter()
            .map(|r| r.iter().map(|&(e, w)| (e as u32, w)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    // 모듈 조립 — 실측 게이트·전문가(K=3)·공유(K=5) 상주.
    let mut m = Ds4MoeCuda::new(d.clone())?;
    let dev = m.device_name().to_string();
    m.set_gate_f32(&gw.weight, gw.bias.as_deref(), gw.tid2eid.as_deref())?;
    let shared = load_expert(ar, &format!("layers.{il}.ffn.shared_experts"))?;
    m.set_shared_f32(&shared.w1, &shared.w2, &shared.w3)?;
    let experts = stage_real_experts(&mut m, ar, il, &need)?;
    // 오라클 종단.
    let oracle_out = moe_forward_mirror(&flat, &tids, &gw, &shared, &experts, d, is_hash, None);
    // (1) 라우팅 이산 — 선택 순열 exact(오름차 순서 포함).
    let got_sel = if is_hash {
        m.moe_route_hash(&xs, &tids)?
    } else {
        m.moe_route_routed(&xs)?
    };
    let mut ids_ok = got_sel.len() == routes.len();
    for (ti, (g, w)) in got_sel.iter().zip(routes.iter()).enumerate() {
        let ge: Vec<u32> = g.iter().map(|&(e, _)| e).collect();
        let we: Vec<u32> = w.iter().map(|&(e, _)| e as u32).collect();
        if ge != we {
            ids_ok = false;
            eprintln!("  route mismatch t{ti}: got {ge:?} want {we:?}");
        }
    }
    // (2) 라우팅 가중치 maxdiff.
    let mut wmd = 0.0f32;
    for (g, w) in got_sel.iter().zip(routes.iter()) {
        for (a, b) in g.iter().zip(w.iter()) {
            wmd = wmd.max((a.1 - b.1).abs());
        }
    }
    // (3) moe_ffn 종단 값.
    let got_out = if is_hash {
        m.moe_ffn_hash(&xs, &tids)?
    } else {
        m.moe_ffn_routed(&xs)?
    };
    let mut md = 0.0f32;
    let mut nan = 0usize;
    for (g, w) in got_out.iter().zip(oracle_out.chunks(dim)) {
        let (m1, n1) = maxdiff_nan(g, w);
        md = md.max(m1);
        nan += n1;
    }
    let np: usize = routes.iter().map(|r| r.len()).sum();
    let pass = ids_ok && wmd <= ROUTE_W_THRESH && md <= MOE_VAL_THRESH && nan == 0;
    println!(
        "device: {dev} | ds4-moe {tag}: dim={} {}e top{} inter{} t={} np={} | route ids {} (w maxdiff={wmd:.3e}) | out maxdiff={md:.3e} nan={nan} | {}",
        d.dim,
        d.n_routed,
        d.n_active,
        d.inter,
        t,
        np,
        if ids_ok { "EXACT" } else { "MISMATCH" },
        if pass { "PASS" } else { "FAIL" }
    );
    let summary = format!(
        "{tag} ids {} w={wmd:.3e} md={md:.3e}",
        if ids_ok { "EXACT" } else { "MISMATCH" }
    );
    let fail = if pass {
        String::new()
    } else {
        format!("{tag} ids_ok={ids_ok} w_maxdiff={wmd:.3e} maxdiff={md:.3e} nan={nan}")
    };
    Ok((dev, summary, pass, fail))
}

/// 합성 전문가 3중(결정론 시드 — 스몰 형상 클램프 케이스).
/// ch 패턴(채널 4그룹×32): g>10(gate 상단 클램프)·g<-10(하단 무클램프)·
/// u>10·u<-10(up ±클램프) — x[0]=4.0 지배 항에 w1/w3[0][ch]=±3.0·2.5.
fn gen_engineered_expert(dim: usize, inter: usize, seed: u64) -> ExpertW {
    let mut rng = Rng::new(seed);
    let mk = |rng: &mut Rng, rows: usize, cols: usize| -> Vec<f32> {
        (0..rows * cols)
            .map(|_| ((rng.next_f64() * 2.0 - 1.0) * 0.02) as f32)
            .collect()
    };
    let mut w1 = mk(&mut rng, dim, inter);
    let mut w3 = mk(&mut rng, dim, inter);
    let w2 = mk(&mut rng, inter, dim);
    for ch in 0..inter {
        let g = match ch % 4 {
            0 => 3.0,  // g ≈ +12 → gate min 10
            1 => -3.0, // g ≈ -12 → 하단 무클램프(silu 원값)
            _ => 2.5,  // g ≈ +10 경계
        };
        let u = match ch % 4 {
            2 => 3.0,  // u ≈ +12 → up clamp +10
            3 => -3.0, // u ≈ -12 → up clamp -10
            _ => 2.5,  // u ≈ +10 경계
        };
        w1[ch] = g; // k-major [k][n]: k=0 행 — w1[0·inter + ch]
        w3[ch] = u;
    }
    ExpertW { w1, w2, w3 }
}

/// 합성 라우티드 게이트(스몰 형상) — 존 설계(계기의 계기로 검증):
/// · hi e=1..5: gw[0][e]=2.0, bias 0 → biased ≈ 2.83(상위 5).
/// · dropout e=6: gw[0][6]=2.0, bias=-1.0 → s는 높아도 biased ≈ 1.83로
///   탈락 — bias가 선택에 반영됨의 이산 증거(무시하면 ids 변화).
/// · tie e=10/200: 열 바이트 동일(gw[0]=0.8)·bias=0.5 동일 → biased 비트
///   동일 동률 — 낮은 id(10)가 슬롯 6 획득(동률 계약), 200은 탈락.
/// top-6 = {1,2,3,4,5,10}.
fn gen_engineered_gate(dim: usize, n_routed: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(0xE7D5_11C0);
    let mut gw = vec![0.0f32; dim * n_routed];
    let mut bias = vec![0.0f32; n_routed];
    for e in 0..n_routed {
        for kk in 1..dim {
            gw[kk * n_routed + e] = ((rng.next_f64() * 2.0 - 1.0) * 0.01) as f32;
        }
    }
    for &e in &[1usize, 2, 3, 4, 5] {
        gw[e] = 2.0;
    }
    gw[6] = 2.0;
    bias[6] = -1.0;
    let mut tie_col = vec![0.0f32; dim];
    tie_col[0] = 0.8;
    let mut rng2 = Rng::new(0xA11C_E5);
    for v in tie_col.iter_mut().skip(1) {
        *v = ((rng2.next_f64() * 2.0 - 1.0) * 0.01) as f32;
    }
    for &e in &[10usize, 200] {
        for (kk, &v) in tie_col.iter().enumerate() {
            gw[kk * n_routed + e] = v; // 동일 바이트 열 — 동률 제작
        }
        bias[e] = 0.5;
    }
    (gw, bias)
}

/// 스몰 형상 케이스(dim=128·inter=128): 라우팅 존(동률·바이어스 낙하) +
/// SwiGLU 클램프 경계(g>10·g<-10·u>10·u<-10 채널) — 오라클과 종단 비교.
/// 반환 (dev, 요약, 통과, 실패, 케이스 자산) — 자산은 음성대조 (a)/(c) 재사용.
struct EngineeredFx {
    d: Ds4MoeDims,
    gw: Vec<f32>,
    bias: Vec<f32>,
    shared: ExpertW,
    experts: HashMap<usize, ExpertW>,
    xs: Vec<Vec<f32>>,
}
impl EngineeredFx {
    fn build(seed: u64) -> Result<Self, String> {
        let d = Ds4MoeDims {
            dim: 128,
            n_routed: 256,
            n_active: 6,
            inter: 128,
            route_scale: 1.5,
            swiglu_limit: 10.0,
        };
        let (gw, bias) = gen_engineered_gate(d.dim, d.n_routed);
        let mut xs = gen_xs(d.dim, 1, seed);
        xs[0][0] = 4.0; // 지배 항 — fp8-sim 왕복 후에도 정확(4/2^-6=256 e4m3)
        let experts: HashMap<usize, ExpertW> = [1usize, 2, 3, 4, 5, 6, 10, 200]
            .iter()
            .map(|&e| {
                (
                    e,
                    gen_engineered_expert(d.dim, d.inter, seed ^ (e as u64).wrapping_mul(0x9E37)),
                )
            })
            .collect();
        let shared = gen_engineered_expert(d.dim, d.inter, seed ^ 0x5EED);
        Ok(EngineeredFx {
            d,
            gw,
            bias,
            shared,
            experts,
            xs,
        })
    }

    /// 존 무결성(계기의 계기): 오라클 선택이 설계 top-6 {1..5,10}와 일치,
    /// dropout(6)·tie 패자(200) 탈락 — 클램프 사전활성이 실제로 한계 초과.
    fn zones_check(&self) -> Result<(), String> {
        let scores = gate_scores_mirror(&self.xs[0], &self.gw, self.d.n_routed, self.d.dim);
        let biased: Vec<f32> = scores
            .iter()
            .zip(&self.bias)
            .map(|(&s, &b)| s + b)
            .collect();
        let top = topk_stable_mirror(&biased, self.d.n_active);
        let ids: Vec<usize> = top.iter().map(|&(i, _)| i).collect();
        // topk_stable은 선택 순서(내림차) 반환 — 집합 비교로 판정(모듈 계약은
        // 오름차 정렬 후 동일). dropout(6)·tie 패자(200) 탈락이 본체.
        let mut ids_sorted = ids.clone();
        ids_sorted.sort_unstable();
        if ids_sorted != vec![1, 2, 3, 4, 5, 10] {
            return Err(format!(
                "zones: 스몰 선택 {ids:?}(정렬 {ids_sorted:?}) ≠ {{1..5,10}} — dropout/bias·동률 설계 결함"
            ));
        }
        // 동률 비트 확인 — tie 쌍 biased 값이 정확히 동일 비트.
        let (a, b) = (biased[10], biased[200]);
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "zones: tie 쌍 biased 비트 상이 {a:e} vs {b:e} — 동률 설계 결함"
            ));
        }
        // 클램프 사전활성 — 선택 전문가 g/u 값이 실제로 ±10 초과.
        let mut xq = self.xs[0].clone();
        fp8_sim_mirror(&mut xq, self.d.dim, 128);
        let (mut g_hi, mut g_lo, mut u_hi, mut u_lo) = (0usize, 0usize, 0usize, 0usize);
        for (_, e) in self
            .experts
            .iter()
            .filter(|&(id, _)| [1, 2, 3, 4, 5].contains(id))
        {
            let mut g = vec![0.0f32; self.d.inter];
            gemm_nt_mirror(&xq, &e.w1, self.d.dim, self.d.inter, &mut g);
            let mut u = vec![0.0f32; self.d.inter];
            gemm_nt_mirror(&xq, &e.w3, self.d.dim, self.d.inter, &mut u);
            g_hi += g.iter().filter(|&&v| v > 10.0).count();
            g_lo += g.iter().filter(|&&v| v < -10.0).count();
            u_hi += u.iter().filter(|&&v| v > 10.0).count();
            u_lo += u.iter().filter(|&&v| v < -10.0).count();
        }
        if g_hi == 0 || g_lo == 0 || u_hi == 0 || u_lo == 0 {
            return Err(format!(
                "zones: 클램프 미도달 g>{g_hi} g<{g_lo} u>{u_hi} u<{u_lo} — 픽스처 결함"
            ));
        }
        Ok(())
    }
}

/// 스몰 형상 케이스 실행 — 라우팅 exact·가중치·종단 값 판정.
fn run_engineered_case(fx: &EngineeredFx) -> Result<(String, String, bool, String), String> {
    fx.zones_check()?;
    let d = fx.d.clone();
    let gw = GateMirror {
        weight: fx.gw.clone(),
        bias: Some(fx.bias.clone()),
        tid2eid: None,
    };
    let flat: Vec<f32> = fx.xs.concat();
    let tids = [0u32];
    let oracle_out =
        moe_forward_mirror(&flat, &tids, &gw, &fx.shared, &fx.experts, &d, false, None);
    let mut m = Ds4MoeCuda::new(d.clone())?;
    let dev = m.device_name().to_string();
    m.set_gate_f32(&fx.gw, Some(&fx.bias), None)?;
    m.set_shared_f32(&fx.shared.w1, &fx.shared.w2, &fx.shared.w3)?;
    let mut need: Vec<usize> = fx.experts.keys().copied().collect();
    need.sort_unstable();
    for e in need {
        let w = &fx.experts[&e];
        m.add_expert_f32(e, &w.w1, &w.w2, &w.w3)?;
    }
    let got_sel = m.moe_route_routed(&fx.xs)?;
    let scores = gate_scores_mirror(&fx.xs[0], &fx.gw, d.n_routed, d.dim);
    let want_sel = route_routed_mirror(&scores, &fx.bias, d.n_active, d.route_scale);
    let ids_ok = got_sel.len() == 1
        && got_sel[0].iter().map(|&(e, _)| e).collect::<Vec<_>>()
            == want_sel.iter().map(|&(e, _)| e as u32).collect::<Vec<_>>();
    let mut wmd = 0.0f32;
    for (g, w) in got_sel[0].iter().zip(want_sel.iter()) {
        wmd = wmd.max((g.1 - w.1).abs());
    }
    let got_out = m.moe_ffn_routed(&fx.xs)?;
    let (md, nan) = maxdiff_nan(&got_out[0], &oracle_out[..d.dim]);
    let pass = ids_ok && wmd <= ROUTE_W_THRESH && md <= MOE_VAL_THRESH && nan == 0;
    println!(
        "device: {dev} | ds4-moe (iii) engineered zones+clamps: dim={} inter={} | route ids {} (w maxdiff={wmd:.3e}) | out maxdiff={md:.3e} nan={nan} | {}",
        d.dim,
        d.inter,
        if ids_ok { "EXACT" } else { "MISMATCH" },
        if pass { "PASS" } else { "FAIL" }
    );
    let summary = format!(
        "(iii) engineered ids {} w={wmd:.3e} md={md:.3e}",
        if ids_ok { "EXACT" } else { "MISMATCH" }
    );
    let fail = if pass {
        String::new()
    } else {
        format!("(iii) ids_ok={ids_ok} w={wmd:.3e} md={md:.3e} nan={nan}")
    };
    Ok((dev, summary, pass, fail))
}

/// ds4-moe — DeepSeek-V4 MoE 스테이지 정합 프로브.
/// (i) 해시 층 L0 실측(실 tid2eid·트렐리스 전문가 K=3·공유 K=5) t=2 —
/// 라우팅 이산 exact가 주 판정. (ii) 라우티드 층 L3 실측 noaux_tc t=2.
/// (iii) 합성 스몰 존(동률 하위 슬롯·바이어스 낙하·클램프 ±10 경계).
/// 하나라도 FAIL이면 Err(→ CLI 비영).
pub fn cuda_ds4_moe_check(dir: Option<&str>) -> Result<String, String> {
    let dir = match dir.map(str::to_string).or_else(|| {
        std::env::var("LLM170_DS4_EXL3")
            .ok()
            .filter(|s| !s.is_empty())
    }) {
        Some(d) => d,
        None => DS4_EXL3_DEFAULT.to_string(),
    };
    let (dims, n_hash, vocab) = ds4_cfg_from_dir(&dir)?;
    // 실측 계약 고정 검증(계기의 계기 — Vision-Exp LLM 타워 2026-10-05).
    if dims.n_routed != 256 || dims.n_active != 6 || dims.inter != 2048 || dims.dim != 4096 {
        return Err(format!(
            "ds4-moe: config 형상 {:?} — 실측 계약(4096·256e·top6·2048) 위반",
            dims
        ));
    }
    if (dims.route_scale - 1.5).abs() > 1e-6 || (dims.swiglu_limit - 10.0).abs() > 1e-6 {
        return Err(format!(
            "ds4-moe: scale/limit {:?} — 실측 계약(1.5·10.0) 위반",
            (dims.route_scale, dims.swiglu_limit)
        ));
    }
    if n_hash < 1 || n_hash as usize + 1 > 42 {
        return Err(format!(
            "ds4-moe: num_hash_layers={n_hash} — L0 해시/L3 라우티드 불가"
        ));
    }
    let ar = StArchive::open(Path::new(&dir))?;
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    let (dev, s, p, f) = run_real_case(
        "(i) hash-L0 real",
        &ar,
        &dims,
        vocab,
        0,
        true,
        2,
        0x05E4_0000_0000_0001,
    )?;
    report.push_str(&s);
    if !p {
        fails.push(f);
    }
    let (_, s, p, f) = run_real_case(
        "(ii) routed-L3 real",
        &ar,
        &dims,
        vocab,
        n_hash,
        false,
        2,
        0x05E4_0000_0000_0002,
    )?;
    report.push_str(" · ");
    report.push_str(&s);
    if !p {
        fails.push(f);
    }
    let fx = EngineeredFx::build(0x05E4_0000_0000_0003)?;
    let (_, s, p, f) = run_engineered_case(&fx)?;
    report.push_str(" · ");
    report.push_str(&s);
    if !p {
        fails.push(f);
    }
    if fails.is_empty() {
        Ok(format!("device: {dev} | ds4-moe {report} | ALL PASS"))
    } else {
        Err(format!("ds4-moe 실패 — {}", fails.join(", ")))
    }
}

/// ds4-moe-neg — 음성대조 3계급(원장 17호: 계기 자체 검증):
/// (a) bias 포함 라우팅 가중치: 오라클이 w = (s+bias)/Σ×1.5(결함식)로
///     계산했을 때의 값과 모듈(정상 s/Σ×1.5)의 라우팅 가중치가 이탈해야
///     한다 — 선택(ids)은 불변인 채 가중치만 이탈(결함 계급 고정).
///     모듈이 이 결함을 갖는 경우와 동일 검출 신호(값 비교 대칭).
/// (b) 해시 표 전치 오독: 모듈에 tid2eid를 [6][vocab] 열 우선으로 오독한
///     표를 먹였을 때(행↔열 전치 — 모듈 입력 데이터 오염) 선택이 완전히
///     달라져 ids exact-match가 깨져야 한다(이산 검출).
/// (c) 클램프 누락: 오라클이 SwiGLU 클램프 없이(limit=∞) 계산했을 때
///     종단 값이 임계 초과 이탈해야 한다(클램프 경계 케이스 (iii)의
///     사전활성이 ±10 초과 → 이탈 보장). 셋 전부 검출 시 NEG-DETECTED.
pub fn cuda_ds4_moe_negative_check(dir: Option<&str>) -> Result<String, String> {
    // (a) bias-in-weights — 스몰 합성(바이어스 0.5·-1.0 존이 결함 이탈 보장).
    let fx = EngineeredFx::build(0x05E4_0000_0000_00AA)?;
    fx.zones_check()?;
    {
        let d = fx.d.clone();
        let mut m = Ds4MoeCuda::new(d.clone())?;
        let dev = m.device_name().to_string();
        m.set_gate_f32(&fx.gw, Some(&fx.bias), None)?;
        m.set_shared_f32(&fx.shared.w1, &fx.shared.w2, &fx.shared.w3)?;
        let mut need: Vec<usize> = fx.experts.keys().copied().collect();
        need.sort_unstable();
        for e in need {
            let w = &fx.experts[&e];
            m.add_expert_f32(e, &w.w1, &w.w2, &w.w3)?;
        }
        let got = m.moe_route_routed(&fx.xs)?;
        // 결함 오라클 — w = (s+bias)/Σ×1.5(route_routed 결함 변형).
        let scores = gate_scores_mirror(&fx.xs[0], &fx.gw, d.n_routed, d.dim);
        let biased: Vec<f32> = scores.iter().zip(&fx.bias).map(|(&s, &b)| s + b).collect();
        let sel: Vec<usize> = topk_stable_mirror(&biased, d.n_active)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        let raw: Vec<f32> = sel.iter().map(|&i| biased[i]).collect(); // bias 포함(결함)
        let sum: f32 = raw.iter().sum();
        let mut defect: Vec<(usize, f32)> = sel
            .into_iter()
            .zip(raw)
            .map(|(i, w)| (i, w / sum * d.route_scale))
            .collect();
        defect.sort_by_key(|&(i, _)| i);
        let ids_same = got.len() == 1
            && got[0].iter().map(|&(e, _)| e).collect::<Vec<_>>()
                == defect.iter().map(|&(e, _)| e as u32).collect::<Vec<_>>();
        let mut md_a = 0.0f32;
        for (g, w) in got[0].iter().zip(defect.iter()) {
            md_a = md_a.max((g.1 - w.1).abs());
        }
        let det_a = ids_same && md_a > ROUTE_W_THRESH;
        println!(
            "device: {dev} | ds4-moe-neg (a) bias-in-weights oracle defect: ids_same={ids_same} w_maxdiff={md_a:.3e} | {}",
            if det_a {
                "FAIL(expected)"
            } else {
                "NOT-DETECTED"
            }
        );
        if !det_a {
            return Err(format!(
                "NEG-MISSED (a) bias-in-weights ids_same={ids_same} maxdiff={md_a:.3e} — 검증계기 결함"
            ));
        }
    }
    // (b) 해시 표 전치 — 실측 L0 표(모듈 입력 데이터 오염).
    {
        let dir = match dir.map(str::to_string).or_else(|| {
            std::env::var("LLM170_DS4_EXL3")
                .ok()
                .filter(|s| !s.is_empty())
        }) {
            Some(d) => d,
            None => DS4_EXL3_DEFAULT.to_string(),
        };
        let (dims, _n_hash, vocab) = ds4_cfg_from_dir(&dir)?;
        let ar = StArchive::open(Path::new(&dir))?;
        let gw = load_gate(&ar, 0, &dims, true)?;
        let t2e = gw.tid2eid.as_ref().unwrap();
        let tids = [7u32];
        let xs = gen_xs(dims.dim, 1, 0x05E4_0000_0000_00BB);
        let scores = gate_scores_mirror(&xs[0], &gw.weight, dims.n_routed, dims.dim);
        let row = &t2e[tids[0] as usize * dims.n_active..][..dims.n_active];
        let oracle_sel = route_hash_mirror(&scores, row, dims.route_scale);
        // 전치 오독 표 — [vocab,6]을 [6,vocab]으로 읽음(행↔열 전치).
        let mut t2e_t = vec![0i64; t2e.len()];
        for tid in 0..vocab {
            for j in 0..dims.n_active {
                t2e_t[j * vocab + tid] = t2e[tid * dims.n_active + j];
            }
        }
        let mut m = Ds4MoeCuda::new(dims.clone())?;
        let dev = m.device_name().to_string();
        m.set_gate_f32(&gw.weight, None, Some(&t2e_t))?;
        let shared = load_expert(&ar, "layers.0.ffn.shared_experts")?;
        m.set_shared_f32(&shared.w1, &shared.w2, &shared.w3)?;
        // 청정·오염 선택 합집합 상주(값 검출 보장 — 모듈 런 가능해야 함).
        let corrupt_row = &t2e_t[tids[0] as usize * dims.n_active..][..dims.n_active];
        let corrupt_sel = route_hash_mirror(&scores, corrupt_row, dims.route_scale);
        let mut need: Vec<u32> = Vec::new();
        for r in [&oracle_sel, &corrupt_sel] {
            for &(e, _) in r.iter() {
                if !need.contains(&(e as u32)) {
                    need.push(e as u32);
                }
            }
        }
        need.sort_unstable();
        stage_real_experts(&mut m, &ar, 0, &need)?;
        let got_sel = m.moe_route_hash(&xs, &tids)?;
        let ids_differ = got_sel.len() != 1
            || got_sel[0].iter().map(|&(e, _)| e).collect::<Vec<_>>()
                != oracle_sel
                    .iter()
                    .map(|&(e, _)| e as u32)
                    .collect::<Vec<_>>();
        println!(
            "device: {dev} | ds4-moe-neg (b) transposed tid2eid (module-side corruption): ids_differ={ids_differ} | {}",
            if ids_differ {
                "FAIL(expected)"
            } else {
                "NOT-DETECTED"
            }
        );
        if !ids_differ {
            return Err(format!(
                "NEG-MISSED (b) transposed tid2eid — 선택 불변(전치 오독 미탐지): 검증계기 결함"
            ));
        }
    }
    // (c) 클램프 누락 — 오라클 결함 주입(limit=∞, (iii)과 동일 픽스처).
    {
        let d = fx.d.clone();
        let gw = GateMirror {
            weight: fx.gw.clone(),
            bias: Some(fx.bias.clone()),
            tid2eid: None,
        };
        let flat: Vec<f32> = fx.xs.concat();
        let oracle_unclamped = moe_forward_mirror(
            &flat,
            &[0],
            &gw,
            &fx.shared,
            &fx.experts,
            &d,
            false,
            Some(f32::INFINITY),
        );
        let mut m = Ds4MoeCuda::new(d.clone())?;
        let dev = m.device_name().to_string();
        m.set_gate_f32(&fx.gw, Some(&fx.bias), None)?;
        m.set_shared_f32(&fx.shared.w1, &fx.shared.w2, &fx.shared.w3)?;
        let mut need: Vec<usize> = fx.experts.keys().copied().collect();
        need.sort_unstable();
        for e in need {
            let w = &fx.experts[&e];
            m.add_expert_f32(e, &w.w1, &w.w2, &w.w3)?;
        }
        let got = m.moe_ffn_routed(&fx.xs)?;
        let (md_c, nan) = maxdiff_nan(&got[0], &oracle_unclamped[..d.dim]);
        let det_c = md_c > MOE_VAL_THRESH && nan == 0;
        println!(
            "device: {dev} | ds4-moe-neg (c) clamp-missing oracle defect: maxdiff={md_c:.3e} nan={nan} | {}",
            if det_c {
                "FAIL(expected)"
            } else {
                "NOT-DETECTED"
            }
        );
        if !det_c {
            return Err(format!(
                "NEG-MISSED (c) clamp-missing maxdiff={md_c:.3e} — 검증계기 결함(클램프 이탈 미탐지)"
            ));
        }
        return Err(format!(
            "NEG-DETECTED (a) bias-in-weights w_maxdiff > {ROUTE_W_THRESH} (b) transposed-tid2eid ids_differ (c) clamp-missing maxdiff={md_c:.3e} > {MOE_VAL_THRESH} — 검증계기 정상(가중치식·해시 표·SwiGLU 클램프 결함 감지)"
        ));
    }
}
