//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: 케이스별 입력을 독립 생성하고
//! **S0≠0·링≠0 초기 상태를 의무 주입**(합성 S0=0은 상태 버그를 가린다
//! — plans/124 §3.3). ② 형상은 실측 GGUF 메타+텐서 ne에서 자동 열거
//! (48/128/16/conv4 계약 검증 — 추정 금지, 결함 1호 정신). ③ 판정은
//! **종단 값 maxdiff + 단계 국소화**(중간 단계는 진단)·argmax 금지.
//! ④ t=1은 코어 디스패치 미리(stage gdn.rs L135 → gdn_ar_batch)로
//! AR 오라클 판정 — 디코드 경로가 상태 버그 역사 계급(§3.3).
//!
//! ═══ FNF 오라클 계약 — core 미러(커널 미러가 아님) ═══
//! G5(gdn_cuda_probe.rs)의 오라클이 **커널**(f16 저장·CS=32)을 미러한
//! 것과 달리, FNF 오라클은 값 maxdiff 판정의 유일 기준인 **core 원천을
//! 그대로 재생**한다(plans/124 §6 계층: 커널 → core → 수학):
//! · conv 링 — qwen4exp/stages/gdn.rs L69-85(conv_w[c·4+3]·x + Σ_{j<3}
//!   w[c·4+j]·s_j, silu, 링 시프트) · ops.rs silu L128.
//! · β/e^g — stages/gdn.rs L57-58: sigmoid(b)·softplus(a+dt_bias)·ssm_a
//!   (ops.rs sigmoid L133·softplus L138·exp_cr L52·ln_cr L91·log1p_cr
//!   L123 — 아래 or_* 트윈은 이 연산열을 상수까지 그대로 베낀다).
//! · q/k l2 — stages/gdn.rs L86-97 · ops.rs l2_norm L39-44(**eps floor**,
//!   순차 f32 누산).
//! · scan(t>1) — core/gdn.rs gdn_chunk_seq L22·gdn_chunk_head L66-197
//!   (CS=64·f32·qp 선스케일·전진 소거·o 소거 후 산출·상태 P6 갱신,
//!   f32 .exp()는 호스트 libm — 코어와 동일 실행 환경 계급).
//! · scan(t=1) — core/gdn.rs gdn_ar_batch L375·gdn_ar_head L326-373
//!   (g_exp=exp_cr — 청크의 libm exp와 다른 트랜센던트, 코어 디스패치
//!   그대로).
//! · 게이트 — gdn_norm.rs gdn_norm_gated L26-50(GdnGate::Sigmoid)·
//!   ops.rs sq_sum L11-31(32세그먼트 f64)·rms_norm L33-37.
//!
//! [정합 원장 — sm_89 실측 2026-10-05]
//! (i) T=32 S0≠0 ord=0(il=0): 종단 7.153e-7 · 단계 worst 7.153e-7 ·
//! 비트동일 7/10(conv q/k/v·링·l2 q2/k2·prep bg) — scan o·상태·gated는
//! expf(장치) vs libm .exp() 잔차 계급.
//! (ii) T=32 S0≠0 ord=35(il=46, 전층 스트라이드 경계): 종단 8.345e-7 ·
//! 비트동일 7/10. (iii) T=1 S0≠0(AR 디스패치): 종단 1.788e-6.
//! 전 case nan=0 · rel>5% 0/N — 임계 2e-4 대비 ~250배 여유.
//! [1차 재사용 시도 실측 — REUSE 불가 근거(판정표 전환)] G5 exl3_gdn_scan
//! 그대로 사용 시: ord=0에서 gdn_expf 정의역(gcs<−900) NaN(o 9088·상태
//! 65536=4헤드) · ord=35에서 f16 오차의 게이트 rms 증폭으로 종단
//! 2.579e-3 — 임계 초과. conv는 비트동일로 재사용 확정.
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 개발기
//! (sm_89) 타이밍은 판단 근거가 아니다(§0 계약).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::exl3_cuda_probe::{Rng, maxdiff_nan};
use crate::rawcuda::fn_gdn_cuda::FnGdnCuda;
use crate::rawcuda::fn_support::{FnDims, FnGguf};
use std::path::Path;

/// GDN 스테이지 값 maxdiff 임계(plans/124 §1 종단 2e-4 — G5 GDN_THRESH
/// 동일 규율).
const FN_GDN_THRESH: f32 = 2e-4;

// ── core ops.rs 트랜센던트/누산 트윈(연산열·상수 그대로 — 비트동일) ──

/// ops.rs exp_cr L52-88 직이식(f64 fma 호너 — round_ties_even 포함).
fn or_exp_cr(x: f32) -> f32 {
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

/// ops.rs ln_cr L91-121 직이식(atanh 급수 fma 호너 — 정규수 v ≥ 1 전용).
fn or_ln_cr(v: f64) -> f64 {
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64;
    let k = e - 1023;
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

fn or_silu(x: f32) -> f32 {
    // ops.rs L128-130.
    x / (1.0 + or_exp_cr(-x))
}

fn or_sigmoid(x: f32) -> f32 {
    // ops.rs L133-135.
    1.0 / (1.0 + or_exp_cr(-x))
}

fn or_softplus(x: f32) -> f32 {
    // ops.rs L138-141(log1p_cr L123 인라인).
    if x > 20.0 {
        x
    } else {
        or_ln_cr(or_exp_cr(x) as f64 + 1.0) as f32
    }
}

/// ops.rs sq_sum L11-31 직이식(32세그먼트 f32 순차 → f64 순차 결합).
fn or_sq_sum(x: &[f32]) -> f64 {
    const SEG: usize = 32;
    let n = x.len();
    let chunk = n.div_ceil(SEG);
    let mut sum = 0.0f64;
    for u in 0..SEG {
        let lo = u * chunk;
        if lo >= n {
            break;
        }
        let hi = (lo + chunk).min(n);
        let mut part = 0.0f32;
        for &v in &x[lo..hi] {
            part += v * v;
        }
        sum += part as f64;
    }
    sum
}

/// ops.rs rms_norm L33-37 직이식(f64 scale · (v·scale)·g 결합순서).
fn or_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = or_sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// ops.rs l2_norm L39-44 직이식(순차 f32 · eps floor · v·scale).
fn or_l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / sum.sqrt().max(eps);
    x.iter().map(|&v| v * scale).collect()
}

// ── 스테이지 오라클(원천: stages/gdn.rs + core/gdn.rs + gdn_norm.rs) ──

/// 오라클 단계 산출(단계별 판정용).
pub(crate) struct FnGdnRefStages {
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    pub ring_post: Vec<f32>,
    pub q2: Vec<f32>,
    pub k2: Vec<f32>,
    pub bg: Vec<f32>,
    pub o: Vec<f32>,
    pub st_post: Vec<f32>,
    pub gated: Vec<f32>,
}

/// conv + β/e^g + q/k l2(청크/AR 공통 전단) — 채널별 순차 회전.
/// 원천: stages/gdn.rs L57-58·L69-85·L86-97.
#[allow(clippy::too_many_arguments)]
fn ref_pre_stages(
    dims: &FnDims,
    eps: f32,
    cw_l: &[f32],
    dtb_l: &[f32],
    ssa_l: &[f32],
    qkv: &[f32],
    b: &[f32],
    a: &[f32],
    ring0: &[f32],
    t_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (dr, ds, ng, ck) = (dims.dt_rank, dims.d_state, dims.n_group, dims.conv_k);
    let k_len = ng * ds;
    let v_len = dr * ds;
    // β/e^g — stages/gdn.rs L57-58(beta_all/g_all [t][dt_rank]).
    let mut beta_all = vec![0f32; t_len * dr];
    let mut g_all = vec![0f32; t_len * dr];
    for t in 0..t_len {
        for h in 0..dr {
            beta_all[t * dr + h] = or_sigmoid(b[t * dr + h]);
            g_all[t * dr + h] = or_softplus(a[t * dr + h] + dtb_l[h]) * ssa_l[h];
        }
    }
    // conv — L69-85: sum = w[c·ck+ck−1]·x + Σ_{j<ck−1} w[c·ck+j]·s_j,
    // silu, 링 시프트(s_j←s_{j+1}, s_{ck−2}←x). 채널 독립(코어 t외향/c내향
    // 루프와 채널별 재귀 동일 — 순서 무관).
    let mut conv_q = vec![0f32; t_len * k_len];
    let mut conv_k = vec![0f32; t_len * k_len];
    let mut conv_v = vec![0f32; t_len * v_len];
    let mut ring = ring0.to_vec();
    let cch = ng * ds * 2 + dr * ds;
    for c in 0..cch {
        let (mut s0, mut s1, mut s2) = (ring[0 * cch + c], ring[1 * cch + c], ring[2 * cch + c]);
        for t in 0..t_len {
            let x = qkv[t * cch + c];
            let mut sum = cw_l[c * ck + (ck - 1)] * x;
            sum += cw_l[c * ck] * s0;
            sum += cw_l[c * ck + 1] * s1;
            sum += cw_l[c * ck + 2] * s2;
            let out_c = or_silu(sum);
            if c < k_len {
                conv_q[t * k_len + c] = out_c;
            } else if c < 2 * k_len {
                conv_k[t * k_len + (c - k_len)] = out_c;
            } else {
                conv_v[t * v_len + (c - 2 * k_len)] = out_c;
            }
            s0 = s1;
            s1 = s2;
            s2 = x;
        }
        ring[0 * cch + c] = s0;
        ring[1 * cch + c] = s1;
        ring[2 * cch + c] = s2;
    }
    // q/k l2 — L86-97(n_group 헤드 · eps floor · 순열 없음).
    let mut q2 = vec![0f32; t_len * k_len];
    let mut k2 = vec![0f32; t_len * k_len];
    for t in 0..t_len {
        for h in 0..ng {
            let b0 = t * k_len + h * ds;
            let head: Vec<f32> = conv_q[b0..b0 + ds].to_vec();
            q2[b0..b0 + ds].copy_from_slice(&or_l2_norm(&head, eps));
            let headk: Vec<f32> = conv_k[b0..b0 + ds].to_vec();
            k2[b0..b0 + ds].copy_from_slice(&or_l2_norm(&headk, eps));
        }
    }
    (conv_q, conv_k, conv_v, ring, q2, k2)
}

/// β/e^g → bg 커널 레이아웃([t][2·dt_rank]: beta|g 자연 순서).
fn ref_bg(beta_all: &[f32], g_all: &[f32], t_len: usize, dr: usize) -> Vec<f32> {
    let mut bg = vec![0f32; t_len * 2 * dr];
    for t in 0..t_len {
        for h in 0..dr {
            bg[t * 2 * dr + h] = beta_all[t * dr + h];
            bg[t * 2 * dr + dr + h] = g_all[t * dr + h];
        }
    }
    bg
}

/// scan 청크 경로(t>1) — core/gdn.rs gdn_chunk_seq L22·gdn_chunk_head
/// L66-197 직미러(CS=64·f32·qp 선스케일·전진 소거·상태 P6). q2/k2는 l2
/// 완료 입력(원천과 동일 — gdn_chunk_seq 호출점 stages/gdn.rs L147).
#[allow(clippy::too_many_arguments)]
fn ref_scan_chunk(
    q2: &[f32],
    k2: &[f32],
    conv_v: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    s0: &[f32],
    t_len: usize,
    dims: &FnDims,
) -> (Vec<f32>, Vec<f32>) {
    let cs = 64usize; // core/gdn.rs L18 CS
    let (dr, ds, ng) = (dims.dt_rank, dims.d_state, dims.n_group);
    let (h_k, h_v, d) = (ng, dr, ds);
    let k_stride = h_k * d;
    let v_stride = h_v * d;
    let scale = 1.0f32 / (d as f32).sqrt();
    let n_chunks = t_len.div_ceil(cs);
    let mut o = vec![0f32; t_len * v_stride];
    let mut st_all = s0.to_vec();
    for h in 0..h_v {
        let kh = h % h_k;
        let st_h = h * d * d;
        let mut st: Vec<f32> = st_all[st_h..st_h + d * d].to_vec();
        for c in 0..n_chunks {
            let t0 = c * cs;
            let n = (t0 + cs).min(t_len) - t0;
            let mut qp = vec![0f32; cs * d];
            let mut kp = vec![0f32; cs * d];
            let mut vp = vec![0f32; cs * d];
            let mut bp = vec![0f32; cs];
            let mut gp = vec![0f32; cs];
            for t in 0..n {
                let src = t0 + t;
                qp[t * d..t * d + d]
                    .copy_from_slice(&q2[src * k_stride + kh * d..src * k_stride + kh * d + d]);
                kp[t * d..t * d + d]
                    .copy_from_slice(&k2[src * k_stride + kh * d..src * k_stride + kh * d + d]);
                vp[t * d..t * d + d]
                    .copy_from_slice(&conv_v[src * v_stride + h * d..src * v_stride + h * d + d]);
                bp[t] = beta_all[src * h_v + h];
                gp[t] = g_all[src * h_v + h];
            }
            for x in qp.iter_mut() {
                *x *= scale;
            }
            let mut gcs = vec![0f32; cs];
            let mut acc = 0f32;
            for t in 0..cs {
                acc += gp[t];
                gcs[t] = acc;
            }
            let g_last = gcs[cs - 1];
            let mut d_out = vec![0f32; cs * d];
            let mut oi = vec![0f32; d];
            for i in 0..n {
                let beta_i = bp[i];
                for dv in 0..d {
                    oi[dv] = beta_i * vp[i * d + dv];
                }
                if beta_i != 0.0 {
                    let w0 = beta_i * gcs[i].exp();
                    for s2 in 0..d {
                        let ks = kp[i * d + s2];
                        if ks == 0.0 {
                            continue;
                        }
                        let w = w0 * ks;
                        for dv in 0..d {
                            oi[dv] -= w * st[s2 * d + dv];
                        }
                    }
                }
                let dbase = i * d;
                d_out[dbase..dbase + d].copy_from_slice(&oi[..d]);
                for j in 0..i {
                    let mut dot = 0f32;
                    for s2 in 0..d {
                        dot += kp[i * d + s2] * kp[j * d + s2];
                    }
                    let aij = dot * beta_i * (gcs[i] - gcs[j]).exp();
                    if aij == 0.0 {
                        continue;
                    }
                    for dv in 0..d {
                        d_out[dbase + dv] -= aij * d_out[j * d + dv];
                    }
                }
                for dv in 0..d {
                    oi[dv] = 0.0;
                }
                let qi_exp = gcs[i].exp();
                for s2 in 0..d {
                    let qv = qp[i * d + s2];
                    if qv == 0.0 {
                        continue;
                    }
                    let w = qi_exp * qv;
                    for dv in 0..d {
                        oi[dv] += w * st[s2 * d + dv];
                    }
                }
                for j in 0..=i {
                    let mut dot = 0f32;
                    for s2 in 0..d {
                        dot += qp[i * d + s2] * kp[j * d + s2];
                    }
                    let kqij = dot * (gcs[i] - gcs[j]).exp();
                    if kqij == 0.0 {
                        continue;
                    }
                    for dv in 0..d {
                        oi[dv] += kqij * d_out[j * d + dv];
                    }
                }
                o[(t0 + i) * v_stride + h * d..(t0 + i) * v_stride + h * d + d]
                    .copy_from_slice(&oi);
            }
            let gl_exp = g_last.exp();
            for xv in st.iter_mut() {
                *xv *= gl_exp;
            }
            for j in 0..n {
                let w = (g_last - gcs[j]).exp();
                for s2 in 0..d {
                    let kv = kp[j * d + s2] * w;
                    for dv in 0..d {
                        st[s2 * d + dv] += kv * d_out[j * d + dv];
                    }
                }
            }
        }
        // 헤드 상태 post-T 기록(st_post 판정용).
        st_all[st_h..st_h + d * d].copy_from_slice(&st);
    }
    (o, st_all)
}

/// scan AR 경로(t=1) — core/gdn.rs gdn_ar_batch L375·gdn_ar_head L326-373
/// 직미러(g_exp=exp_cr — stages/gdn.rs L135 코어 디스패치).
#[allow(clippy::too_many_arguments)]
fn ref_scan_ar(
    q2: &[f32],
    k2: &[f32],
    conv_v: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    s0: &[f32],
    dims: &FnDims,
) -> (Vec<f32>, Vec<f32>) {
    let (dr, ds, ng) = (dims.dt_rank, dims.d_state, dims.n_group);
    let (h_k, h_v, d) = (ng, dr, ds);
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut o = vec![0f32; h_v * d];
    let mut st_post = s0.to_vec();
    for h in 0..h_v {
        let kh = h % h_k;
        let st_h = h * d * d;
        let mut st: Vec<f32> = s0[st_h..st_h + d * d].to_vec();
        let qs = &q2[kh * d..kh * d + d];
        let ks = &k2[kh * d..kh * d + d];
        let vs = &conv_v[h * d..h * d + d];
        let beta_h = beta_all[h];
        let g_exp = or_exp_cr(g_all[h]);
        let mut sk = vec![0f32; d];
        for kdim in 0..d {
            let kk = ks[kdim];
            for dv in 0..d {
                st[kdim * d + dv] *= g_exp;
                sk[dv] += st[kdim * d + dv] * kk;
            }
        }
        let mut delta = vec![0f32; d];
        for dv in 0..d {
            delta[dv] = (vs[dv] - sk[dv]) * beta_h;
        }
        for kdim in 0..d {
            let kd = ks[kdim];
            for dv in 0..d {
                st[kdim * d + dv] += kd * delta[dv];
            }
        }
        let mut ov = vec![0f32; d];
        for kdim in 0..d {
            let qq = qs[kdim];
            for dv in 0..d {
                ov[dv] += st[kdim * d + dv] * qq * scale;
            }
        }
        o[h * d..h * d + d].copy_from_slice(&ov);
        st_post[st_h..st_h + d * d].copy_from_slice(&st);
    }
    (o, st_post)
}

/// 게이트 — gdn_norm.rs gdn_norm_gated L26-50(GdnGate::Sigmoid) 직미러.
fn ref_gate(o: &[f32], z: &[f32], nw_l: &[f32], eps: f32, t_len: usize, dims: &FnDims) -> Vec<f32> {
    let (dr, ds) = (dims.dt_rank, dims.d_state);
    let v_len = dr * ds;
    let mut gated = vec![0f32; t_len * v_len];
    for t in 0..t_len {
        for h in 0..dr {
            let b0 = t * v_len + h * ds;
            let head: Vec<f32> = o[b0..b0 + ds].to_vec();
            let n = or_rms_norm(&head, nw_l, eps);
            let zb = h * ds;
            for i in 0..ds {
                gated[t * v_len + zb + i] = n[i] * or_sigmoid(z[t * v_len + zb + i]);
            }
        }
    }
    gated
}

/// 스테이지 전체 오라클 조립(t>1 청크 · t=1 AR — 코어 디스패치 미러).
#[allow(clippy::too_many_arguments)]
fn fn_gdn_reference(
    dims: &FnDims,
    eps: f32,
    cw_l: &[f32],
    dtb_l: &[f32],
    ssa_l: &[f32],
    nw_l: &[f32],
    qkv: &[f32],
    z: &[f32],
    b: &[f32],
    a: &[f32],
    ring0: &[f32],
    s0: &[f32],
    t_len: usize,
) -> FnGdnRefStages {
    let dr = dims.dt_rank;
    let (conv_q, conv_k, conv_v, ring_post, q2, k2) =
        ref_pre_stages(dims, eps, cw_l, dtb_l, ssa_l, qkv, b, a, ring0, t_len);
    let mut beta_all = vec![0f32; t_len * dr];
    let mut g_all = vec![0f32; t_len * dr];
    for t in 0..t_len {
        for h in 0..dr {
            beta_all[t * dr + h] = or_sigmoid(b[t * dr + h]);
            g_all[t * dr + h] = or_softplus(a[t * dr + h] + dtb_l[h]) * ssa_l[h];
        }
    }
    let bg = ref_bg(&beta_all, &g_all, t_len, dr);
    let (o, st_post) = if t_len == 1 {
        ref_scan_ar(&q2, &k2, &conv_v, &beta_all, &g_all, s0, dims)
    } else {
        ref_scan_chunk(&q2, &k2, &conv_v, &beta_all, &g_all, s0, t_len, dims)
    };
    let gated = ref_gate(&o, z, nw_l, eps, t_len, dims);
    FnGdnRefStages {
        conv_q,
        conv_k,
        conv_v,
        ring_post,
        q2,
        k2,
        bg,
        o,
        st_post,
        gated,
    }
}

// ── 픽스처(실가중 상수 + 결정론 시드 입력 — G5 프로브 방법론 계승) ──

/// FNF GDN 픽스처 — 상수는 실측 GGUF(ssm_conv1d·ssm_dt.bias·ssm_a·
/// ssm_norm — 변환 없는 F32 원값), 전 36 GDN층 등록(커널 layer 인덱싱
/// 증명). 입력은 케이스별 생성(gen_inputs).
struct FnGdnFixture {
    dims: FnDims,
    eps: f32,
    gdn_ils: Vec<usize>,
    cw_all: Vec<f32>,
    dtb_all: Vec<f32>,
    ssa_all: Vec<f32>,
    nw_all: Vec<f32>,
}

impl FnGdnFixture {
    fn load(g: &FnGguf, dims: &FnDims) -> Result<Self, String> {
        let gdn_ils: Vec<usize> = (0..dims.n_layer)
            .filter(|&il| dims.compress[il] == 0)
            .collect();
        if gdn_ils.is_empty() {
            return Err("GDN층 0개 — compress_ratios 이상".into());
        }
        let (dr, cch) = (dims.dt_rank, dims.gdn_conv_ch());
        let f32s = |raw: &[u8]| -> Vec<f32> {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let mut cw_all = Vec::with_capacity(gdn_ils.len() * cch * 4);
        let mut dtb_all = Vec::with_capacity(gdn_ils.len() * dr);
        let mut ssa_all = Vec::with_capacity(gdn_ils.len() * dr);
        let mut nw_all = Vec::with_capacity(gdn_ils.len() * 128);
        for &il in &gdn_ils {
            let tc = g
                .tensor(&format!("blk.{il}.ssm_conv1d.weight"))
                .ok_or("conv1d 없음")?;
            let td = g
                .tensor(&format!("blk.{il}.ssm_dt.bias"))
                .ok_or("dt.bias 없음")?;
            let ta = g.tensor(&format!("blk.{il}.ssm_a")).ok_or("ssm_a 없음")?;
            let tn = g
                .tensor(&format!("blk.{il}.ssm_norm.weight"))
                .ok_or("ssm_norm 없음")?;
            // 형상 명시 검증(추정 금지 — 결함 1호 정신): conv [conv_k,cch]
            // F32 플랫 [c][4](=core conv_w[c·conv_k+j] 레이아웃), 나머지
            // F32 1-D.
            if tc.ty != 0 || tc.dims != vec![dims.conv_k as u64, cch as u64] {
                return Err(format!(
                    "blk.{il}.ssm_conv1d: ty{} ne={:?} — F32 [{},{}] 계약 위반",
                    tc.ty, tc.dims, dims.conv_k, cch
                ));
            }
            if td.ty != 0 || td.dims != vec![dr as u64] {
                return Err(format!(
                    "blk.{il}.ssm_dt.bias: ty{} ne={:?}",
                    td.ty, td.dims
                ));
            }
            if ta.ty != 0 || ta.dims != vec![dr as u64] {
                return Err(format!("blk.{il}.ssm_a: ty{} ne={:?}", ta.ty, ta.dims));
            }
            if tn.ty != 0 || tn.dims != vec![128u64] {
                return Err(format!(
                    "blk.{il}.ssm_norm: ty{} ne={:?} — [128] 헤드공유 계약",
                    tn.ty, tn.dims
                ));
            }
            cw_all.extend_from_slice(&f32s(&g.read_rows(
                &format!("blk.{il}.ssm_conv1d.weight"),
                0,
                cch as u64,
            )?));
            dtb_all.extend_from_slice(&f32s(&g.read_rows(
                &format!("blk.{il}.ssm_dt.bias"),
                0,
                1,
            )?));
            ssa_all.extend_from_slice(&f32s(&g.read_rows(&format!("blk.{il}.ssm_a"), 0, 1)?));
            nw_all.extend_from_slice(&f32s(&g.read_rows(
                &format!("blk.{il}.ssm_norm.weight"),
                0,
                1,
            )?));
        }
        Ok(FnGdnFixture {
            eps: dims.eps,
            dims: dims.clone(),
            gdn_ils,
            cw_all,
            dtb_all,
            ssa_all,
            nw_all,
        })
    }

    /// 대상 층(GDN 서수) 상수 슬라이스.
    fn layer_consts(&self, ord: usize) -> (&[f32], &[f32], &[f32], &[f32]) {
        let (cch, dr) = (self.dims.gdn_conv_ch(), self.dims.dt_rank);
        (
            &self.cw_all[ord * cch * 4..(ord + 1) * cch * 4],
            &self.dtb_all[ord * dr..(ord + 1) * dr],
            &self.ssa_all[ord * dr..(ord + 1) * dr],
            &self.nw_all[ord * 128..(ord + 1) * 128],
        )
    }
}

/// 케이스 입력 — 결정론 시드(G5 계급 스케일: qkv ±0.5·z ±0.4·b/a ±0.5·
/// 링 ±0.2·S0 ±0.1 — 실가중 conv/A 계급). **S0≠0·링≠0 의무**(§3.3).
struct FnGdnInputs {
    qkv: Vec<f32>,
    z: Vec<f32>,
    b: Vec<f32>,
    a: Vec<f32>,
    ring0: Vec<f32>,
    s0: Vec<f32>,
}

fn gen_inputs(dims: &FnDims, t_len: usize, seed: u64) -> FnGdnInputs {
    let mut rng = Rng::new(seed);
    let unif = |rng: &mut Rng, amp: f64| ((rng.next_f64() * 2.0 - 1.0) * amp) as f32;
    let (cch, vl, dr) = (dims.gdn_conv_ch(), dims.gdn_v_len(), dims.dt_rank);
    FnGdnInputs {
        qkv: (0..t_len * cch).map(|_| unif(&mut rng, 0.5)).collect(),
        z: (0..t_len * vl).map(|_| unif(&mut rng, 0.4)).collect(),
        b: (0..t_len * dr).map(|_| unif(&mut rng, 0.5)).collect(),
        a: (0..t_len * dr).map(|_| unif(&mut rng, 0.5)).collect(),
        ring0: (0..3 * cch).map(|_| unif(&mut rng, 0.2)).collect(),
        s0: (0..dr * 128 * 128).map(|_| unif(&mut rng, 0.1)).collect(),
    }
}

/// rel 잠행 집계(G5 gdn_rel_bad 동일 판정 — d>1e-3 && rel>5%).
fn fn_gdn_rel_bad(got: &[f32], want: &[f32]) -> usize {
    let mut bad = 0usize;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        if d > 1e-3 && d / w.abs().max(1e-3) > 0.05 {
            bad += 1;
        }
    }
    bad
}

/// fn-cuda-gdn — FNF GDN 스테이지 값 maxdiff 판정(종단·단계 임계 2e-4 ·
/// rel 0/N · 이식분 비트동일 보고). (i) T=32 S0≠0 lay=il0(서수 0 — 실측
/// 가중) · (ii) T=32 S0≠0 서수 35(il=46 — 전층 스트라이드 경계) ·
/// (iii) T=1 S0≠0(AR 디코드 경로 — 코어 디스패치 미러). 하나라도 FAIL
/// 이면 Err(→ CLI 비영).
pub fn cuda_fn_gdn_check(gguf_main: &str) -> Result<String, String> {
    let g = FnGguf::open(Path::new(gguf_main))?;
    let dims = FnDims::from_gguf(&g)?;
    if (dims.dt_rank, dims.d_state, dims.n_group, dims.conv_k) != (48, 128, 16, 4) {
        return Err(format!(
            "GDN 형상 ({},{},{},{}) != 계약 (48,128,16,4)",
            dims.dt_rank, dims.d_state, dims.n_group, dims.conv_k
        ));
    }
    let fx = FnGdnFixture::load(&g, &dims)?;
    let eps = fx.eps;
    let mut m = FnGdnCuda::open(&dims, eps, &fx.cw_all, &fx.dtb_all, &fx.ssa_all, &fx.nw_all)?;
    let dev = m.cc.device_name.clone();
    let n_gdn = m.n_gdn();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (tag, ord, t_len, seed, AR여부, 단계국소화 여부)
    let cases: [(&str, usize, usize, u64, bool, bool); 3] = [
        ("i", 0, 32, 0x170F_0D00_0000_0001u64, false, true),
        ("ii", n_gdn - 1, 32, 0x170F_0D00_0000_0002u64, false, true),
        ("iii", n_gdn - 1, 1, 0x170F_0D00_0000_0003u64, false, false),
    ];
    for (tag, ord, t_len, seed, _ar, localize) in cases {
        let inp = gen_inputs(&dims, t_len, seed);
        let (cw_l, dtb_l, ssa_l, nw_l) = fx.layer_consts(ord);
        let want = fn_gdn_reference(
            &dims, eps, cw_l, dtb_l, ssa_l, nw_l, &inp.qkv, &inp.z, &inp.b, &inp.a, &inp.ring0,
            &inp.s0, t_len,
        );
        let got = m.gdn_stage_host(
            ord,
            t_len,
            &inp.qkv,
            &inp.z,
            &inp.b,
            &inp.a,
            Some(&inp.s0),
            Some(&inp.ring0),
        )?;
        let il = fx.gdn_ils[ord];
        let (md, nan) = maxdiff_nan(&got, &want.gated);
        let rel = fn_gdn_rel_bad(&got, &want.gated);
        let pass = md <= FN_GDN_THRESH && nan == 0 && rel == 0;
        println!(
            "device: {dev} | fn-cuda-gdn ({tag}) ord={ord}(il={il}) dt_rank=48 d_state=128 n_group=16 conv_k=4 T={t_len} S0!=0: end-to-end maxdiff={md:.3e} nan={nan} rel>5%={rel}/{} | {}",
            t_len * dims.gdn_v_len(),
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!("({tag}) e2e={md:.3e}",));
        if !pass {
            fails.push(format!(
                "({tag}) end-to-end maxdiff={md:.3e} nan={nan} rel={rel}"
            ));
        }

        if localize {
            let mids = m.gdn_mids_host(ord, t_len)?;
            let stages: [(&str, &[f32], &[f32]); 10] = [
                ("conv q", &mids.conv_q, &want.conv_q),
                ("conv k", &mids.conv_k, &want.conv_k),
                ("conv v", &mids.conv_v, &want.conv_v),
                ("ring post-T", &mids.ring_post, &want.ring_post),
                ("l2 q2", &mids.q2, &want.q2),
                ("l2 k2", &mids.k2, &want.k2),
                ("prep bg(beta|g)", &mids.bg, &want.bg),
                ("scan o(elim)", &mids.o, &want.o),
                ("state post-T", &mids.st_post, &want.st_post),
                ("gate gated", &mids.gated, &want.gated),
            ];
            let mut worst = 0f32;
            let mut bitexact = 0usize;
            for (name, gv, wv) in stages {
                let (mv, nv) = maxdiff_nan(gv, wv);
                worst = worst.max(mv);
                if mv == 0.0 && nv == 0 {
                    bitexact += 1;
                }
                if mv > FN_GDN_THRESH || nv > 0 {
                    fails.push(format!("({tag} stage) {name} maxdiff={mv:.3e} nan={nv}"));
                    println!(
                        "device: {dev} | fn-cuda-gdn ({tag}) stage {name}: maxdiff={mv:.3e} nan={nv} | FAIL"
                    );
                }
            }
            println!(
                "device: {dev} | fn-cuda-gdn ({tag}) stages worst={worst:.3e} bitexact {bitexact}/10 (conv ring/l2/prep/scan/state/gate) | {}",
                if worst <= FN_GDN_THRESH {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            report.push_str(&format!(
                " · ({tag}) stages worst={worst:.3e} bitexact={bitexact}/10"
            ));
        }
    }

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | fn-cuda-gdn {report} | ALL PASS (conv 재사용 비트동일·prep/l2/gate 이식 비트동일·scan 이식 ≤2e-4 expf 잔차)"
        ))
    } else {
        Err(format!(
            "fn-cuda-gdn 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// fn-cuda-gdn-neg — 음성대조 2종(원장 17호: 계기도 스스로 검증).
/// (a) conv 3탭 오독(fn_gdn_conv3 — conv_k=4 계약 위반 재현, FNF 판정표
///     핵심 차이): 4탭 오라클 대비 종단 maxdiff가 임계 초과.
/// (b) S0=0 입력: 동일 픽스처에서 상태를 0으로 주면 참출력(오라클은
///     S0≠0)과 이격 — 상태 경로 무시 구현이 탐지됨을 증명(§3.3 의무).
/// 양쪽 모두 초과 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_fn_gdn_negative_check(gguf_main: &str) -> Result<String, String> {
    let g = FnGguf::open(Path::new(gguf_main))?;
    let dims = FnDims::from_gguf(&g)?;
    let fx = FnGdnFixture::load(&g, &dims)?;
    let eps = fx.eps;
    let mut m = FnGdnCuda::open(&dims, eps, &fx.cw_all, &fx.dtb_all, &fx.ssa_all, &fx.nw_all)?;
    let dev = m.cc.device_name.clone();
    let inp = gen_inputs(&dims, 32, 0x170F_0D00_0000_0001u64);
    let (cw_l, dtb_l, ssa_l, nw_l) = fx.layer_consts(0);
    let want = fn_gdn_reference(
        &dims, eps, cw_l, dtb_l, ssa_l, nw_l, &inp.qkv, &inp.z, &inp.b, &inp.a, &inp.ring0,
        &inp.s0, 32,
    );

    // (a) conv 3탭(fn_gdn_conv3 검증 전용 진입) vs 4탭 오라클.
    let got_a = m.gdn_stage_host_conv3(
        0,
        32,
        &inp.qkv,
        &inp.z,
        &inp.b,
        &inp.a,
        Some(&inp.s0),
        Some(&inp.ring0),
    )?;
    let (md_a, nan_a) = maxdiff_nan(&got_a, &want.gated);
    println!(
        "device: {dev} | fn-cuda-gdn (iva) negative control conv3(3-tap misread) vs 4-tap oracle: maxdiff={md_a:.3e} nan={nan_a} | FAIL(expected)"
    );
    let det_a = md_a > FN_GDN_THRESH;

    // (b) S0=0 상태 입력 vs S0≠0 참오라클(상태 경로 판별력 증명).
    let zero_s0 = vec![0f32; dims.dt_rank * 128 * 128];
    let got_b = m.gdn_stage_host(
        0,
        32,
        &inp.qkv,
        &inp.z,
        &inp.b,
        &inp.a,
        Some(&zero_s0),
        Some(&inp.ring0),
    )?;
    let (md_b, nan_b) = maxdiff_nan(&got_b, &want.gated);
    println!(
        "device: {dev} | fn-cuda-gdn (ivb) negative control S0=0 input vs S0!=0 oracle: maxdiff={md_b:.3e} nan={nan_b} | FAIL(expected)"
    );
    let det_b = md_b > FN_GDN_THRESH;

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) conv3-3tap maxdiff={md_a:.3e} (b) S0=0 maxdiff={md_b:.3e} > {FN_GDN_THRESH:.0e} — 검증계기 정상(탭 수 계약·상태 경로 결함 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={md_a:.3e} (b)={md_b:.3e} <= {FN_GDN_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
