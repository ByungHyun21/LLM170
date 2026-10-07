//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: 케이스마다 신규 QsaCuda(제로 캐시) +
//!    오라클 상태를 별도 생성 — 이전 케이스 캐시 행 잔류 없음. 히스토리도
//!    청크로 양측 동일 재생(증분 블록 키 캐시 경계 포함).
//! ② 형상은 실 GGUF 메타에서 자동 열거(FnDims::from_gguf — 추정 금지):
//!    실측 qwen4exp 27B n_head=24·n_kv=2·hd=256·n_rot=64·idx 4×128·top_k
//!    2048·r=4(QSA층 il%4==3 12층). t=1..64·top-k 경계(n_past>2051)·
//!    k_prenormed 경로 포함.
//! ③ 캡처-재생 3방향: 합성 입력(결정론 splitmix64)·실가중 노름(GGUF
//!    blk.{il}.* 4종)·히스토리 청크 재생(2청크 증분).
//! ④ 종단 상태가 유일 불변량: 선택 리스트는 **완전일치**(이산 판정),
//!    연속값은 maxdiff(임계 문서화). 음성대조는 계기 자체 검증(원장 17호).
//!
//! [오라클 계약 — core 직접 미러(파일:행 인용, 전부 D:/LLM170 워크트리)]
//! 오라클은 crates/core/src/qwen4exp/stages/qsa.rs(635행 CPU 황금 계약)를
//! 줄 단위로 베낀 호스트 미러다(전체 워크스페이스가 Windows에서 cargo 불가
//! — G1 원장, G2-G8 노선 계승):
//! - stages/qsa.rs — cpu_attn_row L13-64 · mask_from_list L67-91 ·
//!   qsa_select L99-321(패스 A L112-190 · 블록 키 풀링 L198-208 · 패스 B
//!   L210-307: 4-전개 dot L274-289 · width/n_sel L295-297 ·
//!   select_nth_unstable_by L300 · sort_unstable L303-305) · qsa_sel_list
//!   L323-358 · q rope L508-517(qsa_layer 내) · k_prenormed 분기 L167-181.
//! - crates/core/src/ops.rs — sq_sum L11-31(32세그먼트 f32→f64 결합) ·
//!   rms_norm L33-37 · exp_cr L52-86(게이트 sigmoid용 — 명시적 f64 fma
//!   호너라 **비트동일 직접 미러**) · sigmoid L133-136 · rope_head
//!   L149-163(f32 powf/cos/sin — libm).
//!   [트랜센던트 트윈 계약 — 과제 지정 G5/G6/G7 패턴] 초월함수 3종(rope
//!   theta=base.powf·cos/sin, 소프트맥스 exp)은 호스트 libm(UCRT)과 장치
//!   libdevice가 비트재현 불가(G5 원장: libdevice expf 3.1M 표본 30% ±1ulp)
//!   → 오라클을 twin 모드로도 실행(q_theta/q_sincos_d/q_expf — .cu 트윈과
//!   리터럴까지 동일 DAG)해 **커널↔twin 오라클 비트동일**을 1차 게이트로,
//!   **커널↔core-libm 오라클**(순수 코어 산술 — 이 호스트에서 코어 자신이
//!   계산할 값과 동일) maxdiff를 2차 게이트(임계 문서화)로 판정한다. 선택
//!   리스트는 두 오라클 모두와 **완전일치**를 요구한다(이산 판정 — 이것이
//!   "vs core" 이산 계약). sigmoid(exp_cr)은 트윈 불필요 — 비트동일 미러.
//!
//! [검증층 원장 — sm_89(RTX 4070 SUPER) 실측 2026-10-05, Flash-Next 27B
//! GGUF 실측 형상 n_head=24·n_kv=2·hd=256·n_rot=64·idx 4×128·top_k 2048·r=4]
//! - (a) 결정론 소형 pos0=0 t=64(전체 선택+테일 진화·음수 dot 클리핑 경로):
//!   선택 twin/core 0/0 불일치 · twin attn/kv/bk 0.000e0(비트동일) ·
//!   core-libm attn 4.470e-8 / kv 0.000e0 / bk 0.000e0.
//! - (b) top-k pos0=2560(히스토리 1024+1536 2청크 증분) t=8 — n_past
//!   2561..2568, n_blocks 641 > n_sel 512(129블록 실제 탈락): 선택 0/0 ·
//!   최소 경계갭 2.441e-4(동점 아님 — 이산 판정 근거) · twin 0.000e0 전항 ·
//!   core-libm attn 8.382e-9 / kv 2.384e-7 / bk 1.192e-7.
//! - (c) 실가중 노름(GGUF blk.47.* F32 실측) pos0=2100 t=4: 선택 0/0 ·
//!   경계갭 1.274e-2 · twin 0.000e0 · core-libm attn 1.490e-8 / kv 2.384e-7.
//! - (d) k_prenormed(안티패턴 가드 — layers.rs:170 재적용 금지): 선택 0/0 ·
//!   twin 0.000e0(kv 적립 = 입력 비트동일 복사) · core-libm attn 1.863e-8.
//! - 음성대조: (a) top-k off-by-one cnt 513≠512 — 4/4t 구조적 탐지 ·
//!   (b) 풀링 행 +1 시프트 idx_bk maxdiff 1.988e0·선택 1790건 반전 →
//!   NEG-DETECTED + 비영 exit(원장 17호 — 계기 자체 검증).
//! - 임계: twin 2e-7(실측 0.000e0 — 비트동일 방어 게이트) · core-libm
//!   2e-6(실측 최대 2.384e-7의 ~10배 — 트윈 f64→f32 1회 반올림 대비 UCRT
//!   libm ≤½ulp의 극미 차이가 rope k행에 남는 계급).
//! - [결함 21호 — FND 발견 2026-10-05, 이 프로브의 twin↔libm 교차측정으로
//!   발견] G6 exl3_attn.cu attn_sincos_d(및 attn_cuda_probe.rs 트윈)의
//!   사분면 표가 n=1/n=3 교차 오류((st,−ct)/(−st,ct) — 정확 표는 n=1
//!   (−st,ct)·n=3 (st,−ct), fdlibm __ieee754_rem_pio2 규약). 커널과 오라클이
//!   같은 표를 공유해 자기일치 — 본 QSA 트윈은 바른 표로 수정 후 libm과
//!   ½ulp급 일치(수정 전 kv maxdiff 6.626e0 — 부호 반전급). G6 파일은
//!   원 소유자·리드 회수 대상(본 목표 비간섭 계약).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::exl3_cuda_probe::{Rng, f16_to_f32, gdn_exp_d, gdn_expf, maxdiff_nan};
use crate::rawcuda::fn_support::{FN_GGUF_MAIN, FnDims, FnGguf};
use crate::rawcuda::qsa_cuda::{QsaCuda, QsaNeg};

/// 커널↔twin 오라클 값 maxdiff 임계(G6 어텐션 패리티 2e-7 — 비트동일
/// 기대의 방어 게이트).
const QSA_THRESH_TWIN: f32 = 2e-7;
/// 커널↔core-libm 오라클 값 maxdiff 임계 — 초월함수(θ·cos/sin·exp)의
/// 호스트 libm↔f64 트윈 편차(트윈은 f64→f32 1회 반올림=correctly rounded,
/// UCRT libm ≤½ulp — 양자 일치 시 잔여 ≤~1ulp)가 노름·rope→dot로 전파된
/// 계급. 실측 최대 2.384e-7(rope k행 — 사분면 수정 후, 결함 21호 참조)의
/// ~10배 여유 = 2e-6.
const QSA_THRESH_CORE: f32 = 2e-6;

// ── 트랜센던트 트윈(assets/exl3_fn_qsa.cu qsa_exp_d/qsa_theta/qsa_sincos_d와
// 리터럴까지 동일 DAG — 비트동일 계약, 절대 재작성 금지(원장 17호) ──

/// theta 트윈: e=−2p/n_rot(64·128 전부 2의 거듭제곱 → f64 정확),
/// exp(ln(1e7)·e) — ops.rs L151 powf의 f64 재구성(G6 attn_theta 일반화).
fn q_theta(p: usize, n_rot: usize) -> f32 {
    let e = -(2.0 * p as f64) / n_rot as f64;
    gdn_exp_d(16.11809565095832 * e) as f32
}

/// sincos f64 트윈 — G6 attn_sincos_d와 동일 DAG(Cody-Waite 2분할 환원 +
/// z⁶ Horner + 사분면 k&3). 도메인 0 ≤ a ≤ 2^20.
#[rustfmt::skip]
fn q_sincos_d(a: f64) -> (f64, f64) {
    let invpio2 = 6.36619772367581342433e-01f64;
    let pio2_1 = 1.57079632673412561417e+00f64;
    let pio2_1t = 6.07710050650619224932e-11f64;
    let kd = (a * invpio2 + 0.5).floor();
    let k = kd as i64;
    let mut r = a - k as f64 * pio2_1;
    r -= k as f64 * pio2_1t;
    let z = r * r;
    let st = r
        * (1.0 - z
            * (0.16666666666666666f64
                - z * (0.008333333333333333f64
                    - z * (1.9841269841269841e-04f64
                        - z * (2.7557319223985893e-06f64
                            - z * (2.5052108385441720e-08f64
                                - z * 1.6059043836821613e-10f64))))));
    let ct = 1.0
        - z * (0.5f64
            - z * (0.041666666666666664f64
                - z * (0.0013888888888888889f64
                    - z * (2.4801587301587302e-05f64
                        - z * (2.7557319223985893e-07f64
                            - z * 2.0876756987868100e-09f64)))));
    // 사분면 매핑 — fdlibm 규약(n=1 → (−st, ct) · n=3 → (st, −ct)).
    // [결함 21호 — FND 발견] G6 attn_cuda_probe.rs attn_sincos_d의 표는
    // 1/3이 교차되어 있다(커널과 자기일치해 미발견). 본 트윈은 바른 표.
    match (k & 3) as i32 {
        0 => (ct, st),
        1 => (-st, ct),
        2 => (-ct, -st),
        _ => (st, -ct),
    }
}

// ── 코어 산술 미러(ops.rs — 인용 행 그대로) ──

/// ops.rs sq_sum L11-31 — 32세그먼트 f32 순차 누산 → f64 순차 결합.
fn o_sq_sum(x: &[f32]) -> f64 {
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

/// ops.rs rms_norm L33-37 — scale=1/((sum/len+eps).sqrt() as f32), v·scale·g.
fn o_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = o_sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// ops.rs rope_head L149-163 — twin=false면 코어 그대로(f32 powf·cos·sin
/// libm), true면 트윈(θ=exp(ln·e) f64·sincos f64 → f32 1회 캐스트). 회전
/// f64 정확곱·f32 1회 반올림은 양측 공통 구조.
fn o_rope_head(head: &mut [f32], pos: u32, n_rot: usize, twin: bool) {
    let half = n_rot / 2;
    for p in 0..half {
        let theta = if twin {
            q_theta(p, n_rot)
        } else {
            1e7f32.powf(-(2.0 * p as f32) / n_rot as f32)
        };
        let angle = pos as f32 * theta;
        let (c, s) = if twin {
            let (cd, sd) = q_sincos_d(angle as f64);
            (cd as f32, sd as f32)
        } else {
            (angle.cos(), angle.sin())
        };
        let (x0, x1) = (head[p] as f64, head[p + half] as f64);
        let (cf, sf) = (c as f64, s as f64);
        head[p] = (x0 * cf - x1 * sf) as f32;
        head[p + half] = (x0 * sf + x1 * cf) as f32;
    }
}

/// ops.rs exp_cr L52-86 — 코어 자체가 명시적 f64 fma 호너 DAG(트윈 불필요,
/// 비트동일). 게이트 sigmoid 전용.
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

/// ops.rs sigmoid L133-136 — 1/(1+exp_cr(−x)).
fn o_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + o_exp_cr(-x))
}

// ── QSA 코어 오라클(stages/qsa.rs 직접 미러 — 단일 fi 상태) ──

/// SeqState4의 QSA 슬롯 미러(kv_k/kv_v/idx_k/idx_bk/pos — qsa.rs L26-33).
pub(crate) struct QsaOracle {
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    idx_k: Vec<f32>,
    idx_bk: Vec<f32>,
    pos: usize,
}

impl QsaOracle {
    fn new(cap: usize, idx_dim: usize, n_kv: usize, hd: usize) -> Self {
        QsaOracle {
            kv_k: vec![0.0; cap * n_kv * hd],
            kv_v: vec![0.0; cap * n_kv * hd],
            idx_k: vec![0.0; cap * idx_dim],
            idx_bk: Vec::new(),
            pos: 0,
        }
    }

    /// qsa_select L99-321 미러(단일 스레드 — 코어의 행별 병렬은 순서 무관).
    /// 반환: (sel_blk, sel_cnt, sel_stride, 최소 경계 갭) — 갭은 선택 완전
    /// 유일성의 증거(이산 판정의 근거 보고).
    #[allow(clippy::too_many_arguments)]
    fn select(
        &mut self,
        twin: bool,
        dims: &FnDims,
        knw: &[f32],
        iqw: &[f32],
        ikw: &[f32],
        kk: &[Vec<f32>],
        vv: &[Vec<f32>],
        iq: &[Vec<f32>],
        ik: &[Vec<f32>],
        t_len: usize,
        k_prenormed: bool,
    ) -> Result<(Vec<u32>, Vec<u32>, usize, f32), String> {
        let (n_kv, hd) = (dims.n_kv, dims.head_dim);
        let (n_rot, idx_dim, idx_heads) = (dims.n_rot, dims.idx_dim, dims.idx_heads);
        let (rope_base, eps) = (dims.rope_base, dims.eps);
        let r = 4usize; // 픽스처 계약(compress QSA층 전부 4 — 모듈 ctor 가드와 동일)
        let pos0 = self.pos;
        let _ = rope_base;
        // 패스 A(L112-190): 캐시 적립 + 인덱서 q_rope.
        let mut q_rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); t_len];
        for t in 0..t_len {
            let pos = pos0 + t;
            for h in 0..n_kv {
                let lo = h * hd;
                let dst = (pos0 + t) * n_kv * hd + lo;
                if k_prenormed {
                    // L139-144: 이미 norm+rope된 k — 그대로 적립(재적용 금지).
                    self.kv_k[dst..dst + hd].copy_from_slice(&kk[t][lo..lo + hd]);
                } else {
                    let mut head = o_rms_norm(&kk[t][lo..lo + hd], knw, eps);
                    o_rope_head(&mut head, pos as u32, n_rot, twin);
                    self.kv_k[dst..dst + hd].copy_from_slice(&head);
                }
                self.kv_v[dst..dst + hd].copy_from_slice(&vv[t][lo..lo + hd]);
            }
            self.idx_k[(pos0 + t) * idx_dim..(pos0 + t + 1) * idx_dim]
                .copy_from_slice(&ik[t][..idx_dim]);
            let mut qr: Vec<Vec<f32>> = Vec::with_capacity(idx_heads);
            for h in 0..idx_heads {
                let lo = h * idx_dim;
                let mut qh = o_rms_norm(&iq[t][lo..lo + idx_dim], iqw, eps);
                o_rope_head(&mut qh, pos as u32, idx_dim, twin);
                qr.push(qh);
            }
            q_rows[t] = qr;
        }
        // 블록 키 캐시 증분(L198-208) — 청크 끝까지의 완전 블록.
        let n_blocks_max = (pos0 + t_len) / r;
        if self.idx_bk.len() < n_blocks_max * idx_dim {
            let b0 = self.idx_bk.len() / idx_dim;
            self.idx_bk.resize(n_blocks_max * idx_dim, 0.0);
            for b in b0..n_blocks_max {
                let mut pooled = vec![0.0f32; idx_dim];
                for j in 0..r {
                    let src = (b * r + j) * idx_dim;
                    for i2 in 0..idx_dim {
                        pooled[i2] += self.idx_k[src + i2];
                    }
                }
                for v in pooled.iter_mut() {
                    *v /= r as f32;
                }
                let mut pk = o_rms_norm(&pooled, ikw, eps);
                o_rope_head(&mut pk, (b * r) as u32, idx_dim, twin);
                self.idx_bk[b * idx_dim..(b + 1) * idx_dim].copy_from_slice(&pk);
            }
        }
        // 패스 B(L210-307): 블록 점수 + top-k 선택.
        let idx_top_k = dims.idx_top_k;
        let sel_stride = idx_top_k / r + 2;
        let mut sel_blk: Vec<u32> = vec![0u32; t_len * sel_stride];
        let mut sel_cnt: Vec<u32> = vec![0u32; t_len];
        let mut min_gap = f32::INFINITY;
        for t in 0..t_len {
            let n_past = pos0 + t + 1;
            let n_blocks = n_past / r;
            let bk = &self.idx_bk[..n_blocks * idx_dim];
            let mut block_score = vec![0.0f32; n_blocks];
            for b in 0..n_blocks {
                let pk = &bk[b * idx_dim..(b + 1) * idx_dim];
                for qh in &q_rows[t] {
                    let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let mut i2 = 0usize;
                    while i2 + 4 <= idx_dim {
                        d0 += qh[i2] * pk[i2];
                        d1 += qh[i2 + 1] * pk[i2 + 1];
                        d2 += qh[i2 + 2] * pk[i2 + 2];
                        d3 += qh[i2 + 3] * pk[i2 + 3];
                        i2 += 4;
                    }
                    while i2 < idx_dim {
                        d0 += qh[i2] * pk[i2];
                        i2 += 1;
                    }
                    let dot = (d0 + d1) + (d2 + d3);
                    if dot > 0.0 {
                        block_score[b] += dot;
                    }
                }
            }
            let width = n_past.min(idx_top_k + r - 1);
            let tail_cnt = n_past - n_blocks * r;
            let n_sel_blocks = ((width - tail_cnt) / r).min(n_blocks);
            let mut sel_blocks: Vec<usize> = (0..n_blocks).collect();
            if n_sel_blocks < n_blocks {
                sel_blocks.select_nth_unstable_by(n_sel_blocks, |&a, &b| {
                    block_score[b]
                        .partial_cmp(&block_score[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let sb: Vec<usize> = sel_blocks[..n_sel_blocks].to_vec();
            // 경계 완전 유일성(이산 판정 근거) — 동점이면 픽스처 부적격.
            if n_sel_blocks < n_blocks {
                let sel_min = sb
                    .iter()
                    .map(|&b| block_score[b])
                    .fold(f32::INFINITY, f32::min);
                let unsel_max = (0..n_blocks)
                    .filter(|b| !sb.contains(b))
                    .map(|b| block_score[b])
                    .fold(f32::NEG_INFINITY, f32::max);
                if unsel_max >= sel_min {
                    return Err(format!(
                        "qsa 오라클: t={t} 선택 경계 동점(선택 집합 비유일) — 픽스처 부적격"
                    ));
                }
                min_gap = min_gap.min(sel_min - unsel_max);
            }
            let mut sbs = sb;
            sbs.sort_unstable();
            for (k2, &b) in sbs.iter().enumerate() {
                sel_blk[t * sel_stride + k2] = b as u32;
            }
            sel_cnt[t] = n_sel_blocks as u32;
        }
        self.pos += t_len;
        Ok((sel_blk, sel_cnt, sel_stride, min_gap))
    }

    /// qsa_sel_list L323-358 미러(정수 논리).
    fn sel_list(
        &self,
        sel_blk: &[u32],
        sel_cnt: &[u32],
        sel_stride: usize,
        r: usize,
        pos0: usize,
        n_tok: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut sel_off: Vec<u32> = vec![0u32; n_tok + 1];
        for t2 in 0..n_tok {
            let n_past = pos0 + t2 + 1;
            let tail_cnt = n_past - (n_past / r) * r;
            sel_off[t2 + 1] = sel_off[t2] + sel_cnt[t2] * r as u32 + tail_cnt as u32;
        }
        let mut sel_idx: Vec<u32> = vec![0u32; sel_off[n_tok] as usize];
        for t2 in 0..n_tok {
            let n_past = pos0 + t2 + 1;
            let tail_start = (n_past / r) * r;
            let mut o = sel_off[t2] as usize;
            for k2 in 0..sel_cnt[t2] as usize {
                let b = sel_blk[t2 * sel_stride + k2] as usize;
                for j in 0..r {
                    sel_idx[o] = (b * r + j) as u32;
                    o += 1;
                }
            }
            for j in tail_start..n_past {
                sel_idx[o] = j as u32;
                o += 1;
            }
        }
        (sel_idx, sel_off)
    }

    /// cpu_attn_row L13-64 + mask_from_list L67-91 미러 — 마스크 경로(코어
    /// CPU 참조 산술). twin=false: .exp() libm(코어 원문), true: 트윈.
    #[allow(clippy::too_many_arguments)]
    fn attn_rows(
        &self,
        twin: bool,
        qg: &[Vec<f32>],
        sel_blk: &[u32],
        sel_cnt: &[u32],
        sel_stride: usize,
        r: usize,
        t_len: usize,
        dims: &FnDims,
        kq_scale: f32,
    ) -> Vec<Vec<f32>> {
        let (n_head, n_kv, hd) = (dims.n_head, dims.n_kv, dims.head_dim);
        let pos0 = self.pos - t_len;
        let mut out = vec![vec![0.0f32; n_head * hd]; t_len];
        for t in 0..t_len {
            let n_past = pos0 + t + 1;
            // mask_from_list L78-90(마스크_all 비어있는 경로 — GPU 규약과 동일).
            let mut m = vec![false; n_past];
            let tail_start = (n_past / r) * r;
            for k2 in 0..sel_cnt[t] as usize {
                let b = sel_blk[t * sel_stride + k2] as usize;
                for j in b * r..(b + 1) * r {
                    m[j] = true;
                }
            }
            for j in tail_start..n_past {
                m[j] = true;
            }
            // cpu_attn_row L16-63.
            let q_t = &qg[t];
            for h in 0..n_head {
                let kvh = h / (n_head / n_kv);
                let mut maxv = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_past];
                for (p, sc) in scores.iter_mut().enumerate() {
                    if !m[p] {
                        *sc = f32::NEG_INFINITY;
                        continue;
                    }
                    let b = p * n_kv * hd + kvh * hd;
                    let mut d = 0.0f32;
                    for i in 0..hd {
                        d += q_t[h * 2 * hd + i] * self.kv_k[b + i];
                    }
                    *sc = d * kq_scale;
                    maxv = maxv.max(*sc);
                }
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = if twin {
                        gdn_expf(*sc - maxv)
                    } else {
                        (*sc - maxv).exp()
                    };
                    sum += *sc;
                }
                let ob = h * hd;
                for (p, sc) in scores.iter().enumerate() {
                    let w = sc / sum;
                    if w == 0.0 {
                        continue;
                    }
                    let b = p * n_kv * hd + kvh * hd;
                    for i in 0..hd {
                        out[t][ob + i] += w * self.kv_v[b + i];
                    }
                }
                let gb = h * 2 * hd + hd;
                for i in 0..hd {
                    out[t][ob + i] *= o_sigmoid(q_t[gb + i]);
                }
            }
        }
        out
    }
}

// ── 픽스처 ──

/// 1개 케이스의 입력 일체(청크 순서대로 재생 — 히스토리 증분 포함).
struct QsaFx {
    il: usize,
    knw: Vec<f32>,
    qnw: Vec<f32>,
    iqw: Vec<f32>,
    ikw: Vec<f32>,
    /// (kk, vv, iq, ik, qg) 청크열 — 마지막 청크가 판정 대상.
    chunks: Vec<(
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    )>,
}

/// GGUF 노름 텐서 → f32(ty 0=F32·1=F16·30=BF16 — 형상 명시 판독 계약).
fn gguf_norm_f32(g: &FnGguf, name: &str, want: usize) -> Result<Vec<f32>, String> {
    let t = g
        .tensor(name)
        .ok_or_else(|| format!("gguf 텐서 없음: {name}"))?;
    let rows = t.dims.get(1).copied().unwrap_or(1);
    let raw = g.read_rows(name, 0, rows)?;
    let v: Vec<f32> = match t.ty {
        0 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        1 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        30 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        other => return Err(format!("{name}: gguf ty {other} 미지원(노름 텐서)")),
    };
    if v.len() != want {
        return Err(format!("{name}: {} != {want}", v.len()));
    }
    Ok(v)
}

/// 케이스 생성 — real_norms=true면 대상 층 노름 4종을 실 GGUF에서 판독,
/// 아니면 결정론 합성(0.5..1.5 — G5/G6 합성 노름 계급). 입력 행은 결정론
/// 시드(±amp — 잔류 스트림 계급).
#[allow(clippy::too_many_arguments)]
fn fx_generate(
    dims: &FnDims,
    g: Option<&FnGguf>,
    il: usize,
    real_norms: bool,
    hist_chunks: &[usize],
    t_test: usize,
    seed: u64,
    prenormed: bool,
    posit: bool,
) -> Result<QsaFx, String> {
    let (hd, id) = (dims.head_dim, dims.idx_dim);
    let (knw, qnw, iqw, ikw) = if real_norms {
        let g = g.ok_or("real_norms에 GGUF 필요")?;
        (
            gguf_norm_f32(g, &format!("blk.{il}.attn_k_norm.weight"), hd)?,
            gguf_norm_f32(g, &format!("blk.{il}.attn_q_norm.weight"), hd)?,
            gguf_norm_f32(g, &format!("blk.{il}.indexer.q_norm.weight"), id)?,
            gguf_norm_f32(g, &format!("blk.{il}.indexer.k_norm.weight"), id)?,
        )
    } else {
        let mut rng = Rng::new(seed ^ 0x5EED_0000_0000_0001);
        let syn = |n: usize, r: &mut Rng| -> Vec<f32> {
            (0..n).map(|_| (0.5 + r.next_f64()) as f32).collect()
        };
        (
            syn(hd, &mut rng),
            syn(hd, &mut rng),
            syn(id, &mut rng),
            syn(id, &mut rng),
        )
    };
    let mut rng = Rng::new(seed);
    let unif = |rng: &mut Rng, amp: f64| ((rng.next_f64() * 2.0 - 1.0) * amp) as f32;
    // 공통 저주파 성분(마지막 rope 쌍 (63,127) — θ₆₃=1e7^(−126/128)≈1.3e-7로
    // 위치 의존 회전이 사실상 없다): 인덱서 q·k가 이 방향을 공유하면 블록
    // 점수가 양수 중심이 되어 top-k 경계가 0점 동점 클래스(패스 B의
    // max(dot,0) 클리핑이 만드는 구조적 동점 — select_nth 집합 비유일,
    // qsa.rs L290-291)에 빠지지 않는다. 실측 계급: 학습된 인덱서 어텐션의
    // 국소 양상향. posit=false(경계 없는 소형 케이스)는 대칭 잡음 그대로 —
    // 음수 dot 클리핑 경로를 그쪽에서 검증.
    let common: Vec<f32> = if posit {
        let mut c = vec![0f32; dims.idx_dim];
        c[63] = 0.7;
        c[127] = 0.7;
        c
    } else {
        vec![0f32; dims.idx_dim]
    };
    let mut chunks = Vec::new();
    let mut pos = 0usize;
    let gen_chunk = |rng: &mut Rng,
                     t_len: usize,
                     pos0: usize|
     -> (
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    ) {
        let mut kk = Vec::with_capacity(t_len);
        let mut vv = Vec::with_capacity(t_len);
        let mut iq = Vec::with_capacity(t_len);
        let mut ik = Vec::with_capacity(t_len);
        let mut qg = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let krow: Vec<f32> = (0..dims.n_kv * hd).map(|_| unif(rng, 0.5)).collect();
            // prenormed: k행을 호스트에서 미리 norm+rope(twin — 양측 동일 입력).
            let krow = if prenormed {
                let mut out = vec![0f32; dims.n_kv * hd];
                for h in 0..dims.n_kv {
                    let mut head = o_rms_norm(&krow[h * hd..(h + 1) * hd], &knw, dims.eps);
                    o_rope_head(&mut head, (pos0 + t) as u32, dims.n_rot, true);
                    out[h * hd..(h + 1) * hd].copy_from_slice(&head);
                }
                out
            } else {
                krow
            };
            kk.push(krow);
            vv.push((0..dims.n_kv * hd).map(|_| unif(rng, 0.5)).collect());
            iq.push(
                (0..dims.idx_heads * id)
                    .map(|i| 0.9 * common[i % id] + unif(rng, 0.25))
                    .collect(),
            );
            ik.push((0..id).map(|i| 0.9 * common[i] + unif(rng, 0.25)).collect());
            qg.push((0..dims.n_head * 2 * hd).map(|_| unif(rng, 0.3)).collect());
        }
        (kk, vv, iq, ik, qg)
    };
    for &hc in hist_chunks {
        chunks.push(gen_chunk(&mut rng, hc, pos));
        pos += hc;
    }
    chunks.push(gen_chunk(&mut rng, t_test, pos));
    Ok(QsaFx {
        il,
        knw,
        qnw,
        iqw,
        ikw,
        chunks,
    })
}

/// 선택 목록 비교 — 불일치 원소 수(0이어야 PASS)와 첫 불일치 보고.
fn sel_cmp(
    got_blk: &[u32],
    got_cnt: &[u32],
    want_blk: &[u32],
    want_cnt: &[u32],
) -> (usize, String) {
    let mut bad = 0usize;
    let mut first = String::new();
    for t in 0..want_cnt.len() {
        if got_cnt[t] != want_cnt[t] {
            bad += 1;
            if first.is_empty() {
                first = format!("t={t} cnt {}!={}", got_cnt[t], want_cnt[t]);
            }
            continue;
        }
        for k2 in 0..want_cnt[t] as usize {
            if got_blk[t * (want_blk.len() / want_cnt.len().max(1)) + k2]
                != want_blk[t * (want_blk.len() / want_cnt.len().max(1)) + k2]
            {
                bad += 1;
                if first.is_empty() {
                    first = format!("t={t} k={k2}");
                }
            }
        }
    }
    (bad, first)
}

/// 케이스 실행 — 양 오라클(twin·core-libm)과 모듈을 동일 청크열로 재생,
/// (i) 선택 완전일치(양측) (ii) 종단 값 maxdiff(twin 2e-7·core 문서화 임계)
/// 를 판정. 반환: 보고 라인 모음(원장용).
fn qsa_run_case(
    tag: &str,
    dims: &FnDims,
    fx: &QsaFx,
    prenormed: bool,
    dev: &str,
    fails: &mut Vec<String>,
    report: &mut String,
) -> Result<(), String> {
    let (n_kv, hd) = (dims.n_kv, dims.head_dim);
    let r = 4usize;
    let kq_scale = dims.kq_scale();
    let mut orb_twin = QsaOracle::new(8192, dims.idx_dim, n_kv, hd);
    let mut orb_core = QsaOracle::new(8192, dims.idx_dim, n_kv, hd);
    let mut qm = QsaCuda::new(dims.clone(), 2600)?;
    qm.set_pos(0)?;
    qm.set_norms(fx.il, &fx.knw, &fx.qnw, &fx.iqw, &fx.ikw)?;
    let n_chunks = fx.chunks.len();
    let mut md_twin = 0f32;
    let mut md_core = 0f32;
    let mut md_bk_twin = 0f32;
    let mut md_bk_core = 0f32;
    let mut min_gap = f32::INFINITY;
    let mut md_kv_twin = 0f32;
    let mut md_kv_core = 0f32;
    let mut sel_bad_twin = 0usize;
    let mut sel_bad_core = 0usize;
    for (ci, (kk, vv, iq, ik, qg)) in fx.chunks.iter().enumerate() {
        let t_len = kk.len();
        let last = ci + 1 == n_chunks;
        let pos0 = orb_twin.pos;
        let (want_blk_tw, want_cnt_tw, stride_tw, gap_tw) = orb_twin.select(
            true, dims, &fx.knw, &fx.iqw, &fx.ikw, kk, vv, iq, ik, t_len, prenormed,
        )?;
        let (want_blk_co, want_cnt_co, _, gap_co) = orb_core.select(
            false, dims, &fx.knw, &fx.iqw, &fx.ikw, kk, vv, iq, ik, t_len, prenormed,
        )?;
        min_gap = min_gap.min(gap_tw).min(gap_co);
        // 모듈: 마지막 청크는 스테이지 전체(select+q rope+어텐션), 나머지는 select만.
        let (got_blk, got_cnt, got_stride) =
            qm.qsa_select(fx.il, kk, vv, iq, ik, t_len, prenormed, QsaNeg::Off)?;
        let (b1, f1) = sel_cmp(&got_blk, &got_cnt, &want_blk_tw, &want_cnt_tw);
        let (b2, f2) = sel_cmp(&got_blk, &got_cnt, &want_blk_co, &want_cnt_co);
        sel_bad_twin += b1;
        sel_bad_core += b2;
        if b1 > 0 && fails.len() < 8 {
            fails.push(format!("{tag} chunk{ci} 선택 불일치(twin): {}건 {f1}", b1));
        }
        if b2 > 0 && fails.len() < 8 {
            fails.push(format!("{tag} chunk{ci} 선택 불일치(core): {}건 {f2}", b2));
        }
        if got_stride != stride_tw {
            fails.push(format!(
                "{tag} chunk{ci} sel_stride {got_stride} != {stride_tw}"
            ));
        }
        // 캐시 상태 대조(첫 청크 이후 매 청크 — 증분 검증).
        let fi = qm.full_idx(fx.il)?;
        let (mkv, vkv) = qm.read_kv(fi, pos0 + t_len)?;
        let midx = qm.read_idx_k(fi, pos0 + t_len)?;
        let n_blk = (pos0 + t_len) / r;
        let mbk = qm.read_idx_bk(fi, n_blk)?;
        let (m1, _) = maxdiff_nan(&mkv, &orb_twin.kv_k[..(pos0 + t_len) * n_kv * hd]);
        let (m2, _) = maxdiff_nan(&mkv, &orb_core.kv_k[..(pos0 + t_len) * n_kv * hd]);
        md_kv_twin = md_kv_twin.max(m1);
        md_kv_core = md_kv_core.max(m2);
        let (v1, _) = maxdiff_nan(&vkv, &orb_twin.kv_v[..(pos0 + t_len) * n_kv * hd]);
        let (v2, _) = maxdiff_nan(&vkv, &orb_core.kv_v[..(pos0 + t_len) * n_kv * hd]);
        md_kv_twin = md_kv_twin.max(v1);
        md_kv_core = md_kv_core.max(v2);
        let (i1, _) = maxdiff_nan(&midx, &orb_twin.idx_k[..(pos0 + t_len) * dims.idx_dim]);
        let (i2, _) = maxdiff_nan(&midx, &orb_core.idx_k[..(pos0 + t_len) * dims.idx_dim]);
        md_kv_twin = md_kv_twin.max(i1);
        md_kv_core = md_kv_core.max(i2);
        let (k1, _) = maxdiff_nan(&mbk, &orb_twin.idx_bk[..n_blk * dims.idx_dim]);
        let (k2, _) = maxdiff_nan(&mbk, &orb_core.idx_bk[..n_blk * dims.idx_dim]);
        md_bk_twin = md_bk_twin.max(k1);
        md_bk_core = md_bk_core.max(k2);
        if last {
            // 스테이지 종단 — 어텐션+게이트.
            let got_attn = qm.qsa_stage(fx.il, qg, kk, vv, iq, ik, t_len, prenormed)?;
            // 오라클 q rope(L508-517) 후 어텐션 — twin/libm 각각.
            let mut qg_tw: Vec<Vec<f32>> = qg.clone();
            let mut qg_co: Vec<Vec<f32>> = qg.clone();
            for (t, row) in qg_tw.iter_mut().enumerate() {
                for h in 0..dims.n_head {
                    let lo = h * 2 * hd;
                    let mut qh = o_rms_norm(&row[lo..lo + hd], &fx.qnw, dims.eps);
                    o_rope_head(&mut qh, (pos0 + t) as u32, dims.n_rot, true);
                    row[lo..lo + hd].copy_from_slice(&qh);
                }
            }
            for (t, row) in qg_co.iter_mut().enumerate() {
                for h in 0..dims.n_head {
                    let lo = h * 2 * hd;
                    let mut qh = o_rms_norm(&row[lo..lo + hd], &fx.qnw, dims.eps);
                    o_rope_head(&mut qh, (pos0 + t) as u32, dims.n_rot, false);
                    row[lo..lo + hd].copy_from_slice(&qh);
                }
            }
            let want_tw = orb_twin.attn_rows(
                true,
                &qg_tw,
                &want_blk_tw,
                &want_cnt_tw,
                stride_tw,
                r,
                t_len,
                dims,
                kq_scale,
            );
            let want_co = orb_core.attn_rows(
                false,
                &qg_co,
                &want_blk_co,
                &want_cnt_co,
                stride_tw,
                r,
                t_len,
                dims,
                kq_scale,
            );
            for t in 0..t_len {
                let (a1, n1) = maxdiff_nan(&got_attn[t], &want_tw[t]);
                let (a2, n2) = maxdiff_nan(&got_attn[t], &want_co[t]);
                if n1 + n2 > 0 {
                    fails.push(format!("{tag} t={t} NaN {n1}/{n2}"));
                }
                md_twin = md_twin.max(a1);
                md_core = md_core.max(a2);
            }
        }
        qm.pos_bump(t_len as u32)?;
    }
    let ok_sel = sel_bad_twin == 0 && sel_bad_core == 0;
    let ok_val =
        md_twin <= QSA_THRESH_TWIN && md_core <= QSA_THRESH_CORE && md_kv_twin <= QSA_THRESH_TWIN;
    println!(
        "device: {dev} | qsa {tag}: sel twin/core {sel_bad_twin}/{sel_bad_core}불일치 · min 경계갭 {min_gap:.3e} · twin maxdiff attn={md_twin:.3e} kv={md_kv_twin:.3e} bk={md_bk_twin:.3e} · core-libm attn={md_core:.3e} kv={md_kv_core:.3e} bk={md_bk_core:.3e} | {}",
        if ok_sel && ok_val { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(
        "{tag}: sel_bad={sel_bad_twin}/{sel_bad_core} min_gap={min_gap:.3e} twin(attn/kv/bk)={md_twin:.3e}/{md_kv_twin:.3e}/{md_bk_twin:.3e} core(attn/kv/bk)={md_core:.3e}/{md_kv_core:.3e}/{md_bk_core:.3e}
"
    ));
    if !ok_sel {
        fails.push(format!("{tag}: 선택 리스트 불일치(이산 판정 위반)"));
    }
    if md_twin > QSA_THRESH_TWIN {
        fails.push(format!(
            "{tag}: twin maxdiff {md_twin:.3e} > {QSA_THRESH_TWIN:.0e}"
        ));
    }
    if md_core > QSA_THRESH_CORE {
        fails.push(format!(
            "{tag}: core-libm maxdiff {md_core:.3e} > {QSA_THRESH_CORE:.0e}"
        ));
    }
    if md_kv_twin > QSA_THRESH_TWIN {
        fails.push(format!(
            "{tag}: twin 캐시 maxdiff {md_kv_twin:.3e} > {QSA_THRESH_TWIN:.0e}"
        ));
    }
    if md_bk_twin > QSA_THRESH_TWIN {
        fails.push(format!(
            "{tag}: twin 블록키 maxdiff {md_bk_twin:.3e} > {QSA_THRESH_TWIN:.0e}"
        ));
    }
    Ok(())
}

/// exl3-cuda-qsa — QSA 스테이지 프로브(FND). 케이스:
/// (a) 결정론 소형 pos0=0 t=64(선택 전체 경로+테일 진화)
/// (b) 결정론 top-k pos0=2560(2청크 증분) t=8 — n_past>2051로 실제 top-k
///     탈락(n_blocks 641 > n_sel 512) 발생
/// (c) 실가중 노름(GGUF blk.47.*) pos0=2100 t=4 — top-k 경계 직후
/// (d) k_prenormed 경로(안티패턴 가드 — 재적용 금지 검증) t=4.
pub fn cuda_qsa_check(gguf_main: Option<&str>) -> Result<String, String> {
    let path = gguf_main.unwrap_or(FN_GGUF_MAIN);
    let g = FnGguf::open(std::path::Path::new(path))?;
    let dims = FnDims::from_gguf(&g)?;
    let dev = {
        let cc = crate::rawcuda::ctx::CudaCtx::new()?;
        cc.device_name.clone()
    };
    // 형상 실측 게이트(과제 계약: 4헤드×128×2048).
    if (dims.idx_heads, dims.idx_dim, dims.idx_top_k) != (4, 128, 2048) {
        return Err(format!(
            "qsa: idx {}/{}/{} — 4/128/2048 계약 위반",
            dims.idx_heads, dims.idx_dim, dims.idx_top_k
        ));
    }
    if dims.rope_base != 1e7 {
        return Err(format!("qsa: rope_base={} — 1e7 계약", dims.rope_base));
    }
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();
    let qsa_ils: Vec<usize> = (0..dims.n_layer)
        .filter(|&il| dims.compress[il] != 0)
        .collect();
    report.push_str(&format!(
        "dims: n_head={} n_kv={} hd={} n_rot={} idx={:?} qsa_layers={} r=4 cap=2600
",
        dims.n_head,
        dims.n_kv,
        dims.head_dim,
        dims.n_rot,
        (dims.idx_heads, dims.idx_dim, dims.idx_top_k),
        qsa_ils.len()
    ));
    // (a) 소형 — 첫 QSA층.
    let fx_a = fx_generate(
        &dims,
        Some(&g),
        qsa_ils[0],
        false,
        &[],
        64,
        0x170C_0FD4_0001,
        false,
        false,
    )?;
    qsa_run_case(
        "a-small",
        &dims,
        &fx_a,
        false,
        &dev,
        &mut fails,
        &mut report,
    )?;
    // (b) top-k 대형 — 마지막 QSA층, 히스토리 2청크(증분 블록키).
    let fx_b = fx_generate(
        &dims,
        Some(&g),
        qsa_ils[qsa_ils.len() - 1],
        false,
        &[1024, 1536],
        8,
        0x170C_0FD4_0002,
        false,
        true,
    )?;
    qsa_run_case("b-topk", &dims, &fx_b, false, &dev, &mut fails, &mut report)?;
    // (c) 실가중 노름 — 마지막 QSA층.
    let fx_c = fx_generate(
        &dims,
        Some(&g),
        qsa_ils[qsa_ils.len() - 1],
        true,
        &[2100],
        4,
        0x170C_0FD4_0003,
        false,
        true,
    )?;
    qsa_run_case(
        "c-realnorm",
        &dims,
        &fx_c,
        false,
        &dev,
        &mut fails,
        &mut report,
    )?;
    // (d) k_prenormed — 안티패턴 가드(k 재적용 금지, layers.rs:170).
    let mid = qsa_ils[qsa_ils.len() / 2];
    let fx_d = fx_generate(
        &dims,
        Some(&g),
        mid,
        false,
        &[128],
        4,
        0x170C_0FD4_0004,
        true,
        false,
    )?;
    qsa_run_case(
        "d-prenormed",
        &dims,
        &fx_d,
        true,
        &dev,
        &mut fails,
        &mut report,
    )?;
    if !fails.is_empty() {
        return Err(format!(
            "qsa 프로브 실패:
  {}",
            fails.join(
                "
  "
            )
        ));
    }
    Ok(format!(
        "device: {dev} | flash-next qsa stage PASS (twin 비트동일 게이트 {} · core-libm {} · 선택 완전일치 4케이스)
{report}",
        QSA_THRESH_TWIN, QSA_THRESH_CORE
    ))
}

/// exl3-cuda-qsa-neg — 음성대조(원장 17호). 두 결함류가 모두 탐지되어야:
/// (a) top-k off-by-one(sel_delta=+1 — cnt 513≠512 구조적 불일치)
/// (b) 잘못된 풀링(블록 키 원천 행 +1 — idx_bk 값 이격·선택 반전).
/// 정상 동작 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_qsa_negative_check(gguf_main: Option<&str>) -> Result<String, String> {
    let path = gguf_main.unwrap_or(FN_GGUF_MAIN);
    let g = FnGguf::open(std::path::Path::new(path))?;
    let dims = FnDims::from_gguf(&g)?;
    let dev = {
        let cc = crate::rawcuda::ctx::CudaCtx::new()?;
        cc.device_name.clone()
    };
    let qsa_ils: Vec<usize> = (0..dims.n_layer)
        .filter(|&il| dims.compress[il] != 0)
        .collect();
    let il = qsa_ils[qsa_ils.len() - 1];
    let fx = fx_generate(
        &dims,
        Some(&g),
        il,
        false,
        &[1024, 1536],
        4,
        0x170C_0FD4_00F1,
        false,
        true,
    )?;
    let r = 4usize;
    let kq_scale = dims.kq_scale();
    let mut orb = QsaOracle::new(8192, dims.idx_dim, dims.n_kv, dims.head_dim);
    let mut qm = QsaCuda::new(dims.clone(), 2600)?;
    qm.set_pos(0)?;
    qm.set_norms(il, &fx.knw, &fx.qnw, &fx.iqw, &fx.ikw)?;
    let mut last: Option<(Vec<u32>, Vec<u32>, usize)> = None;
    for (ci, (kk, vv, iq, ik, qg)) in fx.chunks.iter().enumerate() {
        let t_len = kk.len();
        let (blk, cnt, stride, _) = orb.select(
            true, &dims, &fx.knw, &fx.iqw, &fx.ikw, kk, vv, iq, ik, t_len, false,
        )?;
        qm.qsa_select(il, kk, vv, iq, ik, t_len, false, QsaNeg::Off)?;
        if ci + 1 == fx.chunks.len() {
            last = Some((blk, cnt, stride));
        }
        qm.pos_bump(t_len as u32)?;
        let _ = qg;
    }
    let (want_blk, want_cnt, _stride) = last.ok_or("neg: 청크 없음")?;
    // (a) top-k off-by-one — 마지막 청크 재실행(상태는 이미 전진했으므로
    // fresh 모듈로 동일 재생 후 delta만 변경).
    let mut qm2 = QsaCuda::new(dims.clone(), 2600)?;
    qm2.set_pos(0)?;
    qm2.set_norms(il, &fx.knw, &fx.qnw, &fx.iqw, &fx.ikw)?;
    for (kk, vv, iq, ik, _) in fx.chunks.iter() {
        let t_len = kk.len();
        qm2.qsa_select(il, kk, vv, iq, ik, t_len, false, QsaNeg::Off)?;
        qm2.pos_bump(t_len as u32)?;
    }
    // pos가 이미 끝까지 전진 — delta 실행은 set_pos 롤백 후 마지막 청크만.
    qm2.set_pos(2560)?; // 마지막 청크 pos0 = 1024+1536(히스토리 합)
    let (kk_l, vv_l, iq_l, ik_l, _) = fx.chunks.last().unwrap();
    let (neg_blk, neg_cnt, _) =
        qm2.qsa_select(il, kk_l, vv_l, iq_l, ik_l, 4, false, QsaNeg::TopkOff1)?;
    let (bad_a, first_a) = sel_cmp(&neg_blk, &neg_cnt, &want_blk, &want_cnt);
    println!(
        "device: {dev} | qsa-neg (a) top-k off-by-one: 선택 불일치 {bad_a}건({first_a}) | {}",
        if bad_a > 0 { "DETECTED" } else { "MISSED" }
    );
    // (b) 잘못된 풀링 — fresh 모듈, 히스토리 재생 후 shift=1로 블록 키 재구성.
    let mut qm3 = QsaCuda::new(dims.clone(), 2600)?;
    qm3.set_pos(0)?;
    qm3.set_norms(il, &fx.knw, &fx.qnw, &fx.iqw, &fx.ikw)?;
    for (kk, vv, iq, ik, _) in fx.chunks.iter() {
        let t_len = kk.len();
        qm3.qsa_select(il, kk, vv, iq, ik, t_len, false, QsaNeg::PoolShift1)?;
        qm3.pos_bump(t_len as u32)?;
    }
    let fi = qm3.full_idx(il)?;
    let n_blk = 2564 / r;
    let mbk = qm3.read_idx_bk(fi, n_blk)?;
    let (md_bk, _) = maxdiff_nan(&mbk, &orb.idx_bk[..n_blk * dims.idx_dim]);
    // 선택 비교: shift 실행 결과(sel 목록)는 마지막 qsa_select가 반환한 것.
    let (neg2_blk, neg2_cnt, _) = {
        // 마지막 청크 결과를 다시 얻는다(shift 상태에서 재실행 — pos 롤백).
        qm3.set_pos(2560)?;
        let (b, c, s) = qm3.qsa_select(il, kk_l, vv_l, iq_l, ik_l, 4, false, QsaNeg::PoolShift1)?;
        (b, c, s)
    };
    let (bad_b, first_b) = sel_cmp(&neg2_blk, &neg2_cnt, &want_blk, &want_cnt);
    let detected_b = bad_b > 0 || md_bk > QSA_THRESH_TWIN;
    println!(
        "device: {dev} | qsa-neg (b) wrong pooling(+1 shift): idx_bk maxdiff {md_bk:.3e} · 선택 불일치 {bad_b}건({first_b}) | {}",
        if detected_b { "DETECTED" } else { "MISSED" }
    );
    let _ = kq_scale;
    if bad_a > 0 && detected_b {
        Err(format!(
            "NEG-DETECTED (a) top-k off-by-one {bad_a}건 · (b) pooling shift idx_bk maxdiff={md_bk:.3e}/선택 {bad_b}건 — 검증계기 정상(결함 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED a={bad_a} b_sel={bad_b} b_bk={md_bk:.3e} — 검증계기 결함: 결함 미탐지"
        ))
    }
}
