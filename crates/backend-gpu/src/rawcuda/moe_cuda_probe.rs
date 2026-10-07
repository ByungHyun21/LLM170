//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: (i)/(ii-a)/(ii-b)/음성대조 케이스는
//!    각각 독립 MoeCuda(독립 가중치 상주)로 실행 — 케이스 간 디바이스
//!    상태 공유 없음(단일 상주 원칙도 케이스별 drop으로 준수).
//! ② 형상은 실측 config에서 자동 판독: (i) 27B 폭+Flash-Next MoE 헤드
//!    합성 · (ii-a) 35B-A3B config.json · (ii-b) Flash-Next config.json
//!    (과제 지정 경로) — 라우팅 존(포화·동률·근접) 고정 케이스.
//! ③ 캡처-재생: 합성 픽스처(결정론 시드·존 제어) — 모듈과 오라클이 동일
//!    f16 바이트를 소비(전문가 가중치는 동일 생성기 공유).
//! ④ 종단 값이 불변량: 판정은 moe_ffn 출력 값 maxdiff + 라우팅 이산
//!    선택(전문가 id 순열) exact-match — 라우팅은 이산 계약이라 예외
//!    (과제 계약: "routing decisions must match EXACTLY").
//!
//! [오라클 — core qwen4exp 미러(값 maxdiff 판정의 유일 기준)]
//! crates/core/src/qwen4exp/stages/moe.rs moe_ffn(L22-320)을 그대로 재생:
//! - 라우터·sgate 배치 mm(L40-45) → dot 순차 f32 누산(core matmul cpu.rs
//!   L91-94 acc += x[i]*scratch[i] 미러 — 디양자화는 f16→f32 정확 변환).
//! - softmax+선택(L46-70): mx fold → exp → zs 순차(오름차 e) → 나눗셈 →
//!   total_cmp 내림 안정 정렬(동률 id 오름차) → top-n_used → wsum 하한
//!   6.1035156e-5 → w!=0 스킵. [치환 계약] core L48의 std f32::exp는
//!   기기 간 이식 불가(시스템 libm)라 양측(오라클·커널) 모두 core
//!   exp_cr(ops.rs L52-89)의 직이식 미러로 치환한다(G5/G7 트윈 노선,
//!   plans/124 §6 — 단조라 이산 순서 불변·동일 비트 입력에 동일 출력).
//!   silu·sigmoid는 치환 아님: core 자체가 exp_cr을 쓴다(ops.rs
//!   L127-135) — 아래 미러와 비트동일.
//! - 전문가 서브배치 산술(L198-216 토큰-메이저 페어 = L228-297 전문가별
//!   경로와 누산 순서 동일 계약): gate·up·silu·mul(L249-255·silu_rows
//!   L10-16)·down → 토큰별 e-오름차 가중 누산.
//! - shared(L246-262·L303-313) + sgate sigmoid 게이트 가산(L313-318,
//!   sigmoid = ops.rs L133-135).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-05, RTX 4070 SUPER]
//! (i) 27B 폭(5120) 512e top10 ffn640 sh640 t=4 np=31: route ids EXACT
//! (존 4종: 포화 cnt=1/w=1.0 · 동률 비트동일 w · 근접 ulp 순서 · plain
//! 무스킵) · route-w maxdiff 0.000e0 · out maxdiff 0.000e0 nan=0
//! (순차 누산 미러·-fmad=false 계약의 비트동일 실증 — 임계 3e-4 무한대
//! 여유). (ii-a) 35B-A3B cfg(2048·256e top8 512/512) t=4 np=25: 0.000e0 ·
//! (ii-b) Flash-Next cfg(2560·512e top10 640/640) t=4 np=31: 0.000e0.
//! 음성대조: (a) 라우터 가중치 부호 비트 오염 e=7 → ids_differ=true ·
//! maxdiff 6.317e-2 · (b) 전문가 교차 배정(라우팅 불변 ids_same=true) →
//! maxdiff 9.794e-4 > 3e-4 → NEG-DETECTED(비영 exit + 마커).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 개발기
//! (RTX 4070 SUPER, sm_89)은 정합 호스트일 뿐 — 타이밍 판단 근거 아님.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 — scripts/
//! cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::exl3_cuda_probe::{Rng, f16_to_f32, f32_to_f16, maxdiff_nan};
use crate::rawcuda::moe_cuda::{MoeCuda, MoeDims};
use std::collections::HashMap;

/// 라우팅 가중치 maxdiff 임계(softmax 확률 [0,1] — 비트동일 기대).
const ROUTE_W_THRESH: f32 = 1e-6;
/// moe_ffn 출력 값 maxdiff 임계 — plans/124 §1 GEMV 체인 등급(3e-4).
/// 순차 누산 미러·-fmad=false 계약상 비트동일(0.0) 기대, 임계는 여유.
const MOE_VAL_THRESH: f32 = 3e-4;

// ── core exp_cr 직이식(ops.rs L52-89 — 커널 fn_moe_exp_cr과 동일 DAG) ──
/// f64 FMA 호너 13차 exp — 리터럴·연산 순서까지 ops.rs 원본 그대로
/// (변경 금지: 커널과의 비트동일 계약).
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

/// silu — ops.rs L127-131(x/(1.0+exp_cr(-x))) 그대로.
fn silu_mirror(x: f32) -> f32 {
    x / (1.0 + exp_cr_mirror(-x))
}

/// sigmoid — ops.rs L133-135(1.0/(1.0+exp_cr(-x))) 그대로.
fn sigmoid_mirror(x: f32) -> f32 {
    1.0 / (1.0 + exp_cr_mirror(-x))
}

/// f16 행 선형 내적 — core matmul cpu.rs L91-94 순차 누산 미러
/// (acc += x[i]*scratch[i], 오름차 i — 디양자화 f16→f32 정확 변환).
fn dot16(x: &[f32], w: &[u16], row: usize, n_in: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n_in {
        acc += x[i] * f16_to_f32(w[row * n_in + i]);
    }
    acc
}

/// MoE 픽스처 — 모듈·오라클이 소비하는 동일 바이트(전문가는 별도 생성기).
struct MoeFx {
    dims: MoeDims,
    /// 라우터 f16 [n_exp][n_embd] — 원천: ffn_gate_inp(moe.rs L25).
    route: Vec<u16>,
    /// shared 게이트 라우터 f16 [n_embd] — ffn_gate_inp_sh(L26-28).
    route_sh: Vec<u16>,
    /// shared gate f16 [n_ff_sh][n_embd] — ffn_gate_shexp(L29).
    sh_gate: Vec<u16>,
    /// shared up f16 [n_ff_sh][n_embd] — ffn_up_shexp(L30).
    sh_up: Vec<u16>,
    /// shared down f16 [n_embd][n_ff_sh] — ffn_down_shexp(L31-32).
    sh_down: Vec<u16>,
}

/// 전문가 3중 f16(gate/up [n_ff][n_embd]·down [n_embd][n_ff]) — 시드는
/// 전문가 id 파생(선택과 무결 — 모듈·오라클 동일 바이트 보장).
fn gen_expert16(d: &MoeDims, e: u32, seed: u64) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mk = |salt: u64, n: usize| -> Vec<u16> {
        let mut rng = Rng::new(seed ^ salt ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        (0..n)
            .map(|_| f32_to_f16(((rng.next_f64() * 2.0 - 1.0) * 0.03) as f32))
            .collect()
    };
    let g = mk(0xA1, d.n_ff * d.n_embd);
    let u = mk(0xA2, d.n_ff * d.n_embd);
    let dn = mk(0xA3, d.n_embd * d.n_ff);
    (g, u, dn)
}

/// 라우팅 오라클 — moe.rs L40-70 미러(선택 순서 = 확률 내림·동률 id
/// 오름차·w!=0 스킵). 반환: 토큰별 (id, w) 리스트 + sgate 로짓.
fn route_ref(fx: &MoeFx, xs: &[Vec<f32>]) -> (Vec<Vec<(u32, f32)>>, Vec<f32>) {
    let d = &fx.dims;
    let t = xs.len();
    let mut sel_all = Vec::with_capacity(t);
    let mut sgates = Vec::with_capacity(t);
    for ti in 0..t {
        // 1) 라우팅 — 전 토큰 배치(mm_batch L40-42)·sgate(L43-45).
        let mut logits: Vec<f32> = (0..d.n_expert)
            .map(|e| dot16(&xs[ti], &fx.route, e, d.n_embd))
            .collect();
        sgates.push(dot16(&xs[ti], &fx.route_sh, 0, d.n_embd));
        // 2) 선택 — moe.rs L46-70 그대로(exp만 치환 계약).
        let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut zs = 0.0f32;
        for v in logits.iter_mut() {
            *v = exp_cr_mirror(*v - mx);
            zs += *v;
        }
        for v in logits.iter_mut() {
            *v /= zs;
        }
        let mut idx: Vec<usize> = (0..d.n_expert).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let sel = &idx[..d.n_used];
        let mut wsum: f32 = sel.iter().map(|&e| logits[e]).sum();
        wsum = wsum.max(6.103_515_6e-5);
        let mut row = Vec::with_capacity(d.n_used);
        for &e in sel {
            let w = logits[e] / wsum;
            if w != 0.0 {
                row.push((e as u32, w));
            }
        }
        sel_all.push(row);
    }
    (sel_all, sgates)
}

/// moe_ffn 오라클 — moe.rs L22-320 그룹 경로 미러. 페어는 토큰-메이저·
/// 토큰 내 e 오름차(L198-201 정렬 계약) — 전문가별 경로(L228-297)와
/// 토큰별 누산 순서 동일. experts는 (id → 3중) 캐시(모듈과 동일 바이트).
fn moe_ffn_ref(
    fx: &MoeFx,
    xs: &[Vec<f32>],
    experts: &HashMap<u32, (Vec<u16>, Vec<u16>, Vec<u16>)>,
) -> Vec<Vec<f32>> {
    let d = &fx.dims;
    let t = xs.len();
    let (sel_all, sgates) = route_ref(fx, xs);
    let mut out = vec![vec![0.0f32; d.n_embd]; t];
    // 전문가 — 토큰-메이저 페어(L198-216): gate·up·silu·mul·down 후
    // 토큰에 w 가중 누산(e-오름차 순서).
    for ti in 0..t {
        let mut row = sel_all[ti].clone();
        row.sort_by_key(|&(e, _)| e);
        for &(e, w) in &row {
            let (g16, u16, d16) = &experts[&e];
            // gate·up(동일 입력 x — mm_group 계급 L228-237)·활성화
            // (silu_rows L10-16 + r[i]*=u[i] L249-255)·down(L244-245).
            let mut act = vec![0.0f32; d.n_ff];
            let mut up = vec![0.0f32; d.n_ff];
            for o in 0..d.n_ff {
                act[o] = dot16(&xs[ti], g16, o, d.n_embd);
                up[o] = dot16(&xs[ti], u16, o, d.n_embd);
            }
            for o in 0..d.n_ff {
                act[o] = silu_mirror(act[o]) * up[o];
            }
            let eo = (0..d.n_embd)
                .map(|o| dot16(&act, d16, o, d.n_ff))
                .collect::<Vec<_>>();
            let orow = &mut out[ti];
            for i in 0..d.n_embd {
                orow[i] += w * eo[i]; // L313-318 누산 순서(e-오름차)
            }
        }
    }
    // shared 전문가 — 전 토큰 배치(L246-262·L303-313) + sigmoid 게이트
    // 합산(L313-318: o[i] += sigmoid(sgate)·shout[ti][i]).
    for ti in 0..t {
        let mut act = vec![0.0f32; d.n_ff_sh];
        let mut up = vec![0.0f32; d.n_ff_sh];
        for o in 0..d.n_ff_sh {
            act[o] = dot16(&xs[ti], &fx.sh_gate, o, d.n_embd);
            up[o] = dot16(&xs[ti], &fx.sh_up, o, d.n_embd);
        }
        for o in 0..d.n_ff_sh {
            act[o] = silu_mirror(act[o]) * up[o];
        }
        let shout: Vec<f32> = (0..d.n_embd)
            .map(|o| dot16(&act, &fx.sh_down, o, d.n_ff_sh))
            .collect();
        let sh_w = sigmoid_mirror(sgates[ti]);
        let orow = &mut out[ti];
        for i in 0..d.n_embd {
            orow[i] += sh_w * shout[i];
        }
    }
    out
}

/// 라우터 행 가우기(가변 슬라이스 — 폐규칙 회피용 자유 함수).
fn row_of<'a>(r: &'a mut [u16], e: usize, w: usize) -> &'a mut [u16] {
    &mut r[e * w..(e + 1) * w]
}

/// 픽스처 생성 — 라우팅 존(결정론 시드). 존 부스트는 **직교 원소**
/// (sat=elem0 600.0 · tie/near 공통 부스트=elem1 30.0 · near 판별자=
/// elem2 0.125 · near 1-ulp 범프=elem3 0.5): 존 토큰은 자기 원소만
/// +0.5, 나머지 존 원소는 전부 −0.5로 눌러 미부스트 행이 바닥으로 가게
/// 한다(부스트 행이 전 토큰에 공유되므로 — 첫 구현은 이 충돌로 tie/near
/// 토큰이 sat에 잠식당했다).
/// · sat(포화): 로짓 격차 ≈300 ≫ 103.97 → 나머지 exp_cr 언더플로 0 →
///   w==0 스킵 경로(유지 원소 1개·w=1.0) 고정.
/// · tie(동률): e_tie_a ≡ e_tie_b 동일 바이트 → 동일 로짓 비트 — 정렬
///   안정성(동률 id 오름차) 고정.
/// · near(근접): e_near_b = e_near_a에서 원소3을 f16 1 ulp 상향(원소3
///   =0.5 f16 정규수 — 비트+1 = 다음 표현값) — 근접 경계의 결정론 선택.
/// · plain: 존 원소 전부 −0.5 → 순수 필드 top-n_used(스킵 없음).
fn gen_fixture(d: &MoeDims, seed: u64, t: usize) -> (MoeFx, Vec<Vec<f32>>) {
    let mut rng = Rng::new(seed);
    let mk = |rng: &mut Rng, n: usize, amp: f64| -> Vec<u16> {
        (0..n)
            .map(|_| f32_to_f16(((rng.next_f64() * 2.0 - 1.0) * amp) as f32))
            .collect()
    };
    let mut route = mk(&mut rng, d.n_expert * d.n_embd, 0.02);
    let route_sh = mk(&mut rng, d.n_embd, 0.02);
    let sh_gate = mk(&mut rng, d.n_ff_sh * d.n_embd, 0.03);
    let sh_up = mk(&mut rng, d.n_ff_sh * d.n_embd, 0.03);
    let sh_down = mk(&mut rng, d.n_embd * d.n_ff_sh, 0.05);
    let mut xs: Vec<Vec<f32>> = (0..t)
        .map(|_| {
            (0..d.n_embd)
                .map(|_| ((rng.next_f64() * 2.0 - 1.0) * 0.5) as f32)
                .collect()
        })
        .collect();
    let e_sat = 7 % d.n_expert;
    let e_tie_a = 100 % d.n_expert;
    let e_tie_b = 355 % d.n_expert;
    let e_near_a = 200 % d.n_expert;
    let e_near_b = 201 % d.n_expert;
    {
        // 존 행 덮어쓰기 — 존 원소는 전부 직교: sat=elem0(600) · tie/near
        // 공통 부스트=elem1(30) · near 판별자=elem2(0.125 — near행이 tie행과
        // 다른 로짓을 갖게 하는 변별) · near 1-ulp 범프=elem3(0.5).
        row_of(&mut route, e_sat, d.n_embd)[0] = f32_to_f16(600.0);
        let mut base = vec![0u16; d.n_embd];
        base[1] = f32_to_f16(30.0); // tie/near 공통 부스트
        base[3] = f32_to_f16(0.5); // near ulp 기준점(f16 정규수 — 안전)
        let mut rr = Rng::new(seed ^ 0xE7);
        for v in base.iter_mut().skip(4) {
            *v = f32_to_f16(((rr.next_f64() * 2.0 - 1.0) * 0.02) as f32);
        }
        let mut nearbase = base.clone();
        nearbase[2] = f32_to_f16(0.125); // tie행과의 변별(±0.0625 로짓 차)
        let mut near = nearbase.clone();
        near[3] += 1; // f16 비트 +1 = 다음 표현값(양의 정규수 1 ulp)
        row_of(&mut route, e_tie_a, d.n_embd).copy_from_slice(&base);
        row_of(&mut route, e_tie_b, d.n_embd).copy_from_slice(&base);
        row_of(&mut route, e_near_a, d.n_embd).copy_from_slice(&nearbase);
        row_of(&mut route, e_near_b, d.n_embd).copy_from_slice(&near);
        // 존 토큰 배정 — sat=0·tie=1·near=2·plain=3(t 부족시 뒤쪽부터 생략).
        // 비활성 존 원소는 전부 −0.5로 눌러 미부스트 행이 바닥으로 간다.
        let t_sat = 0;
        let t_tie = if t > 1 { 1 } else { 0 };
        let t_near = if t > 2 { 2 } else { t_tie };
        for x in xs.iter_mut() {
            x[0] = -0.5;
            x[1] = -0.5;
            x[2] = -0.5;
            x[3] = -0.5;
        }
        xs[t_sat][0] = 0.5;
        xs[t_tie][1] = 0.5;
        xs[t_near][1] = 0.5;
        xs[t_near][2] = 0.5; // near행을 tie행 위로(근접 경계 상단 배치)
        xs[t_near][3] = 0.5; // ulp 범프 활성(양수 → e_near_b > e_near_a)
    }
    (
        MoeFx {
            dims: d.clone(),
            route,
            route_sh,
            sh_gate,
            sh_up,
            sh_down,
        },
        xs,
    )
}

/// 픽스처 존 무결성 검증(계기의 계기 — 원장 17호): 오라클 라우팅 결과가
/// 설계된 존을 실제로 담는지 단언. t=4(전 존)일 때만 전체 단언.
fn zones_check(d: &MoeDims, sel: &[Vec<(u32, f32)>], t: usize) -> Result<(), String> {
    let e_sat = 7 % d.n_expert;
    let e_tie_a = 100 % d.n_expert;
    let e_tie_b = 355 % d.n_expert;
    let e_near_a = 200 % d.n_expert;
    let e_near_b = 201 % d.n_expert;
    // sat: 유지 원소 1개(e_sat)·w=1.0.
    let s0 = &sel[0];
    if s0.len() != 1 || s0[0].0 != e_sat as u32 || s0[0].1 != 1.0 {
        return Err(format!(
            "zones: sat 토큰 {s0:?} — (e={e_sat}, w=1.0) 단일 유지 기대"
        ));
    }
    if t < 4 {
        return Ok(());
    }
    // tie: 상위 2가 동일 바이트 가중치(모듈로 래핑 id도 커버 — 낮은 id
    // 먼저: 35B n_expert=256에서 e_tie_b=355%256=99 < e_tie_a=100).
    let (lo, hi) = if e_tie_a < e_tie_b {
        (e_tie_a, e_tie_b)
    } else {
        (e_tie_b, e_tie_a)
    };
    let s1 = &sel[1];
    if s1.len() < 2
        || s1[0].0 != lo as u32
        || s1[1].0 != hi as u32
        || s1[0].1.to_bits() != s1[1].1.to_bits()
    {
        return Err(format!(
            "zones: tie 토큰 상위2 {:?} — ({lo},{hi}) 동일 가중치 비트 기대",
            &s1[..2.min(s1.len())]
        ));
    }
    // near: e_near_b(1 ulp 상향)가 e_near_a보다 먼저(순서 고정).
    let s2 = &sel[2];
    let pa = s2.iter().position(|&(e, _)| e == e_near_a as u32);
    let pb = s2.iter().position(|&(e, _)| e == e_near_b as u32);
    match (pa, pb) {
        (Some(a), Some(b)) if b < a => {}
        _ => {
            return Err(format!(
                "zones: near 토큰 {s2:?} — e_near_b({e_near_b})가 e_near_a({e_near_a})보다 먼저 기대"
            ));
        }
    }
    // plain: 스킵 없이 n_used 유지.
    if sel[3].len() != d.n_used {
        return Err(format!(
            "zones: plain 토큰 유지 {} != n_used {} — 존 원소 누설 의심",
            sel[3].len(),
            d.n_used
        ));
    }
    Ok(())
}

/// f16 → 바이트(리틀 엔디언 행 우선 — 모듈 업로드 형식).
fn u16s_bytes(v: &[u16]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// 전문가 상주: 선택 전문가를 생성·등록(모듈·오라클 동일 바이트 공유).
/// 반환 맵은 오라클이 소비(정상 배정 기준).
fn stage_experts(
    m: &mut MoeCuda,
    d: &MoeDims,
    fx: &MoeFx,
    xs: &[Vec<f32>],
    seed: u64,
) -> Result<HashMap<u32, (Vec<u16>, Vec<u16>, Vec<u16>)>, String> {
    let (sel, _) = route_ref(fx, xs);
    let mut need: Vec<u32> = Vec::new();
    for row in &sel {
        for &(e, _) in row {
            if !need.contains(&e) {
                need.push(e);
            }
        }
    }
    let mut experts = HashMap::new();
    for &e in &need {
        let w = gen_expert16(d, e, seed);
        m.add_expert_f16(
            e as usize,
            &u16s_bytes(&w.0),
            &u16s_bytes(&w.1),
            &u16s_bytes(&w.2),
        )?;
        experts.insert(e, w);
    }
    Ok(experts)
}

/// 1개 형상 체크: 모듈 vs 오라클 — 라우팅 이산 exact + 라우팅 가중치
/// maxdiff + moe_ffn 출력 값 maxdiff. 라인 출력 후
/// (디바이스명, 요약, 통과, 실패 상세) 반환.
fn run_case(
    tag: &str,
    d: &MoeDims,
    seed: u64,
    t: usize,
) -> Result<(String, String, bool, String), String> {
    let (fx, xs) = gen_fixture(d, seed, t);
    let (oracle_sel, _) = route_ref(&fx, &xs);
    zones_check(d, &oracle_sel, t)?;
    let mut m = MoeCuda::new(d.clone())?;
    let dev = m.device_name().to_string();
    m.set_router_f16(&u16s_bytes(&fx.route), &u16s_bytes(&fx.route_sh))?;
    m.set_shared_f16(
        &u16s_bytes(&fx.sh_gate),
        &u16s_bytes(&fx.sh_up),
        &u16s_bytes(&fx.sh_down),
    )?;
    let experts = stage_experts(&mut m, d, &fx, &xs, seed)?;
    let oracle_out = moe_ffn_ref(&fx, &xs, &experts);

    // (1) 라우팅 이산 — 선택 순열 exact(순서 포함 — 안정 정렬 계약).
    let got_sel = m.moe_route(&xs)?;
    let mut ids_ok = got_sel.len() == oracle_sel.len();
    for (ti, (g, w)) in got_sel.iter().zip(oracle_sel.iter()).enumerate() {
        if g.len() != w.len() || g.iter().ne(w.iter()) {
            ids_ok = false;
            eprintln!(
                "  route mismatch t{ti}: got {:?} want {:?}",
                g.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
                w.iter().map(|&(e, _)| e).collect::<Vec<_>>()
            );
        }
    }
    // (2) 라우팅 가중치 maxdiff.
    let mut wmd = 0.0f32;
    for (g, w) in got_sel.iter().zip(oracle_sel.iter()) {
        for (a, b) in g.iter().zip(w.iter()) {
            wmd = wmd.max((a.1 - b.1).abs());
        }
    }
    // (3) moe_ffn 종단 값.
    let got_out = m.moe_ffn(&xs)?;
    let mut md = 0.0f32;
    let mut nan = 0usize;
    for (g, w) in got_out.iter().zip(oracle_out.iter()) {
        let (m1, n1) = maxdiff_nan(g, w);
        md = md.max(m1);
        nan += n1;
    }
    let np: usize = oracle_sel.iter().map(|r| r.len()).sum();
    let pass = ids_ok && wmd <= ROUTE_W_THRESH && md <= MOE_VAL_THRESH && nan == 0;
    println!(
        "device: {dev} | fn-cuda-moe {tag}: n_embd={} {}e top{} ffn{} sh{} t={} np={} | route ids {} (w maxdiff={wmd:.3e}) | out maxdiff={md:.3e} nan={nan} | {}",
        d.n_embd,
        d.n_expert,
        d.n_used,
        d.n_ff,
        d.n_ff_sh,
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

/// config.json(text_config) → MoE형상 — JParser(exl3_cuda 최소 JSON) 재사용.
fn moe_dims_from_config(path: &str) -> Result<MoeDims, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let v = JParser { b: &b, p: 0 }.parse()?;
    let tc = v.get("text_config").ok_or("config: text_config 없음")?;
    let g = |k: &str| -> Result<f64, String> {
        tc.get(k)
            .and_then(JVal::as_f64)
            .ok_or_else(|| format!("config: text_config.{k} 없음"))
    };
    Ok(MoeDims {
        n_embd: g("hidden_size")? as usize,
        n_expert: g("num_experts")? as usize,
        n_used: g("num_experts_per_tok")? as usize,
        n_ff: g("moe_intermediate_size")? as usize,
        n_ff_sh: g("shared_expert_intermediate_size")? as usize,
    })
}

/// fn-cuda-moe — FNE MoE FFN 정합 프로브.
/// (i) 27B 폭 합성(n_embd=5120 = Qwen3.8-27B config hidden_size 실측
/// 2026-10-05 + Flash-Next MoE 헤드 512e top10 ffn640 sh640 — 과제 계약
/// "MoE dims from config") 존 4종(plain/포화/동률/근접) — 라우팅 이산
/// exact가 주 판정.
/// (ii-a) 35B-A3B config.json 실측 형상 · (ii-b) Flash-Next config.json
/// (과제 지정 경로) — 종단 값 maxdiff 문서화.
/// 하나라도 FAIL이면 Err(→ CLI 비영).
pub fn cuda_moe_check() -> Result<String, String> {
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (i) 27B-config 합성 — 라우팅 이산 계약의 주 무대.
    let d27 = MoeDims {
        n_embd: 5120,
        n_expert: 512,
        n_used: 10,
        n_ff: 640,
        n_ff_sh: 640,
    };
    let (dev, s, p, f) = run_case("(i) 27B-width zones", &d27, 0x5EED_F00D_0000_0001, 4)?;
    report.push_str(&s);
    if !p {
        fails.push(f);
    }

    // (ii) 실측 MoE 형상 — config.json이 단일 진실(누락 시 비영: 실 파일
    // 증명이 목적 — fn-inv 계급과 동일 계약).
    let d35 = moe_dims_from_config("D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw/config.json")?;
    let (_, s, p, f) = run_case("(ii-a) 35B-A3B cfg", &d35, 0x5EED_F00D_0000_0002, 4)?;
    report.push_str(" · ");
    report.push_str(&s);
    if !p {
        fails.push(f);
    }

    let dfn = moe_dims_from_config("D:/models/Qwen3.8-Flash-Next-exl3-5.05bpw/config.json")?;
    let (_, s, p, f) = run_case("(ii-b) Flash-Next cfg", &dfn, 0x5EED_F00D_0000_0003, 4)?;
    report.push_str(" · ");
    report.push_str(&s);
    if !p {
        fails.push(f);
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | fn-cuda-moe {report} | ALL PASS"))
    } else {
        Err(format!("fn-cuda-moe 실패 — {}", fails.join(", ")))
    }
}

/// fn-cuda-moe-neg — 음성대조 2계급(원장 17호: 계기 자체 검증):
/// (a) 라우팅 가중치 오염: 토큰0 최상위 전문가 행의 원소0 부호 비트
/// 반전(라우터 f16 바이트 오염 — q4-neg 계급의 입력 데이터 오염) →
/// 라우팅 선택 변화·출력 이탈이 검출되어야 한다(이산·값 양측).
/// (b) 전문가 배정 오류: 토큰0 상위 2전문가의 가중치를 뒤바꿔 등록
/// (라우팅은 불변 — ids 동일함을 먼저 확인해 결함 계급을 고정) →
/// 값 이탈 검출. 양쪽 모두 검출 시 NEG-DETECTED 마커와 함께 Err.
pub fn cuda_moe_negative_check() -> Result<String, String> {
    let d = MoeDims {
        n_embd: 2560,
        n_expert: 512,
        n_used: 10,
        n_ff: 640,
        n_ff_sh: 640,
    };
    let seed = 0x5EED_F00D_0000_00AA;
    let (fx, xs) = gen_fixture(&d, seed, 2);
    let (oracle_sel, _) = route_ref(&fx, &xs);

    // (a) 라우팅 가중치 오염 — 토큰0 최상위 전문가 행 원소0 부호 반전.
    let e0 = oracle_sel[0][0].0 as usize;
    let mut rcor = fx.route.clone();
    rcor[e0 * d.n_embd] ^= 0x8000; // f16 부호 비트 — 로짓 부호 반전
    let fxc = MoeFx {
        dims: d.clone(),
        route: rcor,
        route_sh: fx.route_sh.clone(),
        sh_gate: fx.sh_gate.clone(),
        sh_up: fx.sh_up.clone(),
        sh_down: fx.sh_down.clone(),
    };
    let mut m = MoeCuda::new(d.clone())?;
    let dev = m.device_name().to_string();
    m.set_router_f16(&u16s_bytes(&fxc.route), &u16s_bytes(&fxc.route_sh))?;
    m.set_shared_f16(
        &u16s_bytes(&fxc.sh_gate),
        &u16s_bytes(&fxc.sh_up),
        &u16s_bytes(&fxc.sh_down),
    )?;
    // 오염 라우팅이 선택할 수 있는 전문가까지 합집합 상주(값 검출 보장).
    let mut experts = stage_experts(&mut m, &d, &fx, &xs, seed)?;
    let (sel_cor, _) = route_ref(&fxc, &xs);
    for row in &sel_cor {
        for &(e, _) in row {
            if !experts.contains_key(&e) {
                let w = gen_expert16(&d, e, seed);
                m.add_expert_f16(
                    e as usize,
                    &u16s_bytes(&w.0),
                    &u16s_bytes(&w.1),
                    &u16s_bytes(&w.2),
                )?;
                experts.insert(e, w);
            }
        }
    }
    // 기준은 청정 가중치 오라클 — 이산 선택 변화 또는 값 이탈로 검출.
    let got_sel = m.moe_route(&xs)?;
    let ids_differ = got_sel.len() != oracle_sel.len()
        || got_sel
            .iter()
            .zip(oracle_sel.iter())
            .any(|(g, w)| g.len() != w.len() || g.iter().ne(w.iter()));
    let oracle_out = moe_ffn_ref(&fx, &xs, &experts);
    let got = m.moe_ffn(&xs)?;
    let mut md_a = 0.0f32;
    for (g, w) in got.iter().zip(oracle_out.iter()) {
        let (m1, _) = maxdiff_nan(g, w);
        md_a = md_a.max(m1);
    }
    let det_a = ids_differ || md_a > MOE_VAL_THRESH;
    println!(
        "device: {dev} | fn-cuda-moe-neg (a) router-weight corruption e={e0}: ids_differ={ids_differ} maxdiff={md_a:.3e} | {}",
        if det_a {
            "FAIL(expected)"
        } else {
            "NOT-DETECTED"
        }
    );

    // (b) 전문가 배정 오류 — 유지 원소 ≥2인 토큰(포화 토큰0은 cnt=1이므
    // 로 t-1 — 근접 존 토큰)에서 **라우팅 가중치가 다른** 두 전문가를
    // 골라 가중치 교차 등록(동률 tie 쌍은 가중치가 같아 교환이 거의 상쇄
    // 되므로 제외 — 결함 계급 검출 보장).
    let tb = oracle_sel
        .iter()
        .rposition(|r| r.len() >= 2)
        .ok_or("neg-b: 유지 원소 ≥2 토큰 없음 — 픽스처 결함")?;
    let row = &oracle_sel[tb];
    let ea = row[0].0;
    let w0 = row[0].1;
    // 교환 쌍은 라우팅 가중치가 **실질적으로** 다른 짝(상대 5% 이상 —
    // ulp 형제·동률 tie는 교환이 거의 상쇄되어 검출 불능이 된다).
    let eb = row
        .iter()
        .skip(1)
        .find(|&&(_, w)| (w - w0).abs() > 0.05 * w0)
        .map(|&(e, _)| e)
        .ok_or("neg-b: 가중치 실질 상이한 짝 없음 — 픽스처 결함")?;
    let mut m2 = MoeCuda::new(d.clone())?;
    m2.set_router_f16(&u16s_bytes(&fx.route), &u16s_bytes(&fx.route_sh))?;
    m2.set_shared_f16(
        &u16s_bytes(&fx.sh_gate),
        &u16s_bytes(&fx.sh_up),
        &u16s_bytes(&fx.sh_down),
    )?;
    let mut need: Vec<u32> = Vec::new();
    for row in &oracle_sel {
        for &(e, _) in row {
            if !need.contains(&e) {
                need.push(e);
            }
        }
    }
    for &e in &need {
        let w = if e == ea {
            gen_expert16(&d, eb, seed) // 배정 오류: ea 슬롯에 eb 가중치
        } else if e == eb {
            gen_expert16(&d, ea, seed)
        } else {
            gen_expert16(&d, e, seed)
        };
        m2.add_expert_f16(
            e as usize,
            &u16s_bytes(&w.0),
            &u16s_bytes(&w.1),
            &u16s_bytes(&w.2),
        )?;
    }
    // 라우팅은 불변이어야 한다(결함 계급 고정: 배정 오류 ≠ 라우팅 오염).
    let got_sel2 = m2.moe_route(&xs)?;
    let ids_same = got_sel2.len() == oracle_sel.len()
        && got_sel2
            .iter()
            .zip(oracle_sel.iter())
            .all(|(g, w)| g.len() == w.len() && g.iter().eq(w.iter()));
    // 오라클 기준은 정상 배정 가중치 — 청정 쌍으로 계산.
    let clean: HashMap<u32, (Vec<u16>, Vec<u16>, Vec<u16>)> = need
        .iter()
        .map(|&e| (e, gen_expert16(&d, e, seed)))
        .collect();
    let oracle2 = moe_ffn_ref(&fx, &xs, &clean);
    let got2 = m2.moe_ffn(&xs)?;
    let mut md_b = 0.0f32;
    for (g, w) in got2.iter().zip(oracle2.iter()) {
        let (m1, _) = maxdiff_nan(g, w);
        md_b = md_b.max(m1);
    }
    let det_b = ids_same && md_b > MOE_VAL_THRESH;
    println!(
        "device: {dev} | fn-cuda-moe-neg (b) swapped expert assignment ({ea}<->{eb}): ids_same={ids_same} maxdiff={md_b:.3e} | {}",
        if det_b {
            "FAIL(expected)"
        } else {
            "NOT-DETECTED"
        }
    );

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) router corruption ids_differ={ids_differ} maxdiff={md_a:.3e} (b) swapped-assignment ids_same={ids_same} maxdiff={md_b:.3e} > {MOE_VAL_THRESH} — 검증계기 정상(라우팅 가중치 오염·전문가 배정 오류 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={md_a:.3e} (b)={md_b:.3e} — 검증계기 결함: 오염이 탐지되지 않음"
        ))
    }
}
