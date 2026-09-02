//! ew(elementwise) 커널 — 층 전체 GPU 상주(P2-4)의 요소연산 세트.
//!
//! 수치 계약: **CPU 참조(ops.rs·stages)와 동일한 연산 순서** — 토큰 스트림
//! 완전일치가 통과 조건. norm류는 f64 순차 누산(CPU rms_norm/l2_norm과 동일),
//! 나머지는 f32 그대로. ADR-0005: mul_add 금지 — 순수 mul+add.
//!
//! - 스칼라 f32 인수(eps, hc 등)는 params 1원소 텐서로 전달 (gdn_kernel.rs 근거).
//! - 차원 인수는 런타임 usize (gdn_ar 검증 패턴) — comptime 전개 폭발 방지.
//! - 캐스트는 `f64::cast_from`/`f32::cast_from` (Cast 트레이트, cubecl-core
//!   frontend/element/cast.rs). `as` 캐스트는 ADR-0011 오컴파일 회피.
//! - 분기 없는 softplus: x≤80에서 log1p(exp(x))가 f32에서 x와 비트 동일
//!   (반올림 여유 ~1e-6 vs 오차 ~e^{x-2x}) — x>80은 미발생 영역(가드용 clamp).

use cubecl::prelude::*;

// ─── 요소별 활성화 (ABSOLUTE_POS_X 병렬 — 요소 독립, 순서 무관) ───

/// in-place silu: v ← v/(1+e^(−v)).
#[cube(launch_unchecked)]
pub fn ew_silu(t: &mut Tensor<f32>, n: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        let v = t[j];
        t[j] = v / (1.0 + (-v).exp());
    }
}

/// in-place: v ← silu(v / div). hc 저랭크 lo — CPU silu(v/hc) 나눗셈 선행.
#[cube(launch_unchecked)]
pub fn ew_silu_div(t: &mut Tensor<f32>, params: &Tensor<f32>, n: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        let v = t[j] / params[0];
        t[j] = v / (1.0 + (-v).exp());
    }
}

/// in-place sigmoid.
#[cube(launch_unchecked)]
pub fn ew_sigmoid(t: &mut Tensor<f32>, n: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        let v = t[j];
        t[j] = 1.0 / (1.0 + (-v).exp());
    }
}

/// GLU: out[j] = silu(g[j])·u[j] (MoE·shared gate/up 결합).
#[cube(launch_unchecked)]
pub fn ew_silu_mul(g: &Tensor<f32>, u: &Tensor<f32>, out: &mut Tensor<f32>, n: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        let v = g[j];
        out[j] = (v / (1.0 + (-v).exp())) * u[j];
    }
}

// ─── norm류 (큐브당 1행, 유닛 0 순차 — CPU f64 누산과 비트 단위 동일) ───

/// 행별 RMSNorm: out[row·n..] = x[row·n..]·scale·w[wb..], wb = (row % w_reps)·n.
/// CPU ops::rms_norm 순서: f64 순차 누산 → mean+eps → sqrt(f64) → f32 cast →
/// 1/x → (v·scale)·γ.
#[cube(launch_unchecked)]
pub fn rms_rows_part(
    x: &Tensor<f32>,
    part: &mut Tensor<f64>,
    n: usize,
    #[comptime] seg: usize,
) {
    // 32유닛 세그먼트 합 — 유닛 u가 [u·chunk, (u+1)·chunk) f64 순차 누산.
    // CPU ops.rs sq_sum과 동일 순서 (비트 계약, 2026-09-02 P0).
    let row = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u >= seg {
        terminate!();
    }
    let chunk = n.div_ceil(seg);
    let lo = u * chunk;
    if lo >= n {
        part[row * seg + u] = 0.0;
        terminate!();
    }
    let hi = (lo + chunk).min(n);
    let xb = row * n;
    let mut acc = 0.0f64;
    let mut i = lo;
    while i < hi {
        let d = f64::cast_from(x[xb + i]);
        acc += d * d;
        i += 1;
    }
    part[row * seg + u] = acc;
}

/// rms 마무리: 32 세그먼트 f64 부분합을 순차 결합 → 스케일 적용.
/// 유닛 0만 실행 — 결합 체인 32항(짧음) + 스케일 루프는 반복 독립.
#[cube(launch_unchecked)]
pub fn rms_rows_finish(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    part: &Tensor<f64>,
    out: &mut Tensor<f32>,
    params: &Tensor<f32>, // [eps]
    n: usize,
    w_reps: usize,
    #[comptime] seg: usize,
) {
    let row = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u >= seg {
        terminate!();
    }
    // 결합(유닛 0) 후 스케일은 전 유닛 분담 — 원소별이라 순서 무관
    // (280µs 직렬 루프가 병목, 2026-09-02 P2 rocprof).
    let eps = f64::cast_from(params[0]);
    let mut sum = 0.0f64;
    let chunk = n.div_ceil(seg);
    let mut u2 = 0usize;
    while u2 < seg {
        if u2 * chunk < n {
            sum += part[row * seg + u2];
        }
        u2 += 1;
    }
    let len = f64::cast_from(n as u32);
    let scale32 = f32::cast_from((sum / len + eps).sqrt());
    let inv = 1.0f32 / scale32;
    let xb = row * n;
    let wb = (row % w_reps) * n;
    // 스케일 세그먼트: 유닛 u가 [u·schunk, (u+1)·schunk) 분담.
    let schunk = n.div_ceil(seg);
    let lo = u * schunk;
    if lo < n {
        let hi = (lo + schunk).min(n);
        let mut i = lo;
        while i < hi {
            out[xb + i] = x[xb + i] * inv * w[wb + i];
            i += 1;
        }
    }
}

/// norm_gated (GDN 출력): out[row·d..] = rms(o[row·d..])·σ(z[row·d..]).
/// w 반복 = 헤드(dt_rank). CPU gdn.rs 141-151 순서 동일.
#[cube(launch_unchecked)]
pub fn norm_gated_rows(
    o: &Tensor<f32>,
    z: &Tensor<f32>,
    w: &Tensor<f32>,
    out: &mut Tensor<f32>,
    params: &Tensor<f32>, // [eps]
    d: usize,
    n_h: usize,
) {
    let row = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let xb = row * d;
    let wb = (row % n_h) * d;
    let eps = f64::cast_from(params[0]);
    let mut sum = 0.0f64;
    {
        let chunk = d.div_ceil(32);
        for sg in 0..32 {
            let lo = sg * chunk;
            if lo >= d {
                break;
            }
            let hi = (lo + chunk).min(d);
            let mut part = 0.0f64;
            for i in lo..hi {
                let dv = f64::cast_from(o[xb + i]);
                part += dv * dv;
            }
            sum += part;
        }
    }
    let len = f64::cast_from(d as u32);
    let scale32 = f32::cast_from((sum / len + eps).sqrt());
    let inv = 1.0f32 / scale32;
    for i in 0..d {
        let nrm = o[xb + i] * inv * w[wb + i];
        let zz = z[xb + i];
        out[xb + i] = nrm * (1.0 / (1.0 + (-zz).exp()));
    }
}

/// norm_gated silu 게이트 변형 (qwen35): out[row·d..] = rms(o[row·d..])·silu(z[row·d..]).
/// qwen4exp(σ)과의 유일 차이 — CPU model/layers.rs 178-193 순서 동일.
#[cube(launch_unchecked)]
pub fn norm_gated_rows_silu(
    o: &Tensor<f32>,
    z: &Tensor<f32>,
    w: &Tensor<f32>,
    out: &mut Tensor<f32>,
    params: &Tensor<f32>, // [eps]
    d: usize,
    n_h: usize,
) {
    let row = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let xb = row * d;
    let wb = (row % n_h) * d;
    let eps = f64::cast_from(params[0]);
    let mut sum = 0.0f64;
    {
        let chunk = d.div_ceil(32);
        for sg in 0..32 {
            let lo = sg * chunk;
            if lo >= d {
                break;
            }
            let hi = (lo + chunk).min(d);
            let mut part = 0.0f64;
            for i in lo..hi {
                let dv = f64::cast_from(o[xb + i]);
                part += dv * dv;
            }
            sum += part;
        }
    }
    let len = f64::cast_from(d as u32);
    let scale32 = f32::cast_from((sum / len + eps).sqrt());
    let inv = 1.0f32 / scale32;
    for i in 0..d {
        let nrm = o[xb + i] * inv * w[wb + i];
        let zz = z[xb + i];
        out[xb + i] = nrm * (zz / (1.0 + (-zz).exp()));
    }
}

/// 행별 in-place L2 norm (GDN q/k 헤드): x[row·d..] 정규화.
/// CPU ops::l2_norm: Σx² f64 → sqrt → f32 cast → .max(eps) → 1/x.
#[cube(launch_unchecked)]
pub fn l2_rows(x: &mut Tensor<f32>, params: &Tensor<f32>, d: usize) {
    let row = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let xb = row * d;
    let mut sum = 0.0f64;
    for i in 0..d {
        let v = f64::cast_from(x[xb + i]);
        sum += v * v;
    }
    let scale32 = f32::cast_from(sum.sqrt());
    let inv = 1.0f32 / scale32.max(params[0]);
    for i in 0..d {
        x[xb + i] *= inv;
    }
}

// ─── hyper-connection ───

/// hc 게이트 적용 + 스트림 평균: out[t·n+i] = (Σ_s xn·σ(gate)) / hc.
/// 그리드 (n블록, t) × CUBE_DIM 64 — 유닛이 i 1개 담당, s 루프 순차.
/// CPU hc.rs 62-75와 동일 (곱 → 순차 가산 → hc 나눗셈). params: [hc].
#[cube(launch_unchecked)]
pub fn hc_gate_mean(
    xn: &Tensor<f32>,
    gate: &Tensor<f32>,
    out: &mut Tensor<f32>,
    params: &Tensor<f32>,
    n: usize,
    hc: usize,
) {
    let t = CUBE_POS_Y as usize;
    let i = CUBE_POS_X as usize * (CUBE_DIM_X as usize) + UNIT_POS_X as usize;
    if i < n {
        let mut acc = 0.0f32;
        for s in 0..hc {
            let g = gate[t * hc * n + s * n + i];
            acc += xn[t * hc * n + s * n + i] * (1.0 / (1.0 + (-g).exp()));
        }
        out[t * n + i] = acc / params[0];
    }
}

/// hc_combine: res += out·(2·σ(inj/hc)) — layers.rs 272-282와 동일 순서.
/// 레이아웃: res/xn [t][hc·n], out [t][n], inj [t][hc]. 요소 독립 병렬.
#[cube(launch_unchecked)]
pub fn hc_combine(
    res: &mut Tensor<f32>,
    out: &Tensor<f32>,
    inj: &Tensor<f32>,
    params: &Tensor<f32>, // [hc]
    n: usize,
    hc: usize,
    total: usize,
) {
    let j = ABSOLUTE_POS_X as usize;
    if j < total {
        let nhc = hc * n;
        let t = j / nhc;
        let r = j % nhc;
        let s = r / n;
        let i = r % n;
        let isig = 1.0 / (1.0 + (-(inj[t * hc + s] / params[0])).exp());
        res[j] += out[t * n + i] * (2.0 * isig);
    }
}

// ─── GDN ───

/// β/e^g 사전 계산: bg[h·2] = σ(b[h]), bg[h·2+1] = exp(softplus(a[h]+dtb[h])·sa[h]).
/// CPU gdn.rs 53-60 + 117 순서 동일. 유닛 = h.
/// softplus 분기 회피: min(x,80) 클램프 + log1p(exp(x)) — x≤80에서 CPU
/// if(x>20){x}와 f32 비트 동일 (상세는 파일 헤더 주석).
#[cube(launch_unchecked)]
pub fn gdn_beta_g(
    b: &Tensor<f32>,
    a: &Tensor<f32>,
    dtb: &Tensor<f32>,
    sa: &Tensor<f32>,
    bg: &mut Tensor<f32>,
    n_h: usize,
) {
    let h = ABSOLUTE_POS_X as usize;
    if h < n_h {
        let bv = b[h];
        bg[h * 2] = 1.0 / (1.0 + (-bv).exp());
        let x = (a[h] + dtb[h]).min(80.0);
        let sp = x.exp().log1p();
        bg[h * 2 + 1] = (sp * sa[h]).exp();
    }
}

/// conv1d + ring shift + silu: 큐브당 1채널, t 순차 (상태 의존).
/// CPU gdn.rs 68-90과 동일 순서: newest tap 먼저, j=0..k-2 더하고, silu 후
/// 링 시프트, 신규 토큰 링 끝 저장. out은 qkv 동일 레이아웃(분할은 호출부).
#[cube(launch_unchecked)]
pub fn gdn_conv(
    qkv: &Tensor<f32>,
    cw: &Tensor<f32>,          // [ch][k]
    state: &mut Tensor<f32>,   // [(k-1)][ch] ring
    out: &mut Tensor<f32>,     // [t][ch] — silu 적용 결과
    ch: usize,
    k: usize,
    t_len: usize,
) {
    let c = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    for tt in 0..t_len {
        let xb = tt * ch + c;
        let mut sum = cw[c * k + (k - 1)] * qkv[xb];
        for j in 0..(k - 1) {
            sum += cw[c * k + j] * state[j * ch + c];
        }
        let oc = sum / (1.0 + (-sum).exp());
        for j in 0..(k - 2) {
            state[j * ch + c] = state[(j + 1) * ch + c];
        }
        state[(k - 2) * ch + c] = qkv[xb];
        out[xb] = oc;
    }
}

// ─── MoE ───

/// route top-k: softmax(n_exp) + 반복 argmax k회 + 선택 확률 정규화.
/// CPU moe.rs 40-61과 동일 — 안정 정렬(동률 낮은 인덱스) ≡ 반복 argmax
/// (strict >, 첫 승). w=0 행은 소비측에서 0·y 기여(CPU skip과 수치 동일).
/// 출력: ids [t][k] u32, wt [t][k] f32 — 모두 GPU 잔류 (readback 없음).
#[cube(launch_unchecked)]
pub fn moe_top10(
    route: &mut Tensor<f32>, // [t][n_exp] in-place softmax 확률로 전환
    ids: &mut Tensor<u32>,   // [t][k]
    wt: &mut Tensor<f32>,    // [t][k]
    n_exp: usize,
    k_sel: usize,
) {
    let ti = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let rb = ti * n_exp;
    // max (순차 fold — CPU fold(NEG_INF, max)와 동일)
    let mut mx = route[rb];
    for i in 1..n_exp {
        let v = route[rb + i];
        if v > mx {
            mx = v;
        }
    }
    // exp → 확률 (두 패스 — CPU와 동일)
    let mut zs = 0.0f32;
    for i in 0..n_exp {
        let e = (route[rb + i] - mx).exp();
        route[rb + i] = e;
        zs += e;
    }
    for i in 0..n_exp {
        route[rb + i] /= zs;
    }
    // 선택 — 내림차순 (wsum도 이 순서로 누산)
    let mut wsum = 0.0f32;
    for s in 0..k_sel {
        let mut bi = 0;
        let mut bv = route[rb];
        for i in 1..n_exp {
            let v = route[rb + i];
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        ids[ti * k_sel + s] = bi as u32;
        wt[ti * k_sel + s] = bv;
        wsum += bv;
        route[rb + bi] = -1.0;
    }
    wsum = wsum.max(6.103515625e-5);
    for s in 0..k_sel {
        wt[ti * k_sel + s] /= wsum;
    }
}

// ─── RoPE / QSA 인덱서 ───

/// NEOX 페어링 RoPE — ops::rope_head와 동일 수식, cos/sin은 호스트가
/// f32 powf/cos/sin으로 예산산해 테이블 전달 (비트 동일 — 곱셈 순서 보존).
/// pos = (row / rows_per_tok)·pos_mul + pos_base. q/k·블록키 공용.
/// cs 레이아웃: [pos_max][half][2] (cos,sin 인터리브).
#[cube(launch_unchecked)]
pub fn rope_apply(
    x: &mut Tensor<f32>, // [rows][stride] — 대상 헤드 평면
    cs: &Tensor<f32>,
    pos_base: usize,
    rows_per_tok: usize,
    pos_mul: usize,
    stride: usize,
    half: usize,
) {
    let row = CUBE_POS_Y as usize;
    let p = CUBE_POS_X as usize * (CUBE_DIM_X as usize) + UNIT_POS_X as usize;
    if p < half {
        let pos = (row / rows_per_tok) * pos_mul + pos_base;
        let c = cs[pos * half * 2 + p * 2];
        let s = cs[pos * half * 2 + p * 2 + 1];
        let b = row * stride;
        let x0 = x[b + p];
        let x1 = x[b + p + half];
        x[b + p] = x0 * c - x1 * s;
        x[b + p + half] = x0 * s + x1 * c;
    }
}

/// 인덱서 블록키 풀링: out[b·dim+i] = (Σ_j idx_cache[(fb+b)·r+j][i]) / r.
/// CPU qsa.rs 98-108 순서 동일 (j 순차 가산 → r 나눗셈). 유닛 = i.
#[cube(launch_unchecked)]
pub fn idx_pool(
    idx_cache: &Tensor<f32>, // [pos][dim]
    out: &mut Tensor<f32>,   // [신규블록][dim]
    first_block: usize,
    dim: usize,
    r: usize,
) {
    let b = CUBE_POS_X as usize;
    let i = UNIT_POS_X as usize;
    if i < dim {
        let mut acc = 0.0f32;
        for j in 0..r {
            acc += idx_cache[((first_block + b) * r + j) * dim + i];
        }
        out[b * dim + i] = acc / f32::cast_from(r as u32);
    }
}

/// 인덱서 스코어: scores[b] = Σ_h ReLU(qr[h]·bk[b]) — CPU qsa.rs 126-138
/// 순서 동일 (h 외부·i 내부 순차, dot>0만 가산). 큐브 = 블록, 유닛 0.
#[cube(launch_unchecked)]
pub fn idx_scores(
    qr: &Tensor<f32>,     // [idx_heads][dim]
    bk: &Tensor<f32>,     // [n_blocks][dim]
    scores: &mut Tensor<f32>, // [n_blocks]
    idx_heads: usize,
    dim: usize,
) {
    let b = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let mut bs = 0.0f32;
    for h in 0..idx_heads {
        let mut dot = 0.0f32;
        for i in 0..dim {
            dot += qr[h * dim + i] * bk[b * dim + i];
        }
        if dot > 0.0 {
            bs += dot;
        }
    }
    scores[b] = bs;
}

/// in-place 스칼라 곱: v ← v·s (GDN q·scale 등 — CPU는 호스트 사전곱과 동일).
#[cube(launch_unchecked)]
pub fn ew_scale(t: &mut Tensor<f32>, params: &Tensor<f32>, n: usize) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        t[j] *= params[0];
    }
}

/// 행 단위 복사: dst[dst_off + i] = src[src_off + i] (n 요소).
/// QSA kv/idx 캐시 append, 블록키 이어붙이기 — 캐시 상주 경로의 기본 부품.
#[cube(launch_unchecked)]
pub fn copy_rows(
    src: &Tensor<f32>,
    dst: &mut Tensor<f32>,
    src_off: usize,
    dst_off: usize,
    n: usize,
) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        dst[dst_off + j] = src[src_off + j];
    }
}

/// MoE 결합: out[i] = Σ_e wt[e]·ys[e·n+i] — CPU moe.rs 265-274 순서 동일
/// (전문가 가중 합 후 shared 가산은 호출부에서 별도 add).
#[cube(launch_unchecked)]
pub fn moe_weighted_sum(
    ys: &Tensor<f32>,     // [k][n]
    wt: &Tensor<f32>,     // [k]
    out: &mut Tensor<f32>, // [n]
    k: usize,
    n: usize,
) {
    let i = CUBE_POS_X as usize * (CUBE_DIM_X as usize) + UNIT_POS_X as usize;
    if i < n {
        let mut acc = 0.0f32;
        for e in 0..k {
            acc += wt[e] * ys[e * n + i];
        }
        out[i] = acc;
    }
}

/// y[i] += x[i]·s (s는 1원소 버퍼 — MoE shared 가산: σ(sgate)·shout).
/// CPU moe.rs 260-265 순서 동일.
#[cube(launch_unchecked)]
pub fn axpy_scaled(
    y: &mut Tensor<f32>,
    x: &Tensor<f32>,
    s: &Tensor<f32>,
    n: usize,
) {
    let j = ABSOLUTE_POS_X as usize;
    if j < n {
        y[j] += x[j] * s[0];
    }
}

/// qwen35 어텐션 q 프리페어: 헤드별 rms(정준 32-세그)·rope(cs 테이블)·
/// q‖gate 인터리브 기록 — CPU layers.rs norm+rope 순서 동일 (비트 계약).
#[cube(launch_unchecked)]
pub fn attn_q_prep(
    q: &Tensor<f32>,
    w: &Tensor<f32>,
    cs: &Tensor<f32>,
    out: &mut Tensor<f32>,
    params: &Tensor<f32>, // [eps]
    hd: usize,
    pos: usize,
    half: usize,
) {
    let h = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let eps = f64::cast_from(params[0]);
    let qb = h * 2 * hd;
    let mut sum = 0.0f64;
    let chunk = hd.div_ceil(32);
    for sg in 0..32 {
        let lo = sg * chunk;
        if lo >= hd {
            break;
        }
        let hi = (lo + chunk).min(hd);
        let mut part = 0.0f64;
        for i in lo..hi {
            let d = f64::cast_from(q[qb + i]);
            part += d * d;
        }
        sum += part;
    }
    let scale32 = f32::cast_from((sum / f64::cast_from(hd as u32) + eps).sqrt());
    let inv = 1.0f32 / scale32;
    for i in 0..hd {
        out[qb + i] = q[qb + i] * inv * w[i];
    }
    for pp in 0..half {
        let c = cs[pos * half * 2 + pp * 2];
        let si = cs[pos * half * 2 + pp * 2 + 1];
        let x0 = out[qb + pp];
        let x1 = out[qb + pp + half];
        out[qb + pp] = x0 * c - x1 * si;
        out[qb + pp + half] = x0 * si + x1 * c;
    }
    for i in 0..hd {
        out[qb + hd + i] = q[qb + hd + i];
    }
}


/// qwen35 어텐션 k 프리페어: kv-헤드별 rms·rope → 캐시 append(pos 위치).
#[cube(launch_unchecked)]
pub fn attn_k_prep(
    k: &Tensor<f32>,
    w: &Tensor<f32>,
    cs: &Tensor<f32>,
    cache: &mut Tensor<f32>,
    params: &Tensor<f32>, // [eps]
    hd: usize,
    pos: usize,
    n_kv: usize,
    half: usize,
) {
    let h = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u != 0 {
        terminate!();
    }
    let eps = f64::cast_from(params[0]);
    let kb = h * hd;
    let cb = pos * n_kv * hd + h * hd;
    let mut sum = 0.0f64;
    let chunk = hd.div_ceil(32);
    for sg in 0..32 {
        let lo = sg * chunk;
        if lo >= hd {
            break;
        }
        let hi = (lo + chunk).min(hd);
        let mut part = 0.0f64;
        for i in lo..hi {
            let d = f64::cast_from(k[kb + i]);
            part += d * d;
        }
        sum += part;
    }
    let scale32 = f32::cast_from((sum / f64::cast_from(hd as u32) + eps).sqrt());
    let inv = 1.0f32 / scale32;
    for i in 0..hd {
        cache[cb + i] = k[kb + i] * inv * w[i];
    }
    for pp in 0..half {
        let c = cs[pos * half * 2 + pp * 2];
        let si = cs[pos * half * 2 + pp * 2 + 1];
        let x0 = cache[cb + pp];
        let x1 = cache[cb + pp + half];
        cache[cb + pp] = x0 * c - x1 * si;
        cache[cb + pp + half] = x0 * si + x1 * c;
    }
}
