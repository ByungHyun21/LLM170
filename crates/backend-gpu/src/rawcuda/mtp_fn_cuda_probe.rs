//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: **케이스마다** 드래프트 KV·pos를
//!    리셋(reset+fresh 오라클 상태 — 원장 19호: KV는 시퀀스 상태, 재사용
//!    루프는 문맥을 꼬는 사고 계급). 케이스 내 스텝 체인에서는 KV가
//!    누적되어야 한다(core 계약 — mtp_attn_cpu_row가 pos에 KV를 적립).
//! ② 형상은 픽스처 config.json에서 자동 확정(Flash-Next 실측 n=2560·
//!    hc=4·lr=320·24헤드/2KV·hd=256·n_rot=64·512e top10·ffn 640/640·
//!    vocab 248320)+ 3스텝 드래프트 체인(pos 1→3, KV 누적·chain h 인계).
//! ③ 캡처-재생 3방향: 결정론 합성 + 실측 믹서 패치 재생(mtp_hyper_
//!    connection_mixer_patch.safetensors F16 3텐서 — 오프셋 직독) + MTP
//!    GGUF(Q8_0) 실측 가중 재생(eh_proj·hc 믹서·어텐션 투영·라우터·
//!    shared·선택 전문가 — 행 오프셋 직독, 전량 적재 없음).
//! ④ 종단 값이 유일 불변량: 판정은 체인 값 maxdiff+비트 불일치 수
//!    (argmax·토큰은 부가 보고 — plans/124 §6).
//!
//! [오라클 — core qwen4exp 참조 직이식(값 maxdiff 판정의 유일 기준,
//! plans/124 §6). 인용은 전부 D:/LLM170 기준 줄번호, 2026-10-05]
//! - crates/core/src/qwen4exp/frame/mtp.rs — **본 모듈의 황금 계약**:
//!   mtp_draft_frame L362-484(eh_proj L388-396 호스트 cat 조립·t=hc 배치
//!   GEMM → hc_mix_draft L168-229 → dense 어텐션 L398-420 → hc_combine_
//!   draft L232-251 → moe_draft L254-353 → 헤드 L422-456 → argmax·체인
//!   반출 L458-482).
//! - crates/core/src/qwen4exp/layers.rs — mtp_attn_cpu_row L2505-2573
//!   (dense 어텐션 CPU 코어 — cell 0 스킵 L2540·게이트 sigmoid L2556-
//!   2559)·hc_combine L2575-2586(res += out·2σ(inj/hc))·mtp_draft_step_h
//!   L1104-1131(enorm 플랫[n]·hnorm 플랫[hc·n] — vLLM GemmaRMSNorm 평탄
//!   1회, 스트림별 아님).
//! - crates/core/src/qwen4exp/stages/hc.rs — grouped_rms L14-21 ·
//!   hc_mix_ex L25-86(FNC 검증층과 동일 인용).
//! - crates/core/src/qwen4exp/stages/moe.rs — 라우팅 L46-70(softmax·
//!   total_cmp 내림·wsum 하한 6.1035156e-5). 단 softmax exp는 프레임
//!   커널(q4_moe_top10_m — exp_cr 클래스)이 exp_cr를 쓰므로 오라클도
//!   exp_cr로 미러한다(FNE 노선 계승 — exp는 단조라 이산 선택 불변).
//!   전문가 가중 누산은 **프레임 순서(선택 순서=확률 내림)** —
//!   MoeWeightedSum(q4_moe_weighted_sum k순 누산) 미러. 모듈(MoeCuda)
//!   은 e-오름차 누산(moe.rs t=1 fast 값경로 순서) — 수학 동치·반올림
//!   순서만 상이(모듈 머리 [정합 목표] 문서화) → post-MoE 값은 문서화
//!   문턱(§3.4 ≤2e-4) 판정, pre-MoE 값은 비트동일 판정.
//! - crates/core/src/ops.rs — sq_sum L11-31 · rms_norm L33-37 · exp_cr
//!   L63-119 · silu L127-130 · sigmoid L131-133 · rope_head L141-159.
//! - crates/core/src/matmul/cpu.rs — matmul L64-76(행별 f32 순차 누산,
//!   mul+add 무-FMA — ADR-0005)·greedy_from L235-253(동률 최저 인덱스).
//! - crates/core/src/quant/deq.rs — deq_q8_0 L181-185(y[j] = qs·d —
//!   Q8_0 디양자화 미러, 픽스처 재생용).
//!
//! [정합 원장 요약 — 판정 기준]
//! - pre-MoE 스테이지(eh·mix·attn·ao·combine·ffn-mix): bitdiff=0·nan=0
//!   요구(비트동일 — FNC 커널·본 모듈 gemv/combine의 연산별 반올림 미러).
//! - post-MoE(mout·chain_h·hin·logits): maxdiff ≤ 2e-4(plans/124 §3.4)
//!   + bitdiff 보고(MoE 누산 순서 차이의 마지막-ulp 계급 — 측정값 기록).
//! - 캡처 지점(케이스 ii): 체인 h == ffn combine 직후 잔차(pre-mixer)·
//!   ≠ attn combine 직후(결함 10호 계급 판별)·chain 인계 누산 3스텝.
//! - 실측 재생(iii·iv): (iii) 패치 F16→f32 정확 변환 — 비트동일 기대.
//!   (iv) Q8_0 디양자화 정확(값 보존)·MoE 계열 f16 왕복(등록 양측 동일
//!   값) — (i)과 동일 기준. 헤드(output)는 MTP GGUF에 없음(kv nextn_
//!   shared_target_tensors=1 — 본체 공유)으로 전 케이스 합성 f32 사용,
//!   실측 vocab 248320 전폭 판정.
//!
//! [음성대조 — 원장 17호(계기 자체 검증)] 2종 모두 NEG-DETECTED 필:
//! (a) 잘못된 캡처 지점: 체인을 ffn combine **전**(attn combine 직후
//!     잔차)에서 반출한 오라클 변형 vs 모듈 정답 — 임계 초과 검증
//!     (결함 10호 계급).
//! (b) 잘못된 노름 규약: EXL3 MTP의 w−1 저장 규약(§3.4 constant_bias
//!     =1.0 노름)을 qwen4exp 원값 감마에 오적용(등록값 +1)한 **모듈
//!     실경로** vs 정답 오라클 — 임계 초과 검증.
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 합성 헤드
//! f32 [vocab][n] ≈ 2.5GB 1회 상주+h2d(프로브 경제성 계약 — 모듈 머리
//! [메모리 규약] 문서화).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::exl3_cuda_probe::{Rng, f16_to_f32, f32_to_f16, gen_unif, maxdiff_nan};
use crate::rawcuda::fn_support::{FnGguf, FnGgufTensor};
use crate::rawcuda::mtp_fn_cuda::{MTP_KV_CAP, MtpFnCuda, MtpFnDims};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// 기본 픽스처 경로(과제 지정 — 프로브 인자로 대체 가능).
pub const FN_MTP_EXL3_DIR: &str = "D:/models/Qwen3.8-Flash-Next-exl3-5.05bpw";
/// MTP GGUF 기본(Q8_0 — 디양자화가 값 보존이라 재생 판정이 비트동일
/// 계급을 유지한다; Q4_K_M 판은 옵션 인자로 대체 가능).
pub const FN_MTP_GGUF_Q8: &str =
    "D:/models/qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

/// 체인 종단 문서화 문턱(plans/124 §3.4 MTP 드래프트 ≤2e-4) — post-MoE
/// 값 판정(모듈/오라클 MoE 누산 순서 차이의 상한 계급).
const MTP_CHAIN_THRESH: f32 = 2e-4;

// ── core 미러 오라클 프리미티브(인용은 헤드 [오라클] 항) ──

/// ops.rs sq_sum(L11-31) 직이식.
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

/// ops.rs rms_norm(L33-37) 직이식.
fn o_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = o_sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// ops.rs exp_cr(L63-119) 직이식 — 커널 mtp_exp_cr·fn_moe_exp_cr과
/// 리터럴까지 동일.
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

/// ops.rs silu(L127-130) 직이식.
fn o_silu(x: f32) -> f32 {
    x / (1.0 + o_exp_cr(-x))
}

/// ops.rs sigmoid(L131-133) 직이식.
fn o_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + o_exp_cr(-x))
}

/// ops.rs rope_head(L141-159) 직이식 — NEOX 페어링·f64 중간.
fn o_rope_head(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for p in 0..half {
        let theta = base.powf(-(2.0 * p as f32) / n_rot as f32);
        let angle = pos as f32 * theta;
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (head[p] as f64, head[p + half] as f64);
        let (cf, sf) = (c as f64, s as f64);
        head[p] = (x0 * cf - x1 * sf) as f32;
        head[p + half] = (x0 * sf + x1 * cf) as f32;
    }
}

/// cpu.rs matmul(L64-76) 행내적 미러 — f32 순차 누산(mul+add 무-FMA).
fn o_dot(x: &[f32], w: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += x[i] * w[i];
    }
    acc
}

/// cpu.rs greedy_from(L235-253) 미러 — 동률 최저 인덱스(초과 갱신).
fn o_greedy(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// 비트 불일치 수(NaN은 maxdiff_nan이 별도 집계).
fn o_bitdiff(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count()
}

/// 판정 트리플 (maxdiff, bitdiff, nan).
fn o_judge(got: &[f32], want: &[f32]) -> (f32, usize, usize) {
    let (md, nan) = maxdiff_nan(got, want);
    (md, o_bitdiff(got, want), nan)
}

// ── 오라클 체인(계약 지형 — 헤드 [오라클] 인용) ──

/// 합성/실측 공용 오라클 가중세트(전부 f32 — MoE 계열은 f16 왕복값).
struct OW {
    eh: Vec<f32>, //[n][2n]
    wq: Vec<f32>, //[qg][n]
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>, //[n][q_dim]
    qn: Vec<f32>,
    kn: Vec<f32>,
    // hc attn/ffn 믹서(norm [hcn]·down [lr][hcn]·up [hcn][lr]·inj [hc][hcn]).
    an: Vec<f32>,
    ad: Vec<f32>,
    au: Vec<f32>,
    ai: Vec<f32>,
    fn_n: Vec<f32>,
    fd: Vec<f32>,
    fu: Vec<f32>,
    fi: Vec<f32>,
    // nextn.hc_head 믹서(inject 없음).
    hn_: Vec<f32>,
    hd_: Vec<f32>,
    hu_: Vec<f32>,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    route: Vec<f32>,    //[ne][n]
    route_sh: Vec<f32>, //[n]
    sh_g: Vec<f32>,
    sh_u: Vec<f32>,
    sh_d: Vec<f32>,
    wout: Vec<f32>, //[vocab][n] — 빈 벡터면 로짓/토큰 스킵(hin은 항상 산출)
}

/// 오라클 드래프트 시퀀스 상태(KV + pos — SeqState4의 MTP 드래프트 분).
struct OState {
    pos: usize,
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
}

impl OState {
    fn new(pos: usize, kv_dim: usize) -> Self {
        OState {
            pos,
            kv_k: vec![0.0; MTP_KV_CAP * kv_dim],
            kv_v: vec![0.0; MTP_KV_CAP * kv_dim],
        }
    }
    fn clone_state(&self) -> Self {
        OState {
            pos: self.pos,
            kv_k: self.kv_k.clone(),
            kv_v: self.kv_v.clone(),
        }
    }
}

/// 오라클 스텝 산출.
struct OMids {
    eh: Vec<f32>,
    mix1: Vec<f32>,
    inj1: Vec<f32>,
    attn: Vec<f32>,
    ao: Vec<f32>,
    res2: Vec<f32>,
    mix2: Vec<f32>,
    inj2: Vec<f32>,
    mout: Vec<f32>,
    chain: Vec<f32>,
    hin: Vec<f32>,
    logits: Vec<f32>,
    token: Option<u32>,
    /// 라우팅 선택(선택 순서 — 전문가 등록용).
    sel: Vec<(u32, f32)>,
}

/// stages/hc.rs hc_mix_ex(L25-86) 직이식 — (mixed, inject).
fn o_hc_mix(
    dims: &MtpFnDims,
    w_norm: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_inject: Option<&[f32]>,
    res: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (hc, n, lr) = (dims.hc, dims.n, dims.low_rank);
    let hcn = hc * n;
    // 1) grouped RMSNorm(L40-45).
    let mut xn = vec![0.0f32; hcn];
    for s in 0..hc {
        xn[s * n..(s + 1) * n].copy_from_slice(&o_rms_norm(
            &res[s * n..(s + 1) * n],
            &w_norm[s * n..(s + 1) * n],
            dims.eps,
        ));
    }
    // 2) down → inject(동일 입력 xn — D4 그룹 계약).
    let mut lo = vec![0.0f32; lr];
    for o in 0..lr {
        lo[o] = o_dot(&xn, &w_down[o * hcn..(o + 1) * hcn]);
    }
    let inj = match w_inject {
        Some(wi) => (0..hc)
            .map(|o| o_dot(&xn, &wi[o * hcn..(o + 1) * hcn]))
            .collect(),
        None => Vec::new(),
    };
    // silu(lo/hc)(L58-61).
    for v in lo.iter_mut() {
        *v = o_silu(*v / hc as f32);
    }
    // up.
    let mut gate = vec![0.0f32; hcn];
    for o in 0..hcn {
        gate[o] = o_dot(&lo, &w_up[o * lr..(o + 1) * lr]);
    }
    // 게이트 적용 + 스트림 평균(L66-79).
    let mut mix = vec![0.0f32; n];
    for i in 0..n {
        let mut m = 0.0f32;
        for s in 0..hc {
            m += xn[s * n + i] * o_sigmoid(gate[s * n + i]);
        }
        mix[i] = m / hc as f32;
    }
    (mix, inj)
}

/// layers.rs hc_combine(L2575-2586) 직이식 — res += out·(2σ(inj_s/hc)).
fn o_hc_combine(res: &mut [f32], out: &[f32], inj: &[f32], hc: usize, n: usize) {
    for s in 0..hc {
        let w = 2.0 * o_sigmoid(inj[s] / hc as f32);
        let base = s * n;
        for (i, &ov) in out.iter().enumerate() {
            res[base + i] += ov * w;
        }
    }
}

/// layers.rs mtp_attn_cpu_row(L2505-2573) 직이식(모듈 미러와 독립 사본 —
/// 계기 이원 검증). 원본 산술 순서 그대로(cell 0 스킵 포함).
fn o_attn_row(
    dims: &MtpFnDims,
    q_row: &mut [f32],
    k_row: &mut [f32],
    v_row: &[f32],
    st: &mut OState,
    qn: &[f32],
    kn: &[f32],
) -> Vec<f32> {
    let (n_head, n_kv, hd, n_rot) = (dims.n_head, dims.n_kv, dims.head_dim, dims.n_rot);
    let pos = st.pos;
    let kq_scale = dims.kq_scale();
    for h in 0..n_head {
        let lo = h * 2 * hd;
        let mut qh = o_rms_norm(&q_row[lo..lo + hd], qn, dims.eps);
        o_rope_head(&mut qh, pos as u32, n_rot, dims.rope_base);
        q_row[lo..lo + hd].copy_from_slice(&qh);
    }
    let kbase = pos * n_kv * hd;
    for h in 0..n_kv {
        let lo = h * hd;
        let mut kh = o_rms_norm(&k_row[lo..lo + hd], kn, dims.eps);
        o_rope_head(&mut kh, pos as u32, n_rot, dims.rope_base);
        st.kv_k[kbase + lo..kbase + lo + hd].copy_from_slice(&kh);
    }
    st.kv_v[kbase..kbase + n_kv * hd].copy_from_slice(v_row);
    let n_past = pos;
    let mut out = vec![0.0f32; n_head * hd];
    for h in 0..n_head {
        let kvh = h / (n_head / n_kv);
        let mut maxv = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; n_past];
        for (p0, sc) in scores.iter_mut().enumerate() {
            let p = p0 + 1; // cell 0 스킵
            let b = p * n_kv * hd + kvh * hd;
            let mut d = 0.0f32;
            for i in 0..hd {
                d += q_row[h * 2 * hd + i] * st.kv_k[b + i];
            }
            *sc = d * kq_scale;
            maxv = maxv.max(*sc);
        }
        let mut sum = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - maxv).exp();
            sum += *sc;
        }
        let ob = h * hd;
        for (p0, sc) in scores.iter().enumerate() {
            let w = sc / sum;
            if w == 0.0 {
                continue;
            }
            let b = (p0 + 1) * n_kv * hd + kvh * hd;
            for i in 0..hd {
                out[ob + i] += w * st.kv_v[b + i];
            }
        }
        let gb = h * 2 * hd + hd;
        for i in 0..hd {
            out[ob + i] *= o_sigmoid(q_row[gb + i]);
        }
    }
    out
}

/// 라우팅 선택 — moe.rs L46-62 미러(softmax exp_cr·내림차·동률 최저 id·
/// wsum 하한·w==0 스킵). fn_moe_route/q4_moe_top10_m와 동치 계약.
fn o_route_select(dims: &MtpFnDims, w: &OW, mix: &[f32]) -> Vec<(u32, f32)> {
    let (n, ne, nu) = (dims.n, dims.n_expert, dims.n_used);
    let mut lg = vec![0.0f32; ne];
    for e in 0..ne {
        lg[e] = o_dot(mix, &w.route[e * n..(e + 1) * n]);
    }
    let mx = lg.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut zs = 0.0f32;
    for v in lg.iter_mut() {
        *v = o_exp_cr(*v - mx);
        zs += *v;
    }
    for v in lg.iter_mut() {
        *v /= zs;
    }
    let mut taken = vec![false; ne];
    let mut sel: Vec<(u32, f32)> = Vec::with_capacity(nu);
    for _ in 0..nu {
        let mut be = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for e in 0..ne {
            if !taken[e] && lg[e] > bv {
                bv = lg[e];
                be = e;
            }
        }
        taken[be] = true;
        sel.push((be as u32, bv));
    }
    let mut wsum = 0.0f32;
    for &(_, p) in &sel {
        wsum += p;
    }
    wsum = wsum.max(6.103_515_6e-5);
    sel.into_iter()
        .map(|(e, p)| (e, p / wsum))
        .filter(|&(_, wv)| wv != 0.0)
        .collect()
}

/// frame/mtp.rs moe_draft(L254-353) 미러 — 라우팅(선택 순서)·전문가·
/// shared·가중 합. 전문가 가중 (gate,up,down)은 exps에서(e-키).
fn o_moe(
    dims: &MtpFnDims,
    w: &OW,
    exps: &HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    mix: &[f32],
) -> Result<(Vec<f32>, Vec<(u32, f32)>), String> {
    let (n, nff) = (dims.n, dims.n_ff);
    let sgate = o_dot(mix, &w.route_sh);
    let sel = o_route_select(dims, w, mix);
    // 전문가(선택 순서 계산 — MoeWeightedSum k순 누산과 동일 원소).
    let mut ysel: Vec<Vec<f32>> = Vec::with_capacity(sel.len());
    for &(e, _) in &sel {
        let Some((g, u, dwn)) = exps.get(&(e as usize)) else {
            return Err(format!("오라클 MoE: 전문가 e={e} 가중 미생성"));
        };
        let gv: Vec<f32> = (0..nff)
            .map(|i| o_dot(mix, &g[i * n..(i + 1) * n]))
            .collect();
        let uv: Vec<f32> = (0..nff)
            .map(|i| o_dot(mix, &u[i * n..(i + 1) * n]))
            .collect();
        let glu: Vec<f32> = (0..nff).map(|i| o_silu(gv[i]) * uv[i]).collect();
        let mut yo = vec![0.0f32; n];
        for o in 0..n {
            yo[o] = o_dot(&glu, &dwn[o * nff..(o + 1) * nff]);
        }
        ysel.push(yo);
    }
    // 가중 합(선택 순서 k — q4_moe_weighted_sum 미러).
    let mut mout = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        for (k, &(_, wv)) in sel.iter().enumerate() {
            acc += wv * ysel[k][i];
        }
        mout[i] = acc;
    }
    // shared(sigmoid 게이트 — AxpyScaled L345-352). gate/up [nff_sh][n]
    // (실측 nff_sh=nff=640 — core는 n_ff_exp 버퍼 재사용 계약).
    let nff_sh = dims.n_ff_sh;
    let sg: Vec<f32> = (0..nff_sh)
        .map(|i| o_dot(mix, &w.sh_g[i * n..(i + 1) * n]))
        .collect();
    let su: Vec<f32> = (0..nff_sh)
        .map(|i| o_dot(mix, &w.sh_u[i * n..(i + 1) * n]))
        .collect();
    let shglu: Vec<f32> = (0..nff_sh).map(|i| o_silu(sg[i]) * su[i]).collect();
    let mut shout = vec![0.0f32; n];
    for o in 0..n {
        shout[o] = o_dot(&shglu, &w.sh_d[o * nff_sh..(o + 1) * nff_sh]);
    }
    let sw = o_sigmoid(sgate);
    for o in 0..n {
        mout[o] += shout[o] * sw;
    }
    Ok((mout, sel))
}

/// frame/mtp.rs mtp_draft_frame(L362-484) 오라클 — 1스텝 전체.
/// capture_early=true면 체인을 ffn combine 전(res2)에서 반출(음성대조 a).
fn o_draft_step(
    dims: &MtpFnDims,
    w: &OW,
    exps: &mut HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    exp_fn: &dyn Fn(usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>,
    st: &mut OState,
    en: &[f32],
    hn: &[f32],
    capture_early: bool,
) -> Result<OMids, String> {
    let (n, hc, hcn) = (dims.n, dims.hc, dims.hc_dim());
    let (qg, kvd, qd) = (dims.qg_dim(), dims.kv_dim(), dims.q_dim());
    // 1) eh_proj(L388-396) — 스트림별 cat 호스트 dot(t=hc 배치 GEMM과
    //    값 동일 — FNC gemv 순차 누산 계약).
    let mut eh = vec![0.0f32; hcn];
    for s in 0..hc {
        let mut cat = vec![0.0f32; 2 * n];
        cat[..n].copy_from_slice(en);
        cat[n..].copy_from_slice(&hn[s * n..(s + 1) * n]);
        for o in 0..n {
            eh[s * n + o] = o_dot(&cat, &w.eh[o * 2 * n..(o + 1) * 2 * n]);
        }
    }
    // 2) attn 믹서.
    let (mix1, inj1) = o_hc_mix(dims, &w.an, &w.ad, &w.au, Some(&w.ai), &eh);
    // 3) dense 어텐션 — 투영 → CPU 행 → wo.
    let mut q = vec![0.0f32; qg];
    let mut k = vec![0.0f32; kvd];
    let mut v = vec![0.0f32; kvd];
    for o in 0..qg {
        q[o] = o_dot(&mix1, &w.wq[o * n..(o + 1) * n]);
    }
    for o in 0..kvd {
        k[o] = o_dot(&mix1, &w.wk[o * n..(o + 1) * n]);
        v[o] = o_dot(&mix1, &w.wv[o * n..(o + 1) * n]);
    }
    let attn = o_attn_row(dims, &mut q, &mut k, &v, st, &w.qn, &w.kn);
    let mut ao = vec![0.0f32; n];
    for o in 0..n {
        ao[o] = o_dot(&attn, &w.wo[o * qd..(o + 1) * qd]);
    }
    let mut res = eh.clone();
    o_hc_combine(&mut res, &ao, &inj1, hc, n);
    let res2 = res.clone();
    // 4) ffn 믹서 + MoE + combine. 선택 전문가 가중은 exps에 선확보
    //    (모듈 등록·오라클 계산이 동일 가중을 본다 — 이원 검증 계약).
    let (mix2, inj2) = o_hc_mix(dims, &w.fn_n, &w.fd, &w.fu, Some(&w.fi), &res2);
    for &(e, _) in &o_route_select(dims, w, &mix2) {
        if let std::collections::hash_map::Entry::Vacant(ent) = exps.entry(e as usize) {
            ent.insert(exp_fn(e as usize)?);
        }
    }
    let (mout, sel) = o_moe(dims, w, exps, &mix2)?;
    let mut resf = res2.clone();
    o_hc_combine(&mut resf, &mout, &inj2, hc, n);
    // 5) 헤드 — nextn.hc_head 믹서(항상) + output GEMV + greedy(wout 있을 때).
    let (hin, logits, token) = if !w.wout.is_empty() {
        let (h, _) = o_hc_mix(dims, &w.hn_, &w.hd_, &w.hu_, None, &resf);
        let mut lg = vec![0.0f32; dims.vocab];
        for o in 0..dims.vocab {
            lg[o] = o_dot(&h, &w.wout[o * n..(o + 1) * n]);
        }
        let tok = o_greedy(&lg);
        (h, lg, Some(tok))
    } else {
        let (h, _) = o_hc_mix(dims, &w.hn_, &w.hd_, &w.hu_, None, &resf);
        (h, Vec::new(), None)
    };
    // 체인 반출 — ffn combine 직후(pre-mixer). 음성대조 a는 res2.
    let chain = if capture_early {
        res2.clone()
    } else {
        resf.clone()
    };
    Ok(OMids {
        eh,
        mix1,
        inj1,
        attn,
        ao,
        res2,
        mix2,
        inj2,
        mout,
        chain,
        hin,
        logits,
        token,
        sel,
    })
}

// ── 합성 가중 생성(Rng splitmix64 — initializer_range 0.02 계급) ──

fn gen_mat(rows: usize, cols: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * cols)
        .map(|_| (lo + rng.next_f64() * (hi - lo)) as f32)
        .collect()
}

/// 합성 전문가 가중(e 결정론).
fn gen_expert(dims: &MtpFnDims, e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (n, nff) = (dims.n, dims.n_ff);
    let base = 0x5EED_F00D_0000_E000u64 + (e as u64) * 0x100;
    (
        gen_mat(nff, n, base, -0.02, 0.02),
        gen_mat(nff, n, base + 1, -0.02, 0.02),
        gen_mat(n, nff, base + 2, -0.02, 0.02),
    )
}

/// f32 → f16 왕복(MoeCuda 등록값 — 오라클·모듈 양측 동일 값 계약).
fn rt16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// f32 행렬 → f16 등록 바이트(값은 이미 왕복값).
fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for &x in v {
        b.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
    b
}

/// 합성 가중세트(MoE 계열 f16 왕복 — 모듈 등록값과 동일).
fn synth_weights(dims: &MtpFnDims, with_head: bool) -> OW {
    let (n, hc, hcn, lr) = (dims.n, dims.hc, dims.hc_dim(), dims.low_rank);
    let (qg, kvd, qd) = (dims.qg_dim(), dims.kv_dim(), dims.q_dim());
    let s = |k: u64| 0x5EED_F00D_0000_0000u64 + (k << 12);
    let rt = |v: Vec<f32>| v.iter().map(|&x| rt16(x)).collect::<Vec<f32>>();
    OW {
        eh: gen_mat(n, 2 * n, s(1), -0.02, 0.02),
        wq: gen_mat(qg, n, s(2), -0.02, 0.02),
        wk: gen_mat(kvd, n, s(3), -0.02, 0.02),
        wv: gen_mat(kvd, n, s(4), -0.02, 0.02),
        wo: gen_mat(n, qd, s(5), -0.02, 0.02),
        qn: gen_mat(1, dims.head_dim, s(6), 0.8, 1.2),
        kn: gen_mat(1, dims.head_dim, s(7), 0.8, 1.2),
        an: gen_mat(1, hcn, s(8), 0.8, 1.2),
        ad: gen_mat(lr, hcn, s(9), -0.02, 0.02),
        au: gen_mat(hcn, lr, s(10), -0.02, 0.02),
        ai: gen_mat(hc, hcn, s(11), -0.02, 0.02),
        fn_n: gen_mat(1, hcn, s(12), 0.8, 1.2),
        fd: gen_mat(lr, hcn, s(13), -0.02, 0.02),
        fu: gen_mat(hcn, lr, s(14), -0.02, 0.02),
        fi: gen_mat(hc, hcn, s(15), -0.02, 0.02),
        hn_: gen_mat(1, hcn, s(16), 0.8, 1.2),
        hd_: gen_mat(lr, hcn, s(17), -0.02, 0.02),
        hu_: gen_mat(hcn, lr, s(18), -0.02, 0.02),
        enorm: gen_mat(1, n, s(19), 0.8, 1.2),
        hnorm: gen_mat(1, hcn, s(20), 0.8, 1.2),
        route: rt(gen_mat(dims.n_expert, n, s(21), -0.02, 0.02)),
        route_sh: rt(gen_mat(1, n, s(22), -0.02, 0.02)),
        sh_g: rt(gen_mat(dims.n_ff_sh, n, s(23), -0.02, 0.02)),
        sh_u: rt(gen_mat(dims.n_ff_sh, n, s(24), -0.02, 0.02)),
        sh_d: rt(gen_mat(n, dims.n_ff_sh, s(25), -0.02, 0.02)),
        wout: if with_head {
            gen_mat(dims.vocab, n, s(0x77), -0.02, 0.02)
        } else {
            Vec::new()
        },
    }
}

/// 모듈 핵심 가중 등록(케이스당 1회 — 대형 헤드 재업로드 회피 계약).
fn setup_module(m: &mut MtpFnCuda, w: &OW, include_head: bool) -> Result<(), String> {
    m.set_eh_proj(&w.eh)?;
    m.set_attn(&w.wq, &w.wk, &w.wv, &w.wo, &w.qn, &w.kn)?;
    m.hc.register(0, "attn", &w.an, &w.ad, &w.au, Some(&w.ai))?;
    m.hc.register(0, "ffn", &w.fn_n, &w.fd, &w.fu, Some(&w.fi))?;
    m.hc.register(0, "nextn_head", &w.hn_, &w.hd_, &w.hu_, None)?;
    m.moe
        .set_router_f16(&f16_bytes(&w.route), &f16_bytes(&w.route_sh))?;
    m.moe.set_shared_f16(
        &f16_bytes(&w.sh_g),
        &f16_bytes(&w.sh_u),
        &f16_bytes(&w.sh_d),
    )?;
    if include_head && !w.wout.is_empty() {
        m.set_head(&w.wout)?;
    }
    Ok(())
}

/// 선택 전문가 증분 등록(이미 등록된 id는 건너뛴다 — 스텝 간 경제성).
fn ensure_experts(
    m: &mut MtpFnCuda,
    exps: &HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    registered: &mut HashSet<usize>,
) -> Result<(), String> {
    for (e, (g, u, d)) in exps {
        if registered.contains(e) {
            continue;
        }
        m.moe
            .add_expert_f16(*e, &f16_bytes(g), &f16_bytes(u), &f16_bytes(d))?;
        registered.insert(*e);
    }
    Ok(())
}

// ── 실측 픽스처: mtp_hyper_connection_mixer_patch.safetensors(F16 3텐서) ──

/// 패치 믹서 3텐서 오프셋 직독(F16 → f32) — hc_cuda_probe 패턴 동일 계약.
fn patch_mixer(dir: &Path, dims: &MtpFnDims) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let path = dir.join("mtp_hyper_connection_mixer_patch.safetensors");
    let mut f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb).map_err(|e| e.to_string())?;
    let hlen = u64::from_le_bytes(lenb);
    if hlen == 0 || hlen > (1 << 22) {
        return Err(format!("mtp 패치 헤더 길이 {hlen} — 가드(실측 360)"));
    }
    let mut hb = vec![0u8; hlen as usize];
    f.read_exact(&mut hb).map_err(|e| e.to_string())?;
    let data_base = 8 + hlen;
    let v = JParser { b: &hb, p: 0 }.parse()?;
    let obj = v.as_obj().ok_or("mtp 패치: 헤더가 객체 아님")?;
    let (hcn, lr) = (dims.hc_dim(), dims.low_rank);
    let (mut norm, mut down, mut up) = (Vec::new(), Vec::new(), Vec::new());
    for (name, tv) in obj {
        if !name.starts_with("mtp.hyper_connection_mixer.") {
            continue;
        }
        let dt = tv
            .get("dtype")
            .and_then(JVal::as_str)
            .ok_or("mtp 패치: dtype 없음")?;
        if dt != "F16" {
            return Err(format!("mtp 패치: dtype {dt} — F16 고정(픽스처 변경 가드)"));
        }
        let shape = tv
            .get("shape")
            .and_then(JVal::as_arr)
            .ok_or("mtp 패치: shape 없음")?;
        let sh: Vec<usize> = shape
            .iter()
            .filter_map(JVal::as_f64)
            .map(|x| x as usize)
            .collect();
        let offs = tv
            .get("data_offsets")
            .and_then(JVal::as_arr)
            .ok_or("mtp 패치: data_offsets 없음")?;
        let g64 = |i: usize| {
            offs.get(i)
                .and_then(JVal::as_f64)
                .map(|x| x as u64)
                .ok_or_else(|| format!("mtp 패치: data_offsets[{i}] 없음"))
        };
        let (b, e) = (g64(0)?, g64(1)?);
        let want: Vec<usize> = if name.ends_with("hc_norm.weight") {
            vec![hcn]
        } else if name.ends_with("input_mix_weight_down.weight") {
            vec![lr, hcn]
        } else if name.ends_with("input_mix_weight_up.weight") {
            vec![hcn, lr]
        } else {
            continue;
        };
        if sh != want {
            return Err(format!(
                "mtp 패치: {name} shape {sh:?} != {want:?}(dims 불일치 가드)"
            ));
        }
        f.seek(SeekFrom::Start(data_base + b))
            .map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; (e - b) as usize];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        let vals: Vec<f32> = buf
            .as_chunks::<2>().0.iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        if name.ends_with("hc_norm.weight") {
            norm = vals;
        } else if name.ends_with("input_mix_weight_down.weight") {
            down = vals;
        } else {
            up = vals;
        }
    }
    if norm.is_empty() || down.is_empty() || up.is_empty() {
        return Err("mtp 패치: 3텐서(norm/down/up) 인식 실패".into());
    }
    Ok((norm, down, up))
}

// ── 실측 픽스처: MTP GGUF(Q8_0) 행 오프셋 직독 + 디양자화 ──

/// 텐서 행 원시 바이트 직독 — fn_support read_rows(L1042)와 동일 산식에
/// stacked 전문가 스택 확장(row0+nrows가 ne[1]을 넘을 수 있다 — ne[2]
/// 전문가 축: gate/up 스택 [ne0, n_ff, 512]·down 스택 [ne0, n, 512]).
fn gguf_rows(g: &FnGguf, t: &FnGgufTensor, row0: u64, nrows: u64) -> Result<Vec<u8>, String> {
    let rb = FnGguf::row_bytes(t)?;
    let rows_total: u64 = t.dims.iter().skip(1).product::<u64>().max(1);
    if row0 + nrows > rows_total {
        return Err(format!(
            "gguf 판독: row0={row0}+{nrows} > 행수 {rows_total} ({})",
            t.name
        ));
    }
    let mut f = std::fs::File::open(&g.shards[t.shard]).map_err(|e| e.to_string())?;
    let abs = g.data_base[t.shard] + t.off + row0 * rb;
    f.seek(SeekFrom::Start(abs)).map_err(|e| e.to_string())?;
    let n = (rb * nrows) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Q8_0 행 디양자화 — deq.rs deq_q8_0(L181-185) 미러(y[j] = qs·d).
fn deq_q8_0_rows(raw: &[u8], n_in: usize) -> Result<Vec<f32>, String> {
    if !n_in.is_multiple_of(32) {
        return Err(format!("deq q8_0: n_in={n_in} — 32 배수 계약"));
    }
    let nblk = n_in / 32;
    let rowb = nblk * 34;
    if !raw.len().is_multiple_of(rowb) {
        return Err(format!(
            "deq q8_0: bytes={} — 행 바이트 {rowb} 정렬 가드",
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / rowb * n_in);
    for row in raw.chunks_exact(rowb) {
        for b in 0..nblk {
            let blk = &row[b * 34..(b + 1) * 34];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for j in 0..32 {
                out.push(blk[2 + j] as i8 as f32 * d);
            }
        }
    }
    Ok(out)
}

/// GGUF F32 텐서 전행 직독(다중 행 포함 — ne[1..] 적립).
fn gguf_f32(g: &FnGguf, name: &str) -> Result<Vec<f32>, String> {
    let t = g
        .tensor(name)
        .ok_or_else(|| format!("gguf 텐서 없음: {name}"))?;
    let rows: u64 = t.dims.iter().skip(1).product::<u64>().max(1);
    let raw = gguf_rows(g, t, 0, rows)?;
    if t.ty != 0 {
        return Err(format!("gguf f32: {name} ty{} — F32 계약", t.ty));
    }
    Ok(raw
        .as_chunks::<4>().0.iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Q8_0 텐서 행 직독+디양자화(row0..row0+nrows → f32 [nrows][ne0]).
fn gguf_q8_f32(g: &FnGguf, name: &str, row0: u64, nrows: u64) -> Result<Vec<f32>, String> {
    let t = g
        .tensor(name)
        .ok_or_else(|| format!("gguf 텐서 없음: {name}"))?;
    if t.ty != 8 {
        return Err(format!("gguf q8_0: {name} ty{} — Q8_0 계약", t.ty));
    }
    let n_in = t.dims.first().copied().unwrap_or(0) as usize;
    let raw = gguf_rows(g, t, row0, nrows)?;
    deq_q8_0_rows(&raw, n_in)
}

// ── 메인 프로브 ──

/// mtp-frame — Flash-Next MTP 드래프트 프레임 종단 값 판정.
/// (i) 합성 3스텝 체인(pos 1→3, KV 누적·chain 인계·전문가 선택 변화) —
///     pre-MoE 비트동일 + post-MoE §3.4 문턱 + 토큰 일치.
/// (ii) 캡처 지점·시퀀스 정확성 — 체인=ffn combine 직후 증명(반출 지점
///     판별 maxdiff) + chain 인계 판별(스테일 hn 대조).
/// (iii) 실측 패치 픽스처 — nextn_head 믹서 F16 3텐서 재생(1스텝).
/// (iv) 실측 MTP GGUF(Q8_0) — 실측 가중 체인 2스텝(헤드 가중 제외).
pub fn cuda_mtp_frame_check(dir: &str, gguf_arg: &str) -> Result<String, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = MtpFnDims::from_config(&cfg)?;
    let (n, hc, hcn) = (dims.n, dims.hc, dims.hc_dim());
    let mut m = MtpFnCuda::new(dims)?;
    let dev = m.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // ── (i) 합성 체인 — 헤드 포함(전 vocab 실측 폭). ──
    let w = synth_weights(&dims, true);
    let mut exps: HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut registered: HashSet<usize> = HashSet::new();
    let exp_fn =
        |e: usize| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> { Ok(gen_expert(&dims, e)) };
    setup_module(&mut m, &w, true)?;
    m.reset(1)?; // 케이스 시작 — fresh KV·pos=1(오염 점검 ①)
    let mut st = OState::new(1, dims.kv_dim());
    let mut h_pre = gen_unif(hcn, 0x5EED_F00D_0000_9001, 0.5);
    let mut chain_last = (0.0f32, 0usize);
    for step in 0..3usize {
        let e_k = gen_unif(n, 0x5EED_F00D_0000_9100 + step as u64, 0.02);
        let en = o_rms_norm(&e_k, &w.enorm, dims.eps);
        let hn = o_rms_norm(&h_pre, &w.hnorm, dims.eps);
        let o = o_draft_step(&dims, &w, &mut exps, &exp_fn, &mut st, &en, &hn, false)?;
        ensure_experts(&mut m, &exps, &mut registered)?;
        let got = m.mtp_draft_step(&en, &hn)?;
        m.pos += 1; // 호출부 pos 진행 계약(layers.rs L1136 미러)
        // pre-MoE 비트동일 판정(8 버킷).
        let buckets: [(&str, &Vec<f32>, &Vec<f32>); 8] = [
            ("eh", &got.eh, &o.eh),
            ("mix1", &got.mix_attn, &o.mix1),
            ("inj1", &got.inj_attn, &o.inj1),
            ("attn", &got.attn, &o.attn),
            ("ao", &got.ao, &o.ao),
            ("res2", &got.res_attn, &o.res2),
            ("mix2", &got.mix_ffn, &o.mix2),
            ("inj2", &got.inj_ffn, &o.inj2),
        ];
        let mut pre_bad = Vec::new();
        for (name, gv, want) in &buckets {
            let (md, bd, nan) = o_judge(gv, want);
            if bd != 0 || nan != 0 {
                pre_bad.push(format!("{name}={md:.3e}/{bd}b/n{nan}"));
            }
        }
        // post-MoE 문턱 판정.
        let mut post_bad = Vec::new();
        for (name, gv, want) in [
            ("mout", &got.mout, &o.mout),
            ("chain", &got.chain_h, &o.chain),
            ("hin", &got.hin, &o.hin),
            ("logits", &got.logits, &o.logits),
        ] {
            let (md, bd, nan) = o_judge(gv, want);
            if md > MTP_CHAIN_THRESH || nan != 0 {
                post_bad.push(format!("{name}={md:.3e}/{bd}b/n{nan}"));
            }
        }
        let tok_ok = got.token.is_some() && got.token == o.token;
        let pass = pre_bad.is_empty() && post_bad.is_empty() && tok_ok;
        let pre_desc = if pre_bad.is_empty() {
            "bit-exact(8/8)".to_string()
        } else {
            pre_bad.join(",")
        };
        let (mdc, bdc, _) = o_judge(&got.chain_h, &o.chain);
        chain_last = (mdc, bdc);
        println!(
            "device: {dev} | mtp-frame (i) step{} pos={}: pre-MoE {} | post-MoE mout={:.3e} chain={:.3e}/{}b hin={:.3e} logits={:.3e} tok={}/{} | {}",
            step + 1,
            st.pos,
            pre_desc,
            o_judge(&got.mout, &o.mout).0,
            mdc,
            bdc,
            o_judge(&got.hin, &o.hin).0,
            o_judge(&got.logits, &o.logits).0,
            got.token.map(|t| t.to_string()).unwrap_or("-".into()),
            o.token.map(|t| t.to_string()).unwrap_or("-".into()),
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            fails.push(format!(
                "(i step{}) pre=[{}] post=[{}] tok={:?}vs{:?}",
                step + 1,
                pre_bad.join(","),
                post_bad.join(","),
                got.token,
                o.token
            ));
        }
        // 체인 인계(호출부 — layers.rs mtp_draft_step_h 계약).
        h_pre = o.chain.clone();
        st.pos += 1;
    }
    report.push_str(&format!(
        " · (i) chain {0:.3e}/{1}b",
        chain_last.0, chain_last.1
    ));

    // ── (ii) 캡처 지점·시퀀스 판별(계기 증명 — 결함 10호 계급). ──
    {
        // 캡처 지점: 정답 체인 vs ffn-combine-전 반출 변형.
        let mut stx = OState::new(1, dims.kv_dim());
        let e_k = gen_unif(n, 0x5EED_F00D_0000_9100, 0.02);
        let hp0 = gen_unif(hcn, 0x5EED_F00D_0000_9001, 0.5);
        let en = o_rms_norm(&e_k, &w.enorm, dims.eps);
        let hn = o_rms_norm(&hp0, &w.hnorm, dims.eps);
        let good = o_draft_step(&dims, &w, &mut exps, &exp_fn, &mut stx, &en, &hn, false)?;
        let mut sty = OState::new(1, dims.kv_dim());
        let early = o_draft_step(&dims, &w, &mut exps, &exp_fn, &mut sty, &en, &hn, true)?;
        let (md, _, _) = o_judge(&good.chain, &early.chain);
        // 인계 판별: 2스텝 정답 체인 vs 스텝1 hn을 재사용(인계 무시)한 체인.
        let mut run2 = |stale: bool| -> Result<Vec<f32>, String> {
            let mut s = OState::new(1, dims.kv_dim());
            let mut hp = gen_unif(hcn, 0x5EED_F00D_0000_9001, 0.5);
            for k in 0..2usize {
                let e_k = gen_unif(n, 0x5EED_F00D_0000_9100 + k as u64, 0.02);
                let hn_k = if stale && k == 1 {
                    o_rms_norm(
                        &gen_unif(hcn, 0x5EED_F00D_0000_9001, 0.5),
                        &w.hnorm,
                        dims.eps,
                    )
                } else {
                    o_rms_norm(&hp, &w.hnorm, dims.eps)
                };
                let en = o_rms_norm(&e_k, &w.enorm, dims.eps);
                let o = o_draft_step(&dims, &w, &mut exps, &exp_fn, &mut s, &en, &hn_k, false)?;
                hp = o.chain.clone();
                s.pos += 1;
            }
            Ok(hp)
        };
        let correct2 = run2(false)?;
        let stale2 = run2(true)?;
        let (mds, _, _) = o_judge(&correct2, &stale2);
        let cap_ok = md > MTP_CHAIN_THRESH;
        let seq_ok = mds > MTP_CHAIN_THRESH;
        println!(
            "device: {dev} | mtp-frame (ii) capture-point delta={md:.3e} (>{} {}) · handoff(stale-hn) delta={mds:.3e} (>{} {}) | {}",
            MTP_CHAIN_THRESH,
            if cap_ok { "OK" } else { "BAD" },
            MTP_CHAIN_THRESH,
            if seq_ok { "OK" } else { "BAD" },
            if cap_ok && seq_ok { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (ii) cap {md:.3e} hoff {mds:.3e}"));
        if !cap_ok {
            fails.push(format!("(ii) 캡처 지점 판별 실패 delta={md:.3e}"));
        }
        if !seq_ok {
            fails.push(format!("(ii) chain 인계 판별 실패 delta={mds:.3e}"));
        }
    }

    // ── (iii) 실측 패치 픽스처 — nextn_head 믹서 F16 3텐서 재생(1스텝). ──
    {
        let (pn, pd, pu) = patch_mixer(Path::new(dir), &dims)?;
        let mut wr = synth_weights(&dims, true);
        wr.hn_ = pn.clone();
        wr.hd_ = pd.clone();
        wr.hu_ = pu.clone();
        setup_module(&mut m, &wr, false)?; // 헤드 가중 불변 — 재업로드 회피
        let mut st3 = OState::new(1, dims.kv_dim());
        let e_k = gen_unif(n, 0x5EED_F00D_0000_9300, 0.02);
        let h0 = gen_unif(hcn, 0x5EED_F00D_0000_9301, 0.5);
        let en = o_rms_norm(&e_k, &wr.enorm, dims.eps);
        let hn = o_rms_norm(&h0, &wr.hnorm, dims.eps);
        let o = o_draft_step(&dims, &wr, &mut exps, &exp_fn, &mut st3, &en, &hn, false)?;
        ensure_experts(&mut m, &exps, &mut registered)?;
        m.reset(1)?;
        let got = m.mtp_draft_step(&en, &hn)?;
        let (mdh, bdh, nh) = o_judge(&got.hin, &o.hin);
        let (mdc, bdc, nc) = o_judge(&got.chain_h, &o.chain);
        let pass = mdh <= MTP_CHAIN_THRESH && nh == 0 && mdc <= MTP_CHAIN_THRESH && nc == 0;
        println!(
            "device: {dev} | mtp-frame (iii) patch fixture nextn_head (F16 정확변환: {}·{}·{}): hin maxdiff={mdh:.3e}/{bdh}b chain={mdc:.3e}/{bdc}b nan={nh}/{nc} | {}",
            pn.len(),
            pd.len(),
            pu.len(),
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (iii patch) hin {mdh:.3e}/{bdh}b"));
        if !pass {
            fails.push(format!(
                "(iii) hin {mdh:.3e}/{bdh}b chain {mdc:.3e}/{bdc}b nan {nh}/{nc}"
            ));
        }
    }

    // ── (iv) 실측 MTP GGUF(Q8_0) — 실측 가중 체인 2스텝. ──
    let gguf_path = if gguf_arg.is_empty() {
        FN_MTP_GGUF_Q8.to_string()
    } else {
        gguf_arg.to_string()
    };
    if Path::new(&gguf_path).exists() {
        let g = FnGguf::open(Path::new(&gguf_path))?;
        // nextn 블록 인덱스 — GGUF block_count(49 = 트렁크 48 + nextn 1,
        // llama.cpp nextn 규약) − 1 = 48. core mod.rs load_mtp L347 계약.
        let il = g
            .kv_u64("qwen4exp.block_count")
            .ok_or("gguf meta 누락: qwen4exp.block_count(nextn 블록 인덱스)")?
            as usize
            - 1;
        let rt = |v: Vec<f32>| v.iter().map(|&x| rt16(x)).collect::<Vec<f32>>();
        let (pn, pd, pu) = patch_mixer(Path::new(dir), &dims)?; // nextn_head=패치(실측)
        let wr = OW {
            eh: gguf_q8_f32(
                &g,
                &format!("blk.{il}.nextn.eh_proj.weight"),
                0,
                dims.n as u64,
            )?,
            wq: gguf_q8_f32(
                &g,
                &format!("blk.{il}.attn_q.weight"),
                0,
                dims.qg_dim() as u64,
            )?,
            wk: gguf_q8_f32(
                &g,
                &format!("blk.{il}.attn_k.weight"),
                0,
                dims.kv_dim() as u64,
            )?,
            wv: gguf_q8_f32(
                &g,
                &format!("blk.{il}.attn_v.weight"),
                0,
                dims.kv_dim() as u64,
            )?,
            wo: gguf_q8_f32(
                &g,
                &format!("blk.{il}.attn_output.weight"),
                0,
                dims.n as u64,
            )?,
            qn: gguf_f32(&g, &format!("blk.{il}.attn_q_norm.weight"))?,
            kn: gguf_f32(&g, &format!("blk.{il}.attn_k_norm.weight"))?,
            an: gguf_f32(&g, &format!("blk.{il}.hc_attn_norm.weight"))?,
            ad: gguf_q8_f32(
                &g,
                &format!("blk.{il}.hc_attn_down.weight"),
                0,
                dims.low_rank as u64,
            )?,
            au: gguf_q8_f32(
                &g,
                &format!("blk.{il}.hc_attn_up.weight"),
                0,
                dims.hc_dim() as u64,
            )?,
            ai: gguf_q8_f32(
                &g,
                &format!("blk.{il}.hc_attn_inject.weight"),
                0,
                dims.hc as u64,
            )?,
            fn_n: gguf_f32(&g, &format!("blk.{il}.hc_ffn_norm.weight"))?,
            fd: gguf_q8_f32(
                &g,
                &format!("blk.{il}.hc_ffn_down.weight"),
                0,
                dims.low_rank as u64,
            )?,
            fu: gguf_q8_f32(
                &g,
                &format!("blk.{il}.hc_ffn_up.weight"),
                0,
                dims.hc_dim() as u64,
            )?,
            fi: gguf_q8_f32(
                &g,
                &format!("blk.{il}.hc_ffn_inject.weight"),
                0,
                dims.hc as u64,
            )?,
            // nextn_head = 실측 패치(12.5MB F16 — EXL3 아카이브 별도 파일).
            hn_: pn,
            hd_: pd,
            hu_: pu,
            enorm: gguf_f32(&g, &format!("blk.{il}.nextn.enorm.weight"))?,
            hnorm: gguf_f32(&g, &format!("blk.{il}.nextn.hnorm.weight"))?,
            route: rt(gguf_f32(&g, &format!("blk.{il}.ffn_gate_inp.weight"))?),
            route_sh: rt(gguf_f32(
                &g,
                &format!("blk.{il}.ffn_gate_inp_shexp.weight"),
            )?),
            sh_g: rt(gguf_q8_f32(
                &g,
                &format!("blk.{il}.ffn_gate_shexp.weight"),
                0,
                dims.n_ff_sh as u64,
            )?),
            sh_u: rt(gguf_q8_f32(
                &g,
                &format!("blk.{il}.ffn_up_shexp.weight"),
                0,
                dims.n_ff_sh as u64,
            )?),
            sh_d: rt(gguf_q8_f32(
                &g,
                &format!("blk.{il}.ffn_down_shexp.weight"),
                0,
                dims.n as u64,
            )?),
            // 헤드: MTP GGUF에 output 없음(kv nextn_shared_target_tensors=1)
            // — (i) 합성 헤드 유지(합성 wout은 이미 등록됨).
            wout: w.wout.clone(),
        };
        // 전문가 스택: gate/up [ne0, n_ff, 512]·down [ne0, n, 512] —
        // 전문가 e = 행 [e·R, (e+1)·R)(R=ne[1]).
        let read_expert = |e: u64| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
            let rgu = dims.n_ff as u64;
            let rd = dims.n as u64;
            let f16r = |name: &str, rows: u64| -> Result<Vec<f32>, String> {
                Ok(rt(gguf_q8_f32(
                    &g,
                    &format!("blk.{il}.{name}"),
                    e * rows,
                    rows,
                )?))
            };
            Ok((
                f16r("ffn_gate_exps.weight", rgu)?,
                f16r("ffn_up_exps.weight", rgu)?,
                f16r("ffn_down_exps.weight", rd)?,
            ))
        };
        setup_module(&mut m, &wr, false)?;
        let mut exps_r: HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)> = HashMap::new();
        let mut registered_r: HashSet<usize> = HashSet::new();
        let exp_fn_r = |e: usize| read_expert(e as u64);
        let mut st4 = OState::new(1, dims.kv_dim());
        let mut hpre = gen_unif(hcn, 0x5EED_F00D_0000_9400, 0.5);
        m.reset(1)?;
        let mut iv_chain = (0.0f32, 0usize);
        for step in 0..2usize {
            let e_k = gen_unif(n, 0x5EED_F00D_0000_9410 + step as u64, 0.02);
            let en = o_rms_norm(&e_k, &wr.enorm, dims.eps);
            let hn = o_rms_norm(&hpre, &wr.hnorm, dims.eps);
            let o = o_draft_step(
                &dims,
                &wr,
                &mut exps_r,
                &exp_fn_r,
                &mut st4,
                &en,
                &hn,
                false,
            )?;
            ensure_experts(&mut m, &exps_r, &mut registered_r)?;
            let got = m.mtp_draft_step(&en, &hn)?;
            m.pos += 1;
            let mut bad: Vec<String> = Vec::new();
            // res2(=pre-MoE 최종)는 비트동일, chain/hin은 문턱.
            let (mdr, bdr, nr) = o_judge(&got.res_attn, &o.res2);
            if bdr != 0 || nr != 0 {
                bad.push(format!("res2={mdr:.3e}/{bdr}b/n{nr}"));
            }
            for (name, gv, want) in [("chain", &got.chain_h, &o.chain), ("hin", &got.hin, &o.hin)] {
                let (md, bd, nan) = o_judge(gv, want);
                if md > MTP_CHAIN_THRESH || nan != 0 {
                    bad.push(format!("{name}={md:.3e}/{bd}b/n{nan}"));
                }
            }
            let tok_ok = got.token.is_some() && got.token == o.token;
            if !tok_ok {
                bad.push("tok".into());
            }
            let (mdc, bdc, _) = o_judge(&got.chain_h, &o.chain);
            iv_chain = (mdc, bdc);
            println!(
                "device: {dev} | mtp-frame (iv) real-GGUF(Q8_0) step{} pos={}: res2={} chain={:.3e}/{}b hin={:.3e} tok={} | {}",
                step + 1,
                st4.pos,
                if o_judge(&got.res_attn, &o.res2).1 == 0 {
                    "bit-exact"
                } else {
                    "BAD"
                },
                mdc,
                bdc,
                o_judge(&got.hin, &o.hin).0,
                got.token.map(|t| t.to_string()).unwrap_or("-".into()),
                if bad.is_empty() { "PASS" } else { "FAIL" }
            );
            if !bad.is_empty() {
                fails.push(format!("(iv step{}) {}", step + 1, bad.join(",")));
            }
            hpre = o.chain.clone();
            st4.pos += 1;
        }
        report.push_str(&format!(
            " · (iv gguf) chain {0:.3e}/{1}b",
            iv_chain.0, iv_chain.1
        ));
    } else {
        println!("device: {dev} | mtp-frame (iv) real-GGUF SKIP — {gguf_path} 없음(선택 픽스처)");
        report.push_str(" · (iv gguf) skip");
    }

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | dims n={n} hc={hc} lr={} heads={}/{} hd={} rot={} moe={}e/top{}/ffn{} vocab={} |{} | ALL PASS (pre-MoE 비트동일 · post-MoE ≤{:.0e})",
            dims.low_rank,
            dims.n_head,
            dims.n_kv,
            dims.head_dim,
            dims.n_rot,
            dims.n_expert,
            dims.n_used,
            dims.n_ff,
            dims.vocab,
            report,
            MTP_CHAIN_THRESH
        ))
    } else {
        Err(format!(
            "mtp-frame 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// mtp-frame-neg — 음성대조 2종(원장 17호: 검증 계기도 스스로 검증).
/// (a) 잘못된 캡처 지점(결함 10호 계급): ffn combine 전 반출 오라클 vs
///     모듈 정답 체인 — maxdiff > MTP_CHAIN_THRESH 여야 NEG-DETECTED.
/// (b) 잘못된 노름 규약(§3.4): EXL3 w−1 저장 규약을 원값 감마에 오적용
///     (등록값 +1)한 **모듈 실경로** vs 정답 오라클 — 임계 초과 검증.
pub fn cuda_mtp_frame_negative_check(dir: &str) -> Result<String, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = MtpFnDims::from_config(&cfg)?;
    let (n, hcn) = (dims.n, dims.hc_dim());
    let mut m = MtpFnCuda::new(dims)?;
    let dev = m.device_name().to_string();

    // 헤드 없음(음성대조는 체인 값 판정 — 프로브 경제성 계약).
    let w = synth_weights(&dims, false);
    let mut exps: HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut registered: HashSet<usize> = HashSet::new();
    let exp_fn =
        |e: usize| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> { Ok(gen_expert(&dims, e)) };
    let mut st = OState::new(1, dims.kv_dim());
    let e_k = gen_unif(n, 0x5EED_F00D_0000_9500, 0.02);
    let h0 = gen_unif(hcn, 0x5EED_F00D_0000_9501, 0.5);
    let en = o_rms_norm(&e_k, &w.enorm, dims.eps);
    let hn = o_rms_norm(&h0, &w.hnorm, dims.eps);

    // 정답 오라클 + 모듈 1스텝.
    let o = o_draft_step(&dims, &w, &mut exps, &exp_fn, &mut st, &en, &hn, false)?;
    setup_module(&mut m, &w, false)?;
    ensure_experts(&mut m, &exps, &mut registered)?;
    m.reset(1)?;
    let got = m.mtp_draft_step(&en, &hn)?;

    // (a) 캡처 지점 오라클 변형(ffn combine 전 반출) vs 모듈 정답 체인.
    let mut sta = OState::new(1, dims.kv_dim());
    let early = o_draft_step(&dims, &w, &mut exps, &exp_fn, &mut sta, &en, &hn, true)?;
    let (mda, bda, na) = o_judge(&got.chain_h, &early.chain);
    println!(
        "device: {dev} | mtp-frame-neg (a) wrong capture point (pre-ffn-combine oracle) vs module chain: maxdiff={mda:.3e} bitdiff={bda}/{hcn} nan={na} | FAIL(expected)"
    );
    let det_a = mda > MTP_CHAIN_THRESH;

    // (b) 노름 규약 오적용(모듈 실경로) — nextn_head norm 등록값 +1
    // (EXL3 §3.4 w−1 저장 규약을 원값 감마에 적용한 계급).
    let mut wb = synth_weights(&dims, false);
    wb.hn_ = wb.hn_.iter().map(|&v| v + 1.0).collect();
    setup_module(&mut m, &wb, false)?;
    m.reset(1)?;
    let got_b = m.mtp_draft_step(&en, &hn)?;
    let (mdb, bdb, nb) = o_judge(&got_b.hin, &o.hin);
    println!(
        "device: {dev} | mtp-frame-neg (b) wrong norm convention (registered gamma+1) vs correct oracle hin: maxdiff={mdb:.3e} bitdiff={bdb}/{n} nan={nb} | FAIL(expected)"
    );
    let det_b = mdb > MTP_CHAIN_THRESH;

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) capture maxdiff={mda:.3e} (b) norm maxdiff={mdb:.3e} > {MTP_CHAIN_THRESH:.0e} — 검증계기 정상(캡처 지점·노름 규약 편차 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={mda:.3e} (b)={mdb:.3e} <= {MTP_CHAIN_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
