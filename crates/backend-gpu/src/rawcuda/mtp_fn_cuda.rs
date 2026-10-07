//! Flash-Next MTP 드래프트 프레임 CUDA 모듈층 — plans/124 G001(FNA)→FNG,
//! 2026-10-05.
//!
//! [계약] 산술 원천은 crates/core/src/qwen4exp/frame/mtp.rs
//! (MtpFrame + mtp_draft_frame — CPU 황금 계약)이다. 체인:
//!   eh_proj(en‖hn) → hc_mix(attn) → dense 게이트 어텐션(자체 KV) →
//!   hc_combine → hc_mix(ffn) → MoE → hc_combine → nextn.hc_head 믹서 →
//!   output 헤드 → argmax · 체인 h(pre-mixer 잔차) 반출.
//! 입력 en[n]·hn[hc·n]은 호출부가 enorm/hnorm으로 정규화해 전달
//! (core와 동일 계약 — layers.rs mtp_draft_step_h L1104-1128). pos 진행도
//! 호출부 담당(core mtp.rs 헤드 "mtp_st.pos 진행은 호출부가 담당(KV 기입은
//! 내부 pos 기준)" — 본 모듈의 mtp_draft_step는 KV 기입만 내부 pos 기준).
//!
//! [조립 — 기착 스테이지 재사용(신규 커널은 2개뿔)]
//! - hc 믹서 3종: hc_cuda.rs HcCuda(FNC) — hc_mix(attn/ffn)·
//!   hc_mix_nextn_head 그대로 호출(비트동일 원장 승계).
//! - MoE(라우팅·전문가·shared·결합): moe_cuda.rs MoeCuda(FNE) — moe_ffn.
//! - 선형 GEMV 6종(eh_proj·q·k·v·o·output 헤드): llm170_fn_hc_gemv
//!   (assets/exl3_fn_hc.fatbin — cpu.rs matmul L64-76 순차 f32 누산 미러).
//! - hc_combine·argmax: assets/exl3_fn_mtp_frame.fatbin(본 목표 신규).
//! - dense 어텐션의 norm/rope/KV/softmax/게이트: **CPU 유지** — core
//!   프레임 경로 계약(frame/mtp.rs 머리 "norm/rope/KV/softmax는 CPU 유지,
//!   layers.rs mtp_attn_cpu_row 공유"). 본 파일의 mtp_attn_cpu_row 미러는
//!   layers.rs L2505-2573의 직이식(softamx exp만 Rust std f32 exp —
//!   원본과 동일 libm 경로).
//!
//! [메모리 규약] output 헤드 [vocab][n] f32(실측 248320×2560×4B ≈ 2.5GB)는
//! core와 동일하게 상주(등록 1회). set_head 미등록(0)이면 헤드 스텝을
//! 건너뛴다(검증층 음성대조 경제성 계약 — 체인 값만 판정하는_arm은 헤드
//! 없이 구동; FNH 통합 시에는 항상 등록). 대형 h2d는 4MB 청크 분할
//! (G2 원장 패턴 — moe_cuda h2d_chunked와 동일).
//!
//! [정합 목표 — 비트동일(pre-MoE 전 스테이지) + 문서화 문턱(MoE 이후)]
//! eh/믹서/어텐션/combine 구간은 core 미러와 비트동일(FNC 커널·본 파일
//! 커널의 연산별 반올림 미러). MoE 결합만 누산 순서 차이: core 프레임의
//! MoeWeightedSum은 **선택 순서(확률 내림)** 누산(src_q4.hip L1605-1667
//! q4_moe_top10_m 선택 순서 기록 + q4_moe_weighted_sum k순 누산)이고
//! MoeCuda 결합은 **e-오름차**(moe.rs 값경로 t=1 fast "for (k,&e) in
//! sel.iter()" 와 동일 순서 — FNE 비트동일 원장). 수학적 동치·f32
//! 반올림 순서만 상이 → 체인 종단은 문서화 문턱(plans/124 §3.4 ≤2e-4)
//! 으로 판정(검증층 mtp_fn_cuda_probe.rs 원장).
//!
//! [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
//! 드래프트 1스텝 지배 비용 = 헤드 GEMV 2.5GB 스트리밍(HBM2e 하한
//! ~1.7ms)·MoE 전문가 스트리밍(전문가당 3.3MB×선택 10). 본 모듈의
//! 스테이지 간 호스트 스테이징은 plans/124 §4.13(디바이스 체인화)의
//! FNH 목표 과제 — 현재는 정합 우선(속도 칸 '측정 대기 sm_80', 개발기
//! 타이밍은 판단 근거 아님).
//!
//! 단일 상주 원칙(2026-10-04 동결 사고): HcCuda·MoeCuda·본 모듈의
//! CudaCtx는 모두 프라이머리 컨텍스트(ctx.rs new 계약)라 한 프로세스에서
//! 직렬 공존 — 모델 1개 계약 유지.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 —
//! scripts/cuda_probe_shim.rs 단독 컴파일 대상.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;
use crate::rawcuda::hc_cuda::{HcCuda, HcDims};
use crate::rawcuda::moe_cuda::{MoeCuda, MoeDims};

/// 대형 h2d 청크 상한(4MB — moe_cuda G2 원장 패턴 동일).
const H2D_CHUNK: usize = 4 << 20;

/// MTP 드래프트 KV 캐시 위치 상한(호스트 상태 — core SeqState4.kv_* 계급).
pub const MTP_KV_CAP: usize = 1024;

/// Flash-Next MTP 드래프트 형상 — config.json text_config에서 유도
/// (HcDims/MtpDims 방식 계승 — 형상은 명시 등록, 추정 금지).
/// 실측 기준(2026-10-05, D:/models/Qwen3.8-Flash-Next-exl3-5.05bpw/
/// config.json): n=2560·hc=4·lr=320·24헤드/2KV·hd=256·n_rot=64·
/// rope 1e7·512e top10·ffn 640/640·vocab 248320.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpFnDims {
    /// 잔류 폭(hidden_size).
    pub n: usize,
    /// hc 스트림 수(hc_count).
    pub hc: usize,
    /// hc 저랭크 폭(hc_lowrank).
    pub low_rank: usize,
    /// RMSNorm eps(rms_norm_eps).
    pub eps: f32,
    /// q헤드 수(num_attention_heads).
    pub n_head: usize,
    /// KV헤드 수(num_key_value_heads).
    pub n_kv: usize,
    /// 헤드 폭(head_dim).
    pub head_dim: usize,
    /// 회전 차원(head_dim·partial_rotary_factor — 실측 64).
    pub n_rot: usize,
    /// rope theta(mtp.rope_theta — 실측 1e7, 전 모델 공통).
    pub rope_base: f32,
    /// 전문가 수(num_experts).
    pub n_expert: usize,
    /// 토큰당 전문가(num_experts_per_tok).
    pub n_used: usize,
    /// 전문가 FFN 폭(moe_intermediate_size).
    pub n_ff: usize,
    /// shared FFN 폭(shared_expert_intermediate_size).
    pub n_ff_sh: usize,
    /// 어휘(로짓 길이 — 결함 8호: argmax n).
    pub vocab: usize,
}

impl MtpFnDims {
    /// hc·n 형태.
    pub fn hc_dim(&self) -> usize {
        self.hc * self.n
    }
    /// q‖gate 인터리브 폭(n_head·2·head_dim — q행 레이아웃 §3.4).
    pub fn qg_dim(&self) -> usize {
        self.n_head * 2 * self.head_dim
    }
    /// q/어텐션 출력 폭(n_head·head_dim).
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    /// k/v 폭(n_kv·head_dim).
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
    /// kq 스케일 1/√head_dim — 원천: mod.rs Hparams4 L104-106.
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// config.json 본문 → 형상. 키 누락은 Err(형상 추정 금지 — 결함 1호).
    pub fn from_config(cfg: &str) -> Result<Self, String> {
        let v = JParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let tc = v.get("text_config").unwrap_or(&v);
        let num = |k: &str| {
            tc.get(k)
                .and_then(JVal::as_f64)
                .ok_or_else(|| format!("config.json: text_config.{k} 없음"))
        };
        let n = num("hidden_size")? as usize;
        let hc = num("hc_count")? as usize;
        let low_rank = num("hc_lowrank")? as usize;
        let eps = num("rms_norm_eps")? as f32;
        let n_head = num("num_attention_heads")? as usize;
        let n_kv = num("num_key_value_heads")? as usize;
        let head_dim = num("head_dim")? as usize;
        let pfr = tc
            .get("partial_rotary_factor")
            .and_then(JVal::as_f64)
            .or_else(|| {
                tc.get("rope_parameters")
                    .and_then(|rp| rp.get("partial_rotary_factor"))
                    .and_then(JVal::as_f64)
            })
            .ok_or("config.json: partial_rotary_factor 없음")?;
        let rope_base = tc
            .get("mtp")
            .and_then(|m| m.get("rope_theta"))
            .and_then(JVal::as_f64)
            .or_else(|| {
                tc.get("rope_parameters")
                    .and_then(|rp| rp.get("rope_theta"))
                    .and_then(JVal::as_f64)
            })
            .ok_or("config.json: rope_theta 없음")? as f32;
        let n_expert = num("num_experts")? as usize;
        let n_used = num("num_experts_per_tok")? as usize;
        let n_ff = num("moe_intermediate_size")? as usize;
        let n_ff_sh = num("shared_expert_intermediate_size")? as usize;
        let vocab = num("vocab_size")? as usize;
        // n_rot = head_dim·pfr — GGUF qwen4exp.rope.dimension_count와 동치
        // (실측 256·0.25=64).
        let n_rot_f = head_dim as f64 * pfr;
        if (n_rot_f - n_rot_f.round()).abs() > 1e-9 || n_rot_f < 2.0 {
            return Err(format!(
                "mtp-frame: n_rot={n_rot_f} — 2 이상 정수 계약(head_dim·partial_rotary_factor)"
            ));
        }
        let n_rot = n_rot_f as usize;
        if n_rot % 2 != 0 {
            return Err(format!("mtp-frame: n_rot={n_rot} — 짝수 계약(rope 페어링)"));
        }
        if hc == 0 || hc > 8 || n == 0 || n % 32 != 0 || low_rank == 0 || eps <= 0.0 {
            return Err(format!(
                "mtp-frame: n={n} hc={hc} lr={low_rank} eps={eps} — HcDims 도메인 위반"
            ));
        }
        if n_head == 0 || n_kv == 0 || n_head % n_kv != 0 {
            return Err(format!(
                "mtp-frame: n_head={n_head} n_kv={n_kv} — q%kv==0 계약(GQA)"
            ));
        }
        if head_dim == 0 || head_dim % n_rot != 0 {
            return Err(format!(
                "mtp-frame: head_dim={head_dim} n_rot={n_rot} — dim 배수 계약"
            ));
        }
        if n_expert == 0 || n_used == 0 || n_ff == 0 || n_ff_sh == 0 || vocab == 0 {
            return Err("mtp-frame: MoE/vocab 0 형상".into());
        }
        Ok(MtpFnDims {
            n,
            hc,
            low_rank,
            eps,
            n_head,
            n_kv,
            head_dim,
            n_rot,
            rope_base,
            n_expert,
            n_used,
            n_ff,
            n_ff_sh,
            vocab,
        })
    }

    /// HcCuda용 서브 형상.
    pub fn hc_dims(&self) -> HcDims {
        HcDims {
            hc: self.hc,
            n_embd: self.n,
            low_rank: self.low_rank,
            eps: self.eps,
        }
    }
    /// MoeCuda용 서브 형상.
    pub fn moe_dims(&self) -> MoeDims {
        MoeDims {
            n_embd: self.n,
            n_expert: self.n_expert,
            n_used: self.n_used,
            n_ff: self.n_ff,
            n_ff_sh: self.n_ff_sh,
        }
    }
}

/// MTP 드래프트 스텝 중간 산출(검증층 단계별 값 판정용 — G9 MtpMids 노선:
/// 호출은 검증층, 상태는 모듈층 소유).
pub struct MtpMids {
    /// eh_proj 산출 = 프리-믹서 잔차 [hc·n].
    pub eh: Vec<f32>,
    /// attn 반쪽 hc_mix 산출 mixed [n]·inject [hc].
    pub mix_attn: Vec<f32>,
    pub inj_attn: Vec<f32>,
    /// dense 어텐션 출력 [n_head·head_dim](wo 입력).
    pub attn: Vec<f32>,
    /// wo 출력 [n](첫 combine 입력).
    pub ao: Vec<f32>,
    /// attn combine 직후 잔차 [hc·n].
    pub res_attn: Vec<f32>,
    /// ffn 반쪽 hc_mix 산출.
    pub mix_ffn: Vec<f32>,
    pub inj_ffn: Vec<f32>,
    /// MoE 산출(전문가+shared 결합) [n].
    pub mout: Vec<f32>,
    /// 체인 h = ffn combine 직후 잔차(pre-mixer — core mtp.rs 종착
    /// "체인 h 반출" 계약, layers.rs "체인 반출 = pre-mix 멀티 스트림").
    pub chain_h: Vec<f32>,
    /// nextn.hc_head 믹서 출력 [n](헤드 입력).
    pub hin: Vec<f32>,
    /// 로짓 [vocab](헤드 미등록 시 빈 벡터).
    pub logits: Vec<f32>,
    /// greedy 토큰(헤드 미등록 시 None).
    pub token: Option<u32>,
}

/// Flash-Next MTP 드래프트 프레임 모듈 — HcCuda(FNC)·MoeCuda(FNE) 조립 +
/// 자체 GEMV 6종(llm170_fn_hc_gemv)·combine/argmax(exl3_fn_mtp_frame.fatbin).
pub struct MtpFnCuda {
    /// hc 믹서 서브모듈(FNC) — 믹서 등록(register)도 검증층에서 직접.
    pub hc: HcCuda,
    /// MoE 서브모듈(FNE) — 라우터·shared·전문가 등록도 검증층에서 직접.
    pub moe: MoeCuda,
    cc: CudaCtx,
    /// 형상(new 등록).
    pub dims: MtpFnDims,
    // ── 상주 가중치(f32 행우선 [n_out][n_in]) ──
    d_eh: CUdeviceptr,   // eh_proj [n][2n]
    d_wq: CUdeviceptr,   // attn_q [qg_dim][n]
    d_wk: CUdeviceptr,   // attn_k [kv_dim][n]
    d_wv: CUdeviceptr,   // attn_v [kv_dim][n]
    d_wo: CUdeviceptr,   // attn_output [n][q_dim]
    d_qn: CUdeviceptr,   // attn_q_norm [head_dim]
    d_kn: CUdeviceptr,   // attn_k_norm [head_dim]
    d_wout: CUdeviceptr, // output 헤드 [vocab][n] — 0이면 미등록(헤드 스킵)
    // ── 드래프트 KV(호스트 — core 프레임 경로 계약: dense 어텐션 CPU) ──
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    /// 드래프트 시퀀스 위치(진행은 호출부 — core mtp.rs 머리 계약).
    pub pos: usize,
    // ── 작업 버퍼(생성 시 고정 형상) ──
    d_cat: CUdeviceptr,    // [hc][2n]
    d_res: CUdeviceptr,    // [hc·n]
    d_q: CUdeviceptr,      // [qg_dim]
    d_k: CUdeviceptr,      // [kv_dim]
    d_v: CUdeviceptr,      // [kv_dim]
    d_attn: CUdeviceptr,   // [q_dim]
    d_ao: CUdeviceptr,     // [n]
    d_inj: CUdeviceptr,    // [hc]
    d_x: CUdeviceptr,      // [n] GEMV 입력 스테이징
    d_logits: CUdeviceptr, // [vocab]
    d_tok: CUdeviceptr,    // [1] u32
}

/// f32 슬라이스 → 바이트 뷰(LE 호스트 — 복사 없이 h2d 직결, 대형 헤드용).
/// SAFETY: 호출자 슬라이스 수명 내 유효한 재해석(정렬·길이 일치).
fn f32_view(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// ops.rs rms_norm(L33-37) 직이식 — 32세그먼트 f32 누산 → f64 결합.
/// 모듈 내부용(어텐션 q/k 노름·프로브는 검증층 자체 미러 사용).
fn core_sq_sum(x: &[f32]) -> f64 {
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
fn core_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = core_sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// ops.rs rope_head(L141-159) 직이식 — NEOX 페어링(p, p+half), f64 중간.
fn core_rope_head(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
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

/// MTP dense 어텐션 CPU 코어 — layers.rs mtp_attn_cpu_row(L2505-2573)
/// 직이식. q/k/v 원시 행 1개에 norm·rope·KV 적립·softmax(**cell 0 스킵**
/// — 팬텀 0키 결함 P15④-5)·게이트(sigmoid)를 적용해 [n_head·head_dim]
/// 반환. KV 캐시(kv_k/kv_v, [MTP_KV_CAP][kv_dim])는 호출부 소유 — core가
/// SeqState4를 받는 것과 동일 계약.
fn mtp_attn_cpu_row(
    dims: &MtpFnDims,
    q_row: &mut [f32],
    k_row: &mut [f32],
    v_row: &[f32],
    kv_k: &mut [f32],
    kv_v: &mut [f32],
    pos: usize,
    qn: &[f32],
    kn: &[f32],
) -> Result<Vec<f32>, String> {
    let (n_head, n_kv, hd, n_rot) = (dims.n_head, dims.n_kv, dims.head_dim, dims.n_rot);
    let kq_scale = dims.kq_scale();
    if pos >= MTP_KV_CAP {
        return Err(format!("mtp-frame: pos={pos} >= KV cap {MTP_KV_CAP}"));
    }
    // q: 헤드별 norm+rope(전반 hd) — 게이트 후반은 미가공(L2512-2519).
    for h in 0..n_head {
        let lo = h * 2 * hd;
        let mut qh = core_rms_norm(&q_row[lo..lo + hd], qn, dims.eps);
        core_rope_head(&mut qh, pos as u32, n_rot, dims.rope_base);
        q_row[lo..lo + hd].copy_from_slice(&qh);
    }
    // k: kv헤드별 norm+rope → 캐시 적립. v: 원문 그대로(L2520-2526).
    let kbase = pos * n_kv * hd;
    for h in 0..n_kv {
        let lo = h * hd;
        let mut kh = core_rms_norm(&k_row[lo..lo + hd], kn, dims.eps);
        core_rope_head(&mut kh, pos as u32, n_rot, dims.rope_base);
        kv_k[kbase + lo..kbase + lo + hd].copy_from_slice(&kh);
    }
    kv_v[kbase..kbase + n_kv * hd].copy_from_slice(v_row);
    // dense softmax 어텐션 + 게이트 — cell 0 스킵(p=1..=pos, L2527-2555).
    // scores의 exp는 원본과 동일 Rust std f32 exp(libm).
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
                d += q_row[h * 2 * hd + i] * kv_k[b + i];
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
                out[ob + i] += w * kv_v[b + i];
            }
        }
        // 게이트: 후반 hd sigmoid 원소 곱(L2556-2559 — ops.rs sigmoid
        // L131-133: 1/(1+exp_cr(−x))).
        let gb = h * 2 * hd + hd;
        for i in 0..hd {
            let e = exp_cr(-q_row[gb + i]);
            out[ob + i] *= 1.0 / (1.0 + e);
        }
    }
    Ok(out)
}

/// ops.rs exp_cr(L63-119) 직이식 — f64 fma 호너 13단(커널 mtp_exp_cr과
/// 리터럴까지 동일). 어텐션 게이트 sigmoid에 사용(원본 ops::sigmoid 경로).
pub(crate) fn exp_cr(x: f32) -> f32 {
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

impl MtpFnCuda {
    /// exl3_fn_mtp_frame.fatbin 자산 해석 — LLM170_CUDA_FN_MTP_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn mtp_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_FN_MTP_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_fn_mtp_frame.fatbin",
            "src/rawcuda/assets/exl3_fn_mtp_frame.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_fn_mtp_frame.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// exl3_fn_hc.fatbin 바이트(FNC gemv 재사용 — 경로 탐색은 FNC와 동일).
    fn hc_fatbin_bytes() -> Result<Vec<u8>, String> {
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_fn_hc.fatbin",
            "src/rawcuda/assets/exl3_fn_hc.fatbin",
        ];
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_fn_hc.fatbin 없음 — scripts/build_cuda.bat 실행 (탐색: {REL:?})"
        ))
    }

    /// 개방 — 자체 ctx(FNC gemv + 본 목표 fatbin) + HcCuda·MoeCuda 서브모듈.
    pub fn new(dims: MtpFnDims) -> Result<Self, String> {
        let (n, hc) = (dims.n, dims.hc);
        let (hcn, qg, kvd, qd) = (dims.hc_dim(), dims.qg_dim(), dims.kv_dim(), dims.q_dim());
        let image = Self::mtp_fatbin_bytes()?;
        let hc_img = Self::hc_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let (d_cat, d_res, d_q, d_k, d_v, d_attn, d_ao, d_inj, d_x, d_logits, d_tok);
        {
            let _g = cc.guard()?;
            cc.load_fatbin("exl3fnhc", &hc_img, &["llm170_fn_hc_gemv"])?;
            cc.load_fatbin(
                "exl3fnmtp",
                &image,
                &["llm170_fn_mtp_combine", "llm170_fn_mtp_argmax"],
            )?;
            d_cat = cc.alloc(hc * 2 * n * 4)?;
            d_res = cc.alloc(hcn * 4)?;
            d_q = cc.alloc(qg * 4)?;
            d_k = cc.alloc(kvd * 4)?;
            d_v = cc.alloc(kvd * 4)?;
            d_attn = cc.alloc(qd * 4)?;
            d_ao = cc.alloc(n * 4)?;
            d_inj = cc.alloc(hc * 4)?;
            d_x = cc.alloc(n * 4)?;
            d_logits = cc.alloc(dims.vocab * 4)?;
            d_tok = cc.alloc(4)?;
        }
        let kv_len = MTP_KV_CAP * kvd;
        Ok(MtpFnCuda {
            hc: HcCuda::new(dims.hc_dims())?,
            moe: MoeCuda::new(dims.moe_dims())?,
            cc,
            dims,
            d_eh: 0,
            d_wq: 0,
            d_wk: 0,
            d_wv: 0,
            d_wo: 0,
            d_qn: 0,
            d_kn: 0,
            d_wout: 0,
            kv_k: vec![0.0; kv_len],
            kv_v: vec![0.0; kv_len],
            pos: 0,
            d_cat,
            d_res,
            d_q,
            d_k,
            d_v,
            d_attn,
            d_ao,
            d_inj,
            d_x,
            d_logits,
            d_tok,
        })
    }

    /// 디바이스 이름(프로브 보고용).
    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// 대형 h2d 청크 분할 업로드(G2 원장 패턴 미러 — 4MB 청크).
    fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        let mut off = 0usize;
        while off < src.len() {
            let hi = (off + H2D_CHUNK).min(src.len());
            // SAFETY: src는 호출자 소유 슬라이스 — 부분 슬라이스는 호출 내 유효.
            let part = unsafe { std::slice::from_raw_parts(src.as_ptr().add(off), hi - off) };
            cc.h2d(dst + off as u64, part)?;
            off = hi;
        }
        Ok(())
    }

    /// 가중치 슬롯 교체(기존 있으면 해제 후 재할당·업로드 — moe replace_w 패턴).
    fn replace_w(slot: &mut CUdeviceptr, cc: &CudaCtx, bytes: &[u8]) -> Result<(), String> {
        if *slot != 0 {
            // SAFETY: 이전 alloc 산출물 — 교체 시 1회 해제.
            cc.free(*slot)?;
        }
        let p = cc.alloc(bytes.len())?;
        Self::h2d_chunked(cc, p, bytes)?;
        *slot = p;
        Ok(())
    }

    /// eh_proj 등록 — f32 행우선 [n][2n](GGUF blk.48.nextn.eh_proj
    /// [5120,2560] = n_in 5120·n_out 2560와 동일 레이아웃).
    pub fn set_eh_proj(&mut self, w: &[f32]) -> Result<(), String> {
        let want = self.dims.n * 2 * self.dims.n;
        if w.len() != want {
            return Err(format!("mtp-frame: eh_proj {} != n·2n {want}", w.len()));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.d_eh, &self.cc, f32_view(w))
    }

    /// dense 어텐션 투영 등록 — q [qg_dim][n]·k/v [kv_dim][n]·o [n][q_dim]
    /// (q‖gate 인터리브 계약 §3.4) + q/k 노름 가중 [head_dim] 원값.
    pub fn set_attn(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        o: &[f32],
        qn: &[f32],
        kn: &[f32],
    ) -> Result<(), String> {
        let d = &self.dims;
        let (qg, kvd, qd, n, hd) = (d.qg_dim(), d.kv_dim(), d.q_dim(), d.n, d.head_dim);
        if q.len() != qg * n {
            return Err(format!("mtp-frame: attn_q {} != {qg}×{n}", q.len()));
        }
        if k.len() != kvd * n || v.len() != kvd * n {
            return Err(format!(
                "mtp-frame: attn_k/v {}/{} != {kvd}×{n}",
                k.len(),
                v.len()
            ));
        }
        if o.len() != n * qd {
            return Err(format!("mtp-frame: attn_o {} != {n}×{qd}", o.len()));
        }
        if qn.len() != hd || kn.len() != hd {
            return Err(format!(
                "mtp-frame: qn/kn {}/{} != head_dim {hd}",
                qn.len(),
                kn.len()
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.d_wq, &self.cc, f32_view(q))?;
        Self::replace_w(&mut self.d_wk, &self.cc, f32_view(k))?;
        Self::replace_w(&mut self.d_wv, &self.cc, f32_view(v))?;
        Self::replace_w(&mut self.d_wo, &self.cc, f32_view(o))?;
        Self::replace_w(&mut self.d_qn, &self.cc, f32_view(qn))?;
        Self::replace_w(&mut self.d_kn, &self.cc, f32_view(kn))?;
        Ok(())
    }

    /// output 헤드 등록 — f32 행우선 [vocab][n](본체 공유 — MTP GGUF에는
    /// 없음, kv nextn_shared_target_tensors=1 실측). 미등록 시 헤드 스킵.
    pub fn set_head(&mut self, wout: &[f32]) -> Result<(), String> {
        let want = self.dims.vocab * self.dims.n;
        if wout.len() != want {
            return Err(format!("mtp-frame: head {} != vocab·n {want}", wout.len()));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.d_wout, &self.cc, f32_view(wout))
    }

    /// 드래프트 시퀀스 리셋 — KV 제로화 + pos 설정(원장 19호: KV는 시퀀스
    /// 상태 — 재사용 루프는 문맥을 꼬는 사고 계급. 비영 시딩은 §3.3 정신).
    pub fn reset(&mut self, pos: usize) -> Result<(), String> {
        if pos >= MTP_KV_CAP {
            return Err(format!("mtp-frame: reset pos={pos} >= cap {MTP_KV_CAP}"));
        }
        self.kv_k.iter_mut().for_each(|v| *v = 0.0);
        self.kv_v.iter_mut().for_each(|v| *v = 0.0);
        self.pos = pos;
        Ok(())
    }

    /// GEMV 1회 — llm170_fn_hc_gemv(t행 배치, FNC 재사용). 호출부 guard 내.
    fn gemv(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        out: CUdeviceptr,
        t: usize,
        n_in: usize,
        n_out: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("llm170_fn_hc_gemv")?;
        let (mut a0, mut a1, mut a2) = (x, w, out);
        let (mut ni, mut no) = (n_in as i32, n_out as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut ni) as *mut _ as *mut _,
            (&mut no) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, n_out.div_ceil(128) as u32, t as u32, 128, &mut args)
    }

    /// hc_combine 1회 — llm170_fn_mtp_combine(res 제자리 += out·2σ(inj/hc)).
    fn combine(&self, out: CUdeviceptr) -> Result<(), String> {
        let (hcn, hc) = (self.dims.hc_dim(), self.dims.hc);
        let f = self.cc.function("llm170_fn_mtp_combine")?;
        let (mut a0, mut a1, mut a2) = (self.d_res, out, self.d_inj);
        let (mut nn, mut nh) = (self.dims.n as i32, hc as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
            (&mut nh) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, hcn.div_ceil(256) as u32, 1, 256, &mut args)
    }

    /// f32 버퍼 d2h.
    fn read_f32(&self, src: CUdeviceptr, n: usize) -> Result<Vec<f32>, String> {
        let mut b = vec![0u8; n * 4];
        self.cc.d2h(&mut b, src)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n) }.to_vec())
    }

    /// MTP 드래프트 스텝 — core frame/mtp.rs mtp_draft_frame(L362-484) 미러.
    /// 순서 계약: eh_proj(t=hc 배치) → hc_mix(attn) → 투영 q/k/v → CPU
    /// 어텐션行 → wo → combine → hc_mix(ffn) → MoE → combine →
    /// nextn.hc_head → output GEMV + argmax → 체인 h 반출.
    /// pos 진행은 호출부(머리 계약). 헤드 미등록이면 logits/token 생략.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_draft_step(&mut self, en: &[f32], hn: &[f32]) -> Result<MtpMids, String> {
        let d = self.dims;
        let (n, hc, hcn) = (d.n, d.hc, d.hc_dim());
        let (qg, kvd, qd) = (d.qg_dim(), d.kv_dim(), d.q_dim());
        if en.len() != n {
            return Err(format!("mtp-frame: en.len={} != n {n}", en.len()));
        }
        if hn.len() != hcn {
            return Err(format!("mtp-frame: hn.len={} != hc·n {hcn}", hn.len()));
        }
        if self.d_eh == 0 || self.d_wq == 0 || self.d_wo == 0 || self.d_qn == 0 {
            return Err("mtp-frame: eh_proj/attn 미등록 — set_eh_proj·set_attn 먼저".into());
        }
        let _g = self.cc.guard()?;

        // 1) eh_proj — cat[s]=[en ‖ hn_s] 조립 후 t=hc 배치 GEMM 1호출
        //    (core mtp.rs L388-396과 동일 호스트 조립·동일 1호출 기하).
        let mut cat = vec![0.0f32; hc * 2 * n];
        for s in 0..hc {
            let b = s * 2 * n;
            cat[b..b + n].copy_from_slice(en);
            cat[b + n..b + 2 * n].copy_from_slice(&hn[s * n..(s + 1) * n]);
        }
        self.cc.h2d(self.d_cat, f32_view(&cat))?;
        self.gemv(self.d_cat, self.d_eh, self.d_res, hc, 2 * n, n)?;
        let eh = self.read_f32(self.d_res, hcn)?;

        // 2) attn 반쪽 hc_mix — HcCuda(FNC) 재사용(t=1). 반환은 토큰 행
        //    리스트 — 단일 행(스텝 계약)을 소출한다.
        let (mix_attn_rows, inj_attn_rows) = self.hc.hc_mix(0, "attn", &[eh.clone()])?;
        let mix_attn = mix_attn_rows
            .into_iter()
            .next()
            .ok_or("mtp-frame: hc_mix(attn) 빈 반환")?;
        let inj_attn = inj_attn_rows
            .into_iter()
            .next()
            .ok_or("mtp-frame: hc_mix(attn) inject 빈 반환")?;

        // 3) dense 어텐션 — 투영 GPU(3종)·CPU 코어(프레임 경로 계약)·wo GPU.
        let mix_b = f32_view(&mix_attn);
        self.cc.h2d(self.d_x, &mix_b)?;
        self.gemv(self.d_x, self.d_wq, self.d_q, 1, n, qg)?;
        self.gemv(self.d_x, self.d_wk, self.d_k, 1, n, kvd)?;
        self.gemv(self.d_x, self.d_wv, self.d_v, 1, n, kvd)?;
        let mut q = self.read_f32(self.d_q, qg)?;
        let mut k = self.read_f32(self.d_k, kvd)?;
        let v = self.read_f32(self.d_v, kvd)?;
        let qn = self.read_f32(self.d_qn, d.head_dim)?;
        let kn = self.read_f32(self.d_kn, d.head_dim)?;
        // SAFETY: kv_k/kv_v는 정확히 kv 행 길이 — &mut 분리 대여는 인덱스
        // 분할로 보장(두 슬라이스는 서로 다른 Vec).
        let attn = mtp_attn_cpu_row(
            &self.dims,
            &mut q,
            &mut k,
            &v,
            &mut self.kv_k,
            &mut self.kv_v,
            self.pos,
            &qn,
            &kn,
        )?;
        self.cc.h2d(self.d_attn, f32_view(&attn))?;
        self.gemv(self.d_attn, self.d_wo, self.d_ao, 1, qd, n)?;
        let ao = self.read_f32(self.d_ao, n)?;
        self.cc.h2d(self.d_inj, f32_view(&inj_attn))?;
        self.combine(self.d_ao)?;
        let res_attn = self.read_f32(self.d_res, hcn)?;

        // 4) ffn 반쪽 — hc_mix + MoE(FNE) + combine.
        let (mix_ffn_rows, inj_ffn_rows) = self.hc.hc_mix(0, "ffn", &[res_attn.clone()])?;
        let mix_ffn = mix_ffn_rows
            .into_iter()
            .next()
            .ok_or("mtp-frame: hc_mix(ffn) 빈 반환")?;
        let inj_ffn = inj_ffn_rows
            .into_iter()
            .next()
            .ok_or("mtp-frame: hc_mix(ffn) inject 빈 반환")?;
        let moe_rows = self.moe.moe_ffn(&[mix_ffn.clone()])?;
        let mout = moe_rows
            .into_iter()
            .next()
            .ok_or("mtp-frame: moe_ffn 빈 반환")?;
        self.cc.h2d(self.d_ao, f32_view(&mout))?;
        self.cc.h2d(self.d_inj, f32_view(&inj_ffn))?;
        self.combine(self.d_ao)?;
        let chain_h = self.read_f32(self.d_res, hcn)?;

        // 5) 헤드 — nextn.hc_head 믹서(HcCuda) → output GEMV + argmax.
        let hin_rows = self.hc.hc_mix_nextn_head(0, &[chain_h.clone()])?;
        let hin = hin_rows
            .into_iter()
            .next()
            .ok_or("mtp-frame: hc_mix_nextn_head 빈 반환")?;
        let (logits, token) = if self.d_wout != 0 {
            self.cc.h2d(self.d_x, f32_view(&hin))?;
            self.gemv(self.d_x, self.d_wout, self.d_logits, 1, n, d.vocab)?;
            let f = self.cc.function("llm170_fn_mtp_argmax")?;
            let (mut a0, mut a1, mut a2) = (self.d_logits, d.vocab as i32, self.d_tok);
            let mut args: [*mut std::ffi::c_void; 3] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
            ];
            self.cc.launch(f, 1, 1, 32, &mut args)?;
            let mut tb = [0u8; 4];
            self.cc.d2h(&mut tb, self.d_tok)?;
            let lg = self.read_f32(self.d_logits, d.vocab)?;
            (lg, Some(u32::from_le_bytes(tb)))
        } else {
            (Vec::new(), None)
        };
        Ok(MtpMids {
            eh,
            mix_attn,
            inj_attn,
            attn,
            ao,
            res_attn,
            mix_ffn,
            inj_ffn,
            mout,
            chain_h,
            hin,
            logits,
            token,
        })
    }
}

impl Drop for MtpFnCuda {
    fn drop(&mut self) {
        // SAFETY: 각 포인터는 이 ctx의 alloc 산출물이며 drop에서 1회 해제.
        let r = (|| {
            let _g = self.cc.guard();
            for p in [
                &self.d_eh,
                &self.d_wq,
                &self.d_wk,
                &self.d_wv,
                &self.d_wo,
                &self.d_qn,
                &self.d_kn,
                &self.d_wout,
                &self.d_cat,
                &self.d_res,
                &self.d_q,
                &self.d_k,
                &self.d_v,
                &self.d_attn,
                &self.d_ao,
                &self.d_inj,
                &self.d_x,
                &self.d_logits,
                &self.d_tok,
            ] {
                if *p != 0 {
                    self.cc.free(*p)?;
                }
            }
            Ok::<(), String>(())
        })();
        if let Err(e) = r {
            eprintln!("mtp_fn_cuda: drop 해제 실패: {e}");
        }
    }
}
