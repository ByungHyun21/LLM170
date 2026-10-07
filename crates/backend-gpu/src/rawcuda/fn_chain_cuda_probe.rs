//! Flash-Next(Qwen4-Expert) 그리디 디코드 체인 프로브 — plans/124
//! G001(FNA)→FNH, 2026-10-05.
//!
//! [용도] 착지 스테이지(ple·hc·qsa·moe·gdn + FNG mtp_frame)을 **core
//! frame/forward.rs frame_forward_ex의 층 스케줄 그대로** 조립해 T=3
//! 스텝(프리필 t=2 + 디코드 t=1×2) 그리디 체인 + **MTP 스펙 라운드**
//! (드래프트 k=3 → 트렁크 검증 t=2 배치 → 수용/기각 — layers.rs
//! mtp_spec_step L288-386·frame/verify.rs·frame/mtp.rs 계약)을 돌고,
//! 임베딩 방송 → PLE → hc attn mix → GDN|QSA → hc combine → hc ffn mix
//! → MoE → hc combine → output hc head → logits 슬라이스 → argmax 대
//! **코어 CPU 체인 미러 오라클**을 값 판정한다.

//! ═══ MTP 스펙 라운드 계약(확장부 — 원천 core 3파일) ═══
//! - 드래프트: MtpFnCuda(FNG) mtp_draft_step — core frame/mtp.rs
//!   mtp_draft_frame L362-484(eh_proj cat[t=hc] → hc attn mix → dense
//!   게이트드 어텐션(자체 KV·cell 0 스킵 — layers.rs mtp_attn_cpu_row
//!   L2505-2573) → wo → combine → ffn mix → MoE → combine → nextn.hc_head
//!   → output GEMV + argmax → 체인 h 반출). 입력 en=rms_norm(emb,
//!   nextn.enorm)[n]·hn=rms_norm(h_pre, nextn.hnorm)[hc·n] **플랫 전체
//!   RMS**(layers.rs mtp_draft_step_h L1104-1128). 가중은 MTP GGUF
//!   (mtp-*-shared-Q8_0) blk.48.* 실측(Q8_0/F32) + 트렁크 공유 합성 MoE.
//! - 검증: 제안 [t0, d1]을 트렁크 체인 t=2 배치로 전 행 logits 산출
//!   (frame/verify.rs frame_forward_verify — 상태 구간 행별 t=1 준수는
//!   np 불변식으로 모듈 gdn 청크 경로와 동치, 스테이지 게이트로 판정).
//! - 수용: layers.rs mtp_spec_step L350-383 — n_acc = 첫 i where
//!   proposals[i+1] ≠ tgt_out[i]; accepted = [t0] + (수용분) + 보정
//!   토큰. 스펙 라운드 후 프로브 종료(롤백 재실행은 연속 생성 계약 —
//!   프로브는 수용 목록 동일성으로 판정).

//!   ═══ 체인 배선 계약 — 원천 core frame/forward.rs(워크트리 기준 줄번호) ═══
//! - 층 루프 순서: PLE(is_ple) → hc attn mix → GDN(is_recr)|QSA →
//!   hc_combine → hc ffn mix → MoE → hc_combine — frame_forward_ex의
//!   스테이지 발행 순서 그대로.
//! - 임베딩 방송: frame/mod.rs emb_broadcast_write L539-560 — 토큰 행
//!   [n]을 hc 스트림 전부에 복사([t][hc][n]).
//! - hc_combine: layers.rs hc_combine L2576-2588 —
//!   res[t][s·n+i] += out[t][i]·(2·σ(inject[t][s]/hc)).
//! - 헤드: forward.rs 꼬리 — RmsRows(output_hc_norm) → down →
//!   silu(lo/hc) → up → 게이트 적용+스트림 평균(=stages/hc.rs
//!   hc_mix_head, inject 없음) → 마지막 행만 output GEMM → logits.
//! - pos 진행: seq_st.pos += t(청크 종료) — QSA pp[0] 디바이스 진실은
//!   set_pos(0) 후 스텝마다 pos_bump(t) 1회(qsa_cuda.rs 결함 4호 계약).
//!
//! ═══ [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트] ═══
//! ① 선행 단계 공유 버퍼 오염 점검: 체인은 스텝·층 순서로 모듈·오라클을
//!    lock-step 진행 — 스테이지 판정은 **모듈 실제 입력**(경계 포획)으로
//!    스테이지 오라클을 재생한 exact-입력 비교라 선행 드리프트가 게이트를
//!    오염하지 않는다. 순수 스테이지(hc·moe) 재실행은 부작용 없고, 상태
//!    스테이지(gdn·qsa·ple)는 상주 상태 판독(gdn mids·qsa 캐시 판독·ple
//!    mids)으로 동일 효과를 얻는다 — 상태를 오염시키는 이중 실행 회피.
//!    QSA 선택 리스트는 모듈 qsa_select를 판정용으로 1회 선행 호출하고
//!    qsa_stage가 내부 재선택(멱등 — 같은 pos 재기록·블록 키 증분 없음)을
//!    검증한다.
//! ② 형상은 실측 GGUF 메타에서 자동 열거(FnDims::from_gguf — 추정 금지)
//!    + 실측치 대조 가드(48L·2560·hc4·lr320·GDN 48/128/16/4·QSA
//!      24/2/256/64·idx 4×128·top2048·r4·MoE 512e/10/640·ple [1]·ngram3).
//!      ③ 캡처-재생: 실측 토큰(smf64) → 실 token_embd 행 → 실 해시 → 실
//!      IQ4_NL 표 행 pread → 실가중 체인 3스텝(증분 상태 — 2스텝째 진입
//!      상태는 1스텝 산출물로 비영, S0≠0 정신 plans/124 §3.3).
//!      ④ 종단 값이 유일 불변량: 스텝별 logits maxdiff + 그리디 토큰 동일성
//!      (argmax는 체인 수준 토큰 동일성 관찰로만 허용 — 판정 본체는 값
//!      maxdiff. 과제 계약 §6).
//!
//! ═══ 가중치 원장 — REAL vs 합성(형상은 전부 실측) ═══
//! · REAL(GGUF 오프셋 직독, 전량 적재 금지 계약): 메타·스케줄·eps·해시
//!   파라미터(mult/offs/vs+eos), token_embd 체인 토큰 행(Q8_0),
//!   PLE 전체(ple 프로브와 동일 세트 — norms F32·conv1d F32·key/value
//!   Q8_0·IQ4_NL 표 행 pread·EXL3 ngram 헤더 교차 대조·quant 스트림),
//!   GDN 상수 전 36층(ssm_conv1d·dt.bias·ssm_a·ssm_norm F32),
//!   QSA 노름 전 12층(attn_{q,k}_norm·indexer.{q,k}_norm F32),
//!   HC 전 48층 ×2종(hc_{kind}_{norm,down,up,inject} — norm/inject F32·
//!   down/up Q8_0), 헤드(output_hc_norm F32·down/up Q8_0),
//!   output.weight 행 슬라이스 [0,V_SLICE)(Q8_0).
//! · 합성(결정론 smf64, 모듈·오라클 동일 바이트): MoE 전체 — 라우터·
//!   shared·전문가 f16. **전문가 수 512→32 감소 모델**(512e f16 상주는
//!   ~15GB로 예산 밖이고 실 GGUF expert 스택 Q4_K 345MB/층 ×48의 f16
//!   전량 디양자·양방향 소비는 REUSE(G2/G8) 통합 단계 소관 — moe 프로브가
//!   실측 512e 형상 정합을 이미 판정). 체인은 배선이 목적이라 32e top10
//!   감소 모델로 전문가 분기·결합 경로를 완전 커버한다.
//! · 투영(GDN qkv/z/b/a/out·QSA q/k/v/iq/ik/wo)은 **실측 가중**을
//!   스텝·층 방문마다 오프셋 직독해 호스트 순차 f32 내적으로 계산 —
//!   양측 공유 입력(ple/qsa 모듈 계약: "투영은 호출자 입력", REUSE 트랙).
//!   산술은 core matmul cpu.rs L64-76 미러(행별 순차·무-FMA).
//!
//! ═══ 판정 임계(스테이지별 착지 게이트 승계 + 종단) ═══
//! · ple.hash: 정수 완전일치 · ple.gather/ple.block: bitdiff=0(FNB 원장)
//!   · hc.attn/hc.ffn/hc.head: bitdiff=0(FNC) · gdn conv/prep 반쪽
//!   bitdiff=0·scan/gated/링·상태 ≤2e-4(FNF — 실측 ~3e-6) · qsa
//!   kv_k/idx_bk/attn ≤2e-6(FND core-libm 계급 — 실측 ~6e-8)·kv_v/idx_k
//!   bitdiff=0(원값 복사)·sel 리스트 완전일치(이산) · moe.route ids
//!   EXACT·가중치 ≤1e-6·출력 ≤3e-4(FNE — 실측 0.000e0) ·
//!   mtp.* pre-MoE bitdiff=0·post-MoE ≤2e-4(FNG §3.4 계급) ·
//!   **종단 logits maxdiff ≤2e-3**(아래 원장) · 스텝/검증 토큰 동일 ·
//!   수용 목록 동일(스펙 라운드 유일 이산 관찰 — argmax는 체인 수준
//!   토큰 동일성 관찰로만 허용, 판정 본체는 값 maxdiff §6).
//! · [종단 임계 원장 — 2e-3 채택 사유] fn-arm 단일 스테이지 임계
//!   2e-4(fn-gdn)는 **단일 스테이지 게이트** 등급이다. 본 체인 종단은
//!   36 GDN 스테이지의 expf-vs-libm 잔차(실측 스테이지 최악 3.3e-6·
//!   FNF 계급)가 실측 가중 잔차 스트림을 통과하며 누적·증폭된 값으로,
//!   실측 최악 4.66e-4(3스텝 종단) — 2e-4 단일 스테이지 임계로는 48층
//!   체인 판정 불가. plans/124 방식(측정값+기전+여유)에 따라 2e-3
//!   (실측의 ~4.3배)을 체인 종단 임계로 문서화한다 — 스테이지 게이트와
//!   별개 원장.

//! ═══ 음성대조(fn-chain-neg, 원장 17호 — 계기 자체 검증) ═══
//! 결함은 **모듈 체인 조립부**에 주입(오라클은 정상) — 검증계기가 체인
//! 결함 계급을 실제로 잡는지 증명:
//! · (a) 잘못된 층 순서: GDN 서수 5↔6의 (가중·상태) 슬롯 매핑 교환 —
//!   스케줄표 오독 계급(층 순서 결함).
//! · (b) 잘못된 잔차 부착: il=9의 hc_combine이 2σ(inject/4) 가중을
//!   상수 1.0으로 대체 — 잔차 부착 결함 계급.
//! · (c) 잘못된 수용/기각 순서: 수용 판정이 tgt_out[i+1]을 사용(검증
//!   행 순서 오독 — mtp_spec_step L352-354 계급).
//! (a)(b)는 스텝 logits maxdiff > 종단 임계, (c)는 수용 목록 불일치로
//! 각각 NEG-DETECTED(비영 exit).
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 개발기
//! (RTX 4070 SUPER, sm_89) 타이밍은 판단 근거가 아니다.
//!
//! [커널→프루브 매핑(plans/129-cuda C5)] 본 프로브는 신규 커널 없이
//! [커널→프루브 매핑(plans/129-cuda C5)] 본 프로브는 신규 커널 없이
//! 착지 모듈 fatbin(exl3_fn_ple·exl3_fn_hc·exl3_fn_qsa·exl3_fn_moe·
//! exl3_gdn(conv)+exl3_fn_gdn·exl3_fn_mtp_frame)을 소비한다 — 커널→
//! 프루브 매핑은 각 모듈 프로브 머리 원장 승계, 본 파일은 그 조립
//! (체인) 검증.
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 — scripts/
//! cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda_probe::{Rng, f16_to_f32, f32_to_f16, maxdiff_nan};
use crate::rawcuda::fn_gdn_cuda::FnGdnCuda;
use crate::rawcuda::fn_support::{
    FN_EXL3_DIR, FnDims, FnGguf, FnNgramHead, fn_quant_config_stream,
};
use crate::rawcuda::hc_cuda::{HC_HEAD_IL, HcCuda, HcDims};
use crate::rawcuda::moe_cuda::{MoeCuda, MoeDims};
use crate::rawcuda::mtp_fn_cuda::{MtpFnCuda, MtpFnDims, MtpMids};
use crate::rawcuda::ple_cuda::{PLE_TABLE_TENSOR, PleCuda, PleMids, ple_hash_rows};
use crate::rawcuda::qsa_cuda::{QsaCuda, QsaNeg};
use std::collections::HashMap;
use std::path::Path;

/// 종단 logits maxdiff 임계 — 체인 종등 원장(헤드 [종단 임계 원장] 항):
/// 36 GDN 스테이지 expf 잔차(실측 3.3e-6)의 48층 누적·증폭(실측 최악
/// 4.66e-4)을 판정하는 체인 전용 임계 — 단일 스테이지 fn-gdn 2e-4와
/// 별개. 실측의 ~4.3배 여유.
const CHAIN_LOGITS_THRESH: f32 = 2e-3;
/// QSA 값 스테이지 임계(FND core-libm 계급 — 실측 ~2.4e-7의 ~10배).
const CHAIN_QSA_THRESH: f32 = 2e-6;
/// GDN scan/상태 임계(FNF — plans/124 §1 GDN 종단 2e-4, 실측 ~2e-6).
const CHAIN_GDN_THRESH: f32 = 2e-4;
/// MoE 출력 임계(FNE 3e-4 — 실측 0.000e0) + 라우팅 가중치(FNE 1e-6).
const CHAIN_MOE_THRESH: f32 = 3e-4;
/// MTP 드래프트 post-MoE 값 임계(FNG plans/124 §3.4).
const CHAIN_MTP_POST_THRESH: f32 = 2e-4;
const CHAIN_MOE_W_THRESH: f32 = 1e-6;
/// 체인 MoE 감소 모델 전문가 수(헤더 가중치 원장 참조).
const CHAIN_MOE_EXPERTS: usize = 32;
/// logits 판정 어휘 슬라이스(행 오프셋 직독 — 248320 전량은 예산 밖,
/// 양측 동일 슬라이스로 체인 토큰 동일성은 슬라이스 상대로 정의).
const V_SLICE: usize = 8192;
/// 체인 스텝: t=2 프리필 + t=1 디코드 ×2(과제 계약 T=2~3).
const STEPS: usize = 3;
/// QSA 캐시 상한(pos0+t 최대 4 — 4배수 계약에 8 여유).
const QSA_CAP: usize = 8;
/// MTP 스펙 라운드 실측 MTP GGUF(Q8_0 — FNG 프로브 FN_MTP_GGUF_Q8 동일).
pub const CHAIN_MTP_GGUF: &str =
    "D:/models/qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
/// MTP 스펙 k(제안 수 = k-1 = 2: [t0, d1] — 수용 경계 1개 활성).
const MTP_K: usize = 3;

// ── core 트윈 — ops.rs 직이식(리터럴·연산 순서 그대로, 재작성 금지) ──

/// ops.rs exp_cr L52-91 — f64 fma 호너 13차 + 2^k 비트 재구성.
fn exp_cr(x: f32) -> f32 {
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

/// ops.rs ln_cr L91-121(atanh 급수 — softplus 전용).
fn ln_cr(v: f64) -> f64 {
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

fn silu(x: f32) -> f32 {
    // ops.rs L127-131.
    x / (1.0 + exp_cr(-x))
}

fn sigmoid(x: f32) -> f32 {
    // ops.rs L133-135.
    1.0 / (1.0 + exp_cr(-x))
}

fn softplus(x: f32) -> f32 {
    // ops.rs L138-141(log1p_cr 인라인).
    if x > 20.0 {
        x
    } else {
        ln_cr(exp_cr(x) as f64 + 1.0) as f32
    }
}

/// ops.rs sq_sum L11-31 — 32세그먼트 f32 순차 누산 → f64 순차 결합.
fn sq_sum(x: &[f32]) -> f64 {
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

/// ops.rs rms_norm L33-37 — f64 평균+eps → sqrt → f32 역수 → (v·scale)·g.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// stages/hc.rs grouped_rms L14-21 — 스트림별 절단 rms_norm.
fn grouped_rms(x: &[f32], w: &[f32], hc: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut xn = vec![0.0f32; hc * n];
    for s in 0..hc {
        xn[s * n..(s + 1) * n].copy_from_slice(&rms_norm(
            &x[s * n..(s + 1) * n],
            &w[s * n..(s + 1) * n],
            eps,
        ));
    }
    xn
}

/// ops.rs l2_norm L39-44 — 순차 f32 · eps floor.
fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let scale = 1.0 / sum.sqrt().max(eps);
    x.iter().map(|&v| v * scale).collect()
}

/// ops.rs rope_head L149-163 — f32 powf/cos/sin libm(코어 값 경로 원문).
fn rope_head(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
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

// ── 디양자 트윈(deq.rs 직미러 — ple 프로브와 동일 사본) ──

/// KVALUES_IQ4NL — tables.rs L73-75 직이식.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn deq_iq4_nl(blk: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let qs = &blk[2..18];
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        y[16 + j] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

fn deq_q8_0(blk: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    for j in 0..32 {
        y[j] = blk[2 + j] as i8 as f32 * d;
    }
}

/// Q8_0 텐서 원시 행들 → f32 [rows][k].
fn dequant_q8_rows(raw: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let blocks = k / 32;
    let mut out = vec![0.0f32; rows * k];
    for r in 0..rows {
        for b in 0..blocks {
            deq_q8_0(
                &raw[(r * blocks + b) * 34..][..34],
                &mut out[r * k + b * 32..][..32],
            );
        }
    }
    out
}

/// IQ4_NL 표 1행(90B) 디양자.
fn deq_iq4_nl_row(raw: &[u8], hd: usize, out: &mut [f32]) {
    let n_blocks = hd.div_ceil(32);
    let mut tmp = [0.0f32; 512];
    for b in 0..n_blocks {
        deq_iq4_nl(&raw[b * 18..(b + 1) * 18], &mut tmp[b * 32..(b + 1) * 32]);
    }
    out[..hd].copy_from_slice(&tmp[..hd]);
}

// ── 호스트 공유 투영 — core matmul cpu.rs L64-76 미러(행별 순차 f32) ──

fn nthreads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 12)
}

/// out[o] = Σ_i x[i]·w[o·k+i] — 출력 행 분할 병렬(행 내 순차 누산이라
/// 값은 스레드 수와 무관하게 결정론).
fn par_dot_rows(x: &[f32], w: &[f32], n_out: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out];
    let nt = nthreads().min(n_out.max(1));
    if nt <= 1 || n_out < 16 {
        for o in 0..n_out {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[i] * w[o * k + i];
            }
            out[o] = acc;
        }
        return out;
    }
    let per = n_out.div_ceil(nt);
    std::thread::scope(|sc| {
        for (t, chunk) in out.chunks_mut(per).enumerate() {
            let base = t * per;
            sc.spawn(move || {
                for (j, ov) in chunk.iter_mut().enumerate() {
                    let row = &w[(base + j) * k..(base + j + 1) * k];
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += x[i] * row[i];
                    }
                    *ov = acc;
                }
            });
        }
    });
    out
}

// ── HC 오라클 — stages/hc.rs hc_mix_ex L25-86(hc 프로브 사본) ──

#[allow(clippy::too_many_arguments)]
fn hc_mix_ex_ref(
    hc: usize,
    n: usize,
    lr: usize,
    eps: f32,
    w_norm: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_inject: Option<&[f32]>,
    res_hc: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let hcn = hc * n;
    let t = res_hc.len();
    let xn_all: Vec<Vec<f32>> = res_hc
        .iter()
        .map(|x| grouped_rms(x, w_norm, hc, n, eps))
        .collect();
    let mut lo_all = vec![vec![0.0f32; lr]; t];
    for (ti, lo) in lo_all.iter_mut().enumerate() {
        for o in 0..lr {
            let mut acc = 0.0f32;
            for i in 0..hcn {
                acc += xn_all[ti][i] * w_down[o * hcn + i];
            }
            lo[o] = acc;
        }
    }
    let inject_all: Vec<Vec<f32>> = match w_inject {
        Some(wi) => (0..t)
            .map(|ti| {
                (0..hc)
                    .map(|o| {
                        let mut acc = 0.0f32;
                        for i in 0..hcn {
                            acc += xn_all[ti][i] * wi[o * hcn + i];
                        }
                        acc
                    })
                    .collect()
            })
            .collect(),
        None => vec![Vec::new(); t],
    };
    for lo in lo_all.iter_mut() {
        for v in lo.iter_mut() {
            *v = silu(*v / hc as f32);
        }
    }
    let mut gate_all = vec![vec![0.0f32; hcn]; t];
    for (ti, gate) in gate_all.iter_mut().enumerate() {
        for o in 0..hcn {
            let mut acc = 0.0f32;
            for i in 0..lr {
                acc += lo_all[ti][i] * w_up[o * lr + i];
            }
            gate[o] = acc;
        }
    }
    let mut mixed = Vec::with_capacity(t);
    for (gate, xn) in gate_all.iter().zip(xn_all.iter()) {
        let mut m = vec![0.0f32; n];
        for s in 0..hc {
            for i in 0..n {
                m[i] += xn[s * n + i] * sigmoid(gate[s * n + i]);
            }
        }
        for v in m.iter_mut() {
            *v /= hc as f32;
        }
        mixed.push(m);
    }
    (mixed, inject_all)
}

// ── PLE 오라클 — stages/ple.rs L119-235·L276-360(ple 프로브 사본) ──

/// ple_block 미러 — res 제자리 잔차·상태 진화, (gates, gated, conv) 반환.
#[allow(clippy::too_many_arguments)]
fn oracle_ple_block(
    key: &[Vec<f32>],
    value: &[Vec<f32>],
    n_key: &[f32],
    n_query: &[f32],
    n_conv: &[f32],
    conv_w: &[f32],
    res_hc: &mut [Vec<f32>],
    st: &mut [f32],
    hc: usize,
    n_embd: usize,
    kern: usize,
    dil: usize,
    eps: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let hc_dim = hc * n_embd;
    let t = res_hc.len();
    let hist = (kern - 1) * dil;
    let mut gated_hist: Vec<Vec<f32>> = vec![Vec::new(); t];
    let mut gates_hist: Vec<Vec<f32>> = vec![Vec::new(); t];
    for ti in 0..t {
        let k_n = grouped_rms(&key[ti], n_key, hc, n_embd, eps);
        let q_n = grouped_rms(&res_hc[ti], n_query, hc, n_embd, eps);
        let mut gate = vec![0.0f32; hc];
        for s in 0..hc {
            let mut dot = 0.0f32;
            for i in 0..n_embd {
                dot += k_n[s * n_embd + i] * q_n[s * n_embd + i];
            }
            dot /= (n_embd as f32).sqrt();
            let mag = dot.abs().max(1e-6).sqrt();
            gate[s] = sigmoid(if dot >= 0.0 { mag } else { -mag });
        }
        let mut gated = vec![0.0f32; hc_dim];
        for s in 0..hc {
            for i in 0..n_embd {
                gated[s * n_embd + i] = value[ti][i] * gate[s];
            }
        }
        let normalized = grouped_rms(&gated, n_conv, hc, n_embd, eps);
        gates_hist[ti] = gate;
        gated_hist[ti] = normalized;
    }
    let mut padded: Vec<Vec<f32>> = Vec::with_capacity(hist + t);
    for j in 0..hist {
        padded.push(st[j * hc_dim..(j + 1) * hc_dim].to_vec());
    }
    for g in gated_hist.iter() {
        padded.push(g.clone());
    }
    let mut conv_out = vec![vec![0.0f32; hc_dim]; t];
    for ti in 0..t {
        for k in 0..kern {
            let start = hist + ti - (kern - 1 - k) * dil;
            let src = &padded[start];
            for c in 0..hc_dim {
                conv_out[ti][c] += conv_w[c * kern + k] * src[c];
            }
        }
        for c in 0..hc_dim {
            conv_out[ti][c] = silu(conv_out[ti][c]);
        }
    }
    for j in 0..hist {
        let src = &padded[t + j];
        st[j * hc_dim..(j + 1) * hc_dim].copy_from_slice(src);
    }
    for ti in 0..t {
        let gate = &gates_hist[ti];
        for s in 0..hc {
            let g = gate[s];
            for i in 0..n_embd {
                res_hc[ti][s * n_embd + i] += value[ti][i] * g + conv_out[ti][s * n_embd + i];
            }
        }
    }
    (gates_hist, gated_hist, conv_out)
}

/// ple_hash_rows 코어 미러(편집 회귀 감지용 쌍).
#[allow(clippy::too_many_arguments)]
fn oracle_hash_rows(
    hist0: &[u32],
    hist_valid: bool,
    tokens: &[u32],
    ngram: usize,
    hpng: usize,
    mult: &[u64],
    offs: &[u64],
    vs: &[u64],
    eos: u32,
) -> (Vec<u32>, Vec<u32>) {
    let heads = hpng * 2;
    let mut hist: Vec<u32> = if hist_valid {
        hist0.to_vec()
    } else {
        vec![eos; ngram - 1]
    };
    let mut rows = Vec::with_capacity(tokens.len() * heads);
    for (i, &tok) in tokens.iter().enumerate() {
        let mut ctx = vec![tok as u64; ngram];
        let mut cut = false;
        for s in 1..ngram {
            let j = i as i64 - s as i64;
            let prev: u64 = if j >= 0 {
                tokens[j as usize] as u64
            } else {
                let back = s as i64 - i as i64;
                let k = hist0.len() as i64 - back;
                if k >= 0 && (k as usize) < hist0.len() {
                    hist0[k as usize] as u64
                } else {
                    eos as u64
                }
            };
            ctx[s] = if cut { eos as u64 } else { prev };
            if ctx[s] == eos as u64 {
                cut = true;
            }
        }
        for n in 2..=ngram {
            let mut mixed = ctx[0].wrapping_mul(mult[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(mult[j]);
            }
            let base = (n - 2) * hpng;
            for g in 0..hpng {
                let h = base + g;
                rows.push((mixed % vs[h] + offs[h]) as u32);
            }
        }
        hist.push(tok);
        if hist.len() > ngram - 1 {
            let cutn = hist.len() - (ngram - 1);
            hist.drain(..cutn);
        }
    }
    (rows, hist)
}

// ── GDN 오라클 — stages/gdn.rs + core/gdn.rs + gdn_norm.rs
//    (fn_gdn 프로브 직미러 사본 — 인용 줄번호는 해당 파일 머리 원장) ──

/// GDN 스테이지 오라클 산출 일체(단계 판정용).
struct GdnRefOut {
    conv_q: Vec<f32>,
    conv_k: Vec<f32>,
    conv_v: Vec<f32>,
    ring_post: Vec<f32>,
    q2: Vec<f32>,
    k2: Vec<f32>,
    o: Vec<f32>,
    st_post: Vec<f32>,
    gated: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn gdn_ref_pre(
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
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let (dr, ds, ng, ck) = (dims.dt_rank, dims.d_state, dims.n_group, dims.conv_k);
    let k_len = ng * ds;
    let v_len = dr * ds;
    let mut beta_all = vec![0f32; t_len * dr];
    let mut g_all = vec![0f32; t_len * dr];
    for t in 0..t_len {
        for h in 0..dr {
            beta_all[t * dr + h] = sigmoid(b[t * dr + h]);
            g_all[t * dr + h] = softplus(a[t * dr + h] + dtb_l[h]) * ssa_l[h];
        }
    }
    let mut conv_q = vec![0f32; t_len * k_len];
    let mut conv_k = vec![0f32; t_len * k_len];
    let mut conv_v = vec![0f32; t_len * v_len];
    let mut ring = ring0.to_vec();
    let cch = ng * ds * 2 + dr * ds;
    for c in 0..cch {
        let (mut s0, mut s1, mut s2) = (ring[c], ring[cch + c], ring[2 * cch + c]);
        for t in 0..t_len {
            let x = qkv[t * cch + c];
            let mut sum = cw_l[c * ck + (ck - 1)] * x;
            sum += cw_l[c * ck] * s0;
            sum += cw_l[c * ck + 1] * s1;
            sum += cw_l[c * ck + 2] * s2;
            let out_c = silu(sum);
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
        ring[c] = s0;
        ring[cch + c] = s1;
        ring[2 * cch + c] = s2;
    }
    let mut q2 = vec![0f32; t_len * k_len];
    let mut k2 = vec![0f32; t_len * k_len];
    for t in 0..t_len {
        for h in 0..ng {
            let b0 = t * k_len + h * ds;
            let head: Vec<f32> = conv_q[b0..b0 + ds].to_vec();
            q2[b0..b0 + ds].copy_from_slice(&l2_norm(&head, eps));
            let headk: Vec<f32> = conv_k[b0..b0 + ds].to_vec();
            k2[b0..b0 + ds].copy_from_slice(&l2_norm(&headk, eps));
        }
    }
    (conv_q, conv_k, conv_v, ring, q2, k2, beta_all, g_all)
}

#[allow(clippy::too_many_arguments)]
fn gdn_ref_scan_chunk(
    q2: &[f32],
    k2: &[f32],
    conv_v: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    s0: &[f32],
    t_len: usize,
    dims: &FnDims,
) -> (Vec<f32>, Vec<f32>) {
    let cs = 64usize;
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
        st_all[st_h..st_h + d * d].copy_from_slice(&st);
    }
    (o, st_all)
}

#[allow(clippy::too_many_arguments)]
fn gdn_ref_scan_ar(
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
        let g_exp = exp_cr(g_all[h]);
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

/// gdn_norm.rs gdn_norm_gated L26-50(GdnGate::Sigmoid).
fn gdn_ref_gate(
    o: &[f32],
    z: &[f32],
    nw_l: &[f32],
    eps: f32,
    t_len: usize,
    dims: &FnDims,
) -> Vec<f32> {
    let (dr, ds) = (dims.dt_rank, dims.d_state);
    let v_len = dr * ds;
    let mut gated = vec![0f32; t_len * v_len];
    for t in 0..t_len {
        for h in 0..dr {
            let b0 = t * v_len + h * ds;
            let head: Vec<f32> = o[b0..b0 + ds].to_vec();
            let n = rms_norm(&head, nw_l, eps);
            let zb = h * ds;
            for i in 0..ds {
                gated[t * v_len + zb + i] = n[i] * sigmoid(z[t * v_len + zb + i]);
            }
        }
    }
    gated
}

/// GDN 스테이지 전체(청크/AR 코어 디스패치 포함).
#[allow(clippy::too_many_arguments)]
fn gdn_reference(
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
) -> GdnRefOut {
    let (conv_q, conv_k, conv_v, ring_post, q2, k2, beta_all, g_all) =
        gdn_ref_pre(dims, eps, cw_l, dtb_l, ssa_l, qkv, b, a, ring0, t_len);
    let (o, st_post) = if t_len == 1 {
        gdn_ref_scan_ar(&q2, &k2, &conv_v, &beta_all, &g_all, s0, dims)
    } else {
        gdn_ref_scan_chunk(&q2, &k2, &conv_v, &beta_all, &g_all, s0, t_len, dims)
    };
    let gated = gdn_ref_gate(&o, z, nw_l, eps, t_len, dims);
    GdnRefOut {
        conv_q,
        conv_k,
        conv_v,
        ring_post,
        q2,
        k2,
        o,
        st_post,
        gated,
    }
}

// ── QSA 체인 오라클 — stages/qsa.rs(qsa 프로브 QsaOracle 사본, per-fi) ──

/// fi별 캐시 1세트(kv_k/kv_v/idx_k/idx_bk — SeqState4 슬롯 미러).
struct QsaFiOracle {
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    idx_k: Vec<f32>,
    idx_bk: Vec<f32>,
}

struct QsaChainOracle {
    fi: Vec<QsaFiOracle>,
    pos: usize,
}

impl QsaChainOracle {
    fn new(n_full: usize, cap: usize, idx_dim: usize, n_kv: usize, hd: usize) -> Self {
        QsaChainOracle {
            fi: (0..n_full)
                .map(|_| QsaFiOracle {
                    kv_k: vec![0.0; cap * n_kv * hd],
                    kv_v: vec![0.0; cap * n_kv * hd],
                    idx_k: vec![0.0; cap * idx_dim],
                    idx_bk: Vec::new(),
                })
                .collect(),
            pos: 0,
        }
    }

    /// qsa_select L99-321 미러(libm 코어 경로) — 캐시 적립·풀링·top-k.
    #[allow(clippy::too_many_arguments)]
    fn select(
        &mut self,
        fi: usize,
        dims: &FnDims,
        knw: &[f32],
        iqw: &[f32],
        ikw: &[f32],
        kk: &[Vec<f32>],
        vv: &[Vec<f32>],
        iq: &[Vec<f32>],
        ik: &[Vec<f32>],
        t_len: usize,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let (n_kv, hd) = (dims.n_kv, dims.head_dim);
        let (n_rot, idx_dim, idx_heads) = (dims.n_rot, dims.idx_dim, dims.idx_heads);
        let (eps, base) = (dims.eps, dims.rope_base);
        let r = dims.compress.iter().find(|&&c| c != 0).copied().unwrap() as usize;
        let pos0 = self.pos;
        let slot = &mut self.fi[fi];
        let mut q_rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); t_len];
        for t in 0..t_len {
            let pos = pos0 + t;
            for h in 0..n_kv {
                let lo = h * hd;
                let dst = (pos0 + t) * n_kv * hd + lo;
                let mut head = rms_norm(&kk[t][lo..lo + hd], knw, eps);
                rope_head(&mut head, pos as u32, n_rot, base);
                slot.kv_k[dst..dst + hd].copy_from_slice(&head);
                slot.kv_v[dst..dst + hd].copy_from_slice(&vv[t][lo..lo + hd]);
            }
            slot.idx_k[(pos0 + t) * idx_dim..(pos0 + t + 1) * idx_dim]
                .copy_from_slice(&ik[t][..idx_dim]);
            let mut qr: Vec<Vec<f32>> = Vec::with_capacity(idx_heads);
            for h in 0..idx_heads {
                let lo = h * idx_dim;
                let mut qh = rms_norm(&iq[t][lo..lo + idx_dim], iqw, eps);
                rope_head(&mut qh, pos as u32, idx_dim, base);
                qr.push(qh);
            }
            q_rows[t] = qr;
        }
        let n_blocks_max = (pos0 + t_len) / r;
        if slot.idx_bk.len() < n_blocks_max * idx_dim {
            let b0 = slot.idx_bk.len() / idx_dim;
            slot.idx_bk.resize(n_blocks_max * idx_dim, 0.0);
            for b in b0..n_blocks_max {
                slot.idx_bk[b * idx_dim..(b + 1) * idx_dim].copy_from_slice(&pool_block_ref(
                    &slot.idx_k,
                    b,
                    r,
                    idx_dim,
                    ikw,
                    eps,
                    base,
                ));
            }
        }
        let idx_top_k = dims.idx_top_k;
        let sel_stride = idx_top_k / r + 2;
        let mut sel_blk: Vec<u32> = vec![0u32; t_len * sel_stride];
        let mut sel_cnt: Vec<u32> = vec![0u32; t_len];
        for t in 0..t_len {
            let n_past = pos0 + t + 1;
            let n_blocks = n_past / r;
            let (cnt, blocks) = pass_b_select(
                &q_rows[t],
                &slot.idx_bk,
                n_past,
                n_blocks,
                idx_dim,
                idx_top_k,
                r,
            );
            for (k2, &b) in blocks.iter().enumerate() {
                sel_blk[t * sel_stride + k2] = b as u32;
            }
            sel_cnt[t] = cnt as u32;
        }
        // pos 진행은 호출부(스텝 종료 1회) — 층마다 동일 pos0를 봐야 한다
        // (core seq_st.pos는 프레임당 1회 진행 — frame_forward_ex 계약).
        (sel_blk, sel_cnt, sel_stride)
    }

    /// cpu_attn_row L13-64 + mask_from_list — 외부 캐시 판독본 재생 가능.
    #[allow(clippy::too_many_arguments)]
    fn attn_ref(
        kv_k: &[f32],
        kv_v: &[f32],
        qg: &[Vec<f32>],
        qnw: &[f32],
        sel_blk: &[u32],
        sel_cnt: &[u32],
        sel_stride: usize,
        r: usize,
        t_len: usize,
        pos0: usize,
        dims: &FnDims,
    ) -> Vec<Vec<f32>> {
        let (n_head, n_kv, hd) = (dims.n_head, dims.n_kv, dims.head_dim);
        let (n_rot, eps, base) = (dims.n_rot, dims.eps, dims.rope_base);
        let kq_scale = dims.kq_scale();
        let mut out = vec![vec![0.0f32; n_head * hd]; t_len];
        for t in 0..t_len {
            let n_past = pos0 + t + 1;
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
            for h in 0..n_head {
                // q 반 norm+rope(qsa_layer L508-517) — 게이트 반 원값.
                let lo = h * 2 * hd;
                let mut qh = rms_norm(&qg[t][lo..lo + hd], qnw, eps);
                rope_head(&mut qh, (pos0 + t) as u32, n_rot, base);
                let kvh = h / (n_head / n_kv);
                let mut maxv = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_past];
                for (p, sc) in scores.iter_mut().enumerate() {
                    if !m[p] {
                        *sc = f32::NEG_INFINITY;
                        continue;
                    }
                    let bidx = p * n_kv * hd + kvh * hd;
                    let mut d = 0.0f32;
                    for i in 0..hd {
                        d += qh[i] * kv_k[bidx + i];
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
                for (p, sc) in scores.iter().enumerate() {
                    let w = sc / sum;
                    if w == 0.0 {
                        continue;
                    }
                    let bidx = p * n_kv * hd + kvh * hd;
                    for i in 0..hd {
                        out[t][ob + i] += w * kv_v[bidx + i];
                    }
                }
                let gb = h * 2 * hd + hd;
                for i in 0..hd {
                    out[t][ob + i] *= sigmoid(qg[t][gb + i]);
                }
            }
        }
        out
    }
}

/// 블록 키 풀링 1블록 재생(qsa_select L198-208 — mean-pool→rms→rope).
fn pool_block_ref(
    idx_k: &[f32],
    b: usize,
    r: usize,
    idx_dim: usize,
    ikw: &[f32],
    eps: f32,
    base: f32,
) -> Vec<f32> {
    let mut pooled = vec![0.0f32; idx_dim];
    for j in 0..r {
        let src = (b * r + j) * idx_dim;
        for i2 in 0..idx_dim {
            pooled[i2] += idx_k[src + i2];
        }
    }
    for v in pooled.iter_mut() {
        *v /= r as f32;
    }
    let mut pk = rms_norm(&pooled, ikw, eps);
    rope_head(&mut pk, (b * r) as u32, idx_dim, base);
    pk
}

/// 패스 B(top-k 선택 — qsa_select L210-307). 반환 (cnt, 오름차순 블록).
fn pass_b_select(
    q_rows: &[Vec<f32>],
    idx_bk: &[f32],
    n_past: usize,
    n_blocks: usize,
    idx_dim: usize,
    idx_top_k: usize,
    r: usize,
) -> (usize, Vec<usize>) {
    let mut block_score = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        let pk = &idx_bk[b * idx_dim..(b + 1) * idx_dim];
        for qh in q_rows {
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
    let mut sbs: Vec<usize> = sel_blocks[..n_sel_blocks].to_vec();
    sbs.sort_unstable();
    (n_sel_blocks, sbs)
}

/// 패스 B 스테이지 오라클(모듈 iq + 모듈 idx_bk 판독본 — exact-입력 선택
/// 리스트 판정).
fn qsa_stage_pass_b_ref(
    dims: &FnDims,
    iqw: &[f32],
    iq: &[Vec<f32>],
    idx_bk: &[f32],
    pos0: usize,
    t_len: usize,
) -> (Vec<u32>, Vec<u32>, usize) {
    let (idx_dim, idx_heads) = (dims.idx_dim, dims.idx_heads);
    let (eps, base) = (dims.eps, dims.rope_base);
    let r = dims.compress.iter().find(|&&c| c != 0).copied().unwrap() as usize;
    let mut q_rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); t_len];
    for t in 0..t_len {
        let mut qr: Vec<Vec<f32>> = Vec::with_capacity(idx_heads);
        for h in 0..idx_heads {
            let lo = h * idx_dim;
            let mut qh = rms_norm(&iq[t][lo..lo + idx_dim], iqw, eps);
            rope_head(&mut qh, (pos0 + t) as u32, idx_dim, base);
            qr.push(qh);
        }
        q_rows[t] = qr;
    }
    let idx_top_k = dims.idx_top_k;
    let sel_stride = idx_top_k / r + 2;
    let mut sel_blk: Vec<u32> = vec![0u32; t_len * sel_stride];
    let mut sel_cnt: Vec<u32> = vec![0u32; t_len];
    for t in 0..t_len {
        let n_past = pos0 + t + 1;
        let n_blocks = n_past / r;
        let (cnt, blocks) =
            pass_b_select(&q_rows[t], idx_bk, n_past, n_blocks, idx_dim, idx_top_k, r);
        for (k2, &b) in blocks.iter().enumerate() {
            sel_blk[t * sel_stride + k2] = b as u32;
        }
        sel_cnt[t] = cnt as u32;
    }
    (sel_blk, sel_cnt, sel_stride)
}

/// k행 norm+rope 스테이지 오라클(모듈 kk exact-입력 → kv_k 기댓값).
fn qsa_stage_k_rows_ref(
    dims: &FnDims,
    knw: &[f32],
    kk: &[Vec<f32>],
    t_len: usize,
    pos0: usize,
) -> Vec<f32> {
    let (n_kv, hd) = (dims.n_kv, dims.head_dim);
    let (n_rot, eps, base) = (dims.n_rot, dims.eps, dims.rope_base);
    let mut out = vec![0.0f32; t_len * n_kv * hd];
    for t in 0..t_len {
        for h in 0..n_kv {
            let lo = h * hd;
            let dst = t * n_kv * hd + lo;
            let mut head = rms_norm(&kk[t][lo..lo + hd], knw, eps);
            rope_head(&mut head, (pos0 + t) as u32, n_rot, base);
            out[dst..dst + hd].copy_from_slice(&head);
        }
    }
    out
}

// ── MoE 오라클 — stages/moe.rs(moe 프로브 사본, 체인 감소 모델) ──

/// f16 행 내적 — core matmul cpu L91-94 순차 누산 미러.
fn dot16(x: &[f32], w: &[u16], row: usize, n_in: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n_in {
        acc += x[i] * f16_to_f32(w[row * n_in + i]);
    }
    acc
}

/// 라우팅 오라클(moe.rs L40-70 — exp는 exp_cr 치환 계약).
fn moe_route_ref(
    route: &[u16],
    route_sh: &[u16],
    n_expert: usize,
    n_used: usize,
    xs: &[Vec<f32>],
) -> (Vec<Vec<(u32, f32)>>, Vec<f32>) {
    let n_embd = xs.first().map(|x| x.len()).unwrap_or(0);
    let t = xs.len();
    let mut sel_all = Vec::with_capacity(t);
    let mut sgates = Vec::with_capacity(t);
    for ti in 0..t {
        let mut logits: Vec<f32> = (0..n_expert)
            .map(|e| dot16(&xs[ti], route, e, n_embd))
            .collect();
        sgates.push(dot16(&xs[ti], route_sh, 0, n_embd));
        let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut zs = 0.0f32;
        for v in logits.iter_mut() {
            *v = exp_cr(*v - mx);
            zs += *v;
        }
        for v in logits.iter_mut() {
            *v /= zs;
        }
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let sel = &idx[..n_used];
        let mut wsum: f32 = sel.iter().map(|&e| logits[e]).sum();
        wsum = wsum.max(6.103_515_6e-5);
        let mut row = Vec::with_capacity(n_used);
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

/// (ti,e) 페어 1개 산출 — gate·up·silu·mul·down(moe.rs L228-297).
fn moe_pair_out(
    x: &[f32],
    w: &(Vec<u16>, Vec<u16>, Vec<u16>),
    n_embd: usize,
    n_ff: usize,
) -> Vec<f32> {
    let (g16, u16v, d16) = w;
    let mut act = vec![0.0f32; n_ff];
    let mut up = vec![0.0f32; n_ff];
    for o in 0..n_ff {
        act[o] = dot16(x, g16, o, n_embd);
        up[o] = dot16(x, u16v, o, n_embd);
    }
    for o in 0..n_ff {
        act[o] = silu(act[o]) * up[o];
    }
    (0..n_embd).map(|o| dot16(&act, d16, o, n_ff)).collect()
}

/// moe_ffn 오라클(moe.rs L22-320 — 페어 토큰-메이저·e 오름차 누산.
/// 페어 GEMM은 병렬, 가중 누산·shared는 순차(누산 순서 계약)).
fn moe_ffn_ref_chain(
    n_embd: usize,
    n_ff: usize,
    n_ff_sh: usize,
    n_used: usize,
    route: &[u16],
    route_sh: &[u16],
    sh_gate: &[u16],
    sh_up: &[u16],
    sh_down: &[u16],
    experts: &HashMap<u32, (Vec<u16>, Vec<u16>, Vec<u16>)>,
    xs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let (sel_all, sgates) = moe_route_ref(route, route_sh, experts.len(), n_used, xs);
    let t = xs.len();
    let mut out = vec![vec![0.0f32; n_embd]; t];
    let mut pairs: Vec<(usize, u32, f32)> = Vec::with_capacity(t * n_used);
    for (ti, row) in sel_all.iter().enumerate() {
        let mut row: Vec<(u32, f32)> = row.clone();
        row.sort_by_key(|&(e, _)| e);
        for (e, w) in row {
            pairs.push((ti, e, w));
        }
    }
    // 페어 산출 병렬(페어 독립 — 값은 분할 무관).
    let pair_outs: Vec<Vec<f32>> = if pairs.len() > 2 {
        let per = pairs.len().div_ceil(nthreads());
        std::thread::scope(|sc| {
            let hs: Vec<_> = pairs
                .chunks(per.max(1))
                .map(|chunk| {
                    sc.spawn(move || {
                        chunk
                            .iter()
                            .map(|&(ti, e, _)| moe_pair_out(&xs[ti], &experts[&e], n_embd, n_ff))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut all = Vec::new();
            for h in hs {
                all.extend(
                    h.join()
                        .map_err(|_| "moe oracle 스레드 패닉")
                        .unwrap_or_default(),
                );
            }
            all
        })
    } else {
        pairs
            .iter()
            .map(|&(ti, e, _)| moe_pair_out(&xs[ti], &experts[&e], n_embd, n_ff))
            .collect()
    };
    // 가중 누산 — 페어 순서(토큰-메이저·토큰 내 e 오름차) 그대로.
    for ((ti, _, w), eo) in pairs.iter().zip(pair_outs.iter()) {
        let orow = &mut out[*ti];
        for i in 0..n_embd {
            orow[i] += w * eo[i];
        }
    }
    // shared 전문가 + sigmoid 게이트 가산(L246-318).
    for ti in 0..t {
        let mut act = vec![0.0f32; n_ff_sh];
        let mut up = vec![0.0f32; n_ff_sh];
        for o in 0..n_ff_sh {
            act[o] = dot16(&xs[ti], sh_gate, o, n_embd);
            up[o] = dot16(&xs[ti], sh_up, o, n_embd);
        }
        for o in 0..n_ff_sh {
            act[o] = silu(act[o]) * up[o];
        }
        let shout: Vec<f32> = (0..n_embd)
            .map(|o| dot16(&act, sh_down, o, n_ff_sh))
            .collect();
        let sh_w = sigmoid(sgates[ti]);
        let orow = &mut out[ti];
        for i in 0..n_embd {
            orow[i] += sh_w * shout[i];
        }
    }
    out
}

// ── MTP 드래프트 오라클 — frame/mtp.rs mtp_draft_frame L362-484
//    (FNG 프로브 o_draft_step/o_attn_row 사본 — layers.rs mtp_attn_cpu_row
//    L2505-2573 직미리: q/k norm+rope·KV 적립·cell 0 스킵·softmax exp
//    libm·게이트 sigmoid) ──

/// 드래프트 KV 호스트 상태(core SeqState4 mtp 슬롯 계급).
struct MtpOracleSt {
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    pos: usize,
}

impl MtpOracleSt {
    fn new(cap: usize, kv_dim: usize) -> Self {
        MtpOracleSt {
            kv_k: vec![0.0; cap * kv_dim],
            kv_v: vec![0.0; cap * kv_dim],
            pos: 0,
        }
    }
}

/// dense 어텐션 1행 — layers.rs mtp_attn_cpu_row L2505-2573 직미러.
#[allow(clippy::too_many_arguments)]
fn mtp_attn_row_or(
    dims: &FnDims,
    q_row: &mut [f32],
    k_row: &mut [f32],
    v_row: &[f32],
    st: &mut MtpOracleSt,
    qn: &[f32],
    kn: &[f32],
) -> Vec<f32> {
    let (n_head, n_kv, hd, n_rot) = (dims.n_head, dims.n_kv, dims.head_dim, dims.n_rot);
    let (eps, base) = (dims.eps, dims.rope_base);
    let pos = st.pos;
    let kq_scale = dims.kq_scale();
    for h in 0..n_head {
        let lo = h * 2 * hd;
        let mut qh = rms_norm(&q_row[lo..lo + hd], qn, eps);
        rope_head(&mut qh, pos as u32, n_rot, base);
        q_row[lo..lo + hd].copy_from_slice(&qh);
    }
    let kbase = pos * n_kv * hd;
    for h in 0..n_kv {
        let lo = h * hd;
        let mut kh = rms_norm(&k_row[lo..lo + hd], kn, eps);
        rope_head(&mut kh, pos as u32, n_rot, base);
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
            let pcell = p0 + 1; // cell 0 스킵(L2540)
            let b = pcell * n_kv * hd + kvh * hd;
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
            out[ob + i] *= sigmoid(q_row[gb + i]);
        }
    }
    out
}

/// mtp_draft_step 오라클 산출(단계 판정용).
struct MtpOrMids {
    eh: Vec<f32>,
    mix_attn: Vec<f32>,
    inj_attn: Vec<f32>,
    attn: Vec<f32>,
    ao: Vec<f32>,
    res_attn: Vec<f32>,
    mix_ffn: Vec<f32>,
    inj_ffn: Vec<f32>,
    mout: Vec<f32>,
    chain_h: Vec<f32>,
    hin: Vec<f32>,
    logits: Vec<f32>,
    token: u32,
}

/// mtp_draft_frame L362-484 미라 — eh_proj → attn mix → dense attn → wo →
/// combine → ffn mix → MoE → combine → nextn head → logits/argmax.
#[allow(clippy::too_many_arguments)]
fn mtp_draft_oracle(fx: &ChainFx, st: &mut MtpOracleSt, en: &[f32], hn: &[f32]) -> MtpOrMids {
    let dims = &fx.dims;
    let (n, hc, hcn) = (dims.n_embd, dims.hc, dims.hc * dims.n_embd);
    let (lr, eps) = (dims.hc_low_rank, dims.eps);
    let w = &fx.mtp;
    let (qg, kvd, qd) = (
        dims.n_head * 2 * dims.head_dim,
        dims.n_kv * dims.head_dim,
        dims.n_head * dims.head_dim,
    );
    // 1) eh_proj — cat[s]=[en ‖ hn_s] 순차 dot(L388-396과 동일 값).
    let mut eh = vec![0.0f32; hcn];
    for s in 0..hc {
        let mut cat = vec![0.0f32; 2 * n];
        cat[..n].copy_from_slice(en);
        cat[n..].copy_from_slice(&hn[s * n..(s + 1) * n]);
        for o in 0..n {
            let mut acc = 0.0f32;
            for i in 0..2 * n {
                acc += cat[i] * w.eh[o * 2 * n + i];
            }
            eh[s * n + o] = acc;
        }
    }
    // 2) attn 믹서(hc_mix_ex 미러 — nextn 가중).
    let eh_rows: Vec<Vec<f32>> = eh.chunks(hcn).map(<[f32]>::to_vec).collect();
    let (mix1_rows, inj1) =
        hc_mix_ex_ref(hc, n, lr, eps, &w.an, &w.ad, &w.au, Some(&w.ai), &eh_rows);
    let mix1 = mix1_rows.into_iter().next().unwrap();
    // 3) dense 어텐션 — 투영 → CPU 행 → wo.
    let q = par_dot_rows(&mix1, &w.q, qg, n);
    let k = par_dot_rows(&mix1, &w.k, kvd, n);
    let v = par_dot_rows(&mix1, &w.v, kvd, n);
    let mut qm = q.clone();
    let mut km = k.clone();
    let attn = mtp_attn_row_or(dims, &mut qm, &mut km, &v, st, &w.qn, &w.kn);
    let ao = par_dot_rows(&attn, &w.o, n, qd);
    // 4) combine(attn) → ffn 믹서 → MoE → combine.
    let mut res_attn = eh.clone();
    for s in 0..hc {
        let wsg = 2.0 * sigmoid(inj1[0][s] / hc as f32);
        for i in 0..n {
            res_attn[s * n + i] += ao[i] * wsg;
        }
    }
    let ra_rows: Vec<Vec<f32>> = vec![res_attn.clone()];
    let (mix2_rows, inj2) =
        hc_mix_ex_ref(hc, n, lr, eps, &w.fnn, &w.fdd, &w.fu, Some(&w.fi), &ra_rows);
    let mix2 = mix2_rows.into_iter().next().unwrap();
    let mout_v = moe_ffn_ref_chain(
        n,
        dims.n_ff_exp,
        dims.n_ff_shared,
        dims.n_expert_used,
        &fx.moe_route,
        &fx.moe_route_sh,
        &fx.moe_sh_gate,
        &fx.moe_sh_up,
        &fx.moe_sh_down,
        &fx.moe_experts,
        std::slice::from_ref(&mix2),
    )
    .into_iter()
    .next()
    .unwrap();
    let mut chain_h = res_attn.clone();
    for s in 0..hc {
        let wsg = 2.0 * sigmoid(inj2[0][s] / hc as f32);
        for i in 0..n {
            chain_h[s * n + i] += mout_v[i] * wsg;
        }
    }
    // 5) nextn.hc_head 믹서 → logits/argmax(슬라이스 헤드).
    let ch_rows: Vec<Vec<f32>> = vec![chain_h.clone()];
    let (hin_rows, _) = hc_mix_ex_ref(hc, n, lr, eps, &w.hn, &w.hd, &w.hu, None, &ch_rows);
    let hin = hin_rows.into_iter().next().unwrap();
    let logits = par_dot_rows(&hin, &fx.out_rows, V_SLICE, n);
    let token = argmax(&logits);
    MtpOrMids {
        eh,
        mix_attn: mix1,
        inj_attn: inj1.into_iter().next().unwrap_or_default(),
        attn,
        ao,
        res_attn,
        mix_ffn: mix2,
        inj_ffn: inj2.into_iter().next().unwrap_or_default(),
        mout: mout_v,
        chain_h,
        hin,
        logits,
        token,
    }
}

// ── 판정 통계 ──

/// 스테이지별 최악값 집계(표 1행 — 최악 maxdiff·bitdiff·위치).
#[derive(Default)]
struct StageStat {
    max: f32,
    bits: usize,
    nan: usize,
    loc: String,
    checked: usize,
}

impl StageStat {
    fn update(&mut self, name: &str, got: &[f32], want: &[f32]) {
        let (md, nan) = maxdiff_nan(got, want);
        let bd = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        self.checked += 1;
        if md > self.max {
            self.max = md;
        }
        if nan > 0 && self.loc.is_empty() {
            self.loc = name.to_string();
        }
        if bd > 0 && self.loc.is_empty() {
            self.loc = name.to_string();
        }
        self.bits = self.bits.max(bd);
        self.nan += nan;
    }
    fn update_int(&mut self, name: &str, bad: usize) {
        self.checked += 1;
        if bad > 0 {
            self.bits = self.bits.max(bad);
            if self.loc.is_empty() {
                self.loc = name.to_string();
            }
        }
    }
    fn note_max(&mut self, name: &str, md: f32) {
        self.checked += 1;
        if md > self.max {
            self.max = md;
            self.loc = name.to_string();
        }
    }
}

// ── 픽스처 ──

struct GdnProj {
    qkv: Vec<f32>,
    z: Vec<f32>,
    b: Vec<f32>,
    a: Vec<f32>,
    out: Vec<f32>,
}

struct QsaProj {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    iq: Vec<f32>,
    ik: Vec<f32>,
    wo: Vec<f32>,
}

/// MTP nextn 블록 실측 가중(MTP GGUF blk.48.* — FNG 프로브 재생 세트).
struct MtpW {
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    eh: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    an: Vec<f32>,
    ad: Vec<f32>,
    au: Vec<f32>,
    ai: Vec<f32>,
    fnn: Vec<f32>,
    fdd: Vec<f32>,
    fu: Vec<f32>,
    fi: Vec<f32>,
    hn: Vec<f32>,
    hd: Vec<f32>,
    hu: Vec<f32>,
}

/// 체인 실측 픽스처 — 캐시형(스텝 공유) + 방문형(투영 가중).
struct ChainFx {
    g: FnGguf,
    dims: FnDims,
    mult: Vec<u64>,
    offs: Vec<u64>,
    vs: Vec<u64>,
    eos: u32,
    ple_n_key: Vec<f32>,
    ple_n_query: Vec<f32>,
    ple_n_conv: Vec<f32>,
    ple_conv_w: Vec<f32>,
    ple_key_w: Vec<f32>,
    ple_value_w: Vec<f32>,
    gdn_ils: Vec<usize>,
    cw_all: Vec<f32>,
    dtb_all: Vec<f32>,
    ssa_all: Vec<f32>,
    nw_all: Vec<f32>,
    qsa_ils: Vec<usize>,
    qsa_norms: Vec<[Vec<f32>; 4]>,
    head_norm: Vec<f32>,
    head_down: Vec<f32>,
    head_up: Vec<f32>,
    out_rows: Vec<f32>,
    emb_cache: HashMap<u32, Vec<f32>>,
    mtp: MtpW,
    moe_route: Vec<u16>,
    moe_route_sh: Vec<u16>,
    moe_sh_gate: Vec<u16>,
    moe_sh_up: Vec<u16>,
    moe_sh_down: Vec<u16>,
    moe_experts: HashMap<u32, (Vec<u16>, Vec<u16>, Vec<u16>)>,
}

/// GGUF 2-D 가중 → f32 [rows][k] 오프셋 직독(ty 0 F32·8 Q8_0·30 BF16).
fn read_w2d(g: &FnGguf, name: &str) -> Result<Vec<f32>, String> {
    let t = g
        .tensor(name)
        .ok_or_else(|| format!("gguf 텐서 없음: {name}"))?;
    let k = t.dims.first().copied().unwrap_or(0) as usize;
    let rows = t.dims.get(1).copied().unwrap_or(1) as usize;
    let raw = g.read_rows(name, 0, rows as u64)?;
    let v: Vec<f32> = match t.ty {
        0 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        8 => dequant_q8_rows(&raw, rows, k),
        30 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        other => return Err(format!("{name}: gguf ty {other} — 체인 투영 계약 밖")),
    };
    if v.len() != rows * k {
        return Err(format!("{name}: {} != {rows}×{k}", v.len()));
    }
    Ok(v)
}

#[allow(clippy::too_many_lines)]
fn load_fixture(gguf_main: &str) -> Result<ChainFx, String> {
    let g = FnGguf::open(Path::new(gguf_main))?;
    let dims = FnDims::from_gguf(&g)?;
    // 형상 실측 대조(ple 프로브 가드 승계 + 체인 전체).
    let got: [usize; 20] = [
        dims.n_layer,
        dims.n_embd,
        dims.hc,
        dims.hc_low_rank,
        dims.n_head,
        dims.n_kv,
        dims.head_dim,
        dims.n_rot,
        dims.dt_rank,
        dims.d_state,
        dims.n_group,
        dims.conv_k,
        dims.n_expert,
        dims.n_expert_used,
        dims.n_ff_exp,
        dims.n_ff_shared,
        dims.idx_heads,
        dims.idx_dim,
        dims.idx_top_k,
        dims.ple_head_dim,
    ];
    let want: [usize; 20] = [
        48, 2560, 4, 320, 24, 2, 256, 64, 48, 128, 16, 4, 512, 10, 640, 640, 4, 128, 2048, 160,
    ];
    let ple_ok = dims.ple_layers == vec![1usize]
        && dims.ple_ngram == 3
        && dims.ple_heads_per_ngram == 8
        && dims.ple_conv_k == 4;
    if got != want || !ple_ok {
        return Err(format!(
            "fn-chain: 형상 {got:?} ple({:?},{},{},{},{}) ≠ 실측 — 픽스처 변경 가드",
            dims.ple_layers,
            dims.ple_ngram,
            dims.ple_heads_per_ngram,
            dims.ple_conv_k,
            dims.ple_head_dim
        ));
    }
    if dims.compress.iter().filter(|&&c| c == 0).count() != 36
        || dims.compress.iter().filter(|&&c| c == 4).count() != 12
    {
        return Err(format!(
            "fn-chain: compress 구성 {:?} — 36 GDN + 12 QSA(=4) 실측 가드",
            dims.compress
        ));
    }
    // PLE 해시 파라미터 + EXL3 교차 대조(ple 프로브 계약).
    let mult = g
        .kv_arr_u64("qwen4exp.ple.layer_multipliers")
        .ok_or("gguf kv: ple.layer_multipliers 없음")?
        .to_vec();
    let offs = g
        .kv_arr_u64("qwen4exp.ple.head_offsets")
        .ok_or("gguf kv: ple.head_offsets 없음")?
        .to_vec();
    let vs = g
        .kv_arr_u64("qwen4exp.ple.head_vocab_sizes")
        .ok_or("gguf kv: ple.head_vocab_sizes 없음")?
        .to_vec();
    let eos = g
        .kv_u64("qwen4exp.ple.eos_token_id")
        .ok_or("gguf kv: ple.eos_token_id 없음")? as u32;
    if mult.len() != dims.ple_ngram || offs.len() != 16 || vs.len() != 16 {
        return Err("fn-chain: ple 해시 파라미터 길이 가드 위반".into());
    }
    for h in 0..15 {
        if offs[h] + vs[h] != offs[h + 1] {
            return Err(format!("fn-chain: ple 헤드 파티션 끊김 h={h}"));
        }
    }
    let exl3 = Path::new(FN_EXL3_DIR);
    let ng = FnNgramHead::open(exl3)?;
    if ng.head_offsets != offs || ng.head_vocab_sizes != vs || ng.layer_multipliers != mult {
        return Err("fn-chain: GGUF kv ↔ EXL3 ngram 헤더 불일치 — 교차 대조 가드".into());
    }
    let q = fn_quant_config_stream(&exl3.join("quantization_config.json"))?;
    let ple_n_key = read_w2d(&g, "blk.1.ple_norm_key.weight")?;
    let ple_n_query = read_w2d(&g, "blk.1.ple_norm_query.weight")?;
    let ple_n_conv = read_w2d(&g, "blk.1.ple_norm_conv.weight")?;
    let ple_conv_w = read_w2d(&g, "blk.1.ple_conv1d.weight")?;
    let ple_key_w = read_w2d(&g, "blk.1.ple_key.weight")?;
    let ple_value_w = read_w2d(&g, "blk.1.ple_value.weight")?;
    if ple_n_key.len() != 10240
        || ple_conv_w.len() != 40960
        || ple_key_w.len() != 10240 * 2560
        || ple_value_w.len() != 2560 * 2560
    {
        return Err("fn-chain: ple 가중 직독 길이 가드 위반".into());
    }
    // GDN 상수 전 36층(fn_gdn 프로브 FnGdnFixture 계약).
    let gdn_ils: Vec<usize> = (0..dims.n_layer)
        .filter(|&il| dims.compress[il] == 0)
        .collect();
    let (dr, cch) = (dims.dt_rank, dims.gdn_conv_ch());
    let f32s = |raw: &[u8]| -> Vec<f32> {
        raw.as_chunks::<4>()
            .0
            .iter()
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
        if tc.ty != 0 || tc.dims != vec![dims.conv_k as u64, cch as u64] {
            return Err(format!("blk.{il}.ssm_conv1d: 형상 계약 위반"));
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
    // QSA 노름 전 12층(fi 순서 — qsa 프로브 gguf_norm_f32 계약).
    let qsa_ils: Vec<usize> = (0..dims.n_layer)
        .filter(|&il| dims.compress[il] != 0)
        .collect();
    let norm_f32 = |name: &str, want_n: usize| -> Result<Vec<f32>, String> {
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
            other => return Err(format!("{name}: ty {other} 미지원(노름)")),
        };
        if v.len() != want_n {
            return Err(format!("{name}: {} != {want_n}", v.len()));
        }
        Ok(v)
    };
    let mut qsa_norms = Vec::with_capacity(qsa_ils.len());
    for &il in &qsa_ils {
        qsa_norms.push([
            norm_f32(&format!("blk.{il}.attn_k_norm.weight"), dims.head_dim)?,
            norm_f32(&format!("blk.{il}.attn_q_norm.weight"), dims.head_dim)?,
            norm_f32(&format!("blk.{il}.indexer.q_norm.weight"), dims.idx_dim)?,
            norm_f32(&format!("blk.{il}.indexer.k_norm.weight"), dims.idx_dim)?,
        ]);
    }
    // 헤드 + output 슬라이스.
    let head_norm = read_w2d(&g, "output_hc_norm.weight")?;
    let head_down = read_w2d(&g, "output_hc_down.weight")?;
    let head_up = read_w2d(&g, "output_hc_up.weight")?;
    if head_norm.len() != 10240 || head_down.len() != 320 * 10240 || head_up.len() != 10240 * 320 {
        return Err("fn-chain: 헤드 가중 길이 가드 위반".into());
    }
    let out_t = g.tensor("output.weight").ok_or("output.weight 없음")?;
    if out_t.ty != 8 || out_t.dims != vec![2560u64, 248_320u64] {
        return Err(format!(
            "fn-chain: output.weight ty{} {:?} — Q8_0 [2560,248320] 실측 가드",
            out_t.ty, out_t.dims
        ));
    }
    let raw = g.read_rows("output.weight", 0, V_SLICE as u64)?;
    let out_rows = dequant_q8_rows(&raw, V_SLICE, 2560);
    // MTP nextn 블록 실측 가중(MTP GGUF Q8_0 — blk.48.*, core load_mtp
    // 계약: nextn_predict_layers==1·블록 인덱스 = block_count-1).
    let mg = FnGguf::open(Path::new(CHAIN_MTP_GGUF))?;
    let mtp_il = mg
        .kv_u64("qwen4exp.block_count")
        .ok_or("mtp gguf: qwen4exp.block_count 없음")? as usize
        - 1;
    if mtp_il != dims.n_layer {
        return Err(format!(
            "fn-chain: mtp 블록 {mtp_il} != n_layer {}",
            dims.n_layer
        ));
    }
    let mtp = MtpW {
        enorm: read_w2d(&mg, &format!("blk.{mtp_il}.nextn.enorm.weight"))?,
        hnorm: read_w2d(&mg, &format!("blk.{mtp_il}.nextn.hnorm.weight"))?,
        eh: read_w2d(&mg, &format!("blk.{mtp_il}.nextn.eh_proj.weight"))?,
        q: read_w2d(&mg, &format!("blk.{mtp_il}.attn_q.weight"))?,
        k: read_w2d(&mg, &format!("blk.{mtp_il}.attn_k.weight"))?,
        v: read_w2d(&mg, &format!("blk.{mtp_il}.attn_v.weight"))?,
        o: read_w2d(&mg, &format!("blk.{mtp_il}.attn_output.weight"))?,
        qn: read_w2d(&mg, &format!("blk.{mtp_il}.attn_q_norm.weight"))?,
        kn: read_w2d(&mg, &format!("blk.{mtp_il}.attn_k_norm.weight"))?,
        an: read_w2d(&mg, &format!("blk.{mtp_il}.hc_attn_norm.weight"))?,
        ad: read_w2d(&mg, &format!("blk.{mtp_il}.hc_attn_down.weight"))?,
        au: read_w2d(&mg, &format!("blk.{mtp_il}.hc_attn_up.weight"))?,
        ai: read_w2d(&mg, &format!("blk.{mtp_il}.hc_attn_inject.weight"))?,
        fnn: read_w2d(&mg, &format!("blk.{mtp_il}.hc_ffn_norm.weight"))?,
        fdd: read_w2d(&mg, &format!("blk.{mtp_il}.hc_ffn_down.weight"))?,
        fu: read_w2d(&mg, &format!("blk.{mtp_il}.hc_ffn_up.weight"))?,
        fi: read_w2d(&mg, &format!("blk.{mtp_il}.hc_ffn_inject.weight"))?,
        hn: read_w2d(&mg, &format!("blk.{mtp_il}.nextn.hc_head_norm.weight"))?,
        hd: read_w2d(&mg, &format!("blk.{mtp_il}.nextn.hc_head_down.weight"))?,
        hu: read_w2d(&mg, &format!("blk.{mtp_il}.nextn.hc_head_up.weight"))?,
    };
    let (n_, hc_, lr_) = (dims.n_embd, dims.hc, dims.hc_low_rank);
    if mtp.enorm.len() != n_
        || mtp.hnorm.len() != hc_ * n_
        || mtp.eh.len() != n_ * 2 * n_
        || mtp.q.len() != dims.n_head * 2 * dims.head_dim * n_
        || mtp.k.len() != dims.n_kv * dims.head_dim * n_
        || mtp.v.len() != dims.n_kv * dims.head_dim * n_
        || mtp.o.len() != n_ * dims.n_head * dims.head_dim
        || mtp.qn.len() != dims.head_dim
        || mtp.kn.len() != dims.head_dim
        || mtp.an.len() != hc_ * n_
        || mtp.ad.len() != lr_ * hc_ * n_
        || mtp.au.len() != hc_ * n_ * lr_
        || mtp.ai.len() != hc_ * hc_ * n_
        || mtp.fnn.len() != hc_ * n_
        || mtp.fdd.len() != lr_ * hc_ * n_
        || mtp.fu.len() != hc_ * n_ * lr_
        || mtp.fi.len() != hc_ * hc_ * n_
        || mtp.hn.len() != hc_ * n_
        || mtp.hd.len() != lr_ * hc_ * n_
        || mtp.hu.len() != hc_ * n_ * lr_
    {
        return Err("fn-chain: mtp 가중 길이 가드 위반".into());
    }
    // 합성 MoE(감소 모델 — 헤더 가중치 원장).
    let n_exp = CHAIN_MOE_EXPERTS;
    let mut rng = Rng::new(0x5EED_F00D_0000_0E01);
    let mk = |rng: &mut Rng, n: usize, amp: f64| -> Vec<u16> {
        (0..n)
            .map(|_| f32_to_f16(((rng.next_f64() * 2.0 - 1.0) * amp) as f32))
            .collect()
    };
    let moe_route = mk(&mut rng, n_exp * dims.n_embd, 0.02);
    let moe_route_sh = mk(&mut rng, dims.n_embd, 0.02);
    let moe_sh_gate = mk(&mut rng, dims.n_ff_shared * dims.n_embd, 0.03);
    let moe_sh_up = mk(&mut rng, dims.n_ff_shared * dims.n_embd, 0.03);
    let moe_sh_down = mk(&mut rng, dims.n_embd * dims.n_ff_shared, 0.05);
    let mut moe_experts = HashMap::new();
    for e in 0..n_exp as u32 {
        let mut er =
            Rng::new(0x5EED_F00D_0000_0E02 ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let g16 = mk(&mut er, dims.n_ff_exp * dims.n_embd, 0.03);
        let u16v = mk(&mut er, dims.n_ff_exp * dims.n_embd, 0.03);
        let d16 = mk(&mut er, dims.n_embd * dims.n_ff_exp, 0.05);
        moe_experts.insert(e, (g16, u16v, d16));
    }
    eprintln!(
        "[fn-chain] fixture: {gguf_main} | EXL3 ngram 헤더 일치 · quant stream {}B | real: emb/ple/gdn(36L)/qsa-norm(12L)/hc(48L×2)/head/out[0:{V_SLICE}] | synth: moe {}e top{} | proj: real per-visit reads",
        q.consumed_bytes, CHAIN_MOE_EXPERTS, dims.n_expert_used,
    );
    Ok(ChainFx {
        g,
        dims,
        mult,
        offs,
        vs,
        eos,
        ple_n_key,
        ple_n_query,
        ple_n_conv,
        ple_conv_w,
        ple_key_w,
        ple_value_w,
        gdn_ils,
        cw_all,
        dtb_all,
        ssa_all,
        nw_all,
        qsa_ils,
        qsa_norms,
        head_norm,
        head_down,
        head_up,
        out_rows,
        emb_cache: HashMap::new(),
        mtp,
        moe_route,
        moe_route_sh,
        moe_sh_gate,
        moe_sh_up,
        moe_sh_down,
        moe_experts,
    })
}

impl ChainFx {
    /// 토큰 임베딩 행(캐시 — Q8_0 직독 후 f32).
    fn emb_row(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        if let Some(r) = self.emb_cache.get(&tok) {
            return Ok(r.clone());
        }
        let raw = self.g.read_rows("token_embd.weight", tok as u64, 1)?;
        let row = dequant_q8_rows(&raw, 1, self.dims.n_embd);
        self.emb_cache.insert(tok, row.clone());
        Ok(row)
    }

    /// GDN 서수 상수 슬라이스.
    fn gdn_consts(&self, ord: usize) -> (&[f32], &[f32], &[f32], &[f32]) {
        let (cch, dr) = (self.dims.gdn_conv_ch(), self.dims.dt_rank);
        (
            &self.cw_all[ord * cch * 4..(ord + 1) * cch * 4],
            &self.dtb_all[ord * dr..(ord + 1) * dr],
            &self.ssa_all[ord * dr..(ord + 1) * dr],
            &self.nw_all[ord * 128..(ord + 1) * 128],
        )
    }

    /// HC 1세트(kind별) 실측 가중 직독(방문형).
    fn hc_weights(
        &self,
        il: usize,
        kind: &str,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let nrm = read_w2d(&self.g, &format!("blk.{il}.hc_{kind}_norm.weight"))?;
        let d = read_w2d(&self.g, &format!("blk.{il}.hc_{kind}_down.weight"))?;
        let u = read_w2d(&self.g, &format!("blk.{il}.hc_{kind}_up.weight"))?;
        let i = read_w2d(&self.g, &format!("blk.{il}.hc_{kind}_inject.weight"))?;
        let (hcn, lr) = (self.dims.hc * self.dims.n_embd, self.dims.hc_low_rank);
        if nrm.len() != hcn
            || d.len() != lr * hcn
            || u.len() != hcn * lr
            || i.len() != self.dims.hc * hcn
        {
            return Err(format!("hc 가중 형상 위반 il={il} {kind}"));
        }
        Ok((nrm, d, u, i))
    }

    /// GDN 투영 5종(방문형 실측 직독).
    fn gdn_proj(&self, il: usize) -> Result<GdnProj, String> {
        Ok(GdnProj {
            qkv: read_w2d(&self.g, &format!("blk.{il}.attn_qkv.weight"))?,
            z: read_w2d(&self.g, &format!("blk.{il}.attn_gate.weight"))?,
            b: read_w2d(&self.g, &format!("blk.{il}.ssm_beta.weight"))?,
            a: read_w2d(&self.g, &format!("blk.{il}.ssm_alpha.weight"))?,
            out: read_w2d(&self.g, &format!("blk.{il}.ssm_out.weight"))?,
        })
    }

    /// QSA 투영 6종.
    fn qsa_proj(&self, il: usize) -> Result<QsaProj, String> {
        Ok(QsaProj {
            q: read_w2d(&self.g, &format!("blk.{il}.attn_q.weight"))?,
            k: read_w2d(&self.g, &format!("blk.{il}.attn_k.weight"))?,
            v: read_w2d(&self.g, &format!("blk.{il}.attn_v.weight"))?,
            iq: read_w2d(&self.g, &format!("blk.{il}.indexer.q_proj.weight"))?,
            ik: read_w2d(&self.g, &format!("blk.{il}.indexer.k_proj.weight"))?,
            wo: read_w2d(&self.g, &format!("blk.{il}.attn_output.weight"))?,
        })
    }
}

// ── 음성대조 모드(결함은 모듈 체인 조립부에 주입) ──

#[derive(Clone, Copy, PartialEq)]
enum ChainNeg {
    Off,
    /// (a) 잘못된 층 순서 — GDN 서수 a↔b 슬롯(가중+상태) 매핑 교환.
    LayerOrderSwap(usize, usize),
    /// (b) 잘못된 잔차 부착 — il의 combine 가중 2σ(inj/4)→1.0 고정.
    ResidAttachFixed(usize),
    /// (c) 잘못된 수용/기각 순서 — 수용 판정이 tgt_out[i+1]을 사용
    /// (mtp_spec_step L352-354 검증 행 순서 오독 계급).
    AcceptOrderWrong,
}

// ── 체인 러너 ──

struct ChainOut {
    logits_md: Vec<f32>,
    tokens_mod: Vec<u32>,
    tokens_or: Vec<u32>,
    stats: HashMap<String, StageStat>,
    device: String,
    /// MTP 스펙 라운드 산출(제안·검증 토큰·수용 목록·검증 logits 행).
    spec: SpecOut,
}

/// 스펙 라운드 종착(모듈·오라클 쌍).
#[derive(Default)]
struct SpecOut {
    proposals_mod: Vec<u32>,
    proposals_or: Vec<u32>,
    tgt_mod: Vec<u32>,
    tgt_or: Vec<u32>,
    accepted_mod: Vec<u32>,
    accepted_or: Vec<u32>,
    verify_logits_md: Vec<f32>,
}

/// 모듈층 인스턴스 꾸러미(단일 상주 원칙 — 체인 프로브 수명).
struct ChainMods {
    ple: PleCuda,
    hc: HcCuda,
    qsa: QsaCuda,
    moe: MoeCuda,
    gdn: FnGdnCuda,
    mtp: MtpFnCuda,
}

/// 모듈 GDN 상주 상태의 호스트 미러(스테이지 오라클 pre-state용).
struct GdnMirror {
    ring: Vec<Vec<f32>>,
    st: Vec<Vec<f32>>,
}

/// QSA 모듈 캐시의 기댓값 미러(스텝별 append — 판독본 대조).
struct QsaMirror {
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    idx_k: Vec<f32>,
    idx_bk: Vec<f32>,
    n_blocks: usize,
}

fn det_tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| (rng.next_u64() % 262_144) as u32).collect()
}

/// t개 토큰의 PLE 표 원시 행 바이트 pread 팩(행 순서 rows[ti·heads]).
fn pack_table_rows(g: &FnGguf, rows: &[u32]) -> Result<Vec<u8>, String> {
    let mut raw = Vec::with_capacity(rows.len() * 90);
    for &r in rows {
        raw.extend_from_slice(&g.read_rows(PLE_TABLE_TENSOR, r as u64, 1)?);
    }
    Ok(raw)
}

fn stat<'a>(m: &'a mut HashMap<String, StageStat>, k: &str) -> &'a mut StageStat {
    m.entry(k.to_string()).or_default()
}

/// hc_combine — layers.rs L2576-2588(neg (b): il 지정층 가중 1.0 고정).
fn combine(
    res_hc: &mut [Vec<f32>],
    out: &[Vec<f32>],
    inject: &[Vec<f32>],
    hc: usize,
    il: usize,
    neg: ChainNeg,
) {
    for (t, o) in out.iter().enumerate() {
        for s in 0..hc {
            let w = match neg {
                ChainNeg::ResidAttachFixed(bad_il) if bad_il == il => 1.0,
                _ => 2.0 * sigmoid(inject[t][s] / hc as f32),
            };
            let base = s * o.len();
            for (i, ov) in o.iter().enumerate() {
                res_hc[t][base + i] += ov * w;
            }
        }
    }
}

/// 그리디 argmax(체인 수준 토큰 동일성 관찰 전용 — 값 판정은 maxdiff).
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// 체인 본체 — 모듈 체인(neg 결함 주입점)과 오라클 체인을 층·스텝
/// lock-step으로 진행하며 스테이지별 exact-입력 판정을 모은다.
#[allow(clippy::too_many_lines)]
fn run_chain(fx: &mut ChainFx, mods: &mut ChainMods, neg: ChainNeg) -> Result<ChainOut, String> {
    let dims = fx.dims.clone();
    let (n, hc, hcn) = (dims.n_embd, dims.hc, dims.hc * dims.n_embd);
    let (lr, eps) = (dims.hc_low_rank, dims.eps);
    let (cch, v_len_g, dr) = (dims.gdn_conv_ch(), dims.gdn_v_len(), dims.dt_rank);
    let r = 4usize;
    let mut qsa_or = QsaChainOracle::new(
        fx.qsa_ils.len(),
        QSA_CAP,
        dims.idx_dim,
        dims.n_kv,
        dims.head_dim,
    );
    let mut gdn_mirror = GdnMirror {
        ring: vec![vec![0.0; 3 * cch]; fx.gdn_ils.len()],
        st: vec![vec![0.0; dr * 128 * 128]; fx.gdn_ils.len()],
    };
    let mut gdn_mirror_or = GdnMirror {
        ring: vec![vec![0.0; 3 * cch]; fx.gdn_ils.len()],
        st: vec![vec![0.0; dr * 128 * 128]; fx.gdn_ils.len()],
    };
    let mut qsa_mirror: Vec<QsaMirror> = (0..fx.qsa_ils.len())
        .map(|_| QsaMirror {
            kv_k: Vec::new(),
            kv_v: Vec::new(),
            idx_k: Vec::new(),
            idx_bk: Vec::new(),
            n_blocks: 0,
        })
        .collect();
    let ple_hist_len = (dims.ple_conv_k - 1) * dims.ple_ngram;
    let mut ple_hist: Vec<u32> = Vec::new();
    let mut ple_st_or = vec![0.0f32; ple_hist_len * hcn];
    let toks0 = det_tokens(2, 0x5EED_F00D_0000_C0DE);
    let mut prev_tok: Option<u32> = None;
    let mut logits_md_all = Vec::new();
    let mut toks_mod = Vec::new();
    let mut toks_or = Vec::new();
    let mut stats: HashMap<String, StageStat> = HashMap::new();
    let device = mods.gdn.cc.device_name.to_string();

    mods.qsa.set_pos(0)?;

    let mut verify_toks: Vec<u32> = Vec::new();
    let mut spec = SpecOut::default();
    let mut spec_h_prev_mod: Vec<f32> = Vec::new();
    let mut spec_h_prev_or: Vec<f32> = Vec::new();
    for step in 0..STEPS + 1 {
        let verify = step == STEPS;
        let t = if verify || step == 0 { 2 } else { 1 };
        let toks: Vec<u32> = if verify {
            verify_toks.clone()
        } else if step == 0 {
            toks0.clone()
        } else {
            vec![prev_tok.ok_or("체인: argmax 토큰 부재")?]
        };
        let pos0 = qsa_or.pos;
        // ── 임베딩 방송(emb_broadcast_write 미러 — 공유 입력) ──
        let tok_emb: Vec<Vec<f32>> = {
            let mut v = Vec::with_capacity(t);
            for &tok in &toks {
                v.push(fx.emb_row(tok)?);
            }
            v
        };
        let mut res_mod: Vec<Vec<f32>> = tok_emb
            .iter()
            .map(|row| {
                let mut rrow = vec![0.0f32; hcn];
                for s in 0..hc {
                    rrow[s * n..(s + 1) * n].copy_from_slice(row);
                }
                rrow
            })
            .collect();
        let mut res_or = res_mod.clone();
        // ── PLE n-gram 해시(모듈 fn vs 오라클 쌍 — 정수 판정) ──
        let (rows_mod, hist_mod) = ple_hash_rows(
            &ple_hist,
            step > 0,
            &toks,
            dims.ple_ngram,
            dims.ple_heads_per_ngram,
            &fx.mult,
            &fx.offs,
            &fx.vs,
            fx.eos,
        );
        let (rows_or, hist_or) = oracle_hash_rows(
            &ple_hist,
            step > 0,
            &toks,
            dims.ple_ngram,
            dims.ple_heads_per_ngram,
            &fx.mult,
            &fx.offs,
            &fx.vs,
            fx.eos,
        );
        let hash_bad = rows_mod
            .iter()
            .zip(&rows_or)
            .filter(|(a, b)| a != b)
            .count()
            + usize::from(hist_mod != hist_or);
        stat(&mut stats, "ple.hash").update_int(&format!("s{step}"), hash_bad);
        ple_hist = hist_mod;
        let raw_rows = pack_table_rows(&fx.g, &rows_mod)?;
        let heads = dims.ple_heads_per_ngram * 2;
        let emb_w = heads * dims.ple_head_dim;
        let mut ple_emb = vec![0.0f32; t * emb_w];
        for (hi, chunk) in ple_emb.chunks_mut(dims.ple_head_dim).enumerate() {
            deq_iq4_nl_row(&raw_rows[hi * 90..(hi + 1) * 90], dims.ple_head_dim, chunk);
        }
        let ple_emb_rows: Vec<Vec<f32>> = ple_emb.chunks(emb_w).map(<[f32]>::to_vec).collect();

        // ── 층 루프(frame_forward_ex 스테이지 발행 순서) ──
        for il in 0..dims.n_layer {
            // GDN 서수(음성대조 (a): 모듈측 슬롯 교환).
            let ord_correct = fx
                .gdn_ils
                .iter()
                .position(|&x| x == il)
                .unwrap_or(usize::MAX);
            let ord_mod = match neg {
                ChainNeg::LayerOrderSwap(a, b) if ord_correct == a => b,
                ChainNeg::LayerOrderSwap(a, b) if ord_correct == b => a,
                _ => ord_correct,
            };

            // 1) PLE(blk.1 — hc attn mix 앞, res_hc 제자리 잔차).
            if dims.is_ple(il) {
                let key_flat: Vec<f32> = ple_emb_rows
                    .iter()
                    .flat_map(|e| par_dot_rows(e, &fx.ple_key_w, hcn, emb_w))
                    .collect();
                let value_flat: Vec<f32> = ple_emb_rows
                    .iter()
                    .flat_map(|e| par_dot_rows(e, &fx.ple_value_w, n, emb_w))
                    .collect();
                let key: Vec<Vec<f32>> = key_flat.chunks(hcn).map(<[f32]>::to_vec).collect();
                let value: Vec<Vec<f32>> = value_flat.chunks(n).map(<[f32]>::to_vec).collect();
                mods.ple.ple_gather_stage(&raw_rows, t)?;
                let emb_dev = mods.ple.ple_emb_download(t)?;
                stat(&mut stats, "ple.gather").update(&format!("s{step}"), &emb_dev, &ple_emb);
                let pre_state = mods.ple.ple_conv_state()?;
                if step > 0 && pre_state.iter().all(|&v| v == 0.0) {
                    return Err("fn-chain: PLE 2스텝째 상태 전부 0 — S0≠0 정신 위반".into());
                }
                let mut res_mod_flat: Vec<f32> = res_mod.concat();
                let mut mids = PleMids::default();
                mods.ple
                    .ple_block(t, &key_flat, &value_flat, &mut res_mod_flat, &mut mids)?;
                // 스테이지 오라클 — 모듈 입력·모듈 pre 상태로 exact-입력 재생.
                let mut res_stage = res_mod.clone();
                let mut st_stage = pre_state.clone();
                let (g_o, gt_o, c_o) = oracle_ple_block(
                    &key,
                    &value,
                    &fx.ple_n_key,
                    &fx.ple_n_query,
                    &fx.ple_n_conv,
                    &fx.ple_conv_w,
                    &mut res_stage,
                    &mut st_stage,
                    hc,
                    n,
                    dims.ple_conv_k,
                    dims.ple_ngram,
                    eps,
                );
                let sp = &mut stat(&mut stats, "ple.block");
                sp.update(&format!("s{step}.gates"), &mids.gates, &g_o.concat());
                sp.update(&format!("s{step}.gated"), &mids.gated, &gt_o.concat());
                sp.update(&format!("s{step}.conv"), &mids.conv_out, &c_o.concat());
                sp.update(&format!("s{step}.res"), &res_mod_flat, &res_stage.concat());
                let st_dev = mods.ple.ple_conv_state()?;
                sp.update(&format!("s{step}.state"), &st_dev, &st_stage);
                res_mod = res_mod_flat.chunks(hcn).map(<[f32]>::to_vec).collect();
                // 오라클 체인 경로(자기 상태 진화).
                let _ = oracle_ple_block(
                    &key,
                    &value,
                    &fx.ple_n_key,
                    &fx.ple_n_query,
                    &fx.ple_n_conv,
                    &fx.ple_conv_w,
                    &mut res_or,
                    &mut ple_st_or,
                    hc,
                    n,
                    dims.ple_conv_k,
                    dims.ple_ngram,
                    eps,
                );
            }

            // 2) hc attn mix — 실측 가중(레지스트리 키 단일화 (0,kind)).
            let (wn, wd, wu, wi) = fx.hc_weights(il, "attn")?;
            mods.hc.register(0, "attn", &wn, &wd, &wu, Some(&wi))?;
            let (mix_m, inj_m) = mods.hc.hc_mix(0, "attn", &res_mod)?;
            let (mix_stage, inj_stage) =
                hc_mix_ex_ref(hc, n, lr, eps, &wn, &wd, &wu, Some(&wi), &res_mod);
            let sa = &mut stat(&mut stats, "hc.attn");
            sa.update(
                &format!("s{step}.mixed"),
                &mix_m.concat(),
                &mix_stage.concat(),
            );
            sa.update(
                &format!("s{step}.inj"),
                &inj_m.concat(),
                &inj_stage.concat(),
            );
            let (mix_o, inj_o) = hc_mix_ex_ref(hc, n, lr, eps, &wn, &wd, &wu, Some(&wi), &res_or);

            // 3) attention — GDN(is_recr) | QSA.
            let (attn_m, attn_o): (Vec<Vec<f32>>, Vec<Vec<f32>>);
            if dims.is_recr(il) {
                // 모듈: (neg시 교환) 슬롯 가중·상주 상태.
                let pj_m = fx.gdn_proj(fx.gdn_ils[ord_mod])?;
                let qkv_m: Vec<f32> = mix_m
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_m.qkv, cch, n))
                    .collect();
                let z_m: Vec<f32> = mix_m
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_m.z, v_len_g, n))
                    .collect();
                let b_m: Vec<f32> = mix_m
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_m.b, dr, n))
                    .collect();
                let a_m: Vec<f32> = mix_m
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_m.a, dr, n))
                    .collect();
                let gated_m = mods
                    .gdn
                    .gdn_stage_host(ord_mod, t, &qkv_m, &z_m, &b_m, &a_m, None, None)?;
                let mids = mods.gdn.gdn_mids_host(ord_mod, t)?;
                let (cw_l, dtb_l, ssa_l, nw_l) = fx.gdn_consts(ord_mod);
                let refm = gdn_reference(
                    &dims,
                    eps,
                    cw_l,
                    dtb_l,
                    ssa_l,
                    nw_l,
                    &qkv_m,
                    &z_m,
                    &b_m,
                    &a_m,
                    &gdn_mirror.ring[ord_mod],
                    &gdn_mirror.st[ord_mod],
                    t,
                );
                let sg = &mut stat(&mut stats, "gdn.stage");
                // FNF 원장 등급: conv/prep 비트동일, scan/gated/상태 값.
                sg.update(&format!("s{step}.convq"), &mids.conv_q, &refm.conv_q);
                sg.update(&format!("s{step}.convk"), &mids.conv_k, &refm.conv_k);
                sg.update(&format!("s{step}.convv"), &mids.conv_v, &refm.conv_v);
                sg.update(&format!("s{step}.q2"), &mids.q2, &refm.q2);
                sg.update(&format!("s{step}.k2"), &mids.k2, &refm.k2);
                sg.update(&format!("s{step}.o"), &mids.o, &refm.o);
                sg.update(&format!("s{step}.gated"), &mids.gated, &refm.gated);
                sg.update(&format!("s{step}.ring"), &mids.ring_post, &refm.ring_post);
                sg.update(&format!("s{step}.state"), &mids.st_post, &refm.st_post);
                gdn_mirror.ring[ord_mod] = mids.ring_post.clone();
                gdn_mirror.st[ord_mod] = mids.st_post.clone();
                attn_m = gated_m
                    .chunks(v_len_g)
                    .map(|g| par_dot_rows(g, &pj_m.out, n, v_len_g))
                    .collect();
                // 오라클 체인 경로 — 정상 서수 가중·오라클 상태.
                let pj_o = if ord_mod == ord_correct {
                    pj_m
                } else {
                    fx.gdn_proj(il)?
                };
                let (cw_o, dtb_o, ssa_o, nw_o) = fx.gdn_consts(ord_correct);
                let qkv_o: Vec<f32> = mix_o
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_o.qkv, cch, n))
                    .collect();
                let z_o: Vec<f32> = mix_o
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_o.z, v_len_g, n))
                    .collect();
                let b_o: Vec<f32> = mix_o
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_o.b, dr, n))
                    .collect();
                let a_o: Vec<f32> = mix_o
                    .iter()
                    .flat_map(|x| par_dot_rows(x, &pj_o.a, dr, n))
                    .collect();
                let refo = gdn_reference(
                    &dims,
                    eps,
                    cw_o,
                    dtb_o,
                    ssa_o,
                    nw_o,
                    &qkv_o,
                    &z_o,
                    &b_o,
                    &a_o,
                    &gdn_mirror_or.ring[ord_correct],
                    &gdn_mirror_or.st[ord_correct],
                    t,
                );
                gdn_mirror_or.ring[ord_correct] = refo.ring_post.clone();
                gdn_mirror_or.st[ord_correct] = refo.st_post.clone();
                attn_o = refo
                    .gated
                    .chunks(v_len_g)
                    .map(|g| par_dot_rows(g, &pj_o.out, n, v_len_g))
                    .collect();
            } else {
                let fi = fx
                    .qsa_ils
                    .iter()
                    .position(|&x| x == il)
                    .ok_or("fn-chain: QSA fi 매핑 실패")?;
                let pj = fx.qsa_proj(il)?;
                let [knw, qnw, iqw, ikw] = &fx.qsa_norms[fi];
                let qg_m: Vec<Vec<f32>> = mix_m
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.q, dims.n_head * 2 * dims.head_dim, n))
                    .collect();
                let kk_m: Vec<Vec<f32>> = mix_m
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.k, dims.n_kv * dims.head_dim, n))
                    .collect();
                let vv_m: Vec<Vec<f32>> = mix_m
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.v, dims.n_kv * dims.head_dim, n))
                    .collect();
                let iq_m: Vec<Vec<f32>> = mix_m
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.iq, dims.idx_heads * dims.idx_dim, n))
                    .collect();
                let ik_m: Vec<Vec<f32>> = mix_m
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.ik, dims.idx_dim, n))
                    .collect();
                // 모듈: 선택 리스트(판정용 선행 호출) → 스테이지(내부 재선택).
                let (sel_blk_m, sel_cnt_m, sel_stride) =
                    mods.qsa
                        .qsa_select(il, &kk_m, &vv_m, &iq_m, &ik_m, t, false, QsaNeg::Off)?;
                let attn_raw = mods
                    .qsa
                    .qsa_stage(il, &qg_m, &kk_m, &vv_m, &iq_m, &ik_m, t, false)?;
                // 기댓값 미러 갱신(모듈 입력 기반 — 판독본 전체 대조).
                let mir = &mut qsa_mirror[fi];
                mir.kv_k
                    .extend_from_slice(&qsa_stage_k_rows_ref(&dims, knw, &kk_m, t, pos0));
                let kv_v_flat: Vec<f32> = vv_m.concat();
                mir.kv_v.extend_from_slice(&kv_v_flat);
                let ik_flat: Vec<f32> = ik_m.concat();
                mir.idx_k.extend_from_slice(&ik_flat);
                let n_blocks_new = (pos0 + t) / r;
                for b in mir.n_blocks..n_blocks_new {
                    let pk =
                        pool_block_ref(&mir.idx_k, b, r, dims.idx_dim, ikw, eps, dims.rope_base);
                    mir.idx_bk.extend_from_slice(&pk);
                }
                mir.n_blocks = n_blocks_new;
                // 판독본 대조(전체 행 — 과거 스텝 포함 상태 검증).
                let (kv_k_rb, kv_v_rb) = mods.qsa.read_kv(fi, pos0 + t)?;
                let idx_k_rb = mods.qsa.read_idx_k(fi, pos0 + t)?;
                // n_blocks=0(첫 r 토큰 미만)일빈는 판독 생략 — 0길이 d2h는
                // 모듈 계약 밖(qsa_cuda.rs read_idx_bk from_raw_parts 가드).
                let idx_bk_rb = if n_blocks_new > 0 {
                    let rb = mods.qsa.read_idx_bk(fi, n_blocks_new)?;
                    stat(&mut stats, "qsa.bk").update(&format!("s{step}L{il}"), &rb, &mir.idx_bk);
                    rb
                } else {
                    Vec::new()
                };
                stat(&mut stats, "qsa.kvk").update(&format!("s{step}L{il}"), &kv_k_rb, &mir.kv_k);
                stat(&mut stats, "qsa.kvv").update(&format!("s{step}L{il}"), &kv_v_rb, &mir.kv_v);
                stat(&mut stats, "qsa.idxk").update(
                    &format!("s{step}L{il}"),
                    &idx_k_rb,
                    &mir.idx_k,
                );
                stat(&mut stats, "qsa.bk").update(
                    &format!("s{step}L{il}"),
                    &idx_bk_rb,
                    &mir.idx_bk,
                );
                // 선택 리스트 exact-입력 판정(모듈 iq·모듈 idx_bk 판독본).
                let (sel_blk_w, sel_cnt_w, _) = if n_blocks_new > 0 {
                    qsa_stage_pass_b_ref(&dims, iqw, &iq_m, &idx_bk_rb, pos0, t)
                } else {
                    (vec![0u32; t * sel_stride], vec![0u32; t], sel_stride)
                };
                let sel_bad = sel_cnt_m
                    .iter()
                    .zip(&sel_cnt_w)
                    .filter(|(a, b)| a != b)
                    .count()
                    + sel_blk_m
                        .iter()
                        .zip(&sel_blk_w)
                        .filter(|(a, b)| a != b)
                        .count();
                stat(&mut stats, "qsa.sel").update_int(&format!("s{step}L{il}"), sel_bad);
                // 어텐션 exact-입력 판정(모듈 qg·판독 캐시·모듈 sel).
                let attn_want = QsaChainOracle::attn_ref(
                    &kv_k_rb, &kv_v_rb, &qg_m, qnw, &sel_blk_m, &sel_cnt_m, sel_stride, r, t, pos0,
                    &dims,
                );
                stat(&mut stats, "qsa.attn").update(
                    &format!("s{step}L{il}"),
                    &attn_raw.concat(),
                    &attn_want.concat(),
                );
                attn_m = attn_raw
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.wo, n, dims.n_head * dims.head_dim))
                    .collect();
                // 오라클 체인 경로(자기 캐시 진행).
                let qg_o: Vec<Vec<f32>> = mix_o
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.q, dims.n_head * 2 * dims.head_dim, n))
                    .collect();
                let kk_o: Vec<Vec<f32>> = mix_o
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.k, dims.n_kv * dims.head_dim, n))
                    .collect();
                let vv_o: Vec<Vec<f32>> = mix_o
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.v, dims.n_kv * dims.head_dim, n))
                    .collect();
                let iq_o: Vec<Vec<f32>> = mix_o
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.iq, dims.idx_heads * dims.idx_dim, n))
                    .collect();
                let ik_o: Vec<Vec<f32>> = mix_o
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.ik, dims.idx_dim, n))
                    .collect();
                let (sel_o, cnt_o, stride_o) =
                    qsa_or.select(fi, &dims, knw, iqw, ikw, &kk_o, &vv_o, &iq_o, &ik_o, t);
                let slot = &qsa_or.fi[fi];
                let attn_rows_o = QsaChainOracle::attn_ref(
                    &slot.kv_k, &slot.kv_v, &qg_o, qnw, &sel_o, &cnt_o, stride_o, r, t, pos0, &dims,
                );
                attn_o = attn_rows_o
                    .iter()
                    .map(|x| par_dot_rows(x, &pj.wo, n, dims.n_head * dims.head_dim))
                    .collect();
            }
            // 4) hc_combine(attn).
            combine(&mut res_mod, &attn_m, &inj_m, hc, il, neg);
            combine(&mut res_or, &attn_o, &inj_o, hc, il, ChainNeg::Off);

            // 5) hc ffn mix.
            let (wn2, wd2, wu2, wi2) = fx.hc_weights(il, "ffn")?;
            mods.hc.register(0, "ffn", &wn2, &wd2, &wu2, Some(&wi2))?;
            let (mixf_m, injf_m) = mods.hc.hc_mix(0, "ffn", &res_mod)?;
            let (mixf_stage, injf_stage) =
                hc_mix_ex_ref(hc, n, lr, eps, &wn2, &wd2, &wu2, Some(&wi2), &res_mod);
            let sf = &mut stat(&mut stats, "hc.ffn");
            sf.update(
                &format!("s{step}.mixed"),
                &mixf_m.concat(),
                &mixf_stage.concat(),
            );
            sf.update(
                &format!("s{step}.inj"),
                &injf_m.concat(),
                &injf_stage.concat(),
            );
            let (mixf_o, injf_o) =
                hc_mix_ex_ref(hc, n, lr, eps, &wn2, &wd2, &wu2, Some(&wi2), &res_or);

            // 6) MoE(감소 모델 합성 — 모듈·오라클 동일 바이트).
            let out_m = mods.moe.moe_ffn(&mixf_m)?;
            let route_m = mods.moe.moe_route(&mixf_m)?;
            let (route_w, _) = moe_route_ref(
                &fx.moe_route,
                &fx.moe_route_sh,
                CHAIN_MOE_EXPERTS,
                dims.n_expert_used,
                &mixf_m,
            );
            let mut ids_bad = 0usize;
            let mut w_md = 0.0f32;
            for (gm, wm) in route_m.iter().zip(route_w.iter()) {
                if gm.len() != wm.len() || gm.iter().ne(wm.iter()) {
                    ids_bad += 1;
                }
                for (a, b) in gm.iter().zip(wm.iter()) {
                    w_md = w_md.max((a.1 - b.1).abs());
                }
            }
            let sm = &mut stat(&mut stats, "moe.route");
            sm.update_int(&format!("s{step}L{il}.ids"), ids_bad);
            sm.note_max(&format!("s{step}L{il}.w"), w_md);
            let out_w = moe_ffn_ref_chain(
                n,
                dims.n_ff_exp,
                dims.n_ff_shared,
                dims.n_expert_used,
                &fx.moe_route,
                &fx.moe_route_sh,
                &fx.moe_sh_gate,
                &fx.moe_sh_up,
                &fx.moe_sh_down,
                &fx.moe_experts,
                &mixf_m,
            );
            stat(&mut stats, "moe.out").update(
                &format!("s{step}L{il}"),
                &out_m.concat(),
                &out_w.concat(),
            );
            let out_o = moe_ffn_ref_chain(
                n,
                dims.n_ff_exp,
                dims.n_ff_shared,
                dims.n_expert_used,
                &fx.moe_route,
                &fx.moe_route_sh,
                &fx.moe_sh_gate,
                &fx.moe_sh_up,
                &fx.moe_sh_down,
                &fx.moe_experts,
                &mixf_o,
            );

            // 7) hc_combine(ffn).
            combine(&mut res_mod, &out_m, &injf_m, hc, il, neg);
            combine(&mut res_or, &out_o, &injf_o, hc, il, ChainNeg::Off);
        }

        // ── 헤드 + logits + argmax(검증 스텝은 전 행 logits) ──
        mods.hc.register(
            HC_HEAD_IL,
            "head",
            &fx.head_norm,
            &fx.head_down,
            &fx.head_up,
            None,
        )?;
        let hin_m_rows = mods.hc.hc_mix_head(&res_mod)?;
        let (hin_stage, _) = hc_mix_ex_ref(
            hc,
            n,
            lr,
            eps,
            &fx.head_norm,
            &fx.head_down,
            &fx.head_up,
            None,
            &res_mod,
        );
        stat(&mut stats, "hc.head").update(
            &format!("s{step}"),
            &hin_m_rows.concat(),
            &hin_stage.concat(),
        );
        let (hin_or_rows, _) = hc_mix_ex_ref(
            hc,
            n,
            lr,
            eps,
            &fx.head_norm,
            &fx.head_down,
            &fx.head_up,
            None,
            &res_or,
        );
        let mut logits_rows_m: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut logits_rows_o: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (hm, ho) in hin_m_rows.iter().zip(hin_or_rows.iter()) {
            logits_rows_m.push(par_dot_rows(hm, &fx.out_rows, V_SLICE, n));
            logits_rows_o.push(par_dot_rows(ho, &fx.out_rows, V_SLICE, n));
        }
        if verify {
            // 검증 스텝 — 전 행 logits·argmax(frame_forward_verify 종결부).
            let mut vmd = 0.0f32;
            for (i, (lm, lo)) in logits_rows_m.iter().zip(logits_rows_o.iter()).enumerate() {
                let (md, nan) = maxdiff_nan(lm, lo);
                if nan > 0 {
                    return Err(format!("fn-chain: 검증 행{i} logits NaN {nan}건"));
                }
                vmd = vmd.max(md);
                stat(&mut stats, "verify.logits").update(&format!("v{i}"), lm, lo);
            }
            let tgt_m: Vec<u32> = logits_rows_m.iter().map(|l| argmax(l)).collect();
            let tgt_o: Vec<u32> = logits_rows_o.iter().map(|l| argmax(l)).collect();
            spec.tgt_mod = tgt_m.clone();
            spec.tgt_or = tgt_o.clone();
            // 수용/기각 — layers.rs mtp_spec_step L350-383(neg (c): 검증
            // 행 순서 오독 — tgt[i+1]로 판정).
            spec.accepted_mod = accept_prefix(&spec.proposals_mod, &tgt_m, neg);
            spec.accepted_or = accept_prefix(&spec.proposals_or, &tgt_o, ChainNeg::Off);
            eprintln!(
                "[fn-chain] verify t=2 pos0={pos0}: rows logits maxdiff={vmd:.3e} tgt mod={tgt_m:?} or={tgt_o:?} | proposals mod={:?} or={:?} | accepted mod={:?} or={:?}",
                spec.proposals_mod, spec.proposals_or, spec.accepted_mod, spec.accepted_or,
            );
            stat(&mut stats, "spec.tgt").update_int(
                "verify",
                tgt_m.iter().zip(&tgt_o).filter(|(a, b)| a != b).count(),
            );
            stat(&mut stats, "spec.accepted").update_int(
                "round",
                if spec.accepted_mod == spec.accepted_or {
                    0
                } else {
                    1
                },
            );
        } else {
            let logits_m = &logits_rows_m[t - 1];
            let logits_o = &logits_rows_o[t - 1];
            let (md, nan) = maxdiff_nan(logits_m, logits_o);
            if nan > 0 {
                return Err(format!("fn-chain: s{step} logits NaN {nan}건"));
            }
            stat(&mut stats, "logits").update(&format!("s{step}"), logits_m, logits_o);
            let tok_m = argmax(logits_m);
            let tok_o = argmax(logits_o);
            eprintln!(
                "[fn-chain] step{step} t={t} pos0={pos0}: logits maxdiff={md:.3e} tok mod={tok_m} or={tok_o} {}",
                if tok_m == tok_o { "MATCH" } else { "MISMATCH" }
            );
            logits_md_all.push(md);
            toks_mod.push(tok_m);
            toks_or.push(tok_o);
            prev_tok = Some(tok_m);
            if step == 1 {
                // 스펙 h_prev 시드 — 직전 커밋 토큰의 pre-decode 은닉
                // (mtp_spec_step L311-315: spec_h_prev/last_res_hc 캡처).
                spec_h_prev_mod = res_mod[0].clone();
                spec_h_prev_or = res_or[0].clone();
            }
        }
        // QSA pos 진행(청크 종료 1회 — pp[0] += t, 오라클 pos 동일).
        mods.qsa.pos_bump(t as u32)?;
        qsa_or.pos += t;

        // ── MTP 스펙 라운드(그리디 3스텝 직후 — mtp_spec_step L288-386) ──
        if step + 1 == STEPS {
            let h_after_m = res_mod[0].clone();
            let h_after_o = res_or[0].clone();
            let last_tok = toks[0];
            let kv_dim = dims.n_kv * dims.head_dim;
            let mut st_mm = MtpOracleSt::new(QSA_CAP, kv_dim);
            let mut st_or = MtpOracleSt::new(QSA_CAP, kv_dim);
            // 모듈 reset(3)과 동일 시드 pos(드래프트 KV는 트렁크 pos와
            // cell-for-cell — KV[3]가 직전 커밋 토큰 진위치).
            st_mm.pos = 3;
            st_or.pos = 3;
            // KV 시드 — 직전 커밋 토큰 (en, hn=h_prev) 진위치 pos=3 기입
            // (출력 폐기 — mtp_spec_step L320-329 계약).
            let e_seed = fx.emb_row(last_tok)?;
            let en_seed = rms_norm(&e_seed, &fx.mtp.enorm, eps);
            let hn_seed_m = rms_norm(&spec_h_prev_mod, &fx.mtp.hnorm, eps);
            let hn_seed_o = rms_norm(&spec_h_prev_or, &fx.mtp.hnorm, eps);
            mods.mtp.reset(3)?;
            let seed_m = mods.mtp.mtp_draft_step(&en_seed, &hn_seed_m)?;
            let seed_o_mm = mtp_draft_oracle(fx, &mut st_mm, &en_seed, &hn_seed_m);
            gate_mtp(&mut stats, "seed", &seed_m, &seed_o_mm);
            mods.mtp.pos += 1;
            st_mm.pos += 1;
            let _ = mtp_draft_oracle(fx, &mut st_or, &en_seed, &hn_seed_o);
            st_or.pos += 1;
            // 드래프트 체인 — k-1회: draft(next, chain_h) → push(next)
            // (L330-342). t0는 트렁크 그리디(step2 argmax).
            let t0 = prev_tok.ok_or("체인: t0 부재")?;
            let mut chain_h_m = h_after_m;
            let mut chain_h_o = h_after_o;
            let mut next_m = t0;
            let mut next_o = t0;
            let mut proposals_m: Vec<u32> = Vec::new();
            let mut proposals_or: Vec<u32> = Vec::new();
            for i in 0..MTP_K - 1 {
                let e_m = fx.emb_row(next_m)?;
                let en_m = rms_norm(&e_m, &fx.mtp.enorm, eps);
                let hn_m = rms_norm(&chain_h_m, &fx.mtp.hnorm, eps);
                let mids = mods.mtp.mtp_draft_step(&en_m, &hn_m)?;
                let ori = mtp_draft_oracle(fx, &mut st_mm, &en_m, &hn_m);
                gate_mtp(&mut stats, &format!("d{i}"), &mids, &ori);
                proposals_m.push(next_m);
                chain_h_m = mids.chain_h.clone();
                let dm = mids.token.ok_or("mtp draft: token 부재(헤드 미등록)")?;
                mods.mtp.pos += 1;
                st_mm.pos += 1;
                // 오라클 체인 경로(자기 h·토큰).
                let e_o = fx.emb_row(next_o)?;
                let en_o = rms_norm(&e_o, &fx.mtp.enorm, eps);
                let hn_o = rms_norm(&chain_h_o, &fx.mtp.hnorm, eps);
                let oro = mtp_draft_oracle(fx, &mut st_or, &en_o, &hn_o);
                proposals_or.push(next_o);
                chain_h_o = oro.chain_h.clone();
                next_o = oro.token;
                next_m = dm;
                st_or.pos += 1;
            }
            stat(&mut stats, "spec.proposals")
                .update_int("draft", if proposals_m == proposals_or { 0 } else { 1 });
            eprintln!(
                "[fn-chain] mtp draft: proposals mod={proposals_m:?} or={proposals_or:?} (k={MTP_K})",
            );
            spec.proposals_mod = proposals_m.clone();
            spec.proposals_or = proposals_or.clone();
            verify_toks = proposals_m;
        }
    }
    Ok(ChainOut {
        logits_md: logits_md_all,
        tokens_mod: toks_mod,
        tokens_or: toks_or,
        stats,
        device,
        spec,
    })
}

/// MTP 드래프트 스테이지 판정 등록 — pre-MoE 비트동일(FNG 계급:
/// gemv·combine·믹서는 연산별 반올림 미러), post-MoE 값(§3.4 ≤2e-4).
fn gate_mtp(stats: &mut HashMap<String, StageStat>, tag: &str, m: &MtpMids, o: &MtpOrMids) {
    let sp = stat(stats, "mtp.pre");
    sp.update(&format!("{tag}.eh"), &m.eh, &o.eh);
    sp.update(&format!("{tag}.mix1"), &m.mix_attn, &o.mix_attn);
    sp.update(&format!("{tag}.inj1"), &m.inj_attn, &o.inj_attn);
    sp.update(&format!("{tag}.attn"), &m.attn, &o.attn);
    sp.update(&format!("{tag}.ao"), &m.ao, &o.ao);
    sp.update(&format!("{tag}.resattn"), &m.res_attn, &o.res_attn);
    sp.update(&format!("{tag}.mix2"), &m.mix_ffn, &o.mix_ffn);
    sp.update(&format!("{tag}.inj2"), &m.inj_ffn, &o.inj_ffn);
    let sp = stat(stats, "mtp.post");
    sp.update(&format!("{tag}.mout"), &m.mout, &o.mout);
    sp.update(&format!("{tag}.chainh"), &m.chain_h, &o.chain_h);
    sp.update(&format!("{tag}.hin"), &m.hin, &o.hin);
    sp.update(&format!("{tag}.logits"), &m.logits, &o.logits);
    stat(stats, "mtp.token").update_int(tag, usize::from(m.token != Some(o.token)));
}

/// 스펙 수용 접두 판정 — layers.rs mtp_spec_step L350-383. neg (c)는 검증
/// 행 배선을 1 어긋나 오독(수용 판정·보정 토듶 모두 tgt[i+1] —
/// hcmp 토듶/위치 비정렬 계급, 원장 사고 3호).
/// [탐지 불변 조건] 결함과 정답이 같은 목록을 내는 유일한 경우는
/// d1==tgt0==tgt1(전 행 일치 — 어떤 순서로 판정해도 같은 답)뿐이다.
fn accept_prefix(proposals: &[u32], tgt: &[u32], neg: ChainNeg) -> Vec<u32> {
    let len = proposals.len();
    let t0 = proposals[0];
    let mut n_acc = len;
    for i in 0..len {
        let Some(e) = proposals.get(i + 1) else {
            continue;
        };
        let judge = match neg {
            ChainNeg::AcceptOrderWrong => tgt.get(i + 1).copied().unwrap_or(u32::MAX),
            _ => tgt[i],
        };
        if judge != *e {
            n_acc = i;
            break;
        }
    }
    let wrong = matches!(neg, ChainNeg::AcceptOrderWrong);
    let correction = |idx: usize| tgt[idx.min(tgt.len() - 1)];
    let mut accepted = Vec::with_capacity(n_acc + 2);
    accepted.push(t0);
    if n_acc + 1 >= len {
        accepted.extend_from_slice(&proposals[1..]);
        accepted.push(*tgt.last().unwrap_or(&t0));
    } else if wrong {
        accepted.extend_from_slice(&proposals[1..=n_acc]);
        accepted.push(correction(n_acc + 1));
    } else {
        accepted.extend_from_slice(&proposals[1..=n_acc]);
        accepted.push(tgt[n_acc]);
    }
    accepted
}

// ── 스테이지 표 판정 ──

/// 게이트 종류 — Bits(bitdiff=0)·Max(값 임계)·IntBits(정수 불일치 0)+
/// Max 겸용(moe.route).
#[derive(Clone, Copy, PartialEq)]
enum Gate {
    Bits,
    Max(f32),
    MaxInt(f32),
    IntBits,
}

const STAGE_GATES: &[(&str, Gate)] = &[
    ("ple.hash", Gate::IntBits),
    ("ple.gather", Gate::Bits),
    ("ple.block", Gate::Bits),
    ("hc.attn", Gate::Bits),
    ("hc.ffn", Gate::Bits),
    ("hc.head", Gate::Bits),
    ("gdn.stage", Gate::Max(CHAIN_GDN_THRESH)),
    ("qsa.kvk", Gate::Max(CHAIN_QSA_THRESH)),
    ("qsa.kvv", Gate::Bits),
    ("qsa.idxk", Gate::Bits),
    ("qsa.bk", Gate::Max(CHAIN_QSA_THRESH)),
    ("qsa.sel", Gate::IntBits),
    ("qsa.attn", Gate::Max(CHAIN_QSA_THRESH)),
    ("moe.route", Gate::MaxInt(CHAIN_MOE_W_THRESH)),
    ("moe.out", Gate::Max(CHAIN_MOE_THRESH)),
    ("mtp.pre", Gate::Bits),
    ("mtp.post", Gate::Max(CHAIN_MTP_POST_THRESH)),
    ("mtp.token", Gate::IntBits),
    ("verify.logits", Gate::Max(CHAIN_LOGITS_THRESH)),
    ("spec.proposals", Gate::IntBits),
    ("spec.tgt", Gate::IntBits),
    ("spec.accepted", Gate::IntBits),
    ("logits", Gate::Max(CHAIN_LOGITS_THRESH)),
];

/// 스테이지 표 출력 + 게이트 판정 — (표, 실패 목록) 반환.
fn judge(out: &ChainOut) -> (String, Vec<String>) {
    let mut lines = Vec::new();
    let mut fails = Vec::new();
    lines.push(String::from(
        "[fn-chain] per-stage table (worst across 3 steps vs core-mirror oracle)",
    ));
    for (name, gate) in STAGE_GATES {
        let Some(s) = out.stats.get(*name) else {
            fails.push(format!("{name} 통계 부재"));
            continue;
        };
        let gate_s = match gate {
            Gate::Bits => "bit-exact".to_string(),
            Gate::Max(t) | Gate::MaxInt(t) => format!("{t:.0e}"),
            Gate::IntBits => "exact".to_string(),
        };
        let ok = match gate {
            Gate::Bits => s.bits == 0 && s.nan == 0,
            Gate::Max(t) => s.max <= *t && s.nan == 0,
            Gate::MaxInt(t) => s.max <= *t && s.bits == 0 && s.nan == 0,
            Gate::IntBits => s.bits == 0,
        };
        let val = if matches!(gate, Gate::Bits | Gate::IntBits) {
            format!("bad={}", s.bits)
        } else {
            format!("maxdiff={:.3e}", s.max)
        };
        lines.push(format!(
            "  {:<10} {:>18} (gate {gate_s:<9}) checked={} nan={} worst@{} {}",
            name,
            val,
            s.checked,
            s.nan,
            if s.loc.is_empty() { "-" } else { &s.loc },
            if ok { "PASS" } else { "FAIL" },
        ));
        if !ok {
            fails.push(format!(
                "{name}: {val} nan={} gate {gate_s} (worst {})",
                s.nan, s.loc
            ));
        }
    }
    println!(
        "[fn-chain] spec round: proposals={:?}/{:?} tgt={:?}/{:?} accepted={:?}/{:?}",
        out.spec.proposals_mod,
        out.spec.proposals_or,
        out.spec.tgt_mod,
        out.spec.tgt_or,
        out.spec.accepted_mod,
        out.spec.accepted_or,
    );
    if out.spec.accepted_mod != out.spec.accepted_or {
        fails.push(format!(
            "spec accepted mismatch mod={:?} or={:?}",
            out.spec.accepted_mod, out.spec.accepted_or,
        ));
    }
    for (s, (&md, (&tm, &to))) in out
        .logits_md
        .iter()
        .zip(out.tokens_mod.iter().zip(out.tokens_or.iter()))
        .enumerate()
    {
        lines.push(format!(
            "[fn-chain] step{s}: end-to-end logits maxdiff={md:.3e} (gate {CHAIN_LOGITS_THRESH:.0e}) token mod={tm} or={to} {}",
            if tm == to { "MATCH" } else { "MISMATCH" }
        ));
        if md > CHAIN_LOGITS_THRESH {
            fails.push(format!(
                "step{s} logits maxdiff {md:.3e} > {CHAIN_LOGITS_THRESH:.0e}"
            ));
        }
        if tm != to {
            fails.push(format!("step{s} token mismatch mod={tm} or={to}"));
        }
    }
    (lines.join("\n"), fails)
}

// ── 진입점 ──

/// 모듈+가중 등록 공통(체크·음성대조 동일 — 결함은 조립부(run_chain) 주입).
fn open_chain(gguf_main: &str) -> Result<(ChainFx, ChainMods), String> {
    let fx = load_fixture(gguf_main)?;
    let dims = fx.dims.clone();
    let mut mods = ChainMods {
        ple: PleCuda::new(dims.clone())?,
        hc: HcCuda::new(HcDims {
            hc: dims.hc,
            n_embd: dims.n_embd,
            low_rank: dims.hc_low_rank,
            eps: dims.eps,
        })?,
        qsa: QsaCuda::new(dims.clone(), QSA_CAP)?,
        moe: MoeCuda::new(MoeDims {
            n_embd: dims.n_embd,
            n_expert: CHAIN_MOE_EXPERTS,
            n_used: dims.n_expert_used,
            n_ff: dims.n_ff_exp,
            n_ff_sh: dims.n_ff_shared,
        })?,
        gdn: FnGdnCuda::open(
            &dims,
            dims.eps,
            &fx.cw_all,
            &fx.dtb_all,
            &fx.ssa_all,
            &fx.nw_all,
        )?,
        mtp: MtpFnCuda::new(MtpFnDims {
            n: dims.n_embd,
            hc: dims.hc,
            low_rank: dims.hc_low_rank,
            eps: dims.eps,
            n_head: dims.n_head,
            n_kv: dims.n_kv,
            head_dim: dims.head_dim,
            n_rot: dims.n_rot,
            rope_base: dims.rope_base,
            n_expert: CHAIN_MOE_EXPERTS,
            n_used: dims.n_expert_used,
            n_ff: dims.n_ff_exp,
            n_ff_sh: dims.n_ff_shared,
            vocab: V_SLICE,
        })?,
    };
    let u16b = |v: &[u16]| -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 2);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    };
    mods.moe
        .set_router_f16(&u16b(&fx.moe_route), &u16b(&fx.moe_route_sh))?;
    mods.moe.set_shared_f16(
        &u16b(&fx.moe_sh_gate),
        &u16b(&fx.moe_sh_up),
        &u16b(&fx.moe_sh_down),
    )?;
    for e in 0..CHAIN_MOE_EXPERTS {
        let (g, u, d) = &fx.moe_experts[&(e as u32)];
        mods.moe.add_expert_f16(e, &u16b(g), &u16b(u), &u16b(d))?;
    }
    for (fi, &il) in fx.qsa_ils.iter().enumerate() {
        let [knw, qnw, iqw, ikw] = &fx.qsa_norms[fi];
        mods.qsa.set_norms(il, knw, qnw, iqw, ikw)?;
    }
    mods.ple.ple_load_block_weights(
        &fx.ple_n_key,
        &fx.ple_n_query,
        &fx.ple_n_conv,
        &fx.ple_conv_w,
    )?;
    // MTP 드래프트 모듈(FNG) — nextn 실측 가중 + 트렁크 공유 합성 MoE·
    // 출력 슬라이스 헤드(vocab 필드는 슬라이스 길이로 등록 — 아키타
    // argmax n 계약, 결함 8호).
    mods.mtp.set_eh_proj(&fx.mtp.eh)?;
    mods.mtp.set_attn(
        &fx.mtp.q, &fx.mtp.k, &fx.mtp.v, &fx.mtp.o, &fx.mtp.qn, &fx.mtp.kn,
    )?;
    mods.mtp.hc.register(
        0,
        "attn",
        &fx.mtp.an,
        &fx.mtp.ad,
        &fx.mtp.au,
        Some(&fx.mtp.ai),
    )?;
    mods.mtp.hc.register(
        0,
        "ffn",
        &fx.mtp.fnn,
        &fx.mtp.fdd,
        &fx.mtp.fu,
        Some(&fx.mtp.fi),
    )?;
    mods.mtp
        .hc
        .register(0, "nextn_head", &fx.mtp.hn, &fx.mtp.hd, &fx.mtp.hu, None)?;
    mods.mtp
        .moe
        .set_router_f16(&u16b(&fx.moe_route), &u16b(&fx.moe_route_sh))?;
    mods.mtp.moe.set_shared_f16(
        &u16b(&fx.moe_sh_gate),
        &u16b(&fx.moe_sh_up),
        &u16b(&fx.moe_sh_down),
    )?;
    for e in 0..CHAIN_MOE_EXPERTS {
        let (g, u, d) = &fx.moe_experts[&(e as u32)];
        mods.mtp
            .moe
            .add_expert_f16(e, &u16b(g), &u16b(u), &u16b(d))?;
    }
    mods.mtp.set_head(&fx.out_rows)?;
    Ok((fx, mods))
}

/// fn-chain — 착지 5스테이지 그리디 디코드 체인 정합 프로브.
/// 스테이지 표(스테이지별 착지 게이트 승계) + 종단 logits(2e-4) +
/// 스텝 토큰 동일. 하나라도 FAIL이면 Err(비영 exit).
pub fn cuda_fn_chain_check(gguf_main: &str) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    let (mut fx, mut mods) = open_chain(gguf_main)?;
    let out = run_chain(&mut fx, &mut mods, ChainNeg::Off)?;
    let (table, fails) = judge(&out);
    println!("{table}");
    let elapsed = t0.elapsed().as_secs_f64();
    if fails.is_empty() {
        Ok(format!(
            "device: {} | fn-chain PASS — 48L schedule (36 GDN + 12 QSA + PLE@1) × 3 steps (t=2+1+1), vocab slice {V_SLICE} | worst logits maxdiff={:.3e} | {elapsed:.0}s (정합 호스트 — 속도 비골든)",
            out.device,
            out.logits_md.iter().cloned().fold(0.0f32, f32::max),
        ))
    } else {
        Err(format!("fn-chain 실패 — {}", fails.join(" ; ")))
    }
}

/// fn-chain-neg — 체인 결함 계급 음성대조(원장 17호): (a) 층 순서 교환
/// (GDN 서수 5↔6 슬롯), (b) 잔차 부착 결함(il=9 combine 가중 1.0 고정).
/// 결함은 모듈 체인에 주입 — 스텝 logits maxdiff > 2e-4(토큰 반전 보고)
/// 로 탐지되어야 NEG-DETECTED(비영 exit).
pub fn cuda_fn_chain_negative_check(gguf_main: &str) -> Result<String, String> {
    let mut results = Vec::new();
    for (tag, neg, kind) in [
        (
            "layer-order(GDN slot 5<->6)",
            ChainNeg::LayerOrderSwap(5, 6),
            "logits",
        ),
        (
            "resid-attach(il=9 w=1.0)",
            ChainNeg::ResidAttachFixed(9),
            "logits",
        ),
        (
            "accept-order(tgt[i+1])",
            ChainNeg::AcceptOrderWrong,
            "accepted",
        ),
    ] {
        let (mut fx, mut mods) = open_chain(gguf_main)?;
        let out = run_chain(&mut fx, &mut mods, neg)?;
        let worst = out.logits_md.iter().cloned().fold(0.0f32, f32::max);
        let flips = out
            .tokens_mod
            .iter()
            .zip(out.tokens_or.iter())
            .filter(|(a, b)| a != b)
            .count();
        let accepted_diff = usize::from(out.spec.accepted_mod != out.spec.accepted_or);
        println!(
            "[fn-chain-neg] {tag}: worst logits maxdiff={worst:.3e} (gate {CHAIN_LOGITS_THRESH:.0e}) token flips={flips}/{STEPS} accepted mod={:?} or={:?} | FAIL(expected)",
            out.spec.accepted_mod, out.spec.accepted_or,
        );
        results.push((tag, kind, worst, accepted_diff));
    }
    let all_detected = results.iter().all(|(_, kind, w, ad)| match *kind {
        "logits" => *w > CHAIN_LOGITS_THRESH,
        _ => *ad > 0,
    });
    let detail = results
        .iter()
        .map(|(t, kind, w, ad)| format!("{t}({kind}) maxdiff={w:.3e} accepted_diff={ad}"))
        .collect::<Vec<_>>()
        .join(", ");
    if all_detected {
        Err(format!(
            "NEG-DETECTED {detail} — 검증계기 정상(층 순서·잔차 부착·수용 순서 결함 모두 탐지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED {detail} — 검증계기 결함: 탐지되지 않은 결함 계급 존재"
        ))
    }
}
