//! Flash-Next HC(hyper-connection mix) 모듈층 스켈레톤 — plans/124
//! G001(FNA)→FNC, 2026-10-05.
//!
//! [계약 요약 — 전체 판정표는 fn_support.rs 머리 계약 지도]
//! HC는 Flash-Next의 잔차 스트림 구조 자체(hc=4 스트림 — 모든 노름을
//! 대체하는 grouped RMSNorm + 저랭크 게이트 + 스트림 평균 + inject).
//! - NEW: grouped RMSNorm(원천: stages/hc.rs grouped_rms L14-21 — 감마는
//!   (1+w) 폴딩이 아닌 원값 w 직접 곱, rms_norm 계급은 core ops.rs
//!   L33-37)·저랭크 게이트(silu(lo/hc) L60 · xn·sigmoid(gate) L69 ·
//!   스트림 평균 /=hc L78)·combine(s += out·2σ(inject/4) — layers.rs
//!   hc_mix 소비부). norm 코어 적산 순서는 exl3_norm_resid(G3)·
//!   exl3_mtp_rms(G9) 미러 계급으로 정렬(트리 환원·정밀 sqrt).
//! - REUSE(조건부): hc_{kind}_{down,up,inject}·output_hc_{down,up}·
//!   nextn.hc_head_{down,up} 선형 — gemv/gemm2(EXL3, krate≤7 이슈)·
//!   q4 MMQ(GGUF). k=2560·n=320 저랭크(실측).
//! - 3종 진입점: hc_mix(kind="attn"|"ffn", inject 있음 L89)·
//!   hc_mix_head(output_hc_*, inject 없음 L126)·hc_mix_nextn_head
//!   (blk.{il}.nextn.hc_head_*, MTP 드래프트 종결 믹서 L108).
//! - 토큰 축 배치: down/up/inject 전 토큰 1회(층당 GPU 왕복 6회 고정 —
//!   토큰당 288회 왕복이 병목이었던 원장, hc.rs 헤드).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::fn_support::FnCuda;

/// HC 스트림 수(hyper_connection.count 실측 4).
pub const FN_HC_STREAMS: usize = 4;

impl FnCuda {
    /// hc_mix 진입(attn|ffn) — TODO(FNC): stages/hc.rs hc_mix L89-106 →
    /// hc_mix_ex L25-86 미러. 순서: grouped rms(전 토큰) → down+inject
    /// 동일 입력 그룹 1호출 → silu(lo/hc) → up → 게이트 적용+스트림 평균.
    /// 반환 (mixed, inject) — combine은 layers/forward 체인(FNH).
    pub fn hc_mix(
        &mut self,
        _il: usize,
        _kind: &str,
        _res_hc: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        Err("미구현: FNC(fn_hc_cuda) 목표".into())
    }

    /// 출력 헤드용 HC mix(inject 없음) — TODO(FNC): stages/hc.rs
    /// hc_mix_head L126-133(output_hc_{norm,down,up}) 미러.
    pub fn hc_mix_head(&mut self, _res_hc: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNC(fn_hc_cuda) 목표".into())
    }

    /// MTP 드래프트 종결 HC mix — TODO(FNC): stages/hc.rs
    /// hc_mix_nextn_head L108-124(blk.{il}.nextn.hc_head_{norm,down,up})
    /// 미러 — 본체 output_hc와 별개 가중치(plans/109 P15②).
    pub fn hc_mix_nextn_head(
        &mut self,
        _il: usize,
        _res_hc: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNC(fn_hc_cuda) 목표".into())
    }

    /// grouped RMSNorm — TODO(FNC): stages/hc.rs grouped_rms L14-21 미러.
    /// [hc·n] 행을 스트림별로 잘라 각각 rms_norm(스트림 순서 불변 —
    /// 산술은 for s in 0..hc 인라인 판과 동일).
    pub fn hc_grouped_rms(&self, _x: &[f32], _w: &[f32]) -> Result<Vec<f32>, String> {
        Err("미구현: FNC(fn_hc_cuda) 목표".into())
    }
}
