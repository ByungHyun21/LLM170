//! Flash-Next MoE(512 전문가 top-10 + shared) 모듈층 스켈레톤 — plans/124
//! G001(FNA)→FNE, 2026-10-05.
//!
//! [계약 요약 — 전체 판정표는 fn_support.rs 머리 계약 지도]
//! - NEW: 라우팅(softmax max-sub → total_cmp 정렬 → top-10 → 정규화
//!   wsum.max(6.103_515_6e-5)) — 원천: stages/moe.rs L46-70. NaN 내성
//!   total_cmp(107 W11)·w!=0 가드 포함.
//! - REUSE 후보: 전문가 GEMM — GGUF ffn_{gate,up,down}_exps 스택 Q4_K
//!   [2560,640,512]는 G8 moe_down(ids 그룹 런치) 산술 class(원천:
//!   moe.rs L204-216 토큰-메이저 페어). EXL3는 전문가별 분산 trellis
//!   (mlp.experts.{e}.* — krate≤7 이슈) → 퍼-전문가 gemv/gemm2 경로
//!   (원천 L228-297 서브배치). 상주 예산 판정은 FNE(스택 345MB/층 계약
//!   L92-95·원장 18호 소형-T 점유 — kseg/mma 타일링).
//! - REUSE: shared 게이트곱 silu·mul = exl3_ew(G7) 그대로(원천:
//!   moe.rs silu_rows L10-16·L249-255). sigmoid(sgate)·가중 합산은
//!   NEW 소형(원천 L313-318).
//! - 라우터·shared 게이트 입력: EXL3 mlp.gate.weight F16[512,2560]
//!   (트렐리스 아님 — 실측 2026-10-05) — F16 GEMV 경로.
//! - 정합: 값 maxdiff 판정(argmax 금지) + NaN 가드 원장(trace 경로는
//!   core 계약 — rawcuda 프로브는 계기 자체 검증, 원장 17호).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::fn_support::FnCuda;

/// 전문가 수(GGUF qwen4exp.expert_count 실측 512).
pub const FN_MOE_EXPERTS: usize = 512;
/// 토큰당 선택 전문가 수(expert_used_count 실측 10).
pub const FN_MOE_USED: usize = 10;
/// 라우팅 정규화 하한 — 원천: stages/moe.rs L62(wsum.max(6.103_515_6e-5)).
pub const FN_MOE_WSUM_FLOOR: f32 = 6.103_515_6e-5;

impl FnCuda {
    /// MoE FFN 진입 — TODO(FNE): stages/moe.rs moe_ffn L22-320 미러.
    /// 순서 계약: 라우터 2종 배치(L40-45) → 선택(전문가별 (ti,w) 리스트
    /// L46-70) → 전문가 GEMM 3 role(경로 선택은 상주 예산 — ENV 분기 아님)
    /// → shared(게이트곱·down·sigmoid 게이트 합산 L246-318).
    /// 토큰-메이저 페어 정렬 (ti,e,w) — 전문가별 경로의 누산 순서와 동일
    /// (L198-201 계약: 경로 간 산술 동일성).
    pub fn moe_ffn(&mut self, _il: usize, _xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNE(fn_moe_cuda) 목표".into())
    }

    /// 라우팅(softmax·top-k·정규화) — TODO(FNE): moe.rs L46-70 미러.
    /// NaN 로짓 내성 total_cmp 정렬 + wsum 하린(floor) 가드 포함.
    pub fn moe_route(&self, _logits: &[f32]) -> Result<Vec<(usize, f32)>, String> {
        Err("미구현: FNE(fn_moe_cuda) 목표".into())
    }
}
