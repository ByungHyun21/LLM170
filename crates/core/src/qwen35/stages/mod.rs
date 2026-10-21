//! 스테이지 컨텍스트 — 층 구현(gdn/attn)은 모델 뷰(Ctx)와 시퀀스 상태만 받는다.

mod attn;
mod gdn;
// [FN 2단계 2026-10-17] MoE 라우팅(select_topk) 재사용 — qwen4exp 스테이지.
pub(crate) mod moe;

pub(crate) use attn::attn_layer;
pub(crate) use gdn::gdn_layer;
pub(crate) use moe::moe_ffn;
// [fn-moe-check 2026-10-21] 라우팅 미러 공개 — GPU topk 대조 프로브가
// 오라클과 동일한 선택 산출에 쓴다(softmax→k라운드→재정규화).
pub use moe::select_topk;

/// 스테이지 실행 컨텍스트 — 모델 뷰(불변).
pub struct Ctx<'a> {
    pub model: &'a super::Model,
}
