//! 스테이지 컨텍스트 — 층 구현(gdn/attn)은 모델 뷰(Ctx)와 시퀀스 상태만 받는다.

mod attn;
mod gdn;
mod moe;

pub(crate) use attn::attn_layer;
pub(crate) use gdn::gdn_layer;
pub(crate) use moe::moe_ffn;

/// 스테이지 실행 컨텍스트 — 모델 뷰(불변).
pub struct Ctx<'a> {
    pub model: &'a super::Model,
}
