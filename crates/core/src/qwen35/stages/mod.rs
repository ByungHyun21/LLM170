//! 스테이지 컨텍스트 — qwen4exp P1 패턴 재적용 (plans/90 B4 이관).
//! 층 구현(gdn/attn)은 model·가속기 뷰(Ctx)와 시퀀스 상태만 받는다 —
//! Engine 내부 구조와 독립. 수치 불변(이동만).

mod attn;
mod gdn;

pub(crate) use attn::attn_layer;
pub(crate) use gdn::gdn_layer;

use crate::matmul::Acc;

/// 스테이지 실행 컨텍스트 — 모델 뷰(불변) + 가속기 참조.
pub struct Ctx<'a> {
    pub model: &'a super::Model,
    pub acc: &'a Acc,
}
