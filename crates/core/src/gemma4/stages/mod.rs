//! gemma4 스테이지 — 어텐션·FFN(dense MLP + MoE 병렬).

mod attn;
mod ffn;

pub(crate) use attn::attn_layer;
pub(crate) use ffn::{dense_mlp, moe_block, route};
