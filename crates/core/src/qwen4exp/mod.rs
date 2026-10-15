//! [P7 2026-10-15] qwen4_exp(FN = Qwen3.8-Flash-Next) arch 골격 — §2.M 브링업
//! 선행분. **config 판독 + ctx_info만** (serve ctx 파생·기동 가드 입력).
//! 가중치 로드·엔진 경로·PLE/indexer 브링업은 본 트랙(플랜 §2.M).
//!
//! 체크포인트 계약(2026-10-15 config 실측): `architectures =
//! Qwen4ExpForConditionalGeneration` · `model_type = qwen4_exp` · hparams는
//! **`text_config` 중첩**(gemma4와 동일 패턴) · 48층 = layer_types
//! [linear×3, full×1] 반복(full_attention_interval=4 → full 12·linear 36) ·
//! full층 KV = num_key_value_heads(2)×head_dim(256) · linear(GDN류) 상태 =
//! linear_num_value_heads(48)×128×128 · **512 전문가 top-10**(+shared 640) ·
//! PLE = ple_layer_ids([2])·embed 2560 · **MTP 헤드 보유**(mtp 1층 full-attn,
//! hybrid — dense 마커 게이트와 정합) · ctx 262,144.

pub mod bind;

pub use bind::{Qwen4ExpCfg, arch_ok, ctx_info, dir_arch_ok};
