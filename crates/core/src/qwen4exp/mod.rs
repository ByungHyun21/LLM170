//! [P7 2026-10-15] qwen4_exp(FN = Qwen3.8-Flash-Next) arch 골격 — §2.M 브링업
//! [FN 2단계 2026-10-17] CPU 오라클 엔진 골격(engine·stages) 추가 —
//! w4a16-ref fn-probe 진입. 골든은 3차.
//!
//! 체크포인트 계약(2026-10-15 config 실측): `architectures =
//! Qwen4ExpForConditionalGeneration` · `model_type = qwen4_exp` · hparams는
//! **`text_config` 중첩**(gemma4와 동일 패턴) · 48층 = layer_types
//! [linear×3, full×1] 반복(full_attention_interval=4 → full 12·linear 36) ·
//! full층 KV = num_key_value_heads(2)×head_dim(256) · linear(GDN류) 상태 =
//! linear_num_value_heads(48)×128×128 · **512 전문가 top-10**(+shared 640) ·
//! PLE = ple_layer_ids([2])·embed 2560 · **MTP는 config에만 기술**(1층
//! full-attn hybrid)되고 체크포인트 텐서는 미포함 → 엔진 n-gram 초안 폴백
//! (2026-10-17 index.json 실측) · ctx 262,144.

pub mod bind;

// [FN 1단계 2026-10-17] w4a16-load 식별 연결 — bind::report(hparams 전항 +
// ctx_info + PLE/ngram/MTP 스켈레톤 판별). packed 로드·엔진은 본 트랙.
pub use bind::{Qwen4ExpCfg, arch_ok, ctx_info, dir_arch_ok, report};
// [FN 2단계 2026-10-17] CPU 오라클 엔진 골격 — 스토어 직접 접근(gptq 팩).
pub mod engine;
// [FN D1→재팩 2026-10-20] GPTQ→CT 무손실 재패커(니블 전치) — fn_repack 바이너리 진입.
pub mod repack;
pub mod stages;
// [FN D2 2026-10-20] 상태ful CPU 오라클 — 전 48층 forward 체인·greedy 골든.
pub mod oracle;

/// w4a16-ref fn-probe 진입 — engine::probe 재수출.
pub use engine::probe;
