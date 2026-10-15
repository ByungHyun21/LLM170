//! GPU 서빙 엔진 — W4A16 CUDA 디바이스 체인 + GPU head.
//!
//! W3-3 착륙 후 성능 캠페인(2026-10-08): 활성은 디바이스 상주(왕복은
//! 토큰당 임베딩 업로드 + 로짓 판독뿐), head는 head_bf16 커널(bf16
//! →f32 시프트 + k 직렬 f32 내적 — CPU 참조와 동일 순서).
//! 판정 기준은 토큰열(골든)이다.

use llm170_backend_gpu::W4a16Dec;
use std::path::Path;

mod g4;
mod gemma;
mod mtp;
mod qwen;
mod spec;
mod upload;
mod upload_g4;
pub(crate) use self::g4::{g4_chunk, g4_step};
pub(crate) use self::gemma::GemmaEngine;
pub(crate) use self::upload::upload_model;
pub(crate) use self::upload_g4::{G4HostCfg, upload_model_g4};

pub struct QwenEngine {
    model: llm170_core::qwen35::Model,
    dec: W4a16Dec,
    #[allow(dead_code)] // reset_states(진단/복구) 전용
    n_slots: usize,
    /// GPU head 상주 여부(bf16 output.weight 업로드 성공).
    head_gpu: bool,
    /// [gemma4 P2 2026-10-14] 최종 로짓 softcap(0=미적용) — 장치 argmax는
    /// 단조 불변이라 커널 무접촉, **호스트로 나오는 로짓**(샘플링·PPL/eval·
    /// 검증)에만 적용한다. 값은 config.json 파생(모델별).
    softcap: f32,
    /// [MTP 2026-10-12] 모델 자체 MTP 초안기 활성(27B dense v1).
    mtp: bool,
}

/// [R16] 프리필 청크 tmax 정책 — 순수 함수(테스트 표면).
/// 규칙: MoE 스트리밍은 캐시 스테이징 기반 t≤8 청크(종전 t=1 폴백 — 512토큰
/// ~9.3s), 그 외는 `LLM170_PREFILL_T`(진단 오버라이드) 또는 기본 512
/// (가중치 상각), 1..=512.
pub fn prefill_tmax(is_moe: bool, resident: bool, env: Option<&str>) -> usize {
    if is_moe && !resident {
        // [P6 2026-10-15] 8 유지 — t>8 실험: ①union 가드 실효(distinct 71 >
        // 캐시 70 — 구조 상한 8·top_k ≤ k) ②캐시 확대(k=96)+t=12는 그룹 경로
        // illegal access로 파킹. env(≤12)는 허용 — moe_stage_t 가드가 상한을
        // 지킨다(실패는 즉시·명시).
        return env
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, 12);
    }
    env.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        .clamp(1, 512)
}

#[cfg(test)]
mod tests {
    use super::prefill_tmax;

    /// [R16] 프리필 tmax 정책 — 스트리밍 MoE 폴백·env 오버라이드·상한.
    #[test]
    fn prefill_tmax_policy() {
        assert_eq!(
            prefill_tmax(true, false, None),
            8,
            "MoE 스트리밍 = 구조 안전 8"
        );
        assert_eq!(
            prefill_tmax(true, false, Some("8")),
            8,
            "MoE 스트리밍 env 오버라이드"
        );
        assert_eq!(
            prefill_tmax(true, false, Some("999")),
            12,
            "MoE 스트리밍 env 상한 클램프(가드가 지킴)"
        );
        assert_eq!(prefill_tmax(true, true, None), 512, "MoE 상주 = 기본 512");
        assert_eq!(prefill_tmax(false, false, None), 512, "dense = 기본 512");
        assert_eq!(
            prefill_tmax(false, false, Some("128")),
            128,
            "env 오버라이드"
        );
        assert_eq!(
            prefill_tmax(false, false, Some("99999")),
            512,
            "상한 클램프"
        );
        assert_eq!(prefill_tmax(false, false, Some("0")), 1, "하한 클램프");
        assert_eq!(
            prefill_tmax(false, false, Some("abc")),
            512,
            "파싱 실패 = 기본"
        );
    }
}

// ──────────────────── [gemma4 P3 2026-10-14] 서빙 아치 디스패치 ────────────────────
// qwen(QwenEngine)·gemma4(GemmaEngine)를 단일 타입으로 — P2-3 v0 체인 그대로
// (호스트 스테이징·f32 활성 클래스·g4 커널군·g32 GEMV·int8 KV/head).

/// 서빙 엔진 — 아치 디스패치 단일 타입(호출부는 이 타입만 쓴다).
pub enum GpuEngine {
    Qwen(Box<QwenEngine>),
    Gemma(Box<GemmaEngine>),
}

/// 위임 매크로 — &mut self 메서드.
macro_rules! engine_delegate_mut {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
        impl GpuEngine {
            $(
                pub fn $name(&mut self $(, $a: $t)*) -> $r {
                    match self {
                        GpuEngine::Qwen(e) => e.$name($($a),*),
                        GpuEngine::Gemma(e) => e.$name($($a),*),
                    }
                }
            )*
        }
    };
}

/// 위임 매크로 — &self 메서드.
macro_rules! engine_delegate_ref {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
        impl GpuEngine {
            $(
                #[allow(dead_code)]
                pub fn $name(&self $(, $a: $t)*) -> $r {
                    match self {
                        GpuEngine::Qwen(e) => e.$name($($a),*),
                        GpuEngine::Gemma(e) => e.$name($($a),*),
                    }
                }
            )*
        }
    };
}

engine_delegate_mut! {
    prefill(seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String>;
    kv_grow(slot: usize, n_tokens: usize) -> Result<usize, String>;
    kv_release_slot(slot: usize) -> usize;
    prof_report(label: &str) -> Result<String, String>;
    restore_prefix(slot: usize, upto: u32) -> Result<u32, String>;
    spec_rollback(slot: usize, keep: usize) -> Result<(), String>;
    spec_rewind_pos(slot: usize, n: u32) -> Result<(), String>;
    spec_verify(slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String>;
    mtp_first(slot: usize, tok: u32) -> Result<(), String>;
    mtp_single(slot: usize, tok: u32) -> Result<(), String>;
    mtp_pre_round(acc: usize) -> Result<(), String>;
    mtp_post_round(slot: usize, accepted: &[u32], corrected: u32) -> Result<(), String>;
    mtp_slot_kv_reset() -> Result<(), String>;
    mtp_slot_fresh(slot: usize) -> ();
    mtp_slot_reused(slot: usize) -> ();
    mtp_take_draft(slot: usize) -> Option<Vec<u32>>;
    decode(seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<Vec<f32>>, String>;
    decode_np_greedy(seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<u32>, String>;
    decode_greedy(seq: usize, token: u32) -> Result<u32, String>;
    decode_greedy_k(seq: usize, token: u32, k: usize, eos: u32) -> Result<Vec<u32>, String>;
    decode_np_mixed(seq_ids: &[usize], tokens: &[u32]) -> Result<(Vec<u32>, Vec<f32>), String>;
    decode_np_logits(seq_ids: &[usize], tokens: &[u32]) -> Result<Vec<f32>, String>;
    reset_seq(seq: usize) -> Result<(), String>;
    enable_spec() -> Result<(), String>;
    spec_verify_logits(slot: usize, rows: &[f32], t: usize) -> Result<Vec<f32>, String>;
}

engine_delegate_ref! {
    spec_on() -> bool;
    seq_pos(slot: usize) -> u32;
    kv_total_blocks() -> usize;
    kv_free_blocks() -> usize;
    kv_blocks_needed(n_tokens: usize) -> usize;
    reuse_valid(slot: usize, l: usize, prompt_len: usize) -> bool;
    piece(tok: u32) -> String;
    mtp_drafts_on() -> bool;
    mem_stats() -> (u64, u64, u64, u64);
    active_weight_bytes() -> u64;
    moe_mode() -> &'static str;
    mcache_stats() -> (u64, u64, usize);
    copy_stats() -> [(u64, u64, u64); 3];
    eos() -> u32;
    embed_row(tok: u32) -> Result<Vec<f32>, String>;
    vocab() -> usize;
    n_embd() -> usize;
}

impl GpuEngine {
    /// 아치 스니핑 후 해당 엔진 로드 — 서빙 진입 단일 경로.
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
    ) -> Result<Self, String> {
        let arch = llm170_core::qwen35::bind::dir_arch(dir).unwrap_or_default();
        if llm170_core::gemma4::bind::arch_ok(&arch) {
            return Ok(GpuEngine::Gemma(Box::new(GemmaEngine::load(
                dir, n_slots, ctx, kv_budget, kv_reserve,
            )?)));
        }
        Ok(GpuEngine::Qwen(Box::new(QwenEngine::load(
            dir, n_slots, ctx, kv_budget, kv_reserve,
        )?)))
    }
}
