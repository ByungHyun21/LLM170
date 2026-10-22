
use llm170_backend_gpu::W4a16Dec;
use std::path::Path;

mod g4;
mod gemma;
mod mtp;
mod pp;
mod qwen;
mod spec;
mod upload;
mod upload_fn;
mod upload_g4;
pub(crate) use self::g4::{g4_chunk, g4_step};
pub(crate) use self::gemma::GemmaEngine;
pub(crate) use self::pp::PpEngine;
pub(crate) use self::upload::{DirectLoad, upload_model};
pub(crate) use self::upload_fn::{upload_fn, upload_fn_win};
pub(crate) use self::upload_g4::{G4HostCfg, upload_model_g4};

pub struct QwenEngine {
    model: llm170_core::qwen35::Model,
    dec: W4a16Dec,
    #[allow(dead_code)]
    n_slots: usize,
    head_gpu: bool,
    softcap: f32,
    mtp: bool,
    pub tuning: Tuning,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Tuning {
    pub ngram_n: Option<usize>,
    pub ngram_m: Option<usize>,
    pub chunk_t: Option<usize>,
    pub lru_cap: Option<usize>,
    pub lru_frac: Option<u64>,
}
pub fn prefill_tmax(is_moe: bool, resident: bool, env: Option<&str>) -> usize {
    if is_moe && !resident {
        return env
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, 12);
    }
    let hi = if is_moe { 512 } else { 2048 };
    let def = if is_moe { 512 } else { 2048 };
    env.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(def)
        .clamp(1, hi)
}

#[cfg(test)]
mod tests {
    use super::prefill_tmax;

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
        assert_eq!(prefill_tmax(false, false, None), 2048, "dense = 기본 2048");
        assert_eq!(
            prefill_tmax(false, false, Some("128")),
            128,
            "env 오버라이드"
        );
        assert_eq!(
            prefill_tmax(false, false, Some("99999")),
            2048,
            "상한 클램프"
        );
        assert_eq!(prefill_tmax(false, false, Some("0")), 1, "하한 클램프");
        assert_eq!(
            prefill_tmax(false, false, Some("abc")),
            2048,
            "파싱 실패 = 기본"
        );
    }
}


pub enum GpuEngine {
    Qwen(Box<QwenEngine>),
    Gemma(Box<GemmaEngine>),
    Pp(Box<PpEngine>),
}

macro_rules! engine_delegate_mut {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
        impl GpuEngine {
            $(
                pub fn $name(&mut self $(, $a: $t)*) -> $r {
                    match self {
                        GpuEngine::Qwen(e) => e.$name($($a),*),
                        GpuEngine::Gemma(e) => e.$name($($a),*),
                        GpuEngine::Pp(e) => e.$name($($a),*),
                    }
                }
            )*
        }
    };
}

macro_rules! engine_delegate_ref {
    ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
        impl GpuEngine {
            $(
                #[allow(dead_code)]
                pub fn $name(&self $(, $a: $t)*) -> $r {
                    match self {
                        GpuEngine::Qwen(e) => e.$name($($a),*),
                        GpuEngine::Gemma(e) => e.$name($($a),*),
                        GpuEngine::Pp(e) => e.$name($($a),*),
                    }
                }
            )*
        }
    };
}

engine_delegate_mut! {
    prefill(seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String>;
    prefill_greedy(seq: usize, tokens: &[u32]) -> Result<u32, String>;
    prefill_state(seq: usize, tokens: &[u32]) -> Result<(), String>;
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
    apply_tuning(t: Tuning) -> Result<(), String>;
    spec_verify_logits(slot: usize, rows: &[f32], t: usize) -> Result<Vec<f32>, String>;
    warm_spec_graphs(n_slots: usize, eos: u32) -> Result<(), String>;
}

engine_delegate_ref! {
    spec_on() -> bool;
    layer_window() -> llm170_backend_gpu::LayerWindow;
    owns_final_layer() -> bool;
    seq_pos(slot: usize) -> u32;
    kv_total_blocks() -> usize;
    kv_free_blocks() -> usize;
    kv_blocks_needed(n_tokens: usize) -> usize;
    kv_cap_tokens() -> usize;
    kv_used_blocks(slot: usize) -> usize;
    kv_tier_bytes() -> (u64, u64, usize, usize);
    kv_tier_free() -> (usize, usize);
    reuse_valid(slot: usize, l: usize, prompt_len: usize) -> bool;
    piece(tok: u32) -> String;
    mtp_drafts_on() -> bool;
    ngram_knobs() -> (usize, usize);
    mem_stats() -> (u64, u64, u64, u64);
    active_weight_bytes() -> u64;
    moe_mode() -> &'static str;
    mcache_stats() -> (u64, u64, usize);
    copy_stats() -> [(u64, u64, u64); 3];
    eos() -> u32;
    embed_row(tok: u32) -> Result<Vec<f32>, String>;
    vocab() -> usize;
    n_embd() -> usize;
    prefill_mixed_cap() -> usize;
}

impl GpuEngine {
    pub fn load(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
        kv_vram_cap: Option<u64>,
    ) -> Result<Self, String> {
        let arch = llm170_core::qwen35::bind::dir_arch(dir).unwrap_or_default();
        if llm170_core::gemma4::bind::arch_ok(&arch) {
            return Ok(GpuEngine::Gemma(Box::new(GemmaEngine::load(
                dir,
                n_slots,
                ctx,
                kv_budget,
                kv_reserve,
                kv_vram_cap,
            )?)));
        }
        Ok(GpuEngine::Qwen(Box::new(QwenEngine::load(
            dir,
            n_slots,
            ctx,
            kv_budget,
            kv_reserve,
            kv_vram_cap,
        )?)))
    }
}

impl GpuEngine {
    pub fn load_pp(
        dir: &Path,
        n_slots: usize,
        ctx: usize,
        kv_budget: Option<u64>,
        kv_reserve: Option<u64>,
        kv_vram_cap: Option<u64>,
    ) -> Result<Self, String> {
        let arch = llm170_core::qwen35::bind::dir_arch(dir).unwrap_or_default();
        if llm170_core::gemma4::bind::arch_ok(&arch) {
            return Err("PP 2샤드: gemma4 미지원(1차 = qwen dense 27B)".into());
        }
        Ok(GpuEngine::Pp(Box::new(PpEngine::load(
            dir,
            n_slots,
            ctx,
            kv_budget,
            kv_reserve,
            kv_vram_cap,
        )?)))
    }
}
