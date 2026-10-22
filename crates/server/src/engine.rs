
use std::path::PathBuf;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ModelFormat {
    W4A16 { arch: String },
}

pub fn sniff_format(path: &std::path::Path) -> Result<ModelFormat, String> {
    if path.is_file() {
        return Err(format!(
            "파일 모델 미지원(2026-10-08): W4A16 디렉터리만 지원 — {}",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!("모델 경로 없음: {}", path.display()));
    }
    let index = path.join("model.safetensors.index.json");
    let hay: Option<String> = if index.is_file() {
        std::fs::read_to_string(&index).ok()
    } else {
        let shard = std::fs::read_dir(path).ok().and_then(|rd| {
            rd.flatten().map(|e| e.path()).find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model") && n.ends_with(".safetensors"))
            })
        });
        shard.and_then(|p| {
            use std::io::Read;
            let mut f = std::fs::File::open(p).ok()?;
            let mut lenb = [0u8; 8];
            f.read_exact(&mut lenb).ok()?;
            let hlen = u64::from_le_bytes(lenb).min(1 << 20) as usize;
            let mut buf = vec![0u8; hlen];
            f.read_exact(&mut buf).ok()?;
            String::from_utf8_lossy(&buf).into_owned().into()
        })
    };
    let Some(hay) = hay else {
        if path.join("config.json").is_file() {
            return Err(format!(
                "미지원 포맷: HF config 배포(architectures) — {} (지원: W4A16 디렉터리)",
                path.display()
            ));
        }
        return Err(format!(
            "모델 디렉터리 인식 불가(model*.safetensors/index.json 없음): {}",
            path.display()
        ));
    };
    if hay.contains(".trellis") {
        return Err(format!(
            "미지원 quant 스키마(.trellis — 2026-10-08): W4A16 디렉터리만 지원 — {}",
            path.display()
        ));
    }
    if hay.contains("weight_packed") || hay.contains("qweight") {
        let Some(arch) = llm170_core::qwen35::bind::dir_arch(path) else {
            return Err(format!(
                "아키텍처 판별 불가(config.json architectures/model_type 부재): {}",
                path.display()
            ));
        };
        if !llm170_core::qwen35::bind::arch_supported(&arch)
            && !llm170_core::gemma4::bind::arch_ok(&arch)
        {
            return Err(format!(
                "미지원 아키텍처: {arch} (지원: {:?} + gemma4) — {}",
                llm170_core::qwen35::bind::ARCHES,
                path.display()
            ));
        }
        return Ok(ModelFormat::W4A16 { arch });
    }
    if path.join("config.json").is_file() {
        return Err(format!(
            "미지원 포맷: HF config 배포(architectures) — {} (지원: W4A16 디렉터리)",
            path.display()
        ));
    }
    Err(format!(
        "모델 디렉터리 포맷 인식 불가: {} (지원: W4A16 디렉터리)",
        path.display()
    ))
}

pub static SPEC_POLICY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MOE_POLICY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PLE_POLICY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn ple_name() -> &'static str {
    match PLE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
        1 => "ram",
        2 => "vram",
        _ => "ssd",
    }
}

pub fn parse_ple_policy(v: &str) -> Option<u8> {
    match v {
        "ssd" => Some(0),
        "ram" => Some(1),
        "vram" => Some(2),
        _ => None,
    }
}

pub fn parse_spec_policy(v: &str) -> Option<u8> {
    match v {
        "auto" => Some(0),
        "mtp" => Some(1),
        "ngram" => Some(2),
        "off" => Some(3),
        _ => None,
    }
}

pub fn spec_name() -> &'static str {
    match SPEC_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
        1 => "mtp",
        2 => "ngram",
        3 => "off",
        _ => "auto",
    }
}

pub fn moe_name() -> &'static str {
    match MOE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
        1 => "resident",
        2 => "streaming",
        3 => "hybrid",
        _ => "auto",
    }
}

pub fn parse_moe_policy(v: &str) -> Option<u8> {
    match v {
        "auto" => Some(0),
        "resident" => Some(1),
        "streaming" => Some(2),
        "hybrid" => Some(3),
        _ => None,
    }
}

#[derive(Clone)]
pub struct InferRequest {
    pub model: PathBuf,
    pub ctx: usize,
    pub model_max_ctx: usize,
    pub kv_budget: Option<u64>,
    pub kv_vram_cap: Option<u64>,
    pub kv_reserve: Option<u64>,
    pub cards: usize,
}

pub struct InferResult {
    pub tokens: Vec<u32>,
    pub error: Option<String>,
}

pub struct Engine(pub Box<crate::gpu_engine::GpuEngine>);

impl Engine {
    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        self.0.mem_stats()
    }

    pub fn active_weight_bytes(&self) -> u64 {
        self.0.active_weight_bytes()
    }

    pub fn moe_mode(&self) -> &'static str {
        self.0.moe_mode()
    }

    pub fn mcache_stats(&self) -> (u64, u64, usize) {
        self.0.mcache_stats()
    }

    pub fn kv_cap_tokens(&self) -> usize {
        self.0.kv_cap_tokens()
    }

    pub fn kv_tier_bytes(&self) -> (u64, u64, usize, usize) {
        self.0.kv_tier_bytes()
    }

    pub fn kv_tier_free(&self) -> (usize, usize) {
        self.0.kv_tier_free()
    }

    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.0.copy_stats()
    }

    pub fn eos(&self) -> u32 {
        self.0.eos()
    }
}

fn banner(
    model: &std::path::Path,
    fmt: &str,
    runtime: &str,
    offload: &str,
    attach: &str,
    ctx: usize,
    slots: usize,
) {
    let name = model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| model.to_str().unwrap_or("?"));
    eprintln!(
        "# boot: model={name} format={fmt} runtime={runtime} offload={offload} ctx={ctx} slots={slots} attach={attach}"
    );
}

pub fn build_slots(req: InferRequest, n_slots: usize) -> Engine {
    let eng = load_gpu_retry(
        &req.model,
        n_slots,
        req.ctx,
        req.kv_budget,
        req.kv_reserve,
        req.kv_vram_cap,
        req.cards,
    );
    banner(
        &req.model,
        "w4a16",
        "cuda",
        "none",
        "cuda-resident",
        req.ctx,
        n_slots,
    );
    Engine(Box::new(eng))
}

impl Engine {
    pub fn reset_seq(&mut self, seq: usize) {
        let _ = self.0.reset_seq(seq);
    }
}

pub struct Detok {
    buf: Vec<u8>,
}

impl Detok {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn push(&mut self, tok: u32) -> String {
        if let Some(t) = TOKENIZER.get() {
            t.piece_bytes_into(tok, &mut self.buf);
        }
        let v = complete_utf8_prefix(&self.buf);
        let out = String::from_utf8_lossy(&self.buf[..v]).into_owned();
        self.buf.drain(..v);
        out
    }
}

fn complete_utf8_prefix(b: &[u8]) -> usize {
    let mut v = 0usize;
    while v < b.len() {
        let ok2 = v + 1 < b.len() && b[v + 1] & 0xC0 == 0x80;
        let ok3 = v + 2 < b.len() && b[v + 1] & 0xC0 == 0x80 && b[v + 2] & 0xC0 == 0x80;
        let ok4 = v + 3 < b.len() && ok3 && b[v + 3] & 0xC0 == 0x80;
        match b[v] {
            x if x < 0x80 => v += 1,
            0xC0..=0xDF if ok2 => v += 2,
            0xE0..=0xEF if ok3 => v += 3,
            0xF0..=0xF7 if ok4 => v += 4,
            _ => break,
        }
    }
    v
}

use crate::sched::load_gpu_retry;
pub use crate::sched::{SlotJob, slot_loop};

pub static TOKENIZER: std::sync::OnceLock<crate::tokenize::Tokenizer> = std::sync::OnceLock::new();
pub static ARCH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

pub fn arch() -> String {
    ARCH.get().cloned().unwrap_or_default()
}

pub fn greedy_encode(text: &str) -> Vec<u32> {
    TOKENIZER.get().map(|t| t.encode(text)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::complete_utf8_prefix;

    #[test]
    fn detok_holdback_multibyte() {
        assert_eq!(
            complete_utf8_prefix(&[0xED, 0x95]),
            0,
            "미완 3바이트 홀드백"
        );
        assert_eq!(complete_utf8_prefix(&[0xED, 0x95, 0x9C]), 3);
        assert_eq!(complete_utf8_prefix(&[0xF0, 0x9F, 0x98]), 0);
        assert_eq!(complete_utf8_prefix(&[0xF0, 0x9F, 0x98, 0x80]), 4);
        assert_eq!(complete_utf8_prefix(b"ab\xC3"), 2);
        assert_eq!(complete_utf8_prefix(b"ab\xC3\xA9"), 4);
        assert_eq!(complete_utf8_prefix(&[0x80, 0x80]), 0);
        assert_eq!(complete_utf8_prefix(b"abc"), 3);
        assert_eq!(complete_utf8_prefix(b""), 0);
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    fn policy_parsers() {
        assert_eq!(parse_spec_policy("auto"), Some(0));
        assert_eq!(parse_spec_policy("mtp"), Some(1));
        assert_eq!(parse_spec_policy("ngram"), Some(2));
        assert_eq!(parse_spec_policy("off"), Some(3));
        assert_eq!(parse_spec_policy("x"), None);
        assert_eq!(parse_moe_policy("auto"), Some(0));
        assert_eq!(parse_moe_policy("resident"), Some(1));
        assert_eq!(parse_moe_policy("streaming"), Some(2));
        assert_eq!(parse_moe_policy(""), None);
    }
}
